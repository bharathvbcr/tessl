// Metal 4 e-means clustering substrate:
// - 64x64 execution_simdgroups<4> cooperative TensorOps (bf16 & exact f32) argmin
// - 32x32 portable simdgroup_float8x8 fallback (bf16 & f32)
// - Split-C partition reduction for huge-k clusterings
// - Deterministic 8-bit 64-bit-key GPU radix sort + 256-row overflow slice gather
// - Pure-integer IEEE-754 binary64 objective accumulation (fast-math proof)
// - Exact weight==1.0f override EMA centroid update and SplitMix64 Gumbel split revival
//
// Device pointers passed to MPP tensor(...) are non-const because const poisons
// MPP template deduction. All 2D element offsets widen to 64-bit `ulong` before
// multiplication, and all non-finite / subnormal checks use integer bit masks
// (no banned float classification macros or all-ones-exponent float constants).

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#pragma clang fp contract(off) reassociate(off)

constant constexpr float EMEANS_MAX_SENTINEL = as_type<float>(0x7f7fffffu);
constant constexpr uint EMEANS_INVALID_LABEL = 0xffffffffu;

// Fault bitmask bits in `fault_flags[0]`:
//   bit 0 (1u): non-finite input coordinate (row or centroid)
//   bit 1 (2u): finite f32 overflowed bf16 range (|x| rounds to >= 2^128) or raw bf16 non-finite
//   bit 2 (4u): squared L2 row norm overflowed f32
//   bit 3 (8u): squared L2 centroid norm overflowed f32
constant constexpr uint FAULT_INPUT_NONFINITE = 1u;
constant constexpr uint FAULT_BF16_OVERFLOW   = 2u;
constant constexpr uint FAULT_ROW_NORM_OVF    = 4u;
constant constexpr uint FAULT_CENT_NORM_OVF   = 8u;

inline bool f32_bits_is_nonfinite(uint bits) {
    return (bits & 0x7fffffffu) >= 0x7f800000u;
}

inline bool bf16_bits_is_nonfinite(ushort bits) {
    return (bits & 0x7fffu) >= 0x7f80u;
}

// IEEE-754 round-to-nearest-even f32 bits -> bf16 bits, flushing f32 subnormals
// to signed zero to match Apple Silicon GPU bfloat hardware semantics.
inline ushort f32_bits_to_bf16_rte(uint bits) {
    const uint mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) {
        return (ushort)((bits >> 16u) | 0x0040u);
    }
    if (mag < 0x00800000u) {
        return (ushort)((bits >> 16u) & 0x8000u);
    }
    return (ushort)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

inline float bf16_bits_to_f32_clean(ushort bits) {
    const ushort clean = ((bits & 0x7f80u) == 0u) ? (ushort)(bits & 0x8000u) : bits;
    return as_type<float>((uint)clean << 16u);
}

// Fixed-order 256-thread tree sum in threadgroup SRAM.
inline float tg_reduce_sum_256(float val, uint tid, threadgroup float (&scratch)[256]) {
    scratch[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float out = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return out;
}

inline float tg_reduce_max_256(float val, uint tid, threadgroup float (&scratch)[256]) {
    scratch[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float out = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return out;
}

// Exact L2 distance refinement when `||x||^2 + ||c||^2 - 2<x,c>` is near zero:
// eliminates multi-ULP cancellation noise and guarantees exact `0.0f` self-distance
// at `\tilde{x}_i == \tilde{c}_j` across all tile/reduction orders.
inline float finalize_row_dist_bf16(
    device const bfloat *X_bf16,
    device const bfloat *C_bf16,
    uint row,
    uint best_l,
    uint d,
    float rn,
    float best_s,
    uint spherical)
{
    if (spherical != 0u) {
        return 2.0f + best_s;
    }
    const float raw = rn + best_s;
    const float scale = max(rn, 1.0e-30f);
    if (fabs(raw) <= 1024.0f * 1.1920929e-7f * scale) {
        float exact_d = 0.0f;
        const ulong x_base = (ulong)row * (ulong)d;
        const ulong c_base = (ulong)best_l * (ulong)d;
        for (ulong col = 0ul; col < (ulong)d; ++col) {
            const float diff = float(X_bf16[x_base + col]) - float(C_bf16[c_base + col]);
            exact_d += diff * diff;
        }
        return exact_d;
    }
    return max(raw, 0.0f);
}

inline float finalize_row_dist_f32(
    device const float *X_f32,
    device const float *C_f32,
    uint row,
    uint best_l,
    uint d,
    float rn,
    float best_s,
    uint spherical)
{
    if (spherical != 0u) {
        return 2.0f + best_s;
    }
    const float raw = rn + best_s;
    const float scale = max(rn, 1.0e-30f);
    if (fabs(raw) <= 1024.0f * 1.1920929e-7f * scale) {
        float exact_d = 0.0f;
        const ulong x_base = (ulong)row * (ulong)d;
        const ulong c_base = (ulong)best_l * (ulong)d;
        for (ulong col = 0ul; col < (ulong)d; ++col) {
            const float diff = X_f32[x_base + col] - C_f32[c_base + col];
            exact_d += diff * diff;
        }
        return exact_d;
    }
    return max(raw, 0.0f);
}

// =============================================================================
// 1 & 2. Centroid preparation (`bf16` and `f32`)
// =============================================================================

kernel void emeans_prepare_centroids_bf16(
    device const float *centroids [[buffer(0)]],
    device ushort *c_eff_bf16_bits [[buffer(1)]],
    device float *norm2_c [[buffer(2)]],
    device atomic_uint *fault_flags [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    constant uint &spherical [[buffer(6)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (j >= k) return;
    threadgroup float scratch[256];

    const ulong base = (ulong)j * (ulong)d;
    uint local_fault = 0u;
    float local_max_abs = 0.0f;

    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float v = centroids[base + col];
        const uint bits = as_type<uint>(v);
        if (f32_bits_is_nonfinite(bits)) {
            local_fault |= FAULT_INPUT_NONFINITE;
        } else {
            local_max_abs = max(local_max_abs, fabs(v));
        }
    }

    float inv_norm = 1.0f;
    float max_abs = 1.0f;
    if (spherical != 0u) {
        max_abs = tg_reduce_max_256(local_max_abs, tid, scratch);
        float w_sq_sum = 0.0f;
        if (max_abs > 0.0f) {
            for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
                const float v = centroids[base + col];
                const uint bits = as_type<uint>(v);
                const float clean = f32_bits_is_nonfinite(bits) ? 0.0f : precise::divide(v, max_abs);
                w_sq_sum += clean * clean;
            }
        }
        const float total_w_sq = tg_reduce_sum_256(w_sq_sum, tid, scratch);
        inv_norm = (max_abs > 0.0f && total_w_sq > 0.0f) ? precise::rsqrt(total_w_sq) : 0.0f;
    }

    float sq_part = 0.0f;
    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float raw_v = centroids[base + col];
        const uint raw_bits = as_type<uint>(raw_v);
        float v = f32_bits_is_nonfinite(raw_bits) ? 0.0f : raw_v;
        if (spherical != 0u) {
            v = (max_abs > 0.0f) ? (precise::divide(v, max_abs) * inv_norm) : 0.0f;
        }
        const ushort bf = f32_bits_to_bf16_rte(as_type<uint>(v));
        ushort safe_bf = bf;
        if (bf16_bits_is_nonfinite(bf)) {
            local_fault |= FAULT_BF16_OVERFLOW;
            safe_bf = 0u;
        }
        c_eff_bf16_bits[base + col] = safe_bf;
        const float qf = bf16_bits_to_f32_clean(safe_bf);
        sq_part += qf * qf;
    }

    const float cn = tg_reduce_sum_256(sq_part, tid, scratch);
    if (tid == 0u) {
        if (f32_bits_is_nonfinite(as_type<uint>(cn))) {
            local_fault |= FAULT_CENT_NORM_OVF;
            norm2_c[j] = 0.0f;
        } else {
            norm2_c[j] = cn;
        }
    }
    if (local_fault != 0u) {
        atomic_fetch_or_explicit(fault_flags, local_fault, memory_order_relaxed);
    }
}

kernel void emeans_prepare_centroids_f32(
    device const float *centroids [[buffer(0)]],
    device float *c_eff_f32 [[buffer(1)]],
    device float *norm2_c [[buffer(2)]],
    device atomic_uint *fault_flags [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    constant uint &spherical [[buffer(6)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (j >= k) return;
    threadgroup float scratch[256];

    const ulong base = (ulong)j * (ulong)d;
    uint local_fault = 0u;
    float local_max_abs = 0.0f;

    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float v = centroids[base + col];
        const uint bits = as_type<uint>(v);
        if (f32_bits_is_nonfinite(bits)) {
            local_fault |= FAULT_INPUT_NONFINITE;
        } else {
            local_max_abs = max(local_max_abs, fabs(v));
        }
    }

    float inv_norm = 1.0f;
    float max_abs = 1.0f;
    if (spherical != 0u) {
        max_abs = tg_reduce_max_256(local_max_abs, tid, scratch);
        float w_sq_sum = 0.0f;
        if (max_abs > 0.0f) {
            for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
                const float v = centroids[base + col];
                const uint bits = as_type<uint>(v);
                const float clean = f32_bits_is_nonfinite(bits) ? 0.0f : precise::divide(v, max_abs);
                w_sq_sum += clean * clean;
            }
        }
        const float total_w_sq = tg_reduce_sum_256(w_sq_sum, tid, scratch);
        inv_norm = (max_abs > 0.0f && total_w_sq > 0.0f) ? precise::rsqrt(total_w_sq) : 0.0f;
    }

    float sq_part = 0.0f;
    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float raw_v = centroids[base + col];
        const uint raw_bits = as_type<uint>(raw_v);
        float v = f32_bits_is_nonfinite(raw_bits) ? 0.0f : raw_v;
        if (spherical != 0u) {
            v = (max_abs > 0.0f) ? (precise::divide(v, max_abs) * inv_norm) : 0.0f;
        }
        c_eff_f32[base + col] = v;
        sq_part += v * v;
    }

    const float cn = tg_reduce_sum_256(sq_part, tid, scratch);
    if (tid == 0u) {
        if (f32_bits_is_nonfinite(as_type<uint>(cn))) {
            local_fault |= FAULT_CENT_NORM_OVF;
            norm2_c[j] = 0.0f;
        } else {
            norm2_c[j] = cn;
        }
    }
    if (local_fault != 0u) {
        atomic_fetch_or_explicit(fault_flags, local_fault, memory_order_relaxed);
    }
}

// =============================================================================
// 3, 4, 5 & 6. Row staging (upload / streaming) and resident row gathering
// =============================================================================

kernel void emeans_stage_rows_bf16(
    device const ushort *src_bf16_bits [[buffer(0)]],
    device ushort *dst_bf16_bits [[buffer(1)]],
    device float *dst_f32 [[buffer(2)]],
    device float *dst_norm2 [[buffer(3)]],
    device atomic_uint *fault_flags [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    constant uint &write_bf16 [[buffer(7)]],
    constant uint &write_f32 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (row >= n) return;
    threadgroup float scratch[256];

    const ulong base = (ulong)row * (ulong)d;
    uint local_fault = 0u;
    float sq_part = 0.0f;

    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const ushort raw = src_bf16_bits[base + col];
        ushort safe_bf = raw;
        if (bf16_bits_is_nonfinite(raw)) {
            local_fault |= FAULT_INPUT_NONFINITE;
            safe_bf = 0u;
        } else if ((raw & 0x7f80u) == 0u) {
            safe_bf = (ushort)(raw & 0x8000u);
        }
        const float vf = bf16_bits_to_f32_clean(safe_bf);
        if (write_bf16 != 0u) {
            dst_bf16_bits[base + col] = safe_bf;
        }
        if (write_f32 != 0u) {
            dst_f32[base + col] = vf;
        }
        sq_part += vf * vf;
    }

    const float rn = tg_reduce_sum_256(sq_part, tid, scratch);
    if (tid == 0u) {
        if (f32_bits_is_nonfinite(as_type<uint>(rn))) {
            local_fault |= FAULT_ROW_NORM_OVF;
            dst_norm2[row] = 0.0f;
        } else {
            dst_norm2[row] = rn;
        }
    }
    if (local_fault != 0u) {
        atomic_fetch_or_explicit(fault_flags, local_fault, memory_order_relaxed);
    }
}

kernel void emeans_stage_rows_f32(
    device const float *src_f32 [[buffer(0)]],
    device ushort *dst_bf16_bits [[buffer(1)]],
    device float *dst_f32 [[buffer(2)]],
    device float *dst_norm2 [[buffer(3)]],
    device atomic_uint *fault_flags [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    constant uint &write_bf16 [[buffer(7)]],
    constant uint &write_f32 [[buffer(8)]],
    constant uint &norm_from_bf16 [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (row >= n) return;
    threadgroup float scratch[256];

    const ulong base = (ulong)row * (ulong)d;
    uint local_fault = 0u;
    float sq_part = 0.0f;

    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float raw = src_f32[base + col];
        const uint raw_bits = as_type<uint>(raw);
        float vf = raw;
        if (f32_bits_is_nonfinite(raw_bits)) {
            local_fault |= FAULT_INPUT_NONFINITE;
            vf = 0.0f;
        }
        ushort safe_bf = 0u;
        if (write_bf16 != 0u || norm_from_bf16 != 0u) {
            const ushort bf = f32_bits_to_bf16_rte(as_type<uint>(vf));
            if (bf16_bits_is_nonfinite(bf)) {
                local_fault |= FAULT_BF16_OVERFLOW;
                safe_bf = 0u;
            } else {
                safe_bf = bf;
            }
            if (write_bf16 != 0u) {
                dst_bf16_bits[base + col] = safe_bf;
            }
        }
        if (write_f32 != 0u) {
            dst_f32[base + col] = vf;
        }
        const float nv = (norm_from_bf16 != 0u) ? bf16_bits_to_f32_clean(safe_bf) : vf;
        sq_part += nv * nv;
    }

    const float rn = tg_reduce_sum_256(sq_part, tid, scratch);
    if (tid == 0u) {
        if (f32_bits_is_nonfinite(as_type<uint>(rn))) {
            local_fault |= FAULT_ROW_NORM_OVF;
            dst_norm2[row] = 0.0f;
        } else {
            dst_norm2[row] = rn;
        }
    }
    if (local_fault != 0u) {
        atomic_fetch_or_explicit(fault_flags, local_fault, memory_order_relaxed);
    }
}

kernel void emeans_gather_rows_bf16(
    device const ushort *ds_bf16_bits [[buffer(0)]],
    device const float *ds_norm2 [[buffer(1)]],
    device const uint *order [[buffer(2)]],
    device ushort *chunk_bf16_bits [[buffer(3)]],
    device float *chunk_norm2 [[buffer(4)]],
    constant uint &ds_n [[buffer(5)]],
    constant uint &row_offset [[buffer(6)]],
    constant uint &chunk_rows [[buffer(7)]],
    constant uint &d [[buffer(8)]],
    constant uint &copy_norm2 [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (r >= chunk_rows || col >= d) return;
    const uint src_r = min(order[row_offset + r], ds_n - 1u);
    chunk_bf16_bits[(ulong)r * (ulong)d + (ulong)col] =
        ds_bf16_bits[(ulong)src_r * (ulong)d + (ulong)col];
    if (copy_norm2 != 0u && col == 0u) {
        chunk_norm2[r] = ds_norm2[src_r];
    }
}

kernel void emeans_gather_rows_f32(
    device const float *ds_f32 [[buffer(0)]],
    device const float *ds_norm2 [[buffer(1)]],
    device const uint *order [[buffer(2)]],
    device float *chunk_f32 [[buffer(3)]],
    device float *chunk_norm2 [[buffer(4)]],
    constant uint &ds_n [[buffer(5)]],
    constant uint &row_offset [[buffer(6)]],
    constant uint &chunk_rows [[buffer(7)]],
    constant uint &d [[buffer(8)]],
    constant uint &copy_norm2 [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (r >= chunk_rows || col >= d) return;
    const uint src_r = min(order[row_offset + r], ds_n - 1u);
    chunk_f32[(ulong)r * (ulong)d + (ulong)col] =
        ds_f32[(ulong)src_r * (ulong)d + (ulong)col];
    if (copy_norm2 != 0u && col == 0u) {
        chunk_norm2[r] = ds_norm2[src_r];
    }
}

// =============================================================================
// 7 & 8. 64x64 execution_simdgroups<4> Cooperative TensorOps Assignment
// =============================================================================

template <typename ElemT>
inline void emeans_assign_tensorops_body(
    device ElemT *X_assign,
    device ElemT *C_eff,
    device const float *row_norm2,
    device const float *norm2_c,
    device uint *chunk_labels,
    device float *chunk_dists,
    device uint *epoch_labels,
    device float *part_scores,
    device uint *part_labels,
    uint M,
    uint d,
    uint c_start,
    uint c_end,
    uint part_c,
    uint split_c,
    uint spherical,
    uint has_epoch_labels,
    uint epoch_off,
    threadgroup float (&tg_score)[4][64],
    threadgroup uint (&tg_label)[4][64],
    uint2 tgpig,
    uint tid,
    uint sid,
    uint lane)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    const uint ty_u = tgpig.x * (uint)BM;
    if (ty_u >= M || c_start >= c_end) return;

    const uint c_len = c_end - c_start;
    const int ty = (int)ty_u;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, dynamic_length_v<int>, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;

    device ElemT *C_part = C_eff + (ulong)c_start * (ulong)d;
    auto mA = tensor(X_assign, dextents<int, 2>{(int)d, (int)M}, array<int, 2>{1, (int)d});
    auto mB = tensor(C_part, dextents<int, 2>{(int)d, (int)c_len}, array<int, 2>{1, (int)d});
    auto tA = mA.slice(0, ty);
    auto tB0 = mB.slice(0, 0);

    using TA = metal::remove_addrspace_t<decltype(tA)>;
    using TB = metal::remove_addrspace_t<decltype(tB0)>;
    auto cT = op.template get_destination_cooperative_tensor<TA, TB, float>();
    auto bestScoreT = op.template get_destination_cooperative_tensor<TA, TB, float>();
    auto bestLabelT = op.template get_destination_cooperative_tensor<TA, TB, int32_t>();

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < bestScoreT.get_capacity(); ++i) {
        bestScoreT.set(i, EMEANS_MAX_SENTINEL);
        bestLabelT.set(i, -1);
    }

    // Zero threadgroup barriers across the entire cluster-tile loop: accumulators
    // and running (score, label) pairs stay strictly in registers.
    for (uint tx_u = 0u; tx_u < c_len; tx_u += (uint)BN) {
        auto tB = mB.slice(0, (int)tx_u);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            cT.set(i, 0.0f);
        }
        op.run(tA, tB, cT);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            const auto idx = cT.get_multidimensional_index(i);
            const uint ln = (uint)idx[0];
            const uint lm = (uint)idx[1];
            const uint row = ty_u + lm;
            const uint loc_c = tx_u + ln;
            if (row >= M || loc_c >= c_len) continue;

            const uint glob_c = c_start + loc_c;
            const float dot = cT[i];
            const float sc = (spherical != 0u) ? (-2.0f * dot) : (norm2_c[glob_c] - 2.0f * dot);
            const float cur_s = bestScoreT[i];
            const int cur_l = bestLabelT[i];
            if (cur_l < 0 || sc < cur_s || (sc == cur_s && glob_c < (uint)cur_l)) {
                bestScoreT[i] = sc;
                bestLabelT[i] = (int)glob_c;
            }
        }
    }

    // Single post-loop fold into 2 KiB threadgroup SRAM.
    for (uint r = tid; r < 256u; r += 128u) {
        tg_score[r >> 6u][r & 63u] = EMEANS_MAX_SENTINEL;
        tg_label[r >> 6u][r & 63u] = EMEANS_INVALID_LABEL;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0u; step < 32u; ++step) {
        if (lane == step && sid < 4u) {
            for (uint16_t i = 0; i < bestScoreT.get_capacity(); ++i) {
                if (!bestScoreT.is_valid_element(i)) continue;
                const int lbl = bestLabelT[i];
                if (lbl < 0) continue;
                const auto idx = bestScoreT.get_multidimensional_index(i);
                const uint lm = (uint)idx[1];
                if (lm < 64u) {
                    const float sc = bestScoreT[i];
                    const uint ul = (uint)lbl;
                    const float ps = tg_score[sid][lm];
                    const uint pl = tg_label[sid][lm];
                    if (pl == EMEANS_INVALID_LABEL || sc < ps || (sc == ps && ul < pl)) {
                        tg_score[sid][lm] = sc;
                        tg_label[sid][lm] = ul;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        const uint row = ty_u + tid;
        if (row < M) {
            float best_s = EMEANS_MAX_SENTINEL;
            uint best_l = EMEANS_INVALID_LABEL;
            for (uint sg = 0u; sg < 4u; ++sg) {
                const float sc = tg_score[sg][tid];
                const uint ul = tg_label[sg][tid];
                if (ul != EMEANS_INVALID_LABEL
                    && (best_l == EMEANS_INVALID_LABEL || sc < best_s || (sc == best_s && ul < best_l)))
                {
                    best_s = sc;
                    best_l = ul;
                }
            }
            if (best_l == EMEANS_INVALID_LABEL) {
                best_l = c_start;
                best_s = 0.0f;
            }
            if (split_c <= 1u) {
                chunk_labels[row] = best_l;
                if (has_epoch_labels != 0u) {
                    epoch_labels[epoch_off + row] = best_l;
                }
            } else {
                const ulong p_off = (ulong)part_c * (ulong)M + (ulong)row;
                part_scores[p_off] = best_s;
                part_labels[p_off] = best_l;
            }
        }
    }
}

kernel void emeans_assign_bf16_tensorops(
    device bfloat *X_bf16 [[buffer(0)]],
    device bfloat *C_bf16 [[buffer(1)]],
    device const float *row_norm2 [[buffer(2)]],
    device const float *norm2_c [[buffer(3)]],
    device uint *chunk_labels [[buffer(4)]],
    device float *chunk_dists [[buffer(5)]],
    device uint *epoch_labels [[buffer(6)]],
    device float *part_scores [[buffer(7)]],
    device uint *part_labels [[buffer(8)]],
    constant uint &M [[buffer(9)]],
    constant uint &d [[buffer(10)]],
    constant uint &c_start [[buffer(11)]],
    constant uint &c_end [[buffer(12)]],
    constant uint &part_c [[buffer(13)]],
    constant uint &split_c [[buffer(14)]],
    constant uint &spherical [[buffer(15)]],
    constant uint &has_epoch_labels [[buffer(16)]],
    constant uint &epoch_off [[buffer(17)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float tg_score[4][64];
    threadgroup uint tg_label[4][64];
    emeans_assign_tensorops_body<bfloat>(
        X_bf16, C_bf16, row_norm2, norm2_c, chunk_labels, chunk_dists, epoch_labels,
        part_scores, part_labels, M, d, c_start, c_end, part_c, split_c, spherical,
        has_epoch_labels, epoch_off, tg_score, tg_label, tgpig, tid, sid, lane);
    if (split_c <= 1u && tid < 64u) {
        const uint row = tgpig.x * 64u + tid;
        if (row < M && c_start < c_end) {
            float best_s = EMEANS_MAX_SENTINEL;
            uint best_l = EMEANS_INVALID_LABEL;
            for (uint sg = 0u; sg < 4u; ++sg) {
                const float sc = tg_score[sg][tid];
                const uint ul = tg_label[sg][tid];
                if (ul != EMEANS_INVALID_LABEL
                    && (best_l == EMEANS_INVALID_LABEL || sc < best_s || (sc == best_s && ul < best_l)))
                {
                    best_s = sc;
                    best_l = ul;
                }
            }
            if (best_l == EMEANS_INVALID_LABEL) {
                best_l = c_start;
                best_s = 0.0f;
            }
            chunk_dists[row] = finalize_row_dist_bf16(
                X_bf16, C_bf16, row, best_l, d, row_norm2[row], best_s, spherical);
        }
    }
}

kernel void emeans_assign_f32_tensorops(
    device float *X_f32 [[buffer(0)]],
    device float *C_f32 [[buffer(1)]],
    device const float *row_norm2 [[buffer(2)]],
    device const float *norm2_c [[buffer(3)]],
    device uint *chunk_labels [[buffer(4)]],
    device float *chunk_dists [[buffer(5)]],
    device uint *epoch_labels [[buffer(6)]],
    device float *part_scores [[buffer(7)]],
    device uint *part_labels [[buffer(8)]],
    constant uint &M [[buffer(9)]],
    constant uint &d [[buffer(10)]],
    constant uint &c_start [[buffer(11)]],
    constant uint &c_end [[buffer(12)]],
    constant uint &part_c [[buffer(13)]],
    constant uint &split_c [[buffer(14)]],
    constant uint &spherical [[buffer(15)]],
    constant uint &has_epoch_labels [[buffer(16)]],
    constant uint &epoch_off [[buffer(17)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float tg_score[4][64];
    threadgroup uint tg_label[4][64];
    emeans_assign_tensorops_body<float>(
        X_f32, C_f32, row_norm2, norm2_c, chunk_labels, chunk_dists, epoch_labels,
        part_scores, part_labels, M, d, c_start, c_end, part_c, split_c, spherical,
        has_epoch_labels, epoch_off, tg_score, tg_label, tgpig, tid, sid, lane);
    if (split_c <= 1u && tid < 64u) {
        const uint row = tgpig.x * 64u + tid;
        if (row < M && c_start < c_end) {
            float best_s = EMEANS_MAX_SENTINEL;
            uint best_l = EMEANS_INVALID_LABEL;
            for (uint sg = 0u; sg < 4u; ++sg) {
                const float sc = tg_score[sg][tid];
                const uint ul = tg_label[sg][tid];
                if (ul != EMEANS_INVALID_LABEL
                    && (best_l == EMEANS_INVALID_LABEL || sc < best_s || (sc == best_s && ul < best_l)))
                {
                    best_s = sc;
                    best_l = ul;
                }
            }
            if (best_l == EMEANS_INVALID_LABEL) {
                best_l = c_start;
                best_s = 0.0f;
            }
            chunk_dists[row] = finalize_row_dist_f32(
                X_f32, C_f32, row, best_l, d, row_norm2[row], best_s, spherical);
        }
    }
}

// =============================================================================
// 9 & 10. 32x32 Portable simdgroup_float8x8 Assignment (`bf16` and `f32`)
// =============================================================================

kernel void emeans_assign_bf16_simdgroup(
    device const bfloat *X_bf16 [[buffer(0)]],
    device const bfloat *C_bf16 [[buffer(1)]],
    device const float *row_norm2 [[buffer(2)]],
    device const float *norm2_c [[buffer(3)]],
    device uint *chunk_labels [[buffer(4)]],
    device float *chunk_dists [[buffer(5)]],
    device uint *epoch_labels [[buffer(6)]],
    device float *part_scores [[buffer(7)]],
    device uint *part_labels [[buffer(8)]],
    constant uint &M [[buffer(9)]],
    constant uint &d [[buffer(10)]],
    constant uint &c_start [[buffer(11)]],
    constant uint &c_end [[buffer(12)]],
    constant uint &part_c [[buffer(13)]],
    constant uint &split_c [[buffer(14)]],
    constant uint &spherical [[buffer(15)]],
    constant uint &has_epoch_labels [[buffer(16)]],
    constant uint &epoch_off [[buffer(17)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_a[4][64];
    threadgroup float sg_b[4][256];
    threadgroup float sg_c[4][256];

    const uint row0 = tgpig.x * 32u + sid * 8u;
    if (sid >= 4u || row0 >= M || c_start >= c_end) return;

    const uint c_len = c_end - c_start;
    const uint my_r = lane >> 2u;
    const uint my_part = lane & 3u;
    float best_s = EMEANS_MAX_SENTINEL;
    uint best_l = EMEANS_INVALID_LABEL;

    for (uint tx = 0u; tx < c_len; tx += 32u) {
        simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (ulong k0 = 0ul; k0 < (ulong)d; k0 += 8ul) {
            for (uint i = lane; i < 64u; i += 32u) {
                const uint r = i >> 3u;
                const uint kk = i & 7u;
                sg_a[sid][i] = (row0 + r < M && k0 + (ulong)kk < (ulong)d)
                    ? float(X_bf16[(ulong)(row0 + r) * (ulong)d + k0 + (ulong)kk])
                    : 0.0f;
            }
            for (uint i = lane; i < 256u; i += 32u) {
                const uint kk = i >> 5u;
                const uint cn = i & 31u;
                const uint loc_c = tx + cn;
                sg_b[sid][i] = (loc_c < c_len && k0 + (ulong)kk < (ulong)d)
                    ? float(C_bf16[(ulong)(c_start + loc_c) * (ulong)d + k0 + (ulong)kk])
                    : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 a_mat, b0, b1, b2, b3;
            simdgroup_load(a_mat, sg_a[sid], 8);
            simdgroup_load(b0, sg_b[sid] + 0, 32);
            simdgroup_load(b1, sg_b[sid] + 8, 32);
            simdgroup_load(b2, sg_b[sid] + 16, 32);
            simdgroup_load(b3, sg_b[sid] + 24, 32);
            simdgroup_multiply_accumulate(acc0, a_mat, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_mat, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a_mat, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a_mat, b3, acc3);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(acc0, sg_c[sid] + 0, 32);
        simdgroup_store(acc1, sg_c[sid] + 8, 32);
        simdgroup_store(acc2, sg_c[sid] + 16, 32);
        simdgroup_store(acc3, sg_c[sid] + 24, 32);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 + my_r < M) {
            for (uint j = 0u; j < 8u; ++j) {
                const uint cn = my_part * 8u + j;
                const uint loc_c = tx + cn;
                if (loc_c < c_len) {
                    const uint glob_c = c_start + loc_c;
                    const float dot = sg_c[sid][my_r * 32u + cn];
                    const float sc = (spherical != 0u) ? (-2.0f * dot) : (norm2_c[glob_c] - 2.0f * dot);
                    if (best_l == EMEANS_INVALID_LABEL || sc < best_s || (sc == best_s && glob_c < best_l)) {
                        best_s = sc;
                        best_l = glob_c;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ushort off = 1u; off <= 2u; off <<= 1u) {
        const float os = simd_shuffle_xor(best_s, off);
        const uint ol = simd_shuffle_xor(best_l, off);
        if (ol != EMEANS_INVALID_LABEL
            && (best_l == EMEANS_INVALID_LABEL || os < best_s || (os == best_s && ol < best_l)))
        {
            best_s = os;
            best_l = ol;
        }
    }

    const uint row = row0 + my_r;
    if (my_part == 0u && row < M) {
        if (best_l == EMEANS_INVALID_LABEL) {
            best_l = c_start;
            best_s = 0.0f;
        }
        if (split_c <= 1u) {
            chunk_labels[row] = best_l;
            chunk_dists[row] = finalize_row_dist_bf16(
                X_bf16, C_bf16, row, best_l, d, row_norm2[row], best_s, spherical);
            if (has_epoch_labels != 0u) {
                epoch_labels[epoch_off + row] = best_l;
            }
        } else {
            const ulong p_off = (ulong)part_c * (ulong)M + (ulong)row;
            part_scores[p_off] = best_s;
            part_labels[p_off] = best_l;
        }
    }
}

kernel void emeans_assign_f32_simdgroup(
    device const float *X_f32 [[buffer(0)]],
    device const float *C_f32 [[buffer(1)]],
    device const float *row_norm2 [[buffer(2)]],
    device const float *norm2_c [[buffer(3)]],
    device uint *chunk_labels [[buffer(4)]],
    device float *chunk_dists [[buffer(5)]],
    device uint *epoch_labels [[buffer(6)]],
    device float *part_scores [[buffer(7)]],
    device uint *part_labels [[buffer(8)]],
    constant uint &M [[buffer(9)]],
    constant uint &d [[buffer(10)]],
    constant uint &c_start [[buffer(11)]],
    constant uint &c_end [[buffer(12)]],
    constant uint &part_c [[buffer(13)]],
    constant uint &split_c [[buffer(14)]],
    constant uint &spherical [[buffer(15)]],
    constant uint &has_epoch_labels [[buffer(16)]],
    constant uint &epoch_off [[buffer(17)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_a[4][64];
    threadgroup float sg_b[4][256];
    threadgroup float sg_c[4][256];

    const uint row0 = tgpig.x * 32u + sid * 8u;
    if (sid >= 4u || row0 >= M || c_start >= c_end) return;

    const uint c_len = c_end - c_start;
    const uint my_r = lane >> 2u;
    const uint my_part = lane & 3u;
    float best_s = EMEANS_MAX_SENTINEL;
    uint best_l = EMEANS_INVALID_LABEL;

    for (uint tx = 0u; tx < c_len; tx += 32u) {
        simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (ulong k0 = 0ul; k0 < (ulong)d; k0 += 8ul) {
            for (uint i = lane; i < 64u; i += 32u) {
                const uint r = i >> 3u;
                const uint kk = i & 7u;
                sg_a[sid][i] = (row0 + r < M && k0 + (ulong)kk < (ulong)d)
                    ? X_f32[(ulong)(row0 + r) * (ulong)d + k0 + (ulong)kk]
                    : 0.0f;
            }
            for (uint i = lane; i < 256u; i += 32u) {
                const uint kk = i >> 5u;
                const uint cn = i & 31u;
                const uint loc_c = tx + cn;
                sg_b[sid][i] = (loc_c < c_len && k0 + (ulong)kk < (ulong)d)
                    ? C_f32[(ulong)(c_start + loc_c) * (ulong)d + k0 + (ulong)kk]
                    : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 a_mat, b0, b1, b2, b3;
            simdgroup_load(a_mat, sg_a[sid], 8);
            simdgroup_load(b0, sg_b[sid] + 0, 32);
            simdgroup_load(b1, sg_b[sid] + 8, 32);
            simdgroup_load(b2, sg_b[sid] + 16, 32);
            simdgroup_load(b3, sg_b[sid] + 24, 32);
            simdgroup_multiply_accumulate(acc0, a_mat, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_mat, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a_mat, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a_mat, b3, acc3);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(acc0, sg_c[sid] + 0, 32);
        simdgroup_store(acc1, sg_c[sid] + 8, 32);
        simdgroup_store(acc2, sg_c[sid] + 16, 32);
        simdgroup_store(acc3, sg_c[sid] + 24, 32);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (row0 + my_r < M) {
            for (uint j = 0u; j < 8u; ++j) {
                const uint cn = my_part * 8u + j;
                const uint loc_c = tx + cn;
                if (loc_c < c_len) {
                    const uint glob_c = c_start + loc_c;
                    const float dot = sg_c[sid][my_r * 32u + cn];
                    const float sc = (spherical != 0u) ? (-2.0f * dot) : (norm2_c[glob_c] - 2.0f * dot);
                    if (best_l == EMEANS_INVALID_LABEL || sc < best_s || (sc == best_s && glob_c < best_l)) {
                        best_s = sc;
                        best_l = glob_c;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ushort off = 1u; off <= 2u; off <<= 1u) {
        const float os = simd_shuffle_xor(best_s, off);
        const uint ol = simd_shuffle_xor(best_l, off);
        if (ol != EMEANS_INVALID_LABEL
            && (best_l == EMEANS_INVALID_LABEL || os < best_s || (os == best_s && ol < best_l)))
        {
            best_s = os;
            best_l = ol;
        }
    }

    const uint row = row0 + my_r;
    if (my_part == 0u && row < M) {
        if (best_l == EMEANS_INVALID_LABEL) {
            best_l = c_start;
            best_s = 0.0f;
        }
        if (split_c <= 1u) {
            chunk_labels[row] = best_l;
            chunk_dists[row] = finalize_row_dist_f32(
                X_f32, C_f32, row, best_l, d, row_norm2[row], best_s, spherical);
            if (has_epoch_labels != 0u) {
                epoch_labels[epoch_off + row] = best_l;
            }
        } else {
            const ulong p_off = (ulong)part_c * (ulong)M + (ulong)row;
            part_scores[p_off] = best_s;
            part_labels[p_off] = best_l;
        }
    }
}

// =============================================================================
// 11. Split-C Partition Reduction
// =============================================================================

kernel void emeans_reduce_split_c(
    device const float *part_scores [[buffer(0)]],
    device const uint *part_labels [[buffer(1)]],
    device const float *row_norm2 [[buffer(2)]],
    device const void *X_assign [[buffer(3)]],
    device const void *C_eff [[buffer(4)]],
    device uint *chunk_labels [[buffer(5)]],
    device float *chunk_dists [[buffer(6)]],
    device uint *epoch_labels [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    constant uint &d [[buffer(9)]],
    constant uint &split_c [[buffer(10)]],
    constant uint &spherical [[buffer(11)]],
    constant uint &use_bf16_assign [[buffer(12)]],
    constant uint &has_epoch_labels [[buffer(13)]],
    constant uint &epoch_off [[buffer(14)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= M) return;

    float best_s = EMEANS_MAX_SENTINEL;
    uint best_l = 0u;
    bool have = false;
    for (uint p = 0u; p < split_c; ++p) {
        const ulong off = (ulong)p * (ulong)M + (ulong)row;
        const float sc = part_scores[off];
        const uint ul = part_labels[off];
        if (!have || sc < best_s || (sc == best_s && ul < best_l)) {
            best_s = sc;
            best_l = ul;
            have = true;
        }
    }

    chunk_labels[row] = best_l;
    if (use_bf16_assign != 0u) {
        chunk_dists[row] = finalize_row_dist_bf16(
            (device const bfloat *)X_assign,
            (device const bfloat *)C_eff,
            row, best_l, d, row_norm2[row], best_s, spherical);
    } else {
        chunk_dists[row] = finalize_row_dist_f32(
            (device const float *)X_assign,
            (device const float *)C_eff,
            row, best_l, d, row_norm2[row], best_s, spherical);
    }
    if (has_epoch_labels != 0u) {
        epoch_labels[epoch_off + row] = best_l;
    }
}

// =============================================================================
// 12, 13, 14 & 15. Deterministic 8-bit `ulong` GPU Radix Sort
// =============================================================================

kernel void emeans_pack_label_keys(
    device const uint *chunk_labels [[buffer(0)]],
    device ulong *keys_out [[buffer(1)]],
    device uint *overflow_slice_count [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    uint r [[thread_position_in_grid]])
{
    if (r == 0u) {
        overflow_slice_count[0] = 0u;
    }
    if (r >= M) return;
    keys_out[r] = ((ulong)chunk_labels[r] << 16ul) | (ulong)r;
}

kernel void emeans_radix_hist(
    device const ulong *keys_in [[buffer(0)]],
    device uint *radix_hist [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &num_tiles [[buffer(3)]],
    constant uint &shift [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tg >= num_tiles) return;
    threadgroup atomic_uint local_hist[256];
    atomic_store_explicit(&local_hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint idx = tg * 256u + tid;
    if (idx < count) {
        const uint bin = (uint)((keys_in[idx] >> (ulong)shift) & 0xfful);
        atomic_fetch_add_explicit(&local_hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    radix_hist[(ulong)tid * (ulong)num_tiles + (ulong)tg] =
        atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
}

kernel void emeans_radix_prefix(
    device uint *radix_hist [[buffer(0)]],
    constant uint &num_tiles [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint bin_totals[256];
    threadgroup uint bin_offsets[256];

    uint running = 0u;
    for (uint t = 0u; t < num_tiles; ++t) {
        const ulong off = (ulong)tid * (ulong)num_tiles + (ulong)t;
        const uint c = radix_hist[off];
        radix_hist[off] = running;
        running += c;
    }
    bin_totals[tid] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint acc = 0u;
        for (uint b = 0u; b < 256u; ++b) {
            bin_offsets[b] = acc;
            acc += bin_totals[b];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = bin_offsets[tid];
    for (uint t = 0u; t < num_tiles; ++t) {
        const ulong off = (ulong)tid * (ulong)num_tiles + (ulong)t;
        radix_hist[off] += base;
    }
}

kernel void emeans_radix_scatter(
    device const ulong *keys_in [[buffer(0)]],
    device const uint *radix_hist [[buffer(1)]],
    device ulong *keys_out [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &num_tiles [[buffer(4)]],
    constant uint &shift [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tg >= num_tiles) return;
    threadgroup uchar tg_bins[256];
    threadgroup uint tg_base[256];

    const uint idx = tg * 256u + tid;
    const bool valid = idx < count;
    const ulong key = valid ? keys_in[idx] : 0ul;
    const uchar bin = valid ? (uchar)((key >> (ulong)shift) & 0xfful) : (uchar)0u;

    tg_bins[tid] = bin;
    tg_base[tid] = radix_hist[(ulong)tid * (ulong)num_tiles + (ulong)tg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = 0u;
        for (uint q = 0u; q < tid; ++q) {
            rank += (tg_bins[q] == bin) ? 1u : 0u;
        }
        const uint dst = tg_base[(uint)bin] + rank;
        keys_out[dst] = key;
    }
}

// =============================================================================
// 16, 17, 18 & 19. CSR Bucket Bounds + 256-Row Slice Gather + Ascending Fold
// =============================================================================

kernel void emeans_find_bucket_bounds(
    device const ulong *sorted_keys [[buffer(0)]],
    device uint4 *bucket_info [[buffer(1)]],
    device uint2 *slice_meta [[buffer(2)]],
    device atomic_uint *overflow_slice_count [[buffer(3)]],
    device ulong *batch_counts [[buffer(4)]],
    device ulong *epoch_counts [[buffer(5)]],
    device ulong *pass_counts [[buffer(6)]],
    constant uint &M [[buffer(7)]],
    constant uint &k [[buffer(8)]],
    constant uint &final_epoch [[buffer(9)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= k) return;

    uint lo = 0u;
    uint hi = M;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1u);
        if ((uint)(sorted_keys[mid] >> 16ul) < j) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    const uint start = lo;

    hi = M;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1u);
        if ((uint)(sorted_keys[mid] >> 16ul) <= j) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    const uint end = lo;
    const uint n_j = end - start;

    if (n_j > 256u) {
        const uint num_slices = (n_j + 255u) >> 8u;
        const uint base = atomic_fetch_add_explicit(overflow_slice_count, num_slices, memory_order_relaxed);
        for (uint s = 0u; s < num_slices; ++s) {
            const uint s_start = start + (s << 8u);
            const uint s_end = min(s_start + 256u, end);
            slice_meta[base + s] = uint2(s_start, s_end);
        }
        bucket_info[j] = uint4(start, end, base, num_slices);
    } else {
        bucket_info[j] = uint4(start, end, 0u, 0u);
    }

    if (n_j > 0u) {
        batch_counts[j] += (ulong)n_j;
        epoch_counts[j] += (ulong)n_j;
        if (final_epoch != 0u) {
            pass_counts[j] += (ulong)n_j;
        }
    }
}

kernel void emeans_gather_slice_partials_bf16(
    device const ulong *sorted_keys [[buffer(0)]],
    device const bfloat *X_bf16 [[buffer(1)]],
    device const uint2 *slice_meta [[buffer(2)]],
    device const uint *overflow_slice_count [[buffer(3)]],
    device float *slice_partials [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint slice_id = gid.y;
    if (col >= d || slice_id >= overflow_slice_count[0]) return;

    const uint2 bounds = slice_meta[slice_id];
    float acc = 0.0f;
    for (uint p = bounds.x; p < bounds.y; ++p) {
        const uint local_r = (uint)(sorted_keys[p] & 0xfffful);
        acc += float(X_bf16[(ulong)local_r * (ulong)d + (ulong)col]);
    }
    slice_partials[(ulong)slice_id * (ulong)d + (ulong)col] = acc;
}

kernel void emeans_gather_slice_partials_f32(
    device const ulong *sorted_keys [[buffer(0)]],
    device const float *X_f32 [[buffer(1)]],
    device const uint2 *slice_meta [[buffer(2)]],
    device const uint *overflow_slice_count [[buffer(3)]],
    device float *slice_partials [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint slice_id = gid.y;
    if (col >= d || slice_id >= overflow_slice_count[0]) return;

    const uint2 bounds = slice_meta[slice_id];
    float acc = 0.0f;
    for (uint p = bounds.x; p < bounds.y; ++p) {
        const uint local_r = (uint)(sorted_keys[p] & 0xfffful);
        acc += X_f32[(ulong)local_r * (ulong)d + (ulong)col];
    }
    slice_partials[(ulong)slice_id * (ulong)d + (ulong)col] = acc;
}

kernel void emeans_fold_slice_partials(
    device const ulong *sorted_keys [[buffer(0)]],
    device const void *X_stats [[buffer(1)]],
    device const uint4 *bucket_info [[buffer(2)]],
    device const float *slice_partials [[buffer(3)]],
    device float *batch_sums [[buffer(4)]],
    device float *pass_sums [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &d [[buffer(7)]],
    constant uint &stats_is_bf16 [[buffer(8)]],
    constant uint &final_epoch [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint j = gid.y;
    if (col >= d || j >= k) return;

    const uint4 info = bucket_info[j];
    const uint start = info.x;
    const uint end = info.y;
    if (start >= end) return;

    const ulong out_idx = (ulong)j * (ulong)d + (ulong)col;
    const uint num_slices = info.w;
    if (num_slices == 0u) {
        float s_acc = 0.0f;
        if (stats_is_bf16 != 0u) {
            device const bfloat *xb = (device const bfloat *)X_stats;
            for (uint p = start; p < end; ++p) {
                const uint local_r = (uint)(sorted_keys[p] & 0xfffful);
                s_acc += float(xb[(ulong)local_r * (ulong)d + (ulong)col]);
            }
        } else {
            device const float *xf = (device const float *)X_stats;
            for (uint p = start; p < end; ++p) {
                const uint local_r = (uint)(sorted_keys[p] & 0xfffful);
                s_acc += xf[(ulong)local_r * (ulong)d + (ulong)col];
            }
        }
        batch_sums[out_idx] += s_acc;
        if (final_epoch != 0u) {
            pass_sums[out_idx] += s_acc;
        }
    } else {
        const uint base = info.z;
        float b_acc = batch_sums[out_idx];
        float p_acc = (final_epoch != 0u) ? pass_sums[out_idx] : 0.0f;
        for (uint s = 0u; s < num_slices; ++s) {
            const float part = slice_partials[(ulong)(base + s) * (ulong)d + (ulong)col];
            b_acc += part;
            if (final_epoch != 0u) {
                p_acc += part;
            }
        }
        batch_sums[out_idx] = b_acc;
        if (final_epoch != 0u) {
            pass_sums[out_idx] = p_acc;
        }
    }
}

// =============================================================================
// 20. Pure-Integer IEEE-754 binary64 Chunk Objective Reduction
// =============================================================================

inline ulong f32_to_f64_bits(float f) {
    const uint u = as_type<uint>(f);
    const ulong sign = (ulong)(u & 0x80000000u) << 32ul;
    const uint exp32 = (u >> 23u) & 0xffu;
    const uint mant32 = u & 0x007fffffu;
    if (exp32 == 0u) {
        if (mant32 == 0u) return sign;
        uint m = mant32;
        int e = -126;
        while ((m & 0x00800000u) == 0u) {
            m <<= 1u;
            e -= 1;
        }
        const ulong exp64 = (ulong)(e + 1023);
        const ulong mant64 = (ulong)(m & 0x007fffffu) << 29ul;
        return sign | (exp64 << 52ul) | mant64;
    }
    if (exp32 == 0xffu) {
        return sign;
    }
    const ulong exp64 = (ulong)((int)exp32 - 127 + 1023);
    const ulong mant64 = (ulong)mant32 << 29ul;
    return sign | (exp64 << 52ul) | mant64;
}

inline ulong f64_add_bits(ulong a, ulong b) {
    ulong mag_a = a & 0x7ffffffffffffffful;
    ulong mag_b = b & 0x7ffffffffffffffful;
    if (mag_a == 0ul && mag_b == 0ul) {
        return 0ul;
    }
    if (mag_a == 0ul) return b;
    if (mag_b == 0ul) return a;

    ulong hi = a;
    ulong lo = b;
    if (mag_b > mag_a) {
        hi = b;
        lo = a;
        const ulong tmp = mag_a;
        mag_a = mag_b;
        mag_b = tmp;
    }

    const ulong sa = hi >> 63ul;
    const ulong sb = lo >> 63ul;
    int ea = (int)((mag_a >> 52ul) & 0x7fful);
    int eb = (int)((mag_b >> 52ul) & 0x7fful);
    const ulong ma = mag_a & 0x000ffffffffffffful;
    const ulong mb = mag_b & 0x000ffffffffffffful;

    ulong sig_a = (ea == 0) ? (ma << 3ul) : ((0x0010000000000000ul | ma) << 3ul);
    ulong sig_b = (eb == 0) ? (mb << 3ul) : ((0x0010000000000000ul | mb) << 3ul);
    if (ea == 0) ea = 1;
    if (eb == 0) eb = 1;

    const int diff = ea - eb;
    if (diff >= 60) {
        sig_b = (sig_b != 0ul) ? 1ul : 0ul;
    } else if (diff > 0) {
        const ulong mask = (1ul << (uint)diff) - 1ul;
        const ulong sticky = ((sig_b & mask) != 0ul) ? 1ul : 0ul;
        sig_b = (sig_b >> (uint)diff) | sticky;
    }

    int exp_out = ea;
    ulong sum = 0ul;
    if (sa == sb) {
        sum = sig_a + sig_b;
        if ((sum & (1ul << 56ul)) != 0ul) {
            const ulong sticky = sum & 1ul;
            sum = (sum >> 1ul) | sticky;
            exp_out += 1;
        }
    } else {
        sum = sig_a - sig_b;
        if (sum == 0ul) return 0ul;
        while ((sum & (1ul << 55ul)) == 0ul && exp_out > 1) {
            sum <<= 1ul;
            exp_out -= 1;
        }
    }

    const ulong grs = sum & 7ul;
    ulong mant53 = sum >> 3ul;
    if (grs > 4ul || (grs == 4ul && (mant53 & 1ul) != 0ul)) {
        mant53 += 1ul;
        if ((mant53 & (1ul << 53ul)) != 0ul) {
            mant53 >>= 1ul;
            exp_out += 1;
        }
    }

    if ((mant53 & (1ul << 52ul)) == 0ul) {
        exp_out = 0;
    }
    if (exp_out >= 0x7ff) {
        return (sa << 63ul) | 0x7feffffffffffffful;
    }
    return (sa << 63ul) | ((ulong)exp_out << 52ul) | (mant53 & 0x000ffffffffffffful);
}

kernel void emeans_accum_chunk_obj(
    device const float *chunk_dists [[buffer(0)]],
    device ulong *epoch_obj_bits [[buffer(1)]],
    constant uint &M [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup ulong scratch[256];

    ulong acc = 0ul;
    for (uint r = tid; r < M; r += 256u) {
        acc = f64_add_bits(acc, f32_to_f64_bits(chunk_dists[r]));
    }
    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = f64_add_bits(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        epoch_obj_bits[0] = f64_add_bits(epoch_obj_bits[0], scratch[0]);
    }
}

// =============================================================================
// 21. EMA Centroid Update (`weight == 1.0f` exact override + reset batch sums)
// =============================================================================

kernel void emeans_apply_ema(
    device float *centroids [[buffer(0)]],
    device float *usage [[buffer(1)]],
    device float *batch_sums [[buffer(2)]],
    device ulong *batch_counts [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    constant float &alpha [[buffer(6)]],
    uint j [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (j >= k) return;

    const ulong n_j = batch_counts[j];
    const float n_old = usage[j];
    const bool replace = (alpha == 1.0f);
    const ulong base = (ulong)j * (ulong)d;

    if (n_j > 0ul) {
        const float nf = (float)n_j;
        const float n_new = replace ? (alpha * nf) : ((1.0f - alpha) * n_old + alpha * nf);
        const float w = precise::divide(alpha * nf, n_new);
        const bool exact_mean = (w == 1.0f);
        for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
            const float s_j = batch_sums[base + col];
            batch_sums[base + col] = 0.0f;
            const float mean = precise::divide(s_j, nf);
            const float c_old = centroids[base + col];
            centroids[base + col] = exact_mean ? mean : (c_old + w * (mean - c_old));
        }
        if (tid == 0u) {
            usage[j] = n_new;
            batch_counts[j] = 0ul;
        }
    } else {
        for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
            batch_sums[base + col] = 0.0f;
        }
        if (tid == 0u) {
            usage[j] = replace ? 0.0f : ((1.0f - alpha) * n_old);
            batch_counts[j] = 0ul;
        }
    }
}

// =============================================================================
// 22 & 23. SplitMix64 Gumbel Revival Preparation and Split Separation
// =============================================================================

inline ulong splitmix64_mix(ulong z) {
    z += 0x9e3779b97f4a7c15ul;
    z = (z ^ (z >> 30ul)) * 0xbf58476d1ce4e5b9ul;
    z = (z ^ (z >> 27ul)) * 0x94d049bb133111ebul;
    return z ^ (z >> 31ul);
}

inline float emeans_gumbel_key(float usage_j, ulong seed, ulong step, uint cluster_id) {
    const ulong s0 = splitmix64_mix(seed ^ 0xd1b54a32d192ed03ul);
    const ulong s1 = splitmix64_mix(s0 ^ (step * 0x9e3779b97f4a7c15ul));
    const ulong h = splitmix64_mix(s1 ^ ((ulong)cluster_id * 0xbf58476d1ce4e5b9ul));
    const uint odd24 = (((uint)(h >> 41ul)) << 1u) | 1u;
    const float u = (float)odd24 * as_type<float>(0x33800000u); // * 2^-24 in (0, 1)
    return precise::log(usage_j) - precise::log(-precise::log(u));
}

inline uint float_to_ordered_u32(float f) {
    const float clean = (f == 0.0f) ? 0.0f : f;
    const uint u = as_type<uint>(clean);
    return ((u & 0x80000000u) != 0u) ? (~u) : (u ^ 0x80000000u);
}

kernel void emeans_revive_prepare(
    device const float *usage [[buffer(0)]],
    device ulong *donor_keys [[buffer(1)]],
    device uint *unused_ids [[buffer(2)]],
    device uint2 *revive_meta [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant float &unused_threshold [[buffer(5)]],
    constant ulong &seed [[buffer(6)]],
    constant ulong &step [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float scratch[256];
    threadgroup uint tg_counts[2];

    float sum_part = 0.0f;
    for (uint j = tid; j < k; j += 256u) {
        sum_part += usage[j];
    }
    const float total_n = tg_reduce_sum_256(sum_part, tid, scratch);
    const float mean_n = (k > 0u) ? precise::divide(total_n, (float)k) : 0.0f;
    const float thresh = unused_threshold * mean_n;

    if (tid == 0u) {
        uint n_unused = 0u;
        uint n_donors = 0u;
        for (uint j = 0u; j < k; ++j) {
            const float nj = usage[j];
            if (nj < thresh) {
                unused_ids[n_unused] = j;
                n_unused += 1u;
            } else if (nj > 0.0f) {
                n_donors += 1u;
            }
        }
        const uint nsplit = min(n_unused, n_donors);
        tg_counts[0] = nsplit;
        tg_counts[1] = n_unused;
        revive_meta[0] = uint2(nsplit, revive_meta[0].y + nsplit);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < k; j += 256u) {
        const float nj = usage[j];
        const bool is_donor = !(nj < thresh) && (nj > 0.0f);
        if (is_donor) {
            const float g = emeans_gumbel_key(nj, seed, step, j);
            const uint inv_ord = ~float_to_ordered_u32(g);
            donor_keys[j] = ((ulong)inv_ord << 32ul) | (ulong)j;
        } else {
            donor_keys[j] = (0xfffffffful << 32ul) | (ulong)j;
        }
    }
}

kernel void emeans_revive_splits(
    device const ulong *sorted_donor_keys [[buffer(0)]],
    device const uint *unused_ids [[buffer(1)]],
    device const uint2 *revive_meta [[buffer(2)]],
    device float *centroids [[buffer(3)]],
    device float *usage [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    constant float &split_epsilon [[buffer(7)]],
    constant uint &use_bf16_repr [[buffer(8)]],
    uint pair_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint nsplit = revive_meta[0].x;
    if (pair_idx >= nsplit) return;

    const ulong dkey = sorted_donor_keys[pair_idx];
    if ((uint)(dkey >> 32ul) == 0xffffffffu) return;
    const uint donor = (uint)(dkey & 0xfffffffful);
    const uint target = unused_ids[pair_idx];
    if (donor >= k || target >= k || donor == target) return;

    threadgroup atomic_uint any_diff;
    if (tid == 0u) {
        atomic_store_explicit(&any_diff, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong donor_base = (ulong)donor * (ulong)d;
    const ulong target_base = (ulong)target * (ulong)d;

    uint local_diff = 0u;
    for (ulong col = (ulong)tid; col < (ulong)d; col += 256ul) {
        const float c = centroids[donor_base + col];
        const float off = ((col & 1ul) == 0ul) ? split_epsilon : (-split_epsilon);
        const float donor_c = c * (1.0f - off);
        const float child_c = c * (1.0f + off);
        centroids[donor_base + col] = donor_c;
        centroids[target_base + col] = child_c;

        if (use_bf16_repr != 0u) {
            const float dq = bf16_bits_to_f32_clean(f32_bits_to_bf16_rte(as_type<uint>(donor_c)));
            const float cq = bf16_bits_to_f32_clean(f32_bits_to_bf16_rte(as_type<uint>(child_c)));
            if (dq != cq) {
                local_diff = 1u;
            }
        } else {
            const uint du = as_type<uint>(donor_c);
            const uint cu = as_type<uint>(child_c);
            const float df = ((du & 0x7fffffffu) < 0x00800000u) ? 0.0f : donor_c;
            const float cf = ((cu & 0x7fffffffu) < 0x00800000u) ? 0.0f : child_c;
            if (df != cf) {
                local_diff = 1u;
            }
        }
    }
    if (local_diff != 0u) {
        atomic_fetch_or_explicit(&any_diff, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        if (atomic_load_explicit(&any_diff, memory_order_relaxed) == 0u && d > 0u) {
            float child0 = centroids[target_base];
            if (use_bf16_repr != 0u) {
                const ushort b = f32_bits_to_bf16_rte(as_type<uint>(child0));
                if ((b & 0x7fffu) < 0x0080u) {
                    child0 = as_type<float>(0x20800000u);
                } else if ((b & 0x8000u) == 0u) {
                    const ushort b_step = (b >= 0x7f7fu) ? (ushort)(b - 1u) : (ushort)(b + 1u);
                    child0 = as_type<float>((uint)b_step << 16u);
                } else {
                    const ushort b_step = (b == 0x8080u) ? (ushort)0x0000u : (ushort)(b - 1u);
                    child0 = as_type<float>((uint)b_step << 16u);
                }
            } else {
                const uint u = as_type<uint>(child0);
                if ((u & 0x7fffffffu) < 0x00800000u) {
                    child0 = as_type<float>(0x20800000u);
                } else if ((u & 0x80000000u) == 0u) {
                    const uint u_step = (u >= 0x7f7fffffu) ? (u - 1u) : (u + 1u);
                    child0 = as_type<float>(u_step);
                } else {
                    const uint u_step = (u == 0x80800000u) ? 0x00000000u : (u - 1u);
                    child0 = as_type<float>(u_step);
                }
            }
            centroids[target_base] = child0;
        }
        const float half_n = usage[donor] * 0.5f;
        usage[donor] = half_n;
        usage[target] = half_n;
    }
}
