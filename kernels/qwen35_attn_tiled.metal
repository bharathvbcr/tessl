// Qwen3.5 full-attention prefill on the TensorOps matrix units.
//
// `flash_attn_rows` is scalar f32: one simdgroup per query row, one fused
// multiply-add per lane per dim. At T=8192 on an M5 Pro it measured 169 ms per
// layer on Qwen3.5-2B's shapes (8 query / 2 KV heads of 256), about
// 1.6 TFLOP/s and half the whole forward, where the same machine's exact-f32
// TensorOps GEMM runs at 6.4 TFLOP/s (docs/benchmarking.md). The block
// geometry (BQ queries, BK keys, NSG simdgroups) is a template parameter;
// the instantiations are at the end of the file.
//
// This is FlashAttention-2 with both products on `mpp::tensor_ops::matmul2d`:
//
//   S  = Q_blk · K_blkᵀ          BQ x BK, a cooperative tensor (registers)
//   S -> threadgroup; per-row online softmax in scalar code; P overwrites S
//   O  = O · diag(alpha) + P · V_blk      O is BQ x 256, a cooperative tensor
//
// Q, K and V are read straight from device memory through strided tensor
// views: a head's rows are `H*D` (or `Hkv*D`) floats apart, which is just a
// row stride to MPP, so nothing is staged by hand. Only S/P and the per-row
// softmax state live in threadgroup memory.
//
// Same contract as `flash_attn_rows` at `window == 0` and the same buffer
// slots: q_abs = q_pos_offset + t_q, k_abs = kv_pos_offset + t_k, keep
// k_abs <= q_abs; Tkv = min(*Tkv_ptr, kv_capacity); a row with nothing
// unmasked is zeros. The matrix units accumulate in a different order than
// the scalar kernel, so the two agree to rounding, not bit for bit.
//
// Key blocks strictly above a threadgroup's last query are never visited; the
// mask only bites on the blocks that straddle the diagonal and on the partial
// block at Tkv. Out-of-range rows of a K or V block are outside the tensor's
// extents, which MPP bounds-checks, and their scores are masked to -inf so
// their P is exactly 0 either way.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint TILED_ATTN_D = 256;

template <int D, int BQ, int BK, int NSG>
inline void attn_tiled_body(
    device float *Q, device float *K, device float *V, device float *O,
    uint Tq, uint Tkv, uint H, uint Hkv, float scale, ulong q_off,
    ulong kv_off, uint out_bf16, uint kv_capacity, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *m_row, threadgroup float *l_row,
    threadgroup float *a_row)
{
    constexpr int THREADS = NSG * 32;
    constexpr int TPR = THREADS / BQ;      // threads per query row, softmax pass
    constexpr int CPT = BK / TPR;          // columns per thread
    static_assert(THREADS % BQ == 0 && BK % TPR == 0, "softmax mapping");
    static_assert(TPR <= 32 && (TPR & (TPR - 1)) == 0, "row threads in one simdgroup");

    const uint q0 = tgpig.x * (uint)BQ;
    if (q0 >= Tq) { return; }              // uniform across the threadgroup
    const uint nq = min((uint)BQ, Tq - q0);
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;

    const ulong q_row = (ulong)H * (ulong)D;
    const ulong kv_row = (ulong)Hkv * (ulong)D;
    const ulong q_base = (ulong)b * Tq * q_row + (ulong)h * D;
    const ulong kv_base = (ulong)b * kv_capacity * kv_row + (ulong)hkv * D;

    // Keys this threadgroup can see at all: through its last query's position.
    const ulong q_hi = q_off + q0 + nq - 1u;
    const uint t_end = (q_hi >= kv_off)
        ? (uint)min((ulong)Tkv, q_hi - kv_off + 1ul) : 0u;

    constexpr auto qk_desc = matmul2d_descriptor(
        BQ, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        BQ, D, BK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    auto mQ = tensor(Q + q_base, dextents<int, 2>{D, (int)Tq},
                     array<int, 2>{1, (int)q_row});
    auto tQ = mQ.slice(0, (int)q0);
    auto mK = tensor(K + kv_base, dextents<int, 2>{D, (int)Tkv},
                     array<int, 2>{1, (int)kv_row});
    auto mV = tensor(V + kv_base, dextents<int, 2>{D, (int)Tkv},
                     array<int, 2>{1, (int)kv_row});
    auto tP = tensor(S, dextents<int, 2>{BK, BQ}, array<int, 2>{1, BK});

    auto oT = pv_op.template get_destination_cooperative_tensor<
        decltype(tP), decltype(mV.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { oT[i] = 0.0f; }

    if (tid < (uint)BQ) {
        m_row[tid] = -INFINITY;
        l_row[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = tid / (uint)TPR;        // this thread's row in the softmax pass
    const uint part = tid % (uint)TPR;
    const ulong q_abs = q_off + q0 + r;
    const bool row_live = r < nq;

    for (uint kb = 0; kb < t_end; kb += (uint)BK) {
        auto tK = mK.slice(0, (int)kb);
        auto sT = qk_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        qk_op.run(tQ, tK, sT);
        sT.store(tP);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax for row r over this block's columns. The TPR threads
        // of a row are adjacent lanes of one simdgroup, so the row max and sum
        // are xor-butterflies over them.
        float s_loc[CPT];
        float mx = -INFINITY;
        for (int j = 0; j < CPT; ++j) {
            const uint c = part * (uint)CPT + (uint)j;
            const uint t = kb + c;
            float s = S[r * (uint)BK + c] * scale;
            if (!row_live || t >= t_end || kv_off + (ulong)t > q_abs) {
                s = -INFINITY;
            }
            s_loc[j] = s;
            mx = max(mx, s);
        }
        for (uint off = (uint)TPR / 2u; off > 0u; off >>= 1) {
            mx = max(mx, simd_shuffle_xor(mx, (ushort)off));
        }
        const float m_old = m_row[r];
        const float m_new = max(m_old, mx);
        // exp(-inf - -inf) is NaN; a row that has seen nothing has a zero
        // accumulator, so its rescale is exactly zero.
        const float alpha = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
        float sum = 0.0f;
        for (int j = 0; j < CPT; ++j) {
            const float p = (s_loc[j] == -INFINITY) ? 0.0f : exp(s_loc[j] - m_new);
            S[r * (uint)BK + part * (uint)CPT + (uint)j] = p;
            sum += p;
        }
        for (uint off = (uint)TPR / 2u; off > 0u; off >>= 1) {
            sum += simd_shuffle_xor(sum, (ushort)off);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0u) {
            l_row[r] = l_row[r] * alpha + sum;
            m_row[r] = m_new;
            a_row[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) {
                const auto idx = oT.get_multidimensional_index(i);
                oT[i] *= a_row[idx[1]];
            }
        }
        auto tV = mV.slice(0, (int)kb);
        pv_op.run(tP, tV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) {
            const auto idx = oT.get_multidimensional_index(i);
            const float l = l_row[idx[1]];
            oT[i] *= (l > 0.0f) ? (1.0f / l) : 0.0f;
        }
    }
    if (out_bf16 != 0u) {
        auto mO = tensor((device bfloat *)O + q_base, dextents<int, 2>{D, (int)Tq},
                         array<int, 2>{1, (int)q_row});
        auto tO = mO.slice(0, (int)q0);
        auto oB = pv_op.template get_destination_cooperative_tensor<
            decltype(tP), decltype(mV.slice(0, 0)), bfloat>();
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oB.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { oB[i] = bfloat(oT[i]); }
        }
        oB.store(tO);
    } else {
        auto mO = tensor(O + q_base, dextents<int, 2>{D, (int)Tq},
                         array<int, 2>{1, (int)q_row});
        oT.store(mO.slice(0, (int)q0));
    }
}

/// One entry point per block geometry: BQ queries by BK keys, NSG simdgroups.
/// The name spells the geometry (`_q{BQ}_k{BK}_sg{NSG}`), and the host builds
/// its grid from the same numbers, so the name is the contract between them.
/// Buffer slots are `flash_attn_rows`'; `window` (9) must be 0 and `B` (4)
/// is unused, as there.
#define TILED_ATTN_KERNEL(NAME, BQ, BK, NSG)                                   \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *O [[buffer(3)]],                                            \
    constant uint &B [[buffer(4)]],                                           \
    constant uint &Tq [[buffer(5)]],                                          \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    constant uint &out_bf16 [[buffer(13)]],                                   \
    constant uint &kv_capacity [[buffer(14)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[(BQ) * (BK)];                                         \
    threadgroup float m_row[BQ];                                              \
    threadgroup float l_row[BQ];                                              \
    threadgroup float a_row[BQ];                                              \
    /* Clamp mutable device state before it participates in any address. */   \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    /* MPP's type matching rejects const element types; the kernel never      \
       writes Q, K or V. */                                                   \
    attn_tiled_body<TILED_ATTN_D, BQ, BK, NSG>(                               \
        const_cast<device float *>(Q), const_cast<device float *>(K),         \
        const_cast<device float *>(V), O, Tq, Tkv, H, Hkv, scale,             \
        (ulong)(*q_pos_offset_ptr), (ulong)(*kv_pos_offset_ptr), out_bf16,    \
        kv_capacity, tgpig, tid, S, m_row, l_row, a_row);                     \
    (void)B;                                                                  \
    (void)window;                                                             \
}

// Every instantiation keeps the O accumulator at BQ * 256 / (NSG * 32) = 64
// floats per thread; they differ in K/V reuse (BQ) and in how often the
// softmax pass and the O rescale run per key (BK).
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q32_k32_sg4, 32, 32, 4)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q32_k64_sg4, 32, 64, 4)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q64_k32_sg8, 64, 32, 8)
TILED_ATTN_KERNEL(qwen35_attn_tiled_h256_q64_k64_sg8, 64, 64, 8)
