// MLX affine Q4 GEMV (M=1): w = scale * q_u + bias.
// Hot scale+bias banks are interleaved bfloat2 (MLX packing).
// Layouts:
//   gemv_q4_mlx / _wide — row-major + dynamic x_cache + uint peel
//   gemv_q4_mlx_blocked — Hot-repacked [row_block][group][Bn][group_bytes]
//   gemv_q4_mlx_blocked_gate_up_gelu — fused gate∥up → gelu_pytorch_tanh mid
//   gemv_q4_mlx_simd* — true qmv_fast peel + bfloat2 sb; row-major or Interleaved4
//                       (one templated body, gemv_q4_mlx_simd.h)
#include <metal_stdlib>
#include "act.h"
#include "gemv_q4_mlx_simd.h"
using namespace metal;

constant uint GEMV_TG = 128u;
constant uint GEMV_BN = 16u;      // blocked row tile (wider coalesced loads)
constant uint GEMV_LANES = 16u;   // K-lanes per row
constant uint GEMV_BLOCKED_TG = GEMV_BN * GEMV_LANES; // 256
// Cap dynamic x-cache so x + static partials fit in ≈32 KiB TG mem.
constant uint GEMV_X_TILE = 4096u;

inline float dequant_q4_u_bias(
    device const uchar *packed,
    ulong idx,
    float scale,
    float bias)
{
    uchar byte = packed[idx / 2];
    uchar nibble = (idx & 1ul) == 0ul ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
    return scale * (float)nibble + bias;
}

inline float peel_uint_dot8(
    uint w,
    float scale,
    float bias,
    threadgroup float *x_cache,
    ulong xbase)
{
    // xbase is group-aligned (typically 32) → float4-safe.
    float4 x0 = *((threadgroup float4 *)(x_cache + xbase));
    float4 x1 = *((threadgroup float4 *)(x_cache + xbase + 4u));
    float4 q0 = float4(
        float(w & 0x0fu),
        float((w >> 4) & 0x0fu),
        float((w >> 8) & 0x0fu),
        float((w >> 12) & 0x0fu));
    float4 q1 = float4(
        float((w >> 16) & 0x0fu),
        float((w >> 20) & 0x0fu),
        float((w >> 24) & 0x0fu),
        float((w >> 28) & 0x0fu));
    float4 d0 = (scale * q0 + bias) * x0;
    float4 d1 = (scale * q1 + bias) * x1;
    return ((d0.x + d0.y) + (d0.z + d0.w)) + ((d1.x + d1.y) + (d1.z + d1.w));
}

inline float peel_group_dot(
    device const uchar *packed_at_group,
    float scale,
    float bias,
    threadgroup float *x_cache,
    ulong xbase,
    uint group_size)
{
    device const uint *pwords = (device const uint *)packed_at_group;
    float acc = 0.0f;
    ulong i = 0ul;
    for (; i + 32ul <= (ulong)group_size; i += 32ul) {
        const uint4 ww = ((device const uint4 *)(pwords + (i / 8ul)))[0];
        acc += peel_uint_dot8(ww.x, scale, bias, x_cache, xbase + i);
        acc += peel_uint_dot8(ww.y, scale, bias, x_cache, xbase + i + 8u);
        acc += peel_uint_dot8(ww.z, scale, bias, x_cache, xbase + i + 16u);
        acc += peel_uint_dot8(ww.w, scale, bias, x_cache, xbase + i + 24u);
    }
    for (; i + 8ul <= (ulong)group_size; i += 8ul) {
        acc += peel_uint_dot8(pwords[i / 8ul], scale, bias, x_cache, xbase + i);
    }
    for (; i < (ulong)group_size; ++i) {
        uchar byte = packed_at_group[i / 2ul];
        uchar nibble = (i & 1ul) == 0ul ? (byte & 0x0fu) : ((byte >> 4) & 0x0fu);
        acc += (scale * float(nibble) + bias) * x_cache[xbase + i];
    }
    return acc;
}

inline float gemv_q4_mlx_body_acc(
    device const uchar *packed,
    device const bfloat2 *sb,
    device const bfloat *biases_unused,
    threadgroup float *x_cache,
    uint cols,
    uint group_size,
    uint row)
{
    const uint groups_per_row = cols / group_size;
    const ulong row_base = (ulong)row * cols;
    const ulong scale_base = (ulong)row * groups_per_row;
    const ulong packed_row = row_base / 2ul;

    float acc = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        (void)biases_unused;
        const bfloat2 sbv = sb[scale_base + g];
        const float scale = float(sbv.x);
        const float bias = float(sbv.y);
        const ulong xbase = (ulong)g * group_size;
        const ulong pbase = packed_row + ((ulong)g * group_size) / 2ul;
        acc += peel_group_dot(packed + pbase, scale, bias, x_cache, xbase, group_size);
    }
    return acc;
}

inline void gemv_q4_mlx_body(
    device const uchar *packed,
    device const bfloat2 *sb,
    device const bfloat *biases_unused,
    device float *y,
    threadgroup float *x_cache,
    uint cols,
    uint group_size,
    uint row)
{
    y[row] = gemv_q4_mlx_body_acc(packed, sb, biases_unused, x_cache, cols, group_size, row);
}

kernel void gemv_q4_mlx(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float *x_cache [[threadgroup(0)]])
{
    for (ulong i = lid; i < (ulong)cols; i += tptg) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= rows) return;
    gemv_q4_mlx_body(packed, sb, biases_unused, y, x_cache, cols, group_size, gid);
}

kernel void gemv_q4_mlx_wide(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float *x_cache [[threadgroup(0)]])
{
    for (ulong i = lid; i < (ulong)cols; i += tptg) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid >= rows) return;
    gemv_q4_mlx_body(packed, sb, biases_unused, y, x_cache, cols, group_size, gid);
}

inline float peel_uint_dot8_reg(
    uint w,
    float scale,
    float bias,
    float4 x0,
    float4 x1)
{
    float4 q0 = float4(
        float(w & 0x0fu),
        float((w >> 4) & 0x0fu),
        float((w >> 8) & 0x0fu),
        float((w >> 12) & 0x0fu));
    float4 q1 = float4(
        float((w >> 16) & 0x0fu),
        float((w >> 20) & 0x0fu),
        float((w >> 24) & 0x0fu),
        float((w >> 28) & 0x0fu));
    float4 d0 = (scale * q0 + bias) * x0;
    float4 d1 = (scale * q1 + bias) * x1;
    return ((d0.x + d0.y) + (d0.z + d0.w)) + ((d1.x + d1.y) + (d1.z + d1.w));
}

/// Cooperative blocked GEMV: TG = BN × LANES K-lanes.
/// Thread map `row_local = lid % BN`, `lane = lid / BN` so lids 0..BN-1 at a fixed
/// K-group hit consecutive `[Bn][group_bytes]` chunks (coalesced). TG reduce.
/// For cols ≤ GEMV_X_TILE: cache full `x` in TG. For wider (MLP down): stream
/// `x` from device — avoids multi-tile barriers; weight BW dwarfs x re-reads.
kernel void gemv_q4_mlx_blocked(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float *x_cache [[threadgroup(0)]])
{
    threadgroup float partial[GEMV_BLOCKED_TG];

    const uint row_local = lid % GEMV_BN;
    const uint lane = lid / GEMV_BN;
    const uint n_lanes = tptg / GEMV_BN;
    const uint row = tg * GEMV_BN + row_local;
    const uint groups_per_row = cols / group_size;
    const uint bytes_per_group = group_size / 2u;
    const ulong block_bytes = (ulong)groups_per_row * GEMV_BN * bytes_per_group;
    const ulong block_scales = (ulong)groups_per_row * GEMV_BN;
    const ulong block_base = (ulong)tg * block_bytes;
    const ulong scale_block = (ulong)tg * block_scales;
    const bool active = (lane < n_lanes) && (row < rows);
    const bool use_tg_x = (cols <= GEMV_X_TILE);

    if (use_tg_x) {
        for (ulong i = lid; i < (ulong)cols; i += tptg) {
            x_cache[i] = x[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float acc = 0.0f;
    if (active) {
        for (ulong g = lane; g < (ulong)groups_per_row; g += n_lanes) {
            const ulong sg = scale_block + g * GEMV_BN + row_local;
            const bfloat2 sbv = sb[sg];
            const float scale = float(sbv.x);
            const float bias = float(sbv.y);
            const ulong xbase = g * group_size;
            const ulong pbase =
                block_base + g * GEMV_BN * bytes_per_group + row_local * bytes_per_group;
            if (use_tg_x) {
                acc += peel_group_dot(packed + pbase, scale, bias, x_cache, xbase, group_size);
            } else {
                // Device-x peel (wide MLP down).
                device const uint *pwords = (device const uint *)(packed + pbase);
                uint i = 0u;
                for (; i + 32u <= group_size; i += 32u) {
                    const uint4 ww = ((device const uint4 *)(pwords + (i / 8u)))[0];
                    float4 x0 = ((device const float4 *)(x + xbase + i))[0];
                    float4 x1 = ((device const float4 *)(x + xbase + i + 4u))[0];
                    float4 x2 = ((device const float4 *)(x + xbase + i + 8u))[0];
                    float4 x3 = ((device const float4 *)(x + xbase + i + 12u))[0];
                    float4 x4 = ((device const float4 *)(x + xbase + i + 16u))[0];
                    float4 x5 = ((device const float4 *)(x + xbase + i + 20u))[0];
                    float4 x6 = ((device const float4 *)(x + xbase + i + 24u))[0];
                    float4 x7 = ((device const float4 *)(x + xbase + i + 28u))[0];
                    // Manual peel of 8 uint nibbles × float4 pairs.
                    acc += peel_uint_dot8_reg(ww.x, scale, bias, x0, x1);
                    acc += peel_uint_dot8_reg(ww.y, scale, bias, x2, x3);
                    acc += peel_uint_dot8_reg(ww.z, scale, bias, x4, x5);
                    acc += peel_uint_dot8_reg(ww.w, scale, bias, x6, x7);
                }
                for (; i + 8u <= group_size; i += 8u) {
                    float4 x0 = ((device const float4 *)(x + xbase + i))[0];
                    float4 x1 = ((device const float4 *)(x + xbase + i + 4u))[0];
                    acc += peel_uint_dot8_reg(pwords[i / 8u], scale, bias, x0, x1);
                }
            }
        }
    }

    if (lid < GEMV_BLOCKED_TG) {
        partial[lid] = active ? acc : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = n_lanes / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride && lid < GEMV_BLOCKED_TG) {
            partial[lid] += partial[lid + stride * GEMV_BN];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u && row < rows) {
        y[row] = partial[lid];
    }
}

/// Fused gated-MLP mid: mid[row] = gelu(W_gate[row]@x) * (W_up[row]@x).
/// Expects cols ≤ GEMV_X_TILE (E4B/31B gate·up).
kernel void gemv_q4_mlx_blocked_gate_up_gelu(
    device const uchar *gate_packed [[buffer(0)]],
    device const bfloat2 *gate_sb [[buffer(1)]],
    device const bfloat *gate_biases_unused [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const bfloat2 *up_sb [[buffer(4)]],
    device const bfloat *up_biases_unused [[buffer(5)]],
    device const float *x [[buffer(6)]],
    device float *mid [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    constant uint &group_size [[buffer(10)]],
    constant uint &mid_as_bf16 [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float *x_cache [[threadgroup(0)]])
{
    threadgroup float partial_g[GEMV_BLOCKED_TG];
    threadgroup float partial_u[GEMV_BLOCKED_TG];

    const uint row_local = lid % GEMV_BN;
    const uint lane = lid / GEMV_BN;
    const uint n_lanes = tptg / GEMV_BN;
    const uint row = tg * GEMV_BN + row_local;
    const uint groups_per_row = cols / group_size;
    const uint bytes_per_group = group_size / 2u;
    const ulong block_bytes = (ulong)groups_per_row * GEMV_BN * bytes_per_group;
    const ulong block_scales = (ulong)groups_per_row * GEMV_BN;
    const ulong block_base = (ulong)tg * block_bytes;
    const ulong scale_block = (ulong)tg * block_scales;
    const bool active = (lane < n_lanes) && (row < rows);

    for (ulong i = lid; i < (ulong)cols; i += tptg) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (active) {
        for (ulong g = lane; g < (ulong)groups_per_row; g += n_lanes) {
            const ulong sg = scale_block + g * GEMV_BN + row_local;
            (void)gate_biases_unused; (void)up_biases_unused;
            const bfloat2 gsbv = gate_sb[sg];
            const bfloat2 usbv = up_sb[sg];
            const float gs = float(gsbv.x);
            const float gb = float(gsbv.y);
            const float us = float(usbv.x);
            const float ub = float(usbv.y);
            const ulong xbase = g * group_size;
            const ulong pbase =
                block_base + g * GEMV_BN * bytes_per_group + row_local * bytes_per_group;
            acc_g += peel_group_dot(gate_packed + pbase, gs, gb, x_cache, xbase, group_size);
            acc_u += peel_group_dot(up_packed + pbase, us, ub, x_cache, xbase, group_size);
        }
    }

    if (lid < GEMV_BLOCKED_TG) {
        partial_g[lid] = active ? acc_g : 0.0f;
        partial_u[lid] = active ? acc_u : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = n_lanes / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride && lid < GEMV_BLOCKED_TG) {
            partial_g[lid] += partial_g[lid + stride * GEMV_BN];
            partial_u[lid] += partial_u[lid + stride * GEMV_BN];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u && row < rows) {
        float v = tessl_gelu_pytorch_tanh(partial_g[lid]) * partial_u[lid];
        if (mid_as_bf16 != 0u) {
            ((device bfloat *)mid)[row] = bfloat(v);
        } else {
            mid[row] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Simdgroup-cooperative MLX affine Q4 GEMV (M=1): the entry points below own
// only their names and `[[buffer(N)]]` slots. The body is `gemv_q4_mlx_simd.h`
// (`q4_simd_gemv<layout, epilogue>`); the simdgroup geometry (4 rows/SG, 2 SG/TG,
// 16 nibbles/lane, K-block 512) is in `q4_mlx_dot.h`.
// Layouts: row-major + Interleaved4 ([tile][pack2][r0..r3], `_i4`).
// ---------------------------------------------------------------------------

kernel void gemv_q4_mlx_simd(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    q4_simd_gemv<Q4Layout::RowMajor, Q4Epi::Store>(
        Q4Bank{packed, sb}, q4_no_bank(), x, y, nullptr, 0u, rows, cols, group_size,
        q4_simd_row0(tgid, sgid), lane);
}

kernel void gemv_q4_mlx_simd_add(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    device const float *resid [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    q4_simd_gemv<Q4Layout::RowMajor, Q4Epi::AddResid>(
        Q4Bank{packed, sb}, q4_no_bank(), x, y, resid, 0u, rows, cols, group_size,
        q4_simd_row0(tgid, sgid), lane);
}

kernel void gemv_q4_mlx_simd_gate_up_gelu(
    device const uchar *gate_packed [[buffer(0)]],
    device const bfloat2 *gate_sb [[buffer(1)]],
    device const bfloat *gate_biases_unused [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const bfloat2 *up_sb [[buffer(4)]],
    device const bfloat *up_biases_unused [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *mid [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    constant uint &group_size [[buffer(10)]],
    constant uint &mid_as_bf16 [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)gate_biases_unused; (void)up_biases_unused;
    q4_simd_gemv<Q4Layout::RowMajor, Q4Epi::GateUpGelu>(
        Q4Bank{gate_packed, gate_sb}, Q4Bank{up_packed, up_sb}, x, mid, nullptr,
        mid_as_bf16, rows, cols, group_size, q4_simd_row0(tgid, sgid), lane);
}

// Fused K∥V (default-on product path). Bank-partitioned like QKV so each TG
// runs a *solo* gemv reduce (math ≡ `gemv_q4_mlx_simd`). Prior dual-accumulate
// (K+V in one thread loop) drifted ~1 ULP vs solo at Hot E4B k[41].
// Partition: [0, tg_k) → K; [tg_k, 2*tg_k) → V. `tg_k` = ceil(rows / rows_per_tg).
kernel void gemv_q4_mlx_simd_kv(
    device const uchar *k_packed [[buffer(0)]],
    device const bfloat2 *k_sb [[buffer(1)]],
    device const bfloat *k_biases_unused [[buffer(2)]],
    device const uchar *v_packed [[buffer(3)]],
    device const bfloat2 *v_sb [[buffer(4)]],
    device const bfloat *v_biases_unused [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *k_out [[buffer(7)]],
    device float *v_out [[buffer(8)]],
    constant uint &rows [[buffer(9)]],
    constant uint &cols [[buffer(10)]],
    constant uint &group_size [[buffer(11)]],
    constant uint &tg_k [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)k_biases_unused; (void)v_biases_unused;
    const Q4Route at = q4_route_kv(
        tgid, tg_k, Q4Bank{k_packed, k_sb}, k_out, Q4Bank{v_packed, v_sb}, v_out, rows);
    q4_simd_gemv<Q4Layout::RowMajor, Q4Epi::Store>(
        at.bank, q4_no_bank(), x, at.y, nullptr, 0u, at.rows, cols, group_size,
        q4_simd_row0(at.local_tg, sgid), lane);
}

// ---------------------------------------------------------------------------
// Interleaved4 Hot: [tile][uint2_pack][r0..r3] packs + [tile][g][r0..r3] bfloat2 sb
// ---------------------------------------------------------------------------

kernel void gemv_q4_mlx_simd_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    q4_simd_gemv<Q4Layout::Interleaved4, Q4Epi::Store>(
        Q4Bank{packed, sb}, q4_no_bank(), x, y, nullptr, 0u, rows, cols, group_size,
        q4_simd_row0(tgid, sgid), lane);
}

kernel void gemv_q4_mlx_simd_add_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    device const float *resid [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    q4_simd_gemv<Q4Layout::Interleaved4, Q4Epi::AddResid>(
        Q4Bank{packed, sb}, q4_no_bank(), x, y, resid, 0u, rows, cols, group_size,
        q4_simd_row0(tgid, sgid), lane);
}

kernel void gemv_q4_mlx_simd_gate_up_gelu_i4(
    device const uchar *gate_packed [[buffer(0)]],
    device const bfloat2 *gate_sb [[buffer(1)]],
    device const bfloat *gate_biases_unused [[buffer(2)]],
    device const uchar *up_packed [[buffer(3)]],
    device const bfloat2 *up_sb [[buffer(4)]],
    device const bfloat *up_biases_unused [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *mid [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    constant uint &group_size [[buffer(10)]],
    constant uint &mid_as_bf16 [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)gate_biases_unused; (void)up_biases_unused;
    q4_simd_gemv<Q4Layout::Interleaved4, Q4Epi::GateUpGelu>(
        Q4Bank{gate_packed, gate_sb}, Q4Bank{up_packed, up_sb}, x, mid, nullptr,
        mid_as_bf16, rows, cols, group_size, q4_simd_row0(tgid, sgid), lane);
}

// Interleaved4 twin of bank-split `gemv_q4_mlx_simd_kv` (math ≡ `gemv_q4_mlx_simd_i4`).
kernel void gemv_q4_mlx_simd_kv_i4(
    device const uchar *k_packed [[buffer(0)]],
    device const bfloat2 *k_sb [[buffer(1)]],
    device const bfloat *k_biases_unused [[buffer(2)]],
    device const uchar *v_packed [[buffer(3)]],
    device const bfloat2 *v_sb [[buffer(4)]],
    device const bfloat *v_biases_unused [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *k_out [[buffer(7)]],
    device float *v_out [[buffer(8)]],
    constant uint &rows [[buffer(9)]],
    constant uint &cols [[buffer(10)]],
    constant uint &group_size [[buffer(11)]],
    constant uint &tg_k [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)k_biases_unused; (void)v_biases_unused;
    const Q4Route at = q4_route_kv(
        tgid, tg_k, Q4Bank{k_packed, k_sb}, k_out, Q4Bank{v_packed, v_sb}, v_out, rows);
    q4_simd_gemv<Q4Layout::Interleaved4, Q4Epi::Store>(
        at.bank, q4_no_bank(), x, at.y, nullptr, 0u, at.rows, cols, group_size,
        q4_simd_row0(at.local_tg, sgid), lane);
}

kernel void gemv_q4_mlx_tiled(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    (void)biases_unused;
    if (tg >= rows) return;
    threadgroup float partial[GEMV_TG];
    const uint row = tg;
    const uint groups_per_row = cols / group_size;
    float acc = 0.0f;
    for (ulong g = tid; g < (ulong)groups_per_row; g += GEMV_TG) {
        const ulong gi = (ulong)row * groups_per_row + g;
        const bfloat2 sbv = sb[gi];
        const float scale = float(sbv.x);
        const float bias = float(sbv.y);
        const ulong base = (ulong)row * cols + g * group_size;
        const ulong xbase = g * group_size;
        for (ulong i = 0ul; i < (ulong)group_size; ++i) {
            acc += dequant_q4_u_bias(packed, base + i, scale, bias) * x[xbase + i];
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GEMV_TG / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        y[row] = partial[0];
    }
}

// ---------------------------------------------------------------------------
// Fused producer Q∥K∥V simd GEMV (one dispatch for all three projections).
//
// Layer-fusion v1 (opt-in: GEMMA_METAL_FUSE_QKV=1 / GEMMA_METAL_FUSE_LAYER=1).
// Motivation: audit_deep_2026-07-18 F2 — fixed cost ≈37 µs per dispatch, so the
// win is dispatch count, not GEMV math. Q/K/V all read the SAME x (post
// input_norm) and differ only in weight bank + output, so they can share one
// launch with zero redundant activation traffic.
//
// Partitioning: threadgroups are assigned whole banks (never straddling), so
// each TG resolves exactly one (packed, sb, out, rows) tuple:
//     [0, tg_q)               -> Q
//     [tg_q, tg_q + tg_k)     -> K
//     [tg_q + tg_k, ...)      -> V
// `tg_q` / `tg_k` are ceil(rows / (SIMD_SG_PER_TG * SIMD_ROWS)) computed host
// side (`q4_route_qkv`). Bank select depends only on `tgid` (TG-uniform);
// `row0` depends on `sgid` (simdgroup-uniform), so the early-out never splits a
// simdgroup before `simd_sum` — same invariant as `gemv_q4_mlx_simd`.
//
// Math is byte-identical to `gemv_q4_mlx_simd` (same `q4_simd_gemv` body), so
// fused vs unfused must be bit-exact.
// ---------------------------------------------------------------------------
kernel void gemv_q4_mlx_simd_qkv(
    device const uchar *q_packed [[buffer(0)]],
    device const bfloat2 *q_sb [[buffer(1)]],
    device const uchar *k_packed [[buffer(2)]],
    device const bfloat2 *k_sb [[buffer(3)]],
    device const uchar *v_packed [[buffer(4)]],
    device const bfloat2 *v_sb [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *q_out [[buffer(7)]],
    device float *k_out [[buffer(8)]],
    device float *v_out [[buffer(9)]],
    constant uint &rows_q [[buffer(10)]],
    constant uint &rows_kv [[buffer(11)]],
    constant uint &cols [[buffer(12)]],
    constant uint &group_size [[buffer(13)]],
    constant uint &tg_q [[buffer(14)]],
    constant uint &tg_k [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const Q4Route at = q4_route_qkv(
        tgid, tg_q, tg_k, Q4Bank{q_packed, q_sb}, q_out, rows_q, Q4Bank{k_packed, k_sb},
        k_out, Q4Bank{v_packed, v_sb}, v_out, rows_kv);
    q4_simd_gemv<Q4Layout::RowMajor, Q4Epi::Store>(
        at.bank, q4_no_bank(), x, at.y, nullptr, 0u, at.rows, cols, group_size,
        q4_simd_row0(at.local_tg, sgid), lane);
}

// Interleaved4 twin of `gemv_q4_mlx_simd_qkv` (same bank partitioning; i4 pointer walk).
// Opt-in when Hot layout is Interleaved4 (`GEMMA_METAL_GEMV_INTERLEAVE=1`) and
// `GEMMA_METAL_FUSE_QKV` / `FUSE_LAYER` is on. Math ≡ `gemv_q4_mlx_simd_i4`.
kernel void gemv_q4_mlx_simd_qkv_i4(
    device const uchar *q_packed [[buffer(0)]],
    device const bfloat2 *q_sb [[buffer(1)]],
    device const uchar *k_packed [[buffer(2)]],
    device const bfloat2 *k_sb [[buffer(3)]],
    device const uchar *v_packed [[buffer(4)]],
    device const bfloat2 *v_sb [[buffer(5)]],
    device const bfloat *x [[buffer(6)]],
    device float *q_out [[buffer(7)]],
    device float *k_out [[buffer(8)]],
    device float *v_out [[buffer(9)]],
    constant uint &rows_q [[buffer(10)]],
    constant uint &rows_kv [[buffer(11)]],
    constant uint &cols [[buffer(12)]],
    constant uint &group_size [[buffer(13)]],
    constant uint &tg_q [[buffer(14)]],
    constant uint &tg_k [[buffer(15)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const Q4Route at = q4_route_qkv(
        tgid, tg_q, tg_k, Q4Bank{q_packed, q_sb}, q_out, rows_q, Q4Bank{k_packed, k_sb},
        k_out, Q4Bank{v_packed, v_sb}, v_out, rows_kv);
    q4_simd_gemv<Q4Layout::Interleaved4, Q4Epi::Store>(
        at.bank, q4_no_bank(), x, at.y, nullptr, 0u, at.rows, cols, group_size,
        q4_simd_row0(at.local_tg, sgid), lane);
}
