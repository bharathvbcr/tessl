// Backward of the Qwen3.5 row-local ops: RMSNorm, the GDN gated RMSNorm,
// SwiGLU and the attention output gate.
//
// Every operand keeps the forward's layout, windows included (`ld`, `off`),
// so a gradient lands where the next GEMM backward reads it: dgate/dup side by
// side in the fused [T, 2I] projection gradient, the output gate's gradient in
// its column of the fused attention projection gradient.
//
// Weight gradients are sums over every row. They are never accumulated with
// atomics: each threadgroup sums its block of rows into its own row of a
// partial `[nblocks, D]` buffer, and `qwen35_col_sum_blocks_f32` sums the
// blocks in order. Results are therefore deterministic.
//
// All arithmetic is f32; the inputs are the forward's f32 inputs.
#include <metal_stdlib>
#include "reduce_tree.h"
#include "qwen35_act.h"
using namespace metal;

/// Columns a thread can own in a row-block weight-gradient kernel: the host
/// picks the threadgroup size so that `D <= tptg * BWD_MAX_COLS`.
constant uint BWD_MAX_COLS = 16;

/// Simdgroups per threadgroup of `qwen35_gated_rms_norm_bwd_f32` (128 threads).
constant uint GATED_BWD_SG = 4;

/// silu'(g) = s (1 + g (1 - s)), s = sigmoid(g).
inline float qwen35_silu_grad(float g)
{
    const float s = qwen35_sigmoid(g);
    return s * (1.0f + g * (1.0f - s));
}

/// RMSNorm backward, `y = x * rstd * w`, `rstd = rsqrt(mean(x^2) + eps)`:
///
///   dx = rstd * (dy * w) - x * rstd^3 * mean(dy * w * x)
///   dw = sum over rows of dy * x * rstd
///
/// `x`, `dy`, `dx` are dense `[rows, D]`; with `flags & 1` dx is added to
/// what `dx` holds (the residual stream's gradient) instead of stored. Each
/// threadgroup takes `rows_per_block` consecutive rows and writes its dw
/// partial to `dw_part[block, :]`.
///
/// Grid: ceil(rows / rows_per_block) threadgroups of `tptg` threads,
/// `D <= tptg * BWD_MAX_COLS`.
kernel void qwen35_rms_norm_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dx [[buffer(3)]],
    device float *dw_part [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &D [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant uint &rows_per_block [[buffer(8)]],
    constant uint &flags [[buffer(9)]],
    uint blk [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float scratch[2 * REDUCE_MAX_SIMDGROUPS];
    const ulong r0 = (ulong)blk * rows_per_block;
    if (r0 >= rows) return;  // uniform per threadgroup
    const ulong r1 = min((ulong)rows, r0 + rows_per_block);
    float acc[BWD_MAX_COLS];
    for (uint k = 0; k < BWD_MAX_COLS; ++k) {
        acc[k] = 0.0f;
    }
    for (ulong r = r0; r < r1; ++r) {
        device const float *xr = x + r * D;
        device const float *gr = dy + r * D;
        float ss = 0.0f, dot = 0.0f;
        for (uint d = lid; d < D; d += tptg) {
            const float xv = xr[d];
            ss += xv * xv;
            dot += gr[d] * w[d] * xv;
        }
        ss = reduce_row_add(ss, scratch, sgid, lane, tptg);
        // A second scratch region: the first reduction's reads of `scratch`
        // are not fenced from these writes (reduce_row_add's contract).
        dot = reduce_row_add(dot, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg);
        const float rstd = precise::rsqrt(ss / (float)D + eps);
        const float c = rstd * rstd * rstd * dot / (float)D;
        device float *xo = dx + r * D;
        uint k = 0;
        for (uint d = lid; d < D; d += tptg, ++k) {
            const float v = rstd * gr[d] * w[d] - xr[d] * c;
            xo[d] = (flags & 1u) != 0u ? xo[d] + v : v;
            acc[k] += gr[d] * xr[d] * rstd;
        }
        // Every lane has read `scratch` before the next row's reductions
        // rewrite it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint k = 0;
    for (uint d = lid; d < D; d += tptg, ++k) {
        dw_part[(ulong)blk * D + d] = acc[k];
    }
}

/// `out[d] = sum over b of part[b, d]`, in block order.
///
/// Grid: x = column in [0, D).
kernel void qwen35_col_sum_blocks_f32(
    device const float *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &blocks [[buffer(2)]],
    constant uint &D [[buffer(3)]],
    uint d [[thread_position_in_grid]])
{
    if (d >= D) return;
    float s = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        s += part[(ulong)b * D + d];
    }
    out[d] = s;
}

/// Gated RMSNorm backward, per (row, head) unit of `D` values:
/// `y = w * (x * rstd) * silu(z)`, `rstd = rsqrt(mean(x^2) + eps)`.
///
///   dz  = dy * w * xn * silu'(z)                     xn = x * rstd
///   dxn = dy * w * silu(z)
///   dx  = rstd * (dxn - xn * mean(dxn * xn))
///   dw  = sum over units of dy * xn * silu(z)
///
/// One simdgroup per unit, as in the forward. Each threadgroup takes
/// `units_per_block` consecutive units, its simdgroups striding through them;
/// lane `l` owns columns `l, l + 32, ...` of dw, and the simdgroups' sums are
/// combined in simdgroup order into `dw_part[block, :]`.
///
/// Grid: ceil(rows * H / units_per_block) threadgroups of GATED_BWD_SG * 32
/// threads, `D <= 32 * BWD_MAX_COLS`.
kernel void qwen35_gated_rms_norm_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *z [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device const float *dy [[buffer(3)]],
    device float *dx [[buffer(4)]],
    device float *dz [[buffer(5)]],
    device float *dw_part [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &H [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &ld_x [[buffer(10)]],
    constant uint &x_off [[buffer(11)]],
    constant uint &ld_z [[buffer(12)]],
    constant uint &z_off [[buffer(13)]],
    constant uint &ld_dy [[buffer(14)]],
    constant uint &dy_off [[buffer(15)]],
    constant uint &ld_dx [[buffer(16)]],
    constant uint &dx_off [[buffer(17)]],
    constant uint &ld_dz [[buffer(18)]],
    constant uint &dz_off [[buffer(19)]],
    constant float &eps [[buffer(20)]],
    constant uint &units_per_block [[buffer(21)]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_acc[GATED_BWD_SG * BWD_MAX_COLS * 32];
    const uint nsg = GATED_BWD_SG;
    const ulong units = (ulong)rows * H;
    const ulong u0 = (ulong)blk * units_per_block;
    if (u0 >= units) return;  // uniform per threadgroup
    const ulong u1 = min(units, u0 + units_per_block);
    float acc[BWD_MAX_COLS];
    for (uint k = 0; k < BWD_MAX_COLS; ++k) {
        acc[k] = 0.0f;
    }
    for (ulong u = u0 + sg; u < u1; u += nsg) {
        const ulong r = u / H, h = u % H;
        device const float *xr = x + r * ld_x + x_off + h * D;
        device const float *zr = z + r * ld_z + z_off + h * D;
        device const float *gr = dy + r * ld_dy + dy_off + h * D;
        device float *xo = dx + r * ld_dx + dx_off + h * D;
        device float *zo = dz + r * ld_dz + dz_off + h * D;
        float ss = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            ss += xr[d] * xr[d];
        }
        const float rstd = precise::rsqrt(simd_sum(ss) / (float)D + eps);
        float dot = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            const float xn = xr[d] * rstd;
            dot += gr[d] * w[d] * qwen35_silu(zr[d]) * xn;
        }
        const float m = simd_sum(dot) / (float)D;
        uint k = 0;
        for (uint d = lane; d < D; d += 32u, ++k) {
            const float xn = xr[d] * rstd;
            const float zv = zr[d];
            const float s = qwen35_silu(zv);
            const float dxn = gr[d] * w[d] * s;
            xo[d] = rstd * (dxn - xn * m);
            zo[d] = gr[d] * w[d] * xn * qwen35_silu_grad(zv);
            acc[k] += gr[d] * xn * s;
        }
    }
    // Combine the simdgroups' column sums in simdgroup order.
    const uint per = (D + 31u) / 32u;
    for (uint k = 0; k < per; ++k) {
        sg_acc[(sg * per + k) * 32 + lane] = acc[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint k = 0; k < per; ++k) {
            const uint d = lane + 32u * k;
            if (d >= D) continue;
            float s = 0.0f;
            for (uint g = 0; g < nsg; ++g) {
                s += sg_acc[(g * per + k) * 32 + lane];
            }
            dw_part[(ulong)blk * D + d] = s;
        }
    }
}

/// SwiGLU backward, `y = silu(gate) * up`:
/// `dgate = dy * up * silu'(gate)`, `dup = dy * silu(gate)`. Windows as in
/// the forward; `dgate` and `dup` may be two windows of one buffer (the fused
/// projection gradient).
///
/// Grid: x = column in [0, width), y = row.
kernel void qwen35_swiglu_bwd_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dgate [[buffer(3)]],
    device float *dup [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &ld_gate [[buffer(7)]],
    constant uint &gate_off [[buffer(8)]],
    constant uint &ld_up [[buffer(9)]],
    constant uint &up_off [[buffer(10)]],
    constant uint &ld_dy [[buffer(11)]],
    constant uint &dy_off [[buffer(12)]],
    constant uint &ld_dgate [[buffer(13)]],
    constant uint &dgate_off [[buffer(14)]],
    constant uint &ld_dup [[buffer(15)]],
    constant uint &dup_off [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= width || r >= rows) return;
    const float g = gate[(ulong)r * ld_gate + gate_off + col];
    const float u = up[(ulong)r * ld_up + up_off + col];
    const float d = dy[(ulong)r * ld_dy + dy_off + col];
    dgate[(ulong)r * ld_dgate + dgate_off + col] = d * u * qwen35_silu_grad(g);
    dup[(ulong)r * ld_dup + dup_off + col] = d * qwen35_silu(g);
}

/// Attention output gate backward, `out = attn * sigmoid(gate)` with head h's
/// gate at columns `q_off + h*2D + D ..` of the fused projection `p`:
/// `d_attn = dy * s`, `d_gate = dy * attn * s * (1 - s)`, written to the same
/// columns of the fused projection gradient `dp` (`[rows, ld_p]`), whose
/// query columns the attention backward fills.
///
/// Grid: x = column in [0, Hq*D), y = row.
kernel void qwen35_attn_gate_bwd_f32(
    device const float *attn [[buffer(0)]],
    device const float *p [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *d_attn [[buffer(3)]],
    device float *dp [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &Hq [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant uint &ld_p [[buffer(8)]],
    constant uint &q_off [[buffer(9)]],
    constant uint &ld_dy [[buffer(10)]],
    constant uint &dy_off [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= Hq * D || r >= rows) return;
    const uint h = col / D;
    const uint d = col % D;
    const ulong gi = (ulong)r * ld_p + q_off + (ulong)h * 2u * D + D + d;
    const float s = qwen35_sigmoid(p[gi]);
    const float a = attn[(ulong)r * Hq * D + col];
    const float g = dy[(ulong)r * ld_dy + dy_off + col];
    d_attn[(ulong)r * Hq * D + col] = g * s;
    dp[gi] = g * a * s * (1.0f - s);
}
