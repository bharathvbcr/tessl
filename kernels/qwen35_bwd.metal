// Backward of the Qwen3.5 row-local ops: RMSNorm, the GDN gated RMSNorm,
// SwiGLU, the attention output gate, the GDN causal conv + SiLU, the GDN
// gates, the attention Q/K norm + partial RoPE and the embedding gather; and
// a column-window copy for moving operands between fused and dense layouts.
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

// ------------------------------------------------------ causal conv + SiLU ---

/// Longest conv kernel the backward takes (`KW`, as the forward's 2..=8).
constant uint CONV_BWD_MAX_KW = 8;

/// `pre = sum_j w[c, j] * x_ext[t + j]` for row `t` of one batch row, where
/// `x_ext` is `KW - 1` zeros followed by the row (the training forward: no
/// carried state). `xc` points at column c of the batch row's first token.
inline float qwen35_conv_pre(device const float *xc, uint ld_x, device const float *wc, uint KW, uint t)
{
    const uint hist = KW - 1u;
    float acc = 0.0f;
    for (uint j = 0u; j < KW; ++j) {
        const uint e = t + j;
        if (e >= hist) {
            acc += wc[j] * xc[(ulong)(e - hist) * ld_x];
        }
    }
    return acc;
}

/// Input gradient of `y = silu(causal_conv1d(x))` from a zero state:
///
///   dpre[t] = dy[t] * silu'(pre[t])
///   dx[s]   = sum_j w[c, j] * dpre[s + KW - 1 - j],  over t in [0, T) of the
///             same batch row
///
/// with `pre` recomputed from `x` (silu is not invertible, so `y` cannot give
/// it back). `x`, `dy`, `dx` are windows of `C` columns of `[B * T, ld]` rows.
///
/// Grid: x = channel in [0, C), y = row in [0, B * T).
kernel void qwen35_conv1d_silu_bwd_dx_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dx [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &T [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant uint &KW [[buffer(7)]],
    constant uint &ld_x [[buffer(8)]],
    constant uint &x_off [[buffer(9)]],
    constant uint &ld_dy [[buffer(10)]],
    constant uint &dy_off [[buffer(11)]],
    constant uint &ld_dx [[buffer(12)]],
    constant uint &dx_off [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const ulong row = gid.y;
    if (c >= C || row >= (ulong)B * T) return;
    const ulong b = row / T;
    const uint s = (uint)(row % T);
    const uint hist = KW - 1u;
    device const float *xc = x + b * T * (ulong)ld_x + x_off + c;
    device const float *dyc = dy + b * T * (ulong)ld_dy + dy_off + c;
    device const float *wc = w + (ulong)c * KW;
    float acc = 0.0f;
    for (uint j = 0u; j < KW; ++j) {
        const uint t = s + hist - j;  // s + hist >= j, so no wrap
        if (t < T) {
            const float pre = qwen35_conv_pre(xc, ld_x, wc, KW, t);
            acc += wc[j] * dyc[(ulong)t * ld_dy] * qwen35_silu_grad(pre);
        }
    }
    dx[row * ld_dx + dx_off + c] = acc;
}

/// Weight gradient of the same conv, `dw[c, j] = sum over rows of dpre[t] *
/// x_ext[t + j]`, as per-block partials: the thread for (channel c, block)
/// sums rows `[blk * rows_per_block, ..)` of the flattened `B * T` rows into
/// `dw_part[blk, c * KW + j]`, so `qwen35_col_sum_blocks_f32` over `C * KW`
/// columns leaves `dw` in the weight's own `[C, KW]` layout.
///
/// Grid: x = channel in [0, C), y = block.
kernel void qwen35_conv1d_silu_bwd_dw_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dw_part [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &T [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant uint &KW [[buffer(7)]],
    constant uint &ld_x [[buffer(8)]],
    constant uint &x_off [[buffer(9)]],
    constant uint &ld_dy [[buffer(10)]],
    constant uint &dy_off [[buffer(11)]],
    constant uint &rows_per_block [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const ulong blk = gid.y;
    const ulong rows = (ulong)B * T;
    const ulong r0 = blk * rows_per_block;
    if (c >= C || r0 >= rows) return;
    const ulong r1 = min(rows, r0 + rows_per_block);
    const uint hist = KW - 1u;
    device const float *wc = w + (ulong)c * KW;
    float acc[CONV_BWD_MAX_KW];
    for (uint j = 0u; j < CONV_BWD_MAX_KW; ++j) {
        acc[j] = 0.0f;
    }
    for (ulong row = r0; row < r1; ++row) {
        const ulong b = row / T;
        const uint t = (uint)(row % T);
        device const float *xc = x + b * T * (ulong)ld_x + x_off + c;
        const float pre = qwen35_conv_pre(xc, ld_x, wc, KW, t);
        const float dpre = dy[row * ld_dy + dy_off + c] * qwen35_silu_grad(pre);
        for (uint j = 0u; j < KW; ++j) {
            const uint e = t + j;
            if (e >= hist) {
                acc[j] += dpre * xc[(ulong)(e - hist) * ld_x];
            }
        }
    }
    device float *out = dw_part + blk * ((ulong)C * KW) + (ulong)c * KW;
    for (uint j = 0u; j < KW; ++j) {
        out[j] = acc[j];
    }
}

// ------------------------------------------------- Q/K norm + partial RoPE ---

/// Simdgroups per threadgroup of `qwen35_attn_qk_norm_rope_bwd_f32`.
constant uint QK_BWD_SG = 4;

/// Backward of one head row of `qwen35_norm_rope_row`: `src` the forward's
/// input row, `g` the gradient of its output. Writes the input's gradient to
/// `dst` and adds this row's `(1 + w)` gradient to the lane's columns in `acc`.
///
///   n    = x * rstd * (1 + w), then RoPE on the first `rotary_dim` dims
///   dn   = R(pos)^T g on the rotary dims (pair p, p + half), g elsewhere
///   dx   = rstd * (dn (1 + w) - xn * mean(dn (1 + w) xn)),  xn = x * rstd
///   dw  += dn * xn
///
/// Each lane owns columns `lane + 32k` and rotates them itself, so a pair's
/// two columns need not share a lane.
inline void qwen35_norm_rope_row_bwd(
    device const float *src,
    device const float *weight,
    device const float *g,
    device float *dst,
    uint D,
    uint rotary_dim,
    uint pos,
    float theta,
    float eps,
    uint lane,
    thread float *acc)
{
    float ss = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        ss += src[d] * src[d];
    }
    const float rstd = rsqrt(simd_sum(ss) / (float)D + eps);
    const uint half_rot = rotary_dim / 2u;
    float dn[BWD_MAX_COLS];
    float dot = 0.0f;
    uint k = 0;
    for (uint d = lane; d < D; d += 32u, ++k) {
        float v;
        if (d < rotary_dim) {
            const uint p = d < half_rot ? d : d - half_rot;
            const float angle = qwen35_rope_angle(p, rotary_dim, pos, theta);
            const float c = precise::cos(angle);
            const float s = precise::sin(angle);
            // out[p] = n0 c - n1 s, out[p + half] = n1 c + n0 s.
            v = d < half_rot ? g[p] * c + g[p + half_rot] * s : g[d] * c - g[p] * s;
        } else {
            v = g[d];
        }
        dn[k] = v;
        dot += v * (1.0f + weight[d]) * src[d] * rstd;
    }
    const float m = simd_sum(dot) / (float)D;
    k = 0;
    for (uint d = lane; d < D; d += 32u, ++k) {
        const float xn = src[d] * rstd;
        dst[d] = rstd * (dn[k] * (1.0f + weight[d]) - xn * m);
        acc[k] += dn[k] * xn;
    }
}

/// Backward of `qwen35_attn_qk_norm_rope` as training runs it (token t of each
/// batch row at position t, caches `[B, T, Hkv, D]`), over the fused
/// projection window `p` the forward read:
///
///   * query head h: `dq [B*T, Hq, D]` back through RoPE and the q norm into
///     columns `q_off + h*2D ..+D` of `dp` (the gate columns after it are
///     `qwen35_attn_gate_bwd_f32`'s);
///   * key head h: `dk [B*T, Hkv, D]` likewise into `k_off + h*D`;
///   * value head h: `dv [B*T, Hkv, D]` copied into `v_off + h*D`.
///
/// `dp` has `p`'s row stride. One simdgroup per (token, head) unit, as in the
/// forward; each threadgroup takes `rows_per_block` tokens' units, and the
/// (1 + w) gradients are summed per block, simdgroups in order, into
/// `part[block, :]` (q norm) and `part[nblocks + block, :]` (k norm).
///
/// Grid: ceil(B*T / rows_per_block) threadgroups of QK_BWD_SG * 32 threads,
/// `D <= 32 * BWD_MAX_COLS`.
kernel void qwen35_attn_qk_norm_rope_bwd_f32(
    device const float *p [[buffer(0)]],
    device const float *q_norm_w [[buffer(1)]],
    device const float *k_norm_w [[buffer(2)]],
    device const float *dq [[buffer(3)]],
    device const float *dk [[buffer(4)]],
    device const float *dv [[buffer(5)]],
    device float *dp [[buffer(6)]],
    device float *part [[buffer(7)]],
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &Hq [[buffer(10)]],
    constant uint &Hkv [[buffer(11)]],
    constant uint &D [[buffer(12)]],
    constant uint &rotary_dim [[buffer(13)]],
    constant uint &ld_p [[buffer(14)]],
    constant uint &q_off [[buffer(15)]],
    constant uint &k_off [[buffer(16)]],
    constant uint &v_off [[buffer(17)]],
    constant float &theta [[buffer(18)]],
    constant float &eps [[buffer(19)]],
    constant uint &rows_per_block [[buffer(20)]],
    uint blk [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sg_acc[2 * QK_BWD_SG * BWD_MAX_COLS * 32];
    const ulong rows = (ulong)B * T;
    const ulong nblocks = (rows + rows_per_block - 1) / rows_per_block;
    const ulong r0 = (ulong)blk * rows_per_block;
    if (r0 >= rows) return;  // uniform per threadgroup
    const ulong r1 = min(rows, r0 + rows_per_block);
    const uint heads = Hq + 2u * Hkv;
    float acc_q[BWD_MAX_COLS], acc_k[BWD_MAX_COLS];
    for (uint k = 0; k < BWD_MAX_COLS; ++k) {
        acc_q[k] = 0.0f;
        acc_k[k] = 0.0f;
    }
    const ulong u0 = r0 * heads, u1 = r1 * heads;
    for (ulong u = u0 + sg; u < u1; u += QK_BWD_SG) {
        const ulong r = u / heads;
        const uint j = (uint)(u % heads);
        const uint pos = (uint)(r % T);
        device const float *row = p + r * ld_p;
        device float *drow = dp + r * ld_p;
        if (j < Hq) {
            const ulong col = q_off + (ulong)j * 2u * D;
            qwen35_norm_rope_row_bwd(row + col, q_norm_w, dq + (r * Hq + j) * (ulong)D, drow + col,
                                     D, rotary_dim, pos, theta, eps, lane, acc_q);
        } else if (j < Hq + Hkv) {
            const uint h = j - Hq;
            const ulong col = k_off + (ulong)h * D;
            qwen35_norm_rope_row_bwd(row + col, k_norm_w, dk + (r * Hkv + h) * (ulong)D, drow + col,
                                     D, rotary_dim, pos, theta, eps, lane, acc_k);
        } else {
            const uint h = j - Hq - Hkv;
            device const float *src = dv + (r * Hkv + h) * (ulong)D;
            device float *out = drow + v_off + (ulong)h * D;
            for (uint d = lane; d < D; d += 32u) {
                out[d] = src[d];
            }
        }
    }
    // Combine the simdgroups' column sums in simdgroup order.
    const uint per = (D + 31u) / 32u;
    const uint half_acc = QK_BWD_SG * BWD_MAX_COLS * 32;
    for (uint k = 0; k < per; ++k) {
        sg_acc[(sg * per + k) * 32 + lane] = acc_q[k];
        sg_acc[half_acc + (sg * per + k) * 32 + lane] = acc_k[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint k = 0; k < per; ++k) {
            const uint d = lane + 32u * k;
            if (d >= D) continue;
            float sq = 0.0f, sk = 0.0f;
            for (uint g = 0; g < QK_BWD_SG; ++g) {
                sq += sg_acc[(g * per + k) * 32 + lane];
                sk += sg_acc[half_acc + (g * per + k) * 32 + lane];
            }
            part[(ulong)blk * D + d] = sq;
            part[(nblocks + blk) * D + d] = sk;
        }
    }
}

// ------------------------------------------------------ embedding gather ---

/// Backward of the embedding gather, added into the table's gradient:
/// `dw[id, :] += sum of dh[r, :] over the rows r that read id`.
///
/// The host groups the rows by id (it knows the ids): run `u` covers
/// `pos[run_start[u] .. run_start[u + 1])`, rows in ascending order, all
/// reading id `uniq[u]`. One thread per (run, column) sums its run in that
/// order and adds the sum once, and no two runs share an id, so there are no
/// atomics and the result is the same on every run. With a tied LM head this
/// adds onto the head's weight gradient.
///
/// Grid: x = column in [0, hidden), y = run in [0, n_runs).
kernel void qwen35_embed_rows_bwd_f32(
    device const float *dh [[buffer(0)]],
    device const uint *pos [[buffer(1)]],
    device const uint *run_start [[buffer(2)]],
    device const uint *uniq [[buffer(3)]],
    device float *dw [[buffer(4)]],
    constant uint &n_runs [[buffer(5)]],
    constant uint &hidden [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint u = gid.y;
    if (col >= hidden || u >= n_runs) return;
    float s = 0.0f;
    for (uint i = run_start[u]; i < run_start[u + 1u]; ++i) {
        s += dh[(ulong)pos[i] * hidden + col];
    }
    dw[(ulong)uniq[u] * hidden + col] += s;
}

// --------------------------------------------------------------- GDN gates ---

/// Backward of `qwen35_gdn_gates_f32` (`g = -exp(A_log) * softplus(a +
/// dt_bias)`, `beta = sigmoid(b)`), from `dg`, `dbeta` `[rows, H]`:
///
///   da       = dg * -exp(A_log) * softplus'(a + dt_bias)
///   db       = dbeta * beta * (1 - beta)
///   dA_log  += dg * g           ddt_bias += da        (sums over rows)
///
/// with torch's softplus derivative: 1 above the threshold (20), sigmoid
/// below. `da` and `db` go to the a and b columns of the fused projection's
/// gradient `dp` (the projection's row stride). The sums are per block of
/// `rows_per_block` rows: `part[blk, h]` for dA_log, `part[nblocks + blk, h]`
/// for ddt_bias, summed over blocks in order by `qwen35_col_sum_blocks_f32`.
///
/// Grid: x = head in [0, H), y = block.
kernel void qwen35_gdn_gates_bwd_f32(
    device const float *p [[buffer(0)]],
    device const float *a_log [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *dg [[buffer(3)]],
    device const float *dbeta [[buffer(4)]],
    device float *dp [[buffer(5)]],
    device float *part [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &H [[buffer(8)]],
    constant uint &ld [[buffer(9)]],
    constant uint &a_off [[buffer(10)]],
    constant uint &b_off [[buffer(11)]],
    constant uint &rows_per_block [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint h = gid.x;
    const ulong blk = gid.y;
    const ulong r0 = blk * rows_per_block;
    if (h >= H || r0 >= rows) return;
    const ulong r1 = min((ulong)rows, r0 + rows_per_block);
    const ulong nblocks = ((ulong)rows + rows_per_block - 1) / rows_per_block;
    const float neg_a = -exp(a_log[h]);
    float acc_log = 0.0f, acc_dt = 0.0f;
    for (ulong r = r0; r < r1; ++r) {
        device const float *row = p + r * ld;
        device float *drow = dp + r * ld;
        const ulong o = r * H + h;
        const float x = row[a_off + h] + dt_bias[h];
        const float sp_grad = x > 20.0f ? 1.0f : qwen35_sigmoid(x);
        const float da = dg[o] * neg_a * sp_grad;
        const float s = qwen35_sigmoid(row[b_off + h]);
        drow[a_off + h] = da;
        drow[b_off + h] = dbeta[o] * s * (1.0f - s);
        acc_log += dg[o] * qwen35_log_decay(row[a_off + h], a_log[h], dt_bias[h]);
        acc_dt += da;
    }
    part[blk * H + h] = acc_log;
    part[(nblocks + blk) * H + h] = acc_dt;
}

// ------------------------------------------------------------ window copy ---

/// `dst[r, dst_off + c] = src[r, src_off + c]` for `c < width`: a column
/// window of one row-major matrix into another's (the GDN's q, k, v between
/// the conv output and the training op's dense operands, both ways).
///
/// Grid: x = column in [0, width), y = row.
kernel void qwen35_copy_cols_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &ld_src [[buffer(4)]],
    constant uint &src_off [[buffer(5)]],
    constant uint &ld_dst [[buffer(6)]],
    constant uint &dst_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= width || r >= rows) return;
    dst[(ulong)r * ld_dst + dst_off + c] = src[(ulong)r * ld_src + src_off + c];
}
