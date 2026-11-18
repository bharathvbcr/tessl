// TimesFM 3 kernels (src/timesfm3.rs): the query/key preparation and the
// attention, at TimesFM 3's head dimension of 80 (1280 over 16 heads).
//
// Everything else in the forward is a tessl kernel already: the GEMMs (exact
// f32, with ReLU, bias and accumulate epilogues), RMSNorm and the fused
// RMSNorm-plus-residual.
//
// Both kernels are f32 in and out and use `precise::` for each division and
// reciprocal square root (tessl compiles with fast math; see bert.metal).
#include <metal_stdlib>
#include "attn_rows.h"
using namespace metal;

constant uint TFM3_D = 80;
constant uint TFM3_HALF = 40;

/// Upstream's query/key order, in place on a `[rows, H * 80]` buffer: RoPE
/// (half-split pairs `(i, i + 40)`, angles from host tables), then RMSNorm
/// over the head's 80 values (`eps` = f32 epsilon, torch's `nn.RMSNorm`
/// default), then the per-dim gain. The query gain arrives with PerDimScale
/// and the attention scale folded in (src/timesfm3.rs), so attention runs at
/// scale 1. One thread per (row, head) of Q, then of K.
///
/// `pos` of row `m` is `m % n`: a sequence row's patch index (rows are
/// laid out `[b, v, n]`). `rope == 0` (variate attention) skips the rotation
/// and the tables are not read.
kernel void timesfm3_qk_prep_h80(
    device float *Q [[buffer(0)]],
    device float *K [[buffer(1)]],
    device const float *gq [[buffer(2)]],
    device const float *gk [[buffer(3)]],
    device const float *sin_t [[buffer(4)]],
    device const float *cos_t [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &n [[buffer(8)]],
    constant uint &rope [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint per = rows * H;
    if (gid >= 2u * per) { return; }
    const bool is_k = gid >= per;
    const uint idx = is_k ? gid - per : gid;
    const uint m = idx / H;
    const uint h = idx % H;
    device float *x = (is_k ? K : Q) + (ulong)m * H * TFM3_D + (ulong)h * TFM3_D;
    device const float *g = is_k ? gk : gq;
    float v[TFM3_D];
    for (uint i = 0; i < TFM3_D; ++i) { v[i] = x[i]; }
    if (rope != 0u) {
        const ulong t = (ulong)(m % n) * TFM3_HALF;
        for (uint i = 0; i < TFM3_HALF; ++i) {
            const float f = v[i];
            const float s = v[i + TFM3_HALF];
            const float c = cos_t[t + i];
            const float sn = sin_t[t + i];
            v[i] = f * c - s * sn;
            v[i + TFM3_HALF] = s * c + f * sn;
        }
    }
    float ss = 0.0f;
    for (uint i = 0; i < TFM3_D; ++i) { ss += v[i] * v[i]; }
    const float r = precise::rsqrt(precise::divide(ss, (float)TFM3_D) + eps);
    for (uint i = 0; i < TFM3_D; ++i) { x[i] = v[i] * r * g[i]; }
}

/// Attention over strided sequences at head dim 80, one simdgroup per eight
/// query rows (four lanes per row, five float4s each; `attn_rows.h`'s body,
/// so the online softmax is `flash_attn_rows`'s).
///
/// Q, K, V and O share one layout, the model's `[b, v, n, H * 80]` rows, and
/// are never permuted: sequence `s` (of `nseq`), position `t` (of `T`) is
/// model row `(s / P) * S1 + (s % P) * S2 + t * S3`.
///
///   * sequence attention: `s` = (b, v), `t` = patch; `P = 1, S1 = T,
///     S3 = 1`. `causal = 1`: query `t` sees keys `info[s] <= k <= t`, where
///     `info[s]` counts the row's leading fully-masked patches.
///   * variate attention: `s` = (b, n), `t` = variate; `P = n, S1 = v * n,
///     S2 = 1, S3 = n`. `causal = 0`: query `t` sees key `k` iff bit `k` of
///     `info[s]` is set (T <= 32).
///
/// A query that sees no key is written as zeros (torch's SDPA on a row whose
/// boolean mask is all false).
kernel void timesfm3_attn_h80_r4_g16(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &P [[buffer(6)]],
    constant uint &S1 [[buffer(7)]],
    constant uint &S2 [[buffer(8)]],
    constant uint &S3 [[buffer(9)]],
    constant uint &causal [[buffer(10)]],
    device const uint *info [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]])
{
    constexpr uint R = 4;
    constexpr uint SGT = 16;
    constexpr uint RPS = 32u / R;
    constexpr uint DPV = 80u / (4u * R);
    constexpr uint RPT = SGT * RPS;
    const uint tid = tpitg.x;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint sub = lane / R;
    const uint dl = lane % R;

    const uint sh = tgpig.y;
    const uint h = sh % H;
    const uint s = sh / H;
    const ulong base_row = (ulong)tgpig.x * RPT + sg * RPS;
    if (base_row >= (ulong)T) { return; }
    const uint live_rows = min(RPS, (uint)((ulong)T - base_row));
    const bool row_live = sub < live_rows;
    const uint t_q = row_live ? (uint)(base_row + sub) : (uint)base_row;

    const ulong row_stride = (ulong)H * 80u;
    const ulong seq_base = (ulong)(s / P) * S1 + (ulong)(s % P) * S2;
    const ulong head = (ulong)h * 80u;
    const uint inf = info[s];
    const ulong lo = causal != 0u ? min((ulong)inf, (ulong)T) : 0ul;
    const ulong q_hi = base_row + live_rows - 1ul;
    const ulong t_end = causal != 0u ? q_hi + 1ul : (ulong)T;

    const ulong o_off = (seq_base + (ulong)t_q * S3) * row_stride + head;
    float4 q_reg[DPV];
    float4 acc[DPV];
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    if (lo < t_end) {
        device const float4 *Q4 = (device const float4 *)(Q + o_off);
        for (uint j = 0; j < DPV; ++j) {
            q_reg[j] = row_live ? Q4[dl + j * R] : float4(0.0f);
        }
        for (ulong t = lo; t < t_end; ++t) {
            const ulong kv = (seq_base + t * S3) * row_stride + head;
            const float part = attn_rows_dot<DPV, R>(q_reg, (device const float4 *)(K + kv), dl);
            const bool live = row_live
                && (causal != 0u ? t <= (ulong)t_q : ((inf >> (uint)t) & 1u) != 0u);
            attn_rows_step<DPV, R>(part, live, scale, (device const float4 *)(V + kv), dl, acc, m_i, l_i);
        }
    }
    if (!row_live) { return; }
    const float inv_l = l_i > 0.0f ? precise::divide(1.0f, l_i) : 0.0f;
    attn_rows_store<DPV, R>(O, o_off, dl, acc, inv_l, 0u);
}
