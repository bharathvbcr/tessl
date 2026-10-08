// BERT encoder kernels: the embedding sum, LayerNorm (plain and fused with a
// bias and a residual), a bias add, the erf GELU, and the doc-side sparse
// pooling `max_t log1p(relu(logits + bias))` over each sequence's rows.
//
// Every kernel here is f32 in and out and uses `precise::` for each
// transcendental and each division: tessl compiles kernels with fast math,
// under which `exp`, `log`, `rsqrt` and `/` are approximations, and these
// feed a 1e-4 parity bound against torch's fp32 forward (src/bert.rs). The
// division is not a nicety: a fast `sum / D` is `sum * (1/D)`, so the mean of
// seven 2.5s is not 2.5, and LayerNorm's `eps = 1e-12` then multiplies that
// residue by a million.
//
// LayerNorm is two-pass, one threadgroup per row: the mean, then the mean of
// squared deviations from it (torch's biased variance), each reduced with
// `reduce_row_add` into its own scratch region (that helper's contract). A
// one-pass E[x^2] - E[x]^2 cancels catastrophically on a row with a large
// mean, which is the residual stream after a few layers. A row with zero
// variance comes out as exactly `bias`: the deviations are exact zeros, and
// `eps` keeps the reciprocal finite.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// erfc(x) to a relative error below 1.2e-7 everywhere (the Chebyshev fit of
/// Numerical Recipes' `erfcc`). Relative rather than absolute accuracy is the
/// point: GELU's left tail is `0.5 x erfc(|x|/sqrt 2)`, a tiny number that an
/// absolute-error erf (Abramowitz-Stegun 7.1.26) gets wrong in every digit.
/// |x| is capped at 10, past which erfc is below f32's smallest normal; the
/// cap keeps `z * z` finite, which fast math may assume.
static inline float bert_erfc(float x) {
    const float z = min(fabs(x), 10.0f);
    const float t = precise::divide(1.0f, 1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
        + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f
        + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float r = t * precise::exp(poly);
    return x >= 0.0f ? r : 2.0f - r;
}

/// torch's exact GELU, `0.5 x (1 + erf(x / sqrt 2))`, written as
/// `0.5 x erfc(-x / sqrt 2)` so the negative tail keeps its relative accuracy.
static inline float bert_gelu_erf(float x) {
    return 0.5f * x * bert_erfc(-x * 0.70710678118654752f);
}

/// log1p for x >= 0 without losing the small-x digits `log(1 + x)` drops:
/// `u = 1 + x` is rounded, and `log(u) * x / (u - 1)` divides that rounding
/// back out (Goldberg's correction). Exact for x = 0.
static inline float bert_log1p(float x) {
    const float u = 1.0f + x;
    if (u == 1.0f) return x;
    return precise::log(u) * precise::divide(x, u - 1.0f);
}

/// Mean and 1/sqrt(var + eps) of one row whose element `d` is `VALUE(d)`.
/// A macro rather than a function so the three kernels can each say what an
/// element is (a gather sum, a fused residual, a plain read) without a
/// function-pointer or a temporary row.
#define BERT_ROW_STATS(VALUE, D, EPS, MEAN, INV)                              \
    threadgroup float scratch[2u * REDUCE_MAX_SIMDGROUPS];                    \
    float MEAN;                                                               \
    float INV;                                                                \
    {                                                                         \
        float s = 0.0f;                                                       \
        for (ulong d = lid; d < (ulong)(D); d += tptg) { s += (VALUE(d)); }   \
        MEAN = precise::divide(reduce_row_add(s, scratch, sgid, lane, tptg),  \
                               (float)(D));                                   \
        float q = 0.0f;                                                       \
        for (ulong d = lid; d < (ulong)(D); d += tptg) {                      \
            const float c = (VALUE(d)) - MEAN;                                \
            q += c * c;                                                       \
        }                                                                     \
        const float var = precise::divide(                                    \
            reduce_row_add(q, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg), \
            (float)(D));                                                      \
        INV = precise::rsqrt(var + (EPS));                                    \
    }

/// out[r] = LayerNorm(word[ids[r]] + pos[r % seq] + type_row) * w + b.
///
/// `ids` are host-checked below `vocab`; an id past it still cannot read out
/// of bounds here (it is clamped) and its row is written as NaN, the
/// `embed_rows` contract, so a corrupted id is loud rather than a plausible
/// vector. `seq` is host-checked against the position table's length.
kernel void bert_embed_layer_norm_f32(
    device const uint *ids [[buffer(0)]],
    device const float *word [[buffer(1)]],
    device const float *pos [[buffer(2)]],
    device const float *type_row [[buffer(3)]],
    device const float *w [[buffer(4)]],
    device const float *b [[buffer(5)]],
    device float *out [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &seq [[buffer(9)]],
    constant uint &vocab [[buffer(10)]],
    constant float &eps [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    const uint id = ids[row];
    const bool bad = id >= vocab;
    device const float *wr = word + (ulong)(bad ? 0u : id) * dim;
    device const float *pr = pos + (ulong)(row % seq) * dim;
    device float *o = out + (ulong)row * dim;
#define BERT_EMBED_VALUE(d) (wr[(d)] + pr[(d)] + type_row[(d)])
    BERT_ROW_STATS(BERT_EMBED_VALUE, dim, eps, mean, inv)
    // NaN as bits, not a float: fast math may assume no value is NaN and fold
    // a NaN literal away (qwen35_score.metal stores its NaN the same way).
    device uint *ob = (device uint *)o;
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        const float v = (BERT_EMBED_VALUE(d) - mean) * inv * w[d] + b[d];
        ob[d] = bad ? 0x7fc00000u : as_type<uint>(v);
    }
#undef BERT_EMBED_VALUE
}

/// In place, resid[r] = LayerNorm(y[r] + bias + resid[r]) * w + b: BERT's
/// `BertSelfOutput` / `BertOutput` after their dense layer, in one pass.
/// Each lane reads and writes only its own columns of `resid`, so in place
/// is safe.
kernel void bert_bias_residual_layer_norm_f32(
    device const float *y [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *resid [[buffer(2)]],
    device const float *w [[buffer(3)]],
    device const float *b [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    device const float *yr = y + (ulong)row * dim;
    device float *rr = resid + (ulong)row * dim;
#define BERT_RESID_VALUE(d) (yr[(d)] + bias[(d)] + rr[(d)])
    BERT_ROW_STATS(BERT_RESID_VALUE, dim, eps, mean, inv)
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        rr[d] = (BERT_RESID_VALUE(d) - mean) * inv * w[d] + b[d];
    }
#undef BERT_RESID_VALUE
}

/// out[r] = LayerNorm(x[r]) * w + b.
kernel void bert_layer_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *b [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    device const float *xr = x + (ulong)row * dim;
    device float *o = out + (ulong)row * dim;
#define BERT_PLAIN_VALUE(d) (xr[(d)])
    BERT_ROW_STATS(BERT_PLAIN_VALUE, dim, eps, mean, inv)
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        o[d] = (BERT_PLAIN_VALUE(d) - mean) * inv * w[d] + b[d];
    }
#undef BERT_PLAIN_VALUE
}

/// In place, x[r, c] += bias[c] over `rows * cols` elements: the bias of an
/// `nn.Linear` after an exact-f32 GEMM, which has no epilogue.
kernel void bert_bias_add_f32(
    device float *x [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] += bias[gid % cols];
}

/// In place, x[r, c] = gelu_erf(x[r, c] + bias[c]).
kernel void bert_bias_gelu_erf_f32(
    device float *x [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = bert_gelu_erf(x[gid] + bias[gid % cols]);
}

/// pooled[s, v] = max(pooled[s, v], log1p(relu(max over rows r of segment s
/// of logits[r, v] + bias[v]))).
///
/// The doc-side sparse vector is `max_t log1p(relu(logit_t)) * mask_t`. Every
/// term is >= 0 and a masked row contributes 0, so it is the max over the
/// live rows with 0 as the floor; and `log1p . relu` is non-decreasing, so it
/// may be applied once to the largest logit rather than to every one. Running
/// into `pooled` (zeroed once by the host) lets the head run over the rows in
/// blocks without holding every row's logits at once.
///
/// One thread per (segment, vocabulary column): adjacent threads read
/// adjacent columns of a row. `segments` are `[S][2]` (start, end) rows of
/// `logits`; device values, so `end` is clamped to `rows` and an empty range
/// leaves `pooled` as it was.
kernel void bert_segment_sparse_max_f32(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device const uint *segments [[buffer(2)]],
    device float *pooled [[buffer(3)]],
    constant uint &S [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &V [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if ((ulong)gid >= (ulong)S * V) return;
    const uint s = gid / V;
    const uint v = gid % V;
    const uint end = min(segments[2u * s + 1u], rows);
    const uint start = min(segments[2u * s], end);
    if (start == end) return;
    float m = logits[(ulong)start * V + v];
    for (uint r = start + 1u; r < end; ++r) {
        m = max(m, logits[(ulong)r * V + v]);
    }
    const float act = bert_log1p(max(m + bias[v], 0.0f));
    device float *p = pooled + (ulong)s * V + v;
    *p = max(*p, act);
}
