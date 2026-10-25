// Cross-entropy over a vocabulary too large to hold as logits, for the rows
// that are supervised.
//
// Fine-tuning a tied-embedding LM on answers supervises about one position
// per sequence, and full-vocabulary logits for every position are what ran a
// 2B model out of memory on MPS. `tessl::cross_entropy` therefore gathers the
// supervised rows of the hidden states (`ce_gather_rows_*`), then walks the
// vocabulary in chunks: each chunk's logits come from a GEMM against that
// slice of the weight, `ce_lse_update` folds them into a running log-sum-exp
// per row over the classes other than the target (and picks up the target's
// logit when its chunk passes), and on the second walk `ce_softmax_grad` turns
// the recomputed chunk into `(softmax - onehot) * scale` in place for the
// gradient GEMMs, written as `exp(z_j - L) * sigmoid(-margin)` off the target
// and `-sigmoid(-margin)` on it so it survives margins past the f32 cliff.
// Nothing larger than `[rows, chunk]` logits ever exists.
//
// Row indices and targets are validated on the host, which sees them.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// `out[n, c] = h[rows[n] * ld + off + c]` widened to f32, for `c < hidden`,
/// `n < n_rows`; `out` is dense `[n_rows, hidden]`.
///
/// Grid: x = column in [0, hidden), y = row in [0, n_rows).
#define CE_GATHER_KERNEL(NAME, IN_T)                                              \
kernel void NAME(                                                                 \
    device const IN_T *h [[buffer(0)]],                                           \
    device const uint *rows [[buffer(1)]],                                        \
    device float *out [[buffer(2)]],                                              \
    constant uint &n_rows [[buffer(3)]],                                          \
    constant uint &hidden [[buffer(4)]],                                          \
    constant uint &ld [[buffer(5)]],                                              \
    constant uint &off [[buffer(6)]],                                             \
    uint2 gid [[thread_position_in_grid]])                                        \
{                                                                                 \
    const uint c = gid.x;                                                         \
    const uint n = gid.y;                                                         \
    if (c >= hidden || n >= n_rows) return;                                       \
    out[(ulong)n * hidden + c] = float(h[(ulong)rows[n] * ld + off + c]);         \
}

CE_GATHER_KERNEL(ce_gather_rows_f32, float)
CE_GATHER_KERNEL(ce_gather_rows_bf16, bfloat)

/// Classification by bit pattern. The kernels are compiled with
/// `-fmetal-math-mode=fast`, whose `ninf` / `nnan` flags let the compiler
/// assume no operand is infinite or NaN, so a float compare against an
/// infinity, `-FLT_MAX` standing in for one, or NaN may be folded, reordered or
/// answered wrongly. An integer test of the bits cannot be assumed away.
/// `-inf` is exactly `0xff800000`; `s` is zero when its bits, sign dropped, are 0.
inline bool ce_is_neg_inf(float x) { return as_type<uint>(x) == 0xff800000u; }
inline bool ce_is_zero(float x) { return (as_type<uint>(x) << 1) == 0u; }

/// Fold one chunk of logits into each row's running log-sum-exp over the
/// non-target classes.
///
/// `logits` is `[n_rows, width]` with row stride `ld`, the vocabulary columns
/// `[v0, v0 + width)`. Per row, `m` is the running maximum and `s` the running
/// `sum(exp(logit - m))` over every chunk so far, both over the columns that
/// are NOT `targets[n]` (the target is skipped wherever its chunk passes);
/// `first` (the chunk at `v0 = 0`) starts them. When `targets[n]` falls in this
/// chunk its logit is stored in `tlogit[n]`. With `L = m + log(s)`, the
/// log-sum-exp of the other classes, and `margin = tlogit - L`, the per-row
/// loss is `softplus(-margin)`. A row whose other classes are all `-inf` ends
/// with `s == 0`: its loss and gradient are 0.
///
/// Why the target is left out (Barron, "A Fix for Softmax and Cross-Entropy"):
/// folding it in makes `1 + exp(-margin)` round to 1 once the margin passes
/// the f32 cliff near 16.6, and `softmax_t - 1` is then exactly 0 while the
/// other classes' gradients stay positive.
///
/// Grid: one threadgroup per row. The identity for the running max is
/// `-FLT_MAX`, not `-INFINITY` (fast math may assume no value is infinite),
/// and `first` replaces an infinite initial `m`. `reduce_row_max` pads lanes
/// past the last simdgroup with simdgroup 0's partial, so no infinity enters
/// the reduction either. A chunk holding only the target leaves `mx` at the
/// identity and `acc` at 0, and a later chunk's rescale of that `s == 0` is 0.
kernel void ce_lse_update(
    device const float *logits [[buffer(0)]],
    device float *m [[buffer(1)]],
    device float *s [[buffer(2)]],
    device float *tlogit [[buffer(3)]],
    device const uint *targets [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &ld [[buffer(7)]],
    constant uint &v0 [[buffer(8)]],
    constant uint &first [[buffer(9)]],
    uint n [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float scratch[2 * REDUCE_MAX_SIMDGROUPS];
    if (n >= n_rows) return;  // uniform: one threadgroup per row
    device const float *row = logits + (ulong)n * ld;
    const uint t = targets[n];

    float mx = -FLT_MAX;
    for (uint c = lid; c < width; c += tptg) {
        if (v0 + c != t) {
            mx = max(mx, row[c]);
        }
    }
    mx = reduce_row_max(mx, scratch, sgid, lane, tptg);
    const float m_old = first != 0u ? mx : m[n];
    const float m_new = max(m_old, mx);

    float acc = 0.0f;
    for (uint c = lid; c < width; c += tptg) {
        // A `-inf` logit contributes exactly 0. It is skipped, by its bit
        // pattern, instead of being fed to `exp`: with m_new at the identity
        // (a row, or chunk, with no finite non-target logit) `-inf - m_new`
        // is `-inf`, or `-inf - -inf` = NaN if `m_new` is itself `-inf`, and
        // `precise::exp` is compiled under fast math, where an infinite
        // argument is not something it must handle. NaN is not skipped: it
        // reaches `exp`, poisons `s`, and the host reports the row as not
        // finite. A float compare (`row[c] > -FLT_MAX`) cannot do this job:
        // fast math may fold it, and it is false for NaN, dropping the NaN.
        if (v0 + c != t && !ce_is_neg_inf(row[c])) {
            acc += precise::exp(row[c] - m_new);
        }
    }
    // A second scratch region: the max's reads of `scratch` are not fenced
    // from these writes (see reduce_row_add's contract).
    acc = reduce_row_add(acc, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg);

    if (lid == 0u) {
        // `s == 0` (no finite non-target logit yet) has nothing to rescale;
        // multiplying it by `exp(m_old - m_new)` would be 0 * NaN if both
        // maxima sit at the identity.
        const float s_old = first != 0u ? 0.0f : s[n];
        const float carried = ce_is_zero(s_old) ? 0.0f : s_old * precise::exp(m_old - m_new);
        s[n] = carried + acc;
        m[n] = m_new;
        if (t >= v0 && t - v0 < width) {
            tlogit[n] = row[t - v0];
        }
    }
}

/// The gradient of `scale * loss` with respect to this chunk's logits, in
/// place, from the completed first walk: with `L = m + log(s)` (the
/// log-sum-exp of the non-target classes) and `q = sigmoid(L - tlogit)`,
///   non-target `j`:  `exp(logit_j - L) * q * scale`
///   the target:      `-q * scale`.
/// The non-target terms sum to `q` (`sum_j exp(logit_j - L) = 1`), so the row
/// sums to 0. `-q` is the exact value of the target's gradient and does not
/// depend on how the vocabulary was chunked, where a sum of the computed
/// non-target terms would carry their rounding and need a cross-chunk
/// reduction; unlike `softmax_t - 1` it does not cancel to 0 past the f32
/// cliff, because `q` is formed without the `1 + exp(-margin)` that rounds.
/// `q` is evaluated from `exp(-|margin|)`, so it neither overflows nor loses
/// its small side. A row with `s == 0` (every other class `-inf`) has zero
/// gradient.
///
/// Grid: x = column in [0, width), y = row in [0, n_rows).
kernel void ce_softmax_grad(
    device float *logits [[buffer(0)]],
    device const float *m [[buffer(1)]],
    device const float *s [[buffer(2)]],
    device const uint *targets [[buffer(3)]],
    constant uint &n_rows [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &ld [[buffer(6)]],
    constant uint &v0 [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    device const float *tlogit [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint n = gid.y;
    if (c >= width || n >= n_rows) return;
    const ulong i = (ulong)n * ld + c;
    const float sn = s[n];
    if (ce_is_zero(sn)) {
        logits[i] = 0.0f;
        return;
    }
    const float lo = m[n] + precise::log(sn);
    const float margin = tlogit[n] - lo;
    const float e = precise::exp(-fabs(margin));
    const float q = margin >= 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
    // A `-inf` non-target logit has gradient 0; `exp` is not handed `-inf`
    // (see ce_lse_update).
    const float z = logits[i];
    const float g = (v0 + c == targets[n]) ? -q : (ce_is_neg_inf(z) ? 0.0f : precise::exp(z - lo) * q);
    logits[i] = g * scale;
}
