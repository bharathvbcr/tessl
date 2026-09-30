// Cross-entropy over a vocabulary too large to hold as logits, for the rows
// that are supervised.
//
// Fine-tuning a tied-embedding LM on answers supervises about one position
// per sequence, and full-vocabulary logits for every position are what ran a
// 2B model out of memory on MPS. `tessl::cross_entropy` therefore gathers the
// supervised rows of the hidden states (`ce_gather_rows_*`), then walks the
// vocabulary in chunks: each chunk's logits come from a GEMM against that
// slice of the weight, `ce_lse_update` folds them into a running log-sum-exp
// per row (and picks up the target's logit when its chunk passes), and on the
// second walk `ce_softmax_grad` turns the recomputed chunk into
// `(softmax - onehot) * scale` in place for the gradient GEMMs. Nothing larger
// than `[rows, chunk]` logits ever exists.
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

/// Fold one chunk of logits into each row's running log-sum-exp.
///
/// `logits` is `[n_rows, width]` with row stride `ld`, the vocabulary columns
/// `[v0, v0 + width)`. Per row, `m` is the running maximum and `s` the running
/// `sum(exp(logit - m))` over every chunk so far; `first` (the chunk at
/// `v0 = 0`) starts them. When `targets[n]` falls in this chunk its logit is
/// stored in `tlogit[n]`. The per-row loss is `m + log(s) - tlogit`.
///
/// Grid: one threadgroup per row. The identity for the running max is
/// `-FLT_MAX`, not `-INFINITY`, and `first` replaces an infinite initial `m`,
/// so no step depends on infinities surviving fast math.
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

    float mx = -FLT_MAX;
    for (uint c = lid; c < width; c += tptg) {
        mx = max(mx, row[c]);
    }
    mx = reduce_row_max(mx, scratch, sgid, lane, tptg);
    const float m_old = first != 0u ? mx : m[n];
    const float m_new = max(m_old, mx);

    float acc = 0.0f;
    for (uint c = lid; c < width; c += tptg) {
        acc += precise::exp(row[c] - m_new);
    }
    // A second scratch region: the max's reads of `scratch` are not fenced
    // from these writes (see reduce_row_add's contract).
    acc = reduce_row_add(acc, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg);

    if (lid == 0u) {
        const float carried = first != 0u ? 0.0f : s[n] * precise::exp(m_old - m_new);
        s[n] = carried + acc;
        m[n] = m_new;
        const uint t = targets[n];
        if (t >= v0 && t - v0 < width) {
            tlogit[n] = row[t - v0];
        }
    }
}

/// `logits[n, c] = (exp(logits[n, c] - lse[n]) - [v0 + c == targets[n]]) * scale`
/// in place, with `lse = m + log(s)` from the completed first walk: the
/// gradient of the loss with respect to this chunk's logits.
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
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint n = gid.y;
    if (c >= width || n >= n_rows) return;
    const float lse = m[n] + precise::log(s[n]);
    const ulong i = (ulong)n * ld + c;
    float p = precise::exp(logits[i] - lse);
    if (v0 + c == targets[n]) {
        p -= 1.0f;
    }
    logits[i] = p * scale;
}
