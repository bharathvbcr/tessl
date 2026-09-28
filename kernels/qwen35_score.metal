// Score only the answer rows at the slot positions.
//
// A full-vocabulary head at Qwen3.5's 248,320 tokens is 27% of a forward pass's
// FLOPs and a 1-2 GB fp32 logits tensor at prefill length, and a caller that
// only compares a fixed set of answer tokens (17, say) at a handful of slot
// positions reads almost none of it. This computes exactly what is read:
//
//   h      = final_norm(hidden[slot])            (RMSNorm, weight + w_offset)
//   logit  = h . E[answer]                       for each answer token
//   logp   = log_softmax over the answer set only
//
// `w_offset` is 1.0 for Qwen3.5's zero-centred `Qwen3_5RMSNorm` and 0.0 for a
// plain `* w` norm. `logp` is the distribution *restricted to the answer set*;
// it is not the full-vocabulary log-probability, which needs every row.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// A quiet NaN, from its bits. Metal compiles with fast math, under which a
/// NaN produced by arithmetic may be assumed away; stored bits are not.
inline float score_nan()
{
    return as_type<float>(0x7fc00000u);
}

/// Shared body. One threadgroup per slot; `tgm` holds `REDUCE_MAX_SIMDGROUPS`
/// reduction partials followed by `n_ans` logits.
///
/// A slot row `>= rows` or an answer id `>= vocab` scores NaN instead of
/// reading out of bounds: those indices live in device memory, so the host
/// cannot check them without a synchronising read.
template <typename W>
inline void score_rows_impl(
    device const float *hidden_states,
    device const uint *slots,
    device const float *norm_w,
    device const W *emb,
    device const uint *answers,
    device float *logits,
    device float *logprobs,
    uint rows,
    uint hidden,
    uint n_ans,
    uint vocab,
    float eps,
    float w_offset,
    threadgroup float *tgm,
    uint slot,
    uint lid,
    uint sg,
    uint lane,
    uint tptg)
{
    threadgroup float *scratch = tgm;
    threadgroup float *lg = tgm + REDUCE_MAX_SIMDGROUPS;
    const uint row = slots[slot];
    // Uniform: every thread of the threadgroup reads the same slot.
    const bool row_ok = row < rows;
    device const float *h = hidden_states + (ulong)(row_ok ? row : 0u) * hidden;

    float ss = 0.0f;
    for (ulong d = lid; d < (ulong)hidden; d += tptg) {
        ss += h[d] * h[d];
    }
    const float inv = rsqrt(reduce_row_add(ss, scratch, sg, lane, tptg) / (float)hidden + eps);

    const uint n_sg = (tptg + 31u) / 32u;
    for (uint a = sg; a < n_ans; a += n_sg) {
        const uint tok = answers[a];
        const bool ok = row_ok && tok < vocab;
        device const W *e = emb + (ulong)(ok ? tok : 0u) * hidden;
        float dot = 0.0f;
        for (ulong d = lane; d < (ulong)hidden; d += 32u) {
            dot += h[d] * (norm_w[d] + w_offset) * (float)e[d];
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            lg[a] = ok ? dot * inv : score_nan();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // n_ans is small (the host caps it); every thread folds it serially rather
    // than paying two more reductions for a few dozen elements.
    float mx = lg[0];
    for (uint a = 1u; a < n_ans; ++a) {
        mx = max(mx, lg[a]);
    }
    float sum = 0.0f;
    for (uint a = 0u; a < n_ans; ++a) {
        sum += exp(lg[a] - mx);
    }
    const float lse = mx + log(sum);
    for (uint a = lid; a < n_ans; a += tptg) {
        logits[(ulong)slot * n_ans + a] = lg[a];
        logprobs[(ulong)slot * n_ans + a] = lg[a] - lse;
    }
}

/// `emb` is the LM head, [vocab, hidden] row-major (the tied embedding for the
/// small Qwen3.5 checkpoints). Outputs are [n_slots, n_ans]. Grid: one
/// threadgroup per slot.
#define SCORE_ROWS_KERNEL(NAME, W_T)                                              \
kernel void NAME(                                                                 \
    device const float *hidden_states [[buffer(0)]],                              \
    device const uint *slots [[buffer(1)]],                                       \
    device const float *norm_w [[buffer(2)]],                                     \
    device const W_T *emb [[buffer(3)]],                                          \
    device const uint *answers [[buffer(4)]],                                     \
    device float *logits [[buffer(5)]],                                           \
    device float *logprobs [[buffer(6)]],                                         \
    constant uint &rows [[buffer(7)]],                                            \
    constant uint &hidden [[buffer(8)]],                                          \
    constant uint &n_ans [[buffer(9)]],                                           \
    constant uint &vocab [[buffer(10)]],                                          \
    constant float &eps [[buffer(11)]],                                           \
    constant float &w_offset [[buffer(12)]],                                      \
    threadgroup float *tgm [[threadgroup(0)]],                                    \
    uint slot [[threadgroup_position_in_grid]],                                   \
    uint lid [[thread_index_in_threadgroup]],                                     \
    uint sg [[simdgroup_index_in_threadgroup]],                                   \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint tptg [[threads_per_threadgroup]])                                        \
{                                                                                 \
    score_rows_impl<W_T>(hidden_states, slots, norm_w, emb, answers, logits,     \
                         logprobs, rows, hidden, n_ans, vocab, eps, w_offset,     \
                         tgm, slot, lid, sg, lane, tptg);                         \
}

SCORE_ROWS_KERNEL(qwen35_score_rows_f32, float)
SCORE_ROWS_KERNEL(qwen35_score_rows_bf16, bfloat)
