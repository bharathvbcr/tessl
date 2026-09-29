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

/// A quiet NaN's bits. Metal compiles with fast math, under which a NaN that
/// flows through float arithmetic *or a float select* may be assumed away, so
/// results are chosen and stored as integer bits (`store_or_nan`).
constant uint SCORE_NAN_BITS = 0x7fc00000u;

inline void store_or_nan(device float *dst, ulong i, bool ok, float v)
{
    ((device uint *)dst)[i] = ok ? as_type<uint>(v) : SCORE_NAN_BITS;
}

/// Whether answer `a` can be scored: the slot row exists and the token id is
/// inside the LM head. Integer comparisons only, so fast math has nothing to
/// fold.
inline bool answer_ok(device const uint *answers, uint a, bool row_ok, uint vocab)
{
    return row_ok && answers[a] < vocab;
}

/// Shared body. One threadgroup per slot; `tgm` holds `REDUCE_MAX_SIMDGROUPS`
/// reduction partials, then `n_ans` logits.
///
/// A slot row `>= rows` or an answer id `>= vocab` scores NaN instead of
/// reading out of bounds: those indices live in device memory, so the host
/// cannot check them without a synchronising read. Validity is re-derived
/// from the indices (`answer_ok`) rather than tested for as NaN, because fast
/// math may fold `isnan` away, and rather than stored, which would double the
/// threadgroup memory and cap the answer count at half: an invalid answer is left out of its slot's softmax, so the
/// valid answers beside it keep their log-probabilities, and gets NaN (as
/// stored bits) in both outputs. A bad slot row makes the whole row NaN.
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
        const bool ok = answer_ok(answers, a, row_ok, vocab);
        device const W *e = emb + (ulong)(ok ? answers[a] : 0u) * hidden;
        float dot = 0.0f;
        for (ulong d = lane; d < (ulong)hidden; d += 32u) {
            dot += h[d] * (norm_w[d] + w_offset) * (float)e[d];
        }
        dot = simd_sum(dot);
        if (lane == 0u) {
            lg[a] = ok ? dot * inv : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // n_ans is small (the host caps it); every thread folds it serially rather
    // than paying two more reductions for a few dozen elements.
    // Seeded from the first valid logit rather than -INFINITY, which fast
    // math may treat as undefined. No valid answer leaves `any` false.
    bool any = false;
    float mx = 0.0f;
    for (uint a = 0u; a < n_ans; ++a) {
        if (answer_ok(answers, a, row_ok, vocab)) {
            mx = any ? max(mx, lg[a]) : lg[a];
            any = true;
        }
    }
    float sum = 0.0f;
    for (uint a = 0u; a < n_ans; ++a) {
        if (answer_ok(answers, a, row_ok, vocab)) {
            sum += exp(lg[a] - mx);
        }
    }
    // sum >= 1 whenever `any`: the maximum contributes exp(0).
    const float lse = mx + log(max(sum, 1.0f));
    for (uint a = lid; a < n_ans; a += tptg) {
        const bool ok = answer_ok(answers, a, row_ok, vocab);
        store_or_nan(logits, (ulong)slot * n_ans + a, ok, lg[a]);
        store_or_nan(logprobs, (ulong)slot * n_ans + a, ok, lg[a] - lse);
    }
}

/// Threadgroup memory: `REDUCE_MAX_SIMDGROUPS + n_ans` floats.
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

/// The embedding gather, from the same bf16 vocabulary table the scoring
/// kernels read as the (tied) LM head: `out[r, :] = f32(table[ids[r], :])`.
///
/// Token ids live in device memory, so the host cannot check them: an id
/// `>= vocab` makes its row NaN and leaves the others intact. The widening is
/// integer-only (a bf16 is the top half of an f32), so it is exact and fast
/// math has nothing to fold.
///
/// Grid: x = column in [0, hidden), y = row in [0, n).
kernel void qwen35_embed_rows_bf16(
    device const uint *ids [[buffer(0)]],
    device const ushort *table [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &hidden [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= hidden || r >= n) return;
    const uint id = ids[r];
    const bool ok = id < vocab;
    const uint bits = (uint)table[(ulong)(ok ? id : 0u) * hidden + col] << 16;
    ((device uint *)out)[(ulong)r * hidden + col] = ok ? bits : SCORE_NAN_BITS;
}
