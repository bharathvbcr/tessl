// Logit softcap (default 30) + greedy argmax.
// softcap: y = softcap * tanh(x / softcap)
#include <metal_stdlib>
#include "softcap.h"
using namespace metal;

/// Index of an argmax lane holding no logit: past `n`, non-finite, or a
/// partial from a group that had none. The host refuses it as a result.
///
/// A lane's validity is carried by its index, not by a non-finite value:
/// kernels compile with fast math, which may assume no value is infinite or
/// NaN, so `-INFINITY` / `NAN` lane markers are not values the compiler has
/// to preserve. Such a lane's value is `-FLT_MAX`, and it never wins a fold,
/// so a real logit equal to `-FLT_MAX` is still found.
constant uint ARGMAX_NONE = 0xFFFFFFFFu;

/// Fold lane `b` of the threadgroup argmax into lane `a`: a lane holding a
/// logit beats one holding none, then the larger value wins, then the lower
/// vocabulary index.
inline void argmax_fold(threadgroup float *val, threadgroup uint *idx, uint a, uint b) {
    const uint ib = idx[b];
    if (ib == ARGMAX_NONE) return;
    const uint ia = idx[a];
    if (ia == ARGMAX_NONE || val[b] > val[a] || (val[b] == val[a] && ib < ia)) {
        val[a] = val[b];
        idx[a] = ib;
    }
}

/// In-place softcap over logits[0..n).
/// `softcap` from stable device f32 (ICB / encode-once — not const-arena).
kernel void softcap_logits(
    device float *logits [[buffer(0)]],
    device const float *softcap_ptr [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const float softcap = *softcap_ptr;
    if (gid >= n) return;
    logits[gid] = tessl_apply_softcap(logits[gid], softcap);
}

/// Hierarchical argmax: one threadgroup reduces a slice, writes (max, idx) pairs.
/// When `has_idx_in != 0`, `idx_in[i]` carries the original vocab index through
/// subsequent reduce passes (no host remap).
/// When `softcap > 0` and this is the first pass (`has_idx_in == 0`), apply
/// softcap to logits[i] before compare. Later passes reduce partial maxima that
/// are already capped, and tanh is not idempotent: capping again would shrink
/// every value the first pass produced.
/// A group with no finite logit writes `(-FLT_MAX, ARGMAX_NONE)`, which a later
/// pass reads as a lane holding none.
/// `softcap` from stable device f32 (ICB freeze).
kernel void argmax_f32(
    device const float *logits [[buffer(0)]],
    device uint *out_idx [[buffer(1)]],
    device float *out_val [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device const uint *idx_in [[buffer(4)]],
    constant uint &has_idx_in [[buffer(5)]],
    device const float *softcap_ptr [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgpig [[threadgroup_position_in_grid]])
{
    threadgroup float tg_val[256];
    threadgroup uint tg_idx[256];
    const float softcap = *softcap_ptr;

    const uint base = tgpig * tptg;
    const uint i = base + lid;
    float v = -FLT_MAX;
    uint idx = ARGMAX_NONE;
    if (i < n) {
        const float raw = logits[i];
        const uint from = (has_idx_in != 0u) ? idx_in[i] : i;
        if (isfinite(raw) && from != ARGMAX_NONE) {
            v = (has_idx_in == 0u && softcap > 0.0f) ? tessl_apply_softcap(raw, softcap) : raw;
            idx = from;
        }
    }
    tg_val[lid] = v;
    tg_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            argmax_fold(tg_val, tg_idx, lid, lid + stride);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out_idx[tgpig] = tg_idx[0];
        out_val[tgpig] = tg_val[0];
    }
}

/// Fused softcap + single-TG argmax for n <= 256 (unit tests / small heads).
/// `softcap` from stable device f32 (ICB freeze).
kernel void softcap_sample(
    device float *logits [[buffer(0)]],
    device uint *out_token [[buffer(1)]],
    device const float *softcap_ptr [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float tg_val[256];
    threadgroup uint tg_idx[256];
    const float softcap = *softcap_ptr;

    tg_val[lid] = -FLT_MAX;
    tg_idx[lid] = ARGMAX_NONE;
    if (lid < n) {
        const float raw = logits[lid];
        if (isfinite(raw)) {
            const float sc = tessl_apply_softcap(raw, softcap);
            logits[lid] = sc;
            tg_val[lid] = sc;
            tg_idx[lid] = lid;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            argmax_fold(tg_val, tg_idx, lid, lid + stride);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out_token[0] = tg_idx[0];
    }
}

/// Single-pass softcap + argmax for large vocab (e.g. 262144).
/// One threadgroup; each lane scans a strided slice then TG-reduces.
/// Does NOT rewrite logits in place (decode only needs the index).
/// `softcap` from stable device f32 (ICB / encode-once).
kernel void softcap_argmax_one_pass(
    device const float *logits [[buffer(0)]],
    device uint *out_token [[buffer(1)]],
    device const float *softcap_ptr [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    const float softcap = *softcap_ptr;

    float best = -FLT_MAX;
    uint best_i = ARGMAX_NONE;
    for (ulong i = lid; i < (ulong)n; i += tptg) {
        float raw = logits[i];
        if (!isfinite(raw)) continue;
        float v = raw;
        if (softcap > 0.0f) {
            v = tessl_apply_softcap(v, softcap);
        }
        if (!isfinite(v)) continue;
        // A lane visits ascending indices, so an equal later value never
        // replaces the one it has.
        if (best_i == ARGMAX_NONE || v > best) {
            best = v;
            best_i = (uint)i;
        }
    }
    tg_val[lid] = best;
    tg_idx[lid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            argmax_fold(tg_val, tg_idx, lid, lid + stride);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out_token[0] = tg_idx[0];
    }
}
