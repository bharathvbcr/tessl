// Residual / hidden RMSNorm (f32 and bf16).
//
// One threadgroup per row, with the sum of squares reduced as a tree in
// threadgroup memory. Lane `lid` visits `dim[lid], dim[lid + tptg], ...`, so
// `dim` has no ceiling and adjacent lanes read adjacent addresses.
//
// These kernels were one *thread* per row until 2026-08-31, which meant the
// parallelism available was `rows` and each thread walked its row serially
// twice. Measured on an M5 Pro that peaked at 87 GB/s against the 243 GB/s the
// reductions in `reduce.metal` reach on identical traffic, and at the decode
// shape `rows = 1` the entire kernel ran on a single GPU thread: 404 us to move
// 32 KB. RMSNorm runs twice per transformer layer on every token, so that was
// the hot path.
//
// The reduction reassociates where the serial loop did not. That is a real
// change in the low bits — see `rms_norm_f32_matches_cpu_reference`, which
// compares against an f64 reference rather than a matching f32 accumulation
// precisely so it measures error instead of agreeing with itself.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// Sum of squares of `xin[0..dim]`, reduced across the threadgroup.
///
/// Leaves the total in every lane via `scratch[0]`. Callers must not reuse
/// `scratch` until after the barrier this returns past.
inline float row_sum_squares(
    device const float *xin,
    threadgroup float *scratch,
    uint dim,
    uint lid,
    uint tptg)
{
    float ss = 0.0f;
    for (uint d = lid; d < dim; d += tptg) {
        float v = xin[d];
        ss += v * v;
    }
    scratch[lid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    REDUCE_TREE(scratch, tptg, lid, reduce_add)
    return scratch[0];
}

/// out[row, :] = rms_norm(x[row, :], weight[:], eps)
kernel void rms_norm_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float *xin = x + (ulong)row * dim;
    device float *xout = out + (ulong)row * dim;

    // `eps` is what keeps an all-zero row finite: rsqrt(0) is inf, and the row
    // would leave here as inf or NaN without it.
    const float inv = rsqrt(row_sum_squares(xin, scratch, dim, lid, tptg) / (float)dim + eps);
    for (uint d = lid; d < dim; d += tptg) {
        xout[d] = xin[d] * inv * weight[d];
    }
}

/// Same math as rms_norm_f32, writing bf16 (for simd GEMV; kills cast pass).
kernel void rms_norm_bf16(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float *xin = x + (ulong)row * dim;
    device bfloat *xout = out + (ulong)row * dim;

    const float inv = rsqrt(row_sum_squares(xin, scratch, dim, lid, tptg) / (float)dim + eps);
    for (uint d = lid; d < dim; d += tptg) {
        xout[d] = (bfloat)(xin[d] * inv * weight[d]);
    }
}

/// When `layer_scale != 1`, folds end-of-layer `x *= scale`:
/// `resid = scale * (resid + rms_norm(x)*weight)`.
kernel void rms_norm_residual_add_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *resid [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant float &layer_scale [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_TG];
    device const float *xin = x + (ulong)row * dim;
    device float *xout = resid + (ulong)row * dim;

    const float inv = rsqrt(row_sum_squares(xin, scratch, dim, lid, tptg) / (float)dim + eps);
    const float s = layer_scale;
    if (s == 1.0f) {
        for (uint d = lid; d < dim; d += tptg) {
            xout[d] += xin[d] * inv * weight[d];
        }
    } else {
        for (uint d = lid; d < dim; d += tptg) {
            float h = xin[d] * inv * weight[d];
            xout[d] = s * (xout[d] + h);
        }
    }
}
