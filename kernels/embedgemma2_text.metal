// EmbeddingGemma 2 text encoder: the row passes that sit between its GEMMs,
// fused so a layer launches (and moves) less.
//
// Every kernel here is the composition of kernels the encoder used to launch
// one after the other, with each element computed by the same expression in
// the same order, so the exact-f32 forward stays bit for bit what it was:
//
//   * the reduction of a row is `rms_norm.metal`'s: lane `lid` adds
//     `dim[lid], dim[lid + tptg], ...` in ascending order into its own
//     accumulator, and `reduce_row_add` (`reduce_tree.h`) folds the lanes. The
//     host launches these kernels with the threadgroup size `rms_norm_f32`
//     would use for the same `dim` (`nn::reduce_tptg` on a pipeline whose
//     limit is the same 1024), so a row's partial sums are the same sums;
//   * a stage that used to store a value and a later kernel that read it back
//     now keep the value for the next stage in the device buffer the earlier
//     kernel wrote (each lane re-reads only the elements it wrote, in the same
//     lane order, so no barrier separates the two), which keeps the compiler
//     from re-associating a product across what was a kernel boundary;
//   * the per-element expressions are copied from the kernels they replace:
//     `xin * inv * weight` added to, or (with a layer scale) scaled with, the
//     residual, exactly as `rms_norm_residual_add_f32` writes it.
//
// `tests/embedgemma2_kernels.rs` holds each kernel to the unfused composition
// of the kernels it replaces, bit for bit, at several widths and row counts.
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// Sum of squares of `xin[0..dim]` over this lane's elements, folded across the
/// threadgroup (`rms_norm.metal`'s `row_sum_squares`).
inline float eg2t_row_sum_squares(
    device const float *xin,
    threadgroup float *scratch,
    uint dim,
    uint lid,
    uint sgid,
    uint lane,
    uint tptg)
{
    float ss = 0.0f;
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        float v = xin[d];
        ss += v * v;
    }
    return reduce_row_add(ss, scratch, sgid, lane, tptg);
}

/// The post-norm residual add of one row:
/// `resid = layer_scale * (resid + rms_norm(y) * w)` (`layer_scale == 1` is the
/// plain add), returning the sum of squares of the stored row accumulated by
/// this lane (before the cross-lane fold). `rms_norm_residual_add_f32`'s
/// arithmetic, with the next norm's accumulation riding the same loop.
inline float eg2t_resid_add_row(
    device const float *yin,
    device const float *w,
    device float *rout,
    float inv,
    float s,
    uint dim,
    uint lid,
    uint tptg)
{
    float ss = 0.0f;
    if (s == 1.0f) {
        for (ulong d = lid; d < (ulong)dim; d += tptg) {
            rout[d] += yin[d] * inv * w[d];
            const float v = rout[d];
            ss += v * v;
        }
    } else {
        for (ulong d = lid; d < (ulong)dim; d += tptg) {
            float h = yin[d] * inv * w[d];
            rout[d] = s * (rout[d] + h);
            const float v = rout[d];
            ss += v * v;
        }
    }
    return ss;
}

/// `resid[r, :] = layer_scale * (resid[r, :] + rms_norm(y[r, :]) * w_post)` in
/// place, then `out[r, :] = rms_norm(resid[r, :]) * w_next`:
/// `rms_norm_residual_add_f32` and the `rms_norm_f32` after it in one launch.
/// The layer's residual add and the next norm (the pre-feed-forward norm, or
/// the next layer's input norm, or the final norm) share the row, so the row
/// is read once for both statistics and written once per output.
#define EG2T_RESID_NORM_KERNEL(NAME, OUT_T)                                       \
kernel void NAME(                                                                 \
    device const float *y [[buffer(0)]],                                          \
    device const float *w_post [[buffer(1)]],                                     \
    device float *resid [[buffer(2)]],                                            \
    device const float *w_next [[buffer(3)]],                                     \
    device OUT_T *out [[buffer(4)]],                                              \
    constant uint &rows [[buffer(5)]],                                            \
    constant uint &dim [[buffer(6)]],                                             \
    constant float &eps [[buffer(7)]],                                            \
    constant float &layer_scale [[buffer(8)]],                                    \
    uint row [[threadgroup_position_in_grid]],                                    \
    uint lid [[thread_position_in_threadgroup]],                                  \
    uint sgid [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint tptg [[threads_per_threadgroup]])                                        \
{                                                                                 \
    if (row >= rows) return;                                                      \
    threadgroup float scratch[2u * REDUCE_MAX_SIMDGROUPS];                        \
    device const float *yin = y + (ulong)row * dim;                               \
    device float *rrow = resid + (ulong)row * dim;                                \
    device OUT_T *orow = out + (ulong)row * dim;                                  \
    const float inv = rsqrt(                                                      \
        eg2t_row_sum_squares(yin, scratch, dim, lid, sgid, lane, tptg) / (float)dim + eps); \
    const float ss = eg2t_resid_add_row(yin, w_post, rrow, inv, layer_scale, dim, lid, tptg); \
    const float inv2 = rsqrt(                                                     \
        reduce_row_add(ss, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg) / (float)dim + eps); \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        orow[d] = (OUT_T)(rrow[d] * inv2 * w_next[d]);                            \
    }                                                                             \
}

EG2T_RESID_NORM_KERNEL(eg2t_resid_norm_f32, float)
EG2T_RESID_NORM_KERNEL(eg2t_resid_norm_bf16, bfloat)

/// `resid[r, :] = layer_scale * (resid[r, :] + rms_norm(y[r, :]) * w_post)` in
/// place, and `copy[r, :] = bf16(resid[r, :])`: the post-feed-forward add and the
/// rounded copy the bf16 forward's per-layer-input gate reads (it was
/// `rms_norm_residual_add_f32` and `cast_f32_to_bf16`).
kernel void eg2t_resid_add_bf16_copy(
    device const float *y [[buffer(0)]],
    device const float *w_post [[buffer(1)]],
    device float *resid [[buffer(2)]],
    device bfloat *copy [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    constant float &layer_scale [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    device const float *yin = y + (ulong)row * dim;
    device float *rrow = resid + (ulong)row * dim;
    device bfloat *crow = copy + (ulong)row * dim;
    const float inv = rsqrt(eg2t_row_sum_squares(yin, scratch, dim, lid, sgid, lane, tptg) / (float)dim + eps);
    const float s = layer_scale;
    if (s == 1.0f) {
        for (ulong d = lid; d < (ulong)dim; d += tptg) {
            rrow[d] += yin[d] * inv * w_post[d];
            crow[d] = (bfloat)rrow[d];
        }
    } else {
        for (ulong d = lid; d < (ulong)dim; d += tptg) {
            float h = yin[d] * inv * w_post[d];
            rrow[d] = s * (rrow[d] + h);
            crow[d] = (bfloat)rrow[d];
        }
    }
}

/// In place over `rows` rows of `dim`: `x[r, :] = rms_norm(x[r, :] * scale) * w`.
/// The per-layer inputs' `scale_f32_inplace` and `rms_norm_f32` in one launch:
/// the scaled row is stored first (as the scale pass stored it), then the norm
/// reads it back, so the product is rounded to f32 before the reduction and
/// before the normalization, as in the two launches.
kernel void eg2t_scale_rms_norm_f32(
    device float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant float &scale [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    device float *xrow = x + (ulong)row * dim;
    float ss = 0.0f;
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        xrow[d] *= scale;
        const float v = xrow[d];
        ss += v * v;
    }
    const float inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + eps);
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        xrow[d] = xrow[d] * inv * w[d];
    }
}
