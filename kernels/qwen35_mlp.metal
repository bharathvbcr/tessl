// Qwen3.5 MLP activation: `act_fn(gate_proj(x)) * up_proj(x)` with
// `act_fn = silu`, as transformers' `Qwen3_5MLP.forward` computes it before
// `down_proj`.
//
// tessl's generic `mlp_silu` writes f32, and the down GEMM reads bf16, so a
// forward used to spend a second pass casting the f32 product: at Qwen3.5-2B's
// shapes (intermediate 6144) that cast was 0.18 ms of the MLP's 0.52 ms of
// elementwise work per layer at T = 1024. `qwen35_swiglu_bf16` stores the
// product as bf16 directly, in the one pass.
//
// Both operands are column windows of wider rows (`ld`, `off`), like every
// Qwen3.5 kernel's inputs, so a single GEMM that writes `[gate | up]` side by
// side can feed this in place.
#include <metal_stdlib>
#include "gelu.h"
#include "qwen35_act.h"
#include "reduce_tree.h"
using namespace metal;

/// `out[r, out_off + c] = ACT(gate[r, gate_off + c]) * up[r, up_off + c]`
/// for `c < width`, `r < rows`; rows are `ld_*` elements apart. `ACT` is SiLU
/// (`qwen35_swiglu_*`, Qwen3.5) or GELU-tanh (`qwen35_gelu_tanh_glu_*`,
/// EmbeddingGemma 2: the same `tessl_gelu_pytorch_tanh` as `mlp_gelu_tanh`).
///
/// Grid: x = column in [0, width), y = row in [0, rows).
#define GATED_KERNEL(NAME, OUT_T, ACT)                                            \
kernel void NAME(                                                                 \
    device const float *gate [[buffer(0)]],                                       \
    device const float *up [[buffer(1)]],                                         \
    device OUT_T *out [[buffer(2)]],                                              \
    constant uint &rows [[buffer(3)]],                                            \
    constant uint &width [[buffer(4)]],                                           \
    constant uint &ld_gate [[buffer(5)]],                                         \
    constant uint &gate_off [[buffer(6)]],                                        \
    constant uint &ld_up [[buffer(7)]],                                           \
    constant uint &up_off [[buffer(8)]],                                          \
    constant uint &ld_out [[buffer(9)]],                                          \
    constant uint &out_off [[buffer(10)]],                                        \
    uint2 gid [[thread_position_in_grid]])                                        \
{                                                                                 \
    const uint col = gid.x;                                                       \
    const uint r = gid.y;                                                         \
    if (col >= width || r >= rows) return;                                        \
    const float g = gate[(ulong)r * ld_gate + gate_off + col];                    \
    const float u = up[(ulong)r * ld_up + up_off + col];                          \
    out[(ulong)r * ld_out + out_off + col] = (OUT_T)(ACT(g) * u);             \
}

GATED_KERNEL(qwen35_swiglu_f32, float, qwen35_silu)
GATED_KERNEL(qwen35_swiglu_bf16, bfloat, qwen35_silu)
GATED_KERNEL(qwen35_gelu_tanh_glu_f32, float, tessl_gelu_pytorch_tanh)
GATED_KERNEL(qwen35_gelu_tanh_glu_bf16, bfloat, tessl_gelu_pytorch_tanh)

/// `resid[r, resid_off + c] += y[r, y_off + c]` for `c < width`, `r < rows`:
/// the residual add after a projection, for the exact-f32 forward. (The bf16
/// forward folds this into the GEMM epilogue, which exact f32 does not have.)
/// Each element is read and written by one thread, so `resid` is updated in
/// place; `y` must be a different buffer.
///
/// Grid: x = column in [0, width), y = row in [0, rows).
kernel void qwen35_residual_add_f32(
    device const float *y [[buffer(0)]],
    device float *resid [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &ld_y [[buffer(4)]],
    constant uint &y_off [[buffer(5)]],
    constant uint &ld_resid [[buffer(6)]],
    constant uint &resid_off [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint r = gid.y;
    if (col >= width || r >= rows) return;
    resid[(ulong)r * ld_resid + resid_off + col] += y[(ulong)r * ld_y + y_off + col];
}

/// Qwen3.5's zero-centred residual RMSNorm (`Qwen3_5RMSNorm`):
/// `out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * (1 + w[:])`, with
/// `w` stored as the checkpoint holds it and `1 + w` formed here in f32, as
/// transformers forms `1.0 + self.weight.float()`. Storing `w` (not `1 + w`)
/// keeps every write of it exact; `1 + w` rounds `w` to ulp(1 + w).
///
/// The sum of squares is `rms_norm.metal`'s, lane for lane: lane `lid` walks
/// `dim[lid], dim[lid + tptg], ...` and `reduce_row_add` folds the lanes, so
/// with the host's `reduce_tptg` the rows' scale is the one tessl's generic
/// RMSNorm computes.
///
/// Grid: one threadgroup per row.
#define QWEN35_RMS_NORM_KERNEL(NAME, OUT_T)                                       \
kernel void NAME(                                                                 \
    device const float *x [[buffer(0)]],                                          \
    device const float *weight [[buffer(1)]],                                     \
    device OUT_T *out [[buffer(2)]],                                              \
    constant uint &rows [[buffer(3)]],                                            \
    constant uint &dim [[buffer(4)]],                                             \
    constant float &eps [[buffer(5)]],                                            \
    uint row [[threadgroup_position_in_grid]],                                    \
    uint lid [[thread_position_in_threadgroup]],                                  \
    uint sgid [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint tptg [[threads_per_threadgroup]])                                        \
{                                                                                 \
    if (row >= rows) return;                                                      \
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];                             \
    device const float *xin = x + (ulong)row * dim;                               \
    device OUT_T *xout = out + (ulong)row * dim;                                  \
    float ss = 0.0f;                                                              \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        const float v = xin[d];                                                   \
        ss += v * v;                                                              \
    }                                                                             \
    const float inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + eps); \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        const float wp = 1.0f + weight[d];                                        \
        xout[d] = (OUT_T)(xin[d] * inv * wp);                                     \
    }                                                                             \
}

QWEN35_RMS_NORM_KERNEL(qwen35_rms_norm_f32, float)
QWEN35_RMS_NORM_KERNEL(qwen35_rms_norm_bf16, bfloat)
