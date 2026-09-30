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
#include "qwen35_act.h"
using namespace metal;

/// `out[r, out_off + c] = silu(gate[r, gate_off + c]) * up[r, up_off + c]`
/// for `c < width`, `r < rows`; rows are `ld_*` elements apart.
///
/// Grid: x = column in [0, width), y = row in [0, rows).
#define SWIGLU_KERNEL(NAME, OUT_T)                                                \
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
    out[(ulong)r * ld_out + out_off + col] = (OUT_T)(qwen35_silu(g) * u);    \
}

SWIGLU_KERNEL(qwen35_swiglu_f32, float)
SWIGLU_KERNEL(qwen35_swiglu_bf16, bfloat)

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
