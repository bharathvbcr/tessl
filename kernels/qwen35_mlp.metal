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
#include "act.h"
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

GATED_KERNEL(qwen35_swiglu_f32, float, tessl_silu)
GATED_KERNEL(qwen35_swiglu_bf16, bfloat, tessl_silu)
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
/// with the host's `reduce_tptg` the rows' mean square is the one tessl's
/// generic RMSNorm computes. The inverse root is `precise::rsqrt`, not
/// `rsqrt`: `qwen35_rms_norm_bwd_f32` rebuilds `rstd` from `x` with
/// `precise::rsqrt`, and training needs both to be the same bits.
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
    const float inv = precise::rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + eps); \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        const float wp = 1.0f + weight[d];                                        \
        xout[d] = (OUT_T)(xin[d] * inv * wp);                                     \
    }                                                                             \
}

QWEN35_RMS_NORM_KERNEL(qwen35_rms_norm_f32, float)
QWEN35_RMS_NORM_KERNEL(qwen35_rms_norm_bf16, bfloat)

/// Decode's residual add and the norm after it in one pass:
/// `resid[r, :] += y[r, :]`, then `out[r, :] = rms_norm(resid[r, :]) * (1 + w)`
/// with `qwen35_rms_norm_*`'s arithmetic, so the pair of dispatches it
/// replaces (`qwen35_residual_add_f32`, then `qwen35_rms_norm_*`) and this one
/// give the same bits: the sum is the same f32 add, the sum of squares walks
/// the same lanes in the same order and folds with the same `reduce_row_add`,
/// and the scale is `precise::rsqrt` of the same expression.
///
/// Each element of `resid` is read and written by the lane that later reads it
/// back, so the second pass needs no barrier. `y`, `weight` and `resid` are
/// distinct buffers and `out` overlaps none of them.
///
/// Grid: one threadgroup per row.
#define QWEN35_ADD_RMS_NORM_KERNEL(NAME, OUT_T)                                   \
kernel void NAME(                                                                 \
    device const float *y [[buffer(0)]],                                          \
    device float *resid [[buffer(1)]],                                            \
    device const float *weight [[buffer(2)]],                                     \
    device OUT_T *out [[buffer(3)]],                                              \
    constant uint &rows [[buffer(4)]],                                            \
    constant uint &dim [[buffer(5)]],                                             \
    constant float &eps [[buffer(6)]],                                            \
    uint row [[threadgroup_position_in_grid]],                                    \
    uint lid [[thread_position_in_threadgroup]],                                  \
    uint sgid [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint tptg [[threads_per_threadgroup]])                                        \
{                                                                                 \
    if (row >= rows) return;                                                      \
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];                             \
    device const float *yin = y + (ulong)row * dim;                               \
    device float *rin = resid + (ulong)row * dim;                                 \
    device OUT_T *xout = out + (ulong)row * dim;                                  \
    float ss = 0.0f;                                                              \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        const float v = rin[d] + yin[d];                                          \
        rin[d] = v;                                                               \
        ss += v * v;                                                              \
    }                                                                             \
    const float inv = precise::rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + eps); \
    for (ulong d = lid; d < (ulong)dim; d += tptg) {                              \
        const float wp = 1.0f + weight[d];                                        \
        xout[d] = (OUT_T)(rin[d] * inv * wp);                                     \
    }                                                                             \
}

QWEN35_ADD_RMS_NORM_KERNEL(qwen35_add_rms_norm_f32, float)
QWEN35_ADD_RMS_NORM_KERNEL(qwen35_add_rms_norm_bf16, bfloat)

/// Decode's MLP front half in one dispatch, one token:
/// `out[n] = silu(sum_k x[k] * w_gate[k, n]) * (sum_k x[k] * w_up[k, n])`:
/// the two GEMMs `x @ w_gate` and `x @ w_up` and `qwen35_swiglu_*` between
/// them, with the `[1, n]` gate and up vectors never written out. `x`, both
/// weights and `out` are one dtype (f32 for the exact-f32 forward, bf16 for the
/// bf16 one, as every operand of the GEMMs they replace is); the sums are f32
/// either way and the product is rounded once, to `out`'s type.
///
/// Weights are the model's `[k, n]` row-major (`x @ w`). A threadgroup owns
/// QG_COLS columns of both: a simdgroup is 8 column groups of QG_VEC adjacent
/// columns by 4 k-lanes, and the eight simdgroups stride k by 32 between them,
/// so a threadgroup walks k with 32 k-lanes. The four k-lanes of a simdgroup
/// fold with two xor shuffles, the eight simdgroups through threadgroup memory
/// in simdgroup order: a fixed order, so the result is deterministic. It is
/// not the TensorOps GEMM's order, so the two agree to f32 rounding of the
/// sums (the gate's and up's, before the activation), not bit for bit.
///
/// A column group past `n`, and a row pitch that is not a multiple of QG_VEC
/// (`n % 8 != 0`, which would misalign the 16-byte loads), take the scalar
/// path: same sums, same order.
///
/// Grid: ceil(n / QG_COLS) threadgroups of QG_TPTG threads.
#define QG_VEC 8u
#define QG_COLS 64u
#define QG_TPTG 256u

/// Eight consecutive elements as f32, from one 16-byte load (two for f32).
inline void qg_load8(device const float *p, thread float *v)
{
    const float4 a = *(device const float4 *)p;
    const float4 b = *(device const float4 *)(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

inline void qg_load8(device const bfloat *p, thread float *v)
{
    const uint4 a = *(device const uint4 *)p;
    for (uint i = 0u; i < 4u; ++i) {
        v[2u * i] = as_type<float>(a[i] << 16);
        v[2u * i + 1u] = as_type<float>(a[i] & 0xffff0000u);
    }
}

#define QWEN35_GATE_UP_SILU_KERNEL(NAME, T)                                       \
kernel void NAME(                                                                 \
    device const T *x [[buffer(0)]],                                              \
    device const T *w_gate [[buffer(1)]],                                         \
    device const T *w_up [[buffer(2)]],                                           \
    device T *out [[buffer(3)]],                                                  \
    constant uint &k [[buffer(4)]],                                               \
    constant uint &n [[buffer(5)]],                                               \
    uint tg [[threadgroup_position_in_grid]],                                     \
    uint sgid [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]],                                      \
    uint lid [[thread_position_in_threadgroup]])                                  \
{                                                                                 \
    threadgroup float red[16u * QG_COLS];                                         \
    const uint cg = lane & 7u;                                                    \
    const uint kl = lane >> 3;                                                    \
    const ulong col0 = (ulong)tg * QG_COLS + (ulong)cg * QG_VEC;                  \
    const bool vec = ((n & (QG_VEC - 1u)) == 0u) && (col0 + QG_VEC <= (ulong)n);  \
    float ag[QG_VEC];                                                             \
    float au[QG_VEC];                                                             \
    for (uint j = 0u; j < QG_VEC; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }            \
    for (uint kk = sgid * 4u + kl; kk < k; kk += 32u) {                           \
        const float xv = (float)x[kk];                                            \
        const ulong base = (ulong)kk * n + col0;                                  \
        if (vec) {                                                                \
            float wg[QG_VEC];                                                     \
            float wu[QG_VEC];                                                     \
            qg_load8(w_gate + base, wg);                                          \
            qg_load8(w_up + base, wu);                                            \
            for (uint j = 0u; j < QG_VEC; ++j) {                                  \
                ag[j] = fma(xv, wg[j], ag[j]);                                    \
                au[j] = fma(xv, wu[j], au[j]);                                    \
            }                                                                     \
        } else {                                                                  \
            for (uint j = 0u; j < QG_VEC; ++j) {                                  \
                if (col0 + j < (ulong)n) {                                        \
                    ag[j] = fma(xv, (float)w_gate[base + j], ag[j]);              \
                    au[j] = fma(xv, (float)w_up[base + j], au[j]);                \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    }                                                                             \
    for (uint j = 0u; j < QG_VEC; ++j) {                                          \
        ag[j] += simd_shuffle_xor(ag[j], 8u);                                     \
        ag[j] += simd_shuffle_xor(ag[j], 16u);                                    \
        au[j] += simd_shuffle_xor(au[j], 8u);                                     \
        au[j] += simd_shuffle_xor(au[j], 16u);                                    \
    }                                                                             \
    if (kl == 0u) {                                                               \
        for (uint j = 0u; j < QG_VEC; ++j) {                                      \
            red[(sgid * 2u) * QG_COLS + cg * QG_VEC + j] = ag[j];                                \
            red[(sgid * 2u + 1u) * QG_COLS + cg * QG_VEC + j] = au[j];                                \
        }                                                                         \
    }                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    if (lid < QG_COLS) {                                                          \
        const ulong col = (ulong)tg * QG_COLS + lid;                              \
        if (col < (ulong)n) {                                                     \
            float g = 0.0f;                                                       \
            float u = 0.0f;                                                       \
            for (uint s = 0u; s < 8u; ++s) { g += red[(s * 2u) * QG_COLS + lid]; u += red[(s * 2u + 1u) * QG_COLS + lid]; } \
            out[col] = (T)(tessl_silu(g) * u);                                    \
        }                                                                         \
    }                                                                             \
}

QWEN35_GATE_UP_SILU_KERNEL(qwen35_gate_up_silu_f32, float)
QWEN35_GATE_UP_SILU_KERNEL(qwen35_gate_up_silu_bf16, bfloat)
