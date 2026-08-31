// Shared util kernels extracted from metal-native (copy/cast/zero/softcap/transpose).
#include <metal_stdlib>
using namespace metal;

kernel void copy_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

kernel void copy_bf16(
    device const bfloat *in [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

kernel void cast_f32_to_bf16(
    device const float *in [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = bfloat(in[gid]);
}

kernel void cast_bf16_to_f32(
    device const bfloat *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = float(in[gid]);
}

kernel void zero_f32(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = 0.0f;
}

kernel void add_inplace_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] += src[gid];
}

kernel void transpose2d_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * cols) return;
    uint i = gid / cols;
    uint j = gid % cols;
    out[j * rows + i] = in[i * cols + j];
}

/// logits_post = softcap * tanh(logits / softcap).
kernel void softcap_f32(
    device const float *pre [[buffer(0)]],
    device float *post [[buffer(1)]],
    constant float &softcap [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    // Clamp before `tanh`, not after.
    //
    // Metal's `tanh` is evaluated through `exp(2z)`, which leaves float range
    // around |z| ~= 44: the result goes to `inf` and then to `NaN`. Measured at
    // cap=30 this kernel was correct through pre=1250, returned `inf` near
    // pre=1300, and `NaN` from pre=1350 — on the one input class softcapping
    // exists to tame. A single NaN logit poisons its whole softmax row, so the
    // failure is far wider than the element that produced it.
    //
    // In f32, `tanh(z)` has already rounded to exactly +/-1 by |z| ~= 8.7
    // (1 - tanh(z) drops below 2^-24 there), so clamping at 16 is well past
    // saturation and changes no representable result. NaN input still
    // propagates as NaN: that is the caller's data, not an overflow.
    float z = clamp(pre[gid] / softcap, -16.0f, 16.0f);
    post[gid] = softcap * tanh(z);
}

kernel void copy_f16(
    device const half *in [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = in[gid];
}

/// f32 -> IEEE half.
///
/// Unlike bf16, half has only 5 exponent bits, so values above 65504 do not
/// merely lose precision — they become infinity. That is a real difference for
/// activations, and it is why the cast is a conversion rather than a
/// truncation the way `cast_f32_to_bf16` is.
kernel void cast_f32_to_f16(
    device const float *in [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = half(in[gid]);
}

kernel void cast_f16_to_f32(
    device const half *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = float(in[gid]);
}
