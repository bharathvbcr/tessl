#ifndef TESSL_GELU_H
#define TESSL_GELU_H

#include <metal_stdlib>
using namespace metal;

/// Canonical PyTorch tanh-approximate GELU for every fused and standalone
/// Tessl kernel.
///
/// The cubic input is bounded so finite f32 inputs cannot overflow `x^3`, and
/// the precise tanh argument is bounded because the fast Metal lowering can
/// return NaN well past saturation. Only the approximation's cubic is bounded:
/// the outer factor must remain the original `x` so GELU approaches `x` for
/// large positive inputs instead of clipping at the guard value.
static inline float tessl_gelu_pytorch_tanh(float x) {
    const float xc = clamp(x, -20.0f, 20.0f);
    const float x3 = xc * xc * xc;
    const float inner = 0.7978845608028654f * (xc + 0.044715f * x3);
    const float t = precise::tanh(clamp(inner, -10.0f, 10.0f));
    return 0.5f * x * (1.0f + t);
}

#endif
