// The crate's activations: GELU-tanh, sigmoid and SiLU, shared by every
// fused and standalone kernel that applies one.
//
// Each .metal file is its own translation unit, so a kernel that wants an
// activation either includes this or keeps a copy, and copies drift: a GEMM
// epilogue copy of GELU once clipped at 20, and two textbook SiLU copies
// outlived the overflow rule below.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#ifndef TESSL_ACT_H
#define TESSL_ACT_H

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

/// `1 / (1 + e^-x)` without ever forming `e^|x|`: Metal compiles with fast
/// math, which may assume no intermediate is infinite, so the textbook form's
/// `exp(-x) = inf` for x < -88 is not a safe route to 0 on device. Here the
/// denominator stays in [1, 2].
static inline float tessl_sigmoid(float x) {
    const float e = exp(-fabs(x));
    const float r = 1.0f / (1.0f + e);
    return x >= 0.0f ? r : e * r;
}

/// SiLU (swish), `x * sigmoid(x)`, for `nn::mlp_silu`, the GEMM epilogue's
/// `Activation::Silu` and the Qwen3.5 kernels alike.
static inline float tessl_silu(float x) {
    return x * tessl_sigmoid(x);
}

#endif
