// Overflow-free sigmoid and SiLU, torch's softplus and the GDN log decay,
// and the partial-RoPE angle, shared by the Qwen3.5 kernels.
//
// The GDN gates (`beta = sigmoid(b)`, the conv's and the gated norm's SiLU),
// the attention output gate and the MLP's SwiGLU all need the same sigmoid,
// and each .metal file is its own translation unit. Without this header each
// kept a copy (`gdn_sigmoid`, `attn_sigmoid`), which must not drift apart.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// `1 / (1 + e^-x)` without ever forming `e^|x|`: Metal compiles with fast
/// math, which may assume no intermediate is infinite, so the textbook form's
/// `exp(-x) = inf` for x < -88 is not a safe route to 0 on device.
inline float qwen35_sigmoid(float x)
{
    const float e = exp(-fabs(x));
    const float r = 1.0f / (1.0f + e);
    return x >= 0.0f ? r : e * r;
}

inline float qwen35_silu(float x)
{
    return x * qwen35_sigmoid(x);
}

/// transformers' partial-RoPE angle for rotary pair `p` of `rotary_dim` at
/// position `pos`: `pos * theta^(-2p / rotary_dim)`. The attention forward
/// and its backward both rotate with this, so the backward's inverse rotation
/// is exactly the transpose of the forward's.
///
/// torch computes inv_freq, the angle and cos/sin in fp32. `precise::`
/// throughout: the angle reaches tens of thousands of radians, where the fast
/// approximations lose whole digits.
inline float qwen35_rope_angle(uint p, uint rotary_dim, uint pos, float theta)
{
    const float inv_freq =
        precise::divide(1.0f, precise::pow(theta, precise::divide((float)(2u * p), (float)rotary_dim)));
    return (float)pos * inv_freq;
}

/// torch's `F.softplus` at its defaults (beta = 1, threshold = 20): linear above
/// the threshold, `log1p(e^x)` below it.
///
/// MSL has no `log1p`, and `log(1 + e)` is not a substitute where it matters:
/// rounding `1 + e` alone costs `2^-24 / e` relative, and the fast `log` near 1
/// has an absolute error around 2^-21 — together 1-60% of the result for x in
/// [-15, -8], which `a + dt_bias` reaches routinely (Qwen's dt_bias sits around
/// -2 to -7). Below -3 the series `e - e^2/2 + ... - e^8/8` is exact to f32
/// (the first dropped term is 1e-11 relative); above it `1 + e` loses at most
/// 1.2e-6 relative, and `precise::log` adds half an ulp.
inline float qwen35_softplus(float x)
{
    if (x > 20.0f) return x;
    // precise: fast `exp` is documented to 3 + floor(2|x|) ulp, ~33 ulp at
    // x = -15, and below -3 softplus is e^x to within the series. Once per
    // token per head, so the cost is nothing.
    const float e = precise::exp(x);
    if (x < -3.0f) {
        // log1p(e) by Horner: e * (1 - e/2 + e^2/3 - ... - e^7/8).
        float p = -1.0f / 8.0f;
        p = p * e + 1.0f / 7.0f;
        p = p * e - 1.0f / 6.0f;
        p = p * e + 1.0f / 5.0f;
        p = p * e - 1.0f / 4.0f;
        p = p * e + 1.0f / 3.0f;
        p = p * e - 1.0f / 2.0f;
        p = p * e + 1.0f;
        return e * p;
    }
    return precise::log(1.0f + e);
}

/// `g = -exp(A_log) * softplus(a + dt_bias)`: the GDN's log of the per-step
/// decay. Always <= 0. The inference kernels fold it into their loads and the
/// training gate kernel writes it out; both call this.
inline float qwen35_log_decay(float a, float a_log, float dt_bias)
{
    return -exp(a_log) * qwen35_softplus(a + dt_bias);
}
