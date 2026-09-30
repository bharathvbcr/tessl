// Overflow-free sigmoid and SiLU, and the partial-RoPE angle, shared by the
// Qwen3.5 kernels.
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
