// Overflow-free sigmoid and SiLU shared by the Qwen3.5 kernels.
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
