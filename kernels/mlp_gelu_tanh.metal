// Fused gate×up with gelu_pytorch_tanh (NOT SiLU / swish).
// gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// The cubic is clamped so x³ cannot overflow, and precise::tanh's argument is
// clamped because -O2 lowers tanh to fast_tanh, which NaNs past saturation.
// The outer factor stays the original x (see gelu.h).
#include <metal_stdlib>
#include "gelu.h"
using namespace metal;

/// out[i] = gelu_pytorch_tanh(gate[i]) * up[i]
kernel void mlp_gelu_tanh(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = tessl_gelu_pytorch_tanh(gate[gid]) * up[gid];
}

/// Same as mlp_gelu_tanh, writing bf16 (down-proj GEMV input; kills cast pass).
kernel void mlp_gelu_tanh_bf16(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = bfloat(tessl_gelu_pytorch_tanh(gate[gid]) * up[gid]);
}
