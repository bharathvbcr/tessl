// Fused gate×up with gelu_pytorch_tanh (NOT SiLU / swish).
// gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Root cause of prior NaNs: with -O2, MSL `tanh` lowers to `air.fast_tanh`,
// which NaNs for |arg| ≳ ~10. Host/PyTorch use a saturating precise tanh.
// The gelu inner term at |x|≈20 is ~301 → fast_tanh → NaN mid even when
// gate/up are finite. Fix: clamp x (x³ overflow) + precise::tanh on a
// clamped inner (tanh already saturates by |z|≳8).
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
