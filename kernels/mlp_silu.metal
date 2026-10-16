// Fused gate×up with SiLU / swish (the gated-MLP activation), dispatched by `nn::mlp_silu`.
// silu(x) = x * sigmoid(x), the crate's overflow-free one (act.h).
#include <metal_stdlib>
#include "act.h"
using namespace metal;

/// out[i] = silu(gate[i]) * up[i]
kernel void mlp_silu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = tessl_silu(gate[gid]) * up[gid];
}
