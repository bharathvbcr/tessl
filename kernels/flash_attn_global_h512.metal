// Causal global FlashAttention @ head_dim=512 (FA-2 tiled) + GQA.
// The body is `attn_fa2.h`; this file owns the entry name and buffer slots.
// There is no `window` (slot 9 is `scale`). Shared-KV consumers pass densified /
// full-length producer K/V buffers.
// out_bf16=1 (slot 12): write O as bfloat (half-width act scratch for o_proj GEMV).
#include <metal_stdlib>
#include "attn_fa2.h"
using namespace metal;

constant uint HEAD_DIM = 512;
constant uint BR = 4;
constant uint BC = 4;

/// Per-token varying FA scalars from stable device u32s (ICB / encode-once).
kernel void flash_attn_global_h512(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &Tq [[buffer(5)]],
    device const uint *Tkv_ptr [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    device const uint *q_pos_offset_ptr [[buffer(10)]],
    device const uint *kv_pos_offset_ptr [[buffer(11)]],
    constant uint &out_bf16 [[buffer(12)]],
    constant uint &kv_capacity [[buffer(13)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    threadgroup float Oacc[BR * HEAD_DIM];
    threadgroup float Ks[BC * FA2_D_TILE];
    threadgroup float Vs[BC * FA2_D_TILE];
    threadgroup float scores[BR * BC];
    flash_attn_fa2_tiled<HEAD_DIM, BR, BC, false, true>(
        Q, K, V, O, Tq, Tkv_ptr, H, Hkv, 0u, scale, q_pos_offset_ptr,
        kv_pos_offset_ptr, out_bf16, kv_capacity, tgpig, tpitg, tptg_vec, Oacc, Ks,
        Vs, scores);
    (void)B;
}
