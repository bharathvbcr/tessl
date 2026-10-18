// Causal sliding-window FlashAttention @ head_dim=128 (FA-2 tiled); dispatched by `nn::flash_attn_swa`
// for 128-wide heads. The body is `attn_fa2.h`; this file owns the entry name and buffer slots.
// Slot 13 is `kv_capacity`: unlike h256/h512 this kernel has no `out_bf16` and always writes f32.
#include <metal_stdlib>
#include "attn_fa2.h"
using namespace metal;

constant uint HEAD_DIM = 128;
// Prefill Tq>1 benefits from BR=8; decode Tq=1 wastes lanes but FA is tiny vs GEMV.
constant uint BR = 8;
constant uint BC = 8;

/// Per-token varying FA scalars from stable device u32s (ICB / encode-once).
kernel void flash_attn_swa_h128(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &Tq [[buffer(5)]],
    device const uint *Tkv_ptr [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &window [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    device const uint *q_pos_offset_ptr [[buffer(11)]],
    device const uint *kv_pos_offset_ptr [[buffer(12)]],
    constant uint &kv_capacity [[buffer(13)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    threadgroup float Oacc[BR * HEAD_DIM];
    threadgroup float Ks[BC * FA2_D_TILE];
    threadgroup float Vs[BC * FA2_D_TILE];
    threadgroup float scores[BR * BC];
    flash_attn_fa2_tiled<HEAD_DIM, BR, BC, true, false>(
        Q, K, V, O, Tq, Tkv_ptr, H, Hkv, window, scale, q_pos_offset_ptr,
        kv_pos_offset_ptr, 0u, kv_capacity, tgpig, tpitg, tptg_vec, Oacc, Ks, Vs,
        scores);
    (void)B;
}
