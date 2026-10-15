// The encoder attention's matrix-unit entry points: `attn_tiled.h`'s
// FlashAttention-2 body in its `ENCODER` mode behind `encoder_attn.metal`'s
// buffer slots and contract. Shared by `encoder_attn.metal` (the production
// instantiations) and `tune/encoder_attn_tune.metal` (the geometry sweep), so
// a swept kernel is the shipped kernel, not a copy.
//
// Q/O: [B, T, H, D]; K/V: [B, kv_capacity, Hkv, D]; lens: [B] device u32,
// clamped to T before it bounds any extent (the buffer is device-writable).
// The name spells the geometry (`_h{D}_q{BQ}_k{BK}_sg{NSG}`) and the host
// builds its grid from the same numbers, so the name is the contract between
// them.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
#include "attn_tiled.h"
using namespace metal;

/// Operands f32 (Q, K, V, O).
#define ENCODER_TILED_KERNEL(NAME, D, BQ, BK, NSG)                             \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *O [[buffer(3)]],                                            \
    constant uint &T [[buffer(4)]],                                           \
    device const uint *lens [[buffer(5)]],                                    \
    constant uint &H [[buffer(6)]],                                           \
    constant uint &Hkv [[buffer(7)]],                                         \
    constant uint &window [[buffer(8)]],                                      \
    constant float &scale [[buffer(9)]],                                      \
    constant uint &kv_capacity [[buffer(10)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[(BQ) * (BK)];                                         \
    threadgroup float m_row[BQ];                                              \
    threadgroup float l_row[BQ];                                              \
    threadgroup float a_row[BQ];                                              \
    /* Clamp the device-held length before it bounds any extent. */           \
    const uint len = min(lens[tgpig.y / H], T);                               \
    attn_tiled_body<D, BQ, BK, NSG, true>(                                    \
        const_cast<device float *>(Q), const_cast<device float *>(K),         \
        const_cast<device float *>(V), O, nullptr, T, len, H, Hkv, scale,     \
        0ul, 0ul, 0u, kv_capacity, window, tgpig, tid, S, m_row, l_row,       \
        a_row);                                                               \
}

/// Operands bf16: Q, K and V are bf16 in device memory (the q/k-norm-RoPE
/// writer stores them so), the probability tile is `PT` (bfloat, or float for
/// the mixed-operand variant, where P is S in place). Scores, the online
/// softmax, the accumulator and O are f32.
#define ENCODER_TILED_BF16_KERNEL_IMPL(NAME, D, BQ, BK, NSG, PT, P_DECL, P_PTR) \
kernel void NAME(                                                             \
    device const bfloat *Q [[buffer(0)]],                                     \
    device const bfloat *K [[buffer(1)]],                                     \
    device const bfloat *V [[buffer(2)]],                                     \
    device float *O [[buffer(3)]],                                            \
    constant uint &T [[buffer(4)]],                                           \
    device const uint *lens [[buffer(5)]],                                    \
    constant uint &H [[buffer(6)]],                                           \
    constant uint &Hkv [[buffer(7)]],                                         \
    constant uint &window [[buffer(8)]],                                      \
    constant float &scale [[buffer(9)]],                                      \
    constant uint &kv_capacity [[buffer(10)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[(BQ) * (BK)];                                         \
    P_DECL                                                                    \
    threadgroup float m_row[BQ];                                              \
    threadgroup float l_row[BQ];                                              \
    threadgroup float a_row[BQ];                                              \
    const uint len = min(lens[tgpig.y / H], T);                               \
    attn_tiled_core<D, BQ, BK, NSG, true, bfloat, PT>(                        \
        const_cast<device bfloat *>(Q), const_cast<device bfloat *>(K),       \
        const_cast<device bfloat *>(V), O, nullptr, T, len, H, Hkv, scale,    \
        0ul, 0ul, 0u, kv_capacity, window, tgpig, tid, S, P_PTR, m_row,       \
        l_row, a_row);                                                        \
}

/// P rounded to bf16 into its own tile: both operands of P.V are bf16.
#define ENCODER_TILED_BF16_KERNEL(NAME, D, BQ, BK, NSG)                        \
    ENCODER_TILED_BF16_KERNEL_IMPL(NAME, D, BQ, BK, NSG, bfloat,               \
        threadgroup bfloat P[(BQ) * (BK)];, P)

/// P kept f32 (S in place): the mixed-operand variant, f32 P times bf16 V.
#define ENCODER_TILED_BF16_PF32_KERNEL(NAME, D, BQ, BK, NSG)                   \
    ENCODER_TILED_BF16_KERNEL_IMPL(NAME, D, BQ, BK, NSG, float, , S)
