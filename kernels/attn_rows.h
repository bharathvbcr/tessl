// The online-softmax row body shared by the one-simdgroup-per-query-row
// attention kernels: `flash_attn_rows.metal` (causal, sliding or global) and
// `encoder_attn.metal` (bidirectional, symmetric window, per-row lengths).
//
// Each kernel decides which keys a row visits and which of them are live;
// these helpers are what happens per key and at the end, so the two cannot
// drift apart. The lane layout is `flash_attn_rows.metal`'s (read its header
// for why it is fast): R lanes per row, lane `dl` of a row owning the float4s
// `dl, dl + R, dl + 2R, ...` of the head, DPV = D / (4R) of them.
//
// Included, never compiled on its own; `build.rs` tracks `.h` changes.
#pragma once

#include <metal_stdlib>
using namespace metal;

/// `q · k` for one key: this lane's DPV float4 products, then a butterfly over
/// the row's R lanes, which leaves the full sum in every one of them (no
/// broadcast step, and the softmax state below stays uniform across the row).
template <uint DPV, uint R>
inline float attn_rows_dot(thread const float4 *q_reg, device const float4 *K4, uint dl)
{
    float4 dot4 = float4(0.0f);
    for (uint j = 0; j < DPV; ++j) {
        dot4 += q_reg[j] * K4[dl + j * R];
    }
    float part = dot4.x + dot4.y + dot4.z + dot4.w;
    for (uint off = R / 2u; off > 0u; off >>= 1) {
        part += simd_shuffle_xor(part, off);
    }
    return part;
}

/// One key of the online softmax: `part` is its unscaled score, `live` its
/// mask; the row's accumulator, running maximum and sum are updated in place
/// with this lane's slice of the key's value row.
///
/// -FLT_MAX, not -INFINITY, seeds `m_i`: kernels compile with fast math, which
/// may assume no value is infinite. `l_i > 0` is the "has seen a key" flag:
/// the first live key contributes exp(0) = 1, and `l_i` never shrinks below 1
/// after that, because each new maximum adds 1 again. A row that has seen
/// nothing has a zero accumulator, so its rescale is exactly zero; and a
/// masked key's weight is zero by the mask, not by its score, since
/// -FLT_MAX - -FLT_MAX is 0.
template <uint DPV, uint R>
inline void attn_rows_step(
    float part, bool live, float scale, device const float4 *V4, uint dl,
    thread float4 *acc, thread float &m_i, thread float &l_i)
{
    const float s = live ? part * scale : -FLT_MAX;
    const float m_new = max(m_i, s);
    const float alpha = (l_i > 0.0f) ? exp(m_i - m_new) : 0.0f;
    const float p = live ? exp(s - m_new) : 0.0f;
    for (uint j = 0; j < DPV; ++j) {
        acc[j] = acc[j] * alpha + p * V4[dl + j * R];
    }
    l_i = l_i * alpha + p;
    m_i = m_new;
}

/// This lane's slice of the output row, `acc * inv_l`, as f32 or bf16.
template <uint DPV, uint R>
inline void attn_rows_store(
    device float *O, ulong o_off, uint dl, thread const float4 *acc, float inv_l, uint out_bf16)
{
    if (out_bf16 != 0u) {
        device bfloat *Ob = (device bfloat *)O;
        for (uint j = 0; j < DPV; ++j) {
            const ulong d0 = o_off + 4u * (dl + j * R);
            const float4 o4 = acc[j] * inv_l;
            Ob[d0 + 0u] = bfloat(o4.x);
            Ob[d0 + 1u] = bfloat(o4.y);
            Ob[d0 + 2u] = bfloat(o4.z);
            Ob[d0 + 3u] = bfloat(o4.w);
        }
    } else {
        device float4 *O4 = (device float4 *)(O + o_off);
        for (uint j = 0; j < DPV; ++j) {
            O4[dl + j * R] = acc[j] * inv_l;
        }
    }
}
