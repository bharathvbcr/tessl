// Bidirectional (encoder) attention: one simdgroup per query row.
//
// The row mapping, lane layout and float4 dim slicing are
// `flash_attn_rows.metal`'s (read that file's header for why they are fast),
// and the per-key online softmax and the store are the same code
// (`attn_rows.h`); what differs is the masking contract, which is an
// encoder's, not a decoder's:
//
//   * no causal rule: a query sees keys on both sides;
//   * the sliding window is symmetric and inclusive, |t_k - t_q| <= window
//     (transformers' `sliding_window_bidirectional_overlay`), so a row sees up
//     to 2*window + 1 keys; `window == 0` means every key (a global layer);
//   * every batch row has its own live length `lens[b]` (right padding): keys
//     at or past it are masked, and so is every query at or past it, whose
//     output row is written as zeros;
//   * positions are 0..T-1 in every row, so there are no position offsets.
//     K/V hold `kv_capacity >= T` positions per sequence (the host derives it
//     from the buffers' size, as flash attention does), so one pair of K/V
//     buffers serves forwards of different shapes.
//
// It is a separate kernel, not a mode of `flash_attn_rows`, because there the
// causal rule is not a mask but the loop bound (keys above the query are never
// visited), and the live length is one device scalar shared by every row. A
// mode flag would put both contracts in one tuned body that gemma-metal and
// the Qwen3.5 prefill route through.
//
// Q/O: [B, T, H, D]; K/V: [B, kv_capacity, Hkv, D]; lens: [B] device u32. The host
// checks lens[b] in 1..=T for every row; the kernel still clamps to T before
// any address is formed, since the buffer is device-writable.
#include <metal_stdlib>
#include "attn_rows.h"
#include "attn_tiled.h"
using namespace metal;

constant uint ENC_SG_W = 32;

#define ENC_ROWS_KERNEL(NAME, D, R, SGT)                                      \
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
    constant uint &out_bf16 [[buffer(10)]],                                   \
    constant uint &kv_capacity [[buffer(11)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]])                           \
{                                                                             \
    constexpr uint RPS = ENC_SG_W / (R);                                      \
    constexpr uint DPV = (D) / (4u * (R));                                    \
    constexpr uint RPT = (SGT) * RPS;                                         \
    const uint tid = tpitg.x;                                                 \
    const uint sg = tid / ENC_SG_W;                                           \
    const uint lane = tid % ENC_SG_W;                                         \
    const uint sub = lane / (R);                                              \
    const uint dl = lane % (R);                                               \
                                                                              \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    const ulong base_row = (ulong)tgpig.x * RPT + sg * RPS;                  \
    if (base_row >= (ulong)T) { return; }                                     \
    const uint live_rows = min(RPS, (uint)((ulong)T - base_row));             \
    const bool row_live = sub < live_rows;                                    \
    const uint t_q = row_live ? (uint)(base_row + sub) : (uint)base_row;      \
    /* Clamp the device-held length before it bounds any address. */         \
    const ulong len = (ulong)min(lens[b], T);                                 \
    const bool q_valid = row_live && (ulong)t_q < len;                        \
                                                                              \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
    const ulong kv_pos_stride = (ulong)Hkv * (D);                             \
    const ulong kv_head_base = (ulong)b * kv_capacity * kv_pos_stride + (ulong)hkv * (D); \
    const ulong q_pos_stride = (ulong)H * (D);                                \
    const ulong q_head_base = (ulong)b * T * q_pos_stride + (ulong)h * (D);   \
    const ulong w = (ulong)window;                                            \
    const ulong my_lo = (w == 0ul || (ulong)t_q < w) ? 0ul : (ulong)t_q - w;  \
    const ulong my_hi = (w == 0ul) ? len : min(len, (ulong)t_q + w + 1ul);    \
                                                                              \
    /* Union key range over this simdgroup's rows, so every lane runs the     \
       same trip count; rows mask individually inside it. */                 \
    const ulong q_lo = base_row;                                              \
    const ulong q_hi = base_row + live_rows - 1ul;                            \
    const ulong t_start = (w == 0ul || q_lo < w) ? 0ul : q_lo - w;            \
    const ulong t_end = (q_lo >= len) ? 0ul                                   \
        : ((w == 0ul) ? len : min(len, q_hi + w + 1ul));                      \
                                                                              \
    const ulong o_off = q_head_base + (ulong)t_q * q_pos_stride;             \
    float4 q_reg[DPV];                                                        \
    float4 acc[DPV];                                                          \
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }                 \
    /* The seeds and the "has seen a key" flag: see attn_rows_step. */      \
    float m_i = -FLT_MAX;                                                     \
    float l_i = 0.0f;                                                         \
                                                                              \
    if (t_start < t_end) {                                                    \
        device const float4 *Q4 = (device const float4 *)(Q + o_off);         \
        for (uint j = 0; j < DPV; ++j) {                                      \
            q_reg[j] = q_valid ? Q4[dl + j * (R)] : float4(0.0f);             \
        }                                                                     \
        for (ulong t = t_start; t < t_end; ++t) {                             \
            const ulong kv_base = kv_head_base + t * kv_pos_stride;          \
            const float part = attn_rows_dot<DPV, (R)>(                       \
                q_reg, (device const float4 *)(K + kv_base), dl);             \
            const bool live = q_valid && t >= my_lo && t < my_hi;             \
            attn_rows_step<DPV, (R)>(part, live, scale,                       \
                (device const float4 *)(V + kv_base), dl, acc, m_i, l_i);     \
        }                                                                     \
    }                                                                         \
                                                                              \
    if (!row_live) { return; }                                                \
    /* A padding query (or, defensively, a row that saw no key) is zeros. */  \
    const float inv_l = (q_valid && l_i > 0.0f) ? (1.0f / l_i) : 0.0f;        \
    attn_rows_store<DPV, (R)>(O, o_off, dl, acc, inv_l, out_bf16);           \
}

// The lane counts and simdgroups per threadgroup are `flash_attn_rows`'
// measured winners at these head dims (16 dims per lane; `nn::rows_lanes_for`
// and `nn::rows_groups_for`), the same register budget per lane.
ENC_ROWS_KERNEL(encoder_attn_rows_h256_r16_g32, 256, 16, 32)
ENC_ROWS_KERNEL(encoder_attn_rows_h512_r32_g32, 512, 32, 32)
// BERT's head dim (src/bert.rs: 384 hidden over 12 heads). Two lanes per row
// keeps the 16 dims per lane above, so 16 rows share a simdgroup; eight
// simdgroups make a threadgroup of 128 rows, which covers a 512-token
// sequence in four rather than leaving most of a 32-simdgroup group idle.
ENC_ROWS_KERNEL(encoder_attn_rows_h32_r2_g8, 32, 2, 8)
// DistilBERT's (768 over 12): four lanes per row, the same 16 dims per lane
// and 128 rows per threadgroup.
ENC_ROWS_KERNEL(encoder_attn_rows_h64_r4_g16, 64, 4, 16)

/// The same contract on the matrix units: `attn_tiled.h`'s FlashAttention-2
/// body in its `ENCODER` mode, with this file's buffer slots and contract (Q/O `[B, T, H, D]`, K/V `[B, kv_capacity, Hkv, D]`, `lens` `[B]`
/// device u32, clamped to T before it bounds anything), on the matrix units.
/// The name spells the geometry, as above. D=512 halves BQ per simdgroup
/// against D=256 so the O accumulator stays at 64 floats per thread.
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
    constant uint &out_bf16 [[buffer(10)]],                                   \
    constant uint &kv_capacity [[buffer(11)]],                                \
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
        0ul, 0ul, out_bf16, kv_capacity, window, tgpig, tid, S, m_row, l_row, \
        a_row);                                                               \
}

ENCODER_TILED_KERNEL(encoder_attn_tiled_h256_q32_k32_sg4, 256, 32, 32, 4)
ENCODER_TILED_KERNEL(encoder_attn_tiled_h512_q32_k32_sg8, 512, 32, 32, 8)
