// Bidirectional (encoder) attention: one simdgroup per query row.
//
// The row mapping, lane layout, float4 dim slicing and online softmax are
// `flash_attn_rows.metal`'s (read that file's header for why they are fast);
// what differs is the masking contract, which is an encoder's, not a decoder's:
//
//   * no causal rule: a query sees keys on both sides;
//   * the sliding window is symmetric and inclusive, |t_k - t_q| <= window
//     (transformers' `sliding_window_bidirectional_overlay`), so a row sees up
//     to 2*window + 1 keys; `window == 0` means every key (a global layer);
//   * every batch row has its own live length `lens[b]` (right padding): keys
//     at or past it are masked, and so is every query at or past it, whose
//     output row is written as zeros;
//   * positions are 0..T-1 in every row, so there are no position offsets and
//     no KV capacity distinct from T.
//
// It is a separate kernel, not a mode of `flash_attn_rows`, because there the
// causal rule is not a mask but the loop bound (keys above the query are never
// visited), and the live length is one device scalar shared by every row. A
// mode flag would put both contracts in one tuned body that gemma-metal and
// the Qwen3.5 prefill route through.
//
// Q/O: [B, T, H, D]; K/V: [B, T, Hkv, D]; lens: [B] device u32. The host
// checks lens[b] in 1..=T for every row; the kernel still clamps to T before
// any address is formed, since the buffer is device-writable.
#include <metal_stdlib>
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
    const ulong kv_head_base = (ulong)b * T * kv_pos_stride + (ulong)hkv * (D); \
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
    /* -FLT_MAX, not -INFINITY: kernels compile with fast math, which may     \
       assume no value is infinite. l_i > 0 is the "has seen a key" flag: the \
       first live key contributes exp(0) = 1, and l_i never shrinks below 1   \
       after that, because each new maximum adds 1 again. */                  \
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
            device const float4 *K4 = (device const float4 *)(K + kv_base);   \
            float4 dot4 = float4(0.0f);                                       \
            for (uint j = 0; j < DPV; ++j) {                                  \
                dot4 += q_reg[j] * K4[dl + j * (R)];                          \
            }                                                                 \
            float part = dot4.x + dot4.y + dot4.z + dot4.w;                   \
            for (uint off = (R) / 2u; off > 0u; off >>= 1) {                  \
                part += simd_shuffle_xor(part, off);                          \
            }                                                                 \
            const bool live = q_valid && t >= my_lo && t < my_hi;             \
            const float s = live ? part * scale : -FLT_MAX;                   \
            const float m_new = max(m_i, s);                                  \
            /* A row that has seen nothing has a zero accumulator, so its     \
               rescale is exactly zero; and a masked key's weight is zero by  \
               the mask, not by its score, since -FLT_MAX - -FLT_MAX is 0. */ \
            const float alpha = (l_i > 0.0f) ? exp(m_i - m_new) : 0.0f;       \
            const float p = live ? exp(s - m_new) : 0.0f;                     \
            device const float4 *V4 = (device const float4 *)(V + kv_base);   \
            for (uint j = 0; j < DPV; ++j) {                                  \
                acc[j] = acc[j] * alpha + p * V4[dl + j * (R)];               \
            }                                                                 \
            l_i = l_i * alpha + p;                                            \
            m_i = m_new;                                                      \
        }                                                                     \
    }                                                                         \
                                                                              \
    if (!row_live) { return; }                                                \
    /* A padding query (or, defensively, a row that saw no key) is zeros. */  \
    const float inv_l = (q_valid && l_i > 0.0f) ? (1.0f / l_i) : 0.0f;        \
    if (out_bf16 != 0u) {                                                     \
        device bfloat *Ob = (device bfloat *)O;                               \
        for (uint j = 0; j < DPV; ++j) {                                      \
            const ulong d0 = o_off + 4u * (dl + j * (R));                     \
            const float4 o4 = acc[j] * inv_l;                                 \
            Ob[d0 + 0u] = bfloat(o4.x);                                       \
            Ob[d0 + 1u] = bfloat(o4.y);                                       \
            Ob[d0 + 2u] = bfloat(o4.z);                                       \
            Ob[d0 + 3u] = bfloat(o4.w);                                       \
        }                                                                     \
    } else {                                                                  \
        device float4 *O4 = (device float4 *)(O + o_off);                     \
        for (uint j = 0; j < DPV; ++j) {                                      \
            O4[dl + j * (R)] = acc[j] * inv_l;                                \
        }                                                                     \
    }                                                                         \
}

// The lane counts and simdgroups per threadgroup are `flash_attn_rows`'
// measured winners at these head dims (16 dims per lane; `nn::rows_lanes_for`
// and `nn::rows_groups_for`), the same register budget per lane.
ENC_ROWS_KERNEL(encoder_attn_rows_h256_r16_g32, 256, 16, 32)
ENC_ROWS_KERNEL(encoder_attn_rows_h512_r32_g32, 512, 32, 32)
