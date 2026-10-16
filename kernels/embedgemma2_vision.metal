// EmbeddingGemma 2's vision tower (transformers' `Gemma4VisionModel`): the three
// steps no other kernel does. The rest of the tower is tessl's existing kernels
// (exact-f32 GEMMs, `rms_norm_f32`, `gated_act`, `encoder_attn`); see
// `src/embedgemma2_vision.rs` and `docs/embedgemma2.md`.
//
//   vision_qkv_norm_rope_f32   q/k RMSNorm with `* w`, the 2-D axial RoPE, the
//                              weightless V RMSNorm and the K/V store
//                              (`Gemma4VisionAttention.forward`, up to the
//                              attention itself); `_bf16` stores q, k and v as
//                              bf16, the operands of the bf16 attention.
//   vision_pos_embed_add_f32   the patch embedder's `+ position_embeddings`
//                              (`Gemma4VisionPatchEmbedder._position_embeddings`).
//   vision_pool_members_f32    the pooler's k x k average by patch position,
//                              times sqrt(hidden)
//                              (`Gemma4VisionPooler._avg_pool_by_positions`).
//
// Every kernel treats its device-held index data (positions, offsets, member
// rows) as untrusted: a host that has validated them is the contract, and the
// kernel still clamps or skips before any address is formed.
#include <metal_stdlib>
using namespace metal;

/// Normalize one head row (`src * rsqrt(mean(src^2) + eps) * weight`) and apply
/// the axial RoPE, reading `src` and writing `dst` (never the same row).
///
/// transformers splits the head's `D` channels into two halves of `W = D / 2`
/// (`apply_multidimensional_rope`: `2 * (D // (2 * ndim))` per axis). Half 0 is
/// rotated by the patch's x position, half 1 by its y position, each with
/// `rotate_half` over its own `W` channels: pairs `(p, p + W/2)` with
/// `inv_freq[p] = theta^(-2p / W)` (`Gemma4VisionRotaryEmbedding`, whose cos/sin
/// are `cat([x, x, y, y])` over `D / 4` frequencies). So lane-pair `u` of
/// `W = 2 * (W/2)` is axis `u / (W/2)`, frequency `u % (W/2)`.
///
/// The angle is one f32 multiply, `position * inv_freq[p]`, as torch forms it
/// (`position_ids[..., None].float() * inv_freq`); `inv_freq` comes from the
/// host (`nn::rope_inv_freq`), never a device `pow` (see `qwen35_rope_angle`).
/// A padding patch's (-1, -1) rotates by -1: finite, and its keys are masked.
///
/// `OT` is the stored type: `float`, or `bfloat` for the operands of the bf16
/// encoder attention, rounded to nearest even from the f32 result (the one
/// rounding a cast pass would apply, with no pass). The arithmetic before the
/// store is the same in both, and `(float)` of a float is the identity.
template <typename OT>
inline void vision_norm_rope_row(
    device const float *src,
    device const float *weight,
    device OT *dst,
    uint D,
    float pos_x,
    float pos_y,
    constant float *inv_freq,
    float eps,
    uint lane)
{
    float ss = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        ss += src[d] * src[d];
    }
    const float inv = rsqrt(simd_sum(ss) / (float)D + eps);

    const uint W = D / 2u;
    const uint half_w = W / 2u;
    for (uint u = lane; u < W; u += 32u) {
        const uint axis = u / half_w;
        const uint p = u - axis * half_w;
        const uint base = axis * W;
        const float x0 = src[base + p] * inv * weight[base + p];
        const float x1 = src[base + p + half_w] * inv * weight[base + p + half_w];
        const float angle = (axis == 0u ? pos_x : pos_y) * inv_freq[p];
        const float c = precise::cos(angle);
        const float s = precise::sin(angle);
        dst[base + p] = (OT)(x0 * c - x1 * s);
        dst[base + p + half_w] = (OT)(x1 * c + x0 * s);
    }
}

/// One simdgroup's unit of `vision_qkv_norm_rope_*`: Q/K norm + axial RoPE, the
/// weightless V norm and the K/V store, straight from the fused projection
/// `[q | k | v]` (columns `q_off`, `k_off`, `v_off`, rows `ld_p` apart; head
/// `h` of each at `h * D`). Shared by the f32 and bf16 kernels so they cannot
/// drift apart; `OT` is the stored type of q, k and v.
///
/// Row `r = b * T + t` of the projection is patch `t` of image `b`, at
/// `positions[2r] = x`, `positions[2r + 1] = y`. q goes to `q_out`
/// `[B*T, Hq, D]`; k and v to the caches `[B, kv_capacity, Hkv, D]` at slot `t`
/// (the layout `encoder_attn` reads); a token whose slot is past `kv_capacity`
/// is skipped. One simdgroup per (row, head) over Hq query, Hkv key and Hkv
/// value heads.
template <typename OT>
inline void vision_qkv_norm_rope_unit(
    device const float *p,
    device const float *q_norm_w,
    device const float *k_norm_w,
    device OT *q_out,
    device OT *k_cache,
    device OT *v_cache,
    device const int *positions,
    uint B, uint T, uint Hq, uint Hkv, uint D,
    uint ld_p, uint q_off, uint k_off, uint v_off,
    uint kv_capacity, constant float *inv_freq, float eps,
    uint tg, uint sg, uint lane, uint tptg)
{
    const ulong heads = (ulong)Hq + 2ul * Hkv;
    const ulong unit = (ulong)tg * (tptg / 32u) + sg;
    // Uniform per simdgroup; no threadgroup barrier follows.
    if (unit >= (ulong)B * T * heads) return;
    const ulong r = unit / heads;
    const uint j = (uint)(unit % heads);
    const ulong b = r / T;
    const ulong t = r % T;
    if (t >= (ulong)kv_capacity) return;
    device const float *row = p + r * (ulong)ld_p;
    const float pos_x = (float)positions[2ul * r];
    const float pos_y = (float)positions[2ul * r + 1ul];

    if (j < Hq) {
        vision_norm_rope_row(row + q_off + (ulong)j * D, q_norm_w, q_out + (r * Hq + j) * (ulong)D,
                             D, pos_x, pos_y, inv_freq, eps, lane);
        return;
    }
    const uint h = j < Hq + Hkv ? j - Hq : j - Hq - Hkv;
    const ulong slot = ((b * kv_capacity + t) * Hkv + h) * (ulong)D;
    if (j < Hq + Hkv) {
        vision_norm_rope_row(row + k_off + (ulong)h * D, k_norm_w, k_cache + slot,
                             D, pos_x, pos_y, inv_freq, eps, lane);
    } else {
        device const float *src = row + v_off + (ulong)h * D;
        float ss = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            ss += src[d] * src[d];
        }
        const float inv = rsqrt(simd_sum(ss) / (float)D + eps);
        for (uint d = lane; d < D; d += 32u) {
            v_cache[slot + d] = (OT)(src[d] * inv);
        }
    }
}

#define VISION_QKV_KERNEL(NAME, OUT_T) \
kernel void NAME( \
    device const float *p [[buffer(0)]], \
    device const float *q_norm_w [[buffer(1)]], \
    device const float *k_norm_w [[buffer(2)]], \
    device OUT_T *q_out [[buffer(3)]], \
    device OUT_T *k_cache [[buffer(4)]], \
    device OUT_T *v_cache [[buffer(5)]], \
    device const int *positions [[buffer(6)]], \
    constant uint &B [[buffer(7)]], \
    constant uint &T [[buffer(8)]], \
    constant uint &Hq [[buffer(9)]], \
    constant uint &Hkv [[buffer(10)]], \
    constant uint &D [[buffer(11)]], \
    constant uint &ld_p [[buffer(12)]], \
    constant uint &q_off [[buffer(13)]], \
    constant uint &k_off [[buffer(14)]], \
    constant uint &v_off [[buffer(15)]], \
    constant uint &kv_capacity [[buffer(16)]], \
    constant float *inv_freq [[buffer(17)]], \
    constant float &eps [[buffer(18)]], \
    uint tg [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint tptg [[threads_per_threadgroup]]) \
{ \
    vision_qkv_norm_rope_unit(p, q_norm_w, k_norm_w, q_out, k_cache, v_cache, \
                              positions, B, T, Hq, Hkv, D, ld_p, q_off, k_off, v_off, kv_capacity, \
                              inv_freq, eps, tg, sg, lane, tptg); \
}

VISION_QKV_KERNEL(vision_qkv_norm_rope_f32, float)

/// [`vision_qkv_norm_rope_f32`] storing q, k and v as bf16 (rounded to nearest
/// even from the f32 result of the norm and RoPE, the one rounding a cast pass
/// would apply, with no pass): the operands of the bf16 encoder attention
/// (`encoder_attn_tiled_bf16_*`). Same slots and contract; `q_out`,
/// `k_cache` and `v_cache` are bf16 buffers.
VISION_QKV_KERNEL(vision_qkv_norm_rope_bf16, bfloat)

/// `h[r, :] += table[0][x_r] + table[1][y_r]` for a patch at `positions[2r] = x`,
/// `positions[2r + 1] = y`, and `h[r, :] = +0` for a padding patch (both
/// coordinates -1: transformers zeroes its embedding, and the host zeroes its
/// pixels, so the row is exactly +0 either way and stays so through the
/// tower's linear layers). `table`: `[2, table_len, D]` bf16, widened exactly;
/// the sum `x_emb + y_emb` is formed first, then added to the projection, as
/// `hidden_states + position_embeddings`. An index outside `[0, table_len)` is
/// clamped (the host refuses it; this keeps a corrupted buffer in bounds).
///
/// Grid: x = column in [0, D), y = row.
kernel void vision_pos_embed_add_f32(
    device float *h [[buffer(0)]],
    device const bfloat *table [[buffer(1)]],
    device const int *positions [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    constant uint &table_len [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= D || r >= rows) return;
    const ulong o = (ulong)r * D + c;
    const int x = positions[2ul * r];
    const int y = positions[2ul * r + 1ul];
    if (x == -1 && y == -1) {
        h[o] = 0.0f;
        return;
    }
    const int last = (int)table_len - 1;
    const ulong xi = (ulong)clamp(x, 0, last);
    const ulong yi = (ulong)clamp(y, 0, last);
    const float pe = float(table[xi * D + c]) + float(table[((ulong)table_len + yi) * D + c]);
    h[o] = h[o] + pe;
}

/// `out[s, :] = (sum over j in [offsets[s], offsets[s + 1]) of h[members[j], :] *
/// weight) * scale`: output row `s` of the pooler is the sum of its cell's
/// patch rows, each weighted `1 / k^2` (`weights = one_hot(cell) / k_squared`,
/// `weights^T @ hidden`), then multiplied by `sqrt(hidden)`. The members are in
/// ascending patch order, so the sum is sequential and does not depend on the
/// launch shape; a cell with no members is exactly +0. The host builds the
/// cells (`embedgemma2_vision::plan_pooling`); the kernel clamps `offsets` to
/// `n_members` and skips a member row at or past `rows`.
///
/// Grid: x = column in [0, D), y = output row.
kernel void vision_pool_members_f32(
    device const float *h [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uint *members [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &S [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &D [[buffer(6)]],
    constant uint &n_members [[buffer(7)]],
    constant float &weight [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint s = gid.y;
    if (c >= D || s >= S) return;
    const uint end = min(offsets[s + 1u], n_members);
    const uint start = min(offsets[s], end);
    float acc = 0.0f;
    for (uint i = start; i < end; ++i) {
        const uint m = members[i];
        if (m < rows) {
            acc += h[(ulong)m * D + c] * weight;
        }
    }
    out[(ulong)s * D + c] = acc * scale;
}
