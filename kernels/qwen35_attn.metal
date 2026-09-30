// Qwen3.5 full-attention extras: the parts of `Qwen3_5Attention.forward` that
// tessl's existing `rms_qkv_rope` / flash-attention path does not match.
//
//   P          = x @ [W_q | W_k | W_v]        (one fused GEMM; W_q is 2*Hq*D wide)
//   q, gate    = split(P_q.view(T, Hq, 2D))   (per head: D query, then D gate)
//   q          = rms_norm(q) * (1 + w_q)      (zero-centred weight)
//   k          = rms_norm(k) * (1 + w_k)
//   q, k       = partial NeoX RoPE on the first `rotary_dim` dims
//   o          = attention(q, k, v)           -> tessl's flash_attn_rows / decode
//   o          = o * sigmoid(gate)            -> qwen35_attn_gate_*
//
// Three things differ from `rms_qkv_rope`, and each would be a silent wrong
// answer rather than an error:
//
//   * **Partial rotary pairs.** transformers rotates `x[..rotary_dim]` with
//     `rotate_half` *of that slice*, pairing p with p + rotary_dim/2, and takes
//     inv_freq = theta^(-2p / rotary_dim). `rms_qkv_rope` implements Gemma's
//     proportional RoPE, pairing p with p + D/2 over the full head and dividing
//     by D. At Qwen3.5's rotary_dim = 64 of D = 256 the two agree on nothing.
//   * **Zero-centred norm weight.** `Qwen3_5RMSNorm` multiplies by (1 + w), with
//     w initialised to zero. Passing its weights to a `* w` norm zeroes q and k.
//   * **The output gate.** q_proj is twice as wide as the attention output; its
//     second half per head gates the attention output through a sigmoid.
//
// Qwen3.5 carries mRoPE sections for its vision tower. For text the three
// position streams are equal, and the interleaved recomposition then selects
// the same frequency it replaces, so the rotation is plain RoPE at position
// `pos`. Mixed image/text positions are out of scope here.
#include <metal_stdlib>
#include "qwen35_act.h"
using namespace metal;

/// Normalize one head row with a zero-centred weight and apply transformers'
/// partial RoPE, reading `src` and writing `dst` (never the same row).
inline void qwen35_norm_rope_row(
    device const float *src,
    device const float *weight,
    device float *dst,
    uint D,
    uint rotary_dim,
    uint pos,
    float theta,
    float eps,
    uint lane)
{
    float ss = 0.0f;
    for (uint d = lane; d < D; d += 32u) {
        ss += src[d] * src[d];
    }
    const float inv = rsqrt(simd_sum(ss) / (float)D + eps);

    const uint half_rot = rotary_dim / 2u;
    for (uint p = lane; p < half_rot; p += 32u) {
        const float x0 = src[p] * inv * (1.0f + weight[p]);
        const float x1 = src[p + half_rot] * inv * (1.0f + weight[p + half_rot]);
        const float angle = qwen35_rope_angle(p, rotary_dim, pos, theta);
        const float c = precise::cos(angle);
        const float s = precise::sin(angle);
        dst[p] = x0 * c - x1 * s;
        dst[p + half_rot] = x1 * c + x0 * s;
    }
    for (uint d = rotary_dim + lane; d < D; d += 32u) {
        dst[d] = src[d] * inv * (1.0f + weight[d]);
    }
}

/// One simdgroup's unit of `qwen35_attn_qk_norm_rope*`: a (token, head) row
/// over Hq query, Hkv key and Hkv value heads. Shared by the scalar and the
/// device-buffer position variants so they cannot drift apart.
///
/// The position is formed in 64 bits and checked against the capacity before
/// anything is written: `pos_offset + t` in 32 bits could wrap to a small slot
/// and pass the check, and in the `_posbuf` variant the offset is device data
/// the host never sees.
///
/// Row b's offset is `pos_ptr[b * pos_stride]` when `pos_ptr` is given (the
/// `_posbuf` kernel; stride 0 shares one offset, 1 gives each row its own, for
/// ragged continuations) and `pos_scalar` otherwise.
inline void qk_norm_rope_unit(
    device const float *p,
    device const float *q_norm_w,
    device const float *k_norm_w,
    device float *q_out,
    device float *k_cache,
    device float *v_cache,
    uint B, uint T, uint Hq, uint Hkv, uint D, uint rotary_dim,
    uint ld_p, uint q_off, uint k_off, uint v_off,
    device const uint *pos_ptr, uint pos_stride, ulong pos_scalar,
    uint slot_base, uint kv_capacity, float theta, float eps,
    uint tg, uint sg, uint lane, uint tptg)
{
    const ulong heads = (ulong)Hq + 2ul * Hkv;
    const ulong unit = (ulong)tg * (tptg / 32u) + sg;
    // Uniform per simdgroup; no threadgroup barrier follows.
    if (unit >= (ulong)B * T * heads) return;
    const ulong r = unit / heads;
    const uint j = (uint)(unit % heads);
    const ulong b = r / T;
    const ulong pos_offset = pos_ptr ? (ulong)pos_ptr[b * pos_stride] : pos_scalar;
    const ulong pos64 = pos_offset + (r % T);
    // The cache slot is the position less `slot_base`: 0 for a cache that
    // starts at position 0, the shared prefix's length for a suffix cache
    // (see qwen35_attn_prefix_rows). RoPE still uses the absolute position.
    if (pos64 < (ulong)slot_base || pos64 - slot_base >= (ulong)kv_capacity) return;
    const uint pos = (uint)pos64;
    const ulong cache_pos = pos64 - slot_base;
    device const float *row = p + r * (ulong)ld_p;

    if (j < Hq) {
        qwen35_norm_rope_row(row + q_off + (ulong)j * 2u * D, q_norm_w,
                             q_out + (r * Hq + j) * (ulong)D,
                             D, rotary_dim, pos, theta, eps, lane);
        return;
    }
    const uint h = j < Hq + Hkv ? j - Hq : j - Hq - Hkv;
    const ulong slot = (((ulong)b * kv_capacity + cache_pos) * Hkv + h) * (ulong)D;
    if (j < Hq + Hkv) {
        qwen35_norm_rope_row(row + k_off + (ulong)h * D, k_norm_w, k_cache + slot,
                             D, rotary_dim, pos, theta, eps, lane);
    } else {
        device const float *src = row + v_off + (ulong)h * D;
        for (uint d = lane; d < D; d += 32u) {
            v_cache[slot + d] = src[d];
        }
    }
}

/// Q/K norm + partial RoPE, and the K/V cache store, straight from the fused
/// projection output.
///
/// Row `r = b*T + t` of `p` holds, at column offsets:
///   `q_off + h*2D`        query head h (D), followed by its gate (D)
///   `k_off + h*D`         key head h
///   `v_off + h*D`         value head h
///
/// Writes q to `q_out` [B, T, Hq, D] and k/v to the caches [B, kv_capacity,
/// Hkv, D] at slot `pos_offset + t - slot_base` — the layouts
/// `nn::flash_attn_rows` (slot_base 0) and `qwen35_attn_prefix_rows`' suffix
/// cache (slot_base = the shared prefix's length) read. RoPE position is
/// `pos_offset + t`, always absolute. The gate is left where it is, for
/// `qwen35_attn_gate_*`. A token whose slot falls outside `[0, kv_capacity)`
/// is skipped entirely.
///
/// One simdgroup per (token, head) over Hq query, Hkv key and Hkv value heads.
kernel void qwen35_attn_qk_norm_rope(
    device const float *p [[buffer(0)]],
    device const float *q_norm_w [[buffer(1)]],
    device const float *k_norm_w [[buffer(2)]],
    device float *q_out [[buffer(3)]],
    device float *k_cache [[buffer(4)]],
    device float *v_cache [[buffer(5)]],
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &Hq [[buffer(8)]],
    constant uint &Hkv [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant uint &rotary_dim [[buffer(11)]],
    constant uint &ld_p [[buffer(12)]],
    constant uint &q_off [[buffer(13)]],
    constant uint &k_off [[buffer(14)]],
    constant uint &v_off [[buffer(15)]],
    constant uint &pos_offset [[buffer(16)]],
    constant uint &kv_capacity [[buffer(17)]],
    constant float &theta [[buffer(18)]],
    constant float &eps [[buffer(19)]],
    constant uint &slot_base [[buffer(20)]],
    constant uint &pos_stride [[buffer(21)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    qk_norm_rope_unit(p, q_norm_w, k_norm_w, q_out, k_cache, v_cache, B, T, Hq, Hkv, D,
                      rotary_dim, ld_p, q_off, k_off, v_off, nullptr, pos_stride, (ulong)pos_offset,
                      slot_base, kv_capacity,
                      theta, eps, tg, sg, lane, tptg);
}

/// [`qwen35_attn_qk_norm_rope`] with the position offset read from a device
/// buffer (`*pos_offset_ptr`) instead of bound as a scalar, like
/// `rms_qkv_rope_posbuf`. A decode loop replayed from an Indirect Command
/// Buffer freezes its scalar binds; with the position in a buffer the host
/// advances between replays, the same recording rotates and caches each step
/// at its own position.
kernel void qwen35_attn_qk_norm_rope_posbuf(
    device const float *p [[buffer(0)]],
    device const float *q_norm_w [[buffer(1)]],
    device const float *k_norm_w [[buffer(2)]],
    device float *q_out [[buffer(3)]],
    device float *k_cache [[buffer(4)]],
    device float *v_cache [[buffer(5)]],
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &Hq [[buffer(8)]],
    constant uint &Hkv [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant uint &rotary_dim [[buffer(11)]],
    constant uint &ld_p [[buffer(12)]],
    constant uint &q_off [[buffer(13)]],
    constant uint &k_off [[buffer(14)]],
    constant uint &v_off [[buffer(15)]],
    device const uint *pos_offset_ptr [[buffer(16)]],
    constant uint &kv_capacity [[buffer(17)]],
    constant float &theta [[buffer(18)]],
    constant float &eps [[buffer(19)]],
    constant uint &slot_base [[buffer(20)]],
    constant uint &pos_stride [[buffer(21)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    qk_norm_rope_unit(p, q_norm_w, k_norm_w, q_out, k_cache, v_cache, B, T, Hq, Hkv, D,
                      rotary_dim, ld_p, q_off, k_off, v_off, pos_offset_ptr, pos_stride, 0ul,
                      slot_base, kv_capacity,
                      theta, eps, tg, sg, lane, tptg);
}

/// `out = attn * sigmoid(gate)`, with the gate read in place from the fused
/// projection (head h's gate is columns `q_off + h*2D + D ..`).
///
/// `attn` is the attention output [rows, Hq*D]; `out` is `[rows, ld_out]` at
/// `out_off`, f32 or bf16 so it can feed the `o_proj` GEMM directly. The f32
/// variant may write in place over `attn` (same index, read before write).
///
/// Grid: x = column in [0, Hq*D), y = row.
#define ATTN_GATE_KERNEL(NAME, OUT_T)                                             \
kernel void NAME(                                                                 \
    device const float *attn [[buffer(0)]],                                       \
    device const float *p [[buffer(1)]],                                          \
    device OUT_T *out [[buffer(2)]],                                              \
    constant uint &rows [[buffer(3)]],                                            \
    constant uint &Hq [[buffer(4)]],                                              \
    constant uint &D [[buffer(5)]],                                               \
    constant uint &ld_p [[buffer(6)]],                                            \
    constant uint &q_off [[buffer(7)]],                                           \
    constant uint &ld_out [[buffer(8)]],                                          \
    constant uint &out_off [[buffer(9)]],                                         \
    uint2 gid [[thread_position_in_grid]])                                        \
{                                                                                 \
    const uint col = gid.x;                                                       \
    const uint r = gid.y;                                                         \
    if (col >= Hq * D || r >= rows) return;                                       \
    const uint h = col / D;                                                       \
    const uint d = col % D;                                                       \
    const float g = p[(ulong)r * ld_p + q_off + (ulong)h * 2u * D + D + d];       \
    const float a = attn[(ulong)r * Hq * D + col];                                \
    out[(ulong)r * ld_out + out_off + col] = (OUT_T)(a * qwen35_sigmoid(g));       \
}

ATTN_GATE_KERNEL(qwen35_attn_gate_f32, float)
ATTN_GATE_KERNEL(qwen35_attn_gate_bf16, bfloat)

// ---------------------------------------------------- shared-prefix attention

/// Head dim, lanes per query row and simdgroups per threadgroup of
/// `qwen35_attn_prefix_rows`: exactly the instantiation `nn::flash_attn_rows`
/// picks at D = 256 (`rows_lanes_for` / `rows_groups_for`), so the two run the
/// same per-row arithmetic. The host mirrors these.
constant uint PREFIX_ATTN_D = 256;
constant uint PREFIX_ATTN_R = 16;
constant uint PREFIX_ATTN_SGT = 32;

/// Causal flash attention over a **shared prefix** plus a per-row suffix: many
/// questions asked of one prefilled context without copying its K/V per row.
///
///   prefix K/V  `[prefix_cap, Hkv, D]`      no batch dimension (stride 0);
///                                           key `t` is at position `t`
///   suffix K/V  `[B, suffix_cap, Hkv, D]`   row b's own continuation; slot `s`
///                                           is at position `P + s`
///   Q, O        `[B, Tq, H, D]`             query `t` at `q_pos_offset + t`
///
/// Row b attends to keys `[0, P)` of the prefix and `[0, S)` of its own suffix,
/// `S = min(*suffix_len_ptr, suffix_cap)`, keeping positions `<=` the query's.
/// That is `flash_attn_rows_h256_r16_g32` with `window = 0` over a per-row
/// cache `prefix ‖ suffix_b`, and the body below is that kernel's, line for
/// line: only the address of key `t` differs. A masked key is an exact no-op
/// in the online softmax (alpha = 1, p = 0), so visiting the unmasked keys in
/// the same ascending order gives the same bits. A row with nothing unmasked
/// is zeros, not NaN.
///
/// The host guarantees `P + suffix_cap` fits `u32`, so key indices do too.
///
/// Grid: x = ceil(Tq / (SGT * 32/R)), y = B * H; SGT * 32 threads.
kernel void qwen35_attn_prefix_rows(
    device const float *Q [[buffer(0)]],
    device const float *Kp [[buffer(1)]],
    device const float *Vp [[buffer(2)]],
    device const float *Ks [[buffer(3)]],
    device const float *Vs [[buffer(4)]],
    device float *O [[buffer(5)]],
    constant uint &Tq [[buffer(6)]],
    constant uint &P [[buffer(7)]],
    device const uint *suffix_len_ptr [[buffer(8)]],
    constant uint &H [[buffer(9)]],
    constant uint &Hkv [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    device const uint *q_pos_offset_ptr [[buffer(12)]],
    constant uint &out_bf16 [[buffer(13)]],
    constant uint &suffix_cap [[buffer(14)]],
    constant uint &row_stride [[buffer(15)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]])
{
    constexpr uint D = PREFIX_ATTN_D;
    constexpr uint R = PREFIX_ATTN_R;
    constexpr uint RPS = 32u / R;               // query rows per simdgroup
    constexpr uint DPV = D / (4u * R);          // float4s per lane
    constexpr uint RPT = PREFIX_ATTN_SGT * RPS; // query rows per threadgroup
    static_assert(D % (4u * R) == 0u, "float4 lane slices must tile the head");
    const uint tid = tpitg.x;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint sub = lane / R;
    const uint dl = lane % R;

    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    // Clamp mutable device state before it participates in any address. With
    // row_stride 1 each row has its own suffix length and query position.
    const uint S = min(suffix_len_ptr[b * row_stride], suffix_cap);
    const ulong Tkv = (ulong)P + S;
    const ulong base_row = (ulong)tgpig.x * RPT + sg * RPS;
    // Uniform across the simdgroup: the butterfly needs every lane.
    if (base_row >= (ulong)Tq) { return; }
    const uint live_rows = min(RPS, (uint)((ulong)Tq - base_row));
    const bool row_live = sub < live_rows;
    const uint t_q = row_live ? (uint)(base_row + sub) : (uint)base_row;

    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;
    const ulong kv_pos_stride = (ulong)Hkv * D;
    // The prefix has no batch term: every row reads the same K/V.
    const ulong prefix_head_base = (ulong)hkv * D;
    const ulong suffix_head_base = (ulong)b * suffix_cap * kv_pos_stride + (ulong)hkv * D;
    const ulong q_pos_stride = (ulong)H * D;
    const ulong q_head_base = (ulong)b * Tq * q_pos_stride + (ulong)h * D;
    const ulong q_off_i = (ulong)q_pos_offset_ptr[b * row_stride];
    const ulong q_abs = q_off_i + (ulong)t_q;

    // Union key range over this simdgroup's rows; rows mask inside it.
    const ulong q_hi = q_off_i + base_row + live_rows - 1ul;
    const ulong t_end = min(Tkv, q_hi + 1ul);

    const ulong o_off = q_head_base + (ulong)t_q * q_pos_stride;
    float4 q_reg[DPV];
    float4 acc[DPV];
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    if (0ul < t_end) {
        device const float4 *Q4 = (device const float4 *)(Q + o_off);
        for (uint j = 0; j < DPV; ++j) {
            q_reg[j] = row_live ? Q4[dl + j * R] : float4(0.0f);
        }
        for (uint t = 0; t < (uint)t_end; ++t) {
            // The one change from flash_attn_rows: key t is the shared
            // prefix's below P and this row's suffix from P on. Uniform per
            // simdgroup, so the select never diverges.
            const bool in_prefix = t < P;
            const ulong kv_base = in_prefix
                ? prefix_head_base + (ulong)t * kv_pos_stride
                : suffix_head_base + (ulong)(t - P) * kv_pos_stride;
            device const float *Kb = in_prefix ? Kp : Ks;
            device const float *Vb = in_prefix ? Vp : Vs;
            device const float4 *K4 = (device const float4 *)(Kb + kv_base);
            float4 dot4 = float4(0.0f);
            for (uint j = 0; j < DPV; ++j) {
                dot4 += q_reg[j] * K4[dl + j * R];
            }
            float part = dot4.x + dot4.y + dot4.z + dot4.w;
            for (uint off = R / 2u; off > 0u; off >>= 1) {
                part += simd_shuffle_xor(part, off);
            }
            float s = part * scale;
            if (!row_live || (ulong)t > q_abs) {
                s = -INFINITY;
            }
            const float m_new = max(m_i, s);
            const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);
            const float p = (s == -INFINITY) ? 0.0f : exp(s - m_new);
            device const float4 *V4 = (device const float4 *)(Vb + kv_base);
            for (uint j = 0; j < DPV; ++j) {
                acc[j] = acc[j] * alpha + p * V4[dl + j * R];
            }
            l_i = l_i * alpha + p;
            m_i = (m_new == -INFINITY) ? -INFINITY : m_new;
        }
    }

    if (!row_live) { return; }
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
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

/// Keys per chunk and lanes per key of `qwen35_attn_prefix_decode_*`: the
/// `flash_attn_decode` instantiation `nn` picks at D = 256
/// (`decode_chunk_for` / `decode_lanes_for`). The host mirrors these.
constant uint PREFIX_DECODE_CHUNK = 128;
constant uint PREFIX_DECODE_R = 16;

/// Live keys of row b: the prefix, then its suffix clamped to the capacity.
/// Both decode passes derive the row's live chunk count from this.
inline uint prefix_decode_tkv(uint P, device const uint *suffix_len_ptr, uint b, uint row_stride,
                              uint suffix_cap)
{
    return P + min(suffix_len_ptr[b * row_stride], suffix_cap);
}

/// Chunks per (row, head) in the scratch layout: the capacity's count, not the
/// live one. Rows may have different live lengths, and a live-count stride
/// would lay one row's chunks over the next row's.
inline uint prefix_decode_chunk_stride(uint P, uint suffix_cap)
{
    const uint cap = P + suffix_cap;
    return max(cap / PREFIX_DECODE_CHUNK + ((cap % PREFIX_DECODE_CHUNK) != 0u ? 1u : 0u), 1u);
}

/// Single-query attention over a shared prefix plus per-row suffix, split over
/// the keys: partial pass. The layouts are `qwen35_attn_prefix_rows`' with
/// Tq = 1, and the body is `flash_attn_decode_partial_h256_c128_r16`'s (window
/// 0, kv_pos_offset 0) with only the address of key `t` changed, so each
/// chunk's (m, l, acc) is bit-identical to that kernel's over a copied prefix.
///
/// Grid: x = ceil((P + suffix_cap) / CHUNK), y = B * H / sgs; sgs simdgroups,
/// where sgs query heads share a threadgroup (the host's GQA-group choice).
kernel void qwen35_attn_prefix_decode_partial(
    device const float *Q [[buffer(0)]],
    device const float *Kp [[buffer(1)]],
    device const float *Vp [[buffer(2)]],
    device const float *Ks [[buffer(3)]],
    device const float *Vs [[buffer(4)]],
    device float *partials [[buffer(5)]],
    constant uint &P [[buffer(6)]],
    device const uint *suffix_len_ptr [[buffer(7)]],
    constant uint &H [[buffer(8)]],
    constant uint &Hkv [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    device const uint *q_pos_offset_ptr [[buffer(11)]],
    constant uint &suffix_cap [[buffer(12)]],
    constant uint &row_stride [[buffer(13)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]])
{
    constexpr uint D = PREFIX_ATTN_D;
    constexpr uint CH = PREFIX_DECODE_CHUNK;
    constexpr uint R = PREFIX_DECODE_R;
    constexpr uint DPV = D / (4u * R);
    constexpr uint KPG = 32u / R;               // keys in flight per simdgroup
    static_assert(D % (4u * R) == 0u, "float4 lane slices must tile the head");
    const uint sgs = max(tptg.x / 32u, 1u);
    const uint sg = tpitg.x / 32u;
    const uint lane = tpitg.x % 32u;
    const uint grp = lane / R;
    const uint dl = lane % R;
    const uint chunk = tgpig.x;
    const uint blocks = max(H / sgs, 1u);
    const uint b = tgpig.y / blocks;
    // Clamp mutable device state before it participates in any address.
    const uint Tkv = prefix_decode_tkv(P, suffix_len_ptr, b, row_stride, suffix_cap);
    const uint hb = tgpig.y % blocks;
    const uint h = hb * sgs + sg;
    if (h >= H) { return; }
    const ulong bh = (ulong)b * H + h;
    const uint t_k0 = chunk * CH;
    if (t_k0 >= Tkv) { return; }
    const uint n_k = min(CH, Tkv - t_k0);
    const uint group = max(H / Hkv, 1u);
    const uint hkv = h / group;
    const ulong kv_pos_stride = (ulong)Hkv * D;
    const ulong prefix_head_base = (ulong)hkv * D;
    const ulong suffix_head_base = (ulong)b * suffix_cap * kv_pos_stride + (ulong)hkv * D;

    const ulong q_abs = (ulong)q_pos_offset_ptr[b * row_stride];
    const ulong q_off = bh * D;
    device const float4 *Q4 = (device const float4 *)(Q + q_off);
    float4 q_reg[DPV];
    for (uint j = 0; j < DPV; ++j) { q_reg[j] = Q4[dl + j * R]; }

    float4 acc[DPV];
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Live key sub-range of this chunk: [0, q_abs] is the causal rule.
    const ulong local_hi = q_abs + 1ul;
    const ulong lo_i = (ulong)t_k0;
    const ulong hi_i = min((ulong)t_k0 + n_k, local_hi);
    const uint stride0 = D + 2u;
    const ulong base0 = (bh * prefix_decode_chunk_stride(P, suffix_cap) + chunk) * stride0;
    if (lo_i >= hi_i) {
        // Chunk fully masked; uniform across the simdgroup.
        if (lane == 0u) {
            partials[base0] = -INFINITY;
            partials[base0 + 1u] = 0.0f;
        }
        for (uint d = lane; d < D; d += 32u) {
            partials[base0 + 2u + d] = 0.0f;
        }
        return;
    }
    const uint live = (uint)(hi_i - lo_i);
    const uint iters = (live + KPG - 1u) / KPG;
    for (uint it = 0; it < iters; ++it) {
        const uint t = it * KPG + grp;
        const uint tt = min(t, live - 1u);
        const uint key = (uint)lo_i + tt;
        // The one change from flash_attn_decode: key is the shared prefix's
        // below P and this row's suffix from P on.
        const bool in_prefix = key < P;
        const ulong kv_base = in_prefix
            ? prefix_head_base + (ulong)key * kv_pos_stride
            : suffix_head_base + (ulong)(key - P) * kv_pos_stride;
        device const float *Kb = in_prefix ? Kp : Ks;
        device const float *Vb = in_prefix ? Vp : Vs;
        device const float4 *K4 = (device const float4 *)(Kb + kv_base);
        float4 dot4 = float4(0.0f);
        for (uint j = 0; j < DPV; ++j) {
            dot4 += q_reg[j] * K4[dl + j * R];
        }
        float part = dot4.x + dot4.y + dot4.z + dot4.w;
        for (uint off = R / 2u; off > 0u; off >>= 1) {
            part += simd_shuffle_xor(part, off);
        }
        float s = part * scale;
        if (t >= live) { s = -INFINITY; }
        const float m_new = max(m_i, s);
        const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);
        const float p = (s == -INFINITY) ? 0.0f : exp(s - m_new);
        device const float4 *V4 = (device const float4 *)(Vb + kv_base);
        for (uint j = 0; j < DPV; ++j) {
            acc[j] = acc[j] * alpha + p * V4[dl + j * R];
        }
        l_i = l_i * alpha + p;
        m_i = m_new;
    }

    // Combine the KPG key-groups (lanes R apart share a dim slice).
    float m_all = m_i;
    for (uint off = R; off < 32u; off <<= 1) {
        m_all = max(m_all, simd_shuffle_xor(m_all, off));
    }
    const float w = (m_i == -INFINITY || m_all == -INFINITY) ? 0.0f : exp(m_i - m_all);
    float l_all = l_i * w;
    for (uint off = R; off < 32u; off <<= 1) {
        l_all += simd_shuffle_xor(l_all, off);
    }
    for (uint j = 0; j < DPV; ++j) {
        float4 a = acc[j] * w;
        for (uint off = R; off < 32u; off <<= 1) {
            a.x += simd_shuffle_xor(a.x, off);
            a.y += simd_shuffle_xor(a.y, off);
            a.z += simd_shuffle_xor(a.z, off);
            a.w += simd_shuffle_xor(a.w, off);
        }
        acc[j] = a;
    }
    if (grp != 0u) { return; }
    if (lane == 0u) { partials[base0] = m_all; partials[base0 + 1u] = l_all; }
    for (uint j = 0; j < DPV; ++j) {
        const uint d0 = 4u * (dl + j * R);
        partials[base0 + 2u + d0 + 0u] = acc[j].x;
        partials[base0 + 2u + d0 + 1u] = acc[j].y;
        partials[base0 + 2u + d0 + 2u] = acc[j].z;
        partials[base0 + 2u + d0 + 3u] = acc[j].w;
    }
}

/// Reduce pass of the shared-prefix decode: `flash_attn_decode_reduce_h256_c128`
/// with the live key count `P + min(S, suffix_cap)`, the same one the partial
/// pass wrote chunks for. Output `[B, H, D]`, f32 or bf16.
///
/// Grid: y = B * H; any threadgroup width (the output loop strides by it).
kernel void qwen35_attn_prefix_decode_reduce(
    device const float *partials [[buffer(0)]],
    device float *O [[buffer(1)]],
    constant uint &P [[buffer(2)]],
    device const uint *suffix_len_ptr [[buffer(3)]],
    constant uint &H [[buffer(4)]],
    constant uint &out_bf16 [[buffer(5)]],
    constant uint &suffix_cap [[buffer(6)]],
    constant uint &row_stride [[buffer(7)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]])
{
    constexpr uint D = PREFIX_ATTN_D;
    constexpr uint CH = PREFIX_DECODE_CHUNK;
    const uint lid = tpitg.x;
    const uint width = max(tptg.x, 1u);
    const uint bh = tgpig.y;
    const uint Tkv = prefix_decode_tkv(P, suffix_len_ptr, bh / H, row_stride, suffix_cap);
    const uint n_chunks = Tkv / CH + ((Tkv % CH) != 0u ? 1u : 0u);
    const uint stride = D + 2u;
    const ulong chunk0 = (ulong)bh * prefix_decode_chunk_stride(P, suffix_cap);

    float m_all = -INFINITY;
    for (uint c = 0; c < n_chunks; ++c) {
        m_all = max(m_all, partials[(chunk0 + c) * stride]);
    }
    float l_all = 0.0f;
    if (m_all != -INFINITY) {
        for (uint c = 0; c < n_chunks; ++c) {
            const float m_c = partials[(chunk0 + c) * stride];
            if (m_c == -INFINITY) { continue; }
            l_all += partials[(chunk0 + c) * stride + 1u] * exp(m_c - m_all);
        }
    }
    const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
    const ulong o_off = (ulong)bh * D;
    device bfloat *Ob = (device bfloat *)O;
    for (uint d = lid; d < D; d += width) {
        float a = 0.0f;
        if (m_all != -INFINITY) {
            for (uint c = 0; c < n_chunks; ++c) {
                const ulong base = (chunk0 + c) * stride;
                const float m_c = partials[base];
                if (m_c == -INFINITY) { continue; }
                a += partials[base + 2u + d] * exp(m_c - m_all);
            }
        }
        const float o = a * inv_l;
        if (out_bf16 != 0u) { Ob[o_off + d] = bfloat(o); }
        else { O[o_off + d] = o; }
    }
}
