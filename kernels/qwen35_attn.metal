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
        // torch computes inv_freq, the angle and cos/sin in fp32. `precise::`
        // throughout: the angle reaches tens of thousands of radians, where the
        // fast approximations lose whole digits.
        const float inv_freq =
            precise::divide(1.0f, precise::pow(theta, precise::divide((float)(2u * p), (float)rotary_dim)));
        const float angle = (float)pos * inv_freq;
        const float c = precise::cos(angle);
        const float s = precise::sin(angle);
        dst[p] = x0 * c - x1 * s;
        dst[p + half_rot] = x1 * c + x0 * s;
    }
    for (uint d = rotary_dim + lane; d < D; d += 32u) {
        dst[d] = src[d] * inv * (1.0f + weight[d]);
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
/// Hkv, D] at slot `pos_offset + t` — the layouts `nn::flash_attn_rows` reads.
/// RoPE position is also `pos_offset + t`. The gate is left where it is, for
/// `qwen35_attn_gate_*`.
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
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const ulong heads = (ulong)Hq + 2ul * Hkv;
    const ulong unit = (ulong)tg * (tptg / 32u) + sg;
    // Uniform per simdgroup; no threadgroup barrier follows.
    if (unit >= (ulong)B * T * heads) return;
    const ulong r = unit / heads;
    const uint j = (uint)(unit % heads);
    const ulong b = r / T;
    const uint t = (uint)(r % T);
    const uint pos = pos_offset + t;
    // The host checks this too; a stale bind must not write past the cache.
    if (pos >= kv_capacity) return;
    device const float *row = p + r * (ulong)ld_p;

    if (j < Hq) {
        qwen35_norm_rope_row(row + q_off + (ulong)j * 2u * D, q_norm_w,
                             q_out + (r * Hq + j) * (ulong)D,
                             D, rotary_dim, pos, theta, eps, lane);
        return;
    }
    const uint h = j < Hq + Hkv ? j - Hq : j - Hq - Hkv;
    const ulong slot = (((ulong)b * kv_capacity + pos) * Hkv + h) * (ulong)D;
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

/// `sigmoid(x)` without forming `e^|x|`, which fast math may assume finite.
inline float attn_sigmoid(float x)
{
    const float e = exp(-fabs(x));
    const float r = 1.0f / (1.0f + e);
    return x >= 0.0f ? r : e * r;
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
    out[(ulong)r * ld_out + out_off + col] = (OUT_T)(a * attn_sigmoid(g));       \
}

ATTN_GATE_KERNEL(qwen35_attn_gate_f32, float)
ATTN_GATE_KERNEL(qwen35_attn_gate_bf16, bfloat)
