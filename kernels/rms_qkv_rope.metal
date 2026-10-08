// Fused QK-Norm → RoPE (and V-norm). Attention scale after QK-Norm is 1.0.
// V-norm is weight RMS only (no extra scale factor).
//
// Layout: q [T, Hq, D], k/v [T, Hkv, D], weights [D] (shared across heads).
// RoPE: proportional NeoX (MLX traditional=False); p-RoPE rotates first
// rotary_dim/2 pairs across dim/2 (inv-freq denom = full head_dim).
//
// One simdgroup per head row. Lane `l` owns the elements `p` and `p + D/2`
// for every `p ≡ l (mod 32)` below `D/2`, which is exactly one RoPE pair, so
// the rotation never needs a value another lane produced; lane 0 also owns
// the unpaired tail element when `D` is odd. Adjacent lanes touch adjacent
// addresses, and the sum of squares is a `simd_sum` rather than a serial
// walk. Each element is read twice (sum of squares, rewrite) and written
// once.
//
// These kernels were one *thread* per head row until 2026-09-04. At the
// decode shape that put the whole layer on `Hq + 2*Hkv` threads, each walking
// its row three times serially with strided (uncoalesced) neighbours, and at
// prefill it left the GPU with `T * (Hq + 2*Hkv)` threads of parallelism
// against rows of 128-256 elements. The simdgroup reduction reassociates the
// sum of squares where the serial loop did not — a change in the low bits,
// which is why `tests/qkv_rope.rs` compares against an f64 reference rather
// than a matching f32 accumulation.
#include <metal_stdlib>
using namespace metal;

constant uint ROPE_SIMD_WIDTH = 32u;

/// Normalize one head row in place and, when `rotate`, apply RoPE to its
/// first `rotary_dim` lanes. With `STORE`, every rewritten element is also
/// written to `dst` at the same index (the fused KV-cache store), straight
/// from the register that holds it, so no lane re-reads what another wrote.
template <bool STORE>
inline void norm_rope_row(
    device float *row,
    device const float *weight,
    device float *dst,
    uint D,
    uint rotary_dim,
    ulong pos,
    float theta,
    float eps,
    bool rotate,
    uint lane)
{
    float ss = 0.0f;
    for (uint d = lane; d < D; d += ROPE_SIMD_WIDTH) {
        const float v = row[d];
        ss += v * v;
    }
    ss = simd_sum(ss);
    // `eps` is what keeps an all-zero row finite: rsqrt(0) is inf.
    const float inv = rsqrt(ss / (float)D + eps);

    // Proportional NeoX / non-traditional RoPE (MLX `ProportionalRoPE` +
    // `nn.RoPE(traditional=False)`):
    //   pair x[p] with x[p + D/2] for p in [0, rotary_dim/2).
    // When rotary_dim == D (sliding), this is full NeoX over the head.
    // When rotary_dim < D (global p-RoPE), only the first rotary_dim/2
    // pairs rotate; the rest of the D/2 pairs stay unrotated (inf freq).
    // inv_freq denom uses full head `D`.
    const uint half_dim = D / 2;
    const uint n_pairs = rotate ? rotary_dim / 2 : 0u;
    for (uint p = lane; p < half_dim; p += ROPE_SIMD_WIDTH) {
        float x0 = row[p] * inv * weight[p];
        float x1 = row[p + half_dim] * inv * weight[p + half_dim];
        if (p < n_pairs) {
            const float inv_freq = 1.0f / pow(theta, (2.0f * (float)p) / (float)D);
            const float angle = (float)pos * inv_freq;
            const float c = cos(angle);
            const float s = sin(angle);
            const float r0 = x0 * c - x1 * s;
            const float r1 = x0 * s + x1 * c;
            x0 = r0;
            x1 = r1;
        }
        row[p] = x0;
        row[p + half_dim] = x1;
        if (STORE) {
            dst[p] = x0;
            dst[p + half_dim] = x1;
        }
    }
    if ((D & 1u) != 0u && lane == 0u) {
        // The odd tail element pairs with nothing and is never rotated.
        const float tail = row[D - 1u] * inv * weight[D - 1u];
        row[D - 1u] = tail;
        if (STORE) {
            dst[D - 1u] = tail;
        }
    }
}

/// Which head row this simdgroup owns. Threadgroups are a whole number of
/// simdgroups (the host dispatches `rows_per_tg * 32` threads), so the row is
/// uniform across the simdgroup and every `simd_sum` sees all 32 lanes.
inline ulong row_of(uint tg, uint sg, uint tptg)
{
    return (ulong)tg * (ulong)(tptg / ROPE_SIMD_WIDTH) + (ulong)sg;
}

/// One simdgroup per (token, head) row for Q; then K rows; then V rows.
kernel void rms_qkv_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    constant uint &pos_offset [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const ulong total_q = (ulong)T * (ulong)Hq;
    const ulong total_kv = (ulong)T * (ulong)Hkv;
    const ulong gid64 = row_of(tg, sg, tptg);
    if (gid64 >= total_q + 2ul * total_kv) return;

    if (gid64 < total_q) {
        const ulong t = gid64 / (ulong)Hq;
        const ulong h = gid64 % (ulong)Hq;
        device float *row = q + ((t * (ulong)Hq + h) * (ulong)D);
        norm_rope_row<false>(row, q_weight, row, D, rotary_dim,
                             (ulong)pos_offset + t, theta, eps, true, lane);
        return;
    }
    ulong g2 = gid64 - total_q;
    if (g2 < total_kv) {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = k + ((t * (ulong)Hkv + h) * (ulong)D);
        norm_rope_row<false>(row, k_weight, row, D, rotary_dim,
                             (ulong)pos_offset + t, theta, eps, true, lane);
        return;
    }
    g2 -= total_kv;
    {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = v + ((t * (ulong)Hkv + h) * (ulong)D);
        // V-norm: weight RMS only, no RoPE, no attn scale.
        norm_rope_row<false>(row, v_weight, row, D, rotary_dim,
                             0ul, theta, eps, false, lane);
    }
}

/// Encode-once prototype: RoPE pos from a stable device u32 buffer (written once
/// per decode/verify step) instead of a const-arena scalar rebound every layer.
/// Math identical to `rms_qkv_rope` with `pos_offset = *pos_offset_ptr`.
kernel void rms_qkv_rope_posbuf(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    device const uint *pos_offset_ptr [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const ulong pos_offset = (ulong)*pos_offset_ptr;
    const ulong total_q = (ulong)T * (ulong)Hq;
    const ulong total_kv = (ulong)T * (ulong)Hkv;
    const ulong gid64 = row_of(tg, sg, tptg);
    if (gid64 >= total_q + 2ul * total_kv) return;

    if (gid64 < total_q) {
        const ulong t = gid64 / (ulong)Hq;
        const ulong h = gid64 % (ulong)Hq;
        device float *row = q + ((t * (ulong)Hq + h) * (ulong)D);
        norm_rope_row<false>(row, q_weight, row, D, rotary_dim,
                             pos_offset + t, theta, eps, true, lane);
        return;
    }
    ulong g2 = gid64 - total_q;
    if (g2 < total_kv) {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = k + ((t * (ulong)Hkv + h) * (ulong)D);
        norm_rope_row<false>(row, k_weight, row, D, rotary_dim,
                             pos_offset + t, theta, eps, true, lane);
        return;
    }
    g2 -= total_kv;
    {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = v + ((t * (ulong)Hkv + h) * (ulong)D);
        norm_rope_row<false>(row, v_weight, row, D, rotary_dim,
                             0ul, theta, eps, false, lane);
    }
}

/// Layer-fusion: `rms_qkv_rope_posbuf` + `kv_store_timestep_pair` for producers.
/// After K/V norm(+RoPE), also write the timestep into the cache slot at
/// `kv_dst_offset`. Q stays scratch-only. Math identical to the two-pass path
/// (scratch first, then copy) — element-local, no grid sync.
/// Opt-in: `GEMMA_METAL_FUSE_ROPE_KV=1` / `GEMMA_METAL_FUSE_LAYER=1`.
kernel void rms_qkv_rope_kv_store(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_weight [[buffer(3)]],
    device const float *k_weight [[buffer(4)]],
    device const float *v_weight [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &Hq [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant uint &rotary_dim [[buffer(10)]],
    device const uint *pos_offset_ptr [[buffer(11)]],
    constant float &theta [[buffer(12)]],
    constant float &eps [[buffer(13)]],
    device float *dst_k [[buffer(14)]],
    device float *dst_v [[buffer(15)]],
    device const uint *kv_dst_offset_ptr [[buffer(16)]],
    constant uint &kv_capacity [[buffer(17)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    const ulong pos_offset = (ulong)*pos_offset_ptr;
    const ulong kv_dst_offset = (ulong)*kv_dst_offset_ptr;
    const ulong capacity = (ulong)kv_capacity;
    const ulong total_q = (ulong)T * (ulong)Hq;
    const ulong total_kv = (ulong)T * (ulong)Hkv;
    const ulong kv_span = total_kv * (ulong)D;
    const ulong gid64 = row_of(tg, sg, tptg);
    if (gid64 >= total_q + 2ul * total_kv) return;
    // The fused dispatch is atomic with respect to its dynamic cache slot: a
    // stale/hostile offset skips Q/K/V mutation as well as every cache store.
    // Widen before multiplication and use subtraction to make wraparound
    // incapable of turning an invalid span into an in-bounds address.
    if (kv_dst_offset > capacity || kv_span > capacity - kv_dst_offset) return;

    if (gid64 < total_q) {
        const ulong t = gid64 / (ulong)Hq;
        const ulong h = gid64 % (ulong)Hq;
        device float *row = q + ((t * (ulong)Hq + h) * (ulong)D);
        norm_rope_row<false>(row, q_weight, row, D, rotary_dim,
                             pos_offset + t, theta, eps, true, lane);
        return;
    }
    ulong g2 = gid64 - total_q;
    if (g2 < total_kv) {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = k + ((t * (ulong)Hkv + h) * (ulong)D);
        device float *dst = dst_k + (kv_dst_offset + (t * (ulong)Hkv + h) * (ulong)D);
        norm_rope_row<true>(row, k_weight, dst, D, rotary_dim,
                            pos_offset + t, theta, eps, true, lane);
        return;
    }
    g2 -= total_kv;
    {
        const ulong t = g2 / (ulong)Hkv;
        const ulong h = g2 % (ulong)Hkv;
        device float *row = v + ((t * (ulong)Hkv + h) * (ulong)D);
        device float *dst = dst_v + (kv_dst_offset + (t * (ulong)Hkv + h) * (ulong)D);
        norm_rope_row<true>(row, v_weight, dst, D, rotary_dim,
                            0ul, theta, eps, false, lane);
    }
}
