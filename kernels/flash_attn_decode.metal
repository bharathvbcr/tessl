// FlashDecoding: single-query attention, split over the KV sequence.
// K/V use [B,kv_capacity,Hkv,D]; Tkv is the live prefix and never a stride.
//
// The general FA-2 kernels tile over query rows: the grid is
// `ceil(Tq/BR) x B*H` and only `lid < BR` lanes are row-valid. At Tq=1 that is
// one live lane in 32, and the grid collapses to B*H threadgroups -- 8 for an
// 8-head decode. Measured against MLX on an M5 Pro, `global512_decode_4k` ran
// 271x slower, and the slowdown tracked threadgroup count almost monotonically
// (8 -> 271x, 16 -> 47x, 32 -> 27x, 256 -> 10x). The kernels' own comment said
// "decode Tq=1 wastes lanes but FA is tiny vs GEMV"; the benchmark disagreed.
//
// This splits the other way. One simdgroup owns one KV chunk, so the grid is
// `n_chunks x B*H` and every lane is live:
//
//   * lane L owns head dims L, L+32, L+64, ... -- DPL = HEAD_DIM/32 of them,
//     held in registers, so the K/V reads are coalesced across the simdgroup
//     and the P@V accumulate needs no cross-lane communication at all;
//   * a score is one `simd_sum` of the per-lane partial dot, which every lane
//     then holds, so the online-softmax state (m, l) is uniform and the whole
//     kernel is branch-divergence free;
//   * each chunk emits (m, l, acc[D]) and a second pass combines them with the
//     standard rescale, which is exact rather than an approximation.
//
// Masking is transcribed from the general kernels: q_abs = q_pos_offset + t_q,
// k_abs = kv_pos_offset + t_k, keep k_abs <= q_abs, and additionally
// k_abs >= max(0, q_abs - window + 1) when `window` is nonzero. `window == 0`
// is the global rule. A query with nothing unmasked yields zeros, not NaN.
#include <metal_stdlib>
using namespace metal;

constant uint SIMD_W = 32;

// Keys per chunk, and the default the host mirrors. Larger amortises the
// partial write; smaller buys grid parallelism, which is the entire point of
// this kernel -- measured at 21x off memory bound, decode is latency bound, so
// concurrency is the lever rather than arithmetic. Swept by
// `bench_flash_attn` with BENCH_ATTN_DECODE_CHUNK; see the tessl README.
constant uint KV_CHUNK = 256;

/// Partial pass: one simdgroup, one (batch, head, kv-chunk).
///
/// `Tkv` lives on the device, so the host cannot size the grid to the live
/// chunk count and dispatches for the requested, jointly-backed K/V capacity
/// instead. Both passes clamp `Tkv` to that same bound before deriving an
/// address. Chunks past the clamped value return immediately, and the reduce
/// pass visits exactly the chunks the partial pass wrote.
#define DECODE_PARTIAL_KERNEL(NAME, D, CH, R)                                 \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *partials [[buffer(3)]],                                     \
    constant uint &B [[buffer(4)]],                                           \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    constant uint &kv_capacity [[buffer(13)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]],                           \
    uint2 tptg [[threads_per_threadgroup]])                                   \
{                                                                             \
    /* Head dims per lane, as float4s -- a lane owns four *consecutive* dims  \
       per step, so one instruction moves 16R bytes across the R lanes of a   \
       key group instead of 4R. D % (4R) == 0 for every instantiation and     \
       `K + kv_base` is D-aligned, so the reads are aligned.                   \
                                                                              \
       This was measured *twice*, with opposite results, and the order        \
       matters. On the one-simdgroup-per-threadgroup kernel it lost 8-14% at  \
       every head dim: decode is latency bound, and 16 narrow loads left more \
       requests outstanding than 4 wide ones. Once the GQA group became       \
       co-resident -- four simdgroups in a threadgroup walking the same K/V   \
       -- the balance inverted, because the outstanding requests are now      \
       supplied by the other simdgroups and what is scarce is L1 bandwidth,   \
       which wide loads use better. It is worth 1.1x at D=512 and neutral     \
       elsewhere. A tuning result is only valid against the kernel it was     \
       measured on. */                                                        \
    constexpr uint DPV = (D) / (4u * (R));                                    \
    constexpr uint KPG = SIMD_W / (R); /* keys in flight per simdgroup */     \
    /* One simdgroup per query head, and the `H/Hkv` heads that share a KV    \
       head are put in the *same* threadgroup so they walk K and V together.   \
       Nothing is staged in threadgroup memory and there is no barrier: being  \
       co-resident is the whole mechanism, because the four heads of a 4:1     \
       group then touch each line while it is still in L1 instead of pulling   \
       it four separate times.                                                 \
                                                                              \
       That re-read was free of DRAM cost -- the same shape with Hkv = H       \
       issues identical bytes and takes 3x longer, which is what says the      \
       cache was absorbing it -- but not free of *cache* cost: `global512_     \
       decode_4k` was pushing 134 MB through the load path for 33.6 MB of      \
       unique K/V, and sat 2.4x above its own DRAM floor while the no-GQA      \
       shape sat on it.                                                        \
                                                                              \
       grid.y enumerates (batch, kv-head); the host sizes the threadgroup at   \
       `H/Hkv` simdgroups and falls back to one when that exceeds the Metal    \
       maximum, which the `sgs == 1` case below reproduces exactly. */         \
    const uint sgs = max(tptg.x / SIMD_W, 1u);                                \
    const uint sg = tpitg.x / SIMD_W;                                         \
    const uint lane = tpitg.x % SIMD_W;                                       \
    const uint grp = lane / (R);       /* which key-group */                  \
    const uint dl = lane % (R);        /* which dim slice */                  \
    /* Clamp mutable device state before it participates in any address. */   \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    const uint chunk = tgpig.x;                                               \
    /* grid.y enumerates (batch, head-block); a block is the `sgs` consecutive \
       query heads one threadgroup covers. `sgs == H/Hkv` puts exactly the     \
       heads of one KV group together, `sgs == H` puts every head of a batch   \
       item together -- which also makes the threadgroup's per-key read the    \
       full contiguous `[Hkv][D]` row instead of one strided slice of it --    \
       and `sgs == 1` is the original one-head-per-threadgroup dispatch. The   \
       host picks; all three fall out of the same arithmetic. */               \
    const uint blocks = max(H / sgs, 1u);                                     \
    const uint b = tgpig.y / blocks;                                          \
    const uint hb = tgpig.y % blocks;                                         \
    const uint h = hb * sgs + sg;                                             \
    if (h >= H) { return; }                                                   \
    const ulong bh = (ulong)b * H + h;                                        \
    const uint t_k0 = chunk * (CH);                                           \
    if (t_k0 >= Tkv) { return; }                                              \
    const uint n_k = min((uint)(CH), Tkv - t_k0);                             \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
    const ulong kv_pos_stride = (ulong)Hkv * (D);                             \
    const ulong kv_head_base =                                                \
        (ulong)b * kv_capacity * kv_pos_stride + (ulong)hkv * (D);            \
                                                                              \
    const ulong q_abs = (ulong)(*q_pos_offset_ptr);                           \
    const ulong kv_pos_offset = (ulong)(*kv_pos_offset_ptr);                  \
    const ulong window_back = (window == 0u) ? 0ul : (ulong)window - 1ul;     \
    const ulong k_lo = (window == 0u || q_abs < window_back)                  \
        ? 0ul                                                                 \
        : q_abs - window_back;                                                 \
                                                                              \
    /* Tq == 1, so the single query row is index 0. */                        \
    const ulong q_off = bh * (D);                                             \
    device const float4 *Q4 = (device const float4 *)(Q + q_off);             \
    float4 q_reg[DPV];                                                        \
    for (uint j = 0; j < DPV; ++j) { q_reg[j] = Q4[dl + j * (R)]; }           \
                                                                              \
    float4 acc[DPV];                                                          \
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }                 \
    float m_i = -INFINITY;                                                    \
    float l_i = 0.0f;                                                         \
                                                                              \
    /* Live key sub-range of this chunk, in local indices. Tq == 1, so the    \
       mask bounds are a single interval every lane agrees on, which lets the \
       loop visit only unmasked keys instead of reading every one and masking \
       after. That distinction is worth 2x on a windowed decode: at           \
       window=1024 over Tkv=4096 three quarters of the keys are masked, and   \
       reading them costs full K and V traffic for nothing.                   \
                                                                              \
       Restoring this is a regression fix -- the `continue` the pre-R kernel   \
       had did the same job, and dropping it to keep the butterfly uniform    \
       cost 2x on `swa128_decode_b8_4k` before the sweep caught it. */        \
    const ulong local_lo = (k_lo > kv_pos_offset) ? k_lo - kv_pos_offset      \
                                                   : 0ul;                      \
    const ulong local_hi = (q_abs >= kv_pos_offset)                           \
        ? q_abs - kv_pos_offset + 1ul                                         \
        : 0ul;                                                                \
    const ulong lo_i = max((ulong)t_k0, local_lo);                            \
    const ulong hi_i = min((ulong)t_k0 + n_k, local_hi);                      \
    const uint stride0 = (D) + 2u;                                            \
    const uint n_chunks = Tkv / (CH) + ((Tkv % (CH)) != 0u ? 1u : 0u);        \
    const ulong base0 = (bh * n_chunks + chunk) * stride0;                    \
    if (lo_i >= hi_i) {                                                       \
        /* Chunk fully masked. Uniform across the simdgroup, so returning     \
           here cannot strand a butterfly mid-flight. */                      \
        if (lane == 0u) {                                                     \
            partials[base0] = -INFINITY;                                      \
            partials[base0 + 1u] = 0.0f;                                      \
        }                                                                     \
        for (uint d = lane; d < (D); d += SIMD_W) {                           \
            partials[base0 + 2u + d] = 0.0f;                                  \
        }                                                                     \
        return;                                                               \
    }                                                                         \
    const uint live = (uint)(hi_i - lo_i);                                    \
    /* Uniform trip count across the KPG key-groups. The butterfly is a       \
       simdgroup-wide primitive, so a group that exited early would leave the \
       others shuffling against inactive lanes; the tail clamps its read to a \
       live key and masks the score instead. */                               \
    const uint iters = (live + KPG - 1u) / KPG;                               \
    for (uint it = 0; it < iters; ++it) {                                     \
        const uint t = it * KPG + grp;                                        \
        const uint tt = min(t, live - 1u);                                    \
        const uint key = (uint)lo_i + tt;                                     \
        const ulong kv_base = kv_head_base + (ulong)key * kv_pos_stride;      \
        device const float4 *K4 = (device const float4 *)(K + kv_base);       \
        float4 dot4 = float4(0.0f);                                           \
        for (uint j = 0; j < DPV; ++j) {                                      \
            dot4 += q_reg[j] * K4[dl + j * (R)];                              \
        }                                                                     \
        float part = dot4.x + dot4.y + dot4.z + dot4.w;                       \
        for (uint off = (R) / 2u; off > 0u; off >>= 1) {                      \
            part += simd_shuffle_xor(part, off);                              \
        }                                                                     \
        float s = part * scale;                                               \
        if (t >= live) { s = -INFINITY; }                                     \
        const float m_new = max(m_i, s);                                      \
        const float alpha = (m_i == -INFINITY) ? 0.0f : exp(m_i - m_new);     \
        const float p = (s == -INFINITY) ? 0.0f : exp(s - m_new);             \
        device const float4 *V4 = (device const float4 *)(V + kv_base);       \
        for (uint j = 0; j < DPV; ++j) {                                      \
            acc[j] = acc[j] * alpha + p * V4[dl + j * (R)];                   \
        }                                                                     \
        l_i = l_i * alpha + p;                                                \
        m_i = m_new;                                                          \
    }                                                                         \
                                                                              \
    /* Combine the KPG key-groups. Lanes holding the same `dl` sit R apart,   \
       so butterflying with offsets R, 2R, ... 16 sums across groups while    \
       keeping each lane on its own dims. At R=32 KPG is 1 and this loop does \
       not execute, which is the un-split kernel exactly. */                  \
    float m_all = m_i;                                                        \
    for (uint off = (R); off < SIMD_W; off <<= 1) {                           \
        m_all = max(m_all, simd_shuffle_xor(m_all, off));                     \
    }                                                                         \
    /* Both -inf means this group contributed nothing; exp(-inf - -inf) would \
       be NaN and the weight is exactly zero. */                              \
    const float w = (m_i == -INFINITY || m_all == -INFINITY)                  \
        ? 0.0f                                                                \
        : exp(m_i - m_all);                                                   \
    float l_all = l_i * w;                                                    \
    for (uint off = (R); off < SIMD_W; off <<= 1) {                           \
        l_all += simd_shuffle_xor(l_all, off);                                \
    }                                                                         \
    for (uint j = 0; j < DPV; ++j) {                                          \
        float4 a = acc[j] * w;                                                \
        for (uint off = (R); off < SIMD_W; off <<= 1) {                       \
            a.x += simd_shuffle_xor(a.x, off);                                \
            a.y += simd_shuffle_xor(a.y, off);                                \
            a.z += simd_shuffle_xor(a.z, off);                                \
            a.w += simd_shuffle_xor(a.w, off);                                \
        }                                                                     \
        acc[j] = a;                                                           \
    }                                                                         \
                                                                              \
    /* Group 0's R lanes cover dims dl + j*R for all j, i.e. every dim, so it \
       alone writes. The layout stays canonical [m, l, acc[D]] whatever R is, \
       which is why the reduce pass needs no R of its own. */                 \
    if (grp != 0u) { return; }                                                \
    const ulong base = base0;                                                 \
    if (lane == 0u) { partials[base] = m_all; partials[base + 1u] = l_all; }  \
    for (uint j = 0; j < DPV; ++j) {                                          \
        const uint d0 = 4u * (dl + j * (R));                                  \
        partials[base + 2u + d0 + 0u] = acc[j].x;                             \
        partials[base + 2u + d0 + 1u] = acc[j].y;                             \
        partials[base + 2u + d0 + 2u] = acc[j].z;                             \
        partials[base + 2u + d0 + 3u] = acc[j].w;                             \
    }                                                                         \
    (void)B;                                                                  \
}

/// Reduce pass: combine every chunk's (m, l, acc) for one (batch, head).
#define DECODE_REDUCE_KERNEL(NAME, D, CH)                                         \
kernel void NAME(                                                             \
    device const float *partials [[buffer(0)]],                               \
    device float *O [[buffer(1)]],                                            \
    constant uint &B [[buffer(2)]],                                           \
    device const uint *Tkv_ptr [[buffer(3)]],                                 \
    constant uint &H [[buffer(4)]],                                           \
    constant uint &out_bf16 [[buffer(5)]],                                    \
    constant uint &kv_capacity [[buffer(6)]],                                 \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]],                           \
    uint2 tptg [[threads_per_threadgroup]])                                   \
{                                                                             \
    /* Threadgroup width is a *dispatch* parameter, not a compile-time one,   \
       and the loops below are strided by it rather than by SIMD_W. This pass \
       used to run on one simdgroup per (batch, head) -- 8 threadgroups of 32 \
       threads for `global512_decode_4k`, 256 threads in total, folding 16    \
       chunks x 514 floats each. It is a serial tail on an otherwise parallel \
       kernel, and at D=512 it was most of the gap to MLX.                    \
                                                                              \
       Keeping no accumulator array is what lets the width be a runtime       \
       value: each lane recomputes the chunk weights per output dim instead   \
       of holding acc[D/32] in registers. The weights are `n_chunks` scalar   \
       reads from a line every lane in the threadgroup is already touching,   \
       so the recompute is cache-resident and the register cost is gone. */   \
    const uint lid = tpitg.x;                                                 \
    const uint width = max(tptg.x, 1u);                                       \
    /* Match the partial pass's independently bounded scratch layout. */      \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    const uint bh = tgpig.y;                                                  \
    const uint h = bh % H;                                                    \
    const uint b = bh / H;                                                    \
    const uint n_chunks = Tkv / (CH) + ((Tkv % (CH)) != 0u ? 1u : 0u);        \
    const uint stride = D + 2u;                                              \
    const ulong chunk0 = (ulong)bh * n_chunks;                                \
                                                                              \
    float m_all = -INFINITY;                                                  \
    for (uint c = 0; c < n_chunks; ++c) {                                     \
        m_all = max(m_all, partials[(chunk0 + c) * stride]);                  \
    }                                                                         \
    /* Every chunk masked: the general kernels emit zeros rather than NaN,    \
       and exp(-inf - -inf) below would be NaN, so this is both the matching  \
       behaviour and the safe one. */                                         \
    float l_all = 0.0f;                                                       \
    if (m_all != -INFINITY) {                                                 \
        for (uint c = 0; c < n_chunks; ++c) {                                 \
            const float m_c = partials[(chunk0 + c) * stride];                \
            if (m_c == -INFINITY) { continue; }                               \
            l_all += partials[(chunk0 + c) * stride + 1u] * exp(m_c - m_all); \
        }                                                                     \
    }                                                                         \
    const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;               \
    const ulong o_off = (ulong)bh * D;                                        \
    device bfloat *Ob = (device bfloat *)O;                                   \
    for (uint d = lid; d < (D); d += width) {                                 \
        float a = 0.0f;                                                       \
        if (m_all != -INFINITY) {                                             \
            for (uint c = 0; c < n_chunks; ++c) {                             \
                const ulong base = (chunk0 + c) * stride;                     \
                const float m_c = partials[base];                             \
                if (m_c == -INFINITY) { continue; }                           \
                a += partials[base + 2u + d] * exp(m_c - m_all);              \
            }                                                                 \
        }                                                                     \
        const float o = a * inv_l;                                            \
        if (out_bf16 != 0u) { Ob[o_off + d] = bfloat(o); }                    \
        else { O[o_off + d] = o; }                                            \
    }                                                                         \
    (void)B;                                                                  \
}

DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c64_r8, 128, 64, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c64_r16, 128, 64, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c64_r32, 128, 64, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c128_r8, 128, 128, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c128_r16, 128, 128, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c128_r32, 128, 128, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c256_r8, 128, 256, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c256_r16, 128, 256, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h128_c256_r32, 128, 256, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c64_r8, 256, 64, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c64_r16, 256, 64, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c64_r32, 256, 64, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c128_r8, 256, 128, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c128_r16, 256, 128, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c128_r32, 256, 128, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c256_r8, 256, 256, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c256_r16, 256, 256, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h256_c256_r32, 256, 256, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c64_r8, 512, 64, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c64_r16, 512, 64, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c64_r32, 512, 64, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c128_r8, 512, 128, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c128_r16, 512, 128, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c128_r32, 512, 128, 32)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c256_r8, 512, 256, 8)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c256_r16, 512, 256, 16)
DECODE_PARTIAL_KERNEL(flash_attn_decode_partial_h512_c256_r32, 512, 256, 32)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h128_c64, 128, 64)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h128_c128, 128, 128)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h128_c256, 128, 256)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h256_c64, 256, 64)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h256_c128, 256, 128)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h256_c256, 256, 256)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h512_c64, 512, 64)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h512_c128, 512, 128)
DECODE_REDUCE_KERNEL(flash_attn_decode_reduce_h512_c256, 512, 256)
