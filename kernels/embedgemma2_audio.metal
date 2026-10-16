// EmbeddingGemma 2's audio tower (transformers 5.19 `Gemma4AudioModel`), the
// kernels that no other tessl kernel covers. `src/embedgemma2_audio.rs` is
// the host side and holds the contract of each; the activations are f32.
//
// Row spaces. A batch is several clips laid end to end in each of three row
// spaces (frames, the first conv's output rows, the tower rows), described to
// the kernels that need clip boundaries by a clip table of `EGA_CLIP_WORDS`
// u32 per clip:
//   0 frame0    first frame row of the clip        1 frames    its frame count
//   2 mid0      first row after the first conv     3 mid_rows  ceil(frames / 2)
//   4 row0      first tower row                    5 rows      ceil(mid_rows / 2)
// Clips never read each other's rows: a kernel launched over clips takes the
// clip from its grid's y axis and works in clip-local indices.
//
// Clamps. `Gemma4ClippableLinear` clamps its input and output to scalar
// bounds from the checkpoint, and the tower clamps the residual stream to
// `+-gradient_clipping` around its norms. `gemm_epilogue` has no exact-f32
// path, so each clamp is folded into the elementwise pass beside it
// (`eg2a_rms_norm`, `eg2a_ew`, `eg2a_lconv`, the attention loads). "No bound" is
// +-FLT_MAX from the host (never an infinity: see tests/kernel_fast_math.rs).
#include <metal_stdlib>
#include "act.h"
#include "reduce_tree.h"
using namespace metal;

constant uint EGA_CLIP_WORDS = 8u;
/// Channels of the first conv a threadgroup holds (host-checked).
constant uint EGA_MAX_C0 = 512u;
/// Head-dim elements per lane in the attention (host-checked).
constant uint EGA_MAX_E = 8u;

// --- Subsampling: Conv2d(3x3, stride 2, pad 1, no bias) + LayerNorm + ReLU ----

/// The first subsampling layer. `Gemma4AudioSubSampleConvProjectionLayer`
/// with 1 input channel: the frames are multiplied by the frame mask (a masked
/// frame reads as 0), convolved, normalized over the channels with
/// `LayerNorm(eps, weight, no bias)` (biased variance, centred two-pass) and
/// rectified. One threadgroup per output position (`t1`, `f1`), one thread per
/// channel (strided if the channels outnumber the threads).
///
/// feats [Rf, mels]; fmask [Rf] (0/1); w [C0, 9] (the checkpoint's
/// `[C0, 1, 3, 3]`); gamma [C0]; out [R1, mels1, C0], channels last, so the
/// next layer reads a position's channels contiguously.
/// Grid: x = max over clips of `mid_rows * mels1`, y = clip.
kernel void eg2a_sub_conv0_ln_relu(
    device const float *feats [[buffer(0)]],
    device const uint *fmask [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device const float *gamma [[buffer(3)]],
    device float *out [[buffer(4)]],
    device const uint *clips [[buffer(5)]],
    constant uint &mels [[buffer(6)]],
    constant uint &mels1 [[buffer(7)]],
    constant uint &channels [[buffer(8)]],
    constant float &eps [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tptg2 [[threads_per_threadgroup]])
{
    const uint lid = lid2.x;
    const uint tptg = tptg2.x;
    device const uint *c = clips + (ulong)tg.y * EGA_CLIP_WORDS;
    const uint frame0 = c[0];
    const uint frames = c[1];
    const uint mid0 = c[2];
    const uint mid_rows = c[3];
    if (tg.x >= mid_rows * mels1) return;
    const uint t1 = tg.x / mels1;
    const uint f1 = tg.x % mels1;

    float x[9];
    for (uint kh = 0u; kh < 3u; ++kh) {
        for (uint kw = 0u; kw < 3u; ++kw) {
            const int r = (int)(2u * t1 + kh) - 1;
            const int col = (int)(2u * f1 + kw) - 1;
            float v = 0.0f;
            if (r >= 0 && r < (int)frames && col >= 0 && col < (int)mels && fmask[frame0 + (uint)r] != 0u) {
                v = feats[(ulong)(frame0 + (uint)r) * mels + (uint)col];
            }
            x[kh * 3u + kw] = v;
        }
    }

    threadgroup float vals[EGA_MAX_C0];
    threadgroup float scratch[2u * REDUCE_MAX_SIMDGROUPS];
    float s = 0.0f;
    for (uint co = lid; co < channels; co += tptg) {
        float a = 0.0f;
        for (uint k = 0u; k < 9u; ++k) a += w[(ulong)co * 9u + k] * x[k];
        vals[co] = a;
        s += a;
    }
    const float mean = reduce_row_add(s, scratch, sgid, lane, tptg) / (float)channels;
    float sq = 0.0f;
    for (uint co = lid; co < channels; co += tptg) {
        const float d = vals[co] - mean;
        sq += d * d;
    }
    const float var = reduce_row_add(sq, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg) / (float)channels;
    const float inv = rsqrt(var + eps);
    device float *o = out + ((ulong)(mid0 + t1) * mels1 + f1) * channels;
    for (uint co = lid; co < channels; co += tptg) {
        o[co] = max((vals[co] - mean) * inv * gamma[co], 0.0f);
    }
}

/// The second subsampling layer: `C0 -> C1` channels over the first layer's
/// output, whose rows are masked by `mask[::2]` (the frame mask at every
/// other frame) before the convolution, then the channel LayerNorm and ReLU.
/// The output row is `[mels2, C1]`, which is `permute(0, 2, 3,
/// 1).reshape(T, -1)` of transformers' `[C1, T, mels2]`: `index = f2 * C1 + co`.
///
/// One simdgroup per (output row `t2`, block of `EGA_CONV1_P` = 8 frequency
/// positions). Per tap `(kh, kw)` it stages the block's input patch (8
/// positions by `C0` channels, zero where the tap falls outside the frequency
/// range, the rows or the mask) in threadgroup memory, then multiplies it by
/// that tap's `[C0, C1]` weights with 8x8 `simdgroup_float8x8` tiles,
/// accumulating over the nine taps; weight tiles that fit are loaded from
/// device memory as they are, ragged edge tiles through a zero-padded stage.
/// The products are f32 and the sum over the `9 * C0` terms is accumulated
/// tile by tile, so the result differs from a serial sum by f32 rounding only.
/// (The per-position scalar kernel this replaced re-read the whole 147 KB of
/// weights for every output position, which made it the largest non-GEMM
/// stage of a long batch.) The LayerNorm statistics are `simd_sum`s over a
/// position's `C1` accumulators, centred two-pass as before.
///
/// mid [R1, mels1, C0]; w [9, C0, C1] (tap, input channel, output channel:
/// the checkpoint's `[C1, C0, 3, 3]` permuted so a tile's rows are contiguous);
/// gamma [C1]; out [R, mels2 * C1]. Dynamic threadgroup memory 0 holds
/// `8 * c0p + 8 * c1p + 64` floats with `cNp` the channel counts rounded up
/// to 8 (host-checked: `C0 <= EGA_MAX_C0`, `C1 <= EGA_MAX_C1`).
/// Grid: x = max over clips of `rows * ceil(mels2 / 8)`, y = clip; 32 threads.
constant uint EGA_CONV1_P = 8u;
constant uint EGA_MAX_C1 = 128u;

kernel void eg2a_sub_conv1_ln_relu(
    device const float *mid [[buffer(0)]],
    device const uint *fmask [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device const float *gamma [[buffer(3)]],
    device float *out [[buffer(4)]],
    device const uint *clips [[buffer(5)]],
    constant uint &mels1 [[buffer(6)]],
    constant uint &mels2 [[buffer(7)]],
    constant uint &c0 [[buffer(8)]],
    constant uint &c1 [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    threadgroup float *tgm [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const uint *c = clips + (ulong)tg.y * EGA_CLIP_WORDS;
    const uint frame0 = c[0];
    const uint mid0 = c[2];
    const uint mid_rows = c[3];
    const uint row0 = c[4];
    const uint rows = c[5];
    const uint nblk = (mels2 + EGA_CONV1_P - 1u) / EGA_CONV1_P;
    if (tg.x >= rows * nblk) return;
    const uint t2 = tg.x / nblk;
    const uint f2_0 = (tg.x % nblk) * EGA_CONV1_P;

    const uint c0p = (c0 + 7u) & ~7u;
    const uint c1p = (c1 + 7u) & ~7u;
    threadgroup float *a_tile = tgm;                          // [8][c0p]
    threadgroup float *o_tile = tgm + EGA_CONV1_P * c0p;      // [8][c1p]
    threadgroup float *b_edge = o_tile + EGA_CONV1_P * c1p;   // [8][8]

    simdgroup_float8x8 acc[EGA_MAX_C1 / 8u];
#pragma clang loop unroll(full)
    for (uint n = 0u; n < EGA_MAX_C1 / 8u; ++n) acc[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint n_tiles = c1p / 8u;

    for (uint kh = 0u; kh < 3u; ++kh) {
        const int r1 = (int)(2u * t2 + kh) - 1;
        if (r1 < 0 || r1 >= (int)mid_rows) continue;
        // The first layer's row r1 sits at frame 2 * r1 (mask[::2]); 2 * r1 <
        // frames because mid_rows = ceil(frames / 2).
        if (fmask[frame0 + 2u * (uint)r1] == 0u) continue;
        const ulong row_base = (ulong)(mid0 + (uint)r1) * mels1;
        for (uint kw = 0u; kw < 3u; ++kw) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < EGA_CONV1_P * c0p; e += 32u) {
                const uint p = e / c0p;
                const uint ci = e % c0p;
                const uint f2 = f2_0 + p;
                const int f = (int)(2u * f2 + kw) - 1;
                float v = 0.0f;
                if (ci < c0 && f2 < mels2 && f >= 0 && f < (int)mels1) v = mid[(row_base + (uint)f) * c0 + ci];
                a_tile[p * c0p + ci] = v;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            device const float *wk = w + (ulong)(kh * 3u + kw) * c0 * c1;
            for (uint kt = 0u; kt < c0p; kt += 8u) {
                simdgroup_float8x8 a;
                simdgroup_load(a, a_tile + kt, c0p);
#pragma clang loop unroll(full)
                for (uint n = 0u; n < EGA_MAX_C1 / 8u; ++n) {
                    if (n >= n_tiles) continue;
                    simdgroup_float8x8 b;
                    if (kt + 8u <= c0 && n * 8u + 8u <= c1) {
                        simdgroup_load(b, wk + (ulong)kt * c1 + n * 8u, c1);
                    } else {
                        for (uint e = lane; e < 64u; e += 32u) {
                            const uint kk = kt + e / 8u;
                            const uint co = n * 8u + e % 8u;
                            b_edge[e] = (kk < c0 && co < c1) ? wk[(ulong)kk * c1 + co] : 0.0f;
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        simdgroup_load(b, b_edge, 8);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    simdgroup_multiply_accumulate(acc[n], a, b, acc[n]);
                }
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint n = 0u; n < EGA_MAX_C1 / 8u; ++n) {
        if (n < n_tiles) simdgroup_store(acc[n], o_tile + n * 8u, c1p);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint p = 0u; p < EGA_CONV1_P; ++p) {
        const uint f2 = f2_0 + p;
        if (f2 >= mels2) break;
        threadgroup const float *r = o_tile + p * c1p;
        float s = 0.0f;
        for (uint co = lane; co < c1; co += 32u) s += r[co];
        const float mean = simd_sum(s) / (float)c1;
        float sq = 0.0f;
        for (uint co = lane; co < c1; co += 32u) {
            const float d = r[co] - mean;
            sq += d * d;
        }
        const float inv = rsqrt(simd_sum(sq) / (float)c1 + eps);
        device float *o = out + ((ulong)(row0 + t2) * mels2 + f2) * c1;
        for (uint co = lane; co < c1; co += 32u) o[co] = max((r[co] - mean) * inv * gamma[co], 0.0f);
    }
}

// --- Row passes and elementwise passes with the clamps folded in ---------------

/// One row stage's constants: words of `EgaRowStage`, all 4 bytes (the host
/// packs them in this order).
struct EgaRowStage {
    float eps;
    /// Two clamps applied to the input in turn: a clipped linear's output
    /// bounds, then `+-gradient_clipping`.
    float pre1_lo;
    float pre1_hi;
    float pre2_lo;
    float pre2_hi;
    /// The clamp after the norm (and the activation), one per output: the
    /// next clipped linears' input bounds. A stage with one output reads [0].
    float post_lo[3];
    float post_hi[3];
    /// Multiplies the normed value before the residual is added.
    float scale;
    /// bit 0: multiply by `weight`; bit 1: SiLU; bit 2: add `resid[row, d]`;
    /// bit 3: no norm (the pre-clamped value goes on as it is); bit 4: add
    /// `resid[d]` (a per-column bias) instead of bit 2's residual.
    uint flags;
};

/// Words of `EgaNormParams`.
struct EgaNormParams {
    uint rows;
    uint dim;
    /// Outputs of stage `b` (1 to 3); 0: stage `a` alone.
    uint outs;
    EgaRowStage a;
    EgaRowStage b;
};

/// A row stage over one row (`base` is its first element): `o_k[d] = [resid +]
/// scale * clamp_k(act(norm(clamp2(clamp1(src[d]))) [* weight]))` for each of
/// the stage's `n_out` outputs, `norm(v) = v * (mean(v^2) + eps)^-0.5`
/// (`Gemma4RMSNorm`: `* w`, not `* (1 + w)`; a stage with bit 3 has none). The
/// calling threadgroup's threads each handle the elements `lid, lid + tptg,
/// ...`, so a stage reads back only what its own thread wrote, and an output
/// may be `src` (each thread reads an element before it writes that element).
inline void ega_row_stage(
    device const float *src,
    device const float *weight,
    device const float *resid,
    device float *o0,
    device float *o1,
    device float *o2,
    uint n_out,
    constant EgaRowStage &s,
    ulong base,
    uint dim,
    threadgroup float *scratch,
    uint sgid,
    uint lane,
    uint lid,
    uint tptg)
{
    float inv = 1.0f;
    if ((s.flags & 8u) == 0u) {
        float ss = 0.0f;
        for (uint d = lid; d < dim; d += tptg) {
            const float v = clamp(clamp(src[base + d], s.pre1_lo, s.pre1_hi), s.pre2_lo, s.pre2_hi);
            ss += v * v;
        }
        inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + s.eps);
    }
    for (uint d = lid; d < dim; d += tptg) {
        float v = clamp(clamp(src[base + d], s.pre1_lo, s.pre1_hi), s.pre2_lo, s.pre2_hi) * inv;
        if ((s.flags & 1u) != 0u) v *= weight[d];
        if ((s.flags & 2u) != 0u) v = tessl_silu(v);
        float r = clamp(v, s.post_lo[0], s.post_hi[0]) * s.scale;
        if ((s.flags & 4u) != 0u) r += resid[base + d];
        if ((s.flags & 16u) != 0u) r += resid[d];
        if (n_out > 1u) o1[base + d] = clamp(v, s.post_lo[1], s.post_hi[1]) * s.scale;
        if (n_out > 2u) o2[base + d] = clamp(v, s.post_lo[2], s.post_hi[2]) * s.scale;
        o0[base + d] = r;
    }
}

/// One or two row stages per threadgroup (one row each): stage `a` reads `x`
/// and writes `out`; with `p.outs > 0`, stage `b` then reads `out` (what `a`
/// wrote) and writes `out_b0` (and `out_b1`, `out_b2`: the same value under
/// other output clamps, so the three clipped linears that read one norm each
/// get their own clamped copy from one pass). Two stages in one launch is
/// the residual add and the norm that follows it, which are both per-row: the
/// row is read once and the second stage reads the first's output while it is
/// still cached. `out` may be `x`; `resid` may be `out`; `out_b0` may be `out`.
kernel void eg2a_rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *resid [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant EgaNormParams &p [[buffer(4)]],
    device const float *weight_b [[buffer(5)]],
    device float *out_b0 [[buffer(6)]],
    device float *out_b1 [[buffer(7)]],
    device float *out_b2 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= p.rows) return;
    threadgroup float scratch[2u * REDUCE_MAX_SIMDGROUPS];
    const ulong base = (ulong)row * p.dim;
    ega_row_stage(x, weight, resid, out, out, out, 1u, p.a, base, p.dim, scratch, sgid, lane, lid, tptg);
    if (p.outs != 0u) {
        ega_row_stage(
            out, weight_b, out, out_b0, out_b1, out_b2, p.outs, p.b, base, p.dim,
            scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, lid, tptg);
    }
}

/// Words of `EgaEwParams`.
struct EgaEwParams {
    uint n;
    float lo1;
    float hi1;
    float lo2;
    float hi2;
    /// bit 1: SiLU between the clamps.
    uint flags;
};

/// `out[i] = clamp2(act(clamp1(x[i])))`: the pass between two clipped linears
/// (output clamp, SiLU, input clamp). `out` may be `x`.
kernel void eg2a_ew(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant EgaEwParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    float v = clamp(x[i], p.lo1, p.hi1);
    if ((p.flags & 2u) != 0u) v = tessl_silu(v);
    out[i] = clamp(v, p.lo2, p.hi2);
}

/// Hidden width the light conv's threadgroup holds (host-checked).
constant uint EGA_MAX_HIDDEN = 4096u;

/// Words of `EgaLconvParams`.
struct EgaLconvParams {
    uint hidden;
    uint taps;
    float eps;
    /// The linear_start output clamp the GLU applies to both halves.
    float glu_lo;
    float glu_hi;
    /// `+-gradient_clipping`, applied to the conv output before the norm.
    float pre_lo;
    float pre_hi;
    /// The next clipped linear's input clamp.
    float post_lo;
    float post_hi;
};

/// The middle of `Gemma4AudioLightConv1d` in one pass per row: the GLU, the
/// depthwise causal conv and the conv norm with its SiLU,
///
///   g[s, c]  = clamp(x[s, c]) * sigmoid(clamp(x[s, H + c]))
///   y[t, c]  = sum_j w[c, j] * g[t - (taps - 1) + j, c]       rows before the clip's first read 0
///   out[t]   = clamp_post(silu(rms_norm(clamp_pre(y[t])) * norm_w))
///
/// One threadgroup per (tower row `t`, clip), threads across the channels. The
/// GLU is recomputed for each of the `taps` rows a row's conv reads (they are
/// the rows its neighbours read too, so they come from cache), which is
/// cheaper than writing the GLU's output and the conv's to memory and reading
/// each back. Clips never read each other's rows.
///
/// x [R, 2 * hidden] (the linear_start GEMM's output); w [hidden, taps] (the
/// checkpoint's `[hidden, 1, taps]`); norm_w [hidden]; out [R, hidden].
/// Grid: x = max over clips of `rows`, y = clip.
kernel void eg2a_lconv(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *norm_w [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const uint *clips [[buffer(4)]],
    constant EgaLconvParams &p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tptg2 [[threads_per_threadgroup]])
{
    const uint lid = lid2.x;
    const uint tptg = tptg2.x;
    device const uint *c = clips + (ulong)tg.y * EGA_CLIP_WORDS;
    const uint row0 = c[4];
    const uint t = tg.x;
    if (t >= c[5]) return;
    threadgroup float vals[EGA_MAX_HIDDEN];
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    const uint hidden = p.hidden;
    float ss = 0.0f;
    for (uint ch = lid; ch < hidden; ch += tptg) {
        float acc = 0.0f;
        for (uint j = 0u; j < p.taps; ++j) {
            const int s = (int)t - (int)(p.taps - 1u) + (int)j;
            if (s >= 0) {
                device const float *g = x + (ulong)(row0 + (uint)s) * 2u * hidden;
                const float a = clamp(g[ch], p.glu_lo, p.glu_hi);
                const float b = clamp(g[hidden + ch], p.glu_lo, p.glu_hi);
                acc += w[(ulong)ch * p.taps + j] * (a * tessl_sigmoid(b));
            }
        }
        const float v = clamp(acc, p.pre_lo, p.pre_hi);
        vals[ch] = v;
        ss += v * v;
    }
    const float inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)hidden + p.eps);
    device float *o = out + (ulong)(row0 + t) * hidden;
    for (uint ch = lid; ch < hidden; ch += tptg) {
        o[ch] = clamp(tessl_silu(vals[ch] * inv * norm_w[ch]), p.post_lo, p.post_hi);
    }
}

// --- Chunked local attention with Transformer-XL relative positions -----------

/// Past frames the kernel holds partial logits for (host-checked).
constant uint EGA_MAX_PAST = 16u;

/// Words of `EgaAttnParams`.
struct EgaAttnParams {
    uint heads;
    uint head_dim;
    /// Queries per block (`attention_chunk_size`).
    uint chunk;
    /// Past frames the mask admits (`attention_context_left - 1`); a block's
    /// window holds `ctx = chunk + past` keys.
    uint past;
    /// Rows of the relative-position table (`ctx / 2 + 1`).
    uint rel_rows;
    uint ctx;
    /// `ln(1 + e) / ln 2`, applied to every key.
    float k_scale;
    /// `attention_logit_cap`. (A masked logit is `attention_invalid_logits_value`,
    /// which the host guarantees is far enough below every unmasked one that
    /// its softmax weight is exactly 0; see the kernel's comment.)
    float cap;
    /// Output clamps of the q, k and v projections, and the input clamp of
    /// the output projection (`post`), applied to what the kernel loads and
    /// stores.
    float q_lo;
    float q_hi;
    float k_lo;
    float k_hi;
    float v_lo;
    float v_hi;
    float o_lo;
    float o_hi;
    /// Elements between the rows of q, k and v, and the column each starts
    /// at: `hidden` and 0 for three buffers of their own, `3 * hidden` and
    /// `0, hidden, 2 * hidden` for one fused q|k|v projection.
    uint stride;
    uint q_off;
    uint k_off;
    uint v_off;
};

/// Sums each of 16 per-lane values across the 32 lanes of a simdgroup, and
/// leaves the sum of value `l >> 1` in lane `l` (lanes `2i` and `2i + 1` hold
/// value `i`). A butterfly that halves the values it carries at each of four
/// steps, so it moves 8 + 4 + 2 + 1 + 1 values where 16 `simd_sum`s move 80.
inline float ega_scatter_sum16(thread float *v, uint lane) {
    bool hi = (lane & 16u) != 0u;
    float a[8];
    for (uint i = 0u; i < 8u; ++i) {
        a[i] = (hi ? v[i + 8u] : v[i]) + simd_shuffle_xor(hi ? v[i] : v[i + 8u], 16u);
    }
    hi = (lane & 8u) != 0u;
    float b[4];
    for (uint i = 0u; i < 4u; ++i) {
        b[i] = (hi ? a[i + 4u] : a[i]) + simd_shuffle_xor(hi ? a[i] : a[i + 4u], 8u);
    }
    hi = (lane & 4u) != 0u;
    float c[2];
    for (uint i = 0u; i < 2u; ++i) {
        c[i] = (hi ? b[i + 2u] : b[i]) + simd_shuffle_xor(hi ? b[i] : b[i + 2u], 4u);
    }
    hi = (lane & 2u) != 0u;
    const float d = (hi ? c[1] : c[0]) + simd_shuffle_xor(hi ? c[0] : c[1], 2u);
    return d + simd_shuffle_xor(d, 1u);
}

/// `Gemma4AudioAttention.forward` after the projections, for the queries of
/// every clip: one simdgroup per (query, head).
///
/// A query `t` in block `blk = t / chunk` (position `i = t % chunk` in it)
/// sees the block's window of `ctx` keys; window slot `j` holds frame
/// `kp = blk * chunk + j - past`. The logit is
///
///   softcap(ac + bd),  ac = (q * q_mult) . (k * k_scale),
///                      bd = (q * q_mult) . rel[j - i]   when 0 <= j - i < rel_rows, else 0,
///
/// where `q_mult[d] = head_dim^-0.5 / ln 2 * softplus(per_dim_scale[d])` and
/// `rel = relative_k_proj(sinusoidal table)` are host constants. `bd` is
/// transformers' `_rel_shift` written out: it pads each row of the `[chunk,
/// rel_rows]` table product to `ctx + 1`, flattens, drops the tail and views
/// the rest as `[chunk, ctx]`, so row `i` column `j` reads source column `j -
/// i` of row `i` for `j >= i`, and (host-checked) reads padding for `j < i`.
/// Row `p` of `rel` holds relative position `ctx / 2 - p`, so slot `j` of
/// query `i` has `q - kp = past - (j - i)` frames of distance: the table is
/// indexed `rel[past - (t - kp)]` directly, with no shift to materialize.
///
/// A slot is visible iff `0 <= t - kp < past` (`sliding_window_mask_function
/// ((past, 0))`: the query and the `past - 1` frames before it), `kp` is a
/// frame of the clip and `omask[kp]` (the padding mask, on keys only). Those
/// are the slots `j = i + 1 .. i + past`, so the kernel computes the logits of
/// those `past` candidates only: each lane holds its share of `ac + bd` for
/// each candidate (the head dimension is spread over the lanes), one
/// `ega_scatter_sum16` finishes all of them, and the softcap, the maximum and
/// the exponentials are then one value per lane pair instead of a serial loop
/// per key. An invisible slot has a masked logit (a finite -1e9, as
/// `masked_fill` writes it), whose softmax weight is exactly 0 (the host
/// requires it to sit at least 104 below every unmasked one, past `exp`'s
/// underflow), unless nothing is visible at all: then every slot of the window
/// has the same logit, and the row softmaxes to the uniform `1 / ctx` over
/// its window, values included, exactly as transformers computes (and then
/// nothing reads that row). That case is the second loop below.
///
/// q, k, v are read at `row * stride + off + column` (`stride`, `*_off` in
/// the params); out [R, heads * head_dim]; rel [rel_rows, heads * head_dim];
/// q_mult [head_dim]; omask [R] (0/1). Grid: x = max rows * heads, y = clip;
/// 32 threads.
kernel void eg2a_attn(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *rel [[buffer(3)]],
    device const float *q_mult [[buffer(4)]],
    device const uint *omask [[buffer(5)]],
    device const uint *clips [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant EgaAttnParams &p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const uint *c = clips + (ulong)tg.y * EGA_CLIP_WORDS;
    const uint row0 = c[4];
    const uint rows = c[5];
    const uint t = tg.x / p.heads;
    const uint h = tg.x % p.heads;
    if (t >= rows) return;
    const uint hidden = p.heads * p.head_dim;
    const ulong col0 = (ulong)h * p.head_dim;

    float qs[EGA_MAX_E];
    for (uint e = 0u; e < EGA_MAX_E; ++e) {
        const uint d = lane + 32u * e;
        qs[e] = d < p.head_dim
            ? clamp(q[(ulong)(row0 + t) * p.stride + p.q_off + col0 + d], p.q_lo, p.q_hi) * q_mult[d]
            : 0.0f;
    }

    const uint blk = t / p.chunk;
    const uint i = t % p.chunk;
    // Candidate jj is window slot j = i + 1 + jj, frame kp = blk * chunk + j - past.
    float part[EGA_MAX_PAST];
    uint visible = 0u;
    for (uint jj = 0u; jj < EGA_MAX_PAST; ++jj) {
        float s = 0.0f;
        const uint j = i + 1u + jj;
        const int kp = (int)(blk * p.chunk + j) - (int)p.past;
        if (jj < p.past && kp >= 0 && kp < (int)rows && omask[row0 + (uint)kp] != 0u) {
            visible |= 1u << jj;
            device const float *kr = k + (ulong)(row0 + (uint)kp) * p.stride + p.k_off + col0;
            for (uint e = 0u; e < EGA_MAX_E; ++e) {
                const uint d = lane + 32u * e;
                if (d < p.head_dim) s += qs[e] * (clamp(kr[d], p.k_lo, p.k_hi) * p.k_scale);
            }
            if (j - i < p.rel_rows) {
                device const float *rr = rel + (ulong)(j - i) * hidden + col0;
                for (uint e = 0u; e < EGA_MAX_E; ++e) {
                    const uint d = lane + 32u * e;
                    if (d < p.head_dim) s += qs[e] * rr[d];
                }
            }
        }
        part[jj] = s;
    }

    float acc[EGA_MAX_E];
    for (uint e = 0u; e < EGA_MAX_E; ++e) acc[e] = 0.0f;

    if (visible != 0u) {
        const float total = ega_scatter_sum16(part, lane);
        const uint mine = lane >> 1;
        const bool vis = ((visible >> mine) & 1u) != 0u;
        // `tanh(s / cap) * cap`; plain tanh lowers to a fast tanh that is not
        // trusted past |10|, and tanh has saturated well before 16.
        const float s = precise::tanh(clamp(total / p.cap, -16.0f, 16.0f)) * p.cap;
        const float m = simd_max(vis ? s : -FLT_MAX);
        const float ex = vis ? exp(s - m) : 0.0f;
        const float den = simd_sum((lane & 1u) == 0u ? ex : 0.0f);
        const float wgt = ex / den;
        for (uint jj = 0u; jj < p.past; ++jj) {
            if (((visible >> jj) & 1u) == 0u) continue;
            const float w = simd_broadcast(wgt, (ushort)(2u * jj));
            const uint kp = blk * p.chunk + i + 1u + jj - p.past;
            device const float *vr = v + (ulong)(row0 + kp) * p.stride + p.v_off + col0;
            for (uint e = 0u; e < EGA_MAX_E; ++e) {
                const uint d = lane + 32u * e;
                if (d < p.head_dim) acc[e] += w * clamp(vr[d], p.v_lo, p.v_hi);
            }
        }
    } else {
        // Nothing visible: the uniform weight over the whole window. Slots
        // outside the clip hold zero values (`F.pad`).
        const float w = 1.0f / (float)p.ctx;
        for (uint j = 0u; j < p.ctx; ++j) {
            const int kp = (int)(blk * p.chunk + j) - (int)p.past;
            if (kp < 0 || kp >= (int)rows) continue;
            device const float *vr = v + (ulong)(row0 + (uint)kp) * p.stride + p.v_off + col0;
            for (uint e = 0u; e < EGA_MAX_E; ++e) {
                const uint d = lane + 32u * e;
                if (d < p.head_dim) acc[e] += w * clamp(vr[d], p.v_lo, p.v_hi);
            }
        }
    }
    device float *o = out + (ulong)(row0 + t) * hidden + col0;
    for (uint e = 0u; e < EGA_MAX_E; ++e) {
        const uint d = lane + 32u * e;
        if (d < p.head_dim) o[d] = clamp(acc[e], p.o_lo, p.o_hi);
    }
}
