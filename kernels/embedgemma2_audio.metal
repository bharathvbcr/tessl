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
// (`eg2a_rms_norm`, `eg2a_ew`, `eg2a_glu`, the attention loads). "No bound" is
// +-FLT_MAX from the host (never an infinity: see tests/kernel_fast_math.rs).
#include <metal_stdlib>
#include "act.h"
#include "reduce_tree.h"
using namespace metal;

constant uint EGA_CLIP_WORDS = 8u;
/// Channels of the first conv a threadgroup holds (host-checked).
constant uint EGA_MAX_C0 = 512u;
/// Attention window positions and head-dim elements per lane (host-checked).
constant uint EGA_MAX_CTX = 64u;
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
/// other frame) before the convolution. One simdgroup per output position,
/// one lane per channel (up to four per lane); the LayerNorm statistics are
/// `simd_sum`s. The output row is `[mels2, C1]`, which is `permute(0, 2, 3,
/// 1).reshape(T, -1)` of transformers' `[C1, T, mels2]`: `index = f2 * C1 + co`.
///
/// mid [R1, mels1, C0]; w [9, C0, C1] (tap, input channel, output channel:
/// the checkpoint's `[C1, C0, 3, 3]` permuted so lanes read adjacent
/// addresses); gamma [C1]; out [R, mels2 * C1].
/// Grid: x = max over clips of `rows * mels2`, y = clip; 32 threads.
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
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    device const uint *c = clips + (ulong)tg.y * EGA_CLIP_WORDS;
    const uint frame0 = c[0];
    const uint mid0 = c[2];
    const uint mid_rows = c[3];
    const uint row0 = c[4];
    const uint rows = c[5];
    if (tg.x >= rows * mels2) return;
    const uint t2 = tg.x / mels2;
    const uint f2 = tg.x % mels2;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint kh = 0u; kh < 3u; ++kh) {
        const int r1 = (int)(2u * t2 + kh) - 1;
        if (r1 < 0 || r1 >= (int)mid_rows) continue;
        // The first layer's row r1 sits at frame 2 * r1 (mask[::2]); 2 * r1 <
        // frames because mid_rows = ceil(frames / 2).
        if (fmask[frame0 + 2u * (uint)r1] == 0u) continue;
        for (uint kw = 0u; kw < 3u; ++kw) {
            const int f = (int)(2u * f2 + kw) - 1;
            if (f < 0 || f >= (int)mels1) continue;
            device const float *xin = mid + ((ulong)(mid0 + (uint)r1) * mels1 + (uint)f) * c0;
            device const float *wk = w + (ulong)(kh * 3u + kw) * c0 * c1;
            for (uint ci = 0u; ci < c0; ++ci) {
                const float xv = xin[ci];
                for (uint i = 0u; i < 4u; ++i) {
                    const uint co = lane + 32u * i;
                    if (co < c1) acc[i] += wk[(ulong)ci * c1 + co] * xv;
                }
            }
        }
    }

    float s = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        if (lane + 32u * i < c1) s += acc[i];
    }
    const float mean = simd_sum(s) / (float)c1;
    float sq = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        if (lane + 32u * i < c1) {
            const float d = acc[i] - mean;
            sq += d * d;
        }
    }
    const float inv = rsqrt(simd_sum(sq) / (float)c1 + eps);
    device float *o = out + ((ulong)(row0 + t2) * mels2 + f2) * c1;
    for (uint i = 0u; i < 4u; ++i) {
        const uint co = lane + 32u * i;
        if (co < c1) o[co] = max((acc[i] - mean) * inv * gamma[co], 0.0f);
    }
}

// --- Row norm and elementwise passes with the clamps folded in -----------------

/// Words of `EgaNormParams`, in this order, all 4 bytes (the host packs them).
struct EgaNormParams {
    uint rows;
    uint dim;
    float eps;
    /// Two clamps applied to the input in turn: a clipped linear's output
    /// bounds, then `+-gradient_clipping`.
    float pre1_lo;
    float pre1_hi;
    float pre2_lo;
    float pre2_hi;
    /// The clamp after the norm (and the activation): the next clipped
    /// linear's input bounds.
    float post_lo;
    float post_hi;
    /// Multiplies the normed value before the residual is added.
    float scale;
    /// bit 0: multiply by `weight`; bit 1: SiLU; bit 2: add `resid`.
    uint flags;
};

/// `out[r] = [resid[r] +] scale * clamp_post(act(rms_norm(clamp2(clamp1(x[r]))) [* weight]))`
/// with `rms_norm(v) = v * (mean(v^2) + eps)^-0.5` (`Gemma4RMSNorm`: `* w`, not
/// `* (1 + w)`; a missing weight is the multimodal embedder's weightless norm).
/// One threadgroup per row. `out` may be `x`, and `resid` may be `out`: each
/// thread reads an element before it writes that element, and the reduction
/// between the two passes separates every thread's reads from every write.
kernel void eg2a_rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *resid [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant EgaNormParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= p.rows) return;
    threadgroup float scratch[REDUCE_MAX_SIMDGROUPS];
    const ulong base = (ulong)row * p.dim;
    float ss = 0.0f;
    for (uint d = lid; d < p.dim; d += tptg) {
        const float v = clamp(clamp(x[base + d], p.pre1_lo, p.pre1_hi), p.pre2_lo, p.pre2_hi);
        ss += v * v;
    }
    const float inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)p.dim + p.eps);
    for (uint d = lid; d < p.dim; d += tptg) {
        float v = clamp(clamp(x[base + d], p.pre1_lo, p.pre1_hi), p.pre2_lo, p.pre2_hi) * inv;
        if ((p.flags & 1u) != 0u) v *= weight[d];
        if ((p.flags & 2u) != 0u) v = tessl_silu(v);
        v = clamp(v, p.post_lo, p.post_hi) * p.scale;
        if ((p.flags & 4u) != 0u) v += resid[base + d];
        out[base + d] = v;
    }
}

/// Words of `EgaEwParams`.
struct EgaEwParams {
    uint n;
    /// Columns of the row `aux` is indexed by in bias mode.
    uint cols;
    float lo1;
    float hi1;
    float lo2;
    float hi2;
    /// bit 1: SiLU between the clamps; bit 2: add `aux[i]` (a residual);
    /// bit 3: add `aux[i % cols]` (a per-column bias).
    uint flags;
};

/// `out[i] = clamp2(act(clamp1(x[i]))) [+ aux[i] | + aux[i % cols]]`: the pass
/// between two clipped linears (output clamp, SiLU, input clamp), a linear's
/// output clamp plus the residual add, an input clamp on its own, and the
/// output projection's bias. `out` may be `x`.
kernel void eg2a_ew(
    device const float *x [[buffer(0)]],
    device const float *aux [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant EgaEwParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n) return;
    float v = clamp(x[i], p.lo1, p.hi1);
    if ((p.flags & 2u) != 0u) v = tessl_silu(v);
    v = clamp(v, p.lo2, p.hi2);
    if ((p.flags & 4u) != 0u) v += aux[i];
    if ((p.flags & 8u) != 0u) v += aux[i % p.cols];
    out[i] = v;
}

/// `torch.nn.functional.glu(x, dim=-1)` after the clipped linear's output
/// clamp: `out[r, c] = clamp(x[r, c]) * sigmoid(clamp(x[r, H + c]))`. x [rows, 2H].
kernel void eg2a_glu(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant float &lo [[buffer(4)]],
    constant float &hi [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const ulong r = i / hidden;
    const uint c = i % hidden;
    const float a = clamp(x[r * 2u * hidden + c], lo, hi);
    const float b = clamp(x[r * 2u * hidden + hidden + c], lo, hi);
    out[i] = a * tessl_sigmoid(b);
}

/// `Gemma4AudioCausalConv1d` (depthwise, kernel `taps`, left pad `taps - 1`):
/// `out[t, c] = sum_j w[c, j] * x[t - (taps - 1) + j, c]`, rows before the
/// clip's first reading as 0. x, out [R, hidden]; w [hidden, taps] (the
/// checkpoint's `[hidden, 1, taps]`). Grid: x = channels, y = clip-local row,
/// z = clip.
kernel void eg2a_dwconv1d_causal(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    device const uint *clips [[buffer(3)]],
    constant uint &hidden [[buffer(4)]],
    constant uint &taps [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint ch = gid.x;
    if (ch >= hidden) return;
    device const uint *c = clips + (ulong)gid.z * EGA_CLIP_WORDS;
    const uint row0 = c[4];
    const uint t = gid.y;
    if (t >= c[5]) return;
    float acc = 0.0f;
    for (uint j = 0u; j < taps; ++j) {
        const int s = (int)t - (int)(taps - 1u) + (int)j;
        if (s >= 0) acc += w[(ulong)ch * taps + j] * x[(ulong)(row0 + (uint)s) * hidden + ch];
    }
    out[(ulong)(row0 + t) * hidden + ch] = acc;
}

// --- Chunked local attention with Transformer-XL relative positions -----------

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
    /// `attention_logit_cap`, and the value a masked logit takes
    /// (`attention_invalid_logits_value`).
    float cap;
    float invalid;
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
};

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
/// frame of the clip and `omask[kp]` (the padding mask, on keys only). An
/// invisible logit is `invalid` (a finite -1e9, as `masked_fill` writes it),
/// so a row whose keys are all invisible softmaxes to the uniform `1 / ctx`
/// over its window, values included, exactly as transformers computes (and
/// then nothing reads that row).
///
/// q, k, v, out [R, heads * head_dim]; rel [rel_rows, heads * head_dim];
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
            ? clamp(q[(ulong)(row0 + t) * hidden + col0 + d], p.q_lo, p.q_hi) * q_mult[d]
            : 0.0f;
    }

    const uint blk = t / p.chunk;
    const uint i = t % p.chunk;
    float lg[EGA_MAX_CTX];
    for (uint j = 0u; j < p.ctx; ++j) {
        const int kp = (int)(blk * p.chunk + j) - (int)p.past;
        const int dist = (int)t - kp;
        float s = p.invalid;
        if (dist >= 0 && dist < (int)p.past && kp >= 0 && kp < (int)rows && omask[row0 + (uint)kp] != 0u) {
            device const float *kr = k + (ulong)(row0 + (uint)kp) * hidden + col0;
            float ac = 0.0f;
            for (uint e = 0u; e < EGA_MAX_E; ++e) {
                const uint d = lane + 32u * e;
                if (d < p.head_dim) ac += qs[e] * (clamp(kr[d], p.k_lo, p.k_hi) * p.k_scale);
            }
            float bd = 0.0f;
            if (j >= i && j - i < p.rel_rows) {
                device const float *rr = rel + (ulong)(j - i) * hidden + col0;
                for (uint e = 0u; e < EGA_MAX_E; ++e) {
                    const uint d = lane + 32u * e;
                    if (d < p.head_dim) bd += qs[e] * rr[d];
                }
            }
            s = simd_sum(ac) + simd_sum(bd);
            // `tanh(s / cap) * cap`; plain tanh lowers to a fast tanh that
            // is not trusted past |10|, and tanh has saturated well before 16.
            s = precise::tanh(clamp(s / p.cap, -16.0f, 16.0f)) * p.cap;
        }
        lg[j] = s;
    }

    float m = lg[0];
    for (uint j = 1u; j < p.ctx; ++j) m = max(m, lg[j]);
    float den = 0.0f;
    for (uint j = 0u; j < p.ctx; ++j) {
        lg[j] = exp(lg[j] - m);
        den += lg[j];
    }

    float acc[EGA_MAX_E];
    for (uint e = 0u; e < EGA_MAX_E; ++e) acc[e] = 0.0f;
    for (uint j = 0u; j < p.ctx; ++j) {
        const int kp = (int)(blk * p.chunk + j) - (int)p.past;
        const float wgt = lg[j] / den;
        // Window slots outside the clip hold zero values (`F.pad`).
        if (kp < 0 || kp >= (int)rows || wgt == 0.0f) continue;
        device const float *vr = v + (ulong)(row0 + (uint)kp) * hidden + col0;
        for (uint e = 0u; e < EGA_MAX_E; ++e) {
            const uint d = lane + 32u * e;
            if (d < p.head_dim) acc[e] += wgt * clamp(vr[d], p.v_lo, p.v_hi);
        }
    }
    device float *o = out + (ulong)(row0 + t) * hidden + col0;
    for (uint e = 0u; e < EGA_MAX_E; ++e) {
        const uint d = lane + 32u * e;
        if (d < p.head_dim) o[d] = clamp(acc[e], p.o_lo, p.o_hi);
    }
}
