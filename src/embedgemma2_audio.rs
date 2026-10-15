//! EmbeddingGemma 2's audio tower and its multimodal embedder
//! (`google/embeddinggemma-2`'s `audio_tower.*` and `embed_audio.*`), composed
//! from tessl kernels and loaded straight from `model.safetensors`.
//!
//! The input is what the checkpoint's audio processor produces: a 128-bin
//! log-mel `input_features` `[frames, 128]` and its frame mask `[frames]`
//! (preprocessing stays the caller's, as tokenization does). The output is the
//! 512-wide soft tokens the text encoder (`crate::embedgemma2`) takes in place
//! of the audio placeholder rows: one per kept subsampled frame.
//!
//! # Forward (transformers 5.19 `Gemma4AudioModel`, `EmbeddingGemma2MultimodalEmbedder`)
//!
//! ```text
//! x      = frames * mask                                   a masked frame reads 0
//! x      = relu(layer_norm_C(conv3x3_s2(x)))               C: 1 -> 128, then (frames * mask[::2]) -> 32
//! x      = flatten(x, (freq, channel)) @ input_proj        [T, 32 * 32] -> [T, 1024]; T = ceil(ceil(F/2)/2)
//! per layer, `clip` = clamp(+-gradient_clipping), `L` = a clipped linear (clamp in, matmul, clamp out):
//!   x  = x + 0.5 * post_norm(clip(L2(silu(L1(pre_norm(clip(x)))))))            feed_forward1
//!   a  = attn(norm_pre_attn(clip(x)));  x = x + norm_post_attn(clip(post(a)))
//!   x  = x + linear_end(silu(conv_norm(clip(depthwise_causal_conv5(glu(linear_start(pre_norm(x))))))))
//!   x  = x + 0.5 * post_norm(clip(L2(silu(L1(pre_norm(clip(x)))))))            feed_forward2
//!   x  = norm_out(clip(x))
//! tower  = x @ output_proj + bias                          [T, 1536]
//! soft   = rms_norm_noweight(tower) @ embedding_projection [T, 512], kept where the tower's mask is true
//! ```
//!
//! Every norm is `Gemma4RMSNorm` (`x * (mean(x^2) + eps)^-0.5 * w`, in f32). The
//! tower's mask is `mask[::4]` (each conv keeps every other frame). The
//! attention is chunked and local ([`rel_shift_source`], [`key_visible`]):
//! queries come in blocks of `chunk` (12), each block sees a window of
//! `chunk + past` (24) keys, a query attends to itself and the `past - 1` (11)
//! frames before it, with Transformer-XL relative-position logits from the
//! sinusoidal table through `relative_k_proj`, a `softplus(per_dim_scale)`
//! query scale, a tanh softcap of 50 and masked logits at -1e9.
//!
//! # The attention mask, and a transformers eager bug
//!
//! `Gemma4AudioAttention.forward` masks with `attention_mask.logical_not()`,
//! which is right for the bool mask the sdpa path builds. Under
//! `attn_implementation="eager"`, `create_bidirectional_mask` returns a FLOAT
//! additive mask (0.0 visible, `finfo.min` not), and `logical_not()` of that
//! is true exactly where the key IS visible: the visible keys are filled with
//! -1e9 and the keys outside the window are attended to. The model never calls
//! `scaled_dot_product_attention`, so the setting changes nothing but the mask's
//! dtype; sdpa is the specified behaviour, and this module implements it.
//! `tools/embedgemma2_ref/make_audio_sdpa_reference.py` writes the audio
//! references with the bool mask. `docs/embedgemma2.md` has the measured
//! difference: a reference taken under eager disagrees from layer 0.
//!
//! # Precision
//!
//! Weights are widened from the checkpoint's bf16 to f32 exactly, every
//! activation is f32 and every GEMM is exact f32: transformers' fp32 forward up
//! to operation order. It needs the runtime's relaxed-f32 GEMM switched off
//! (the default). A clip bound of infinity (the module's untrained default) is
//! held as `+-f32::MAX`, which clamps nothing.
//!
//! # Rows
//!
//! A batch is its clips laid end to end in each row space (frames, the first
//! conv's rows, tower rows) and run as one forward, up to
//! [`MAX_FRAMES_PER_FORWARD`] frames (more clips are more forwards, all queued
//! before the one wait). A clip's rows never read another's: the convolutions
//! and the attention work in clip-local indices and the GEMMs and norms are
//! row-wise, so a clip's output does not depend on what it is batched with. The
//! one place a padded (masked) frame influences a kept one is the subsampling's
//! zeroing before each conv, which is defined by the mask.
//!
//! # Kernels
//!
//! New, in `kernels/embedgemma2_audio.metal`: [`sub_conv0_ln_relu`] and
//! [`sub_conv1_ln_relu`] (the strided convs with the frame mask, channel
//! LayerNorm and ReLU), [`rms_norm`] and [`ew`] (the norms and elementwise
//! passes with the clipped linears' and the gradient clamps folded in),
//! [`glu`], [`dwconv1d_causal`] and [`attention`]. The matmuls are tessl's
//! exact-f32 [`gemm`]; `gemm_epilogue` has no exact-f32 path, so the clamps
//! are not fused into them. The sinusoidal table, `relative_k_proj` of it and
//! the query scale are computed once per layer at load.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_gpu_buf, set_u32};
use crate::gemm::{gemm, select_backend, GemmBackend};
use crate::json::{self, Json, Syntax};
use crate::loader::Loader;
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes};
use crate::qwen35_model::Precision;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{gpu_copy, DType, GpuBuffer, Tensor};

/// Frames (rows of `input_features`) one forward holds. Bounds the largest
/// activation, the first conv's output `[ceil(F/2), 64 * 128]`: 268 MB of f32
/// at this size. [`EmbedGemma2Audio::encode_audio`] splits a batch into as
/// many forwards as this needs, so it bounds memory, not what a caller may
/// pass; a single clip above it is refused.
pub const MAX_FRAMES_PER_FORWARD: usize = 16_384;
/// Clips per [`EmbedGemma2Audio::encode_audio`] call.
pub const MAX_CLIPS: usize = 256;

/// Words per clip in the table the kernels read (`EGA_CLIP_WORDS`).
const CLIP_WORDS: usize = 8;
/// Limits the kernels are compiled for (`EGA_MAX_*`).
const MAX_C0: u32 = 512;
const MAX_C1: u32 = 128;
const MAX_CTX: u32 = 64;
const MAX_HEAD_DIM: u32 = 256;

const PREFIX_TOWER: &str = "audio_tower.";
const PREFIX_EMBED: &str = "embed_audio.";

// ---------------------------------------------------------------------------
// Host definitions of the attention's index math
// ---------------------------------------------------------------------------

/// Frames the output of a stride-2, pad-1, 3x3 convolution has: `ceil(n / 2)`.
pub fn conv_out(n: usize) -> usize {
    n.div_ceil(2)
}

/// Rows of the tower for a clip of `frames` frames: two convolutions, so
/// `ceil(ceil(frames / 2) / 2)`. The tower's mask is `mask[::4]` of the same
/// length (`mask[::2]` per convolution).
pub fn sub_rows(frames: usize) -> usize {
    conv_out(conv_out(frames))
}

/// Whether the key at frame `kp` is visible to the query at frame `t` in a clip
/// of `rows` tower rows whose padding mask is `omask`
/// (`create_bidirectional_mask(..., and_mask_function=sliding_window_mask_function
/// ((past, 0)))` followed by `_convert_4d_mask_to_blocked_5d`).
///
/// The window function admits `0 <= t - kp < past` (the query and the `past - 1`
/// frames before it, no future); the padding mask is on the key only; a window
/// slot outside the clip `[0, rows)` is padding, `False`. `past` is
/// `attention_context_left - 1` (12), so a query sees 12 keys, not the 13 the
/// config's "context left 13" suggests.
pub fn key_visible(t: usize, kp: i64, past: usize, rows: usize, omask: &[bool]) -> bool {
    let dist = t as i64 - kp;
    dist >= 0 && dist < past as i64 && kp >= 0 && (kp as usize) < rows && omask[kp as usize]
}

/// Where logit `bd(i, j)` of block row `i` and window slot `j` reads its value
/// in the `[chunk, rel_rows]` product of the queries with the relative-key
/// table: `Some(column)` of row `i`, or `None` for the zero padding.
///
/// transformers' `_rel_shift` pads each row of that product to `ctx + 1`
/// columns, flattens the block, drops the tail and views the rest as `[chunk,
/// ctx]`: output `(i, j)` is flat index `i * ctx + j = i * (ctx + 1) + (j - i)`,
/// which for `j >= i` is column `j - i` of row `i` (padding if `j - i >=
/// rel_rows`), and for `j < i` is column `ctx + 1 + j - i` of row `i - 1`,
/// which is padding too whenever `ctx + 1 + j - i >= rel_rows` for every `j <
/// i` ([`AudioConfig`] refuses a configuration where it is not). Row `p` of the
/// table holds relative position `ctx / 2 - p` (the table is built from
/// `arange(ctx / 2, -1, -1)`), so a visible slot, at distance `t - kp = past -
/// (j - i)`, reads row `j - i`, and row 0 (distance `past`) is never visible.
pub fn rel_shift_source(i: usize, j: usize, rel_rows: usize) -> Option<usize> {
    (j >= i && j - i < rel_rows).then_some(j - i)
}

/// The sinusoidal relative-position table `[ctx / 2 + 1, hidden]`
/// (`Gemma4AudioRelPositionalEncoding`): row `p` is `[sin(pos * f_d) for d <
/// hidden/2] ++ [cos(pos * f_d)]` at `pos = ctx / 2 - p` and `f_d = exp(-d *
/// ln(10000) / max(hidden/2 - 1, 1))`. Computed in f64 (transformers rounds
/// each step to f32).
pub fn rel_position_table(hidden: usize, ctx: usize) -> Vec<f64> {
    let n = hidden / 2;
    let inc = (10000.0f64 / 1.0).ln() / (n.saturating_sub(1).max(1) as f64);
    let rows = ctx / 2 + 1;
    let mut out = vec![0.0f64; rows * hidden];
    for p in 0..rows {
        let pos = (ctx / 2 - p) as f64;
        for d in 0..n {
            let t = pos * (-(d as f64) * inc).exp();
            out[p * hidden + d] = t.sin();
            out[p * hidden + n + d] = t.cos();
        }
    }
    out
}

/// `relative_k_proj(table)`: `table [rows, hidden] @ w^T` with `w [hidden,
/// hidden]` as `nn.Linear` stores it, summed in f64 and rounded once.
fn relative_keys(table: &[f64], w: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for n in 0..hidden {
        let wrow = &w[n * hidden..(n + 1) * hidden];
        for p in 0..rows {
            let trow = &table[p * hidden..(p + 1) * hidden];
            let s: f64 = trow.iter().zip(wrow).map(|(a, b)| a * f64::from(*b)).sum();
            out[p * hidden + n] = s as f32;
        }
    }
    out
}

/// `softplus(x)` as torch computes it (threshold 20, above which it is `x`).
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The audio tower's shapes, read from `config.json` (`audio_config`, and the
/// text config's hidden size for the embedder's output).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioConfig {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    /// Depthwise conv taps (`conv_kernel_size`).
    pub conv_kernel: u32,
    /// Output channels of the two subsampling convolutions.
    pub sub_channels: [u32; 2],
    /// `output_proj_dims`: the tower's output width.
    pub out_dims: u32,
    /// The text encoder's hidden size: the soft tokens' width.
    pub embed_dim: u32,
    pub residual_weight: f32,
    pub logit_cap: f32,
    /// The logit a masked key takes (`attention_invalid_logits_value`).
    pub invalid_logit: f32,
    /// Queries per attention block (`attention_chunk_size`).
    pub chunk: u32,
    /// `attention_context_left`; the mask admits `context_left - 1` frames.
    pub context_left: u32,
    pub eps: f32,
    /// `min(gradient_clipping, f32::MAX)`.
    pub gradient_clip: f32,
    /// Whether the linears carry clip bounds (`use_clipped_linears`).
    pub clipped_linears: bool,
}

impl AudioConfig {
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.heads
    }

    /// `attention_context_left - 1`: how many frames, the query included, a
    /// query sees.
    pub fn past(&self) -> u32 {
        self.context_left - 1
    }

    /// Keys in a block's window (`chunk + past`; no future context).
    pub fn ctx(&self) -> u32 {
        self.chunk + self.past()
    }

    /// Rows of the relative-position table (`ctx / 2 + 1`).
    pub fn rel_rows(&self) -> u32 {
        self.ctx() / 2 + 1
    }

    /// Mel bins `input_features` carries. transformers sizes the input
    /// projection `(c0 / 4) * c1` for it, which holds only when the bins equal
    /// the first conv's channels: the checkpoint's 128.
    pub fn mel_bins(&self) -> u32 {
        self.sub_channels[0]
    }

    /// Features the input projection takes: frequency bins left after the two
    /// convolutions (`mel_bins / 4`) times the second one's channels.
    pub fn proj_in(&self) -> u32 {
        self.sub_channels[0] / 4 * self.sub_channels[1]
    }

    /// Parse the checkpoint's `config.json` (the multimodal root with an
    /// `audio_config` and a `text_config`, or a bare audio config plus
    /// `text_hidden`). What the forward does not implement is refused by name.
    pub fn from_config_json(text: &str) -> Result<Self, String> {
        const SYNTAX: Syntax = Syntax {
            what: "config.json",
            max_depth: 16,
            uints_only: false,
            literals: true,
        };
        let root = json::parse(text, SYNTAX)?;
        if !matches!(root, Json::Object(_)) {
            return Err("config.json: the root is not an object".into());
        }
        let c = root
            .get("audio_config")
            .ok_or("config.json: missing \"audio_config\"")?;
        let uint_in = |obj: &Json, k: &str| -> Result<u32, String> {
            match obj.get(k) {
                Some(Json::Num { uint: Some(n), .. }) => {
                    u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32"))
                }
                Some(v) => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        let uint = |k: &str| uint_in(c, k);
        let num = |k: &str| -> Result<f64, String> {
            match c.get(k) {
                Some(Json::Num { value, .. }) => Ok(*value),
                Some(v) => Err(format!("config.json: {k} must be a number, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        if let Some(Json::Str(t)) = c.get("model_type") {
            if t != "gemma4_audio" {
                return Err(format!("config.json: audio model_type {t:?} is not gemma4_audio"));
            }
        }
        match c.get("hidden_act") {
            Some(Json::Str(a)) if a == "silu" => {}
            other => return Err(format!("config.json: audio hidden_act must be silu, got {other:?}")),
        }
        let clipped_linears = match c.get("use_clipped_linears") {
            Some(Json::Bool(b)) => *b,
            other => {
                return Err(format!(
                    "config.json: use_clipped_linears must be a bool, got {other:?}"
                ))
            }
        };
        let right = uint("attention_context_right")?;
        if right != 0 {
            return Err(format!(
                "config.json: attention_context_right = {right} is not supported (the checkpoint has none)"
            ));
        }
        let channels: Vec<u32> = match c.get("subsampling_conv_channels") {
            Some(Json::Array(a)) => a
                .iter()
                .map(|v| match v {
                    Json::Num { uint: Some(n), .. } => {
                        u32::try_from(*n).map_err(|_| "config.json: a subsampling channel exceeds u32".to_string())
                    }
                    v => Err(format!("config.json: subsampling_conv_channels entry {v:?}")),
                })
                .collect::<Result<_, _>>()?,
            other => {
                return Err(format!(
                    "config.json: subsampling_conv_channels must be an array, got {other:?}"
                ))
            }
        };
        let [c0, c1] = channels[..] else {
            return Err(format!(
                "config.json: subsampling_conv_channels has {} entries, need 2",
                channels.len()
            ));
        };
        let text_hidden = match root.get("text_config") {
            Some(t) => uint_in(t, "hidden_size")?,
            None => {
                return Err("config.json: missing \"text_config\" (the soft tokens' width is its hidden_size)".into())
            }
        };
        let cfg = Self {
            hidden: uint("hidden_size")?,
            layers: uint("num_hidden_layers")?,
            heads: uint("num_attention_heads")?,
            conv_kernel: uint("conv_kernel_size")?,
            sub_channels: [c0, c1],
            out_dims: uint("output_proj_dims")?,
            embed_dim: text_hidden,
            residual_weight: num("residual_weight")? as f32,
            logit_cap: num("attention_logit_cap")? as f32,
            invalid_logit: num("attention_invalid_logits_value")? as f32,
            chunk: uint("attention_chunk_size")?,
            context_left: uint("attention_context_left")?,
            eps: num("rms_norm_eps")? as f32,
            gradient_clip: num("gradient_clipping")?.min(f64::from(f32::MAX)) as f32,
            clipped_linears,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Everything the kernels are compiled for, and the identities the index
    /// math relies on.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |m: String| Err(format!("config.json: audio {m}"));
        for (k, v) in [
            ("hidden_size", self.hidden),
            ("num_hidden_layers", self.layers),
            ("num_attention_heads", self.heads),
            ("conv_kernel_size", self.conv_kernel),
            ("output_proj_dims", self.out_dims),
            ("text hidden_size", self.embed_dim),
            ("attention_chunk_size", self.chunk),
            ("subsampling_conv_channels[0]", self.sub_channels[0]),
            ("subsampling_conv_channels[1]", self.sub_channels[1]),
        ] {
            if v == 0 {
                return bad(format!("{k} is zero"));
            }
        }
        if self.context_left < 2 {
            return bad(format!(
                "attention_context_left = {} leaves no past frame (need at least 2)",
                self.context_left
            ));
        }
        if self.hidden % self.heads != 0 || self.hidden % 2 != 0 {
            return bad(format!(
                "hidden_size {} must be even and a multiple of num_attention_heads {}",
                self.hidden, self.heads
            ));
        }
        if self.head_dim() > MAX_HEAD_DIM {
            return bad(format!("head dim {} exceeds {MAX_HEAD_DIM}", self.head_dim()));
        }
        if self.ctx() > MAX_CTX {
            return bad(format!(
                "attention window chunk + past = {} exceeds {MAX_CTX}",
                self.ctx()
            ));
        }
        // `rel_shift_source`'s second branch (j < i) must always be padding.
        if self.ctx() + 2 < self.chunk + self.rel_rows() {
            return bad(format!(
                "chunk {} with context_left {} reads shifted rows of the relative table; unsupported",
                self.chunk, self.context_left
            ));
        }
        if self.sub_channels[0] % 4 != 0 || self.sub_channels[0] > MAX_C0 || self.sub_channels[1] > MAX_C1 {
            return bad(format!(
                "subsampling_conv_channels {:?}: the first must be a multiple of 4 and at most {MAX_C0}, the second at most {MAX_C1}",
                self.sub_channels
            ));
        }
        for (k, v) in [
            ("rms_norm_eps", self.eps),
            ("attention_logit_cap", self.logit_cap),
            ("gradient_clipping", self.gradient_clip),
        ] {
            if !(v > 0.0 && v.is_finite()) {
                return bad(format!("{k} must be positive and finite, got {v}"));
            }
        }
        if !(self.residual_weight.is_finite() && self.invalid_logit.is_finite()) {
            return bad("residual_weight and attention_invalid_logits_value must be finite".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

const WHAT: &str = "embedgemma2_audio";

/// A clamp `[lo, hi]`. "No bound" is `+-f32::MAX`: the kernels never see an
/// infinity (fast math may assume there is none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clamp {
    pub lo: f32,
    pub hi: f32,
}

impl Clamp {
    pub const NONE: Clamp = Clamp {
        lo: -f32::MAX,
        hi: f32::MAX,
    };

    /// `[lo, hi]` with an infinite bound held as `+-f32::MAX`. NaN is refused.
    /// `lo > hi` is allowed and gives `hi`, as `torch.clamp` does.
    pub fn new(lo: f32, hi: f32) -> Result<Clamp, String> {
        if lo.is_nan() || hi.is_nan() {
            return Err(format!("{WHAT}: a clamp bound is NaN ([{lo}, {hi}])"));
        }
        Ok(Clamp {
            lo: lo.max(-f32::MAX),
            hi: hi.min(f32::MAX),
        })
    }

    /// `[-b, b]`.
    pub fn symmetric(b: f32) -> Clamp {
        Clamp {
            lo: -b.min(f32::MAX),
            hi: b.min(f32::MAX),
        }
    }
}

fn word(f: f32) -> u32 {
    f.to_bits()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

fn elems(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| format!("{what}: element count {dims:?} overflows usize"))
}

/// One clip's rows in each row space (see the kernel file's header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRows {
    pub frame0: usize,
    pub frames: usize,
    pub mid0: usize,
    pub mid_rows: usize,
    pub row0: usize,
    pub rows: usize,
}

/// The clip table the kernels read, and the sizes of the batch it describes.
pub struct ClipTable {
    pub buffer: GpuBuffer,
    pub clips: Vec<ClipRows>,
    /// Totals over the clips: frames, first-conv rows, tower rows.
    pub frames: usize,
    pub mid_rows: usize,
    pub rows: usize,
}

impl ClipTable {
    /// The table for clips of `frames[i]` frames each, uploaded without waiting
    /// for the GPU ([`GpuRuntime::alloc_buffer_from_u32`]).
    pub fn new(rt: &Arc<GpuRuntime>, frames: &[usize]) -> Result<ClipTable, String> {
        if frames.is_empty() || frames.len() > MAX_CLIPS {
            return Err(format!("{WHAT}: {} clips; need 1..={MAX_CLIPS}", frames.len()));
        }
        let (mut clips, mut words) = (Vec::with_capacity(frames.len()), Vec::new());
        let (mut f0, mut m0, mut r0) = (0usize, 0usize, 0usize);
        for (i, &f) in frames.iter().enumerate() {
            if f == 0 || f > MAX_FRAMES_PER_FORWARD {
                return Err(format!(
                    "{WHAT}: clip {i} has {f} frames; need 1..={MAX_FRAMES_PER_FORWARD}"
                ));
            }
            let c = ClipRows {
                frame0: f0,
                frames: f,
                mid0: m0,
                mid_rows: conv_out(f),
                row0: r0,
                rows: sub_rows(f),
            };
            for v in [c.frame0, c.frames, c.mid0, c.mid_rows, c.row0, c.rows] {
                words.push(u32::try_from(v).map_err(|_| format!("{WHAT}: row index {v} exceeds u32"))?);
            }
            words.extend([0u32; CLIP_WORDS - 6]);
            (f0, m0, r0) = (f0 + f, m0 + c.mid_rows, r0 + c.rows);
            clips.push(c);
        }
        if f0 > MAX_FRAMES_PER_FORWARD {
            return Err(format!(
                "{WHAT}: {f0} frames in one forward; at most {MAX_FRAMES_PER_FORWARD}"
            ));
        }
        Ok(ClipTable {
            buffer: rt.alloc_buffer_from_u32(&words)?,
            clips,
            frames: f0,
            mid_rows: m0,
            rows: r0,
        })
    }

    fn max_mid_rows(&self) -> usize {
        self.clips.iter().map(|c| c.mid_rows).max().unwrap_or(0)
    }

    fn max_rows(&self) -> usize {
        self.clips.iter().map(|c| c.rows).max().unwrap_or(0)
    }
}

fn check_table(rt: &GpuRuntime, t: &ClipTable, what: &str) -> Result<(), String> {
    require::<u32>(rt, &t.buffer, t.clips.len() * CLIP_WORDS, what)
}

/// The first subsampling layer: for every clip, `out = relu(layer_norm_C(
/// conv3x3_s2(feats * fmask)))` with `C = channels`, laid out `[mid rows,
/// mels1, channels]`. `feats [frames, mels]`, `fmask [frames]` (u32 0/1),
/// `w [channels, 9]`, `gamma [channels]`, `out [mid rows, mels1 * channels]`
/// where `mels1 = ceil(mels / 2)`. `channels <= 512`.
#[allow(clippy::too_many_arguments)]
pub fn sub_conv0_ln_relu(
    rt: &Arc<GpuRuntime>,
    feats: &GpuBuffer,
    fmask: &GpuBuffer,
    w: &GpuBuffer,
    gamma: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    mels: u32,
    channels: u32,
    eps: f32,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::sub_conv0_ln_relu";
    if mels == 0 || channels == 0 || channels > MAX_C0 || !(eps > 0.0 && eps.is_finite()) {
        return Err(format!(
            "{E}: need mels > 0, 1 <= channels <= {MAX_C0}, eps > 0 (got {mels}, {channels}, {eps})"
        ));
    }
    let mels1 = conv_out(mels as usize) as u32;
    check_table(rt, clips, E)?;
    require::<f32>(rt, feats, elems(&[clips.frames as u32, mels], E)?, "sub_conv0 feats")?;
    require::<u32>(rt, fmask, clips.frames, "sub_conv0 fmask")?;
    require::<f32>(rt, w, channels as usize * 9, "sub_conv0 w")?;
    require::<f32>(rt, gamma, channels as usize, "sub_conv0 gamma")?;
    require::<f32>(
        rt,
        out,
        elems(&[clips.mid_rows as u32, mels1, channels], E)?,
        "sub_conv0 out",
    )?;
    require_disjoint_writes(
        E,
        &[("out", out)],
        &[
            ("feats", feats),
            ("fmask", fmask),
            ("w", w),
            ("gamma", gamma),
            ("clips", &clips.buffer),
        ],
    )?;
    let p = rt.pipeline("eg2a_sub_conv0_ln_relu")?;
    let tptg = (channels as usize)
        .next_multiple_of(32)
        .min(256)
        .min(p.maxTotalThreadsPerThreadgroup());
    dispatch_2d_tg(
        rt,
        &p,
        clips.max_mid_rows() * mels1 as usize,
        clips.clips.len(),
        tptg,
        |bnd| {
            set_gpu_buf(bnd, feats, 0);
            set_gpu_buf(bnd, fmask, 1);
            set_gpu_buf(bnd, w, 2);
            set_gpu_buf(bnd, gamma, 3);
            set_gpu_buf(bnd, out, 4);
            set_gpu_buf(bnd, &clips.buffer, 5);
            set_u32(bnd, mels, 6);
            set_u32(bnd, mels1, 7);
            set_u32(bnd, channels, 8);
            crate::dispatch::set_f32(bnd, eps, 9);
        },
    )
}

/// The second subsampling layer: `out[t2, f2 * c1 + co] = relu(layer_norm_c1(
/// conv3x3_s2(mid * mask[::2])))` for the tower rows of every clip. `mid
/// [mid rows, mels1 * c0]` is [`sub_conv0_ln_relu`]'s output, `fmask [frames]`
/// the frame mask (read at every other frame), `w [9, c0, c1]` (tap, input
/// channel, output channel), `gamma [c1]`, `out [rows, mels2 * c1]` with
/// `mels2 = ceil(mels1 / 2)`. `c1 <= 128`.
#[allow(clippy::too_many_arguments)]
pub fn sub_conv1_ln_relu(
    rt: &Arc<GpuRuntime>,
    mid: &GpuBuffer,
    fmask: &GpuBuffer,
    w: &GpuBuffer,
    gamma: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    mels1: u32,
    c0: u32,
    c1: u32,
    eps: f32,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::sub_conv1_ln_relu";
    if mels1 == 0 || c0 == 0 || c1 == 0 || c1 > MAX_C1 || !(eps > 0.0 && eps.is_finite()) {
        return Err(format!(
            "{E}: need mels1, c0 > 0, 1 <= c1 <= {MAX_C1}, eps > 0 (got {mels1}, {c0}, {c1}, {eps})"
        ));
    }
    let mels2 = conv_out(mels1 as usize) as u32;
    check_table(rt, clips, E)?;
    require::<f32>(rt, mid, elems(&[clips.mid_rows as u32, mels1, c0], E)?, "sub_conv1 mid")?;
    require::<u32>(rt, fmask, clips.frames, "sub_conv1 fmask")?;
    require::<f32>(rt, w, elems(&[9, c0, c1], E)?, "sub_conv1 w")?;
    require::<f32>(rt, gamma, c1 as usize, "sub_conv1 gamma")?;
    require::<f32>(rt, out, elems(&[clips.rows as u32, mels2, c1], E)?, "sub_conv1 out")?;
    require_disjoint_writes(
        E,
        &[("out", out)],
        &[
            ("mid", mid),
            ("fmask", fmask),
            ("w", w),
            ("gamma", gamma),
            ("clips", &clips.buffer),
        ],
    )?;
    let p = rt.pipeline("eg2a_sub_conv1_ln_relu")?;
    dispatch_2d_tg(
        rt,
        &p,
        clips.max_rows() * mels2 as usize,
        clips.clips.len(),
        32,
        |bnd| {
            set_gpu_buf(bnd, mid, 0);
            set_gpu_buf(bnd, fmask, 1);
            set_gpu_buf(bnd, w, 2);
            set_gpu_buf(bnd, gamma, 3);
            set_gpu_buf(bnd, out, 4);
            set_gpu_buf(bnd, &clips.buffer, 5);
            set_u32(bnd, mels1, 6);
            set_u32(bnd, mels2, 7);
            set_u32(bnd, c0, 8);
            set_u32(bnd, c1, 9);
            crate::dispatch::set_f32(bnd, eps, 10);
        },
    )
}

/// What [`rms_norm`] computes per row: `[resid +] scale * clamp(post)(act(rms_norm(
/// clamp(pre[1])(clamp(pre[0])(x))) [* weight]))`.
pub struct RmsNorm<'a> {
    pub eps: f32,
    /// `None`: the weightless norm of the multimodal embedder.
    pub weight: Option<&'a GpuBuffer>,
    /// Two clamps applied to the input in turn: a clipped linear's output
    /// bounds, then `+-gradient_clipping`.
    pub pre: [Clamp; 2],
    pub silu: bool,
    /// The next clipped linear's input bounds.
    pub post: Clamp,
    /// Multiplies the normed value before the residual is added.
    pub scale: f32,
    pub resid: Option<&'a GpuBuffer>,
}

impl<'a> RmsNorm<'a> {
    /// `rms_norm(x) * weight`, no clamps.
    pub fn new(eps: f32, weight: Option<&'a GpuBuffer>) -> Self {
        Self {
            eps,
            weight,
            pre: [Clamp::NONE; 2],
            silu: false,
            post: Clamp::NONE,
            scale: 1.0,
            resid: None,
        }
    }
}

/// [`RmsNorm`] over `rows` rows of `dim` each, `x` into `out` (`out` may be `x`,
/// and `resid` may be `out`).
pub fn rms_norm(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    spec: &RmsNorm<'_>,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::rms_norm";
    if dim == 0 || !(spec.eps > 0.0 && spec.eps.is_finite()) || !spec.scale.is_finite() {
        return Err(format!("{E}: need dim > 0, eps > 0 and a finite scale"));
    }
    let n = elems(&[rows, dim], E)?;
    require::<f32>(rt, x, n, "rms_norm x")?;
    require::<f32>(rt, out, n, "rms_norm out")?;
    let mut flags = 0u32;
    if let Some(w) = spec.weight {
        require::<f32>(rt, w, dim as usize, "rms_norm weight")?;
        flags |= 1;
    }
    if spec.silu {
        flags |= 2;
    }
    if let Some(r) = spec.resid {
        require::<f32>(rt, r, n, "rms_norm resid")?;
        flags |= 4;
    }
    if rows == 0 {
        return Ok(());
    }
    // `out` is written element by element after a barrier-separated read of
    // the same element by the same thread, so it may be `x` or `resid`.
    if let Some(w) = spec.weight {
        require_disjoint_writes(E, &[("out", out)], &[("weight", w)])?;
    }
    let p = rt.pipeline("eg2a_rms_norm")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    let params = bytes(&[
        rows,
        dim,
        word(spec.eps),
        word(spec.pre[0].lo),
        word(spec.pre[0].hi),
        word(spec.pre[1].lo),
        word(spec.pre[1].hi),
        word(spec.post.lo),
        word(spec.post.hi),
        word(spec.scale),
        flags,
    ]);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, spec.weight.unwrap_or(x), 1);
        set_gpu_buf(bnd, spec.resid.unwrap_or(x), 2);
        set_gpu_buf(bnd, out, 3);
        bnd.bind_bytes(&params, 4);
    })
}

/// What [`ew`] adds after its clamps.
pub enum EwAdd<'a> {
    None,
    /// `+ aux[i]`: a residual of the same size.
    Residual(&'a GpuBuffer),
    /// `+ aux[i % cols]`: a per-column bias.
    Bias(&'a GpuBuffer, u32),
}

/// What [`ew`] computes per element: `clamp(post)(act(clamp(pre)(x))) + add`.
pub struct Ew<'a> {
    pub pre: Clamp,
    pub silu: bool,
    pub post: Clamp,
    pub add: EwAdd<'a>,
}

/// [`Ew`] over `n` elements, `x` into `out` (`out` may be `x`).
pub fn ew(rt: &Arc<GpuRuntime>, x: &GpuBuffer, out: &GpuBuffer, n: u32, spec: &Ew<'_>) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::ew";
    require::<f32>(rt, x, n as usize, "ew x")?;
    require::<f32>(rt, out, n as usize, "ew out")?;
    let (mut flags, mut cols, mut aux) = (if spec.silu { 2u32 } else { 0 }, 1u32, x);
    match spec.add {
        EwAdd::None => {}
        EwAdd::Residual(r) => {
            require::<f32>(rt, r, n as usize, "ew residual")?;
            (flags, aux) = (flags | 4, r);
        }
        EwAdd::Bias(b, c) => {
            if c == 0 || n % c != 0 {
                return Err(format!("{E}: bias of {c} columns for {n} elements"));
            }
            require::<f32>(rt, b, c as usize, "ew bias")?;
            (flags, cols, aux) = (flags | 8, c, b);
        }
    }
    if n == 0 {
        return Ok(());
    }
    if let EwAdd::Bias(b, _) = spec.add {
        require_disjoint_writes(E, &[("out", out)], &[("bias", b)])?;
    }
    let p = rt.pipeline("eg2a_ew")?;
    let params = bytes(&[
        n,
        cols,
        word(spec.pre.lo),
        word(spec.pre.hi),
        word(spec.post.lo),
        word(spec.post.hi),
        flags,
    ]);
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, aux, 1);
        set_gpu_buf(bnd, out, 2);
        bnd.bind_bytes(&params, 3);
    })
}

/// `glu(x)` after a clamp: `out[r, c] = clamp(x[r, c]) * sigmoid(clamp(x[r,
/// hidden + c]))`, `x [rows, 2 * hidden]`, `out [rows, hidden]`.
pub fn glu(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    hidden: u32,
    clamp: Clamp,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::glu";
    let n = elems(&[rows, hidden], E)?;
    require::<f32>(rt, x, elems(&[rows, hidden, 2], E)?, "glu x")?;
    require::<f32>(rt, out, n, "glu out")?;
    if n == 0 {
        return Ok(());
    }
    if n > u32::MAX as usize {
        return Err(format!("{E}: {n} elements exceed u32"));
    }
    require_disjoint_writes(E, &[("out", out)], &[("x", x)])?;
    let p = rt.pipeline("eg2a_glu")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, out, 1);
        set_u32(bnd, n as u32, 2);
        set_u32(bnd, hidden, 3);
        crate::dispatch::set_f32(bnd, clamp.lo, 4);
        crate::dispatch::set_f32(bnd, clamp.hi, 5);
    })
}

/// The depthwise causal convolution of `Gemma4AudioLightConv1d` over every
/// clip: `out[t, c] = sum_j w[c, j] * x[t - (taps - 1) + j, c]`, rows before a
/// clip's first reading as 0. `x`, `out [rows, hidden]`, `w [hidden, taps]`.
pub fn dwconv1d_causal(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    hidden: u32,
    taps: u32,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::dwconv1d_causal";
    if hidden == 0 || taps == 0 {
        return Err(format!("{E}: need hidden > 0 and taps > 0"));
    }
    check_table(rt, clips, E)?;
    let n = elems(&[clips.rows as u32, hidden], E)?;
    require::<f32>(rt, x, n, "dwconv x")?;
    require::<f32>(rt, out, n, "dwconv out")?;
    require::<f32>(rt, w, elems(&[hidden, taps], E)?, "dwconv w")?;
    require_disjoint_writes(E, &[("out", out)], &[("x", x), ("w", w), ("clips", &clips.buffer)])?;
    let p = rt.pipeline("eg2a_dwconv1d_causal")?;
    let groups_x = clips.max_rows() * (hidden as usize).div_ceil(32);
    dispatch_2d_tg(rt, &p, groups_x, clips.clips.len(), 32, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_gpu_buf(bnd, out, 2);
        set_gpu_buf(bnd, &clips.buffer, 3);
        set_u32(bnd, hidden, 4);
        set_u32(bnd, taps, 5);
    })
}

/// Shape and constants of one [`attention`] call.
#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    pub heads: u32,
    pub head_dim: u32,
    pub chunk: u32,
    /// `attention_context_left - 1`.
    pub past: u32,
    /// `ln(1 + e) / ln 2`.
    pub k_scale: f32,
    pub cap: f32,
    pub invalid: f32,
    /// Output clamps of the q, k and v projections, applied as the kernel
    /// loads them, and the input clamp of the output projection, applied as
    /// it stores.
    pub q: Clamp,
    pub k: Clamp,
    pub v: Clamp,
    pub out: Clamp,
}

/// The chunked local relative-position attention of `Gemma4AudioAttention`
/// after its projections, for every clip; see `kernels/embedgemma2_audio.metal`
/// (`eg2a_attn`) for the exact rule. `q`, `k`, `v`, `out [rows, heads *
/// head_dim]`; `rel [ctx / 2 + 1, heads * head_dim]` is `relative_k_proj` of
/// [`rel_position_table`]; `q_mult [head_dim]` is `head_dim^-0.5 / ln 2 *
/// softplus(per_dim_scale)`; `omask [rows]` (u32 0/1) is the tower's mask.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    rel: &GpuBuffer,
    q_mult: &GpuBuffer,
    omask: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    spec: &AttentionSpec,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::attention";
    let ctx = spec.chunk + spec.past;
    let rel_rows = ctx / 2 + 1;
    if spec.heads == 0
        || spec.head_dim == 0
        || spec.head_dim > MAX_HEAD_DIM
        || spec.chunk == 0
        || spec.past == 0
        || ctx > MAX_CTX
        || ctx + 2 < spec.chunk + rel_rows
        || !(spec.cap > 0.0 && spec.cap.is_finite())
        || !(spec.k_scale.is_finite() && spec.invalid.is_finite())
    {
        return Err(format!("{E}: unsupported shape or constants {spec:?}"));
    }
    check_table(rt, clips, E)?;
    let hidden = spec.heads * spec.head_dim;
    let n = elems(&[clips.rows as u32, hidden], E)?;
    for (b, name) in [(q, "q"), (k, "k"), (v, "v"), (out, "out")] {
        require::<f32>(rt, b, n, name)?;
    }
    require::<f32>(rt, rel, elems(&[rel_rows, hidden], E)?, "rel")?;
    require::<f32>(rt, q_mult, spec.head_dim as usize, "q_mult")?;
    require::<u32>(rt, omask, clips.rows, "omask")?;
    require_disjoint_writes(
        E,
        &[("out", out)],
        &[
            ("q", q),
            ("k", k),
            ("v", v),
            ("rel", rel),
            ("q_mult", q_mult),
            ("omask", omask),
            ("clips", &clips.buffer),
        ],
    )?;
    let p = rt.pipeline("eg2a_attn")?;
    let params = bytes(&[
        spec.heads,
        spec.head_dim,
        spec.chunk,
        spec.past,
        rel_rows,
        ctx,
        word(spec.k_scale),
        word(spec.cap),
        word(spec.invalid),
        word(spec.q.lo),
        word(spec.q.hi),
        word(spec.k.lo),
        word(spec.k.hi),
        word(spec.v.lo),
        word(spec.v.hi),
        word(spec.out.lo),
        word(spec.out.hi),
    ]);
    dispatch_2d_tg(
        rt,
        &p,
        clips.max_rows() * spec.heads as usize,
        clips.clips.len(),
        32,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, k, 1);
            set_gpu_buf(bnd, v, 2);
            set_gpu_buf(bnd, rel, 3);
            set_gpu_buf(bnd, q_mult, 4);
            set_gpu_buf(bnd, omask, 5);
            set_gpu_buf(bnd, &clips.buffer, 6);
            set_gpu_buf(bnd, out, 7);
            bnd.bind_bytes(&params, 8);
        },
    )
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Clip bounds of a `Gemma4ClippableLinear`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds {
    input: Clamp,
    output: Clamp,
}

/// A clipped linear: the weight packed `[in, out]` and its bounds.
struct Linear {
    w: Tensor,
    bounds: Bounds,
}

struct Ffn {
    pre_norm: GpuBuffer,
    post_norm: GpuBuffer,
    l1: Linear,
    l2: Linear,
}

struct Layer {
    ff1: Ffn,
    ff2: Ffn,
    norm_pre_attn: GpuBuffer,
    norm_post_attn: GpuBuffer,
    norm_out: GpuBuffer,
    q: Linear,
    k: Linear,
    v: Linear,
    post: Linear,
    /// `relative_k_proj(table)`, `[rel_rows, hidden]`.
    rel: GpuBuffer,
    /// `head_dim^-0.5 / ln 2 * softplus(per_dim_scale)`, `[head_dim]`.
    q_mult: GpuBuffer,
    lc_pre_norm: GpuBuffer,
    lc_start: Linear,
    /// `[hidden, taps]`.
    lc_conv: GpuBuffer,
    lc_conv_norm: GpuBuffer,
    lc_end: Linear,
}

/// A loaded audio tower and multimodal embedder: weights on the device in the
/// layouts the kernels read.
pub struct EmbedGemma2Audio {
    cfg: AudioConfig,
    rt: Arc<GpuRuntime>,
    /// `[C0, 9]`, `[C0]`.
    conv0_w: GpuBuffer,
    conv0_norm: GpuBuffer,
    /// `[9, C0, C1]`, `[C1]`.
    conv1_w: GpuBuffer,
    conv1_norm: GpuBuffer,
    input_proj: Tensor,
    layers: Vec<Layer>,
    output_proj: Tensor,
    output_bias: GpuBuffer,
    embed_proj: Tensor,
}

/// One clip: the processor's `input_features` and `input_features_mask`.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioInput {
    /// `[frames, mel_bins]`, row-major, finite.
    pub input_features: Vec<f32>,
    /// `[frames]`: `true` for a real frame. The frames come first in a
    /// processor's output, but any pattern is defined.
    pub mask: Vec<bool>,
}

/// One clip's intermediates, under `trace`.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioClipTrace {
    /// After the subsampling projection, `[T, hidden]`.
    pub subsample: Vec<f32>,
    /// The residual stream after each layer, `[T, hidden]` each.
    pub layers: Vec<Vec<f32>>,
    /// After `output_proj`, `[T, out_dims]`.
    pub tower_out: Vec<f32>,
}

/// What [`EmbedGemma2Audio::encode_audio`] returns, in the order of the clips.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioOutput {
    /// Per clip the soft tokens `[S, embed_dim]` of the tower rows whose mask
    /// is true, in order (`S = out_masks[i].iter().filter(..).count()`).
    pub soft_tokens: Vec<Vec<f32>>,
    /// Per clip the tower's mask `[T]`, `T = ceil(ceil(frames / 2) / 2)`:
    /// `mask[::4]`.
    pub out_masks: Vec<Vec<bool>>,
    /// Per clip, with `trace`; empty otherwise.
    pub trace: Vec<AudioClipTrace>,
    /// How many forwards the batch was split into (see [`MAX_FRAMES_PER_FORWARD`]).
    pub forwards: u32,
}

/// Every activation of an `encode_audio`'s forwards, allocated once for its
/// largest forward and reused by each. A forward over fewer rows works on a
/// prefix of every buffer.
///
/// Nothing here is zeroed on the host ([`GpuRuntime::alloc_tensor_unzeroed`]):
/// every element a forward reads, an earlier kernel of the same forward wrote.
/// The convolutions write every position of their clips' rows, the GEMMs and
/// the elementwise passes every element of their rows, and the attention
/// reads the q, k, v rows its GEMMs wrote and a mask the host wrote.
/// `tests/embedgemma2_audio_tiny.rs` runs the forward with every unzeroed
/// allocation poisoned to NaN.
struct Acts {
    /// First conv's output, `[mid rows, mels1 * C0]`.
    mid: Tensor,
    /// Second conv's output, `[rows, proj_in]`.
    sub: Tensor,
    /// The residual stream.
    x: Tensor,
    /// The norms' outputs, then the attention's and the GLU's.
    n: Tensor,
    /// A clamped copy of a q/k/v projection's input.
    c: Tensor,
    y: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// The feed-forward's `[rows, 4 * hidden]`; the GLU's input is its
    /// `[rows, 2 * hidden]` prefix.
    big: Tensor,
    tower: Tensor,
    emb: Tensor,
}

/// What one forward alone owns, because the host writes it before the forward
/// runs and reads it after: a later forward must not rewrite it while the GPU
/// may still be reading it, so each gets its own set, and `encode_audio` keeps
/// it alive until its one wait.
struct ForwardIo {
    clips: ClipTable,
    feats: GpuBuffer,
    fmask: GpuBuffer,
    omask: GpuBuffer,
    /// `[rows, embed_dim]`.
    soft: Tensor,
    /// With `trace`: the subsample output, each layer's residual, then the
    /// tower output, each `[rows, width]`.
    trace: Vec<Tensor>,
}

impl EmbedGemma2Audio {
    /// Load the tower from `st`, whose tensors are named
    /// `{prefix}audio_tower.*` and `{prefix}embed_audio.*`. Every tensor's
    /// shape is checked against `cfg`; a missing one is an error, and so is
    /// any tensor under either prefix the forward does not read.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, prefix: &str, cfg: AudioConfig) -> Result<Self, String> {
        cfg.validate()?;
        let tower_prefix = format!("{prefix}{PREFIX_TOWER}");
        let embed_prefix = format!("{prefix}{PREFIX_EMBED}");
        let ld = Loader::new(rt, st, &tower_prefix);
        let le = Loader::new(rt, st, &embed_prefix);
        let (h, hd) = (cfg.hidden as usize, cfg.head_dim() as usize);
        let (c0, c1) = (cfg.sub_channels[0] as usize, cfg.sub_channels[1] as usize);
        let f32p = Precision::F32;

        let sub = |s: &str| format!("subsample_conv_projection.{s}");
        let conv0_w = ld.buf(&ld.f32(&sub("layer0.conv.weight"), &[c0, 1, 3, 3])?)?;
        let conv0_norm = ld.norm(&sub("layer0.norm.weight"), c0)?;
        // [co, ci, kh, kw] -> [kh * 3 + kw, ci, co]: lanes read adjacent co.
        let w1 = ld.f32(&sub("layer1.conv.weight"), &[c1, c0, 3, 3])?;
        let mut w1p = vec![0.0f32; 9 * c0 * c1];
        for co in 0..c1 {
            for ci in 0..c0 {
                for tap in 0..9 {
                    w1p[(tap * c0 + ci) * c1 + co] = w1[(co * c0 + ci) * 9 + tap];
                }
            }
        }
        let conv1_w = ld.buf(&w1p)?;
        let conv1_norm = ld.norm(&sub("layer1.norm.weight"), c1)?;
        let input_proj = ld.linear(&[(&sub("input_proj_linear.weight"), h)], cfg.proj_in() as usize, f32p)?;

        let bound = |name: &str| -> Result<Bounds, String> {
            if !cfg.clipped_linears {
                return Ok(Bounds {
                    input: Clamp::NONE,
                    output: Clamp::NONE,
                });
            }
            let get = |s: &str| -> Result<f32, String> {
                let v = ld.f32(&format!("{name}.{s}"), &[])?;
                match v[..] {
                    [x] if !x.is_nan() => Ok(x),
                    ref other => Err(format!("{tower_prefix}{name}.{s}: expected one number, got {other:?}")),
                }
            };
            Ok(Bounds {
                input: Clamp::new(get("input_min")?, get("input_max")?)?,
                output: Clamp::new(get("output_min")?, get("output_max")?)?,
            })
        };
        let clipped = |name: &str, out: usize, inp: usize| -> Result<Linear, String> {
            Ok(Linear {
                w: ld.linear(&[(&format!("{name}.linear.weight"), out)], inp, f32p)?,
                bounds: bound(name)?,
            })
        };

        let ctx = cfg.ctx() as usize;
        let rel_rows = cfg.rel_rows() as usize;
        let table = rel_position_table(h, ctx);
        let q_scale = (hd as f64).powf(-0.5) / std::f64::consts::LN_2;
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let ffn = |name: &str| -> Result<Ffn, String> {
                Ok(Ffn {
                    pre_norm: ld.norm(&p(&format!("{name}.pre_layer_norm.weight")), h)?,
                    post_norm: ld.norm(&p(&format!("{name}.post_layer_norm.weight")), h)?,
                    l1: clipped(&p(&format!("{name}.ffw_layer_1")), 4 * h, h)?,
                    l2: clipped(&p(&format!("{name}.ffw_layer_2")), h, 4 * h)?,
                })
            };
            let rel_w = ld.f32(&p("self_attn.relative_k_proj.weight"), &[h, h])?;
            let per_dim = ld.f32(&p("self_attn.per_dim_scale"), &[hd])?;
            let q_mult: Vec<f32> = per_dim
                .iter()
                .map(|&s| (q_scale * softplus(f64::from(s))) as f32)
                .collect();
            layers.push(Layer {
                ff1: ffn("feed_forward1")?,
                ff2: ffn("feed_forward2")?,
                norm_pre_attn: ld.norm(&p("norm_pre_attn.weight"), h)?,
                norm_post_attn: ld.norm(&p("norm_post_attn.weight"), h)?,
                norm_out: ld.norm(&p("norm_out.weight"), h)?,
                q: clipped(&p("self_attn.q_proj"), h, h)?,
                k: clipped(&p("self_attn.k_proj"), h, h)?,
                v: clipped(&p("self_attn.v_proj"), h, h)?,
                post: clipped(&p("self_attn.post"), h, h)?,
                rel: ld.buf(&relative_keys(&table, &rel_w, rel_rows, h))?,
                q_mult: ld.buf(&q_mult)?,
                lc_pre_norm: ld.norm(&p("lconv1d.pre_layer_norm.weight"), h)?,
                lc_start: clipped(&p("lconv1d.linear_start"), 2 * h, h)?,
                lc_conv: ld.buf(&ld.f32(&p("lconv1d.depthwise_conv1d.weight"), &[h, 1, cfg.conv_kernel as usize])?)?,
                lc_conv_norm: ld.norm(&p("lconv1d.conv_norm.weight"), h)?,
                lc_end: clipped(&p("lconv1d.linear_end"), h, h)?,
            });
        }
        let out_dims = cfg.out_dims as usize;
        let output_proj = ld.linear(&[("output_proj.weight", out_dims)], h, f32p)?;
        let output_bias = ld.norm("output_proj.bias", out_dims)?;
        let embed_proj = le.linear(
            &[("embedding_projection.weight", cfg.embed_dim as usize)],
            out_dims,
            f32p,
        )?;
        ld.refuse_unread()?;
        le.refuse_unread()?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            conv0_w,
            conv0_norm,
            conv1_w,
            conv1_norm,
            input_proj,
            layers,
            output_proj,
            output_bias,
            embed_proj,
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.cfg
    }

    /// Embed a batch of clips into soft tokens. Each clip's output is its own
    /// tower rows through the embedder, kept where the tower's mask is true;
    /// it does not depend on what else is in the batch beyond GEMM tiling.
    ///
    /// The batch runs as one or more forwards (clips in order, cut at
    /// [`MAX_FRAMES_PER_FORWARD`] frames), all queued before the host waits
    /// once. `trace` also returns each clip's subsample output, per-layer
    /// residuals and tower output. On a runtime without `async_scope`, as
    /// this one, the wait is one `synchronize()` after the last forward; the
    /// runtime's `async_encode` setting decides whether the forwards overlap
    /// the host encoding the next.
    pub fn encode_audio(&self, clips: &[AudioInput], trace: bool) -> Result<AudioOutput, String> {
        const E: &str = "EmbedGemma2Audio::encode_audio";
        let cfg = &self.cfg;
        if clips.is_empty() {
            return Err(format!("{E}: empty batch"));
        }
        if clips.len() > MAX_CLIPS {
            return Err(format!(
                "{E}: {} clips exceeds the batch limit {MAX_CLIPS}",
                clips.len()
            ));
        }
        if self.rt.f32_gemm_runs_relaxed() {
            return Err(format!(
                "{E}: the forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        let mels = cfg.mel_bins() as usize;
        for (i, c) in clips.iter().enumerate() {
            let frames = c.mask.len();
            if frames == 0 {
                return Err(format!("{E}: clip {i} has no frames"));
            }
            if frames > MAX_FRAMES_PER_FORWARD {
                return Err(format!(
                    "{E}: clip {i} has {frames} frames; the limit is {MAX_FRAMES_PER_FORWARD} (split it)"
                ));
            }
            if c.input_features.len() != frames * mels {
                return Err(format!(
                    "{E}: clip {i} has {} feature values for {frames} frames of {mels} mel bins ({} expected)",
                    c.input_features.len(),
                    frames * mels
                ));
            }
            if let Some(at) = c.input_features.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "{E}: clip {i}: input_features[{at}] = {} is not finite",
                    c.input_features[at]
                ));
            }
        }

        // Clips in order, cut where a forward would pass the frame budget.
        let mut plan: Vec<Vec<usize>> = Vec::new();
        let mut used = 0usize;
        for (i, c) in clips.iter().enumerate() {
            let f = c.mask.len();
            match plan.last_mut() {
                Some(g) if used + f <= MAX_FRAMES_PER_FORWARD => {
                    g.push(i);
                    used += f;
                }
                _ => {
                    plan.push(vec![i]);
                    used = f;
                }
            }
        }
        let dims = |g: &Vec<usize>| -> (usize, usize, usize) {
            let f: usize = g.iter().map(|&i| clips[i].mask.len()).sum();
            let m: usize = g.iter().map(|&i| conv_out(clips[i].mask.len())).sum();
            let r: usize = g.iter().map(|&i| sub_rows(clips[i].mask.len())).sum();
            (f, m, r)
        };
        let (max_m, max_r) = plan.iter().fold((0, 0), |(m, r), g| {
            let (_, gm, gr) = dims(g);
            (m.max(gm), r.max(gr))
        });
        let acts = self.acts(max_m, max_r)?;

        let mut pending: Vec<(Vec<usize>, ForwardIo)> = Vec::with_capacity(plan.len());
        let queued: Result<(), String> = (|| {
            for g in &plan {
                let io = self.forward_io(clips, g, trace)?;
                pending.push((g.clone(), io));
                let (_, io) = pending.last().expect("just pushed");
                self.forward(&acts, io)?;
            }
            Ok(())
        })();
        // Wait even when queuing failed: the buffers in `pending` and `acts`
        // must outlive every command already queued on them (Metal 4 command
        // buffers do not retain what they bind).
        let waited = self.rt.synchronize();
        queued?;
        waited?;

        let embed = cfg.embed_dim as usize;
        let mut out = AudioOutput {
            soft_tokens: vec![Vec::new(); clips.len()],
            out_masks: vec![Vec::new(); clips.len()],
            trace: if trace {
                vec![
                    AudioClipTrace {
                        subsample: Vec::new(),
                        layers: Vec::new(),
                        tower_out: Vec::new(),
                    };
                    clips.len()
                ]
            } else {
                Vec::new()
            },
            forwards: pending.len() as u32,
        };
        for (g, io) in &pending {
            // One host mapping at a time (the runtime's access is exclusive).
            let read =
                |t: &Tensor| -> Result<Vec<f32>, String> { Ok(t.buffer.try_contents_f32()?[..t.numel()].to_vec()) };
            let soft = read(&io.soft)?;
            let traced: Vec<Vec<f32>> = io.trace.iter().map(read).collect::<Result<_, _>>()?;
            for (&i, cr) in g.iter().zip(&io.clips.clips) {
                let omask = out_mask(&clips[i].mask);
                debug_assert_eq!(omask.len(), cr.rows);
                let mut tokens = Vec::new();
                for (t, &keep) in omask.iter().enumerate() {
                    if keep {
                        let r = cr.row0 + t;
                        tokens.extend_from_slice(&soft[r * embed..(r + 1) * embed]);
                    }
                }
                out.soft_tokens[i] = tokens;
                out.out_masks[i] = omask;
                if trace {
                    let h = cfg.hidden as usize;
                    let rows = |m: &[f32], w: usize| m[cr.row0 * w..(cr.row0 + cr.rows) * w].to_vec();
                    let layers = cfg.layers as usize;
                    out.trace[i] = AudioClipTrace {
                        subsample: rows(&traced[0], h),
                        layers: (0..layers).map(|l| rows(&traced[1 + l], h)).collect(),
                        tower_out: rows(&traced[1 + layers], cfg.out_dims as usize),
                    };
                }
            }
        }
        Ok(out)
    }

    /// [`Acts`] for forwards of up to `mid` first-conv rows and `rows` tower
    /// rows.
    fn acts(&self, mid: usize, rows: usize) -> Result<Acts, String> {
        let cfg = &self.cfg;
        let rt = &self.rt;
        let (h, mid, rows) = (cfg.hidden as usize, mid.max(1), rows.max(1));
        let mels1 = conv_out(cfg.mel_bins() as usize);
        // Every elementwise pass counts its elements in a u32.
        let widest = rows
            .checked_mul(4 * h.max(cfg.out_dims as usize))
            .filter(|&n| n <= u32::MAX as usize);
        if widest.is_none() {
            return Err(format!(
                "EmbedGemma2Audio::encode_audio: {rows} rows of width {} overflow a 32-bit element count",
                4 * h.max(cfg.out_dims as usize)
            ));
        }
        let f32s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::F32);
        Ok(Acts {
            mid: f32s(&[mid, mels1 * cfg.sub_channels[0] as usize])?,
            sub: f32s(&[rows, cfg.proj_in() as usize])?,
            x: f32s(&[rows, h])?,
            n: f32s(&[rows, h])?,
            c: f32s(&[rows, h])?,
            y: f32s(&[rows, h])?,
            q: f32s(&[rows, h])?,
            k: f32s(&[rows, h])?,
            v: f32s(&[rows, h])?,
            big: f32s(&[rows, 4 * h])?,
            tower: f32s(&[rows, cfg.out_dims as usize])?,
            emb: f32s(&[rows, cfg.out_dims as usize])?,
        })
    }

    /// This forward's own host-written inputs and host-read outputs for the
    /// clips `g`, written without waiting for the GPU.
    fn forward_io(&self, clips: &[AudioInput], g: &[usize], trace: bool) -> Result<ForwardIo, String> {
        let cfg = &self.cfg;
        let rt = &self.rt;
        let frames: Vec<usize> = g.iter().map(|&i| clips[i].mask.len()).collect();
        let table = ClipTable::new(rt, &frames)?;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let feats: Vec<u32> = g.iter().flat_map(|&i| bits(&clips[i].input_features)).collect();
        let fmask: Vec<u32> = g
            .iter()
            .flat_map(|&i| clips[i].mask.iter().map(|&m| u32::from(m)).collect::<Vec<_>>())
            .collect();
        let omask: Vec<u32> = g
            .iter()
            .flat_map(|&i| out_mask(&clips[i].mask).into_iter().map(u32::from).collect::<Vec<_>>())
            .collect();
        let rows = table.rows;
        let f32s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::F32);
        let mut traced = Vec::new();
        if trace {
            for _ in 0..=cfg.layers {
                traced.push(f32s(&[rows, cfg.hidden as usize])?);
            }
            traced.push(f32s(&[rows, cfg.out_dims as usize])?);
        }
        Ok(ForwardIo {
            feats: rt.alloc_buffer_from_u32(&feats)?,
            fmask: rt.alloc_buffer_from_u32(&fmask)?,
            omask: rt.alloc_buffer_from_u32(&omask)?,
            soft: f32s(&[rows, cfg.embed_dim as usize])?,
            trace: traced,
            clips: table,
        })
    }

    /// Queue one forward over `io`'s clips in `a`, without waiting for it. The
    /// soft tokens are in `io.soft` once the caller has synchronized.
    fn forward(&self, a: &Acts, io: &ForwardIo) -> Result<(), String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let backend: GemmBackend = select_backend(rt);
        let (rows, mid) = (io.clips.rows, io.clips.mid_rows);
        let (h, rows_u) = (cfg.hidden, rows as u32);
        let (eps, hd) = (cfg.eps, cfg.head_dim());
        let at = |t: &Tensor, cols: u32| t.try_view(&[rows, cols as usize], 0);
        let (x, n, c, y) = (at(&a.x, h)?, at(&a.n, h)?, at(&a.c, h)?, at(&a.y, h)?);
        let (q, k, v) = (at(&a.q, h)?, at(&a.k, h)?, at(&a.v, h)?);
        let big = at(&a.big, 4 * h)?;
        let big2 = at(&a.big, 2 * h)?;
        let gc = Clamp::symmetric(cfg.gradient_clip);
        let none = Clamp::NONE;
        let mels = cfg.mel_bins();
        let (c0, c1) = (cfg.sub_channels[0], cfg.sub_channels[1]);
        let mels1 = conv_out(mels as usize) as u32;

        // Subsampling.
        let mid_t = a.mid.try_view(&[mid, (mels1 * c0) as usize], 0)?;
        sub_conv0_ln_relu(
            rt,
            &io.feats,
            &io.fmask,
            &self.conv0_w,
            &self.conv0_norm,
            &mid_t.buffer,
            &io.clips,
            mels,
            c0,
            eps,
        )?;
        let sub_t = at(&a.sub, cfg.proj_in())?;
        sub_conv1_ln_relu(
            rt,
            &mid_t.buffer,
            &io.fmask,
            &self.conv1_w,
            &self.conv1_norm,
            &sub_t.buffer,
            &io.clips,
            mels1,
            c0,
            c1,
            eps,
        )?;
        gemm(&sub_t, &self.input_proj, &x, backend)?;
        if let Some(t) = io.trace.first() {
            gpu_copy(&x, t)?;
        }

        let ffn = |f: &Ffn| -> Result<(), String> {
            let mut pre = RmsNorm::new(eps, Some(&f.pre_norm));
            pre.pre = [gc, none];
            pre.post = f.l1.bounds.input;
            rms_norm(rt, &x.buffer, &n.buffer, rows_u, h, &pre)?;
            gemm(&n, &f.l1.w, &big, backend)?;
            ew(
                rt,
                &big.buffer,
                &big.buffer,
                rows_u * 4 * h,
                &Ew {
                    pre: f.l1.bounds.output,
                    silu: true,
                    post: f.l2.bounds.input,
                    add: EwAdd::None,
                },
            )?;
            gemm(&big, &f.l2.w, &y, backend)?;
            let mut post = RmsNorm::new(eps, Some(&f.post_norm));
            post.pre = [f.l2.bounds.output, gc];
            post.scale = cfg.residual_weight;
            post.resid = Some(&x.buffer);
            rms_norm(rt, &y.buffer, &x.buffer, rows_u, h, &post)
        };

        for (li, l) in self.layers.iter().enumerate() {
            ffn(&l.ff1)?;

            // Attention.
            let mut pre = RmsNorm::new(eps, Some(&l.norm_pre_attn));
            pre.pre = [gc, none];
            rms_norm(rt, &x.buffer, &n.buffer, rows_u, h, &pre)?;
            for (lin, dst) in [(&l.q, &q), (&l.k, &k), (&l.v, &v)] {
                ew(
                    rt,
                    &n.buffer,
                    &c.buffer,
                    rows_u * h,
                    &Ew {
                        pre: lin.bounds.input,
                        silu: false,
                        post: none,
                        add: EwAdd::None,
                    },
                )?;
                gemm(&c, &lin.w, dst, backend)?;
            }
            attention(
                rt,
                &q.buffer,
                &k.buffer,
                &v.buffer,
                &l.rel,
                &l.q_mult,
                &io.omask,
                &n.buffer,
                &io.clips,
                &AttentionSpec {
                    heads: cfg.heads,
                    head_dim: hd,
                    chunk: cfg.chunk,
                    past: cfg.past(),
                    k_scale: ((1.0f64 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32,
                    cap: cfg.logit_cap,
                    invalid: cfg.invalid_logit,
                    q: l.q.bounds.output,
                    k: l.k.bounds.output,
                    v: l.v.bounds.output,
                    out: l.post.bounds.input,
                },
            )?;
            gemm(&n, &l.post.w, &y, backend)?;
            let mut post = RmsNorm::new(eps, Some(&l.norm_post_attn));
            post.pre = [l.post.bounds.output, gc];
            post.resid = Some(&x.buffer);
            rms_norm(rt, &y.buffer, &x.buffer, rows_u, h, &post)?;

            // Light conv.
            let mut pre = RmsNorm::new(eps, Some(&l.lc_pre_norm));
            pre.post = l.lc_start.bounds.input;
            rms_norm(rt, &x.buffer, &n.buffer, rows_u, h, &pre)?;
            gemm(&n, &l.lc_start.w, &big2, backend)?;
            glu(rt, &big2.buffer, &n.buffer, rows_u, h, l.lc_start.bounds.output)?;
            dwconv1d_causal(rt, &n.buffer, &l.lc_conv, &y.buffer, &io.clips, h, cfg.conv_kernel)?;
            let mut cn = RmsNorm::new(eps, Some(&l.lc_conv_norm));
            cn.pre = [gc, none];
            cn.silu = true;
            cn.post = l.lc_end.bounds.input;
            rms_norm(rt, &y.buffer, &n.buffer, rows_u, h, &cn)?;
            gemm(&n, &l.lc_end.w, &y, backend)?;
            ew(
                rt,
                &y.buffer,
                &x.buffer,
                rows_u * h,
                &Ew {
                    pre: l.lc_end.bounds.output,
                    silu: false,
                    post: none,
                    add: EwAdd::Residual(&x.buffer),
                },
            )?;

            ffn(&l.ff2)?;
            let mut out = RmsNorm::new(eps, Some(&l.norm_out));
            out.pre = [gc, none];
            rms_norm(rt, &x.buffer, &x.buffer, rows_u, h, &out)?;
            if let Some(t) = io.trace.get(1 + li) {
                gpu_copy(&x, t)?;
            }
        }

        // Output projection (with bias) and the multimodal embedder.
        let tower = at(&a.tower, cfg.out_dims)?;
        gemm(&x, &self.output_proj, &tower, backend)?;
        ew(
            rt,
            &tower.buffer,
            &tower.buffer,
            rows_u * cfg.out_dims,
            &Ew {
                pre: none,
                silu: false,
                post: none,
                add: EwAdd::Bias(&self.output_bias, cfg.out_dims),
            },
        )?;
        if let Some(t) = io.trace.last() {
            gpu_copy(&tower, t)?;
        }
        let emb = at(&a.emb, cfg.out_dims)?;
        rms_norm(
            rt,
            &tower.buffer,
            &emb.buffer,
            rows_u,
            cfg.out_dims,
            &RmsNorm::new(eps, None),
        )?;
        gemm(&emb, &self.embed_proj, &io.soft, backend)
    }
}

/// The tower's mask for a clip's frame mask: `mask[::2][::2]`, one entry per
/// tower row.
pub fn out_mask(mask: &[bool]) -> Vec<bool> {
    mask.iter()
        .step_by(2)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .step_by(2)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audio_config of google/embeddinggemma-2's config.json.
    const REAL: &str = r#"{"text_config":{"hidden_size":512},"audio_config":{
"attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,
"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,"conv_kernel_size":5,
"gradient_clipping":10000000000.0,"hidden_act":"silu","hidden_size":1024,"model_type":"gemma4_audio",
"num_attention_heads":8,"num_hidden_layers":12,"output_proj_dims":1536,"residual_weight":0.5,
"rms_norm_eps":1e-06,"subsampling_conv_channels":[128,32],"use_clipped_linears":true}}"#;

    fn real() -> AudioConfig {
        AudioConfig::from_config_json(REAL).unwrap()
    }

    #[test]
    fn config_reads_the_checkpoint() {
        let c = real();
        assert_eq!((c.hidden, c.layers, c.heads, c.head_dim()), (1024, 12, 8, 128));
        assert_eq!((c.chunk, c.past(), c.ctx(), c.rel_rows()), (12, 12, 24, 13));
        assert_eq!(
            (c.mel_bins(), c.proj_in(), c.out_dims, c.embed_dim),
            (128, 1024, 1536, 512)
        );
        assert_eq!(c.gradient_clip, 1e10);
        assert!(c.clipped_linears);
    }

    #[test]
    fn config_refuses_what_the_forward_does_not_implement() {
        let swap = |from: &str, to: &str| REAL.replace(from, to);
        for (from, to, why) in [
            (
                "\"attention_context_right\":0",
                "\"attention_context_right\":2",
                "future context",
            ),
            ("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"", "activation"),
            (
                "\"num_attention_heads\":8",
                "\"num_attention_heads\":7",
                "heads that do not divide",
            ),
            (
                "\"subsampling_conv_channels\":[128,32]",
                "\"subsampling_conv_channels\":[128]",
                "one conv",
            ),
            (
                "\"subsampling_conv_channels\":[128,32]",
                "\"subsampling_conv_channels\":[130,32]",
                "channels % 4",
            ),
            (
                "\"subsampling_conv_channels\":[128,32]",
                "\"subsampling_conv_channels\":[128,200]",
                "wide second conv",
            ),
            (
                "\"attention_chunk_size\":12",
                "\"attention_chunk_size\":0",
                "zero chunk",
            ),
            (
                "\"attention_chunk_size\":12",
                "\"attention_chunk_size\":60",
                "window past 64",
            ),
            (
                "\"attention_context_left\":13",
                "\"attention_context_left\":1",
                "no past",
            ),
            ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":0.0", "zero eps"),
            ("\"use_clipped_linears\":true", "\"use_clipped_linears\":1", "non-bool"),
            (
                "\"model_type\":\"gemma4_audio\"",
                "\"model_type\":\"gemma4_vision\"",
                "model type",
            ),
        ] {
            let e = AudioConfig::from_config_json(&swap(from, to)).expect_err(why);
            assert!(e.contains("config.json"), "{why}: {e}");
        }
        assert!(AudioConfig::from_config_json(&REAL.replace("\"text_config\":{\"hidden_size\":512},", "")).is_err());
        // The same window with a chunk the shifted rows of the table would reach.
        let e = AudioConfig::from_config_json(
            &REAL
                .replace("\"attention_chunk_size\":12", "\"attention_chunk_size\":30")
                .replace("\"attention_context_left\":13", "\"attention_context_left\":3"),
        )
        .expect_err("rel shift");
        assert!(e.contains("shifted rows"), "{e}");
    }

    /// Every number the module quotes of the table, from first principles.
    #[test]
    fn the_position_table_is_transformers() {
        let (hidden, ctx) = (8usize, 6usize);
        let t = rel_position_table(hidden, ctx);
        assert_eq!(t.len(), (ctx / 2 + 1) * hidden);
        // Row p is position ctx/2 - p; the last row is position 0: sin 0, cos 1.
        let last = &t[3 * hidden..];
        assert_eq!(&last[..4], &[0.0; 4]);
        assert_eq!(&last[4..], &[1.0; 4]);
        // inv_timescales = exp(-d * ln(10000) / 3) for d = 0..4.
        for (p, pos) in [(0usize, 3.0f64), (1, 2.0), (2, 1.0)] {
            for d in 0..4usize {
                let f = 10000f64.powf(-(d as f64) / 3.0);
                assert!((t[p * hidden + d] - (pos * f).sin()).abs() < 1e-12);
                assert!((t[p * hidden + 4 + d] - (pos * f).cos()).abs() < 1e-12);
            }
        }
        assert_eq!(
            rel_position_table(2, 2).len(),
            2 * 2,
            "one timescale: the divisor is max(n - 1, 1)"
        );
    }

    /// `_rel_shift` literally: pad each row of `[chunk, rel_rows]` to `ctx + 1`,
    /// flatten, drop the tail, view as `[chunk, ctx]`; the entry says which
    /// source it came from (tagged `row * 1000 + col + 1`, 0 for padding).
    fn rel_shift_literal(chunk: usize, rel_rows: usize, ctx: usize) -> Vec<Vec<usize>> {
        let mut flat = Vec::new();
        for i in 0..chunk {
            for c in 0..ctx + 1 {
                flat.push(if c < rel_rows { i * 1000 + c + 1 } else { 0 });
            }
        }
        flat.truncate(chunk * ctx);
        flat.chunks(ctx).map(<[usize]>::to_vec).collect()
    }

    #[test]
    fn rel_shift_source_is_transformers_rel_shift() {
        for (chunk, past) in [(12usize, 12usize), (4, 4), (6, 9), (1, 1), (3, 12), (8, 5)] {
            let ctx = chunk + past;
            let rel_rows = ctx / 2 + 1;
            let lit = rel_shift_literal(chunk, rel_rows, ctx);
            // Only configurations the config accepts read no shifted rows.
            let accepted = ctx + 2 >= chunk + rel_rows;
            for (i, row) in lit.iter().enumerate() {
                for (j, &want) in row.iter().enumerate() {
                    match rel_shift_source(i, j, rel_rows) {
                        Some(col) => assert_eq!(want, i * 1000 + col + 1, "chunk {chunk} past {past} ({i},{j})"),
                        None if accepted => assert_eq!(want, 0, "chunk {chunk} past {past} ({i},{j})"),
                        None => {}
                    }
                }
            }
        }
    }

    /// `_convert_4d_mask_to_blocked_5d` literally: the 4-D mask is the padding
    /// mask on keys and the window function, padded to whole blocks, cut into
    /// blocks of `chunk` queries, each query's keys padded `past` on the left,
    /// then each block gathers `ctx` keys starting at `block * chunk`.
    fn blocked_mask_literal(omask: &[bool], chunk: usize, past: usize) -> Vec<Vec<Vec<bool>>> {
        let t = omask.len();
        let blocks = t.div_ceil(chunk);
        let padded = blocks * chunk;
        let ctx = chunk + past;
        let m4 = |q: usize, kv: usize| -> bool {
            if q >= t || kv >= t {
                return false;
            }
            let dist = q as i64 - kv as i64;
            let window = dist >= 0 && dist < past as i64;
            window && omask[kv]
        };
        (0..blocks)
            .map(|b| {
                (0..chunk)
                    .map(|i| {
                        (0..ctx)
                            .map(|j| {
                                // F.pad(past, 0) on the key axis, then gather block_start + offset.
                                let col = b * chunk + j;
                                col >= past && col - past < padded && m4(b * chunk + i, col - past)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn key_visible_is_the_blocked_mask() {
        let patterns: Vec<Vec<bool>> = vec![
            vec![true; 30],
            (0..30).map(|i| i < 19).collect(),
            (0..30).map(|i| !(7..13).contains(&i)).collect(),
            vec![true; 1],
            vec![false; 5],
            (0..25).map(|i| i % 3 != 0).collect(),
        ];
        for (chunk, past) in [(12usize, 12usize), (4, 4), (5, 7)] {
            for omask in &patterns {
                let lit = blocked_mask_literal(omask, chunk, past);
                let t = omask.len();
                for (b, block) in lit.iter().enumerate() {
                    for (i, row) in block.iter().enumerate() {
                        let q = b * chunk + i;
                        if q >= t {
                            continue;
                        }
                        for (j, &want) in row.iter().enumerate() {
                            let kp = (b * chunk + j) as i64 - past as i64;
                            assert_eq!(
                                key_visible(q, kp, past, t, omask),
                                want,
                                "chunk {chunk} past {past} len {t} block {b} ({i},{j})"
                            );
                        }
                    }
                }
            }
        }
        // 12 keys per query in the interior of a long clip, not 13.
        let all = vec![true; 60];
        let seen = (0..60).filter(|&kp| key_visible(40, kp, 12, 60, &all)).count();
        assert_eq!(seen, 12);
        assert!(key_visible(40, 29, 12, 60, &all) && !key_visible(40, 28, 12, 60, &all));
        assert!(!key_visible(40, 41, 12, 60, &all), "no future");
    }

    #[test]
    fn row_spaces() {
        assert_eq!(
            [1usize, 2, 3, 4, 5, 99, 330].map(sub_rows),
            [1, 1, 1, 1, 2, 25, 83],
            "ceil(ceil(F/2)/2): the goldens' 99 frames give 25 tokens and 330 give 83"
        );
        let mask: Vec<bool> = (0..330).map(|i| i < 329).collect();
        let om = out_mask(&mask);
        assert_eq!(om.len(), 83);
        assert!(om.iter().all(|&m| m), "frame 329 is dropped by [::4]; 328 is kept");
        let om = out_mask(&[true, true, true, true, true, false, false, false, false]);
        assert_eq!(om, vec![true, true, false]);
    }

    #[test]
    fn clamps_hold_infinite_bounds_finite() {
        let c = Clamp::new(f32::NEG_INFINITY, f32::INFINITY).unwrap();
        assert_eq!(c, Clamp::NONE);
        assert!(Clamp::new(f32::NAN, 1.0).is_err());
        assert_eq!(Clamp::symmetric(f32::INFINITY), Clamp::NONE);
        assert_eq!(Clamp::symmetric(2.0), Clamp { lo: -2.0, hi: 2.0 });
        assert_eq!(Clamp::symmetric(1e10), Clamp { lo: -1e10, hi: 1e10 });
    }

    #[test]
    fn softplus_follows_torch_threshold() {
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(-30.0) - (-30.0f64).exp()).abs() < 1e-25);
    }
}
