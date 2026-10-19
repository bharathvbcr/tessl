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
//! `tools/embedgemma2_ref/make_mm_reference.py` runs sdpa by default and so
//! writes the audio references with the bool mask. `docs/embedgemma2.md` has the measured
//! difference: a reference taken under eager disagrees from layer 0.
//!
//! # Precision
//!
//! [`EmbedGemma2Audio::load`] is [`GemmOperands::ExactF32`]: weights widened from
//! the checkpoint's bf16 to f32 exactly, every activation f32 and every GEMM
//! exact f32, transformers' fp32 forward up to operation order. It needs the
//! runtime's relaxed-f32 GEMM switched off (the default). Its output is
//! unchanged by the bf16 path below (the kernels' f32 arithmetic is the same
//! code). A clip bound of infinity (the module's untrained default) is held as
//! `+-f32::MAX`, which clamps nothing.
//!
//! [`EmbedGemma2Audio::load_with`] with [`GemmOperands::Bf16`] keeps every
//! projection as the checkpoint stores it, bf16, on the device (half the
//! weight bytes, [`EmbedGemma2Audio::weight_bytes`]) and runs every GEMM with
//! bf16 operands and f32 accumulation. The operand of each GEMM is stored bf16
//! by the pass that produces it ([`ActStore`]: the row passes, the feed-forward
//! activation, the light conv and the attention), with no cast per GEMM; the
//! two ends of the forward, which no audio pass writes, take one cast each (the
//! subsampling's output into the input projection, the last layer's residual
//! stream into the output projection). The residual stream, the norms' and
//! clamps' arithmetic, the convolutions, the soft tokens and every GEMM's
//! output stay f32. The attention is not the text and vision towers' bf16 tile
//! kernel (`encoder_attn_bf16`): it is chunked with relative-position logits,
//! a softcap and a window mask, which that kernel does not have, so it stays
//! the scalar f32 kernel [`attention`] reading the projection's f32 output, and
//! only stores its result bf16. No GEMM of this forward is an f32 one, so it
//! accepts a runtime with relaxed f32 GEMMs on.
//!
//! # Rows
//!
//! A batch is its clips laid end to end in each row space (frames, the first
//! conv's rows, tower rows) and run as one forward, up to
//! [`MAX_FRAMES_PER_FORWARD`] frames and [`MAX_CLIPS`] clips (more clips are
//! more forwards, all queued before the one wait; a call's total is bounded by
//! [`MAX_CALL_FRAMES`] and [`MAX_CALL_CLIPS`], the memory its forwards' inputs
//! and outputs hold until that wait, refused by name before any allocation). A
//! clip's rows never read another's: the convolutions
//! and the attention work in clip-local indices and the GEMMs and norms are
//! row-wise, so a clip's output does not depend on what it is batched with. The
//! one place a padded (masked) frame influences a kept one is the subsampling's
//! zeroing before each conv, which is defined by the mask.
//!
//! # Kernels
//!
//! New, in `kernels/embedgemma2_audio.metal`: [`sub_conv0_ln_relu`] and
//! [`sub_conv1_ln_relu`] (the strided convs with the frame mask, channel
//! LayerNorm and ReLU; the second on 8x8 simdgroup tiles), [`rms_norm`] and
//! [`rms_norm_then`] (the row passes, one or two per launch, with the clipped
//! linears' and the gradient clamps folded in: a layer's residual add and the
//! norm after it are one launch), [`ew`] (the clamp, SiLU and clamp between a
//! feed-forward's linears), [`lconv`] (the GLU, the depthwise conv and the
//! conv norm of the light conv in one pass) and [`attention`]. The matmuls are
//! tessl's [`gemm`] (exact f32) or `gemm_bf16`; the clamps are not fused
//! into them (`gemm_epilogue` has an exact-f32 path now, but not one for
//! these clamps). The passes that feed
//! a GEMM take the type of their output from the host ([`ActStore`]) in the
//! same entry, so the bf16 mode adds no kernel entry. q, k and v are one
//! `[hidden, 3 * hidden]` GEMM on one clamped input whenever their input clamps
//! are equal (every layer of the checkpoint). The sinusoidal table,
//! `relative_k_proj` of it and the query scale are computed once per layer at
//! load.

use std::borrow::Borrow;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::config_json;
use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_gpu_buf, set_u32};
use crate::embedgemma2::SoftSegment;
use crate::gemm::{cast_f32_to_bf16_into, gemm, select_backend, GemmBackend, GemmOperands};
use crate::json::{Json, NumForm};
use crate::loader::Loader;
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes};
use crate::qwen35_model::Precision;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::safetensors::SafeTensors;
use crate::tensor::{gpu_copy, DType, GpuBuffer, Tensor};

/// Frames (rows of `input_features`) one forward holds. Bounds the largest
/// activation, the first conv's output `[ceil(F/2), 64 * 128]`: 268 MB of f32
/// at this size. [`EmbedGemma2Audio::encode_audio`] splits a batch into as
/// many forwards as this needs, so it bounds memory, not what a caller may
/// pass; a single clip above it is refused.
pub const MAX_FRAMES_PER_FORWARD: usize = 16_384;
/// Clips per forward (the clip table's size). A call may hold more
/// ([`MAX_CALL_CLIPS`], [`MAX_CALL_FRAMES`]): it runs as several forwards
/// behind its one wait.
pub const MAX_CLIPS: usize = 256;
/// Clips per [`EmbedGemma2Audio::encode_audio`] /
/// [`EmbedGemma2Audio::queue_audio`] call (4096, sixteen forwards of
/// [`MAX_CLIPS`]): a sanity bound beside [`MAX_CALL_FRAMES`], which is the
/// memory one. A call over it is refused naming it.
pub const MAX_CALL_CLIPS: usize = 4096;
/// Frames (rows of `input_features`) per call: the memory bound of a call that
/// is cut into several forwards. Every forward's uploaded inputs (the features,
/// 512 bytes a frame at the checkpoint's 128 mel bins, and the masks) and its
/// soft tokens (`embed_dim` f32 per tower row, a quarter of a frame's rows) are
/// held until the call's one wait, about 1.25 KiB a frame, so a call holds up to
/// 2.5 GiB of them at this bound, beside one activation set of at most
/// [`MAX_FRAMES_PER_FORWARD`] frames. A call over it is refused naming it,
/// before any allocation; split the call (each call waits once) to go past it.
/// 2^21 frames is 128 full forwards, 5.8 hours of audio at 10 ms a frame, and 2.7
/// times the 768,000 frames of 256 clips of 30 s.
pub const MAX_CALL_FRAMES: usize = 1 << 21;

/// Words per clip in the table the kernels read (`EGA_CLIP_WORDS`).
const CLIP_WORDS: usize = 8;
/// Limits the kernels are compiled for (`EGA_MAX_*`).
const MAX_C0: u32 = 512;
const MAX_C1: u32 = 128;
const MAX_CTX: u32 = 64;
const MAX_HEAD_DIM: u32 = 256;
/// Frequency positions one simdgroup of the second subsampling conv holds
/// (`EGA_CONV1_P`).
const CONV1_POSITIONS: usize = 8;

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
        let root = config_json::parse_object(text)?;
        let c = root
            .get("audio_config")
            .ok_or("config.json: missing \"audio_config\"")?;
        let uint = |k: &str| config_json::uint(c, k);
        let num = |k: &str| config_json::float(config_json::field(c, k)?, k);
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
                    Json::Num {
                        form: NumForm::Uint,
                        int: n,
                        ..
                    } => u32::try_from(*n).map_err(|_| "config.json: a subsampling channel exceeds u32".to_string()),
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
            Some(t) => config_json::uint(t, "hidden_size")?,
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
        if self.past() > MAX_PAST {
            return bad(format!(
                "attention_context_left {} means {} past frames; the kernel holds {MAX_PAST}",
                self.context_left,
                self.past()
            ));
        }
        // A masked logit's softmax weight must be exactly 0 beside any unmasked
        // one (the softcap keeps those within +-cap), which `exp` gives from
        // 104 below; the kernel computes only the unmasked slots.
        if self.invalid_logit > -(self.logit_cap + 104.0) {
            return bad(format!(
                "attention_invalid_logits_value {} must be at most -(attention_logit_cap + 104) = {}",
                self.invalid_logit,
                -(self.logit_cap + 104.0)
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

const WHAT: &str = "embedgemma2_audio";

/// The element type a pass stores its output in: f32, or bf16 where the output
/// is the operand of a bf16 GEMM ([`GemmOperands::Bf16`]), rounded to nearest
/// even as the GPU's own cast does. A bf16 output is a buffer of its own: a
/// pass refuses one that is also an input, or the f32 output of the same
/// launch (two bf16 elements share a word, so an in-place store would race
/// with the reads of its neighbour).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActStore {
    #[default]
    F32,
    Bf16,
}

impl ActStore {
    fn is_bf16(self) -> bool {
        self == ActStore::Bf16
    }

    /// The device buffer `buf` holds `n` elements of this type.
    fn require(self, rt: &GpuRuntime, buf: &GpuBuffer, n: usize, what: &str) -> Result<(), String> {
        match self {
            ActStore::F32 => require::<f32>(rt, buf, n, what),
            ActStore::Bf16 => require::<u16>(rt, buf, n, what),
        }
    }
}

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
/// `mels2 = ceil(mels1 / 2)`. `c0 <= 512`, `c1 <= 128`.
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
    if mels1 == 0 || c0 == 0 || c0 > MAX_C0 || c1 == 0 || c1 > MAX_C1 || !(eps > 0.0 && eps.is_finite()) {
        return Err(format!(
            "{E}: need mels1 > 0, 1 <= c0 <= {MAX_C0}, 1 <= c1 <= {MAX_C1}, eps > 0 (got {mels1}, {c0}, {c1}, {eps})"
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
    // One simdgroup per (row, block of CONV1_POSITIONS frequency positions);
    // threadgroup memory 0: the staged input patch, the accumulators, and an
    // 8x8 weight-edge tile (`kernels/embedgemma2_audio.metal`).
    let groups_x = clips.max_rows() * (mels2 as usize).div_ceil(CONV1_POSITIONS);
    let tg_bytes = 4 * (CONV1_POSITIONS * c0.next_multiple_of(8) as usize + 8 * c1.next_multiple_of(8) as usize + 64);
    if groups_x == 0 {
        return Ok(());
    }
    crate::dispatch::check_threadgroup_limit(rt, &p, 32)?;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
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
        bnd.set_threadgroup_memory(0, tg_bytes);
        bnd.dispatch(mtl_size(groups_x, clips.clips.len(), 1), mtl_size(32, 1, 1));
        Ok(())
    })
}

/// What [`rms_norm`] computes per row: `[resid | bias +] scale * clamp(post)(
/// act(rms_norm(clamp(pre[1])(clamp(pre[0])(x))) [* weight]))`. With `norm`
/// false there is no normalization: the clamped value goes on to the
/// activation, the scale and the add (the residual add after a clipped
/// linear's output clamp; the output projection's bias).
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
    /// Multiplies the normed value before the add.
    pub scale: f32,
    /// `+ resid[row, d]`: a buffer of the input's size.
    pub resid: Option<&'a GpuBuffer>,
    /// `+ bias[d]`: a buffer of one row. At most one of `resid` and `bias`.
    pub bias: Option<&'a GpuBuffer>,
    pub norm: bool,
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
            bias: None,
            norm: true,
        }
    }
}

/// Words of the kernel's `EgaRowStage`.
const STAGE_WORDS: usize = 13;

/// A stage's checked kernel words, for an input of `n` elements in rows of
/// `dim`; `posts` are the clamps of its outputs (one to three) and `stores`
/// their element types.
fn stage_words(
    rt: &GpuRuntime,
    spec: &RmsNorm<'_>,
    posts: &[Clamp],
    stores: &[ActStore],
    n: usize,
    dim: u32,
    e: &str,
) -> Result<[u32; STAGE_WORDS], String> {
    if dim == 0 || !(spec.eps > 0.0 && spec.eps.is_finite()) || !spec.scale.is_finite() {
        return Err(format!("{e}: need dim > 0, eps > 0 and a finite scale"));
    }
    let mut flags = 0u32;
    if let Some(w) = spec.weight {
        require::<f32>(rt, w, dim as usize, "rms_norm weight")?;
        flags |= 1;
    }
    if spec.silu {
        flags |= 2;
    }
    match (spec.resid, spec.bias) {
        (Some(_), Some(_)) => return Err(format!("{e}: a stage adds a residual or a bias, not both")),
        (Some(r), None) => {
            require::<f32>(rt, r, n, "rms_norm resid")?;
            flags |= 4;
        }
        (None, Some(b)) => {
            require::<f32>(rt, b, dim as usize, "rms_norm bias")?;
            flags |= 16;
        }
        (None, None) => {}
    }
    if !spec.norm {
        flags |= 8;
    }
    for (i, s) in stores.iter().enumerate().take(3) {
        if s.is_bf16() {
            flags |= 32 << i;
        }
    }
    let mut w = [0u32; STAGE_WORDS];
    w[..5].copy_from_slice(&[
        word(spec.eps),
        word(spec.pre[0].lo),
        word(spec.pre[0].hi),
        word(spec.pre[1].lo),
        word(spec.pre[1].hi),
    ]);
    // Unused outputs hold an empty clamp; the kernel never reads them.
    for i in 0..3 {
        let c = posts.get(i).copied().unwrap_or(Clamp::NONE);
        (w[5 + i], w[8 + i]) = (word(c.lo), word(c.hi));
    }
    (w[11], w[12]) = (word(spec.scale), flags);
    Ok(w)
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
    norm_launch(
        rt,
        "embedgemma2_audio::rms_norm",
        x,
        out,
        rows,
        dim,
        spec,
        ActStore::F32,
        None,
        [ActStore::F32; 3],
    )
}

/// [`rms_norm`] storing `out` as `store`. A bf16 `out` is the operand of the
/// GEMM that reads it and is a buffer of its own (not `x`, and not `spec`'s
/// residual).
pub fn rms_norm_store(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    spec: &RmsNorm<'_>,
    store: ActStore,
) -> Result<(), String> {
    norm_launch(
        rt,
        "embedgemma2_audio::rms_norm_store",
        x,
        out,
        rows,
        dim,
        spec,
        store,
        None,
        [ActStore::F32; 3],
    )
}

/// A second row stage chained after [`rms_norm`]'s, and where it writes.
pub struct NormThen<'a> {
    /// The second stage. Its `post` clamp, `scale`, `resid` and `bias` are
    /// refused (they stay at `new`'s values): the outputs carry the clamps.
    pub spec: RmsNorm<'a>,
    /// One to three outputs: the second stage's value under each output's
    /// clamp, in its own buffer. `outs[0]` may be the first stage's `out`
    /// (the norm in place); the others are disjoint from every other buffer.
    pub outs: Vec<(&'a GpuBuffer, Clamp)>,
}

/// Two row stages in one launch, one threadgroup per row: `first` reads `x`
/// and writes `out`, exactly as [`rms_norm`] does, and `then.spec` reads
/// `out` (what `first` just wrote) and writes `then.outs`. The row is read
/// from memory once; the result is `rms_norm` twice, in sequence, to the last
/// bit (the reductions have the same shape). The layer's residual add and
/// the norm that follows it; the attention norm into the three clamped
/// copies q, k and v read.
pub fn rms_norm_then(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    first: &RmsNorm<'_>,
    then: &NormThen<'_>,
) -> Result<(), String> {
    norm_launch(
        rt,
        "embedgemma2_audio::rms_norm_then",
        x,
        out,
        rows,
        dim,
        first,
        ActStore::F32,
        Some(then),
        [ActStore::F32; 3],
    )
}

/// [`rms_norm_then`] storing the second stage's outputs as `stores` (`stores[i]`
/// is the element type of `then.outs[i]`; entries past the outputs are unused).
/// The first stage's `out` stays f32, since the second stage reads it. A bf16
/// output is a buffer of its own: not `x`, `out`, a residual, or another
/// output.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_then_store(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    first: &RmsNorm<'_>,
    then: &NormThen<'_>,
    stores: [ActStore; 3],
) -> Result<(), String> {
    norm_launch(
        rt,
        "embedgemma2_audio::rms_norm_then_store",
        x,
        out,
        rows,
        dim,
        first,
        ActStore::F32,
        Some(then),
        stores,
    )
}

#[allow(clippy::too_many_arguments)]
fn norm_launch(
    rt: &Arc<GpuRuntime>,
    e: &str,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    first: &RmsNorm<'_>,
    first_store: ActStore,
    then: Option<&NormThen<'_>>,
    then_stores: [ActStore; 3],
) -> Result<(), String> {
    let n = elems(&[rows, dim], e)?;
    require::<f32>(rt, x, n, "rms_norm x")?;
    first_store.require(rt, out, n, "rms_norm out")?;
    // A chained launch's first output is read back by the second stage as
    // f32, so only a lone stage stores its output bf16 (`rms_norm_then_store`
    // has no way to ask otherwise).
    debug_assert!(then.is_none() || !first_store.is_bf16());
    if first_store.is_bf16() {
        // A bf16 element shares its word with its neighbour: no in-place store.
        let mut reads = vec![("x", x)];
        reads.extend(first.resid.map(|r| ("resid", r)));
        reads.extend(first.bias.map(|b| ("bias", b)));
        reads.extend(first.weight.map(|w| ("weight", w)));
        require_disjoint_writes(e, &[("out", out)], &reads)?;
    }
    let a = stage_words(rt, first, &[first.post], &[first_store], n, dim, e)?;
    let (b, outs) = match then {
        None => ([0u32; STAGE_WORDS], &[][..]),
        Some(t) => {
            let s = &t.spec;
            if t.outs.is_empty() || t.outs.len() > 3 {
                return Err(format!("{e}: {} outputs; need 1..=3", t.outs.len()));
            }
            if s.post != Clamp::NONE || s.scale != 1.0 || s.resid.is_some() || s.bias.is_some() {
                return Err(format!(
                    "{e}: the second stage's clamp, scale and add belong to the first stage and the outputs"
                ));
            }
            for (&(o, _), st) in t.outs.iter().zip(then_stores) {
                st.require(rt, o, n, "rms_norm_then output")?;
            }
            let posts: Vec<Clamp> = t.outs.iter().map(|&(_, c)| c).collect();
            (
                stage_words(rt, s, &posts, &then_stores[..t.outs.len()], n, dim, e)?,
                &t.outs[..],
            )
        }
    };
    if rows == 0 {
        return Ok(());
    }
    // `out` is written element by element after a barrier-separated read of
    // the same element by the same thread, so it may be `x` or `resid`.
    let mut reads: Vec<(&str, &GpuBuffer)> = Vec::new();
    if let Some(w) = first.weight {
        reads.push(("weight", w));
    }
    if let Some(b) = first.bias {
        reads.push(("bias", b));
    }
    if let Some(t) = then {
        if let Some(w) = t.spec.weight {
            reads.push(("weight_b", w));
        }
        // The extra outputs overlap nothing; output 0 may be `out` (the second
        // stage's norm in place) but not a weight.
        const NAMES: [&str; 3] = ["output 0", "output 1", "output 2"];
        for (k, &(o, _)) in t.outs.iter().enumerate().skip(1) {
            let mut others = reads.clone();
            others.extend([("x", x), ("out", out)]);
            others.extend(
                t.outs
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(j, &(b, _))| (NAMES[j], b)),
            );
            require_disjoint_writes(e, &[(NAMES[k], o)], &others)?;
        }
        let mut others = reads.clone();
        others.extend(t.outs.iter().enumerate().skip(1).map(|(j, &(b, _))| (NAMES[j], b)));
        require_disjoint_writes(e, &[(NAMES[0], t.outs[0].0)], &others)?;
        // A bf16 output is read by no stage of the launch: it overlaps neither
        // the input, the first stage's output nor its residual.
        for (k, (&(o, _), st)) in t.outs.iter().zip(then_stores).enumerate() {
            if st.is_bf16() {
                let mut inputs = vec![("x", x), ("out", out)];
                inputs.extend(first.resid.map(|r| ("resid", r)));
                require_disjoint_writes(e, &[(NAMES[k], o)], &inputs)?;
            }
        }
    }
    require_disjoint_writes(e, &[("out", out)], &reads)?;
    let p = rt.pipeline("eg2a_rms_norm")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    let mut words = vec![rows, dim, outs.len() as u32];
    words.extend_from_slice(&a);
    words.extend_from_slice(&b);
    let params = bytes(&words);
    // Unused slots bind a buffer that exists; the kernel never touches them.
    let slot = |i: usize| outs.get(i).map_or(out, |&(o, _)| o);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, first.weight.unwrap_or(x), 1);
        set_gpu_buf(bnd, first.resid.or(first.bias).unwrap_or(x), 2);
        set_gpu_buf(bnd, out, 3);
        bnd.bind_bytes(&params, 4);
        set_gpu_buf(bnd, then.and_then(|t| t.spec.weight).unwrap_or(x), 5);
        set_gpu_buf(bnd, slot(0), 6);
        set_gpu_buf(bnd, slot(1), 7);
        set_gpu_buf(bnd, slot(2), 8);
    })
}

/// What [`ew`] computes per element: `clamp(post)(act(clamp(pre)(x)))`.
pub struct Ew {
    pub pre: Clamp,
    pub silu: bool,
    pub post: Clamp,
}

/// [`Ew`] over `n` elements, `x` into `out` (`out` may be `x`).
pub fn ew(rt: &Arc<GpuRuntime>, x: &GpuBuffer, out: &GpuBuffer, n: u32, spec: &Ew) -> Result<(), String> {
    ew_store(rt, x, out, n, spec, ActStore::F32)
}

/// [`ew`] storing `out` as `store`. A bf16 `out` is a buffer of its own, not
/// `x`: the second feed-forward GEMM's operand, from the first one's f32
/// output.
pub fn ew_store(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    spec: &Ew,
    store: ActStore,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::ew_store";
    require::<f32>(rt, x, n as usize, "ew x")?;
    store.require(rt, out, n as usize, "ew out")?;
    if store.is_bf16() {
        require_disjoint_writes(E, &[("out", out)], &[("x", x)])?;
    }
    if n == 0 {
        return Ok(());
    }
    let p = rt.pipeline("eg2a_ew")?;
    let params = bytes(&[
        n,
        word(spec.pre.lo),
        word(spec.pre.hi),
        word(spec.post.lo),
        word(spec.post.hi),
        (if spec.silu { 2 } else { 0 }) | (if store.is_bf16() { 4 } else { 0 }),
    ]);
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, out, 1);
        bnd.bind_bytes(&params, 2);
    })
}

/// Hidden width [`lconv`] holds in a threadgroup (`EGA_MAX_HIDDEN`).
const MAX_HIDDEN: u32 = 4096;

/// What [`lconv`] needs besides its buffers.
pub struct Lconv<'a> {
    pub eps: f32,
    /// The conv norm's weight, `[hidden]`.
    pub norm_weight: &'a GpuBuffer,
    /// `linear_start`'s output clamp, applied to both halves of the GLU.
    pub glu: Clamp,
    /// `+-gradient_clipping`, applied to the conv output before the norm.
    pub pre: Clamp,
    /// The next clipped linear's input clamp, applied last.
    pub post: Clamp,
}

/// The light conv's middle in one pass: for every clip's tower rows,
/// `out[t] = clamp(post)(silu(rms_norm(clamp(pre)(conv[t])) * norm_weight))`
/// with `conv[t, c] = sum_j w[c, j] * glu[t - (taps - 1) + j, c]` (rows before
/// the clip's first read 0) and `glu[s, c] = clamp(x[s, c]) * sigmoid(clamp(
/// x[s, hidden + c]))`. `x [rows, 2 * hidden]`, `w [hidden, taps]`,
/// `out [rows, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn lconv(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    hidden: u32,
    taps: u32,
    spec: &Lconv<'_>,
) -> Result<(), String> {
    lconv_store(rt, x, w, out, clips, hidden, taps, spec, ActStore::F32)
}

/// [`lconv`] storing `out` as `store`: the output of the light conv's middle is
/// `linear_end`'s GEMM operand.
#[allow(clippy::too_many_arguments)]
pub fn lconv_store(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    out: &GpuBuffer,
    clips: &ClipTable,
    hidden: u32,
    taps: u32,
    spec: &Lconv<'_>,
    store: ActStore,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::lconv";
    if hidden == 0 || hidden > MAX_HIDDEN || taps == 0 || !(spec.eps > 0.0 && spec.eps.is_finite()) {
        return Err(format!(
            "{E}: need 1 <= hidden <= {MAX_HIDDEN}, taps > 0 and eps > 0 (got {hidden}, {taps}, {})",
            spec.eps
        ));
    }
    check_table(rt, clips, E)?;
    let rows = clips.rows as u32;
    require::<f32>(rt, x, elems(&[rows, hidden, 2], E)?, "lconv x")?;
    store.require(rt, out, elems(&[rows, hidden], E)?, "lconv out")?;
    require::<f32>(rt, w, elems(&[hidden, taps], E)?, "lconv w")?;
    require::<f32>(rt, spec.norm_weight, hidden as usize, "lconv norm weight")?;
    require_disjoint_writes(
        E,
        &[("out", out)],
        &[
            ("x", x),
            ("w", w),
            ("norm_weight", spec.norm_weight),
            ("clips", &clips.buffer),
        ],
    )?;
    let p = rt.pipeline("eg2a_lconv")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), hidden as usize);
    let params = bytes(&[
        hidden,
        taps,
        word(spec.eps),
        word(spec.glu.lo),
        word(spec.glu.hi),
        word(spec.pre.lo),
        word(spec.pre.hi),
        word(spec.post.lo),
        word(spec.post.hi),
        u32::from(store.is_bf16()),
    ]);
    dispatch_2d_tg(rt, &p, clips.max_rows(), clips.clips.len(), tptg, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_gpu_buf(bnd, spec.norm_weight, 2);
        set_gpu_buf(bnd, out, 3);
        set_gpu_buf(bnd, &clips.buffer, 4);
        bnd.bind_bytes(&params, 5);
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
    /// `attention_invalid_logits_value`, the logit of a masked slot. It must
    /// be at most `-(cap + 104)`, so that every masked slot's softmax weight
    /// is exactly 0 beside any unmasked one (the real value is `-1e9`); the
    /// kernel relies on that and does not read it.
    pub invalid: f32,
    /// Output clamps of the q, k and v projections, applied as the kernel
    /// loads them, and the input clamp of the output projection, applied as
    /// it stores.
    pub q: Clamp,
    pub k: Clamp,
    pub v: Clamp,
    pub out: Clamp,
    /// Elements between the rows of `q`, `k` and `v`, and the column each
    /// starts at: `heads * head_dim` and 0 for three buffers of their own,
    /// `3 * heads * head_dim` and `0, h, 2h` (`h = heads * head_dim`) for one
    /// fused q|k|v projection passed three times.
    pub stride: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
}

/// Past frames the attention kernel carries partial logits for
/// (`EGA_MAX_PAST`).
const MAX_PAST: u32 = 16;

/// The chunked local relative-position attention of `Gemma4AudioAttention`
/// after its projections, for every clip; see `kernels/embedgemma2_audio.metal`
/// (`eg2a_attn`) for the exact rule. `q`, `k`, `v` (read at the spec's stride and offsets), `out [rows, heads *
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
    attention_store(rt, q, k, v, rel, q_mult, omask, out, clips, spec, ActStore::F32)
}

/// [`attention`] storing `out` as `store`: the output projection's GEMM
/// operand. `q`, `k` and `v` are f32 either way (the projection GEMM's
/// accumulators are f32 and the kernel is not a tile kernel: it has the
/// relative-position logits, the softcap and the window mask).
#[allow(clippy::too_many_arguments)]
pub fn attention_store(
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
    store: ActStore,
) -> Result<(), String> {
    const E: &str = "embedgemma2_audio::attention";
    let ctx = spec.chunk + spec.past;
    let rel_rows = ctx / 2 + 1;
    if spec.heads == 0
        || spec.head_dim == 0
        || spec.head_dim > MAX_HEAD_DIM
        || spec.chunk == 0
        || spec.past == 0
        || spec.past > MAX_PAST
        || ctx > MAX_CTX
        || ctx + 2 < spec.chunk + rel_rows
        || !(spec.cap > 0.0 && spec.cap.is_finite())
        || !(spec.k_scale.is_finite() && spec.invalid.is_finite())
        || spec.invalid > -(spec.cap + 104.0)
    {
        return Err(format!("{E}: unsupported shape or constants {spec:?}"));
    }
    check_table(rt, clips, E)?;
    let hidden = spec.heads * spec.head_dim;
    let n = elems(&[clips.rows as u32, hidden], E)?;
    // Each of q, k, v is read at `row * stride + off + column`.
    let n_in = elems(&[clips.rows as u32, spec.stride], E)?;
    for (off, name) in [(spec.q_off, "q"), (spec.k_off, "k"), (spec.v_off, "v")] {
        if u64::from(off) + u64::from(hidden) > u64::from(spec.stride) {
            return Err(format!(
                "{E}: {name} at column {off} of {hidden} does not fit a row stride of {}",
                spec.stride
            ));
        }
    }
    for (b, name) in [(q, "q"), (k, "k"), (v, "v")] {
        require::<f32>(rt, b, n_in, name)?;
    }
    store.require(rt, out, n, "out")?;
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
        word(spec.q.lo),
        word(spec.q.hi),
        word(spec.k.lo),
        word(spec.k.hi),
        word(spec.v.lo),
        word(spec.v.hi),
        word(spec.out.lo),
        word(spec.out.hi),
        spec.stride,
        spec.q_off,
        spec.k_off,
        spec.v_off,
        u32::from(store.is_bf16()),
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

/// The attention's q, k and v projections: three clipped linears over one
/// input.
struct Qkv {
    /// q's, k's and v's bounds.
    bounds: [Bounds; 3],
    w: QkvWeights,
}

enum QkvWeights {
    /// One `[hidden, 3 * hidden]` weight (q | k | v): the input clamps are
    /// equal, so one clamped copy of the normed input feeds one GEMM.
    Fused(Tensor),
    /// Three `[hidden, hidden]` weights, for input clamps that differ: each
    /// reads its own clamped copy.
    Split([Tensor; 3]),
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
    qkv: Qkv,
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
    /// Some layer's q, k and v have different input clamps, so the forward
    /// keeps three projection outputs and three clamped inputs.
    split_qkv: bool,
    /// Which [`AudioStage`]s a forward runs (all, unless a profiler narrowed it).
    stage_mask: AtomicU32,
    /// How the projections are held and the GEMMs run
    /// ([`EmbedGemma2Audio::load_with`]).
    operands: GemmOperands,
}

/// One dispatch group of the forward, so a profiler can run a stage alone or
/// leave it out ([`EmbedGemma2Audio::set_stage_mask`]). A fused dispatch
/// belongs to the stage named for its main work, and says what else it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AudioStage {
    SubConv0,
    SubConv1,
    InputProj,
    /// The first feed-forward's pre-norm (the second's is in `LconvResid`).
    FfnNormPre,
    FfnUp,
    FfnAct,
    FfnDown,
    /// The feed-forward's post-norm and residual add, and the norm after it:
    /// the attention norm (into q's, k's and v's clamped inputs) after the
    /// first feed-forward, `norm_out` after the second.
    FfnNormPost,
    QkvGemm,
    Attention,
    AttnOutGemm,
    /// The attention's post-norm and residual add, and the light conv's
    /// pre-norm.
    AttnNormPost,
    LconvStart,
    /// The GLU, the depthwise conv and the conv norm with its SiLU.
    Lconv,
    LconvEnd,
    /// `linear_end`'s output clamp and the residual add, and the second
    /// feed-forward's pre-norm.
    LconvResid,
    OutProj,
    /// The output bias, and the embedder's weightless norm after it.
    OutBias,
    EmbedProj,
}

impl AudioStage {
    /// Every stage, in forward order.
    pub const ALL: [AudioStage; 19] = [
        AudioStage::SubConv0,
        AudioStage::SubConv1,
        AudioStage::InputProj,
        AudioStage::FfnNormPre,
        AudioStage::FfnUp,
        AudioStage::FfnAct,
        AudioStage::FfnDown,
        AudioStage::FfnNormPost,
        AudioStage::QkvGemm,
        AudioStage::Attention,
        AudioStage::AttnOutGemm,
        AudioStage::AttnNormPost,
        AudioStage::LconvStart,
        AudioStage::Lconv,
        AudioStage::LconvEnd,
        AudioStage::LconvResid,
        AudioStage::OutProj,
        AudioStage::OutBias,
        AudioStage::EmbedProj,
    ];

    /// The mask with every stage set.
    pub const MASK_ALL: u32 = (1u32 << Self::ALL.len() as u32) - 1;

    /// This stage's bit in a mask.
    pub const fn bit(self) -> u32 {
        1u32 << self as u32
    }

    pub const fn name(self) -> &'static str {
        match self {
            AudioStage::SubConv0 => "subsample conv0+ln+relu",
            AudioStage::SubConv1 => "subsample conv1+ln+relu",
            AudioStage::InputProj => "input_proj GEMM",
            AudioStage::FfnNormPre => "ffn1 pre-norm",
            AudioStage::FfnUp => "ffn up GEMM (1024->4096)",
            AudioStage::FfnAct => "ffn clamp+silu",
            AudioStage::FfnDown => "ffn down GEMM (4096->1024)",
            AudioStage::FfnNormPost => "ffn post-norm+resid (+next norm)",
            AudioStage::QkvGemm => "q/k/v GEMMs",
            AudioStage::Attention => "attention",
            AudioStage::AttnOutGemm => "attn out GEMM",
            AudioStage::AttnNormPost => "attn post-norm+resid (+lconv norm)",
            AudioStage::LconvStart => "lconv start GEMM (1024->2048)",
            AudioStage::Lconv => "lconv glu+dwconv+norm+silu",
            AudioStage::LconvEnd => "lconv end GEMM",
            AudioStage::LconvResid => "lconv resid (+ffn2 pre-norm)",
            AudioStage::OutProj => "output_proj GEMM",
            AudioStage::OutBias => "output bias (+embedder norm)",
            AudioStage::EmbedProj => "embedding_projection GEMM",
        }
    }
}

/// Set in the stage mask once a profiler has called
/// [`EmbedGemma2Audio::set_stage_mask`]: the activations are then zeroed on
/// allocation, even with every stage on, so a profiler's variants all pay the
/// same allocation.
const PROFILED: u32 = 1 << 31;

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
    y: Tensor,
    /// q, k and v of a split projection (see [`QkvWeights`]); `None` when
    /// every layer fuses them, which then land in a prefix of `big`.
    split: Option<[Tensor; 3]>,
    /// The feed-forward's `[rows, 4 * hidden]`; the GLU's input is its
    /// `[rows, 2 * hidden]` prefix.
    big: Tensor,
    /// `[rows, out_dims]` (under `ExactF32` `hidden` wide at least: k's clamped
    /// input of a split projection is in it, free until the tail).
    tower: Tensor,
    /// The GEMM operands under [`GemmOperands::ExactF32`]; `None` under `Bf16`.
    ex: Option<ExactActs>,
    /// The GEMM operands under [`GemmOperands::Bf16`]; `None` under `ExactF32`.
    bf: Option<Bf16Acts>,
}

/// The f32 operands of the exact forward's GEMMs.
struct ExactActs {
    /// The norms' outputs, then the attention's and the GLU's.
    n: Tensor,
    /// The attention norm, clamped for q's input (k's and v's are in `tower`
    /// and `emb`).
    c: Tensor,
    /// The embedder's normed input; also v's clamped input of a split
    /// projection, free until the tail (`[rows, max(hidden, out_dims)]`).
    emb: Tensor,
}

/// The bf16 operands of the GEMMs, each written by the pass that produces it
/// (see [`ActStore`]), except the input and output projections' (one cast of
/// the subsampling's output and of the residual stream per forward). A GEMM's
/// output stays in the f32 buffers of [`Acts`].
struct Bf16Acts {
    /// `[rows, max(proj_in, hidden, out_dims)]`: the operand of whichever GEMM
    /// is next of the ones that read a `[rows, hidden]`-wide norm or pass
    /// output (the feed-forwards' and the attention output's and light conv's
    /// linears), and at the ends of the forward the input projection's (the
    /// subsampling's output cast) and the output projection's (the residual
    /// stream cast) and the embedder's (the weightless norm's output).
    /// Each is read by its GEMM before the next is written: the runtime separates
    /// dispatches with barriers, and the f32 forward reuses its `n` the same way.
    n: Tensor,
    /// q's clamped attention input; a split projection's k's and v's are `ck`
    /// and `cv` (`None` when every layer fuses them).
    c: Tensor,
    ck: Option<Tensor>,
    cv: Option<Tensor>,
    /// The feed-forwards' activation, `[rows, 4 * hidden]`: the down
    /// projection's operand.
    big: Tensor,
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

/// A batch of clips, owned or borrowed, as references.
fn refs<C: Borrow<AudioInput>>(clips: &[C]) -> Vec<&AudioInput> {
    clips.iter().map(Borrow::borrow).collect()
}

/// What a forward leaves for the host once it has run: its output and, with
/// `trace`, the traced tensors, and where each clip's rows sit in them. The
/// rest of [`ForwardIo`] (the features, masks and clip table the GPU reads) is
/// not kept.
struct ForwardOut {
    /// `[rows, embed_dim]`.
    soft: Tensor,
    trace: Vec<Tensor>,
    clips: Vec<ClipRows>,
}

/// What a queued [`EmbedGemma2Audio::queue_audio`] call keeps alive: each
/// forward's device-written outputs, and the clips' frame masks the host
/// splits the output by. The activations and each forward's host-written
/// inputs are not here: they were handed to the runtime's deferred release when
/// the call was queued, so the caller's wait recycles them.
struct AudioPendingState {
    /// Per forward: the clips it ran (indices into the call) and its outputs.
    outs: Vec<(Vec<usize>, ForwardOut)>,
    /// Each clip's `[frames]` mask.
    masks: Vec<Vec<bool>>,
    trace: bool,
}

/// An [`EmbedGemma2Audio::queue_audio`] call whose forwards are queued and not
/// yet waited for. [`Self::collect`] waits for them (if the caller has not) and
/// reads the soft tokens; dropping it without collecting waits for the queued
/// work and discards the result, since Metal 4 command buffers do not retain
/// the buffers they bind. It holds only each forward's outputs: the
/// activations and the forwards' uploaded inputs are the runtime's, released at
/// the wait, so a call followed by another holds one activation set.
pub struct PendingAudio {
    rt: Arc<GpuRuntime>,
    hidden: usize,
    layers: usize,
    out_dims: usize,
    embed: usize,
    state: Option<AudioPendingState>,
}

impl PendingAudio {
    /// How many forwards were queued.
    pub fn forwards(&self) -> u32 {
        self.state.as_ref().map_or(0, |s| s.outs.len() as u32)
    }

    /// Hand the call's soft tokens to a caller that keeps them on the device:
    /// per clip, in the order of the call, the rows of its forward's output
    /// that its tower mask keeps ([`out_mask`]), as one segment per maximal run
    /// of kept rows (a clip whose mask is a prefix, as a processor's is, has
    /// one; a clip with nothing kept has none). The rows are not compacted:
    /// the consumer reads them where the tower wrote them. Consumes the call
    /// **without waiting** (so the caller owes the one wait, after queuing
    /// whatever reads these rows); the segments hold the output buffers, which
    /// the runtime keeps until that wait has seen the GPU finish.
    pub(crate) fn into_device(mut self) -> Vec<Vec<SoftSegment>> {
        let state = self.state.take().expect("into_device consumes the pending call");
        let mut clips: Vec<Vec<SoftSegment>> = vec![Vec::new(); state.masks.len()];
        for (g, out) in &state.outs {
            debug_assert_eq!(out.soft.byte_offset(), 0, "a forward's soft tokens start their buffer");
            for (&i, cr) in g.iter().zip(&out.clips) {
                let omask = out_mask(&state.masks[i]);
                debug_assert_eq!(omask.len(), cr.rows);
                clips[i].extend(kept_runs(&omask).into_iter().map(|r| SoftSegment {
                    buf: out.soft.buffer.clone(),
                    row: (cr.row0 + r.start) as u32,
                    rows: r.len() as u32,
                }));
            }
        }
        clips
    }

    /// The call's [`AudioOutput`]. Waits for the queued forwards first if
    /// nothing has (the wait is a no-op when the caller already waited).
    pub fn collect(mut self) -> Result<AudioOutput, String> {
        let state = self.state.take().expect("collect consumes the pending call");
        let AudioPendingState { outs, masks, trace } = state;
        // The buffers in `outs` must outlive the queued work, so a failed
        // wait still holds them until the end of this function.
        self.rt.synchronize()?;
        let n_clips = masks.len();
        let (embed, layers) = (self.embed, self.layers);
        let mut out = AudioOutput {
            soft_tokens: vec![Vec::new(); n_clips],
            out_masks: vec![Vec::new(); n_clips],
            trace: if trace {
                vec![
                    AudioClipTrace {
                        subsample: Vec::new(),
                        layers: Vec::new(),
                        tower_out: Vec::new(),
                    };
                    n_clips
                ]
            } else {
                Vec::new()
            },
            forwards: outs.len() as u32,
        };
        for (g, io) in &outs {
            // One host mapping at a time (the runtime's access is exclusive).
            let read =
                |t: &Tensor| -> Result<Vec<f32>, String> { Ok(t.buffer.try_contents_f32()?[..t.numel()].to_vec()) };
            let soft = read(&io.soft)?;
            let traced: Vec<Vec<f32>> = io.trace.iter().map(read).collect::<Result<_, _>>()?;
            for (&i, cr) in g.iter().zip(&io.clips) {
                let omask = out_mask(&masks[i]);
                debug_assert_eq!(omask.len(), cr.rows);
                let mut tokens = Vec::new();
                for run in kept_runs(&omask) {
                    tokens.extend_from_slice(&soft[(cr.row0 + run.start) * embed..(cr.row0 + run.end) * embed]);
                }
                out.soft_tokens[i] = tokens;
                out.out_masks[i] = omask;
                if trace {
                    let h = self.hidden;
                    let rows = |m: &[f32], w: usize| m[cr.row0 * w..(cr.row0 + cr.rows) * w].to_vec();
                    out.trace[i] = AudioClipTrace {
                        subsample: rows(&traced[0], h),
                        layers: (0..layers).map(|l| rows(&traced[1 + l], h)).collect(),
                        tower_out: rows(&traced[1 + layers], self.out_dims),
                    };
                }
            }
        }
        Ok(out)
    }
}

impl Drop for PendingAudio {
    fn drop(&mut self) {
        if self.state.is_some() {
            // The result cannot be returned; the wait is for the buffers. A
            // failure leaves the runtime poisoned, which its own checks report.
            let _ = self.rt.synchronize();
        }
    }
}

/// A feed-forward's pre-norm: `clip` of the residual stream, the norm, then
/// the first linear's input clamp.
fn ffn_pre_norm(f: &Ffn, eps: f32, gc: Clamp) -> RmsNorm<'_> {
    let mut pre = RmsNorm::new(eps, Some(&f.pre_norm));
    pre.pre = [gc, Clamp::NONE];
    pre.post = f.l1.bounds.input;
    pre
}

/// A feed-forward's post-norm and residual add (`resid` is the residual
/// stream, which the pass writes in place).
fn ffn_post_norm<'a>(f: &'a Ffn, eps: f32, gc: Clamp, scale: f32, resid: &'a GpuBuffer) -> RmsNorm<'a> {
    let mut post = RmsNorm::new(eps, Some(&f.post_norm));
    post.pre = [f.l2.bounds.output, gc];
    post.scale = scale;
    post.resid = Some(resid);
    post
}

impl EmbedGemma2Audio {
    /// Load the tower from `st`, whose tensors are named
    /// `{prefix}audio_tower.*` and `{prefix}embed_audio.*`. Every tensor's
    /// shape is checked against `cfg`; a missing one is an error, and so is
    /// any tensor under either prefix the forward does not read.
    ///
    /// This is [`GemmOperands::ExactF32`] ([`Self::load_with`]): every
    /// projection widened to f32 and every GEMM exact f32.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, prefix: &str, cfg: AudioConfig) -> Result<Self, String> {
        Self::load_with(rt, st, prefix, cfg, GemmOperands::ExactF32)
    }

    /// [`Self::load`] with the operand precision chosen (the module docs,
    /// Precision).
    ///
    /// * [`GemmOperands::ExactF32`]: every projection widened to f32 on the
    ///   device, every GEMM exact f32 (transformers' fp32 forward up to
    ///   operation order). `encode_audio` refuses a runtime whose relaxed f32
    ///   GEMMs are on.
    /// * [`GemmOperands::Bf16`]: every projection (the input projection, each
    ///   layer's feed-forwards, q|k|v, the attention output and the light
    ///   conv's two linears, the output projection and the embedder's) kept as
    ///   the checkpoint stores it, bf16, with no f32 copy, and every GEMM run
    ///   with bf16 operands and f32 accumulation. The passes that feed a GEMM
    ///   store its operand bf16 themselves ([`ActStore`]), so no cast pass is
    ///   added per GEMM. The checkpoint's projections must be bf16 (a tensor
    ///   stored in another dtype is refused by name rather than rounded) and
    ///   the device needs TensorOps. `encode_audio` accepts a runtime with
    ///   relaxed f32 GEMMs on, which no GEMM of this forward uses.
    pub fn load_with(
        rt: &Arc<GpuRuntime>,
        st: &SafeTensors,
        prefix: &str,
        cfg: AudioConfig,
        operands: GemmOperands,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let proj = match operands {
            GemmOperands::ExactF32 => Precision::F32,
            GemmOperands::Bf16 => {
                if !rt.has_tensorops() {
                    return Err(
                        "EmbedGemma2Audio::load_with: bf16 operands need TensorOps, which this device lacks".into(),
                    );
                }
                Precision::Bf16
            }
        };
        let tower_prefix = format!("{prefix}{PREFIX_TOWER}");
        let embed_prefix = format!("{prefix}{PREFIX_EMBED}");
        let ld = Loader::new(rt, st, &tower_prefix);
        let le = Loader::new(rt, st, &embed_prefix);
        let (h, hd) = (cfg.hidden as usize, cfg.head_dim() as usize);
        let (c0, c1) = (cfg.sub_channels[0] as usize, cfg.sub_channels[1] as usize);

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
        let input_proj = ld.linear(&[(&sub("input_proj_linear.weight"), h)], cfg.proj_in() as usize, proj)?;

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
                w: ld.linear(&[(&format!("{name}.linear.weight"), out)], inp, proj)?,
                bounds: bound(name)?,
            })
        };

        // q, k and v read one tensor, so with equal input clamps (every layer
        // of the checkpoint) they are one clamped input and one GEMM.
        let qkv = |base: &str| -> Result<Qkv, String> {
            let names = ["q_proj", "k_proj", "v_proj"].map(|n| format!("{base}.{n}"));
            let bounds = [bound(&names[0])?, bound(&names[1])?, bound(&names[2])?];
            let [wq, wk, wv] = names.map(|n| format!("{n}.linear.weight"));
            let w = if bounds[0].input == bounds[1].input && bounds[1].input == bounds[2].input {
                QkvWeights::Fused(ld.linear(&[(&wq, h), (&wk, h), (&wv, h)], h, proj)?)
            } else {
                QkvWeights::Split([
                    ld.linear(&[(&wq, h)], h, proj)?,
                    ld.linear(&[(&wk, h)], h, proj)?,
                    ld.linear(&[(&wv, h)], h, proj)?,
                ])
            };
            Ok(Qkv { bounds, w })
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
                qkv: qkv(&p("self_attn"))?,
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
        let output_proj = ld.linear(&[("output_proj.weight", out_dims)], h, proj)?;
        let output_bias = ld.norm("output_proj.bias", out_dims)?;
        let embed_proj = le.linear(
            &[("embedding_projection.weight", cfg.embed_dim as usize)],
            out_dims,
            proj,
        )?;
        ld.refuse_unread()?;
        le.refuse_unread()?;
        rt.synchronize()?;
        let split_qkv = layers.iter().any(|l| matches!(l.qkv.w, QkvWeights::Split(_)));
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
            split_qkv,
            stage_mask: AtomicU32::new(AudioStage::MASK_ALL),
            operands,
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.cfg
    }

    /// The operand precision this model was loaded with.
    pub fn operands(&self) -> GemmOperands {
        self.operands
    }

    /// Bytes of weights held on the device: every projection in the dtype the
    /// mode holds it in (f32 or bf16), the convolutions, the norms, the
    /// relative-key tables and the query scales (f32). The activations are not
    /// counted.
    pub fn weight_bytes(&self) -> usize {
        let t = |t: &Tensor| t.nbytes_logical();
        let b = |b: &GpuBuffer| b.nbytes();
        let ffn = |f: &Ffn| b(&f.pre_norm) + b(&f.post_norm) + t(&f.l1.w) + t(&f.l2.w);
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                let qkv = match &l.qkv.w {
                    QkvWeights::Fused(w) => t(w),
                    QkvWeights::Split(ws) => ws.iter().map(t).sum(),
                };
                ffn(&l.ff1)
                    + ffn(&l.ff2)
                    + [
                        &l.norm_pre_attn,
                        &l.norm_post_attn,
                        &l.norm_out,
                        &l.lc_pre_norm,
                        &l.lc_conv_norm,
                    ]
                    .into_iter()
                    .map(b)
                    .sum::<usize>()
                    + qkv
                    + t(&l.post.w)
                    + b(&l.rel)
                    + b(&l.q_mult)
                    + t(&l.lc_start.w)
                    + b(&l.lc_conv)
                    + t(&l.lc_end.w)
            })
            .sum();
        b(&self.conv0_w)
            + b(&self.conv0_norm)
            + b(&self.conv1_w)
            + b(&self.conv1_norm)
            + t(&self.input_proj)
            + layers
            + t(&self.output_proj)
            + b(&self.output_bias)
            + t(&self.embed_proj)
    }

    /// A profiler's switch: later forwards run only the [`AudioStage`]s whose
    /// bits are set in `mask`. The output of a narrowed forward is
    /// meaningless. From the first call on, activations are zeroed on
    /// allocation (even with every stage on), so a stage that reads what a
    /// skipped one would have written reads zeros, not another run's values.
    #[doc(hidden)]
    pub fn set_stage_mask(&self, mask: u32) {
        self.stage_mask
            .store((mask & AudioStage::MASK_ALL) | PROFILED, Ordering::Release);
    }

    /// Layers whose q, k and v run as one fused GEMM on one clamped input
    /// (their input clamps are equal, as in every layer of the checkpoint);
    /// the others project each of q, k, v on its own.
    #[doc(hidden)]
    pub fn fused_qkv_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.qkv.w, QkvWeights::Fused(_)))
            .count()
    }

    /// Undo [`Self::set_stage_mask`]: every stage, unzeroed activations.
    #[doc(hidden)]
    pub fn clear_stage_mask(&self) {
        self.stage_mask.store(AudioStage::MASK_ALL, Ordering::Release);
    }

    /// The runtime this model was loaded on.
    pub(crate) fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }

    /// Everything [`Self::encode_audio`] checks before it touches the device,
    /// with no allocation and no wait: the batch is non-empty and within
    /// [`MAX_CALL_CLIPS`] and [`MAX_CALL_FRAMES`] (a call of more than
    /// [`MAX_CLIPS`] clips is cut into several forwards behind its one wait),
    /// the runtime's f32 GEMMs are exact (under
    /// [`GemmOperands::ExactF32`]; a bf16 forward runs none), and each clip has
    /// frames (at most [`MAX_FRAMES_PER_FORWARD`]), `frames * mel_bins` finite
    /// features. Returns the number of soft tokens each clip will produce (the
    /// true entries of its tower mask), which is what a caller needs to lay out
    /// the placeholders of its text sequence before the tower has run.
    pub fn check_clips<C: Borrow<AudioInput>>(&self, clips: &[C]) -> Result<Vec<usize>, String> {
        let clips = refs(clips);
        self.check_call("EmbedGemma2Audio::check_clips", &clips)?;
        Ok(clips
            .iter()
            .map(|c| out_mask(&c.mask).into_iter().filter(|&k| k).count())
            .collect())
    }

    /// [`Self::check_clips`]'s checks.
    fn check_call(&self, what: &str, clips: &[&AudioInput]) -> Result<(), String> {
        let cfg = &self.cfg;
        if clips.is_empty() {
            return Err(format!("{what}: empty batch"));
        }
        if clips.len() > MAX_CALL_CLIPS {
            return Err(format!(
                "{what}: {} clips exceeds the per-call limit MAX_CALL_CLIPS = {MAX_CALL_CLIPS}",
                clips.len()
            ));
        }
        let total_frames = clips.iter().fold(0usize, |acc, c| acc.saturating_add(c.mask.len()));
        if total_frames > MAX_CALL_FRAMES {
            return Err(format!(
                "{what}: {} clips make {total_frames} frames; the per-call memory bound MAX_CALL_FRAMES is \
                 {MAX_CALL_FRAMES} (split the call)",
                clips.len()
            ));
        }
        // No GEMM of the bf16 forward is an f32 one, so the switch does not
        // touch it.
        if self.operands == GemmOperands::ExactF32 && self.rt.f32_gemm_runs_relaxed() {
            return Err(format!(
                "{what}: the forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        let mels = cfg.mel_bins() as usize;
        for (i, c) in clips.iter().enumerate() {
            let frames = c.mask.len();
            if frames == 0 {
                return Err(format!("{what}: clip {i} has no frames"));
            }
            if frames > MAX_FRAMES_PER_FORWARD {
                return Err(format!(
                    "{what}: clip {i} has {frames} frames; the limit is {MAX_FRAMES_PER_FORWARD} (split it)"
                ));
            }
            if c.input_features.len() != frames * mels {
                return Err(format!(
                    "{what}: clip {i} has {} feature values for {frames} frames of {mels} mel bins ({} expected)",
                    c.input_features.len(),
                    frames * mels
                ));
            }
            if let Some(at) = c.input_features.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "{what}: clip {i}: input_features[{at}] = {} is not finite",
                    c.input_features[at]
                ));
            }
        }
        Ok(())
    }

    /// Embed a batch of clips into soft tokens. Each clip's output is its own
    /// tower rows through the embedder, kept where the tower's mask is true;
    /// it does not depend on what else is in the batch beyond GEMM tiling.
    ///
    /// The batch runs as one or more forwards (clips in order, cut at
    /// [`MAX_FRAMES_PER_FORWARD`] frames), all queued under a
    /// [`GpuRuntime::async_scope`] (whatever the runtime's `async_encode`
    /// setting) before the host waits once. `trace` also returns each clip's
    /// subsample output, per-layer residuals and tower output. This is
    /// [`Self::queue_audio`], the wait, and [`PendingAudio::collect`].
    pub fn encode_audio(&self, clips: &[AudioInput], trace: bool) -> Result<AudioOutput, String> {
        let pending = self.queue_impl("EmbedGemma2Audio::encode_audio", &refs(clips), trace)?;
        self.rt.synchronize()?;
        pending.collect()
    }

    /// [`Self::encode_audio`] up to the wait: the call is checked and its
    /// forwards are queued, and the host does not wait for them. The returned
    /// [`PendingAudio`] holds every buffer the queued work binds; a caller with
    /// other work to queue behind it (the vision tower, the text encoder) does
    /// so, waits once, and then [`PendingAudio::collect`]s. Nothing is queued
    /// when the call is refused, and a call that fails while queuing waits for
    /// what it queued before it returns the error.
    pub fn queue_audio<C: Borrow<AudioInput>>(&self, clips: &[C], trace: bool) -> Result<PendingAudio, String> {
        self.queue_impl("EmbedGemma2Audio::queue_audio", &refs(clips), trace)
    }

    fn queue_impl(&self, what: &str, clips: &[&AudioInput], trace: bool) -> Result<PendingAudio, String> {
        self.check_call(what, clips)?;

        // Clips in order, cut where a forward would pass the frame budget or
        // the clip table's size.
        let mut plan: Vec<Vec<usize>> = Vec::new();
        let mut used = 0usize;
        for (i, c) in clips.iter().enumerate() {
            let f = c.mask.len();
            match plan.last_mut() {
                Some(g) if g.len() < MAX_CLIPS && used + f <= MAX_FRAMES_PER_FORWARD => {
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

        // Held by the returned `PendingAudio` until the work is waited for
        // (Metal 4 command buffers do not retain what they bind), including
        // the buffers of a forward that fails midway.
        let scope = self.rt.async_scope();
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
        // Hand everything the GPU only reads to the runtime *before* the wait:
        // the activations and each forward's uploaded inputs and clip table. A
        // dropped pooled buffer is parked in the runtime's deferred-release
        // queue, which holds it until a waiting commit has seen the GPU finish
        // and that wait then returns it to the pool's freelist (see
        // `EmbedGemma2Model::encode`). Held until `collect` instead, they
        // reached the freelist only at the next commit, after the next call had
        // allocated its own set. Only each forward's outputs stay; a forward
        // that failed midway is released the same way.
        let outs: Vec<(Vec<usize>, ForwardOut)> = pending
            .into_iter()
            .map(|(g, io)| {
                let ForwardIo { clips, soft, trace, .. } = io;
                (
                    g,
                    ForwardOut {
                        soft,
                        trace,
                        clips: clips.clips,
                    },
                )
            })
            .collect();
        drop(acts);
        let out = PendingAudio {
            rt: Arc::clone(&self.rt),
            hidden: self.cfg.hidden as usize,
            layers: self.cfg.layers as usize,
            out_dims: self.cfg.out_dims as usize,
            embed: self.cfg.embed_dim as usize,
            state: Some(AudioPendingState {
                outs,
                masks: clips.iter().map(|c| c.mask.clone()).collect(),
                trace,
            }),
        };
        drop(scope);
        // On an error `out` is dropped, which waits for what was queued (see
        // its `Drop`).
        queued?;
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
        let out_dims = cfg.out_dims as usize;
        let narrowed = self.stage_mask.load(Ordering::Acquire) & PROFILED != 0;
        let f32s = |shape: &[usize]| {
            if narrowed {
                rt.alloc_tensor_f32(shape)
            } else {
                rt.alloc_tensor_unzeroed(shape, DType::F32)
            }
        };
        let bf16s = |shape: &[usize]| {
            if narrowed {
                rt.alloc_tensor_bf16(shape)
            } else {
                rt.alloc_tensor_unzeroed(shape, DType::BF16)
            }
        };
        let (ex, bf, tower_w) = match self.operands {
            // `tower` and `emb` also hold k's and v's clamped input, `hidden` wide.
            GemmOperands::ExactF32 => {
                let tail_w = out_dims.max(h);
                let ex = ExactActs {
                    n: f32s(&[rows, h])?,
                    c: f32s(&[rows, h])?,
                    emb: f32s(&[rows, tail_w])?,
                };
                (Some(ex), None, tail_w)
            }
            GemmOperands::Bf16 => {
                let bf = Bf16Acts {
                    n: bf16s(&[rows, (cfg.proj_in() as usize).max(h).max(out_dims)])?,
                    c: bf16s(&[rows, h])?,
                    ck: if self.split_qkv { Some(bf16s(&[rows, h])?) } else { None },
                    cv: if self.split_qkv { Some(bf16s(&[rows, h])?) } else { None },
                    big: bf16s(&[rows, 4 * h])?,
                };
                (None, Some(bf), out_dims)
            }
        };
        Ok(Acts {
            mid: f32s(&[mid, mels1 * cfg.sub_channels[0] as usize])?,
            sub: f32s(&[rows, cfg.proj_in() as usize])?,
            x: f32s(&[rows, h])?,
            y: f32s(&[rows, h])?,
            split: if self.split_qkv {
                Some([f32s(&[rows, h])?, f32s(&[rows, h])?, f32s(&[rows, h])?])
            } else {
                None
            },
            big: f32s(&[rows, 4 * h])?,
            tower: f32s(&[rows, tower_w])?,
            ex,
            bf,
        })
    }

    /// This forward's own host-written inputs and host-read outputs for the
    /// clips `g`, written without waiting for the GPU.
    fn forward_io(&self, clips: &[&AudioInput], g: &[usize], trace: bool) -> Result<ForwardIo, String> {
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
        let (x, y) = (at(&a.x, h)?, at(&a.y, h)?);
        // What each GEMM reads, in the mode's operand type: f32 buffers that
        // the passes fill under `ExactF32`, bf16 ones under `Bf16` (every pass
        // that feeds a GEMM stores its output in `store`'s type, so no GEMM
        // reads a buffer cast for it, except at the two ends of the forward).
        //   n    the norm and pass outputs the feed-forwards', the attention
        //        output's and the light conv's linears read;
        //   cq, ck, cv  the attention norm, clamped for q's, k's and v's input
        //        (k's and v's only for a split projection);
        //   act  the feed-forward's activation, the down projection's operand.
        let store = if a.bf.is_some() { ActStore::Bf16 } else { ActStore::F32 };
        let (n, cq, ck, cv, act_buf, emb_wide) = match (&a.ex, &a.bf) {
            // Under `ExactF32`, k's and v's clamped input of a split projection
            // are in `tower` and `emb`, free until the tail of the forward.
            (Some(ex), None) => (
                at(&ex.n, h)?,
                at(&ex.c, h)?,
                Some(at(&a.tower, h)?),
                Some(at(&ex.emb, h)?),
                at(&a.big, 4 * h)?,
                at(&ex.emb, cfg.out_dims)?,
            ),
            (None, Some(bf)) => (
                at(&bf.n, h)?,
                at(&bf.c, h)?,
                bf.ck.as_ref().map(|t| at(t, h)).transpose()?,
                bf.cv.as_ref().map(|t| at(t, h)).transpose()?,
                at(&bf.big, 4 * h)?,
                at(&bf.n, cfg.out_dims)?,
            ),
            _ => return Err("EmbedGemma2Audio: the activations do not match the operand precision".into()),
        };
        // The fused projection's output q | k | v lands in a prefix of `big`;
        // a split one's in three buffers of their own.
        let qkv_fused = at(&a.big, 3 * h)?;
        let split = match &a.split {
            Some([q, k, v]) => Some([at(q, h)?, at(k, h)?, at(v, h)?]),
            None => None,
        };
        // The feed-forward's up projection's output (f32), and where the
        // activation pass leaves its result: in place under `ExactF32`, in its
        // own bf16 buffer under `Bf16`.
        let big = at(&a.big, 4 * h)?;
        let big2 = at(&a.big, 2 * h)?;
        let operands = self.operands;
        // `out = inp @ w` at the mode's precision. Under `Bf16` the operand
        // must already be bf16: `gemm_bf16` would cast an f32 one into a
        // temporary, which is a dispatch and an allocation this forward has
        // none of.
        let mm = |inp: &Tensor, w: &Tensor, out: &Tensor| -> Result<(), String> {
            match operands {
                GemmOperands::ExactF32 => gemm(inp, w, out, backend),
                GemmOperands::Bf16 => {
                    if inp.dtype != DType::BF16 {
                        return Err(format!(
                            "EmbedGemma2Audio::forward: a bf16 GEMM was handed an {:?} operand",
                            inp.dtype
                        ));
                    }
                    GemmOperands::Bf16.nn(inp, w, out)
                }
            }
        };
        let gc = Clamp::symmetric(cfg.gradient_clip);
        let none = Clamp::NONE;
        let mels = cfg.mel_bins();
        let (c0, c1) = (cfg.sub_channels[0], cfg.sub_channels[1]);
        let mels1 = conv_out(mels as usize) as u32;

        let mask = self.stage_mask.load(Ordering::Acquire);
        let on = |s: AudioStage| mask & s.bit() != 0;

        // Subsampling.
        let mid_t = a.mid.try_view(&[mid, (mels1 * c0) as usize], 0)?;
        if on(AudioStage::SubConv0) {
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
        }
        let sub_t = at(&a.sub, cfg.proj_in())?;
        if on(AudioStage::SubConv1) {
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
        }
        if on(AudioStage::InputProj) {
            match &a.bf {
                None => mm(&sub_t, &self.input_proj, &x)?,
                // The convolution stage's output, rounded for the GEMM: one
                // pass for the forward.
                Some(bf) => {
                    let op = at(&bf.n, cfg.proj_in())?;
                    cast_f32_to_bf16_into(&sub_t, &op)?;
                    mm(&op, &self.input_proj, &x)?;
                }
            }
        }
        if let Some(t) = io.trace.first() {
            gpu_copy(&x, t)?;
        }

        // The feed-forward between its norms: the two linears and the clamp,
        // SiLU and clamp between them.
        let ffn_mid = |f: &Ffn| -> Result<(), String> {
            if on(AudioStage::FfnUp) {
                mm(&n, &f.l1.w, &big)?;
            }
            if on(AudioStage::FfnAct) {
                ew_store(
                    rt,
                    &big.buffer,
                    &act_buf.buffer,
                    rows_u * 4 * h,
                    &Ew {
                        pre: f.l1.bounds.output,
                        silu: true,
                        post: f.l2.bounds.input,
                    },
                    store,
                )?;
            }
            if on(AudioStage::FfnDown) {
                mm(&act_buf, &f.l2.w, &y)?;
            }
            Ok(())
        };

        const NO_SPLIT: &str = "EmbedGemma2Audio: a split q/k/v projection without its activations";
        // k's and v's clamped inputs of a split projection.
        let split_inputs = ck.as_ref().zip(cv.as_ref());

        for (li, l) in self.layers.iter().enumerate() {
            if on(AudioStage::FfnNormPre) {
                rms_norm_store(
                    rt,
                    &x.buffer,
                    &n.buffer,
                    rows_u,
                    h,
                    &ffn_pre_norm(&l.ff1, eps, gc),
                    store,
                )?;
            }
            ffn_mid(&l.ff1)?;

            // The first feed-forward's post-norm, then the attention's norm into
            // the clamped input(s) q, k and v read: one when their clamps are
            // equal, else one each.
            let fused = matches!(l.qkv.w, QkvWeights::Fused(_));
            if on(AudioStage::FfnNormPost) {
                let mut attn_norm = RmsNorm::new(eps, Some(&l.norm_pre_attn));
                attn_norm.pre = [gc, none];
                let mut outs = vec![(&cq.buffer, l.qkv.bounds[0].input)];
                if !fused {
                    let (ck, cv) = split_inputs.ok_or(NO_SPLIT)?;
                    outs.push((&ck.buffer, l.qkv.bounds[1].input));
                    outs.push((&cv.buffer, l.qkv.bounds[2].input));
                }
                rms_norm_then_store(
                    rt,
                    &y.buffer,
                    &x.buffer,
                    rows_u,
                    h,
                    &ffn_post_norm(&l.ff1, eps, gc, cfg.residual_weight, &x.buffer),
                    &NormThen { spec: attn_norm, outs },
                    [store; 3],
                )?;
            }

            // Attention.
            if on(AudioStage::QkvGemm) {
                match (&l.qkv.w, &split) {
                    (QkvWeights::Fused(w), _) => mm(&cq, w, &qkv_fused)?,
                    (QkvWeights::Split(ws), Some(outs)) => {
                        let (ck, cv) = split_inputs.ok_or(NO_SPLIT)?;
                        for ((src, w), dst) in [&cq, ck, cv].into_iter().zip(ws).zip(outs) {
                            mm(src, w, dst)?;
                        }
                    }
                    (QkvWeights::Split(_), None) => return Err(NO_SPLIT.into()),
                }
            }
            if on(AudioStage::Attention) {
                // The fused projection's output is read three times, at q's, k's
                // and v's columns of its rows.
                let (q, k, v, stride, offs) = match &split {
                    Some([q, k, v]) if !fused => (&q.buffer, &k.buffer, &v.buffer, h, [0, 0, 0]),
                    _ => (
                        &qkv_fused.buffer,
                        &qkv_fused.buffer,
                        &qkv_fused.buffer,
                        3 * h,
                        [0, h, 2 * h],
                    ),
                };
                attention_store(
                    rt,
                    q,
                    k,
                    v,
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
                        q: l.qkv.bounds[0].output,
                        k: l.qkv.bounds[1].output,
                        v: l.qkv.bounds[2].output,
                        out: l.post.bounds.input,
                        stride,
                        q_off: offs[0],
                        k_off: offs[1],
                        v_off: offs[2],
                    },
                    store,
                )?;
            }
            if on(AudioStage::AttnOutGemm) {
                mm(&n, &l.post.w, &y)?;
            }

            // The attention's post-norm, then the light conv's pre-norm.
            if on(AudioStage::AttnNormPost) {
                let mut post = RmsNorm::new(eps, Some(&l.norm_post_attn));
                post.pre = [l.post.bounds.output, gc];
                post.resid = Some(&x.buffer);
                let mut lc_norm = RmsNorm::new(eps, Some(&l.lc_pre_norm));
                lc_norm.pre = [none, none];
                rms_norm_then_store(
                    rt,
                    &y.buffer,
                    &x.buffer,
                    rows_u,
                    h,
                    &post,
                    &NormThen {
                        spec: lc_norm,
                        outs: vec![(&n.buffer, l.lc_start.bounds.input)],
                    },
                    [store; 3],
                )?;
            }

            // Light conv.
            if on(AudioStage::LconvStart) {
                mm(&n, &l.lc_start.w, &big2)?;
            }
            if on(AudioStage::Lconv) {
                lconv_store(
                    rt,
                    &big2.buffer,
                    &l.lc_conv,
                    &n.buffer,
                    &io.clips,
                    h,
                    cfg.conv_kernel,
                    &Lconv {
                        eps,
                        norm_weight: &l.lc_conv_norm,
                        glu: l.lc_start.bounds.output,
                        pre: gc,
                        post: l.lc_end.bounds.input,
                    },
                    store,
                )?;
            }
            if on(AudioStage::LconvEnd) {
                mm(&n, &l.lc_end.w, &y)?;
            }

            // `linear_end`'s output clamp and the residual add, then the second
            // feed-forward's pre-norm.
            if on(AudioStage::LconvResid) {
                let mut add = RmsNorm::new(eps, None);
                add.norm = false;
                add.pre = [l.lc_end.bounds.output, none];
                add.resid = Some(&x.buffer);
                rms_norm_then_store(
                    rt,
                    &y.buffer,
                    &x.buffer,
                    rows_u,
                    h,
                    &add,
                    &NormThen {
                        spec: RmsNorm {
                            pre: [gc, none],
                            ..RmsNorm::new(eps, Some(&l.ff2.pre_norm))
                        },
                        outs: vec![(&n.buffer, l.ff2.l1.bounds.input)],
                    },
                    [store; 3],
                )?;
            }

            ffn_mid(&l.ff2)?;
            // The second feed-forward's post-norm, then `norm_out`, in place.
            if on(AudioStage::FfnNormPost) {
                let mut norm_out = RmsNorm::new(eps, Some(&l.norm_out));
                norm_out.pre = [gc, none];
                rms_norm_then(
                    rt,
                    &y.buffer,
                    &x.buffer,
                    rows_u,
                    h,
                    &ffn_post_norm(&l.ff2, eps, gc, cfg.residual_weight, &x.buffer),
                    &NormThen {
                        spec: norm_out,
                        outs: vec![(&x.buffer, none)],
                    },
                )?;
            }
            if let Some(t) = io.trace.get(1 + li) {
                gpu_copy(&x, t)?;
            }
        }

        // Output projection (with bias) and the multimodal embedder.
        let tower = at(&a.tower, cfg.out_dims)?;
        if on(AudioStage::OutProj) {
            match &a.bf {
                None => mm(&x, &self.output_proj, &tower)?,
                // The residual stream, rounded for the GEMM: the last
                // layer's `norm_out` is the one GEMM input that is also the
                // next layer's f32 input, so it is cast once, here.
                Some(bf) => {
                    let op = at(&bf.n, h)?;
                    cast_f32_to_bf16_into(&x, &op)?;
                    mm(&op, &self.output_proj, &tower)?;
                }
            }
        }
        let emb = emb_wide;
        if on(AudioStage::OutBias) {
            // The bias in place on `tower` (the tower's output, traced below),
            // then the embedder's weightless norm of it.
            let mut bias = RmsNorm::new(eps, None);
            bias.norm = false;
            bias.bias = Some(&self.output_bias);
            rms_norm_then_store(
                rt,
                &tower.buffer,
                &tower.buffer,
                rows_u,
                cfg.out_dims,
                &bias,
                &NormThen {
                    spec: RmsNorm::new(eps, None),
                    outs: vec![(&emb.buffer, none)],
                },
                [store; 3],
            )?;
        }
        if let Some(t) = io.trace.last() {
            gpu_copy(&tower, t)?;
        }
        if on(AudioStage::EmbedProj) {
            mm(&emb, &self.embed_proj, &io.soft)?;
        }
        Ok(())
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

/// The maximal runs of consecutive `true` entries of a tower mask, as row
/// ranges: the pieces of a clip's tower output that are its soft tokens. A
/// processor's mask is a prefix (one run, or none); any other pattern is
/// defined and gives one run per stretch.
fn kept_runs(omask: &[bool]) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut t = 0usize;
    while t < omask.len() {
        if !omask[t] {
            t += 1;
            continue;
        }
        let start = t;
        while t < omask.len() && omask[t] {
            t += 1;
        }
        runs.push(start..t);
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clip's soft rows are the maximal runs of its tower mask, so a hole in
    /// the mask is two segments and the rows between them are never read.
    #[test]
    fn kept_rows_are_the_masks_maximal_runs() {
        let runs = |m: &[bool]| -> Vec<(usize, usize)> { kept_runs(m).into_iter().map(|r| (r.start, r.end)).collect() };
        assert!(runs(&[]).is_empty());
        assert!(runs(&[false, false]).is_empty(), "nothing kept");
        assert_eq!(runs(&[true; 5]), [(0, 5)], "a full clip is one run");
        assert_eq!(runs(&[true, true, false, false]), [(0, 2)], "a processor's prefix");
        assert_eq!(
            runs(&[true, false, true, true, false, true]),
            [(0, 1), (2, 4), (5, 6)],
            "a hole splits the run"
        );
        // The tower mask is `mask[::4]`: frames 8..16 masked drop rows 2 and 3.
        let mask: Vec<bool> = (0..30).map(|f| !(8..16).contains(&f)).collect();
        let om = out_mask(&mask);
        assert_eq!(om.len(), 8);
        assert_eq!(runs(&om), [(0, 2), (4, 8)]);
    }

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
            (
                "\"attention_context_left\":13",
                "\"attention_context_left\":18",
                "more past frames than the kernel holds",
            ),
            (
                "\"attention_invalid_logits_value\":-1000000000.0",
                "\"attention_invalid_logits_value\":-100.0",
                "a masked logit that is not far enough below",
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
