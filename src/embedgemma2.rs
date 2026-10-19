//! EmbeddingGemma 2's text encoder (`google/embeddinggemma-2`), composed from
//! tessl kernels and loaded straight from its `model.safetensors`.
//!
//! The text path is a 24-layer bidirectional encoder over a Gemma-4-style
//! block, then a mean pool and an L2 normalize: what sentence-transformers'
//! `Transformer -> Pooling(mean, include_prompt) -> Normalize` computes for
//! this checkpoint. This model does not load the vision or audio towers in
//! the same file ([`crate::embedgemma2_vision`] holds the vision tower); their
//! soft tokens enter through [`EmbedGemma2Model::encode_multimodal`] (see
//! "Multimodal input" below).
//!
//! # Layer (transformers 5.19 `EmbeddingGemma2EncoderLayer`)
//!
//! ```text
//! x      = rms_norm(resid) * input_layernorm.w           (* w, not * (1 + w))
//! q,k,v  = x @ [Wq | Wk | Wv]
//! q, k   = rope(rms_norm(q) * q_norm.w), rope(rms_norm(k) * k_norm.w)   per head
//! v      = rms_norm(v)                                    per head, no weight
//! o      = softmax(q kᵀ * 1.0 + mask) v                   bidirectional
//! resid += rms_norm(o @ Wo) * post_attention_layernorm.w
//! x      = rms_norm(resid) * pre_feedforward_layernorm.w
//! resid += rms_norm(down(gelu_tanh(gate x) * up x)) * post_feedforward_layernorm.w
//! resid  = layer_scalar * (resid + rms_norm(proj(gelu_tanh(gate_ple resid) * ple_i)) * w)
//! ```
//!
//! with `ple_i = rms_norm(embeds @ Wple[i] * hidden^-0.5) * ple_norm.w` (the
//! projection-only per-layer input), `embeds = embed_tokens[ids] *
//! sqrt(hidden)`. The sliding layers see keys with `|i - j| <= window`, the
//! full layers every key; both see only the keys of the row's own sequence.
//! After the last layer: `rms_norm(resid) * norm.w`, the mean over each
//! sequence's tokens, `@ embedding_projection` (512 -> 768), and an L2
//! normalize (over a Matryoshka prefix when `encode` is given `truncate_dim`).
//! The projection is linear, so pooling before it is the same function as
//! transformers' project-then-pool, at 1/T the GEMM.
//!
//! # Precision
//!
//! [`EmbedGemma2Model::load`] is [`GemmOperands::ExactF32`]: weights are
//! widened from the checkpoint's bf16 to f32 exactly (the embedding table
//! stays bf16 and its gather widens exactly), every activation is f32 and every
//! GEMM is exact f32: transformers' fp32 forward up to operation order, which
//! is what `tests/embedgemma2_model.rs` holds it to. It needs the runtime's
//! relaxed-f32 GEMM switched off (the default).
//!
//! [`EmbedGemma2Model::load_with`] with [`GemmOperands::Bf16`] keeps every
//! projection as the checkpoint stores it, bf16, on the device (no f32 copy),
//! and runs every GEMM with bf16 operands and f32 accumulation. The inputs of
//! the GEMMs are rounded to bf16 where they are produced when a kernel can
//! write bf16 (the norms, the attention, the feed-forward residual add, the
//! GELU gates) and by a cast pass where one cannot (the embeddings once per
//! forward, the pooled vector); the residual stream,
//! the norms, the attention, the pooling and the outputs of every GEMM stay f32.
//! The embedding table is bf16 under both modes, so it does not change. No f32
//! GEMM runs, so the runtime's relaxed-f32 switch does not matter and `encode`
//! does not refuse it. `tests/embedgemma2_model.rs` holds this forward to
//! transformers' bf16 forward the way `tests/qwen35_model.rs` holds Qwen3.5's.
//!
//! # Multimodal input
//!
//! `EmbeddingGemma2Model.forward` embeds the ids with every image, video and
//! audio placeholder id (`image_token_id`, `video_token_id`, `audio_token_id`)
//! replaced by `pad_token_id`, so the placeholder rows are the pad row times
//! `sqrt(hidden)`, then `masked_scatter`s the towers' soft tokens over the
//! placeholder rows, in order and unscaled, and runs the text model on the
//! result. [`EmbedGemma2Model::encode_multimodal`] is that entry with the soft
//! tokens given: the scatter lands in the activation the per-layer inputs and
//! the residual stream both read, so the rest of the forward is unchanged.
//! [`EmbedGemma2Model::encode`] is the case with no soft rows and no
//! placeholder ids (a text id equal to a placeholder id is an ordinary token
//! there, as before). [`EmbedGemma2Config::image_sequence`],
//! [`EmbedGemma2Config::audio_sequence`] and
//! [`EmbedGemma2Config::video_sequence`] build the processor's id layout.
//!
//! # Kernels
//!
//! New for this model: [`encoder_attn`] (`kernels/encoder_attn.metal`: the
//! TensorOps tiled kernel of `kernels/attn_tiled.h` (head dims 32 to 512; bf16 operand tiles at 64, 256 and 512),
//! or one simdgroup per query row; [`encoder_attn_bf16_out`] stores bf16),
//! [`segment_mean_rows`] and [`l2_normalize_rows`] (`kernels/embed_pool.metal`),
//! and the fused row passes of `kernels/embedgemma2_text.metal`
//! ([`resid_add_norm_f32`] and [`resid_add_norm_bf16`]: a post-norm residual
//! add with the layer scalar and the norm after it, in one launch;
//! [`resid_add_bf16_copy`]: the add and its rounded bf16 copy;
//! [`scale_rms_norm_inplace`]: the per-layer inputs' scale and norm), each the
//! composition of the tessl kernels it replaced, bit for bit. The rest are
//! tessl's: GEMMs, exact f32 or bf16 (NN, and NT for the per-layer-input
//! projection), `nn::rms_norm_f32` (`nn::rms_norm_bf16` into a bf16 GEMM's
//! input: layer 0's input norm, the only one that stands alone),
//! `nn::rms_norm_residual_add_f32` (the exact-f32 feed-forward add),
//! `qwen35::embed_rows_scaled` (the
//! gather and the `sqrt(hidden)` scale), `qwen35::gated_act` (GELU-tanh over
//! column windows), `qwen35::attn_qk_norm_rope_columns` (`* w` norms,
//! full-width RoPE, and the weightless V norm) and `qwen35_bwd::copy_cols`
//! (one flat copy per run of consecutive placeholder rows, for the soft
//! tokens; no new kernel).

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::config_json;
use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_gpu_buf, set_u32};
use crate::embedgemma2_tok::Tokenizer;
use crate::gemm::{cast_f32_to_bf16_into, gemm, gemm_nt_f32, gemm_small_m, GemmBackend, GemmOperands};
use crate::json::Json;
use crate::loader::Loader;
use crate::nn::{
    dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes, rms_norm_bf16, rms_norm_f32,
    rms_norm_residual_add_f32,
};
use crate::qwen35::{self, AttnShape, AttnTargets, Cols, GatedAct, LmHead, OutCols, QkvColumns};
use crate::qwen35_bwd::copy_cols;
use crate::qwen35_model::Precision;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{gpu_copy, DType, GpuBuffer, Tensor};

const BACKEND: GemmBackend = GemmBackend::TensorOps;

/// Tokens per sequence [`EmbedGemma2Model::encode`] accepts unless raised
/// with [`EmbedGemma2Model::set_max_tokens`]: the model card's context.
pub const DEFAULT_MAX_TOKENS: u32 = 8192;
/// Sequences per [`EmbedGemma2Model::encode`] call.
pub const MAX_BATCH: usize = 256;
/// Padded tokens (`sequences * longest`) per forward. Bounds the activations:
/// at this size the largest buffer, the q|k|v projection of a 512-wide full
/// layer (`(4 + 2) * 512` columns), is 384 MiB, and all of them about 2 GiB.
/// [`EmbedGemma2Model::encode`] splits a batch into as many forwards as
/// this needs, so it bounds memory, not what a caller may pass.
pub const MAX_BATCH_TOKENS: usize = 32_768;
/// The largest frame time, in seconds, that
/// [`EmbedGemma2Config::video_sequence_with_timestamps`] labels (about 31
/// years): past it the `mm:ss` text is a minutes count no video has, and a
/// label that long is a caller's error, not a layout.
pub const MAX_TIMESTAMP_SECONDS: f64 = 1.0e9;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Shape and masking rule of one [`encoder_attn`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderAttnDims {
    pub batch: u32,
    /// Padded sequence length: every row of Q/K/V/O holds `seq` positions.
    pub seq: u32,
    pub heads: u32,
    pub heads_kv: u32,
    /// 32, 64, 256 or 512 (the instantiated head dims).
    pub head_dim: u32,
    /// `0`: every key of the row's sequence. Otherwise keys with
    /// `|t_k - t_q| <= window` (inclusive, both sides).
    pub window: u32,
    /// Multiplies `q · k` (EmbeddingGemma 2 uses 1.0).
    pub scale: f32,
}

fn encoder_attn_entry(head_dim: u32) -> Option<(&'static str, usize, usize)> {
    // (entry, lanes per row, simdgroups per threadgroup) as compiled.
    match head_dim {
        32 => Some(("encoder_attn_rows_h32_r2_g8", 2, 8)),
        64 => Some(("encoder_attn_rows_h64_r4_g16", 4, 16)),
        256 => Some(("encoder_attn_rows_h256_r16_g32", 16, 32)),
        512 => Some(("encoder_attn_rows_h512_r32_g32", 32, 32)),
        _ => None,
    }
}

/// Which kernel runs [`encoder_attn`]'s contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderAttnKernel {
    /// One simdgroup per query row, scalar f32 (`encoder_attn_rows_*`): every
    /// head dim.
    Rows,
    /// FlashAttention-2 on the TensorOps matrix units
    /// (`encoder_attn_tiled_*`, `kernels/attn_tiled.h`'s body in its encoder
    /// mode, the one `kernels/qwen35_attn_tiled.metal` runs causally): head
    /// dims 32, 64, 256 and 512, f32 operands, on a device with TensorOps.
    Tiled,
    /// [`Self::Tiled`] with bf16 operand tiles (`encoder_attn_tiled_bf16_*`):
    /// Q, K and V are bf16 buffers, and so is the probability tile that
    /// multiplies V; scores, the online softmax, the accumulator and the
    /// output stay f32. Head dims 64, 256 and 512. Never picked by
    /// [`encoder_attn_kernel`]: the caller owns the operand dtype
    /// ([`encoder_attn_bf16`]).
    TiledBf16,
    /// [`Self::TiledBf16`] storing the output as bf16 as well
    /// (`encoder_attn_tiled_bf16o_*`, [`encoder_attn_bf16_out`]): the kernel
    /// rounds the normalized accumulator to nearest even at the store, the
    /// value a cast pass after [`Self::TiledBf16`] makes. Never picked by
    /// [`encoder_attn_kernel`].
    TiledBf16Out,
}

/// The kernel [`encoder_attn`] picks: [`EncoderAttnKernel::Tiled`] where it
/// is compiled and the device has the matrix units, else the rows kernel.
pub fn encoder_attn_kernel(rt: &GpuRuntime, head_dim: u32) -> EncoderAttnKernel {
    if rt.has_tensorops() && encoder_attn_tiled_entry(head_dim).is_some() {
        EncoderAttnKernel::Tiled
    } else {
        EncoderAttnKernel::Rows
    }
}

/// (entry, queries per threadgroup, simdgroups per threadgroup) as compiled;
/// the entry spells its geometry. Each head dim's geometry is the winner of
/// the sweep in `bench/results/embedgemma2_attention_m5pro.txt`.
fn encoder_attn_tiled_entry(head_dim: u32) -> Option<(&'static str, usize, usize)> {
    match head_dim {
        32 => Some(("encoder_attn_tiled_h32_q16_k32_sg1", 16, 1)),
        64 => Some(("encoder_attn_tiled_h64_q32_k32_sg2", 32, 2)),
        256 => Some(("encoder_attn_tiled_h256_q32_k32_sg4", 32, 4)),
        512 => Some(("encoder_attn_tiled_h512_q32_k32_sg8", 32, 8)),
        _ => None,
    }
}

/// [`encoder_attn_tiled_entry`] for [`EncoderAttnKernel::TiledBf16`].
fn encoder_attn_tiled_bf16_entry(head_dim: u32) -> Option<(&'static str, usize, usize)> {
    match head_dim {
        64 => Some(("encoder_attn_tiled_bf16_h64_q32_k32_sg2", 32, 2)),
        256 => Some(("encoder_attn_tiled_bf16_h256_q32_k64_sg4", 32, 4)),
        512 => Some(("encoder_attn_tiled_bf16_h512_q32_k128_sg8", 32, 8)),
        _ => None,
    }
}

/// [`encoder_attn_tiled_bf16_entry`] for [`EncoderAttnKernel::TiledBf16Out`].
fn encoder_attn_tiled_bf16o_entry(head_dim: u32) -> Option<(&'static str, usize, usize)> {
    match head_dim {
        64 => Some(("encoder_attn_tiled_bf16o_h64_q32_k32_sg2", 32, 2)),
        256 => Some(("encoder_attn_tiled_bf16o_h256_q32_k64_sg4", 32, 4)),
        512 => Some(("encoder_attn_tiled_bf16o_h512_q32_k128_sg8", 32, 8)),
        _ => None,
    }
}

/// Whether [`encoder_attn_bf16`] has a kernel for `head_dim` (64, 256 or 512),
/// so a model that runs its attention in bf16 can refuse a config at load
/// rather than at its first forward.
pub fn encoder_attn_bf16_supports(head_dim: u32) -> bool {
    encoder_attn_tiled_bf16_entry(head_dim).is_some()
}

/// `out = a @ w` at `operands`' precision, the one GEMM entry every
/// EmbeddingGemma 2 forward (the text encoder's, the vision tower's) calls.
/// `a` is in the mode's activation dtype (f32, or bf16 under
/// [`GemmOperands::Bf16`], as the forward stages it), so neither lane rounds
/// or allocates anything of its own here. The exact lane does not check the
/// runtime's relaxed-f32 switch: the model's `encode` does, once per call.
pub(crate) fn project(operands: GemmOperands, a: &Tensor, w: &Tensor, out: &Tensor) -> Result<(), String> {
    match operands {
        GemmOperands::ExactF32 => gemm(a, w, out, BACKEND),
        GemmOperands::Bf16 => GemmOperands::Bf16.nn(a, w, out),
    }
}

/// Bidirectional attention over right-padded sequences, on the kernel
/// [`encoder_attn_kernel`] picks ([`encoder_attn_with`] names one).
///
/// `q`/`o`: `[batch, seq, heads, head_dim]`; `k`/`v`: `[batch, capacity,
/// heads_kv, head_dim]` with `capacity >= seq` derived from their size
/// ([`crate::nn::attn_kv_capacity`], as the K/V writer derives it), all f32
/// (the bf16-operand forms are [`encoder_attn_bf16`] and
/// [`encoder_attn_bf16_out`]). `lens`: `[batch]` u32,
/// each sequence's live length. Keys at or past a row's length are masked,
/// and every query at or past it is written as zeros. The lengths are device
/// values the host cannot see; the kernel clamps each to `seq`, and a length
/// of 0 gives an all-zero row. Callers that know the lengths should check
/// them (as [`EmbedGemma2Model::encode`] does).
#[allow(clippy::too_many_arguments)]
pub fn encoder_attn(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lens: &GpuBuffer,
    dims: EncoderAttnDims,
) -> Result<(), String> {
    let kernel = encoder_attn_kernel(rt, dims.head_dim);
    encoder_attn_with(rt, q, k, v, o, lens, dims, kernel)
}

/// [`encoder_attn`] with bf16 operand tiles ([`EncoderAttnKernel::TiledBf16`]):
/// `q`, `k` and `v` are bf16 buffers (same shapes, `k`/`v` capacity derived
/// by [`crate::nn::attn_kv_capacity_bf16`]), `o` stays f32. Needs a TensorOps
/// device and head dim 64, 256 or 512; anything else is an error, never a
/// silent fall back to a different precision.
#[allow(clippy::too_many_arguments)]
pub fn encoder_attn_bf16(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lens: &GpuBuffer,
    dims: EncoderAttnDims,
) -> Result<(), String> {
    encoder_attn_with(rt, q, k, v, o, lens, dims, EncoderAttnKernel::TiledBf16)
}

/// [`encoder_attn_bf16`] with `o` a bf16 buffer too
/// ([`EncoderAttnKernel::TiledBf16Out`]): the output is the f32 kernel's
/// accumulator rounded to nearest even as it is stored, bit for bit what
/// [`encoder_attn_bf16`] followed by a cast to bf16 gives. The bf16 forward
/// feeds it straight to the output projection.
#[allow(clippy::too_many_arguments)]
pub fn encoder_attn_bf16_out(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lens: &GpuBuffer,
    dims: EncoderAttnDims,
) -> Result<(), String> {
    encoder_attn_with(rt, q, k, v, o, lens, dims, EncoderAttnKernel::TiledBf16Out)
}

/// [`encoder_attn`] on a named kernel (for the tests and the paired
/// benchmark that compare them). The contract and the checks are the same,
/// except that [`EncoderAttnKernel::TiledBf16`] takes bf16 `q`, `k` and `v`,
/// and [`EncoderAttnKernel::TiledBf16Out`] a bf16 `o` as well.
#[allow(clippy::too_many_arguments)]
pub fn encoder_attn_with(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lens: &GpuBuffer,
    dims: EncoderAttnDims,
    kernel: EncoderAttnKernel,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::encoder_attn";
    let (entry, lanes, groups) = encoder_attn_entry(dims.head_dim)
        .ok_or_else(|| format!("{WHAT}: head dim {} has no kernel (32, 64, 256 or 512)", dims.head_dim))?;
    if dims.heads == 0 || dims.heads_kv == 0 || dims.heads % dims.heads_kv != 0 {
        return Err(format!(
            "{WHAT}: heads ({}) must be a non-zero multiple of heads_kv ({})",
            dims.heads, dims.heads_kv
        ));
    }
    if !dims.scale.is_finite() {
        return Err(format!("{WHAT}: scale must be finite"));
    }
    let q_elems = elems(&[dims.batch, dims.seq, dims.heads, dims.head_dim], WHAT)?;
    let bf16_out = kernel == EncoderAttnKernel::TiledBf16Out;
    let bf16 = bf16_out || kernel == EncoderAttnKernel::TiledBf16;
    if bf16 {
        require::<u16>(rt, q, q_elems, "encoder_attn q (bf16)")?;
    } else {
        require::<f32>(rt, q, q_elems, "encoder_attn q")?;
    }
    if bf16_out {
        require::<u16>(rt, o, q_elems, "encoder_attn o (bf16)")?;
    } else {
        require::<f32>(rt, o, q_elems, "encoder_attn o")?;
    }
    require::<u32>(rt, lens, dims.batch as usize, "encoder_attn lens")?;
    if q_elems == 0 {
        return Ok(());
    }
    let kv_capacity = if bf16 {
        crate::nn::attn_kv_capacity_bf16(k, v, dims.batch, dims.heads_kv, dims.head_dim)?
    } else {
        crate::nn::attn_kv_capacity(k, v, dims.batch, dims.heads_kv, dims.head_dim)?
    };
    if kv_capacity < dims.seq {
        return Err(format!(
            "{WHAT}: K/V hold {kv_capacity} positions per sequence; need at least seq = {}",
            dims.seq
        ));
    }
    require_disjoint_writes(WHAT, &[("o", o)], &[("q", q), ("k", k), ("v", v), ("lens", lens)])?;
    if kernel != EncoderAttnKernel::Rows {
        let (entry, bq, sg) = if bf16 {
            let table = if bf16_out {
                encoder_attn_tiled_bf16o_entry
            } else {
                encoder_attn_tiled_bf16_entry
            };
            table(dims.head_dim).ok_or_else(|| {
                format!(
                    "{WHAT}: the bf16 tiled kernel is compiled for head dims 64, 256 and 512, not {}",
                    dims.head_dim
                )
            })?
        } else {
            encoder_attn_tiled_entry(dims.head_dim).ok_or_else(|| {
                format!(
                    "{WHAT}: the tiled kernel is compiled for head dims 32, 64, 256 and 512, not {}",
                    dims.head_dim
                )
            })?
        };
        if !rt.has_tensorops() {
            return Err(format!(
                "{WHAT}: the tiled kernel needs TensorOps, which this device lacks"
            ));
        }
        // MPP tensor views index one (batch, head) plane with i32 extents
        // and strides, so a plane's last element must be addressable in i32.
        let q_plane = u64::from(dims.seq) * u64::from(dims.heads) * u64::from(dims.head_dim);
        let kv_plane = u64::from(kv_capacity) * u64::from(dims.heads_kv) * u64::from(dims.head_dim);
        if q_plane > i32::MAX as u64 || kv_plane > i32::MAX as u64 {
            return Err(format!(
                "{WHAT}: a head's plane exceeds i32 indexing (seq = {}, kv capacity = {kv_capacity})",
                dims.seq
            ));
        }
        let threads = sg * 32;
        let p = rt.pipeline(entry)?;
        let max = p.maxTotalThreadsPerThreadgroup();
        if max < threads {
            return Err(format!(
                "{WHAT}: {entry} needs {threads} threads per threadgroup, the pipeline allows {max}"
            ));
        }
        let groups_y = elems(&[dims.batch, dims.heads], WHAT)?;
        return dispatch_2d_tg(rt, &p, (dims.seq as usize).div_ceil(bq), groups_y, threads, |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, k, 1);
            set_gpu_buf(bnd, v, 2);
            set_gpu_buf(bnd, o, 3);
            set_u32(bnd, dims.seq, 4);
            set_gpu_buf(bnd, lens, 5);
            set_u32(bnd, dims.heads, 6);
            set_u32(bnd, dims.heads_kv, 7);
            set_u32(bnd, dims.window, 8);
            crate::dispatch::set_f32(bnd, dims.scale, 9);
            set_u32(bnd, kv_capacity, 10);
        });
    }
    let rows_per_tg = groups * (32 / lanes);
    let groups_x = (dims.seq as usize).div_ceil(rows_per_tg);
    let groups_y = elems(&[dims.batch, dims.heads], WHAT)?;
    let p = rt.pipeline(entry)?;
    dispatch_2d_tg(rt, &p, groups_x, groups_y, groups * 32, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_u32(bnd, dims.seq, 4);
        set_gpu_buf(bnd, lens, 5);
        set_u32(bnd, dims.heads, 6);
        set_u32(bnd, dims.heads_kv, 7);
        set_u32(bnd, dims.window, 8);
        crate::dispatch::set_f32(bnd, dims.scale, 9);
        set_u32(bnd, kv_capacity, 10);
    })
}

/// `[start, end)` row ranges for [`segment_mean_rows`], checked against
/// `rows` and uploaded: `[n, 2]` u32.
pub fn upload_segments(rt: &Arc<GpuRuntime>, segments: &[(u32, u32)], rows: u32) -> Result<GpuBuffer, String> {
    crate::encode_batch::upload_segments(rt, "embedgemma2", segments, rows)
}

/// `out[s] = mean(x[start_s..end_s])` over `x: [rows, dim]` f32, into `out:
/// [n_segments, dim]`, with `segments` the `[n_segments, 2]` u32 ranges
/// [`upload_segments`] checks. They are device values: the kernel clamps
/// `end` to `rows` and writes zeros for an empty range.
pub fn segment_mean_rows(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    segments: &GpuBuffer,
    out: &GpuBuffer,
    n_segments: u32,
    rows: u32,
    dim: u32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::segment_mean_rows";
    require::<f32>(rt, x, elems(&[rows, dim], WHAT)?, "segment_mean_rows x")?;
    require::<u32>(
        rt,
        segments,
        elems(&[n_segments, 2], WHAT)?,
        "segment_mean_rows segments",
    )?;
    let n = elems(&[n_segments, dim], WHAT)?;
    require::<f32>(rt, out, n, "segment_mean_rows out")?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("out", out)], &[("x", x), ("segments", segments)])?;
    let p = rt.pipeline("segment_mean_rows_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, segments, 1);
        set_gpu_buf(bnd, out, 2);
        set_u32(bnd, n_segments, 3);
        set_u32(bnd, rows, 4);
        set_u32(bnd, dim, 5);
    })
}

/// In place, `x[r, :dim] /= max(||x[r, :dim]||, 1e-12)` for `rows` rows
/// `ld` elements apart. `dim < ld` normalizes a Matryoshka prefix (the
/// columns past it are left as they are).
pub fn l2_normalize_rows(rt: &Arc<GpuRuntime>, x: &GpuBuffer, rows: u32, dim: u32, ld: u32) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::l2_normalize_rows";
    if dim == 0 || dim > ld {
        return Err(format!("{WHAT}: dim must be in 1..=ld, got dim {dim}, ld {ld}"));
    }
    if rows == 0 {
        return Ok(());
    }
    // The last row reaches dim, not ld, elements.
    let need = (rows as usize - 1)
        .checked_mul(ld as usize)
        .and_then(|n| n.checked_add(dim as usize))
        .ok_or_else(|| format!("{WHAT}: rows * ld overflows"))?;
    require::<f32>(rt, x, need, "l2_normalize_rows x")?;
    let p = rt.pipeline("l2_normalize_rows_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_u32(bnd, rows, 1);
        set_u32(bnd, dim, 2);
        set_u32(bnd, ld, 3);
    })
}

/// Launch `entry` (one of `kernels/embedgemma2_text.metal`'s row passes) with
/// one threadgroup per row, at the width `nn::rms_norm_f32` gives a row of the
/// same `dim`: the fused passes reduce a row in the same lane order, so their
/// sums are the unfused kernels' sums (see that file's header).
fn dispatch_row_pass(
    rt: &Arc<GpuRuntime>,
    entry: &str,
    rows: u32,
    dim: u32,
    bind: impl FnOnce(&mut crate::dispatch::Binder<'_>),
) -> Result<(), String> {
    let p = rt.pipeline(entry)?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, bind)
}

/// The checks every fused row pass shares: a finite `layer_scale`, a valid
/// `eps` and `dim`, and `[rows, dim]` f32 buffers (`n` elements) for `y` and
/// `resid`.
#[allow(clippy::too_many_arguments)]
fn check_row_pass(
    what: &str,
    rt: &GpuRuntime,
    y: &GpuBuffer,
    resid: &GpuBuffer,
    n: usize,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<(), String> {
    crate::nn::validate_rms_scalars(dim, eps, what)?;
    if !layer_scale.is_finite() {
        return Err(format!("{what}: layer_scale must be finite"));
    }
    require::<f32>(rt, y, n, "fused row pass y").map_err(|e| format!("{what}: {e}"))?;
    require::<f32>(rt, resid, n, "fused row pass resid").map_err(|e| format!("{what}: {e}"))
}

/// `resid = layer_scale * (resid + rms_norm(y) * w_post)` in place, then
/// `out = rms_norm(resid) * w_next` (`[rows, dim]`, f32 `out`): the post-norm
/// residual add and the norm after it in one launch,
/// [`crate::nn::rms_norm_residual_add_f32`] then [`crate::nn::rms_norm_f32`] with
/// bit-identical results. `out` must not alias `resid` or `y`.
#[allow(clippy::too_many_arguments)]
pub fn resid_add_norm_f32(
    rt: &Arc<GpuRuntime>,
    y: &GpuBuffer,
    w_post: &GpuBuffer,
    resid: &GpuBuffer,
    w_next: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::resid_add_norm_f32";
    resid_add_norm(
        rt,
        WHAT,
        "eg2t_resid_norm_f32",
        y,
        w_post,
        resid,
        w_next,
        out,
        rows,
        dim,
        eps,
        layer_scale,
        4,
    )
}

/// [`resid_add_norm_f32`] writing `out` as bf16 (`[rows, dim]`, rounded to
/// nearest even as [`crate::nn::rms_norm_bf16`] rounds), for the next GEMM of a
/// [`GemmOperands::Bf16`] forward.
#[allow(clippy::too_many_arguments)]
pub fn resid_add_norm_bf16(
    rt: &Arc<GpuRuntime>,
    y: &GpuBuffer,
    w_post: &GpuBuffer,
    resid: &GpuBuffer,
    w_next: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::resid_add_norm_bf16";
    resid_add_norm(
        rt,
        WHAT,
        "eg2t_resid_norm_bf16",
        y,
        w_post,
        resid,
        w_next,
        out,
        rows,
        dim,
        eps,
        layer_scale,
        2,
    )
}

#[allow(clippy::too_many_arguments)]
fn resid_add_norm(
    rt: &Arc<GpuRuntime>,
    what: &str,
    entry: &str,
    y: &GpuBuffer,
    w_post: &GpuBuffer,
    resid: &GpuBuffer,
    w_next: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
    out_elem_bytes: usize,
) -> Result<(), String> {
    let n = elems(&[rows, dim], what)?;
    check_row_pass(what, rt, y, resid, n, dim, eps, layer_scale)?;
    let weight = |w: &GpuBuffer, name: &str| {
        require::<f32>(rt, w, dim as usize, "fused row pass weight").map_err(|e| format!("{what} {name}: {e}"))
    };
    weight(w_post, "w_post")?;
    weight(w_next, "w_next")?;
    // `u8`: the capacity is in bytes, which is f32's or bf16's elements times their width.
    let out_bytes = n
        .checked_mul(out_elem_bytes)
        .ok_or_else(|| format!("{what}: out size overflows usize"))?;
    require::<u8>(rt, out, out_bytes, "fused row pass out").map_err(|e| format!("{what}: {e}"))?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        what,
        &[("resid", resid), ("out", out)],
        &[("y", y), ("w_post", w_post), ("w_next", w_next)],
    )?;
    dispatch_row_pass(rt, entry, rows, dim, |bnd| {
        set_gpu_buf(bnd, y, 0);
        set_gpu_buf(bnd, w_post, 1);
        set_gpu_buf(bnd, resid, 2);
        set_gpu_buf(bnd, w_next, 3);
        set_gpu_buf(bnd, out, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, dim, 6);
        crate::dispatch::set_f32(bnd, eps, 7);
        crate::dispatch::set_f32(bnd, layer_scale, 8);
    })
}

/// `resid = layer_scale * (resid + rms_norm(y) * w_post)` in place and
/// `copy = bf16(resid)` (`[rows, dim]`, rounded to nearest even):
/// [`crate::nn::rms_norm_residual_add_f32`] then a cast to bf16 in one launch,
/// bit-identical to the pair. The bf16 forward's feed-forward residual add, whose
/// result its per-layer-input gate reads rounded.
#[allow(clippy::too_many_arguments)]
pub fn resid_add_bf16_copy(
    rt: &Arc<GpuRuntime>,
    y: &GpuBuffer,
    w_post: &GpuBuffer,
    resid: &GpuBuffer,
    copy: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::resid_add_bf16_copy";
    let n = elems(&[rows, dim], WHAT)?;
    check_row_pass(WHAT, rt, y, resid, n, dim, eps, layer_scale)?;
    require::<f32>(rt, w_post, dim as usize, "resid_add_bf16_copy w_post")?;
    require::<u16>(rt, copy, n, "resid_add_bf16_copy copy")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("resid", resid), ("copy", copy)],
        &[("y", y), ("w_post", w_post)],
    )?;
    dispatch_row_pass(rt, "eg2t_resid_add_bf16_copy", rows, dim, |bnd| {
        set_gpu_buf(bnd, y, 0);
        set_gpu_buf(bnd, w_post, 1);
        set_gpu_buf(bnd, resid, 2);
        set_gpu_buf(bnd, copy, 3);
        set_u32(bnd, rows, 4);
        set_u32(bnd, dim, 5);
        crate::dispatch::set_f32(bnd, eps, 6);
        crate::dispatch::set_f32(bnd, layer_scale, 7);
    })
}

/// In place over `[rows, dim]`: `x = rms_norm(x * scale) * w`
/// ([`crate::nn::scale_f32_inplace`] then [`crate::nn::rms_norm_f32`] with
/// bit-identical results): the per-layer inputs' scale and norm in one launch.
pub fn scale_rms_norm_inplace(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    scale: f32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2::scale_rms_norm_inplace";
    crate::nn::validate_rms_scalars(dim, eps, WHAT)?;
    if !scale.is_finite() {
        return Err(format!("{WHAT}: scale must be finite"));
    }
    let n = elems(&[rows, dim], WHAT)?;
    require::<f32>(rt, x, n, "scale_rms_norm_inplace x")?;
    require::<f32>(rt, w, dim as usize, "scale_rms_norm_inplace w")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("x", x)], &[("w", w)])?;
    dispatch_row_pass(rt, "eg2t_scale_rms_norm_f32", rows, dim, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_u32(bnd, rows, 2);
        set_u32(bnd, dim, 3);
        crate::dispatch::set_f32(bnd, eps, 4);
        crate::dispatch::set_f32(bnd, scale, 5);
    })
}

fn elems(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| format!("{what}: element count {dims:?} overflows usize"))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// One layer's attention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerSpec {
    /// `true`: the symmetric sliding window; `false`: every key.
    pub sliding: bool,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub rope_theta: f32,
}

/// The text encoder's shapes, read from `config.json` (`text_config`).
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedGemma2Config {
    pub hidden: u32,
    pub ple_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub embedding_dim: u32,
    pub q_heads: u32,
    pub sliding_window: u32,
    pub rms_norm_eps: f32,
    pub layers: Vec<LayerSpec>,
    /// The special token ids the multimodal entry needs; each is `None` when
    /// `config.json` does not carry it (a text-only config).
    pub special_tokens: SpecialTokens,
}

/// The special token ids of `config.json`, each present only if the file has
/// it. [`EmbedGemma2Model::encode`] needs none of them;
/// [`EmbedGemma2Model::encode_multimodal`] needs [`Self::multimodal`], and the
/// sequence builders ([`EmbedGemma2Config::image_sequence`],
/// [`EmbedGemma2Config::audio_sequence`]) the ones they place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    /// `text_config.pad_token_id`: what a placeholder is embedded as.
    pub pad: Option<u32>,
    /// `text_config.bos_token_id` and `eos_token_id`: the ends of a sequence.
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /// Top-level `image_token_id`, `video_token_id` and `audio_token_id`: the
    /// placeholders that soft tokens replace.
    pub image: Option<u32>,
    pub video: Option<u32>,
    pub audio: Option<u32>,
    /// `boi_token_id`, `eoi_token_id`, `boa_token_id` and `eoa_token_index`
    /// (spelled so in the checkpoint's config): the markers around an image's
    /// and an audio clip's placeholders.
    pub boi: Option<u32>,
    pub eoi: Option<u32>,
    pub boa: Option<u32>,
    pub eoa: Option<u32>,
}

/// One piece of an item that interleaves text with media, for
/// [`EmbedGemma2Config::mixed_sequence`]: what the placeholder layout needs of
/// it and nothing more (the soft-token counts, not the media).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayoutSegment<'a> {
    /// Text, tokenized together with the text next to it.
    Text(&'a str),
    /// An image of `soft` soft tokens.
    Image { soft: usize },
    /// A clip of `soft` soft tokens.
    Audio { soft: usize },
    /// A video: the soft tokens of each of its frames, and with `timestamps`
    /// (one time in seconds per frame) the `mm:ss` labels the processor writes
    /// under `add_timestamps = true`.
    Video {
        soft_per_frame: &'a [usize],
        timestamps: Option<&'a [f64]>,
    },
}

/// The ids [`EmbedGemma2Model::encode_multimodal`] works with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmTokens {
    pub pad: u32,
    pub image: u32,
    pub video: u32,
    pub audio: u32,
}

impl MmTokens {
    /// Whether `id` is a placeholder that a soft token replaces.
    pub fn is_placeholder(&self, id: u32) -> bool {
        id == self.image || id == self.video || id == self.audio
    }
}

impl SpecialTokens {
    /// Every id of `named`, or an error naming each missing one by its
    /// `config.json` key.
    fn all<const N: usize>(what: &str, named: [(&str, Option<u32>); N]) -> Result<[u32; N], String> {
        Self::require(what, &named)?;
        Ok(named.map(|(_, v)| v.unwrap_or_default()))
    }

    /// `Ok` when every id of `named` is present, else the error naming each
    /// missing one by its `config.json` key.
    fn require(what: &str, named: &[(&str, Option<u32>)]) -> Result<(), String> {
        let missing: Vec<&str> = named.iter().filter(|(_, v)| v.is_none()).map(|(k, _)| *k).collect();
        if !missing.is_empty() {
            return Err(format!("{what}: config.json has no {}", missing.join(", ")));
        }
        Ok(())
    }

    /// What the multimodal entry needs: `pad_token_id`, the three
    /// placeholders, and the four markers (`boi_token_id`, `eoi_token_id`,
    /// `boa_token_id`, `eoa_token_index`) of a config that serves it. An
    /// error names every missing key.
    pub fn multimodal(&self, what: &str) -> Result<MmTokens, String> {
        let [pad, image, video, audio, _, _, _, _] = Self::all(
            what,
            [
                ("text_config.pad_token_id", self.pad),
                ("image_token_id", self.image),
                ("video_token_id", self.video),
                ("audio_token_id", self.audio),
                ("boi_token_id", self.boi),
                ("eoi_token_id", self.eoi),
                ("boa_token_id", self.boa),
                ("eoa_token_index", self.eoa),
            ],
        )?;
        Ok(MmTokens {
            pad,
            image,
            video,
            audio,
        })
    }
}

impl EmbedGemma2Config {
    /// The processor's ids for one image of `n_soft` soft tokens:
    /// `[<bos>, <boi>, <image> x n_soft, <eoi>, <eos>]`. Refused when
    /// `config.json` lacks one of those ids, or for `n_soft == 0` (a
    /// processor never emits an empty run).
    pub fn image_sequence(&self, n_soft: usize) -> Result<Vec<u32>, String> {
        let t = &self.special_tokens;
        let [bos, boi, image, eoi, eos] = SpecialTokens::all(
            "image_sequence",
            [
                ("text_config.bos_token_id", t.bos),
                ("boi_token_id", t.boi),
                ("image_token_id", t.image),
                ("eoi_token_id", t.eoi),
                ("text_config.eos_token_id", t.eos),
            ],
        )?;
        Self::framed("image_sequence", [bos, boi], image, [eoi, eos], n_soft)
    }

    /// The processor's ids for one audio clip of `n_soft` soft tokens:
    /// `[<bos>, <boa>, <audio> x n_soft, <eoa>, <eos>]`. Refused as
    /// [`Self::image_sequence`] is.
    pub fn audio_sequence(&self, n_soft: usize) -> Result<Vec<u32>, String> {
        let t = &self.special_tokens;
        let [bos, boa, audio, eoa, eos] = SpecialTokens::all(
            "audio_sequence",
            [
                ("text_config.bos_token_id", t.bos),
                ("boa_token_id", t.boa),
                ("audio_token_id", t.audio),
                ("eoa_token_index", t.eoa),
                ("text_config.eos_token_id", t.eos),
            ],
        )?;
        Self::framed("audio_sequence", [bos, boa], audio, [eoa, eos], n_soft)
    }

    /// The processor's ids for one video whose frames have `soft_per_frame[f]`
    /// soft tokens: `[<bos>, (<boi>, <video> x n_f, <eoi>) for each frame,
    /// <eos>]`, with no separator and no timestamps, which is
    /// `EmbeddingGemma2Processor.replace_video_token` under
    /// `add_timestamps = false` (the snapshot's setting) wrapped in the
    /// tokenizer's `<bos>` and `<eos>`. Refused as [`Self::image_sequence`] is,
    /// and for a video of no frames or a frame of no soft tokens.
    pub fn video_sequence(&self, soft_per_frame: &[usize]) -> Result<Vec<u32>, String> {
        self.video_layout("video_sequence", soft_per_frame, None)
    }

    /// [`Self::video_sequence`] under `add_timestamps = true`
    /// (`replace_video_token`, processing:255-268): every frame's block is
    /// preceded by its `mm:ss` text, and the blocks are joined by single
    /// spaces,
    ///
    /// ```text
    /// 00:00 <boi><video> x n_0<eoi> 00:01 <boi><video> x n_1<eoi> ...
    /// ```
    ///
    /// then the tokenizer wraps the whole in `<bos>` and `<eos>`. The text
    /// between two added tokens is tokenized on its own (the added tokens
    /// split the text before the model runs), so the ids are `[<bos>`, then
    /// for the first frame the ids of `"mm:ss "`, for every later frame the
    /// ids of `" mm:ss "`, each followed by the frame's block, `<eos>]`; the
    /// label's ids come from `tokenizer` ([`Tokenizer::encode_raw`] without
    /// special tokens), so a space is whatever the tokenizer makes of it (a
    /// `▁` in the snapshot's). `timestamps[f]` is frame `f`'s time in seconds
    /// (`PreparedVideo::timestamps`), formatted by
    /// [`crate::embedgemma2_pre::timestamp_label`]; each must be finite, not
    /// negative, and below [`MAX_TIMESTAMP_SECONDS`]. One timestamp per frame
    /// is required. Otherwise refused as [`Self::video_sequence`] is.
    pub fn video_sequence_with_timestamps(
        &self,
        tokenizer: &Tokenizer,
        timestamps: &[f64],
        soft_per_frame: &[usize],
    ) -> Result<Vec<u32>, String> {
        self.video_layout(
            "video_sequence_with_timestamps",
            soft_per_frame,
            Some((tokenizer, timestamps)),
        )
    }

    /// A video's ids, with or without the timestamps' labels in front of the
    /// blocks: a mixed layout of that one video ([`Self::layout`]).
    fn video_layout(
        &self,
        what: &str,
        soft_per_frame: &[usize],
        labels: Option<(&Tokenizer, &[f64])>,
    ) -> Result<Vec<u32>, String> {
        let segment = LayoutSegment::Video {
            soft_per_frame,
            timestamps: labels.map(|(_, stamps)| stamps),
        };
        self.layout(what, labels.map(|(tokenizer, _)| tokenizer), &[segment])
    }

    /// The ids of an item that interleaves text with media, built the way the
    /// processor builds them. The snapshot's chat template joins the text
    /// pieces and the media (as the processor's placeholder tokens) in order
    /// with nothing between them; the processor expands each placeholder to its
    /// block (an image's `<boi> <image> x n <eoi>`, a clip's `<boa> <audio> x n
    /// <eoa>`, a video's one such block per frame) and the tokenizer wraps the
    /// whole in `<bos>` and `<eos>`. The tokenizer splits the text at those
    /// added tokens, so the text between two blocks is tokenized as one stretch
    /// on its own: adjacent text segments are one stretch, and a timestamp's
    /// label and its spaces ([`Self::video_sequence_with_timestamps`]) join the
    /// text before them. `soft` counts are the towers' (the caller gets them
    /// from the vision and audio towers' `check_images` / `check_clips`, or
    /// from [`crate::embedgemma2_pre::image_soft_tokens`],
    /// [`crate::embedgemma2_pre::audio_soft_tokens`] and
    /// [`crate::embedgemma2_pre::video_soft_tokens`] before anything runs), so
    /// a caller never counts placeholders.
    ///
    /// `[Text("a photo of"), Image { soft: 256 }, Text("on a table")]` is
    /// `[<bos>, ids("a photo of"), <boi>, <image> x 256, <eoi>, ids("on a
    /// table"), <eos>]`. Refused: no segments, a medium of no soft tokens, a
    /// video of no frames, timestamps that are not one per frame or not times
    /// in `0..=`[`MAX_TIMESTAMP_SECONDS`], a stretch of text of
    /// [`crate::embedgemma2_tok::MAX_TEXT_BYTES`] bytes or more, and a config
    /// that lacks an id a segment places (naming each key).
    pub fn mixed_sequence(&self, tokenizer: &Tokenizer, segments: &[LayoutSegment<'_>]) -> Result<Vec<u32>, String> {
        self.layout("mixed_sequence", Some(tokenizer), segments)
    }

    /// The one builder of an interleaved item's ids ([`Self::mixed_sequence`]);
    /// `tokenizer` is needed only by text and timestamp labels.
    fn layout(
        &self,
        what: &str,
        tokenizer: Option<&Tokenizer>,
        segments: &[LayoutSegment<'_>],
    ) -> Result<Vec<u32>, String> {
        let t = &self.special_tokens;
        if segments.is_empty() {
            return Err(format!("{what}: no segments"));
        }
        // The ids the segments place, every missing one named at once.
        let mut need: Vec<(&str, Option<u32>)> =
            vec![("text_config.bos_token_id", t.bos), ("text_config.eos_token_id", t.eos)];
        let mut want = |keys: [(&'static str, Option<u32>); 3]| {
            for key in keys {
                if !need.iter().any(|(name, _)| *name == key.0) {
                    need.push(key);
                }
            }
        };
        for seg in segments {
            match seg {
                LayoutSegment::Text(_) => {}
                LayoutSegment::Image { .. } => want([
                    ("boi_token_id", t.boi),
                    ("image_token_id", t.image),
                    ("eoi_token_id", t.eoi),
                ]),
                LayoutSegment::Audio { .. } => want([
                    ("boa_token_id", t.boa),
                    ("audio_token_id", t.audio),
                    ("eoa_token_index", t.eoa),
                ]),
                LayoutSegment::Video { .. } => want([
                    ("boi_token_id", t.boi),
                    ("video_token_id", t.video),
                    ("eoi_token_id", t.eoi),
                ]),
            }
        }
        SpecialTokens::require(what, &need)?;
        let (bos, eos) = (t.bos.unwrap_or_default(), t.eos.unwrap_or_default());
        // Every other refusal comes before the first token is written.
        let mut len = 2usize;
        for (s, seg) in segments.iter().enumerate() {
            let at = |e: String| format!("{what}: segment {s}: {e}");
            match *seg {
                LayoutSegment::Text(_) => {}
                LayoutSegment::Image { soft } | LayoutSegment::Audio { soft } => {
                    len = Self::block_len(len, soft).map_err(at)?;
                }
                LayoutSegment::Video {
                    soft_per_frame,
                    timestamps,
                } => {
                    if soft_per_frame.is_empty() {
                        return Err(at("the video has no frames".into()));
                    }
                    if let Some(f) = soft_per_frame.iter().position(|&n| n == 0) {
                        return Err(at(format!("frame {f} has no soft tokens")));
                    }
                    for &n in soft_per_frame {
                        len = Self::block_len(len, n).map_err(at)?;
                    }
                    if let Some(stamps) = timestamps {
                        if tokenizer.is_none() {
                            return Err(at("timestamp labels need the tokenizer".into()));
                        }
                        if stamps.len() != soft_per_frame.len() {
                            return Err(at(format!(
                                "{} timestamps for {} frames; one per frame",
                                stamps.len(),
                                soft_per_frame.len()
                            )));
                        }
                        if let Some(f) = stamps
                            .iter()
                            .position(|s| !(s.is_finite() && (0.0..=MAX_TIMESTAMP_SECONDS).contains(s)))
                        {
                            return Err(at(format!(
                                "frame {f}'s timestamp {} is not a time in 0..={MAX_TIMESTAMP_SECONDS} seconds",
                                stamps[f]
                            )));
                        }
                    }
                }
            }
        }
        // Text between two blocks, written as one stretch and tokenized when the
        // next block (or the end) arrives.
        let mut stretch = String::new();
        let mut ids = Vec::with_capacity(len);
        ids.push(bos);
        let flush = |stretch: &mut String, ids: &mut Vec<u32>| -> Result<(), String> {
            if stretch.is_empty() {
                return Ok(());
            }
            let tokenizer = tokenizer.ok_or_else(|| format!("{what}: text needs the tokenizer"))?;
            ids.extend(tokenizer.encode_raw(stretch, false));
            stretch.clear();
            Ok(())
        };
        let push = |stretch: &mut String, s: usize, text: &str| -> Result<(), String> {
            // `encode_raw` panics on a text this long, and a caller's text is theirs.
            if stretch.len().saturating_add(text.len()) >= crate::embedgemma2_tok::MAX_TEXT_BYTES {
                return Err(format!(
                    "{what}: segment {s}: the text between two media reaches {} bytes; it must stay under {}",
                    stretch.len().saturating_add(text.len()),
                    crate::embedgemma2_tok::MAX_TEXT_BYTES
                ));
            }
            stretch.push_str(text);
            Ok(())
        };
        for (s, seg) in segments.iter().enumerate() {
            match *seg {
                LayoutSegment::Text(text) => push(&mut stretch, s, text)?,
                LayoutSegment::Image { soft } => {
                    flush(&mut stretch, &mut ids)?;
                    ids.push(t.boi.unwrap_or_default());
                    ids.resize(ids.len() + soft, t.image.unwrap_or_default());
                    ids.push(t.eoi.unwrap_or_default());
                }
                LayoutSegment::Audio { soft } => {
                    flush(&mut stretch, &mut ids)?;
                    ids.push(t.boa.unwrap_or_default());
                    ids.resize(ids.len() + soft, t.audio.unwrap_or_default());
                    ids.push(t.eoa.unwrap_or_default());
                }
                LayoutSegment::Video {
                    soft_per_frame,
                    timestamps,
                } => {
                    for (f, &n) in soft_per_frame.iter().enumerate() {
                        if let Some(stamps) = timestamps {
                            // The label and the spaces that join it to the
                            // neighbouring blocks, one stretch with the text before it.
                            let label = crate::embedgemma2_pre::timestamp_label(stamps[f]);
                            let spaced = if f == 0 {
                                format!("{label} ")
                            } else {
                                format!(" {label} ")
                            };
                            push(&mut stretch, s, &spaced)?;
                        }
                        flush(&mut stretch, &mut ids)?;
                        ids.push(t.boi.unwrap_or_default());
                        ids.resize(ids.len() + n, t.video.unwrap_or_default());
                        ids.push(t.eoi.unwrap_or_default());
                    }
                }
            }
        }
        flush(&mut stretch, &mut ids)?;
        ids.push(eos);
        Ok(ids)
    }

    /// `len` plus a block of `soft` placeholders and its two markers; a block
    /// of none is refused.
    fn block_len(len: usize, soft: usize) -> Result<usize, String> {
        if soft == 0 {
            return Err("the medium has no soft tokens".into());
        }
        len.checked_add(soft)
            .and_then(|l| l.checked_add(2))
            .ok_or_else(|| format!("the sequence length overflows with {soft} soft tokens"))
    }

    fn framed(what: &str, head: [u32; 2], placeholder: u32, tail: [u32; 2], n_soft: usize) -> Result<Vec<u32>, String> {
        if n_soft == 0 {
            return Err(format!("{what}: n_soft is zero"));
        }
        let len = n_soft
            .checked_add(4)
            .ok_or_else(|| format!("{what}: n_soft {n_soft} overflows"))?;
        let mut ids = Vec::with_capacity(len);
        ids.extend(head);
        ids.resize(2 + n_soft, placeholder);
        ids.extend(tail);
        Ok(ids)
    }

    /// Parse the checkpoint's `config.json` (the multimodal root with a
    /// `text_config`, or a bare text config). Everything the forward does not
    /// implement is refused by name.
    pub fn from_config_json(text: &str) -> Result<Self, String> {
        let root = config_json::parse_object(text)?;
        let c = root.get("text_config").unwrap_or(&root);
        let uint = |k: &str| config_json::uint(c, k);
        let num = |obj: &Json, k: &str| config_json::float(config_json::field(obj, k)?, k);
        // An id that may be absent (or null: a text-only config); a present one
        // must be a u32.
        let opt_uint = |obj: &Json, k: &str| -> Result<Option<u32>, String> {
            match obj.get(k) {
                None | Some(Json::Null) => Ok(None),
                Some(_) => config_json::uint(obj, k).map(Some),
            }
        };
        let string = |k: &str| config_json::string(c, k);

        if let Some(t) = string("model_type")? {
            if t != "embedding_gemma2_text" {
                return Err(format!("config.json: model_type {t:?} is not embedding_gemma2_text"));
            }
        }
        match string("hidden_activation")? {
            Some("gelu_pytorch_tanh") => {}
            other => {
                return Err(format!(
                    "config.json: hidden_activation must be gelu_pytorch_tanh, got {other:?}"
                ))
            }
        }
        if let Some(Json::Bool(true)) = c.get("attention_bias") {
            return Err("config.json: attention_bias is not supported".into());
        }

        let hidden = uint("hidden_size")?;
        let ple_dim = uint("hidden_size_per_layer_input")?;
        let intermediate = uint("intermediate_size")?;
        let vocab = uint("vocab_size")?;
        let embedding_dim = uint("embedding_dim")?;
        let q_heads = uint("num_attention_heads")?;
        let n_layers = uint("num_hidden_layers")? as usize;
        let sliding_window = uint("sliding_window")?;
        let head_dim = uint("head_dim")?;
        let kv_heads = uint("num_key_value_heads")?;
        let eps = num(c, "rms_norm_eps")?;
        for (k, v) in [
            ("hidden_size", hidden),
            ("hidden_size_per_layer_input", ple_dim),
            ("intermediate_size", intermediate),
            ("vocab_size", vocab),
            ("embedding_dim", embedding_dim),
            ("num_attention_heads", q_heads),
            ("sliding_window", sliding_window),
        ] {
            if v == 0 {
                return Err(format!("config.json: {k} is zero"));
            }
        }
        if n_layers == 0 {
            return Err("config.json: num_hidden_layers is zero".into());
        }
        if !(eps > 0.0 && eps.is_finite()) {
            return Err(format!("config.json: rms_norm_eps must be positive, got {eps}"));
        }

        let types: Vec<bool> = match c.get("layer_types") {
            Some(Json::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(i, t)| match t {
                    Json::Str(s) if s == "sliding_attention" => Ok(true),
                    Json::Str(s) if s == "full_attention" => Ok(false),
                    t => Err(format!(
                        "config.json: layer_types[{i}] = {t:?} is not a known layer type"
                    )),
                })
                .collect::<Result<_, _>>()?,
            Some(v) => return Err(format!("config.json: layer_types must be an array, got {v:?}")),
            None => return Err("config.json: missing \"layer_types\"".into()),
        };
        if types.len() != n_layers {
            return Err(format!(
                "config.json: layer_types has {} entries but num_hidden_layers is {n_layers}",
                types.len()
            ));
        }

        // Per-layer overrides, keyed by layer index ("05" in the checkpoint).
        let mut overrides: Vec<(Option<u32>, Option<u32>)> = vec![(None, None); n_layers];
        match c.get("per_layer_config") {
            None | Some(Json::Null) => {}
            Some(Json::Object(fields)) => {
                for (key, v) in fields {
                    let i: usize = key
                        .parse()
                        .map_err(|_| format!("config.json: per_layer_config key {key:?} is not a layer index"))?;
                    if i >= n_layers {
                        return Err(format!("config.json: per_layer_config names layer {i} of {n_layers}"));
                    }
                    if !matches!(v, Json::Object(_)) {
                        return Err(format!("config.json: per_layer_config[{key:?}] is not an object"));
                    }
                    if let Json::Object(inner) = v {
                        for (name, _) in inner {
                            if name != "head_dim" && name != "num_key_value_heads" {
                                return Err(format!(
                                    "config.json: per_layer_config[{key:?}].{name} is not supported"
                                ));
                            }
                        }
                    }
                    let get = |k: &str| v.get(k).map(|_| config_json::uint(v, k)).transpose();
                    overrides[i] = (get("head_dim")?, get("num_key_value_heads")?);
                }
            }
            Some(v) => return Err(format!("config.json: per_layer_config must be an object, got {v:?}")),
        }

        let rope = c
            .get("rope_parameters")
            .ok_or("config.json: missing \"rope_parameters\"")?;
        let theta_for = |kind: &str| -> Result<f32, String> {
            let p = rope
                .get(kind)
                .ok_or_else(|| format!("config.json: rope_parameters.{kind} is missing"))?;
            match p.get("rope_type") {
                Some(Json::Str(t)) if t == "default" => {}
                other => {
                    return Err(format!(
                        "config.json: rope_parameters.{kind}.rope_type must be \"default\", got {other:?}"
                    ))
                }
            }
            if p.get("partial_rotary_factor").is_some() {
                return Err(format!(
                    "config.json: rope_parameters.{kind}.partial_rotary_factor is not supported"
                ));
            }
            let theta = num(p, "rope_theta")?;
            if !(theta > 0.0 && theta.is_finite()) {
                return Err(format!(
                    "config.json: rope_parameters.{kind}.rope_theta must be positive"
                ));
            }
            Ok(theta as f32)
        };
        let theta_sliding = theta_for("sliding_attention")?;
        let theta_full = theta_for("full_attention")?;

        let mut layers = Vec::with_capacity(n_layers);
        for (i, &sliding) in types.iter().enumerate() {
            let (hd, kv) = overrides[i];
            let spec = LayerSpec {
                sliding,
                head_dim: hd.unwrap_or(head_dim),
                kv_heads: kv.unwrap_or(kv_heads),
                rope_theta: if sliding { theta_sliding } else { theta_full },
            };
            if encoder_attn_entry(spec.head_dim).is_none() {
                return Err(format!(
                    "config.json: layer {i} head_dim {} has no attention kernel (256 or 512)",
                    spec.head_dim
                ));
            }
            if spec.kv_heads == 0 || q_heads % spec.kv_heads != 0 {
                return Err(format!(
                    "config.json: layer {i}: num_attention_heads {q_heads} is not a multiple of kv heads {}",
                    spec.kv_heads
                ));
            }
            layers.push(spec);
        }
        let special_tokens = SpecialTokens {
            pad: opt_uint(c, "pad_token_id")?,
            bos: opt_uint(c, "bos_token_id")?,
            eos: opt_uint(c, "eos_token_id")?,
            image: opt_uint(&root, "image_token_id")?,
            video: opt_uint(&root, "video_token_id")?,
            audio: opt_uint(&root, "audio_token_id")?,
            boi: opt_uint(&root, "boi_token_id")?,
            eoi: opt_uint(&root, "eoi_token_id")?,
            boa: opt_uint(&root, "boa_token_id")?,
            eoa: opt_uint(&root, "eoa_token_index")?,
        };
        Ok(Self {
            hidden,
            ple_dim,
            intermediate,
            vocab,
            embedding_dim,
            q_heads,
            sliding_window,
            rms_norm_eps: eps as f32,
            layers,
            special_tokens,
        })
    }

    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The projections (`w_*`, `gate_up`, `down`, `ple_*`) are f32 under
/// [`GemmOperands::ExactF32`] and bf16 under [`GemmOperands::Bf16`]; the norms
/// are f32 under both.
struct Layer {
    spec: LayerSpec,
    input_norm: GpuBuffer,
    post_attn_norm: GpuBuffer,
    pre_ffn_norm: GpuBuffer,
    post_ffn_norm: GpuBuffer,
    /// `[hidden, (q_heads + 2 kv_heads) * head_dim]`: `q | k | v`.
    w_qkv: Tensor,
    w_o: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    /// `[hidden, 2 * intermediate]`: `gate | up`.
    gate_up: Tensor,
    down: Tensor,
    ple_gate: Tensor,
    ple_out: Tensor,
    post_ple_norm: GpuBuffer,
    layer_scalar: f32,
}

/// A loaded text encoder: weights on the device in the layouts the kernels read.
pub struct EmbedGemma2Model {
    cfg: EmbedGemma2Config,
    rt: Arc<GpuRuntime>,
    /// `[vocab, hidden]` bf16, as the checkpoint holds it, under either mode.
    embed: Tensor,
    /// The per-layer-input projection as the checkpoint holds it,
    /// `[n_layers * ple_dim, hidden]`: layer `i`'s rows are `[i * ple_dim,
    /// (i + 1) * ple_dim)`, so a run of layers is a run of rows, the right
    /// operand of one NT GEMM.
    ple_w: Tensor,
    ple_norm: GpuBuffer,
    final_norm: GpuBuffer,
    /// `[hidden, embedding_dim]`.
    projection: Tensor,
    layers: Vec<Layer>,
    /// How the projections are held and the GEMMs run
    /// ([`EmbedGemma2Model::load_with`]).
    operands: GemmOperands,
    max_tokens: u32,
    /// See [`EmbedGemma2Model::set_ple_run_bytes`].
    ple_run_bytes: usize,
}

/// What [`EmbedGemma2Model::encode`] returns.
pub struct EncodeOutput {
    /// `[batch, truncate_dim.unwrap_or(embedding_dim)]`, each row L2-normalized.
    pub embeddings: Vec<f32>,
    /// With `trace` (one sequence only): the residual stream after each
    /// layer, `[tokens, hidden]`, then the final norm's output. Empty otherwise.
    pub trace: Vec<Vec<f32>>,
    /// How many forwards the batch was split into (see [`MAX_BATCH_TOKENS`]).
    pub forwards: u32,
}

/// A text-encoder call whose forwards are queued and not yet waited for: only
/// each forward's output is held (everything else was released to the runtime
/// before this was returned). The caller waits, whatever
/// [`EmbedGemma2Model::queue_multimodal`] returned, and passes the result of
/// its wait to [`Self::finish`].
pub(crate) struct PendingEncode {
    outs: Vec<(Vec<usize>, Tensor)>,
    /// How queuing ended: a failure is reported by [`Self::finish`], after the
    /// caller's wait.
    queued: Result<(), String>,
    len: usize,
    /// Columns returned per sequence, and columns of the output buffer.
    dim: usize,
    full: usize,
    trace: Vec<Vec<f32>>,
    forwards: u32,
}

impl PendingEncode {
    /// The call's [`EncodeOutput`], once the caller has waited (`waited` is
    /// that wait's result). A queuing failure is returned before a wait
    /// failure.
    pub(crate) fn finish(self, waited: Result<(), String>) -> Result<EncodeOutput, String> {
        self.queued?;
        waited?;
        let mut embeddings = vec![0.0f32; self.len * self.dim];
        for (chunk, out) in &self.outs {
            // Read the prefix columns straight from the mapping: one copy.
            let out = out.buffer.try_contents_f32()?;
            for (n, &i) in chunk.iter().enumerate() {
                embeddings[i * self.dim..(i + 1) * self.dim]
                    .copy_from_slice(&out[n * self.full..n * self.full + self.dim]);
            }
        }
        Ok(EncodeOutput {
            embeddings,
            trace: self.trace,
            forwards: self.forwards,
        })
    }
}

/// Bytes of per-layer inputs one GEMM forms: every layer's at once would be
/// `rows * n_layers * ple_dim` floats, 1.5 GiB at [`MAX_BATCH_TOKENS`] with
/// the checkpoint's 24 layers of 512. A layer's inputs are 2 KiB a row, so
/// under this bound a forward of up to 341 rows forms all 24 in one GEMM,
/// one of 2048 rows 4 layers per GEMM, and one of 8192 rows or more one layer
/// per GEMM (the buffer is never smaller than one layer's: 16 MiB at 8192
/// rows, 64 MiB at 32768).
///
/// It is 16 MiB, not the 256 MiB it was: the buffer is part of every
/// forward's activation set, and at 8192 rows 256 MiB of the set's 736 MiB
/// was this one (240 MiB more than a layer needs). The bound moves no output
/// bit and no measurable time (`bench/results/embedgemma2_memory_m5pro.txt`).
pub const PLE_RUN_BYTES: usize = 16 << 20;

/// Elements of the per-layer-input buffer for a forward of `rows` padded
/// rows: as many layers' worth as fit in `run_bytes`, at least one layer's,
/// at most every layer's.
fn ple_run_elements(run_bytes: usize, rows: usize, ple_dim: usize, n_layers: usize) -> usize {
    (run_bytes / 4).max(rows * ple_dim).min(rows * ple_dim * n_layers)
}

/// A forward's fixed cost in padded rows, for [`pack_forwards`], measured as
/// the intercept over the slope of `encode` time against padded rows (short
/// uniform sequences, so attention is negligible: 1 to 128 sequences of 32
/// tokens, min of 5 rounds, an ordinary least squares fit over the 8 shapes)
/// on an M5 Pro with the real checkpoint.
///
/// Measured with every dispatch committing and waiting (what `encode` ran
/// before it opened a [`GpuRuntime::async_scope`], and what a runtime set to
/// serialize its dispatches still runs) it is `54.9 ms + 0.0525 ms/row`, so
/// [`FORWARD_COST_ROWS_SYNC`] is 1046. The fixed part is a forward's
/// dispatches each committing and waiting (inferred from the size of the gap
/// to the queued fit).
///
/// With the forwards queued and one host wait, as `encode` runs them, it is
/// `3.42 ms + 0.0515 ms/row`, so [`FORWARD_COST_ROWS_ASYNC`] is 67. The fixed
/// part is what a forward costs however few rows it has, most likely
/// re-reading the f32 weights (inferred, not separately measured).
///
/// The data and the fits are `bench/results/embedgemma2_forward_cost_m5pro.json`
/// (residuals within 2.2 ms of the queued fit, 6.8 ms of the synchronous
/// one). The cost only steers how a batch is split, never what a sequence's
/// embedding is.
const FORWARD_COST_ROWS_SYNC: usize = 1046;
const FORWARD_COST_ROWS_ASYNC: usize = 67;

/// Split `batch` into forwards of at most `max_rows` padded rows and at most
/// [`MAX_BATCH`] sequences. Indices are sorted longest first (ties in input
/// order) and cut into contiguous runs minimizing `sum(run_len * longest) +
/// forward_cost_rows * runs`, so a long document does not pad short queries
/// to its length. A sequence longer than `max_rows` runs alone.
fn pack_forwards(batch: &[&[u32]], max_rows: usize, forward_cost_rows: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..batch.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(batch[i].len()));
    let n = order.len();
    // best[k]: least cost of the first k sorted sequences; cut[k]: where the
    // last run of that solution starts.
    let mut best = vec![usize::MAX; n + 1];
    let mut cut = vec![0usize; n + 1];
    best[0] = 0;
    for start in 0..n {
        if best[start] == usize::MAX {
            continue;
        }
        let longest = batch[order[start]].len();
        for end in start + 1..=n.min(start + MAX_BATCH) {
            let rows = (end - start).saturating_mul(longest);
            if rows > max_rows && end > start + 1 {
                break;
            }
            let cost = best[start].saturating_add(rows).saturating_add(forward_cost_rows);
            if cost < best[end] {
                best[end] = cost;
                cut[end] = start;
            }
        }
    }
    let mut runs = Vec::new();
    let mut end = n;
    while end > 0 {
        let start = cut[end];
        runs.push(order[start..end].to_vec());
        end = start;
    }
    runs.reverse();
    runs
}

/// Every activation of an encode's forwards, allocated once per
/// [`EmbedGemma2Model::encode`] for its largest forward (`rows` padded rows,
/// `nb` sequences) and reused by each. A forward over fewer rows works on a
/// prefix of every buffer (`[rows, cols]` views).
///
/// Nothing here is zeroed on the host ([`GpuRuntime::alloc_tensor_unzeroed`]):
/// every element a forward reads, an earlier kernel of the same forward wrote
/// (the gather, a GEMM, a norm, the K/V writer for positions `< seq`, the
/// attention for every query row, padding included), and attention reads no
/// key at or past its sequence's length. `tests/embedgemma2_tiny.rs` runs the
/// forward with every unzeroed allocation poisoned to NaN.
struct Acts {
    rows: usize,
    nb: usize,
    /// `embed_tokens[ids] * sqrt(hidden)`, kept for the per-layer inputs.
    embeds: Tensor,
    resid: Tensor,
    x: Tensor,
    /// The `q|k|v` GEMM's output, `[rows, qkv_width]`. The same buffer as
    /// `gate_up` (this is a clone of it, [`Self::gate_up`]'s width or wider):
    /// the q/k-norm-RoPE kernel is the last reader of `qkv` in a layer, and
    /// the MLP's `gate | up` GEMM is the next writer of that storage, after
    /// the attention and the output projection; the next layer's `q|k|v`
    /// GEMM writes it again only after the gated activation has read
    /// `gate | up`. Each dispatch is barriered against the one before it, so
    /// the two live ranges never overlap, in either [`GemmOperands`] mode
    /// (both outputs are f32 in both).
    qkv: Tensor,
    q: Tensor,
    /// K and normed V, `[batch, capacity, kv_heads, head_dim]`, sized for
    /// the widest layer. The K/V writer and [`encoder_attn`] both derive the
    /// per-sequence capacity from the buffer's size
    /// ([`crate::nn::attn_kv_capacity`]), so every layer and every forward
    /// gets a consistent stride whatever its width and shape; it is at least
    /// `seq`, and the positions past `seq` are never written or read.
    k: Tensor,
    v: Tensor,
    /// The attention output; then, under [`GemmOperands::ExactF32`], once the
    /// output projection has read it, the MLP's `gelu_tanh(gate) * up` (sized
    /// for the wider of the two). Under `Bf16` the MLP's lives in
    /// [`Bf16Acts::wide`] and this is the attention output's width only.
    attn: Tensor,
    y: Tensor,
    /// The MLP's `gate | up`, `[rows, 2 * intermediate]`, from one GEMM; sized
    /// for the wider of that and the `q|k|v` row it shares its storage with
    /// ([`Self::qkv`]).
    gate_up: Tensor,
    /// The normed per-layer inputs of a run of layers, `[rows, run * ple_dim]`
    /// (see [`PLE_RUN_BYTES`]).
    ple: Tensor,
    ple_gate: Tensor,
    /// `gelu_tanh(ple_gate) * ple input`, what the layer's output projection
    /// reads: f32, or bf16 under `Bf16`.
    ple_mid: Tensor,
    pooled: Tensor,
    bf16: Option<Bf16Acts>,
}

/// What [`GemmOperands::Bf16`] adds to [`Acts`]: the bf16 form of each GEMM
/// input that no kernel writes in bf16 itself (the fused row passes, the
/// attention and the GELU gate do, into `x`, `wide` and `ple_mid`; only the
/// embeddings and the pooled vector are cast). Persistent and unzeroed like the
/// rest, so a GEMM never allocates its own rounded copy of an operand.
struct Bf16Acts {
    /// `[rows, hidden]`: the normed input of the `q|k|v` and `gate|up` GEMMs
    /// (the fused residual-add-and-norm passes write bf16 into it), then the
    /// rounded residual stream that feeds the per-layer-input gate (the
    /// feed-forward add writes it) once those have read it.
    x: Tensor,
    /// `[rows, max(q_heads * head_dim, intermediate)]`: the rounded attention
    /// output, then, once the output projection has read it, the MLP's
    /// `gelu_tanh(gate) * up` (written bf16 by the gate kernel).
    wide: Tensor,
    /// `[rows, hidden]`: the rounded embeddings, the per-layer-input GEMM's
    /// left operand, read by every run of layers.
    embeds: Tensor,
    /// `[sequences, hidden]`: the rounded pooled vectors.
    pooled: Tensor,
}

/// Queue `chunks` in order: for each, `make` its [`ForwardIo`], push it into
/// `pending` and only then `run` the forward on it, so a forward that fails
/// after queuing some dispatches still has its io held in `pending` until the
/// caller's wait (Metal 4 command buffers do not retain what they bind).
fn queue_forwards<Io>(
    chunks: Vec<Vec<usize>>,
    pending: &mut Vec<(Vec<usize>, Io)>,
    mut make: impl FnMut(&[usize]) -> Result<Io, String>,
    mut run: impl FnMut(&[usize], &Io) -> Result<(), String>,
) -> Result<(), String> {
    for chunk in chunks {
        let io = make(&chunk)?;
        pending.push((chunk, io));
        let (chunk, io) = pending.last().expect("just pushed");
        run(chunk, io)?;
    }
    Ok(())
}

/// What one forward alone owns, because the host writes it before the forward
/// runs and reads it after: a later forward must not rewrite it while the GPU
/// may still be reading it, so each gets its own (small) set.
/// [`EmbedGemma2Model::encode`] keeps all of it until the forwards are
/// queued, then releases everything but `out` to the runtime's deferred
/// queue, which holds it until its one wait has finished; `out` it reads after.
struct ForwardIo {
    /// The ids as the embedding gather reads them: padding is id 0, and under
    /// [`EmbedGemma2Model::encode_multimodal`] each placeholder is `pad_token_id`.
    ids: GpuBuffer,
    lens: GpuBuffer,
    /// `[batch, 2]`: each sequence's live rows, `[b * seq, b * seq + len)`.
    segments: GpuBuffer,
    /// `[batch, embedding_dim]`: the embeddings.
    out: Tensor,
    /// The forward's soft tokens, when it has any.
    soft: Option<SoftIo>,
}

/// One input sequence of [`EmbedGemma2Model::encode_multimodal`]: its token
/// ids, and the soft tokens that replace its placeholder ids.
#[derive(Clone, Copy, Debug)]
pub struct MmSequence<'a> {
    /// The token ids, placeholders included (`[<bos>, <boi>, <image> x S,
    /// <eoi>, <eos>]` for [`EmbedGemma2Config::image_sequence`]).
    pub ids: &'a [u32],
    /// `[n_placeholders, hidden]` row-major, finite: row `j` replaces the
    /// `j`-th row of `ids` whose id is an image, video or audio placeholder,
    /// in order, as it is (not scaled by `sqrt(hidden)`).
    pub soft: &'a [f32],
}

/// A run of `rows` consecutive soft-token rows (`hidden` f32 each) of a device
/// buffer, from row `row`. The towers describe their output this way: a vision
/// image's rows are one segment, an audio clip's are one segment per maximal
/// run of rows its output mask keeps. Nothing is copied or compacted; the
/// text encoder's scatter reads the rows where the tower wrote them.
#[derive(Clone)]
pub(crate) struct SoftSegment {
    pub(crate) buf: GpuBuffer,
    pub(crate) row: u32,
    pub(crate) rows: u32,
}

/// Where one sequence's soft rows come from, in the order of its placeholder
/// ids: a host slice (what [`MmSequence`] carries) or device segments (what
/// the towers' queued forwards produced).
#[derive(Clone, Copy)]
pub(crate) enum SoftSource<'a> {
    /// `[n, hidden]` row-major.
    Host(&'a [f32]),
    Device(&'a [SoftSegment]),
}

impl SoftSource<'_> {
    /// Soft rows, given the row width.
    fn rows(&self, hidden: usize) -> usize {
        match self {
            SoftSource::Host(v) => v.len() / hidden,
            SoftSource::Device(segs) => segs.iter().map(|s| s.rows as usize).sum(),
        }
    }
}

/// One input sequence of [`EmbedGemma2Model::encode_impl`]: ids and where its
/// soft rows come from. [`MmSequence`] is the public form, with host rows.
#[derive(Clone, Copy)]
pub(crate) struct Seq<'a> {
    pub(crate) ids: &'a [u32],
    pub(crate) soft: SoftSource<'a>,
}

impl<'a> From<MmSequence<'a>> for Seq<'a> {
    fn from(s: MmSequence<'a>) -> Self {
        Seq {
            ids: s.ids,
            soft: SoftSource::Host(s.soft),
        }
    }
}

/// One flat copy of the scatter: `(source, first source row, first padded
/// row, row count)`, the source an index into [`SoftIo::sources`].
type SoftRun = (usize, u32, u32, u32);

/// A forward's soft tokens: the device buffers they sit in, and where each run
/// of them lands. Host rows are uploaded (without a wait) into one buffer;
/// device rows are the towers' own. Either way the scatter is the same.
struct SoftIo {
    sources: Vec<GpuBuffer>,
    runs: Vec<SoftRun>,
}

/// Where the soft tokens of a chunk land in a forward padded to `seq`: one
/// run per maximal stretch of consecutive padded rows whose soft rows are also
/// consecutive in one source. `ids[b]` is sequence `b`'s ids and `segs[b]` its
/// soft rows in placeholder order as `(source, first row, row count)`; a run
/// is cut by a non-placeholder row, by padding, and by a change of source or a
/// gap between two segments, and is continued across a sequence boundary when
/// the last row of one and the first of the next are both placeholders and
/// their sources line up. A sequence whose placeholders and soft rows differ in
/// number is an error ([`EmbedGemma2Model::encode_impl`] checked the counts).
fn soft_runs(
    ids: &[&[u32]],
    segs: &[Vec<(usize, u32, u32)>],
    seq: usize,
    tokens: &MmTokens,
) -> Result<Vec<SoftRun>, String> {
    const WHAT: &str = "EmbedGemma2Model::forward";
    let mut runs: Vec<SoftRun> = Vec::new();
    for (b, (seq_ids, seq_segs)) in ids.iter().zip(segs).enumerate() {
        let mut rows = seq_segs
            .iter()
            .flat_map(|&(src, row, n)| (row..row + n).map(move |r| (src, r)));
        for (t, &id) in seq_ids.iter().enumerate() {
            if !tokens.is_placeholder(id) {
                continue;
            }
            let (src, src_row) = rows
                .next()
                .ok_or_else(|| format!("{WHAT}: sequence {b} has more placeholder ids than soft rows"))?;
            let dst = (b * seq + t) as u32;
            match runs.last_mut() {
                Some((s, sr, dr, n)) if *s == src && *sr + *n == src_row && *dr + *n == dst => *n += 1,
                _ => runs.push((src, src_row, dst, 1)),
            }
        }
        if rows.next().is_some() {
            return Err(format!("{WHAT}: sequence {b} has more soft rows than placeholder ids"));
        }
    }
    Ok(runs)
}

impl EmbedGemma2Model {
    /// Load the text encoder from `st`, whose tensors are named `{prefix}...`
    /// (`"language_model."` in `google/embeddinggemma-2`'s `model.safetensors`).
    /// Every tensor's shape is checked against `cfg`; a missing one is an
    /// error, and so is any tensor under `prefix` the forward does not read
    /// (with an empty prefix, any tensor in the file).
    ///
    /// This is [`GemmOperands::ExactF32`] ([`Self::load_with`]): the
    /// projections widened to f32, the GEMMs exact f32.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, prefix: &str, cfg: EmbedGemma2Config) -> Result<Self, String> {
        Self::load_with(rt, st, prefix, cfg, GemmOperands::ExactF32)
    }

    /// [`Self::load`] with the operand precision chosen
    /// (see the module docs, Precision).
    ///
    /// * [`GemmOperands::ExactF32`]: every projection widened to f32 on the
    ///   device, every GEMM exact f32 (transformers' fp32 forward up to
    ///   operation order). `encode` refuses a runtime whose relaxed f32 GEMMs
    ///   are on.
    /// * [`GemmOperands::Bf16`]: every projection kept as the checkpoint stores
    ///   it, bf16, with no f32 copy (the real checkpoint's weights are 0.54 GB
    ///   against 0.82 GB, and each forward reads the projections at half the
    ///   width), and every GEMM run with bf16 operands and f32 accumulation.
    ///   The checkpoint's projections must be bf16 (a tensor stored in another
    ///   dtype is refused by name rather than rounded), and the device needs
    ///   TensorOps. `encode` accepts a runtime with relaxed f32 GEMMs on, which
    ///   no GEMM of this forward uses.
    pub fn load_with(
        rt: &Arc<GpuRuntime>,
        st: &SafeTensors,
        prefix: &str,
        cfg: EmbedGemma2Config,
        operands: GemmOperands,
    ) -> Result<Self, String> {
        let ld = Loader::new(rt, st, prefix);
        let (h, ple, inter, vocab) = (
            cfg.hidden as usize,
            cfg.ple_dim as usize,
            cfg.intermediate as usize,
            cfg.vocab as usize,
        );
        let n_layers = cfg.layers.len();
        let proj = match operands {
            GemmOperands::ExactF32 => Precision::F32,
            GemmOperands::Bf16 => {
                if !rt.has_tensorops() {
                    return Err(
                        "EmbedGemma2Model::load_with: bf16 operands need TensorOps, which this device lacks".into(),
                    );
                }
                // The attention's operands are bf16 too, and its bf16 kernel
                // is compiled for these head dims only.
                if let Some(s) = cfg
                    .layers
                    .iter()
                    .find(|s| encoder_attn_tiled_bf16_entry(s.head_dim).is_none())
                {
                    return Err(format!(
                        "EmbedGemma2Model::load_with: bf16 operands have no attention kernel for head dim {} \
                         (64, 256 and 512 are compiled)",
                        s.head_dim
                    ));
                }
                Precision::Bf16
            }
        };

        // Read straight into the device table: no host copy.
        let embed = rt.alloc_tensor_bf16_hot(&[vocab, h])?;
        ld.bf16_into("embed_tokens.weight", &[vocab, h], &embed)?;
        // In the checkpoint's layout, straight into the device tensor: widened
        // to f32, or as stored.
        let ple_name = "ple.per_layer_model_projection.weight";
        let ple_w = match proj {
            Precision::F32 => {
                let t = rt.alloc_tensor_f32_hot(&[n_layers * ple, h])?;
                ld.f32_into(ple_name, &[n_layers * ple, h], &t)?;
                t
            }
            Precision::Bf16 => {
                let t = rt.alloc_tensor_bf16_hot(&[n_layers * ple, h])?;
                ld.bf16_into(ple_name, &[n_layers * ple, h], &t)?;
                t
            }
        };
        let ple_norm = ld.norm("ple.per_layer_projection_norm.weight", ple)?;
        let final_norm = ld.norm("norm.weight", h)?;
        let projection = ld.linear(&[("embedding_projection.weight", cfg.embedding_dim as usize)], h, proj)?;

        let mut layers = Vec::with_capacity(n_layers);
        for (i, &spec) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{i}.{s}");
            let (d, q, kv) = (spec.head_dim as usize, cfg.q_heads as usize, spec.kv_heads as usize);
            layers.push(Layer {
                spec,
                input_norm: ld.norm(&p("input_layernorm.weight"), h)?,
                post_attn_norm: ld.norm(&p("post_attention_layernorm.weight"), h)?,
                pre_ffn_norm: ld.norm(&p("pre_feedforward_layernorm.weight"), h)?,
                post_ffn_norm: ld.norm(&p("post_feedforward_layernorm.weight"), h)?,
                w_qkv: ld.linear(
                    &[
                        (&p("self_attn.q_proj.weight"), q * d),
                        (&p("self_attn.k_proj.weight"), kv * d),
                        (&p("self_attn.v_proj.weight"), kv * d),
                    ],
                    h,
                    proj,
                )?,
                w_o: ld.linear(&[(&p("self_attn.o_proj.weight"), h)], q * d, proj)?,
                q_norm: ld.norm(&p("self_attn.q_norm.weight"), d)?,
                k_norm: ld.norm(&p("self_attn.k_norm.weight"), d)?,
                gate_up: ld.linear(
                    &[(&p("mlp.gate_proj.weight"), inter), (&p("mlp.up_proj.weight"), inter)],
                    h,
                    proj,
                )?,
                down: ld.linear(&[(&p("mlp.down_proj.weight"), h)], inter, proj)?,
                ple_gate: ld.linear(&[(&p("ple_block.per_layer_input_gate.weight"), ple)], h, proj)?,
                ple_out: ld.linear(&[(&p("ple_block.per_layer_projection.weight"), h)], ple, proj)?,
                post_ple_norm: ld.norm(&p("ple_block.post_per_layer_input_norm.weight"), h)?,
                layer_scalar: ld.scalar(&p("layer_scalar"))?,
            });
        }
        ld.refuse_unread()?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            embed,
            ple_w,
            ple_norm,
            final_norm,
            projection,
            layers,
            operands,
            max_tokens: DEFAULT_MAX_TOKENS,
            ple_run_bytes: PLE_RUN_BYTES,
        })
    }

    pub fn config(&self) -> &EmbedGemma2Config {
        &self.cfg
    }

    /// The runtime this model was loaded on.
    pub(crate) fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }

    /// The operand precision this model was loaded with.
    pub fn operands(&self) -> GemmOperands {
        self.operands
    }

    /// Bytes of weights held on the device: the embedding table, every
    /// projection in the dtype the mode holds it in, and the norms (f32). The
    /// activations are not counted.
    pub fn weight_bytes(&self) -> usize {
        let t = |t: &Tensor| t.nbytes_logical();
        let l: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.w_qkv, &l.w_o, &l.gate_up, &l.down, &l.ple_gate, &l.ple_out]
                    .into_iter()
                    .map(t)
                    .sum::<usize>()
                    + [
                        &l.input_norm,
                        &l.post_attn_norm,
                        &l.pre_ffn_norm,
                        &l.post_ffn_norm,
                        &l.q_norm,
                        &l.k_norm,
                        &l.post_ple_norm,
                    ]
                    .into_iter()
                    .map(GpuBuffer::nbytes)
                    .sum::<usize>()
            })
            .sum();
        t(&self.embed) + t(&self.ple_w) + t(&self.projection) + self.ple_norm.nbytes() + self.final_norm.nbytes() + l
    }

    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Raise or lower the per-sequence token limit, up to
    /// [`MAX_BATCH_TOKENS`] (32768). RoPE has no position table, so the
    /// checkpoint's `max_position_embeddings` (262144) is not a limit here;
    /// the model card states 8192 ([`DEFAULT_MAX_TOKENS`]).
    pub fn set_max_tokens(&mut self, max_tokens: u32) -> Result<(), String> {
        if max_tokens == 0 || max_tokens as usize > MAX_BATCH_TOKENS {
            return Err(format!(
                "max_tokens must be in 1..={MAX_BATCH_TOKENS}, got {max_tokens}"
            ));
        }
        self.max_tokens = max_tokens;
        Ok(())
    }

    /// Bound the per-layer-input buffer (default [`PLE_RUN_BYTES`]): each
    /// forward forms the per-layer inputs of as many layers per GEMM as fit
    /// in it (at least one). Smaller saves memory at more GEMMs; the
    /// embeddings agree to GEMM rounding whatever the size.
    pub fn set_ple_run_bytes(&mut self, bytes: usize) -> Result<(), String> {
        if bytes == 0 {
            return Err("ple_run_bytes must be positive".into());
        }
        self.ple_run_bytes = bytes;
        Ok(())
    }

    /// Embed a batch of tokenized sequences (with the tokenizer's BOS/EOS and
    /// any task prompt already in the ids). Returns one L2-normalized row per
    /// sequence, in the order given. A sequence's embedding does not depend
    /// on what else is in the batch beyond GEMM tiling (padded keys are
    /// masked, padded queries are not pooled).
    ///
    /// `truncate_dim`: `None` returns `embedding_dim` columns. `Some(d)`
    /// returns the Matryoshka prefix of `d` columns, normalized on the GPU
    /// over those `d` columns: sentence-transformers'
    /// `encode(truncate_dim=d, normalize_embeddings=True)`. The model card's
    /// sizes are 768, 512, 256 and 128; any `d` in `1..=embedding_dim` runs.
    ///
    /// The batch runs as one or more forwards: sequences sorted longest
    /// first, then cut into contiguous runs of at most [`MAX_BATCH_TOKENS`]
    /// padded rows and [`MAX_BATCH`] sequences, choosing the cuts (by dynamic
    /// programming) that minimize padded rows plus a fixed per-forward cost,
    /// so one long document does not pad every short query to its length.
    /// `trace` (per-layer residuals, for parity tests) takes one sequence.
    ///
    /// The forwards' dispatches are queued and waited for once, on a runtime
    /// whatever its [`GpuRuntime::set_async_encode`] flag says: `encode` opens
    /// a [`GpuRuntime::async_scope`] and leaves the flag as it found it. The
    /// runtime is single-threaded, so no other caller dispatches on it while
    /// `encode` runs; work the caller queued before the call is waited for by
    /// `encode`'s own wait. A runtime set to serialize its dispatches
    /// ([`GpuRuntime::set_serialized_dispatch`]) still commits and waits for
    /// each.
    pub fn encode(&self, batch: &[&[u32]], truncate_dim: Option<u32>, trace: bool) -> Result<EncodeOutput, String> {
        let seqs: Vec<Seq<'_>> = batch
            .iter()
            .map(|&ids| Seq {
                ids,
                soft: SoftSource::Host(&[]),
            })
            .collect();
        self.encode_impl("EmbedGemma2Model::encode", &seqs, None, truncate_dim, trace)
    }

    /// Tokenize `texts` with `tokenizer` and embed them: the tokenizer's ids
    /// ([`Tokenizer::encode_batch_with_prompt`]: the `<bos>`/`<eos>` and the
    /// task prompt `prompt_name` names, as sentence-transformers feeds them),
    /// then [`Self::encode`]. A text the tokenizer refuses or an id past this
    /// model's vocabulary is an error before any GPU work; `truncate_dim` and
    /// the rest are [`Self::encode`]'s.
    pub fn encode_texts(
        &self,
        tokenizer: &Tokenizer,
        prompt_name: Option<&str>,
        texts: &[&str],
        truncate_dim: Option<u32>,
    ) -> Result<EncodeOutput, String> {
        let ids = tokenizer
            .encode_batch_with_prompt(prompt_name, texts)
            .map_err(|e| format!("EmbedGemma2Model::encode_texts: {e}"))?;
        let batch: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        self.encode(&batch, truncate_dim, false)
    }

    /// [`Self::encode`] for sequences whose placeholder ids (`image_token_id`,
    /// `video_token_id`, `audio_token_id`) are replaced by soft tokens, what
    /// `EmbeddingGemma2Model.forward` does with the towers' outputs (see the
    /// module docs, Multimodal input): the placeholders are embedded as
    /// `pad_token_id`, then each sequence's `soft` rows overwrite its
    /// placeholder rows in order, unscaled, before the per-layer inputs and the
    /// residual stream are formed. The mean pool takes every token of the
    /// sequence, soft ones included (`include_prompt`).
    ///
    /// Refused before any GPU work: a config without the ids
    /// ([`SpecialTokens::multimodal`]), a `soft` whose length is not a
    /// multiple of `hidden`, a non-finite soft value, a number of soft rows
    /// that differs from the number of placeholder ids in `ids`, and
    /// everything [`Self::encode`] refuses (a placeholder id is exempt from
    /// the `vocab` bound, as in transformers, since it is never looked up).
    /// A sequence with no placeholders and no soft rows is a text sequence.
    /// Batching, packing and queueing are [`Self::encode`]'s: one host wait
    /// per call, and a sequence's embedding does not depend on its batch
    /// beyond GEMM tiling.
    pub fn encode_multimodal(
        &self,
        batch: &[MmSequence<'_>],
        truncate_dim: Option<u32>,
        trace: bool,
    ) -> Result<EncodeOutput, String> {
        const WHAT: &str = "EmbedGemma2Model::encode_multimodal";
        let tokens = self.cfg.special_tokens.multimodal(WHAT)?;
        let seqs: Vec<Seq<'_>> = batch.iter().map(|&s| Seq::from(s)).collect();
        self.encode_impl(WHAT, &seqs, Some(tokens), truncate_dim, trace)
    }

    /// [`Self::encode_multimodal`] for soft rows that are already on the
    /// device, up to the wait: the call is checked and its forwards are
    /// queued, and the host does not wait for them. The multimodal
    /// composition ([`crate::embedgemma2_mm`]) queues the towers, then this,
    /// then waits once, and [`PendingEncode::finish`]es. A call that fails
    /// while queuing still has its buffers held by the runtime until the
    /// caller's wait, which the caller must make whatever this returns.
    pub(crate) fn queue_multimodal(
        &self,
        what: &'static str,
        batch: &[Seq<'_>],
        truncate_dim: Option<u32>,
    ) -> Result<PendingEncode, String> {
        let tokens = self.cfg.special_tokens.multimodal(what)?;
        self.queue_impl(what, batch, Some(tokens), truncate_dim, false)
    }

    /// Everything about a batch of id sequences that [`Self::encode`] and
    /// [`Self::encode_multimodal`] refuse before any GPU work, without
    /// allocating or waiting: `truncate_dim` in range, a non-empty batch of at
    /// most [`MAX_BATCH`] sequences, `trace` only for one, exact-f32 GEMMs when
    /// loaded `ExactF32`, `pad_token_id` inside the vocabulary (`tokens` given),
    /// and for each sequence: non-empty, at most [`Self::max_tokens`], every
    /// id inside the vocabulary (a placeholder id is exempt when `tokens` is
    /// given, since it is never looked up). Returns each sequence's count of
    /// placeholder ids (zero for all when `tokens` is `None`), which the
    /// caller matches against its soft rows. The multimodal composition
    /// ([`crate::embedgemma2_mm`]) runs this before it queues a tower.
    pub(crate) fn check_sequences(
        &self,
        what: &str,
        ids: &[&[u32]],
        tokens: Option<MmTokens>,
        truncate_dim: Option<u32>,
        trace: bool,
    ) -> Result<Vec<usize>, String> {
        let cfg = &self.cfg;
        let dim_out = truncate_dim.unwrap_or(cfg.embedding_dim);
        if dim_out == 0 || dim_out > cfg.embedding_dim {
            return Err(format!(
                "{what}: truncate_dim {dim_out} must be in 1..={}",
                cfg.embedding_dim
            ));
        }
        crate::encode_batch::check_batch_shape(what, ids.len(), MAX_BATCH, trace)?;
        if self.operands == GemmOperands::ExactF32 && self.rt.f32_gemm_runs_relaxed() {
            return Err(format!(
                "{what}: the forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        if let Some(t) = &tokens {
            if t.pad >= cfg.vocab {
                return Err(format!("{what}: pad_token_id {} >= vocab {}", t.pad, cfg.vocab));
            }
        }
        // A placeholder is looked up as the pad id, so only the other ids are
        // bounded by the vocabulary.
        let is_ph = |id: u32| tokens.is_some_and(|t| t.is_placeholder(id));
        let mut placeholders = Vec::with_capacity(ids.len());
        for (b, seq) in ids.iter().enumerate() {
            if seq.is_empty() {
                return Err(format!("{what}: sequence {b} is empty"));
            }
            if seq.len() > self.max_tokens as usize {
                return Err(format!(
                    "{what}: sequence {b} has {} tokens; the limit is {} (truncate it, or raise it with set_max_tokens)",
                    seq.len(),
                    self.max_tokens
                ));
            }
            if let Some(&bad) = seq.iter().find(|&&id| id >= cfg.vocab && !is_ph(id)) {
                return Err(format!("{what}: sequence {b}: token id {bad} >= vocab {}", cfg.vocab));
            }
            placeholders.push(seq.iter().filter(|&&id| is_ph(id)).count());
        }
        Ok(placeholders)
    }

    /// The batching, packing and queueing of [`Self::encode`] and
    /// [`Self::encode_multimodal`] (`tokens`: `Some` for the latter, whose
    /// placeholders and soft rows it then checks and applies; `None` for the
    /// former, whose `soft` is always empty), and the one wait.
    fn encode_impl(
        &self,
        what: &'static str,
        batch: &[Seq<'_>],
        tokens: Option<MmTokens>,
        truncate_dim: Option<u32>,
        trace: bool,
    ) -> Result<EncodeOutput, String> {
        let pending = self.queue_impl(what, batch, tokens, truncate_dim, trace)?;
        // Wait even when queuing failed: the runtime holds the buffers of
        // what was queued until the wait ends (see `queue_impl`).
        let waited = self.rt.synchronize();
        pending.finish(waited)
    }

    /// [`Self::encode_impl`] up to the wait. The returned [`PendingEncode`]
    /// holds only what the host reads after it: each forward's output.
    fn queue_impl(
        &self,
        what: &'static str,
        batch: &[Seq<'_>],
        tokens: Option<MmTokens>,
        truncate_dim: Option<u32>,
        trace: bool,
    ) -> Result<PendingEncode, String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden as usize;
        let ids: Vec<&[u32]> = batch.iter().map(|s| s.ids).collect();
        let placeholders = self.check_sequences(what, &ids, tokens, truncate_dim, trace)?;
        let dim_out = truncate_dim.unwrap_or(cfg.embedding_dim);
        for (b, s) in batch.iter().enumerate() {
            if let SoftSource::Host(soft) = s.soft {
                if soft.len() % hidden != 0 {
                    return Err(format!(
                        "{what}: sequence {b}: {} soft values is not a multiple of hidden {hidden}",
                        soft.len()
                    ));
                }
                if let Some(i) = soft.iter().position(|v| !v.is_finite()) {
                    return Err(format!(
                        "{what}: sequence {b}: soft value {} (row {}, column {}) is not finite",
                        soft[i],
                        i / hidden,
                        i % hidden
                    ));
                }
            }
            // Device rows are not read back to be checked for finiteness: a
            // caller that needs that checks what the forward returns.
            let have = s.soft.rows(hidden);
            if placeholders[b] != have {
                return Err(format!(
                    "{what}: sequence {b} has {} placeholder ids but {have} soft rows",
                    placeholders[b],
                ));
            }
        }

        let dim = dim_out as usize;
        let mut out_trace = Vec::new();
        let mut forwards = 0u32;
        // Queue every dispatch, whatever the caller set
        // ([`GpuRuntime::async_scope`] leaves the runtime's `async_encode`
        // flag alone). Opened before the cost is chosen, which is what a
        // dispatch costs under it. It ends when this returns, so a failed
        // forward or a panic ends it too.
        let scope = self.rt.async_scope();
        let cost = if self.rt.dispatch_batching() {
            FORWARD_COST_ROWS_ASYNC
        } else {
            FORWARD_COST_ROWS_SYNC
        };
        let id_lists: Vec<&[u32]> = batch.iter().map(|s| s.ids).collect();
        let chunks = pack_forwards(&id_lists, MAX_BATCH_TOKENS, cost);
        // One set of activations for every forward: the most padded rows and
        // the most sequences any of them has (a chunk is padded to its first,
        // longest sequence).
        let rows = chunks.iter().map(|c| c.len() * id_lists[c[0]].len()).max().unwrap_or(0);
        let nb = chunks.iter().map(Vec::len).max().unwrap_or(0);
        let acts = self.acts(nb, rows)?;
        // The forwards are queued back to back and the host waits once, after
        // the last. Each keeps its own host-written inputs and host-read
        // output ([`ForwardIo`]), held in `pending` until then. The shared
        // activations need no copy per forward: forward n + 1 rewrites them
        // after forward n's last read of them, the same order of dependent
        // dispatches (the runtime barriers each against the one before, and
        // each command buffer against the queue) that one forward's layers
        // already rely on for `resid`. The GPU runs ahead of the host
        // encoding the next forward because of the scope above; only a runtime
        // set to serialize its dispatches commits and waits for each.
        let mut pending: Vec<(Vec<usize>, ForwardIo)> = Vec::with_capacity(chunks.len());
        let queued = queue_forwards(
            chunks,
            &mut pending,
            |chunk| {
                let seqs: Vec<Seq<'_>> = chunk.iter().map(|&i| batch[i]).collect();
                self.forward_io(&seqs, tokens.as_ref())
            },
            |chunk, io| {
                let seqs: Vec<&[u32]> = chunk.iter().map(|&i| id_lists[i]).collect();
                out_trace = self.forward(&acts, io, &seqs, dim_out, trace)?;
                forwards += 1;
                Ok(())
            },
        );
        // Hand everything the GPU only reads back to the runtime *before* the
        // wait: `acts` and each forward's uploaded inputs (and the soft rows
        // they hold, which the towers' device buffers are). A dropped pooled
        // buffer is parked in the runtime's deferred-release queue, which holds
        // it (the buffer outlives every queued command, though Metal 4 command
        // buffers do not retain what they bind) until a waiting commit has seen
        // the GPU finish, and that same wait then returns it to the pool's
        // freelist. Dropped after the wait instead, the buffers reach the
        // freelist only at the *next* commit, which the next `encode` comes
        // after its `acts()` allocation: two sets of activations were live
        // across consecutive calls (1472 MiB at 8192 rows, 736 with a wait in
        // between). Only the output of each forward stays, to be read after
        // the wait. A failed forward's io (which `queue_forwards` registers
        // before running it) is held by the runtime the same way, so the
        // caller waits even when this reports a failure ([`PendingEncode`]).
        let outs: Vec<(Vec<usize>, Tensor)> = pending.into_iter().map(|(chunk, io)| (chunk, io.out)).collect();
        drop(acts);
        drop(scope);
        Ok(PendingEncode {
            outs,
            queued,
            len: batch.len(),
            dim,
            full: cfg.embedding_dim as usize,
            trace: out_trace,
            forwards,
        })
    }

    /// This forward's own host-written inputs and host-read output for
    /// `batch` padded to its longest sequence, written without waiting for
    /// the GPU ([`GpuRuntime::alloc_buffer_from_u32`]; writing into a buffer
    /// already bound would wait for every queued forward). The GPU reads them
    /// when the forward runs, which may be after the host has queued the next
    /// one.
    ///
    /// With `tokens` (the multimodal entry) each placeholder id is uploaded as
    /// `pad_token_id`, and the sequences' soft rows, in order, are uploaded
    /// too, as raw f32 bits through the same no-wait path (the runtime has no
    /// f32 twin of `alloc_buffer_from_u32`). A forward with no soft rows
    /// allocates nothing for them.
    fn forward_io(&self, batch: &[Seq<'_>], tokens: Option<&MmTokens>) -> Result<ForwardIo, String> {
        const WHAT: &str = "EmbedGemma2Model::forward";
        let rt = &self.rt;
        let seq = batch.iter().map(|s| s.ids.len()).max().unwrap_or(0);
        let rows = batch.len() * seq;
        if seq == 0 || rows > MAX_BATCH_TOKENS {
            return Err(format!(
                "{WHAT}: {} sequences padded to {seq} tokens is {rows} rows; need 1..={MAX_BATCH_TOKENS}",
                batch.len()
            ));
        }
        let mut padded = vec![0u32; rows];
        for (b, s) in batch.iter().enumerate() {
            let dst = &mut padded[b * seq..b * seq + s.ids.len()];
            match tokens {
                Some(t) => {
                    for (d, &id) in dst.iter_mut().zip(s.ids) {
                        *d = if t.is_placeholder(id) { t.pad } else { id };
                    }
                }
                None => dst.copy_from_slice(s.ids),
            }
        }
        let lens: Vec<u32> = batch.iter().map(|s| s.ids.len() as u32).collect();
        let segs: Vec<u32> = batch
            .iter()
            .enumerate()
            .flat_map(|(b, s)| [(b * seq) as u32, (b * seq + s.ids.len()) as u32])
            .collect();
        let hidden = self.cfg.hidden as usize;
        let soft = match tokens {
            Some(t) if batch.iter().any(|s| s.soft.rows(hidden) > 0) => Some(self.soft_io(batch, seq, t)?),
            _ => None,
        };
        Ok(ForwardIo {
            ids: rt.alloc_buffer_from_u32(&padded)?,
            lens: rt.alloc_buffer_from_u32(&lens)?,
            segments: rt.alloc_buffer_from_u32(&segs)?,
            out: rt.alloc_tensor_unzeroed(&[batch.len(), self.cfg.embedding_dim as usize], DType::F32)?,
            soft,
        })
    }

    /// The soft rows of a forward's `batch` as device buffers and the runs
    /// that copy them over the placeholder rows. Host rows are uploaded, all
    /// of the forward's into one buffer, as raw f32 bits through the same
    /// no-wait path as the ids (the runtime has no f32 twin of
    /// `alloc_buffer_from_u32`); device segments are used where they are, so
    /// nothing is uploaded or copied for them. This is the one place the two
    /// sources meet: after it the scatter does not know which it has.
    fn soft_io(&self, batch: &[Seq<'_>], seq: usize, tokens: &MmTokens) -> Result<SoftIo, String> {
        const WHAT: &str = "EmbedGemma2Model::forward";
        let hidden = self.cfg.hidden as usize;
        let bits: Vec<u32> = batch
            .iter()
            .flat_map(|s| match s.soft {
                SoftSource::Host(v) => v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                SoftSource::Device(_) => Vec::new(),
            })
            .collect();
        let mut sources: Vec<GpuBuffer> = Vec::new();
        if !bits.is_empty() {
            sources.push(self.rt.alloc_buffer_from_u32(&bits)?);
        }
        let mut host_row = 0u32;
        let mut segs: Vec<Vec<(usize, u32, u32)>> = Vec::with_capacity(batch.len());
        for s in batch {
            let mut mine = Vec::new();
            match s.soft {
                SoftSource::Host(v) => {
                    let n = u32::try_from(v.len() / hidden)
                        .map_err(|_| format!("{WHAT}: {} soft rows exceed u32", v.len() / hidden))?;
                    if n > 0 {
                        mine.push((0, host_row, n));
                    }
                    host_row = host_row
                        .checked_add(n)
                        .ok_or_else(|| format!("{WHAT}: soft rows exceed u32"))?;
                }
                SoftSource::Device(device) => {
                    for seg in device.iter().filter(|g| g.rows > 0) {
                        let at = match sources.iter().position(|b| b.aliases(&seg.buf)) {
                            Some(i) => i,
                            None => {
                                sources.push(seg.buf.clone());
                                sources.len() - 1
                            }
                        };
                        mine.push((at, seg.row, seg.rows));
                    }
                }
            }
            segs.push(mine);
        }
        let ids: Vec<&[u32]> = batch.iter().map(|s| s.ids).collect();
        Ok(SoftIo {
            sources,
            runs: soft_runs(&ids, &segs, seq, tokens)?,
        })
    }

    /// Overwrite the placeholder rows of `embeds` (`rows` rows of `hidden`,
    /// just gathered as the pad id) with the forward's soft rows, unscaled:
    /// one flat [`copy_cols`] per run of consecutive placeholder rows whose
    /// soft rows are consecutive in one source, a single row of the run's
    /// whole width. A document with one image or clip has one run; the runs
    /// number at most `(rows + 1) / 2`.
    fn scatter_soft(&self, soft: &SoftIo, embeds: &GpuBuffer, rows: u32) -> Result<(), String> {
        const WHAT: &str = "EmbedGemma2Model::forward";
        let h = self.cfg.hidden;
        let flat = |count: u32, what: &str| {
            count
                .checked_mul(h)
                .ok_or_else(|| format!("{WHAT}: {what} ({count} rows of {h}) exceeds u32 indexing"))
        };
        let embeds_len = flat(rows, "padded rows")?;
        for &(source, src_row, dst_row, len) in &soft.runs {
            let buf = soft
                .sources
                .get(source)
                .ok_or_else(|| format!("{WHAT}: soft run names source {source} of {}", soft.sources.len()))?;
            let end = src_row
                .checked_add(len)
                .ok_or_else(|| format!("{WHAT}: soft run (src {src_row}, len {len}) exceeds u32"))?;
            if dst_row.checked_add(len).is_none_or(|e| e > rows) {
                return Err(format!(
                    "{WHAT}: soft run (dst {dst_row}, len {len}) exceeds {rows} padded rows"
                ));
            }
            copy_cols(
                &self.rt,
                Cols {
                    buf,
                    ld: flat(end, "soft rows")?,
                    off: src_row * h,
                },
                Cols {
                    buf: embeds,
                    ld: embeds_len,
                    off: dst_row * h,
                },
                1,
                len * h,
            )?;
        }
        Ok(())
    }

    /// Queue one forward over `batch` padded to its longest sequence
    /// (`io` from [`Self::forward_io`] for the same `batch`) in `a`, without
    /// waiting for it. The embeddings are in `io.out` once the caller has
    /// synchronized, `[batch, embedding_dim]` with the first `dim_out`
    /// columns normalized. With `trace` the forward waits per layer (one
    /// sequence, one forward) and returns the residual stream after each.
    fn forward(
        &self,
        a: &Acts,
        io: &ForwardIo,
        batch: &[&[u32]],
        dim_out: u32,
        trace: bool,
    ) -> Result<Vec<Vec<f32>>, String> {
        const WHAT: &str = "EmbedGemma2Model::forward";
        let rt = &self.rt;
        let cfg = &self.cfg;
        let seq = batch.iter().map(|ids| ids.len() as u32).max().unwrap_or(0);
        let nb = batch.len() as u32;
        let rows = nb as usize * seq as usize;
        if rows > a.rows || batch.len() > a.nb {
            return Err(format!(
                "{WHAT}: {rows} rows of {nb} sequences exceed the activations' {} rows of {}",
                a.rows, a.nb
            ));
        }

        let (h, rows_u) = (cfg.hidden, rows as u32);
        let eps = cfg.rms_norm_eps;
        let at = |t: &Tensor, cols: u32| view(t, rows_u, cols);
        let (embeds, resid, x, y) = (at(&a.embeds, h)?, at(&a.resid, h)?, at(&a.x, h)?, at(&a.y, h)?);
        let inter = cfg.intermediate;
        // Under `Bf16`, each GEMM input that has a bf16 staging reads that
        // instead of its f32 buffer. Each costs the kernel that writes it
        // nothing (a norm, the attention, the feed-forward residual add or a
        // GELU gate writes bf16 in place of f32) or one elementwise cast pass:
        // the embeddings and the pooled vectors once per forward. A cast reads
        // 4 bytes and writes 2 per element, against the GEMM reading the
        // operand at half the width and no per-GEMM copy: the rounded copies
        // are made once, into these buffers, never allocated.
        //
        // The row passes between the GEMMs are fused (`kernels/embedgemma2_text.metal`):
        // a layer's residual add and the norm that follows it are one launch
        // (the attention's add with the pre-feed-forward norm, the per-layer
        // block's add with the next layer's input norm, the last layer's with the
        // final norm), so only layer 0's input norm stands alone.
        let bf = a.bf16.as_ref();
        let (x_in, embeds_in) = match bf {
            Some(b) => (at(&b.x, h)?, at(&b.embeds, h)?),
            None => (x.clone(), embeds.clone()),
        };
        let gate_up = at(&a.gate_up, 2 * inter)?;
        let mid = match bf {
            Some(b) => view(&b.wide, rows_u, inter)?,
            None => at(&a.attn, inter)?,
        };
        let (ple_gate, ple_mid) = (at(&a.ple_gate, cfg.ple_dim)?, at(&a.ple_mid, cfg.ple_dim)?);
        let (n_layers, pd) = (self.layers.len(), cfg.ple_dim as usize);
        // Layers whose per-layer inputs one GEMM forms: as many as `a.ple`
        // holds at this forward's rows.
        let run = (a.ple.numel() / (rows * pd)).clamp(1, n_layers);
        let mut ple_ld = 0u32;
        // transformers multiplies by `embed_scale.to(weight.dtype)`: sqrt(512)
        // as an f32 in the fp32 forward this mirrors, here inside the gather.
        qwen35::embed_rows_scaled(
            rt,
            &io.ids,
            rows_u,
            LmHead {
                weight: &self.embed.buffer,
                dtype: DType::BF16,
                vocab: cfg.vocab,
            },
            h,
            (h as f32).sqrt(),
            &embeds.buffer,
        )?;
        // The multimodal entry: soft tokens over the placeholder rows, before
        // anything reads `embeds` (the residual stream, the bf16 copy, the
        // per-layer-input GEMM).
        if let Some(soft) = &io.soft {
            self.scatter_soft(soft, &a.embeds.buffer, rows_u)?;
        }
        gpu_copy(&embeds, &resid)?;
        if bf.is_some() {
            cast_f32_to_bf16_into(&embeds, &embeds_in)?;
        }
        // Layer 0's normed input, for the GEMMs that read `x_in`: bf16
        // straight from the norm under `Bf16`. Every later layer's is written
        // by the previous layer's last residual add.
        match bf {
            Some(_) => rms_norm_bf16(
                rt,
                &resid.buffer,
                &self.layers[0].input_norm,
                &x_in.buffer,
                rows_u,
                h,
                eps,
            ),
            None => rms_norm_f32(
                rt,
                &resid.buffer,
                &self.layers[0].input_norm,
                &x_in.buffer,
                rows_u,
                h,
                eps,
            ),
        }?;

        let mut out_trace = Vec::new();
        let ple_scale = (h as f32).powf(-0.5);
        for (i, layer) in self.layers.iter().enumerate() {
            let s = layer.spec;
            // Attention.
            let qkv_width = (cfg.q_heads + 2 * s.kv_heads) * s.head_dim;
            let qkv = at(&a.qkv, qkv_width)?;
            self.project(&x_in, &layer.w_qkv, &qkv, &a.attn)?;
            qwen35::attn_qk_norm_rope_columns(
                rt,
                &AttnShape {
                    batch: nb,
                    seq,
                    q_heads: cfg.q_heads,
                    kv_heads: s.kv_heads,
                    head_dim: s.head_dim,
                    rotary_dim: s.head_dim,
                },
                Cols::dense(&qkv.buffer, qkv_width),
                QkvColumns {
                    q_head_stride: s.head_dim,
                    k_col: cfg.q_heads * s.head_dim,
                    v_col: (cfg.q_heads + s.kv_heads) * s.head_dim,
                    v_norm: true,
                },
                0.0,
                &layer.q_norm,
                &layer.k_norm,
                &AttnTargets {
                    q_out: &a.q.buffer,
                    k_cache: &a.k.buffer,
                    v_cache: &a.v.buffer,
                },
                a.q.dtype,
                0,
                s.rope_theta,
                eps,
            )?;
            // Q, K and V are in the attention's operand type: f32, or under
            // `Bf16` the writer above stored them bf16.
            let attn_dims = EncoderAttnDims {
                batch: nb,
                seq,
                heads: cfg.q_heads,
                heads_kv: s.kv_heads,
                head_dim: s.head_dim,
                window: if s.sliding { cfg.sliding_window } else { 0 },
                scale: 1.0,
            };
            let attn_width = cfg.q_heads * s.head_dim;
            // The output projection reads the attention's output: under `Bf16`
            // the kernel stores it bf16, rounded as a cast pass would.
            match bf {
                Some(b) => {
                    let attn_in = view(&b.wide, rows_u, attn_width)?;
                    encoder_attn_bf16_out(
                        rt,
                        &a.q.buffer,
                        &a.k.buffer,
                        &a.v.buffer,
                        &attn_in.buffer,
                        &io.lens,
                        attn_dims,
                    )?;
                    self.project(&attn_in, &layer.w_o, &y, &a.qkv)?;
                }
                None => {
                    encoder_attn(
                        rt,
                        &a.q.buffer,
                        &a.k.buffer,
                        &a.v.buffer,
                        &a.attn.buffer,
                        &io.lens,
                        attn_dims,
                    )?;
                    let attn = at(&a.attn, attn_width)?;
                    self.project(&attn, &layer.w_o, &y, &a.qkv)?;
                }
            }
            // The attention's residual add and the pre-feed-forward norm: one
            // launch, the normed row written where the GEMM reads it.
            self.add_norm(
                &y,
                &layer.post_attn_norm,
                &resid,
                &layer.pre_ffn_norm,
                &x_in,
                rows_u,
                1.0,
            )?;

            // MLP.
            self.project(&x_in, &layer.gate_up, &gate_up, &a.attn)?;
            let window = |off: u32| Cols {
                buf: &gate_up.buffer,
                ld: 2 * inter,
                off,
            };
            qwen35::gated_act(
                rt,
                GatedAct::GeluTanh,
                window(0),
                window(inter),
                OutCols {
                    cols: Cols::dense(&mid.buffer, inter),
                    dtype: mid.dtype,
                },
                rows_u,
                inter,
            )?;
            self.project(&mid, &layer.down, &y, &a.qkv)?;
            match bf {
                // The residual stream stays f32; `x_in` is free again (the
                // GEMMs that read the normed input have), so the add leaves
                // the rounded copy the gate projection reads there.
                Some(_) => resid_add_bf16_copy(
                    rt,
                    &y.buffer,
                    &layer.post_ffn_norm,
                    &resid.buffer,
                    &x_in.buffer,
                    rows_u,
                    h,
                    eps,
                    1.0,
                )?,
                None => {
                    rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_ffn_norm, &resid.buffer, rows_u, h, eps, 1.0)?
                }
            }

            // Per-layer input: at the first layer of each run, one GEMM over
            // the embeddings for the whole run, `rms_norm(embeds @ Wpleᵀ *
            // hidden^-0.5) * ple_norm.w` in place (the scale and the norm one
            // launch); then this layer's block reads its column window.
            if i % run == 0 {
                let k = run.min(n_layers - i);
                ple_ld = (k * pd) as u32;
                let w = self.ple_w.try_view(&[k * pd, h as usize], i * pd * h as usize)?;
                let out = at(&a.ple, ple_ld)?;
                match self.operands {
                    GemmOperands::ExactF32 => gemm_nt_f32(&embeds_in, &w, &out, BACKEND)?,
                    GemmOperands::Bf16 => GemmOperands::Bf16.nt(&embeds_in, &w, &out)?,
                }
                scale_rms_norm_inplace(
                    rt,
                    &out.buffer,
                    &self.ple_norm,
                    rows_u * k as u32,
                    cfg.ple_dim,
                    eps,
                    ple_scale,
                )?;
            }
            match bf {
                Some(_) => self.project(&x_in, &layer.ple_gate, &ple_gate, &a.qkv)?,
                None => self.project(&resid, &layer.ple_gate, &ple_gate, &a.qkv)?,
            }
            qwen35::gated_act(
                rt,
                GatedAct::GeluTanh,
                Cols::dense(&ple_gate.buffer, cfg.ple_dim),
                Cols {
                    buf: &a.ple.buffer,
                    ld: ple_ld,
                    off: ((i % run) * pd) as u32,
                },
                OutCols {
                    cols: Cols::dense(&ple_mid.buffer, cfg.ple_dim),
                    dtype: ple_mid.dtype,
                },
                rows_u,
                cfg.ple_dim,
            )?;
            self.project(&ple_mid, &layer.ple_out, &y, &a.qkv)?;
            // The layer's last residual add (with its layer scalar) and the
            // norm that follows it: the next layer's input norm into `x_in`, or
            // after the last layer the final norm into `x` (f32 under both
            // modes: the pooling reads it).
            match self.layers.get(i + 1) {
                Some(next) => self.add_norm(
                    &y,
                    &layer.post_ple_norm,
                    &resid,
                    &next.input_norm,
                    &x_in,
                    rows_u,
                    layer.layer_scalar,
                )?,
                None => resid_add_norm_f32(
                    rt,
                    &y.buffer,
                    &layer.post_ple_norm,
                    &resid.buffer,
                    &self.final_norm,
                    &x.buffer,
                    rows_u,
                    h,
                    eps,
                    layer.layer_scalar,
                )?,
            }
            if trace {
                rt.synchronize()?;
                out_trace.push(resid.buffer.try_contents_f32()?[..rows * h as usize].to_vec());
            }
        }

        let (pooled, out) = (view(&a.pooled, nb, h)?, view(&io.out, nb, cfg.embedding_dim)?);
        segment_mean_rows(rt, &x.buffer, &io.segments, &pooled.buffer, nb, rows_u, h)?;
        match bf {
            Some(b) => {
                let pooled_in = view(&b.pooled, nb, h)?;
                cast_f32_to_bf16_into(&pooled, &pooled_in)?;
                self.project(&pooled_in, &self.projection, &out, &a.qkv)?;
            }
            None => self.project(&pooled, &self.projection, &out, &a.qkv)?,
        }
        // A Matryoshka prefix normalizes its own columns; the rest are not read.
        l2_normalize_rows(rt, &out.buffer, nb, dim_out, cfg.embedding_dim)?;
        if trace {
            rt.synchronize()?;
            out_trace.push(x.buffer.try_contents_f32()?[..rows * h as usize].to_vec());
        }
        Ok(out_trace)
    }

    /// A residual add and the norm after it ([`resid_add_norm_f32`], or
    /// [`resid_add_norm_bf16`] under `Bf16`, so `out` is in the mode's
    /// activation dtype): `resid = layer_scale * (resid + rms_norm(y) * w_post)`
    /// in place, then `out = rms_norm(resid) * w_next`.
    #[allow(clippy::too_many_arguments)]
    fn add_norm(
        &self,
        y: &Tensor,
        w_post: &GpuBuffer,
        resid: &Tensor,
        w_next: &GpuBuffer,
        out: &Tensor,
        rows: u32,
        layer_scale: f32,
    ) -> Result<(), String> {
        let pass = match self.operands {
            GemmOperands::ExactF32 => resid_add_norm_f32,
            GemmOperands::Bf16 => resid_add_norm_bf16,
        };
        pass(
            &self.rt,
            &y.buffer,
            w_post,
            &resid.buffer,
            w_next,
            &out.buffer,
            rows,
            self.cfg.hidden,
            self.cfg.rms_norm_eps,
            layer_scale,
        )
    }

    /// `out = a @ w` at the model's operand precision. `a` is in the mode's
    /// activation dtype (f32, or bf16 under `Bf16`, as `forward` stages it), so
    /// neither lane rounds or allocates anything of its own here.
    ///
    /// This is the text encoder's own entry and the only caller that opts in
    /// to the small-M split-K ([`gemm_small_m`]): at 16 rows the down, o and
    /// per-layer-input GEMMs run 2-3x faster, and the exact-f32 sum moves only
    /// in its order (alone against batched, within `BATCH_ABS` of
    /// `tests/embedgemma2_model.rs`). The free [`project`] the vision tower
    /// calls keeps the default route, so its clips stay bit for bit what they
    /// are alone. `Bf16` takes the route it always had.
    ///
    /// `scratch` is a buffer of the forward's activations that is idle at this
    /// GEMM (neither its input nor its output, and nothing reads it after):
    /// the lane's K-partition slices go there, so a forward holds no device
    /// memory beyond [`Acts`]. One too small for the shape's partitions keeps
    /// the single dispatch.
    fn project(&self, a: &Tensor, w: &Tensor, out: &Tensor, scratch: &Tensor) -> Result<(), String> {
        match self.operands {
            GemmOperands::ExactF32 => gemm_small_m(a, w, out, BACKEND, Some(scratch)),
            GemmOperands::Bf16 => project(self.operands, a, w, out),
        }
    }

    /// [`Acts`] for forwards of up to `rows` padded rows over up to `nb`
    /// sequences.
    fn acts(&self, nb: usize, rows: usize) -> Result<Acts, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let r = rows.max(1);
        let nb = nb.max(1);
        let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
        // Widest attention layer: the q|k|v row, the query/output buffers and
        // K/V, which narrower layers read through a prefix.
        let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
        for s in &cfg.layers {
            let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
            qkv_w = qkv_w.max((q + 2 * kv) * d);
            q_w = q_w.max(q * d);
            kv_w = kv_w.max(kv * d);
        }
        let f32s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::F32);
        let bf16 = self.operands == GemmOperands::Bf16;
        let bf16s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::BF16);
        let gate_up = f32s(&[r, qkv_w.max(2 * inter)])?;
        Ok(Acts {
            rows: r,
            nb,
            embeds: f32s(&[r, h])?,
            resid: f32s(&[r, h])?,
            x: f32s(&[r, h])?,
            qkv: gate_up.clone(),
            // The attention's operands: bf16 under `Bf16`, written so by the
            // q/k-norm-RoPE kernel (no cast pass, and half the bytes read
            // back by the attention).
            q: if bf16 { bf16s(&[r, q_w])? } else { f32s(&[r, q_w])? },
            k: if bf16 { bf16s(&[r, kv_w])? } else { f32s(&[r, kv_w])? },
            v: if bf16 { bf16s(&[r, kv_w])? } else { f32s(&[r, kv_w])? },
            // Under `Bf16` the MLP's activation lives in `Bf16Acts::wide`.
            attn: f32s(&[r, if bf16 { q_w } else { q_w.max(inter) }])?,
            y: f32s(&[r, h])?,
            gate_up,
            // A run of per-layer inputs: within `ple_run_bytes`, or one layer's
            // when even that is larger, and never more than every layer's.
            ple: f32s(&[ple_run_elements(self.ple_run_bytes, r, ple, cfg.layers.len())])?,
            ple_gate: f32s(&[r, ple])?,
            ple_mid: if bf16 { bf16s(&[r, ple])? } else { f32s(&[r, ple])? },
            pooled: f32s(&[nb, h])?,
            bf16: if bf16 {
                Some(Bf16Acts {
                    x: bf16s(&[r, h])?,
                    wide: bf16s(&[r, q_w.max(inter)])?,
                    embeds: bf16s(&[r, h])?,
                    pooled: bf16s(&[nb, h])?,
                })
            } else {
                None
            },
        })
    }
}

/// A `[rows, cols]` view over the front of `t`'s buffer: the activations
/// are sized for the widest layer and narrower layers use a prefix.
fn view(t: &Tensor, rows: u32, cols: u32) -> Result<Tensor, String> {
    t.try_view(&[rows as usize, cols as usize], 0)
}

#[cfg(test)]
mod tests {
    use super::{
        encoder_attn_tiled_entry, pack_forwards, queue_forwards, soft_runs, MmTokens, SpecialTokens,
        FORWARD_COST_ROWS_ASYNC, FORWARD_COST_ROWS_SYNC, MAX_BATCH,
    };

    /// The soft tokens' destinations: soft rows numbered in chunk order,
    /// padded rows `b * seq + t`, a run broken by any non-placeholder row or
    /// padding but continued across a sequence boundary when the last row of
    /// one and the first of the next are both placeholders.
    #[test]
    fn soft_runs_follow_the_padded_layout() {
        let tokens = MmTokens {
            pad: 0,
            image: 90,
            video: 91,
            audio: 92,
        };
        // The host form: one source (0), each sequence's rows one segment,
        // numbered in chunk order. Runs are `(source, first soft row, first
        // padded row, count)`.
        let run = |chunk: &[&[u32]], seq: usize| {
            let mut next = 0u32;
            let segs: Vec<Vec<(usize, u32, u32)>> = chunk
                .iter()
                .map(|ids| {
                    let n = ids.iter().filter(|&&id| tokens.is_placeholder(id)).count() as u32;
                    next += n;
                    if n == 0 {
                        Vec::new()
                    } else {
                        vec![(0, next - n, n)]
                    }
                })
                .collect();
            soft_runs(chunk, &segs, seq, &tokens).unwrap()
        };
        assert!(run(&[&[1, 2, 3]], 3).is_empty(), "no placeholders");
        // One image: a single run over the placeholders.
        assert_eq!(run(&[&[1, 90, 90, 90, 2]], 5), [(0, 0, 1, 3)]);
        // Three kinds, scattered: runs split at text, kinds merge when adjacent.
        assert_eq!(
            run(&[&[90, 91, 5, 92, 5, 5, 90]], 7),
            [(0, 0, 0, 2), (0, 2, 3, 1), (0, 3, 6, 1)]
        );
        // Padding separates: a sequence of 2 padded to 3 ends a run at its last token.
        assert_eq!(run(&[&[5, 90], &[90, 5, 5]], 3), [(0, 0, 1, 1), (0, 1, 3, 1)]);
        // Full-length sequences touch: rows 2 and 3 are one run, its soft rows
        // are consecutive.
        assert_eq!(
            run(&[&[5, 5, 90], &[90, 90, 5], &[5, 5, 90]], 3),
            [(0, 0, 2, 3), (0, 3, 8, 1)]
        );
        // A second sequence's rows continue the numbering.
        assert_eq!(run(&[&[90, 5], &[5, 91]], 2), [(0, 0, 0, 1), (0, 1, 3, 1)]);
    }

    /// Device sources: one placeholder run is cut wherever its soft rows are
    /// not consecutive in one source, and a run continues where they are.
    /// `segs[b]` is sequence `b`'s rows `(source, first row, count)` in
    /// placeholder order.
    #[test]
    fn soft_runs_split_where_the_sources_do() {
        let tokens = MmTokens {
            pad: 0,
            image: 90,
            video: 91,
            audio: 92,
        };
        let run = |ids: &[&[u32]], segs: &[&[(usize, u32, u32)]], seq: usize| {
            let segs: Vec<Vec<(usize, u32, u32)>> = segs.iter().map(|s| s.to_vec()).collect();
            soft_runs(ids, &segs, seq, &tokens)
        };
        // Two adjacent images in one buffer, back to back: one copy.
        assert_eq!(
            run(&[&[1, 90, 90, 90, 90, 2]], &[&[(0, 0, 2), (0, 2, 2)]], 6).unwrap(),
            [(0, 0, 1, 4)]
        );
        // The same two images in two buffers: one placeholder run, two copies.
        assert_eq!(
            run(&[&[1, 90, 90, 90, 90, 2]], &[&[(0, 0, 2), (1, 0, 2)]], 6).unwrap(),
            [(0, 0, 1, 2), (1, 0, 3, 2)]
        );
        // Two images in one buffer with a gap between (rows 2 and 3 not kept):
        // a clip whose output mask has a hole.
        assert_eq!(
            run(&[&[90, 90, 90, 90]], &[&[(0, 4, 2), (0, 8, 2)]], 4).unwrap(),
            [(0, 4, 0, 2), (0, 8, 2, 2)]
        );
        // Image and audio placeholders interleaved, each taking its next row
        // from its own tower's buffer, the way the facade orders them.
        assert_eq!(
            run(
                &[&[90, 92, 90, 5, 92]],
                &[&[(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]],
                5
            )
            .unwrap(),
            [(0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 2, 1), (1, 1, 4, 1)]
        );
        // A run continues across a sequence boundary when the sources line up.
        assert_eq!(
            run(&[&[5, 90], &[90, 5]], &[&[(0, 3, 1)], &[(0, 4, 1)]], 2).unwrap(),
            [(0, 3, 1, 2)]
        );
        // Sequences with no soft rows contribute nothing.
        assert_eq!(
            run(&[&[5, 5], &[90, 90]], &[&[], &[(0, 0, 2)]], 2).unwrap(),
            [(0, 0, 2, 2)]
        );
        // A count that does not match the placeholders is an error, not a
        // short or long copy.
        assert!(run(&[&[90, 90]], &[&[(0, 0, 1)]], 2).is_err());
        assert!(run(&[&[90]], &[&[(0, 0, 2)]], 1).is_err());
    }

    /// The special tokens: every key named when missing, the multimodal set
    /// complete or refused, and the placeholders (not the markers) test true.
    #[test]
    fn special_tokens_name_what_is_missing() {
        let none = SpecialTokens::default();
        let e = none.multimodal("w").unwrap_err();
        for key in [
            "text_config.pad_token_id",
            "image_token_id",
            "video_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
            "boa_token_id",
            "eoa_token_index",
        ] {
            assert!(e.contains(key), "{e}");
        }
        assert!(e.starts_with("w: "));
        let all = SpecialTokens {
            pad: Some(0),
            bos: Some(2),
            eos: Some(1),
            image: Some(10),
            video: Some(11),
            audio: Some(12),
            boi: Some(20),
            eoi: Some(21),
            boa: Some(22),
            eoa: Some(23),
        };
        let m = all.multimodal("w").unwrap();
        assert_eq!(
            m,
            MmTokens {
                pad: 0,
                image: 10,
                video: 11,
                audio: 12
            }
        );
        for id in 0..30 {
            assert_eq!(m.is_placeholder(id), (10..=12).contains(&id), "{id}");
        }
        let e = SpecialTokens { eoa: None, ..all }.multimodal("w").unwrap_err();
        assert!(e.contains("eoa_token_index") && !e.contains("boa_token_id"), "{e}");
    }

    /// A forward that fails after queuing dispatches must still have its io
    /// held (not dropped) when `queue_forwards` returns, so the caller's wait
    /// precedes its release.
    #[test]
    fn a_failed_forward_keeps_its_io_until_the_caller_waits() {
        use std::cell::Cell;
        struct Io<'a>(&'a Cell<usize>);
        impl Drop for Io<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut pending = Vec::new();
        let r = queue_forwards(
            vec![vec![0], vec![1], vec![2]],
            &mut pending,
            |_| Ok(Io(&drops)),
            |chunk, _| {
                if chunk[0] == 1 {
                    Err("mid-forward".to_string())
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(r, Err("mid-forward".to_string()));
        assert_eq!(
            pending.len(),
            2,
            "the failing forward's io is registered before it runs"
        );
        assert_eq!(drops.get(), 0, "no io dropped before the caller's wait");
        drop(pending);
        assert_eq!(drops.get(), 2);
    }

    /// Adversarial: the call order is make, run per chunk in input order, a
    /// failure stops later chunks from being made or run, and no io is
    /// released by `queue_forwards` itself, success or failure.
    #[test]
    fn queue_forwards_orders_calls_and_never_releases_io() {
        use std::cell::RefCell;
        use std::rc::Rc;
        struct Io(usize, Rc<RefCell<Vec<String>>>);
        impl Drop for Io {
            fn drop(&mut self) {
                self.1.borrow_mut().push(format!("drop{}", self.0));
            }
        }
        for fail_at in [None, Some(0usize), Some(2)] {
            let log = Rc::new(RefCell::new(Vec::<String>::new()));
            let mut pending: Vec<(Vec<usize>, Io)> = Vec::new();
            let (l1, l2) = (log.clone(), log.clone());
            let r = queue_forwards(
                vec![vec![0], vec![1], vec![2], vec![3]],
                &mut pending,
                |c| {
                    l1.borrow_mut().push(format!("make{}", c[0]));
                    Ok(Io(c[0], l1.clone()))
                },
                |c, io| {
                    assert_eq!(io.0, c[0], "run gets the io made for its own chunk");
                    l2.borrow_mut().push(format!("run{}", c[0]));
                    if Some(c[0]) == fail_at {
                        Err("boom".to_string())
                    } else {
                        Ok(())
                    }
                },
            );
            let expect_queued = fail_at.map_or(4, |f| f + 1);
            assert_eq!(r.is_err(), fail_at.is_some());
            assert_eq!(pending.len(), expect_queued, "fail_at {fail_at:?}");
            let want: Vec<String> = (0..expect_queued)
                .flat_map(|i| [format!("make{i}"), format!("run{i}")])
                .collect();
            assert_eq!(
                *log.borrow(),
                want,
                "no later chunk is made or run, and nothing dropped"
            );
            let ids: Vec<usize> = pending.iter().map(|(c, _)| c[0]).collect();
            assert_eq!(ids, (0..expect_queued).collect::<Vec<_>>(), "input order");
        }
    }

    /// A panicking forward (not just an Err) leaves its io in `pending`, which
    /// the caller owns and unwinds after the wait.
    #[test]
    fn a_panicking_run_leaves_its_io_registered() {
        let mut pending: Vec<(Vec<usize>, u8)> = Vec::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = queue_forwards(
                vec![vec![7]],
                &mut pending,
                |_| Ok(1u8),
                |_, _| -> Result<(), String> { panic!("run panics") },
            );
        }));
        assert!(r.is_err());
        assert_eq!(pending.len(), 1);
    }

    /// Existing entries in `pending` are kept and appended to; no chunks is a
    /// no-op that calls neither closure.
    #[test]
    fn queue_forwards_appends_and_handles_no_chunks() {
        let mut pending = vec![(vec![9usize], 0u8)];
        let r = queue_forwards(
            Vec::new(),
            &mut pending,
            |_| -> Result<u8, String> { panic!("make called") },
            |_, _| -> Result<(), String> { panic!("run called") },
        );
        assert_eq!(r, Ok(()));
        assert_eq!(pending.len(), 1);
        queue_forwards(vec![vec![1]], &mut pending, |_| Ok(1u8), |_, _| Ok(())).unwrap();
        assert_eq!(pending.iter().map(|p| p.0[0]).collect::<Vec<_>>(), vec![9, 1]);
    }

    /// Adversarial packing: pack_forwards on degenerate budgets still yields a
    /// forward per sequence at worst, every index once, nothing over the cap.
    #[test]
    fn pack_forwards_degenerate_budgets() {
        let seqs: Vec<Vec<u32>> = vec![vec![0; 3], vec![0; 1], vec![0; 2]];
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        for max_rows in [0usize, 1, 2, usize::MAX] {
            for cost in [0usize, 1, usize::MAX] {
                let packs = pack_forwards(&batch, max_rows, cost);
                let mut seen: Vec<usize> = packs.iter().flatten().copied().collect();
                seen.sort_unstable();
                assert_eq!(seen, vec![0, 1, 2], "max_rows {max_rows} cost {cost}");
                for p in &packs {
                    assert!(p.len() <= MAX_BATCH);
                    assert_eq!(batch[p[0]].len(), p.iter().map(|&i| batch[i].len()).max().unwrap());
                }
            }
        }
        assert!(pack_forwards(&[], 10, 1).is_empty());
        let many = vec![vec![0u32; 1]; MAX_BATCH * 2 + 1];
        let b: Vec<&[u32]> = many.iter().map(Vec::as_slice).collect();
        assert!(pack_forwards(&b, usize::MAX, 0).iter().all(|p| p.len() <= MAX_BATCH));
    }

    /// A failing `make` queues nothing for that chunk.
    #[test]
    fn a_failed_io_allocation_stops_the_queue() {
        let mut pending: Vec<(Vec<usize>, ())> = Vec::new();
        let r = queue_forwards(
            vec![vec![0], vec![1]],
            &mut pending,
            |c| if c[0] == 1 { Err("alloc".into()) } else { Ok(()) },
            |_, _| Ok(()),
        );
        assert_eq!(r, Err("alloc".to_string()));
        assert_eq!(pending.len(), 1);
    }

    /// The per-layer-input buffer at the checkpoint's shape (24 layers of
    /// 512) under the default bound: one layer's 16 MiB at 8192 rows, never
    /// more than one layer's from there up, every layer's at a few hundred
    /// rows. The default was 256 MiB, which at 8192 rows was 256 of the
    /// activation set's 736 MiB.
    #[test]
    fn the_per_layer_input_run_is_bounded_at_the_default() {
        use super::{ple_run_elements, PLE_RUN_BYTES};
        const MIB: usize = 1 << 20;
        let at = |rows: usize| ple_run_elements(PLE_RUN_BYTES, rows, 512, 24) * 4;
        assert_eq!(PLE_RUN_BYTES, 16 * MIB);
        assert_eq!(at(8192), 16 * MIB, "one layer at the largest single forward");
        assert_eq!(at(2048), 16 * MIB, "4 layers of 4 MiB");
        assert_eq!(at(341), 341 * 2048 * 24, "every layer fits");
        assert_eq!(at(342), 16 * MIB, "one row more and the 24th layer no longer does");
        assert_eq!(at(16), 16 * 2048 * 24, "a short query holds all 24 layers' inputs");
        assert_eq!(at(32768), 64 * MIB, "the floor is one layer, however many rows");
        // The old default, for contrast: this is what the bound removed.
        assert_eq!(ple_run_elements(256 * MIB, 8192, 512, 24) * 4, 256 * MIB);
        // Whatever the bound, one layer at least and every layer at most.
        for bound in [1usize, 4, 1 << 20, usize::MAX / 8] {
            for rows in [1usize, 7, 8192] {
                let n = ple_run_elements(bound, rows, 512, 24);
                assert!(
                    (rows * 512..=rows * 512 * 24).contains(&n),
                    "bound {bound} rows {rows}: {n}"
                );
            }
        }
    }

    /// Each tiled entry's name spells the geometry the host builds its grid
    /// from (`_q{BQ}_k{BK}_sg{NSG}`), and the source instantiates it with
    /// those numbers: the rule `tools/msl_emu/check_qwen35.py` holds the
    /// Qwen3.5 instantiations to.
    #[test]
    fn tiled_entries_spell_their_geometry() {
        let src = include_str!("../kernels/encoder_attn.metal");
        for d in [256u32, 512] {
            let (entry, bq, sg) = encoder_attn_tiled_entry(d).unwrap();
            assert!(entry.ends_with(&format!("_h{d}_q{bq}_k32_sg{sg}")), "{entry}");
            let inst = format!("ENCODER_TILED_KERNEL({entry}, {d}, {bq}, 32, {sg})");
            assert!(src.contains(&inst), "missing {inst}");
        }
    }

    /// [`check_at`] under the synchronous runtime's forward cost.
    fn check(lens: &[usize], max_rows: usize) -> Vec<Vec<usize>> {
        let packs = check_at(lens, max_rows, FORWARD_COST_ROWS_SYNC);
        check_at(lens, max_rows, FORWARD_COST_ROWS_ASYNC);
        packs
    }

    /// Every index exactly once; each forward within the row budget and the
    /// sequence cap; each forward padded to its first (longest) sequence;
    /// never costlier than the two simple splits it replaces.
    fn check_at(lens: &[usize], max_rows: usize, forward_cost: usize) -> Vec<Vec<usize>> {
        let seqs: Vec<Vec<u32>> = lens.iter().map(|&n| vec![0; n]).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let packs = pack_forwards(&batch, max_rows, forward_cost);
        let mut seen: Vec<usize> = packs.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>(), "{lens:?}");
        for p in &packs {
            assert!(!p.is_empty() && p.len() <= MAX_BATCH);
            let longest = p.iter().map(|&i| lens[i]).max().unwrap();
            assert_eq!(longest, lens[p[0]], "pads to its first sequence");
            assert!(p.len() * longest <= max_rows || p.len() == 1, "{p:?} over {max_rows}");
        }
        // Never worse than the two simple splits it replaces: all in one
        // forward when that fits, and greedy first-fit.
        let cost = |packs: &[Vec<usize>]| -> usize { packs.iter().map(|p| p.len() * lens[p[0]] + forward_cost).sum() };
        let mut order: Vec<usize> = (0..lens.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        let mut greedy: Vec<Vec<usize>> = Vec::new();
        for i in order {
            match greedy.last_mut() {
                Some(g) if (g.len() + 1) * lens[g[0]] <= max_rows && g.len() < MAX_BATCH => g.push(i),
                _ => greedy.push(vec![i]),
            }
        }
        assert!(
            cost(&packs) <= cost(&greedy),
            "{lens:?}: {} > greedy {}",
            cost(&packs),
            cost(&greedy)
        );
        packs
    }

    #[test]
    fn packs_the_reference_texts_by_length() {
        // tools/embedgemma2_ref's eight texts: the 6147- and 1658-token
        // documents each alone, the six short texts together (6 * 42 rows),
        // rather than everything padded to 6147 (8 * 6147 is over the budget)
        // or the short texts padded to 1658.
        let packs = check(&[20, 27, 10, 28, 42, 17, 1658, 6147], 32_768);
        assert_eq!(packs, vec![vec![7], vec![6], vec![4, 3, 1, 0, 5, 2]]);
    }

    /// A forward's fixed cost steers the split: two sequences of 300 and 200
    /// tokens cost 600 padded rows together and 500 apart plus one more
    /// forward's fixed cost. Where a forward costs more than 100 rows (the
    /// synchronous runtime's 1046) they share one; where it costs less (the
    /// queued runtime's 67) they split. A single constant, 256 before it was
    /// measured, packed them together in both.
    #[test]
    fn the_forward_cost_decides_whether_a_small_saving_splits() {
        let seqs = [vec![0u32; 300], vec![0u32; 200]];
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        assert_eq!(pack_forwards(&batch, 32_768, FORWARD_COST_ROWS_SYNC), vec![vec![0, 1]]);
        assert_eq!(
            pack_forwards(&batch, 32_768, FORWARD_COST_ROWS_ASYNC),
            vec![vec![0], vec![1]]
        );
    }

    #[test]
    fn packing_edges() {
        assert_eq!(check(&[5], 5), vec![vec![0]]);
        assert_eq!(check(&[9], 4), vec![vec![0]], "longer than the budget: alone");
        assert_eq!(check(&[4, 4], 8), vec![vec![0, 1]], "exactly at the budget");
        assert_eq!(check(&[4, 4], 7), vec![vec![0], vec![1]], "one row over");
        assert_eq!(check(&[3, 9, 3], 9), vec![vec![1], vec![0, 2]]);
        assert_eq!(check(&[1; 600], 32_768).len(), 3, "the sequence cap splits, not rows");
        // Equal lengths sort in input order, and of two equal-cost splits the
        // one found first is kept, so the split is deterministic.
        assert_eq!(check(&[2, 2, 2], 4), vec![vec![0], vec![1, 2]]);
        // Equal lengths never split while they fit: a split only adds a forward.
        assert_eq!(check(&[512; 64], 32_768).len(), 1);
        // A long one and many short ones: the short ones are not padded to it.
        let packs = check(&[4000, 30, 30, 30, 30, 30, 30, 30], 32_768);
        assert_eq!(packs[0], vec![0]);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..200 {
            let lens: Vec<usize> = (0..1 + (x % 300) as usize)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    1 + (x % 8192) as usize
                })
                .collect();
            check(&lens, 32_768);
        }
    }
}
