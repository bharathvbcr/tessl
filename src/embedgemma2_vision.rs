//! EmbeddingGemma 2's vision tower and multimodal embedder
//! (`google/embeddinggemma-2`'s `vision_tower.*` and `embed_vision.*`),
//! composed from tessl kernels and loaded straight from its
//! `model.safetensors`.
//!
//! It maps the image processor's patches to the 512-wide soft tokens the text
//! encoder ([`crate::embedgemma2`]) substitutes for an image's placeholder
//! tokens: transformers 5.19's `Gemma4VisionModel` (sentence-transformers'
//! `Transformer` calls it through `EmbeddingGemma2Model.get_image_features`),
//! then `EmbeddingGemma2MultimodalEmbedder`. Merging the soft tokens into a
//! text sequence is not here.
//!
//! # Forward (one image of `P` patches, `n` of them real)
//!
//! ```text
//! x      = 2 * (pixels - 0.5)                                  (the host, in f32)
//! h      = x @ Wproj + tab[0][pos_x] + tab[1][pos_y]           (zero on padding)
//! per layer (the text encoder's four norms, `* w`):
//!   a    = rms_norm(h) * input_layernorm.w
//!   q,k,v = a @ [Wq | Wk | Wv]
//!   q, k = rope2d(rms_norm(q) * q_norm.w), rope2d(rms_norm(k) * k_norm.w)   per head
//!   v    = rms_norm(v)                                           per head, no weight
//!   o    = softmax(q kᵀ * 1.0) v                  over the n real patches' keys
//!   h   += rms_norm(o @ Wo) * post_attention_layernorm.w
//!   a    = rms_norm(h) * pre_feedforward_layernorm.w
//!   h   += rms_norm(down(gelu_tanh(gate a) * up a)) * post_feedforward_layernorm.w
//! pooled = (k x k average of h by patch position) * sqrt(hidden)
//! soft   = rms_norm(pooled) (no weight) @ embedding_projection
//! ```
//!
//! `rope2d` splits a head's channels into two halves, rotates the first by the
//! patch's x position and the second by its y position, each with
//! `rotate_half` over its own channels and `inv_freq[p] = theta^(-2p / W)` for
//! a half of `W` channels (`apply_multidimensional_rope`,
//! `Gemma4VisionRotaryEmbedding`; theta 100). The checkpoint's linears are
//! plain (`use_clipped_linears` and `standardize` are false, and the config
//! refuses a checkpoint that sets them).
//!
//! # Padding and batching
//!
//! The processor pads every image to the same `P` (2520 patches for 280 soft
//! tokens) with `(-1, -1)` positions at the end. Padding patches never reach a
//! real one in transformers (keys are masked, and the pool gives them zero
//! hidden state), so they are stripped on the host: only the `n` real patches
//! are run, and `P` matters only to the pooler, which sizes its output as
//! `P / k²` cells exactly as `_avg_pool_by_positions` does. Padding must be a
//! suffix; anything else is refused. Images of different `n` share a forward
//! padded to the longest, with per-image lengths for attention (the text
//! encoder's scheme); the padded rows are exactly zero through the patch
//! embedder and their keys are masked.
//!
//! # Pooling
//!
//! A patch at `(x, y)` goes to cell `x/k + (max_x/k) * (y/k)` with
//! `max_x = max(x) + 1` of its own image; the output has one row per occupied
//! cell in cell order, plus cell 0 when the image has padding (transformers
//! counts padding as cell 0's, so cell 0 is always kept). A cell index at or
//! past `P / k²` is an error there (`one_hot`), and here. [`plan_pooling`] is
//! that mapping on the host; `vision_pool_members_f32` sums each cell's rows.
//!
//! # Precision
//!
//! As the text encoder ([`crate::embedgemma2`]), the mode is chosen when the
//! tower is loaded ([`EmbedGemma2VisionModel::load_with`]).
//!
//! [`EmbedGemma2VisionModel::load`] is [`GemmOperands::ExactF32`]: weights
//! widened from the checkpoint's bf16 exactly, every activation f32, every GEMM
//! exact f32 (it refuses a runtime with relaxed f32 GEMMs on). Its results are
//! unchanged by the bf16 path: the tiny checkpoint's and the real one's output
//! hash identically before and after.
//!
//! [`GemmOperands::Bf16`] keeps every projection as the checkpoint stores it,
//! bf16, on the device (no f32 copy), and runs every GEMM with bf16 operands and
//! f32 accumulation. The inputs of the GEMMs are rounded to bf16 where they are
//! produced when a kernel can write bf16 (the pixels, on the host as they are
//! scaled; the norms; the GELU gate), and by one cast pass per layer where one
//! cannot (the attention output, [`crate::embedgemma2::encoder_attn_bf16`]
//! writes f32). q, k and v are stored bf16 by the axial kernel and the
//! attention runs on bf16 tiles. The residual stream, the norms' arithmetic,
//! the rotation, the pooling and the soft tokens stay f32. A runtime with
//! relaxed f32 GEMMs on is accepted: no GEMM here is an f32 one. The audio tower
//! has no such mode ([`crate::embedgemma2_audio`] stays exact f32).
//!
//! # Kernels
//!
//! New, in `kernels/embedgemma2_vision.metal`: `vision_qkv_norm_rope_f32` and
//! `_bf16` ([`axial_qkv_norm_rope`], [`axial_qkv_norm_rope_bf16`]: one body,
//! two stored types), `vision_pos_embed_add_f32`
//! ([`patch_position_embed_add`]) and `vision_pool_members_f32`
//! ([`pool_by_members`]). The 1-D RoPE path ([`crate::qwen35`]'s) rotates a
//! head by one position per token and is paired with a backward kernel that
//! must stay its transpose, so the axial mode is its own kernel rather than a
//! mode of that one. Everything else is tessl's: GEMMs (exact f32, or bf16
//! operands), [`crate::nn::rms_norm_f32`] ([`crate::nn::rms_norm_bf16`] into a
//! bf16 GEMM's input; the embedder's weightless norm is the same kernel with a
//! weight of ones, which multiplies exactly),
//! [`crate::nn::rms_norm_residual_add_f32`], [`crate::qwen35::gated_act`] and
//! [`crate::embedgemma2::encoder_attn`] ([`crate::embedgemma2::encoder_attn_bf16`])
//! at head dim 64 (window 0, scale 1.0: every real key of the image, none past
//! its length).
//!
//! # Waiting
//!
//! Every forward of an [`EmbedGemma2VisionModel::encode_images`] call is queued
//! under a [`GpuRuntime::async_scope`] (on a runtime whatever its
//! `set_async_encode` flag says) behind one `synchronize`.
//! [`EmbedGemma2VisionModel::queue_images`] is that call up to the wait: it
//! returns a [`PendingImages`] so a caller can queue more work (the audio
//! tower, the text encoder) behind the same wait, then
//! [`PendingImages::collect`] the soft tokens.

use std::sync::Arc;

use crate::dispatch::{dispatch_2d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
use crate::embedgemma2::{
    encoder_attn, encoder_attn_bf16, encoder_attn_bf16_supports, project, EncoderAttnDims, SoftSegment,
};
use crate::gemm::{cast_f32_to_bf16_into, GemmOperands};
use crate::json::{self, Json, Syntax};
use crate::loader::Loader;
use crate::nn::{
    attn_kv_capacity, attn_kv_capacity_bf16, bind_rope_inv_freq, require, require_disjoint_writes, rms_norm_bf16,
    rms_norm_f32, rms_norm_residual_add_f32,
};
use crate::qwen35::{self, AttnTargets, Cols, GatedAct, OutCols};
use crate::qwen35_model::Precision;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{f32_to_bf16_bits_gpu, DType, GpuBuffer, Tensor};

/// Images per [`EmbedGemma2VisionModel::encode_images`] call.
pub const MAX_IMAGES: usize = 256;
/// Padded patches (`images * longest`) per forward, and the most real patches
/// one image may have. Bounds the activations: at this size the widest
/// buffer, the gate|up projection (`2 * 3072` columns), is 384 MiB, and all of
/// them about 1 GiB (the sum of the activation buffers' sizes). `encode_images` splits a call into as many forwards as
/// this needs, so it bounds memory, not what a caller may pass.
pub const MAX_BATCH_PATCHES: usize = 16_384;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// The largest head dim [`axial_qkv_norm_rope`] takes: the kernel keeps a
/// head's rotation table (one cos/sin pair per rotated pair, a lane's share of
/// `head_dim / 2`) in registers, sized for 512 (the checkpoint's is 64).
const MAX_ROPE_HEAD_DIM: u32 = 512;

/// Consecutive heads of one kind (q, k or v) of one row that one simdgroup of
/// `vision_qkv_norm_rope_*` takes: the rotation depends on the patch and not
/// the head, so they share one table of `cos`/`sin` (the dominant arithmetic)
/// that one simdgroup per head formed again for each.
const ROPE_HEADS_PER_UNIT: u32 = 6;

/// Shape of one [`axial_qkv_norm_rope`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxialQkvShape {
    pub batch: u32,
    /// Patches per image: row `b * seq + t` of the projection is patch `t` of image `b`.
    pub seq: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// Divisible by 4: each of the two axes rotates `head_dim / 2` channels in
    /// pairs.
    pub head_dim: u32,
}

/// Q/K RMSNorm with `* w`, the 2-D axial RoPE at each patch's `(x, y)`, the
/// weightless V RMSNorm and the K/V store, reading the fused projection
/// `[q | k | v]` in place (`Gemma4VisionAttention.forward` up to the
/// attention). Same `eps` for all three norms.
///
/// `proj`: rows `batch * seq`, q head `j` at column `proj.off + j * head_dim`,
/// k head `h` at `proj.off + k_col + h * head_dim`, v head `h` at
/// `proj.off + v_col + h * head_dim`. `positions`: `[batch * seq, 2]` i32
/// `(x, y)` (stored as u32 bits; a padding row carries `(-1, -1)`, which
/// rotates by -1 and is masked as a key). q goes to `targets.q_out`
/// `[batch, seq, q_heads, head_dim]`, k and v to the caches `[batch, capacity,
/// kv_heads, head_dim]` at slot `t`, the layout [`encoder_attn`] reads
/// (capacity derived from the cache buffers, [`crate::nn::attn_kv_capacity`],
/// as the attention derives it, and at least `seq`).
///
/// The frequencies are `nn::rope_inv_freq(head_dim / 4, head_dim / 2, theta)`,
/// computed on the host (a device `pow` is an ulp off at most pairs, which the
/// position multiplies).
///
/// `q_out`, `k_cache` and `v_cache` are f32; [`axial_qkv_norm_rope_bf16`] is
/// this with bf16 outputs.
#[allow(clippy::too_many_arguments)]
pub fn axial_qkv_norm_rope(
    rt: &Arc<GpuRuntime>,
    shape: &AxialQkvShape,
    proj: Cols<'_>,
    k_col: u32,
    v_col: u32,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    positions: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    axial_qkv_norm_rope_to(
        rt, shape, proj, k_col, v_col, q_norm_w, k_norm_w, targets, positions, theta, eps, false,
    )
}

/// [`axial_qkv_norm_rope`] storing q, k and v as bf16 (`vision_qkv_norm_rope_bf16`):
/// rounded to nearest even from the f32 result of the norm and the rotation,
/// the one rounding a cast pass would apply, with no pass. These are the
/// operands of [`encoder_attn_bf16`]; `targets`' buffers are bf16 (two bytes an
/// element) and the caches' capacity is derived from them as bf16
/// ([`crate::nn::attn_kv_capacity_bf16`]).
#[allow(clippy::too_many_arguments)]
pub fn axial_qkv_norm_rope_bf16(
    rt: &Arc<GpuRuntime>,
    shape: &AxialQkvShape,
    proj: Cols<'_>,
    k_col: u32,
    v_col: u32,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    positions: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    axial_qkv_norm_rope_to(
        rt, shape, proj, k_col, v_col, q_norm_w, k_norm_w, targets, positions, theta, eps, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn axial_qkv_norm_rope_to(
    rt: &Arc<GpuRuntime>,
    shape: &AxialQkvShape,
    proj: Cols<'_>,
    k_col: u32,
    v_col: u32,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    positions: &GpuBuffer,
    theta: f32,
    eps: f32,
    bf16: bool,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2_vision::axial_qkv_norm_rope";
    let s = shape;
    if s.head_dim == 0
        || s.head_dim % 4 != 0
        || s.head_dim > MAX_ROPE_HEAD_DIM
        || s.q_heads == 0
        || s.kv_heads == 0
        || s.q_heads % s.kv_heads != 0
    {
        return Err(format!(
            "{WHAT}: head_dim must be a positive multiple of 4 up to {MAX_ROPE_HEAD_DIM} and q_heads a multiple of \
             kv_heads, got head_dim {}, q_heads {}, kv_heads {}",
            s.head_dim, s.q_heads, s.kv_heads
        ));
    }
    if !(theta.is_finite() && theta > 0.0 && eps.is_finite() && eps > 0.0) {
        return Err(format!("{WHAT}: theta and eps must be positive and finite"));
    }
    if s.batch == 0 || s.seq == 0 {
        return Ok(());
    }
    let d = u64::from(s.head_dim);
    let span = |heads: u32| u64::from(heads) * d;
    let width = span(s.q_heads)
        .max(u64::from(k_col) + span(s.kv_heads))
        .max(u64::from(v_col) + span(s.kv_heads));
    let rows = u64::from(s.batch) * u64::from(s.seq);
    qwen35::require_window::<f32>(rt, proj, rows, width, "axial_qkv_norm_rope proj")?;
    let kv_capacity = if bf16 {
        attn_kv_capacity_bf16(targets.k_cache, targets.v_cache, s.batch, s.kv_heads, s.head_dim)?
    } else {
        attn_kv_capacity(targets.k_cache, targets.v_cache, s.batch, s.kv_heads, s.head_dim)?
    };
    if kv_capacity < s.seq {
        return Err(format!(
            "{WHAT}: the caches hold {kv_capacity} positions per image; need at least seq = {}",
            s.seq
        ));
    }
    require::<f32>(rt, q_norm_w, s.head_dim as usize, "axial_qkv_norm_rope q_norm weight")?;
    require::<f32>(rt, k_norm_w, s.head_dim as usize, "axial_qkv_norm_rope k_norm weight")?;
    require::<u32>(rt, positions, 2 * rows as usize, "axial_qkv_norm_rope positions")?;
    let q_elems = rows as usize * s.q_heads as usize * s.head_dim as usize;
    let cache_elems = s.batch as usize * kv_capacity as usize * s.kv_heads as usize * s.head_dim as usize;
    if bf16 {
        require::<u16>(rt, targets.q_out, q_elems, "axial_qkv_norm_rope q_out (bf16)")?;
        require::<u16>(rt, targets.k_cache, cache_elems, "axial_qkv_norm_rope k_cache (bf16)")?;
        require::<u16>(rt, targets.v_cache, cache_elems, "axial_qkv_norm_rope v_cache (bf16)")?;
    } else {
        require::<f32>(rt, targets.q_out, q_elems, "axial_qkv_norm_rope q_out")?;
        require::<f32>(rt, targets.k_cache, cache_elems, "axial_qkv_norm_rope k_cache")?;
        require::<f32>(rt, targets.v_cache, cache_elems, "axial_qkv_norm_rope v_cache")?;
    }
    require_disjoint_writes(
        WHAT,
        &[
            ("q_out", targets.q_out),
            ("k_cache", targets.k_cache),
            ("v_cache", targets.v_cache),
        ],
        &[
            ("proj", proj.buf),
            ("q_norm", q_norm_w),
            ("k_norm", k_norm_w),
            ("positions", positions),
        ],
    )?;
    // One simdgroup per (row, head); eight per threadgroup.
    const SIMDGROUPS: usize = 8;
    let units = rows as usize
        * (s.q_heads.div_ceil(ROPE_HEADS_PER_UNIT) + 2 * s.kv_heads.div_ceil(ROPE_HEADS_PER_UNIT)) as usize;
    let p = rt.pipeline(if bf16 {
        "vision_qkv_norm_rope_bf16"
    } else {
        "vision_qkv_norm_rope_f32"
    })?;
    dispatch_2d_tg(rt, &p, units.div_ceil(SIMDGROUPS), 1, SIMDGROUPS * 32, |bnd| {
        set_gpu_buf(bnd, proj.buf, 0);
        set_gpu_buf(bnd, q_norm_w, 1);
        set_gpu_buf(bnd, k_norm_w, 2);
        set_gpu_buf(bnd, targets.q_out, 3);
        set_gpu_buf(bnd, targets.k_cache, 4);
        set_gpu_buf(bnd, targets.v_cache, 5);
        set_gpu_buf(bnd, positions, 6);
        set_u32(bnd, s.batch, 7);
        set_u32(bnd, s.seq, 8);
        set_u32(bnd, s.q_heads, 9);
        set_u32(bnd, s.kv_heads, 10);
        set_u32(bnd, s.head_dim, 11);
        set_u32(bnd, proj.ld, 12);
        set_u32(bnd, proj.off, 13);
        set_u32(bnd, proj.off + k_col, 14);
        set_u32(bnd, proj.off + v_col, 15);
        set_u32(bnd, kv_capacity, 16);
        bind_rope_inv_freq(bnd, s.head_dim / 4, s.head_dim / 2, theta, 17);
        set_f32(bnd, eps, 18);
        set_u32(bnd, ROPE_HEADS_PER_UNIT, 19);
    })
}

/// `h[r, :] += table[0][x_r] + table[1][y_r]` for each row's `(x, y)` in
/// `positions` (`[rows, 2]` i32 as u32 bits), and `h[r, :] = +0` for a padding
/// row (`(-1, -1)`): `Gemma4VisionPatchEmbedder`'s `hidden_states +
/// position_embeddings`. `table`: `[2, table_len, dim]` bf16, widened exactly.
/// The host must keep every real position in `[0, table_len)` (the kernel
/// clamps, so a corrupted buffer stays in bounds, but a clamped lookup is a
/// wrong embedding, not an error); [`EmbedGemma2VisionModel::encode_images`]
/// checks them.
pub fn patch_position_embed_add(
    rt: &Arc<GpuRuntime>,
    h: &GpuBuffer,
    table: &GpuBuffer,
    positions: &GpuBuffer,
    rows: u32,
    dim: u32,
    table_len: u32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2_vision::patch_position_embed_add";
    if dim == 0 || table_len == 0 {
        return Err(format!("{WHAT}: dim and table_len must be non-zero"));
    }
    let n = rows as usize * dim as usize;
    require::<f32>(rt, h, n, "patch_position_embed_add h")?;
    require::<u16>(
        rt,
        table,
        2 * table_len as usize * dim as usize,
        "patch_position_embed_add table",
    )?;
    require::<u32>(rt, positions, 2 * rows as usize, "patch_position_embed_add positions")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("h", h)], &[("table", table), ("positions", positions)])?;
    let p = rt.pipeline("vision_pos_embed_add_f32")?;
    dispatch_2d(rt, &p, dim as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, h, 0);
        set_gpu_buf(bnd, table, 1);
        set_gpu_buf(bnd, positions, 2);
        set_u32(bnd, rows, 3);
        set_u32(bnd, dim, 4);
        set_u32(bnd, table_len, 5);
    })
}

/// `out[s, :] = (sum of h[members[j], :] * weight for j in offsets[s]..offsets[s +
/// 1]) * scale` for `s` in `0..out_rows`: the pooler's cell sums
/// ([`plan_pooling`] makes `offsets` and `members`), with `weight = 1 / k²` and
/// `scale = sqrt(hidden)`. A cell with no members is exactly +0. `offsets` and
/// `members` are device values: the kernel clamps `offsets` to `n_members` and
/// skips a member row at or past `rows`.
#[allow(clippy::too_many_arguments)]
pub fn pool_by_members(
    rt: &Arc<GpuRuntime>,
    h: &GpuBuffer,
    offsets: &GpuBuffer,
    members: &GpuBuffer,
    out: &GpuBuffer,
    out_rows: u32,
    rows: u32,
    dim: u32,
    n_members: u32,
    weight: f32,
    scale: f32,
) -> Result<(), String> {
    const WHAT: &str = "embedgemma2_vision::pool_by_members";
    if !(weight.is_finite() && scale.is_finite()) {
        return Err(format!("{WHAT}: weight and scale must be finite"));
    }
    require::<f32>(rt, h, rows as usize * dim as usize, "pool_by_members h")?;
    require::<u32>(rt, offsets, out_rows as usize + 1, "pool_by_members offsets")?;
    require::<u32>(rt, members, n_members as usize, "pool_by_members members")?;
    require::<f32>(rt, out, out_rows as usize * dim as usize, "pool_by_members out")?;
    if out_rows == 0 || dim == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("out", out)],
        &[("h", h), ("offsets", offsets), ("members", members)],
    )?;
    let p = rt.pipeline("vision_pool_members_f32")?;
    dispatch_2d(rt, &p, dim as usize, out_rows as usize, |bnd| {
        set_gpu_buf(bnd, h, 0);
        set_gpu_buf(bnd, offsets, 1);
        set_gpu_buf(bnd, members, 2);
        set_gpu_buf(bnd, out, 3);
        set_u32(bnd, out_rows, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, dim, 6);
        set_u32(bnd, n_members, 7);
        set_f32(bnd, weight, 8);
        set_f32(bnd, scale, 9);
    })
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The vision tower's shapes, read from `config.json` (`vision_config`), and
/// the width of the text encoder it feeds (`text_config.hidden_size`).
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub patch_size: u32,
    /// Side of the average-pool cell (`pooling_kernel_size`): at least 2.
    pub pooling_kernel: u32,
    /// Rows of each axis' position table.
    pub position_embedding_size: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Width of a soft token: the text encoder's hidden size.
    pub text_hidden: u32,
}

impl VisionConfig {
    /// Columns of one patch: `3 * patch_size²`.
    pub fn patch_dim(&self) -> u32 {
        3 * self.patch_size * self.patch_size
    }

    /// Parse the checkpoint's `config.json` (the multimodal root with a
    /// `vision_config` and a `text_config`). Everything the forward does not
    /// implement is refused by name.
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
            .get("vision_config")
            .ok_or("config.json: missing \"vision_config\"")?;
        let text_cfg = root.get("text_config").ok_or("config.json: missing \"text_config\"")?;
        let uint_of = |obj: &Json, k: &str| -> Result<u32, String> {
            match obj.get(k) {
                Some(Json::Num { uint: Some(n), .. }) => {
                    u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32"))
                }
                Some(v) => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        let num_of = |obj: &Json, k: &str| -> Result<f64, String> {
            match obj.get(k) {
                Some(Json::Num { value, .. }) => Ok(*value),
                Some(v) => Err(format!("config.json: {k} must be a number, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        let str_of = |obj: &Json, k: &str| -> Result<Option<String>, String> {
            match obj.get(k) {
                None => Ok(None),
                Some(Json::Str(s)) => Ok(Some(s.clone())),
                Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
            }
        };
        let flag = |k: &str| -> Result<bool, String> {
            match c.get(k) {
                None => Ok(false),
                Some(Json::Bool(b)) => Ok(*b),
                Some(v) => Err(format!("config.json: vision_config.{k} must be a boolean, got {v:?}")),
            }
        };

        if let Some(t) = str_of(c, "model_type")? {
            if t != "gemma4_vision" {
                return Err(format!(
                    "config.json: vision_config.model_type {t:?} is not gemma4_vision"
                ));
            }
        }
        match str_of(c, "hidden_activation")?.as_deref() {
            Some("gelu_pytorch_tanh") => {}
            other => {
                return Err(format!(
                    "config.json: vision_config.hidden_activation must be gelu_pytorch_tanh, got {other:?}"
                ))
            }
        }
        for k in ["attention_bias", "use_clipped_linears", "standardize"] {
            if flag(k)? {
                return Err(format!("config.json: vision_config.{k} is not supported"));
            }
        }
        match c.get("attention_dropout") {
            None => {}
            Some(Json::Num { value, .. }) if *value == 0.0 => {}
            Some(v) => {
                return Err(format!(
                    "config.json: vision_config.attention_dropout must be 0, got {v:?}"
                ))
            }
        }

        let hidden = uint_of(c, "hidden_size")?;
        let layers = uint_of(c, "num_hidden_layers")?;
        let heads = uint_of(c, "num_attention_heads")?;
        let kv_heads = uint_of(c, "num_key_value_heads")?;
        let head_dim = uint_of(c, "head_dim")?;
        let intermediate = uint_of(c, "intermediate_size")?;
        let patch_size = uint_of(c, "patch_size")?;
        let pooling_kernel = uint_of(c, "pooling_kernel_size")?;
        let position_embedding_size = uint_of(c, "position_embedding_size")?;
        let eps = num_of(c, "rms_norm_eps")?;
        let text_hidden = uint_of(text_cfg, "hidden_size")?;
        for (k, v) in [
            ("hidden_size", hidden),
            ("num_hidden_layers", layers),
            ("num_attention_heads", heads),
            ("head_dim", head_dim),
            ("intermediate_size", intermediate),
            ("patch_size", patch_size),
            ("position_embedding_size", position_embedding_size),
            ("text_config.hidden_size", text_hidden),
        ] {
            if v == 0 {
                return Err(format!("config.json: {k} is zero"));
            }
        }
        if kv_heads != heads {
            return Err(format!(
                "config.json: num_key_value_heads {kv_heads} differs from num_attention_heads {heads} \
                 (the vision tower is multi-head attention)"
            ));
        }
        if !matches!(head_dim, 32 | 64 | 256 | 512) {
            return Err(format!(
                "config.json: vision_config.head_dim {head_dim} has no attention kernel (32, 64, 256 or 512)"
            ));
        }
        if head_dim % 4 != 0 {
            return Err(format!(
                "config.json: vision_config.head_dim {head_dim} must be divisible by 4 (two axes, rotated in pairs)"
            ));
        }
        // `Gemma4VisionPooler` returns its padding mask as the validity mask
        // when it does not pool, which inverts it: only a real pool is supported.
        if pooling_kernel < 2 {
            return Err(format!(
                "config.json: pooling_kernel_size must be at least 2, got {pooling_kernel}"
            ));
        }
        if !(eps > 0.0 && eps.is_finite()) {
            return Err(format!("config.json: rms_norm_eps must be positive, got {eps}"));
        }
        let rope = c
            .get("rope_parameters")
            .ok_or("config.json: vision_config is missing \"rope_parameters\"")?;
        match str_of(rope, "rope_type")?.as_deref() {
            Some("axial") => {}
            other => {
                return Err(format!(
                    "config.json: vision_config.rope_parameters.rope_type must be \"axial\", got {other:?}"
                ))
            }
        }
        let theta = num_of(rope, "rope_theta")?;
        if !(theta > 0.0 && theta.is_finite()) {
            return Err("config.json: vision_config.rope_parameters.rope_theta must be positive".into());
        }
        Ok(Self {
            hidden,
            layers,
            heads,
            head_dim,
            intermediate,
            patch_size,
            pooling_kernel,
            position_embedding_size,
            rms_norm_eps: eps as f32,
            rope_theta: theta as f32,
            text_hidden,
        })
    }

    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Host side: validation and the pooling plan
// ---------------------------------------------------------------------------

/// One image as the processor produced it.
#[derive(Clone, Copy, Debug)]
pub struct ImageInput<'a> {
    /// `[P, 3 * patch_size²]` f32 in `[0, 1]`: the patches, padding included.
    pub pixel_values: &'a [f32],
    /// `[P, 2]` i32 `(x, y)` patch positions; padding patches are `(-1, -1)`
    /// and must be a suffix.
    pub position_ids: &'a [i32],
}

/// The pooler's mapping from patches to output rows for one image
/// ([`plan_pooling`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolPlan {
    /// The cell id of each output row, ascending.
    pub cells: Vec<u32>,
    /// `cells.len() + 1` offsets into `members`.
    pub offsets: Vec<u32>,
    /// Patch indices (into the image's real patches), ascending within a cell.
    pub members: Vec<u32>,
}

/// The pooler's cells for an image of `positions.len() / 2` patches, the first
/// `valid` of them real (`_avg_pool_by_positions`).
///
/// `output_length = P / k²` must satisfy `k² * output_length == P`; a patch at
/// `(x, y)` goes to cell `x/k + (max_x/k) * (y/k)` with `max_x = max(x) + 1`
/// over the real patches, and a cell at or past `output_length` is an error
/// (transformers' `one_hot` raises). The output rows are the cells that hold a
/// patch, in cell order, plus cell 0 when there is padding (transformers
/// clamps a padding position to `(0, 0)`, so cell 0 always counts it).
pub fn plan_pooling(positions: &[i32], valid: usize, kernel: u32) -> Result<PoolPlan, String> {
    const WHAT: &str = "plan_pooling";
    let total = positions.len() / 2;
    if positions.len() % 2 != 0 || valid == 0 || valid > total {
        return Err(format!(
            "{WHAT}: {} position values with {valid} real patches",
            positions.len()
        ));
    }
    let k = i64::from(kernel);
    let kk = (k * k) as usize;
    if kernel < 2 || total % kk != 0 || total / kk == 0 {
        return Err(format!(
            "{WHAT}: cannot pool {total} patches with a {kernel}x{kernel} kernel: k² times the output length \
             must equal the patch count"
        ));
    }
    let length = total / kk;
    let mut max_x = 0i64;
    for (i, xy) in positions[..2 * valid].chunks_exact(2).enumerate() {
        if xy[0] < 0 || xy[1] < 0 {
            return Err(format!("{WHAT}: real patch {i} has position ({}, {})", xy[0], xy[1]));
        }
        max_x = max_x.max(i64::from(xy[0]));
    }
    let row_cells = (max_x + 1) / k;
    let cell_of = |i: usize| -> i64 {
        let (x, y) = (i64::from(positions[2 * i]), i64::from(positions[2 * i + 1]));
        x / k + row_cells * (y / k)
    };
    let mut counts = vec![0u32; length];
    let mut occupied = vec![false; length];
    // Padding clamps to (0, 0): cell 0 is occupied whenever there is padding.
    occupied[0] = valid < total;
    for i in 0..valid {
        let cell = cell_of(i);
        if cell < 0 || cell as usize >= length {
            return Err(format!(
                "{WHAT}: patch {i} at ({}, {}) falls in cell {cell}, outside the {length} output cells",
                positions[2 * i],
                positions[2 * i + 1]
            ));
        }
        counts[cell as usize] += 1;
        occupied[cell as usize] = true;
    }
    let cells: Vec<u32> = (0..length as u32).filter(|&c| occupied[c as usize]).collect();
    let mut offsets = Vec::with_capacity(cells.len() + 1);
    let mut row_of = vec![u32::MAX; length];
    let mut run = 0u32;
    for (row, &c) in cells.iter().enumerate() {
        offsets.push(run);
        row_of[c as usize] = row as u32;
        run += counts[c as usize];
    }
    offsets.push(run);
    let mut cursor = offsets.clone();
    let mut members = vec![0u32; valid];
    for i in 0..valid {
        let row = row_of[cell_of(i) as usize] as usize;
        members[cursor[row] as usize] = i as u32;
        cursor[row] += 1;
    }
    Ok(PoolPlan {
        cells,
        offsets,
        members,
    })
}

/// An image checked and reduced to what the forward needs.
struct Prepared {
    /// Real patches.
    n: usize,
    plan: PoolPlan,
}

impl Prepared {
    fn rows_out(&self) -> usize {
        self.plan.cells.len()
    }
}

/// Greedy forwards over images sorted longest first: contiguous runs of at
/// most [`MAX_BATCH_PATCHES`] padded patches and [`MAX_IMAGES`] images, so a
/// big image does not pad small ones to its length. An image over the budget
/// runs alone.
fn pack_images(valid: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..valid.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(valid[i]));
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match runs.last_mut() {
            Some(run) if run.len() < MAX_IMAGES && (run.len() + 1) * valid[run[0]] <= MAX_BATCH_PATCHES => run.push(i),
            _ => runs.push(vec![i]),
        }
    }
    runs
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// The projections (`w_*`, `gate_up`, `down`) are f32 under
/// [`GemmOperands::ExactF32`] and bf16 under [`GemmOperands::Bf16`]; the norms
/// are f32 under both.
struct Layer {
    input_norm: GpuBuffer,
    post_attn_norm: GpuBuffer,
    pre_ffn_norm: GpuBuffer,
    post_ffn_norm: GpuBuffer,
    /// `[hidden, 3 * heads * head_dim]`: `q | k | v`.
    w_qkv: Tensor,
    /// `[heads * head_dim, hidden]`.
    w_o: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    /// `[hidden, 2 * intermediate]`: `gate | up`.
    gate_up: Tensor,
    /// `[intermediate, hidden]`.
    down: Tensor,
}

/// A loaded vision tower and multimodal embedder: weights on the device in the
/// layouts the kernels read.
pub struct EmbedGemma2VisionModel {
    cfg: VisionConfig,
    rt: Arc<GpuRuntime>,
    /// `[patch_dim, hidden]`.
    patch_proj: Tensor,
    /// `[2, position_embedding_size, hidden]` bf16, as the checkpoint holds it.
    pos_table: Tensor,
    layers: Vec<Layer>,
    /// `[hidden, text_hidden]`.
    embed_proj: Tensor,
    /// `[hidden]` ones: the embedder's weightless norm through `rms_norm_f32`
    /// (`rms_norm_bf16` under [`GemmOperands::Bf16`]).
    ones: GpuBuffer,
    /// How the projections are held and the GEMMs run
    /// ([`EmbedGemma2VisionModel::load_with`]).
    operands: GemmOperands,
}

/// Per-layer residuals and the pooled pre-embedder output of an
/// [`EmbedGemma2VisionModel::encode_images`] call run with `trace`. Only real
/// patches (and real output rows) are kept.
#[derive(Clone, Debug, Default)]
pub struct VisionTrace {
    /// Per image: the patch embedder's output, `[n_i, hidden]`.
    pub patch_embed: Vec<Vec<f32>>,
    /// `layers[l][i]`: image `i`'s residual stream after layer `l`, `[n_i, hidden]`.
    pub layers: Vec<Vec<Vec<f32>>>,
    /// Per image: the pooler's output (after `* sqrt(hidden)`), `[S_i, hidden]`.
    pub pooled: Vec<Vec<f32>>,
}

/// What [`EmbedGemma2VisionModel::encode_images`] returns.
#[derive(Clone, Debug)]
pub struct VisionOutput {
    /// Per image: its soft tokens, `[S_i, text_hidden]`, in the order the text
    /// encoder substitutes them for the image's placeholders.
    pub soft: Vec<Vec<f32>>,
    /// `S_i`: soft tokens per image.
    pub tokens: Vec<usize>,
    /// Real patches per image (`n_i`).
    pub patches: Vec<usize>,
    /// With `trace`; empty otherwise.
    pub trace: VisionTrace,
    /// How many forwards the call was split into (see [`MAX_BATCH_PATCHES`]).
    pub forwards: u32,
}

/// Every activation of a call's forwards, allocated once for its largest
/// forward and reused by each (a forward over fewer rows works on a prefix of
/// every buffer). None of it is zeroed on the host
/// ([`GpuRuntime::alloc_tensor_unzeroed`]): every element a forward reads, an
/// earlier kernel of the same forward wrote (the patch GEMM covers every row,
/// the K/V writer every position `< seq`, the attention every query row), and
/// attention reads no key at or past its image's length.
/// `tests/embedgemma2_vision_tiny.rs` runs the forward with every unzeroed
/// allocation poisoned to NaN.
struct Acts {
    rows: usize,
    nb: usize,
    pooled_rows: usize,
    resid: Tensor,
    /// The normed input of the `q|k|v` and `gate|up` GEMMs: f32, or bf16 under
    /// `Bf16` (the norms write it so).
    x: Tensor,
    qkv: Tensor,
    /// The attention operands q, k and normed v: f32, or bf16 under `Bf16` (the
    /// axial kernel stores them so, with no cast pass).
    q: Tensor,
    /// K and normed V, `[batch, capacity, heads, head_dim]`: the capacity both
    /// the K/V writer and [`encoder_attn`] derive from the buffer's size.
    k: Tensor,
    v: Tensor,
    /// The attention output; then, under [`GemmOperands::ExactF32`], once the
    /// output projection has read it, the MLP's `gelu_tanh(gate) * up` (sized
    /// for the wider of the two). Under `Bf16` the MLP's lives in
    /// [`Acts::wide`] and this is the attention output's width only.
    attn: Tensor,
    y: Tensor,
    gate_up: Tensor,
    pooled: Tensor,
    /// The embedder's normed input: f32, or bf16 under `Bf16`.
    normed: Tensor,
    /// Under [`GemmOperands::Bf16`]: `[rows, max(heads * head_dim,
    /// intermediate)]` bf16, the rounded attention output and then, once the
    /// output projection has read it, the MLP's `gelu_tanh(gate) * up`
    /// (written bf16 by the gate kernel). `None` under `ExactF32`, whose
    /// `attn` holds both in turn.
    wide: Option<Tensor>,
}

/// What one forward alone owns, because the host writes it before the forward
/// runs and reads it after: each forward gets its own set, kept alive until
/// the call's one wait.
struct ForwardIo {
    /// `[rows, patch_dim]` scaled pixels, padded rows zero: f32, or under
    /// `Bf16` rounded to bf16 on the host as they are written (the GPU's
    /// rounding, [`f32_to_bf16_bits_gpu`]), the patch GEMM's operand.
    pix: Tensor,
    /// `[rows, 2]` i32 positions as u32 bits, padded rows `(-1, -1)`.
    positions: GpuBuffer,
    /// `[batch]` real patches per image.
    lens: GpuBuffer,
    /// The concatenated pool plans: `[pooled_rows + 1]`, offsets into `members`.
    offsets: GpuBuffer,
    members: GpuBuffer,
    n_members: u32,
    /// `[pooled_rows, text_hidden]`: the soft tokens.
    soft: Tensor,
}

/// A forward's traced buffers, whole (the caller splits them per image).
struct RawTrace {
    embed: Vec<f32>,
    layers: Vec<Vec<f32>>,
    pooled: Vec<f32>,
}

/// What a queued [`EmbedGemma2VisionModel::queue_images`] call keeps alive:
/// each forward's device-written output (its soft tokens), and what the host
/// needs to split that output per image. The activations and each forward's
/// host-written inputs are not here: they were handed to the runtime's
/// deferred release when the call was queued, so the caller's wait recycles
/// them.
struct PendingState {
    /// Per forward: the images it ran (indices into the call) and its soft
    /// tokens, `[pooled_rows, text_hidden]`, the images' rows end to end.
    outs: Vec<(Vec<usize>, Tensor)>,
    raw: Vec<Option<RawTrace>>,
    prepared: Vec<Prepared>,
    valid: Vec<usize>,
    trace: bool,
}

/// An [`EmbedGemma2VisionModel::queue_images`] call whose forwards are queued
/// and not yet waited for. [`Self::collect`] waits for them (if the caller has
/// not) and reads the soft tokens; dropping it without collecting waits for the
/// queued work and discards the result, since Metal 4 command buffers do not
/// retain the buffers they bind. It holds only each forward's output: the
/// activations and the forwards' uploaded inputs are the runtime's, released at
/// the wait, so a call followed by another holds one activation set.
pub struct PendingImages {
    rt: Arc<GpuRuntime>,
    hidden: usize,
    n_layers: usize,
    text_hidden: usize,
    state: Option<PendingState>,
}

impl PendingImages {
    /// How many forwards were queued.
    pub fn forwards(&self) -> u32 {
        self.state.as_ref().map_or(0, |s| s.outs.len() as u32)
    }

    /// Hand the call's soft tokens to a caller that keeps them on the device:
    /// per image, in the order of the call, the rows of its forward's output
    /// that are its soft tokens (one segment, since the pooler writes an
    /// image's rows end to end and nothing else). Consumes the call **without
    /// waiting** (so the caller owes the one wait, after queuing whatever
    /// reads these rows); the segments hold the output buffers, which the
    /// runtime keeps until that wait has seen the GPU finish.
    pub(crate) fn into_device(mut self) -> Vec<SoftSegment> {
        let state = self.state.take().expect("into_device consumes the pending call");
        let mut segments: Vec<Option<SoftSegment>> = vec![None; state.valid.len()];
        for (chunk, soft) in &state.outs {
            debug_assert_eq!(soft.byte_offset(), 0, "a forward's soft tokens start their buffer");
            let mut row = 0u32;
            for &i in chunk {
                let rows = state.prepared[i].rows_out() as u32;
                segments[i] = Some(SoftSegment {
                    buf: soft.buffer.clone(),
                    row,
                    rows,
                });
                row += rows;
            }
        }
        segments
            .into_iter()
            .map(|s| s.expect("every image is in one forward"))
            .collect()
    }

    /// The call's [`VisionOutput`]. Waits for the queued forwards first if
    /// nothing has (the wait is a no-op when the caller already waited).
    pub fn collect(mut self) -> Result<VisionOutput, String> {
        let state = self.state.take().expect("collect consumes the pending call");
        let PendingState {
            outs,
            raw,
            prepared,
            valid,
            trace,
        } = state;
        // The buffers in `outs` must outlive the queued work, so a failed
        // wait still holds them until the end of this function.
        self.rt.synchronize()?;
        let (text_h, h) = (self.text_hidden, self.hidden);
        let n_images = valid.len();
        let n_layers = self.n_layers;
        let mut soft = vec![Vec::new(); n_images];
        let mut out_trace = VisionTrace {
            patch_embed: vec![Vec::new(); if trace { n_images } else { 0 }],
            layers: vec![vec![Vec::new(); if trace { n_images } else { 0 }]; if trace { n_layers } else { 0 }],
            pooled: vec![Vec::new(); if trace { n_images } else { 0 }],
        };
        for (c, (chunk, soft_out)) in outs.iter().enumerate() {
            let seq = valid[chunk[0]];
            let all_soft = soft_out.buffer.try_contents_f32()?;
            let mut out_row = 0usize;
            for (b, &i) in chunk.iter().enumerate() {
                let s = prepared[i].rows_out();
                soft[i] = all_soft[out_row * text_h..(out_row + s) * text_h].to_vec();
                if let Some(t) = raw[c].as_ref() {
                    let n = prepared[i].n;
                    out_trace.patch_embed[i] = t.embed[b * seq * h..(b * seq + n) * h].to_vec();
                    for (l, layer) in t.layers.iter().enumerate() {
                        out_trace.layers[l][i] = layer[b * seq * h..(b * seq + n) * h].to_vec();
                    }
                    out_trace.pooled[i] = t.pooled[out_row * h..(out_row + s) * h].to_vec();
                }
                out_row += s;
            }
        }
        Ok(VisionOutput {
            tokens: prepared.iter().map(Prepared::rows_out).collect(),
            patches: valid,
            soft,
            trace: out_trace,
            forwards: outs.len() as u32,
        })
    }
}

impl Drop for PendingImages {
    fn drop(&mut self) {
        if self.state.is_some() {
            // The result cannot be returned; the wait is for the buffers. A
            // failure leaves the runtime poisoned, which its own checks report.
            let _ = self.rt.synchronize();
        }
    }
}

/// A `[rows, cols]` view over the front of `t`'s buffer.
fn view(t: &Tensor, rows: u32, cols: u32) -> Result<Tensor, String> {
    t.try_view(&[rows as usize, cols as usize], 0)
}

impl EmbedGemma2VisionModel {
    /// Load the vision tower and embedder from `st` (`vision_tower.*` and
    /// `embed_vision.*` of `google/embeddinggemma-2`'s `model.safetensors`).
    /// Every tensor's shape is checked against `cfg`; a missing one is an
    /// error, and so is any tensor under either prefix the forward does not
    /// read.
    ///
    /// This is [`GemmOperands::ExactF32`] ([`Self::load_with`]): the
    /// projections widened to f32, the GEMMs exact f32.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: VisionConfig) -> Result<Self, String> {
        Self::load_with(rt, st, cfg, GemmOperands::ExactF32)
    }

    /// [`Self::load`] with the operand precision chosen (the module docs,
    /// Precision).
    ///
    /// * [`GemmOperands::ExactF32`]: every projection widened to f32 on the
    ///   device, every GEMM exact f32 (transformers' fp32 forward up to
    ///   operation order). `encode_images` refuses a runtime whose relaxed f32
    ///   GEMMs are on.
    /// * [`GemmOperands::Bf16`]: every projection kept as the checkpoint stores
    ///   it, bf16, with no f32 copy, and every GEMM run with bf16 operands and
    ///   f32 accumulation; the attention runs on bf16 tiles
    ///   ([`encoder_attn_bf16`]). The checkpoint's projections must be bf16 (a
    ///   tensor stored in another dtype is refused by name rather than
    ///   rounded), the device needs TensorOps, and the head dim must be one the
    ///   bf16 attention is compiled for (64, 256 or 512). `encode_images`
    ///   accepts a runtime with relaxed f32 GEMMs on, which no GEMM of this
    ///   forward uses.
    pub fn load_with(
        rt: &Arc<GpuRuntime>,
        st: &SafeTensors,
        cfg: VisionConfig,
        operands: GemmOperands,
    ) -> Result<Self, String> {
        let proj = match operands {
            GemmOperands::ExactF32 => Precision::F32,
            GemmOperands::Bf16 => {
                if !rt.has_tensorops() {
                    return Err(
                        "EmbedGemma2VisionModel::load_with: bf16 operands need TensorOps, which this device lacks"
                            .into(),
                    );
                }
                if !encoder_attn_bf16_supports(cfg.head_dim) {
                    return Err(format!(
                        "EmbedGemma2VisionModel::load_with: bf16 operands have no attention kernel for head dim {} \
                         (64, 256 and 512 are compiled)",
                        cfg.head_dim
                    ));
                }
                Precision::Bf16
            }
        };
        let tower = Loader::new(rt, st, "vision_tower.");
        let embed = Loader::new(rt, st, "embed_vision.");
        let (h, inter, patch_dim, text_h) = (
            cfg.hidden as usize,
            cfg.intermediate as usize,
            cfg.patch_dim() as usize,
            cfg.text_hidden as usize,
        );
        let qw = (cfg.heads * cfg.head_dim) as usize;
        let d = cfg.head_dim as usize;

        let patch_proj = tower.linear(&[("patch_embedder.input_proj.weight", h)], patch_dim, proj)?;
        let size = cfg.position_embedding_size as usize;
        // Read straight into the device table: no host copy.
        let pos_table = rt.alloc_tensor_bf16_hot(&[2, size, h])?;
        tower.bf16_into("patch_embedder.position_embedding_table", &[2, size, h], &pos_table)?;

        let mut layers = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("encoder.layers.{i}.{s}");
            layers.push(Layer {
                input_norm: tower.norm(&p("input_layernorm.weight"), h)?,
                post_attn_norm: tower.norm(&p("post_attention_layernorm.weight"), h)?,
                pre_ffn_norm: tower.norm(&p("pre_feedforward_layernorm.weight"), h)?,
                post_ffn_norm: tower.norm(&p("post_feedforward_layernorm.weight"), h)?,
                w_qkv: tower.linear(
                    &[
                        (&p("self_attn.q_proj.linear.weight"), qw),
                        (&p("self_attn.k_proj.linear.weight"), qw),
                        (&p("self_attn.v_proj.linear.weight"), qw),
                    ],
                    h,
                    proj,
                )?,
                w_o: tower.linear(&[(&p("self_attn.o_proj.linear.weight"), h)], qw, proj)?,
                q_norm: tower.norm(&p("self_attn.q_norm.weight"), d)?,
                k_norm: tower.norm(&p("self_attn.k_norm.weight"), d)?,
                gate_up: tower.linear(
                    &[
                        (&p("mlp.gate_proj.linear.weight"), inter),
                        (&p("mlp.up_proj.linear.weight"), inter),
                    ],
                    h,
                    proj,
                )?,
                down: tower.linear(&[(&p("mlp.down_proj.linear.weight"), h)], inter, proj)?,
            });
        }
        let embed_proj = embed.linear(&[("embedding_projection.weight", text_h)], h, proj)?;
        tower.refuse_unread()?;
        embed.refuse_unread()?;
        let ones = tower.buf(&vec![1.0f32; h])?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            patch_proj,
            pos_table,
            layers,
            embed_proj,
            ones,
            operands,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// The operand precision this model was loaded with.
    pub fn operands(&self) -> GemmOperands {
        self.operands
    }

    /// Bytes of weights held on the device: the position table, every
    /// projection in the dtype the mode holds it in, and the norms (f32). The
    /// activations are not counted.
    pub fn weight_bytes(&self) -> usize {
        let t = |t: &Tensor| t.nbytes_logical();
        let layers: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.w_qkv, &l.w_o, &l.gate_up, &l.down]
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
                    ]
                    .into_iter()
                    .map(GpuBuffer::nbytes)
                    .sum::<usize>()
            })
            .sum();
        t(&self.patch_proj) + t(&self.pos_table) + t(&self.embed_proj) + self.ones.nbytes() + layers
    }

    /// The runtime this model was loaded on.
    pub(crate) fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }

    /// Check one image and reduce it to its real patches and pooling plan.
    fn prepare(&self, i: usize, img: &ImageInput<'_>) -> Result<Prepared, String> {
        let what = format!("EmbedGemma2VisionModel::encode_images: image {i}");
        let cfg = &self.cfg;
        let pd = cfg.patch_dim() as usize;
        if img.position_ids.len() % 2 != 0 || img.position_ids.is_empty() {
            return Err(format!(
                "{what}: position_ids has {} values; need [P, 2] with P > 0",
                img.position_ids.len()
            ));
        }
        let total = img.position_ids.len() / 2;
        if img.pixel_values.len() != total * pd {
            return Err(format!(
                "{what}: pixel_values has {} values; need [{total}, {pd}] for {total} patches",
                img.pixel_values.len()
            ));
        }
        // Padding is (-1, -1) in both coordinates, and a suffix.
        let pad = |t: usize| img.position_ids[2 * t] == -1 && img.position_ids[2 * t + 1] == -1;
        let valid = (0..total).position(pad).unwrap_or(total);
        if valid == 0 {
            return Err(format!("{what}: no real patches (all {total} are padding)"));
        }
        if let Some(t) = (valid..total).find(|&t| !pad(t)) {
            return Err(format!(
                "{what}: patch {t} is real but follows the padding that starts at patch {valid}; \
                 padding must be a suffix"
            ));
        }
        if valid > MAX_BATCH_PATCHES {
            return Err(format!(
                "{what}: {valid} real patches; the limit is {MAX_BATCH_PATCHES}"
            ));
        }
        let size = i64::from(cfg.position_embedding_size);
        for t in 0..valid {
            let (x, y) = (img.position_ids[2 * t], img.position_ids[2 * t + 1]);
            if !(0..size).contains(&i64::from(x)) || !(0..size).contains(&i64::from(y)) {
                return Err(format!(
                    "{what}: patch {t} at ({x}, {y}) is outside the position table [0, {size})"
                ));
            }
        }
        if let Some(j) = img.pixel_values[..valid * pd].iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "{what}: pixel value {j} (patch {}) is {}",
                j / pd,
                img.pixel_values[j]
            ));
        }
        let plan = plan_pooling(img.position_ids, valid, cfg.pooling_kernel).map_err(|e| format!("{what}: {e}"))?;
        Ok(Prepared { n: valid, plan })
    }

    /// Everything [`Self::encode_images`] checks before it touches the device,
    /// without doing any GPU work, no allocation and no wait: the batch is
    /// non-empty and within [`MAX_IMAGES`], the runtime's f32 GEMMs are exact,
    /// and each image is well formed ([`ImageInput`]). Returns the number of soft
    /// tokens each image will produce, which is what a caller needs to lay out
    /// the placeholders of its text sequence before the towers have run.
    pub fn check_images(&self, images: &[ImageInput<'_>]) -> Result<Vec<usize>, String> {
        let prepared = self.check_call("EmbedGemma2VisionModel::check_images", images)?;
        Ok(prepared.iter().map(Prepared::rows_out).collect())
    }

    /// [`Self::check_images`]'s checks, keeping what they computed.
    fn check_call(&self, what: &str, images: &[ImageInput<'_>]) -> Result<Vec<Prepared>, String> {
        if images.is_empty() {
            return Err(format!("{what}: empty batch"));
        }
        if images.len() > MAX_IMAGES {
            return Err(format!(
                "{what}: {} images exceeds the batch limit {MAX_IMAGES}",
                images.len()
            ));
        }
        // No GEMM of the bf16 forward is an f32 one, so the switch does not
        // touch it.
        if self.operands == GemmOperands::ExactF32 && self.rt.f32_gemm_runs_relaxed() {
            return Err(format!(
                "{what}: the forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        images.iter().enumerate().map(|(i, img)| self.prepare(i, img)).collect()
    }

    /// Encode a batch of images to soft tokens. Each [`ImageInput`] is one
    /// image's processor tensors; the result's `soft[i]` is its `[S_i,
    /// text_hidden]` soft tokens in placeholder order. An image's tokens do
    /// not depend on what else is in the call beyond GEMM tiling (padded keys
    /// are masked, padded rows are zero).
    ///
    /// The call runs as one or more forwards (images sorted longest first, cut
    /// into runs of at most [`MAX_BATCH_PATCHES`] padded patches) queued
    /// behind one wait, on a runtime whatever its
    /// [`GpuRuntime::set_async_encode`] flag says (the forwards are queued
    /// under a [`GpuRuntime::async_scope`]). `trace` also returns the patch
    /// embedder's output, the residual stream after each layer and the pooled
    /// output. This is [`Self::queue_images`], the wait, and
    /// [`PendingImages::collect`].
    pub fn encode_images(&self, images: &[ImageInput<'_>], trace: bool) -> Result<VisionOutput, String> {
        let pending = self.queue_impl("EmbedGemma2VisionModel::encode_images", images, trace)?;
        self.rt.synchronize()?;
        pending.collect()
    }

    /// [`Self::encode_images`] up to the wait: the call is checked, its forwards
    /// are queued, and the host does not wait for them. The returned
    /// [`PendingImages`] holds the forwards' outputs, and the runtime holds
    /// every other buffer the queued work binds until the wait; a caller that
    /// has other work to queue behind it (the text encoder, another tower) does
    /// so, waits once, and then [`PendingImages::collect`]s. Nothing is queued
    /// when the call is refused, and a call that fails while queuing waits for
    /// what it queued before it returns the error.
    pub fn queue_images(&self, images: &[ImageInput<'_>], trace: bool) -> Result<PendingImages, String> {
        self.queue_impl("EmbedGemma2VisionModel::queue_images", images, trace)
    }

    fn queue_impl(&self, what: &str, images: &[ImageInput<'_>], trace: bool) -> Result<PendingImages, String> {
        let prepared = self.check_call(what, images)?;
        let valid: Vec<usize> = prepared.iter().map(|p| p.n).collect();
        let chunks = pack_images(&valid);
        let dims = |c: &[usize]| (c.len(), valid[c[0]]);
        let rows = chunks.iter().map(|c| c.len() * valid[c[0]]).max().unwrap_or(0);
        let nb = chunks.iter().map(Vec::len).max().unwrap_or(0);
        let pooled_rows = chunks
            .iter()
            .map(|c| c.iter().map(|&i| prepared[i].rows_out()).sum::<usize>())
            .max()
            .unwrap_or(0);
        let acts = self.acts(nb, rows, pooled_rows)?;

        // The forwards are queued back to back and the host waits once, after
        // the last (the caller's wait, or `encode_images`'). Each keeps its own
        // host-written inputs and host-read output, held in `pending` until
        // then (Metal 4 command buffers do not retain what they bind),
        // including a forward that fails midway.
        let scope = self.rt.async_scope();
        let mut pending: Vec<(Vec<usize>, ForwardIo)> = Vec::with_capacity(chunks.len());
        let mut raw: Vec<Option<RawTrace>> = Vec::with_capacity(chunks.len());
        let mut queued: Result<(), String> = Ok(());
        for chunk in chunks {
            let (n_img, seq) = dims(&chunk);
            let io = match self.forward_io(images, &prepared, &chunk, seq) {
                Ok(io) => io,
                Err(e) => {
                    queued = Err(e);
                    break;
                }
            };
            pending.push((chunk, io));
            let (_, io) = pending.last().expect("just pushed");
            match self.forward(&acts, io, n_img as u32, seq as u32, trace) {
                Ok(t) => raw.push(t),
                Err(e) => {
                    queued = Err(e);
                    break;
                }
            }
        }
        // Hand everything the GPU only reads to the runtime *before* the wait:
        // the activations and each forward's uploaded inputs. A dropped pooled
        // buffer is parked in the runtime's deferred-release queue, which holds
        // it until a waiting commit has seen the GPU finish and that wait then
        // returns it to the pool's freelist (see `EmbedGemma2Model::encode`).
        // Held until `collect` instead, they reached the freelist only at the
        // next commit, after the next call had allocated its own set. Only
        // each forward's soft tokens stay, to be read (or copied from) after
        // the wait; a forward that failed midway is released the same way.
        let outs: Vec<(Vec<usize>, Tensor)> = pending.into_iter().map(|(chunk, io)| (chunk, io.soft)).collect();
        drop(acts);
        let state = PendingState {
            outs,
            raw,
            prepared,
            valid,
            trace,
        };
        let out = PendingImages {
            rt: Arc::clone(&self.rt),
            hidden: self.cfg.hidden as usize,
            n_layers: self.layers.len(),
            text_hidden: self.cfg.text_hidden as usize,
            state: Some(state),
        };
        drop(scope);
        // On an error `out` is dropped, which waits for what was queued (see
        // its `Drop`).
        queued?;
        Ok(out)
    }

    /// This forward's own host-written inputs and host-read output for the
    /// images `chunk` (indices into `images`), padded to `seq` rows each,
    /// written without waiting for the GPU ([`GpuRuntime::alloc_buffer_from_u32`]).
    fn forward_io(
        &self,
        images: &[ImageInput<'_>],
        prepared: &[Prepared],
        chunk: &[usize],
        seq: usize,
    ) -> Result<ForwardIo, String> {
        const WHAT: &str = "EmbedGemma2VisionModel::forward";
        let rt = &self.rt;
        let pd = self.cfg.patch_dim() as usize;
        let rows = chunk.len() * seq;
        if seq == 0 || rows > MAX_BATCH_PATCHES.max(seq) {
            return Err(format!(
                "{WHAT}: {} images padded to {seq} patches is {rows} rows; need 1..={MAX_BATCH_PATCHES}",
                chunk.len()
            ));
        }
        // 2 * (pixel - 0.5), in f32 and in that order, as the patch embedder
        // does; padded rows stay zero, so the projection of one is zero.
        let mut pix = vec![0u32; rows * pd];
        let mut pos = vec![u32::MAX; rows * 2]; // -1, -1
        let mut lens = Vec::with_capacity(chunk.len());
        let mut offsets = vec![0u32];
        let mut members: Vec<u32> = Vec::new();
        for (b, &i) in chunk.iter().enumerate() {
            let (img, p) = (&images[i], &prepared[i]);
            for (dst, &v) in pix[b * seq * pd..(b * seq + p.n) * pd]
                .iter_mut()
                .zip(&img.pixel_values[..p.n * pd])
            {
                *dst = (2.0f32 * (v - 0.5f32)).to_bits();
            }
            for (dst, &v) in pos[2 * b * seq..2 * (b * seq + p.n)]
                .iter_mut()
                .zip(&img.position_ids[..2 * p.n])
            {
                *dst = v as u32;
            }
            lens.push(p.n as u32);
            let base = members.len() as u32;
            let row0 = (b * seq) as u32;
            members.extend(p.plan.members.iter().map(|&m| row0 + m));
            offsets.extend(p.plan.offsets[1..].iter().map(|&o| base + o));
        }
        let pooled_rows = offsets.len() - 1;
        let pix = match self.operands {
            GemmOperands::ExactF32 => {
                Tensor::from_buffer(rt, rt.alloc_buffer_from_u32(&pix)?, &[rows, pd], DType::F32, 0)?
            }
            // The patch GEMM's operand is bf16: round the scaled pixels as the
            // GPU's cast would, two to a word (little endian, the first in the
            // low half), so there is no cast pass and half the upload.
            GemmOperands::Bf16 => {
                let words: Vec<u32> = pix
                    .chunks(2)
                    .map(|c| {
                        let half = |bits: u32| u32::from(f32_to_bf16_bits_gpu(f32::from_bits(bits)));
                        half(c[0]) | c.get(1).map_or(0, |&b| half(b) << 16)
                    })
                    .collect();
                Tensor::from_buffer(rt, rt.alloc_buffer_from_u32(&words)?, &[rows, pd], DType::BF16, 0)?
            }
        };
        Ok(ForwardIo {
            pix,
            positions: rt.alloc_buffer_from_u32(&pos)?,
            lens: rt.alloc_buffer_from_u32(&lens)?,
            offsets: rt.alloc_buffer_from_u32(&offsets)?,
            n_members: members.len() as u32,
            members: rt.alloc_buffer_from_u32(&members)?,
            soft: rt.alloc_tensor_unzeroed(&[pooled_rows, self.cfg.text_hidden as usize], DType::F32)?,
        })
    }

    /// Queue one forward over `nb` images of `seq` rows each (`io` from
    /// [`Self::forward_io`] for the same chunk) in `a`, without waiting for
    /// it. The soft tokens are in `io.soft` once the caller has synchronized.
    /// With `trace` the forward waits per layer and returns the traced buffers.
    fn forward(&self, a: &Acts, io: &ForwardIo, nb: u32, seq: u32, trace: bool) -> Result<Option<RawTrace>, String> {
        const WHAT: &str = "EmbedGemma2VisionModel::forward";
        let rt = &self.rt;
        let cfg = &self.cfg;
        let rows = nb as usize * seq as usize;
        let pooled_rows = io.soft.shape()[0] as u32;
        if rows > a.rows || nb as usize > a.nb || pooled_rows as usize > a.pooled_rows {
            return Err(format!(
                "{WHAT}: {rows} rows of {nb} images ({pooled_rows} pooled) exceed the activations' {} rows of {} ({} pooled)",
                a.rows, a.nb, a.pooled_rows
            ));
        }
        let (h, rows_u) = (cfg.hidden, rows as u32);
        let eps = cfg.rms_norm_eps;
        let qw = cfg.heads * cfg.head_dim;
        let inter = cfg.intermediate;
        let at = |t: &Tensor, cols: u32| view(t, rows_u, cols);
        let (resid, y) = (at(&a.resid, h)?, at(&a.y, h)?);
        // Under `Bf16`, each GEMM input that has a bf16 staging reads that
        // instead of its f32 buffer: the norms (`rms_norm_bf16`) and the GELU
        // gate write bf16 directly, at no extra dispatch, and one cast pass
        // per layer covers the attention output (`encoder_attn` writes f32
        // only). The rounded copies are made once, into `a.bf16`'s buffers.
        let bf = a.wide.as_ref();
        let x_in = at(&a.x, h)?;
        let (qkv, gate_up) = (at(&a.qkv, 3 * qw)?, at(&a.gate_up, 2 * inter)?);
        let mid = match bf {
            Some(wide) => at(wide, inter)?,
            None => at(&a.attn, inter)?,
        };
        // The normed input of the GEMMs that read `x_in`: bf16 straight from
        // the norm under `Bf16`.
        let norm_x_in = |weight: &GpuBuffer| match bf {
            Some(_) => rms_norm_bf16(rt, &resid.buffer, weight, &x_in.buffer, rows_u, h, eps),
            None => rms_norm_f32(rt, &resid.buffer, weight, &x_in.buffer, rows_u, h, eps),
        };

        // Patch embedder: the projection of the scaled pixels (zero rows give
        // zero rows), plus the position embeddings (zero on padding).
        project(self.operands, &io.pix, &self.patch_proj, &resid)?;
        patch_position_embed_add(
            rt,
            &resid.buffer,
            &self.pos_table.buffer,
            &io.positions,
            rows_u,
            h,
            cfg.position_embedding_size,
        )?;
        let mut raw = trace.then(|| RawTrace {
            embed: Vec::new(),
            layers: Vec::new(),
            pooled: Vec::new(),
        });
        if let Some(t) = raw.as_mut() {
            rt.synchronize()?;
            t.embed = resid.buffer.try_contents_f32()?[..rows * h as usize].to_vec();
        }

        for layer in &self.layers {
            // Attention.
            norm_x_in(&layer.input_norm)?;
            project(self.operands, &x_in, &layer.w_qkv, &qkv)?;
            // Q, K and V are in the attention's operand type: f32, or under
            // `Bf16` the writer stores them bf16.
            let rope = if bf.is_some() {
                axial_qkv_norm_rope_bf16
            } else {
                axial_qkv_norm_rope
            };
            rope(
                rt,
                &AxialQkvShape {
                    batch: nb,
                    seq,
                    q_heads: cfg.heads,
                    kv_heads: cfg.heads,
                    head_dim: cfg.head_dim,
                },
                Cols::dense(&qkv.buffer, 3 * qw),
                qw,
                2 * qw,
                &layer.q_norm,
                &layer.k_norm,
                &AttnTargets {
                    q_out: &a.q.buffer,
                    k_cache: &a.k.buffer,
                    v_cache: &a.v.buffer,
                },
                &io.positions,
                cfg.rope_theta,
                eps,
            )?;
            let attend = if bf.is_some() { encoder_attn_bf16 } else { encoder_attn };
            attend(
                rt,
                &a.q.buffer,
                &a.k.buffer,
                &a.v.buffer,
                &a.attn.buffer,
                &io.lens,
                EncoderAttnDims {
                    batch: nb,
                    seq,
                    heads: cfg.heads,
                    heads_kv: cfg.heads,
                    head_dim: cfg.head_dim,
                    window: 0,
                    scale: 1.0,
                },
            )?;
            let attn = at(&a.attn, qw)?;
            match bf {
                Some(wide) => {
                    let attn_in = at(wide, qw)?;
                    cast_f32_to_bf16_into(&attn, &attn_in)?;
                    project(self.operands, &attn_in, &layer.w_o, &y)?;
                }
                None => project(self.operands, &attn, &layer.w_o, &y)?,
            }
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_attn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;

            // MLP.
            norm_x_in(&layer.pre_ffn_norm)?;
            project(self.operands, &x_in, &layer.gate_up, &gate_up)?;
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
            project(self.operands, &mid, &layer.down, &y)?;
            rms_norm_residual_add_f32(rt, &y.buffer, &layer.post_ffn_norm, &resid.buffer, rows_u, h, eps, 1.0)?;
            if let Some(t) = raw.as_mut() {
                rt.synchronize()?;
                t.layers
                    .push(resid.buffer.try_contents_f32()?[..rows * h as usize].to_vec());
            }
        }

        // Pooler: the k x k cell sums, 1/k² each, times sqrt(hidden) (an f32
        // product, as torch forms it); then the embedder.
        let kk = (cfg.pooling_kernel * cfg.pooling_kernel) as f32;
        let (pooled, normed) = (view(&a.pooled, pooled_rows, h)?, view(&a.normed, pooled_rows, h)?);
        let soft = view(&io.soft, pooled_rows, cfg.text_hidden)?;
        pool_by_members(
            rt,
            &resid.buffer,
            &io.offsets,
            &io.members,
            &pooled.buffer,
            pooled_rows,
            rows_u,
            h,
            io.n_members,
            1.0f32 / kk,
            (h as f32).sqrt(),
        )?;
        // `EmbeddingGemma2MultimodalEmbedder`: a weightless RMSNorm (the norm
        // kernel with a weight of ones), then the projection.
        match bf {
            Some(_) => rms_norm_bf16(rt, &pooled.buffer, &self.ones, &normed.buffer, pooled_rows, h, eps)?,
            None => rms_norm_f32(rt, &pooled.buffer, &self.ones, &normed.buffer, pooled_rows, h, eps)?,
        }
        project(self.operands, &normed, &self.embed_proj, &soft)?;
        if let Some(t) = raw.as_mut() {
            rt.synchronize()?;
            t.pooled = pooled.buffer.try_contents_f32()?[..pooled_rows as usize * h as usize].to_vec();
        }
        Ok(raw)
    }

    /// [`Acts`] for forwards of up to `rows` padded rows over up to `nb`
    /// images pooling to up to `pooled_rows` rows.
    fn acts(&self, nb: usize, rows: usize, pooled_rows: usize) -> Result<Acts, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let (r, nb, s) = (rows.max(1), nb.max(1), pooled_rows.max(1));
        let (h, inter) = (cfg.hidden as usize, cfg.intermediate as usize);
        let qw = (cfg.heads * cfg.head_dim) as usize;
        let f32s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::F32);
        let bf16 = self.operands == GemmOperands::Bf16;
        let bf16s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::BF16);
        // A GEMM input or attention operand: bf16 under `Bf16`, else f32.
        let operand = |shape: &[usize]| if bf16 { bf16s(shape) } else { f32s(shape) };
        Ok(Acts {
            rows: r,
            nb,
            pooled_rows: s,
            resid: f32s(&[r, h])?,
            x: operand(&[r, h])?,
            qkv: f32s(&[r, 3 * qw])?,
            q: operand(&[r, qw])?,
            k: operand(&[r, qw])?,
            v: operand(&[r, qw])?,
            attn: f32s(&[r, if bf16 { qw } else { qw.max(inter) }])?,
            y: f32s(&[r, h])?,
            gate_up: f32s(&[r, 2 * inter])?,
            pooled: f32s(&[s, h])?,
            normed: operand(&[s, h])?,
            wide: if bf16 { Some(bf16s(&[r, qw.max(inter)])?) } else { None },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_images, plan_pooling, MAX_BATCH_PATCHES, MAX_IMAGES};

    /// A `w x h` grid of patches in raster order, then `pad` padding patches.
    fn grid(w: i32, h: i32, pad: usize) -> Vec<i32> {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.extend([x, y]);
            }
        }
        v.extend(std::iter::repeat_n(-1, 2 * pad));
        v
    }

    #[test]
    fn a_grid_pools_to_its_cells_in_raster_order() {
        // 6 x 3 real patches padded to 36, k = 3: 36 / 9 = 4 output cells.
        let pos = grid(6, 3, 18);
        let plan = plan_pooling(&pos, 18, 3).unwrap();
        assert_eq!(plan.cells, vec![0, 1], "6x3 patches fill two cells of the 4");
        assert_eq!(plan.offsets, vec![0, 9, 18]);
        assert_eq!(&plan.members[..9], &[0, 1, 2, 6, 7, 8, 12, 13, 14]);
    }

    #[test]
    fn padding_keeps_cell_zero_even_without_a_patch_there() {
        // One real patch at (3, 0) -> cell 1 of 2; the 17 padding patches
        // count toward cell 0, which is kept, empty.
        let mut pos = vec![3, 0];
        pos.extend(std::iter::repeat_n(-1, 34));
        let plan = plan_pooling(&pos, 1, 3).unwrap();
        assert_eq!(plan.cells, vec![0, 1]);
        assert_eq!(plan.offsets, vec![0, 0, 1]);
        assert_eq!(plan.members, vec![0]);
        // No padding: cell 0 is kept because a patch is in it, not forced.
        let plan = plan_pooling(&grid(3, 3, 0), 9, 3).unwrap();
        assert_eq!(plan.cells, vec![0]);
        assert_eq!(plan.members, (0..9).collect::<Vec<u32>>());
    }

    #[test]
    fn plan_pooling_refuses_what_transformers_raises_on() {
        // 10 patches do not divide by k² = 9.
        assert!(plan_pooling(&grid(10, 1, 0), 10, 3).is_err());
        // A cell past the output length: 9 patches, 1 cell, a patch at x = 5.
        assert!(plan_pooling(&[5, 0, 0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2], 9, 3).is_err());
        // Negative real position, no real patches, a kernel under 2.
        assert!(plan_pooling(&[0, -2, -1, -1, -1, -1, -1, -1], 1, 2).is_err());
        assert!(plan_pooling(&grid(2, 2, 0), 0, 2).is_err());
        assert!(plan_pooling(&grid(2, 2, 0), 4, 1).is_err());
        assert!(plan_pooling(&[0, 0, 1], 1, 2).is_err(), "odd value count");
    }

    #[test]
    fn packing_splits_by_budget_and_keeps_every_image_once() {
        let valid = [
            10usize,
            MAX_BATCH_PATCHES / 2,
            3,
            MAX_BATCH_PATCHES / 2,
            MAX_BATCH_PATCHES + 1,
        ];
        let runs = pack_images(&valid);
        let mut seen: Vec<usize> = runs.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        for r in &runs {
            let longest = valid[r[0]];
            assert!(longest == valid[*r.iter().max_by_key(|&&i| valid[i]).unwrap()]);
            assert!(r.len() == 1 || r.len() * longest <= MAX_BATCH_PATCHES, "{r:?}");
            assert!(r.len() <= MAX_IMAGES);
        }
        assert_eq!(runs[0], vec![4], "the oversized image runs alone");
        let many = vec![1usize; MAX_IMAGES * 2 + 1];
        assert!(pack_images(&many).iter().all(|r| r.len() <= MAX_IMAGES));
    }
}
