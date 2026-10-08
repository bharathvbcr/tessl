//! EmbeddingGemma 2's text encoder (`google/embeddinggemma-2`), composed from
//! tessl kernels and loaded straight from its `model.safetensors`.
//!
//! The text path is a 24-layer bidirectional encoder over a Gemma-4-style
//! block, then a mean pool and an L2 normalize: what sentence-transformers'
//! `Transformer -> Pooling(mean, include_prompt) -> Normalize` computes for
//! this checkpoint. The vision and audio towers in the same file are not
//! loaded.
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
//! normalize. The projection is linear, so pooling before it is the same
//! function as transformers' project-then-pool, at 1/T the GEMM.
//!
//! # Precision
//!
//! Weights are widened from the checkpoint's bf16 to f32 exactly (the
//! embedding table stays bf16 and its gather widens exactly), every
//! activation is f32 and every GEMM is exact f32: transformers' fp32 forward
//! up to operation order, which is what `tests/embedgemma2_model.rs` holds it
//! to. It needs the runtime's relaxed-f32 GEMM switched off (the default).
//!
//! # Kernels
//!
//! New for this model: [`encoder_attn`] (`kernels/encoder_attn.metal`),
//! [`segment_mean_rows`] and [`l2_normalize_rows`] (`kernels/embed_pool.metal`).
//! The rest are tessl's: exact-f32 GEMM, `nn::rms_norm_f32`,
//! `nn::rms_norm_residual_add_f32` (the post-norms and the layer scalar),
//! `nn::mlp_gelu_tanh`, `nn::scale_f32_inplace`, `qwen35::embed_rows` and
//! `qwen35::attn_qk_norm_rope_columns` (`* w` norms, full-width RoPE).

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_gpu_buf, set_u32};
use crate::gemm::{gemm, GemmBackend};
use crate::json::{self, Json, Syntax};
use crate::nn::{
    dispatch_tg_1d, mlp_gelu_tanh, reduce_tptg, require, require_disjoint_writes, rms_norm_f32,
    rms_norm_residual_add_f32, scale_f32_inplace,
};
use crate::qwen35::{self, AttnShape, AttnTargets, Cols, LmHead, QkvColumns};
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
/// at this size the largest buffer, a full layer's queries, is 256 MiB.
/// [`EmbedGemma2Model::encode`] splits a batch into as many forwards as
/// this needs, so it bounds memory, not what a caller may pass.
pub const MAX_BATCH_TOKENS: usize = 32_768;

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

/// Bidirectional attention over right-padded sequences.
///
/// `q`/`o`: `[batch, seq, heads, head_dim]`; `k`/`v`: `[batch, seq, heads_kv,
/// head_dim]`, all f32 (`o` bf16 when `out_bf16`). `lens`: `[batch]` u32,
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
    out_bf16: bool,
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
    let kv_elems = elems(&[dims.batch, dims.seq, dims.heads_kv, dims.head_dim], WHAT)?;
    require::<f32>(rt, q, q_elems, "encoder_attn q")?;
    require::<f32>(rt, k, kv_elems, "encoder_attn k")?;
    require::<f32>(rt, v, kv_elems, "encoder_attn v")?;
    if out_bf16 {
        require::<u16>(rt, o, q_elems, "encoder_attn o")?;
    } else {
        require::<f32>(rt, o, q_elems, "encoder_attn o")?;
    }
    require::<u32>(rt, lens, dims.batch as usize, "encoder_attn lens")?;
    if q_elems == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("o", o)], &[("q", q), ("k", k), ("v", v), ("lens", lens)])?;
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
        set_u32(bnd, u32::from(out_bf16), 10);
    })
}

/// `[start, end)` row ranges for [`segment_mean_rows`], checked against
/// `rows` and uploaded: `[n, 2]` u32.
pub fn upload_segments(rt: &Arc<GpuRuntime>, segments: &[(u32, u32)], rows: u32) -> Result<GpuBuffer, String> {
    for (i, &(start, end)) in segments.iter().enumerate() {
        if start >= end || end > rows {
            return Err(format!(
                "embedgemma2::upload_segments: segment {i} is [{start}, {end}); need start < end <= {rows}"
            ));
        }
    }
    let flat: Vec<u32> = segments.iter().flat_map(|&(a, b)| [a, b]).collect();
    let buf = rt.alloc_buffer(flat.len().max(1) * 4)?;
    buf.write_u32(&flat);
    Ok(buf)
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
    require::<u32>(rt, segments, elems(&[n_segments, 2], WHAT)?, "segment_mean_rows segments")?;
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
}

impl EmbedGemma2Config {
    /// Parse the checkpoint's `config.json` (the multimodal root with a
    /// `text_config`, or a bare text config). Everything the forward does not
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
        let c = root.get("text_config").unwrap_or(&root);
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
        let num = |obj: &Json, k: &str| -> Result<f64, String> {
            match obj.get(k) {
                Some(Json::Num { value, .. }) => Ok(*value),
                Some(v) => Err(format!("config.json: {k} must be a number, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        let string = |k: &str| -> Result<Option<&str>, String> {
            match c.get(k) {
                None => Ok(None),
                Some(Json::Str(s)) => Ok(Some(s.as_str())),
                Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
            }
        };

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
                    t => Err(format!("config.json: layer_types[{i}] = {t:?} is not a known layer type")),
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
                    let get = |k: &str| v.get(k).map(|_| uint_in(v, k)).transpose();
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
                return Err(format!("config.json: rope_parameters.{kind}.partial_rotary_factor is not supported"));
            }
            let theta = num(p, "rope_theta")?;
            if !(theta > 0.0 && theta.is_finite()) {
                return Err(format!("config.json: rope_parameters.{kind}.rope_theta must be positive"));
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
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    /// This layer's slice of the per-layer-input projection, `[hidden, ple_dim]`.
    ple_in: Tensor,
    ple_gate: Tensor,
    ple_out: Tensor,
    post_ple_norm: GpuBuffer,
    layer_scalar: f32,
}

/// A loaded text encoder: weights on the device in the layouts the kernels read.
pub struct EmbedGemma2Model {
    cfg: EmbedGemma2Config,
    rt: Arc<GpuRuntime>,
    /// `[vocab, hidden]` bf16, as the checkpoint holds it.
    embed: Tensor,
    ple_norm: GpuBuffer,
    final_norm: GpuBuffer,
    /// `[hidden, embedding_dim]`.
    projection: Tensor,
    layers: Vec<Layer>,
    /// All-ones `[head_dim]` weights for the weightless value norm, per head dim.
    ones: Vec<(u32, GpuBuffer)>,
    max_tokens: u32,
}

/// What [`EmbedGemma2Model::encode`] returns.
pub struct EncodeOutput {
    /// `[batch, embedding_dim]`, each row L2-normalized.
    pub embeddings: Vec<f32>,
    /// With `trace` (one sequence only): the residual stream after each
    /// layer, `[tokens, hidden]`, then the final norm's output. Empty otherwise.
    pub trace: Vec<Vec<f32>>,
    /// How many forwards the batch was split into (see [`MAX_BATCH_TOKENS`]).
    pub forwards: u32,
}

struct Loader<'a> {
    st: &'a SafeTensors,
    prefix: &'a str,
    rt: &'a Arc<GpuRuntime>,
}

impl Loader<'_> {
    fn name(&self, rest: &str) -> String {
        format!("{}{rest}", self.prefix)
    }

    fn f32(&self, rest: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let name = self.name(rest);
        let (got, data) = self.st.read_f32(&name)?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(data)
    }

    fn buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let b = self.rt.alloc_buffer(data.len().max(1) * 4)?;
        b.write_f32(data);
        Ok(b)
    }

    fn norm(&self, rest: &str, dim: u32) -> Result<GpuBuffer, String> {
        self.buf(&self.f32(rest, &[dim as usize])?)
    }

    /// `nn.Linear` weights `[out_i, in]` packed into the right operand
    /// `[in, sum(out_i)]` of one exact-f32 GEMM.
    fn linear(&self, parts: &[(&str, u32)], in_features: u32) -> Result<Tensor, String> {
        let mut data = Vec::with_capacity(parts.len());
        for &(rest, out) in parts {
            data.push(self.f32(rest, &[out as usize, in_features as usize])?);
        }
        self.pack(&data.iter().map(Vec::as_slice).collect::<Vec<_>>(), parts.iter().map(|p| p.1).collect(), in_features)
    }

    fn pack(&self, parts: &[&[f32]], widths: Vec<u32>, in_features: u32) -> Result<Tensor, String> {
        let widths: Vec<usize> = widths.into_iter().map(|w| w as usize).collect();
        let total: usize = widths.iter().sum();
        let packed = qwen35::pack_linear_weights_f32(parts, &widths, in_features as usize)?;
        let t = self.rt.alloc_tensor_f32(&[in_features as usize, total])?;
        t.buffer.write_f32(&packed);
        Ok(t)
    }
}

/// A forward's fixed cost in padded rows, for [`pack_forwards`]: each forward
/// re-reads every weight (~0.6 GB of f32 outside the embedding table), about
/// what a few hundred rows of compute cost. An estimate, not a measurement;
/// it only steers how a batch is split, never what a sequence's embedding is.
const FORWARD_COST_ROWS: usize = 256;

/// Split `batch` into forwards of at most `max_rows` padded rows and at most
/// [`MAX_BATCH`] sequences. Indices are sorted longest first (ties in input
/// order) and cut into contiguous runs minimizing `sum(run_len * longest) +
/// FORWARD_COST_ROWS * runs`, so a long document does not pad short queries
/// to its length. A sequence longer than `max_rows` runs alone.
fn pack_forwards(batch: &[&[u32]], max_rows: usize) -> Vec<Vec<usize>> {
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
            let cost = best[start].saturating_add(rows).saturating_add(FORWARD_COST_ROWS);
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

/// Every activation of one forward, sized for `rows = batch * seq`.
struct Acts {
    rows: u32,
    ids: GpuBuffer,
    lens: GpuBuffer,
    /// `[batch, 2]`: each sequence's live rows, `[b * seq, b * seq + len)`.
    segments: GpuBuffer,
    /// `embed_tokens[ids] * sqrt(hidden)`, kept for the per-layer inputs.
    embeds: Tensor,
    resid: Tensor,
    x: Tensor,
    qkv: Tensor,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    v_normed: GpuBuffer,
    attn: Tensor,
    y: Tensor,
    gate: Tensor,
    up: Tensor,
    mid: Tensor,
    ple: Tensor,
    ple_gate: Tensor,
    ple_mid: Tensor,
    pooled: Tensor,
    out: Tensor,
}

impl EmbedGemma2Model {
    /// Load the text encoder from `st`, whose tensors are named `{prefix}...`
    /// (`"language_model."` in `google/embeddinggemma-2`'s `model.safetensors`).
    /// Every tensor's shape is checked against `cfg`; a missing one is an error.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, prefix: &str, cfg: EmbedGemma2Config) -> Result<Self, String> {
        let ld = Loader { st, prefix, rt };
        let (h, ple, inter, vocab) = (cfg.hidden, cfg.ple_dim, cfg.intermediate, cfg.vocab);
        let n_layers = cfg.layers.len() as u32;

        let embed = {
            let name = ld.name("embed_tokens.weight");
            let (shape, bits) = st.read_bf16_bits(&name)?;
            if shape != [vocab as usize, h as usize] {
                return Err(format!("{name}: shape {shape:?}, expected [{vocab}, {h}]"));
            }
            let t = rt.alloc_tensor_bf16(&[vocab as usize, h as usize])?;
            t.buffer.write_bf16_bits(&bits);
            t
        };
        let ple_w = ld.f32("ple.per_layer_model_projection.weight", &[(n_layers * ple) as usize, h as usize])?;
        let ple_norm = ld.norm("ple.per_layer_projection_norm.weight", ple)?;
        let final_norm = ld.norm("norm.weight", h)?;
        let projection = ld.linear(&[("embedding_projection.weight", cfg.embedding_dim)], h)?;

        let mut layers = Vec::with_capacity(cfg.layers.len());
        let slice = (ple * h) as usize;
        for (i, &spec) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{i}.{s}");
            let (d, q, kv) = (spec.head_dim, cfg.q_heads, spec.kv_heads);
            let (_, scalar) = st.read_f32(&ld.name(&p("layer_scalar")))?;
            let layer_scalar = match scalar.as_slice() {
                [s] if s.is_finite() => *s,
                other => return Err(format!("{}: expected one finite value, got {other:?}", ld.name(&p("layer_scalar")))),
            };
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
                )?,
                w_o: ld.linear(&[(&p("self_attn.o_proj.weight"), h)], q * d)?,
                q_norm: ld.norm(&p("self_attn.q_norm.weight"), d)?,
                k_norm: ld.norm(&p("self_attn.k_norm.weight"), d)?,
                gate: ld.linear(&[(&p("mlp.gate_proj.weight"), inter)], h)?,
                up: ld.linear(&[(&p("mlp.up_proj.weight"), inter)], h)?,
                down: ld.linear(&[(&p("mlp.down_proj.weight"), h)], inter)?,
                ple_in: ld.pack(&[&ple_w[i * slice..(i + 1) * slice]], vec![ple], h)?,
                ple_gate: ld.linear(&[(&p("ple_block.per_layer_input_gate.weight"), ple)], h)?,
                ple_out: ld.linear(&[(&p("ple_block.per_layer_projection.weight"), h)], ple)?,
                post_ple_norm: ld.norm(&p("ple_block.post_per_layer_input_norm.weight"), h)?,
                layer_scalar,
            });
        }
        let mut ones = Vec::new();
        for spec in &cfg.layers {
            if !ones.iter().any(|(d, _)| *d == spec.head_dim) {
                ones.push((spec.head_dim, ld.buf(&vec![1.0f32; spec.head_dim as usize])?));
            }
        }
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            embed,
            ple_norm,
            final_norm,
            projection,
            layers,
            ones,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    pub fn config(&self) -> &EmbedGemma2Config {
        &self.cfg
    }

    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Raise or lower the per-sequence token limit (the position table goes
    /// to 262144; the model card states 8192).
    pub fn set_max_tokens(&mut self, max_tokens: u32) -> Result<(), String> {
        if max_tokens == 0 || max_tokens as usize > MAX_BATCH_TOKENS {
            return Err(format!("max_tokens must be in 1..={MAX_BATCH_TOKENS}, got {max_tokens}"));
        }
        self.max_tokens = max_tokens;
        Ok(())
    }

    /// Embed a batch of tokenized sequences (with the tokenizer's BOS/EOS and
    /// any task prompt already in the ids). Returns one L2-normalized
    /// `embedding_dim` row per sequence, in the order given. A sequence's
    /// embedding does not depend on what else is in the batch beyond GEMM
    /// tiling (padded keys are masked, padded queries are not pooled).
    ///
    /// The batch runs as one or more forwards: sequences sorted longest first
    /// and packed greedily while `sequences * longest <= MAX_BATCH_TOKENS`, so
    /// one long document does not pad every short query to its length.
    /// `trace` (per-layer residuals, for parity tests) takes one sequence.
    pub fn encode(&self, batch: &[&[u32]], trace: bool) -> Result<EncodeOutput, String> {
        const WHAT: &str = "EmbedGemma2Model::encode";
        let cfg = &self.cfg;
        if batch.is_empty() {
            return Err(format!("{WHAT}: empty batch"));
        }
        if batch.len() > MAX_BATCH {
            return Err(format!("{WHAT}: {} sequences exceeds the batch limit {MAX_BATCH}", batch.len()));
        }
        if trace && batch.len() != 1 {
            return Err(format!("{WHAT}: trace takes one sequence, got {}", batch.len()));
        }
        if self.rt.relaxed_precision() {
            return Err(format!(
                "{WHAT}: the forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        for (b, ids) in batch.iter().enumerate() {
            if ids.is_empty() {
                return Err(format!("{WHAT}: sequence {b} is empty"));
            }
            if ids.len() > self.max_tokens as usize {
                return Err(format!(
                    "{WHAT}: sequence {b} has {} tokens; the limit is {} (truncate it, or raise it with set_max_tokens)",
                    ids.len(),
                    self.max_tokens
                ));
            }
            if let Some(&bad) = ids.iter().find(|&&id| id >= cfg.vocab) {
                return Err(format!("{WHAT}: sequence {b}: token id {bad} >= vocab {}", cfg.vocab));
            }
        }

        let dim = cfg.embedding_dim as usize;
        let mut embeddings = vec![0.0f32; batch.len() * dim];
        let mut out_trace = Vec::new();
        let mut forwards = 0u32;
        for chunk in pack_forwards(batch, MAX_BATCH_TOKENS) {
            let seqs: Vec<&[u32]> = chunk.iter().map(|&i| batch[i]).collect();
            let (e, t) = self.forward(&seqs, trace)?;
            for (n, &i) in chunk.iter().enumerate() {
                embeddings[i * dim..(i + 1) * dim].copy_from_slice(&e[n * dim..(n + 1) * dim]);
            }
            out_trace = t;
            forwards += 1;
        }
        Ok(EncodeOutput {
            embeddings,
            trace: out_trace,
            forwards,
        })
    }

    /// One forward over `batch` padded to its longest sequence (validated by
    /// [`Self::encode`]; `batch.len() * longest <= MAX_BATCH_TOKENS`).
    fn forward(&self, batch: &[&[u32]], trace: bool) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let seq = batch.iter().map(|ids| ids.len() as u32).max().unwrap_or(0);
        let nb = batch.len() as u32;
        let rows = nb as usize * seq as usize;
        if seq == 0 || rows > MAX_BATCH_TOKENS {
            return Err(format!(
                "EmbedGemma2Model::forward: {nb} sequences padded to {seq} tokens is {rows} rows; need 1..={MAX_BATCH_TOKENS}"
            ));
        }
        let a = self.acts(nb, seq)?;
        let mut padded = vec![0u32; rows];
        for (b, ids) in batch.iter().enumerate() {
            padded[b * seq as usize..b * seq as usize + ids.len()].copy_from_slice(ids);
        }
        a.ids.write_u32(&padded);
        a.lens.write_u32(&batch.iter().map(|ids| ids.len() as u32).collect::<Vec<_>>());
        let segs: Vec<u32> = batch
            .iter()
            .enumerate()
            .flat_map(|(b, ids)| [b as u32 * seq, b as u32 * seq + ids.len() as u32])
            .collect();
        a.segments.write_u32(&segs);

        let (h, rows_u) = (cfg.hidden, a.rows);
        let eps = cfg.rms_norm_eps;
        qwen35::embed_rows(
            rt,
            &a.ids,
            rows_u,
            LmHead {
                weight: &self.embed.buffer,
                dtype: DType::BF16,
                vocab: cfg.vocab,
            },
            h,
            &a.embeds.buffer,
        )?;
        // transformers multiplies by `embed_scale.to(weight.dtype)`: sqrt(512)
        // as an f32 in the fp32 forward this mirrors.
        scale_f32_inplace(rt, &a.embeds.buffer, (h as f32).sqrt(), rows_u * h)?;
        gpu_copy(&a.embeds, &a.resid)?;

        let mut out_trace = Vec::new();
        let ple_scale = (h as f32).powf(-0.5);
        for layer in &self.layers {
            let s = layer.spec;
            // Attention.
            rms_norm_f32(rt, &a.resid.buffer, &layer.input_norm, &a.x.buffer, rows_u, h, eps)?;
            let qkv_width = (cfg.q_heads + 2 * s.kv_heads) * s.head_dim;
            let qkv = view(&a.qkv, rows_u, qkv_width)?;
            gemm(&a.x, &layer.w_qkv, &qkv, BACKEND)?;
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
                },
                0.0,
                &layer.q_norm,
                &layer.k_norm,
                &AttnTargets {
                    q_out: &a.q,
                    k_cache: &a.k,
                    v_cache: &a.v,
                },
                0,
                s.rope_theta,
                eps,
            )?;
            let ones = &self.ones.iter().find(|(d, _)| *d == s.head_dim).expect("ones for every head dim").1;
            rms_norm_f32(rt, &a.v, ones, &a.v_normed, rows_u * s.kv_heads, s.head_dim, eps)?;
            encoder_attn(
                rt,
                &a.q,
                &a.k,
                &a.v_normed,
                &a.attn.buffer,
                &a.lens,
                EncoderAttnDims {
                    batch: nb,
                    seq,
                    heads: cfg.q_heads,
                    heads_kv: s.kv_heads,
                    head_dim: s.head_dim,
                    window: if s.sliding { cfg.sliding_window } else { 0 },
                    scale: 1.0,
                },
                false,
            )?;
            let attn = view(&a.attn, rows_u, cfg.q_heads * s.head_dim)?;
            gemm(&attn, &layer.w_o, &a.y, BACKEND)?;
            rms_norm_residual_add_f32(rt, &a.y.buffer, &layer.post_attn_norm, &a.resid.buffer, rows_u, h, eps, 1.0)?;

            // MLP.
            rms_norm_f32(rt, &a.resid.buffer, &layer.pre_ffn_norm, &a.x.buffer, rows_u, h, eps)?;
            gemm(&a.x, &layer.gate, &a.gate, BACKEND)?;
            gemm(&a.x, &layer.up, &a.up, BACKEND)?;
            mlp_gelu_tanh(rt, &a.gate.buffer, &a.up.buffer, &a.mid.buffer, rows_u * cfg.intermediate)?;
            gemm(&a.mid, &layer.down, &a.y, BACKEND)?;
            rms_norm_residual_add_f32(rt, &a.y.buffer, &layer.post_ffn_norm, &a.resid.buffer, rows_u, h, eps, 1.0)?;

            // Per-layer input, then its block and the layer scalar.
            gemm(&a.embeds, &layer.ple_in, &a.ple, BACKEND)?;
            scale_f32_inplace(rt, &a.ple.buffer, ple_scale, rows_u * cfg.ple_dim)?;
            rms_norm_f32(rt, &a.ple.buffer, &self.ple_norm, &a.ple_mid.buffer, rows_u, cfg.ple_dim, eps)?;
            gemm(&a.resid, &layer.ple_gate, &a.ple_gate, BACKEND)?;
            mlp_gelu_tanh(rt, &a.ple_gate.buffer, &a.ple_mid.buffer, &a.ple.buffer, rows_u * cfg.ple_dim)?;
            gemm(&a.ple, &layer.ple_out, &a.y, BACKEND)?;
            rms_norm_residual_add_f32(
                rt,
                &a.y.buffer,
                &layer.post_ple_norm,
                &a.resid.buffer,
                rows_u,
                h,
                eps,
                layer.layer_scalar,
            )?;
            if trace {
                rt.synchronize()?;
                out_trace.push(a.resid.buffer.read_f32()[..rows_u as usize * h as usize].to_vec());
            }
        }

        rms_norm_f32(rt, &a.resid.buffer, &self.final_norm, &a.x.buffer, rows_u, h, eps)?;
        segment_mean_rows(rt, &a.x.buffer, &a.segments, &a.pooled.buffer, nb, rows_u, h)?;
        gemm(&a.pooled, &self.projection, &a.out, BACKEND)?;
        l2_normalize_rows(rt, &a.out.buffer, nb, cfg.embedding_dim, cfg.embedding_dim)?;
        rt.synchronize()?;
        if trace {
            out_trace.push(a.x.buffer.read_f32()[..rows_u as usize * h as usize].to_vec());
        }
        let n = nb as usize * cfg.embedding_dim as usize;
        Ok((a.out.buffer.read_f32()[..n].to_vec(), out_trace))
    }

    fn acts(&self, nb: u32, seq: u32) -> Result<Acts, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let rows = nb * seq;
        let r = rows as usize;
        let (h, ple, inter) = (cfg.hidden as usize, cfg.ple_dim as usize, cfg.intermediate as usize);
        // Widest attention layer: the q|k|v row and the per-head buffers.
        let (mut qkv_w, mut q_w, mut kv_w) = (0usize, 0usize, 0usize);
        for s in &cfg.layers {
            let (q, kv, d) = (cfg.q_heads as usize, s.kv_heads as usize, s.head_dim as usize);
            qkv_w = qkv_w.max((q + 2 * kv) * d);
            q_w = q_w.max(q * d);
            kv_w = kv_w.max(kv * d);
        }
        let f32s = |n: usize| rt.alloc_buffer(n.max(1) * 4);
        Ok(Acts {
            rows,
            ids: rt.alloc_buffer(r.max(1) * 4)?,
            lens: rt.alloc_buffer(nb.max(1) as usize * 4)?,
            segments: rt.alloc_buffer(nb.max(1) as usize * 8)?,
            embeds: rt.alloc_tensor_f32(&[r, h])?,
            resid: rt.alloc_tensor_f32(&[r, h])?,
            x: rt.alloc_tensor_f32(&[r, h])?,
            qkv: rt.alloc_tensor_f32(&[r, qkv_w])?,
            q: f32s(r * q_w)?,
            k: f32s(r * kv_w)?,
            v: f32s(r * kv_w)?,
            v_normed: f32s(r * kv_w)?,
            attn: rt.alloc_tensor_f32(&[r, q_w])?,
            y: rt.alloc_tensor_f32(&[r, h])?,
            gate: rt.alloc_tensor_f32(&[r, inter])?,
            up: rt.alloc_tensor_f32(&[r, inter])?,
            mid: rt.alloc_tensor_f32(&[r, inter])?,
            ple: rt.alloc_tensor_f32(&[r, ple])?,
            ple_gate: rt.alloc_tensor_f32(&[r, ple])?,
            ple_mid: rt.alloc_tensor_f32(&[r, ple])?,
            pooled: rt.alloc_tensor_f32(&[nb as usize, h])?,
            out: rt.alloc_tensor_f32(&[nb as usize, cfg.embedding_dim as usize])?,
        })
    }
}

/// A `[rows, cols]` view over the front of `t`'s buffer: the activations
/// are sized for the widest layer and narrower layers use a prefix.
fn view(t: &Tensor, rows: u32, cols: u32) -> Result<Tensor, String> {
    t.try_view(&[rows as usize, cols as usize], 0)
}

/// Re-normalize a prefix of already-normalized embeddings on the host: the
/// Matryoshka sizes (768, 512, 256, 128). sentence-transformers'
/// `truncate_dim` slices *without* renormalizing; this renormalizes, which is
/// what cosine search over the prefix wants and leaves cosine scores equal.
pub fn truncate_renormalize(embeddings: &[f32], dim_in: usize, dim_out: usize) -> Result<Vec<f32>, String> {
    if dim_out == 0 || dim_out > dim_in || dim_in == 0 || embeddings.len() % dim_in != 0 {
        return Err(format!(
            "truncate_renormalize: need 1 <= dim_out ({dim_out}) <= dim_in ({dim_in}) and whole rows"
        ));
    }
    let mut out = Vec::with_capacity(embeddings.len() / dim_in * dim_out);
    for row in embeddings.chunks_exact(dim_in) {
        let p = &row[..dim_out];
        let norm = p.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt().max(1e-12);
        out.extend(p.iter().map(|&v| (f64::from(v) / norm) as f32));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{pack_forwards, MAX_BATCH};

    /// Every index exactly once; each forward within the row budget and the
    /// sequence cap; each forward padded to its first (longest) sequence.
    fn check(lens: &[usize], max_rows: usize) -> Vec<Vec<usize>> {
        let seqs: Vec<Vec<u32>> = lens.iter().map(|&n| vec![0; n]).collect();
        let batch: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
        let packs = pack_forwards(&batch, max_rows);
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
        let cost = |packs: &[Vec<usize>]| -> usize {
            packs.iter().map(|p| p.len() * lens[p[0]] + super::FORWARD_COST_ROWS).sum()
        };
        let mut order: Vec<usize> = (0..lens.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        let mut greedy: Vec<Vec<usize>> = Vec::new();
        for i in order {
            match greedy.last_mut() {
                Some(g) if (g.len() + 1) * lens[g[0]] <= max_rows && g.len() < MAX_BATCH => g.push(i),
                _ => greedy.push(vec![i]),
            }
        }
        assert!(cost(&packs) <= cost(&greedy), "{lens:?}: {} > greedy {}", cost(&packs), cost(&greedy));
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
