//! BERT-family masked-LM encoders as doc-side learned sparse encoders,
//! composed from tessl kernels and loaded straight from `model.safetensors`.
//!
//! The model is `BertForMaskedLM` or `DistilBertForMaskedLM` (the
//! `opensearch-neural-sparse-encoding-doc-*` family is both), and the output is
//! not the masked-LM logits but what those models are trained to produce from
//! them: per sequence,
//!
//! ```text
//! sparse[v] = max over the sequence's tokens t of log1p(relu(logits[t, v]))
//! ```
//!
//! which a search index stores as the document's term weights. Padding tokens
//! take no part (torch multiplies them by the attention mask, and every term
//! is >= 0, so a masked token contributes 0).
//!
//! # Layer (transformers `BertLayer` / DistilBERT `TransformerBlock`)
//!
//! ```text
//! x      = LayerNorm(word[ids] + pos[t] + type[0])        (DistilBERT: no type)
//! q,k,v  = x Wq + bq, x Wk + bk, x Wv + bv
//! a      = softmax(q kᵀ / sqrt(head_dim) + mask) v         bidirectional
//! x      = LayerNorm(a Wo + bo + x)
//! x      = LayerNorm(gelu_erf(x Wi + bi) Wout + bout + x)
//! ```
//!
//! and the head: `LayerNorm(gelu_erf(x Wt + bt)) @ word_embeddingsᵀ + bias`.
//! Both checkpoints tie the decoder to the word embeddings (neither stores a
//! decoder matrix); one that does store `cls.predictions.decoder.weight` is
//! read from it instead.
//!
//! # Precision
//!
//! Weights and activations are f32 and every GEMM is exact f32, which is
//! torch's fp32 forward up to operation order. The kernels new for this model
//! are in `kernels/bert.metal` (LayerNorm plain and fused with a bias and a
//! residual, the embedding sum, a bias add, the erf GELU, and the segment
//! sparse max), each against an f64 reference in `tests/bert_kernels.rs`;
//! attention is [`crate::embedgemma2::encoder_attn`] at head dims 32 and 64.
//! The whole forward is held to torch by DevCouncil's `dc-sparse-encode`,
//! which records the reference.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_1d, set_f32, set_gpu_buf, set_u32};
use crate::embedgemma2::{encoder_attn, EncoderAttnDims};
use crate::gemm::{gemm, gemm_nt_f32, GemmBackend};
use crate::json::{self, Json, Syntax};
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes};
use crate::qwen35;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{GpuBuffer, Tensor};

const BACKEND: GemmBackend = GemmBackend::TensorOps;

/// Sequences per [`BertSparseModel::encode`] call.
pub const MAX_BATCH: usize = 256;
/// Padded tokens (`sequences * longest`) per forward. At this size the widest
/// activation, the FFN's `[rows, intermediate]` at DistilBERT's 3072, is
/// 96 MiB. [`BertSparseModel::encode`] splits a batch into as many forwards as
/// this needs, so it bounds memory, not what a caller may pass.
pub const MAX_BATCH_TOKENS: usize = 8192;
/// Rows of masked-LM logits held at once. The head's output is `[rows,
/// vocab]` — 30522 columns — so it runs over whole sequences in blocks of at
/// most this many rows (about 250 MB of logits) and pools each block before
/// the next, rather than materializing a forward's full logits.
pub const HEAD_BLOCK_ROWS: usize = 2048;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

fn elems(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| format!("{what}: element count {dims:?} overflows usize"))
}

fn check_eps(eps: f32, what: &str) -> Result<(), String> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(format!("{what}: eps must be finite and positive, got {eps}"))
    }
}

/// The tables [`embed_layer_norm`] reads.
#[derive(Clone, Copy)]
pub struct EmbedTables<'a> {
    /// `[vocab, dim]` f32.
    pub word: &'a GpuBuffer,
    pub vocab: u32,
    /// `[positions, dim]` f32; row `t` is added to token `t` of each sequence.
    pub pos: &'a GpuBuffer,
    pub positions: u32,
    /// `[dim]` f32 added to every token: token type 0's row, or zeros for a
    /// model with no token types.
    pub type_row: &'a GpuBuffer,
    /// LayerNorm weight and bias, `[dim]` each.
    pub ln_w: &'a GpuBuffer,
    pub ln_b: &'a GpuBuffer,
}

/// `out[r] = LayerNorm(word[ids[r]] + pos[r % seq] + type_row) * w + b` over
/// `rows = batch * seq` rows of `dim` (right-padded sequences of `seq`
/// positions each). An id `>= vocab` cannot be checked here (it is a device
/// value); its row is written as NaN and every other row is unaffected.
#[allow(clippy::too_many_arguments)]
pub fn embed_layer_norm(
    rt: &Arc<GpuRuntime>,
    ids: &GpuBuffer,
    tables: EmbedTables<'_>,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    seq: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "bert::embed_layer_norm";
    check_eps(eps, WHAT)?;
    if dim == 0 || tables.vocab == 0 {
        return Err(format!("{WHAT}: dim and vocab must be non-zero"));
    }
    if seq == 0 || seq > tables.positions {
        return Err(format!(
            "{WHAT}: seq {seq} must be in 1..={} (the position table)",
            tables.positions
        ));
    }
    if rows % seq != 0 {
        return Err(format!(
            "{WHAT}: rows {rows} is not a whole number of {seq}-position sequences"
        ));
    }
    require::<u32>(rt, ids, rows as usize, "embed_layer_norm ids")?;
    require::<f32>(
        rt,
        tables.word,
        elems(&[tables.vocab, dim], WHAT)?,
        "embed_layer_norm word",
    )?;
    require::<f32>(
        rt,
        tables.pos,
        elems(&[tables.positions, dim], WHAT)?,
        "embed_layer_norm pos",
    )?;
    require::<f32>(rt, tables.type_row, dim as usize, "embed_layer_norm type_row")?;
    require::<f32>(rt, tables.ln_w, dim as usize, "embed_layer_norm ln_w")?;
    require::<f32>(rt, tables.ln_b, dim as usize, "embed_layer_norm ln_b")?;
    require::<f32>(rt, out, elems(&[rows, dim], WHAT)?, "embed_layer_norm out")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("out", out)],
        &[
            ("ids", ids),
            ("word", tables.word),
            ("pos", tables.pos),
            ("type_row", tables.type_row),
            ("ln_w", tables.ln_w),
            ("ln_b", tables.ln_b),
        ],
    )?;
    let p = rt.pipeline("bert_embed_layer_norm_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, ids, 0);
        set_gpu_buf(bnd, tables.word, 1);
        set_gpu_buf(bnd, tables.pos, 2);
        set_gpu_buf(bnd, tables.type_row, 3);
        set_gpu_buf(bnd, tables.ln_w, 4);
        set_gpu_buf(bnd, tables.ln_b, 5);
        set_gpu_buf(bnd, out, 6);
        set_u32(bnd, rows, 7);
        set_u32(bnd, dim, 8);
        set_u32(bnd, seq, 9);
        set_u32(bnd, tables.vocab, 10);
        set_f32(bnd, eps, 11);
    })
}

/// In place, `resid[r] = LayerNorm(y[r] + bias + resid[r]) * w + b` over
/// `rows` rows of `dim`: a dense layer's bias, the residual add and the
/// post-norm in one dispatch.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_norm(
    rt: &Arc<GpuRuntime>,
    y: &GpuBuffer,
    bias: &GpuBuffer,
    resid: &GpuBuffer,
    w: &GpuBuffer,
    b: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "bert::bias_residual_layer_norm";
    check_eps(eps, WHAT)?;
    if dim == 0 {
        return Err(format!("{WHAT}: dim must be non-zero"));
    }
    let n = elems(&[rows, dim], WHAT)?;
    require::<f32>(rt, y, n, "bias_residual_layer_norm y")?;
    require::<f32>(rt, resid, n, "bias_residual_layer_norm resid")?;
    for (name, buf) in [("bias", bias), ("w", w), ("b", b)] {
        require::<f32>(rt, buf, dim as usize, format_args!("bias_residual_layer_norm {name}"))?;
    }
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("resid", resid)],
        &[("y", y), ("bias", bias), ("w", w), ("b", b)],
    )?;
    let p = rt.pipeline("bert_bias_residual_layer_norm_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, y, 0);
        set_gpu_buf(bnd, bias, 1);
        set_gpu_buf(bnd, resid, 2);
        set_gpu_buf(bnd, w, 3);
        set_gpu_buf(bnd, b, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, dim, 6);
        set_f32(bnd, eps, 7);
    })
}

/// `out[r] = LayerNorm(x[r]) * w + b` over `rows` rows of `dim`, with torch's
/// biased variance.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "bert::layer_norm";
    check_eps(eps, WHAT)?;
    if dim == 0 {
        return Err(format!("{WHAT}: dim must be non-zero"));
    }
    let n = elems(&[rows, dim], WHAT)?;
    require::<f32>(rt, x, n, "layer_norm x")?;
    require::<f32>(rt, out, n, "layer_norm out")?;
    require::<f32>(rt, w, dim as usize, "layer_norm w")?;
    require::<f32>(rt, b, dim as usize, "layer_norm b")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("out", out)], &[("x", x), ("w", w), ("b", b)])?;
    let p = rt.pipeline("bert_layer_norm_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_gpu_buf(bnd, b, 2);
        set_gpu_buf(bnd, out, 3);
        set_u32(bnd, rows, 4);
        set_u32(bnd, dim, 5);
        set_f32(bnd, eps, 6);
    })
}

fn bias_elementwise(
    rt: &Arc<GpuRuntime>,
    entry: &str,
    what: &str,
    x: &GpuBuffer,
    bias: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    if cols == 0 {
        return Err(format!("{what}: cols must be non-zero"));
    }
    let n = elems(&[rows, cols], what)?;
    if n > u32::MAX as usize {
        return Err(format!("{what}: {n} elements exceeds the kernel's u32 index"));
    }
    require::<f32>(rt, x, n, format_args!("{what} x"))?;
    require::<f32>(rt, bias, cols as usize, format_args!("{what} bias"))?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(what, &[("x", x)], &[("bias", bias)])?;
    let p = rt.pipeline(entry)?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, bias, 1);
        set_u32(bnd, n as u32, 2);
        set_u32(bnd, cols, 3);
    })
}

/// In place, `x[r, c] += bias[c]` over `[rows, cols]`.
pub fn bias_add(rt: &Arc<GpuRuntime>, x: &GpuBuffer, bias: &GpuBuffer, rows: u32, cols: u32) -> Result<(), String> {
    bias_elementwise(rt, "bert_bias_add_f32", "bert::bias_add", x, bias, rows, cols)
}

/// In place, `x[r, c] = gelu(x[r, c] + bias[c])` with torch's exact (erf)
/// GELU, not the tanh approximation `nn::mlp_gelu_tanh` computes.
pub fn bias_gelu_erf(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    bias: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    bias_elementwise(rt, "bert_bias_gelu_erf_f32", "bert::bias_gelu_erf", x, bias, rows, cols)
}

/// `[start, end)` row ranges for [`segment_sparse_max`], checked against
/// `rows` and uploaded: `[n, 2]` u32.
pub fn upload_segments(rt: &Arc<GpuRuntime>, segments: &[(u32, u32)], rows: u32) -> Result<GpuBuffer, String> {
    for (i, &(start, end)) in segments.iter().enumerate() {
        if start >= end || end > rows {
            return Err(format!(
                "bert::upload_segments: segment {i} is [{start}, {end}); need start < end <= {rows}"
            ));
        }
    }
    let flat: Vec<u32> = segments.iter().flat_map(|&(a, b)| [a, b]).collect();
    let buf = rt.alloc_buffer(flat.len().max(1) * 4)?;
    buf.write_u32(&flat);
    Ok(buf)
}

/// `pooled[s, v] = max(pooled[s, v], log1p(relu(max_{r in segment s}
/// logits[r, v] + bias[v])))` for `logits: [rows, vocab]`, `segments: [n, 2]`
/// row ranges of `logits` ([`upload_segments`]) and `pooled: [n, vocab]`.
///
/// It accumulates, so a caller zeroes `pooled` once and may then pool a
/// sequence's rows over several calls; the result is the doc-side sparse
/// vector over every row seen. Ranges are device values: `end` is clamped to
/// `rows` and an empty range leaves its row of `pooled` unchanged.
#[allow(clippy::too_many_arguments)]
pub fn segment_sparse_max(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    bias: &GpuBuffer,
    segments: &GpuBuffer,
    pooled: &GpuBuffer,
    n_segments: u32,
    rows: u32,
    vocab: u32,
) -> Result<(), String> {
    const WHAT: &str = "bert::segment_sparse_max";
    if vocab == 0 {
        return Err(format!("{WHAT}: vocab must be non-zero"));
    }
    require::<f32>(rt, logits, elems(&[rows, vocab], WHAT)?, "segment_sparse_max logits")?;
    require::<f32>(rt, bias, vocab as usize, "segment_sparse_max bias")?;
    require::<u32>(
        rt,
        segments,
        elems(&[n_segments, 2], WHAT)?,
        "segment_sparse_max segments",
    )?;
    let n = elems(&[n_segments, vocab], WHAT)?;
    require::<f32>(rt, pooled, n, "segment_sparse_max pooled")?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("pooled", pooled)],
        &[("logits", logits), ("bias", bias), ("segments", segments)],
    )?;
    let p = rt.pipeline("bert_segment_sparse_max_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, bias, 1);
        set_gpu_buf(bnd, segments, 2);
        set_gpu_buf(bnd, pooled, 3);
        set_u32(bnd, n_segments, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, vocab, 6);
    })
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Which checkpoint layout the tensors are named in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BertFamily {
    /// `BertForMaskedLM`: `bert.*` and `cls.predictions.*`.
    Bert,
    /// `DistilBertForMaskedLM`: `distilbert.*` and `vocab_*`; no token types.
    DistilBert,
}

/// The encoder's shapes, read from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct BertConfig {
    pub family: BertFamily,
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub max_positions: u32,
    /// Token-type rows (BERT's `type_vocab_size`); 0 for DistilBERT.
    pub type_vocab: u32,
    pub layer_norm_eps: f32,
}

impl BertConfig {
    /// Head dimension, `hidden / heads`.
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.heads
    }

    /// Parse a `config.json`. Everything the forward does not implement — a
    /// relative position embedding, a tanh GELU, a sinusoidal DistilBERT, a
    /// head dim with no attention kernel — is refused by name rather than
    /// computed as something else.
    pub fn from_config_json(text: &str) -> Result<Self, String> {
        const SYNTAX: Syntax = Syntax {
            what: "config.json",
            max_depth: 16,
            uints_only: false,
            literals: true,
        };
        let c = json::parse(text, SYNTAX)?;
        if !matches!(c, Json::Object(_)) {
            return Err("config.json: the root is not an object".into());
        }
        let uint = |k: &str| -> Result<u32, String> {
            match c.get(k) {
                Some(Json::Num { uint: Some(n), .. }) => {
                    u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32"))
                }
                Some(v) => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
                None => Err(format!("config.json: missing {k:?}")),
            }
        };
        let string = |k: &str| -> Result<Option<&str>, String> {
            match c.get(k) {
                None | Some(Json::Null) => Ok(None),
                Some(Json::Str(s)) => Ok(Some(s.as_str())),
                Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
            }
        };
        let family = match string("model_type")? {
            Some("bert") => BertFamily::Bert,
            Some("distilbert") => BertFamily::DistilBert,
            other => {
                return Err(format!(
                    "config.json: model_type must be bert or distilbert, got {other:?}"
                ))
            }
        };
        let (act_key, hidden_k, layers_k, heads_k, inter_k) = match family {
            BertFamily::Bert => (
                "hidden_act",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
            ),
            BertFamily::DistilBert => ("activation", "dim", "n_layers", "n_heads", "hidden_dim"),
        };
        match string(act_key)? {
            Some("gelu") => {}
            other => {
                return Err(format!(
                    "config.json: {act_key} must be gelu (the erf form), got {other:?}"
                ))
            }
        }
        let (type_vocab, layer_norm_eps) = match family {
            BertFamily::Bert => {
                match string("position_embedding_type")? {
                    None | Some("absolute") => {}
                    other => {
                        return Err(format!(
                            "config.json: position_embedding_type must be absolute, got {other:?}"
                        ))
                    }
                }
                let eps = match c.get("layer_norm_eps") {
                    Some(Json::Num { value, .. }) => *value,
                    Some(v) => return Err(format!("config.json: layer_norm_eps must be a number, got {v:?}")),
                    None => return Err("config.json: missing \"layer_norm_eps\"".into()),
                };
                (uint("type_vocab_size")?, eps)
            }
            BertFamily::DistilBert => {
                if let Some(Json::Bool(true)) = c.get("sinusoidal_pos_embds") {
                    return Err("config.json: sinusoidal_pos_embds is not supported".into());
                }
                // transformers' DistilBERT hard-codes LayerNorm(eps=1e-12).
                (0, 1e-12)
            }
        };
        let cfg = Self {
            family,
            hidden: uint(hidden_k)?,
            layers: uint(layers_k)?,
            heads: uint(heads_k)?,
            intermediate: uint(inter_k)?,
            vocab: uint("vocab_size")?,
            max_positions: uint("max_position_embeddings")?,
            type_vocab,
            layer_norm_eps: layer_norm_eps as f32,
        };
        for (k, v) in [
            (hidden_k, cfg.hidden),
            (layers_k, cfg.layers),
            (heads_k, cfg.heads),
            (inter_k, cfg.intermediate),
            ("vocab_size", cfg.vocab),
            ("max_position_embeddings", cfg.max_positions),
        ] {
            if v == 0 {
                return Err(format!("config.json: {k} is zero"));
            }
        }
        if family == BertFamily::Bert && cfg.type_vocab == 0 {
            return Err("config.json: type_vocab_size is zero".into());
        }
        if !(layer_norm_eps > 0.0 && layer_norm_eps.is_finite() && cfg.layer_norm_eps > 0.0) {
            return Err(format!(
                "config.json: layer_norm_eps must be positive, got {layer_norm_eps}"
            ));
        }
        if cfg.hidden % cfg.heads != 0 {
            return Err(format!(
                "config.json: {hidden_k} {} is not a multiple of {heads_k} {}",
                cfg.hidden, cfg.heads
            ));
        }
        let d = cfg.head_dim();
        if !matches!(d, 32 | 64) {
            return Err(format!("config.json: head dim {d} has no attention kernel (32 or 64)"));
        }
        Ok(cfg)
    }

    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A dense layer: `[in, out]` weight for `x @ w`, and `[out]` bias.
struct Dense {
    w: Tensor,
    b: GpuBuffer,
}

struct Norm {
    w: GpuBuffer,
    b: GpuBuffer,
}

struct Layer {
    q: Dense,
    k: Dense,
    v: Dense,
    o: Dense,
    attn_norm: Norm,
    inter: Dense,
    out: Dense,
    out_norm: Norm,
}

/// Tensor names of one checkpoint layout.
struct Names {
    embed: &'static str,
    layer: fn(u32, &str) -> String,
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
    attn_norm: &'static str,
    inter: &'static str,
    out: &'static str,
    out_norm: &'static str,
    transform: &'static str,
    transform_norm: &'static str,
    decoder: &'static str,
    decoder_bias: &'static str,
}

fn names(family: BertFamily) -> Names {
    match family {
        BertFamily::Bert => Names {
            embed: "bert.embeddings.",
            layer: |i, rest| format!("bert.encoder.layer.{i}.{rest}"),
            q: "attention.self.query",
            k: "attention.self.key",
            v: "attention.self.value",
            o: "attention.output.dense",
            attn_norm: "attention.output.LayerNorm",
            inter: "intermediate.dense",
            out: "output.dense",
            out_norm: "output.LayerNorm",
            transform: "cls.predictions.transform.dense",
            transform_norm: "cls.predictions.transform.LayerNorm",
            decoder: "cls.predictions.decoder.weight",
            decoder_bias: "cls.predictions.bias",
        },
        BertFamily::DistilBert => Names {
            embed: "distilbert.embeddings.",
            layer: |i, rest| format!("distilbert.transformer.layer.{i}.{rest}"),
            q: "attention.q_lin",
            k: "attention.k_lin",
            v: "attention.v_lin",
            o: "attention.out_lin",
            attn_norm: "sa_layer_norm",
            inter: "ffn.lin1",
            out: "ffn.lin2",
            out_norm: "output_layer_norm",
            transform: "vocab_transform",
            transform_norm: "vocab_layer_norm",
            decoder: "vocab_projector.weight",
            decoder_bias: "vocab_projector.bias",
        },
    }
}

struct Loader<'a> {
    st: &'a SafeTensors,
    rt: &'a Arc<GpuRuntime>,
}

impl Loader<'_> {
    fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (got, data) = self.st.read_f32(name)?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        if let Some(i) = data.iter().position(|v| !v.is_finite()) {
            return Err(format!("{name}: element {i} is {}", data[i]));
        }
        Ok(data)
    }

    fn buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let b = self.rt.alloc_buffer(data.len().max(1) * 4)?;
        b.write_f32(data);
        Ok(b)
    }

    fn vector(&self, name: &str, dim: u32) -> Result<GpuBuffer, String> {
        self.buf(&self.f32(name, &[dim as usize])?)
    }

    fn norm(&self, prefix: &str, dim: u32) -> Result<Norm, String> {
        Ok(Norm {
            w: self.vector(&format!("{prefix}.weight"), dim)?,
            b: self.vector(&format!("{prefix}.bias"), dim)?,
        })
    }

    /// `nn.Linear(in, out)`: weight `[out, in]` transposed to `[in, out]`.
    fn dense(&self, prefix: &str, in_features: u32, out_features: u32) -> Result<Dense, String> {
        let w = self.f32(
            &format!("{prefix}.weight"),
            &[out_features as usize, in_features as usize],
        )?;
        let packed = qwen35::pack_linear_weights_f32(&[&w], &[out_features as usize], in_features as usize)?;
        let t = self
            .rt
            .alloc_tensor_f32(&[in_features as usize, out_features as usize])?;
        t.buffer.write_f32(&packed);
        Ok(Dense {
            w: t,
            b: self.vector(&format!("{prefix}.bias"), out_features)?,
        })
    }
}

/// A loaded encoder: weights on the device in the layouts the kernels read.
pub struct BertSparseModel {
    cfg: BertConfig,
    rt: Arc<GpuRuntime>,
    /// `[vocab, hidden]`: the embedding gather's table.
    word: Tensor,
    /// `[vocab, hidden]`: the decoder, `word` itself unless the checkpoint
    /// stores its own.
    decoder: Option<Tensor>,
    decoder_bias: GpuBuffer,
    pos: GpuBuffer,
    type_row: GpuBuffer,
    embed_norm: Norm,
    layers: Vec<Layer>,
    transform: Dense,
    transform_norm: Norm,
}

/// What [`BertSparseModel::encode`] returns.
pub struct SparseOutput {
    /// `[batch, vocab]`: each sequence's `max_t log1p(relu(logits[t]))`.
    pub pooled: Vec<f32>,
    /// With `trace` (one sequence only): the embedding LayerNorm's output and
    /// then the residual stream after each layer, `[tokens, hidden]` each.
    /// Empty otherwise.
    pub trace: Vec<Vec<f32>>,
    /// How many forwards the batch was split into (see [`MAX_BATCH_TOKENS`]).
    pub forwards: u32,
}

/// Every activation of one forward, sized for `rows = batch * seq`.
struct Acts {
    ids: GpuBuffer,
    lens: GpuBuffer,
    resid: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn: Tensor,
    y: Tensor,
    mid: Tensor,
    x: Tensor,
    logits: Tensor,
    pooled: GpuBuffer,
}

impl BertSparseModel {
    /// Load the encoder and its masked-LM head from `st`. Every tensor's
    /// shape is checked against `cfg`, and every weight must be finite; a
    /// missing one is an error.
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: BertConfig) -> Result<Self, String> {
        let ld = Loader { st, rt };
        let n = names(cfg.family);
        let (h, inter, vocab) = (cfg.hidden, cfg.intermediate, cfg.vocab);
        let e = |rest: &str| format!("{}{rest}", n.embed);

        let word_data = ld.f32(&e("word_embeddings.weight"), &[vocab as usize, h as usize])?;
        let word = rt.alloc_tensor_f32(&[vocab as usize, h as usize])?;
        word.buffer.write_f32(&word_data);
        let decoder = match st.info(n.decoder) {
            Ok(_) => {
                let d = ld.f32(n.decoder, &[vocab as usize, h as usize])?;
                // Stored but tied is still tied: keep one table.
                if d == word_data {
                    None
                } else {
                    let t = rt.alloc_tensor_f32(&[vocab as usize, h as usize])?;
                    t.buffer.write_f32(&d);
                    Some(t)
                }
            }
            Err(_) => None,
        };
        let pos = ld.buf(&ld.f32(
            &e("position_embeddings.weight"),
            &[cfg.max_positions as usize, h as usize],
        )?)?;
        let type_row = match cfg.family {
            BertFamily::Bert => {
                let t = ld.f32(
                    &e("token_type_embeddings.weight"),
                    &[cfg.type_vocab as usize, h as usize],
                )?;
                ld.buf(&t[..h as usize])?
            }
            BertFamily::DistilBert => ld.buf(&vec![0.0f32; h as usize])?,
        };
        let embed_norm = ld.norm(&e("LayerNorm"), h)?;

        let mut layers = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers {
            let p = |rest: &str| (n.layer)(i, rest);
            layers.push(Layer {
                q: ld.dense(&p(n.q), h, h)?,
                k: ld.dense(&p(n.k), h, h)?,
                v: ld.dense(&p(n.v), h, h)?,
                o: ld.dense(&p(n.o), h, h)?,
                attn_norm: ld.norm(&p(n.attn_norm), h)?,
                inter: ld.dense(&p(n.inter), h, inter)?,
                out: ld.dense(&p(n.out), inter, h)?,
                out_norm: ld.norm(&p(n.out_norm), h)?,
            });
        }
        let transform = ld.dense(n.transform, h, h)?;
        let transform_norm = ld.norm(n.transform_norm, h)?;
        let decoder_bias = ld.vector(n.decoder_bias, vocab)?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            word,
            decoder,
            decoder_bias,
            pos,
            type_row,
            embed_norm,
            layers,
            transform,
            transform_norm,
        })
    }

    pub fn config(&self) -> &BertConfig {
        &self.cfg
    }

    /// Encode a batch of tokenized sequences, each with the tokenizer's
    /// `[CLS]` and `[SEP]` already in the ids and at most `max_positions`
    /// long. Returns each sequence's sparse vector, in the order given. A
    /// sequence's vector does not depend on what else is in the batch beyond
    /// GEMM tiling: padded keys are masked and padded rows are not pooled.
    ///
    /// The batch runs as one or more forwards, cut in the order given so
    /// that `sequences * longest <= MAX_BATCH_TOKENS`; a caller that sorts
    /// by length wastes the least on padding. `trace` takes one sequence.
    pub fn encode(&self, batch: &[&[u32]], trace: bool) -> Result<SparseOutput, String> {
        const WHAT: &str = "BertSparseModel::encode";
        let cfg = &self.cfg;
        if batch.is_empty() {
            return Err(format!("{WHAT}: empty batch"));
        }
        if batch.len() > MAX_BATCH {
            return Err(format!(
                "{WHAT}: {} sequences exceeds the batch limit {MAX_BATCH}",
                batch.len()
            ));
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
            if ids.len() > cfg.max_positions as usize {
                return Err(format!(
                    "{WHAT}: sequence {b} has {} tokens; the model has {} positions (truncate it)",
                    ids.len(),
                    cfg.max_positions
                ));
            }
            if let Some(&bad) = ids.iter().find(|&&id| id >= cfg.vocab) {
                return Err(format!("{WHAT}: sequence {b}: token id {bad} >= vocab {}", cfg.vocab));
            }
        }

        let v = cfg.vocab as usize;
        let mut pooled = vec![0.0f32; batch.len() * v];
        let mut out_trace = Vec::new();
        let mut forwards = 0u32;
        let mut start = 0usize;
        while start < batch.len() {
            // Grow the forward while it stays inside the row budget.
            let mut end = start + 1;
            let mut longest = batch[start].len();
            while end < batch.len() {
                let l = longest.max(batch[end].len());
                if (end + 1 - start) * l > MAX_BATCH_TOKENS {
                    break;
                }
                longest = l;
                end += 1;
            }
            let (p, t) = self.forward(&batch[start..end], trace)?;
            pooled[start * v..end * v].copy_from_slice(&p);
            out_trace = t;
            forwards += 1;
            start = end;
        }
        Ok(SparseOutput {
            pooled,
            trace: out_trace,
            forwards,
        })
    }

    /// One forward over `batch` right-padded to its longest sequence
    /// (validated by [`Self::encode`]).
    fn forward(&self, batch: &[&[u32]], trace: bool) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let seq = batch.iter().map(|ids| ids.len() as u32).max().unwrap_or(0);
        let nb = batch.len() as u32;
        let rows = nb * seq;
        let (h, inter, vocab, eps) = (cfg.hidden, cfg.intermediate, cfg.vocab, cfg.layer_norm_eps);
        let a = self.acts(nb, seq)?;

        let mut padded = vec![0u32; rows as usize];
        for (b, ids) in batch.iter().enumerate() {
            padded[b * seq as usize..b * seq as usize + ids.len()].copy_from_slice(ids);
        }
        a.ids.write_u32(&padded);
        a.lens
            .write_u32(&batch.iter().map(|ids| ids.len() as u32).collect::<Vec<_>>());

        let mut out_trace = Vec::new();
        let snap = |t: &Tensor, out: &mut Vec<Vec<f32>>| -> Result<(), String> {
            rt.synchronize()?;
            out.push(t.buffer.try_contents_f32()?[..(rows * h) as usize].to_vec());
            Ok(())
        };

        embed_layer_norm(
            rt,
            &a.ids,
            EmbedTables {
                word: &self.word.buffer,
                vocab,
                pos: &self.pos,
                positions: cfg.max_positions,
                type_row: &self.type_row,
                ln_w: &self.embed_norm.w,
                ln_b: &self.embed_norm.b,
            },
            &a.resid.buffer,
            rows,
            h,
            seq,
            eps,
        )?;
        if trace {
            snap(&a.resid, &mut out_trace)?;
        }
        let dims = EncoderAttnDims {
            batch: nb,
            seq,
            heads: cfg.heads,
            heads_kv: cfg.heads,
            head_dim: cfg.head_dim(),
            window: 0,
            scale: 1.0 / (cfg.head_dim() as f32).sqrt(),
        };
        for layer in &self.layers {
            for (dense, out) in [(&layer.q, &a.q), (&layer.k, &a.k), (&layer.v, &a.v)] {
                gemm(&a.resid, &dense.w, out, BACKEND)?;
                bias_add(rt, &out.buffer, &dense.b, rows, h)?;
            }
            encoder_attn(
                rt,
                &a.q.buffer,
                &a.k.buffer,
                &a.v.buffer,
                &a.attn.buffer,
                &a.lens,
                dims,
                false,
            )?;
            gemm(&a.attn, &layer.o.w, &a.y, BACKEND)?;
            let n = &layer.attn_norm;
            bias_residual_layer_norm(rt, &a.y.buffer, &layer.o.b, &a.resid.buffer, &n.w, &n.b, rows, h, eps)?;
            gemm(&a.resid, &layer.inter.w, &a.mid, BACKEND)?;
            bias_gelu_erf(rt, &a.mid.buffer, &layer.inter.b, rows, inter)?;
            gemm(&a.mid, &layer.out.w, &a.y, BACKEND)?;
            let n = &layer.out_norm;
            bias_residual_layer_norm(rt, &a.y.buffer, &layer.out.b, &a.resid.buffer, &n.w, &n.b, rows, h, eps)?;
            if trace {
                snap(&a.resid, &mut out_trace)?;
            }
        }

        // Head: transform, then the decoder over whole sequences in blocks.
        gemm(&a.resid, &self.transform.w, &a.y, BACKEND)?;
        bias_gelu_erf(rt, &a.y.buffer, &self.transform.b, rows, h)?;
        let tn = &self.transform_norm;
        layer_norm(rt, &a.y.buffer, &tn.w, &tn.b, &a.x.buffer, rows, h, eps)?;
        let decoder = self.decoder.as_ref().unwrap_or(&self.word);
        let per_block = (HEAD_BLOCK_ROWS / seq as usize).max(1);
        let v = vocab as usize;
        let mut out = vec![0.0f32; nb as usize * v];
        let mut b0 = 0usize;
        while b0 < batch.len() {
            let b1 = (b0 + per_block).min(batch.len());
            let block_rows = ((b1 - b0) * seq as usize) as u32;
            let x =
                a.x.try_view(&[block_rows as usize, h as usize], b0 * seq as usize * h as usize)?;
            let logits = a.logits.try_view(&[block_rows as usize, vocab as usize], 0)?;
            gemm_nt_f32(&x, decoder, &logits, BACKEND)?;
            let segs: Vec<(u32, u32)> = (b0..b1)
                .map(|b| {
                    let r = ((b - b0) * seq as usize) as u32;
                    (r, r + batch[b].len() as u32)
                })
                .collect();
            let segments = upload_segments(rt, &segs, block_rows)?;
            // Every sequence lies wholly inside one block, so the block's
            // pooled rows start from zero and are final when it ends.
            a.pooled.zero();
            segment_sparse_max(
                rt,
                &a.logits.buffer,
                &self.decoder_bias,
                &segments,
                &a.pooled,
                (b1 - b0) as u32,
                block_rows,
                vocab,
            )?;
            // The next block reuses the logits and pooled buffers.
            rt.synchronize()?;
            out[b0 * v..b1 * v].copy_from_slice(&a.pooled.try_contents_f32()?[..(b1 - b0) * v]);
            b0 = b1;
        }
        Ok((out, out_trace))
    }

    fn acts(&self, nb: u32, seq: u32) -> Result<Acts, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let r = (nb * seq) as usize;
        let (h, inter, v) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.vocab as usize);
        let per_block = (HEAD_BLOCK_ROWS / seq as usize).max(1).min(nb as usize);
        let block = per_block * seq as usize;
        Ok(Acts {
            ids: rt.alloc_buffer(r.max(1) * 4)?,
            lens: rt.alloc_buffer(nb.max(1) as usize * 4)?,
            resid: rt.alloc_tensor_f32(&[r, h])?,
            q: rt.alloc_tensor_f32(&[r, h])?,
            k: rt.alloc_tensor_f32(&[r, h])?,
            v: rt.alloc_tensor_f32(&[r, h])?,
            attn: rt.alloc_tensor_f32(&[r, h])?,
            y: rt.alloc_tensor_f32(&[r, h])?,
            mid: rt.alloc_tensor_f32(&[r, inter])?,
            x: rt.alloc_tensor_f32(&[r, h])?,
            logits: rt.alloc_tensor_f32(&[block, v])?,
            pooled: rt.alloc_buffer(per_block.max(1) * v * 4)?,
        })
    }
}
