//! The Qwen3.5 text model, composed from [`crate::qwen35`]'s kernels and
//! loaded straight from a `.safetensors` checkpoint: the prefill forward, and
//! decode continuing it a token at a time.
//!
//! Each kernel is checked against transformers on its own (`docs/qwen35.md`);
//! this is the composition: the layer order, norm offsets, weight layouts and
//! LM head of inference, written once in `Qwen35Model::layer` for the
//! prefill, the staged prefill and decode alike, and what
//! `tests/qwen35_model.rs` checks end to end against transformers' logits.
//! Training ([`crate::qwen35_train`]) runs its own layer forward, on the
//! training kernels and keeping what its backward reads;
//! `tests/qwen35_train.rs` pins its hidden states to this forward's.
//!
//! # Layer
//!
//! ```text
//! x      = rms_norm(resid) * (1 + input_layernorm.w)
//! resid += mixer(x)                  GDN ("linear_attention") or attention
//! x      = rms_norm(resid) * (1 + post_attention_layernorm.w)
//! resid += down(silu(gate(x)) * up(x))
//! ```
//!
//! and after the last layer `logits = rms_norm(resid) * (1 + norm.w) @ headᵀ`
//! (Qwen3.5 up to the 4B ties the LM head to the embedding, so `head` is
//! `embed`; the 9B's `tie_word_embeddings: false` gives it its own
//! `lm_head.weight`, read from the checkpoint root). Zero-centred norm weights are
//! stored as the checkpoint holds them, `w`, and their kernels form `1 + w`
//! in f32 as transformers does (the residual norms in `qwen35_rms_norm_*`,
//! the attention Q/K norms in theirs); the GDN gated norm's weight is used
//! as stored.
//!
//! # Loading
//!
//! [`Qwen35Model::load`] reads every tensor of the model by name under the
//! checkpoint's prefix and refuses the checkpoint if any tensor under that
//! prefix was not read (`Loader::refuse_unread`): a longer layer stack than
//! the config's, a projection bias, an extra norm. Such a tensor is a
//! parameter of a model this forward does not implement, and ignoring it would
//! return the wrong model's logits without a word. Tensors outside the prefix
//! (the vision tower, `mtp.`) are not looked at, except `lm_head.weight`: an
//! untied config requires it and a tied one refuses a checkpoint that carries
//! it.
//!
//! # Precision
//!
//! [`Precision::Bf16`] is the production forward: bf16 weights, and every GEMM
//! input rounded to bf16 (the norms, gated norm, output gate and SwiGLU write
//! bf16 directly); accumulation, the residual stream, the GDN state and all
//! attention arithmetic stay f32. [`Precision::F32`] widens the weights to f32
//! (exactly) and keeps every activation f32 with exact-f32 GEMMs: the same
//! computation as transformers in fp32, up to operation order, which makes it
//! the tight oracle for the composition. It needs the runtime's relaxed-f32
//! GEMM switched off (the default), and checks that it is.
//!
//! # Prefill and decode
//!
//! One sequence, batch 1, positions from 0. [`Qwen35Model::forward`] and
//! [`Qwen35Model::begin`] prefill and keep nothing: the GDN and conv states
//! start at zero and end with the call, and every attention layer's K/V lives
//! in one buffer the next attention layer overwrites.
//!
//! Decode is tessl's to compose, not the caller's: the layer order is written
//! here once, and a caller wiring `gdn_recurrent` and `attn_prefix_decode`
//! into its own loop would be a second copy of it. [`Qwen35Model::prefill`]
//! runs the same prefill keeping what decode continues from, per layer: the
//! GDN recurrent state (`[v_heads, 128, v_dim]` f32, from `gdn_chunk_forward`'s
//! `state_out`), the conv state (the last `conv_kernel - 1` inputs, from
//! `conv1d_silu`'s), and each attention layer's own K/V, which becomes the
//! shared prefix [`crate::qwen35::attn_prefix_decode`] reads. [`Decode::step`]
//! then runs one token through the same layers on the decode kernels
//! (`conv1d_silu` on carried state, `gdn_recurrent` updating the state in
//! place, the token's K/V written to a suffix cache of `max_new` slots with
//! `attn_qk_norm_rope_suffix`) and returns its logits. The decode kernels sum
//! in a different order from the prefill ones (the recurrent rule against the
//! chunked one, split-KV attention against tiled), so a decoded token's
//! logits equal the prefill's at that position to f32 rounding, not bit for
//! bit; `tests/qwen35_model.rs` holds them to a bound.
//!
//! # Logits
//!
//! [`Qwen35Model::forward_rows`] and [`Staged::logits`] compute the head for
//! chosen positions only ([`LogitRows`]): `[vocab]` f32 per row is about 1 MB
//! at the 2B's 248,320, so a caller that samples the next token asks for
//! [`LogitRows::Last`]. [`Staged::score_answers`] scores a few answer tokens
//! at chosen positions with no logits row at all
//! ([`crate::qwen35::score_answer_rows`]), on a model with or without the
//! packed head.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::cross_entropy::gather_rows_f32;
use crate::gemm::{gemm, gemm_nt_f32, use_relaxed_f32, GemmBackend};
use crate::json::{self, Json, Syntax};
use crate::loader::Loader;
use crate::nn::{AttnDims, DecodeScratch};
use crate::qwen35::{
    self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnDims, GdnParams, GdnProjLayout, GdnScanSlice, GdnWorkspace,
    LmHead, OutCols, SharedPrefix, StateIn,
};
use crate::runtime::{BufferKind, GpuRuntime};
use crate::safetensors::TensorSource;
use crate::tensor::{DType, GpuBuffer, Tensor};

const BACKEND: GemmBackend = GemmBackend::TensorOps;

/// Which mixer a layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// The gated delta net (`"linear_attention"` in the config).
    LinearAttention,
    /// Softmax attention with an output gate (`"full_attention"`).
    FullAttention,
}

/// The text model's shape (a checkpoint's `text_config`).
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Config {
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub layers: Vec<LayerKind>,
    pub gdn: GdnProjLayout,
    /// `linear_conv_kernel_dim`.
    pub conv_kernel: u32,
    pub attn: AttnProjLayout,
    /// `head_dim * partial_rotary_factor`.
    pub rotary_dim: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// `tie_word_embeddings`: the LM head is the embedding (Qwen3.5-0.8B to
    /// 4B). `false` (the 9B and up) gives the model its own `lm_head.weight`,
    /// a `[vocab, hidden]` parameter beside `embed_tokens.weight` that the
    /// loader reads from the checkpoint root (outside the tower's prefix, as
    /// transformers names it) and that the gradients, optimizer and
    /// parameter table carry as an entry of its own.
    pub tie_word_embeddings: bool,
}

/// The checkpoint tensor that is an untied model's LM head, at the
/// checkpoint root. A tied checkpoint holds none.
pub const LM_HEAD_NAME: &str = "lm_head.weight";

impl Qwen35Config {
    /// `Qwen/Qwen3.5-2B-Base`'s `config.json` `text_config`: 24 layers, every
    /// fourth full attention; GDN 16 key / 16 value heads of 128; attention 8
    /// query / 2 KV heads of 256 with a quarter of each head rotated
    /// (theta 1e7); MLP 6144; vocabulary 248320.
    pub fn qwen35_2b() -> Result<Self, String> {
        let cfg = Self {
            hidden: 2048,
            intermediate: 6144,
            vocab: 248_320,
            layers: (0..24)
                .map(|l| {
                    if l % 4 == 3 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect(),
            gdn: GdnProjLayout::new(16, 16, 128)?,
            conv_kernel: 4,
            attn: AttnProjLayout::new(8, 2, 256)?,
            rotary_dim: 64,
            rope_theta: 1e7,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The text model a Hugging Face `config.json` describes: its
    /// `text_config` (a `Qwen3_5ForConditionalGeneration` checkpoint) or the
    /// root (a text-only one, `model_type: "qwen3_5_text"`).
    ///
    /// Everything the forward depends on is read, and everything it does not
    /// implement is refused by name rather than ignored: MoE, attention bias, an ungated attention output, an activation other
    /// than SiLU, `mlp_only_layers`, a RoPE other than the default, a
    /// `mamba_ssm_dtype` other than `"float32"`, a non-zero `attention_dropout`,
    /// and key or head dims the kernels are not compiled for.
    ///
    /// Keys the forward has no use for are left unread: `dtype` (the caller
    /// picks the [`Precision`]), `eos_token_id`, `initializer_range`,
    /// `max_position_embeddings`, `use_cache`, `model_type`, the `mtp_*` keys
    /// (the checkpoint's `mtp.` tensors are outside the prefix), and
    /// `rope_parameters.mrope_*` (text positions are one stream, so the
    /// multimodal sections do not apply).
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
        let field = |k: &str| c.get(k).ok_or_else(|| format!("config.json: missing {k:?}"));
        let uint = |k: &str| -> Result<u32, String> {
            match field(k)? {
                Json::Num { uint: Some(n), .. } => {
                    u32::try_from(*n).map_err(|_| format!("config.json: {k} = {n} exceeds u32"))
                }
                v => Err(format!("config.json: {k} must be a non-negative integer, got {v:?}")),
            }
        };
        let float = |v: &Json, k: &str| -> Result<f64, String> {
            match v {
                Json::Num { value, .. } => Ok(*value),
                v => Err(format!("config.json: {k} must be a number, got {v:?}")),
            }
        };
        let flag = |k: &str| -> Result<Option<bool>, String> {
            match c.get(k) {
                None => Ok(None),
                Some(Json::Bool(b)) => Ok(Some(*b)),
                Some(v) => Err(format!("config.json: {k} must be true or false, got {v:?}")),
            }
        };
        let string = |k: &str| -> Result<Option<&str>, String> {
            match c.get(k) {
                None => Ok(None),
                Some(Json::Str(s)) => Ok(Some(s.as_str())),
                Some(v) => Err(format!("config.json: {k} must be a string, got {v:?}")),
            }
        };

        // `tie_word_embeddings`, at the root and in `text_config`: whichever
        // are present must be booleans and agree, and one must be present
        // (transformers defaults an absent key to tied; a config that does not
        // say is refused rather than guessed, as the head is the whole
        // difference between two checkpoint layouts).
        let mut tie: Option<bool> = None;
        for t in [root.get("tie_word_embeddings"), c.get("tie_word_embeddings")]
            .into_iter()
            .flatten()
        {
            let Json::Bool(b) = t else {
                return Err(format!(
                    "config.json: tie_word_embeddings must be true or false, got {t:?}"
                ));
            };
            if tie.replace(*b).is_some_and(|prev| prev != *b) {
                return Err("config.json: tie_word_embeddings disagrees between the root and text_config".into());
            }
        }
        let tie_word_embeddings =
            tie.ok_or("config.json: tie_word_embeddings is not set; the LM head must be stated as tied or untied")?;

        // Features the forward does not implement.
        for k in ["num_experts", "num_local_experts", "moe_intermediate_size"] {
            if c.get(k).is_some() {
                return Err(format!(
                    "config.json: {k} is set; mixture-of-experts models are not supported"
                ));
            }
        }
        if flag("attention_bias")? == Some(true) {
            return Err("config.json: attention_bias is not supported".into());
        }
        // The GDN rules run in float32 whatever this says: transformers casts
        // them unconditionally (modeling_qwen3_5.py:266, 347) and declares no
        // `mamba_ssm_dtype` at all, so an absent key is float32 too (as in
        // ojas-qwen35). Any other dtype would be a request this forward cannot
        // honour.
        if let Some(d) = string("mamba_ssm_dtype")? {
            if d != "float32" {
                return Err(format!(
                    "config.json: mamba_ssm_dtype {d:?} is not supported (the GDN state is float32)"
                ));
            }
        }
        // Inference and dropout-free training only: no kernel drops attention
        // probabilities (transformers' default is 0.0,
        // configuration_qwen3_5.py:96).
        if let Some(v) = c.get("attention_dropout") {
            let p = float(v, "attention_dropout")?;
            if p != 0.0 {
                return Err(format!("config.json: attention_dropout {p} is not supported (only 0)"));
            }
        }
        if flag("attn_output_gate")? == Some(false) {
            return Err("config.json: an ungated attention output is not supported".into());
        }
        if let Some(act) = string("hidden_act")? {
            if act != "silu" {
                return Err(format!(
                    "config.json: hidden_act {act:?} is not supported (only \"silu\")"
                ));
            }
        }
        match c.get("mlp_only_layers") {
            None => {}
            Some(Json::Array(a)) if a.is_empty() => {}
            Some(v) => return Err(format!("config.json: mlp_only_layers {v:?} is not supported")),
        }

        let hidden = uint("hidden_size")?;
        let n_layers = uint("num_hidden_layers")?;
        let layers: Vec<LayerKind> = match c.get("layer_types") {
            Some(Json::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(i, t)| match t {
                    Json::Str(s) if s == "linear_attention" => Ok(LayerKind::LinearAttention),
                    Json::Str(s) if s == "full_attention" => Ok(LayerKind::FullAttention),
                    t => Err(format!(
                        "config.json: layer_types[{i}] = {t:?} is not a known layer type"
                    )),
                })
                .collect::<Result<_, _>>()?,
            Some(v) => return Err(format!("config.json: layer_types must be an array, got {v:?}")),
            None => {
                let every = uint("full_attention_interval")?;
                if every == 0 {
                    return Err("config.json: full_attention_interval must be non-zero".into());
                }
                (0..n_layers)
                    .map(|l| {
                        if (l + 1) % every == 0 {
                            LayerKind::FullAttention
                        } else {
                            LayerKind::LinearAttention
                        }
                    })
                    .collect()
            }
        };
        if layers.len() != n_layers as usize {
            return Err(format!(
                "config.json: layer_types has {} entries but num_hidden_layers is {n_layers}",
                layers.len()
            ));
        }

        let key_dim = uint("linear_key_head_dim")?;
        if key_dim != qwen35::GDN_KEY_DIM {
            return Err(format!(
                "config.json: linear_key_head_dim {key_dim} is not supported (the GDN kernels are \
                 compiled for {})",
                qwen35::GDN_KEY_DIM
            ));
        }
        let gdn = GdnProjLayout::new(
            uint("linear_num_key_heads")?,
            uint("linear_num_value_heads")?,
            uint("linear_value_head_dim")?,
        )?;
        let head_dim = uint("head_dim")?;
        let attn = AttnProjLayout::new(uint("num_attention_heads")?, uint("num_key_value_heads")?, head_dim)?;

        // RoPE: rope_parameters (transformers 5) or the older flat fields.
        let rope = c.get("rope_parameters").unwrap_or(c);
        if let Some(t) = rope.get("rope_type").or_else(|| rope.get("type")) {
            if t != &Json::Str("default".into()) {
                return Err(format!(
                    "config.json: rope_type {t:?} is not supported (only \"default\")"
                ));
            }
        }
        let theta = float(
            rope.get("rope_theta").ok_or("config.json: missing \"rope_theta\"")?,
            "rope_theta",
        )?;
        let factor = match rope.get("partial_rotary_factor") {
            Some(v) => float(v, "partial_rotary_factor")?,
            None => 1.0,
        };
        let rotary = f64::from(head_dim) * factor;
        if !(factor > 0.0 && factor <= 1.0) || rotary.fract() != 0.0 {
            return Err(format!(
                "config.json: head_dim {head_dim} x partial_rotary_factor {factor} is not a whole \
                 number of rotated dims"
            ));
        }
        let eps = float(field("rms_norm_eps")?, "rms_norm_eps")?;

        let cfg = Self {
            hidden,
            intermediate: uint("intermediate_size")?,
            vocab: uint("vocab_size")?,
            layers,
            gdn,
            conv_kernel: uint("linear_conv_kernel_dim")?,
            attn,
            rotary_dim: rotary as u32,
            rope_theta: theta as f32,
            rms_norm_eps: eps as f32,
            tie_word_embeddings,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// [`Self::from_config_json`] on a file.
    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.intermediate == 0 || self.vocab == 0 || self.layers.is_empty() {
            return Err("Qwen35Config: hidden, intermediate, vocab and layers must be non-zero".into());
        }
        if !qwen35::CONV_KERNEL_WIDTHS.contains(&self.conv_kernel) {
            return Err(format!(
                "Qwen35Config: conv_kernel {} is not supported (the conv kernels are compiled for {:?})",
                self.conv_kernel,
                qwen35::CONV_KERNEL_WIDTHS
            ));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0)
            || !(self.rope_theta.is_finite() && self.rope_theta > 0.0)
        {
            return Err("Qwen35Config: rms_norm_eps and rope_theta must be positive".into());
        }
        if self.attn.q_heads() % self.attn.kv_heads() != 0 {
            return Err(format!(
                "Qwen35Config: {} query heads do not group over {} kv_heads (num_attention_heads must be a \
                 multiple of num_key_value_heads)",
                self.attn.q_heads(),
                self.attn.kv_heads()
            ));
        }
        if self.rotary_dim % 2 != 0 || self.rotary_dim > self.attn.head_dim() {
            return Err("Qwen35Config: rotary_dim must be even and at most head_dim".into());
        }
        if self.attn.head_dim() != qwen35::PREFIX_ATTN_HEAD_DIM {
            return Err(format!(
                "Qwen35Config: attention head_dim {} is not supported (the prefill attention \
                 kernels are compiled for {})",
                self.attn.head_dim(),
                qwen35::PREFIX_ATTN_HEAD_DIM
            ));
        }
        Ok(())
    }
}

/// Weight and activation precision of the forward (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Bf16,
    F32,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Precision::Bf16 => DType::BF16,
            Precision::F32 => DType::F32,
        }
    }
}

pub(crate) struct GdnWeights {
    pub(crate) w_in: Tensor,
    pub(crate) w_out: Tensor,
    pub(crate) conv_w: GpuBuffer,
    pub(crate) a_log: GpuBuffer,
    pub(crate) dt_bias: GpuBuffer,
    pub(crate) norm_w: GpuBuffer,
}

pub(crate) struct AttnWeights {
    pub(crate) w_in: Tensor,
    pub(crate) w_out: Tensor,
    pub(crate) q_norm: GpuBuffer,
    pub(crate) k_norm: GpuBuffer,
}

pub(crate) enum Mixer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

pub(crate) struct Layer {
    pub(crate) input_norm: GpuBuffer,
    pub(crate) post_norm: GpuBuffer,
    pub(crate) mixer: Mixer,
    pub(crate) gate: Tensor,
    pub(crate) up: Tensor,
    pub(crate) down: Tensor,
}

/// A loaded model: weights on the device in the layouts the kernels read.
pub struct Qwen35Model {
    pub(crate) cfg: Qwen35Config,
    pub(crate) precision: Precision,
    pub(crate) rt: Arc<GpuRuntime>,
    /// `[vocab, hidden]` in the forward's precision: the embedding (the
    /// gather's table; tied, also the head, in f32 the head GEMM's transposed
    /// operand, and what training reads and writes).
    pub(crate) embed: Tensor,
    /// An untied model's own LM head (`lm_head.weight`), `[vocab, hidden]` in
    /// the forward's precision; `None` when tied, where `embed` is the head.
    /// Read through [`Self::head_table`]: the head GEMM (f32), the packed
    /// bf16 copy's source, `score_answers` and training's cross-entropy.
    pub(crate) lm_head: Option<Tensor>,
    /// Bf16 only: the head (the tied embedding, or `lm_head`) packed `[hidden, vocab]` for an NN GEMM,
    /// which measured faster than the NT GEMM over `embed` at the 2B's
    /// head, even with the NT tile tuned for it (5-11%,
    /// `bench/results/bf16_nt_lm_head_m5pro.txt`). The
    /// bf16 model is not trained, so the two copies never diverge.
    pub(crate) lm_head_bf16: Option<Tensor>,
    pub(crate) final_norm: GpuBuffer,
    pub(crate) layers: Vec<Layer>,
    /// The GDN chunk scan's column slice for every prefill this model runs
    /// ([`Self::with_gdn_scan_slice`]); the library default unless chosen.
    gdn_scan: GdnScanSlice,
    /// Bumped by every write to the weights after load
    /// ([`Self::bump_param_generation`]). A [`crate::qwen35_train::PendingStep`]
    /// records it at the forward, and the backward, which rebuilds each layer
    /// from the weights it finds, refuses a step whose weights have moved.
    param_generation: AtomicU64,
    /// Test hook ([`Self::fail_layer_after_for_test`]): layers still to run
    /// before one fails; `usize::MAX` is off.
    layer_fault: AtomicUsize,
    /// Test hook ([`Self::fail_host_read_after_for_test`]): host reads still
    /// to succeed before one fails; `usize::MAX` is off.
    read_fault: AtomicUsize,
}

impl Qwen35Model {
    /// How many times the weights have been written since load.
    pub(crate) fn param_generation(&self) -> u64 {
        self.param_generation.load(Ordering::Relaxed)
    }

    /// Record a write to the weights. Called before the write is encoded, so
    /// a write that fails part way still invalidates a pending step.
    pub(crate) fn bump_param_generation(&self) {
        self.param_generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Make the layer that runs after `n` more successful ones fail, once,
    /// with an error: how a test reaches the failure after the embed and
    /// after earlier layers' encodes that [`Staged::advance_to`] and
    /// [`Decode::step`] must survive as poisoned. Tests only.
    #[doc(hidden)]
    pub fn fail_layer_after_for_test(&self, n: usize) {
        self.layer_fault.store(n, Ordering::Relaxed);
    }

    /// Make the host read that comes after `n` more successful ones fail with
    /// an error, once, after the work ran and the runtime synchronized. The
    /// reads, in order: one per layer of [`Staged::advance_to`]'s trace,
    /// [`Staged::logits`], [`Staged::score_answers`], [`Decode::step`]'s
    /// logits, and the final-norm row of [`Qwen35Model::forward`]'s trace. How
    /// a test reaches the failure each must return as `Err`, and a step must
    /// survive without advancing. Tests only.
    #[doc(hidden)]
    pub fn fail_host_read_after_for_test(&self, n: usize) {
        self.read_fault.store(n, Ordering::Relaxed);
    }

    fn take_read_fault(&self) -> Result<(), String> {
        count_down_fault(
            &self.read_fault,
            "host read: injected fault (fail_host_read_after_for_test)",
        )
    }

    fn take_layer_fault(&self) -> Result<(), String> {
        count_down_fault(&self.layer_fault, "layer: injected fault (fail_layer_after_for_test)")
    }
}

/// What [`Qwen35Model::forward`] and [`Qwen35Model::forward_rows`] return.
pub struct ForwardOutput {
    /// `[rows, vocab]` f32, one row per position the [`LogitRows`] chose
    /// (`[tokens, vocab]` from [`Qwen35Model::forward`]).
    pub logits: Vec<f32>,
    /// With `trace`: the residual stream after each layer, `[tokens, hidden]`
    /// each; then the final norm's output. Empty otherwise.
    pub trace: Vec<Vec<f32>>,
}

/// Which positions' logits a forward computes and returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitRows<'a> {
    /// Every position: `[tokens, vocab]`.
    All,
    /// The last position only, `[1, vocab]`: what choosing the next token reads.
    Last,
    /// These positions, in this order (repeats allowed): `[rows.len(), vocab]`.
    Rows(&'a [u32]),
}

/// What [`Staged::score_answers`] returns, `[positions, answers]` f32 each.
pub struct AnswerScores {
    /// The answer tokens' logits at each position.
    pub logits: Vec<f32>,
    /// Their log-softmax over the answer set alone (not the vocabulary).
    pub logprobs: Vec<f32>,
}

impl Qwen35Model {
    /// Load the text tower from `st`, whose tensors are named `{prefix}...`
    /// (`"model.language_model."` in the Qwen3.5 checkpoints). Every tensor's
    /// shape is checked against `cfg`; a missing one is an error.
    pub fn load(
        rt: &Arc<GpuRuntime>,
        st: &dyn TensorSource,
        prefix: &str,
        cfg: Qwen35Config,
        precision: Precision,
    ) -> Result<Self, String> {
        Self::load_impl(rt, st, prefix, cfg, precision, true)
    }

    /// [`Self::load`] without the LM head: for a model that is read through
    /// [`Self::begin`] (hidden states, never logits). At bf16 this skips the
    /// `[hidden, vocab]` packed head copy of the embedding (1.27 GB at the
    /// 4B's shapes), and the model then has no head: [`Self::forward`],
    /// [`Self::prefill`] and [`Staged::logits`] refuse it. At f32 the
    /// embedding is the head (the tied one), so an f32 tower runs all of
    /// them. [`Staged::score_answers`] reads the embedding's rows directly and
    /// runs on any tower.
    pub fn load_tower(
        rt: &Arc<GpuRuntime>,
        st: &dyn TensorSource,
        prefix: &str,
        cfg: Qwen35Config,
        precision: Precision,
    ) -> Result<Self, String> {
        Self::load_impl(rt, st, prefix, cfg, precision, false)
    }

    fn load_impl(
        rt: &Arc<GpuRuntime>,
        st: &dyn TensorSource,
        prefix: &str,
        cfg: Qwen35Config,
        precision: Precision,
        with_head: bool,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let ld = Loader::new(rt, st, prefix);
        let (h, inter, vocab) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.vocab as usize);

        // The embedding: one [vocab, hidden] table in the model's precision,
        // which the gather reads by row and, tied, the LM head as the
        // transposed right operand of one GEMM. Untied, the head is its own
        // table read from the checkpoint root; a tied checkpoint carrying one
        // is refused (it would be read by nothing).
        let tied = cfg.tie_word_embeddings;
        if tied && ld.has_root(LM_HEAD_NAME) {
            return Err(format!(
                "Qwen35Model::load: the config ties the word embeddings but the checkpoint carries {LM_HEAD_NAME}; \
                 the tied forward would ignore it (tie_word_embeddings is false for such a model)"
            ));
        }
        let (embed, tied_head_bf16) = Self::load_table(
            rt,
            &ld,
            "embed_tokens.weight",
            false,
            [vocab, h],
            precision,
            with_head && tied,
        )?;
        let (lm_head, untied_head_bf16) = if tied {
            (None, None)
        } else {
            let (t, packed) = Self::load_table(rt, &ld, LM_HEAD_NAME, true, [vocab, h], precision, with_head)?;
            (Some(t), packed)
        };
        let lm_head_bf16 = tied_head_bf16.or(untied_head_bf16);
        let final_norm = ld.norm("norm.weight", h)?;

        let (g, a) = (cfg.gdn, cfg.attn);
        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (l, kind) in cfg.layers.iter().enumerate() {
            let p = |s: &str| format!("layers.{l}.{s}");
            let mixer = match kind {
                LayerKind::LinearAttention => {
                    let pw = g.part_widths();
                    let conv_dim = g.conv_dim() as usize;
                    let kw = cfg.conv_kernel as usize;
                    let vh = g.v_heads() as usize;
                    Mixer::Gdn(GdnWeights {
                        w_in: ld.linear(
                            &[
                                (&p("linear_attn.in_proj_qkv.weight"), pw[0]),
                                (&p("linear_attn.in_proj_z.weight"), pw[1]),
                                (&p("linear_attn.in_proj_b.weight"), pw[2]),
                                (&p("linear_attn.in_proj_a.weight"), pw[3]),
                            ],
                            h,
                            precision,
                        )?,
                        w_out: ld.linear(
                            &[(&p("linear_attn.out_proj.weight"), h)],
                            g.value_dim() as usize,
                            precision,
                        )?,
                        conv_w: ld.buf(&ld.f32(&p("linear_attn.conv1d.weight"), &[conv_dim, 1, kw])?)?,
                        a_log: ld.buf(&ld.f32(&p("linear_attn.A_log"), &[vh])?)?,
                        dt_bias: ld.buf(&ld.f32(&p("linear_attn.dt_bias"), &[vh])?)?,
                        norm_w: ld.buf(&ld.f32(&p("linear_attn.norm.weight"), &[g.v_dim() as usize])?)?,
                    })
                }
                LayerKind::FullAttention => {
                    let pw = a.part_widths();
                    let d = a.head_dim() as usize;
                    Mixer::Attn(AttnWeights {
                        w_in: ld.linear(
                            &[
                                (&p("self_attn.q_proj.weight"), pw[0]),
                                (&p("self_attn.k_proj.weight"), pw[1]),
                                (&p("self_attn.v_proj.weight"), pw[2]),
                            ],
                            h,
                            precision,
                        )?,
                        w_out: ld.linear(
                            &[(&p("self_attn.o_proj.weight"), h)],
                            (a.q_heads() * a.head_dim()) as usize,
                            precision,
                        )?,
                        // The kernel applies (1 + w) itself.
                        q_norm: ld.buf(&ld.f32(&p("self_attn.q_norm.weight"), &[d])?)?,
                        k_norm: ld.buf(&ld.f32(&p("self_attn.k_norm.weight"), &[d])?)?,
                    })
                }
            };
            layers.push(Layer {
                input_norm: ld.norm(&p("input_layernorm.weight"), h)?,
                post_norm: ld.norm(&p("post_attention_layernorm.weight"), h)?,
                mixer,
                gate: ld.linear(&[(&p("mlp.gate_proj.weight"), inter)], h, precision)?,
                up: ld.linear(&[(&p("mlp.up_proj.weight"), inter)], h, precision)?,
                down: ld.linear(&[(&p("mlp.down_proj.weight"), h)], inter, precision)?,
            });
        }
        // Every tensor under the prefix is a parameter of this model: a longer
        // layer stack, a bias or an extra norm is a different model.
        ld.refuse_unread()?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            precision,
            rt: Arc::clone(rt),
            embed,
            lm_head,
            lm_head_bf16,
            final_norm,
            layers,
            gdn_scan: GdnScanSlice::default(),
            param_generation: AtomicU64::new(0),
            layer_fault: AtomicUsize::new(usize::MAX),
            read_fault: AtomicUsize::new(usize::MAX),
        })
    }

    /// One `[vocab, hidden]` table (the embedding, or an untied head at the
    /// checkpoint `root`) read into a device tensor in `precision`; with
    /// `pack` at bf16, also packed `[hidden, vocab]` for the head's NN GEMM.
    /// The one place a table is read, so the embedding and an untied head
    /// cannot load differently.
    fn load_table(
        rt: &Arc<GpuRuntime>,
        ld: &Loader<'_>,
        name: &str,
        root: bool,
        shape: [usize; 2],
        precision: Precision,
        pack: bool,
    ) -> Result<(Tensor, Option<Tensor>), String> {
        let [vocab, h] = shape;
        match precision {
            Precision::Bf16 => {
                let t = rt.alloc_tensor_bf16_hot(&[vocab, h])?;
                if pack {
                    // One host copy of the table, placed into both device
                    // tensors and dropped; never a second, packed host copy.
                    let bits = if root {
                        ld.bf16_root(name, &[vocab, h])?
                    } else {
                        ld.bf16(name, &[vocab, h])?
                    };
                    t.buffer
                        .try_write_bf16_bits(&bits)
                        .map_err(|e| format!("Qwen35Model::load: {name}: {e}"))?;
                    Ok((t, Some(Self::pack_head(rt, &bits, vocab, h)?)))
                } else {
                    // No head to transpose into: read straight into the table.
                    if root {
                        ld.bf16_into_root(name, &[vocab, h], &t)?;
                    } else {
                        ld.bf16_into(name, &[vocab, h], &t)?;
                    }
                    Ok((t, None))
                }
            }
            Precision::F32 => {
                // Widened straight into the table: no host copy at all.
                let t = rt.alloc_tensor_f32_hot(&[vocab, h])?;
                if root {
                    ld.f32_into_root(name, &[vocab, h], &t)?;
                } else {
                    ld.f32_into(name, &[vocab, h], &t)?;
                }
                Ok((t, None))
            }
        }
    }

    /// The tied head packed `[hidden, vocab]` from the embedding's bf16 `bits`
    /// (`[vocab, hidden]`): the one place [`Self::load`] and [`Self::random`]
    /// build it.
    fn pack_head(rt: &Arc<GpuRuntime>, bits: &[u16], vocab: usize, h: usize) -> Result<Tensor, String> {
        let head = rt.alloc_tensor_bf16_hot(&[h, vocab])?;
        qwen35::place_linear_part(&mut head.buffer.try_contents_u16()?, vocab, 0, bits, vocab, h)?;
        Ok(head)
    }

    /// A model of `cfg`'s shapes with seeded random weights and no checkpoint,
    /// loaded as [`Self::load`] loads (at bf16, with the packed LM head, so
    /// [`Self::forward`] and [`Self::prefill`] run): for benches and tests at
    /// shapes with no local checkpoint. Every matrix and the conv weights are
    /// uniform with standard deviation 0.02 (transformers' `initializer_range`),
    /// the norms' `w`, `A_log` and `dt_bias` zero. Not a trained model. The
    /// head is the embedding's own bits, so it costs one more `[hidden,
    /// vocab]` table (1.0 GB at the 2B's) and [`Self::random_tower`]'s weights
    /// are this model's, bit for bit.
    pub fn random(rt: &Arc<GpuRuntime>, cfg: Qwen35Config, precision: Precision, seed: u64) -> Result<Self, String> {
        Self::random_impl(rt, cfg, precision, seed, true)
    }

    /// [`Self::random`] loaded as [`Self::load_tower`] loads (no packed LM
    /// head): for memory probes and tests that read hidden states only. At
    /// bf16, [`Self::forward`], [`Self::prefill`] and [`Staged::logits`] refuse it.
    pub fn random_tower(
        rt: &Arc<GpuRuntime>,
        cfg: Qwen35Config,
        precision: Precision,
        seed: u64,
    ) -> Result<Self, String> {
        Self::random_impl(rt, cfg, precision, seed, false)
    }

    fn random_impl(
        rt: &Arc<GpuRuntime>,
        cfg: Qwen35Config,
        precision: Precision,
        seed: u64,
        with_head: bool,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        // splitmix64, then a uniform in [-a, a) with variance 0.02^2.
        let half = 0.02f32 * 3f32.sqrt();
        let mut uniform = move |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    z ^= z >> 31;
                    ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * half
                })
                .collect()
        };
        let (h, inter, vocab) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.vocab as usize);
        // The embedding's draw is the first of the stream whether or not a head
        // is built, so the head reuses its bits and the tower stays the same.
        let embed_data = uniform(vocab * h);
        let upload = |shape: &[usize], p: Precision, data: &[f32]| -> Result<Tensor, String> {
            match p {
                Precision::Bf16 => {
                    let t = rt.alloc_tensor_bf16_hot(shape)?;
                    t.buffer
                        .try_write_bf16_bits(&crate::tensor::f32_slice_to_bf16(data))
                        .map_err(|e| format!("Qwen35Model::random: {e}"))?;
                    Ok(t)
                }
                Precision::F32 => {
                    let t = rt.alloc_tensor_f32_hot(shape)?;
                    t.buffer
                        .try_write_f32(data)
                        .map_err(|e| format!("Qwen35Model::random: {e}"))?;
                    Ok(t)
                }
            }
        };
        let mut tensor = |shape: &[usize], p: Precision| -> Result<Tensor, String> {
            let data = uniform(shape.iter().product());
            upload(shape, p, &data)
        };
        let zeros = |n: usize| -> Result<GpuBuffer, String> {
            let b = rt.alloc_buffer_hot(n.max(1) * 4)?;
            b.try_zero().map_err(|e| format!("Qwen35Model::random: {e}"))?;
            Ok(b)
        };
        let (g, a) = (cfg.gdn, cfg.attn);
        let embed = upload(&[vocab, h], precision, &embed_data)?;
        // Tied, the head is the embedding's own draw; an untied head is drawn
        // after every layer, so the tower's weights are the tied model's.
        let tied_head_bf16 = if with_head && cfg.tie_word_embeddings && precision == Precision::Bf16 {
            Some(Self::pack_head(
                rt,
                &crate::tensor::f32_slice_to_bf16(&embed_data),
                vocab,
                h,
            )?)
        } else {
            None
        };
        drop(embed_data);
        let mut layers = Vec::with_capacity(cfg.layers.len());
        for kind in &cfg.layers {
            let mixer = match kind {
                LayerKind::LinearAttention => {
                    let conv_w = tensor(&[(g.conv_dim() * cfg.conv_kernel) as usize], Precision::F32)?.buffer;
                    Mixer::Gdn(GdnWeights {
                        w_in: tensor(&[h, g.width() as usize], precision)?,
                        w_out: tensor(&[g.value_dim() as usize, h], precision)?,
                        conv_w,
                        a_log: zeros(g.v_heads() as usize)?,
                        dt_bias: zeros(g.v_heads() as usize)?,
                        norm_w: zeros(g.v_dim() as usize)?,
                    })
                }
                LayerKind::FullAttention => Mixer::Attn(AttnWeights {
                    w_in: tensor(&[h, a.width() as usize], precision)?,
                    w_out: tensor(&[(a.q_heads() * a.head_dim()) as usize, h], precision)?,
                    q_norm: zeros(a.head_dim() as usize)?,
                    k_norm: zeros(a.head_dim() as usize)?,
                }),
            };
            layers.push(Layer {
                input_norm: zeros(h)?,
                post_norm: zeros(h)?,
                mixer,
                gate: tensor(&[h, inter], precision)?,
                up: tensor(&[h, inter], precision)?,
                down: tensor(&[inter, h], precision)?,
            });
        }
        let final_norm = zeros(h)?;
        let (lm_head, lm_head_bf16) = if cfg.tie_word_embeddings {
            (None, tied_head_bf16)
        } else {
            let data = uniform(vocab * h);
            let packed = if with_head && precision == Precision::Bf16 {
                Some(Self::pack_head(rt, &crate::tensor::f32_slice_to_bf16(&data), vocab, h)?)
            } else {
                None
            };
            (Some(upload(&[vocab, h], precision, &data)?), packed)
        };
        rt.synchronize()?;
        Ok(Self {
            cfg,
            precision,
            rt: Arc::clone(rt),
            embed,
            lm_head,
            lm_head_bf16,
            final_norm,
            layers,
            gdn_scan: GdnScanSlice::default(),
            param_generation: AtomicU64::new(0),
            layer_fault: AtomicUsize::new(usize::MAX),
            read_fault: AtomicUsize::new(usize::MAX),
        })
    }

    /// This model with the GDN chunk scan run in `slice`-column slices by
    /// every prefill (`forward`, `begin`, `prefill`). The slices are
    /// bit-identical, so this changes speed only; the default is the
    /// library's ([`GdnScanSlice::default`]). Decode's recurrence has its own
    /// width and is not affected.
    pub fn with_gdn_scan_slice(mut self, slice: GdnScanSlice) -> Self {
        self.gdn_scan = slice;
        self
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// What the model is and how it stores itself, in one line: its shape,
    /// the matrices' dtype (the embedding and the packed projections) and the
    /// rest's (always f32), and whether it carries the inference forward's
    /// packed LM head.
    pub fn describe(&self) -> String {
        let c = &self.cfg;
        let attn = c.layers.iter().filter(|&&k| k == LayerKind::FullAttention).count();
        let matrices = match self.precision {
            Precision::Bf16 => "bf16",
            Precision::F32 => "f32",
        };
        format!(
            "Qwen3.5 text: hidden {}, intermediate {}, vocab {}, {} layers ({} GDN, {attn} attention); \
             matrices {matrices}, norms/conv/gates f32; LM head {}",
            c.hidden,
            c.intermediate,
            c.vocab,
            c.layers.len(),
            c.layers.len() - attn,
            if self.lm_head_bf16.is_some() {
                "packed bf16 copy"
            } else if self.precision == Precision::F32 {
                if self.lm_head.is_some() {
                    "untied (its own table)"
                } else {
                    "tied to the embedding"
                }
            } else {
                "none (load_tower / random_tower; only begin / Staged::score_answers read it)"
            }
        )
    }

    /// Prefill `ids` (one sequence, positions from 0) and return every
    /// position's logits; with `trace`, also the residual stream after each
    /// layer and the final norm's output (one host read per layer, so slower).
    pub fn forward(&self, ids: &[u32], trace: bool) -> Result<ForwardOutput, String> {
        self.forward_rows(ids, LogitRows::All, trace)
    }

    /// [`Self::forward`] with the LM head run for the positions `rows`
    /// chooses only: the logits are `[rows, vocab]`, and the `[tokens,
    /// vocab]` of every position is neither allocated nor read unless `rows`
    /// is [`LogitRows::All`]. The trace is the same whatever `rows` is.
    pub fn forward_rows(&self, ids: &[u32], rows: LogitRows<'_>, trace: bool) -> Result<ForwardOutput, String> {
        self.require_head("Qwen35Model::forward")?;
        check_rows("Qwen35Model::forward_rows", rows, ids.len())?;
        // The logits the head writes, `[rows, vocab]` f32, after the activations.
        let n_rows = match rows {
            LogitRows::All => ids.len(),
            LogitRows::Last => ids.len().min(1),
            LogitRows::Rows(r) => r.len(),
        };
        let logits_bytes = f32_bytes(n_rows.saturating_mul(self.cfg.vocab as usize));
        let mut s = self.begin_impl("Qwen35Model::forward", ids, trace, None, logits_bytes)?;
        s.advance_to(self.layers.len())?;
        let logits = s.logits(rows)?;
        let mut out_trace = s.trace.take().unwrap_or_default();
        if trace {
            // The final norm of every row (the head's own pass covers only
            // the chosen rows unless all were chosen); the same kernel on the
            // same stream, so the same bits either way.
            self.norm(&s.a.resid, &self.final_norm, &s.a.x)?;
            self.rt.synchronize()?;
            self.take_read_fault()?;
            out_trace.push(read_rows(&s.a.x, self.precision)?);
        }
        Ok(ForwardOutput {
            logits,
            trace: out_trace,
        })
    }

    /// Start a prefill of `ids` (one sequence, positions from 0) that runs
    /// only as many layers as asked: [`Staged::advance_to`] continues it,
    /// [`Staged::final_norm_f32`] reads the final norm of the residual stream
    /// at the layer reached, [`Staged::logits`] the head for chosen rows and
    /// [`Staged::score_answers`] answer tokens' scores. Nothing of the LM head
    /// is allocated until one of the last two asks, so this works on a
    /// [`Self::load_tower`] model and at sequence lengths whose `[tokens,
    /// vocab]` logits would not fit.
    pub fn begin(&self, ids: &[u32]) -> Result<Staged<'_>, String> {
        self.begin_impl("Qwen35Model::begin", ids, false, None, 0)
    }

    /// Prefill `ids` (one sequence, positions from 0) for decoding up to
    /// `max_new` tokens after it. Returns the [`Decode`] session that
    /// continues the sequence and the last position's logits (`[vocab]` f32:
    /// the distribution of the first new token).
    ///
    /// The prefill is [`Self::forward`]'s, layer for layer, except that each
    /// layer keeps its state (see the module docs): every attention layer
    /// holds `ids.len() + max_new` positions of K/V and every GDN layer its
    /// recurrent and conv states until the session is dropped. Refused on a
    /// bf16 [`Self::load_tower`] model, which has no head.
    pub fn prefill(&self, ids: &[u32], max_new: u32) -> Result<(Decode<'_>, Vec<f32>), String> {
        const WHAT: &str = "Qwen35Model::prefill";
        self.require_head(WHAT)?;
        let t = self.check_ids(WHAT, ids)?;
        if max_new == 0 {
            return Err(format!("{WHAT}: max_new must be at least 1"));
        }
        let positions = t
            .checked_add(max_new)
            .ok_or_else(|| format!("{WHAT}: {t} tokens plus max_new {max_new} exceed u32 positions"))?;
        let (rt, cfg) = (&self.rt, &self.cfg);
        // Everything the prefill allocates, checked as a training step is
        // before any GPU work: past the recommended working set Metal pages
        // the resident set or the system runs out of memory, rather than an
        // allocation failing where the caller can see it. That is what the
        // session keeps (state, decode activations, logits, scratch) and what
        // only the prefill holds (its activations for all `t` tokens, with
        // the GDN workspace, and the head's `[1, vocab]` row). The prefill's
        // activations are counted again by `begin_impl`, against what is
        // allocated by then.
        let (q_heads, head_dim) = (cfg.attn.q_heads(), cfg.attn.head_dim());
        let logits_bytes = f32_bytes(cfg.vocab as usize);
        let need = self
            .carry_bytes(t, max_new)
            .saturating_add(DecodeScratch::bytes(1, q_heads, positions as usize, head_dim)? as u64)
            .saturating_add(Acts::bytes(cfg, self.precision, 1, false)?)
            .saturating_add(logits_bytes.saturating_mul(2))
            .saturating_add(Acts::bytes(cfg, self.precision, t, false)?);
        self.check_working_set(
            WHAT,
            need,
            &format!("a session of {t} tokens and max_new {max_new} needs"),
            "shorten the prompt or max_new, or free device memory",
        )?;
        // Everything the session holds is allocated before the prefill runs,
        // so a failed allocation never throws a finished prefill away.
        let carry = self.alloc_carry(t, max_new)?;
        let a = Acts::new(rt, cfg, self.precision, 1, false)?;
        let logits_t = rt.alloc_tensor_f32(&[1, cfg.vocab as usize])?;
        let scratch = DecodeScratch::new(rt, 1, q_heads, positions as usize, head_dim)?;
        let mut s = self.begin_impl(WHAT, ids, false, Some(carry), 0)?;
        s.advance_to(self.layers.len())?;
        let logits = s.logits(LogitRows::Last)?;
        let carry = s
            .carry
            .take()
            .ok_or_else(|| format!("{WHAT}: the prefill lost its state"))?;
        let decode = Decode {
            model: self,
            carry,
            a,
            logits: logits_t,
            scratch,
            prefix_len: t,
            max_new,
            decoded: 0,
            broken: false,
            param_generation: s.param_generation,
        };
        Ok((decode, logits))
    }

    fn begin_impl(
        &self,
        what: &str,
        ids: &[u32],
        trace: bool,
        carry: Option<Vec<Carry>>,
        extra_bytes: u64,
    ) -> Result<Staged<'_>, String> {
        // Read before any weight is: a write that lands during the prefill
        // then stops it too.
        let param_generation = self.param_generation();
        let t = self.check_ids(what, ids)?;
        // Counted against what is allocated now, so a prefill that has
        // already allocated its session's state is not charged for it twice.
        let need = Acts::bytes(&self.cfg, self.precision, t, carry.is_none())?.saturating_add(extra_bytes);
        self.check_working_set(
            what,
            need,
            &format!("a forward of {t} tokens needs"),
            "shorten the sequence, or free device memory",
        )?;
        let mut a = Acts::new(&self.rt, &self.cfg, self.precision, t, carry.is_none())?;
        a.g_ws.set_scan_slice(self.gdn_scan);
        self.embed_ids(ids, &a.resid)?;
        Ok(Staged {
            model: self,
            a,
            next: 0,
            trace: trace.then(Vec::new),
            carry,
            param_generation,
            broken: false,
        })
    }

    /// Refuse, before any GPU work, an allocation of `need` bytes that would
    /// take the device's current allocation past its recommended working set.
    /// `what` names the entry point, `subject` what needs the bytes and `hint`
    /// what the caller can do.
    fn check_working_set(&self, what: &str, need: u64, subject: &str, hint: &str) -> Result<(), String> {
        let (have, limit) = (
            self.rt.current_allocated_bytes(),
            self.rt.memory_info().recommended_working_set,
        );
        if have.saturating_add(need) > limit {
            return Err(format!(
                "{what}: {subject} {need} B on top of the {have} B allocated, over the device's \
                 recommended working set of {limit} B; refused before any GPU work ({hint})"
            ));
        }
        Ok(())
    }

    /// Refuse to continue a prefill or session begun at `since` once the
    /// weights have been written (write_parameters, write_parameter_host, adamw_step): its state
    /// is of weights the model no longer has.
    fn check_generation(&self, what: &str, since: u64) -> Result<(), String> {
        if since != self.param_generation() {
            return Err(format!(
                "{what}: the model's parameters were written after this prefill began (write_parameters, \
                 write_parameter_host or adamw_step), so its state is of weights the model no longer has; prefill again"
            ));
        }
        Ok(())
    }

    /// What every run checks of its tokens and the runtime: at least one id,
    /// each below the vocabulary, and exact-f32 GEMMs for the F32 forward.
    /// Returns the token count.
    fn check_ids(&self, what: &str, ids: &[u32]) -> Result<u32, String> {
        if ids.is_empty() {
            return Err(format!("{what}: no tokens"));
        }
        if let Some(&bad) = ids.iter().find(|&&id| id >= self.cfg.vocab) {
            return Err(format!("{what}: token id {bad} >= vocab {}", self.cfg.vocab));
        }
        if self.precision == Precision::F32 && self.rt.f32_gemm_runs_relaxed() {
            return Err(relaxed_refusal(what));
        }
        u32::try_from(ids.len()).map_err(|_| format!("{what}: too many tokens"))
    }

    /// Refuse what needs the LM head on a model loaded without one.
    fn require_head(&self, what: &str) -> Result<(), String> {
        if self.precision == Precision::Bf16 && self.lm_head_bf16.is_none() {
            return Err(format!(
                "{what}: this model was built without an LM head (load_tower or random_tower); use begin / advance_to / \
                 final_norm_f32, or Staged::score_answers"
            ));
        }
        Ok(())
    }

    /// `out[r] = embed[ids[r]]`, the residual stream's first value (`out` f32
    /// `[ids.len(), hidden]`).
    fn embed_ids(&self, ids: &[u32], out: &Tensor) -> Result<(), String> {
        let id_buf = self.rt.alloc_buffer_from_u32(ids)?;
        qwen35::embed_rows(
            &self.rt,
            &id_buf,
            ids.len() as u32,
            LmHead {
                weight: &self.embed.buffer,
                dtype: self.embed.dtype,
                vocab: self.cfg.vocab,
            },
            self.cfg.hidden,
            &out.buffer,
        )
    }

    /// Fresh state for [`Self::prefill`] of `t` tokens and `max_new` after
    /// them, one per layer. Nothing is read before it is written: the
    /// prefill writes every state and prefix slot, and decode reads a suffix
    /// slot only after writing it.
    fn alloc_carry(&self, t: u32, max_new: u32) -> Result<Vec<Carry>, String> {
        let f32s = |n: usize| self.rt.alloc_buffer(n.max(1) * 4);
        self.cfg
            .layers
            .iter()
            .map(|&kind| {
                Ok(match (kind, self.carry_elems(kind, t, max_new)) {
                    (LayerKind::LinearAttention, [conv, state]) => Carry::Gdn(GdnCarry {
                        conv: [f32s(conv)?, f32s(conv)?],
                        state: f32s(state)?,
                    }),
                    (LayerKind::FullAttention, [prefix, suffix]) => Carry::Attn(AttnCarry {
                        prefix_k: f32s(prefix)?,
                        prefix_v: f32s(prefix)?,
                        suffix_k: f32s(suffix)?,
                        suffix_v: f32s(suffix)?,
                    }),
                })
            })
            .collect()
    }

    /// f32 elements of one layer's kept state buffers: a GDN layer's conv
    /// state (one side; there are two) and recurrent state, or an attention
    /// layer's prefix and suffix (one of K and V; there are both). What
    /// [`Self::alloc_carry`] allocates and [`Self::carry_bytes`] counts.
    fn carry_elems(&self, kind: LayerKind, t: u32, max_new: u32) -> [usize; 2] {
        let (g, l) = (self.cfg.gdn, self.cfg.attn);
        match kind {
            LayerKind::LinearAttention => [
                g.conv_dim() as usize * (self.cfg.conv_kernel as usize - 1),
                g.dims(1, 1).state_elems_per_row(),
            ],
            LayerKind::FullAttention => {
                let kv_row = (l.kv_heads() * l.head_dim()) as usize;
                [t as usize * kv_row, max_new as usize * kv_row]
            }
        }
    }

    /// Bytes [`Self::alloc_carry`] allocates, saturating.
    fn carry_bytes(&self, t: u32, max_new: u32) -> u64 {
        let buf = |n: usize| (n.max(1) as u64).saturating_mul(4);
        self.cfg
            .layers
            .iter()
            .map(|&kind| match (kind, self.carry_elems(kind, t, max_new)) {
                (LayerKind::LinearAttention, [conv, state]) => buf(conv).saturating_mul(2).saturating_add(buf(state)),
                (LayerKind::FullAttention, [prefix, suffix]) => {
                    buf(prefix).saturating_add(buf(suffix)).saturating_mul(2)
                }
            })
            .fold(0, u64::saturating_add)
    }

    /// One decoder layer on `a.resid`, in place: the layer order of every
    /// inference path (the prefill, the staged prefill and decode). `step`
    /// picks the mixer's kernels and where its state starts and goes.
    fn layer(&self, l: &Layer, a: &Acts, step: Step<'_>) -> Result<(), String> {
        self.take_layer_fault()?;
        LayerPart::ORDER
            .into_iter()
            .try_for_each(|part| self.layer_part(l, a, step, part))
    }

    /// One [`LayerPart`] of a layer: the only place the parts' bodies are
    /// written, and [`Self::layer`] runs them in [`LayerPart::ORDER`].
    fn layer_part(&self, l: &Layer, a: &Acts, step: Step<'_>, part: LayerPart) -> Result<(), String> {
        match part {
            LayerPart::InputNorm => self.norm(&a.resid, &l.input_norm, &a.x),
            LayerPart::Mixer => match &l.mixer {
                Mixer::Gdn(w) => self.gdn(w, a, step),
                Mixer::Attn(w) => self.attention(w, a, step),
            },
            LayerPart::PostNorm => self.norm(&a.resid, &l.post_norm, &a.x),
            LayerPart::Mlp => self.mlp(l, a),
        }
    }

    /// `out = (rms_norm(resid) * (1 + norm.w)) @ headᵀ` for every row of
    /// `resid` (f32 `[rows, hidden]`), through `x` (`[rows, hidden]` in the
    /// activation dtype): the final norm and the LM head (tied to the embedding, or its own). `out` is f32
    /// `[rows, vocab]`. Encoded, not waited for.
    fn head(&self, resid: &Tensor, x: &Tensor, out: &Tensor) -> Result<(), String> {
        self.norm(resid, &self.final_norm, x)?;
        match &self.lm_head_bf16 {
            Some(head) => gemm(x, head, out, BACKEND),
            None => gemm_nt_f32(x, self.head_table(), out, BACKEND),
        }
    }

    /// The `[vocab, hidden]` table the LM head reads: an untied model's own
    /// `lm_head`, else the tied embedding.
    pub(crate) fn head_table(&self) -> &Tensor {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// `out = rms_norm(x) * (1 + w)` in the forward's activation dtype.
    fn norm(&self, x: &Tensor, w: &GpuBuffer, out: &Tensor) -> Result<(), String> {
        let (rows, dim) = (x.shape[0] as u32, x.shape[1] as u32);
        qwen35::rms_norm(
            &self.rt,
            &x.buffer,
            w,
            &out.buffer,
            self.precision.dtype(),
            rows,
            dim,
            self.cfg.rms_norm_eps,
        )
    }

    /// `resid += y @ w_out`.
    fn project_residual(&self, y: &Tensor, w_out: &Tensor, a: &Acts) -> Result<(), String> {
        match self.precision {
            Precision::Bf16 => qwen35::project_residual(y, w_out, &a.resid, BACKEND),
            Precision::F32 => {
                gemm(y, w_out, &a.proj_out, BACKEND)?;
                let width = self.cfg.hidden;
                qwen35::residual_add(
                    &self.rt,
                    Cols::dense(&a.proj_out.buffer, width),
                    Cols::dense(&a.resid.buffer, width),
                    a.t,
                    width,
                )
            }
        }
    }

    fn gdn(&self, w: &GdnWeights, a: &Acts, step: Step<'_>) -> Result<(), String> {
        let (rt, g, t) = (&self.rt, self.cfg.gdn, a.t);
        gemm(&a.x, &w.w_in, &a.g_proj, BACKEND)?;
        let proj = &a.g_proj.buffer;
        let x = Cols::dense(proj, g.width());
        let kw = self.cfg.conv_kernel;
        let (dims, qkv, gates) = (g.dims(1, t), g.conv_qkv(&a.g_qkv), g.gates(proj));
        let params = GdnParams {
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
        };
        let out = Cols::dense(&a.g_o, g.value_dim());
        match step {
            // From zero state over the whole sequence: the chunked rule. A
            // prefill that keeps its state leaves the conv's last inputs in
            // side 0 and the recurrent state in place.
            Step::Prefill | Step::Keep(_) => {
                let keep = match step {
                    Step::Keep(c) => Some(c.gdn()?),
                    _ => None,
                };
                qwen35::conv1d_silu(
                    rt,
                    x,
                    &w.conv_w,
                    kw,
                    StateIn::Zero,
                    &a.g_qkv,
                    keep.map(|c| &c.conv[0]),
                    1,
                    t,
                    g.conv_dim(),
                )?;
                qwen35::gdn_chunk_forward(
                    rt,
                    &dims,
                    &qkv,
                    &gates,
                    &params,
                    StateIn::Zero,
                    &a.g_ws,
                    out,
                    keep.map(|c| &c.state),
                )?;
            }
            // One token on the carried state: the conv reads one side and
            // writes the other (its output state may not be its input), and
            // the recurrent rule updates the state in place.
            Step::Decode(c, io) => {
                let c = c.gdn()?;
                qwen35::conv1d_silu(
                    rt,
                    x,
                    &w.conv_w,
                    kw,
                    StateIn::PerBatch(&c.conv[io.side]),
                    &a.g_qkv,
                    Some(&c.conv[1 - io.side]),
                    1,
                    t,
                    g.conv_dim(),
                )?;
                qwen35::gdn_recurrent(
                    rt,
                    &dims,
                    &qkv,
                    &gates,
                    &params,
                    StateIn::PerBatch(&c.state),
                    out,
                    Some(&c.state),
                )?;
            }
        }
        qwen35::gated_rms_norm(
            rt,
            out,
            g.z(proj),
            &w.norm_w,
            OutCols {
                cols: Cols::dense(&a.g_y.buffer, g.value_dim()),
                dtype: self.precision.dtype(),
            },
            t,
            g.v_heads(),
            g.v_dim(),
            self.cfg.rms_norm_eps,
        )?;
        self.project_residual(&a.g_y, &w.w_out, a)
    }

    fn attention(&self, w: &AttnWeights, a: &Acts, step: Step<'_>) -> Result<(), String> {
        let (rt, l, t) = (&self.rt, self.cfg.attn, a.t);
        gemm(&a.x, &w.w_in, &a.a_proj, BACKEND)?;
        let pc = Cols::dense(&a.a_proj.buffer, l.width());
        let shape = AttnShape {
            batch: 1,
            seq: t,
            q_heads: l.q_heads(),
            kv_heads: l.kv_heads(),
            head_dim: l.head_dim(),
            rotary_dim: self.cfg.rotary_dim,
        };
        let dims = AttnDims {
            batch: 1,
            tq: t,
            heads: l.q_heads(),
            heads_kv: l.kv_heads(),
            window: 0,
            scale: 1.0 / (l.head_dim() as f32).sqrt(),
        };
        let (theta, eps) = (self.cfg.rope_theta, self.cfg.rms_norm_eps);
        match step {
            // Positions 0..t into a cache of the prefill's own K/V: the shared
            // buffers, or this layer's prefix when the state is kept.
            Step::Prefill | Step::Keep(_) => {
                let (k, v) = match step {
                    Step::Keep(c) => {
                        let c = c.attn()?;
                        (&c.prefix_k, &c.prefix_v)
                    }
                    _ => a.kv()?,
                };
                qwen35::attn_qk_norm_rope(
                    rt,
                    &shape,
                    pc,
                    &w.q_norm,
                    &w.k_norm,
                    &AttnTargets {
                        q_out: &a.a_q,
                        k_cache: k,
                        v_cache: v,
                    },
                    0,
                    theta,
                    eps,
                )?;
                qwen35::attn_prefill(rt, &a.a_q, k, v, &a.a_o, &a.tkv, &a.zero, &a.zero, dims, false)?;
            }
            // One token at position prefix_len + slot: cached at its suffix
            // slot, attending to the prefix and the suffix so far.
            Step::Decode(c, io) => {
                let c = c.attn()?;
                qwen35::attn_qk_norm_rope_suffix(
                    rt,
                    &shape,
                    pc,
                    &w.q_norm,
                    &w.k_norm,
                    &AttnTargets {
                        q_out: &a.a_q,
                        k_cache: &c.suffix_k,
                        v_cache: &c.suffix_v,
                    },
                    io.prefix_len,
                    io.slot,
                    theta,
                    eps,
                )?;
                qwen35::attn_prefix_decode(
                    rt,
                    &a.a_q,
                    SharedPrefix {
                        k: &c.prefix_k,
                        v: &c.prefix_v,
                        len: io.prefix_len,
                    },
                    &c.suffix_k,
                    &c.suffix_v,
                    &io.suffix_len,
                    &io.q_pos,
                    &a.a_o,
                    io.scratch,
                    dims,
                    false,
                )?;
            }
        }
        let width = l.q_heads() * l.head_dim();
        qwen35::attn_output_gate(
            rt,
            &a.a_o,
            pc,
            OutCols {
                cols: Cols::dense(&a.a_y.buffer, width),
                dtype: self.precision.dtype(),
            },
            t,
            l.q_heads(),
            l.head_dim(),
        )?;
        self.project_residual(&a.a_y, &w.w_out, a)
    }

    fn mlp(&self, layer: &Layer, a: &Acts) -> Result<(), String> {
        let inter = self.cfg.intermediate;
        gemm(&a.x, &layer.gate, &a.m_gate, BACKEND)?;
        gemm(&a.x, &layer.up, &a.m_up, BACKEND)?;
        qwen35::swiglu(
            &self.rt,
            Cols::dense(&a.m_gate.buffer, inter),
            Cols::dense(&a.m_up.buffer, inter),
            OutCols {
                cols: Cols::dense(&a.m_mid.buffer, inter),
                dtype: self.precision.dtype(),
            },
            a.t,
            inter,
        )?;
        self.project_residual(&a.m_mid, &layer.down, a)
    }
}

/// One of the four encoded steps of a decoder layer (see the module docs'
/// layer), in the order [`Self::ORDER`] runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerPart {
    /// `x = rms_norm(resid) * (1 + input_layernorm.w)`.
    InputNorm,
    /// The GDN or attention mixer: `resid += mixer(x)`.
    Mixer,
    /// `x = rms_norm(resid) * (1 + post_attention_layernorm.w)`.
    PostNorm,
    /// `resid += down(silu(gate(x)) * up(x))`.
    Mlp,
}

impl LayerPart {
    /// The order a layer runs its parts in.
    pub const ORDER: [LayerPart; 4] = [Self::InputNorm, Self::Mixer, Self::PostNorm, Self::Mlp];
}

/// How one layer's mixer runs, and where its state starts and goes.
#[derive(Clone, Copy)]
enum Step<'c> {
    /// Positions `0..t` from zero state, keeping nothing: every attention
    /// layer's K/V in [`Acts`]' one pair of buffers.
    Prefill,
    /// Positions `0..t` from zero state, leaving the layer's state in its
    /// carry ([`Qwen35Model::prefill`]).
    Keep(&'c Carry),
    /// One token after the carry's prefix and the tokens decoded before it.
    Decode(&'c Carry, &'c DecodeIo<'c>),
}

/// What one layer hands from the prefill to decode, and decode from token
/// to token.
enum Carry {
    Gdn(GdnCarry),
    Attn(AttnCarry),
}

struct GdnCarry {
    /// The conv's last `conv_kernel - 1` inputs, `[conv_dim, conv_kernel -
    /// 1]`, twice: a step reads one and writes the other.
    conv: [GpuBuffer; 2],
    /// The recurrent state, `[v_heads, 128, v_dim]`, updated in place.
    state: GpuBuffer,
}

struct AttnCarry {
    /// The prefill's K/V, `[prefill tokens, kv_heads, head_dim]`: the shared
    /// prefix every decode step reads.
    prefix_k: GpuBuffer,
    prefix_v: GpuBuffer,
    /// The decoded tokens' K/V, `[max_new, kv_heads, head_dim]`; slot `s` is
    /// position `prefix + s`.
    suffix_k: GpuBuffer,
    suffix_v: GpuBuffer,
}

impl Carry {
    fn gdn(&self) -> Result<&GdnCarry, String> {
        match self {
            Carry::Gdn(c) => Ok(c),
            Carry::Attn(_) => Err("Qwen35Model: a GDN layer was handed an attention layer's state".into()),
        }
    }

    fn attn(&self) -> Result<&AttnCarry, String> {
        match self {
            Carry::Attn(c) => Ok(c),
            Carry::Gdn(_) => Err("Qwen35Model: an attention layer was handed a GDN layer's state".into()),
        }
    }
}

/// What a decode step's mixers read besides their carry.
struct DecodeIo<'s> {
    /// Prefill tokens: the shared prefix's length.
    prefix_len: u32,
    /// Tokens decoded before this one: its suffix slot.
    slot: u32,
    /// The conv state side this step reads; it writes the other.
    side: usize,
    /// `slot + 1` and `prefix_len + slot`, one device u32 each.
    suffix_len: GpuBuffer,
    q_pos: GpuBuffer,
    scratch: &'s DecodeScratch,
}

/// A prefill stopped between layers, from [`Qwen35Model::begin`].
///
/// The residual stream after the last layer run stays on the device; run more
/// layers with [`Self::advance_to`] or read the final norm of it with
/// [`Self::final_norm_f32`]. Advancing in steps computes exactly what one
/// [`Qwen35Model::forward`] computes: the same dispatches in the same order
/// on the same buffers, only with reads in between.
pub struct Staged<'m> {
    model: &'m Qwen35Model,
    a: Acts,
    next: usize,
    trace: Option<Vec<Vec<f32>>>,
    /// [`Qwen35Model::prefill`]'s: where each layer leaves its state.
    carry: Option<Vec<Carry>>,
    /// The model's parameter generation when the prefill began.
    param_generation: u64,
    /// An advance failed part way, so the residual is past `next`: the
    /// session refuses to continue (the same latch as [`Decode`]'s).
    broken: bool,
}

impl Staged<'_> {
    /// Tokens in the prefill.
    pub fn tokens(&self) -> u32 {
        self.a.t
    }

    /// Decoder layers run so far: the residual stream is the output of layer
    /// `layers_done() - 1` (transformers' `hidden_states[layers_done()]`).
    pub fn layers_done(&self) -> usize {
        self.next
    }

    /// What every entry point checks before encoding: the session is not
    /// poisoned, the weights are the ones it began with, and an F32 model
    /// is not now running relaxed GEMMs (the switch can be thrown after
    /// `begin`).
    fn check_usable(&self, what: &str) -> Result<(), String> {
        let m = self.model;
        if self.broken {
            return Err(format!(
                "{what}: an earlier advance failed part way, so the residual stream is past \
                 layers_done(); begin again"
            ));
        }
        m.check_generation(what, self.param_generation)?;
        if m.precision == Precision::F32 && use_relaxed_f32(&m.rt, BACKEND) {
            return Err(relaxed_refusal(what));
        }
        Ok(())
    }

    /// Run layers `layers_done()..layer`. `layer` may equal `layers_done()`
    /// (nothing to run) but not go back, or past the model's layer count.
    pub fn advance_to(&mut self, layer: usize) -> Result<(), String> {
        let m = self.model;
        self.check_usable("Staged::advance_to")?;
        if layer < self.next || layer > m.layers.len() {
            return Err(format!(
                "Staged::advance_to({layer}): already at layer {}, the model has {}",
                self.next,
                m.layers.len()
            ));
        }
        if layer == self.next {
            return Ok(());
        }
        // From the first layer on, a failure leaves the residual advanced.
        self.broken = true;
        let a = &self.a;
        for (i, l) in m.layers.iter().enumerate().take(layer).skip(self.next) {
            let step = match &self.carry {
                Some(c) => Step::Keep(&c[i]),
                None => Step::Prefill,
            };
            m.layer(l, a, step)?;
            if let Some(trace) = self.trace.as_mut() {
                m.rt.synchronize()?;
                m.take_read_fault()?;
                trace.push(a.resid.buffer.try_contents_f32()?[..(a.t * m.cfg.hidden) as usize].to_vec());
            }
        }
        self.next = layer;
        self.broken = false;
        Ok(())
    }

    /// Encode one `part` of layer `layer` on the session's activations, as a
    /// prefill runs it (nothing kept), without advancing [`Self::layers_done`]
    /// and without waiting: for timing a layer's parts one at a time (several
    /// per command buffer under async encode). The residual stream is
    /// updated in place by the mixer and the MLP, so repeating a part moves
    /// it; the numbers a session returns after this are not a forward's.
    /// Refused on a session that keeps state (only [`Qwen35Model::prefill`]'s
    /// internal one does), on a broken session, and for a layer the model does
    /// not have.
    pub fn encode_part(&self, layer: usize, part: LayerPart) -> Result<(), String> {
        const WHAT: &str = "Staged::encode_part";
        let m = self.model;
        self.check_usable(WHAT)?;
        if self.carry.is_some() {
            return Err(format!("{WHAT}: a session that keeps state runs whole layers only"));
        }
        let l = m
            .layers
            .get(layer)
            .ok_or_else(|| format!("{WHAT}: layer {layer} of a model with {}", m.layers.len()))?;
        m.layer_part(l, &self.a, Step::Prefill, part)
    }

    /// `out = rms_norm(resid) * (1 + norm.w)` in f32, `[tokens, hidden]`: the
    /// model's final norm applied to the residual stream at the layer reached
    /// (transformers' last `hidden_states` entry when every layer has run).
    /// Encoded, not waited for: read `out` after a synchronize.
    pub fn final_norm_f32(&self, out: &Tensor) -> Result<(), String> {
        let (m, t) = (self.model, self.a.t);
        self.check_usable("Staged::final_norm_f32")?;
        if out.dtype != DType::F32 || out.shape() != [t as usize, m.cfg.hidden as usize] {
            return Err(format!(
                "Staged::final_norm_f32: out must be f32 [{t}, {}], got {:?} {:?}",
                m.cfg.hidden,
                out.dtype,
                out.shape()
            ));
        }
        qwen35::rms_norm(
            &m.rt,
            &self.a.resid.buffer,
            &m.final_norm,
            &out.buffer,
            DType::F32,
            t,
            m.cfg.hidden,
            m.cfg.rms_norm_eps,
        )
    }

    /// The final norm and LM head on the residual stream at the layer
    /// reached, for the positions `rows` chooses: `[rows, vocab]` f32 (after
    /// every layer, the model's logits). Only the chosen rows are normed,
    /// multiplied and read back. Waits for the GPU. Refused on a bf16
    /// [`Qwen35Model::load_tower`] model, which has no head (an f32 tower
    /// has one: the embedding).
    pub fn logits(&self, rows: LogitRows<'_>) -> Result<Vec<f32>, String> {
        const WHAT: &str = "Staged::logits";
        let (m, a, t) = (self.model, &self.a, self.a.t);
        m.require_head(WHAT)?;
        self.check_usable(WHAT)?;
        let last = [t - 1];
        // None: every row, in order, through the stream's own buffers.
        let pick: Option<&[u32]> = match rows {
            LogitRows::All => None,
            LogitRows::Last if t == 1 => None,
            LogitRows::Last => Some(&last),
            LogitRows::Rows(r) => Some(r),
        };
        check_rows(WHAT, rows, t as usize)?;
        let (rt, h, vocab) = (&m.rt, m.cfg.hidden as usize, m.cfg.vocab as usize);
        let n = pick.map_or(t as usize, <[u32]>::len);
        if n == 0 {
            return Ok(Vec::new());
        }
        let out = rt.alloc_tensor_f32(&[n, vocab])?;
        match pick {
            None => m.head(&a.resid, &a.x, &out)?,
            Some(rows) => {
                let picked = rt.alloc_tensor_f32(&[n, h])?;
                gather_rows_f32(rt, WHAT, &a.resid, rows, &picked)?;
                m.head(&picked, &activation(rt, m.precision, n, h)?, &out)?;
            }
        }
        rt.synchronize()?;
        m.take_read_fault()?;
        Ok(out.buffer.try_contents_f32()?[..n * vocab].to_vec())
    }

    /// The `answers` tokens' logits at positions `rows`, and their
    /// log-softmax over the answers alone, from the residual stream at the
    /// layer reached: [`qwen35::score_answer_rows`], which applies the final
    /// norm and dots each row with only the answers' rows of the LM head,
    /// in f32 (a bf16 embedding is read as stored; the activations are not
    /// rounded to bf16 as [`Self::logits`]' bf16 GEMM rounds them). No
    /// logits row is formed, so it runs on a [`Qwen35Model::load_tower`]
    /// model too. `answers` holds 1..=[`qwen35::MAX_ANSWERS`] token ids; a
    /// token listed twice is scored twice and counts twice in the
    /// log-softmax over the answers (legal, and not deduplicated).
    /// Waits for the GPU.
    pub fn score_answers(&self, rows: &[u32], answers: &[u32]) -> Result<AnswerScores, String> {
        const WHAT: &str = "Staged::score_answers";
        let (m, t) = (self.model, self.a.t);
        let (rt, cfg) = (&m.rt, &m.cfg);
        self.check_usable(WHAT)?;
        if let Some(&bad) = rows.iter().find(|&&r| r >= t) {
            return Err(format!("{WHAT}: position {bad} >= {t} tokens"));
        }
        if answers.is_empty() || answers.len() > qwen35::MAX_ANSWERS as usize {
            return Err(format!(
                "{WHAT}: {} answers; 1..={} are scored per call",
                answers.len(),
                qwen35::MAX_ANSWERS
            ));
        }
        if let Some(&bad) = answers.iter().find(|&&id| id >= cfg.vocab) {
            return Err(format!("{WHAT}: answer {bad} >= vocab {}", cfg.vocab));
        }
        let n_rows = u32::try_from(rows.len()).map_err(|_| format!("{WHAT}: too many positions"))?;
        let n_answers = u32::try_from(answers.len()).map_err(|_| format!("{WHAT}: too many answers"))?;
        let n = rows.len() * answers.len();
        let (slots, ans) = (rt.alloc_buffer_from_u32(rows)?, rt.alloc_buffer_from_u32(answers)?);
        let (logits, logprobs) = (rt.alloc_buffer(n.max(1) * 4)?, rt.alloc_buffer(n.max(1) * 4)?);
        qwen35::score_answer_rows(
            rt,
            &self.a.resid.buffer,
            t,
            cfg.hidden,
            &slots,
            n_rows,
            &m.final_norm,
            1.0,
            cfg.rms_norm_eps,
            LmHead {
                weight: &m.head_table().buffer,
                dtype: m.head_table().dtype,
                vocab: cfg.vocab,
            },
            &ans,
            n_answers,
            &logits,
            &logprobs,
        )?;
        rt.synchronize()?;
        m.take_read_fault()?;
        // One host mapping at a time: two live at once read as a busy runtime.
        let logits = logits.try_contents_f32()?[..n].to_vec();
        let logprobs = logprobs.try_contents_f32()?[..n].to_vec();
        Ok(AnswerScores { logits, logprobs })
    }
}

/// A sequence being decoded a token at a time, from
/// [`Qwen35Model::prefill`]: every layer's state, and room for `max_new`
/// tokens after the prefill. Each step runs the inference layer order on the
/// decode kernels (see the module docs).
pub struct Decode<'m> {
    model: &'m Qwen35Model,
    carry: Vec<Carry>,
    /// One token's intermediates, reused by every step.
    a: Acts,
    /// `[1, vocab]` f32.
    logits: Tensor,
    scratch: DecodeScratch,
    prefix_len: u32,
    max_new: u32,
    decoded: u32,
    /// A step failed part way, so some layers' state may have advanced and
    /// others not: the session refuses to continue.
    broken: bool,
    /// The model's parameter generation when the prefill began.
    param_generation: u64,
}

impl Decode<'_> {
    /// The position the next token takes: the prefill's tokens plus those
    /// decoded since.
    pub fn position(&self) -> u32 {
        self.prefix_len + self.decoded
    }

    /// Tokens this session can still decode.
    pub fn remaining(&self) -> u32 {
        self.max_new - self.decoded
    }

    /// Run `id` at [`Self::position`] through every layer, continuing each
    /// layer's state, and return its logits (`[vocab]` f32: the distribution
    /// of the token after it). Waits for the GPU. Refused, before any work,
    /// once `max_new` tokens have been decoded, once the model's weights
    /// have been written since the prefill, and for good after a step that
    /// failed part way.
    pub fn step(&mut self, id: u32) -> Result<Vec<f32>, String> {
        const WHAT: &str = "Decode::step";
        let m = self.model;
        if self.broken {
            return Err(format!(
                "{WHAT}: an earlier step failed part way, so the layers' state is inconsistent; prefill again"
            ));
        }
        if self.decoded >= self.max_new {
            return Err(format!(
                "{WHAT}: all {} tokens the session was prefilled for are decoded; prefill again with a \
                 larger max_new",
                self.max_new
            ));
        }
        m.check_generation(WHAT, self.param_generation)?;
        m.check_ids(WHAT, &[id])?;
        let rt = &m.rt;
        let slot = self.decoded;
        let io = DecodeIo {
            prefix_len: self.prefix_len,
            slot,
            side: (slot % 2) as usize,
            suffix_len: rt.alloc_buffer_from_u32(&[slot + 1])?,
            q_pos: rt.alloc_buffer_from_u32(&[self.prefix_len + slot])?,
            scratch: &self.scratch,
        };
        m.embed_ids(&[id], &self.a.resid)?;
        // From the first layer on, a failure leaves the state part advanced.
        self.broken = true;
        for (l, c) in m.layers.iter().zip(&self.carry) {
            m.layer(l, &self.a, Step::Decode(c, &io))?;
        }
        m.head(&self.a.resid, &self.a.x, &self.logits)?;
        rt.synchronize()?;
        // Read before advancing: a failed read leaves `position()` where it was,
        // and the session poisoned, since the layers' state has moved past it.
        m.take_read_fault()?;
        let row = self.logits.buffer.try_contents_f32()?[..m.cfg.vocab as usize].to_vec();
        self.decoded += 1;
        self.broken = false;
        Ok(row)
    }
}

/// A test hook's countdown: `usize::MAX` is off, `0` fires once (and turns
/// the hook off), `n` lets `n` more calls pass.
fn count_down_fault(hook: &AtomicUsize, message: &str) -> Result<(), String> {
    match hook.load(Ordering::Relaxed) {
        usize::MAX => Ok(()),
        0 => {
            hook.store(usize::MAX, Ordering::Relaxed);
            Err(message.into())
        }
        n => {
            hook.store(n - 1, Ordering::Relaxed);
            Ok(())
        }
    }
}

fn relaxed_refusal(what: &str) -> String {
    format!("{what}: the F32 forward needs exact-f32 GEMMs; switch the runtime's relaxed precision off")
}

/// Refuse a [`LogitRows::Rows`] position past a sequence of `t` tokens.
fn check_rows(what: &str, rows: LogitRows<'_>, t: usize) -> Result<(), String> {
    if let LogitRows::Rows(r) = rows {
        if let Some(&bad) = r.iter().find(|&&p| p as usize >= t) {
            return Err(format!("{what}: position {bad} >= {t} tokens"));
        }
    }
    Ok(())
}

/// A `[rows, cols]` activation read back as f32 (bf16 widened exactly).
fn read_rows(x: &Tensor, precision: Precision) -> Result<Vec<f32>, String> {
    let n = x.shape.iter().product::<usize>();
    Ok(match precision {
        Precision::F32 => x.buffer.try_contents_f32()?[..n].to_vec(),
        Precision::Bf16 => x.buffer.try_contents_u16()?[..n]
            .iter()
            .map(|&b| crate::tensor::bf16_bits_to_f32(b))
            .collect(),
    })
}

/// A `[rows, cols]` activation in the forward's dtype: what the norms, gated
/// norm, output gate and SwiGLU write and the GEMMs read. Unzeroed: each of
/// those writes it in full before anything reads it.
fn activation(rt: &Arc<GpuRuntime>, p: Precision, rows: usize, cols: usize) -> Result<Tensor, String> {
    rt.alloc_tensor_unzeroed(&[rows, cols], p.dtype())
}

/// Every intermediate of one forward at `t` tokens, shared by all layers (the
/// runtime orders the reuse between dispatches).
struct Acts {
    t: u32,
    resid: Tensor,
    /// Norm output, the GEMMs' left operand.
    x: Tensor,
    /// `y @ w_out` before the add, F32 forward only.
    proj_out: Tensor,
    g_proj: Tensor,
    g_qkv: GpuBuffer,
    g_o: GpuBuffer,
    g_y: Tensor,
    g_ws: GdnWorkspace,
    a_proj: Tensor,
    a_q: GpuBuffer,
    /// The attention layers' K/V, `[t, kv_heads, head_dim]` each, which every
    /// attention layer overwrites; none when each layer keeps its own
    /// ([`Carry`]).
    kv: Option<(GpuBuffer, GpuBuffer)>,
    a_o: GpuBuffer,
    a_y: Tensor,
    tkv: GpuBuffer,
    zero: GpuBuffer,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
}

/// The allocated size of an `n`-element f32 buffer, as [`Acts`] and the
/// session's other buffers make it (never empty).
fn f32_bytes(n: usize) -> u64 {
    GpuRuntime::allocated_bytes_for(n.max(1).saturating_mul(4), BufferKind::Cold)
}

/// What [`Acts`] allocates at `t` tokens: the one place its sizes are
/// decided, which [`Acts::new`] allocates from and [`Acts::bytes`] sums.
struct ActsPlan {
    t: usize,
    p: Precision,
    hidden: usize,
    intermediate: usize,
    /// Columns of the GDN projection, its value width and the attention
    /// projection.
    g_width: usize,
    g_value: usize,
    a_width: usize,
    /// Elements of the GDN conv input and of the attention's `[t, q_heads *
    /// head_dim]` buffers.
    g_qkv: usize,
    qd: usize,
    /// Elements of one of the shared K and V; none when each layer keeps its own.
    kv: Option<usize>,
    gdn: GdnDims,
}

impl ActsPlan {
    fn new(cfg: &Qwen35Config, p: Precision, t: u32, shared_kv: bool) -> Self {
        let tu = t as usize;
        let (g, l) = (cfg.gdn, cfg.attn);
        let qd = (l.q_heads() * l.head_dim()) as usize;
        Self {
            t: tu,
            p,
            hidden: cfg.hidden as usize,
            intermediate: cfg.intermediate as usize,
            g_width: g.width() as usize,
            g_value: g.value_dim() as usize,
            a_width: l.width() as usize,
            g_qkv: tu.saturating_mul(g.conv_dim() as usize),
            qd,
            kv: shared_kv.then(|| tu.saturating_mul((l.kv_heads() * l.head_dim()) as usize)),
            gdn: g.dims(1, t),
        }
    }

    /// Bytes of a `[t, cols]` tensor in f32 and in the forward's dtype.
    fn f32_rows(&self, cols: usize) -> u64 {
        f32_bytes(self.t.saturating_mul(cols))
    }

    fn act_rows(&self, cols: usize) -> u64 {
        let nbytes = self.t.saturating_mul(cols).saturating_mul(self.p.dtype().size_of());
        GpuRuntime::allocated_bytes_for(nbytes, BufferKind::Cold)
    }
}

impl Acts {
    fn new(rt: &Arc<GpuRuntime>, cfg: &Qwen35Config, p: Precision, t: u32, shared_kv: bool) -> Result<Self, String> {
        let plan = ActsPlan::new(cfg, p, t, shared_kv);
        let tu = plan.t;
        // Every activation is written in full before anything reads it — the
        // embedding gather fills `resid`, the norms fill `x`, the GEMMs fill
        // their outputs, the gate and activation kernels fill `g_y`, `a_y`
        // and `m_mid` — so none is zeroed on the host. `tests/qwen35_train.rs`
        // holds this with `set_poison_unzeroed`.
        let f32t = |cols: usize| rt.alloc_tensor_unzeroed(&[tu, cols], DType::F32);
        let act = |cols: usize| activation(rt, p, tu, cols);
        let f32s = |n: usize| rt.alloc_buffer(n.max(1) * 4);
        // Fresh buffers, written without a GPU wait.
        let u32_buf = |v: u32| rt.alloc_buffer_from_u32(&[v]);
        Ok(Self {
            t,
            resid: f32t(plan.hidden)?,
            x: act(plan.hidden)?,
            proj_out: f32t(plan.hidden)?,
            g_proj: f32t(plan.g_width)?,
            g_qkv: f32s(plan.g_qkv)?,
            g_o: f32s(tu * plan.g_value)?,
            g_y: act(plan.g_value)?,
            g_ws: GdnWorkspace::new(rt, &plan.gdn)?,
            a_proj: f32t(plan.a_width)?,
            a_q: f32s(tu * plan.qd)?,
            kv: match plan.kv {
                Some(n) => Some((f32s(n)?, f32s(n)?)),
                None => None,
            },
            a_o: f32s(tu * plan.qd)?,
            a_y: act(plan.qd)?,
            tkv: u32_buf(t)?,
            zero: u32_buf(0)?,
            m_gate: f32t(plan.intermediate)?,
            m_up: f32t(plan.intermediate)?,
            m_mid: act(plan.intermediate)?,
        })
    }

    /// Bytes [`Self::new`] allocates at `t` tokens, as the device counts them
    /// (each buffer at its allocated size; the GDN workspace at
    /// [`GdnWorkspace::bytes_for`]'s). What a caller checks against the
    /// device's memory before allocating.
    fn bytes(cfg: &Qwen35Config, p: Precision, t: u32, shared_kv: bool) -> Result<u64, String> {
        let plan = ActsPlan::new(cfg, p, t, shared_kv);
        let ws = GdnWorkspace::bytes_for(&plan.gdn)? as u64;
        let scalar = f32_bytes(1);
        let kv = plan.kv.map_or(0, |n| f32_bytes(n).saturating_mul(2));
        Ok([
            plan.f32_rows(plan.hidden),
            plan.act_rows(plan.hidden),
            plan.f32_rows(plan.hidden),
            plan.f32_rows(plan.g_width),
            f32_bytes(plan.g_qkv),
            plan.f32_rows(plan.g_value),
            plan.act_rows(plan.g_value),
            ws,
            plan.f32_rows(plan.a_width),
            plan.f32_rows(plan.qd),
            kv,
            plan.f32_rows(plan.qd),
            plan.act_rows(plan.qd),
            scalar,
            scalar,
            plan.f32_rows(plan.intermediate),
            plan.f32_rows(plan.intermediate),
            plan.act_rows(plan.intermediate),
        ]
        .into_iter()
        .fold(0, u64::saturating_add))
    }

    /// The shared K/V of a prefill that keeps nothing.
    fn kv(&self) -> Result<(&GpuBuffer, &GpuBuffer), String> {
        self.kv
            .as_ref()
            .map(|(k, v)| (k, v))
            .ok_or_else(|| "Qwen35Model: a prefill that keeps its state has no shared K/V".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Acts::bytes` is what `Acts::new` allocates: every buffer's allocated
    /// size summed, the GDN workspace at `GdnWorkspace::bytes_for`'s.
    #[test]
    fn acts_bytes_is_what_acts_new_allocates() {
        let cfg = Qwen35Config::from_config_file(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train/config.json"),
        )
        .unwrap();
        let rt = GpuRuntime::new().expect("runtime");
        let alloc = |b: &GpuBuffer| GpuRuntime::allocated_bytes_for(b.nbytes(), BufferKind::Cold);
        for p in [Precision::F32, Precision::Bf16] {
            for t in [1u32, 5, 64, 65, 200] {
                for shared in [false, true] {
                    let a = Acts::new(&rt, &cfg, p, t, shared).unwrap();
                    let mut got = [&a.resid, &a.x, &a.proj_out, &a.g_proj, &a.g_y, &a.a_proj, &a.a_y]
                        .iter()
                        .chain([&&a.m_gate, &&a.m_up, &&a.m_mid])
                        .map(|x| alloc(&x.buffer))
                        .sum::<u64>();
                    got += [&a.g_qkv, &a.g_o, &a.a_q, &a.a_o, &a.tkv, &a.zero]
                        .into_iter()
                        .map(alloc)
                        .sum::<u64>();
                    got += a.kv.as_ref().map_or(0, |(k, v)| alloc(k) + alloc(v));
                    got += GdnWorkspace::bytes_for(&cfg.gdn.dims(1, t)).unwrap() as u64;
                    let want = Acts::bytes(&cfg, p, t, shared).unwrap();
                    assert_eq!(want, got, "{p:?} t={t} shared_kv={shared}");
                }
            }
        }
    }
}
