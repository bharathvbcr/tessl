//! TimesFM 3 (Google Research's time-series foundation model, 2026) on the
//! GPU: the residual block, the stacked mixing transformer and the output
//! head, composed from tessl kernels and loaded straight from
//! `model.safetensors`, with [`crate::timesfm3_host`] doing everything around
//! them. No Python anywhere on this path.
//!
//! The port follows upstream's PyTorch implementation (google-research/timesfm
//! at 918df02, `src/timesfm3/torch/`, Apache-2.0). The checkpoint
//! (`google/timesfm-3.0-pytorch`) is under the TimesFM Non-Commercial License
//! v1.0: this crate reads it, never ships it.
//!
//! # Forward (per patch row, `rows = b * v * n`)
//!
//! ```text
//! x  = relu(in Whᵀ) Woᵀ + in Wrᵀ                       in: [rows, 2 (p + o)]
//! per layer:
//!   a  = SeqAttn(rms(x))          causal over the n patches of one (b, v),
//!                                 leading fully-masked patches masked
//!   x += rms_post(a Woᵀ)
//!   a  = VarAttn(rms(x))          over the v variates of one (b, n), masked
//!                                 variates' keys masked; v == 1: a = x Wvᵀ
//!   x += rms_post(a Woᵀ)
//!   x += rms_post(relu(rms(x) W0ᵀ) W1ᵀ)
//! logits = x Wheadᵀ + b_head                          [rows, o * q]
//! ```
//!
//! Attention: Q, K, V projections (no bias), RoPE on the sequence axis only,
//! then RMSNorm per head with a gain, then PerDimScale on the query
//! (`x · 1.442695 / √d · softplus(s)`), and SDPA at scale √d (upstream's
//! memory-efficient-attention convention). The query's gain, PerDimScale and
//! the √d fold into one per-dim query gain at load, so the kernel runs at
//! scale 1. Every RMSNorm uses torch's default epsilon for f32,
//! `f32::EPSILON`.
//!
//! With one variate, upstream skips the variate attention's Q, K, softmax and
//! mask (attention over one key is its value) and so does this port.
//!
//! # Precision
//!
//! Weights and activations are f32 and every GEMM is exact f32 (the runtime's
//! relaxed precision must be off), torch's fp32 forward up to operation order.
//! `tests/timesfm3_tiny.rs` holds it to upstream's own forward.

use std::sync::Arc;

use crate::config_json;
use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
use crate::gemm::{gemm, gemm_epilogue, Activation, Epilogue, GemmBackend};
use crate::loader::Loader;
use crate::nn::{rms_norm_f32, rms_norm_residual_add_f32};
use crate::qwen35;
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, GpuBuffer, Tensor};
use crate::timesfm3_forecast::{self as forecast, PredictOptions, Series, SeriesForecast};
use crate::timesfm3_host::{self as host, DecodeInput, HostConfig};

const BACKEND: GemmBackend = GemmBackend::TensorOps;
/// The head dimension the attention kernels are compiled for.
pub const HEAD_DIM: u32 = 80;
/// Variates per series the variate attention's key bitmask holds (and
/// upstream's `max_variates`).
pub const MAX_VARIATES: usize = 32;
/// Patch rows (`b * v * n`) per forward. At d = 1280 the widest activation is
/// `[rows, 1280]` f32, 40 MiB at this size; [`Timesfm3Model::decode`] refuses
/// more rather than allocating without bound.
pub const MAX_ROWS: usize = 8192;
/// torch's `nn.RMSNorm(eps=None)` for f32.
const RMS_EPS: f32 = f32::EPSILON;
/// `_RECIPROCAL_OF_SOFTPLUS_0` in upstream's `normalization.py`, kept as
/// upstream writes it: a truncated `1 / ln 2`, not `LOG2_E`.
#[allow(clippy::approx_constant)]
const RECIPROCAL_OF_SOFTPLUS_0: f64 = 1.442695041;
/// RoPE's `max_timescale` (min 1).
const ROPE_MAX_TIMESCALE: f64 = 10000.0;

/// What `config.json` says, restricted to what this port computes. Every other
/// value of a recognised switch is refused by name.
#[derive(Clone, Debug, PartialEq)]
pub struct Timesfm3Config {
    pub host: HostConfig,
    pub model_dims: u32,
    pub ff_hidden: u32,
    pub heads: u32,
    pub layers: u32,
    pub resblock_hidden: u32,
    pub use_variate_attention: bool,
    /// `use_memory_efficient_attention`: SDPA at scale √d (true) or 1.
    pub mea_scale: bool,
    /// The quantile levels the head predicts, in output order (`quantiles`).
    pub quantile_levels: Vec<f64>,
}

impl Timesfm3Config {
    pub fn from_config_json(text: &str) -> Result<Self, String> {
        let c = config_json::parse_object(text)?;
        let uint = |o: &ojas_io::json::JsonValue, k: &str| config_json::uint(o, k);
        let float = |o: &ojas_io::json::JsonValue, k: &str| -> Result<f64, String> {
            config_json::float(config_json::field(o, k)?, k)
        };
        let want_str =
            |o: &ojas_io::json::JsonValue, k: &str, want: &str, default: Option<&str>| -> Result<(), String> {
                match (config_json::string(o, k)?, default) {
                    (Some(s), _) if s == want => Ok(()),
                    (None, Some(d)) if d == want => Ok(()),
                    (got, _) => Err(format!("config.json: {k} must be {want:?}, got {got:?}")),
                }
            };
        let want_flag = |o: &ojas_io::json::JsonValue, k: &str, want: bool, default: bool| -> Result<(), String> {
            let got = config_json::flag(o, k)?.unwrap_or(default);
            if got != want {
                return Err(format!("config.json: {k} = {got} is not supported (needs {want})"));
            }
            Ok(())
        };
        let flag = |o: &ojas_io::json::JsonValue, k: &str, default: bool| -> Result<bool, String> {
            Ok(config_json::flag(o, k)?.unwrap_or(default))
        };
        let quantile_levels = match config_json::field(&c, "quantiles")? {
            ojas_io::json::JsonValue::Array(a) => a
                .iter()
                .map(|x| config_json::float(x, "quantiles"))
                .collect::<Result<Vec<f64>, String>>()?,
            v => return Err(format!("config.json: quantiles must be an array, got {v:?}")),
        };
        let quantiles = quantile_levels.len();
        want_str(&c, "input_transform", "identity", Some("identity"))?;
        want_flag(&c, "use_frozen_running_stats", false, false)?;
        let rb = config_json::field(&c, "residual_block_config")?;
        want_str(rb, "activation", "relu", None)?;
        want_str(rb, "prenorm", "none", Some("none"))?;
        want_flag(rb, "use_bias", false, false)?;
        want_flag(rb, "identity_skip", false, false)?;
        let tc = config_json::field(&c, "transformer_config")?;
        let t = config_json::field(tc, "transformer")?;
        for k in ["attention_norm", "feedforward_norm", "qk_norm"] {
            want_str(t, k, "rms", None)?;
        }
        want_str(t, "v_norm", "none", Some("none"))?;
        want_str(t, "ff_activation", "relu", None)?;
        want_flag(t, "use_bias", false, false)?;
        want_flag(t, "use_rope_seq", true, true)?;
        want_flag(t, "use_rope_var", false, false)?;
        want_flag(t, "causal_attention", true, true)?;
        want_flag(t, "debug_no_masking", false, false)?;
        want_flag(t, "paired_token_skip_second", false, false)?;
        let cfg = Self {
            host: HostConfig {
                input_patch_len: uint(&c, "input_patch_len")? as usize,
                output_patch_len: uint(&c, "output_patch_len")? as usize,
                num_quantiles: quantiles,
                value_clip: float(&c, "value_clip")? as f32,
                use_stitching: flag(&c, "use_stitching", true)?,
                use_linear_detrending: flag(&c, "use_linear_detrending", true)?,
                linear_detrending_threshold: float(&c, "linear_detrending_threshold")? as f32,
                use_iterative_cpm_revin: flag(&c, "use_iterative_cpm_revin", true)?,
            },
            model_dims: uint(t, "model_dims")?,
            ff_hidden: uint(t, "hidden_dims")?,
            heads: uint(t, "num_heads")?,
            layers: uint(tc, "num_layers")?,
            resblock_hidden: uint(rb, "hidden_dims")?,
            use_variate_attention: flag(&c, "use_variate_attention", true)?,
            mea_scale: flag(t, "use_memory_efficient_attention", true)?,
            quantile_levels,
        };
        if uint(rb, "output_dims")? != cfg.model_dims {
            return Err("config.json: residual_block_config.output_dims must equal model_dims".into());
        }
        if [
            cfg.model_dims,
            cfg.ff_hidden,
            cfg.heads,
            cfg.layers,
            cfg.resblock_hidden,
        ]
        .contains(&0)
        {
            return Err("config.json: a width, head or layer count is zero".into());
        }
        if !cfg.model_dims.is_multiple_of(cfg.heads) || cfg.model_dims / cfg.heads != HEAD_DIM {
            return Err(format!(
                "config.json: head dim {} / {} has no kernel (needs {HEAD_DIM})",
                cfg.model_dims, cfg.heads
            ));
        }
        if !(cfg.host.value_clip.is_finite() && cfg.host.value_clip > 0.0) {
            return Err(format!(
                "config.json: value_clip {} must be positive and finite in f32",
                cfg.host.value_clip
            ));
        }
        Ok(cfg)
    }

    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let text = config_json::read_file(path)?;
        Self::from_config_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// One attention block's weights in the layouts the kernels read.
struct Attn {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    /// Query gain with PerDimScale and the SDPA scale folded in.
    gq: GpuBuffer,
    gk: GpuBuffer,
}

struct Layer {
    pre_seq: GpuBuffer,
    post_seq: GpuBuffer,
    seq: Attn,
    var: Option<(GpuBuffer, GpuBuffer, Attn)>,
    pre_ff: GpuBuffer,
    post_ff: GpuBuffer,
    ff0: Tensor,
    ff1: Tensor,
}

/// A loaded TimesFM 3: weights on the device, `nn.Linear` weights transposed
/// to `[in, out]`.
pub struct Timesfm3Model {
    cfg: Timesfm3Config,
    rt: Arc<GpuRuntime>,
    res_hidden: Tensor,
    res_out: Tensor,
    res_skip: Tensor,
    layers: Vec<Layer>,
    head: Tensor,
    head_bias: Tensor,
}

/// The device forward's output (see [`Timesfm3Model::forward`]).
pub struct ForwardOutput {
    /// `[rows, o * q]`: the output head, before RevIN is undone.
    pub raw_logits: Vec<f32>,
    /// With `trace`: `transformer_input` and then the residual stream after
    /// each layer, `[rows, d]` each. Empty otherwise.
    pub trace: Vec<Vec<f32>>,
}

/// A forecast: `[b, v, horizon, q]` for every variate (targets first, then
/// covariates, as upstream's `decode()` returns them).
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    pub batch: usize,
    pub variates: usize,
    pub horizon: usize,
    pub quantiles: usize,
    pub values: Vec<f32>,
}

struct Weights<'a> {
    ld: Loader<'a>,
    rt: &'a Arc<GpuRuntime>,
}

impl Weights<'_> {
    fn vec(&self, name: &str, n: u32) -> Result<Vec<f32>, String> {
        self.ld.f32_finite(name, &[n as usize])
    }

    fn buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        self.ld.buf(data)
    }

    /// `nn.Linear(in, out)` without bias: `[out, in]` stored as `[in, out]`.
    fn linear(&self, name: &str, in_f: u32, out_f: u32) -> Result<Tensor, String> {
        let w = self.ld.f32_finite(name, &[out_f as usize, in_f as usize])?;
        let packed = qwen35::pack_linear_weights_f32(&[&w], &[out_f as usize], in_f as usize)?;
        let t = self.rt.alloc_tensor_f32(&[in_f as usize, out_f as usize])?;
        t.buffer
            .try_write_f32(&packed)
            .map_err(|e| format!("timesfm3 load {name}: {e}"))?;
        Ok(t)
    }

    fn attn(&self, prefix: &str, d: u32, mea_scale: bool) -> Result<Attn, String> {
        let p = |r: &str| format!("{prefix}.{r}");
        let gq = self.vec(&p("query_ln.weight"), HEAD_DIM)?;
        let gk = self.vec(&p("key_ln.weight"), HEAD_DIM)?;
        let pds = self.vec(&p("per_dim_scale.per_dim_scale"), HEAD_DIM)?;
        let sdpa = if mea_scale { (HEAD_DIM as f64).sqrt() } else { 1.0 };
        let fold: Vec<f32> = gq
            .iter()
            .zip(&pds)
            .map(|(&g, &s)| {
                let softplus = (s as f64).exp().ln_1p();
                (g as f64 * RECIPROCAL_OF_SOFTPLUS_0 / (HEAD_DIM as f64).sqrt() * softplus * sdpa) as f32
            })
            .collect();
        Ok(Attn {
            q: self.linear(&p("query_proj.weight"), d, d)?,
            k: self.linear(&p("key_proj.weight"), d, d)?,
            v: self.linear(&p("value_proj.weight"), d, d)?,
            o: self.linear(&p("out_proj.weight"), d, d)?,
            gq: self.buf(&fold)?,
            gk: self.buf(&gk)?,
        })
    }
}

/// Activations of one forward over `rows` patch rows.
struct Acts {
    input: Tensor,
    hidden: Tensor,
    x: Tensor,
    xn: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    y: Tensor,
    logits: Tensor,
    sin: GpuBuffer,
    cos: GpuBuffer,
    lead: GpuBuffer,
    bits: GpuBuffer,
}

impl Timesfm3Model {
    /// Load from `st`. Every tensor's shape is checked against `cfg`, every
    /// weight must be finite, and a tensor the checkpoint carries that this
    /// model does not read is refused (`Loader::refuse_unread`).
    pub fn load(rt: &Arc<GpuRuntime>, st: &SafeTensors, cfg: Timesfm3Config) -> Result<Self, String> {
        let w = Weights {
            ld: Loader::new(rt, st, ""),
            rt,
        };
        let d = cfg.model_dims;
        let rin = cfg.host.resblock_in() as u32;
        let rh = cfg.resblock_hidden;
        let res_hidden = w.linear("pre_transformer_resblock.hidden_layer.weight", rin, rh)?;
        let res_out = w.linear("pre_transformer_resblock.output_layer.weight", rh, d)?;
        let res_skip = w.linear("pre_transformer_resblock.residual_layer.weight", rin, d)?;
        let mut layers = Vec::with_capacity(cfg.layers as usize);
        for i in 0..cfg.layers {
            let p = |r: &str| format!("transformer_stack.layers.{i}.{r}");
            let var = if cfg.use_variate_attention {
                Some((
                    w.buf(&w.vec(&p("pre_var_attn_ln.weight"), d)?)?,
                    w.buf(&w.vec(&p("post_var_attn_ln.weight"), d)?)?,
                    w.attn(&p("var_attn"), d, cfg.mea_scale)?,
                ))
            } else {
                None
            };
            layers.push(Layer {
                pre_seq: w.buf(&w.vec(&p("pre_seq_attn_ln.weight"), d)?)?,
                post_seq: w.buf(&w.vec(&p("post_seq_attn_ln.weight"), d)?)?,
                seq: w.attn(&p("seq_attn"), d, cfg.mea_scale)?,
                var,
                pre_ff: w.buf(&w.vec(&p("pre_ff_ln.weight"), d)?)?,
                post_ff: w.buf(&w.vec(&p("post_ff_ln.weight"), d)?)?,
                ff0: w.linear(&p("ff0.weight"), d, cfg.ff_hidden)?,
                ff1: w.linear(&p("ff1.weight"), cfg.ff_hidden, d)?,
            });
        }
        let ho = cfg.host.head_out() as u32;
        let head = w.linear("output_head.weight", d, ho)?;
        let hb = w.vec("output_head.bias", ho)?;
        let head_bias = rt.alloc_tensor_f32(&[ho as usize])?;
        head_bias
            .buffer
            .try_write_f32(&hb)
            .map_err(|e| format!("timesfm3 load head bias: {e}"))?;
        w.ld.refuse_unread()?;
        rt.synchronize()?;
        Ok(Self {
            cfg,
            rt: Arc::clone(rt),
            res_hidden,
            res_out,
            res_skip,
            layers,
            head,
            head_bias,
        })
    }

    pub fn config(&self) -> &Timesfm3Config {
        &self.cfg
    }

    /// Forecast a batch (upstream's `decode()`): host preparation, one device
    /// forward, host post-processing.
    pub fn decode(&self, input: &DecodeInput) -> Result<Forecast, String> {
        let hc = &self.cfg.host;
        let pr = host::prepare(hc, input)?;
        let pre = host::preprocess(hc, &pr)?;
        let fwd = self.forward(
            pr.batch,
            pr.variates,
            pr.patches,
            &pre.resblock_input,
            &pre.leading_masked,
            false,
        )?;
        let values = host::postprocess(hc, &pr, &pre, &fwd.raw_logits)?;
        Ok(Forecast {
            batch: pr.batch,
            variates: pr.variates,
            horizon: pr.horizon,
            quantiles: hc.num_quantiles,
            values,
        })
    }

    /// Upstream's `TimesFM3Forecaster.predict_batch()`: one quantile forecast
    /// per series, `[targets, horizon, quantiles]` (see
    /// [`crate::timesfm3_forecast`] for what is done to the series around
    /// the model). The plan's calls (groups of `per_core_batch_size`
    /// queries) whose shapes agree run as shared forwards of whole series, up
    /// to [`MAX_ROWS`] patch rows each ([`forecast::pack`]); calls of another
    /// shape run apart. Series in one batch never interact and a row's bits do
    /// not depend on how many others share its forward, so a series' forecast
    /// is the same bit for bit packed, split or alone.
    pub fn predict_batch(
        &self,
        series: &[Series],
        horizon: usize,
        opts: &PredictOptions,
    ) -> Result<Vec<SeriesForecast>, String> {
        self.predict_batch_capped(series, horizon, opts, MAX_ROWS)
    }

    /// [`Self::predict_batch`] with at most `max_rows` patch rows per
    /// forward (tests lower it to exercise the split).
    #[doc(hidden)]
    pub fn predict_batch_capped(
        &self,
        series: &[Series],
        horizon: usize,
        opts: &PredictOptions,
        max_rows: usize,
    ) -> Result<Vec<SeriesForecast>, String> {
        let plan = forecast::plan(&self.cfg.host, series, horizon, opts)?;
        let packing = forecast::pack(&self.cfg.host, &plan.calls, max_rows)?;
        let outs = packing
            .forwards
            .iter()
            .map(|forward| self.decode(forward))
            .collect::<Result<Vec<_>, _>>()?;
        forecast::finish(&plan, &packing.unpack(&outs)?)
    }

    /// [`Self::decode`] in pieces of whole batch elements, each at most
    /// `max_rows` (and [`MAX_ROWS`]) patch rows; the pieces' forecasts are
    /// concatenated. Refuses, naming the sizes, a single batch element larger
    /// than that. (The cut is [`forecast::pack`]'s, so `predict_batch` and
    /// this split a call at the same elements.)
    pub fn decode_capped(&self, input: &DecodeInput, max_rows: usize) -> Result<Forecast, String> {
        let packing = forecast::pack(&self.cfg.host, std::slice::from_ref(input), max_rows)?;
        let outs = packing
            .forwards
            .iter()
            .map(|forward| self.decode(forward))
            .collect::<Result<Vec<_>, _>>()?;
        packing
            .unpack(&outs)?
            .pop()
            .ok_or_else(|| "timesfm3: empty batch".to_string())
    }

    /// The device forward on `b * v * n` patch rows: `resblock_input` is
    /// `[b, v, n, 2 (p + o)]` and `leading_masked` `[b * v]` (from
    /// [`host::preprocess`]). Returns the raw head output.
    pub fn forward(
        &self,
        b: usize,
        v: usize,
        n: usize,
        resblock_input: &[f32],
        leading_masked: &[u32],
        trace: bool,
    ) -> Result<ForwardOutput, String> {
        const WHAT: &str = "Timesfm3Model::forward";
        let rt = &self.rt;
        let cfg = &self.cfg;
        if rt.f32_gemm_runs_relaxed() {
            return Err(format!(
                "{WHAT}: the forward needs exact-f32 GEMMs; switch relaxed precision off"
            ));
        }
        let rows = b
            .checked_mul(v)
            .and_then(|x| x.checked_mul(n))
            .ok_or_else(|| format!("{WHAT}: rows overflow"))?;
        if rows == 0 {
            return Err(format!("{WHAT}: empty batch"));
        }
        if rows > MAX_ROWS {
            return Err(format!(
                "{WHAT}: {rows} patch rows exceed MAX_ROWS = {MAX_ROWS}; split the batch"
            ));
        }
        if v > MAX_VARIATES {
            return Err(format!("{WHAT}: {v} variates exceed {MAX_VARIATES}"));
        }
        let rin = cfg.host.resblock_in();
        if resblock_input.len() != rows * rin {
            return Err(format!(
                "{WHAT}: resblock_input holds {}, expected {}",
                resblock_input.len(),
                rows * rin
            ));
        }
        if leading_masked.len() != b * v {
            return Err(format!(
                "{WHAT}: leading_masked holds {}, expected {}",
                leading_masked.len(),
                b * v
            ));
        }
        if let Some(&bad) = leading_masked.iter().find(|&&l| l as usize > n) {
            return Err(format!("{WHAT}: leading_masked {bad} exceeds {n} patches"));
        }
        if resblock_input.iter().any(|x| !x.is_finite()) {
            return Err(format!("{WHAT}: resblock_input holds a non-finite value"));
        }
        let scope = rt.async_scope();
        let ran = self.forward_inner(b, v, n, rows, resblock_input, leading_masked, trace);
        let waited = rt.synchronize();
        drop(scope);
        let out = ran?;
        waited?;
        Ok(out)
    }

    fn acts(&self, b: usize, v: usize, n: usize, rows: usize, leading: &[u32]) -> Result<Acts, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let d = cfg.model_dims as usize;
        let wide = (cfg.ff_hidden as usize).max(cfg.resblock_hidden as usize);
        let f32s = |shape: &[usize]| rt.alloc_tensor_unzeroed(shape, DType::F32);
        // RoPE tables, computed as upstream does in f32: timescale_i =
        // 10000^(2i/D), angle = pos / timescale.
        let half = (HEAD_DIM / 2) as usize;
        let mut sin = vec![0.0f32; n * half];
        let mut cos = vec![0.0f32; n * half];
        for i in 0..half {
            let fraction = 2.0f32 * i as f32 / HEAD_DIM as f32;
            let ts = (ROPE_MAX_TIMESCALE as f32).powf(fraction);
            for t in 0..n {
                let a = t as f32 / ts;
                sin[t * half + i] = a.sin();
                cos[t * half + i] = a.cos();
            }
        }
        // Variate key bitmask per (b, n): bit vi set where (b, vi) is past
        // its leading fully-masked patches at patch n.
        let mut bits = vec![0u32; b * n];
        for bi in 0..b {
            for ni in 0..n {
                for vi in 0..v {
                    if ni as u32 >= leading[bi * v + vi] {
                        bits[bi * n + ni] |= 1 << vi;
                    }
                }
            }
        }
        let sin_b = rt.alloc_buffer(sin.len() * 4)?;
        sin_b.try_write_f32(&sin)?;
        let cos_b = rt.alloc_buffer(cos.len() * 4)?;
        cos_b.try_write_f32(&cos)?;
        Ok(Acts {
            input: f32s(&[rows, cfg.host.resblock_in()])?,
            hidden: f32s(&[rows, wide])?,
            x: f32s(&[rows, d])?,
            xn: f32s(&[rows, d])?,
            q: f32s(&[rows, d])?,
            k: f32s(&[rows, d])?,
            v: f32s(&[rows, d])?,
            o: f32s(&[rows, d])?,
            y: f32s(&[rows, d])?,
            logits: f32s(&[rows, cfg.host.head_out()])?,
            sin: sin_b,
            cos: cos_b,
            lead: rt.alloc_buffer_from_u32(leading)?,
            bits: rt.alloc_buffer_from_u32(&bits)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        b: usize,
        v: usize,
        n: usize,
        rows: usize,
        resblock_input: &[f32],
        leading: &[u32],
        trace: bool,
    ) -> Result<ForwardOutput, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        let d = cfg.model_dims;
        let r32 = rows as u32;
        let a = self.acts(b, v, n, rows, leading)?;
        a.input.buffer.try_write_f32(resblock_input)?;
        let mut out_trace = Vec::new();
        let snap = |t: &Tensor, out: &mut Vec<Vec<f32>>| -> Result<(), String> {
            rt.synchronize()?;
            out.push(t.buffer.try_contents_f32()?[..rows * d as usize].to_vec());
            Ok(())
        };
        let relu = Epilogue {
            activation: Activation::Relu,
            ..Epilogue::default()
        };
        let accumulate = Epilogue {
            beta: 1.0,
            ..Epilogue::default()
        };

        // Residual block.
        let rh = a.hidden.try_view(&[rows, cfg.resblock_hidden as usize], 0)?;
        gemm_epilogue(&a.input, &self.res_hidden, &rh, BACKEND, relu)?;
        gemm(&rh, &self.res_out, &a.x, BACKEND)?;
        gemm_epilogue(&a.input, &self.res_skip, &a.x, BACKEND, accumulate)?;
        if trace {
            snap(&a.x, &mut out_trace)?;
        }

        let ffh = a.hidden.try_view(&[rows, cfg.ff_hidden as usize], 0)?;
        for layer in &self.layers {
            // Sequence attention.
            rms_norm_f32(rt, &a.x.buffer, &layer.pre_seq, &a.xn.buffer, r32, d, RMS_EPS)?;
            self.attention(&a, &layer.seq, rows, n, AttnAxis::Sequence { seqs: b * v })?;
            gemm(&a.o, &layer.seq.o, &a.y, BACKEND)?;
            rms_norm_residual_add_f32(rt, &a.y.buffer, &layer.post_seq, &a.x.buffer, r32, d, RMS_EPS, 1.0)?;
            // Variate attention.
            if let Some((pre, post, var)) = &layer.var {
                rms_norm_f32(rt, &a.x.buffer, pre, &a.xn.buffer, r32, d, RMS_EPS)?;
                if v == 1 {
                    gemm(&a.xn, &var.v, &a.o, BACKEND)?;
                } else {
                    self.attention(&a, var, rows, n, AttnAxis::Variate { b, v })?;
                }
                gemm(&a.o, &var.o, &a.y, BACKEND)?;
                rms_norm_residual_add_f32(rt, &a.y.buffer, post, &a.x.buffer, r32, d, RMS_EPS, 1.0)?;
            }
            // Feed-forward.
            rms_norm_f32(rt, &a.x.buffer, &layer.pre_ff, &a.xn.buffer, r32, d, RMS_EPS)?;
            gemm_epilogue(&a.xn, &layer.ff0, &ffh, BACKEND, relu)?;
            gemm(&ffh, &layer.ff1, &a.y, BACKEND)?;
            rms_norm_residual_add_f32(rt, &a.y.buffer, &layer.post_ff, &a.x.buffer, r32, d, RMS_EPS, 1.0)?;
            if trace {
                snap(&a.x, &mut out_trace)?;
            }
        }
        let head = Epilogue {
            bias: Some(&self.head_bias),
            ..Epilogue::default()
        };
        gemm_epilogue(&a.x, &self.head, &a.logits, BACKEND, head)?;
        rt.synchronize()?;
        let raw_logits = a.logits.buffer.try_contents_f32()?[..rows * cfg.host.head_out()].to_vec();
        Ok(ForwardOutput {
            raw_logits,
            trace: out_trace,
        })
    }

    /// Q, K, V from `xn`, the QK preparation, and attention into `o`.
    fn attention(&self, a: &Acts, w: &Attn, rows: usize, n: usize, axis: AttnAxis) -> Result<(), String> {
        let rt = &self.rt;
        let h = self.cfg.heads;
        gemm(&a.xn, &w.q, &a.q, BACKEND)?;
        gemm(&a.xn, &w.k, &a.k, BACKEND)?;
        gemm(&a.xn, &w.v, &a.v, BACKEND)?;
        let rope = matches!(axis, AttnAxis::Sequence { .. });
        let prep = rt.pipeline("timesfm3_qk_prep_h80")?;
        let threads = 2 * rows * h as usize;
        dispatch_1d(rt, &prep, threads, |bnd| {
            set_gpu_buf(bnd, &a.q.buffer, 0);
            set_gpu_buf(bnd, &a.k.buffer, 1);
            set_gpu_buf(bnd, &w.gq, 2);
            set_gpu_buf(bnd, &w.gk, 3);
            set_gpu_buf(bnd, &a.sin, 4);
            set_gpu_buf(bnd, &a.cos, 5);
            set_u32(bnd, rows as u32, 6);
            set_u32(bnd, h, 7);
            set_u32(bnd, n as u32, 8);
            set_u32(bnd, rope as u32, 9);
            set_f32(bnd, RMS_EPS, 10);
        })?;
        // (T, sequences, P, S1, S2, S3, causal, info)
        let (t, seqs, p, s1, s2, s3, causal, info) = match axis {
            AttnAxis::Sequence { seqs } => (n, seqs, 1, n, 0, 1, 1u32, &a.lead),
            AttnAxis::Variate { b, v } => (v, b * n, n, v * n, 1, n, 0u32, &a.bits),
        };
        let attn = rt.pipeline("timesfm3_attn_h80_r4_g16")?;
        const ROWS_PER_TG: usize = 16 * (32 / 4);
        dispatch_2d_tg(rt, &attn, t.div_ceil(ROWS_PER_TG), seqs * h as usize, 16 * 32, |bnd| {
            set_gpu_buf(bnd, &a.q.buffer, 0);
            set_gpu_buf(bnd, &a.k.buffer, 1);
            set_gpu_buf(bnd, &a.v.buffer, 2);
            set_gpu_buf(bnd, &a.o.buffer, 3);
            set_u32(bnd, t as u32, 4);
            set_u32(bnd, h, 5);
            set_u32(bnd, p as u32, 6);
            set_u32(bnd, s1 as u32, 7);
            set_u32(bnd, s2 as u32, 8);
            set_u32(bnd, s3 as u32, 9);
            set_u32(bnd, causal, 10);
            set_gpu_buf(bnd, info, 11);
            set_f32(bnd, 1.0, 12);
        })
    }
}

#[derive(Clone, Copy)]
enum AttnAxis {
    /// Causal over the n patches of each of `seqs = b * v` rows.
    Sequence { seqs: usize },
    /// Bidirectional over the v variates of each (b, n).
    Variate { b: usize, v: usize },
}
