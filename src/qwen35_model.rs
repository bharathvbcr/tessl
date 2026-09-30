//! The Qwen3.5 text model's prefill forward, composed from [`crate::qwen35`]'s
//! kernels and loaded straight from a `.safetensors` checkpoint.
//!
//! Each kernel is checked against transformers on its own (`docs/qwen35.md`);
//! this is the composition, the one place the layer order, norm offsets,
//! weight layouts and tied LM head are written down, and what
//! `tests/qwen35_model.rs` checks end to end against transformers' logits.
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
//! and after the last layer `logits = rms_norm(resid) * (1 + norm.w) @ embedᵀ`
//! (Qwen3.5 ties the LM head to the embedding). Zero-centred norm weights are
//! folded to `1 + w` in f32 on the host, which is the value transformers
//! multiplies by; the GDN gated norm's weight is used as stored, and the
//! attention Q/K norms apply their own `1 + w` in the kernel.
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
//! Prefill only, batch 1, positions from 0: the GDN state and K/V caches start
//! empty and are not returned. Decode and many-question continuation use the
//! kernels directly (`qwen35::gdn_recurrent`, `attn_prefix_rows`, ...).

use std::sync::Arc;

use crate::gemm::{gemm, GemmBackend};
use crate::nn::{self, AttnDims};
use crate::qwen35::{
    self, AttnProjLayout, AttnShape, AttnTargets, Cols, GdnParams, GdnProjLayout, GdnWorkspace,
    LmHead, OutCols, StateIn,
};
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
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
}

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
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.intermediate == 0 || self.vocab == 0 || self.layers.is_empty() {
            return Err("Qwen35Config: hidden, intermediate, vocab and layers must be non-zero".into());
        }
        if self.conv_kernel == 0 {
            return Err("Qwen35Config: conv_kernel must be non-zero".into());
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0)
            || !(self.rope_theta.is_finite() && self.rope_theta > 0.0)
        {
            return Err("Qwen35Config: rms_norm_eps and rope_theta must be positive".into());
        }
        if self.rotary_dim % 2 != 0 || self.rotary_dim > self.attn.head_dim() {
            return Err("Qwen35Config: rotary_dim must be even and at most head_dim".into());
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

struct GdnWeights {
    w_in: Tensor,
    w_out: Tensor,
    conv_w: GpuBuffer,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    norm_w: GpuBuffer,
}

struct AttnWeights {
    w_in: Tensor,
    w_out: Tensor,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
}

enum Mixer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

struct Layer {
    input_norm: GpuBuffer,
    post_norm: GpuBuffer,
    mixer: Mixer,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

/// A loaded model: weights on the device in the layouts the kernels read.
pub struct Qwen35Model {
    cfg: Qwen35Config,
    precision: Precision,
    rt: Arc<GpuRuntime>,
    /// `[vocab, hidden]` bf16: the embedding gather's table.
    embed: GpuBuffer,
    /// `[hidden, vocab]` in the forward's precision: the tied LM head, as the
    /// right operand of one GEMM.
    lm_head: Tensor,
    final_norm: GpuBuffer,
    layers: Vec<Layer>,
}

/// What [`Qwen35Model::forward`] returns.
pub struct ForwardOutput {
    /// `[tokens, vocab]` f32.
    pub logits: Vec<f32>,
    /// With `trace`: the residual stream after each layer, `[tokens, hidden]`
    /// each; then the final norm's output. Empty otherwise.
    pub trace: Vec<Vec<f32>>,
}

/// Host-side loader state: one checkpoint, one tensor-name prefix.
struct Loader<'a> {
    st: &'a SafeTensors,
    prefix: &'a str,
    rt: &'a Arc<GpuRuntime>,
}

impl Loader<'_> {
    fn name(&self, rest: &str) -> String {
        format!("{}{rest}", self.prefix)
    }

    /// A tensor widened to f32, with its shape checked.
    fn f32(&self, rest: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let name = self.name(rest);
        let (got, data) = self.st.read_f32(&name)?;
        if got != shape {
            return Err(format!("{name}: shape {got:?}, expected {shape:?}"));
        }
        Ok(data)
    }

    fn f32_buf(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let b = self.rt.alloc_buffer(data.len().max(1) * 4)?;
        b.write_f32(data);
        Ok(b)
    }

    /// A zero-centred norm weight, folded to `1 + w` in f32.
    fn norm_plus_one(&self, rest: &str, dim: usize) -> Result<GpuBuffer, String> {
        let w: Vec<f32> = self.f32(rest, &[dim])?.iter().map(|w| 1.0 + w).collect();
        self.f32_buf(&w)
    }

    /// `nn.Linear` weights `[out_i, in]` packed side by side into the right
    /// operand `[in, sum(out_i)]` of one GEMM, in `precision`.
    fn linear(
        &self,
        parts: &[(&str, usize)],
        in_features: usize,
        precision: Precision,
    ) -> Result<Tensor, String> {
        let widths: Vec<usize> = parts.iter().map(|&(_, o)| o).collect();
        let total: usize = widths.iter().sum();
        match precision {
            Precision::Bf16 => {
                let mut bits = Vec::with_capacity(parts.len());
                for &(rest, out) in parts {
                    let name = self.name(rest);
                    let (shape, b) = self.st.read_bf16_bits(&name)?;
                    if shape != [out, in_features] {
                        return Err(format!("{name}: shape {shape:?}, expected [{out}, {in_features}]"));
                    }
                    bits.push(b);
                }
                let refs: Vec<&[u16]> = bits.iter().map(Vec::as_slice).collect();
                let packed = qwen35::pack_linear_weights_bf16(&refs, &widths, in_features)?;
                let t = self.rt.alloc_tensor_bf16(&[in_features, total])?;
                t.buffer.write_bf16_bits(&packed);
                Ok(t)
            }
            Precision::F32 => {
                let mut data = Vec::with_capacity(parts.len());
                for &(rest, out) in parts {
                    data.push(self.f32(rest, &[out, in_features])?);
                }
                let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
                let packed = qwen35::pack_linear_weights_f32(&refs, &widths, in_features)?;
                let t = self.rt.alloc_tensor_f32(&[in_features, total])?;
                t.buffer.write_f32(&packed);
                Ok(t)
            }
        }
    }
}

impl Qwen35Model {
    /// Load the text tower from `st`, whose tensors are named `{prefix}...`
    /// (`"model.language_model."` in the Qwen3.5 checkpoints). Every tensor's
    /// shape is checked against `cfg`; a missing one is an error.
    pub fn load(
        rt: &Arc<GpuRuntime>,
        st: &SafeTensors,
        prefix: &str,
        cfg: Qwen35Config,
        precision: Precision,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let ld = Loader { st, prefix, rt };
        let (h, inter, vocab) = (cfg.hidden as usize, cfg.intermediate as usize, cfg.vocab as usize);

        // Embedding (bf16 table for the gather) and the tied LM head.
        let name = ld.name("embed_tokens.weight");
        let (shape, embed_bits) = st.read_bf16_bits(&name)?;
        if shape != [vocab, h] {
            return Err(format!("{name}: shape {shape:?}, expected [{vocab}, {h}]"));
        }
        let embed = rt.alloc_buffer(embed_bits.len() * 2)?;
        embed.write_bf16_bits(&embed_bits);
        let lm_head = match precision {
            Precision::Bf16 => {
                let packed = qwen35::pack_linear_weights_bf16(&[&embed_bits], &[vocab], h)?;
                let t = rt.alloc_tensor_bf16(&[h, vocab])?;
                t.buffer.write_bf16_bits(&packed);
                t
            }
            Precision::F32 => {
                let wide: Vec<f32> = embed_bits
                    .iter()
                    .map(|&b| crate::tensor::bf16_bits_to_f32(b))
                    .collect();
                let packed = qwen35::pack_linear_weights_f32(&[&wide], &[vocab], h)?;
                let t = rt.alloc_tensor_f32(&[h, vocab])?;
                t.buffer.write_f32(&packed);
                t
            }
        };
        drop(embed_bits);
        let final_norm = ld.norm_plus_one("norm.weight", h)?;

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
                        conv_w: ld.f32_buf(&ld.f32(&p("linear_attn.conv1d.weight"), &[conv_dim, 1, kw])?)?,
                        a_log: ld.f32_buf(&ld.f32(&p("linear_attn.A_log"), &[vh])?)?,
                        dt_bias: ld.f32_buf(&ld.f32(&p("linear_attn.dt_bias"), &[vh])?)?,
                        norm_w: ld.f32_buf(&ld.f32(&p("linear_attn.norm.weight"), &[g.v_dim() as usize])?)?,
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
                        q_norm: ld.f32_buf(&ld.f32(&p("self_attn.q_norm.weight"), &[d])?)?,
                        k_norm: ld.f32_buf(&ld.f32(&p("self_attn.k_norm.weight"), &[d])?)?,
                    })
                }
            };
            layers.push(Layer {
                input_norm: ld.norm_plus_one(&p("input_layernorm.weight"), h)?,
                post_norm: ld.norm_plus_one(&p("post_attention_layernorm.weight"), h)?,
                mixer,
                gate: ld.linear(&[(&p("mlp.gate_proj.weight"), inter)], h, precision)?,
                up: ld.linear(&[(&p("mlp.up_proj.weight"), inter)], h, precision)?,
                down: ld.linear(&[(&p("mlp.down_proj.weight"), h)], inter, precision)?,
            });
        }
        rt.synchronize()?;
        Ok(Self {
            cfg,
            precision,
            rt: Arc::clone(rt),
            embed,
            lm_head,
            final_norm,
            layers,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Prefill `ids` (one sequence, positions from 0) and return every
    /// position's logits; with `trace`, also the residual stream after each
    /// layer and the final norm's output (one host read per layer, so slower).
    pub fn forward(&self, ids: &[u32], trace: bool) -> Result<ForwardOutput, String> {
        let rt = &self.rt;
        let cfg = &self.cfg;
        if ids.is_empty() {
            return Err("Qwen35Model::forward: no tokens".into());
        }
        if let Some(&bad) = ids.iter().find(|&&id| id >= cfg.vocab) {
            return Err(format!("Qwen35Model::forward: token id {bad} >= vocab {}", cfg.vocab));
        }
        if self.precision == Precision::F32 && rt.relaxed_precision() {
            return Err(
                "Qwen35Model::forward: the F32 forward needs exact-f32 GEMMs; switch the \
                 runtime's relaxed precision off"
                    .into(),
            );
        }
        let t = u32::try_from(ids.len()).map_err(|_| "Qwen35Model::forward: too many tokens")?;
        let a = Acts::new(rt, cfg, self.precision, t)?;

        let id_buf = rt.alloc_buffer(ids.len() * 4)?;
        id_buf.write_u32(ids);
        qwen35::embed_rows(
            rt,
            &id_buf,
            t,
            LmHead {
                weight: &self.embed,
                dtype: DType::BF16,
                vocab: cfg.vocab,
            },
            cfg.hidden,
            &a.resid.buffer,
        )?;

        let mut out_trace = Vec::new();
        for layer in &self.layers {
            self.norm(&a.resid, &layer.input_norm, &a.x)?;
            match &layer.mixer {
                Mixer::Gdn(w) => self.gdn(w, &a)?,
                Mixer::Attn(w) => self.attention(w, &a)?,
            }
            self.norm(&a.resid, &layer.post_norm, &a.x)?;
            self.mlp(layer, &a)?;
            if trace {
                rt.synchronize()?;
                out_trace.push(a.resid.buffer.read_f32()[..(t * cfg.hidden) as usize].to_vec());
            }
        }
        self.norm(&a.resid, &self.final_norm, &a.x)?;
        gemm(&a.x, &self.lm_head, &a.logits, BACKEND)?;
        rt.synchronize()?;
        if trace {
            out_trace.push(read_rows(&a.x, self.precision)?);
        }
        let logits = a.logits.buffer.read_f32()[..t as usize * cfg.vocab as usize].to_vec();
        Ok(ForwardOutput {
            logits,
            trace: out_trace,
        })
    }

    /// `out = rms_norm(x) * w` in the forward's activation dtype.
    fn norm(&self, x: &Tensor, w: &GpuBuffer, out: &Tensor) -> Result<(), String> {
        let (rows, dim) = (x.shape[0] as u32, x.shape[1] as u32);
        let eps = self.cfg.rms_norm_eps;
        match self.precision {
            Precision::Bf16 => nn::rms_norm_bf16(&self.rt, &x.buffer, w, &out.buffer, rows, dim, eps),
            Precision::F32 => nn::rms_norm_f32(&self.rt, &x.buffer, w, &out.buffer, rows, dim, eps),
        }
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

    fn gdn(&self, w: &GdnWeights, a: &Acts) -> Result<(), String> {
        let (rt, g, t) = (&self.rt, self.cfg.gdn, a.t);
        gemm(&a.x, &w.w_in, &a.g_proj, BACKEND)?;
        let proj = &a.g_proj.buffer;
        qwen35::conv1d_silu(
            rt,
            Cols::dense(proj, g.width()),
            &w.conv_w,
            self.cfg.conv_kernel,
            StateIn::Zero,
            &a.g_qkv,
            None,
            1,
            t,
            g.conv_dim(),
        )?;
        qwen35::gdn_chunk_forward(
            rt,
            &g.dims(1, t),
            &g.conv_qkv(&a.g_qkv),
            &g.gates(proj),
            &GdnParams {
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
            },
            StateIn::Zero,
            &a.g_ws,
            Cols::dense(&a.g_o, g.value_dim()),
            None,
        )?;
        qwen35::gated_rms_norm(
            rt,
            Cols::dense(&a.g_o, g.value_dim()),
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

    fn attention(&self, w: &AttnWeights, a: &Acts) -> Result<(), String> {
        let (rt, l, t) = (&self.rt, self.cfg.attn, a.t);
        gemm(&a.x, &w.w_in, &a.a_proj, BACKEND)?;
        let pc = Cols::dense(&a.a_proj.buffer, l.width());
        qwen35::attn_qk_norm_rope(
            rt,
            &AttnShape {
                batch: 1,
                seq: t,
                q_heads: l.q_heads(),
                kv_heads: l.kv_heads(),
                head_dim: l.head_dim(),
                rotary_dim: self.cfg.rotary_dim,
            },
            pc,
            &w.q_norm,
            &w.k_norm,
            &AttnTargets {
                q_out: &a.a_q,
                k_cache: &a.a_k,
                v_cache: &a.a_v,
            },
            0,
            self.cfg.rope_theta,
            self.cfg.rms_norm_eps,
        )?;
        qwen35::attn_prefill(
            rt,
            &a.a_q,
            &a.a_k,
            &a.a_v,
            &a.a_o,
            &a.tkv,
            &a.zero,
            &a.zero,
            AttnDims {
                batch: 1,
                tq: t,
                heads: l.q_heads(),
                heads_kv: l.kv_heads(),
                window: 0,
                scale: 1.0 / (l.head_dim() as f32).sqrt(),
            },
            false,
        )?;
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

/// A `[rows, cols]` activation read back as f32 (bf16 widened exactly).
fn read_rows(x: &Tensor, precision: Precision) -> Result<Vec<f32>, String> {
    let n = x.shape.iter().product::<usize>();
    Ok(match precision {
        Precision::F32 => x.buffer.read_f32()[..n].to_vec(),
        Precision::Bf16 => x.buffer.contents_u16()[..n]
            .iter()
            .map(|&b| crate::tensor::bf16_bits_to_f32(b))
            .collect(),
    })
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
    a_k: GpuBuffer,
    a_v: GpuBuffer,
    a_o: GpuBuffer,
    a_y: Tensor,
    tkv: GpuBuffer,
    zero: GpuBuffer,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
    logits: Tensor,
}

impl Acts {
    fn new(rt: &Arc<GpuRuntime>, cfg: &Qwen35Config, p: Precision, t: u32) -> Result<Self, String> {
        let tu = t as usize;
        let (g, l) = (cfg.gdn, cfg.attn);
        let act = |cols: usize| match p {
            Precision::Bf16 => rt.alloc_tensor_bf16(&[tu, cols]),
            Precision::F32 => rt.alloc_tensor_f32(&[tu, cols]),
        };
        let f32s = |n: usize| rt.alloc_buffer(n.max(1) * 4);
        let qd = (l.q_heads() * l.head_dim()) as usize;
        let kv = tu * (l.kv_heads() * l.head_dim()) as usize;
        let u32_buf = |v: u32| -> Result<GpuBuffer, String> {
            let b = rt.alloc_buffer(4)?;
            b.write_u32(&[v]);
            Ok(b)
        };
        Ok(Self {
            t,
            resid: rt.alloc_tensor_f32(&[tu, cfg.hidden as usize])?,
            x: act(cfg.hidden as usize)?,
            proj_out: rt.alloc_tensor_f32(&[tu, cfg.hidden as usize])?,
            g_proj: rt.alloc_tensor_f32(&[tu, g.width() as usize])?,
            g_qkv: f32s(tu * g.conv_dim() as usize)?,
            g_o: f32s(tu * g.value_dim() as usize)?,
            g_y: act(g.value_dim() as usize)?,
            g_ws: GdnWorkspace::new(rt, &g.dims(1, t))?,
            a_proj: rt.alloc_tensor_f32(&[tu, l.width() as usize])?,
            a_q: f32s(tu * qd)?,
            a_k: f32s(kv)?,
            a_v: f32s(kv)?,
            a_o: f32s(tu * qd)?,
            a_y: act(qd)?,
            tkv: u32_buf(t)?,
            zero: u32_buf(0)?,
            m_gate: rt.alloc_tensor_f32(&[tu, cfg.intermediate as usize])?,
            m_up: rt.alloc_tensor_f32(&[tu, cfg.intermediate as usize])?,
            m_mid: act(cfg.intermediate as usize)?,
            logits: rt.alloc_tensor_f32(&[tu, cfg.vocab as usize])?,
        })
    }
}
