//! One training step of the Qwen3.5 text model: the forward with what the
//! backward needs saved, the causal-LM loss, and every parameter's gradient.
//!
//! The loss is transformers' `ForCausalLMLoss` for one unpadded sequence:
//! position `t` predicts `ids[t + 1]`, cross-entropy averaged over the
//! `T - 1` predicted positions. The step runs in f32 ([`Precision::F32`]),
//! with the training kernels where the inference forward's cannot give a
//! backward what it needs: [`crate::gdn_train`] for the gated delta rule
//! (checkpointed state), [`crate::attn_train`] for attention (log-sum-exp),
//! and [`crate::qwen35::gdn_gates`] for the gates as values. Everything else
//! is the inference forward's kernels, and the backward is
//! [`crate::qwen35_bwd`], [`crate::cross_entropy`] and exact-f32 GEMMs.
//!
//! Every gradient is in the layout of the weight it belongs to: the fused
//! projections' packed `[in, out]` right operands, the conv weight
//! `[channels, kernel_width]`, and `[vocab, hidden]` for the tied embedding /
//! LM head (both uses summed). The zero-centred norms are stored as `1 + w`,
//! whose gradient is the gradient of `w`. Every reduction runs in a fixed
//! order, so a step's gradients are the same bits on every run.
//!
//! Scope: one sequence, positions from 0, value heads equal to key heads in
//! the GDN (Qwen3.5-2B's 16 and 16; `gdn_train` has no head grouping).
//!
//! What the forward keeps for the backward is [`Activations`]: every layer's
//! intermediates ([`Activations::Saved`]), or only the residual stream into
//! each layer, with one layer's intermediates rebuilt at a time just before
//! its backward ([`Activations::Recomputed`]). The kernels are deterministic,
//! so the rebuilt intermediates are the forward's bits and both modes return
//! the same loss and gradients bit for bit.

use std::sync::Arc;

use crate::attn_train::{attn_train_backward, attn_train_forward, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace};
use crate::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use crate::gdn_train::{
    gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace, GDN_TRAIN_DK,
};
use crate::gemm::{gemm, gemm_nt_f32, gemm_tn_f32, GemmBackend};
use crate::nn;
use crate::qwen35::{self, AttnShape, AttnTargets, Cols, GdnParams, OutCols, StateIn};
use crate::qwen35_bwd::{
    attn_gate_bwd, attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, conv1d_silu_bwd, conv1d_silu_bwd_part_len,
    copy_cols, embed_rows_bwd, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, gdn_gates_bwd, gdn_gates_bwd_part_len,
    rms_norm_bwd, rms_norm_bwd_part_len, swiglu_bwd, AttnQkvGrads, EmbedBwdWorkspace,
};
use crate::qwen35_model::{AttnWeights, GdnWeights, Layer, Mixer, Precision, Qwen35Model};
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, GpuBuffer, Tensor};

const BACKEND: GemmBackend = GemmBackend::TensorOps;
/// Vocabulary columns per cross-entropy chunk.
const CE_CHUNK: u32 = 8192;

/// The gradients of a GDN layer's mixer, in its weights' layouts.
pub struct GdnGrads {
    /// `[hidden, width]`: `in_proj_qkv | in_proj_z | in_proj_b | in_proj_a`.
    pub w_in: Tensor,
    /// `[value_dim, hidden]`.
    pub w_out: Tensor,
    /// `[conv_dim, kernel_width]`.
    pub conv_w: GpuBuffer,
    pub a_log: GpuBuffer,
    pub dt_bias: GpuBuffer,
    /// `[v_dim]`, the gated norm's weight.
    pub norm_w: GpuBuffer,
}

/// The gradients of an attention layer's mixer, in its weights' layouts.
pub struct AttnGrads {
    /// `[hidden, width]`: `q_proj (query and gate per head) | k_proj | v_proj`.
    pub w_in: Tensor,
    /// `[q_heads * head_dim, hidden]`.
    pub w_out: Tensor,
    pub q_norm: GpuBuffer,
    pub k_norm: GpuBuffer,
}

pub enum MixerGrads {
    Gdn(GdnGrads),
    Attn(AttnGrads),
}

/// One layer's gradients.
pub struct LayerGrads {
    pub input_norm: GpuBuffer,
    pub post_norm: GpuBuffer,
    pub mixer: MixerGrads,
    /// `[hidden, intermediate]`.
    pub gate: Tensor,
    /// `[hidden, intermediate]`.
    pub up: Tensor,
    /// `[intermediate, hidden]`.
    pub down: Tensor,
}

/// Every parameter's gradient.
pub struct Qwen35Grads {
    /// `[vocab, hidden]`: the tied embedding and LM head, both uses summed.
    pub embed: Tensor,
    pub final_norm: GpuBuffer,
    pub layers: Vec<LayerGrads>,
}

/// What [`Qwen35Model::train_step`] keeps from the forward for the backward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activations {
    /// Every layer's intermediates: no recomputation, most memory.
    Saved,
    /// The residual stream into each layer (`T x hidden` f32 per layer); each
    /// layer's forward runs again just before its backward. About one more
    /// forward's work for a fraction of the memory.
    Recomputed,
}

/// What [`Qwen35Model::train_step`] returns.
pub struct TrainStep {
    /// Mean cross-entropy over the `T - 1` predicted positions.
    pub loss: f64,
    pub grads: Qwen35Grads,
    /// Bytes of per-layer activations held from the forward until the
    /// backward, summed over the tensors actually kept (logical sizes).
    pub activation_bytes: u64,
}

/// What one GDN layer's forward keeps for its backward.
struct SavedGdn {
    proj: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    ckpt: Tensor,
    o: Tensor,
    y: Tensor,
}

/// What one attention layer's forward keeps for its backward.
struct SavedAttn {
    proj: Tensor,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    o: GpuBuffer,
    lse: GpuBuffer,
    y: Tensor,
}

enum SavedMixer {
    Gdn(Box<SavedGdn>),
    Attn(SavedAttn),
}

impl SavedGdn {
    fn bytes(&self) -> u64 {
        [&self.proj, &self.q, &self.k, &self.v, &self.g, &self.beta, &self.ckpt, &self.o, &self.y]
            .iter()
            .map(|t| t.nbytes_logical() as u64)
            .sum()
    }
}

impl SavedAttn {
    fn bytes(&self) -> u64 {
        let bufs: u64 = [&self.q, &self.k, &self.v, &self.o, &self.lse].iter().map(|b| b.nbytes() as u64).sum();
        bufs + (self.proj.nbytes_logical() + self.y.nbytes_logical()) as u64
    }
}

/// What one layer's forward keeps: the residual stream into each norm, each
/// norm's output (the GEMMs' left operand), the mixer's, and the MLP's.
struct Saved {
    resid_in: Tensor,
    x1: Tensor,
    mixer: SavedMixer,
    resid_mid: Tensor,
    x2: Tensor,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
}

impl Saved {
    fn bytes(&self) -> u64 {
        let own: usize = [&self.resid_in, &self.x1, &self.resid_mid, &self.x2, &self.m_gate, &self.m_up, &self.m_mid]
            .iter()
            .map(|t| t.nbytes_logical())
            .sum();
        own as u64
            + match &self.mixer {
                SavedMixer::Gdn(s) => s.bytes(),
                SavedMixer::Attn(s) => s.bytes(),
            }
    }
}

/// What the forward hands one layer's backward.
enum Kept {
    Saved(Box<Saved>),
    /// The residual stream into the layer, to rebuild [`Saved`] from.
    Input(Tensor),
}

/// The scratch every layer's backward reuses.
struct Scratch {
    t: u32,
    /// The residual stream's gradient.
    dresid: Tensor,
    /// A norm input's gradient before it joins `dresid`.
    dx: Tensor,
    tmp_h: Tensor,
    d_mid: Tensor,
    d_gate: Tensor,
    d_up: Tensor,
    norm_part: GpuBuffer,
    gdn: Option<GdnScratch>,
    attn: Option<AttnScratch>,
}

struct GdnScratch {
    dy: Tensor,
    d_o: Tensor,
    dproj: Tensor,
    d_conv: GpuBuffer,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    dg: Tensor,
    dbeta: Tensor,
    ws: GdnTrainWorkspace,
    norm_part: GpuBuffer,
    gates_part: GpuBuffer,
    conv_part: GpuBuffer,
}

struct AttnScratch {
    dy: Tensor,
    d_o: GpuBuffer,
    dproj: Tensor,
    dq: GpuBuffer,
    dk: GpuBuffer,
    dv: GpuBuffer,
    ws: AttnTrainWorkspace,
    part: GpuBuffer,
}

fn f32s(rt: &Arc<GpuRuntime>, n: usize) -> Result<GpuBuffer, String> {
    rt.alloc_buffer(n.max(1) * std::mem::size_of::<f32>())
}

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, String> {
    rt.alloc_tensor_f32(shape)
}

impl Qwen35Model {
    /// One training step on `ids` (one sequence, positions from 0): the loss
    /// transformers' `Qwen3_5ForCausalLM(input_ids=ids, labels=ids)` reports
    /// and every parameter's gradient of it, keeping `activations` from the
    /// forward for the backward.
    pub fn train_step(&self, ids: &[u32], activations: Activations) -> Result<TrainStep, String> {
        const WHAT: &str = "Qwen35Model::train_step";
        let (rt, cfg) = (&self.rt, &self.cfg);
        if self.precision != Precision::F32 {
            return Err(format!("{WHAT}: training runs in f32; load the model with Precision::F32"));
        }
        if rt.relaxed_precision() {
            return Err(format!("{WHAT}: the step needs exact-f32 GEMMs; switch the runtime's relaxed precision off"));
        }
        if cfg.gdn.k_heads() != cfg.gdn.v_heads() {
            return Err(format!(
                "{WHAT}: GDN value heads ({}) must equal key heads ({}); gdn_train has no head grouping",
                cfg.gdn.v_heads(),
                cfg.gdn.k_heads()
            ));
        }
        if ids.len() < 2 {
            return Err(format!("{WHAT}: a step needs at least two tokens (one prediction)"));
        }
        if let Some(&bad) = ids.iter().find(|&&id| id >= cfg.vocab) {
            return Err(format!("{WHAT}: token id {bad} >= vocab {}", cfg.vocab));
        }
        let t = u32::try_from(ids.len()).map_err(|_| format!("{WHAT}: too many tokens"))?;
        let (tu, h) = (t as usize, cfg.hidden as usize);

        // ---- forward --------------------------------------------------------
        let id_buf = rt.alloc_buffer(tu * std::mem::size_of::<u32>())?;
        id_buf.write_u32(ids);
        let mut resid = tensor(rt, &[tu, h])?;
        qwen35::embed_rows(
            rt,
            &id_buf,
            t,
            qwen35::LmHead { weight: &self.embed, dtype: DType::BF16, vocab: cfg.vocab },
            cfg.hidden,
            &resid.buffer,
        )?;
        let mut kept = Vec::with_capacity(self.layers.len());
        let mut activation_bytes = 0u64;
        for layer in &self.layers {
            let (s, out) = self.train_layer_forward(layer, resid, t, true)?;
            let out = out.ok_or("Qwen35Model::train_step: a layer's forward returned no output")?;
            kept.push(match activations {
                Activations::Saved => {
                    activation_bytes += s.bytes();
                    Kept::Saved(Box::new(s))
                }
                // Everything but the layer's input is dropped here, back to
                // the pool for the next layer.
                Activations::Recomputed => {
                    activation_bytes += s.resid_in.nbytes_logical() as u64;
                    Kept::Input(s.resid_in)
                }
            });
            resid = out;
        }
        let xf = tensor(rt, &[tu, h])?;
        nn::rms_norm_f32(rt, &resid.buffer, &self.final_norm, &xf.buffer, t, cfg.hidden, cfg.rms_norm_eps)?;

        // ---- loss and the LM head ------------------------------------------
        let n = tu - 1;
        let rows: Vec<u32> = (0..n as u32).collect();
        let targets = &ids[1..];
        let embed = Tensor::from_buffer(rt, self.embed.clone(), &[cfg.vocab as usize, h], DType::BF16, 0)?;
        let ce_ws = CeWorkspace::new(rt, n as u32, cfg.hidden, CE_CHUNK.min(cfg.vocab), DType::BF16)?;
        let d_embed = tensor(rt, &[cfg.vocab as usize, h])?;
        let dxf = tensor(rt, &[tu, h])?;
        // The last position predicts nothing: its gradient row stays zero.
        dxf.buffer.zero();
        let out = cross_entropy_rows(
            rt,
            CeHidden { rows: &xf, off: 0 },
            &embed,
            &rows,
            targets,
            Reduction::Mean,
            &ce_ws,
            Some(CeGrads { dh: &dxf.view(&[n, h], 0), dw: &d_embed, scale: 1.0 }),
        )?;

        // ---- backward -------------------------------------------------------
        let mut sc = self.scratch(t)?;
        let final_norm = f32s(rt, h)?;
        rms_norm_bwd(
            rt,
            &resid.buffer,
            &self.final_norm,
            &dxf.buffer,
            &sc.dresid.buffer,
            &final_norm,
            &sc.norm_part,
            t,
            cfg.hidden,
            cfg.rms_norm_eps,
            false,
        )?;
        let mut layers = Vec::with_capacity(self.layers.len());
        // Popped from the back, so a layer's kept activations are released
        // as soon as its backward is encoded.
        for layer in self.layers.iter().rev() {
            let s = match kept.pop().ok_or("Qwen35Model::train_step: fewer kept activations than layers")? {
                Kept::Saved(s) => *s,
                Kept::Input(resid_in) => self.train_layer_forward(layer, resid_in, t, false)?.0,
            };
            layers.push(self.train_layer_backward(layer, &s, &mut sc)?);
        }
        layers.reverse();
        let emb_ws = EmbedBwdWorkspace::new(rt, t)?;
        embed_rows_bwd(rt, ids, &sc.dresid.buffer, &d_embed.buffer, cfg.vocab, cfg.hidden, &emb_ws)?;
        rt.synchronize()?;
        Ok(TrainStep { loss: out.loss, grads: Qwen35Grads { embed: d_embed, final_norm, layers }, activation_bytes })
    }

    fn scratch(&self, t: u32) -> Result<Scratch, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let (g, a) = (cfg.gdn, cfg.attn);
        let has = |k: crate::qwen35_model::LayerKind| cfg.layers.contains(&k);
        let gdn = if has(crate::qwen35_model::LayerKind::LinearAttention) {
            let (hv, dv) = (g.v_heads() as usize, g.v_dim() as usize);
            let dims = GdnTrainDims { batch: 1, seq: t, heads: g.v_heads(), v_dim: g.v_dim() };
            let dk = GDN_TRAIN_DK as usize;
            Some(GdnScratch {
                dy: tensor(rt, &[tu, hv * dv])?,
                d_o: tensor(rt, &[1, tu, hv, dv])?,
                dproj: tensor(rt, &[tu, g.width() as usize])?,
                d_conv: f32s(rt, tu * g.conv_dim() as usize)?,
                dq: tensor(rt, &[1, tu, hv, dk])?,
                dk: tensor(rt, &[1, tu, hv, dk])?,
                dv: tensor(rt, &[1, tu, hv, dv])?,
                dg: tensor(rt, &[1, tu, hv])?,
                dbeta: tensor(rt, &[1, tu, hv])?,
                ws: GdnTrainWorkspace::new(rt, dims)?,
                norm_part: f32s(rt, gated_rms_norm_bwd_part_len(t, g.v_heads(), g.v_dim()))?,
                gates_part: f32s(rt, gdn_gates_bwd_part_len(t, g.v_heads()))?,
                conv_part: f32s(rt, conv1d_silu_bwd_part_len(1, t, g.conv_dim(), cfg.conv_kernel))?,
            })
        } else {
            None
        };
        let attn = if has(crate::qwen35_model::LayerKind::FullAttention) {
            let qd = (a.q_heads() * a.head_dim()) as usize;
            let kvd = (a.kv_heads() * a.head_dim()) as usize;
            Some(AttnScratch {
                dy: tensor(rt, &[tu, qd])?,
                d_o: f32s(rt, tu * qd)?,
                dproj: tensor(rt, &[tu, a.width() as usize])?,
                dq: f32s(rt, tu * qd)?,
                dk: f32s(rt, tu * kvd)?,
                dv: f32s(rt, tu * kvd)?,
                ws: AttnTrainWorkspace::new(rt, self.attn_dims(t))?,
                part: f32s(rt, attn_qk_norm_rope_bwd_part_len(&self.attn_shape(t)))?,
            })
        } else {
            None
        };
        Ok(Scratch {
            t,
            dresid: tensor(rt, &[tu, h])?,
            dx: tensor(rt, &[tu, h])?,
            tmp_h: tensor(rt, &[tu, h])?,
            d_mid: tensor(rt, &[tu, i])?,
            d_gate: tensor(rt, &[tu, i])?,
            d_up: tensor(rt, &[tu, i])?,
            norm_part: f32s(rt, rms_norm_bwd_part_len(t, cfg.hidden))?,
            gdn,
            attn,
        })
    }

    fn attn_shape(&self, t: u32) -> AttnShape {
        let a = self.cfg.attn;
        AttnShape {
            batch: 1,
            seq: t,
            q_heads: a.q_heads(),
            kv_heads: a.kv_heads(),
            head_dim: a.head_dim(),
            rotary_dim: self.cfg.rotary_dim,
        }
    }

    fn attn_dims(&self, t: u32) -> AttnTrainDims {
        let a = self.cfg.attn;
        AttnTrainDims {
            batch: 1,
            seq: t,
            q_heads: a.q_heads(),
            kv_heads: a.kv_heads(),
            scale: 1.0 / (a.head_dim() as f32).sqrt(),
        }
    }

    /// `out = rms_norm(x) * w`, f32.
    fn norm_f32(&self, x: &Tensor, w: &GpuBuffer, out: &Tensor, t: u32) -> Result<(), String> {
        nn::rms_norm_f32(&self.rt, &x.buffer, w, &out.buffer, t, self.cfg.hidden, self.cfg.rms_norm_eps)
    }

    /// A fresh residual stream `resid + y @ w_out`, leaving `resid` as the
    /// saved input of the norm that read it.
    fn residual(&self, resid: &Tensor, y: &Tensor, w_out: &Tensor, t: u32) -> Result<Tensor, String> {
        let (rt, h) = (&self.rt, self.cfg.hidden);
        let out = tensor(rt, &[t as usize, h as usize])?;
        gemm(y, w_out, &out, BACKEND)?;
        qwen35::residual_add(rt, Cols::dense(&resid.buffer, h), Cols::dense(&out.buffer, h), t, h)?;
        Ok(out)
    }

    /// One layer's forward from `resid_in`, keeping what its backward reads;
    /// with `output`, also the residual stream out of it (a recomputation
    /// needs only the former, and skips the `down` projection's product).
    fn train_layer_forward(
        &self,
        layer: &Layer,
        resid_in: Tensor,
        t: u32,
        output: bool,
    ) -> Result<(Saved, Option<Tensor>), String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let x1 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_in, &layer.input_norm, &x1, t)?;
        let (mixer, resid_mid) = match &layer.mixer {
            Mixer::Gdn(w) => {
                let s = self.gdn_forward(w, &x1, t)?;
                let r = self.residual(&resid_in, &s.y, &w.w_out, t)?;
                (SavedMixer::Gdn(Box::new(s)), r)
            }
            Mixer::Attn(w) => {
                let s = self.attn_forward(w, &x1, t)?;
                let r = self.residual(&resid_in, &s.y, &w.w_out, t)?;
                (SavedMixer::Attn(s), r)
            }
        };
        let x2 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_mid, &layer.post_norm, &x2, t)?;
        let (m_gate, m_up, m_mid) = (tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?);
        gemm(&x2, &layer.gate, &m_gate, BACKEND)?;
        gemm(&x2, &layer.up, &m_up, BACKEND)?;
        qwen35::swiglu(
            rt,
            Cols::dense(&m_gate.buffer, cfg.intermediate),
            Cols::dense(&m_up.buffer, cfg.intermediate),
            OutCols { cols: Cols::dense(&m_mid.buffer, cfg.intermediate), dtype: DType::F32 },
            t,
            cfg.intermediate,
        )?;
        let resid_out = if output { Some(self.residual(&resid_mid, &m_mid, &layer.down, t)?) } else { None };
        Ok((Saved { resid_in, x1, mixer, resid_mid, x2, m_gate, m_up, m_mid }, resid_out))
    }

    fn gdn_forward(&self, w: &GdnWeights, x1: &Tensor, t: u32) -> Result<SavedGdn, String> {
        let (rt, g) = (&self.rt, self.cfg.gdn);
        let (tu, hv, dv) = (t as usize, g.v_heads() as usize, g.v_dim() as usize);
        let dk = GDN_TRAIN_DK as usize;
        let proj = tensor(rt, &[tu, g.width() as usize])?;
        gemm(x1, &w.w_in, &proj, BACKEND)?;
        let conv = f32s(rt, tu * g.conv_dim() as usize)?;
        qwen35::conv1d_silu(
            rt,
            Cols::dense(&proj.buffer, g.width()),
            &w.conv_w,
            self.cfg.conv_kernel,
            StateIn::Zero,
            &conv,
            None,
            1,
            t,
            g.conv_dim(),
        )?;
        let qkv = g.conv_qkv(&conv);
        let (q, k, v) = (tensor(rt, &[1, tu, hv, dk])?, tensor(rt, &[1, tu, hv, dk])?, tensor(rt, &[1, tu, hv, dv])?);
        for (off, dst, width) in [(qkv.q_off, &q, g.key_dim()), (qkv.k_off, &k, g.key_dim()), (qkv.v_off, &v, g.value_dim())] {
            copy_cols(rt, Cols { buf: &conv, ld: qkv.ld, off }, Cols::dense(&dst.buffer, width), t, width)?;
        }
        let (gt, beta) = (tensor(rt, &[1, tu, hv])?, tensor(rt, &[1, tu, hv])?);
        qwen35::gdn_gates(
            rt,
            &g.gates(&proj.buffer),
            &GdnParams { a_log: &w.a_log, dt_bias: &w.dt_bias },
            &gt.buffer,
            &beta.buffer,
            t,
            g.v_heads(),
        )?;
        let dims = GdnTrainDims { batch: 1, seq: t, heads: g.v_heads(), v_dim: g.v_dim() };
        let ckpt = tensor(rt, &dims.checkpoint_shape())?;
        let o = tensor(rt, &[1, tu, hv, dv])?;
        let inputs = GdnTrainInputs { q: &q, k: &k, v: &v, g: &gt, beta: &beta, s0: None };
        gdn_train_forward(rt, dims, inputs, &o, None, &ckpt)?;
        let y = tensor(rt, &[tu, hv * dv])?;
        qwen35::gated_rms_norm(
            rt,
            Cols::dense(&o.buffer, g.value_dim()),
            g.z(&proj.buffer),
            &w.norm_w,
            OutCols { cols: Cols::dense(&y.buffer, g.value_dim()), dtype: DType::F32 },
            t,
            g.v_heads(),
            g.v_dim(),
            self.cfg.rms_norm_eps,
        )?;
        Ok(SavedGdn { proj, q, k, v, g: gt, beta, ckpt, o, y })
    }

    fn attn_forward(&self, w: &AttnWeights, x1: &Tensor, t: u32) -> Result<SavedAttn, String> {
        let (rt, a) = (&self.rt, self.cfg.attn);
        let tu = t as usize;
        let (qd, kvd) = ((a.q_heads() * a.head_dim()) as usize, (a.kv_heads() * a.head_dim()) as usize);
        let proj = tensor(rt, &[tu, a.width() as usize])?;
        gemm(x1, &w.w_in, &proj, BACKEND)?;
        let (q, k, v) = (f32s(rt, tu * qd)?, f32s(rt, tu * kvd)?, f32s(rt, tu * kvd)?);
        qwen35::attn_qk_norm_rope(
            rt,
            &self.attn_shape(t),
            Cols::dense(&proj.buffer, a.width()),
            &w.q_norm,
            &w.k_norm,
            &AttnTargets { q_out: &q, k_cache: &k, v_cache: &v },
            0,
            self.cfg.rope_theta,
            self.cfg.rms_norm_eps,
        )?;
        let dims = self.attn_dims(t);
        let ws = AttnTrainWorkspace::new(rt, dims)?;
        let (o, lse) = (f32s(rt, tu * qd)?, f32s(rt, dims.lse_len())?);
        attn_train_forward(rt, &dims, &q, &k, &v, &o, &lse, &ws)?;
        let y = tensor(rt, &[tu, qd])?;
        qwen35::attn_output_gate(
            rt,
            &o,
            Cols::dense(&proj.buffer, a.width()),
            OutCols { cols: Cols::dense(&y.buffer, qd as u32), dtype: DType::F32 },
            t,
            a.q_heads(),
            a.head_dim(),
        )?;
        Ok(SavedAttn { proj, q, k, v, o, lse, y })
    }

    /// One layer's backward. On entry `sc.dresid` is the gradient of the
    /// layer's output; on return, of its input.
    fn train_layer_backward(&self, layer: &Layer, s: &Saved, sc: &mut Scratch) -> Result<LayerGrads, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (t, h, i) = (sc.t, cfg.hidden, cfg.intermediate);
        let (hu, iu) = (h as usize, i as usize);
        let eps = cfg.rms_norm_eps;

        // MLP: resid_out = resid_mid + swiglu(x2 @ gate, x2 @ up) @ down.
        let down = tensor(rt, &[iu, hu])?;
        gemm_tn_f32(&s.m_mid, &sc.dresid, &down, BACKEND)?;
        gemm_nt_f32(&sc.dresid, &layer.down, &sc.d_mid, BACKEND)?;
        swiglu_bwd(
            rt,
            Cols::dense(&s.m_gate.buffer, i),
            Cols::dense(&s.m_up.buffer, i),
            Cols::dense(&sc.d_mid.buffer, i),
            Cols::dense(&sc.d_gate.buffer, i),
            Cols::dense(&sc.d_up.buffer, i),
            t,
            i,
        )?;
        let (gate, up) = (tensor(rt, &[hu, iu])?, tensor(rt, &[hu, iu])?);
        gemm_tn_f32(&s.x2, &sc.d_gate, &gate, BACKEND)?;
        gemm_tn_f32(&s.x2, &sc.d_up, &up, BACKEND)?;
        gemm_nt_f32(&sc.d_gate, &layer.gate, &sc.dx, BACKEND)?;
        gemm_nt_f32(&sc.d_up, &layer.up, &sc.tmp_h, BACKEND)?;
        qwen35::residual_add(rt, Cols::dense(&sc.tmp_h.buffer, h), Cols::dense(&sc.dx.buffer, h), t, h)?;
        let post_norm = f32s(rt, hu)?;
        rms_norm_bwd(rt, &s.resid_mid.buffer, &layer.post_norm, &sc.dx.buffer, &sc.dresid.buffer, &post_norm, &sc.norm_part, t, h, eps, true)?;

        // Mixer: resid_mid = resid_in + mixer(x1), x1 = norm(resid_in).
        let mixer = match (&layer.mixer, &s.mixer) {
            (Mixer::Gdn(w), SavedMixer::Gdn(sv)) => MixerGrads::Gdn(self.gdn_backward(w, sv, &s.x1, sc)?),
            (Mixer::Attn(w), SavedMixer::Attn(sv)) => MixerGrads::Attn(self.attn_backward(w, sv, &s.x1, sc)?),
            _ => return Err("Qwen35Model::train_step: a layer's saved state is not its mixer's".into()),
        };
        let input_norm = f32s(rt, hu)?;
        rms_norm_bwd(rt, &s.resid_in.buffer, &layer.input_norm, &sc.dx.buffer, &sc.dresid.buffer, &input_norm, &sc.norm_part, t, h, eps, true)?;
        Ok(LayerGrads { input_norm, post_norm, mixer, gate, up, down })
    }

    /// The GDN mixer's backward from `sc.dresid`; leaves the gradient of its
    /// input `x1` in `sc.dx`.
    fn gdn_backward(&self, w: &GdnWeights, s: &SavedGdn, x1: &Tensor, sc: &mut Scratch) -> Result<GdnGrads, String> {
        let (rt, cfg, g) = (&self.rt, &self.cfg, self.cfg.gdn);
        let t = sc.t;
        let (hu, vd) = (cfg.hidden as usize, g.value_dim() as usize);
        let gs = sc.gdn.as_ref().ok_or("Qwen35Model::train_step: no GDN scratch")?;
        let w_out = tensor(rt, &[vd, hu])?;
        gemm_tn_f32(&s.y, &sc.dresid, &w_out, BACKEND)?;
        gemm_nt_f32(&sc.dresid, &w.w_out, &gs.dy, BACKEND)?;
        // y = gated_rms_norm(o, z): d_o, and dz into the projection gradient.
        let norm_w = f32s(rt, g.v_dim() as usize)?;
        let dproj = &gs.dproj.buffer;
        gated_rms_norm_bwd(
            rt,
            Cols::dense(&s.o.buffer, g.value_dim()),
            g.z(&s.proj.buffer),
            &w.norm_w,
            Cols::dense(&gs.dy.buffer, g.value_dim()),
            Cols::dense(&gs.d_o.buffer, g.value_dim()),
            g.z(dproj),
            &norm_w,
            &gs.norm_part,
            t,
            g.v_heads(),
            g.v_dim(),
            cfg.rms_norm_eps,
        )?;
        let dims = GdnTrainDims { batch: 1, seq: t, heads: g.v_heads(), v_dim: g.v_dim() };
        gdn_train_backward(
            rt,
            dims,
            GdnTrainInputs { q: &s.q, k: &s.k, v: &s.v, g: &s.g, beta: &s.beta, s0: None },
            &s.ckpt,
            &gs.d_o,
            None,
            &gs.ws,
            GdnTrainGrads { dq: &gs.dq, dk: &gs.dk, dv: &gs.dv, dg: &gs.dg, dbeta: &gs.dbeta, ds0: None },
        )?;
        // The gates' logits (a, b columns) and their parameters.
        let (a_log, dt_bias) = (f32s(rt, g.v_heads() as usize)?, f32s(rt, g.v_heads() as usize)?);
        gdn_gates_bwd(
            rt,
            &g.gates(&s.proj.buffer),
            &GdnParams { a_log: &w.a_log, dt_bias: &w.dt_bias },
            &gs.dg.buffer,
            &gs.dbeta.buffer,
            dproj,
            &a_log,
            &dt_bias,
            &gs.gates_part,
            t,
            g.v_heads(),
        )?;
        // q, k, v back into the conv output's layout, then the conv into the
        // projection's qkv columns.
        let qkv = g.conv_qkv(&gs.d_conv);
        for (src, off, width) in [(&gs.dq, qkv.q_off, g.key_dim()), (&gs.dk, qkv.k_off, g.key_dim()), (&gs.dv, qkv.v_off, g.value_dim())] {
            copy_cols(rt, Cols::dense(&src.buffer, width), Cols { buf: &gs.d_conv, ld: qkv.ld, off }, t, width)?;
        }
        let conv_w = f32s(rt, (g.conv_dim() * cfg.conv_kernel) as usize)?;
        conv1d_silu_bwd(
            rt,
            Cols::dense(&s.proj.buffer, g.width()),
            &w.conv_w,
            cfg.conv_kernel,
            Cols::dense(&gs.d_conv, g.conv_dim()),
            Cols::dense(dproj, g.width()),
            &conv_w,
            &gs.conv_part,
            1,
            t,
            g.conv_dim(),
        )?;
        let w_in = tensor(rt, &[hu, g.width() as usize])?;
        gemm_tn_f32(x1, &gs.dproj, &w_in, BACKEND)?;
        gemm_nt_f32(&gs.dproj, &w.w_in, &sc.dx, BACKEND)?;
        Ok(GdnGrads { w_in, w_out, conv_w, a_log, dt_bias, norm_w })
    }

    /// The attention mixer's backward from `sc.dresid`; leaves the gradient
    /// of its input `x1` in `sc.dx`.
    fn attn_backward(&self, w: &AttnWeights, s: &SavedAttn, x1: &Tensor, sc: &mut Scratch) -> Result<AttnGrads, String> {
        let (rt, cfg, a) = (&self.rt, &self.cfg, self.cfg.attn);
        let t = sc.t;
        let hu = cfg.hidden as usize;
        let qd = a.q_heads() * a.head_dim();
        let asc = sc.attn.as_ref().ok_or("Qwen35Model::train_step: no attention scratch")?;
        let w_out = tensor(rt, &[qd as usize, hu])?;
        gemm_tn_f32(&s.y, &sc.dresid, &w_out, BACKEND)?;
        gemm_nt_f32(&sc.dresid, &w.w_out, &asc.dy, BACKEND)?;
        let proj = Cols::dense(&s.proj.buffer, a.width());
        let dproj = &asc.dproj.buffer;
        // y = o * sigmoid(gate): d_o, and the gate columns of dproj.
        attn_gate_bwd(rt, &s.o, proj, Cols::dense(&asc.dy.buffer, qd), &asc.d_o, dproj, t, a.q_heads(), a.head_dim())?;
        let grads = AttnTrainGrads { dq: &asc.dq, dk: &asc.dk, dv: &asc.dv };
        attn_train_backward(rt, &self.attn_dims(t), &s.q, &s.k, &s.v, &s.o, &s.lse, &asc.d_o, &grads, &asc.ws)?;
        // q, k, v back through RoPE and the norms into their dproj columns.
        let (q_norm, k_norm) = (f32s(rt, a.head_dim() as usize)?, f32s(rt, a.head_dim() as usize)?);
        attn_qk_norm_rope_bwd(
            rt,
            &self.attn_shape(t),
            proj,
            &w.q_norm,
            &w.k_norm,
            &AttnQkvGrads { dq: &asc.dq, dk: &asc.dk, dv: &asc.dv },
            dproj,
            &q_norm,
            &k_norm,
            &asc.part,
            cfg.rope_theta,
            cfg.rms_norm_eps,
        )?;
        let w_in = tensor(rt, &[hu, a.width() as usize])?;
        gemm_tn_f32(x1, &asc.dproj, &w_in, BACKEND)?;
        gemm_nt_f32(&asc.dproj, &w.w_in, &sc.dx, BACKEND)?;
        Ok(AttnGrads { w_in, w_out, q_norm, k_norm })
    }
}

#[cfg(test)]
mod tests {
    //! Whether `train_step`'s gradients are the gradients of tessl's own
    //! forward, where they differ from transformers': central differences of
    //! the loss along the direction in which the two gradients disagree.

    use std::path::{Path, PathBuf};

    use super::*;
    use crate::npy::read_npy;
    use crate::qwen35_model::Qwen35Config;
    use crate::safetensors::SafeTensors;

    fn npy(path: &Path) -> Vec<f64> {
        let a = read_npy(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if let Ok(s) = a.f32_slice() {
            s.iter().map(|&x| f64::from(x)).collect()
        } else if let Ok(s) = a.f64_slice() {
            s.to_vec()
        } else {
            a.i64_slice().unwrap().iter().map(|&x| x as f64).collect()
        }
    }

    /// One parameter tensor to probe: the model's buffer, tessl's gradient,
    /// transformers' gradient in tessl's layout, and whether the disagreement
    /// is large enough for finite differences to resolve.
    struct FdCase<'a> {
        name: String,
        param: &'a GpuBuffer,
        grad: &'a GpuBuffer,
        n: usize,
        torch: Vec<f64>,
        resolved: bool,
        steps: [f64; 4],
    }

    /// The loss of the inference forward's logits (the training forward's
    /// to 2e-7), in f64 on the host.
    fn loss(model: &Qwen35Model, ids: &[u32]) -> f64 {
        let logits = model.forward(ids, false).unwrap().logits;
        let v = model.cfg.vocab as usize;
        let mut ce = 0.0f64;
        for (t, row) in logits.chunks(v).take(ids.len() - 1).enumerate() {
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z: f64 = row.iter().map(|&x| (f64::from(x) - m).exp()).sum();
            ce += m + z.ln() - f64::from(row[ids[t + 1] as usize]);
        }
        ce / (ids.len() - 1) as f64
    }

    /// Qwen3.5-2B-Base against `make_train_fixture.py 2b`'s gradients.
    ///
    /// tessl's and transformers' f32 forwards differ (logits by 1.9e-6 of
    /// the largest), so their gradients differ too (up to 4e-3 of a
    /// parameter's largest, tests/qwen35_train.rs). This decides whose side a
    /// disagreement is on: along `v = (g_tessl - g_torch) / |d|` the two
    /// gradients predict slopes `|d|` apart, and a Richardson-extrapolated
    /// central difference of tessl's own loss must land on tessl's.
    #[test]
    #[ignore]
    fn real_2b_gradients_are_those_of_tessls_forward() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_train_ref");
        let st = SafeTensors::open(Path::new(&std::env::var("QWEN35_2B_SAFETENSORS").expect("QWEN35_2B_SAFETENSORS"))).unwrap();
        let rt = GpuRuntime::new().unwrap();
        let model = Qwen35Model::load(&rt, &st, "model.language_model.", Qwen35Config::qwen35_2b().unwrap(), Precision::F32).unwrap();
        drop(st);
        let ids: Vec<u32> = npy(&dir.join("ids.npy")).iter().map(|&x| x as u32).collect();
        let step = model.train_step(&ids, Activations::Saved).unwrap();
        let base = loss(&model, &ids);
        // (name, the model's f32 buffer, tessl's gradient buffer, length).
        let mixer = |l: usize| match (&model.layers[l].mixer, &step.grads.layers[l].mixer) {
            (Mixer::Gdn(w), MixerGrads::Gdn(g)) => (w, g),
            _ => panic!("layer {l} is not a GDN layer"),
        };
        let h = model.cfg.hidden as usize;
        let cases: Vec<(String, &GpuBuffer, &GpuBuffer, usize)> = vec![
            ("model.layers.20.linear_attn.conv1d.weight".into(), &mixer(20).0.conv_w, &mixer(20).1.conv_w, 6144 * 4),
            ("model.layers.23.input_layernorm.weight".into(), &model.layers[23].input_norm, &step.grads.layers[23].input_norm, h),
            ("model.layers.8.linear_attn.dt_bias".into(), &mixer(8).0.dt_bias, &mixer(8).1.dt_bias, 16),
            ("model.layers.8.linear_attn.A_log".into(), &mixer(8).0.a_log, &mixer(8).1.a_log, 16),
            ("model.norm.weight".into(), &model.final_norm, &step.grads.final_norm, h),
            ("model.layers.22.input_layernorm.weight".into(), &model.layers[22].input_norm, &step.grads.layers[22].input_norm, h),
            ("model.layers.8.post_attention_layernorm.weight".into(), &model.layers[8].post_norm, &step.grads.layers[8].post_norm, h),
            ("model.layers.6.input_layernorm.weight".into(), &model.layers[6].input_norm, &step.grads.layers[6].input_norm, h),
            ("model.layers.20.linear_attn.norm.weight".into(), &mixer(20).0.norm_w, &mixer(20).1.norm_w, 128),
        ];
        // The matrices, whose disagreement is large in absolute terms and
        // spread thinly over millions of weights: torch's [out, in]
        // transposed into the packed [in, out] tessl stores.
        let i = model.cfg.intermediate as usize;
        let attn = |l: usize| match (&model.layers[l].mixer, &step.grads.layers[l].mixer) {
            (Mixer::Attn(w), MixerGrads::Attn(g)) => (w, g),
            _ => panic!("layer {l} is not an attention layer"),
        };
        let qd = (model.cfg.attn.q_heads() * model.cfg.attn.head_dim()) as usize;
        let matrices: Vec<(String, &GpuBuffer, &GpuBuffer, usize, usize)> = vec![
            ("model.layers.0.mlp.gate_proj.weight".into(), &model.layers[0].gate.buffer, &step.grads.layers[0].gate.buffer, h, i),
            ("model.layers.0.mlp.down_proj.weight".into(), &model.layers[0].down.buffer, &step.grads.layers[0].down.buffer, i, h),
            ("model.layers.3.self_attn.o_proj.weight".into(), &attn(3).0.w_out.buffer, &attn(3).1.w_out.buffer, qd, h),
        ];
        // Resolved: the disagreement is large enough in absolute terms (|d| h
        // well above the loss's f32 rounding, ~1e-6) for central differences
        // to tell the two gradients apart. The conv and the matrices are; the
        // 1-D tensors' |d| h is at the rounding level, so they are printed and
        // not asserted.
        let mut all_cases: Vec<FdCase<'_>> = cases
            .into_iter()
            .map(|(name, param, grad, n)| {
                let torch = npy(&dir.join(format!("grad.{name}.npy")));
                // The hidden-size norms act as (1 + w) around 1, so a step of
                // 0.05-0.4 along a unit direction moves each of their 2048
                // entries by at most ~0.01 and stays near-linear, while lifting
                // |d| h far above the loss's f32 rounding. The GDN gated norm's
                // 128 weights do not resolve: steps of 0.4-0.05 and 0.2-0.025
                // extrapolate 4e-5 apart, more than its |d| (3e-5), the first
                // nearer tessl's gradient and the second nearer transformers'.
                let norm = name.ends_with("layernorm.weight") || name == "model.norm.weight";
                let resolved = norm || name.ends_with("conv1d.weight");
                let steps = if norm || name.ends_with("linear_attn.norm.weight") {
                    [0.4, 0.2, 0.1, 0.05]
                } else {
                    [1e-2, 5e-3, 2.5e-3, 1.25e-3]
                };
                FdCase { name, param, grad, n, torch, resolved, steps }
            })
            .collect();
        for (name, p, g, rows, cols) in matrices {
            let t = npy(&dir.join(format!("grad.{name}.npy")));
            let packed: Vec<f64> = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c))).map(|(r, c)| t[c * rows + r]).collect();
            all_cases.push(FdCase { name, param: p, grad: g, n: rows * cols, torch: packed, resolved: true, steps: [1e-2, 5e-3, 2.5e-3, 1.25e-3] });
        }
        for FdCase { name, param, grad, n, torch: g_torch, resolved, steps } in all_cases {
            let g_tessl: Vec<f64> = grad.read_f32()[..n].iter().map(|&x| f64::from(x)).collect();
            let d: Vec<f64> = g_tessl.iter().zip(&g_torch).map(|(a, b)| a - b).collect();
            let dn = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            let v: Vec<f64> = d.iter().map(|x| x / dn).collect();
            let dot = |g: &[f64]| g.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
            let (pt, pr) = (dot(&g_tessl), dot(&g_torch));
            let orig = param.read_f32()[..n].to_vec();
            let mut line = format!("{name}: |d| {dn:.3e}, tessl.v {pt:.6e}, torch.v {pr:.6e}; FD");
            let mut fds = Vec::new();
            for step_h in steps {
                let set = |s: f64| {
                    let p: Vec<f32> = orig.iter().zip(&v).map(|(&x, &u)| (f64::from(x) + s * u) as f32).collect();
                    let mut all = param.read_f32();
                    all[..n].copy_from_slice(&p);
                    param.write_f32(&all);
                };
                set(step_h);
                let up = loss(&model, &ids);
                set(-step_h);
                let down = loss(&model, &ids);
                let mut all = param.read_f32();
                all[..n].copy_from_slice(&orig);
                param.write_f32(&all);
                let fd = (up - down) / (2.0 * step_h);
                fds.push(fd);
                line += &format!(" h={step_h:e}: {fd:.5e}");
            }
            // Richardson: the central difference's h^2 term cancels between
            // h and h/2.
            let n_fd = fds.len();
            let extrap = (4.0 * fds[n_fd - 1] - fds[n_fd - 2]) / 3.0;
            let extrap2 = (4.0 * fds[n_fd - 2] - fds[n_fd - 3]) / 3.0;
            let fd = (extrap + extrap2) / 2.0;
            line += &format!("; extrapolated {fd:.5e}: off tessl {:+.2e}, off torch {:+.2e}", fd - pt, fd - pr);
            eprintln!("{line}");
            assert_eq!(loss(&model, &ids).to_bits(), base.to_bits(), "{name}: the parameter was not restored");
            if resolved {
                assert!((fd - pt).abs() <= 0.3 * dn, "{name}: tessl's loss moves as {fd:.6e} along v, its gradient says {pt:.6e}");
                assert!((fd - pt).abs() < (fd - pr).abs(), "{name}: the finite difference is closer to transformers' gradient");
            }
        }
    }
}
