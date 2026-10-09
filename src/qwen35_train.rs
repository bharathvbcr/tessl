//! One training step of the Qwen3.5 text model: the forward with what the
//! backward needs saved, the causal-LM loss, and every parameter's gradient.
//!
//! The loss is transformers' `ForCausalLMLoss` for one unpadded sequence:
//! position `t` predicts `ids[t + 1]`, cross-entropy averaged over the
//! `T - 1` predicted positions. The step runs in f32 ([`Precision::F32`]:
//! f32 weights, activations and gradients), its GEMMs on the caller's
//! [`GemmOperands`]: exact f32, or operands rounded to bf16 with f32
//! accumulation. It uses the training kernels where the inference forward's cannot give a
//! backward what it needs: [`crate::gdn_train`] for the gated delta rule
//! (checkpointed state), [`crate::attn_train`] for attention (log-sum-exp),
//! and [`crate::qwen35::gdn_gates`] for the gates as values. Everything else
//! is the inference forward's kernels, and the backward is
//! [`crate::qwen35_bwd`], [`crate::cross_entropy`] and the GEMMs.
//!
//! Every gradient is in the layout of the weight it belongs to: the fused
//! projections' packed `[in, out]` right operands, the conv weight
//! `[channels, kernel_width]`, and `[vocab, hidden]` for the tied embedding /
//! LM head (both uses summed). The zero-centred norms are stored as `w`, and
//! their gradient is that of `w` (the same as `1 + w`'s). Every reduction runs in a fixed
//! order, so a step's gradients are the same bits on every run.
//!
//! Scope: one sequence, positions from 0. A GDN with more value heads than
//! key heads (the 4B's 32 over 16) repeats each key head's q and k across its
//! value heads before `gdn_train`, as transformers does, and sums their
//! gradients back over the group.
//! Several sequences' gradients sum in a bank through
//! [`Qwen35Model::train_step_into`], one sequence at a time, and a step can
//! score chosen positions against given tokens instead ([`Supervise::Rows`]).
//!
//! The forward keeps only the residual stream into each layer (`T x hidden`
//! f32, or bf16 on a bf16-stored model); each layer's intermediates are
//! rebuilt from it just before that layer's backward. The kernels are
//! deterministic, so on an f32 model the rebuilt intermediates are the
//! forward's bits; a bf16 model rebuilds from the rounded input (see
//! `step_forward`). Keeping every layer's
//! intermediates instead was measured at 10% faster for 22x the activation
//! memory at T = 2048 on the 2B (`docs/qwen35.md`) and removed.

use std::sync::Arc;

use crate::attn_train::{attn_train_backward, attn_train_forward, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace};
use crate::cross_entropy::{
    cross_entropy_rows, cross_entropy_rows_accumulating, gather_rows_f32, CeGrads, CeHidden, CeWorkspace, Reduction,
};
use crate::dispatch::{dispatch_1d, set_gpu_buf_offset, set_u32};
use crate::gdn_train::{
    gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace, GDN_TRAIN_DK,
};
use crate::gemm::{cast_bf16_to_f32_into, cast_f32_to_bf16_into, GemmOperands};
use crate::nn::require_runtime;
use crate::qwen35::{self, AttnShape, AttnTargets, Cols, GdnParams, OutCols, StateIn};
use crate::qwen35_bwd::{
    attn_gate_bwd, attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, check_scatter_rows, conv1d_silu_bwd,
    conv1d_silu_bwd_part_len, copy_cols, embed_rows_bwd, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len,
    gdn_gates_bwd, gdn_gates_bwd_part_len, rms_norm_bwd, rms_norm_bwd_part_len, scatter_add_rows, swiglu_bwd,
    AttnQkvGrads, EmbedBwdWorkspace,
};
use crate::qwen35_model::{AttnWeights, GdnWeights, Layer, Mixer, Precision, Qwen35Model};
use crate::runtime::{BufferKind, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};

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

/// One gradient (or weight) buffer: the buffer, the byte offset of its
/// elements, how many elements it holds, and their type (a matrix is f32 or
/// bf16; a [`GpuBuffer`] part is always f32).
type Part<'a> = (&'a GpuBuffer, usize, usize, DType);

fn tensor_part(t: &Tensor) -> Part<'_> {
    (&t.buffer, t.byte_offset(), t.numel(), t.dtype)
}

fn buf_part(b: &GpuBuffer) -> Part<'_> {
    (b, 0, b.nbytes() / 4, DType::F32)
}

/// A bank tensor for weight `t`, in `t`'s dtype: a bf16 model's matrices
/// bank their gradients in bf16.
fn zeros_tensor(t: &Tensor) -> Result<Tensor, String> {
    // Hot: a bank (accumulated gradients) stays resident. Allocations come
    // back zeroed.
    let rt = t.runtime();
    match t.dtype {
        DType::BF16 => rt.alloc_tensor_bf16_hot(t.shape()),
        _ => rt.alloc_tensor_f32_hot(t.shape()),
    }
}

fn zeros_buf(model: &Qwen35Model, b: &GpuBuffer) -> Result<GpuBuffer, String> {
    let out = model.rt.alloc_buffer_hot(b.nbytes())?;
    out.try_zero()?;
    Ok(out)
}

impl LayerGrads {
    /// Every buffer, in the order [`layer_weight_parts`] walks the weights.
    fn parts(&self) -> Vec<Part<'_>> {
        let mut v = vec![buf_part(&self.input_norm), buf_part(&self.post_norm)];
        match &self.mixer {
            MixerGrads::Gdn(g) => v.extend([
                tensor_part(&g.w_in),
                tensor_part(&g.w_out),
                buf_part(&g.conv_w),
                buf_part(&g.a_log),
                buf_part(&g.dt_bias),
                buf_part(&g.norm_w),
            ]),
            MixerGrads::Attn(g) => v.extend([
                tensor_part(&g.w_in),
                tensor_part(&g.w_out),
                buf_part(&g.q_norm),
                buf_part(&g.k_norm),
            ]),
        }
        v.extend([tensor_part(&self.gate), tensor_part(&self.up), tensor_part(&self.down)]);
        v
    }
}

/// A layer's weights, in the order [`LayerGrads::parts`] walks its gradients.
fn layer_weight_parts(layer: &Layer) -> Vec<Part<'_>> {
    let mut v = vec![buf_part(&layer.input_norm), buf_part(&layer.post_norm)];
    match &layer.mixer {
        Mixer::Gdn(w) => v.extend([
            tensor_part(&w.w_in),
            tensor_part(&w.w_out),
            buf_part(&w.conv_w),
            buf_part(&w.a_log),
            buf_part(&w.dt_bias),
            buf_part(&w.norm_w),
        ]),
        Mixer::Attn(w) => v.extend([
            tensor_part(&w.w_in),
            tensor_part(&w.w_out),
            buf_part(&w.q_norm),
            buf_part(&w.k_norm),
        ]),
    }
    v.extend([
        tensor_part(&layer.gate),
        tensor_part(&layer.up),
        tensor_part(&layer.down),
    ]);
    v
}

impl Qwen35Grads {
    /// Zeroed gradients shaped like every parameter of `model`, each in its
    /// weight's dtype (bf16 matrices on a bf16 model): a bank for
    /// [`Qwen35Model::train_step_into`], resident.
    pub fn zeros_like(model: &Qwen35Model) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let mixer = match &layer.mixer {
                Mixer::Gdn(w) => MixerGrads::Gdn(GdnGrads {
                    w_in: zeros_tensor(&w.w_in)?,
                    w_out: zeros_tensor(&w.w_out)?,
                    conv_w: zeros_buf(model, &w.conv_w)?,
                    a_log: zeros_buf(model, &w.a_log)?,
                    dt_bias: zeros_buf(model, &w.dt_bias)?,
                    norm_w: zeros_buf(model, &w.norm_w)?,
                }),
                Mixer::Attn(w) => MixerGrads::Attn(AttnGrads {
                    w_in: zeros_tensor(&w.w_in)?,
                    w_out: zeros_tensor(&w.w_out)?,
                    q_norm: zeros_buf(model, &w.q_norm)?,
                    k_norm: zeros_buf(model, &w.k_norm)?,
                }),
            };
            layers.push(LayerGrads {
                input_norm: zeros_buf(model, &layer.input_norm)?,
                post_norm: zeros_buf(model, &layer.post_norm)?,
                mixer,
                gate: zeros_tensor(&layer.gate)?,
                up: zeros_tensor(&layer.up)?,
                down: zeros_tensor(&layer.down)?,
            });
        }
        Ok(Self {
            embed: zeros_tensor(&model.embed)?,
            final_norm: zeros_buf(model, &model.final_norm)?,
            layers,
        })
    }
}

/// `dst = src`, or `dst += src` when `add`, for each pair of parts (same
/// counts, checked by the caller). `src` is f32 (a step's gradients, from
/// f32-accumulating GEMMs and kernels). Elementwise, so exact into f32: a
/// copy is the source's bits, and one add is one f32 rounding. Into a bf16
/// part, the copy or the f32 sum is rounded to nearest once.
fn deliver(rt: &Arc<GpuRuntime>, src: &[Part<'_>], dst: &[Part<'_>], add: bool) -> Result<(), String> {
    for (&(sb, so, n, st), &(db, doff, _, dt)) in src.iter().zip(dst) {
        let n32 = u32::try_from(n).map_err(|_| format!("deliver: {n} elements exceed u32"))?;
        if st != DType::F32 {
            return Err(format!("deliver: a {st:?} gradient; the step makes f32 ones"));
        }
        // Written in place by its GEMM ([`Qwen35Model::weight_grad`]).
        if dt == DType::F32 && sb.aliases(db) && so == doff {
            continue;
        }
        match dt {
            DType::F32 => {
                let p = rt.pipeline(if add { "add_inplace_f32" } else { "copy_f32" })?;
                dispatch_1d(rt, &p, n, |bnd| {
                    if add {
                        set_gpu_buf_offset(bnd, db, doff, 0);
                        set_gpu_buf_offset(bnd, sb, so, 1);
                    } else {
                        set_gpu_buf_offset(bnd, sb, so, 0);
                        set_gpu_buf_offset(bnd, db, doff, 1);
                    }
                    set_u32(bnd, n32, 2);
                })?;
            }
            DType::BF16 => {
                let p = rt.pipeline("qwen35_deliver_f32_to_bf16")?;
                dispatch_1d(rt, &p, n, |bnd| {
                    set_gpu_buf_offset(bnd, sb, so, 0);
                    set_gpu_buf_offset(bnd, db, doff, 1);
                    set_u32(bnd, n32, 2);
                    set_u32(bnd, u32::from(add), 3);
                })?;
            }
            DType::F16 => return Err("deliver: an f16 bank is not supported".into()),
        }
    }
    Ok(())
}

/// What a step scores.
#[derive(Clone, Copy, Debug)]
pub enum Supervise<'a> {
    /// transformers' causal-LM loss: position `t` predicts `ids[t + 1]`, the
    /// mean over the `T - 1` predicted positions.
    Causal,
    /// The hidden state at `positions[i]` scored against token `targets[i]`
    /// (for a next-token loss, `targets[i] = ids[positions[i] + 1]`). The step
    /// returns the sum of these cross-entropies and the gradients of
    /// `scale` times it, so a batch mean over `N` rows spread across several
    /// steps is `scale = 1 / N` in each. Positions are distinct and below `T`;
    /// none at all is allowed (the gradients are then those of whatever
    /// else the step is given, here none).
    Rows {
        positions: &'a [u32],
        targets: &'a [u32],
        scale: f32,
    },
}

/// A step between its forward and its backward
/// ([`Qwen35Model::train_forward`], [`Qwen35Model::train_backward_into`]):
/// the loss, the loss's gradient at the final norm's output, and what the
/// backward rebuilds each layer from. It holds each layer's input
/// (`T x hidden` f32 per layer) until the backward consumes it.
pub struct PendingStep {
    ids: Vec<u32>,
    t: u32,
    operands: GemmOperands,
    inputs: Vec<Tensor>,
    resid: Tensor,
    xf: Tensor,
    dxf: Tensor,
    /// The LM head's weight gradient; none when the step scores nothing in
    /// tessl, whose embedding gradient is then the gather's alone, or when
    /// the head went straight into the bank (`head_in_bank`).
    d_embed: Option<Tensor>,
    /// The cross-entropy wrote (or added) the head gradient into the bank's
    /// f32 embedding itself, which the gather then adds onto.
    head_in_bank: bool,
    loss: f64,
    /// The step's one attention workspace (a model with attention layers):
    /// every attention layer's forward, rebuild and backward use it.
    attn_ws: Option<AttnTrainWorkspace>,
    /// The model's embedding buffer: which model made this step.
    embed: GpuBuffer,
    /// The model's parameter generation when the forward began: the backward
    /// rebuilds each layer from the weights it finds, so it runs only on the
    /// weights the forward saw.
    param_generation: u64,
}

impl PendingStep {
    /// The loss the forward computed (as [`Supervise`] defines it).
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Tokens in the step's sequence.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the sequence is empty (never: a step needs a token).
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Rows `positions` of the final norm's output (transformers'
    /// `last_hidden_state`) into `out`, dense f32 `[positions.len(), hidden]`,
    /// for a loss outside tessl; its gradient comes back through
    /// [`Qwen35Model::train_backward_into`]. Positions may repeat. Waits for
    /// the GPU, so `out` is readable when this returns.
    pub fn hidden(&self, positions: &[u32], out: &Tensor) -> Result<(), String> {
        const WHAT: &str = "PendingStep::hidden";
        let (n, h) = (positions.len(), self.xf.shape()[1]);
        if out.shape() != [n, h] || out.dtype != DType::F32 {
            return Err(format!(
                "{WHAT}: out must be f32 [{n}, {h}], got {:?} {:?}",
                out.dtype,
                out.shape()
            ));
        }
        if let Some(&bad) = positions.iter().find(|&&p| p >= self.t) {
            return Err(format!("{WHAT}: position {bad} >= {} tokens", self.t));
        }
        let rt = self.xf.runtime();
        require_runtime(rt, &out.buffer, format_args!("{WHAT}: out"))?;
        if out.overlaps(&self.xf) {
            return Err(format!("{WHAT}: out overlaps the step's own storage"));
        }
        if n == 0 {
            return Ok(());
        }
        gather_rows_f32(rt, WHAT, &self.xf, positions, out)?;
        rt.synchronize()
    }
}

/// Zero one part on the GPU, in order with the work around it.
fn zero_part(rt: &Arc<GpuRuntime>, (b, off, n, dtype): Part<'_>) -> Result<(), String> {
    if dtype != DType::F32 {
        return Err(format!("zero_part: a {dtype:?} part; only f32 parts are zeroed here"));
    }
    let n32 = u32::try_from(n).map_err(|_| format!("zero_part: {n} elements exceed u32"))?;
    let p = rt.pipeline("zero_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf_offset(bnd, b, off, 0);
        set_u32(bnd, n32, 1);
    })
}

/// Whether a bank's embedding gradient takes the head's and the gather's
/// gradients in place: f32 (a GEMM accumulates into it) and a whole buffer
/// (the gather's kernel addresses the buffer from its start).
fn embed_in_place(t: &Tensor) -> bool {
    t.dtype == DType::F32 && t.byte_offset() == 0 && t.nbytes_logical() == t.buffer.nbytes()
}

/// What [`Qwen35Model::train_step`] returns.
pub struct TrainStep {
    /// Mean cross-entropy over the `T - 1` predicted positions.
    pub loss: f64,
    pub grads: Qwen35Grads,
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

/// The scratch every layer's backward reuses.
struct Scratch {
    t: u32,
    /// The residual stream's gradient.
    dresid: Tensor,
    /// A norm input's gradient before it joins `dresid`.
    dx: Tensor,
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
    /// Grouped heads: dq or dk summed back to key-head width.
    dqk: Option<GpuBuffer>,
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

/// An f32 tensor a kernel writes in full before anything reads it: every
/// one the step allocates. Not zeroed on the host
/// ([`GpuRuntime::alloc_tensor_unzeroed`]); what must start at zero is
/// zeroed on the GPU in order with the work around it ([`zero_part`]).
fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, String> {
    rt.alloc_tensor_unzeroed(shape, DType::F32)
}

/// `t * k_heads`: the rows of a `[t, k_heads, 128]` operand taken one head
/// at a time.
fn head_rows(t: u32, k_heads: u32) -> Result<u32, String> {
    t.checked_mul(k_heads)
        .ok_or_else(|| format!("Qwen35Model::train_step: {t} tokens x {k_heads} GDN key heads exceed u32"))
}

/// `dst` (`[t, k_heads * r, 128]`) = each key head of `src` (`[t, k_heads,
/// 128]`, dense) repeated over its `r` value heads, as transformers'
/// `repeat_interleave(r, dim=2)`: value head `h * r + j` is key head `h`.
fn repeat_heads(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    t: u32,
    k_heads: u32,
    r: u32,
) -> Result<(), String> {
    let (rows, dk) = (head_rows(t, k_heads)?, GDN_TRAIN_DK);
    for j in 0..r {
        copy_cols(
            rt,
            Cols {
                buf: src,
                ld: dk,
                off: 0,
            },
            Cols {
                buf: dst,
                ld: r * dk,
                off: j * dk,
            },
            rows,
            dk,
        )?;
    }
    Ok(())
}

/// The gradient of [`repeat_heads`]: `dst` (`[t, k_heads, 128]`, dense) =
/// the sum of `src` (`[t, k_heads * r, 128]`) over each key head's `r` value
/// heads, added in value-head order.
fn sum_heads(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    t: u32,
    k_heads: u32,
    r: u32,
) -> Result<(), String> {
    let (rows, dk) = (head_rows(t, k_heads)?, GDN_TRAIN_DK);
    let group = |j: u32| Cols {
        buf: src,
        ld: r * dk,
        off: j * dk,
    };
    copy_cols(
        rt,
        group(0),
        Cols {
            buf: dst,
            ld: dk,
            off: 0,
        },
        rows,
        dk,
    )?;
    for j in 1..r {
        qwen35::residual_add(
            rt,
            group(j),
            Cols {
                buf: dst,
                ld: dk,
                off: 0,
            },
            rows,
            dk,
        )?;
    }
    Ok(())
}

impl Qwen35Model {
    /// One training step on `ids` (one sequence, positions from 0): the loss
    /// transformers' `Qwen3_5ForCausalLM(input_ids=ids, labels=ids)` reports
    /// and every parameter's gradient of it.
    pub fn train_step(&self, ids: &[u32], operands: GemmOperands) -> Result<TrainStep, String> {
        let (loss, grads) = self.step(ids, operands, Supervise::Causal, None)?;
        let grads = grads.ok_or("Qwen35Model::train_step: the step returned no gradients")?;
        Ok(TrainStep { loss, grads })
    }

    /// [`Self::train_step`] with its gradients written into `bank` (from
    /// [`Qwen35Grads::zeros_like`]) instead of fresh tensors: over what `bank`
    /// holds, or added to it when `accumulate`, so the gradients of several
    /// sequences sum in place. Each layer's gradients go into the bank as
    /// soon as its backward is encoded and is then released: its buffers go
    /// back to the pool at the next GPU wait (a few layers' on Qwen3.5, every
    /// layer's on a model with no wait inside the backward), plus the 2 GB
    /// embedding's on the 2B, rather than a second copy of every gradient
    /// held to the end. The bank is checked against the model before anything
    /// runs. `sup` says what the loss scores; returns that loss.
    pub fn train_step_into(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
        bank: &Qwen35Grads,
        accumulate: bool,
    ) -> Result<f64, String> {
        self.check_bank("Qwen35Model::train_step_into", bank)?;
        Ok(self.step(ids, operands, sup, Some((bank, accumulate)))?.0)
    }

    /// `bank` has a buffer of each weight's size for every weight.
    fn check_bank(&self, what: &str, bank: &Qwen35Grads) -> Result<(), String> {
        if bank.layers.len() != self.layers.len() {
            return Err(format!(
                "{what}: the bank has {} layers, the model {}",
                bank.layers.len(),
                self.layers.len()
            ));
        }
        let top = [
            ("embed", tensor_part(&bank.embed), tensor_part(&self.embed)),
            ("final_norm", buf_part(&bank.final_norm), buf_part(&self.final_norm)),
        ];
        let dtype_ok = |d: DType| matches!(d, DType::F32 | DType::BF16);
        for (name, (b, _, n, d), (_, _, want, _)) in top {
            require_runtime(&self.rt, b, format_args!("{what}: the bank's {name}"))?;
            if n != want || b.nbytes() == 0 || !dtype_ok(d) {
                return Err(format!(
                    "{what}: the bank's {name} holds {n} {d:?} values, the weight {want} (f32 or bf16)"
                ));
            }
        }
        for (i, (g, layer)) in bank.layers.iter().zip(&self.layers).enumerate() {
            let (have, want) = (g.parts(), layer_weight_parts(layer));
            if have.len() != want.len() {
                return Err(format!("{what}: layer {i}'s gradients are not its mixer's"));
            }
            for (k, ((b, _, n, d), (_, _, w, _))) in have.iter().zip(&want).enumerate() {
                require_runtime(&self.rt, b, format_args!("{what}: layer {i} buffer {k}"))?;
                if n != w || !dtype_ok(*d) {
                    return Err(format!(
                        "{what}: layer {i} buffer {k} holds {n} {d:?} values, its weight {w} (f32 or bf16)"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The step: into fresh gradients (returned) or into `bank`.
    fn step(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
        bank: Option<(&Qwen35Grads, bool)>,
    ) -> Result<(f64, Option<Qwen35Grads>), String> {
        let head = bank
            .filter(|(b, _)| embed_in_place(&b.embed))
            .map(|(b, add)| (&b.embed, add));
        let p = self.step_forward(ids, operands, sup, bank.is_none(), head)?;
        let loss = p.loss;
        Ok((loss, self.backward(p, None, bank)?))
    }

    /// A step's forward and its loss (`sup`), kept for
    /// [`Self::train_backward_into`]. In between, [`PendingStep::hidden`]
    /// gives the final norm's output at chosen positions to a loss outside
    /// tessl, whose gradient the backward adds to the step's own.
    ///
    /// Refused before any GPU work when the device's current allocation
    /// plus [`Self::train_step_bytes`] exceeds its recommended working set:
    /// past that, Metal pages the resident set and the step's command
    /// buffers time out (poisoning the runtime), or the system runs out of
    /// memory, rather than failing where the caller can see it.
    pub fn train_forward(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
    ) -> Result<PendingStep, String> {
        self.step_forward(ids, operands, sup, false, None)
    }

    /// [`Self::train_forward`]; `fresh` when the backward will return fresh
    /// gradients rather than deliver them into a bank, which the pre-flight
    /// then counts too. `head` is the bank's f32 embedding (and whether to
    /// add into it) when the head gradient goes straight there: no
    /// `[vocab, hidden]` tensor and no copy of it (2 GB on the 2B).
    fn step_forward(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
        fresh: bool,
        head: Option<(&Tensor, bool)>,
    ) -> Result<PendingStep, String> {
        const WHAT: &str = "Qwen35Model::train_step";
        // Read before any weight is: a write that lands during the forward
        // then invalidates the step too.
        let param_generation = self.param_generation();
        let (rt, cfg) = (&self.rt, &self.cfg);
        self.require_trainable(WHAT)?;
        if self.precision == Precision::Bf16 && operands != GemmOperands::Bf16 {
            return Err(format!(
                "{WHAT}: a model stored in bf16 trains on GemmOperands::Bf16 (its weights are the bf16 operands)"
            ));
        }
        if rt.relaxed_precision() {
            return Err(format!(
                "{WHAT}: the step needs exact-f32 GEMMs; switch the runtime's relaxed precision off"
            ));
        }
        match sup {
            Supervise::Causal if ids.len() < 2 => {
                return Err(format!("{WHAT}: a step needs at least two tokens (one prediction)"));
            }
            Supervise::Rows { .. } if ids.is_empty() => return Err(format!("{WHAT}: no tokens")),
            _ => {}
        }
        if let Some(&bad) = ids.iter().find(|&&id| id >= cfg.vocab) {
            return Err(format!("{WHAT}: token id {bad} >= vocab {}", cfg.vocab));
        }
        if let Supervise::Rows {
            positions,
            targets,
            scale,
        } = sup
        {
            if positions.len() != targets.len() {
                return Err(format!(
                    "{WHAT}: {} positions but {} targets",
                    positions.len(),
                    targets.len()
                ));
            }
            if let Some(&bad) = targets.iter().find(|&&id| id >= cfg.vocab) {
                return Err(format!("{WHAT}: target {bad} >= vocab {}", cfg.vocab));
            }
            let mut seen = vec![false; ids.len()];
            for &p in positions {
                let slot = seen
                    .get_mut(p as usize)
                    .ok_or_else(|| format!("{WHAT}: position {p} >= {} tokens", ids.len()))?;
                if std::mem::replace(slot, true) {
                    return Err(format!("{WHAT}: position {p} is supervised twice"));
                }
            }
            if !scale.is_finite() {
                return Err(format!("{WHAT}: scale {scale} must be finite"));
            }
        }
        let t = u32::try_from(ids.len()).map_err(|_| format!("{WHAT}: too many tokens"))?;
        let (tu, h) = (t as usize, cfg.hidden as usize);

        // ---- forward --------------------------------------------------------
        let id_buf = rt.alloc_buffer(tu * std::mem::size_of::<u32>())?;
        // A waited commit: what earlier work freed is recycled, so the
        // allocation read next is what the step really starts from.
        id_buf.try_write_u32(ids)?;
        let (have, need) = (
            rt.current_allocated_bytes(),
            self.step_bytes(t, operands, fresh, head.is_none()),
        );
        let limit = rt.memory_info().recommended_working_set;
        if have.saturating_add(need) > limit {
            return Err(format!(
                "{WHAT}: a step on {t} tokens may allocate {need} B on top of the {have} B allocated, \
                 over the device's recommended working set of {limit} B; refused before any GPU work \
                 (shorten the sequence, or free device memory)"
            ));
        }
        let mut resid = tensor(rt, &[tu, h])?;
        qwen35::embed_rows(
            rt,
            &id_buf,
            t,
            qwen35::LmHead {
                weight: &self.embed.buffer,
                dtype: self.embed.dtype,
                vocab: cfg.vocab,
            },
            cfg.hidden,
            &resid.buffer,
        )?;
        // Each layer's input; everything else its forward made goes back to
        // the pool for the next layer. A bf16 model keeps a bf16 copy of each
        // input and runs the forward on the f32 stream: the backward rebuilds
        // each layer from its rounded input, so a layer's gradient is taken
        // up to 2^-9 relative from where its forward ran, and that error
        // stays local. Running the forward on the rounded stream instead (the
        // rebuild then the forward's bits) compounds the rounding through
        // every layer above: on the 2B (`real_2b_step_on_bf16_storage_stays_near_the_f32_step`)
        // it put layer 0's `dt_bias` gradient 1.1e-1 from the f32 step's,
        // against 2.7e-2 this way.
        let mut inputs = Vec::with_capacity(self.layers.len());
        let attn_ws = self.attn_workspace(t)?;
        for layer in &self.layers {
            let kept = if self.precision == Precision::Bf16 {
                let b = rt.alloc_tensor_unzeroed(&[tu, h], DType::BF16)?;
                cast_f32_to_bf16_into(&resid, &b)?;
                Some(b)
            } else {
                None
            };
            let (s, out) = self.train_layer_forward(layer, resid, t, true, operands, attn_ws.as_ref())?;
            let out = out.ok_or("Qwen35Model::train_step: a layer's forward returned no output")?;
            inputs.push(kept.unwrap_or(s.resid_in));
            resid = out;
        }
        let xf = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid, &self.final_norm, &xf, t)?;

        // ---- loss and the LM head ------------------------------------------
        let scores = !matches!(sup, Supervise::Rows { positions: [], .. });
        let d_embed = if scores && head.is_none() {
            Some(tensor(rt, &[cfg.vocab as usize, h])?)
        } else {
            None
        };
        let dxf = tensor(rt, &[tu, h])?;
        // Positions nothing scores keep a zero gradient row. Zeroed on the
        // GPU: a host zero would wait for the whole forward first.
        zero_part(rt, tensor_part(&dxf))?;
        let ce = |rows: &[u32], targets: &[u32], reduction, dh: &Tensor, scale| {
            let ws = CeWorkspace::new(
                rt,
                rows.len() as u32,
                cfg.hidden,
                CE_CHUNK.min(cfg.vocab),
                self.embed.dtype,
            )?;
            let hidden = CeHidden { rows: &xf, off: 0 };
            let (dw, add) = match (head, &d_embed) {
                (Some((bank, add)), _) => (bank, add),
                (None, Some(d)) => (d, false),
                (None, None) => return Err("a scored step has a head gradient".to_string()),
            };
            let grads = CeGrads { dh, dw, scale };
            if add {
                cross_entropy_rows_accumulating(rt, hidden, &self.embed, rows, targets, reduction, operands, &ws, grads)
            } else {
                cross_entropy_rows(
                    rt,
                    hidden,
                    &self.embed,
                    rows,
                    targets,
                    reduction,
                    operands,
                    &ws,
                    Some(grads),
                )
            }
        };
        let loss = match sup {
            Supervise::Causal => {
                // Rows 0..n are the first n rows of dxf: the gradient lands in place.
                let n = tu - 1;
                let rows: Vec<u32> = (0..n as u32).collect();
                ce(&rows, &ids[1..], Reduction::Mean, &dxf.view(&[n, h], 0), 1.0)?.loss
            }
            // No loss: the head adds nothing.
            Supervise::Rows { positions: [], .. } => 0.0,
            Supervise::Rows {
                positions,
                targets,
                scale,
            } => {
                let dh = tensor(rt, &[positions.len(), h])?;
                let out = ce(positions, targets, Reduction::Sum, &dh, scale)?;
                scatter_add_rows(rt, &dh, positions, &dxf)?;
                out.loss
            }
        };
        Ok(PendingStep {
            ids: ids.to_vec(),
            t,
            operands,
            inputs,
            resid,
            xf,
            dxf,
            d_embed,
            head_in_bank: scores && head.is_some(),
            loss,
            attn_ws,
            embed: self.embed.buffer.clone(),
            param_generation,
        })
    }

    /// The backward of [`Self::train_forward`]'s `p` into `bank` (over it, or
    /// added to it when `accumulate`, as [`Self::train_step_into`]). `dh`
    /// adds the gradient of a loss outside tessl at the final norm's output:
    /// rows `positions` (distinct) of a dense f32 `[positions.len(), hidden]`
    /// tensor, such as torch's for what [`PendingStep::hidden`] gave it. The
    /// step's own loss gradient is already in `p` ([`Supervise::Rows`] with
    /// no positions has none). Everything is checked before anything runs.
    /// The parameters must be as they were at the forward: each layer is
    /// rebuilt from its input with the weights it finds, so a step whose
    /// forward preceded [`Self::write_parameters`] or [`Self::adamw_step`] is
    /// refused. Tensors from another runtime are refused.
    pub fn train_backward_into(
        &self,
        p: PendingStep,
        dh: Option<(&[u32], &Tensor)>,
        bank: &Qwen35Grads,
        accumulate: bool,
    ) -> Result<(), String> {
        const WHAT: &str = "Qwen35Model::train_backward_into";
        self.check_pending(WHAT, &p, dh)?;
        self.check_bank(WHAT, bank)?;
        self.backward(p, dh, Some((bank, accumulate))).map(|_| ())
    }

    /// `p` is this model's, and `dh` fits it.
    pub(crate) fn check_pending(
        &self,
        what: &str,
        p: &PendingStep,
        dh: Option<(&[u32], &Tensor)>,
    ) -> Result<(), String> {
        if !p.embed.aliases(&self.embed.buffer) || p.inputs.len() != self.layers.len() {
            return Err(format!("{what}: the pending step is another model's"));
        }
        if p.param_generation != self.param_generation() {
            return Err(format!(
                "{what}: the model's parameters were written after the step's forward \
                 (write_parameters or adamw_step); its gradients would be of weights the \
                 forward never ran. Run the forward again"
            ));
        }
        if let Some((pos, g)) = dh {
            require_runtime(&self.rt, &g.buffer, format_args!("{what}: dh"))?;
            check_scatter_rows(what, g, pos, p.t as usize, self.cfg.hidden as usize)?;
            if g.overlaps(&p.dxf) {
                return Err(format!("{what}: dh overlaps the step's own storage"));
            }
        }
        Ok(())
    }

    /// The backward of `p`, with `dh` added at the final norm's output
    /// first: into fresh gradients (returned) or into `bank`.
    fn backward(
        &self,
        p: PendingStep,
        dh: Option<(&[u32], &Tensor)>,
        bank: Option<(&Qwen35Grads, bool)>,
    ) -> Result<Option<Qwen35Grads>, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let PendingStep {
            ids,
            t,
            operands,
            mut inputs,
            resid,
            dxf,
            d_embed,
            head_in_bank,
            attn_ws,
            ..
        } = p;
        let h = cfg.hidden as usize;
        if let Some((pos, g)) = dh {
            scatter_add_rows(rt, g, pos, &dxf)?;
        }

        // ---- backward -------------------------------------------------------
        let mut sc = self.scratch(t, attn_ws)?;
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
        if let Some((b, add)) = bank {
            deliver(rt, &[buf_part(&final_norm)], &[buf_part(&b.final_norm)], add)?;
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        // Popped from the back: a layer's rebuilt intermediates are released
        // as soon as its backward is encoded, and its gradients too when they
        // go into a bank.
        for (li, layer) in self.layers.iter().enumerate().rev() {
            let kept = inputs
                .pop()
                .ok_or("Qwen35Model::train_step: fewer layer inputs than layers")?;
            let resid_in = if kept.dtype == DType::BF16 {
                let f = tensor(rt, kept.shape())?;
                cast_bf16_to_f32_into(&kept, &f)?;
                f
            } else {
                kept
            };
            let ws = sc.attn.as_ref().map(|a| &a.ws);
            let s = self.train_layer_forward(layer, resid_in, t, false, operands, ws)?.0;
            let into = bank.map(|(b, add)| (&b.layers[li], add));
            let g = self.train_layer_backward(layer, &s, &mut sc, operands, into)?;
            match bank {
                Some((b, add)) => deliver(rt, &g.parts(), &b.layers[li].parts(), add)?,
                None => layers.push(g),
            }
        }
        layers.reverse();
        // The gather's gradient adds onto the head's. Into an f32 bank whose
        // embedding is a whole buffer it adds straight into the bank: onto
        // the head gradient the cross-entropy already wrote there, or, with
        // no head gradient, onto the bank zeroed first unless accumulating.
        // No 2 GB tensor and no copy on the 2B. A bf16 bank takes it through
        // an f32 tensor, rounded once on delivery.
        let (dw, direct) = match (d_embed, bank) {
            (Some(d), _) => (d, false),
            (None, Some((b, add))) if embed_in_place(&b.embed) => {
                if !add && !head_in_bank {
                    zero_part(rt, tensor_part(&b.embed))?;
                }
                (b.embed.clone(), true)
            }
            (None, _) => {
                let d = tensor(rt, &[cfg.vocab as usize, h])?;
                zero_part(rt, tensor_part(&d))?;
                (d, false)
            }
        };
        let emb_ws = EmbedBwdWorkspace::new(rt, t)?;
        embed_rows_bwd(rt, &ids, &sc.dresid.buffer, &dw.buffer, cfg.vocab, cfg.hidden, &emb_ws)?;
        if let Some((b, add)) = bank {
            if !direct {
                deliver(rt, &[tensor_part(&dw)], &[tensor_part(&b.embed)], add)?;
            }
            rt.synchronize()?;
            return Ok(None);
        }
        rt.synchronize()?;
        Ok(Some(Qwen35Grads {
            embed: dw,
            final_norm,
            layers,
        }))
    }

    /// The step's attention workspace: one per step, made before the
    /// forward encodes anything (none for a model without attention).
    fn attn_workspace(&self, t: u32) -> Result<Option<AttnTrainWorkspace>, String> {
        if self.cfg.layers.contains(&crate::qwen35_model::LayerKind::FullAttention) {
            Ok(Some(AttnTrainWorkspace::new(&self.rt, self.attn_dims(t))?))
        } else {
            Ok(None)
        }
    }

    /// The backward's scratch, holding the step's attention workspace.
    fn scratch(&self, t: u32, attn_ws: Option<AttnTrainWorkspace>) -> Result<Scratch, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let (g, a) = (cfg.gdn, cfg.attn);
        let has = |k: crate::qwen35_model::LayerKind| cfg.layers.contains(&k);
        let gdn = if has(crate::qwen35_model::LayerKind::LinearAttention) {
            let (hv, dv) = (g.v_heads() as usize, g.v_dim() as usize);
            let dims = GdnTrainDims {
                batch: 1,
                seq: t,
                heads: g.v_heads(),
                v_dim: g.v_dim(),
            };
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
                dqk: if g.v_heads() > g.k_heads() {
                    Some(f32s(rt, tu * g.key_dim() as usize)?)
                } else {
                    None
                },
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
                ws: attn_ws.ok_or("Qwen35Model::train_step: the step has no attention workspace")?,
                part: f32s(rt, attn_qk_norm_rope_bwd_part_len(&self.attn_shape(t)))?,
            })
        } else {
            None
        };
        Ok(Scratch {
            t,
            dresid: tensor(rt, &[tu, h])?,
            dx: tensor(rt, &[tu, h])?,
            d_mid: tensor(rt, &[tu, i])?,
            d_gate: tensor(rt, &[tu, i])?,
            d_up: tensor(rt, &[tu, i])?,
            norm_part: f32s(rt, rms_norm_bwd_part_len(t, cfg.hidden))?,
            gdn,
            attn,
        })
    }

    /// A bound on the device bytes one step on `t` tokens allocates on top
    /// of what is allocated when it starts: a step into a bank
    /// ([`Self::train_step_into`], or [`Self::train_forward`] then
    /// [`Self::train_backward_into`]) that scores its rows, so it holds the
    /// `[vocab, hidden]` head gradient. Every buffer counts at the size the
    /// pool makes it ([`GpuRuntime::allocated_bytes_for`]).
    ///
    /// A freed buffer stays allocated until the next waited commit. Without
    /// async encode (the default) every dispatch is one, so a layer's
    /// buffers are recycled once the next layer's first kernel has run;
    /// with it ([`GpuRuntime::set_async_encode`]) a step waits only at each
    /// attention layer (a deliberate wait, forward and rebuild), at the
    /// cross-entropy's loss, and at its start and end. So the bound is what the step holds
    /// throughout (each layer's input, the final norm's output and its
    /// gradient, the head gradient, the backward's scratch), plus the most
    /// it allocates between two waits (two adjacent layers' intermediates,
    /// gradients and GEMM temporaries, or under async encode every layer's
    /// from one attention layer to the next, both counted whole; or the
    /// cross-entropy's workspace and GEMM temporaries), plus the freelist's
    /// cap (`pool_cache_cap`), which is how far recycled buffers can grow
    /// the pool past the step's own. It reads the runtime's encode mode and
    /// cap when called. [`Self::train_forward`] refuses a step whose bound
    /// and the device's current allocation exceed its recommended working
    /// set.
    pub fn train_step_bytes(&self, t: u32, operands: GemmOperands) -> u64 {
        self.step_bytes(t, operands, false, true)
    }

    /// [`Self::train_step_bytes`]; `fresh` adds every layer's gradients held
    /// to the end, as [`Self::train_step`] returns them instead of
    /// delivering each into a bank, and `head` the `[vocab, hidden]` head
    /// gradient, which [`Self::train_step_into`] writes straight into an f32
    /// bank instead.
    fn step_bytes(&self, t: u32, mm: GemmOperands, fresh: bool, head: bool) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let (h, v) = (cfg.hidden as usize, cfg.vocab as usize);
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold);
        let sum = |parts: &[u64]| parts.iter().fold(0u64, |a, &b| a.saturating_add(b));
        let th = f(tu * h);
        let layers = self.layers.len() as u64;
        // Each layer's kept input: bf16 on a bf16 model, which the backward
        // widens into an f32 tensor for the layer it rebuilds.
        let bf16 = self.precision == Precision::Bf16;
        let (kept, widened) = if bf16 {
            (GpuRuntime::allocated_bytes_for(tu * h * 2, BufferKind::Cold), th)
        } else {
            (th, 0)
        };
        // The ids, each layer's input and the stream out of the last, the
        // final norm's output and its gradient, the head gradient, and the
        // step's attention workspace.
        let attn_ws = if cfg.layers.contains(&crate::qwen35_model::LayerKind::FullAttention) {
            AttnTrainWorkspace::allocated_bytes_for(self.attn_dims(t))
        } else {
            0
        };
        let held = sum(&[
            f(tu),
            kept.saturating_mul(layers),
            th.saturating_mul(3),
            if head { f(v * h) } else { 0 },
            attn_ws,
        ]);
        let fwd: Vec<u64> = self
            .layers
            .iter()
            .map(|l| self.layer_fwd_bytes(l, t, mm, true))
            .collect();
        let bwd: Vec<(u64, u64)> = self
            .layers
            .iter()
            .map(|l| {
                let (grads, temps) = self.layer_bwd_bytes(l, t, mm);
                (
                    grads,
                    sum(&[self.layer_fwd_bytes(l, t, mm, false), grads, temps, widened]),
                )
            })
            .collect();
        let fresh_grads = if fresh {
            bwd.iter().fold(f(h), |a, &(g, _)| a.saturating_add(g))
        } else {
            0
        };
        let between_waits = |bytes: &mut dyn Iterator<Item = (bool, u64)>| {
            let (mut most, mut run) = (0u64, 0u64);
            for (waits, b) in bytes {
                run = run.saturating_add(b);
                if waits {
                    most = most.max(run);
                    run = b;
                }
            }
            most.max(run)
        };
        // Without async encode every dispatch is a waited commit, so a
        // layer's buffers are recycled once the next layer's first kernel
        // runs; with it, only at an attention layer's deliberate wait.
        let batched = self.rt.async_encode_enabled();
        let is_attn = |l: &Layer| !batched || matches!(l.mixer, Mixer::Attn(_));
        let fwd_most = between_waits(&mut self.layers.iter().map(is_attn).zip(fwd.iter().copied()));
        let bwd_most = between_waits(
            &mut self
                .layers
                .iter()
                .rev()
                .map(is_attn)
                .zip(bwd.iter().rev().map(|&(_, b)| b)),
        );
        let ce = self.ce_bytes(t, mm, batched);
        let backward_held = sum(&[
            self.scratch_bytes(t),
            f(h),
            EmbedBwdWorkspace::allocated_bytes_for(t),
            fresh_grads,
        ]);
        let pool = self.rt.memory_info().pool_cache_cap as u64;
        sum(&[held, fwd_most.max(ce).max(backward_held.saturating_add(bwd_most)), pool])
    }

    /// One layer's forward intermediates, with its GEMMs' temporaries;
    /// `output` adds the `down` projection's product (its stream out is the
    /// next layer's input, which [`Self::step_bytes`] holds).
    fn layer_fwd_bytes(&self, layer: &Layer, t: u32, mm: GemmOperands, output: bool) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
        let wd = self.embed.dtype;
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold);
        let mut b = vec![
            f(tu * h), // x1
            f(tu * h), // resid_mid
            f(tu * h), // x2
            f(tu * i),
            f(tu * i),
            f(tu * i), // m_gate, m_up, m_mid
            2 * mm.nn_scratch_bytes(tu, i, h, wd),
        ];
        if output {
            b.push(mm.nn_scratch_bytes(tu, h, i, wd));
        }
        match &layer.mixer {
            Mixer::Gdn(_) => {
                let g = cfg.gdn;
                let (hv, dv, dk) = (g.v_heads() as usize, g.v_dim() as usize, GDN_TRAIN_DK as usize);
                let ckpt = GdnTrainDims {
                    batch: 1,
                    seq: t,
                    heads: g.v_heads(),
                    v_dim: g.v_dim(),
                }
                .checkpoint_shape();
                b.extend([
                    f(tu * g.width() as usize), // proj
                    mm.nn_scratch_bytes(tu, g.width() as usize, h, wd),
                    f(tu * g.conv_dim() as usize),
                    f(tu * hv * dk),
                    f(tu * hv * dk),
                    f(tu * hv * dv), // q, k, v
                    f(tu * hv),
                    f(tu * hv), // g, beta
                    f(ckpt.iter().product()),
                    f(tu * hv * dv), // o
                    f(tu * hv * dv), // y
                    mm.nn_scratch_bytes(tu, h, hv * dv, wd),
                ]);
                if g.v_heads() > g.k_heads() {
                    b.push(f(tu * g.key_dim() as usize)); // q, k staged at key-head width
                }
            }
            Mixer::Attn(_) => {
                let a = cfg.attn;
                let (qd, kvd) = (
                    (a.q_heads() * a.head_dim()) as usize,
                    (a.kv_heads() * a.head_dim()) as usize,
                );
                let dims = self.attn_dims(t);
                b.extend([
                    f(tu * a.width() as usize), // proj
                    mm.nn_scratch_bytes(tu, a.width() as usize, h, wd),
                    f(tu * qd),
                    f(tu * kvd),
                    f(tu * kvd),       // q, k, v
                    f(tu * qd),        // o
                    f(dims.lse_len()), // lse
                    f(tu * qd),        // y
                    mm.nn_scratch_bytes(tu, h, qd, wd),
                ]);
            }
        }
        b.iter().fold(0, |a, &x| a.saturating_add(x))
    }

    /// One layer's backward: its gradients, and its GEMMs' temporaries.
    fn layer_bwd_bytes(&self, layer: &Layer, t: u32, mm: GemmOperands) -> (u64, u64) {
        let (cfg, tu) = (&self.cfg, t as usize);
        let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
        let tc = self.rt.has_tensorops();
        let wd = self.embed.dtype;
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold);
        // down, gate, up, post_norm, input_norm.
        let mut grads = vec![f(i * h), f(h * i), f(h * i), f(h), f(h)];
        let mut temps = vec![
            mm.tn_scratch_bytes(tc, i, h, tu),
            mm.nt_scratch_bytes(tc, tu, i, h, wd),
            2 * mm.tn_scratch_bytes(tc, h, i, tu),
            2 * mm.nt_scratch_bytes(tc, tu, h, i, wd),
        ];
        let (width, y_cols) = match &layer.mixer {
            Mixer::Gdn(_) => {
                let g = cfg.gdn;
                let (hv, vd) = (g.v_heads() as usize, g.value_dim() as usize);
                grads.extend([
                    f(g.v_dim() as usize), // norm_w
                    f(hv),
                    f(hv), // a_log, dt_bias
                    f((g.conv_dim() * cfg.conv_kernel) as usize),
                ]);
                (g.width() as usize, vd)
            }
            Mixer::Attn(_) => {
                let a = cfg.attn;
                grads.extend([f(a.head_dim() as usize), f(a.head_dim() as usize)]);
                (a.width() as usize, (a.q_heads() * a.head_dim()) as usize)
            }
        };
        // w_out [y_cols, h] and w_in [h, width], and their GEMMs.
        grads.extend([f(y_cols * h), f(h * width)]);
        temps.extend([
            mm.tn_scratch_bytes(tc, y_cols, h, tu),
            mm.nt_scratch_bytes(tc, tu, y_cols, h, wd),
            mm.tn_scratch_bytes(tc, h, width, tu),
            mm.nt_scratch_bytes(tc, tu, h, width, wd),
        ]);
        let total = |v: &[u64]| v.iter().fold(0u64, |a, &b| a.saturating_add(b));
        (total(&grads), total(&temps))
    }

    /// The backward's scratch ([`Self::scratch`]).
    fn scratch_bytes(&self, t: u32) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.max(1).saturating_mul(4), BufferKind::Cold);
        let has = |k: crate::qwen35_model::LayerKind| cfg.layers.contains(&k);
        let mut b = vec![
            2 * f(tu * h), // dresid, dx
            3 * f(tu * i), // d_mid, d_gate, d_up
            f(rms_norm_bwd_part_len(t, cfg.hidden)),
        ];
        if has(crate::qwen35_model::LayerKind::LinearAttention) {
            let g = cfg.gdn;
            let (hv, dv, dk) = (g.v_heads() as usize, g.v_dim() as usize, GDN_TRAIN_DK as usize);
            b.extend([
                2 * f(tu * hv * dv), // dy, d_o
                f(tu * g.width() as usize),
                f(tu * g.conv_dim() as usize),
                2 * f(tu * hv * dk), // dq, dk
                f(tu * hv * dv),     // dv
                2 * f(tu * hv),      // dg, dbeta
                GdnTrainWorkspace::allocated_bytes_for(GdnTrainDims {
                    batch: 1,
                    seq: t,
                    heads: g.v_heads(),
                    v_dim: g.v_dim(),
                }),
                f(gated_rms_norm_bwd_part_len(t, g.v_heads(), g.v_dim())),
                f(gdn_gates_bwd_part_len(t, g.v_heads())),
                f(conv1d_silu_bwd_part_len(1, t, g.conv_dim(), cfg.conv_kernel)),
            ]);
            if g.v_heads() > g.k_heads() {
                b.push(f(tu * g.key_dim() as usize)); // dqk
            }
        }
        if has(crate::qwen35_model::LayerKind::FullAttention) {
            let a = cfg.attn;
            let (qd, kvd) = (
                (a.q_heads() * a.head_dim()) as usize,
                (a.kv_heads() * a.head_dim()) as usize,
            );
            b.extend([
                3 * f(tu * qd),             // dy, d_o, dq
                f(tu * a.width() as usize), // dproj
                2 * f(tu * kvd),            // dk, dv
                f(attn_qk_norm_rope_bwd_part_len(&self.attn_shape(t))),
            ]);
        }
        b.iter().fold(0, |a, &x| a.saturating_add(x))
    }

    /// The cross-entropy over `t` rows: its workspace, its GEMMs'
    /// temporaries, and a [`Supervise::Rows`] step's `dh`. `batched` (async
    /// encode) holds every vocabulary chunk's temporaries to the one wait
    /// at its end; otherwise each dispatch waits, and one chunk's four
    /// GEMMs bound them.
    fn ce_bytes(&self, t: u32, mm: GemmOperands, batched: bool) -> u64 {
        let cfg = &self.cfg;
        let (n, h) = (t as usize, cfg.hidden as usize);
        let chunk = CE_CHUNK.min(cfg.vocab);
        let tc = self.rt.has_tensorops();
        let per_chunk = |w: usize| {
            [
                2 * mm.nt_scratch_bytes(tc, n, w, h, DType::F32),
                mm.nn_scratch_bytes(n, h, w, DType::F32),
                mm.tn_scratch_bytes(tc, w, h, n),
            ]
            .iter()
            .fold(0u64, |a, &b| a.saturating_add(b))
        };
        let (full, rest) = ((cfg.vocab / chunk) as u64, (cfg.vocab % chunk) as usize);
        let gemms = if batched {
            per_chunk(chunk as usize)
                .saturating_mul(full)
                .saturating_add(if rest > 0 { per_chunk(rest) } else { 0 })
        } else {
            per_chunk(chunk as usize)
        };
        [
            CeWorkspace::allocated_bytes_for(t, cfg.hidden, chunk, self.embed.dtype),
            gemms,
            GpuRuntime::allocated_bytes_for(n * h * 4, BufferKind::Cold),
        ]
        .iter()
        .fold(0, |a, &b| a.saturating_add(b))
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

    /// `out = rms_norm(x) * (1 + w)`, f32.
    fn norm_f32(&self, x: &Tensor, w: &GpuBuffer, out: &Tensor, t: u32) -> Result<(), String> {
        qwen35::rms_norm(
            &self.rt,
            &x.buffer,
            w,
            &out.buffer,
            DType::F32,
            t,
            self.cfg.hidden,
            self.cfg.rms_norm_eps,
        )
    }

    /// A fresh residual stream `resid + y @ w_out`, leaving `resid` as the
    /// saved input of the norm that read it: `resid` copied out, then the
    /// product added in its GEMM (no product tensor, no read-modify-write
    /// pass over the stream).
    fn residual(&self, resid: &Tensor, y: &Tensor, w_out: &Tensor, t: u32, mm: GemmOperands) -> Result<Tensor, String> {
        let (rt, h) = (&self.rt, self.cfg.hidden);
        let out = tensor(rt, &[t as usize, h as usize])?;
        deliver(rt, &[tensor_part(resid)], &[tensor_part(&out)], false)?;
        mm.nn_acc(y, w_out, &out)?;
        Ok(out)
    }

    /// One layer's forward from `resid_in`, keeping what its backward reads;
    /// with `output`, also the residual stream out of it (a recomputation
    /// needs only the former, and skips the `down` projection's product).
    /// `attn_ws` is the step's attention workspace (an attention layer needs it).
    /// The layer order is inference's (`Qwen35Model::layer`) on the training
    /// kernels; `tests/qwen35_train.rs`
    /// (`train_forward_hidden_states_are_the_inference_forwards`) pins the
    /// two to the same hidden states.
    fn train_layer_forward(
        &self,
        layer: &Layer,
        resid_in: Tensor,
        t: u32,
        output: bool,
        mm: GemmOperands,
        attn_ws: Option<&AttnTrainWorkspace>,
    ) -> Result<(Saved, Option<Tensor>), String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let x1 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_in, &layer.input_norm, &x1, t)?;
        let (mixer, resid_mid) = match &layer.mixer {
            Mixer::Gdn(w) => {
                let s = self.gdn_forward(w, &x1, t, mm)?;
                let r = self.residual(&resid_in, &s.y, &w.w_out, t, mm)?;
                (SavedMixer::Gdn(Box::new(s)), r)
            }
            Mixer::Attn(w) => {
                let ws = attn_ws.ok_or("Qwen35Model::train_step: an attention layer without the step's workspace")?;
                let s = self.attn_forward(w, &x1, t, mm, ws)?;
                let r = self.residual(&resid_in, &s.y, &w.w_out, t, mm)?;
                (SavedMixer::Attn(s), r)
            }
        };
        let x2 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_mid, &layer.post_norm, &x2, t)?;
        let (m_gate, m_up, m_mid) = (tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?);
        mm.nn(&x2, &layer.gate, &m_gate)?;
        mm.nn(&x2, &layer.up, &m_up)?;
        qwen35::swiglu(
            rt,
            Cols::dense(&m_gate.buffer, cfg.intermediate),
            Cols::dense(&m_up.buffer, cfg.intermediate),
            OutCols {
                cols: Cols::dense(&m_mid.buffer, cfg.intermediate),
                dtype: DType::F32,
            },
            t,
            cfg.intermediate,
        )?;
        let resid_out = if output {
            Some(self.residual(&resid_mid, &m_mid, &layer.down, t, mm)?)
        } else {
            None
        };
        Ok((
            Saved {
                resid_in,
                x1,
                mixer,
                resid_mid,
                x2,
                m_gate,
                m_up,
                m_mid,
            },
            resid_out,
        ))
    }

    fn gdn_forward(&self, w: &GdnWeights, x1: &Tensor, t: u32, mm: GemmOperands) -> Result<SavedGdn, String> {
        let (rt, g) = (&self.rt, self.cfg.gdn);
        let (tu, hv, dv) = (t as usize, g.v_heads() as usize, g.v_dim() as usize);
        let dk = GDN_TRAIN_DK as usize;
        let proj = tensor(rt, &[tu, g.width() as usize])?;
        mm.nn(x1, &w.w_in, &proj)?;
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
        let (q, k, v) = (
            tensor(rt, &[1, tu, hv, dk])?,
            tensor(rt, &[1, tu, hv, dk])?,
            tensor(rt, &[1, tu, hv, dv])?,
        );
        // Grouped heads: q and k land at key-head width, then repeat.
        let r = g.v_heads() / g.k_heads();
        let staged = if r > 1 {
            Some(f32s(rt, tu * g.key_dim() as usize)?)
        } else {
            None
        };
        for (off, dst, width, repeat) in [
            (qkv.q_off, &q, g.key_dim(), true),
            (qkv.k_off, &k, g.key_dim(), true),
            (qkv.v_off, &v, g.value_dim(), false),
        ] {
            let land = match (&staged, repeat) {
                (Some(s), true) => s,
                _ => &dst.buffer,
            };
            copy_cols(
                rt,
                Cols {
                    buf: &conv,
                    ld: qkv.ld,
                    off,
                },
                Cols::dense(land, width),
                t,
                width,
            )?;
            if let (Some(s), true) = (&staged, repeat) {
                repeat_heads(rt, s, &dst.buffer, t, g.k_heads(), r)?;
            }
        }
        let (gt, beta) = (tensor(rt, &[1, tu, hv])?, tensor(rt, &[1, tu, hv])?);
        qwen35::gdn_gates(
            rt,
            &g.gates(&proj.buffer),
            &GdnParams {
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
            },
            &gt.buffer,
            &beta.buffer,
            t,
            g.v_heads(),
        )?;
        let dims = GdnTrainDims {
            batch: 1,
            seq: t,
            heads: g.v_heads(),
            v_dim: g.v_dim(),
        };
        let ckpt = tensor(rt, &dims.checkpoint_shape())?;
        let o = tensor(rt, &[1, tu, hv, dv])?;
        let inputs = GdnTrainInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &gt,
            beta: &beta,
            s0: None,
        };
        gdn_train_forward(rt, dims, inputs, &o, None, &ckpt)?;
        let y = tensor(rt, &[tu, hv * dv])?;
        qwen35::gated_rms_norm(
            rt,
            Cols::dense(&o.buffer, g.value_dim()),
            g.z(&proj.buffer),
            &w.norm_w,
            OutCols {
                cols: Cols::dense(&y.buffer, g.value_dim()),
                dtype: DType::F32,
            },
            t,
            g.v_heads(),
            g.v_dim(),
            self.cfg.rms_norm_eps,
        )?;
        Ok(SavedGdn {
            proj,
            q,
            k,
            v,
            g: gt,
            beta,
            ckpt,
            o,
            y,
        })
    }

    fn attn_forward(
        &self,
        w: &AttnWeights,
        x1: &Tensor,
        t: u32,
        mm: GemmOperands,
        ws: &AttnTrainWorkspace,
    ) -> Result<SavedAttn, String> {
        let (rt, a) = (&self.rt, self.cfg.attn);
        let tu = t as usize;
        let (qd, kvd) = (
            (a.q_heads() * a.head_dim()) as usize,
            (a.kv_heads() * a.head_dim()) as usize,
        );
        let proj = tensor(rt, &[tu, a.width() as usize])?;
        mm.nn(x1, &w.w_in, &proj)?;
        let (q, k, v) = (f32s(rt, tu * qd)?, f32s(rt, tu * kvd)?, f32s(rt, tu * kvd)?);
        qwen35::attn_qk_norm_rope(
            rt,
            &self.attn_shape(t),
            Cols::dense(&proj.buffer, a.width()),
            &w.q_norm,
            &w.k_norm,
            &AttnTargets {
                q_out: &q,
                k_cache: &k,
                v_cache: &v,
            },
            0,
            self.cfg.rope_theta,
            self.cfg.rms_norm_eps,
        )?;
        let dims = self.attn_dims(t);
        // The step's one deliberate wait per attention layer (forward and
        // rebuild). Under async encode the pool recycles what earlier layers
        // freed only at a wait, and [`Self::step_bytes`] bounds the step's
        // memory by the work between two of them; without async encode every
        // dispatch has already waited and this costs nothing. (It used to be
        // the drain of a fresh workspace's host writes, made here per layer.)
        rt.synchronize()?;
        let (o, lse) = (f32s(rt, tu * qd)?, f32s(rt, dims.lse_len())?);
        attn_train_forward(rt, &dims, &q, &k, &v, &o, &lse, ws)?;
        let y = tensor(rt, &[tu, qd])?;
        qwen35::attn_output_gate(
            rt,
            &o,
            Cols::dense(&proj.buffer, a.width()),
            OutCols {
                cols: Cols::dense(&y.buffer, qd as u32),
                dtype: DType::F32,
            },
            t,
            a.q_heads(),
            a.head_dim(),
        )?;
        Ok(SavedAttn {
            proj,
            q,
            k,
            v,
            o,
            lse,
            y,
        })
    }

    /// A weight's gradient `A^T B`: straight into its bank tensor when that
    /// is f32 (over it, or added to it in the GEMM), else into a fresh
    /// tensor delivered after the layer. Returns the tensor it is in, the
    /// bank's own when it went there ([`deliver`] then skips it).
    fn weight_grad(
        &self,
        mm: GemmOperands,
        a_km: &Tensor,
        b_kn: &Tensor,
        bank: Option<(&Tensor, bool)>,
    ) -> Result<Tensor, String> {
        match bank {
            Some((dst, add)) if dst.dtype == DType::F32 => {
                if add {
                    mm.tn_acc(a_km, b_kn, dst)?;
                } else {
                    mm.tn(a_km, b_kn, dst)?;
                }
                Ok(dst.clone())
            }
            _ => {
                let g = tensor(&self.rt, &[a_km.shape()[1], b_kn.shape()[1]])?;
                mm.tn(a_km, b_kn, &g)?;
                Ok(g)
            }
        }
    }

    /// One layer's backward. On entry `sc.dresid` is the gradient of the
    /// layer's output; on return, of its input. `bank` is the layer's
    /// gradients in a bank (and whether to add into them): its f32 weight
    /// matrices take their gradients in place ([`Self::weight_grad`]).
    fn train_layer_backward(
        &self,
        layer: &Layer,
        s: &Saved,
        sc: &mut Scratch,
        mm: GemmOperands,
        bank: Option<(&LayerGrads, bool)>,
    ) -> Result<LayerGrads, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (t, h, i) = (sc.t, cfg.hidden, cfg.intermediate);
        let hu = h as usize;
        let eps = cfg.rms_norm_eps;

        // MLP: resid_out = resid_mid + swiglu(x2 @ gate, x2 @ up) @ down.
        let into = |pick: fn(&LayerGrads) -> &Tensor| bank.map(|(b, add)| (pick(b), add));
        let down = self.weight_grad(mm, &s.m_mid, &sc.dresid, into(|b| &b.down))?;
        mm.nt(&sc.dresid, &layer.down, &sc.d_mid)?;
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
        let gate = self.weight_grad(mm, &s.x2, &sc.d_gate, into(|b| &b.gate))?;
        let up = self.weight_grad(mm, &s.x2, &sc.d_up, into(|b| &b.up))?;
        // dx = d_gate @ gate^T + d_up @ up^T, the second added in its GEMM.
        mm.nt(&sc.d_gate, &layer.gate, &sc.dx)?;
        mm.nt_acc(&sc.d_up, &layer.up, &sc.dx)?;
        let post_norm = f32s(rt, hu)?;
        rms_norm_bwd(
            rt,
            &s.resid_mid.buffer,
            &layer.post_norm,
            &sc.dx.buffer,
            &sc.dresid.buffer,
            &post_norm,
            &sc.norm_part,
            t,
            h,
            eps,
            true,
        )?;

        // Mixer: resid_mid = resid_in + mixer(x1), x1 = norm(resid_in).
        const MISMATCH: &str = "Qwen35Model::train_step: a layer's saved state or bank is not its mixer's";
        let mixer = match (&layer.mixer, &s.mixer, bank.map(|(b, add)| (&b.mixer, add))) {
            (Mixer::Gdn(w), SavedMixer::Gdn(sv), into) => {
                let into = match into {
                    Some((MixerGrads::Gdn(g), add)) => Some((&g.w_in, &g.w_out, add)),
                    Some(_) => return Err(MISMATCH.into()),
                    None => None,
                };
                MixerGrads::Gdn(self.gdn_backward(w, sv, &s.x1, sc, mm, into)?)
            }
            (Mixer::Attn(w), SavedMixer::Attn(sv), into) => {
                let into = match into {
                    Some((MixerGrads::Attn(g), add)) => Some((&g.w_in, &g.w_out, add)),
                    Some(_) => return Err(MISMATCH.into()),
                    None => None,
                };
                MixerGrads::Attn(self.attn_backward(w, sv, &s.x1, sc, mm, into)?)
            }
            _ => return Err(MISMATCH.into()),
        };
        let input_norm = f32s(rt, hu)?;
        rms_norm_bwd(
            rt,
            &s.resid_in.buffer,
            &layer.input_norm,
            &sc.dx.buffer,
            &sc.dresid.buffer,
            &input_norm,
            &sc.norm_part,
            t,
            h,
            eps,
            true,
        )?;
        Ok(LayerGrads {
            input_norm,
            post_norm,
            mixer,
            gate,
            up,
            down,
        })
    }

    /// The GDN mixer's backward from `sc.dresid`; leaves the gradient of its
    /// input `x1` in `sc.dx`.
    /// `bank` is the layer's `(w_in, w_out)` bank gradients (and whether to
    /// add into them), as [`Self::train_layer_backward`] takes them.
    fn gdn_backward(
        &self,
        w: &GdnWeights,
        s: &SavedGdn,
        x1: &Tensor,
        sc: &mut Scratch,
        mm: GemmOperands,
        bank: Option<(&Tensor, &Tensor, bool)>,
    ) -> Result<GdnGrads, String> {
        let (rt, cfg, g) = (&self.rt, &self.cfg, self.cfg.gdn);
        let t = sc.t;
        let gs = sc.gdn.as_ref().ok_or("Qwen35Model::train_step: no GDN scratch")?;
        let w_out = self.weight_grad(mm, &s.y, &sc.dresid, bank.map(|(_, o, add)| (o, add)))?;
        mm.nt(&sc.dresid, &w.w_out, &gs.dy)?;
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
        let dims = GdnTrainDims {
            batch: 1,
            seq: t,
            heads: g.v_heads(),
            v_dim: g.v_dim(),
        };
        gdn_train_backward(
            rt,
            dims,
            GdnTrainInputs {
                q: &s.q,
                k: &s.k,
                v: &s.v,
                g: &s.g,
                beta: &s.beta,
                s0: None,
            },
            &s.ckpt,
            &gs.d_o,
            None,
            &gs.ws,
            GdnTrainGrads {
                dq: &gs.dq,
                dk: &gs.dk,
                dv: &gs.dv,
                dg: &gs.dg,
                dbeta: &gs.dbeta,
                ds0: None,
            },
        )?;
        // The gates' logits (a, b columns) and their parameters.
        let (a_log, dt_bias) = (f32s(rt, g.v_heads() as usize)?, f32s(rt, g.v_heads() as usize)?);
        gdn_gates_bwd(
            rt,
            &g.gates(&s.proj.buffer),
            &GdnParams {
                a_log: &w.a_log,
                dt_bias: &w.dt_bias,
            },
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
        // Grouped heads: dq and dk sum over each key head's value heads first.
        let qkv = g.conv_qkv(&gs.d_conv);
        let r = g.v_heads() / g.k_heads();
        for (src, off, width, grouped) in [
            (&gs.dq, qkv.q_off, g.key_dim(), true),
            (&gs.dk, qkv.k_off, g.key_dim(), true),
            (&gs.dv, qkv.v_off, g.value_dim(), false),
        ] {
            let src = match (&gs.dqk, grouped) {
                (Some(sum), true) => {
                    sum_heads(rt, &src.buffer, sum, t, g.k_heads(), r)?;
                    sum
                }
                _ => &src.buffer,
            };
            copy_cols(
                rt,
                Cols::dense(src, width),
                Cols {
                    buf: &gs.d_conv,
                    ld: qkv.ld,
                    off,
                },
                t,
                width,
            )?;
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
        let w_in = self.weight_grad(mm, x1, &gs.dproj, bank.map(|(i, _, add)| (i, add)))?;
        mm.nt(&gs.dproj, &w.w_in, &sc.dx)?;
        Ok(GdnGrads {
            w_in,
            w_out,
            conv_w,
            a_log,
            dt_bias,
            norm_w,
        })
    }

    /// The attention mixer's backward from `sc.dresid`; leaves the gradient
    /// of its input `x1` in `sc.dx`.
    /// `bank` as [`Self::gdn_backward`] takes it.
    fn attn_backward(
        &self,
        w: &AttnWeights,
        s: &SavedAttn,
        x1: &Tensor,
        sc: &mut Scratch,
        mm: GemmOperands,
        bank: Option<(&Tensor, &Tensor, bool)>,
    ) -> Result<AttnGrads, String> {
        let (rt, cfg, a) = (&self.rt, &self.cfg, self.cfg.attn);
        let t = sc.t;
        let qd = a.q_heads() * a.head_dim();
        let asc = sc
            .attn
            .as_ref()
            .ok_or("Qwen35Model::train_step: no attention scratch")?;
        let w_out = self.weight_grad(mm, &s.y, &sc.dresid, bank.map(|(_, o, add)| (o, add)))?;
        mm.nt(&sc.dresid, &w.w_out, &asc.dy)?;
        let proj = Cols::dense(&s.proj.buffer, a.width());
        let dproj = &asc.dproj.buffer;
        // y = o * sigmoid(gate): d_o, and the gate columns of dproj.
        attn_gate_bwd(
            rt,
            &s.o,
            proj,
            Cols::dense(&asc.dy.buffer, qd),
            &asc.d_o,
            dproj,
            t,
            a.q_heads(),
            a.head_dim(),
        )?;
        let grads = AttnTrainGrads {
            dq: &asc.dq,
            dk: &asc.dk,
            dv: &asc.dv,
        };
        attn_train_backward(
            rt,
            &self.attn_dims(t),
            &s.q,
            &s.k,
            &s.v,
            &s.o,
            &s.lse,
            &asc.d_o,
            &grads,
            &asc.ws,
        )?;
        // q, k, v back through RoPE and the norms into their dproj columns.
        let (q_norm, k_norm) = (f32s(rt, a.head_dim() as usize)?, f32s(rt, a.head_dim() as usize)?);
        attn_qk_norm_rope_bwd(
            rt,
            &self.attn_shape(t),
            proj,
            &w.q_norm,
            &w.k_norm,
            &AttnQkvGrads {
                dq: &asc.dq,
                dk: &asc.dk,
                dv: &asc.dv,
            },
            dproj,
            &q_norm,
            &k_norm,
            &asc.part,
            cfg.rope_theta,
            cfg.rms_norm_eps,
        )?;
        let w_in = self.weight_grad(mm, x1, &asc.dproj, bank.map(|(i, _, add)| (i, add)))?;
        mm.nt(&asc.dproj, &w.w_in, &sc.dx)?;
        Ok(AttnGrads {
            w_in,
            w_out,
            q_norm,
            k_norm,
        })
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
        let st = SafeTensors::open(Path::new(
            &std::env::var("QWEN35_2B_SAFETENSORS").expect("QWEN35_2B_SAFETENSORS"),
        ))
        .unwrap();
        let rt = GpuRuntime::new().unwrap();
        let model = Qwen35Model::load(
            &rt,
            &st,
            "model.language_model.",
            Qwen35Config::qwen35_2b().unwrap(),
            Precision::F32,
        )
        .unwrap();
        drop(st);
        let ids: Vec<u32> = npy(&dir.join("ids.npy")).iter().map(|&x| x as u32).collect();
        let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let base = loss(&model, &ids);
        // (name, the model's f32 buffer, tessl's gradient buffer, length).
        let mixer = |l: usize| match (&model.layers[l].mixer, &step.grads.layers[l].mixer) {
            (Mixer::Gdn(w), MixerGrads::Gdn(g)) => (w, g),
            _ => panic!("layer {l} is not a GDN layer"),
        };
        let h = model.cfg.hidden as usize;
        let cases: Vec<(String, &GpuBuffer, &GpuBuffer, usize)> = vec![
            (
                "model.layers.20.linear_attn.conv1d.weight".into(),
                &mixer(20).0.conv_w,
                &mixer(20).1.conv_w,
                6144 * 4,
            ),
            (
                "model.layers.23.input_layernorm.weight".into(),
                &model.layers[23].input_norm,
                &step.grads.layers[23].input_norm,
                h,
            ),
            (
                "model.layers.8.linear_attn.dt_bias".into(),
                &mixer(8).0.dt_bias,
                &mixer(8).1.dt_bias,
                16,
            ),
            (
                "model.layers.8.linear_attn.A_log".into(),
                &mixer(8).0.a_log,
                &mixer(8).1.a_log,
                16,
            ),
            ("model.norm.weight".into(), &model.final_norm, &step.grads.final_norm, h),
            (
                "model.layers.22.input_layernorm.weight".into(),
                &model.layers[22].input_norm,
                &step.grads.layers[22].input_norm,
                h,
            ),
            (
                "model.layers.8.post_attention_layernorm.weight".into(),
                &model.layers[8].post_norm,
                &step.grads.layers[8].post_norm,
                h,
            ),
            (
                "model.layers.6.input_layernorm.weight".into(),
                &model.layers[6].input_norm,
                &step.grads.layers[6].input_norm,
                h,
            ),
            (
                "model.layers.20.linear_attn.norm.weight".into(),
                &mixer(20).0.norm_w,
                &mixer(20).1.norm_w,
                128,
            ),
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
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                &model.layers[0].gate.buffer,
                &step.grads.layers[0].gate.buffer,
                h,
                i,
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                &model.layers[0].down.buffer,
                &step.grads.layers[0].down.buffer,
                i,
                h,
            ),
            (
                "model.layers.3.self_attn.o_proj.weight".into(),
                &attn(3).0.w_out.buffer,
                &attn(3).1.w_out.buffer,
                qd,
                h,
            ),
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
                FdCase {
                    name,
                    param,
                    grad,
                    n,
                    torch,
                    resolved,
                    steps,
                }
            })
            .collect();
        for (name, p, g, rows, cols) in matrices {
            let t = npy(&dir.join(format!("grad.{name}.npy")));
            let packed: Vec<f64> = (0..rows)
                .flat_map(|r| (0..cols).map(move |c| (r, c)))
                .map(|(r, c)| t[c * rows + r])
                .collect();
            all_cases.push(FdCase {
                name,
                param: p,
                grad: g,
                n: rows * cols,
                torch: packed,
                resolved: true,
                steps: [1e-2, 5e-3, 2.5e-3, 1.25e-3],
            });
        }
        for FdCase {
            name,
            param,
            grad,
            n,
            torch: g_torch,
            resolved,
            steps,
        } in all_cases
        {
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
                    let p: Vec<f32> = orig
                        .iter()
                        .zip(&v)
                        .map(|(&x, &u)| (f64::from(x) + s * u) as f32)
                        .collect();
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
            line += &format!(
                "; extrapolated {fd:.5e}: off tessl {:+.2e}, off torch {:+.2e}",
                fd - pt,
                fd - pr
            );
            eprintln!("{line}");
            assert_eq!(
                loss(&model, &ids).to_bits(),
                base.to_bits(),
                "{name}: the parameter was not restored"
            );
            if resolved {
                assert!(
                    (fd - pt).abs() <= 0.3 * dn,
                    "{name}: tessl's loss moves as {fd:.6e} along v, its gradient says {pt:.6e}"
                );
                assert!(
                    (fd - pt).abs() < (fd - pr).abs(),
                    "{name}: the finite difference is closer to transformers' gradient"
                );
            }
        }
    }
}
