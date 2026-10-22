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
    cross_entropy_rows, cross_entropy_rows_accumulating, cross_entropy_rows_chunked, gather_rows_f32, CeGrads,
    CeHidden, CeWorkspace, Reduction,
};
use crate::dispatch::{dispatch_1d, dispatch_2d, set_gpu_buf, set_gpu_buf_offset, set_u32};
use crate::gdn_train::{
    gdn_train_backward_windowed, gdn_train_forward_transient_bytes, gdn_train_forward_windowed, GdnTrainDims,
    GdnTrainWindowedGrads, GdnTrainWindowedInputs, GdnTrainWorkspace, GdnWindow,
};
use crate::gemm::{cast_bf16_to_f32_into, cast_f32_to_bf16_into, GemmOperands};
use crate::nn::require_runtime;
use crate::qwen35::{self, AttnShape, AttnTargets, Cols, GdnParams, OutCols, StateIn};
use crate::qwen35_adamw::{fused_table_bytes, AdamW, AdamWConfig, AdamWHyper, FusedAdamW};
use crate::qwen35_bwd::{
    attn_gate_bwd, attn_qk_norm_rope_bwd, attn_qk_norm_rope_bwd_part_len, check_scatter_rows, conv1d_silu_bwd,
    conv1d_silu_bwd_part_len, embed_rows_bwd_once, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len, gdn_gates_bwd,
    gdn_gates_bwd_part_len, rms_norm_bwd, rms_norm_bwd_part_len, scatter_add_rows, swiglu_bwd, AttnQkvGrads,
    EmbedBwdWorkspace,
};
use crate::qwen35_model::{AttnWeights, GdnWeights, Layer, Mixer, Precision, Qwen35Model};
use crate::qwen35_params::{embed_slot, layer_slots, tail_slots};
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

/// The directory `tools/qwen35_ref/make_train_fixture.py 2b` writes the real
/// 2B's reference gradients to: `QWEN35_TRAIN_REF_DIR`, else
/// `target/qwen35_train_ref` under this crate. The one owner of that
/// resolution for the ignored real-2B tests (`tests/qwen35_train.rs` and this
/// file's), so a missing reference is named in one place. Test support, not
/// API.
///
/// # Errors
/// The directory has no `ids.npy`: the message names the variable to set and
/// the path tried.
#[doc(hidden)]
pub fn train_reference_dir() -> Result<std::path::PathBuf, String> {
    reference_dir_from(std::env::var_os("QWEN35_TRAIN_REF_DIR"))
}

/// [`train_reference_dir`] with the variable's value passed in.
fn reference_dir_from(var: Option<std::ffi::OsString>) -> Result<std::path::PathBuf, String> {
    reference_dir_named(
        var,
        "QWEN35_TRAIN_REF_DIR",
        "target/qwen35_train_ref",
        "make_train_fixture.py 2b",
    )
}

/// The f64 autograd reference's directory (`make_train_fixture.py 2b --f64`):
/// `QWEN35_TRAIN_REF_F64_DIR`, else `target/qwen35_train_ref_f64`. The same
/// resolution and the same refusal as [`train_reference_dir`], under its own
/// variable. Test support, not API.
///
/// # Errors
/// The directory has no `ids.npy`: the message names the variable to set and
/// the path tried.
#[doc(hidden)]
pub fn train_reference_f64_dir() -> Result<std::path::PathBuf, String> {
    reference_dir_named(
        std::env::var_os("QWEN35_TRAIN_REF_F64_DIR"),
        "QWEN35_TRAIN_REF_F64_DIR",
        "target/qwen35_train_ref_f64",
        "make_train_fixture.py 2b --f64",
    )
}

/// The one resolution behind both reference directories: `var` (the value of
/// the variable `name`), else `default` under this crate; the directory must
/// hold `ids.npy`.
fn reference_dir_named(
    var: Option<std::ffi::OsString>,
    name: &str,
    default: &str,
    command: &str,
) -> Result<std::path::PathBuf, String> {
    let (dir, from) = match var {
        Some(d) => (std::path::PathBuf::from(d), name.to_string()),
        None => (
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(default),
            format!("the default {default}"),
        ),
    };
    if dir.join("ids.npy").is_file() {
        Ok(dir)
    } else {
        Err(format!(
            "no 2B reference at {} (from {from}): set {name} to the directory \
             `tools/qwen35_ref/{command}` wrote (it holds ids.npy)",
            dir.display()
        ))
    }
}

/// Every parameter's gradient.
pub struct Qwen35Grads {
    /// `[vocab, hidden]`: the embedding's gradient. Tied, the embedding is
    /// also the LM head and this holds both uses summed.
    pub embed: Tensor,
    /// `[vocab, hidden]`: an untied model's LM head gradient (`Some` exactly
    /// when [`Qwen35Model`] has its own `lm_head`); the embedding's is then
    /// the gather's alone.
    pub lm_head: Option<Tensor>,
    pub final_norm: GpuBuffer,
    pub layers: Vec<LayerGrads>,
}

/// One gradient (or weight) buffer: the buffer, the byte offset of its
/// elements, how many elements it holds, and their type (a matrix is f32 or
/// bf16; a [`GpuBuffer`] part is always f32).
type Part<'a> = (&'a GpuBuffer, usize, usize, DType);

/// The one `GdnTrainDims` the trainer builds: a single sequence of `t` tokens over
/// the layout's value heads.
fn single_seq_gdn_dims(g: &qwen35::GdnProjLayout, t: u32) -> GdnTrainDims {
    GdnTrainDims {
        batch: 1,
        seq: t,
        heads: g.v_heads(),
        v_dim: g.v_dim(),
    }
}

fn tensor_part(t: &Tensor) -> Part<'_> {
    (&t.buffer, t.byte_offset(), t.numel(), t.dtype)
}

fn buf_part(b: &GpuBuffer) -> Part<'_> {
    (b, 0, b.nbytes() / 4, DType::F32)
}

/// Bytes the pool makes a fresh buffer for `part`'s elements (a fused
/// step's scratch is each weight's size and dtype).
fn scratch_part_bytes(&(_, _, n, dtype): &Part<'_>) -> u64 {
    let elem = if dtype == DType::BF16 { 2 } else { 4 };
    GpuRuntime::allocated_bytes_for(n.saturating_mul(elem), BufferKind::Cold)
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

impl MixerGrads {
    /// The mixer's buffers, in the order [`layer_weight_parts`] walks its weights.
    fn parts(&self) -> Vec<Part<'_>> {
        match self {
            MixerGrads::Gdn(g) => vec![
                tensor_part(&g.w_in),
                tensor_part(&g.w_out),
                buf_part(&g.conv_w),
                buf_part(&g.a_log),
                buf_part(&g.dt_bias),
                buf_part(&g.norm_w),
            ],
            MixerGrads::Attn(g) => vec![
                tensor_part(&g.w_in),
                tensor_part(&g.w_out),
                buf_part(&g.q_norm),
                buf_part(&g.k_norm),
            ],
        }
    }
}

impl LayerGrads {
    /// Every buffer, in the order [`layer_weight_parts`] walks the weights.
    fn parts(&self) -> Vec<Part<'_>> {
        let mut v = vec![buf_part(&self.input_norm), buf_part(&self.post_norm)];
        v.extend(self.mixer.parts());
        v.extend([tensor_part(&self.gate), tensor_part(&self.up), tensor_part(&self.down)]);
        v
    }

    /// [`Self::parts`] but the mixer's: the ragged backward delivers each
    /// sequence's mixer gradients as it makes them.
    fn parts_without_mixer(&self) -> Vec<Part<'_>> {
        vec![
            buf_part(&self.input_norm),
            buf_part(&self.post_norm),
            tensor_part(&self.gate),
            tensor_part(&self.up),
            tensor_part(&self.down),
        ]
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
            lm_head: model.lm_head.as_ref().map(zeros_tensor).transpose()?,
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
    /// The LM head's weight gradient (tied: the embedding's, which the
    /// gather adds onto; untied: the head's own, which the gather never
    /// touches); none when the step scores nothing in tessl, whose embedding
    /// gradient is then the gather's alone, or when the head went straight
    /// into the bank (`head_in_bank`, `token_rows`).
    d_embed: Option<Tensor>,
    /// The cross-entropy wrote (or added) the head gradient into the bank
    /// itself: tied, into the f32 embedding the gather then adds onto;
    /// untied, into the head's matrix (in place or a chunk at a time).
    head_in_bank: bool,
    /// The cross-entropy delivered the head gradient into a bank embedding
    /// a chunk at a time ([`Head::ByChunk`]), all but these rows, the
    /// step's tokens, which wait in f32 for the gather.
    token_rows: Option<TokenRows>,
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
    /// Set by [`Qwen35Model::train_step_ragged_into`] alone.
    ragged: Option<RaggedPending>,
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

/// Rows `[v0, v0 + w)` of the `[vocab, hidden]` matrix `t`, as a [`Part`].
fn matrix_rows_part(t: &Tensor, v0: usize, w: usize) -> Part<'_> {
    let hidden = t.shape()[1];
    let elem = if t.dtype == DType::BF16 { 2 } else { 4 };
    (&t.buffer, t.byte_offset() + v0 * hidden * elem, w * hidden, t.dtype)
}

/// Zero a `[vocab, hidden]` bank matrix: f32 in one dispatch, bf16 (which
/// [`zero_part`] does not write) from a zeroed f32 chunk delivered over each
/// run of rows, so no `[vocab, hidden]` f32 tensor is made.
fn zero_matrix(rt: &Arc<GpuRuntime>, t: &Tensor) -> Result<(), String> {
    if t.dtype != DType::BF16 {
        return zero_part(rt, tensor_part(t));
    }
    let (vocab, hidden) = (t.shape()[0], t.shape()[1]);
    let chunk = CE_CHUNK.min(vocab as u32) as usize;
    let zeros = tensor(rt, &[chunk, hidden])?;
    zero_part(rt, tensor_part(&zeros))?;
    let mut v0 = 0;
    while v0 < vocab {
        let w = chunk.min(vocab - v0);
        deliver(
            rt,
            &[matrix_rows_part(&zeros, 0, w)],
            &[matrix_rows_part(t, v0, w)],
            false,
        )?;
        v0 += w;
    }
    Ok(())
}

/// Whether a bank's embedding gradient takes the head's and the gather's
/// gradients in place: f32 (a GEMM accumulates into it) and a whole buffer
/// (the gather's kernel addresses the buffer from its start).
fn embed_in_place(t: &Tensor) -> bool {
    t.dtype == DType::F32 && t.byte_offset() == 0 && t.nbytes_logical() == t.buffer.nbytes()
}

/// Where a step's head gradient (the tied embedding's, from its loss) goes.
#[derive(Clone, Copy)]
enum Head<'a> {
    /// A fresh f32 `[vocab, hidden]` tensor the pending step keeps: a step
    /// whose gradients come back fresh, or [`Qwen35Model::train_forward`],
    /// which does not know the bank yet.
    Fresh,
    /// Written (or added, `true`) by the cross-entropy's GEMMs into a bank's
    /// f32 embedding, a whole buffer ([`embed_in_place`]), onto which the
    /// gather then adds.
    InPlace(&'a Tensor, bool),
    /// Into any other bank embedding (bf16, or an f32 view), added to it
    /// when `true`: a chunk at a time, each element rounded once
    /// ([`head_rows_to_bank`]), but for the step's token rows
    /// ([`TokenRows`]), which take the gather's gradient first. No
    /// `[vocab, hidden]` f32 tensor (2 GB on the 2B, 2.5 GB at the 4B's
    /// shapes).
    ByChunk(&'a Tensor, bool),
}

/// Where the head gradient goes for a step into `bank` (fresh gradients when
/// none): the bank matrix that takes it is an untied model's own, else the
/// embedding's.
fn bank_head(bank: Option<(&Qwen35Grads, bool)>) -> Head<'_> {
    match bank {
        None => Head::Fresh,
        Some((b, add)) => {
            let target = b.lm_head.as_ref().unwrap_or(&b.embed);
            if embed_in_place(target) {
                Head::InPlace(target, add)
            } else {
                Head::ByChunk(target, add)
            }
        }
    }
}

/// Where a step's backward puts the gradients, for the pre-flight's count.
#[derive(Clone, Copy)]
enum Sink {
    /// Into a bank the caller made (or one the forward does not know yet).
    Bank,
    /// Fresh tensors, returned to the caller.
    Fresh,
    /// Into per-layer scratch each layer's AdamW update (under this config)
    /// consumes ([`Qwen35Model::train_step_fused_adamw`]).
    Fused(AdamWConfig),
}

impl Sink {
    fn fresh(self) -> bool {
        matches!(self, Self::Fresh)
    }

    fn fused(self) -> Option<AdamWConfig> {
        match self {
            Self::Fused(c) => Some(c),
            _ => None,
        }
    }
}

/// [`Head`]'s kind, for [`Qwen35Model::step_bytes`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum HeadKind {
    Fresh,
    InPlace,
    ByChunk,
}

impl Head<'_> {
    fn kind(&self) -> HeadKind {
        match self {
            Self::Fresh => HeadKind::Fresh,
            Self::InPlace(..) => HeadKind::InPlace,
            Self::ByChunk(..) => HeadKind::ByChunk,
        }
    }
}

/// The rows of a bank embedding the embedding gather adds to (the step's
/// distinct tokens) and their f32 gradient until it has: the head's part
/// (or zero), then the gather's added, then delivered once
/// ([`token_rows_to_bank`]).
struct TokenRows {
    /// The distinct ids, ascending, on the device.
    uniq: GpuBuffer,
    n_uniq: u32,
    /// Each position's index into `uniq`: the gather's ids into `side`,
    /// grouped in the order of the ids themselves, so each row's sum is
    /// formed exactly as into the embedding.
    compact: Vec<u32>,
    /// f32 `[n_uniq, hidden]`: row `u` for vocabulary row `uniq[u]`.
    side: Tensor,
}

impl TokenRows {
    fn new(rt: &Arc<GpuRuntime>, ids: &[u32], hidden: usize) -> Result<Self, String> {
        let mut uniq = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let compact = ids
            .iter()
            .map(|id| uniq.binary_search(id).map(|u| u as u32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "TokenRows: an id missing from its own set".to_string())?;
        Ok(Self {
            uniq: rt.alloc_buffer_from_u32(&uniq)?,
            n_uniq: uniq.len() as u32,
            compact,
            // Written in full before it is read: every token row lies in one
            // head chunk, or is zeroed when the step has no head gradient.
            side: rt.alloc_tensor_unzeroed(&[uniq.len(), hidden], DType::F32)?,
        })
    }
}

/// The bank kernel `base` for `bank`'s dtype.
fn bank_kernel(base: &'static str, bank: &Tensor) -> Result<&'static str, String> {
    Ok(match (base, bank.dtype) {
        ("head", DType::F32) => "qwen35_head_rows_to_bank_f32",
        ("head", DType::BF16) => "qwen35_head_rows_to_bank_bf16",
        ("token", DType::F32) => "qwen35_token_rows_to_bank_f32",
        ("token", DType::BF16) => "qwen35_token_rows_to_bank_bf16",
        (_, d) => return Err(format!("the bank's embedding is {d:?}; f32 or bf16 only")),
    })
}

/// Rows `[v0, v0 + w)` of the head's weight gradient (`src`, f32
/// `[w, hidden]`; zeros when `None`) into `bank` (over it, or added when
/// `add`, rounded once), but for `rows`' token rows, which go unrounded into
/// `rows.side`.
fn head_rows_to_bank(
    rt: &Arc<GpuRuntime>,
    bank: &Tensor,
    add: bool,
    rows: &TokenRows,
    v0: usize,
    w: usize,
    src: Option<&Tensor>,
) -> Result<(), String> {
    let hidden = bank.shape()[1];
    let p = rt.pipeline(bank_kernel("head", bank)?)?;
    dispatch_2d(rt, &p, hidden, w, |bnd| {
        match src {
            Some(s) => set_gpu_buf_offset(bnd, &s.buffer, s.byte_offset(), 0),
            // Never read without a source.
            None => set_gpu_buf(bnd, &rows.uniq, 0),
        }
        set_gpu_buf_offset(bnd, &bank.buffer, bank.byte_offset(), 1);
        set_gpu_buf(bnd, &rows.uniq, 2);
        set_gpu_buf_offset(bnd, &rows.side.buffer, rows.side.byte_offset(), 3);
        set_u32(bnd, rows.n_uniq, 4);
        set_u32(bnd, v0 as u32, 5);
        set_u32(bnd, w as u32, 6);
        set_u32(bnd, hidden as u32, 7);
        set_u32(bnd, u32::from(add), 8);
        set_u32(bnd, u32::from(src.is_some()), 9);
    })
}

/// `rows.side` into the bank's token rows, rounded once (added to the
/// bank when `add`).
fn token_rows_to_bank(rt: &Arc<GpuRuntime>, bank: &Tensor, add: bool, rows: &TokenRows) -> Result<(), String> {
    let hidden = bank.shape()[1];
    let p = rt.pipeline(bank_kernel("token", bank)?)?;
    dispatch_2d(rt, &p, hidden, rows.n_uniq as usize, |bnd| {
        set_gpu_buf_offset(bnd, &rows.side.buffer, rows.side.byte_offset(), 0);
        set_gpu_buf_offset(bnd, &bank.buffer, bank.byte_offset(), 1);
        set_gpu_buf(bnd, &rows.uniq, 2);
        set_u32(bnd, rows.n_uniq, 3);
        set_u32(bnd, hidden as u32, 4);
        set_u32(bnd, u32::from(add), 5);
    })
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
    /// The conv output `[t, conv_dim]`, `q | k | v`: the GDN kernels read
    /// them through windows of it, at key-head width.
    conv: Tensor,
    g: Tensor,
    beta: Tensor,
    ckpt: Tensor,
    o: Tensor,
    y: Tensor,
    /// `y`'s [`bf16_twin`]: read by the forward's out-projection and the
    /// backward's weight gradient.
    y16: Option<Tensor>,
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
    /// `y`'s [`bf16_twin`], as [`SavedGdn::y16`].
    y16: Option<Tensor>,
}

enum SavedMixer {
    Gdn(Box<SavedGdn>),
    Attn(SavedAttn),
    /// A ragged step: one mixer's saved state per packed sequence.
    Ragged(Vec<SavedMixer>),
}

/// The packed sequences of a ragged step: `(first row, rows)` of each, and
/// each one's attention workspace (made for its own length; none for a model
/// without attention layers).
struct RaggedRef<'a> {
    spans: &'a [(usize, usize)],
    attn_ws: &'a [Option<&'a AttnTrainWorkspace>],
}

/// A ragged backward's per-sequence scratch, each at its sequence's length.
struct RaggedScratch {
    spans: Vec<(usize, usize)>,
    scratch: Vec<MixerScratch>,
}

/// What a ragged step's [`PendingStep`] keeps of its sequences.
struct RaggedPending {
    spans: Vec<(usize, usize)>,
    attn_ws: Vec<Option<AttnTrainWorkspace>>,
    /// Each supervised row's cross-entropy, in packed order.
    per_row: Vec<f64>,
}

/// What [`Qwen35Model::train_step_ragged_into`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct RaggedLoss {
    /// The loss the step's gradients are of: the cross-entropy summed over
    /// every sequence's predicted positions, over their count.
    pub loss: f64,
    /// Each sequence's mean cross-entropy over its own `len - 1` predicted
    /// positions: the loss [`Qwen35Model::train_step`] reports for it alone.
    pub per_sequence: Vec<f64>,
}

/// The sequences `cu_seqlens` cuts `total` packed tokens into, as row ranges.
///
/// `cu_seqlens` holds the prefix offsets (`n_seq + 1` values): it starts at
/// 0, rises strictly, and ends at `total`. A zero-length sequence is refused
/// (it has no token to carry, and its offsets would repeat), and so is a
/// sequence of one token (a step scores each token's successor, so it has
/// nothing to score: [`Qwen35Model::train_step`] refuses it too). `total`
/// must fit a `u32`, the width of the kernels' row counts.
///
/// # Errors
/// Any of those rules broken, named.
pub fn check_cu_seqlens(cu_seqlens: &[u32], total: usize) -> Result<Vec<std::ops::Range<usize>>, String> {
    if u32::try_from(total).is_err() {
        return Err(format!("{total} tokens overflow u32"));
    }
    let (&first, &last) = match (cu_seqlens.first(), cu_seqlens.last()) {
        (Some(f), Some(l)) if cu_seqlens.len() >= 2 => (f, l),
        _ => {
            return Err(format!(
                "cu_seqlens has {} entries; it needs at least 2",
                cu_seqlens.len()
            ))
        }
    };
    if first != 0 {
        return Err(format!("cu_seqlens starts at {first}, not 0"));
    }
    if last as usize != total {
        return Err(format!("cu_seqlens ends at {last}, but there are {total} tokens"));
    }
    let mut out = Vec::with_capacity(cu_seqlens.len() - 1);
    for (i, w) in cu_seqlens.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        if b < a {
            return Err(format!("cu_seqlens decreases at {i}: {a} then {b}"));
        }
        match b - a {
            0 => return Err(format!("sequence {i} is empty (cu_seqlens repeats {a})")),
            1 => {
                return Err(format!(
                    "sequence {i} has one token; a step needs at least two (one prediction)"
                ))
            }
            _ => out.push(a as usize..b as usize),
        }
    }
    Ok(out)
}

/// What one layer's forward keeps: the residual stream into each norm, each
/// norm's output (the GEMMs' left operand), the mixer's, and the MLP's.
struct Saved {
    resid_in: Tensor,
    x1: Tensor,
    /// `x1`'s [`bf16_twin`]: the in-projection's operand, forward and backward.
    x1_16: Option<Tensor>,
    mixer: SavedMixer,
    resid_mid: Tensor,
    x2: Tensor,
    /// `x2`'s [`bf16_twin`]: gate and up, forward and backward.
    x2_16: Option<Tensor>,
    m_gate: Tensor,
    m_up: Tensor,
    m_mid: Tensor,
}

/// The scratch every layer's backward reuses.
struct Scratch {
    mx: MixerScratch,
    d_mid: Tensor,
    d_gate: Tensor,
    d_up: Tensor,
    norm_part: GpuBuffer,
}

/// What a mixer's backward reads and writes: the residual stream's gradient
/// in and its input's out, and the mixer's own scratch. A ragged step's
/// per-sequence ones hold views of the packed step's `dresid` and `dx`.
struct MixerScratch {
    t: u32,
    /// The residual stream's gradient.
    dresid: Tensor,
    /// A norm input's gradient before it joins `dresid`.
    dx: Tensor,
    gdn: Option<GdnScratch>,
    attn: Option<AttnScratch>,
}

struct GdnScratch {
    dy: Tensor,
    d_o: Tensor,
    dproj: Tensor,
    /// The conv output's gradient `[t, conv_dim]`: the GDN backward writes
    /// dq, dk and dv into their columns.
    d_conv: Tensor,
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

/// Where a `[n]` vector gradient is written. Every vector gradient's kernel
/// overwrites its output, so into a bank that is overwritten (not added to)
/// it goes straight into the bank's buffer, and [`deliver`] skips the
/// aliased part: no temporary and no copy dispatch. Otherwise a fresh buffer.
fn vec_grad(rt: &Arc<GpuRuntime>, n: usize, bank: Option<(&GpuBuffer, bool)>) -> Result<GpuBuffer, String> {
    match bank {
        Some((b, false)) => Ok(b.clone()),
        _ => f32s(rt, n),
    }
}

/// An f32 tensor a kernel writes in full before anything reads it: every
/// one the step allocates. Not zeroed on the host
/// ([`GpuRuntime::alloc_tensor_unzeroed`]); what must start at zero is
/// zeroed on the GPU in order with the work around it ([`zero_part`]).
fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, String> {
    rt.alloc_tensor_unzeroed(shape, DType::F32)
}

/// `x` rounded to bf16 once, for an f32 activation that several GEMMs read
/// under [`GemmOperands::Bf16`]; `None` on the exact lane. Each such GEMM
/// would otherwise round it into a fresh temporary of its own (`x2` four
/// times a layer, `dresid` four). Same cast kernel, so the GEMMs see the same
/// bits. Pair with [`op`].
fn bf16_twin(mm: GemmOperands, x: &Tensor) -> Result<Option<Tensor>, String> {
    if mm != GemmOperands::Bf16 || x.dtype != DType::F32 {
        return Ok(None);
    }
    let t = x.runtime().alloc_tensor_unzeroed(x.shape(), DType::BF16)?;
    cast_f32_to_bf16_into(x, &t)?;
    Ok(Some(t))
}

/// The tensor a GEMM takes for `x`: its bf16 twin when there is one.
fn op<'a>(x: &'a Tensor, twin: &'a Option<Tensor>) -> &'a Tensor {
    twin.as_ref().unwrap_or(x)
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
    /// layer's on a model with no wait inside the backward), rather than a
    /// second copy of every gradient held to the end. The embedding's
    /// gradient (the tied head's and the gather's) is never a tensor of its
    /// own (2 GB on the 2B): an f32 bank takes it in place, a bf16 bank a
    /// vocabulary chunk at a time, each element rounded once. The bank is
    /// checked against the model before anything runs. `sup` says what the
    /// loss scores; returns that loss.
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
        let mut top = vec![
            ("embed", tensor_part(&bank.embed), tensor_part(&self.embed)),
            ("final_norm", buf_part(&bank.final_norm), buf_part(&self.final_norm)),
        ];
        match (&bank.lm_head, &self.lm_head) {
            (Some(g), Some(w)) => top.push(("lm_head", tensor_part(g), tensor_part(w))),
            (None, None) => {}
            (Some(_), None) => return Err(format!("{what}: the bank has an lm_head gradient, the model is tied")),
            (None, Some(_)) => {
                return Err(format!(
                    "{what}: the model has an untied lm_head and the bank none (Qwen35Grads::zeros_like of this model)"
                ))
            }
        }
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
        let head = bank_head(bank);
        let p = self.step_forward(
            ids,
            operands,
            sup,
            if bank.is_none() { Sink::Fresh } else { Sink::Bank },
            head,
            None,
        )?;
        let loss = p.loss;
        Ok((loss, self.backward(p, None, bank, None)?))
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
        self.step_forward(ids, operands, sup, Sink::Bank, Head::Fresh, None)
    }

    /// A training step on several sequences packed end to end in `ids`, cut
    /// at `cu_seqlens` ([`check_cu_seqlens`]: prefix offsets, `n_seq + 1`
    /// values from 0 to `ids.len()`), with the gradients written into `bank`
    /// (over it, or added when `accumulate`, as [`Self::train_step_into`]).
    ///
    /// Each sequence is trained as [`Self::train_step`] would train it alone:
    /// positions restart at 0, the GDN recurrence and the causal convolution
    /// start from zero state, and attention sees only its own sequence, so no
    /// token reads another sequence. Every sequence's mixers run on their
    /// own rows with the single-sequence kernels; the projections, norms,
    /// MLPs and the loss run once over all rows.
    ///
    /// The loss is the token mean: every sequence's next-token cross-entropy
    /// summed, over the number of predicted positions in the batch (each
    /// sequence of `n` tokens contributes `n - 1`), as [`Supervise::Rows`]
    /// with `scale = 1 / predictions`. The gradients are those of that loss,
    /// so they equal the sum over sequences of `(n_i - 1) / predictions` times
    /// the gradients of [`Self::train_step`] on sequence `i` alone. Each
    /// sequence's own mean loss comes back in [`RaggedLoss::per_sequence`].
    ///
    /// Refused before any GPU work: a bad `cu_seqlens`, a sequence under two
    /// tokens, a bank not shaped like the model, and everything
    /// [`Self::train_forward`] refuses. The memory pre-flight bounds the step
    /// as one sequence of `ids.len()` tokens; the per-sequence workspaces sum
    /// to about that.
    ///
    /// # Errors
    /// As above, and a failed dispatch.
    pub fn train_step_ragged_into(
        &self,
        ids: &[u32],
        cu_seqlens: &[u32],
        operands: GemmOperands,
        bank: &Qwen35Grads,
        accumulate: bool,
    ) -> Result<RaggedLoss, String> {
        const WHAT: &str = "Qwen35Model::train_step_ragged_into";
        let ranges = check_cu_seqlens(cu_seqlens, ids.len()).map_err(|e| format!("{WHAT}: {e}"))?;
        self.check_bank(WHAT, bank)?;
        let mut positions = Vec::with_capacity(ids.len());
        let mut targets = Vec::with_capacity(ids.len());
        for r in &ranges {
            for p in r.start..r.end - 1 {
                positions.push(p as u32);
                targets.push(ids[p + 1]);
            }
        }
        let scale = 1.0 / positions.len() as f32;
        let spans: Vec<(usize, usize)> = ranges.iter().map(|r| (r.start, r.len())).collect();
        let sup = Supervise::Rows {
            positions: &positions,
            targets: &targets,
            scale,
        };
        let head = bank_head(Some((bank, accumulate)));
        let p = self.step_forward(ids, operands, sup, Sink::Bank, head, Some(&spans))?;
        let per_row = p
            .ragged
            .as_ref()
            .map(|r| r.per_row.clone())
            .ok_or_else(|| format!("{WHAT}: the step kept no per-row losses"))?;
        let mut per_sequence = Vec::with_capacity(spans.len());
        let mut at = 0;
        for &(_, n) in &spans {
            let rows = &per_row[at..at + n - 1];
            per_sequence.push(rows.iter().sum::<f64>() / (n - 1) as f64);
            at += n - 1;
        }
        let loss = per_row.iter().sum::<f64>() / per_row.len() as f64;
        self.backward(p, None, Some((bank, accumulate)), None)?;
        Ok(RaggedLoss { loss, per_sequence })
    }

    /// [`Self::train_forward`]; `sink` says where the backward will put the
    /// gradients, which the pre-flight counts: [`Sink::Fresh`] when it will
    /// return fresh gradients rather than deliver them into a bank, which the pre-flight
    /// then counts too. `head` says where the head gradient goes: into a
    /// bank's embedding, there is no `[vocab, hidden]` tensor and no copy of
    /// it (2 GB on the 2B).
    fn step_forward(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
        sink: Sink,
        head: Head<'_>,
        ragged: Option<&[(usize, usize)]>,
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
        if rt.f32_gemm_runs_relaxed() {
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
        let lens: Option<Vec<u32>> = ragged.map(|s| s.iter().map(|&(_, n)| n as u32).collect());
        let fused = sink.fused();
        let (have, need) = (
            rt.current_allocated_bytes(),
            // A fused step made its top bank before the forward, so the
            // allocation read above already holds it.
            self.step_bytes(t, operands, sink.fresh(), head.kind(), lens.as_deref(), fused)
                .saturating_sub(if fused.is_some() {
                    self.fused_top_bank_bytes()
                } else {
                    0
                }),
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
        // A ragged step's attention workspaces are per sequence (each at its
        // own length); the one for the whole step is not made.
        let (attn_ws, ragged_ws) = match ragged {
            None => (self.attn_workspace(t)?, Vec::new()),
            Some(spans) => (
                None,
                spans
                    .iter()
                    .map(|&(_, n)| self.attn_workspace(n as u32))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        let ws_refs: Vec<Option<&AttnTrainWorkspace>> = ragged_ws.iter().map(Option::as_ref).collect();
        let rr = ragged.map(|spans| RaggedRef {
            spans,
            attn_ws: &ws_refs,
        });
        for layer in &self.layers {
            let kept = if self.precision == Precision::Bf16 {
                let b = rt.alloc_tensor_unzeroed(&[tu, h], DType::BF16)?;
                cast_f32_to_bf16_into(&resid, &b)?;
                Some(b)
            } else {
                None
            };
            let (s, out) = self.train_layer_forward(layer, resid, t, true, operands, attn_ws.as_ref(), rr.as_ref())?;
            let out = out.ok_or("Qwen35Model::train_step: a layer's forward returned no output")?;
            inputs.push(kept.unwrap_or(s.resid_in));
            resid = out;
        }
        let xf = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid, &self.final_norm, &xf, t)?;

        // ---- loss and the LM head ------------------------------------------
        let scores = !matches!(sup, Supervise::Rows { positions: [], .. });
        let d_embed = if scores && head.kind() == HeadKind::Fresh {
            Some(tensor(rt, &[cfg.vocab as usize, h])?)
        } else {
            None
        };
        let untied = self.lm_head.is_some();
        // Tied, the head's by-chunk rows wait for the gather on the token
        // rows; an untied head's go straight into its own bank matrix.
        let token_rows = if scores && head.kind() == HeadKind::ByChunk && !untied {
            Some(TokenRows::new(rt, ids, h)?)
        } else {
            None
        };
        let dxf = tensor(rt, &[tu, h])?;
        // Positions nothing scores keep a zero gradient row. Zeroed on the
        // GPU: a host zero would wait for the whole forward first. The causal
        // loss writes rows `0..t - 1` in full (its first vocabulary chunk's
        // GEMM overwrites them), so only the last row is zeroed there.
        match (sup, tu) {
            (Supervise::Causal, 1..) => {
                zero_part(rt, (&dxf.buffer, dxf.byte_offset() + (tu - 1) * h * 4, h, DType::F32))?
            }
            _ => zero_part(rt, tensor_part(&dxf))?,
        }
        let head_w = self.head_table();
        let ce = |rows: &[u32], targets: &[u32], reduction, dh: &Tensor, scale| {
            let chunk = CE_CHUNK.min(cfg.vocab);
            let ws = CeWorkspace::new(rt, rows.len() as u32, cfg.hidden, chunk, head_w.dtype, operands)?;
            let hidden = CeHidden { rows: &xf, off: 0 };
            let (dw, add) = match (head, &d_embed) {
                (Head::InPlace(bank, add), _) => (bank, add),
                (Head::Fresh, Some(d)) => (d, false),
                (Head::ByChunk(bank, add), _) => {
                    // Each chunk's rows, written by its GEMM and delivered
                    // before the next chunk's overwrites them.
                    let scratch = tensor(rt, &[chunk as usize, h])?;
                    let mut sink = |v0: usize, rows: &Tensor| match token_rows.as_ref() {
                        Some(tr) => head_rows_to_bank(rt, bank, add, tr, v0, rows.shape()[0], Some(rows)),
                        // Untied: the rows are the head's alone.
                        None => deliver(
                            rt,
                            &[tensor_part(rows)],
                            &[matrix_rows_part(bank, v0, rows.shape()[0])],
                            add,
                        ),
                    };
                    return cross_entropy_rows_chunked(
                        rt, hidden, head_w, rows, targets, reduction, operands, &ws, dh, scale, &scratch, &mut sink,
                    );
                }
                (Head::Fresh, None) => return Err("a scored step has a head gradient".to_string()),
            };
            let grads = CeGrads { dh, dw, scale };
            if add {
                cross_entropy_rows_accumulating(rt, hidden, head_w, rows, targets, reduction, operands, &ws, grads)
            } else {
                cross_entropy_rows(rt, hidden, head_w, rows, targets, reduction, operands, &ws, Some(grads))
            }
        };
        let mut per_row = Vec::new();
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
                if ragged.is_some() {
                    per_row = out.per_row;
                }
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
            // Tied: the head went in place into the embedding bank. Untied:
            // into the head's bank matrix, in place or by chunk.
            head_in_bank: scores
                && match head.kind() {
                    HeadKind::Fresh => false,
                    HeadKind::InPlace => true,
                    HeadKind::ByChunk => untied,
                },
            token_rows,
            loss,
            attn_ws,
            embed: self.embed.buffer.clone(),
            param_generation,
            ragged: ragged.map(|spans| RaggedPending {
                spans: spans.to_vec(),
                attn_ws: ragged_ws,
                per_row,
            }),
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
        self.backward(p, dh, Some((bank, accumulate)), None).map(|_| ())
    }

    /// One training step that updates the model with AdamW as the backward
    /// runs: [`Self::train_step_into`] then [`Self::adamw_step`] without the
    /// gradient bank. Each layer's gradients are written to a scratch that
    /// the pool reuses from layer to layer, and the layer is updated from it
    /// at once, with the kernels, step scalars, per-parameter state and
    /// stochastic-rounding salts `adamw_step` uses; the embedding, the final
    /// norm and an untied head are updated after the gather. What stays
    /// allocated beyond the step's own working set is one layer's gradients
    /// and the embedding's (and an untied head's), not every parameter's
    /// ([`Self::train_step_fused_adamw_bytes`]). On the 2B in f32 that is
    /// the 7 GiB bank saved.
    ///
    /// Every parameter, moment and auxiliary value ends bit for bit as the
    /// two calls would leave it, and the loss (as `sup` defines it, returned)
    /// is the same. `state`'s step count advances by one.
    ///
    /// # Why updating mid-backward is sound
    ///
    /// The backward walks the layers top-down, and each layer reads only its
    /// own weights: the rebuild of layer `L`'s forward from its saved input,
    /// then its backward. After `train_layer_backward(L)` has delivered
    /// layer `L`'s gradient, nothing later reads layer `L`'s weights again:
    /// the layers below read their own, the gather's backward reads no
    /// weight, and the final norm's backward (the one read of a weight
    /// outside the layers) ran before the first layer. The embedding and the
    /// tied head are read in the forward alone. The GPU runs the update
    /// after the layer's backward (the runtime's encoder barrier orders
    /// dependent dispatches), and the update writes only that layer's
    /// parameters and state. So every gradient is that of the weights the
    /// forward ran on, which is what [`PendingStep`]'s generation guard
    /// protects: the guard stays as it is for every other entry point, and
    /// this step, which never leaves a forward pending, bumps the
    /// generation once at its end (even on failure after an update was
    /// encoded), so a [`PendingStep`] taken before it is refused after.
    ///
    /// # Refused before anything is written
    ///
    /// `hyper.grad_scale` other than 1 (a clip coefficient needs the global
    /// gradient norm, which does not exist until the backward ends; clip
    /// with [`Self::train_step_into`], [`Self::grad_sq_norm`] and
    /// [`Self::adamw_step`]), a `state` or `weight_decay` not shaped like the
    /// model, a parameter whose step kernel does not exist under the state's
    /// config, and everything [`Self::train_forward`] refuses (including a
    /// step the bound does not fit). One sequence only: a ragged batch has
    /// no entry here, because its mixers deliver gradients per sequence into
    /// a bank that is summed layer by layer, which a scratch that is
    /// updated and dropped per layer does not hold.
    ///
    /// `weight_decay[i]` is for [`Self::parameter_table`] entry `i`.
    ///
    /// # Errors
    /// As above, and a failed dispatch. A failure after the first layer's
    /// update was encoded leaves the model partly updated (as a failed
    /// [`Self::adamw_step`] does); the generation has moved, so no stale
    /// pending step runs on it.
    pub fn train_step_fused_adamw(
        &self,
        ids: &[u32],
        operands: GemmOperands,
        sup: Supervise<'_>,
        state: &mut AdamW,
        hyper: &AdamWHyper,
        weight_decay: &[f32],
    ) -> Result<f64, String> {
        const WHAT: &str = "Qwen35Model::train_step_fused_adamw";
        let config = state.config();
        let mut fused = FusedAdamW::new(WHAT, self, state, hyper, weight_decay)?;
        let top = self.fused_top_bank()?;
        let head = bank_head(Some((&top, false)));
        let p = self.step_forward(ids, operands, sup, Sink::Fused(config), head, None)?;
        let loss = p.loss;
        // A bank step lets the caller look at the gradients before
        // `adamw_step`; this one has no such moment, and a non-finite loss
        // would write NaN into every parameter and moment. Refused here,
        // before the backward and any update, so nothing has moved.
        if !loss.is_finite() {
            return Err(format!(
                "{WHAT}: the loss is {loss}, not finite; refused before any update, so the model and the state \
                 are as they were"
            ));
        }
        let ran = self.backward(p, None, Some((&top, false)), Some(&mut fused));
        if fused.touched() {
            self.bump_param_generation();
        }
        if let Err(e) = &ran {
            // From the first update's dispatch on, a failure leaves some
            // parameters and moments at the step and the rest before it: the
            // model and the state refuse every later step until reloaded.
            if fused.touched() {
                fused.tear(self, e);
            }
        }
        ran?;
        fused.confirm();
        Ok(loss)
    }

    /// A bound on the device bytes [`Self::train_step_fused_adamw`] on `t`
    /// tokens allocates on top of what is allocated when it starts, as
    /// [`Self::train_step_bytes`] bounds a step into a bank (every buffer at
    /// the size the pool makes it): the step's working set with the head
    /// gradient going into the embedding gradient in place (f32 weights) or
    /// a chunk at a time (bf16), plus the top gradients it holds throughout
    /// (the embedding's, an untied head's, the final norm's), plus each
    /// layer's gradient scratch and AdamW's slot tables between waits. It
    /// excludes the model and `config`'s state, which [`AdamW::allocated_bytes_for`]
    /// bounds, and the gradient bank a step into one needs: this step has
    /// none.
    pub fn train_step_fused_adamw_bytes(&self, t: u32, operands: GemmOperands, config: AdamWConfig) -> u64 {
        self.step_bytes(t, operands, false, self.fused_head_kind(), None, Some(config))
    }

    /// Where a fused step's head gradient goes: its top bank's embedding is
    /// f32 or bf16 as the weight is.
    fn fused_head_kind(&self) -> HeadKind {
        if self.embed.dtype == DType::F32 {
            HeadKind::InPlace
        } else {
            HeadKind::ByChunk
        }
    }

    /// A buffer of `like`'s size and dtype that a kernel fills before
    /// anything reads it: a fused step's gradient storage. Not zeroed (the
    /// step writes over it, as it does a bank it does not accumulate into).
    fn scratch_like(&self, like: &Tensor) -> Result<Tensor, String> {
        let dtype = if like.dtype == DType::BF16 {
            DType::BF16
        } else {
            DType::F32
        };
        self.rt.alloc_tensor_unzeroed(like.shape(), dtype)
    }

    fn scratch_buf(&self, like: &GpuBuffer) -> Result<GpuBuffer, String> {
        self.rt.alloc_buffer(like.nbytes())
    }

    /// A fused step's gradients for everything but the layers, which each
    /// have a scratch of their own ([`Self::layer_scratch`]): the bank the
    /// head and the gather write into. Held through the step.
    fn fused_top_bank(&self) -> Result<Qwen35Grads, String> {
        Ok(Qwen35Grads {
            embed: self.scratch_like(&self.embed)?,
            lm_head: self.lm_head.as_ref().map(|t| self.scratch_like(t)).transpose()?,
            final_norm: self.scratch_buf(&self.final_norm)?,
            layers: Vec::new(),
        })
    }

    /// The fused step's update of what is not a layer, from the top bank `b`
    /// the gather has just finished: nothing when the step is not fused.
    fn fused_top(&self, fused: Option<&mut FusedAdamW<'_>>, b: &Qwen35Grads) -> Result<(), String> {
        match fused {
            Some(f) => f.top(self, &b.embed, &b.final_norm, b.lm_head.as_ref()),
            None => Ok(()),
        }
    }

    /// Bytes [`Self::fused_top_bank`] allocates, at the pool's rounding.
    fn fused_top_bank_bytes(&self) -> u64 {
        let mut parts = vec![tensor_part(&self.embed), buf_part(&self.final_norm)];
        parts.extend(self.lm_head.as_ref().map(tensor_part));
        parts.iter().fold(0u64, |a, p| a.saturating_add(scratch_part_bytes(p)))
    }

    /// One layer's gradients, in the layer's weights' dtypes (as
    /// [`Qwen35Grads::zeros_like`] makes a bank's, but from the pool and not
    /// zeroed): a fused step's scratch, released once the layer is updated.
    fn layer_scratch(&self, layer: &Layer) -> Result<LayerGrads, String> {
        let mixer = match &layer.mixer {
            Mixer::Gdn(w) => MixerGrads::Gdn(GdnGrads {
                w_in: self.scratch_like(&w.w_in)?,
                w_out: self.scratch_like(&w.w_out)?,
                conv_w: self.scratch_buf(&w.conv_w)?,
                a_log: self.scratch_buf(&w.a_log)?,
                dt_bias: self.scratch_buf(&w.dt_bias)?,
                norm_w: self.scratch_buf(&w.norm_w)?,
            }),
            Mixer::Attn(w) => MixerGrads::Attn(AttnGrads {
                w_in: self.scratch_like(&w.w_in)?,
                w_out: self.scratch_like(&w.w_out)?,
                q_norm: self.scratch_buf(&w.q_norm)?,
                k_norm: self.scratch_buf(&w.k_norm)?,
            }),
        };
        Ok(LayerGrads {
            input_norm: self.scratch_buf(&layer.input_norm)?,
            post_norm: self.scratch_buf(&layer.post_norm)?,
            mixer,
            gate: self.scratch_like(&layer.gate)?,
            up: self.scratch_like(&layer.up)?,
            down: self.scratch_like(&layer.down)?,
        })
    }

    /// Bytes [`Self::layer_scratch`] allocates for `layer`, at the pool's
    /// rounding: (in f32 buffers, in bf16 ones). A step's bound
    /// ([`Self::layer_bwd_bytes`]) already charges every gradient at f32, so
    /// only the bf16 buffers are added to it.
    fn layer_scratch_bytes(&self, layer: &Layer) -> (u64, u64) {
        layer_weight_parts(layer).iter().fold((0u64, 0u64), |(f, b), p| {
            let n = scratch_part_bytes(p);
            if p.3 == DType::BF16 {
                (f, b.saturating_add(n))
            } else {
                (f.saturating_add(n), b)
            }
        })
    }

    /// The bytes a fused step's update of `layer` adds to a step's bound
    /// beyond [`Self::layer_bwd_bytes`]: its bf16 scratch buffers and its
    /// AdamW slot tables.
    fn fused_layer_bytes(&self, l: usize, config: AdamWConfig) -> u64 {
        let tables = layer_slots(self, l, None).map_or(0, |ps| fused_table_bytes(config, &ps));
        self.layer_scratch_bytes(&self.layers[l]).1.saturating_add(tables)
    }

    /// The bytes the update after the gather adds: the embedding's, the
    /// final norm's and an untied head's slot tables.
    fn fused_tail_bytes(&self, config: AdamWConfig) -> u64 {
        let mut ps = vec![embed_slot(self, None)];
        ps.extend(tail_slots(self, None).unwrap_or_default());
        fused_table_bytes(config, &ps)
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
                 (write_parameters, write_parameter_host or adamw_step); its gradients would be of weights the \
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
    ///
    /// With `fused` (and `bank` the step's top bank, [`Self::fused_top_bank`],
    /// with no layers), each layer's gradients go into a scratch of their own
    /// ([`Self::layer_scratch`]), and the layer's AdamW update is encoded as
    /// soon as the scratch is filled; the embedding's, the final norm's and
    /// an untied head's follow the gather. See
    /// [`Self::train_step_fused_adamw`] for why that is sound.
    fn backward(
        &self,
        p: PendingStep,
        dh: Option<(&[u32], &Tensor)>,
        bank: Option<(&Qwen35Grads, bool)>,
        mut fused: Option<&mut FusedAdamW<'_>>,
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
            token_rows,
            attn_ws,
            xf,
            ragged,
            ..
        } = p;
        // The final norm's output fed the loss, which is done: released now
        // rather than held through the backward (a [T, hidden] f32 tensor,
        // which the layers' own then reuse).
        drop(xf);
        let h = cfg.hidden as usize;
        if let Some((pos, g)) = dh {
            scatter_add_rows(rt, g, pos, &dxf)?;
        }

        // ---- backward -------------------------------------------------------
        // A ragged step's mixers run per sequence, each on a scratch of its
        // own length; the packed scratch then holds the rows-wise parts alone.
        let (mut sc, mut rscratch) = match ragged {
            None => (self.scratch(t, attn_ws, true)?, None),
            Some(r) => {
                if bank.is_none() {
                    return Err("Qwen35Model::train_step_ragged_into: a ragged step delivers into a bank".into());
                }
                let sc = self.scratch(t, None, false)?;
                // Each sequence's mixer reads and writes its own rows of the
                // packed gradient streams in place: no copy, no second set
                // of row-wise buffers.
                let scratch = r
                    .spans
                    .iter()
                    .zip(r.attn_ws)
                    .map(|(&(start, n), ws)| {
                        self.mixer_scratch(
                            n as u32,
                            ws,
                            true,
                            sc.mx.dresid.view(&[n, h], start * h),
                            sc.mx.dx.view(&[n, h], start * h),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    sc,
                    Some(RaggedScratch {
                        spans: r.spans,
                        scratch,
                    }),
                )
            }
        };
        let final_norm = vec_grad(rt, h, bank.map(|(b, add)| (&b.final_norm, add)))?;
        rms_norm_bwd(
            rt,
            &resid.buffer,
            &self.final_norm,
            &dxf.buffer,
            &sc.mx.dresid.buffer,
            &final_norm,
            &sc.norm_part,
            t,
            cfg.hidden,
            cfg.rms_norm_eps,
            false,
        )?;
        // The final norm's input and the gradient at its output are read
        // there alone: released before the layers' backward, as `xf` was.
        drop((resid, dxf));
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
            let s = match &rscratch {
                None => {
                    let ws = sc.mx.attn.as_ref().map(|a| &a.ws);
                    self.train_layer_forward(layer, resid_in, t, false, operands, ws, None)?
                        .0
                }
                Some(rs) => {
                    let refs: Vec<Option<&AttnTrainWorkspace>> =
                        rs.scratch.iter().map(|x| x.attn.as_ref().map(|a| &a.ws)).collect();
                    let rr = RaggedRef {
                        spans: &rs.spans,
                        attn_ws: &refs,
                    };
                    self.train_layer_forward(layer, resid_in, t, false, operands, None, Some(&rr))?
                        .0
                }
            };
            // A fused step's layer gradients go to a scratch of the layer's
            // own, replaced in the pool by the next layer's.
            let scratch = match fused {
                Some(_) => Some(self.layer_scratch(layer)?),
                None => None,
            };
            let dest = match (&scratch, bank) {
                (Some(sg), _) => Some((sg, false)),
                (None, Some((b, add))) => Some((&b.layers[li], add)),
                (None, None) => None,
            };
            let ragged_step = rscratch.is_some();
            let g = self.train_layer_backward(layer, &s, &mut sc, operands, dest, rscratch.as_mut())?;
            match dest {
                // A ragged step's mixer gradients went in per sequence.
                Some((d, add)) if ragged_step => {
                    deliver(rt, &g.parts_without_mixer(), &d.parts_without_mixer(), add)?;
                }
                Some((d, add)) => deliver(rt, &g.parts(), &d.parts(), add)?,
                None => layers.push(g),
            }
            if let (Some(f), Some(sg)) = (fused.as_deref_mut(), &scratch) {
                f.layer(self, li, sg)?;
            }
        }
        layers.reverse();
        // An untied head's gradient is settled here, and the gather then
        // proceeds as a step with no head gradient: onto nothing, into the
        // embedding alone.
        let (mut d_embed, mut head_in_bank, mut token_rows) = (d_embed, head_in_bank, token_rows);
        let mut head_fresh = None;
        if self.lm_head.is_some() {
            match bank {
                Some((b, add)) => {
                    let hb = b
                        .lm_head
                        .as_ref()
                        .ok_or("Qwen35Model::train_step: the bank has no lm_head for an untied model")?;
                    match d_embed.take() {
                        Some(d) => deliver(rt, &[tensor_part(&d)], &[tensor_part(hb)], add)?,
                        // Written in the bank already by the cross-entropy.
                        None if head_in_bank => {}
                        // Nothing scored: the bank's head gradient is zero.
                        None if !add => zero_matrix(rt, hb)?,
                        None => {}
                    }
                }
                None => {
                    head_fresh = Some(match d_embed.take() {
                        Some(d) => d,
                        None => {
                            let d = tensor(rt, &[cfg.vocab as usize, h])?;
                            zero_part(rt, tensor_part(&d))?;
                            d
                        }
                    });
                }
            }
            (head_in_bank, token_rows) = (false, None);
        }
        // The gather's gradient adds onto the head's. Into an f32 bank whose
        // embedding is a whole buffer it adds straight into the bank: onto
        // the head gradient the cross-entropy already wrote there, or, with
        // no head gradient, onto the bank zeroed first unless accumulating.
        // Into any other bank (bf16) it adds onto the token rows' f32
        // gradient the cross-entropy held back, which then goes into the
        // bank rounded once; every other row took the head's gradient
        // already, or takes none. No 2 GB tensor and no copy on the 2B.
        let (dw, direct) = match (d_embed, bank) {
            (Some(d), _) => (d, false),
            (None, Some((b, add))) if embed_in_place(&b.embed) => {
                if !add && !head_in_bank {
                    zero_part(rt, tensor_part(&b.embed))?;
                }
                (b.embed.clone(), true)
            }
            (None, Some((b, add))) => {
                let rows = match token_rows {
                    Some(rows) => rows,
                    // No head gradient: the token rows start at zero, and
                    // the other rows take nothing (zero, over the bank).
                    None => {
                        let rows = TokenRows::new(rt, &ids, h)?;
                        if add {
                            zero_part(rt, tensor_part(&rows.side))?;
                        } else {
                            head_rows_to_bank(rt, &b.embed, false, &rows, 0, cfg.vocab as usize, None)?;
                        }
                        rows
                    }
                };
                embed_rows_bwd_once(
                    rt,
                    &rows.compact,
                    &sc.mx.dresid.buffer,
                    &rows.side.buffer,
                    rows.n_uniq,
                    cfg.hidden,
                )?;
                token_rows_to_bank(rt, &b.embed, add, &rows)?;
                self.fused_top(fused, b)?;
                rt.synchronize()?;
                return Ok(None);
            }
            (None, None) => {
                let d = tensor(rt, &[cfg.vocab as usize, h])?;
                zero_part(rt, tensor_part(&d))?;
                (d, false)
            }
        };
        // A one-call grouping: fresh exact-size buffers written without a
        // wait, rather than a workspace filled by staged uploads and copies.
        embed_rows_bwd_once(rt, &ids, &sc.mx.dresid.buffer, &dw.buffer, cfg.vocab, cfg.hidden)?;
        if let Some((b, add)) = bank {
            if !direct {
                deliver(rt, &[tensor_part(&dw)], &[tensor_part(&b.embed)], add)?;
            }
            self.fused_top(fused, b)?;
            rt.synchronize()?;
            return Ok(None);
        }
        rt.synchronize()?;
        Ok(Some(Qwen35Grads {
            embed: dw,
            lm_head: head_fresh,
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
    /// `mixers` makes the GDN and attention scratch too (not the packed
    /// scratch of a ragged step, whose mixers run per sequence).
    fn scratch(&self, t: u32, attn_ws: Option<AttnTrainWorkspace>, mixers: bool) -> Result<Scratch, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let mx = self.mixer_scratch(t, attn_ws, mixers, tensor(rt, &[tu, h])?, tensor(rt, &[tu, h])?)?;
        Ok(Scratch {
            mx,
            d_mid: tensor(rt, &[tu, i])?,
            d_gate: tensor(rt, &[tu, i])?,
            d_up: tensor(rt, &[tu, i])?,
            norm_part: f32s(rt, rms_norm_bwd_part_len(t, cfg.hidden))?,
        })
    }

    /// The mixer scratch for `t` rows over `dresid` and `dx` (`[t, hidden]`,
    /// the caller's: views of the packed step's in a ragged one), with the
    /// GDN and attention scratch when `mixers`.
    fn mixer_scratch(
        &self,
        t: u32,
        attn_ws: Option<AttnTrainWorkspace>,
        mixers: bool,
        dresid: Tensor,
        dx: Tensor,
    ) -> Result<MixerScratch, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let tu = t as usize;
        let (g, a) = (cfg.gdn, cfg.attn);
        let has = |k: crate::qwen35_model::LayerKind| cfg.layers.contains(&k);
        let gdn = if mixers && has(crate::qwen35_model::LayerKind::LinearAttention) {
            let (hv, dv) = (g.v_heads() as usize, g.v_dim() as usize);
            let dims = single_seq_gdn_dims(&g, t);
            Some(GdnScratch {
                dy: tensor(rt, &[tu, hv * dv])?,
                d_o: tensor(rt, &[1, tu, hv, dv])?,
                dproj: tensor(rt, &[tu, g.width() as usize])?,
                d_conv: tensor(rt, &[tu, g.conv_dim() as usize])?,
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
        let attn = if mixers && has(crate::qwen35_model::LayerKind::FullAttention) {
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
        Ok(MixerScratch {
            t,
            dresid,
            dx,
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
    /// [`Self::train_step_into`] alone holds less: an f32 bank takes the head
    /// gradient in place, a bf16 bank a chunk at a time (its own pre-flight
    /// counts what it holds).
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
    /// the pool past the step's own. It reads the runtime's encode mode
    /// ([`GpuRuntime::dispatch_batching`]: async encode or an open
    /// [`GpuRuntime::async_scope`] batch, serialized dispatch does not) and
    /// cap when called. [`Self::train_forward`] refuses a step whose bound
    /// and the device's current allocation exceed its recommended working
    /// set.
    pub fn train_step_bytes(&self, t: u32, operands: GemmOperands) -> u64 {
        self.step_bytes(t, operands, false, HeadKind::Fresh, None, None)
    }

    /// [`Self::train_step_bytes`] for [`Self::train_step_ragged_into`] on
    /// `cu_seqlens` (as [`check_cu_seqlens`] reads it), into a bank whose
    /// head gradient goes `in_place` (an f32 embedding that is a whole
    /// buffer) or a chunk at a time: what the ragged step allocates, each
    /// sequence's mixer counted at its own length.
    ///
    /// # Errors
    /// `cu_seqlens` is malformed for `total` tokens.
    pub fn train_step_ragged_bytes(
        &self,
        cu_seqlens: &[u32],
        total: usize,
        operands: GemmOperands,
        in_place: bool,
    ) -> Result<u64, String> {
        let ranges = check_cu_seqlens(cu_seqlens, total)?;
        let lens: Vec<u32> = ranges.iter().map(|r| r.len() as u32).collect();
        let head = if in_place { HeadKind::InPlace } else { HeadKind::ByChunk };
        Ok(self.step_bytes(total as u32, operands, false, head, Some(&lens), None))
    }

    /// [`Self::train_step_bytes`]; `fresh` adds every layer's gradients held
    /// to the end, as [`Self::train_step`] returns them instead of
    /// delivering each into a bank, and `head` is where the head gradient
    /// goes: a `[vocab, hidden]` tensor held to the end ([`Head::Fresh`],
    /// what [`Self::train_forward`] makes, so what the public bound counts),
    /// nothing beyond the bank ([`Head::InPlace`]), or one `[chunk, hidden]`
    /// chunk through the cross-entropy and the token rows held to the end
    /// ([`Head::ByChunk`]).
    ///
    /// `lens` are a ragged step's sequence lengths (summing to `t`): every
    /// mixer's forward, backward and scratch, and the attention workspace,
    /// are counted once per sequence at its own length (each buffer at the
    /// pool's rounding), the row-wise parts once over `t` rows.
    ///
    /// `fused` is a [`Self::train_step_fused_adamw`] under that config: the
    /// top bank it holds throughout, each layer's bf16 scratch and slot
    /// tables between waits (its f32 scratch is the gradients
    /// [`Self::layer_bwd_bytes`] already charges), and the tail's tables.
    fn step_bytes(
        &self,
        t: u32,
        mm: GemmOperands,
        fresh: bool,
        head: HeadKind,
        lens: Option<&[u32]>,
        fused: Option<AdamWConfig>,
    ) -> u64 {
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
            lens.unwrap_or(&[t]).iter().fold(0u64, |a, &n| {
                a.saturating_add(AttnTrainWorkspace::allocated_bytes_for(self.attn_dims(n)))
            })
        } else {
            0
        };
        // A mixer's bytes over the step's sequences (one, unless ragged).
        let over = |per: &dyn Fn(u32) -> u64| lens.unwrap_or(&[t]).iter().fold(0u64, |a, &n| a.saturating_add(per(n)));
        let layer_fwd = |l: &Layer, output: bool| self.layer_fwd_step_bytes(l, t, mm, output, lens);
        // At most `t` distinct tokens: their f32 rows and their ids.
        let head_held = match head {
            // Untied, the gather's gradient is a second [vocab, hidden] tensor.
            HeadKind::Fresh => f(v * h).saturating_mul(1 + u64::from(self.lm_head.is_some())),
            HeadKind::InPlace => 0,
            // Tied, the token rows wait for the gather; an untied head's
            // chunks go straight into its own bank matrix, so nothing waits.
            HeadKind::ByChunk if self.lm_head.is_none() => sum(&[th, f(tu)]),
            HeadKind::ByChunk => 0,
        };
        let top_bank = if fused.is_some() {
            self.fused_top_bank_bytes()
        } else {
            0
        };
        let held = sum(&[
            f(tu),
            kept.saturating_mul(layers),
            th.saturating_mul(3),
            head_held,
            attn_ws,
            top_bank,
        ]);
        let fwd: Vec<u64> = self.layers.iter().map(|l| layer_fwd(l, true)).collect();
        let bwd: Vec<(u64, u64)> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| {
                let (grads, temps) = self.layer_bwd_step_bytes(l, t, mm, lens);
                let fused_extra = fused.map_or(0, |c| self.fused_layer_bytes(li, c));
                (grads, sum(&[layer_fwd(l, false), grads, temps, widened, fused_extra]))
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
        // Unbatched, every dispatch is a waited commit, so a layer's buffers
        // are recycled once the next layer's first kernel runs; batched
        // (async encode or an open async_scope, unless serialized), only at
        // an attention layer's deliberate wait.
        let batched = self.rt.dispatch_batching();
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
        let ce = self
            .ce_bytes(t, mm, batched)
            .saturating_add(if head == HeadKind::ByChunk {
                f(CE_CHUNK.min(cfg.vocab) as usize * h)
            } else {
                0
            });
        // An untied bf16 bank head is zeroed, when nothing scores, from one
        // zeroed f32 `[chunk, hidden]` tensor ([`zero_matrix`]).
        let zero_chunk = if head != HeadKind::Fresh && self.lm_head.is_some() && bf16 {
            f(CE_CHUNK.min(cfg.vocab) as usize * h)
        } else {
            0
        };
        let scratch = match lens {
            None => self.scratch_bytes(t),
            // The packed scratch holds the row-wise buffers alone; each
            // sequence's mixer scratch views its rows of `dresid` and `dx`.
            Some(_) => self
                .rowwise_scratch_bytes(t)
                .saturating_add(over(&|n| self.mixer_scratch_bytes(n))),
        };
        let backward_held = sum(&[
            scratch,
            zero_chunk,
            f(h),
            EmbedBwdWorkspace::allocated_bytes_for(t),
            fresh_grads,
            fused.map_or(0, |c| self.fused_tail_bytes(c)),
        ]);
        let pool = self.rt.memory_info().pool_cache_cap as u64;
        sum(&[held, fwd_most.max(ce).max(backward_held.saturating_add(bwd_most)), pool])
    }

    /// A layer's forward over a step: the row-wise part over `t` rows and
    /// the mixer over each sequence (the whole layer's, with `lens` of one).
    fn layer_fwd_step_bytes(&self, l: &Layer, t: u32, mm: GemmOperands, output: bool, lens: Option<&[u32]>) -> u64 {
        lens.unwrap_or(&[t]).iter().fold(
            self.layer_fwd_bytes(l, t, mm, output)
                .saturating_sub(self.mixer_fwd_bytes(l, t, mm)),
            |a, &n| a.saturating_add(self.mixer_fwd_bytes(l, n, mm)),
        )
    }

    /// A layer's backward (gradients, temporaries) over a step, the mixer's
    /// part over each sequence as [`Self::layer_fwd_step_bytes`].
    fn layer_bwd_step_bytes(&self, l: &Layer, t: u32, mm: GemmOperands, lens: Option<&[u32]>) -> (u64, u64) {
        let (grads, temps) = self.layer_bwd_bytes(l, t, mm);
        let (mg, mt) = self.mixer_bwd_bytes(l, t, mm);
        lens.unwrap_or(&[t])
            .iter()
            .fold((grads.saturating_sub(mg), temps.saturating_sub(mt)), |(g, tm), &n| {
                let (ng, nt) = self.mixer_bwd_bytes(l, n, mm);
                (g.saturating_add(ng), tm.saturating_add(nt))
            })
    }

    /// One layer's forward intermediates, with its GEMMs' temporaries;
    /// `output` adds the `down` projection's product (its stream out is the
    /// next layer's input, which [`Self::step_bytes`] holds).
    ///
    /// Under bf16 operands the bound over-counts by one bf16 `[rows, hidden]`
    /// copy per sequence (the mixer's `w_in` GEMM is charged a copy of `x1`,
    /// but reads the layer's bf16 twin, which the gate/up charge covers).
    /// That is safe: an over-count only raises a bound whose purpose is to
    /// refuse a step that might not fit, and the exact tests pin the
    /// difference rather than letting it grow unnoticed.
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
        b.push(self.mixer_fwd_bytes(layer, t, mm));
        b.iter().fold(0, |a, &x| a.saturating_add(x))
    }

    /// [`Self::layer_fwd_bytes`]'s mixer part on `t` rows: a ragged step
    /// runs it once per sequence, each at its own length.
    fn mixer_fwd_bytes(&self, layer: &Layer, t: u32, mm: GemmOperands) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let h = cfg.hidden as usize;
        let wd = self.embed.dtype;
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold);
        let mut b = Vec::new();
        match &layer.mixer {
            Mixer::Gdn(_) => {
                let g = cfg.gdn;
                let (hv, dv) = (g.v_heads() as usize, g.v_dim() as usize);
                let ckpt = single_seq_gdn_dims(&g, t).checkpoint_shape();
                b.extend([
                    f(tu * g.width() as usize), // proj
                    mm.nn_scratch_bytes(tu, g.width() as usize, h, wd),
                    f(tu * g.conv_dim() as usize), // conv: q, k, v
                    f(tu * hv),
                    f(tu * hv), // g, beta
                    f(ckpt.iter().product()),
                    f(tu * hv * dv), // o
                    f(tu * hv * dv), // y
                    mm.nn_scratch_bytes(tu, h, hv * dv, wd),
                    // The chunked kernel's normalized q and k, decay, beta and
                    // C x C products, made and freed inside the forward call.
                    gdn_train_forward_transient_bytes(single_seq_gdn_dims(&g, t), g.k_heads()),
                ]);
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
    ///
    /// Two deliberate over-counts, both safe for the same reason (they only
    /// raise a refuse-if-it-might-not-fit bound, and the exact tests pin
    /// each): every gradient is charged, though a step into an f32 bank
    /// writes its matrices in place; and under bf16 operands each GEMM is
    /// charged the bf16 copy of an operand whose twin the backward already
    /// holds (the twin is allocated once).
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
        let (mg, mt) = self.mixer_bwd_bytes(layer, t, mm);
        grads.push(mg);
        temps.push(mt);
        let total = |v: &[u64]| v.iter().fold(0u64, |a, &b| a.saturating_add(b));
        (total(&grads), total(&temps))
    }

    /// [`Self::layer_bwd_bytes`]'s mixer part on `t` rows (gradients,
    /// temporaries): a ragged step runs it once per sequence.
    fn mixer_bwd_bytes(&self, layer: &Layer, t: u32, mm: GemmOperands) -> (u64, u64) {
        let (cfg, tu) = (&self.cfg, t as usize);
        let h = cfg.hidden as usize;
        let tc = self.rt.has_tensorops();
        let wd = self.embed.dtype;
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.saturating_mul(4), BufferKind::Cold);
        let mut grads = Vec::new();
        let mut temps = Vec::new();
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
        self.rowwise_scratch_bytes(t)
            .saturating_add(self.mixer_scratch_bytes(t))
    }

    /// [`Self::scratch_bytes`]'s row-wise buffers (a ragged step's packed
    /// scratch, which holds the per-sequence mixers' `dresid` and `dx` rows).
    fn rowwise_scratch_bytes(&self, t: u32) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.max(1).saturating_mul(4), BufferKind::Cold);
        [
            2 * f(tu * h), // dresid, dx
            3 * f(tu * i), // d_mid, d_gate, d_up
            f(rms_norm_bwd_part_len(t, cfg.hidden)),
        ]
        .iter()
        .fold(0, |a, &x| a.saturating_add(x))
    }

    /// [`Self::scratch_bytes`]'s mixer scratch on `t` rows: a ragged step
    /// holds one per sequence, each at its own length.
    fn mixer_scratch_bytes(&self, t: u32) -> u64 {
        let (cfg, tu) = (&self.cfg, t as usize);
        let f = |n: usize| GpuRuntime::allocated_bytes_for(n.max(1).saturating_mul(4), BufferKind::Cold);
        let has = |k: crate::qwen35_model::LayerKind| cfg.layers.contains(&k);
        let mut b: Vec<u64> = Vec::new();
        if has(crate::qwen35_model::LayerKind::LinearAttention) {
            let g = cfg.gdn;
            let (hv, dv) = (g.v_heads() as usize, g.v_dim() as usize);
            b.extend([
                2 * f(tu * hv * dv), // dy, d_o
                f(tu * g.width() as usize),
                f(tu * g.conv_dim() as usize), // d_conv: dq, dk, dv
                2 * f(tu * hv),                // dg, dbeta
                GdnTrainWorkspace::allocated_bytes_for(single_seq_gdn_dims(&g, t)),
                f(gated_rms_norm_bwd_part_len(t, g.v_heads(), g.v_dim())),
                f(gdn_gates_bwd_part_len(t, g.v_heads())),
                f(conv1d_silu_bwd_part_len(1, t, g.conv_dim(), cfg.conv_kernel)),
            ]);
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
        // Exact f32: every operand is f32 (a bf16 weight chunk widened in
        // the workspace). bf16 operands: every operand reaches its GEMM
        // already bf16 (rounded once into the workspace, or the bf16
        // weight itself), so the GEMMs copy nothing.
        let per_chunk = |w: usize| match mm {
            GemmOperands::Bf16 => 0,
            GemmOperands::ExactF32 => [
                2 * mm.nt_scratch_bytes(tc, n, w, h, DType::F32),
                mm.nn_scratch_bytes(n, h, w, DType::F32),
                mm.tn_scratch_bytes(tc, w, h, n),
            ]
            .iter()
            .fold(0u64, |a, &b| a.saturating_add(b)),
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
            CeWorkspace::allocated_bytes_for(t, cfg.hidden, chunk, self.embed.dtype, mm),
            gemms,
            GpuRuntime::allocated_bytes_for(n * h * 4, BufferKind::Cold),
            // `cross_entropy_rows` stages its row and target ids through two
            // `[n]` u32 buffers (`upload_u32`), freed with the call.
            2 * GpuRuntime::allocated_bytes_for(n * 4, BufferKind::Cold),
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
        self.residual_into(resid, y, w_out, &out, mm)?;
        Ok(out)
    }

    /// [`Self::residual`] into `out` (a view of a larger stream, in a ragged
    /// step: the rows of one sequence).
    fn residual_into(
        &self,
        resid: &Tensor,
        y: &Tensor,
        w_out: &Tensor,
        out: &Tensor,
        mm: GemmOperands,
    ) -> Result<(), String> {
        deliver(&self.rt, &[tensor_part(resid)], &[tensor_part(out)], false)?;
        mm.nn_acc(y, w_out, out)
    }

    /// One layer's forward from `resid_in`, keeping what its backward reads;
    /// with `output`, also the residual stream out of it (a recomputation
    /// needs only the former, and skips the `down` projection's product).
    /// `attn_ws` is the step's attention workspace (an attention layer needs it).
    /// The layer order is inference's (`Qwen35Model::layer`) on the training
    /// kernels; `tests/qwen35_train.rs`
    /// (`train_forward_hidden_states_are_the_inference_forwards`) pins the
    /// two to the same hidden states.
    #[allow(clippy::too_many_arguments)]
    fn train_layer_forward(
        &self,
        layer: &Layer,
        resid_in: Tensor,
        t: u32,
        output: bool,
        mm: GemmOperands,
        attn_ws: Option<&AttnTrainWorkspace>,
        ragged: Option<&RaggedRef<'_>>,
    ) -> Result<(Saved, Option<Tensor>), String> {
        // Test hook (`fail_layer_after_for_test`): off unless a test sets it.
        // A step's forward runs each layer once and its backward rebuilds
        // each, so a count past the layers fails a rebuild.
        self.take_layer_fault()?;
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (tu, h, i) = (t as usize, cfg.hidden as usize, cfg.intermediate as usize);
        let x1 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_in, &layer.input_norm, &x1, t)?;
        let x1_16 = bf16_twin(mm, &x1)?;
        let (mixer, resid_mid) = match (&layer.mixer, ragged) {
            (mixer, Some(r)) => self.ragged_mixer_forward(mixer, op(&x1, &x1_16), &resid_in, mm, r)?,
            (Mixer::Gdn(w), None) => {
                let s = self.gdn_forward(w, op(&x1, &x1_16), t, mm)?;
                let r = self.residual(&resid_in, op(&s.y, &s.y16), &w.w_out, t, mm)?;
                (SavedMixer::Gdn(Box::new(s)), r)
            }
            (Mixer::Attn(w), None) => {
                let ws = attn_ws.ok_or("Qwen35Model::train_step: an attention layer without the step's workspace")?;
                let s = self.attn_forward(w, op(&x1, &x1_16), t, mm, ws)?;
                let r = self.residual(&resid_in, op(&s.y, &s.y16), &w.w_out, t, mm)?;
                (SavedMixer::Attn(s), r)
            }
        };
        let x2 = tensor(rt, &[tu, h])?;
        self.norm_f32(&resid_mid, &layer.post_norm, &x2, t)?;
        let x2_16 = bf16_twin(mm, &x2)?;
        let (m_gate, m_up, m_mid) = (tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?, tensor(rt, &[tu, i])?);
        mm.nn(op(&x2, &x2_16), &layer.gate, &m_gate)?;
        mm.nn(op(&x2, &x2_16), &layer.up, &m_up)?;
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
                x1_16,
                mixer,
                resid_mid,
                x2,
                x2_16,
                m_gate,
                m_up,
                m_mid,
            },
            resid_out,
        ))
    }

    /// A ragged step's mixer and its residual: each packed sequence's rows
    /// of `xs` (the norm's output, or its bf16 twin) through the mixer alone,
    /// as a step on that sequence would run it (positions from 0, zero GDN
    /// and convolution state, attention over its own rows), then added to its
    /// rows of `resid_in`. Returns each sequence's saved state and the packed
    /// residual stream after the mixer.
    fn ragged_mixer_forward(
        &self,
        mixer: &Mixer,
        xs: &Tensor,
        resid_in: &Tensor,
        mm: GemmOperands,
        r: &RaggedRef<'_>,
    ) -> Result<(SavedMixer, Tensor), String> {
        let (rt, h) = (&self.rt, self.cfg.hidden as usize);
        let total: usize = r.spans.iter().map(|&(_, n)| n).sum();
        let out = tensor(rt, &[total, h])?;
        let mut saved = Vec::with_capacity(r.spans.len());
        for (k, &(start, n)) in r.spans.iter().enumerate() {
            let nu = n as u32;
            let x_k = xs.view(&[n, h], start * h);
            let rin = resid_in.view(&[n, h], start * h);
            let out_k = out.view(&[n, h], start * h);
            let sv = match mixer {
                Mixer::Gdn(w) => {
                    let s = self.gdn_forward(w, &x_k, nu, mm)?;
                    self.residual_into(&rin, op(&s.y, &s.y16), &w.w_out, &out_k, mm)?;
                    SavedMixer::Gdn(Box::new(s))
                }
                Mixer::Attn(w) => {
                    let ws = r.attn_ws.get(k).copied().flatten().ok_or(
                        "Qwen35Model::train_step_ragged_into: an attention layer without a sequence's workspace",
                    )?;
                    let s = self.attn_forward(w, &x_k, nu, mm, ws)?;
                    self.residual_into(&rin, op(&s.y, &s.y16), &w.w_out, &out_k, mm)?;
                    SavedMixer::Attn(s)
                }
            };
            saved.push(sv);
        }
        Ok((SavedMixer::Ragged(saved), out))
    }

    /// A ragged step's mixer backward from the packed `dresid`: each
    /// sequence's rows through [`Self::gdn_backward`] or
    /// [`Self::attn_backward`] on a scratch of its own length, its weight
    /// gradients delivered into the layer's bank as they are made (the first
    /// sequence over the bank unless `add`, every later one added), and its
    /// input gradient into its rows of the packed `dx` (the scratch holds views of both). Returns the last sequence's
    /// gradients, already delivered.
    fn ragged_mixer_backward(
        &self,
        layer: &Layer,
        s: &Saved,
        mm: GemmOperands,
        bank: Option<(&LayerGrads, bool)>,
        rs: &mut RaggedScratch,
    ) -> Result<MixerGrads, String> {
        const MISMATCH: &str = "Qwen35Model::train_step_ragged_into: a layer's saved state or bank is not its mixer's";
        let (rt, h) = (&self.rt, self.cfg.hidden as usize);
        let (bank_layer, add) = bank.ok_or(MISMATCH)?;
        let SavedMixer::Ragged(saved) = &s.mixer else {
            return Err(MISMATCH.into());
        };
        let x1 = op(&s.x1, &s.x1_16);
        let mut last = None;
        for (k, (&(start, n), sv)) in rs.spans.iter().zip(saved).enumerate() {
            let sck = rs.scratch.get_mut(k).ok_or(MISMATCH)?;
            let add_k = add || k > 0;
            let x_k = x1.view(&[n, h], start * h);
            let g = match (&layer.mixer, sv, &bank_layer.mixer) {
                (Mixer::Gdn(w), SavedMixer::Gdn(svk), MixerGrads::Gdn(b)) => {
                    MixerGrads::Gdn(self.gdn_backward(w, svk, &x_k, sck, mm, Some((b, add_k)))?)
                }
                (Mixer::Attn(w), SavedMixer::Attn(svk), MixerGrads::Attn(b)) => {
                    MixerGrads::Attn(self.attn_backward(w, svk, &x_k, sck, mm, Some((b, add_k)))?)
                }
                _ => return Err(MISMATCH.into()),
            };
            deliver(rt, &g.parts(), &bank_layer.mixer.parts(), add_k)?;
            last = Some(g);
        }
        last.ok_or_else(|| MISMATCH.to_string())
    }

    fn gdn_forward(&self, w: &GdnWeights, x1: &Tensor, t: u32, mm: GemmOperands) -> Result<SavedGdn, String> {
        let (rt, g) = (&self.rt, self.cfg.gdn);
        let (tu, hv, dv) = (t as usize, g.v_heads() as usize, g.v_dim() as usize);
        let proj = tensor(rt, &[tu, g.width() as usize])?;
        mm.nn(x1, &w.w_in, &proj)?;
        let conv = tensor(rt, &[tu, g.conv_dim() as usize])?;
        qwen35::conv1d_silu(
            rt,
            Cols::dense(&proj.buffer, g.width()),
            &w.conv_w,
            self.cfg.conv_kernel,
            StateIn::Zero,
            &conv.buffer,
            None,
            1,
            t,
            g.conv_dim(),
        )?;
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
        let dims = single_seq_gdn_dims(&g, t);
        let ckpt = tensor(rt, &dims.checkpoint_shape())?;
        let o = tensor(rt, &[1, tu, hv, dv])?;
        let qkv = g.conv_qkv(&conv.buffer);
        let win = |off| GdnWindow {
            tensor: &conv,
            ld: qkv.ld,
            off,
        };
        let inputs = GdnTrainWindowedInputs {
            q: win(qkv.q_off),
            k: win(qkv.k_off),
            v: win(qkv.v_off),
            key_heads: g.k_heads(),
            g: &gt,
            beta: &beta,
            s0: None,
        };
        gdn_train_forward_windowed(rt, dims, inputs, &o, None, &ckpt)?;
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
        let y16 = bf16_twin(mm, &y)?;
        Ok(SavedGdn {
            proj,
            conv,
            g: gt,
            beta,
            ckpt,
            o,
            y,
            y16,
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
        let y16 = bf16_twin(mm, &y)?;
        Ok(SavedAttn {
            proj,
            q,
            k,
            v,
            o,
            lse,
            y,
            y16,
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
        ragged: Option<&mut RaggedScratch>,
    ) -> Result<LayerGrads, String> {
        let (rt, cfg) = (&self.rt, &self.cfg);
        let (t, h, i) = (sc.mx.t, cfg.hidden, cfg.intermediate);
        let hu = h as usize;
        let eps = cfg.rms_norm_eps;

        // MLP: resid_out = resid_mid + swiglu(x2 @ gate, x2 @ up) @ down.
        let into = |pick: fn(&LayerGrads) -> &Tensor| bank.map(|(b, add)| (pick(b), add));
        let dresid16 = bf16_twin(mm, &sc.mx.dresid)?;
        let down = self.weight_grad(mm, &s.m_mid, op(&sc.mx.dresid, &dresid16), into(|b| &b.down))?;
        mm.nt(op(&sc.mx.dresid, &dresid16), &layer.down, &sc.d_mid)?;
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
        let (d_gate16, d_up16) = (bf16_twin(mm, &sc.d_gate)?, bf16_twin(mm, &sc.d_up)?);
        let x2 = op(&s.x2, &s.x2_16);
        let gate = self.weight_grad(mm, x2, op(&sc.d_gate, &d_gate16), into(|b| &b.gate))?;
        let up = self.weight_grad(mm, x2, op(&sc.d_up, &d_up16), into(|b| &b.up))?;
        // dx = d_gate @ gate^T + d_up @ up^T, the second added in its GEMM.
        mm.nt(op(&sc.d_gate, &d_gate16), &layer.gate, &sc.mx.dx)?;
        mm.nt_acc(op(&sc.d_up, &d_up16), &layer.up, &sc.mx.dx)?;
        let post_norm = vec_grad(rt, hu, bank.map(|(b, add)| (&b.post_norm, add)))?;
        rms_norm_bwd(
            rt,
            &s.resid_mid.buffer,
            &layer.post_norm,
            &sc.mx.dx.buffer,
            &sc.mx.dresid.buffer,
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
            _ if ragged.is_some() => {
                let rs = ragged.ok_or(MISMATCH)?;
                self.ragged_mixer_backward(layer, s, mm, bank, rs)?
            }
            (Mixer::Gdn(w), SavedMixer::Gdn(sv), into) => {
                let into = match into {
                    Some((MixerGrads::Gdn(g), add)) => Some((g, add)),
                    Some(_) => return Err(MISMATCH.into()),
                    None => None,
                };
                MixerGrads::Gdn(self.gdn_backward(w, sv, op(&s.x1, &s.x1_16), &mut sc.mx, mm, into)?)
            }
            (Mixer::Attn(w), SavedMixer::Attn(sv), into) => {
                let into = match into {
                    Some((MixerGrads::Attn(g), add)) => Some((g, add)),
                    Some(_) => return Err(MISMATCH.into()),
                    None => None,
                };
                MixerGrads::Attn(self.attn_backward(w, sv, op(&s.x1, &s.x1_16), &mut sc.mx, mm, into)?)
            }
            _ => return Err(MISMATCH.into()),
        };
        let input_norm = vec_grad(rt, hu, bank.map(|(b, add)| (&b.input_norm, add)))?;
        rms_norm_bwd(
            rt,
            &s.resid_in.buffer,
            &layer.input_norm,
            &sc.mx.dx.buffer,
            &sc.mx.dresid.buffer,
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

    /// The GDN mixer's backward from `sc.mx.dresid`; leaves the gradient of its
    /// input `x1` in `sc.mx.dx`.
    /// `bank` is the layer's `(w_in, w_out)` bank gradients (and whether to
    /// add into them), as [`Self::train_layer_backward`] takes them.
    fn gdn_backward(
        &self,
        w: &GdnWeights,
        s: &SavedGdn,
        x1: &Tensor,
        sc: &mut MixerScratch,
        mm: GemmOperands,
        bank: Option<(&GdnGrads, bool)>,
    ) -> Result<GdnGrads, String> {
        let (rt, cfg, g) = (&self.rt, &self.cfg, self.cfg.gdn);
        let t = sc.t;
        let gs = sc.gdn.as_ref().ok_or("Qwen35Model::train_step: no GDN scratch")?;
        // `dresid` has had the post-norm's gradient added since the MLP's twin.
        let dresid16 = bf16_twin(mm, &sc.dresid)?;
        let w_out = self.weight_grad(
            mm,
            op(&s.y, &s.y16),
            op(&sc.dresid, &dresid16),
            bank.map(|(b, add)| (&b.w_out, add)),
        )?;
        mm.nt(op(&sc.dresid, &dresid16), &w.w_out, &gs.dy)?;
        // y = gated_rms_norm(o, z): d_o, and dz into the projection gradient.
        let norm_w = vec_grad(rt, g.v_dim() as usize, bank.map(|(b, add)| (&b.norm_w, add)))?;
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
        let dims = single_seq_gdn_dims(&g, t);
        let qkv = g.conv_qkv(&s.conv.buffer);
        let dqkv = g.conv_qkv(&gs.d_conv.buffer);
        let ld = qkv.ld;
        let win = |tensor, off| GdnWindow { tensor, ld, off };
        gdn_train_backward_windowed(
            rt,
            dims,
            GdnTrainWindowedInputs {
                q: win(&s.conv, qkv.q_off),
                k: win(&s.conv, qkv.k_off),
                v: win(&s.conv, qkv.v_off),
                key_heads: g.k_heads(),
                g: &s.g,
                beta: &s.beta,
                s0: None,
            },
            &s.ckpt,
            &gs.d_o,
            None,
            &gs.ws,
            // dq, dk (summed over each key head's value heads) and dv land in
            // the conv output's gradient, in its layout.
            GdnTrainWindowedGrads {
                dq: win(&gs.d_conv, dqkv.q_off),
                dk: win(&gs.d_conv, dqkv.k_off),
                dv: win(&gs.d_conv, dqkv.v_off),
                dg: &gs.dg,
                dbeta: &gs.dbeta,
                ds0: None,
            },
        )?;
        // The gates' logits (a, b columns) and their parameters.
        let (a_log, dt_bias) = (
            vec_grad(rt, g.v_heads() as usize, bank.map(|(b, add)| (&b.a_log, add)))?,
            vec_grad(rt, g.v_heads() as usize, bank.map(|(b, add)| (&b.dt_bias, add)))?,
        );
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
        // The conv into the projection's qkv columns.
        let conv_w = vec_grad(
            rt,
            (g.conv_dim() * cfg.conv_kernel) as usize,
            bank.map(|(b, add)| (&b.conv_w, add)),
        )?;
        conv1d_silu_bwd(
            rt,
            Cols::dense(&s.proj.buffer, g.width()),
            &w.conv_w,
            cfg.conv_kernel,
            Cols::dense(&gs.d_conv.buffer, g.conv_dim()),
            Cols::dense(dproj, g.width()),
            &conv_w,
            &gs.conv_part,
            1,
            t,
            g.conv_dim(),
        )?;
        let dproj16 = bf16_twin(mm, &gs.dproj)?;
        let w_in = self.weight_grad(mm, x1, op(&gs.dproj, &dproj16), bank.map(|(b, add)| (&b.w_in, add)))?;
        mm.nt(op(&gs.dproj, &dproj16), &w.w_in, &sc.dx)?;
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
        sc: &mut MixerScratch,
        mm: GemmOperands,
        bank: Option<(&AttnGrads, bool)>,
    ) -> Result<AttnGrads, String> {
        let (rt, cfg, a) = (&self.rt, &self.cfg, self.cfg.attn);
        let t = sc.t;
        let qd = a.q_heads() * a.head_dim();
        let asc = sc
            .attn
            .as_ref()
            .ok_or("Qwen35Model::train_step: no attention scratch")?;
        // `dresid` has had the post-norm's gradient added since the MLP's twin.
        let dresid16 = bf16_twin(mm, &sc.dresid)?;
        let w_out = self.weight_grad(
            mm,
            op(&s.y, &s.y16),
            op(&sc.dresid, &dresid16),
            bank.map(|(b, add)| (&b.w_out, add)),
        )?;
        mm.nt(op(&sc.dresid, &dresid16), &w.w_out, &asc.dy)?;
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
        let (q_norm, k_norm) = (
            vec_grad(rt, a.head_dim() as usize, bank.map(|(b, add)| (&b.q_norm, add)))?,
            vec_grad(rt, a.head_dim() as usize, bank.map(|(b, add)| (&b.k_norm, add)))?,
        );
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
        let dproj16 = bf16_twin(mm, &asc.dproj)?;
        let w_in = self.weight_grad(mm, x1, op(&asc.dproj, &dproj16), bank.map(|(b, add)| (&b.w_in, add)))?;
        mm.nt(op(&asc.dproj, &dproj16), &w.w_in, &sc.dx)?;
        Ok(AttnGrads {
            w_in,
            w_out,
            q_norm,
            k_norm,
        })
    }
}

// The peak-footprint reader the bench bins and tests/ share, for the 2B test's
// closing line.
#[cfg(test)]
#[path = "bin/common/footprint.rs"]
mod footprint;

#[cfg(test)]
mod tests {
    //! Whether `train_step`'s gradients are the gradients of tessl's own
    //! forward, where they differ from transformers': central differences of
    //! the loss along the direction in which the two gradients disagree.

    use std::path::Path;

    use super::*;
    use crate::npy::read_npy;
    use crate::qwen35_adamw::{MomentStorage, UpdateRule};
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
    /// transformers' gradient in tessl's layout, and the finite-difference
    /// steps to probe it with (whether they resolve is measured, not declared).
    struct FdCase<'a> {
        name: String,
        param: &'a GpuBuffer,
        grad: &'a GpuBuffer,
        n: usize,
        torch: Vec<f64>,
        steps: [f64; 6],
    }

    /// What the finite-difference sweep of one parameter says.
    struct FdVerdict {
        /// Every adjacent Richardson estimate, coarsest h first.
        estimates: Vec<f64>,
        /// The value the bound compares: the mean of the three estimates in
        /// the chosen window (the finest qualifying one when resolved, the
        /// finest window otherwise).
        fd: f64,
        /// Max minus min of the chosen window's estimates when resolved,
        /// else the smallest spread over all windows.
        spread: f64,
        /// True only when every difference and `dn` are finite, `dn > 0`, and
        /// some window of three consecutive estimates spans at most `0.3 |d|`.
        resolved: bool,
    }

    /// Richardson-extrapolate a halving sequence of central differences `fds`
    /// (h, h/2, h/4, ...; at least five, giving at least three estimates) and
    /// decide whether the sweep resolves the parameter at the scale of
    /// `dn` = |d|, the norm of the tessl-minus-torch gradient difference.
    ///
    /// Resolved means three consecutive adjacent Richardson estimates all lie
    /// within 0.3 |d| of each other (max minus min). 0.3 |d| is the width of
    /// the bound the test then applies (|fd - tessl| <= 0.3 |d|): an FD whose
    /// own estimates scatter by more than that cannot place tessl against the
    /// bound, nor separate tessl from torch (a distance of |d|) any finer than
    /// its scatter. Two agreeing estimates are not enough: on the GPU they
    /// agreed within 0.3 |d| at layers.20 conv1d.weight by coincidence (the
    /// successive-difference ratios were 1.96 and 3.7, not 4), and the bound
    /// then failed. Of several qualifying windows the finest-h one is used.
    /// A non-finite difference, or a non-finite or zero `dn`, never resolves.
    fn fd_verdict(fds: &[f64], dn: f64) -> FdVerdict {
        assert!(
            fds.len() >= 5,
            "three Richardson estimates need five central differences, got {}",
            fds.len()
        );
        let mut estimates: Vec<f64> = fds.windows(2).map(|w| (4.0 * w[1] - w[0]) / 3.0).collect();
        // Only the four finest estimates count (the owner's design for the
        // six-step sweep: three windows of three); the coarsest h only feeds
        // the printed differences.
        estimates.drain(..estimates.len().saturating_sub(4));
        let tol = 0.3 * dn;
        let finite = fds.iter().all(|x| x.is_finite()) && dn.is_finite() && dn > 0.0;
        let window = |w: &[f64]| {
            let hi = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lo = w.iter().copied().fold(f64::INFINITY, f64::min);
            (hi - lo, w.iter().sum::<f64>() / 3.0)
        };
        let windows: Vec<(f64, f64)> = estimates.windows(3).map(window).collect();
        let best = windows.iter().rposition(|(s, _)| finite && *s <= tol);
        let resolved = best.is_some();
        let (spread, fd) = match best {
            Some(i) => windows[i],
            None => (
                windows.iter().map(|w| w.0).fold(f64::INFINITY, f64::min),
                windows[windows.len() - 1].1,
            ),
        };
        FdVerdict {
            estimates,
            fd,
            spread,
            resolved,
        }
    }

    #[test]
    fn fd_verdict_resolves_a_clean_h_squared_sequence_and_never_a_noisy_or_nan_one() {
        // fd(h) = g + c h^2: Richardson cancels the h^2 term exactly.
        let clean: Vec<f64> = [1e-2f64, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4]
            .iter()
            .map(|h| -0.88 + 40.0 * h * h)
            .collect();
        let v = fd_verdict(&clean, 3.6e-3);
        assert!(v.resolved, "{:?}", v.estimates);
        assert!((v.fd + 0.88).abs() < 1e-12, "{}", v.fd);
        // The conv1d.weight of layer 20 (|d| 3.636e-3): central differences
        // that swing between -0.875 and -0.924 at fine h, rounding noise.
        let noisy = [-0.875, -0.924, -0.89, -0.91, -0.88, -0.93];
        let v = fd_verdict(&noisy, 3.636e-3);
        assert!(!v.resolved, "{:?}", v.estimates);
        // Only two adjacent estimates agree and a third differs: a coincidence
        // of the kind seen at conv20, so not resolved. Perturbing f[2] moves
        // the first two of the four estimates, leaving the last two exact.
        let mut two = clean.clone();
        two[2] += 0.03;
        let v = fd_verdict(&two, 3.6e-3);
        assert!(
            (v.estimates[2] - v.estimates[3]).abs() <= 0.3 * 3.6e-3,
            "{:?}",
            v.estimates
        );
        assert!(!v.resolved, "{:?}", v.estimates);
        // A coarse outlier does not hide a finer agreeing window.
        let mut late = clean.clone();
        late[1] += 0.03;
        assert!(fd_verdict(&late, 3.6e-3).resolved);
        // A NaN anywhere, even one the estimates would not otherwise show.
        for i in 0..6 {
            let mut f = clean.clone();
            f[i] = f64::NAN;
            assert!(!fd_verdict(&f, 3.6e-3).resolved, "NaN at {i}");
        }
        let mut f = clean.clone();
        f[0] = f64::INFINITY;
        assert!(!fd_verdict(&f, 3.6e-3).resolved);
        assert!(!fd_verdict(&clean, f64::NAN).resolved);
        assert!(!fd_verdict(&clean, 0.0).resolved);
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

    fn pooled(b: &GpuBuffer) -> u64 {
        GpuRuntime::allocated_bytes_for(b.nbytes(), BufferKind::Cold)
    }

    fn pooled_t(t: &Tensor) -> u64 {
        pooled(&t.buffer)
    }

    /// The pool-rounded bytes a [`MixerScratch`] holds of its own: the
    /// buffers `mixer_scratch` made, read off the allocated tensors, not its
    /// `dresid`/`dx` (the caller's) nor the attention workspace (the step's,
    /// counted with what the step holds throughout). Both destructurings are
    /// exhaustive: a field added to the scratch fails to compile here until
    /// it is counted. The GDN workspace has no instance size; its own static
    /// `allocated_bytes_for` stands in for it.
    fn mixer_held(model: &Qwen35Model, m: &MixerScratch) -> u64 {
        let MixerScratch {
            t,
            dresid: _,
            dx: _,
            gdn,
            attn,
        } = m;
        let mut n = 0;
        if let Some(GdnScratch {
            dy,
            d_o,
            dproj,
            d_conv,
            dg,
            dbeta,
            ws: _,
            norm_part,
            gates_part,
            conv_part,
        }) = gdn
        {
            n += [dy, d_o, dproj, d_conv, dg, dbeta]
                .into_iter()
                .map(pooled_t)
                .sum::<u64>();
            n += [norm_part, gates_part, conv_part].into_iter().map(pooled).sum::<u64>();
            n += GdnTrainWorkspace::allocated_bytes_for(single_seq_gdn_dims(&model.cfg.gdn, *t));
        }
        if let Some(AttnScratch {
            dy,
            d_o,
            dproj,
            dq,
            dk,
            dv,
            ws: _,
            part,
        }) = attn
        {
            n += [dy, dproj].into_iter().map(pooled_t).sum::<u64>();
            n += [d_o, dq, dk, dv, part].into_iter().map(pooled).sum::<u64>();
        }
        n
    }

    /// Every buffer a [`Scratch`] holds, mixers included.
    fn scratch_held(model: &Qwen35Model, s: &Scratch) -> u64 {
        let Scratch {
            mx,
            d_mid,
            d_gate,
            d_up,
            norm_part,
        } = s;
        pooled_t(&mx.dresid)
            + pooled_t(&mx.dx)
            + mixer_held(model, mx)
            + [d_mid, d_gate, d_up].into_iter().map(pooled_t).sum::<u64>()
            + pooled(norm_part)
    }

    fn tiny_model() -> Qwen35Model {
        tiny_model_as(Precision::F32)
    }

    /// [`tiny_model_as`] as [`Qwen35Model::load_tower`] loads it (no packed
    /// LM head): the model a bf16 step can train, which `AdamW` accepts.
    fn tiny_tower_as(precision: Precision) -> Qwen35Model {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
        let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
        let rt = GpuRuntime::new().unwrap();
        let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
        Qwen35Model::load_tower(&rt, &st, "model.", cfg, precision).unwrap()
    }

    fn tiny_model_as(precision: Precision) -> Qwen35Model {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
        let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
        let rt = GpuRuntime::new().unwrap();
        let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
        Qwen35Model::load(&rt, &st, "model.", cfg, precision).unwrap()
    }

    /// The model's runtime allocation total ([`GpuRuntime::requested_alloc_bytes`]),
    /// which counts every pool request at the pool's rounding whether the
    /// freelist or the device served it, so recycled buffers hide nothing.
    fn requested(model: &Qwen35Model) -> u64 {
        model.rt.requested_alloc_bytes()
    }

    /// The three operand lanes a model can train on: an f32 model on exact
    /// f32 and on bf16 operands, and a bf16-stored model (bf16 weights, the
    /// only lane it accepts).
    fn lanes() -> Vec<(&'static str, Precision, GemmOperands)> {
        vec![
            ("f32/exact", Precision::F32, GemmOperands::ExactF32),
            ("f32/bf16-operands", Precision::F32, GemmOperands::Bf16),
            ("bf16-model/bf16-operands", Precision::Bf16, GemmOperands::Bf16),
        ]
    }

    /// One sequence (`None`) or the packed ragged sequences of a step.
    fn shapes() -> Vec<(u32, Option<Vec<usize>>)> {
        vec![
            (17, None),
            (64, None),
            (300, None),
            (22, Some(vec![5, 17])),
            (110, Some(vec![30, 3, 64, 13])),
        ]
    }

    /// What `train_layer_forward` allocated for layer `li` (as the step's
    /// forward or the backward's rebuild calls it: `output` is whether it
    /// makes the stream out), with the saved state it returns. The
    /// attention workspace(s) and the input stream are made before the mark,
    /// as the step makes them.
    fn forward_alloc(
        model: &Qwen35Model,
        li: usize,
        t: u32,
        lens: Option<&[usize]>,
        mm: GemmOperands,
        output: bool,
    ) -> (u64, Saved) {
        let h = model.cfg.hidden as usize;
        let resid_in = tensor(&model.rt, &[t as usize, h]).unwrap();
        let spans: Option<Vec<(usize, usize)>> = lens.map(|l| {
            l.iter()
                .scan(0, |s, &n| {
                    let r = (*s, n);
                    *s += n;
                    Some(r)
                })
                .collect()
        });
        let (single_ws, ragged_ws): (_, Vec<_>) = match &spans {
            None => (model.attn_workspace(t).unwrap(), Vec::new()),
            Some(sp) => (
                None,
                sp.iter()
                    .map(|&(_, n)| model.attn_workspace(n as u32).unwrap())
                    .collect(),
            ),
        };
        let refs: Vec<Option<&AttnTrainWorkspace>> = ragged_ws.iter().map(Option::as_ref).collect();
        let rr = spans.as_deref().map(|spans| RaggedRef { spans, attn_ws: &refs });
        let before = requested(model);
        let (saved, _out) = model
            .train_layer_forward(
                &model.layers[li],
                resid_in,
                t,
                output,
                mm,
                single_ws.as_ref(),
                rr.as_ref(),
            )
            .unwrap();
        (requested(model) - before, saved)
    }

    /// The sequence lengths a step's mixers run over (one, unless ragged).
    fn seq_lens(t: u32, lens: Option<&[usize]>) -> Vec<u32> {
        lens.map_or(vec![t], |l| l.iter().map(|&n| n as u32).collect())
    }

    /// The pool bytes of a bf16 `[n, hidden]` tensor.
    fn bf16_rows(model: &Qwen35Model, n: u32) -> u64 {
        GpuRuntime::allocated_bytes_for(n as usize * model.cfg.hidden as usize * 2, BufferKind::Cold)
    }

    /// Signed `got - bound` a forward's cell must show, from the two
    /// documented places the bound and the allocators part (the bound
    /// counts the GDN kernel's transients, which an earlier version of
    /// these tests found it did not):
    ///
    /// * `+` the stream out (`output`): the next layer's input, which the
    ///   bound leaves to `step_bytes`'s held part.
    /// * `-` under bf16 operands, one bf16 `[n, hidden]` copy per sequence:
    ///   each mixer's `w_in` GEMM is charged a copy of `x1` (`nn_scratch_bytes`)
    ///   but reads the layer's one bf16 twin of `x1`, whose size the
    ///   gate/up charge covers.
    fn forward_slack(model: &Qwen35Model, t: u32, lens: Option<&[usize]>, mm: GemmOperands, output: bool) -> i64 {
        let seqs = seq_lens(t, lens);
        let mut s = 0i64;
        if output {
            s += GpuRuntime::allocated_bytes_for(t as usize * model.cfg.hidden as usize * 4, BufferKind::Cold) as i64;
        }
        if mm == GemmOperands::Bf16 {
            s -= seqs.iter().map(|&n| bf16_rows(model, n) as i64).sum::<i64>();
        }
        s
    }

    /// Every (lane, shape, layer, output) cell: the layer forward allocates
    /// what `layer_fwd_step_bytes` says plus exactly [`forward_slack`].
    #[test]
    fn layer_forward_bound_equals_the_bytes_the_forward_allocates() {
        let mut bad = Vec::new();
        let mut cells = 0;
        for (name, precision, mm) in lanes() {
            let model = tiny_model_as(precision);
            for (t, lens) in shapes() {
                let lens32 = lens.as_ref().map(|l| l.iter().map(|&n| n as u32).collect::<Vec<_>>());
                for li in 0..model.layers.len() {
                    for output in [true, false] {
                        let (got, _saved) = forward_alloc(&model, li, t, lens.as_deref(), mm, output);
                        let bound = model.layer_fwd_step_bytes(&model.layers[li], t, mm, output, lens32.as_deref());
                        let slack = forward_slack(&model, t, lens.as_deref(), mm, output);
                        cells += 1;
                        if got as i64 - bound as i64 != slack {
                            bad.push(format!(
                                "{name} t={t} lens={lens:?} layer {li} output={output}: allocated {got}, \
                                 bound {bound}, expected got-bound {slack}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {cells} cells differ:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// What `train_layer_backward` allocated for layer `li`: its gradients
    /// (fresh, as a bankless single-sequence step makes them; a ragged step
    /// delivers into a bank, here an accumulating one, which still makes them
    /// before adding) and its temporaries. The scratch and the rebuilt
    /// forward's saved state are made before the mark, as the step makes
    /// them.
    fn backward_alloc(model: &Qwen35Model, li: usize, t: u32, lens: Option<&[usize]>, mm: GemmOperands) -> (u64, u64) {
        let h = model.cfg.hidden as usize;
        let layer = &model.layers[li];
        let resid_in = tensor(&model.rt, &[t as usize, h]).unwrap();
        let bank = Qwen35Grads::zeros_like(model).unwrap();
        match lens {
            None => {
                let mut sc = model.scratch(t, model.attn_workspace(t).unwrap(), true).unwrap();
                let ws = sc.mx.attn.as_ref().map(|a| &a.ws);
                let (s, _) = model
                    .train_layer_forward(layer, resid_in, t, false, mm, ws, None)
                    .unwrap();
                let before = requested(model);
                let _g = model.train_layer_backward(layer, &s, &mut sc, mm, None, None).unwrap();
                (requested(model) - before, 0)
            }
            Some(l) => {
                let spans: Vec<(usize, usize)> = l
                    .iter()
                    .scan(0, |s, &n| {
                        let r = (*s, n);
                        *s += n;
                        Some(r)
                    })
                    .collect();
                let mut sc = model.scratch(t, None, false).unwrap();
                let scratch: Vec<MixerScratch> = spans
                    .iter()
                    .map(|&(start, n)| {
                        model
                            .mixer_scratch(
                                n as u32,
                                model.attn_workspace(n as u32).unwrap(),
                                true,
                                sc.mx.dresid.view(&[n, h], start * h),
                                sc.mx.dx.view(&[n, h], start * h),
                            )
                            .unwrap()
                    })
                    .collect();
                let mut rs = RaggedScratch { spans, scratch };
                let refs: Vec<Option<&AttnTrainWorkspace>> =
                    rs.scratch.iter().map(|x| x.attn.as_ref().map(|a| &a.ws)).collect();
                let rr = RaggedRef {
                    spans: &rs.spans,
                    attn_ws: &refs,
                };
                let (s, _) = model
                    .train_layer_forward(layer, resid_in, t, false, mm, None, Some(&rr))
                    .unwrap();
                let before = requested(model);
                let _g = model
                    .train_layer_backward(layer, &s, &mut sc, mm, Some((&bank.layers[li], true)), Some(&mut rs))
                    .unwrap();
                let got = requested(model) - before;
                // The matrices an f32 bank takes in place: once for the MLP's,
                // and per sequence for the mixer's.
                let inplace_of = |m: &Tensor| if m.dtype == DType::F32 { pooled_t(m) } else { 0 };
                let bl = &bank.layers[li];
                let mixer = match &bl.mixer {
                    MixerGrads::Gdn(g) => inplace_of(&g.w_in) + inplace_of(&g.w_out),
                    MixerGrads::Attn(g) => inplace_of(&g.w_in) + inplace_of(&g.w_out),
                };
                let inplace = inplace_of(&bl.down) + inplace_of(&bl.gate) + inplace_of(&bl.up) + mixer * l.len() as u64;
                (got, inplace)
            }
        }
    }

    /// Signed `got - (grads + temps)` a backward cell must show:
    ///
    /// * `-` the f32 bank matrices the GEMMs write in place
    ///   ([`Qwen35Model::weight_grad`]): the bound charges every gradient,
    ///   as a step with a bf16 bank (fresh tensors, delivered after) makes
    ///   them. Only the ragged cells run into a bank (`inplace`, which
    ///   `backward_alloc` reads off the bank's own tensors).
    /// * `-` under bf16 operands, the copies the bound charges per GEMM for
    ///   an activation whose bf16 twin the backward already holds (the twin
    ///   is allocated once, and charged once by the GEMM that makes it
    ///   from the same tensor as a right operand): the `dresid` twin's second
    ///   reader (down `nt`), the saved `x2` twin on gate and up `tn`, the
    ///   `d_gate`/`d_up` twins' `nt` readers, and per sequence the mixer's
    ///   saved `y` twin (`w_out` `tn`), `dresid` twin (`nt`), saved `x1` twin
    ///   (`w_in` `tn`) and `dproj` twin (`nt`).
    fn backward_slack(
        model: &Qwen35Model,
        li: usize,
        t: u32,
        lens: Option<&[usize]>,
        mm: GemmOperands,
        inplace: u64,
    ) -> i64 {
        let cfg = &model.cfg;
        let (h, i) = (cfg.hidden as usize, cfg.intermediate as usize);
        let c = |elems: usize| GpuRuntime::allocated_bytes_for(elems * 2, BufferKind::Cold) as i64;
        let (width, y_cols) = match model.layers[li].mixer {
            Mixer::Gdn(_) => (cfg.gdn.width() as usize, cfg.gdn.value_dim() as usize),
            Mixer::Attn(_) => (
                cfg.attn.width() as usize,
                (cfg.attn.q_heads() * cfg.attn.head_dim()) as usize,
            ),
        };
        let mut s = -(inplace as i64);
        if mm == GemmOperands::Bf16 {
            let tu = t as usize;
            let mut twins = 3 * c(tu * h) + 2 * c(tu * i);
            for n in seq_lens(t, lens) {
                let n = n as usize;
                twins += c(n * y_cols) + 2 * c(n * h) + c(n * width);
            }
            s -= twins;
        }
        s
    }

    /// Every (lane, shape, layer) cell: the layer backward allocates what
    /// `layer_bwd_step_bytes` says (gradients and temporaries) plus exactly
    /// [`backward_slack`].
    #[test]
    fn layer_backward_bound_equals_the_bytes_the_backward_allocates() {
        let mut bad = Vec::new();
        let mut cells = 0;
        for (name, precision, mm) in lanes() {
            let model = tiny_model_as(precision);
            for (t, lens) in shapes() {
                let lens32 = lens.as_ref().map(|l| l.iter().map(|&n| n as u32).collect::<Vec<_>>());
                for li in 0..model.layers.len() {
                    let (got, inplace) = backward_alloc(&model, li, t, lens.as_deref(), mm);
                    let (g, tm) = model.layer_bwd_step_bytes(&model.layers[li], t, mm, lens32.as_deref());
                    let slack = backward_slack(&model, li, t, lens.as_deref(), mm, inplace);
                    cells += 1;
                    if got as i64 - (g + tm) as i64 != slack {
                        bad.push(format!(
                            "{name} t={t} lens={lens:?} layer {li}: allocated {got}, grads {g} + temps {tm}, \
                             expected got-bound {slack}"
                        ));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {cells} cells differ:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// What the cross-entropy phase allocates for `t` supervised rows
    /// ([`Supervise::Rows`] over every row, so its `dh` is made): the
    /// workspace, `dh`, and the GEMMs' temporaries across every vocabulary
    /// chunk. The model's `vocab` is set to `vocab` and the head is a
    /// matching `[vocab, hidden]` tensor of the embedding's dtype, so a
    /// vocabulary past [`CE_CHUNK`] runs several chunks; the hidden rows and
    /// the head gradient are made before the mark (the step holds them).
    fn ce_alloc(model: &mut Qwen35Model, t: u32, vocab: u32, mm: GemmOperands) -> u64 {
        model.cfg.vocab = vocab;
        let (rt, h) = (model.rt.clone(), model.cfg.hidden as usize);
        let head_w = rt
            .alloc_tensor_unzeroed(&[vocab as usize, h], model.embed.dtype)
            .unwrap();
        let xf = tensor(&rt, &[t as usize, h]).unwrap();
        let dw = tensor(&rt, &[vocab as usize, h]).unwrap();
        let rows: Vec<u32> = (0..t).collect();
        let targets: Vec<u32> = (0..t).map(|i| i % vocab).collect();
        let before = requested(model);
        let chunk = CE_CHUNK.min(vocab);
        let ws = CeWorkspace::new(&rt, t, model.cfg.hidden, chunk, head_w.dtype, mm).unwrap();
        let dh = tensor(&rt, &[t as usize, h]).unwrap();
        let grads = CeGrads {
            dh: &dh,
            dw: &dw,
            scale: 1.0,
        };
        let hidden = CeHidden { rows: &xf, off: 0 };
        cross_entropy_rows(
            &rt,
            hidden,
            &head_w,
            &rows,
            &targets,
            Reduction::Sum,
            mm,
            &ws,
            Some(grads),
        )
        .unwrap();
        requested(model) - before
    }

    /// The cross-entropy allocates exactly `ce_bytes(batched)` (the bound
    /// counts its two `u32` staging buffers, which an earlier version of this
    /// test found it did not), on one chunk and on several (a
    /// vocabulary of one chunk, a chunk and a bit, two and a bit). Unbatched
    /// the bound counts one chunk's GEMM temporaries, so it equals the batched
    /// one on a single chunk and never exceeds it. (The exact lane's GEMM
    /// temporaries are zero on a TensorOps device and the bf16 lane's are
    /// not counted per chunk, so this test asserts no batched/unbatched gap.)
    #[test]
    fn cross_entropy_bound_equals_the_bytes_the_cross_entropy_allocates() {
        let mut bad = Vec::new();
        let mut cells = 0;
        for (name, precision, mm) in lanes() {
            let mut model = tiny_model_as(precision);
            for vocab in [64u32, CE_CHUNK, CE_CHUNK + 40, 2 * CE_CHUNK + 40] {
                for t in [2u32, 17, 64, 300] {
                    let got = ce_alloc(&mut model, t, vocab, mm);
                    let bound = model.ce_bytes(t, mm, true);
                    let single = model.ce_bytes(t, mm, false);
                    cells += 1;
                    let chunks = vocab.div_ceil(CE_CHUNK);
                    if got != bound || single > bound || (chunks == 1 && single != bound) {
                        bad.push(format!(
                            "{name} vocab={vocab} t={t}: allocated {got}, batched bound {bound}, unbatched {single}"
                        ));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} of {cells} cells differ:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// Prints the step bounds for the real Qwen3.5-2B configuration at the
    /// sizes a trainer uses, from the config's arithmetic alone: the tiny
    /// fixture's runtime and layer shapes stand in for the 2B's (the bound
    /// reads only the config, the layer kinds, the embedding dtype and the
    /// runtime's pool cap). Run with
    /// `cargo test --lib report_2b_step_bounds -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints a report"]
    fn report_2b_step_bounds() {
        let mut model = tiny_model();
        let cfg = Qwen35Config::qwen35_2b().unwrap();
        let mut layers = Vec::new();
        for kind in &cfg.layers {
            let idx = usize::from(*kind == crate::qwen35_model::LayerKind::FullAttention);
            layers.push(tiny_model().layers.swap_remove(idx));
        }
        model.cfg = cfg;
        model.layers = layers;
        for (name, mm) in [("exact", GemmOperands::ExactF32), ("bf16", GemmOperands::Bf16)] {
            for t in [512u32, 2048, 4096, 8192] {
                println!("BOUND {name} single t={t} {}", model.train_step_bytes(t, mm));
            }
            let cu = [0u32, 2048, 4096, 6144, 8192];
            println!(
                "BOUND {name} ragged 4x2048 {}",
                model.train_step_ragged_bytes(&cu, 8192, mm, true).unwrap()
            );
        }
    }

    /// The attention workspace bound is the bytes `AttnTrainWorkspace::new`
    /// allocates, for a step's length and for each ragged sequence's.
    #[test]
    fn attention_workspace_bound_equals_the_bytes_it_allocates() {
        let model = tiny_model();
        for n in [1u32, 2, 5, 17, 64, 110, 300] {
            let dims = model.attn_dims(n);
            let before = requested(&model);
            let _ws = AttnTrainWorkspace::new(&model.rt, dims).unwrap();
            assert_eq!(
                requested(&model) - before,
                AttnTrainWorkspace::allocated_bytes_for(dims),
                "n={n}"
            );
        }
    }

    /// The scratch bounds equal the bytes the allocators made, exactly: the
    /// whole scratch, its row-wise part (the packed scratch of a ragged step)
    /// and each sequence's mixer scratch. A tolerance would let a missing
    /// mirror hide in the pool-cap slack; this does not.
    #[test]
    fn scratch_bounds_equal_the_bytes_the_scratch_allocators_make() {
        let model = tiny_model();
        for t in [2u32, 3, 17, 64, 256, 1000] {
            let ws = model.attn_workspace(t).unwrap();
            let sc = model.scratch(t, ws, true).unwrap();
            assert_eq!(
                scratch_held(&model, &sc),
                model.scratch_bytes(t),
                "scratch_bytes at t={t}"
            );
            let packed = model.scratch(t, None, false).unwrap();
            assert_eq!(
                scratch_held(&model, &packed),
                model.rowwise_scratch_bytes(t),
                "rowwise_scratch_bytes at t={t}"
            );
            let h = model.cfg.hidden as usize;
            let n = t as usize;
            let mx = model
                .mixer_scratch(
                    t,
                    model.attn_workspace(t).unwrap(),
                    true,
                    tensor(&model.rt, &[n, h]).unwrap(),
                    tensor(&model.rt, &[n, h]).unwrap(),
                )
                .unwrap();
            assert_eq!(
                mixer_held(&model, &mx),
                model.mixer_scratch_bytes(t),
                "mixer_scratch_bytes at t={t}"
            );
        }
    }

    /// What a fused step allocates beyond a bank step's working set equals
    /// its bounds, exactly: the top bank, each layer's scratch (whose f32
    /// buffers are the gradients `layer_bwd_bytes` already charges, and whose
    /// bf16 ones the fused bound adds) and the slot tables of each layer's
    /// update and of the tail's. Read off the runtime's request counter, so
    /// recycled buffers hide nothing.
    #[test]
    fn fused_bounds_equal_the_bytes_the_fused_step_allocates() {
        for (name, precision, mm) in lanes() {
            let model = tiny_tower_as(precision);
            let config = if precision == Precision::F32 {
                AdamWConfig::F32
            } else {
                AdamWConfig {
                    update: UpdateRule::bf16_default(3),
                    moments: MomentStorage::F32,
                }
            };
            let before = requested(&model);
            let top = model.fused_top_bank().unwrap();
            assert_eq!(
                requested(&model) - before,
                model.fused_top_bank_bytes(),
                "{name}: the top bank"
            );
            let mut state = AdamW::with_config(&model, config).unwrap();
            let wd = vec![0.0f32; model.parameter_table().unwrap().len()];
            let mut fused = FusedAdamW::new("test", &model, &mut state, &AdamWHyper::default(), &wd).unwrap();
            for (li, layer) in model.layers.iter().enumerate() {
                let before = requested(&model);
                let scratch = model.layer_scratch(layer).unwrap();
                let (in_f32, in_bf16) = model.layer_scratch_bytes(layer);
                assert_eq!(
                    requested(&model) - before,
                    in_f32 + in_bf16,
                    "{name}: layer {li}'s scratch"
                );
                let (grads, _) = model.layer_bwd_bytes(layer, 8, mm);
                if precision == Precision::F32 {
                    assert_eq!(
                        in_f32, grads,
                        "{name}: layer {li}: the f32 scratch is what the bound charges"
                    );
                } else {
                    assert!(
                        in_f32 <= grads,
                        "{name}: layer {li}: f32 scratch {in_f32} over the charge {grads}"
                    );
                }
                let before = requested(&model);
                fused.layer(&model, li, &scratch).unwrap();
                let tables = fused_table_bytes(config, &layer_slots(&model, li, None).unwrap());
                assert_eq!(requested(&model) - before, tables, "{name}: layer {li}'s slot tables");
                assert_eq!(
                    model.fused_layer_bytes(li, config),
                    in_bf16 + tables,
                    "{name}: layer {li}'s fused bound"
                );
            }
            let before = requested(&model);
            fused
                .top(&model, &top.embed, &top.final_norm, top.lm_head.as_ref())
                .unwrap();
            assert_eq!(
                requested(&model) - before,
                model.fused_tail_bytes(config),
                "{name}: the tail's slot tables"
            );
        }
    }

    /// A ragged step's bound: the packed row-wise scratch plus one mixer
    /// scratch per sequence at its own length, as `train_step_ragged_into`
    /// builds them (per-sequence `dresid`/`dx` are views, so not allocated).
    #[test]
    fn ragged_scratch_bound_equals_the_packed_plus_per_sequence_allocations() {
        let model = tiny_model();
        let h = model.cfg.hidden as usize;
        for lens in [vec![5usize, 17, 30], vec![2, 64, 3, 40, 100], vec![256, 2]] {
            let total: usize = lens.iter().sum();
            let sc = model.scratch(total as u32, None, false).unwrap();
            let mut held = scratch_held(&model, &sc);
            let mut start = 0;
            for &n in &lens {
                let mx = model
                    .mixer_scratch(
                        n as u32,
                        model.attn_workspace(n as u32).unwrap(),
                        true,
                        sc.mx.dresid.view(&[n, h], start * h),
                        sc.mx.dx.view(&[n, h], start * h),
                    )
                    .unwrap();
                held += mixer_held(&model, &mx);
                start += n;
            }
            let bound = model.rowwise_scratch_bytes(total as u32)
                + lens.iter().map(|&n| model.mixer_scratch_bytes(n as u32)).sum::<u64>();
            assert_eq!(held, bound, "ragged lens {lens:?}");
        }
    }

    /// The reference directory is `QWEN35_TRAIN_REF_DIR` when set, and a
    /// directory without `ids.npy` is refused naming the variable and the
    /// path (ft-19f4bd16: the in-crate 2B test once read a fixed path and
    /// panicked on a missing `ids.npy`).
    #[test]
    fn reference_dir_honours_the_variable_and_names_it_when_missing() {
        let root = std::env::temp_dir().join(format!("tessl_ref_dir_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let missing = reference_dir_from(Some(root.clone().into_os_string())).unwrap_err();
        assert!(missing.contains("QWEN35_TRAIN_REF_DIR"), "{missing}");
        assert!(missing.contains(&root.display().to_string()), "{missing}");
        std::fs::write(root.join("ids.npy"), b"").unwrap();
        let found = reference_dir_from(Some(root.clone().into_os_string())).unwrap();
        assert_eq!(found, root);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The reference directory, the runtime, the f32 2B model and the
    /// reference ids, shared by the 2B finite-difference tests.
    fn real_2b_setup() -> (std::path::PathBuf, Arc<GpuRuntime>, Qwen35Model, Vec<u32>) {
        let dir = train_reference_dir().unwrap_or_else(|e| panic!("{e}"));
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
        (dir, rt, model, ids)
    }

    /// The model's f32 buffer, tessl's gradient buffer and the element count
    /// for a parameter the 2B finite-difference diagnostics know: a conv1d,
    /// GDN gated-norm, `input_layernorm`, `post_attention_layernorm` or
    /// `model.norm.weight` name. Panics on any other.
    fn fd_param<'a>(model: &'a Qwen35Model, step: &'a TrainStep, name: &str) -> (&'a GpuBuffer, &'a GpuBuffer, usize) {
        let layer = || -> usize {
            name.strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("{name}: no layer index"))
        };
        let hidden = model.cfg.hidden as usize;
        let gdn = |l: usize| match (&model.layers[l].mixer, &step.grads.layers[l].mixer) {
            (Mixer::Gdn(w), MixerGrads::Gdn(g)) => (w, g),
            _ => panic!("{name}: layer {l} is not a GDN layer"),
        };
        if name.ends_with("conv1d.weight") {
            let (w, g) = gdn(layer());
            (
                &w.conv_w,
                &g.conv_w,
                (model.cfg.gdn.conv_dim() * model.cfg.conv_kernel) as usize,
            )
        } else if name.ends_with("linear_attn.norm.weight") {
            let (w, g) = gdn(layer());
            (&w.norm_w, &g.norm_w, 128)
        } else if name.ends_with(".input_layernorm.weight") {
            let l = layer();
            (&model.layers[l].input_norm, &step.grads.layers[l].input_norm, hidden)
        } else if name.ends_with(".post_attention_layernorm.weight") {
            let l = layer();
            (&model.layers[l].post_norm, &step.grads.layers[l].post_norm, hidden)
        } else if name == "model.norm.weight" {
            (&model.final_norm, &step.grads.final_norm, hidden)
        } else {
            panic!("{name}: not a parameter this diagnostic knows")
        }
    }

    /// Diagnostic for ft-1d2ab5b004a2f7bebbd1bf03896428ff: the central
    /// difference's convergence for one parameter of the 2B model along the
    /// same `v = (g_tessl - g_torch) / |d|` as the bound test. The parameter
    /// is `TESSL_FD_PARAM` (default the layer-20 conv weight; a conv1d,
    /// `input_layernorm`, `post_attention_layernorm` or `model.norm.weight`
    /// name). It prints, and asserts only that the parameter is restored.
    #[test]
    #[ignore = "diagnostic: prints the FD convergence for one parameter"]
    fn real_2b_fd_convergence_for_one_parameter() {
        let name =
            std::env::var("TESSL_FD_PARAM").unwrap_or_else(|_| "model.layers.20.linear_attn.conv1d.weight".into());
        let (dir, _rt, model, ids) = real_2b_setup();
        let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let base = loss(&model, &ids);
        let (param, grad, n) = fd_param(&model, &step, &name);
        let g_torch = npy(&dir.join(format!("grad.{name}.npy")));
        let g_tessl: Vec<f64> = grad.read_f32()[..n].iter().map(|&x| f64::from(x)).collect();
        let d: Vec<f64> = g_tessl.iter().zip(&g_torch).map(|(a, b)| a - b).collect();
        let dn = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v: Vec<f64> = d.iter().map(|x| x / dn).collect();
        let dot = |g: &[f64]| g.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let orig = param.read_f32()[..n].to_vec();
        let max_abs = orig.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        eprintln!(
            "{name}: n {n}, |d| {dn:.3e}, tessl.v {:.6e}, torch.v {:.6e}, base loss {base:.9e}, max|x| {max_abs:.3e}, max|v| {:.3e}, |g_tessl| {:.3e}",
            dot(&g_tessl),
            dot(&g_torch),
            v.iter().fold(0.0f64, |a, &b| a.max(b.abs())),
            g_tessl.iter().map(|x| x * x).sum::<f64>().sqrt(),
        );
        // Both perturbations are formed from the f32 original. "f64 once" is
        // the bound test's: x + s*u in f64, rounded to f32 once. "f32
        // arithmetic" rounds s*u to f32 first and adds in f32, so a differing
        // row shows what the second rounding costs.
        let perturb = |s: f64, f64_once: bool| -> Vec<f32> {
            orig.iter()
                .zip(&v)
                .map(|(&x, &u)| {
                    if f64_once {
                        (f64::from(x) + s * u) as f32
                    } else {
                        x + (s * u) as f32
                    }
                })
                .collect()
        };
        let write = |p: &[f32]| {
            let mut all = param.read_f32();
            all[..n].copy_from_slice(p);
            param.write_f32(&all);
        };
        let hs: Vec<f64> = (0..=14)
            .map(|k| 1e-1 / f64::from(1u32 << k))
            .take_while(|&h| h >= 1e-5)
            .collect();
        for f64_once in [true, false] {
            eprintln!(
                "-- perturbation {}",
                if f64_once {
                    "f64 once, rounded once"
                } else {
                    "f32 arithmetic"
                }
            );
            let mut fds: Vec<f64> = Vec::new();
            for &h in &hs {
                let (xp, xm) = (perturb(h, f64_once), perturb(-h, f64_once));
                let unchanged = |p: &[f32]| p.iter().zip(&orig).filter(|(a, b)| a == b).count();
                let (cp, cm) = (unchanged(&xp), unchanged(&xm));
                // The step the f32 parameters really took along v.
                let realised = xp
                    .iter()
                    .zip(&xm)
                    .zip(&v)
                    .map(|((&a, &b), &u)| (f64::from(a) - f64::from(b)) * u)
                    .sum::<f64>()
                    / 2.0;
                write(&xp);
                let up = loss(&model, &ids);
                write(&xm);
                let down = loss(&model, &ids);
                write(&orig);
                let fd = (up - down) / (2.0 * h);
                let fd_real = (up - down) / (2.0 * realised);
                // Observed order from three successive differences:
                // (fd(4h') - fd(2h')) / (fd(2h') - fd(h')) = 2^p.
                let k = fds.len();
                let order = if k >= 2 {
                    let r = (fds[k - 2] - fds[k - 1]) / (fds[k - 1] - fd);
                    format!("diff ratio {r:.3} (order {:.2})", r.abs().log2())
                } else {
                    "-".to_string()
                };
                let rich = fds
                    .last()
                    .map_or("-".to_string(), |p| format!("{:.6e}", (4.0 * fd - p) / 3.0));
                eprintln!(
                    "h {h:.4e}: up-base {:+.3e} down-base {:+.3e} fd {fd:+.6e} fd(realised h) {fd_real:+.6e} (realised/h {:.6}) {order} richardson(2h,h) {rich} unchanged f32 elems +h {cp} -h {cm} of {n}",
                    up - base,
                    down - base,
                    realised / h,
                );
                fds.push(fd);
            }
            assert_eq!(
                loss(&model, &ids).to_bits(),
                base.to_bits(),
                "{name}: the parameter was not restored"
            );
        }
    }

    /// Decorrelation probe for ft-1d2ab5b004a2f7bebbd1bf03896428ff. The
    /// central difference along `v` has a deterministic roughness (~3e-7 in
    /// the loss, rounding flips across the f32 layers). Averaging K central
    /// differences, each taken about a shifted base `x0 + delta_k v` with the
    /// same `h`, removes it only if the roughness decorrelates between nearby
    /// points. For `TESSL_FD_PARAM` (default `model.norm.weight`) at the bound
    /// test's finest `h` for that class and at 2x and 4x it, this prints the
    /// K shifted differences, their mean, std, std/sqrt(K), the lag-1
    /// autocorrelation (in run order and in `delta` order), a disjoint-halves
    /// check and the mean's distance from the unshifted difference. The
    /// unshifted difference is also taken K times and must be bit-identical
    /// (the run itself is deterministic). The `delta` grid is 12 magnitudes in
    /// [h/64, h/4], each used with both signs, so the linear (`f''`) shift
    /// bias cancels and the `h^2` term remains.
    #[test]
    #[ignore = "diagnostic: FD noise decorrelation probe"]
    fn real_2b_fd_noise_decorrelation_probe() {
        const K: usize = 24;
        let name = std::env::var("TESSL_FD_PARAM").unwrap_or_else(|_| "model.norm.weight".into());
        let (dir, _rt, model, ids) = real_2b_setup();
        let step = model.train_step(&ids, GemmOperands::ExactF32).unwrap();
        let base = loss(&model, &ids);
        let (param, grad, n) = fd_param(&model, &step, &name);
        let g_torch = npy(&dir.join(format!("grad.{name}.npy")));
        let g_tessl: Vec<f64> = grad.read_f32()[..n].iter().map(|&x| f64::from(x)).collect();
        let d: Vec<f64> = g_tessl.iter().zip(&g_torch).map(|(a, b)| a - b).collect();
        let dn = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v: Vec<f64> = d.iter().map(|x| x / dn).collect();
        let orig = param.read_f32()[..n].to_vec();
        // The bound test's finest step for this parameter class.
        let h0 = if name.ends_with("layernorm.weight")
            || name == "model.norm.weight"
            || name.ends_with("linear_attn.norm.weight")
        {
            0.0125
        } else {
            3.125e-4
        };
        let write = |s: f64| {
            let p: Vec<f32> = orig
                .iter()
                .zip(&v)
                .map(|(&x, &u)| (f64::from(x) + s * u) as f32)
                .collect();
            let mut all = param.read_f32();
            all[..n].copy_from_slice(&p);
            param.write_f32(&all);
        };
        let mut forwards = 0usize;
        let mut fd_at = |delta: f64, h: f64| -> f64 {
            write(delta + h);
            let up = loss(&model, &ids);
            write(delta - h);
            let down = loss(&model, &ids);
            forwards += 2;
            (up - down) / (2.0 * h)
        };
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        let std = |x: &[f64]| {
            let m = mean(x);
            (x.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
        };
        let lag1 = |x: &[f64]| {
            let m = mean(x);
            let num: f64 = x.windows(2).map(|w| (w[0] - m) * (w[1] - m)).sum();
            num / x.iter().map(|a| (a - m).powi(2)).sum::<f64>()
        };
        let t0 = std::time::Instant::now();
        eprintln!(
            "{name}: n {n}, |d| {dn:.3e}, 0.3|d| {:.3e}, base loss {base:.9e}, K {K}",
            0.3 * dn
        );
        for mult in [1.0, 2.0, 4.0] {
            let h = h0 * mult;
            let unshifted: Vec<f64> = (0..K).map(|_| fd_at(0.0, h)).collect();
            assert!(
                unshifted.iter().all(|x| x.to_bits() == unshifted[0].to_bits()),
                "{name}: h {h}: the unshifted difference is not deterministic: {unshifted:?}"
            );
            let fd0 = unshifted[0];
            // 12 magnitudes in [h/64, h/4], each with both signs, run order
            // alternating + and -.
            let mags: Vec<f64> = (0..K / 2)
                .map(|j| h / 64.0 + (h / 4.0 - h / 64.0) * j as f64 / (K / 2 - 1) as f64)
                .collect();
            let deltas: Vec<f64> = (0..K)
                .map(|k| if k % 2 == 0 { mags[k / 2] } else { -mags[k / 2] })
                .collect();
            let fds: Vec<f64> = deltas.iter().map(|&dl| fd_at(dl, h)).collect();
            let mut by_delta: Vec<(f64, f64)> = deltas.iter().copied().zip(fds.iter().copied()).collect();
            by_delta.sort_by(|a, b| a.0.total_cmp(&b.0));
            let sorted: Vec<f64> = by_delta.iter().map(|p| p.1).collect();
            let (m, sd) = (mean(&fds), std(&fds));
            let se = sd / (K as f64).sqrt();
            let (ra, rb) = (lag1(&fds), lag1(&sorted));
            // Disjoint halves by magnitude-index parity (each holds both signs).
            let half = |par: usize| -> Vec<f64> { (0..K).filter(|k| (k / 2) % 2 == par).map(|k| fds[k]).collect() };
            let (ma, mb) = (mean(&half(0)), mean(&half(1)));
            eprintln!(
                "-- h {h:.5e} (x{mult}), shifts |delta| in [{:.3e}, {:.3e}]",
                h / 64.0,
                h / 4.0
            );
            eprintln!("unshifted fd (x{K}, bit-identical): {fd0:+.6e}");
            for (k, (dl, f)) in deltas.iter().zip(&fds).enumerate() {
                eprintln!("  k {k:2} delta {dl:+.4e} fd {f:+.6e} (fd-unshifted {:+.3e})", f - fd0);
            }
            eprintln!(
                "mean {m:+.6e} std {sd:.3e} std/sqrtK {se:.3e} (single/mean {:.2}) lag1 run-order {ra:+.3} delta-order {rb:+.3}",
                sd / se
            );
            eprintln!(
                "halves (|delta| index parity): mean_a {ma:+.6e} mean_b {mb:+.6e} diff {:+.3e}, expected sd of diff if independent {:.3e}",
                ma - mb,
                sd * (2.0 / (K / 2) as f64).sqrt()
            );
            eprintln!(
                "mean - unshifted {:+.3e} ({:.2} std/sqrtK); 3 std/sqrtK {:.3e} vs 0.3|d| {:.3e}",
                m - fd0,
                (m - fd0) / se,
                3.0 * se,
                0.3 * dn
            );
        }
        write(0.0);
        assert_eq!(
            loss(&model, &ids).to_bits(),
            base.to_bits(),
            "{name}: the parameter was not restored"
        );
        eprintln!(
            "cost: {forwards} forwards for the probes (+1 train_step, 2 base loss), {:.1}s wall",
            t0.elapsed().as_secs_f64()
        );
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
        let (dir, _rt, model, ids) = real_2b_setup();
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
        // Whether a parameter resolves is measured from its own sweep
        // (`fd_verdict`); unresolved ones are reported by name, never passed.
        let mut all_cases: Vec<FdCase<'_>> = cases
            .into_iter()
            .map(|(name, param, grad, n)| {
                let torch = npy(&dir.join(format!("grad.{name}.npy")));
                // The hidden-size norms act as (1 + w) around 1, so a step of
                // 0.05-0.4 along a unit direction moves each of their 2048
                // entries by at most ~0.01 and stays near-linear, while lifting
                // |d| h far above the loss's f32 rounding. The GDN gated norm's
                // 128 weights did not resolve (steps of 0.4-0.05 and 0.2-0.025
                // extrapolated 4e-5 apart, more than its |d| of 3e-5); whether
                // a parameter resolves is now measured by `fd_verdict`.
                let norm = name.ends_with("layernorm.weight") || name == "model.norm.weight";
                let steps = if norm || name.ends_with("linear_attn.norm.weight") {
                    [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
                } else {
                    [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4]
                };
                FdCase {
                    name,
                    param,
                    grad,
                    n,
                    torch,
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
                steps: [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4],
            });
        }
        let mut resolved_names: Vec<String> = Vec::new();
        let mut unresolved_names: Vec<String> = Vec::new();
        for FdCase {
            name,
            param,
            grad,
            n,
            torch: g_torch,
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
            // h and h/2. Six steps give four adjacent estimates; three
            // consecutive ones must agree for the parameter to resolve.
            let FdVerdict {
                estimates,
                fd,
                spread,
                resolved,
            } = fd_verdict(&fds, dn);
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
                resolved_names.push(name.clone());
                assert!(
                    (fd - pt).abs() <= 0.3 * dn,
                    "{name}: tessl's loss moves as {fd:.6e} along v, its gradient says {pt:.6e}"
                );
                assert!(
                    (fd - pt).abs() < (fd - pr).abs(),
                    "{name}: the finite difference is closer to transformers' gradient"
                );
            } else {
                eprintln!(
                    "UNRESOLVED {name}: adjacent Richardson estimates {}; best window of three spans {spread:.3e} > 0.3·|d| {:.3e}",
                    estimates.iter().map(|e| format!("{e:.6e}")).collect::<Vec<_>>().join(", "),
                    0.3 * dn
                );
                unresolved_names.push(name.clone());
            }
        }
        let total = resolved_names.len() + unresolved_names.len();
        eprintln!(
            "{} of {total} parameters resolved; unresolved: {}",
            resolved_names.len(),
            if unresolved_names.is_empty() {
                "none".to_string()
            } else {
                unresolved_names.join(", ")
            }
        );
        assert!(
            !resolved_names.is_empty(),
            "no parameter resolved: the test has no evidence (unresolved: {})",
            unresolved_names.join(", ")
        );
        let peak = super::footprint::peak_footprint().unwrap_or_else(|e| panic!("peak footprint unreadable: {e}"));
        eprintln!(
            "real_2b_gradients_are_those_of_tessls_forward: peak footprint {:.2} GiB (memguard cap 32 GiB)",
            peak as f64 / f64::from(1u32 << 30)
        );
    }
}
