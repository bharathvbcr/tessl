//! Qwen3.5 layer kernels: the gated delta net (GDN), the full-attention
//! extras, and answer-row scoring.
//!
//! transformers runs Qwen3.5's GDN layers on Apple silicon through a pure-torch
//! fallback: a Python loop over every 64-token chunk with a 63-step triangular
//! solve inside it, all in fp32, which is roughly 13-16k tiny kernel launches
//! per forward pass. On CUDA the `fla` library fuses the same work into a few
//! kernels. This module is that fusion for Metal:
//!
//! | step | kernel(s) | replaces |
//! |---|---|---|
//! | fused in-projection | one [`gemm`](fn@crate::gemm) over [`pack_linear_weights_f32`] output | four `nn.Linear` |
//! | conv + SiLU | [`conv1d_silu`] | `causal_conv1d_fn` / `causal_conv1d_update` |
//! | gates, l2norm, q scale, delta rule (prefill) | [`gdn_chunk_forward`] (2 dispatches) | `torch_chunk_gated_delta_rule` |
//! | same, decode / short suffixes | [`gdn_recurrent`] | `torch_recurrent_gated_delta_rule` |
//! | gated norm | [`gated_rms_norm`] | `Qwen3_5RMSNormGated` |
//! | out-projection + residual | [`project_residual`] | `out_proj` + the residual add |
//! | Q/K norm, partial RoPE, K/V cache store | [`attn_qk_norm_rope`] | `q_norm`/`k_norm`/`apply_rotary_pos_emb` |
//! | output gate | [`attn_output_gate`] | `attn_output * sigmoid(gate)` |
//! | answer scoring | [`score_answer_rows`] | the full-vocabulary LM head |
//!
//! # Layout
//!
//! Activations are row-major f32, one row per token (`b * seq + t`). Every
//! input is a column window of a wider row — a [`Cols`] — so the kernels read
//! the fused projection's output in place instead of splitting it into copies.
//! [`GdnProjLayout`] and [`AttnProjLayout`] give the column offsets that
//! [`pack_linear_weights_f32`] produces.
//!
//! # Semantics
//!
//! Each kernel matches `transformers/models/qwen3_5/modeling_qwen3_5.py`, and
//! `tools/msl_emu/check_qwen35.py` checks that against the model code itself
//! (see `docs/qwen35.md`). In particular the GDN decay is
//! `g = -exp(A_log) * softplus(a + dt_bias)` and there is no gate clamp: that is
//! Qwen3.5's gate, and it is *not* the nanolab GDN that `tests/common/gdn.rs`
//! was written for, whose `sigmoid` gate and clamp are a different model.
//!
//! Every function validates buffer capacities, runtime ownership and aliasing on
//! the host before encoding, like [`crate::nn`]. The kernels guard their grid
//! and whatever only the device can see: they clamp device-side lengths and
//! positions (`seq_lens`, `tkv`, suffix lengths, position buffers) to the
//! validated capacities, and turn an out-of-range slot, answer or token id into
//! a NaN row instead of an out-of-bounds read.

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_2d, set_f32, set_gpu_buf, set_gpu_buf_offset, set_u32, Binder};
use crate::gemm::{gemm, gemm_epilogue, Epilogue, GemmBackend};
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes, require_runtime, validate_rms_scalars};
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};

/// Key head dim the GDN kernels are compiled for (`linear_key_head_dim`, 128 at
/// every published Qwen3.5 size).
pub const GDN_KEY_DIM: u32 = 128;
/// Tokens per chunk in [`gdn_chunk_forward`].
pub const GDN_CHUNK: u32 = 64;
/// Value columns one scan / decode threadgroup owns; `v_dim` must be a multiple.
pub const GDN_VALUE_BLOCK: u32 = 32;

// Threadgroup memory, mirroring `GDN_*_TG_FLOATS` in kernels/qwen35_gdn.metal.
const PREP_TG_BYTES: usize = 4 * (64 * 65 + 4 * 64);
const SCAN_TG_BYTES: usize = 4 * (128 * 36 + 64 * 36 + 4 * 128 + 4 * 64);
const SCAN16_TG_BYTES: usize = 4 * (128 * 20 + 64 * 20 + 4 * 128 + 4 * 64);
const REC_TG_BYTES: usize = 4 * (2 * 4 * 32);
const PREP_THREADS: usize = 256;
const SCAN_THREADS: usize = 128;
/// Simdgroups per threadgroup for the one-simdgroup-per-row kernels.
const ROWS_PER_TG: usize = 8;
/// `REDUCE_MAX_SIMDGROUPS` in kernels/reduce_tree.h.
const REDUCE_MAX_SIMDGROUPS: usize = 32;
const SCORE_THREADS: usize = 256;
/// Threads per conv threadgroup, along the channel axis. The kernel indexes by
/// grid position, so this is an occupancy choice only: 32-thread groups (one
/// per 32 channels x 1 token) left the per-core threadgroup limit the cap.
const CONV_THREADS: usize = 256;
/// Answers per scoring call. Far above any real answer set; it bounds the
/// threadgroup memory the logits occupy, which `score_tg_bytes` must keep
/// inside 32 KB at this maximum (a unit test holds it to that).
pub const MAX_ANSWERS: u32 = 4096;

/// Threadgroup memory of one scoring call: the reduction partials, then one
/// logit per answer.
fn score_tg_bytes(n_answers: u32) -> usize {
    ((REDUCE_MAX_SIMDGROUPS + n_answers as usize) * 4).next_multiple_of(16)
}

// ---------------------------------------------------------------- views ---

/// A column window of a row-major f32 matrix: row `r` of the window starts at
/// element `r * ld + off` of `buf`.
#[derive(Clone, Copy, Debug)]
pub struct Cols<'a> {
    pub buf: &'a GpuBuffer,
    /// Row stride, in elements.
    pub ld: u32,
    /// Column of the window's first element.
    pub off: u32,
}

impl<'a> Cols<'a> {
    /// A dense matrix `width` columns wide.
    pub fn dense(buf: &'a GpuBuffer, width: u32) -> Self {
        Self { buf, ld: width, off: 0 }
    }
}

/// Elements a `rows x width` window at `off` of stride `ld` reaches, or an error
/// if it does not fit in its row.
pub(crate) fn window_elems(rows: u64, ld: u32, off: u32, width: u64, what: &str) -> Result<usize, String> {
    if u64::from(off) + width > u64::from(ld) {
        return Err(format!(
            "{what}: window [{off}, {off} + {width}) does not fit a row of {ld}"
        ));
    }
    if rows == 0 || width == 0 {
        return Ok(0);
    }
    let last = (rows - 1)
        .checked_mul(u64::from(ld))
        .and_then(|n| n.checked_add(u64::from(off) + width))
        .ok_or_else(|| format!("{what}: extent overflows"))?;
    usize::try_from(last).map_err(|_| format!("{what}: extent overflows usize"))
}

/// Check a window's buffer: right runtime, enough `T`-sized elements.
pub(crate) fn require_window<T>(rt: &GpuRuntime, c: Cols<'_>, rows: u64, width: u64, what: &str) -> Result<(), String> {
    let need = window_elems(rows, c.ld, c.off, width, what)?;
    require::<T>(rt, c.buf, need, what)
}

fn u32_product(parts: &[u32], what: &str) -> Result<u32, String> {
    parts
        .iter()
        .try_fold(1u32, |acc, &p| acc.checked_mul(p))
        .ok_or_else(|| format!("{what}: product exceeds u32"))
}

fn usize_product(parts: &[usize], what: &str) -> Result<usize, String> {
    parts
        .iter()
        .try_fold(1usize, |acc, &p| acc.checked_mul(p))
        .ok_or_else(|| format!("{what}: product overflows usize"))
}

fn pipeline_for(
    rt: &GpuRuntime,
    name: &str,
    threads: usize,
    tg_bytes: usize,
) -> Result<objc2::rc::Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let p = rt.pipeline(name)?;
    let max = p.maxTotalThreadsPerThreadgroup();
    if max < threads {
        return Err(format!(
            "{name}: needs {threads} threads per threadgroup, the pipeline allows {max}"
        ));
    }
    if tg_bytes > rt.max_threadgroup_memory() {
        return Err(format!(
            "{name}: needs {tg_bytes} bytes of threadgroup memory, the device has {}",
            rt.max_threadgroup_memory()
        ));
    }
    Ok(p)
}

/// One dispatch of `groups` threadgroups of `threads` threads, with optional
/// threadgroup memory at index 0.
fn dispatch_groups(
    rt: &Arc<GpuRuntime>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups: (usize, usize, usize),
    threads: usize,
    tg_bytes: usize,
    encode: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups.0 == 0 || groups.1 == 0 || groups.2 == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode(bnd);
        if tg_bytes > 0 {
            bnd.set_threadgroup_memory(0, tg_bytes);
        }
        bnd.dispatch(mtl_size(groups.0, groups.1, groups.2), mtl_size(threads, 1, 1));
        Ok(())
    })
}

// -------------------------------------------------------- projection layout ---

/// Column layout of a GDN layer's fused in-projection output:
/// `[q | k | v | z | b | a]`, as [`pack_linear_weights_f32`] builds it from
/// `[in_proj_qkv, in_proj_z, in_proj_b, in_proj_a]`.
///
/// Built by [`GdnProjLayout::new`], which checks the shape the GDN kernels
/// accept and that every offset fits `u32`, so the getters cannot overflow.
/// Take the call shape from [`GdnProjLayout::dims`] rather than restating it:
/// a layout and a [`GdnDims`] that disagree on `v_heads` still pass every
/// capacity check, and read the wrong columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnProjLayout {
    k_heads: u32,
    v_heads: u32,
    v_dim: u32,
}

impl GdnProjLayout {
    pub fn new(k_heads: u32, v_heads: u32, v_dim: u32) -> Result<Self, String> {
        let l = Self {
            k_heads,
            v_heads,
            v_dim,
        };
        l.dims(0, 0).validate("GdnProjLayout")?;
        // The widest offset: 2*key + 2*value + 2*v_heads.
        let key = u64::from(k_heads) * u64::from(GDN_KEY_DIM);
        let value = u64::from(v_heads) * u64::from(v_dim);
        if 2 * key + 2 * value + 2 * u64::from(v_heads) > u64::from(u32::MAX) {
            return Err("GdnProjLayout: projection width exceeds u32".into());
        }
        Ok(l)
    }
    pub fn k_heads(&self) -> u32 {
        self.k_heads
    }
    pub fn v_heads(&self) -> u32 {
        self.v_heads
    }
    pub fn v_dim(&self) -> u32 {
        self.v_dim
    }
    /// The GDN call shape for `batch` sequences of `seq` tokens in this layout.
    pub fn dims(&self, batch: u32, seq: u32) -> GdnDims {
        GdnDims {
            batch,
            seq,
            k_heads: self.k_heads,
            v_heads: self.v_heads,
            v_dim: self.v_dim,
        }
    }
    pub fn key_dim(&self) -> u32 {
        self.k_heads * GDN_KEY_DIM
    }
    pub fn value_dim(&self) -> u32 {
        self.v_heads * self.v_dim
    }
    /// Channels the causal conv runs over: `q | k | v`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_dim() + self.value_dim()
    }
    pub fn z_off(&self) -> u32 {
        self.conv_dim()
    }
    pub fn b_off(&self) -> u32 {
        self.z_off() + self.value_dim()
    }
    pub fn a_off(&self) -> u32 {
        self.b_off() + self.v_heads
    }
    /// Total columns of the fused projection.
    pub fn width(&self) -> u32 {
        self.a_off() + self.v_heads
    }
    /// Output-row widths of the four `nn.Linear`s, in packing order.
    pub fn part_widths(&self) -> [usize; 4] {
        [
            self.conv_dim() as usize,
            self.value_dim() as usize,
            self.v_heads as usize,
            self.v_heads as usize,
        ]
    }
    /// Where the conv output (a dense `[rows, conv_dim]` matrix) holds q, k, v.
    pub fn conv_qkv<'a>(&self, conv_out: &'a GpuBuffer) -> GdnQkv<'a> {
        GdnQkv {
            buf: conv_out,
            ld: self.conv_dim(),
            q_off: 0,
            k_off: self.key_dim(),
            v_off: 2 * self.key_dim(),
        }
    }
    /// Where the fused projection holds the raw gate logits.
    pub fn gates<'a>(&self, proj: &'a GpuBuffer) -> GdnGateLogits<'a> {
        GdnGateLogits {
            buf: proj,
            ld: self.width(),
            a_off: self.a_off(),
            b_off: self.b_off(),
        }
    }
    /// Where the fused projection holds `z`, the gated norm's gate.
    pub fn z<'a>(&self, proj: &'a GpuBuffer) -> Cols<'a> {
        Cols {
            buf: proj,
            ld: self.width(),
            off: self.z_off(),
        }
    }
}

/// Column layout of a full-attention layer's fused projection:
/// `[q+gate (2*Hq*D, per head: D query then D gate) | k (Hkv*D) | v (Hkv*D)]`,
/// from `[q_proj, k_proj, v_proj]`. [`AttnProjLayout::new`] checks every
/// offset fits `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnProjLayout {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

impl AttnProjLayout {
    pub fn new(q_heads: u32, kv_heads: u32, head_dim: u32) -> Result<Self, String> {
        if q_heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err("AttnProjLayout: head counts and head_dim must be non-zero".into());
        }
        let width = (2 * u64::from(q_heads) + 2 * u64::from(kv_heads)) * u64::from(head_dim);
        if width > u64::from(u32::MAX) {
            return Err("AttnProjLayout: projection width exceeds u32".into());
        }
        Ok(Self {
            q_heads,
            kv_heads,
            head_dim,
        })
    }
    pub fn q_heads(&self) -> u32 {
        self.q_heads
    }
    pub fn kv_heads(&self) -> u32 {
        self.kv_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }
    pub fn q_off(&self) -> u32 {
        0
    }
    pub fn k_off(&self) -> u32 {
        2 * self.q_heads * self.head_dim
    }
    pub fn v_off(&self) -> u32 {
        self.k_off() + self.kv_heads * self.head_dim
    }
    pub fn width(&self) -> u32 {
        self.v_off() + self.kv_heads * self.head_dim
    }
    pub fn part_widths(&self) -> [usize; 3] {
        [
            (2 * self.q_heads * self.head_dim) as usize,
            (self.kv_heads * self.head_dim) as usize,
            (self.kv_heads * self.head_dim) as usize,
        ]
    }
}

/// Pack `nn.Linear` weights (`[out_i, in]` row-major, as a checkpoint stores
/// them) into the right operand of one GEMM: `[in, sum(out_i)]`, the parts'
/// transposes side by side. `x @ packed` is then every projection at once, laid
/// out as [`GdnProjLayout`] / [`AttnProjLayout`] describe.
pub fn pack_linear_weights_f32(
    parts: &[&[f32]],
    out_features: &[usize],
    in_features: usize,
) -> Result<Vec<f32>, String> {
    pack_linear_weights(parts, out_features, in_features)
}

/// [`pack_linear_weights_f32`] for bf16 bit patterns.
pub fn pack_linear_weights_bf16(
    parts: &[&[u16]],
    out_features: &[usize],
    in_features: usize,
) -> Result<Vec<u16>, String> {
    pack_linear_weights(parts, out_features, in_features)
}

fn pack_linear_weights<T: Copy + Default>(
    parts: &[&[T]],
    out_features: &[usize],
    in_features: usize,
) -> Result<Vec<T>, String> {
    if parts.len() != out_features.len() {
        return Err(format!(
            "pack_linear_weights: {} parts but {} widths",
            parts.len(),
            out_features.len()
        ));
    }
    // Checked: with `in_features = 0` every part is empty and passes its length
    // check whatever its width, so a wrapped sum would reach the loops below.
    let total = out_features
        .iter()
        .try_fold(0usize, |acc, &o| acc.checked_add(o))
        .ok_or_else(|| "pack_linear_weights: output widths overflow usize".to_string())?;
    for (i, (p, &o)) in parts.iter().zip(out_features).enumerate() {
        let want = usize_product(&[o, in_features], "pack_linear_weights")?;
        if p.len() != want {
            return Err(format!(
                "pack_linear_weights: part {i} has {} elements, expected {o} x {in_features}",
                p.len()
            ));
        }
    }
    let mut out = vec![T::default(); usize_product(&[in_features, total], "pack_linear_weights")?];
    if out.is_empty() {
        // Nothing to place; and with `in_features = 0` the row loop below
        // would still walk every one of `total` (possibly enormous) rows.
        return Ok(out);
    }
    let mut col0 = 0;
    for (p, &o) in parts.iter().zip(out_features) {
        for r in 0..o {
            let src = &p[r * in_features..(r + 1) * in_features];
            for (k, &v) in src.iter().enumerate() {
                out[k * total + col0 + r] = v;
            }
        }
        col0 += o;
    }
    Ok(out)
}

/// `proj = x @ packed`: every projection of a layer in one GEMM. `x` is
/// `[rows, hidden]`, `packed` is [`pack_linear_weights_f32`]'s `[hidden, width]`
/// (bf16 for the bf16 path), `proj` is `[rows, width]` f32.
pub fn fused_projection(x: &Tensor, packed: &Tensor, proj: &Tensor, backend: GemmBackend) -> Result<(), String> {
    gemm(x, packed, proj, backend)
}

/// `residual += y @ w_out` in one dispatch: the output projection with the
/// residual add folded into the GEMM epilogue (`beta = 1` accumulates into `C`
/// while the product is still in registers). Needs the epilogue path — bf16
/// operands, or f32 under `PrecisionMode::Relaxed`, on TensorOps.
pub fn project_residual(y: &Tensor, w_out: &Tensor, residual: &Tensor, backend: GemmBackend) -> Result<(), String> {
    gemm_epilogue(
        y,
        w_out,
        residual,
        backend,
        Epilogue {
            beta: 1.0,
            ..Epilogue::default()
        },
    )
}

// ------------------------------------------------------------------ conv1d ---

/// Where a recurrence starts.
#[derive(Clone, Copy, Debug)]
pub enum StateIn<'a> {
    /// All zeros; nothing is read.
    Zero,
    /// One state per batch row.
    PerBatch(&'a GpuBuffer),
    /// One state shared, read-only, by every batch row: many continuations of
    /// one prefilled prefix without copying or perturbing it.
    Snapshot(&'a GpuBuffer),
}

impl<'a> StateIn<'a> {
    fn buffer(&self) -> Option<&'a GpuBuffer> {
        match *self {
            StateIn::Zero => None,
            StateIn::PerBatch(b) | StateIn::Snapshot(b) => Some(b),
        }
    }
    /// (flags bit, batch stride, elements to validate) for a per-row state size.
    fn plan(&self, per_row: u32, batch: u32) -> (u32, u32, usize) {
        match self {
            StateIn::Zero => (0, 0, 0),
            StateIn::PerBatch(_) => (1, per_row, per_row as usize * batch as usize),
            StateIn::Snapshot(_) => (1, 0, per_row as usize),
        }
    }
}

/// Depthwise causal conv over time, then SiLU:
/// `y[b, t, c] = silu(sum_j w[c, j] * x_ext[b, t + j, c])`, where `x_ext` is the
/// `kernel_width - 1` state entries (oldest first) followed by `x`.
///
/// - `x`: rows `batch * seq`, channels `[x.off, x.off + channels)`.
/// - `weight`: `[channels, kernel_width]` (transformers' `conv1d.weight.squeeze(1)`).
/// - `state`: `[.., channels, kernel_width - 1]`.
/// - `y`: dense `[batch * seq, channels]`.
/// - `state_out`: when given, `[batch, channels, kernel_width - 1]` receives the
///   last `kernel_width - 1` inputs of each row, for the next call. It must not
///   be the input state.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    weight: &GpuBuffer,
    kernel_width: u32,
    state: StateIn<'_>,
    y: &GpuBuffer,
    state_out: Option<&GpuBuffer>,
    batch: u32,
    seq: u32,
    channels: u32,
) -> Result<(), String> {
    conv1d_silu_impl(
        rt,
        x,
        weight,
        kernel_width,
        state,
        y,
        state_out,
        batch,
        seq,
        channels,
        None,
    )
}

/// [`conv1d_silu`] with a length per row.
///
/// For ragged batches: `seq_lens` holds `batch` u32s on the device, row b
/// being `min(seq_lens[b], seq)` tokens long (rows stay `seq` apart, so
/// right-pad each row to `seq`). Each row is exactly what the equal-length
/// call computes for it alone at that length, `state_out` included; outputs
/// past a row's length are not written.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_varlen(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    weight: &GpuBuffer,
    kernel_width: u32,
    state: StateIn<'_>,
    y: &GpuBuffer,
    state_out: Option<&GpuBuffer>,
    batch: u32,
    seq: u32,
    channels: u32,
    seq_lens: &GpuBuffer,
) -> Result<(), String> {
    conv1d_silu_impl(
        rt,
        x,
        weight,
        kernel_width,
        state,
        y,
        state_out,
        batch,
        seq,
        channels,
        Some(seq_lens),
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_silu_impl(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    weight: &GpuBuffer,
    kernel_width: u32,
    state: StateIn<'_>,
    y: &GpuBuffer,
    state_out: Option<&GpuBuffer>,
    batch: u32,
    seq: u32,
    channels: u32,
    seq_lens: Option<&GpuBuffer>,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::conv1d_silu";
    if !(2..=8).contains(&kernel_width) {
        return Err(format!("{WHAT}: kernel_width must be 2..=8, got {kernel_width}"));
    }
    let hist = kernel_width - 1;
    // The kernel's grid (and its `T + KW - 1` bound) is seq + hist positions.
    let positions = seq
        .checked_add(hist)
        .ok_or_else(|| format!("{WHAT}: seq + kernel_width - 1 exceeds u32"))?;
    let rows = u64::from(batch) * u64::from(seq);
    require_window::<f32>(rt, x, rows, u64::from(channels), "conv1d_silu x")?;
    require::<f32>(
        rt,
        weight,
        usize_product(&[channels as usize, kernel_width as usize], WHAT)?,
        "conv1d_silu weight",
    )?;
    require::<f32>(
        rt,
        y,
        usize_product(&[rows as usize, channels as usize], WHAT)?,
        "conv1d_silu y",
    )?;
    let per_row = u32_product(&[channels, hist], WHAT)?;
    let (in_flag, bstride, in_need) = state.plan(per_row, batch);
    if let Some(s) = state.buffer() {
        require::<f32>(rt, s, in_need, "conv1d_silu state_in")?;
    }
    let state_elems = per_row as usize * batch as usize;
    if let Some(s) = state_out {
        require::<f32>(rt, s, state_elems, "conv1d_silu state_out")?;
    }
    if let Some(l) = seq_lens {
        require::<u32>(rt, l, batch as usize, "conv1d_silu seq_lens")?;
    }
    // seq == 0 with a state_out still runs: the kernel copies the input state
    // through, so a caller swapping state buffers each step never reads a
    // stale one after an empty step.
    if batch == 0 || channels == 0 || (seq == 0 && state_out.is_none()) {
        return Ok(());
    }
    // state_out may never be the input state: output slot j of channel c is
    // input slot j + seq of the extended sequence, which another thread reads.
    let mut writes = vec![("y", y)];
    if let Some(s) = state_out {
        writes.push(("state_out", s));
    }
    let mut reads = vec![("x", x.buf), ("weight", weight)];
    if let Some(s) = state.buffer() {
        reads.push(("state_in", s));
    }
    if let Some(l) = seq_lens {
        reads.push(("seq_lens", l));
    }
    require_disjoint_writes(WHAT, &writes, &reads)?;

    let flags = in_flag | if state_out.is_some() { 2 } else { 0 } | if seq_lens.is_some() { 4 } else { 0 };
    let p = pipeline_for(rt, "qwen35_conv1d_silu", CONV_THREADS, 0)?;
    dispatch_groups(
        rt,
        &p,
        (
            (channels as usize).div_ceil(CONV_THREADS),
            positions as usize,
            batch as usize,
        ),
        CONV_THREADS,
        0,
        |bnd| {
            set_gpu_buf(bnd, x.buf, 0);
            set_gpu_buf(bnd, weight, 1);
            // Unread when the flag is off; any resident buffer satisfies the bind.
            set_gpu_buf(bnd, state.buffer().unwrap_or(weight), 2);
            set_gpu_buf(bnd, y, 3);
            set_gpu_buf(bnd, state_out.unwrap_or(y), 4);
            set_u32(bnd, batch, 5);
            set_u32(bnd, seq, 6);
            set_u32(bnd, channels, 7);
            set_u32(bnd, kernel_width, 8);
            set_u32(bnd, x.ld, 9);
            set_u32(bnd, x.off, 10);
            set_u32(bnd, bstride, 11);
            set_u32(bnd, flags, 12);
            // Unread without flag 4.
            set_gpu_buf(bnd, seq_lens.unwrap_or(weight), 13);
        },
    )
}

// --------------------------------------------------------------------- GDN ---

/// The GDN gates as values, for the training op: `g[r, h] = -exp(A_log[h]) *
/// softplus(a[r, h] + dt_bias[h])` and `beta[r, h] = sigmoid(b[r, h])`, each
/// written dense `[rows, heads]` f32, from the raw logits in the fused
/// projection. The inference kernels compute the same values in their loads,
/// with the same helpers (`kernels/qwen35_act.h`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_gates(
    rt: &Arc<GpuRuntime>,
    logits: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    g: &GpuBuffer,
    beta: &GpuBuffer,
    rows: u32,
    heads: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::gdn_gates";
    let r = u64::from(rows);
    for (off, name) in [(logits.a_off, "a"), (logits.b_off, "b")] {
        let c = Cols {
            buf: logits.buf,
            ld: logits.ld,
            off,
        };
        require_window::<f32>(rt, c, r, u64::from(heads), &format!("{WHAT} {name}"))?;
    }
    let n = usize_product(&[rows as usize, heads as usize], WHAT)?;
    require::<f32>(rt, params.a_log, heads as usize, &format!("{WHAT} a_log"))?;
    require::<f32>(rt, params.dt_bias, heads as usize, &format!("{WHAT} dt_bias"))?;
    require::<f32>(rt, g, n, &format!("{WHAT} g"))?;
    require::<f32>(rt, beta, n, &format!("{WHAT} beta"))?;
    require_disjoint_writes(
        WHAT,
        &[("g", g), ("beta", beta)],
        &[
            ("logits", logits.buf),
            ("a_log", params.a_log),
            ("dt_bias", params.dt_bias),
        ],
    )?;
    if n == 0 {
        return Ok(());
    }
    let p = rt.pipeline("qwen35_gdn_gates_f32")?;
    dispatch_2d(rt, &p, heads as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, logits.buf, 0);
        set_gpu_buf(bnd, params.a_log, 1);
        set_gpu_buf(bnd, params.dt_bias, 2);
        set_gpu_buf(bnd, g, 3);
        set_gpu_buf(bnd, beta, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, heads, 6);
        set_u32(bnd, logits.ld, 7);
        set_u32(bnd, logits.a_off, 8);
        set_u32(bnd, logits.b_off, 9);
    })
}

/// Shape of one GDN call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDims {
    pub batch: u32,
    pub seq: u32,
    /// `linear_num_key_heads`. Key head dim is fixed at [`GDN_KEY_DIM`].
    pub k_heads: u32,
    /// `linear_num_value_heads`; a multiple of `k_heads`.
    pub v_heads: u32,
    /// `linear_value_head_dim`; a multiple of [`GDN_VALUE_BLOCK`].
    pub v_dim: u32,
}

impl GdnDims {
    fn validate(&self, what: &str) -> Result<(), String> {
        if self.k_heads == 0 || self.v_heads == 0 || self.v_heads % self.k_heads != 0 {
            return Err(format!(
                "{what}: v_heads ({}) must be a non-zero multiple of k_heads ({})",
                self.v_heads, self.k_heads
            ));
        }
        if self.v_dim == 0 || self.v_dim % GDN_VALUE_BLOCK != 0 {
            return Err(format!(
                "{what}: v_dim must be a non-zero multiple of {GDN_VALUE_BLOCK}, got {}",
                self.v_dim
            ));
        }
        // Every per-row / per-head extent the kernels index in u32.
        u32_product(&[self.v_heads, GDN_KEY_DIM, self.v_dim], what)?;
        u32_product(&[self.k_heads, GDN_KEY_DIM, 2], what)?;
        self.chunks()
            .checked_mul(GDN_CHUNK)
            .ok_or_else(|| format!("{what}: seq too long"))?;
        Ok(())
    }

    /// Chunks of [`GDN_CHUNK`] tokens, rounding up.
    pub fn chunks(&self) -> u32 {
        self.seq.div_ceil(GDN_CHUNK)
    }

    /// Elements of one recurrent state `[v_heads, key_dim, v_dim]`.
    pub fn state_elems_per_row(&self) -> usize {
        self.v_heads as usize * GDN_KEY_DIM as usize * self.v_dim as usize
    }
}

/// q, k and v as column windows of one buffer (the conv output):
/// q head `h` is columns `q_off + h*128 ..`, likewise k; v head `h` is
/// `v_off + h*v_dim ..`.
#[derive(Clone, Copy, Debug)]
pub struct GdnQkv<'a> {
    pub buf: &'a GpuBuffer,
    pub ld: u32,
    pub q_off: u32,
    pub k_off: u32,
    pub v_off: u32,
}

/// The raw `a` and `b` logits (before softplus / sigmoid), one column per value
/// head, as the fused projection holds them.
#[derive(Clone, Copy, Debug)]
pub struct GdnGateLogits<'a> {
    pub buf: &'a GpuBuffer,
    pub ld: u32,
    pub a_off: u32,
    pub b_off: u32,
}

/// Per-head parameters, `[v_heads]` f32 each.
#[derive(Clone, Copy, Debug)]
pub struct GdnParams<'a> {
    pub a_log: &'a GpuBuffer,
    pub dt_bias: &'a GpuBuffer,
}

/// How many value columns one threadgroup of the chunked rule's sequential scan
/// owns. Each output element runs the same arithmetic either way, so the
/// results are bit-identical; `Cols16` launches twice the threadgroups, which
/// pays when a small batch leaves the GPU underfilled (`probe_gdn_scan`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GdnScanSlice {
    /// `qwen35_gdn_chunk_scan`: `v_dim / 32` threadgroups per head.
    Cols32,
    /// `qwen35_gdn_chunk_scan_bv16`: `v_dim / 16` threadgroups per head.
    ///
    /// Default. A paired batch-1 measurement (`probe_gdn_scan --paired`) had
    /// this faster than [`Self::Cols32`] at T = 200 and T = 8192, with the
    /// same bits.
    #[default]
    Cols16,
}

/// Scratch for [`gdn_chunk_forward`]'s two passes. Allocate once for the
/// largest call and reuse it across layers. It also carries the scan's
/// [`GdnScanSlice`], so every chunked entry point that takes it honours it.
pub struct GdnWorkspace {
    scan: GdnScanSlice,
    k: GpuBuffer,
    q: GpuBuffer,
    g: GpuBuffer,
    beta: GpuBuffer,
    w: GpuBuffer,
    aq: GpuBuffer,
}

/// Element counts of the workspace buffers for `dims`: (rows, blocks), where
/// `k`/`q` hold `rows * 128`, `g`/`beta` hold `rows`, and `w`/`aq` hold
/// `blocks * 64 * 64`.
fn workspace_extent(dims: &GdnDims) -> Result<(usize, usize), String> {
    let heads = usize_product(&[dims.batch as usize, dims.v_heads as usize], "GdnWorkspace")?;
    let blocks = usize_product(&[heads, dims.chunks() as usize], "GdnWorkspace")?;
    let rows = usize_product(&[blocks, GDN_CHUNK as usize], "GdnWorkspace")?;
    Ok((rows, blocks))
}

impl GdnWorkspace {
    /// Workspace sized for calls up to `max` (batch, seq and heads).
    pub fn new(rt: &GpuRuntime, max: &GdnDims) -> Result<Self, String> {
        let (rows, blocks) = workspace_extent(max)?;
        let f = std::mem::size_of::<f32>();
        let kq = usize_product(&[rows, GDN_KEY_DIM as usize, f], "GdnWorkspace")?;
        let row_bytes = usize_product(&[rows, f], "GdnWorkspace")?;
        let blk = usize_product(&[blocks, (GDN_CHUNK * GDN_CHUNK) as usize, f], "GdnWorkspace")?;
        // A zero-length Metal buffer is not a buffer; keep every slot non-empty.
        let alloc = |n: usize| rt.alloc_buffer(n.max(16));
        Ok(Self {
            scan: GdnScanSlice::default(),
            k: alloc(kq)?,
            q: alloc(kq)?,
            g: alloc(row_bytes)?,
            beta: alloc(row_bytes)?,
            w: alloc(blk)?,
            aq: alloc(blk)?,
        })
    }

    /// This workspace with the scan run in `slice`-column slices.
    pub fn with_scan_slice(mut self, slice: GdnScanSlice) -> Self {
        self.scan = slice;
        self
    }

    /// Point later scans on this workspace at `slice`.
    ///
    /// The scratch buffers stay put. Prep does not read the slice, so one prep
    /// can feed both widths.
    pub fn set_scan_slice(&mut self, slice: GdnScanSlice) {
        self.scan = slice;
    }

    /// Bytes this workspace needs for `dims`.
    pub fn bytes_for(dims: &GdnDims) -> Result<usize, String> {
        let (rows, blocks) = workspace_extent(dims)?;
        let per_row = usize_product(&[rows, 2 * GDN_KEY_DIM as usize + 2], "GdnWorkspace")?;
        let per_blk = usize_product(&[blocks, 2 * (GDN_CHUNK * GDN_CHUNK) as usize], "GdnWorkspace")?;
        let floats = per_row
            .checked_add(per_blk)
            .ok_or_else(|| "GdnWorkspace: size overflows usize".to_string())?;
        usize_product(&[floats, 4], "GdnWorkspace")
    }

    fn check(&self, rt: &GpuRuntime, dims: &GdnDims) -> Result<(), String> {
        let (rows, blocks) = workspace_extent(dims)?;
        let kq = usize_product(&[rows, GDN_KEY_DIM as usize], "GdnWorkspace")?;
        let blk = usize_product(&[blocks, (GDN_CHUNK * GDN_CHUNK) as usize], "GdnWorkspace")?;
        require::<f32>(rt, &self.k, kq, "GdnWorkspace k (too small for these dims)")?;
        require::<f32>(rt, &self.q, kq, "GdnWorkspace q")?;
        require::<f32>(rt, &self.g, rows, "GdnWorkspace g")?;
        require::<f32>(rt, &self.beta, rows, "GdnWorkspace beta")?;
        require::<f32>(rt, &self.w, blk, "GdnWorkspace w")?;
        require::<f32>(rt, &self.aq, blk, "GdnWorkspace aq")
    }

    fn buffers(&self) -> [(&'static str, &GpuBuffer); 6] {
        [
            ("ws.k", &self.k),
            ("ws.q", &self.q),
            ("ws.g", &self.g),
            ("ws.beta", &self.beta),
            ("ws.w", &self.w),
            ("ws.aq", &self.aq),
        ]
    }
}

/// Validation shared by the chunked and recurrent paths. Returns the
/// (flags, state batch stride) the kernels take.
#[allow(clippy::too_many_arguments)]
fn validate_gdn(
    rt: &GpuRuntime,
    what: &str,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    extra_writes: &[(&'static str, &GpuBuffer)],
    seq_lens: Option<&GpuBuffer>,
) -> Result<(u32, u32), String> {
    dims.validate(what)?;
    let rows = u64::from(dims.batch) * u64::from(dims.seq);
    let key_w = u64::from(dims.k_heads * GDN_KEY_DIM);
    let val_w = u64::from(dims.v_heads * dims.v_dim);
    let q = Cols {
        buf: qkv.buf,
        ld: qkv.ld,
        off: qkv.q_off,
    };
    let k = Cols {
        buf: qkv.buf,
        ld: qkv.ld,
        off: qkv.k_off,
    };
    let v = Cols {
        buf: qkv.buf,
        ld: qkv.ld,
        off: qkv.v_off,
    };
    require_window::<f32>(rt, q, rows, key_w, "gdn q")?;
    require_window::<f32>(rt, k, rows, key_w, "gdn k")?;
    require_window::<f32>(rt, v, rows, val_w, "gdn v")?;
    let hv = u64::from(dims.v_heads);
    let a = Cols {
        buf: gates.buf,
        ld: gates.ld,
        off: gates.a_off,
    };
    let b = Cols {
        buf: gates.buf,
        ld: gates.ld,
        off: gates.b_off,
    };
    require_window::<f32>(rt, a, rows, hv, "gdn a (gate logits)")?;
    require_window::<f32>(rt, b, rows, hv, "gdn b (gate logits)")?;
    require::<f32>(rt, params.a_log, dims.v_heads as usize, "gdn A_log")?;
    require::<f32>(rt, params.dt_bias, dims.v_heads as usize, "gdn dt_bias")?;
    require_window::<f32>(rt, out, rows, val_w, "gdn out")?;

    let per_row = dims.state_elems_per_row();
    let per_row_u32 = u32::try_from(per_row).map_err(|_| format!("{what}: state exceeds u32"))?;
    let (in_flag, bstride, in_need) = state.plan(per_row_u32, dims.batch);
    if let Some(s) = state.buffer() {
        require::<f32>(rt, s, in_need, "gdn state_in")?;
    }
    if let Some(s) = state_out {
        require::<f32>(
            rt,
            s,
            usize_product(&[per_row, dims.batch as usize], what)?,
            "gdn state_out",
        )?;
    }

    // `state_out` may be the per-batch input state itself: each thread reads
    // the elements it later writes, and nothing else touches them. It may not
    // be a snapshot, which every batch row reads.
    let in_place = matches!((state, state_out), (StateIn::PerBatch(i), Some(o)) if i.aliases(o));
    let mut writes: Vec<(&str, &GpuBuffer)> = vec![("out", out.buf)];
    writes.extend_from_slice(extra_writes);
    if let Some(s) = state_out {
        // In place, the state is one buffer that is both read and written; as
        // a write it is still checked against every input below, so it cannot
        // also be, say, the qkv buffer other threadgroups are reading.
        writes.push((if in_place { "state (in place)" } else { "state_out" }, s));
    }
    let mut reads: Vec<(&str, &GpuBuffer)> = vec![
        ("qkv", qkv.buf),
        ("gate logits", gates.buf),
        ("A_log", params.a_log),
        ("dt_bias", params.dt_bias),
    ];
    if let (Some(s), false) = (state.buffer(), in_place) {
        reads.push(("state_in", s));
    }
    if let Some(l) = seq_lens {
        require::<u32>(rt, l, dims.batch as usize, "gdn seq_lens")?;
        reads.push(("seq_lens", l));
    }
    require_disjoint_writes(what, &writes, &reads)?;
    let flags = in_flag | if state_out.is_some() { 2 } else { 0 } | if seq_lens.is_some() { 4 } else { 0 };
    Ok((flags, bstride))
}

/// The gated delta rule over a whole sequence, chunked: transformers'
/// `torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`, plus the
/// gate computation (`beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a +
/// dt_bias)`) that `Qwen3_5GatedDeltaNet.forward` does before it.
///
/// Two dispatches: `qwen35_gdn_chunk_prep`, parallel over (batch, head, chunk),
/// does the norms, gates and the 64x64 triangular solve; `qwen35_gdn_chunk_scan`
/// carries the state across chunks. `ws` holds what passes between them.
///
/// `out` receives value head `h` of token row `r` at columns
/// `out.off + h*v_dim ..`. `state_out` ([batch, v_heads, 128, v_dim]) receives
/// the final state and may be the [`StateIn::PerBatch`] buffer itself.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_forward(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    ws: &GdnWorkspace,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
) -> Result<(), String> {
    gdn_chunk_forward_impl(
        rt,
        dims,
        qkv,
        gates,
        params,
        state,
        ws,
        out,
        state_out,
        None,
        ChunkPhases::Both,
    )
}

/// [`gdn_chunk_forward`] with a length per row.
///
/// For ragged batches: `seq_lens` holds `batch` u32s on the device, row b
/// being `min(seq_lens[b], seq)` tokens long (rows stay `seq` apart, so
/// right-pad each row to `seq`). Each row is exactly what the equal-length
/// call computes for it alone at that length, `state_out` included; outputs
/// past a row's length are not written.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_forward_varlen(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    ws: &GdnWorkspace,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    seq_lens: &GpuBuffer,
) -> Result<(), String> {
    gdn_chunk_forward_impl(
        rt,
        dims,
        qkv,
        gates,
        params,
        state,
        ws,
        out,
        state_out,
        Some(seq_lens),
        ChunkPhases::Both,
    )
}

/// One of [`gdn_chunk_forward`]'s two dispatches, for timing them apart.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnChunkPhase {
    /// `qwen35_gdn_chunk_prep`: norms, gates, the C x C products and the
    /// solve, into `ws`.
    Prep,
    /// `qwen35_gdn_chunk_scan`: the state pass over `ws`. Only meaningful
    /// after a `Prep` over the same inputs has filled the workspace.
    Scan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkPhases {
    Both,
    Only(GdnChunkPhase),
}

/// [`gdn_chunk_forward`] running only one of its two dispatches, with the same
/// validation. For benchmarks that attribute the chunked rule's time; a
/// forward pass calls [`gdn_chunk_forward`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_phase(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    ws: &GdnWorkspace,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    phase: GdnChunkPhase,
) -> Result<(), String> {
    gdn_chunk_forward_impl(
        rt,
        dims,
        qkv,
        gates,
        params,
        state,
        ws,
        out,
        state_out,
        None,
        ChunkPhases::Only(phase),
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_chunk_forward_impl(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    ws: &GdnWorkspace,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    seq_lens: Option<&GpuBuffer>,
    phases: ChunkPhases,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::gdn_chunk_forward";
    dims.validate(WHAT)?;
    ws.check(rt, dims)?;
    let (flags, bstride) = validate_gdn(
        rt,
        WHAT,
        dims,
        qkv,
        gates,
        params,
        state,
        out,
        state_out,
        &ws.buffers(),
        seq_lens,
    )?;
    // seq == 0 with a state_out still dispatches: it copies the start state
    // through, so a caller alternating state buffers never reads a stale one.
    if dims.batch == 0 || (dims.seq == 0 && state_out.is_none()) {
        return Ok(());
    }

    let prep = pipeline_for(rt, "qwen35_gdn_chunk_prep", PREP_THREADS, PREP_TG_BYTES)?;
    // Literal names, so the emulator's host-contract check sees both kernels
    // this site binds.
    let scan_name = match ws.scan {
        GdnScanSlice::Cols32 => "qwen35_gdn_chunk_scan",
        GdnScanSlice::Cols16 => "qwen35_gdn_chunk_scan_bv16",
    };
    let (scan_cols, scan_bytes) = match ws.scan {
        GdnScanSlice::Cols32 => (GDN_VALUE_BLOCK, SCAN_TG_BYTES),
        GdnScanSlice::Cols16 => (16, SCAN16_TG_BYTES),
    };
    let scan = pipeline_for(rt, scan_name, SCAN_THREADS, scan_bytes)?;
    let run_prep = phases != ChunkPhases::Only(GdnChunkPhase::Scan);
    let run_scan = phases != ChunkPhases::Only(GdnChunkPhase::Prep);
    if run_prep {
        dispatch_groups(
            rt,
            &prep,
            (dims.chunks() as usize, dims.v_heads as usize, dims.batch as usize),
            PREP_THREADS,
            PREP_TG_BYTES,
            |bnd| {
                set_gpu_buf(bnd, qkv.buf, 0);
                set_gpu_buf(bnd, gates.buf, 1);
                set_gpu_buf(bnd, params.a_log, 2);
                set_gpu_buf(bnd, params.dt_bias, 3);
                set_gpu_buf(bnd, &ws.k, 4);
                set_gpu_buf(bnd, &ws.q, 5);
                set_gpu_buf(bnd, &ws.g, 6);
                set_gpu_buf(bnd, &ws.beta, 7);
                set_gpu_buf(bnd, &ws.w, 8);
                set_gpu_buf(bnd, &ws.aq, 9);
                set_u32(bnd, dims.seq, 10);
                set_u32(bnd, dims.k_heads, 11);
                set_u32(bnd, dims.v_heads, 12);
                set_u32(bnd, qkv.ld, 13);
                set_u32(bnd, qkv.q_off, 14);
                set_u32(bnd, qkv.k_off, 15);
                set_u32(bnd, gates.ld, 16);
                set_u32(bnd, gates.a_off, 17);
                set_u32(bnd, gates.b_off, 18);
                // Unread when use_lens is 0.
                set_gpu_buf(bnd, seq_lens.unwrap_or(qkv.buf), 19);
                set_u32(bnd, u32::from(seq_lens.is_some()), 20);
            },
        )?;
    }
    if !run_scan {
        return Ok(());
    }
    // The binder orders this after the prep dispatch (a Dispatch->Dispatch
    // barrier after every dispatch, or before the next in hazard mode).
    dispatch_groups(
        rt,
        &scan,
        (
            (dims.v_dim / scan_cols) as usize,
            dims.v_heads as usize,
            dims.batch as usize,
        ),
        SCAN_THREADS,
        scan_bytes,
        |bnd| {
            set_gpu_buf(bnd, qkv.buf, 0);
            set_gpu_buf(bnd, &ws.k, 1);
            set_gpu_buf(bnd, &ws.q, 2);
            set_gpu_buf(bnd, &ws.g, 3);
            set_gpu_buf(bnd, &ws.beta, 4);
            set_gpu_buf(bnd, &ws.w, 5);
            set_gpu_buf(bnd, &ws.aq, 6);
            set_gpu_buf(bnd, state.buffer().unwrap_or(qkv.buf), 7);
            set_gpu_buf(bnd, out.buf, 8);
            set_gpu_buf(bnd, state_out.unwrap_or(out.buf), 9);
            set_u32(bnd, dims.seq, 10);
            set_u32(bnd, dims.v_heads, 11);
            set_u32(bnd, dims.v_dim, 12);
            set_u32(bnd, qkv.ld, 13);
            set_u32(bnd, qkv.v_off, 14);
            set_u32(bnd, out.ld, 15);
            set_u32(bnd, out.off, 16);
            set_u32(bnd, bstride, 17);
            set_u32(bnd, flags, 18);
            set_gpu_buf(bnd, seq_lens.unwrap_or(qkv.buf), 19);
        },
    )
}

/// The gated delta rule token by token: transformers'
/// `torch_recurrent_gated_delta_rule`, with the same folded gates and norms as
/// [`gdn_chunk_forward`]. For decode (`seq = 1`) and short suffixes; prefer the
/// chunked path past a few dozen tokens.
///
/// With [`StateIn::Snapshot`] every batch row continues from one prefilled
/// state that is never written, so one prefix can answer many questions.
/// `state_out` may be the [`StateIn::PerBatch`] buffer itself.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
) -> Result<(), String> {
    gdn_recurrent_impl(rt, dims, qkv, gates, params, state, out, state_out, None)
}

/// [`gdn_recurrent`] with a length per row.
///
/// For ragged batches: `seq_lens` holds `batch` u32s on the device, row b
/// being `min(seq_lens[b], seq)` tokens long (rows stay `seq` apart, so
/// right-pad each row to `seq`). Each row is exactly what the equal-length
/// call computes for it alone at that length, `state_out` included; outputs
/// past a row's length are not written.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_varlen(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    seq_lens: &GpuBuffer,
) -> Result<(), String> {
    gdn_recurrent_impl(rt, dims, qkv, gates, params, state, out, state_out, Some(seq_lens))
}

#[allow(clippy::too_many_arguments)]
fn gdn_recurrent_impl(
    rt: &Arc<GpuRuntime>,
    dims: &GdnDims,
    qkv: &GdnQkv<'_>,
    gates: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    state: StateIn<'_>,
    out: Cols<'_>,
    state_out: Option<&GpuBuffer>,
    seq_lens: Option<&GpuBuffer>,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::gdn_recurrent";
    let (flags, bstride) = validate_gdn(rt, WHAT, dims, qkv, gates, params, state, out, state_out, &[], seq_lens)?;
    // seq == 0 with a state_out still dispatches: it copies the start state
    // through, so a caller alternating state buffers never reads a stale one.
    if dims.batch == 0 || (dims.seq == 0 && state_out.is_none()) {
        return Ok(());
    }
    let p = pipeline_for(rt, "qwen35_gdn_recurrent", SCAN_THREADS, REC_TG_BYTES)?;
    dispatch_groups(
        rt,
        &p,
        (
            (dims.v_dim / GDN_VALUE_BLOCK) as usize,
            dims.v_heads as usize,
            dims.batch as usize,
        ),
        SCAN_THREADS,
        REC_TG_BYTES,
        |bnd| {
            set_gpu_buf(bnd, qkv.buf, 0);
            set_gpu_buf(bnd, gates.buf, 1);
            set_gpu_buf(bnd, params.a_log, 2);
            set_gpu_buf(bnd, params.dt_bias, 3);
            set_gpu_buf(bnd, state.buffer().unwrap_or(qkv.buf), 4);
            set_gpu_buf(bnd, out.buf, 5);
            set_gpu_buf(bnd, state_out.unwrap_or(out.buf), 6);
            set_u32(bnd, dims.seq, 7);
            set_u32(bnd, dims.k_heads, 8);
            set_u32(bnd, dims.v_heads, 9);
            set_u32(bnd, dims.v_dim, 10);
            set_u32(bnd, qkv.ld, 11);
            set_u32(bnd, qkv.q_off, 12);
            set_u32(bnd, qkv.k_off, 13);
            set_u32(bnd, qkv.v_off, 14);
            set_u32(bnd, gates.ld, 15);
            set_u32(bnd, gates.a_off, 16);
            set_u32(bnd, gates.b_off, 17);
            set_u32(bnd, out.ld, 18);
            set_u32(bnd, out.off, 19);
            set_u32(bnd, bstride, 20);
            set_u32(bnd, flags, 21);
            set_gpu_buf(bnd, seq_lens.unwrap_or(qkv.buf), 22);
        },
    )
}

// ------------------------------------------------------------ gated norm ---

/// An output window with its element type ([`DType::F32`] or [`DType::BF16`]).
#[derive(Clone, Copy, Debug)]
pub struct OutCols<'a> {
    pub cols: Cols<'a>,
    pub dtype: DType,
}

fn out_kernel(base: &str, dtype: DType, what: &str) -> Result<String, String> {
    match dtype {
        DType::F32 => Ok(format!("{base}_f32")),
        DType::BF16 => Ok(format!("{base}_bf16")),
        other => Err(format!("{what}: dtype must be F32 or BF16, got {other:?}")),
    }
}

fn require_out_window(rt: &GpuRuntime, out: OutCols<'_>, rows: u64, width: u64, what: &str) -> Result<(), String> {
    match out.dtype {
        DType::BF16 => require_window::<u16>(rt, out.cols, rows, width, what),
        _ => require_window::<f32>(rt, out.cols, rows, width, what),
    }
}

/// `out = rms_norm(x) * w * silu(z)` per head of `dim`: `Qwen3_5RMSNormGated`.
/// `x`, `z` and `out` hold `heads * dim` columns per row; `weight` is `[dim]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    z: Cols<'_>,
    weight: &GpuBuffer,
    out: OutCols<'_>,
    rows: u32,
    heads: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::gated_rms_norm";
    let name = out_kernel("qwen35_gated_rms_norm", out.dtype, WHAT)?;
    if dim == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("{WHAT}: dim must be non-zero and eps positive"));
    }
    let width = u64::from(u32_product(&[heads, dim], WHAT)?);
    require_window::<f32>(rt, x, u64::from(rows), width, "gated_rms_norm x")?;
    require_window::<f32>(rt, z, u64::from(rows), width, "gated_rms_norm z")?;
    require::<f32>(rt, weight, dim as usize, "gated_rms_norm weight")?;
    require_out_window(rt, out, u64::from(rows), width, "gated_rms_norm out")?;
    let units = rows as usize * heads as usize;
    if units == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("out", out.cols.buf)],
        &[("x", x.buf), ("z", z.buf), ("weight", weight)],
    )?;
    let p = pipeline_for(rt, &name, ROWS_PER_TG * 32, 0)?;
    dispatch_groups(
        rt,
        &p,
        (units.div_ceil(ROWS_PER_TG), 1, 1),
        ROWS_PER_TG * 32,
        0,
        |bnd| {
            set_gpu_buf(bnd, x.buf, 0);
            set_gpu_buf(bnd, z.buf, 1);
            set_gpu_buf(bnd, weight, 2);
            set_gpu_buf(bnd, out.cols.buf, 3);
            set_u32(bnd, rows, 4);
            set_u32(bnd, heads, 5);
            set_u32(bnd, dim, 6);
            set_u32(bnd, x.ld, 7);
            set_u32(bnd, x.off, 8);
            set_u32(bnd, z.ld, 9);
            set_u32(bnd, z.off, 10);
            set_u32(bnd, out.cols.ld, 11);
            set_u32(bnd, out.cols.off, 12);
            set_f32(bnd, eps, 13);
        },
    )
}

// ------------------------------------------------------ attention extras ---

/// Shape of one full-attention call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub batch: u32,
    pub seq: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Leading dims that rotate: `head_dim * partial_rotary_factor` (64 of 256
    /// for Qwen3.5). Even, and at most `head_dim`.
    pub rotary_dim: u32,
}

/// Where [`attn_qk_norm_rope`] writes. The caches are `[batch, capacity,
/// kv_heads, head_dim]`, the layout [`crate::nn::flash_attn_rows`] reads.
///
/// There is no capacity field: it is derived from the cache buffers with
/// [`crate::nn::attn_kv_capacity`], exactly as flash attention derives it. A
/// capacity passed alongside the buffers could disagree with the one attention
/// reads, and every batch row after the first would land in the wrong slots.
#[derive(Clone, Copy, Debug)]
pub struct AttnTargets<'a> {
    /// `[batch, seq, q_heads, head_dim]`.
    pub q_out: &'a GpuBuffer,
    pub k_cache: &'a GpuBuffer,
    pub v_cache: &'a GpuBuffer,
}

/// Q/K RMSNorm with zero-centred weights (`* (1 + w)`), transformers' partial
/// RoPE (pairs `p, p + rotary_dim/2`, `inv_freq = theta^(-2p/rotary_dim)`),
/// and the K/V cache store, reading the fused projection in place.
///
/// `proj` rows are laid out as [`AttnProjLayout`] says (`proj.off` is the
/// window's first column, i.e. where `q` starts). Token `t` is rotated to
/// position `pos_offset + t` and cached at that slot.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    pos_offset: u32,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::Scalar(pos_offset),
        0,
        theta,
        eps,
        QkRead::Qwen,
    )
}

/// [`attn_qk_norm_rope`] with an explicit query-head stride.
///
/// [`attn_qk_norm_rope`] reads query head `j` at column `j * 2 * head_dim`
/// (Qwen packs `[q, gate]`) and multiplies the RMSNorm by `(1 + w)`. This
/// entry takes the stride. Pass `head_dim` for nanolab's packed `[T, H, D]`
/// layout, and `2 * head_dim` to match the Qwen query columns.
///
/// Key and value head `h` are column `h * head_dim` of the same window, so
/// one residual can be Q, K, and V together. `weight_bias` is added to the
/// RMSNorm weight: `0` is a plain learnable scale (`* w`, nanolab, `eps`
/// typically `1e-6`), `1` is Qwen's `(1 + w)`.
///
/// RoPE is the half-split on `shape.rotary_dim` leading dims:
/// `x * cos + cat(-x2, x1) * sin`, with `theta^(-2p / rotary_dim)`. Set
/// `rotary_dim` to `head_dim` to rotate the whole head.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_packed(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_head_stride: u32,
    weight_bias: f32,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    pos_offset: u32,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::Scalar(pos_offset),
        0,
        theta,
        eps,
        QkRead::Packed {
            q_head_stride,
            weight_bias,
        },
    )
}

/// [`attn_qk_norm_rope`] with the position offset read from `pos_offset`, a
/// one-element u32 device buffer, instead of bound as a scalar.
///
/// This is the variant for a decode loop replayed from an Indirect Command
/// Buffer: replay freezes scalar binds, so a scalar position would rotate and
/// cache every step at the recorded one. Advance the buffer between replays
/// instead. The host cannot see the value, so it cannot reject a position past
/// the caches' capacity: the kernel skips any token whose position is at or
/// past it (computed in 64 bits, so an offset near `u32::MAX` cannot wrap back
/// into range). Everything else is validated as for [`attn_qk_norm_rope`].
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_posbuf(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    pos_offset: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::Buffer(pos_offset),
        0,
        theta,
        eps,
        QkRead::Qwen,
    )
}

/// [`attn_qk_norm_rope`] for a continuation of a shared prefix: token `t` is
/// rotated to the absolute position `prefix_len + suffix_offset + t` but
/// cached at slot `suffix_offset + t` of `targets`' suffix caches, the layout
/// [`attn_prefix_rows`] reads (slot `s` is position `prefix_len + s`).
///
/// With `prefix_len = 0` it is exactly [`attn_qk_norm_rope`] at
/// `pos_offset = suffix_offset`.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_suffix(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    prefix_len: u32,
    suffix_offset: u32,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    let pos_offset = prefix_len.checked_add(suffix_offset).ok_or_else(|| {
        format!(
            "qwen35::attn_qk_norm_rope: prefix length {prefix_len} plus suffix offset \
             {suffix_offset} exceeds u32 positions"
        )
    })?;
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::Scalar(pos_offset),
        prefix_len,
        theta,
        eps,
        QkRead::Qwen,
    )
}

/// [`attn_qk_norm_rope_suffix`] with the **absolute** position of token 0,
/// `prefix_len + suffix_offset`, read from `pos_offset` (one device u32)
/// instead of bound as a scalar: the suffix writer for a decode loop replayed
/// from an Indirect Command Buffer, as [`attn_qk_norm_rope_posbuf`] is for an
/// ordinary cache. Advance the buffer between replays; `attn_prefix_decode`
/// reads its `q_pos_offset` the same way, so one buffer can serve both.
///
/// The host cannot see the position. The kernel skips any token whose slot,
/// `position - prefix_len`, falls outside `[0, capacity)`, computed in 64
/// bits.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_suffix_posbuf(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    prefix_len: u32,
    pos_offset: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::Buffer(pos_offset),
        prefix_len,
        theta,
        eps,
        QkRead::Qwen,
    )
}

/// [`attn_qk_norm_rope_suffix_posbuf`] with a position **per row**:
/// `positions` is `[batch]` device u32s, row b's token 0 at the absolute
/// position `positions[b]`, cached at slot `positions[b] - prefix_len`. For
/// continuations of different lengths (see [`attn_prefix_decode_varlen`]):
/// each row's next token lands at its own position.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_suffix_rows(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    prefix_len: u32,
    positions: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    qk_norm_rope_impl(
        rt,
        shape,
        proj,
        q_norm_w,
        k_norm_w,
        targets,
        RopePos::PerRow(positions),
        prefix_len,
        theta,
        eps,
        QkRead::Qwen,
    )
}

/// How query heads are read, and how the RMSNorm weight is applied.
#[derive(Clone, Copy)]
enum QkRead {
    /// Query stride `2 * head_dim`, weight `*(1 + w)`, [`AttnProjLayout`] columns.
    Qwen,
    /// Query head `j` at column `j * q_head_stride`. Key and value head `h`
    /// at column `h * head_dim`. Offsets are relative to `proj.off`.
    /// `weight_bias` is added to the RMSNorm weight (`0` is `* w`, `1` is
    /// Qwen's `*(1 + w)`).
    Packed { q_head_stride: u32, weight_bias: f32 },
}

/// Where the RoPE / cache position comes from.
#[derive(Clone, Copy)]
enum RopePos<'a> {
    Scalar(u32),
    /// One device u32 shared by every row.
    Buffer(&'a GpuBuffer),
    /// `[batch]` device u32s, row b's offset at element b.
    PerRow(&'a GpuBuffer),
}

#[allow(clippy::too_many_arguments)]
fn qk_norm_rope_impl(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    targets: &AttnTargets<'_>,
    pos: RopePos<'_>,
    slot_base: u32,
    theta: f32,
    eps: f32,
    read: QkRead,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_qk_norm_rope";
    let s = shape;
    if s.head_dim == 0 || s.q_heads == 0 || s.kv_heads == 0 {
        return Err(format!("{WHAT}: head_dim and head counts must be non-zero"));
    }
    if s.rotary_dim % 2 != 0 || s.rotary_dim > s.head_dim {
        return Err(format!(
            "{WHAT}: rotary_dim must be even and at most head_dim, got {} of {}",
            s.rotary_dim, s.head_dim
        ));
    }
    if !theta.is_finite() || theta <= 0.0 || !eps.is_finite() || eps <= 0.0 {
        return Err(format!("{WHAT}: theta and eps must be positive and finite"));
    }
    let layout = AttnProjLayout::new(s.q_heads, s.kv_heads, s.head_dim)?;
    let (q_col, k_col, v_col, row_width, q_head_stride, weight_bias) = match read {
        QkRead::Qwen => {
            let stride = s
                .head_dim
                .checked_mul(2)
                .ok_or_else(|| format!("{WHAT}: 2 * head_dim overflows u32"))?;
            (
                layout.q_off(),
                layout.k_off(),
                layout.v_off(),
                layout.width(),
                stride,
                1.0,
            )
        }
        QkRead::Packed {
            q_head_stride,
            weight_bias,
        } => {
            if q_head_stride < s.head_dim {
                return Err(format!(
                    "{WHAT}: query head stride {q_head_stride} is shorter than head_dim {}",
                    s.head_dim
                ));
            }
            if !weight_bias.is_finite() {
                return Err(format!("{WHAT}: weight_bias must be finite"));
            }
            let span = |heads: u32, stride: u32| -> Result<u64, String> {
                u64::from(heads - 1)
                    .checked_mul(u64::from(stride))
                    .and_then(|n| n.checked_add(u64::from(s.head_dim)))
                    .ok_or_else(|| format!("{WHAT}: packed columns overflow"))
            };
            let width = span(s.q_heads, q_head_stride)?.max(span(s.kv_heads, s.head_dim)?);
            let width_u = u32::try_from(width).map_err(|_| format!("{WHAT}: packed width exceeds u32"))?;
            (0, 0, 0, width_u, q_head_stride, weight_bias)
        }
    };
    if s.batch == 0 || s.seq == 0 {
        // Nothing to write; and with no batch rows the caches imply no
        // capacity, which must not read as "positions past capacity".
        return Ok(());
    }
    let kv_capacity = crate::nn::attn_kv_capacity(targets.k_cache, targets.v_cache, s.batch, s.kv_heads, s.head_dim)?;
    // Both are known on the host whatever the position mode. With no slots
    // every token would be skipped as out of range, a call that "succeeds"
    // and writes nothing. And a cache whose last slot's absolute position
    // `slot_base + capacity - 1` passes u32 would let the kernel's u32
    // position wrap for in-range tokens; this is the same bound
    // `validate_prefix_attn` puts on the reader of these caches.
    if kv_capacity == 0 {
        return Err(format!("{WHAT}: the caches hold no positions (capacity 0)"));
    }
    if u64::from(slot_base) + u64::from(kv_capacity) > u64::from(u32::MAX) {
        return Err(format!(
            "{WHAT}: cache positions [{slot_base}, {slot_base} + {kv_capacity}) exceed u32"
        ));
    }
    match pos {
        RopePos::Scalar(pos_offset) => {
            if pos_offset < slot_base {
                return Err(format!(
                    "{WHAT}: position {pos_offset} precedes the cache's first position {slot_base}"
                ));
            }
            if u64::from(pos_offset - slot_base) + u64::from(s.seq) > u64::from(kv_capacity) {
                return Err(format!(
                    "{WHAT}: positions [{pos_offset}, {pos_offset} + {}) exceed the caches' capacity {kv_capacity}",
                    s.seq
                ));
            }
        }
        RopePos::Buffer(b) => require::<u32>(rt, b, 1, "attn_qk_norm_rope pos_offset")?,
        RopePos::PerRow(b) => require::<u32>(rt, b, s.batch as usize, "attn_qk_norm_rope per-row pos_offset")?,
    }
    let width = u64::from(row_width);
    let rows = u64::from(s.batch) * u64::from(s.seq);
    require_window::<f32>(rt, proj, rows, width, "attn_qk_norm_rope proj")?;
    require::<f32>(rt, q_norm_w, s.head_dim as usize, "attn q_norm weight")?;
    require::<f32>(rt, k_norm_w, s.head_dim as usize, "attn k_norm weight")?;
    let q_elems = usize_product(&[rows as usize, s.q_heads as usize, s.head_dim as usize], WHAT)?;
    let cache_elems = usize_product(
        &[
            s.batch as usize,
            kv_capacity as usize,
            s.kv_heads as usize,
            s.head_dim as usize,
        ],
        WHAT,
    )?;
    require::<f32>(rt, targets.q_out, q_elems, "attn q_out")?;
    require::<f32>(rt, targets.k_cache, cache_elems, "attn k_cache")?;
    require::<f32>(rt, targets.v_cache, cache_elems, "attn v_cache")?;
    if rows == 0 {
        return Ok(());
    }
    let mut reads = vec![("proj", proj.buf), ("q_norm", q_norm_w), ("k_norm", k_norm_w)];
    if let RopePos::Buffer(b) | RopePos::PerRow(b) = pos {
        reads.push(("pos_offset", b));
    }
    require_disjoint_writes(
        WHAT,
        &[
            ("q_out", targets.q_out),
            ("k_cache", targets.k_cache),
            ("v_cache", targets.v_cache),
        ],
        &reads,
    )?;
    let units = usize_product(&[rows as usize, s.q_heads as usize + 2 * s.kv_heads as usize], WHAT)?;
    let name = match pos {
        RopePos::Scalar(_) => "qwen35_attn_qk_norm_rope",
        RopePos::Buffer(_) | RopePos::PerRow(_) => "qwen35_attn_qk_norm_rope_posbuf",
    };
    let p = pipeline_for(rt, name, ROWS_PER_TG * 32, 0)?;
    dispatch_groups(
        rt,
        &p,
        (units.div_ceil(ROWS_PER_TG), 1, 1),
        ROWS_PER_TG * 32,
        0,
        |bnd| {
            set_gpu_buf(bnd, proj.buf, 0);
            set_gpu_buf(bnd, q_norm_w, 1);
            set_gpu_buf(bnd, k_norm_w, 2);
            set_gpu_buf(bnd, targets.q_out, 3);
            set_gpu_buf(bnd, targets.k_cache, 4);
            set_gpu_buf(bnd, targets.v_cache, 5);
            set_u32(bnd, s.batch, 6);
            set_u32(bnd, s.seq, 7);
            set_u32(bnd, s.q_heads, 8);
            set_u32(bnd, s.kv_heads, 9);
            set_u32(bnd, s.head_dim, 10);
            set_u32(bnd, s.rotary_dim, 11);
            set_u32(bnd, proj.ld, 12);
            set_u32(bnd, proj.off + q_col, 13);
            set_u32(bnd, proj.off + k_col, 14);
            set_u32(bnd, proj.off + v_col, 15);
            match pos {
                RopePos::Scalar(v) => set_u32(bnd, v, 16),
                RopePos::Buffer(b) | RopePos::PerRow(b) => set_gpu_buf(bnd, b, 16),
            }
            set_u32(bnd, kv_capacity, 17);
            set_f32(bnd, theta, 18);
            set_f32(bnd, eps, 19);
            set_u32(bnd, slot_base, 20);
            set_u32(bnd, u32::from(matches!(pos, RopePos::PerRow(_))), 21);
            set_u32(bnd, q_head_stride, 22);
            set_f32(bnd, weight_bias, 23);
        },
    )
}

/// `out = attn * sigmoid(gate)`, the gate read in place from the fused
/// projection (`proj` as [`attn_qk_norm_rope`] takes it). `attn` is the dense
/// attention output `[rows, q_heads * head_dim]`. An f32 `out` may be `attn`
/// itself when it is dense (`ld = q_heads * head_dim`, `off = 0`).
pub fn attn_output_gate(
    rt: &Arc<GpuRuntime>,
    attn: &GpuBuffer,
    proj: Cols<'_>,
    out: OutCols<'_>,
    rows: u32,
    q_heads: u32,
    head_dim: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_output_gate";
    let name = out_kernel("qwen35_attn_gate", out.dtype, WHAT)?;
    let width = u32_product(&[q_heads, head_dim], WHAT)?;
    let gate_w = u64::from(u32_product(&[width, 2], WHAT)?);
    require::<f32>(
        rt,
        attn,
        usize_product(&[rows as usize, width as usize], WHAT)?,
        "attn_output_gate attn",
    )?;
    require_window::<f32>(rt, proj, u64::from(rows), gate_w, "attn_output_gate proj")?;
    require_out_window(rt, out, u64::from(rows), u64::from(width), "attn_output_gate out")?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let in_place = out.cols.buf.aliases(attn) && out.dtype == DType::F32 && out.cols.ld == width && out.cols.off == 0;
    if in_place {
        require_disjoint_writes(WHAT, &[("out", out.cols.buf)], &[("proj", proj.buf)])?;
    } else {
        require_disjoint_writes(WHAT, &[("out", out.cols.buf)], &[("attn", attn), ("proj", proj.buf)])?;
    }
    let p = rt.pipeline(&name)?;
    dispatch_2d(rt, &p, width as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, attn, 0);
        set_gpu_buf(bnd, proj.buf, 1);
        set_gpu_buf(bnd, out.cols.buf, 2);
        set_u32(bnd, rows, 3);
        set_u32(bnd, q_heads, 4);
        set_u32(bnd, head_dim, 5);
        set_u32(bnd, proj.ld, 6);
        set_u32(bnd, proj.off, 7);
        set_u32(bnd, out.cols.ld, 8);
        set_u32(bnd, out.cols.off, 9);
    })
}

// ---------------------------------------------------------------------- MLP ---

/// `out = silu(gate) * up`, elementwise over `rows x width`: transformers'
/// `Qwen3_5MLP` between its projections. Both inputs are f32 column windows
/// (they may be two windows of one buffer, as a fused `[gate | up]` GEMM would
/// write them). A bf16 `out` is what the down projection's GEMM reads, so it
/// needs no separate cast pass. `out` may not overlap either input.
pub fn swiglu(
    rt: &Arc<GpuRuntime>,
    gate: Cols<'_>,
    up: Cols<'_>,
    out: OutCols<'_>,
    rows: u32,
    width: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::swiglu";
    let name = out_kernel("qwen35_swiglu", out.dtype, WHAT)?;
    let (r, w) = (u64::from(rows), u64::from(width));
    require_window::<f32>(rt, gate, r, w, "swiglu gate")?;
    require_window::<f32>(rt, up, r, w, "swiglu up")?;
    require_out_window(rt, out, r, w, "swiglu out")?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("out", out.cols.buf)], &[("gate", gate.buf), ("up", up.buf)])?;
    let p = rt.pipeline(&name)?;
    dispatch_2d(rt, &p, width as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, gate.buf, 0);
        set_gpu_buf(bnd, up.buf, 1);
        set_gpu_buf(bnd, out.cols.buf, 2);
        set_u32(bnd, rows, 3);
        set_u32(bnd, width, 4);
        set_u32(bnd, gate.ld, 5);
        set_u32(bnd, gate.off, 6);
        set_u32(bnd, up.ld, 7);
        set_u32(bnd, up.off, 8);
        set_u32(bnd, out.cols.ld, 9);
        set_u32(bnd, out.cols.off, 10);
    })
}

/// `resid += y`, elementwise over `rows x width` column windows, in exact f32:
/// the residual add after an output projection when the projection is an
/// exact-f32 GEMM. ([`project_residual`] folds the add into the GEMM epilogue,
/// which only the bf16 and relaxed-f32 GEMMs have.)
///
/// `y` must be a different buffer from `resid`.
pub fn residual_add(rt: &Arc<GpuRuntime>, y: Cols<'_>, resid: Cols<'_>, rows: u32, width: u32) -> Result<(), String> {
    residual_add_at(rt, y, 0, resid, 0, rows, width)
}

/// [`residual_add`] with each window's element 0 living `*_byte` bytes into its buffer.
///
/// A dense tensor view stores that base as [`crate::tensor::Tensor::byte_offset`],
/// which [`Cols::dense`] does not carry. Binding the Metal buffer there keeps
/// `off` as a column offset inside the view.
pub(crate) fn residual_add_at(
    rt: &Arc<GpuRuntime>,
    y: Cols<'_>,
    y_byte: usize,
    resid: Cols<'_>,
    resid_byte: usize,
    rows: u32,
    width: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::residual_add";
    let (r, w) = (u64::from(rows), u64::from(width));
    require_window_at::<f32>(rt, y, y_byte, r, w, "residual_add y")?;
    require_window_at::<f32>(rt, resid, resid_byte, r, w, "residual_add resid")?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("resid", resid.buf)], &[("y", y.buf)])?;
    let p = rt.pipeline("qwen35_residual_add_f32")?;
    dispatch_2d(rt, &p, width as usize, rows as usize, |bnd| {
        set_gpu_buf_offset(bnd, y.buf, y_byte, 0);
        set_gpu_buf_offset(bnd, resid.buf, resid_byte, 1);
        set_u32(bnd, rows, 2);
        set_u32(bnd, width, 3);
        set_u32(bnd, y.ld, 4);
        set_u32(bnd, y.off, 5);
        set_u32(bnd, resid.ld, 6);
        set_u32(bnd, resid.off, 7);
    })
}

/// [`require_window`] plus a byte base. The kernel is bound at `byte_off`, so the
/// window's elements start that many bytes into the allocation.
fn require_window_at<T>(
    rt: &GpuRuntime,
    c: Cols<'_>,
    byte_off: usize,
    rows: u64,
    width: u64,
    what: &str,
) -> Result<(), String> {
    let elem = std::mem::size_of::<T>();
    if elem == 0 || byte_off % elem != 0 {
        return Err(format!("{what}: byte offset {byte_off} is not aligned to {elem}"));
    }
    let base = byte_off / elem;
    let span = window_elems(rows, c.ld, c.off, width, what)?;
    let need = base
        .checked_add(span)
        .ok_or_else(|| format!("{what}: extent overflows"))?;
    require::<T>(rt, c.buf, need, what)
}

/// Qwen3.5's zero-centred RMSNorm over `rows` dense rows of `dim`:
/// `out = x * rsqrt(mean(x^2) + eps) * (1 + w)`, with `w` as the checkpoint
/// holds it (`Qwen3_5RMSNorm`), `1 + w` formed in the kernel. `out` is f32 or
/// bf16 (`out_dtype`); the scale is reduced as [`crate::nn::rms_norm_f32`]
/// reduces it. An f32 `out` may be `x` (each element is read and written by
/// one lane); neither may be `w`, and a bf16 `out` may not overlap `x`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    out: &GpuBuffer,
    out_dtype: DType,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::rms_norm";
    let name = out_kernel("qwen35_rms_norm", out_dtype, WHAT)?;
    validate_rms_scalars(dim, eps, WHAT)?;
    let n = (rows as usize)
        .checked_mul(dim as usize)
        .ok_or_else(|| format!("{WHAT}: rows x dim overflows usize"))?;
    require::<f32>(rt, x, n, "rms_norm x")?;
    require::<f32>(rt, w, dim as usize, "rms_norm w")?;
    if out_dtype == DType::BF16 {
        require::<u16>(rt, out, n, "rms_norm out")?;
    } else {
        require::<f32>(rt, out, n, "rms_norm out")?;
    }
    if rows == 0 {
        return Ok(());
    }
    if out_dtype == DType::BF16 {
        require_disjoint_writes(WHAT, &[("out", out)], &[("x", x), ("w", w)])?;
    } else {
        require_disjoint_writes(WHAT, &[("out", out)], &[("w", w)])?;
    }
    let p = rt.pipeline(&name)?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_gpu_buf(bnd, out, 2);
        set_u32(bnd, rows, 3);
        set_u32(bnd, dim, 4);
        set_f32(bnd, eps, 5);
    })
}

// ------------------------------------------------ matrix-unit prefill attention ---

/// Block geometry of [`attn_prefill_with_tile`]: queries per threadgroup (BQ)
/// by keys per step (BK), on NSG simdgroups. Each is its own entry point,
/// `qwen35_attn_tiled_h256_q{BQ}_k{BK}_sg{NSG}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnTile {
    Q32K32Sg4,
    Q32K64Sg4,
    Q64K32Sg8,
    Q64K64Sg8,
}

impl AttnTile {
    pub const ALL: [AttnTile; 4] = [
        AttnTile::Q32K32Sg4,
        AttnTile::Q32K64Sg4,
        AttnTile::Q64K32Sg8,
        AttnTile::Q64K64Sg8,
    ];

    /// `(BQ, BK, NSG)`.
    pub fn geometry(self) -> (u32, u32, u32) {
        match self {
            AttnTile::Q32K32Sg4 => (32, 32, 4),
            AttnTile::Q32K64Sg4 => (32, 64, 4),
            AttnTile::Q64K32Sg8 => (64, 32, 8),
            AttnTile::Q64K64Sg8 => (64, 64, 8),
        }
    }

    /// The name a bench flag or log uses: `q{BQ}_k{BK}_sg{NSG}`.
    pub fn label(self) -> String {
        let (bq, bk, sg) = self.geometry();
        format!("q{bq}_k{bk}_sg{sg}")
    }
}

/// The tile [`attn_prefill`] uses.
pub const ATTN_PREFILL_TILE: AttnTile = AttnTile::Q32K32Sg4;

/// Which kernel [`attn_prefill_by_length`] dispatches for a query length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAttnKernel {
    /// [`attn_prefill`] ([`ATTN_PREFILL_TILE`]).
    Tiled,
}

/// [`PrefillAttnKernel::Tiled`] at every `tq`. There is no length cutoff.
///
/// Paired and interleaved in one process on an M5 Pro (2026-10-03,
/// `bench_qwen35_layers --paired-attn`, same Q/K/V, ABBA, 9 rounds).
/// `attn_prefill` beat `nn::flash_attn_rows` at both lengths, and the
/// sample ranges did not overlap. Per launch, median / min:
/// T = 200, 0.116 / 0.098 ms vs 0.232 / 0.221 ms; T = 8192, 55.3 / 53.4 ms
/// vs 176.2 / 173.7 ms. A cutoff that kept the scalar kernel below 8192
/// was the slower kernel at T = 200.
pub fn prefill_attn_kernel(tq: u32) -> PrefillAttnKernel {
    let _ = tq;
    PrefillAttnKernel::Tiled
}

/// [`crate::nn::flash_attn_rows`] at head_dim 256 and `window = 0`, with both
/// products on the TensorOps matrix units: `S = Q·Kᵀ` and `P·V` are
/// `matmul2d` over query-by-key blocks ([`ATTN_PREFILL_TILE`]), with an f32 online softmax
/// between them. Same buffers, layouts and masking contract as
/// `flash_attn_rows` (`q`/`o` `[batch, tq, heads, 256]`, `k`/`v`
/// `[batch, capacity, kv_heads, 256]`, live `min(*tkv, capacity)`, query `t`
/// at `*q_pos_offset + t`, key `t` at `*kv_pos_offset + t`, causal).
///
/// Always this tile. [`attn_prefill_by_length`] calls this at every query
/// length ([`prefill_attn_kernel`]). Callers that pin these bits — training
/// attention, the tile sweep — keep calling this function.
///
/// The matrix units sum in a different order from the scalar kernel, so the
/// two agree to f32 rounding, not bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    attn_prefill_with_tile(
        rt,
        q,
        k,
        v,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        dims,
        out_bf16,
        ATTN_PREFILL_TILE,
    )
}

/// Prefill attention. [`prefill_attn_kernel`] selects [`attn_prefill`] at
/// every `dims.tq`.
///
/// Same buffers, causal mask (`window` must be 0), f32 accumulate and
/// `out_bf16` as [`attn_prefill`]. [`attn_prefill`] and
/// [`attn_prefill_with_tile`] stay the TensorOps tile.
/// [`crate::nn::flash_attn_rows`] stays the scalar row kernel.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_by_length(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_prefill_by_length";
    if dims.window != 0 {
        return Err(format!(
            "{WHAT}: window must be 0 (Qwen3.5's full attention is global), got {}",
            dims.window
        ));
    }
    match prefill_attn_kernel(dims.tq) {
        PrefillAttnKernel::Tiled => attn_prefill(rt, q, k, v, o, tkv, q_pos_offset, kv_pos_offset, dims, out_bf16),
    }
}

/// [`attn_prefill`] at an explicit block geometry, for the tuning sweep.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_with_tile(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
    tile: AttnTile,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_prefill";
    const D: u32 = PREFIX_ATTN_HEAD_DIM;
    if dims.window != 0 {
        return Err(format!(
            "{WHAT}: window must be 0 (Qwen3.5's full attention is global), got {}",
            dims.window
        ));
    }
    let kv_capacity = crate::nn::validate_rows_attn_call(
        rt,
        q,
        k,
        v,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        &dims,
        D,
        out_bf16,
        WHAT,
    )?;
    // MPP tensor views index one (batch, head) plane with i32 extents and
    // strides, so a plane's last element must be addressable in i32.
    let q_plane = u64::from(dims.tq) * u64::from(dims.heads) * u64::from(D);
    let kv_plane = u64::from(kv_capacity) * u64::from(dims.heads_kv) * u64::from(D);
    if q_plane > i32::MAX as u64 || kv_plane > i32::MAX as u64 {
        return Err(format!(
            "{WHAT}: a head's plane exceeds i32 indexing (tq = {}, kv capacity = {kv_capacity})",
            dims.tq
        ));
    }
    // Literal names, so the emulator's host-contract check can see which
    // kernels this site binds; it also holds each name's spelled geometry to
    // the kernel's instantiation, and the GPU tests run every tile.
    let entry = match tile {
        AttnTile::Q32K32Sg4 => "qwen35_attn_tiled_h256_q32_k32_sg4",
        AttnTile::Q32K64Sg4 => "qwen35_attn_tiled_h256_q32_k64_sg4",
        AttnTile::Q64K32Sg8 => "qwen35_attn_tiled_h256_q64_k32_sg8",
        AttnTile::Q64K64Sg8 => "qwen35_attn_tiled_h256_q64_k64_sg8",
    };
    let (bq, _, sg) = tile.geometry();
    let threads = sg as usize * 32;
    let groups_y = usize_product(&[dims.batch as usize, dims.heads as usize], WHAT)?;
    let p = pipeline_for(rt, entry, threads, 0)?;
    dispatch_groups(
        rt,
        &p,
        ((dims.tq as usize).div_ceil(bq as usize), groups_y, 1),
        threads,
        0,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, k, 1);
            set_gpu_buf(bnd, v, 2);
            set_gpu_buf(bnd, o, 3);
            set_u32(bnd, dims.batch, 4);
            set_u32(bnd, dims.tq, 5);
            set_gpu_buf(bnd, tkv, 6);
            set_u32(bnd, dims.heads, 7);
            set_u32(bnd, dims.heads_kv, 8);
            set_u32(bnd, dims.window, 9);
            set_f32(bnd, dims.scale, 10);
            set_gpu_buf(bnd, q_pos_offset, 11);
            set_gpu_buf(bnd, kv_pos_offset, 12);
            set_u32(bnd, u32::from(out_bf16), 13);
            set_u32(bnd, kv_capacity, 14);
        },
    )
}

// --------------------------------------------------- shared-prefix attention ---

/// Head dim of [`attn_prefix_rows`]: Qwen3.5's full-attention heads.
pub const PREFIX_ATTN_HEAD_DIM: u32 = 256;
/// Lanes per query row and simdgroups per threadgroup of
/// `qwen35_attn_prefix_rows`: the kernel is `flash_attn_rows` at the
/// instantiation [`crate::nn::rows_lanes_for`] / [`crate::nn::rows_groups_for`]
/// pick for D = 256, so it runs the same per-row arithmetic.
const PREFIX_ATTN_LANES: usize = 16;
const PREFIX_ATTN_SIMDGROUPS: usize = 32;

/// A K/V prefix shared by every batch row: `[capacity, kv_heads, head_dim]`
/// with no batch dimension, holding positions `0 .. len`.
#[derive(Clone, Copy, Debug)]
pub struct SharedPrefix<'a> {
    pub k: &'a GpuBuffer,
    pub v: &'a GpuBuffer,
    pub len: u32,
}

/// Causal attention for many continuations of one prefilled prefix, without
/// copying the prefix's K/V per row.
///
/// Row `b` attends to the shared `prefix` (positions `0 .. prefix.len`) and
/// then to its own suffix cache, `[batch, suffix_capacity, kv_heads, 256]`,
/// whose slot `s` is position `prefix.len + s`; the live suffix length is
/// `min(*suffix_len, suffix_capacity)`, one u32 on the device shared by all
/// rows, like `flash_attn_rows`' `tkv`. Query `t` of `q` (`[batch, tq, heads,
/// 256]`) is at position `*q_pos_offset + t`. Write the suffix with
/// [`attn_qk_norm_rope_suffix`], which rotates to the absolute position but
/// caches at the suffix-relative slot.
///
/// The result is bit-identical to [`crate::nn::flash_attn_rows`] over a per-row
/// cache `prefix ‖ suffix_b` with `window = 0`: this is that kernel with only
/// the key address changed. `dims.window` must be 0 (Qwen3.5's full attention
/// is global), and only head_dim 256 is compiled.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefix_rows(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    suffix_len: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    prefix_rows_impl(
        rt,
        q,
        prefix,
        suffix_k,
        suffix_v,
        RowLens::Shared {
            suffix_len,
            q_pos_offset,
        },
        o,
        dims,
        out_bf16,
    )
}

/// [`attn_prefix_rows`] for continuations of **different lengths**:
/// `suffix_lens` and `q_pos_offsets` are `[batch]` device u32s, row b's live
/// suffix length and the absolute position of its query 0. Each row is
/// exactly what [`attn_prefix_rows`] computes for that row alone with those
/// two values. Rows are right-padded to `dims.tq`; outputs for a row's padded
/// queries are computed but meaningless.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefix_rows_varlen(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    suffix_lens: &GpuBuffer,
    q_pos_offsets: &GpuBuffer,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    prefix_rows_impl(
        rt,
        q,
        prefix,
        suffix_k,
        suffix_v,
        RowLens::PerRow {
            suffix_lens,
            q_pos_offsets,
        },
        o,
        dims,
        out_bf16,
    )
}

/// The device lengths the shared-prefix kernels read: one value for every
/// row, or one per row (the kernels' `row_stride` 0 or 1).
#[derive(Clone, Copy)]
enum RowLens<'a> {
    Shared {
        suffix_len: &'a GpuBuffer,
        q_pos_offset: &'a GpuBuffer,
    },
    PerRow {
        suffix_lens: &'a GpuBuffer,
        q_pos_offsets: &'a GpuBuffer,
    },
}

impl<'a> RowLens<'a> {
    fn buffers(self) -> (&'a GpuBuffer, &'a GpuBuffer) {
        match self {
            RowLens::Shared {
                suffix_len,
                q_pos_offset,
            } => (suffix_len, q_pos_offset),
            RowLens::PerRow {
                suffix_lens,
                q_pos_offsets,
            } => (suffix_lens, q_pos_offsets),
        }
    }

    fn row_stride(self) -> u32 {
        u32::from(matches!(self, RowLens::PerRow { .. }))
    }
}

#[allow(clippy::too_many_arguments)]
fn prefix_rows_impl(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    lens: RowLens<'_>,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_prefix_rows";
    let Some(suffix_cap) = validate_prefix_attn(rt, WHAT, q, prefix, suffix_k, suffix_v, lens, o, &dims, out_bf16)?
    else {
        return Ok(());
    };
    let (suffix_len, q_pos_offset) = lens.buffers();
    let rows_per_tg = PREFIX_ATTN_SIMDGROUPS * (32 / PREFIX_ATTN_LANES);
    let threads = PREFIX_ATTN_SIMDGROUPS * 32;
    let groups_y = usize_product(&[dims.batch as usize, dims.heads as usize], WHAT)?;
    let p = pipeline_for(rt, "qwen35_attn_prefix_rows", threads, 0)?;
    dispatch_groups(
        rt,
        &p,
        ((dims.tq as usize).div_ceil(rows_per_tg), groups_y, 1),
        threads,
        0,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, prefix.k, 1);
            set_gpu_buf(bnd, prefix.v, 2);
            set_gpu_buf(bnd, suffix_k, 3);
            set_gpu_buf(bnd, suffix_v, 4);
            set_gpu_buf(bnd, o, 5);
            set_u32(bnd, dims.tq, 6);
            set_u32(bnd, prefix.len, 7);
            set_gpu_buf(bnd, suffix_len, 8);
            set_u32(bnd, dims.heads, 9);
            set_u32(bnd, dims.heads_kv, 10);
            set_f32(bnd, dims.scale, 11);
            set_gpu_buf(bnd, q_pos_offset, 12);
            set_u32(bnd, u32::from(out_bf16), 13);
            set_u32(bnd, suffix_cap, 14);
            set_u32(bnd, lens.row_stride(), 15);
        },
    )
}

/// Keys per chunk and lanes per key of the shared-prefix decode: the
/// [`crate::nn::flash_attn_decode`] instantiation nn picks at D = 256
/// ([`crate::nn::decode_chunk_for`], [`crate::nn::decode_lanes_for`]).
const PREFIX_DECODE_CHUNK: usize = 128;
// No host arithmetic depends on it; it exists to be pinned against nn (unit
// test) and against the kernel's constant (the emulator's host_contract).
#[allow(dead_code)]
const PREFIX_DECODE_LANES: usize = 16;
/// Reduce-pass width: [`crate::nn::DECODE_REDUCE_THREADS`], capped at D as nn
/// caps it.
const PREFIX_DECODE_REDUCE_THREADS: usize = 256;

/// [`attn_prefix_rows`] for a single query per row (`dims.tq == 1`), split
/// over the keys like [`crate::nn::flash_attn_decode`]: one simdgroup per
/// 128-key chunk per head, then a reduce over the chunks. The rows kernel runs
/// one simdgroup per (row, head) through all `P + S` keys in series, which is
/// the shape `flash_attn_decode` exists to avoid.
///
/// Same arguments and layouts as [`attn_prefix_rows`]; `o` is `[batch, heads,
/// 256]`. The result is bit-identical to `nn::flash_attn_decode` over a per-row
/// cache `prefix ‖ suffix_b` with `window = 0` and `kv_pos_offset = 0`: both
/// passes are that kernel's D = 256 instantiation with only the key address
/// (partial) and the live key count (both) changed.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefix_decode(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    suffix_len: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    prefix_decode_impl(
        rt,
        q,
        prefix,
        suffix_k,
        suffix_v,
        RowLens::Shared {
            suffix_len,
            q_pos_offset,
        },
        o,
        dims,
        out_bf16,
    )
}

/// [`attn_prefix_decode`] with a live suffix length and query position per
/// row (`[batch]` device u32s), for continuations of different lengths. Each
/// row is exactly what [`attn_prefix_decode`] computes for it alone.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefix_decode_varlen(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    suffix_lens: &GpuBuffer,
    q_pos_offsets: &GpuBuffer,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    prefix_decode_impl(
        rt,
        q,
        prefix,
        suffix_k,
        suffix_v,
        RowLens::PerRow {
            suffix_lens,
            q_pos_offsets,
        },
        o,
        dims,
        out_bf16,
    )
}

#[allow(clippy::too_many_arguments)]
fn prefix_decode_impl(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    lens: RowLens<'_>,
    o: &GpuBuffer,
    dims: crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_prefix_decode";
    if dims.tq != 1 {
        return Err(format!(
            "{WHAT}: one query per row (tq = 1), got tq = {}; use attn_prefix_rows",
            dims.tq
        ));
    }
    let Some(suffix_cap) = validate_prefix_attn(rt, WHAT, q, prefix, suffix_k, suffix_v, lens, o, &dims, out_bf16)?
    else {
        return Ok(());
    };
    let (suffix_len, q_pos_offset) = lens.buffers();
    // The device suffix length is unknown here, so the grid covers every key
    // the capacities allow; both passes clamp it and derive the same live
    // chunk count, so every chunk the reduce reads was written by this
    // partial pass (flash_attn_decode's invariant, unchanged).
    let key_cap = prefix.len as usize + suffix_cap as usize;
    let chunks = key_cap.div_ceil(PREFIX_DECODE_CHUNK).max(1);
    let heads = dims.heads as usize;
    let group = (dims.heads / dims.heads_kv) as usize;
    // The GQA group shares a threadgroup, as nn's D = 256 decode does.
    let sgs = crate::nn::DecodeHeadBlock::Group.simdgroups(heads, group).unwrap_or(1);
    let bh = usize_product(&[dims.batch as usize, heads], WHAT)?;
    let scratch_bytes = usize_product(
        &[
            bh,
            chunks,
            PREFIX_ATTN_HEAD_DIM as usize + 2,
            std::mem::size_of::<f32>(),
        ],
        WHAT,
    )?;
    // Resolve both pipelines before encoding either: a failure after the
    // partial pass was encoded would leave a producer with no consumer.
    let partial = pipeline_for(rt, "qwen35_attn_prefix_decode_partial", sgs * 32, 0)?;
    let reduce = pipeline_for(rt, "qwen35_attn_prefix_decode_reduce", PREFIX_DECODE_REDUCE_THREADS, 0)?;
    let scratch = rt.alloc_buffer(scratch_bytes)?;
    dispatch_groups(
        rt,
        &partial,
        (chunks, dims.batch as usize * (heads / sgs), 1),
        sgs * 32,
        0,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, prefix.k, 1);
            set_gpu_buf(bnd, prefix.v, 2);
            set_gpu_buf(bnd, suffix_k, 3);
            set_gpu_buf(bnd, suffix_v, 4);
            set_gpu_buf(bnd, &scratch, 5);
            set_u32(bnd, prefix.len, 6);
            set_gpu_buf(bnd, suffix_len, 7);
            set_u32(bnd, dims.heads, 8);
            set_u32(bnd, dims.heads_kv, 9);
            set_f32(bnd, dims.scale, 10);
            set_gpu_buf(bnd, q_pos_offset, 11);
            set_u32(bnd, suffix_cap, 12);
            set_u32(bnd, lens.row_stride(), 13);
        },
    )?;
    // The binder orders this after the partial pass, as for the GDN prep and
    // scan.
    dispatch_groups(rt, &reduce, (1, bh, 1), PREFIX_DECODE_REDUCE_THREADS, 0, |bnd| {
        set_gpu_buf(bnd, &scratch, 0);
        set_gpu_buf(bnd, o, 1);
        set_u32(bnd, prefix.len, 2);
        set_gpu_buf(bnd, suffix_len, 3);
        set_u32(bnd, dims.heads, 4);
        set_u32(bnd, u32::from(out_bf16), 5);
        set_u32(bnd, suffix_cap, 6);
        set_u32(bnd, lens.row_stride(), 7);
    })
}

/// The host checks both shared-prefix entry points make. Returns the suffix
/// capacity, or `None` when the shape has no work.
#[allow(clippy::too_many_arguments)]
fn validate_prefix_attn(
    rt: &GpuRuntime,
    what: &str,
    q: &GpuBuffer,
    prefix: SharedPrefix<'_>,
    suffix_k: &GpuBuffer,
    suffix_v: &GpuBuffer,
    lens: RowLens<'_>,
    o: &GpuBuffer,
    dims: &crate::nn::AttnDims,
    out_bf16: bool,
) -> Result<Option<u32>, String> {
    let (suffix_len, q_pos_offset) = lens.buffers();
    let len_elems = match lens {
        RowLens::Shared { .. } => 1,
        RowLens::PerRow { .. } => dims.batch as usize,
    };
    const D: u32 = PREFIX_ATTN_HEAD_DIM;
    if dims.window != 0 {
        return Err(format!(
            "{what}: window must be 0 (global causal attention), got {}",
            dims.window
        ));
    }
    for (name, b) in [
        ("q", q),
        ("prefix k", prefix.k),
        ("prefix v", prefix.v),
        ("suffix k", suffix_k),
        ("suffix v", suffix_v),
        ("o", o),
    ] {
        require_runtime(rt, b, &format!("{what} {name}"))?;
    }
    // Q/O extents, head grouping, scale, and o against q and the suffix.
    let suffix_cap = crate::nn::validate_attn_storage(dims, D, q, suffix_k, suffix_v, o, out_bf16)
        .map_err(|e| format!("{what}: {e}"))?;
    let prefix_cap = crate::nn::attn_kv_capacity(prefix.k, prefix.v, 1, dims.heads_kv, D)
        .map_err(|e| format!("{what}: prefix {e}"))?;
    if prefix.len > prefix_cap {
        return Err(format!(
            "{what}: prefix length {} exceeds the prefix K/V capacity {prefix_cap}",
            prefix.len
        ));
    }
    if u64::from(prefix.len) + u64::from(suffix_cap) > u64::from(u32::MAX) {
        return Err(format!(
            "{what}: prefix length {} plus suffix capacity {suffix_cap} exceeds u32 positions",
            prefix.len
        ));
    }
    require::<u32>(rt, suffix_len, len_elems, &format!("{what} suffix_len"))?;
    require::<u32>(rt, q_pos_offset, len_elems, &format!("{what} q_pos_offset"))?;
    if dims.batch == 0 || dims.tq == 0 || dims.heads == 0 {
        return Ok(None);
    }
    require_disjoint_writes(
        what,
        &[("o", o)],
        &[
            ("prefix k", prefix.k),
            ("prefix v", prefix.v),
            ("suffix_len", suffix_len),
            ("q_pos_offset", q_pos_offset),
        ],
    )?;
    Ok(Some(suffix_cap))
}

// ----------------------------------------------------------------- scoring ---

/// The LM head rows to score against.
#[derive(Clone, Copy, Debug)]
pub struct LmHead<'a> {
    /// `[vocab, hidden]` row-major, [`DType::F32`] or [`DType::BF16`].
    pub weight: &'a GpuBuffer,
    pub dtype: DType,
    pub vocab: u32,
}

/// Final norm + LM head, for the answer tokens at the slot rows only.
///
/// For each slot `s` (a row index into `hidden_states` `[rows, hidden]`) and
/// answer `a` (a token id):
///
/// ```text
/// logits[s, a]   = rms_norm(hidden_states[slots[s]]) * (norm_w + w_offset) . lm_head[answers[a]]
/// logprobs[s, :] = log_softmax(logits[s, :])        (over the answer set only)
/// ```
///
/// `w_offset = 1.0` is Qwen3.5's zero-centred final norm. `slots` and `answers`
/// are u32 device buffers; an out-of-range entry scores NaN rather than reading
/// out of bounds, since the host cannot see their contents without a sync.
#[allow(clippy::too_many_arguments)]
pub fn score_answer_rows(
    rt: &Arc<GpuRuntime>,
    hidden_states: &GpuBuffer,
    rows: u32,
    hidden: u32,
    slots: &GpuBuffer,
    n_slots: u32,
    norm_w: &GpuBuffer,
    w_offset: f32,
    eps: f32,
    lm_head: LmHead<'_>,
    answers: &GpuBuffer,
    n_answers: u32,
    logits: &GpuBuffer,
    logprobs: &GpuBuffer,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::score_answer_rows";
    let name = out_kernel("qwen35_score_rows", lm_head.dtype, "qwen35::score_answer_rows lm_head")?;
    if hidden == 0 || n_answers == 0 || n_answers > MAX_ANSWERS {
        return Err(format!(
            "{WHAT}: hidden must be non-zero and n_answers in 1..={MAX_ANSWERS}"
        ));
    }
    // An invalid slot or answer id still reads row 0 (then discards it) rather
    // than branching around the load, so row 0 must exist in both tables.
    if rows == 0 || lm_head.vocab == 0 {
        return Err(format!(
            "{WHAT}: rows and vocab must be non-zero (invalid indices read row 0)"
        ));
    }
    if !eps.is_finite() || eps <= 0.0 || !w_offset.is_finite() {
        return Err(format!("{WHAT}: eps must be positive and w_offset finite"));
    }
    require::<f32>(
        rt,
        hidden_states,
        usize_product(&[rows as usize, hidden as usize], WHAT)?,
        "score hidden_states",
    )?;
    require::<u32>(rt, slots, n_slots as usize, "score slots")?;
    require::<f32>(rt, norm_w, hidden as usize, "score norm weight")?;
    let head_elems = usize_product(&[lm_head.vocab as usize, hidden as usize], WHAT)?;
    match lm_head.dtype {
        DType::BF16 => require::<u16>(rt, lm_head.weight, head_elems, "score lm_head")?,
        _ => require::<f32>(rt, lm_head.weight, head_elems, "score lm_head")?,
    }
    require::<u32>(rt, answers, n_answers as usize, "score answers")?;
    let out_elems = usize_product(&[n_slots as usize, n_answers as usize], WHAT)?;
    require::<f32>(rt, logits, out_elems, "score logits")?;
    require::<f32>(rt, logprobs, out_elems, "score logprobs")?;
    require_runtime(rt, lm_head.weight, "score lm_head")?;
    if n_slots == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        WHAT,
        &[("logits", logits), ("logprobs", logprobs)],
        &[
            ("hidden_states", hidden_states),
            ("slots", slots),
            ("norm_w", norm_w),
            ("lm_head", lm_head.weight),
            ("answers", answers),
        ],
    )?;
    // Reduction partials, then the answers' logits.
    let tg_bytes = score_tg_bytes(n_answers);
    let p = pipeline_for(rt, &name, SCORE_THREADS, tg_bytes)?;
    dispatch_groups(rt, &p, (n_slots as usize, 1, 1), SCORE_THREADS, tg_bytes, |bnd| {
        set_gpu_buf(bnd, hidden_states, 0);
        set_gpu_buf(bnd, slots, 1);
        set_gpu_buf(bnd, norm_w, 2);
        set_gpu_buf(bnd, lm_head.weight, 3);
        set_gpu_buf(bnd, answers, 4);
        set_gpu_buf(bnd, logits, 5);
        set_gpu_buf(bnd, logprobs, 6);
        set_u32(bnd, rows, 7);
        set_u32(bnd, hidden, 8);
        set_u32(bnd, n_answers, 9);
        set_u32(bnd, lm_head.vocab, 10);
        set_f32(bnd, eps, 11);
        set_f32(bnd, w_offset, 12);
    })
}

// --------------------------------------------------------------- embedding ---

/// The embedding gather on the device: `out[r, :] = table[ids[r], :]`, bf16
/// widened exactly to f32. `table` is the `[vocab, hidden]` vocabulary matrix
/// (Qwen3.5 ties it to the LM head, so it is the [`LmHead`] that
/// [`score_answer_rows`] reads); only [`DType::BF16`] is compiled. `ids` is
/// `n` u32 token ids on the device and `out` is dense `[n, hidden]` f32, the
/// residual stream's first value, so the whole forward stays in one command
/// buffer with no host gather.
///
/// An id `>= vocab` cannot be seen by the host; its row comes out NaN and
/// every other row is unaffected.
pub fn embed_rows(
    rt: &Arc<GpuRuntime>,
    ids: &GpuBuffer,
    n: u32,
    table: LmHead<'_>,
    hidden: u32,
    out: &GpuBuffer,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::embed_rows";
    let kernel = match table.dtype {
        DType::BF16 => "qwen35_embed_rows_bf16",
        DType::F32 => "qwen35_embed_rows_f32",
        d => return Err(format!("{WHAT}: bf16 and f32 tables are compiled, got {d:?}")),
    };
    if table.vocab == 0 || hidden == 0 {
        return Err(format!("{WHAT}: vocab and hidden must be non-zero"));
    }
    let table_elems = usize_product(&[table.vocab as usize, hidden as usize], WHAT)?;
    if table.dtype == DType::BF16 {
        require::<u16>(rt, table.weight, table_elems, "embed_rows table")?;
    } else {
        require::<f32>(rt, table.weight, table_elems, "embed_rows table")?;
    }
    require::<u32>(rt, ids, n as usize, "embed_rows ids")?;
    require::<f32>(
        rt,
        out,
        usize_product(&[n as usize, hidden as usize], WHAT)?,
        "embed_rows out",
    )?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(WHAT, &[("out", out)], &[("ids", ids), ("table", table.weight)])?;
    let p = rt.pipeline(kernel)?;
    dispatch_2d(rt, &p, hidden as usize, n as usize, |bnd| {
        set_gpu_buf(bnd, ids, 0);
        set_gpu_buf(bnd, table.weight, 1);
        set_gpu_buf(bnd, out, 2);
        set_u32(bnd, n, 3);
        set_u32(bnd, hidden, 4);
        set_u32(bnd, table.vocab, 5);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_puts_each_linear_transposed_side_by_side() {
        // Two linears on in_features = 2: A is 3x2, B is 1x2.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0];
        let packed = pack_linear_weights_f32(&[&a, &b], &[3, 1], 2).unwrap();
        // [in=2, out=4]: row k holds A[:, k] then B[:, k].
        assert_eq!(packed, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
        assert!(pack_linear_weights_f32(&[&a], &[2], 2).is_err());
    }

    #[test]
    fn gdn_layout_matches_the_packing_order() {
        let l = GdnProjLayout::new(16, 32, 128).unwrap();
        assert_eq!(l.conv_dim(), 2 * 2048 + 4096);
        assert_eq!(l.z_off(), l.conv_dim());
        assert_eq!(l.b_off(), l.conv_dim() + 4096);
        assert_eq!(l.a_off(), l.b_off() + 32);
        assert_eq!(l.width() as usize, l.part_widths().iter().sum::<usize>());
        assert!(
            GdnProjLayout::new(16, 24, 128).is_err(),
            "v_heads not a multiple of k_heads"
        );
        assert!(GdnProjLayout::new(1, 1, 48).is_err(), "v_dim not a multiple of 32");
        assert!(AttnProjLayout::new(u32::MAX / 4, 1, 4).is_err(), "width overflow");
    }

    #[test]
    fn window_extent_is_exact_and_rejects_overhang() {
        assert_eq!(window_elems(3, 10, 2, 5, "t").unwrap(), 2 * 10 + 7);
        assert!(window_elems(3, 10, 6, 5, "t").is_err());
        assert_eq!(window_elems(0, 10, 0, 5, "t").unwrap(), 0);
    }

    #[test]
    fn prefill_attn_is_tiled_at_every_length() {
        // Paired kernel times (see `prefill_attn_kernel`) had the tile faster
        // at T = 200 and at T = 8192. A length cutoff is the bug that test
        // used to pin.
        for t in [0, 200, 2048, 8191, 8192, 8193] {
            assert_eq!(prefill_attn_kernel(t), PrefillAttnKernel::Tiled, "tq={t}");
        }
    }

    #[test]
    fn prefix_attention_is_flash_attn_rows_instantiation() {
        // `qwen35_attn_prefix_rows` copies flash_attn_rows' body at these
        // knobs; its bit-for-bit equality with `nn::flash_attn_rows` holds only
        // while nn picks the same ones for D = 256.
        let d = PREFIX_ATTN_HEAD_DIM;
        assert_eq!(crate::nn::rows_lanes_for(d).width(), PREFIX_ATTN_LANES);
        assert_eq!(crate::nn::rows_groups_for(d).count(), PREFIX_ATTN_SIMDGROUPS);
        // Likewise the decode passes and flash_attn_decode.
        assert_eq!(crate::nn::decode_chunk_for(d).keys(), PREFIX_DECODE_CHUNK);
        assert_eq!(crate::nn::decode_lanes_for(d).width(), PREFIX_DECODE_LANES);
        assert_eq!(crate::nn::decode_head_block_for(d), crate::nn::DecodeHeadBlock::Group);
        assert_eq!(
            crate::nn::DECODE_REDUCE_THREADS.min(d as usize),
            PREFIX_DECODE_REDUCE_THREADS
        );
    }

    #[test]
    fn threadgroup_budgets_fit_32k() {
        for bytes in [
            PREP_TG_BYTES,
            SCAN_TG_BYTES,
            SCAN16_TG_BYTES,
            REC_TG_BYTES,
            score_tg_bytes(MAX_ANSWERS),
        ] {
            assert!(bytes <= 32 * 1024 && bytes % 16 == 0, "{bytes}");
        }
    }
}
