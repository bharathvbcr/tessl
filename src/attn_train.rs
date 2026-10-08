//! Qwen3.5's full attention for training: a forward that also saves each
//! query row's log-sum-exp, and the backward that rebuilds the attention
//! probabilities from it.
//!
//! Causal over positions `0..seq` on both sides, no window, grouped KV heads
//! (`q_heads / kv_heads` query heads per KV head), head dim
//! [`ATTN_TRAIN_HEAD_DIM`]: the attention Qwen3.5 trains, where the inference
//! path ([`crate::qwen35::attn_prefill`]) also serves offsets, ragged lengths
//! and caches larger than the sequence.
//!
//! [`attn_train_forward`] is the inference forward's tiled kernel
//! (`qwen35_attn_tiled_lse_*`, the same body as [`crate::qwen35::attn_prefill`]
//! at its default geometry) with the log-sum-exp written out.
//! [`attn_train_backward`] is FlashAttention-2's backward on the matrix units
//! (`kernels/qwen35_attn_bwd.metal`): dQ per query block, dK and dV per key
//! block, each gradient written once by the threadgroup that owns it, so
//! gradients are deterministic.
//!
//! All operands are dense f32: `q`, `o`, `d_o`, `dq` `[B, T, Hq, D]`; `k`,
//! `v`, `dk`, `dv` `[B, T, Hkv, D]`; `lse` `[B, Hq, T]`.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_2d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
use crate::nn::{require, require_disjoint_writes};
use crate::qwen35::PREFIX_ATTN_HEAD_DIM;
use crate::runtime::{BufferKind, GpuRuntime};
use crate::tensor::GpuBuffer;

/// Head dim the training kernels are compiled for (Qwen3.5's at every size).
pub const ATTN_TRAIN_HEAD_DIM: u32 = PREFIX_ATTN_HEAD_DIM;
/// Query rows and key rows per threadgroup, and simdgroups per threadgroup,
/// as spelled in the entry points' names.
const BQ: u32 = 32;
const BK: u32 = 32;
const THREADS: usize = 4 * 32;

/// Shape of one training attention call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnTrainDims {
    pub batch: u32,
    pub seq: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// The score scale, `head_dim^-0.5` for Qwen3.5.
    pub scale: f32,
}

impl AttnTrainDims {
    fn validate(&self, what: &str) -> Result<(), String> {
        if self.q_heads == 0 || self.kv_heads == 0 || self.q_heads % self.kv_heads != 0 {
            return Err(format!(
                "{what}: q_heads must be a non-zero multiple of kv_heads, got {} and {}",
                self.q_heads, self.kv_heads
            ));
        }
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(format!("{what}: scale must be finite and positive"));
        }
        // MPP tensor views index one (batch, head) plane with i32 extents
        // and strides.
        let plane = u64::from(self.seq) * u64::from(self.q_heads) * u64::from(ATTN_TRAIN_HEAD_DIM);
        if plane > i32::MAX as u64 {
            return Err(format!(
                "{what}: a head's plane exceeds i32 indexing (seq = {})",
                self.seq
            ));
        }
        let lse = u64::from(self.batch) * u64::from(self.q_heads) * u64::from(self.seq);
        if lse > u64::from(u32::MAX) {
            return Err(format!("{what}: batch x q_heads x seq exceeds u32"));
        }
        Ok(())
    }

    /// Elements of `q`, `o`, `d_o`, `dq`.
    fn q_len(&self) -> usize {
        self.rows() * self.q_heads as usize * ATTN_TRAIN_HEAD_DIM as usize
    }

    /// Elements of `k`, `v`, `dk`, `dv`.
    fn kv_len(&self) -> usize {
        self.rows() * self.kv_heads as usize * ATTN_TRAIN_HEAD_DIM as usize
    }

    /// Elements of `lse`.
    pub fn lse_len(&self) -> usize {
        self.batch as usize * self.q_heads as usize * self.seq as usize
    }

    fn rows(&self) -> usize {
        self.batch as usize * self.seq as usize
    }
}

/// What a training attention call needs besides its operands: the key count
/// and zero position offset the tiled forward reads from device memory, and
/// the backward's `rowsum(dO ∘ O)` `[B, Hq, T]`.
pub struct AttnTrainWorkspace {
    dims: AttnTrainDims,
    tkv: GpuBuffer,
    zero: GpuBuffer,
    dvec: GpuBuffer,
}

impl AttnTrainWorkspace {
    pub fn new(rt: &Arc<GpuRuntime>, dims: AttnTrainDims) -> Result<Self, String> {
        dims.validate("AttnTrainWorkspace")?;
        // Fresh buffers, written without waiting for the GPU.
        Ok(Self {
            dims,
            tkv: rt.alloc_buffer_from_u32(&[dims.seq])?,
            zero: rt.alloc_buffer_from_u32(&[0])?,
            dvec: rt.alloc_buffer(dims.lse_len().max(1) * std::mem::size_of::<f32>())?,
        })
    }

    pub fn dims(&self) -> AttnTrainDims {
        self.dims
    }

    /// Device bytes [`Self::new`] allocates for `dims`, each buffer at the
    /// size the pool makes it ([`GpuRuntime::allocated_bytes_for`]).
    pub fn allocated_bytes_for(dims: AttnTrainDims) -> u64 {
        let u32_slot = GpuRuntime::allocated_bytes_for(std::mem::size_of::<u32>(), BufferKind::Cold);
        let dvec = dims.lse_len().max(1) * std::mem::size_of::<f32>();
        2 * u32_slot + GpuRuntime::allocated_bytes_for(dvec, BufferKind::Cold)
    }

    fn check(&self, dims: &AttnTrainDims, what: &str) -> Result<(), String> {
        if *dims != self.dims {
            return Err(format!(
                "{what}: the workspace was made for {:?}, not {dims:?}",
                self.dims
            ));
        }
        Ok(())
    }

    fn buffers(&self) -> [(&'static str, &GpuBuffer); 3] {
        [
            ("workspace tkv", &self.tkv),
            ("workspace offsets", &self.zero),
            ("workspace dvec", &self.dvec),
        ]
    }
}

fn pipeline(
    rt: &Arc<GpuRuntime>,
    name: &str,
    what: &str,
) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let p = rt.pipeline(name)?;
    if p.maxTotalThreadsPerThreadgroup() < THREADS {
        return Err(format!("{what}: {name} cannot run {THREADS} threads per threadgroup"));
    }
    Ok(p)
}

/// Causal attention `o = softmax(scale * q kᵀ) v` per head, query head h
/// reading KV head `h / (q_heads / kv_heads)`, and `lse[b, h, t]`, the
/// log-sum-exp of row t's scaled scores, for [`attn_train_backward`].
#[allow(clippy::too_many_arguments)]
pub fn attn_train_forward(
    rt: &Arc<GpuRuntime>,
    dims: &AttnTrainDims,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lse: &GpuBuffer,
    ws: &AttnTrainWorkspace,
) -> Result<(), String> {
    const WHAT: &str = "attn_train::attn_train_forward";
    dims.validate(WHAT)?;
    ws.check(dims, WHAT)?;
    for (b, len, name) in [
        (q, dims.q_len(), "q"),
        (k, dims.kv_len(), "k"),
        (v, dims.kv_len(), "v"),
        (o, dims.q_len(), "o"),
        (lse, dims.lse_len(), "lse"),
    ] {
        require::<f32>(rt, b, len, format_args!("{WHAT} {name}"))?;
    }
    let [w0, w1, w2] = ws.buffers();
    require_disjoint_writes(
        WHAT,
        &[("o", o), ("lse", lse)],
        &[("q", q), ("k", k), ("v", v), w0, w1, w2],
    )?;
    if dims.batch == 0 || dims.seq == 0 {
        return Ok(());
    }
    let p = pipeline(rt, "qwen35_attn_tiled_lse_h256_q32_k32_sg4", WHAT)?;
    let groups_y = dims.batch as usize * dims.q_heads as usize;
    dispatch_2d_tg(
        rt,
        &p,
        (dims.seq as usize).div_ceil(BQ as usize),
        groups_y,
        THREADS,
        |bnd| {
            set_gpu_buf(bnd, q, 0);
            set_gpu_buf(bnd, k, 1);
            set_gpu_buf(bnd, v, 2);
            set_gpu_buf(bnd, o, 3);
            set_u32(bnd, dims.batch, 4);
            set_u32(bnd, dims.seq, 5);
            set_gpu_buf(bnd, &ws.tkv, 6);
            set_u32(bnd, dims.q_heads, 7);
            set_u32(bnd, dims.kv_heads, 8);
            set_u32(bnd, 0, 9);
            set_f32(bnd, dims.scale, 10);
            set_gpu_buf(bnd, &ws.zero, 11);
            set_gpu_buf(bnd, &ws.zero, 12);
            set_u32(bnd, 0, 13);
            set_u32(bnd, dims.seq, 14);
            set_gpu_buf(bnd, lse, 15);
        },
    )
}

/// The backward's three block kernels.
#[derive(Clone, Copy)]
enum AttnBwdStage {
    Dq,
    Dk,
    Dv,
}

/// The gradients of [`attn_train_backward`]'s outputs.
#[derive(Clone, Copy, Debug)]
pub struct AttnTrainGrads<'a> {
    pub dq: &'a GpuBuffer,
    pub dk: &'a GpuBuffer,
    pub dv: &'a GpuBuffer,
}

/// The backward of [`attn_train_forward`] from `d_o`, given its inputs, its
/// output `o` and the `lse` it saved. `dq`, `dk`, `dv` are overwritten.
#[allow(clippy::too_many_arguments)]
pub fn attn_train_backward(
    rt: &Arc<GpuRuntime>,
    dims: &AttnTrainDims,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    lse: &GpuBuffer,
    d_o: &GpuBuffer,
    grads: &AttnTrainGrads<'_>,
    ws: &AttnTrainWorkspace,
) -> Result<(), String> {
    const WHAT: &str = "attn_train::attn_train_backward";
    dims.validate(WHAT)?;
    ws.check(dims, WHAT)?;
    for (b, len, name) in [
        (q, dims.q_len(), "q"),
        (k, dims.kv_len(), "k"),
        (v, dims.kv_len(), "v"),
        (o, dims.q_len(), "o"),
        (lse, dims.lse_len(), "lse"),
        (d_o, dims.q_len(), "d_o"),
        (grads.dq, dims.q_len(), "dq"),
        (grads.dk, dims.kv_len(), "dk"),
        (grads.dv, dims.kv_len(), "dv"),
    ] {
        require::<f32>(rt, b, len, format_args!("{WHAT} {name}"))?;
    }
    require_disjoint_writes(
        WHAT,
        &[
            ("dq", grads.dq),
            ("dk", grads.dk),
            ("dv", grads.dv),
            ("workspace dvec", &ws.dvec),
        ],
        &[("q", q), ("k", k), ("v", v), ("o", o), ("lse", lse), ("d_o", d_o)],
    )?;
    if dims.batch == 0 || dims.seq == 0 {
        return Ok(());
    }
    let bh = dims.batch * dims.q_heads;
    let p = rt.pipeline("qwen35_attn_bwd_dvec_f32")?;
    dispatch_2d(rt, &p, dims.seq as usize, bh as usize, |bnd| {
        set_gpu_buf(bnd, o, 0);
        set_gpu_buf(bnd, d_o, 1);
        set_gpu_buf(bnd, &ws.dvec, 2);
        set_u32(bnd, dims.seq, 3);
        set_u32(bnd, dims.q_heads, 4);
        set_u32(bnd, bh, 5);
    })?;
    // dQ per query block of each query head; dK and dV per key block of each
    // KV head. Literal names in one match, so the emulator's host-contract
    // check can see which kernels this site binds.
    for stage in [AttnBwdStage::Dq, AttnBwdStage::Dk, AttnBwdStage::Dv] {
        let name = match stage {
            AttnBwdStage::Dq => "qwen35_attn_bwd_dq_h256_q32_k32_sg4",
            AttnBwdStage::Dk => "qwen35_attn_bwd_dk_h256_q32_k32_sg4",
            AttnBwdStage::Dv => "qwen35_attn_bwd_dv_h256_q32_k32_sg4",
        };
        let (out, rows_per_group, heads) = match stage {
            AttnBwdStage::Dq => (grads.dq, BQ, dims.q_heads),
            AttnBwdStage::Dk => (grads.dk, BK, dims.kv_heads),
            AttnBwdStage::Dv => (grads.dv, BK, dims.kv_heads),
        };
        let p = pipeline(rt, name, WHAT)?;
        let groups_y = dims.batch as usize * heads as usize;
        dispatch_2d_tg(
            rt,
            &p,
            (dims.seq as usize).div_ceil(rows_per_group as usize),
            groups_y,
            THREADS,
            |bnd| {
                set_gpu_buf(bnd, q, 0);
                set_gpu_buf(bnd, k, 1);
                set_gpu_buf(bnd, v, 2);
                set_gpu_buf(bnd, d_o, 3);
                set_gpu_buf(bnd, lse, 4);
                set_gpu_buf(bnd, &ws.dvec, 5);
                set_gpu_buf(bnd, out, 6);
                set_u32(bnd, dims.seq, 7);
                set_u32(bnd, dims.q_heads, 8);
                set_u32(bnd, dims.kv_heads, 9);
                set_f32(bnd, dims.scale, 10);
            },
        )?;
    }
    Ok(())
}
