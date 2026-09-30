//! The gated delta rule for training: forward and backward at transformers'
//! op seam, `torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state,
//! use_qk_l2norm_in_kernel=True)`.
//!
//! [`crate::qwen35`]'s GDN kernels are the inference path: they take the raw
//! gate logits and run the chunked form. These take `g` (the log decay) and
//! `beta` as given, which is what a training framework hands over, and are
//! built around what the backward needs to keep. [`gdn_train_forward`] saves
//! the state every [`GDN_TRAIN_CKPT`] tokens and nothing else;
//! [`gdn_train_backward`] walks the chunks in reverse, recomputing each one's
//! states from its checkpoint into a bounded [`GdnTrainWorkspace`] before
//! running the reverse-mode recurrence over it. At Qwen3.5-2B's shapes and
//! T = 2048 that is 32 MiB of checkpoints per layer, where transformers'
//! torch fallback saves 435 MiB (`docs/qwen35.md`, Training memory).
//!
//! Per head, with the state `S` `[128, Dv]` (see `kernels/gdn_train.metal`):
//!
//! ```text
//! q^ = l2norm(q) / sqrt(128),  k^ = l2norm(k)
//! S_t = exp(g_t) S_{t-1} + k^ (beta_t (v_t - (exp(g_t) S_{t-1})^T k^))^T
//! o_t = S_t^T q^
//! ```
//!
//! All operands are dense f32: `q`, `k` `[B, T, H, 128]`, `v` `[B, T, H, Dv]`,
//! `g`, `beta` `[B, T, H]`, states `[B, H, 128, Dv]`. Grouped heads are the
//! caller's (transformers repeats them before the call). Every reduction is in
//! a fixed order, so results are deterministic.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_2d_tg, set_gpu_buf, set_tensor, set_u32};
use crate::nn::dispatch_tg_1d;
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, Tensor};

/// Key head dim the kernels are compiled for.
pub const GDN_TRAIN_DK: u32 = 128;
/// Value columns per threadgroup; `v_dim` must be a multiple.
pub const GDN_TRAIN_BV: u32 = 16;
/// Tokens between saved states.
pub const GDN_TRAIN_CKPT: u32 = 64;
const THREADS: usize = GDN_TRAIN_DK as usize;

/// Problem shape. `heads` is the value-head count (after any repeat).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnTrainDims {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub v_dim: u32,
}

impl GdnTrainDims {
    fn validate(&self, what: &str) -> Result<(), String> {
        if self.batch == 0 || self.seq == 0 || self.heads == 0 {
            return Err(format!("{what}: batch, seq and heads must be non-zero"));
        }
        if self.v_dim == 0 || self.v_dim % GDN_TRAIN_BV != 0 {
            return Err(format!("{what}: v_dim must be a non-zero multiple of {GDN_TRAIN_BV}, got {}", self.v_dim));
        }
        // Every buffer and grid below is sized from these; check the largest.
        let rows = u64::from(self.batch) * u64::from(self.seq) * u64::from(self.heads);
        let widest = rows * u64::from(GDN_TRAIN_DK.max(self.v_dim)) * u64::from(self.slices());
        if rows > u64::from(u32::MAX) || widest > i64::MAX as u64 / 4 {
            return Err(format!("{what}: shape {self:?} is too large"));
        }
        Ok(())
    }

    /// `ceil(seq / GDN_TRAIN_CKPT)`: the checkpoints the forward saves.
    pub fn checkpoints(&self) -> u32 {
        self.seq.div_ceil(GDN_TRAIN_CKPT)
    }

    fn slices(&self) -> u32 {
        self.v_dim / GDN_TRAIN_BV
    }

    fn rows(&self) -> usize {
        self.batch as usize * self.seq as usize * self.heads as usize
    }

    /// The checkpoint tensor's shape, `[B, H, NC, 128, Dv]`.
    pub fn checkpoint_shape(&self) -> [usize; 5] {
        [
            self.batch as usize,
            self.heads as usize,
            self.checkpoints() as usize,
            GDN_TRAIN_DK as usize,
            self.v_dim as usize,
        ]
    }
}

/// Scratch for [`gdn_train_backward`]: one chunk of recomputed states per
/// threadgroup, and the per-slice partial gradients. Sized for one shape and
/// reusable across layers and steps of that shape or smaller.
pub struct GdnTrainWorkspace {
    dims: GdnTrainDims,
    scratch: Tensor,
    dq_part: Tensor,
    dk_part: Tensor,
    dg_part: Tensor,
    dbeta_part: Tensor,
}

impl GdnTrainWorkspace {
    pub fn new(rt: &Arc<GpuRuntime>, dims: GdnTrainDims) -> Result<Self, String> {
        dims.validate("GdnTrainWorkspace")?;
        let (bh, ns, rows) = (
            dims.batch as usize * dims.heads as usize,
            dims.slices() as usize,
            dims.rows(),
        );
        let dk = GDN_TRAIN_DK as usize;
        Ok(Self {
            dims,
            scratch: rt.alloc_tensor_f32(&[bh * ns * GDN_TRAIN_CKPT as usize * dk * GDN_TRAIN_BV as usize])?,
            dq_part: rt.alloc_tensor_f32(&[ns * rows * dk])?,
            dk_part: rt.alloc_tensor_f32(&[ns * rows * dk])?,
            dg_part: rt.alloc_tensor_f32(&[ns * rows])?,
            dbeta_part: rt.alloc_tensor_f32(&[ns * rows])?,
        })
    }

    /// Device bytes the workspace holds for `dims`.
    pub fn bytes_for(dims: GdnTrainDims) -> usize {
        let (bh, ns, rows) = (
            dims.batch as usize * dims.heads as usize,
            (dims.v_dim / GDN_TRAIN_BV) as usize,
            dims.batch as usize * dims.seq as usize * dims.heads as usize,
        );
        let dk = GDN_TRAIN_DK as usize;
        4 * (bh * ns * GDN_TRAIN_CKPT as usize * dk * GDN_TRAIN_BV as usize + 2 * ns * rows * dk + 2 * ns * rows)
    }

    pub fn dims(&self) -> GdnTrainDims {
        self.dims
    }
}

fn want(t: &Tensor, shape: &[usize], name: &str, what: &str, rt: &Arc<GpuRuntime>) -> Result<(), String> {
    t.validate().map_err(|e| format!("{what}: {name}: {e}"))?;
    if !Arc::ptr_eq(t.runtime(), rt) {
        return Err(format!("{what}: {name} belongs to another runtime"));
    }
    if t.dtype != DType::F32 || t.shape() != shape {
        return Err(format!("{what}: {name} must be f32 {shape:?}, got {:?} {:?}", t.dtype, t.shape()));
    }
    Ok(())
}

fn disjoint(what: &str, outs: &[(&str, &Tensor)], ins: &[(&str, &Tensor)]) -> Result<(), String> {
    for (i, (a, ta)) in outs.iter().enumerate() {
        for (b, tb) in &outs[i + 1..] {
            if ta.overlaps(tb) {
                return Err(format!("{what}: outputs {a} and {b} overlap"));
            }
        }
        for (b, tb) in ins {
            if ta.overlaps(tb) {
                return Err(format!("{what}: output {a} overlaps input {b}"));
            }
        }
    }
    Ok(())
}

fn pipeline_128(
    rt: &Arc<GpuRuntime>,
    name: &str,
) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let p = rt.pipeline(name)?;
    if p.maxTotalThreadsPerThreadgroup() < THREADS {
        return Err(format!(
            "{name}: the pipeline allows {} threads per threadgroup; the kernel needs {THREADS}",
            p.maxTotalThreadsPerThreadgroup()
        ));
    }
    Ok(p)
}

/// The inputs both directions read.
#[derive(Clone, Copy)]
pub struct GdnTrainInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
    /// The initial state; zeros when `None`.
    pub s0: Option<&'a Tensor>,
}

impl GdnTrainInputs<'_> {
    fn check(&self, rt: &Arc<GpuRuntime>, d: &GdnTrainDims, what: &str) -> Result<(), String> {
        let (b, t, h, dv, dk) = (d.batch as usize, d.seq as usize, d.heads as usize, d.v_dim as usize, GDN_TRAIN_DK as usize);
        want(self.q, &[b, t, h, dk], "q", what, rt)?;
        want(self.k, &[b, t, h, dk], "k", what, rt)?;
        want(self.v, &[b, t, h, dv], "v", what, rt)?;
        want(self.g, &[b, t, h], "g", what, rt)?;
        want(self.beta, &[b, t, h], "beta", what, rt)?;
        if let Some(s0) = self.s0 {
            want(s0, &[b, h, dk, dv], "s0", what, rt)?;
        }
        Ok(())
    }

    fn named(&self) -> Vec<(&'static str, &Tensor)> {
        let mut v = vec![("q", self.q), ("k", self.k), ("v", self.v), ("g", self.g), ("beta", self.beta)];
        if let Some(s0) = self.s0 {
            v.push(("s0", s0));
        }
        v
    }
}

/// `o` `[B, T, H, Dv]`, the final state into `s_fin` when given, and the
/// checkpoints [`gdn_train_backward`] needs into `ckpt`
/// ([`GdnTrainDims::checkpoint_shape`]).
pub fn gdn_train_forward(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_forward";
    dims.validate(WHAT)?;
    x.check(rt, &dims, WHAT)?;
    let (b, t, h, dv, dk) = (dims.batch as usize, dims.seq as usize, dims.heads as usize, dims.v_dim as usize, GDN_TRAIN_DK as usize);
    want(o, &[b, t, h, dv], "o", WHAT, rt)?;
    want(ckpt, &dims.checkpoint_shape(), "ckpt", WHAT, rt)?;
    let mut outs = vec![("o", o), ("ckpt", ckpt)];
    if let Some(sf) = s_fin {
        want(sf, &[b, h, dk, dv], "s_fin", WHAT, rt)?;
        outs.push(("s_fin", sf));
    }
    disjoint(WHAT, &outs, &x.named())?;
    let flags = u32::from(x.s0.is_some()) | (u32::from(s_fin.is_some()) << 1);
    let p = pipeline_128(rt, "gdn_train_fwd")?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.q, 0);
        set_tensor(bnd, x.k, 1);
        set_tensor(bnd, x.v, 2);
        set_tensor(bnd, x.g, 3);
        set_tensor(bnd, x.beta, 4);
        // Unused slots get a live buffer the kernel never touches.
        set_tensor(bnd, x.s0.unwrap_or(x.q), 5);
        set_tensor(bnd, o, 6);
        set_tensor(bnd, s_fin.unwrap_or(ckpt), 7);
        set_tensor(bnd, ckpt, 8);
        set_u32(bnd, dims.batch, 9);
        set_u32(bnd, dims.seq, 10);
        set_u32(bnd, dims.heads, 11);
        set_u32(bnd, dims.v_dim, 12);
        set_u32(bnd, flags, 13);
    })
}

/// Gradients of every input, written in full.
pub struct GdnTrainGrads<'a> {
    pub dq: &'a Tensor,
    pub dk: &'a Tensor,
    pub dv: &'a Tensor,
    pub dg: &'a Tensor,
    pub dbeta: &'a Tensor,
    /// Required exactly when the forward had an initial state.
    pub ds0: Option<&'a Tensor>,
}

/// The backward of [`gdn_train_forward`], from `d_o` (and `d_fin`, the final
/// state's gradient, when the forward's final state was used), given the
/// same inputs and the checkpoints it saved.
#[allow(clippy::too_many_arguments)]
pub fn gdn_train_backward(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    ws: &GdnTrainWorkspace,
    grads: GdnTrainGrads<'_>,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_backward";
    dims.validate(WHAT)?;
    x.check(rt, &dims, WHAT)?;
    let (b, t, h, dv, dk) = (dims.batch as usize, dims.seq as usize, dims.heads as usize, dims.v_dim as usize, GDN_TRAIN_DK as usize);
    if ws.dims != dims {
        return Err(format!("{WHAT}: the workspace is for {:?}, not {dims:?}", ws.dims));
    }
    want(ckpt, &dims.checkpoint_shape(), "ckpt", WHAT, rt)?;
    want(d_o, &[b, t, h, dv], "d_o", WHAT, rt)?;
    if let Some(df) = d_fin {
        want(df, &[b, h, dk, dv], "d_fin", WHAT, rt)?;
    }
    want(grads.dq, &[b, t, h, dk], "dq", WHAT, rt)?;
    want(grads.dk, &[b, t, h, dk], "dk", WHAT, rt)?;
    want(grads.dv, &[b, t, h, dv], "dv", WHAT, rt)?;
    want(grads.dg, &[b, t, h], "dg", WHAT, rt)?;
    want(grads.dbeta, &[b, t, h], "dbeta", WHAT, rt)?;
    match (x.s0, grads.ds0) {
        (Some(_), Some(ds0)) => want(ds0, &[b, h, dk, dv], "ds0", WHAT, rt)?,
        (None, None) => {}
        (Some(_), None) => return Err(format!("{WHAT}: the forward had an initial state; ds0 is required")),
        (None, Some(_)) => return Err(format!("{WHAT}: ds0 given but the forward had no initial state")),
    }
    let mut outs = vec![
        ("dq", grads.dq),
        ("dk", grads.dk),
        ("dv", grads.dv),
        ("dg", grads.dg),
        ("dbeta", grads.dbeta),
    ];
    if let Some(ds0) = grads.ds0 {
        outs.push(("ds0", ds0));
    }
    let mut ins = x.named();
    ins.push(("ckpt", ckpt));
    ins.push(("d_o", d_o));
    if let Some(df) = d_fin {
        ins.push(("d_fin", df));
    }
    disjoint(WHAT, &outs, &ins)?;
    for (name, o) in &outs {
        for (wname, w) in [
            ("scratch", &ws.scratch),
            ("dq_part", &ws.dq_part),
            ("dk_part", &ws.dk_part),
            ("dg_part", &ws.dg_part),
            ("dbeta_part", &ws.dbeta_part),
        ] {
            if o.buffer.aliases(&w.buffer) {
                return Err(format!("{WHAT}: {name} aliases the workspace's {wname}"));
            }
        }
    }

    let flags = u32::from(x.s0.is_some()) | (u32::from(d_fin.is_some()) << 1);
    let p = pipeline_128(rt, "gdn_train_bwd")?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.q, 0);
        set_tensor(bnd, x.k, 1);
        set_tensor(bnd, x.v, 2);
        set_tensor(bnd, x.g, 3);
        set_tensor(bnd, x.beta, 4);
        set_tensor(bnd, d_o, 5);
        set_tensor(bnd, d_fin.unwrap_or(d_o), 6);
        set_tensor(bnd, ckpt, 7);
        set_gpu_buf(bnd, &ws.scratch.buffer, 8);
        set_tensor(bnd, grads.dv, 9);
        set_gpu_buf(bnd, &ws.dq_part.buffer, 10);
        set_gpu_buf(bnd, &ws.dk_part.buffer, 11);
        set_gpu_buf(bnd, &ws.dg_part.buffer, 12);
        set_gpu_buf(bnd, &ws.dbeta_part.buffer, 13);
        set_tensor(bnd, grads.ds0.unwrap_or(grads.dv), 14);
        set_u32(bnd, dims.batch, 15);
        set_u32(bnd, dims.seq, 16);
        set_u32(bnd, dims.heads, 17);
        set_u32(bnd, dims.v_dim, 18);
        set_u32(bnd, flags, 19);
    })?;
    let fin = pipeline_128(rt, "gdn_train_bwd_finish")?;
    let rows = dims.rows();
    dispatch_tg_1d(rt, &fin, rows, THREADS, None, |bnd| {
        set_tensor(bnd, x.q, 0);
        set_tensor(bnd, x.k, 1);
        set_gpu_buf(bnd, &ws.dq_part.buffer, 2);
        set_gpu_buf(bnd, &ws.dk_part.buffer, 3);
        set_gpu_buf(bnd, &ws.dg_part.buffer, 4);
        set_gpu_buf(bnd, &ws.dbeta_part.buffer, 5);
        set_tensor(bnd, grads.dq, 6);
        set_tensor(bnd, grads.dk, 7);
        set_tensor(bnd, grads.dg, 8);
        set_tensor(bnd, grads.dbeta, 9);
        set_u32(bnd, rows as u32, 10);
        set_u32(bnd, dims.slices(), 11);
    })
}
