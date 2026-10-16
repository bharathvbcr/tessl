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
//! All operands are f32. `g`, `beta` are `[B, T, H]`, states `[B, H, 128, Dv]`,
//! `o` and `d_o` `[B, T, H, Dv]`. `q`, `k`, `v` and their gradients are
//! column windows ([`GdnWindow`]) of a matrix with one row per token, so the
//! kernels read them where a model keeps them (Qwen3.5's conv output is
//! `[q | k | v]`) and write `dq`, `dk`, `dv` back the same way; `q` and `k`
//! have `key_heads` heads that the `H` value heads share (`H / key_heads` to a
//! key head, as transformers' `repeat_interleave`), and `dq`, `dk` come back
//! summed over each key head's value heads. [`gdn_train_forward_windowed`] and
//! [`gdn_train_backward_windowed`] take that form; [`gdn_train_forward`] and
//! [`gdn_train_backward`] take dense `[B, T, H, 128]` operands (a window with
//! `ld = H * 128`, `off = 0`, `key_heads = H`) and delegate to them. Every
//! reduction is in a fixed order, so results are deterministic.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_2d_tg, set_gpu_buf, set_tensor, set_u32};
use crate::nn::dispatch_tg_1d;
use crate::runtime::{BufferKind, GpuRuntime};
use crate::tensor::{DType, Tensor};

/// Key head dim the kernels are compiled for.
pub const GDN_TRAIN_DK: u32 = 128;
/// Value columns per threadgroup; `v_dim` must be a multiple.
pub const GDN_TRAIN_BV: u32 = 16;
/// Tokens between saved states.
pub const GDN_TRAIN_CKPT: u32 = 64;
const THREADS: usize = GDN_TRAIN_DK as usize;
/// Threads of `gdn_train_c_prep`: a quad of lanes per column of its solve.
const PREP_THREADS: usize = 2 * THREADS;

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
            return Err(format!(
                "{what}: v_dim must be a non-zero multiple of {GDN_TRAIN_BV}, got {}",
                self.v_dim
            ));
        }
        // Every buffer and grid below is sized from these; check the largest.
        let rows = u64::from(self.batch) * u64::from(self.seq) * u64::from(self.heads);
        let widest = rows * u64::from(GDN_TRAIN_DK.max(self.v_dim)) * u64::from(self.slices());
        if rows > u64::from(u32::MAX) || widest > i64::MAX as u64 / 4 {
            return Err(format!("{what}: shape {self:?} is too large"));
        }
        // A dense row (`heads * 128` or `heads * v_dim` columns) is a window
        // stride in u32; refuse a width that does not fit rather than wrap.
        if u64::from(self.heads) * u64::from(GDN_TRAIN_DK.max(self.v_dim)) > u64::from(u32::MAX) {
            return Err(format!("{what}: shape {self:?} has a row width past u32"));
        }
        Ok(())
    }

    /// `ceil(seq / GDN_TRAIN_CKPT)`: the checkpoints the forward saves.
    pub fn checkpoints(&self) -> u32 {
        self.seq.div_ceil(GDN_TRAIN_CKPT)
    }

    /// `seq` rounded up to whole chunks: the row count of the chunked
    /// kernels' head-major buffers.
    fn padded_seq(&self) -> usize {
        self.checkpoints() as usize * GDN_TRAIN_CKPT as usize
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

/// Which implementation of the gated delta rule a call runs. Both read and
/// write the same layouts and the same checkpoints (the state at every
/// [`GDN_TRAIN_CKPT`] tokens), so a forward of one composes with a backward of
/// the other.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnTrainKernel {
    /// One dependent step per token (`gdn_train_fwd`, `gdn_train_bwd`): the
    /// parity oracle for the chunked form.
    Sequential,
    /// The chunked (WY) form (`gdn_train_c_*`): matrix products per 64-token
    /// chunk, only the state sequential across chunks.
    Chunked,
}

/// What [`gdn_train_forward`] and [`gdn_train_backward`] run.
/// See `bench/results/gdn_train_chunked_m5pro.txt` for the paired measurement
/// this routing rests on. The sequential kernel stays reachable (see
/// [`GdnTrainKernel`]) as the oracle the parity tests hold the chunked one to:
/// its callers are those tests and the benchmark.
#[doc(hidden)]
pub const GDN_TRAIN_PRODUCTION: GdnTrainKernel = GdnTrainKernel::Chunked;

/// Scratch for [`gdn_train_backward`], sized for exactly one
/// [`GdnTrainDims`] and reusable across layers and steps of that same shape;
/// [`gdn_train_backward`] returns an error for any other dims, smaller ones
/// included. [`GdnTrainWorkspace::new`] makes the one the production kernel
/// needs.
pub struct GdnTrainWorkspace {
    dims: GdnTrainDims,
    bufs: WsBufs,
}

enum WsBufs {
    /// One chunk of recomputed states per threadgroup, and the per-slice
    /// partial gradients.
    /// `scratch`, `dq_part`, `dk_part`, `dg_part`, `dbeta_part`.
    Sequential(Box<[Tensor; 5]>),
    Chunked(Box<ChunkedWs>),
}

/// The chunked backward's buffers (layouts in `kernels/gdn_train.metal`).
struct ChunkedWs {
    prep: ChunkPrep,
    dsbuf: Tensor,
    dbuf: Tensor,
    drbuf: Tensor,
    dop: Tensor,
    dscore: Tensor,
    das: Tensor,
    gtot: Tensor,
    dbs: Tensor,
    dgpart: Tensor,
    dbpart: Tensor,
    dqh: Tensor,
    dkh: Tensor,
}

/// What the chunked kernels derive from q, k, g and beta alone: the
/// normalized operands (once per key head), the chunk-local decay and beta,
/// and the two C x C products. The forward makes one per call; the backward
/// rebuilds it into its workspace.
struct ChunkPrep {
    kh: Tensor,
    qh: Tensor,
    gc: Tensor,
    bt: Tensor,
    wm: Tensor,
    aq: Tensor,
}

/// Elements of [`ChunkPrep`]'s buffers, sized for `heads` key heads (the most
/// there can be): `kh`, `qh`, `gc`, `bt`, `wm`, `aq`.
fn prep_lens(dims: GdnTrainDims) -> [usize; 6] {
    let bh = dims.batch as usize * dims.heads as usize;
    let tp = dims.padded_seq();
    let mats = bh * dims.checkpoints() as usize * (GDN_TRAIN_CKPT * GDN_TRAIN_CKPT) as usize;
    let rows = bh * tp * GDN_TRAIN_DK as usize;
    [rows, rows, bh * tp, bh * tp, mats, mats]
}

impl ChunkPrep {
    fn new(rt: &Arc<GpuRuntime>, dims: GdnTrainDims) -> Result<Self, String> {
        let [kh, qh, gc, bt, wm, aq] = prep_lens(dims);
        // Not zeroed: gdn_train_c_norm writes every element (zero past T),
        // and the later stages read only what the earlier ones wrote.
        let f32s = |n: usize| rt.alloc_tensor_unzeroed(&[n], DType::F32);
        Ok(Self {
            kh: f32s(kh)?,
            qh: f32s(qh)?,
            gc: f32s(gc)?,
            bt: f32s(bt)?,
            wm: f32s(wm)?,
            aq: f32s(aq)?,
        })
    }
}

impl GdnTrainWorkspace {
    /// The workspace for [`GDN_TRAIN_PRODUCTION`].
    pub fn new(rt: &Arc<GpuRuntime>, dims: GdnTrainDims) -> Result<Self, String> {
        Self::new_for(rt, dims, GDN_TRAIN_PRODUCTION)
    }

    /// The workspace `kernel`'s backward needs.
    #[doc(hidden)]
    pub fn new_for(rt: &Arc<GpuRuntime>, dims: GdnTrainDims, kernel: GdnTrainKernel) -> Result<Self, String> {
        dims.validate("GdnTrainWorkspace")?;
        // Not zeroed: each backward writes every element it later reads.
        let f32s = |n: usize| rt.alloc_tensor_unzeroed(&[n], DType::F32);
        let bufs = match kernel {
            GdnTrainKernel::Sequential => {
                let [scratch, dq_part, dk_part, dg_part, dbeta_part] = Self::seq_lens(dims);
                WsBufs::Sequential(Box::new([
                    f32s(scratch)?,
                    f32s(dq_part)?,
                    f32s(dk_part)?,
                    f32s(dg_part)?,
                    f32s(dbeta_part)?,
                ]))
            }
            GdnTrainKernel::Chunked => {
                let [dsbuf, dbuf, dop_drbuf, mats, parts, dqh] = Self::chunked_lens(dims);
                WsBufs::Chunked(Box::new(ChunkedWs {
                    prep: ChunkPrep::new(rt, dims)?,
                    dsbuf: f32s(dsbuf)?,
                    dbuf: f32s(dbuf)?,
                    drbuf: f32s(dop_drbuf)?,
                    dop: f32s(dop_drbuf)?,
                    dscore: f32s(mats)?,
                    das: f32s(mats)?,
                    gtot: f32s(mats)?,
                    dbs: f32s(mats)?,
                    dgpart: f32s(parts)?,
                    dbpart: f32s(parts)?,
                    dqh: f32s(dqh)?,
                    dkh: f32s(dqh)?,
                }))
            }
        };
        Ok(Self { dims, bufs })
    }

    /// Elements of the sequential kernel's `scratch`, `dq_part`, `dk_part`,
    /// `dg_part`, `dbeta_part`.
    fn seq_lens(dims: GdnTrainDims) -> [usize; 5] {
        let (bh, ns, rows) = (
            dims.batch as usize * dims.heads as usize,
            (dims.v_dim / GDN_TRAIN_BV) as usize,
            dims.rows(),
        );
        let dk = GDN_TRAIN_DK as usize;
        let part = ns * rows * dk;
        [
            bh * ns * GDN_TRAIN_CKPT as usize * dk * GDN_TRAIN_BV as usize,
            part,
            part,
            ns * rows,
            ns * rows,
        ]
    }

    /// Elements of the chunked backward's buffers beyond [`ChunkPrep`]'s:
    /// `dsbuf`, `dbuf`, each of `drbuf` and `dop`, each of the four C x C
    /// matrices, each of the two slice partials, each of `dqh` and `dkh`.
    fn chunked_lens(dims: GdnTrainDims) -> [usize; 6] {
        let bh = dims.batch as usize * dims.heads as usize;
        let tp = dims.padded_seq();
        let dv = dims.v_dim as usize;
        let ck = dims.checkpoints() as usize;
        let mats = bh * ck * (GDN_TRAIN_CKPT * GDN_TRAIN_CKPT) as usize;
        [
            bh * ck * GDN_TRAIN_DK as usize * dv,
            bh * tp * dv,
            bh * tp * dv,
            mats,
            dims.slices() as usize * bh * tp,
            dims.rows() * GDN_TRAIN_DK as usize,
        ]
    }

    /// Elements of every buffer the production kernel's backward holds.
    fn lens(dims: GdnTrainDims) -> Vec<usize> {
        match GDN_TRAIN_PRODUCTION {
            GdnTrainKernel::Sequential => Self::seq_lens(dims).to_vec(),
            GdnTrainKernel::Chunked => {
                let [dsbuf, dbuf, drop_, mats, parts, dqh] = Self::chunked_lens(dims);
                let mut v = prep_lens(dims).to_vec();
                v.extend([dsbuf, dbuf, drop_, drop_]);
                v.extend([mats; 4]);
                v.extend([parts; 2]);
                v.extend([dqh; 2]);
                v
            }
        }
    }

    /// Device bytes the workspace holds for `dims`.
    pub fn bytes_for(dims: GdnTrainDims) -> usize {
        4 * Self::lens(dims).iter().sum::<usize>()
    }

    /// [`Self::bytes_for`] as the device charges it: each buffer at the
    /// size the pool makes it ([`GpuRuntime::allocated_bytes_for`]).
    pub fn allocated_bytes_for(dims: GdnTrainDims) -> u64 {
        Self::lens(dims)
            .iter()
            .map(|&n| GpuRuntime::allocated_bytes_for(n * 4, BufferKind::Cold))
            .fold(0, u64::saturating_add)
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
        return Err(format!(
            "{what}: {name} must be f32 {shape:?}, got {:?} {:?}",
            t.dtype,
            t.shape()
        ));
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

fn pipeline_threads(
    rt: &Arc<GpuRuntime>,
    name: &str,
    threads: usize,
) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let p = rt.pipeline(name)?;
    if p.maxTotalThreadsPerThreadgroup() < threads {
        return Err(format!(
            "{name}: the pipeline allows {} threads per threadgroup; the kernel needs {threads}",
            p.maxTotalThreadsPerThreadgroup()
        ));
    }
    Ok(p)
}

/// A column window of an f32 matrix with one row per token (`b * T + t`):
/// element `(token, c)` is at `token * ld + off + c` of `tensor`. The matrix
/// may be wider than the window and `tensor` a view into a larger buffer.
#[derive(Clone, Copy)]
pub struct GdnWindow<'a> {
    pub tensor: &'a Tensor,
    /// Elements between consecutive tokens' rows.
    pub ld: u32,
    /// First element of the window in a row.
    pub off: u32,
}

impl<'a> GdnWindow<'a> {
    /// The whole of `tensor` read as one row of `width` per token.
    pub fn dense(tensor: &'a Tensor, width: u32) -> Self {
        Self {
            tensor,
            ld: width,
            off: 0,
        }
    }

    /// `width` columns of every one of `tokens` rows must lie inside `tensor`.
    fn check(&self, rt: &Arc<GpuRuntime>, name: &str, tokens: usize, width: usize, what: &str) -> Result<(), String> {
        let t = self.tensor;
        t.validate().map_err(|e| format!("{what}: {name}: {e}"))?;
        if !Arc::ptr_eq(t.runtime(), rt) {
            return Err(format!("{what}: {name} belongs to another runtime"));
        }
        if t.dtype != DType::F32 {
            return Err(format!("{what}: {name} must be f32, got {:?}", t.dtype));
        }
        let (ld, off) = (self.ld as usize, self.off as usize);
        if off + width > ld {
            return Err(format!(
                "{what}: {name}'s window of {width} columns at offset {off} does not fit its row stride {ld}"
            ));
        }
        let need = (tokens - 1)
            .checked_mul(ld)
            .and_then(|n| n.checked_add(off + width))
            .ok_or_else(|| format!("{what}: {name}'s window is too large"))?;
        let have = t.try_numel().map_err(|e| format!("{what}: {name}: {e}"))?;
        if have < need {
            return Err(format!(
                "{what}: {name} holds {have} elements; {tokens} rows of stride {ld} with a window of \
                 {width} at {off} need {need}"
            ));
        }
        Ok(())
    }

    /// Whether this window (`width` columns) and `other` can touch the same
    /// element: the same matrix at the same stride with disjoint columns do
    /// not; anything else that shares a byte does.
    fn overlaps(&self, width: usize, other: &GdnWindow<'_>, other_width: usize) -> bool {
        if !self.tensor.overlaps(other.tensor) {
            return false;
        }
        let same_matrix = self.tensor.byte_offset() == other.tensor.byte_offset()
            && self.ld == other.ld
            && self.tensor.try_numel().ok() == other.tensor.try_numel().ok();
        if !same_matrix {
            return true;
        }
        let (a0, b0) = (self.off as usize, other.off as usize);
        a0 < b0 + other_width && b0 < a0 + width
    }
}

/// The inputs both directions read, dense: `q`, `k` `[B, T, H, 128]`, `v`
/// `[B, T, H, Dv]`.
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

impl<'a> GdnTrainInputs<'a> {
    fn check(&self, rt: &Arc<GpuRuntime>, d: &GdnTrainDims, what: &str) -> Result<(), String> {
        let (b, t, h, dv, dk) = (
            d.batch as usize,
            d.seq as usize,
            d.heads as usize,
            d.v_dim as usize,
            GDN_TRAIN_DK as usize,
        );
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

    /// The same operands as windows: every row's `H * 128` (or `H * Dv`)
    /// columns, `key_heads = H`.
    fn windowed(&self, d: &GdnTrainDims) -> GdnTrainWindowedInputs<'a> {
        let (qk, v) = (d.heads * GDN_TRAIN_DK, d.heads * d.v_dim);
        GdnTrainWindowedInputs {
            q: GdnWindow::dense(self.q, qk),
            k: GdnWindow::dense(self.k, qk),
            v: GdnWindow::dense(self.v, v),
            key_heads: d.heads,
            g: self.g,
            beta: self.beta,
            s0: self.s0,
        }
    }
}

/// The inputs both directions read, as windows. `q` and `k` hold `key_heads`
/// heads of 128 columns, `v` holds `dims.heads` heads of `dims.v_dim`;
/// `dims.heads` must be a multiple of `key_heads`.
#[derive(Clone, Copy)]
pub struct GdnTrainWindowedInputs<'a> {
    pub q: GdnWindow<'a>,
    pub k: GdnWindow<'a>,
    pub v: GdnWindow<'a>,
    pub key_heads: u32,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
    /// The initial state; zeros when `None`.
    pub s0: Option<&'a Tensor>,
}

impl GdnTrainWindowedInputs<'_> {
    fn check(&self, rt: &Arc<GpuRuntime>, d: &GdnTrainDims, what: &str) -> Result<(), String> {
        let (b, t, h, dv, dk) = (
            d.batch as usize,
            d.seq as usize,
            d.heads as usize,
            d.v_dim as usize,
            GDN_TRAIN_DK as usize,
        );
        if self.key_heads == 0 || d.heads % self.key_heads != 0 {
            return Err(format!(
                "{what}: {} value heads are not a multiple of {} key heads",
                d.heads, self.key_heads
            ));
        }
        let (tokens, hk) = (b * t, self.key_heads as usize);
        self.q.check(rt, "q", tokens, hk * dk, what)?;
        self.k.check(rt, "k", tokens, hk * dk, what)?;
        self.v.check(rt, "v", tokens, h * dv, what)?;
        want(self.g, &[b, t, h], "g", what, rt)?;
        want(self.beta, &[b, t, h], "beta", what, rt)?;
        if let Some(s0) = self.s0 {
            want(s0, &[b, h, dk, dv], "s0", what, rt)?;
        }
        Ok(())
    }

    fn named(&self) -> Vec<(&'static str, &Tensor)> {
        let mut v = vec![
            ("q", self.q.tensor),
            ("k", self.k.tensor),
            ("v", self.v.tensor),
            ("g", self.g),
            ("beta", self.beta),
        ];
        if let Some(s0) = self.s0 {
            v.push(("s0", s0));
        }
        v
    }
}

/// `o` `[B, T, H, Dv]`, the final state into `s_fin` when given, and the
/// checkpoints [`gdn_train_backward`] needs into `ckpt`
/// ([`GdnTrainDims::checkpoint_shape`]). Dense operands; see
/// [`gdn_train_forward_windowed`].
pub fn gdn_train_forward(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
) -> Result<(), String> {
    gdn_train_forward_with(GDN_TRAIN_PRODUCTION, rt, dims, x, o, s_fin, ckpt)
}

/// [`gdn_train_forward`] on a chosen kernel.
#[doc(hidden)]
pub fn gdn_train_forward_with(
    kernel: GdnTrainKernel,
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
) -> Result<(), String> {
    dims.validate("gdn_train_forward")?;
    x.check(rt, &dims, "gdn_train_forward")?;
    gdn_train_forward_windowed_with(kernel, rt, dims, x.windowed(&dims), o, s_fin, ckpt)
}

/// [`gdn_train_forward`] with `q`, `k`, `v` read through column windows and
/// key heads shared by value heads.
pub fn gdn_train_forward_windowed(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainWindowedInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
) -> Result<(), String> {
    gdn_train_forward_windowed_with(GDN_TRAIN_PRODUCTION, rt, dims, x, o, s_fin, ckpt)
}

/// [`gdn_train_forward_windowed`] on a chosen kernel.
#[doc(hidden)]
pub fn gdn_train_forward_windowed_with(
    kernel: GdnTrainKernel,
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainWindowedInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_forward";
    dims.validate(WHAT)?;
    x.check(rt, &dims, WHAT)?;
    let (b, t, h, dv, dk) = (
        dims.batch as usize,
        dims.seq as usize,
        dims.heads as usize,
        dims.v_dim as usize,
        GDN_TRAIN_DK as usize,
    );
    want(o, &[b, t, h, dv], "o", WHAT, rt)?;
    want(ckpt, &dims.checkpoint_shape(), "ckpt", WHAT, rt)?;
    let mut outs = vec![("o", o), ("ckpt", ckpt)];
    if let Some(sf) = s_fin {
        want(sf, &[b, h, dk, dv], "s_fin", WHAT, rt)?;
        outs.push(("s_fin", sf));
    }
    disjoint(WHAT, &outs, &x.named())?;
    let flags = u32::from(x.s0.is_some()) | (u32::from(s_fin.is_some()) << 1);
    match kernel {
        GdnTrainKernel::Sequential => forward_sequential(rt, dims, &x, o, s_fin, ckpt, flags),
        GdnTrainKernel::Chunked => {
            let prep = ChunkPrep::new(rt, dims)?;
            chunk_prep(rt, dims, &x, &prep)?;
            forward_chunked(rt, dims, &x, &prep, o, s_fin, ckpt, flags)
        }
    }
}

fn forward_sequential(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
    flags: u32,
) -> Result<(), String> {
    let (b, h) = (dims.batch as usize, dims.heads as usize);
    let p = pipeline_threads(rt, "gdn_train_fwd", THREADS)?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.q.tensor, 0);
        set_tensor(bnd, x.k.tensor, 1);
        set_tensor(bnd, x.v.tensor, 2);
        set_tensor(bnd, x.g, 3);
        set_tensor(bnd, x.beta, 4);
        // Unused slots get a live buffer the kernel never touches.
        set_tensor(bnd, x.s0.unwrap_or(x.q.tensor), 5);
        set_tensor(bnd, o, 6);
        set_tensor(bnd, s_fin.unwrap_or(ckpt), 7);
        set_tensor(bnd, ckpt, 8);
        set_u32(bnd, dims.batch, 9);
        set_u32(bnd, dims.seq, 10);
        set_u32(bnd, dims.heads, 11);
        set_u32(bnd, dims.v_dim, 12);
        set_u32(bnd, flags, 13);
        set_u32(bnd, x.q.ld, 14);
        set_u32(bnd, x.q.off, 15);
        set_u32(bnd, x.k.ld, 16);
        set_u32(bnd, x.k.off, 17);
        set_u32(bnd, x.v.ld, 18);
        set_u32(bnd, x.v.off, 19);
        set_u32(bnd, x.key_heads, 20);
    })
}

/// The chunked kernels' first two stages: q and k normalized once per key
/// head (`gdn_train_c_norm`, grid = chunks x batch x *key* heads), then the
/// per-value-head decay, beta, solve and C x C product (`gdn_train_c_prep`).
fn chunk_prep(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    prep: &ChunkPrep,
) -> Result<(), String> {
    let (b, h, hk, nc) = (
        dims.batch as usize,
        dims.heads as usize,
        x.key_heads as usize,
        dims.checkpoints() as usize,
    );
    let norm = pipeline_threads(rt, "gdn_train_c_norm", THREADS)?;
    dispatch_2d_tg(rt, &norm, nc, b * hk, THREADS, |bnd| {
        set_tensor(bnd, x.q.tensor, 0);
        set_tensor(bnd, x.k.tensor, 1);
        set_tensor(bnd, &prep.kh, 2);
        set_tensor(bnd, &prep.qh, 3);
        set_u32(bnd, dims.batch, 4);
        set_u32(bnd, dims.seq, 5);
        set_u32(bnd, x.key_heads, 6);
        set_u32(bnd, x.q.ld, 7);
        set_u32(bnd, x.q.off, 8);
        set_u32(bnd, x.k.ld, 9);
        set_u32(bnd, x.k.off, 10);
    })?;
    let p = pipeline_threads(rt, "gdn_train_c_prep", PREP_THREADS)?;
    dispatch_2d_tg(rt, &p, nc, b * h, PREP_THREADS, |bnd| {
        set_tensor(bnd, &prep.kh, 0);
        set_tensor(bnd, &prep.qh, 1);
        set_tensor(bnd, x.g, 2);
        set_tensor(bnd, x.beta, 3);
        set_tensor(bnd, &prep.gc, 4);
        set_tensor(bnd, &prep.bt, 5);
        set_tensor(bnd, &prep.wm, 6);
        set_tensor(bnd, &prep.aq, 7);
        set_u32(bnd, dims.batch, 8);
        set_u32(bnd, dims.seq, 9);
        set_u32(bnd, dims.heads, 10);
        set_u32(bnd, x.key_heads, 11);
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_chunked(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    prep: &ChunkPrep,
    o: &Tensor,
    s_fin: Option<&Tensor>,
    ckpt: &Tensor,
    flags: u32,
) -> Result<(), String> {
    let (b, h) = (dims.batch as usize, dims.heads as usize);
    let p = pipeline_threads(rt, "gdn_train_c_fwd", THREADS)?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.v.tensor, 0);
        set_tensor(bnd, &prep.kh, 1);
        set_tensor(bnd, &prep.qh, 2);
        set_tensor(bnd, &prep.gc, 3);
        set_tensor(bnd, &prep.bt, 4);
        set_tensor(bnd, &prep.wm, 5);
        set_tensor(bnd, &prep.aq, 6);
        // Unused slots get a live buffer the kernel never touches.
        set_tensor(bnd, x.s0.unwrap_or(&prep.kh), 7);
        set_tensor(bnd, o, 8);
        set_tensor(bnd, s_fin.unwrap_or(ckpt), 9);
        set_tensor(bnd, ckpt, 10);
        set_u32(bnd, dims.batch, 11);
        set_u32(bnd, dims.seq, 12);
        set_u32(bnd, dims.heads, 13);
        set_u32(bnd, dims.v_dim, 14);
        set_u32(bnd, flags, 15);
        set_u32(bnd, x.v.ld, 16);
        set_u32(bnd, x.v.off, 17);
        set_u32(bnd, x.key_heads, 18);
    })
}

/// Gradients of every input, written in full. Dense: `dq`, `dk`
/// `[B, T, H, 128]`, `dv` `[B, T, H, Dv]`.
pub struct GdnTrainGrads<'a> {
    pub dq: &'a Tensor,
    pub dk: &'a Tensor,
    pub dv: &'a Tensor,
    pub dg: &'a Tensor,
    pub dbeta: &'a Tensor,
    /// Required exactly when the forward had an initial state.
    pub ds0: Option<&'a Tensor>,
}

impl<'a> GdnTrainGrads<'a> {
    fn windowed(&self, d: &GdnTrainDims) -> GdnTrainWindowedGrads<'a> {
        let (qk, v) = (d.heads * GDN_TRAIN_DK, d.heads * d.v_dim);
        GdnTrainWindowedGrads {
            dq: GdnWindow::dense(self.dq, qk),
            dk: GdnWindow::dense(self.dk, qk),
            dv: GdnWindow::dense(self.dv, v),
            dg: self.dg,
            dbeta: self.dbeta,
            ds0: self.ds0,
        }
    }
}

/// [`GdnTrainGrads`] with `dq`, `dk`, `dv` written through column windows;
/// `dq` and `dk` hold the forward's `key_heads` heads, summed over each key
/// head's value heads. The three may be disjoint windows of one matrix.
pub struct GdnTrainWindowedGrads<'a> {
    pub dq: GdnWindow<'a>,
    pub dk: GdnWindow<'a>,
    pub dv: GdnWindow<'a>,
    pub dg: &'a Tensor,
    pub dbeta: &'a Tensor,
    /// Required exactly when the forward had an initial state.
    pub ds0: Option<&'a Tensor>,
}

/// Everything [`gdn_train_backward`] checks that does not need its workspace:
/// dims, input and gradient shapes, and that no output overlaps an input. The
/// C ABI runs it before it replaces a cached workspace.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_gdn_backward(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    grads: &GdnTrainGrads<'_>,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_backward";
    dims.validate(WHAT)?;
    x.check(rt, &dims, WHAT)?;
    let (b, t, h, dv, dk) = (
        dims.batch as usize,
        dims.seq as usize,
        dims.heads as usize,
        dims.v_dim as usize,
        GDN_TRAIN_DK as usize,
    );
    want(grads.dq, &[b, t, h, dk], "dq", WHAT, rt)?;
    want(grads.dk, &[b, t, h, dk], "dk", WHAT, rt)?;
    want(grads.dv, &[b, t, h, dv], "dv", WHAT, rt)?;
    validate_windowed_backward(rt, dims, &x.windowed(&dims), ckpt, d_o, d_fin, &grads.windowed(&dims))
}

/// [`validate_gdn_backward`] for windowed operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_windowed_backward(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    grads: &GdnTrainWindowedGrads<'_>,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_backward";
    dims.validate(WHAT)?;
    x.check(rt, &dims, WHAT)?;
    let (b, t, h, dv, dk) = (
        dims.batch as usize,
        dims.seq as usize,
        dims.heads as usize,
        dims.v_dim as usize,
        GDN_TRAIN_DK as usize,
    );
    want(ckpt, &dims.checkpoint_shape(), "ckpt", WHAT, rt)?;
    want(d_o, &[b, t, h, dv], "d_o", WHAT, rt)?;
    if let Some(df) = d_fin {
        want(df, &[b, h, dk, dv], "d_fin", WHAT, rt)?;
    }
    let (tokens, hk) = (b * t, x.key_heads as usize);
    let wins = [
        ("dq", &grads.dq, hk * dk),
        ("dk", &grads.dk, hk * dk),
        ("dv", &grads.dv, h * dv),
    ];
    for (name, w, width) in &wins {
        w.check(rt, name, tokens, *width, WHAT)?;
    }
    want(grads.dg, &[b, t, h], "dg", WHAT, rt)?;
    want(grads.dbeta, &[b, t, h], "dbeta", WHAT, rt)?;
    match (x.s0, grads.ds0) {
        (Some(_), Some(ds0)) => want(ds0, &[b, h, dk, dv], "ds0", WHAT, rt)?,
        (None, None) => {}
        (Some(_), None) => return Err(format!("{WHAT}: the forward had an initial state; ds0 is required")),
        (None, Some(_)) => return Err(format!("{WHAT}: ds0 given but the forward had no initial state")),
    }
    let mut outs = vec![
        ("dq", grads.dq.tensor),
        ("dk", grads.dk.tensor),
        ("dv", grads.dv.tensor),
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
    // dq, dk and dv may share a matrix, in columns of their own: those pairs
    // are checked as windows, every other pair by tensor.
    for (i, (a, wa, na)) in wins.iter().enumerate() {
        for (b, wb, nb) in &wins[i + 1..] {
            if wa.overlaps(*na, wb, *nb) {
                return Err(format!("{WHAT}: outputs {a} and {b} overlap"));
            }
        }
    }
    let window = |n: &str| matches!(n, "dq" | "dk" | "dv");
    for (i, (a, ta)) in outs.iter().enumerate() {
        for (b, tb) in &outs[i + 1..] {
            if !(window(a) && window(b)) && ta.overlaps(tb) {
                return Err(format!("{WHAT}: outputs {a} and {b} overlap"));
            }
        }
        for (b, tb) in &ins {
            if ta.overlaps(tb) {
                return Err(format!("{WHAT}: output {a} overlaps input {b}"));
            }
        }
    }
    Ok(())
}

/// The backward of [`gdn_train_forward`], from `d_o` (and `d_fin`, the final
/// state's gradient, when the forward's final state was used), given the
/// same inputs and the checkpoints it saved. Dense operands; see
/// [`gdn_train_backward_windowed`].
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
    validate_gdn_backward(rt, dims, &x, ckpt, d_o, d_fin, &grads)?;
    gdn_train_backward_windowed(rt, dims, x.windowed(&dims), ckpt, d_o, d_fin, ws, grads.windowed(&dims))
}

/// [`gdn_train_backward`] with `q`, `k`, `v` read, and `dq`, `dk`, `dv`
/// written, through column windows, key heads shared by value heads.
#[allow(clippy::too_many_arguments)]
pub fn gdn_train_backward_windowed(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: GdnTrainWindowedInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    ws: &GdnTrainWorkspace,
    grads: GdnTrainWindowedGrads<'_>,
) -> Result<(), String> {
    const WHAT: &str = "gdn_train_backward";
    validate_windowed_backward(rt, dims, &x, ckpt, d_o, d_fin, &grads)?;
    if ws.dims != dims {
        return Err(format!("{WHAT}: the workspace is for {:?}, not {dims:?}", ws.dims));
    }
    let mut outs = vec![
        ("dq", grads.dq.tensor),
        ("dk", grads.dk.tensor),
        ("dv", grads.dv.tensor),
        ("dg", grads.dg),
        ("dbeta", grads.dbeta),
    ];
    if let Some(ds0) = grads.ds0 {
        outs.push(("ds0", ds0));
    }
    let ws_bufs: Vec<(&str, &Tensor)> = match &ws.bufs {
        WsBufs::Sequential(t) => {
            let [scratch, dq_part, dk_part, dg_part, dbeta_part] = &**t;
            vec![
                ("scratch", scratch),
                ("dq_part", dq_part),
                ("dk_part", dk_part),
                ("dg_part", dg_part),
                ("dbeta_part", dbeta_part),
            ]
        }
        WsBufs::Chunked(c) => vec![
            ("kh", &c.prep.kh),
            ("qh", &c.prep.qh),
            ("gc", &c.prep.gc),
            ("bt", &c.prep.bt),
            ("wm", &c.prep.wm),
            ("aq", &c.prep.aq),
            ("dsbuf", &c.dsbuf),
            ("dbuf", &c.dbuf),
            ("drbuf", &c.drbuf),
            ("dop", &c.dop),
            ("dscore", &c.dscore),
            ("das", &c.das),
            ("gtot", &c.gtot),
            ("dbs", &c.dbs),
            ("dgpart", &c.dgpart),
            ("dbpart", &c.dbpart),
            ("dqh", &c.dqh),
            ("dkh", &c.dkh),
        ],
    };
    for (name, o) in &outs {
        for (wname, w) in &ws_bufs {
            if o.buffer.aliases(&w.buffer) {
                return Err(format!("{WHAT}: {name} aliases the workspace's {wname}"));
            }
        }
    }
    let flags = u32::from(x.s0.is_some()) | (u32::from(d_fin.is_some()) << 1);
    match &ws.bufs {
        WsBufs::Sequential(t) => {
            let [scratch, dq_part, dk_part, dg_part, dbeta_part] = &**t;
            backward_sequential(
                rt,
                dims,
                &x,
                ckpt,
                d_o,
                d_fin,
                [scratch, dq_part, dk_part, dg_part, dbeta_part],
                &grads,
                flags,
            )
        }
        WsBufs::Chunked(c) => backward_chunked(rt, dims, &x, ckpt, d_o, d_fin, c, &grads, flags),
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_sequential(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    [scratch, dq_part, dk_part, dg_part, dbeta_part]: [&Tensor; 5],
    grads: &GdnTrainWindowedGrads<'_>,
    flags: u32,
) -> Result<(), String> {
    let (b, h) = (dims.batch as usize, dims.heads as usize);
    let p = pipeline_threads(rt, "gdn_train_bwd", THREADS)?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.q.tensor, 0);
        set_tensor(bnd, x.k.tensor, 1);
        set_tensor(bnd, x.v.tensor, 2);
        set_tensor(bnd, x.g, 3);
        set_tensor(bnd, x.beta, 4);
        set_tensor(bnd, d_o, 5);
        set_tensor(bnd, d_fin.unwrap_or(d_o), 6);
        set_tensor(bnd, ckpt, 7);
        set_gpu_buf(bnd, &scratch.buffer, 8);
        set_tensor(bnd, grads.dv.tensor, 9);
        set_gpu_buf(bnd, &dq_part.buffer, 10);
        set_gpu_buf(bnd, &dk_part.buffer, 11);
        set_gpu_buf(bnd, &dg_part.buffer, 12);
        set_gpu_buf(bnd, &dbeta_part.buffer, 13);
        set_tensor(bnd, grads.ds0.unwrap_or(grads.dv.tensor), 14);
        set_u32(bnd, dims.batch, 15);
        set_u32(bnd, dims.seq, 16);
        set_u32(bnd, dims.heads, 17);
        set_u32(bnd, dims.v_dim, 18);
        set_u32(bnd, flags, 19);
        set_u32(bnd, x.q.ld, 20);
        set_u32(bnd, x.q.off, 21);
        set_u32(bnd, x.k.ld, 22);
        set_u32(bnd, x.k.off, 23);
        set_u32(bnd, x.v.ld, 24);
        set_u32(bnd, x.v.off, 25);
        set_u32(bnd, x.key_heads, 26);
        set_u32(bnd, grads.dv.ld, 27);
        set_u32(bnd, grads.dv.off, 28);
    })?;
    let fin = pipeline_threads(rt, "gdn_train_bwd_finish", THREADS)?;
    let toks = dims.batch * dims.seq;
    dispatch_tg_1d(rt, &fin, toks as usize * x.key_heads as usize, THREADS, None, |bnd| {
        set_tensor(bnd, x.q.tensor, 0);
        set_tensor(bnd, x.k.tensor, 1);
        set_gpu_buf(bnd, &dq_part.buffer, 2);
        set_gpu_buf(bnd, &dk_part.buffer, 3);
        set_gpu_buf(bnd, &dg_part.buffer, 4);
        set_gpu_buf(bnd, &dbeta_part.buffer, 5);
        set_tensor(bnd, grads.dq.tensor, 6);
        set_tensor(bnd, grads.dk.tensor, 7);
        set_tensor(bnd, grads.dg, 8);
        set_tensor(bnd, grads.dbeta, 9);
        set_u32(bnd, dims.rows() as u32, 10);
        set_u32(bnd, dims.slices(), 11);
        set_u32(bnd, x.q.ld, 12);
        set_u32(bnd, x.q.off, 13);
        set_u32(bnd, x.k.ld, 14);
        set_u32(bnd, x.k.off, 15);
        set_u32(bnd, grads.dq.ld, 16);
        set_u32(bnd, grads.dq.off, 17);
        set_u32(bnd, grads.dk.ld, 18);
        set_u32(bnd, grads.dk.off, 19);
        set_u32(bnd, toks, 20);
        set_u32(bnd, dims.heads, 21);
        set_u32(bnd, x.key_heads, 22);
    })
}

/// The chunked backward: the prep rebuilt into the workspace, the reverse
/// state pass, the full-width C x C stage, the gate gradients, dq and dk per
/// value head, and the l2norm backward. Seven dispatches, one writer per
/// element, no atomics.
#[allow(clippy::too_many_arguments)]
fn backward_chunked(
    rt: &Arc<GpuRuntime>,
    dims: GdnTrainDims,
    x: &GdnTrainWindowedInputs<'_>,
    ckpt: &Tensor,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
    c: &ChunkedWs,
    grads: &GdnTrainWindowedGrads<'_>,
    flags: u32,
) -> Result<(), String> {
    let (b, h, nc) = (dims.batch as usize, dims.heads as usize, dims.checkpoints() as usize);
    chunk_prep(rt, dims, x, &c.prep)?;
    let p = pipeline_threads(rt, "gdn_train_c_bwd_a", THREADS)?;
    dispatch_2d_tg(rt, &p, dims.slices() as usize, b * h, THREADS, |bnd| {
        set_tensor(bnd, x.v.tensor, 0);
        set_tensor(bnd, &c.prep.kh, 1);
        set_tensor(bnd, &c.prep.qh, 2);
        set_tensor(bnd, &c.prep.gc, 3);
        set_tensor(bnd, &c.prep.bt, 4);
        set_tensor(bnd, &c.prep.wm, 5);
        set_tensor(bnd, &c.prep.aq, 6);
        set_tensor(bnd, d_o, 7);
        set_tensor(bnd, d_fin.unwrap_or(d_o), 8);
        set_tensor(bnd, ckpt, 9);
        set_tensor(bnd, &c.dsbuf, 10);
        set_tensor(bnd, &c.dbuf, 11);
        set_tensor(bnd, &c.drbuf, 12);
        set_tensor(bnd, &c.dop, 13);
        set_tensor(bnd, grads.dv.tensor, 14);
        set_tensor(bnd, &c.dgpart, 15);
        set_tensor(bnd, &c.dbpart, 16);
        set_tensor(bnd, grads.ds0.unwrap_or(grads.dv.tensor), 17);
        set_u32(bnd, dims.batch, 18);
        set_u32(bnd, dims.seq, 19);
        set_u32(bnd, dims.heads, 20);
        set_u32(bnd, dims.v_dim, 21);
        set_u32(bnd, flags, 22);
        set_u32(bnd, x.v.ld, 23);
        set_u32(bnd, x.v.off, 24);
        set_u32(bnd, x.key_heads, 25);
        set_u32(bnd, grads.dv.ld, 26);
        set_u32(bnd, grads.dv.off, 27);
    })?;
    let mats = pipeline_threads(rt, "gdn_train_c_mats", THREADS)?;
    dispatch_2d_tg(rt, &mats, nc, b * h, THREADS, |bnd| {
        set_tensor(bnd, &c.dop, 0);
        set_tensor(bnd, &c.dbuf, 1);
        set_tensor(bnd, &c.drbuf, 2);
        set_tensor(bnd, &c.prep.kh, 3);
        set_tensor(bnd, &c.prep.gc, 4);
        set_tensor(bnd, &c.prep.bt, 5);
        set_tensor(bnd, &c.prep.aq, 6);
        set_tensor(bnd, &c.dscore, 7);
        set_tensor(bnd, &c.das, 8);
        set_tensor(bnd, &c.gtot, 9);
        set_tensor(bnd, &c.dbs, 10);
        set_u32(bnd, dims.batch, 11);
        set_u32(bnd, dims.seq, 12);
        set_u32(bnd, dims.heads, 13);
        set_u32(bnd, dims.v_dim, 14);
        set_u32(bnd, x.key_heads, 15);
    })?;
    let gates = pipeline_threads(rt, "gdn_train_c_gates", THREADS)?;
    dispatch_2d_tg(rt, &gates, nc, b * h, THREADS, |bnd| {
        set_tensor(bnd, &c.gtot, 0);
        set_tensor(bnd, &c.dbs, 1);
        set_tensor(bnd, &c.dgpart, 2);
        set_tensor(bnd, &c.dbpart, 3);
        set_tensor(bnd, grads.dg, 4);
        set_tensor(bnd, grads.dbeta, 5);
        set_u32(bnd, dims.batch, 6);
        set_u32(bnd, dims.seq, 7);
        set_u32(bnd, dims.heads, 8);
        set_u32(bnd, dims.slices(), 9);
    })?;
    let dqk = pipeline_threads(rt, "gdn_train_c_dqk", THREADS)?;
    dispatch_2d_tg(rt, &dqk, nc, b * h, THREADS, |bnd| {
        set_tensor(bnd, &c.prep.kh, 0);
        set_tensor(bnd, &c.prep.qh, 1);
        set_tensor(bnd, &c.prep.gc, 2);
        set_tensor(bnd, &c.prep.bt, 3);
        set_tensor(bnd, &c.dop, 4);
        set_tensor(bnd, &c.dbuf, 5);
        set_tensor(bnd, &c.drbuf, 6);
        set_tensor(bnd, ckpt, 7);
        set_tensor(bnd, &c.dsbuf, 8);
        set_tensor(bnd, &c.dscore, 9);
        set_tensor(bnd, &c.das, 10);
        set_tensor(bnd, &c.dqh, 11);
        set_tensor(bnd, &c.dkh, 12);
        set_u32(bnd, dims.batch, 13);
        set_u32(bnd, dims.seq, 14);
        set_u32(bnd, dims.heads, 15);
        set_u32(bnd, dims.v_dim, 16);
        set_u32(bnd, x.key_heads, 17);
    })?;
    let fin = pipeline_threads(rt, "gdn_train_c_finish", THREADS)?;
    let toks = dims.batch * dims.seq;
    dispatch_tg_1d(rt, &fin, toks as usize * x.key_heads as usize, THREADS, None, |bnd| {
        set_tensor(bnd, x.q.tensor, 0);
        set_tensor(bnd, x.k.tensor, 1);
        set_tensor(bnd, &c.dqh, 2);
        set_tensor(bnd, &c.dkh, 3);
        set_tensor(bnd, grads.dq.tensor, 4);
        set_tensor(bnd, grads.dk.tensor, 5);
        set_u32(bnd, x.q.ld, 6);
        set_u32(bnd, x.q.off, 7);
        set_u32(bnd, x.k.ld, 8);
        set_u32(bnd, x.k.off, 9);
        set_u32(bnd, grads.dq.ld, 10);
        set_u32(bnd, grads.dq.off, 11);
        set_u32(bnd, grads.dk.ld, 12);
        set_u32(bnd, grads.dk.off, 13);
        set_u32(bnd, toks, 14);
        set_u32(bnd, dims.heads, 15);
        set_u32(bnd, x.key_heads, 16);
    })
}
