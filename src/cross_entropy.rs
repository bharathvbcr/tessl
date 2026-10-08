//! Cross-entropy of a tied LM head over the supervised rows only, with its
//! gradients, without ever forming a `[rows, vocab]` logits matrix.
//!
//! Fine-tuning on answers supervises about one position per sequence, and the
//! full-vocabulary logits (and their gradient) for every position are what
//! ran a 2B model out of memory on MPS: at 248 320 tokens of vocabulary and
//! 8192 positions that is 8 GB before softmax. [`cross_entropy_rows`] takes
//! the hidden states, the `[vocab, hidden]` weight and the supervised
//! `(row, target)` pairs, and:
//!
//! 1. gathers the `n` supervised rows into f32 `[n, hidden]` (bf16 widened
//!    exactly);
//! 2. walks the vocabulary in chunks: `logits = h @ W[v0..v0+w]ᵀ` (the
//!    caller's [`GemmOperands`]; a bf16 weight chunk is widened exactly first), folded into a
//!    running log-sum-exp per row, the target's logit picked up on the way;
//! 3. with gradients requested, walks it again: the recomputed chunk becomes
//!    `dlogits = (softmax - onehot) * scale`, then `dh += dlogits @ W_c` (the
//!    first chunk writes `dh`, the rest accumulate into it in the GEMM) and
//!    `dW[v0..v0+w] = dlogitsᵀ @ h` (or `+=`, with
//!    [`cross_entropy_rows_accumulating`], into a gradient bank).
//!
//! Every buffer it works in comes from a [`CeWorkspace`] sized for at most
//! `max_rows` rows and `chunk` vocabulary columns, so the scratch is bounded
//! by construction: `[max_rows, chunk]` logits, `[chunk, hidden]` of widened
//! weight, and a few `[max_rows, hidden]` rows. `dW` is `[vocab, hidden]` f32
//! because a softmax gradient reaches every vocabulary row.
//!
//! The loss is `mean` (or `sum`) over the supplied rows of
//! `logsumexp(logits) - logits[target]`, which is
//! `torch.nn.functional.cross_entropy` over those rows. With gradients, `scale`
//! is the upstream gradient of the loss (1 for a plain `backward()`); `dh` and
//! `dW` are the gradients of `scale * loss`.
//!
//! The four GEMMs (two logit walks, `dh`, `dW`) run on the caller's
//! [`GemmOperands`]: exact f32, or operands rounded to bf16 with f32
//! accumulation. Everything else is exact f32 (softmax exponentials and logs
//! use `precise::`), and a runtime with relaxed-precision GEMMs switched on is
//! refused either way.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_2d, set_f32, set_gpu_buf, set_tensor, set_u32};
use crate::gemm::{cast_bf16_to_f32_into, GemmOperands};
use crate::nn::{dispatch_tg_1d, reduce_tptg};
use crate::runtime::{BufferKind, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};

/// How the per-row losses combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Mean over the supplied rows (`reduction='mean'`).
    Mean,
    /// Sum over the supplied rows (`reduction='sum'`).
    Sum,
}

/// The hidden states: `rows` is a `[T, ld]` f32 or bf16 tensor (at any byte
/// offset, so a view into a larger storage works), and the hidden columns
/// are `[off, off + hidden)` of each row. `hidden` is the weight's width.
#[derive(Clone, Copy, Debug)]
pub struct CeHidden<'a> {
    pub rows: &'a Tensor,
    pub off: u32,
}

/// Where the gradients go. `dh` is dense f32 `[n, hidden]`, row `i` the
/// gradient for the `i`-th supplied row (scatter it back by `rows[i]`);
/// `dw` is dense f32 `[vocab, hidden]`. Both are overwritten
/// ([`cross_entropy_rows_accumulating`] adds into `dw` instead).
pub struct CeGrads<'a> {
    pub dh: &'a Tensor,
    pub dw: &'a Tensor,
    /// The upstream gradient of the loss.
    pub scale: f32,
}

/// What [`cross_entropy_rows`] returns.
#[derive(Clone, Debug)]
pub struct CeOutput {
    /// The reduced loss.
    pub loss: f64,
    /// `logsumexp - target logit` for each supplied row.
    pub per_row: Vec<f64>,
}

/// Scratch for [`cross_entropy_rows`], sized once for at most `max_rows`
/// rows, `hidden` columns and `chunk` vocabulary columns per step.
pub struct CeWorkspace {
    max_rows: u32,
    hidden: u32,
    chunk: u32,
    weight_dtype: DType,
    rows: GpuBuffer,
    targets: GpuBuffer,
    /// Gathered rows `[max_rows, hidden]` f32.
    h: Tensor,
    /// One chunk of logits `[max_rows, chunk]` f32.
    logits: Tensor,
    /// A bf16 weight chunk widened to f32 `[chunk, hidden]` (bf16 weights only).
    w32: Option<Tensor>,
    m: GpuBuffer,
    s: GpuBuffer,
    tlogit: GpuBuffer,
}

impl CeWorkspace {
    pub fn new(
        rt: &Arc<GpuRuntime>,
        max_rows: u32,
        hidden: u32,
        chunk: u32,
        weight_dtype: DType,
    ) -> Result<Self, String> {
        const WHAT: &str = "CeWorkspace";
        if max_rows == 0 || hidden == 0 || chunk == 0 {
            return Err(format!("{WHAT}: max_rows, hidden and chunk must be non-zero"));
        }
        // 16-byte GEMM operand alignment for every chunk's weight view (v0 * hidden
        // elements in), in either dtype.
        if hidden % 8 != 0 {
            return Err(format!("{WHAT}: hidden must be a multiple of 8, got {hidden}"));
        }
        if !matches!(weight_dtype, DType::F32 | DType::BF16) {
            return Err(format!("{WHAT}: weight must be f32 or bf16, got {weight_dtype:?}"));
        }
        let (n, h, c) = (max_rows as usize, hidden as usize, chunk as usize);
        let f32s = |len: usize| rt.alloc_buffer(len.max(1) * 4);
        // Not zeroed: each call writes the rows it reads (the gather into
        // `h`, every logit chunk by its GEMM, `w32` by the widening) before
        // reading them.
        Ok(Self {
            max_rows,
            hidden,
            chunk,
            weight_dtype,
            rows: f32s(n)?,
            targets: f32s(n)?,
            h: rt.alloc_tensor_unzeroed(&[n, h], DType::F32)?,
            logits: rt.alloc_tensor_unzeroed(&[n, c], DType::F32)?,
            w32: match weight_dtype {
                DType::BF16 => Some(rt.alloc_tensor_unzeroed(&[c, h], DType::F32)?),
                _ => None,
            },
            m: f32s(n)?,
            s: f32s(n)?,
            tlogit: f32s(n)?,
        })
    }

    /// Device bytes this workspace holds, the bound on the call's scratch.
    pub fn bytes(&self) -> usize {
        Self::bytes_for(self.max_rows, self.hidden, self.chunk, self.weight_dtype)
    }

    /// [`Self::bytes`] before allocating.
    pub fn bytes_for(max_rows: u32, hidden: u32, chunk: u32, weight_dtype: DType) -> usize {
        4 * Self::lens(max_rows, hidden, chunk, weight_dtype).iter().sum::<usize>()
    }

    /// [`Self::bytes_for`] as the device charges it: each buffer at the
    /// size the pool makes it ([`GpuRuntime::allocated_bytes_for`]).
    pub fn allocated_bytes_for(max_rows: u32, hidden: u32, chunk: u32, weight_dtype: DType) -> u64 {
        Self::lens(max_rows, hidden, chunk, weight_dtype)
            .iter()
            .filter(|&&n| n > 0)
            .map(|&n| GpuRuntime::allocated_bytes_for(n * 4, BufferKind::Cold))
            .fold(0, u64::saturating_add)
    }

    /// f32 elements of each buffer [`Self::new`] allocates (`w32` is 0 for an
    /// f32 weight, which has none): `rows`, `targets`, `m`, `s`, `tlogit`,
    /// `h`, `logits`, `w32`.
    fn lens(max_rows: u32, hidden: u32, chunk: u32, weight_dtype: DType) -> [usize; 8] {
        let (n, h, c) = (max_rows as usize, hidden as usize, chunk as usize);
        let widened = if weight_dtype == DType::BF16 { c * h } else { 0 };
        [n, n, n, n, n, n * h, n * c, widened]
    }

    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    /// The most rows one call may supply.
    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }
}

/// See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_rows(
    rt: &Arc<GpuRuntime>,
    h: CeHidden<'_>,
    weight: &Tensor,
    rows: &[u32],
    targets: &[u32],
    reduction: Reduction,
    operands: GemmOperands,
    ws: &CeWorkspace,
    grads: Option<CeGrads<'_>>,
) -> Result<CeOutput, String> {
    ce_rows(
        "cross_entropy_rows",
        rt,
        h,
        weight,
        rows,
        targets,
        reduction,
        operands,
        ws,
        grads,
        false,
    )
}

/// [`cross_entropy_rows`] adding the weight gradient into `grads.dw`
/// (`dW += dlogitsᵀ @ h`, in the GEMM) instead of overwriting it: the head
/// gradient goes straight into an f32 gradient bank that already holds other
/// sequences' (or, overwritten first, nothing). `dh` is still overwritten.
/// Accumulating in the GEMM rounds the sum in a different order from
/// computing `dW` apart and adding it, so the bits differ from that by
/// rounding (`tests/cross_entropy.rs` bounds it).
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_rows_accumulating(
    rt: &Arc<GpuRuntime>,
    h: CeHidden<'_>,
    weight: &Tensor,
    rows: &[u32],
    targets: &[u32],
    reduction: Reduction,
    operands: GemmOperands,
    ws: &CeWorkspace,
    grads: CeGrads<'_>,
) -> Result<CeOutput, String> {
    const WHAT: &str = "cross_entropy_rows_accumulating";
    ce_rows(
        WHAT,
        rt,
        h,
        weight,
        rows,
        targets,
        reduction,
        operands,
        ws,
        Some(grads),
        true,
    )
}

/// [`cross_entropy_rows`]; `dw_add` adds into `dw` rather than overwriting it.
#[allow(clippy::too_many_arguments)]
fn ce_rows(
    what: &'static str,
    rt: &Arc<GpuRuntime>,
    h: CeHidden<'_>,
    weight: &Tensor,
    rows: &[u32],
    targets: &[u32],
    reduction: Reduction,
    operands: GemmOperands,
    ws: &CeWorkspace,
    grads: Option<CeGrads<'_>>,
    dw_add: bool,
) -> Result<CeOutput, String> {
    let n = rows.len();
    if n == 0 {
        return Err(format!("{what}: no supervised rows (an empty selection has no mean)"));
    }
    if targets.len() != n {
        return Err(format!("{what}: {n} rows but {} targets", targets.len()));
    }
    if n > ws.max_rows as usize {
        return Err(format!("{what}: {n} rows exceed the workspace's {}", ws.max_rows));
    }
    if rt.relaxed_precision() {
        return Err(format!("{what}: needs exact-f32 GEMMs; switch relaxed precision off"));
    }
    for (name, t) in [("hidden", h.rows), ("weight", weight)] {
        t.validate().map_err(|e| format!("{what}: {name}: {e}"))?;
        if !Arc::ptr_eq(t.runtime(), rt) {
            return Err(format!("{what}: {name} belongs to another runtime"));
        }
        if t.shape().len() != 2 || !matches!(t.dtype, DType::F32 | DType::BF16) {
            return Err(format!(
                "{what}: {name} must be a 2-D f32 or bf16 tensor, got {:?} {:?}",
                t.dtype,
                t.shape()
            ));
        }
    }
    let (vs, hs) = (weight.shape()[0], weight.shape()[1]);
    let (t_rows, ld) = (h.rows.shape()[0], h.rows.shape()[1]);
    if vs == 0 {
        return Err(format!("{what}: empty vocabulary"));
    }
    if hs != ws.hidden as usize || weight.dtype != ws.weight_dtype {
        return Err(format!(
            "{what}: workspace is for hidden {} / {:?} weights, not {hs} / {:?}",
            ws.hidden, ws.weight_dtype, weight.dtype
        ));
    }
    if vs > u32::MAX as usize || t_rows > u32::MAX as usize || ld > u32::MAX as usize {
        return Err(format!("{what}: vocabulary, rows and ld must fit u32"));
    }
    if (h.off as usize).checked_add(hs).is_none_or(|end| end > ld) {
        return Err(format!("{what}: hidden window [{}, +{hs}) exceeds ld {ld}", h.off));
    }
    if let Some((i, &r)) = rows.iter().enumerate().find(|(_, &r)| r as usize >= t_rows) {
        return Err(format!("{what}: rows[{i}] = {r} is past the {t_rows} hidden rows"));
    }
    if let Some((i, &t)) = targets.iter().enumerate().find(|(_, &t)| t as usize >= vs) {
        return Err(format!("{what}: targets[{i}] = {t} is past the vocabulary {vs}"));
    }
    if let Some(g) = &grads {
        if !g.scale.is_finite() {
            return Err(format!("{what}: scale must be finite"));
        }
        for (name, t, want) in [("dh", g.dh, [n, hs]), ("dw", g.dw, [vs, hs])] {
            t.validate().map_err(|e| format!("{what}: {name}: {e}"))?;
            if !Arc::ptr_eq(t.runtime(), rt) {
                return Err(format!("{what}: {name} belongs to another runtime"));
            }
            if t.dtype != DType::F32 || t.shape() != want {
                return Err(format!(
                    "{what}: {name} must be f32 {want:?}, got {:?} {:?}",
                    t.dtype,
                    t.shape()
                ));
            }
        }
        if g.dh.overlaps(g.dw) {
            return Err(format!("{what}: dh and dw overlap"));
        }
        for (out, o) in [("dh", g.dh), ("dw", g.dw)] {
            for (inp, i) in [("hidden", h.rows), ("weight", weight)] {
                if o.overlaps(i) {
                    return Err(format!("{what}: {out} overlaps the {inp} it is computed from"));
                }
            }
            for (name, b) in ws_buffers(ws) {
                if o.buffer.aliases(b) {
                    return Err(format!("{what}: {out} aliases the workspace's {name}"));
                }
            }
        }
    }
    let hidden = hs as u32;

    // In order with the queued work rather than waiting for it.
    rt.upload_u32(&ws.rows, &pad(rows, ws.max_rows))?;
    rt.upload_u32(&ws.targets, &pad(targets, ws.max_rows))?;
    let n32 = n as u32;

    // 1. Gather the supervised rows into f32.
    let gather = rt.pipeline(match h.rows.dtype {
        DType::F32 => "ce_gather_rows_f32",
        _ => "ce_gather_rows_bf16",
    })?;
    dispatch_2d(rt, &gather, hs, n, |bnd| {
        set_tensor(bnd, h.rows, 0);
        set_gpu_buf(bnd, &ws.rows, 1);
        set_gpu_buf(bnd, &ws.h.buffer, 2);
        set_u32(bnd, n32, 3);
        set_u32(bnd, hidden, 4);
        set_u32(bnd, ld as u32, 5);
        set_u32(bnd, h.off, 6);
    })?;
    let h_rows = ws.h.try_view(&[n, hs], 0)?;

    let chunk = ws.chunk as usize;
    let lse = rt.pipeline("ce_lse_update")?;
    // The f32 weight rows [v0, v0 + w) for GEMMs, widening a bf16 chunk first.
    let weight_chunk = |v0: usize, w: usize| -> Result<Tensor, String> {
        let src = weight.try_view(&[w, hs], v0 * hs)?;
        match weight.dtype {
            DType::F32 => Ok(src),
            _ => {
                let dst = ws
                    .w32
                    .as_ref()
                    .ok_or("CeWorkspace: bf16 weight without a widening buffer")?
                    .try_view(&[w, hs], 0)?;
                cast_bf16_to_f32_into(&src, &dst)?;
                Ok(dst)
            }
        }
    };

    // 2. First walk: running log-sum-exp and the target logits.
    for v0 in (0..vs).step_by(chunk) {
        let w = chunk.min(vs - v0);
        let wc = weight_chunk(v0, w)?;
        let logits = ws.logits.try_view(&[n, w], 0)?;
        operands.nt(&h_rows, &wc, &logits)?;
        let tptg = reduce_tptg(lse.maxTotalThreadsPerThreadgroup(), w);
        dispatch_tg_1d(rt, &lse, n, tptg, None, |bnd| {
            set_gpu_buf(bnd, &logits.buffer, 0);
            set_gpu_buf(bnd, &ws.m, 1);
            set_gpu_buf(bnd, &ws.s, 2);
            set_gpu_buf(bnd, &ws.tlogit, 3);
            set_gpu_buf(bnd, &ws.targets, 4);
            set_u32(bnd, n32, 5);
            set_u32(bnd, w as u32, 6);
            set_u32(bnd, w as u32, 7);
            set_u32(bnd, v0 as u32, 8);
            set_u32(bnd, u32::from(v0 == 0), 9);
        })?;
    }

    // 3. Second walk: the gradients.
    if let Some(g) = &grads {
        let scale = match reduction {
            Reduction::Mean => g.scale / n as f32,
            Reduction::Sum => g.scale,
        };
        let sgrad = rt.pipeline("ce_softmax_grad")?;
        for v0 in (0..vs).step_by(chunk) {
            let w = chunk.min(vs - v0);
            let wc = weight_chunk(v0, w)?;
            let logits = ws.logits.try_view(&[n, w], 0)?;
            operands.nt(&h_rows, &wc, &logits)?;
            dispatch_2d(rt, &sgrad, w, n, |bnd| {
                set_gpu_buf(bnd, &logits.buffer, 0);
                set_gpu_buf(bnd, &ws.m, 1);
                set_gpu_buf(bnd, &ws.s, 2);
                set_gpu_buf(bnd, &ws.targets, 3);
                set_u32(bnd, n32, 4);
                set_u32(bnd, w as u32, 5);
                set_u32(bnd, w as u32, 6);
                set_u32(bnd, v0 as u32, 7);
                set_f32(bnd, scale, 8);
            })?;
            // dh (+)= dlogits @ W_c: the first chunk writes, the rest add in
            // the GEMM.
            if v0 == 0 {
                operands.nn(&logits, &wc, g.dh)?;
            } else {
                operands.nn_acc(&logits, &wc, g.dh)?;
            }
            // dW rows [v0, v0 + w) (+)= dlogitsᵀ @ h.
            let dw_rows = g.dw.try_view(&[w, hs], v0 * hs)?;
            if dw_add {
                operands.tn_acc(&logits, &h_rows, &dw_rows)?;
            } else {
                operands.tn(&logits, &h_rows, &dw_rows)?;
            }
        }
    }

    rt.synchronize()?;
    let (m, s, t) = (ws.m.try_read_f32()?, ws.s.try_read_f32()?, ws.tlogit.try_read_f32()?);
    let per_row: Vec<f64> = (0..n)
        .map(|i| f64::from(m[i]) + f64::from(s[i]).ln() - f64::from(t[i]))
        .collect();
    if let Some(i) = per_row.iter().position(|l| !l.is_finite()) {
        return Err(format!(
            "{what}: row {i}'s loss is not finite (non-finite hidden states or weights?)"
        ));
    }
    let total: f64 = per_row.iter().sum();
    let loss = match reduction {
        Reduction::Mean => total / n as f64,
        Reduction::Sum => total,
    };
    Ok(CeOutput { loss, per_row })
}

/// `v` padded with zeros to `len` (the workspace buffers' full mapping).
fn pad(v: &[u32], len: u32) -> Vec<u32> {
    let mut out = v.to_vec();
    out.resize(len as usize, 0);
    out
}

fn ws_buffers(ws: &CeWorkspace) -> Vec<(&'static str, &GpuBuffer)> {
    let mut v = vec![
        ("rows", &ws.rows),
        ("targets", &ws.targets),
        ("h", &ws.h.buffer),
        ("logits", &ws.logits.buffer),
        ("m", &ws.m),
        ("s", &ws.s),
        ("tlogit", &ws.tlogit),
    ];
    if let Some(w) = &ws.w32 {
        v.push(("w32", &w.buffer));
    }
    v
}
