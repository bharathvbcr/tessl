//! Backward of the Qwen3.5 row-local ops (`kernels/qwen35_bwd.metal`): the
//! RMSNorm, the GDN gated RMSNorm, SwiGLU and the attention output gate.
//!
//! Each entry point takes the forward's inputs (f32, as the forward reads
//! them) and the output's gradient, in the forward's layouts, and writes the
//! inputs' gradients in the same layouts: a gradient that belongs in a fused
//! projection's gradient (`dgate`/`dup`, the output gate's column) is written
//! into that window, where the projection's GEMM backward reads it.
//!
//! Weight gradients are summed without atomics: per-block partials into a
//! caller scratch, then an in-order sum over blocks, so they are deterministic.
//! The scratch a call needs is given by the matching `*_part_len`.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_1d, dispatch_2d, set_f32, set_gpu_buf, set_u32};
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes};
use crate::qwen35::{require_window, Cols};
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// Columns one thread of a weight-gradient kernel can own
/// (`BWD_MAX_COLS` in the kernel).
const MAX_COLS: u32 = 16;
/// Rows per threadgroup of the RMSNorm backward.
const RMS_ROWS_PER_BLOCK: u32 = 32;
/// (row, head) units per threadgroup of the gated-norm backward, and its
/// fixed threadgroup size (`GATED_BWD_SG` simdgroups).
const GATED_UNITS_PER_BLOCK: u32 = 64;
const GATED_THREADS: usize = 128;

/// Whether two windows of `rows` rows can touch the same element. Windows of
/// one buffer with the same row stride are disjoint exactly when their column
/// ranges are; any other pair sharing a buffer is treated as overlapping.
pub(crate) fn windows_overlap(a: Cols<'_>, a_width: u32, b: Cols<'_>, b_width: u32) -> bool {
    if !a.buf.aliases(b.buf) {
        return false;
    }
    if a.ld != b.ld {
        return true;
    }
    let (a0, a1) = (u64::from(a.off), u64::from(a.off) + u64::from(a_width));
    let (b0, b1) = (u64::from(b.off), u64::from(b.off) + u64::from(b_width));
    a0 < b1 && b0 < a1
}

fn no_overlap(what: &str, outs: &[(&str, Cols<'_>, u32)], ins: &[(&str, Cols<'_>, u32)]) -> Result<(), String> {
    for (i, (an, a, aw)) in outs.iter().enumerate() {
        for (bn, b, bw) in outs[i + 1..].iter().chain(ins.iter()) {
            if windows_overlap(*a, *aw, *b, *bw) {
                return Err(format!("{what}: {an} overlaps {bn}"));
            }
        }
    }
    Ok(())
}

fn blocks(n: u64, per: u32) -> u64 {
    n.div_ceil(u64::from(per))
}

/// f32 elements of scratch [`rms_norm_bwd`] needs.
pub fn rms_norm_bwd_part_len(rows: u32, dim: u32) -> usize {
    blocks(u64::from(rows), RMS_ROWS_PER_BLOCK) as usize * dim as usize
}

/// f32 elements of scratch [`gated_rms_norm_bwd`] needs.
pub fn gated_rms_norm_bwd_part_len(rows: u32, heads: u32, dim: u32) -> usize {
    blocks(u64::from(rows) * u64::from(heads), GATED_UNITS_PER_BLOCK) as usize * dim as usize
}

fn col_sum_blocks(rt: &Arc<GpuRuntime>, part: &GpuBuffer, out: &GpuBuffer, nblocks: u64, dim: u32) -> Result<(), String> {
    let nb = u32::try_from(nblocks).map_err(|_| "weight-gradient blocks exceed u32".to_string())?;
    let p = rt.pipeline("qwen35_col_sum_blocks_f32")?;
    dispatch_1d(rt, &p, dim as usize, |bnd| {
        set_gpu_buf(bnd, part, 0);
        set_gpu_buf(bnd, out, 1);
        set_u32(bnd, nb, 2);
        set_u32(bnd, dim, 3);
    })
}

/// Backward of `y = rms_norm(x) * w` (the weight as the forward used it,
/// so `1 + w` for Qwen3.5's zero-centred norms, whose gradient is the same).
/// `x`, `dy`, `dx` are dense `[rows, dim]` f32; `dw` is `[dim]`, overwritten.
/// With `accumulate`, dx is added to `dx` (the residual stream's gradient)
/// instead of stored. `part` holds [`rms_norm_bwd_part_len`] floats.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_bwd(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    w: &GpuBuffer,
    dy: &GpuBuffer,
    dx: &GpuBuffer,
    dw: &GpuBuffer,
    part: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    accumulate: bool,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::rms_norm_bwd";
    if dim == 0 {
        return Err(format!("{WHAT}: dim must be non-zero"));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!("{WHAT}: eps must be finite and positive"));
    }
    let n = (rows as usize)
        .checked_mul(dim as usize)
        .ok_or_else(|| format!("{WHAT}: rows x dim overflows"))?;
    for (b, len, name) in [(x, n, "x"), (dy, n, "dy"), (dx, n, "dx"), (w, dim as usize, "w"), (dw, dim as usize, "dw")] {
        require::<f32>(rt, b, len, &format!("{WHAT} {name}"))?;
    }
    let nb = blocks(u64::from(rows), RMS_ROWS_PER_BLOCK);
    require::<f32>(rt, part, rms_norm_bwd_part_len(rows, dim), &format!("{WHAT} part"))?;
    require_disjoint_writes(WHAT, &[("dx", dx), ("dw", dw), ("part", part)], &[("x", x), ("w", w), ("dy", dy)])?;
    let p = rt.pipeline("qwen35_rms_norm_bwd_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    if (tptg as u64) * u64::from(MAX_COLS) < u64::from(dim) {
        return Err(format!("{WHAT}: dim {dim} exceeds {} x {MAX_COLS} columns per threadgroup", tptg));
    }
    dispatch_tg_1d(rt, &p, nb as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, w, 1);
        set_gpu_buf(bnd, dy, 2);
        set_gpu_buf(bnd, dx, 3);
        set_gpu_buf(bnd, part, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, dim, 6);
        set_f32(bnd, eps, 7);
        set_u32(bnd, RMS_ROWS_PER_BLOCK, 8);
        set_u32(bnd, u32::from(accumulate), 9);
    })?;
    col_sum_blocks(rt, part, dw, nb, dim)
}

/// Backward of [`crate::qwen35::gated_rms_norm`] (`y = w * rms_norm(x) *
/// silu(z)` per head of `dim`). `x`, `z`, `dy`, `dx`, `dz` are windows of
/// `heads * dim` columns; `dw` is `[dim]`, overwritten; `part` holds
/// [`gated_rms_norm_bwd_part_len`] floats.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_bwd(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    z: Cols<'_>,
    w: &GpuBuffer,
    dy: Cols<'_>,
    dx: Cols<'_>,
    dz: Cols<'_>,
    dw: &GpuBuffer,
    part: &GpuBuffer,
    rows: u32,
    heads: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::gated_rms_norm_bwd";
    if heads == 0 || dim == 0 || dim > 32 * MAX_COLS {
        return Err(format!("{WHAT}: heads must be non-zero and dim in 1..={}", 32 * MAX_COLS));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!("{WHAT}: eps must be finite and positive"));
    }
    let width = heads
        .checked_mul(dim)
        .ok_or_else(|| format!("{WHAT}: heads x dim exceeds u32"))?;
    let (r, wd) = (u64::from(rows), u64::from(width));
    for (c, name) in [(x, "x"), (z, "z"), (dy, "dy"), (dx, "dx"), (dz, "dz")] {
        require_window::<f32>(rt, c, r, wd, &format!("{WHAT} {name}"))?;
    }
    require::<f32>(rt, w, dim as usize, &format!("{WHAT} w"))?;
    require::<f32>(rt, dw, dim as usize, &format!("{WHAT} dw"))?;
    let nb = blocks(r * u64::from(heads), GATED_UNITS_PER_BLOCK);
    require::<f32>(rt, part, gated_rms_norm_bwd_part_len(rows, heads, dim), &format!("{WHAT} part"))?;
    no_overlap(
        WHAT,
        &[("dx", dx, width), ("dz", dz, width)],
        &[("x", x, width), ("z", z, width), ("dy", dy, width)],
    )?;
    let buffers_out = [("dx", dx.buf), ("dz", dz.buf)];
    require_disjoint_writes(
        WHAT,
        &[("dw", dw), ("part", part)],
        &[("x", x.buf), ("z", z.buf), ("dy", dy.buf), ("w", w), buffers_out[0], buffers_out[1]],
    )?;
    let p = rt.pipeline("qwen35_gated_rms_norm_bwd_f32")?;
    if p.maxTotalThreadsPerThreadgroup() < GATED_THREADS {
        return Err(format!("{WHAT}: the pipeline cannot run {GATED_THREADS} threads"));
    }
    dispatch_tg_1d(rt, &p, nb as usize, GATED_THREADS, None, |bnd| {
        set_gpu_buf(bnd, x.buf, 0);
        set_gpu_buf(bnd, z.buf, 1);
        set_gpu_buf(bnd, w, 2);
        set_gpu_buf(bnd, dy.buf, 3);
        set_gpu_buf(bnd, dx.buf, 4);
        set_gpu_buf(bnd, dz.buf, 5);
        set_gpu_buf(bnd, part, 6);
        set_u32(bnd, rows, 7);
        set_u32(bnd, heads, 8);
        set_u32(bnd, dim, 9);
        for (i, c) in [x, z, dy, dx, dz].iter().enumerate() {
            set_u32(bnd, c.ld, 10 + 2 * i);
            set_u32(bnd, c.off, 11 + 2 * i);
        }
        set_f32(bnd, eps, 20);
        set_u32(bnd, GATED_UNITS_PER_BLOCK, 21);
    })?;
    col_sum_blocks(rt, part, dw, nb, dim)
}

/// Backward of [`crate::qwen35::swiglu`]: `dgate = dy * up * silu'(gate)`,
/// `dup = dy * silu(gate)`, all `rows x width` f32 windows. `dgate` and `dup`
/// may be two windows of one buffer (the fused projection's gradient).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_bwd(
    rt: &Arc<GpuRuntime>,
    gate: Cols<'_>,
    up: Cols<'_>,
    dy: Cols<'_>,
    dgate: Cols<'_>,
    dup: Cols<'_>,
    rows: u32,
    width: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::swiglu_bwd";
    let (r, w) = (u64::from(rows), u64::from(width));
    for (c, name) in [(gate, "gate"), (up, "up"), (dy, "dy"), (dgate, "dgate"), (dup, "dup")] {
        require_window::<f32>(rt, c, r, w, &format!("{WHAT} {name}"))?;
    }
    no_overlap(
        WHAT,
        &[("dgate", dgate, width), ("dup", dup, width)],
        &[("gate", gate, width), ("up", up, width), ("dy", dy, width)],
    )?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let p = rt.pipeline("qwen35_swiglu_bwd_f32")?;
    dispatch_2d(rt, &p, width as usize, rows as usize, |bnd| {
        for (i, c) in [gate, up, dy, dgate, dup].iter().enumerate() {
            set_gpu_buf(bnd, c.buf, i);
        }
        set_u32(bnd, rows, 5);
        set_u32(bnd, width, 6);
        for (i, c) in [gate, up, dy, dgate, dup].iter().enumerate() {
            set_u32(bnd, c.ld, 7 + 2 * i);
            set_u32(bnd, c.off, 8 + 2 * i);
        }
    })
}

/// Backward of [`crate::qwen35::attn_output_gate`] (`out = attn *
/// sigmoid(gate)`, head h's gate at columns `q_off + h*2D + D ..` of the fused
/// projection `p`). `attn` and `d_attn` are dense `[rows, heads * dim]`; the
/// gate's gradient is written to the same columns of `dp`, the fused
/// projection's gradient (same `ld` as `p`), and nothing else of `dp` is
/// touched.
#[allow(clippy::too_many_arguments)]
pub fn attn_gate_bwd(
    rt: &Arc<GpuRuntime>,
    attn: &GpuBuffer,
    p: Cols<'_>,
    dy: Cols<'_>,
    d_attn: &GpuBuffer,
    dp: &GpuBuffer,
    rows: u32,
    heads: u32,
    dim: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::attn_gate_bwd";
    let width = heads
        .checked_mul(dim)
        .ok_or_else(|| format!("{WHAT}: heads x dim exceeds u32"))?;
    let gate_span = u64::from(width) * 2;
    let r = u64::from(rows);
    require_window::<f32>(rt, p, r, gate_span, &format!("{WHAT} p"))?;
    require_window::<f32>(rt, Cols { buf: dp, ..p }, r, gate_span, &format!("{WHAT} dp"))?;
    require_window::<f32>(rt, dy, r, u64::from(width), &format!("{WHAT} dy"))?;
    let n = (rows as usize)
        .checked_mul(width as usize)
        .ok_or_else(|| format!("{WHAT}: rows x width overflows"))?;
    require::<f32>(rt, attn, n, &format!("{WHAT} attn"))?;
    require::<f32>(rt, d_attn, n, &format!("{WHAT} d_attn"))?;
    require_disjoint_writes(
        WHAT,
        &[("d_attn", d_attn), ("dp", dp)],
        &[("attn", attn), ("p", p.buf), ("dy", dy.buf)],
    )?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let pl = rt.pipeline("qwen35_attn_gate_bwd_f32")?;
    dispatch_2d(rt, &pl, width as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, attn, 0);
        set_gpu_buf(bnd, p.buf, 1);
        set_gpu_buf(bnd, dy.buf, 2);
        set_gpu_buf(bnd, d_attn, 3);
        set_gpu_buf(bnd, dp, 4);
        set_u32(bnd, rows, 5);
        set_u32(bnd, heads, 6);
        set_u32(bnd, dim, 7);
        set_u32(bnd, p.ld, 8);
        set_u32(bnd, p.off, 9);
        set_u32(bnd, dy.ld, 10);
        set_u32(bnd, dy.off, 11);
    })
}
