//! Backward of the Qwen3.5 row-local ops (`kernels/qwen35_bwd.metal`): the
//! RMSNorm, the GDN gated RMSNorm, SwiGLU, the attention output gate and
//! the GDN causal conv + SiLU, the GDN gates, the attention Q/K norm +
//! partial RoPE, and the embedding gather; and [`copy_cols`], which moves a
//! column window between fused and dense layouts.
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

use crate::dispatch::{dispatch_1d, dispatch_2d, set_f32, set_gpu_buf, set_gpu_buf_offset, set_u32};
use crate::nn::{dispatch_tg_1d, reduce_tptg, require, require_disjoint_writes};
use crate::qwen35::{require_window, AttnProjLayout, AttnShape, Cols, GdnGateLogits, GdnParams};
use crate::runtime::{BufferKind, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};

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

fn col_sum_blocks(
    rt: &Arc<GpuRuntime>,
    part: &GpuBuffer,
    part_off: usize,
    out: &GpuBuffer,
    nblocks: u64,
    dim: u32,
) -> Result<(), String> {
    let nb = u32::try_from(nblocks).map_err(|_| "weight-gradient blocks exceed u32".to_string())?;
    let p = rt.pipeline("qwen35_col_sum_blocks_f32")?;
    dispatch_1d(rt, &p, dim as usize, |bnd| {
        set_gpu_buf_offset(bnd, part, part_off * std::mem::size_of::<f32>(), 0);
        set_gpu_buf(bnd, out, 1);
        set_u32(bnd, nb, 2);
        set_u32(bnd, dim, 3);
    })
}

/// Backward of Qwen3.5's zero-centred `y = rms_norm(x) * (1 + w)`
/// ([`crate::qwen35::rms_norm`]), with `w` as stored; the kernel forms
/// `1 + w`, and `dw` is the gradient of `w` (that of `1 + w`).
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
    for (b, len, name) in [
        (x, n, "x"),
        (dy, n, "dy"),
        (dx, n, "dx"),
        (w, dim as usize, "w"),
        (dw, dim as usize, "dw"),
    ] {
        require::<f32>(rt, b, len, &format!("{WHAT} {name}"))?;
    }
    let nb = blocks(u64::from(rows), RMS_ROWS_PER_BLOCK);
    require::<f32>(rt, part, rms_norm_bwd_part_len(rows, dim), &format!("{WHAT} part"))?;
    require_disjoint_writes(
        WHAT,
        &[("dx", dx), ("dw", dw), ("part", part)],
        &[("x", x), ("w", w), ("dy", dy)],
    )?;
    let p = rt.pipeline("qwen35_rms_norm_bwd_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    if (tptg as u64) * u64::from(MAX_COLS) < u64::from(dim) {
        return Err(format!(
            "{WHAT}: dim {dim} exceeds {} x {MAX_COLS} columns per threadgroup",
            tptg
        ));
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
    col_sum_blocks(rt, part, 0, dw, nb, dim)
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
        return Err(format!(
            "{WHAT}: heads must be non-zero and dim in 1..={}",
            32 * MAX_COLS
        ));
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
    require::<f32>(
        rt,
        part,
        gated_rms_norm_bwd_part_len(rows, heads, dim),
        &format!("{WHAT} part"),
    )?;
    no_overlap(
        WHAT,
        &[("dx", dx, width), ("dz", dz, width)],
        &[("x", x, width), ("z", z, width), ("dy", dy, width)],
    )?;
    let buffers_out = [("dx", dx.buf), ("dz", dz.buf)];
    require_disjoint_writes(
        WHAT,
        &[("dw", dw), ("part", part)],
        &[
            ("x", x.buf),
            ("z", z.buf),
            ("dy", dy.buf),
            ("w", w),
            buffers_out[0],
            buffers_out[1],
        ],
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
    col_sum_blocks(rt, part, 0, dw, nb, dim)
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

/// Flattened `batch * seq` rows per weight-gradient block of the conv backward.
const CONV_ROWS_PER_BLOCK: u32 = 256;
/// Largest conv kernel width (`CONV_BWD_MAX_KW` in the kernel).
const CONV_MAX_KW: u32 = 8;

/// f32 elements of scratch [`conv1d_silu_bwd`] needs.
pub fn conv1d_silu_bwd_part_len(batch: u32, seq: u32, channels: u32, kernel_width: u32) -> usize {
    let nb = blocks(u64::from(batch) * u64::from(seq), CONV_ROWS_PER_BLOCK);
    nb as usize * channels as usize * kernel_width as usize
}

/// Backward of [`crate::qwen35::conv1d_silu`] as training runs it: from a
/// zero state (transformers' no-cache path, a conv padded with
/// `kernel_width - 1` zeros), with no carried state, no `state_out` and every
/// row `seq` long. `x`, `dy` and `dx` are windows of `channels` columns of
/// `batch * seq` rows (`dx` may be a window of the fused projection's
/// gradient); `weight` is `[channels, kernel_width]`; `dw`, the same shape,
/// is overwritten (zeros when there are no rows). `part` holds
/// [`conv1d_silu_bwd_part_len`] floats.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_bwd(
    rt: &Arc<GpuRuntime>,
    x: Cols<'_>,
    weight: &GpuBuffer,
    kernel_width: u32,
    dy: Cols<'_>,
    dx: Cols<'_>,
    dw: &GpuBuffer,
    part: &GpuBuffer,
    batch: u32,
    seq: u32,
    channels: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::conv1d_silu_bwd";
    if !(2..=CONV_MAX_KW).contains(&kernel_width) {
        return Err(format!(
            "{WHAT}: kernel_width must be 2..={CONV_MAX_KW}, got {kernel_width}"
        ));
    }
    let rows = u64::from(batch) * u64::from(seq);
    let rows32 = u32::try_from(rows).map_err(|_| format!("{WHAT}: batch x seq exceeds u32"))?;
    let (r, c) = (rows, u64::from(channels));
    for (w, name) in [(x, "x"), (dy, "dy"), (dx, "dx")] {
        require_window::<f32>(rt, w, r, c, &format!("{WHAT} {name}"))?;
    }
    let taps = channels
        .checked_mul(kernel_width)
        .ok_or_else(|| format!("{WHAT}: channels x kernel_width exceeds u32"))?;
    let wlen = taps as usize;
    require::<f32>(rt, weight, wlen, &format!("{WHAT} weight"))?;
    require::<f32>(rt, dw, wlen, &format!("{WHAT} dw"))?;
    let nb = blocks(rows, CONV_ROWS_PER_BLOCK);
    require::<f32>(
        rt,
        part,
        conv1d_silu_bwd_part_len(batch, seq, channels, kernel_width),
        &format!("{WHAT} part"),
    )?;
    no_overlap(
        WHAT,
        &[("dx", dx, channels)],
        &[("x", x, channels), ("dy", dy, channels)],
    )?;
    require_disjoint_writes(
        WHAT,
        &[("dw", dw), ("part", part)],
        &[("x", x.buf), ("weight", weight), ("dy", dy.buf), ("dx", dx.buf)],
    )?;
    require_disjoint_writes(WHAT, &[("dx", dx.buf)], &[("weight", weight)])?;
    if channels == 0 {
        return Ok(());
    }
    let bind = |bnd: &mut crate::dispatch::Binder<'_>| {
        set_u32(bnd, batch, 4);
        set_u32(bnd, seq, 5);
        set_u32(bnd, channels, 6);
        set_u32(bnd, kernel_width, 7);
        set_u32(bnd, x.ld, 8);
        set_u32(bnd, x.off, 9);
        set_u32(bnd, dy.ld, 10);
        set_u32(bnd, dy.off, 11);
    };
    let p = rt.pipeline("qwen35_conv1d_silu_bwd_dx_f32")?;
    dispatch_2d(rt, &p, channels as usize, rows32 as usize, |bnd| {
        set_gpu_buf(bnd, x.buf, 0);
        set_gpu_buf(bnd, weight, 1);
        set_gpu_buf(bnd, dy.buf, 2);
        set_gpu_buf(bnd, dx.buf, 3);
        bind(bnd);
        set_u32(bnd, dx.ld, 12);
        set_u32(bnd, dx.off, 13);
    })?;
    let p = rt.pipeline("qwen35_conv1d_silu_bwd_dw_f32")?;
    dispatch_2d(rt, &p, channels as usize, nb as usize, |bnd| {
        set_gpu_buf(bnd, x.buf, 0);
        set_gpu_buf(bnd, weight, 1);
        set_gpu_buf(bnd, dy.buf, 2);
        set_gpu_buf(bnd, part, 3);
        bind(bnd);
        set_u32(bnd, CONV_ROWS_PER_BLOCK, 12);
    })?;
    col_sum_blocks(rt, part, 0, dw, nb, taps)
}

/// Tokens per threadgroup of the Q/K norm + RoPE backward.
const QK_ROWS_PER_BLOCK: u32 = 16;
/// Its threadgroup size (`QK_BWD_SG` simdgroups).
const QK_THREADS: usize = 128;

/// f32 elements of scratch [`attn_qk_norm_rope_bwd`] needs: a q-norm and a
/// k-norm partial per block of tokens.
pub fn attn_qk_norm_rope_bwd_part_len(shape: &AttnShape) -> usize {
    let nb = blocks(u64::from(shape.batch) * u64::from(shape.seq), QK_ROWS_PER_BLOCK);
    2 * nb as usize * shape.head_dim as usize
}

/// The gradients [`attn_qk_norm_rope_bwd`] starts from: of the rotated
/// queries and keys and of the values, each dense `[batch * seq, heads,
/// head_dim]` (the layouts the forward wrote, caches of capacity `seq`).
#[derive(Clone, Copy, Debug)]
pub struct AttnQkvGrads<'a> {
    pub dq: &'a GpuBuffer,
    pub dk: &'a GpuBuffer,
    pub dv: &'a GpuBuffer,
}

/// Backward of [`crate::qwen35::attn_qk_norm_rope`] as training runs it:
/// token `t` of each batch row at position `t` (`pos_offset` 0), caches of
/// capacity `seq`. `proj` is the fused projection window the forward read
/// (laid out as [`AttnProjLayout`] says); the q, k and v gradients are
/// written to the same columns of `dproj` (same row stride and offset), and
/// the gate columns are left for [`attn_gate_bwd`]. `dq_norm_w` and
/// `dk_norm_w` (`[head_dim]`) receive the norm weights' gradients,
/// overwritten; `part` holds [`attn_qk_norm_rope_bwd_part_len`] floats.
#[allow(clippy::too_many_arguments)]
pub fn attn_qk_norm_rope_bwd(
    rt: &Arc<GpuRuntime>,
    shape: &AttnShape,
    proj: Cols<'_>,
    q_norm_w: &GpuBuffer,
    k_norm_w: &GpuBuffer,
    grads: &AttnQkvGrads<'_>,
    dproj: &GpuBuffer,
    dq_norm_w: &GpuBuffer,
    dk_norm_w: &GpuBuffer,
    part: &GpuBuffer,
    theta: f32,
    eps: f32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::attn_qk_norm_rope_bwd";
    let s = shape;
    let layout = AttnProjLayout::new(s.q_heads, s.kv_heads, s.head_dim).map_err(|e| format!("{WHAT}: {e}"))?;
    if s.head_dim > 32 * MAX_COLS {
        return Err(format!(
            "{WHAT}: head_dim must be at most {}, got {}",
            32 * MAX_COLS,
            s.head_dim
        ));
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
    let rows = u64::from(s.batch) * u64::from(s.seq);
    let span = 2 * (u64::from(s.q_heads) + u64::from(s.kv_heads)) * u64::from(s.head_dim);
    require_window::<f32>(rt, proj, rows, span, &format!("{WHAT} proj"))?;
    require_window::<f32>(rt, Cols { buf: dproj, ..proj }, rows, span, &format!("{WHAT} dproj"))?;
    let d = s.head_dim as usize;
    let per = |heads: u32| -> Result<usize, String> {
        (rows as usize)
            .checked_mul(heads as usize)
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| format!("{WHAT}: rows x heads x head_dim overflows"))
    };
    let (nq, nkv) = (per(s.q_heads)?, per(s.kv_heads)?);
    for (b, len, name) in [
        (grads.dq, nq, "dq"),
        (grads.dk, nkv, "dk"),
        (grads.dv, nkv, "dv"),
        (q_norm_w, d, "q_norm_w"),
        (k_norm_w, d, "k_norm_w"),
        (dq_norm_w, d, "dq_norm_w"),
        (dk_norm_w, d, "dk_norm_w"),
    ] {
        require::<f32>(rt, b, len, &format!("{WHAT} {name}"))?;
    }
    let nb = blocks(rows, QK_ROWS_PER_BLOCK);
    require::<f32>(rt, part, attn_qk_norm_rope_bwd_part_len(s), &format!("{WHAT} part"))?;
    require_disjoint_writes(
        WHAT,
        &[
            ("dproj", dproj),
            ("dq_norm_w", dq_norm_w),
            ("dk_norm_w", dk_norm_w),
            ("part", part),
        ],
        &[
            ("proj", proj.buf),
            ("q_norm_w", q_norm_w),
            ("k_norm_w", k_norm_w),
            ("dq", grads.dq),
            ("dk", grads.dk),
            ("dv", grads.dv),
        ],
    )?;
    let p = rt.pipeline("qwen35_attn_qk_norm_rope_bwd_f32")?;
    if p.maxTotalThreadsPerThreadgroup() < QK_THREADS {
        return Err(format!("{WHAT}: the pipeline cannot run {QK_THREADS} threads"));
    }
    dispatch_tg_1d(rt, &p, nb as usize, QK_THREADS, None, |bnd| {
        set_gpu_buf(bnd, proj.buf, 0);
        set_gpu_buf(bnd, q_norm_w, 1);
        set_gpu_buf(bnd, k_norm_w, 2);
        set_gpu_buf(bnd, grads.dq, 3);
        set_gpu_buf(bnd, grads.dk, 4);
        set_gpu_buf(bnd, grads.dv, 5);
        set_gpu_buf(bnd, dproj, 6);
        set_gpu_buf(bnd, part, 7);
        set_u32(bnd, s.batch, 8);
        set_u32(bnd, s.seq, 9);
        set_u32(bnd, s.q_heads, 10);
        set_u32(bnd, s.kv_heads, 11);
        set_u32(bnd, s.head_dim, 12);
        set_u32(bnd, s.rotary_dim, 13);
        set_u32(bnd, proj.ld, 14);
        set_u32(bnd, proj.off + layout.q_off(), 15);
        set_u32(bnd, proj.off + layout.k_off(), 16);
        set_u32(bnd, proj.off + layout.v_off(), 17);
        crate::nn::bind_rope_inv_freq(bnd, s.rotary_dim / 2, s.rotary_dim, theta, 18);
        set_f32(bnd, eps, 19);
        set_u32(bnd, QK_ROWS_PER_BLOCK, 20);
    })?;
    col_sum_blocks(rt, part, 0, dq_norm_w, nb, s.head_dim)?;
    col_sum_blocks(rt, part, nb as usize * d, dk_norm_w, nb, s.head_dim)
}

/// Scratch for [`embed_rows_bwd`]: the rows grouped by id, for at most
/// `max_rows` rows per call.
pub struct EmbedBwdWorkspace {
    max_rows: u32,
    pos: GpuBuffer,
    run_start: GpuBuffer,
    uniq: GpuBuffer,
}

impl EmbedBwdWorkspace {
    pub fn new(rt: &Arc<GpuRuntime>, max_rows: u32) -> Result<Self, String> {
        if max_rows == 0 {
            return Err("EmbedBwdWorkspace: max_rows must be non-zero".into());
        }
        let n = max_rows as usize;
        let alloc = |len: usize| rt.alloc_buffer(len * std::mem::size_of::<u32>());
        Ok(Self {
            max_rows,
            pos: alloc(n)?,
            run_start: alloc(n + 1)?,
            uniq: alloc(n)?,
        })
    }

    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }

    /// Device bytes [`Self::new`] allocates for `max_rows`, each buffer at
    /// the size the pool makes it ([`GpuRuntime::allocated_bytes_for`]).
    pub fn allocated_bytes_for(max_rows: u32) -> u64 {
        let n = max_rows as usize;
        [n, n + 1, n]
            .iter()
            .map(|&len| GpuRuntime::allocated_bytes_for(len * std::mem::size_of::<u32>(), BufferKind::Cold))
            .fold(0, u64::saturating_add)
    }
}

/// Backward of the embedding gather ([`crate::qwen35::embed_rows`]):
/// `dw[ids[r], :] += dh[r, :]` for every row, **added** to
/// what `dw` holds. With a tied LM head, `dw` is the head's weight gradient
/// ([`crate::cross_entropy::cross_entropy_rows`] overwrites it, so call this
/// after); for an untied table, zero `dw` first.
///
/// `ids` are host-known in training, so they are checked here (`< vocab`,
/// at most `ws.max_rows()`) and grouped by id on the host; each id's rows
/// are summed in ascending row order and added once, without atomics, so the
/// gradient is deterministic. `dh` is dense `[ids.len(), hidden]` f32, `dw`
/// dense `[vocab, hidden]` f32.
pub fn embed_rows_bwd(
    rt: &Arc<GpuRuntime>,
    ids: &[u32],
    dh: &GpuBuffer,
    dw: &GpuBuffer,
    vocab: u32,
    hidden: u32,
    ws: &EmbedBwdWorkspace,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::embed_rows_bwd";
    if ids.len() > ws.max_rows as usize {
        return Err(format!(
            "{WHAT}: {} rows exceed the workspace's {}",
            ids.len(),
            ws.max_rows
        ));
    }
    if let Some((r, &id)) = ids.iter().enumerate().find(|(_, &id)| id >= vocab) {
        return Err(format!("{WHAT}: ids[{r}] = {id} is not below vocab {vocab}"));
    }
    let rows = ids.len();
    let h = hidden as usize;
    let n_dh = rows
        .checked_mul(h)
        .ok_or_else(|| format!("{WHAT}: rows x hidden overflows"))?;
    let n_dw = (vocab as usize)
        .checked_mul(h)
        .ok_or_else(|| format!("{WHAT}: vocab x hidden overflows"))?;
    require::<f32>(rt, dh, n_dh, &format!("{WHAT} dh"))?;
    require::<f32>(rt, dw, n_dw, &format!("{WHAT} dw"))?;
    require_disjoint_writes(
        WHAT,
        &[("dw", dw)],
        &[
            ("dh", dh),
            ("pos", &ws.pos),
            ("run_start", &ws.run_start),
            ("uniq", &ws.uniq),
        ],
    )?;
    if rows == 0 || hidden == 0 {
        return Ok(());
    }
    // Rows sorted by (id, row): each id's rows form one run, in row order.
    let mut order: Vec<u32> = (0..rows as u32).collect();
    order.sort_by_key(|&r| (ids[r as usize], r));
    let mut starts = Vec::with_capacity(rows + 1);
    let mut uniq = Vec::with_capacity(rows);
    for (i, &r) in order.iter().enumerate() {
        let id = ids[r as usize];
        if uniq.last() != Some(&id) {
            uniq.push(id);
            starts.push(i as u32);
        }
    }
    starts.push(rows as u32);
    let n_runs = uniq.len() as u32;
    let pad = |v: &[u32], len: usize| {
        let mut out = v.to_vec();
        out.resize(len, 0);
        out
    };
    let n = ws.max_rows as usize;
    ws.pos.try_write_u32(&pad(&order, n))?;
    ws.run_start.try_write_u32(&pad(&starts, n + 1))?;
    ws.uniq.try_write_u32(&pad(&uniq, n))?;
    let p = rt.pipeline("qwen35_embed_rows_bwd_f32")?;
    dispatch_2d(rt, &p, h, n_runs as usize, |bnd| {
        set_gpu_buf(bnd, dh, 0);
        set_gpu_buf(bnd, &ws.pos, 1);
        set_gpu_buf(bnd, &ws.run_start, 2);
        set_gpu_buf(bnd, &ws.uniq, 3);
        set_gpu_buf(bnd, dw, 4);
        set_u32(bnd, n_runs, 5);
        set_u32(bnd, hidden, 6);
    })
}

/// Rows per weight-gradient block of the GDN gates' backward.
const GATES_ROWS_PER_BLOCK: u32 = 256;

/// f32 elements of scratch [`gdn_gates_bwd`] needs: a dA_log and a ddt_bias
/// partial per block of rows.
pub fn gdn_gates_bwd_part_len(rows: u32, heads: u32) -> usize {
    2 * blocks(u64::from(rows), GATES_ROWS_PER_BLOCK) as usize * heads as usize
}

/// Backward of [`crate::qwen35::gdn_gates`] from `dg`, `dbeta` (dense
/// `[rows, heads]`): `da`, `db` into the a and b columns of `dproj` (the
/// fused projection's gradient, with the logits' row stride; nothing else of
/// it is touched), and `da_log`, `ddt_bias` (`[heads]`) overwritten, summed
/// over rows in a fixed order. `part` holds [`gdn_gates_bwd_part_len`]
/// floats. The softplus derivative is torch's: 1 above 20.
#[allow(clippy::too_many_arguments)]
pub fn gdn_gates_bwd(
    rt: &Arc<GpuRuntime>,
    logits: &GdnGateLogits<'_>,
    params: &GdnParams<'_>,
    dg: &GpuBuffer,
    dbeta: &GpuBuffer,
    dproj: &GpuBuffer,
    da_log: &GpuBuffer,
    ddt_bias: &GpuBuffer,
    part: &GpuBuffer,
    rows: u32,
    heads: u32,
) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::gdn_gates_bwd";
    let r = u64::from(rows);
    for (buf, name) in [(logits.buf, "logits"), (dproj, "dproj")] {
        for (off, col) in [(logits.a_off, "a"), (logits.b_off, "b")] {
            let c = Cols {
                buf,
                ld: logits.ld,
                off,
            };
            require_window::<f32>(rt, c, r, u64::from(heads), &format!("{WHAT} {name} {col}"))?;
        }
    }
    let (a, b) = (u64::from(logits.a_off), u64::from(logits.b_off));
    if a < b + u64::from(heads) && b < a + u64::from(heads) {
        return Err(format!("{WHAT}: the a and b windows overlap"));
    }
    let n = (rows as usize)
        .checked_mul(heads as usize)
        .ok_or_else(|| format!("{WHAT}: rows x heads overflows"))?;
    for (buf, len, name) in [
        (params.a_log, heads as usize, "a_log"),
        (params.dt_bias, heads as usize, "dt_bias"),
        (dg, n, "dg"),
        (dbeta, n, "dbeta"),
        (da_log, heads as usize, "da_log"),
        (ddt_bias, heads as usize, "ddt_bias"),
    ] {
        require::<f32>(rt, buf, len, &format!("{WHAT} {name}"))?;
    }
    let nb = blocks(r, GATES_ROWS_PER_BLOCK);
    require::<f32>(rt, part, gdn_gates_bwd_part_len(rows, heads), &format!("{WHAT} part"))?;
    require_disjoint_writes(
        WHAT,
        &[
            ("dproj", dproj),
            ("da_log", da_log),
            ("ddt_bias", ddt_bias),
            ("part", part),
        ],
        &[
            ("logits", logits.buf),
            ("a_log", params.a_log),
            ("dt_bias", params.dt_bias),
            ("dg", dg),
            ("dbeta", dbeta),
        ],
    )?;
    if heads == 0 {
        return Ok(());
    }
    let p = rt.pipeline("qwen35_gdn_gates_bwd_f32")?;
    dispatch_2d(rt, &p, heads as usize, nb as usize, |bnd| {
        set_gpu_buf(bnd, logits.buf, 0);
        set_gpu_buf(bnd, params.a_log, 1);
        set_gpu_buf(bnd, params.dt_bias, 2);
        set_gpu_buf(bnd, dg, 3);
        set_gpu_buf(bnd, dbeta, 4);
        set_gpu_buf(bnd, dproj, 5);
        set_gpu_buf(bnd, part, 6);
        set_u32(bnd, rows, 7);
        set_u32(bnd, heads, 8);
        set_u32(bnd, logits.ld, 9);
        set_u32(bnd, logits.a_off, 10);
        set_u32(bnd, logits.b_off, 11);
        set_u32(bnd, GATES_ROWS_PER_BLOCK, 12);
    })?;
    col_sum_blocks(rt, part, 0, da_log, nb, heads)?;
    col_sum_blocks(rt, part, nb as usize * heads as usize, ddt_bias, nb, heads)
}

/// What [`scatter_add_rows`] checks of `src` and `pos` for a `[rows, width]`
/// destination: `src` is dense f32 `[pos.len(), width]`, and `pos` is in
/// range with no repeats. A caller that must refuse before doing anything
/// else calls it first.
pub fn check_scatter_rows(what: &str, src: &Tensor, pos: &[u32], rows: usize, width: usize) -> Result<(), String> {
    let (n, ss) = (pos.len(), src.shape());
    if ss != [n, width] || src.dtype != DType::F32 {
        return Err(format!(
            "{what}: src must be f32 [{n}, {width}], got {:?} {ss:?}",
            src.dtype
        ));
    }
    let mut seen = vec![false; rows];
    for &p in pos {
        let slot = seen
            .get_mut(p as usize)
            .ok_or_else(|| format!("{what}: position {p} >= {rows} rows"))?;
        if std::mem::replace(slot, true) {
            return Err(format!("{what}: position {p} appears twice"));
        }
    }
    Ok(())
}

/// `dst[pos[i], :] += src[i, :]`: the dense f32 `[pos.len(), width]` rows of
/// `src` added into rows `pos` of the dense f32 `[rows, width]` `dst` (the
/// gradient of chosen positions back into the whole sequence's). `pos` must
/// be in range and have no repeats (checked here), so the adds never race;
/// each element gets one f32 add. `src` and `dst` must not overlap.
pub fn scatter_add_rows(rt: &Arc<GpuRuntime>, src: &Tensor, pos: &[u32], dst: &Tensor) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::scatter_add_rows";
    let ds = dst.shape();
    if ds.len() != 2 || dst.dtype != DType::F32 {
        return Err(format!(
            "{WHAT}: dst must be f32 [rows, width], got {:?} {ds:?}",
            dst.dtype
        ));
    }
    check_scatter_rows(WHAT, src, pos, ds[0], ds[1])?;
    if src.overlaps(dst) {
        return Err(format!("{WHAT}: src and dst overlap"));
    }
    let (n, width) = (pos.len(), ds[1]);
    if n == 0 || width == 0 {
        return Ok(());
    }
    let (n32, w32) = (
        u32::try_from(n).map_err(|_| format!("{WHAT}: {n} rows exceed u32"))?,
        u32::try_from(width).map_err(|_| format!("{WHAT}: width {width} exceeds u32"))?,
    );
    let pos_buf = rt.alloc_buffer(std::mem::size_of_val(pos))?;
    pos_buf.try_write_u32(pos)?;
    let p = rt.pipeline("qwen35_scatter_add_rows_f32")?;
    dispatch_2d(rt, &p, width, n, |bnd| {
        set_gpu_buf_offset(bnd, &src.buffer, src.byte_offset(), 0);
        set_gpu_buf(bnd, &pos_buf, 1);
        set_gpu_buf_offset(bnd, &dst.buffer, dst.byte_offset(), 2);
        set_u32(bnd, n32, 3);
        set_u32(bnd, w32, 4);
    })
}

/// `dst`'s window `= src`'s window: `rows x width` f32 between two column
/// windows (the GDN's q, k, v between the conv output and the training op's
/// dense operands). The windows may share a buffer only when they are
/// disjoint.
pub fn copy_cols(rt: &Arc<GpuRuntime>, src: Cols<'_>, dst: Cols<'_>, rows: u32, width: u32) -> Result<(), String> {
    const WHAT: &str = "qwen35_bwd::copy_cols";
    require_window::<f32>(rt, src, u64::from(rows), u64::from(width), &format!("{WHAT} src"))?;
    require_window::<f32>(rt, dst, u64::from(rows), u64::from(width), &format!("{WHAT} dst"))?;
    no_overlap(WHAT, &[("dst", dst, width)], &[("src", src, width)])?;
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let p = rt.pipeline("qwen35_copy_cols_f32")?;
    dispatch_2d(rt, &p, width as usize, rows as usize, |bnd| {
        set_gpu_buf(bnd, src.buf, 0);
        set_gpu_buf(bnd, dst.buf, 1);
        set_u32(bnd, rows, 2);
        set_u32(bnd, width, 3);
        set_u32(bnd, src.ld, 4);
        set_u32(bnd, src.off, 5);
        set_u32(bnd, dst.ld, 6);
        set_u32(bnd, dst.off, 7);
    })
}
