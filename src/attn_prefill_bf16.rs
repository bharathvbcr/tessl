//! Qwen3.5's prefill attention with bf16 Q/K/V tiles: the host side of the
//! `qwen35_attn_tiled_bf16*_h256_*` kernels (`kernels/qwen35_attn_tiled.metal`).
//!
//! The kernels are [`crate::qwen35::attn_prefill_with_tile`]'s body
//! (`kernels/attn_tiled.h`) with `q`, `k` and `v` read as bf16, so both matrix
//! products run on the matrix units' bf16 path. Scores, the online softmax and
//! the output accumulator stay f32, and the buffer slots, the causal contract
//! (`q_abs = *q_pos_offset + t_q`, `k_abs = *kv_pos_offset + t_k`, keep
//! `k_abs <= q_abs`, live keys `min(*tkv, capacity)`, a row with no key is
//! zeros) and the grid are the f32 entry points'. The one difference at the call
//! site is the operand type:
//!
//! - `q`: bf16 `[batch, tq, heads, 256]`; `k`, `v`: bf16
//!   `[batch, capacity, kv_heads, 256]`. A caller that holds f32 tensors casts
//!   them once with the producer's store (or [`cast_f32_to_bf16`]); the kernel
//!   does not cast on load (see the kernel file for why).
//! - `o`: f32 `[batch, tq, heads, 256]`, or bf16 with `out_bf16`.
//!
//! [`crate::qwen35_model`] calls this for a bf16 prefill and a staged prefill.
//! [`crate::qwen35::attn_prefill`] stays the f32 forward, and decode attention
//! stays on [`crate::qwen35::attn_prefix_decode`] in both precisions. The cast
//! below is `dispatch_1d` on the `cast_f32_to_bf16` utility kernel (the one the
//! GEMM boundaries use, round to nearest even, pinned to the host rounding by
//! `tests/bf16_cast.rs`): the attention host-contract scan does not treat that
//! dispatch as an attention bind.

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
use crate::nn::{
    attn_kv_capacity_bf16, require, require_disjoint_writes, validate_attn_output_scalar_aliases, AttnDims,
};
use crate::qwen35::{AttnTile, PREFIX_ATTN_HEAD_DIM};
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// The probability tile of the `P.V` product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnProbTile {
    /// P rounded to bf16 into a tile of its own (both operands of `P.V` bf16).
    /// Every [`AttnTile`] geometry is compiled.
    Bf16,
    /// P kept f32 (S in place): f32 `P` times bf16 `V`. Only
    /// [`AttnTile::Q32K32Sg4`] is compiled.
    F32,
}

/// The entry point for a geometry and probability tile, or the reason there is
/// none.
pub fn attn_prefill_bf16_entry(tile: AttnTile, p: AttnProbTile) -> Result<&'static str, String> {
    match (p, tile) {
        (AttnProbTile::Bf16, AttnTile::Q32K32Sg4) => Ok("qwen35_attn_tiled_bf16_h256_q32_k32_sg4"),
        (AttnProbTile::Bf16, AttnTile::Q32K64Sg4) => Ok("qwen35_attn_tiled_bf16_h256_q32_k64_sg4"),
        (AttnProbTile::Bf16, AttnTile::Q64K32Sg8) => Ok("qwen35_attn_tiled_bf16_h256_q64_k32_sg8"),
        (AttnProbTile::Bf16, AttnTile::Q64K64Sg8) => Ok("qwen35_attn_tiled_bf16_h256_q64_k64_sg8"),
        (AttnProbTile::F32, AttnTile::Q32K32Sg4) => Ok("qwen35_attn_tiled_bf16pf32_h256_q32_k32_sg4"),
        (AttnProbTile::F32, other) => Err(format!(
            "qwen35::attn_prefill_bf16: the f32-P variant is compiled only at {}, not {}",
            AttnTile::Q32K32Sg4.label(),
            other.label()
        )),
    }
}

/// Causal prefill attention over bf16 `q`/`k`/`v` at an explicit block geometry
/// and probability tile. See the module docs for the layouts.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_bf16_with_tile(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    out_bf16: bool,
    tile: AttnTile,
    p: AttnProbTile,
) -> Result<(), String> {
    const WHAT: &str = "qwen35::attn_prefill_bf16";
    const D: u32 = PREFIX_ATTN_HEAD_DIM;
    const THREADS_PER_SIMDGROUP: usize = 32;
    if dims.window != 0 {
        return Err(format!(
            "{WHAT}: window must be 0 (Qwen3.5's full attention is global), got {}",
            dims.window
        ));
    }
    if dims.heads_kv == 0 || dims.heads % dims.heads_kv != 0 {
        return Err(format!(
            "{WHAT}: heads {} must be a non-zero multiple of heads_kv {}",
            dims.heads, dims.heads_kv
        ));
    }
    if !dims.scale.is_finite() {
        return Err(format!("{WHAT}: scale must be finite, got {}", dims.scale));
    }
    let entry = attn_prefill_bf16_entry(tile, p)?;
    let overflow = || format!("{WHAT}: a dimension product overflows");
    let grid_y = (dims.batch as usize)
        .checked_mul(dims.heads as usize)
        .ok_or_else(overflow)?;
    if grid_y > u32::MAX as usize {
        return Err(format!("{WHAT}: B*H grid extent {grid_y} exceeds Metal uint indexing"));
    }
    let q_elems = [dims.batch, dims.tq, dims.heads, D]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x as usize))
        .ok_or_else(overflow)?;
    require::<u16>(rt, q, q_elems, format_args!("{WHAT} q"))?;
    if out_bf16 {
        require::<u16>(rt, o, q_elems, format_args!("{WHAT} o (bf16)"))?;
    } else {
        require::<f32>(rt, o, q_elems, format_args!("{WHAT} o"))?;
    }
    require::<u16>(rt, k, 0, format_args!("{WHAT} k"))?;
    require::<u16>(rt, v, 0, format_args!("{WHAT} v"))?;
    require::<u32>(rt, tkv, 1, format_args!("{WHAT} tkv"))?;
    require::<u32>(rt, q_pos_offset, 1, format_args!("{WHAT} q_pos_offset"))?;
    require::<u32>(rt, kv_pos_offset, 1, format_args!("{WHAT} kv_pos_offset"))?;
    let kv_capacity = attn_kv_capacity_bf16(k, v, dims.batch, dims.heads_kv, D)?;
    let has_work = dims.batch != 0 && dims.tq != 0 && dims.heads != 0;
    if !has_work {
        return Ok(());
    }
    if kv_capacity == 0 {
        return Err(format!(
            "{WHAT}: K/V buffers do not jointly back one complete KV position"
        ));
    }
    require_disjoint_writes(WHAT, &[("o", o)], &[("q", q), ("k", k), ("v", v)])?;
    validate_attn_output_scalar_aliases(
        &dims,
        o,
        &[
            ("tkv", tkv),
            ("q_pos_offset", q_pos_offset),
            ("kv_pos_offset", kv_pos_offset),
        ],
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
    let (bq, _, sg) = tile.geometry();
    let threads = sg as usize * THREADS_PER_SIMDGROUP;
    let pipeline = rt.pipeline(entry)?;
    dispatch_2d_tg(
        rt,
        &pipeline,
        (dims.tq as usize).div_ceil(bq as usize),
        grid_y,
        threads,
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

/// Round the first `n` f32 values of `src` to bf16 into the first `n` of `dst`
/// (`cast_f32_to_bf16`, round to nearest even, as every bf16 GEMM operand of the
/// model is rounded).
pub fn cast_f32_to_bf16(rt: &Arc<GpuRuntime>, src: &GpuBuffer, dst: &GpuBuffer, n: usize) -> Result<(), String> {
    const WHAT: &str = "attn_prefill_bf16::cast_f32_to_bf16";
    let n32 = u32::try_from(n).map_err(|_| format!("{WHAT}: {n} elements exceed u32"))?;
    require::<f32>(rt, src, n, format_args!("{WHAT} src"))?;
    require::<u16>(rt, dst, n, format_args!("{WHAT} dst"))?;
    require_disjoint_writes(WHAT, &[("dst", dst)], &[("src", src)])?;
    if n == 0 {
        return Ok(());
    }
    let pipeline = rt.pipeline("cast_f32_to_bf16")?;
    dispatch_1d(rt, &pipeline, n, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        set_u32(bnd, n32, 2);
    })
}
