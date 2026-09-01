//! Neural-network kernels promoted out of `gemma-metal`.
//!
//! These entry points shipped for months inside one model's crate, reachable
//! only as raw strings through an overlay metallib. They are model-agnostic —
//! RMSNorm, gated MLP activations, quantized GEMV, KV-cache stores — so they
//! live here now, and this module is the typed surface over them.
//!
//! # Every function validates its buffers
//!
//! The kernels guard `gid >= n` and nothing else. `rms_norm_f32` reads
//! `x[gid * dim .. gid * dim + dim]` for every `gid < rows`, so a buffer
//! holding fewer than `rows * dim` floats is an out-of-bounds *device* read:
//! no bounds check fires, no error is raised, and the result is whatever
//! happened to be resident. The wrappers below reject that on the host, before
//! encoding, because it is the only place it can still be caught.
//!
//! # The `_with_scalars` seam
//!
//! Each kernel has two entry points. The plain one binds its scalar operands
//! through the runtime's const arena and is what most callers want. The
//! `_with_scalars` one takes a closure that binds them itself, for callers
//! that need *stable* GPU addresses across encodes — const-arena offsets move
//! from one encode to the next, which breaks an Indirect Command Buffer that
//! froze its binds. `gemma-metal` drives these from a persistent scalar pool
//! for exactly that reason.
//!
//! The closure receives the binder and must fill the scalar indices named in
//! each function's docs. Buffer operands and the dispatch shape are bound here,
//! so the two paths cannot disagree about them.

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32, Binder};
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

/// Elements a buffer can hold at `size_of::<T>()` bytes each.
fn capacity_of<T>(buf: &GpuBuffer) -> usize {
    buf.nbytes() / std::mem::size_of::<T>()
}

/// `rows * dim`, or an error naming the overflow rather than wrapping.
fn elems(rows: u32, dim: u32, what: &str) -> Result<usize, String> {
    (rows as usize)
        .checked_mul(dim as usize)
        .ok_or_else(|| format!("{what}: rows {rows} x dim {dim} overflows usize"))
}

/// Reject a buffer that cannot hold `need` elements of `T`.
fn require<T>(buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    let have = capacity_of::<T>(buf);
    if have < need {
        return Err(format!(
            "{what}: buffer holds {have} elements, kernel reads/writes {need}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- RMSNorm ---

/// `out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * weight[:]`, f32 out.
///
/// Scalar indices for `_with_scalars`: 3 = `rows` (u32), 4 = `dim` (u32),
/// 5 = `eps` (f32).
pub fn rms_norm_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    rms_norm_f32_with_scalars(rt, x, weight, out, rows, dim, |bnd| {
        set_u32(bnd, rows, 3);
        set_u32(bnd, dim, 4);
        set_f32(bnd, eps, 5);
    })
}

/// [`rms_norm_f32`] with caller-supplied scalar binds. See the module docs.
pub fn rms_norm_f32_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_f32")?;
    require::<f32>(x, n, "rms_norm_f32 x")?;
    require::<f32>(weight, dim as usize, "rms_norm_f32 weight")?;
    require::<f32>(out, n, "rms_norm_f32 out")?;
    if rows == 0 {
        return Ok(());
    }
    let p = rt.pipeline("rms_norm_f32")?;
    // One threadgroup per row with a tree reduction, matching `row_reduce`.
    // Was `dispatch_1d(rt, &p, rows)` — one thread per row — which capped
    // parallelism at `rows` and ran the whole kernel on a single GPU thread at
    // the decode shape.
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, weight, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// [`rms_norm_f32`] writing bf16, to feed a bf16 GEMV without a cast pass.
///
/// Scalar indices for `_with_scalars`: 3 = `rows`, 4 = `dim`, 5 = `eps`.
pub fn rms_norm_bf16(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<(), String> {
    rms_norm_bf16_with_scalars(rt, x, weight, out, rows, dim, |bnd| {
        set_u32(bnd, rows, 3);
        set_u32(bnd, dim, 4);
        set_f32(bnd, eps, 5);
    })
}

/// [`rms_norm_bf16`] with caller-supplied scalar binds.
pub fn rms_norm_bf16_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_bf16")?;
    require::<f32>(x, n, "rms_norm_bf16 x")?;
    require::<f32>(weight, dim as usize, "rms_norm_bf16 weight")?;
    // bf16 output: two bytes per element, not four.
    require::<u16>(out, n, "rms_norm_bf16 out")?;
    if rows == 0 {
        return Ok(());
    }
    let p = rt.pipeline("rms_norm_bf16")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, weight, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// Fused `resid = layer_scale * (resid + rms_norm(x) * weight)`, in place.
///
/// Collapses a norm and a residual add into one dispatch. `layer_scale == 1.0`
/// is the plain residual add.
///
/// Scalar indices for `_with_scalars`: 3 = `rows`, 4 = `dim`, 5 = `eps`,
/// 6 = `layer_scale` (f32).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_add_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    resid: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<(), String> {
    rms_norm_residual_add_f32_with_scalars(rt, x, weight, resid, rows, dim, |bnd| {
        set_u32(bnd, rows, 3);
        set_u32(bnd, dim, 4);
        set_f32(bnd, eps, 5);
        set_f32(bnd, layer_scale, 6);
    })
}

/// [`rms_norm_residual_add_f32`] with caller-supplied scalar binds.
pub fn rms_norm_residual_add_f32_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    resid: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_residual_add_f32")?;
    require::<f32>(x, n, "rms_norm_residual_add_f32 x")?;
    require::<f32>(weight, dim as usize, "rms_norm_residual_add_f32 weight")?;
    require::<f32>(resid, n, "rms_norm_residual_add_f32 resid")?;
    if rows == 0 {
        return Ok(());
    }
    let p = rt.pipeline("rms_norm_residual_add_f32")?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), dim as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, weight, 1);
        set_gpu_buf(bnd, resid, 2);
        scalars(bnd);
    })
}

// ------------------------------------------------------- Gated MLP acts ---

/// `out[i] = silu(gate[i]) * up[i]`, where `silu(x) = x * sigmoid(x)`.
///
/// Scalar index for `_with_scalars`: 3 = `n` (u32).
pub fn mlp_silu(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    mlp_silu_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3))
}

/// [`mlp_silu`] with caller-supplied scalar binds.
pub fn mlp_silu_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(gate, n_us, "mlp_silu gate")?;
    require::<f32>(up, n_us, "mlp_silu up")?;
    require::<f32>(out, n_us, "mlp_silu out")?;
    let p = rt.pipeline("mlp_silu")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, gate, 0);
        set_gpu_buf(bnd, up, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]`.
///
/// The kernel clamps before cubing and uses `precise::tanh`: at `-O2` MSL
/// lowers plain `tanh` to `air.fast_tanh`, which returns NaN for arguments
/// beyond roughly 10, and the GELU inner term reaches ~301 at `|x| = 20`.
///
/// Scalar index for `_with_scalars`: 3 = `n`.
pub fn mlp_gelu_tanh(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    mlp_gelu_tanh_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3))
}

/// [`mlp_gelu_tanh`] with caller-supplied scalar binds.
pub fn mlp_gelu_tanh_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(gate, n_us, "mlp_gelu_tanh gate")?;
    require::<f32>(up, n_us, "mlp_gelu_tanh up")?;
    require::<f32>(out, n_us, "mlp_gelu_tanh out")?;
    let p = rt.pipeline("mlp_gelu_tanh")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, gate, 0);
        set_gpu_buf(bnd, up, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// [`mlp_gelu_tanh`] writing bf16, to feed a bf16 down-projection GEMV.
///
/// Scalar index for `_with_scalars`: 3 = `n`.
pub fn mlp_gelu_tanh_bf16(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    mlp_gelu_tanh_bf16_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3))
}

/// [`mlp_gelu_tanh_bf16`] with caller-supplied scalar binds.
pub fn mlp_gelu_tanh_bf16_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(gate, n_us, "mlp_gelu_tanh_bf16 gate")?;
    require::<f32>(up, n_us, "mlp_gelu_tanh_bf16 up")?;
    require::<u16>(out, n_us, "mlp_gelu_tanh_bf16 out")?;
    let p = rt.pipeline("mlp_gelu_tanh_bf16")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, gate, 0);
        set_gpu_buf(bnd, up, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

// ------------------------------------------------------------- Elementwise ---

/// `x[i] *= scale`, in place.
///
/// Scalar indices for `_with_scalars`: 1 = `scale` (f32), 2 = `n` (u32).
pub fn scale_f32_inplace(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    scale: f32,
    n: u32,
) -> Result<(), String> {
    scale_f32_inplace_with_scalars(rt, x, n, |bnd| {
        set_f32(bnd, scale, 1);
        set_u32(bnd, n, 2);
    })
}

/// [`scale_f32_inplace`] with caller-supplied scalar binds.
pub fn scale_f32_inplace_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    require::<f32>(x, n as usize, "scale_f32_inplace x")?;
    let p = rt.pipeline("scale_f32_inplace")?;
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, x, 0);
        scalars(bnd);
    })
}

// ------------------------------------------------------------------ GEMV ---

/// `y[rows] = W[rows, cols] @ x[cols]` with group-wise affine Q8 weights.
///
/// `packed` is row-major `int8` of `rows * cols`; `scales` and `zeros` hold
/// `rows * (cols / group_size)` entries each, grouped along `cols`. The
/// dequantization is `w = scale * (packed - zero)`.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`,
/// 7 = `group_size`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q8(
    rt: &Arc<GpuRuntime>,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<(), String> {
    gemv_q8_with_scalars(
        rt,
        packed,
        scales,
        zeros,
        x,
        y,
        rows,
        cols,
        group_size,
        |bnd| {
            set_u32(bnd, rows, 5);
            set_u32(bnd, cols, 6);
            set_u32(bnd, group_size, 7);
        },
    )
}

/// [`gemv_q8`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q8_with_scalars(
    rt: &Arc<GpuRuntime>,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    // The kernel computes `cols / group_size` with integer division and then
    // strides by `group_size`, so a ragged final group is silently dropped
    // rather than half-read. Refusing here makes that a caller error instead of
    // a quiet wrong answer.
    if group_size == 0 {
        return Err("gemv_q8: group_size must be non-zero".into());
    }
    if cols % group_size != 0 {
        return Err(format!(
            "gemv_q8: cols {cols} is not a multiple of group_size {group_size}; \
             the kernel would silently drop the ragged tail group"
        ));
    }
    let weights = elems(rows, cols, "gemv_q8")?;
    let groups = elems(rows, cols / group_size, "gemv_q8 groups")?;
    require::<i8>(packed, weights, "gemv_q8 packed")?;
    require::<f32>(scales, groups, "gemv_q8 scales")?;
    require::<f32>(zeros, groups, "gemv_q8 zeros")?;
    require::<f32>(x, cols as usize, "gemv_q8 x")?;
    require::<f32>(y, rows as usize, "gemv_q8 y")?;
    if rows == 0 {
        return Ok(());
    }
    let p = rt.pipeline("gemv_q8")?;
    // One simdgroup per `SIMD_ROWS` output rows with lanes striding K, the same
    // geometry the MLX Q4 simd GEMVs use. Was `dispatch_1d(rt, &p, rows)` — one
    // thread per row — which left adjacent threads reading `cols` bytes apart,
    // so nothing in a simdgroup's loads coalesced.
    dispatch_tg_1d(
        rt,
        &p,
        simd_gemv_threadgroups(rows),
        SIMD_TPTG,
        None,
        |bnd| {
            set_gpu_buf(bnd, packed, 0);
            set_gpu_buf(bnd, scales, 1);
            set_gpu_buf(bnd, zeros, 2);
            set_gpu_buf(bnd, x, 3);
            set_gpu_buf(bnd, y, 4);
            scalars(bnd);
        },
    )
}

// -------------------------------------------------------------- KV cache ---

/// `dst[*dst_offset + i] = src[i]` for `i < n`.
///
/// `dst_offset` is read from a device `u32` rather than passed as a constant:
/// the offset changes every timestep, and an Indirect Command Buffer that
/// froze its binds needs a stable address whose *contents* move, not a new
/// const-arena slot each encode.
///
/// Scalar index for `_with_scalars`: 2 = `n`. Buffer 3 (`dst_offset`) is bound
/// here in both paths — it is a device buffer, not a scalar.
pub fn kv_store_timestep(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    kv_store_timestep_with_scalars(rt, src, dst, dst_offset, n, |bnd| set_u32(bnd, n, 2))
}

/// [`kv_store_timestep`] with caller-supplied scalar binds.
pub fn kv_store_timestep_with_scalars(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    require::<f32>(src, n as usize, "kv_store_timestep src")?;
    require::<u32>(dst_offset, 1, "kv_store_timestep dst_offset")?;
    // `dst` is indexed at `*dst_offset + gid`, and the offset lives on the
    // device — its value is not knowable here. Only the floor is checkable.
    require::<f32>(dst, n as usize, "kv_store_timestep dst")?;
    let p = rt.pipeline("kv_store_timestep")?;
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        scalars(bnd);
        set_gpu_buf(bnd, dst_offset, 3);
    })
}

/// [`kv_store_timestep`] for K and V in one dispatch.
///
/// Scalar index for `_with_scalars`: 4 = `n`. Buffer 5 is `dst_offset`.
pub fn kv_store_timestep_pair(
    rt: &Arc<GpuRuntime>,
    src_k: &GpuBuffer,
    src_v: &GpuBuffer,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    kv_store_timestep_pair_with_scalars(rt, src_k, src_v, dst_k, dst_v, dst_offset, n, |bnd| {
        set_u32(bnd, n, 4)
    })
}

/// [`kv_store_timestep_pair`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn kv_store_timestep_pair_with_scalars(
    rt: &Arc<GpuRuntime>,
    src_k: &GpuBuffer,
    src_v: &GpuBuffer,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(src_k, n_us, "kv_store_timestep_pair src_k")?;
    require::<f32>(src_v, n_us, "kv_store_timestep_pair src_v")?;
    require::<f32>(dst_k, n_us, "kv_store_timestep_pair dst_k")?;
    require::<f32>(dst_v, n_us, "kv_store_timestep_pair dst_v")?;
    require::<u32>(dst_offset, 1, "kv_store_timestep_pair dst_offset")?;
    let p = rt.pipeline("kv_store_timestep_pair")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, src_k, 0);
        set_gpu_buf(bnd, src_v, 1);
        set_gpu_buf(bnd, dst_k, 2);
        set_gpu_buf(bnd, dst_v, 3);
        scalars(bnd);
        set_gpu_buf(bnd, dst_offset, 5);
    })
}

/// Chronological densify from a ring buffer: `dst[t] = src[(start + t) % capacity]`.
///
/// `filled` and `start` are device `u32`s for the same reason as
/// [`kv_store_timestep`]'s offset.
///
/// Scalar indices for `_with_scalars`: 2 = `n_slot`, 3 = `capacity`. Buffers
/// 4 and 5 are `filled` and `start`.
pub fn kv_ring_densify(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    filled: &GpuBuffer,
    start: &GpuBuffer,
    n_slot: u32,
    capacity: u32,
) -> Result<(), String> {
    kv_ring_densify_with_scalars(rt, src, dst, filled, start, n_slot, capacity, |bnd| {
        set_u32(bnd, n_slot, 2);
        set_u32(bnd, capacity, 3);
    })
}

/// [`kv_ring_densify`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn kv_ring_densify_with_scalars(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    filled: &GpuBuffer,
    start: &GpuBuffer,
    n_slot: u32,
    capacity: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if capacity == 0 {
        return Err("kv_ring_densify: capacity must be non-zero (kernel takes % capacity)".into());
    }
    let ring = elems(capacity, n_slot, "kv_ring_densify")?;
    require::<f32>(src, ring, "kv_ring_densify src")?;
    require::<f32>(dst, ring, "kv_ring_densify dst")?;
    require::<u32>(filled, 1, "kv_ring_densify filled")?;
    require::<u32>(start, 1, "kv_ring_densify start")?;
    let p = rt.pipeline("kv_ring_densify")?;
    dispatch_1d(rt, &p, ring, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        scalars(bnd);
        set_gpu_buf(bnd, filled, 4);
        set_gpu_buf(bnd, start, 5);
    })
}

// --------------------------------------------------------- Dispatch helper ---

/// Encode a 1-D threadgroup dispatch with an explicit group count.
///
/// [`dispatch_1d`] derives its threadgroup size from `threadExecutionWidth`,
/// which is right for elementwise kernels and wrong for every kernel below:
/// a simdgroup-cooperative GEMV or a threadgroup-wide reduction needs a
/// specific number of threads per group, and a wrong one silently changes what
/// the kernel computes rather than failing.
fn dispatch_tg_1d(
    rt: &Arc<GpuRuntime>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups: usize,
    threads_per_tg: usize,
    tg_memory: Option<(usize, usize)>,
    encode: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups == 0 || threads_per_tg == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode(bnd);
        if let Some((index, bytes)) = tg_memory {
            bnd.set_threadgroup_memory(index, bytes);
        }
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(threads_per_tg, 1, 1));
        Ok(())
    })
}

// ------------------------------------------------------- Flash attention ---

/// Head dimension a flash-attention entry point is compiled for.
///
/// Each kernel bakes `HEAD_DIM` in as a `constant`, so the head dimension
/// selects the kernel rather than being passed to it. Calling one with the
/// wrong `D` reads past the end of every head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnHeadDim {
    /// `flash_attn_swa_h128` / sliding window, `BR = 8`.
    D128,
    /// `flash_attn_swa_h256` / sliding window, `BR = 8`.
    D256,
}

impl AttnHeadDim {
    fn dim(self) -> u32 {
        match self {
            Self::D128 => 128,
            Self::D256 => 256,
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::D128 => "flash_attn_swa_h128",
            Self::D256 => "flash_attn_swa_h256",
        }
    }

    /// Query-block rows per threadgroup, from the kernel's `constant uint BR`.
    fn br(self) -> usize {
        8
    }
}

/// Sliding-window flash attention.
///
/// `Q` and `O` are `[B, Tq, H, D]`; `K` and `V` are `[B, Tkv, Hkv, D]` with
/// `H` a multiple of `Hkv` (grouped-query attention).
///
/// `tkv`, `q_pos_offset` and `kv_pos_offset` are device `u32` buffers, not
/// constants: during decode they change every token, and an Indirect Command
/// Buffer that froze its binds needs a stable address whose contents move.
///
/// `Tkv` therefore is not knowable on the host, so `k` and `v` extents cannot
/// be validated here — only `q` and `o`, against `B * Tq * H * D`.
///
/// Scalar indices for `_with_scalars`: 4 = `B`, 5 = `Tq`, 7 = `H`, 8 = `Hkv`,
/// 9 = `window`, 10 = `scale` (f32). Buffers 6, 11, 12 are bound here.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_swa(
    rt: &Arc<GpuRuntime>,
    head_dim: AttnHeadDim,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
) -> Result<(), String> {
    flash_attn_swa_with_scalars(
        rt,
        head_dim,
        q,
        k,
        v,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        dims,
        |bnd| {
            set_u32(bnd, dims.batch, 4);
            set_u32(bnd, dims.tq, 5);
            set_u32(bnd, dims.heads, 7);
            set_u32(bnd, dims.heads_kv, 8);
            set_u32(bnd, dims.window, 9);
            set_f32(bnd, dims.scale, 10);
        },
    )
}

/// Shapes and scalars shared by the attention entry points.
#[derive(Clone, Copy, Debug)]
pub struct AttnDims {
    /// Batch size.
    pub batch: u32,
    /// Query positions in this dispatch (1 during decode, `T` during prefill).
    pub tq: u32,
    /// Query heads.
    pub heads: u32,
    /// Key/value heads. `heads` must be a multiple of this.
    pub heads_kv: u32,
    /// Sliding-window span. Ignored by the global entry point.
    pub window: u32,
    /// Softmax scale, conventionally `1 / sqrt(D)`.
    pub scale: f32,
}

/// [`flash_attn_swa`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_swa_with_scalars(
    rt: &Arc<GpuRuntime>,
    head_dim: AttnHeadDim,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let d = head_dim.dim();
    // The sliding-window kernels always write f32.
    validate_attn_dims(&dims, d, q, o, "flash_attn_swa", false)?;
    require::<u32>(tkv, 1, "flash_attn_swa tkv")?;
    require::<u32>(q_pos_offset, 1, "flash_attn_swa q_pos_offset")?;
    require::<u32>(kv_pos_offset, 1, "flash_attn_swa kv_pos_offset")?;
    // K/V hold at least one position each; `Tkv` lives on the device.
    require::<f32>(
        k,
        elems(dims.batch * dims.heads_kv, d, "flash_attn_swa k")?,
        "flash_attn_swa k",
    )?;
    require::<f32>(
        v,
        elems(dims.batch * dims.heads_kv, d, "flash_attn_swa v")?,
        "flash_attn_swa v",
    )?;

    let p = rt.pipeline(head_dim.entry())?;
    let groups_x = (dims.tq as usize).div_ceil(head_dim.br());
    let groups_y = (dims.batch as usize).saturating_mul(dims.heads as usize);
    dispatch_2d_tg(rt, &p, groups_x, groups_y, 32, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf(bnd, tkv, 6);
        scalars(bnd);
        set_gpu_buf(bnd, q_pos_offset, 11);
        set_gpu_buf(bnd, kv_pos_offset, 12);
    })
}

/// Global (non-windowed) flash attention at head dimension 512.
///
/// A separate entry point rather than a variant of [`flash_attn_swa`] because
/// the kernel's buffer layout differs: there is no `window`, and index 12
/// carries an `out_bf16` flag instead of a position offset.
///
/// Scalar indices for `_with_scalars`: 4 = `B`, 5 = `Tq`, 7 = `H`, 8 = `Hkv`,
/// 9 = `scale` (f32), 12 = `out_bf16`. Buffers 6, 10, 11 are bound here.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_global_h512(
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
) -> Result<(), String> {
    flash_attn_global_h512_with_scalars(
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
        |bnd| {
            set_u32(bnd, dims.batch, 4);
            set_u32(bnd, dims.tq, 5);
            set_u32(bnd, dims.heads, 7);
            set_u32(bnd, dims.heads_kv, 8);
            set_f32(bnd, dims.scale, 9);
            set_u32(bnd, u32::from(out_bf16), 12);
        },
    )
}

/// [`flash_attn_global_h512`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_global_h512_with_scalars(
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
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    const D: u32 = 512;
    // `BR = 4` for this kernel, not 8 — see its `constant uint BR`.
    const BR: usize = 4;
    validate_attn_dims(&dims, D, q, o, "flash_attn_global_h512", out_bf16)?;
    require::<u32>(tkv, 1, "flash_attn_global_h512 tkv")?;
    require::<u32>(q_pos_offset, 1, "flash_attn_global_h512 q_pos_offset")?;
    require::<u32>(kv_pos_offset, 1, "flash_attn_global_h512 kv_pos_offset")?;
    require::<f32>(
        k,
        elems(dims.batch * dims.heads_kv, D, "k")?,
        "flash_attn_global_h512 k",
    )?;
    require::<f32>(
        v,
        elems(dims.batch * dims.heads_kv, D, "v")?,
        "flash_attn_global_h512 v",
    )?;

    let p = rt.pipeline("flash_attn_global_h512")?;
    let groups_x = (dims.tq as usize).div_ceil(BR);
    let groups_y = (dims.batch as usize).saturating_mul(dims.heads as usize);
    dispatch_2d_tg(rt, &p, groups_x, groups_y, 32, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf(bnd, tkv, 6);
        set_gpu_buf(bnd, q_pos_offset, 10);
        set_gpu_buf(bnd, kv_pos_offset, 11);
        scalars(bnd);
    })
}

/// Shared validation for the attention entry points.
///
/// `o` is checked as f32 here; the h512 path re-checks it as bf16 when its
/// `out_bf16` flag is set.
fn validate_attn_dims(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    o: &GpuBuffer,
    what: &str,
    out_bf16: bool,
) -> Result<(), String> {
    if dims.heads_kv == 0 {
        return Err(format!("{what}: heads_kv must be non-zero"));
    }
    if dims.heads % dims.heads_kv != 0 {
        return Err(format!(
            "{what}: heads {} is not a multiple of heads_kv {} — grouped-query \
             attention maps a whole number of query heads onto each kv head",
            dims.heads, dims.heads_kv
        ));
    }
    if !dims.scale.is_finite() {
        return Err(format!("{what}: scale must be finite, got {}", dims.scale));
    }
    let n = elems(dims.batch * dims.tq * dims.heads, d, what)?;
    require::<f32>(q, n, &format!("{what} q"))?;
    if out_bf16 {
        // `out_bf16` exists to halve this buffer — the kernel writes `bfloat`
        // into it. Validating `o` as f32 regardless demanded twice the memory
        // the kernel touches, so a caller who sized it correctly for bf16 got
        // "buffer holds 2560 elements, kernel reads/writes 5120" and the
        // documented half-width scratch was unreachable.
        require::<u16>(o, n, &format!("{what} o (bf16)"))?;
    } else {
        require::<f32>(o, n, &format!("{what} o"))?;
    }
    Ok(())
}

// ------------------------------------------------- Fused RMSNorm+QKV+RoPE ---

/// Shapes and scalars for the fused QKV normalization + rotary embedding.
#[derive(Clone, Copy, Debug)]
pub struct QkvRopeDims {
    /// Positions in this dispatch.
    pub t: u32,
    /// Query heads.
    pub heads_q: u32,
    /// Key/value heads.
    pub heads_kv: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Leading slice of each head that RoPE rotates. `<= head_dim`.
    pub rotary_dim: u32,
    /// RoPE base frequency (10000 in the original formulation).
    pub theta: f32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl QkvRopeDims {
    /// Heads processed per dispatch: all of Q, plus K and V when they are
    /// written.
    fn head_count(&self, q_only: bool) -> Result<usize, String> {
        let q = (self.t as usize).checked_mul(self.heads_q as usize);
        let kv = (self.t as usize).checked_mul(self.heads_kv as usize);
        match (q, kv) {
            (Some(q), Some(_)) if q_only => Ok(q),
            (Some(q), Some(kv)) => q
                .checked_add(2 * kv)
                .ok_or_else(|| "qkv_rope: head count overflows usize".to_string()),
            _ => Err("qkv_rope: head count overflows usize".to_string()),
        }
    }

    fn validate(&self, what: &str) -> Result<(), String> {
        if self.head_dim == 0 {
            return Err(format!("{what}: head_dim must be non-zero"));
        }
        if self.rotary_dim > self.head_dim {
            return Err(format!(
                "{what}: rotary_dim {} exceeds head_dim {} — RoPE would rotate \
                 past the end of each head",
                self.rotary_dim, self.head_dim
            ));
        }
        if self.rotary_dim % 2 != 0 {
            return Err(format!(
                "{what}: rotary_dim {} is odd; RoPE rotates (even, odd) pairs",
                self.rotary_dim
            ));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(format!("{what}: theta must be finite and positive"));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(format!("{what}: eps must be finite and non-negative"));
        }
        Ok(())
    }
}

/// Which fused QKV+RoPE entry point to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkvRopeVariant {
    /// `rms_qkv_rope` — position offset passed as a constant.
    PosConst,
    /// `rms_qkv_rope_posbuf` — position offset read from a device `u32`, so an
    /// ICB that froze its binds can advance the position without re-encoding.
    PosBuffer,
    /// `rms_qkv_rope_kv_store` — as [`Self::PosBuffer`], and additionally
    /// writes the rotated K and V straight into the KV cache, saving a
    /// separate [`kv_store_timestep_pair`] dispatch.
    PosBufferKvStore,
}

impl QkvRopeVariant {
    fn entry(self) -> &'static str {
        match self {
            Self::PosConst => "rms_qkv_rope",
            Self::PosBuffer => "rms_qkv_rope_posbuf",
            Self::PosBufferKvStore => "rms_qkv_rope_kv_store",
        }
    }
}

/// Destination for [`QkvRopeVariant::PosBufferKvStore`].
#[derive(Clone, Copy)]
pub struct KvStoreTarget<'a> {
    /// Key cache.
    pub dst_k: &'a GpuBuffer,
    /// Value cache.
    pub dst_v: &'a GpuBuffer,
    /// Device `u32` element offset into both caches.
    pub dst_offset: &'a GpuBuffer,
}

/// Fused per-head RMSNorm, QKV projection scaling, and rotary embedding.
///
/// `q`, `k` and `v` are read and written in place: they arrive holding the raw
/// projection output and leave normalized and rotated.
///
/// `pos_offset` is a constant for [`QkvRopeVariant::PosConst`] and a device
/// `u32` buffer for the other two. Exactly one must be supplied; passing the
/// wrong one for the variant is refused rather than silently ignored.
///
/// Scalar indices for `_with_scalars`: 6 = `T`, 7 = `Hq`, 8 = `Hkv`,
/// 9 = `D`, 10 = `rotary_dim`, 11 = `pos_offset` (`PosConst` only),
/// 12 = `theta` (f32), 13 = `eps` (f32).
#[allow(clippy::too_many_arguments)]
pub fn rms_qkv_rope(
    rt: &Arc<GpuRuntime>,
    variant: QkvRopeVariant,
    qkv: QkvBuffers<'_>,
    dims: QkvRopeDims,
    pos_offset: u32,
    pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>,
    q_only: bool,
) -> Result<(), String> {
    rms_qkv_rope_with_scalars(
        rt,
        variant,
        qkv,
        dims,
        pos_offset_buf,
        kv_store,
        q_only,
        |bnd| {
            set_u32(bnd, dims.t, 6);
            set_u32(bnd, dims.heads_q, 7);
            set_u32(bnd, dims.heads_kv, 8);
            set_u32(bnd, dims.head_dim, 9);
            set_u32(bnd, dims.rotary_dim, 10);
            if variant == QkvRopeVariant::PosConst {
                set_u32(bnd, pos_offset, 11);
            }
            set_f32(bnd, dims.theta, 12);
            set_f32(bnd, dims.eps, 13);
        },
    )
}

/// The six in/out buffers every fused QKV+RoPE entry point takes.
#[derive(Clone, Copy)]
pub struct QkvBuffers<'a> {
    /// Query activations, rewritten in place.
    pub q: &'a GpuBuffer,
    /// Key activations, rewritten in place.
    pub k: &'a GpuBuffer,
    /// Value activations, rewritten in place.
    pub v: &'a GpuBuffer,
    /// Per-channel RMSNorm weight for Q.
    pub q_weight: &'a GpuBuffer,
    /// Per-channel RMSNorm weight for K.
    pub k_weight: &'a GpuBuffer,
    /// Per-channel RMSNorm weight for V.
    pub v_weight: &'a GpuBuffer,
}

/// [`rms_qkv_rope`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn rms_qkv_rope_with_scalars(
    rt: &Arc<GpuRuntime>,
    variant: QkvRopeVariant,
    qkv: QkvBuffers<'_>,
    dims: QkvRopeDims,
    pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>,
    q_only: bool,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    dims.validate("rms_qkv_rope")?;

    // The variant selects the kernel, and the kernel decides which of these
    // operands exist. Accepting a mismatched pair and ignoring the extra one
    // is how a caller ends up reading a stale position for a whole session.
    match (variant, pos_offset_buf) {
        (QkvRopeVariant::PosConst, Some(_)) => {
            return Err("rms_qkv_rope: PosConst takes a constant offset, not a buffer".into())
        }
        (QkvRopeVariant::PosConst, None) => {}
        (_, None) => return Err("rms_qkv_rope: PosBuffer variants require pos_offset_buf".into()),
        (_, Some(b)) => require::<u32>(b, 1, "rms_qkv_rope pos_offset_buf")?,
    }
    match (variant, kv_store.is_some()) {
        (QkvRopeVariant::PosBufferKvStore, false) => {
            return Err("rms_qkv_rope: PosBufferKvStore requires a KvStoreTarget".into())
        }
        (QkvRopeVariant::PosBufferKvStore, true) | (_, false) => {}
        (_, true) => return Err("rms_qkv_rope: only PosBufferKvStore writes the KV cache".into()),
    }

    let d = dims.head_dim as usize;
    let q_heads = (dims.t as usize).saturating_mul(dims.heads_q as usize);
    let kv_heads = (dims.t as usize).saturating_mul(dims.heads_kv as usize);
    require::<f32>(qkv.q, q_heads * d, "rms_qkv_rope q")?;
    require::<f32>(qkv.q_weight, d, "rms_qkv_rope q_weight")?;
    if !q_only {
        require::<f32>(qkv.k, kv_heads * d, "rms_qkv_rope k")?;
        require::<f32>(qkv.v, kv_heads * d, "rms_qkv_rope v")?;
        require::<f32>(qkv.k_weight, d, "rms_qkv_rope k_weight")?;
        require::<f32>(qkv.v_weight, d, "rms_qkv_rope v_weight")?;
    }
    if let Some(t) = &kv_store {
        require::<f32>(t.dst_k, kv_heads * d, "rms_qkv_rope dst_k")?;
        require::<f32>(t.dst_v, kv_heads * d, "rms_qkv_rope dst_v")?;
        require::<u32>(t.dst_offset, 1, "rms_qkv_rope kv_dst_offset")?;
    }

    let n = dims.head_count(q_only)?;
    let p = rt.pipeline(variant.entry())?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, qkv.q, 0);
        set_gpu_buf(bnd, qkv.k, 1);
        set_gpu_buf(bnd, qkv.v, 2);
        set_gpu_buf(bnd, qkv.q_weight, 3);
        set_gpu_buf(bnd, qkv.k_weight, 4);
        set_gpu_buf(bnd, qkv.v_weight, 5);
        scalars(bnd);
        if let Some(b) = pos_offset_buf {
            set_gpu_buf(bnd, b, 11);
        }
        if let Some(t) = kv_store {
            set_gpu_buf(bnd, t.dst_k, 14);
            set_gpu_buf(bnd, t.dst_v, 15);
            set_gpu_buf(bnd, t.dst_offset, 16);
        }
    })
}

// -------------------------------------------------------------- Sampling ---

/// Largest threadgroup a tree reduction can use here.
///
/// The reduction loop halves `tptg` each round and reads `tg_val[lid + stride]`,
/// so a non-power-of-two threadgroup drops elements silently — it produces a
/// plausible token rather than an error. Every entry point below rounds down to
/// a power of two and clamps to the pipeline's own limit.
fn reduction_tptg(max_threads: usize, want: usize, tg_array_len: usize) -> usize {
    let cap = max_threads.min(tg_array_len).min(want).max(1);
    // `prev_power_of_two`: 1 << floor(log2(cap)).
    1usize << (usize::BITS - 1 - cap.leading_zeros()) as usize
}

/// `logits[i] = softcap * tanh(logits[i] / softcap)`, in place.
///
/// `softcap` is a device `f32` buffer rather than a constant so an ICB that
/// froze its binds can change the cap without re-encoding.
///
/// Scalar index for `_with_scalars`: 2 = `n`. Buffer 1 is `softcap`.
pub fn softcap_logits(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    softcap_logits_with_scalars(rt, logits, softcap, n, |bnd| set_u32(bnd, n, 2))
}

/// [`softcap_logits`] with caller-supplied scalar binds.
pub fn softcap_logits_with_scalars(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    require::<f32>(logits, n as usize, "softcap_logits logits")?;
    require::<f32>(softcap, 1, "softcap_logits softcap")?;
    let p = rt.pipeline("softcap_logits")?;
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, softcap, 1);
        scalars(bnd);
    })
}

/// One reduction pass of a multi-pass GPU argmax.
///
/// Writes one `(index, value)` pair per threadgroup into `out_idx` / `out_val`,
/// so a full argmax over `n` logits is this called repeatedly until one group
/// remains. The first pass fuses the softcap on read and passes `idx_in = None`;
/// later passes pass the previous pass's `out_idx` so original vocabulary
/// indices propagate rather than being re-derived from partial offsets.
///
/// `out_idx` and `out_val` must each hold [`argmax_pass_groups`] elements.
///
/// Scalar indices for `_with_scalars`: 3 = `n`, 5 = `has_idx_in`.
/// Buffers 4 (`idx_in`) and 6 (`softcap`) are bound here.
#[allow(clippy::too_many_arguments)]
pub fn argmax_f32_pass(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_idx: &GpuBuffer,
    out_val: &GpuBuffer,
    idx_in: Option<&GpuBuffer>,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    let has = u32::from(idx_in.is_some());
    argmax_f32_pass_with_scalars(rt, logits, out_idx, out_val, idx_in, softcap, n, |bnd| {
        set_u32(bnd, n, 3);
        set_u32(bnd, has, 5);
    })
}

/// Threadgroups [`argmax_f32_pass`] launches for `n` inputs, and therefore the
/// number of partial results it writes.
pub fn argmax_pass_groups(n: u32) -> usize {
    (n as usize).div_ceil(ARGMAX_TG)
}

/// Threads per group for the argmax reduction, matching the kernel's
/// `threadgroup` array extents.
const ARGMAX_TG: usize = 256;

/// [`argmax_f32_pass`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn argmax_f32_pass_with_scalars(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_idx: &GpuBuffer,
    out_val: &GpuBuffer,
    idx_in: Option<&GpuBuffer>,
    softcap: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Err("argmax_f32_pass: n must be non-zero".into());
    }
    let groups = argmax_pass_groups(n);
    require::<f32>(logits, n as usize, "argmax_f32_pass logits")?;
    require::<u32>(out_idx, groups, "argmax_f32_pass out_idx")?;
    require::<f32>(out_val, groups, "argmax_f32_pass out_val")?;
    require::<f32>(softcap, 1, "argmax_f32_pass softcap")?;
    if let Some(b) = idx_in {
        require::<u32>(b, n as usize, "argmax_f32_pass idx_in")?;
    }

    let p = rt.pipeline("argmax_f32")?;
    // Buffer 4 must be bound even on the first pass: the kernel reads the
    // binding unconditionally and gates on `has_idx_in`, so leaving the slot
    // empty is an unbound-buffer fault, not a no-op.
    let placeholder;
    let idx_buf = match idx_in {
        Some(b) => b,
        None => {
            placeholder = rt.alloc_buffer(4)?;
            &placeholder
        }
    };
    dispatch_tg_1d(rt, &p, groups, ARGMAX_TG, None, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, out_idx, 1);
        set_gpu_buf(bnd, out_val, 2);
        scalars(bnd);
        set_gpu_buf(bnd, idx_buf, 4);
        set_gpu_buf(bnd, softcap, 6);
    })
}

/// Softcap `logits` in place and write the argmax index to `out_token`.
///
/// Single threadgroup, so `n` may not exceed the threadgroup size — the kernel
/// stages `logits[lid]` one per lane and never strides. For a full vocabulary
/// use [`softcap_argmax_one_pass`], which does stride.
///
/// Scalar index for `_with_scalars`: 3 = `n`. Buffer 2 is `softcap`.
pub fn softcap_sample(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    softcap_sample_with_scalars(rt, logits, out_token, softcap, n, |bnd| set_u32(bnd, n, 3))
}

/// [`softcap_sample`] with caller-supplied scalar binds.
pub fn softcap_sample_with_scalars(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Err("softcap_sample: n must be non-zero".into());
    }
    require::<f32>(logits, n as usize, "softcap_sample logits")?;
    require::<u32>(out_token, 1, "softcap_sample out_token")?;
    require::<f32>(softcap, 1, "softcap_sample softcap")?;

    let p = rt.pipeline("softcap_sample")?;
    let max_threads = p.maxTotalThreadsPerThreadgroup();
    let tptg = reduction_tptg(max_threads, (n as usize).next_power_of_two(), 256);
    if (n as usize) > tptg {
        return Err(format!(
            "softcap_sample: n = {n} exceeds the {tptg}-lane threadgroup this \
             kernel reduces over; logits past lane {tptg} would be ignored. \
             Use softcap_argmax_one_pass for a full vocabulary."
        ));
    }
    dispatch_tg_1d(rt, &p, 1, tptg, None, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, out_token, 1);
        set_gpu_buf(bnd, softcap, 2);
        scalars(bnd);
    })
}

/// Softcap-and-argmax over an arbitrarily large `logits`, in one dispatch.
///
/// One threadgroup whose lanes each scan a strided slice, then reduce. Unlike
/// [`softcap_sample`] it does **not** rewrite `logits`: decode only needs the
/// index, and skipping the write avoids restating a full vocabulary.
///
/// Scalar index for `_with_scalars`: 3 = `n`. Buffer 2 is `softcap`.
pub fn softcap_argmax_one_pass(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    softcap_argmax_one_pass_with_scalars(rt, logits, out_token, softcap, n, |bnd| {
        set_u32(bnd, n, 3)
    })
}

/// [`softcap_argmax_one_pass`] with caller-supplied scalar binds.
pub fn softcap_argmax_one_pass_with_scalars(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Err("softcap_argmax_one_pass: n must be non-zero".into());
    }
    require::<f32>(logits, n as usize, "softcap_argmax_one_pass logits")?;
    require::<u32>(out_token, 1, "softcap_argmax_one_pass out_token")?;
    require::<f32>(softcap, 1, "softcap_argmax_one_pass softcap")?;

    let p = rt.pipeline("softcap_argmax_one_pass")?;
    // The kernel's threadgroup arrays are 1024 long and every lane writes its
    // own slot, so the group may not exceed that regardless of device limits.
    let tptg = reduction_tptg(p.maxTotalThreadsPerThreadgroup(), 1024, 1024);
    dispatch_tg_1d(rt, &p, 1, tptg, None, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, out_token, 1);
        set_gpu_buf(bnd, softcap, 2);
        scalars(bnd);
    })
}

// ------------------------------------------------- Quantized weight banks ---

/// Shape of a group-wise quantized weight matrix.
///
/// Quantization is affine and grouped along `cols`: each run of `group_size`
/// consecutive weights in a row shares one scale and one zero point (or bias).
#[derive(Clone, Copy, Debug)]
pub struct QuantShape {
    /// Output rows.
    pub rows: u32,
    /// Reduction length.
    pub cols: u32,
    /// Weights per quantization group. Must divide `cols`.
    pub group_size: u32,
}

impl QuantShape {
    /// Quantization groups in the whole matrix.
    pub fn groups(&self) -> Result<usize, String> {
        self.validate("QuantShape")?;
        elems(self.rows, self.cols / self.group_size, "QuantShape groups")
    }

    fn validate(&self, what: &str) -> Result<(), String> {
        if self.group_size == 0 {
            return Err(format!("{what}: group_size must be non-zero"));
        }
        if self.cols % self.group_size != 0 {
            return Err(format!(
                "{what}: cols {} is not a multiple of group_size {}; the kernels \
                 compute `cols / group_size` with integer division and would \
                 silently drop the ragged tail group",
                self.cols, self.group_size
            ));
        }
        Ok(())
    }
}

/// Q4 weights with separate f32 scale and zero-point tables.
///
/// `packed` holds two 4-bit weights per byte, low nibble first. The nibble is
/// **signed**: the kernels sign-extend it with `(int)(n << 28) >> 28`, so a
/// stored 8 means -8 and the value range is -8..=7. Dequantization is
/// `w = scale * (q - zero)`.
///
/// [`Q4MlxBank`] does not share this convention — it reads the nibble unsigned
/// and adds a bias. The two are not interchangeable.
#[derive(Clone, Copy)]
pub struct Q4Bank<'a> {
    /// Packed nibbles, `rows * cols / 2` bytes.
    pub packed: &'a GpuBuffer,
    /// One f32 per group.
    pub scales: &'a GpuBuffer,
    /// One f32 per group.
    pub zeros: &'a GpuBuffer,
}

impl Q4Bank<'_> {
    fn validate(&self, shape: &QuantShape, what: &str) -> Result<(), String> {
        shape.validate(what)?;
        let groups = shape.groups()?;
        let weights = elems(shape.rows, shape.cols, what)?;
        require::<u8>(self.packed, weights.div_ceil(2), &format!("{what} packed"))?;
        require::<f32>(self.scales, groups, &format!("{what} scales"))?;
        require::<f32>(self.zeros, groups, &format!("{what} zeros"))?;
        Ok(())
    }
}

/// MLX-format Q4 weights: packed nibbles plus interleaved `(scale, bias)` pairs.
///
/// The scale and bias for a group sit adjacent as a `bfloat2`, so one 4-byte
/// load fetches both. Dequantization is `w = scale * nibble + bias` — an add,
/// not the subtract [`Q4Bank`] uses.
///
/// The kernels also take a third `biases` buffer at the following index and
/// never read it (`(void)biases_unused` in the source). These wrappers bind
/// `scales_biases` there rather than making callers carry a buffer that exists
/// only to fill a slot.
#[derive(Clone, Copy)]
pub struct Q4MlxBank<'a> {
    /// Packed nibbles, `rows * cols / 2` bytes.
    pub packed: &'a GpuBuffer,
    /// Interleaved `bfloat2` scale/bias pairs, one per group.
    pub scales_biases: &'a GpuBuffer,
}

impl Q4MlxBank<'_> {
    fn validate(&self, shape: &QuantShape, what: &str) -> Result<(), String> {
        shape.validate(what)?;
        let groups = shape.groups()?;
        let weights = elems(shape.rows, shape.cols, what)?;
        require::<u8>(self.packed, weights.div_ceil(2), &format!("{what} packed"))?;
        // One bfloat2 = two u16 = 4 bytes per group.
        require::<u32>(
            self.scales_biases,
            groups,
            &format!("{what} scales_biases (bfloat2 per group)"),
        )?;
        Ok(())
    }
}

// -------------------------------------------------------------- Q4 GEMV ---

/// `y[rows] = W[rows, cols] @ x[cols]` with [`Q4Bank`] weights.
///
/// One thread per output row. Each threadgroup stages the whole `x` vector in
/// threadgroup memory, so `cols * 4` bytes must fit the device limit — checked
/// against [`GpuRuntime::max_threadgroup_memory`] before encoding, because the
/// dispatch-time failure names neither this kernel nor `cols`.
///
/// Set `tiled` to use `gemv_q4_tiled`, which walks one threadgroup per row tile
/// instead of one thread per row.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`, 7 = `group_size`.
pub fn gemv_q4(
    rt: &Arc<GpuRuntime>,
    bank: Q4Bank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    tiled: bool,
) -> Result<(), String> {
    gemv_q4_with_scalars(rt, bank, x, y, shape, tiled, |bnd| {
        set_u32(bnd, shape.rows, 5);
        set_u32(bnd, shape.cols, 6);
        set_u32(bnd, shape.group_size, 7);
    })
}

/// [`gemv_q4`] with caller-supplied scalar binds.
pub fn gemv_q4_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4Bank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    tiled: bool,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = if tiled { "gemv_q4_tiled" } else { "gemv_q4" };
    bank.validate(&shape, entry)?;
    require::<f32>(x, shape.cols as usize, &format!("{entry} x"))?;
    require::<f32>(y, shape.rows as usize, &format!("{entry} y"))?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    // The two kernels take *different* grids, and until 2026-08-31 both were
    // dispatched with the one-thread-per-row geometry in the `else` arm.
    //
    // `gemv_q4_tiled` indexes its output row by `threadgroup_position_in_grid`
    // and returns when that exceeds `rows`, so it needs one threadgroup per row.
    // Handing it `rows.div_ceil(128)` groups meant it wrote the first
    // `rows / 128` rows and left every other row of `y` untouched — no error, no
    // partial-write signal, just whatever was in the buffer before. Measured at
    // 512 rows it wrote 4 and left 508 holding a sentinel. The benchmark is what
    // caught it: 3,077 GB/s is not a number this machine can produce, and it was
    // doing 0.8% of the work.
    //
    // It also declares its scratch statically (`threadgroup float
    // partial[GEMV_TG]`) and never caches `x`, so the dynamic threadgroup
    // allocation, and the `cols` ceiling that exists to bound it, belong to the
    // one-thread-per-row kernel alone.
    let (groups, tptg, tg_mem) = if tiled {
        (shape.rows as usize, GEMV_TILED_TPTG, None)
    } else {
        let bytes = (shape.cols as usize).saturating_mul(4);
        let limit = rt.max_threadgroup_memory();
        if bytes > limit {
            return Err(format!(
                "{entry}: caching x needs {bytes} bytes of threadgroup memory but this \
                 device allows {limit}; cols {} is too large for this kernel",
                shape.cols
            ));
        }
        let t = reduction_tptg(
            p.maxTotalThreadsPerThreadgroup(),
            GEMV_ROW_TPTG,
            GEMV_ROW_TPTG,
        )
        .min(shape.rows as usize)
        .max(1);
        ((shape.rows as usize).div_ceil(t), t, Some((0, bytes)))
    };
    dispatch_tg_1d(rt, &p, groups, tptg, tg_mem, |bnd| {
        set_gpu_buf(bnd, bank.packed, 0);
        set_gpu_buf(bnd, bank.scales, 1);
        set_gpu_buf(bnd, bank.zeros, 2);
        set_gpu_buf(bnd, x, 3);
        set_gpu_buf(bnd, y, 4);
        scalars(bnd);
    })
}

/// Threads per group for the one-thread-per-row Q4 GEMV kernels.
///
/// 128 amortizes the shared `x` cache across enough rows to pay for staging it,
/// without making the tail group wasteful on short matrices.
const GEMV_ROW_TPTG: usize = 128;

/// Threads per threadgroup for `gemv_q4_tiled`.
///
/// Must equal `GEMV_TG` in `kernels/gemv_q4.metal`: the kernel sizes its
/// `partial[]` scratch and its tree reduction by that constant, so a smaller
/// launch leaves the upper half of the array uninitialised and a larger one
/// overruns it.
const GEMV_TILED_TPTG: usize = 128;

// ---------------------------------------------------- Embedding lookup ---

/// Gather `n_tokens` embedding rows from a quantized table into `out`.
///
/// Dequantizes on the GPU, so a quantized embedding table never has to be
/// expanded host-side each step. Token ids at or beyond `vocab` yield a zero
/// row rather than reading out of bounds.
///
/// Scalar indices for `_with_scalars`: 5 = `hidden`, 6 = `group_size`,
/// 7 = `vocab`, 8 = `n_tokens`.
#[allow(clippy::too_many_arguments)]
pub fn embed_lookup_q4(
    rt: &Arc<GpuRuntime>,
    bank: Q4Bank<'_>,
    token_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
    hidden: u32,
    group_size: u32,
    n_tokens: u32,
) -> Result<(), String> {
    embed_lookup_q4_with_scalars(
        rt,
        bank,
        token_ids,
        out,
        vocab,
        hidden,
        group_size,
        n_tokens,
        |bnd| {
            set_u32(bnd, hidden, 5);
            set_u32(bnd, group_size, 6);
            set_u32(bnd, vocab, 7);
            set_u32(bnd, n_tokens, 8);
        },
    )
}

/// [`embed_lookup_q4`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn embed_lookup_q4_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4Bank<'_>,
    token_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
    hidden: u32,
    group_size: u32,
    n_tokens: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let shape = QuantShape {
        rows: vocab,
        cols: hidden,
        group_size,
    };
    bank.validate(&shape, "embed_lookup_q4")?;
    let total = elems(n_tokens, hidden, "embed_lookup_q4")?;
    require::<u32>(token_ids, n_tokens as usize, "embed_lookup_q4 token_ids")?;
    require::<f32>(out, total, "embed_lookup_q4 out")?;

    let p = rt.pipeline("embed_lookup_q4")?;
    dispatch_1d(rt, &p, total, |bnd| {
        set_gpu_buf(bnd, bank.packed, 0);
        set_gpu_buf(bnd, bank.scales, 1);
        set_gpu_buf(bnd, bank.zeros, 2);
        set_gpu_buf(bnd, token_ids, 3);
        set_gpu_buf(bnd, out, 4);
        scalars(bnd);
    })
}

/// [`embed_lookup_q4`] for an MLX-format table.
///
/// Scalar indices for `_with_scalars`: 5 = `hidden`, 6 = `group_size`,
/// 7 = `vocab`, 8 = `n_tokens`.
#[allow(clippy::too_many_arguments)]
pub fn embed_lookup_q4_mlx(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    token_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
    hidden: u32,
    group_size: u32,
    n_tokens: u32,
) -> Result<(), String> {
    embed_lookup_q4_mlx_with_scalars(
        rt,
        bank,
        token_ids,
        out,
        vocab,
        hidden,
        group_size,
        n_tokens,
        |bnd| {
            set_u32(bnd, hidden, 5);
            set_u32(bnd, group_size, 6);
            set_u32(bnd, vocab, 7);
            set_u32(bnd, n_tokens, 8);
        },
    )
}

/// [`embed_lookup_q4_mlx`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn embed_lookup_q4_mlx_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    token_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
    hidden: u32,
    group_size: u32,
    n_tokens: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let shape = QuantShape {
        rows: vocab,
        cols: hidden,
        group_size,
    };
    bank.validate(&shape, "embed_lookup_q4_mlx")?;
    let total = elems(n_tokens, hidden, "embed_lookup_q4_mlx")?;
    require::<u32>(
        token_ids,
        n_tokens as usize,
        "embed_lookup_q4_mlx token_ids",
    )?;
    require::<f32>(out, total, "embed_lookup_q4_mlx out")?;

    let p = rt.pipeline("embed_lookup_q4_mlx")?;
    dispatch_1d(rt, &p, total, |bnd| {
        set_gpu_buf(bnd, bank.packed, 0);
        set_gpu_buf(bnd, bank.scales_biases, 1);
        // Slot 2 is the kernel's `biases_unused`; see `Q4MlxBank`.
        set_gpu_buf(bnd, bank.scales_biases, 2);
        set_gpu_buf(bnd, token_ids, 3);
        set_gpu_buf(bnd, out, 4);
        scalars(bnd);
    })
}

// --------------------------------------------------------- MLX Q4 GEMV ---

/// Row packing of an MLX Q4 bank.
///
/// The two layouts hold identical weights; they differ in the order nibbles sit
/// within a byte, which changes how a simdgroup lane gathers them. Passing the
/// wrong one produces a plausible, wrong result rather than an error, so it
/// selects the kernel rather than being a hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4MlxLayout {
    /// Plain row-major nibbles.
    RowMajor,
    /// Interleaved for 4-bit lane gather (`_i4` entry points).
    Interleaved4,
}

/// Output rows a single simdgroup-cooperative threadgroup covers.
///
/// `SIMD_SG_PER_TG (2) * SIMD_ROWS (4)` in the kernel sources. The threadgroup
/// is those two simdgroups, so 64 threads.
const SIMD_ROWS_PER_TG: usize = 8;
const SIMD_TPTG: usize = 64;

/// Threadgroups a simdgroup-cooperative GEMV needs for `rows` outputs.
///
/// Public because the fused K∥V and Q∥K∥V entry points partition one grid
/// across several matrices, and the caller has to be able to reason about where
/// the boundaries fall.
pub fn simd_gemv_threadgroups(rows: u32) -> usize {
    (rows as usize).div_ceil(SIMD_ROWS_PER_TG)
}

/// Which one-thread-per-row MLX Q4 GEMV kernel to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4MlxRowVariant {
    /// `gemv_q4_mlx` — tuned for very tall matrices (lm_head-class).
    Standard,
    /// `gemv_q4_mlx_wide` — same math, tuned for wide-and-short projections.
    Wide,
    /// `gemv_q4_mlx_tiled` — one threadgroup per row tile.
    Tiled,
}

impl Q4MlxRowVariant {
    fn entry(self) -> &'static str {
        match self {
            Self::Standard => "gemv_q4_mlx",
            Self::Wide => "gemv_q4_mlx_wide",
            Self::Tiled => "gemv_q4_mlx_tiled",
        }
    }
}

/// `y = W @ x` with MLX Q4 weights and an **f32** activation vector.
///
/// One thread per output row, staging `x` in threadgroup memory. For the
/// bandwidth-limited decode shape prefer [`gemv_q4_mlx_simd`], which reads a
/// bf16 activation stream and halves the traffic.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`, 7 = `group_size`.
pub fn gemv_q4_mlx(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    variant: Q4MlxRowVariant,
) -> Result<(), String> {
    gemv_q4_mlx_with_scalars(rt, bank, x, y, shape, variant, |bnd| {
        set_u32(bnd, shape.rows, 5);
        set_u32(bnd, shape.cols, 6);
        set_u32(bnd, shape.group_size, 7);
    })
}

/// [`gemv_q4_mlx`] with caller-supplied scalar binds.
pub fn gemv_q4_mlx_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    variant: Q4MlxRowVariant,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = variant.entry();
    bank.validate(&shape, entry)?;
    require::<f32>(x, shape.cols as usize, &format!("{entry} x"))?;
    require::<f32>(y, shape.rows as usize, &format!("{entry} y"))?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    // `Tiled` takes a different grid from `Standard` and `Wide`, and until
    // 2026-08-31 all three got the one-thread-per-row geometry below.
    //
    // `gemv_q4_mlx_tiled` indexes its output row by
    // `threadgroup_position_in_grid`, so it needs one threadgroup per row.
    // With `rows.div_ceil(128)` groups it wrote the first `rows / 128` rows and
    // left the rest of `y` untouched, returning no error. This is the same
    // defect `gemv_q4_tiled` had, in the sibling family — found by giving the
    // three variants one shared numeric test rather than testing `Standard`
    // alone.
    //
    // The tiled kernel also declares its scratch statically and does not cache
    // `x`, so the dynamic threadgroup allocation and its `cols` ceiling belong
    // to the other two.
    let (groups, tptg, tg_mem) = if variant == Q4MlxRowVariant::Tiled {
        (shape.rows as usize, GEMV_TILED_TPTG, None)
    } else {
        let bytes = (shape.cols as usize).saturating_mul(4);
        let limit = rt.max_threadgroup_memory();
        if bytes > limit {
            return Err(format!(
                "{entry}: caching x needs {bytes} bytes of threadgroup memory but this \
                 device allows {limit}; cols {} is too large for this kernel",
                shape.cols
            ));
        }
        let t = reduction_tptg(
            p.maxTotalThreadsPerThreadgroup(),
            GEMV_ROW_TPTG,
            GEMV_ROW_TPTG,
        )
        .min(shape.rows as usize)
        .max(1);
        ((shape.rows as usize).div_ceil(t), t, Some((0, bytes)))
    };
    dispatch_tg_1d(rt, &p, groups, tptg, tg_mem, |bnd| {
        bind_mlx_bank(bnd, &bank, 0);
        set_gpu_buf(bnd, x, 3);
        set_gpu_buf(bnd, y, 4);
        scalars(bnd);
    })
}

/// Bind an MLX bank's three consecutive slots starting at `base`.
///
/// Slot `base + 2` is the kernel's `biases_unused`; see [`Q4MlxBank`].
fn bind_mlx_bank(bnd: &mut Binder<'_>, bank: &Q4MlxBank<'_>, base: usize) {
    set_gpu_buf(bnd, bank.packed, base);
    set_gpu_buf(bnd, bank.scales_biases, base + 1);
    set_gpu_buf(bnd, bank.scales_biases, base + 2);
}

/// Row tile a blocked MLX Q4 GEMV threadgroup covers (`GEMV_BN`), the K-lanes
/// per row (`GEMV_LANES`), and the largest `x` slice it stages at once.
const GEMV_BN: usize = 16;
const GEMV_LANES: usize = 16;
const GEMV_X_TILE: usize = 4096;

/// `y = W @ x` with MLX Q4 weights, one threadgroup per 16-row block.
///
/// Each row gets 16 K-lanes that `simd_sum` their partial products, and `x` is
/// staged a tile at a time rather than whole — so unlike [`gemv_q4_mlx`] this
/// has no `cols` ceiling from threadgroup memory.
///
/// # The bank must be block-interleaved, not row-major
///
/// This is the one entry point whose [`Q4MlxBank`] is **not** laid out the way
/// every other one here expects, and the type cannot express the difference —
/// `Q4MlxBank` carries no layout tag, so passing a row-major bank compiles,
/// dispatches, and returns wrong numbers with no error.
///
/// The kernel indexes within a 16-row block: for block `b`, group `g` and row
/// `r` inside the block, it reads scale/bias at
/// `b * groups_per_row * 16 + g * 16 + r` and the matching nibbles at the same
/// index, rather than the row-major `row * groups_per_row + g`. Repack with
/// that mapping before calling.
///
/// The two layouts coincide only when `groups_per_row == 1`, which is why a
/// single-group matrix appears to work and anything wider silently does not.
/// Measured on a 64x256 matrix with `group_size` 64: 63 of 64 rows wrong with a
/// row-major bank, 0 of 64 once repacked.
/// `promoted_numeric.rs` carries a reference repacking.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`, 7 = `group_size`.
pub fn gemv_q4_mlx_blocked(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    gemv_q4_mlx_blocked_with_scalars(rt, bank, x, y, shape, |bnd| {
        set_u32(bnd, shape.rows, 5);
        set_u32(bnd, shape.cols, 6);
        set_u32(bnd, shape.group_size, 7);
    })
}

/// [`gemv_q4_mlx_blocked`] with caller-supplied scalar binds.
pub fn gemv_q4_mlx_blocked_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    bank.validate(&shape, "gemv_q4_mlx_blocked")?;
    require::<f32>(x, shape.cols as usize, "gemv_q4_mlx_blocked x")?;
    require::<f32>(y, shape.rows as usize, "gemv_q4_mlx_blocked y")?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline("gemv_q4_mlx_blocked")?;
    let groups = (shape.rows as usize).div_ceil(GEMV_BN);
    let tg_mem = (shape.cols as usize).min(GEMV_X_TILE) * 4;
    dispatch_tg_1d(
        rt,
        &p,
        groups,
        GEMV_BN * GEMV_LANES,
        Some((0, tg_mem)),
        |bnd| {
            bind_mlx_bank(bnd, &bank, 0);
            set_gpu_buf(bnd, x, 3);
            set_gpu_buf(bnd, y, 4);
            scalars(bnd);
        },
    )
}

/// `y = W @ x` with MLX Q4 weights and a **bf16** activation vector.
///
/// The simdgroup-cooperative decode path: four rows per simdgroup, two
/// simdgroups per threadgroup. `x` must already be bf16 — half the activation
/// traffic of [`gemv_q4_mlx`], which is what makes this the faster shape on a
/// bandwidth-bound decode.
///
/// Passing `resid` adds it elementwise (`gemv_q4_mlx_simd_add*`), folding a
/// residual connection into the same dispatch.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`, 7 = `group_size`.
/// Buffer 8 (`resid`) is bound here.
pub fn gemv_q4_mlx_simd(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    layout: Q4MlxLayout,
    resid: Option<&GpuBuffer>,
) -> Result<(), String> {
    gemv_q4_mlx_simd_with_scalars(rt, bank, x_bf16, y, shape, layout, resid, |bnd| {
        set_u32(bnd, shape.rows, 5);
        set_u32(bnd, shape.cols, 6);
        set_u32(bnd, shape.group_size, 7);
    })
}

/// [`gemv_q4_mlx_simd`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_simd_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    layout: Q4MlxLayout,
    resid: Option<&GpuBuffer>,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = match (resid.is_some(), layout) {
        (false, Q4MlxLayout::RowMajor) => "gemv_q4_mlx_simd",
        (false, Q4MlxLayout::Interleaved4) => "gemv_q4_mlx_simd_i4",
        (true, Q4MlxLayout::RowMajor) => "gemv_q4_mlx_simd_add",
        (true, Q4MlxLayout::Interleaved4) => "gemv_q4_mlx_simd_add_i4",
    };
    bank.validate(&shape, entry)?;
    require::<u16>(x_bf16, shape.cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(y, shape.rows as usize, &format!("{entry} y"))?;
    if let Some(r) = resid {
        require::<f32>(r, shape.rows as usize, &format!("{entry} resid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    let groups = simd_gemv_threadgroups(shape.rows);
    dispatch_tg_1d(rt, &p, groups, SIMD_TPTG, None, |bnd| {
        bind_mlx_bank(bnd, &bank, 0);
        set_gpu_buf(bnd, x_bf16, 3);
        set_gpu_buf(bnd, y, 4);
        scalars(bnd);
        if let Some(r) = resid {
            set_gpu_buf(bnd, r, 8);
        }
    })
}

/// Dispatch strategy for the fused gate/up GELU GEMV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUpDispatch {
    /// Simdgroup-cooperative, bf16 activations.
    Simd(Q4MlxLayout),
    /// Blocked 16-row tiles, f32 activations.
    Blocked,
}

/// `mid = gelu_pytorch_tanh(W_gate @ x) * (W_up @ x)` in one dispatch.
///
/// Both projections and the gating collapse into a single launch, so the
/// intermediate `gate` and `up` vectors are never written to device memory.
/// `mid_as_bf16` writes the result as bf16 to feed the down-projection GEMV
/// without a cast pass.
///
/// `x` is bf16 for [`GateUpDispatch::Simd`] and f32 for
/// [`GateUpDispatch::Blocked`], matching the kernels.
///
/// Scalar indices for `_with_scalars`: 8 = `rows`, 9 = `cols`,
/// 10 = `group_size`, 11 = `mid_as_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_gate_up_gelu(
    rt: &Arc<GpuRuntime>,
    gate: Q4MlxBank<'_>,
    up: Q4MlxBank<'_>,
    x: &GpuBuffer,
    mid: &GpuBuffer,
    shape: QuantShape,
    dispatch: GateUpDispatch,
    mid_as_bf16: bool,
) -> Result<(), String> {
    gemv_q4_mlx_gate_up_gelu_with_scalars(
        rt,
        gate,
        up,
        x,
        mid,
        shape,
        dispatch,
        mid_as_bf16,
        |bnd| {
            set_u32(bnd, shape.rows, 8);
            set_u32(bnd, shape.cols, 9);
            set_u32(bnd, shape.group_size, 10);
            set_u32(bnd, u32::from(mid_as_bf16), 11);
        },
    )
}

/// [`gemv_q4_mlx_gate_up_gelu`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_gate_up_gelu_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: Q4MlxBank<'_>,
    up: Q4MlxBank<'_>,
    x: &GpuBuffer,
    mid: &GpuBuffer,
    shape: QuantShape,
    dispatch: GateUpDispatch,
    mid_as_bf16: bool,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = match dispatch {
        GateUpDispatch::Simd(l) => match l {
            Q4MlxLayout::RowMajor => "gemv_q4_mlx_simd_gate_up_gelu",
            Q4MlxLayout::Interleaved4 => "gemv_q4_mlx_simd_gate_up_gelu_i4",
        },
        GateUpDispatch::Blocked => "gemv_q4_mlx_blocked_gate_up_gelu",
    };
    gate.validate(&shape, &format!("{entry} gate"))?;
    up.validate(&shape, &format!("{entry} up"))?;
    match dispatch {
        GateUpDispatch::Simd(_) => require::<u16>(x, shape.cols as usize, &format!("{entry} x"))?,
        GateUpDispatch::Blocked => require::<f32>(x, shape.cols as usize, &format!("{entry} x"))?,
    }
    if mid_as_bf16 {
        require::<u16>(mid, shape.rows as usize, &format!("{entry} mid (bf16)"))?;
    } else {
        require::<f32>(mid, shape.rows as usize, &format!("{entry} mid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    let (groups, tptg, tg_mem) = match dispatch {
        GateUpDispatch::Simd(_) => (simd_gemv_threadgroups(shape.rows), SIMD_TPTG, None),
        GateUpDispatch::Blocked => (
            (shape.rows as usize).div_ceil(GEMV_BN),
            GEMV_BN * GEMV_LANES,
            Some((0, (shape.cols as usize).min(GEMV_X_TILE) * 4)),
        ),
    };
    dispatch_tg_1d(rt, &p, groups, tptg, tg_mem, |bnd| {
        bind_mlx_bank(bnd, &gate, 0);
        bind_mlx_bank(bnd, &up, 3);
        set_gpu_buf(bnd, x, 6);
        set_gpu_buf(bnd, mid, 7);
        scalars(bnd);
    })
}

/// `k_out = W_k @ x` and `v_out = W_v @ x` in one dispatch.
///
/// The grid is partitioned: the first `simd_gemv_threadgroups(k_rows)` groups
/// compute K and the rest compute V, with the boundary passed to the kernel so
/// each group knows which matrix it owns. K and V must share `cols`,
/// `group_size` and row count.
///
/// Scalar indices for `_with_scalars`: 9 = `rows`, 10 = `cols`,
/// 11 = `group_size`, 12 = `tg_k` (the partition boundary).
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_kv(
    rt: &Arc<GpuRuntime>,
    k: Q4MlxBank<'_>,
    v: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
    shape: QuantShape,
    layout: Q4MlxLayout,
) -> Result<(), String> {
    let tg_k = simd_gemv_threadgroups(shape.rows) as u32;
    gemv_q4_mlx_kv_with_scalars(rt, k, v, x_bf16, k_out, v_out, shape, layout, |bnd| {
        set_u32(bnd, shape.rows, 9);
        set_u32(bnd, shape.cols, 10);
        set_u32(bnd, shape.group_size, 11);
        set_u32(bnd, tg_k, 12);
    })
}

/// [`gemv_q4_mlx_kv`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_kv_with_scalars(
    rt: &Arc<GpuRuntime>,
    k: Q4MlxBank<'_>,
    v: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
    shape: QuantShape,
    layout: Q4MlxLayout,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = match layout {
        Q4MlxLayout::RowMajor => "gemv_q4_mlx_simd_kv",
        Q4MlxLayout::Interleaved4 => "gemv_q4_mlx_simd_kv_i4",
    };
    k.validate(&shape, &format!("{entry} k"))?;
    v.validate(&shape, &format!("{entry} v"))?;
    require::<u16>(x_bf16, shape.cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(k_out, shape.rows as usize, &format!("{entry} k_out"))?;
    require::<f32>(v_out, shape.rows as usize, &format!("{entry} v_out"))?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    let per_matrix = simd_gemv_threadgroups(shape.rows);
    dispatch_tg_1d(rt, &p, per_matrix * 2, SIMD_TPTG, None, |bnd| {
        bind_mlx_bank(bnd, &k, 0);
        bind_mlx_bank(bnd, &v, 3);
        set_gpu_buf(bnd, x_bf16, 6);
        set_gpu_buf(bnd, k_out, 7);
        set_gpu_buf(bnd, v_out, 8);
        scalars(bnd);
    })
}

/// `q_out`, `k_out`, `v_out` from one shared activation in one dispatch.
///
/// As [`gemv_q4_mlx_kv`], with a three-way grid partition. Q may have a
/// different row count from K and V (grouped-query attention); all three share
/// `cols` and `group_size`.
///
/// These entry points take **two**-slot banks: unlike every other MLX kernel
/// they omit the unused `biases` slot, so the packed/scale pairs sit at
/// consecutive indices 0..6.
///
/// Scalar indices for `_with_scalars`: 10 = `rows_q`, 11 = `rows_kv`,
/// 12 = `cols`, 13 = `group_size`, 14 = `tg_q`, 15 = `tg_k`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_qkv(
    rt: &Arc<GpuRuntime>,
    q: Q4MlxBank<'_>,
    k: Q4MlxBank<'_>,
    v: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    out: QkvOutputs<'_>,
    rows_q: u32,
    rows_kv: u32,
    cols: u32,
    group_size: u32,
    layout: Q4MlxLayout,
) -> Result<(), String> {
    let tg_q = simd_gemv_threadgroups(rows_q) as u32;
    let tg_k = simd_gemv_threadgroups(rows_kv) as u32;
    gemv_q4_mlx_qkv_with_scalars(
        rt,
        q,
        k,
        v,
        x_bf16,
        out,
        rows_q,
        rows_kv,
        cols,
        group_size,
        layout,
        |bnd| {
            set_u32(bnd, rows_q, 10);
            set_u32(bnd, rows_kv, 11);
            set_u32(bnd, cols, 12);
            set_u32(bnd, group_size, 13);
            set_u32(bnd, tg_q, 14);
            set_u32(bnd, tg_k, 15);
        },
    )
}

/// The three destination buffers of [`gemv_q4_mlx_qkv`].
#[derive(Clone, Copy)]
pub struct QkvOutputs<'a> {
    /// Query projection output, `rows_q` f32.
    pub q_out: &'a GpuBuffer,
    /// Key projection output, `rows_kv` f32.
    pub k_out: &'a GpuBuffer,
    /// Value projection output, `rows_kv` f32.
    pub v_out: &'a GpuBuffer,
}

/// [`gemv_q4_mlx_qkv`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_mlx_qkv_with_scalars(
    rt: &Arc<GpuRuntime>,
    q: Q4MlxBank<'_>,
    k: Q4MlxBank<'_>,
    v: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    out: QkvOutputs<'_>,
    rows_q: u32,
    rows_kv: u32,
    cols: u32,
    group_size: u32,
    layout: Q4MlxLayout,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = match layout {
        Q4MlxLayout::RowMajor => "gemv_q4_mlx_simd_qkv",
        Q4MlxLayout::Interleaved4 => "gemv_q4_mlx_simd_qkv_i4",
    };
    let q_shape = QuantShape {
        rows: rows_q,
        cols,
        group_size,
    };
    let kv_shape = QuantShape {
        rows: rows_kv,
        cols,
        group_size,
    };
    q.validate(&q_shape, &format!("{entry} q"))?;
    k.validate(&kv_shape, &format!("{entry} k"))?;
    v.validate(&kv_shape, &format!("{entry} v"))?;
    require::<u16>(x_bf16, cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(out.q_out, rows_q as usize, &format!("{entry} q_out"))?;
    require::<f32>(out.k_out, rows_kv as usize, &format!("{entry} k_out"))?;
    require::<f32>(out.v_out, rows_kv as usize, &format!("{entry} v_out"))?;
    if rows_q == 0 && rows_kv == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    let groups = simd_gemv_threadgroups(rows_q) + 2 * simd_gemv_threadgroups(rows_kv);
    dispatch_tg_1d(rt, &p, groups, SIMD_TPTG, None, |bnd| {
        // Two slots per bank here, not three — these kernels drop the unused
        // biases buffer that the rest of the MLX family carries.
        set_gpu_buf(bnd, q.packed, 0);
        set_gpu_buf(bnd, q.scales_biases, 1);
        set_gpu_buf(bnd, k.packed, 2);
        set_gpu_buf(bnd, k.scales_biases, 3);
        set_gpu_buf(bnd, v.packed, 4);
        set_gpu_buf(bnd, v.scales_biases, 5);
        set_gpu_buf(bnd, x_bf16, 6);
        set_gpu_buf(bnd, out.q_out, 7);
        set_gpu_buf(bnd, out.k_out, 8);
        set_gpu_buf(bnd, out.v_out, 9);
        scalars(bnd);
    })
}

// --------------------------------------------------------- MLX Q4 GEMM ---

/// Largest `M` the MLX Q4 GEMM kernels handle, from `constant uint GEMM_MAX_M`.
///
/// The kernel takes `min(M, GEMM_MAX_M)` internally, so a larger `M` is
/// silently truncated — rows past the eighth simply are not computed, and the
/// destination keeps whatever it held. [`gemm_q4_mlx`] refuses instead.
pub const GEMM_Q4_MLX_MAX_M: u32 = 8;

/// `Y[M, rows] = X[M, cols] @ W[rows, cols]^T` with MLX Q4 weights.
///
/// The small-batch companion to [`gemv_q4_mlx_simd`]: same simdgroup structure,
/// with each simdgroup carrying up to [`GEMM_Q4_MLX_MAX_M`] activation rows
/// through one weight read. Passing `resid` adds it elementwise.
///
/// Scalar indices for `_with_scalars`: 5 = `rows`, 6 = `cols`,
/// 7 = `group_size`, 8 = `M`. Buffer 9 (`resid`) is bound here.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_mlx(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    m: u32,
    layout: Q4MlxLayout,
    resid: Option<&GpuBuffer>,
) -> Result<(), String> {
    gemm_q4_mlx_with_scalars(rt, bank, x_bf16, y, shape, m, layout, resid, |bnd| {
        set_u32(bnd, shape.rows, 5);
        set_u32(bnd, shape.cols, 6);
        set_u32(bnd, shape.group_size, 7);
        set_u32(bnd, m, 8);
    })
}

/// [`gemm_q4_mlx`] with caller-supplied scalar binds.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_mlx_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    m: u32,
    layout: Q4MlxLayout,
    resid: Option<&GpuBuffer>,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = match (resid.is_some(), layout) {
        (false, Q4MlxLayout::RowMajor) => "gemm_q4_mlx_simd",
        (false, Q4MlxLayout::Interleaved4) => "gemm_q4_mlx_simd_i4",
        (true, Q4MlxLayout::RowMajor) => "gemm_q4_mlx_simd_add",
        (true, Q4MlxLayout::Interleaved4) => "gemm_q4_mlx_simd_add_i4",
    };
    if m == 0 {
        return Ok(());
    }
    if m > GEMM_Q4_MLX_MAX_M {
        return Err(format!(
            "{entry}: M = {m} exceeds GEMM_MAX_M = {GEMM_Q4_MLX_MAX_M}; the kernel \
             clamps to that internally, so rows {GEMM_Q4_MLX_MAX_M}..{m} would be \
             left unwritten rather than computed"
        ));
    }
    bank.validate(&shape, entry)?;
    let out_elems = elems(m, shape.rows, entry)?;
    require::<u16>(
        x_bf16,
        elems(m, shape.cols, entry)?,
        &format!("{entry} x_bf16"),
    )?;
    require::<f32>(y, out_elems, &format!("{entry} y"))?;
    if let Some(r) = resid {
        require::<f32>(r, out_elems, &format!("{entry} resid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    let groups = simd_gemv_threadgroups(shape.rows);
    dispatch_tg_1d(rt, &p, groups, SIMD_TPTG, None, |bnd| {
        bind_mlx_bank(bnd, &bank, 0);
        set_gpu_buf(bnd, x_bf16, 3);
        set_gpu_buf(bnd, y, 4);
        scalars(bnd);
        if let Some(r) = resid {
            set_gpu_buf(bnd, r, 9);
        }
    })
}

// ------------------------------------------------------------ Reductions ---

/// Threadgroup scratch depth in `reduce.metal`. Every lane writes its own slot,
/// so a launch may never exceed this.
const REDUCE_MAX_TG: usize = 1024;

/// Threads per group for a row reduction: a power of two, capped by the
/// kernel's scratch depth and the pipeline's own limit, and never more than
/// the row is wide.
fn reduce_tptg(max_threads: usize, cols: usize) -> usize {
    reduction_tptg(max_threads, cols.next_power_of_two().max(1), REDUCE_MAX_TG)
}

/// `out[r, :] = softmax(x[r, :])` over `rows` rows of `cols` each.
///
/// Numerically stable: the row maximum is subtracted before exponentiating.
/// Without that a single logit above about 88 overflows `exp` in f32 and takes
/// the whole row to NaN, which is an ordinary input for attention scores rather
/// than a pathological one.
///
/// A row whose exponentials sum to zero — every entry `-inf`, which is what a
/// fully masked attention row looks like — yields a uniform distribution rather
/// than NaN.
///
/// `x` and `out` may be the same buffer.
///
/// Scalar index for `_with_scalars`: 2 = `cols`.
pub fn softmax_rows_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    row_reduce(rt, "softmax_rows_f32", x, out, rows, cols, cols)
}

/// `out[r] = sum(x[r, :])`. `out` holds one f32 per row.
pub fn row_sum_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    row_reduce(rt, "row_sum_f32", x, out, rows, cols, 1)
}

/// `out[r] = max(x[r, :])`. `out` holds one f32 per row.
pub fn row_max_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    row_reduce(rt, "row_max_f32", x, out, rows, cols, 1)
}

/// Shared dispatch for the row reductions: one threadgroup per row.
///
/// `out_per_row` is how many f32 each row writes — `cols` for softmax, 1 for a
/// scalar reduction — and is what `out`'s extent is checked against.
fn row_reduce(
    rt: &Arc<GpuRuntime>,
    entry: &str,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
    out_per_row: u32,
) -> Result<(), String> {
    if cols == 0 {
        return Err(format!("{entry}: cols must be non-zero"));
    }
    require::<f32>(x, elems(rows, cols, entry)?, &format!("{entry} x"))?;
    require::<f32>(
        out,
        elems(rows, out_per_row, entry)?,
        &format!("{entry} out"),
    )?;
    if rows == 0 {
        return Ok(());
    }
    let p = rt.pipeline(entry)?;
    let tptg = reduce_tptg(p.maxTotalThreadsPerThreadgroup(), cols as usize);
    dispatch_tg_1d(rt, &p, rows as usize, tptg, None, |bnd| {
        set_gpu_buf(bnd, x, 0);
        set_gpu_buf(bnd, out, 1);
        set_u32(bnd, cols, 2);
    })
}

// ------------------------------------------------- Quantized int8 GEMM ---

/// `C[m,n] = (A_i8 @ B_i8) * a_scale * b_scale[n]`, dequantized in registers.
///
/// MPP TensorOps accumulates `int8 x int8` into `int32` natively. The products
/// are exact in int32 for any `k` below 2^17 at full int8 range, so unlike the
/// f32 paths this accumulation carries **no rounding at all** — the only
/// approximation is the caller's quantization of the operands.
///
/// `b_scale` is per output column, which is where a per-channel weight scale
/// lives, and is optional. The dequantization is applied between the
/// accumulate and the store, while the values are still in registers; applied
/// afterwards it would be an extra full read and write of `C`.
///
/// # Why this is not behind `quant-prep`
///
/// It does not need a host-side `MTLTensor`. The kernel builds its tensors from
/// raw device pointers, so `MTLTensorDataType::Int8` being bound or not is
/// irrelevant here — that binding is for descriptors this path never creates.
/// The same is true of Int4: `metal::int4b_format` exists in the shading
/// language and TensorOps accepts it, and what is actually missing is the
/// tensor constructor for a sub-byte element type, not an objc2 binding.
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_dequant(
    rt: &Arc<GpuRuntime>,
    a: &GpuBuffer,
    b: &GpuBuffer,
    c: &GpuBuffer,
    m: u32,
    n: u32,
    k: u32,
    a_scale: f32,
    b_scale: Option<&GpuBuffer>,
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("gemm_i8_dequant: m, n and k must be non-zero".into());
    }
    if !a_scale.is_finite() {
        return Err(format!(
            "gemm_i8_dequant: a_scale must be finite, got {a_scale}"
        ));
    }
    // int32 accumulation is exact only while the running sum fits. Full-range
    // int8 products reach 127*127 = 16129, so k above 2^31/16129 could
    // overflow; refusing well below that keeps the "no rounding" claim true
    // rather than nearly true.
    const MAX_K_EXACT: u32 = 131_072;
    if k > MAX_K_EXACT {
        return Err(format!(
            "gemm_i8_dequant: k = {k} exceeds {MAX_K_EXACT}, past which an int32 \
             accumulator can overflow at full int8 range and the result wraps \
             silently rather than saturating"
        ));
    }
    require::<i8>(a, elems(m, k, "gemm_i8_dequant")?, "gemm_i8_dequant a")?;
    require::<i8>(b, elems(k, n, "gemm_i8_dequant")?, "gemm_i8_dequant b")?;
    require::<f32>(c, elems(m, n, "gemm_i8_dequant")?, "gemm_i8_dequant c")?;
    if let Some(sc) = b_scale {
        require::<f32>(sc, n as usize, "gemm_i8_dequant b_scale")?;
    }

    let p = rt.pipeline("matmul2d_tensorops_i8_f32")?;
    // Geometry must match the I8_DEQUANT_KERNEL instantiation.
    const SM: usize = 128;
    const SN: usize = 64;
    let tiles_n = (n as usize).div_ceil(SN);
    let tiles_m = (m as usize).div_ceil(SM);
    let groups = tiles_n * tiles_m;
    let tptg = 32 * 4; // NSG = 4 simdgroups
    let has_scale = u32::from(b_scale.is_some());
    // Buffer 8 is declared, so it must be bound even when unread: Metal faults
    // on a declared-but-unbound buffer. `has_scale` gates the dereference.
    let scale_buf = b_scale.unwrap_or(c);
    dispatch_tg_1d(rt, &p, groups, tptg, None, |bnd| {
        set_gpu_buf(bnd, a, 0);
        set_gpu_buf(bnd, b, 1);
        set_gpu_buf(bnd, c, 2);
        set_u32(bnd, m, 3);
        set_u32(bnd, n, 4);
        set_u32(bnd, k, 5);
        set_u32(bnd, tiles_n as u32, 6);
        set_u32(bnd, tiles_m as u32, 7);
        set_gpu_buf(bnd, scale_buf, 8);
        set_f32(bnd, a_scale, 9);
        set_u32(bnd, has_scale, 10);
    })
}
