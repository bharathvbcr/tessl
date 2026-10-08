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
//! # Safety of the `_with_scalars` seam
//!
//! Each kernel has two entry points. The plain one binds its scalar operands
//! through the runtime's const arena and is what most callers want. The
//! `unsafe` `_with_scalars` one takes a closure that binds them itself, for callers
//! that need *stable* GPU addresses across encodes — const-arena offsets move
//! from one encode to the next, which breaks an Indirect Command Buffer that
//! froze its binds. `gemma-metal` drives these from a persistent scalar pool
//! for exactly that reason.
//!
//! The closure receives the full [`Binder`], so the type system cannot stop it
//! from replacing a validated data-buffer bind, changing the pipeline, or
//! binding a device scalar whose contents disagree with the host values used
//! for validation and dispatch. A violation can turn a safe-looking call into
//! an out-of-bounds GPU access. Every `_with_scalars` caller must therefore:
//!
//! - bind every scalar index documented by that function exactly once, with the
//!   documented ABI type and the exact value passed to the host wrapper;
//! - bind no other index and perform no pipeline, dispatch, ICB, or resource
//!   mutation through the supplied binder; and
//! - ensure every buffer used as stable scalar storage belongs to `rt`, covers
//!   the bound value at its offset, and remains alive and resident until the
//!   encoded work completes (or through every replay of an ICB that froze it).
//!
//! If stable scalar storage is mutable, these requirements apply to its value
//! at every execution, not only when the closure first binds its GPU address.
//! Prefer the plain safe entry points unless stable addresses are required.
//!
//! Calling a raw variant without acknowledging this contract does not compile:
//!
//! ```compile_fail
//! use std::sync::Arc;
//! use tessl::{nn, GpuBuffer, GpuRuntime};
//!
//! fn invalid_safe_call(rt: &Arc<GpuRuntime>, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer) {
//!     nn::rms_norm_f32_with_scalars(rt, x, w, out, 1, 1, |_| {});
//! }
//! ```

use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::dispatch::{
    dispatch_1d, dispatch_2d_tg, set_f32, set_gpu_buf, set_u32, validate_dispatch_geometry, Binder,
};
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

/// Elements a buffer can hold at `size_of::<T>()` bytes each.
fn capacity_of<T>(buf: &GpuBuffer) -> usize {
    buf.nbytes() / std::mem::size_of::<T>()
}

/// `rows * dim`, or an error naming the overflow rather than wrapping.
fn elems(rows: u32, dim: u32, what: &str) -> Result<usize, String> {
    elems_product(&[rows, dim], what)
}

/// Product of device `u32` dimensions, widened before every multiplication.
/// Writing `a * b` first and only then passing it to [`elems`] wrapped or
/// panicked before the checked host arithmetic ever saw the value.
fn elems_product(dims: &[u32], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| format!("{what}: dimension product overflows usize"))
}

/// Tessl's 1D NN shaders expose `thread_position_in_grid` as Metal `uint`.
/// Reject a larger host extent before pipeline lookup or binder creation rather
/// than relying on a late dispatch-layer refusal.
fn require_1d_indexable(n: usize, what: &str) -> Result<(), String> {
    if n > u32::MAX as usize {
        return Err(format!("{what}: 1D extent {n} exceeds Metal uint indexing"));
    }
    Ok(())
}

/// MPP tensor extents and the I8 kernel's tile coordinates are signed `int`.
/// Keep every flattened matrix below that boundary before any buffer lookup or
/// dispatch so a large valid `u32` shape cannot truncate into a negative index.
fn require_i32_indexable(n: usize, what: &str) -> Result<(), String> {
    if n > i32::MAX as usize {
        return Err(format!(
            "{what}: extent {n} exceeds signed 32-bit kernel indexing"
        ));
    }
    Ok(())
}

fn i8_gemm_extents(m: u32, n: u32, k: u32) -> Result<(usize, usize, usize), String> {
    let a = elems(m, k, "gemm_i8_dequant A")?;
    let b = elems(k, n, "gemm_i8_dequant B")?;
    let c = elems(m, n, "gemm_i8_dequant C")?;
    for (name, extent) in [("A", a), ("B", b), ("C", c)] {
        require_i32_indexable(extent, &format!("gemm_i8_dequant {name}"))?;
    }
    Ok((a, b, c))
}

/// Reject a buffer that was allocated by another runtime.
fn require_runtime(rt: &GpuRuntime, buf: &GpuBuffer, what: &str) -> Result<(), String> {
    if !buf.belongs_to(rt) {
        return Err(format!("{what}: buffer belongs to another runtime"));
    }
    Ok(())
}

fn checked_buffer_byte_range(
    nbytes: usize,
    byte_offset: usize,
    byte_len: usize,
    what: &str,
) -> Result<(), String> {
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| format!("{what}: byte range overflows usize"))?;
    if end > nbytes {
        return Err(format!(
            "{what}: byte range {byte_offset}..{end} exceeds buffer capacity {nbytes}"
        ));
    }
    Ok(())
}

/// Validate an owned raw buffer range without allocating, staging scalars, or
/// opening an encoder.
///
/// Host adapters that issue several dependent dispatches use this preflight to
/// prove every operand before the first dispatch can mutate device state. A
/// zero-length range is valid when its offset is at most `buf.nbytes()`.
pub fn validate_buffer_byte_range(
    rt: &GpuRuntime,
    buf: &GpuBuffer,
    byte_offset: usize,
    byte_len: usize,
    what: &str,
) -> Result<(), String> {
    require_runtime(rt, buf, what)?;
    checked_buffer_byte_range(buf.nbytes(), byte_offset, byte_len, what)
}

/// Reject a buffer that cannot hold `need` elements of `T`.
fn require_capacity<T>(buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    let have = capacity_of::<T>(buf);
    if have < need {
        return Err(format!(
            "{what}: buffer holds {have} elements, kernel reads/writes {need}"
        ));
    }
    Ok(())
}

/// Validate both allocation ownership and element capacity at every raw-buffer
/// API boundary. Keeping the two checks in one seam prevents new kernels from
/// remembering the extent check while forgetting runtime identity (or vice
/// versa).
fn require<T>(rt: &GpuRuntime, buf: &GpuBuffer, need: usize, what: &str) -> Result<(), String> {
    require_runtime(rt, buf, what)?;
    require_capacity::<T>(buf, need, what)
}

/// Reject unordered device writes through aliased Metal allocations.
///
/// `writes` contains every buffer this dispatch may mutate and `reads`
/// contains only read-only operands. A kernel with a proven in-place contract
/// deliberately omits that one logical input from `reads`; all other operands
/// still flow through this single check.
fn require_disjoint_writes(
    entry: &str,
    writes: &[(&str, &GpuBuffer)],
    reads: &[(&str, &GpuBuffer)],
) -> Result<(), String> {
    for (i, &(lhs_name, lhs)) in writes.iter().enumerate() {
        for &(rhs_name, rhs) in &writes[i + 1..] {
            if lhs.aliases(rhs) {
                return Err(format!(
                    "{entry}: writable buffers {lhs_name} and {rhs_name} overlap"
                ));
            }
        }
        for &(read_name, read) in reads {
            if lhs.aliases(read) {
                return Err(format!(
                    "{entry}: writable buffer {lhs_name} overlaps read-only buffer {read_name}"
                ));
            }
        }
    }
    Ok(())
}

/// Maximum complete KV positions jointly backed by `k` and `v`.
///
/// Attention's live `Tkv` is a device-side `u32`, so a host wrapper cannot
/// validate its current value without synchronizing. Every attention kernel
/// instead receives this independently derived upper bound and clamps the live
/// value before indexing either buffer. A partial trailing position is ignored.
/// A live value of zero is a valid empty-history state and overwrites every live
/// output row with zeros on all attention paths.
/// K and V must imply the same fixed per-batch capacity, and that capacity must
/// fit the device `u32`; otherwise there is no single stride the kernels can
/// safely use for both buffers.
pub fn attn_kv_capacity(
    k: &GpuBuffer,
    v: &GpuBuffer,
    batch: u32,
    heads_kv: u32,
    head_dim: u32,
) -> Result<u32, String> {
    attn_kv_capacity_for(k, v, batch, heads_kv, head_dim, "attention")
}

fn attn_kv_capacity_for(
    k: &GpuBuffer,
    v: &GpuBuffer,
    batch: u32,
    heads_kv: u32,
    head_dim: u32,
    what: &str,
) -> Result<u32, String> {
    if heads_kv == 0 {
        return Err(format!("{what}: heads_kv must be non-zero"));
    }
    if head_dim == 0 {
        return Err(format!("{what}: head_dim must be non-zero"));
    }
    let per_position = elems_product(&[batch, heads_kv, head_dim], what)?;
    if per_position == 0 {
        return Ok(0);
    }
    attn_kv_capacity_from_elements(
        capacity_of::<f32>(k),
        capacity_of::<f32>(v),
        per_position,
        what,
    )
}

fn attn_kv_capacity_from_elements(
    k_elements: usize,
    v_elements: usize,
    per_position: usize,
    what: &str,
) -> Result<u32, String> {
    debug_assert_ne!(per_position, 0);
    let k_positions = k_elements / per_position;
    let v_positions = v_elements / per_position;
    if k_positions != v_positions {
        return Err(format!(
            "{what}: K and V imply different fixed capacities ({k_positions} vs \
             {v_positions} positions)"
        ));
    }
    u32::try_from(k_positions)
        .map_err(|_| format!("{what}: KV capacity {k_positions} exceeds the device u32 range"))
}

fn validate_rms_scalars(dim: u32, eps: f32, what: &str) -> Result<(), String> {
    if dim == 0 {
        return Err(format!("{what}: dim must be non-zero"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(format!("{what}: eps must be finite and positive"));
    }
    Ok(())
}

// ---------------------------------------------------------------- RMSNorm ---

/// `out[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * weight[:]`, f32 out.
/// `dim` and `eps` must both be positive, and `eps` must be finite.
/// `out` may alias `x`: each row is fully reduced before its lanes rewrite
/// their own elements. `out` must not alias the shared `weight`, which other
/// row threadgroups can still be reading.
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
    validate_rms_scalars(dim, eps, "rms_norm_f32")?;
    // SAFETY: the closure binds only the documented scalar slots, using the
    // exact host values this call validates; the runtime owns their const-arena
    // storage through execution.
    unsafe {
        rms_norm_f32_with_scalars(rt, x, weight, out, rows, dim, |bnd| {
            set_u32(bnd, rows, 3);
            set_u32(bnd, dim, 4);
            set_f32(bnd, eps, 5);
        })
    }
}

/// [`rms_norm_f32`] with caller-supplied scalar binds. See the module docs.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn rms_norm_f32_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_f32")?;
    require::<f32>(rt, x, n, "rms_norm_f32 x")?;
    require::<f32>(rt, weight, dim as usize, "rms_norm_f32 weight")?;
    require::<f32>(rt, out, n, "rms_norm_f32 out")?;
    if rows == 0 {
        return Ok(());
    }
    // `out == x` is intentionally supported; see the public contract above.
    require_disjoint_writes("rms_norm_f32", &[("out", out)], &[("weight", weight)])?;
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
/// `dim` and `eps` have the same validated domain as [`rms_norm_f32`].
/// Its bf16 output must not alias either f32 input: the two-byte stores overlap
/// different four-byte input elements that other lanes or rows may still read.
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
    validate_rms_scalars(dim, eps, "rms_norm_bf16")?;
    // SAFETY: the closure binds only the documented scalar slots, using the
    // exact host values this call validates; the runtime owns their const-arena
    // storage through execution.
    unsafe {
        rms_norm_bf16_with_scalars(rt, x, weight, out, rows, dim, |bnd| {
            set_u32(bnd, rows, 3);
            set_u32(bnd, dim, 4);
            set_f32(bnd, eps, 5);
        })
    }
}

/// [`rms_norm_bf16`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn rms_norm_bf16_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_bf16")?;
    require::<f32>(rt, x, n, "rms_norm_bf16 x")?;
    require::<f32>(rt, weight, dim as usize, "rms_norm_bf16 weight")?;
    // bf16 output: two bytes per element, not four.
    require::<u16>(rt, out, n, "rms_norm_bf16 out")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "rms_norm_bf16",
        &[("out", out)],
        &[("x", x), ("weight", weight)],
    )?;
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
/// is the plain residual add. `dim`, `eps`, and `layer_scale` are validated as
/// finite parameters before any work is encoded; `dim` and `eps` must be
/// positive.
///
/// `resid` may alias `x`: the row reduction finishes before each lane reads
/// and rewrites its own element. It must not alias the shared `weight`, which
/// other row threadgroups can still be reading.
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
    validate_rms_scalars(dim, eps, "rms_norm_residual_add_f32")?;
    if !layer_scale.is_finite() {
        return Err("rms_norm_residual_add_f32: layer_scale must be finite".into());
    }
    // SAFETY: the closure binds only the documented scalar slots, using the
    // exact host values this call validates; the runtime owns their const-arena
    // storage through execution.
    unsafe {
        rms_norm_residual_add_f32_with_scalars(rt, x, weight, resid, rows, dim, |bnd| {
            set_u32(bnd, rows, 3);
            set_u32(bnd, dim, 4);
            set_f32(bnd, eps, 5);
            set_f32(bnd, layer_scale, 6);
        })
    }
}

/// [`rms_norm_residual_add_f32`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn rms_norm_residual_add_f32_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    resid: &GpuBuffer,
    rows: u32,
    dim: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n = elems(rows, dim, "rms_norm_residual_add_f32")?;
    require::<f32>(rt, x, n, "rms_norm_residual_add_f32 x")?;
    require::<f32>(rt, weight, dim as usize, "rms_norm_residual_add_f32 weight")?;
    require::<f32>(rt, resid, n, "rms_norm_residual_add_f32 resid")?;
    if rows == 0 {
        return Ok(());
    }
    // `resid == x` is intentional and folds the residual into one allocation.
    require_disjoint_writes(
        "rms_norm_residual_add_f32",
        &[("resid", resid)],
        &[("weight", weight)],
    )?;
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
/// `out` may alias `gate`, `up`, or both: each thread loads only its own two
/// operands before writing that same element.
///
/// Scalar index for `_with_scalars`: 3 = `n` (u32).
pub fn mlp_silu(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slot 3 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe { mlp_silu_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3)) }
}

/// [`mlp_silu`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn mlp_silu_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(rt, gate, n_us, "mlp_silu gate")?;
    require::<f32>(rt, up, n_us, "mlp_silu up")?;
    require::<f32>(rt, out, n_us, "mlp_silu out")?;
    if n == 0 {
        return Ok(());
    }
    let p = rt.pipeline("mlp_silu")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, gate, 0);
        set_gpu_buf(bnd, up, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// `out[i] = gelu_pytorch_tanh(gate[i]) * up[i]`.
/// `out` may alias `gate`, `up`, or both: each thread loads only its own two
/// operands before writing that same element.
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
    // SAFETY: the closure binds only documented slot 3 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe { mlp_gelu_tanh_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3)) }
}

/// [`mlp_gelu_tanh`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn mlp_gelu_tanh_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(rt, gate, n_us, "mlp_gelu_tanh gate")?;
    require::<f32>(rt, up, n_us, "mlp_gelu_tanh up")?;
    require::<f32>(rt, out, n_us, "mlp_gelu_tanh out")?;
    if n == 0 {
        return Ok(());
    }
    let p = rt.pipeline("mlp_gelu_tanh")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, gate, 0);
        set_gpu_buf(bnd, up, 1);
        set_gpu_buf(bnd, out, 2);
        scalars(bnd);
    })
}

/// [`mlp_gelu_tanh`] writing bf16, to feed a bf16 down-projection GEMV.
/// The bf16 output must not alias either f32 input: a two-byte store can
/// overwrite a different four-byte operand that another thread has yet to
/// read.
///
/// Scalar index for `_with_scalars`: 3 = `n`.
pub fn mlp_gelu_tanh_bf16(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slot 3 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe { mlp_gelu_tanh_bf16_with_scalars(rt, gate, up, out, n, |bnd| set_u32(bnd, n, 3)) }
}

/// [`mlp_gelu_tanh_bf16`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn mlp_gelu_tanh_bf16_with_scalars(
    rt: &Arc<GpuRuntime>,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(rt, gate, n_us, "mlp_gelu_tanh_bf16 gate")?;
    require::<f32>(rt, up, n_us, "mlp_gelu_tanh_bf16 up")?;
    require::<u16>(rt, out, n_us, "mlp_gelu_tanh_bf16 out")?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "mlp_gelu_tanh_bf16",
        &[("out", out)],
        &[("gate", gate), ("up", up)],
    )?;
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
    // SAFETY: the closure binds only the documented scalar slots to this
    // call's values; the runtime owns their const-arena storage through execution.
    unsafe {
        scale_f32_inplace_with_scalars(rt, x, n, |bnd| {
            set_f32(bnd, scale, 1);
            set_u32(bnd, n, 2);
        })
    }
}

/// [`scale_f32_inplace`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn scale_f32_inplace_with_scalars(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    require::<f32>(rt, x, n as usize, "scale_f32_inplace x")?;
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
/// `y` must not alias the weights, quantization tables, or `x`; output rows are
/// written while other threadgroups can still be reading those shared inputs.
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
    // SAFETY: the closure binds only the documented scalar slots to the shape
    // values validated below; the runtime owns their const-arena storage.
    unsafe {
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
}

/// [`gemv_q8`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_q8_with_scalars(
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
    require::<i8>(rt, packed, weights, "gemv_q8 packed")?;
    require::<f32>(rt, scales, groups, "gemv_q8 scales")?;
    require::<f32>(rt, zeros, groups, "gemv_q8 zeros")?;
    require::<f32>(rt, x, cols as usize, "gemv_q8 x")?;
    require::<f32>(rt, y, rows as usize, "gemv_q8 y")?;
    if rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "gemv_q8",
        &[("y", y)],
        &[
            ("packed", packed),
            ("scales", scales),
            ("zeros", zeros),
            ("x", x),
        ],
    )?;
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
/// `dst_capacity` is the fixed logical capacity of `dst`, in f32 elements. It
/// must cover `n` and be backed by the allocation. The kernel checks the live
/// device offset against this bound with widened arithmetic; an offset at or
/// beyond the end, an offset-plus-length crossing the end, and arithmetic
/// wraparound all make the complete store a no-op rather than an OOB write.
///
/// Scalar indices for `_with_scalars`: 2 = `n`, 4 = the callback's validated
/// `dst_capacity`. Buffer 3 (`dst_offset`) is bound here in both paths — it is
/// a device buffer, not a scalar.
/// `dst` must not alias `src` or `dst_offset`: the device-controlled offset can
/// select an overlapping region that the host cannot prove race-free.
pub fn kv_store_timestep(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    dst_capacity: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slots 2 and 4 to this call's
    // validated `n` and capacity; the runtime owns the const-arena storage.
    unsafe {
        kv_store_timestep_with_scalars(
            rt,
            src,
            dst,
            dst_offset,
            n,
            dst_capacity,
            |bnd, capacity| {
                set_u32(bnd, n, 2);
                set_u32(bnd, capacity, 4);
            },
        )
    }
}

/// [`kv_store_timestep`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn kv_store_timestep_with_scalars(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    dst_capacity: u32,
    scalars: impl FnOnce(&mut Binder<'_>, u32),
) -> Result<(), String> {
    require::<f32>(rt, src, n as usize, "kv_store_timestep src")?;
    require::<u32>(rt, dst_offset, 1, "kv_store_timestep dst_offset")?;
    if dst_capacity < n {
        return Err(format!(
            "kv_store_timestep: dst_capacity {dst_capacity} is smaller than n {n}"
        ));
    }
    require::<f32>(
        rt,
        dst,
        dst_capacity as usize,
        "kv_store_timestep dst capacity",
    )?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "kv_store_timestep",
        &[("dst", dst)],
        &[("src", src), ("dst_offset", dst_offset)],
    )?;
    let p = rt.pipeline("kv_store_timestep")?;
    dispatch_1d(rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        scalars(bnd, dst_capacity);
        set_gpu_buf(bnd, dst_offset, 3);
    })
}

/// [`kv_store_timestep`] for K and V in one dispatch.
///
/// Both destinations share one explicit logical capacity. Scalar indices for
/// `_with_scalars`: 4 = `n`, 6 = the callback's validated `dst_capacity`.
/// Buffer 5 is `dst_offset`.
/// The two destinations must be distinct and neither may alias either source
/// or the device-controlled offset.
#[allow(clippy::too_many_arguments)]
pub fn kv_store_timestep_pair(
    rt: &Arc<GpuRuntime>,
    src_k: &GpuBuffer,
    src_v: &GpuBuffer,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    dst_capacity: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slots 4 and 6 to this call's
    // validated `n` and capacity; the runtime owns the const-arena storage.
    unsafe {
        kv_store_timestep_pair_with_scalars(
            rt,
            src_k,
            src_v,
            dst_k,
            dst_v,
            dst_offset,
            n,
            dst_capacity,
            |bnd, capacity| {
                set_u32(bnd, n, 4);
                set_u32(bnd, capacity, 6);
            },
        )
    }
}

/// [`kv_store_timestep_pair`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_store_timestep_pair_with_scalars(
    rt: &Arc<GpuRuntime>,
    src_k: &GpuBuffer,
    src_v: &GpuBuffer,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    dst_offset: &GpuBuffer,
    n: u32,
    dst_capacity: u32,
    scalars: impl FnOnce(&mut Binder<'_>, u32),
) -> Result<(), String> {
    let n_us = n as usize;
    require::<f32>(rt, src_k, n_us, "kv_store_timestep_pair src_k")?;
    require::<f32>(rt, src_v, n_us, "kv_store_timestep_pair src_v")?;
    require::<u32>(rt, dst_offset, 1, "kv_store_timestep_pair dst_offset")?;
    if dst_capacity < n {
        return Err(format!(
            "kv_store_timestep_pair: dst_capacity {dst_capacity} is smaller than n {n}"
        ));
    }
    require::<f32>(
        rt,
        dst_k,
        dst_capacity as usize,
        "kv_store_timestep_pair dst_k capacity",
    )?;
    require::<f32>(
        rt,
        dst_v,
        dst_capacity as usize,
        "kv_store_timestep_pair dst_v capacity",
    )?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "kv_store_timestep_pair",
        &[("dst_k", dst_k), ("dst_v", dst_v)],
        &[
            ("src_k", src_k),
            ("src_v", src_v),
            ("dst_offset", dst_offset),
        ],
    )?;
    let p = rt.pipeline("kv_store_timestep_pair")?;
    dispatch_1d(rt, &p, n_us, |bnd| {
        set_gpu_buf(bnd, src_k, 0);
        set_gpu_buf(bnd, src_v, 1);
        set_gpu_buf(bnd, dst_k, 2);
        set_gpu_buf(bnd, dst_v, 3);
        scalars(bnd, dst_capacity);
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
/// `dst` must be separate from the ring and its device metadata; an in-place
/// ring permutation has cross-thread read/write dependencies and is unordered.
pub fn kv_ring_densify(
    rt: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    filled: &GpuBuffer,
    start: &GpuBuffer,
    n_slot: u32,
    capacity: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only the documented scalar slots to the shape
    // values validated below; the runtime owns their const-arena storage.
    unsafe {
        kv_ring_densify_with_scalars(rt, src, dst, filled, start, n_slot, capacity, |bnd| {
            set_u32(bnd, n_slot, 2);
            set_u32(bnd, capacity, 3);
        })
    }
}

/// [`kv_ring_densify`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kv_ring_densify_with_scalars(
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
    require_1d_indexable(ring, "kv_ring_densify fixed grid")?;
    require::<f32>(rt, src, ring, "kv_ring_densify src")?;
    require::<f32>(rt, dst, ring, "kv_ring_densify dst")?;
    require::<u32>(rt, filled, 1, "kv_ring_densify filled")?;
    require::<u32>(rt, start, 1, "kv_ring_densify start")?;
    if ring == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "kv_ring_densify",
        &[("dst", dst)],
        &[("src", src), ("filled", filled), ("start", start)],
    )?;
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

    /// The Metal entry point this head dimension dispatches to.
    ///
    /// Exposed so a benchmark can report the kernel it actually ran rather
    /// than a second mapping of its own that could drift from this one.
    pub fn kernel(self) -> &'static str {
        self.entry()
    }

    /// Query-block rows per threadgroup, from the kernel's `constant uint BR`.
    fn br(self) -> usize {
        8
    }
}

/// The shader's declared default chunk size. Kept mirrored so
/// `tests/attention.rs` can catch a drift between the two files; the value the
/// router actually uses comes from [`decode_chunk_for`].
pub const DECODE_KV_CHUNK: usize = 256;

/// Keys per chunk for a decode dispatch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeChunk {
    C64,
    C128,
    C256,
}

impl DecodeChunk {
    pub fn keys(self) -> usize {
        match self {
            DecodeChunk::C64 => 64,
            DecodeChunk::C128 => 128,
            DecodeChunk::C256 => 256,
        }
    }

    pub fn parse(v: &str) -> Result<Self, String> {
        match v {
            "64" => Ok(DecodeChunk::C64),
            "128" => Ok(DecodeChunk::C128),
            "256" => Ok(DecodeChunk::C256),
            other => Err(format!(
                "decode chunk must be 64, 128 or 256; got {other:?}"
            )),
        }
    }
}

/// Which query heads share a threadgroup in the decode partial pass.
///
/// They walk the same K/V, so co-residency decides how many times a line is
/// pulled through the load path. A *policy* rather than a count, because the
/// useful widths are shape-derived: `Group` is `H/Hkv` simdgroups, `AllHeads`
/// is `H`.
///
/// Measured, `bench/attn_tune.py --knob decode-sgs --batched 32`, median ms:
///
/// | config | one | group | all |
/// |---|---|---|---|
/// | `swa128_decode_1k` | 0.037 | **0.035** | 0.058 |
/// | `swa128_decode_4k` | 0.046 | **0.041** | 0.062 |
/// | `swa128_decode_b8_4k` | 0.439 | **0.339** | 0.340 |
/// | `swa256_decode_4k` | 0.057 | **0.054** | 0.080 |
/// | `global512_decode_4k` | 0.193 | 0.170 | **0.168** |
/// | `global512_decode_4k_mha` | 0.610 | 0.585 | **0.578** |
///
/// `Group` beats one-head-per-threadgroup by **1.3–1.7x everywhere**, which is
/// far outside the ~3% run-to-run noise floor and is the finding here.
///
/// `AllHeads` is the marginal one. It makes a threadgroup's per-key read the
/// whole contiguous `[Hkv][D]` row instead of a strided slice, and at D=512
/// (where `H` is 8) it wins — but by 4% in one sweep and 1.2% in another, so
/// the honest reading is a tie that two independent sweeps broke the same way,
/// not a measured gain. At D=128 `H` is 32, so it asks for 1024-thread
/// threadgroups and loses **1.7x** to the occupancy that costs; that half is
/// unambiguous. Chosen per head dim for that reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeHeadBlock {
    /// One query head per threadgroup: the original dispatch.
    One,
    /// The `H/Hkv` heads that share a KV head.
    Group,
    /// Every head of a batch item.
    AllHeads,
}

impl DecodeHeadBlock {
    /// Simdgroups per threadgroup for this shape, or `None` if the width does
    /// not divide `H` or does not fit a threadgroup — the caller steps down.
    ///
    /// Public because it is the rule that keeps a threadgroup's head-block
    /// arithmetic exact: `grid.y` is decoded as `(batch, block)` with
    /// `H / sgs` blocks, so an `sgs` that does not divide `H` would address the
    /// wrong head rather than fail. Nothing but a reference check would see
    /// that, so the rule is pinned by a test.
    pub fn simdgroups(self, heads: usize, group: usize) -> Option<usize> {
        let n = match self {
            DecodeHeadBlock::One => 1,
            DecodeHeadBlock::Group => group,
            DecodeHeadBlock::AllHeads => heads,
        };
        ((1..=32).contains(&n) && heads % n == 0).then_some(n)
    }

    pub fn parse(v: &str) -> Result<Self, String> {
        match v {
            "one" => Ok(DecodeHeadBlock::One),
            "group" => Ok(DecodeHeadBlock::Group),
            "all" => Ok(DecodeHeadBlock::AllHeads),
            other => Err(format!(
                "decode head block must be one, group or all; got {other:?}"
            )),
        }
    }
}

pub const DECODE_HEAD_BLOCK_D128: DecodeHeadBlock = DecodeHeadBlock::Group;
pub const DECODE_HEAD_BLOCK_D256: DecodeHeadBlock = DecodeHeadBlock::Group;
pub const DECODE_HEAD_BLOCK_D512: DecodeHeadBlock = DecodeHeadBlock::AllHeads;

pub fn decode_head_block_for(d: u32) -> DecodeHeadBlock {
    match d {
        512 => DECODE_HEAD_BLOCK_D512,
        256 => DECODE_HEAD_BLOCK_D256,
        _ => DECODE_HEAD_BLOCK_D128,
    }
}

/// Threads per threadgroup in the decode reduce pass.
///
/// Swept with `bench/attn_tune.py --knob reduce-w --batched 32`; it is a
/// dispatch parameter, not a compiled constant, so the values are not kernels.
pub const DECODE_REDUCE_THREADS: usize = 256;

/// Chunk size chosen per head dimension.
///
/// Measured with `bench/attn_tune.py --knob decode --batched 32`, five
/// interleaved rounds (median ms). The `--batched 32` matters: swept at one
/// launch per submit, ~88% of every number is the host round trip, which
/// compresses the margins towards 1.0x and decides near-ties on noise.
///
/// | config | 64 | 128 | 256 |
/// |---/// |---/// |---/// |---|
/// | `swa128_decode_1k` | 0.035 | **0.031** | 0.033 |
/// | `swa128_decode_4k` | 0.054 | 0.040 | **0.037** |
/// | `swa128_decode_b8_4k` | 0.360 | 0.316 | **0.304** |
/// | `swa256_decode_4k` | 0.053 | **0.044** | 0.056 |
/// | `global512_decode_4k` | 0.183 | **0.154** | 0.160 |
///
/// Two of the three are clear and one is not. **D=256 is 128**, ahead of 64 by
/// 21% — a real margin. **D=512 is 128**, ahead of 256 by 4%, which is at the
/// edge of what this measurement resolves. **D=128 is 256 by 1.4% on the
/// geometric mean over its three configs (0.0719 against 0.0729)**, which is
/// inside the ~3% run-to-run noise floor and is therefore recorded as a tie
/// broken by measurement, not as a rule. Per config it splits: `decode_1k`
/// prefers 128, `decode_4k` and `decode_b8_4k` prefer 256.
///
/// Unlike the rows knob there is no clean rule here: the trade is grid
/// parallelism against the number of partials the reduce pass combines, and
/// where that balances depends on how many threadgroups `B*H` already supplies
/// and how much K/V reuse a threadgroup already has.
pub const DECODE_CHUNK_D128: DecodeChunk = DecodeChunk::C256;
pub const DECODE_CHUNK_D256: DecodeChunk = DecodeChunk::C128;
pub const DECODE_CHUNK_D512: DecodeChunk = DecodeChunk::C128;

pub fn decode_chunk_for(d: u32) -> DecodeChunk {
    match d {
        512 => DECODE_CHUNK_D512,
        256 => DECODE_CHUNK_D256,
        _ => DECODE_CHUNK_D128,
    }
}

/// Head dimensions the FlashDecoding path is compiled for.
fn decode_entries(d: u32, c: DecodeChunk, r: RowsLanes) -> Option<(String, String)> {
    if !matches!(d, 128 | 256 | 512) {
        return None;
    }
    Some((
        format!(
            "flash_attn_decode_partial_h{d}_c{}_r{}",
            c.keys(),
            r.width()
        ),
        format!("flash_attn_decode_reduce_h{d}_c{}", c.keys()),
    ))
}

/// Lanes per key in the decode partial pass, per head dimension.
///
/// Measured with `bench/attn_tune.py --knob decode-r --batched 32`, five
/// interleaved rounds (median ms):
///
/// | config | R=8 | R=16 | R=32 |
/// |---/// |---/// |---/// |---|
/// | `swa128_decode_1k` | **0.032** | 0.047 | 0.076 |
/// | `swa128_decode_4k` | **0.037** | 0.051 | 0.080 |
/// | `swa128_decode_b8_4k` | **0.304** | 0.312 | 0.342 |
/// | `swa256_decode_4k` | 0.102 | **0.044** | 0.058 |
/// | `global512_decode_4k` | 0.345 | 0.268 | **0.154** |
///
/// Kept separate from [`rows_lanes_for`] because the two kernels are bound by
/// different things. Prefill is ALU bound and lands cleanly on `D/R = 16`.
/// Decode *was* latency bound — 0.6% of ALU peak and 5% of bandwidth when these
/// values were first chosen — and is now bandwidth bound at 48-65% of the
/// memory roof and under 10% of the ALU roof, which is why the later rounds of
/// tuning stopped paying. D=256 and D=512 land on `D/R = 16`; D=128 does not,
/// and takes R=8.
///
/// D=128 was R=16 until this sweep was re-run kernel-only. At one launch per
/// submit the two were a 3% tie that R=16 won; with the ~88% dispatch share
/// removed, R=8 is ahead or level at every D=128 config.
///
/// Choosing wrong is expensive even where choosing right gains little: R=16 at
/// D=512 is **9.3x** slower than R=32.
pub const DECODE_LANES_D128: RowsLanes = RowsLanes::R8;
pub const DECODE_LANES_D256: RowsLanes = RowsLanes::R16;
pub const DECODE_LANES_D512: RowsLanes = RowsLanes::R32;

pub fn decode_lanes_for(d: u32) -> RowsLanes {
    match d {
        512 => DECODE_LANES_D512,
        256 => DECODE_LANES_D256,
        _ => DECODE_LANES_D128,
    }
}

/// Single-query attention, split over the KV sequence (FlashDecoding).
///
/// The general kernels tile over query rows, so at `Tq == 1` they run one live
/// lane in 32 over a grid of `B*H` threadgroups. This splits over KV instead:
/// `n_chunks x B*H` threadgroups, every lane live. `window == 0` selects the
/// global (causal) rule; anything else is the sliding window.
///
/// `Tkv` is a device value, so the host cannot size the grid to the live chunk
/// count and dispatches for the fixed `kv_capacity` instead. Both passes clamp
/// the live value to that capacity and derive the same live chunk count. Every
/// chunk the reduce pass visits was therefore written by this partial pass;
/// capacity-only chunks return without writing and are never reduced, so the
/// scratch needs no zero fill.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_decode(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    head_dim: u32,
    // Declared fixed capacity of both K and V. This is checked against their
    // logical sizes before it is used as either the grid bound or batch stride.
    kv_capacity: usize,
    out_bf16: bool,
) -> Result<(), String> {
    flash_attn_decode_with_chunk(
        rt,
        q,
        k,
        v,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        dims,
        head_dim,
        kv_capacity,
        decode_chunk_for(head_dim),
        decode_lanes_for(head_dim),
        None,
        None,
        out_bf16,
    )
}

/// [`flash_attn_decode`] with an explicit chunk size, for the tuning sweep.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_decode_with_chunk(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    head_dim: u32,
    kv_capacity: usize,
    chunk: DecodeChunk,
    lanes: RowsLanes,
    // Threads per threadgroup for the reduce pass; `None` takes
    // [`DECODE_REDUCE_THREADS`]. A dispatch parameter, so a sweep of it costs
    // no extra kernels.
    reduce_threads: Option<usize>,
    // Which query heads share a threadgroup in the partial pass; `None` takes
    // [`decode_head_block_for`]. A width that does not divide `H` or does not
    // fit a threadgroup steps down rather than mis-indexing.
    head_block: Option<DecodeHeadBlock>,
    out_bf16: bool,
) -> Result<(), String> {
    let (partial_entry, reduce_entry) =
        decode_entries(head_dim, chunk, lanes).ok_or_else(|| {
            format!("flash_attn_decode: head dim {head_dim} has no decode kernel (128, 256 or 512)")
        })?;
    if dims.tq != 1 {
        return Err(format!(
            "flash_attn_decode is the Tq == 1 path; got Tq = {}. Use flash_attn_swa \
             or flash_attn_global_h512 for prefill.",
            dims.tq
        ));
    }
    require_attn_runtime(rt, q, k, v, o, "flash_attn_decode")?;
    let actual_kv_capacity =
        validate_attn_storage_for(&dims, head_dim, q, k, v, o, "flash_attn_decode", out_bf16)?;
    validate_attn_live_scalar_aliases(
        &dims,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        "flash_attn_decode",
    )?;
    require::<u32>(rt, tkv, 1, "flash_attn_decode tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_decode q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_decode kv_pos_offset")?;
    let kv_capacity = u32::try_from(kv_capacity)
        .map_err(|_| "flash_attn_decode: kv_capacity exceeds the device u32 range")?;
    if kv_capacity == 0 {
        return Err("flash_attn_decode: kv_capacity must be at least 1".into());
    }

    let bh = elems_product(&[dims.batch, dims.heads], "flash_attn_decode B*H")?;
    if bh == 0 {
        return Ok(());
    }
    if kv_capacity != actual_kv_capacity {
        return Err(format!(
            "flash_attn_decode: declared kv_capacity {kv_capacity} does not match the fixed K/V \
             layout capacity {actual_kv_capacity}"
        ));
    }
    let chunks = (kv_capacity as usize).div_ceil(chunk.keys()).max(1);
    let stride = head_dim as usize + 2;
    let scratch_elems = bh
        .checked_mul(chunks)
        .and_then(|x| x.checked_mul(stride))
        .ok_or("flash_attn_decode: partial scratch size overflows")?;
    let scratch_bytes = scratch_elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("flash_attn_decode: partial scratch byte size overflows")?;
    let scratch = rt.alloc_buffer(scratch_bytes)?;
    // Deliberately not zeroed. The reduce pass reads chunks
    // `0 .. ceil(Tkv/KV_CHUNK)`, and the partial pass returns early only for
    // `chunk * KV_CHUNK >= Tkv` -- which by construction is exactly the chunks
    // outside that range. So every slot the reduce reads was written by this
    // dispatch, and zero-filling was a 66 KB host allocation and memcpy on
    // every decode step. `decode_is_immune_to_a_recycled_scratch` is what
    // holds that invariant: it alternates long and short histories through the
    // pooled buffer, so a stale partial would be exactly what it reads.

    // Query heads that walk the same K/V go in one threadgroup, so grid.y
    // enumerates (batch, head-block) and the threadgroup is `sgs` simdgroups
    // wide. [`DecodeHeadBlock`] carries the measurements behind the choice.
    // Metal caps a threadgroup at 1024 threads, and past that the kernel's
    // `sgs == 1` path is the original one-head-per-threadgroup dispatch.
    let heads = dims.heads.max(1) as usize;
    let group = (dims.heads / dims.heads_kv.max(1)).max(1) as usize;
    let want = head_block.unwrap_or_else(|| decode_head_block_for(head_dim));
    // Step down through the policies rather than rounding a count, so a shape
    // the requested width cannot serve lands on one that can — ultimately the
    // original one-head-per-threadgroup dispatch, which always divides.
    let partial_sgs = [want, DecodeHeadBlock::Group, DecodeHeadBlock::One]
        .into_iter()
        .find_map(|p| p.simdgroups(heads, group))
        .unwrap_or(1);
    let partial_y = (dims.batch as usize)
        .checked_mul(heads / partial_sgs)
        .ok_or("flash_attn_decode: partial grid height overflows")?;
    let p = rt.pipeline(&partial_entry)?;
    let r = rt.pipeline(&reduce_entry)?;
    // The reduce pass is a serial tail: one threadgroup per (batch, head), so
    // at B*H = 8 the whole GPU folds partials on 8 threadgroups. Its width is a
    // dispatch parameter rather than a compiled-in one -- the kernel strides
    // its output loop by `threads_per_threadgroup` -- so widening it costs no
    // extra kernel. Capped at D because a lane past the head dim does nothing,
    // and at the Metal maximum of 1024.
    let reduce_width = reduce_threads
        .unwrap_or(DECODE_REDUCE_THREADS)
        .min(head_dim as usize)
        .max(32);
    let partial_groups = mtl_size(chunks, partial_y, 1);
    let partial_threads = mtl_size(partial_sgs * 32, 1, 1);
    let reduce_groups = mtl_size(1, bh, 1);
    let reduce_threads = mtl_size(reduce_width, 1, 1);
    // Validate both commands before opening a binder. If the reduce geometry
    // is invalid, encoding only the producer would leave an incomplete op in
    // the active batch and poison the next caller's view of `scratch`.
    validate_dispatch_geometry(
        partial_groups,
        partial_threads,
        Some(p.maxTotalThreadsPerThreadgroup()),
    )?;
    validate_dispatch_geometry(
        reduce_groups,
        reduce_threads,
        Some(r.maxTotalThreadsPerThreadgroup()),
    )?;

    // Keep the producer and consumer in one binder scope. Besides avoiding a
    // second access/residency pass, this makes the scratch RAW edge explicit
    // in the one place that knows whether automatic dispatch barriers were
    // disabled for this encoder.
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, &scratch, 3);
        set_u32(bnd, dims.batch, 4);
        set_gpu_buf(bnd, tkv, 6);
        set_u32(bnd, dims.heads, 7);
        set_u32(bnd, dims.heads_kv, 8);
        set_u32(bnd, dims.window, 9);
        set_f32(bnd, dims.scale, 10);
        set_gpu_buf(bnd, q_pos_offset, 11);
        set_gpu_buf(bnd, kv_pos_offset, 12);
        set_u32(bnd, kv_capacity, 13);
        bnd.dispatch(partial_groups, partial_threads);
        if bnd.needs_explicit_barriers() {
            bnd.barrier();
        }

        bnd.set_pipeline(&r);
        set_gpu_buf(bnd, &scratch, 0);
        set_gpu_buf(bnd, o, 1);
        set_u32(bnd, dims.batch, 2);
        set_gpu_buf(bnd, tkv, 3);
        set_u32(bnd, dims.heads, 4);
        set_u32(bnd, u32::from(out_bf16), 5);
        set_u32(bnd, kv_capacity, 6);
        bnd.dispatch(reduce_groups, reduce_threads);
        Ok(())
    })
}

/// Simdgroups per threadgroup in the row-parallel kernels.
///
/// Every simdgroup in a threadgroup walks the same key range, so this is how
/// much K/V reuse one global read buys: a threadgroup's lines are read once
/// from L2 and served `SGT` times from L1. Rows per threadgroup are
/// `SGT * 32/R`, which is why it interacts with [`RowsLanes`] and is swept
/// against it rather than chosen alone. 32 simdgroups is 1024 threads, the
/// Metal maximum. Compiled into the kernel name; `tests/attention.rs` pins the
/// host and shader agreeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowsGroups {
    G8,
    G16,
    G32,
}

impl RowsGroups {
    pub fn count(self) -> usize {
        match self {
            RowsGroups::G8 => 8,
            RowsGroups::G16 => 16,
            RowsGroups::G32 => 32,
        }
    }

    pub fn parse(v: &str) -> Result<Self, String> {
        match v {
            "8" => Ok(RowsGroups::G8),
            "16" => Ok(RowsGroups::G16),
            "32" => Ok(RowsGroups::G32),
            other => Err(format!(
                "rows simdgroups-per-threadgroup must be 8, 16 or 32; got {other:?}"
            )),
        }
    }
}

/// Lanes per query row in the row-parallel kernels.
///
/// The reduction that turns per-lane partial dots into a score costs log2(R)
/// shuffle-and-add steps against 2*D/R fused multiply-adds, so narrowing R
/// trades reduction overhead for per-lane work and puts 32/R query rows in one
/// simdgroup. Which value wins is measured per head dimension, not assumed --
/// `bench_flash_attn` sweeps it with `BENCH_ATTN_ROWS_R`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowsLanes {
    R8,
    R16,
    R32,
}

impl RowsLanes {
    pub fn width(self) -> usize {
        match self {
            RowsLanes::R8 => 8,
            RowsLanes::R16 => 16,
            RowsLanes::R32 => 32,
        }
    }

    pub fn parse(v: &str) -> Result<Self, String> {
        match v {
            "8" => Ok(RowsLanes::R8),
            "16" => Ok(RowsLanes::R16),
            "32" => Ok(RowsLanes::R32),
            other => Err(format!(
                "rows lanes-per-row must be 8, 16 or 32; got {other:?}"
            )),
        }
    }
}

/// Lanes per row chosen for each head dimension.
///
/// Measured on an M5 Pro; see the tessl README. Kept as a table rather than a
/// formula because the winner is a cache and register-pressure outcome, not a
/// derivable one.
pub fn rows_lanes_for(d: u32) -> RowsLanes {
    match d {
        512 => ROWS_LANES_D512,
        256 => ROWS_LANES_D256,
        _ => ROWS_LANES_D128,
    }
}

/// Measured with `bench/attn_tune.py --knob rows`, five interleaved rounds
/// (median ms) -- generated from
/// `bench/results/attn_tune_rows_m5pro.json`, not transcribed:
///
/// | config | R=8 | R=16 | R=32 | winner |
/// |---|---|---|---|---|
/// | `swa128_prefill_512` | **1.096** | 1.283 | 1.672 | 8 |
/// | `swa128_prefill_2048` | **11.493** | 17.062 | 21.064 | 8 |
/// | `swa128_prefill_4096` | **27.291** | 44.191 | 52.045 | 8 |
/// | `swa256_prefill_2048` | 22.973 | **14.891** | 16.667 | 16 |
/// | `global512_prefill_1024` | 21.898 | 8.208 | **5.195** | 32 |
///
/// The winners are not arbitrary: all three land at **D/R = 16 dims per lane**.
/// Below that the log2(R) reduction steps dominate the 2*D/R multiply-adds;
/// above it `q_reg[D/R] + acc[D/R]` exceeds 32 floats per lane and the register
/// file spills — which is the cliff visible at D=256/R=8 and D=512/R=16, both
/// of which want 32 dims per lane.
pub const ROWS_LANES_D128: RowsLanes = RowsLanes::R8;
pub const ROWS_LANES_D256: RowsLanes = RowsLanes::R16;
pub const ROWS_LANES_D512: RowsLanes = RowsLanes::R32;

/// Simdgroups per threadgroup, per head dimension.
///
/// Swept with `bench/attn_tune.py --knob rows-g`. It was a single constant 8
/// for every head dim, and that is what left D=512 behind: rows per threadgroup
/// are `SGT * 32/R`, so at R=32 eight simdgroups gave 8 rows of reuse per K/V
/// line against 32 at D=128/R=8 — the same arithmetic per byte over four times
/// the L1 traffic. Median ms, 5 interleaved rounds:
///
/// | config | 8 | 16 | 32 |
/// |---|---|---|---|
/// | `swa128_prefill_512` | **1.125** | 1.197 | 1.282 |
/// | `swa128_prefill_2048` | **12.278** | 13.743 | 12.943 |
/// | `swa128_prefill_4096` | **29.006** | 35.975 | 31.021 |
/// | `swa256_prefill_2048` | 15.110 | 16.624 | **14.243** |
/// | `global512_prefill_1024` | 7.298 | 7.234 | **5.986** |
///
/// Prefill is ~100% kernel, so this is swept at one launch per submit: the
/// batched arm buys nothing here and costs 32x the wall clock.
pub const ROWS_GROUPS_D128: RowsGroups = RowsGroups::G8;
pub const ROWS_GROUPS_D256: RowsGroups = RowsGroups::G32;
pub const ROWS_GROUPS_D512: RowsGroups = RowsGroups::G32;

pub fn rows_groups_for(d: u32) -> RowsGroups {
    match d {
        512 => ROWS_GROUPS_D512,
        256 => ROWS_GROUPS_D256,
        _ => ROWS_GROUPS_D128,
    }
}

fn rows_entry(d: u32, r: RowsLanes, g: RowsGroups) -> Option<String> {
    if !matches!(d, 128 | 256 | 512) {
        return None;
    }
    Some(format!(
        "flash_attn_rows_h{d}_r{}_g{}",
        r.width(),
        g.count()
    ))
}

/// Row-parallel flash attention: one simdgroup per query row.
///
/// The tiled kernels put BR query rows in a 32-thread threadgroup and guard the
/// inner loops with `lid < BR`, so 8 lanes in 32 do the arithmetic. This gives
/// each simdgroup its own row and each lane its own slice of the head
/// dimension, which makes every lane live, removes the `Oacc` threadgroup array
/// and its barriers, and — because a simdgroup owns one row rather than a tile
/// — lets each row walk its exact key range instead of the union window over
/// the tile followed by masking inside it.
///
/// `window == 0` selects the global (causal) rule.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_rows(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    head_dim: u32,
    out_bf16: bool,
) -> Result<(), String> {
    flash_attn_rows_with_lanes(
        rt,
        q,
        k,
        v,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        dims,
        head_dim,
        rows_lanes_for(head_dim),
        rows_groups_for(head_dim),
        out_bf16,
    )
}

/// [`flash_attn_rows`] with an explicit lanes-per-row, for the tuning sweep.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_rows_with_lanes(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    head_dim: u32,
    lanes: RowsLanes,
    groups: RowsGroups,
    out_bf16: bool,
) -> Result<(), String> {
    let entry = rows_entry(head_dim, lanes, groups).ok_or_else(|| {
        format!("flash_attn_rows: head dim {head_dim} has no kernel (128, 256 or 512)")
    })?;
    require_attn_runtime(rt, q, k, v, o, "flash_attn_rows")?;
    let kv_capacity =
        validate_attn_storage_for(&dims, head_dim, q, k, v, o, "flash_attn_rows", out_bf16)?;
    validate_attn_live_scalar_aliases(
        &dims,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        "flash_attn_rows",
    )?;
    require::<u32>(rt, tkv, 1, "flash_attn_rows tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_rows q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_rows kv_pos_offset")?;

    // Rows per threadgroup is `SGT` simdgroups times 32/R rows each, so the
    // grid depends on both values the kernel was compiled for.
    let rows_per_tg = groups.count() * (32 / lanes.width());
    let groups_x = (dims.tq as usize).div_ceil(rows_per_tg);
    let groups_y = elems_product(&[dims.batch, dims.heads], "flash_attn_rows grid")?;
    let p = rt.pipeline(&entry)?;
    dispatch_2d_tg(rt, &p, groups_x, groups_y, groups.count() * 32, |bnd| {
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
    })
}

/// `TESSL_ATTN_TILED=1` forces the original tiled kernels.
///
/// Read once: this sits on the attention dispatch path, and the A/B harness
/// sets it per process rather than per call.
fn tiled_attn_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var_os("TESSL_ATTN_TILED").is_some())
}

/// Which kernel an attention dispatch routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKernel {
    /// FlashDecoding: `n_chunks x B*H` threadgroups plus a reduce pass.
    SplitKv,
    /// One simdgroup per query row: `Tq x B*H` threadgroups, single pass.
    Rows,
}

/// The routing rule, as a pure function of what it actually depends on.
///
/// Extracted so the rule can be pinned by a test rather than inferred from a
/// timing run. The rule it replaced -- `Tq == 1 && B*H < 128` -- was wrong for
/// two years' worth of batch sizes and nothing failed when it changed, because
/// both kernels compute the same thing and only the clock could tell them
/// apart.
///
/// `kv_capacity` is the shared fixed position capacity of K and V. The split
/// kernel sizes its grid from it, because the live `Tkv` is a device value, so
/// buffers too small to hold one position have no grid to launch and fall back
/// rather than dispatching an empty one.
pub fn attn_kernel_for(tq: u32, kv_capacity: usize) -> AttnKernel {
    if tq == 1 && kv_capacity > 0 {
        AttnKernel::SplitKv
    } else {
        AttnKernel::Rows
    }
}

/// Pick the attention kernel for a dispatch and run it.
#[allow(clippy::too_many_arguments)]
fn route_attn(
    rt: &Arc<GpuRuntime>,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    dims: AttnDims,
    head_dim: u32,
    out_bf16: bool,
) -> Result<(), String> {
    // Every `Tq == 1` dispatch takes the KV split. There used to be a
    // `B*H < 128` threshold here, on the evidence that at `B*H = 256` the split
    // lost 0.96 ms to the row kernel's 0.65 -- but that was measured at one
    // launch per submit, where ~85% of a decode call is the host round trip and
    // the split pays two submits to the row kernel's one. Measured kernel-only,
    // 32 launches per submit, the split wins at every batch the config set
    // reaches, and by more as the batch grows:
    //
    // | `B*H` | split | rows | |
    // |---|---|---|---|
    // | 32 | 0.033 | 0.327 | 9.9x |
    // | 256 | 0.344 | 0.494 | 1.4x |
    // | 1024 | 1.229 | 2.775 | 2.3x |
    // | 2048 | 2.476 | 5.127 | 2.1x |
    //
    // It also wins at one launch per submit once the decode kernel's own
    // constants were retuned on a kernel-only signal (1.18x at 32, 1.31x at
    // 256, 1.86x at 2048), so the threshold was not trading one protocol
    // against the other -- it was reading dispatch cost as kernel cost.
    // `Tkv` is a device value, so route from the complete positions jointly
    // backed by K and V. The selected kernel independently clamps to this same
    // bound before indexing either buffer.
    let capacity = attn_kv_capacity_for(k, v, dims.batch, dims.heads_kv, head_dim, "flash_attn")?;
    match attn_kernel_for(dims.tq, capacity as usize) {
        AttnKernel::SplitKv => flash_attn_decode(
            rt,
            q,
            k,
            v,
            o,
            tkv,
            q_pos_offset,
            kv_pos_offset,
            dims,
            head_dim,
            capacity as usize,
            out_bf16,
        ),
        AttnKernel::Rows => flash_attn_rows(
            rt,
            q,
            k,
            v,
            o,
            tkv,
            q_pos_offset,
            kv_pos_offset,
            dims,
            head_dim,
            out_bf16,
        ),
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
/// `Tkv` therefore is not knowable on the host without a stall. The wrapper
/// derives a capacity from the complete positions jointly backed by `k` and
/// `v`; the kernel clamps its live value to that capacity before any indexing.
///
/// Scalar indices for `_with_scalars`: 4 = `B`, 5 = `Tq`, 7 = `H`, 8 = `Hkv`,
/// 9 = `window`, 10 = `scale` (f32), plus the callback's validated capacity
/// argument at slot 13 for D=128 or slot 14 for D=256. Buffers 6, 11 and 12 are
/// bound here. For D=256 slot 13 is the kernel's `out_bf16` flag; this entry
/// point validates `o` as f32, so the wrapper binds that slot to 0 after the
/// callback whatever the callback did with it.
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
    // The tiled kernel is 4.5-6.4x slower on prefill -- 8 of its 32 lanes do
    // the arithmetic -- and 12-22x slower on decode. It stays reachable as
    // [`flash_attn_swa_tiled`], which is what the A/B benchmark and
    // `tests/attention.rs` compare the fast paths against.
    if !tiled_attn_forced() {
        return route_attn(
            rt,
            q,
            k,
            v,
            o,
            tkv,
            q_pos_offset,
            kv_pos_offset,
            dims,
            head_dim.dim(),
            false,
        );
    }
    flash_attn_swa_tiled(
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
    )
}

/// The original BR-tiled sliding-window kernel, unrouted.
///
/// Kept as the A/B baseline the fast paths are measured and tested against.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_swa_tiled(
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
    // SAFETY: the closure binds only the documented scalar slots to `dims`;
    // the runtime owns their const-arena storage through execution.
    unsafe {
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
            |bnd, kv_capacity| {
                set_u32(bnd, dims.batch, 4);
                set_u32(bnd, dims.tq, 5);
                set_u32(bnd, dims.heads, 7);
                set_u32(bnd, dims.heads_kv, 8);
                set_u32(bnd, dims.window, 9);
                set_f32(bnd, dims.scale, 10);
                match head_dim {
                    AttnHeadDim::D128 => set_u32(bnd, kv_capacity, 13),
                    AttnHeadDim::D256 => {
                        // D=256 shares its shader with Gemma's bf16-output variant.
                        set_u32(bnd, 0, 13);
                        set_u32(bnd, kv_capacity, 14);
                    }
                }
            },
        )
    }
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
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, including its validated
/// `kv_capacity` argument at the head-dimension-specific slot, mutate no other
/// binder state, and keep all stable scalar storage alive and resident for
/// every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_swa_with_scalars(
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
    scalars: impl FnOnce(&mut Binder<'_>, u32),
) -> Result<(), String> {
    let d = head_dim.dim();
    // The sliding-window kernels always write f32.
    require_attn_runtime(rt, q, k, v, o, "flash_attn_swa")?;
    let kv_capacity = validate_attn_storage_for(&dims, d, q, k, v, o, "flash_attn_swa", false)?;
    validate_attn_live_scalar_aliases(
        &dims,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        "flash_attn_swa",
    )?;
    require::<u32>(rt, tkv, 1, "flash_attn_swa tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_swa q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_swa kv_pos_offset")?;

    let p = rt.pipeline(head_dim.entry())?;
    let groups_x = (dims.tq as usize).div_ceil(head_dim.br());
    let groups_y = elems_product(&[dims.batch, dims.heads], "flash_attn_swa grid")?;
    dispatch_2d_tg(rt, &p, groups_x, groups_y, 32, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf(bnd, tkv, 6);
        set_gpu_buf(bnd, q_pos_offset, 11);
        set_gpu_buf(bnd, kv_pos_offset, 12);
        scalars(bnd, kv_capacity);
        if head_dim == AttnHeadDim::D256 {
            // The D=256 kernel reads `out_bf16` from slot 13, which the D=128
            // kernel uses for its capacity. `o` was validated as f32 above, so
            // a callback that left the slot unbound (a stale bind from an
            // earlier dispatch in this scope) or set it would have the kernel
            // write two-byte values into a four-byte output. This wrapper
            // owns the slot: bound to 0 after the callback, whatever it did.
            set_u32(bnd, 0, 13);
        }
    })
}

/// Global (non-windowed) flash attention at head dimension 512.
///
/// A separate entry point rather than a variant of [`flash_attn_swa`] because
/// the kernel's buffer layout differs: there is no `window`, and index 12
/// carries an `out_bf16` flag instead of a position offset.
///
/// Scalar indices for `_with_scalars`: 4 = `B`, 5 = `Tq`, 7 = `H`, 8 = `Hkv`,
/// 9 = `scale` (f32), 12 = `out_bf16`, and the callback's validated capacity
/// argument at slot 13. Buffers 6, 10 and 11 are bound here.
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
    if !tiled_attn_forced() {
        // This entry point's contract is that `window` is ignored: the tiled
        // h512 kernel has no window parameter at all. The routed kernels take
        // one and treat 0 as "global", so it is zeroed here rather than passed
        // through -- otherwise a caller who left a window set in `dims` would
        // silently get sliding-window attention from the global entry point.
        let global = AttnDims { window: 0, ..dims };
        return route_attn(
            rt,
            q,
            k,
            v,
            o,
            tkv,
            q_pos_offset,
            kv_pos_offset,
            global,
            512,
            out_bf16,
        );
    }
    flash_attn_global_h512_tiled(
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
    )
}

/// The original BR-tiled global kernel, unrouted. A/B baseline.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_global_h512_tiled(
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
    // SAFETY: the closure binds only the documented scalar slots to this
    // call's dimensions/output mode; the runtime owns their const-arena storage.
    unsafe {
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
            |bnd, kv_capacity| {
                set_u32(bnd, dims.batch, 4);
                set_u32(bnd, dims.tq, 5);
                set_u32(bnd, dims.heads, 7);
                set_u32(bnd, dims.heads_kv, 8);
                set_f32(bnd, dims.scale, 9);
                set_u32(bnd, u32::from(out_bf16), 12);
                set_u32(bnd, kv_capacity, 13);
            },
        )
    }
}

/// [`flash_attn_global_h512`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, including its validated
/// `kv_capacity` argument at slot 13, mutate no other binder state, and keep all
/// stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_global_h512_with_scalars(
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
    scalars: impl FnOnce(&mut Binder<'_>, u32),
) -> Result<(), String> {
    const D: u32 = 512;
    // `BR = 4` for this kernel, not 8 — see its `constant uint BR`.
    const BR: usize = 4;
    require_attn_runtime(rt, q, k, v, o, "flash_attn_global_h512")?;
    let kv_capacity =
        validate_attn_storage_for(&dims, D, q, k, v, o, "flash_attn_global_h512", out_bf16)?;
    validate_attn_live_scalar_aliases(
        &dims,
        o,
        tkv,
        q_pos_offset,
        kv_pos_offset,
        "flash_attn_global_h512",
    )?;
    require::<u32>(rt, tkv, 1, "flash_attn_global_h512 tkv")?;
    require::<u32>(rt, q_pos_offset, 1, "flash_attn_global_h512 q_pos_offset")?;
    require::<u32>(rt, kv_pos_offset, 1, "flash_attn_global_h512 kv_pos_offset")?;

    let p = rt.pipeline("flash_attn_global_h512")?;
    let groups_x = (dims.tq as usize).div_ceil(BR);
    let groups_y = elems_product(&[dims.batch, dims.heads], "flash_attn_global grid")?;
    dispatch_2d_tg(rt, &p, groups_x, groups_y, 32, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf(bnd, tkv, 6);
        set_gpu_buf(bnd, q_pos_offset, 10);
        set_gpu_buf(bnd, kv_pos_offset, 11);
        scalars(bnd, kv_capacity);
    })
}

/// Validate shared attention storage and return the safe fixed KV capacity.
///
/// This is exposed for host adapters that bind Tessl's Metal entry points
/// directly to stable scalar pools. It validates dimensions, Q/O extents,
/// output-vs-input aliasing, and derives the complete position count jointly
/// backed by K and V. The returned capacity is the batch stride required by the
/// kernels as well as the upper bound for their live device-side `Tkv`.
/// F32 output may alias Q: one-pass kernels retain a query row until its output
/// store, and decode finishes every Q read in the partial pass before reduce
/// writes. BF16 output may not alias Q because its packed addresses overlap
/// different f32 query rows. Output never aliases K or V.
pub fn validate_attn_storage(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    out_bf16: bool,
) -> Result<u32, String> {
    validate_attn_storage_for(dims, d, q, k, v, o, "attention", out_bf16)
}

#[allow(clippy::too_many_arguments)]
fn validate_attn_storage_for(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    what: &str,
    out_bf16: bool,
) -> Result<u32, String> {
    validate_attn_dims(dims, d, q, o, what, out_bf16)?;
    let capacity = attn_kv_capacity_for(k, v, dims.batch, dims.heads_kv, d, what)?;
    let has_work = dims.batch != 0 && dims.tq != 0 && dims.heads != 0;
    if !has_work {
        return Ok(capacity);
    }
    if out_bf16 && o.aliases(q) {
        return Err(format!(
            "{what}: bf16 output must not alias the f32 q input"
        ));
    }
    for (input, name) in [(k, "k"), (v, "v")] {
        if o.aliases(input) {
            return Err(format!(
                "{what}: output must not alias read-only {name} input"
            ));
        }
    }
    if capacity == 0 {
        return Err(format!(
            "{what}: K/V buffers do not jointly back one complete KV position"
        ));
    }
    Ok(capacity)
}

fn require_attn_runtime(
    rt: &GpuRuntime,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    what: &str,
) -> Result<(), String> {
    for (name, buffer) in [("q", q), ("k", k), ("v", v), ("o", o)] {
        require_runtime(rt, buffer, &format!("{what} {name}"))?;
    }
    Ok(())
}

fn validate_attn_live_scalar_aliases(
    dims: &AttnDims,
    o: &GpuBuffer,
    tkv: &GpuBuffer,
    q_pos_offset: &GpuBuffer,
    kv_pos_offset: &GpuBuffer,
    what: &str,
) -> Result<(), String> {
    validate_attn_output_scalar_aliases_for(
        dims,
        o,
        &[
            ("tkv", tkv),
            ("q_pos_offset", q_pos_offset),
            ("kv_pos_offset", kv_pos_offset),
        ],
        what,
    )
}

/// Reject output aliases with caller-owned scalar storage used by attention.
///
/// Direct adapters that bind Tessl kernels from stable scalar pools must call
/// this alongside [`validate_attn_storage`]. Read/read scalar aliases remain
/// valid. A zero-work shape is a clean no-op and accepts placeholder aliases.
pub fn validate_attn_output_scalar_aliases(
    dims: &AttnDims,
    o: &GpuBuffer,
    scalars: &[(&str, &GpuBuffer)],
) -> Result<(), String> {
    validate_attn_output_scalar_aliases_for(dims, o, scalars, "attention")
}

fn validate_attn_output_scalar_aliases_for(
    dims: &AttnDims,
    o: &GpuBuffer,
    scalars: &[(&str, &GpuBuffer)],
    what: &str,
) -> Result<(), String> {
    if dims.batch == 0 || dims.tq == 0 || dims.heads == 0 {
        return Ok(());
    }
    for &(name, scalar) in scalars {
        if o.aliases(scalar) {
            return Err(format!("{what}: output must not alias live {name} scalar"));
        }
    }
    Ok(())
}

fn validate_attn_dims(
    dims: &AttnDims,
    d: u32,
    q: &GpuBuffer,
    o: &GpuBuffer,
    what: &str,
    out_bf16: bool,
) -> Result<(), String> {
    if d == 0 {
        return Err(format!("{what}: head_dim must be non-zero"));
    }
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
    let grid_y = elems_product(&[dims.batch, dims.heads], what)?;
    if grid_y > u32::MAX as usize {
        return Err(format!(
            "{what}: B*H grid extent {grid_y} exceeds Metal uint indexing"
        ));
    }
    let n = elems_product(&[dims.batch, dims.tq, dims.heads, d], what)?;
    require_capacity::<f32>(q, n, &format!("{what} q"))?;
    if out_bf16 {
        // `out_bf16` exists to halve this buffer — the kernel writes `bfloat`
        // into it. Validating `o` as f32 regardless demanded twice the memory
        // the kernel touches, so a caller who sized it correctly for bf16 got
        // "buffer holds 2560 elements, kernel reads/writes 5120" and the
        // documented half-width scratch was unreachable.
        require_capacity::<u16>(o, n, &format!("{what} o (bf16)"))?;
    } else {
        require_capacity::<f32>(o, n, &format!("{what} o"))?;
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
            (Some(q), Some(kv)) => kv
                .checked_mul(2)
                .and_then(|kv_twice| q.checked_add(kv_twice))
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
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(format!("{what}: eps must be finite and positive"));
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
    /// Fixed logical capacity of each cache, in f32 elements.
    ///
    /// This may be smaller than the backing allocation when multiple logical
    /// regions share a slab. The host proves both buffers cover it; the shader
    /// refuses the complete fused operation when its live offset plus the K/V
    /// span would cross it.
    pub capacity: u32,
}

/// Fused per-head RMSNorm, QKV projection scaling, and rotary embedding.
///
/// `q`, `k` and `v` are read and written in place: they arrive holding the raw
/// projection output and leave normalized and rotated.
/// Every active in/out buffer must name a distinct allocation and must not
/// alias the shared weights or device offsets. The fused cache destinations
/// are active writes as well. When `q_only` is true, K/V and cache targets are
/// not dispatched and therefore are excluded from this alias contract; the
/// cache-store variant is rejected because its device-side cache guard runs
/// before the Q branch and could otherwise suppress an ostensibly Q-only op.
///
/// `pos_offset` is a constant for [`QkvRopeVariant::PosConst`] and a device
/// `u32` buffer for the other two. Exactly one must be supplied; passing the
/// wrong one for the variant is refused rather than silently ignored.
///
/// Scalar indices for `_with_scalars`: 6 = `T`, 7 = `Hq`, 8 = `Hkv`,
/// 9 = `D`, 10 = `rotary_dim`, 11 = `pos_offset` (`PosConst` only),
/// 12 = `theta` (f32), 13 = `eps` (f32), and for
/// [`QkvRopeVariant::PosBufferKvStore`] 17 = the callback's validated cache
/// capacity.
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
    // SAFETY: the closure binds only the scalar slots documented for the
    // selected variant, with the same values this call validates; the runtime
    // owns their const-arena storage through execution.
    unsafe {
        rms_qkv_rope_with_scalars(
            rt,
            variant,
            qkv,
            dims,
            pos_offset_buf,
            kv_store,
            q_only,
            |bnd, kv_capacity| {
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
                if let Some(capacity) = kv_capacity {
                    set_u32(bnd, capacity, 17);
                }
            },
        )
    }
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

/// Validate the complete [`rms_qkv_rope`] storage/variant contract without
/// looking up a pipeline, staging scalars, or encoding work.
///
/// Stable-scalar adapters should call this before reserving slots in their own
/// scalar pool. [`rms_qkv_rope_with_scalars`] calls the same function, so the
/// preflight cannot drift from the eventual dispatch boundary.
#[allow(clippy::too_many_arguments)]
pub fn validate_rms_qkv_rope(
    rt: &Arc<GpuRuntime>,
    variant: QkvRopeVariant,
    qkv: QkvBuffers<'_>,
    dims: QkvRopeDims,
    pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>,
    q_only: bool,
) -> Result<(), String> {
    dims.validate("rms_qkv_rope")?;
    if q_only && variant == QkvRopeVariant::PosBufferKvStore {
        return Err(
            "rms_qkv_rope: q_only cannot use PosBufferKvStore because its cache guard can suppress Q"
                .into(),
        );
    }

    // The variant selects the kernel, and the kernel decides which of these
    // operands exist. Accepting a mismatched pair and ignoring the extra one
    // is how a caller ends up reading a stale position for a whole session.
    match (variant, pos_offset_buf) {
        (QkvRopeVariant::PosConst, Some(_)) => {
            return Err("rms_qkv_rope: PosConst takes a constant offset, not a buffer".into())
        }
        (QkvRopeVariant::PosConst, None) => {}
        (_, None) => return Err("rms_qkv_rope: PosBuffer variants require pos_offset_buf".into()),
        (_, Some(b)) => require::<u32>(rt, b, 1, "rms_qkv_rope pos_offset_buf")?,
    }
    match (variant, kv_store.is_some()) {
        (QkvRopeVariant::PosBufferKvStore, false) => {
            return Err("rms_qkv_rope: PosBufferKvStore requires a KvStoreTarget".into())
        }
        (QkvRopeVariant::PosBufferKvStore, true) | (_, false) => {}
        (_, true) => return Err("rms_qkv_rope: only PosBufferKvStore writes the KV cache".into()),
    }

    let d = dims.head_dim as usize;
    let q_heads = elems_product(&[dims.t, dims.heads_q], "rms_qkv_rope q heads")?;
    let kv_heads = elems_product(&[dims.t, dims.heads_kv], "rms_qkv_rope kv heads")?;
    let q_elems = q_heads
        .checked_mul(d)
        .ok_or("rms_qkv_rope q extent overflows usize")?;
    let kv_elems = kv_heads
        .checked_mul(d)
        .ok_or("rms_qkv_rope kv extent overflows usize")?;
    require::<f32>(rt, qkv.q, q_elems, "rms_qkv_rope q")?;
    require::<f32>(rt, qkv.q_weight, d, "rms_qkv_rope q_weight")?;
    if !q_only {
        require::<f32>(rt, qkv.k, kv_elems, "rms_qkv_rope k")?;
        require::<f32>(rt, qkv.v, kv_elems, "rms_qkv_rope v")?;
        require::<f32>(rt, qkv.k_weight, d, "rms_qkv_rope k_weight")?;
        require::<f32>(rt, qkv.v_weight, d, "rms_qkv_rope v_weight")?;
    }
    if let Some(t) = &kv_store {
        if (t.capacity as usize) < kv_elems {
            return Err(format!(
                "rms_qkv_rope: cache capacity {} is smaller than the K/V span {kv_elems}",
                t.capacity
            ));
        }
        require::<f32>(
            rt,
            t.dst_k,
            t.capacity as usize,
            "rms_qkv_rope dst_k capacity",
        )?;
        require::<f32>(
            rt,
            t.dst_v,
            t.capacity as usize,
            "rms_qkv_rope dst_v capacity",
        )?;
        require::<u32>(rt, t.dst_offset, 1, "rms_qkv_rope kv_dst_offset")?;
    }

    let n = dims.head_count(q_only)?;
    require_1d_indexable(n, "rms_qkv_rope head grid")?;
    if n == 0 {
        return Ok(());
    }
    if q_only {
        match pos_offset_buf {
            Some(pos) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q)],
                &[("q_weight", qkv.q_weight), ("pos_offset_buf", pos)],
            )?,
            None => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q)],
                &[("q_weight", qkv.q_weight)],
            )?,
        }
    } else {
        match (pos_offset_buf, kv_store) {
            (None, None) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v)],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                ],
            )?,
            (Some(pos), None) => require_disjoint_writes(
                "rms_qkv_rope",
                &[("q", qkv.q), ("k", qkv.k), ("v", qkv.v)],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("pos_offset_buf", pos),
                ],
            )?,
            (Some(pos), Some(target)) => require_disjoint_writes(
                "rms_qkv_rope",
                &[
                    ("q", qkv.q),
                    ("k", qkv.k),
                    ("v", qkv.v),
                    ("dst_k", target.dst_k),
                    ("dst_v", target.dst_v),
                ],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("pos_offset_buf", pos),
                    ("dst_offset", target.dst_offset),
                ],
            )?,
            // Variant validation above makes a cache target without a device
            // position impossible, but keep this exhaustive and fail closed if
            // the variant contract changes.
            (None, Some(target)) => require_disjoint_writes(
                "rms_qkv_rope",
                &[
                    ("q", qkv.q),
                    ("k", qkv.k),
                    ("v", qkv.v),
                    ("dst_k", target.dst_k),
                    ("dst_v", target.dst_v),
                ],
                &[
                    ("q_weight", qkv.q_weight),
                    ("k_weight", qkv.k_weight),
                    ("v_weight", qkv.v_weight),
                    ("dst_offset", target.dst_offset),
                ],
            )?,
        }
    }
    Ok(())
}

/// [`rms_qkv_rope`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
/// For [`QkvRopeVariant::PosBufferKvStore`] only, the callback may additionally
/// replace slot 16 with an aligned, in-bounds byte-offset view of the validated
/// [`KvStoreTarget::dst_offset`] buffer. This supports a stable scalar pool;
/// the callback must preserve that buffer's runtime, lifetime, and `u32` ABI.
///
/// When `q_only` is set, slot 8 (`Hkv`) is rebound to zero after `scalars`
/// returns. The q-only grid is `T * Hq` head rows rounded up to a whole
/// threadgroup of rows (one simdgroup per row), and the kernel's bounds guard
/// counts `2 * T * Hkv` K/V rows after the Q ones, so a non-zero `Hkv` would
/// let the padding simdgroups run the K/V branches over Q storage. The
/// callback may still bind slot 8 as documented; the override is this
/// wrapper's responsibility, not the adapter's.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_qkv_rope_with_scalars(
    rt: &Arc<GpuRuntime>,
    variant: QkvRopeVariant,
    qkv: QkvBuffers<'_>,
    dims: QkvRopeDims,
    pos_offset_buf: Option<&GpuBuffer>,
    kv_store: Option<KvStoreTarget<'_>>,
    q_only: bool,
    scalars: impl FnOnce(&mut Binder<'_>, Option<u32>),
) -> Result<(), String> {
    validate_rms_qkv_rope(rt, variant, qkv, dims, pos_offset_buf, kv_store, q_only)?;
    let n = dims.head_count(q_only)?;
    if n == 0 {
        return Ok(());
    }
    let kv_capacity = kv_store.map(|target| target.capacity);
    let p = rt.pipeline(variant.entry())?;
    let (rows_per_tg, threads_per_tg) = rope_row_geometry(&p)?;
    dispatch_tg_1d(
        rt,
        &p,
        n.div_ceil(rows_per_tg),
        threads_per_tg,
        None,
        |bnd| {
            set_gpu_buf(bnd, qkv.q, 0);
            set_gpu_buf(bnd, qkv.q_weight, 3);
            if q_only {
                // These slots are part of the fixed argument-table ABI but the
                // q-only grid never reaches either K/V branch. Bind already
                // validated Q storage rather than touching caller-provided
                // inactive placeholders (which may intentionally be foreign or
                // empty under this contract).
                set_gpu_buf(bnd, qkv.q, 1);
                set_gpu_buf(bnd, qkv.q, 2);
                set_gpu_buf(bnd, qkv.q_weight, 4);
                set_gpu_buf(bnd, qkv.q_weight, 5);
            } else {
                set_gpu_buf(bnd, qkv.k, 1);
                set_gpu_buf(bnd, qkv.v, 2);
                set_gpu_buf(bnd, qkv.k_weight, 4);
                set_gpu_buf(bnd, qkv.v_weight, 5);
            }
            if let Some(b) = pos_offset_buf {
                set_gpu_buf(bnd, b, 11);
            }
            if let Some(t) = kv_store {
                set_gpu_buf(bnd, t.dst_k, 14);
                set_gpu_buf(bnd, t.dst_v, 15);
                set_gpu_buf(bnd, t.dst_offset, 16);
            }
            // The unsafe stable-scalar seam runs last so a fused-cache adapter may
            // replace slot 16 with a validated byte-offset view of `dst_offset`.
            scalars(bnd, kv_capacity);
            if q_only {
                // The grid is `T * Hq` rows rounded up to whole threadgroups and
                // the kernel's guard is `T*Hq + 2*T*Hkv`: with the real `Hkv`
                // bound, the padding simdgroups fall into the K/V branches, which
                // in this mode point at Q and re-normalize rows other simdgroups
                // own. Force `Hkv = 0` after the callback so the guard is exactly
                // the Q grid, whatever slot 8 held.
                set_u32(bnd, 0, 8);
            }
        },
    )
}

/// Lanes in the simdgroup the RoPE kernels give each head row.
const ROPE_SIMD_WIDTH: usize = 32;
/// Head rows per threadgroup: eight simdgroups, 256 threads, the usual
/// occupancy point for a row kernel with no threadgroup memory.
const ROPE_ROWS_PER_TG: usize = 8;

/// `(rows_per_tg, threads_per_tg)` for the one-simdgroup-per-row RoPE
/// kernels.
///
/// The kernel folds each row's sum of squares with `simd_sum` over exactly
/// 32 lanes and derives its row from `threads_per_threadgroup / 32`, so a
/// pipeline whose execution width is not 32 would reduce the wrong lanes and
/// address the wrong rows. That is refused here rather than dispatched. The
/// row count per group bends to the pipeline's thread limit so a register
/// -heavy compile still gets whole simdgroups.
fn rope_row_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(usize, usize), String> {
    let width = pipeline.threadExecutionWidth();
    if width != ROPE_SIMD_WIDTH {
        return Err(format!(
            "rms_qkv_rope: kernel assumes a {ROPE_SIMD_WIDTH}-lane simdgroup, \
             pipeline reports an execution width of {width}"
        ));
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if max_threads < ROPE_SIMD_WIDTH {
        return Err(format!(
            "rms_qkv_rope: pipeline allows {max_threads} threads per \
             threadgroup, fewer than one simdgroup"
        ));
    }
    let rows = (max_threads / ROPE_SIMD_WIDTH).min(ROPE_ROWS_PER_TG);
    Ok((rows, rows * ROPE_SIMD_WIDTH))
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
/// A non-positive or non-finite device value disables softcapping; this keeps a
/// mutable scalar fault from poisoning finite logits with zeros or NaNs.
/// `softcap` must not alias `logits`: every lane reads the shared scalar while
/// lane zero may rewrite the same allocation.
///
/// Scalar index for `_with_scalars`: 2 = `n`. Buffer 1 is `softcap`.
pub fn softcap_logits(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slot 2 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe { softcap_logits_with_scalars(rt, logits, softcap, n, |bnd| set_u32(bnd, n, 2)) }
}

/// [`softcap_logits`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn softcap_logits_with_scalars(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    require::<f32>(rt, logits, n as usize, "softcap_logits logits")?;
    require::<f32>(rt, softcap, 1, "softcap_logits softcap")?;
    if n == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "softcap_logits",
        &[("logits", logits)],
        &[("softcap", softcap)],
    )?;
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
/// indices propagate rather than being re-derived from partial offsets. The
/// kernel applies the cap only on that first pass (`has_idx_in == 0`): the
/// partials it writes are already capped and `tanh` is not idempotent, so a
/// pass that capped them again would shrink every value.
///
/// `out_idx` and `out_val` must each hold [`argmax_pass_groups`] elements.
/// They must be distinct from one another and from every input. Threadgroups
/// are unordered, so an output prefix cannot safely reuse an input allocation
/// even though each output is smaller than its source.
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
    // SAFETY: the closure binds only the documented scalar slots to `n` and
    // the presence of the validated optional index buffer; the runtime owns
    // their const-arena storage through execution.
    unsafe {
        argmax_f32_pass_with_scalars(rt, logits, out_idx, out_val, idx_in, softcap, n, |bnd| {
            set_u32(bnd, n, 3);
            set_u32(bnd, has, 5);
        })
    }
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
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn argmax_f32_pass_with_scalars(
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
    require::<f32>(rt, logits, n as usize, "argmax_f32_pass logits")?;
    require::<u32>(rt, out_idx, groups, "argmax_f32_pass out_idx")?;
    require::<f32>(rt, out_val, groups, "argmax_f32_pass out_val")?;
    require::<f32>(rt, softcap, 1, "argmax_f32_pass softcap")?;
    if let Some(b) = idx_in {
        require::<u32>(rt, b, n as usize, "argmax_f32_pass idx_in")?;
    }
    match idx_in {
        Some(indices) => require_disjoint_writes(
            "argmax_f32_pass",
            &[("out_idx", out_idx), ("out_val", out_val)],
            &[
                ("logits", logits),
                ("idx_in", indices),
                ("softcap", softcap),
            ],
        )?,
        None => require_disjoint_writes(
            "argmax_f32_pass",
            &[("out_idx", out_idx), ("out_val", out_val)],
            &[("logits", logits), ("softcap", softcap)],
        )?,
    }

    let p = rt.pipeline("argmax_f32")?;
    // Buffer 4 must be bound even on the first pass: the kernel reads the
    // binding unconditionally and gates on `has_idx_in`, so leaving the slot
    // empty is an unbound-buffer fault, not a no-op.
    // Reuse `logits` as the inert first-pass binding instead of allocating a
    // throwaway four-byte Metal buffer on every argmax. A Metal buffer is
    // untyped and `logits` has already been proved to contain at least `n`
    // four-byte words, so even an eager load through the `uint*` declaration
    // stays in bounds; `has_idx_in == 0` makes those bits semantically inert.
    let idx_buf = idx_in.unwrap_or(logits);
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
/// A non-positive or non-finite device softcap disables softcapping.
/// `out_token`, `logits`, and `softcap` must use distinct allocations. Sharing
/// `logits` with the scalar creates an unordered read/write; sharing either
/// destination corrupts one of the documented results.
///
/// Scalar index for `_with_scalars`: 3 = `n`. Buffer 2 is `softcap`.
pub fn softcap_sample(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slot 3 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe {
        softcap_sample_with_scalars(rt, logits, out_token, softcap, n, |bnd| set_u32(bnd, n, 3))
    }
}

/// [`softcap_sample`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn softcap_sample_with_scalars(
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
    require::<f32>(rt, logits, n as usize, "softcap_sample logits")?;
    require::<u32>(rt, out_token, 1, "softcap_sample out_token")?;
    require::<f32>(rt, softcap, 1, "softcap_sample softcap")?;
    require_disjoint_writes(
        "softcap_sample",
        &[("logits", logits), ("out_token", out_token)],
        &[("softcap", softcap)],
    )?;

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
/// A non-positive or non-finite device softcap disables softcapping.
/// `out_token` must not alias `logits` or `softcap`: the function promises to
/// leave both read-only inputs unchanged, while the token store overwrites its
/// first four bytes.
///
/// Scalar index for `_with_scalars`: 3 = `n`. Buffer 2 is `softcap`.
pub fn softcap_argmax_one_pass(
    rt: &Arc<GpuRuntime>,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: &GpuBuffer,
    n: u32,
) -> Result<(), String> {
    // SAFETY: the closure binds only documented slot 3 to this call's `n`;
    // the runtime owns the const-arena storage through execution.
    unsafe {
        softcap_argmax_one_pass_with_scalars(rt, logits, out_token, softcap, n, |bnd| {
            set_u32(bnd, n, 3)
        })
    }
}

/// [`softcap_argmax_one_pass`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn softcap_argmax_one_pass_with_scalars(
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
    require::<f32>(rt, logits, n as usize, "softcap_argmax_one_pass logits")?;
    require::<u32>(rt, out_token, 1, "softcap_argmax_one_pass out_token")?;
    require::<f32>(rt, softcap, 1, "softcap_argmax_one_pass softcap")?;
    require_disjoint_writes(
        "softcap_argmax_one_pass",
        &[("out_token", out_token)],
        &[("logits", logits), ("softcap", softcap)],
    )?;

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
    fn validate(&self, rt: &GpuRuntime, shape: &QuantShape, what: &str) -> Result<(), String> {
        shape.validate(what)?;
        if shape.group_size % 8 != 0 {
            return Err(format!(
                "{what}: group_size {} must be a multiple of 8; the kernels peel each \
                 group through 4-byte `uint` loads at `packed + row * cols / 2 + \
                 g * group_size / 2`, which is aligned for every row and group only \
                 when group_size is a multiple of 8",
                shape.group_size
            ));
        }
        let groups = shape.groups()?;
        let weights = elems(shape.rows, shape.cols, what)?;
        require::<u8>(
            rt,
            self.packed,
            weights.div_ceil(2),
            &format!("{what} packed"),
        )?;
        require::<f32>(rt, self.scales, groups, &format!("{what} scales"))?;
        require::<f32>(rt, self.zeros, groups, &format!("{what} zeros"))?;
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
    /// Packed nibbles: `rows * cols / 2` bytes row-major. Interleaved4 stores
    /// rows in tiles of four, so it holds `rows.div_ceil(4) * 4` rows' worth
    /// (see [`Q4MlxLayout`]); the kernels read the padding rows.
    pub packed: &'a GpuBuffer,
    /// Interleaved `bfloat2` scale/bias pairs, one per group, padded the same
    /// way for Interleaved4.
    pub scales_biases: &'a GpuBuffer,
}

impl Q4MlxBank<'_> {
    /// Validate the group domain and the stored extent for `layout`.
    ///
    /// The row kernels peel each group through 16-byte `uint4` loads, so a
    /// group's packed bytes (`group_size / 2`) must be a multiple of 16, and
    /// the simdgroup kernels walk K in 512-wide blocks stepping their scale
    /// pointer by `512 / group_size`, so the group must also divide 512. That
    /// is `{32, 64, 128, 256, 512}`; MLX itself quantizes with 32, 64 or 128.
    ///
    /// Interleaved4 stores rows in tiles of [`I4_TILE_ROWS`], and the kernels
    /// read every row of a tile (nibbles and scale pairs) before the
    /// `row < rows` guard discards the surplus lanes, so the bank has to exist
    /// at the tile-padded extent, not the row-major one.
    fn validate(
        &self,
        rt: &GpuRuntime,
        shape: &QuantShape,
        layout: Q4MlxLayout,
        what: &str,
    ) -> Result<(), String> {
        shape.validate(what)?;
        const SIMD_BLOCK: u32 = 512;
        if shape.group_size % 32 != 0 || SIMD_BLOCK % shape.group_size != 0 {
            return Err(format!(
                "{what}: group_size {} is outside the MLX kernel domain \
                 {{32, 64, 128, 256, 512}}; the row kernels peel each group through \
                 16-byte loads and the simdgroup kernels stride their scale pointer \
                 by 512 / group_size",
                shape.group_size
            ));
        }
        let (stored_rows, layout_note) = match layout {
            Q4MlxLayout::RowMajor => (shape.rows, String::new()),
            Q4MlxLayout::Interleaved4 => (
                shape.rows.div_ceil(I4_TILE_ROWS) * I4_TILE_ROWS,
                format!(
                    " (Interleaved4: {} rows padded to tiles of {I4_TILE_ROWS})",
                    shape.rows
                ),
            ),
        };
        let weights = elems(stored_rows, shape.cols, what)?;
        let groups = elems(stored_rows, shape.cols / shape.group_size, what)?;
        require::<u8>(
            rt,
            self.packed,
            weights.div_ceil(2),
            &format!("{what} packed{layout_note}"),
        )?;
        // One bfloat2 = two u16 = 4 bytes per group.
        require::<u32>(
            rt,
            self.scales_biases,
            groups,
            &format!("{what} scales_biases (bfloat2 per group){layout_note}"),
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
/// `y` must not alias the bank or `x`; output rows are written while other
/// lanes and threadgroups can still be consuming those shared inputs.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape values; the runtime owns the backing constant storage.
    unsafe {
        gemv_q4_with_scalars(rt, bank, x, y, shape, tiled, |bnd| {
            set_u32(bnd, shape.rows, 5);
            set_u32(bnd, shape.cols, 6);
            set_u32(bnd, shape.group_size, 7);
        })
    }
}

/// Validate every host-side invariant for [`gemv_q4`] without allocating
/// scalar storage or encoding work.
///
/// Stable-scalar adapters can call this before reserving their scalar arena,
/// so malformed public inputs cannot consume persistent slots before the
/// dispatch wrapper rejects them. A zero-row shape remains a no-op, but its
/// quantization shape and buffer ownership/capacity are still validated.
pub fn validate_gemv_q4(
    rt: &GpuRuntime,
    bank: Q4Bank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    tiled: bool,
) -> Result<(), String> {
    let entry = if tiled { "gemv_q4_tiled" } else { "gemv_q4" };
    bank.validate(rt, &shape, entry)?;
    require::<f32>(rt, x, shape.cols as usize, &format!("{entry} x"))?;
    require::<f32>(rt, y, shape.rows as usize, &format!("{entry} y"))?;
    if shape.rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        entry,
        &[("y", y)],
        &[
            ("packed", bank.packed),
            ("scales", bank.scales),
            ("zeros", bank.zeros),
            ("x", x),
        ],
    )?;
    Ok(())
}

/// [`gemv_q4`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn gemv_q4_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4Bank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    tiled: bool,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = if tiled { "gemv_q4_tiled" } else { "gemv_q4" };
    validate_gemv_q4(rt, bank, x, y, shape, tiled)?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline(entry)?;
    // The two kernels take *different* grids.
    //
    // `gemv_q4` is one simdgroup per `SIMD_ROWS_PER_TG / 2` output rows with
    // lanes striding K, the geometry `gemv_q8` and the MLX simd GEMVs use. It
    // was one thread per row with `x` staged in dynamic threadgroup memory
    // until 2026-09-05, which is where the `cols` ceiling this wrapper used to
    // enforce came from; there is no cache and no ceiling now.
    //
    // `gemv_q4_tiled` indexes its output row by `threadgroup_position_in_grid`
    // and returns when that exceeds `rows`, so it needs one threadgroup per row.
    // Until 2026-08-31 it was handed the row kernel's grid and wrote the first
    // `rows / 128` rows, leaving every other row of `y` untouched — no error, no
    // partial-write signal. The benchmark caught it: 3,077 GB/s is not a number
    // this machine can produce. It declares its scratch statically
    // (`threadgroup float partial[GEMV_TG]`).
    let (groups, tptg) = if tiled {
        (shape.rows as usize, GEMV_TILED_TPTG)
    } else {
        (simd_gemv_threadgroups(shape.rows), SIMD_TPTG)
    };
    dispatch_tg_1d(rt, &p, groups, tptg, None, |bnd| {
        set_gpu_buf(bnd, bank.packed, 0);
        set_gpu_buf(bnd, bank.scales, 1);
        set_gpu_buf(bnd, bank.zeros, 2);
        set_gpu_buf(bnd, x, 3);
        set_gpu_buf(bnd, y, 4);
        scalars(bnd);
    })
}

/// Threads per group for the one-thread-per-row MLX Q4 GEMV kernels
/// (`gemv_q4_mlx`, `gemv_q4_mlx_wide`), which still stage `x` in dynamic
/// threadgroup memory.
///
/// 128 amortizes the shared `x` cache across enough rows to pay for staging it,
/// without making the tail group wasteful on short matrices. The signed-nibble
/// `gemv_q4` left this geometry on 2026-09-05 for one simdgroup per four rows.
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
/// `out` must not alias the table or `token_ids`; independent output threads
/// may otherwise overwrite a later lookup's read-only input.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated lookup dimensions; the runtime owns the constant storage.
    unsafe {
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
}

/// [`embed_lookup_q4`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn embed_lookup_q4_with_scalars(
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
    bank.validate(rt, &shape, "embed_lookup_q4")?;
    let total = elems(n_tokens, hidden, "embed_lookup_q4")?;
    require_1d_indexable(total, "embed_lookup_q4")?;
    require::<u32>(
        rt,
        token_ids,
        n_tokens as usize,
        "embed_lookup_q4 token_ids",
    )?;
    require::<f32>(rt, out, total, "embed_lookup_q4 out")?;
    if total == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "embed_lookup_q4",
        &[("out", out)],
        &[
            ("packed", bank.packed),
            ("scales", bank.scales),
            ("zeros", bank.zeros),
            ("token_ids", token_ids),
        ],
    )?;

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
/// `out` must not alias either bank buffer or `token_ids`.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated lookup dimensions; the runtime owns the constant storage.
    unsafe {
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
}

/// [`embed_lookup_q4_mlx`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn embed_lookup_q4_mlx_with_scalars(
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
    bank.validate(rt, &shape, Q4MlxLayout::RowMajor, "embed_lookup_q4_mlx")?;
    let total = elems(n_tokens, hidden, "embed_lookup_q4_mlx")?;
    require_1d_indexable(total, "embed_lookup_q4_mlx")?;
    require::<u32>(
        rt,
        token_ids,
        n_tokens as usize,
        "embed_lookup_q4_mlx token_ids",
    )?;
    require::<f32>(rt, out, total, "embed_lookup_q4_mlx out")?;
    if total == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "embed_lookup_q4_mlx",
        &[("out", out)],
        &[
            ("packed", bank.packed),
            ("scales_biases", bank.scales_biases),
            ("token_ids", token_ids),
        ],
    )?;

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

/// Rows per Interleaved4 tile (`SIMD_ROWS` in `gemv_q4_mlx.metal`). An
/// interleaved bank is stored at `rows.div_ceil(I4_TILE_ROWS) * I4_TILE_ROWS`
/// rows, and [`Q4MlxBank`] is validated at that extent.
pub const I4_TILE_ROWS: u32 = 4;

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
/// `y` must not alias either bank buffer or `x`.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape values; the runtime owns the backing constant storage.
    unsafe {
        gemv_q4_mlx_with_scalars(rt, bank, x, y, shape, variant, |bnd| {
            set_u32(bnd, shape.rows, 5);
            set_u32(bnd, shape.cols, 6);
            set_u32(bnd, shape.group_size, 7);
        })
    }
}

/// Validate every host-side invariant for [`gemv_q4_mlx`] without allocating
/// scalar storage or encoding work.
///
/// This is the preflight seam for model adapters that bind dimensions from a
/// persistent scalar pool. It deliberately validates malformed shapes before
/// treating `rows == 0` as a no-op, preventing divide-by-zero dimensions from
/// being hidden by an empty launch.
pub fn validate_gemv_q4_mlx(
    rt: &GpuRuntime,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    variant: Q4MlxRowVariant,
) -> Result<(), String> {
    let entry = variant.entry();
    validate_gemv_q4_mlx_f32_inputs(rt, bank, x, y, shape, entry)?;
    if shape.rows == 0 {
        return Ok(());
    }
    if variant != Q4MlxRowVariant::Tiled {
        let bytes = (shape.cols as usize).saturating_mul(4);
        let limit = rt.max_threadgroup_memory();
        if bytes > limit {
            return Err(format!(
                "{entry}: caching x needs {bytes} bytes of threadgroup memory but this \
                 device allows {limit}; cols {} is too large for this kernel",
                shape.cols
            ));
        }
    }
    Ok(())
}

/// Validate the common f32-input MLX Q4 GEMV contract before an adapter picks
/// a row, tiled, or bf16-SIMD implementation.
///
/// Unlike [`validate_gemv_q4_mlx`], this does not impose a row-kernel
/// threadgroup-memory ceiling. It is intended for auto-routing adapters that
/// must reject malformed banks, foreign/undersized buffers, and output aliases
/// before they allocate scratch or encode a conversion dispatch.
pub fn validate_gemv_q4_mlx_inputs(
    rt: &GpuRuntime,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    validate_gemv_q4_mlx_f32_inputs(rt, bank, x, y, shape, "gemv_q4_mlx")
}

fn validate_gemv_q4_mlx_f32_inputs(
    rt: &GpuRuntime,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    entry: &str,
) -> Result<(), String> {
    bank.validate(rt, &shape, Q4MlxLayout::RowMajor, entry)?;
    require::<f32>(rt, x, shape.cols as usize, &format!("{entry} x"))?;
    require::<f32>(rt, y, shape.rows as usize, &format!("{entry} y"))?;
    if shape.rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        entry,
        &[("y", y)],
        &[
            ("packed", bank.packed),
            ("scales_biases", bank.scales_biases),
            ("x", x),
        ],
    )?;
    Ok(())
}

/// [`gemv_q4_mlx`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn gemv_q4_mlx_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    variant: Q4MlxRowVariant,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    let entry = variant.entry();
    validate_gemv_q4_mlx(rt, bank, x, y, shape, variant)?;
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
/// `y` must not alias either bank buffer or `x`.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape values; the runtime owns the backing constant storage.
    unsafe {
        gemv_q4_mlx_blocked_with_scalars(rt, bank, x, y, shape, |bnd| {
            set_u32(bnd, shape.rows, 5);
            set_u32(bnd, shape.cols, 6);
            set_u32(bnd, shape.group_size, 7);
        })
    }
}

/// Validate every host-side invariant for [`gemv_q4_mlx_blocked`] without
/// allocating scalar storage or encoding work.
pub fn validate_gemv_q4_mlx_blocked(
    rt: &GpuRuntime,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
) -> Result<(), String> {
    bank.validate(rt, &shape, Q4MlxLayout::RowMajor, "gemv_q4_mlx_blocked")?;
    require::<f32>(rt, x, shape.cols as usize, "gemv_q4_mlx_blocked x")?;
    require::<f32>(rt, y, shape.rows as usize, "gemv_q4_mlx_blocked y")?;
    if shape.rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        "gemv_q4_mlx_blocked",
        &[("y", y)],
        &[
            ("packed", bank.packed),
            ("scales_biases", bank.scales_biases),
            ("x", x),
        ],
    )
}

/// [`gemv_q4_mlx_blocked`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
pub unsafe fn gemv_q4_mlx_blocked_with_scalars(
    rt: &Arc<GpuRuntime>,
    bank: Q4MlxBank<'_>,
    x: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    scalars: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    validate_gemv_q4_mlx_blocked(rt, bank, x, y, shape)?;
    if shape.rows == 0 {
        return Ok(());
    }

    let p = rt.pipeline("gemv_q4_mlx_blocked")?;
    let groups = (shape.rows as usize).div_ceil(GEMV_BN);
    // The kernel caches x in threadgroup memory only when the whole row fits
    // (`cols <= GEMV_X_TILE`); past that it reads x from device memory and the
    // slot goes unused, so reserve the minimum rather than a full tile.
    let cols = shape.cols as usize;
    let tg_mem = if cols <= GEMV_X_TILE { cols * 4 } else { 16 };
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
/// `y` must not alias the bank or `x`. It may alias `resid`: lane zero reads
/// exactly the residual element it then overwrites, after the projection
/// reduction has completed.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape values; the runtime owns the backing constant storage.
    unsafe {
        gemv_q4_mlx_simd_with_scalars(rt, bank, x_bf16, y, shape, layout, resid, |bnd| {
            set_u32(bnd, shape.rows, 5);
            set_u32(bnd, shape.cols, 6);
            set_u32(bnd, shape.group_size, 7);
        })
    }
}

/// Validate every host-side invariant for [`gemv_q4_mlx_simd`] without
/// allocating scalar storage or encoding work.
///
/// Stable-scalar adapters use this before reserving arena slots. As with the
/// dispatch wrapper, `y == resid` is the one intentional writable alias.
pub fn validate_gemv_q4_mlx_simd(
    rt: &GpuRuntime,
    bank: Q4MlxBank<'_>,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    shape: QuantShape,
    layout: Q4MlxLayout,
    resid: Option<&GpuBuffer>,
) -> Result<(), String> {
    let entry = match (resid.is_some(), layout) {
        (false, Q4MlxLayout::RowMajor) => "gemv_q4_mlx_simd",
        (false, Q4MlxLayout::Interleaved4) => "gemv_q4_mlx_simd_i4",
        (true, Q4MlxLayout::RowMajor) => "gemv_q4_mlx_simd_add",
        (true, Q4MlxLayout::Interleaved4) => "gemv_q4_mlx_simd_add_i4",
    };
    bank.validate(rt, &shape, layout, entry)?;
    require::<u16>(rt, x_bf16, shape.cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(rt, y, shape.rows as usize, &format!("{entry} y"))?;
    if let Some(r) = resid {
        require::<f32>(rt, r, shape.rows as usize, &format!("{entry} resid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }
    // `y == resid` is an intentional fused in-place residual operation.
    require_disjoint_writes(
        entry,
        &[("y", y)],
        &[
            ("packed", bank.packed),
            ("scales_biases", bank.scales_biases),
            ("x_bf16", x_bf16),
        ],
    )
}

/// [`gemv_q4_mlx_simd`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_q4_mlx_simd_with_scalars(
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
    validate_gemv_q4_mlx_simd(rt, bank, x_bf16, y, shape, layout, resid)?;
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
/// The blocked kernel caches the complete activation row in a fixed 4096-float
/// threadgroup array, so that variant rejects `cols > 4096`; the simd variants
/// stream `x` and do not have this limit.
/// `mid` must not alias either weight bank or `x`; separate threadgroups write
/// output rows while continuing to read all three inputs.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape and output-format values; the runtime owns the constant
    // storage.
    unsafe {
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
}

/// [`gemv_q4_mlx_gate_up_gelu`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_q4_mlx_gate_up_gelu_with_scalars(
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
    shape.validate(entry)?;
    if dispatch == GateUpDispatch::Blocked && shape.cols as usize > GEMV_X_TILE {
        return Err(format!(
            "{entry}: cols {} exceeds the blocked kernel x-cache capacity {GEMV_X_TILE}",
            shape.cols
        ));
    }
    let bank_layout = match dispatch {
        GateUpDispatch::Simd(layout) => layout,
        GateUpDispatch::Blocked => Q4MlxLayout::RowMajor,
    };
    gate.validate(rt, &shape, bank_layout, &format!("{entry} gate"))?;
    up.validate(rt, &shape, bank_layout, &format!("{entry} up"))?;
    match dispatch {
        GateUpDispatch::Simd(_) => {
            require::<u16>(rt, x, shape.cols as usize, &format!("{entry} x"))?
        }
        GateUpDispatch::Blocked => {
            require::<f32>(rt, x, shape.cols as usize, &format!("{entry} x"))?
        }
    }
    if mid_as_bf16 {
        require::<u16>(rt, mid, shape.rows as usize, &format!("{entry} mid (bf16)"))?;
    } else {
        require::<f32>(rt, mid, shape.rows as usize, &format!("{entry} mid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        entry,
        &[("mid", mid)],
        &[
            ("gate_packed", gate.packed),
            ("gate_scales_biases", gate.scales_biases),
            ("up_packed", up.packed),
            ("up_scales_biases", up.scales_biases),
            ("x", x),
        ],
    )?;

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
/// `k_out` and `v_out` must be distinct and neither may alias a bank or `x`.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape and derived dispatch boundary; the runtime owns the
    // backing constant storage.
    unsafe {
        gemv_q4_mlx_kv_with_scalars(rt, k, v, x_bf16, k_out, v_out, shape, layout, |bnd| {
            set_u32(bnd, shape.rows, 9);
            set_u32(bnd, shape.cols, 10);
            set_u32(bnd, shape.group_size, 11);
            set_u32(bnd, tg_k, 12);
        })
    }
}

/// [`gemv_q4_mlx_kv`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_q4_mlx_kv_with_scalars(
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
    k.validate(rt, &shape, layout, &format!("{entry} k"))?;
    v.validate(rt, &shape, layout, &format!("{entry} v"))?;
    require::<u16>(rt, x_bf16, shape.cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(rt, k_out, shape.rows as usize, &format!("{entry} k_out"))?;
    require::<f32>(rt, v_out, shape.rows as usize, &format!("{entry} v_out"))?;
    if shape.rows == 0 {
        return Ok(());
    }
    require_disjoint_writes(
        entry,
        &[("k_out", k_out), ("v_out", v_out)],
        &[
            ("k_packed", k.packed),
            ("k_scales_biases", k.scales_biases),
            ("v_packed", v.packed),
            ("v_scales_biases", v.scales_biases),
            ("x_bf16", x_bf16),
        ],
    )?;

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
/// All active outputs must use distinct allocations, disjoint from their banks
/// and the shared activation. A zero row count makes that projection inactive
/// and excludes its otherwise-unused operands from the alias contract.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated dimensions and derived dispatch boundaries; the runtime owns
    // the backing constant storage.
    unsafe {
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
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_q4_mlx_qkv_with_scalars(
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
    q.validate(rt, &q_shape, layout, &format!("{entry} q"))?;
    k.validate(rt, &kv_shape, layout, &format!("{entry} k"))?;
    v.validate(rt, &kv_shape, layout, &format!("{entry} v"))?;
    require::<u16>(rt, x_bf16, cols as usize, &format!("{entry} x_bf16"))?;
    require::<f32>(rt, out.q_out, rows_q as usize, &format!("{entry} q_out"))?;
    require::<f32>(rt, out.k_out, rows_kv as usize, &format!("{entry} k_out"))?;
    require::<f32>(rt, out.v_out, rows_kv as usize, &format!("{entry} v_out"))?;
    if rows_q == 0 && rows_kv == 0 {
        return Ok(());
    }
    match (rows_q == 0, rows_kv == 0) {
        (true, false) => require_disjoint_writes(
            entry,
            &[("k_out", out.k_out), ("v_out", out.v_out)],
            &[
                ("k_packed", k.packed),
                ("k_scales_biases", k.scales_biases),
                ("v_packed", v.packed),
                ("v_scales_biases", v.scales_biases),
                ("x_bf16", x_bf16),
            ],
        )?,
        (false, true) => require_disjoint_writes(
            entry,
            &[("q_out", out.q_out)],
            &[
                ("q_packed", q.packed),
                ("q_scales_biases", q.scales_biases),
                ("x_bf16", x_bf16),
            ],
        )?,
        (false, false) => require_disjoint_writes(
            entry,
            &[
                ("q_out", out.q_out),
                ("k_out", out.k_out),
                ("v_out", out.v_out),
            ],
            &[
                ("q_packed", q.packed),
                ("q_scales_biases", q.scales_biases),
                ("k_packed", k.packed),
                ("k_scales_biases", k.scales_biases),
                ("v_packed", v.packed),
                ("v_scales_biases", v.scales_biases),
                ("x_bf16", x_bf16),
            ],
        )?,
        (true, true) => unreachable!("zero-output case returned above"),
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
/// `y` must not alias the bank or `x`. It may alias `resid`: the single writer
/// of each element loads that same residual value before overwriting it.
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
    // SAFETY: the closure binds only the documented scalar slots to the exact
    // validated shape and batch values; the runtime owns the backing constant
    // storage.
    unsafe {
        gemm_q4_mlx_with_scalars(rt, bank, x_bf16, y, shape, m, layout, resid, |bnd| {
            set_u32(bnd, shape.rows, 5);
            set_u32(bnd, shape.cols, 6);
            set_u32(bnd, shape.group_size, 7);
            set_u32(bnd, m, 8);
        })
    }
}

/// [`gemm_q4_mlx`] with caller-supplied scalar binds.
///
/// # Safety
///
/// `scalars` must obey the module-level `_with_scalars` contract: bind exactly
/// the documented scalar slots and values, mutate no other binder state, and
/// keep all stable scalar storage alive and resident for every execution.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_q4_mlx_with_scalars(
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
    bank.validate(rt, &shape, layout, entry)?;
    let out_elems = elems(m, shape.rows, entry)?;
    require::<u16>(
        rt,
        x_bf16,
        elems(m, shape.cols, entry)?,
        &format!("{entry} x_bf16"),
    )?;
    require::<f32>(rt, y, out_elems, &format!("{entry} y"))?;
    if let Some(r) = resid {
        require::<f32>(rt, r, out_elems, &format!("{entry} resid"))?;
    }
    if shape.rows == 0 {
        return Ok(());
    }
    // `y == resid` is an intentional fused in-place residual operation.
    require_disjoint_writes(
        entry,
        &[("y", y)],
        &[
            ("packed", bank.packed),
            ("scales_biases", bank.scales_biases),
            ("x_bf16", x_bf16),
        ],
    )?;

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

#[derive(Clone, Copy)]
struct RowReduceLayout {
    out_per_row: u32,
    allow_in_place: bool,
}

/// Threads per group for a row reduction: a power of two, capped by the
/// kernel's scratch depth and the pipeline's own limit, and never more than
/// the row is wide.
/// Most lanes a row reduction (softmax, row sum/max, RMSNorm) is given.
///
/// Rows wider than this stride the extra elements through the same lanes.
/// Measured 2026-09-05 against the 1024-lane maximum with the simdgroup-first
/// reductions (bench/results/row_reduction_simd_m5pro.txt): 256 lanes were
/// 1.9x faster on `row_sum` and 1.18x on softmax at 1024 columns, 1.27x on
/// RMSNorm at 64 rows, and within noise everywhere wider — a 1024-lane group
/// leaves a core little else to schedule, and the barrier it pays scales with
/// its size.
const REDUCE_ROW_MAX_LANES: usize = 256;

fn reduce_tptg(max_threads: usize, cols: usize) -> usize {
    reduction_tptg(
        max_threads,
        cols.next_power_of_two().max(1),
        REDUCE_ROW_MAX_LANES.min(REDUCE_MAX_TG),
    )
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
    row_reduce(
        rt,
        "softmax_rows_f32",
        x,
        out,
        rows,
        cols,
        RowReduceLayout {
            out_per_row: cols,
            allow_in_place: true,
        },
    )
}

/// `out[r] = sum(x[r, :])`. `out` holds one f32 per row and must not alias
/// `x`, because different row threadgroups are unordered.
pub fn row_sum_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    row_reduce(
        rt,
        "row_sum_f32",
        x,
        out,
        rows,
        cols,
        RowReduceLayout {
            out_per_row: 1,
            allow_in_place: false,
        },
    )
}

/// `out[r] = max(x[r, :])`. `out` holds one f32 per row and must not alias
/// `x`, because different row threadgroups are unordered.
pub fn row_max_f32(
    rt: &Arc<GpuRuntime>,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
) -> Result<(), String> {
    row_reduce(
        rt,
        "row_max_f32",
        x,
        out,
        rows,
        cols,
        RowReduceLayout {
            out_per_row: 1,
            allow_in_place: false,
        },
    )
}

/// Shared dispatch for the row reductions: one threadgroup per row.
///
/// `layout.out_per_row` is how many f32 each row writes — `cols` for softmax,
/// 1 for a scalar reduction — and is what `out`'s extent is checked against.
fn row_reduce(
    rt: &Arc<GpuRuntime>,
    entry: &str,
    x: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    cols: u32,
    layout: RowReduceLayout,
) -> Result<(), String> {
    if cols == 0 {
        return Err(format!("{entry}: cols must be non-zero"));
    }
    require::<f32>(rt, x, elems(rows, cols, entry)?, &format!("{entry} x"))?;
    require::<f32>(
        rt,
        out,
        elems(rows, layout.out_per_row, entry)?,
        &format!("{entry} out"),
    )?;
    if rows == 0 {
        return Ok(());
    }
    if !layout.allow_in_place {
        require_disjoint_writes(entry, &[("out", out)], &[("x", x)])?;
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
/// MPP TensorOps accumulates `int8 x int8` into `int32` natively, and the sum
/// is exact in int32 for every admitted `k` (below 2^17 at full int8 range).
/// Rounding happens once, at the store: the int32 sum is converted to f32
/// before the scales are applied, so a sum within ±2^24 (`k <= 1024` at full
/// range, or any admitted `k` with smaller products) is exact and a larger one
/// is the correctly rounded integer (f32 spacing is 2 up to 2^25, 4 up to
/// 2^26, ...). The only other approximation is the caller's quantization of
/// the operands.
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
///
/// `c` must not alias `a`, `b`, or a supplied `b_scale`: independent output
/// tiles are written while other tiles can still be reading those operands.
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
    // int32 accumulation is exact only while the running sum fits. The largest
    // signed int8 product is -128 * -128 = 16_384, so k = 2^17 already reaches
    // 2^31 and overflows i32. Keep the boundary derived from those exact limits
    // so every admitted full-range input accumulates exactly before the single
    // int32 -> f32 rounding at the store.
    const MAX_I8_PRODUCT: u32 = 16_384;
    const MAX_K_EXACT: u32 = i32::MAX as u32 / MAX_I8_PRODUCT;
    if k > MAX_K_EXACT {
        return Err(format!(
            "gemm_i8_dequant: k = {k} exceeds {MAX_K_EXACT}, past which an int32 \
             accumulator can overflow at full int8 range and the result wraps \
             silently rather than saturating"
        ));
    }
    let (a_elems, b_elems, c_elems) = i8_gemm_extents(m, n, k)?;
    require::<i8>(rt, a, a_elems, "gemm_i8_dequant a")?;
    require::<i8>(rt, b, b_elems, "gemm_i8_dequant b")?;
    require::<f32>(rt, c, c_elems, "gemm_i8_dequant c")?;
    if let Some(sc) = b_scale {
        require::<f32>(rt, sc, n as usize, "gemm_i8_dequant b_scale")?;
    }
    match b_scale {
        Some(sc) => require_disjoint_writes(
            "gemm_i8_dequant",
            &[("c", c)],
            &[("a", a), ("b", b), ("b_scale", sc)],
        )?,
        None => require_disjoint_writes("gemm_i8_dequant", &[("c", c)], &[("a", a), ("b", b)])?,
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

#[cfg(test)]
mod scalar_api_contract_tests {
    use super::{
        attn_kv_capacity_from_elements, checked_buffer_byte_range, elems_product, i8_gemm_extents,
    };

    #[test]
    fn buffer_byte_range_preflight_rejects_overflow_and_truncation() {
        assert!(checked_buffer_byte_range(16, 0, 16, "buffer").is_ok());
        assert!(checked_buffer_byte_range(16, 16, 0, "buffer").is_ok());

        let short = checked_buffer_byte_range(16, 8, 9, "buffer")
            .expect_err("range crossing the physical allocation must fail");
        assert!(short.contains("exceeds buffer capacity 16"));

        let overflow = checked_buffer_byte_range(usize::MAX, usize::MAX, 1, "buffer")
            .expect_err("wrapped byte range must fail closed");
        assert!(overflow.contains("overflows usize"));
    }

    #[test]
    fn i8_gemm_rejects_every_signed_index_extent_boundary() {
        let limit = i32::MAX as u32;
        assert!(i8_gemm_extents(limit, 1, 1).is_ok());

        for (label, dims) in [
            ("A", (limit / 8 + 1, 1, 8)),
            ("B", (1, limit / 8 + 1, 8)),
            ("C", (limit / 2 + 1, 2, 1)),
            ("all", (u32::MAX, u32::MAX, 8)),
        ] {
            let err = i8_gemm_extents(dims.0, dims.1, dims.2)
                .expect_err("signed-index overflow must be rejected");
            assert!(
                err.contains("signed 32-bit kernel indexing"),
                "{label}: unexpected error: {err}"
            );
        }
    }

    #[test]
    fn every_public_with_scalars_entry_is_explicitly_unsafe() {
        let source = include_str!("nn.rs");
        let entries: Vec<_> = source
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with("pub ") && line.contains("_with_scalars("))
            .collect();

        assert!(
            !entries.is_empty(),
            "expected raw scalar-binding entry points"
        );
        for declaration in entries {
            assert!(
                declaration.starts_with("pub unsafe fn "),
                "raw scalar-binding entry point must be unsafe: {declaration}"
            );
        }
    }

    #[test]
    fn kv_capacity_requires_one_exact_fixed_stride_for_k_and_v() {
        const PER_POSITION: usize = 8;

        assert_eq!(
            attn_kv_capacity_from_elements(
                4 * PER_POSITION + (PER_POSITION - 1),
                4 * PER_POSITION,
                PER_POSITION,
                "attention",
            )
            .unwrap(),
            4,
            "a partial trailing position must not alter the shared stride"
        );
        assert!(
            attn_kv_capacity_from_elements(
                3 * PER_POSITION,
                5 * PER_POSITION + 1,
                PER_POSITION,
                "attention",
            )
            .is_err(),
            "different complete K/V capacities have no safe shared batch stride"
        );
        assert!(
            attn_kv_capacity_from_elements(
                u32::MAX as usize + 1,
                u32::MAX as usize + 1,
                1,
                "attention",
            )
            .is_err(),
            "a fixed stride cannot be saturated without changing the layout"
        );
    }

    #[test]
    fn kv_capacity_dimension_product_fails_closed_on_host_overflow() {
        assert!(
            elems_product(&[u32::MAX, u32::MAX, u32::MAX], "attention capacity").is_err(),
            "wrapped dimensions could turn a huge device extent into a small accepted one"
        );
    }
}
