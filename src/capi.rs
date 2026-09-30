//! A C ABI over the parts of tessl a Python training loop calls, so torch can
//! use tessl through `ctypes` without a C++ extension.
//!
//! The crate builds a `cdylib` (`libtessl.dylib`) exporting the `tessl_*`
//! functions below; `python/tessl_torch` loads it. Everything crosses the
//! boundary as plain C: an opaque runtime handle, `#[repr(C)]` tensor
//! descriptors naming an `id<MTLBuffer>` plus a byte offset, and an `i32`
//! status with the message written into a caller buffer.
//!
//! # The contract a caller upholds
//!
//! - A [`TesslTensorRef`]'s `buffer` is a live `id<MTLBuffer>` on the
//!   system default device. From torch that is
//!   `t.untyped_storage().data_ptr()` for an MPS tensor (ATen's own
//!   `getMTLBufferStorage` is that pointer, bit-cast), with
//!   `t.storage_offset() * t.element_size()` as `byte_offset`.
//! - Cross-queue order ([`crate::Tensor::from_mtl_buffer`]'s safety
//!   contract): the caller finishes its own queue's work on every buffer it
//!   passes before the call (`torch.mps.synchronize()`), and does not touch
//!   them from another queue until the call returns. Each entry point
//!   synchronizes tessl before it returns, so outputs are complete.
//! - A handle is used from the thread that created it; calls from any other
//!   thread are refused (tessl's runtime is thread-affine, and `ctypes`
//!   releases the GIL around foreign calls).
//!
//! Status codes: [`TESSL_OK`], [`TESSL_ERR`] (the message says why), and
//! [`TESSL_PANIC`] (a bug in tessl, caught at the boundary rather than
//! unwinding into C; the handle should be freed).

use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread::ThreadId;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, Tensor};

pub const TESSL_OK: i32 = 0;
pub const TESSL_ERR: i32 = 1;
pub const TESSL_PANIC: i32 = 2;

/// Bumped on any change to a `#[repr(C)]` layout or an entry point's
/// signature; the Python side refuses a library whose version differs.
pub const TESSL_ABI_VERSION: u32 = 1;

/// Largest tensor rank a [`TesslTensorRef`] carries.
pub const TESSL_MAX_DIMS: usize = 4;

/// `dtype` codes in a [`TesslTensorRef`].
pub const TESSL_F32: u32 = 0;
pub const TESSL_BF16: u32 = 1;
pub const TESSL_F16: u32 = 2;

/// A dense, row-major tensor inside a caller's `MTLBuffer`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TesslTensorRef {
    /// `id<MTLBuffer>`.
    pub buffer: *mut c_void,
    pub byte_offset: u64,
    /// [`TESSL_F32`], [`TESSL_BF16`] or [`TESSL_F16`].
    pub dtype: u32,
    pub ndim: u32,
    /// The first `ndim` entries are used.
    pub shape: [u64; TESSL_MAX_DIMS],
}

/// Arguments of [`tessl_cross_entropy_rows`]; see
/// [`crate::cross_entropy::cross_entropy_rows`] for their meaning.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TesslCeArgs {
    /// `[T, ld]`, f32 or bf16.
    pub hidden: TesslTensorRef,
    /// First hidden column in each row.
    pub col_off: u32,
    /// `[V, H]`, f32 or bf16.
    pub weight: TesslTensorRef,
    /// `n` row indices into `hidden` and `n` target ids.
    pub rows: *const u32,
    pub targets: *const u32,
    pub n: u64,
    /// 0 = mean, 1 = sum.
    pub reduction: u32,
    /// Vocabulary columns per step; 0 picks [`DEFAULT_CE_CHUNK`].
    pub chunk: u32,
    /// Non-zero: also write `dh` (`[n, H]` f32) and `dw` (`[V, H]` f32),
    /// scaled by `scale`.
    pub want_grads: u32,
    pub scale: f32,
    pub dh: TesslTensorRef,
    pub dw: TesslTensorRef,
}

/// Vocabulary columns per step when the caller passes 0: 4096 columns of f32
/// logits per supervised row, and 4096 x H of widened bf16 weight.
pub const DEFAULT_CE_CHUNK: u32 = 4096;

/// What a [`tessl_runtime_new`] pointer owns.
pub struct TesslRuntime {
    rt: Arc<GpuRuntime>,
    owner: ThreadId,
    /// The last cross-entropy workspace, reused while it fits.
    ce_ws: Option<CeWorkspace>,
    ce_ws_key: (u32, u32, DType),
}

/// The ABI version this library implements ([`TESSL_ABI_VERSION`]).
#[no_mangle]
pub extern "C" fn tessl_abi_version() -> u32 {
    TESSL_ABI_VERSION
}

/// A new runtime on the system default device, or null with the reason in
/// `err`.
///
/// # Safety
/// `err` is null or points to `err_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn tessl_runtime_new(err: *mut c_char, err_len: usize) -> *mut TesslRuntime {
    let made = catch_unwind(GpuRuntime::new);
    match made {
        Ok(Ok(rt)) => Box::into_raw(Box::new(TesslRuntime {
            rt,
            owner: std::thread::current().id(),
            ce_ws: None,
            ce_ws_key: (0, 0, DType::F32),
        })),
        Ok(Err(e)) => {
            // SAFETY: forwarded from this function's contract.
            unsafe { write_err(err, err_len, &e) };
            std::ptr::null_mut()
        }
        Err(_) => {
            // SAFETY: forwarded from this function's contract.
            unsafe { write_err(err, err_len, "panic while creating the runtime") };
            std::ptr::null_mut()
        }
    }
}

/// Free a handle from [`tessl_runtime_new`]. Null is ignored.
///
/// # Safety
/// `handle` is null or a pointer [`tessl_runtime_new`] returned that has not
/// been freed, and no other call on it is running.
#[no_mangle]
pub unsafe extern "C" fn tessl_runtime_free(handle: *mut TesslRuntime) {
    if handle.is_null() {
        return;
    }
    // SAFETY: by the contract, this is the Box tessl_runtime_new leaked, freed once.
    let boxed = unsafe { Box::from_raw(handle) };
    // Dropping sync-waits outstanding work; a panic there must not unwind into C.
    let _ = catch_unwind(AssertUnwindSafe(move || drop(boxed)));
}

/// Wait for every piece of work the runtime has submitted.
///
/// # Safety
/// As [`tessl_cross_entropy_rows`] for `handle`, `err` and `err_len`.
#[no_mangle]
pub unsafe extern "C" fn tessl_synchronize(
    handle: *mut TesslRuntime,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { guarded(handle, err, err_len, |h| h.rt.synchronize()) }
}

/// The length in bytes of an `id<MTLBuffer>`, or 0 for null. A diagnostic
/// for bindings: it checks that the pointer a framework hands out really is
/// the buffer object before anything is wrapped.
///
/// # Safety
/// `buffer` is null or a live `id<MTLBuffer>`.
#[no_mangle]
pub unsafe extern "C" fn tessl_mtl_buffer_length(buffer: *mut c_void) -> u64 {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: a live MTLBuffer by the contract; borrowed, not retained.
    let buf = unsafe { &*(buffer as *const ProtocolObject<dyn MTLBuffer>) };
    buf.length() as u64
}

/// [`crate::cross_entropy::cross_entropy_rows`] over caller buffers.
/// Writes the reduced loss to `*loss` and, when `per_row` is not null, the
/// `n` per-row losses to it.
///
/// # Safety
/// `handle` comes from [`tessl_runtime_new`], is not freed, and is used on
/// its creating thread (checked). `args` points to a valid [`TesslCeArgs`]
/// whose `rows` and `targets` point to `n` readable `u32`s and whose buffers
/// satisfy the module's contract. `loss` is writable; `per_row` is null or
/// points to `n` writable `f64`s; `err` is null or points to `err_len`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn tessl_cross_entropy_rows(
    handle: *mut TesslRuntime,
    args: *const TesslCeArgs,
    loss: *mut f64,
    per_row: *mut f64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(handle, err, err_len, |h| {
            if args.is_null() || loss.is_null() {
                return Err("tessl_cross_entropy_rows: null args or loss".into());
            }
            let a = &*args;
            let out = ce(h, a)?;
            *loss = out.loss;
            if !per_row.is_null() {
                std::ptr::copy_nonoverlapping(out.per_row.as_ptr(), per_row, out.per_row.len());
            }
            Ok(())
        })
    }
}

/// # Safety
/// `a`'s pointers satisfy [`tessl_cross_entropy_rows`]'s contract.
unsafe fn ce(h: &mut TesslRuntime, a: &TesslCeArgs) -> Result<crate::cross_entropy::CeOutput, String> {
    const WHAT: &str = "tessl_cross_entropy_rows";
    let n = usize::try_from(a.n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
    if n == 0 {
        return Err(format!("{WHAT}: no supervised rows (an empty selection has no mean)"));
    }
    if a.rows.is_null() || a.targets.is_null() {
        return Err(format!("{WHAT}: null rows or targets"));
    }
    let n32 = u32::try_from(n).map_err(|_| format!("{WHAT}: {n} rows exceed u32"))?;
    // SAFETY: n readable u32s each, by the contract.
    let (rows, targets) = unsafe {
        (std::slice::from_raw_parts(a.rows, n), std::slice::from_raw_parts(a.targets, n))
    };
    let reduction = match a.reduction {
        0 => Reduction::Mean,
        1 => Reduction::Sum,
        r => return Err(format!("{WHAT}: reduction {r} is neither 0 (mean) nor 1 (sum)")),
    };
    let rt = Arc::clone(&h.rt);
    // SAFETY (each wrap): live MTLBuffers under the module contract.
    let hidden = unsafe { wrap(&rt, &a.hidden, "hidden") }?;
    let weight = unsafe { wrap(&rt, &a.weight, "weight") }?;
    if weight.shape().len() != 2 {
        return Err(format!("{WHAT}: weight must be 2-D, got {:?}", weight.shape()));
    }
    let hs = u32::try_from(weight.shape()[1]).map_err(|_| format!("{WHAT}: hidden exceeds u32"))?;
    let chunk = if a.chunk == 0 { DEFAULT_CE_CHUNK } else { a.chunk };
    let key = (hs, chunk, weight.dtype);
    let fits = h.ce_ws.as_ref().is_some_and(|ws| ws.max_rows() >= n32) && h.ce_ws_key == key;
    if !fits {
        // Grow to the next power of two so a slowly growing batch does not
        // reallocate every step.
        let max_rows = n32.checked_next_power_of_two().unwrap_or(n32);
        h.ce_ws = None;
        h.ce_ws = Some(CeWorkspace::new(&rt, max_rows, hs, chunk, weight.dtype)?);
        h.ce_ws_key = key;
    }
    let ws = h.ce_ws.as_ref().ok_or_else(|| format!("{WHAT}: workspace missing"))?;
    let (dh, dw);
    let grads = if a.want_grads != 0 {
        // SAFETY: as above.
        dh = unsafe { wrap(&rt, &a.dh, "dh") }?;
        dw = unsafe { wrap(&rt, &a.dw, "dw") }?;
        Some(CeGrads { dh: &dh, dw: &dw, scale: a.scale })
    } else {
        None
    };
    cross_entropy_rows(
        &rt,
        CeHidden { rows: &hidden, off: a.col_off },
        &weight,
        rows,
        targets,
        reduction,
        ws,
        grads,
    )
}

/// # Safety
/// `t.buffer` is null (refused) or a live `id<MTLBuffer>` under the module
/// contract.
unsafe fn wrap(rt: &Arc<GpuRuntime>, t: &TesslTensorRef, name: &str) -> Result<Tensor, String> {
    if t.buffer.is_null() {
        return Err(format!("{name}: null MTLBuffer"));
    }
    let dtype = match t.dtype {
        TESSL_F32 => DType::F32,
        TESSL_BF16 => DType::BF16,
        TESSL_F16 => DType::F16,
        d => return Err(format!("{name}: unknown dtype code {d}")),
    };
    let ndim = t.ndim as usize;
    if ndim == 0 || ndim > TESSL_MAX_DIMS {
        return Err(format!("{name}: rank {ndim} is outside 1..={TESSL_MAX_DIMS}"));
    }
    let shape = t.shape[..ndim]
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| format!("{name}: dimension {d} overflows usize")))
        .collect::<Result<Vec<_>, _>>()?;
    let byte_offset =
        usize::try_from(t.byte_offset).map_err(|_| format!("{name}: byte offset overflows usize"))?;
    // SAFETY: a live MTLBuffer by the contract; retained for the wrap's lifetime.
    let buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
        unsafe { Retained::retain(t.buffer as *mut ProtocolObject<dyn MTLBuffer>) }
            .ok_or_else(|| format!("{name}: null MTLBuffer"))?;
    // SAFETY: the cross-queue half of from_mtl_buffer's contract is this
    // module's contract, which the caller upholds.
    unsafe { Tensor::from_mtl_buffer(rt, buffer, &shape, dtype, byte_offset) }
        .map_err(|e| format!("{name}: {e}"))
}

/// Run `f` on the handle with every failure turned into a status: a null or
/// foreign-thread handle, an `Err`, or a panic. A successful call leaves the
/// runtime synchronized.
///
/// # Safety
/// `handle` is null or a live handle from [`tessl_runtime_new`] with no other
/// call on it running; `err` is null or points to `err_len` writable bytes.
unsafe fn guarded(
    handle: *mut TesslRuntime,
    err: *mut c_char,
    err_len: usize,
    f: impl FnOnce(&mut TesslRuntime) -> Result<(), String>,
) -> i32 {
    if handle.is_null() {
        // SAFETY: forwarded.
        unsafe { write_err(err, err_len, "null runtime handle") };
        return TESSL_ERR;
    }
    // SAFETY: live and unshared by the contract.
    let h = unsafe { &mut *handle };
    if std::thread::current().id() != h.owner {
        // SAFETY: forwarded.
        unsafe {
            write_err(
                err,
                err_len,
                "tessl runtime used from a thread other than the one that created it",
            )
        };
        return TESSL_ERR;
    }
    match catch_unwind(AssertUnwindSafe(|| f(h).and_then(|()| h.rt.synchronize()))) {
        Ok(Ok(())) => TESSL_OK,
        Ok(Err(e)) => {
            // SAFETY: forwarded.
            unsafe { write_err(err, err_len, &e) };
            TESSL_ERR
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            // SAFETY: forwarded.
            unsafe { write_err(err, err_len, &format!("panic in tessl: {msg}")) };
            TESSL_PANIC
        }
    }
}

/// Copy `msg` into `err` as a NUL-terminated string, truncated to fit (at a
/// UTF-8 boundary).
///
/// # Safety
/// `err` is null or points to `err_len` writable bytes.
unsafe fn write_err(err: *mut c_char, err_len: usize, msg: &str) {
    if err.is_null() || err_len == 0 {
        return;
    }
    let mut end = msg.len().min(err_len - 1);
    while !msg.is_char_boundary(end) {
        end -= 1;
    }
    // SAFETY: end < err_len bytes of `err` are writable, plus the NUL.
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), err.cast::<u8>(), end);
        *err.add(end) = 0;
    }
}
