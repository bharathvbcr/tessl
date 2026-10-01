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
use crate::gdn_train::{
    gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace,
};
use crate::gemm::GemmOperands;
use crate::qwen35_adamw::{excluded_from_weight_decay, AdamW, AdamWHyper, Moment};
use crate::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use crate::qwen35_params::ParamInfo;
use crate::qwen35_train::{PendingStep, Qwen35Grads, Supervise};
use crate::runtime::GpuRuntime;
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};

pub const TESSL_OK: i32 = 0;
pub const TESSL_ERR: i32 = 1;
pub const TESSL_PANIC: i32 = 2;

/// Bumped on any change to a `#[repr(C)]` layout or an entry point's
/// signature; the Python side refuses a library whose version differs.
pub const TESSL_ABI_VERSION: u32 = 9;

/// Largest tensor rank a [`TesslTensorRef`] carries.
pub const TESSL_MAX_DIMS: usize = 6;

/// `dtype` codes in a [`TesslTensorRef`].
pub const TESSL_F32: u32 = 0;
pub const TESSL_BF16: u32 = 1;
pub const TESSL_F16: u32 = 2;

/// GEMM operand codes ([`GemmOperands`]): exact f32, or operands rounded to
/// bf16 with f32 accumulation. Any other value is refused.
pub const TESSL_OPERANDS_EXACT_F32: u32 = 0;
pub const TESSL_OPERANDS_BF16: u32 = 1;

fn parse_operands(code: u32, what: &str) -> Result<GemmOperands, String> {
    match code {
        TESSL_OPERANDS_EXACT_F32 => Ok(GemmOperands::ExactF32),
        TESSL_OPERANDS_BF16 => Ok(GemmOperands::Bf16),
        c => Err(format!(
            "{what}: operands code {c} is neither 0 (exact f32) nor 1 (bf16)"
        )),
    }
}

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
    /// The GEMMs' operands: [`TESSL_OPERANDS_EXACT_F32`] or
    /// [`TESSL_OPERANDS_BF16`].
    pub operands: u32,
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
    /// The last GDN backward workspace, reused for the same shape.
    gdn_ws: Option<GdnTrainWorkspace>,
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
            gdn_ws: None,
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

/// Free a handle from [`tessl_runtime_new`]. Null is ignored ([`TESSL_OK`]).
///
/// From a thread other than the creator's the handle is leaked, not freed,
/// and [`TESSL_ERR`] is returned: the runtime is thread-affine, and a
/// garbage-collected wrapper (Python's thread-local at interpreter shutdown)
/// can run its finalizer on any thread. A leaked runtime holds its buffers
/// until the process exits; dropping it on the wrong thread is undefined.
///
/// # Safety
/// `handle` is null or a pointer [`tessl_runtime_new`] returned that has not
/// been freed, and no other call on it is running.
#[no_mangle]
pub unsafe extern "C" fn tessl_runtime_free(handle: *mut TesslRuntime) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_handle(handle) }
}

/// Wait for every piece of work the runtime has submitted.
///
/// # Safety
/// As [`tessl_cross_entropy_rows`] for `handle`, `err` and `err_len`.
#[no_mangle]
pub unsafe extern "C" fn tessl_synchronize(handle: *mut TesslRuntime, err: *mut c_char, err_len: usize) -> i32 {
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
        (
            std::slice::from_raw_parts(a.rows, n),
            std::slice::from_raw_parts(a.targets, n),
        )
    };
    let reduction = match a.reduction {
        0 => Reduction::Mean,
        1 => Reduction::Sum,
        r => return Err(format!("{WHAT}: reduction {r} is neither 0 (mean) nor 1 (sum)")),
    };
    let operands = parse_operands(a.operands, WHAT)?;
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
        Some(CeGrads {
            dh: &dh,
            dw: &dw,
            scale: a.scale,
        })
    } else {
        None
    };
    cross_entropy_rows(
        &rt,
        CeHidden {
            rows: &hidden,
            off: a.col_off,
        },
        &weight,
        rows,
        targets,
        reduction,
        operands,
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
    let byte_offset = usize::try_from(t.byte_offset).map_err(|_| format!("{name}: byte offset overflows usize"))?;
    // SAFETY: a live MTLBuffer by the contract; retained for the wrap's lifetime.
    let buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
        unsafe { Retained::retain(t.buffer as *mut ProtocolObject<dyn MTLBuffer>) }
            .ok_or_else(|| format!("{name}: null MTLBuffer"))?;
    // SAFETY: the cross-queue half of from_mtl_buffer's contract is this
    // module's contract, which the caller upholds.
    unsafe { Tensor::from_mtl_buffer(rt, buffer, &shape, dtype, byte_offset) }.map_err(|e| format!("{name}: {e}"))
}

/// A thread-affine handle the ABI hands out.
trait Handle {
    /// What the handle is, in messages.
    const KIND: &'static str;
    fn owner(&self) -> ThreadId;
    fn runtime(&self) -> &Arc<GpuRuntime>;
}

impl Handle for TesslRuntime {
    const KIND: &'static str = "runtime";
    fn owner(&self) -> ThreadId {
        self.owner
    }
    fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.rt
    }
}

/// Free a boxed handle on its creating thread; see [`tessl_runtime_free`].
///
/// # Safety
/// As [`tessl_runtime_free`], for a handle of type `H`.
unsafe fn free_handle<H: Handle>(handle: *mut H) -> i32 {
    if handle.is_null() {
        return TESSL_OK;
    }
    // SAFETY: live by the contract; only read here.
    if unsafe { (*handle).owner() } != std::thread::current().id() {
        return TESSL_ERR;
    }
    // SAFETY: by the contract, the Box its constructor leaked, freed once.
    let boxed = unsafe { Box::from_raw(handle) };
    // Dropping sync-waits outstanding work; a panic there must not unwind into C.
    match catch_unwind(AssertUnwindSafe(move || drop(boxed))) {
        Ok(()) => TESSL_OK,
        Err(_) => TESSL_PANIC,
    }
}

/// Run `f` on the handle with every failure turned into a status: a null or
/// foreign-thread handle, an `Err`, or a panic. A successful call leaves the
/// runtime synchronized.
///
/// # Safety
/// `handle` is null or a live handle from its constructor with no other call
/// on it running; `err` is null or points to `err_len` writable bytes.
unsafe fn guarded<H: Handle>(
    handle: *mut H,
    err: *mut c_char,
    err_len: usize,
    f: impl FnOnce(&mut H) -> Result<(), String>,
) -> i32 {
    if handle.is_null() {
        // SAFETY: forwarded.
        unsafe { write_err(err, err_len, &format!("null {} handle", H::KIND)) };
        return TESSL_ERR;
    }
    // SAFETY: live and unshared by the contract.
    let h = unsafe { &mut *handle };
    if std::thread::current().id() != h.owner() {
        let msg = format!(
            "tessl {} used from a thread other than the one that created it",
            H::KIND
        );
        // SAFETY: forwarded.
        unsafe { write_err(err, err_len, &msg) };
        return TESSL_ERR;
    }
    match catch_unwind(AssertUnwindSafe(|| f(h).and_then(|()| h.runtime().synchronize()))) {
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

/// Arguments of [`tessl_gdn_train_forward`] and [`tessl_gdn_train_backward`];
/// see [`crate::gdn_train`]. Every tensor is dense f32. A null `buffer` means
/// "absent" for `s0`, `s_fin`, `d_fin` and `ds0`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TesslGdnArgs {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub v_dim: u32,
    /// `[B, T, H, 128]`.
    pub q: TesslTensorRef,
    pub k: TesslTensorRef,
    /// `[B, T, H, Dv]`.
    pub v: TesslTensorRef,
    /// `[B, T, H]`.
    pub g: TesslTensorRef,
    pub beta: TesslTensorRef,
    /// `[B, H, 128, Dv]`, optional.
    pub s0: TesslTensorRef,
    /// `[B, H, NC, 128, Dv]`, NC = ceil(T / 64): written by the forward,
    /// read by the backward.
    pub ckpt: TesslTensorRef,
    /// Forward: `o` `[B, T, H, Dv]` and optionally `s_fin` `[B, H, 128, Dv]`.
    pub o: TesslTensorRef,
    pub s_fin: TesslTensorRef,
    /// Backward: `d_o`, optionally `d_fin`, and the gradients (`ds0` exactly
    /// when `s0` is given).
    pub d_o: TesslTensorRef,
    pub d_fin: TesslTensorRef,
    pub dq: TesslTensorRef,
    pub dk: TesslTensorRef,
    pub dv: TesslTensorRef,
    pub dg: TesslTensorRef,
    pub dbeta: TesslTensorRef,
    pub ds0: TesslTensorRef,
}

/// # Safety
/// As [`wrap`] for a non-null `t.buffer`.
unsafe fn wrap_opt(rt: &Arc<GpuRuntime>, t: &TesslTensorRef, name: &str) -> Result<Option<Tensor>, String> {
    if t.buffer.is_null() {
        Ok(None)
    } else {
        // SAFETY: forwarded.
        unsafe { wrap(rt, t, name) }.map(Some)
    }
}

/// The inputs both directions read, wrapped.
struct GdnWrapped {
    dims: GdnTrainDims,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Option<Tensor>,
    ckpt: Tensor,
}

impl GdnWrapped {
    /// # Safety
    /// `a`'s buffers satisfy the module contract.
    unsafe fn new(rt: &Arc<GpuRuntime>, a: &TesslGdnArgs) -> Result<Self, String> {
        // SAFETY (each wrap): forwarded from this function's contract.
        unsafe {
            Ok(Self {
                dims: GdnTrainDims {
                    batch: a.batch,
                    seq: a.seq,
                    heads: a.heads,
                    v_dim: a.v_dim,
                },
                q: wrap(rt, &a.q, "q")?,
                k: wrap(rt, &a.k, "k")?,
                v: wrap(rt, &a.v, "v")?,
                g: wrap(rt, &a.g, "g")?,
                beta: wrap(rt, &a.beta, "beta")?,
                s0: wrap_opt(rt, &a.s0, "s0")?,
                ckpt: wrap(rt, &a.ckpt, "ckpt")?,
            })
        }
    }

    fn inputs(&self) -> GdnTrainInputs<'_> {
        GdnTrainInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: self.s0.as_ref(),
        }
    }
}

/// [`crate::gdn_train::gdn_train_forward`] over caller buffers.
///
/// # Safety
/// As [`tessl_cross_entropy_rows`], with `args` a valid [`TesslGdnArgs`].
#[no_mangle]
pub unsafe extern "C" fn tessl_gdn_train_forward(
    handle: *mut TesslRuntime,
    args: *const TesslGdnArgs,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(handle, err, err_len, |h| {
            if args.is_null() {
                return Err("tessl_gdn_train_forward: null args".into());
            }
            let a = &*args;
            let rt = Arc::clone(&h.rt);
            let x = GdnWrapped::new(&rt, a)?;
            let o = wrap(&rt, &a.o, "o")?;
            let s_fin = wrap_opt(&rt, &a.s_fin, "s_fin")?;
            gdn_train_forward(&rt, x.dims, x.inputs(), &o, s_fin.as_ref(), &x.ckpt)
        })
    }
}

/// [`crate::gdn_train::gdn_train_backward`] over caller buffers. The
/// backward workspace is cached per handle and per shape.
///
/// # Safety
/// As [`tessl_gdn_train_forward`].
#[no_mangle]
pub unsafe extern "C" fn tessl_gdn_train_backward(
    handle: *mut TesslRuntime,
    args: *const TesslGdnArgs,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(handle, err, err_len, |h| {
            if args.is_null() {
                return Err("tessl_gdn_train_backward: null args".into());
            }
            let a = &*args;
            let rt = Arc::clone(&h.rt);
            let x = GdnWrapped::new(&rt, a)?;
            let d_o = wrap(&rt, &a.d_o, "d_o")?;
            let d_fin = wrap_opt(&rt, &a.d_fin, "d_fin")?;
            let (dq, dk, dv) = (
                wrap(&rt, &a.dq, "dq")?,
                wrap(&rt, &a.dk, "dk")?,
                wrap(&rt, &a.dv, "dv")?,
            );
            let (dg, dbeta) = (wrap(&rt, &a.dg, "dg")?, wrap(&rt, &a.dbeta, "dbeta")?);
            let ds0 = wrap_opt(&rt, &a.ds0, "ds0")?;
            if h.gdn_ws.as_ref().map(GdnTrainWorkspace::dims) != Some(x.dims) {
                h.gdn_ws = None;
                h.gdn_ws = Some(GdnTrainWorkspace::new(&rt, x.dims)?);
            }
            let ws = h.gdn_ws.as_ref().ok_or("tessl_gdn_train_backward: workspace missing")?;
            gdn_train_backward(
                &rt,
                x.dims,
                x.inputs(),
                &x.ckpt,
                &d_o,
                d_fin.as_ref(),
                ws,
                GdnTrainGrads {
                    dq: &dq,
                    dk: &dk,
                    dv: &dv,
                    dg: &dg,
                    dbeta: &dbeta,
                    ds0: ds0.as_ref(),
                },
            )
        })
    }
}

// ------------------------------------------------------------ Qwen3.5 ---

/// Longest parameter name a [`TesslParamInfo`] carries, NUL included.
pub const TESSL_NAME_LEN: usize = 128;

/// [`tessl_qwen35_copy`] directions.
pub const TESSL_READ_PARAMS: u32 = 0;
pub const TESSL_READ_GRADS: u32 = 1;
pub const TESSL_WRITE_PARAMS: u32 = 2;
/// AdamW state ([`tessl_qwen35_adamw_init`]): torch's `exp_avg` and
/// `exp_avg_sq`, read or written in the parameters' layouts.
pub const TESSL_READ_ADAMW_M: u32 = 3;
pub const TESSL_READ_ADAMW_V: u32 = 4;
pub const TESSL_WRITE_ADAMW_M: u32 = 5;
pub const TESSL_WRITE_ADAMW_V: u32 = 6;

/// One entry of a model's parameter table; see
/// [`crate::qwen35_params::ParamInfo`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TesslParamInfo {
    /// transformers' name below the text tower, NUL-terminated.
    pub name: [c_char; TESSL_NAME_LEN],
    pub ndim: u32,
    /// Non-zero: the tensors a copy takes hold the `[in, out]` transpose of
    /// the 2-D `shape`.
    pub transposed: u32,
    /// Non-zero: transformers' Trainer excludes this parameter from weight
    /// decay ([`crate::qwen35_adamw::excluded_from_weight_decay`]), the
    /// default [`tessl_qwen35_adamw_step`] callers start from.
    pub decay_excluded: u32,
    /// transformers' shape; the first `ndim` entries are used.
    pub shape: [u64; TESSL_MAX_DIMS],
}

/// What the handle's gradient bank holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankState {
    /// Zeros, as allocated: nothing has run into it.
    Empty,
    /// A finished step's gradients (or several, accumulated).
    Ready,
    /// A step into it was refused or failed part way: neither.
    Dirty,
}

/// What a [`tessl_qwen35_load`] pointer owns: an f32 model, its gradient
/// bank (allocated by the first step and reused by every later one), a step
/// between [`tessl_qwen35_train_forward`] and [`tessl_qwen35_train_backward`],
/// and AdamW state once [`tessl_qwen35_adamw_init`] made it.
pub struct TesslQwen35 {
    model: Qwen35Model,
    table: Vec<ParamInfo>,
    grads: Option<Qwen35Grads>,
    bank: BankState,
    pending: Option<PendingStep>,
    adamw: Option<AdamW>,
    owner: ThreadId,
}

impl TesslQwen35 {
    /// The bank, allocated (zeroed) on first use.
    fn bank(&mut self) -> Result<&Qwen35Grads, String> {
        if self.grads.is_none() {
            self.grads = Some(Qwen35Grads::zeros_like(&self.model)?);
            self.bank = BankState::Empty;
        }
        self.grads.as_ref().ok_or_else(|| "no gradient bank".to_string())
    }

    fn no_pending(&self, what: &str) -> Result<(), String> {
        if self.pending.is_some() {
            return Err(format!(
                "{what}: a step is pending; call tessl_qwen35_train_backward or tessl_qwen35_train_discard first"
            ));
        }
        Ok(())
    }
}

/// The bank's gradients, when they are a finished step's and no step is
/// pending.
fn ready_grads<'a>(
    what: &str,
    grads: &'a Option<Qwen35Grads>,
    bank: BankState,
    pending: &Option<PendingStep>,
) -> Result<&'a Qwen35Grads, String> {
    if pending.is_some() {
        return Err(format!(
            "{what}: a step is pending; call tessl_qwen35_train_backward or tessl_qwen35_train_discard first"
        ));
    }
    match (grads, bank) {
        (Some(g), BankState::Ready) => Ok(g),
        _ => Err(format!("{what}: no gradients yet; run tessl_qwen35_train_step first")),
    }
}

impl Handle for TesslQwen35 {
    const KIND: &'static str = "model";
    fn owner(&self) -> ThreadId {
        self.owner
    }
    fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.model.rt
    }
}

/// # Safety
/// `s` is non-null and NUL-terminated (null is refused).
unsafe fn c_str<'a>(s: *const c_char, what: &str) -> Result<&'a str, String> {
    if s.is_null() {
        return Err(format!("null {what}"));
    }
    // SAFETY: NUL-terminated by the contract.
    unsafe { std::ffi::CStr::from_ptr(s) }
        .to_str()
        .map_err(|_| format!("{what} is not UTF-8"))
}

/// Load the text tower of a Qwen3.5 checkpoint in f32 for training, on
/// `runtime`'s device, into a new handle written to `*out`.
///
/// `safetensors` is the `.safetensors` file, `config_json` its `config.json`
/// (with or without `text_config`), and `prefix` the tensor-name prefix
/// (`"model.language_model."` in the Qwen3.5 checkpoints). Free the handle
/// with [`tessl_qwen35_free`], on this thread.
///
/// # Safety
/// As [`tessl_cross_entropy_rows`] for `runtime`, `err` and `err_len`; the
/// three strings are NUL-terminated (null is refused); `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_load(
    runtime: *mut TesslRuntime,
    safetensors: *const c_char,
    config_json: *const c_char,
    prefix: *const c_char,
    out: *mut *mut TesslQwen35,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(runtime, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_load";
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            *out = std::ptr::null_mut();
            let path = c_str(safetensors, "safetensors path").map_err(|e| format!("{WHAT}: {e}"))?;
            let config = c_str(config_json, "config path").map_err(|e| format!("{WHAT}: {e}"))?;
            let prefix = c_str(prefix, "prefix").map_err(|e| format!("{WHAT}: {e}"))?;
            let cfg = Qwen35Config::from_config_file(std::path::Path::new(config))?;
            let st = SafeTensors::open(std::path::Path::new(path))?;
            let model = Qwen35Model::load(&h.rt, &st, prefix, cfg, Precision::F32)?;
            let table = model.parameter_table()?;
            if let Some(p) = table
                .iter()
                .find(|p| p.name.len() >= TESSL_NAME_LEN || p.shape.len() > TESSL_MAX_DIMS)
            {
                return Err(format!("{WHAT}: {} does not fit a TesslParamInfo", p.name));
            }
            *out = Box::into_raw(Box::new(TesslQwen35 {
                model,
                table,
                grads: None,
                bank: BankState::Empty,
                pending: None,
                adamw: None,
                owner: h.owner,
            }));
            Ok(())
        })
    }
}

/// Free a handle from [`tessl_qwen35_load`], as [`tessl_runtime_free`] frees
/// a runtime (null is ignored; from another thread it is leaked and
/// [`TESSL_ERR`] returned).
///
/// # Safety
/// As [`tessl_runtime_free`], for a [`tessl_qwen35_load`] handle.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_free(model: *mut TesslQwen35) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_handle(model) }
}

/// The number of entries in the model's parameter table.
///
/// # Safety
/// As [`tessl_cross_entropy_rows`], for a [`tessl_qwen35_load`] handle;
/// `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_param_count(
    model: *mut TesslQwen35,
    out: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            if out.is_null() {
                return Err("tessl_qwen35_param_count: null out".into());
            }
            *out = h.table.len() as u64;
            Ok(())
        })
    }
}

/// Entry `index` of the model's parameter table.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `out` points to a writable
/// [`TesslParamInfo`].
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_param_info(
    model: *mut TesslQwen35,
    index: u64,
    out: *mut TesslParamInfo,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_param_info";
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            let p = usize::try_from(index)
                .ok()
                .and_then(|i| h.table.get(i))
                .ok_or_else(|| format!("{WHAT}: index {index} is outside the table's {} entries", h.table.len()))?;
            let mut info = TesslParamInfo {
                name: [0; TESSL_NAME_LEN],
                ndim: p.shape.len() as u32,
                transposed: u32::from(p.transposed),
                decay_excluded: u32::from(excluded_from_weight_decay(&p.name)),
                shape: [0; TESSL_MAX_DIMS],
            };
            // Lengths were checked at load.
            for (d, &b) in info.name.iter_mut().zip(p.name.as_bytes()) {
                *d = b as c_char;
            }
            for (d, &s) in info.shape.iter_mut().zip(&p.shape) {
                *d = s as u64;
            }
            *out = info;
            Ok(())
        })
    }
}

/// One training step on the `n` token ids at `ids` (one sequence), its GEMMs
/// on `operands` ([`TESSL_OPERANDS_EXACT_F32`] or [`TESSL_OPERANDS_BF16`]):
/// writes the loss to `*loss` and every parameter's gradient into the
/// handle's bank for [`tessl_qwen35_copy`], over the previous step's (a
/// refused or failed step leaves none). Refused while a step is pending,
/// which leaves the bank as it was.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `ids` points to `n` readable `u32`s and
/// `loss` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_train_step(
    model: *mut TesslQwen35,
    ids: *const u32,
    n: u64,
    operands: u32,
    loss: *mut f64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_train_step";
            if ids.is_null() || loss.is_null() {
                return Err(format!("{WHAT}: null ids or loss"));
            }
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            let ids = std::slice::from_raw_parts(ids, n);
            // A pending step's bank is left alone; any other refusal leaves
            // no gradients.
            h.no_pending(WHAT)?;
            h.bank()?;
            h.bank = BankState::Dirty;
            let operands = parse_operands(operands, WHAT)?;
            let bank = h.grads.as_ref().ok_or("no gradient bank")?;
            *loss = h.model.train_step_into(ids, operands, Supervise::Causal, bank, false)?;
            h.bank = BankState::Ready;
            Ok(())
        })
    }
}

/// [`tessl_qwen35_train_forward`]: transformers' causal-LM loss, the mean
/// over every next-token prediction ([`Supervise::Causal`]).
pub const TESSL_SUPERVISE_CAUSAL: u32 = 0;
/// [`tessl_qwen35_train_forward`]: the hidden state at `positions[i]` scored
/// against `targets[i]`, the sum, with the gradients of `scale` times it
/// ([`Supervise::Rows`]; no rows at all is allowed).
pub const TESSL_SUPERVISE_ROWS: u32 = 1;

/// The forward and loss of one step on the `n` token ids at `ids`, kept in
/// the handle for [`tessl_qwen35_hidden`] and [`tessl_qwen35_train_backward`]
/// ([`Qwen35Model::train_forward`]). `supervise` picks the loss:
/// [`TESSL_SUPERVISE_CAUSAL`] (`positions` and `targets` null, `n_rows` 0) or
/// [`TESSL_SUPERVISE_ROWS`] (`n_rows` positions and targets; null only when
/// `n_rows` is 0). The loss goes to `*loss`. Refused while a step is pending.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `ids` points to `n` readable `u32`s,
/// `positions` and `targets` to `n_rows` each, and `loss` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_train_forward(
    model: *mut TesslQwen35,
    ids: *const u32,
    n: u64,
    operands: u32,
    supervise: u32,
    positions: *const u32,
    targets: *const u32,
    n_rows: u64,
    scale: f32,
    loss: *mut f64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_train_forward";
            if ids.is_null() || loss.is_null() {
                return Err(format!("{WHAT}: null ids or loss"));
            }
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            let n_rows = usize::try_from(n_rows).map_err(|_| format!("{WHAT}: n_rows overflows usize"))?;
            let ids = std::slice::from_raw_parts(ids, n);
            let operands = parse_operands(operands, WHAT)?;
            let rows = |p: *const u32, name: &str| -> Result<&[u32], String> {
                match (p.is_null(), n_rows) {
                    (true, 0) => Ok(&[]),
                    (true, _) => Err(format!("{WHAT}: null {name} for {n_rows} rows")),
                    (false, k) => Ok(std::slice::from_raw_parts(p, k)),
                }
            };
            let sup = match supervise {
                TESSL_SUPERVISE_CAUSAL => {
                    if n_rows != 0 || !positions.is_null() || !targets.is_null() {
                        return Err(format!("{WHAT}: the causal loss takes no positions or targets"));
                    }
                    Supervise::Causal
                }
                TESSL_SUPERVISE_ROWS => Supervise::Rows {
                    positions: rows(positions, "positions")?,
                    targets: rows(targets, "targets")?,
                    scale,
                },
                s => {
                    return Err(format!(
                    "{WHAT}: supervise {s} is not {TESSL_SUPERVISE_CAUSAL} (causal) or {TESSL_SUPERVISE_ROWS} (rows)"
                ))
                }
            };
            h.no_pending(WHAT)?;
            let p = h.model.train_forward(ids, operands, sup)?;
            *loss = p.loss();
            h.pending = Some(p);
            Ok(())
        })
    }
}

/// Rows `positions` of the pending step's final-norm output (transformers'
/// `last_hidden_state`) into `out`, dense f32 `[n, hidden]`
/// ([`PendingStep::hidden`]). Positions may repeat.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `positions` points to `n` readable
/// `u32`s (null only when `n` is 0) and `out` to a [`TesslTensorRef`] whose
/// buffer satisfies the module contract.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_hidden(
    model: *mut TesslQwen35,
    positions: *const u32,
    n: u64,
    out: *const TesslTensorRef,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_hidden";
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            if out.is_null() || (positions.is_null() && n != 0) {
                return Err(format!("{WHAT}: null positions or out"));
            }
            let positions = if n == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(positions, n)
            };
            let p = h
                .pending
                .as_ref()
                .ok_or_else(|| format!("{WHAT}: no step is pending; call tessl_qwen35_train_forward first"))?;
            let out = wrap(&h.model.rt, &*out, "out")?;
            p.hidden(positions, &out)
        })
    }
}

/// The pending step's backward into the handle's bank: over it, or added to
/// it when `accumulate` is non-zero ([`Qwen35Model::train_backward_into`]).
/// `dh`, when not null, is the gradient of a loss outside tessl at rows
/// `positions` (distinct) of the final norm's output, dense f32
/// `[n, hidden]`; null with `n` 0 adds none. Everything is checked before
/// anything runs, and a refusal keeps the step pending; once the backward
/// runs the step is consumed. Accumulating onto a bank a failed step left
/// part-written is refused. The parameters must not change in between: a
/// [`TESSL_WRITE_PARAMS`] copy is refused while a step is pending.
///
/// # Safety
/// As [`tessl_qwen35_hidden`], for `positions` and `dh`.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_train_backward(
    model: *mut TesslQwen35,
    positions: *const u32,
    n: u64,
    dh: *const TesslTensorRef,
    accumulate: u32,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_train_backward";
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            let rt = Arc::clone(&h.model.rt);
            let dh = match (dh.is_null(), positions.is_null(), n) {
                (true, true, 0) => None,
                (false, false, k) => Some((std::slice::from_raw_parts(positions, k), wrap(&rt, &*dh, "dh")?)),
                (false, true, 0) => Some((&[][..], wrap(&rt, &*dh, "dh")?)),
                _ => {
                    return Err(format!(
                        "{WHAT}: dh and positions must both be given, or both null with n 0"
                    ))
                }
            };
            let dh = dh.as_ref().map(|(p, t)| (*p, t));
            let accumulate = accumulate != 0;
            {
                let p = h
                    .pending
                    .as_ref()
                    .ok_or_else(|| format!("{WHAT}: no step is pending; call tessl_qwen35_train_forward first"))?;
                h.model.check_pending(WHAT, p, dh)?;
            }
            h.bank()?;
            if accumulate && h.bank == BankState::Dirty {
                return Err(format!(
                    "{WHAT}: the bank holds a failed step's partial gradients; start again with accumulate = 0"
                ));
            }
            let p = h.pending.take().ok_or("no pending step")?;
            h.bank = BankState::Dirty;
            let bank = h.grads.as_ref().ok_or("no gradient bank")?;
            h.model.train_backward_into(p, dh, bank, accumulate)?;
            h.bank = BankState::Ready;
            Ok(())
        })
    }
}

/// Drop the pending step, if any, without its backward.
///
/// # Safety
/// As [`tessl_qwen35_param_count`].
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_train_discard(model: *mut TesslQwen35, err: *mut c_char, err_len: usize) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            h.pending = None;
            Ok(())
        })
    }
}

/// Copy between the model and `n` caller tensors, one per parameter-table
/// entry in order, each dense f32 of the entry's shape (transposed when the
/// entry says so): [`TESSL_READ_PARAMS`] and [`TESSL_READ_GRADS`] (the last
/// step's) fill them, [`TESSL_WRITE_PARAMS`] sets the parameters from them,
/// and [`TESSL_READ_ADAMW_M`] ... [`TESSL_WRITE_ADAMW_V`] read or set the AdamW
/// moments in the same layouts. Values are transformers' (see
/// [`crate::qwen35_params`] for the layouts tessl keeps them in). Every
/// tensor is checked before anything is copied.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `tensors` points to `n` readable
/// [`TesslTensorRef`]s whose buffers satisfy the module contract.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_copy(
    model: *mut TesslQwen35,
    direction: u32,
    tensors: *const TesslTensorRef,
    n: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_copy";
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            if n != h.table.len() {
                return Err(format!("{WHAT}: {n} tensors for {} parameters", h.table.len()));
            }
            if tensors.is_null() {
                return Err(format!("{WHAT}: null tensors"));
            }
            let refs = std::slice::from_raw_parts(tensors, n);
            let rt = Arc::clone(&h.model.rt);
            let ts = refs
                .iter()
                .zip(&h.table)
                .map(|(r, p)| wrap(&rt, r, &p.name))
                .collect::<Result<Vec<_>, _>>()?;
            match direction {
                TESSL_READ_PARAMS => h.model.read_parameters(&ts),
                TESSL_READ_GRADS => {
                    let g = ready_grads(WHAT, &h.grads, h.bank, &h.pending)?;
                    h.model.read_gradients(g, &ts)
                }
                TESSL_WRITE_PARAMS => {
                    // The backward rebuilds each layer from the forward's inputs
                    // with the weights it finds: they must be the forward's.
                    h.no_pending(WHAT)?;
                    h.model.write_parameters(&ts)
                }
                TESSL_READ_ADAMW_M | TESSL_READ_ADAMW_V | TESSL_WRITE_ADAMW_M | TESSL_WRITE_ADAMW_V => {
                    let state = h
                        .adamw
                        .as_mut()
                        .ok_or_else(|| format!("{WHAT}: no AdamW state; call tessl_qwen35_adamw_init first"))?;
                    let which = if matches!(direction, TESSL_READ_ADAMW_M | TESSL_WRITE_ADAMW_M) {
                        Moment::First
                    } else {
                        Moment::Second
                    };
                    if direction <= TESSL_READ_ADAMW_V {
                        h.model.read_adamw_moment(state, which, &ts)
                    } else {
                        h.model.write_adamw_moment(state, which, &ts)
                    }
                }
                d => Err(format!(
                    "{WHAT}: direction {d} is not 0 (read params), 1 (read grads), 2 (write params), \
                     3 or 4 (read AdamW m or v) or 5 or 6 (write AdamW m or v)"
                )),
            }
        })
    }
}

/// Make the handle's AdamW state: both moments zeroed, twice the parameters'
/// memory (16 GB on the 2B), step count 0. Refused if the handle has one.
///
/// # Safety
/// As [`tessl_qwen35_param_count`].
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_adamw_init(model: *mut TesslQwen35, err: *mut c_char, err_len: usize) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            if h.adamw.is_some() {
                return Err("tessl_qwen35_adamw_init: the model already has AdamW state".into());
            }
            h.adamw = Some(AdamW::new(&h.model)?);
            Ok(())
        })
    }
}

/// Drop the handle's AdamW state, if any.
///
/// # Safety
/// As [`tessl_qwen35_param_count`].
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_adamw_free(model: *mut TesslQwen35, err: *mut c_char, err_len: usize) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            h.adamw = None;
            Ok(())
        })
    }
}

/// One AdamW step ([`crate::qwen35_adamw`]) on every parameter from the last
/// [`tessl_qwen35_train_step`]'s gradients, which stay in the handle, with
/// `weight_decay[i]` for parameter-table entry `i` (`n` must be the table's
/// length) and every gradient multiplied by `grad_scale` first (1, or the
/// clip coefficient formed from [`tessl_qwen35_grad_sq_norm`]; the
/// gradients themselves are left as they are). Everything is checked before
/// anything moves.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `weight_decay` points to `n` readable
/// `f32`s.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_adamw_step(
    model: *mut TesslQwen35,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    grad_scale: f64,
    weight_decay: *const f32,
    n: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_adamw_step";
            let n = usize::try_from(n).map_err(|_| format!("{WHAT}: n overflows usize"))?;
            if n != h.table.len() {
                return Err(format!("{WHAT}: {n} weight decays for {} parameters", h.table.len()));
            }
            if weight_decay.is_null() {
                return Err(format!("{WHAT}: null weight_decay"));
            }
            let wd = std::slice::from_raw_parts(weight_decay, n);
            let grads = ready_grads(WHAT, &h.grads, h.bank, &h.pending)?;
            let state = h
                .adamw
                .as_mut()
                .ok_or_else(|| format!("{WHAT}: no AdamW state; call tessl_qwen35_adamw_init first"))?;
            h.model.adamw_step(
                grads,
                state,
                &AdamWHyper {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                },
                wd,
            )
        })
    }
}

/// The sum of squares of every gradient the last [`tessl_qwen35_train_step`]
/// left ([`Qwen35Model::grad_sq_norm`]), written to `*out`: the square of
/// the global norm `clip_grad_norm_` takes, before any gradients outside
/// the model are added.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `out` points to a writable `f64`.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_grad_sq_norm(
    model: *mut TesslQwen35,
    out: *mut f64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_grad_sq_norm";
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            let grads = ready_grads(WHAT, &h.grads, h.bank, &h.pending)?;
            *out = h.model.grad_sq_norm(grads)?;
            Ok(())
        })
    }
}

/// Set the AdamW step count, restoring a checkpoint together with the
/// moments written through [`tessl_qwen35_copy`].
///
/// # Safety
/// As [`tessl_qwen35_param_count`].
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_adamw_set_step_count(
    model: *mut TesslQwen35,
    step: u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            let state = h
                .adamw
                .as_mut()
                .ok_or("tessl_qwen35_adamw_set_step_count: no AdamW state")?;
            state.set_step_count(step);
            Ok(())
        })
    }
}

/// The AdamW steps taken so far, written to `*out`.
///
/// # Safety
/// As [`tessl_qwen35_param_count`]; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tessl_qwen35_adamw_step_count(
    model: *mut TesslQwen35,
    out: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        guarded(model, err, err_len, |h| {
            const WHAT: &str = "tessl_qwen35_adamw_step_count";
            if out.is_null() {
                return Err(format!("{WHAT}: null out"));
            }
            let state = h.adamw.as_ref().ok_or_else(|| format!("{WHAT}: no AdamW state"))?;
            *out = state.step_count();
            Ok(())
        })
    }
}
