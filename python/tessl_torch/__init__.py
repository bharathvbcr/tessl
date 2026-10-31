"""torch bindings for tessl's Metal kernels, through the C ABI in src/capi.rs.

    import tessl_torch
    loss = tessl_torch.cross_entropy(hidden, weight, targets, mask)
    loss.backward()

    model = tessl_torch.Qwen35(safetensors_path, config_json_path)
    params = model.parameters()           # an f32 copy of the weights, on MPS

The Qwen3.5 training step, its gradients and its AdamW retired with ABI 14:
ojas (the Tape over ojas-metal) owns Metal training, and tessl keeps the
kernels. `Qwen35` here loads a checkpoint and reads and writes its
parameters.

Only the standard library and torch are needed: the library is loaded with
ctypes, and MPS tensors are handed over as the MTLBuffer behind their storage
(``untyped_storage().data_ptr()``, which is what ATen's own
``getMTLBufferStorage`` bit-casts) plus a byte offset.

Build the library first (``cargo build --release``) or point ``TESSL_LIB`` at
a ``libtessl.dylib``. It loads its kernels from the metallib path baked in at
build time, so keep the build's ``target/`` directory in place.

Torch's MPS queue and tessl's are separate. A call that hands tessl a torch
buffer first waits for torch's queue (``torch.mps.synchronize()``), and every
call returns only after tessl's queue is idle, on failure as well as success,
so the two never touch a buffer at once and a buffer torch recycles after an
exception is never still in use. tessl's runtime batches a call's kernels
(async encode) and waits once, at the end: one tessl wait per call instead of
one per kernel. See README.md, "How the buffers cross", for why torch's side
of the handoff stays a CPU wait.
"""

from __future__ import annotations

import ctypes
import math
import operator
import os
import threading
import warnings
import weakref
from pathlib import Path

import torch

__all__ = [
    "cross_entropy",
    "cross_entropy_rows",
    "clip_coef",
    "Qwen35",
    "TesslError",
    "TesslPanic",
    "TesslPoisoned",
    "library_path",
]

_ABI_VERSION = 14
_MAX_DIMS = 6
_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_ERR_LEN = 1024
# GEMM operand codes: exact f32, or operands rounded to bf16 with f32 accumulation.
_OPERANDS_CODE = {"f32": 0, "bf16": 1}


_STATUS_PANIC = 2  # TESSL_PANIC: a bug in tessl, caught at the C boundary


class TesslError(RuntimeError):
    """A tessl call refused its arguments or failed on the device."""


class TesslPanic(TesslError):
    """A tessl call hit a bug in tessl (the C ABI's ``TESSL_PANIC``) and was
    caught at the boundary. The handle it came from is closed (the library
    says it should be freed): later calls on that object raise
    :class:`TesslError` without reaching the library."""


class TesslPoisoned(TesslError):
    """A tessl call failed and left the runtime poisoned (an encode, submit or
    GPU fault latched it; the library refuses all further work on it). The
    message names the original error. The runtime and every model bound to it
    are closed; the next call on this thread builds a fresh runtime, and a
    model must be loaded again."""


def _status_error(status: int, err) -> TesslError:
    """The exception for a non-zero status, with the library's message."""
    cls = TesslPanic if status == _STATUS_PANIC else TesslError
    return cls(err.value.decode(errors="replace"))


def _flag(name: str, value) -> int:
    """A boolean flag as the C ABI's 0 or 1: a bool or the ints 0 and 1,
    nothing else (``2**32`` would wrap to 0 in a c_uint32)."""
    if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
        return int(value)
    raise TesslError(f"{name} must be a bool, not {value!r}")


def _check_uint(name: str, value, bits: int) -> int:
    """``value`` as an int in ``[0, 2**bits)``, the range of the C field it
    is about to fill. ctypes wraps out-of-range integers silently (``-1``
    becomes ``2**32 - 1``, ``2**32`` becomes 0), so every integer headed for
    a ctypes field is checked here first."""
    if isinstance(value, bool):
        raise TesslError(f"{name} must be an integer, not bool")
    try:
        v = operator.index(value)
    except TypeError:
        raise TesslError(f"{name} must be an integer, not {type(value).__name__}") from None
    if v < 0:
        raise TesslError(f"{name} {v} is negative")
    if v >= 1 << bits:
        raise TesslError(f"{name} {v} is outside [0, 2**{bits})")
    return v


class _Owned:
    """A tessl handle with one owner. ``close()`` frees it once (a second call
    does nothing) and after it every use raises :class:`TesslError` without
    reaching the library. Copying and pickling are refused: a copy would own
    the same pointer, and the library frees a handle exactly once. Also a
    context manager (``with`` closes on exit).

    Subclasses name the C free function in ``_free_name`` and read the live
    handle through ``_live()``."""

    _free_name = ""
    _kind = "handle"
    _h = None
    _lib = None

    def _own(self, lib, handle) -> None:
        self._lib = lib
        self._h = handle

    def _live(self):
        if self._h is None:
            raise TesslError(f"this tessl {self._kind} is closed")
        return self._h

    @property
    def closed(self) -> bool:
        return self._h is None

    def close(self) -> None:
        """Free the handle. Idempotent: the object is closed after the first
        call whatever the library says, and later calls do nothing.

        Raises :class:`TesslPanic` if the free itself hit a bug in tessl, and
        :class:`TesslError` if it was called from a thread other than the one
        that made the handle: the library then refuses and leaks the handle
        (held until the process exits) rather than drop a thread-affine
        runtime on the wrong thread."""
        h, self._h = self._h, None
        if h is None or self._lib is None:
            return
        status = getattr(self._lib, self._free_name)(h)
        if status == _STATUS_PANIC:
            raise TesslPanic(f"tessl {self._kind}: freeing the handle hit a bug in tessl")
        if status != 0:
            raise TesslError(
                f"tessl {self._kind} closed from a thread other than the one that made it: "
                f"the handle is leaked until the process exits"
            )

    def _close_after_panic(self) -> None:
        """Close the object whose call panicked. A failing free is dropped:
        the panic being raised is the news."""
        try:
            self.close()
        except TesslError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except TesslError as e:
            # A finalizer must not raise; a leak is still worth saying.
            try:
                warnings.warn(str(e), ResourceWarning, stacklevel=2)
            except Exception:  # interpreter shutdown
                pass

    def __copy__(self):
        raise TypeError(f"tessl {self._kind}s own a C handle and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError(f"tessl {self._kind}s own a C handle and cannot be copied")

    def __reduce__(self):
        raise TypeError(f"tessl {self._kind}s own a C handle and cannot be pickled")


def _operands_code(operands: str) -> int:
    if operands not in _OPERANDS_CODE:
        raise TesslError(f"operands must be 'f32' or 'bf16', not {operands!r}")
    return _OPERANDS_CODE[operands]


def clip_coef(total_norm: float, max_norm: float) -> float:
    """``torch.nn.utils.clip_grad_norm_``'s coefficient for a global norm,
    in its f32 arithmetic, bit for bit: ``max_norm * (1 / (total_norm + 1e-6))``
    (torch's ``max_norm / tensor`` multiplies the reciprocal), at most 1. The
    arithmetic is ``tessl_clip_coef``'s, in Rust. Pass it as
    ``Qwen35.adamw_step(grad_scale=...)`` and scale any gradients outside
    tessl by it too."""
    return float(_load().tessl_clip_coef(float(total_norm), float(max_norm)))


class _TensorRef(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("byte_offset", ctypes.c_uint64),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * _MAX_DIMS),
    ]


class _CeArgs(ctypes.Structure):
    _fields_ = [
        ("hidden", _TensorRef),
        ("col_off", ctypes.c_uint32),
        ("weight", _TensorRef),
        ("rows", ctypes.POINTER(ctypes.c_uint32)),
        ("targets", ctypes.POINTER(ctypes.c_uint32)),
        ("n", ctypes.c_uint64),
        ("reduction", ctypes.c_uint32),
        ("operands", ctypes.c_uint32),
        ("chunk", ctypes.c_uint32),
        ("want_grads", ctypes.c_uint32),
        ("scale", ctypes.c_float),
        ("dh", _TensorRef),
        ("dw", _TensorRef),
    ]




_NAME_LEN = 128


class _ParamInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * _NAME_LEN),
        ("ndim", ctypes.c_uint32),
        ("transposed", ctypes.c_uint32),
        ("decay_excluded", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * _MAX_DIMS),
    ]


def library_path() -> Path:
    """Where the tessl C library is loaded from."""
    env = os.environ.get("TESSL_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "target" / "release" / "libtessl.dylib"


_lib = None
_lib_lock = threading.Lock()


def _load():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = library_path()
        if not path.exists():
            raise TesslError(f"{path} does not exist; run `cargo build --release` or set TESSL_LIB")
        lib = ctypes.CDLL(str(path))
        lib.tessl_abi_version.restype = ctypes.c_uint32
        lib.tessl_abi_version.argtypes = []
        version = lib.tessl_abi_version()
        if version != _ABI_VERSION:
            raise TesslError(f"{path} implements ABI {version}, this binding expects {_ABI_VERSION}")
        lib.tessl_runtime_new.restype = ctypes.c_void_p
        lib.tessl_runtime_new.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.tessl_runtime_free.restype = ctypes.c_int32
        lib.tessl_runtime_free.argtypes = [ctypes.c_void_p]
        lib.tessl_runtime_poisoned.restype = ctypes.c_int32
        lib.tessl_runtime_poisoned.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_size_t]
        lib.tessl_runtime_set_async_encode.restype = ctypes.c_int32
        lib.tessl_runtime_set_async_encode.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
        lib.tessl_synchronize.restype = ctypes.c_int32
        lib.tessl_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.tessl_mtl_buffer_length.restype = ctypes.c_uint64
        lib.tessl_mtl_buffer_length.argtypes = [ctypes.c_void_p]
        lib.tessl_cross_entropy_rows.restype = ctypes.c_int32
        lib.tessl_cross_entropy_rows.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CeArgs),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        err_args = [ctypes.c_char_p, ctypes.c_size_t]
        lib.tessl_qwen35_load.restype = ctypes.c_int32
        lib.tessl_qwen35_load.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ] + err_args
        lib.tessl_qwen35_free.restype = ctypes.c_int32
        lib.tessl_qwen35_free.argtypes = [ctypes.c_void_p]
        lib.tessl_qwen35_param_count.restype = ctypes.c_int32
        lib.tessl_qwen35_param_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)] + err_args
        lib.tessl_qwen35_param_info.restype = ctypes.c_int32
        lib.tessl_qwen35_param_info.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(_ParamInfo)] + err_args
        f32p = ctypes.POINTER(ctypes.c_float)
        lib.tessl_qwen35_read_parameter_host.restype = ctypes.c_int32
        lib.tessl_qwen35_read_parameter_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64, f32p, ctypes.c_uint64] + err_args
        lib.tessl_qwen35_write_parameter_host.restype = ctypes.c_int32
        lib.tessl_qwen35_write_parameter_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64, f32p, ctypes.c_uint64] + err_args
        lib.tessl_qwen35_copy.restype = ctypes.c_int32
        lib.tessl_qwen35_copy.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_TensorRef), ctypes.c_uint64,
        ] + err_args
        lib.tessl_qwen35_describe.restype = ctypes.c_int32
        lib.tessl_qwen35_describe.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        ] + err_args
        lib.tessl_clip_coef.restype = ctypes.c_float
        lib.tessl_clip_coef.argtypes = [ctypes.c_double, ctypes.c_double]
        _lib = lib
        return lib


class _Runtime(_Owned):
    """One tessl runtime per thread: tessl's runtime is thread-affine."""

    _free_name = "tessl_runtime_free"
    _kind = "runtime"
    handle = property(lambda self: self._live())

    def __init__(self):
        lib = _load()
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = lib.tessl_runtime_new(err, _ERR_LEN)
        if not handle:
            raise _status_error(1, err)
        self.lib = lib
        self._own(lib, ctypes.c_void_p(handle))
        # Async encode: a call's kernels are batched into command buffers and
        # waited for once, when the call returns (on failure too: `guarded` in
        # src/capi.rs finishes what a refused call already submitted, so the
        # buffers torch frees after an exception are never still in use). The
        # library's own default is synchronous, one waited commit per kernel.
        status = lib.tessl_runtime_set_async_encode(self.handle, 1, err, _ERR_LEN)
        if status:
            error = _status_error(status, err)
            self._close_after_panic()
            raise error
        # Models loaded on this runtime, closed with it if it is poisoned.
        self._models = weakref.WeakSet()

    def is_poisoned(self) -> bool:
        """Whether the library reports this runtime poisoned. False when it is
        closed or the query itself is refused (the caller reports the original
        error then)."""
        if self.closed:
            return False
        out = ctypes.c_int32()
        err = ctypes.create_string_buffer(_ERR_LEN)
        status = self.lib.tessl_runtime_poisoned(self.handle, ctypes.byref(out), err, _ERR_LEN)
        return status == 0 and out.value == 1

    def _close_poisoned(self) -> None:
        """Close the models bound to this runtime and then the runtime, and
        drop it as the thread's runtime so the next call builds a fresh one.

        This is also what keeps torch's buffers safe after a poisoned call. A
        poisoned runtime refuses the wait a failing call otherwise ends with,
        so tessl work for that call may still be in flight; freeing the runtime
        waits for the GPU (bounded by the library's event-wait budget, 30 s by
        default). The call's tensors are locals of the frames the exception is
        still unwinding through, so they outlive this close and torch cannot
        recycle their storage before it returns. A caller that catches the
        exception and drops its buffers does so after the wait; one that
        sees ``TesslPoisoned`` need keep nothing alive."""
        for model in list(self._models):
            model._close_after_panic()
        self._close_after_panic()
        if getattr(_local, "rt", None) is self:
            _local.rt = None

    # close() from a foreign thread (interpreter shutdown) is refused by the
    # library, which leaks the runtime rather than drop it on the wrong thread.


_local = threading.local()


def _runtime_status(rt: "_Runtime", status: int, err) -> None:
    """Raise for a non-zero status of a call on the thread's runtime. A panic
    closes that runtime; a refusal that left it poisoned closes it and its
    models and raises :class:`TesslPoisoned`; the next call on this thread
    builds a fresh runtime."""
    if status == 0:
        return
    if status == _STATUS_PANIC:
        rt._close_after_panic()
        raise _status_error(status, err)
    raise _error_or_poisoned(rt, status, err)


def _error_or_poisoned(rt: "_Runtime", status: int, err) -> TesslError:
    """The exception for a failed call on ``rt``: :class:`TesslPoisoned` (after
    closing ``rt`` and its models) when the library says the call poisoned the
    runtime, else the plain status error. The message is read before the
    query, which reuses no buffer of the caller's."""
    error = _status_error(status, err)
    if rt.is_poisoned():
        rt._close_poisoned()
        return TesslPoisoned(f"tessl runtime poisoned: {error}")
    return error


def _runtime() -> _Runtime:
    rt = getattr(_local, "rt", None)
    if rt is None or rt.closed:
        rt = _Runtime()
        _local.rt = rt
    return rt


def _finish_torch_queue() -> None:
    """Wait for torch's MPS queue before tessl reads or writes a torch buffer.

    This is the torch-to-tessl half of the handoff and it stays a CPU wait.
    tessl's side has a shared-event handoff (``GpuRuntime::shared_event()`` and
    ``last_signaled_value()``: tessl signals it after each commit), and the
    tessl-to-torch half is already that, enforced by the library: every call
    returns with tessl's queue idle. The other half would need torch to signal
    an ``MTLSharedEvent`` that tessl then waits on, and torch (2.14) exposes no
    way to do that: ``torch.mps.Event`` wraps an event torch creates itself, with
    no handle to share, and nothing lets Python encode a wait or signal for an
    external one. Signalling tessl's own event from torch's side is the one
    thing that must not be done: tessl recycles memory off its private lifetime
    event, and the handoff event is separate so that foreign code signalling it
    cannot free memory early. So each call that gives tessl a torch buffer
    (cross-entropy, ``copy``)
    waits here once; the calls that give it only host arrays do not."""
    torch.mps.synchronize()


def _ref(t: torch.Tensor, name: str) -> _TensorRef:
    """Describe an MPS tensor as the MTLBuffer behind its storage.

    The tensor must be row-major contiguous (the ABI carries no strides).
    """
    if t.device.type != "mps":
        raise TesslError(f"{name} must be on the mps device, not {t.device}")
    if t.dtype not in _DTYPE_CODE:
        raise TesslError(f"{name}: dtype {t.dtype} is not f32, bf16 or f16")
    if not t.is_contiguous():
        raise TesslError(f"{name} must be contiguous")
    if not 1 <= t.dim() <= _MAX_DIMS:
        raise TesslError(f"{name}: rank {t.dim()} is outside 1..{_MAX_DIMS}")
    storage = t.untyped_storage()
    offset = t.storage_offset() * t.element_size()
    end = offset + t.numel() * t.element_size()
    # tessl checks against the MTLBuffer's length, which torch's allocator
    # rounds up; the storage is the real extent.
    if end > storage.nbytes():
        raise TesslError(f"{name}: bytes [{offset}, {end}) exceed its storage ({storage.nbytes()})")
    ref = _TensorRef()
    ref.buffer = storage.data_ptr()
    ref.byte_offset = offset
    ref.dtype = _DTYPE_CODE[t.dtype]
    ref.ndim = t.dim()
    for i, d in enumerate(t.shape):
        ref.shape[i] = d
    return ref


def _row_view(hidden: torch.Tensor):
    """``hidden`` ([..., H]) as a ``[T, ld]`` tensor over its storage plus a
    function mapping flat row indices of ``hidden`` to rows of that tensor.

    Works without copying whenever the last dimension is unit-stride and
    every other stride is a multiple of the second-to-last one, which covers
    contiguous tensors and slices along leading dimensions such as
    ``hidden[:, :-1]``. Otherwise the tensor is copied contiguous first.
    """
    h = hidden if hidden.dim() >= 2 else hidden.unsqueeze(0)
    H = h.shape[-1]
    ld = h.stride(-2) if h.dim() >= 2 and h.shape[-2] > 1 else H
    lead_strides = h.stride()[:-1]
    ok = (
        h.stride(-1) == 1
        and ld >= H
        and all(s % ld == 0 for s in lead_strides)
    )
    if not ok:
        h = h.contiguous()
        ld = H
        lead_strides = h.stride()[:-1]
    storage = h.untyped_storage()
    elems = storage.nbytes() // h.element_size()
    base = h.storage_offset()
    # The [T, ld] tensor starts at the storage offset and runs as many whole
    # rows as the storage holds past it.
    t_rows = (elems - base) // ld
    base_view = torch.empty(0, dtype=h.dtype, device=h.device).set_(storage, base, (t_rows, ld), (ld, 1))
    lead_shape = h.shape[:-1]
    row_steps = torch.tensor([s // ld for s in lead_strides], dtype=torch.int64)

    def to_rows(flat: torch.Tensor) -> torch.Tensor:
        idx = torch.stack(torch.unravel_index(flat.cpu(), lead_shape), dim=-1)
        return (idx * row_steps).sum(-1)

    return base_view, to_rows


def cross_entropy_rows(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    rows: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
    chunk: int = 0,
    grads: bool = False,
    scale: float = 1.0,
    operands: str = "f32",
    per_row: bool = True,
):
    """The raw call: ``hidden`` is a contiguous ``[T, ld]`` MPS tensor whose
    first ``H`` columns are the hidden states, ``weight`` a contiguous
    ``[V, H]`` one; ``rows`` and ``targets`` are the ``n`` supervised row
    indices and token ids. Returns ``(loss, per_row, dh, dw)``: the loss as
    a Python float, per-row losses as a CPU float64 tensor, and with
    ``grads`` the f32 ``[n, H]`` and ``[V, H]`` gradients of ``scale * loss``
    (``None`` otherwise). ``operands`` is the GEMMs': ``"f32"`` (exact) or
    ``"bf16"`` (operands rounded to bf16, f32 accumulation). With
    ``per_row=False`` the per-row losses are not copied out and ``None`` is
    returned in their place.
    """
    chunk = _check_uint("chunk", chunk, 32)
    if reduction not in ("mean", "sum"):
        raise TesslError(f"reduction must be 'mean' or 'sum', not {reduction!r}")
    operands_code = _operands_code(operands)
    if hidden.dim() != 2 or weight.dim() != 2:
        raise TesslError("hidden must be [T, ld] and weight [V, H]")
    n = rows.numel()
    if n == 0:
        raise TesslError("no supervised rows (an empty selection has no mean)")
    if targets.numel() != n:
        raise TesslError(f"{n} rows but {targets.numel()} targets")
    rows_u32 = _u32(rows, hidden.shape[0], "rows")
    targets_u32 = _u32(targets, weight.shape[0], "targets")
    rt = _runtime()
    V, H = weight.shape
    dh = dw = None
    if grads:
        dh = torch.empty((n, H), dtype=torch.float32, device=hidden.device)
        dw = torch.empty((V, H), dtype=torch.float32, device=hidden.device)
    args = _CeArgs()
    args.hidden = _ref(hidden, "hidden")
    args.col_off = 0
    args.weight = _ref(weight, "weight")
    args.rows = rows_u32.ctypes_ptr
    args.targets = targets_u32.ctypes_ptr
    args.n = n
    args.reduction = 0 if reduction == "mean" else 1
    args.operands = operands_code
    args.chunk = chunk
    args.want_grads = 1 if grads else 0
    args.scale = scale
    if grads:
        args.dh = _ref(dh, "dh")
        args.dw = _ref(dw, "dw")
    loss = ctypes.c_double()
    per_row_out = torch.empty(n, dtype=torch.float64) if per_row else None
    err = ctypes.create_string_buffer(_ERR_LEN)
    _finish_torch_queue()
    status = rt.lib.tessl_cross_entropy_rows(
        rt.handle,
        ctypes.byref(args),
        ctypes.byref(loss),
        ctypes.cast(per_row_out.data_ptr(), ctypes.POINTER(ctypes.c_double)) if per_row_out is not None else None,
        err,
        _ERR_LEN,
    )
    # Keep the host arrays alive across the call.
    del rows_u32, targets_u32
    _runtime_status(rt, status, err)
    return loss.value, per_row_out, dh, dw


class _HostU32:
    """A host uint32 copy and a ctypes pointer into it."""

    def __init__(self, t: torch.Tensor):
        self.t = t.contiguous()
        self.ctypes_ptr = ctypes.cast(self.t.data_ptr(), ctypes.POINTER(ctypes.c_uint32))


def _index_tensor(v: torch.Tensor, name: str) -> torch.Tensor:
    """``v`` flattened to a CPU int64 tensor. Integer dtypes only: a float
    id (3.5) would truncate to 3 silently and a bool is a mask, not an index.
    An empty tensor of any dtype is fine (``torch.as_tensor([])`` is f32)."""
    if v.numel() and (v.dtype == torch.bool or v.dtype.is_floating_point or v.dtype.is_complex):
        raise TesslError(f"{name} must be an integer tensor, not {v.dtype}")
    return v.reshape(-1).to("cpu", torch.int64)


def _u32(v: torch.Tensor, bound: int, name: str) -> _HostU32:
    flat = _index_tensor(v, name)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= bound):
        raise TesslError(f"{name} must lie in [0, {bound})")
    return _HostU32(flat.to(torch.int32).view(torch.int32))


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, mask, reduction, chunk, operands):
        base, to_rows = _row_view(hidden)
        flat_mask = mask.reshape(-1).to("cpu")
        positions = flat_mask.nonzero().reshape(-1)
        if positions.numel() == 0:
            raise TesslError("the mask selects no positions (an empty selection has no mean)")
        rows = to_rows(positions)
        if int(rows.max()) >= base.shape[0]:
            # A column slice (hidden[..., c:]) whose last row ends before a
            # whole ld-wide row does: not expressible as [T, ld], so copy.
            base, to_rows = _row_view(hidden.contiguous())
            rows = to_rows(positions)
        tgt = targets.reshape(-1).to("cpu")[positions]
        H = weight.shape[1]
        if base.shape[1] < H:
            raise TesslError(f"hidden width {base.shape[1]} is less than the weight's {H}")
        want = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        loss, _, dh, dw = cross_entropy_rows(
            base, weight.contiguous(), rows, tgt, reduction=reduction, chunk=chunk, grads=want, operands=operands,
            per_row=False,
        )
        # Gradients of the loss itself (scale 1); backward scales them by the
        # upstream gradient, which is linear, so nothing is recomputed.
        ctx.save_for_backward(dh if dh is not None else torch.empty(0), dw if dw is not None else torch.empty(0), positions)
        ctx.hidden_shape = hidden.shape
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        return torch.tensor(loss, dtype=torch.float32, device=hidden.device)

    @staticmethod
    def backward(ctx, grad_out):
        dh, dw, positions = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        grad_hidden = grad_weight = None
        if ctx.needs_input_grad[0]:
            flat = torch.zeros(
                (math.prod(ctx.hidden_shape[:-1]), ctx.hidden_shape[-1]),
                dtype=ctx.hidden_dtype,
                device=dh.device,
            )
            flat[positions.to(dh.device)] = (dh * g).to(ctx.hidden_dtype)
            grad_hidden = flat.reshape(ctx.hidden_shape)
        if ctx.needs_input_grad[1]:
            # dW accumulated in f32 over the whole vocabulary walk, cast once.
            grad_weight = (dw * g).to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None, None, None


def cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
    chunk: int = 0,
    operands: str = "f32",
) -> torch.Tensor:
    """``F.cross_entropy(hidden[mask] @ weight.T, targets[mask])`` without
    forming the logits, differentiable in ``hidden`` and ``weight``.

    ``hidden`` is ``[..., H]`` on MPS (f32 or bf16; strided slices along the
    leading dimensions, like ``hidden[:, :-1]``, are read in place), ``weight``
    ``[V, H]``, ``targets`` integer ids shaped like ``hidden[..., 0]``, and
    ``mask`` a bool tensor of that shape selecting the supervised positions
    (all of them when ``None``). An empty selection is an error. The weight
    gradient is accumulated in f32 and cast to the weight's dtype once.
    ``operands`` is the GEMMs': ``"f32"`` (exact) or ``"bf16"`` (operands
    rounded to bf16, f32 accumulation).

    The loss is the same function as ``F.cross_entropy``, evaluated with the
    target left out of the log-sum-exp: with ``L = logsumexp(z[j != t])`` and
    ``margin = z[t] - L`` it is ``softplus(-margin)``, and the gradient is
    ``exp(z[j] - L) * sigmoid(-margin)`` for ``j != t`` and ``-sigmoid(-margin)``
    at the target. ``softmax - onehot`` computed directly rounds the target's
    gradient to 0 once the margin passes about 16.6 in f32 (Barron, "A Fix for
    Softmax and Cross-Entropy"); this form keeps it, and each row sums to 0.
    """
    chunk = _check_uint("chunk", chunk, 32)
    if mask is None:
        mask = torch.ones(hidden.shape[:-1], dtype=torch.bool)
    if tuple(mask.shape) != tuple(hidden.shape[:-1]) or tuple(targets.shape) != tuple(hidden.shape[:-1]):
        raise TesslError(
            f"targets {tuple(targets.shape)} and mask {tuple(mask.shape)} must be shaped like "
            f"hidden[..., 0] {tuple(hidden.shape[:-1])}"
        )
    if mask.dtype != torch.bool:
        raise TesslError(f"mask must be bool, not {mask.dtype}")
    _operands_code(operands)
    return _CrossEntropy.apply(hidden, weight, targets, mask, reduction, chunk, operands)


# ------------------------------------------------------ Qwen3.5 model ---

_READ_PARAMS, _WRITE_PARAMS = 0, 2
# Stored precision codes (TESSL_F32, TESSL_BF16).
_PRECISION_CODE = {"f32": 0, "bf16": 1}


class Qwen35(_Owned):
    """A Qwen3.5 text model trained by tessl's ``train_step``, with torch
    running the optimizer.

        model = tessl_torch.Qwen35(safetensors_path, config_json_path)
        params = model.parameters()            # an f32 copy, on MPS
        opt = torch.optim.AdamW(params.values(), lr=1e-5)
        grads = None
        loss = model.train_step(ids, operands="bf16")  # tessl: loss and every gradient
        grads = model.grads(into=grads)        # reused after the first step
        for name, g in grads.items():
            params[name].grad = g
        opt.step()
        model.load_parameters(params)          # write the update back

    Parameters and gradients are keyed by transformers' names below the text
    tower (``layers.3.mlp.gate_proj.weight``) and have transformers' shapes
    and values (the zero-centred norms as ``w``, as tessl stores them).
    Linear weights come back as transposed views of ``[in, out]``
    tensors, which is how tessl lays them out; ``load_parameters`` accepts
    any layout. With ``precision="f32"`` (the default) the model runs
    entirely in f32, the tied embedding included, so ``load_parameters`` is
    exact. ``operands="bf16"`` rounds only the GEMMs' operands; the default
    ``"f32"`` is exact, and it is the lane every stated parity bound is for.

    ``precision="bf16"`` stores the matrices, their gradients and each
    layer's saved input in bf16, with f32 arithmetic: half the memory, and
    the 4B's training fits a 48 GiB Mac with ``adamw_init`` keeping 8-bit
    moments. Its steps take ``operands="bf16"``, ``load_parameters`` rounds
    the matrices to nearest, and ``parameters()`` and ``grads()`` return the
    stored values widened to f32.

    The handle belongs to the thread that made it, like every tessl call.
    Each call synchronizes torch's MPS stream first and returns after tessl's
    queue is idle.
    """

    _free_name = "tessl_qwen35_free"
    _kind = "model"
    _handle = property(lambda self: self._live())

    def __init__(self, safetensors, config_json, prefix: str = "model.language_model.", precision: str = "f32"):
        if precision not in _PRECISION_CODE:
            raise TesslError(f"precision must be 'f32' or 'bf16', not {precision!r}")
        rt = _runtime()
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = ctypes.c_void_p()
        status = rt.lib.tessl_qwen35_load(
            rt.handle, str(safetensors).encode(), str(config_json).encode(), prefix.encode(),
            _PRECISION_CODE[precision], ctypes.byref(handle), err, _ERR_LEN,
        )
        _runtime_status(rt, status, err)
        self._rt = rt
        self._own(rt.lib, handle)
        rt._models.add(self)
        self._precision = precision
        count = ctypes.c_uint64()
        self._check(rt.lib.tessl_qwen35_param_count(handle, ctypes.byref(count), err, _ERR_LEN), err)
        table = []
        decay_excluded = []
        for i in range(count.value):
            info = _ParamInfo()
            self._check(rt.lib.tessl_qwen35_param_info(handle, i, ctypes.byref(info), err, _ERR_LEN), err)
            shape = tuple(info.shape[d] for d in range(info.ndim))
            table.append((info.name.decode(), shape, bool(info.transposed)))
            decay_excluded.append(bool(info.decay_excluded))
        self._table = table
        self._decay_excluded = decay_excluded
        self._hidden = dict((n, s) for n, s, _ in table)["norm.weight"][0]

    def _check(self, status: int, err):
        """Raise for a non-zero status. A panic (a bug in tessl) closes the
        model: the library says its handle should be freed."""
        if status != 0:
            if status == _STATUS_PANIC:
                self._close_after_panic()
                # The runtime the model ran on is dropped too: a bug in tessl
                # may have left it unusable. A live model never depends on the
                # Python runtime object's handle: the Rust model holds its own
                # Arc of the GpuRuntime (capi.rs `Handle::runtime`), and every
                # model call passes the model's handle, so closing the thread's
                # runtime cannot free anything a live model uses.
                self._rt._close_after_panic()
                raise _status_error(status, err)
            raise _error_or_poisoned(self._rt, status, err)

    @property
    def shapes(self) -> dict:
        """Every parameter's transformers shape, in the table's order."""
        return {name: shape for name, shape, _ in self._table}

    def _storage(self, device) -> list:
        """One dense f32 tensor per entry, laid out as tessl copies it."""
        return [
            torch.empty(tuple(reversed(shape)) if tr else shape, dtype=torch.float32, device=device)
            for _, shape, tr in self._table
        ]

    def _copy(self, direction: int, tensors: list):
        refs = (_TensorRef * len(tensors))(*[_ref(t, name) for t, (name, _, _) in zip(tensors, self._table)])
        err = ctypes.create_string_buffer(_ERR_LEN)
        _finish_torch_queue()
        self._check(self._rt.lib.tessl_qwen35_copy(self._handle, direction, refs, len(tensors), err, _ERR_LEN), err)

    def _check_names(self, what: str, d: dict) -> None:
        """Refuse ``d`` unless its keys are exactly the parameter names; ``what``
        leads the message."""
        names = [name for name, _, _ in self._table]
        missing = [n for n in names if n not in d]
        extra = [n for n in d if n not in set(names)]
        if missing or extra:
            raise TesslError(f"{what} missing {missing[:3]}{'...' if len(missing) > 3 else ''}, "
                             f"unexpected {extra[:3]}{'...' if len(extra) > 3 else ''}")

    def _staged(self, what: str, tensors: dict) -> list:
        """``tensors`` (name -> tensor of the parameter's shape, any dtype,
        device and layout) as tessl copies them in, all checked first."""
        self._check_names(f"{what}:", tensors)
        ts = []
        for name, shape, tr in self._table:
            p = tensors[name]
            if tuple(p.shape) != shape:
                raise TesslError(f"{what}: {name} must be {shape}, got {tuple(p.shape)}")
            p = p.detach()
            ts.append((p.t() if tr else p).to(device="mps", dtype=torch.float32).contiguous())
        return ts

    def _views(self, tensors: list) -> dict:
        return {name: (t.t() if tr else t) for t, (name, _, tr) in zip(tensors, self._table)}

    def parameters(self, device: str = "mps") -> dict:
        """A fresh f32 copy of every parameter, keyed by name."""
        ts = self._storage(torch.device(device))
        self._copy(_READ_PARAMS, ts)
        return self._views(ts)

    def load_parameters(self, params) -> None:
        """Set every parameter from ``params`` (name -> tensor of the
        parameter's shape, any dtype, device and layout). All of them are
        required, and all are checked before any is written.

        Under an AdamW state that keeps auxiliary state (``update="f32-master"``
        or ``"bf16-kahan"``) the written weights are not what the next
        ``adamw_step`` starts from: it rebuilds each bf16 matrix from the
        master (or compensation), so a write that must stick needs that state
        restored with it (``load_adamw_state`` with its ``"aux"``)."""
        self._copy(_WRITE_PARAMS, self._staged("load_parameters", params))

    def _entry(self, what: str, name: str):
        """``(index, shape, transposed)`` of the parameter ``name``."""
        if not isinstance(name, str):
            raise TesslError(f"{what}: the parameter name must be a str, got {type(name).__name__}")
        for i, (n, shape, tr) in enumerate(self._table):
            if n == name:
                return i, shape, tr
        raise TesslError(f"{what}: no parameter named {name!r}")

    def read_parameter(self, name: str, device: str = "cpu") -> torch.Tensor:
        """One parameter's value as a fresh f32 tensor of its shape. Only that
        parameter is copied, on the host, with no GPU copy and no wait on
        torch's queue: ``parameters()[name]`` copies the whole model (about
        8 GB of f32 on the 2B) to give you the same values."""
        i, shape, tr = self._entry("read_parameter", name)
        host = torch.empty(tuple(reversed(shape)) if tr else shape, dtype=torch.float32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(
            self._rt.lib.tessl_qwen35_read_parameter_host(
                self._handle, i, ctypes.cast(host.data_ptr(), ctypes.POINTER(ctypes.c_float)), host.numel(),
                err, _ERR_LEN,
            ),
            err,
        )
        return (host.t() if tr else host).to(device)

    def write_parameter(self, name: str, value: torch.Tensor) -> None:
        """Set one parameter from ``value`` (the parameter's shape; any dtype,
        device and layout). Only that parameter is written, on the host, as
        ``load_parameters`` would write it. Refused while a step is pending.

        Also refused, before anything moves, for a bf16 matrix whose AdamW
        state keeps auxiliary state (``update="f32-master"`` or
        ``"bf16-kahan"``): the next ``adamw_step`` rebuilds the weight from the
        master (or the Kahan compensation) and would silently overwrite this
        write. There is no per-parameter write of that state: restore the
        parameters and the state together (``load_parameters`` then
        ``load_adamw_state``, which carries ``"aux"``), or write before
        ``adamw_init``. f32 parameters, the other update rules and a model with
        no AdamW state take the write as asked."""
        i, shape, tr = self._entry("write_parameter", name)
        if tuple(value.shape) != shape:
            raise TesslError(f"write_parameter: {name} must be {shape}, got {tuple(value.shape)}")
        v = value.detach().to(device="cpu", dtype=torch.float32)
        host = (v.t() if tr else v).contiguous()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(
            self._rt.lib.tessl_qwen35_write_parameter_host(
                self._handle, i, ctypes.cast(host.data_ptr(), ctypes.POINTER(ctypes.c_float)), host.numel(),
                err, _ERR_LEN,
            ),
            err,
        )

    def describe(self) -> str:
        """One line naming the model's shapes and stored precision, then the
        AdamW state's update rule, moment storage and step count once
        ``adamw_init`` made it: what a run's log or checkpoint records."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        needed = ctypes.c_uint64()
        out = ctypes.create_string_buffer(1024)
        status = self._rt.lib.tessl_qwen35_describe(
            self._handle, out, len(out), ctypes.byref(needed), err, _ERR_LEN)
        if status != 0 and needed.value > len(out):
            out = ctypes.create_string_buffer(needed.value)
            status = self._rt.lib.tessl_qwen35_describe(
                self._handle, out, len(out), ctypes.byref(needed), err, _ERR_LEN)
        self._check(status, err)
        return out.value.decode()



from .embedgemma2 import EmbedGemma2  # noqa: E402  (after the definitions it imports from this module)

__all__ += ["EmbedGemma2"]
