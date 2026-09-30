"""torch bindings for tessl's Metal kernels, through the C ABI in src/capi.rs.

    import tessl_torch
    loss = tessl_torch.cross_entropy(hidden, weight, targets, mask)
    loss.backward()

Only the standard library and torch are needed: the library is loaded with
ctypes, and MPS tensors are handed over as the MTLBuffer behind their storage
(``untyped_storage().data_ptr()``, which is what ATen's own
``getMTLBufferStorage`` bit-casts) plus a byte offset.

Build the library first (``cargo build --release``) or point ``TESSL_LIB`` at
a ``libtessl.dylib``. It loads its kernels from the metallib path baked in at
build time, so keep the build's ``target/`` directory in place.

Every call synchronizes torch's MPS stream before handing buffers to tessl
and returns only after tessl's queue has finished, so the two queues never
touch a buffer at the same time. That is correct, not fast: each call waits
for the GPU twice.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from pathlib import Path

import torch

__all__ = ["cross_entropy", "cross_entropy_rows", "TesslError", "library_path"]

_ABI_VERSION = 1
_MAX_DIMS = 4
_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_ERR_LEN = 1024


class TesslError(RuntimeError):
    """A tessl call refused its arguments or failed on the device."""


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
        ("chunk", ctypes.c_uint32),
        ("want_grads", ctypes.c_uint32),
        ("scale", ctypes.c_float),
        ("dh", _TensorRef),
        ("dw", _TensorRef),
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
        lib.tessl_runtime_free.restype = None
        lib.tessl_runtime_free.argtypes = [ctypes.c_void_p]
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
        _lib = lib
        return lib


class _Runtime:
    """One tessl runtime per thread: tessl's runtime is thread-affine."""

    def __init__(self):
        lib = _load()
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = lib.tessl_runtime_new(err, _ERR_LEN)
        if not handle:
            raise TesslError(err.value.decode(errors="replace"))
        self.lib = lib
        self.handle = ctypes.c_void_p(handle)

    def __del__(self):
        lib, handle = getattr(self, "lib", None), getattr(self, "handle", None)
        if lib is not None and handle:
            lib.tessl_runtime_free(handle)


_local = threading.local()


def _runtime() -> _Runtime:
    rt = getattr(_local, "rt", None)
    if rt is None:
        rt = _Runtime()
        _local.rt = rt
    return rt


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
):
    """The raw call: ``hidden`` is a contiguous ``[T, ld]`` MPS tensor whose
    first ``H`` columns are the hidden states, ``weight`` a contiguous
    ``[V, H]`` one; ``rows`` and ``targets`` are the ``n`` supervised row
    indices and token ids. Returns ``(loss, per_row, dh, dw)``: the loss as
    a Python float, per-row losses as a CPU float64 tensor, and with
    ``grads`` the f32 ``[n, H]`` and ``[V, H]`` gradients of ``scale * loss``
    (``None`` otherwise).
    """
    if reduction not in ("mean", "sum"):
        raise TesslError(f"reduction must be 'mean' or 'sum', not {reduction!r}")
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
    args.chunk = chunk
    args.want_grads = 1 if grads else 0
    args.scale = scale
    if grads:
        args.dh = _ref(dh, "dh")
        args.dw = _ref(dw, "dw")
    loss = ctypes.c_double()
    per_row = torch.empty(n, dtype=torch.float64)
    err = ctypes.create_string_buffer(_ERR_LEN)
    # tessl reads buffers torch's queue may still be writing: finish it.
    torch.mps.synchronize()
    status = rt.lib.tessl_cross_entropy_rows(
        rt.handle,
        ctypes.byref(args),
        ctypes.byref(loss),
        ctypes.cast(per_row.data_ptr(), ctypes.POINTER(ctypes.c_double)),
        err,
        _ERR_LEN,
    )
    # Keep the host arrays alive across the call.
    del rows_u32, targets_u32
    if status != 0:
        raise TesslError(err.value.decode(errors="replace"))
    return loss.value, per_row, dh, dw


class _HostU32:
    """A host uint32 copy and a ctypes pointer into it."""

    def __init__(self, t: torch.Tensor):
        self.t = t.contiguous()
        self.ctypes_ptr = ctypes.cast(self.t.data_ptr(), ctypes.POINTER(ctypes.c_uint32))


def _u32(v: torch.Tensor, bound: int, name: str) -> _HostU32:
    flat = v.reshape(-1).to("cpu", torch.int64)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= bound):
        raise TesslError(f"{name} must lie in [0, {bound})")
    return _HostU32(flat.to(torch.int32).view(torch.int32))


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, mask, reduction, chunk):
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
            base, weight.contiguous(), rows, tgt, reduction=reduction, chunk=chunk, grads=want
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
        return grad_hidden, grad_weight, None, None, None, None


def cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
    chunk: int = 0,
) -> torch.Tensor:
    """``F.cross_entropy(hidden[mask] @ weight.T, targets[mask])`` without
    forming the logits, differentiable in ``hidden`` and ``weight``.

    ``hidden`` is ``[..., H]`` on MPS (f32 or bf16; strided slices along the
    leading dimensions, like ``hidden[:, :-1]``, are read in place), ``weight``
    ``[V, H]``, ``targets`` integer ids shaped like ``hidden[..., 0]``, and
    ``mask`` a bool tensor of that shape selecting the supervised positions
    (all of them when ``None``). An empty selection is an error. The weight
    gradient is accumulated in f32 and cast to the weight's dtype once.
    """
    if mask is None:
        mask = torch.ones(hidden.shape[:-1], dtype=torch.bool)
    if tuple(mask.shape) != tuple(hidden.shape[:-1]) or tuple(targets.shape) != tuple(hidden.shape[:-1]):
        raise TesslError(
            f"targets {tuple(targets.shape)} and mask {tuple(mask.shape)} must be shaped like "
            f"hidden[..., 0] {tuple(hidden.shape[:-1])}"
        )
    if mask.dtype != torch.bool:
        raise TesslError(f"mask must be bool, not {mask.dtype}")
    return _CrossEntropy.apply(hidden, weight, targets, mask, reduction, chunk)
