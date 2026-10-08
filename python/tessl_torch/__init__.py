"""torch bindings for tessl's Metal kernels, through the C ABI in src/capi.rs.

    import tessl_torch
    loss = tessl_torch.cross_entropy(hidden, weight, targets, mask)
    loss.backward()

    model = tessl_torch.Qwen35(safetensors_path, config_json_path)
    loss = model.train_step(ids)          # a whole Qwen3.5 step in tessl
    grads = model.grads()                 # for a torch optimizer

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

__all__ = [
    "cross_entropy",
    "cross_entropy_rows",
    "chunk_gated_delta_rule",
    "clip_coef",
    "patch_transformers_qwen3_5",
    "Qwen35",
    "TesslError",
    "library_path",
]

_ABI_VERSION = 10
_MAX_DIMS = 6
_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_ERR_LEN = 1024
# GEMM operand codes: exact f32, or operands rounded to bf16 with f32 accumulation.
_OPERANDS_CODE = {"f32": 0, "bf16": 1}
# What Qwen35.train_forward scores: the causal-LM loss, or chosen rows.
_SUPERVISE_CAUSAL, _SUPERVISE_ROWS = 0, 1


class TesslError(RuntimeError):
    """A tessl call refused its arguments or failed on the device."""


def _operands_code(operands: str) -> int:
    if operands not in _OPERANDS_CODE:
        raise TesslError(f"operands must be 'f32' or 'bf16', not {operands!r}")
    return _OPERANDS_CODE[operands]


def clip_coef(total_norm: float, max_norm: float) -> float:
    """``torch.nn.utils.clip_grad_norm_``'s coefficient for a global norm,
    in its f32 arithmetic: ``max_norm / (total_norm + 1e-6)``, at most 1.
    Pass it as ``Qwen35.adamw_step(grad_scale=...)`` and scale any gradients
    outside tessl by it too."""
    norm = torch.tensor(total_norm, dtype=torch.float32)
    return torch.clamp(max_norm / (norm + 1e-6), max=1.0).item()


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


class _GdnArgs(ctypes.Structure):
    _fields_ = [
        ("batch", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("heads", ctypes.c_uint32),
        ("v_dim", ctypes.c_uint32),
    ] + [
        (name, _TensorRef)
        for name in (
            "q", "k", "v", "g", "beta", "s0", "ckpt", "o", "s_fin",
            "d_o", "d_fin", "dq", "dk", "dv", "dg", "dbeta", "ds0",
        )
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
        for name in ("tessl_gdn_train_forward", "tessl_gdn_train_backward"):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int32
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GdnArgs), ctypes.c_char_p, ctypes.c_size_t]
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
        lib.tessl_qwen35_train_step.restype = ctypes.c_int32
        lib.tessl_qwen35_train_step.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
        ] + err_args
        u32p = ctypes.POINTER(ctypes.c_uint32)
        lib.tessl_qwen35_train_forward.restype = ctypes.c_int32
        lib.tessl_qwen35_train_forward.argtypes = [
            ctypes.c_void_p, u32p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, u32p, u32p, ctypes.c_uint64,
            ctypes.c_float, ctypes.POINTER(ctypes.c_double),
        ] + err_args
        lib.tessl_qwen35_hidden.restype = ctypes.c_int32
        lib.tessl_qwen35_hidden.argtypes = [
            ctypes.c_void_p, u32p, ctypes.c_uint64, ctypes.POINTER(_TensorRef),
        ] + err_args
        lib.tessl_qwen35_train_backward.restype = ctypes.c_int32
        lib.tessl_qwen35_train_backward.argtypes = [
            ctypes.c_void_p, u32p, ctypes.c_uint64, ctypes.POINTER(_TensorRef), ctypes.c_uint32,
        ] + err_args
        lib.tessl_qwen35_train_discard.restype = ctypes.c_int32
        lib.tessl_qwen35_train_discard.argtypes = [ctypes.c_void_p] + err_args
        lib.tessl_qwen35_copy.restype = ctypes.c_int32
        lib.tessl_qwen35_copy.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_TensorRef), ctypes.c_uint64,
        ] + err_args
        lib.tessl_qwen35_adamw_init.restype = ctypes.c_int32
        lib.tessl_qwen35_adamw_init.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32,
        ] + err_args
        lib.tessl_qwen35_adamw_free.restype = ctypes.c_int32
        lib.tessl_qwen35_adamw_free.argtypes = [ctypes.c_void_p] + err_args
        lib.tessl_qwen35_describe.restype = ctypes.c_int32
        lib.tessl_qwen35_describe.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        ] + err_args
        lib.tessl_qwen35_adamw_step.restype = ctypes.c_int32
        lib.tessl_qwen35_adamw_step.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
        ] + err_args
        lib.tessl_qwen35_adamw_step_count.restype = ctypes.c_int32
        lib.tessl_qwen35_adamw_step_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)] + err_args
        lib.tessl_qwen35_adamw_set_step_count.restype = ctypes.c_int32
        lib.tessl_qwen35_adamw_set_step_count.argtypes = [ctypes.c_void_p, ctypes.c_uint64] + err_args
        lib.tessl_qwen35_grad_sq_norm.restype = ctypes.c_int32
        lib.tessl_qwen35_grad_sq_norm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)] + err_args
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
        # From a foreign thread (interpreter shutdown) the library refuses and
        # leaks the runtime rather than drop it on the wrong thread.
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
    operands: str = "f32",
):
    """The raw call: ``hidden`` is a contiguous ``[T, ld]`` MPS tensor whose
    first ``H`` columns are the hidden states, ``weight`` a contiguous
    ``[V, H]`` one; ``rows`` and ``targets`` are the ``n`` supervised row
    indices and token ids. Returns ``(loss, per_row, dh, dw)``: the loss as
    a Python float, per-row losses as a CPU float64 tensor, and with
    ``grads`` the f32 ``[n, H]`` and ``[V, H]`` gradients of ``scale * loss``
    (``None`` otherwise). ``operands`` is the GEMMs': ``"f32"`` (exact) or
    ``"bf16"`` (operands rounded to bf16, f32 accumulation).
    """
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
            base, weight.contiguous(), rows, tgt, reduction=reduction, chunk=chunk, grads=want, operands=operands
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
    _operands_code(operands)
    return _CrossEntropy.apply(hidden, weight, targets, mask, reduction, chunk, operands)


# ------------------------------------------------------ gated delta rule ---

_GDN_DK = 128
_GDN_BV = 16
_GDN_CKPT = 64
_NULL_REF = _TensorRef()


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).contiguous()


def _gdn_args(B, T, H, Dv, **refs) -> _GdnArgs:
    a = _GdnArgs()
    a.batch, a.seq, a.heads, a.v_dim = B, T, H, Dv
    for name, t in refs.items():
        setattr(a, name, _NULL_REF if t is None else _ref(t, name))
    return a


def _gdn_call(fn_name: str, args: _GdnArgs):
    rt = _runtime()
    err = ctypes.create_string_buffer(_ERR_LEN)
    torch.mps.synchronize()
    status = getattr(rt.lib, fn_name)(rt.handle, ctypes.byref(args), err, _ERR_LEN)
    if status != 0:
        raise TesslError(err.value.decode(errors="replace"))


class _GdnChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state):
        B, T, H, Dk = q.shape
        Dv = v.shape[-1]
        qf, kf, vf, gf, bf = (_f32(x) for x in (q, k, v, g, beta))
        s0 = None if initial_state is None else _f32(initial_state)
        dev = q.device
        o = torch.empty((B, T, H, Dv), dtype=torch.float32, device=dev)
        s_fin = torch.empty((B, H, Dk, Dv), dtype=torch.float32, device=dev)
        nc = -(-T // _GDN_CKPT)
        ckpt = torch.empty((B, H, nc, Dk, Dv), dtype=torch.float32, device=dev)
        _gdn_call(
            "tessl_gdn_train_forward",
            _gdn_args(B, T, H, Dv, q=qf, k=kf, v=vf, g=gf, beta=bf, s0=s0, ckpt=ckpt, o=o, s_fin=s_fin),
        )
        # The inputs as given (bf16 stays bf16) and one state per 64 tokens:
        # everything else the backward needs it recomputes.
        ctx.save_for_backward(q, k, v, g, beta, initial_state if initial_state is not None else torch.empty(0), ckpt)
        ctx.has_s0 = initial_state is not None
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, None if initial_state is None else initial_state.dtype)
        return o.to(v.dtype), s_fin

    @staticmethod
    def backward(ctx, d_o, d_fin):
        q, k, v, g, beta, s0, ckpt = ctx.saved_tensors
        B, T, H, Dk = q.shape
        Dv = v.shape[-1]
        qf, kf, vf, gf, bf = (_f32(x) for x in (q, k, v, g, beta))
        s0f = _f32(s0) if ctx.has_s0 else None
        dev = q.device
        d_of = _f32(d_o) if d_o is not None else torch.zeros((B, T, H, Dv), dtype=torch.float32, device=dev)
        d_finf = _f32(d_fin) if d_fin is not None else None
        dq = torch.empty((B, T, H, Dk), dtype=torch.float32, device=dev)
        dk = torch.empty_like(dq)
        dv = torch.empty((B, T, H, Dv), dtype=torch.float32, device=dev)
        dg = torch.empty((B, T, H), dtype=torch.float32, device=dev)
        dbeta = torch.empty_like(dg)
        ds0 = torch.empty((B, H, Dk, Dv), dtype=torch.float32, device=dev) if ctx.has_s0 else None
        _gdn_call(
            "tessl_gdn_train_backward",
            _gdn_args(
                B, T, H, Dv, q=qf, k=kf, v=vf, g=gf, beta=bf, s0=s0f, ckpt=ckpt,
                d_o=d_of, d_fin=d_finf, dq=dq, dk=dk, dv=dv, dg=dg, dbeta=dbeta, ds0=ds0,
            ),
        )
        tq, tk, tv, tg, tb, ts = ctx.dtypes
        return (
            dq.to(tq), dk.to(tk), dv.to(tv), dg.to(tg), dbeta.to(tb),
            ds0.to(ts) if ds0 is not None else None,
        )


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
):
    """transformers' ``torch_chunk_gated_delta_rule``, on tessl's Metal kernels.

    Same arguments and results: ``query``/``key`` ``[B, T, H, 128]``,
    ``value`` ``[B, T, H, Dv]`` (``Dv`` a multiple of 16), ``g`` (the log
    decay) and ``beta`` ``[B, T, H]``; returns ``(out [B, T, H, Dv],
    final_state [B, H, 128, Dv] or None)``, differentiable in every input and
    the initial state. The forward keeps only its inputs and one state per 64
    tokens for the backward. ``chunk_size`` does not change the result (the
    kernels checkpoint every 64 tokens regardless). Only the l2-normalized
    form Qwen3.5 uses is implemented, and not ragged batches.
    """
    if not use_qk_l2norm_in_kernel:
        raise TesslError("tessl's GDN kernels implement use_qk_l2norm_in_kernel=True only")
    if cu_seqlens is not None:
        raise TesslError("ragged batches (cu_seqlens) are not supported")
    if query.dim() != 4 or query.shape != key.shape or query.shape[-1] != _GDN_DK:
        raise TesslError(f"query and key must both be [B, T, H, {_GDN_DK}], got {tuple(query.shape)} and {tuple(key.shape)}")
    B, T, H, _ = query.shape
    if value.dim() != 4 or tuple(value.shape[:3]) != (B, T, H) or value.shape[-1] % _GDN_BV != 0:
        raise TesslError(f"value must be [B, T, H, Dv] with Dv a multiple of {_GDN_BV}, got {tuple(value.shape)}")
    for name, t in (("g", g), ("beta", beta)):
        if tuple(t.shape) != (B, T, H):
            raise TesslError(f"{name} must be [B, T, H] = {(B, T, H)}, got {tuple(t.shape)}")
    if initial_state is not None and tuple(initial_state.shape) != (B, H, _GDN_DK, value.shape[-1]):
        raise TesslError(f"initial_state must be {(B, H, _GDN_DK, value.shape[-1])}, got {tuple(initial_state.shape)}")
    out, final = _GdnChunk.apply(query, key, value, g, beta, initial_state)
    return out, (final if output_final_state else None)


def patch_transformers_qwen3_5():
    """Route transformers' Qwen3.5 chunked GDN through tessl.

    Replaces ``modeling_qwen3_5.torch_chunk_gated_delta_rule`` (the torch
    fallback transformers uses when ``fla`` is absent, as on macOS), which the
    layer looks up at call time. Returns the function it replaced.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    previous = m.torch_chunk_gated_delta_rule
    m.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    return previous


# ------------------------------------------------------ Qwen3.5 training ---

_READ_PARAMS, _READ_GRADS, _WRITE_PARAMS = 0, 1, 2
_READ_ADAMW_M, _READ_ADAMW_V, _WRITE_ADAMW_M, _WRITE_ADAMW_V = 3, 4, 5, 6
_READ_ADAMW_AUX, _WRITE_ADAMW_AUX = 7, 8
# Stored precision codes (TESSL_F32, TESSL_BF16), and the AdamW update rules
# and moment storages by the names tessl's AdamW::describe prints.
_PRECISION_CODE = {"f32": 0, "bf16": 1}
_UPDATE_CODE = {"f32": 0, "f32-master": 1, "bf16-kahan": 2, "bf16-stochastic": 3}
_MOMENTS_CODE = {"f32": 0, "bf16": 1, "block8": 2}
# The rules that keep auxiliary state: an f32 master, or a Kahan compensation.
_AUX_RULES = ("f32-master", "bf16-kahan")


class Qwen35:
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
        if status != 0:
            raise TesslError(err.value.decode(errors="replace"))
        self._rt = rt
        self._handle = handle
        self._precision = precision
        # What adamw_init made the AdamW state with, or None.
        self._adamw_config = None
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
        self._has_grads = False
        # A step between train_forward and train_backward, and whether the
        # bank held a finished step's gradients before it (a discard leaves
        # the bank as it was).
        self._pending = False
        self._grads_before_pending = False
        self._hidden = dict((n, s) for n, s, _ in table)["norm.weight"][0]

    @staticmethod
    def _check(status: int, err):
        if status != 0:
            raise TesslError(err.value.decode(errors="replace"))

    def __del__(self):
        rt, handle = getattr(self, "_rt", None), getattr(self, "_handle", None)
        if rt is not None and handle:
            rt.lib.tessl_qwen35_free(handle)

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
        torch.mps.synchronize()
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

    def grads(self, device: str = "mps", into: dict | None = None) -> dict:
        """The last ``train_step``'s gradients, laid out as ``parameters()``.

        With ``into`` (a dict an earlier ``grads()`` or ``parameters()``
        returned, such as the optimizer's ``.grad`` tensors), the gradients
        are written into those tensors and ``into`` is returned: no new
        memory, where a fresh call allocates another copy of every gradient
        while the previous one is usually still alive. Every tensor is checked
        before any is written."""
        if not self._has_grads:
            raise TesslError("no gradients yet; call train_step first")
        if into is None:
            ts = self._storage(torch.device(device))
            self._copy(_READ_GRADS, ts)
            return self._views(ts)
        self._check_names("grads(into=...):", into)
        ts = []
        for name, shape, tr in self._table:
            t = into[name]
            if tuple(t.shape) != shape or t.dtype != torch.float32 or t.device.type != "mps":
                raise TesslError(f"grads(into=...): {name} must be f32 {shape} on mps, "
                                 f"got {t.dtype} {tuple(t.shape)} on {t.device}")
            storage = t.t() if tr else t
            if not storage.is_contiguous():
                raise TesslError(f"grads(into=...): {name} is not laid out as grads() returns it"
                                 f"{' (a transposed view of a contiguous [in, out] tensor)' if tr else ''}")
            ts.append(storage)
        self._copy(_READ_GRADS, ts)
        return into

    def load_parameters(self, params) -> None:
        """Set every parameter from ``params`` (name -> tensor of the
        parameter's shape, any dtype, device and layout). All of them are
        required, and all are checked before any is written."""
        self._copy(_WRITE_PARAMS, self._staged("load_parameters", params))

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

    def adamw_init(self, update: str | None = None, moments: str = "f32", seed: int = 0) -> None:
        """Make AdamW state inside tessl: moments zeroed, step count 0. With
        it a training loop needs no torch copy of the parameters or gradients.

        An f32 model takes ``update="f32"`` (the default for it): torch's
        AdamW, bit for bit. A bf16 model names how its bf16 weights take
        their updates: ``"f32-master"`` (an f32 master of each weight),
        ``"bf16-kahan"`` (a bf16 compensation of what rounding dropped) or
        ``"bf16-stochastic"`` (stochastic rounding from ``seed``, the only
        rule that takes one). ``moments`` is ``"f32"`` (twice the parameters'
        memory, 16 GB on the 2B), ``"bf16"`` or ``"block8"`` (8-bit codes,
        one f32 scale per 256 elements)."""
        if update is None:
            if self._precision != "f32":
                raise TesslError("adamw_init: a bf16 model takes update='f32-master', 'bf16-kahan' or "
                                 "'bf16-stochastic'")
            update = "f32"
        if update not in _UPDATE_CODE:
            raise TesslError(f"adamw_init: update must be one of {sorted(_UPDATE_CODE)}, not {update!r}")
        if moments not in _MOMENTS_CODE:
            raise TesslError(f"adamw_init: moments must be one of {sorted(_MOMENTS_CODE)}, not {moments!r}")
        seed = int(seed)
        if not 0 <= seed < 2 ** 64:
            raise TesslError(f"adamw_init: seed {seed} is outside [0, 2**64)")
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(
            self._rt.lib.tessl_qwen35_adamw_init(
                self._handle, _UPDATE_CODE[update], seed, _MOMENTS_CODE[moments], err, _ERR_LEN),
            err,
        )
        self._adamw_config = {"update": update, "seed": seed, "moments": moments}

    def adamw_free(self) -> None:
        """Drop the AdamW state, if any."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_qwen35_adamw_free(self._handle, err, _ERR_LEN), err)
        self._adamw_config = None

    @property
    def adamw_step_count(self) -> int:
        """AdamW steps taken so far (torch's ``state["step"]``)."""
        count = ctypes.c_uint64()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_qwen35_adamw_step_count(self._handle, ctypes.byref(count), err, _ERR_LEN), err)
        return count.value

    def adamw_state(self, device: str = "mps") -> dict:
        """A checkpoint of the AdamW state: ``{"step": int, "config": {"update",
        "seed", "moments"}, "exp_avg": {name: tensor}, "exp_avg_sq": {name:
        tensor}}``, the moments laid out as ``parameters()`` (f32, decoded from
        bf16 or 8-bit storage) and named as torch's AdamW names them, plus
        ``"aux"`` (the f32 masters or the Kahan compensations, widened) under
        a rule that keeps them. Saved with the parameters, it resumes the run
        bit for bit through ``load_adamw_state``."""
        if self._adamw_config is None:
            raise TesslError("adamw_state: no AdamW state; call adamw_init first")
        state = {"step": self.adamw_step_count, "config": dict(self._adamw_config)}
        keys = [("exp_avg", _READ_ADAMW_M), ("exp_avg_sq", _READ_ADAMW_V)]
        if self._adamw_config["update"] in _AUX_RULES:
            keys.append(("aux", _READ_ADAMW_AUX))
        for key, direction in keys:
            ts = self._storage(torch.device(device))
            self._copy(direction, ts)
            state[key] = self._views(ts)
        return state

    def load_adamw_state(self, state: dict) -> None:
        """Restore what ``adamw_state`` returned (or torch's AdamW state for
        the same parameters, rearranged by name, into an f32 model's f32
        AdamW) into the AdamW state that ``adamw_init()`` made. A checkpoint
        without ``"config"`` is torch's AdamW: f32 rule and moments. A
        checkpoint made under another configuration is refused, as is one
        whose ``"aux"`` the rule needs and lacks or does not need and has.
        Everything is checked before anything is written."""
        required = {"step", "exp_avg", "exp_avg_sq"}
        if not required <= set(state) or set(state) - required - {"config", "aux"}:
            raise TesslError(
                f"load_adamw_state: want keys step, exp_avg and exp_avg_sq (and config and aux as adamw_state "
                f"writes them), got {sorted(state)}")
        if self._adamw_config is None:
            raise TesslError("load_adamw_state: no AdamW state; call tessl_qwen35_adamw_init first")
        saved = state.get("config", {"update": "f32", "seed": 0, "moments": "f32"})
        if saved != self._adamw_config:
            raise TesslError(f"load_adamw_state: the checkpoint's AdamW is {saved}, this model's "
                             f"{self._adamw_config}; call adamw_init with the checkpoint's configuration")
        wants_aux = self._adamw_config["update"] in _AUX_RULES
        if wants_aux != ("aux" in state):
            raise TesslError(f"load_adamw_state: update {self._adamw_config['update']!r} "
                             f"{'needs' if wants_aux else 'keeps no'} aux state, and the checkpoint "
                             f"{'has none' if wants_aux else 'has some'}")
        step = int(state["step"])
        if step < 0:
            raise TesslError(f"load_adamw_state: step {step} is negative")
        m = self._staged("load_adamw_state: exp_avg", state["exp_avg"])
        v = self._staged("load_adamw_state: exp_avg_sq", state["exp_avg_sq"])
        aux = self._staged("load_adamw_state: aux", state["aux"]) if wants_aux else None
        self._copy(_WRITE_ADAMW_M, m)
        self._copy(_WRITE_ADAMW_V, v)
        if aux is not None:
            self._copy(_WRITE_ADAMW_AUX, aux)
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_qwen35_adamw_set_step_count(self._handle, step, err, _ERR_LEN), err)

    def grad_sq_norm(self) -> float:
        """The sum of squares of every gradient the last ``train_step`` left:
        the square of the global norm ``clip_grad_norm_`` takes. Add the
        squares of any gradients outside tessl (a head of your own) before
        forming ``clip_coef``."""
        if not self._has_grads:
            raise TesslError("no gradients yet; call train_step first")
        out = ctypes.c_double()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_qwen35_grad_sq_norm(self._handle, ctypes.byref(out), err, _ERR_LEN), err)
        return out.value

    def adamw_step(self, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay=0.01,
                   grad_scale: float = 1.0) -> None:
        """One ``torch.optim.AdamW`` step (amsgrad and maximize off) on every
        parameter, in tessl, from the last ``train_step``'s gradients.

        ``weight_decay`` is a float, applied to every parameter except those
        transformers' Trainer excludes (every norm and ``linear_attn.dt_bias``),
        or a dict giving every parameter name its own. ``grad_scale``
        multiplies every gradient before the update, as ``clip_grad_norm_``
        scales ``.grad``: pass ``clip_coef(norm, max_norm)``. The gradients
        ``grads()`` reads stay unscaled. Call ``adamw_init()`` first."""
        if isinstance(weight_decay, dict):
            self._check_names("adamw_step: weight_decay is", weight_decay)
            wd = [float(weight_decay[name]) for name, _, _ in self._table]
        else:
            wd = [0.0 if ex else float(weight_decay) for ex in self._decay_excluded]
        beta1, beta2 = betas
        arr = (ctypes.c_float * len(wd))(*wd)
        err = ctypes.create_string_buffer(_ERR_LEN)
        torch.mps.synchronize()
        self._check(
            self._rt.lib.tessl_qwen35_adamw_step(
                self._handle, float(lr), float(beta1), float(beta2), float(eps), float(grad_scale), arr, len(wd), err,
                _ERR_LEN,
            ),
            err,
        )

    def _ids(self, ids) -> "_HostU32":
        ids = torch.as_tensor(ids)
        if ids.dim() != 1:
            raise TesslError(f"ids must be 1-D (one sequence), got shape {tuple(ids.shape)}")
        return _u32(ids, 1 << 32, "ids")

    def train_forward(self, ids, operands: str = "f32", positions=None, targets=None, scale: float = 1.0) -> float:
        """The forward and loss of one step, kept for ``hidden()`` and
        ``train_backward()``; returns the loss.

        With ``positions`` and ``targets`` (1-D, equal length, positions
        distinct; both empty is allowed), the loss is the sum of the
        cross-entropies of the hidden state at ``positions[i]`` against token
        ``targets[i]``, and the gradients are those of ``scale`` times it: a
        batch mean over ``N`` supervised rows split across several sequences
        is ``scale=1/N`` on each. Without them it is transformers' causal-LM
        loss, as ``train_step``. In transformers' shifted frame, the position
        that predicts ``ids[p + 1]`` is ``p``."""
        code = _operands_code(operands)
        host = self._ids(ids)
        n_ids = host.t.numel()
        if (positions is None) != (targets is None):
            raise TesslError("train_forward: give positions and targets together")
        if positions is None:
            sup, pos, tgt, n = _SUPERVISE_CAUSAL, None, None, 0
        else:
            pos = _u32(torch.as_tensor(positions), 1 << 32, "positions")
            tgt = _u32(torch.as_tensor(targets), 1 << 32, "targets")
            if pos.t.numel() != tgt.t.numel():
                raise TesslError(f"train_forward: {pos.t.numel()} positions but {tgt.t.numel()} targets")
            sup, n = _SUPERVISE_ROWS, pos.t.numel()
        loss = ctypes.c_double()
        err = ctypes.create_string_buffer(_ERR_LEN)
        torch.mps.synchronize()
        self._check(
            self._rt.lib.tessl_qwen35_train_forward(
                self._handle, host.ctypes_ptr, n_ids, code, sup,
                pos.ctypes_ptr if n else None, tgt.ctypes_ptr if n else None, n, float(scale),
                ctypes.byref(loss), err, _ERR_LEN,
            ),
            err,
        )
        self._grads_before_pending, self._has_grads, self._pending = self._has_grads, False, True
        return loss.value

    def hidden(self, positions) -> torch.Tensor:
        """Rows ``positions`` of the pending step's final-norm output
        (transformers' ``last_hidden_state``), a new f32 ``[n, hidden]``
        tensor on MPS, for a loss outside tessl."""
        pos = _u32(torch.as_tensor(positions), 1 << 32, "positions")
        n = pos.t.numel()
        out = torch.empty((n, self._hidden), dtype=torch.float32, device="mps")
        if n == 0:
            if not self._pending:
                raise TesslError("hidden: no step is pending; call train_forward first")
            return out
        err = ctypes.create_string_buffer(_ERR_LEN)
        torch.mps.synchronize()
        ref = _ref(out, "out")
        self._check(
            self._rt.lib.tessl_qwen35_hidden(self._handle, pos.ctypes_ptr, n, ctypes.byref(ref), err, _ERR_LEN),
            err,
        )
        return out

    def train_backward(self, dh=None, positions=None, accumulate: bool = False) -> None:
        """The pending step's backward: every parameter's gradient for
        ``grads()``, over the previous ones or, with ``accumulate``, added to
        them. ``dh`` (``[n, hidden]``, any dtype and device) is the gradient
        of a loss outside tessl at rows ``positions`` of what ``hidden()``
        reads, added to the step's own; rows of a repeated position are
        summed first. A refused call keeps the step pending
        (``train_discard()`` drops it)."""
        if (dh is None) != (positions is None):
            raise TesslError("train_backward: give dh and positions together")
        err = ctypes.create_string_buffer(_ERR_LEN)
        if dh is None:
            pos_ptr, n, ref = None, 0, None
        else:
            positions = torch.as_tensor(positions).reshape(-1).to("cpu", torch.int64)
            if tuple(dh.shape) != (positions.numel(), self._hidden):
                raise TesslError(
                    f"train_backward: dh must be [{positions.numel()}, {self._hidden}], got {tuple(dh.shape)}"
                )
            dh32 = dh.detach().to(device="mps", dtype=torch.float32)
            uniq, inv = torch.unique(positions, return_inverse=True)
            if uniq.numel() != positions.numel():
                dh32 = torch.zeros((uniq.numel(), self._hidden), dtype=torch.float32, device="mps").index_add_(
                    0, inv.to("mps"), dh32
                )
                positions = uniq
            dh32 = dh32.contiguous()
            pos = _u32(positions, 1 << 32, "positions")
            n = pos.t.numel()
            pos_ptr = pos.ctypes_ptr if n else None
            ref = ctypes.byref(_ref(dh32, "dh")) if n else None
        torch.mps.synchronize()
        self._check(
            self._rt.lib.tessl_qwen35_train_backward(self._handle, pos_ptr, n, ref, int(accumulate), err, _ERR_LEN),
            err,
        )
        self._has_grads, self._pending = True, False

    def train_discard(self) -> None:
        """Drop the pending step, if any, without its backward; the
        gradients are then as they were before it."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_qwen35_train_discard(self._handle, err, _ERR_LEN), err)
        if self._pending:
            self._has_grads, self._pending = self._grads_before_pending, False

    def train_step(self, ids, operands: str = "f32") -> float:
        """One training step on one sequence of token ids (a 1-D tensor or a
        sequence of ints): returns transformers' causal-LM loss and keeps
        every parameter's gradient for ``grads()``. The forward keeps only
        each layer's input; each layer reruns just before its backward.
        ``operands`` is the GEMMs': ``"f32"`` (exact) or ``"bf16"`` (operands
        rounded to bf16, f32 accumulation; weights, activations and
        gradients stay f32)."""
        code = _operands_code(operands)
        host = self._ids(ids)
        loss = ctypes.c_double()
        err = ctypes.create_string_buffer(_ERR_LEN)
        torch.mps.synchronize()
        self._has_grads = False
        status = self._rt.lib.tessl_qwen35_train_step(
            self._handle, host.ctypes_ptr, host.t.numel(), code, ctypes.byref(loss), err, _ERR_LEN,
        )
        del host
        self._check(status, err)
        self._has_grads = True
        return loss.value
