"""EmbeddingGemma 2 text embeddings from Python, through the C ABI in
src/capi_embedgemma2.rs.

    from tessl_torch.embedgemma2 import EmbedGemma2

    model = EmbedGemma2.from_pretrained(snapshot_dir)   # a local snapshot directory
    emb = model.encode([[2, 4521, 1], [2, 17, 99, 1]])  # float32 [2, 768], rows L2-normalized
    emb = model.encode_texts(["how do I ...?"], tokenizer, prompt_name="query")

``encode`` takes token ids, so it needs nothing beyond the standard library,
numpy and the built library (``cargo build --release``, or ``TESSL_LIB``). The
ids are the tokenizer's, ``<bos>`` and ``<eos>`` and any task prompt included.
``encode_texts`` does that tokenization with a tokenizer *you* pass in: a
``tokenizers.Tokenizer`` or a ``transformers`` tokenizer, recognised by the
methods it has. Neither package is imported here, and neither is a dependency.

Images and audio need the towers, so load with ``multimodal=True``::

    model = EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)
    out = processor(images=images, return_tensors="np")      # transformers' processor
    emb = model.encode_images(out["pixel_values"], out["image_position_ids"])
    emb = model.embed_images(images, processor)              # the same, in one call
    emb = model.encode_audio(feats["input_features"], feats["input_features_mask"])

``encode_images`` and ``encode_audio`` take a processor's outputs (numpy arrays
or torch tensors, recognised by their methods and never imported) and
``embed_images`` / ``embed_audio`` call a processor you pass in, as
``encode_texts`` calls a tokenizer. ``transformers`` is not a dependency.
``operands="bf16"`` (multimodal loads) is the GEMM operands of the text encoder
and the vision tower; the audio tower is exact f32.

The model belongs to the thread that made it, like every tessl handle, and
each call returns after tessl's queue is idle. It reads no torch tensor and
writes none: the embeddings come back as a numpy array.
"""

from __future__ import annotations

import ctypes
import json
import os
import stat
from pathlib import Path

from . import (
    _ERR_LEN,
    _STATUS_PANIC,
    TesslError,
    _check_uint,
    _error_or_poisoned,
    _operands_code,
    _Owned,
    _runtime,
    _runtime_status,
    _status_error,
)

__all__ = ["EmbedGemma2"]

# The snapshot file that names the task prompts, as sentence-transformers reads it.
_PROMPTS_FILE = "config_sentence_transformers.json"
# Largest prompts file read; the checkpoint's is under 2 KiB.
_MAX_PROMPTS_BYTES = 1 << 20


def _declare(lib) -> None:
    """Give ctypes the signatures of the EmbeddingGemma 2 entries, once per
    library. A library built before they existed is named as such, not left to
    an ``AttributeError`` at the first call. A stand-in object (the host-only
    tests') carries no ctypes attributes and is used as it is."""
    if not isinstance(lib, ctypes.CDLL) or getattr(lib, "_tessl_embedgemma2_declared", False):
        return
    err_args = [ctypes.c_char_p, ctypes.c_size_t]
    u32p = ctypes.POINTER(ctypes.c_uint32)
    try:
        lib.tessl_embedgemma2_load.restype = ctypes.c_int32
        lib.tessl_embedgemma2_load.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p),
        ] + err_args
        lib.tessl_embedgemma2_free.restype = ctypes.c_int32
        lib.tessl_embedgemma2_free.argtypes = [ctypes.c_void_p]
        lib.tessl_embedgemma2_embedding_dim.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embedding_dim.argtypes = [ctypes.c_void_p, u32p] + err_args
        lib.tessl_embedgemma2_max_tokens.restype = ctypes.c_int32
        lib.tessl_embedgemma2_max_tokens.argtypes = [ctypes.c_void_p, u32p] + err_args
        lib.tessl_embedgemma2_set_max_tokens.restype = ctypes.c_int32
        lib.tessl_embedgemma2_set_max_tokens.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + err_args
        lib.tessl_embedgemma2_encode.restype = ctypes.c_int32
        lib.tessl_embedgemma2_encode.argtypes = [
            ctypes.c_void_p, u32p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
        ] + err_args
    except AttributeError:
        raise TesslError(
            f"{lib._name} has no tessl_embedgemma2_* entry points; it was built before they existed: "
            f"run `cargo build --release`"
        ) from None
    lib._tessl_embedgemma2_declared = True


def _declare_multimodal(lib) -> None:
    """Give ctypes the signatures of the multimodal entries, once per library,
    apart from :func:`_declare`: a library with the text entries and without
    these is named as such."""
    if not isinstance(lib, ctypes.CDLL) or getattr(lib, "_tessl_embedgemma2_mm_declared", False):
        return
    err_args = [ctypes.c_char_p, ctypes.c_size_t]
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    try:
        lib.tessl_embedgemma2_load_multimodal.restype = ctypes.c_int32
        lib.tessl_embedgemma2_load_multimodal.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        ] + err_args
        lib.tessl_embedgemma2_encode_images.restype = ctypes.c_int32
        lib.tessl_embedgemma2_encode_images.argtypes = [
            ctypes.c_void_p, f32p, ctypes.c_uint64, i32p, ctypes.c_uint64, u64p, ctypes.c_uint64,
            ctypes.c_uint32, f32p, ctypes.c_uint64,
        ] + err_args
        lib.tessl_embedgemma2_encode_audio.restype = ctypes.c_int32
        lib.tessl_embedgemma2_encode_audio.argtypes = [
            ctypes.c_void_p, f32p, ctypes.c_uint64, u8p, ctypes.c_uint64, u64p, ctypes.c_uint64,
            ctypes.c_uint32, f32p, ctypes.c_uint64,
        ] + err_args
    except AttributeError:
        raise TesslError(
            f"{lib._name} has no tessl_embedgemma2_load_multimodal / _encode_images / _encode_audio entry "
            f"points; it was built before they existed: run `cargo build --release`"
        ) from None
    lib._tessl_embedgemma2_mm_declared = True


def _numpy():
    """numpy, imported when first needed (torch, which this package imports
    anyway, depends on it)."""
    import numpy as np

    return np


def _ids_array(ids):
    """``ids`` (a non-empty sequence of non-empty sequences of integers in
    ``[0, 2**32)``) as the C ABI's CSR pair: a flat uint32 array and uint64
    offsets, ``len(ids) + 1`` long. Everything is checked before a ctypes field
    could wrap it (``-1`` would become ``2**32 - 1``)."""
    np = _numpy()
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__len__") or not hasattr(ids, "__iter__"):
        raise TesslError(f"ids must be a sequence of token-id sequences, not {type(ids).__name__}")
    if len(ids) == 0:
        raise TesslError("ids is empty: there is nothing to embed")
    parts = []
    for i, seq in enumerate(ids):
        if isinstance(seq, (str, bytes)) or not hasattr(seq, "__len__") or not hasattr(seq, "__iter__"):
            raise TesslError(f"ids[{i}] must be a sequence of integers, not {type(seq).__name__}")
        if len(seq) == 0:
            raise TesslError(f"ids[{i}] is empty: every sequence needs at least one token")
        try:
            arr = np.asarray(seq)
        except (ValueError, OverflowError) as e:
            raise TesslError(f"ids[{i}] is not a flat sequence of integers: {e}") from None
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise TesslError(
                f"ids[{i}] must be a flat sequence of integers (no bool, float or nesting), "
                f"got dtype {arr.dtype} with shape {arr.shape}"
            )
        if arr.min() < 0:
            raise TesslError(f"ids[{i}] has a negative token id {int(arr.min())}")
        if arr.max() >= 1 << 32:
            raise TesslError(f"ids[{i}] has token id {int(arr.max())}, outside [0, 2**32)")
        parts.append(arr.astype(np.uint32))
    flat = np.ascontiguousarray(np.concatenate(parts), dtype=np.uint32)
    offsets = np.zeros(len(parts) + 1, dtype=np.uint64)
    np.cumsum([len(p) for p in parts], dtype=np.uint64, out=offsets[1:])
    return flat, offsets


def _as_numpy(x):
    """``x`` as a numpy array: a torch tensor (recognised by ``detach``, never
    imported; bf16 and fp16 widened to f32 first, which numpy cannot hold),
    an array, or anything ``numpy.asarray`` takes."""
    np = _numpy()
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if str(getattr(x, "dtype", "")) in ("torch.bfloat16", "torch.float16") and hasattr(x, "float"):
            x = x.float()
        if hasattr(x, "numpy"):
            x = x.numpy()
    try:
        return np.asarray(x)
    except (ValueError, TypeError, OverflowError) as e:
        raise TesslError(f"cannot read the input as an array: {e}") from None


def _per_item(x, name: str, item_ndim: int) -> list:
    """``x`` as a list of per-item arrays of ``item_ndim`` dimensions. ``x`` is
    one item's array, a batch with a leading item axis (a processor's
    ``[N, ...]`` output), or a list or tuple of either (a ragged batch, or one
    processor output per sample). Items keep their own lengths."""
    if isinstance(x, (str, bytes)):
        raise TesslError(f"{name} must be an array or a list of arrays, not {type(x).__name__}")
    parts = list(x) if isinstance(x, (list, tuple)) else [x]
    items = []
    for k, part in enumerate(parts):
        arr = _as_numpy(part)
        if arr.ndim == item_ndim:
            items.append(arr)
        elif arr.ndim == item_ndim + 1:
            items.extend(arr[i] for i in range(arr.shape[0]))
        else:
            raise TesslError(
                f"{name}{f'[{k}]' if isinstance(x, (list, tuple)) else ''} has {arr.ndim} dimensions "
                f"(shape {arr.shape}); expected {item_ndim} for one item or {item_ndim + 1} with a leading item axis"
            )
    return items


def _csr_offsets(lengths: list):
    """``len(lengths) + 1`` uint64 offsets starting at 0."""
    np = _numpy()
    offsets = np.zeros(len(lengths) + 1, dtype=np.uint64)
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])
    return offsets


def _image_arrays(pixel_values, image_position_ids):
    """A processor's ``pixel_values`` and ``image_position_ids`` as the C
    ABI's flat float32 ``[rows, patch_dim]`` and int32 ``[rows, 2]`` arrays
    with uint64 row offsets per image. Everything Python can see is checked
    here (the library checks the rest before any GPU work)."""
    np = _numpy()
    px = _per_item(pixel_values, "pixel_values", 2)
    pos = _per_item(image_position_ids, "image_position_ids", 2)
    if not px:
        raise TesslError("pixel_values is empty: there is nothing to embed")
    if len(px) != len(pos):
        raise TesslError(f"{len(px)} images in pixel_values but {len(pos)} in image_position_ids")
    width = px[0].shape[1]
    for i, (a, b) in enumerate(zip(px, pos)):
        if a.dtype.kind != "f":
            raise TesslError(
                f"pixel_values[{i}] must be floating point (the processor's rescaled values), not {a.dtype}"
            )
        if b.dtype.kind not in "iu":
            raise TesslError(f"image_position_ids[{i}] must be integers, not {b.dtype}")
        if a.shape[0] == 0:
            raise TesslError(f"pixel_values[{i}] has no patches")
        if a.shape[1] != width:
            raise TesslError(f"pixel_values[{i}] has {a.shape[1]} columns, pixel_values[0] has {width}")
        if b.shape != (a.shape[0], 2):
            raise TesslError(
                f"image_position_ids[{i}] has shape {b.shape}; expected ({a.shape[0]}, 2) for the "
                f"{a.shape[0]} patches of pixel_values[{i}]"
            )
        if b.min() < -(1 << 31) or b.max() >= 1 << 31:
            raise TesslError(f"image_position_ids[{i}] has a value outside int32")
    flat_px = np.ascontiguousarray(np.concatenate(px), dtype=np.float32)
    flat_pos = np.ascontiguousarray(np.concatenate(pos), dtype=np.int32)
    return flat_px, flat_pos, _csr_offsets([a.shape[0] for a in px])


def _audio_arrays(input_features, input_features_mask):
    """A processor's ``input_features`` and ``input_features_mask`` as the C
    ABI's flat float32 ``[frames, mel_bins]`` and uint8 ``[frames]`` (0 or 1)
    arrays with uint64 frame offsets per clip."""
    np = _numpy()
    feats = _per_item(input_features, "input_features", 2)
    masks = _per_item(input_features_mask, "input_features_mask", 1)
    if not feats:
        raise TesslError("input_features is empty: there is nothing to embed")
    if len(feats) != len(masks):
        raise TesslError(f"{len(feats)} clips in input_features but {len(masks)} in input_features_mask")
    width = feats[0].shape[1]
    for i, (f, m) in enumerate(zip(feats, masks)):
        if f.dtype.kind != "f":
            raise TesslError(f"input_features[{i}] must be floating point, not {f.dtype}")
        if m.dtype.kind not in "biu":
            raise TesslError(f"input_features_mask[{i}] must be bool or 0/1 integers, not {m.dtype}")
        if f.shape[0] == 0:
            raise TesslError(f"input_features[{i}] has no frames")
        if f.shape[1] != width:
            raise TesslError(f"input_features[{i}] has {f.shape[1]} columns, input_features[0] has {width}")
        if m.shape != (f.shape[0],):
            raise TesslError(
                f"input_features_mask[{i}] has shape {m.shape}; expected ({f.shape[0]},) for the "
                f"{f.shape[0]} frames of input_features[{i}]"
            )
        if m.dtype.kind != "b" and ((m != 0) & (m != 1)).any():
            raise TesslError(f"input_features_mask[{i}] holds a value other than 0 and 1")
    flat_f = np.ascontiguousarray(np.concatenate(feats), dtype=np.float32)
    flat_m = np.ascontiguousarray(np.concatenate(masks).astype(np.uint8))
    return flat_f, flat_m, _csr_offsets([f.shape[0] for f in feats])


def _item_list(items, name: str) -> list:
    """``items`` as a non-empty list; a single string or a bare mapping is not
    a batch."""
    if isinstance(items, (str, bytes, dict)) or not hasattr(items, "__iter__"):
        raise TesslError(f"{name} must be a list of items, not {type(items).__name__}")
    items = list(items)
    if not items:
        raise TesslError(f"{name} is empty: there is nothing to embed")
    return items


def _processor_outputs(processor, kwargs: dict, *keys: str):
    """``processor(**kwargs, return_tensors="np")``, which must hold ``keys``.
    The processor is asked, not imported or type-checked."""
    if not callable(processor):
        raise TesslError(f"processor must be a callable transformers processor, not {type(processor).__name__}")
    out = processor(**kwargs, return_tensors="np")
    missing = [k for k in keys if k not in out]
    if missing:
        raise TesslError(f"the processor's output has no {', '.join(missing)}; is it the checkpoint's processor?")
    return out


def _token_ids(tokenizer, texts: list) -> list:
    """The token ids of ``texts`` with the tokenizer's special tokens, from a
    ``tokenizers.Tokenizer`` (which has ``encode_batch``) or a ``transformers``
    tokenizer (which is callable). Nothing is imported: the object is asked."""
    if hasattr(tokenizer, "encode_batch"):
        if getattr(tokenizer, "padding", None) is not None:
            raise TesslError("the tokenizer pads its output; encode_texts needs unpadded ids (call no_padding())")
        return [list(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=True)]
    if callable(tokenizer):
        batch = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        return [list(row) for row in batch["input_ids"]]
    raise TesslError(
        f"tokenizer must be a tokenizers.Tokenizer or a transformers tokenizer, not {type(tokenizer).__name__}"
    )


class EmbedGemma2(_Owned):
    """The EmbeddingGemma 2 text encoder, loaded from a Hugging Face snapshot
    directory (``config.json`` and the single-file ``model.safetensors`` of
    ``google/embeddinggemma-2``) and run by tessl in exact f32.

        model = EmbedGemma2.from_pretrained(snapshot_dir)
        emb = model.encode(ids)                       # [len(ids), 768], float32
        emb = model.encode(ids, truncate_dim=256)     # the Matryoshka prefix, re-normalized

    Each row is L2-normalized and equals sentence-transformers' ``encode`` of
    the same ids to ``1e-4`` (the bound ``tests/embedgemma2_model.rs`` holds the
    Rust model to). A sequence's embedding does not depend on the batch it is
    in beyond GEMM rounding. The batch is limited to 256 sequences and each
    sequence to ``max_tokens`` (8192 by default; up to 32768 when raised); a
    larger batch is split into forwards by tessl, not by the caller.

    The handle belongs to the thread that made it, like every tessl call.
    """

    _free_name = "tessl_embedgemma2_free"
    _kind = "embedgemma2 model"

    def __init__(self, snapshot_dir, multimodal=False, operands="f32"):
        path = os.fspath(snapshot_dir)
        if isinstance(path, bytes):
            raise TesslError("snapshot_dir must be a str or a path, not bytes")
        try:
            raw = path.encode("utf-8")
        except UnicodeEncodeError:
            raise TesslError(f"snapshot_dir {path!r} is not valid UTF-8") from None
        if not isinstance(multimodal, bool):
            raise TesslError(f"multimodal must be a bool, not {type(multimodal).__name__}")
        operands_code = _operands_code(operands)
        if operands_code != 0 and not multimodal:
            raise TesslError("operands applies to the text encoder and vision tower of a multimodal load: pass multimodal=True")
        rt = _runtime()
        _declare(rt.lib)
        if multimodal:
            _declare_multimodal(rt.lib)
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = ctypes.c_void_p()
        if multimodal:
            status = rt.lib.tessl_embedgemma2_load_multimodal(
                rt.handle, raw, len(raw), operands_code, ctypes.byref(handle), err, _ERR_LEN
            )
        else:
            status = rt.lib.tessl_embedgemma2_load(rt.handle, raw, len(raw), ctypes.byref(handle), err, _ERR_LEN)
        _runtime_status(rt, status, err)
        self._rt = rt
        self._multimodal = multimodal
        self._own(rt.lib, handle)
        rt._models.add(self)
        self._snapshot = Path(path)
        # None until read; False when the snapshot has no prompts file.
        self._prompt_config_cache = None
        dim = ctypes.c_uint32()
        self._check(rt.lib.tessl_embedgemma2_embedding_dim(handle, ctypes.byref(dim), err, _ERR_LEN), err)
        self._dim = dim.value

    @classmethod
    def from_pretrained(cls, snapshot_dir, multimodal=False, operands="f32") -> "EmbedGemma2":
        """Load the snapshot directory ``snapshot_dir`` (a local path; nothing
        is downloaded). With ``multimodal=True`` the vision and audio towers
        are loaded too (about 1.7 GB on the device against 0.8 GB), which
        :meth:`encode_images` and :meth:`encode_audio` need; ``operands`` is
        then the GEMM operands of the *text encoder and the vision tower*, ``"f32"``
        (exact) or ``"bf16"``. The audio tower is exact f32 under either."""
        return cls(snapshot_dir, multimodal=multimodal, operands=operands)

    def _check(self, status: int, err) -> None:
        """Raise for a non-zero status. A panic (a bug in tessl) closes the
        model and the thread's runtime: the library says the handle should be
        freed. A refusal that poisoned the runtime closes it and its models."""
        if status == 0:
            return
        if status == _STATUS_PANIC:
            self._close_after_panic()
            self._rt._close_after_panic()
            raise _status_error(status, err)
        raise _error_or_poisoned(self._rt, status, err)

    @property
    def embedding_dim(self) -> int:
        """The width of an untruncated embedding, and the largest ``truncate_dim``."""
        self._live()
        return self._dim

    @property
    def max_tokens(self) -> int:
        """The most tokens one sequence may have; ``encode`` refuses longer ones.
        Assign to change it (1 to 32768)."""
        handle = self._live()
        out = ctypes.c_uint32()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_embedgemma2_max_tokens(handle, ctypes.byref(out), err, _ERR_LEN), err)
        return out.value

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        handle = self._live()
        v = _check_uint("max_tokens", value, 32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_embedgemma2_set_max_tokens(handle, v, err, _ERR_LEN), err)

    def encode(self, ids, truncate_dim=None):
        """Embed token-id sequences: ``ids`` is a list of lists of ints, each
        sequence non-empty, with the tokenizer's ``<bos>``/``<eos>`` and any
        prompt already in it. Returns a float32 numpy array
        ``[len(ids), truncate_dim or embedding_dim]``, rows L2-normalized, in
        the order given. ``truncate_dim`` (``None`` for the full width) must be
        in ``1..=embedding_dim``. Raises :class:`TesslError` for anything the
        library refuses, before any GPU work."""
        handle = self._live()
        np = _numpy()
        flat, offsets = _ids_array(ids)
        if truncate_dim is None:
            code, dim = 0, self._dim
        else:
            code = _check_uint("truncate_dim", truncate_dim, 32)
            if code == 0:
                raise TesslError("truncate_dim must be positive (None means the full width)")
            dim = code
        out = np.empty((len(offsets) - 1, dim), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        status = self._rt.lib.tessl_embedgemma2_encode(
            handle,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), flat.size,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(offsets) - 1,
            code,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size,
            err, _ERR_LEN,
        )
        self._check(status, err)
        return out

    def _multimodal_handle(self, what: str):
        handle = self._live()
        if not self._multimodal:
            raise TesslError(
                f"{what} needs the vision and audio towers: load with "
                f"EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)"
            )
        return handle

    def _truncate_code(self, truncate_dim):
        if truncate_dim is None:
            return 0, self._dim
        code = _check_uint("truncate_dim", truncate_dim, 32)
        if code == 0:
            raise TesslError("truncate_dim must be positive (None means the full width)")
        return code, code

    def encode_images(self, pixel_values, image_position_ids, truncate_dim=None):
        """Embed images from a transformers processor's outputs: one row per
        image, float32 ``[n_images, truncate_dim or embedding_dim]``, rows
        L2-normalized, the embedding of the processor's image sequence
        (``<bos> <boi> <image> x S <eoi> <eos>``) with the vision tower's soft
        tokens in place of the placeholders.

        ``pixel_values`` is ``[N, P, 768]`` (one image: ``[P, 768]``) and
        ``image_position_ids`` ``[N, P, 2]`` (``[P, 2]``), numpy arrays or torch
        tensors (recognised by their methods, never imported), exactly as
        ``processor(images=...)`` returns them, padding patches ``(-1, -1)``
        included. A list of per-image arrays of different ``P`` works too. The
        image embeddings do not depend on the batch beyond GEMM rounding. The
        model must have been loaded with ``multimodal=True``. Raises
        :class:`TesslError` for anything Python or the library refuses, before
        any GPU work."""
        handle = self._multimodal_handle("encode_images")
        np = _numpy()
        flat_px, flat_pos, offsets = _image_arrays(pixel_values, image_position_ids)
        code, dim = self._truncate_code(truncate_dim)
        n = len(offsets) - 1
        out = np.empty((n, dim), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        status = self._rt.lib.tessl_embedgemma2_encode_images(
            handle,
            flat_px.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), flat_px.size,
            flat_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), flat_pos.size,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), n,
            code,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size,
            err, _ERR_LEN,
        )
        self._check(status, err)
        return out

    def encode_audio(self, input_features, input_features_mask, truncate_dim=None):
        """Embed audio clips from a transformers processor's outputs: one row
        per clip, float32 ``[n_clips, truncate_dim or embedding_dim]``, rows
        L2-normalized, the embedding of the processor's audio sequence
        (``<bos> <boa> <audio> x S <eoa> <eos>``) with the audio tower's soft
        tokens in place of the placeholders.

        ``input_features`` is ``[N, F, 128]`` (one clip: ``[F, 128]``) and
        ``input_features_mask`` ``[N, F]`` (``[F]``; bool, or 0/1), as
        ``processor(audio=...)`` returns them, padded frames included (a clip's
        ``F`` is its padded length; the mask says which frames are real). A
        list of per-clip arrays of different ``F`` works too. A clip whose
        mask leaves no soft token is refused. Otherwise as
        :meth:`encode_images`."""
        handle = self._multimodal_handle("encode_audio")
        np = _numpy()
        flat_f, flat_m, offsets = _audio_arrays(input_features, input_features_mask)
        code, dim = self._truncate_code(truncate_dim)
        n = len(offsets) - 1
        out = np.empty((n, dim), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        status = self._rt.lib.tessl_embedgemma2_encode_audio(
            handle,
            flat_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), flat_f.size,
            flat_m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), flat_m.size,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), n,
            code,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size,
            err, _ERR_LEN,
        )
        self._check(status, err)
        return out

    def embed_images(self, images, processor, truncate_dim=None):
        """Embed ``images`` (a list of PIL images, or whatever ``processor``
        takes as ``images=``) with a processor *you* supply, one embedding per
        image: ``processor(images=list(images), return_tensors="np")`` and
        :meth:`encode_images` on its ``pixel_values`` and
        ``image_position_ids``. ``processor`` is the checkpoint's
        ``AutoProcessor.from_pretrained(snapshot)``, recognised by being
        callable and returning those two keys; nothing is imported."""
        self._multimodal_handle("embed_images")
        out = _processor_outputs(processor, {"images": _item_list(images, "images")}, "pixel_values", "image_position_ids")
        return self.encode_images(out["pixel_values"], out["image_position_ids"], truncate_dim=truncate_dim)

    def embed_audio(self, clips, processor, truncate_dim=None):
        """Embed audio ``clips`` (a list of 1-D float arrays at the processor's
        sampling rate, 16 kHz) with a processor you supply, one embedding per
        clip: ``processor(audio=list(clips), return_tensors="np")`` and
        :meth:`encode_audio` on its ``input_features`` and
        ``input_features_mask``. As :meth:`embed_images`."""
        self._multimodal_handle("embed_audio")
        out = _processor_outputs(processor, {"audio": _item_list(clips, "clips")}, "input_features", "input_features_mask")
        return self.encode_audio(out["input_features"], out["input_features_mask"], truncate_dim=truncate_dim)

    def _prompt_config(self):
        """``(prompts, default_prompt_name)`` from the snapshot's
        ``config_sentence_transformers.json``, read once, or ``None`` when the
        snapshot has no such file. A file that is there and wrong is an error."""
        if self._prompt_config_cache is None:
            path = self._snapshot / _PROMPTS_FILE
            try:
                with open(path, "rb") as f:
                    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                        raise TesslError(f"{path} is not a regular file")
                    data = f.read(_MAX_PROMPTS_BYTES + 1)
            except FileNotFoundError:
                self._prompt_config_cache = False
                return None
            except OSError as e:
                raise TesslError(f"cannot read {path}: {e}") from None
            if len(data) > _MAX_PROMPTS_BYTES:
                raise TesslError(f"{path} is larger than {_MAX_PROMPTS_BYTES} bytes")
            try:
                config = json.loads(data)
            except ValueError as e:
                raise TesslError(f"{path} is not JSON: {e}") from None
            prompts = config.get("prompts") if isinstance(config, dict) else None
            if not isinstance(prompts, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()
            ):
                raise TesslError(f'{path} has no "prompts" mapping of names to strings')
            default = config.get("default_prompt_name")
            if default is not None and default not in prompts:
                raise TesslError(f"{path}: default_prompt_name {default!r} is not one of its prompts")
            self._prompt_config_cache = (prompts, default)
        return self._prompt_config_cache or None

    def prompts(self) -> dict:
        """The task prompts of the snapshot's ``config_sentence_transformers.json``
        (``{"query": "task: search result | query: ", "document": "title: none | text: ", ...}``)."""
        self._live()
        config = self._prompt_config()
        if config is None:
            raise TesslError(f"{self._snapshot} has no {_PROMPTS_FILE}")
        return dict(config[0])

    def encode_texts(self, texts, tokenizer, prompt_name=None, truncate_dim=None):
        """Embed ``texts`` (a list of str) with a tokenizer you supply.

        ``tokenizer`` is a ``tokenizers.Tokenizer`` (``Tokenizer.from_file(snapshot
        / "tokenizer.json")``) or a ``transformers`` tokenizer
        (``AutoTokenizer.from_pretrained(snapshot)``); it is asked for ids with
        special tokens, so each text becomes ``<bos> ... <eos>``. With
        ``prompt_name`` (a key of :meth:`prompts`, such as ``"query"`` or
        ``"document"``) the snapshot's prompt is prepended to each text first,
        as sentence-transformers' ``encode(prompt_name=...)`` does; without
        one, the snapshot's ``default_prompt_name`` applies if it names one
        (this checkpoint's is null). Returns what :meth:`encode` returns."""
        self._live()
        if isinstance(texts, (str, bytes)) or not hasattr(texts, "__iter__"):
            raise TesslError("texts must be a list of str, not a single string")
        texts = list(texts)
        if not texts:
            raise TesslError("texts is empty: there is nothing to embed")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TesslError(f"texts[{i}] must be a str, not {type(t).__name__}")
        config = self._prompt_config()
        name = prompt_name if prompt_name is not None else (config[1] if config else None)
        if name is not None:
            if config is None:
                raise TesslError(f"prompt_name {name!r} was given but {self._snapshot} has no {_PROMPTS_FILE}")
            if name not in config[0]:
                raise TesslError(f"unknown prompt_name {name!r}; the snapshot has {sorted(config[0])}")
            texts = [config[0][name] + t for t in texts]
        return self.encode(_token_ids(tokenizer, texts), truncate_dim=truncate_dim)
