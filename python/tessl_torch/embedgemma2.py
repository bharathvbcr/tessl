"""EmbeddingGemma 2 embeddings from Python, through the C ABI in
src/capi_embedgemma2.rs.

    from tessl_torch.embedgemma2 import EmbedGemma2

    model = EmbedGemma2.from_pretrained(snapshot_dir)   # a local snapshot directory
    emb = model.encode_texts(["how do I ...?"], prompt_name="query")   # float32 [1, 768]
    emb = model.encode([[2, 4521, 1], [2, 17, 99, 1]])                 # token ids, [2, 768]

Every front end runs in Rust, from the snapshot's own files: the tokenizer
(``tokenizer.json`` and its sidecars, the task prompts of
``config_sentence_transformers.json``) and the preprocessing
(``processor_config.json``). Nothing here needs ``transformers``, ``tokenizers``
or a processor; the requirements are numpy and the built library
(``cargo build --release``, or ``TESSL_LIB``). The rows are L2-normalized and
agree with sentence-transformers' to ``1e-4`` (the tests hold them to it).

Images, audio and video need the towers, so load with ``multimodal=True``::

    model = EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)
    emb = model.embed_images([rgb_uint8_hwc, gray_uint8_hw])    # arrays, 1 to 4 channels
    emb = model.embed_audio([pcm_float32_16khz])                # mono clips at 16 kHz
    emb = model.embed_video(frames_uint8_thwc, fps=24.0, timestamps=True)
    emb = model.media([{"ids": ids, "images": [rgb], "audio": [pcm]}])   # ids of your own

``tokenize`` and ``decode`` expose the tokenizer, ``prompts()`` the named task
prompts. A tokenizer or processor of your own is still welcome: ``encode_texts``
takes ``tokenizer=``, and ``encode_images`` / ``encode_audio`` take a
processor's output tensors (numpy arrays or torch tensors, recognised by their
methods and never imported). ``operands="bf16"`` (multimodal loads) is the GEMM
operands of the text encoder and both towers.

The model belongs to the thread that made it, like every tessl handle, and
each call returns after tessl's queue is idle. It reads no torch tensor and
writes none: the embeddings come back as a numpy array.
"""

from __future__ import annotations

import ctypes
import numbers
import os

from . import (
    _ERR_LEN,
    _STATUS_PANIC,
    TesslError,
    _check_uint,
    _error_or_poisoned,
    _flag,
    _operands_code,
    _Owned,
    _runtime,
    _runtime_status,
    _status_error,
)

__all__ = ["EmbedGemma2"]

# The extractor's sampling rate; the library refuses audio at another rate.
_SAMPLING_RATE = 16000


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


def _declare_front_ends(lib) -> None:
    """Give ctypes the signatures of the entries that run the snapshot's own
    tokenizer, prompts and preprocessing, once per library. A library without
    them is named as such."""
    if not isinstance(lib, ctypes.CDLL) or getattr(lib, "_tessl_embedgemma2_raw_declared", False):
        return
    err_args = [ctypes.c_char_p, ctypes.c_size_t]
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    h = ctypes.c_void_p
    u64 = ctypes.c_uint64
    u32 = ctypes.c_uint32
    try:
        lib.tessl_embedgemma2_prompt_count.restype = ctypes.c_int32
        lib.tessl_embedgemma2_prompt_count.argtypes = [h, u32p] + err_args
        lib.tessl_embedgemma2_prompt_name.restype = ctypes.c_int32
        lib.tessl_embedgemma2_prompt_name.argtypes = [h, u32, u8p, u64, u64p] + err_args
        lib.tessl_embedgemma2_prompt_get.restype = ctypes.c_int32
        lib.tessl_embedgemma2_prompt_get.argtypes = [h, ctypes.c_char_p, u64, u8p, u64, u64p] + err_args
        lib.tessl_embedgemma2_default_prompt.restype = ctypes.c_int32
        lib.tessl_embedgemma2_default_prompt.argtypes = [h, i32p] + err_args
        lib.tessl_embedgemma2_tokenize.restype = ctypes.c_int32
        lib.tessl_embedgemma2_tokenize.argtypes = [
            h, u8p, u64, u64p, u64, ctypes.c_char_p, u64, u32p, u64, u64p, u64p,
        ] + err_args
        lib.tessl_embedgemma2_decode.restype = ctypes.c_int32
        lib.tessl_embedgemma2_decode.argtypes = [h, u32p, u64, u32, u8p, u64, u64p] + err_args
        lib.tessl_embedgemma2_embed_texts.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embed_texts.argtypes = [
            h, u8p, u64, u64p, u64, ctypes.c_char_p, u64, u32, f32p, u64,
        ] + err_args
        lib.tessl_embedgemma2_embed_images.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embed_images.argtypes = [h, u8p, u64, u32p, u64, u32, f32p, u64] + err_args
        lib.tessl_embedgemma2_embed_audio.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embed_audio.argtypes = [h, f32p, u64, u64p, u64, u32, u32, f32p, u64] + err_args
        lib.tessl_embedgemma2_embed_videos.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embed_videos.argtypes = [
            h, u8p, u64, u32p, u64p, f64p, u64, u32, u32, f32p, u64,
        ] + err_args
        lib.tessl_embedgemma2_embed_media.restype = ctypes.c_int32
        lib.tessl_embedgemma2_embed_media.argtypes = [
            h, u32p, u64, u64p, u64,
            u8p, u64, u32p, u64, u64p,
            f32p, u64, u64p, u64, u64p, u32,
            u8p, u64, u32p, u64p, f64p, u64, u64p,
            u32, f32p, u64,
        ] + err_args
    except AttributeError:
        raise TesslError(
            f"{lib._name} has no tessl_embedgemma2_embed_texts / _tokenize / _embed_images / ... entry points; "
            f"it was built before they existed: run `cargo build --release`"
        ) from None
    lib._tessl_embedgemma2_raw_declared = True


def _numpy():
    """numpy, imported when first needed (torch, which this package imports
    anyway, depends on it)."""
    import numpy as np

    return np


def _ptr(array, ctype):
    """``array``'s data as a ``POINTER(ctype)``."""
    return array.ctypes.data_as(ctypes.POINTER(ctype))


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
    return flat, _csr_offsets([len(p) for p in parts])


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


def _text_list(texts) -> list:
    """``texts`` as a non-empty list of str; a single string is not a batch."""
    if isinstance(texts, (str, bytes)) or not hasattr(texts, "__iter__"):
        raise TesslError("texts must be a list of str, not a single string")
    texts = list(texts)
    if not texts:
        raise TesslError("texts is empty: there is nothing to embed")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TesslError(f"texts[{i}] must be a str, not {type(t).__name__}")
    return texts


def _utf8_texts(texts):
    """``texts`` as the C ABI's one UTF-8 byte array and ``len(texts) + 1``
    uint64 offsets (an empty text is two offsets alike)."""
    np = _numpy()
    encoded = []
    for i, t in enumerate(texts):
        try:
            encoded.append(t.encode("utf-8"))
        except UnicodeEncodeError:
            raise TesslError(f"texts[{i}] is not valid UTF-8 (a lone surrogate)") from None
    flat = np.frombuffer(b"".join(encoded), dtype=np.uint8) if encoded else np.empty(0, dtype=np.uint8)
    return flat, _csr_offsets([len(e) for e in encoded])


def _name_arg(prompt_name):
    """``prompt_name`` as ``(bytes or None, length)`` for the C ABI; ``None``
    is the snapshot's default prompt."""
    if prompt_name is None:
        return None, 0
    if not isinstance(prompt_name, str):
        raise TesslError(f"prompt_name must be a str or None, not {type(prompt_name).__name__}")
    try:
        raw = prompt_name.encode("utf-8")
    except UnicodeEncodeError:
        raise TesslError(f"prompt_name {prompt_name!r} is not valid UTF-8") from None
    return raw, len(raw)


def _dim_fits(name: str, shape) -> None:
    """Each of ``shape`` fits the C ABI's uint32 dimension (``astype`` would wrap)."""
    if max(shape) >= 1 << 32:
        raise TesslError(f"{name} has a dimension past 2**32: shape {tuple(shape)}")


def _uint8_array(x, name: str):
    """``x`` as a numpy array of uint8: nothing else is accepted, because a
    float or wider integer cast to uint8 wraps silently."""
    arr = _as_numpy(x)
    if arr.dtype != _numpy().uint8:
        raise TesslError(f"{name} must be uint8 (0 to 255), not {arr.dtype}")
    return arr


def _raw_image_list(images, name: str = "images") -> list:
    """``images`` as a list of uint8 ``[H, W, C]`` arrays, C in 1 to 4: a list or
    tuple of images, one ``[H, W, C]`` or ``[H, W]`` array (one image), or a
    ``[N, H, W, C]`` array. An ``[H, W]`` image is gray. Anything else the
    library refuses by name."""
    np = _numpy()
    if isinstance(images, (str, bytes, dict)):
        raise TesslError(f"{name} must be arrays, not {type(images).__name__}")
    if isinstance(images, (list, tuple)):
        parts = list(images)
    else:
        arr = _as_numpy(images)
        parts = list(arr) if arr.ndim == 4 else [arr]
    out = []
    for i, part in enumerate(parts):
        arr = _uint8_array(part, f"{name}[{i}]")
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise TesslError(
                f"{name}[{i}] has {arr.ndim} dimensions (shape {arr.shape}); expected [H, W, C] or [H, W]"
            )
        _dim_fits(f"{name}[{i}]", arr.shape)
        out.append(np.ascontiguousarray(arr))
    return out


def _image_buffers(arrays: list):
    """The C ABI's flat uint8 pixels and ``[n, 3]`` uint32 ``(H, W, C)`` table."""
    np = _numpy()
    if not arrays:
        return np.empty(0, dtype=np.uint8), np.empty((0, 3), dtype=np.uint32)
    flat = np.concatenate([a.reshape(-1) for a in arrays])
    dims = np.array([a.shape for a in arrays], dtype=np.uint32)
    return np.ascontiguousarray(flat), np.ascontiguousarray(dims)


def _raw_clip_list(clips, name: str = "audio") -> list:
    """``clips`` as a list of 1-D float32 arrays: a list or tuple of clips, one
    1-D array, or a 2-D ``[N, samples]`` array. Floating point only (an int16
    clip would need scaling this binding cannot guess)."""
    np = _numpy()
    if isinstance(clips, (str, bytes, dict)):
        raise TesslError(f"{name} must be arrays, not {type(clips).__name__}")
    if isinstance(clips, (list, tuple)):
        parts = list(clips)
    else:
        arr = _as_numpy(clips)
        parts = list(arr) if arr.ndim == 2 else [arr]
    out = []
    for i, part in enumerate(parts):
        arr = _as_numpy(part)
        if arr.ndim != 1:
            raise TesslError(f"{name}[{i}] has {arr.ndim} dimensions (shape {arr.shape}); expected a mono 1-D clip")
        if arr.dtype.kind != "f":
            raise TesslError(f"{name}[{i}] must be floating point PCM in [-1, 1], not {arr.dtype}")
        out.append(np.ascontiguousarray(arr, dtype=np.float32))
    return out


def _clip_buffers(clips: list):
    """The C ABI's flat float32 samples and uint64 sample offsets."""
    np = _numpy()
    flat = np.concatenate(clips) if clips else np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(flat), _csr_offsets([len(c) for c in clips])


def _raw_video(frames, name: str):
    """One video as a uint8 ``[T, H, W, 3]`` array: that array, or a list of
    ``[H, W, 3]`` frames."""
    np = _numpy()
    if isinstance(frames, (str, bytes, dict)):
        raise TesslError(f"{name} must be frames, not {type(frames).__name__}")
    if isinstance(frames, (list, tuple)):
        if not frames:
            raise TesslError(f"{name} has no frames")
        parts = [_uint8_array(f, f"{name}[{i}]") for i, f in enumerate(frames)]
        shapes = {p.shape for p in parts}
        if len(shapes) != 1:
            raise TesslError(f"{name}: the frames have different shapes {sorted(shapes)}")
        arr = np.stack(parts)
    else:
        arr = _uint8_array(frames, name)
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise TesslError(f"{name} has shape {arr.shape}; expected [frames, H, W, 3] (decoded RGB frames)")
    if arr.shape[0] == 0:
        raise TesslError(f"{name} has no frames")
    _dim_fits(name, arr.shape)
    return np.ascontiguousarray(arr)


def _video_buffers(videos: list, fps: list):
    """The C ABI's flat uint8 frames, ``[n, 2]`` uint32 ``(H, W)`` table, uint64
    frame offsets and float64 source rates."""
    np = _numpy()
    if not videos:
        return (
            np.empty(0, dtype=np.uint8),
            np.empty((0, 2), dtype=np.uint32),
            _csr_offsets([]),
            np.empty(0, dtype=np.float64),
        )
    flat = np.concatenate([v.reshape(-1) for v in videos])
    dims = np.array([v.shape[1:3] for v in videos], dtype=np.uint32)
    return (
        np.ascontiguousarray(flat),
        np.ascontiguousarray(dims),
        _csr_offsets([v.shape[0] for v in videos]),
        np.array(fps, dtype=np.float64),
    )


def _fps_list(fps, n: int) -> list:
    """One source frame rate per video: ``fps`` is a number (every video) or
    one per video. Positivity is the library's to check."""
    if isinstance(fps, (list, tuple)):
        values = list(fps)
        if len(values) != n:
            raise TesslError(f"{len(values)} frame rates for {n} videos")
    else:
        values = [fps] * n
    out = []
    for i, f in enumerate(values):
        if isinstance(f, bool) or not isinstance(f, numbers.Real):
            raise TesslError(f"fps[{i}] must be a number, not {type(f).__name__}")
        out.append(float(f))
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
        emb = model.encode_texts(["a text"])          # [1, 768], float32
        emb = model.encode(ids)                       # [len(ids), 768], float32
        emb = model.encode(ids, truncate_dim=256)     # the Matryoshka prefix, re-normalized

    Each row is L2-normalized and equals sentence-transformers' ``encode`` of
    the same text to ``1e-4`` (the bound ``tests/embedgemma2_model.rs`` holds the
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
        _declare_front_ends(rt.lib)
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
        dim = ctypes.c_uint32()
        self._check(rt.lib.tessl_embedgemma2_embedding_dim(handle, ctypes.byref(dim), err, _ERR_LEN), err)
        self._dim = dim.value

    @classmethod
    def from_pretrained(cls, snapshot_dir, multimodal=False, operands="f32") -> "EmbedGemma2":
        """Load the snapshot directory ``snapshot_dir`` (a local path; nothing
        is downloaded). With ``multimodal=True`` the vision and audio towers
        are loaded too (about 1.7 GB on the device against 0.8 GB), which
        the image, audio, video and media methods need; ``operands`` is
        then the GEMM operands of the *text encoder and both towers*, ``"f32"``
        (exact) or ``"bf16"``. The tokenizer and the preprocessing settings are
        read from the snapshot on the first call that needs them."""
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

    def _truncate_code(self, truncate_dim):
        """``(the C ABI's truncate_dim, the output width)``."""
        if truncate_dim is None:
            return 0, self._dim
        code = _check_uint("truncate_dim", truncate_dim, 32)
        if code == 0:
            raise TesslError("truncate_dim must be positive (None means the full width)")
        return code, code

    def _embed(self, call, n: int, truncate_dim):
        """One embedding call: ``call(code, out_pointer, out_len, err, err_len)``
        returns the C status, and ``n`` rows of the output come back."""
        np = _numpy()
        code, dim = self._truncate_code(truncate_dim)
        out = np.empty((n, dim), dtype=np.float32)
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(call(code, _ptr(out, ctypes.c_float), out.size, err, _ERR_LEN), err)
        return out

    def encode(self, ids, truncate_dim=None):
        """Embed token-id sequences: ``ids`` is a list of lists of ints, each
        sequence non-empty, with the tokenizer's ``<bos>``/``<eos>`` and any
        prompt already in it. Returns a float32 numpy array
        ``[len(ids), truncate_dim or embedding_dim]``, rows L2-normalized, in
        the order given. ``truncate_dim`` (``None`` for the full width) must be
        in ``1..=embedding_dim``. Raises :class:`TesslError` for anything the
        library refuses, before any GPU work."""
        handle = self._live()
        flat, offsets = _ids_array(ids)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_encode(
                handle, _ptr(flat, ctypes.c_uint32), flat.size, _ptr(offsets, ctypes.c_uint64), len(offsets) - 1,
                code, out, out_len, err, err_len,
            ),
            len(offsets) - 1,
            truncate_dim,
        )

    # -- the tokenizer and the prompts ------------------------------------------------------

    def _query_string(self, call) -> str:
        """A string the library hands back: ``call(buffer, buffer_len, needed, err,
        err_len)`` is asked for the size first, then for the bytes."""
        needed = ctypes.c_uint64()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(call(None, 0, ctypes.byref(needed), err, _ERR_LEN), err)
        buf = (ctypes.c_uint8 * max(needed.value, 1))()
        again = ctypes.c_uint64()
        self._check(call(buf, needed.value, ctypes.byref(again), err, _ERR_LEN), err)
        return bytes(buf[: again.value]).decode("utf-8")

    def prompts(self) -> dict:
        """The task prompts of the snapshot's ``config_sentence_transformers.json``
        (``{"query": "task: search result | query: ", "document": "title: none | text: ", ...}``),
        an empty dict when the snapshot has none."""
        handle = self._live()
        lib = self._rt.lib
        count = ctypes.c_uint32()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(lib.tessl_embedgemma2_prompt_count(handle, ctypes.byref(count), err, _ERR_LEN), err)
        names = [
            self._query_string(lambda b, n, need, e, el, i=i: lib.tessl_embedgemma2_prompt_name(handle, i, b, n, need, e, el))
            for i in range(count.value)
        ]
        return {name: self._prompt_text(name) for name in names}

    def _prompt_text(self, name: str) -> str:
        handle = self._live()
        raw, length = _name_arg(name)
        return self._query_string(
            lambda b, n, need, e, el: self._rt.lib.tessl_embedgemma2_prompt_get(handle, raw, length, b, n, need, e, el)
        )

    @property
    def default_prompt_name(self):
        """The snapshot's ``default_prompt_name`` (``None`` when it names none,
        as this checkpoint's does not): the prompt that texts get when no
        ``prompt_name`` is given."""
        handle = self._live()
        index = ctypes.c_int32()
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._check(self._rt.lib.tessl_embedgemma2_default_prompt(handle, ctypes.byref(index), err, _ERR_LEN), err)
        if index.value < 0:
            return None
        lib = self._rt.lib
        return self._query_string(
            lambda b, n, need, e, el: lib.tessl_embedgemma2_prompt_name(handle, index.value, b, n, need, e, el)
        )

    def tokenize(self, texts, prompt_name=None) -> list:
        """The ids the model is given for each of ``texts``, as a list of lists of
        ints: ``<bos>`` and ``<eos>`` around the task prompt ``prompt_name`` (a key
        of :meth:`prompts`; ``None`` is the snapshot's default, none for this
        checkpoint) and the text, as sentence-transformers feeds them. The
        tokenizer is Rust's, from the snapshot's ``tokenizer.json``."""
        handle = self._live()
        np = _numpy()
        flat, offsets = _utf8_texts(_text_list(texts))
        name, name_len = _name_arg(prompt_name)
        n = len(offsets) - 1
        lib = self._rt.lib
        total = ctypes.c_uint64()
        err = ctypes.create_string_buffer(_ERR_LEN)

        def run(ids, cuts):
            return lib.tessl_embedgemma2_tokenize(
                handle, _ptr(flat, ctypes.c_uint8), flat.size, _ptr(offsets, ctypes.c_uint64), n, name, name_len,
                None if ids is None else _ptr(ids, ctypes.c_uint32), 0 if ids is None else ids.size,
                None if cuts is None else _ptr(cuts, ctypes.c_uint64), ctypes.byref(total), err, _ERR_LEN,
            )

        self._check(run(None, None), err)
        ids = np.empty(total.value, dtype=np.uint32)
        cuts = np.empty(n + 1, dtype=np.uint64)
        self._check(run(ids, cuts), err)
        return [ids[cuts[i] : cuts[i + 1]].tolist() for i in range(n)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """The text of the token ids ``ids`` (a flat sequence of ints), as the
        tokenizer library's ``decode``: ``<bos>``, ``<eos>`` and the other special
        tokens are left out unless ``skip_special_tokens`` is false."""
        handle = self._live()
        np = _numpy()
        skip = 1 if _flag("skip_special_tokens", skip_special_tokens) else 0
        arr = np.asarray(ids)
        if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
            raise TesslError(f"ids must be a flat sequence of integers, got dtype {arr.dtype} with shape {arr.shape}")
        if arr.size and (arr.min() < 0 or arr.max() >= 1 << 32):
            raise TesslError("ids has a token id outside [0, 2**32)")
        flat = np.ascontiguousarray(arr, dtype=np.uint32)
        lib = self._rt.lib
        return self._query_string(
            lambda b, n, need, e, el: lib.tessl_embedgemma2_decode(
                handle, _ptr(flat, ctypes.c_uint32), flat.size, skip, b, n, need, e, el
            )
        )

    def encode_texts(self, texts, tokenizer=None, prompt_name=None, truncate_dim=None):
        """Embed ``texts`` (a list of str): tokenized in Rust from the snapshot's
        own ``tokenizer.json``, so each text becomes ``<bos> ... <eos>``. With
        ``prompt_name`` (a key of :meth:`prompts`, such as ``"query"`` or
        ``"document"``) the snapshot's prompt is written in front of each text, as
        sentence-transformers' ``encode(prompt_name=...)`` does; without one, the
        snapshot's ``default_prompt_name`` applies if it names one (this
        checkpoint's is null). Returns what :meth:`encode` returns.

        ``tokenizer`` is optional: a ``tokenizers.Tokenizer`` or a ``transformers``
        tokenizer of your own is asked for the ids instead (the prompt is still the
        snapshot's), for callers that tokenize with theirs. Pass it by keyword;
        nothing is imported."""
        handle = self._live()
        if isinstance(tokenizer, str):
            raise TesslError("the second argument of encode_texts is the optional tokenizer; pass prompt_name=... by keyword")
        texts = _text_list(texts)
        if tokenizer is not None:
            name = prompt_name if prompt_name is not None else self.default_prompt_name
            if name is not None:
                prompt = self._prompt_text(name)
                texts = [prompt + t for t in texts]
            return self.encode(_token_ids(tokenizer, texts), truncate_dim=truncate_dim)
        flat, offsets = _utf8_texts(texts)
        name, name_len = _name_arg(prompt_name)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_embed_texts(
                handle, _ptr(flat, ctypes.c_uint8), flat.size, _ptr(offsets, ctypes.c_uint64), len(offsets) - 1,
                name, name_len, code, out, out_len, err, err_len,
            ),
            len(offsets) - 1,
            truncate_dim,
        )

    # -- images, audio and video ---------------------------------------------------------

    def _multimodal_handle(self, what: str):
        handle = self._live()
        if not self._multimodal:
            raise TesslError(
                f"{what} needs the vision and audio towers: load with "
                f"EmbedGemma2.from_pretrained(snapshot_dir, multimodal=True)"
            )
        return handle

    def encode_images(self, pixel_values, image_position_ids, truncate_dim=None):
        """Embed images from a transformers processor's outputs: one row per
        image, float32 ``[n_images, truncate_dim or embedding_dim]``, rows
        L2-normalized, the embedding of the processor's image sequence
        (``<bos> <boi> <image> x S <eoi> <eos>``) with the vision tower's soft
        tokens in place of the placeholders. (:meth:`embed_images` does the
        preprocessing itself, from raw arrays.)

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
        flat_px, flat_pos, offsets = _image_arrays(pixel_values, image_position_ids)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_encode_images(
                handle, _ptr(flat_px, ctypes.c_float), flat_px.size, _ptr(flat_pos, ctypes.c_int32), flat_pos.size,
                _ptr(offsets, ctypes.c_uint64), len(offsets) - 1, code, out, out_len, err, err_len,
            ),
            len(offsets) - 1,
            truncate_dim,
        )

    def encode_audio(self, input_features, input_features_mask, truncate_dim=None):
        """Embed audio clips from a transformers processor's outputs: one row
        per clip, float32 ``[n_clips, truncate_dim or embedding_dim]``, rows
        L2-normalized, the embedding of the processor's audio sequence
        (``<bos> <boa> <audio> x S <eoa> <eos>``) with the audio tower's soft
        tokens in place of the placeholders. (:meth:`embed_audio` does the
        feature extraction itself, from raw samples.)

        ``input_features`` is ``[N, F, 128]`` (one clip: ``[F, 128]``) and
        ``input_features_mask`` ``[N, F]`` (``[F]``; bool, or 0/1), as
        ``processor(audio=...)`` returns them, padded frames included (a clip's
        ``F`` is its padded length; the mask says which frames are real). A
        list of per-clip arrays of different ``F`` works too. A clip whose
        mask leaves no soft token is refused. Otherwise as
        :meth:`encode_images`."""
        handle = self._multimodal_handle("encode_audio")
        flat_f, flat_m, offsets = _audio_arrays(input_features, input_features_mask)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_encode_audio(
                handle, _ptr(flat_f, ctypes.c_float), flat_f.size, _ptr(flat_m, ctypes.c_uint8), flat_m.size,
                _ptr(offsets, ctypes.c_uint64), len(offsets) - 1, code, out, out_len, err, err_len,
            ),
            len(offsets) - 1,
            truncate_dim,
        )

    def embed_images(self, images, truncate_dim=None):
        """Embed raw ``images``, one row each (float32 ``[n_images,
        truncate_dim or embedding_dim]``, rows L2-normalized): resized, rescaled
        and patchified in Rust with the snapshot's ``processor_config.json``
        (bit for bit the torchvision processor's resize), then the vision tower.
        No processor and no ``transformers``.

        ``images`` is a list of uint8 arrays ``[H, W, C]`` (C = 1 gray, 2 gray
        and alpha, 3 RGB, 4 RGBA; alpha is dropped), ``[H, W]`` for gray, one
        such array, or a ``[N, H, W, C]`` array. A PIL image works
        (``numpy.asarray`` reads it). Only uint8 is accepted: a float image
        would wrap on the cast. Needs ``multimodal=True``."""
        handle = self._multimodal_handle("embed_images")
        arrays = _raw_image_list(images)
        if not arrays:
            raise TesslError("images is empty: there is nothing to embed")
        flat, dims = _image_buffers(arrays)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_embed_images(
                handle, _ptr(flat, ctypes.c_uint8), flat.size, _ptr(dims, ctypes.c_uint32), len(arrays),
                code, out, out_len, err, err_len,
            ),
            len(arrays),
            truncate_dim,
        )

    def embed_audio(self, clips, sampling_rate: int = _SAMPLING_RATE, truncate_dim=None):
        """Embed raw mono ``clips``, one row each: the log-mel features are
        computed in Rust (the snapshot's feature extractor), then the audio tower.
        ``clips`` is a list of 1-D float arrays of PCM in ``[-1, 1]`` (a clip is
        cut to the extractor's 30 s), one such array, or a 2-D ``[N, samples]``
        array. ``sampling_rate`` must be the extractor's 16000: audio at another
        rate is refused, never resampled. Needs ``multimodal=True``."""
        handle = self._multimodal_handle("embed_audio")
        arrays = _raw_clip_list(clips)
        if not arrays:
            raise TesslError("clips is empty: there is nothing to embed")
        rate = _check_uint("sampling_rate", sampling_rate, 32)
        flat, offsets = _clip_buffers(arrays)
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_embed_audio(
                handle, _ptr(flat, ctypes.c_float), flat.size, _ptr(offsets, ctypes.c_uint64), len(arrays), rate,
                code, out, out_len, err, err_len,
            ),
            len(arrays),
            truncate_dim,
        )

    def embed_videos(self, videos, fps, timestamps: bool = False, truncate_dim=None):
        """Embed raw ``videos``, one row each. Each video is a uint8 array
        ``[frames, H, W, 3]`` of decoded RGB frames (or a list of ``[H, W, 3]``
        frames) and ``fps`` its source frame rate (a number for all, or one per
        video). Frames are sampled and resized in Rust as the processor does
        (the snapshot's ``fps`` and ``max_frames``), every sampled frame goes
        through the vision tower. ``timestamps=True`` writes each frame's
        ``mm:ss`` time in front of its block (the processor's
        ``add_timestamps``). Needs ``multimodal=True``."""
        handle = self._multimodal_handle("embed_videos")
        if isinstance(videos, (str, bytes, dict)) or not hasattr(videos, "__iter__"):
            raise TesslError("videos must be a list of videos")
        arrays = [_raw_video(v, f"videos[{i}]") for i, v in enumerate(videos)]
        if not arrays:
            raise TesslError("videos is empty: there is nothing to embed")
        stamps = 1 if _flag("timestamps", timestamps) else 0
        flat, dims, cuts, rates = _video_buffers(arrays, _fps_list(fps, len(arrays)))
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_embed_videos(
                handle, _ptr(flat, ctypes.c_uint8), flat.size, _ptr(dims, ctypes.c_uint32),
                _ptr(cuts, ctypes.c_uint64), _ptr(rates, ctypes.c_double), len(arrays), stamps,
                code, out, out_len, err, err_len,
            ),
            len(arrays),
            truncate_dim,
        )

    def embed_video(self, frames, fps, timestamps: bool = False, truncate_dim=None):
        """:meth:`embed_videos` for one video: ``frames`` is a uint8 array
        ``[frames, H, W, 3]`` (or a list of frames) and ``fps`` its source frame
        rate. Returns ``[1, truncate_dim or embedding_dim]``."""
        return self.embed_videos([frames], fps, timestamps=timestamps, truncate_dim=truncate_dim)

    def media(self, items, sampling_rate: int = _SAMPLING_RATE, truncate_dim=None):
        """Embed items of token ids of your own and the raw media their placeholders
        stand for, one row per item. Each item is a mapping::

            {"ids": [...],            # token ids, placeholders included
             "images": [array, ...],  # as embed_images
             "audio": [array, ...],   # as embed_audio
             "videos": [(frames, fps), ...]}   # as embed_videos, never timestamped

        ``images``, ``audio`` and ``videos`` may be left out. The ``j``-th image,
        audio or video placeholder id of an item's ids takes the ``j``-th soft token
        row of its images, clips or video frames (a video's sampled frames), and
        the counts must match exactly or the call is refused. The ids are yours: the
        placeholder ids and the number of soft tokens each medium makes are the
        checkpoint's (``config.json``, ``processor_config.json``). Needs
        ``multimodal=True``."""
        np = _numpy()
        handle = self._multimodal_handle("media")
        if isinstance(items, (str, bytes, dict)) or not hasattr(items, "__iter__"):
            raise TesslError("items must be a list of mappings")
        items = list(items)
        if not items:
            raise TesslError("items is empty: there is nothing to embed")
        rate = _check_uint("sampling_rate", sampling_rate, 32)
        ids, images, clips, videos, rates = [], [], [], [], []
        counts = {"images": [0], "audio": [0], "videos": [0]}
        for i, item in enumerate(items):
            if not hasattr(item, "get") or not hasattr(item, "keys"):
                raise TesslError(f"items[{i}] must be a mapping, not {type(item).__name__}")
            unknown = set(item.keys()) - {"ids", "images", "audio", "videos"}
            if unknown:
                raise TesslError(f"items[{i}] has unknown keys {sorted(unknown)}; expected ids, images, audio, videos")
            if "ids" not in item:
                raise TesslError(f"items[{i}] has no ids")
            ids.append(item["ids"])
            got = _raw_image_list(item.get("images", ()), f"items[{i}]['images']")
            images.extend(got)
            counts["images"].append(counts["images"][-1] + len(got))
            got = _raw_clip_list(item.get("audio", ()), f"items[{i}]['audio']")
            clips.extend(got)
            counts["audio"].append(counts["audio"][-1] + len(got))
            vids = item.get("videos", ())
            if isinstance(vids, (str, bytes, dict)) or not hasattr(vids, "__iter__"):
                raise TesslError(f"items[{i}]['videos'] must be a list of (frames, fps) pairs")
            vids = list(vids)
            for j, pair in enumerate(vids):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise TesslError(f"items[{i}]['videos'][{j}] must be a (frames, fps) pair")
                videos.append(_raw_video(pair[0], f"items[{i}]['videos'][{j}]"))
                rates.extend(_fps_list(pair[1], 1))
            counts["videos"].append(counts["videos"][-1] + len(vids))
        flat_ids, id_cuts = _ids_array(ids)
        flat_px, dims = _image_buffers(images)
        flat_pcm, sample_cuts = _clip_buffers(clips)
        flat_fr, vdims, frame_cuts, fps_arr = _video_buffers(videos, rates)
        image_cuts, clip_cuts, video_cuts = (np.array(counts[k], dtype=np.uint64) for k in ("images", "audio", "videos"))
        return self._embed(
            lambda code, out, out_len, err, err_len: self._rt.lib.tessl_embedgemma2_embed_media(
                handle,
                _ptr(flat_ids, ctypes.c_uint32), flat_ids.size, _ptr(id_cuts, ctypes.c_uint64), len(items),
                _ptr(flat_px, ctypes.c_uint8), flat_px.size, _ptr(dims, ctypes.c_uint32), len(images),
                _ptr(image_cuts, ctypes.c_uint64),
                _ptr(flat_pcm, ctypes.c_float), flat_pcm.size, _ptr(sample_cuts, ctypes.c_uint64), len(clips),
                _ptr(clip_cuts, ctypes.c_uint64), rate,
                _ptr(flat_fr, ctypes.c_uint8), flat_fr.size, _ptr(vdims, ctypes.c_uint32),
                _ptr(frame_cuts, ctypes.c_uint64), _ptr(fps_arr, ctypes.c_double), len(videos),
                _ptr(video_cuts, ctypes.c_uint64),
                code, out, out_len, err, err_len,
            ),
            len(items),
            truncate_dim,
        )

