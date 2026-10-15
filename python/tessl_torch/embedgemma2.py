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

from . import _ERR_LEN, _STATUS_PANIC, TesslError, _check_uint, _error_or_poisoned, _Owned, _runtime, _runtime_status, _status_error

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

    def __init__(self, snapshot_dir):
        path = os.fspath(snapshot_dir)
        if isinstance(path, bytes):
            raise TesslError("snapshot_dir must be a str or a path, not bytes")
        try:
            raw = path.encode("utf-8")
        except UnicodeEncodeError:
            raise TesslError(f"snapshot_dir {path!r} is not valid UTF-8") from None
        rt = _runtime()
        _declare(rt.lib)
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = ctypes.c_void_p()
        status = rt.lib.tessl_embedgemma2_load(rt.handle, raw, len(raw), ctypes.byref(handle), err, _ERR_LEN)
        _runtime_status(rt, status, err)
        self._rt = rt
        self._own(rt.lib, handle)
        rt._models.add(self)
        self._snapshot = Path(path)
        # None until read; False when the snapshot has no prompts file.
        self._prompt_config_cache = None
        dim = ctypes.c_uint32()
        self._check(rt.lib.tessl_embedgemma2_embedding_dim(handle, ctypes.byref(dim), err, _ERR_LEN), err)
        self._dim = dim.value

    @classmethod
    def from_pretrained(cls, snapshot_dir) -> "EmbedGemma2":
        """Load the snapshot directory ``snapshot_dir`` (a local path; nothing
        is downloaded)."""
        return cls(snapshot_dir)

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
