"""tessl_torch.embedgemma2's argument handling and ownership, with no GPU.

The library is replaced by a recording stand-in (``tessl_torch._load`` is
patched), so these run anywhere torch imports, with or without a build:

    python3 -m unittest python/tests/test_embedgemma2_host.py -v

The built library and the real checkpoint are ``test_embedgemma2.py``.
"""

import ctypes
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import tessl_torch  # noqa: E402
from tessl_torch import TesslError, TesslPanic, TesslPoisoned  # noqa: E402
from tessl_torch import embedgemma2 as eg2  # noqa: E402
from tessl_torch.embedgemma2 import EmbedGemma2  # noqa: E402
from test_binding_host import RUNTIME, FakeLib  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EG2 = 0xE62E0000


class FakeEg2Lib(FakeLib):
    """The EmbeddingGemma 2 entries, recording what they were handed. ``encode``
    fills the output with ``0, 1, 2, ...`` so a test can see it came back."""

    dim = 8

    def __init__(self):
        super().__init__()
        self.encode_status = 0
        self.max_tokens = 8192
        self.encoded = []
        self.loaded = []
        # The snapshot's prompts, in file order, and the index of the default one (-1: none).
        self.prompts = {}
        self.default_prompt = -1
        self.front_status = 0
        self.embedded_texts = []
        self.tokenized = []
        self.decoded = []

    @staticmethod
    def _give(data, buf, buf_len, needed, err):
        """The size-first protocol of the string entries: the length is always
        written, the bytes only to a buffer that holds them."""
        needed._obj.value = len(data)
        if buf is None:
            return 0
        if buf_len < len(data):
            err.value = b"buffer too short"
            return 1
        for i, b in enumerate(data):
            buf[i] = b
        return 0

    @staticmethod
    def _split(flat, offsets, n):
        """The texts of the CSR pair, decoded."""
        raw = bytes(flat[i] for i in range(offsets[n]))
        return [raw[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(n)]

    def tessl_embedgemma2_prompt_count(self, h, out, err, n):
        if self.front_status:
            err.value = b"front end failed"
            return self.front_status
        out._obj.value = len(self.prompts)
        return 0

    def tessl_embedgemma2_prompt_name(self, h, index, buf, buf_len, needed, err, n):
        names = list(self.prompts)
        if index >= len(names):
            err.value = b"index past the prompts"
            return 1
        return self._give(names[index].encode(), buf, buf_len, needed, err)

    def tessl_embedgemma2_prompt_get(self, h, name, name_len, buf, buf_len, needed, err, n):
        key = name[:name_len].decode()
        if key not in self.prompts:
            err.value = f"no prompt named {key!r} (the snapshot has {list(self.prompts)})".encode()
            return 1
        return self._give(self.prompts[key].encode(), buf, buf_len, needed, err)

    def tessl_embedgemma2_default_prompt(self, h, out, err, n):
        out._obj.value = self.default_prompt
        return 0

    def tessl_embedgemma2_tokenize(self, h, flat, flat_len, offs, n, name, name_len, ids, cap, cuts, total, err, nn):
        texts = self._split(flat, offs, n)
        self.tokenized.append({"texts": texts, "prompt": None if name is None else name[:name_len].decode()})
        if self.front_status:
            err.value = b"tokenize failed"
            return self.front_status
        rows = [[2, *[ord(c) % 50 for c in t], 1] for t in texts]
        total._obj.value = sum(len(r) for r in rows)
        if ids is None:
            return 0
        at = 0
        for i, row in enumerate(rows):
            cuts[i] = at
            for v in row:
                ids[at] = v
                at += 1
        cuts[n] = at
        return 0

    def tessl_embedgemma2_decode(self, h, ids, n_ids, skip, buf, buf_len, needed, err, n):
        self.decoded.append({"ids": [ids[i] for i in range(n_ids)], "skip": skip})
        return self._give("".join(chr(97 + ids[i] % 26) for i in range(n_ids)).encode(), buf, buf_len, needed, err)

    def tessl_embedgemma2_embed_texts(self, h, flat, flat_len, offs, n, name, name_len, trunc, out, out_len, err, nn):
        self.embedded_texts.append(
            {
                "texts": self._split(flat, offs, n),
                "offsets": [offs[i] for i in range(n + 1)],
                "prompt": None if name is None else name[:name_len].decode(),
                "prompt_len": name_len,
                "truncate_dim": trunc,
                "out_len": out_len,
            }
        )
        if self.encode_status:
            err.value = b"embed_texts failed"
            return self.encode_status
        for i in range(out_len):
            out[i] = float(i)
        return 0

    def tessl_embedgemma2_load(self, rt, path, n, out, err, nn):
        self.loaded.append((path, n))
        if self.load_status:
            err.value = b"load failed"
            return self.load_status
        out._obj.value = EG2
        return 0

    def tessl_embedgemma2_free(self, h):
        self._rec("tessl_embedgemma2_free", h.value)
        return self.free_status

    def tessl_embedgemma2_embedding_dim(self, h, out, err, n):
        out._obj.value = self.dim
        return 0

    def tessl_embedgemma2_max_tokens(self, h, out, err, n):
        out._obj.value = self.max_tokens
        return 0

    def tessl_embedgemma2_set_max_tokens(self, h, v, err, n):
        self._rec("tessl_embedgemma2_set_max_tokens", v)
        if v == 0:
            err.value = b"max_tokens must be in range"
            return 1
        self.max_tokens = v
        return 0

    def tessl_embedgemma2_encode(self, h, ids, n_ids, offs, n_seqs, trunc, out, out_len, err, n):
        self.encoded.append(
            {
                "ids": [ids[i] for i in range(n_ids)],
                "offsets": [offs[i] for i in range(n_seqs + 1)],
                "n_seqs": n_seqs,
                "truncate_dim": trunc,
                "out_len": out_len,
            }
        )
        if self.encode_status:
            err.value = b"encode failed"
            return self.encode_status
        for i in range(out_len):
            out[i] = float(i)
        return 0


class HostCase(unittest.TestCase):
    def setUp(self):
        self.lib = FakeEg2Lib()
        patch = mock.patch.object(tessl_torch, "_load", return_value=self.lib)
        patch.start()
        self.addCleanup(patch.stop)
        local = mock.patch.object(tessl_torch, "_local", threading.local())
        local.start()
        self.addCleanup(local.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = Path(self.tmp.name)

    def model(self):
        return EmbedGemma2.from_pretrained(self.snapshot)

    def calls(self, name):
        return [c for c in self.lib.calls if c[0] == name]


class Loading(HostCase):
    def test_the_path_reaches_the_library_as_utf8_with_its_length(self):
        sub = self.snapshot / "snapshot é"
        sub.mkdir()
        for given in (sub, str(sub)):
            with EmbedGemma2(given) as model:
                self.assertEqual(model.embedding_dim, FakeEg2Lib.dim)
        want = str(sub).encode("utf-8")
        self.assertEqual(self.lib.loaded, [(want, len(want))] * 2)
        self.assertNotEqual(len(want), len(str(sub)))

    def test_a_bytes_path_is_refused_without_calling_the_library(self):
        with self.assertRaisesRegex(TesslError, "bytes"):
            EmbedGemma2(b"/x")
        self.assertEqual(self.lib.loaded, [])

    def test_a_load_refusal_keeps_the_runtime_and_a_panic_drops_it(self):
        self.lib.load_status = 1
        with self.assertRaisesRegex(TesslError, "load failed") as caught:
            self.model()
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertEqual(self.calls("tessl_runtime_free"), [])
        self.lib.load_status = 2
        with self.assertRaises(TesslPanic):
            self.model()
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])
        self.assertEqual(self.calls("tessl_embedgemma2_free"), [])

    def test_a_model_is_freed_once_and_a_closed_one_never_reaches_the_library(self):
        model = self.model()
        model.close()
        model.close()
        del model
        self.assertEqual(self.calls("tessl_embedgemma2_free"), [("tessl_embedgemma2_free", EG2)])
        model = self.model()
        model.close()
        before = len(self.lib.calls)
        for call in (lambda: model.encode([[1]]), lambda: model.max_tokens, lambda: model.embedding_dim):
            with self.assertRaisesRegex(TesslError, "closed"):
                call()
        self.assertEqual(len(self.lib.calls), before)

    def test_it_cannot_be_copied_or_pickled(self):
        import copy
        import pickle

        model = self.model()
        for f in (copy.copy, copy.deepcopy, pickle.dumps):
            with self.assertRaises(TypeError):
                f(model)

    def test_the_declarations_skip_a_stand_in_and_name_a_library_without_the_entries(self):
        eg2._declare(self.lib)
        with self.assertRaisesRegex(TesslError, "tessl_embedgemma2_.*cargo build"):
            eg2._declare(ctypes.CDLL(None))


class Encoding(HostCase):
    def test_lists_become_the_csr_pair_and_the_output_is_a_float32_array(self):
        with self.model() as model:
            out = model.encode([[2, 5, 1], [2, 4294967295, 7, 1], [9]])
        self.assertEqual(out.shape, (3, FakeEg2Lib.dim))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ravel().tolist(), [float(i) for i in range(3 * FakeEg2Lib.dim)])
        (call,) = self.lib.encoded
        self.assertEqual(call["ids"], [2, 5, 1, 2, 4294967295, 7, 1, 9])
        self.assertEqual(call["offsets"], [0, 3, 7, 8])
        self.assertEqual((call["n_seqs"], call["truncate_dim"], call["out_len"]), (3, 0, 3 * FakeEg2Lib.dim))

    def test_tuples_and_numpy_arrays_of_any_integer_dtype_are_accepted(self):
        with self.model() as model:
            model.encode([(1, 2), np.array([3, 4, 5], dtype=np.int64), np.array([6], dtype=np.uint16)])
            model.encode(np.array([[1, 2], [3, 4]], dtype=np.uint32))
        a, b = self.lib.encoded
        self.assertEqual((a["ids"], a["offsets"]), ([1, 2, 3, 4, 5, 6], [0, 2, 5, 6]))
        self.assertEqual((b["ids"], b["offsets"]), ([1, 2, 3, 4], [0, 2, 4]))

    def test_truncate_dim_sets_the_width_and_reaches_the_library(self):
        with self.model() as model:
            out = model.encode([[1, 2], [3]], truncate_dim=3)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(self.lib.encoded[0]["truncate_dim"], 3)
        self.assertEqual(self.lib.encoded[0]["out_len"], 6)

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        big = 1 << 32
        bad_ids = {
            "an empty batch": [],
            "an empty sequence": [[1], []],
            "a string batch": "abc",
            "a string sequence": ["abc"],
            "a bytes sequence": [b"ab"],
            "None": None,
            "an int": 5,
            "a flat list": [1, 2, 3],
            "a negative id": [[1, -1]],
            "an id past u32": [[1, big]],
            "a huge id": [[1, 1 << 70]],
            "a float id": [[1, 2.0]],
            "a bool id": [[True, False]],
            "a numpy bool": [np.array([True])],
            "a nested sequence": [[[1, 2]]],
            "a generator": [(i for i in range(3))],
            "a ragged mix": [[1, "a"]],
            "a 2-D array row": [np.zeros((2, 2), dtype=np.int64)],
            "a negative numpy id": [np.array([-1], dtype=np.int64)],
        }
        with self.model() as model:
            for name, ids in bad_ids.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.encode(ids)
            for name, dim in {
                "zero": 0, "negative": -1, "bool": True, "past u32": big, "a string": "8", "a float": 2.5,
            }.items():
                with self.subTest(f"truncate_dim {name}"), self.assertRaises(TesslError):
                    model.encode([[1]], truncate_dim=dim)
        self.assertEqual(self.lib.encoded, [])

    def test_a_refusal_raises_and_leaves_the_model_open_a_panic_closes_it(self):
        model = self.model()
        self.lib.encode_status = 1
        with self.assertRaisesRegex(TesslError, "encode failed") as caught:
            model.encode([[1]])
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertFalse(model.closed)
        self.lib.encode_status = 2
        with self.assertRaises(TesslPanic):
            model.encode([[1]])
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_embedgemma2_free"), [("tessl_embedgemma2_free", EG2)])
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_a_refusal_that_poisoned_the_runtime_closes_everything(self):
        model = self.model()
        self.lib.encode_status = 1
        self.lib.poisoned = 1
        with self.assertRaisesRegex(TesslPoisoned, "encode failed"):
            model.encode([[1]])
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_max_tokens_reads_and_writes_through_the_library(self):
        with self.model() as model:
            self.assertEqual(model.max_tokens, 8192)
            model.max_tokens = 4096
            self.assertEqual(model.max_tokens, 4096)
            for bad in (-1, 1 << 32, True, 2.5):
                with self.subTest(bad), self.assertRaises(TesslError):
                    model.max_tokens = bad
            with self.assertRaisesRegex(TesslError, "range"):
                model.max_tokens = 0
        self.assertEqual(self.calls("tessl_embedgemma2_set_max_tokens"), [("tessl_embedgemma2_set_max_tokens", 4096), ("tessl_embedgemma2_set_max_tokens", 0)])


class FakeTokenizersTokenizer:
    """The surface of ``tokenizers.Tokenizer`` that encode_texts uses."""

    padding = None

    def __init__(self):
        self.seen = []

    def encode_batch(self, texts, add_special_tokens=True):
        assert add_special_tokens
        self.seen.append(list(texts))
        return [mock.Mock(ids=[2, *[ord(c) % 50 for c in t], 1]) for t in texts]


class FakeTransformersTokenizer:
    """The surface of a ``transformers`` tokenizer that encode_texts uses."""

    def __init__(self):
        self.seen = []

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False):
        assert add_special_tokens and padding is False and truncation is False
        self.seen.append(list(texts))
        return {"input_ids": [[2, *[ord(c) % 50 for c in t], 1] for t in texts], "attention_mask": None}


class Texts(HostCase):
    """``encode_texts`` marshals UTF-8 and a prompt name into the library, which
    owns the tokenizer, the prompts and the snapshot's files: the stand-in
    library is the only thing these tests give a prompt, and the snapshot
    directory they load is empty."""

    PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}

    def setUp(self):
        super().setUp()
        self.lib.prompts = dict(self.PROMPTS)

    def test_texts_reach_the_library_as_utf8_with_offsets_and_need_no_tokenizer_or_file(self):
        with self.model() as model:
            out = model.encode_texts(["hi", "", "héllo 中文 \U0001f389", "yo!"])
        self.assertEqual((out.shape, out.dtype), ((4, FakeEg2Lib.dim), np.float32))
        self.assertEqual(out.ravel().tolist(), [float(i) for i in range(4 * FakeEg2Lib.dim)])
        (call,) = self.lib.embedded_texts
        self.assertEqual(call["texts"], ["hi", "", "héllo 中文 \U0001f389", "yo!"])
        self.assertEqual(call["offsets"], [0, 2, 2, 2 + len("héllo 中文 \U0001f389".encode()), call["offsets"][-1]])
        self.assertEqual((call["prompt"], call["prompt_len"], call["truncate_dim"]), (None, 0, 0))
        self.assertEqual(self.lib.encoded, [], "the ids path was not used")
        self.assertEqual(list(self.snapshot.iterdir()), [], "nothing was read from or written to the snapshot")

    def test_the_prompt_name_and_truncate_dim_reach_the_library(self):
        with self.model() as model:
            out = model.encode_texts(["a"], prompt_name="query", truncate_dim=3)
            model.encode_texts(["a"], prompt_name="déjà")
        self.assertEqual(out.shape, (1, 3))
        a, b = self.lib.embedded_texts
        self.assertEqual((a["prompt"], a["prompt_len"], a["truncate_dim"], a["out_len"]), ("query", 5, 3, 3))
        self.assertEqual((b["prompt"], b["prompt_len"]), ("déjà", len("déjà".encode())))

    def test_both_caller_tokenizer_kinds_are_asked_for_special_tokens_and_agree(self):
        with self.model() as model:
            hf, tr = FakeTokenizersTokenizer(), FakeTransformersTokenizer()
            a = model.encode_texts(["hi", "yo!"], tokenizer=hf)
            b = model.encode_texts(["hi", "yo!"], tokenizer=tr)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(hf.seen, [["hi", "yo!"]])
        self.assertEqual(self.lib.encoded[0]["ids"], self.lib.encoded[1]["ids"])
        self.assertEqual(self.lib.encoded[0]["offsets"], [0, 4, 9])
        self.assertEqual(self.lib.embedded_texts, [], "a caller's tokenizer means the id entry")

    def test_a_caller_tokenizer_gets_the_snapshots_prompt_text_from_the_library(self):
        with self.model() as model:
            tok = FakeTokenizersTokenizer()
            model.encode_texts(["a", "b"], tokenizer=tok, prompt_name="query")
            model.encode_texts(["a"], tokenizer=tok, prompt_name="document")
            model.encode_texts(["a"], tokenizer=tok)
            self.assertEqual(model.prompts(), self.PROMPTS)
        self.assertEqual(
            tok.seen,
            [
                [self.PROMPTS["query"] + "a", self.PROMPTS["query"] + "b"],
                [self.PROMPTS["document"] + "a"],
                ["a"],
            ],
        )
        self.lib.default_prompt = 1
        with self.model() as model:
            tok = FakeTokenizersTokenizer()
            model.encode_texts(["a"], tokenizer=tok)
        self.assertEqual(tok.seen, [[self.PROMPTS["document"] + "a"]], "the snapshot's default prompt applies")

    def test_an_unknown_prompt_is_the_librarys_refusal_and_nothing_is_embedded(self):
        with self.model() as model:
            with self.assertRaisesRegex(TesslError, "no prompt named 'nope'"):
                model.encode_texts(["a"], tokenizer=FakeTokenizersTokenizer(), prompt_name="nope")
        self.assertEqual((self.lib.encoded, self.lib.embedded_texts), ([], []))

    def test_prompts_lists_the_librarys_and_none_is_an_empty_dict(self):
        with self.model() as model:
            self.assertEqual(model.prompts(), self.PROMPTS)
            self.assertIsNone(model.default_prompt_name)
            self.lib.default_prompt = 1
            self.assertEqual(model.default_prompt_name, "document")
            self.lib.prompts = {}
            self.lib.default_prompt = -1
            self.assertEqual(model.prompts(), {})
        self.lib.front_status = 1
        with self.model() as model, self.assertRaisesRegex(TesslError, "front end failed"):
            model.prompts()

    def test_tokenize_asks_for_the_size_then_the_ids_and_splits_them_per_text(self):
        with self.model() as model:
            ids = model.tokenize(["ab", "", "c"], prompt_name="query")
        self.assertEqual(ids, [[2, 97 % 50, 98 % 50, 1], [2, 1], [2, 99 % 50, 1]])
        self.assertTrue(all(isinstance(i, int) for row in ids for i in row))
        self.assertEqual(len(self.lib.tokenized), 2, "a size query, then the ids")
        self.assertEqual(self.lib.tokenized[0]["texts"], ["ab", "", "c"])
        self.assertEqual(self.lib.tokenized[1]["prompt"], "query")

    def test_decode_asks_for_the_size_then_the_text(self):
        with self.model() as model:
            self.assertEqual(model.decode([0, 1, 2]), "abc")
            self.assertEqual(model.decode(np.array([3, 4], dtype=np.int64), skip_special_tokens=False), "de")
            self.assertEqual(model.decode([]), "")
        self.assertEqual(
            [(d["ids"], d["skip"]) for d in self.lib.decoded],
            [([0, 1, 2], 1)] * 2 + [([3, 4], 0)] * 2 + [([], 1)] * 2,
        )

    def test_bad_decode_arguments_are_refused_before_the_library_is_called(self):
        big = 1 << 32
        with self.model() as model:
            for name, ids in {
                "nested": [[1, 2]],
                "a float": [1.5],
                "negative": [-1],
                "past u32": [big],
                "a string": "abc",
                "a bool": [True],
            }.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.decode(ids)
            with self.assertRaises(TesslError):
                model.decode([1], skip_special_tokens=2)
        self.assertEqual(self.lib.decoded, [])

    def test_bad_texts_are_refused_before_the_library_is_called(self):
        with self.model() as model:
            for name, texts in {
                "a bare string": "abc",
                "bytes": b"abc",
                "an empty list": [],
                "a non-str": ["a", 5],
                "a lone surrogate": ["a\ud800"],
            }.items():
                with self.subTest(name):
                    for call in (model.encode_texts, model.tokenize):
                        with self.assertRaises(TesslError):
                            call(texts)
            for bad in (5, b"query", ["query"]):
                with self.subTest(f"prompt_name {bad!r}"), self.assertRaises(TesslError):
                    model.encode_texts(["a"], prompt_name=bad)
            with self.assertRaisesRegex(TesslError, "optional tokenizer"):
                model.encode_texts(["a"], "query")
            with self.assertRaisesRegex(TesslError, "pads"):
                model.encode_texts(["a"], tokenizer=type("P", (FakeTokenizersTokenizer,), {"padding": {"d": 1}})())
            with self.assertRaisesRegex(TesslError, "tokenizer must be"):
                model.encode_texts(["a"], tokenizer=object())
            for dim in (0, -1, True, 1 << 32, "8"):
                with self.subTest(f"truncate_dim {dim!r}"), self.assertRaises(TesslError):
                    model.encode_texts(["a"], truncate_dim=dim)
        self.assertEqual((self.lib.encoded, self.lib.embedded_texts, self.lib.tokenized), ([], [], []))

    def test_a_refusal_raises_and_leaves_the_model_open_a_panic_closes_it(self):
        model = self.model()
        self.lib.encode_status = 1
        with self.assertRaisesRegex(TesslError, "embed_texts failed") as caught:
            model.encode_texts(["a"])
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertFalse(model.closed)
        self.lib.encode_status = 2
        with self.assertRaises(TesslPanic):
            model.encode_texts(["a"])
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_importing_the_module_imports_neither_tokenizers_nor_transformers(self):
        # Checked in a fresh interpreter: this process may have imported them already.
        import subprocess

        code = (
            "import sys; sys.path.insert(0, %r); import tessl_torch.embedgemma2 as m; "
            "bad = [n for n in ('tokenizers', 'transformers') if n in sys.modules]; "
            "sys.exit(1 if bad else 0)" % str(ROOT / "python")
        )
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_front_end_declarations_skip_a_stand_in_and_name_a_library_without_the_entries(self):
        eg2._declare_front_ends(self.lib)
        with self.assertRaisesRegex(TesslError, "tessl_embedgemma2_embed_texts.*cargo build"):
            eg2._declare_front_ends(ctypes.CDLL(None))
