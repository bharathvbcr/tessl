"""tessl_torch.embedgemma2's multimodal argument handling, with no GPU.

The library is replaced by a recording stand-in (``tessl_torch._load`` is
patched), as in ``test_embedgemma2_host.py``, so these run anywhere torch
imports, with or without a build:

    python3 -m unittest python/tests/test_embedgemma2_mm_host.py -v

The built library and the real checkpoint are ``test_embedgemma2_mm.py``.
"""

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
from tessl_torch import TesslError, TesslPanic  # noqa: E402
from tessl_torch import embedgemma2 as eg2  # noqa: E402
from tessl_torch.embedgemma2 import EmbedGemma2  # noqa: E402
from test_binding_host import RUNTIME  # noqa: E402
from test_embedgemma2_host import EG2, FakeEg2Lib  # noqa: E402

DIM = FakeEg2Lib.dim
PD = 6  # patch width of the stand-in's images
MELS = 5  # mel bins of the stand-in's clips


class FakeMmLib(FakeEg2Lib):
    """The multimodal entries, recording what they were handed (the arrays as
    lists, so a test sees exactly what crossed). The output is ``0, 1, 2, ...``."""

    def __init__(self):
        super().__init__()
        self.loaded_mm = []
        self.images = []
        self.audio = []
        self.mm_status = 0

    def tessl_embedgemma2_load_multimodal(self, rt, path, n, operands, out, err, nn):
        self.loaded_mm.append((path, n, operands))
        if self.load_status:
            err.value = b"load failed"
            return self.load_status
        out._obj.value = EG2
        return 0

    def _fill(self, out, out_len):
        for i in range(out_len):
            out[i] = float(i)

    def tessl_embedgemma2_encode_images(self, h, px, px_len, pos, pos_len, offs, n, trunc, out, out_len, err, nn):
        self.images.append(
            {
                "pixels": [px[i] for i in range(px_len)],
                "positions": [pos[i] for i in range(pos_len)],
                "offsets": [offs[i] for i in range(n + 1)],
                "n": n,
                "truncate_dim": trunc,
                "out_len": out_len,
            }
        )
        if self.mm_status:
            err.value = b"encode_images failed"
            return self.mm_status
        self._fill(out, out_len)
        return 0

    def tessl_embedgemma2_encode_audio(self, h, feats, f_len, mask, m_len, offs, n, trunc, out, out_len, err, nn):
        self.audio.append(
            {
                "features": [feats[i] for i in range(f_len)],
                "mask": [mask[i] for i in range(m_len)],
                "offsets": [offs[i] for i in range(n + 1)],
                "n": n,
                "truncate_dim": trunc,
                "out_len": out_len,
            }
        )
        if self.mm_status:
            err.value = b"encode_audio failed"
            return self.mm_status
        self._fill(out, out_len)
        return 0


class FakeTensor:
    """What a torch tensor looks like to the binding: ``detach``, ``cpu``,
    ``float`` and ``numpy``, and a dtype whose name it can read. Counts the
    calls so a test can see the binding went through them."""

    def __init__(self, array, dtype_name):
        self.array, self.dtype, self.calls = array, dtype_name, []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def float(self):
        self.calls.append("float")
        widened = FakeTensor(self.array.astype(np.float32), "torch.float32")
        widened.calls = self.calls  # one record of the whole chain
        return widened

    def numpy(self):
        self.calls.append("numpy")
        if self.dtype == "torch.bfloat16":
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self.array


class FakeProcessor:
    """A callable that records its arguments and returns what a transformers
    processor returns with ``return_tensors="np"`` (a mapping of arrays)."""

    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs


def image(rows, seed=0, pad=0):
    """A float32 ``[rows + pad, PD]`` patch array and its int64 ``[., 2]``
    positions (a raster, then ``(-1, -1)`` padding)."""
    rng = np.random.default_rng(seed)
    px = rng.random((rows + pad, PD), dtype=np.float32)
    pos = np.array([[i % 3, i // 3] for i in range(rows)] + [[-1, -1]] * pad, dtype=np.int64).reshape(-1, 2)
    return px, pos


def clip(frames, seed=0, valid=None):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((frames, MELS)).astype(np.float32)
    mask = np.arange(frames) < (frames if valid is None else valid)
    return feats, mask


class MmCase(unittest.TestCase):
    def setUp(self):
        self.lib = FakeMmLib()
        patch = mock.patch.object(tessl_torch, "_load", return_value=self.lib)
        patch.start()
        self.addCleanup(patch.stop)
        local = mock.patch.object(tessl_torch, "_local", threading.local())
        local.start()
        self.addCleanup(local.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = Path(self.tmp.name)

    def mm(self, **kw):
        return EmbedGemma2.from_pretrained(self.snapshot, multimodal=True, **kw)

    def calls(self, name):
        return [c for c in self.lib.calls if c[0] == name]


class Loading(MmCase):
    def test_multimodal_loads_through_its_own_entry_with_the_operands_code(self):
        want = str(self.snapshot).encode("utf-8")
        with self.mm() as model:
            self.assertEqual(model.embedding_dim, DIM)
        with self.mm(operands="bf16"):
            pass
        self.assertEqual(self.lib.loaded_mm, [(want, len(want), 0), (want, len(want), 1)])
        self.assertEqual(self.lib.loaded, [], "the text-only entry was not used")
        with EmbedGemma2.from_pretrained(self.snapshot):
            pass
        self.assertEqual(self.lib.loaded, [(want, len(want))])
        self.assertEqual(len(self.lib.loaded_mm), 2)

    def test_bad_load_arguments_are_refused_before_the_library_is_called(self):
        for kw, pattern in (
            ({"operands": "bf16"}, "multimodal=True"),
            ({"operands": "fp16"}, "operands"),
            ({"multimodal": 1}, "multimodal must be a bool"),
        ):
            with self.subTest(kw), self.assertRaisesRegex(TesslError, pattern):
                EmbedGemma2(self.snapshot, **{"multimodal": False, **kw})
        with self.assertRaisesRegex(TesslError, "operands"):
            self.mm(operands="int8")
        self.assertEqual((self.lib.loaded, self.lib.loaded_mm), ([], []))

    def test_a_multimodal_load_refusal_keeps_the_runtime_and_a_panic_drops_it(self):
        self.lib.load_status = 1
        with self.assertRaisesRegex(TesslError, "load failed") as caught:
            self.mm()
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertEqual(self.calls("tessl_runtime_free"), [])
        self.lib.load_status = 2
        with self.assertRaises(TesslPanic):
            self.mm()
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_a_text_only_model_refuses_media_without_calling_the_library(self):
        with EmbedGemma2.from_pretrained(self.snapshot) as model:
            px, pos = image(4)
            f, m = clip(8)
            for call in (
                lambda: model.encode_images(px, pos),
                lambda: model.encode_audio(f, m),
                lambda: model.embed_images([object()], lambda **k: {}),
                lambda: model.embed_audio([object()], lambda **k: {}),
            ):
                with self.assertRaisesRegex(TesslError, "multimodal=True"):
                    call()
        self.assertEqual((self.lib.images, self.lib.audio), ([], []))

    def test_the_declarations_skip_a_stand_in_and_name_a_library_without_the_entries(self):
        import ctypes

        eg2._declare_multimodal(self.lib)
        with self.assertRaisesRegex(TesslError, "load_multimodal.*cargo build"):
            eg2._declare_multimodal(ctypes.CDLL(None))


class Images(MmCase):
    def test_a_batch_array_becomes_flat_float32_int32_and_row_offsets(self):
        a, pa = image(4, 1, pad=2)
        b, pb = image(3, 2, pad=3)
        assert a.shape == b.shape == (6, PD)
        with self.mm() as model:
            out = model.encode_images(np.stack([a, b]).astype(np.float64), np.stack([pa, pb]))
        self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
        self.assertEqual(out.ravel().tolist(), [float(i) for i in range(2 * DIM)])
        (call,) = self.lib.images
        self.assertEqual(call["pixels"], np.concatenate([a, b]).astype(np.float32).ravel().tolist())
        self.assertEqual(call["positions"], np.concatenate([pa, pb]).astype(np.int32).ravel().tolist())
        self.assertEqual(call["offsets"], [0, 6, 12])
        self.assertEqual((call["n"], call["truncate_dim"], call["out_len"]), (2, 0, 2 * DIM))
        self.assertEqual(call["positions"][-2:], [-1, -1])

    def test_one_image_a_ragged_list_and_per_sample_outputs_are_the_same_calls(self):
        a, pa = image(4, 1, pad=2)
        b, pb = image(5, 2)
        with self.mm() as model:
            model.encode_images(a, pa)  # [P, D]
            model.encode_images([a, b], [pa, pb])  # ragged
            model.encode_images([a[None], b[None]], [pa[None], pb[None]])  # [1, P, D] per sample
            model.encode_images((a, b), (pa, pb))  # tuples
        one, ragged, per_sample, tup = self.lib.images
        self.assertEqual(one["offsets"], [0, 6])
        self.assertEqual(ragged["offsets"], [0, 6, 11])
        self.assertEqual(per_sample, ragged)
        self.assertEqual(tup, ragged)

    def test_torch_tensors_are_read_through_their_methods_and_bf16_is_widened(self):
        a, pa = image(4, 1)
        t = FakeTensor(a[None], "torch.bfloat16")
        p = FakeTensor(pa[None], "torch.int64")
        with self.mm() as model:
            model.encode_images(t, p)
        self.assertEqual(t.calls, ["detach", "cpu", "float", "numpy"])
        self.assertEqual(p.calls, ["detach", "cpu", "numpy"])
        self.assertEqual(self.lib.images[0]["pixels"], a.ravel().tolist())

    def test_truncate_dim_sets_the_width_and_reaches_the_library(self):
        a, pa = image(4)
        with self.mm() as model:
            out = model.encode_images(a, pa, truncate_dim=3)
        self.assertEqual(out.shape, (1, 3))
        self.assertEqual((self.lib.images[0]["truncate_dim"], self.lib.images[0]["out_len"]), (3, 3))

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        a, pa = image(4, 1, pad=1)
        b, pb = image(3, 2)
        wide = np.zeros((5, PD + 1), dtype=np.float32)
        cases = {
            "no images": ([], []),
            "a string": ("abc", pa),
            "None": (None, None),
            "a 1-D array": (np.zeros(PD, dtype=np.float32), pa),
            "a 4-D array": (np.zeros((1, 1, 5, PD), dtype=np.float32), pa),
            "more images than positions": ([a, b], [pa]),
            "fewer images than positions": ([a], [pa, pb]),
            "positions of another length": (a, pa[:-1]),
            "positions with three columns": (a, np.zeros((5, 3), dtype=np.int64)),
            "integer pixels": (a.astype(np.int64), pa),
            "float positions": (a, pa.astype(np.float32)),
            "bool positions": (a, pa.astype(bool)),
            "a position past int32": (a, np.full_like(pa, 1 << 31)),
            "a position below int32": (a, np.full_like(pa, -(1 << 31) - 1)),
            "images of different widths": ([a, wide], [pa, np.zeros((5, 2), dtype=np.int64)]),
            "an image with no patches": (np.zeros((0, PD), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)),
        }
        with self.mm() as model:
            for name, (px, pos) in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.encode_images(px, pos)
            for name, dim in {"zero": 0, "negative": -1, "bool": True, "a string": "8", "a float": 2.5}.items():
                with self.subTest(f"truncate_dim {name}"), self.assertRaises(TesslError):
                    model.encode_images(a, pa, truncate_dim=dim)
        self.assertEqual(self.lib.images, [])

    def test_a_refusal_raises_and_leaves_the_model_open_a_panic_closes_it(self):
        a, pa = image(4)
        model = self.mm()
        self.lib.mm_status = 1
        with self.assertRaisesRegex(TesslError, "encode_images failed") as caught:
            model.encode_images(a, pa)
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertFalse(model.closed)
        self.lib.mm_status = 2
        with self.assertRaises(TesslPanic):
            model.encode_images(a, pa)
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_embedgemma2_free"), [("tessl_embedgemma2_free", EG2)])


class Audio(MmCase):
    def test_a_batch_array_becomes_flat_float32_uint8_and_frame_offsets(self):
        fa, ma = clip(7, 1, valid=5)
        fb, mb = clip(7, 2)
        with self.mm() as model:
            out = model.encode_audio(np.stack([fa, fb]).astype(np.float64), np.stack([ma, mb]))
        self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
        (call,) = self.lib.audio
        self.assertEqual(call["features"], np.concatenate([fa, fb]).ravel().tolist())
        self.assertEqual(call["mask"], [1, 1, 1, 1, 1, 0, 0] + [1] * 7)
        self.assertEqual(call["offsets"], [0, 7, 14])
        self.assertEqual((call["n"], call["truncate_dim"], call["out_len"]), (2, 0, 2 * DIM))

    def test_one_clip_a_ragged_list_and_integer_masks_are_accepted(self):
        fa, ma = clip(7, 1, valid=5)
        fb, mb = clip(4, 2)
        with self.mm() as model:
            model.encode_audio(fa, ma)
            model.encode_audio([fa, fb], [ma, mb])
            model.encode_audio([fa[None], fb[None]], [ma[None], mb[None]])
            model.encode_audio(fa, ma.astype(np.int64))
            model.encode_audio(fa, FakeTensor(ma[None], "torch.bool"))
        one, ragged, per_sample, ints, tensor = self.lib.audio
        self.assertEqual(one["offsets"], [0, 7])
        self.assertEqual(ragged["offsets"], [0, 7, 11])
        self.assertEqual(per_sample, ragged)
        self.assertEqual(ints, one)
        self.assertEqual(tensor, one)

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        fa, ma = clip(7, 1, valid=5)
        fb, mb = clip(4, 2)
        cases = {
            "no clips": ([], []),
            "a string": ("abc", ma),
            "more clips than masks": ([fa, fb], [ma]),
            "a mask of another length": (fa, ma[:-1]),
            "a 2-D mask for a 2-D clip": (fa, ma[None, None]),
            "integer features": (fa.astype(np.int32), ma),
            "a float mask": (fa, ma.astype(np.float32)),
            "a mask holding 2": (fa, np.array([1, 1, 2, 0, 0, 0, 0])),
            "a negative mask": (fa, np.array([1, 1, -1, 0, 0, 0, 0])),
            "clips of different widths": ([fa, np.zeros((4, MELS + 1), dtype=np.float32)], [ma, mb]),
            "a clip with no frames": (np.zeros((0, MELS), dtype=np.float32), np.zeros(0, dtype=bool)),
        }
        with self.mm() as model:
            for name, (f, m) in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.encode_audio(f, m)
            with self.assertRaises(TesslError):
                model.encode_audio(fa, ma, truncate_dim=0)
        self.assertEqual(self.lib.audio, [])

    def test_a_refusal_raises_and_leaves_the_model_open(self):
        fa, ma = clip(7)
        with self.mm() as model:
            self.lib.mm_status = 1
            with self.assertRaisesRegex(TesslError, "encode_audio failed"):
                model.encode_audio(fa, ma)
            self.assertFalse(model.closed)


class Processors(MmCase):
    def test_embed_images_calls_the_processor_with_the_images_and_encodes_its_outputs(self):
        a, pa = image(4, 1, pad=2)
        b, pb = image(3, 2, pad=3)
        proc = FakeProcessor({"pixel_values": np.stack([a, b]), "image_position_ids": np.stack([pa, pb])})
        pics = [object(), object()]
        with self.mm() as model:
            out = model.embed_images(pics, proc, truncate_dim=4)
        self.assertEqual(out.shape, (2, 4))
        (kwargs,) = proc.calls
        self.assertEqual(set(kwargs), {"images", "return_tensors"})
        self.assertEqual((kwargs["images"], kwargs["return_tensors"]), (pics, "np"))
        self.assertEqual(self.lib.images[0]["offsets"], [0, 6, 12])

    def test_embed_audio_calls_the_processor_with_the_clips_and_encodes_its_outputs(self):
        fa, ma = clip(7, 1, valid=5)
        fb, mb = clip(7, 2)
        proc = FakeProcessor({"input_features": np.stack([fa, fb]), "input_features_mask": np.stack([ma, mb])})
        clips = [np.zeros(3, dtype=np.float32), np.zeros(5, dtype=np.float32)]
        with self.mm() as model:
            model.embed_audio(tuple(clips), proc)
        (kwargs,) = proc.calls
        self.assertEqual(set(kwargs), {"audio", "return_tensors"})
        self.assertEqual((len(kwargs["audio"]), kwargs["return_tensors"]), (2, "np"))
        self.assertEqual(self.lib.audio[0]["offsets"], [0, 7, 14])

    def test_a_bad_processor_or_input_list_is_refused_before_the_library_is_called(self):
        a, pa = image(4)
        good = FakeProcessor({"pixel_values": a[None], "image_position_ids": pa[None]})
        with self.mm() as model:
            for name, call, pattern in (
                ("a non-callable processor", lambda: model.embed_images([object()], "processor"), "callable"),
                ("a missing key", lambda: model.embed_images([object()], FakeProcessor({"pixel_values": a[None]})), "image_position_ids"),
                ("audio keys for images", lambda: model.embed_images([object()], FakeProcessor({"input_features": a})), "pixel_values"),
                ("no images", lambda: model.embed_images([], good), "empty"),
                ("a string", lambda: model.embed_images("cat.png", good), "list of items"),
                ("a dict", lambda: model.embed_images({"a": 1}, good), "list of items"),
                ("a single non-iterable", lambda: model.embed_audio(5, good), "list of items"),
                ("no clips", lambda: model.embed_audio([], good), "empty"),
            ):
                with self.subTest(name), self.assertRaisesRegex(TesslError, pattern):
                    call()
        self.assertEqual((self.lib.images, self.lib.audio), ([], []))
        self.assertEqual(good.calls, [])


if __name__ == "__main__":
    unittest.main()
