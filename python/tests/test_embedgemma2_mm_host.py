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
        self.raw_images = []
        self.raw_audio = []
        self.raw_videos = []
        self.raw_media = []
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

    def _raw(self, kind, record, out, out_len, err):
        """One raw-media call: recorded, refused with ``raw_status`` or answered with ``0, 1, 2, ...``."""
        getattr(self, kind).append(record)
        if self.mm_status:
            err.value = f"{kind} failed".encode()
            return self.mm_status
        self._fill(out, out_len)
        return 0

    def tessl_embedgemma2_embed_images(self, h, px, px_len, dims, n, trunc, out, out_len, err, nn):
        record = {
            "pixels": [px[i] for i in range(px_len)],
            "dims": [dims[i] for i in range(3 * n)],
            "n": n,
            "truncate_dim": trunc,
            "out_len": out_len,
        }
        return self._raw("raw_images", record, out, out_len, err)

    def tessl_embedgemma2_embed_audio(self, h, s, s_len, offs, n, rate, trunc, out, out_len, err, nn):
        record = {
            "samples": [s[i] for i in range(s_len)],
            "offsets": [offs[i] for i in range(n + 1)],
            "n": n,
            "rate": rate,
            "truncate_dim": trunc,
            "out_len": out_len,
        }
        return self._raw("raw_audio", record, out, out_len, err)

    def tessl_embedgemma2_embed_videos(self, h, fr, fr_len, dims, offs, fps, n, ts, trunc, out, out_len, err, nn):
        record = {
            "frames": bytes(fr[i] for i in range(fr_len)),
            "dims": [dims[i] for i in range(2 * n)],
            "offsets": [offs[i] for i in range(n + 1)],
            "fps": [fps[i] for i in range(n)],
            "n": n,
            "timestamps": ts,
            "truncate_dim": trunc,
            "out_len": out_len,
        }
        return self._raw("raw_videos", record, out, out_len, err)

    def tessl_embedgemma2_embed_media(
        self, h, ids, n_ids, id_offs, n_items, px, px_len, idims, n_img, img_cuts,
        s, s_len, s_offs, n_clips, clip_cuts, rate, fr, fr_len, vdims, f_offs, fps, n_vid, vid_cuts,
        trunc, out, out_len, err, nn,
    ):  # fmt: skip
        record = {
            "ids": [ids[i] for i in range(n_ids)],
            "id_offsets": [id_offs[i] for i in range(n_items + 1)],
            "n_items": n_items,
            "pixels": bytes(px[i] for i in range(px_len)),
            "image_dims": [idims[i] for i in range(3 * n_img)],
            "n_images": n_img,
            "image_cuts": [img_cuts[i] for i in range(n_items + 1)],
            "samples": [s[i] for i in range(s_len)],
            "sample_offsets": [s_offs[i] for i in range(n_clips + 1)] if n_clips else [],
            "n_clips": n_clips,
            "clip_cuts": [clip_cuts[i] for i in range(n_items + 1)],
            "rate": rate,
            "frames": bytes(fr[i] for i in range(fr_len)),
            "video_dims": [vdims[i] for i in range(2 * n_vid)],
            "frame_offsets": [f_offs[i] for i in range(n_vid + 1)],
            "fps": [fps[i] for i in range(n_vid)],
            "n_videos": n_vid,
            "video_cuts": [vid_cuts[i] for i in range(n_items + 1)],
            "truncate_dim": trunc,
            "out_len": out_len,
        }
        return self._raw("raw_media", record, out, out_len, err)

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


def rgb(h, w, c=3, seed=0):
    """A uint8 ``[h, w, c]`` image."""
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, c), dtype=np.uint8)


def video(frames, h, w, seed=0):
    """A uint8 ``[frames, h, w, 3]`` video."""
    return np.random.default_rng(seed).integers(0, 256, size=(frames, h, w, 3), dtype=np.uint8)


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
            pic, pcm, vid = rgb(4, 4), np.zeros(8, dtype=np.float32), video(2, 4, 4)
            for call in (
                lambda: model.encode_images(px, pos),
                lambda: model.encode_audio(f, m),
                lambda: model.embed_images([pic]),
                lambda: model.embed_audio([pcm]),
                lambda: model.embed_video(vid, 1.0),
                lambda: model.embed_videos([vid], 1.0),
                lambda: model.media([{"ids": [1, 2]}]),
            ):
                with self.assertRaisesRegex(TesslError, "multimodal=True"):
                    call()
        self.assertEqual((self.lib.images, self.lib.audio), ([], []))
        self.assertEqual(
            (self.lib.raw_images, self.lib.raw_audio, self.lib.raw_videos, self.lib.raw_media), ([], [], [], [])
        )

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


class RawImages(MmCase):
    def test_images_become_flat_uint8_and_an_hwc_table(self):
        a, b, g = rgb(3, 4, 3, 1), rgb(2, 5, 4, 2), rgb(6, 2, 1, 3)
        with self.mm() as model:
            out = model.embed_images([a, b, g[:, :, 0], g])
        self.assertEqual((out.shape, out.dtype), ((4, DIM), np.float32))
        self.assertEqual(out.ravel().tolist(), [float(i) for i in range(4 * DIM)])
        (call,) = self.lib.raw_images
        self.assertEqual(call["pixels"], np.concatenate([a.ravel(), b.ravel(), g.ravel(), g.ravel()]).tolist())
        self.assertEqual(call["dims"], [3, 4, 3, 2, 5, 4, 6, 2, 1, 6, 2, 1])
        self.assertEqual((call["n"], call["truncate_dim"], call["out_len"]), (4, 0, 4 * DIM))

    def test_one_image_a_batch_array_and_a_pil_like_object_are_accepted(self):
        a = rgb(3, 4)

        class Pil:
            """What PIL images look like to numpy: the array interface."""

            __array_interface__ = a.__array_interface__

        with self.mm() as model:
            model.embed_images(a)
            model.embed_images(np.stack([a, a]))
            model.embed_images([Pil()])
            model.embed_images((a,), truncate_dim=3)
        one, batch, pil, tup = self.lib.raw_images
        self.assertEqual((one["n"], batch["n"], pil["n"]), (1, 2, 1))
        self.assertEqual(pil["pixels"], one["pixels"])
        self.assertEqual(batch["dims"], [3, 4, 3] * 2)
        self.assertEqual((tup["truncate_dim"], tup["out_len"]), (3, 3))

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        a = rgb(3, 4)
        cases = {
            "no images": [],
            "a string": "cat.png",
            "a dict": {"a": 1},
            "None": None,
            "floats": a.astype(np.float32),
            "uint16": a.astype(np.uint16),
            "int64 (a wrap if cast)": a.astype(np.int64),
            "bool": a > 100,
            "a 1-D array": np.zeros(5, dtype=np.uint8),
            "a 5-D array": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
            "a list holding a float image": [a, a.astype(np.float64)],
            "a list holding a 1-D array": [a, np.zeros(4, dtype=np.uint8)],
        }
        with self.mm() as model:
            for name, images in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.embed_images(images)
            for name, dim in {"zero": 0, "negative": -1, "bool": True, "a string": "8", "a float": 2.5}.items():
                with self.subTest(f"truncate_dim {name}"), self.assertRaises(TesslError):
                    model.embed_images(a, truncate_dim=dim)
            with self.assertRaisesRegex(TesslError, "dimension past 2"):
                # Not allocated: a view whose shape is past u32.
                big = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.uint8), shape=(1 << 32, 1, 1), strides=(0, 0, 0))
                model.embed_images([big])
        self.assertEqual(self.lib.raw_images, [])

    def test_a_refusal_raises_and_leaves_the_model_open_a_panic_closes_it(self):
        a = rgb(3, 4)
        model = self.mm()
        self.lib.mm_status = 1
        with self.assertRaisesRegex(TesslError, "raw_images failed") as caught:
            model.embed_images(a)
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertFalse(model.closed)
        self.lib.mm_status = 2
        with self.assertRaises(TesslPanic):
            model.embed_images(a)
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_embedgemma2_free"), [("tessl_embedgemma2_free", EG2)])


class RawAudio(MmCase):
    def test_clips_become_flat_float32_and_sample_offsets(self):
        a = np.linspace(-1, 1, 5, dtype=np.float32)
        b = np.linspace(-1, 1, 3, dtype=np.float64)
        with self.mm() as model:
            out = model.embed_audio([a, b])
            model.embed_audio(a, sampling_rate=16000, truncate_dim=2)
            model.embed_audio(np.stack([a, a]))
        self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
        two, one, stacked = self.lib.raw_audio
        self.assertEqual(two["samples"], a.tolist() + b.astype(np.float32).tolist())
        self.assertEqual((two["offsets"], two["rate"], two["n"]), ([0, 5, 8], 16000, 2))
        self.assertEqual((one["offsets"], one["truncate_dim"], one["out_len"]), ([0, 5], 2, 2))
        self.assertEqual((stacked["offsets"], stacked["n"]), ([0, 5, 10], 2))

    def test_the_sampling_rate_reaches_the_library_to_refuse_it(self):
        with self.mm() as model:
            model.embed_audio([np.zeros(4, dtype=np.float32)], sampling_rate=8000)
        self.assertEqual(self.lib.raw_audio[0]["rate"], 8000)

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        pcm = np.zeros(8, dtype=np.float32)
        cases = {
            "no clips": [],
            "a string": "x.wav",
            "None": None,
            "int16 samples": np.zeros(8, dtype=np.int16),
            "a stereo clip": np.zeros((2, 8, 2), dtype=np.float32),
            "a list holding a 2-D clip": [pcm, np.zeros((2, 4), dtype=np.float32)],
            "a list holding ints": [pcm, np.zeros(4, dtype=np.int32)],
        }
        with self.mm() as model:
            for name, clips in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.embed_audio(clips)
            for name, rate in {"negative": -1, "past u32": 1 << 32, "a float": 16000.0, "bool": True}.items():
                with self.subTest(f"rate {name}"), self.assertRaises(TesslError):
                    model.embed_audio(pcm, sampling_rate=rate)
            with self.assertRaises(TesslError):
                model.embed_audio(pcm, truncate_dim=0)
        self.assertEqual(self.lib.raw_audio, [])


class RawVideo(MmCase):
    def test_a_video_becomes_flat_frames_a_table_offsets_and_rates(self):
        a, b = video(3, 4, 6, 1), video(2, 2, 2, 2)
        with self.mm() as model:
            out = model.embed_videos([a, b], fps=[24.0, 5], timestamps=True)
            model.embed_video(list(a), 30)
            model.embed_videos([a, a], 12)
        self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
        two, one, shared = self.lib.raw_videos
        self.assertEqual(two["frames"], a.tobytes() + b.tobytes())
        self.assertEqual(
            (two["dims"], two["offsets"], two["fps"], two["timestamps"], two["n"]),
            ([4, 6, 2, 2], [0, 3, 5], [24.0, 5.0], 1, 2),
        )
        self.assertEqual((one["offsets"], one["fps"], one["timestamps"], one["out_len"]), ([0, 3], [30.0], 0, DIM))
        self.assertEqual(shared["fps"], [12.0, 12.0])

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        v = video(2, 3, 3)
        cases = {
            "no videos": ([], 1.0),
            "a string": ("clip.mp4", 1.0),
            "a dict": ({"a": 1}, 1.0),
            "float frames": ([v.astype(np.float32)], 1.0),
            "grayscale frames": ([v[..., 0]], 1.0),
            "four channels": ([np.zeros((2, 3, 3, 4), dtype=np.uint8)], 1.0),
            "no frames": ([np.zeros((0, 3, 3, 3), dtype=np.uint8)], 1.0),
            "frames of different sizes": ([[v[0], video(1, 4, 4)[0]]], 1.0),
            "an empty frame list": ([[]], 1.0),
            "fps count mismatch": ([v, v], [1.0]),
            "a string fps": ([v], "30"),
            "a bool fps": ([v], True),
            "None fps": ([v], None),
        }
        with self.mm() as model:
            for name, (videos, fps) in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.embed_videos(videos, fps)
            for name, stamps in {"2": 2, "a string": "yes", "None": None}.items():
                with self.subTest(f"timestamps {name}"), self.assertRaises(TesslError):
                    model.embed_videos([v], 1.0, timestamps=stamps)
            with self.assertRaises(TesslError):
                model.embed_video(v, 1.0, truncate_dim=0)
        self.assertEqual(self.lib.raw_videos, [])

    def test_the_library_judges_the_rate(self):
        with self.mm() as model:
            model.embed_video(video(2, 3, 3), -1.0)  # positivity is the library's to check
        self.assertEqual(self.lib.raw_videos[0]["fps"], [-1.0])


class Media(MmCase):
    def test_items_become_ids_media_and_per_item_cuts(self):
        img, img2 = rgb(3, 4, 3, 1), rgb(2, 2, 1, 2)
        pcm = np.linspace(-1, 1, 6, dtype=np.float32)
        vid = video(2, 2, 2, 3)
        items = [
            {"ids": [2, 60, 60, 1], "images": [img, img2]},
            {"ids": [2, 1]},
            {"ids": [2, 61, 1], "audio": [pcm], "videos": [(vid, 24.0), (list(vid), 8)]},
        ]
        with self.mm() as model:
            out = model.media(items, truncate_dim=4)
        self.assertEqual(out.shape, (3, 4))
        (call,) = self.lib.raw_media
        self.assertEqual(call["ids"], [2, 60, 60, 1, 2, 1, 2, 61, 1])
        self.assertEqual((call["id_offsets"], call["n_items"]), ([0, 4, 6, 9], 3))
        self.assertEqual(call["pixels"], img.tobytes() + img2.tobytes())
        self.assertEqual((call["image_dims"], call["n_images"], call["image_cuts"]), ([3, 4, 3, 2, 2, 1], 2, [0, 2, 2, 2]))
        self.assertEqual(call["samples"], pcm.tolist())
        self.assertEqual((call["sample_offsets"], call["n_clips"], call["clip_cuts"]), ([0, 6], 1, [0, 0, 0, 1]))
        self.assertEqual(call["rate"], 16000)
        self.assertEqual(call["frames"], vid.tobytes() * 2)
        self.assertEqual(
            (call["video_dims"], call["frame_offsets"], call["fps"], call["n_videos"], call["video_cuts"]),
            ([2, 2, 2, 2], [0, 2, 4], [24.0, 8.0], 2, [0, 0, 0, 2]),
        )
        self.assertEqual((call["truncate_dim"], call["out_len"]), (4, 12))

    def test_a_call_with_no_media_passes_empty_modalities(self):
        with self.mm() as model:
            model.media([{"ids": [2, 5, 1]}, {"ids": [2, 6, 1]}], sampling_rate=16000)
        (call,) = self.lib.raw_media
        self.assertEqual(
            (call["n_images"], call["n_clips"], call["n_videos"], call["image_cuts"], call["clip_cuts"], call["video_cuts"]),
            (0, 0, 0, [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        )
        self.assertEqual((call["pixels"], call["samples"], call["frames"]), (b"", [], b""))

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        img, pcm, vid = rgb(2, 2), np.zeros(4, dtype=np.float32), video(1, 2, 2)
        cases = {
            "no items": [],
            "a string": "items",
            "a dict instead of a list": {"ids": [1]},
            "an item that is not a mapping": [[1, 2]],
            "an item without ids": [{"images": [img]}],
            "an unknown key": [{"ids": [1], "image": [img]}],
            "empty ids": [{"ids": []}],
            "float ids": [{"ids": [1.5]}],
            "float images": [{"ids": [1], "images": [img.astype(np.float32)]}],
            "int audio": [{"ids": [1], "audio": [np.zeros(4, dtype=np.int16)]}],
            "a video that is not a pair": [{"ids": [1], "videos": [vid]}],
            "a video pair of three": [{"ids": [1], "videos": [(vid, 1.0, 2)]}],
            "a video with a bad frame array": [{"ids": [1], "videos": [(vid.astype(np.float32), 1.0)]}],
            "videos that is a string": [{"ids": [1], "videos": "v"}],
        }
        with self.mm() as model:
            for name, items in cases.items():
                with self.subTest(name), self.assertRaises(TesslError):
                    model.media(items)
            with self.assertRaises(TesslError):
                model.media([{"ids": [1], "audio": [pcm]}], sampling_rate=-1)
            with self.assertRaises(TesslError):
                model.media([{"ids": [1]}], truncate_dim=0)
        self.assertEqual(self.lib.raw_media, [])

    def test_the_front_end_declarations_skip_a_stand_in_and_name_a_library_without_the_entries(self):
        import ctypes

        eg2._declare_front_ends(self.lib)
        with self.assertRaisesRegex(TesslError, "embed_texts.*cargo build"):
            eg2._declare_front_ends(ctypes.CDLL(None))



if __name__ == "__main__":
    unittest.main()
