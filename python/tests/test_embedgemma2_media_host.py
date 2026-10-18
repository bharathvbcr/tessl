"""tessl_torch.embedgemma2's file, bytes, soft-token-count and segment entries: argument
handling with no GPU.

The library is replaced by a recording stand-in (``tessl_torch._load`` is patched), as in
``test_embedgemma2_mm_host.py``, so these run anywhere torch imports, with or without a build:

    python3 -m unittest python/tests/test_embedgemma2_media_host.py -v

The built library, the tiny snapshot and the real checkpoint are ``test_embedgemma2_media.py``.
"""

import ctypes
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tessl_torch import TesslError  # noqa: E402
from tessl_torch import embedgemma2 as eg2  # noqa: E402
from test_embedgemma2_mm_host import DIM, FakeMmLib, MmCase, rgb, video  # noqa: E402

PATHS, BYTES, RAW = 1, 2, 0
TEXT, IMAGE, AUDIO, VIDEO, VIDEO_T = 0, 1, 2, 3, 4


def cut(blob, offsets, n):
    """The ``n`` pieces of a blob and its byte cuts."""
    return [bytes(blob[offsets[i] : offsets[i + 1]]) for i in range(n)]


class FakeMediaLib(FakeMmLib):
    """The seven entries, recording what they were handed. Embeddings are ``0, 1, 2, ...``;
    the counts are ``10 + i`` (images, clips) and ``(4, 9 + i)`` (videos)."""

    def __init__(self):
        super().__init__()
        self.sources = []
        self.counts = []
        self.segments = []
        self.media_status = 0

    def _answer(self, kind, record, out, out_len, err):
        getattr(self, kind).append(record)
        if self.media_status:
            err.value = f"{record.get('entry', kind)} failed".encode()
            return self.media_status
        self._fill(out, out_len)
        return 0

    def _source_record(self, entry, source, blob, blob_len, offs, n, trunc, out_len, **extra):
        blob_bytes = bytes(blob[i] for i in range(blob_len)) if blob is not None else b""
        return {
            "entry": entry,
            "source": source,
            "pieces": cut(blob_bytes, [offs[i] for i in range(n + 1)], n),
            "n": n,
            "truncate_dim": trunc,
            "out_len": out_len,
            **extra,
        }

    def tessl_embedgemma2_embed_image_sources(self, h, source, blob, blob_len, offs, n, trunc, out, out_len, err, nn):
        record = self._source_record("image_sources", source, blob, blob_len, offs, n, trunc, out_len)
        return self._answer("sources", record, out, out_len, err)

    def tessl_embedgemma2_embed_audio_sources(self, h, source, blob, blob_len, offs, n, trunc, out, out_len, err, nn):
        record = self._source_record("audio_sources", source, blob, blob_len, offs, n, trunc, out_len)
        return self._answer("sources", record, out, out_len, err)

    def tessl_embedgemma2_embed_video_sources(
        self, h, source, blob, blob_len, offs, n, ext, ext_len, stamps, trunc, out, out_len, err, nn
    ):
        record = self._source_record(
            "video_sources", source, blob, blob_len, offs, n, trunc, out_len,
            extension=ext, extension_len=ext_len, timestamps=stamps,
        )  # fmt: skip
        return self._answer("sources", record, out, out_len, err)

    def tessl_embedgemma2_image_soft_token_count(self, h, dims, n, out, out_len, err, nn):
        self.counts.append({"entry": "image", "dims": [dims[i] for i in range(2 * n)], "n": n, "out_len": out_len})
        if self.media_status:
            err.value = b"image count failed"
            return self.media_status
        for i in range(out_len):
            out[i] = 10 + i
        return 0

    def tessl_embedgemma2_audio_soft_token_count(self, h, samples, n, rate, out, out_len, err, nn):
        self.counts.append(
            {"entry": "audio", "samples": [samples[i] for i in range(n)], "rate": rate, "n": n, "out_len": out_len}
        )
        if self.media_status:
            err.value = b"audio count failed"
            return self.media_status
        for i in range(out_len):
            out[i] = 10 + i
        return 0

    def tessl_embedgemma2_video_soft_token_count(self, h, dims, frames, fps, n, out, out_len, err, nn):
        self.counts.append(
            {
                "entry": "video",
                "dims": [dims[i] for i in range(2 * n)],
                "frames": [frames[i] for i in range(n)],
                "fps": [fps[i] for i in range(n)],
                "n": n,
                "out_len": out_len,
            }
        )
        if self.media_status:
            err.value = b"video count failed"
            return self.media_status
        for i in range(n):
            out[2 * i], out[2 * i + 1] = 4, 9 + i
        return 0

    def tessl_embedgemma2_embed_segments(
        self, h, tb, tb_len, toffs, n_texts, kinds, refs, n_seg, ioffs, n_items,
        isrc, iblob, iblob_len, ioff, idims, n_img,
        asrc, ablob, ablob_len, aoff, rate, n_clip,
        vsrc, vblob, vblob_len, voff, vdims, vfps, n_vid,
        ext, ext_len, trunc, out, out_len, err, nn,
    ):  # fmt: skip
        def blob_of(b, n_bytes):
            return bytes(b[i] for i in range(n_bytes)) if b is not None else None

        def offsets(o, n):
            return [o[i] for i in range(n + 1)] if o is not None else None

        record = {
            "texts": cut(blob_of(tb, tb_len) or b"", offsets(toffs, n_texts) or [0], n_texts),
            "n_texts": n_texts,
            "kinds": [kinds[i] for i in range(n_seg)],
            "refs": [refs[i] for i in range(n_seg)],
            "item_offsets": [ioffs[i] for i in range(n_items + 1)],
            "image": {"source": isrc, "blob": blob_of(iblob, iblob_len), "offsets": offsets(ioff, n_img),
                      "dims": [idims[i] for i in range(3 * n_img)] if idims is not None else None, "n": n_img},
            "audio": {"source": asrc, "blob": blob_of(ablob, ablob_len), "offsets": offsets(aoff, n_clip),
                      "rate": rate, "n": n_clip},
            "video": {"source": vsrc, "blob": blob_of(vblob, vblob_len), "offsets": offsets(voff, n_vid),
                      "dims": [vdims[i] for i in range(2 * n_vid)] if vdims is not None else None,
                      "fps": [vfps[i] for i in range(n_vid)] if vfps is not None else None, "n": n_vid},
            "extension": ext,
            "extension_len": ext_len,
            "truncate_dim": trunc,
            "out_len": out_len,
        }  # fmt: skip
        return self._answer("segments", record, out, out_len, err)


class MediaCase(MmCase):
    def setUp(self):
        super().setUp()
        self.lib = FakeMediaLib()
        # MmCase.setUp patched _load with its own stand-in; point it at this one.
        import tessl_torch
        from unittest import mock

        patch = mock.patch.object(tessl_torch, "_load", return_value=self.lib)
        patch.start()
        self.addCleanup(patch.stop)


class Sources(MediaCase):
    def test_image_files_become_one_blob_of_utf8_paths_and_byte_cuts(self):
        with self.mm() as model:
            out = model.embed_image_files(["a.png", Path("dir/b é.jpg")])
            self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
            self.assertEqual(out.ravel().tolist(), list(map(float, range(2 * DIM))))
            (call,) = self.lib.sources
            self.assertEqual(call["entry"], "image_sources")
            self.assertEqual((call["source"], call["n"], call["truncate_dim"], call["out_len"]), (PATHS, 2, 0, 2 * DIM))
            self.assertEqual(call["pieces"], [b"a.png", "dir/b é.jpg".encode()])
            # One path is a batch of one.
            self.assertEqual(model.embed_image_files("only.png", truncate_dim=3).shape, (1, 3))
            self.assertEqual(self.lib.sources[-1]["pieces"], [b"only.png"])
            self.assertEqual(self.lib.sources[-1]["truncate_dim"], 3)

    def test_encoded_bytes_of_any_buffer_type_are_laid_end_to_end(self):
        with self.mm() as model:
            model.embed_image_bytes([b"\x89PNG", bytearray(b"\xff\xd8"), memoryview(b"GIF8")])
            self.assertEqual(self.lib.sources[-1]["source"], BYTES)
            self.assertEqual(self.lib.sources[-1]["pieces"], [b"\x89PNG", b"\xff\xd8", b"GIF8"])
            model.embed_audio_bytes(b"RIFF....")
            self.assertEqual(self.lib.sources[-1]["entry"], "audio_sources")
            self.assertEqual(self.lib.sources[-1]["pieces"], [b"RIFF...."])
            model.embed_audio_files(["x.wav"])
            self.assertEqual((self.lib.sources[-1]["entry"], self.lib.sources[-1]["source"]), ("audio_sources", PATHS))

    def test_videos_carry_the_extension_and_the_timestamps_flag(self):
        with self.mm() as model:
            model.embed_video_files(["a.mp4"], timestamps=True)
            call = self.lib.sources[-1]
            self.assertEqual((call["entry"], call["source"], call["timestamps"]), ("video_sources", PATHS, 1))
            self.assertEqual((call["extension"], call["extension_len"]), (None, 0))
            model.embed_video_bytes([b"\x00\x00ftyp"], extension="mov")
            call = self.lib.sources[-1]
            self.assertEqual((call["source"], call["timestamps"], call["extension"], call["extension_len"]), (BYTES, 0, b"mov", 3))
            model.embed_video_bytes(b"v")
            self.assertEqual(self.lib.sources[-1]["extension"], b"mp4")

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        long = "x" * 4
        with self.mm() as model:
            for what, call in {
                "no paths": lambda: model.embed_image_files([]),
                "a dict": lambda: model.embed_image_files({"a": 1}),
                "a number": lambda: model.embed_image_files(3),
                "bytes as a path": lambda: model.embed_image_files([b"a.png"]),
                "a path as bytes": lambda: model.embed_image_bytes(["a.png"]),
                "a number in a list": lambda: model.embed_audio_files(["a.wav", 7]),
                "a lone surrogate": lambda: model.embed_image_files(["\ud800.png"]),
                "an array as bytes": lambda: model.embed_audio_bytes([np.zeros(4, np.uint8)]),
                "an extension that is not a str": lambda: model.embed_video_bytes([b"v"], extension=3),
                "timestamps that are not a bool": lambda: model.embed_video_files(["a.mp4"], timestamps="yes"),
                "truncate_dim 0": lambda: model.embed_image_files([long], truncate_dim=0),
            }.items():
                with self.subTest(what), self.assertRaises(TesslError):
                    call()
        self.assertEqual(self.lib.sources, [])

    def test_a_text_only_load_names_the_multimodal_load(self):
        with eg2.EmbedGemma2.from_pretrained(self.snapshot) as model:
            for call in (
                lambda: model.embed_image_files(["a.png"]),
                lambda: model.embed_audio_bytes([b"a"]),
                lambda: model.embed_video_files(["a.mp4"]),
                lambda: model.media([{"segments": ["a"]}]),
            ):
                with self.assertRaisesRegex(TesslError, "multimodal=True"):
                    call()
        self.assertEqual(self.lib.sources + self.lib.segments, [])

    def test_a_refusal_raises_the_librarys_message_and_leaves_the_model_open(self):
        with self.mm() as model:
            self.lib.media_status = 1
            with self.assertRaisesRegex(TesslError, "image_sources failed"):
                model.embed_image_files(["a.png"])
            self.lib.media_status = 0
            self.assertEqual(model.embed_image_files(["a.png"]).shape, (1, DIM))


class Counts(MediaCase):
    def test_image_sizes_cross_as_a_pair_table_and_come_back_as_ints(self):
        with self.text_or_mm() as model:
            self.assertEqual(model.image_soft_token_count((480, 640)), [10])
            self.assertEqual(model.image_soft_token_count([(8, 8), np.array([9, 33])]), [10, 11])
            self.assertEqual(self.lib.counts[-1]["dims"], [8, 8, 9, 33])
            self.assertTrue(all(type(v) is int for v in model.image_soft_token_count([(1, 2)])))

    def test_clip_samples_cross_with_the_rate(self):
        with self.text_or_mm() as model:
            self.assertEqual(model.audio_soft_token_count(16000), [10])
            self.assertEqual(model.audio_soft_token_count([16000, 480000], sampling_rate=8000), [10, 11])
            call = self.lib.counts[-1]
            self.assertEqual((call["samples"], call["rate"]), ([16000, 480000], 8000))

    def test_videos_come_back_as_sampled_frames_and_tokens_per_frame(self):
        with self.text_or_mm() as model:
            self.assertEqual(model.video_soft_token_count((48, 64), 12, 10.0), [(4, 9)])
            self.assertEqual(model.video_soft_token_count([(8, 8), (9, 9)], [12, 30], [10, 24.0]), [(4, 9), (4, 10)])
            call = self.lib.counts[-1]
            self.assertEqual((call["dims"], call["frames"], call["fps"]), ([8, 8, 9, 9], [12, 30], [10.0, 24.0]))
            # One frame count for every video.
            model.video_soft_token_count([(8, 8), (9, 9)], 12, 5)
            self.assertEqual(self.lib.counts[-1]["frames"], [12, 12])

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        with self.text_or_mm() as model:
            for what, call in {
                "no sizes": lambda: model.image_soft_token_count([]),
                "three numbers": lambda: model.image_soft_token_count([(1, 2, 3)]),
                "a float size": lambda: model.image_soft_token_count([(1.5, 2)]),
                "a negative size": lambda: model.image_soft_token_count([(-1, 2)]),
                "a size past u32": lambda: model.image_soft_token_count([(1 << 32, 2)]),
                "a ragged list": lambda: model.image_soft_token_count([(1, 2), (3,)]),
                "no samples": lambda: model.audio_soft_token_count([]),
                "negative samples": lambda: model.audio_soft_token_count([-3]),
                "float samples": lambda: model.audio_soft_token_count([1.5]),
                "a bad rate": lambda: model.audio_soft_token_count([100], sampling_rate=-1),
                "frames of another count": lambda: model.video_soft_token_count([(8, 8), (9, 9)], [1, 2, 3], 5),
                "fps of another count": lambda: model.video_soft_token_count([(8, 8)], 4, [1, 2]),
                "fps that is a str": lambda: model.video_soft_token_count([(8, 8)], 4, "5"),
            }.items():
                with self.subTest(what), self.assertRaises(TesslError):
                    call()
        self.assertEqual(self.lib.counts, [])

    def test_the_library_judges_the_values(self):
        with self.text_or_mm() as model:
            self.lib.media_status = 1
            with self.assertRaisesRegex(TesslError, "image count failed"):
                model.image_soft_token_count((0, 8))

    def text_or_mm(self):
        # The counts work on either kind of load: use the text-only one.
        return eg2.EmbedGemma2.from_pretrained(self.snapshot)


class Segments(MediaCase):
    def call(self, items, **kw):
        with self.mm() as model:
            out = model.media(items, **kw)
        return out, self.lib.segments[-1]

    def test_text_is_deduplicated_and_items_are_cut_by_segment(self):
        out, c = self.call([{"segments": ["a", "b", "a"]}, {"segments": ["b"]}])
        self.assertEqual((out.shape, out.dtype), ((2, DIM), np.float32))
        self.assertEqual(c["texts"], [b"a", b"b"])
        self.assertEqual((c["kinds"], c["refs"], c["item_offsets"]), ([TEXT] * 4, [0, 1, 0, 1], [0, 3, 4]))
        for modality in ("image", "audio", "video"):
            self.assertEqual(c[modality]["n"], 0)
            self.assertIsNone(c[modality]["blob"])
        self.assertIsNone(c["extension"])
        self.assertEqual((c["video"]["offsets"], c["image"]["dims"], c["video"]["dims"]), (None, None, None))

    def test_raw_media_cross_with_dims_byte_cuts_and_a_named_medium_once(self):
        a, b = rgb(3, 4, 3, 1), rgb(2, 2, 1, 2)
        pcm = np.linspace(-1, 1, 5, dtype=np.float32)
        v = video(2, 4, 5, 3)
        items = [
            {"segments": ["x", {"image": a}, {"audio": pcm}, {"video": (v, 3.0), "timestamps": True}, {"image": a}]},
            {"segments": [{"image": b}, "x", {"video": (v, 3.0)}]},
        ]
        out, c = self.call(items)
        self.assertEqual(c["texts"], [b"x"])
        self.assertEqual(c["kinds"], [TEXT, IMAGE, AUDIO, VIDEO_T, IMAGE, IMAGE, TEXT, VIDEO])
        self.assertEqual(c["refs"], [0, 0, 0, 0, 0, 1, 0, 1], "a is one image, named twice; the second (v, 3.0) is its own")
        self.assertEqual(c["item_offsets"], [0, 5, 8])
        img = c["image"]
        self.assertEqual((img["source"], img["n"], img["dims"]), (RAW, 2, [3, 4, 3, 2, 2, 1]))
        self.assertEqual((img["offsets"], img["blob"]), ([0, 36, 40], a.tobytes() + b.tobytes()))
        aud = c["audio"]
        self.assertEqual((aud["source"], aud["n"], aud["rate"]), (RAW, 1, 16000))
        self.assertEqual((aud["offsets"], aud["blob"]), ([0, 20], pcm.tobytes()), "byte cuts, not sample cuts")
        vid = c["video"]
        self.assertEqual((vid["source"], vid["n"], vid["dims"], vid["fps"]), (RAW, 2, [4, 5, 4, 5], [3.0, 3.0]))
        self.assertEqual(vid["offsets"], [0, 2 * 4 * 5 * 3, 4 * 4 * 5 * 3])
        self.assertIsNone(c["extension"])

    def test_the_same_object_is_one_medium_but_equal_arrays_are_two(self):
        a = rgb(3, 3, 3, 1)
        _, c = self.call([{"segments": [{"image": a}, {"image": a}, {"image": a.copy()}]}])
        self.assertEqual((c["refs"], c["image"]["n"]), ([0, 0, 1], 2))

    def test_paths_and_bytes_are_a_modalitys_source_and_a_sampling_rate_is_for_pcm_only(self):
        items = [{"segments": [{"image": "a.png"}, "t", {"audio": Path("c.wav")}, {"video": "v.mp4", "timestamps": False}]}]
        _, c = self.call(items, sampling_rate=44100)
        self.assertEqual((c["image"]["source"], c["image"]["blob"], c["image"]["offsets"]), (PATHS, b"a.png", [0, 5]))
        self.assertEqual((c["audio"]["source"], c["audio"]["blob"], c["audio"]["rate"]), (PATHS, b"c.wav", 0))
        self.assertEqual((c["video"]["source"], c["video"]["blob"], c["video"]["dims"]), (PATHS, b"v.mp4", None))
        self.assertEqual((c["extension"], c["extension_len"]), (None, 0), "no extension for paths")
        self.assertEqual(c["kinds"], [IMAGE, TEXT, AUDIO, VIDEO])

        items = [{"segments": [{"image": b"\x89PNG"}, {"audio": b"RIFF"}, {"video": b"\x00ftyp"}]}]
        _, c = self.call(items, video_extension="mov")
        self.assertEqual([c[m]["source"] for m in ("image", "audio", "video")], [BYTES] * 3)
        self.assertEqual((c["extension"], c["extension_len"]), (b"mov", 3))

    def test_the_other_forms_and_options_reach_the_library(self):
        _, c = self.call([{"segments": ["a"]}], truncate_dim=4)
        self.assertEqual((c["truncate_dim"], c["out_len"]), (4, 4))
        # The ids form still goes to embed_media.
        with self.mm() as model:
            model.media([{"ids": [1, 2]}])
        self.assertEqual(len(self.lib.raw_media), 1)

    def test_bad_arguments_are_refused_before_the_library_is_called(self):
        a = rgb(3, 3, 3, 1)
        cases = {
            "an item that is not a mapping": ["a"],
            "segments not a list": [{"segments": "abc"}],
            "no segments": [{"segments": []}],
            "a segment that is bytes": [{"segments": [b"abc"]}],
            "a segment that is a number": [{"segments": [3]}],
            "an unknown key": [{"segments": [{"imag": a}]}],
            "two media in one segment": [{"segments": [{"image": a, "audio": a}]}],
            "no medium": [{"segments": [{"timestamps": True}]}],
            "timestamps on an image": [{"segments": [{"image": a, "timestamps": True}]}],
            "timestamps not a bool": [{"segments": [{"video": ("v.mp4"), "timestamps": 2}]}],
            "segments beside ids": [{"segments": ["a"], "ids": [1]}],
            "segments in one item and ids in another": [{"segments": ["a"]}, {"ids": [1]}],
            "an image that is not uint8": [{"segments": [{"image": a.astype(np.float32)}]}],
            "a clip that is not float": [{"segments": [{"audio": np.zeros(4, np.int16)}]}],
            "a video that is not a pair": [{"segments": [{"video": video(2, 4, 4)}]}],
            "images of two sources": [{"segments": [{"image": a}, {"image": "a.png"}]}],
            "files and bytes": [{"segments": [{"audio": "a.wav"}, {"audio": b"RIFF"}]}],
            "a lone surrogate path": [{"segments": [{"image": "\ud800"}]}],
            "an extension that is not a str": [{"segments": [{"video": b"v"}]}],
        }
        with self.mm() as model:
            for what, items in cases.items():
                kw = {"video_extension": 3} if what == "an extension that is not a str" else {}
                with self.subTest(what), self.assertRaises(TesslError):
                    model.media(items, **kw)
            with self.assertRaises(TesslError):
                model.media([{"segments": ["a"]}], truncate_dim=0)
        self.assertEqual(self.lib.segments, [])

    def test_a_mixed_source_error_names_both_segments(self):
        a = rgb(3, 3, 3, 1)
        with self.mm() as model:
            with self.assertRaisesRegex(
                TesslError, r"images of one call must all be arrays, all paths or all encoded bytes: "
                r"items\[0\]\['segments'\]\[0\]\['image'\] is an array but items\[0\]\['segments'\]\[1\]\['image'\] is a path"
            ):
                model.media([{"segments": [{"image": a}, {"image": "a.png"}]}])

    def test_a_refusal_raises_and_leaves_the_model_open(self):
        with self.mm() as model:
            self.lib.media_status = 1
            with self.assertRaisesRegex(TesslError, "segments failed"):
                model.media([{"segments": ["a"]}])
            self.lib.media_status = 0
            self.assertEqual(model.media([{"segments": ["a"]}]).shape, (1, DIM))


class Declarations(unittest.TestCase):
    def test_the_media_declarations_skip_a_stand_in_and_name_a_library_without_the_entries(self):
        eg2._declare_media(FakeMediaLib())
        with self.assertRaisesRegex(TesslError, "embed_image_sources.*cargo build"):
            eg2._declare_media(ctypes.CDLL(None))


if __name__ == "__main__":
    unittest.main()
