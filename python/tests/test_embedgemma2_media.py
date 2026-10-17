"""tessl_torch.embedgemma2's media files, bytes, soft-token counts and segment-built items
against the built library and a device, with no ``transformers``, ``tokenizers`` or processor.

Two layers (argument handling against a stand-in library is ``test_embedgemma2_media_host.py``):

* ``Library``: the built ``libtessl.dylib`` and a device, on the tiny multimodal snapshot that
  ``cargo test --release --test capi_embedgemma2_media`` writes under ``target/`` and the media of
  ``tests/fixtures/media``: a file, the same file's bytes and its decoded array give the same row;
  a batch is its rows; a segment-built item is the embed_* call it spells; what the library refuses
  that Python cannot see names the file;
* ``RealCheckpoint``: the real ``google/embeddinggemma-2`` snapshot against the references
  ``tools/embedgemma2_ref/make_segments_reference.py`` wrote with the transformers processor.
  Skipped unless ``EMBEDGEMMA2_SNAPSHOT`` is set and the references exist:

      cargo build --release
      ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_segments_reference.py
      EMBEDGEMMA2_SNAPSHOT=/path/to/snapshot \\
      EMBEDGEMMA2_SEG_REF_DIR=/path/to/target/embedgemma2_seg_ref \\
          python3 -m unittest python/tests/test_embedgemma2_media.py -v

  The bounds are ``tests/embedgemma2_segments_model.rs``'s: max abs ``1e-4`` and cosine
  ``>= 0.99999``; the segment-built item of each case is held to the processor's embedding, and the
  soft-token counts to the placeholders of the processor's ids.
"""

import json
import os
import struct
import sys
import tempfile
import unittest
import wave
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402
from tessl_torch.embedgemma2 import EmbedGemma2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "media"
EMB_ABS = 1e-4
EMB_COS = 0.99999
BATCH_ABS = 1e-5

LIB_BUILT = tessl_torch.library_path().exists()
TINY = ROOT / "target" / "capi_embedgemma2_media_images"
TINY_DIM = 96


def eprint(*args):
    print(*args, file=sys.stderr)


def max_abs_and_cosine(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return np.abs(got - want).max(), cos.min()


def png_bytes(array):
    """A lossless 8-bit PNG of a uint8 ``[H, W, 3]`` array (standard library only)."""
    h, w, _ = array.shape
    raw = b"".join(b"\x00" + array[y].tobytes() for y in range(h))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def wav_bytes(samples_i16, rate=16000):
    """A mono PCM16 WAV of an int16 array."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples_i16.astype("<i2").tobytes())
    return buf.getvalue()


def rgb(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(
    TINY.exists(), "run `cargo test --release --test capi_embedgemma2_media` to write the tiny snapshot"
)
class Library(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)

    def write(self, name, data):
        path = Path(self.tmp.name) / name
        path.write_bytes(data)
        return path

    def test_an_image_file_its_bytes_and_its_array_are_the_same_row(self):
        pixels = rgb(40, 56, 1)
        path = self.write("a.png", png_bytes(pixels))
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            want = model.embed_images([pixels])
            self.assertEqual((want.shape, want.dtype), ((1, TINY_DIM), np.float32))
            np.testing.assert_array_equal(model.embed_image_files([path]), want)
            np.testing.assert_array_equal(model.embed_image_files(str(path)), want)
            np.testing.assert_array_equal(model.embed_image_bytes([path.read_bytes()]), want)

    def test_fixture_images_batch_by_file_and_by_bytes(self):
        names = ["rgb.png", "gray.png", "rgb.bmp", "rgb.tiff", "rgb.jpg"]
        paths = [FIXTURES / n for n in names]
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            batch = model.embed_image_files(paths)
            self.assertEqual(batch.shape, (len(names), TINY_DIM))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            np.testing.assert_array_equal(model.embed_image_bytes([p.read_bytes() for p in paths]), batch)
            for i, p in enumerate(paths):
                np.testing.assert_allclose(model.embed_image_files([p])[0], batch[i], atol=BATCH_ABS)
            cut = model.embed_image_files(paths, truncate_dim=40)
            want = batch[:, :40] / np.linalg.norm(batch[:, :40], axis=1, keepdims=True)
            np.testing.assert_allclose(cut, want, atol=1e-5)

    def test_an_audio_file_its_bytes_and_its_samples_are_the_same_row(self):
        samples = (12000 * np.sin(np.arange(8000) * 0.05)).astype(np.int16)
        path = self.write("a.wav", wav_bytes(samples))
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            want = model.embed_audio([samples.astype(np.float32) / 32768.0])
            got = model.embed_audio_files(path)
            np.testing.assert_allclose(got, want, atol=1e-6)
            np.testing.assert_array_equal(model.embed_audio_bytes(path.read_bytes()), got)
            names = ["pcm16_16k_mono.wav", "pcm16_44k_stereo.wav", "tone_44k_stereo.flac", "tone_16k_mono.mp3"]
            paths = [FIXTURES / n for n in names]
            batch = model.embed_audio_files(paths)
            self.assertEqual(batch.shape, (len(names), TINY_DIM))
            np.testing.assert_array_equal(model.embed_audio_bytes([p.read_bytes() for p in paths]), batch)

    def test_a_video_file_and_its_bytes_are_the_same_row_and_timestamps_change_it(self):
        path = FIXTURES / "colors.mp4"
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            plain = model.embed_video_files([path])
            self.assertEqual(plain.shape, (1, TINY_DIM))
            np.testing.assert_array_equal(model.embed_video_bytes([path.read_bytes()]), plain)
            np.testing.assert_array_equal(model.embed_video_bytes(path.read_bytes(), extension="mp4"), plain)
            stamped = model.embed_video_files(path, timestamps=True)
            self.assertGreater(np.abs(stamped - plain).max(), 1e-4, "the timestamps changed nothing")
            np.testing.assert_array_equal(model.embed_video_bytes(path.read_bytes(), timestamps=True), stamped)
            # 12 frames at 10 fps, sampled at the snapshot's 5 fps and capped at 4 frames.
            (kept, per_frame), = model.video_soft_token_count((48, 64), 12, 10.0)
            self.assertEqual(kept, 4)
            self.assertGreater(per_frame, 0)

    def test_soft_token_counts_are_positive_bounded_and_work_without_the_towers(self):
        with EmbedGemma2.from_pretrained(TINY) as model:
            counts = model.image_soft_token_count([(8, 8), (64, 64), (200, 31)])
            self.assertTrue(all(0 < c <= 70 for c in counts), counts)
            self.assertEqual(model.image_soft_token_count((64, 64)), [counts[1]])
            audio = model.audio_soft_token_count([1600, 16000, 480000, 480001])
            self.assertEqual(audio[2], audio[3], "a clip is cut to 30 s")
            self.assertLessEqual(audio[2], 750)
            self.assertLess(audio[0], audio[1])
            with self.assertRaisesRegex(TesslError, "clip 1 has 0 samples"):
                model.audio_soft_token_count([16000, 0])
            with self.assertRaisesRegex(TesslError, "image 1 is 0x8"):
                model.image_soft_token_count([(8, 8), (0, 8)])
            with self.assertRaisesRegex(TesslError, "8000 Hz"):
                model.audio_soft_token_count([16000], sampling_rate=8000)

    def test_a_segment_item_is_the_call_it_spells_and_files_equal_arrays(self):
        pixels = rgb(40, 56, 2)
        path = self.write("seg.png", png_bytes(pixels))
        pcm = (0.4 * np.sin(np.arange(6400) * 0.05)).astype(np.float32)
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            alone = model.media([{"segments": [{"image": pixels}]}])
            np.testing.assert_allclose(alone, model.embed_images([pixels]), atol=1e-6)
            clip = model.media([{"segments": [{"audio": pcm}]}])
            np.testing.assert_allclose(clip, model.embed_audio([pcm]), atol=1e-6)

            raw = model.media([{"segments": ["1:", {"image": pixels}, "0 0"]}])
            as_file = model.media([{"segments": ["1:", {"image": path}, "0 0"]}])
            as_bytes = model.media([{"segments": ["1:", {"image": path.read_bytes()}, "0 0"]}])
            np.testing.assert_array_equal(as_file, raw)
            np.testing.assert_array_equal(as_bytes, raw)
            self.assertGreater(np.abs(raw - alone).max(), 1e-4, "the text changed nothing")

            # An item in a batch is the item alone (to GEMM rounding), and a medium named
            # twice is decoded once and goes through its tower twice.
            items = [
                {"segments": ["1:", {"image": path}, "0 0"]},
                {"segments": [{"audio": FIXTURES / "pcm16_16k_mono.wav"}, "00"]},
                {"segments": [{"image": path}, {"image": path}, {"video": FIXTURES / "colors.mp4", "timestamps": True}]},
            ]
            batch = model.media(items)
            self.assertEqual(batch.shape, (3, TINY_DIM))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            for i, item in enumerate(items):
                np.testing.assert_allclose(model.media([item])[0], batch[i], atol=BATCH_ABS)
            cut = model.media(items, truncate_dim=40)
            want = batch[:, :40] / np.linalg.norm(batch[:, :40], axis=1, keepdims=True)
            np.testing.assert_allclose(cut, want, atol=1e-5)

            # Video bytes need the extension AVFoundation sniffs them as.
            vb = (FIXTURES / "colors.mp4").read_bytes()
            as_bytes = model.media([{"segments": ["0", {"video": vb}]}], video_extension="mp4")
            as_file = model.media([{"segments": ["0", {"video": FIXTURES / "colors.mp4"}]}])
            np.testing.assert_array_equal(as_bytes, as_file)

    def test_the_library_names_the_file_it_refuses(self):
        good = FIXTURES / "rgb.png"
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            with self.assertRaisesRegex(TesslError, "file 1: cannot read /no/such/missing.png"):
                model.embed_image_files([good, "/no/such/missing.png"])
            with self.assertRaisesRegex(TesslError, "bytes 1: image:"):
                model.embed_image_bytes([good.read_bytes(), b"not an image"])
            with self.assertRaisesRegex(TesslError, "file 0: audio:"):
                model.embed_audio_files([FIXTURES / "rgb.png"])
            with self.assertRaisesRegex(TesslError, "video"):
                model.embed_video_bytes([b"not a video"])
            with self.assertRaisesRegex(TesslError, "file 0: cannot read /no/such/missing.png"):
                model.media([{"segments": ["x", {"image": "/no/such/missing.png"}]}])
            with self.assertRaisesRegex(TesslError, "n_images is 257"):
                model.embed_image_files([good] * 257)
            with self.assertRaisesRegex(TesslError, "text 0 is not named|never empty|empty"):
                model.media([{"segments": []}])
            self.assertEqual(model.embed_image_files([good]).shape, (1, TINY_DIM), "usable after the refusals")

    def test_no_transformers_or_tokenizers_are_imported(self):
        import subprocess

        code = (
            "import sys; from pathlib import Path; sys.path.insert(0, %r)\n"
            "from tessl_torch.embedgemma2 import EmbedGemma2\n"
            "m = EmbedGemma2.from_pretrained(%r, multimodal=True)\n"
            "m.embed_image_files(%r); m.media([{'segments': ['0', {'image': %r}]}]); m.image_soft_token_count((8, 8))\n"
            "bad = sorted(k for k in ('transformers', 'tokenizers', 'PIL') if k in sys.modules)\n"
            "print('IMPORTED', bad)\n"
            % (str(ROOT / "python"), str(TINY), str(FIXTURES / "rgb.png"), str(FIXTURES / "rgb.png"))
        )
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("IMPORTED []", out.stdout)


# ---------------------------------------------------------------------------
# The real checkpoint against the processor's references.
# ---------------------------------------------------------------------------

SNAPSHOT = os.environ.get("EMBEDGEMMA2_SNAPSHOT")
SEG_REF = Path(os.environ.get("EMBEDGEMMA2_SEG_REF_DIR", ROOT / "target" / "embedgemma2_seg_ref"))
VIDEO_FPS = 4.0

CASES = {
    "t_i_t": ["a photo of", "IMAGE", "on a table"],
    "t_a_t": ["listen:", "AUDIO", "(a chirp)"],
    "t_v_t": ["watch", "VIDEO", "now"],
    "i_t": ["IMAGE", "alone with text"],
    "t_i_a_v": ["x", "IMAGE", "y", "AUDIO", "z", "VIDEO"],
    "v_v": ["VIDEO", "VIDEO"],
    "t_vt_t": ["at", "VIDEO_T", "end"],
}


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(SNAPSHOT, "set EMBEDGEMMA2_SNAPSHOT to the google/embeddinggemma-2 snapshot directory")
@unittest.skipUnless(
    (SEG_REF / "cases.json").exists(),
    "run tools/embedgemma2_ref/make_segments_reference.py to write the references",
)
class RealCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = np.load(SEG_REF / "seg_video_raw.npy").astype(np.uint8)
        cls.image_path = SEG_REF / "seg_image.png"
        cls.clip_path = SEG_REF / "seg_clip.wav"

    def segments(self, pieces, how):
        """The pieces as segments; the image and the clip as ``how`` (``files`` or ``bytes``),
        the video as the raw frames the reference stored."""
        medium = {
            "files": {"IMAGE": self.image_path, "AUDIO": self.clip_path},
            "bytes": {"IMAGE": self.image_path.read_bytes(), "AUDIO": self.clip_path.read_bytes()},
        }[how]
        out = []
        for p in pieces:
            if p == "IMAGE":
                out.append({"image": medium["IMAGE"]})
            elif p == "AUDIO":
                out.append({"audio": medium["AUDIO"]})
            elif p == "VIDEO":
                out.append({"video": (self.video, VIDEO_FPS)})
            elif p == "VIDEO_T":
                out.append({"video": (self.video, VIDEO_FPS), "timestamps": True})
            else:
                out.append(p)
        return out

    def test_a_segment_built_item_matches_the_processor_built_reference(self):
        worst, worst_cos = 0.0, 1.0
        with EmbedGemma2.from_pretrained(SNAPSHOT, multimodal=True) as model:
            for how in ("files", "bytes"):
                items = [{"segments": self.segments(pieces, how)} for pieces in CASES.values()]
                got = model.media(items)
                for (name, _), row in zip(CASES.items(), got):
                    want = np.load(SEG_REF / f"{name}_emb_f32.npy")
                    err, cos = max_abs_and_cosine(row, want)
                    eprint(f"{name} ({how}, one call): max abs {err:.3e}, cosine {cos:.8f}")
                    worst, worst_cos = max(worst, err), min(worst_cos, cos)
                    self.assertLessEqual(err, EMB_ABS, f"{name} ({how})")
                    self.assertGreaterEqual(cos, EMB_COS, f"{name} ({how})")
        eprint(f"worst over {len(CASES)} cases, files and bytes: max abs {worst:.3e}, cosine {worst_cos:.8f}")

    def test_the_soft_token_counts_are_the_processors_placeholder_counts(self):
        config = json.loads((Path(SNAPSHOT) / "config.json").read_text())
        image_id, audio_id = config["image_token_id"], config["audio_token_id"]
        video_id = config["video_token_id"]
        with EmbedGemma2.from_pretrained(SNAPSHOT) as model:  # the counts need no tower
            image_ids = np.load(SEG_REF / "i_t_ids.npy")
            self.assertEqual(model.image_soft_token_count((224, 224)), [int((image_ids == image_id).sum())])
            clip_ids = np.load(SEG_REF / "t_a_t_ids.npy")
            self.assertEqual(model.audio_soft_token_count(16000), [int((clip_ids == audio_id).sum())])
            frames, h, w, _ = self.video.shape
            (kept, per_frame), = model.video_soft_token_count((h, w), frames, VIDEO_FPS)
            video_ids = np.load(SEG_REF / "t_v_t_ids.npy")
            self.assertEqual(kept * per_frame, int((video_ids == video_id).sum()))


if __name__ == "__main__":
    unittest.main()
