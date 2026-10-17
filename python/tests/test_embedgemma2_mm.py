"""tessl_torch.embedgemma2's images and audio against the built library and a
device.

Two layers (argument handling against a stand-in library is
``test_embedgemma2_mm_host.py``):

* ``Library``: the built ``libtessl.dylib`` and a device, on the tiny
  multimodal checkpoint that ``cargo test --release --test
  capi_embedgemma2_multimodal`` writes under ``target/``: what the library
  refuses that Python cannot see, and rows that are unit length and do not
  depend on the batch;
* ``RealCheckpoint``: the real ``google/embeddinggemma-2`` snapshot against the
  references ``tools/embedgemma2_ref/make_mm_reference.py`` wrote from
  sentence-transformers (fp32, sdpa, CPU): the processor's own pixels and
  log-mel features, saved as the model received them. Skipped unless
  ``EMBEDGEMMA2_SNAPSHOT`` is set:

      cargo build --release
      EMBEDGEMMA2_SNAPSHOT=/path/to/snapshot \\
      EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \\
          python3 -m unittest python/tests/test_embedgemma2_mm.py -v

  The bounds are ``tests/embedgemma2_e2e_model.rs``'s, fixed before the first
  run: max abs ``1e-4`` and cosine ``>= 0.99999`` against ``emb_f32``; an
  input alone vs in a batch within ``1e-5``; the processor path (which needs
  transformers and PIL and rebuilds the generator's synthetic inputs) must hand
  the model the saved ``pixel_values`` and ``input_features`` to ``1e-6``, and
  ``encode_images`` / ``encode_audio`` on its output embed to the same bounds.
  (Raw arrays through ``embed_images`` / ``embed_audio`` / ``embed_video``, with
  no processor, are ``test_embedgemma2_raw.py``.)
"""

import importlib.util
import os
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402
from tessl_torch.embedgemma2 import EmbedGemma2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EMB_ABS = 1e-4
EMB_COS = 0.99999
BATCH_ABS = 1e-5
INPUT_ABS = 1e-6

LIB_BUILT = tessl_torch.library_path().exists()
TINY = ROOT / "target" / "capi_embedgemma2_mm_tiny"
TINY_DIM = 96


def max_abs_and_cosine(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return np.abs(got - want).max(), cos.min()


# ---------------------------------------------------------------------------
# The built library on a device, with the tiny checkpoint.
# ---------------------------------------------------------------------------

def tiny_image(w, h, pad, seed, patch_dim=48):
    rng = np.random.default_rng(seed)
    n = w * h
    pos = [[x, y] for y in range(h) for x in range(w)] + [[-1, -1]] * pad
    return rng.random((n + pad, patch_dim), dtype=np.float32), np.array(pos, dtype=np.int64)


def tiny_clip(frames, seed, valid=None, mels=32):
    rng = np.random.default_rng(seed)
    return 2 * rng.random((frames, mels), dtype=np.float32) - 1, np.arange(frames) < (valid or frames)


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(
    TINY.exists(), "run `cargo test --release --test capi_embedgemma2_multimodal` to write the tiny checkpoint"
)
class Library(unittest.TestCase):
    def test_images_and_audio_embed_to_unit_rows_that_do_not_depend_on_the_batch(self):
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            self.assertEqual(model.embedding_dim, TINY_DIM)
            imgs = [tiny_image(8, 6, 16, 1), tiny_image(5, 4, 28, 2), tiny_image(2, 2, 0, 3)]
            batch = model.encode_images([p for p, _ in imgs], [q for _, q in imgs])
            self.assertEqual((batch.shape, batch.dtype), ((3, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            for i, (p, q) in enumerate(imgs):
                np.testing.assert_allclose(model.encode_images(p, q)[0], batch[i], atol=BATCH_ABS)
            # The same call twice is the same bits; a batch array equals the list.
            np.testing.assert_array_equal(model.encode_images([p for p, _ in imgs], [q for _, q in imgs]), batch)
            same = tiny_image(2, 2, 0, 3)
            np.testing.assert_array_equal(
                model.encode_images(np.stack([same[0], same[0]]), np.stack([same[1], same[1]])),
                np.stack([batch[2], batch[2]]),
            )
            cut = model.encode_images(imgs[0][0], imgs[0][1], truncate_dim=40)
            self.assertEqual(cut.shape, (1, 40))
            want = batch[0, :40] / np.linalg.norm(batch[0, :40])
            np.testing.assert_allclose(cut[0], want, atol=1e-5)

            clips = [tiny_clip(41, 1), tiny_clip(58, 2, valid=45), tiny_clip(30, 3)]
            a = model.encode_audio([f for f, _ in clips], [m for _, m in clips])
            self.assertEqual((a.shape, a.dtype), ((3, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
            for i, (f, m) in enumerate(clips):
                np.testing.assert_allclose(model.encode_audio(f, m)[0], a[i], atol=BATCH_ABS)
            # Text still works on the same handle.
            self.assertEqual(model.encode([[2, 5, 1]]).shape, (1, TINY_DIM))

    def test_bf16_text_operands_load_and_embed(self):
        with EmbedGemma2.from_pretrained(TINY, multimodal=True, operands="bf16") as model:
            p, q = tiny_image(2, 2, 0, 3)
            with EmbedGemma2.from_pretrained(TINY, multimodal=True) as exact:
                got, want = model.encode_images(p, q), exact.encode_images(p, q)
            self.assertEqual(got.shape, want.shape)
            cos = (got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want))
            self.assertGreater(cos, 0.999)

    def test_the_library_refuses_what_python_cannot_see(self):
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            p, q = tiny_image(2, 2, 0, 3)
            bad = q.copy()
            bad[1] = [999, 0]
            with self.assertRaisesRegex(TesslError, "outside the position table"):
                model.encode_images(p, bad)
            nan = p.copy()
            nan[0, 3] = np.nan
            with self.assertRaisesRegex(TesslError, "pixel value 3"):
                model.encode_images(nan, q)
            with self.assertRaisesRegex(TesslError, "pixel_values_len"):
                model.encode_images(p[:, :40], q)
            with self.assertRaisesRegex(TesslError, "truncate_dim"):
                model.encode_images(p, q, truncate_dim=TINY_DIM + 1)
            f, m = tiny_clip(12, 4)
            with self.assertRaisesRegex(TesslError, "n_soft is zero"):
                model.encode_audio(f, np.zeros(12, dtype=bool))
            with self.assertRaisesRegex(TesslError, "input_features_len"):
                model.encode_audio(f[:, :30], m)
            model.max_tokens = 3
            with self.assertRaisesRegex(TesslError, "limit is 3"):
                model.encode_images(p, q)
            model.max_tokens = 8192
            # Refusals leave the model usable.
            self.assertEqual(model.encode_images(p, q).shape, (1, TINY_DIM))

    def test_a_text_only_load_refuses_images_in_python_and_a_model_is_thread_affine(self):
        with EmbedGemma2.from_pretrained(TINY) as text_only:
            p, q = tiny_image(2, 2, 0, 3)
            with self.assertRaisesRegex(TesslError, "multimodal=True"):
                text_only.encode_images(p, q)
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            seen = []

            def other():
                try:
                    model.encode_images(p, q)
                except TesslError as e:
                    seen.append(str(e))

            t = threading.Thread(target=other)
            t.start()
            t.join()
            self.assertEqual(len(seen), 1)
            self.assertIn("thread other than the one that created it", seen[0])
            self.assertEqual(model.encode_images(p, q).shape, (1, TINY_DIM))


# ---------------------------------------------------------------------------
# The real checkpoint.
# ---------------------------------------------------------------------------

SNAPSHOT = os.environ.get("EMBEDGEMMA2_SNAPSHOT")
REF_DIR = Path(os.environ.get("EMBEDGEMMA2_MM_REF_DIR") or ROOT / "target" / "embedgemma2_mm_ref")
GENERATOR = ROOT / "tools" / "embedgemma2_ref" / "make_mm_reference.py"


def load_generator():
    """The reference script as a module (its ``make_image`` / ``make_audio`` are
    the inputs' definition), or ``None`` when its imports are not installed."""
    spec = importlib.util.spec_from_file_location("make_mm_reference", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GENERATOR.parent))
    try:
        spec.loader.exec_module(module)
    except ImportError:
        return None
    finally:
        sys.path.remove(str(GENERATOR.parent))
    return module


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(
    SNAPSHOT, "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_MM_REF_DIR) to run against the real checkpoint"
)
class RealCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing reference is a failure, not a skip: the opt-in was made.
        cls.images = [
            {k: np.load(REF_DIR / f"image{n}_{k}.npy") for k in ("pixel_values", "image_position_ids", "emb_f32")}
            for n in (0, 1)
        ]
        cls.clips = [
            {k: np.load(REF_DIR / f"audio{n}_{k}.npy") for k in ("input_features", "input_features_mask", "emb_f32")}
            for n in (0, 1)
        ]
        cls.model = EmbedGemma2.from_pretrained(SNAPSHOT, multimodal=True)
        cls.addClassCleanup(cls.model.close)

    def test_the_reference_set_is_what_these_tests_assume(self):
        for im in self.images:
            self.assertEqual(im["pixel_values"].shape, (1, 2520, 768))
            self.assertEqual(im["image_position_ids"].shape, (1, 2520, 2))
        # Different real-patch counts, so a batch is ragged inside its padding.
        real = [int((im["image_position_ids"][0, :, 0] >= 0).sum()) for im in self.images]
        self.assertNotEqual(real[0], real[1])
        for c in self.clips:
            self.assertEqual(c["input_features"].shape[2], 128)
        self.assertNotEqual(self.clips[0]["input_features"].shape[1], self.clips[1]["input_features"].shape[1])

    def test_images_match_sentence_transformers_alone_and_in_one_batch(self):
        want = np.stack([im["emb_f32"] for im in self.images])
        # The processor's tensors as it returned them: [1, P, 768] per input.
        batch = self.model.encode_images(
            [im["pixel_values"] for im in self.images], [im["image_position_ids"] for im in self.images]
        )
        self.assertEqual((batch.shape, batch.dtype), (want.shape, np.float32))
        err, cos = max_abs_and_cosine(batch, want)
        eprint(f"images in one call: max abs {err:.3e}, cosine {cos:.9f}")
        self.assertLessEqual(err, EMB_ABS)
        self.assertGreaterEqual(cos, EMB_COS)
        for i, im in enumerate(self.images):
            alone = self.model.encode_images(im["pixel_values"], im["image_position_ids"])
            err, cos = max_abs_and_cosine(alone[0], want[i])
            eprint(f"image {i} alone: max abs {err:.3e}, cosine {cos:.9f}")
            self.assertLessEqual(err, EMB_ABS, f"image {i} alone")
            self.assertGreaterEqual(cos, EMB_COS, f"image {i} alone")
            drift = np.abs(alone[0] - batch[i]).max()
            eprint(f"image {i}: alone vs in the batch {drift:.3e}")
            self.assertLessEqual(drift, BATCH_ABS, f"image {i}: alone vs in the batch")

    def test_audio_matches_sentence_transformers_alone_and_in_one_batch(self):
        want = np.stack([c["emb_f32"] for c in self.clips])
        # Different frame counts: one list of per-clip arrays, not a stack.
        batch = self.model.encode_audio(
            [c["input_features"] for c in self.clips], [c["input_features_mask"] for c in self.clips]
        )
        self.assertEqual((batch.shape, batch.dtype), (want.shape, np.float32))
        err, cos = max_abs_and_cosine(batch, want)
        eprint(f"clips in one call: max abs {err:.3e}, cosine {cos:.9f}")
        self.assertLessEqual(err, EMB_ABS)
        self.assertGreaterEqual(cos, EMB_COS)
        for i, c in enumerate(self.clips):
            alone = self.model.encode_audio(c["input_features"], c["input_features_mask"])
            err, cos = max_abs_and_cosine(alone[0], want[i])
            eprint(f"clip {i} alone: max abs {err:.3e}, cosine {cos:.9f}")
            self.assertLessEqual(err, EMB_ABS, f"clip {i} alone")
            self.assertGreaterEqual(cos, EMB_COS, f"clip {i} alone")
            drift = np.abs(alone[0] - batch[i]).max()
            eprint(f"clip {i}: alone vs in the batch {drift:.3e}")
            self.assertLessEqual(drift, BATCH_ABS, f"clip {i}: alone vs in the batch")

    def test_matryoshka_prefixes_of_an_image_follow_the_full_embedding(self):
        im = self.images[0]
        full = self.model.encode_images(im["pixel_values"], im["image_position_ids"])[0]
        for d in (512, 256, 128):
            cut = self.model.encode_images(im["pixel_values"], im["image_position_ids"], truncate_dim=d)[0]
            want = full[:d] / np.linalg.norm(full[:d])
            self.assertLessEqual(np.abs(cut - want).max(), 1e-5, f"truncate_dim {d}")

    def test_the_processor_path_hands_the_model_the_saved_tensors(self):
        try:
            from transformers import AutoProcessor
        except ImportError:
            self.skipTest("the transformers package is not installed")
        gen = load_generator()
        if gen is None:
            self.skipTest("the reference script's imports (torch, PIL, sentence-transformers) are not installed")
        processor = AutoProcessor.from_pretrained(SNAPSHOT)
        pics = [gen.make_image(w, h, s) for (w, h, s) in gen.IMAGES]
        out = processor(images=pics, return_tensors="np")
        for i, im in enumerate(self.images):
            d = np.abs(out["pixel_values"][i] - im["pixel_values"][0]).max()
            eprint(f"image {i}: processor pixel_values vs saved: max abs {d:.3e}")
            self.assertLessEqual(d, INPUT_ABS, f"image {i} pixel_values")
            np.testing.assert_array_equal(out["image_position_ids"][i], im["image_position_ids"][0])
        want = np.stack([im["emb_f32"] for im in self.images])
        got = self.model.encode_images(out["pixel_values"], out["image_position_ids"])
        err, cos = max_abs_and_cosine(got, want)
        eprint(f"encode_images on the processor's output: max abs {err:.3e}, cosine {cos:.9f}")
        self.assertLessEqual(err, EMB_ABS)
        self.assertGreaterEqual(cos, EMB_COS)

        wavs = [gen.make_audio(sec, s) for (sec, s) in gen.AUDIOS]
        feats = processor(audio=wavs, return_tensors="np")
        for i, c in enumerate(self.clips):
            n = c["input_features"].shape[1]
            d = np.abs(feats["input_features"][i, :n] - c["input_features"][0]).max()
            eprint(f"clip {i}: processor input_features vs saved: max abs {d:.3e}")
            self.assertLessEqual(d, INPUT_ABS, f"clip {i} input_features")
            np.testing.assert_array_equal(
                feats["input_features_mask"][i, :n].astype(np.int64), c["input_features_mask"][0]
            )
        want = np.stack([c["emb_f32"] for c in self.clips])
        got = self.model.encode_audio(feats["input_features"], feats["input_features_mask"])
        err, cos = max_abs_and_cosine(got, want)
        eprint(f"encode_audio on the processor's output: max abs {err:.3e}, cosine {cos:.9f}")
        # Batched clips are padded to the longest, which the saved (alone) ones
        # are not; the mask makes that no difference to the embedding.
        self.assertLessEqual(err, EMB_ABS)
        self.assertGreaterEqual(cos, EMB_COS)


def eprint(*args):
    print(*args, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
