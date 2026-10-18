"""tessl_torch.embedgemma2's Rust front ends (texts, raw images, raw audio, video,
media) against the built library and a device, with no ``transformers``,
``tokenizers`` or processor anywhere.

Two layers (argument handling against a stand-in library is
``test_embedgemma2_host.py`` and ``test_embedgemma2_mm_host.py``):

* ``Library``: the built ``libtessl.dylib`` and a device, on the tiny multimodal
  snapshot (checkpoint, tokenizer, prompts and processor config) that
  ``cargo test --release --test capi_embedgemma2_raw`` writes under ``target/``:
  rows that are unit length and do not depend on the batch, what the library
  refuses that Python cannot see, and calls that agree with the id and tensor
  entries they are built from;
* ``RealCheckpoint``: the real ``google/embeddinggemma-2`` snapshot against the
  references ``tools/embedgemma2_ref/make_reference.py`` (texts) and
  ``make_mm_reference.py`` (images, audio, video and timestamped video) wrote
  from sentence-transformers (fp32, CPU). Skipped unless ``EMBEDGEMMA2_SNAPSHOT``
  is set:

      cargo build --release
      EMBEDGEMMA2_SNAPSHOT=/path/to/snapshot \\
      EMBEDGEMMA2_REF_DIR=/path/to/target/embedgemma2_ref \\
      EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \\
          python3 -m unittest python/tests/test_embedgemma2_raw.py -v

  The bounds are ``tests/embedgemma2_e2e_model.rs``'s, fixed before the first
  run: max abs ``1e-4`` and cosine ``>= 0.99999``. Texts are also held to the
  reference *ids*. The last test runs every kind in a fresh interpreter and
  asserts that neither ``transformers`` nor ``tokenizers`` was imported by the
  calls.
"""

import json
import os
import subprocess
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

LIB_BUILT = tessl_torch.library_path().exists()
TINY = ROOT / "target" / "capi_embedgemma2_raw_text"
TINY_DIM = 96


def max_abs_and_cosine(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return np.abs(got - want).max(), cos.min()


def eprint(*args):
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------
# The built library on a device, with the tiny snapshot.
# ---------------------------------------------------------------------------

def rgb(h, w, c=3, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, c), dtype=np.uint8)


def tone(n, seed=0):
    rng = np.random.default_rng(seed)
    return (0.4 * np.sin(np.arange(n) * 0.05) + 0.1 * rng.standard_normal(n)).astype(np.float32)


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(
    TINY.exists(), "run `cargo test --release --test capi_embedgemma2_raw` to write the tiny snapshot"
)
class Library(unittest.TestCase):
    def test_texts_are_tokenized_in_rust_and_embed_to_unit_rows_that_do_not_depend_on_the_batch(self):
        texts = ["1: 2:3", "", "00 0 00 1000", "héllo 中文 \U0001f389"]
        with EmbedGemma2.from_pretrained(TINY) as model:
            self.assertEqual(model.prompts(), {"query": "00 ", "document": "1: "})
            self.assertIsNone(model.default_prompt_name)
            ids = model.tokenize(texts)
            self.assertEqual([r[0] for r in ids], [2] * 4, "every text starts with <bos>")
            self.assertEqual([r[-1] for r in ids], [1] * 4, "and ends with <eos>")
            self.assertEqual(ids[1], [2, 1], "an empty text is <bos> <eos>")
            self.assertEqual(model.tokenize(["1: 2:3"], prompt_name="document")[0][1:4], [5, 14, 15], "the prompt \"1: \" comes first")
            batch = model.encode_texts(texts)
            self.assertEqual((batch.shape, batch.dtype), ((4, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            np.testing.assert_array_equal(batch, model.encode(ids))
            for i, t in enumerate(texts):
                np.testing.assert_allclose(model.encode_texts([t])[0], batch[i], atol=BATCH_ABS)
            prompted = model.encode_texts(texts, prompt_name="query")
            self.assertGreater(np.abs(prompted - batch).max(), 1e-3, "the prompt changed nothing")
            np.testing.assert_array_equal(
                prompted, model.encode(model.tokenize(texts, prompt_name="query"))
            )
            cut = model.encode_texts(texts, truncate_dim=40)
            want = batch[:, :40] / np.linalg.norm(batch[:, :40], axis=1, keepdims=True)
            np.testing.assert_allclose(cut, want, atol=1e-5)
            self.assertEqual(model.decode([4, 5, 6]), "012")
            self.assertEqual(model.decode([2, 4, 1]), "0")
            self.assertEqual(model.decode([2, 4, 1], skip_special_tokens=False), "<bos>0<eos>")
            self.assertEqual(model.decode([]), "")

    def test_the_library_refuses_what_python_cannot_see(self):
        with EmbedGemma2.from_pretrained(TINY) as model:
            with self.assertRaisesRegex(TesslError, r'no prompt named "nope".*query'):
                model.encode_texts(["1"], prompt_name="nope")
            with self.assertRaisesRegex(TesslError, "placeholder"):
                model.encode_texts(["a <|image|> b"])
            model.max_tokens = 6
            with self.assertRaisesRegex(TesslError, "text 1 has .* the limit is 6"):
                model.encode_texts(["1", "0123456789 0123456789"])
            model.max_tokens = 8192
            with self.assertRaisesRegex(TesslError, "truncate_dim"):
                model.encode_texts(["1"], truncate_dim=TINY_DIM + 1)
            with self.assertRaisesRegex(TesslError, "n_texts is 257"):
                model.encode_texts(["1"] * 257)
            self.assertEqual(model.encode_texts(["1"]).shape, (1, TINY_DIM), "usable after the refusals")

    def test_raw_images_embed_to_unit_rows_that_do_not_depend_on_the_batch(self):
        gray, rgba = rgb(40, 40, 1, 1), rgb(100, 50, 4, 2)
        images = [rgb(64, 56, 3, 3), gray[:, :, 0], rgba, rgb(13, 77, 3, 4)]
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            batch = model.embed_images(images)
            self.assertEqual((batch.shape, batch.dtype), ((4, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            for i, im in enumerate(images):
                np.testing.assert_allclose(model.embed_images([im])[0], batch[i], atol=BATCH_ABS)
            np.testing.assert_array_equal(model.embed_images(images), batch)
            # A gray image is a replicated RGB one; alpha is dropped.
            np.testing.assert_array_equal(model.embed_images(gray), model.embed_images(np.repeat(gray, 3, axis=2)))
            np.testing.assert_array_equal(model.embed_images(rgba), model.embed_images(rgba[:, :, :3]))
            np.testing.assert_array_equal(model.embed_images(np.stack([images[0], images[0]]))[0], batch[0])
            with self.assertRaisesRegex(TesslError, "5 channels"):
                model.embed_images(rgb(8, 8, 5))
            with self.assertRaisesRegex(TesslError, "truncate_dim"):
                model.embed_images(images[0], truncate_dim=TINY_DIM + 1)
            self.assertEqual(model.embed_images(images[0], truncate_dim=40).shape, (1, 40))

    def test_raw_audio_embeds_to_unit_rows_and_refuses_another_rate(self):
        clips = [tone(1600, 1), tone(4000, 2), tone(16000, 3)]
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            batch = model.embed_audio(clips)
            self.assertEqual((batch.shape, batch.dtype), ((3, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            for i, c in enumerate(clips):
                np.testing.assert_allclose(model.embed_audio(c)[0], batch[i], atol=BATCH_ABS)
            np.testing.assert_array_equal(model.embed_audio([c.astype(np.float64) for c in clips]), batch)
            with self.assertRaisesRegex(TesslError, "does not resample"):
                model.embed_audio(clips, sampling_rate=8000)
            nan = clips[0].copy()
            nan[5] = np.nan
            with self.assertRaisesRegex(TesslError, "sample 5"):
                model.embed_audio([nan])
            with self.assertRaisesRegex(TesslError, "truncate_dim"):
                model.embed_audio(clips, truncate_dim=TINY_DIM + 1)

    def test_video_embeds_with_and_without_timestamps(self):
        rng = np.random.default_rng(5)
        video = rng.integers(0, 256, size=(4, 24, 32, 3), dtype=np.uint8)
        other = rng.integers(0, 256, size=(8, 40, 16, 3), dtype=np.uint8)
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            plain = model.embed_video(video, 2.0)
            self.assertEqual((plain.shape, plain.dtype), ((1, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(plain, axis=1), 1.0, atol=1e-5)
            np.testing.assert_array_equal(model.embed_video(list(video), 2.0), plain)
            stamped = model.embed_video(video, 2.0, timestamps=True)
            self.assertGreater(np.abs(stamped - plain).max(), 1e-4, "timestamps changed nothing")
            both = model.embed_videos([video, other], [2.0, 4.0])
            np.testing.assert_allclose(both[0], plain[0], atol=BATCH_ABS)
            np.testing.assert_allclose(model.embed_video(other, 4.0)[0], both[1], atol=BATCH_ABS)
            with self.assertRaisesRegex(TesslError, "fps"):
                model.embed_video(video, 0.0)
            with self.assertRaisesRegex(TesslError, "timestamps"):
                model.embed_video(video, 2.0, timestamps=2)

    def test_media_with_ids_of_your_own_is_the_id_entry_when_it_holds_no_media(self):
        with EmbedGemma2.from_pretrained(TINY, multimodal=True) as model:
            ids = [[2, 5, 6, 1], [2, 9, 1]]
            np.testing.assert_array_equal(model.media([{"ids": i} for i in ids]), model.encode(ids))
            with self.assertRaisesRegex(TesslError, "image placeholder ids"):
                model.media([{"ids": [2, 5, 1], "images": [rgb(16, 16)]}])
            self.assertEqual(model.media([{"ids": [2, 5, 1]}]).shape, (1, TINY_DIM), "usable after the refusal")

    def test_a_text_only_load_refuses_media_in_python_and_a_model_is_thread_affine(self):
        with EmbedGemma2.from_pretrained(TINY) as text_only:
            with self.assertRaisesRegex(TesslError, "multimodal=True"):
                text_only.embed_images([rgb(8, 8)])
            self.assertEqual(text_only.encode_texts(["1"]).shape, (1, TINY_DIM))
        with EmbedGemma2.from_pretrained(TINY) as model:
            seen = []

            def other():
                for call in (lambda: model.encode_texts(["1"]), lambda: model.tokenize(["1"]), model.prompts):
                    try:
                        call()
                    except TesslError as e:
                        seen.append(str(e))

            t = threading.Thread(target=other)
            t.start()
            t.join()
            self.assertEqual(len(seen), 3)
            self.assertTrue(all("thread other than the one that created it" in s for s in seen), seen)
            self.assertEqual(model.encode_texts(["1"]).shape, (1, TINY_DIM))


# ---------------------------------------------------------------------------
# The real checkpoint.
# ---------------------------------------------------------------------------

SNAPSHOT = os.environ.get("EMBEDGEMMA2_SNAPSHOT")
TEXT_REF = Path(os.environ.get("EMBEDGEMMA2_REF_DIR") or ROOT / "target" / "embedgemma2_ref")
MM_REF = Path(os.environ.get("EMBEDGEMMA2_MM_REF_DIR") or ROOT / "target" / "embedgemma2_mm_ref")

# The child of the fresh-interpreter test: every kind through the raw entries,
# the metrics as JSON on stdout, the module check last.
CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from tessl_torch.embedgemma2 import EmbedGemma2

snapshot, text_ref, mm_ref = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])

def metrics(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return [float(np.abs(got - want).max()), float(cos.min())]

out = {}
model = EmbedGemma2.from_pretrained(snapshot, multimodal=True)
texts = json.loads((text_ref / "texts.json").read_text())["texts"]
want = np.load(text_ref / "emb_f32.npy")
ids_flat, offs = np.load(text_ref / "ids.npy"), np.load(text_ref / "offsets.npy")
want_ids = [ids_flat[offs[i]:offs[i + 1]].tolist() for i in range(len(offs) - 1)]
got = np.empty_like(want)
ids_equal = True
for name in sorted({t["prompt_name"] for t in texts}):
    rows = [i for i, t in enumerate(texts) if t["prompt_name"] == name]
    batch = [texts[i]["text"] for i in rows]
    ids_equal &= model.tokenize(batch, prompt_name=name) == [want_ids[i] for i in rows]
    got[rows] = model.encode_texts(batch, prompt_name=name)
out["text_ids_equal"] = bool(ids_equal)
out["text"] = metrics(got, want)
images = [np.load(mm_ref / f"image{n}_raw.npy").astype(np.uint8) for n in (0, 1)]
out["images"] = metrics(model.embed_images(images), np.stack([np.load(mm_ref / f"image{n}_emb_f32.npy") for n in (0, 1)]))
clips = [np.load(mm_ref / f"audio{n}_raw.npy") for n in (0, 1)]
out["audio"] = metrics(model.embed_audio(clips), np.stack([np.load(mm_ref / f"audio{n}_emb_f32.npy") for n in (0, 1)]))
for n in (0, 1):
    frames = np.load(mm_ref / f"video{n}_raw.npy").astype(np.uint8)
    fps = float(np.load(mm_ref / f"video{n}_fps.npy"))
    out[f"video{n}"] = metrics(model.embed_video(frames, fps), np.load(mm_ref / f"video{n}_emb_f32.npy")[None])
frames = np.load(mm_ref / "video0_raw.npy").astype(np.uint8)
fps = float(np.load(mm_ref / "video0_fps.npy"))
out["video0t"] = metrics(model.embed_video(frames, fps, timestamps=True), np.load(mm_ref / "video0t_emb_f32.npy")[None])
model.close()
out["imported"] = sorted(m for m in ("transformers", "tokenizers", "sentence_transformers", "PIL") if m in sys.modules)
print(json.dumps(out))
"""


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(
    SNAPSHOT, "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_REF_DIR, EMBEDGEMMA2_MM_REF_DIR) to run against the real checkpoint"
)
class RealCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing reference is a failure, not a skip: the opt-in was made.
        cls.texts = json.loads((TEXT_REF / "texts.json").read_text())["texts"]
        cls.text_ref = np.load(TEXT_REF / "emb_f32.npy")
        flat, offs = np.load(TEXT_REF / "ids.npy"), np.load(TEXT_REF / "offsets.npy")
        cls.ids = [flat[offs[i] : offs[i + 1]].tolist() for i in range(len(offs) - 1)]
        cls.model = EmbedGemma2.from_pretrained(SNAPSHOT, multimodal=True)
        cls.addClassCleanup(cls.model.close)

    def check(self, what, got, want):
        err, cos = max_abs_and_cosine(got, want)
        eprint(f"{what}: max abs {err:.3e}, cosine {cos:.9f}")
        self.assertLessEqual(err, EMB_ABS, what)
        self.assertGreaterEqual(cos, EMB_COS, what)

    def by_prompt(self):
        for name in sorted({t["prompt_name"] for t in self.texts}):
            rows = [i for i, t in enumerate(self.texts) if t["prompt_name"] == name]
            yield name, rows, [self.texts[i]["text"] for i in rows]

    def test_the_rust_tokenizer_makes_the_references_ids_and_the_embeddings_match(self):
        got = np.empty_like(self.text_ref)
        for name, rows, batch in self.by_prompt():
            # Token ids first: a wrong prompt, <bos> or <eos> is visible here
            # before it can hide in an embedding that is merely close.
            self.assertEqual(self.model.tokenize(batch, prompt_name=name), [self.ids[i] for i in rows], name)
            got[rows] = self.model.encode_texts(batch, prompt_name=name)
        self.check("texts through the Rust tokenizer", got, self.text_ref)
        # A text alone is its row of the batch.
        i = len(self.texts) // 2
        alone = self.model.encode_texts([self.texts[i]["text"]], prompt_name=self.texts[i]["prompt_name"])[0]
        self.assertLessEqual(np.abs(alone - got[i]).max(), BATCH_ABS)

    def test_decode_inverts_the_references_ids(self):
        for i in (0, 3):
            self.assertEqual(self.model.decode(self.ids[i]), self.model.decode(self.model.tokenize(
                [self.texts[i]["text"]], prompt_name=self.texts[i]["prompt_name"])[0]))
        prompts = self.model.prompts()
        self.assertEqual(set(prompts), {"query", "document"} | (set(prompts) - {"query", "document"}))
        self.assertIsNone(self.model.default_prompt_name)

    def test_raw_images_match_sentence_transformers(self):
        images = [np.load(MM_REF / f"image{n}_raw.npy").astype(np.uint8) for n in (0, 1)]
        want = np.stack([np.load(MM_REF / f"image{n}_emb_f32.npy") for n in (0, 1)])
        self.assertEqual({im.shape[2] for im in images}, {3})
        self.check("raw images in one call", self.model.embed_images(images), want)
        for i, im in enumerate(images):
            alone = self.model.embed_images([im])
            self.check(f"raw image {i} alone", alone[0], want[i])

    def test_raw_audio_matches_sentence_transformers(self):
        clips = [np.load(MM_REF / f"audio{n}_raw.npy") for n in (0, 1)]
        want = np.stack([np.load(MM_REF / f"audio{n}_emb_f32.npy") for n in (0, 1)])
        self.assertEqual({c.dtype for c in clips}, {np.dtype(np.float32)})
        self.check("raw audio in one call", self.model.embed_audio(clips), want)
        for i, c in enumerate(clips):
            self.check(f"raw clip {i} alone", self.model.embed_audio([c])[0], want[i])

    def test_raw_video_matches_sentence_transformers_with_and_without_timestamps(self):
        for n in (0, 1):
            frames = np.load(MM_REF / f"video{n}_raw.npy").astype(np.uint8)
            fps = float(np.load(MM_REF / f"video{n}_fps.npy"))
            want = np.load(MM_REF / f"video{n}_emb_f32.npy")
            self.check(f"raw video {n}", self.model.embed_video(frames, fps)[0], want)
        frames = np.load(MM_REF / "video0_raw.npy").astype(np.uint8)
        fps = float(np.load(MM_REF / "video0_fps.npy"))
        want = np.load(MM_REF / "video0t_emb_f32.npy")
        self.check("raw video 0 with timestamps", self.model.embed_video(frames, fps, timestamps=True)[0], want)

    def test_a_fresh_interpreter_embeds_every_kind_without_transformers(self):
        done = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, str(ROOT / "python"), SNAPSHOT, str(TEXT_REF), str(MM_REF)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        out = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(out.pop("imported"), [], "the calls imported a front end's library")
        self.assertTrue(out.pop("text_ids_equal"), "the Rust tokenizer's ids differ from the reference's")
        for kind, (err, cos) in out.items():
            eprint(f"fresh interpreter, {kind}: max abs {err:.3e}, cosine {cos:.9f}")
            self.assertLessEqual(err, EMB_ABS, kind)
            self.assertGreaterEqual(cos, EMB_COS, kind)
        self.assertEqual(sorted(out), ["audio", "images", "text", "video0", "video0t", "video1"])


if __name__ == "__main__":
    unittest.main()
