"""tessl_torch.embedgemma2 against the built library and a device.

Two layers (argument handling against a stand-in library is
``test_embedgemma2_host.py``):

* ``Library``: the built ``libtessl.dylib`` and a device. Refusals need only
  those; the encode checks also need the tiny checkpoint that
  ``cargo test --release --test capi_embedgemma2`` writes under ``target/``;
* ``RealCheckpoint``: the real ``google/embeddinggemma-2`` snapshot against the
  references ``tools/embedgemma2_ref/make_reference.py`` wrote from
  sentence-transformers. Skipped unless ``EMBEDGEMMA2_SNAPSHOT`` is set:

      cargo build --release
      EMBEDGEMMA2_SNAPSHOT=/path/to/snapshot \\
      EMBEDGEMMA2_REF_DIR=/path/to/target/embedgemma2_ref \\
          python3 -m unittest python/tests/test_embedgemma2.py -v

  The bounds are ``tests/embedgemma2_model.rs``'s: max abs ``1e-4`` and cosine
  ``>= 0.99999`` against the fp32 reference, a text alone vs in a batch within
  ``1e-5``. ``encode_texts`` is also held to the reference *ids*, so a wrong
  prompt or a missing ``<bos>``/``<eos>`` is caught before it moves an
  embedding. ``EMBEDGEMMA2_COMPARE_ST=1`` additionally runs sentence-transformers
  itself on the same texts.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402
from tessl_torch import embedgemma2 as eg2  # noqa: E402
from tessl_torch.embedgemma2 import EmbedGemma2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EMB_ABS = 1e-4
EMB_COS = 0.99999
BATCH_ABS = 1e-5


# ---------------------------------------------------------------------------
# The built library on a device.
# ---------------------------------------------------------------------------

LIB_BUILT = tessl_torch.library_path().exists()
TINY = ROOT / "target" / "capi_embedgemma2_tiny"
TINY_DIM, TINY_VOCAB = 96, 48


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
class Library(unittest.TestCase):
    def test_a_bad_snapshot_is_a_tessl_error_and_the_next_load_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            for target, pattern in ((Path(d) / "absent", "config.json"), (Path(d), "config.json")):
                with self.subTest(str(target)), self.assertRaisesRegex(TesslError, pattern):
                    EmbedGemma2(target)
            with self.assertRaisesRegex(TesslError, "snapshot_dir is 0 bytes"):
                EmbedGemma2("")

    @unittest.skipUnless(TINY.exists(), "run `cargo test --release --test capi_embedgemma2` to write the tiny checkpoint")
    def test_the_tiny_checkpoint_embeds_unit_rows_that_do_not_depend_on_the_batch(self):
        with EmbedGemma2(TINY) as model:
            self.assertEqual(model.embedding_dim, TINY_DIM)
            seqs = [[1, 2, 3, 4, 5], [7], [3, 9, 11], [1, 2, 3, 4, 5]]
            batch = model.encode(seqs)
            self.assertEqual((batch.shape, batch.dtype), ((4, TINY_DIM), np.float32))
            np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
            for i, s in enumerate(seqs):
                np.testing.assert_allclose(model.encode([s])[0], batch[i], atol=BATCH_ABS)
            np.testing.assert_allclose(batch[0], batch[3], atol=BATCH_ABS)
            # A prefix is re-normalized over its own columns.
            cut = model.encode(seqs, truncate_dim=64)
            self.assertEqual(cut.shape, (4, 64))
            want = batch[:, :64] / np.linalg.norm(batch[:, :64], axis=1, keepdims=True)
            np.testing.assert_allclose(cut, want, atol=1e-5)
            # The same call twice is the same bits.
            np.testing.assert_array_equal(model.encode(seqs), batch)

    @unittest.skipUnless(TINY.exists(), "run `cargo test --release --test capi_embedgemma2` to write the tiny checkpoint")
    def test_the_library_refuses_what_python_cannot_see(self):
        with EmbedGemma2(TINY) as model:
            with self.assertRaisesRegex(TesslError, f"token id {TINY_VOCAB}"):
                model.encode([[1, TINY_VOCAB]])
            with self.assertRaisesRegex(TesslError, "truncate_dim"):
                model.encode([[1]], truncate_dim=TINY_DIM + 1)
            with self.assertRaisesRegex(TesslError, "n_seqs"):
                model.encode([[1]] * 257)
            model.max_tokens = 4
            self.assertEqual(model.max_tokens, 4)
            with self.assertRaisesRegex(TesslError, "limit is 4"):
                model.encode([[1, 2, 3, 4, 5]])
            with self.assertRaisesRegex(TesslError, "max_tokens"):
                model.max_tokens = 0
            self.assertEqual(model.max_tokens, 4)
            # Refusals leave the model usable.
            self.assertEqual(model.encode([[1, 2, 3, 4]]).shape, (1, TINY_DIM))
            model.max_tokens = 5
            self.assertEqual(model.encode([[1, 2, 3, 4, 5]]).shape, (1, TINY_DIM))

    @unittest.skipUnless(TINY.exists(), "run `cargo test --release --test capi_embedgemma2` to write the tiny checkpoint")
    def test_a_model_used_from_another_thread_is_refused(self):
        with EmbedGemma2(TINY) as model:
            seen = []

            def other():
                try:
                    model.encode([[1, 2]])
                except TesslError as e:
                    seen.append(str(e))

            t = threading.Thread(target=other)
            t.start()
            t.join()
            self.assertEqual(len(seen), 1)
            self.assertIn("thread other than the one that created it", seen[0])
            self.assertEqual(model.encode([[1, 2]]).shape, (1, TINY_DIM))


# ---------------------------------------------------------------------------
# The real checkpoint.
# ---------------------------------------------------------------------------

SNAPSHOT = os.environ.get("EMBEDGEMMA2_SNAPSHOT")
REF_DIR = Path(os.environ.get("EMBEDGEMMA2_REF_DIR") or ROOT / "target" / "embedgemma2_ref")


def max_abs_and_cosine(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
    return np.abs(got - want).max(), cos.min()


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
@unittest.skipUnless(SNAPSHOT, "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_REF_DIR) to run against the real checkpoint")
class RealCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing reference is a failure, not a skip: the opt-in was made.
        cls.ids_flat = np.load(REF_DIR / "ids.npy")
        cls.offsets = np.load(REF_DIR / "offsets.npy")
        cls.ref = np.load(REF_DIR / "emb_f32.npy")
        cls.texts = json.loads((REF_DIR / "texts.json").read_text())["texts"]
        cls.seqs = [cls.ids_flat[cls.offsets[i]:cls.offsets[i + 1]].tolist() for i in range(len(cls.offsets) - 1)]
        cls.model = EmbedGemma2.from_pretrained(SNAPSHOT)
        cls.addClassCleanup(cls.model.close)

    def test_the_reference_set_is_what_these_tests_assume(self):
        self.assertEqual(len(self.seqs), len(self.texts))
        self.assertEqual(self.ref.shape, (len(self.seqs), self.model.embedding_dim))
        self.assertGreater(max(map(len, self.seqs)), 2 * 512, "the set must include a text past twice the window")

    def test_embeddings_match_sentence_transformers_alone_and_in_one_ragged_batch(self):
        batch = self.model.encode(self.seqs)
        self.assertEqual((batch.shape, batch.dtype), (self.ref.shape, np.float32))
        err, cos = max_abs_and_cosine(batch, self.ref)
        self.assertLessEqual(err, EMB_ABS, "batch vs reference")
        self.assertGreaterEqual(cos, EMB_COS, "batch vs reference")
        for i, s in enumerate(self.seqs):
            alone = self.model.encode([s])
            err, cos = max_abs_and_cosine(alone[0], self.ref[i])
            self.assertLessEqual(err, EMB_ABS, f"text {i} alone vs reference")
            self.assertGreaterEqual(cos, EMB_COS, f"text {i} alone vs reference")
            self.assertLessEqual(np.abs(alone[0] - batch[i]).max(), BATCH_ABS, f"text {i} alone vs in the batch")

    def test_matryoshka_prefixes_match_the_reference(self):
        for d in (512, 256, 128):
            with self.subTest(truncate_dim=d):
                got = self.model.encode(self.seqs, truncate_dim=d)
                want = np.load(REF_DIR / f"emb_f32_trunc{d}.npy")
                self.assertEqual(got.shape, want.shape)
                err, cos = max_abs_and_cosine(got, want)
                self.assertLessEqual(err, EMB_ABS)
                self.assertGreaterEqual(cos, EMB_COS)

    def test_the_default_token_limit_is_the_model_cards(self):
        self.assertEqual(self.model.max_tokens, 8192)
        with self.assertRaisesRegex(TesslError, "limit is 8192"):
            self.model.encode([[2] + [1000] * 8192])

    def _prompted_texts(self):
        prompts = self.model.prompts()
        return [prompts[t["prompt_name"]] + t["text"] for t in self.texts]

    def _check_tokenizer(self, tokenizer):
        texts = [t["text"] for t in self.texts]
        # Token ids first: a wrong prompt, <bos>, or <eos> is visible here
        # before it can hide in an embedding that is merely close.
        ids = eg2._token_ids(tokenizer, self._prompted_texts())
        self.assertEqual(ids, self.seqs)
        # One prompt name per text in the reference; encode_texts takes one per
        # call, so group by it and put the rows back in order.
        got = np.empty_like(self.ref)
        for name in sorted({t["prompt_name"] for t in self.texts}):
            rows = [i for i, t in enumerate(self.texts) if t["prompt_name"] == name]
            got[rows] = self.model.encode_texts([texts[i] for i in rows], tokenizer, prompt_name=name)
        err, cos = max_abs_and_cosine(got, self.ref)
        self.assertLessEqual(err, EMB_ABS)
        self.assertGreaterEqual(cos, EMB_COS)

    def test_encode_texts_with_a_tokenizers_tokenizer(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("the tokenizers package is not installed")
        self._check_tokenizer(Tokenizer.from_file(str(Path(SNAPSHOT) / "tokenizer.json")))

    def test_encode_texts_with_a_transformers_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("the transformers package is not installed")
        self._check_tokenizer(AutoTokenizer.from_pretrained(SNAPSHOT))

    @unittest.skipUnless(os.environ.get("EMBEDGEMMA2_COMPARE_ST") == "1", "set EMBEDGEMMA2_COMPARE_ST=1 to run sentence-transformers live")
    def test_encode_texts_matches_sentence_transformers_encode_with_prompt_name(self):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        import torch

        # The reference is the fp32 forward (the checkpoint itself is bf16).
        st = SentenceTransformer(SNAPSHOT, device="cpu", model_kwargs={"attn_implementation": "eager", "dtype": torch.float32})
        tok = AutoTokenizer.from_pretrained(SNAPSHOT)
        for name in ("query", "document"):
            texts = [t["text"] for t in self.texts if t["prompt_name"] == name]
            want = st.encode(texts, prompt_name=name, convert_to_numpy=True, batch_size=1)
            got = self.model.encode_texts(texts, tok, prompt_name=name)
            err, cos = max_abs_and_cosine(got, want)
            self.assertLessEqual(err, EMB_ABS, name)
            self.assertGreaterEqual(cos, EMB_COS, name)


if __name__ == "__main__":
    unittest.main()
