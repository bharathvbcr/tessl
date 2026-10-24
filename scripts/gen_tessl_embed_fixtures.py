#!/usr/bin/env python3
"""Write the Hugging Face oracle for go/cmd/tessl-embed's real-checkpoint tests.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_tessl_embed_fixtures.py [SNAPSHOT_DIR]

SNAPSHOT_DIR defaults to $EMBEDGEMMA2_SNAPSHOT. Needs torch, transformers >= 5.19
and sentence-transformers >= 6.1 (the versions tessl's other EmbeddingGemma 2
references were taken with). Writes go/cmd/tessl-embed/testdata/hf_reference.json
(committed: a reference oracle, never shipped):

  ids        the ids sentence-transformers feeds the model for each text under each
             of the server's prompts (`SentenceTransformer.preprocess([text],
             prompt=...)`, one text at a time), checked against `tokenizers` alone
  embedding  `SentenceTransformer.encode(texts, prompt_name=...)` in float32 on the
             CPU: the 768 values each row holds, as exact float32 values

Deterministic: fixed texts, and the oracle is a function of the snapshot.
"""

import importlib.metadata
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "go", "cmd", "tessl-embed", "testdata", "hf_reference.json")

# Career-knowledge-base-like prose, scripts the byte fallback and the added
# tokens meet, whitespace the normalizer turns into U+2581, and the empty text.
TEXTS = [
    "Built a Metal GEMM kernel that runs EmbeddingGemma 2 on Apple silicon.",
    "how do I speed up attention?",
    "",
    "日本語のテキストと中文文本",
    "हिन्दी पाठ और বাংলা",
    "family 👨‍👩‍👧‍👦, flag 🇮🇳, keycap 1️⃣",
    "é combining marks and ñ",
    "  leading, double  spaces,\ttab\nnewline\r\nCRLF NBSP  ",
    "fn main() { println!(\"{}\", 1 << 3); } // code",
    "<start_of_turn> and <|image> are text here",
]

# The server's names: embeddinggemma-2-tessl (the snapshot default, none),
# :document and :query.
PROMPTS = [None, "document", "query"]


def snapshot():
    if len(sys.argv) > 1:
        return sys.argv[1]
    snap = os.environ.get("EMBEDGEMMA2_SNAPSHOT")
    if not snap:
        sys.exit("pass SNAPSHOT_DIR or set EMBEDGEMMA2_SNAPSHOT")
    return snap


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

    snap = os.path.realpath(snapshot())
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    model = SentenceTransformer(snap, device="cpu", model_kwargs={"dtype": torch.float32})
    assert model.default_prompt_name is None, model.default_prompt_name
    tf = model[0]
    prompts = dict(model.prompts)

    cases = []
    for name in PROMPTS:
        prompt = prompts[name] if name else None
        emb = model.encode(TEXTS, prompt_name=name, convert_to_numpy=True, batch_size=len(TEXTS))
        assert emb.dtype == np.float32 and emb.shape == (len(TEXTS), 768), (emb.dtype, emb.shape)
        for i, text in enumerate(TEXTS):
            feats = tf.preprocess([text], prompt=prompt)
            mask = feats["attention_mask"][0]
            ids = feats["input_ids"][0][: int(mask.sum())].tolist()
            lib = tok.encode((prompt or "") + text, add_special_tokens=True).ids
            assert ids == lib, f"preprocess and tokenizers disagree on {text!r}"
            cases.append({
                "prompt": name or "",
                "text": text,
                "ids": ids,
                # The shortest decimal that reads back as the same float32.
                "embedding": [float(str(x)) for x in emb[i]],
            })

    out = {
        "snapshot_revision": os.path.basename(snap),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "sentence-transformers", "tokenizers")},
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(cases)} cases to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
