#!/usr/bin/env python3
"""Reference outputs for tests/embedgemma2_model.rs, from sentence-transformers.

    ~/.venvs/ml/bin/python tools/embedgemma2_ref/make_reference.py [SNAPSHOT_DIR]

SNAPSHOT_DIR is google/embeddinggemma-2 (config.json, model.safetensors,
tokenizer.json, ...); default: the Hugging Face cache. Writes into
target/embedgemma2_ref/ (override with EMBEDGEMMA2_REF_DIR):

  texts.json           the inputs, each with its prompt name
  ids.npy, offsets.npy every text's token ids (the tokenizer's own BOS/EOS
                       included), flattened, and where each starts
  emb_f32.npy          [N, 768] sentence-transformers' embeddings, fp32 model,
                       eager attention: the oracle the f32 forward is held to
  emb_bf16.npy         [N, 768] the same in bf16 (the checkpoint's dtype), so
                       the test can report how far bf16 itself moves
  trace_l{i}.npy       text 0's residual stream after layer i, [T, 512]
  trace_final.npy      text 0 after the final norm, [T, 512]

Needs torch, numpy, transformers >= 5.19 and sentence-transformers >= 6.1.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EMBEDGEMMA2_REF_DIR", os.path.join(HERE, "..", "..", "target", "embedgemma2_ref"))


def snapshot():
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        sys.exit("no google/embeddinggemma-2 snapshot; pass its directory")
    return os.path.dirname(hits[-1])


def long_text(n_words):
    # Deterministic, varied vocabulary so the tokens are not one repeated id.
    words = ("metal kernel attention window encoder residual gradient tensor "
             "embedding retrieval query document cosine latency batch pooling "
             "matryoshka normalize sequence token position rotary frequency").split()
    return " ".join(f"{words[(i * 7) % len(words)]}{i % 13}" for i in range(n_words))


TEXTS = [
    ("query", "How do I speed up attention kernels on Apple silicon?"),
    ("document", "Bidirectional sliding-window attention lets each token see 512 neighbours on both sides."),
    ("query", "x"),
    ("document", "混合精度训练与推理 — 東京で会いましょう 🚀🔥 café naïve résumé é́"),
    ("CodeRetrieval", "fn main() {\n    let v: Vec<u32> = (0..10).collect();\n    println!(\"{:?}\", v);\n}"),
    ("Classification", "   leading and trailing whitespace\t\tand tabs   "),
    ("document", long_text(700)),     # past one window on both sides
    ("document", long_text(2600)),    # several windows; the global layers span it all
]


def main():
    path = snapshot()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    texts = [{"prompt_name": p, "text": t} for p, t in TEXTS]
    with open(os.path.join(OUT, "texts.json"), "w") as f:
        json.dump({"snapshot": path, "texts": texts}, f, ensure_ascii=False, indent=1)

    m32 = SentenceTransformer(path, device="cpu",
                              model_kwargs={"attn_implementation": "eager", "dtype": torch.float32})
    prompts = m32.prompts

    ids, offsets = [], [0]
    for p, t in TEXTS:
        feats = m32.tokenize([prompts[p] + t])
        row = feats["input_ids"][0].tolist()
        ids.extend(row)
        offsets.append(len(ids))
    np.save(os.path.join(OUT, "ids.npy"), np.array(ids, dtype=np.int64))
    np.save(os.path.join(OUT, "offsets.npy"), np.array(offsets, dtype=np.int64))

    def embed(model):
        rows = []
        for p, t in TEXTS:
            e = model.encode([t], prompt_name=p, convert_to_tensor=True, batch_size=1)
            rows.append(e[0].float().cpu())
        return torch.stack(rows).numpy()

    with torch.no_grad():
        np.save(os.path.join(OUT, "emb_f32.npy"), embed(m32).astype(np.float32))

        lm = [mod for mod in m32.modules() if mod.__class__.__name__ == "EmbeddingGemma2TextModel"][0]
        trace = []
        hooks = [layer.register_forward_hook(lambda _m, _i, out: trace.append(
            (out[0] if isinstance(out, tuple) else out)[0].float().cpu().numpy()))
            for layer in lm.layers]
        final = []
        hooks.append(lm.norm.register_forward_hook(lambda _m, _i, out: final.append(out[0].float().cpu().numpy())))
        x = torch.tensor([ids[offsets[0]:offsets[1]]])
        lm(input_ids=x, attention_mask=torch.ones_like(x))
        for h in hooks:
            h.remove()
        assert len(trace) == len(lm.layers) and len(final) == 1, (len(trace), len(final))
        for i, t in enumerate(trace):
            np.save(os.path.join(OUT, f"trace_l{i}.npy"), t.astype(np.float32))
        np.save(os.path.join(OUT, "trace_final.npy"), final[0].astype(np.float32))

    del m32
    m16 = SentenceTransformer(path, device="cpu", model_kwargs={"attn_implementation": "eager"})
    with torch.no_grad():
        np.save(os.path.join(OUT, "emb_bf16.npy"), embed(m16).astype(np.float32))
    print("wrote", OUT, "tokens per text:", [offsets[i + 1] - offsets[i] for i in range(len(TEXTS))])


if __name__ == "__main__":
    main()
