#!/usr/bin/env python3
"""Generate the BertSparseModel golden fixtures in tests/fixtures/bert_tiny/ from transformers.

    ~/.venvs/rsijev-oracle/bin/python scripts/gen_bert_fixtures.py

Python is the reference oracle only: this script builds a tiny, seeded, random
`BertForMaskedLM` and `DistilBertForMaskedLM` (fp32, eager attention), saves
each as `config.json` + `model.safetensors` exactly as `save_pretrained` writes
it (decoder tied to the word embeddings, so no decoder tensor), runs
transformers on every sequence on its own with no padding, and records what
`tessl::bert::BertSparseModel::encode` must reproduce:

  <family>/ids.npy       int64 [sum(lens)]  token ids, sequences concatenated
  <family>/lens.npy      int64 [batch]      each sequence's length
  <family>/pooled.npy    f32 [batch, vocab] max_t log1p(relu(MLM logits[t]))
  <family>/trace.npy     f32 [layers + 1, len0, hidden]  for sequence 0: the
                         embedding LayerNorm output, then each layer's output
                         (transformers' `hidden_states`)

`manifest.json` records the torch, transformers and numpy versions, the seed,
the generator's sha256 and every output file's sha256. Deterministic for a
given set of versions: rerunning rewrites identical files.

The weights are re-drawn (normal, std 0.2; LayerNorm weights 1 + 0.1 * normal;
biases 0.1 * normal) because transformers' 0.02 init leaves the logits ~0 and
the goldens would not distinguish a wrong layer from a right one.
"""

import hashlib
import importlib.metadata
import json
import os
import platform

import numpy as np
import torch
from safetensors import safe_open
from transformers import BertConfig, BertForMaskedLM, DistilBertConfig, DistilBertForMaskedLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures", "bert_tiny")
SEED = 20260713
HIDDEN, HEADS, LAYERS, INTER, VOCAB, POSITIONS = 64, 2, 2, 96, 96, 16  # head dim 32
LENS = [16, 3, 9, 12, 5, 16]  # the first and last are full length; the rest pad in a batch


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(family):
    if family == "bert":
        cfg = BertConfig(
            hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
            intermediate_size=INTER, vocab_size=VOCAB, max_position_embeddings=POSITIONS,
            type_vocab_size=2, hidden_act="gelu", layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )
        cfg._attn_implementation = "eager"
        return BertForMaskedLM(cfg)
    cfg = DistilBertConfig(
        dim=HIDDEN, n_layers=LAYERS, n_heads=HEADS, hidden_dim=INTER, vocab_size=VOCAB,
        max_position_embeddings=POSITIONS, activation="gelu", dropout=0.0, attention_dropout=0.0,
        sinusoidal_pos_embds=False,
    )
    cfg._attn_implementation = "eager"
    return DistilBertForMaskedLM(cfg)


def randomize(model, gen):
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            noise = torch.randn(p.shape, generator=gen, dtype=torch.float32)
            if "LayerNorm" in name or "layer_norm" in name:
                p.copy_(noise * 0.1 + (1.0 if name.endswith("weight") else 0.0))
            elif name.endswith("bias"):
                p.copy_(noise * 0.1)
            else:
                p.copy_(noise * 0.2)
    model.tie_weights()


def sparse(logits):
    """What BertSparseModel pools: max over tokens of log1p(relu(logits))."""
    return torch.log1p(torch.relu(logits)).max(dim=0).values


def family_fixture(family, seed):
    gen = torch.Generator().manual_seed(seed)
    model = build(family).eval()
    randomize(model, gen)
    d = os.path.join(OUT, family)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    gen_cfg = os.path.join(d, "generation_config.json")
    if os.path.exists(gen_cfg):
        os.remove(gen_cfg)
    with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
        names = sorted(f.keys())
    assert not any("decoder.weight" in n or "vocab_projector.weight" in n for n in names), names

    seqs = [torch.randint(0, VOCAB, (n,), generator=gen) for n in LENS]
    pooled, trace = [], None
    with torch.no_grad():
        for i, ids in enumerate(seqs):
            out = model(input_ids=ids[None], output_hidden_states=True)
            pooled.append(sparse(out.logits[0]))
            if i == 0:
                assert len(out.hidden_states) == LAYERS + 1
                trace = torch.stack([h[0] for h in out.hidden_states])
    pooled = torch.stack(pooled)
    # A golden near zero would pass any broken model: require real signal.
    assert (pooled > 0.5).float().mean() > 0.2 and pooled.max() > 1.5, "goldens carry too little signal"
    np.save(os.path.join(d, "ids.npy"), torch.cat(seqs).numpy().astype(np.int64))
    np.save(os.path.join(d, "lens.npy"), np.asarray(LENS, dtype=np.int64))
    np.save(os.path.join(d, "pooled.npy"), np.ascontiguousarray(pooled.numpy().astype(np.float32)))
    np.save(os.path.join(d, "trace.npy"), np.ascontiguousarray(trace.numpy().astype(np.float32)))
    return d


def write_manifest(dirs):
    files = {}
    for d in dirs:
        for fn in sorted(os.listdir(d)):
            files[f"{os.path.basename(d)}/{fn}"] = sha256(os.path.join(d, fn))
    info = {
        "generator": "scripts/gen_bert_fixtures.py",
        "generator_sha256": sha256(os.path.abspath(__file__)),
        "seed": SEED,
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "safetensors")},
        "shape": {"hidden": HIDDEN, "heads": HEADS, "layers": LAYERS, "intermediate": INTER,
                  "vocab": VOCAB, "positions": POSITIONS, "lens": LENS},
        "files_sha256": files,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(info, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    dirs = [family_fixture("bert", SEED), family_fixture("distilbert", SEED + 1)]
    write_manifest(dirs)
    print("wrote", sorted(os.listdir(OUT)))
