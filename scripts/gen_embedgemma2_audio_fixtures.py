#!/usr/bin/env python3
"""Generate the EmbeddingGemma 2 audio-tower fixtures in tests/fixtures/embedgemma2/ from transformers.

    ~/.venvs/ml-2026-10/bin/python scripts/gen_embedgemma2_audio_fixtures.py

Needs torch, numpy and a transformers that ships `models.gemma4` and
`models.embedding_gemma2` (5.19.0 or later). A tiny `Gemma4AudioModel` with
random weights and finite clip bounds is run on three clips (all frames valid,
trailing frames masked, a hole of masked frames in the middle) under
`attn_implementation="sdpa"`, with the library's own `EmbeddingGemma2Multimodal
Embedder` after it, so `tests/embedgemma2_audio_fixtures.rs` can hold the Rust
f64 reference of `tests/common/embedgemma2_audio.rs` (which the kernels and the
whole-tower test are held to) to transformers on a Mac with no Python
installed. The geometry is the checkpoint's attention (chunk 12, context left
13, no right context) at a small hidden size, with a gradient clip and a logit
cap that bite.

Why sdpa: under eager, `Gemma4AudioAttention.forward` calls
`attention_mask.logical_not()` on the FLOAT mask `create_bidirectional_mask`
returns, which inverts the local attention (see src/embedgemma2_audio.rs). The
clip with trailing masked frames is also run under eager and saved (`..._eager_tower.npy`) so the
test can pin that the two differ.

Files, `eg2a_audio_*`: `config.json` (the config both sides read),
`weights.npy` (every tensor concatenated, f32) with `index.txt` (one line per
tensor: name, then its dims; names as in the checkpoint), and per clip `c{n}_`:
`feats.npy` [F, 16], `mask.npy` [F] (int64), `subsample.npy`, `layer0.npy`,
`tower.npy` [T, 24], `out_mask.npy` [T] (int64), `soft.npy` [kept, 8].
Deterministic: rerunning rewrites identical files.
"""

import importlib.metadata
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "embedgemma2")
TEXT_HIDDEN = 8
AUDIO = dict(
    hidden_size=32, num_hidden_layers=1, num_attention_heads=2, subsampling_conv_channels=[16, 8],
    output_proj_dims=24, use_clipped_linears=True, attention_chunk_size=12, attention_context_left=13,
    attention_context_right=0, conv_kernel_size=5, residual_weight=0.5, attention_logit_cap=5.0,
    attention_invalid_logits_value=-1e9, gradient_clipping=4.0, rms_norm_eps=1e-6, hidden_act="silu",
)
CLIPS = [  # (frames, valid(frame))
    (61, lambda f: True),
    (70, lambda f: f < 52),
    (45, lambda f: not (10 <= f < 19)),
]


def save(name, arr):
    np.save(os.path.join(OUT, f"eg2a_audio_{name}.npy"), np.ascontiguousarray(arr))


def build(impl):
    cfg = Gemma4AudioConfig(**AUDIO)
    cfg._attn_implementation = impl
    return cfg, Gemma4AudioModel(cfg).eval()


def randomize(m, embed):
    g = torch.Generator().manual_seed(2026)
    for n, p in m.named_parameters():
        if "norm" in n:
            p.data = 1.0 + 0.3 * torch.randn(p.shape, generator=g)
        elif "per_dim_scale" in n:
            p.data = 0.6 * torch.randn(p.shape, generator=g)
        elif "bias" in n:
            p.data = 0.2 * torch.randn(p.shape, generator=g)
        elif "depthwise" in n or "conv.weight" in n and p.shape[1] == 1:
            p.data = 0.6 * torch.randn(p.shape, generator=g)
        else:
            fan = p.shape[-1] if p.dim() > 1 else 1
            if "conv.weight" in n:
                fan = p.shape[1] * 9
            p.data = 1.7 * torch.randn(p.shape, generator=g) / np.sqrt(max(fan, 1))
    for n, b in m.named_buffers():
        if n.endswith("_min"):
            b.data = torch.tensor(-(1.3 + 0.5 * torch.rand(1, generator=g).item()))
        elif n.endswith("_max"):
            b.data = torch.tensor(1.3 + 0.5 * torch.rand(1, generator=g).item())
    embed.embedding_projection.weight.data = torch.randn(embed.embedding_projection.weight.shape, generator=g) / 4.0


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg, tower = build("sdpa")
    embed = EmbeddingGemma2MultimodalEmbedder(cfg, SimpleNamespace(hidden_size=TEXT_HIDDEN)).eval()
    randomize(tower, embed)
    _, eager = build("eager")
    eager.load_state_dict(tower.state_dict())

    with open(os.path.join(OUT, "eg2a_audio_config.json"), "w") as f:
        json.dump({"text_config": {"hidden_size": TEXT_HIDDEN},
                   "audio_config": {**AUDIO, "model_type": "gemma4_audio"}}, f, indent=1)
        f.write("\n")

    tensors = [(f"audio_tower.{k}", v) for k, v in tower.state_dict().items()]
    tensors.append(("embed_audio.embedding_projection.weight", embed.embedding_projection.weight))
    flat = np.concatenate([v.detach().double().numpy().astype(np.float32).ravel() for _, v in tensors])
    save("weights", flat)
    with open(os.path.join(OUT, "eg2a_audio_index.txt"), "w") as f:
        for k, v in tensors:
            f.write(" ".join([k] + [str(d) for d in v.shape]) + "\n")

    g = torch.Generator().manual_seed(7)
    for n, (frames, valid) in enumerate(CLIPS):
        feats = 2.0 * (2 * torch.rand(1, frames, 16, generator=g) - 1)
        mask = torch.tensor([[1 if valid(f) else 0 for f in range(frames)]])
        cap = {}
        h1 = tower.subsample_conv_projection.register_forward_hook(lambda _m, _i, out: cap.__setitem__("sub", out[0]))
        h2 = tower.layers[0].register_forward_hook(lambda _m, _i, out: cap.__setitem__("l0", out[0] if isinstance(out, tuple) else out))
        with torch.no_grad():
            out = tower(feats, mask)
            soft = embed(out.last_hidden_state)
            if n == 1:
                e = eager(feats, mask)
        h1.remove()
        h2.remove()
        om = out.attention_mask[0].bool()  # the tower returns its mask as int64
        save(f"c{n}_feats", feats[0].numpy())
        save(f"c{n}_mask", mask[0].numpy().astype(np.int64))
        save(f"c{n}_subsample", cap["sub"][0].numpy())
        save(f"c{n}_layer0", cap["l0"][0].numpy())
        save(f"c{n}_tower", out.last_hidden_state[0].numpy())
        save(f"c{n}_out_mask", om.numpy().astype(np.int64))
        save(f"c{n}_soft", soft[0][om].numpy())
        if n == 1:
            save(f"c{n}_eager_tower", e.last_hidden_state[0].numpy())
        print(f"clip {n}: {frames} frames, {int(mask.sum())} valid, tower rows {om.shape[0]}, kept {int(om.sum())}")

    with open(os.path.join(OUT, "eg2a_audio_provenance.json"), "w") as f:
        json.dump({"versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "transformers")},
                   "generator": "scripts/gen_embedgemma2_audio_fixtures.py"}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
