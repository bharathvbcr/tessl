#!/usr/bin/env python3
"""Audio-tower references for EmbeddingGemma 2 with the intended attention mask.

    ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_audio_sdpa_reference.py [SNAPSHOT_DIR]

Why this exists. `make_mm_reference.py` runs sentence-transformers with
`attn_implementation="eager"`. In transformers 5.19, `Gemma4AudioModel` builds
its chunked local-attention mask with `create_bidirectional_mask`, which under
eager returns a FLOAT additive mask (0.0 where a key is visible, `finfo.min`
where it is not), and `Gemma4AudioAttention.forward` then calls
`attention_mask.logical_not()` on it, assuming a bool. On a float mask that
flips the meaning: the visible keys are filled with `attention_invalid_logits_value`
and the keys the window forbids are the ones attended to. Under sdpa the mask
is bool and the same code is right. The model never calls
`scaled_dot_product_attention` (the attention is hand-written matmuls), so
`_attn_implementation` selects only the mask's dtype; sdpa is the intended
behaviour and the identical arithmetic otherwise.

So the audio goldens of `make_mm_reference.py` (subsample.npy aside, which does
not touch attention) describe the inverted mask. This script writes the same
audio files with the mask the model is specified to apply, from the SAME
`audio{n}_input_features.npy` / `_input_features_mask.npy` the eager run saved
(read from EMBEDGEMMA2_MM_REF_DIR, default target/embedgemma2_mm_ref), in
float32 on the CPU, into EMBEDGEMMA2_MM_REF_SDPA_DIR (default
target/embedgemma2_mm_ref_sdpa). Files, for n = 0, 1 (prefix `audio{n}_`):

  subsample.npy [T, 1024], layer{0..11}.npy [T, 1024], tower_out.npy [T, 1536],
  out_mask.npy [T] bool, soft.npy [S, 512] (valid rows), plus a copy of
  input_features.npy and input_features_mask.npy so the directory is
  self-contained for tests/embedgemma2_audio_model.rs.

The soft tokens are `EmbeddingGemma2MultimodalEmbedder.forward`: a weightless
RMSNorm over 1536 at the audio eps, then `embed_audio.embedding_projection`.
"""

import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

HF = os.path.expanduser("~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("EMBEDGEMMA2_MM_REF_DIR", os.path.join(HERE, "..", "..", "target", "embedgemma2_mm_ref"))
OUT = os.environ.get(
    "EMBEDGEMMA2_MM_REF_SDPA_DIR", os.path.join(HERE, "..", "..", "target", "embedgemma2_mm_ref_sdpa")
)
CLIPS = (0, 1)


def snapshot():
    if len(sys.argv) > 1:
        return sys.argv[1]
    snaps = sorted(os.listdir(HF))
    if not snaps:
        raise SystemExit(f"no snapshot under {HF}")
    return os.path.join(HF, snaps[-1])


def main():
    snap = snapshot()
    with open(os.path.join(snap, "config.json")) as f:
        acfg = json.load(f)["audio_config"]
    skip = ("architectures", "id2label", "label2id", "model_type", "dtype")
    cfg = Gemma4AudioConfig(**{k: v for k, v in acfg.items() if k not in skip})
    # The mask dtype is the whole point: bool, as the code expects.
    cfg._attn_implementation = "sdpa"
    tower = Gemma4AudioModel(cfg).eval()

    tower_sd, embed_w = {}, None
    with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("audio_tower."):
                tower_sd[k[len("audio_tower."):]] = f.get_tensor(k).float()
            elif k == "embed_audio.embedding_projection.weight":
                embed_w = f.get_tensor(k).float()
    result = tower.load_state_dict(tower_sd, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    assert embed_w is not None
    eps = cfg.rms_norm_eps

    os.makedirs(OUT, exist_ok=True)
    for n in CLIPS:
        prefix = f"audio{n}_"
        feats = torch.tensor(np.load(os.path.join(SRC, prefix + "input_features.npy")))
        mask = torch.tensor(np.load(os.path.join(SRC, prefix + "input_features_mask.npy")))
        cap = {"layers": []}
        hooks = [
            tower.subsample_conv_projection.register_forward_hook(
                lambda _m, _i, out: cap.__setitem__("subsample", out[0])),
        ]
        hooks += [
            layer.register_forward_hook(lambda _m, _i, out: cap["layers"].append(out[0] if isinstance(out, tuple) else out))
            for layer in tower.layers
        ]
        with torch.no_grad():
            out = tower(feats, mask)
        for h in hooks:
            h.remove()
        assert len(cap["layers"]) == len(tower.layers)
        out_mask = out.attention_mask[0].bool()
        tower_out = out.last_hidden_state[0]
        normed = tower_out * torch.pow(tower_out.pow(2).mean(-1, keepdim=True) + eps, -0.5)
        soft = F.linear(normed, embed_w)[out_mask]

        def save(name, arr):
            np.save(os.path.join(OUT, prefix + name + ".npy"), arr)

        save("subsample", cap["subsample"][0].numpy().astype(np.float32))
        for i, t in enumerate(cap["layers"]):
            save(f"layer{i}", t[0].numpy().astype(np.float32))
        save("tower_out", tower_out.numpy().astype(np.float32))
        save("out_mask", out_mask.numpy())
        save("soft", soft.numpy().astype(np.float32))
        for name in ("input_features", "input_features_mask"):
            shutil.copyfile(os.path.join(SRC, prefix + name + ".npy"), os.path.join(OUT, prefix + name + ".npy"))
        print(prefix, "frames", tuple(feats.shape), "valid", int(mask.sum()), "soft", tuple(soft.shape))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
