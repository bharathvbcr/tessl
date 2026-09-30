"""Reference outputs of Qwen3.5-2B-Base from transformers, for tessl's
full-model parity test (tests/qwen35_model.rs).

    python3 tools/qwen35_ref/make_reference.py            # writes target/qwen35_ref/

Loads the text tower (`Qwen3_5ForCausalLM`) from the checkpoint in the Hugging
Face cache, runs one fixed prompt, and writes:

  ids.npy                 int64 [T]            the prompt's token ids
  logits_f32.npy          float32 [T, vocab]   fp32 forward (weights widened from bf16)
  hidden_f32_{l}.npy      float32 [T, hidden]  the residual stream after layer l
                                               (l = 0..23), fp32 forward
  final_norm_f32.npy      float32 [T, hidden]  after the final norm, fp32
  logits_bf16.npy         float32 [T, vocab]   the same forward in bf16, as the
                                               scale of bf16 rounding noise
  meta.txt                what was run

Every checkpoint tensor the text tower needs must load: a missing or
unexpected key aborts, since a model with a freshly initialised layer still
runs and produces plausible-looking logits.

Needs ~10 GB free RAM (fp32 weights plus the bf16 copy, one at a time).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

# Long enough for three 64-token GDN chunks, so state crosses chunk edges.
PROMPT = (
    "The gated delta rule updates a fast-weight memory with a learned decay "
    "and a delta correction. Each token writes its value under its key, after "
    "first erasing whatever the memory already returned for that key, and the "
    "whole memory fades by a per-head factor between tokens. Linear attention "
    "layers of this kind are cheap at long context because their state has a "
    "fixed size, while full attention layers keep every past key and value. "
    "Hybrid models interleave the two so that most layers are cheap and a few "
    "can still look up exact tokens from far back in the context. In a model "
    "with twenty-four layers where every fourth one is full attention, "
    "eighteen layers are linear and six are full. Question: which layer types "
    "does Qwen3.5 mix, and in what ratio? Answer:"
)
MIN_TOKENS = 130


def find_snapshot():
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/*/")
    snaps = sorted(glob.glob(pat))
    if len(snaps) != 1:
        sys.exit(f"expected one snapshot at {pat}, found {snaps}; pass --model-dir")
    return snaps[0]


def load(model_dir, dtype):
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        model_dir, dtype=dtype, output_loading_info=True)
    # The checkpoint also holds the vision tower, which the text model does
    # not use; everything else must match exactly.
    unexpected = [k for k in info["unexpected_keys"] if ".visual." not in k and not k.startswith("visual.")]
    for kind, keys in (("missing", info["missing_keys"]), ("unexpected", unexpected),
                       ("mismatched", info.get("mismatched_keys", []))):
        if keys:
            sys.exit(f"{kind} keys loading the text tower: {keys[:10]} ({len(keys)} total)")
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="target/qwen35_ref")
    args = ap.parse_args()
    model_dir = args.model_dir or find_snapshot()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    T = ids.shape[1]
    if T < MIN_TOKENS:
        sys.exit(f"prompt is {T} tokens; the test needs at least {MIN_TOKENS} (three GDN chunks)")
    np.save(os.path.join(args.out, "ids.npy"), ids[0].numpy().astype(np.int64))

    model = load(model_dir, torch.float32)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # embeddings, then each layer; the last is post-norm
    n_layers = model.config.num_hidden_layers
    if len(hs) != n_layers + 1:
        sys.exit(f"expected {n_layers + 1} hidden states, got {len(hs)}")
    for l in range(n_layers - 1):
        np.save(os.path.join(args.out, f"hidden_f32_{l}.npy"), hs[l + 1][0].float().numpy())
    # transformers replaces the last layer's output with the final norm's.
    np.save(os.path.join(args.out, "final_norm_f32.npy"), hs[n_layers][0].float().numpy())
    np.save(os.path.join(args.out, "logits_f32.npy"), out.logits[0].float().numpy())
    # A sanity line for the log: the model's own next-token guess.
    top = int(out.logits[0, -1].argmax())
    next_token = tok.decode([top])
    del model, out, hs

    model = load(model_dir, torch.bfloat16)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    np.save(os.path.join(args.out, "logits_bf16.npy"), out.logits[0].float().numpy())

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"model_dir {model_dir}\nT {T}\ntorch {torch.__version__}\n")
        import transformers
        f.write(f"transformers {transformers.__version__}\nprompt {PROMPT!r}\n")
        f.write(f"fp32 next token {next_token!r}\n")
    print(f"wrote {args.out}: T={T}, fp32 next token {next_token!r}")


if __name__ == "__main__":
    main()
