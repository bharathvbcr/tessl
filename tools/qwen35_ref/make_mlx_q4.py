"""Convert the local Qwen3.5-2B-Base to MLX affine 4-bit and dump mlx-lm's logits,
for tessl's MLX-Q4 tower test (`mlx_q4_2b_matches_mlx_lm_logits` in
tests/qwen35_mlx_q4.rs). A reference/fixture tool only; nothing here ships.

    python3 tools/qwen35_ref/make_mlx_q4.py            # writes target/qwen35_2b_mlx_q4/
    python3 tools/qwen35_ref/make_mlx_q4.py --force    # replace an earlier output

Run it with the venv that has mlx-lm (mlx-lm 0.32.0, mlx 0.32.3):

    /Users/bharath/.venvs/ml-2026-10/bin/python tools/qwen35_ref/make_mlx_q4.py

What it does:

  1. `mlx_lm.convert(quantize=True, q_bits=4, q_group_size=64)` on the local
     bf16 snapshot (the same one make_reference.py reads), into
     target/qwen35_2b_mlx_q4/ (config.json, model.safetensors, tokenizer files).
     This is `mlx_lm.convert -q --q-bits 4 --q-group-size 64`.
  2. Loads that converted model with mlx-lm and writes, for each fixed prompt,
     the token ids and mlx-lm's logits at every position:

       ref/default/ids.npy, ref/default/logits_mlx.npy   the default prompt of
                                                         make_reference.py
       ref/text/ids.npy,    ref/text/logits_mlx.npy      tests/fixtures/qwen35_text_ids.npy

     ids are int64 [T]; logits are float32 [T, vocab] (mlx-lm's bf16 logits,
     widened). The ids are the ones tools/qwen35_ref uses: the default prompt is
     tokenized with the same tokenizer, and compared with
     target/qwen35_ref/ids.npy when that exists (a mismatch aborts).
  3. Writes tensors.txt, every tensor of the converted checkpoint with its
     dtype and shape, which is what the Rust loader's key spelling
     (`language_model.model.*`), conv1d layout and untouched tensors are
     checked against, and meta.txt.

Offline only. HF_HUB_OFFLINE is set before anything is imported, and the
download entry points are replaced by functions that abort, so an attempt to
reach the hub fails the script instead of fetching.
"""
import argparse
import glob
import os
import shutil
import sys

# Offline before any hub-aware import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DEFAULT = os.path.join(REPO, "target", "qwen35_2b_mlx_q4")
TEXT_IDS = os.path.join(REPO, "tests", "fixtures", "qwen35_text_ids.npy")
HF_REF_IDS = os.path.join(REPO, "target", "qwen35_ref", "ids.npy")

# make_reference.py's PROMPT, verbatim (kept in step by the ids comparison
# against its ids.npy below, not by importing it: that module pulls in torch).
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


def block_hub():
    """Make every route to the hub raise."""

    def refuse(*a, **k):
        raise RuntimeError("hub access attempted; this script is offline-only")

    import huggingface_hub

    huggingface_hub.snapshot_download = refuse
    huggingface_hub.hf_hub_download = refuse
    import mlx_lm.utils as utils

    if hasattr(utils, "snapshot_download"):
        utils.snapshot_download = refuse


def find_snapshot():
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/*/")
    snaps = sorted(glob.glob(pat))
    if len(snaps) != 1:
        sys.exit(f"expected one snapshot at {pat}, found {snaps}; pass --model-dir")
    return snaps[0]


def under_target(path):
    target = os.path.realpath(os.path.join(REPO, "target"))
    return os.path.realpath(path).startswith(target + os.sep)


def dump_logits(model, ids, out_dir):
    import mlx.core as mx

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ids.npy"), ids.astype(np.int64))
    logits = model(mx.array(ids.astype(np.int64))[None])
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    arr = np.array(logits)[0]
    if not np.isfinite(arr).all():
        sys.exit(f"{out_dir}: mlx-lm produced non-finite logits")
    np.save(os.path.join(out_dir, "logits_mlx.npy"), arr.astype(np.float32))
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="the bf16 snapshot (default: the local Qwen3.5-2B-Base)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--force", action="store_true", help="remove an existing --out under target/ first")
    args = ap.parse_args()
    model_dir = args.model_dir or find_snapshot()
    if not os.path.isdir(model_dir):
        sys.exit(f"{model_dir} is not a local directory; this script never downloads")

    block_hub()
    import mlx_lm
    from mlx_lm import convert, load

    if os.path.exists(args.out):
        if not args.force:
            sys.exit(f"{args.out} exists; mlx_lm.convert refuses to overwrite. Re-run with --force "
                     "(removes it, only under target/) or remove it yourself.")
        if not under_target(args.out):
            sys.exit(f"--force removes only paths under {os.path.join(REPO, 'target')}, not {args.out}")
        shutil.rmtree(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    convert(model_dir, mlx_path=args.out, quantize=True, q_bits=4, q_group_size=64)

    # The converted checkpoint, as tessl will read it.
    import json

    with open(os.path.join(args.out, "config.json")) as f:
        cfg = json.load(f)
    q = cfg.get("quantization")
    if not q or q.get("bits") != 4 or q.get("group_size") != 64 or q.get("mode", "affine") != "affine":
        sys.exit(f"converted config has quantization {q!r}, expected affine 4-bit group 64")
    extra = [k for k in q if k not in ("bits", "group_size", "mode")]
    if extra:
        sys.exit(f"converted config has per-layer quantization entries {extra[:5]} (a mixed checkpoint)")

    import mlx.core as mx

    lines = []
    for path in sorted(glob.glob(os.path.join(args.out, "*.safetensors"))):
        for name, arr in sorted(mx.load(path).items()):
            lines.append(f"{name}\t{arr.dtype}\t{list(arr.shape)}")
    with open(os.path.join(args.out, "tensors.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    model, tokenizer = load(args.out)
    ref = os.path.join(args.out, "ref")

    default_ids = np.array(tokenizer.encode(PROMPT), dtype=np.int64)
    if default_ids.shape[0] < MIN_TOKENS:
        sys.exit(f"default prompt is {default_ids.shape[0]} tokens, expected at least {MIN_TOKENS}")
    if os.path.exists(HF_REF_IDS):
        hf_ids = np.load(HF_REF_IDS).astype(np.int64)
        if hf_ids.shape != default_ids.shape or (hf_ids != default_ids).any():
            sys.exit(f"mlx-lm's tokenization of the default prompt differs from {HF_REF_IDS}")
    else:
        print(f"note: {HF_REF_IDS} not found; the default prompt's ids were not cross-checked")
    text_ids = np.load(TEXT_IDS).astype(np.int64).reshape(-1)

    summary = []
    for name, ids in (("default", default_ids), ("text", text_ids)):
        arr = dump_logits(model, ids, os.path.join(ref, name))
        summary.append(f"{name}: T={ids.shape[0]} vocab={arr.shape[1]} next token {int(arr[-1].argmax())}")
        print(summary[-1])

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"model_dir {model_dir}\nmlx_lm {mlx_lm.__version__}\nmlx {mx.__version__}\n")
        f.write(f"quantization {q}\n")
        f.write("\n".join(summary) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
