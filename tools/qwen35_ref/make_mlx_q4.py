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
     widened). It then casts every floating parameter of the same model to
     float32 (`to_float32`: the bf16 scales and biases widen exactly, so the
     4-bit values and the dequantized weights are unchanged; the dtype is
     checked at runtime) and writes `logits_mlx_f32.npy` beside each: mlx-lm's
     activations in float32, the reference for attributing a tessl-vs-mlx-lm gap
     to mlx-lm's own bfloat16 rounding. `--refs-only` rewrites only these
     files, keeping the converted checkpoint. The ids are the ones
     tools/qwen35_ref uses: the default prompt is
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


def dump_logits(model, ids, out_dir, file="logits_mlx.npy", native=None):
    """Run `model` on `ids` and save its logits widened to float32. `native` is
    the dtype the model's own logits must have (checked before the widening), so
    a "float32" run that quietly stayed bfloat16 aborts."""
    import mlx.core as mx

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "ids.npy"), ids.astype(np.int64))
    logits = model(mx.array(ids.astype(np.int64))[None])
    mx.eval(logits)
    if native is not None and logits.dtype != native:
        sys.exit(f"{out_dir}/{file}: the model's logits are {logits.dtype}, expected {native}")
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    arr = np.array(logits)[0]
    if not np.isfinite(arr).all():
        sys.exit(f"{out_dir}/{file}: mlx-lm produced non-finite logits")
    np.save(os.path.join(out_dir, file), arr.astype(np.float32))
    return arr


def to_float32(model):
    """Run the same 4-bit weights with float32 activations.

    mlx-lm 0.32.0 has no activation-dtype switch: every layer computes in the
    dtype of its floating parameters and inputs (the ids are integers). The
    quantized layers take their dtype from their `scales` / `biases`
    (`nn.QuantizedEmbedding` dequantizes to `scales.dtype`; `quantized_matmul`
    runs in the dtype of `x` and the scales), the norms from their weights. So
    casting every FLOATING parameter to float32 (the bf16 scales, biases, norms,
    conv, `dt_bias`, ...) runs the whole model in float32 on the identical
    4-bit values; the packed uint32 weights are not floating and are left
    alone. A bf16 value widens exactly, so the dequantized weights are the same
    numbers. Verified at runtime below, not assumed."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    before = {k: v.dtype for k, v in tree_flatten(model.parameters())}
    model.update(
        tree_map(lambda x: x.astype(mx.float32) if mx.issubdtype(x.dtype, mx.floating) else x, model.parameters())
    )
    mx.eval(model.parameters())
    wrong = []
    for k, v in tree_flatten(model.parameters()):
        if mx.issubdtype(before[k], mx.floating):
            if v.dtype != mx.float32:
                wrong.append((k, str(v.dtype)))
        elif v.dtype != before[k]:
            wrong.append((k, str(v.dtype)))
    if wrong:
        sys.exit(f"float32 cast left or changed parameters: {wrong[:5]} ({len(wrong)} total)")
    probe = model.language_model.model.embed_tokens(mx.array([[0]]))
    mx.eval(probe)
    if probe.dtype != mx.float32:
        sys.exit(f"the embedding output is {probe.dtype} after the float32 cast, expected float32")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="the bf16 snapshot (default: the local Qwen3.5-2B-Base)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--force", action="store_true", help="remove an existing --out under target/ first")
    ap.add_argument("--refs-only", action="store_true",
                    help="keep the converted checkpoint in --out and only rewrite tensors.txt, meta.txt and ref/ "
                         "(no conversion: the weights the Rust test loads stay byte for byte what they were)")
    args = ap.parse_args()
    model_dir = args.model_dir or find_snapshot()
    if not os.path.isdir(model_dir):
        sys.exit(f"{model_dir} is not a local directory; this script never downloads")

    block_hub()
    import mlx_lm
    from mlx_lm import convert, load

    if args.refs_only:
        if not os.path.exists(os.path.join(args.out, "config.json")):
            sys.exit(f"--refs-only needs a converted checkpoint in {args.out}")
    else:
        if os.path.exists(args.out):
            if not args.force:
                sys.exit(f"{args.out} exists; mlx_lm.convert refuses to overwrite. Re-run with --force "
                         "(removes it, only under target/), --refs-only, or remove it yourself.")
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
    prompts = (("default", default_ids), ("text", text_ids))
    bf16 = {}
    for name, ids in prompts:
        arr = dump_logits(model, ids, os.path.join(ref, name), native=mx.bfloat16)
        bf16[name] = arr
        summary.append(f"{name}: T={ids.shape[0]} vocab={arr.shape[1]} next token {int(arr[-1].argmax())}")
        print(summary[-1])

    # The same 4-bit weights with float32 activations, the attribution
    # reference: how far is mlx-lm's own bfloat16 rounding from the exact
    # computation on these weights, against how far tessl is.
    to_float32(model)
    for name, ids in prompts:
        arr = dump_logits(model, ids, os.path.join(ref, name), file="logits_mlx_f32.npy", native=mx.float32)
        diff = float(np.abs(arr - bf16[name]).max())
        if diff == 0.0:
            sys.exit(f"{name}: the float32 and bfloat16 runs are identical; the float32 run did not change the arithmetic")
        summary.append(f"{name}: float32 vs bfloat16 activations, max |dlogit| {diff:.4e}")
        print(summary[-1])

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"model_dir {model_dir}\nmlx_lm {mlx_lm.__version__}\nmlx {mx.__version__}\n")
        f.write(f"quantization {q}\n")
        f.write("\n".join(summary) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
