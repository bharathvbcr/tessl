#!/usr/bin/env python3
"""The PyTorch lane for bench/paired_embedgemma2.py.

sentence-transformers' google/embeddinggemma-2 (Transformer -> mean Pooling ->
Dense -> Normalize) on MPS, fed the same token ids as bench_embedgemma2 (no
tokenization in either lane). Prints one JSON array like the Rust lane's.

    ~/.venvs/ml/bin/python bench/embedgemma2_torch.py --dtype f32

--dtype f32 is the like-for-like lane (tessl's forward is f32 with exact-f32
GEMMs); bf16 is the checkpoint's dtype, reported as what a torch deployment
would run. Attention is transformers' default implementation for the model
(--attn overrides it).
"""
import argparse
import glob
import json
import os
import statistics
import time

import torch
from sentence_transformers import SentenceTransformer


def ids(seq, length):
    # bench_embedgemma2's ids, bit for bit.
    return [2 if i == 0 else 1 if i == length - 1 else 1000 + (seq * 7919 + i * 104729) % 200000
            for i in range(length)]


def snapshot():
    if os.environ.get("EMBEDGEMMA2_SNAPSHOT"):
        return os.environ["EMBEDGEMMA2_SNAPSHOT"]
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/*/config.json")))
    if not hits:
        raise SystemExit("no google/embeddinggemma-2 snapshot; set EMBEDGEMMA2_SNAPSHOT")
    return os.path.dirname(hits[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--attn", default=None)
    ap.add_argument("--workloads", default=os.environ.get("BENCH_WORKLOADS", "1x16,64x32,32x256,8x1024,2x4096"))
    ap.add_argument("--iters", type=int, default=int(os.environ.get("BENCH_ITERS", "10")))
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", "3")))
    args = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    kwargs = {"dtype": torch.float32 if args.dtype == "f32" else torch.bfloat16}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    model = SentenceTransformer(snapshot(), device="mps", model_kwargs=kwargs).eval()

    rows = []
    for w in args.workloads.split(","):
        b, t = (int(x) for x in w.strip().split("x"))
        x = torch.tensor([ids(s, t) for s in range(b)], dtype=torch.long, device="mps")
        feats = {"input_ids": x, "attention_mask": torch.ones_like(x)}

        def run():
            with torch.no_grad():
                e = model(dict(feats))["sentence_embedding"]
            return e.float().cpu()

        e = run()
        norms = e.norm(dim=1)
        if not torch.isfinite(e).all() or (norms - 1).abs().max().item() > 1e-2:
            raise SystemExit(f"{w}: embeddings are not finite unit vectors")
        for _ in range(args.warmup):
            run()
        ms = []
        for _ in range(args.iters):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            run()
            ms.append((time.perf_counter() - t0) * 1e3)
        ms.sort()
        rows.append({"workload": f"{b}x{t}", "backend": f"torch-mps-{args.dtype}",
                     "ms_min": round(ms[0], 4), "ms_median": round(statistics.median(ms), 4),
                     "iters": args.iters})
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
