#!/usr/bin/env python3
"""The PyTorch lane that `bench_paired` alternates with `bench_embedgemma2`.

sentence-transformers' google/embeddinggemma-2 (modules.json: Transformer ->
mean Pooling -> Normalize; there is no Dense module, the 512 -> 768
embedding_projection is inside the Transformer) on MPS, fed the same token ids
as bench_embedgemma2 (no tokenization in either lane). Prints one JSON array like the Rust lane's.

    ~/.venvs/ml/bin/python bench/embedgemma2_torch.py --dtype f32

A workload runs as sentence-transformers' `encode` runs it: sorted longest
first and cut into batches of --batch-size (its default, 32), each padded to
its own longest sequence. One batch of a ragged workload padded to its
longest asked MPS for more memory than the machine has (16 GiB for one
attention at 1x6147 beside seven short texts).

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


def lengths(workload):
    # bench_embedgemma2's grammar: `+`-joined terms, each BxT or BxLO-HI, the
    # i-th sequence of a BxLO-HI term LO + (i * 2654435761) % (HI - LO + 1) long.
    lens = []
    for term in workload.strip().split("+"):
        b, t = term.split("x")
        lo, _, hi = t.partition("-")
        lo, hi = int(lo), int(hi or lo)
        lens += [lo + (i * 2654435761) % (hi - lo + 1) for i in range(int(b))]
    return lens


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
    ap.add_argument("--workloads", default=os.environ.get(
        "BENCH_WORKLOADS", "1x16,64x32,32x256,8x1024,2x4096,128x8-512,1x4096+63x32,1x6147+1x1658+6x10-42"))
    ap.add_argument("--batch-size", type=int, default=32)
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
        lens = lengths(w)
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        batches = []
        for c in range(0, len(order), args.batch_size):
            idx = order[c:c + args.batch_size]
            longest = lens[idx[0]]
            # Right-padded to the batch's longest, as the tokenizer pads.
            x = torch.tensor([ids(i, lens[i]) + [0] * (longest - lens[i]) for i in idx],
                             dtype=torch.long, device="mps")
            mask = torch.tensor([[1] * lens[i] + [0] * (longest - lens[i]) for i in idx],
                                dtype=torch.long, device="mps")
            batches.append({"input_ids": x, "attention_mask": mask})

        def run():
            with torch.no_grad():
                es = [model(dict(f))["sentence_embedding"] for f in batches]
            return torch.cat(es).float().cpu()

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
        rows.append({"workload": w.strip(), "backend": f"torch-mps-{args.dtype}",
                     "ms_min": round(ms[0], 4), "ms_median": round(statistics.median(ms), 4),
                     "iters": args.iters, "forwards": len(batches)})
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
