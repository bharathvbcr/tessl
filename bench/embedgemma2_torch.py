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
attention at 1x6147 beside seven short texts). Batches of 32 still do it for
two of the default workloads on a 64 GiB M5 Pro: 1x4096+63x32 (32 texts padded
to 4096; stopped at the 32 GiB cap in f32 and bf16, and once refused by MPS itself) and
1x6147+1x1658+6x10-42 (eight texts padded to 6147; one f32 run finished in
17.5 s, the next was stopped at the machine's 32 GiB per-process cap). A run
that has to finish leaves both out with BENCH_WORKLOADS.

The MPS cache is emptied between workloads, outside the timed region, so one
workload's cached buffers are not held through the next. Each workload's
starting MPS driver memory goes to stderr.

Memory, per workload, from one further untimed pass after the timed iterations
(`memory_pass`): `tensor_mib` is the most `torch.mps.current_allocated_memory()`
(the bytes live tensors hold) read at any module boundary of the forward, above
what was live before the pass (the weights and the padded input batches);
`tensor_base_mib` is that baseline; `driver_mib` is
`torch.mps.driver_allocated_memory()` after the pass. `tensor_mib` is the
memory metric: it is the same every round, and `bench_paired` carries it into
its report (`tensor_mib_max`). `driver_mib` includes the caching allocator's
retained pool, which does not repeat from round to round at the same work, so
it is recorded in the row (kept verbatim in the report's raw rows and in the
lane's stdout under `bench_paired --out PATH.lanes/`) and on stderr, and is
not summarised (no `driver_mib_max`). `tensor_mib` is
a lower bound on the true peak (torch has no MPS peak counter, and a module's
internal temporaries are seen only at its boundaries).

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
import sys
import time

import torch


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


def mib(n):
    return n / 2**20


class PeakSampler:
    """Largest `current()` seen at the entry and exit of every module of `model`
    while the context is open. The hooks are removed on exit, also on an error."""

    def __init__(self, model, current):
        self.model, self.current, self.peak, self.handles = model, current, 0, []

    def sample(self, *_):
        self.peak = max(self.peak, self.current())

    def __enter__(self):
        self.sample()
        for m in self.model.modules():
            self.handles.append(m.register_forward_pre_hook(self.sample))
            self.handles.append(m.register_forward_hook(self.sample))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.sample()
        return False


def memory_pass(model, run, current, driver, sync):
    """One untimed `run()` with the live tensor bytes (`current`, bytes) sampled
    at every module boundary. Returns {tensor_mib, tensor_base_mib, driver_mib}:
    the peak sample above the bytes live before the pass, that baseline, and the
    driver's allocation (`driver`, bytes) after the pass, which is recorded, not
    gated."""
    sync()
    base = current()
    with PeakSampler(model, current) as s:
        run()
        sync()
    peak = max(s.peak, current())
    return {"tensor_mib": round(mib(max(peak - base, 0)), 1),
            "tensor_base_mib": round(mib(base), 1),
            "driver_mib": round(mib(driver()), 1)}


def workload_row(workload, dtype, ms_sorted, iters, forwards, mem):
    """The row `bench_paired --format workloads` reads: `ms_min` is the fastest
    of the sorted `ms_sorted`, and the `memory_pass` fields ride along
    (`tensor_mib` is carried into the report, `driver_mib` is recorded only)."""
    return {"workload": workload.strip(), "backend": f"torch-mps-{dtype}",
            "ms_min": round(ms_sorted[0], 4), "ms_median": round(statistics.median(ms_sorted), 4),
            "iters": iters, "forwards": forwards, **mem}


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

    from sentence_transformers import SentenceTransformer

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

        print(f"{w}: MPS driver holds {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB",
              file=sys.stderr, flush=True)
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
        mem = memory_pass(model, run, torch.mps.current_allocated_memory,
                          torch.mps.driver_allocated_memory, torch.mps.synchronize)
        print(f"{w}: tensor_mib={mem['tensor_mib']} (above {mem['tensor_base_mib']}) "
              f"driver_mib={mem['driver_mib']} (recorded, not gated)", file=sys.stderr, flush=True)
        rows.append(workload_row(w, args.dtype, ms, args.iters, len(batches), mem))
        batches.clear()
        del e
        torch.mps.empty_cache()
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
