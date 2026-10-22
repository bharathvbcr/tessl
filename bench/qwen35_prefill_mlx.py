#!/usr/bin/env python3
"""The mlx-lm lane of the Qwen3.5-2B prefill+score head-to-head.

mlx-lm 0.32.0's `qwen3_5` model (mlx 0.32.3) on the Qwen3.5-2B-Base snapshot's own
bf16 weights, not quantized (like for like with the torch and tessl lanes), fed
the ids file `bench_qwen35_layers --prefill-score` reads. The work, the output
rows, the knobs and the exact `bench_paired --evidence` command are in
`bench/prefill_lane.py`'s docstring; this file is only the MLX half.

  ~/.venvs/ml-2026-10/bin/python bench/qwen35_prefill_mlx.py --lengths 200,1024,2048,8192

The body is `model.model` (embedding gather, 24 layers over a fresh
`make_cache()` that is evaluated with the output, so the GDN and KV state are
kept, and the final norm), one pass for the whole prompt (no chunked prefill);
the head is the f32 dot of the last normed row with the 17 answer rows of the
tied embedding table, then a log-softmax over the 17, read back to the host. No
`[T, vocab]` logits. `--device cpu` exists to smoke-test the lane without the
GPU; its rows say so.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_lane as pl  # noqa: E402


def build_parser():
    return pl.base_parser(__doc__.split("\n\n")[0])


def main(argv=None):
    args = build_parser().parse_args(argv)
    iters, warmup = pl.resolve_counts(args)
    doc, ids, digests = pl.prepare(args)

    from pathlib import Path

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.utils import load_model

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    snap = args.snapshot or pl.snapshot_default()
    model, config = load_model(Path(snap))
    model.eval()
    body = model.model
    table = body.embed_tokens.weight
    if table.dtype != mx.bfloat16:
        raise SystemExit(f"embedding table is {table.dtype}, not bfloat16: refusing a non-like-for-like lane")
    if table.shape[0] != doc["vocab"]:
        raise SystemExit(f"model vocab {table.shape[0]} but the ids file says {doc['vocab']}")
    answer_rows = table[mx.array(doc["answers"], dtype=mx.int32)].astype(mx.float32)
    mx.eval(answer_rows)

    def forward(ids_list):
        x = mx.array([ids_list], dtype=mx.int32)
        cache = model.make_cache()
        h = body(x, cache)
        logits = h[:, -1].astype(mx.float32) @ answer_rows.T
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        state = [a for c in cache for a in c.state if a is not None]
        mx.eval(lp, *state)
        return lp[0].tolist(), h.shape[1]

    def one(t):
        ms, (lp, ran) = pl.timed(lambda: forward(ids[t]))
        return ms, lp, ran

    backend = f"mlx-lm-{mlx_lm.__version__}-mlx-{mx.__version__}-{args.device}-bf16"
    print(f"{backend}: snapshot {os.path.basename(snap.rstrip('/'))}, iters {iters}, warmup {warmup}",
          file=sys.stderr, flush=True)
    samples = pl.run_rounds(args.lengths, iters, warmup, one, pl.sink_scores)
    pl.emit(pl.format_rows(backend, args.lengths, samples, iters, len(doc["answers"]), digests))


if __name__ == "__main__":
    main()
