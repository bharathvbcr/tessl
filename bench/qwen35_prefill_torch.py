#!/usr/bin/env python3
"""The PyTorch MPS lane of the Qwen3.5-2B prefill+score head-to-head.

transformers (5.19.0) `AutoModelForCausalLM` on the Qwen3.5-2B-Base snapshot,
bf16 (the checkpoint's dtype; the 2026-10-04 PyTorch 2.12.1 run was bf16 sdpa too),
on MPS, fed the ids file `bench_qwen35_layers --prefill-score` reads. The work,
the output rows, the knobs and the exact `bench_paired --evidence` command are in
`bench/prefill_lane.py`'s docstring; this file is only the torch half.

  ~/.venvs/ml-2026-10/bin/python bench/qwen35_prefill_torch.py --lengths 200,1024,2048,8192

The body is `model.model` (embedding gather, 24 layers with `use_cache=True`, so
the GDN and KV state are kept, and the final norm); the head is the f32 dot of
the last normed row with the 17 answer rows of the tied embedding table, then a
log-softmax over the 17, read back to the host. No `[T, vocab]` logits.
`--device cpu` exists to smoke-test the lane without the GPU; its rows say so.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_lane as pl  # noqa: E402


def build_parser():
    ap = pl.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16",
                    help="bf16 is the checkpoint's dtype and what the earlier run used")
    ap.add_argument("--attn", default=None, help="attn_implementation (default: transformers' own, sdpa)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    iters, warmup = pl.resolve_counts(args)
    doc, ids, digests = pl.prepare(args)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    device = "cpu" if args.device == "cpu" else "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    snap = args.snapshot or pl.snapshot_default()
    kwargs = {"dtype": torch.bfloat16 if args.dtype == "bf16" else torch.float32}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    model = AutoModelForCausalLM.from_pretrained(snap, **kwargs).to(device).eval()
    if model.config.vocab_size != doc["vocab"]:
        raise SystemExit(f"model vocab {model.config.vocab_size} but the ids file says {doc['vocab']}")
    body = model.model
    table = model.get_input_embeddings().weight
    answer_rows = table[torch.tensor(doc["answers"], device=device)].float()

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def forward(ids_list):
        x = torch.tensor([ids_list], dtype=torch.long, device=device)
        with torch.no_grad():
            h = body(input_ids=x, use_cache=True).last_hidden_state
            logits = h[:, -1].float() @ answer_rows.T
            lp = torch.log_softmax(logits, dim=-1)[0].cpu().tolist()
        sync()
        return lp, h.shape[1]

    def one(t):
        sync()
        ms, (lp, ran) = pl.timed(lambda: forward(ids[t]))
        return ms, lp, ran

    backend = f"torch-{torch.__version__}-{transformers.__version__}-{device}-{args.dtype}"
    print(f"{backend}: snapshot {os.path.basename(snap.rstrip('/'))}, iters {iters}, warmup {warmup}",
          file=sys.stderr, flush=True)
    samples = pl.run_rounds(args.lengths, iters, warmup, one, pl.sink_scores)
    pl.emit(pl.format_rows(backend, args.lengths, samples, iters, len(doc["answers"]), digests))


if __name__ == "__main__":
    main()
