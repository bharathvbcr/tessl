"""How far float32 gradients of the 2B training step move for a given
movement of the forward: the scale tessl's gradients are compared on.

    python3 tools/qwen35_ref/train_noise_floor.py                  # after make_train_fixture.py 2b
    python3 tools/qwen35_ref/train_noise_floor.py --perturb 1e-6

Recomputes the loss and the gradients make_train_fixture.py 2b kept, with
the same weights and ids but a different float32 evaluation order (SDPA
attention instead of eager, a different CPU thread count, so every
reduction is summed differently), and prints, per parameter, the same
`max|a - b| / max|b|` tests/qwen35_train.rs reports for tessl.

With no option that is transformers disagreeing with itself. With
`--perturb eps` every weight is also scaled by `1 + eps * N(0, 1)`; choosing
eps so the loss moves by as much as tessl's forward differs from
transformers' shows how far the gradients move for that much forward
difference, which separates a forward-borne gradient difference from a
backward bug. Needs the same ~17 GB as make_train_fixture.py 2b; run it
alone.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import make_reference as mr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="also scale every weight by (1 + eps * N(0, 1))")
    args = ap.parse_args()
    ref = os.path.join(ROOT, "target", "qwen35_train_ref")
    ids = torch.from_numpy(np.load(os.path.join(ref, "ids.npy")))
    want_loss = float(np.load(os.path.join(ref, "loss.npy"))[0])
    torch.set_num_threads(max(1, os.cpu_count() // 2 - 1))
    model = mr.load(mr.find_snapshot(), torch.float32)
    model.config._attn_implementation = "sdpa"
    # The forward movement is measured where the Qwen3.5 parity test measures
    # tessl's (tests/qwen35_model.rs): logits, max|a - b| / max|b|.
    with torch.no_grad():
        base = model(input_ids=ids[None]).logits[0].double()
    if args.perturb:
        gen = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(1 + args.perturb * torch.randn(p.shape, generator=gen))
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids[None], labels=ids[None])
    out.loss.backward()
    loss = out.loss.item()
    moved = (out.logits[0].detach().double() - base).abs().max() / base.abs().max()
    print(f"logits moved by {moved.item():.2e} of max|logit| (tessl's F32 forward: 1.9e-6, docs/qwen35.md)")
    print(f"loss: sdpa, perturb {args.perturb:g}: {loss:.8f} vs eager {want_loss:.8f} "
          f"(rel {abs(loss - want_loss) / want_loss:.2e})")
    rows = np.load(os.path.join(ref, "embed_rows.npy"))
    results = []
    for n, p in model.named_parameters():
        g = p.grad.detach().float().numpy()
        path = os.path.join(ref, f"grad.{n}.npy")
        if n.endswith("embed_tokens.weight"):
            path, g = os.path.join(ref, f"grad.{n}.rows.npy"), g[rows]
        if not os.path.exists(path):
            continue
        want = np.load(path).astype(np.float64)
        r = np.abs(g - want).max() / max(np.abs(want).max(), 1e-30)
        results.append((r, n))
    results.sort()
    for r, n in results[-12:]:
        print(f"{r:.2e} {n}")
    print(f"worst {results[-1][0]:.2e} over {len(results)} parameters; median {results[len(results) // 2][0]:.2e}")


if __name__ == "__main__":
    main()
