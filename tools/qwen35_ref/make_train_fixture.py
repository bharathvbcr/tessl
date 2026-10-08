"""The training-step reference for tessl's `Qwen35Model::train_step`
(tests/qwen35_train.rs): a Qwen3.5 model's loss and every parameter's
gradient from transformers' autograd.

    python3 tools/qwen35_ref/make_train_fixture.py tiny     # tests/fixtures/qwen35_train/
    python3 tools/qwen35_ref/make_train_fixture.py tiny --grouped   # tests/fixtures/qwen35_train_grouped/
    python3 tools/qwen35_ref/make_train_fixture.py 2b --tokens 128   # target/qwen35_train_ref/

`tiny` builds a small random `Qwen3_5ForCausalLM` of the 2B's shape family
(head_dim 256, GDN key heads of 128 with as many value heads, grouped
attention KV heads, tied embeddings), one GDN and one attention layer, and
commits it: a bf16 checkpoint, its config.json, the token ids, the loss and
one float32 .npy per parameter gradient. Every parameter is re-initialised at
a scale where every term matters (the zero-initialised norm weights would
otherwise hide a weight applied in the wrong order) and rounded to bf16, so
the checkpoint tessl loads and the float32 model torch differentiates hold the
same values.

`tiny --grouped` (tests/fixtures/qwen35_train_grouped/) is the committed tiny
model with two GDN key heads over four value heads, the 4B's ratio (16 key
heads, 32 value heads): transformers repeats each key head's q and k across
its value heads (`repeat_interleave` over heads) before the delta rule. Two
key heads tell that order from a tiled repeat.

`tiny --layers 24 --out target/qwen35_train_deep` is the same model 24 layers
deep in the 2B's layer pattern, for tests/qwen35_train.rs's
`deep_tiny_step_matches_transformers` (how gradient agreement changes with
depth alone, everything else equal).

`2b` runs Qwen/Qwen3.5-2B-Base from the Hugging Face cache in float32 on the
CPU (about 17 GB: weights, gradients, activations) over the first `--tokens`
tokens of make_reference.py's prompt, and writes the loss and a subset of the
gradients: every 1-D parameter and conv weight, every matrix of layers 0 (GDN)
and 3 (attention), and the embedding gradient's rows for the prompt's ids and
64 other ids. Run it alone: tessl's side loads the same model afterwards.

Both run the loss exactly as `Qwen3_5ForCausalLM(input_ids=ids, labels=ids)`
reports it: the mean cross-entropy of positions 0..T-2 predicting ids 1..T-1.
The attention is transformers' eager implementation, in float32.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TINY_T = 70  # past one 64-token GDN chunk and two 32-row attention blocks


def tiny_config(layers=2, k_heads=1, v_heads=1):
    # The 2B's pattern: three GDN layers, then full attention. Two layers
    # (the committed fixture) are one of each.
    types = ["linear_attention", "full_attention"] if layers == 2 else [
        "full_attention" if l % 4 == 3 else "linear_attention" for l in range(layers)]
    return mq.Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        layer_types=types,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=k_heads,
        linear_num_value_heads=v_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=64,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )


def reinit(model, gen):
    """Every parameter at a scale where it matters, then rounded to bf16."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.25)
            elif name.endswith("conv1d.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("A_log"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5 - 1.0)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif p.dim() == 1:
                # The zero-centred norms, applied as (1 + w).
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))
            else:
                fan_in = p.shape[1]
                p.copy_(torch.randn(p.shape, generator=gen) / fan_in ** 0.5)
            p.copy_(p.to(torch.bfloat16).float())


def step(model, ids):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids[None], labels=ids[None])
    out.loss.backward()
    return out.loss.item(), {n: p.grad.detach().float().cpu().numpy() for n, p in model.named_parameters()}


def write_common(out, loss, ids):
    np.save(os.path.join(out, "ids.npy"), ids.numpy().astype(np.int64))
    np.save(os.path.join(out, "loss.npy"), np.array([loss], dtype=np.float64))


def tiny(args):
    name = "qwen35_train_grouped" if args.grouped else "qwen35_train"
    out = args.out or os.path.join(ROOT, "tests", "fixtures", name)
    os.makedirs(out, exist_ok=True)
    gen = torch.Generator().manual_seed(20260930)
    cfg = tiny_config(args.layers, *((2, 4) if args.grouped else (1, 1)))
    torch.manual_seed(0)
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    reinit(model, gen)
    ids = torch.randint(0, cfg.vocab_size, (TINY_T,), generator=gen)
    loss, grads = step(model, ids)
    # The checkpoint, bf16 as published ones are; the tied lm_head is the
    # embedding and is not stored twice.
    state = {n: p.detach().to(torch.bfloat16).contiguous() for n, p in model.named_parameters()}
    save_file(state, os.path.join(out, "model.safetensors"))
    cfg_json = json.loads(cfg.to_json_string())
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=1, sort_keys=True)
    write_common(out, loss, ids)
    for n, g in grads.items():
        np.save(os.path.join(out, f"grad.{n}.npy"), g.astype(np.float32))
    print(f"tiny: loss {loss:.6f}, {len(grads)} gradients -> {out}")


def two_b(args):
    sys.path.insert(0, os.path.dirname(__file__))
    import make_reference as mr  # noqa: E402

    out = os.path.join(ROOT, "target", "qwen35_train_ref")
    os.makedirs(out, exist_ok=True)
    model_dir = args.model_dir or mr.find_snapshot()
    model = mr.load(model_dir, torch.float32)
    model.config._attn_implementation = "eager"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(mr.PROMPT, return_tensors="pt").input_ids[0][: args.tokens]
    if len(ids) < args.tokens:
        sys.exit(f"the prompt has {len(ids)} tokens, fewer than --tokens {args.tokens}")
    loss, grads = step(model, ids)
    write_common(out, loss, ids)
    keep = {}
    for n, g in grads.items():
        layer = n.split("layers.")[1].split(".")[0] if "layers." in n else None
        if g.ndim == 1 or "conv1d" in n or layer in ("0", "3"):
            keep[n] = g
    emb = [n for n in grads if n.endswith("embed_tokens.weight")][0]
    gen = torch.Generator().manual_seed(7)
    others = torch.randperm(grads[emb].shape[0], generator=gen)[:64].numpy()
    rows = np.unique(np.concatenate([ids.numpy(), others])).astype(np.int64)
    np.save(os.path.join(out, "embed_rows.npy"), rows)
    keep[emb + ".rows"] = grads[emb][rows]
    for n, g in keep.items():
        np.save(os.path.join(out, f"grad.{n}.npy"), g.astype(np.float32))
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"model {model_dir}\ntokens {args.tokens}\nloss {loss!r}\ngradients {sorted(keep)}\n")
    print(f"2b: loss {loss:.6f}, {len(keep)} gradients -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["tiny", "2b"])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--model-dir")
    ap.add_argument("--layers", type=int, default=2,
                    help="tiny: depth (the 2B's pattern of three GDN layers then attention); the "
                         "committed fixture is 2")
    ap.add_argument("--grouped", action="store_true",
                    help="tiny: two GDN key heads over four value heads, the 4B's ratio")
    ap.add_argument("--out", help="tiny: where to write (default: the committed fixture)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    (tiny if args.which == "tiny" else two_b)(args)


if __name__ == "__main__":
    main()
