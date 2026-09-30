"""Where a Qwen3.5 training step's memory goes, per op, on MPS.

Every tensor autograd saves for backward, parameters excepted, is attributed to the innermost
module (or GDN function) that was running when it was saved, deduplicated by
storage. The model is Qwen3.5-2B's text config cut to one repeat of its layer
pattern (3 GDN layers + 1 attention layer) with random weights: what a layer
saves depends on shapes, not weights, so per-layer numbers scale to 24 layers
by 6x. The LM head's full-vocabulary loss is measured separately.

    python3 tools/qwen35_ref/saved_memory.py --tokens 512 1024 2048
"""

import argparse
import collections
import contextlib

import torch
import torch.nn.functional as F
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

MPS = torch.device("mps")
_label = ["(outside)"]


@contextlib.contextmanager
def labelled(name):
    _label.append(name)
    try:
        yield
    finally:
        _label.pop()


def wrap_fn(name):
    fn = getattr(mq, name)

    def wrapped(*a, **k):
        with labelled(name):
            return fn(*a, **k)

    setattr(mq, name, wrapped)


def attach(model):
    for name, mod in model.named_modules():
        if not name or list(mod.children()) and not name.endswith(("linear_attn", "self_attn", "mlp")):
            continue
        short = name.split(".", 2)[-1] if name.startswith("layers.") else name
        # layers.<i>.<rest>: keep the layer kind, drop the index.
        parts = name.split(".")
        if parts[0] == "layers" and len(parts) > 2:
            short = ".".join(parts[2:])
        mod.register_forward_pre_hook(lambda m, i, s=short: _label.append(s))
        mod.register_forward_hook(lambda m, i, o: (_label.pop(), None)[1])


def measure(model, tokens, vocab):
    # Parameters are resident anyway; a matmul saving the weight costs nothing.
    seen = {(p.untyped_storage().data_ptr(), p.device.type): 0 for p in model.parameters()}
    by = collections.Counter()

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.device.type)
        if key not in seen:
            nbytes = t.untyped_storage().nbytes()
            seen[key] = nbytes
            by[_label[-1]] += nbytes
        return t

    ids = torch.randint(0, vocab, (1, tokens), device=MPS)
    torch.mps.synchronize()
    torch.mps.empty_cache()
    base = torch.mps.driver_allocated_memory()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        hidden = model(input_ids=ids).last_hidden_state
        with labelled("lm_head+loss (full vocab)"):
            logits = hidden[0, :-1] @ model.embed_tokens.weight.T
            loss = F.cross_entropy(logits.float(), ids[0, 1:])
    torch.mps.synchronize()
    peak_fwd = torch.mps.driver_allocated_memory() - base
    loss.backward()
    torch.mps.synchronize()
    peak_all = torch.mps.driver_allocated_memory() - base
    return by, peak_fwd, peak_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024])
    args = ap.parse_args()
    cfg = Qwen3_5TextConfig(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=248_320,
        tie_word_embeddings=True,
    )
    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        wrap_fn(name)
    torch.manual_seed(0)
    model = mq.Qwen3_5TextModel(cfg).to(MPS, torch.bfloat16)
    model.train()
    attach(model)
    print(f"torch {torch.__version__}; bf16 on MPS; 3 GDN + 1 attention layer, Qwen3.5-2B dims")
    for t in args.tokens:
        by, peak_fwd, peak_all = measure(model, t, cfg.vocab_size)
        total = sum(by.values())
        print(f"\nT = {t}: saved for backward {total / 2**20:.0f} MiB; "
              f"driver memory above baseline: after forward {peak_fwd / 2**20:.0f} MiB, "
              f"after backward {peak_all / 2**20:.0f} MiB")
        for label, n in by.most_common():
            print(f"  {n / 2**20:9.1f} MiB  {100 * n / total:5.1f}%  {label}")
        for p in model.parameters():
            p.grad = None


if __name__ == "__main__":
    main()
