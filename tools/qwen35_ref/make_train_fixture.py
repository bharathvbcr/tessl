"""The training-step reference for tessl's `Qwen35Model::train_step`
(tests/qwen35_train.rs): a Qwen3.5 model's loss and every parameter's
gradient from transformers' autograd.

    python3 tools/qwen35_ref/make_train_fixture.py tiny     # tests/fixtures/qwen35_train/
    python3 tools/qwen35_ref/make_train_fixture.py tiny --grouped   # tests/fixtures/qwen35_train_grouped/
    python3 tools/qwen35_ref/make_train_fixture.py tiny --untied    # tests/fixtures/qwen35_train_untied/
    python3 tools/qwen35_ref/make_train_fixture.py 2b --tokens 128   # target/qwen35_train_ref/
    python3 tools/qwen35_ref/make_train_fixture.py 2b --tokens 128 --f64   # target/qwen35_train_ref_f64/

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

`tiny --untied` (tests/fixtures/qwen35_train_untied/) is the committed tiny
model with `tie_word_embeddings: false`, the 9B's layout: the checkpoint
carries a separate `lm_head.weight` at its root, and the gradients include
`grad.lm_head.weight.npy` (the embedding's is then the gather's alone). It
also writes `logits.npy`, the float32 model's `[T, vocab]` logits, for the
forward-logits comparison.

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

`2b --f64` (target/qwen35_train_ref_f64/) is the same gradients from the exact
function: the whole forward and backward in float64, for
tests/qwen35_train.rs's `real_2b_step_matches_the_f64_gradient` (tessl's f32
step against the true gradient, with no finite differences). transformers
hard-codes float32 in the Qwen3.5 path (the GDN's `to(torch.float32)`, the
gated norm, the softmax's `dtype=`, `.float()` in the norms, the rotary and
the loss), so a float64 model alone would still round to float32 in the
middle; `F64Only` rewrites every such request to float64 and raises on any
floating result that is not float64. The model is the checkpoint's bf16
values exactly (every bf16 is a float64). Only the parameters the fixture
keeps carry gradients (see `step_f64` for the memory this buys). The files
are the fp32 fixture's, the gradients float64, and meta.txt says `dtype f64`.

Both run the loss exactly as `Qwen3_5ForCausalLM(input_ids=ids, labels=ids)`
reports it: the mean cross-entropy of positions 0..T-2 predicting ids 1..T-1.
The attention is transformers' eager implementation, in float32.
"""
import argparse
import collections
import json
import os
import resource
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from torch.overrides import TorchFunctionMode
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixture_manifest as fm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GENERATOR = "tools/qwen35_ref/make_train_fixture.py"
TINY_T = 70  # past one 64-token GDN chunk and two 32-row attention blocks


def tiny_config(layers=2, k_heads=1, v_heads=1, tie=True):
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
        tie_word_embeddings=tie,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )


def reinit(model, gen):
    """Every parameter at a scale where it matters, then rounded to bf16."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
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


def train_manifest(directory, out, grouped, role, reproduction):
    """MANIFEST.json for a committed `tiny` fixture directory (the format and its
    meaning are in fixture_manifest.py). Every file here is a golden of record: the
    loss and the gradients come from transformers' own autograd, and the checkpoint
    and ids are what they were differentiated at."""
    files = {}
    for name in sorted(os.listdir(out)):
        if name == "config.json":
            files[name] = dict(role="config", published=True, note="the tiny model's config.json")
        elif name == "model.safetensors":
            files[name] = dict(role="model", published=True,
                               note="bf16 checkpoint; the float32 model torch differentiated holds the same values")
        elif name == "ids.npy":
            files[name] = dict(role="input", published=True, note="token ids the loss and gradients are at")
        elif name == "loss.npy":
            files[name] = dict(role="golden", published=True, note="mean cross-entropy, float64")
        else:
            files[name] = dict(role="golden", published=True,
                               note="float32 gradient from transformers' loss.backward()")
    drift = None
    if reproduction is not None:
        worst = max(reproduction["max_relative_to_peak"].values(), default=0.0)
        drift = (f"goldens reproduce to {worst:.1e} of each file's peak under torch "
                 f"{fm.versions()['torch']} / transformers {fm.versions()['transformers']} (this run: "
                 f"{len(reproduction['byte_identical'])} files byte-identical incl. model.safetensors and ids.npy, "
                 f"the gradients and loss moved by arithmetic only); the audit of 2026-10-09 measured, under torch "
                 f"2.13.0 / transformers 5.14.1, 25 of 29 gradients bit-identical and the other 4 within 2.4e-7 of "
                 f"peak (unverified here). The committed files were not regenerated.")
    heads = "2 key heads over 4 value heads" if grouped else "1 key head over 1 value head"
    fm.write_manifest(
        directory, GENERATOR,
        f"drawn: torch.Generator().manual_seed(20260930) feeds the parameter init and the token ids; the "
        f"drawn model (model.safetensors) and ids.npy are committed, so the draws are the inputs. {heads}.",
        files, role, reproduction=reproduction,
        extra={**({"drift": drift} if drift else {}), "gdn_path": "chunked: Qwen3_5ForCausalLM runs with no cache, so every GDN layer calls "
                           "transformers' torch_chunk_gated_delta_rule (modeling_qwen3_5.py, 5.17.0 :301) and "
                           "autograd differentiates it; the recurrent rule runs only for a cached single-token "
                           "step. Both read the decayed state, which is the published rule (arXiv:2412.06464 eq. 8). "
                           "Checked by wrapping both entry points over one forward at T=70."})


def tiny(args):
    if args.untied and args.grouped:
        sys.exit("--untied and --grouped are separate fixtures")
    name = "qwen35_train_untied" if args.untied else "qwen35_train_grouped" if args.grouped else "qwen35_train"
    out = args.out or os.path.join(ROOT, "tests", "fixtures", name)
    os.makedirs(out, exist_ok=True)
    gen = torch.Generator().manual_seed(20260930)
    cfg = tiny_config(args.layers, *((2, 4) if args.grouped else (1, 1)), tie=not args.untied)
    torch.manual_seed(0)
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    reinit(model, gen)
    ids = torch.randint(0, cfg.vocab_size, (TINY_T,), generator=gen)
    if args.untied:
        with torch.no_grad():
            logits = model(input_ids=ids[None]).logits[0].float().cpu().numpy()
        np.save(os.path.join(out, "logits.npy"), logits.astype(np.float32))
    loss, grads = step(model, ids)
    # The checkpoint, bf16 as published ones are; a tied lm_head is the
    # embedding and is not stored twice (an untied one is its own tensor).
    state = {n: p.detach().to(torch.bfloat16).contiguous() for n, p in model.named_parameters()}
    save_file(state, os.path.join(out, "model.safetensors"))
    cfg_json = json.loads(cfg.to_json_string())
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=1, sort_keys=True)
    write_common(out, loss, ids)
    for n, g in grads.items():
        np.save(os.path.join(out, f"grad.{n}.npy"), g.astype(np.float32))
    if args.reproduces:
        # The committed files predate the manifest: compare this run with them and
        # record that, rather than claiming they were produced by it.
        train_manifest(args.reproduces, out, args.grouped, "reproduced", fm.compare(args.reproduces, out))
    else:
        train_manifest(out, out, args.grouped, "produced", None)
    print(f"tiny: loss {loss:.6f}, {len(grads)} gradients -> {out}")


def keep_parameter(name, ndim):
    """The 2b fixture's kept gradients (the embedding is kept as rows, apart):
    every 1-D parameter and conv weight, and every parameter of layers 0 (GDN)
    and 3 (attention)."""
    layer = name.split("layers.")[1].split(".")[0] if "layers." in name else None
    return ndim == 1 or "conv1d" in name or layer in ("0", "3")


def _floats(x):
    if isinstance(x, torch.Tensor):
        if x.is_floating_point():
            yield x
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _floats(y)


class F64Only(TorchFunctionMode):
    """Runs everything under it in float64, or raises.

    transformers 5.19.0's Qwen3.5 path (and its loss) ask for float32 in
    places, which on a float64 model would round the middle of the computation
    to float32 and make the "exact" gradient an fp32 one. Every torch call
    under this mode has each float32 request (a positional or `dtype=` float32,
    `torch.float` included) rewritten to float64 and `Tensor.float()` run as
    `Tensor.double()`; and every floating tensor any call returns is checked
    to be float64, so a cast this does not recognise (a half, a bfloat16, a
    float32 made some other way) is a RuntimeError naming the call, never a
    quiet float32. `sites` counts each rewrite by the transformers line that
    asked for it; the cast sites (installed transformers 5.19.0, under
    site-packages/transformers/). Rewritten on a tiny Qwen3.5 text model's
    forward and backward (the other 2B-path sites run the same code):
      models/qwen3_5/modeling_qwen3_5.py:  177 (rotary inv_freq
      `torch.arange(dtype=torch.float)`), 186 (rotary positions `.float()` and
      `inv_freq.to(dtype=torch.float)`), 217 and 222 (gated norm
      `.to(torch.float32)`), 329 (chunked delta rule `.to(torch.float32)`), 609
      (GDN decay `A_log.float()`, `a.float()`), 731 (eager attention
      `softmax(dtype=torch.float32)`), 841 and 844 (RMSNorm `.float()`);
      loss/loss_utils.py: 59 (`logits.float()`).
    Present in the file and covered by the same rewrite, not reached by a
    text-only no-cache step: 449 (the recurrent delta rule, one cached token)
    and 113, 119, 122, 992, 993 (the vision tower's rotary).
    """

    def __init__(self):
        super().__init__()
        self.sites = collections.Counter()

    def _site(self, what):
        f = sys._getframe(1)
        while f is not None and ("/torch/" in f.f_code.co_filename or f.f_code is self.__torch_function__.__code__):
            f = f.f_back
        where = "?"
        if f is not None:
            path = f.f_code.co_filename
            where = f"{path.split('site-packages/')[-1]}:{f.f_lineno}"
        self.sites[f"{where}  {what}"] += 1

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.float:
            self._site(".float() -> .double()")
            func = torch.Tensor.double
        if any(a is torch.float32 for a in args) or any(v is torch.float32 for v in kwargs.values()):
            self._site(f"{getattr(func, '__name__', func)}(float32) -> float64")
            args = tuple(torch.float64 if a is torch.float32 else a for a in args)
            kwargs = {k: torch.float64 if v is torch.float32 else v for k, v in kwargs.items()}
        out = func(*args, **kwargs)
        for t in _floats(out):
            if t.dtype != torch.float64:
                raise RuntimeError(f"{getattr(func, '__name__', func)} returned {t.dtype}, not float64")
        return out


def step_f64(model, ids, rows):
    """The loss and the kept gradients of a float64 model, exact: returns
    (loss, {name: float64 array}, meta lines).

    Memory (about 1.88 B parameters, 248,320 x 2048 embedding, 128 tokens;
    estimated, not measured on this machine: the meta.txt it writes records
    the peak resident set it actually had): the weights are 1.88e9 x 8 B =
    14.0 GiB. Only the kept parameters have requires_grad: the embedding's
    lm_head side (3.8 GiB dense) and layers 0 and 3 and the 1-D and conv
    parameters (about 0.9 GiB), 4.7 GiB of gradients. The tied embedding is
    split into two parameters on one storage (no copy), the lookup's with a
    sparse gradient (only the rows the ids touch), so the 3.8 GiB tied
    gradient is never built twice or summed out of place (that would have been
    11.4 GiB at the end of the backward; the sum is taken on the kept rows
    only). The activations (128 tokens: the 128 x 248,320 f64 logits are 0.24
    GiB, and the loss keeps a few copies of them; the layers' are smaller) are
    about 2-3 GiB. Peak about 14.0 + 4.7 + 3 + interpreter and torch about 1
    = 22-23 GiB, under memguard's 32 GiB cap.
    """
    if not model.config.tie_word_embeddings:
        sys.exit("--f64 handles the 2B's tied embedding; this model is untied")
    emb_mod = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head.weight is not emb_mod.weight:
        sys.exit("lm_head.weight is not the embedding: the tie this fixture splits is not there")
    emb_name = "model.embed_tokens.weight"
    for n, p in model.named_parameters():
        if p.dtype != torch.float64:
            sys.exit(f"{n} is {p.dtype}: the model was not loaded in float64")
    model.requires_grad_(False)
    kept = {n: p for n, p in model.named_parameters() if n != emb_name and keep_parameter(n, p.ndim)}
    for p in kept.values():
        p.requires_grad_(True)
    # The tie split: lm_head's own Parameter over the embedding's storage.
    head.weight = torch.nn.Parameter(emb_mod.weight.detach())
    emb_mod.weight.requires_grad_(True)
    emb_mod.sparse = True
    # The rotary tables were built in float32 at construction; rebuild them in
    # float64 (inv_freq = 1 / theta^(2i/d) is not exact in float32).
    with F64Only() as mode:
        for m in model.modules():
            if hasattr(m, "compute_default_rope_parameters") and hasattr(m, "inv_freq"):
                if m.rope_type != "default":
                    sys.exit(f"rope type {m.rope_type}: only the default rotary is rebuilt in float64")
                inv_freq, _ = m.compute_default_rope_parameters(m.config)
                m.inv_freq = inv_freq
                m.original_inv_freq = inv_freq.clone()
        out = model(input_ids=ids[None], labels=ids[None], use_cache=False)
        out.loss.backward()
    sites = mode.sites
    if out.loss.dtype != torch.float64:
        sys.exit(f"the loss is {out.loss.dtype}")
    loss = out.loss.item()
    grads = {}
    for n, p in kept.items():
        if p.grad is None or p.grad.dtype != torch.float64:
            sys.exit(f"{n}: gradient {None if p.grad is None else p.grad.dtype}, not a float64 gradient")
        grads[n] = p.grad.detach().cpu().numpy()
    # The tied embedding's gradient on the kept rows: lm_head's side plus the
    # lookup's (the sparse rows, each id's rows summed by coalesce).
    if not emb_mod.weight.grad.is_sparse or head.weight.grad.is_sparse:
        sys.exit("the embedding's gradient layout is not the sparse lookup + dense head this assumes")
    want = head.weight.grad[torch.from_numpy(rows)].clone()
    look = emb_mod.weight.grad.coalesce()
    idx = look.indices()[0].numpy()
    pos = np.searchsorted(rows, idx)
    if not (pos < len(rows)).all() or not (rows[pos] == idx).all():
        sys.exit("the lookup touched a row outside the kept rows")
    want[torch.from_numpy(pos)] += look.values()
    grads[emb_name + ".rows"] = want.numpy()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gib = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    notes = f"peak_rss_gib {rss_gib:.2f}\nf64_cast_sites_rewritten:\n" + "".join(
        f"  {n}x {s}\n" for s, n in sorted(sites.items()))
    return loss, grads, notes


def two_b(args):
    sys.path.insert(0, os.path.dirname(__file__))
    import make_reference as mr  # noqa: E402

    f64 = args.f64
    out = os.path.join(ROOT, "target", "qwen35_train_ref_f64" if f64 else "qwen35_train_ref")
    os.makedirs(out, exist_ok=True)
    model_dir = args.model_dir or mr.find_snapshot()
    model = mr.load(model_dir, torch.float64 if f64 else torch.float32)
    model.config._attn_implementation = "eager"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(mr.PROMPT, return_tensors="pt").input_ids[0][: args.tokens]
    if len(ids) < args.tokens:
        sys.exit(f"the prompt has {len(ids)} tokens, fewer than --tokens {args.tokens}")
    emb = "model.embed_tokens.weight"
    gen = torch.Generator().manual_seed(7)
    others = torch.randperm(model.get_input_embeddings().weight.shape[0], generator=gen)[:64].numpy()
    rows = np.unique(np.concatenate([ids.numpy(), others])).astype(np.int64)
    if f64:
        loss, keep, notes = step_f64(model, ids, rows)
        dtype = np.float64
    else:
        loss, grads = step(model, ids)
        keep = {n: g for n, g in grads.items() if keep_parameter(n, g.ndim)}
        keep[emb + ".rows"] = grads[emb][rows]
        notes, dtype = "", np.float32
    write_common(out, loss, ids)
    np.save(os.path.join(out, "embed_rows.npy"), rows)
    for n, g in keep.items():
        np.save(os.path.join(out, f"grad.{n}.npy"), g.astype(dtype))
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"model {model_dir}\ntokens {args.tokens}\nloss {loss!r}\ngradients {sorted(keep)}\n")
        if f64:
            f.write(f"dtype f64\n{notes}")
    print(f"2b{' f64' if f64 else ''}: loss {loss:.6f}, {len(keep)} gradients -> {out}")


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
    ap.add_argument("--untied", action="store_true",
                    help="tiny: tie_word_embeddings false, a separate lm_head.weight (the 9B's layout)")
    ap.add_argument("--f64", action="store_true",
                    help="2b: the exact reference, the whole forward and backward in float64, into "
                         "target/qwen35_train_ref_f64 (about 23 GiB; see step_f64)")
    ap.add_argument("--out", help="tiny: where to write (default: the committed fixture)")
    ap.add_argument("--reproduces", metavar="DIR",
                    help="tiny, with --out: DIR is a committed fixture. Compare this run with it and write "
                         "DIR/MANIFEST.json recording the comparison (versions_role: reproduced)")
    args = ap.parse_args()
    if args.f64 and args.which != "2b":
        sys.exit("--f64 is for 2b only")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    (tiny if args.which == "tiny" else two_b)(args)


if __name__ == "__main__":
    main()
