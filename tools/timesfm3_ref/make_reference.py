#!/usr/bin/env python3
"""Reference outputs of TimesFM 3 from Google's own PyTorch port, for tessl's
TimesFM 3 parity tests (tests/timesfm3_*.rs).

Python is the oracle only. The upstream source is not installed or vendored: it
is read by path from a checkout of google-research/timesfm at the pinned commit

    PINNED = 918df02c9d366b2145bc5876135fa2131d3ed2a6

which tools/timesfm3_ref/fetch_upstream.sh unpacks into
target/timesfm3_upstream/<PINNED>/ (Apache-2.0 code; see THIRD_PARTY_NOTICES.md).

    ~/.venvs/ml/bin/python tools/timesfm3_ref/make_reference.py --tiny \
        --out tests/fixtures/timesfm3_tiny            # committed
    ~/.venvs/ml/bin/python tools/timesfm3_ref/make_reference.py \
        --model-dir <hf snapshot of google/timesfm-3.0-pytorch> \
        --out target/timesfm3_ref_real                # never committed

The real checkpoint is under the TimesFM Non-Commercial License v1.0, so nothing
computed from it is committed: real-weight goldens live under target/ only.

--tiny draws a seeded random model with the real model's structure (patch 32,
output patch 64, nine quantiles, head dim 80, sandwich RMSNorm, QK norm,
PerDimScale, sequence RoPE, variate attention) but two heads, two layers and
d_model 160, with the FFN and residual-block hidden widths different from
d_model so a transposed weight cannot pass. Weights are re-drawn because a
zero-initialised PerDimScale and unit norm gains would let a missing gain pass.

Every file is flat in --out, named `<case>.<name>` (so one MANIFEST.json can account
for the directory), all little-endian .npy (bool arrays as float32 0/1, which ojas-io's npy reader
takes):

  in_target.npy / in_target_mask.npy           the decode() inputs
  in_po.npy / in_po_mask.npy                   past-only covariates (if any)
  in_pf.npy / in_pf_mask.npy                   past+future covariates (if any)
  fwd_values.npy  f32 [b, v, n, 32]            what decode() hands forward()
  fwd_masks.npy   0/1 [b, v, n, 32]
  fwd_patch_is_target.npy 0/1 [b, v, n]
  fwd_cpm.npy     0/1 [b, n]                   horizon CPM mask
  resblock_input.npy  f32 [b, v, n, 192]
  transformer_input.npy f32 [b, v, n, d]
  effective_patch_mask.npy 0/1 [b, v, n]       cumprod(all-masked) per row
  layer_{l}.npy   f32 [b, v, n, d]             residual stream after layer l
  revin_mean.npy / revin_std.npy f32 [b, v, n] running stats before CPM refine
  raw_logits.npy  f32 [b, v, n, 576]           the output head, before RevIN is undone
  logits.npy      f32 [b, v, n, 64, 9]         forward()'s final logits
  out.npy         f32 [b, v, horizon, 9]       decode()'s forecast
  case.json       horizon, shapes, NaN counts and what was run
  f64.<name>      (--tiny) float64 copies of fwd_values, revin_mean/std,
                  resblock_input, raw_logits and out from the same model and
                  inputs cast to f64
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import torch

PINNED = "918df02c9d366b2145bc5876135fa2131d3ed2a6"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEED = 20261118


def import_upstream(path):
    src = os.path.join(path, "src")
    if not os.path.isdir(os.path.join(src, "timesfm3", "torch")):
        sys.exit(f"no timesfm3/torch under {src}; run tools/timesfm3_ref/fetch_upstream.sh")
    sys.path.insert(0, src)
    from timesfm3.torch import model as model_lib

    return model_lib


def tiny_model(model_lib):
    model = model_lib.TimesFM3Torch(
        input_patch_len=32,
        output_patch_len=64,
        residual_block_config=dict(hidden_dims=176, output_dims=160, use_bias=False, activation="relu"),
        transformer_config=dict(
            num_layers=2,
            transformer=dict(
                model_dims=160, hidden_dims=224, num_heads=2, attention_norm="rms",
                feedforward_norm="rms", qk_norm="rms", use_bias=False, use_rope_seq=True,
                use_rope_var=False, ff_activation="relu", deterministic=True,
                use_memory_efficient_attention=True, use_sdpa=True,
            ),
        ),
    )
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("per_dim_scale"):
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            elif p.dim() == 1 and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[1]
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(fan_in))
    return model


def real_model(model_lib, model_dir):
    from safetensors.torch import load_file

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = model_lib.TimesFM3Torch(**cfg)
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=True), None
    del missing, unexpected
    return model, cfg


def series(rng, length, trend=0.0, scale=1.0, offset=0.0):
    t = np.arange(length, dtype=np.float64)
    period = rng.integers(7, 40)
    y = offset + scale * (np.sin(2 * np.pi * t / period) + 0.3 * rng.standard_normal(length)) + trend * t
    return y.astype(np.float32)


def cases(rng, big):
    out = []
    # a: one series, context not a multiple of 32 (left pad), horizon past one
    # output patch (stitching), no covariates, v == 1 fast path.
    out.append(dict(name="a_univariate", horizon=100,
                    target=np.stack([series(rng, 200, offset=5.0)])[None]))
    # b: two batch rows, two targets, one past-only and one past+future
    # covariate, gaps, a leading run of >= 32 masked points (a fully masked
    # leading patch) and a strong trend (linear detrending fires).
    ctx, hor = 256, 70
    tgt = np.stack([
        np.stack([series(rng, ctx, trend=0.5, scale=0.2), series(rng, ctx, scale=3.0, offset=-2.0)]),
        np.stack([series(rng, ctx, offset=100.0), series(rng, ctx, trend=-0.1)]),
    ])
    tmask = np.zeros_like(tgt, dtype=bool)
    tmask[0, 1, 50:58] = True
    tmask[1, :, :40] = True
    tmask[1, 0, 120:121] = True
    po = np.stack([series(rng, ctx)[None], series(rng, ctx, scale=0.1)[None]])
    pf = np.stack([series(rng, ctx + hor)[None], series(rng, ctx + hor, trend=0.2)[None]])
    out.append(dict(name="b_covariates", horizon=hor, target=tgt, target_mask=tmask, po=po, pf=pf))
    # c: one target with a constant context (sigma 0) and a horizon of exactly
    # one output patch.
    out.append(dict(name="c_constant", horizon=64,
                    target=np.full((1, 1, 96), 3.25, dtype=np.float32)))
    if big:
        out.append(dict(name="d_batch4_ctx1024", horizon=128,
                        target=np.stack([series(rng, 1024, offset=float(i)) for i in range(4)])[:, None]))
        out.append(dict(name="e_multivariate8_ctx512", horizon=96,
                        target=np.stack([series(rng, 512, scale=1.0 + i) for i in range(8)])[None]))
    return out


def u8(x):
    return x.detach().cpu().numpy().astype(np.float32)


def f32(x):
    # float32 for the fp32 pass; the f64 pass's save() keeps float64.
    x = x.detach().cpu().numpy()
    return x if x.dtype == np.float64 else x.astype(np.float32)


# What the f64 pass records: the host stages and the forecast, whose fp32
# error is dominated by conditioning (detrending a steep series, then dividing
# by a small sigma). Tests hold the port to "no further from f64 than
# upstream's own fp32 run", which a fixed tolerance cannot state.
F64_KEEP = ("fwd_values.npy", "revin_mean.npy", "revin_std.npy", "resblock_input.npy", "raw_logits.npy", "out.npy")


def run_case(model, case, out_dir, f64=False):
    prefix = case["name"] + (".f64." if f64 else ".")
    captured = {}
    orig_forward = model.forward

    def forward(inputs, freeze_after=None, patch_cpm_mask=None, return_aux_outputs=False):
        captured["values"] = inputs["values"].clone()
        captured["masks"] = inputs["masks"].clone()
        captured["patch_is_target"] = inputs["patch_is_target"].clone()
        captured["cpm"] = patch_cpm_mask.clone()
        captured["freeze_after"] = freeze_after
        return orig_forward(inputs, freeze_after=freeze_after, patch_cpm_mask=patch_cpm_mask,
                            return_aux_outputs=True)

    layers = []
    head = []
    hooks = [layer.register_forward_hook(lambda m, i, o: layers.append(o[0].detach().clone()))
             for layer in model.transformer_stack.layers]
    hooks.append(model.output_head.register_forward_hook(lambda m, i, o: head.append(o.detach().clone())))
    model.forward = forward
    try:
        def t(x):
            if x is None:
                return None
            x = torch.from_numpy(np.ascontiguousarray(x))
            return x.double() if f64 and x.is_floating_point() else x

        out, aux = model.decode(
            t(case["target"]), horizon=case["horizon"],
            past_only_covariates=t(case.get("po")),
            past_future_covariates=t(case.get("pf")),
            target_mask=t(case.get("target_mask")),
            return_aux_outputs=True,
        )
    finally:
        model.forward = orig_forward
        for h in hooks:
            h.remove()

    files = {}

    def save(name, arr):
        if f64:
            if name not in F64_KEEP:
                return
            arr = arr.astype(np.float64)
        np.save(os.path.join(out_dir, prefix + name), arr)
        files[name] = list(arr.shape)

    save("in_target.npy", case["target"].astype(np.float32))
    save("in_target_mask.npy", (case.get("target_mask") if case.get("target_mask") is not None
                                else np.zeros(case["target"].shape, dtype=bool)).astype(np.float32))
    for key in ("po", "pf"):
        if case.get(key) is not None:
            save(f"in_{key}.npy", case[key].astype(np.float32))
            save(f"in_{key}_mask.npy", np.zeros(case[key].shape, dtype=np.float32))
    save("fwd_values.npy", f32(captured["values"]))
    save("fwd_masks.npy", u8(captured["masks"]))
    save("fwd_patch_is_target.npy", u8(captured["patch_is_target"]))
    save("fwd_cpm.npy", u8(captured["cpm"]))
    save("resblock_input.npy", f32(aux["__call__:resblock_input"]))
    save("transformer_input.npy", f32(aux["__call__:transformer_input"]))
    # Recompute the mask forward() hands the stack, from the resblock input's
    # mask half, exactly as forward() does.
    p2 = aux["__call__:resblock_input"].shape[-1] // 2
    patch_mask = aux["__call__:resblock_input"][..., p2:].bool().all(dim=3)
    save("effective_patch_mask.npy", u8(torch.cumprod(patch_mask.int(), dim=2).bool()))
    for i, h in enumerate(layers):
        save(f"layer_{i}.npy", f32(h))
    mean, std = aux["revin_stats"]
    save("revin_mean.npy", f32(mean))
    save("revin_std.npy", f32(std))
    assert len(head) == 1, "the output head ran more than once"
    save("raw_logits.npy", f32(head[0]))
    save("logits.npy", f32(aux["logits"]))
    save("out.npy", f32(out))
    if f64:
        return {"files": files}
    meta = {
        "horizon": case["horizon"], "freeze_after": captured["freeze_after"], "files": files,
        "nan": {k: int(np.isnan(np.load(os.path.join(out_dir, prefix + k))).sum())
                for k in files if k.endswith(".npy") and not k.startswith(("in_", "fwd_m", "fwd_p", "fwd_c", "eff"))},
        "upstream": PINNED, "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, prefix + "case.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    files["case.json"] = None
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.join(ROOT, "target", "timesfm3_upstream", PINNED))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", action="store_true", help="write MANIFEST.json (needs the generator committed)")
    args = ap.parse_args()
    if args.tiny == bool(args.model_dir):
        sys.exit("pass exactly one of --tiny or --model-dir")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model_lib = import_upstream(args.upstream)
    os.makedirs(args.out, exist_ok=True)
    if args.tiny:
        model = tiny_model(model_lib)
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  os.path.join(args.out, "model.safetensors"))
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(model.to_dict(), f, indent=1, sort_keys=True)
    else:
        model, _ = real_model(model_lib, args.model_dir)
    model.eval()
    rng = np.random.default_rng(SEED)
    metas = {}
    with torch.no_grad():
        for case in cases(rng, big=not args.tiny):
            metas[case["name"]] = run_case(model, case, args.out)
            if args.tiny:
                f64_files = run_case(copy.deepcopy(model).double(), case, args.out, f64=True)["files"]
                metas[case["name"]]["files"].update({f"f64.{k}": v for k, v in f64_files.items()})
    if args.tiny and args.manifest:
        write_manifest(args.out, metas)
    print(json.dumps({k: v["nan"] for k, v in metas.items()}, indent=1))


def write_manifest(out_dir, metas):
    sys.path.insert(0, os.path.join(ROOT, "tools", "qwen35_ref"))
    import fixture_manifest as fm

    def role(name):
        if name.startswith("f64."):
            return "golden"
        if name.startswith("in_") or name.startswith("fwd_"):
            return "input"
        if name == "case.json":
            return "config"
        return "golden"

    files = {
        "model.safetensors": {"role": "model", "published": True,
                              "note": "the seeded tiny TimesFM 3 (2 layers, d 160, 2 heads of 80)"},
        "config.json": {"role": "config", "published": True, "note": "TimesFM3Torch.to_dict() of the tiny model"},
    }
    for case, meta in metas.items():
        for name in meta["files"]:
            files[f"{case}.{name}"] = {
                "role": role(name), "published": True,
                "note": {"golden": (f"float64 CPU forward of upstream timesfm3.torch at {PINNED} (the model and "
                                    "inputs cast to f64)" if name.startswith("f64.") else
                                    f"float32 CPU forward of upstream timesfm3.torch at {PINNED}"),
                         "config": "the case's horizon, shapes, NaN counts and versions",
                         "input": "decode() input or the forward() input decode() built"}[role(name)],
            }
    fm.write_manifest(
        out_dir, "tools/timesfm3_ref/make_reference.py",
        f"drawn: numpy default_rng({SEED}) draws the series and torch.Generator().manual_seed({SEED}) "
        "the weights; both are committed (model.safetensors and the in_*.npy files), so the draws are the inputs.",
        files, "produced")


if __name__ == "__main__":
    main()
