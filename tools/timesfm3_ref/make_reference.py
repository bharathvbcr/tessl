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
import dataclasses
import hashlib
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


def import_forecaster():
    # import_upstream() put the pinned checkout first on sys.path.
    from timesfm3.torch import timesfm3_forecaster

    return timesfm3_forecaster


def tiny_model(model_lib, use_mea=True, **switches):
    """The seeded tiny model; `switches` are TimesFM3Torch constructor switches
    (use_stitching, ...) and `use_mea` the transformer's
    use_memory_efficient_attention, for the off-branch variants."""
    model = model_lib.TimesFM3Torch(
        **switches,
        input_patch_len=32,
        output_patch_len=64,
        residual_block_config=dict(hidden_dims=176, output_dims=160, use_bias=False, activation="relu"),
        transformer_config=dict(
            num_layers=2,
            transformer=dict(
                model_dims=160, hidden_dims=224, num_heads=2, attention_norm="rms",
                feedforward_norm="rms", qk_norm="rms", use_bias=False, use_rope_seq=True,
                use_rope_var=False, ff_activation="relu", deterministic=True,
                use_memory_efficient_attention=use_mea, use_sdpa=True,
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


# Off-branch cases: one TimesFM 3 config switch off (the rest on), on the inputs
# of an existing case, so the committed all-on fixtures stay as they are. Each
# has its own <case>.config.json; a model with no variate attention also has
# its own <case>.model.safetensors (its state dict lacks the variate-attention
# tensors), every other variant loads the base model.safetensors.
OFF_CASES = (
    ("off_stitching", "a_univariate", dict(use_stitching=False)),
    ("off_detrending", "b_covariates", dict(use_linear_detrending=False)),
    ("off_cpm_refine", "b_covariates", dict(use_iterative_cpm_revin=False)),
    ("off_variate_attention", "b_covariates", dict(use_variate_attention=False)),
    ("off_mea_scale", "b_covariates", dict(use_mea=False)),
    ("off_all", "b_covariates", dict(use_stitching=False, use_linear_detrending=False,
                                     use_iterative_cpm_revin=False, use_variate_attention=False,
                                     use_mea=False)),
)
# Off and on must differ by more than fp32 noise (2e-5) on the case's forecast,
# relative to its peak, or the fixture could not tell the branches apart.
OFF_MIN_REL_DIFF = 1e-3


def off_model(model_lib, base, switches):
    """The tiny model with `switches` off and the base model's weights."""
    model = tiny_model(model_lib, **{k: False for k in switches})
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not missing, f"off model lacks {missing}"
    assert all("var_attn" in k or "var_attn_ln" in k for k in unexpected), unexpected
    return model.eval()


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


# Forecaster cases: upstream's TimesFM3Forecaster.predict_batch() on top of
# decode(), from their own generator so the decode cases above are unchanged.
FC_SEED = SEED + 1


def forecast_cases(rng, big):
    def s(n, **kw):
        return series(rng, n, **kw)

    out = []
    # fa: ragged univariate contexts over two groups of four (each group pads
    # to its own context), a length-1 context, a leading NaN run (trimmed),
    # interior and trailing NaNs (interpolated) and an all-NaN series (zeros).
    ctxs = [s(100, offset=5.0), s(37), s(300, trend=0.05), s(64, scale=2.0), s(1, offset=3.0), s(250)]
    ctxs[1][:10] = np.nan
    ctxs[2][40:47] = np.nan
    ctxs[2][-1] = np.nan
    ctxs[5][:] = np.nan
    out.append(dict(name="fa_ragged", horizon=50, contexts=ctxs))
    # fb: z-normalisation, symmetric averaging and make_positive together, with
    # an odd per_core_batch_size so a +/- pair straddles two groups; one series
    # non-negative with NaNs, one constant (sigma below the threshold).
    ctxs = [np.abs(s(128)), s(96, offset=-1.0), np.abs(s(200, offset=4.0)),
            np.full(80, 2.5, dtype=np.float32), s(160, trend=0.2)]
    ctxs[2][[5, 60]] = np.nan
    out.append(dict(name="fb_options", horizon=100, contexts=ctxs, per_core_batch_size=3,
                    use_znorm=True, use_symmetric_averaging=True, make_positive=True))
    # fc: two target variates per series; past-only covariates on some series
    # and past+future on others, so groups mix real covariates with masked
    # placeholders; an all-NaN variate (interpolates to zeros) and a NaN run
    # in one variate only (not trimmed: only all-variate NaN columns are).
    hor = 70
    ctxs, po, pf = [], [], []
    for i, n in enumerate([150, 90, 200, 64, 120]):
        ctxs.append(np.stack([s(n, offset=float(i)), s(n, scale=0.5)]))
        po.append(s(n)[None] if i in (0, 2) else None)
        pf.append(np.stack([s(n + hor), s(n + hor, trend=0.1)]) if i in (1, 2, 4) else None)
    ctxs[3][1, :] = np.nan
    ctxs[0][0, :20] = np.nan
    out.append(dict(name="fc_multivariate_cov", horizon=hor, contexts=ctxs, po=po, pf=pf))
    # fd: edge padding of the past+future covariates to the patch-rounded
    # horizon, and quantiles left unsorted.
    ctxs = [s(n) for n in (120, 75, 64)]
    pf = [s(n + hor)[None] for n in (120, 75, 64)]
    out.append(dict(name="fd_edge", horizon=hor, contexts=ctxs, pf=pf, padding_mode="edge",
                    sort_quantiles=False))
    # fe: contexts past the global context (15360) are truncated, past+future
    # covariates with them; a second series one patch short of it.
    n0, n1 = 15360 + 300, 15360 - 25
    ctxs = [np.stack([s(n0, offset=1.0), s(n0, scale=0.3)]), np.stack([s(n1), s(n1, trend=0.001)])]
    pf = [s(n0 + 64)[None], None]
    out.append(dict(name="fe_long", horizon=64, contexts=ctxs, pf=pf))
    # fg: the options on two target variates with covariates: past-only and
    # past+future covariates z-normalised and negated with the series (mixed
    # within groups), make_positive per variate (the first variate is
    # non-negative, the second is not) and on an all-NaN series (nothing
    # finite, so not clamped), edge padding, an odd group size.
    hor = 40
    ctxs, po, pf = [], [], []
    for i, n in enumerate([96, 130, 64, 75]):
        ctxs.append(np.stack([np.abs(s(n, offset=3.0)), s(n, offset=-0.5)]))
        po.append(s(n, scale=2.0)[None] if i in (0, 3) else None)
        pf.append(np.stack([s(n + hor), s(n + hor, trend=0.05)]) if i in (1, 3) else None)
    ctxs[2][:, :] = np.nan
    out.append(dict(name="fg_options_cov", horizon=hor, contexts=ctxs, po=po, pf=pf, per_core_batch_size=3,
                    use_znorm=True, use_symmetric_averaging=True, make_positive=True, padding_mode="edge"))
    if big:
        # ff: one group of four 8-variate series at the global context, about
        # 15.4k patch rows: more than tessl runs in one forward (MAX_ROWS), so
        # the port must split the group without changing any series.
        out.append(dict(name="ff_over_cap", horizon=128,
                        contexts=[np.stack([s(15360, offset=float(i + j)) for j in range(8)]) for i in range(4)]))
    return out


FC_OPTIONS = ("use_symmetric_averaging", "make_positive", "sort_quantiles", "use_znorm", "padding_mode")


def run_forecast_case(fc, case, out_dir):
    """Runs upstream's predict_batch() and records every decode() call it makes."""
    prefix = case["name"] + "."
    calls = []
    orig_decode = fc.model.decode

    def decode(**kw):
        out = orig_decode(**kw)
        calls.append((kw, out))
        return out

    fc.model.decode = decode
    orig_bs = fc.config.per_core_batch_size
    if "per_core_batch_size" in case:
        fc.config = dataclasses.replace(fc.config, per_core_batch_size=case["per_core_batch_size"])
    try:
        opts = {k: case[k] for k in FC_OPTIONS if k in case}
        results = list(fc.predict_batch(
            contexts=case["contexts"], horizon=case["horizon"],
            past_only_covariates=case.get("po"), past_future_covariates=case.get("pf"),
            return_quantiles=True, **opts))
    finally:
        fc.model.decode = orig_decode
        per_core = fc.config.per_core_batch_size
        fc.config = dataclasses.replace(fc.config, per_core_batch_size=orig_bs)

    files = {}

    def save(name, arr):
        np.save(os.path.join(out_dir, prefix + name), arr)
        files[name] = list(arr.shape)

    for i, ctx in enumerate(case["contexts"]):
        save(f"in_{i}.npy", np.asarray(ctx, dtype=np.float32))
        for key in ("po", "pf"):
            cov = (case.get(key) or [None] * len(case["contexts"]))[i]
            if cov is not None:
                save(f"in_{key}_{i}.npy", np.asarray(cov, dtype=np.float32))
        save(f"out_{i}.npy", results[i].quantiles.astype(np.float32))
        save(f"point_{i}.npy", results[i].forecast.astype(np.float32))
    call_meta = []
    for k, (kw, out) in enumerate(calls):
        c = f"call_{k}."
        save(c + "target.npy", kw["target"].numpy().astype(np.float32))
        save(c + "mask.npy", kw["mask"].numpy().astype(np.float32))
        for key, mkey in (("past_only_covariates", "past_only_mask"), ("past_future_covariates", "past_future_mask")):
            short = "po" if key.startswith("past_only") else "pf"
            if kw.get(key) is not None:
                save(c + f"{short}.npy", kw[key].numpy().astype(np.float32))
                save(c + f"{short}_mask.npy", kw[mkey].numpy().astype(np.float32))
        save(c + "out.npy", out.numpy().astype(np.float32))
        call_meta.append({"horizon": int(kw["horizon"])})
    meta = {
        "horizon": case["horizon"], "series": len(case["contexts"]), "per_core_batch_size": per_core,
        "options": {k: case.get(k) for k in FC_OPTIONS if k in case}, "calls": call_meta,
        "median_quantile_index": fc.config.median_quantile_index, "files": files,
        "nan": {k: int(np.isnan(np.load(os.path.join(out_dir, prefix + k))).sum())
                for k in files if k.startswith(("out_", "point_")) or k.endswith(".out.npy")},
        "upstream": PINNED, "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, prefix + "forecast.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    files["forecast.json"] = None
    return meta


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


def run_case(model, case, out_dir, f64=False, extra=None):
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
        **(extra or {}),
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
    extra_files = {}
    if args.tiny:
        off_metas, extra_files = run_off_cases(model_lib, model, args.out)
        metas.update(off_metas)
    forecaster_lib = import_forecaster()
    fc = forecaster_lib.TimesFM3Forecaster.from_pretrained(
        args.out if args.tiny else args.model_dir, device="cpu", local_files_only=True)
    with torch.no_grad():
        for case in forecast_cases(np.random.default_rng(FC_SEED), big=not args.tiny):
            metas[case["name"]] = run_forecast_case(fc, case, args.out)
    if args.tiny and args.manifest:
        write_manifest(args.out, metas, extra_files)
    print(json.dumps({k: v["nan"] for k, v in metas.items()}, indent=1))


def run_off_cases(model_lib, base, out_dir):
    """Runs every OFF_CASES entry and asserts its forecast differs from the
    all-on model's on the same inputs. Returns the cases' metas and the
    manifest entries of the config and weight files it wrote."""
    from safetensors.torch import save_file

    inputs = {c["name"]: c for c in cases(np.random.default_rng(SEED), big=False)}
    metas, entries = {}, {}
    for name, source, switches in OFF_CASES:
        case = {**inputs[source], "name": name}
        model = off_model(model_lib, base, switches)
        config_keys = ["use_memory_efficient_attention" if k == "use_mea" else k for k in switches]
        with open(os.path.join(out_dir, f"{name}.config.json"), "w") as f:
            json.dump(model.to_dict(), f, indent=1, sort_keys=True)
        entries[f"{name}.config.json"] = {
            "role": "config", "published": True,
            "note": f"TimesFM3Torch.to_dict() of the tiny model with {sorted(config_keys)} off, the rest on"}
        if not switches.get("use_variate_attention", True):
            save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                      os.path.join(out_dir, f"{name}.model.safetensors"))
            entries[f"{name}.model.safetensors"] = {
                "role": "model", "published": True,
                "note": "the tiny model's weights without the variate-attention tensors (a model with that switch off has none)"}
        meta = run_case(model, case, out_dir, extra={"off": sorted(config_keys), "inputs_of": source})
        meta_f64 = run_case(copy.deepcopy(model).double(), case, out_dir, f64=True)["files"]
        want = np.load(os.path.join(out_dir, f"{source}.out.npy"))
        got = np.load(os.path.join(out_dir, f"{name}.out.npy"))
        rel = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-6))
        print(f"{name}: max |off - on| / peak = {rel:.3e} on the inputs of {source}")
        assert rel > OFF_MIN_REL_DIFF, f"{name}: off and on agree to {rel:.3e}; the fixture cannot tell them apart"
        meta["files"].update({f"f64.{k}": v for k, v in meta_f64.items()})
        metas[name] = meta
    return metas, entries


def write_manifest(out_dir, metas, extra_files=None):
    sys.path.insert(0, os.path.join(ROOT, "tools", "qwen35_ref"))
    import fixture_manifest as fm

    def role(name):
        if name.startswith("f64."):
            return "golden"
        if name.startswith("in_") or name.startswith("fwd_"):
            return "input"
        if name.startswith("call_"):
            return "golden" if name.endswith(".out.npy") else "input"
        if name in ("case.json", "forecast.json"):
            return "config"
        return "golden"

    files = {
        **(extra_files or {}),
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
                                    f"TimesFM3Forecaster.predict_batch() output (float32 CPU) at {PINNED}"
                                    if name.startswith(("out_", "point_")) else
                                    f"float32 CPU forward of upstream timesfm3.torch at {PINNED}"),
                         "config": "the case's horizon, shapes, options, NaN counts and versions",
                         "input": ("a decode() call TimesFM3Forecaster.predict_batch() made" if name.startswith("call_")
                                   else "decode() or predict_batch() input, or the forward() input decode() built")}[role(name)],
            }
    # Every file's sha256, so tests/fixture_manifests.rs checks the bytes and
    # not only the names.
    for name, meta in files.items():
        with open(os.path.join(out_dir, name), "rb") as f:
            meta["sha256"] = hashlib.sha256(f.read()).hexdigest()
    fm.write_manifest(
        out_dir, "tools/timesfm3_ref/make_reference.py",
        f"drawn: numpy default_rng({SEED}) draws the series and torch.Generator().manual_seed({SEED}) "
        "the weights; both are committed (model.safetensors and the in_*.npy files), so the draws are the inputs.",
        files, "produced")


if __name__ == "__main__":
    main()
