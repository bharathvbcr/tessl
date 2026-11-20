#!/usr/bin/env python3
"""Baseline lane for bench_paired: upstream's own PyTorch model on MPS (timesfm3.torch)
on the same workloads and synthetic series as src/bin/bench_timesfm3.rs.

A comparison lane only: nothing in tessl runs or needs Python. Reads upstream's
source by path (tools/timesfm3_ref/fetch_upstream.sh) and the checkpoint from
TIMESFM3_DIR. Prints the same JSON as bench_timesfm3: one object per workload
with ms_min (min of BENCH_ITERS) and check (mean median forecast over targets).

BENCH_WORKLOADS (BxVxC:H) times upstream's model.decode. BENCH_PREDICT (NxL:H,
NxVxL:H, each optionally +PO+PF) times upstream's TimesFM3Forecaster.predict_batch
(device mps, per_core_batch_size = BENCH_PREDICT_BATCH, default 4) on the ragged
synthetic series the module docs of src/bin/bench_timesfm3.rs define; as there,
setting only BENCH_PREDICT drops the default decode set. Its rows add mode,
series, per_core_batch_size, groups (one decode per group) and forwards (upstream's
count: one forward per group, so equal to groups here; tessl's `forwards` is the
count after it packs groups of equal shape, a different quantity). The stage
split (BENCH_STAGES) is tessl-only: this lane refuses it rather than report an
unsplit time as a split one.

    TIMESFM3_DIR=<snapshot> ~/.venvs/ml/bin/python bench/timesfm3_torch.py
"""
import json
import math
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PINNED = "918df02c9d366b2145bc5876135fa2131d3ed2a6"
# TIMESFM3_UPSTREAM: an upstream checkout at PINNED elsewhere (a worktree without its own target/).
UPSTREAM = os.environ.get("TIMESFM3_UPSTREAM") or os.path.join(ROOT, "target", "timesfm3_upstream", PINNED)
sys.path.insert(0, os.path.join(UPSTREAM, "src"))
DEFAULT_WORKLOADS = "1x1x512:128,16x1x2048:128,64x1x512:64,1x8x1024:96,4x4x2048:256"


def series(i, ctx):
    t = np.arange(ctx, dtype=np.float64)
    period = 7.0 + (i % 29)
    slope = 3.0 * t / ctx if i % 3 == 0 else 0.0
    return (np.sin(2 * math.pi * t / period) + slope + (i % 5)).astype(np.float32)


MAX_COUNT = 999_999
MAX_PREDICT_ELEMENTS = 1 << 28


def count(name, raw, minimum):
    """bench_timesfm3's parse_count: ASCII digits, no leading zero, minimum..MAX_COUNT."""
    if not (raw.isascii() and raw.isdigit()) or (raw != "0" and raw.startswith("0")):
        raise SystemExit(f"{name}={raw!r} is not a non-negative integer (ASCII digits, no sign or leading zero)")
    v = int(raw)
    if v < minimum or v > MAX_COUNT:
        raise SystemExit(f"{name}={v} is outside {minimum}..{MAX_COUNT}")
    return v


def env_count(name, default, minimum):
    raw = os.environ.get(name)
    return default if raw is None else count(name, raw, minimum)


def series_len(i, length):
    return length - (i * 61) % (length // 4 + 1)


def channels(base, rows, n):
    return np.stack([series(base + 7 * k, n) for k in range(rows)])


def parse_predict(w):
    """NxL:H or NxVxL:H, optionally +PO+PF; returns (name, n, v, length, horizon, po, pf)."""
    var = "BENCH_PREDICT"
    parts = w.split("+")
    if len(parts) not in (1, 3) or ":" not in parts[0]:
        raise SystemExit(f"{var}: workload {w!r}: expected NxL:H or NxVxL:H, optionally +PO+PF")
    shape, h = parts[0].split(":", 1)
    dims = [count(var, x, 1) for x in shape.split("x")]
    if len(dims) == 2:
        dims.insert(1, 1)
    if len(dims) != 3:
        raise SystemExit(f"{var}: workload {w!r}: the shape has two or three dimensions")
    n, v, length = dims
    horizon = count(var, h, 1)
    po, pf = (count(var, parts[1], 0), count(var, parts[2], 0)) if len(parts) == 3 else (0, 0)
    if len(parts) == 3 and po == 0 and pf == 0:
        raise SystemExit(f"{var}: workload {w!r}: +0+0 has no covariates; leave the suffix off")
    if (v + po + pf) * n * (length + horizon) > MAX_PREDICT_ELEMENTS:
        raise SystemExit(f"{var}: workload {w!r}: more than {MAX_PREDICT_ELEMENTS} input elements")
    return f"predict:{w}", n, v, length, horizon, po, pf


def predict_inputs(n, v, length, horizon, po, pf):
    """Per series: target ([len] for v == 1 else [v, len]), past-only, past+future (or None)."""
    ctxs, pos, pfs = [], [], []
    for i in range(n):
        ln = series_len(i, length)
        rows = np.stack([series(i * v + r, ln) for r in range(v)])
        ctxs.append(rows[0] if v == 1 else rows)
        pos.append(channels(i + 1000, po, ln) if po else None)
        pfs.append(channels(i + 2000, pf, ln + horizon) if pf else None)
    return ctxs, pos, pfs


def workloads():
    out = []
    env_w, env_p = os.environ.get("BENCH_WORKLOADS"), os.environ.get("BENCH_PREDICT")
    if env_w is None and env_p is not None:
        return out
    for w in (env_w if env_w is not None else DEFAULT_WORKLOADS).split(","):
        shape, h = w.split(":")
        b, v, c = (int(x) for x in shape.split("x"))
        x = np.stack([series(i, c) for i in range(b * v)]).reshape(b, v, c)
        out.append((w, b, v, c, int(h), x))
    return out


def main():
    import torch
    from timesfm3.torch.model import TimesFM3Torch
    from safetensors.torch import load_file

    d = os.environ["TIMESFM3_DIR"]
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    model = TimesFM3Torch(**cfg)
    model.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=True)
    dev = torch.device("mps")
    model = model.to(dev).eval()
    iters = env_count("BENCH_ITERS", 10, 1)
    warmup = env_count("BENCH_WARMUP", 3, 0)
    if os.environ.get("BENCH_STAGES", "0") != "0":
        raise SystemExit("BENCH_STAGES: the stage split is tessl-only; unset it for this lane")
    per_core = env_count("BENCH_PREDICT_BATCH", 4, 1)
    predicts = [parse_predict(w) for w in os.environ["BENCH_PREDICT"].split(",")] if "BENCH_PREDICT" in os.environ else []
    work = workloads()
    rows = []
    with torch.no_grad():
        for name, b, v, c, h, x in work:
            def run():
                out = model.decode(torch.from_numpy(x).to(dev), horizon=h)
                return out.cpu().numpy()
            first = run()
            assert np.isfinite(first).all(), name
            check = float(first[:, :v, :, first.shape[-1] // 2].mean())
            for _ in range(warmup):
                run()
            ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                run()
                ms.append((time.perf_counter() - t0) * 1e3)
            ms.sort()
            rows.append({"workload": name, "backend": "torch-mps-f32", "ms_min": ms[0], "ms_median": ms[len(ms) // 2],
                         "iters": iters, "check": check,
                         "peak_footprint_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
    if predicts:
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

        fc = TimesFM3Forecaster(checkpoint_path=d, device="mps", per_core_batch_size=per_core, local_files_only=True)
        for name, n, v, length, h, po, pf in predicts:
            ctxs, pos, pfs = predict_inputs(n, v, length, h, po, pf)

            def run():
                return list(fc.predict_batch(
                    contexts=ctxs, horizon=h,
                    past_only_covariates=pos if po else None,
                    past_future_covariates=pfs if pf else None))

            first = run()
            assert all(np.isfinite(o.forecast).all() for o in first), name
            # mean over series, variates and steps: every forecast has v * h points
            check = float(sum(np.sum(o.forecast, dtype=np.float64) for o in first) / (n * v * h))
            for _ in range(warmup):
                run()
            ms = []
            for _ in range(iters):
                t0 = time.perf_counter()
                run()
                ms.append((time.perf_counter() - t0) * 1e3)
            ms.sort()
            groups = -(-n // per_core)
            rows.append({"workload": name, "mode": "predict", "backend": "torch-mps-f32", "ms_min": ms[0],
                         "ms_median": ms[len(ms) // 2], "iters": iters, "series": n, "per_core_batch_size": per_core,
                         "groups": groups, "forwards": groups, "check": check,
                         "peak_footprint_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
