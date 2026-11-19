#!/usr/bin/env python3
"""Baseline lane for bench_paired: upstream's own MLX port (timesfm3.mlx, mx.compile on)
on the same workloads and synthetic series as src/bin/bench_timesfm3.rs.

A comparison lane only: nothing in tessl runs or needs Python. Reads upstream's
source by path (tools/timesfm3_ref/fetch_upstream.sh) and the checkpoint from
TIMESFM3_DIR. Prints the same JSON as bench_timesfm3: one object per workload
with ms_min (min of BENCH_ITERS) and check (mean median forecast over targets).

    TIMESFM3_DIR=<snapshot> ~/.venvs/ml/bin/python bench/timesfm3_mlx.py
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
sys.path.insert(0, os.path.join(ROOT, "target", "timesfm3_upstream", PINNED, "src"))
DEFAULT_WORKLOADS = "1x1x512:128,16x1x2048:128,64x1x512:64,1x8x1024:96,4x4x2048:256"


def series(i, ctx):
    t = np.arange(ctx, dtype=np.float64)
    period = 7.0 + (i % 29)
    slope = 3.0 * t / ctx if i % 3 == 0 else 0.0
    return (np.sin(2 * math.pi * t / period) + slope + (i % 5)).astype(np.float32)


def workloads():
    out = []
    for w in os.environ.get("BENCH_WORKLOADS", DEFAULT_WORKLOADS).split(","):
        shape, h = w.split(":")
        b, v, c = (int(x) for x in shape.split("x"))
        x = np.stack([series(i, c) for i in range(b * v)]).reshape(b, v, c)
        out.append((w, b, v, c, int(h), x))
    return out


def main():
    import mlx.core as mx
    from timesfm3.mlx.model import TimesFM3Mlx

    model = TimesFM3Mlx.from_pretrained(os.environ["TIMESFM3_DIR"], compile=True)
    iters = max(int(os.environ.get("BENCH_ITERS", "10")), 1)
    warmup = int(os.environ.get("BENCH_WARMUP", "3"))
    rows = []
    for name, b, v, c, h, x in workloads():
        def run():
            out = model.decode(mx.array(x), horizon=h)
            mx.eval(out)
            return np.array(out)
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
        rows.append({"workload": name, "backend": "mlx-f32", "ms_min": ms[0], "ms_median": ms[len(ms) // 2],
                     "iters": iters, "check": check,
                     "peak_footprint_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
