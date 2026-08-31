#!/usr/bin/env python3
"""MLX + PyTorch MPS lanes of the GEMM shape sweep.

Protocol is pinned to `src/bin/bench_gemm_sweep.rs`: same shapes, same
warmup/iters, **synchronize every iteration** (mx.eval / torch.mps.synchronize)
so no lane hides dispatch cost behind pipelining, median over iters.

  python3 bench/gemm_sweep_mlx.py --iters 50 --warmup 10 --out results.json
  python3 bench/gemm_sweep_mlx.py --parity-dir /path/with/parity_*.npy
"""
import argparse, json, statistics, sys, time
import numpy as np

# (M, N, K, label) — must match SHAPES in bench_gemm_sweep.rs.
SHAPES = [
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
]

def shapes_from_env():
    """BENCH_SHAPES="MxNxK,..." — same override the Rust lane honours, so both
    lanes can be pointed at an identical diagnostic grid."""
    import os
    raw = os.environ.get("BENCH_SHAPES")
    if not raw:
        return None
    out = []
    for spec in filter(None, (x.strip() for x in raw.split(","))):
        m, n, k = (int(v) for v in spec.split("x"))
        out.append((m, n, k, f"{m}x{n}x{k}"))
    return out


def bench_mlx(shapes, warmup, iters, dtype="f32"):
    import mlx.core as mx
    mdt = {"f32": mx.float32, "bf16": mx.bfloat16}[dtype]
    rows = []
    for (m, n, k, label) in shapes:
        a = (mx.zeros((m, k), dtype=mdt) + 0.5).astype(mdt)
        b = (mx.zeros((k, n), dtype=mdt) + 0.5).astype(mdt)
        mx.eval(a, b)
        for _ in range(warmup):
            mx.eval(mx.matmul(a, b))
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(mx.matmul(a, b))
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(samples)
        rows.append(dict(shape=label, backend="mlx-" + dtype, runtime="mlx", m=m, n=n, k=k,
                         median_ms=med, best_ms=min(samples),
                         gflops=(2.0 * m * n * k) / (med * 1e6)))
        print(f"{label:<12} {'mlx-'+dtype:<14} M={m} N={n} K={k}  {med:8.3f} ms  "
              f"{rows[-1]['gflops']:8.1f} GFLOP/s", file=sys.stderr)
    return rows


def bench_torch(shapes, warmup, iters, dtype="f32"):
    import torch
    tdt = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    if not torch.backends.mps.is_available():
        print("torch MPS unavailable; skipping lane", file=sys.stderr)
        return []
    dev = torch.device("mps")
    rows = []
    for (m, n, k, label) in shapes:
        a = torch.full((m, k), 0.5, dtype=tdt, device=dev)
        b = torch.full((k, n), 0.5, dtype=tdt, device=dev)
        torch.mps.synchronize()
        for _ in range(warmup):
            torch.matmul(a, b)
            torch.mps.synchronize()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            torch.matmul(a, b)
            torch.mps.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(samples)
        rows.append(dict(shape=label, backend="mps-" + dtype, runtime="torch", m=m, n=n, k=k,
                         median_ms=med, best_ms=min(samples),
                         gflops=(2.0 * m * n * k) / (med * 1e6)))
        print(f"{label:<12} {'torch-mps-'+dtype:<14} M={m} N={n} K={k}  {med:8.3f} ms  "
              f"{rows[-1]['gflops']:8.1f} GFLOP/s", file=sys.stderr)
    return rows


def parity(parity_dir):
    """float64 numpy reference vs each dumped metal-native C. Returns rows."""
    import os
    a = np.load(os.path.join(parity_dir, "parity_a.npy"))
    b = np.load(os.path.join(parity_dir, "parity_b.npy"))
    ref = (a.astype(np.float64) @ b.astype(np.float64))
    scale = np.abs(ref).max()
    rows = []
    for fn in sorted(os.listdir(parity_dir)):
        if not fn.startswith("parity_c_"):
            continue
        name = fn[len("parity_c_"):-len(".npy")]
        c = np.load(os.path.join(parity_dir, fn)).astype(np.float64)
        err = np.abs(c - ref)
        rows.append(dict(lane=name, max_abs_err=float(err.max()),
                         max_rel_err=float(err.max() / scale),
                         mean_abs_err=float(err.mean())))
        print(f"parity {name:<12} max_abs={err.max():.3e}  "
              f"max_rel={err.max()/scale:.3e}", file=sys.stderr)
    # Same reference vs MLX and torch, so the tolerance bar is calibrated.
    try:
        import mlx.core as mx
        cm = np.array(mx.matmul(mx.array(a), mx.array(b)), copy=False).astype(np.float64)
        e = np.abs(cm - ref)
        rows.append(dict(lane="mlx", max_abs_err=float(e.max()),
                         max_rel_err=float(e.max() / scale), mean_abs_err=float(e.mean())))
        print(f"parity {'mlx':<12} max_abs={e.max():.3e}  max_rel={e.max()/scale:.3e}",
              file=sys.stderr)
    except Exception as exc:
        print(f"mlx parity skipped: {exc}", file=sys.stderr)
    try:
        import torch
        if torch.backends.mps.is_available():
            ct = torch.matmul(torch.from_numpy(a).to("mps"),
                              torch.from_numpy(b).to("mps")).cpu().numpy().astype(np.float64)
            e = np.abs(ct - ref)
            rows.append(dict(lane="torch-mps", max_abs_err=float(e.max()),
                             max_rel_err=float(e.max() / scale), mean_abs_err=float(e.mean())))
            print(f"parity {'torch-mps':<12} max_abs={e.max():.3e}  "
                  f"max_rel={e.max()/scale:.3e}", file=sys.stderr)
    except Exception as exc:
        print(f"torch parity skipped: {exc}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--parity-dir")
    ap.add_argument("--lanes", default="mlx,torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    args = ap.parse_args()

    if args.parity_dir:
        rows = parity(args.parity_dir)
        print(json.dumps(rows, indent=2))
        return

    lanes = args.lanes.split(",")
    shapes = shapes_from_env() or SHAPES
    rows = []
    for dt in args.dtypes.split(","):
        if "mlx" in lanes:
            rows += bench_mlx(shapes, args.warmup, args.iters, dt)
        if "torch" in lanes:
            rows += bench_torch(shapes, args.warmup, args.iters, dt)
    out = json.dumps(rows, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
