#!/usr/bin/env python3
"""MLX + PyTorch MPS lanes of the GEMM shape sweep.

Protocol is pinned to `src/bin/bench_gemm_sweep.rs`: same shapes, same
warmup/iters, **synchronize every iteration** (mx.eval / torch.mps.synchronize)
so no lane hides dispatch cost behind pipelining, median over iters.

  python3 bench/gemm_sweep_mlx.py --iters 50 --warmup 10 --out results.json
  python3 bench/gemm_sweep_mlx.py --external-lanes /path/with/parity_manifest.json

This file is only the comparison runtimes. The numeric-parity *scorer* (the
float64 reference, the per-element error budget, the verdicts, the ladder) is
Rust: `gemm_parity score DIR` and `gemm_parity ladder`. `--external-lanes DIR`
runs MLX and torch-MPS on the operands a `bench_gemm_sweep --dump-parity DIR`
wrote and puts what they returned next to the dump, where the scorer reads it:

  DIR/external_lanes.json           per lane, per seed: out_dtype, or why it was skipped
  DIR/seed_NN/parity_c_<lane>.npy   the product, one file per scored (lane, seed)
"""
import argparse, json, os, statistics, sys, time
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


def _mlx_matmul(a, b, dtype):
    import mlx.core as mx
    am, bm = mx.array(a), mx.array(b)
    if dtype == "bf16":
        am, bm = am.astype(mx.bfloat16), bm.astype(mx.bfloat16)
    c = mx.matmul(am, bm)
    # Reported rather than assumed: MLX returns bf16 from a bf16 matmul, which
    # is a rounding step tessl's f32 destination does not pay.
    out = "bf16" if c.dtype == mx.bfloat16 else "f32"
    return np.array(c.astype(mx.float32), copy=False), out


def _torch_matmul(a, b, dtype):
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS unavailable")
    at, bt = torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps")
    if dtype == "bf16":
        at, bt = at.to(torch.bfloat16), bt.to(torch.bfloat16)
    c = torch.matmul(at, bt)
    out = "bf16" if c.dtype == torch.bfloat16 else "f32"
    return c.to(torch.float32).cpu().numpy(), out


# (lane name, callable, operand dtype). Each is run independently: a failure
# in one must not be attributed to another. An earlier version wrapped both MLX
# lanes in one try, so an f32 lane that had already scored also got a "skipped"
# row when the bf16 lane raised. The lane names are the keys of the scorer's
# LANE_PRECISION table (src/bin/common/parity.rs), which refuses any other.
EXTERNAL_LANES = [
    ("mlx-f32", _mlx_matmul, "f32"),
    ("mlx-bf16", _mlx_matmul, "bf16"),
    ("torch-mps-f32", _torch_matmul, "f32"),
    ("torch-mps-bf16", _torch_matmul, "bf16"),
]


def _dump_seeds(parity_dir):
    """The seed directories of a dump, from its manifest.

    Only the part of the manifest this script needs. The scorer validates the
    rest, and refuses a dump without a manifest, which `bench_gemm_sweep` writes
    last and only on success.
    """
    path = os.path.join(parity_dir, "parity_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no parity_manifest.json in {parity_dir}. It is written last and only "
            "on a clean run, so this dump is absent or partial. Re-run "
            "`bench_gemm_sweep --dump-parity`.")
    try:
        with open(path) as f:
            man = json.load(f)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"parity_manifest.json is unreadable: {exc}")
    seeds = man.get("seeds") if isinstance(man, dict) else None
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, str) for s in seeds):
        raise SystemExit("parity_manifest.json: 'seeds' must be a non-empty list of strings")
    return seeds


def write_external_lanes(parity_dir, external_lanes=None):
    """Run each comparison runtime on every seed of a dump and write the results
    next to it, for `gemm_parity score DIR`.

    `external_lanes` defaults to EXTERNAL_LANES (live MLX / torch-MPS). A caller
    that needs a result independent of which runtimes are installed, and of how
    the installed GPU rounds an f32 matmul, passes its own (lane, fn, dtype)
    list.

    A lane that raises on a seed is recorded as skipped, with its reason, on that
    seed; the scorer turns a lane skipped on every seed into a visible SKIPPED
    row and a lane skipped on only some into an error, never a partial average.
    The manifest `external_lanes.json` is written last, so a run that died
    halfway leaves none and the dump is scored with no comparison runtimes.
    """
    if external_lanes is None:
        external_lanes = EXTERNAL_LANES
    seeds = _dump_seeds(parity_dir)
    stale = os.path.join(parity_dir, "external_lanes.json")
    if os.path.exists(stale):
        os.remove(stale)
    # One list per entry, not per name: the scorer is the one that refuses a name
    # listed twice, so the writer must not merge two entries into one.
    results = [[] for _ in external_lanes]
    for seed in seeds:
        sd = os.path.join(parity_dir, seed)
        try:
            a = np.load(os.path.join(sd, "parity_a.npy"))
            b = np.load(os.path.join(sd, "parity_b.npy"))
        except OSError as exc:
            raise SystemExit(f"{seed}: operands unreadable: {exc}")
        for i, (lane, fn, dt) in enumerate(external_lanes):
            try:
                c, out_dtype = fn(a, b, dt)
            except Exception as exc:
                results[i].append(dict(seed=seed, skipped=str(exc)))
                print(f"external {lane:<16} {seed}: skipped -- {exc}", file=sys.stderr)
                continue
            np.save(os.path.join(sd, f"parity_c_{lane}.npy"), np.asarray(c))
            results[i].append(dict(seed=seed, out_dtype=out_dtype))
    doc = dict(lanes=[dict(lane=lane, dtype=dt, seeds=results[i])
                      for i, (lane, _, dt) in enumerate(external_lanes)])
    tmp = stale + ".partial"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, stale)
    print(f"external lanes: {len(external_lanes)} runtimes x {len(seeds)} seeds -> {parity_dir}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--external-lanes", metavar="DIR",
                    help="run MLX / torch-MPS on the operands of this parity dump and write "
                         "their outputs next to it, for `gemm_parity score DIR`")
    ap.add_argument("--lanes", default="mlx,torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    args = ap.parse_args()

    def emit(rows):
        out = json.dumps(rows, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out)
        else:
            print(out)

    if args.external_lanes:
        write_external_lanes(args.external_lanes)
        return

    # statistics.median() raises on an empty sample, and a zero-iteration sweep
    # is a request for a number nothing measured.
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")

    lanes = args.lanes.split(",")
    unknown = [x for x in lanes if x not in ("mlx", "torch")]
    if unknown:
        raise SystemExit(f"--lanes: unknown lane(s) {unknown}; expected mlx and/or torch")
    dtypes = args.dtypes.split(",")
    bad = [x for x in dtypes if x not in ("f32", "bf16")]
    if bad:
        raise SystemExit(f"--dtypes: unknown dtype(s) {bad}; expected f32 and/or bf16")

    shapes = shapes_from_env() or SHAPES
    rows = []
    for dt in dtypes:
        if "mlx" in lanes:
            rows += bench_mlx(shapes, args.warmup, args.iters, dt)
        if "torch" in lanes:
            rows += bench_torch(shapes, args.warmup, args.iters, dt)
    emit(rows)


if __name__ == "__main__":
    main()
