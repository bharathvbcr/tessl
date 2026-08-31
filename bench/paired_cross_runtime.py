#!/usr/bin/env python3
"""Alternate the tessl and PyTorch/MLX GEMM lanes so clock drift cancels.

Running the Rust sweep once and the Python sweep once — minutes apart, in
separate processes — puts all the drift between them into the ratio. Two such
runs of the identical benchmark disagreed by 16-21% on the torch lane alone,
which is larger than most of the differences being reported.

This alternates the lanes round by round and reports the median of the
per-round ratios, plus the observed spread, so a claim can be checked against
its own noise floor instead of resting on a single ordering.
"""
import argparse, json, math, os, statistics, subprocess, sys

def run_rust(shapes_env, iters, warmup):
    env = dict(os.environ, BENCH_SHAPES=shapes_env, BENCH_ITERS=str(iters),
               BENCH_WARMUP=str(warmup))
    out = subprocess.run(["./target/release/bench_gemm_sweep"], env=env,
                         capture_output=True, text=True, check=True).stdout
    return {(x["shape"], x["backend"]): x["gflops"] for x in json.loads(out)}

def run_py(shapes_env, iters, warmup, lanes, dtypes):
    env = dict(os.environ, BENCH_SHAPES=shapes_env)
    out = subprocess.run(
        [sys.executable, "bench/gemm_sweep_mlx.py", "--lanes", lanes,
         "--dtypes", dtypes, "--iters", str(iters), "--warmup", str(warmup)],
        env=env, capture_output=True, text=True, check=True).stdout
    return {(x["shape"], x["backend"]): x["gflops"] for x in json.loads(out)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lanes", default="torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    ap.add_argument("--shapes", default=(
        "512x512x512,1024x1024x1024,2048x2048x2048,4096x4096x4096,"
        "2048x768x768,8192x3072x768,8192x768x3072,4096x4096x1024"))
    args = ap.parse_args()

    labels = args.shapes.split(",")
    per_round = []
    for r in range(args.rounds):
        # Alternate which lane goes first so a warm-up asymmetry cannot favour
        # the same side every round.
        if r % 2 == 0:
            a = run_rust(args.shapes, args.iters, args.warmup)
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
        else:
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
            a = run_rust(args.shapes, args.iters, args.warmup)
        per_round.append((a, b))
        print(f"round {r+1}/{args.rounds} done", file=sys.stderr)

    pairs = [("tensorops-f32", "mps-f32", "tessl f32-exact vs torch f32"),
             ("tensorops-f32relaxed", "mps-f32", "tessl tf32 vs torch f32"),
             ("tensorops-bf16", "mps-bf16", "tessl bf16 vs torch bf16"),
             ("tensorops-f32relaxed", "mlx-f32", "tessl tf32 vs MLX f32"),
             ("tensorops-bf16", "mlx-bf16", "tessl bf16 vs MLX bf16")]
    for mine, theirs, label in pairs:
        rows = []
        for s in labels:
            rr = [a[(s, mine)] / b[(s, theirs)]
                  for a, b in per_round if (s, mine) in a and (s, theirs) in b]
            if rr:
                rows.append((s, statistics.median(rr), min(rr), max(rr)))
        if not rows:
            continue
        print(f"\n{label}   ({args.rounds} alternating rounds)")
        print(f"  {'shape':<16}{'median':>9}{'min':>9}{'max':>9}")
        for s, med, lo, hi in rows:
            print(f"  {s:<16}{med:>8.2f}×{lo:>8.2f}×{hi:>8.2f}×")
        g = math.exp(sum(math.log(m) for _, m, _, _ in rows) / len(rows))
        print(f"  {'GEOMEAN of medians':<16}{g:>8.2f}×"
              f"   worst single round {min(lo for _, _, lo, _ in rows):.2f}×")

if __name__ == "__main__":
    main()
