#!/usr/bin/env python3
"""Alternate tessl's EmbeddingGemma 2 encoder and PyTorch MPS round by round.

    cargo build --release --bin bench_embedgemma2
    ~/.venvs/ml/bin/python bench/paired_embedgemma2.py --rounds 5 --out bench/results/embedgemma2_<machine>.json

Each round runs both lanes as fresh processes (the order alternates every
round, so a warm-up asymmetry cannot favour one side), each lane timing every
workload BENCH_ITERS times after BENCH_WARMUP. Per lane and workload the
round's number is the minimum (min-of-N); the reported ratio is the median of
the per-round ratios torch / tessl, with the lowest and highest per-round
ratio as the spread. A ratio whose spread straddles 1.0 is not a result.

Both lanes take the same token ids (no tokenization); see the two lanes'
docstrings for what each times.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, env):
    out = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed ({out.returncode}):\n{out.stderr[-4000:]}")
    return {r["workload"]: r for r in json.loads(out.stdout.strip().splitlines()[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workloads", default="1x16,64x32,32x256,8x1024,2x4096")
    ap.add_argument("--torch-dtypes", default="f32,bf16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2: a single round has no spread")

    env = dict(os.environ, BENCH_WORKLOADS=args.workloads, BENCH_ITERS=str(args.iters),
               BENCH_WARMUP=str(args.warmup))
    rust = [os.path.join(ROOT, "target", "release", "bench_embedgemma2")]
    lanes = {"tessl-f32": rust}
    for d in args.torch_dtypes.split(","):
        lanes[f"torch-mps-{d}"] = [sys.executable, os.path.join(HERE, "embedgemma2_torch.py"), "--dtype", d]

    names = list(lanes)
    rounds = []
    for r in range(args.rounds):
        order = names if r % 2 == 0 else names[::-1]
        rounds.append({name: run(lanes[name], env) for name in order})
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    workloads = [w.strip() for w in args.workloads.split(",")]
    summary = []
    for w in workloads:
        row = {"workload": w}
        for name in names:
            mins = [rd[name][w]["ms_min"] for rd in rounds]
            row[name] = {"ms_min_median": statistics.median(mins), "ms_min_all": mins}
        for name in names[1:]:
            ratios = [rd[name][w]["ms_min"] / rd["tessl-f32"][w]["ms_min"] for rd in rounds]
            row[f"{name}/tessl-f32"] = {"median": statistics.median(ratios), "lo": min(ratios), "hi": max(ratios)}
        summary.append(row)

    print(f"{'workload':>9} {'tessl f32 ms':>13} " + " ".join(f"{n + ' ms':>18} {'ratio [lo, hi]':>20}" for n in names[1:]))
    for row in summary:
        line = f"{row['workload']:>9} {row['tessl-f32']['ms_min_median']:>13.3f} "
        for n in names[1:]:
            q = row[f"{n}/tessl-f32"]
            line += f"{row[n]['ms_min_median']:>18.3f} {q['median']:>6.2f}x [{q['lo']:.2f}, {q['hi']:.2f}] "
        print(line)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rounds": args.rounds, "iters": args.iters, "warmup": args.warmup,
                       "summary": summary, "raw": rounds}, f, indent=1)
        print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
