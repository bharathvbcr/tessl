#!/usr/bin/env python3
"""Alternate the tessl and PyTorch/MLX GEMM lanes so clock drift cancels.

Running the Rust sweep once and the Python sweep once -- minutes apart, in
separate processes -- puts all the drift between them into the ratio. Two such
runs of the identical benchmark disagreed by 16-21% on the torch lane alone,
which is larger than most of the differences being reported.

This alternates the lanes round by round and reports the median of the
per-round ratios, plus the observed spread, so a claim can be checked against
its own noise floor instead of resting on a single ordering.

Every reported geomean covers the **whole** requested ladder or the run fails.
A geomean over the three shapes that happened to report used to print exactly
like a geomean over all eight.

  python3 bench/paired_cross_runtime.py --lanes torch,mlx \\
      --rounds 6 --out bench/results/gemm_speed_ladder.json
"""
import argparse
import json
import math
import os
import subprocess
import sys

from benchmark_evidence import (
    DEFAULT_MAX_RATIO_SPREAD,
    DEFAULT_OUTER_ROUNDS,
    ENVIRONMENT_PROBE_POLICY,
    EvidenceOutput,
    clean_benchmark_env,
    embedded_metallib_path,
    enforce_ratio_spread,
    finish_provenance,
    geometric_mean,
    parse_ratio_spread_limit,
    requested_output_path,
    series_summary,
    start_required_provenance,
    validate_evidence_sample_counts,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
# Resolved from this file, not the caller's working directory. The sibling
# audit script already had to learn this.
RUST_BIN = os.path.join(CRATE, "target", "release", "bench_gemm_sweep")
PY_SWEEP = os.path.join(HERE, "gemm_sweep_mlx.py")

DEFAULT_SHAPES = ("512x512x512,1024x1024x1024,2048x2048x2048,4096x4096x4096,"
                  "2048x768x768,8192x3072x768,8192x768x3072,4096x4096x1024")

# (tessl lane, comparison lane, label). The comparison lane names which
# external runtime and dtype must be selected for the pair to be measurable.
PAIRS = [
    ("tensorops-f32", "mps-f32", "torch", "f32", "tessl f32-exact vs torch f32"),
    ("tensorops-tf32", "mps-f32", "torch", "f32", "tessl tf32 vs torch f32"),
    ("tensorops-bf16", "mps-bf16", "torch", "bf16", "tessl bf16 vs torch bf16"),
    ("tensorops-f32", "mlx-f32", "mlx", "f32", "tessl f32-exact vs MLX f32"),
    ("tensorops-tf32", "mlx-f32", "mlx", "f32", "tessl tf32 vs MLX f32"),
    ("tensorops-bf16", "mlx-bf16", "mlx", "bf16", "tessl bf16 vs MLX bf16"),
]


def missing_coverage(per_round, labels, mine, theirs):
    """(shape, lane) pairs absent from at least one round.

    Formed before any ratio is, because a shape missing from one side is a
    measurement that did not happen -- and a geomean that quietly drops it is
    indistinguishable from one that covered the whole ladder.
    """
    gaps = set()
    for s in labels:
        for rd in per_round:
            if (s, mine) not in rd[0]:
                gaps.add(f"{s}/{mine}")
            if (s, theirs) not in rd[1]:
                gaps.add(f"{s}/{theirs}")
    return sorted(gaps)


def summarize_comparison(per_round, labels, mine, theirs, label, max_ratio_spread):
    """Summarize paired ratios without taking independent best-of-run samples."""
    per_shape = []
    ratios_by_round = [[] for _ in per_round]
    for shape in labels:
        ratios = []
        for round_index, (tessl, other) in enumerate(per_round):
            ratio = tessl[(shape, mine)] / other[(shape, theirs)]
            ratios.append(ratio)
            ratios_by_round[round_index].append(ratio)
        summary = series_summary(ratios, label=f"{label}/{shape}")
        enforce_ratio_spread(
            summary, limit=max_ratio_spread, label=f"{label}/{shape}"
        )
        per_shape.append({"shape": shape, **summary})

    per_round_geomeans = [
        geometric_mean(values, label=f"{label}/round-{index + 1}")
        for index, values in enumerate(ratios_by_round)
    ]
    paired_geomean = series_summary(
        per_round_geomeans, label=f"{label}/paired-geomean"
    )
    enforce_ratio_spread(
        paired_geomean, limit=max_ratio_spread, label=f"{label}/paired-geomean"
    )
    medians = [row["median"] for row in per_shape]
    worst_single = min(per_shape, key=lambda row: row["min"])
    return {
        "label": label,
        "tessl_lane": mine,
        "other_lane": theirs,
        # Kept for consumers of the original schema. The explicit paired field
        # below is the canonical estimate because it preserves run pairing.
        "geomean": geometric_mean(medians, label=f"{label}/shape-medians"),
        "geomean_statistic": (
            "compatibility field: geometric mean of per-shape medians; "
            "paired_geomean.median is canonical"
        ),
        "paired_geomean": paired_geomean,
        "shapes_covered": len(per_shape),
        "shapes_requested": len(labels),
        "worst_shape": min(per_shape, key=lambda row: row["median"])["shape"],
        "worst_median": min(medians),
        "best_median": max(medians),
        "worst_single_round": worst_single["min"],
        "worst_single_round_shape": worst_single["shape"],
        "per_shape": per_shape,
    }


def summarize_throughput(per_round, labels):
    """Choose the best shape by its round median, never by a one-off maximum."""
    lane_names = sorted(
        {lane for rd in per_round for (_, lane) in rd[0]}
        | {lane for rd in per_round for (_, lane) in rd[1]}
    )
    summaries = []
    for lane in lane_names:
        shape_summaries = []
        for shape in labels:
            samples = []
            seen = False
            for tessl, other in per_round:
                in_tessl = (shape, lane) in tessl
                in_other = (shape, lane) in other
                if in_tessl and in_other:
                    raise ValueError(f"{shape}/{lane}: lane reported by both runtimes")
                if in_tessl or in_other:
                    seen = True
                    samples.append((tessl if in_tessl else other)[(shape, lane)])
                elif seen:
                    raise ValueError(f"{shape}/{lane}: missing from a later round")
            if seen:
                if len(samples) != len(per_round):
                    raise ValueError(
                        f"{shape}/{lane}: present in {len(samples)}/{len(per_round)} rounds"
                    )
                shape_summaries.append(
                    {"shape": shape, **series_summary(samples, label=f"{shape}/{lane}")}
                )
        if not shape_summaries:
            continue
        best = max(shape_summaries, key=lambda row: row["median"])
        summaries.append({
            "lane": lane,
            # Compatibility key: now the robust median at the best-median
            # shape, not an independent maximum over all rounds.
            "peak_gflops": best["median"],
            "shape": best["shape"],
            "statistic": "max_shape_of_round_medians",
            "round_summary": {
                key: value for key, value in best.items() if key != "shape"
            },
            "per_shape": shape_summaries,
        })
    return summaries


def raw_measurements(values):
    return [
        {"shape": shape, "lane": lane, "gflops": gflops}
        for (shape, lane), gflops in sorted(values.items())
    ]


def raw_round_record(round_number, execution_order, tessl, comparison):
    return {
        "round": round_number,
        "execution_order": list(execution_order),
        "tessl": raw_measurements(tessl),
        "comparison": raw_measurements(comparison),
    }


def _run(cmd, env, what):
    """Surface the child's stderr. `check=True` raised a CalledProcessError
    whose message omitted the one thing that explains the failure."""
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{what} exited {r.returncode}:\n{r.stderr.strip()}")
    try:
        rows = json.loads(r.stdout)
    except ValueError as exc:
        raise SystemExit(f"{what}: stdout is not JSON ({exc}):\n{r.stdout[:400]}")
    out = {}
    for x in rows:
        g = x["gflops"]
        if not (isinstance(g, (int, float)) and math.isfinite(g) and g > 0):
            raise SystemExit(f"{what}: {x['shape']}/{x['backend']} reported gflops={g}")
        key = (x["shape"], x["backend"])
        if key in out:
            raise SystemExit(f"{what}: duplicate measurement for {key[0]}/{key[1]}")
        out[key] = g
    return out


def run_rust(shapes, iters, warmup):
    env = clean_benchmark_env({
        "BENCH_SHAPES": shapes,
        "BENCH_ITERS": str(iters),
        "BENCH_WARMUP": str(warmup),
    })
    return _run([RUST_BIN], env, "bench_gemm_sweep")


def run_py(shapes, iters, warmup, lanes, dtypes):
    env = clean_benchmark_env({"BENCH_SHAPES": shapes})
    return _run([sys.executable, PY_SWEEP, "--lanes", lanes, "--dtypes", dtypes,
                 "--iters", str(iters), "--warmup", str(warmup)], env,
                "gemm_sweep_mlx.py")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--rounds", type=int, default=DEFAULT_OUTER_ROUNDS)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lanes", default="torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument(
        "--max-ratio-spread",
        type=parse_ratio_spread_limit,
        default=DEFAULT_MAX_RATIO_SPREAD,
        help="fail if any paired max/min ratio spread exceeds this bounded cap",
    )
    ap.add_argument("--out")
    output = EvidenceOutput(
        requested_output_path(sys.argv), driver_path=__file__, argv=sys.argv
    )
    output.begin()
    args = ap.parse_args()

    # Fewer than two complete AB/BA pairs cannot support both a balanced order
    # and a meaningful paired drift estimate.
    try:
        validate_evidence_sample_counts(args.rounds, args.iters)
    except ValueError as exc:
        raise SystemExit(f"insufficient evidence samples: {exc}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    labels = [s.strip() for s in args.shapes.split(",") if s.strip()]
    if not labels:
        raise SystemExit("--shapes selects no shape")
    for s in labels:
        d = s.split("x")
        if len(d) != 3 or not all(p.isdigit() and int(p) > 0 for p in d):
            raise SystemExit(f"--shapes entry must be MxNxK with positive dims, got {s!r}")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--shapes contains duplicates: {labels}")

    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    if len(set(lanes)) != len(lanes):
        raise SystemExit(f"--lanes contains duplicates: {lanes}")
    if len(set(dtypes)) != len(dtypes):
        raise SystemExit(f"--dtypes contains duplicates: {dtypes}")
    unknown_lanes = sorted(set(lanes) - {"torch", "mlx"})
    unknown_dtypes = sorted(set(dtypes) - {"f32", "bf16"})
    if unknown_lanes:
        raise SystemExit(f"--lanes contains unknown values: {unknown_lanes}")
    if unknown_dtypes:
        raise SystemExit(f"--dtypes contains unknown values: {unknown_dtypes}")
    pairs = [p for p in PAIRS if p[2] in lanes and p[3] in dtypes]
    if not pairs:
        raise SystemExit(
            f"--lanes {lanes} x --dtypes {dtypes} selects no comparison pair; "
            f"available: {sorted({(p[2], p[3]) for p in PAIRS})}")

    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"{RUST_BIN} not built. "
                         "cargo build --release --bin bench_gemm_sweep")
    try:
        metallib = embedded_metallib_path(RUST_BIN)
    except ValueError as exc:
        raise SystemExit(f"cannot establish exact shader provenance: {exc}")

    provenance = start_required_provenance(
        driver_path=__file__,
        argv=sys.argv,
        repo_scope=CRATE,
        executable_inputs={
            "rust_benchmark": RUST_BIN,
            "python_comparison": PY_SWEEP,
            "evidence_helper": os.path.join(HERE, "benchmark_evidence.py"),
            "metal_library": metallib,
        },
        benchmark_config={
            "rounds": args.rounds,
            "iters": args.iters,
            "warmup": args.warmup,
            "shapes": labels,
            "lanes": lanes,
            "dtypes": dtypes,
            "max_ratio_spread": args.max_ratio_spread,
            "child_processes": [
                {
                    "name": "tessl",
                    "argv": [RUST_BIN],
                    "environment": {
                        "BENCH_SHAPES": args.shapes,
                        "BENCH_ITERS": str(args.iters),
                        "BENCH_WARMUP": str(args.warmup),
                    },
                },
                {
                    "name": "comparison",
                    "argv": [
                        sys.executable,
                        PY_SWEEP,
                        "--lanes",
                        args.lanes,
                        "--dtypes",
                        args.dtypes,
                        "--iters",
                        str(args.iters),
                        "--warmup",
                        str(args.warmup),
                    ],
                    "environment": {"BENCH_SHAPES": args.shapes},
                },
            ],
        },
    )

    per_round = []
    raw_rounds = []
    for r in range(args.rounds):
        # Alternate which lane goes first so a warm-up asymmetry cannot favour
        # the same side every round.
        if r % 2 == 0:
            order = ["tessl", "comparison"]
            a = run_rust(args.shapes, args.iters, args.warmup)
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
        else:
            order = ["comparison", "tessl"]
            b = run_py(args.shapes, args.iters, args.warmup, args.lanes, args.dtypes)
            a = run_rust(args.shapes, args.iters, args.warmup)
        per_round.append((a, b))
        raw_rounds.append(raw_round_record(r + 1, order, a, b))
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    # Coverage is checked before any ratio is formed. A shape missing from one
    # side is a measurement that did not happen, and a geomean that quietly
    # drops it is indistinguishable from one that covered everything.
    for mine, theirs, _, _, label in pairs:
        missing = missing_coverage(per_round, labels, mine, theirs)
        if missing:
            raise SystemExit(
                f"{label}: {len(missing)} (shape, lane) pair(s) absent from at "
                f"least one round -- {missing[:6]}. Refusing to report a geomean "
                "over part of the ladder as if it covered all of it.")

    doc: dict = dict(
        schema_version=2,
        rounds=args.rounds,
        iters=args.iters,
        warmup=args.warmup,
        shapes=labels,
        lanes=lanes,
        dtypes=dtypes,
        max_ratio_spread=args.max_ratio_spread,
        methodology={
            "ratio_pairing": "same outer round",
            "execution_order": "exact AB/BA pairs; even outer-round count",
            "central_estimate": "median of paired per-round ratios",
            "canonical_aggregate": "median of per-round geometric means",
            "recorded_round_value": (
                "one child-process aggregate median GFLOP/s; inner iteration "
                "timings are not emitted by the child"
            ),
            "inner_iteration_samples_recorded": False,
            "drift_gate": "max/min paired ratio spread per shape and aggregate",
            "environment_probe_policy": ENVIRONMENT_PROBE_POLICY,
        },
        comparisons=[],
        peak_gflops=[],
        raw_rounds=raw_rounds,
        provenance=provenance,
    )

    for mine, theirs, _, _, label in pairs:
        try:
            comparison = summarize_comparison(
                per_round, labels, mine, theirs, label, args.max_ratio_spread
            )
        except ValueError as exc:
            raise SystemExit(f"unstable paired benchmark; refusing evidence artifact: {exc}")
        doc["comparisons"].append(comparison)

    # Validate every remaining summary before presenting any performance
    # number. A later failing lane must not leave earlier console rows looking
    # like a successful evidence report.
    try:
        doc["peak_gflops"] = summarize_throughput(per_round, labels)
    except ValueError as exc:
        raise SystemExit(f"incomplete throughput evidence: {exc}")

    for comparison in doc["comparisons"]:
        label = comparison["label"]
        rows = comparison["per_shape"]
        paired = comparison["paired_geomean"]

        print(f"\n{label}   ({args.rounds} alternating rounds, "
              f"{len(rows)}/{len(labels)} shapes)")
        print(f"  {'shape':<16}{'median':>9}{'min':>9}{'max':>9}")
        for row in rows:
            print(
                f"  {row['shape']:<16}{row['median']:>8.2f}x"
                f"{row['min']:>8.2f}x{row['max']:>8.2f}x"
            )
        print(
            f"  {'PAIRED GEOMEAN':<16}{paired['median']:>8.2f}x   "
            f"range {paired['min']:.2f}x..{paired['max']:.2f}x"
        )

    print(f"\n{'lane':<18}{'best median GFLOP/s':>22}  at")
    for row in doc["peak_gflops"]:
        print(f"{row['lane']:<18}{row['peak_gflops']:>22.0f}  {row['shape']}")

    finish_provenance(provenance)

    if args.out:
        output.publish(doc)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
