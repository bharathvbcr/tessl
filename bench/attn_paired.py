#!/usr/bin/env python3
"""Alternate the tessl and torch/MLX attention lanes so clock drift cancels.

Same protocol as `paired_cross_runtime.py`, for the attention configs. Drift is
15-20% on this hardware, so a paired sweep is what separates a real ratio from
a scheduling artifact -- and it stays necessary as the gap narrows, even where
today's ratios are far outside any drift band.

Every reported ratio covers the whole requested config set or the run fails.

  python3 bench/attn_paired.py --rounds 6 --out bench/results/attn_ladder.json
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
RUST_BIN = os.path.join(CRATE, "target", "release", "bench_flash_attn")
PY_LANE = os.path.join(HERE, "flash_attn_torch_mlx.py")

sys.path.insert(0, HERE)
from flash_attn_torch_mlx import BY_LABEL  # noqa: E402


def _run(cmd, env, what, key, expect_batch=None):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{what} exited {r.returncode}:\n{r.stderr.strip()}")
    try:
        rows = json.loads(r.stdout)
    except ValueError as exc:
        raise SystemExit(f"{what}: stdout is not JSON ({exc}):\n{r.stdout[:400]}")
    out = {}
    for x in rows:
        ms = x["median_ms"]
        if not (isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0):
            raise SystemExit(f"{what}: {x['cfg']} reported median_ms={ms}")
        # Both lanes echo the batch they ran. Comparing a lane measured at
        # batch=1 against one at batch=32 is an ~11x apples-to-oranges error
        # that looks like a plausible ratio, and this driver produced exactly
        # that when one side's env plumbing was missed.
        if expect_batch is not None:
            got = x.get("batched")
            if got is None:
                raise SystemExit(
                    f"{what}: rows carry no 'batched' field, so the driver cannot "
                    "confirm both lanes ran the same batching. Refusing to form a ratio.")
            if int(got) != int(expect_batch):
                raise SystemExit(
                    f"{what}: ran at batch={got}, driver asked for {expect_batch}. "
                    "A ratio across different batching is meaningless.")
        row_key = (x["cfg"], key(x))
        if row_key in out:
            raise SystemExit(f"{what}: duplicate measurement for {row_key[0]}/{row_key[1]}")
        out[row_key] = ms
    return out


def run_rust(cfgs, iters, warmup, batched):
    env = clean_benchmark_env({
        "BENCH_ATTN_CFGS": ",".join(cfgs),
        "BENCH_ITERS": str(iters),
        "BENCH_WARMUP": str(warmup),
        "BENCH_ATTN_BATCHED": str(batched),
    })
    # The binary emits one row per implementation, so the key has to be the
    # runtime tag; keying every row "tessl" silently kept only the last.
    return _run([RUST_BIN], env, "bench_flash_attn", lambda x: x["runtime"], batched)


def run_py(cfgs, iters, warmup, lanes, batched):
    return _run([sys.executable, PY_LANE, "--cfgs", ",".join(cfgs), "--lanes", lanes,
                 "--iters", str(iters), "--warmup", str(warmup)],
                clean_benchmark_env({"BENCH_ATTN_BATCHED": str(batched)}),
                "flash_attn_torch_mlx.py", lambda x: x["runtime"], batched)


def missing_coverage(per_round, cfgs, lanes, tessl_key="tessl"):
    """(cfg, lane) pairs absent from at least one round."""
    gaps = set()
    for c in cfgs:
        for a, b in per_round:
            if (c, tessl_key) not in a:
                gaps.add(f"{c}/{tessl_key}")
            for ln in lanes:
                if (c, ln) not in b:
                    gaps.add(f"{c}/{ln}")
    return sorted(gaps)


def validate_requested_configs(cfgs):
    """Reject selections that could duplicate or silently shrink an aggregate."""
    if not cfgs or any(not cfg for cfg in cfgs):
        raise ValueError("--cfgs selects an empty config name")
    if len(set(cfgs)) != len(cfgs):
        raise ValueError(f"--cfgs contains duplicates: {cfgs}")
    bad = [cfg for cfg in cfgs if cfg not in BY_LABEL]
    if bad:
        raise ValueError(f"--cfgs: unknown {bad}; expected from {list(BY_LABEL)}")


def summarize_comparison(
    per_round, cfgs, tessl_impl, other_lane, max_ratio_spread
):
    """Keep pairing intact and retain every absolute outer-round child median."""
    rows = []
    ratios_by_round = [[] for _ in per_round]
    for cfg in cfgs:
        ours = [tessl[(cfg, tessl_impl)] for tessl, _ in per_round]
        theirs = [other[(cfg, other_lane)] for _, other in per_round]
        ratios = []
        for round_index, (ours_ms, theirs_ms) in enumerate(zip(ours, theirs)):
            ratio = ours_ms / theirs_ms
            ratios.append(ratio)
            ratios_by_round[round_index].append(ratio)
        ratio_summary = series_summary(
            ratios, label=f"{tessl_impl}/{other_lane}/{cfg}"
        )
        enforce_ratio_spread(
            ratio_summary,
            limit=max_ratio_spread,
            label=f"{tessl_impl}/{other_lane}/{cfg}",
        )
        tessl_ms = series_summary(
            ours, label=f"{tessl_impl}/{other_lane}/{cfg}/tessl-ms"
        )
        other_ms = series_summary(
            theirs, label=f"{tessl_impl}/{other_lane}/{cfg}/other-ms"
        )
        rows.append({
            "cfg": cfg,
            **ratio_summary,
            "tessl_ms": tessl_ms,
            "other_ms": other_ms,
            # Original flat keys retained for existing readers.
            "tessl_median_ms": tessl_ms["median"],
            "other_median_ms": other_ms["median"],
        })

    per_round_geomeans = [
        geometric_mean(
            ratios, label=f"{tessl_impl}/{other_lane}/round-{round_index + 1}"
        )
        for round_index, ratios in enumerate(ratios_by_round)
    ]
    paired_geomean = series_summary(
        per_round_geomeans, label=f"{tessl_impl}/{other_lane}/paired-geomean"
    )
    enforce_ratio_spread(
        paired_geomean,
        limit=max_ratio_spread,
        label=f"{tessl_impl}/{other_lane}/paired-geomean",
    )
    medians = [row["median"] for row in rows]
    worst = max(rows, key=lambda row: row["median"])
    return {
        "lane": other_lane,
        "tessl_impl": tessl_impl,
        # Compatibility aggregate from the original schema.
        "geomean_tessl_over_other": geometric_mean(
            medians, label=f"{tessl_impl}/{other_lane}/config-medians"
        ),
        "geomean_statistic": (
            "compatibility field: geometric mean of per-config medians; "
            "paired_geomean_tessl_over_other.median is canonical"
        ),
        "paired_geomean_tessl_over_other": paired_geomean,
        "configs_covered": len(rows),
        "configs_requested": len(cfgs),
        "worst_config": worst["cfg"],
        "worst_median": worst["median"],
        "best_median": min(medians),
        "per_config": rows,
    }


def raw_measurements(values):
    return [
        {"cfg": cfg, "runtime": runtime, "median_ms": median_ms}
        for (cfg, runtime), median_ms in sorted(values.items())
    ]


def raw_round_record(round_number, execution_order, tessl, comparison):
    return {
        "round": round_number,
        "execution_order": list(execution_order),
        "tessl": raw_measurements(tessl),
        "comparison": raw_measurements(comparison),
    }


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--rounds", type=int, default=DEFAULT_OUTER_ROUNDS)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lanes", default="torch,mlx")
    ap.add_argument("--impl", default="tessl-decode",
                    choices=["tessl", "tessl-tiled", "tessl-decode", "tessl-rows"],
                    help="which tessl implementation to compare")
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--batched", type=int, default=1,
                    help="launches per submit. 1 matches mx.eval / "
                         "torch.mps.synchronize semantics; larger amortises the "
                         "command-buffer submit and isolates the kernel.")
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
    if args.batched < 1:
        raise SystemExit(f"--batched must be >= 1, got {args.batched}")

    try:
        validate_evidence_sample_counts(args.rounds, args.iters)
    except ValueError as exc:
        raise SystemExit(f"insufficient evidence samples: {exc}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    cfgs: list[str] = (
        [str(label) for label in BY_LABEL]
        if args.cfgs == "all"
        else [x.strip() for x in args.cfgs.split(",")]
    )
    try:
        validate_requested_configs(cfgs)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.impl == "tessl-decode":  # noqa: SIM102
        # FlashDecoding is defined only at Tq == 1. Narrowing is announced
        # rather than silent: a geomean over a quietly reduced config set is
        # the same defect this driver refuses elsewhere.
        decode_only = [c for c in cfgs if BY_LABEL[c]["tq"] == 1]
        dropped = [c for c in cfgs if c not in decode_only]
        if not decode_only:
            raise SystemExit(
                f"--impl tessl-decode needs at least one Tq == 1 config; "
                f"{cfgs} are all prefill")
        if dropped:
            print(f"--impl tessl-decode: skipping {len(dropped)} prefill config(s) "
                  f"{dropped} -- the decode path is Tq == 1 only",
                  file=sys.stderr)
        cfgs = decode_only

    sel = [x.strip() for x in args.lanes.split(",") if x.strip()]
    if not sel:
        raise SystemExit("--lanes selects no comparison lane")
    if len(set(sel)) != len(sel):
        raise SystemExit(f"--lanes contains duplicates: {sel}")
    bad = [x for x in sel if x not in ("torch", "mlx")]
    if bad:
        raise SystemExit(f"--lanes: unknown {bad}; expected torch and/or mlx")
    lane_keys = [{"torch": "torch-mps", "mlx": "mlx"}[x] for x in sel]

    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"{RUST_BIN} not built. "
                         "cargo build --release --bin bench_flash_attn")
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
            "python_comparison": PY_LANE,
            "evidence_helper": os.path.join(HERE, "benchmark_evidence.py"),
            "metal_library": metallib,
        },
        benchmark_config={
            "rounds": args.rounds,
            "iters": args.iters,
            "warmup": args.warmup,
            "batched": args.batched,
            "configs": cfgs,
            "lanes": lane_keys,
            "tessl_impl": args.impl,
            "max_ratio_spread": args.max_ratio_spread,
            "child_processes": [
                {
                    "name": "tessl",
                    "argv": [RUST_BIN],
                    "environment": {
                        "BENCH_ATTN_CFGS": ",".join(cfgs),
                        "BENCH_ITERS": str(args.iters),
                        "BENCH_WARMUP": str(args.warmup),
                        "BENCH_ATTN_BATCHED": str(args.batched),
                    },
                },
                {
                    "name": "comparison",
                    "argv": [
                        sys.executable,
                        PY_LANE,
                        "--cfgs",
                        ",".join(cfgs),
                        "--lanes",
                        args.lanes,
                        "--iters",
                        str(args.iters),
                        "--warmup",
                        str(args.warmup),
                    ],
                    "environment": {"BENCH_ATTN_BATCHED": str(args.batched)},
                },
            ],
        },
    )

    per_round = []
    raw_rounds = []
    for r in range(args.rounds):
        if r % 2 == 0:
            order = ["tessl", "comparison"]
            a = run_rust(cfgs, args.iters, args.warmup, args.batched)
            b = run_py(cfgs, args.iters, args.warmup, args.lanes, args.batched)
        else:
            order = ["comparison", "tessl"]
            b = run_py(cfgs, args.iters, args.warmup, args.lanes, args.batched)
            a = run_rust(cfgs, args.iters, args.warmup, args.batched)
        per_round.append((a, b))
        raw_rounds.append(raw_round_record(r + 1, order, a, b))
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    gaps = missing_coverage(per_round, cfgs, lane_keys, args.impl)
    if gaps:
        raise SystemExit(
            f"{len(gaps)} (config, lane) pair(s) absent from at least one round -- "
            f"{gaps[:6]}. Refusing to report ratios over part of the set.")

    doc: dict = dict(
        schema_version=2,
        rounds=args.rounds,
        iters=args.iters,
        warmup=args.warmup,
        batched=args.batched,
        cfgs=cfgs,
        lanes=lane_keys,
        tessl_impl=args.impl,
        max_ratio_spread=args.max_ratio_spread,
        methodology={
            "ratio_direction": ">1 means tessl is slower",
            "ratio_pairing": "same outer round",
            "execution_order": "exact AB/BA pairs; even outer-round count",
            "central_estimate": "median of paired per-round ratios",
            "canonical_aggregate": "median of per-round geometric means",
            "recorded_round_value": (
                "one child-process aggregate median in milliseconds; inner iteration "
                "timings are not emitted by the child"
            ),
            "inner_iteration_samples_recorded": False,
            "drift_gate": "max/min paired ratio spread per config and aggregate",
            "environment_probe_policy": ENVIRONMENT_PROBE_POLICY,
        },
        comparisons=[],
        raw_rounds=raw_rounds,
        provenance=provenance,
    )
    for ln in lane_keys:
        try:
            comparison = summarize_comparison(
                per_round, cfgs, args.impl, ln, args.max_ratio_spread
            )
        except ValueError as exc:
            raise SystemExit(f"unstable paired benchmark; refusing evidence artifact: {exc}")
        doc["comparisons"].append(comparison)

    # Only present numbers after every selected lane has passed its drift gate.
    for comparison in doc["comparisons"]:
        ln = comparison["lane"]
        rows = comparison["per_config"]
        paired = comparison["paired_geomean_tessl_over_other"]
        what = ("wall clock, submit-and-wait per call" if args.batched == 1
                else f"kernel only, {args.batched} launches per submit")
        print(f"\n{args.impl} / {ln}   {what}   ({args.rounds} alternating rounds, "
              f"{len(rows)}/{len(cfgs)} configs).  >1 means tessl is slower.")
        print(f"  {'config':<24}{'median':>9}{'min':>9}{'max':>9}"
              f"{'tessl ms':>12}{ln + ' ms':>14}")
        for row in rows:
            print(
                f"  {row['cfg']:<24}{row['median']:>8.1f}x"
                f"{row['min']:>8.1f}x{row['max']:>8.1f}x"
                f"{row['tessl_median_ms']:>12.4f}{row['other_median_ms']:>14.4f}"
            )
        print(
            f"  {'PAIRED GEOMEAN':<24}{paired['median']:>8.1f}x   "
            f"range {paired['min']:.1f}x..{paired['max']:.1f}x"
        )

    finish_provenance(provenance)

    if args.out:
        output.publish(doc)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
