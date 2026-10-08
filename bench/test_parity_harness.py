#!/usr/bin/env python3
"""Adversarial tests for the parity scorer in `gemm_sweep_mlx.py`.

Most cases fabricate a dump on disk and assert the scorer's response. The two
CLI-contract sections launch benchmark binaries only when permitted by the
selected mode; `--pure` is runnable anywhere without touching a GPU.

The bar each case defends is the same one: a check that could not run must
never report the same result as a check that ran and passed. Every case below
corresponds to a way the previous harness returned a clean-looking report over
a dump it had not actually verified.

  python3 bench/test_parity_harness.py               # best effort GPU probes
  python3 bench/test_parity_harness.py --pure        # never launch a GPU binary
  python3 bench/test_parity_harness.py --require-gpu # a skipped GPU lane is fatal
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE_BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_BENCH)
from gemm_sweep_mlx import parity  # noqa: E402

M, N, K = 32, 24, 16
LANES = ["tensorops-f32", "simdgroup-f32", "tensorops-bf16", "tensorops-tf32"]
FAILURES = []


class Skip:
    """A section that could not run, distinct from a successful return."""

    def __init__(self, reason):
        self.reason = reason


class PartialSkip:
    """One sub-check was unavailable, although the rest of its section ran."""

    def __init__(self, name, reason):
        self.name = name
        self.reason = reason


class HarnessSummary:
    """Machine-readable section accounting for one harness invocation."""

    def __init__(self, mode):
        self.mode = mode
        self.ran = []
        self.skipped = []
        self.failed = []

    def skip(self, name, reason, required=False, partial=False):
        self.skipped.append(
            dict(name=name, reason=reason, required=required, partial=partial)
        )

    def as_dict(self, assertion_failures=0):
        required_skips = [x["name"] for x in self.skipped if x["required"]]
        return dict(
            mode=self.mode,
            ran=self.ran,
            skipped=self.skipped,
            failed=self.failed,
            counts=dict(
                ran=len(self.ran),
                skipped=len(self.skipped),
                failed=len(self.failed),
                assertion_failures=assertion_failures,
            ),
            required_gpu_skips=required_skips,
        )

    def verdict(self, assertion_failures=0):
        if (
            assertion_failures
            or self.failed
            or any(x["required"] for x in self.skipped)
        ):
            return "FAIL"
        if self.skipped:
            return "PASS_WITH_SKIPS"
        return "PASS"

    def exit_code(self, assertion_failures=0):
        return 1 if self.verdict(assertion_failures) == "FAIL" else 0


def run_section(summary, name, fn, required=False, failures=None):
    """Run one section and record exactly one ran/skipped/failed outcome."""
    failures = FAILURES if failures is None else failures
    before = len(failures)
    try:
        outcome = fn()
    except SystemExit as exc:
        failures.append(f"{name}: exited unexpectedly -- {exc}")
        print(f"  FAIL  {name}: unexpected exit: {exc}")
        outcome = None
    except Exception as exc:  # keep the final accounting even on a broken contract
        failures.append(f"{name}: raised {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        outcome = None

    if isinstance(outcome, Skip):
        if len(failures) != before:
            failures.append(f"{name}: recorded both failure and skip")
            summary.failed.append(name)
        else:
            summary.skip(name, outcome.reason, required=required)
        return
    if isinstance(outcome, PartialSkip):
        summary.skip(
            f"{name}: {outcome.name}",
            outcome.reason,
            required=False,
            partial=True,
        )
        if len(failures) == before:
            summary.ran.append(name)
        else:
            summary.failed.append(name)
        return
    if outcome is not None:
        failures.append(f"{name}: returned unsupported outcome {outcome!r}")
    if len(failures) == before:
        summary.ran.append(name)
    else:
        summary.failed.append(name)


def run_gpu_section(summary, mode, name, fn):
    """Apply the selected GPU policy without accidentally calling in pure mode."""
    if mode == "pure":
        reason = "disabled by --pure; GPU contract intentionally unverified"
        print(f"\n-- {name} --\n  SKIP  {reason}")
        summary.skip(name, reason, required=False)
        return
    run_section(summary, name, fn, required=(mode == "require-gpu"))


def build_dump(root, seeds=2, lanes=LANES, err_scale=None):
    """A structurally valid dump. `err_scale[i]` perturbs seed i's results so a
    test can pin down which seed the aggregate reports."""
    os.makedirs(root, exist_ok=True)
    names = []
    for i in range(seeds):
        sd = os.path.join(root, f"seed_{i:02}")
        os.makedirs(sd, exist_ok=True)
        names.append(f"seed_{i:02}")
        rng = np.random.default_rng(1000 + i)
        a = rng.uniform(-1, 1, (M, K)).astype(np.float32)
        b = rng.uniform(-1, 1, (K, N)).astype(np.float32)
        np.save(os.path.join(sd, "parity_a.npy"), a)
        np.save(os.path.join(sd, "parity_b.npy"), b)
        c = (a.astype(np.float64) @ b.astype(np.float64))
        if err_scale is not None:
            # Relative to the reference, so the perturbation scales with each
            # element's own error budget instead of blowing it on small ones.
            c = c * (1.0 + err_scale[i])
        for lane in lanes:
            np.save(os.path.join(sd, f"parity_c_{lane}.npy"), c.astype(np.float32))
    with open(os.path.join(root, "parity_manifest.json"), "w") as f:
        json.dump(dict(shape="synthetic", dist="uniform", m=M, n=N, k=K,
                       lanes=lanes, seeds=names), f)
    return root


def edit_manifest(root, **kw):
    path = os.path.join(root, "parity_manifest.json")
    with open(path) as f:
        man = json.load(f)
    man.update(kw)
    with open(path, "w") as f:
        json.dump(man, f)


def check(name, fn, expect_reject, needle=None):
    """`expect_reject` -- the scorer must refuse. Otherwise it must succeed."""
    try:
        rows = fn()
    except SystemExit as exc:
        if not expect_reject:
            FAILURES.append(f"{name}: rejected a valid dump -- {exc}")
            print(f"  FAIL  {name}: unexpected rejection: {exc}")
            return None
        msg = str(exc)
        if needle and needle.lower() not in msg.lower():
            FAILURES.append(f"{name}: rejected, but message lacks {needle!r}: {msg}")
            print(f"  FAIL  {name}: message lacks {needle!r}")
            return None
        print(f"  ok    {name}: rejected -- {msg.splitlines()[0][:88]}")
        return None
    except Exception as exc:  # a raw traceback is not a usable diagnosis
        FAILURES.append(f"{name}: raised {type(exc).__name__} instead of SystemExit: {exc}")
        print(f"  FAIL  {name}: raw {type(exc).__name__}: {exc}")
        return None
    if expect_reject:
        FAILURES.append(f"{name}: ACCEPTED a dump it should have refused")
        print(f"  FAIL  {name}: accepted")
        return None
    print(f"  ok    {name}: accepted")
    return rows


# (label, argv tail, env overrides, expected substring in the error).
# Every one of these either panicked or silently did the wrong thing before.
CLI_CASES = [
    ("--dump-parity with no directory", ["--dump-parity"], {}, "requires a directory"),
    ("--dump-parity followed by a flag", ["--dump-parity", "--x"], {}, "requires a directory"),
    ("BENCH_ITERS=0", [], {"BENCH_ITERS": "0"}, "below the minimum"),
    ("BENCH_ITERS=abc", [], {"BENCH_ITERS": "abc"}, "not a non-negative integer"),
    ("BENCH_ITERS=-1", [], {"BENCH_ITERS": "-1"}, "not a non-negative integer"),
    ("BENCH_WARMUP=xyz", [], {"BENCH_WARMUP": "xyz"}, "not a non-negative integer"),
    ("BENCH_SHAPES=garbage", [], {"BENCH_SHAPES": "garbage"}, "must be MxNxK"),
    ("BENCH_SHAPES=64x64", [], {"BENCH_SHAPES": "64x64"}, "must be MxNxK"),
    ("BENCH_SHAPES with a zero dim", [], {"BENCH_SHAPES": "0x64x64"}, "zero dimension"),
    ("BENCH_SHAPES=64xAx64", [], {"BENCH_SHAPES": "64xAx64"}, "not an integer"),
    ("BENCH_PARITY_SEEDS=0", ["--dump-parity", "{tmp}"], {"BENCH_PARITY_SEEDS": "0"},
     "below the minimum"),
    ("BENCH_PARITY_SEEDS=abc", ["--dump-parity", "{tmp}"], {"BENCH_PARITY_SEEDS": "abc"},
     "not a non-negative integer"),
    ("BENCH_SHAPES alongside --dump-parity", ["--dump-parity", "{tmp}"],
     {"BENCH_SHAPES": "64x64x64"}, "does not apply"),
    ("BENCH_PARITY_SHAPE=nonsense", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": "nonsense"}, "not a ladder label"),
    ("BENCH_PARITY_SHAPE=512x512x512 (dims, not a label)", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": "512x512x512"}, "not a ladder label"),
    ("BENCH_PARITY_SHAPE empty", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_SHAPE": ""}, "not a ladder label"),
    ("BENCH_PARITY_DIST=gaussian", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_DIST": "gaussian"}, "not a distribution"),
    ("BENCH_PARITY_DIST empty", ["--dump-parity", "{tmp}"],
     {"BENCH_PARITY_DIST": ""}, "not a distribution"),
]


def cli_contract():
    """Argument and environment handling in `bench_gemm_sweep`.

    Needs the built binary and a working Metal runtime (main() initialises the
    GPU before it parses anything), so this section announces a skip rather
    than passing vacuously where neither is available.
    """
    import subprocess
    print("\n-- bench_gemm_sweep argument / environment contract --")
    binary = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "target", "release", "bench_gemm_sweep")
    if not os.path.exists(binary):
        reason = (f"{binary} not built "
                  "(cargo build --release --bin bench_gemm_sweep); CLI contract unverified")
        print(f"  SKIP  {reason}")
        return Skip(reason)
    probe = subprocess.run([binary, "--dump-parity"], capture_output=True, text=True)
    if "panicked" not in probe.stderr and "requires a directory" not in probe.stderr:
        reason = f"no usable Metal runtime: {probe.stderr.strip()[:90]}"
        print(f"  SKIP  {reason}")
        return Skip(reason)

    tmp = tempfile.mkdtemp(prefix="parity-cli-")
    try:
        for label, argv, env, needle in CLI_CASES:
            argv = [x.replace("{tmp}", tmp) for x in argv]
            e = dict(os.environ, **env)
            # A stale value from the caller's shell would silently change the case.
            for var in ("BENCH_ITERS", "BENCH_WARMUP", "BENCH_SHAPES",
                        "BENCH_PARITY_SEEDS", "BENCH_PARITY_SHAPE",
                        "BENCH_PARITY_DIST"):
                if var not in env:
                    e.pop(var, None)
            r = subprocess.run([binary] + argv, env=e, capture_output=True, text=True)
            out = r.stderr + r.stdout
            if "panicked at" in out or r.returncode < 0:
                FAILURES.append(f"CLI {label}: panicked instead of erroring cleanly")
                print(f"  FAIL  {label}: panic")
            elif r.returncode == 0:
                FAILURES.append(f"CLI {label}: exited 0, accepting bad input")
                print(f"  FAIL  {label}: accepted (exit 0)")
            elif needle.lower() not in out.lower():
                FAILURES.append(f"CLI {label}: error lacks {needle!r}: {out.strip()[:120]}")
                print(f"  FAIL  {label}: message lacks {needle!r}")
            else:
                print(f"  ok    {label}: exit {r.returncode}, names the cause")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ladder_merge():
    """`parity_ladder.merge` -- the cross-shape aggregation.

    A per-shape sweep introduces a second way to under-report: a lane that
    covers only part of the ladder, averaged into one number that reads as a
    full sweep. Same rule as the seed axis, one level up.
    """
    from parity_ladder import merge
    print("\n-- ladder merge (cross-shape x distribution aggregation) --")
    cells = [("square_512", "uniform"), ("square_1024", "uniform"),
             ("mlp_down", "heavy_tail")]
    labels = [c[0] for c in cells]

    def row(lane, cell, rel, dtype="f32"):
        return dict(lane=lane, shape=cell[0], dist=cell[1], out_dtype=dtype, seeds=2,
                    max_rel_err=rel, median_rel_err=rel, min_rel_err=rel,
                    max_abs_err=rel, mean_abs_err=rel, per_seed_rel_err=[rel],
                    max_elem_rel_err=rel, max_higham_ratio=rel, per_seed_higham=[rel])

    full = [row("L", c, r) for c, r in zip(cells, [1e-6, 5e-6, 9e-6])]
    out = check("full grid merges", lambda: merge(full, cells), expect_reject=False)

    check("bare labels instead of (shape, dist) pairs",
          lambda: merge(full, labels), True, "would be split into characters")
    if out:
        got = out[0]
        if got["worst_shape"] != "mlp_down" or abs(got["worst_rel_err"] - 9e-6) > 1e-18:
            FAILURES.append(f"merge picked {got['worst_shape']} @ {got['worst_rel_err']}")
            print(f"  FAIL  worst is {got['worst_shape']} @ {got['worst_rel_err']:.2e}")
        else:
            print(f"  ok    worst across ladder = {got['worst_rel_err']:.1e} "
                  f"at {got['worst_shape']}")

    check("lane absent from one cell", lambda: merge(full[:2], cells),
          True, "whole grid")

    partial = full[:2] + [dict(lane="L", shape="mlp_down", dist="heavy_tail",
                               skipped="mlx missing")]
    check("lane scored on some cells, skipped on others",
          lambda: merge(partial, cells), True, "whole grid")

    allskip = [dict(lane="L", shape=c[0], dist=c[1], skipped="mlx missing")
               for c in cells]
    out = check("lane skipped on the whole grid", lambda: merge(allskip, cells),
                expect_reject=False)
    if out and "skipped" not in out[0]:
        FAILURES.append("an all-skipped lane was reported as scored")
        print("  FAIL  all-skipped lane reported as scored")
    elif out:
        print("  ok    all-skipped lane reported as skipped, not scored")

    mixed = [row("L", cells[0], 1e-6, "f32"), row("L", cells[1], 1e-6, "bf16"),
             row("L", cells[2], 1e-6, "f32")]
    check("output dtype varies across cells", lambda: merge(mixed, cells),
          True, "dtype varies")


def speed_coverage():
    """`paired_cross_runtime.missing_coverage` -- the timing sweep's version of
    the same rule. A geomean over three of eight shapes used to print exactly
    like a geomean over eight."""
    import types
    import paired_cross_runtime
    from paired_cross_runtime import (
        missing_coverage,
        raw_measurements,
        raw_round_record,
        summarize_comparison,
        summarize_throughput,
    )
    print("\n-- paired timing sweep: ladder coverage, drift, and robust summaries --")
    labels = ["512x512x512", "1024x1024x1024", "2048x2048x2048"]
    full_a = {(s, "tensorops-f32"): 1.0 for s in labels}
    full_b = {(s, "mps-f32"): 1.0 for s in labels}

    rounds = [(dict(full_a), dict(full_b)) for _ in range(3)]
    gaps = missing_coverage(rounds, labels, "tensorops-f32", "mps-f32")
    if gaps:
        FAILURES.append(f"complete rounds reported gaps: {gaps}")
        print(f"  FAIL  complete rounds reported gaps: {gaps}")
    else:
        print("  ok    complete coverage reports no gaps")

    # torch MPS dropping out mid-sweep: one round returns nothing for a shape.
    holed = [(dict(full_a), dict(full_b)) for _ in range(3)]
    del holed[1][1][("1024x1024x1024", "mps-f32")]
    gaps = missing_coverage(holed, labels, "tensorops-f32", "mps-f32")
    if gaps != ["1024x1024x1024/mps-f32"]:
        FAILURES.append(f"a hole in one round was not caught: {gaps}")
        print(f"  FAIL  hole in one round not caught: {gaps}")
    else:
        print(f"  ok    a shape missing from one round is caught: {gaps}")

    # The comparison runtime absent entirely -- every shape, every round.
    gaps = missing_coverage([(dict(full_a), {}) for _ in range(3)], labels,
                            "tensorops-f32", "mps-f32")
    if len(gaps) != len(labels):
        FAILURES.append(f"a fully absent lane was not fully reported: {gaps}")
        print(f"  FAIL  fully absent lane: {gaps}")
    else:
        print(f"  ok    a fully absent comparison lane names all {len(gaps)} shapes")

    # Paired aggregation must survive through the cross-shape summary. With
    # crossed ratios, each actual round's geometric mean is sqrt(1.25), while
    # the geomean of the two independently-computed medians is 1.125.
    crossed = [
        (
            {(labels[0], "tensorops-f32"): 1.0,
             (labels[1], "tensorops-f32"): 1.25},
            {(labels[0], "mps-f32"): 1.0, (labels[1], "mps-f32"): 1.0},
        ),
        (
            {(labels[0], "tensorops-f32"): 1.25,
             (labels[1], "tensorops-f32"): 1.0},
            {(labels[0], "mps-f32"): 1.0, (labels[1], "mps-f32"): 1.0},
        ),
    ]
    result = summarize_comparison(
        crossed, labels[:2], "tensorops-f32", "mps-f32", "synthetic", 1.25
    )
    paired = result["paired_geomean"]
    expected = 1.25 ** 0.5
    if (abs(paired["median"] - expected) < 1e-12
            and all(
                abs(value - expected) < 1e-12
                for value in paired["outer_round_values"]
            )):
        print("  ok    aggregate is the median of paired per-round geomeans")
    else:
        FAILURES.append(f"paired aggregate lost pairing: {paired}")
        print(f"  FAIL  paired aggregate: {paired}")

    unstable = [
        ({(labels[0], "tensorops-f32"): 1.0}, {(labels[0], "mps-f32"): 1.0}),
        ({(labels[0], "tensorops-f32"): 2.0}, {(labels[0], "mps-f32"): 1.0}),
    ]
    try:
        summarize_comparison(
            unstable, labels[:1], "tensorops-f32", "mps-f32", "unstable", 1.25
        )
    except ValueError as exc:
        if "exceeds bounded limit" in str(exc):
            print("  ok    excessive paired ratio drift fails closed")
        else:
            FAILURES.append(f"ratio drift raised wrong error: {exc}")
            print(f"  FAIL  ratio drift raised wrong error: {exc}")
    else:
        FAILURES.append("2x paired ratio spread passed a 1.25x drift gate")
        print("  FAIL  excessive paired ratio drift was accepted")

    # A single 100x outlier must not become the reported peak. Shape B's
    # repeatable 2 GFLOP/s beats shape A's median of 1 GFLOP/s.
    throughput_rounds = []
    for a_value, b_value in ((1.0, 2.0), (1.0, 2.0), (100.0, 2.0)):
        throughput_rounds.append(({
            (labels[0], "tensorops-f32"): a_value,
            (labels[1], "tensorops-f32"): b_value,
        }, {}))
    throughput = summarize_throughput(throughput_rounds, labels[:2])
    if (len(throughput) == 1 and throughput[0]["shape"] == labels[1]
            and throughput[0]["peak_gflops"] == 2.0
            and throughput[0]["statistic"] == "max_shape_of_round_medians"):
        print("  ok    throughput peak is selected from per-shape medians, not maxima")
    else:
        FAILURES.append(f"throughput summary selected an outlier: {throughput}")
        print(f"  FAIL  throughput summary: {throughput}")

    raw = raw_measurements({
        (labels[0], "tensorops-f32"): 3.0,
        (labels[1], "tensorops-f32"): 4.0,
    })
    if {row["gflops"] for row in raw} == {3.0, 4.0} and len(raw) == 2:
        print("  ok    outer-round child aggregates retain every lane measurement")
    else:
        FAILURES.append(f"raw GEMM measurements were lost: {raw}")
        print(f"  FAIL  raw GEMM measurements: {raw}")
    round_record = raw_round_record(
        2,
        ["comparison", "tessl"],
        {(labels[0], "tensorops-f32"): 3.0},
        {(labels[0], "mps-f32"): 2.0},
    )
    if (round_record["round"] == 2
            and round_record["execution_order"] == ["comparison", "tessl"]
            and len(round_record["tessl"]) == len(round_record["comparison"]) == 1):
        print("  ok    outer-round aggregate record retains pairing and execution order")
    else:
        FAILURES.append(f"GEMM raw round lost pairing/order: {round_record}")
        print(f"  FAIL  GEMM raw round record: {round_record}")

    duplicate_rows = [
        dict(shape=labels[0], backend="tensorops-f32", gflops=1.0),
        dict(shape=labels[0], backend="tensorops-f32", gflops=2.0),
    ]
    real_run = paired_cross_runtime.subprocess.run
    try:
        paired_cross_runtime.subprocess.run = lambda *a, **kw: types.SimpleNamespace(
            returncode=0, stdout=json.dumps(duplicate_rows), stderr=""
        )
        try:
            paired_cross_runtime._run(["synthetic"], {}, "synthetic")
        except SystemExit as exc:
            if "duplicate measurement" in str(exc):
                print("  ok    duplicate GEMM measurements fail instead of overwriting")
            else:
                FAILURES.append(f"duplicate GEMM row raised wrong error: {exc}")
                print(f"  FAIL  duplicate GEMM row raised wrong error: {exc}")
        else:
            FAILURES.append("duplicate GEMM rows silently overwrote each other")
            print("  FAIL  duplicate GEMM rows were accepted")
    finally:
        paired_cross_runtime.subprocess.run = real_run


def attn_paired_contract():
    """`attn_paired` -- the same coverage rule, plus the batching guard.

    The batching guard exists because this driver once set BENCH_ATTN_BATCHED
    for the Python lane and not the Rust one, compared tessl at batch=1 against
    MLX at batch=32, and printed a perfectly plausible 11.6x. Both lanes now
    echo the batch they ran and the driver refuses the ratio; that refusal is
    what is tested here.
    """
    import types
    import attn_paired
    print("\n-- attn_paired: coverage rule and batching guard --")

    cfgs = ["swa128_decode_1k", "swa128_decode_4k"]
    a = {(c, "tessl-decode"): 1.0 for c in cfgs}
    b = {(c, "mlx"): 1.0 for c in cfgs}
    rounds = [(dict(a), dict(b)) for _ in range(3)]
    gaps = attn_paired.missing_coverage(rounds, cfgs, ["mlx"], "tessl-decode")
    if gaps:
        FAILURES.append(f"attn complete rounds reported gaps: {gaps}")
        print(f"  FAIL  complete rounds reported gaps: {gaps}")
    else:
        print("  ok    complete coverage reports no gaps")

    holed = [(dict(a), dict(b)) for _ in range(3)]
    del holed[2][1][("swa128_decode_4k", "mlx")]
    gaps = attn_paired.missing_coverage(holed, cfgs, ["mlx"], "tessl-decode")
    if gaps != ["swa128_decode_4k/mlx"]:
        FAILURES.append(f"attn hole in one round not caught: {gaps}")
        print(f"  FAIL  hole in one round not caught: {gaps}")
    else:
        print(f"  ok    a config missing from one round is caught: {gaps}")

    try:
        attn_paired.validate_requested_configs([cfgs[0], cfgs[0]])
    except ValueError as exc:
        if "duplicates" in str(exc):
            print("  ok    duplicate requested configs cannot weight the aggregate twice")
        else:
            FAILURES.append(f"duplicate configs raised wrong error: {exc}")
            print(f"  FAIL  duplicate configs raised wrong error: {exc}")
    else:
        FAILURES.append("duplicate requested configs were accepted")
        print("  FAIL  duplicate requested configs were accepted")

    stable = [
        (
            {(cfgs[0], "tessl-decode"): ours},
            {(cfgs[0], "mlx"): other},
        )
        for ours, other in ((2.0, 1.0), (2.2, 1.0), (2.1, 1.0))
    ]
    summary = attn_paired.summarize_comparison(
        stable, cfgs[:1], "tessl-decode", "mlx", 1.25
    )
    row = summary["per_config"][0]
    if (row["outer_round_values"] == [2.0, 2.2, 2.1]
            and row["tessl_ms"]["outer_round_values"] == [2.0, 2.2, 2.1]
            and summary["paired_geomean_tessl_over_other"]["median"] == 2.1):
        print("  ok    attention evidence retains paired ratios and absolute rounds")
    else:
        FAILURES.append(f"attention raw/paired summary incomplete: {summary}")
        print(f"  FAIL  attention raw/paired summary: {summary}")

    unstable = [
        ({(cfgs[0], "tessl-decode"): 1.0}, {(cfgs[0], "mlx"): 1.0}),
        ({(cfgs[0], "tessl-decode"): 1.5}, {(cfgs[0], "mlx"): 1.0}),
    ]
    try:
        attn_paired.summarize_comparison(
            unstable, cfgs[:1], "tessl-decode", "mlx", 1.25
        )
    except ValueError as exc:
        if "exceeds bounded limit" in str(exc):
            print("  ok    attention paired drift above the cap fails closed")
        else:
            FAILURES.append(f"attention drift raised wrong error: {exc}")
            print(f"  FAIL  attention drift raised wrong error: {exc}")
    else:
        FAILURES.append("attention ratio spread 1.5x passed a 1.25x gate")
        print("  FAIL  excessive attention ratio drift was accepted")

    def with_rows(rows):
        def fake_run(cmd, env=None, **kw):
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        return fake_run

    real_run = attn_paired.subprocess.run
    try:
        # A lane that ran at a different batch than the driver asked for.
        attn_paired.subprocess.run = with_rows(
            [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=1.0, batched=1)])
        try:
            attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 32)
        except SystemExit as exc:
            if "different batching" in str(exc):
                print("  ok    a lane at the wrong batch refuses to form a ratio")
            else:
                FAILURES.append(f"batch mismatch raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("a batch mismatch was accepted")
            print("  FAIL  batch mismatch accepted")

        # A lane that reports no batch at all cannot be confirmed, and an
        # unconfirmable check must not pass like a confirmed one.
        attn_paired.subprocess.run = with_rows(
            [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=1.0)])
        try:
            attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 32)
        except SystemExit as exc:
            if "no 'batched' field" in str(exc):
                print("  ok    a lane that reports no batch is refused, not trusted")
            else:
                FAILURES.append(f"missing batch field raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("a lane reporting no batch was accepted")
            print("  FAIL  missing batch field accepted")

        # A non-finite or zero median is a failed measurement, not a fast one.
        for bad in (0.0, float("nan"), -1.0):
            attn_paired.subprocess.run = with_rows(
                [dict(cfg="swa128_decode_1k", runtime="mlx", median_ms=bad, batched=1)])
            try:
                attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 1)
            except SystemExit:
                pass
            else:
                FAILURES.append(f"median_ms={bad} was accepted as a timing")
                print(f"  FAIL  median_ms={bad} accepted")
                break
        else:
            print("  ok    zero, NaN and negative medians are all refused")

        duplicate_rows = [
            dict(cfg=cfgs[0], runtime="mlx", median_ms=1.0, batched=1),
            dict(cfg=cfgs[0], runtime="mlx", median_ms=2.0, batched=1),
        ]
        attn_paired.subprocess.run = with_rows(duplicate_rows)
        try:
            attn_paired._run(["x"], {}, "fake lane", lambda x: x["runtime"], 1)
        except SystemExit as exc:
            if "duplicate measurement" in str(exc):
                print("  ok    duplicate attention rows fail instead of overwriting")
            else:
                FAILURES.append(f"duplicate attention row raised wrong error: {exc}")
                print(f"  FAIL  duplicate attention row raised wrong error: {exc}")
        else:
            FAILURES.append("duplicate attention rows silently overwrote each other")
            print("  FAIL  duplicate attention rows were accepted")

        raw = attn_paired.raw_measurements({
            (cfgs[0], "tessl-decode"): 2.0,
            (cfgs[0], "mlx"): 1.0,
        })
        if len(raw) == 2 and {row["median_ms"] for row in raw} == {1.0, 2.0}:
            print("  ok    outer-round attention aggregates retain both lanes")
        else:
            FAILURES.append(f"raw attention measurements were lost: {raw}")
            print(f"  FAIL  raw attention measurements: {raw}")
        round_record = attn_paired.raw_round_record(
            1,
            ["tessl", "comparison"],
            {(cfgs[0], "tessl-decode"): 2.0},
            {(cfgs[0], "mlx"): 1.0},
        )
        if (round_record["round"] == 1
                and round_record["execution_order"] == ["tessl", "comparison"]
                and len(round_record["tessl"]) == len(round_record["comparison"]) == 1):
            print("  ok    attention aggregate round retains pairing and execution order")
        else:
            FAILURES.append(f"attention raw round lost pairing/order: {round_record}")
            print(f"  FAIL  attention raw round record: {round_record}")
    finally:
        attn_paired.subprocess.run = real_run


ATTN_CLI_CASES = [
    ("--dump-parity with no directory", ["--dump-parity"], {}, "requires a directory"),
    ("BENCH_ITERS=0", [], {"BENCH_ITERS": "0"}, "below the minimum"),
    ("BENCH_ITERS=abc", [], {"BENCH_ITERS": "abc"}, "not a non-negative integer"),
    ("BENCH_ATTN_DIST=gaussian", [], {"BENCH_ATTN_DIST": "gaussian"}, "not a distribution"),
    ("BENCH_ATTN_CFGS=nope", [], {"BENCH_ATTN_CFGS": "nope"}, "not a configuration"),
    ("BENCH_ATTN_CFGS=swa128_prefill_5120", [],
     {"BENCH_ATTN_CFGS": "swa128_prefill_5120"}, "not a configuration"),
    # A parity artifact must describe the shipping configuration. Under a
    # tuning override it would describe a kernel no caller reaches, which is
    # exactly how the forced-decode lane came to be dumped at chunk 256 while
    # the library shipped 128.
    ("--dump-parity with BENCH_ATTN_DECODE_CHUNK", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_DECODE_CHUNK": "64"}, "shipping configuration"),
    ("--dump-parity with BENCH_ATTN_DECODE_R", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_DECODE_R": "32"}, "shipping configuration"),
    ("--dump-parity with BENCH_ATTN_ROWS_R", ["--dump-parity", "/tmp/attn-refused"],
     {"BENCH_ATTN_ROWS_R": "32"}, "shipping configuration"),
    ("--dump-parity with TESSL_ATTN_TILED", ["--dump-parity", "/tmp/attn-refused"],
     {"TESSL_ATTN_TILED": "1"}, "shipping configuration"),
]


def attention_semantics():
    """The masking rule and GQA mapping the Python lane transcribes.

    If these were wrong, torch and MLX would be wrong in exactly the same way
    as each other and a speed ratio would still look plausible -- which is why
    the parity run scores the comparison runtimes against the f64 reference too,
    and why the rule itself is checked here against hand-computed cases.
    """
    from flash_attn_torch_mlx import keep_mask, CFGS, BY_LABEL, live_pairs
    print("\n-- attention masking rule and config mirror --")

    # Causal prefill from position 0: row t keeps exactly t+1 keys.
    c = dict(tq=6, tkv=6, window=-1, q_off=0, kv_off=0)
    m = keep_mask(c)
    if list(m.sum(axis=1)) == [1, 2, 3, 4, 5, 6]:
        print("  ok    global causal: row t keeps t+1 keys")
    else:
        FAILURES.append(f"causal row counts wrong: {list(m.sum(axis=1))}")
        print(f"  FAIL  causal row counts: {list(m.sum(axis=1))}")

    # Sliding window w: row t keeps min(t+1, w).
    c = dict(tq=6, tkv=6, window=3, q_off=0, kv_off=0)
    m = keep_mask(c)
    if list(m.sum(axis=1)) == [1, 2, 3, 3, 3, 3]:
        print("  ok    sliding window: row t keeps min(t+1, w) keys")
    else:
        FAILURES.append(f"window row counts wrong: {list(m.sum(axis=1))}")
        print(f"  FAIL  window row counts: {list(m.sum(axis=1))}")

    # Decode with an offset: q_abs = q_off, so a window of w keeps the last w.
    c = dict(tq=1, tkv=100, window=8, q_off=99, kv_off=0)
    m = keep_mask(c)
    if m.sum() == 8 and m[0, 92] and m[0, 99] and not m[0, 91]:
        print("  ok    offset decode: window lands on the last w keys")
    else:
        FAILURES.append(f"offset decode mask wrong: sum={m.sum()}")
        print(f"  FAIL  offset decode mask: sum={m.sum()}")

    # A window wider than the history is inert -- plain causal.
    a = keep_mask(dict(tq=8, tkv=8, window=999, q_off=0, kv_off=0))
    b = keep_mask(dict(tq=8, tkv=8, window=-1, q_off=0, kv_off=0))
    if (a == b).all():
        print("  ok    a window wider than the history is inert")
    else:
        FAILURES.append("wide window differs from causal")
        print("  FAIL  wide window differs from causal")

    # Nothing in the shipped set masks a row completely: a fully masked row is
    # a legitimate decode state the kernels answer with zeros, but torch SDPA
    # returns NaN for it, so a config that hit it would compare noise.
    empty = [c["label"] for c in CFGS if (keep_mask(c).sum(axis=1) == 0).any()]
    if empty:
        FAILURES.append(f"configs with a fully masked row: {empty}")
        print(f"  FAIL  fully masked rows in {empty}")
    else:
        print(f"  ok    no config in the set has a fully masked row ({len(CFGS)} checked)")

    # Live-pair counts must match the Rust FLOP accounting exactly, or the two
    # sides report GFLOP/s against different denominators.
    want = {"swa128_prefill_512": 131328, "swa128_decode_1k": 1024,
            "global512_decode_4k": 4096, "swa128_prefill_2048": 1573376}
    biffed = {k: (live_pairs(BY_LABEL[k]), v) for k, v in want.items()
              if live_pairs(BY_LABEL[k]) != v}
    if biffed:
        FAILURES.append(f"live-pair counts disagree with the Rust lane: {biffed}")
        print(f"  FAIL  live-pair counts: {biffed}")
    else:
        print(f"  ok    live-pair counts match the Rust FLOP accounting ({len(want)} checked)")


def tune_knob_contract():
    """The knob table and the environment it actually hands the binary.

    `attn_tune.run` clears every knob it is not sweeping, because a knob left
    over from the caller's shell would silently be measured as part of the
    value under test. That clearing list used to be a second hardcoded copy of
    the knob names, so adding a knob to KNOBS and forgetting the copy leaked
    it. Both the table's shape and the env the subprocess receives are checked
    here.
    """
    import types
    import attn_tune
    from flash_attn_torch_mlx import BY_LABEL
    print("\n-- attn_tune knob table and subprocess environment --")

    declared = {env for env, *_ in attn_tune.KNOBS.values()}
    if declared == set(attn_tune.KNOB_ENV):
        print(f"  ok    KNOB_ENV covers all {len(declared)} knob variables")
    else:
        FAILURES.append(f"KNOB_ENV {attn_tune.KNOB_ENV} != declared {sorted(declared)}")
        print("  FAIL  KNOB_ENV does not cover every knob")

    lanes = {"tessl", "tessl-tiled", "tessl-decode", "tessl-rows"}
    for knob, (env, values, lane, cfgs) in sorted(attn_tune.KNOBS.items()):
        bad = [c for c in cfgs if c not in BY_LABEL]
        if bad:
            FAILURES.append(f"knob {knob}: unknown configs {bad}")
            print(f"  FAIL  {knob}: unknown configs {bad}")
        elif lane not in lanes:
            FAILURES.append(f"knob {knob}: lane {lane!r} is not one the binary emits")
            print(f"  FAIL  {knob}: unknown lane {lane!r}")
        elif len(set(values)) != len(values):
            FAILURES.append(f"knob {knob}: duplicate values {values}")
            print(f"  FAIL  {knob}: duplicate values {values}")
        else:
            print(f"  ok    {knob}: {env}={values} on {lane}, {len(cfgs)} configs")

    # The decomposition report divides batched by solo, so the values are part
    # of the contract, not a default someone may reorder.
    if attn_tune.KNOBS["batched"][1] == ["1", "32"]:
        print("  ok    batched knob is ('1', '32'), the order the share assumes")
    else:
        FAILURES.append(f"batched values reordered: {attn_tune.KNOBS['batched'][1]}")
        print("  FAIL  batched knob values are not ('1', '32')")

    # A knob left in the caller's environment must not survive into a sweep of
    # a different knob.
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen.clear()
        seen.update(env or {})
        rows = [dict(cfg=c, runtime="tessl-rows", median_ms=1.0)
                for c in attn_tune.KNOBS["rows"][3]]
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    real_run, real_env = attn_tune.subprocess.run, dict(os.environ)
    try:
        attn_tune.subprocess.run = fake_run
        os.environ["BENCH_ATTN_DECODE_CHUNK"] = "256"
        os.environ["BENCH_ATTN_BATCHED"] = "32"
        env_name, _, lane, cfgs = attn_tune.KNOBS["rows"]
        attn_tune.run(env_name, "8", lane, cfgs, 3, 1)
    finally:
        attn_tune.subprocess.run = real_run
        os.environ.clear()
        os.environ.update(real_env)

    # BENCH_ATTN_BATCHED is not cleared but *set*, so a stray value in the
    # caller's shell is overridden rather than inherited: a sweep silently run
    # at someone else's batching is a sweep of the wrong thing.
    leaked = [k for k in attn_tune.KNOB_ENV
              if k not in ("BENCH_ATTN_ROWS_R", "BENCH_ATTN_BATCHED") and k in seen]
    if leaked:
        FAILURES.append(f"knobs leaked into a rows sweep: {leaked}")
        print(f"  FAIL  knobs leaked into the child env: {leaked}")
    elif seen.get("BENCH_ATTN_ROWS_R") != "8":
        FAILURES.append(f"swept knob not set: BENCH_ATTN_ROWS_R={seen.get('BENCH_ATTN_ROWS_R')}")
        print("  FAIL  the swept knob did not reach the child env")
    elif seen.get("BENCH_ATTN_BATCHED") != "1":
        FAILURES.append(f"stray BENCH_ATTN_BATCHED inherited: {seen.get('BENCH_ATTN_BATCHED')}")
        print(f"  FAIL  inherited BENCH_ATTN_BATCHED={seen.get('BENCH_ATTN_BATCHED')}")
    else:
        print("  ok    sweeping one knob clears the others and pins the batching")

    # A sweep of launches-per-submit cannot also be handed a fixed batching.
    import subprocess as _sp
    r = _sp.run([sys.executable, os.path.join(HERE_BENCH, "attn_tune.py"),
                 "--knob", "batched", "--batched", "32"],
                capture_output=True, text=True)
    if r.returncode != 0 and "Drop one of them" in (r.stderr + r.stdout):
        print("  ok    --knob batched with --batched is refused")
    else:
        FAILURES.append("--knob batched --batched 32 was accepted")
        print("  FAIL  --knob batched --batched 32 accepted")


def attn_cli_contract():
    import subprocess
    print("\n-- bench_flash_attn argument / environment contract --")
    binary = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "target", "release", "bench_flash_attn")
    if not os.path.exists(binary):
        reason = f"{binary} not built; CLI contract unverified"
        print(f"  SKIP  {reason}")
        return Skip(reason)
    probe = subprocess.run([binary, "--dump-parity"], capture_output=True, text=True)
    if "requires a directory" not in probe.stderr and "panicked" not in probe.stderr:
        reason = f"no usable Metal runtime: {probe.stderr.strip()[:90]}"
        print(f"  SKIP  {reason}")
        return Skip(reason)
    for label, argv, env, needle in ATTN_CLI_CASES:
        e = dict(os.environ, **env)
        for var in ("BENCH_ITERS", "BENCH_WARMUP", "BENCH_ATTN_DIST", "BENCH_ATTN_CFGS",
                    "BENCH_ATTN_DECODE_CHUNK", "BENCH_ATTN_DECODE_R",
                    "BENCH_ATTN_ROWS_R", "TESSL_ATTN_TILED"):
            if var not in env:
                e.pop(var, None)
        r = subprocess.run([binary] + argv, env=e, capture_output=True, text=True)
        out = r.stderr + r.stdout
        if "panicked at" in out or r.returncode < 0:
            FAILURES.append(f"attn CLI {label}: panicked")
            print(f"  FAIL  {label}: panic")
        elif r.returncode == 0:
            FAILURES.append(f"attn CLI {label}: exited 0, accepting bad input")
            print(f"  FAIL  {label}: accepted")
        elif needle.lower() not in out.lower():
            FAILURES.append(f"attn CLI {label}: error lacks {needle!r}")
            print(f"  FAIL  {label}: message lacks {needle!r}")
        else:
            print(f"  ok    {label}: exit {r.returncode}, names the cause")


def coverage_inventory():
    """`kernel_coverage.inventory` -- the kernel census.

    This is where the worst error in the audit came from: a scan for
    `kernel void` missed every kernel declared through the `NN_COOP_KERNEL`
    family of macros, so the inventory read 67 when the true count was 83 and
    16 kernels could never have shown as uncovered. The guard that makes that
    impossible is the one worth testing.
    """
    from kernel_coverage import inventory
    print("\n-- kernel inventory (the census coverage is measured against) --")
    partial_skip = None
    tmp = tempfile.mkdtemp(prefix="kcov-")
    try:
        def write(body):
            d = tempfile.mkdtemp(dir=tmp)
            with open(os.path.join(d, "k.metal"), "w") as f:
                f.write(body)
            return d

        d = write("kernel void alpha(device float* x) {}\n"
                  "NN_COOP_KERNEL(beta, bfloat, 64, 64, 4, false)\n"
                  "TN_NT_COOP_KERNEL( gamma , float, 128, 64)\n")
        names, _ = inventory(d, metallib="/nonexistent")
        if names == {"alpha", "beta", "gamma"}:
            print("  ok    plain and macro-declared kernels are both counted")
        else:
            FAILURES.append(f"inventory missed declarations: {sorted(names)}")
            print(f"  FAIL  inventory got {sorted(names)}")

        # A macro *body* containing `kernel void NAME(` must not be scanned:
        # NAME is the macro parameter, and counting it invents a kernel that is
        # in no metallib. The metallib cross-check caught this for real when
        # the decode kernels landed.
        # The macro name must end in KERNEL for the pattern to recognise it,
        # which is the convention every kernel-declaring macro in the tree uses.
        d = write("#define MK_KERNEL(NAME, D) \\\n"
                  "kernel void NAME(device float* x) {}\n"
                  "MK_KERNEL(real_one, 128)\n")
        names, _ = inventory(d, metallib="/nonexistent")
        if names == {"real_one"}:
            print("  ok    a macro body's parameter is not counted as a kernel")
        else:
            FAILURES.append(f"macro body leaked into the inventory: {sorted(names)}")
            print(f"  FAIL  macro body leaked: {sorted(names)}")

        # A macro whose first argument is not an identifier cannot be parsed
        # for a kernel name, and must fail rather than be skipped.
        d = write("kernel void alpha(device float* x) {}\n"
                  'WEIRD_NEW_KERNEL("delta", bfloat)\n')
        try:
            inventory(d, metallib="/nonexistent")
        except SystemExit as exc:
            if "not parsed" in str(exc):
                print("  ok    an unparsed kernel macro fails loudly")
            else:
                FAILURES.append(f"unparsed macro raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("an unparsed kernel macro was silently undercounted")
            print("  FAIL  unparsed macro silently undercounted")

        # An empty kernel directory is a broken run, not zero kernels.
        d = tempfile.mkdtemp(dir=tmp)
        try:
            inventory(d, metallib="/nonexistent")
        except SystemExit:
            print("  ok    an empty kernel directory is refused, not read as zero")
        else:
            FAILURES.append("an empty kernel directory was accepted")
            print("  FAIL  empty kernel directory accepted")

        # Reporting source coverage without checking the compiled library used
        # to print the same PASS when the library or metal-nm was unavailable.
        # Library consumers may request a source-only census, but the CLI's
        # actual coverage measurement must fail closed.
        d = write("kernel void alpha(device float* x) {}\n")
        try:
            inventory(d, metallib="/definitely/missing.metallib", require_metallib=True)
        except SystemExit as exc:
            if "metallib missing" in str(exc):
                print("  ok    required metallib cross-check fails closed when absent")
            else:
                FAILURES.append(f"missing metallib raised wrong error: {exc}")
                print(f"  FAIL  wrong missing-metallib error: {exc}")
        else:
            FAILURES.append("required missing metallib was accepted")
            print("  FAIL  required missing metallib accepted")

        # A source scan that disagrees with the compiled metallib is wrong,
        # whichever way it disagrees, and must not be measured against.
        from kernel_coverage import metallib_symbols
        real = metallib_symbols()
        if real is None:
            reason = "metal-nm unavailable; metallib cross-check unverified"
            print(f"  SKIP  {reason}")
            partial_skip = PartialSkip("compiled metallib cross-check", reason)
        else:
            d = write("kernel void not_a_real_kernel(device float* x) {}\n")
            try:
                inventory(d)
            except SystemExit as exc:
                if "disagrees with the compiled metallib" in str(exc):
                    print(f"  ok    source scan vs metallib disagreement is fatal "
                          f"({len(real)} exported)")
                else:
                    FAILURES.append(f"metallib mismatch raised wrong error: {exc}")
                    print(f"  FAIL  wrong error: {str(exc)[:70]}")
            else:
                FAILURES.append("a source scan disagreeing with the metallib was accepted")
                print("  FAIL  metallib disagreement accepted")

        # The real tree must agree with the union of both declaration forms.
        names, _ = inventory()
        if len(names) >= 80 and "matmul2d_tensorops_bf16_f32_64x64_sg4" in names:
            print(f"  ok    real tree: {len(names)} entry points, macro-declared ones included")
        else:
            FAILURES.append(f"real inventory looks wrong: {len(names)} names")
            print(f"  FAIL  real inventory: {len(names)} names")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return partial_skip


def tile_audit_contract():
    """The static tile audit must reject missing or unverifiable geometry."""
    import importlib.util

    path = os.path.join(os.path.dirname(HERE_BENCH), "scripts", "audit_gemm_tiles.py")
    spec = importlib.util.spec_from_file_location("audit_gemm_tiles", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    print("\n-- static TileGeom audit fails closed --")
    kernels = {"matmul_ok": (32, 64, 4)}
    tiles = {"TILE_OK": (32, 64, 4)}
    cases = [
        ("matching geometry", kernels, tiles, "matmul_ok", "TILE_OK", False),
        ("missing kernel", kernels, tiles, "matmul_missing", "TILE_OK", True),
        ("missing Rust tile", kernels, tiles, "matmul_ok", "TILE_MISSING", True),
        ("kernel with no geometry", {"matmul_ok": (None, None, None)},
         tiles, "matmul_ok", "TILE_OK", True),
        ("mismatched geometry", kernels, {"TILE_OK": (64, 64, 4)},
         "matmul_ok", "TILE_OK", True),
    ]
    for name, kt, rt, kernel, tile, expect_error in cases:
        error = audit.geometry_error(kt, rt, kernel, tile)
        if (error is not None) != expect_error:
            FAILURES.append(f"tile audit {name}: unexpected result {error!r}")
            print(f"  FAIL  {name}: {error!r}")
        else:
            print(f"  ok    {name}: {error or 'accepted'}")


def execution_policy_contract():
    """Pure regression for mode routing, accounting, verdicts, and exit codes."""
    print("\n-- harness execution policy and accounting --")

    def require(condition, message):
        if condition:
            print(f"  ok    {message}")
        else:
            FAILURES.append(f"execution policy: {message}")
            print(f"  FAIL  {message}")

    called = []
    pure = HarnessSummary("pure")
    run_gpu_section(pure, "pure", "synthetic GPU lane", lambda: called.append(True))
    pure_doc = pure.as_dict()
    require(not called, "--pure does not invoke a GPU-backed section")
    require(
        pure_doc["counts"] == dict(ran=0, skipped=1, failed=0, assertion_failures=0),
        "--pure records the excluded GPU lane as skipped",
    )
    require(
        pure.verdict() == "PASS_WITH_SKIPS" and pure.exit_code() == 0,
        "an intentional pure-mode skip is visible but not fatal",
    )

    optional = HarnessSummary("default")
    run_gpu_section(optional, "default", "synthetic GPU lane", lambda: Skip("unavailable"))
    require(
        optional.verdict() == "PASS_WITH_SKIPS" and optional.exit_code() == 0,
        "a default-mode GPU skip cannot masquerade as a complete PASS",
    )

    partial = HarnessSummary("default")
    run_section(
        partial,
        "synthetic host section",
        lambda: PartialSkip("optional cross-check", "tool unavailable"),
    )
    require(
        partial.as_dict()["counts"]["ran"] == 1
        and partial.as_dict()["counts"]["skipped"] == 1
        and partial.skipped[0]["partial"]
        and partial.verdict() == "PASS_WITH_SKIPS",
        "a partial sub-check skip is visible alongside completed section work",
    )

    required = HarnessSummary("require-gpu")
    run_gpu_section(required, "require-gpu", "synthetic GPU lane", lambda: Skip("unavailable"))
    required_doc = required.as_dict()
    require(
        required_doc["required_gpu_skips"] == ["synthetic GPU lane"]
        and required.verdict() == "FAIL"
        and required.exit_code() == 1,
        "--require-gpu turns a skipped GPU lane into a failing exit",
    )

    local_failures = []
    failed = HarnessSummary("default")
    run_section(
        failed,
        "synthetic failure",
        lambda: local_failures.append("deliberate"),
        failures=local_failures,
    )
    require(
        failed.failed == ["synthetic failure"]
        and failed.verdict(len(local_failures)) == "FAIL"
        and failed.exit_code(len(local_failures)) == 1,
        "a section failure is counted and exits nonzero",
    )

    ran = HarnessSummary("default")
    run_section(ran, "synthetic success", lambda: None)
    require(
        ran.as_dict()["counts"]["ran"] == 1 and ran.verdict() == "PASS",
        "a completed section is counted as ran",
    )


def benchmark_evidence_contract():
    """Pure regressions for drift policy, provenance, and atomic publication."""
    import subprocess

    from benchmark_evidence import (
        DEFAULT_MAX_RATIO_SPREAD,
        DEFAULT_OUTER_ROUNDS,
        MAX_RATIO_SPREAD,
        EvidenceOutput,
        clean_benchmark_env,
        embedded_metallib_path,
        finish_provenance,
        parse_ratio_spread_limit,
        requested_output_path,
        start_provenance,
        validate_evidence_sample_counts,
    )

    print("\n-- cross-runtime evidence provenance and bounded policy --")
    if (DEFAULT_MAX_RATIO_SPREAD == 1.10 and MAX_RATIO_SPREAD == 1.25
            and DEFAULT_OUTER_ROUNDS == 6
            and parse_ratio_spread_limit("1.25") == 1.25):
        print("  ok    defaults are 6 balanced rounds/10% drift; override caps at 25%")
    else:
        FAILURES.append(
            f"drift policy is default={DEFAULT_MAX_RATIO_SPREAD}, max={MAX_RATIO_SPREAD}"
        )
        print("  FAIL  ratio-spread safety defaults are wrong")
    for invalid in ("nan", "inf", "0.99", "1.251"):
        try:
            parse_ratio_spread_limit(invalid)
        except ValueError:
            continue
        FAILURES.append(f"unbounded ratio-spread limit {invalid!r} accepted")
        print(f"  FAIL  unbounded ratio-spread limit {invalid!r} accepted")
        break
    else:
        print("  ok    NaN, infinity, sub-unit, and unbounded drift caps are refused")
    selected = requested_output_path(
        ["driver.py", "--out", "first.json", "--out=second.json"]
    )
    if selected == "second.json":
        print("  ok    pre-validation output discovery deterministically uses the last value")
    else:
        FAILURES.append(f"pre-validation output discovery selected {selected!r}")
        print(f"  FAIL  pre-validation output discovery selected {selected!r}")
    old_knob = os.environ.get("TESSL_GEMM_ACCUM")
    try:
        os.environ["TESSL_GEMM_ACCUM"] = "1"
        clean_env = clean_benchmark_env({"BENCH_ITERS": "7"})
    finally:
        if old_knob is None:
            os.environ.pop("TESSL_GEMM_ACCUM", None)
        else:
            os.environ["TESSL_GEMM_ACCUM"] = old_knob
    if ("TESSL_GEMM_ACCUM" not in clean_env and clean_env.get("BENCH_ITERS") == "7"
            and clean_env.get("PATH") == os.environ.get("PATH")):
        print("  ok    inherited tuning knobs are cleared; only explicit overrides survive")
    else:
        FAILURES.append("benchmark environment sanitizer leaked or dropped the wrong values")
        print("  FAIL  benchmark environment sanitizer")
    validate_evidence_sample_counts(4, 3)
    rejected_counts = []
    invalid_counts = (
        (0, 3), (1, 3), (2, 3), (3, 3), (5, 3),
        (4, 0), (4, 1), (4, 2),
    )
    for rounds, iters in invalid_counts:
        try:
            validate_evidence_sample_counts(rounds, iters)
        except ValueError:
            rejected_counts.append((rounds, iters))
    if rejected_counts == list(invalid_counts):
        print("  ok    low/odd round counts and sub-median inner samples are refused")
    else:
        FAILURES.append(f"insufficient sample counts accepted: {rejected_counts}")
        print(f"  FAIL  insufficient sample-count gate: {rejected_counts}")

    tmp = tempfile.mkdtemp(prefix="benchmark-evidence-")
    try:
        metallib = os.path.join(tmp, "default-synthetic.metallib")
        binary = os.path.join(tmp, "benchmark")
        with open(metallib, "wb") as handle:
            handle.write(b"synthetic metal library")
        with open(binary, "wb") as handle:
            handle.write(b"prefix" + metallib.encode() + b"\x00suffix")
        if embedded_metallib_path(binary) == os.path.realpath(metallib):
            print("  ok    the exact metallib path embedded in the binary is resolved")
        else:
            FAILURES.append("embedded metallib resolver selected the wrong file")
            print("  FAIL  embedded metallib resolver selected the wrong file")

        hostile_dir = os.path.join(tmp, "checkout @([βeta])")
        os.makedirs(hostile_dir)
        hostile_metallib = os.path.join(hostile_dir, "default [tuned]@2.metallib")
        hostile_binary = os.path.join(tmp, "benchmark-hostile-path")
        with open(hostile_metallib, "wb") as handle:
            handle.write(b"synthetic metal library")
        with open(hostile_binary, "wb") as handle:
            handle.write(b"prefix" + hostile_metallib.encode() + b"\x00suffix")
        if embedded_metallib_path(hostile_binary) == os.path.realpath(hostile_metallib):
            print("  ok    @, parentheses/brackets, and non-ASCII path bytes are resolved")
        else:
            FAILURES.append("embedded metallib resolver rejected a valid hostile path")
            print("  FAIL  embedded metallib resolver rejected a valid hostile path")

        provenance = start_provenance(
            driver_path=__file__,
            argv=[__file__, "--synthetic", "value with spaces"],
            repo_scope=os.path.dirname(HERE_BENCH),
            executable_inputs={"fixture": __file__, "metal_library": metallib},
            benchmark_config={"rounds": 6, "synthetic": True, "child_path": __file__},
            probe_device=False,
        )
        finish_provenance(provenance)
        required = (
            provenance.get("git", {}).get("revision"),
            provenance.get("host", {}).get("machine"),
            provenance.get("os", {}).get("platform"),
            provenance.get("runtime", {}).get("python"),
            provenance.get("load", {}).get("start"),
            provenance.get("load", {}).get("finish"),
            provenance.get("power", {}).get("thermal_pressure"),
            provenance.get("power", {}).get("thermal_pressure_finish"),
            provenance.get("power", {}).get("power_source_finish"),
            provenance.get("environment_policy"),
            provenance.get("path_policy"),
            provenance.get("inputs", {}).get("fixture", {}).get("sha256"),
            provenance.get("inputs", {}).get("metal_library", {}).get("sha256"),
            provenance.get("invocation_shell"),
            provenance.get("finished_at_utc"),
        )
        dirty_recorded = isinstance(provenance.get("git", {}).get("dirty"), bool)
        no_full_environment = "environment" not in provenance
        serialized_provenance = json.dumps(provenance)
        home = os.path.abspath(os.path.expanduser("~"))
        home_redacted = (
            home == os.path.sep
            or (home not in serialized_provenance and "$HOME" in serialized_provenance)
        )
        if (all(required) and dirty_recorded and no_full_environment
                and home_redacted and provenance["benchmark_config"]["rounds"] == 6):
            try:
                json.dumps(provenance)
            except (TypeError, ValueError) as exc:
                FAILURES.append(f"provenance is not JSON serializable: {exc}")
                print(f"  FAIL  provenance is not JSON serializable: {exc}")
            else:
                print("  ok    revision/dirty, host/runtime, power/load, invocation, and hashes persist")
        else:
            FAILURES.append(f"provenance is missing required evidence: {required}")
            print(f"  FAIL  provenance is missing required evidence: {required}")

        output_path = os.path.join(tmp, "result.json")
        with open(output_path, "w") as handle:
            json.dump({"generation": "old"}, handle)
        publisher = EvidenceOutput(
            output_path, driver_path=__file__, argv=[__file__, "--out", output_path]
        )
        publisher.begin()
        with open(output_path) as handle:
            preserved = json.load(handle)
        with open(f"{output_path}.attempt.json") as handle:
            pending = json.load(handle)
        serialized_pending = json.dumps(pending)
        if (preserved == {"generation": "old"}
                and pending["status"] == "not_published"
                and pending["prior_output"]["sha256"]
                and (home == os.path.sep or home not in serialized_pending)):
            print(
                "  ok    a new attempt marks an older output stale without "
                "overwriting it or disclosing the home path"
            )
        else:
            FAILURES.append(f"failed-attempt marker is incomplete: {pending}")
            print(f"  FAIL  failed-attempt marker is incomplete: {pending}")

        publisher.publish({"generation": "new"})
        with open(output_path) as handle:
            published = json.load(handle)
        with open(f"{output_path}.attempt.json") as handle:
            marker = json.load(handle)
        staged_left = [name for name in os.listdir(tmp) if name.startswith(".result.json.")]
        if (published == {"generation": "new"} and marker["status"] == "published"
                and marker["published_output"]["sha256"] and not staged_left
                and os.path.dirname(publisher.marker_path) == os.path.dirname(output_path)):
            print("  ok    successful evidence publication is atomic and hash-recorded")
        else:
            FAILURES.append(
                f"atomic publication failed: published={published}, marker={marker}, temp={staged_left}"
            )
            print("  FAIL  atomic evidence publication contract")

        nonfinite_output = os.path.join(tmp, "nonfinite.json")
        with open(nonfinite_output, "w") as handle:
            json.dump({"generation": "old"}, handle)
        nonfinite_publisher = EvidenceOutput(
            nonfinite_output,
            driver_path=__file__,
            argv=[__file__, "--out", nonfinite_output],
        )
        nonfinite_publisher.begin()
        try:
            nonfinite_publisher.publish({"ratio": float("nan")})
        except ValueError:
            with open(nonfinite_output) as handle:
                nonfinite_preserved = json.load(handle)
            with open(f"{nonfinite_output}.attempt.json") as handle:
                nonfinite_marker = json.load(handle)
            nonfinite_temps = [
                name for name in os.listdir(tmp) if name.startswith(".nonfinite.json.")
            ]
            if (nonfinite_preserved == {"generation": "old"}
                    and nonfinite_marker["status"] == "not_published"
                    and not nonfinite_temps):
                print("  ok    non-finite JSON cannot replace prior evidence")
            else:
                FAILURES.append("non-finite publication damaged the prior artifact")
                print("  FAIL  non-finite publication damaged the prior artifact")
        else:
            FAILURES.append("non-finite JSON was published as evidence")
            print("  FAIL  non-finite JSON was published as evidence")

        missing_parent = os.path.join(tmp, "missing-parent")
        missing_output = os.path.join(missing_parent, "result.json")
        missing_publisher = EvidenceOutput(
            missing_output,
            driver_path=__file__,
            argv=[__file__, "--out", missing_output],
        )
        try:
            missing_publisher.begin()
        except FileNotFoundError:
            if not os.path.exists(missing_parent):
                print("  ok    output parents must pre-exist and are never created implicitly")
            else:
                FAILURES.append("missing output parent was partially created")
                print("  FAIL  missing output parent was partially created")
        else:
            FAILURES.append("missing output parent was silently created")
            print("  FAIL  missing output parent did not fail closed")

        # The marker is created before argparse validates any other option, so
        # even a malformed rerun makes a preserved older output visibly stale.
        probes = (
            ("cap", ["--max-ratio-spread", "9"], "--max-ratio-spread"),
            ("round3", ["--rounds", "3"], "even integer >= 4"),
            ("round5", ["--rounds", "5"], "even integer >= 4"),
        )
        for script in ("paired_cross_runtime.py", "attn_paired.py"):
            for probe_name, bad_args, needle in probes:
                rejected_output = os.path.join(tmp, f"{script}.{probe_name}.json")
                with open(rejected_output, "w") as handle:
                    json.dump({"generation": "old"}, handle)
                proc = subprocess.run(
                    [
                        sys.executable,
                        os.path.join(HERE_BENCH, script),
                        *bad_args,
                        "--out",
                        rejected_output,
                    ],
                    capture_output=True,
                    text=True,
                )
                with open(f"{rejected_output}.attempt.json") as handle:
                    rejected_marker = json.load(handle)
                with open(rejected_output) as handle:
                    rejected_preserved = json.load(handle)
                combined = proc.stdout + proc.stderr
                if (proc.returncode != 0
                        and rejected_marker["status"] == "not_published"
                        and rejected_preserved == {"generation": "old"}
                        and needle in combined and "round 1/" not in combined):
                    continue
                FAILURES.append(
                    f"{script}/{probe_name}: pre-validation rejection failed: "
                    f"rc={proc.returncode}, marker={rejected_marker}, output="
                    f"{rejected_preserved}, text={combined[:120]!r}"
                )
                print(f"  FAIL  {script}/{probe_name} did not fail before child launch")
                break
            else:
                continue
            break
        else:
            print("  ok    both drivers mark stale and reject bad/odd policy before child launch")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_mode(argv=None):
    parser = argparse.ArgumentParser(
        description="Adversarial parity-harness contracts with explicit GPU skip policy."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--pure",
        action="store_true",
        help="run only host-side contracts; never launch a GPU-backed binary",
    )
    modes.add_argument(
        "--require-gpu",
        action="store_true",
        help="run GPU-backed CLI contracts and fail if either one is skipped",
    )
    args = parser.parse_args(argv)
    if args.pure:
        return "pure"
    if args.require_gpu:
        return "require-gpu"
    return "default"


def main(argv=None):
    mode = parse_mode(argv)
    FAILURES.clear()
    summary = HarnessSummary(mode)
    core_before = len(FAILURES)
    tmp = tempfile.mkdtemp(prefix="parity-harness-")
    try:
        print("\n-- a valid dump is accepted, and reported exactly once per lane --")
        good = build_dump(os.path.join(tmp, "good"), seeds=3)
        rows = check("valid dump", lambda: parity(good), expect_reject=False)
        if rows is not None:
            names = [r["lane"] for r in rows]
            dupes = {x for x in names if names.count(x) > 1}
            if dupes:
                FAILURES.append(f"duplicate lane rows: {sorted(dupes)}")
                print(f"  FAIL  duplicate lane rows: {sorted(dupes)}")
            else:
                print(f"  ok    {len(names)} lanes, no duplicates: {names}")
            for lane in LANES:
                row = next((r for r in rows if r["lane"] == lane), None)
                if row is None or row.get("seeds") != 3:
                    FAILURES.append(f"{lane} did not report all 3 seeds: {row}")
                    print(f"  FAIL  {lane} seeds={row and row.get('seeds')}")
            if all(r.get("seeds") == 3 for r in rows if r["lane"] in LANES):
                print("  ok    every dumped lane covers all 3 seeds")

        print("\n-- the aggregate reports the WORST seed, not the last or the mean --")
        # Seed 1 is perturbed hardest; a scorer that keeps the last seed or
        # averages would under-report it.
        w = build_dump(os.path.join(tmp, "worst"), seeds=3, err_scale=[1e-8, 5e-7, 1e-8])
        rows = check("worst-of-seeds", lambda: parity(w), expect_reject=False)
        if rows:
            row = next(r for r in rows if r["lane"] == "tensorops-f32")
            per = row["per_seed_rel_err"]
            if abs(row["max_rel_err"] - max(per)) > 1e-12:
                FAILURES.append(f"max_rel_err {row['max_rel_err']} != max(per_seed) {max(per)}")
                print("  FAIL  max_rel_err is not the worst seed")
            elif not (per[1] > per[0] * 10):
                FAILURES.append(f"perturbation did not register: {per}")
                print(f"  FAIL  perturbation did not register: {per}")
            else:
                print(f"  ok    worst seed dominates: per_seed={['%.2e' % x for x in per]}")

        print("\n-- structural rejections --")
        d = build_dump(os.path.join(tmp, "nomanifest"))
        os.remove(os.path.join(d, "parity_manifest.json"))
        check("missing manifest", lambda: parity(d), True, "manifest")

        d = build_dump(os.path.join(tmp, "badjson"))
        with open(os.path.join(d, "parity_manifest.json"), "w") as f:
            f.write("{not json")
        check("malformed manifest", lambda: parity(d), True, "unreadable")

        d = build_dump(os.path.join(tmp, "lanestr"))
        edit_manifest(d, lanes="tensorops-f32")
        check("manifest lanes not a list", lambda: parity(d), True, "non-empty list")

        d = build_dump(os.path.join(tmp, "laneint"))
        edit_manifest(d, lanes=[1, 2])
        check("manifest lanes not strings", lambda: parity(d), True, "strings")

        d = build_dump(os.path.join(tmp, "emptylanes"))
        edit_manifest(d, lanes=[])
        check("manifest lanes empty", lambda: parity(d), True, "non-empty list")

        d = build_dump(os.path.join(tmp, "badk"))
        edit_manifest(d, k=0)
        check("manifest zero dim", lambda: parity(d), True, "positive integer")

        d = build_dump(os.path.join(tmp, "notobj"))
        with open(os.path.join(d, "parity_manifest.json"), "w") as f:
            json.dump([1, 2, 3], f)
        check("manifest not an object", lambda: parity(d), True, "object")

        print("\n-- coverage rejections (the original bug's shape) --")
        # Exactly the pre-fix state: the two reduced-precision lanes absent
        # while the manifest still claims four.
        d = build_dump(os.path.join(tmp, "shortlanes"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                for lane in ("tensorops-bf16", "tensorops-tf32"):
                    os.remove(os.path.join(d, s, f"parity_c_{lane}.npy"))
        check("lane in manifest, npy absent", lambda: parity(d), True, "missing")

        d = build_dump(os.path.join(tmp, "shortseed"), seeds=3)
        shutil.rmtree(os.path.join(d, "seed_02"))
        check("seed in manifest, directory absent", lambda: parity(d), True, "absent")

        print("\n-- numeric rejections --")
        for label, bad in (("NaN", np.nan), ("+Inf", np.inf), ("-Inf", -np.inf)):
            d = build_dump(os.path.join(tmp, f"nf{label}"))
            fn = os.path.join(d, "seed_00", "parity_c_tensorops-tf32.npy")
            c = np.load(fn)
            c[3, 4] = bad
            np.save(fn, c)
            check(f"{label} in a result", lambda d=d: parity(d), True, "non-finite")

        d = build_dump(os.path.join(tmp, "wrongshape"))
        np.save(os.path.join(d, "seed_00", "parity_c_tensorops-f32.npy"),
                np.zeros((M, N + 3), dtype=np.float32))
        check("result shape mismatch", lambda: parity(d), True, "shape")

        d = build_dump(os.path.join(tmp, "zeroops"))
        np.save(os.path.join(d, "seed_00", "parity_a.npy"), np.zeros((M, K), np.float32))
        check("all-zero operands (scale 0)", lambda: parity(d), True, "division by zero")

        d = build_dump(os.path.join(tmp, "nanops"))
        a = np.load(os.path.join(d, "seed_00", "parity_a.npy"))
        a[0, 0] = np.nan
        np.save(os.path.join(d, "seed_00", "parity_a.npy"), a)
        check("non-finite operands", lambda: parity(d), True, "non-finite")

        d = build_dump(os.path.join(tmp, "opshape"))
        np.save(os.path.join(d, "seed_00", "parity_b.npy"),
                np.ones((K + 2, N), dtype=np.float32))
        check("operand shape contradicts manifest", lambda: parity(d), True, "contradict")

        print("\n-- per-element error budget --")
        # A lane that writes nothing. The Rust side pre-zeroes C so this cannot
        # inherit the previous lane's result; the budget gate then rejects it
        # outright rather than merely scoring it 1.0.
        d = build_dump(os.path.join(tmp, "zeroed"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                np.save(os.path.join(d, s, "parity_c_tensorops-tf32.npy"),
                        np.zeros((M, N), dtype=np.float32))
        check("no-op lane (all-zero result)", lambda: parity(d), True, "budget exceeded")

        # A drift of 1e-5 relative: innocuous by the normwise metric, and ~2x
        # the f32 lane's per-element budget. This is the case the old harness
        # could not see at all.
        d = build_dump(os.path.join(tmp, "drifted"))
        for s in os.listdir(d):
            if s.startswith("seed_"):
                fn = os.path.join(d, s, "parity_c_tensorops-f32.npy")
                np.save(fn, (np.load(fn).astype(np.float64) * (1 + 1e-5)).astype(np.float32))
        check("f32 lane drifted 1e-5 (normwise would shrug)",
              lambda: parity(d), True, "kernel defect")

        # A lane with no declared precision must not be scored against a
        # budget nobody chose for it.
        d = build_dump(os.path.join(tmp, "unknownlane"), lanes=LANES + ["tensorops-fp8"])
        check("lane with no LANE_PRECISION entry", lambda: parity(d), True,
              "LANE_PRECISION")

        print("\n-- budget verdicts (which response the breach calls for) --")
        from gemm_sweep_mlx import adjudicate
        for name, over, needle in (
            ("tessl only", {"tensorops-bf16": 2.0}, "kernel defect"),
            ("comparison runtime only", {"torch-mps-bf16": 2.0}, "other runtime is the outlier"),
            ("every runtime", {"tensorops-bf16": 2.0, "torch-mps-bf16": 2.1, "mlx-bf16": 2.2},
             "points at the bound"),
        ):
            msg = adjudicate(over)
            if needle in msg:
                print(f"  ok    {name}: {needle!r}")
            else:
                FAILURES.append(f"adjudicate({name}) lacks {needle!r}: {msg}")
                print(f"  FAIL  {name}: lacks {needle!r}")
    except SystemExit as exc:
        FAILURES.append(f"parity scorer contracts: exited unexpectedly -- {exc}")
        print(f"  FAIL  parity scorer contracts: unexpected exit: {exc}")
    except Exception as exc:
        FAILURES.append(
            f"parity scorer contracts: raised {type(exc).__name__}: {exc}"
        )
        print(f"  FAIL  parity scorer contracts: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if len(FAILURES) == core_before:
        summary.ran.append("parity scorer contracts")
    else:
        summary.failed.append("parity scorer contracts")

    run_gpu_section(summary, mode, "bench_gemm_sweep CLI contract", cli_contract)
    run_section(summary, "ladder merge contract", ladder_merge)
    run_section(summary, "paired timing coverage contract", speed_coverage)
    run_section(summary, "attention semantics contract", attention_semantics)
    run_gpu_section(summary, mode, "bench_flash_attn CLI contract", attn_cli_contract)
    run_section(summary, "attention paired contract", attn_paired_contract)
    run_section(summary, "attention tuning contract", tune_knob_contract)
    run_section(summary, "kernel inventory contract", coverage_inventory)
    run_section(summary, "static tile audit contract", tile_audit_contract)
    run_section(summary, "benchmark evidence contract", benchmark_evidence_contract)
    run_section(summary, "execution policy contract", execution_policy_contract)

    print()
    report = summary.as_dict(assertion_failures=len(FAILURES))
    print("HARNESS_SUMMARY " + json.dumps(report, sort_keys=True))
    verdict = summary.verdict(assertion_failures=len(FAILURES))
    if verdict == "FAIL":
        print(
            f"FAIL: {len(FAILURES)} assertion failure(s), "
            f"{len(report['required_gpu_skips'])} required GPU skip(s)"
        )
        for f in FAILURES:
            print(f"  - {f}")
        for name in report["required_gpu_skips"]:
            print(f"  - required GPU section skipped: {name}")
        return 1
    if verdict == "PASS_WITH_SKIPS":
        print(
            "PASS WITH SKIPS: all executed adversarial contracts passed; "
            "the skipped sections above remain unverified"
        )
        return 0
    print("PASS: every adversarial contract ran and passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
