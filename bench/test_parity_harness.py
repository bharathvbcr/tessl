#!/usr/bin/env python3
"""Adversarial tests for the benchmark tooling that is still Python.

The GEMM parity scorer, the external-lane contract, the ladder merge and the
`bench_gemm_sweep` / `bench_flash_attn`-style CLI contracts moved to Rust
(`tests/gemm_parity.rs`, `tests/bench_gemm_sweep_cli.rs`); the attention masking
and external-lane writer checks live in `bench/test_comparator_lanes.py`. What
is left here is waiting for its own port.

The bar each case defends is the same one: a check that could not run must
never report the same result as a check that ran and passed.

  python3 bench/test_parity_harness.py               # best effort GPU probes
  python3 bench/test_parity_harness.py --pure        # never launch a GPU binary
  python3 bench/test_parity_harness.py --require-gpu # a skipped GPU lane is fatal
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE_BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE_BENCH)

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

    run_gpu_section(summary, mode, "bench_flash_attn CLI contract", attn_cli_contract)
    run_section(summary, "attention tuning contract", tune_knob_contract)
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
