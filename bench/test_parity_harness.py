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
import contextlib  # noqa: E402
import io  # noqa: E402

import gemm_sweep_mlx  # noqa: E402
from gemm_sweep_mlx import adjudicate, parity as _parity  # noqa: E402


def _exact_runtime(a, b, dtype):
    """Stand-in comparison runtime: the float64 product rounded once to f32.

    The scorer-contract fixtures must not depend on which of MLX / torch-MPS is
    installed or on how the local GPU rounds an f32 matmul. On this machine MLX
    0.32.3's GPU f32 matmul errs by ~3.5e-3 absolute where its CPU stream errs by
    ~7e-7 (it is reduced precision), so a live MLX lane busts the f32 budget by
    ~600x on a perfectly valid dump -- and which interpreter happened to have
    `mlx` importable decided whether this suite passed.
    """
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32), "f32"


# Same lane names and scoring path as the live ones, deterministic outputs.
STUB_EXTERNAL_LANES = [(lane, _exact_runtime, "f32")
                       for lane in ("mlx-f32", "torch-mps-f32")]


def parity(parity_dir):
    return _parity(parity_dir, external_lanes=STUB_EXTERNAL_LANES)


def _bf16(x):
    """Round float32 to bfloat16 (nearest-even) and return it as float64."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def _bf16_runtime(a, b, dtype):
    """Stand-in bf16 runtime: bf16 operands, exact product, bf16 result."""
    c = _bf16(a) @ _bf16(b)
    return _bf16(c.astype(np.float32)).astype(np.float32), "bf16"


BF16_STUB_LANES = [("mlx-bf16", _bf16_runtime, "bf16"),
                   ("torch-mps-bf16", _bf16_runtime, "bf16")]


def _tf32(x):
    """Round float32 to tf32 (10 explicit mantissa bits, nearest-even), as float64."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def _tf32_runtime(a, b, dtype):
    """Stand-in for MLX's measured GPU f32 matmul: tf32 operands, exact product."""
    return (_tf32(a) @ _tf32(b)).astype(np.float32), "f32"


def mlx_f32_precision_class(tmp):
    """`mlx-f32` is scored as the tf32-class lane it measures as, not f32-exact."""
    good = build_dump(os.path.join(tmp, "mlx_tf32"), seeds=3)
    lanes = [("mlx-f32", _tf32_runtime, "f32"), ("torch-mps-f32", _exact_runtime, "f32")]

    print("\n-- mlx-f32: tf32-class error is inside its budget --")
    rows = check("tf32-class mlx-f32 stub", lambda: _parity(good, external_lanes=lanes),
                 expect_reject=False)
    if rows is not None:
        print("  ok    tf32-class mlx-f32 scored without abort")

    print("\n-- mlx-f32: the same error against an f32-exact claim aborts --")
    real = gemm_sweep_mlx.LANE_PRECISION["mlx-f32"]
    gemm_sweep_mlx.LANE_PRECISION["mlx-f32"] = (0.0, 0.0)
    try:
        check("f32-exact claim for mlx-f32", lambda: _parity(good, external_lanes=lanes),
              True, "mlx-f32")
    finally:
        gemm_sweep_mlx.LANE_PRECISION["mlx-f32"] = real

    print("\n-- torch-mps-f32 stays f32-exact: a tf32-class torch lane aborts --")
    bad = [("mlx-f32", _exact_runtime, "f32"), ("torch-mps-f32", _tf32_runtime, "f32")]
    check("tf32-class torch-mps-f32 stub", lambda: _parity(good, external_lanes=bad),
          True, "torch-mps-f32")


def external_lane_contracts(tmp):
    """`parity(external_lanes=...)`: validation, skip/abort paths, defaults."""
    good = build_dump(os.path.join(tmp, "ext"), seeds=3)
    both = STUB_EXTERNAL_LANES + BF16_STUB_LANES

    def run(lanes):
        return _parity(good, external_lanes=lanes)

    print("\n-- external_lanes: bf16 stubs are scored through the out_u budget --")
    rows = check("f32 + bf16 stub lanes", lambda: run(both), expect_reject=False)
    if rows is not None:
        got = {r["lane"]: r for r in rows}
        for lane in ("mlx-bf16", "torch-mps-bf16"):
            r = got.get(lane)
            if r is None or r.get("out_dtype") != "bf16" or r.get("seeds") != 3:
                FAILURES.append(f"{lane} stub not scored as bf16 on 3 seeds: {r}")
                print(f"  FAIL  {lane} stub not scored")
            else:
                print(f"  ok    {lane} scored as bf16 on 3 seeds, "
                      f"budget {r['max_higham_ratio']:.3f}x")
        if any(r.get("external_lanes_scored") != 4 for r in rows if "skipped" not in r):
            FAILURES.append("rows do not report 4 comparison runtimes scored")
            print("  FAIL  external_lanes_scored != 4")

    print("\n-- external_lanes: a lane that raises on every seed is a visible skip --")
    def boom(a, b, dt):
        raise RuntimeError("stub boom")
    rows = check("raises on every seed",
                 lambda: run([("mlx-f32", boom, "f32"), STUB_EXTERNAL_LANES[1]]),
                 expect_reject=False)
    if rows is not None:
        r = next((x for x in rows if x["lane"] == "mlx-f32"), None)
        if not r or "stub boom" not in r.get("skipped", "") or r.get("seeds") != 0:
            FAILURES.append(f"skip reason not surfaced: {r}")
            print(f"  FAIL  skip reason not surfaced: {r}")
        else:
            print("  ok    skipped row carries the reason 'stub boom'")

    print("\n-- external_lanes: partial-seed and dtype-flip lanes abort --")
    calls = {"n": 0}
    def boom_on_third(a, b, dt):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("stub seed-2 failure")
        return _exact_runtime(a, b, dt)
    check("raises on seed 2 only",
          lambda: run([("mlx-f32", boom_on_third, "f32")]), True,
          "cover every seed")
    flips = {"n": 0}
    def flip(a, b, dt):
        flips["n"] += 1
        c, _ = _exact_runtime(a, b, dt)
        return c, ("f32" if flips["n"] == 1 else "bf16")
    check("dtype label flips between seeds",
          lambda: run([("mlx-f32", flip, "f32")]), True, "dtype changed")

    print("\n-- external_lanes: malformed lists are refused up front --")
    ok = STUB_EXTERNAL_LANES[0]
    for label, lanes, needle in (
        ("duplicate name", [ok, ok], "more than once"),
        ("collides with a dumped lane", [("tensorops-f32", _exact_runtime, "f32")],
         "collides"),
        ("unknown name", [("cuda-f32", _exact_runtime, "f32")], "LANE_PRECISION"),
        ("not a 3-tuple", [("mlx-f32", _exact_runtime)], "3-tuple"),
        ("fn not callable", [("mlx-f32", None, "f32")], "callable"),
        ("bad dtype", [("mlx-f32", _exact_runtime, "f16")], "dtype"),
        ("not iterable", 7, "list of"),
    ):
        check(f"external_lanes {label}", lambda l=lanes: run(l), True, needle)

    print("\n-- external_lanes: None means the live EXTERNAL_LANES, by identity --")
    sentinel = {"called": 0}
    def stub_live(a, b, dt):
        sentinel["called"] += 1
        return _exact_runtime(a, b, dt)
    real = gemm_sweep_mlx.EXTERNAL_LANES
    gemm_sweep_mlx.EXTERNAL_LANES = [("mlx-f32", stub_live, "f32")]
    try:
        rows = check("external_lanes=None", lambda: _parity(good), expect_reject=False)
    finally:
        gemm_sweep_mlx.EXTERNAL_LANES = real
    if rows is not None:
        if sentinel["called"] == 3:
            print("  ok    None resolved to the module's EXTERNAL_LANES (3 seeds)")
        else:
            FAILURES.append(f"None did not use EXTERNAL_LANES: {sentinel}")
            print(f"  FAIL  None did not use EXTERNAL_LANES: {sentinel}")

    print("\n-- external_lanes=[]: loud note, and no claimed corroboration --")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rows = check("empty external_lanes", lambda: run([]), expect_reject=False)
    if rows is not None:
        if "NO COMPARISON RUNTIMES SCORED" not in err.getvalue():
            FAILURES.append("empty external_lanes printed no loud note")
            print("  FAIL  no loud note for empty external_lanes")
        elif any(r.get("external_lanes_scored") != 0 for r in rows):
            FAILURES.append("rows claim comparison runtimes that were not scored")
            print("  FAIL  rows claim external runtimes")
        else:
            print("  ok    loud note printed; rows report external_lanes_scored=0")
    d = build_dump(os.path.join(tmp, "ext_drift"))
    for sd in os.listdir(d):
        if sd.startswith("seed_"):
            fn = os.path.join(d, sd, "parity_c_tensorops-f32.npy")
            np.save(fn, (np.load(fn).astype(np.float64) * (1 + 1e-5)).astype(np.float32))
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _parity(d, external_lanes=[])
        FAILURES.append("drifted lane accepted with no comparison runtimes")
        print("  FAIL  drifted lane accepted")
    except SystemExit as exc:
        msg = str(exc)
        if "unconfirmed" in msg and "stayed inside" not in msg:
            print("  ok    tessl-only breach with no externals is called unconfirmed")
        else:
            FAILURES.append(f"verdict claims corroboration it lacks: {msg}")
            print(f"  FAIL  verdict claims corroboration: {msg}")
    if "stayed inside" in adjudicate({"tensorops-f32": 2.0}, externals_scored=False):
        FAILURES.append("adjudicate claims externals stayed inside with none scored")
        print("  FAIL  adjudicate(externals_scored=False) claims corroboration")

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

        # Macro families that stamp kernels under names the old pattern could not
        # see: a name not ending in KERNEL, `##` pasting, and a macro that stamps
        # through another macro (the shapes of the 29 qwen35 kernels it missed).
        d = write("#define STAMP(NAME, T) \\\n"
                  "kernel void NAME(device T* x) {}\n"
                  "#define STAMP_RULE(R, T) \\\n"
                  "    STAMP(fam_##R##_a, T) \\\n"
                  "    STAMP(fam_##R##_b, T)\n"
                  "STAMP(plain_stamp, float)\n"
                  "STAMP_RULE(kahan, bfloat)\n"
                  "STAMP_RULE(sr,\n"
                  "           float)\n")
        names, _ = inventory(d, metallib="/nonexistent")
        want = {"plain_stamp", "fam_kahan_a", "fam_kahan_b", "fam_sr_a", "fam_sr_b"}
        if names == want:
            print("  ok    defined macro families are expanded (non-KERNEL names, ##, nesting)")
        else:
            FAILURES.append(f"macro expansion got {sorted(names)}")
            print(f"  FAIL  macro expansion got {sorted(names)}")

        # A binary that cannot dump its metallib (stale, built before
        # `--dump-metallib`) is a hard failure naming the fix, never a library
        # with nothing in it.
        from kernel_coverage import dump_embedded_metallib
        try:
            dump_embedded_metallib("/usr/bin/true", tmp)
        except SystemExit as exc:
            if "cargo build --release --bins" in str(exc):
                print("  ok    a binary that dumps no metallib fails, naming the rebuild")
            else:
                FAILURES.append(f"stale-binary error does not name the fix: {exc}")
                print(f"  FAIL  stale-binary error: {exc}")
        else:
            FAILURES.append("a binary that dumped no metallib was accepted")
            print("  FAIL  dump-less binary accepted")

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
    kernels = {"matmul_ok": (32, 64, 4, "constexpr")}
    tiles = {"TILE_OK": (32, 64, 4)}
    cases = [
        ("matching geometry", kernels, tiles, "matmul_ok", "TILE_OK", False),
        ("missing kernel", kernels, tiles, "matmul_missing", "TILE_OK", True),
        ("missing Rust tile", kernels, tiles, "matmul_ok", "TILE_MISSING", True),
        ("kernel with no geometry", {"matmul_ok": (None, None, None, "no compile-time SM/SN")},
         tiles, "matmul_ok", "TILE_OK", True),
    ]
    for name, kt, rt, kernel, tile, expect_error in cases:
        error = audit.geometry_error(kt, rt, kernel, tile)
        if (error is not None) != expect_error:
            FAILURES.append(f"tile audit {name}: unexpected result {error!r}")
            print(f"  FAIL  {name}: {error!r}")
        else:
            print(f"  ok    {name}: {error or 'accepted'}")
    for name, tile, expect_match in [
        ("agreeing geometry", (32, 64, 4), True),
        ("different tile rows", (64, 64, 4), False),
        ("different simdgroups", (32, 64, 2), False),
    ]:
        if audit.geometry_matches(kernels["matmul_ok"], tile) != expect_match:
            FAILURES.append(f"tile audit {name}: wrong verdict")
            print(f"  FAIL  {name}")
        else:
            print(f"  ok    {name}: {'matches' if expect_match else 'refused'}")


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
        external_lane_contracts(tmp)
        mlx_f32_precision_class(tmp)
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
    run_section(summary, "attention semantics contract", attention_semantics)
    run_gpu_section(summary, mode, "bench_flash_attn CLI contract", attn_cli_contract)
    run_section(summary, "attention tuning contract", tune_knob_contract)
    run_section(summary, "kernel inventory contract", coverage_inventory)
    run_section(summary, "static tile audit contract", tile_audit_contract)
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
