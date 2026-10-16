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


def coverage_adversarial():
    """`bench/test_kernel_coverage_adversarial.py`: phantom and missed kernels,
    exit statuses, stale-binary refusal. A subprocess, so its unittest run is
    one section here; any failure fails the harness."""
    print("\n-- kernel coverage adversarial contract --")
    r = subprocess.run([sys.executable, "-I",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "test_kernel_coverage_adversarial.py")],
                       capture_output=True, text=True, timeout=600)
    tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    if r.returncode == 0:
        print(f"  ok    adversarial tests passed ({tail})")
    else:
        FAILURES.append(f"kernel coverage adversarial tests failed:\n{r.stderr[-1500:]}")
        print(f"  FAIL  adversarial tests: exit {r.returncode}\n{r.stderr[-1500:]}")


def metallib_unavailable(reason, hosted=None):
    """The metallib cross-check could not run. Locally that is a visible partial
    skip. On the hosted macOS job (TESSL_HOSTED=1; the runner has Xcode and has
    just built the metallib) it is a misconfiguration, so it fails instead of
    reading as a pass."""
    if hosted is None:
        hosted = bool(os.environ.get("TESSL_HOSTED"))
    if hosted:
        FAILURES.append(f"{reason} (TESSL_HOSTED is set: a skipped cross-check is a failure)")
        print(f"  FAIL  {reason}; TESSL_HOSTED is set")
        return None
    print(f"  SKIP  {reason}")
    return PartialSkip("compiled metallib cross-check", reason)


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
                  "#define NN_COOP_KERNEL(NAME, T, A, B, C, D) kernel void NAME(device T* x) {}\n"
                  "#define TN_NT_COOP_KERNEL(NAME, T, A, B) kernel void NAME(device T* x) {}\n"
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

        # A source the preprocessor rejects is a failed census, never a smaller one.
        d = write("kernel void alpha(device float* x) {}\n"
                  'WEIRD_NEW_KERNEL("delta", bfloat)\n#error not a kernel source\n')
        try:
            inventory(d, metallib="/nonexistent")
        except SystemExit as exc:
            if "does not preprocess" in str(exc):
                print("  ok    a source that does not preprocess fails loudly")
            else:
                FAILURES.append(f"unpreprocessable source raised the wrong error: {exc}")
                print(f"  FAIL  wrong error: {exc}")
        else:
            FAILURES.append("a source that does not preprocess was silently undercounted")
            print("  FAIL  unpreprocessable source silently undercounted")

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
            if "metallib missing" in str(exc) and getattr(exc, "code", None) == 77:
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
            partial_skip = metallib_unavailable("metal-nm unavailable; metallib cross-check unverified")
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

        # TESSL_HOSTED turns the metal-nm skip into a failure; without it, a visible skip.
        before = len(FAILURES)
        with contextlib.redirect_stdout(io.StringIO()):
            skipped = metallib_unavailable("probe", hosted=False)
            failed_quietly = skipped is None or len(FAILURES) != before
            after_skip = len(FAILURES)
            hosted_result = metallib_unavailable("probe", hosted=True)
        if failed_quietly or hosted_result is not None or len(FAILURES) != after_skip + 1:
            FAILURES.append("metallib_unavailable does not separate a local skip from a hosted failure")
            print("  FAIL  hosted/local metallib skip policy")
        else:
            del FAILURES[after_skip:]
            print("  ok    a missing metal-nm is a visible skip locally and a failure when TESSL_HOSTED is set")

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
    run_section(summary, "kernel inventory contract", coverage_inventory)
    run_section(summary, "kernel coverage adversarial contract", coverage_adversarial)
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
