#!/usr/bin/env python3
"""Which kernels does the benchmark suite actually dispatch?

Coverage used to be inferred: grep a kernel's name out of the bench sources.
That was wrong in both directions -- `matmul2d_tensorops_*` is reached through
a dispatcher and looked untimed, while a name in a comment looked timed. Worse,
the inventory itself was wrong: kernels declared through the `NN_COOP_KERNEL`
family of macros have no `kernel void` line, so a source scan missed 16 of 83
entry points outright.

This measures instead. `TESSL_KERNEL_TRACE=1` makes the runtime record every
name passed to `GpuRuntime::pipeline`, which is the single site where a kernel
is selected; each bench binary prints its trace on exit, on every exit path.

  python3 bench/kernel_coverage.py            # report
  python3 bench/kernel_coverage.py --check    # gate: see the exit status below

Exit status (the same on every path; documented in docs/benchmarking.md):
  0   the check ran and every kernel entry point is dispatched
  1   the check ran and found gaps (undispatched kernels, a census that
      disagrees with the metallib, a kernel traced that is not in the census)
  77  the check could not run: no toolchain (`xcrun metal` / `metal-nm`), no
      binary, a stale binary, no metallib, a suite that failed or printed no
      trace. 77 is the repo's SKIPPED status (scripts/check_kernel_ir.sh) and is
      never success, and never the same status as "ran and found gaps".
  2   bad command line (argparse)

The kernel census comes from the real preprocessor: `xcrun metal -E`, with the
flags build.rs's `metal_args` gives every compile, on each kernels/*.metal, and
the entry points are read from the preprocessed text (comments, `#if 0`, and
disabled `#ifdef` blocks are gone; macro stamping, `##` pasting, and
`[[host_name]]` / `template` instantiations are resolved). It is not
reimplemented here, so a toolchain that is missing is "could not run".

The census is cross-checked against the metallib the binaries run. The runtime embeds the
library's bytes (`include_bytes!`), so no binary carries a path to look up;
`target/release/bench_gemm_sweep --dump-metallib PATH` writes those bytes out and
`metal-nm` reads them. Pass `--metallib PATH` to audit an explicit artifact
instead. The build never writes a compatibility copy into the crate source tree.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
KDIR = os.path.join(CRATE, "kernels")

EXIT_GAPS = 1
EXIT_CANNOT_RUN = 77


class CannotRun(SystemExit):
    """The check could not run. Distinct from a run that found gaps: `main`
    exits EXIT_CANNOT_RUN for it, where a plain SystemExit(message) is 1."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.code = EXIT_CANNOT_RUN


def bin_dir():
    """`target/release` of this crate, honouring CARGO_TARGET_DIR."""
    target = os.environ.get("CARGO_TARGET_DIR")
    return os.path.join(os.path.abspath(target) if target else os.path.join(CRATE, "target"),
                        "release")


def newest_input(crate=None):
    """Newest mtime among what the metallib and the binaries are built from."""
    crate = crate or CRATE
    paths = [os.path.join(crate, "build.rs")]
    kdir = os.path.join(crate, "kernels")
    if os.path.isdir(kdir):
        paths += [os.path.join(kdir, f) for f in os.listdir(kdir) if f.endswith((".metal", ".h"))]
    return max((os.path.getmtime(x) for x in paths if os.path.exists(x)), default=0.0)


def require_fresh(binary, crate=None):
    """A bench binary older than the kernel sources or build.rs embeds a metallib
    (and dispatches kernels) the source tree no longer describes."""
    if os.path.getmtime(binary) < newest_input(crate):
        raise CannotRun(
            f"{binary} is older than kernels/*.metal, kernels/*.h or build.rs, so it "
            "embeds a stale metallib; rebuild with `cargo build --release --bins`")


def metal_toolchain():
    try:
        r = subprocess.run(["xcrun", "-f", "metal"], capture_output=True, text=True)
        sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                             capture_output=True, text=True)
    except OSError as exc:
        raise CannotRun(f"kernel census needs `xcrun metal` (Xcode Metal Toolchain): {exc}")
    if r.returncode != 0 or not r.stdout.strip() or sdk.returncode != 0:
        raise CannotRun("kernel census needs `xcrun metal` (Xcode Metal Toolchain); "
                        "no kernel source was preprocessed")
    return r.stdout.strip(), sdk.stdout.strip()


def build_rs_flags(sdk):
    """The flag list build.rs's `metal_args` gives every compile, read from
    build.rs itself (no second copy to drift). `-std` is chosen by the caller."""
    with open(os.path.join(CRATE, "build.rs")) as fh:
        text = fh.read()
    m = re.search(r"fn metal_args\(.*?let mut args: Vec<String> = \[(.*?)\]\s*\.iter\(\)",
                  text, re.S)
    if not m:
        raise CannotRun("could not read build.rs `metal_args`; the preprocessor flags are unknown")
    flags = []
    for lit, ident in re.findall(r'"([^"]*)"|\b(sdk)\b', m.group(1)):
        if ident:
            flags.append(sdk)
        elif "{" not in lit:  # `-std={metal_std}` is chosen per compile
            flags.append(lit)
    if "-isysroot" not in flags or sdk not in flags:
        raise CannotRun(f"build.rs metal_args parsed to {flags}, with no -isysroot SDK")
    return flags


ENTRY = re.compile(
    r"(template\s*(?:<[^<>]*>)?\s*)?((?:\[\[[^\]]*\]\]\s*)*)"
    r"\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)")
HOST_NAME = re.compile(r'host_name\(\s*"([^"]+)"\s*\)')


def entry_points(preprocessed, where="<source>"):
    """Exported kernel names in preprocessed MSL.

    An entry point is `kernel void NAME`; `[[host_name("x")]]` renames it to x
    (what the metallib exports). A `template <params>` kernel is a generic
    definition, not an entry point, until an explicit instantiation
    (`template [[host_name("x")]] kernel void f<...>`) names it.
    """
    out = set()
    for m in ENTRY.finditer(preprocessed):
        tmpl, attrs, name = m.group(1), m.group(2), m.group(3)
        host = HOST_NAME.search(attrs)
        if tmpl and re.search(r"<\s*[^>\s]", tmpl):
            continue  # generic definition
        if tmpl and not host:
            raise SystemExit(f"{where}: template instantiation of `{name}` has no "
                             "[[host_name]], so its exported name cannot be derived")
        out.add(host.group(1) if host else name)
    return out


def preprocess(metal, flags, src):
    """`metal -E` of one source, with build.rs's std fallback (metal4.0 then
    metal3.2). Returns the preprocessed text."""
    errs = []
    with tempfile.TemporaryDirectory(prefix="kernel-census-") as d:
        out = os.path.join(d, "pp.i")
        for std in ("metal4.0", "metal3.2"):
            try:
                r = subprocess.run([metal, "-E", f"-std={std}", *flags, src, "-o", out],
                                   capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CannotRun(f"could not run {metal} -E on {src}: {exc}")
            if r.returncode == 0 and os.path.isfile(out):
                with open(out, errors="replace") as fh:
                    return fh.read()
            errs.append(f"{std}: {r.stderr.strip()[:300]}")
    raise SystemExit(f"{os.path.basename(src)} does not preprocess ({'; '.join(errs)})")


def inventory(kdir=None, metallib=None, require_metallib=False):
    """Every kernel entry point in the shipped kernel sources.

    Read from the real preprocessor's output, so the set is right by
    construction and needs no cross-check to be trusted. `metallib` is still
    compared when it can be read (the library is what the runtime looks a
    pipeline up in), and `require_metallib` makes an unreadable library a
    failure; `--check` always sets it. Raises CannotRun when the toolchain is
    absent: that is never answered with a different scan.
    """
    kdir = kdir or KDIR
    sources = sorted(f for f in os.listdir(kdir) if f.endswith(".metal"))
    if not sources:
        raise SystemExit(f"no kernels found under {kdir}")
    metal, sdk = metal_toolchain()
    flags = build_rs_flags(sdk)
    names, by_file = set(), {}
    for f in sources:
        found = entry_points(preprocess(metal, flags, os.path.join(kdir, f)), f)
        names |= found
        by_file[f[:-6]] = found
    if not names:
        raise SystemExit(f"no kernels found under {kdir}")

    # Cross-check against the compiled metallib, which is ground truth for what
    # the runtime can look up. Any disagreement means the census is wrong.
    exported = metallib_symbols(metallib, required=require_metallib)
    if exported is not None:
        only_lib = sorted(exported - names)
        only_src = sorted(names - exported)
        if only_lib or only_src:
            raise SystemExit(
                "kernel inventory disagrees with the compiled metallib.\n"
                f"  in metallib, missed by the census ({len(only_lib)}): {only_lib[:8]}\n"
                f"  in census, absent from metallib ({len(only_src)}): {only_src[:8]}\n"
                "The census is wrong; fix it rather than measuring coverage against it.")
    return names, by_file


def metallib_symbols(path=None, required=False):
    """Exported kernel names from the built metallib, or None if unavailable.

    `None` and `set()` are deliberately different: a toolchain without
    `metal-nm` must not read as a library with no kernels in it. Set `required`
    for a coverage measurement: a cross-check that could not run is then a
    CannotRun rather than the same PASS as one that actually verified the binary.
    """
    if path is None:
        # The runtime embeds the library's bytes, so the census is read from
        # those bytes, dumped by the binary being measured.
        binary = os.path.join(bin_dir(), "bench_gemm_sweep")
        if not os.path.isfile(binary):
            if required:
                raise CannotRun(
                    f"benchmark binary missing at {binary}; run "
                    "`cargo build --release --bins` before measuring kernel coverage")
            return None
        with tempfile.TemporaryDirectory(prefix="kernel-coverage-lib-") as d:
            try:
                require_fresh(binary)
                return metallib_symbols(dump_embedded_metallib(binary, d), required)
            except SystemExit:
                if required:
                    raise
                return None
    if not os.path.exists(path):
        if required:
            raise CannotRun(
                f"compiled metallib missing at {path}; build the release binaries "
                "before measuring kernel coverage")
        return None
    try:
        r = subprocess.run(["xcrun", "metal-nm", path], capture_output=True, text=True)
    except OSError as exc:
        if required:
            raise CannotRun(f"could not execute xcrun metal-nm: {exc}")
        return None
    if r.returncode != 0:
        if required:
            detail = r.stderr.strip()[:400] or "no diagnostic"
            raise CannotRun(f"metal-nm failed for {path}: {detail}")
        return None
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        # `<addr> T <name>` marks an exported kernel entry point.
        if len(parts) == 3 and parts[1] == "T":
            out.add(parts[2])
    if not out and required:
        raise CannotRun(
            f"metal-nm exported no kernel entry points from {path}; refusing to "
            "treat an unreadable or empty library as verified")
    return out or None


def dump_embedded_metallib(binary, directory):
    """Write the metallib `binary` embeds into `directory`; return its path.

    Raises CannotRun naming the fix when the binary cannot produce it, so a
    stale binary (built before `--dump-metallib` existed) is not mistaken for
    a library with nothing in it.
    """
    out = os.path.join(directory, "embedded.metallib")
    fix = "rebuild with `cargo build --release --bins`"
    try:
        r = subprocess.run([binary, "--dump-metallib", out], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CannotRun(f"could not run {binary} --dump-metallib: {exc}; {fix}")
    if r.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) == 0:
        detail = r.stderr.strip()[:400] or "no diagnostic"
        raise CannotRun(
            f"{binary} --dump-metallib exited {r.returncode} without a library "
            f"({detail}); the binary is stale or embeds no metallib: {fix}")
    return out


# Benchmarks to run, and the arguments that make them cover their own breadth.
# Kept small: this is a coverage probe, not a measurement.
SUITES = [
    ("bench_gemm_sweep", [], dict(BENCH_SHAPES="256x256x256,192x520x1024",
                                  BENCH_ITERS="1", BENCH_WARMUP="0")),
    ("bench_gemm_sweep", ["--dump-parity", "{tmp}/gemmparity"],
     dict(BENCH_PARITY_SHAPE="square_512", BENCH_PARITY_SEEDS="1")),
    # Decode configs reach the KV-split kernels; one prefill config at each
    # head dim reaches the row-parallel ones. Both families are needed: the
    # public entry points route by Tq, so a decode-only run leaves the prefill
    # kernels undispatched.
    ("bench_flash_attn", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0",
                                  BENCH_ATTN_CFGS="swa128_decode_1k,swa256_decode_4k,"
                                                  "global512_decode_4k,swa128_prefill_512,"
                                                  "swa256_prefill_2048,"
                                                  "global512_prefill_1024")),
    # Every decode (lanes-per-key x chunk) pair, and every rows lanes-per-row.
    # Both are compile-time constants, so each combination is its own kernel and
    # sweeping one axis at the other's default would leave the cross product
    # undispatched -- which is exactly what the gate caught.
    *[
        ("bench_flash_attn", [], dict(
            BENCH_ITERS="1", BENCH_WARMUP="0",
            BENCH_ATTN_DECODE_R=r, BENCH_ATTN_DECODE_CHUNK=ch,
            BENCH_ATTN_CFGS="swa128_decode_1k,swa256_decode_4k,global512_decode_4k"))
        for r in ("8", "16", "32")
        for ch in ("64", "128", "256")
    ],
    *[
        ("bench_flash_attn", [], dict(
            BENCH_ITERS="1", BENCH_WARMUP="0", BENCH_ATTN_ROWS_R=r,
            BENCH_ATTN_ROWS_SGT=g,
            BENCH_ATTN_CFGS="swa128_prefill_512,swa256_prefill_2048,"
                            "global512_prefill_1024"))
        for r in ("8", "16", "32")
        for g in ("8", "16", "32")
    ],
    # The tiled kernels remain the A/B baseline and are still dispatched.
    ("bench_flash_attn", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0",
                                  TESSL_ATTN_TILED="1",
                                  BENCH_ATTN_CFGS="swa128_decode_1k,swa256_decode_4k,"
                                                  "global512_decode_4k")),
    ("bench_nn_kernels", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0")),
    # The TN/NT accumulate lanes reach the accumulate kernels with no opt-in
    # since TESSL_GEMM_ACCUM / _ACCUM_DX were retired (2026-10-10).
    ("bench_gemm_variants", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0")),
    # The routing lanes dispatch the parallel bf16 split-K and its
    # accumulating reduce, and the f16 / relaxed-f32 64x64 epilogue kernels,
    # which no other suite reaches. One round, so the drift gate cannot fire.
    ("bench_gemm_routes", [], dict(BENCH_REPS="1", BENCH_ITERS="1",
                                   BENCH_WARMUP="0", BENCH_ROUNDS="1")),
]


def run_suites(tmp):
    """Union of the kernels every suite dispatched, plus the per-suite split."""
    traced, per_suite, failures = set(), {}, []
    for name, argv, env in SUITES:
        binary = os.path.join(bin_dir(), name)
        if not os.path.exists(binary):
            failures.append(f"{name}: not built (cargo build --release --bins)")
            continue
        try:
            require_fresh(binary)
        except CannotRun as exc:
            failures.append(exc.message)
            continue
        argv = [a.replace("{tmp}", tmp) for a in argv]
        e = dict(os.environ, TESSL_KERNEL_TRACE="1", **env)
        r = subprocess.run([binary] + argv, env=e, capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"{name} {' '.join(argv)}: exited {r.returncode}\n"
                            f"{r.stderr.strip()[:400]}")
            continue
        line = [x for x in r.stderr.splitlines() if x.startswith("KERNEL_TRACE ")]
        if not line:
            # An empty trace and a missing trace must not read the same.
            failures.append(f"{name}: emitted no KERNEL_TRACE line; is the runtime "
                            "honouring TESSL_KERNEL_TRACE?")
            continue
        got = {x for x in line[-1][len("KERNEL_TRACE "):].split(",") if x}
        tag = " ".join(f"{k}={v}" for k, v in sorted(env.items())
                       if k.startswith("TESSL_") or k in ("BENCH_ATTN_ROWS_R",
                                 "BENCH_ATTN_ROWS_SGT", "BENCH_ATTN_DECODE_CHUNK",
                                 "BENCH_ATTN_DECODE_R"))
        key = f"{name}{' ' + argv[0] if argv else ''}{' ' + tag if tag else ''}"
        per_suite[key] = got
        traced |= got
    return traced, per_suite, failures


def main():
    try:
        run()
    except CannotRun as exc:
        print(f"COULD NOT RUN: {exc.message}", file=sys.stderr)
        sys.exit(EXIT_CANNOT_RUN)


def run():
    ap = argparse.ArgumentParser(
        epilog="exit status: 0 ran and every kernel is dispatched; 1 ran and found "
               "gaps (or a census/metallib disagreement); 77 could not run (no "
               "toolchain, binary, metallib or trace, a stale binary, or a failed "
               "suite); 2 bad command line.")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any kernel is never dispatched (77 if the check cannot run)")
    ap.add_argument(
        "--metallib",
        help="explicit compiled metallib to inventory (default: the library bench_gemm_sweep embeds, via --dump-metallib)",
    )
    ap.add_argument("--out")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="kernel-coverage-")
    try:
        # A command-line coverage result is evidence about the shipped binary,
        # not just a source grep. If the compiled-library census cannot run,
        # stop before executing suites or printing a percentage.
        names, by_file = inventory(metallib=args.metallib, require_metallib=True)
        traced, per_suite, failures = run_suites(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        for f in failures:
            print(f"ERROR  {f}", file=sys.stderr)
        raise CannotRun("coverage run incomplete; refusing to report a partial "
                        "sweep as a coverage measurement")

    # A traced name absent from the inventory means the inventory is wrong --
    # exactly the failure this tool exists to prevent.
    stray = sorted(traced - names)
    if stray:
        raise SystemExit(
            f"dispatched {len(stray)} kernel(s) absent from the source inventory: "
            f"{stray[:8]}. The inventory is undercounting; fix it before reading "
            "any coverage number from it.")

    covered = sorted(traced)
    missing = sorted(names - traced)
    print(f"kernel entry points : {len(names)}")
    print(f"dispatched by a bench: {len(covered)}  ({100 * len(covered) / len(names):.0f}%)")
    print(f"never dispatched     : {len(missing)}\n")
    for suite, got in per_suite.items():
        print(f"  {suite:<40}{len(got):>4} kernels")
    if missing:
        print("\nuncovered, by file:")
        for f, ks in sorted(by_file.items()):
            gap = sorted(ks - traced)
            if gap:
                print(f"  {f:<24}{len(gap):>3}  {', '.join(gap[:4])}"
                      f"{' ...' if len(gap) > 4 else ''}")
    doc = dict(entry_points=len(names), covered=covered, missing=missing,
               per_suite={k: sorted(v) for k, v in per_suite.items()})
    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(doc, indent=2))
    if args.check and missing:
        raise SystemExit(f"\n{len(missing)} kernel(s) have no timing lane")  # EXIT_GAPS
    print("\nPASS: every kernel entry point is dispatched by a benchmark"
          if not missing else "")


if __name__ == "__main__":
    main()
