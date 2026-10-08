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
  python3 bench/kernel_coverage.py --check    # non-zero if anything is uncovered

By default the compiled library is resolved from the absolute `OUT_DIR` path
embedded in `target/release/bench_gemm_sweep`. Pass `--metallib PATH` to audit
an explicit artifact. The build never writes a compatibility copy into the
crate source tree.
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
BIN = os.path.join(CRATE, "target", "release")

PLAIN = re.compile(r'^kernel void ([A-Za-z0-9_]+)', re.M)
MACRO = re.compile(r'^([A-Z][A-Z0-9_]*KERNEL)\(\s*([A-Za-z0-9_]+)', re.M)
# Any line that looks like a kernel-declaring macro must match MACRO's shape.
# A new macro family that this does not parse is a silent undercount, which is
# the bug that produced the original 67-vs-83 error.
MACROISH = re.compile(r'^([A-Z][A-Z0-9_]*KERNEL)\(', re.M)

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
    ("bench_gemm_variants", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0")),
    # The native TN/NT accumulate kernels are behind an opt-in: without it the
    # accumulate lanes take a temp-buffer + add_inplace fallback and the real
    # kernels are never dispatched.
    ("bench_gemm_variants", [], dict(BENCH_ITERS="1", BENCH_WARMUP="0",
                                     TESSL_GEMM_ACCUM="1", TESSL_GEMM_ACCUM_DX="1")),
]


def strip_macro_bodies(text):
    """Blank out `#define` continuation blocks, keeping line count stable."""
    out, in_define = [], False
    for line in text.splitlines():
        starts = line.lstrip().startswith("#define")
        if starts or in_define:
            out.append("")
            in_define = line.rstrip().endswith("\\")
        else:
            out.append(line)
    return "\n".join(out)


def inventory(kdir=None, metallib=None, require_metallib=False):
    """Every kernel entry point declared in the shipped kernel sources.

    `kdir` is a parameter so the guard below is testable: a macro family this
    does not parse must fail loudly, and that is the single behaviour standing
    between a correct inventory and the silent 67-vs-83 undercount.
    """
    kdir = kdir or KDIR
    names, by_file = set(), {}
    for f in sorted(os.listdir(kdir)):
        if not f.endswith(".metal"):
            continue
        text = open(os.path.join(kdir, f)).read()
        # A macro *body* can contain `kernel void NAME(` at line start, where
        # NAME is the macro parameter rather than a kernel. Scanning it yields a
        # phantom entry point that exists in no metallib. Strip `#define` blocks
        # -- which continue while lines end in a backslash -- before scanning;
        # the invocations below the block are what name the real kernels.
        text = strip_macro_bodies(text)
        found = set(PLAIN.findall(text)) | {m[1] for m in MACRO.findall(text)}
        seen_macro = {m for m in MACROISH.findall(text)}
        parsed_macro = {m[0] for m in MACRO.findall(text)}
        unparsed = seen_macro - parsed_macro
        if unparsed:
            raise SystemExit(
                f"{f}: kernel-declaring macro(s) {sorted(unparsed)} present but not "
                "parsed by MACRO. Extend the pattern rather than undercounting -- "
                "a missed macro family silently shrinks the inventory.")
        names |= found
        by_file[f[:-6]] = found
    if not names:
        raise SystemExit(f"no kernels found under {kdir}")

    # Cross-check against the compiled metallib, which is ground truth: it is
    # what the runtime can actually look a pipeline up in. The source scan is
    # kept because it attributes each kernel to a file, but any disagreement
    # means the scan is wrong -- a macro form it cannot parse, or one whose
    # first argument is not the kernel name.
    exported = metallib_symbols(metallib, required=require_metallib)
    if exported is not None:
        only_lib = sorted(exported - names)
        only_src = sorted(names - exported)
        if only_lib or only_src:
            raise SystemExit(
                "kernel inventory disagrees with the compiled metallib.\n"
                f"  in metallib, missed by the source scan ({len(only_lib)}): {only_lib[:8]}\n"
                f"  in source scan, absent from metallib ({len(only_src)}): {only_src[:8]}\n"
                "The scan is wrong; fix it rather than measuring coverage against it.")
    return names, by_file


def metallib_symbols(path=None, required=False):
    """Exported kernel names from the built metallib, or None if unavailable.

    `None` and `set()` are deliberately different: a toolchain without
    `metal-nm` must not read as a library with no kernels in it. Set `required`
    for a coverage measurement: a cross-check that could not run is then a hard
    failure rather than the same PASS as one that actually verified the binary.
    """
    if path is None:
        # The runtime embeds build.rs's immutable OUT_DIR artifact. Resolving
        # that exact path keeps the census tied to the binaries being measured
        # and avoids relying on a stale, source-tree `default.metallib` copy.
        binary = os.path.join(BIN, "bench_gemm_sweep")
        if not os.path.isfile(binary):
            if required:
                raise SystemExit(
                    f"benchmark binary missing at {binary}; run "
                    "`cargo build --release --bins` before measuring kernel coverage")
            return None
        from benchmark_evidence import embedded_metallib_path
        try:
            path = embedded_metallib_path(binary)
        except ValueError as exc:
            if required:
                raise SystemExit(f"cannot resolve the benchmark's embedded metallib: {exc}")
            return None
    if not os.path.exists(path):
        if required:
            raise SystemExit(
                f"compiled metallib missing at {path}; build the release binaries "
                "before measuring kernel coverage")
        return None
    try:
        r = subprocess.run(["xcrun", "metal-nm", path], capture_output=True, text=True)
    except OSError as exc:
        if required:
            raise SystemExit(f"could not execute xcrun metal-nm: {exc}")
        return None
    if r.returncode != 0:
        if required:
            detail = r.stderr.strip()[:400] or "no diagnostic"
            raise SystemExit(f"metal-nm failed for {path}: {detail}")
        return None
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        # `<addr> T <name>` marks an exported kernel entry point.
        if len(parts) == 3 and parts[1] == "T":
            out.add(parts[2])
    if not out and required:
        raise SystemExit(
            f"metal-nm exported no kernel entry points from {path}; refusing to "
            "treat an unreadable or empty library as verified")
    return out or None


def run_suites(tmp):
    """Union of the kernels every suite dispatched, plus the per-suite split."""
    traced, per_suite, failures = set(), {}, []
    for name, argv, env in SUITES:
        binary = os.path.join(BIN, name)
        if not os.path.exists(binary):
            failures.append(f"{name}: not built (cargo build --release --bins)")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any kernel is never dispatched")
    ap.add_argument(
        "--metallib",
        help="explicit compiled metallib to inventory (default: path embedded in the release binary)",
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
        raise SystemExit("coverage run incomplete; refusing to report a partial "
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
        raise SystemExit(f"\n{len(missing)} kernel(s) have no timing lane")
    print("\nPASS: every kernel entry point is dispatched by a benchmark"
          if not missing else "")


if __name__ == "__main__":
    main()
