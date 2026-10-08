#!/usr/bin/env python3
"""Numeric-parity check across the whole shape ladder, one shape at a time.

Dumping every shape at once is ~12 GB of .npy at eight draws, so this drives
`bench_gemm_sweep --dump-parity` per label: dump -> score -> delete. Peak disk
is one shape's worth (~3.5 GB at the widest), and the per-shape results merge
into a single artifact.

Why a grid and not one cell: GEMM forward error grows with K, so one shape is a
floor rather than a bound -- and the operand *distribution* moves the fraction
of the error budget a lane consumes by up to 15x, in the opposite direction from
shape. bf16 sits at 0.076x of budget on uniform operands and 0.936x on
heavy-tailed ones. Either axis alone misleads.

  python3 bench/parity_ladder.py --dists all --seeds 4 \
      --out bench/results/gemm_parity_grid_m5pro.json
"""
import argparse, json, os, shutil, statistics, subprocess, sys, tempfile
import numpy as np  # noqa: F401  (imported so a missing numpy fails here, not mid-sweep)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemm_sweep_mlx import SHAPES, parity  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BINARY = os.path.join(HERE, "target", "release", "bench_gemm_sweep")


DISTS = ["uniform", "normal", "log_uniform", "near_cancel", "heavy_tail"]


def run_shape(binary, label, dist, seeds, keep_dir=None):
    """Dump one (shape, distribution), score it, delete the .npy."""
    tmp = keep_dir or tempfile.mkdtemp(prefix=f"parity-{label}-{dist}-")
    try:
        env = dict(os.environ, BENCH_PARITY_SHAPE=label, BENCH_PARITY_DIST=dist,
                   BENCH_PARITY_SEEDS=str(seeds))
        # BENCH_SHAPES is rejected outright by the dump path; drop an inherited
        # one rather than have the whole sweep die on the caller's shell state.
        env.pop("BENCH_SHAPES", None)
        r = subprocess.run([binary, "--dump-parity", tmp], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"{label}/{dist}: bench_gemm_sweep exited {r.returncode}\n"
                f"{r.stderr.strip()}")

        rows = parity(tmp)

        # Cross-check the Rust ladder against the Python mirror of it. These are
        # two hand-maintained lists; if they drift, every number below is being
        # attributed to the wrong shape.
        with open(os.path.join(tmp, "parity_manifest.json")) as f:
            man = json.load(f)
        want = next((s for s in SHAPES if s[3] == label), None)
        if want is None:
            raise SystemExit(f"{label} is not in the Python SHAPES mirror")
        if man["shape"] != label or (man["m"], man["n"], man["k"]) != want[:3]:
            raise SystemExit(
                f"ladder drift: Rust dumped {man['shape']} "
                f"{man['m']}x{man['n']}x{man['k']}, Python expected "
                f"{label} {want[0]}x{want[1]}x{want[2]}")
        # Same class of check on the distribution axis: a silently ignored
        # BENCH_PARITY_DIST would attribute uniform numbers to heavy_tail.
        if man.get("dist") != dist:
            raise SystemExit(
                f"distribution drift: asked for {dist!r}, dump reports "
                f"{man.get('dist')!r}")
        for row in rows:
            row["shape"], row["dist"] = label, dist
            row["m"], row["n"], row["k"] = want[0], want[1], want[2]
        return rows
    finally:
        if keep_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def merge(per_shape, cells):
    """Worst across every (shape, distribution) cell, per lane.

    `cells` is the full grid that was requested. A lane must cover all of it or
    none of it -- the same rule the seed axis already enforces, one level up.
    """
    lanes = []
    for row in per_shape:
        if row["lane"] not in lanes:
            lanes.append(row["lane"])
    # `tuple("square_512")` silently becomes a tuple of characters, so a caller
    # that passes bare labels instead of (shape, dist) pairs would compare
    # against nonsense rather than fail. Reject the shape, do not coerce it.
    norm = set()
    for c in cells:
        if isinstance(c, str) or len(tuple(c)) != 2:
            raise SystemExit(
                f"merge() expects (shape, dist) pairs, got {c!r}. A bare label "
                "would be split into characters and silently match nothing.")
        norm.add(tuple(c))
    cells = norm
    out = []
    for lane in lanes:
        rows = [r for r in per_shape if r["lane"] == lane]
        scored = [r for r in rows if "skipped" not in r]
        covered = {(r["shape"], r.get("dist", "uniform")) for r in rows}
        if covered != cells:
            raise SystemExit(
                f"lane {lane} is missing from {sorted(cells - covered)}; "
                "a lane must cover the whole grid or none of it")
        if not scored:
            out.append(dict(lane=lane, skipped=rows[0]["skipped"], cells=0))
            continue
        if len(scored) != len(rows):
            # Half a grid averaged into one number reads as a full sweep.
            missing = sorted({(r["shape"], r.get("dist")) for r in rows} -
                             {(r["shape"], r.get("dist")) for r in scored})
            raise SystemExit(
                f"lane {lane} scored on {len(scored)}/{len(rows)} cells, "
                f"absent on {missing}. A lane must cover the whole grid or none.")
        dtypes = {r["out_dtype"] for r in scored}
        if len(dtypes) != 1:
            raise SystemExit(f"lane {lane}: output dtype varies across cells: {dtypes}")
        worst = max(scored, key=lambda r: r["max_rel_err"])
        # The budget ratio is the one that decides whether a lane is *correct*;
        # the others describe how wrong it is when it is still inside.
        wb = max(scored, key=lambda r: r["max_higham_ratio"])
        we = max(scored, key=lambda r: r["max_elem_rel_err"])
        out.append(dict(
            lane=lane, out_dtype=dtypes.pop(),
            cells=len(scored), seeds_per_cell=scored[0]["seeds"],
            worst_rel_err=worst["max_rel_err"],
            worst_shape=worst["shape"], worst_dist=worst.get("dist"),
            best_rel_err=min(r["max_rel_err"] for r in scored),
            median_cell_rel_err=statistics.median(r["max_rel_err"] for r in scored),
            worst_higham_ratio=wb["max_higham_ratio"],
            worst_higham_shape=wb["shape"], worst_higham_dist=wb.get("dist"),
            worst_elem_rel_err=we["max_elem_rel_err"],
            worst_elem_shape=we["shape"], worst_elem_dist=we.get("dist"),
            per_cell_rel_err={f"{r['shape']}/{r.get('dist')}": r["max_rel_err"]
                              for r in scored},
            per_cell_higham={f"{r['shape']}/{r.get('dist')}": r["max_higham_ratio"]
                             for r in scored}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--shapes", default="all",
                    help="comma-separated ladder labels, or 'all'")
    ap.add_argument("--dists", default="uniform",
                    help=f"comma-separated operand distributions, or 'all'. "
                         f"Known: {','.join(DISTS)}")
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.seeds < 1:
        raise SystemExit(f"--seeds must be >= 1, got {args.seeds}")
    if not os.path.exists(args.binary):
        raise SystemExit(
            f"{args.binary} not built. "
            "cargo build --release --bin bench_gemm_sweep")

    known = [s[3] for s in SHAPES]
    labels = known if args.shapes == "all" else [x.strip() for x in args.shapes.split(",")]
    unknown = [x for x in labels if x not in known]
    if unknown:
        raise SystemExit(f"unknown ladder label(s) {unknown}; expected from {known}")
    dists = DISTS if args.dists == "all" else [x.strip() for x in args.dists.split(",")]
    bad = [x for x in dists if x not in DISTS]
    if bad:
        raise SystemExit(f"unknown distribution(s) {bad}; expected from {DISTS}")

    cells = [(lab, d) for lab in labels for d in dists]
    per_shape = []
    for i, (label, dist) in enumerate(cells, 1):
        print(f"\n===== [{i}/{len(cells)}] {label} / {dist} x {args.seeds} seeds =====",
              file=sys.stderr)
        per_shape += run_shape(args.binary, label, dist, args.seeds)

    per_lane = merge(per_shape, cells)
    doc = dict(ladder=labels, dists=dists, seeds=args.seeds,
               cells=[list(c) for c in cells], per_lane=per_lane, per_shape=per_shape)

    print(f"\n{'lane':<20}{'out':<6}{'worst norm':>12}{'worst elem':>12}"
          f"{'worst budget':>14}  budget worst cell", file=sys.stderr)
    for r in per_lane:
        if "skipped" in r:
            print(f"{r['lane']:<20}SKIPPED  {r['skipped'][:48]}", file=sys.stderr)
            continue
        print(f"{r['lane']:<20}{r['out_dtype']:<6}{r['worst_rel_err']:>12.2e}"
              f"{r['worst_elem_rel_err']:>12.2e}{r['worst_higham_ratio']:>13.3f}x"
              f"  {r['worst_higham_shape']}/{r['worst_higham_dist']}", file=sys.stderr)

    text = json.dumps(doc, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nwrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
