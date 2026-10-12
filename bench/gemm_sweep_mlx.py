#!/usr/bin/env python3
"""MLX + PyTorch MPS lanes of the GEMM shape sweep.

Protocol is pinned to `src/bin/bench_gemm_sweep.rs`: same shapes, same
warmup/iters, **synchronize every iteration** (mx.eval / torch.mps.synchronize)
so no lane hides dispatch cost behind pipelining, median over iters.

  python3 bench/gemm_sweep_mlx.py --iters 50 --warmup 10 --out results.json
  python3 bench/gemm_sweep_mlx.py --parity-dir /path/with/parity_*.npy
"""
import argparse, json, statistics, sys, time
import numpy as np

# (M, N, K, label) — must match SHAPES in bench_gemm_sweep.rs.
SHAPES = [
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
]

def shapes_from_env():
    """BENCH_SHAPES="MxNxK,..." — same override the Rust lane honours, so both
    lanes can be pointed at an identical diagnostic grid."""
    import os
    raw = os.environ.get("BENCH_SHAPES")
    if not raw:
        return None
    out = []
    for spec in filter(None, (x.strip() for x in raw.split(","))):
        m, n, k = (int(v) for v in spec.split("x"))
        out.append((m, n, k, f"{m}x{n}x{k}"))
    return out


def bench_mlx(shapes, warmup, iters, dtype="f32"):
    import mlx.core as mx
    mdt = {"f32": mx.float32, "bf16": mx.bfloat16}[dtype]
    rows = []
    for (m, n, k, label) in shapes:
        a = (mx.zeros((m, k), dtype=mdt) + 0.5).astype(mdt)
        b = (mx.zeros((k, n), dtype=mdt) + 0.5).astype(mdt)
        mx.eval(a, b)
        for _ in range(warmup):
            mx.eval(mx.matmul(a, b))
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(mx.matmul(a, b))
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(samples)
        rows.append(dict(shape=label, backend="mlx-" + dtype, runtime="mlx", m=m, n=n, k=k,
                         median_ms=med, best_ms=min(samples),
                         gflops=(2.0 * m * n * k) / (med * 1e6)))
        print(f"{label:<12} {'mlx-'+dtype:<14} M={m} N={n} K={k}  {med:8.3f} ms  "
              f"{rows[-1]['gflops']:8.1f} GFLOP/s", file=sys.stderr)
    return rows


def bench_torch(shapes, warmup, iters, dtype="f32"):
    import torch
    tdt = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    if not torch.backends.mps.is_available():
        print("torch MPS unavailable; skipping lane", file=sys.stderr)
        return []
    dev = torch.device("mps")
    rows = []
    for (m, n, k, label) in shapes:
        a = torch.full((m, k), 0.5, dtype=tdt, device=dev)
        b = torch.full((k, n), 0.5, dtype=tdt, device=dev)
        torch.mps.synchronize()
        for _ in range(warmup):
            torch.matmul(a, b)
            torch.mps.synchronize()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            torch.matmul(a, b)
            torch.mps.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(samples)
        rows.append(dict(shape=label, backend="mps-" + dtype, runtime="torch", m=m, n=n, k=k,
                         median_ms=med, best_ms=min(samples),
                         gflops=(2.0 * m * n * k) / (med * 1e6)))
        print(f"{label:<12} {'torch-mps-'+dtype:<14} M={m} N={n} K={k}  {med:8.3f} ms  "
              f"{rows[-1]['gflops']:8.1f} GFLOP/s", file=sys.stderr)
    return rows


def _mlx_matmul(a, b, dtype):
    import mlx.core as mx
    am, bm = mx.array(a), mx.array(b)
    if dtype == "bf16":
        am, bm = am.astype(mx.bfloat16), bm.astype(mx.bfloat16)
    c = mx.matmul(am, bm)
    # Reported rather than assumed: MLX returns bf16 from a bf16 matmul, which
    # is a rounding step tessl's f32 destination does not pay.
    out = "bf16" if c.dtype == mx.bfloat16 else "f32"
    return np.array(c.astype(mx.float32), copy=False), out


def _torch_matmul(a, b, dtype):
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS unavailable")
    at, bt = torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps")
    if dtype == "bf16":
        at, bt = at.to(torch.bfloat16), bt.to(torch.bfloat16)
    c = torch.matmul(at, bt)
    out = "bf16" if c.dtype == torch.bfloat16 else "f32"
    return c.to(torch.float32).cpu().numpy(), out


# (lane name, callable, operand dtype). Each is scored independently: a failure
# in one must not be attributed to another. An earlier version wrapped both MLX
# lanes in one try, so an f32 lane that had already scored also got a "skipped"
# row when the bf16 lane raised.
EXTERNAL_LANES = [
    ("mlx-f32", _mlx_matmul, "f32"),
    ("mlx-bf16", _mlx_matmul, "bf16"),
    ("torch-mps-f32", _torch_matmul, "f32"),
    ("torch-mps-bf16", _torch_matmul, "bf16"),
]


def adjudicate(over):
    """Message for a set of {lane: worst budget ratio} that exceeded 1.0.

    Split out so all three verdicts are reachable from a test. Which one
    applies decides the response, and they are opposite: a tessl-only breach is
    a kernel to fix, an everyone-breached result is a bound to re-derive.
    """
    tessl = sorted(l for l in over if l.startswith(("tensorops-", "simdgroup-")))
    other = sorted(l for l in over if l not in tessl)
    detail = ", ".join(f"{l} {over[l]:.3f}x" for l in sorted(over))
    if tessl and not other:
        verdict = ("Only tessl lanes exceeded it, and the comparison runtimes "
                   "stayed inside on the same operands: this is a kernel defect.")
    elif other and not tessl:
        verdict = ("Only the comparison runtimes exceeded it. tessl is inside the "
                   "bound; the other runtime is the outlier.")
    else:
        verdict = ("Every runtime exceeded it on the same operands, which points "
                   "at the bound rather than at any one kernel -- re-derive it "
                   "for this operand distribution before reading it as a defect.")
    return f"per-element error budget exceeded: {detail}. {verdict}"


def _load_manifest(parity_dir):
    """Read and structurally validate the dump manifest.

    Required, not optional. `bench_gemm_sweep` writes it last and only on
    success, so its absence means the dump was interrupted or never ran -- and
    a partial dump must not be scorable as a complete one.
    """
    import os
    path = os.path.join(parity_dir, "parity_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no parity_manifest.json in {parity_dir}. It is written last and only "
            "on a clean run, so this dump is absent or partial. Re-run "
            "`bench_gemm_sweep --dump-parity`.")
    try:
        with open(path) as f:
            man = json.load(f)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"parity_manifest.json is unreadable: {exc}")
    if not isinstance(man, dict):
        raise SystemExit(f"parity_manifest.json must be an object, got {type(man).__name__}")
    for key, want in (("lanes", list), ("seeds", list)):
        if not isinstance(man.get(key), want) or not man[key]:
            raise SystemExit(f"parity_manifest.json: '{key}' must be a non-empty list")
        if not all(isinstance(v, str) for v in man[key]):
            raise SystemExit(f"parity_manifest.json: '{key}' must contain only strings")
    for key in ("m", "n", "k"):
        if not isinstance(man.get(key), int) or man[key] <= 0:
            raise SystemExit(f"parity_manifest.json: '{key}' must be a positive integer")
    return man


# Unit roundoffs, mirroring tests/common/mod.rs. U_BF16 doubles as a safe upper
# bound for tf32-class formats (2^-11), which is why the relaxed path shares it.
U_F32 = 2.0 ** -24
U_BF16 = 2.0 ** -8

# lane -> (operand_u, out_u).
#
# `operand_u` is nonzero when the kernel reads operands narrower than the
# reference does. Note this differs from the Rust unit tests on purpose: those
# round their operands to bf16 *before* upload, so the GPU sees exactly the
# reference's values and operand width contributes nothing. This harness dumps
# the f32 originals as the reference, so the bf16 lane really does narrow.
#
# `out_u` is nonzero only for lanes that return bf16 -- MLX and torch do, tessl
# accumulates and returns f32 -- which is a rounding of the result itself.
LANE_PRECISION = {
    "tensorops-f32": (0.0, 0.0),
    "simdgroup-f32": (0.0, 0.0),
    "tensorops-tf32": (U_BF16, 0.0),
    "tensorops-bf16": (U_BF16, 0.0),
    "mlx-f32": (0.0, 0.0),
    "mlx-bf16": (U_BF16, U_BF16),
    "torch-mps-f32": (0.0, 0.0),
    "torch-mps-bf16": (U_BF16, U_BF16),
}


def _gamma(n, u):
    """gamma_n = n*u / (1 - n*u), the classical dot-product bound."""
    nu = n * u
    if nu >= 1.0:
        raise SystemExit(f"error bound degenerate at K={n}: n*u = {nu} >= 1")
    return nu / (1.0 - nu)


def _score(lane, c, ref, mag, scale, k, where):
    """One lane, one seed, three metrics -- and a bound, not just numbers.

    `normwise` (max|err| / max|ref|) is what this harness used to report alone.
    It cannot see a per-element failure: an output whose true value is near zero
    from cancellation can be wrong by orders of magnitude in relative terms and
    still move max|err| almost not at all. Measured on a plain uniform draw, a
    bf16 GEMM's worst element-wise relative error is ~6e+2 while its normwise
    error is ~3e-3 -- five orders apart, both correct, describing different
    things.

    `higham` is the one that decides pass or fail: err / ((gamma_{K+8} + 2*u_in)
    * sum|a.b| + u_out*|ref|), the same per-element budget
    `tests/common/mod.rs` asserts against. Above 1.0 the kernel is genuinely out
    of tolerance, and that is fatal here rather than reported.
    """
    c = np.asarray(c)
    if c.shape != ref.shape:
        raise SystemExit(f"{where} lane {lane}: shape {c.shape} != reference {ref.shape}")
    if not np.isfinite(c).all():
        bad = int((~np.isfinite(c)).sum())
        raise SystemExit(
            f"{where} lane {lane}: {bad} non-finite element(s). That is a broken "
            "kernel, not an accuracy number -- refusing to score it.")
    if lane not in LANE_PRECISION:
        # A permissive default for an unrecognised lane would let a new kernel
        # pass a budget nobody chose for it.
        raise SystemExit(
            f"{where}: lane {lane!r} has no entry in LANE_PRECISION; add its "
            f"operand/output unit roundoff rather than scoring it against a "
            f"budget picked by default. Known: {sorted(LANE_PRECISION)}")
    operand_u, out_u = LANE_PRECISION[lane]

    err = np.abs(c.astype(np.float64) - ref)
    ref_abs = np.abs(ref)
    live = ref_abs > scale * 1e-9
    elem_rel = float((err[live] / ref_abs[live]).max()) if live.any() else 0.0

    budget = (_gamma(k + 8, U_F32) + 2.0 * operand_u) * mag + out_u * ref_abs
    # An exactly-zero budget means every product was exactly zero, so the only
    # acceptable output is exactly zero.
    ratio = np.where(budget > 0, err / np.maximum(budget, np.finfo(np.float64).tiny),
                     np.where(err == 0, 0.0, np.inf))
    worst = float(ratio.max())
    return (float(err.max()), float(err.max() / scale), float(err.mean()),
            elem_rel, worst)


def parity(parity_dir):
    """Score every dumped lane on every seed against one float64 reference per
    seed, alongside MLX and torch on those same operands.

    A reduced-precision lane is *expected* to sit further from the reference --
    that is what it trades for speed -- so the question is never whether it
    matches bit-for-bit (tf32 and bf16 never can, by construction). It is how
    far out it lands next to what the runtime it is being compared against
    costs on the same inputs.

    Reported per lane is the **worst** relative error across all seeds, not the
    last and not the mean: one draw cannot tell a bound from a lucky sample.
    """
    import os
    man = _load_manifest(parity_dir)
    lanes, seeds = man["lanes"], man["seeds"]
    m, n, k = man["m"], man["n"], man["k"]
    print(f"parity: {len(lanes)} lanes x {len(seeds)} seeds at "
          f"{man.get('shape', '?')} (M={m} N={n} K={k})", file=sys.stderr)

    acc = {}          # lane -> per-metric lists across seeds
    skipped = {}      # lane -> {seed: reason}
    KEYS = ("abs", "rel", "mean", "elem", "higham")

    def record(lane, out_dtype, scores):
        e = acc.setdefault(lane, dict({k: [] for k in KEYS}, out=out_dtype))
        if e["out"] != out_dtype:
            raise SystemExit(
                f"lane {lane}: output dtype changed between seeds "
                f"({e['out']} then {out_dtype})")
        for key, v in zip(KEYS, scores):
            e[key].append(v)

    for seed in seeds:
        sd = os.path.join(parity_dir, seed)
        if not os.path.isdir(sd):
            raise SystemExit(f"manifest lists seed directory {seed!r}, which is absent")
        try:
            a = np.load(os.path.join(sd, "parity_a.npy"))
            b = np.load(os.path.join(sd, "parity_b.npy"))
        except OSError as exc:
            raise SystemExit(f"{seed}: operands unreadable: {exc}")
        if a.shape != (m, k) or b.shape != (k, n):
            raise SystemExit(
                f"{seed}: operand shapes {a.shape} @ {b.shape} contradict the "
                f"manifest ({m}x{k} @ {k}x{n})")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise SystemExit(f"{seed}: operands contain non-finite values")

        A64, B64 = a.astype(np.float64), b.astype(np.float64)
        ref = A64 @ B64
        # sum|a_ik * b_kj| -- the conditioning of each output element's sum, and
        # the numerator of the per-element budget. Costs a second f64 GEMM.
        mag = np.abs(A64) @ np.abs(B64)
        scale = float(np.abs(ref).max())
        if not np.isfinite(scale) or scale == 0.0:
            raise SystemExit(
                f"{seed}: reference peak magnitude is {scale}; every relative "
                "error would be a division by zero or a NaN")

        for lane in lanes:
            fn = os.path.join(sd, f"parity_c_{lane}.npy")
            if not os.path.exists(fn):
                raise SystemExit(
                    f"{seed}: manifest lists lane {lane!r} but "
                    f"parity_c_{lane}.npy is missing. Re-run "
                    "`bench_gemm_sweep --dump-parity`. Refusing to report a "
                    "partial check as a complete one.")
            record(lane, "f32", _score(lane, np.load(fn), ref, mag, scale, k, seed))

        for lane, fn, dt in EXTERNAL_LANES:
            try:
                c, out_dtype = fn(a, b, dt)
            except Exception as exc:
                skipped.setdefault(lane, {})[seed] = str(exc)
                continue
            record(lane, out_dtype, _score(lane, c, ref, mag, scale, k, seed))

    # A lane over budget is adjudicated only once every lane has been scored.
    # Aborting on the first one cannot tell "tessl is out of tolerance" from
    # "this bound is too tight for these operands", and those need opposite
    # responses. If MLX and torch blow the same budget on the same inputs, the
    # bound is the thing that is wrong.
    over = {lane: max(e["higham"]) for lane, e in acc.items() if max(e["higham"]) > 1.0}
    if over:
        raise SystemExit(adjudicate(over))

    rows = []
    for lane in list(acc) + [l for l in skipped if l not in acc]:
        miss = skipped.get(lane, {})
        if lane not in acc:
            reason = sorted(set(miss.values()))
            rows.append(dict(lane=lane, skipped="; ".join(reason), seeds=0))
            print(f"parity {lane:<22} SKIPPED on all {len(seeds)} seeds -- "
                  f"{reason[0]}", file=sys.stderr)
            continue
        if miss:
            # Averaging over a lane that ran on some seeds and not others would
            # quietly report a partial sample as a full one.
            raise SystemExit(
                f"lane {lane} scored on {len(acc[lane]['rel'])} seeds but failed on "
                f"{sorted(miss)}: {sorted(set(miss.values()))[0]}. A lane must "
                "cover every seed or none.")
        e = acc[lane]
        rows.append(dict(
            lane=lane, out_dtype=e["out"], seeds=len(e["rel"]),
            max_rel_err=max(e["rel"]),
            median_rel_err=statistics.median(e["rel"]),
            min_rel_err=min(e["rel"]),
            max_abs_err=max(e["abs"]),
            mean_abs_err=statistics.fmean(e["mean"]),
            # Worst single output element's relative error. Orders of magnitude
            # above max_rel_err wherever cancellation makes an output near zero,
            # and that is the honest number for "how wrong can one result be".
            max_elem_rel_err=max(e["elem"]),
            # Fraction of the per-element error budget used. <= 1.0 or the run
            # would already have aborted; this says how much headroom is left.
            max_higham_ratio=max(e["higham"]),
            per_seed_rel_err=e["rel"],
            per_seed_higham=e["higham"]))
        print(f"parity {lane:<22} out={e['out']:<4} norm={max(e['rel']):.2e}  "
              f"elem={max(e['elem']):.2e}  budget={max(e['higham']):.3f}x  "
              f"({len(e['rel'])} seeds)", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--parity-dir")
    ap.add_argument("--lanes", default="mlx,torch")
    ap.add_argument("--dtypes", default="f32,bf16")
    args = ap.parse_args()

    def emit(rows):
        out = json.dumps(rows, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out)
        else:
            print(out)

    if args.parity_dir:
        # --out applies here too; it used to be silently ignored on this path.
        emit(parity(args.parity_dir))
        return

    # statistics.median() raises on an empty sample, and a zero-iteration sweep
    # is a request for a number nothing measured.
    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")

    lanes = args.lanes.split(",")
    unknown = [x for x in lanes if x not in ("mlx", "torch")]
    if unknown:
        raise SystemExit(f"--lanes: unknown lane(s) {unknown}; expected mlx and/or torch")
    dtypes = args.dtypes.split(",")
    bad = [x for x in dtypes if x not in ("f32", "bf16")]
    if bad:
        raise SystemExit(f"--dtypes: unknown dtype(s) {bad}; expected f32 and/or bf16")

    shapes = shapes_from_env() or SHAPES
    rows = []
    for dt in dtypes:
        if "mlx" in lanes:
            rows += bench_mlx(shapes, args.warmup, args.iters, dt)
        if "torch" in lanes:
            rows += bench_torch(shapes, args.warmup, args.iters, dt)
    emit(rows)


if __name__ == "__main__":
    main()
