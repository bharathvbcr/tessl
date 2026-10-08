#!/usr/bin/env python3
"""torch-MPS and MLX lanes for the flash-attention sweep, plus the f64 scorer.

Protocol is pinned to `src/bin/bench_flash_attn.rs`: same configs, same
warmup/iters, synchronize every iteration, median over iters.

The masking rule is transcribed from the kernels, via `tests/attention.rs`:

  * q_abs = q_off + t_q, k_abs = kv_off + t_k
  * sliding window keeps max(0, q_abs - window + 1) <= k_abs <= q_abs
  * global keeps k_abs <= q_abs and ignores window
  * a row with nothing unmasked is zeros, not NaN

Layout note: tessl is [B, Tq, H, D] and [B, Tkv, Hkv, D]; torch and MLX both
want head-major [B, H, T, D], so every lane permutes and permutes back. GQA is
`h // (H/Hkv)`, which is repeat_interleave -- torch's `enable_gqa` and MLX's
native GQA use the same convention, and the f64 reference is scored against all
three so a convention error surfaces as error rather than as a ratio.

  python3 bench/flash_attn_torch_mlx.py --parity-dir /tmp/attn
  python3 bench/flash_attn_torch_mlx.py --iters 30 --warmup 10
"""
import argparse, json, os, statistics, sys, time
import numpy as np

# Must match CFGS in src/bin/bench_flash_attn.rs. Cross-checked against the
# dump manifest on every parity run, so drift is caught rather than assumed.
CFGS = [
    dict(label="swa128_prefill_512",    b=1, tq=512,  tkv=512,  h=32, hkv=8, d=128, window=1024, q_off=0,    kv_off=0),
    dict(label="swa128_prefill_2048",   b=1, tq=2048, tkv=2048, h=32, hkv=8, d=128, window=1024, q_off=0,    kv_off=0),
    dict(label="swa128_prefill_4096",   b=1, tq=4096, tkv=4096, h=32, hkv=8, d=128, window=1024, q_off=0,    kv_off=0),
    dict(label="swa128_decode_1k",      b=1, tq=1,    tkv=1024, h=32, hkv=8, d=128, window=1024, q_off=1023, kv_off=0),
    dict(label="swa128_decode_4k",      b=1, tq=1,    tkv=4096, h=32, hkv=8, d=128, window=1024, q_off=4095, kv_off=0),
    dict(label="swa128_decode_b8_4k",   b=8, tq=1,    tkv=4096, h=32, hkv=8, d=128, window=1024, q_off=4095, kv_off=0),
    dict(label="swa128_decode_b32_1k",  b=32, tq=1,   tkv=1024, h=32, hkv=8, d=128, window=1024, q_off=1023, kv_off=0),
    dict(label="swa128_decode_b64_1k",  b=64, tq=1,   tkv=1024, h=32, hkv=8, d=128, window=1024, q_off=1023, kv_off=0),
    dict(label="swa256_prefill_2048",   b=1, tq=2048, tkv=2048, h=16, hkv=4, d=256, window=1024, q_off=0,    kv_off=0),
    dict(label="swa256_decode_4k",      b=1, tq=1,    tkv=4096, h=16, hkv=4, d=256, window=1024, q_off=4095, kv_off=0),
    dict(label="global512_prefill_1024", b=1, tq=1024, tkv=1024, h=8, hkv=2, d=512, window=-1,   q_off=0,    kv_off=0),
    dict(label="global512_decode_4k",   b=1, tq=1,    tkv=4096, h=8,  hkv=2, d=512, window=-1,   q_off=4095, kv_off=0),
    dict(label="global512_decode_4k_mha", b=1, tq=1,  tkv=4096, h=8,  hkv=8, d=512, window=-1,   q_off=4095, kv_off=0),
    dict(label="global512_decode_4k_mqa", b=1, tq=1,  tkv=4096, h=8,  hkv=1, d=512, window=-1,   q_off=4095, kv_off=0),
]
BY_LABEL = {c["label"]: c for c in CFGS}


def keep_mask(c):
    """[Tq, Tkv] bool, True = attend. The kernels' rule, verbatim."""
    q_abs = c["q_off"] + np.arange(c["tq"], dtype=np.int64)[:, None]
    k_abs = c["kv_off"] + np.arange(c["tkv"], dtype=np.int64)[None, :]
    keep = k_abs <= q_abs
    if c["window"] >= 0:
        keep &= k_abs >= np.maximum(0, q_abs - c["window"] + 1)
    return keep


def reference(q, k, v, c, chunk=512):
    """f64 attention reference in tessl's [B, Tq, H, D] layout.

    Computed per (batch, head) and chunked over query rows: a 4096x4096 f64
    score matrix is 134 MB, and materialising one per head at once is 4 GB.
    """
    b, tq, h, d, tkv, hkv = c["b"], c["tq"], c["h"], c["d"], c["tkv"], c["hkv"]
    group = max(h // hkv, 1)
    keep = keep_mask(c)
    out = np.zeros((b, tq, h, d), dtype=np.float64)
    scale = 1.0 / np.sqrt(d)
    for bi in range(b):
        for hi in range(h):
            hk = hi // group
            qh = q[bi, :, hi, :].astype(np.float64)          # [Tq, D]
            kh = k[bi, :, hk, :].astype(np.float64)          # [Tkv, D]
            vh = v[bi, :, hk, :].astype(np.float64)
            for lo in range(0, tq, chunk):
                hi_ = min(lo + chunk, tq)
                s = (qh[lo:hi_] @ kh.T) * scale              # [chunk, Tkv]
                m = np.where(keep[lo:hi_], s, -np.inf)
                mx = m.max(axis=1, keepdims=True)
                # A fully masked row: the kernels emit zeros rather than NaN.
                live = np.isfinite(mx[:, 0])
                p = np.zeros_like(m)
                p[live] = np.exp(m[live] - mx[live])
                l = p.sum(axis=1, keepdims=True)
                inv = np.where(l > 0, 1.0 / np.maximum(l, np.finfo(np.float64).tiny), 0.0)
                out[bi, lo:hi_, hi, :] = (p @ vh) * inv
    return out


def _to_headmajor(x):
    """[B, T, H, D] -> [B, H, T, D]."""
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)))


def torch_attn(q, k, v, c, time_it=False, warmup=0, iters=1):
    import torch
    import torch.nn.functional as F
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS unavailable")
    dev = torch.device("mps")
    qt = torch.from_numpy(_to_headmajor(q)).to(dev)
    kt = torch.from_numpy(_to_headmajor(k)).to(dev)
    vt = torch.from_numpy(_to_headmajor(v)).to(dev)
    scale = 1.0 / float(np.sqrt(c["d"]))
    mask = torch.from_numpy(keep_mask(c)).to(dev)

    def run():
        return F.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=mask, scale=scale, enable_gqa=(c["h"] != c["hkv"]))

    if not time_it:
        o = run()
        torch.mps.synchronize()
        return np.transpose(o.float().cpu().numpy(), (0, 2, 1, 3))
    batch = int(os.environ.get("BENCH_ATTN_BATCHED", "1"))
    for _ in range(warmup):
        for _ in range(batch):
            run()
        torch.mps.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(batch):
            run()
        torch.mps.synchronize()
        s.append((time.perf_counter() - t0) * 1000.0 / batch)
    return s


def mlx_attn(q, k, v, c, time_it=False, warmup=0, iters=1):
    import mlx.core as mx
    qm = mx.array(_to_headmajor(q))
    km = mx.array(_to_headmajor(k))
    vm = mx.array(_to_headmajor(v))
    scale = 1.0 / float(np.sqrt(c["d"]))
    # Additive float mask rather than boolean: MLX accepts either, and the
    # f64 scorer below is what confirms whichever convention was taken.
    add = np.where(keep_mask(c), 0.0, -np.inf).astype(np.float32)
    mask = mx.array(add)

    def run():
        return mx.fast.scaled_dot_product_attention(qm, km, vm, scale=scale, mask=mask)

    if not time_it:
        o = run()
        mx.eval(o)
        return np.transpose(np.array(o, copy=False), (0, 2, 1, 3))
    # `batch > 1` queues that many attentions before a single eval, which is
    # what separates MLX's kernel from its submit cost the same way
    # BENCH_ATTN_BATCHED does for tessl.
    batch = int(os.environ.get("BENCH_ATTN_BATCHED", "1"))
    for _ in range(warmup):
        mx.eval([run() for _ in range(batch)])
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval([run() for _ in range(batch)])
        s.append((time.perf_counter() - t0) * 1000.0 / batch)
    return s


def live_pairs(c):
    return int(keep_mask(c).sum())


def live_flop(c):
    return 4.0 * c["d"] * c["b"] * c["h"] * live_pairs(c)


def parity(parity_dir):
    """Score tessl's dumped O, plus torch and MLX on the same inputs, against
    one f64 reference per config.

    torch and MLX are scored too, deliberately. If the mask convention or the
    GQA mapping in this file were wrong, the comparison lanes would be wrong in
    exactly the same way and a ratio would still look plausible -- an error
    against the reference is the only thing that catches it.
    """
    mpath = os.path.join(parity_dir, "attn_manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(
            f"no attn_manifest.json in {parity_dir}. It is written last and only "
            "on a clean run, so this dump is absent or partial.")
    with open(mpath) as f:
        man = json.load(f)
    rows = []
    for mc in man["configs"]:
        label = mc["cfg"]
        if label not in BY_LABEL:
            raise SystemExit(f"dump has config {label!r} absent from this file's CFGS mirror")
        c = BY_LABEL[label]
        for key in ("b", "tq", "tkv", "h", "hkv", "d", "window", "q_off", "kv_off"):
            if mc[key] != c[key]:
                raise SystemExit(
                    f"config drift on {label}: dump says {key}={mc[key]}, "
                    f"this file says {c[key]}")
        d = os.path.join(parity_dir, label)
        q = np.load(os.path.join(d, "q.npy"))
        k = np.load(os.path.join(d, "k.npy"))
        v = np.load(os.path.join(d, "v.npy"))
        if q.shape != (c["b"], c["tq"], c["h"], c["d"]):
            raise SystemExit(f"{label}: q shape {q.shape} contradicts the config")
        ref = reference(q, k, v, c)
        scale = float(np.abs(ref).max())
        if not np.isfinite(scale) or scale == 0.0:
            raise SystemExit(f"{label}: reference peak is {scale}")

        # Every implementation the dump ran, scored against the same f64
        # reference under the name of the dispatch that produced it. The
        # manifest lists them, so a lane the binary wrote and this scorer
        # skipped -- or one the manifest names and the dump does not hold --
        # is an error rather than a quietly shorter report.
        declared = mc.get("lanes")
        if not declared:
            raise SystemExit(
                f"{label}: manifest names no lanes. A dump whose lanes cannot "
                "be identified must not be scored as if they had been.")
        lanes = {}
        for entry in declared:
            name = entry["lane"]
            fn = os.path.join(d, f"o_{name}.npy")
            if not os.path.exists(fn):
                raise SystemExit(
                    f"{label}: manifest names lane {name!r} ({entry['kernel']}) "
                    f"but {os.path.basename(fn)} is absent. A missing lane must "
                    "not read as a pass.")
            lanes[name] = np.load(fn)
        stray = sorted(x[2:-4] for x in os.listdir(d)
                       if x.startswith("o_") and x.endswith(".npy")
                       and x[2:-4] not in lanes)
        if stray:
            raise SystemExit(
                f"{label}: dump holds unscored lane(s) {stray} that the manifest "
                "does not name. Scoring a subset silently is the defect this "
                "harness exists to prevent.")
        if c["tq"] == 1 and "tessl-decode" not in lanes:
            raise SystemExit(
                f"{label}: Tq == 1 but the dump has no tessl-decode lane. The "
                "KV-split path is expected here.")
        # At Tq == 1 the routed path *is* the KV split, so the two lanes are the
        # same kernel on the same data and must agree bit for bit. They did not:
        # the benchmark's forced-decode lane defaulted to chunk 256 for every
        # head dim while the library ships 128, so a lane labelled "the decode
        # kernel" was a kernel the library never dispatches. Nothing else in
        # this harness could see that -- both outputs were correct, just not the
        # same one.
        if c["tq"] == 1 and not np.array_equal(lanes["tessl"], lanes["tessl-decode"]):
            n = int((lanes["tessl"] != lanes["tessl-decode"]).sum())
            worst = float(np.abs(lanes["tessl"].astype(np.float64)
                                 - lanes["tessl-decode"].astype(np.float64)).max())
            raise SystemExit(
                f"{label}: the routed lane and the forced KV-split lane differ "
                f"in {n} of {lanes['tessl'].size} elements (max {worst:.3e}). At "
                "Tq == 1 routing takes the split path, so these are the same "
                "kernel on the same data -- a difference means the benchmark's "
                "lane is configured differently from the shipping one.")
        for name, fn in (("torch-mps", torch_attn), ("mlx", mlx_attn)):
            try:
                lanes[name] = fn(q, k, v, c)
            except Exception as exc:
                rows.append(dict(cfg=label, lane=name, skipped=str(exc)))
                print(f"attn {label:<22} {name:<10} SKIPPED -- {exc}", file=sys.stderr)
        for name, o in lanes.items():
            o = np.asarray(o)
            if o.shape != ref.shape:
                raise SystemExit(f"{label}/{name}: shape {o.shape} != reference {ref.shape}")
            if not np.isfinite(o).all():
                raise SystemExit(
                    f"{label}/{name}: {int((~np.isfinite(o)).sum())} non-finite output(s)")
            err = np.abs(o.astype(np.float64) - ref)
            rows.append(dict(cfg=label, lane=name, max_abs_err=float(err.max()),
                             max_rel_err=float(err.max() / scale),
                             mean_abs_err=float(err.mean())))
            print(f"attn {label:<22} {name:<10} max_rel={err.max() / scale:.3e}",
                  file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lanes", default="torch,mlx")
    ap.add_argument("--cfgs", default="all")
    ap.add_argument("--out")
    ap.add_argument("--parity-dir")
    args = ap.parse_args()

    def emit(rows):
        text = json.dumps(rows, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
        else:
            print(text)

    if args.parity_dir:
        emit(parity(args.parity_dir))
        return

    if args.iters < 1:
        raise SystemExit(f"--iters must be >= 1, got {args.iters}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}")
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    bad = [x for x in lanes if x not in ("torch", "mlx")]
    if bad:
        raise SystemExit(f"--lanes: unknown {bad}; expected torch and/or mlx")
    want = list(BY_LABEL) if args.cfgs == "all" else [x.strip() for x in args.cfgs.split(",")]
    bad = [x for x in want if x not in BY_LABEL]
    if bad:
        raise SystemExit(f"--cfgs: unknown {bad}; expected from {list(BY_LABEL)}")

    rng = np.random.default_rng(0)
    rows = []
    for label in want:
        c = BY_LABEL[label]
        q = rng.uniform(-1, 1, (c["b"], c["tq"], c["h"], c["d"])).astype(np.float32)
        k = rng.uniform(-1, 1, (c["b"], c["tkv"], c["hkv"], c["d"])).astype(np.float32)
        v = rng.uniform(-1, 1, (c["b"], c["tkv"], c["hkv"], c["d"])).astype(np.float32)
        flop = live_flop(c)
        for name, fn in (("torch-mps", torch_attn), ("mlx", mlx_attn)):
            if name.split("-")[0] not in lanes:
                continue
            try:
                s = fn(q, k, v, c, time_it=True, warmup=args.warmup, iters=args.iters)
            except Exception as exc:
                print(f"{label:<22} {name:<10} SKIPPED -- {exc}", file=sys.stderr)
                continue
            med = statistics.median(s)
            rows.append(dict(cfg=label, kernel=f"sdpa-{name}", runtime=name,
                             batched=int(os.environ.get("BENCH_ATTN_BATCHED", "1")),
                             **{x: c[x] for x in ("b", "tq", "tkv", "h", "hkv", "d",
                                                  "window", "q_off", "kv_off")},
                             median_ms=med, best_ms=min(s),
                             live_pairs=live_pairs(c), gflops=flop / (med * 1e6)))
            print(f"{label:<22} {name:<10} {med:8.3f} ms  {flop / (med * 1e6):9.1f} GFLOP/s",
                  file=sys.stderr)
    emit(rows)


if __name__ == "__main__":
    main()
