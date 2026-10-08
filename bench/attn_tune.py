#!/usr/bin/env python3
"""Sweep the attention kernels' tuning knobs and report the winner per config.

Both fast paths have one free parameter each, and neither is derivable:

  * `BENCH_ATTN_ROWS_R` -- lanes per query row. The reduction costs log2(R)
    steps against 2*D/R multiply-adds, so small R trades reduction for per-lane
    work until `q_reg + acc` spills the register file.
  * `BENCH_ATTN_ROWS_SGT` -- simdgroups per threadgroup. Every simdgroup in a
    threadgroup walks the same key range, so this is how many query rows one
    global K/V read serves. It interacts with `R` (rows per threadgroup are
    `SGT * 32/R`), so the two are swept against each other, not alone.
  * `BENCH_ATTN_DECODE_CHUNK` -- keys per KV chunk. Smaller buys grid
    parallelism, which is what a latency-bound decode wants, at the cost of
    more partials for the reduce pass to combine.

Each is a *compile-time* constant in the shader, so a value is a kernel, not a
flag -- which is why they are swept rather than guessed.

`--knob batched` is not a tuning parameter: it sweeps launches-per-submit to
*decompose* a measurement rather than to pick a winner. One launch per submit
is what `mx.eval` and `torch.mps.synchronize` do, so it charges the host round
trip; 32 amortises it away. The ratio is the share of the wall clock that is
actually kernel, and it is printed as such instead of as a winner.

  python3 bench/attn_tune.py --knob rows
  python3 bench/attn_tune.py --knob decode --rounds 3
  python3 bench/attn_tune.py --knob batched --rounds 3
"""
import argparse, json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "target", "release", "bench_flash_attn")

KNOBS = {
    "rows": ("BENCH_ATTN_ROWS_R", ["8", "16", "32"], "tessl-rows",
             ["swa128_prefill_512", "swa128_prefill_2048", "swa128_prefill_4096",
              "swa256_prefill_2048", "global512_prefill_1024"]),
    "decode": ("BENCH_ATTN_DECODE_CHUNK", ["64", "128", "256"], "tessl-decode",
               ["swa128_decode_1k", "swa128_decode_4k", "swa128_decode_b8_4k",
                "swa256_decode_4k", "global512_decode_4k"]),
    "rows-g": ("BENCH_ATTN_ROWS_SGT", ["8", "16", "32"], "tessl-rows",
               ["swa128_prefill_512", "swa128_prefill_2048", "swa128_prefill_4096",
                "swa256_prefill_2048", "global512_prefill_1024"]),
    "decode-r": ("BENCH_ATTN_DECODE_R", ["8", "16", "32"], "tessl-decode",
                 ["swa128_decode_1k", "swa128_decode_4k", "swa128_decode_b8_4k",
                  "swa256_decode_4k", "global512_decode_4k"]),
    # Which query heads share a threadgroup. `group` is the H/Hkv heads that
    # share a KV head; `all` additionally makes the per-key read the whole
    # contiguous [Hkv][D] row, at the cost of H-simdgroup threadgroups.
    "decode-sgs": ("BENCH_ATTN_DECODE_SGS", ["one", "group", "all"], "tessl-decode",
                   ["swa128_decode_1k", "swa128_decode_4k", "swa128_decode_b8_4k",
                    "swa256_decode_4k", "global512_decode_4k",
                    "global512_decode_4k_mha"]),
    "reduce-w": ("BENCH_ATTN_REDUCE_W", ["32", "128", "256"], "tessl-decode",
                 ["swa128_decode_1k", "swa128_decode_4k", "swa128_decode_b8_4k",
                  "swa256_decode_4k", "global512_decode_4k"]),
    # Decomposition, not tuning: the shipping routed lane over both decode and
    # prefill, because prefill is the control that says the decode split is
    # real and not an artefact of the batched arm.
    "batched": ("BENCH_ATTN_BATCHED", ["1", "32"], "tessl",
                ["swa128_decode_1k", "swa128_decode_4k", "swa256_decode_4k",
                 "global512_decode_4k", "swa128_prefill_2048",
                 "swa256_prefill_2048", "global512_prefill_1024"]),
}

# One canonical list of the knob environment variables, so `run` clears every
# knob it is not sweeping. A hardcoded second copy silently leaks whichever
# knob was added to KNOBS but forgotten here.
KNOB_ENV = sorted({env for env, *_ in KNOBS.values()})


def run(env_name, value, lane, cfgs, iters, warmup, batched=1):
    env = dict(os.environ, BENCH_ITERS=str(iters), BENCH_WARMUP=str(warmup),
               BENCH_ATTN_CFGS=",".join(cfgs))
    for other in KNOB_ENV:
        env.pop(other, None)
    # Batching is applied to every arm of the sweep, then the swept knob is set
    # last so `--knob batched` still overrides it. A decode sweep at one launch
    # per submit is ~85% host round trip, which compresses every gain it
    # reports towards 1.0x; the knob that wins can still be read off it, but
    # the margin cannot.
    env["BENCH_ATTN_BATCHED"] = str(batched)
    env[env_name] = value
    r = subprocess.run([BIN], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{env_name}={value}: exited {r.returncode}\n{r.stderr.strip()}")
    rows = json.loads(r.stdout)
    out = {x["cfg"]: x["median_ms"] for x in rows if x["runtime"] == lane}
    missing = [c for c in cfgs if c not in out]
    if missing:
        raise SystemExit(
            f"{env_name}={value}: lane {lane!r} absent for {missing}. Refusing to "
            "pick a winner from a partial sweep.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", choices=sorted(KNOBS), required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batched", type=int, default=1,
                    help="launches per submit for every arm. >1 amortises the "
                         "command-buffer submit, which at decode sizes is most "
                         "of the wall clock and none of the kernel.")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.rounds < 1:
        raise SystemExit(f"--rounds must be >= 1, got {args.rounds}")
    if args.batched < 1:
        raise SystemExit(f"--batched must be >= 1, got {args.batched}")
    if args.knob == "batched" and args.batched != 1:
        raise SystemExit(
            "--knob batched sweeps launches-per-submit; --batched would fix the "
            "very thing being swept. Drop one of them.")
    if not os.path.exists(BIN):
        raise SystemExit(f"{BIN} not built. cargo build --release --bin bench_flash_attn")

    env_name, values, lane, cfgs = KNOBS[args.knob]
    # Round-robin over the values rather than all rounds of one then the next,
    # so thermal drift cannot favour whichever was measured first.
    per = {v: {c: [] for c in cfgs} for v in values}
    for r in range(args.rounds):
        for v in values:
            got = run(env_name, v, lane, cfgs, args.iters, args.warmup, args.batched)
            for c in cfgs:
                per[v][c].append(got[c])
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    med = {v: {c: statistics.median(per[v][c]) for c in cfgs} for v in values}
    what = ("wall clock, submit-and-wait per call" if args.batched == 1
            else f"kernel only, {args.batched} launches per submit")
    print(f"\n{env_name}  (median ms over {args.rounds} interleaved rounds, {what})")

    if args.knob == "batched":
        # No winner is reported here on purpose. 32 launches per submit is a
        # measurement protocol, not a setting to ship, and an artifact that
        # named it "winner" would read as a recommendation.
        solo, amortised = values[0], values[-1]
        if (solo, amortised) != ("1", "32"):
            raise SystemExit(
                f"--knob batched expects values ('1', '32'), got {values}. The "
                "share is defined against one launch per submit.")
        print(f"  {'config':<24}{'solo ms':>10}{'batched ms':>12}"
              f"{'submit ms':>11}{'kernel share':>14}")
        share = {}
        for c in cfgs:
            hi, lo = med[solo][c], med[amortised][c]
            share[c] = lo / hi
            print(f"  {c:<24}{hi:>10.4f}{lo:>12.4f}{hi - lo:>11.4f}"
                  f"{100 * share[c]:>13.1f}%")
        doc = dict(knob=args.knob, env=env_name, values=values, rounds=args.rounds,
                   iters=args.iters, batched=args.batched, median_ms=med,
                   winner=None, kernel_share=share)
    else:
        print(f"  {'config':<24}" + "".join(f"{v:>10}" for v in values)
              + f"{'winner':>10}{'gain':>8}")
        winners = {}
        for c in cfgs:
            best = min(values, key=lambda v: med[v][c])
            worst = max(med[v][c] for v in values)
            winners[c] = best
            print(f"  {c:<24}" + "".join(f"{med[v][c]:>10.3f}" for v in values)
                  + f"{best:>10}{worst / med[best][c]:>7.2f}x")
        doc = dict(knob=args.knob, env=env_name, values=values, rounds=args.rounds,
                   iters=args.iters, batched=args.batched, median_ms=med,
                   winner=winners)
    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(doc, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
