"""Builds bench/results/embedgemma2_memory_m5pro.{json,txt} from the raw runs (analysis only; ships nothing).

    python embedgemma2_memory_report.py <raw dir> <output dir>

The raw directory holds what the lane H session produced:

* ``table_{before,after}_{1,2}.json``: ``profile_embedgemma2`` ``PROFILE_MODE=memory-table``, two runs of
  each build (``before`` is 16f5190, ``after`` the change);
* ``hash_{before,after,after_2}.txt``: ``PROFILE_MODE=hash`` on the real checkpoint;
* ``tiny_{before,after}.txt``: the ``TINY`` lines of ``tests/embedgemma2_memory.rs tiny_embedding_hashes``;
* ``paired_{f32,bf16}_a.json.lanes/``: ``bench_paired --out``'s per-lane run files (``rNN-I-lane.stdout``
  and ``.meta.json``), lanes ``before``, ``after``, ``after_ple256``;
* ``paired_f32.json.lanes/``: an earlier 6-round run with a fourth lane, ``after_ple64``;
* ``bounds.txt``: the bounds, written before the changed code was measured.
"""

import glob
import json
import statistics as st
import sys
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MIB_BOUND = 1.03  # B3/B4: median of per-round ratios


def table(path):
    """{(operands, workload, config): peak_above_model_mib}."""
    doc = json.loads(Path(path).read_text())
    out = {}
    for r in doc["runs"]:
        x = r["result"]
        out[(r["operands"], x["workload"], r["label"])] = x["peak_above_model_mib"]
    return out


def lanes_dir(path):
    """{lane: {workload: [ms_min per round]}}, the per-round normalized 1-minute load, the rounds."""
    runs, loads = {}, []
    for meta_path in sorted(glob.glob(str(Path(path) / "*.meta.json"))):
        meta = json.loads(Path(meta_path).read_text())
        rows = json.loads(Path(meta_path[: -len(".meta.json")] + ".stdout").read_text().strip().splitlines()[-1])
        for r in rows:
            runs.setdefault(meta["lane"], {}).setdefault(r["workload"], {})[meta["round"]] = r["ms_min"]
        loads.append(meta["load_at_start"]["normalized_loadavg_1m"])
    rounds = sorted({k for lane in runs.values() for w in lane.values() for k in w})
    series = {l: {w: [d[k] for k in rounds] for w, d in ws.items()} for l, ws in runs.items()}
    return series, loads, rounds


def ratio_stats(series, lane, base="before"):
    out = {}
    for w, xs in series[lane].items():
        bs = series[base][w]
        ratios = [a / b for a, b in zip(xs, bs)]
        out[w] = {
            "min_of_n_ms": min(xs),
            "base_min_of_n_ms": min(bs),
            "min_of_n_ratio": min(xs) / min(bs),
            "median_of_round_ratios": st.median(ratios),
            "ratio_lo": min(ratios),
            "ratio_hi": max(ratios),
            "spread": max(ratios) / min(ratios),
            "rounds_within_bound": sum(r <= MIB_BOUND for r in ratios),
            "rounds": len(ratios),
        }
    return out


def read_hashes(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.startswith(("HASH", "TINY"))]


doc = {"bounds": (RAW / "bounds.txt").read_text()}
tb = [table(RAW / f"table_before_{i}.json") for i in (1, 2)]
ta = [table(RAW / f"table_after_{i}.json") for i in (1, 2)]
doc["memory"] = {
    "cells": [
        {"operands": k[0], "workload": k[1], "config": k[2], "before_mib": tb[0][k], "after_mib": ta[0][k]}
        for k in tb[0]
    ],
    "before_stable": tb[0] == tb[1],
    "after_stable": ta[0] == ta[1],
}
doc["hashes"] = {
    "real_before": read_hashes(RAW / "hash_before.txt"),
    "real_after": read_hashes(RAW / "hash_after.txt"),
    "real_after_repeat": read_hashes(RAW / "hash_after_2.txt"),
    "tiny_before": read_hashes(RAW / "tiny_before.txt"),
    "tiny_after": read_hashes(RAW / "tiny_after.txt"),
}
doc["hashes"]["real_identical"] = doc["hashes"]["real_before"] == doc["hashes"]["real_after"]
doc["hashes"]["tiny_identical"] = doc["hashes"]["tiny_before"] == doc["hashes"]["tiny_after"]
doc["paired"] = {}
for name, d in (("f32", "paired_f32_a.json.lanes"), ("bf16", "paired_bf16_a.json.lanes"), ("f32_six_rounds_four_lanes", "paired_f32.json.lanes")):
    series, loads, rounds = lanes_dir(RAW / d)
    doc["paired"][name] = {
        "rounds": rounds,
        "series_ms": series,
        "load_1m_normalized": {"min": min(loads), "median": st.median(loads), "max": max(loads)},
        "vs_before": {l: ratio_stats(series, l) for l in series if l != "before"},
        "ple256_vs_after": ratio_stats(series, "after_ple256", base="after"),
    }
doc["ple_sweeps"] = {}
for name, d in (
    ("f32 1024-4096 rows", "ple_f32_a.json.lanes"),
    ("bf16 1024-4096 rows, first run (machine at load 14-38 per core)", "ple_bf16_a.json.lanes"),
    ("bf16 1024-4096 rows", "ple_bf16_b.json.lanes"),
    ("f32 8192 rows", "ple_f32_big.json.lanes"),
    ("f32 2x4096 repeat (the first run's median was 1.0303)", "ple_f32_big2.json.lanes"),
    ("bf16 8192 rows", "ple_bf16_big.json.lanes"),
):
    series, loads, rounds = lanes_dir(RAW / d)
    doc["ple_sweeps"][name] = {
        "rounds": rounds,
        "series_ms": series,
        "load_1m_normalized": {"min": min(loads), "median": st.median(loads), "max": max(loads)},
        "vs_ple256": {l: ratio_stats(series, l, base="ple256") for l in ("ple16", "ple64")},
    }
(OUT / "embedgemma2_memory_m5pro.json").write_text(json.dumps(doc, indent=1) + "\n")

# ---- text ----
L = []
w = L.append
w("EMBEDGEMMA 2: DEVICE MEMORY OF encode, lane H (M5 Pro, real checkpoint)")
w("")
w("Before = 16f5190 (the library), built in its own worktree and target dir. After = the lane H change:")
w("  (a) encode hands its activations and uploaded inputs back to the runtime before its one wait;")
w("  (b) PLE_RUN_BYTES 256 -> 16 MiB; (c) the q|k|v GEMM output and the MLP gate|up share one buffer.")
w("The harness is the same source for both builds (src/bin/profile_embedgemma2.rs mode memory-table / hash,")
w("src/bin/bench_embedgemma2.rs through bench_paired); only the library differs.")
w("")
w(doc["bounds"].rstrip())
w("")
w("RESULTS: DEVICE PEAK ABOVE THE LOADED MODEL, MiB (one fresh process per cell, deterministic)")
w("  single = one encode; back-to-back = the last of four encodes with no wait between;")
w("  sync-between = the last of three with rt.synchronize() between. Two runs of each build agree on every cell:")
w(f"  before stable: {doc['memory']['before_stable']}, after stable: {doc['memory']['after_stable']}.")
w("")
w(f"  {'mode':5} {'workload':8} | {'single':>9} {'back2back':>10} {'sync-betw':>10} | {'single':>9} {'back2back':>10} {'sync-betw':>10} | back2back")
w(f"  {'':5} {'':8} | {'BEFORE':>31} | {'AFTER':>31} | change")
cells = {(c["operands"], c["workload"], c["config"]): c for c in doc["memory"]["cells"]}
for ops in ("f32", "bf16"):
    for wl in ("1x16", "64x32", "32x256", "8x1024", "2x4096"):
        b = [cells[(ops, wl, c)]["before_mib"] for c in ("single", "back-to-back", "sync-between")]
        a = [cells[(ops, wl, c)]["after_mib"] for c in ("single", "back-to-back", "sync-between")]
        w(f"  {ops:5} {wl:8} | {b[0]:9.2f} {b[1]:10.2f} {b[2]:10.2f} | {a[0]:9.2f} {a[1]:10.2f} {a[2]:10.2f} | {a[1] / b[1] - 1:+.1%}")
w("")
w("RESULTS: OUTPUT BITS (FNV-1a of the f32 bit patterns; three consecutive encodes agree in every cell)")
w(f"  real checkpoint, 6 workloads x 2 modes x {{full, truncate_dim 128}}: before == after: {doc['hashes']['real_identical']}"
  f" (after run twice: {doc['hashes']['real_after'] == doc['hashes']['real_after_repeat']})")
w(f"  tiny model, 3 batches x 2 modes: before == after: {doc['hashes']['tiny_identical']}")
for ln in doc["hashes"]["real_after"]:
    w("    " + ln)
for ln in doc["hashes"]["tiny_after"]:
    w("    " + ln)
w("")
for name, title in (("f32", "RESULTS: TIME, ExactF32"), ("bf16", "RESULTS: TIME, Bf16"), ("f32_six_rounds_four_lanes", "SUPPLEMENT: TIME, ExactF32, first 6-round run, with a 64 MiB lane")):
    p = doc["paired"][name]
    ld = p["load_1m_normalized"]
    w(f"{title}: bench_paired, {len(p['rounds'])} rounds, order reversed on odd rounds, each value the min of 10 encodes")
    w(f"  machine load at lane start (1-minute load average / 18 cores): min {ld['min']:.2f}, median {ld['median']:.2f}, max {ld['max']:.2f};")
    w("  other lanes and builds share the GPU and the cores, so a single round can be several times slower.")
    for lane, stats in p["vs_before"].items():
        w(f"  lane {lane} against before:")
        w(f"    {'workload':14} {'before ms':>10} {'lane ms':>9} | {'min-of-N':>9} {'median':>7} {'[lo, hi]':>16} {'spread':>7} {'rounds<=1.03':>13}")
        for wl, s in stats.items():
            w(f"    {wl:14} {s['base_min_of_n_ms']:10.2f} {s['min_of_n_ms']:9.2f} | {s['min_of_n_ratio']:9.3f} {s['median_of_round_ratios']:7.3f}"
              f" [{s['ratio_lo']:6.3f},{s['ratio_hi']:6.3f}] {s['spread']:7.2f} {s['rounds_within_bound']:>9}/{s['rounds']}")
    w("  after_ple256 (the change with the old 256 MiB bound) against after (the PLE bound's own cost, B4):")
    w(f"    {'workload':14} | {'min-of-N':>9} {'median':>7} {'[lo, hi]':>16} {'spread':>7}")
    for wl, s in p["ple256_vs_after"].items():
        w(f"    {wl:14} | {s['min_of_n_ratio']:9.3f} {s['median_of_round_ratios']:7.3f} [{s['ratio_lo']:6.3f},{s['ratio_hi']:6.3f}] {s['spread']:7.2f}")
    w("")
w("RESULTS: THE PER-LAYER-INPUT BOUND ALONE (B4): one changed binary at 16, 64 and 256 MiB (BENCH_PLE_RUN_BYTES), interleaved")
w("  Each ratio is lane / the 256 MiB lane, over the rounds; min-of-N is the minimum over all rounds of each lane's per-process minimum.")
for name, sweep in doc["ple_sweeps"].items():
    ld = sweep["load_1m_normalized"]
    w(f"  {name}: {len(sweep['rounds'])} rounds, load/core median {ld['median']:.1f}, max {ld['max']:.1f}")
    w(f"    {'workload':10} {'lane':6} | {'256 MiB ms':>10} {'lane ms':>9} | {'min-of-N':>9} {'median':>7} {'[lo, hi]':>16} {'spread':>7}")
    for lane, stats in sweep["vs_ple256"].items():
        for wl, s in stats.items():
            w(f"    {wl:10} {lane:6} | {s['base_min_of_n_ms']:10.2f} {s['min_of_n_ms']:9.2f} | {s['min_of_n_ratio']:9.3f} {s['median_of_round_ratios']:7.3f}"
              f" [{s['ratio_lo']:6.3f},{s['ratio_hi']:6.3f}] {s['spread']:7.2f}")
w("")
w("READING (each claim labelled verified / inferred / unverified)")
w("  * Memory (verified, deterministic, repeated): B2 holds. One encode and four back to back peak at the same figure within 0.3 MiB")
w("    (the second call's own small output buffer); 400.5 / 400.3 / 400.3 MiB at 32x256 / 8x1024 / 2x4096 against the 400 + 1 bound; no cell")
w("    above its 16f5190 value. 736.5 -> 400.5 is the 240 (PLE) + 96 (alias) the lane D accounting predicted; the 96 was inferred there and is measured here.")
w("  * Output bits (verified): B1 holds on the real checkpoint (6 workloads, both modes, full and truncated, each from three consecutive")
w("    encodes) and on the tiny model (3 batches, both modes, one of them two forwards). The 16 and 64 MiB PLE bounds were also hashed on 16f5190")
w("    itself (the library unchanged) before any change was made: identical, so the PLE bound moves no bit by itself.")
w("  * PLE bound time (B4, verified to the extent the noise allows): one binary at 16 MiB against 256 MiB. The three small-row sweeps and the")
w("    8192-row sweeps put the ratio at 0.96 to 1.03 (median of rounds) and 0.90 to 1.00 (min-of-N); the only cell at 1.0303 (f32 2x4096, 8 rounds, load")
w("    21 per core, spread 1.75) was repeated with 12 rounds and read 1.005 / 0.991, so 16 MiB is the default. A first bf16 sweep at 1024-4096 rows ran at a")
w("    load of 14-38 per core and gave min-of-N 1.047 at 32x32 with medians 0.90-0.96 (spreads 2.2-4.6): it was repeated once (spreads 1.04-1.10, ratios 0.98-1.00).")
w("    The 64x32 f32 cell that read 1.045 and 1.074 in the two paired runs is not the PLE bound: the quiet sweep reads 0.993 / 0.984 there.")
w("  * Time against 16f5190 (B3): NOT met on the registered statistic, and the registered response (repeat) did not converge. Every series of")
w("    the paired runs has a max/min round spread of 1.2 to 28 (the machine ran at 100 to 360 load average on 18 cores, with other lanes on the GPU),")
w("    so the median over rounds of lane/before is not a stable statistic here: f32 medians 0.81-1.25 (1x16: 1.247 with a spread of 28), 2x4096 1.065 in the first run;")
w("    bf16 medians 0.94-1.00. The statistic that survives the contention is min-of-N over all rounds (10 rounds x 10 encodes per lane): after / before")
w("    = 0.92 / 1.05 / 0.90 / 0.98 / 0.96 / 0.86 (f32, 10 rounds) and 1.00 / 1.05 / 1.00 / 1.02 / 1.04 / 1.04 (f32, 6 rounds), 1.03 / 0.98 / 1.00 / 1.01 / 0.88 / 0.88 (bf16)")
w("    at 1x16 / 64x32 / 32x256 / 8x1024 / 2x4096 / ragged. So: no slowdown beyond +5.4 % shows in any series (verified as a bound of about 5 %, which is")
w("    the noise floor of this machine), and the 5.4 % cell (64x32) is not the PLE bound (above) and not the alias or the early release either: the changed")
w("    build with the old 256 MiB bound reads 0.98 and 1.01 there. Whether the change is within 3 % of 16f5190 at every workload is UNVERIFIED on this machine.")
w("  * Why none is expected (inferred): the change adds no dispatch to a forward except 22 more per-layer-input GEMM+scale+norm triples at 8192 rows")
w("    (66 extra dispatches of a few us each, against 372 in a forward and hundreds of ms), the alias changes which buffer a GEMM writes, not what it computes,")
w("    and the early release moves buffer ownership, not GPU work.")
w("")
(OUT / "embedgemma2_memory_m5pro.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
