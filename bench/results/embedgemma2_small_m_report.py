"""Merge the raw runs behind bench/results/embedgemma2_small_m_m5pro.txt (analysis only; ships nothing).

    python embedgemma2_small_m_report.py <results dir> <raw dir> <lane-before> <lane-after> <lane-control>

<raw dir> holds, from `bench_paired` (one run per precision and workload group):
    paired_lat_f32.json    (1x16, 1x64)
    paired_thr64_f32.json  (64x32)
    paired_thr8_f32.json   (8x1024)
and, from `bench_gemm_routes` (BENCH_LANES=nnsmall), the stdout of
    nnsmall_f32.txt
It writes embedgemma2_small_m_m5pro.json into <results dir> and prints the
tables the .txt quotes.

Statistics, all computed here from the per-round figures in the raw files:
  * "all rounds": the median over rounds of the per-round ratio after / before,
    with the lowest and highest round, and the same for the control lane (the
    before binary run a second time), which is what noise looks like.
  * "min of rounds": the house method, min over rounds of each lane's min-of-N,
    after against before.
  * "clean rounds": rounds in which the control lane is within 10 % of the
    before lane on that workload (nothing else was loading the GPU unevenly
    across the three runs of that round); the median ratio over them and how
    many there were. A delta whose [lo, hi] over all rounds straddles 1.0 and
    whose control spread is as wide is noise.
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

results = Path(sys.argv[1])
raw = Path(sys.argv[2])
BEFORE, AFTER, CONTROL = sys.argv[3], sys.argv[4], sys.argv[5]
CLEAN = 0.10


def load(name):
    return json.loads((raw / name).read_text())


def stats(row):
    b, a, c = (row[n]["ms_min_all"] for n in (BEFORE, AFTER, CONTROL))
    ratio = [x / y for x, y in zip(a, b)]
    noise = [x / y for x, y in zip(c, b)]
    clean = [r for r, n in zip(ratio, noise) if abs(n - 1) <= CLEAN]
    return {
        "rounds": len(b),
        "before_min": min(b),
        "after_min": min(a),
        "control_min": min(c),
        "before_median": st.median(b),
        "after_median": st.median(a),
        "ratio_median": st.median(ratio),
        "ratio_lo": min(ratio),
        "ratio_hi": max(ratio),
        "control_median": st.median(noise),
        "control_lo": min(noise),
        "control_hi": max(noise),
        "min_ratio": min(a) / min(b),
        "clean_rounds": len(clean),
        "clean_ratio_median": st.median(clean) if clean else None,
        "clean_ratio_lo": min(clean) if clean else None,
        "clean_ratio_hi": max(clean) if clean else None,
    }


def parse_nnsmall(path):
    """[(label, {arm: us})] from `bench_gemm_routes` lane nnsmall's stdout."""
    out, cur = [], None
    for line in path.read_text().splitlines():
        m = re.match(r"nnsmall (f32|bf16) (.+?) M=(\d+) N=(\d+) K=(\d+)", line)
        if m:
            cur = {"dtype": m[1], "name": m[2], "m": int(m[3]), "n": int(m[4]), "k": int(m[5]), "arms": {}, "ratios": {}}
            out.append(cur)
            continue
        m = re.match(r"\s+(\S+)\s+([0-9.]+) us\s+plain/this ([0-9.]+) \[([0-9.]+)–([0-9.]+)\]", line)
        if m and cur is not None:
            cur["arms"][m[1]] = float(m[2])
            cur["ratios"][m[1]] = [float(m[3]), float(m[4]), float(m[5])]
    return out


doc = {
    "lane_names": {"before": BEFORE, "after": AFTER, "control": CONTROL},
    "workloads": [],
    "encode": {},
    "stats": {},
}
for prec in ("f32",):
    rows = []
    for grp in ("lat", "thr64", "thr8"):
        d = load(f"paired_{grp}_{prec}.json")
        rows += d["summary"]
        doc.setdefault("env", {})[f"{grp}_{prec}"] = d.get("env", {})
    doc["encode"][prec] = {"summary": rows}
    doc["stats"][prec] = {r["workload"]: stats(r) for r in rows}
    if not doc["workloads"]:
        doc["workloads"] = [r["workload"] for r in rows]

gemm = {}
for prec in ("f32",):
    cases = parse_nnsmall(raw / f"nnsmall_{prec}.txt")
    gemm[prec] = cases
doc["gemm"] = gemm
g16 = {}
for prec, cases in gemm.items():
    g16[prec] = [
        {"name": c["name"], "plain_us": c["arms"]["plain"], "production_us": c["arms"]["production"]}
        for c in cases
        if c["m"] == 16
    ]
doc["gemm_16_rows"] = g16
(results / "embedgemma2_small_m_m5pro.json").write_text(json.dumps(doc, indent=1))

print("WHOLE ENCODE: after / before (ms = min of rounds' min-of-N)")
print(f"{'prec':5} {'workload':8} {'rounds':>6} {'before':>9} {'after':>9} {'min-of-rounds':>14} {'median ratio [lo,hi]':>24} {'control [lo,hi]':>22} {'clean n':>8} {'clean median':>13}")
for prec, ws in doc["stats"].items():
    for w, s in ws.items():
        cm = f"{s['clean_ratio_median']:.3f}" if s["clean_ratio_median"] else "-"
        print(
            f"{prec:5} {w:8} {s['rounds']:>6} {s['before_min']:>9.3f} {s['after_min']:>9.3f} {s['min_ratio']:>14.3f} "
            f"{s['ratio_median']:>8.3f} [{s['ratio_lo']:.2f},{s['ratio_hi']:.2f}]"
            f" {s['control_median']:>8.3f} [{s['control_lo']:.2f},{s['control_hi']:.2f}] {s['clean_rounds']:>8} {cm:>13}"
        )
print()
print("ONE GEMM, us per call (median of interleaved rounds); plain / this = median of per-round ratios [lo, hi]")
for prec, cases in gemm.items():
    for c in cases:
        arms = "  ".join(f"{k} {v:.1f}" + (f" ({c['ratios'][k][0]:.2f}x [{c['ratios'][k][1]:.2f},{c['ratios'][k][2]:.2f}])" if k != "plain" else "") for k, v in c["arms"].items())
        print(f"{prec:5} {c['name']:<14} M={c['m']:<3} K={c['k']:<5} N={c['n']:<5} {arms}")
