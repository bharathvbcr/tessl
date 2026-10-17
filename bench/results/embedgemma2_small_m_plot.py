"""Plot for bench/results/embedgemma2_small_m_m5pro.txt (analysis only; ships nothing).

Reads embedgemma2_small_m_m5pro.json and writes embedgemma2_small_m_m5pro.png:
left, the whole-encode time after the small-M GEMM lane against before it (the
ratio of every interleaved round, with the median and the control lane's
spread as the noise band), exact f32; right, the per-GEMM time at 16
rows (`bench_gemm_routes` lane `nnsmall`: plain single dispatch against the
split-K lane, interleaved).

    python embedgemma2_small_m_plot.py <results dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_small_m_m5pro.json").read_text())

BEFORE, AFTER, CONTROL = doc["lane_names"]["before"], doc["lane_names"]["after"], doc["lane_names"]["control"]
WORKLOADS = doc["workloads"]
COLORS = {"f32": "#4c72b0", "bf16": "#dd8452"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.15, 1]})

# Left: after / before per round, per workload.
width = 0.36
for j, prec in enumerate(doc["encode"]):
    paired = doc["encode"][prec]
    xs, meds = [], []
    for i, w in enumerate(WORKLOADS):
        row = next(r for r in paired["summary"] if r["workload"] == w)
        before, after, control = (row[n]["ms_min_all"] for n in (BEFORE, AFTER, CONTROL))
        ratios = [a / b for a, b in zip(after, before)]
        noise = [c / b for c, b in zip(control, before)]
        x = i + (j - 0.5) * width
        ax.vlines(x, min(ratios), max(ratios), color=COLORS[prec], lw=6, alpha=0.35)
        ax.plot([x] * len(ratios), ratios, "o", ms=4, color=COLORS[prec], label=f"{prec}: after / before" if i == 0 else None)
        med = sorted(ratios)[len(ratios) // 2]
        ax.plot([x - width / 3, x + width / 3], [med, med], color="black", lw=2)
        # The control lane (the before binary again) against the first: what noise looks like.
        ax.plot([x + width / 2.2] * len(noise), noise, "x", ms=4, color="#555555", label="control: before again / before" if (i == 0 and j == 0) else None)
ax.set_yscale("log")
ax.set_yticks([0.25, 0.5, 0.75, 1, 1.5, 2, 4], ["0.25", "0.5", "0.75", "1", "1.5", "2", "4"])
ax.axhline(1.0, color="black", lw=0.8, ls="--")
ax.set_xticks(range(len(WORKLOADS)), WORKLOADS)
ax.set_xlabel("workload (sequences x tokens)")
ax.set_ylabel("encode time, after / before (lower is faster)")
ax.set_title("Whole encode, interleaved rounds (log scale; bar: range; black: median)")
ax.legend(loc="lower right", fontsize=8)

# Right: per-GEMM time at 16 rows.
g = doc["gemm_16_rows"]
names = [r["name"] for r in g["f32"]]
pos = range(len(names))
for j, prec in enumerate(doc["encode"]):
    plain = [r["plain_us"] for r in g[prec]]
    prod = [r["production_us"] for r in g[prec]]
    off = (j - 0.5) * 0.36
    bx.bar([p + off - 0.08 for p in pos], plain, 0.16, color=COLORS[prec], alpha=0.45, label=f"{prec}: single dispatch")
    bx.bar([p + off + 0.08 for p in pos], prod, 0.16, color=COLORS[prec], label=f"{prec}: split-K lane")
bx.set_xticks(list(pos), names, rotation=30, ha="right", fontsize=8)
bx.set_ylabel("us per GEMM (median of interleaved rounds)")
bx.set_title("One GEMM at 16 rows (median of interleaved rounds)")
bx.legend(fontsize=8)

fig.tight_layout()
fig.savefig(root / "embedgemma2_small_m_m5pro.png", dpi=140)
print("wrote", root / "embedgemma2_small_m_m5pro.png")
