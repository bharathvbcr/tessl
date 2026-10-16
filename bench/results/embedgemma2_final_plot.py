"""Plot bench/results/embedgemma2_final_m5pro.json (analysis only; ships nothing).

The JSON is `bench_paired --format workloads` output for four lanes of `bench_embedgemma2`:
`before_async` (60a6240, the session's starting point, in its best mode, BENCH_ASYNC=1; the base),
`control` (a copy of that binary, so the run measures its own noise), `after_f32` and
`after_bf16` (eg2/integrate, ExactF32 and Bf16). 60a6240's default mode (every dispatch committed
and waited, what a library caller got) is not a lane: its 1x16 ratio series exceeded the 1.25
spread gate in the first attempt, and its cost is measured in embedgemma2_before_after_m5pro.txt. Left: each lane's median over rounds of its min-of-N encode
time, log scale. Middle: speed-up against `before_async` (1 / the median per-round ratio, whiskers
at the lowest and highest round). Right: device peak above the loaded model.

    uv run --no-project --with matplotlib python embedgemma2_final_plot.py <results dir>

Writes embedgemma2_final_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_final_m5pro.json").read_text())
rows = doc["summary"]
lanes = [l["name"] for l in doc["lanes"]]
base = lanes[0]
works = [r["workload"].replace("+", "+\n") for r in rows]

COLORS = {
    "before_async": "#2a78d6",
    "control": "#8a8985",
    "after_f32": "#eb6834",
    "after_bf16": "#1f9e6e",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (ax_t, ax_s, ax_m) = plt.subplots(1, 3, figsize=(18, 6.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
w = 0.8 / len(lanes)
for i, lane in enumerate(lanes):
    xs = [k + (i - (len(lanes) - 1) / 2) * w for k in range(len(rows))]
    ax_t.bar(xs, [r[lane]["ms_min_median"] for r in rows], width=w, color=COLORS.get(lane, MUTED), label=lane)
ax_t.set_yscale("log")
ax_t.set_xticks(range(len(rows)), works, fontsize=8, color=MUTED, rotation=35, ha="right")
ax_t.set_ylabel("encode, ms (median over rounds of min-of-N; log)", fontsize=9, color=MUTED)
ax_t.set_title("Encode time per workload", loc="left", fontsize=11, color=INK)
ax_t.legend(fontsize=8.5, frameon=False, ncol=2)
style(ax_t)

others = [l for l in lanes if l != base]
w = 0.8 / len(others)
for i, lane in enumerate(others):
    xs = [k + (i - (len(others) - 1) / 2) * w for k in range(len(rows))]
    sp = [1 / r[lane]["vs_base"]["median"] for r in rows]
    lo = [1 / r[lane]["vs_base"]["hi"] for r in rows]
    hi = [1 / r[lane]["vs_base"]["lo"] for r in rows]
    ax_s.bar(xs, sp, width=w, color=COLORS.get(lane, MUTED), label=lane)
    ax_s.errorbar(xs, sp, yerr=[[s - l for s, l in zip(sp, lo)], [h - s for s, h in zip(sp, hi)]],
                  fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    for x, s in zip(xs, sp):
        ax_s.text(x, s * 1.04, f"{s:.2f}x", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
ax_s.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--")
ax_s.set_yscale("log")
ax_s.set_xticks(range(len(rows)), works, fontsize=8, color=MUTED, rotation=35, ha="right")
ax_s.yaxis.set_major_locator(FixedLocator([0.5, 1, 1.5, 2, 3, 4]))
ax_s.yaxis.set_minor_locator(NullLocator())
ax_s.yaxis.set_major_formatter(ScalarFormatter())
ax_s.set_ylim(0.85, 4.2)
ax_s.set_ylabel(f"speed-up vs {base} (log; whiskers: lowest/highest round)", fontsize=9, color=MUTED)
ax_s.set_title(f"Speed-up against 60a6240 ({base})", loc="left", fontsize=11, color=INK)
ax_s.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
style(ax_s)

w = 0.8 / len(lanes)
for i, lane in enumerate(lanes):
    xs = [k + (i - (len(lanes) - 1) / 2) * w for k in range(len(rows))]
    ax_m.bar(xs, [r[lane].get("device_peak_mib_max", 0.0) for r in rows], width=w,
             color=COLORS.get(lane, MUTED), label=lane)
ax_m.set_xticks(range(len(rows)), works, fontsize=8, color=MUTED, rotation=35, ha="right")
ax_m.set_ylabel("device peak above the model, MiB (max over rounds)", fontsize=9, color=MUTED)
ax_m.set_title("Device memory", loc="left", fontsize=11, color=INK)
style(ax_m)

fig.suptitle("EmbeddingGemma 2 text encode on Apple M5 Pro: 60a6240 against eg2/integrate "
             f"({doc.get('rounds', '?')} interleaved rounds)", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(root / "embedgemma2_final_m5pro.png", dpi=150, facecolor="#fcfcfb")
print("wrote", root / "embedgemma2_final_m5pro.png")
