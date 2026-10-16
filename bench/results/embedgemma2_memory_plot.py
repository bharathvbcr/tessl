"""Plots for bench/results/embedgemma2_memory_m5pro.txt (analysis only; ships nothing).

Reads embedgemma2_memory_m5pro.json (written by embedgemma2_memory_report.py) and writes
embedgemma2_memory_m5pro.png.

    python embedgemma2_memory_plot.py <results dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_memory_m5pro.json").read_text())
cells = {(c["operands"], c["workload"], c["config"]): c for c in doc["memory"]["cells"]}
WLS = ["1x16", "64x32", "32x256", "8x1024", "2x4096"]
BEFORE, AFTER, GREY = "#c44e52", "#4c72b0", "#8c8c8c"

fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

# Panel row 1: device peak above the model, back-to-back encodes, both builds, both modes.
for col, ops in enumerate(("f32", "bf16")):
    ax = fig.add_subplot(gs[0, col])
    x = range(len(WLS))
    b = [cells[(ops, w, "back-to-back")]["before_mib"] for w in WLS]
    a = [cells[(ops, w, "back-to-back")]["after_mib"] for w in WLS]
    b1 = [cells[(ops, w, "single")]["before_mib"] for w in WLS]
    ax.bar([i - 0.2 for i in x], b, 0.38, color=BEFORE, label="16f5190, back-to-back encodes")
    ax.bar([i + 0.2 for i in x], a, 0.38, color=AFTER, label="changed, back-to-back encodes")
    ax.scatter([i - 0.2 for i in x], b1, marker="_", s=420, color="black", zorder=3, label="16f5190, one encode")
    for i, (bv, av) in enumerate(zip(b, a)):
        ax.text(i - 0.2, bv + 20, f"{bv:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, av + 20, f"{av:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x), WLS)
    ax.set_ylim(0, 2100)
    ax.set_ylabel("device peak above the loaded model (MiB)")
    ax.set_title(f"{'ExactF32' if ops == 'f32' else 'Bf16'}: peak of the last of four encodes")
    ax.grid(axis="y", alpha=0.3)
    if col == 0:
        ax.legend(fontsize=8, loc="upper left")

# Panel row 1, right: where the 736 -> 400 MiB of one set at 8192 rows goes (f32 itemization).
ax = fig.add_subplot(gs[0, 2])
steps = [("16f5190\none set", 736.47), ("PLE bound\n256 -> 16 MiB", -240.0), ("q|k|v = gate|up\none buffer", -96.0)]
run = 0.0
for i, (label, v) in enumerate(steps):
    if i == 0:
        ax.bar(i, v, color=BEFORE)
        run = v
        ax.text(i, v + 12, f"{v:.0f}", ha="center", fontsize=9)
    else:
        ax.bar(i, -v, bottom=run + v, color=GREY)
        run += v
        ax.text(i, run + -v + 12, f"{v:+.0f}", ha="center", fontsize=9)
ax.bar(len(steps), run, color=AFTER)
ax.text(len(steps), run + 12, f"{run:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["changed\none set"], fontsize=8)
ax.set_ylabel("MiB, 8192 rows, ExactF32")
ax.set_title("one activation set (single encode)")
ax.grid(axis="y", alpha=0.3)

# Row 2: time, after / before per workload, both modes: every round (dots), median (bar), min-of-N (diamond).
for col, ops in enumerate(("f32", "bf16")):
    ax = fig.add_subplot(gs[1, col])
    p = doc["paired"][ops]
    series = p["series_ms"]
    wls = list(series["before"].keys())
    for i, wl in enumerate(wls):
        ratios = [a / b for a, b in zip(series["after"][wl], series["before"][wl])]
        ax.scatter([i] * len(ratios), ratios, s=14, color=AFTER, alpha=0.45, zorder=2)
        s = p["vs_before"]["after"][wl]
        ax.hlines(s["median_of_round_ratios"], i - 0.3, i + 0.3, color="black", lw=2, zorder=3)
        ax.scatter([i + 0.0], [s["min_of_n_ratio"]], marker="D", color="#dd8452", zorder=4, s=36)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.axhline(1.03, color=BEFORE, lw=0.8, ls="--")
    ax.set_xticks(range(len(wls)), wls, fontsize=7, rotation=15)
    lo, hi = 0.5, 2.0
    ax.set_ylim(lo, hi)
    off = sum(
        not lo <= a / b <= hi for wl in wls for a, b in zip(series["after"][wl], series["before"][wl])
    )
    ax.text(0.99, 0.02, f"{off} of {len(wls) * len(p['rounds'])} rounds off scale", transform=ax.transAxes,
            ha="right", fontsize=7, color=GREY)
    ax.set_ylabel("changed / 16f5190 (encode ms, per round)")
    ld = p["load_1m_normalized"]
    ax.set_title(
        f"{'ExactF32' if ops == 'f32' else 'Bf16'}: {len(p['rounds'])} paired rounds (dots), median (bar), min-of-N (diamond)\n"
        f"shared machine, load/core median {ld['median']:.1f}, max {ld['max']:.1f}",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)

# Row 2, right: the PLE bound's own cost, one binary at 16 MiB over the same at 256 MiB, by padded rows.
ax = fig.add_subplot(gs[1, 2])
sweeps = doc["ple_sweeps"]
ROWS = {"32x32": 1024, "64x32": 2048, "128x32": 4096, "32x256": 8192, "8x1024": 8192, "2x4096": 8192}
for k, (ops, color) in enumerate((("f32", AFTER), ("bf16", "#55a868"))):
    pts = []
    for name, sweep in sweeps.items():
        # The first bf16 run at 1024-4096 rows ran at a load of 14-38 per core and is in the text only.
        # A "repeat" is the same cell run again (text only).
        if not name.startswith(ops) or "first run" in name or "repeat" in name:
            continue
        for wl, s in sweep["vs_ple256"]["ple16"].items():
            pts.append((wl, s))
    pts.sort(key=lambda p: (ROWS[p[0]], p[0]))
    xs = [i + (k - 0.5) * 0.3 for i in range(len(pts))]
    ax.bar(xs, [s["median_of_round_ratios"] for _, s in pts], 0.28, color=color,
           label=f"{'ExactF32' if ops == 'f32' else 'Bf16'}: median of rounds")
    ax.scatter(xs, [s["min_of_n_ratio"] for _, s in pts], marker="D", color="#dd8452", zorder=4, s=28,
               label="min-of-N" if k == 0 else None)
    labels = [p[0] for p in pts]
ax.axhline(1.0, color="black", lw=0.8)
ax.axhline(1.03, color=BEFORE, lw=0.8, ls="--")
ax.set_ylim(0.8, 1.25)
ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=20)
ax.set_ylabel("16 MiB bound / 256 MiB bound (encode time)")
ax.set_title("time cost of the 16 MiB per-layer-input bound\n(dashed: the 1.03 limit fixed in advance)", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("EmbeddingGemma 2 encode, device memory and time: lane H against 16f5190 (M5 Pro, real checkpoint)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(root / "embedgemma2_memory_m5pro.png", dpi=130)
print("wrote", root / "embedgemma2_memory_m5pro.png")
