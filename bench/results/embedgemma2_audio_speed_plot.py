"""Plot bench/results/embedgemma2_audio_speed_m5pro.json (analysis only; ships nothing).

The JSON is `bench_paired --format workloads --evidence` output for three lanes of
`bench_embedgemma2_audio`: `before` (df5d329), `after` (the lane's head) and `control` (the
`before` binary again, so the run measures its own noise). Per workload and lane it holds each
round's min-of-N encode time (`ms_min_all`); the figure shows every round as a dot, the median as
a bar, and the median per-round ratio against `before` with its lowest and highest round.

    python embedgemma2_audio_speed_plot.py <results dir>

Writes embedgemma2_audio_speed_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_audio_speed_m5pro.json").read_text())
rows = doc["summary"]
lanes = [l["name"] for l in doc["lanes"]]
base = lanes[0]

# Categorical slots 1 (blue), 2 (orange) of the reference palette, validated; the control is the
# muted ink so it reads as "same code as before", not as a third thing.
COLORS = {"before": "#2a78d6", "after": "#eb6834", "control": "#8a8985"}
INK, MUTED = "#0b0b0b", "#52514e"

fig, axes = plt.subplots(1, len(rows), figsize=(3.9 * len(rows), 4.6))
if len(rows) == 1:
    axes = [axes]
for ax, row in zip(axes, rows):
    for x, lane in enumerate(lanes):
        d = row[lane]
        c = COLORS.get(lane, MUTED)
        ax.hlines(d["ms_min_median"], x - 0.3, x + 0.3, color=c, linewidth=3)
        ax.scatter([x + (i - 2.5) * 0.045 for i in range(len(d["ms_min_all"]))], d["ms_min_all"],
                   s=26, color=c, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        if lane != base:
            r = d["vs_base"]
            ax.text(x, d["ms_min_median"], f"{r['median']:.2f}x\n[{r['lo']:.2f}, {r['hi']:.2f}]",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(lanes)), lanes, fontsize=9, color=MUTED)
    ax.set_title(row["workload"].replace("_", " "), loc="left", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(max(row[l]["ms_min_all"]) for l in lanes) * 1.28)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("encode_audio, ms (min of N per round; median bar)", fontsize=9, color=MUTED)
fig.suptitle("EmbeddingGemma 2 audio tower: before (df5d329) vs after, interleaved rounds, Apple M5 Pro "
             "(label: median per-round ratio to before, [lowest, highest])\n"
             "control = the before binary again: its own ratio and range are the noise floor of this run; "
             "a difference inside it is not a result", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(root / "embedgemma2_audio_speed_m5pro.png", dpi=160)
