"""Plot bench/results/embedgemma2_pre_m5pro.json (analysis only; ships nothing).

The JSON is `bench_paired --format workloads` output for two lanes: `tessl`
(`bench_embedgemma2_pre`, tessl::embedgemma2_pre) and `hf` (`bench/embedgemma2_pre_hf.py`,
transformers 5.19's processors on the same inputs). Per workload and lane it holds each round's
min-of-N time (`ms_min_all`); the figure shows every round as a dot, the median over rounds as a
bar (log scale: the workloads span three decades), and above each pair the median per-round ratio
hf / tessl with the lowest and highest round. A ratio range that straddles 1 is not a result.

    uv run --no-project --with matplotlib python embedgemma2_pre_plot.py [results dir]

Writes embedgemma2_pre_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_pre_m5pro.json").read_text())
rows = doc["summary"]
COLORS = {"tessl": "#eb6834", "hf": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(13, 5.6))
width = 0.38
for i, row in enumerate(rows):
    for j, lane in enumerate(("tessl", "hf")):
        d = row[lane]
        x = i + (j - 0.5) * width
        ax.bar(x, d["ms_min_median"], width * 0.92, color=COLORS[lane], alpha=0.28, zorder=1)
        ax.hlines(d["ms_min_median"], x - width * 0.46, x + width * 0.46, color=COLORS[lane], linewidth=2.5, zorder=2)
        n = len(d["ms_min_all"])
        ax.scatter([x + (k - (n - 1) / 2) * width * 0.9 / n for k in range(n)], d["ms_min_all"], s=14,
                   color=COLORS[lane], edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
    r = row["hf"]["vs_base"]
    top = max(max(row[l]["ms_min_all"]) for l in ("tessl", "hf"))
    ax.text(i, top * 1.25, f"{r['median']:.2f}x\n[{r['lo']:.2f}, {r['hi']:.2f}]", ha="center", va="bottom",
            fontsize=8.5, color=INK)
ax.set_yscale("log")
ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
ax.set_xticks(range(len(rows)), [r["workload"] for r in rows], rotation=20, ha="right", fontsize=9, color=MUTED)
ax.set_ylabel("ms per call (min of 15 per round; bar = median over rounds; dots = rounds)", fontsize=9, color=MUTED)
ax.tick_params(labelsize=8.5, colors=MUTED)
ax.grid(axis="y", color="#e6e5e1", linewidth=0.7, which="both")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("tessl", "hf")]
ax.legend(handles, ["tessl (embedgemma2_pre)", "transformers 5.19 (AutoProcessor)"], frameon=False, fontsize=9,
          loc="upper left")
fig.suptitle("EmbeddingGemma 2 preprocessing, tessl native vs transformers processors, same inputs, Apple M5 Pro "
             "(shared host)\nlabel: median per-round ratio hf / tessl [lowest, highest round]; above 1 = tessl faster",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(root / "embedgemma2_pre_m5pro.png", dpi=160)
