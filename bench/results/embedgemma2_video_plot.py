"""Plot bench/results/embedgemma2_video_m5pro.json (analysis only; ships nothing).

The JSON is `bench_embedgemma2_video` output, three consecutive runs on a machine shared with other
work. Per workload (CPU preprocessing, the vision tower, the whole call) and precision the figure
shows each run's min-of-15 as a dot and the best of the three as a bar, with the machine's load
average at the start of each run's workload beside it: the spread between runs is the point, the
bar is the least-disturbed run, and none of it is a clean measurement.

    python embedgemma2_video_plot.py <results dir>

Writes embedgemma2_video_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_video_m5pro.json").read_text())

# Categorical slots 1 (blue) and 2 (orange) of the reference palette; the ink is muted for text.
COLORS = {"f32": "#2a78d6", "bf16": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"
WORKLOADS = [("preprocess", "preprocess_video, CPU\n(32 frames, 640x360)"),
             ("tower", "vision tower\n(32 frames)"),
             ("encode", "whole call\n(tower + text encoder, one wait)")]

by = {}
for run in doc["runs"]:
    for r in run["rows"]:
        by.setdefault((r["workload"], r["precision"]), []).append((r["ms_min"], r["load_avg_1m"]))

fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(11.5, 4.6))
for ax, (w, title) in zip(axes, WORKLOADS):
    precisions = ["f32"] if w == "preprocess" else ["f32", "bf16"]
    for x, p in enumerate(precisions):
        pts = by[(w, p)]
        best = min(ms for ms, _ in pts)
        ax.bar(x, best, width=0.55, color=COLORS[p], alpha=0.35)
        for k, (ms, load) in enumerate(pts):
            ax.plot([x + (k - 1) * 0.12], [ms], "o", color=COLORS[p], ms=6)
            ax.annotate(f"{load:.0f}", (x + (k - 1) * 0.12, ms), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=MUTED)
        ax.annotate(f"{best:.0f} ms", (x, 0), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=INK)
    ax.set_xticks(range(len(precisions)), precisions if w != "preprocess" else ["CPU"], color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ms, min of 15 calls", color=INK)
fig.suptitle("One 32-frame video, EmbeddingGemma 2 on an Apple M5 Pro (dots: three runs, labelled by 1-min "
             "load average; bar: best run)", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(root / "embedgemma2_video_m5pro.png", dpi=150)
print("wrote", root / "embedgemma2_video_m5pro.png")
