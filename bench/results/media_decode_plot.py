"""Plot bench/results/media_decode_m5pro.json (analysis only; ships nothing).

The JSON is the last stdout line of `bench_media_decode` (src/bin/bench_media_decode.rs):
minimum-of-N and median wall time per workload, native decoding by ImageIO, AudioToolbox and
AVFoundation, one call per workload, on a machine that was doing other work. Indicative, not a
paired comparison. Bars are the minimum; ticks mark the median. Log scale: the workloads span four
decades.

    ~/.venvs/ml-2026-10/bin/python media_decode_plot.py [results dir]

Writes media_decode_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
rows = json.loads((root / "media_decode_m5pro.json").read_text())
COLORS = {"megapixel": "#eb6834", "audio_s": "#2a78d6", "frame": "#1a9e6a"}
INK, MUTED = "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(11, 5.4))
labels = [f"{r['workload']}  ({r['units']:g} {r['unit'].replace('_', ' ')})" for r in rows]
ys = range(len(rows))
ax.barh(ys, [r["ms_min"] for r in rows], color=[COLORS[r["unit"]] for r in rows], alpha=0.85, height=0.62)
ax.scatter([r["ms_median"] for r in rows], ys, marker="|", s=260, color=INK, zorder=3, label="median")
for y, r in zip(ys, rows):
    ax.text(max(r["ms_min"], r["ms_median"]) * 1.12, y, f"{r['ms_min']:.1f} ms", va="center", fontsize=8.5, color=INK)
ax.set_yticks(list(ys), labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("wall time per call, ms (min of N; tick = median)", color=MUTED)
ax.set_title("tessl::media native decoding, one call per workload", loc="left", color=INK)
ax.grid(axis="x", alpha=0.25)
ax.legend(frameon=False, loc="upper center")
fig.tight_layout()
fig.savefig(root / "media_decode_m5pro.png", dpi=150)
print("wrote", root / "media_decode_m5pro.png")
