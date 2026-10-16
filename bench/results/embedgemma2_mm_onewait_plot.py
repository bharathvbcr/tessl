"""Plot bench/results/embedgemma2_mm_onewait_m5pro.json (analysis only; ships nothing).

The JSON is `bench_paired --format workloads` output for three lanes of `bench_embedgemma2_mm`
(the facade's `encode` over texts, images and clips): `before` (27a0407, two host waits for a call
with media), `after` (the lane's head, one) and `control` (the `before` binary again, so the run
measures its own noise). Per workload and lane it holds each round's min-of-N call time
(`ms_min_all`); the top row shows every round as a dot, the median as a bar, and the median
per-round ratio against `before` with its lowest and highest round. The bottom row is the device
peak above the loaded model (`device_peak_mib_max`) and the host waits of one call. The waits
are the `sync_waits` field each lane printed (`infer_trace`, counted on a separate call before
the timed ones); `bench_paired` does not carry it into the JSON, so WAITS below restates what
all 24 processes of each lane printed (before and control 2, after 1; see the results `.txt`).

    python embedgemma2_mm_onewait_plot.py <results dir>

Writes embedgemma2_mm_onewait_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_mm_onewait_m5pro.json").read_text())
rows = doc["summary"]
lanes = [l["name"] for l in doc["lanes"]]
base = lanes[0]
WAITS = {"before": 2, "after": 1, "control": 2}

COLORS = {"before": "#2a78d6", "after": "#eb6834", "control": "#8a8985"}
INK, MUTED = "#0b0b0b", "#52514e"

fig, axes = plt.subplots(2, len(rows), figsize=(3.9 * len(rows), 8.2), gridspec_kw={"height_ratios": [3, 1.4]})
for col, row in enumerate(rows):
    ax = axes[0][col]
    for x, lane in enumerate(lanes):
        d = row[lane]
        c = COLORS.get(lane, MUTED)
        n = len(d["ms_min_all"])
        ax.hlines(d["ms_min_median"], x - 0.3, x + 0.3, color=c, linewidth=3)
        ax.scatter([x + (i - (n - 1) / 2) * 0.045 for i in range(n)], d["ms_min_all"],
                   s=26, color=c, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        if lane != base:
            r = d["vs_base"]
            ax.text(x, d["ms_min_median"], f"{r['median']:.2f}x\n[{r['lo']:.2f}, {r['hi']:.2f}]",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(lanes)), lanes, fontsize=9, color=MUTED)
    ax.set_title(row["workload"].replace("_", " "), loc="left", fontsize=10.5, color=INK)
    lo_all = min(min(row[l]["ms_min_all"]) for l in lanes)
    hi_all = max(max(row[l]["ms_min_all"]) for l in lanes)
    ax.set_ylim(lo_all - 0.08 * (hi_all - lo_all) - 0.01 * lo_all, hi_all + 0.35 * (hi_all - lo_all))
    for x, lane in enumerate(lanes):
        ax.text(x, lo_all - 0.04 * (hi_all - lo_all) - 0.005 * lo_all, f"best {min(row[lane]['ms_min_all']):.1f}",
                ha="center", va="top", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    bx = axes[1][col]
    peaks = [row[l]["device_peak_mib_max"] for l in lanes]
    bx.bar(range(len(lanes)), peaks, color=[COLORS.get(l, MUTED) for l in lanes], width=0.55)
    for x, (lane, p) in enumerate(zip(lanes, peaks)):
        w = WAITS.get(lane)
        label = f"{p:.1f}" + (f"\n{w} wait{'s' if w != 1 else ''}" if w is not None else "")
        bx.text(x, p, label, ha="center", va="bottom", fontsize=8.5, color=INK)
    bx.set_xticks(range(len(lanes)), lanes, fontsize=9, color=MUTED)
    bx.set_ylim(0, max(peaks) * 1.35)
    bx.tick_params(labelsize=8.5, colors=MUTED)
    bx.grid(axis="y", color="#e6e5e1", linewidth=0.7)
    bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
axes[0][0].set_ylabel("encode, ms (min of 8 per round; bar = median)", fontsize=9, color=MUTED)
axes[1][0].set_ylabel("device peak above model, MiB", fontsize=9, color=MUTED)
fig.suptitle("EmbeddingGemma 2 multimodal facade: before (27a0407, 2 host waits) vs after (1 wait), "
             "8 interleaved rounds, Apple M5 Pro\n"
             "label: median per-round ratio to before, [lowest, highest round]; 'best' = fastest round. "
             "Control = the before binary again:\n"
             "its own ratio and range are the noise floor of this run; a difference inside it is not a result. "
             "Time axes are zoomed.",
             x=0.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(root / "embedgemma2_mm_onewait_m5pro.png", dpi=140)
