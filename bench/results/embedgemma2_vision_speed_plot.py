"""Plots for bench/results/embedgemma2_vision_speed_m5pro.txt (analysis only; ships nothing).

Reads embedgemma2_vision_speed_m5pro.json (bench_paired's report: every round of
every lane, the summary) and embedgemma2_vision_profile_m5pro.json (written by
`results_report vision`), and writes embedgemma2_vision_speed_m5pro.png.

    python embedgemma2_vision_speed_plot.py <results dir>

Three panels:
  1. encode_images time per lane at 1 image and at 8 images: the bar is the
     median over rounds of each round's min-of-N, the dots are the rounds;
  2. the same as ratios: each lane's time over the first lane's, per round
     (median, and the lowest and highest round as the whisker) - the spread is
     the honest error bar on a GPU other sessions share, a whisker that crosses
     1.0 is not a result;
  3. where the 1-image forward's time goes, per stage, for the three profile
     labels (each stage alone on the real shapes and weights, ms).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
speed = json.loads((root / "embedgemma2_vision_speed_m5pro.json").read_text())
profile = json.loads((root / "embedgemma2_vision_profile_m5pro.json").read_text())["runs"]

lanes = [lane["name"] for lane in speed["lanes"]]
workloads = [row["workload"] for row in speed["summary"]]
COLORS = ["#8c8c8c", "#4c72b0", "#dd8452"]
rows = {row["workload"]: row for row in speed["summary"]}

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.6]})

ax = axes[0]
width = 0.8 / len(lanes)
for li, lane in enumerate(lanes):
    xs = [wi + (li - (len(lanes) - 1) / 2) * width for wi in range(len(workloads))]
    meds = [rows[w][lane]["ms_min_median"] for w in workloads]
    ax.bar(xs, meds, width * 0.92, color=COLORS[li % len(COLORS)], label=lane)
    for x, w in zip(xs, workloads):
        ax.plot([x] * len(rows[w][lane]["ms_min_all"]), rows[w][lane]["ms_min_all"], "k.", markersize=4)
    for x, m in zip(xs, meds):
        ax.text(x, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(workloads)))
ax.set_xticklabels([w.replace("img", " image" if w == "1img" else " images") for w in workloads])
ax.set_yscale("log")
ax.set_ylabel("encode_images, ms (median over rounds of min-of-N; dots = rounds)")
ax.set_title("Wall time of one encode_images call")
ax.legend(fontsize=8)

ax = axes[1]
base = lanes[0]
for li, lane in enumerate(lanes[1:], start=1):
    xs = [wi + (li - len(lanes) / 2) * width * 1.4 for wi in range(len(workloads))]
    med = [rows[w][lane]["vs_base"]["median"] for w in workloads]
    lo = [rows[w][lane]["vs_base"]["lo"] for w in workloads]
    hi = [rows[w][lane]["vs_base"]["hi"] for w in workloads]
    ax.bar(xs, med, width * 1.3, color=COLORS[li % len(COLORS)], label=f"{lane} / {base}")
    ax.errorbar(xs, med, yerr=[[m - a for m, a in zip(med, lo)], [b - m for m, b in zip(med, hi)]], fmt="none", ecolor="k", capsize=4)
    for x, m in zip(xs, med):
        ax.text(x, m, f"{m:.2f}x", ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xticks(range(len(workloads)))
ax.set_xticklabels([w.replace("img", " image" if w == "1img" else " images") for w in workloads])
ax.set_ylabel(f"time relative to {base} (lower is faster); whiskers = lowest/highest round")
ax.set_title("Paired, interleaved rounds (order reversed on odd rounds)")
ax.legend(fontsize=8)


def stage_ms(label, workload="1img"):
    """Stage -> min over passes of the `only` ms per call, for one label."""
    out = {}
    for doc in profile[label]:
        wl = next(w for w in doc["workloads"] if w["workload"] == workload)
        for ci, name in enumerate(doc["classes"]):
            for v in wl["variants"]:
                if v["kind"] == "only" and v["class"] == ci and v["dispatches"] > 0:
                    out[name] = min(out.get(name, float("inf")), min(v["ms_per_call"]))
    return out


labels = list(profile)
per_label = {k: stage_ms(k) for k in labels}
stages = sorted({s for d in per_label.values() for s in d}, key=lambda s: -max(d.get(s, 0.0) for d in per_label.values()))
ax = axes[2]
bar = 0.8 / len(labels)
for li, label in enumerate(labels):
    ys = [si + (li - (len(labels) - 1) / 2) * bar for si in range(len(stages))]
    ax.barh(ys, [per_label[label].get(s, 0.0) for s in stages], bar * 0.92, color=COLORS[li % len(COLORS)], label=label)
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages, fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("ms for all 16 layers of one stage, run alone (1 image, 2304 padded rows)")
ax.set_title("Where the time goes (profile_embedgemma2_vision)")
ax.legend(fontsize=8)

fig.suptitle("EmbeddingGemma 2 vision tower on an Apple M5 Pro, real checkpoint", fontsize=12)
fig.tight_layout()
fig.savefig(root / "embedgemma2_vision_speed_m5pro.png", dpi=130)
print(f"wrote {root / 'embedgemma2_vision_speed_m5pro.png'}")
