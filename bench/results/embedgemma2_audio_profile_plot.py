"""Plot and tabulate bench/results/embedgemma2_audio_profile_m5pro.json (analysis only; ships nothing).

The JSON has two profiles of the audio tower, `before` (the 26-stage forward at a4eb1b8, which is df5d329's
kernels with the stage hook) and `after` (the 19-stage forward at the lane's head). Each is
`profile_embedgemma2_audio`'s output: per workload, every stage's `alone_ms` (the stage run alone with the
harness floor taken off) and the variants' raw per-round times. The stages differ between the two (fused
dispatches), so they are compared by the work they do: GEMMs, attention, the subsampling convs, and the
row passes (norms, clamps, residual adds, GLU, depthwise conv, bias).

    python embedgemma2_audio_profile_plot.py <results dir>

Writes embedgemma2_audio_profile_m5pro.png and prints the grouped table.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_audio_profile_m5pro.json").read_text())

GROUPS = ["GEMMs (exact f32)", "attention", "subsampling convs", "row passes"]


def group_of(stage):
    if "GEMM" in stage:
        return GROUPS[0]
    if stage.startswith("attention"):
        return GROUPS[1]
    if stage.startswith("subsample"):
        return GROUPS[2]
    return GROUPS[3]


def grouped(profile, workload):
    w = next(x for x in profile["workloads"] if x["workload"] == workload)
    out = dict.fromkeys(GROUPS, 0.0)
    for s in w["stages"]:
        out[group_of(s["stage"])] += s["alone_ms"]
    enc = min(next(v for v in w["variants"] if v["name"] == "encode")["rounds_ms"])
    return out, enc


# Colors: categorical slots 1-4 of the reference palette (validated), one per group, in fixed order.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DARK_TEXT, MUTED = "#0b0b0b", "#52514e"

workloads = [w["workload"] for w in doc["before"]["workloads"]]
fig, axes = plt.subplots(len(workloads), 1, figsize=(9.5, 1.55 * len(workloads) + 1.1), sharex=False)
if len(workloads) == 1:
    axes = [axes]

# The GEMM kernels are the same before and after, so on a shared GPU their time is the in-run control:
# "other / GEMM" (everything but the GEMMs, per ms of GEMM) does not move with the load the way the
# absolute times do.
print(f"{'workload':<14}{'side':<8}" + "".join(f"{g:>22}" for g in GROUPS) + f"{'encode ms':>12}{'other/GEMM':>12}")
for ax, wl in zip(axes, workloads):
    rows = []
    for side in ("before", "after"):
        g, enc = grouped(doc[side], wl)
        rows.append((side, g, enc))
        other = sum(v for k, v in g.items() if k != GROUPS[0])
        print(f"{wl:<14}{side:<8}" + "".join(f"{g[k]:>22.2f}" for k in GROUPS) + f"{enc:>12.2f}{other / g[GROUPS[0]]:>12.3f}")
    for yi, (side, g, enc) in enumerate(rows):
        left = 0.0
        for k, c in zip(GROUPS, LIGHT):
            ax.barh(yi, g[k], left=left, color=c, height=0.62, edgecolor="#fcfcfb", linewidth=2,
                    label=k if yi == 0 else None)
            left += g[k]
        ax.text(left + 0.01 * max(r[1][k] for r in rows for k in GROUPS) * 6, yi, f"{left:.1f} ms (sum of stages)",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks([0, 1], ["before", "after"])
    ax.invert_yaxis()
    ax.set_title(wl.replace("_", " "), loc="left", fontsize=10, color=DARK_TEXT)
    ax.set_xlabel("stage time alone, ms (min over rounds, harness floor removed)", fontsize=8.5, color=MUTED)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.28)
axes[0].legend(ncol=4, fontsize=8.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.22))
fig.suptitle("EmbeddingGemma 2 audio tower, per-stage GPU time (Apple M5 Pro, shared GPU)\n"
             "the GEMMs did not change: read the other bars against the GEMM bar of the same row",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(root / "embedgemma2_audio_profile_m5pro.png", dpi=160)
