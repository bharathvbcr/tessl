"""Plot bench/results/embedgemma2_text_fusion_profile_m5pro.json (analysis only; ships nothing).

The JSON is `results_report text-fusion`'s merge of `profile_embedgemma2`
stage runs of the text encoder at 1x16, 64x32 and 8x1024 on the real checkpoint, three
passes per label (a class's time is its minimum over the passes): `f32 unfused`, `f32 fused`, `bf16 unfused`, `bf16 fused` (the replica of
`EmbedGemma2Model::forward` with and without the fused row passes). Each bar stacks the
stage-alone time of every class (min over the run's samples), grouped as the report
groups them; the row passes and the casts are the stages the fusion changed. Left
panels: time per forward by category. Right panels: dispatches per forward.

    uv run --no-project --with matplotlib python embedgemma2_text_fusion_profile_plot.py <results dir>

Writes embedgemma2_text_fusion_profile_m5pro.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_text_fusion_profile_m5pro.json").read_text())
runs = doc["runs"]
cats = doc["categories"]
labels = [l for l in ("f32 unfused", "f32 fused", "bf16 unfused", "bf16 fused") if l in runs]
workloads = ["1x16", "64x32", "8x1024"]

PREFIXES = {
    "GEMM": ["qkv GEMM", "o GEMM", "gate|up GEMM", "down GEMM", "PLE projection", "ple_gate", "ple_out"],
    "attention": ["attention"],
    "row passes": ["rms_norm", "residual add", "norm+residual", "PLE scale", "bf16 casts"],
    "gelu*up": ["gelu*up", "ple gelu"],
    "other elementwise": ["gather", "qk-norm"],
    "tail": ["tail"],
}
COLORS = {
    "GEMM": "#2a78d6",
    "attention": "#8a8985",
    "row passes": "#eb6834",
    "gelu*up": "#1f9e6e",
    "other elementwise": "#c9a227",
    "tail": "#b3b2ac",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def category(cls):
    for name, prefixes in PREFIXES.items():
        if any(cls.startswith(p) for p in prefixes):
            return name
    raise KeyError(cls)


def per_category(label, wl):
    """Each class's stage-alone time is the minimum over every pass of the label (the
    house min-of-N method); a category sums its classes."""
    best = {}
    dispatches = None
    for run in doc["passes"][label]:
        w = next(x for x in run["workloads"] if x["workload"] == wl)
        dispatches = w["dispatches_full"]
        for v in w["variants"]:
            if v["kind"] == "only":
                cls = v["name"].split(":", 1)[1]
                best[cls] = min(best.get(cls, float("inf")), min(v["ms_per_forward"]))
    ms = {c: 0.0 for c in cats}
    for cls, t in best.items():
        ms[category(cls)] += t
    return ms, dispatches


fig, axes = plt.subplots(2, len(workloads), figsize=(15, 8.2))
for j, wl in enumerate(workloads):
    ax_t, ax_d = axes[0][j], axes[1][j]
    bottoms = [0.0] * len(labels)
    for c in cats:
        vals = [per_category(l, wl)[0][c] for l in labels]
        ax_t.bar(range(len(labels)), vals, bottom=bottoms, color=COLORS[c], label=c, width=0.7)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for k, b in enumerate(bottoms):
        ax_t.text(k, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax_t.set_xticks(range(len(labels)), labels, fontsize=8.5, color=MUTED, rotation=20, ha="right")
    ax_t.set_title(f"{wl}: stage time per forward (ms, sum of stages alone)", loc="left", fontsize=10, color=INK)
    ax_d.bar(range(len(labels)), [per_category(l, wl)[1] for l in labels],
             color=["#8a8985", "#eb6834", "#8a8985", "#eb6834"][: len(labels)], width=0.7)
    for k, l in enumerate(labels):
        d = per_category(l, wl)[1]
        ax_d.text(k, d, str(d), ha="center", va="bottom", fontsize=8, color=INK)
    ax_d.set_xticks(range(len(labels)), labels, fontsize=8.5, color=MUTED, rotation=20, ha="right")
    ax_d.set_title(f"{wl}: dispatches per forward", loc="left", fontsize=10, color=INK)
    for ax in (ax_t, ax_d):
        ax.tick_params(labelsize=8.5, colors=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
handles, names = axes[0][0].get_legend_handles_labels()
fig.legend(handles, names, fontsize=8.5, frameon=False, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, 0.0))
fig.suptitle("EmbeddingGemma 2 text encoder, stage profile before and after the row-pass fusion (M5 Pro, shared GPU, min over 3 passes)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(root / "embedgemma2_text_fusion_profile_m5pro.png", dpi=140, facecolor="#fcfcfb")
print("wrote", root / "embedgemma2_text_fusion_profile_m5pro.png")
