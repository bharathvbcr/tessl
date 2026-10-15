"""Plots for bench/results/embedgemma2_stage_profile_m5pro.txt (analysis only; ships nothing).

Reads embedgemma2_stage_profile_m5pro.json (written by
embedgemma2_stage_profile_report.py) and writes embedgemma2_stage_profile_m5pro.png.

    python embedgemma2_stage_profile_plot.py <results dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
doc = json.loads((root / "embedgemma2_stage_profile_m5pro.json").read_text())
passes = {k: v for k, v in doc["passes"].items() if k not in doc["contended_passes"]}
gemm, mem, layout = doc["gemm"], doc["memory"], doc["layout"]

GROUPS = {
    "GEMMs, exact f32": ["qkv GEMM", "o GEMM", "gate|up GEMM", "down GEMM", "ple_gate GEMM", "ple_out GEMM", "PLE projection"],
    "attention sliding (hd256)": ["attention sliding"],
    "attention global (hd512)": ["attention global"],
    "elementwise: norms, residual adds, gelu, rope, scale, gather, tail": [
        "gelu*up", "ple gelu", "rms_norm", "norm+residual", "qk-norm", "PLE scale", "gather", "tail"],
}
COLORS = ["#4c72b0", "#dd8452", "#c44e52", "#8c8c8c"]


def group_of(name):
    for g, keys in GROUPS.items():
        if any(name.startswith(k) for k in keys):
            return g
    raise KeyError(name)


def best(w, kind, cls=None):
    vals = []
    for p in passes.values():
        for ww in p["workloads"]:
            if ww["workload"] != w:
                continue
            for v in ww["variants"]:
                if v["kind"] == kind and (cls is None or v["name"].split(":", 1)[-1] == cls):
                    vals.append(min(v["ms_per_forward"]))
    return min(vals)


wls = sorted({ww["workload"] for p in passes.values() for ww in p["workloads"]}, key=lambda s: (int(s.split("x")[0]) * int(s.split("x")[1]), int(s.split("x")[1])))
classes = next(iter(passes.values()))["classes"]
share = {g: [] for g in GROUPS}
ms = {g: [] for g in GROUPS}
for w in wls:
    full = best(w, "full")
    for g in GROUPS:
        t = sum(best(w, "only", c) for c in classes if group_of(c) == g)
        ms[g].append(t)
        share[g].append(100.0 * t / full)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax = axes[0][0]
bottom = [0.0] * len(wls)
for (g, vals), col in zip(share.items(), COLORS):
    ax.bar(wls, vals, bottom=bottom, label=g, color=col)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_ylabel("% of the full forward (stage alone, summed)")
ax.set_title("Where a forward's GPU time goes (async; each stage alone, real shapes and weights)")
ax.legend(fontsize=7, loc="lower left")

ax = axes[0][1]
bottom = [0.0] * len(wls)
for (g, vals), col in zip(ms.items(), COLORS):
    ax.bar(wls, vals, bottom=bottom, color=col)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.plot(wls, [best(w, "full") for w in wls], "k_", markersize=26, mew=2, label="full forward (min)")
ax.set_yscale("log")
ax.set_ylabel("ms per forward (log)")
ax.set_title("Absolute time; black ticks are the whole replica forward")
ax.legend(fontsize=8)

ax = axes[1][0]
rows8 = [r for r in gemm["gemm_shapes"] if r["rows"] == 8192]
names = [r["gemm"] for r in rows8]
f32 = [r["flops"] / (min(r["f32_ms"]) * 1e-3) / 1e12 for r in rows8]
bf = [r["flops"] / (min(r["bf16_ms"]) * 1e-3) / 1e12 for r in rows8]
xs = range(len(names))
ax.bar([x - 0.2 for x in xs], f32, 0.4, label="exact f32 (what runs today)", color="#4c72b0")
ax.bar([x + 0.2 for x in xs], bf, 0.4, label="bf16 operands, f32 accumulate (not in the forward)", color="#55a868")
ax.axhline(gemm["ceilings"]["f32_gemm_4096_tflops"], color="#4c72b0", ls="--", lw=1, label="f32 4096^3 (measured)")
ax.axhline(gemm["ceilings"]["bf16_gemm_4096_tflops"], color="#55a868", ls="--", lw=1, label="bf16 4096^3 (measured)")
ax.set_xticks(list(xs))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("TFLOP/s")
ax.set_title("The forward's GEMM shapes at 8192 rows, a distinct weight per layer")
ax.legend(fontsize=7)

ax = axes[1][1]
wl8 = next(w for w in layout["workloads"] if w["rows"] == 8192)
runs = {r["label"] + "|" + r["result"]["workload"]: r["result"] for r in mem["runs"]}
labels = [
    "2a7cb8b\nset",
    "HEAD\nset",
    "HEAD set,\nPLE run 16 MiB",
    "HEAD,\nencode;encode",
    "bench table\n(after 1x16, 64x32)",
]
vals = [
    wl8["2a7cb8b_total_mib"],
    wl8["head_total_mib"],
    runs["ple16|8x1024"]["trace"][0]["allocated_above_model_mib"],
    runs["alone-back-to-back|8x1024"]["peak_above_model_mib"],
    runs["bench-order|32x256"]["peak_above_model_mib"],
]
ax.bar(labels, vals, color=["#8c8c8c", "#4c72b0", "#55a868", "#c44e52", "#c44e52"])
for i, v in enumerate(vals):
    ax.text(i, v + 15, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("MiB above the loaded model, 8192 rows")
ax.set_title("Device memory: one activation set, and what the bench table measured")
plt.setp(ax.get_xticklabels(), fontsize=8)

fig.tight_layout()
out = root / "embedgemma2_stage_profile_m5pro.png"
fig.savefig(out, dpi=130)
print("wrote", out)
