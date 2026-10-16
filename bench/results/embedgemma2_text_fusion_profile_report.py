"""Tabulate `profile_embedgemma2` stage profiles of the text encoder, before and after
the row-pass fusion (G7). Analysis only; ships nothing.

    python3 embedgemma2_text_fusion_profile_report.py OUT_DIR LABEL=FILE.json [LABEL=FILE.json ...]

Each FILE is one `PROFILE_MODE=stages` run (`PROFILE_JSON`). A class's time is the
minimum over the kept samples of its `only:` variant (the stage alone, every other
dispatch skipped); a category sums its classes; the share is of the sum of all
classes (the stages add up to the whole forward to within the run-to-run noise,
as `embedgemma2_stage_profile_m5pro.txt` showed). The classes are grouped as:

  GEMM              every exact-f32 or bf16 GEMM (qkv, o, gate|up, down, the three
                    per-layer-input ones, and the tail's projection is in `tail`)
  attention         sliding and global
  row passes        every norm and residual add, the per-layer inputs' scale+norm
                    and (bf16) the casts: the classes the fusion changed
  gelu*up           the two gated activations
  other elementwise gather+copy and qk-norm+rope+kv store
  tail              final norm (unfused), pooling, projection, l2 normalize

Prints the tables; writes OUT_DIR/embedgemma2_text_fusion_profile_m5pro.json (every
input file merged under its label) and the tables to
OUT_DIR/embedgemma2_text_fusion_profile_tables.txt.
"""

import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
runs = {}
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    runs[label] = json.loads(Path(path).read_text())

CATEGORIES = [
    ("GEMM", ["qkv GEMM", "o GEMM", "gate|up GEMM", "down GEMM", "PLE projection", "ple_gate", "ple_out"]),
    ("attention", ["attention"]),
    # `rms_norm (input, pre-ffn)` / `norm+residual` are the names before the
    # fusion lane renamed the classes; both spellings are the same stages.
    ("row passes", ["rms_norm", "residual add", "norm+residual", "PLE scale", "bf16 casts"]),
    ("gelu*up", ["gelu*up", "ple gelu"]),
    ("other elementwise", ["gather", "qk-norm"]),
    ("tail", ["tail"]),
]


def category(cls):
    for name, prefixes in CATEGORIES:
        if any(cls.startswith(p) for p in prefixes):
            return name
    raise KeyError(cls)


def workload_key(w):
    return (w["rows"], w["seq"])


lines = []
P = lines.append
summary = {}
labels = list(runs)
workloads = sorted({w["workload"] for d in runs.values() for w in d["workloads"]}, key=lambda s: (int(s.split("x")[0]) * int(s.split("x")[1]), s))

for wl in workloads:
    P(f"== workload {wl}")
    hdr = f"  {'run':<18} {'disp':>5} {'full ms':>9} {'sum(only)':>10} " + "".join(f"{c:>19}" for c, _ in CATEGORIES)
    P(hdr)
    for label in labels:
        w = next((x for x in runs[label]["workloads"] if x["workload"] == wl), None)
        if w is None:
            continue
        only = {}
        for v in w["variants"]:
            if v["kind"] == "only":
                only[v["name"].split(":", 1)[1]] = (min(v["ms_per_forward"]), v["dispatches"])
        full = next(v for v in w["variants"] if v["kind"] == "full")
        full_ms = min(full["ms_per_forward"])
        total = sum(ms for ms, _ in only.values())
        cat_ms = {c: 0.0 for c, _ in CATEGORIES}
        cat_disp = {c: 0 for c, _ in CATEGORIES}
        for cls, (ms, d) in only.items():
            cat_ms[category(cls)] += ms
            cat_disp[category(cls)] += d
        summary[(label, wl)] = dict(
            full_ms=full_ms, sum_only=total, cat_ms=cat_ms, cat_disp=cat_disp, dispatches=w["dispatches_full"],
            util=(w["gpu_util_start_pct"], w["gpu_util_end_pct"]), contended=(w["samples_contended"], w["samples_total"]),
        )
        cells = "".join(f"{cat_ms[c]:>10.3f} {100 * cat_ms[c] / total:>5.1f}% " for c, _ in CATEGORIES)
        P(f"  {label:<18} {w['dispatches_full']:>5} {full_ms:>9.3f} {total:>10.3f} {cells}")
    P("")

P("Row passes (the classes the fusion changed): ms, dispatches, share of the sum of stages")
P(f"  {'workload':<9} {'run':<18} {'disp':>5} {'ms':>9} {'share':>7}   classes (ms)")
for wl in workloads:
    for label in labels:
        s = summary.get((label, wl))
        if s is None:
            continue
        w = next(x for x in runs[label]["workloads"] if x["workload"] == wl)
        parts = []
        for v in w["variants"]:
            if v["kind"] == "only":
                cls = v["name"].split(":", 1)[1]
                if category(cls) == "row passes":
                    parts.append(f"{cls.split(' (')[0]} {min(v['ms_per_forward']):.3f}")
        P(
            f"  {wl:<9} {label:<18} {s['cat_disp']['row passes']:>5} {s['cat_ms']['row passes']:>9.3f} "
            f"{100 * s['cat_ms']['row passes'] / s['sum_only']:>6.1f}%   " + "; ".join(parts)
        )
P("")
P("Machine: GPU utilization at the start and end of each workload (ioreg, %), and samples taken with another tessl GPU process seen")
for wl in workloads:
    for label in labels:
        s = summary.get((label, wl))
        if s:
            P(f"  {wl:<9} {label:<18} util {s['util'][0]}% -> {s['util'][1]}%   contended samples {s['contended'][0]} of {s['contended'][1]}")

text = "\n".join(lines)
print(text)
(out_dir / "embedgemma2_text_fusion_profile_tables.txt").write_text(text + "\n")
(out_dir / "embedgemma2_text_fusion_profile_m5pro.json").write_text(
    json.dumps({"runs": runs, "categories": [c for c, _ in CATEGORIES]}, separators=(",", ":"))
)
print("wrote", out_dir / "embedgemma2_text_fusion_profile_m5pro.json", file=sys.stderr)
