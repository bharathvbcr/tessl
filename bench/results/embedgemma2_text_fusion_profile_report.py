"""Tabulate `profile_embedgemma2` stage profiles of the text encoder, before and after
the row-pass fusion (G7). Analysis only; ships nothing.

    python3 embedgemma2_text_fusion_profile_report.py OUT_DIR LABEL=FILE.json [LABEL=FILE.json ...]

Each FILE is one `PROFILE_MODE=stages` run (`PROFILE_JSON`); a LABEL may be given for
several files (passes), and its numbers are then the minimum over all of them (the
house method: min-of-N over every kept sample of every pass), with the range of the
per-pass minima next to it. A class's time is the minimum over a pass's samples of its
`only:` variant (the stage alone, every other dispatch skipped); a category sums its
classes; the share is of the sum of all classes (the stages add up to the whole forward
to within the run-to-run noise, as `embedgemma2_stage_profile_m5pro.txt` showed). The
classes are grouped as:

  GEMM              every exact-f32 or bf16 GEMM (qkv, o, gate|up, down, the three
                    per-layer-input ones; the tail's projection is in `tail`)
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
from collections import defaultdict
from pathlib import Path

out_dir = Path(sys.argv[1])
passes = defaultdict(list)  # label -> [doc, ...] in argument order
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    passes[label].append(json.loads(Path(path).read_text()))
labels = list(passes)

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


def stages(label, wl):
    """{class: ([per-pass min ms], dispatches)} for one label and workload, plus the
    per-pass `full` minima and the workload record of the first pass."""
    per = defaultdict(list)
    disp = {}
    full = []
    first = None
    for doc in passes[label]:
        w = next((x for x in doc["workloads"] if x["workload"] == wl), None)
        if w is None:
            continue
        first = first or w
        for v in w["variants"]:
            if v["kind"] == "only":
                cls = v["name"].split(":", 1)[1]
                per[cls].append(min(v["ms_per_forward"]))
                disp[cls] = v["dispatches"]
            elif v["kind"] == "full":
                full.append(min(v["ms_per_forward"]))
    return per, disp, full, first


lines = []
P = lines.append
workloads = sorted(
    {w["workload"] for docs in passes.values() for d in docs for w in d["workloads"]},
    key=lambda s: (int(s.split("x")[0]) * int(s.split("x")[1]), s),
)
summary = {}
for wl in workloads:
    for label in labels:
        per, disp, full, first = stages(label, wl)
        if not per:
            continue
        best = {c: min(v) for c, v in per.items()}
        total = sum(best.values())
        cat_ms = {c: 0.0 for c, _ in CATEGORIES}
        cat_disp = {c: 0 for c, _ in CATEGORIES}
        for cls, ms in best.items():
            cat_ms[category(cls)] += ms
            cat_disp[category(cls)] += disp[cls]
        # The same sum per pass, to show how far the passes disagree.
        n_pass = len(passes[label])
        pass_sums = [
            sum(v[i] for v in per.values() if len(v) > i)
            for i in range(max(len(v) for v in per.values()))
        ]
        summary[(label, wl)] = dict(
            best=best, disp=disp, total=total, cat_ms=cat_ms, cat_disp=cat_disp, full=full,
            pass_sums=pass_sums, dispatches=first["dispatches_full"], n_pass=n_pass,
        )

for wl in workloads:
    P(f"== workload {wl}   (ms per forward; stage alone, min over {max(s['n_pass'] for (l, w), s in summary.items() if w == wl)} pass(es))")
    P(f"  {'run':<30} {'disp':>5} {'full ms':>9} {'sum(only)':>10} " + "".join(f"{c:>19}" for c, _ in CATEGORIES))
    for label in labels:
        s = summary.get((label, wl))
        if s is None:
            continue
        cells = "".join(f"{s['cat_ms'][c]:>10.3f} {100 * s['cat_ms'][c] / s['total']:>5.1f}% " for c, _ in CATEGORIES)
        P(f"  {label:<30} {s['dispatches']:>5} {min(s['full']):>9.3f} {s['total']:>10.3f} {cells}")
    P("")

P("Row passes (the classes the fusion changed): dispatches, ms (min over passes) and [range of the per-pass sums], share of the sum of stages")
P(f"  {'workload':<9} {'run':<30} {'disp':>5} {'ms':>9} {'passes':>16} {'share':>7}   classes (ms)")
for wl in workloads:
    for label in labels:
        s = summary.get((label, wl))
        if s is None:
            continue
        rp = [c for c in s["best"] if category(c) == "row passes"]
        passes_rp = []
        per, _, _, _ = stages(label, wl)
        for i in range(s["n_pass"]):
            passes_rp.append(sum(per[c][i] for c in rp if len(per[c]) > i))
        P(
            f"  {wl:<9} {label:<30} {s['cat_disp']['row passes']:>5} {s['cat_ms']['row passes']:>9.3f} "
            f"{min(passes_rp):>7.2f}-{max(passes_rp):<7.2f} {100 * s['cat_ms']['row passes'] / s['total']:>6.1f}%   "
            + "; ".join(f"{c.split(' (')[0]} {s['best'][c]:.3f}" for c in rp)
        )
P("")
P("Machine: GPU utilization at the start and end of each workload (ioreg, %, first pass), and samples taken with another tessl GPU process seen")
for wl in workloads:
    for label in labels:
        s = summary.get((label, wl))
        if s is None:
            continue
        _, _, _, first = stages(label, wl)
        P(
            f"  {wl:<9} {label:<30} util {first['gpu_util_start_pct']}% -> {first['gpu_util_end_pct']}%   "
            f"contended samples {first['samples_contended']} of {first['samples_total']}"
        )

text = "\n".join(lines)
print(text)
(out_dir / "embedgemma2_text_fusion_profile_tables.txt").write_text(text + "\n")
merged = {"runs": {l: passes[l][-1] for l in labels}, "passes": {l: passes[l] for l in labels}, "categories": [c for c, _ in CATEGORIES]}
(out_dir / "embedgemma2_text_fusion_profile_m5pro.json").write_text(json.dumps(merged, separators=(",", ":")))
print("wrote", out_dir / "embedgemma2_text_fusion_profile_m5pro.json", file=sys.stderr)
