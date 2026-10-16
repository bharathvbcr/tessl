"""Merge and tabulate the output of `profile_embedgemma2_vision` (analysis only; ships nothing).

    python embedgemma2_vision_profile_report.py OUT_DIR LABEL=PASS.json [LABEL=PASS.json ...]

A LABEL names a build and operand mode (`before-f32`, `after-f32`, `after-bf16`)
and may repeat: every pass with that label is merged. Writes
OUT_DIR/embedgemma2_vision_profile_m5pro.json (every pass's raw document under
its label) and prints the ranking tables.

A stage's time is the minimum over every kept sample of every pass with the
label (the harness keeps the samples taken while no other tessl GPU process
ran, or all of them when none was clean); the table also shows the range of the
per-pass minima, the honest error bar on a GPU other sessions share, and how
many samples the harness flagged as taken beside another process. `only` is a
stage run alone with the real shapes and weights of all 16 layers; the share
column divides by the sum of the stages' `only` times (what the forward costs
if its stages add up) and `full` is the whole forward's own minimum, so the gap
between them is the cost the stages do not show alone (overlap, caches).
`full - skip` is reported by the harness and kept in the JSON; it is not
printed, because on a shared GPU its difference of two noisy minima is not
trustworthy (the raw samples are there to check).
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

out_dir = Path(sys.argv[1])
passes = OrderedDict()
for arg in sys.argv[2:]:
    label, _, path = arg.partition("=")
    passes.setdefault(label, []).append(json.loads(Path(path).read_text()))

(out_dir / "embedgemma2_vision_profile_m5pro.json").write_text(
    json.dumps({"runs": {k: v for k, v in passes.items()}}, separators=(",", ":")) + "\n"
)


def variant(doc_wl, kind, cls=None):
    for v in doc_wl["variants"]:
        if v["kind"] == kind and (cls is None or v["class"] == cls):
            return v
    raise KeyError((kind, cls))


def workloads_of(docs):
    names = []
    for d in docs:
        for w in d["workloads"]:
            if w["workload"] not in names:
                names.append(w["workload"])
    return names


for label, docs in passes.items():
    for name in workloads_of(docs):
        wls = [w for d in docs for w in d["workloads"] if w["workload"] == name]
        first = wls[0]
        classes = next(d["classes"] for d in docs)
        full_min = [min(variant(w, "full")["ms_per_call"]) for w in wls]
        enc_min = [min(w["encode_images_ms"]) for w in wls]
        flagged = sum(w["samples_contended"] for w in wls)
        total = sum(w["samples_total"] for w in wls)
        print(
            f"== {label}, workload {name}: {first['forwards']} forward(s), {first['padded_rows']} padded rows, "
            f"{len(wls)} pass(es), operands {first['operands']}"
        )
        print(
            f"   full replica {min(full_min):.2f} ms (min; per-pass minima {min(full_min):.1f}..{max(full_min):.1f}); "
            f"encode_images {min(enc_min):.2f} ms (per-pass {min(enc_min):.1f}..{max(enc_min):.1f}); "
            f"host io build {min(w['host_io_ms'] for w in wls):.2f} ms" if "host_io_ms" in first else
            f"   full replica {min(full_min):.2f} ms (min; per-pass minima {min(full_min):.1f}..{max(full_min):.1f}); "
            f"encode_images {min(enc_min):.2f} ms (per-pass {min(enc_min):.1f}..{max(enc_min):.1f})"
        )
        print(f"   samples beside another tessl GPU process: {flagged} of {total}; gpu util at start/end {first['gpu_util_start_pct']}/{first['gpu_util_end_pct']} %")
        rows = []
        for ci, cname in enumerate(classes):
            vs = [variant(w, "only", ci) for w in wls]
            if vs[0]["dispatches"] == 0:
                continue
            per_pass = [min(v["ms_per_call"]) for v in vs]
            cost = first["costs"][ci]
            only = min(per_pass)
            rows.append((cname, vs[0]["dispatches"], only, min(per_pass), max(per_pass), cost))
        total_only = sum(r[2] for r in rows)
        print(f"   {'stage':<38} {'disp':>5} {'only ms':>9} {'share':>6} {'pass range':>15} {'GB/s':>6} {'TFLOP/s':>8}")
        for cname, disp, only, lo, hi, cost in sorted(rows, key=lambda r: -r[2]):
            gbs = cost["bytes_min"] / (only * 1e-3) / 1e9
            tf = cost["flops"] / (only * 1e-3) / 1e12
            print(
                f"   {cname:<38} {disp:>5} {only:>9.3f} {100 * only / total_only:>5.1f}% {f'{lo:.2f}..{hi:.2f}':>15} "
                f"{gbs:>6.0f} {(f'{tf:.2f}' if cost['flops'] else '-'):>8}"
            )
        print(f"   sum(only) {total_only:.2f} ms = {100 * total_only / min(full_min):.1f}% of the full replica's minimum")
        print()
