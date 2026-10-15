"""Merge and tabulate the output of `profile_embedgemma2` (analysis only; ships nothing).

    python embedgemma2_stage_profile_report.py OUT_DIR PASS.json [PASS.json ...] \
        --gemm G.json --memory M.json --layout L.json --ledger LG.json --alloc A.json --ple P.json

Writes OUT_DIR/embedgemma2_stage_profile_m5pro.json (every raw sample of every
pass, and the other modes' files) and prints the ranking tables. A stage's time
is the minimum over every kept sample of every pass; the table also shows the
spread of the per-pass minima, which is the honest error bar on this machine.
"""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("passes", nargs="+")
ap.add_argument("--gemm")
ap.add_argument("--memory")
ap.add_argument("--layout")
ap.add_argument("--ledger")
ap.add_argument("--alloc")
ap.add_argument("--ple")
ap.add_argument("--contended", nargs="*", default=[], help="pass files taken while the machine was busy")
args = ap.parse_args()

out_dir = Path(args.out_dir)
passes = {Path(p).stem: json.loads(Path(p).read_text()) for p in args.passes}

# Class groups for the ranking (prefixes of the class names).
GROUPS = [
    ("gate|up GEMM", ["gate|up GEMM"]),
    ("attention sliding (hd256)", ["attention sliding"]),
    ("attention global (hd512)", ["attention global"]),
    ("down GEMM", ["down GEMM"]),
    ("qkv GEMMs (sliding + global)", ["qkv GEMM"]),
    ("o GEMMs (sliding + global)", ["o GEMM"]),
    ("per-layer-input GEMMs (proj, ple_gate, ple_out)", ["PLE projection", "ple_gate", "ple_out"]),
    ("gelu*up (mlp)", ["gelu*up"]),
    ("qk-norm + rope + kv store", ["qk-norm"]),
    ("rms_norm x48 + norm-residual x72", ["rms_norm", "norm+residual"]),
    ("per-layer-input scale+norm, gelu*input", ["PLE scale", "ple gelu"]),
    ("gather, final norm, pool, projection, l2", ["gather", "tail"]),
]


def group_of(cls):
    for g, keys in GROUPS:
        if any(cls.startswith(k) for k in keys):
            return g
    raise KeyError(cls)


def var_min(w, kind, cls=None):
    """Minimum ms over the kept samples of one variant, and per-pass minima."""
    per_pass = {}
    for pname, doc in passes.items():
        if pname in args.contended:
            continue
        for ww in doc["workloads"]:
            if ww["workload"] != w:
                continue
            for v in ww["variants"]:
                if v["kind"] == kind and (cls is None or v["name"].split(":", 1)[-1] == cls):
                    per_pass[pname] = min(v["ms_per_forward"])
    return per_pass


workloads = sorted({ww["workload"] for d in passes.values() for ww in d["workloads"]}, key=lambda s: (int(s.split("x")[0]) * int(s.split("x")[1]), int(s.split("x")[1])))
classes = next(iter(passes.values()))["classes"]
lines = []
P = lines.append

summary = {}
for w in workloads:
    full = var_min(w, "full")
    full_best = min(full.values())
    enc = [min(ww["encode_ms"]) for pn, d in passes.items() if pn not in args.contended for ww in d["workloads"] if ww["workload"] == w]
    only = {c: var_min(w, "only", c) for c in classes}
    sum_only_best = sum(min(v.values()) for v in only.values())
    summary[w] = dict(full=full, full_best=full_best, encode_best=min(enc), only=only, sum_only=sum_only_best)

# 1. Whole forward, replica vs library, sum of stages.
P("Whole forward: replica (async, forwards queued) against EmbedGemma2Model::encode, and the sum of its stages")
P(f"  {'workload':<9} {'rows':>6} {'replica ms':>11} {'per-pass min':>22} {'encode() ms':>12} {'sum(stages) ms':>15} {'sum/replica':>12}")
for w in workloads:
    s = summary[w]
    rows = int(w.split("x")[0]) * int(w.split("x")[1])
    pp = " ".join(f"{v:.1f}" for v in s["full"].values())
    P(f"  {w:<9} {rows:>6} {s['full_best']:>11.2f} {pp:>22} {s['encode_best']:>12.2f} {s['sum_only']:>15.2f} {s['sum_only'] / s['full_best']:>11.1%}")
P("")

# 2. Ranking by group.
P("Ranking by stage group: ms per forward (best over passes) and % of the replica's forward")
hdr = f"  {'group':<50}" + "".join(f"{w:>17}" for w in workloads)
P(hdr)
rank_rows = []
for g, _ in GROUPS:
    cells = {}
    for w in workloads:
        s = summary[w]
        ms = sum(min(s["only"][c].values()) for c in classes if group_of(c) == g)
        cells[w] = (ms, 100.0 * ms / s["full_best"])
    rank_rows.append((g, cells))
rank_rows.sort(key=lambda r: -r[1][workloads[-2]][0])
for g, cells in rank_rows:
    P(f"  {g:<50}" + "".join(f"{cells[w][0]:>10.2f} {cells[w][1]:>5.1f}%" for w in workloads))
P("")
P("Top 5 groups at each workload")
for w in workloads:
    top = sorted(rank_rows, key=lambda r: -r[1][w][0])[:5]
    P(f"  {w:<8} " + "; ".join(f"{g} {c[w][1]:.1f}%" for g, c in top))
P("")

# 3. Per class tables with rates.
for w in workloads:
    s = summary[w]
    doc = next(iter(passes.values()))
    ww = next(x for d in passes.values() for x in d["workloads"] if x["workload"] == w)
    costs = {c["class"]: c for c in ww["costs"]}
    disp = {v["name"].split(":", 1)[1]: v["dispatches"] for v in ww["variants"] if v["kind"] == "only"}
    P(f"Per class, {w} ({ww['rows']} rows, {ww['dispatches_full']} dispatches per forward); only = that class alone, best over passes [lowest, highest per-pass minimum]")
    P(f"  {'class':<34} {'disp':>4} {'only ms':>9} {'% of fwd':>8} {'pass range':>16} {'us/disp':>8} {'GB/s':>6} {'TFLOP/s':>8}")
    order = sorted(classes, key=lambda c: -min(s["only"][c].values()))
    for c in order:
        vals = list(s["only"][c].values())
        ms = min(vals)
        co = costs[c]
        gbs = co["bytes_min"] / (ms * 1e-3) / 1e9
        tf = co["flops"] / (ms * 1e-3) / 1e12
        P(
            f"  {c:<34} {disp[c]:>4} {ms:>9.3f} {100 * ms / s['full_best']:>7.1f}% {min(vals):>8.3f}-{max(vals):<7.3f} {1e3 * ms / max(disp[c], 1):>8.1f} {gbs:>6.0f} {tf if tf > 0 else float('nan'):>8.2f}"
        )
    P("")

# 4. Ablation cross-check where a pass has skip variants.
for pname, doc in passes.items():
    if pname in args.contended:
        continue
    for ww in doc["workloads"]:
        if not any(v["kind"] == "skip" for v in ww["variants"]):
            continue
        full = min(next(v for v in ww["variants"] if v["kind"] == "full")["ms_per_forward"])
        P(f"Ablation cross-check, pass {pname}, {ww['workload']}: full {full:.1f} ms; for the six largest stages, full - skip against the stage alone")
        tops = sorted((v for v in ww["variants"] if v["kind"] == "only"), key=lambda v: -min(v["ms_per_forward"]))[:6]
        for v in tops:
            c = v["name"].split(":", 1)[1]
            sk = next(x for x in ww["variants"] if x["kind"] == "skip" and x["name"] == "skip:" + c)
            P(f"    {c:<34} alone {min(v['ms_per_forward']):>8.2f}   full-skip {full - min(sk['ms_per_forward']):>8.2f}")
        tot = sum(full - min(x["ms_per_forward"]) for x in ww["variants"] if x["kind"] == "skip")
        P(f"    sum over all 21 stages of full-skip: {tot:.1f} ms = {tot / full:.0%} of full")
P("")

text = "\n".join(lines)
print(text)
(out_dir / "embedgemma2_stage_profile_tables.txt").write_text(text + "\n")

merged = {"passes": passes, "contended_passes": args.contended}
for key in ("gemm", "memory", "layout", "ledger", "alloc", "ple"):
    f = getattr(args, key)
    if f:
        merged[key] = json.loads(Path(f).read_text())
(out_dir / "embedgemma2_stage_profile_m5pro.json").write_text(json.dumps(merged, separators=(",", ":")))
print("wrote", out_dir / "embedgemma2_stage_profile_m5pro.json", file=sys.stderr)
