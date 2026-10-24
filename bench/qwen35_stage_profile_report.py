#!/usr/bin/env python3
"""Merge the tessl and mlx stage profiles into one comparison (card ft-2bc8f1e951d0f5afff0f72303155f940).

  python3 bench/qwen35_stage_profile_report.py TESSL_LOG MLX_LOG [--json OUT.json]

Both logs are the stdout of the two lanes; only their `STAGE_PROFILE_JSON {...}`
lines are read. For each T prints, per stage, ms per call on each side, the
count per forward, and total ms per forward (ms x count), the mlx/tessl ratio of
the totals, and the share of the whole-forward gap each stage accounts for.
Stages are joined on (group, stage); a stage only one side has is shown with
the other column empty. Throwaway analysis; no dependencies beyond the stdlib.
"""
import json
import sys


def load(path):
    out = {}
    for line in open(path):
        if line.startswith("STAGE_PROFILE_JSON "):
            rec = json.loads(line[len("STAGE_PROFILE_JSON "):])
            out[rec["tokens"]] = rec
    if not out:
        raise SystemExit(f"{path}: no STAGE_PROFILE_JSON lines")
    return out


def table(tl, ml):
    t, rows = tl["tokens"], {}
    for s in tl["stages"]:
        rows[(s["group"], s["stage"])] = {"tessl": s}
    for s in ml["stages"]:
        rows.setdefault((s["group"], s["stage"]), {})["mlx"] = s
    lines = [f"=== T = {t} ===",
             f"{'group':<7}{'stage':<24}{'tessl ms':>10}{'mlx ms':>10}{'count':>6}{'tessl tot':>11}{'mlx tot':>10}{'mlx-tessl':>11}{'mlx/tessl':>10}"]
    out_rows = []
    for (g, n), d in rows.items():
        a, b = d.get("tessl"), d.get("mlx")
        cnt = (a or b)["count"]
        ta = a["ms"] * cnt if a else None
        tb = b["ms"] * cnt if b else None
        delta = tb - ta if a and b else None
        ratio = tb / ta if a and b and ta else None
        f = lambda v, w, p=3: f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
        lines.append(f"{g:<7}{n:<24}{f(a['ms'] if a else None, 10, 4)}{f(b['ms'] if b else None, 10, 4)}{cnt:>6}"
                     f"{f(ta, 11)}{f(tb, 10)}{f(delta, 11)}{f(ratio, 10, 2)}")
        out_rows.append({"group": g, "stage": n, "count": cnt, "tessl_ms": a and a["ms"], "mlx_ms": b and b["ms"],
                         "tessl_total_ms": ta, "mlx_total_ms": tb})
    comparable = [r for r in out_rows if r["group"] != "detail" and r["tessl_total_ms"] is not None
                  and r["mlx_total_ms"] is not None and r["stage"] not in ("gdn_core_prep", "gdn_core_scan")]
    st, sm = sum(r["tessl_total_ms"] for r in comparable), sum(r["mlx_total_ms"] for r in comparable)
    lines.append(f"sum of comparable stages x counts: tessl {st:.2f} ms, mlx {sm:.2f} ms, mlx-tessl {sm - st:+.2f} ms")
    tt, mt = tl["timeline_ms"]["total"], ml["forward_ms"]["total"]
    lines.append(f"whole forward (min sample, this process): tessl {tt:.2f} ms, mlx {mt:.2f} ms, mlx/tessl {mt / tt:.3f} "
                 f"(h2h min-round ratio reported separately)")
    lines.append(f"tessl timeline: begin {tl['timeline_ms']['begin']:.2f}, encode-return {tl['timeline_ms']['advance_return']:.2f}, "
                 f"score wait {tl['timeline_ms']['score_wait']:.2f}; submit floor {tl['submit_floor_ms']:.3f} ms")
    lines.append(f"mlx forward: python graph build {ml['forward_ms']['build']:.2f} (serial before the GPU), "
                 f"eval {ml['forward_ms']['eval']:.2f}, tolist {ml['forward_ms']['tolist']:.2f}")
    return lines, {"tokens": t, "rows": out_rows, "sum_tessl_ms": st, "sum_mlx_ms": sm,
                   "tessl_total_ms": tt, "mlx_total_ms": mt}


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    tess, mlx = load(argv[1]), load(argv[2])
    merged = []
    for t in sorted(set(tess) & set(mlx)):
        lines, rec = table(tess[t], mlx[t])
        print("\n".join(lines) + "\n")
        merged.append(rec)
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump({"tessl": list(tess.values()), "mlx": list(mlx.values()), "comparison": merged}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv)
