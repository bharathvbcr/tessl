"""A Qwen3.5 training step in transformers on MPS, with and without tessl's GDN.

One arm runs transformers as it ships on macOS (its torch fallback
``torch_chunk_gated_delta_rule``); the other swaps in tessl's kernels with
``tessl_torch.patch_transformers_qwen3_5()``. Nothing else changes: same
config, same seed, same random weights and token ids, bf16 on MPS.

A step is forward with labels (the full-vocabulary loss) plus backward, then
``torch.mps.synchronize()``; no optimizer. Each arm runs in a process of its
own, so its peak memory is its own: ``ri_lifetime_max_phys_footprint`` from
``proc_pid_rusage``, which counts Metal memory that torch's allocator does
not see (tessl's buffers). The model is built directly on MPS, so no CPU copy
of the weights inflates that peak; the footprint after building is printed
alongside. Rounds alternate the arm order (torch first in odd rounds, tessl
first in even ones), and the per-round ratio torch / tessl is reported with
its spread.

Each process checks that the GDN function it meant to run was called on every
step (a patch that silently stopped applying would otherwise time two copies
of the torch arm) and that every loss is finite. Across processes, the two
arms' first-step losses must agree within each round (``--loss-rtol``) and
each arm's must not move between rounds (``--determinism-rtol``: the seed
builds the same weights every time), and with ``--max-spread`` the per-round
ratios' max/min is bounded. A run that fails any of these still prints and
writes its numbers, marked ``"valid": false`` with the reasons, and exits 3.
``OUT.partial`` holds the finished rounds while the run is going.

    cargo build --release      # libtessl.dylib, which tessl_torch loads
    python3 tools/qwen35_ref/gdn_speed.py --tokens 2048 --rounds 6 --out OUT.json
    python3 tools/qwen35_ref/gdn_speed.py --config .../Qwen3.5-0.8B-Base/config.json

Without ``--config`` the model is ``saved_memory.cut_2b_config()``: Qwen3.5-2B
cut to 3 GDN layers and 1 attention layer.
"""

import argparse
import ctypes
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ARMS = ("torch", "tessl")


# ---------------------------------------------------------------- footprint ---

_RUSAGE_INFO_V4 = 4
_V4_FIELDS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
    "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
    "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
    "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
    "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
    "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
    "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
)


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _V4_FIELDS]


def footprint():
    """(current, lifetime max) phys_footprint of this process, in bytes."""
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    info = _RusageInfoV4()
    if libc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint


# -------------------------------------------------------------------- child ---

def child(args):
    import torch
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    mps = torch.device("mps")
    expected = mq.torch_chunk_gated_delta_rule
    if args.child == "tessl":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
        import tessl_torch

        tessl_torch.patch_transformers_qwen3_5()
        expected = tessl_torch.chunk_gated_delta_rule
    if mq.torch_chunk_gated_delta_rule is not expected:
        raise SystemExit(f"{args.child}: modeling_qwen3_5.torch_chunk_gated_delta_rule is not the function this arm runs")
    calls = [0]

    def counted(*a, **k):
        calls[0] += 1
        return expected(*a, **k)

    mq.torch_chunk_gated_delta_rule = counted

    if args.config:
        raw = json.loads(Path(args.config).read_text())
        cfg = Qwen3_5TextConfig(**raw.get("text_config", raw))
    else:
        from saved_memory import cut_2b_config

        cfg = cut_2b_config()
    gdn_layers = sum(t == "linear_attention" for t in cfg.layer_types)
    if gdn_layers == 0:
        raise SystemExit("the config has no linear_attention (GDN) layers")

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    with mps:
        model = mq.Qwen3_5ForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    model.train()
    gen = torch.Generator().manual_seed(args.seed + 1)
    ids = torch.randint(0, cfg.vocab_size, (1, args.tokens), generator=gen).to(mps)
    torch.mps.synchronize()
    after_build, _ = footprint()

    def step():
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        torch.mps.synchronize()
        loss = out.loss.item()
        if not math.isfinite(loss):
            raise SystemExit(f"{args.child}: non-finite loss {loss}")
        for p in model.parameters():
            p.grad = None
        return loss

    first_loss = None
    ms = []
    for k in range(args.warmup + args.iters):
        before = calls[0]
        t0 = time.perf_counter()
        loss = step()
        dt = (time.perf_counter() - t0) * 1e3
        if calls[0] - before < gdn_layers:
            raise SystemExit(f"{args.child}: step {k} made {calls[0] - before} GDN calls, expected >= {gdn_layers}")
        if first_loss is None:
            first_loss = loss
        if k >= args.warmup:
            ms.append(dt)
    current, peak = footprint()
    print(json.dumps({
        "arm": args.child,
        "torch": torch.__version__,
        "params": sum(p.numel() for p in model.parameters()),
        "gdn_layers": gdn_layers,
        "layers": cfg.num_hidden_layers,
        "tokens": args.tokens,
        "first_loss": first_loss,
        "min_ms": min(ms),
        "median_ms": statistics.median(ms),
        "ms": ms,
        "footprint_after_build_mib": after_build / 2**20,
        "footprint_peak_mib": peak / 2**20,
        "footprint_end_mib": current / 2**20,
        "mps_driver_end_mib": torch.mps.driver_allocated_memory() / 2**20,
    }))


# ------------------------------------------------------------------- parent ---

def run_arm(args, arm):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", arm,
           "--tokens", str(args.tokens), "--iters", str(args.iters),
           "--warmup", str(args.warmup), "--seed", str(args.seed)]
    if args.config:
        cmd += ["--config", args.config]
    load_before = os.getloadavg()[0]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        sys.stderr.write(e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        raise SystemExit(f"arm {arm} did not finish within --timeout {args.timeout:.0f} s") from None
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"arm {arm} exited with {proc.returncode}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if len(lines) != 1:
        raise SystemExit(f"arm {arm}: expected one JSON line on stdout, got {len(lines)}")
    rec = json.loads(lines[0])
    if rec.get("arm") != arm:
        raise SystemExit(f"arm {arm}: the child reported arm {rec.get('arm')!r}")
    rec["load_before"] = load_before
    rec["load_after"] = os.getloadavg()[0]
    return rec


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def problems(rounds, loss_rtol, determinism_rtol, max_spread):
    """Why these rounds are not a result, or [] when they are.

    - Within a round both arms must compute the same loss (to ``loss_rtol``):
      otherwise the timings are of different computations.
    - Each arm's first-step loss must not move between rounds (to
      ``determinism_rtol``): every process builds the same weights from the
      same seed, so a moving loss means the seed does not control the build
      and the rounds are not comparable.
    - With ``max_spread`` set, the per-round ratios' max/min must not exceed it.
    """
    out = []
    for x in rounds:
        d = rel(x["torch"]["first_loss"], x["tessl"]["first_loss"])
        if not d <= loss_rtol:
            out.append(f"round {x['round']}: first-step losses differ by {d:.2e} relative (> {loss_rtol}): "
                       f"torch {x['torch']['first_loss']}, tessl {x['tessl']['first_loss']}")
    for a in ARMS:
        first = rounds[0][a]["first_loss"]
        for x in rounds[1:]:
            d = rel(first, x[a]["first_loss"])
            if not d <= determinism_rtol:
                out.append(f"{a}: round {x['round']}'s first-step loss {x[a]['first_loss']} differs from round 1's "
                           f"{first} by {d:.2e} relative (> {determinism_rtol})")
    ratios = [x["ratio_torch_over_tessl"] for x in rounds]
    if max_spread is not None and max(ratios) / min(ratios) > max_spread:
        out.append(f"ratio spread {max(ratios) / min(ratios):.3f} exceeds --max-spread {max_spread}")
    return out


def parent(args):
    if args.config and not Path(args.config).is_file():
        raise SystemExit(f"--config {args.config}: no such file")
    partial = Path(args.out + ".partial") if args.out else None
    rounds = []
    for r in range(1, args.rounds + 1):
        order = ARMS if r % 2 else ARMS[::-1]
        recs = {arm: run_arm(args, arm) for arm in order}
        ratio = recs["torch"]["median_ms"] / recs["tessl"]["median_ms"]
        rounds.append({"round": r, "order": list(order), "ratio_torch_over_tessl": ratio, **{a: recs[a] for a in ARMS}})
        if partial:
            partial.write_text(json.dumps({"complete": False, "rounds_detail": rounds}, indent=1) + "\n")
        print(f"round {r} ({' then '.join(order)}): torch {recs['torch']['median_ms']:.1f} ms, "
              f"tessl {recs['tessl']['median_ms']:.1f} ms, ratio {ratio:.3f}; "
              f"peak {recs['torch']['footprint_peak_mib']:.0f} vs {recs['tessl']['footprint_peak_mib']:.0f} MiB; "
              f"load {recs[order[0]]['load_before']:.1f}", file=sys.stderr)

    losses = {a: rounds[0][a]["first_loss"] for a in ARMS}
    ratios = [x["ratio_torch_over_tessl"] for x in rounds]
    found = problems(rounds, args.loss_rtol, args.determinism_rtol, args.max_spread)

    def arm_summary(a):
        return {
            "median_of_round_medians_ms": statistics.median(x[a]["median_ms"] for x in rounds),
            "min_ms": min(x[a]["min_ms"] for x in rounds),
            "tokens_per_s_at_median": args.tokens * 1e3 / statistics.median(x[a]["median_ms"] for x in rounds),
            "footprint_peak_mib_max": max(x[a]["footprint_peak_mib"] for x in rounds),
            "footprint_after_build_mib": rounds[0][a]["footprint_after_build_mib"],
        }

    summary = {
        "config": args.config or "saved_memory.cut_2b_config()",
        "tokens": args.tokens,
        "rounds": args.rounds,
        "iters": args.iters,
        "warmup": args.warmup,
        "complete": True,
        "valid": not found,
        "problems": found,
        "first_loss": losses,
        "first_loss_rel_diff": rel(losses["torch"], losses["tessl"]),
        "loss_rtol": args.loss_rtol,
        "determinism_rtol": args.determinism_rtol,
        "max_spread": args.max_spread,
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "ratio_spread": max(ratios) / min(ratios),
        "load_max": max(max(x[a]["load_before"], x[a]["load_after"]) for x in rounds for a in ARMS),
        "arms": {a: arm_summary(a) for a in ARMS},
        "rounds_detail": rounds,
    }
    if args.out:
        # Written whole, then renamed: a reader never sees half a file.
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(summary, indent=1) + "\n")
        tmp.replace(args.out)
        partial.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rounds_detail"}, indent=1))
    if found:
        # Exit 3, as bench_paired does for drift: the numbers above are not a result.
        print("NOT A RESULT:\n  " + "\n  ".join(found), file=sys.stderr)
        raise SystemExit(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="a Qwen3.5 config.json (its text_config is used); default: the 2B cut")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-rtol", type=float, default=1e-2,
                    help="largest relative difference between the arms' first-step losses (bf16)")
    ap.add_argument("--determinism-rtol", type=float, default=1e-5,
                    help="largest relative change of one arm's first-step loss between rounds")
    ap.add_argument("--max-spread", type=float,
                    help="refuse (exit 3) when the per-round ratios' max/min exceeds this, e.g. 1.10")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per arm process")
    ap.add_argument("--out", help="write the full JSON summary here (OUT.partial after each round)")
    ap.add_argument("--child", choices=ARMS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.rounds < 2 or args.iters < 1 or args.warmup < 1 or args.tokens < 1:
        ap.error("need --rounds >= 2 (one of each order), --iters >= 1, --warmup >= 1, --tokens >= 1")
    for name in ("loss_rtol", "determinism_rtol", "timeout"):
        v = getattr(args, name)
        if not (v > 0 and v != float("inf")):
            ap.error(f"--{name.replace('_', '-')} must be a positive finite number, got {v}")
    if args.max_spread is not None and not (1.0 <= args.max_spread < float("inf")):
        ap.error(f"--max-spread must be at least 1, got {args.max_spread}")
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
