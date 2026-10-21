"""A transformers Qwen3.5 forward+backward on MPS, with and without
``tessl_torch.patch_transformers_qwen3_5()`` (a measurement script: it prints
``METRIC name value`` lines for ``bench_paired --format metrics`` and ships no
behaviour).

The model is the Qwen3.5-2B text config cut to ``--layers`` layers (default 4,
which keeps the 2B's three-GDN-then-attention pattern), float32, B=1, at
``--seq-len`` tokens (default 2048). The cut layers load the real checkpoint's
weights (``--init real``: the embedding, the first ``--layers`` layers and the
final norm, mapped from ``model.language_model.*`` to the text-only model's
``model.*``); ``--init random`` is a seeded init instead. Everything is seeded
by ``--seed`` so both arms build the same model on the same ids.

Timing (one process is one arm, so ``bench_paired`` interleaves the arms):

    python bench/qwen35_gdn_patch_torch.py --arm torch|tessl [--seq-len N]
        [--steps N] [--warmup N] [--layers N] [--init real|random]
        [--snapshot DIR]

``--warmup`` and ``--steps`` default to ``BENCH_WARMUP`` (2) and
``BENCH_ITERS`` (5) where set. Each step is ``zero_grad``, the forward with
``labels=ids``, ``backward`` and ``torch.mps.synchronize()``, timed from a
synchronized start. One further untimed step samples memory, after a
``torch.mps.synchronize()``, after the forward and after the backward. The
metrics on stdout, all lower-is-better:

    qwen35_gdn_step_ms_min, qwen35_gdn_step_ms_median
    qwen35_gdn_tensor_mib_peak   (larger of the two samples)
    qwen35_gdn_tensor_mib_after  (after the backward)

The memory metrics are ``torch.mps.current_allocated_memory()``, the bytes
live tensors hold, which is the same every round. The gated metrics must be
that: ``torch.mps.driver_allocated_memory()`` includes the caching
allocator's retained pool, which varied by 12% across torch rounds (16123 to
18043 MiB) at the same work, and no drift gate passes that. The driver
samples are printed on stderr only (``driver_mib_peak=... driver_mib_after=...``),
so they land in the lane logs without being gated.

``--arm tessl`` calls ``tessl_torch.patch_transformers_qwen3_5()`` before the
model is built (the layer looks the function up at call time). Build the
library first (``cargo build --release``) or point ``TESSL_LIB`` at a
``libtessl.dylib``; ``bench_paired --clean-env`` drops ``TESSL_*``, so give it
to the lane line.

Gradient parity (``--check``): one step on the same weights and ids, the
unpatched model first and then the patched one in the same process (the
previous function restored afterwards), compared with the bounds of
``python/tests/test_gdn.py``'s ``ThroughAQwen35Model``: the loss within 1e-4
relative, and the worst parameter gradient within 1e-3 of its own peak
(``max|got-want| / max|want|``). It prints the worst parameter and exits
non-zero on a miss, on a missing or extra gradient, or on a non-finite value.
Both modes also count the calls that reach tessl (a counting wrapper on
``tessl_torch.chunk_gated_delta_rule``, which the patch installs) and fail
when the patch is not taking effect: ``--check`` needs 0 calls in the
unpatched step and at least one per GDN layer in the patched one, ``--arm
tessl`` at least one per GDN layer per forward (3 for the 4-layer cut). The
count is on stderr as ``tessl_gdn_calls`` (0 for the torch arm).
Those bounds are for float32.

The commands for the locked GPU job, from the repository root with
``cargo build --release --bin bench_paired`` built (``--evidence`` wants a
clean tree and an even round count of at least 4):

    python bench/qwen35_gdn_patch_torch.py --check --seq-len 512

(``--check`` logs ``phase=<name> rss_mib=... rss_max_mib=... driver_mib=...
tensor_mib=...`` on stderr at each phase boundary, to find where host memory
goes. It holds both steps' gradients and the model, so it runs at 512:
at 1024 and 2048 memguard's 32 GiB cap killed it.)

    target/release/bench_paired --format metrics --evidence --clean-env \\
        --rounds 6 --compare tessl:torch \\
        --out bench/results/qwen35_gdn_patch_torch_m5pro.json \\
        "torch:BENCH_WARMUP=2,BENCH_ITERS=5=python bench/qwen35_gdn_patch_torch.py --arm torch --seq-len 2048" \\
        "tessl:BENCH_WARMUP=2,BENCH_ITERS=5=python bench/qwen35_gdn_patch_torch.py --arm tessl --seq-len 2048"

(the lane command is split on whitespace, so name the venv's python by its
full path there if ``python`` is not it.) A ratio is the ``tessl`` lane over
the ``torch`` lane, so above 1 tessl is slower or larger; report every
metric where it is, not only the wins.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots"
    / "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
)
# python/tests/test_gdn.py ThroughAQwen35Model: loss assertAlmostEqual(delta=
# 1e-4 * |want|) and the worst rel_err of any parameter gradient <= 1e-3.
LOSS_RTOL = 1e-4
GRAD_REL = 1e-3
# Config keys of the checkpoint's text_config that Qwen3_5TextConfig does not
# take (the multi-token-prediction head, the stored dtype, the model type).
_SKIP_CONFIG = ("model_type", "dtype", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")
_PREFIX = "model.language_model."


def env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("torch", "tessl"), help="the timed lane (required unless --check)")
    ap.add_argument("--check", action="store_true", help="gradient parity, unpatched then patched, one process")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=env_int("BENCH_ITERS", 5))
    ap.add_argument("--warmup", type=int, default=env_int("BENCH_WARMUP", 2))
    ap.add_argument("--init", choices=("real", "random"), default="real")
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.check == (args.arm is not None):
        ap.error("exactly one of --arm and --check")
    if args.seq_len < 1 or args.steps < 1 or args.warmup < 0:
        ap.error("--seq-len and --steps must be at least 1, --warmup at least 0")
    if args.layers < 1:
        ap.error("--layers must be at least 1")
    return args


def summarize(step_ms):
    """Min and median of the timed steps, in ms."""
    if not step_ms:
        raise ValueError("no timed steps")
    return min(step_ms), statistics.median(step_ms)


def metric_lines(step_ms, mem_peak_mib, mem_after_mib):
    lo, med = summarize(step_ms)
    return [
        f"METRIC qwen35_gdn_step_ms_min {lo:.3f}",
        f"METRIC qwen35_gdn_step_ms_median {med:.3f}",
        f"METRIC qwen35_gdn_tensor_mib_peak {mem_peak_mib:.1f}",
        f"METRIC qwen35_gdn_tensor_mib_after {mem_after_mib:.1f}",
    ]


REL_ERR_SLICE = 1 << 24


def rel_err(got, want, slice_elems=REL_ERR_SLICE):
    """test_gdn.py's rel_err: the largest absolute error over the want's peak,
    in double. The same value as widening whole tensors, taken a slice at a time:
    the embedding gradient is 508M elements, and four whole-tensor double
    temporaries of it were 16 GB of host memory."""
    got = got.detach().cpu().reshape(-1)
    want = want.detach().cpu().reshape(-1)
    diff = peak = 0.0
    for lo in range(0, want.numel(), slice_elems):
        g = got[lo:lo + slice_elems].double()
        w = want[lo:lo + slice_elems].double()
        diff = max(diff, (g - w).abs().max().item())
        peak = max(peak, w.abs().max().item())
    return diff / max(peak, 1e-30)


def compare_grads(got_loss, want_loss, got, want):
    """Score a patched step against an unpatched one with test_gdn.py's bounds.

    Returns ``(ok, worst_name, worst_err, reasons)``; a gradient missing from
    one side, extra on the other, or non-finite is a reason, not a skipped row.
    """
    import torch

    reasons = []
    if not (got_loss == got_loss and want_loss == want_loss) or abs(got_loss - want_loss) > LOSS_RTOL * abs(want_loss):
        reasons.append(f"loss {got_loss!r} vs {want_loss!r} outside {LOSS_RTOL:g} relative")
    for name in sorted(set(want) - set(got)):
        reasons.append(f"{name}: no gradient in the patched model")
    for name in sorted(set(got) - set(want)):
        reasons.append(f"{name}: no gradient in the unpatched model")
    worst_name, worst_err = "", 0.0
    for name in sorted(set(got) & set(want)):
        if not (torch.isfinite(got[name]).all() and torch.isfinite(want[name]).all()):
            reasons.append(f"{name}: non-finite gradient")
            continue
        err = rel_err(got[name], want[name])
        if err > worst_err:
            worst_name, worst_err = name, err
    if worst_err > GRAD_REL:
        reasons.append(f"{worst_name}: gradient error {worst_err:.3e} over {GRAD_REL:g}")
    return not reasons, worst_name, worst_err, reasons


def text_config(snapshot, layers):
    """The checkpoint's text config cut to ``layers`` layers, keeping the
    leading layer types (so 4 layers is linear, linear, linear, full)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    cfg = json.loads((Path(snapshot) / "config.json").read_text())
    text = dict(cfg.get("text_config", cfg))
    for key in _SKIP_CONFIG:
        text.pop(key, None)
    kinds = text["layer_types"]
    if layers > len(kinds):
        raise SystemExit(f"--layers {layers} is over the checkpoint's {len(kinds)}")
    text["num_hidden_layers"] = layers
    text["layer_types"] = kinds[:layers]
    return mq.Qwen3_5TextConfig(**text)


def load_real_weights(model, snapshot, layers):
    """Copy the embedding, the first ``layers`` layers and the final norm from
    the checkpoint into a text-only model (bf16 on disk, float32 here)."""
    import torch
    from safetensors import safe_open

    path = Path(snapshot) / "model.safetensors-00001-of-00001.safetensors"
    if not path.is_file():
        raise SystemExit(f"{path}: no checkpoint (use --init random or --snapshot)")
    state = model.state_dict()
    loaded = set()
    with safe_open(str(path), "pt") as f:
        for key in f.keys():
            if not key.startswith(_PREFIX):
                continue
            name = "model." + key[len(_PREFIX):]
            parts = name.split(".")
            if parts[1] == "layers" and int(parts[2]) >= layers:
                continue
            if name not in state:
                raise SystemExit(f"checkpoint tensor {key} has no parameter {name} in the model")
            # Straight into the parameter, one tensor at a time: no second
            # float32 copy of the model is held.
            with torch.no_grad():
                state[name].copy_(f.get_tensor(key))
            loaded.add(name)
    # The tied lm_head is the embedding; every other parameter must be covered.
    missing = [n for n in state if n not in loaded and n != "lm_head.weight" and "rotary" not in n]
    if missing:
        raise SystemExit(f"checkpoint lacks {len(missing)} parameters, first {missing[0]}")


def build_model(args, device):
    """A seeded float32 model on ``device`` (real weights or seeded init)."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    torch.manual_seed(args.seed)
    model = mq.Qwen3_5ForCausalLM(text_config(args.snapshot, args.layers)).float()
    if args.init == "real":
        load_real_weights(model, args.snapshot, args.layers)
    model.tie_weights()
    return model.to(device)


def make_ids(args, vocab, device):
    import torch

    gen = torch.Generator().manual_seed(args.seed + 1)
    return torch.randint(0, vocab, (1, args.seq_len), generator=gen).to(device)


def sync(device):
    import torch

    if device.type == "mps":
        torch.mps.synchronize()


def train_step(model, ids, device):
    """One forward and backward; returns the loss as a float."""
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    sync(device)
    return loss.item()


def time_steps(model, ids, device, warmup, steps):
    for _ in range(warmup):
        train_step(model, ids, device)
    out = []
    for _ in range(steps):
        sync(device)
        t0 = time.perf_counter()
        train_step(model, ids, device)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def memory_step(model, ids, device):
    """Memory after the forward and after the backward of one untimed step, in
    MiB, sampled after a synchronize: ``{"tensor": (peak, after), "driver":
    (peak, after)}``. ``tensor`` is the live tensors' bytes (gated, repeatable),
    ``driver`` the driver's allocation including the allocator's cache
    (recorded, not gated)."""
    import torch

    def sample():
        sync(device)
        mib = 1.0 / (1 << 20)
        return torch.mps.current_allocated_memory() * mib, torch.mps.driver_allocated_memory() * mib

    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    t_fwd, d_fwd = sample()
    loss.backward()
    t_after, d_after = sample()
    return {"tensor": (max(t_fwd, t_after), t_after), "driver": (max(d_fwd, d_after), d_after)}


def require_mps():
    import torch

    if not torch.backends.mps.is_available():
        sys.exit("qwen35_gdn_patch_torch: MPS is not available")
    return torch.device("mps")


class CallCounter:
    """A transparent wrapper that counts calls into ``fn``."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def spy_on(module, name="chunk_gated_delta_rule"):
    """Replace ``module.<name>`` with a counting wrapper; returns the counter
    and the original. ``patch_transformers_qwen3_5`` resolves the name in the
    module at call time, so the function it installs is the counted one, and a
    call that reaches tessl through the patch is counted. Put the original
    back with ``setattr(module, name, original)``."""
    original = getattr(module, name)
    counter = CallCounter(original)
    setattr(module, name, counter)
    return counter, original


def count_error(calls, at_least, label):
    """A message when ``calls`` into tessl is not what the run needs, else
    None. ``at_least == 0`` means the count must be exactly 0 (the unpatched
    step); otherwise at least ``at_least`` calls (one per GDN layer per
    forward) are required, or the patch is not taking effect."""
    if at_least == 0:
        return f"{label}: {calls} tessl GDN calls, expected 0" if calls else None
    if calls < at_least:
        return (f"{label}: {calls} tessl GDN calls, expected at least {at_least} "
                "(one per GDN layer per forward): the patch is not taking effect")
    return None


def gdn_layers(model):
    return list(model.config.layer_types).count("linear_attention")


def run_arm(args):
    device = require_mps()
    counter = None
    if args.arm == "tessl":
        sys.path.insert(0, str(ROOT / "python"))
        import tessl_torch

        counter, _ = spy_on(tessl_torch)
        tessl_torch.patch_transformers_qwen3_5()
    model = build_model(args, device)
    ids = make_ids(args, model.config.vocab_size, device)
    step_ms = time_steps(model, ids, device, args.warmup, args.steps)
    mem = memory_step(model, ids, device)
    peak, after = mem["tensor"]
    calls = counter.calls if counter else 0
    print(f"arm={args.arm} layers={args.layers} seq_len={args.seq_len} init={args.init} tessl_gdn_calls={calls}",
          file=sys.stderr)
    print(f"driver_mib_peak={mem['driver'][0]:.1f} driver_mib_after={mem['driver'][1]:.1f}", file=sys.stderr)
    if counter:
        forwards = args.warmup + args.steps + 1
        err = count_error(calls, gdn_layers(model) * forwards, "arm tessl")
        if err:
            print(err, file=sys.stderr)
            return 1
    print("\n".join(metric_lines(step_ms, peak, after)))
    return 0


def host_copy(t):
    """A host copy that owns its storage (``.cpu()`` of a device tensor is one
    copy already; of a CPU tensor it is the tensor itself)."""
    t = t.detach()
    return t.cpu() if t.device.type != "cpu" else t.clone()


def grads_of(model):
    return {n: host_copy(p.grad) for n, p in model.named_parameters() if p.grad is not None}


def rss_now_mib():
    """This process's current resident set (ps; the stdlib has only the max)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip()) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return float("nan")


def phase_log(name):
    """One stderr line of memory at a phase boundary of ``--check``: the max and
    current resident set, and, where MPS is up, the driver's and the live
    tensors' allocation (all MiB)."""
    import resource

    import torch

    mib = 1.0 / (1 << 20)
    # ru_maxrss is bytes on macOS and KiB elsewhere.
    rss_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (mib if sys.platform == "darwin" else 1 / 1024.0)
    line = f"phase={name} rss_mib={rss_now_mib():.0f} rss_max_mib={rss_max:.0f}"
    if torch.backends.mps.is_available():
        line += (f" driver_mib={torch.mps.driver_allocated_memory() * mib:.0f}"
                 f" tensor_mib={torch.mps.current_allocated_memory() * mib:.0f}")
    print(line, file=sys.stderr, flush=True)


def run_check(args):
    device = require_mps()
    sys.path.insert(0, str(ROOT / "python"))
    import tessl_torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    reference = mq.torch_chunk_gated_delta_rule
    counter, tessl_fn = spy_on(tessl_torch)
    phase_log("start")
    model = build_model(args, device)
    ids = make_ids(args, model.config.vocab_size, device)
    phase_log("model_built")
    want_loss = train_step(model, ids, device)
    phase_log("unpatched_step")
    want = grads_of(model)
    phase_log("unpatched_grads_on_host")
    unpatched_calls = counter.calls
    previous = tessl_torch.patch_transformers_qwen3_5()
    phase_log("patched")
    try:
        if previous is not reference:
            sys.exit("check: the unpatched function was not transformers' torch fallback")
        got_loss = train_step(model, ids, device)
        phase_log("patched_step")
        got = grads_of(model)
        phase_log("patched_grads_on_host")
    finally:
        mq.torch_chunk_gated_delta_rule = previous
        tessl_torch.chunk_gated_delta_rule = tessl_fn
    patched_calls = counter.calls - unpatched_calls
    print(f"tessl GDN calls: unpatched step {unpatched_calls}, patched step {patched_calls}")
    miss = [e for e in (count_error(unpatched_calls, 0, "unpatched step"),
                        count_error(patched_calls, gdn_layers(model), "patched step")) if e]
    ok, name, err, reasons = compare_grads(got_loss, want_loss, got, want)
    phase_log("compared")
    reasons = miss + reasons
    ok = not reasons
    print(f"loss unpatched {want_loss:.6f} patched {got_loss:.6f}")
    print(f"worst parameter gradient error {err:.3e} ({name}) over {len(want)} parameters, bound {GRAD_REL:g}")
    for r in reasons:
        print(f"MISS {r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.check:
        return run_check(args)
    return run_arm(args)


if __name__ == "__main__":
    sys.exit(main())
