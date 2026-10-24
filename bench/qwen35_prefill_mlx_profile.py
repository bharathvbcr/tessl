#!/usr/bin/env python3
"""Stage profile of the mlx-lm lane of the Qwen3.5-2B prefill+score head-to-head.

Card ft-2bc8f1e951d0f5afff0f72303155f940. The MLX half of
`bench_qwen35_layers --stage-profile`: the same work definition as the h2h
(`bench/qwen35_prefill_mlx.py`: embedding gather, 24 layers over a fresh cache,
final norm, 17 answer rows), the same ids from tests/fixtures/qwen35_prefill_ids.json,
the same snapshot (bf16, not quantized), and the model's own modules.

  ~/.venvs/ml-2026-10/bin/python bench/qwen35_prefill_mlx_profile.py --lengths 1024,8192 --out mlx_profile.json

How a stage is timed (mlx 0.32 evaluates lazily, so a stage is a subgraph):
`reps` independent calls of the stage's code on fixed, already-evaluated inputs
are built, then ONE `mx.eval` over all of them, so the per-eval submit/wait
floor is paid once per `reps` calls, as the tessl lane's REPS-per-command-buffer
does. Two times are kept apart: `build_us` (Python graph construction per call,
host only, before any GPU work starts) and the eval wall (`ms`, per call, which
includes the host encode of that subgraph). Stages are taken from the module
code (mlx_lm/models/qwen3_5.py, qwen3_next.py, gated_delta.py), line for line.

Cross-checks printed with every T: the sum of the stages against the real
`DecoderLayer.__call__` timed the same way (layer rows), and against the whole
forward split into `build` (Python graph construction, serial before the GPU
starts) and `eval` (encode + GPU + wait). `--insitu` also times the real forward
with an `mx.eval` after each of the four layer parts (norm, mixer + add, post
norm, MLP + add), which pays a sync per part and is reported as its own group.

There is no mx.metal per-kernel timer. `MTL_CAPTURE_ENABLED=1` with
`mx.metal.start_capture` writes a .gputrace that only Xcode reads, so it is not
used here (NOT RUN).

Python here is a bench lane, not shipped code.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_lane as pl  # noqa: E402

STAGE_REPS = 8


def build_parser():
    ap = pl.base_parser(__doc__.split("\n\n")[0])
    ap.set_defaults(lengths=[1024, 8192])
    ap.add_argument("--out", default=None, help="write the JSON here (also printed as STAGE_PROFILE_JSON lines)")
    ap.add_argument("--reps", type=int, default=STAGE_REPS)
    ap.add_argument("--insitu", action="store_true", help="also time the real forward with an eval per layer part")
    return ap


def gpu_utilization():
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        return out.split('"Device Utilization %"=')[1].split(",")[0].split("}")[0]
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def main(argv=None):
    args = build_parser().parse_args(argv)
    iters, warmup = pl.resolve_counts(args)
    doc, ids, digests = pl.prepare(args)

    from pathlib import Path

    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk
    from mlx_lm.utils import load_model

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    snap = args.snapshot or pl.snapshot_default()
    model, config = load_model(Path(snap))
    model.eval()
    body = model.model
    table = body.embed_tokens.weight
    if table.dtype != mx.bfloat16:
        raise SystemExit(f"embedding table is {table.dtype}, not bfloat16")
    answer_rows = table[mx.array(doc["answers"], dtype=mx.int32)].astype(mx.float32)
    mx.eval(answer_rows)
    layers = body.layers
    gdn_idx = next(i for i, l in enumerate(layers) if l.is_linear)
    attn_idx = next(i for i, l in enumerate(layers) if not l.is_linear)
    n_gdn = sum(1 for l in layers if l.is_linear)
    n_attn = len(layers) - n_gdn
    backend = f"mlx-lm-{mlx_lm.__version__}-mlx-{mx.__version__}-{args.device}-bf16"
    reps = args.reps

    def timed_stage(fn):
        """(median ms per call, min ms per call, median build us per call) of `fn`, `reps` calls per eval."""
        def once():
            t0 = time.perf_counter()
            outs = [fn() for _ in range(reps)]
            t1 = time.perf_counter()
            mx.eval(outs)
            t2 = time.perf_counter()
            return (t2 - t0) * 1e3 / reps, (t1 - t0) * 1e6 / reps
        for _ in range(warmup):
            once()
        s = [once() for _ in range(iters)]
        ms = [a for a, _ in s]
        return statistics.median(ms), min(ms), statistics.median(b for _, b in s)

    results = []
    print(f"{backend}: snapshot {os.path.basename(snap.rstrip('/'))}, iters {iters}, warmup {warmup}, reps {reps}")
    print(f"GPU utilization at start: {gpu_utilization()}%  load: {os.getloadavg()}")
    for t in args.lengths:
        tok = mx.array([ids[t]], dtype=mx.int32)
        print(f"\n=== T = {t} (fnv1a32 {digests[t]}) ===  GPU utilization at start of T: {gpu_utilization()}%")
        x0 = body.embed_tokens(tok)  # [1, T, 2048] bf16
        g_layer, a_layer = layers[gdn_idx], layers[attn_idx]
        lin, at = g_layer.linear_attn, a_layer.self_attn
        B, S, _ = x0.shape

        # Inputs: every stage reads an evaluated array, as in a forward where its producer already ran.
        xn = g_layer.input_layernorm(x0)
        qkv, z, b, a = lin.in_proj_qkv(xn), lin.in_proj_z(xn), lin.in_proj_b(xn), lin.in_proj_a(xn)
        mx.eval(xn, qkv, z, b, a)
        z4 = z.reshape(B, S, lin.num_v_heads, lin.head_v_dim)
        conv_state0 = mx.zeros((B, lin.conv_kernel_size - 1, lin.conv_dim), dtype=xn.dtype)

        def g_conv():
            ci = mx.concatenate([conv_state0, qkv], axis=1)
            keep = mx.contiguous(ci[:, -(lin.conv_kernel_size - 1):, :])
            return [nn.silu(lin.conv1d(ci)), keep]

        conv_out = g_conv()[0]
        mx.eval(conv_out)

        def split_qkv(co):
            return [
                tt.reshape(B, S, h, d)
                for tt, h, d in zip(
                    mx.split(co, [lin.key_dim, 2 * lin.key_dim], -1),
                    [lin.num_k_heads, lin.num_k_heads, lin.num_v_heads],
                    [lin.head_k_dim, lin.head_k_dim, lin.head_v_dim],
                )
            ]

        q0, k0, v0 = split_qkv(conv_out)
        qn, kn = normalize_qk(q0, k0, inv_scale=lin.head_k_dim ** -0.5, eps=1e-6)
        beta0 = mx.sigmoid(b)
        gg0 = compute_g(lin.A_log, a, lin.dt_bias)
        state0 = mx.zeros((B, lin.num_v_heads, lin.head_v_dim, lin.head_k_dim), dtype=mx.float32)
        mx.eval(q0, k0, v0, qn, kn, beta0, gg0, state0)

        def g_core():
            q, k, v = split_qkv(conv_out)
            qq, kk = normalize_qk(q, k, inv_scale=lin.head_k_dim ** -0.5, eps=1e-6)
            from mlx_lm.models.gated_delta import gated_delta_update
            return list(gated_delta_update(qq, kk, v, a, b, lin.A_log, lin.dt_bias, None, None, use_kernel=True))

        out0, _ = g_core()
        mx.eval(out0)
        y0 = lin.norm(out0, z4)
        mx.eval(y0)

        # Attention layer inputs.
        xa = a_layer.input_layernorm(x0)
        qp, kp, vp = at.q_proj(xa), at.k_proj(xa), at.v_proj(xa)
        mx.eval(xa, qp, kp, vp)

        def a_qk_rope():
            from mlx_lm.models.cache import KVCache
            cache = KVCache()
            queries, gate = mx.split(qp.reshape(B, S, at.num_attention_heads, -1), 2, axis=-1)
            gate = gate.reshape(B, S, -1)
            queries = at.q_norm(queries).transpose(0, 2, 1, 3)
            keys = at.k_norm(kp.reshape(B, S, at.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
            values = vp.reshape(B, S, at.num_key_value_heads, -1).transpose(0, 2, 1, 3)
            queries = at.rope(queries, offset=cache.offset)
            keys = at.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
            return [queries, keys, values, gate]

        queries, keys, values, gate = a_qk_rope()
        mx.eval(queries, keys, values, gate)
        amask = create_attention_mask(x0, None)

        def a_sdpa():
            return scaled_dot_product_attention(queries, keys, values, cache=None, scale=at.scale, mask=amask)

        sd = a_sdpa()
        mx.eval(sd)

        def a_gate():
            o = sd.transpose(0, 2, 1, 3).reshape(B, S, -1)
            return o * mx.sigmoid(gate)

        og = a_gate()
        mx.eval(og)

        # MLP inputs (layer 0's MLP; the shapes are the same on every layer).
        mlp = g_layer.mlp
        h0 = x0 + g_layer.linear_attn.out_proj(y0.reshape(B, S, -1))
        mx.eval(h0)
        hn = g_layer.post_attention_layernorm(h0)
        gt, up = mlp.gate_proj(hn), mlp.up_proj(hn)
        mx.eval(hn, gt, up)
        mid = swiglu(gt, up)
        mx.eval(mid)

        stages = []

        def add(group, name, count, fn):
            ms, mn, build = timed_stage(fn)
            stages.append({"group": group, "stage": name, "ms": ms, "min_ms": mn, "count": count, "build_us": build})

        add("embed", "gather", 1, lambda: body.embed_tokens(tok))
        add("gdn", "input_norm", n_gdn, lambda: g_layer.input_layernorm(x0))
        add("gdn", "in_proj_gemm", n_gdn, lambda: [lin.in_proj_qkv(xn), lin.in_proj_z(xn), lin.in_proj_b(xn), lin.in_proj_a(xn)])
        add("gdn", "conv1d_silu", n_gdn, g_conv)
        add("gdn", "gdn_core", n_gdn, g_core)
        add("gdn", "gated_norm", n_gdn, lambda: lin.norm(out0, z4))
        add("gdn", "out_proj_resid", n_gdn, lambda: x0 + lin.out_proj(y0.reshape(B, S, -1)))
        add("attn", "input_norm", n_attn, lambda: a_layer.input_layernorm(x0))
        add("attn", "in_proj_gemm", n_attn, lambda: [at.q_proj(xa), at.k_proj(xa), at.v_proj(xa)])
        add("attn", "qk_norm_rope", n_attn, a_qk_rope)
        add("attn", "sdpa", n_attn, a_sdpa)
        add("attn", "out_gate", n_attn, a_gate)
        add("attn", "out_proj_resid", n_attn, lambda: x0 + at.o_proj(og))
        add("mlp", "post_norm", len(layers), lambda: g_layer.post_attention_layernorm(h0))
        add("mlp", "gate_up_gemm", len(layers), lambda: [mlp.gate_proj(hn), mlp.up_proj(hn)])
        add("mlp", "swiglu", len(layers), lambda: swiglu(gt, up))
        add("mlp", "down_proj_resid", len(layers), lambda: h0 + mlp.down_proj(mid))
        # Detail rows (not summed): the pieces of gdn_core and the GEMMs.
        add("detail", "gdn_qk_l2norm", n_gdn, lambda: normalize_qk(q0, k0, inv_scale=lin.head_k_dim ** -0.5, eps=1e-6))
        add("detail", "gdn_gate_prep", n_gdn, lambda: [mx.sigmoid(b), compute_g(lin.A_log, a, lin.dt_bias)])
        add("detail", "gdn_recurrence_kernel", n_gdn, lambda: list(gated_delta_kernel(qn, kn, v0, gg0, beta0, state0)))
        add("detail", "gdn_in_proj_qkv", n_gdn, lambda: lin.in_proj_qkv(xn))
        add("detail", "gdn_in_proj_z", n_gdn, lambda: lin.in_proj_z(xn))
        add("detail", "gdn_in_proj_ba", n_gdn, lambda: [lin.in_proj_b(xn), lin.in_proj_a(xn)])
        add("detail", "gdn_out_proj_gemm_only", n_gdn, lambda: lin.out_proj(y0.reshape(B, S, -1)))
        add("detail", "resid_add_only", len(layers) * 2, lambda: x0 + h0)
        add("detail", "mlp_gate_gemm", len(layers), lambda: mlp.gate_proj(hn))
        add("detail", "mlp_up_gemm", len(layers), lambda: mlp.up_proj(hn))

        # Whole real layers, same method (the analogue of tessl's encode_part cross-check).
        def real_layer(layer, idx):
            def fn():
                cache = model.make_cache()[idx]
                mask = create_ssm_mask(x0, cache) if layer.is_linear else create_attention_mask(x0, cache)
                return layer(x0, mask=mask, cache=cache)
            return fn

        lay_g = timed_stage(real_layer(g_layer, gdn_idx))
        lay_a = timed_stage(real_layer(a_layer, attn_idx))

        # Whole forward, split into Python graph build and eval.
        def forward_split():
            t0 = time.perf_counter()
            cache = model.make_cache()
            h = body(tok, cache)
            lg = h[:, -1].astype(mx.float32) @ answer_rows.T
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            state = [aa for c in cache for aa in c.state if aa is not None]
            t1 = time.perf_counter()
            mx.eval(lp, *state)
            t2 = time.perf_counter()
            lp[0].tolist()
            t3 = time.perf_counter()
            return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3

        for _ in range(warmup):
            forward_split()
        fs = [forward_split() for _ in range(iters)]
        best = min(fs, key=lambda r: r[3])
        med_total = statistics.median(r[3] for r in fs)

        insitu = None
        if args.insitu:
            def insitu_once():
                acc = {}
                def tick(name, t0):
                    acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
                cache = model.make_cache()
                h = body.embed_tokens(tok)
                mx.eval(h)
                fa = create_attention_mask(h, cache[body.fa_idx])
                sm = create_ssm_mask(h, cache[body.ssm_idx])
                for layer, c in zip(layers, cache):
                    kind = "gdn" if layer.is_linear else "attn"
                    mask = sm if layer.is_linear else fa
                    t0 = time.perf_counter(); n = layer.input_layernorm(h); mx.eval(n); tick(f"{kind}_input_norm", t0)
                    t0 = time.perf_counter()
                    r = (layer.linear_attn if layer.is_linear else layer.self_attn)(n, mask, c)
                    mx.eval(r); tick(f"{kind}_mixer", t0)
                    t0 = time.perf_counter(); h = h + r; mx.eval(h); tick("resid_add_1", t0)
                    t0 = time.perf_counter(); n = layer.post_attention_layernorm(h); mx.eval(n); tick("post_norm", t0)
                    t0 = time.perf_counter(); m = layer.mlp(n); mx.eval(m); tick("mlp", t0)
                    t0 = time.perf_counter(); h = h + m; mx.eval(h); tick("resid_add_2", t0)
                state = [aa for cc in cache for aa in cc.state if aa is not None]
                mx.eval(*state)
                return acc
            for _ in range(warmup):
                insitu_once()
            runs = [insitu_once() for _ in range(iters)]
            insitu = {k: statistics.median(r[k] for r in runs) for k in runs[0]}

        # Report.
        print(f"{'group':<7}{'stage':<26}{'ms/call':>10}{'count':>6}{'total ms':>10}{'build us':>10}")
        for s in stages:
            tot = s["ms"] * s["count"] if s["group"] != "detail" else float("nan")
            print(f"{s['group']:<7}{s['stage']:<26}{s['ms']:>10.4f}{s['count']:>6}{tot:>10.3f}{s['build_us']:>10.1f}")
        summed = sum(s["ms"] * s["count"] for s in stages if s["group"] != "detail")
        g_sum = sum(s["ms"] for s in stages if s["group"] == "gdn")
        a_sum = sum(s["ms"] for s in stages if s["group"] == "attn")
        m_sum = sum(s["ms"] for s in stages if s["group"] == "mlp")
        print(f"sum of stages x counts: {summed:.3f} ms")
        print(f"cross-check: real GDN layer {lay_g[0]:.4f} ms vs stages {g_sum + m_sum:.4f}; "
              f"real attention layer {lay_a[0]:.4f} vs stages {a_sum + m_sum:.4f}")
        print(f"whole forward min-total sample: python build {best[0]:.3f} ms, eval {best[1]:.3f} ms, tolist {best[2]:.3f} ms, total {best[3]:.3f} ms "
              f"(median total {med_total:.3f})")
        if insitu:
            print("in-situ (eval after each part, summed over 24 layers, median of runs): " +
                  ", ".join(f"{k} {v:.2f}" for k, v in sorted(insitu.items())) + f"; sum {sum(insitu.values()):.2f} ms")
        rec = {
            "backend": backend, "tokens": t, "iters": iters, "reps": reps, "stages": stages,
            "layers_ms": {"gdn_layer": lay_g[0], "attn_layer": lay_a[0]},
            "forward_ms": {"build": best[0], "eval": best[1], "tolist": best[2], "total": best[3], "total_median": med_total},
            "insitu_ms": insitu,
        }
        results.append(rec)
        print("STAGE_PROFILE_JSON " + json.dumps(rec))
        sys.stdout.flush()

    if args.out:
        meta = {"host": platform.platform(), "python": sys.version.split()[0], "load": os.getloadavg(),
                "gpu_util_end": gpu_utilization()}
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "runs": results}, f, indent=1)


if __name__ == "__main__":
    main()
