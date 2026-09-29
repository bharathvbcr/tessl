#!/usr/bin/env python3
"""Check tessl's Qwen3.5 kernels, run on the CPU emulator, against transformers.

    python3 tools/msl_emu/check_qwen35.py [-k NAME_SUBSTRING]

Needs `torch` and `transformers` (any version that ships
`transformers.models.qwen3_5`). The oracle is the model code itself —
`torch_chunk_gated_delta_rule`, `causal_conv1d_update`, `Qwen3_5RMSNormGated`,
`Qwen3_5TextRotaryEmbedding`, `apply_rotary_pos_emb` — plus an independent f64
sequential recurrence for the delta rule, because transformers computes it in
fp32 and a reference must not be a meaningful source of the error it measures.

Every GDN case reports two numbers: the kernel's error against f64 and
transformers' own fp32 error against f64. The kernel is held to a bound that
scales with the second, so the check says "as accurate as the reference
implementation", not "close to a number we picked".
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DK = 128
FAILURES = []
# --fast-math: perturb the kernels' fast-math transcendentals by a few ulps and
# hold the GDN to tests/qwen35_kernels.rs's bound (1e-4 of max|y|) instead of
# to transformers' own fp32 error, which no perturbed kernel can match.
FAST_MATH = False


def build():
    out = os.environ.get("MSL_EMU_OUT", os.path.join(HERE, "build"))
    env = dict(os.environ, MSL_EMU_OUT=out)
    subprocess.run([os.path.join(HERE, "build.sh")], check=True, env=env, stdout=subprocess.DEVNULL)
    return os.path.join(out, "harness")


HARNESS = None


# Threadgroup orders every case runs in. A GPU promises none, so a correct
# kernel's output is bitwise identical in all of them; one whose threadgroups
# write each other's outputs is not, even when a single order happens to pass.
ORDERS = ["forward", "reverse", "shuffle"]


def run(kernel, params, inputs, outputs):
    """Write inputs, run the harness in each threadgroup order, and return the
    outputs as flat tensors after checking every order agrees bit for bit."""
    results = []
    for order in ORDERS:
        results.append(_run_once(kernel, params, inputs, outputs, order))
    first = results[0]
    for order, res in zip(ORDERS[1:], results[1:]):
        for name in outputs:
            a, b = first[name], res[name]
            same = torch.equal(a.view(torch.int32), b.view(torch.int32)) if a.dtype == torch.float32 \
                else torch.equal(a, b)
            if not same:
                FAILURES.append(f"{kernel}: output {name} depends on threadgroup order ({order})")
                print(f"  [FAIL] {kernel}: output {name} differs between forward and {order} threadgroup order")
    return first


def _run_once(kernel, params, inputs, outputs, order):
    with tempfile.TemporaryDirectory() as d:
        for name, t in inputs.items():
            t = t.contiguous()
            if t.dtype == torch.float64:
                t = t.float()
            with open(os.path.join(d, name + ".bin"), "wb") as f:
                f.write(t.numpy().tobytes())
        for name, (dtype, n) in outputs.items():
            if name in inputs:
                continue  # in-place: the input is the initial contents
            # Pre-fill outputs with NaN so an element the kernel never wrote is
            # visible as such rather than as a plausible zero.
            fill = torch.full((n,), float("nan"), dtype=torch.float32)
            data = fill.to(torch.bfloat16).view(torch.int16) if dtype == "bf16" else fill
            with open(os.path.join(d, name + ".bin"), "wb") as f:
                f.write(data.numpy().tobytes())
        with open(os.path.join(d, "params.txt"), "w") as f:
            f.write(f"kernel {kernel}\n")
            f.write("outputs " + " ".join(outputs) + "\n")
            for k, v in params.items():
                f.write(f"{k} {v}\n")
        subprocess.run([HARNESS, d], check=True, env=dict(os.environ, MSL_EMU_TG_ORDER=order))
        res = {}
        for name, (dtype, n) in outputs.items():
            raw = open(os.path.join(d, name + ".bin"), "rb").read()
            if not raw:
                res[name] = torch.zeros(0)
            elif dtype == "bf16":
                res[name] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
            else:
                res[name] = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        return res


def check(name, got, want, atol, rtol=0.0):
    got = got.double().flatten()
    want = want.double().flatten()
    assert got.numel() == want.numel(), f"{name}: {got.numel()} vs {want.numel()} elements"
    if got.numel() == 0:
        print(f"  [ok  ] {name}: empty")
        return
    bad_nan = torch.isnan(got) & ~torch.isnan(want)
    err = (got - want).abs()
    err[torch.isnan(want) & torch.isnan(got)] = 0
    tol = atol + rtol * want.abs()
    worst = (err - tol).argmax().item()
    ok = not bad_nan.any() and bool((err <= tol).all())
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}: max abs err {err.max().item():.3e} (tol {atol:.1e}+{rtol:.1e}|x|)"
          + ("" if ok else f" worst @ {worst}: got {got[worst].item()} want {want[worst].item()}"
             + (f"; {int(bad_nan.sum())} unexpected NaN" if bad_nan.any() else "")))
    if not ok:
        FAILURES.append(name)


def seeded(seed):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------- conv1d


def case_conv(B, T, C, KW, with_state, bcast_state, seed):
    g = seeded(seed)
    ld_x, x_off = C + 13, 5
    x_full = torch.randn(B * T, ld_x, generator=g)
    w = torch.randn(C, KW, generator=g) * 0.5
    x = x_full[:, x_off:x_off + C].reshape(B, T, C).transpose(1, 2)  # [B, C, T] as transformers holds it
    if with_state:
        st = torch.randn(1 if bcast_state else B, C, KW - 1, generator=g)
        st_b = st.expand(B, C, KW - 1).clone()
    else:
        st = None
        st_b = torch.zeros(B, C, KW - 1)
    ref_state = st_b.clone()
    if T == 0:
        # Nothing to convolve (transformers' conv rejects the empty input); the
        # state passes through unchanged.
        ref = torch.zeros(B, C, 0)
    else:
        ref = m.causal_conv1d_update(x.clone(), ref_state, w, None, "silu")  # updates ref_state in place
    if not with_state and T > 0:
        # Zero state is the prefill path; hold the two transformers paths to each other.
        ref_fn = m.causal_conv1d_fn(x.clone(), w, None, activation="silu")
        check(f"conv1d ref self-consistency B{B} T{T}", ref_fn, ref, 1e-6)
    inputs = {"x": x_full, "w": w}
    if st is not None:
        inputs["state_in"] = st
    out = run("qwen35_conv1d_silu",
              dict(B=B, T=T, C=C, KW=KW, ld_x=ld_x, x_off=x_off,
                   state_bstride=0 if bcast_state else C * (KW - 1),
                   flags=(1 if with_state else 0) | 2),
              inputs, {"y": ("f32", B * T * C), "state_out": ("f32", B * C * (KW - 1))})
    tag = f"conv1d B{B} T{T} C{C} KW{KW} state={with_state} bcast={bcast_state}"
    check(tag + " y", out["y"], ref.transpose(1, 2).reshape(-1), 1e-5, 1e-5)
    check(tag + " state", out["state_out"], ref_state.reshape(-1), 0.0)


# ------------------------------------------------------------------- the GDN


def gdn_layout(B, T, Hk, Hv, Dv, seed, a_scale=1.0, a_log_lo=-2.0, a_log_hi=1.0, dt_shift=0.0):
    """Random operands in the fused-projection layouts the kernels read."""
    g = seeded(seed)
    key_dim, value_dim = Hk * DK, Hv * Dv
    pad = 7
    ld_qkv = 2 * key_dim + value_dim + pad
    q_off, k_off, v_off = 3, 3 + key_dim, 3 + 2 * key_dim
    qkv = torch.randn(B * T, ld_qkv, generator=g)
    # z | b | a, the tail of the projection, in its own buffer.
    ld_ab = value_dim + 2 * Hv + 5
    b_off, a_off = value_dim + 1, value_dim + 1 + Hv
    ab = torch.randn(B * T, ld_ab, generator=g) * a_scale
    a_log = torch.empty(Hv).uniform_(a_log_lo, a_log_hi, generator=g)
    dt_bias = torch.randn(Hv, generator=g) + dt_shift
    return dict(qkv=qkv, ab=ab, a_log=a_log, dt_bias=dt_bias, ld_qkv=ld_qkv, q_off=q_off, k_off=k_off,
                v_off=v_off, ld_ab=ld_ab, a_off=a_off, b_off=b_off)


def gdn_operands(L, B, T, Hk, Hv, Dv, dtype):
    """transformers' pre-rule operands, from the raw layout, in `dtype`."""
    qkv = L["qkv"].to(dtype)
    q = qkv[:, L["q_off"]:L["q_off"] + Hk * DK].reshape(B, T, Hk, DK)
    k = qkv[:, L["k_off"]:L["k_off"] + Hk * DK].reshape(B, T, Hk, DK)
    v = qkv[:, L["v_off"]:L["v_off"] + Hv * Dv].reshape(B, T, Hv, Dv)
    ab = L["ab"].to(dtype)
    b = ab[:, L["b_off"]:L["b_off"] + Hv].reshape(B, T, Hv)
    a = ab[:, L["a_off"]:L["a_off"] + Hv].reshape(B, T, Hv)
    # Qwen3_5GatedDeltaNet.forward, verbatim.
    beta = b.sigmoid()
    g = -L["a_log"].to(dtype).exp() * F.softplus(a + L["dt_bias"].to(dtype))
    if Hv // Hk > 1:
        q = q.repeat_interleave(Hv // Hk, dim=2)
        k = k.repeat_interleave(Hv // Hk, dim=2)
    return q, k, v, g, beta


def gdn_f64(q, k, v, g, beta, state):
    """Sequential gated delta rule in f64 (transformers' recurrent form)."""
    q = m.l2norm(q.double(), dim=-1, eps=1e-6) * DK ** -0.5
    k = m.l2norm(k.double(), dim=-1, eps=1e-6)
    v, g, beta = v.double(), g.double(), beta.double()
    B, T, H, _ = q.shape
    S = state.double().clone()
    out = torch.zeros_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]
        kv = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        out[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t])
    return out, S


def case_gdn(kernel, B, T, Hk, Hv, Dv, seed, state_mode="none", **layout_kw):
    L = gdn_layout(B, T, Hk, Hv, Dv, seed, **layout_kw)
    g_ = seeded(seed + 1000)
    snap = None
    if state_mode == "batch":
        snap = torch.randn(B, Hv, DK, Dv, generator=g_) * 0.1
        state0 = snap
    elif state_mode == "snapshot":
        snap = torch.randn(1, Hv, DK, Dv, generator=g_) * 0.1
        state0 = snap.expand(B, Hv, DK, Dv)
    else:
        state0 = torch.zeros(B, Hv, DK, Dv)

    ref, ref_state = gdn_f64(*gdn_operands(L, B, T, Hk, Hv, Dv, torch.float64), state0)
    q, k, v, g, beta = gdn_operands(L, B, T, Hk, Hv, Dv, torch.float32)
    init = None if state_mode == "none" else state0.contiguous().float()
    if T == 0:
        # transformers has no empty-sequence path; the answer is the start state.
        hf, hf_state = ref.float(), state0.float()
    elif kernel == "qwen35_gdn_chunk":
        hf, hf_state = m.torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=init,
                                                      output_final_state=True, use_qk_l2norm_in_kernel=True)
    else:
        hf, hf_state = m.torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=init,
                                                          output_final_state=True, use_qk_l2norm_in_kernel=True)

    ld_out, out_off = Hv * Dv + 9, 4
    params = dict(B=B, T=T, Hk=Hk, Hv=Hv, Dv=Dv, ld_qkv=L["ld_qkv"], q_off=L["q_off"], k_off=L["k_off"],
                  v_off=L["v_off"], ld_ab=L["ld_ab"], a_off=L["a_off"], b_off=L["b_off"], ld_out=ld_out,
                  out_off=out_off, state_bstride=0 if state_mode == "snapshot" else Hv * DK * Dv,
                  flags=(0 if state_mode == "none" else 1) | 2)
    inputs = {k_: L[k_] for k_ in ("qkv", "ab", "a_log", "dt_bias")}
    if snap is not None:
        inputs["state_in"] = snap
    out = run(kernel, params, inputs,
              {"out": ("f32", B * T * ld_out), "state_out": ("f32", B * Hv * DK * Dv)})

    got = out["out"].reshape(B * T, ld_out)
    got_heads = got[:, out_off:out_off + Hv * Dv]
    # Untouched columns of `out` must stay NaN: the kernel writes its window only.
    outside = torch.cat([got[:, :out_off], got[:, out_off + Hv * Dv:]], dim=1)
    tag = f"{kernel} B{B} T{T} Hk{Hk} Hv{Hv} Dv{Dv} state={state_mode}" + \
        ("" if not layout_kw else " " + ",".join(f"{a}={b}" for a, b in layout_kw.items()))
    if not torch.isnan(outside).all():
        print(f"  [FAIL] {tag}: wrote outside its output window")
        FAILURES.append(tag + " window")

    want = ref.reshape(B * T, Hv * Dv)
    amax = lambda t: t.abs().max().item() if t.numel() else 0.0
    hf_err = amax(hf.double().reshape(B * T, Hv * Dv) - want)
    hf_state_err = amax(hf_state.double() - ref_state)
    scale = max(amax(want), 1e-3)
    print(f"  {tag}: transformers fp32 err vs f64 {hf_err:.2e} (out), {hf_state_err:.2e} (state); |y|max {scale:.2e}")
    # The bound: 8x what transformers' own fp32 run misses by, floored at a few
    # ulps of the output scale for cases where the fp32 reference is exact.
    state_scale = max(ref_state.abs().max().item(), 1e-3)
    if FAST_MATH:
        atol, satol = 1e-4 * scale, 1e-4 * state_scale
    else:
        atol = max(8 * hf_err, 64 * 2 ** -24 * scale)
        satol = max(8 * hf_state_err, 64 * 2 ** -24 * state_scale)
    if T > 0:
        check(tag + " out", got_heads, want, atol)
    check(tag + " state", out["state_out"], ref_state.reshape(-1), satol)


def case_recurrent_in_place(seed):
    """state_out == state_in, one state per batch row."""
    B, T, Hk, Hv, Dv = 2, 3, 1, 2, 64
    L = gdn_layout(B, T, Hk, Hv, Dv, seed)
    state0 = torch.randn(B, Hv, DK, Dv, generator=seeded(seed + 1)) * 0.1
    ref, ref_state = gdn_f64(*gdn_operands(L, B, T, Hk, Hv, Dv, torch.float64), state0)
    params = dict(B=B, T=T, Hk=Hk, Hv=Hv, Dv=Dv, ld_qkv=L["ld_qkv"], q_off=L["q_off"], k_off=L["k_off"],
                  v_off=L["v_off"], ld_ab=L["ld_ab"], a_off=L["a_off"], b_off=L["b_off"], ld_out=Hv * Dv,
                  out_off=0, state_bstride=Hv * DK * Dv, flags=3, in_place=1)
    inputs = {k_: L[k_] for k_ in ("qkv", "ab", "a_log", "dt_bias")}
    inputs["state_in"] = state0
    out = run("qwen35_gdn_recurrent", params, inputs,
              {"out": ("f32", B * T * Hv * Dv), "state_in": ("f32", B * Hv * DK * Dv)})
    check("gdn_recurrent in-place out", out["out"], ref.reshape(-1), 1e-5)
    check("gdn_recurrent in-place state", out["state_in"], ref_state.reshape(-1), 1e-5)


def case_chunk_ws(seed):
    """W must invert (I + A) and Aq must match transformers' intra-chunk attention."""
    B, T, Hk, Hv, Dv = 1, 70, 1, 1, 32
    L = gdn_layout(B, T, Hk, Hv, Dv, seed)
    params = dict(B=B, T=T, Hk=Hk, Hv=Hv, Dv=Dv, ld_qkv=L["ld_qkv"], q_off=L["q_off"], k_off=L["k_off"],
                  v_off=L["v_off"], ld_ab=L["ld_ab"], a_off=L["a_off"], b_off=L["b_off"], ld_out=Hv * Dv,
                  out_off=0, state_bstride=0, flags=0, dump_ws=1)
    inputs = {k_: L[k_] for k_ in ("qkv", "ab", "a_log", "dt_bias")}
    out = run("qwen35_gdn_chunk", params, inputs, {"out": ("f32", B * T * Hv * Dv),
                                                    "ws_w": ("f32", 2 * 64 * 64), "ws_aq": ("f32", 2 * 64 * 64)})
    q, k, v, g, beta = gdn_operands(L, B, T, Hk, Hv, Dv, torch.float64)
    q = m.l2norm(q, dim=-1, eps=1e-6)[0, :, 0] * DK ** -0.5
    k = m.l2norm(k, dim=-1, eps=1e-6)[0, :, 0]
    g, beta = g[0, :, 0], beta[0, :, 0]
    W = out["ws_w"].double().reshape(2, 64, 64)
    Aq = out["ws_aq"].double().reshape(2, 64, 64)
    for c in range(2):
        rows = slice(c * 64, min(T, c * 64 + 64))
        n = rows.stop - rows.start
        G = g[rows].cumsum(0)
        gam = (G[:, None] - G[None, :]).tril().exp().tril()
        A = ((beta[rows, None] * (k[rows] @ k[rows].T)) * gam).tril(-1)
        eye = torch.eye(n, dtype=torch.float64)
        check(f"chunk ws W (I+A) = I chunk {c}", W[c, :n, :n] @ (eye + A), eye, 2e-5)
        check(f"chunk ws Aq chunk {c}", Aq[c, :n, :n], ((q[rows] @ k[rows].T) * gam), 2e-5)
        if n < 64:
            check(f"chunk ws W padding is identity chunk {c}", W[c, n:, n:], torch.eye(64 - n), 0.0)


# ----------------------------------------------------------------- the norm


def case_gated_norm(bf16, seed, H=4, rows=37):
    g = seeded(seed)
    D = 128
    ld_x, x_off, ld_z, z_off, ld_out, out_off = H * D + 3, 2, H * D + 11, 7, H * D + 5, 1
    x = torch.randn(rows, ld_x, generator=g)
    z = torch.randn(rows, ld_z, generator=g) * 2
    w = torch.randn(D, generator=g) * 0.2 + 1
    eps = 1e-6
    norm = m.Qwen3_5RMSNormGated(D, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(w)
        ref = norm(x[:, x_off:x_off + H * D].reshape(-1, D), z[:, z_off:z_off + H * D].reshape(-1, D))
    kname = "qwen35_gated_rms_norm_bf16" if bf16 else "qwen35_gated_rms_norm_f32"
    out = run(kname, dict(rows=rows, H=H, D=D, ld_x=ld_x, x_off=x_off, ld_z=ld_z, z_off=z_off, ld_out=ld_out,
                          out_off=out_off, eps=eps),
              {"x": x, "z": z, "w": w}, {"out": ("bf16" if bf16 else "f32", rows * ld_out)})
    got = out["out"].reshape(rows, ld_out)[:, out_off:out_off + H * D]
    want = ref.reshape(rows, H * D)
    if bf16:
        check(kname, got, want.to(torch.bfloat16).float(), 0.0, 2 ** -7)
    else:
        check(kname, got, want, 1e-5, 1e-5)


# -------------------------------------------------------- attention extras


def case_qk_norm_rope(seed, Hq=4, Hkv=2, B=2, T=9):
    g = seeded(seed)
    D = 256
    theta, eps, pos_offset, cap = 1e7, 1e-6, 30000, 30016
    cfg = Qwen3_5TextConfig(head_dim=D, num_attention_heads=Hq, num_key_value_heads=Hkv, hidden_size=Hq * D,
                            rope_parameters={"rope_type": "default", "rope_theta": theta,
                                             "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                             "mrope_interleaved": True})
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    R = rot.inv_freq.numel() * 2
    ld_p = 2 * Hq * D + 2 * Hkv * D + 6
    q_off, k_off, v_off = 1, 1 + 2 * Hq * D, 1 + 2 * Hq * D + Hkv * D
    p = torch.randn(B * T, ld_p, generator=g)
    qw = torch.randn(D, generator=g) * 0.1
    kw = torch.randn(D, generator=g) * 0.1
    # Qwen3_5Attention.forward, verbatim, from the projection output onward.
    qg = p[:, q_off:q_off + 2 * Hq * D].reshape(B, T, Hq, 2 * D)
    q, _gate = torch.chunk(qg, 2, dim=-1)
    qn = m.Qwen3_5RMSNorm(D, eps=eps)
    kn = m.Qwen3_5RMSNorm(D, eps=eps)
    with torch.no_grad():
        qn.weight.copy_(qw)
        kn.weight.copy_(kw)
        q = qn(q).transpose(1, 2)
        k = kn(p[:, k_off:k_off + Hkv * D].reshape(B, T, Hkv, D)).transpose(1, 2)
        pos = (torch.arange(T) + pos_offset)[None, None, :].expand(3, B, T)
        cos, sin = rot(q, pos)
        q, k = m.apply_rotary_pos_emb(q, k, cos, sin)
    v = p[:, v_off:v_off + Hkv * D].reshape(B, T, Hkv, D)
    out = run("qwen35_attn_qk_norm_rope",
              dict(B=B, T=T, Hq=Hq, Hkv=Hkv, D=D, rotary_dim=R, ld_p=ld_p, q_off=q_off, k_off=k_off, v_off=v_off,
                   pos_offset=pos_offset, kv_capacity=cap, theta=theta, eps=eps),
              {"p": p, "q_norm_w": qw, "k_norm_w": kw},
              {"q_out": ("f32", B * T * Hq * D), "k_cache": ("f32", B * cap * Hkv * D),
               "v_cache": ("f32", B * cap * Hkv * D)})
    tag = f"attn_qk_norm_rope D{D} R{R} pos{pos_offset}"
    check(tag + " q", out["q_out"], q.transpose(1, 2).reshape(-1), 2e-4)
    kc = out["k_cache"].reshape(B, cap, Hkv, D)
    vc = out["v_cache"].reshape(B, cap, Hkv, D)
    check(tag + " k", kc[:, pos_offset:pos_offset + T], k.transpose(1, 2), 2e-4)
    check(tag + " v", vc[:, pos_offset:pos_offset + T], v, 0.0)
    untouched = torch.cat([kc[:, :pos_offset].flatten(), kc[:, pos_offset + T:].flatten()])
    if not torch.isnan(untouched).all():
        print(f"  [FAIL] {tag}: wrote cache slots outside [pos, pos+T)")
        FAILURES.append(tag + " cache window")


def case_qk_rope_posbuf(seed):
    """The device-buffer position variant: identical to the scalar one, and
    positions at or past the capacity are skipped, in 64 bits."""
    g = seeded(seed)
    B, T, Hq, Hkv, D, cap = 2, 5, 2, 1, 256, 12
    ld_p = 2 * Hq * D + 2 * Hkv * D
    p = torch.randn(B * T, ld_p, generator=g)
    qw, kw = torch.randn(D, generator=g) * 0.1, torch.randn(D, generator=g) * 0.1
    base = dict(B=B, T=T, Hq=Hq, Hkv=Hkv, D=D, rotary_dim=64, ld_p=ld_p, q_off=0, k_off=2 * Hq * D,
                v_off=2 * Hq * D + Hkv * D, kv_capacity=cap, theta=1e7, eps=1e-6)
    outs = {"q_out": ("f32", B * T * Hq * D), "k_cache": ("f32", B * cap * Hkv * D),
            "v_cache": ("f32", B * cap * Hkv * D)}
    inputs = {"p": p, "q_norm_w": qw, "k_norm_w": kw}

    def via_buf(pos):
        return run("qwen35_attn_qk_norm_rope", dict(base, pos_offset=0, posbuf=1),
                   dict(inputs, pos_buf=torch.tensor([pos], dtype=torch.int64).to(torch.int32)
                        if pos < 2 ** 31 else torch.tensor([pos - 2 ** 32], dtype=torch.int32)), outs)

    scalar = run("qwen35_attn_qk_norm_rope", dict(base, pos_offset=4), inputs, outs)
    buffered = via_buf(4)
    same = all(torch.equal(scalar[k].view(torch.int32), buffered[k].view(torch.int32)) for k in outs)
    print(f"  [{'ok  ' if same else 'FAIL'}] qk_norm_rope_posbuf bit-identical to the scalar variant")
    if not same:
        FAILURES.append("qk_norm_rope_posbuf vs scalar")
    # Positions 10..14 with capacity 12: tokens 0 and 1 land, 2..4 are skipped.
    part = via_buf(10)
    q = part["q_out"].reshape(B, T, Hq * D)
    kc = part["k_cache"].reshape(B, cap, Hkv * D)
    ok = (not torch.isnan(q[:, :2]).any() and torch.isnan(q[:, 2:]).all()
          and not torch.isnan(kc[:, 10:12]).any() and torch.isnan(kc[:, :10]).all())
    print(f"  [{'ok  ' if ok else 'FAIL'}] qk_norm_rope_posbuf skips tokens at or past the capacity")
    if not ok:
        FAILURES.append("qk_norm_rope_posbuf capacity skip")
    # An offset near u32::MAX: in 32 bits pos + t wraps to 0..2 and would
    # overwrite the first cache slots. Everything must be skipped.
    wrap = via_buf(2 ** 32 - 2)
    ok = all(torch.isnan(wrap[k]).all() for k in outs)
    print(f"  [{'ok  ' if ok else 'FAIL'}] qk_norm_rope_posbuf: an offset that would wrap writes nothing")
    if not ok:
        FAILURES.append("qk_norm_rope_posbuf wrap")


def case_prefix_rows(seed, B, P, S, s_cap, Tq, q_pos):
    """Shared-prefix attention: bit-identical to flash_attn_rows over each row's
    copied `prefix ‖ suffix`, and equal to torch's causal softmax. K/V slots
    past every live length are NaN, so reading one would show."""
    g = seeded(seed)
    H, Hkv, D = 8, 2, 256
    p_cap = P + 2
    kp, vp = torch.randn(p_cap, Hkv, D, generator=g), torch.randn(p_cap, Hkv, D, generator=g)
    ks, vs = torch.randn(B, s_cap, Hkv, D, generator=g), torch.randn(B, s_cap, Hkv, D, generator=g)
    for t in (kp, vp):
        t[P:] = float("nan")
    for t in (ks, vs):
        t[:, S:] = float("nan")
    q = torch.randn(B, Tq, H, D, generator=g)
    scale = D ** -0.5
    n = B * Tq * H * D
    got = run("qwen35_attn_prefix_rows",
              dict(B=B, Tq=Tq, H=H, Hkv=Hkv, P=P, suffix_cap=s_cap, scale=scale),
              {"q": q, "kp": kp, "vp": vp, "ks": ks, "vs": vs,
               "suffix_len": torch.tensor([S], dtype=torch.int32),
               "q_pos": torch.tensor([q_pos], dtype=torch.int32)},
              {"o": ("f32", n)})["o"]
    kf = torch.cat([kp[:P].expand(B, P, Hkv, D), ks], dim=1).contiguous()
    vf = torch.cat([vp[:P].expand(B, P, Hkv, D), vs], dim=1).contiguous()
    want = run("flash_attn_rows", dict(B=B, Tq=Tq, H=H, Hkv=Hkv, D=D, kv_capacity=P + s_cap, scale=scale),
               {"q": q, "k": kf, "v": vf, "tkv": torch.tensor([P + S], dtype=torch.int32),
                "q_pos": torch.tensor([q_pos], dtype=torch.int32),
                "kv_pos": torch.tensor([0], dtype=torch.int32)},
               {"o": ("f32", n)})["o"]
    tag = f"attn_prefix_rows B{B} P{P} S{S}/{s_cap} Tq{Tq} q@{q_pos}"
    same = torch.equal(got.view(torch.int32), want.view(torch.int32)) and not torch.isnan(got).any()
    print(f"  [{'ok  ' if same else 'FAIL'}] {tag}: bit-identical to flash_attn_rows on a copied prefix")
    if not same:
        FAILURES.append(tag + " vs flash_attn_rows")
    n_kv = P + S
    kk = kf[:, :n_kv].double().repeat_interleave(H // Hkv, dim=2)
    vv = vf[:, :n_kv].double().repeat_interleave(H // Hkv, dim=2)
    sc = torch.einsum("bthd,bshd->bhts", q.double(), kk) * scale
    mask = torch.arange(n_kv)[None, :] > (q_pos + torch.arange(Tq))[:, None]
    ref = torch.einsum("bhts,bshd->bthd", sc.masked_fill(mask, float("-inf")).softmax(-1), vv)
    check(tag + " vs torch", got, ref, 1e-5, 1e-5)


def case_prefix_decode(seed, B, P, S, s_cap, q_pos):
    """Shared-prefix split-KV decode (one query per row) against torch's causal
    softmax over prefix ‖ suffix. Dead K/V slots are NaN."""
    g = seeded(seed)
    H, Hkv, D = 8, 2, 256
    p_cap = P + 2
    kp, vp = torch.randn(p_cap, Hkv, D, generator=g), torch.randn(p_cap, Hkv, D, generator=g)
    ks, vs = torch.randn(B, s_cap, Hkv, D, generator=g), torch.randn(B, s_cap, Hkv, D, generator=g)
    for t in (kp, vp):
        t[P:] = float("nan")
    for t in (ks, vs):
        t[:, S:] = float("nan")
    q = torch.randn(B, 1, H, D, generator=g)
    scale = D ** -0.5
    got = run("qwen35_attn_prefix_decode", dict(B=B, H=H, Hkv=Hkv, P=P, suffix_cap=s_cap, scale=scale),
              {"q": q, "kp": kp, "vp": vp, "ks": ks, "vs": vs,
               "suffix_len": torch.tensor([S], dtype=torch.int32),
               "q_pos": torch.tensor([q_pos], dtype=torch.int32)},
              {"o": ("f32", B * H * D)})["o"]
    n_kv = min(P + S, q_pos + 1)
    kf = torch.cat([kp[:P].expand(B, P, Hkv, D), ks], dim=1)[:, :n_kv].double()
    vf = torch.cat([vp[:P].expand(B, P, Hkv, D), vs], dim=1)[:, :n_kv].double()
    kk, vv = kf.repeat_interleave(H // Hkv, dim=2), vf.repeat_interleave(H // Hkv, dim=2)
    sc = torch.einsum("bthd,bshd->bhts", q.double(), kk) * scale
    ref = torch.einsum("bhts,bshd->bthd", sc.softmax(-1), vv)
    check(f"attn_prefix_decode B{B} P{P} S{S}/{s_cap} q@{q_pos} vs torch", got, ref, 1e-5, 1e-5)


def case_embed_rows(seed):
    """The bf16 embedding gather: exact against indexing the table in torch;
    an out-of-range id is a NaN row and the others are intact."""
    g = seeded(seed)
    vocab, hidden = 50, 96
    table = torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    ids = torch.tensor([0, 49, 7, 7, 50, 3, 2 ** 31 - 1], dtype=torch.int32)
    out = run("qwen35_embed_rows_bf16", dict(n=len(ids), hidden=hidden, vocab=vocab),
              {"ids": ids, "table": table.view(torch.int16)}, {"out": ("f32", len(ids) * hidden)})["out"]
    out = out.reshape(len(ids), hidden)
    good = ids < vocab
    want = table.float()[ids[good].long()]
    ok = torch.equal(out[good].view(torch.int32), want.view(torch.int32)) and bool(torch.isnan(out[~good]).all())
    print(f"  [{'ok  ' if ok else 'FAIL'}] embed_rows_bf16: exact gather, bad ids NaN")
    if not ok:
        FAILURES.append("embed_rows_bf16")


def case_prefix_varlen(seed, decode):
    """Per-row suffix lengths and query positions (row_stride 1): each row of
    a ragged batch is bit-identical to that row run alone with shared values."""
    g = seeded(seed)
    H, Hkv, D, P, s_cap = 8, 2, 256, 20, 6
    rows = [(0, 19), (1, 20), (6, 25), (3, 22)] if decode else [(0, 20), (2, 20), (6, 21), (9, 20)]
    Tq = 1 if decode else 3
    B = len(rows)
    kp, vp = torch.randn(P, Hkv, D, generator=g), torch.randn(P, Hkv, D, generator=g)
    ks, vs = torch.randn(B, s_cap, Hkv, D, generator=g), torch.randn(B, s_cap, Hkv, D, generator=g)
    for b, (n, _) in enumerate(rows):
        ks[b, min(n, s_cap):] = float("nan")
        vs[b, min(n, s_cap):] = float("nan")
    q = torch.randn(B, Tq, H, D, generator=g)
    kname = "qwen35_attn_prefix_decode" if decode else "qwen35_attn_prefix_rows"
    base = dict(Tq=Tq, H=H, Hkv=Hkv, P=P, suffix_cap=s_cap, scale=D ** -0.5)
    per = Tq * H * D
    got = run(kname, dict(base, B=B, row_stride=1),
              {"q": q, "kp": kp, "vp": vp, "ks": ks, "vs": vs,
               "suffix_len": torch.tensor([n for n, _ in rows], dtype=torch.int32),
               "q_pos": torch.tensor([p for _, p in rows], dtype=torch.int32)},
              {"o": ("f32", B * per)})["o"].reshape(B, per)
    ok = True
    for b, (n, p) in enumerate(rows):
        one = run(kname, dict(base, B=1),
                  {"q": q[b:b + 1], "kp": kp, "vp": vp, "ks": ks[b:b + 1], "vs": vs[b:b + 1],
                   "suffix_len": torch.tensor([min(n, s_cap)], dtype=torch.int32),
                   "q_pos": torch.tensor([p], dtype=torch.int32)},
                  {"o": ("f32", per)})["o"]
        ok &= torch.equal(got[b].view(torch.int32), one.view(torch.int32))
    tag = f"{kname} varlen: each row bit-identical to it alone"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {tag}")
    if not ok:
        FAILURES.append(tag)


def case_gdn_varlen(kname, seed, T, lens, Dv=32):
    """Ragged rows (flags & 4, per-row seq_lens) on a GDN path: each row's
    outputs and final state bit-identical to that row run alone at its own
    length, and its rows past that length never written."""
    g = seeded(seed)
    B, Hk, Hv, DK = len(lens), 1, 2, 128
    key_w, val_w = Hk * DK, Hv * Dv
    ld_qkv = 2 * key_w + val_w
    qkv = torch.randn(B, T, ld_qkv, generator=g)
    ab = torch.randn(B, T, 2 * Hv, generator=g) * 2
    a_log = torch.rand(Hv, generator=g) * 1.5 - 0.5
    dt_bias = torch.randn(Hv, generator=g)
    st = torch.randn(B, Hv, DK, Dv, generator=g) * 0.1
    per_state = Hv * DK * Dv

    def params(b, t, extra):
        return dict(B=b, T=t, Hk=Hk, Hv=Hv, Dv=Dv, ld_qkv=ld_qkv, q_off=0, k_off=key_w, v_off=2 * key_w,
                    ld_ab=2 * Hv, a_off=0, b_off=Hv, ld_out=val_w, out_off=0, state_bstride=per_state, **extra)

    outs = lambda b, t: {"out": ("f32", b * t * val_w), "state_out": ("f32", b * per_state)}
    got = run(kname, params(B, T, dict(flags=1 | 2 | 4)),
              {"qkv": qkv, "ab": ab, "a_log": a_log, "dt_bias": dt_bias, "state_in": st,
               "seq_lens": torch.tensor(lens, dtype=torch.int32)}, outs(B, T))
    y, so = got["out"].reshape(B, T, val_w), got["state_out"].reshape(B, per_state)
    ok = True
    for b, n in enumerate(lens):
        live = min(n, T)
        one = run(kname, params(1, live, dict(flags=1 | 2)),
                  {"qkv": qkv[b, :live], "ab": ab[b, :live], "a_log": a_log, "dt_bias": dt_bias,
                   "state_in": st[b]}, outs(1, live))
        ok &= torch.equal(y[b, :live].reshape(-1).view(torch.int32), one["out"].view(torch.int32))
        ok &= bool(torch.isnan(y[b, live:]).all())
        ok &= torch.equal(so[b].view(torch.int32), one["state_out"].view(torch.int32))
    tag = f"{kname} varlen T{T} lens {lens}: each row bit-identical to it alone"
    print(f"  [{'ok  ' if ok else 'FAIL'}] {tag}")
    if not ok:
        FAILURES.append(tag)


def case_conv_varlen(seed):
    """The conv's ragged form: outputs and carried state per row equal the row
    run alone, including rows shorter than the kernel's history."""
    g = seeded(seed)
    T, Ch, KW = 12, 40, 4
    lens = [0, 1, 2, 3, 7, 12, 15]
    B, hist = len(lens), KW - 1
    x = torch.randn(B, T, Ch, generator=g)
    w = torch.randn(Ch, KW, generator=g)
    st = torch.randn(B, Ch, hist, generator=g)
    base = lambda b, t, flags: dict(B=b, T=t, C=Ch, KW=KW, ld_x=Ch, x_off=0, state_bstride=Ch * hist,
                                    flags=flags)
    got = run("qwen35_conv1d_silu", base(B, T, 1 | 2 | 4),
              {"x": x, "w": w, "state_in": st, "seq_lens": torch.tensor(lens, dtype=torch.int32)},
              {"y": ("f32", B * T * Ch), "state_out": ("f32", B * Ch * hist)})
    y, so = got["y"].reshape(B, T, Ch), got["state_out"].reshape(B, Ch * hist)
    ok = True
    for b, n in enumerate(lens):
        live = min(n, T)
        one = run("qwen35_conv1d_silu", base(1, live, 1 | 2),
                  {"x": x[b, :live], "w": w, "state_in": st[b]},
                  {"y": ("f32", live * Ch), "state_out": ("f32", Ch * hist)})
        ok &= torch.equal(y[b, :live].reshape(-1).view(torch.int32), one["y"].view(torch.int32))
        ok &= bool(torch.isnan(y[b, live:]).all())
        ok &= torch.equal(so[b].view(torch.int32), one["state_out"].view(torch.int32))
    print(f"  [{'ok  ' if ok else 'FAIL'}] conv1d_silu varlen: each row bit-identical to it alone")
    if not ok:
        FAILURES.append("conv1d_silu varlen")


def case_qk_rope_slot_base(seed):
    """`slot_base` moves only the cache slot: a suffix cached relative to a
    prefix of P holds, bit for bit, what a cache from position 0 holds at P.."""
    g = seeded(seed)
    B, T, Hq, Hkv, D, P, cap = 2, 4, 2, 1, 256, 9, 6
    ld_p = 2 * Hq * D + 2 * Hkv * D
    p = torch.randn(B * T, ld_p, generator=g)
    qw, kw = torch.randn(D, generator=g) * 0.1, torch.randn(D, generator=g) * 0.1
    base = dict(B=B, T=T, Hq=Hq, Hkv=Hkv, D=D, rotary_dim=64, ld_p=ld_p, q_off=0, k_off=2 * Hq * D,
                v_off=2 * Hq * D + Hkv * D, theta=1e7, eps=1e-6)
    inputs = {"p": p, "q_norm_w": qw, "k_norm_w": kw}

    def outs(c):
        return {"q_out": ("f32", B * T * Hq * D), "k_cache": ("f32", B * c * Hkv * D),
                "v_cache": ("f32", B * c * Hkv * D)}

    whole = run("qwen35_attn_qk_norm_rope", dict(base, pos_offset=P + 1, kv_capacity=P + cap),
                inputs, outs(P + cap))
    rel = run("qwen35_attn_qk_norm_rope", dict(base, pos_offset=P + 1, kv_capacity=cap, slot_base=P),
              inputs, outs(cap))
    ok = torch.equal(whole["q_out"].view(torch.int32), rel["q_out"].view(torch.int32))
    for name in ("k_cache", "v_cache"):
        w = whole[name].reshape(B, P + cap, Hkv * D)
        r = rel[name].reshape(B, cap, Hkv * D)
        ok &= torch.equal(w[:, P + 1:P + 1 + T].view(torch.int32), r[:, 1:1 + T].view(torch.int32))
        ok &= bool(torch.isnan(r[:, :1]).all() and torch.isnan(r[:, 1 + T:]).all())
    print(f"  [{'ok  ' if ok else 'FAIL'}] qk_norm_rope slot_base: absolute RoPE, relative slot, nothing else written")
    if not ok:
        FAILURES.append("qk_norm_rope slot_base")


def case_attn_gate(bf16, in_place, seed):
    g = seeded(seed)
    rows, Hq, D = 11, 3, 64
    ld_p, q_off = 2 * Hq * D + 3, 2
    p = torch.randn(rows, ld_p, generator=g) * 3
    attn = torch.randn(rows, Hq * D, generator=g)
    gate = p[:, q_off:q_off + 2 * Hq * D].reshape(rows, Hq, 2 * D)[..., D:].reshape(rows, Hq * D)
    ref = attn * torch.sigmoid(gate)  # Qwen3_5Attention.forward
    kname = "qwen35_attn_gate_bf16" if bf16 else "qwen35_attn_gate_f32"
    ld_out, out_off = (Hq * D, 0) if in_place else (Hq * D + 4, 3)
    params = dict(rows=rows, Hq=Hq, D=D, ld_p=ld_p, q_off=q_off, ld_out=ld_out, out_off=out_off)
    outputs = {"attn": ("f32", rows * Hq * D)} if in_place else {"out": ("bf16" if bf16 else "f32", rows * ld_out)}
    if in_place:
        params["in_place"] = 1
    out = run(kname, params, {"p": p, "attn": attn}, outputs)
    got = out["attn"] if in_place else out["out"].reshape(rows, ld_out)[:, out_off:out_off + Hq * D]
    tag = kname + (" in-place" if in_place else "")
    if bf16:
        check(tag, got, ref.to(torch.bfloat16).float(), 0.0, 2 ** -7)
    else:
        check(tag, got, ref, 1e-6, 1e-6)


# ------------------------------------------------------------------ scoring


def case_score(bf16, seed):
    g = seeded(seed)
    rows, hidden, vocab = 20, 300, 50
    h = torch.randn(rows, hidden, generator=g)
    nw = torch.randn(hidden, generator=g) * 0.1
    emb = torch.randn(vocab, hidden, generator=g) * 0.05
    if bf16:
        emb = emb.to(torch.bfloat16)
    answers = torch.randint(0, vocab, (17,), generator=g, dtype=torch.int32)
    slots = torch.tensor([3, 19, 0, 7], dtype=torch.int32)
    norm = m.Qwen3_5RMSNorm(hidden, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(nw)
        logits = norm(h[slots.long()]) @ emb.float()[answers.long()].T
    logp = torch.log_softmax(logits.double(), dim=-1)
    kname = "qwen35_score_rows_bf16" if bf16 else "qwen35_score_rows_f32"
    emb_in = emb.view(torch.int16) if bf16 else emb
    out = run(kname, dict(rows=rows, hidden=hidden, n_ans=17, vocab=vocab, n_slots=len(slots), eps=1e-6,
                          w_offset=1.0),
              {"h": h, "norm_w": nw, "emb": emb_in, "answers": answers, "slots": slots},
              {"logits": ("f32", len(slots) * 17), "logprobs": ("f32", len(slots) * 17)})
    check(kname + " logits", out["logits"], logits, 1e-5, 1e-5)
    check(kname + " logprobs", out["logprobs"], logp, 1e-5, 1e-5)
    # Out-of-range indices score NaN rather than reading out of bounds.
    bad = run(kname, dict(rows=rows, hidden=hidden, n_ans=3, vocab=vocab, n_slots=2, eps=1e-6, w_offset=1.0),
              {"h": h, "norm_w": nw, "emb": emb_in,
               "answers": torch.tensor([1, vocab, 2], dtype=torch.int32),
               "slots": torch.tensor([rows, 1], dtype=torch.int32)},
              {"logits": ("f32", 6), "logprobs": ("f32", 6)})
    lg = bad["logits"].reshape(2, 3)
    lp = bad["logprobs"].reshape(2, 3)
    ok = bool(torch.isnan(lg[0]).all() and torch.isnan(lp[0]).all()
              and torch.isnan(lg[1, 1]) and torch.isnan(lp[1, 1]))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {kname} out-of-range slot/answer -> NaN in both outputs")
    if not ok:
        FAILURES.append(kname + " out-of-range")
    # The valid answers beside the bad one are a two-way softmax, unpoisoned.
    with torch.no_grad():
        l2 = norm(h[1:2]) @ emb.float()[[1, 2]].T
    check(kname + " valid logits beside a bad answer", lg[1, [0, 2]], l2[0], 1e-5, 1e-5)
    check(kname + " valid logprobs beside a bad answer", lp[1, [0, 2]],
          torch.log_softmax(l2[0].double(), dim=-1), 1e-5, 1e-5)


# ------------------------------------------------------------ host contract

ROOT = os.path.dirname(os.path.dirname(HERE))


def _kernel_signatures():
    """{kernel name: [(buffer index, 'buf'|'u32'|'f32')]} from the .metal sources,
    expanding the macro-generated kernels through their instantiations."""
    import re
    sigs, macros = {}, {}
    for f in ("qwen35_gdn", "qwen35_attn", "qwen35_score"):
        src = open(os.path.join(ROOT, "kernels", f + ".metal")).read()
        for m in re.finditer(r"kernel void (\w+)\((.*?)\)\s*\\?\s*\{", src, re.S):
            params = []
            for p in re.finditer(r"([^,()]*?)\b\w+\s*\[\[buffer\((\d+)\)\]\]", m.group(2)):
                decl = p.group(1)
                kind = "buf" if "device" in decl else ("f32" if "float" in decl else "u32")
                params.append((int(p.group(2)), kind))
            (macros if m.group(1) == "NAME" else sigs)[m.group(1)] = params
        for m in re.finditer(r"^#define (\w+)\(NAME", src, re.M):
            body = src[m.start():]
            body_sig = re.search(r"kernel void NAME\((.*?)\)\s*\\?\s*\{", body, re.S).group(1)
            params = []
            for p in re.finditer(r"([^,()]*?)\b\w+\s*\[\[buffer\((\d+)\)\]\]", body_sig):
                decl = p.group(1)
                kind = "buf" if "device" in decl else ("f32" if "float" in decl else "u32")
                params.append((int(p.group(2)), kind))
            for inst in re.finditer(rf"^{m.group(1)}\((\w+),", src, re.M):
                sigs[inst.group(1)] = params
    return sigs


def _host_binds():
    """{kernel name: [(index, kind)]} from src/qwen35.rs's dispatch closures."""
    import re
    rs = open(os.path.join(ROOT, "src", "qwen35.rs")).read()
    # `let name = out_kernel("base", ..)` picks the _f32 or _bf16 variant; like
    # the pipelines below, the name is reused, so resolve it by position.
    name_defs = [(m.start(), m.group(1), [m.group(2) + "_f32", m.group(2) + "_bf16"])
                 for m in re.finditer(r'let (\w+) = out_kernel\(\s*"(\w+)"', rs)]
    # `let name = match .. { A => "kernel_a", B => "kernel_b" };` — one dispatch
    # site serving several kernels.
    name_defs += [(m.start(), m.group(1), re.findall(r'"(qwen35_\w+)"', m.group(2)))
                  for m in re.finditer(r'let (\w+) = match [^{]*\{(.*?)\};', rs, re.S)]
    name_defs.sort()
    # (position, variable, kernels): a dispatch resolves its pipeline variable
    # to the nearest `let` before it, since most functions reuse the name `p`.
    pipe_defs = []
    for m in re.finditer(r'let (\w+) = (?:pipeline_for\(\s*rt,\s*|rt\.pipeline\(\s*)(&?)("?)(\w+)', rs):
        var, lit, name = m.group(1), m.group(3), m.group(4)
        if lit:
            kernels = [name]
        else:
            found = [k for pos, v, k in name_defs if v == name and pos < m.start()]
            if not found:
                continue  # `pipeline_for`'s own body, not a dispatch site
            kernels = found[-1]
        pipe_defs.append((m.start(), var, kernels))
    binds = {}
    # Calls only: the helper's own `fn dispatch_groups(` definition is not one.
    for m in re.finditer(r"(?<!fn )\b(dispatch_groups|dispatch_2d)\(", rs):
        depth, i = 0, m.end() - 1
        while True:
            depth += {"(": 1, ")": -1}.get(rs[i], 0)
            if depth == 0:
                break
            i += 1
        call = rs[m.end():i]
        var = re.search(r"&(\w+)", call).group(1)
        kinds = {"gpu_buf": "buf", "u32": "u32", "f32": "f32"}
        b = [(int(x.group(2)), kinds[x.group(1)])
             for x in re.finditer(r"set_(gpu_buf|u32|f32)\(bnd,[^;]*?, (\d+)\)", call, re.S)]
        kernels = [k for pos, v, k in pipe_defs if v == var and pos < m.start()][-1]
        for k in kernels:
            assert k not in binds, f"{k} is dispatched twice; the contract check needs one site"
            binds[k] = b
    consts = {}
    for m in re.finditer(r"^const (\w+): usize = ([^;]+);", rs, re.M):
        consts[m.group(1)] = eval(m.group(2))
    for m in re.finditer(r"^pub const (\w+): u32 = ([^;]+);", rs, re.M):
        consts[m.group(1)] = eval(m.group(2))
    return binds, consts


def case_host_contract():
    sigs = _kernel_signatures()
    binds, consts = _host_binds()
    for k, sig in sorted(sigs.items()):
        got = sorted(binds.get(k, []))
        # A site serving several kernels may bind one slot in a `match`, one
        # kind per arm: the kernel's kind must be among the host's for that
        # slot, and the slots must be exactly the kernel's.
        host_kinds = {}
        for i, kind in got:
            host_kinds.setdefault(i, set()).add(kind)
        ok = set(host_kinds) == {i for i, _ in sig} and all(kind in host_kinds[i] for i, kind in sig)
        print(f"  [{'ok  ' if ok else 'FAIL'}] binds {k}: {len(sig)} slots"
              + ("" if ok else f"\n      kernel {sorted(sig)}\n      host   {got}"))
        if not ok:
            FAILURES.append(f"host binds for {k}")
    kc = dict(l.split() for l in subprocess.run([HARNESS, "--constants"], check=True, capture_output=True,
                                                   text=True).stdout.splitlines())
    kc = {k: int(v) for k, v in kc.items()}
    for host, kernel, scale in [("GDN_KEY_DIM", "GDN_DK", 1), ("GDN_CHUNK", "GDN_C", 1),
                                ("GDN_VALUE_BLOCK", "GDN_BV", 1), ("PREP_THREADS", "GDN_PREP_THREADS", 1),
                                ("SCAN_THREADS", "GDN_SCAN_THREADS", 1),
                                ("PREP_TG_BYTES", "GDN_PREP_TG_FLOATS", 4), ("SCAN_TG_BYTES", "GDN_SCAN_TG_FLOATS", 4),
                                ("REC_TG_BYTES", "GDN_REC_TG_FLOATS", 4),
                                ("REDUCE_MAX_SIMDGROUPS", "REDUCE_MAX_SIMDGROUPS", 1),
                                ("PREFIX_ATTN_HEAD_DIM", "PREFIX_ATTN_D", 1),
                                ("PREFIX_ATTN_LANES", "PREFIX_ATTN_R", 1),
                                ("PREFIX_ATTN_SIMDGROUPS", "PREFIX_ATTN_SGT", 1),
                                ("PREFIX_DECODE_CHUNK", "PREFIX_DECODE_CHUNK", 1),
                                ("PREFIX_DECODE_LANES", "PREFIX_DECODE_R", 1)]:
        ok = consts[host] == kc[kernel] * scale
        print(f"  [{'ok  ' if ok else 'FAIL'}] {host} = {consts[host]} vs kernel {kernel} x{scale} = {kc[kernel] * scale}")
        if not ok:
            FAILURES.append(f"constant {host}")


CASES = [
    ("host_contract", case_host_contract),
    ("conv", lambda: [case_conv(2, 37, 100, 4, s, b, 1) for s, b in ((False, False), (True, False), (True, True))]),
    ("conv_short", lambda: case_conv(3, 2, 64, 4, True, False, 2)),  # T < KW-1: state carries old entries
    ("chunk_ws", lambda: case_chunk_ws(3)),
    ("chunk_T1", lambda: case_gdn("qwen35_gdn_chunk", 1, 1, 1, 1, 32, 4)),
    ("chunk_T64", lambda: case_gdn("qwen35_gdn_chunk", 1, 64, 1, 1, 64, 5)),
    ("chunk_T65_gqa", lambda: case_gdn("qwen35_gdn_chunk", 2, 65, 1, 2, 64, 6)),
    ("chunk_T130_state", lambda: case_gdn("qwen35_gdn_chunk", 2, 130, 2, 4, 32, 7, state_mode="batch")),
    ("chunk_T100_snapshot", lambda: case_gdn("qwen35_gdn_chunk", 3, 100, 1, 1, 128, 8, state_mode="snapshot")),
    ("chunk_strong_decay", lambda: case_gdn("qwen35_gdn_chunk", 1, 150, 1, 1, 32, 9, a_scale=4.0,
                                            a_log_lo=1.0, a_log_hi=2.5)),
    ("chunk_T200", lambda: case_gdn("qwen35_gdn_chunk", 1, 200, 1, 1, 128, 10)),
    ("chunk_softplus_series", lambda: case_gdn("qwen35_gdn_chunk", 1, 130, 1, 2, 32, 19, dt_shift=-10.0)),
    ("recurrent_softplus_series", lambda: case_gdn("qwen35_gdn_recurrent", 1, 9, 1, 2, 32, 20, dt_shift=-10.0)),
    ("chunk_T1000", lambda: case_gdn("qwen35_gdn_chunk", 1, 1000, 1, 1, 32, 21)),
    # Long context: 64 chunks carried through one state, against transformers.
    ("chunk_T4096", lambda: case_gdn("qwen35_gdn_chunk", 1, 4096, 1, 1, 32, 30)),
    ("chunk_T0_passthrough", lambda: case_gdn("qwen35_gdn_chunk", 2, 0, 1, 2, 32, 22, state_mode="batch")),
    ("recurrent_T0_passthrough", lambda: case_gdn("qwen35_gdn_recurrent", 2, 0, 1, 2, 32, 23, state_mode="snapshot")),
    ("conv_T0_passthrough", lambda: case_conv(2, 0, 64, 4, True, True, 24)),
    # transformers' Qwen3_5TextConfig() defaults: 16 key heads shared by 32
    # value heads of 128 in the GDN, 16 query heads over 4 KV heads of 256 in
    # attention. These are NOT the 2B's (see config_2b_* below); they are kept
    # because Hv = 2*Hk exercises grouped value heads. Every other case uses a
    # handful of heads.
    ("config_chunk", lambda: case_gdn("qwen35_gdn_chunk", 1, 130, 16, 32, 128, 25, state_mode="batch")),
    ("config_recurrent", lambda: case_gdn("qwen35_gdn_recurrent", 2, 2, 16, 32, 128, 26, state_mode="snapshot")),
    ("config_gated_norm", lambda: case_gated_norm(False, 27, H=32, rows=9)),
    ("config_qk_norm_rope", lambda: case_qk_norm_rope(28, Hq=16, Hkv=4, B=1, T=5)),
    # Qwen/Qwen3.5-2B-Base's config.json: linear_num_key_heads 16,
    # linear_num_value_heads 16 (no value-head grouping), 8 query over 2 KV
    # attention heads of 256.
    ("config_2b_chunk", lambda: case_gdn("qwen35_gdn_chunk", 1, 130, 16, 16, 128, 34, state_mode="batch")),
    ("config_2b_recurrent", lambda: case_gdn("qwen35_gdn_recurrent", 2, 2, 16, 16, 128, 35,
                                             state_mode="snapshot")),
    ("config_2b_qk_norm_rope", lambda: case_qk_norm_rope(36, Hq=8, Hkv=2, B=1, T=5)),
    ("recurrent_T1_snapshot", lambda: case_gdn("qwen35_gdn_recurrent", 4, 1, 2, 4, 64, 11, state_mode="snapshot")),
    ("recurrent_T7_state", lambda: case_gdn("qwen35_gdn_recurrent", 2, 7, 1, 2, 128, 12, state_mode="batch")),
    ("recurrent_T20", lambda: case_gdn("qwen35_gdn_recurrent", 1, 20, 1, 1, 32, 13)),
    ("recurrent_in_place", lambda: case_recurrent_in_place(14)),
    ("gated_norm", lambda: [case_gated_norm(bf, 15) for bf in (False, True)]),
    ("qk_norm_rope", lambda: case_qk_norm_rope(16)),
    ("qk_rope_posbuf", lambda: case_qk_rope_posbuf(29)),
    ("qk_rope_slot_base", lambda: case_qk_rope_slot_base(31)),
    # Shared-prefix attention at Qwen3.5's full-attention heads (8 query over 2
    # KV heads of 256): no prefix, a one-token prefix, a prefix one past the
    # 64-row query tile with a partial second tile of queries, and queries
    # inside the prefix with no suffix.
    # Split-KV decode: 128-key chunks, so a prefix inside the first chunk, one
    # at a chunk edge with the suffix starting a new chunk, and a chunk that
    # straddles the prefix/suffix boundary.
    ("prefix_decode", lambda: [case_prefix_decode(40 + i, *c) for i, c in enumerate(
        [(2, 5, 2, 3, 6), (2, 128, 1, 2, 128), (3, 120, 20, 24, 139)])]),
    ("gdn_varlen", lambda: [case_gdn_varlen("qwen35_gdn_chunk", 70, 130, [0, 1, 63, 64, 65, 130, 150]),
                            case_gdn_varlen("qwen35_gdn_recurrent", 71, 7, [0, 1, 4, 7, 9])]),
    ("conv_varlen", lambda: case_conv_varlen(72)),
    ("prefix_varlen", lambda: [case_prefix_varlen(60, False), case_prefix_varlen(61, True)]),
    ("prefix_rows", lambda: [case_prefix_rows(32 + i, *c) for i, c in enumerate(
        [(2, 0, 3, 4, 3, 0), (3, 1, 2, 2, 2, 1), (2, 65, 66, 66, 66, 65), (2, 30, 0, 1, 2, 28)])]),
    ("attn_gate", lambda: [case_attn_gate(bf, ip, 17) for bf, ip in ((False, False), (True, False), (False, True))]),
    ("embed_rows", lambda: case_embed_rows(50)),
    ("score", lambda: [case_score(bf, 18) for bf in (False, True)]),
]


def main():
    global HARNESS
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="run only cases whose name contains one of these (comma-separated)")
    ap.add_argument("--fast-math", action="store_true",
                    help="perturb fast-math transcendentals by 4 ulps; use the on-device bounds")
    args = ap.parse_args()
    global FAST_MATH
    if args.fast_math:
        FAST_MATH = True
        os.environ.setdefault("MSL_EMU_ULP", "4")
    torch.set_num_threads(1)
    HARNESS = build()
    selected = [(name, fn) for name, fn in CASES if any(k in name for k in args.k.split(","))]
    if not selected:
        # A filter that matches nothing must not report "all checks passed".
        print(f"no case matches -k {args.k!r}; cases: {', '.join(n for n, _ in CASES)}")
        sys.exit(2)
    for name, fn in selected:
        print(f"{name}:")
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
