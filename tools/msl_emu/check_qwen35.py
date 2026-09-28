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


def build():
    out = os.environ.get("MSL_EMU_OUT", os.path.join(HERE, "build"))
    env = dict(os.environ, MSL_EMU_OUT=out)
    subprocess.run([os.path.join(HERE, "build.sh")], check=True, env=env, stdout=subprocess.DEVNULL)
    return os.path.join(out, "harness")


HARNESS = None


def run(kernel, params, inputs, outputs):
    """Write inputs, run the harness, return outputs as flat tensors."""
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
        subprocess.run([HARNESS, d], check=True)
        res = {}
        for name, (dtype, n) in outputs.items():
            raw = open(os.path.join(d, name + ".bin"), "rb").read()
            if dtype == "bf16":
                res[name] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
            else:
                res[name] = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        return res


def check(name, got, want, atol, rtol=0.0):
    got = got.double().flatten()
    want = want.double().flatten()
    assert got.numel() == want.numel(), f"{name}: {got.numel()} vs {want.numel()} elements"
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
    ref = m.causal_conv1d_update(x.clone(), ref_state, w, None, "silu")  # updates ref_state in place
    if not with_state:
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


def gdn_layout(B, T, Hk, Hv, Dv, seed, a_scale=1.0, a_log_lo=-2.0, a_log_hi=1.0):
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
    dt_bias = torch.randn(Hv, generator=g)
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
    if kernel == "qwen35_gdn_chunk":
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
    hf_err = (hf.double().reshape(B * T, Hv * Dv) - want).abs().max().item()
    hf_state_err = (hf_state.double() - ref_state).abs().max().item()
    scale = max(want.abs().max().item(), 1e-3)
    print(f"  {tag}: transformers fp32 err vs f64 {hf_err:.2e} (out), {hf_state_err:.2e} (state); |y|max {scale:.2e}")
    # The bound: 8x what transformers' own fp32 run misses by, floored at a few
    # ulps of the output scale for cases where the fp32 reference is exact.
    atol = max(8 * hf_err, 64 * 2 ** -24 * scale)
    check(tag + " out", got_heads, want, atol)
    satol = max(8 * hf_state_err, 64 * 2 ** -24 * max(ref_state.abs().max().item(), 1e-3))
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


def case_gated_norm(bf16, seed):
    g = seeded(seed)
    rows, H, D = 37, 4, 128
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


def case_qk_norm_rope(seed):
    g = seeded(seed)
    B, T, Hq, Hkv, D = 2, 9, 4, 2, 256
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
    ok = bool(torch.isnan(lg[0]).all() and torch.isnan(lg[1, 1]) and not torch.isnan(lg[1, 0]))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {kname} out-of-range slot/answer -> NaN")
    if not ok:
        FAILURES.append(kname + " out-of-range")


CASES = [
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
    ("recurrent_T1_snapshot", lambda: case_gdn("qwen35_gdn_recurrent", 4, 1, 2, 4, 64, 11, state_mode="snapshot")),
    ("recurrent_T7_state", lambda: case_gdn("qwen35_gdn_recurrent", 2, 7, 1, 2, 128, 12, state_mode="batch")),
    ("recurrent_T20", lambda: case_gdn("qwen35_gdn_recurrent", 1, 20, 1, 1, 32, 13)),
    ("recurrent_in_place", lambda: case_recurrent_in_place(14)),
    ("gated_norm", lambda: [case_gated_norm(bf, 15) for bf in (False, True)]),
    ("qk_norm_rope", lambda: case_qk_norm_rope(16)),
    ("attn_gate", lambda: [case_attn_gate(bf, ip, 17) for bf, ip in ((False, False), (True, False), (False, True))]),
    ("score", lambda: [case_score(bf, 18) for bf in (False, True)]),
]


def main():
    global HARNESS
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="run only cases whose name contains this")
    args = ap.parse_args()
    torch.set_num_threads(1)
    HARNESS = build()
    for name, fn in CASES:
        if args.k in name:
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
