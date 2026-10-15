#!/usr/bin/env python3
"""Generate the Qwen3.5 golden fixtures in tests/fixtures/qwen35/ from transformers.

    python3 scripts/gen_qwen35_fixtures.py                    # recompute the goldens
    python3 scripts/gen_qwen35_fixtures.py --init-gate-inputs # once: write the gate inputs

Needs torch, numpy and a transformers that ships `models.qwen3_5`; the pinned
oracle environment is ~/.venvs/rsijev-oracle (torch 2.14.1, transformers 5.17.0).
The goldens come from the model code itself -- `torch_chunk_gated_delta_rule`,
`Qwen3_5GatedDeltaNet.forward`, `Qwen3_5RMSNorm`, `Qwen3_5TextRotaryEmbedding`,
`apply_rotary_pos_emb` -- so `tests/qwen35_kernels.rs` can hold its Rust f64
references to transformers on a Mac with no Python installed.

**Inputs are stored, goldens are recomputed.** Every input array (`role: input` in
MANIFEST.json) is read from the committed .npy, never drawn from
`torch.randn(generator=)`: a torch generator's stream is not stable across torch
builds (a rerun on another build moved 16 of 18 files by an ulp-scale input
change), and a golden that moves with the generator's RNG is not a golden. What
this script rewrites is the goldens, which move only by the arithmetic of the
installed torch (measured: at most ~7.8e-7 relative to the file's peak across
builds). The goldens are therefore reproducible to ~1e-6 across torch versions,
not bit-for-bit. The gate inputs are drawn
once, by `--init-gate-inputs`, from numpy's `default_rng`, and stored like the rest.

The fixtures of the other generators are `tools/qwen35_ref/make_train_fixture.py`'s.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "qwen35_ref"))
import fixture_manifest as fm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "qwen35")
DK = 128
GENERATOR = "scripts/gen_qwen35_fixtures.py"

# The gate fixture's shape: the 4B's 32 value heads, B x T = 2 x 37 rows (a count
# that divides by nothing the kernel launches in), and extreme logits mixed in.
GATE_B, GATE_T, GATE_H = 2, 37, 32
GATE_SEED = 20261015


def path(name):
    return os.path.join(OUT, f"qwen35_{name}.npy")


def save(name, t):
    t = t.detach()
    arr = t.numpy().astype(np.float64 if t.dtype == torch.float64 else np.float32)
    np.save(path(name), arr)


def load(name, dtype=torch.float32):
    """A stored input array, exactly as committed."""
    return torch.from_numpy(np.load(path(name))).to(dtype)


def gdn_f64(q, k, v, g, beta, state):
    """Sequential gated delta rule in f64, the recurrent form."""
    q = m.l2norm(q, dim=-1, eps=1e-6) * DK ** -0.5
    k = m.l2norm(k, dim=-1, eps=1e-6)
    S = state.clone()
    out = torch.zeros_like(v)
    for t in range(q.shape[1]):
        S = S * g[:, t].exp()[:, :, None, None]
        kv = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        out[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t])
    return out, S


def gdn():
    """B=2, T=70 (a full chunk plus a ragged one), 1 key head shared by 2 value heads."""
    q, k, v = load("gdn_q"), load("gdn_k"), load("gdn_v")
    a, b = load("gdn_a"), load("gdn_b")
    a_log, dt_bias, state0 = load("gdn_a_log"), load("gdn_dt_bias"), load("gdn_state0")
    Hk, Hv = q.shape[2], v.shape[2]

    def gates(dtype):
        # Qwen3_5GatedDeltaNet.forward, verbatim.
        beta = b.to(dtype).sigmoid()
        gg = -a_log.to(dtype).exp() * F.softplus(a.to(dtype) + dt_bias.to(dtype))
        return gg, beta

    def expand(x, dtype):
        x = x.to(dtype)
        return x.repeat_interleave(Hv // Hk, dim=2) if Hv // Hk > 1 else x

    gg, beta = gates(torch.float32)
    y, s = m.torch_chunk_gated_delta_rule(expand(q, torch.float32), expand(k, torch.float32), v, gg, beta,
                                          initial_state=state0, output_final_state=True,
                                          use_qk_l2norm_in_kernel=True)
    save("gdn_y_hf", y)
    save("gdn_state_hf", s)
    gg, beta = gates(torch.float64)
    y, s = gdn_f64(expand(q, torch.float64), expand(k, torch.float64), v.double(), gg, beta, state0.double())
    save("gdn_y_f64", y)
    save("gdn_state_f64", s)


def init_gate_inputs():
    """Draw the gate inputs once, from numpy's `default_rng`, and store them. Logits
    spread over several units, with the values the kernel's branches turn on mixed
    in: a + dt_bias past softplus' linear threshold (20) either side, b at +-20."""
    rng = np.random.default_rng(GATE_SEED)
    a = (rng.standard_normal((GATE_B, GATE_T, GATE_H)) * 3.0).astype(np.float32)
    b = (rng.standard_normal((GATE_B, GATE_T, GATE_H)) * 3.0).astype(np.float32)
    a_log = rng.uniform(-2.0, 1.0, GATE_H).astype(np.float32)
    dt_bias = rng.standard_normal(GATE_H).astype(np.float32)
    a[0, 0, :4] = [-30.0, 30.0, 25.0, -25.0]
    b[0, 0, :4] = [-20.0, 20.0, 0.0, -0.0]
    a[-1, -1, :2] = -dt_bias[:2]  # a + dt_bias == 0 exactly: softplus(0) = ln 2
    for name, arr in dict(a=a, b=b, a_log=a_log, dt_bias=dt_bias).items():
        np.save(path(f"gates_{name}"), arr)


def gates():
    """The `qwen35_gdn_gates_f32` golden, from `Qwen3_5GatedDeltaNet.forward` itself.

    transformers computes the gates inline (`beta = b.sigmoid()`,
    `g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)`), with no function
    to call, so the module's own forward is run and the `g` and `beta` it hands to
    the delta rule are captured. `in_proj_a` / `in_proj_b` are set to select the two
    halves of the input, which makes the module's `a` and `b` exactly the stored
    ones; forward hooks read back what the module computed and the script refuses
    to continue unless they equal the stored arrays bit for bit.
    """
    a, b = load("gates_a"), load("gates_b")
    a_log, dt_bias = load("gates_a_log"), load("gates_dt_bias")
    H = a.shape[-1]
    cfg = Qwen3_5TextConfig(hidden_size=2 * H, linear_num_value_heads=H, linear_num_key_heads=16,
                            linear_key_head_dim=DK, linear_value_head_dim=DK, num_hidden_layers=1,
                            layer_types=["linear_attention"])
    mod = m.Qwen3_5GatedDeltaNet(cfg, 0)
    eye, zero = torch.eye(H), torch.zeros(H, H)
    seen, hooks = {}, []
    real = m.torch_chunk_gated_delta_rule

    def capture(query, key, value, g=None, beta=None, **kw):
        seen["g"], seen["beta"] = g.detach().clone(), beta.detach().clone()
        return real(query, key, value, g=g, beta=beta, **kw)

    m.torch_chunk_gated_delta_rule = capture
    try:
        with torch.no_grad():
            mod.in_proj_a.weight.copy_(torch.cat([eye, zero], 1))
            mod.in_proj_b.weight.copy_(torch.cat([zero, eye], 1))
            mod.A_log.copy_(a_log)
            mod.dt_bias.copy_(dt_bias)
            hooks.append(mod.in_proj_a.register_forward_hook(lambda _m, _i, o: seen.__setitem__("a", o.clone())))
            hooks.append(mod.in_proj_b.register_forward_hook(lambda _m, _i, o: seen.__setitem__("b", o.clone())))
            mod(torch.cat([a, b], -1))
    finally:
        m.torch_chunk_gated_delta_rule = real
        for h in hooks:
            h.remove()
    if not (torch.equal(seen["a"], a) and torch.equal(seen["b"], b)):
        sys.exit("the module's a/b logits are not the stored inputs; the identity projection failed")
    save("gates_g", seen["g"])
    save("gates_beta", seen["beta"])


def rope():
    """Zero-centred Q/K norm + partial RoPE at a large position, D=256, rotary 64."""
    T, Hq, Hkv, D, theta, eps, pos0 = 5, 2, 1, 256, 1e7, 1e-6, 20000
    cfg = Qwen3_5TextConfig(head_dim=D, num_attention_heads=Hq, num_key_value_heads=Hkv, hidden_size=Hq * D,
                            rope_parameters={"rope_type": "default", "rope_theta": theta,
                                             "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                             "mrope_interleaved": True})
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    q, k = load("rope_q_in"), load("rope_k_in")
    qw, kw = load("rope_q_norm_w"), load("rope_k_norm_w")
    qn, kn = m.Qwen3_5RMSNorm(D, eps=eps), m.Qwen3_5RMSNorm(D, eps=eps)
    with torch.no_grad():
        qn.weight.copy_(qw)
        kn.weight.copy_(kw)
        qs, ks = qn(q).transpose(1, 2), kn(k).transpose(1, 2)
        pos = (torch.arange(T) + pos0)[None, None, :].expand(3, 1, T)
        cos, sin = rot(qs, pos)
        qs, ks = m.apply_rotary_pos_emb(qs, ks, cos, sin)
    save("rope_q_out", qs.transpose(1, 2))
    save("rope_k_out", ks.transpose(1, 2))


# role, published, note -- per file, in the manifest. `published` is true for a golden of
# record or an input to one; false for files kept only for the tests that already read
# them (see tools/qwen35_ref/fixture_manifest.py).
LEGACY = ("kept for tests/qwen35_kernels.rs's gdn_* tests and tools/msl_emu/check_qwen35.py, which read it "
          "by path; the operator is transformers' Qwen3.5 form with the weights drawn here, and nothing new "
          "uses it as a golden (the model-level golden is tests/fixtures/qwen35_train*)")


def file_table():
    files = {"config_qwen35_2b_base.json": dict(
        role="config", published=True,
        note="Qwen3.5-2B-Base's real text config, the shape source for tests that need the 2B's dimensions")}
    for n in ("q", "k", "v", "a", "b", "a_log", "dt_bias", "state0"):
        files[f"qwen35_gdn_{n}.npy"] = dict(role="input", published=False, note=LEGACY)
    for n in ("y_hf", "state_hf", "y_f64", "state_f64"):
        files[f"qwen35_gdn_{n}.npy"] = dict(role="golden", published=False, note=LEGACY)
    for n in ("q_in", "k_in", "q_norm_w", "k_norm_w"):
        files[f"qwen35_rope_{n}.npy"] = dict(role="input", published=True, note="stored input")
    for n in ("q_out", "k_out"):
        files[f"qwen35_rope_{n}.npy"] = dict(role="golden", published=True, note="Qwen3_5RMSNorm + partial RoPE")
    for n in ("a", "b", "a_log", "dt_bias"):
        files[f"qwen35_gates_{n}.npy"] = dict(role="input", published=True,
                                              note="numpy default_rng stream, drawn once by --init-gate-inputs")
    for n in ("g", "beta"):
        files[f"qwen35_gates_{n}.npy"] = dict(
            role="golden", published=True,
            note="captured from Qwen3_5GatedDeltaNet.forward (g = -exp(A_log) * softplus(a + dt_bias); "
                 "beta = sigmoid(b)): the golden of qwen35_gdn_gates_f32")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-gate-inputs", action="store_true",
                    help="write the stored gate inputs (once) and stop; they are inputs, not goldens")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.init_gate_inputs:
        init_gate_inputs()
        print("wrote the gate inputs")
        return
    before = tempfile.mkdtemp(prefix="qwen35_before_")
    try:
        for n in os.listdir(OUT):
            if n.endswith(".npy"):
                shutil.copy(os.path.join(OUT, n), before)
        gdn()
        rope()
        gates()
        moved = fm.compare(before, OUT)["max_relative_to_peak"]
        prior = {}
        manifest_path = os.path.join(OUT, "MANIFEST.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                prior = json.load(f).get("regeneration", {})
        # A run that moves nothing must not erase the record of the run that did.
        regeneration = {"max_relative_to_peak_vs_previous_build": moved} if moved else prior
        fm.write_manifest(
            OUT, GENERATOR,
            "stored: every role=input file is the committed array and is never redrawn. The gdn and rope "
            "inputs are the original torch.randn(generator=) draws, extracted from the first committed "
            "fixtures; the gate inputs are numpy default_rng draws. Only the goldens are recomputed.",
            file_table(), "produced", extra={"regeneration": regeneration} if regeneration else None)
    finally:
        shutil.rmtree(before)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
