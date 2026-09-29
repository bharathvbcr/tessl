#!/usr/bin/env python3
"""Generate the Qwen3.5 golden fixtures in tests/fixtures/qwen35/ from transformers.

    python3 scripts/gen_qwen35_fixtures.py

Needs torch, numpy and a transformers that ships `models.qwen3_5`. The goldens
come from the model code itself — `torch_chunk_gated_delta_rule`,
`Qwen3_5RMSNorm`, `Qwen3_5TextRotaryEmbedding`, `apply_rotary_pos_emb` — so
`tests/qwen35_kernels.rs` can hold its Rust f64 references to transformers on a
Mac with no Python installed. Deterministic: rerunning rewrites identical files.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "qwen35")
DK = 128


def save(name, t):
    t = t.detach()
    arr = t.numpy().astype(np.float64 if t.dtype == torch.float64 else np.float32)
    np.save(os.path.join(OUT, f"qwen35_{name}.npy"), arr)


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
    g = torch.Generator().manual_seed(35)
    B, T, Hk, Hv, Dv = 2, 70, 1, 2, 32
    q = torch.randn(B, T, Hk, DK, generator=g)
    k = torch.randn(B, T, Hk, DK, generator=g)
    v = torch.randn(B, T, Hv, Dv, generator=g)
    a = torch.randn(B, T, Hv, generator=g)
    b = torch.randn(B, T, Hv, generator=g)
    a_log = torch.empty(Hv).uniform_(-2.0, 1.0, generator=g)
    dt_bias = torch.randn(Hv, generator=g)
    state0 = torch.randn(B, Hv, DK, Dv, generator=g) * 0.1
    for name, t in dict(q=q, k=k, v=v, a=a, b=b, a_log=a_log, dt_bias=dt_bias, state0=state0).items():
        save(f"gdn_{name}", t)

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


def rope():
    """Zero-centred Q/K norm + partial RoPE at a large position, D=256, rotary 64."""
    g = torch.Generator().manual_seed(36)
    T, Hq, Hkv, D, theta, eps, pos0 = 5, 2, 1, 256, 1e7, 1e-6, 20000
    cfg = Qwen3_5TextConfig(head_dim=D, num_attention_heads=Hq, num_key_value_heads=Hkv, hidden_size=Hq * D,
                            rope_parameters={"rope_type": "default", "rope_theta": theta,
                                             "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                             "mrope_interleaved": True})
    rot = m.Qwen3_5TextRotaryEmbedding(cfg)
    q = torch.randn(1, T, Hq, D, generator=g)
    k = torch.randn(1, T, Hkv, D, generator=g)
    qw = torch.randn(D, generator=g) * 0.1
    kw = torch.randn(D, generator=g) * 0.1
    save("rope_q_in", q)
    save("rope_k_in", k)
    save("rope_q_norm_w", qw)
    save("rope_k_norm_w", kw)
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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    gdn()
    rope()
    print("wrote", sorted(os.listdir(OUT)))
