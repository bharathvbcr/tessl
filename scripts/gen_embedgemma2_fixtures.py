#!/usr/bin/env python3
"""Generate the EmbeddingGemma 2 golden fixtures in tests/fixtures/embedgemma2/ from transformers.

    ~/.venvs/ml/bin/python scripts/gen_embedgemma2_fixtures.py

Needs torch, numpy and a transformers that ships `models.embedding_gemma2`
(5.19.0 or later). The goldens come from the model code itself —
`EmbeddingGemma2Attention`'s pieces (`eager_attention_forward`,
`create_bidirectional_sliding_window_mask`, `create_bidirectional_mask`),
`EmbeddingGemma2RMSNorm`, `EmbeddingGemma2RotaryEmbedding`,
`apply_rotary_pos_emb`, `EmbeddingGemma2TextPLE` and
`EmbeddingGemma2TextPLEBlock` — so `tests/embedgemma2_kernels.rs` can hold its
Rust f64 references to transformers on a Mac with no Python installed.
Deterministic: rerunning rewrites identical files.

Shapes are the model's own head dims (256 sliding, 512 global) at small T, so
the kernels are exercised at their real instantiations. The sliding window is
shrunk to 7 so that T = 24 crosses it in both directions; the Rust reference,
once held to these files, checks the kernels at the real window of 512.
"""

import os

import numpy as np
import torch
from transformers import masking_utils
from transformers.models.embedding_gemma2 import modeling_embedding_gemma2 as m
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "embedgemma2")
EPS = 1e-6


def save(name, t):
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.dtype in (torch.int32, torch.int64, torch.bool):
            arr = t.to(torch.int64).numpy()
        else:
            arr = t.to(torch.float64 if t.dtype == torch.float64 else torch.float32).numpy()
    else:
        arr = np.asarray(t)
    np.save(os.path.join(OUT, f"eg2_{name}.npy"), np.ascontiguousarray(arr))


# The checkpoint's own text_config (google/embeddinggemma-2, config.json), so
# no shape here rests on a constructor default.
REAL = {
    "hidden_size": 512, "hidden_size_per_layer_input": 512, "intermediate_size": 2048,
    "num_hidden_layers": 24, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
    "embedding_dim": 768, "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": EPS,
    "sliding_window": 512, "vocab_size": 262144, "max_position_embeddings": 262144,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 4,
    "per_layer_config": {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 1} for i in (5, 11, 17, 23)},
    "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
}


def text_config(**kw):
    cfg = EmbeddingGemma2TextConfig(**{**REAL, **kw})
    cfg._attn_implementation = "eager"
    return cfg


class _Attn:
    """The two attributes `eager_attention_forward` reads from its module."""

    def __init__(self, hq, hkv, d):
        self.num_key_value_groups = hq // hkv
        self.head_dim = d
        self.training = False


def attention(name, *, hq, hkv, d, window, lens, seed):
    """Bidirectional attention, scale 1.0, keys masked past each row's length.

    window = None is a full-attention layer (padding mask only); otherwise the
    symmetric inclusive window |i - j| <= window. q/k/v are [B, T, H, D], the
    output [B, T, Hq, D]. Rows past a sequence's length are padding: their
    outputs are written but meaningless, so the Rust side compares only
    t < len.
    """
    g = torch.Generator().manual_seed(seed)
    B, T = len(lens), max(lens)
    q = torch.randn(B, T, hq, d, generator=g)
    k = torch.randn(B, T, hkv, d, generator=g)
    v = torch.randn(B, T, hkv, d, generator=g)
    # Scale 1.0 on unit-variance D=256 vectors would put every softmax at one
    # key; the model's q/k are RMS-normed with learned weights, so scale the
    # inputs to keep the softmax spread like the real one.
    q = q * d ** -0.5
    attn_mask = torch.zeros(B, T, dtype=torch.long)
    for b, n in enumerate(lens):
        attn_mask[b, :n] = 1
    cfg = text_config(sliding_window=window if window is not None else 512)
    embeds = torch.zeros(B, T, 8)
    if window is None:
        mask = masking_utils.create_bidirectional_mask(config=cfg, inputs_embeds=embeds, attention_mask=attn_mask)
    else:
        mask = masking_utils.create_bidirectional_sliding_window_mask(
            config=cfg, inputs_embeds=embeds, attention_mask=attn_mask)
    out, _ = m.eager_attention_forward(_Attn(hq, hkv, d), q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2), mask, dropout=0.0, scaling=1.0)
    for n, t in dict(q=q, k=k, v=v, out=out, lens=torch.tensor(lens)).items():
        save(f"{name}_{n}", t)
    save(f"{name}_window", np.array([-1 if window is None else window], dtype=np.int64))


def qkv_norm_rope(name, *, hq, hkv, d, theta, pos0, seed):
    """q/k: weighted RMSNorm over head_dim (times w), v: weightless RMSNorm,
    then full-width rotate_half RoPE on q and k at positions pos0 + t."""
    g = torch.Generator().manual_seed(seed)
    T = 6
    q = torch.randn(1, T, hq, d, generator=g) * 3.0
    k = torch.randn(1, T, hkv, d, generator=g) * 3.0
    v = torch.randn(1, T, hkv, d, generator=g) * 3.0
    qw = 1.0 + torch.randn(d, generator=g) * 0.2
    kw = 1.0 + torch.randn(d, generator=g) * 0.2
    qn, kn = m.EmbeddingGemma2RMSNorm(d, eps=EPS), m.EmbeddingGemma2RMSNorm(d, eps=EPS)
    vn = m.EmbeddingGemma2RMSNorm(d, eps=EPS, with_scale=False)
    with torch.no_grad():
        qn.weight.copy_(qw)
        kn.weight.copy_(kw)
        qs, ks, vs = qn(q).transpose(1, 2), kn(k).transpose(1, 2), vn(v).transpose(1, 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        pos = (torch.arange(T) + pos0).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        qs, ks = m.apply_rotary_pos_emb(qs, ks, cos, sin)
    for n, t in dict(q_in=q, k_in=k, v_in=v, q_norm_w=qw, k_norm_w=kw,
                     q_out=qs.transpose(1, 2), k_out=ks.transpose(1, 2), v_out=vs.transpose(1, 2)).items():
        save(f"{name}_{n}", t)
    save(f"{name}_meta", np.array([pos0], dtype=np.int64))


def rope_matches_model():
    """The hand-built cos/sin above must be what the model's rotary module
    computes, per layer type (theta 1e4 at D=256, 1e6 at D=512)."""
    cfg = text_config()
    rot = m.EmbeddingGemma2RotaryEmbedding(cfg)
    x = torch.zeros(1, 4, 8)
    pos = torch.arange(4)[None] + 8000
    for layer_type, d, theta in (("sliding_attention", 256, 1e4), ("full_attention", 512, 1e6)):
        cos, sin = rot(x, pos, layer_type)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        freqs = torch.outer(pos[0].float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        assert cos.shape[-1] == d, (layer_type, cos.shape)
        assert torch.allclose(cos[0], emb.cos(), atol=1e-6) and torch.allclose(sin[0], emb.sin(), atol=1e-6), layer_type


def ple(seed):
    """Projection-only per-layer inputs and one PLE block, at the model's widths.

    Two layers at width 64 instead of 24 at 512: the projection is one GEMM
    whose output columns split per layer and every op here is width-generic,
    so this pins the layout and the arithmetic at a fixture size the repo can
    carry (the real [24*512, 512] weight alone would be 25 MB of f32)."""
    g = torch.Generator().manual_seed(seed)
    cfg = text_config(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"], per_layer_config={},
                      hidden_size=64, hidden_size_per_layer_input=64)
    B, T, H, L = 2, 5, cfg.hidden_size, cfg.num_hidden_layers
    embeds = torch.randn(B, T, H, generator=g) * 4.0
    hidden = torch.randn(B, T, H, generator=g)
    proj = m.EmbeddingGemma2TextPLE(cfg)
    block = m.EmbeddingGemma2TextPLEBlock(cfg)
    with torch.no_grad():
        proj.per_layer_model_projection.weight.copy_(torch.randn(L * H, H, generator=g) * H ** -0.5)
        proj.per_layer_projection_norm.weight.copy_(1.0 + torch.randn(H, generator=g) * 0.2)
        block.per_layer_input_gate.weight.copy_(torch.randn(H, H, generator=g) * H ** -0.5)
        block.per_layer_projection.weight.copy_(torch.randn(H, H, generator=g) * H ** -0.5)
        block.post_per_layer_input_norm.weight.copy_(1.0 + torch.randn(H, generator=g) * 0.2)
        per_layer = proj(embeds)
        layer = 1
        out = block(hidden, per_layer[:, :, layer, :])
    for n, t in dict(embeds=embeds, hidden=hidden,
                     proj_w=proj.per_layer_model_projection.weight,
                     proj_norm_w=proj.per_layer_projection_norm.weight,
                     gate_w=block.per_layer_input_gate.weight,
                     out_proj_w=block.per_layer_projection.weight,
                     post_norm_w=block.post_per_layer_input_norm.weight,
                     per_layer=per_layer, block_out=out).items():
        save(f"ple_{n}", t)
    save("ple_layer", np.array([layer], dtype=np.int64))


def pool(seed):
    """Masked mean over valid tokens, then L2 normalize (sentence-transformers
    Pooling(mean, include_prompt) + Normalize), and the Matryoshka prefixes."""
    g = torch.Generator().manual_seed(seed)
    lens = [9, 4, 1]
    B, T, D = len(lens), max(lens), 768
    x = torch.randn(B, T, D, generator=g)
    mask = torch.zeros(B, T)
    for b, n in enumerate(lens):
        mask[b, :n] = 1
    summed = (x * mask[..., None]).sum(1)
    mean = summed / mask.sum(1, keepdim=True).clamp(min=1e-9)
    out = torch.nn.functional.normalize(mean, p=2, dim=-1)
    save("pool_x", x)
    save("pool_lens", torch.tensor(lens))
    save("pool_out", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    rope_matches_model()
    attention("swa", hq=2, hkv=1, d=256, window=7, lens=[24, 17], seed=201)
    attention("global", hq=2, hkv=1, d=512, window=None, lens=[24, 17], seed=202)
    attention("swa_tiny", hq=2, hkv=1, d=256, window=7, lens=[1, 3], seed=203)
    qkv_norm_rope("qkv256", hq=2, hkv=1, d=256, theta=1e4, pos0=3, seed=204)
    qkv_norm_rope("qkv512", hq=2, hkv=1, d=512, theta=1e6, pos0=3, seed=205)
    # Far from 0, transformers' own f32 angles (pos * inv_freq in f32) carry
    # ~pos * 2^-24 rad of rounding, so this one is held to that, not to f64.
    qkv_norm_rope("qkv256_far", hq=2, hkv=1, d=256, theta=1e4, pos0=8000, seed=208)
    ple(seed=206)
    pool(seed=207)
    print("wrote", sorted(os.listdir(OUT)))
