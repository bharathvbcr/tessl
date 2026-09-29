#!/usr/bin/env python3
"""The Qwen3.5 kernels against the *model*, not just its functions.

    python3 tools/msl_emu/check_qwen35_model.py

check_qwen35.py holds each kernel to the transformers function it replaces.
That leaves a gap: how `Qwen3_5GatedDeltaNet` and `Qwen3_5Attention` wire
those functions together (which columns of which projection feed which input,
how the conv weight is laid out, how heads group, where the gate sits, how a
cached decode continues a prefill). This builds a small random
`Qwen3_5ForCausalLM` and runs its whole forward pass with every Qwen3.5-specific
step on the emulated kernels, using the weight packing and column layouts
`src/qwen35.rs` defines, and the attention itself on tessl's existing
`flash_attn_rows` kernel, emulated too, as `nn::flash_attn_rows` dispatches it,
so the seam between the Qwen3.5 kernels and the attention they feed is tested.
torch does only what tessl's GEMM and existing norm/MLP kernels would: the
matrix products, the MLP and the pre-attention norms. The model's own logits
are the oracle.

Three flows:

- **prefill**: one sequence, logits for a set of answer tokens at slot rows;
- **decode**: prefill, then one token with the carried conv/GDN state and KV
  cache, against the model's own cached decode;
- **snapshot**: one prefilled prefix answers several questions at once through
  `StateIn::Snapshot`, against the model run separately on each full sequence;
- **shared_prefix**: the same, with the attention's KV prefix shared too
  (`qwen35_attn_prefix_rows`), then one decode step per question
  (`qwen35_attn_prefix_decode`).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_qwen35 as cq  # noqa: E402  (the harness driver and its helpers)
from transformers.models.qwen3_5 import modeling_qwen3_5 as m  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

DK = 128
HK, HV, DV = 1, 2, 32          # GDN heads: one key head shared by two value heads
HQ, HKV, D = 2, 1, 256         # attention heads; Qwen3.5's head_dim, rotary 64
HIDDEN, KW, EPS = 128, 4, 1e-6
VOCAB = 96
ANSWERS = torch.tensor([3, 17, 40, 41, 77], dtype=torch.int32)


def build_model(seed):
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=HQ, num_key_value_heads=HKV, head_dim=D, linear_key_head_dim=DK,
        linear_value_head_dim=DV, linear_num_key_heads=HK, linear_num_value_heads=HV,
        linear_conv_kernel_dim=KW, tie_word_embeddings=True, rms_norm_eps=EPS,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = m.Qwen3_5ForCausalLM(cfg).eval()
    # The default init (std 0.02, zero norm weights, dt_bias = 1) leaves most of
    # the arithmetic near zero, where a wrong column or a missing (1 + w) can
    # hide. Perturb everything into the regime a trained model lives in.
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 2 - 3)  # reaches the softplus series range
            elif name.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(-1.0, 2.0, generator=g))
            elif name.endswith("norm.weight") and ".linear_attn." in name:
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))       # RMSNormGated: * w
            elif p.dim() == 1:
                p.copy_(0.2 * torch.randn(p.shape, generator=g))           # zero-centred: * (1 + w)
            elif name.endswith("conv1d.weight"):
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5)
    return model


def pack(*linears):
    """`pack_linear_weights_f32`: [in, sum(out)], each part transposed."""
    return torch.cat([l.weight.detach().T for l in linears], dim=1).contiguous()


class Layout:
    """`GdnProjLayout` / `AttnProjLayout` from src/qwen35.rs."""
    key_dim, value_dim = HK * DK, HV * DV
    conv_dim = 2 * key_dim + value_dim
    z_off = conv_dim
    b_off = z_off + value_dim
    a_off = b_off + HV
    width = a_off + HV
    attn_k_off = 2 * HQ * D
    attn_v_off = attn_k_off + HKV * D
    attn_width = attn_v_off + HKV * D


def kernel(name, params, inputs, outputs):
    return cq.run(name, params, inputs, {k: ("f32", n) for k, n in outputs.items()})


# ------------------------------------------------------------------- layers


def gdn_layer(attn, h, B, T, conv_state=None, gdn_state=None, snapshot=False, path="chunk"):
    """One Qwen3_5GatedDeltaNet on the kernels. Returns (out, conv_state, gdn_state)."""
    L = Layout
    proj = (h.reshape(B * T, HIDDEN) @ pack(attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_b,
                                            attn.in_proj_a)).contiguous()           # the fused GEMM
    hist = KW - 1
    conv_in = {"x": proj, "w": attn.conv1d.weight.detach().squeeze(1).contiguous()}
    flags = 2
    if conv_state is not None:
        conv_in["state_in"] = conv_state
        flags |= 1
    c = kernel("qwen35_conv1d_silu",
               dict(B=B, T=T, C=L.conv_dim, KW=KW, ld_x=L.width, x_off=0,
                    state_bstride=0 if snapshot else L.conv_dim * hist, flags=flags),
               conv_in, {"y": B * T * L.conv_dim, "state_out": B * L.conv_dim * hist})
    g_in = {"qkv": c["y"], "ab": proj, "a_log": attn.A_log.detach(), "dt_bias": attn.dt_bias.detach()}
    flags = 2
    if gdn_state is not None:
        g_in["state_in"] = gdn_state
        flags |= 1
    o = kernel("qwen35_gdn_chunk" if path == "chunk" else "qwen35_gdn_recurrent",
               dict(B=B, T=T, Hk=HK, Hv=HV, Dv=DV, ld_qkv=L.conv_dim, q_off=0, k_off=L.key_dim,
                    v_off=2 * L.key_dim, ld_ab=L.width, a_off=L.a_off, b_off=L.b_off, ld_out=L.value_dim,
                    out_off=0, state_bstride=0 if snapshot else HV * DK * DV, flags=flags),
               g_in, {"out": B * T * L.value_dim, "state_out": B * HV * DK * DV})
    y = kernel("qwen35_gated_rms_norm_f32",
               dict(rows=B * T, H=HV, D=DV, ld_x=L.value_dim, x_off=0, ld_z=L.width, z_off=L.z_off,
                    ld_out=L.value_dim, out_off=0, eps=EPS),
               {"x": o["out"], "z": proj, "w": attn.norm.weight.detach()}, {"out": B * T * L.value_dim})
    out = y["out"].reshape(B * T, L.value_dim) @ attn.out_proj.weight.detach().T
    return out.reshape(B, T, HIDDEN), c["state_out"], o["state_out"]


def attn_layer(sa, h, B, T, pos, k_cache, v_cache, prefix=None):
    """One Qwen3_5Attention on the kernels, appending to `[B, cap, Hkv, D]` caches.

    With `prefix = (kp, vp, P)` the caches are suffix caches (slot s is position
    P + s, written with slot_base = P, as `attn_qk_norm_rope_suffix` does) and
    the attention reads the shared `[P, Hkv, D]` prefix at batch stride 0:
    `qwen35_attn_prefix_rows`, or `qwen35_attn_prefix_decode` for one token."""
    L = Layout
    cap = k_cache.shape[1]
    P = 0 if prefix is None else prefix[2]
    proj = (h.reshape(B * T, HIDDEN) @ pack(sa.q_proj, sa.k_proj, sa.v_proj)).contiguous()
    r = kernel("qwen35_attn_qk_norm_rope",
               dict(B=B, T=T, Hq=HQ, Hkv=HKV, D=D, rotary_dim=D // 4, ld_p=L.attn_width, q_off=0,
                    k_off=L.attn_k_off, v_off=L.attn_v_off, pos_offset=pos, kv_capacity=cap,
                    theta=1e7, eps=EPS, slot_base=P),
               {"p": proj, "q_norm_w": sa.q_norm.weight.detach(), "k_norm_w": sa.k_norm.weight.detach(),
                "k_cache": k_cache, "v_cache": v_cache},
               {"q_out": B * T * HQ * D, "k_cache": k_cache.numel(), "v_cache": v_cache.numel()})
    q = r["q_out"].reshape(B, T, HQ, D)
    kc = r["k_cache"].reshape(B, cap, HKV, D)
    vc = r["v_cache"].reshape(B, cap, HKV, D)
    n = pos + T
    q_pos = torch.tensor([pos], dtype=torch.int32)
    if prefix is None:
        # The attention: tessl's flash_attn_rows, reading q_out and the caches
        # exactly as attn_qk_norm_rope wrote them. Tkv is the live prefix, the
        # query rows sit at absolute positions pos.., the keys at 0.., scale
        # D^-0.5.
        name = "flash_attn_rows"
        fa = cq.run(name, dict(B=B, Tq=T, H=HQ, Hkv=HKV, D=D, kv_capacity=cap, scale=D ** -0.5),
                    {"q": q.contiguous(), "k": kc.contiguous(), "v": vc.contiguous(),
                     "tkv": torch.tensor([n], dtype=torch.int32), "q_pos": q_pos,
                     "kv_pos": torch.tensor([0], dtype=torch.int32)},
                    {"o": ("f32", B * T * HQ * D)})
        k_all, v_all = kc[:, :n], vc[:, :n]
    else:
        kp, vp, _ = prefix
        name = "qwen35_attn_prefix_decode" if T == 1 else "qwen35_attn_prefix_rows"
        fa = cq.run(name, dict(B=B, Tq=T, H=HQ, Hkv=HKV, P=P, suffix_cap=cap, scale=D ** -0.5),
                    {"q": q.contiguous(), "kp": kp, "vp": vp, "ks": kc.contiguous(), "vs": vc.contiguous(),
                     "suffix_len": torch.tensor([n - P], dtype=torch.int32), "q_pos": q_pos},
                    {"o": ("f32", B * T * HQ * D)})
        k_all = torch.cat([kp.expand(B, P, HKV, D), kc[:, :n - P]], dim=1)
        v_all = torch.cat([vp.expand(B, P, HKV, D), vc[:, :n - P]], dim=1)
    a = fa["o"].reshape(B * T, HQ * D).contiguous()
    # The same attention in torch, so a disagreement names the attention kernel
    # rather than surfacing only as wrong logits three steps later.
    kk = k_all.repeat_interleave(HQ // HKV, dim=2)
    vv = v_all.repeat_interleave(HQ // HKV, dim=2)
    sc = torch.einsum("bthd,bshd->bhts", q, kk) * D ** -0.5
    mask = torch.arange(n)[None, :] > (pos + torch.arange(T))[:, None]
    ref = torch.einsum("bhts,bshd->bthd", sc.masked_fill(mask, float("-inf")).softmax(-1), vv)
    cq.check(f"{name} vs torch (B{B} T{T} pos{pos})", a, ref.reshape(B * T, HQ * D), 1e-5, 1e-5)
    gt = kernel("qwen35_attn_gate_f32",
                dict(rows=B * T, Hq=HQ, D=D, ld_p=L.attn_width, q_off=0, ld_out=HQ * D, out_off=0),
                {"attn": a, "p": proj}, {"out": B * T * HQ * D})
    out = gt["out"].reshape(B * T, HQ * D) @ sa.o_proj.weight.detach().T
    return out.reshape(B, T, HIDDEN), kc.contiguous(), vc.contiguous()


class State:
    """Per-layer carried state: conv + GDN for linear layers, KV for attention."""

    def __init__(self):
        self.conv, self.gdn, self.k, self.v = {}, {}, {}, {}


def forward(model, ids, pos=0, state=None, cap=None, snapshot=False, paths=None, shared=None):
    """Run the stack on the kernels. `ids` [B, T]. Returns (hidden [B*T, H], state).

    `shared` runs attention over a shared prefix: {"P": P, "prefix": {layer:
    (kp, vp)}, "cap": suffix capacity, "suffix": {layer: (kc, vc)} or absent
    for a fresh suffix}. The state's K/V are then the suffix caches."""
    B, T = ids.shape
    cap = cap or pos + T
    st_in = state
    st = State()
    x = model.model.embed_tokens(ids).detach()
    for i, layer in enumerate(model.model.layers):
        with torch.no_grad():
            h = layer.input_layernorm(x)
        if layer.block_type == "linear_attention":
            out, st.conv[i], st.gdn[i] = gdn_layer(
                layer.linear_attn, h, B, T,
                conv_state=None if st_in is None else st_in.conv[i],
                gdn_state=None if st_in is None else st_in.gdn[i],
                snapshot=snapshot, path=(paths or {}).get(i, "chunk"))
        elif shared is not None:
            kp, vp = shared["prefix"][i]
            if "suffix" in shared:
                kc, vc = shared["suffix"][i]
            else:
                kc = torch.full((B, shared["cap"], HKV, D), float("nan"))
                vc = torch.full((B, shared["cap"], HKV, D), float("nan"))
            out, st.k[i], st.v[i] = attn_layer(layer.self_attn, h, B, T, pos, kc, vc,
                                               prefix=(kp, vp, shared["P"]))
        else:
            if st_in is None:
                kc = torch.full((B, cap, HKV, D), float("nan"))
                vc = torch.full((B, cap, HKV, D), float("nan"))
            else:
                # A shared prefix's KV, copied per row: the attention kernel
                # has a batch dimension (docs/qwen35.md, "Not done").
                kc = torch.full((B, cap, HKV, D), float("nan"))
                vc = torch.full((B, cap, HKV, D), float("nan"))
                kc[:, :pos] = st_in.k[i][:, :pos]
                vc[:, :pos] = st_in.v[i][:, :pos]
            out, st.k[i], st.v[i] = attn_layer(layer.self_attn, h, B, T, pos, kc, vc)
        x = x + out
        with torch.no_grad():
            x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x.reshape(B * T, HIDDEN).contiguous(), st


def score(model, hidden, slots):
    r = cq.run("qwen35_score_rows_f32",
               dict(rows=hidden.shape[0], hidden=HIDDEN, n_ans=len(ANSWERS), vocab=VOCAB, n_slots=len(slots),
                    eps=EPS, w_offset=1.0),
               {"h": hidden, "norm_w": model.model.norm.weight.detach(),
                "emb": model.lm_head.weight.detach().contiguous(), "answers": ANSWERS,
                "slots": torch.tensor(slots, dtype=torch.int32)},
               {"logits": ("f32", len(slots) * len(ANSWERS)), "logprobs": ("f32", len(slots) * len(ANSWERS))})
    return r["logits"].reshape(len(slots), -1), r["logprobs"].reshape(len(slots), -1)


def compare(name, got_logits, got_logp, want_logits):
    want_logits = want_logits[:, ANSWERS.long()].double()
    scale = want_logits.abs().max().item()
    # The chain is ~20 f32 kernels and GEMMs deep and lands ~2e-7 of the logit
    # scale from the model; 2e-5 leaves ~10x headroom and is still far below
    # any wiring error, each of which moved the logits by >1e-3 of scale.
    cq.check(f"{name} logits", got_logits, want_logits, 2e-5 * scale)
    cq.check(f"{name} answer log-softmax", got_logp, torch.log_softmax(want_logits, -1), 2e-5 * max(scale, 1.0))


# -------------------------------------------------------------------- flows


def flow_prefill(model):
    T = 70                                  # a full chunk and a ragged one
    ids = torch.randint(0, VOCAB, (1, T), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want = model(ids).logits[0]
    hidden, _ = forward(model, ids)
    slots = [0, 31, 63, 64, 69]
    lg, lp = score(model, hidden, slots)
    compare("prefill T=70", lg, lp, want[slots])


def flow_decode(model):
    T = 40
    ids = torch.randint(0, VOCAB, (1, T + 2), generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        pre = model(ids[:, :T], use_cache=True)
        step1 = model(ids[:, T:T + 1], past_key_values=pre.past_key_values, use_cache=True)
        step2 = model(ids[:, T + 1:T + 2], past_key_values=step1.past_key_values, use_cache=True)
    _, st = forward(model, ids[:, :T], cap=T + 2)
    rec = {i: "recurrent" for i in range(3)}
    h1, st = forward(model, ids[:, T:T + 1], pos=T, state=st, cap=T + 2, paths=rec)
    h2, _ = forward(model, ids[:, T + 1:T + 2], pos=T + 1, state=st, cap=T + 2, paths=rec)
    for name, h, want in (("decode step 1", h1, step1), ("decode step 2", h2, step2)):
        lg, lp = score(model, h, [0])
        compare(name, lg, lp, want.logits[0, -1:])


def flow_snapshot(model):
    P, S, N = 66, 5, 3
    g = torch.Generator().manual_seed(7)
    prefix = torch.randint(0, VOCAB, (1, P), generator=g)
    suffixes = torch.randint(0, VOCAB, (N, S), generator=g)
    _, st = forward(model, prefix, cap=P + S)
    # Mixed paths: the recurrent kernel on two layers, the chunked one on the
    # third, both starting every row from the one shared snapshot.
    paths = {0: "recurrent", 1: "chunk", 2: "recurrent"}
    h, _ = forward(model, suffixes, pos=P, state=st, cap=P + S, snapshot=True, paths=paths)
    slots = [b * S + S - 1 for b in range(N)] + [S - 3]       # each question's last token, and one earlier
    lg, lp = score(model, h, slots)
    with torch.no_grad():
        want = torch.stack([model(torch.cat([prefix, suffixes[b:b + 1]], 1)).logits[0, -1] for b in range(N)]
                           + [model(torch.cat([prefix, suffixes[:1]], 1)).logits[0, P + S - 3]])
    compare(f"snapshot: {N} questions from one {P}-token prefix", lg, lp, want)


def flow_shared_prefix(model):
    """The snapshot flow with the attention prefix shared too: the prefix's K/V
    is stored once, with no batch dimension, and each question keeps only
    its own suffix cache. That is `attn_qk_norm_rope_suffix` into
    `attn_prefix_rows` for the questions, then one step of
    `attn_prefix_decode` per question, continuing its suffix cache and
    per-row GDN/conv state."""
    P, S, N = 66, 5, 3
    g = torch.Generator().manual_seed(7)
    prefix = torch.randint(0, VOCAB, (1, P), generator=g)
    suffixes = torch.randint(0, VOCAB, (N, S), generator=g)
    nxt = torch.randint(0, VOCAB, (N, 1), generator=g)
    _, st = forward(model, prefix, cap=P)
    shared = {"P": P, "cap": S + 1,
              "prefix": {i: (st.k[i][0].contiguous(), st.v[i][0].contiguous()) for i in st.k}}
    paths = {0: "recurrent", 1: "chunk", 2: "recurrent"}
    h, st2 = forward(model, suffixes, pos=P, state=st, snapshot=True, paths=paths, shared=shared)
    slots = [b * S + S - 1 for b in range(N)] + [S - 3]
    lg, lp = score(model, h, slots)
    with torch.no_grad():
        want = torch.stack([model(torch.cat([prefix, suffixes[b:b + 1]], 1)).logits[0, -1] for b in range(N)]
                           + [model(torch.cat([prefix, suffixes[:1]], 1)).logits[0, P + S - 3]])
    compare(f"shared prefix: {N} questions over one {P}-token KV prefix", lg, lp, want)
    step = dict(shared, suffix={i: (st2.k[i], st2.v[i]) for i in st2.k})
    rec = {i: "recurrent" for i in range(3)}
    h2, _ = forward(model, nxt, pos=P + S, state=st2, paths=rec, shared=step)
    lg, lp = score(model, h2, list(range(N)))
    with torch.no_grad():
        want = torch.stack([model(torch.cat([prefix, suffixes[b:b + 1], nxt[b:b + 1]], 1)).logits[0, -1]
                            for b in range(N)])
    compare(f"shared prefix: one decode step for each of {N} questions", lg, lp, want)


def main():
    torch.set_num_threads(1)
    cq.HARNESS = cq.build()
    # Order invariance is check_qwen35.py's job; one order keeps this quick.
    cq.ORDERS = ["forward"]
    model = build_model(35)
    for name, flow in (("prefill", flow_prefill), ("decode", flow_decode), ("snapshot", flow_snapshot),
                       ("shared_prefix", flow_shared_prefix)):
        print(f"{name}:")
        flow(model)
    if cq.FAILURES:
        print(f"\n{len(cq.FAILURES)} check(s) FAILED:")
        for f in cq.FAILURES:
            print("  " + f)
        sys.exit(1)
    print("\nall model checks passed")


if __name__ == "__main__":
    main()
