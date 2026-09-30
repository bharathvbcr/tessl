"""tessl_torch.chunk_gated_delta_rule against transformers' own
torch_chunk_gated_delta_rule, through torch autograd.

The reference is the function transformers runs on macOS (its torch
fallback), on the CPU in float32 (it casts to float32 internally), so this is
an oracle independent of tessl's own f64 reference in tests/gdn_train.rs.

    cargo build --release
    python3 -m unittest discover -s python/tests -v
"""

import sys
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402

MPS = torch.device("mps")
REFERENCE = mq.torch_chunk_gated_delta_rule


def rel_err(got, want):
    got = got.detach().cpu().double()
    want = want.detach().cpu().double()
    return (got - want).abs().max().item() / max(want.abs().max().item(), 1e-30)


def inputs(seed, B, T, H, Dv, dtype, with_state):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, 128, generator=gen)
    k = torch.randn(B, T, H, 128, generator=gen)
    v = torch.randn(B, T, H, Dv, generator=gen)
    g = -torch.rand(B, T, H, generator=gen) * 1.5 - 1e-3
    beta = torch.rand(B, T, H, generator=gen) * 0.9 + 0.05
    s0 = torch.randn(B, H, 128, Dv, generator=gen) * 0.3 if with_state else None
    w_o = torch.randn(B, T, H, Dv, generator=gen)
    w_s = torch.randn(B, H, 128, Dv, generator=gen)
    xs = [q.to(dtype), k.to(dtype), v.to(dtype), g, beta.to(dtype)]
    return xs, s0, w_o, w_s


def run(fn, xs, s0, w_o, w_s, device):
    leaves = [x.detach().to(device).requires_grad_(True) for x in xs]
    st = s0.detach().to(device).requires_grad_(True) if s0 is not None else None
    out, fin = fn(*leaves, initial_state=st, output_final_state=True, use_qk_l2norm_in_kernel=True)
    loss = (out.float() * w_o.to(device)).sum() + (fin.float() * w_s.to(device)).sum()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    grads = [x.grad for x in leaves] + ([st.grad] if st is not None else [])
    return out, fin, grads


class MatchesTransformers(unittest.TestCase):
    def compare(self, B, T, H, Dv, dtype, with_state, tol, seed=0):
        xs, s0, w_o, w_s = inputs(seed, B, T, H, Dv, dtype, with_state)
        # The reference in f32 on the CPU from the same (rounded) operands.
        ref_xs = [x.float() for x in xs]
        want_o, want_f, want_g = run(REFERENCE, ref_xs, s0, w_o, w_s, torch.device("cpu"))
        got_o, got_f, got_g = run(tessl_torch.chunk_gated_delta_rule, xs, s0, w_o, w_s, MPS)
        self.assertEqual(got_o.dtype, xs[2].dtype)
        self.assertLessEqual(rel_err(got_o, want_o), tol, "out")
        self.assertLessEqual(rel_err(got_f, want_f), tol, "final state")
        names = ["dq", "dk", "dv", "dg", "dbeta", "ds0"]
        for name, gg, wg in zip(names, got_g, want_g):
            self.assertIsNotNone(gg, name)
            self.assertLessEqual(rel_err(gg, wg), tol, name)

    def test_f32_with_state_across_a_partial_chunk(self):
        self.compare(2, 70, 4, 32, torch.float32, True, 1e-4)

    def test_f32_one_token_and_no_state(self):
        self.compare(1, 1, 2, 16, torch.float32, False, 1e-4)

    def test_bf16_operands_at_the_2b_head_shape(self):
        # Outputs and gradients are cast to bf16 once: 2^-8 relative.
        self.compare(1, 130, 16, 128, torch.bfloat16, True, 2 ** -8)


class ThroughAQwen35Model(unittest.TestCase):
    def test_the_patched_model_trains_like_the_unpatched_one(self):
        # A small Qwen3.5 (grouped GDN heads: 2 key heads repeated to 4 value
        # heads) in f32 on MPS: loss and every parameter's gradient with the
        # torch fallback and with tessl.
        cfg = mq.Qwen3_5TextConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=256,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
            vocab_size=512,
            tie_word_embeddings=True,
        )
        torch.manual_seed(0)
        model = mq.Qwen3_5ForCausalLM(cfg).to(MPS)
        ids = torch.randint(0, 512, (2, 90), device=MPS)

        def step():
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            torch.mps.synchronize()
            return loss.item(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

        want_loss, want = step()
        previous = tessl_torch.patch_transformers_qwen3_5()
        try:
            self.assertIs(previous, REFERENCE)
            got_loss, got = step()
        finally:
            mq.torch_chunk_gated_delta_rule = previous
        self.assertAlmostEqual(got_loss, want_loss, delta=1e-4 * abs(want_loss))
        self.assertEqual(set(got), set(want))
        worst = max(rel_err(got[n], want[n]) for n in want)
        self.assertLessEqual(worst, 1e-3, "parameter gradients")


class Refuses(unittest.TestCase):
    def setUp(self):
        xs, s0, _, _ = inputs(1, 1, 4, 2, 32, torch.float32, True)
        self.xs = [x.to(MPS) for x in xs]
        self.s0 = s0.to(MPS)

    def refuses(self, needle, *xs, **kw):
        kw.setdefault("use_qk_l2norm_in_kernel", True)
        with self.assertRaises(TesslError) as cm:
            tessl_torch.chunk_gated_delta_rule(*xs, **kw)
        self.assertIn(needle, str(cm.exception))

    def test_unsupported_forms(self):
        self.refuses("use_qk_l2norm_in_kernel=True only", *self.xs, use_qk_l2norm_in_kernel=False)
        self.refuses("cu_seqlens", *self.xs, cu_seqlens=torch.tensor([0, 4]))

    def test_shapes(self):
        q, k, v, g, b = self.xs
        self.refuses("[B, T, H, 128]", q[..., :64], k[..., :64], v, g, b)
        self.refuses("multiple of 16", q, k, v[..., :24], g, b)
        self.refuses("g must be", q, k, v, g[:, :3], b)
        self.refuses("initial_state must be", q, k, v, g, b, initial_state=self.s0[..., :16])


if __name__ == "__main__":
    unittest.main()
