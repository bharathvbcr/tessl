"""tessl_torch.Qwen35 against transformers' own Qwen3_5ForCausalLM.

The oracle is transformers' autograd run here, on the CPU in float32 with
eager attention, on the committed tiny fixture (tests/fixtures/qwen35_train):
independent of the gradients tools/qwen35_ref/make_train_fixture.py saved.

    cargo build --release
    python3 -m unittest discover -s python/tests -v
"""

import json
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "qwen35_train"
SAFETENSORS = FIXTURE / "model.safetensors"
CONFIG = FIXTURE / "config.json"


def reference():
    cfg = mq.Qwen3_5TextConfig(**json.loads(CONFIG.read_text()))
    cfg._attn_implementation = "eager"
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    state = {k: v.float() for k, v in load_file(str(SAFETENSORS)).items()}
    # Only the tied head is absent from the checkpoint.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model


def ids():
    import numpy as np

    return torch.from_numpy(np.load(FIXTURE / "ids.npy"))


def torch_step(model, x):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=x[None], labels=x[None])
    out.loss.backward()
    return out.loss.item(), {n[len("model."):]: p.grad.detach().clone() for n, p in model.named_parameters()}


def rel_err(got, want):
    got = got.detach().cpu().double()
    want = want.detach().cpu().double()
    return (got - want).abs().max().item() / max(want.abs().max().item(), 1e-30)


class Qwen35Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = ids()

    def model(self):
        return tessl_torch.Qwen35(SAFETENSORS, CONFIG, prefix="model.")

    def test_parameters_are_the_checkpoints(self):
        m = self.model()
        params = m.parameters()
        state = load_file(str(SAFETENSORS))
        self.assertEqual(sorted("model." + n for n in params), sorted(state))
        for name, p in params.items():
            want = state["model." + name].float()
            self.assertEqual(tuple(p.shape), tuple(want.shape), name)
            # 1 + w round-trips w within f32's rounding at 1.
            tol = 2.0 ** -23 if name.endswith("layernorm.weight") or name == "norm.weight" else 0.0
            self.assertLessEqual((p.cpu() - want).abs().max().item(), tol, name)

    def test_step_matches_transformers_autograd(self):
        m = self.model()
        loss = m.train_step(self.ids)
        want_loss, want = torch_step(reference(), self.ids)
        self.assertLessEqual(abs(loss - want_loss) / abs(want_loss), 1e-5)
        grads = m.grads()
        self.assertEqual(sorted(grads), sorted(want))
        worst = max(rel_err(grads[n], want[n]) for n in want)
        self.assertLessEqual(worst, 1e-4)

    def test_a_bf16_operand_step_stays_near_transformers(self):
        # The same bounds as tests/qwen35_train.rs (set for this fixture before
        # its first run): loss within 2^-8, every gradient within 2^-5 of its
        # own peak. And not the exact step's loss: something was rounded.
        m = self.model()
        loss = m.train_step(self.ids, operands="bf16")
        want_loss, want = torch_step(reference(), self.ids)
        self.assertLessEqual(abs(loss - want_loss) / abs(want_loss), 2.0 ** -8)
        grads = m.grads()
        worst = max(rel_err(grads[n], want[n]) for n in want)
        self.assertLessEqual(worst, 2.0 ** -5)
        self.assertNotEqual(loss, m.train_step(self.ids))

    def test_grads_into_reuses_the_callers_tensors(self):
        # A second step's gradients written into the first step's tensors:
        # the same bits as a fresh grads(), in the same storage, with no MPS
        # memory allocated by the call.
        m = self.model()
        m.train_step(self.ids)
        held = m.grads()
        first = {n: t.clone() for n, t in held.items()}
        ptrs = {n: t.data_ptr() for n, t in held.items()}
        m.train_step(self.ids[: len(self.ids) // 2])
        torch.mps.synchronize()
        before = torch.mps.current_allocated_memory()
        out = m.grads(into=held)
        torch.mps.synchronize()
        self.assertEqual(torch.mps.current_allocated_memory(), before)
        self.assertIs(out, held)
        fresh = m.grads()
        for n, t in held.items():
            self.assertEqual(t.data_ptr(), ptrs[n], n)
            self.assertTrue(torch.equal(t, fresh[n]), n)
        # The half-length step's gradients are not the first step's, so the
        # tensors were written, not left as they were.
        self.assertTrue(any(not torch.equal(held[n], first[n]) for n in held))

    def test_grads_into_checks_every_tensor_before_writing(self):
        m = self.model()
        m.train_step(self.ids)
        held = m.grads()
        name, shape, tr = next(e for e in m._table if e[2])
        bad = dict(held)
        del bad[name]
        with self.assertRaisesRegex(TesslError, "missing"):
            m.grads(into=bad)
        bad = dict(held)
        bad[name] = held[name].half()
        with self.assertRaisesRegex(TesslError, f"{name} must be f32"):
            m.grads(into=bad)
        bad = dict(held)
        bad[name] = held[name].contiguous()  # right shape, wrong layout for a transposed entry
        with self.assertRaisesRegex(TesslError, f"{name} is not laid out as grads\\(\\) returns it"):
            m.grads(into=bad)
        # A refusal writes nothing: the other tensors in the dict are untouched.
        # Every entry valid (zeroed, in grads()'s layout) except the one.
        layout = {n: t for n, _, t in m._table}
        zeros = {n: (torch.zeros_like(t.t()).t() if layout[n] else torch.zeros_like(t)) for n, t in held.items()}
        zeros[name] = held[name].contiguous()
        before = {n: z.clone() for n, z in zeros.items()}
        with self.assertRaises(TesslError):
            m.grads(into=zeros)
        for n in zeros:
            self.assertTrue(torch.equal(zeros[n], before[n]), n)

    def test_an_optimizer_step_written_back_is_transformers_after_the_same_step(self):
        m = self.model()
        ref = reference()
        params = m.parameters()
        m.train_step(self.ids)
        grads = m.grads()
        opt = torch.optim.AdamW(params.values(), lr=1e-2, weight_decay=0.1)
        for name, p in params.items():
            p.grad = grads[name]
        opt.step()
        m.load_parameters(params)
        loss = m.train_step(self.ids)
        # transformers with the same values, the embedding included (tessl's
        # table is f32, so nothing is rounded on the way in).
        with torch.no_grad():
            for name, p in ref.named_parameters():
                p.copy_(params[name[len("model."):]].cpu())
        want_loss, want = torch_step(ref, self.ids)
        self.assertLessEqual(abs(loss - want_loss) / abs(want_loss), 1e-5)
        got = m.grads()
        self.assertLessEqual(max(rel_err(got[n], want[n]) for n in want), 1e-4)
        # And the parameters read back are what was written.
        back = m.parameters()
        for name, p in params.items():
            tol = 2.0 ** -22 if name.endswith("layernorm.weight") or name == "norm.weight" else 0.0
            self.assertLessEqual((back[name] - p).abs().max().item(), tol, name)

    def test_refusals(self):
        m = self.model()
        with self.assertRaisesRegex(TesslError, "no gradients yet"):
            m.grads()
        with self.assertRaisesRegex(TesslError, "token id 64 >= vocab 64"):
            m.train_step([1, 64])
        with self.assertRaisesRegex(TesslError, "one sequence"):
            m.train_step(torch.zeros(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(TesslError, "operands must be 'f32' or 'bf16', not 'fp8'"):
            m.train_step(self.ids, operands="fp8")
        params = m.parameters()
        bad = dict(params)
        del bad["norm.weight"]
        with self.assertRaisesRegex(TesslError, "missing \\['norm.weight'\\]"):
            m.load_parameters(bad)
        bad = dict(params)
        bad["norm.weight"] = torch.zeros(3)
        with self.assertRaisesRegex(TesslError, "norm.weight must be"):
            m.load_parameters(bad)
        with self.assertRaisesRegex(TesslError, "does-not-exist"):
            tessl_torch.Qwen35(FIXTURE / "does-not-exist.safetensors", CONFIG, prefix="model.")
        # A refused step after a good one leaves the gradients unavailable,
        # not the good step's, on both sides of the ABI.
        m.train_step(self.ids)
        m.grads()
        with self.assertRaisesRegex(TesslError, "token id 64 >= vocab 64"):
            m.train_step([1, 64])
        with self.assertRaisesRegex(TesslError, "no gradients yet"):
            m.grads()
        m._has_grads = True
        with self.assertRaisesRegex(TesslError, "no gradients yet; run tessl_qwen35_train_step"):
            m.grads()


if __name__ == "__main__":
    unittest.main()
