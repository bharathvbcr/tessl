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
        # transformers with the same values; tessl holds the embedding as
        # bf16, so the reference gets the rounded table.
        with torch.no_grad():
            for name, p in ref.named_parameters():
                v = params[name[len("model."):]].cpu()
                if name == "model.embed_tokens.weight":
                    v = v.to(torch.bfloat16).float()
                p.copy_(v)
        want_loss, want = torch_step(ref, self.ids)
        self.assertLessEqual(abs(loss - want_loss) / abs(want_loss), 1e-5)
        got = m.grads()
        self.assertLessEqual(max(rel_err(got[n], want[n]) for n in want), 1e-4)
        # And the parameters read back are what was written.
        back = m.parameters()
        for name, p in params.items():
            want_p = p.to(torch.bfloat16).float() if name == "embed_tokens.weight" else p
            tol = 2.0 ** -22 if name.endswith("layernorm.weight") or name == "norm.weight" else 0.0
            self.assertLessEqual((back[name] - want_p).abs().max().item(), tol, name)

    def test_refusals(self):
        m = self.model()
        with self.assertRaisesRegex(TesslError, "no gradients yet"):
            m.grads()
        with self.assertRaisesRegex(TesslError, "token id 64 >= vocab 64"):
            m.train_step([1, 64])
        with self.assertRaisesRegex(TesslError, "one sequence"):
            m.train_step(torch.zeros(2, 3, dtype=torch.long))
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
