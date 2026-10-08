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
            # The checkpoint's bits, the zero-centred norms' w included.
            self.assertTrue(torch.equal(p.cpu(), want), name)

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

    def test_a_bf16_stored_step_stays_near_transformers(self):
        # tests/qwen35_train.rs's bf16-storage bounds (set before its first
        # run): loss within 2^-7, every gradient within 2^-4 of its own peak.
        # The forward is the bf16-operand forward, so the loss is that
        # step's; the gradients come back as f32.
        m = tessl_torch.Qwen35(SAFETENSORS, CONFIG, prefix="model.", precision="bf16")
        loss = m.train_step(self.ids, operands="bf16")
        want_loss, want = torch_step(reference(), self.ids)
        self.assertLessEqual(abs(loss - want_loss) / abs(want_loss), 2.0 ** -7)
        grads = m.grads()
        self.assertEqual(sorted(grads), sorted(want))
        self.assertTrue(all(g.dtype == torch.float32 for g in grads.values()))
        worst = max(rel_err(grads[n], want[n]) for n in want)
        self.assertLessEqual(worst, 2.0 ** -4)
        self.assertEqual(loss, self.model().train_step(self.ids, operands="bf16"))
        with self.assertRaisesRegex(TesslError, "trains on GemmOperands::Bf16"):
            m.train_step(self.ids)
        self.assertIn("matrices bf16", m.describe())

    def test_a_bf16_adamw_checkpoint_resumes_bit_for_bit(self):
        # For each update rule and moment storage of a bf16 model: two steps,
        # a checkpoint (parameters plus adamw_state, which records the
        # configuration and any aux state), a third step; a fresh bf16 model
        # restored from it takes the third step to the same bits.
        def step(model):
            model.train_step(self.ids, operands="bf16")
            model.adamw_step(1e-2, weight_decay=0.1)

        def bf16():
            return tessl_torch.Qwen35(SAFETENSORS, CONFIG, prefix="model.", precision="bf16")

        for update, moments, seed in (("f32-master", "bf16", 0), ("bf16-kahan", "block8", 0),
                                      ("bf16-stochastic", "block8", 7)):
            with self.subTest(update=update, moments=moments):
                m = bf16()
                m.adamw_init(update, moments, seed)
                step(m)
                step(m)
                params, state = m.parameters(), m.adamw_state()
                self.assertEqual(state["config"], {"update": update, "seed": seed, "moments": moments})
                self.assertEqual("aux" in state, update != "bf16-stochastic")
                self.assertIn(f"update={update}", m.describe())
                self.assertTrue(m.describe().endswith("step=2"), m.describe())
                step(m)
                want = m.parameters()
                r = bf16()
                r.load_parameters(params)
                r.adamw_init(update, moments, seed)
                r.load_adamw_state(state)
                step(r)
                got = r.parameters()
                for n in want:
                    self.assertTrue(torch.equal(got[n], want[n]), n)

    def test_bf16_storage_refusals(self):
        with self.assertRaisesRegex(TesslError, "precision must be 'f32' or 'bf16', not 'f16'"):
            tessl_torch.Qwen35(SAFETENSORS, CONFIG, prefix="model.", precision="f16")
        m = tessl_torch.Qwen35(SAFETENSORS, CONFIG, prefix="model.", precision="bf16")
        with self.assertRaisesRegex(TesslError, "a bf16 model takes update="):
            m.adamw_init()
        with self.assertRaisesRegex(TesslError, "does not apply to a Bf16 model"):
            m.adamw_init("f32")
        with self.assertRaisesRegex(TesslError, "moments must be one of"):
            m.adamw_init("bf16-kahan", "int4")
        with self.assertRaisesRegex(TesslError, "seed 3 given to a rule that takes none"):
            m.adamw_init("bf16-kahan", "block8", 3)
        m.adamw_init("bf16-kahan", "block8")
        state = m.adamw_state()
        m.adamw_free()
        m.adamw_init("f32-master", "block8")
        with self.assertRaisesRegex(TesslError, "the checkpoint's AdamW is .*bf16-kahan.*call adamw_init"):
            m.load_adamw_state(state)
        m.adamw_free()
        m.adamw_init("bf16-kahan", "block8")
        with self.assertRaisesRegex(TesslError, "'bf16-kahan' needs aux state, and the checkpoint has none"):
            m.load_adamw_state({k: v for k, v in state.items() if k != "aux"})
        # A torch-style checkpoint (no config) is f32 AdamW, which this is not.
        with self.assertRaisesRegex(TesslError, "the checkpoint's AdamW is .*'f32'"):
            m.load_adamw_state({k: state[k] for k in ("step", "exp_avg", "exp_avg_sq")})
        self.assertEqual(m.adamw_step_count, 0)
        f = self.model()
        f.adamw_init()
        with self.assertRaisesRegex(TesslError, "'f32' keeps no aux state, and the checkpoint has some"):
            f.load_adamw_state({**f.adamw_state(), "aux": state["aux"]})

    def test_adamw_in_tessl_is_torch_adamw(self):
        # torch.optim.AdamW on CPU copies, with the same two groups (Trainer's
        # exclusions take no decay), fed tessl's gradients each step. Bound
        # set before the first run: 1e-6 per element (lr 1e-2: a semantic
        # error is 1e-5 or more), every parameter alike.
        m = self.model()
        ref = {n: p.cpu().clone().requires_grad_(True) for n, p in m.parameters().items()}
        excluded = dict(zip((n for n, _, _ in m._table), m._decay_excluded))
        self.assertTrue(excluded["layers.0.linear_attn.dt_bias"] and excluded["norm.weight"])
        self.assertFalse(excluded["embed_tokens.weight"] or excluded["layers.0.linear_attn.A_log"])
        opt = torch.optim.AdamW(
            [
                {"params": [p for n, p in ref.items() if not excluded[n]], "weight_decay": 0.1},
                {"params": [p for n, p in ref.items() if excluded[n]], "weight_decay": 0.0},
            ],
            lr=1e-2,
        )
        m.adamw_init()
        losses = []
        for step in range(1, 4):
            losses.append(m.train_step(self.ids))
            grads = m.grads()
            for n, p in ref.items():
                p.grad = grads[n].cpu()
            opt.step()
            m.adamw_step(1e-2, weight_decay=0.1)
            self.assertEqual(m.adamw_step_count, step)
            got = m.parameters()
            for n, p in ref.items():
                err = (got[n].cpu() - p.detach()).abs().max().item()
                self.assertLessEqual(err, 1e-6, f"step {step} {n}: {err:.3e}")
        # Training on one sequence lowers its loss.
        self.assertLess(m.train_step(self.ids), losses[0])

    def test_a_padded_batch_with_an_outside_head_is_row_by_row_steps(self):
        # The reference: transformers on a right-padded batch of two rows
        # (attention mask from the lengths), one loss over both. A letter row
        # supervises one position through the tied head (mean over the
        # batch's letter rows); a span row feeds a pointer head in torch
        # (bias-free projection of the query row against candidate rows plus
        # a learned abstain vector, start and end, mean over the batch's
        # pointer decisions). tessl: each row trimmed to its length, one
        # forward/backward per row accumulated in the bank, the span row's
        # gradient coming back from torch through hidden()/train_backward().
        # Bounds as test_step_matches_transformers_autograd, set before the
        # first run; the global norm (tessl's plus the head's) to 1e-5.
        torch.manual_seed(0)
        x = self.ids
        a, b = x[:13], x[4:15]
        pad = 0
        width = max(a.numel(), b.numel())
        batch = torch.full((2, width), pad, dtype=x.dtype)
        batch[0, : a.numel()], batch[1, : b.numel()] = a, b
        lengths = torch.tensor([a.numel(), b.numel()])
        mask = (torch.arange(width)[None] < lengths[:, None]).long()
        letter_pos = a.numel() - 2  # predicts the last real token
        query, cands = b.numel() - 2, [0, 3, 6]  # the query repeats as no candidate
        start_t, end_t = 1, len(cands)  # a candidate, and abstain
        H = self.model()._hidden
        proj = torch.nn.Linear(H, H, bias=False)
        abstain = torch.nn.Parameter(torch.randn(H) * 0.1)

        def span_loss(h_query, h_cands):
            q = proj(h_query)
            scores = torch.cat([h_cands @ q, (abstain @ q)[None]])
            ce = torch.nn.functional.cross_entropy
            return (ce(scores[None], torch.tensor([start_t])) + ce(scores[None], torch.tensor([end_t]))) / 2

        ref = reference()
        ref.zero_grad(set_to_none=True)
        hs = ref.model(input_ids=batch, attention_mask=mask).last_hidden_state
        letter = torch.nn.functional.cross_entropy(ref.lm_head(hs[0, letter_pos])[None], a[letter_pos + 1][None])
        span = span_loss(hs[1, query], hs[1, cands])
        (letter + span).backward()
        want = {n[len("model."):]: p.grad.detach().clone() for n, p in ref.named_parameters()}
        head_want = [proj.weight.grad.clone(), abstain.grad.clone()]
        proj.zero_grad(set_to_none=True)
        abstain.grad = None

        m = self.model()
        got_letter = m.train_forward(a, positions=[letter_pos], targets=[int(a[letter_pos + 1])], scale=1.0)
        m.train_backward()
        m.train_forward(b, positions=[], targets=[])
        rows = [query] + cands
        h = m.hidden(rows).cpu().requires_grad_(True)
        got_span = span_loss(h[0], h[1:])
        got_span.backward()
        m.train_backward(dh=h.grad, positions=rows, accumulate=True)

        self.assertLessEqual(abs(got_letter - letter.item()) / abs(letter.item()), 1e-5)
        self.assertLessEqual(abs(got_span.item() - span.item()) / abs(span.item()), 1e-5)
        grads = m.grads()
        worst = max((rel_err(grads[n], want[n]), n) for n in want)
        self.assertLessEqual(worst[0], 1e-4, worst)
        for got, w in zip([proj.weight.grad, abstain.grad], head_want):
            self.assertLessEqual(rel_err(got, w), 1e-4)
        # The global norm clip_grad_norm_ takes over the model and the head:
        # tessl's grad_sq_norm plus the head's squares, against the norm of
        # the reference's own per-tensor norms (its gradients, both parts).
        total = (m.grad_sq_norm() + sum(float((g.double() ** 2).sum()) for g in (proj.weight.grad, abstain.grad))) ** 0.5
        ref_norms = [g.norm() for g in want.values()] + [g.norm() for g in head_want]
        want_total = torch.linalg.vector_norm(torch.stack(ref_norms)).item()
        self.assertLessEqual(abs(total - want_total), 1e-5 * want_total)

    def test_two_phase_refusals(self):
        m = self.model()
        with self.assertRaisesRegex(TesslError, "no step is pending"):
            m.hidden([0])
        with self.assertRaisesRegex(TesslError, "no step is pending"):
            m.train_backward()
        m.train_forward(self.ids, positions=[1], targets=[2])
        with self.assertRaisesRegex(TesslError, "a step is pending"):
            m.train_step(self.ids)
        with self.assertRaisesRegex(TesslError, "a step is pending"):
            m.load_parameters(m.parameters())
        with self.assertRaisesRegex(TesslError, "dh must be \\[2, "):
            m.train_backward(dh=torch.zeros(1, m._hidden), positions=[0, 1])
        with self.assertRaisesRegex(TesslError, "position 70 >= 70"):
            m.train_backward(dh=torch.zeros(1, m._hidden), positions=[70])
        # Still pending after the refusals; a discard drops it.
        m.hidden([0, 0])
        m.train_discard()
        with self.assertRaisesRegex(TesslError, "no step is pending"):
            m.hidden([0])
        with self.assertRaisesRegex(TesslError, "position 3 is supervised twice"):
            m.train_forward(self.ids, positions=[3, 3], targets=[1, 2])
        with self.assertRaisesRegex(TesslError, "give positions and targets together"):
            m.train_forward(self.ids, positions=[3])
        with self.assertRaisesRegex(TesslError, "no step is pending"):
            m.hidden([])

    def test_a_refused_or_discarded_step_keeps_the_gradients(self):
        # The library's bank decides: a refused forward and a discarded one
        # leave the last finished step's gradients readable.
        m = self.model()
        m.train_step(self.ids)
        want = m.grads()
        with self.assertRaisesRegex(TesslError, "position 3 is supervised twice"):
            m.train_forward(self.ids, positions=[3, 3], targets=[1, 2])
        got = m.grads()
        self.assertTrue(all(torch.equal(got[n], want[n]) for n in want))
        m.train_forward(self.ids, positions=[3], targets=[1])
        m.train_discard()
        got = m.grads()
        self.assertTrue(all(torch.equal(got[n], want[n]) for n in want))

    def test_repeated_positions_in_dh_are_summed(self):
        # train_backward merges the rows of a repeated position: the same
        # gradients as one row holding their sum.
        m = self.model()
        rows = [5, 9, 5]
        dh = torch.randn(len(rows), m._hidden) * 1e-2
        m.train_forward(self.ids, positions=[], targets=[])
        m.train_backward(dh=dh, positions=rows)
        got = m.grads()
        m.train_forward(self.ids, positions=[], targets=[])
        m.train_backward(dh=torch.stack([dh[0] + dh[2], dh[1]]), positions=[5, 9])
        want = m.grads()
        worst = max(rel_err(got[n], want[n]) for n in want)
        self.assertLessEqual(worst, 1e-6)

    def test_clipping_is_clip_grad_norm_then_adamw(self):
        # torch: clip_grad_norm_(max_norm) over every parameter, then AdamW.
        # tessl: grad_sq_norm() -> clip_coef -> adamw_step(grad_scale=...).
        # max_norm is below the tiny fixture's gradient norm so it clips; if
        # it does not, lower max_norm rather than drop the check.
        # Adam cancels a gradient scale except against eps, so eps is 1e-2
        # here (the Rust test shows the scale then moves the result by far
        # more than the bound). Bounds as test_adamw_in_tessl_is_torch_adamw;
        # the norm to 1e-5 relative (torch reduces in another order).
        m = self.model()
        ref = {n: p.cpu().clone().requires_grad_(True) for n, p in m.parameters().items()}
        opt = torch.optim.AdamW(list(ref.values()), lr=1e-2, eps=1e-2, weight_decay=0.0)
        m.adamw_init()
        coefs = []
        for step in range(1, 3):
            m.train_step(self.ids)
            grads = m.grads()
            for n, p in ref.items():
                p.grad = grads[n].cpu()
            max_norm = 0.5
            want_norm = torch.nn.utils.clip_grad_norm_(list(ref.values()), max_norm).item()
            got_norm = m.grad_sq_norm() ** 0.5
            self.assertLessEqual(abs(got_norm - want_norm), 1e-5 * want_norm)
            # One coefficient for both sides, so a parameter mismatch is the
            # update's, not the norm's reduction order (checked just above).
            coef = tessl_torch.clip_coef(want_norm, max_norm)
            coefs.append(coef)
            opt.step()
            m.adamw_step(1e-2, eps=1e-2, weight_decay=0.0, grad_scale=coef)
            got = m.parameters()
            for n, p in ref.items():
                tol = 1e-6 + (2.0 ** -22 if n.endswith("layernorm.weight") or n == "norm.weight" else 0.0)
                err = (got[n].cpu() - p.detach()).abs().max().item()
                self.assertLessEqual(err, tol, f"step {step} {n}: {err:.3e}")
        self.assertTrue(all(c < 1.0 for c in coefs), coefs)
        # The gradients tessl keeps are not clipped.
        kept = m.grads()
        self.assertTrue(all(torch.equal(kept[k], grads[k]) for k in grads))
        self.assertEqual(tessl_torch.clip_coef(0.1, 1.0), 1.0)
        with self.assertRaisesRegex(TesslError, "grad_scale NaN must be finite and >= 0"):
            m.adamw_step(1e-2, grad_scale=float("nan"))

    def test_an_adamw_checkpoint_resumes_bit_for_bit(self):
        # Two steps, a checkpoint (parameters plus adamw_state), a third step.
        # A fresh model restored from the checkpoint takes the same third
        # step to the same bits; one given the parameters and a fresh AdamW
        # state does not, so the moments and step count are what carried it.
        def step(model):
            model.train_step(self.ids)
            model.adamw_step(1e-2, weight_decay=0.1)

        m = self.model()
        m.adamw_init()
        step(m)
        step(m)
        params, state = m.parameters(), m.adamw_state()
        self.assertEqual(state["step"], 2)
        self.assertEqual(set(state["exp_avg"]), set(params))
        step(m)
        want = m.parameters()

        def resumed(restore):
            r = self.model()
            r.load_parameters(params)
            r.adamw_init()
            if restore:
                r.load_adamw_state(state)
                self.assertEqual(r.adamw_step_count, 2)
                back = r.adamw_state()
                for key in ("exp_avg", "exp_avg_sq"):
                    for n, t in state[key].items():
                        self.assertTrue(torch.equal(back[key][n], t), f"{key} {n}")
            step(r)
            return r.parameters()

        got = resumed(True)
        for n in want:
            self.assertTrue(torch.equal(got[n], want[n]), n)
        cold = resumed(False)
        self.assertTrue(any(not torch.equal(cold[n], want[n]) for n in want))

    def test_adamw_state_refusals(self):
        m = self.model()
        with self.assertRaisesRegex(TesslError, "no AdamW state"):
            m.adamw_state()
        m.adamw_init()
        state = m.adamw_state()
        m.adamw_free()
        with self.assertRaisesRegex(TesslError, "no AdamW state; call tessl_qwen35_adamw_init first"):
            m.load_adamw_state(state)
        m.adamw_init()
        with self.assertRaisesRegex(TesslError, "want keys step, exp_avg and exp_avg_sq"):
            m.load_adamw_state({"step": 1, "exp_avg": state["exp_avg"]})
        with self.assertRaisesRegex(TesslError, "step -1 is negative"):
            m.load_adamw_state({**state, "step": -1})
        bad = dict(state["exp_avg_sq"])
        bad["norm.weight"] = bad["norm.weight"][:-1]
        with self.assertRaisesRegex(TesslError, "load_adamw_state: exp_avg_sq: norm.weight must be"):
            m.load_adamw_state({**state, "exp_avg_sq": bad, "step": 5})
        # Refused before anything was written: still the fresh state.
        self.assertEqual(m.adamw_step_count, 0)
        after = m.adamw_state()
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertTrue(all(not t.any() for t in after[key].values()), key)

    def test_adamw_refusals(self):
        m = self.model()
        m.train_step(self.ids)
        with self.assertRaisesRegex(TesslError, "no AdamW state; call tessl_qwen35_adamw_init first"):
            m.adamw_step(1e-2)
        m.adamw_init()
        with self.assertRaisesRegex(TesslError, "already has AdamW state"):
            m.adamw_init()
        with self.assertRaisesRegex(TesslError, "weight_decay is missing \\['embed_tokens.weight'"):
            m.adamw_step(1e-2, weight_decay={"norm.weight": 0.0})
        with self.assertRaisesRegex(TesslError, "beta2 1 must lie in"):
            m.adamw_step(1e-2, betas=(0.9, 1.0))
        self.assertEqual(m.adamw_step_count, 0)
        m.adamw_free()
        with self.assertRaisesRegex(TesslError, "no AdamW state"):
            m.adamw_step_count

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
            self.assertTrue(torch.equal(back[name], p), name)

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
