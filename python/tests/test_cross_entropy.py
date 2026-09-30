"""tessl_torch.cross_entropy against torch.nn.functional.cross_entropy.

The reference runs on the CPU in float64 from exactly the operands the GPU
reads, through torch's own autograd. Run with the library built:

    cargo build --release
    python3 -m unittest discover -s python/tests -v
"""

import sys
import threading
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tessl_torch  # noqa: E402
from tessl_torch import TesslError  # noqa: E402

MPS = torch.device("mps")


def reference(hidden, weight, targets, mask, reduction, upstream=1.0):
    h = hidden.detach().cpu().double().requires_grad_(True)
    w = weight.detach().cpu().double().requires_grad_(True)
    m = mask.to("cpu")
    logits = h[m] @ w.T
    loss = F.cross_entropy(logits, targets.to("cpu")[m], reduction=reduction)
    (loss * upstream).backward()
    return loss.item(), h.grad, w.grad


def run(hidden, weight, targets, mask, reduction, upstream=1.0, **kw):
    h = hidden.detach().clone().requires_grad_(True) if hidden.is_leaf else hidden
    w = weight.detach().clone().requires_grad_(True)
    if not hidden.is_leaf:
        base = kw.pop("base")
        base.grad = None
    loss = tessl_torch.cross_entropy(h, w, targets, mask, reduction=reduction, **kw)
    (loss * upstream).backward()
    torch.mps.synchronize()
    gh = (kw.get("base_ref") or h).grad if hidden.is_leaf else base.grad
    return loss.item(), gh, w.grad


def rel_err(got, want):
    got = got.detach().cpu().double()
    peak = want.abs().max().item()
    return (got - want).abs().max().item() / max(peak, 1e-300)


def problem(seed, lead, H, V, dtype, hscale=1.0):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(*lead, H, generator=g) * hscale).to(dtype).to(MPS)
    weight = (torch.randn(V, H, generator=g) * 0.25).to(dtype).to(MPS)
    targets = torch.randint(0, V, lead, generator=g)
    mask = torch.rand(lead, generator=g) < 0.4
    mask.view(-1)[0] = True  # never empty
    return hidden, weight, targets, mask


class CrossEntropyMatchesTorch(unittest.TestCase):
    def check(self, hidden, weight, targets, mask, reduction, upstream=1.0, tol=1e-4, **kw):
        want_loss, want_gh, want_gw = reference(hidden, weight, targets, mask, reduction, upstream)
        h = hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        loss = tessl_torch.cross_entropy(h, w, targets, mask, reduction=reduction, **kw)
        (loss * upstream).backward()
        torch.mps.synchronize()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertLessEqual(abs(loss.item() - want_loss), 1e-5 + 1e-5 * abs(want_loss))
        self.assertLessEqual(rel_err(h.grad, want_gh), tol, "grad hidden")
        self.assertLessEqual(rel_err(w.grad, want_gw), tol, "grad weight")
        self.assertEqual(h.grad.dtype, hidden.dtype)
        self.assertEqual(w.grad.dtype, weight.dtype)
        # Rows the mask leaves out get exactly zero gradient.
        self.assertEqual(h.grad[~mask.to(MPS)].abs().max().item() if (~mask).any() else 0.0, 0.0)
        return loss.item()

    def test_f32_mean_and_sum_with_partial_chunks(self):
        hidden, weight, targets, mask = problem(1, (2, 9), 64, 997, torch.float32)
        for reduction in ("mean", "sum"):
            self.check(hidden, weight, targets, mask, reduction, chunk=128)
        self.check(hidden, weight, targets, mask, "mean", upstream=3.5, chunk=0)

    def test_bf16_inputs_within_bf16_rounding_of_the_gradients(self):
        hidden, weight, targets, mask = problem(2, (3, 7), 64, 1000, torch.bfloat16, hscale=2.0)
        # Gradients come back cast to bf16: one rounding, 2^-8 relative.
        self.check(hidden, weight, targets, mask, "mean", tol=2 ** -8, chunk=256)

    def test_bf16_operands_match_the_reference_on_the_rounded_operands(self):
        # f32 inputs, bf16 GEMM operands: the reference is formed from the
        # bf16-rounded hidden states and weight, which the logit walk reads
        # exactly; dh and dW also round the softmax gradient (bound 2^-7, as
        # tests/cross_entropy.rs). The loss must be off the exact reference by
        # more than twice its bound, or nothing was rounded.
        hidden, weight, targets, mask = problem(4, (2, 9), 64, 997, torch.float32)
        exact_loss, _, _ = reference(hidden, weight, targets, mask, "mean")
        want_loss, want_gh, want_gw = reference(
            hidden.bfloat16().float(), weight.bfloat16().float(), targets, mask, "mean"
        )
        h = hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        loss = tessl_torch.cross_entropy(h, w, targets, mask, chunk=128, operands="bf16")
        loss.backward()
        torch.mps.synchronize()
        bound = 1e-5 + 1e-5 * abs(want_loss)
        self.assertLessEqual(abs(loss.item() - want_loss), bound)
        self.assertLessEqual(rel_err(h.grad, want_gh), 2 ** -7, "grad hidden")
        self.assertLessEqual(rel_err(w.grad, want_gw), 2 ** -7, "grad weight")
        self.assertGreater(abs(exact_loss - want_loss), 2 * bound)

    def test_a_shifted_slice_is_read_in_place(self):
        # hidden[:, :-1] of a [B, T, H] tensor: rows b*T + t of the storage.
        full, weight, _, _ = problem(3, (2, 10), 64, 500, torch.float32)
        g = torch.Generator().manual_seed(33)
        targets = torch.randint(0, 500, (2, 9), generator=g)
        mask = torch.rand((2, 9), generator=g) < 0.5
        mask[0, 0] = True
        leaf = full.detach().clone().requires_grad_(True)
        sliced = leaf[:, :-1]
        self.assertFalse(sliced.is_contiguous())
        want_loss, want_gh, want_gw = reference(sliced, weight, targets, mask, "mean")
        w = weight.detach().clone().requires_grad_(True)
        loss = tessl_torch.cross_entropy(sliced, w, targets, mask)
        loss.backward()
        torch.mps.synchronize()
        self.assertLessEqual(abs(loss.item() - want_loss), 1e-5 + 1e-5 * abs(want_loss))
        self.assertLessEqual(rel_err(leaf.grad[:, :-1], want_gh), 1e-4)
        self.assertEqual(leaf.grad[:, -1].abs().max().item(), 0.0)
        self.assertLessEqual(rel_err(w.grad, want_gw), 1e-4)

    def test_tensors_at_a_storage_offset(self):
        # hidden[:, 1:] starts H elements into its storage and weight[5:] 5
        # rows in: both are passed as the storage's MTLBuffer plus an offset.
        full, wfull, _, _ = problem(11, (2, 10), 64, 505, torch.float32)
        g = torch.Generator().manual_seed(111)
        targets = torch.randint(0, 500, (2, 9), generator=g)
        mask = torch.rand((2, 9), generator=g) < 0.6
        mask[1, 8] = True
        leaf = full.detach().clone().requires_grad_(True)
        wleaf = wfull.detach().clone().requires_grad_(True)
        hidden, weight = leaf[:, 1:], wleaf[5:]
        self.assertGreater(hidden.storage_offset(), 0)
        self.assertGreater(weight.storage_offset(), 0)
        self.assertTrue(weight.is_contiguous())
        want_loss, want_gh, want_gw = reference(hidden, weight, targets, mask, "sum")
        loss = tessl_torch.cross_entropy(hidden, weight, targets, mask, reduction="sum")
        loss.backward()
        torch.mps.synchronize()
        self.assertLessEqual(abs(loss.item() - want_loss), 1e-5 + 1e-5 * abs(want_loss))
        self.assertLessEqual(rel_err(leaf.grad[:, 1:], want_gh), 1e-4)
        self.assertEqual(leaf.grad[:, 0].abs().max().item(), 0.0)
        self.assertLessEqual(rel_err(wleaf.grad[5:], want_gw), 1e-4)
        self.assertEqual(wleaf.grad[:5].abs().max().item(), 0.0)

    def test_a_column_slice_falls_back_to_a_copy(self):
        g = torch.Generator().manual_seed(4)
        wide = torch.randn(3, 5, 96, generator=g).to(MPS)
        weight = (torch.randn(300, 64, generator=g) * 0.25).to(MPS)
        targets = torch.randint(0, 300, (3, 5), generator=g)
        mask = torch.ones((3, 5), dtype=torch.bool)
        for cols in (slice(0, 64), slice(32, 96)):
            self.check(wide[..., cols], weight, targets, mask, "sum")

    def test_the_real_vocabulary(self):
        hidden, weight, targets, mask = problem(5, (2, 6), 128, 248_320, torch.bfloat16)
        targets[0, 0] = 248_319
        self.check(hidden, weight, targets, mask, "mean", tol=2 ** -8)

    def test_no_grad_forward_matches(self):
        hidden, weight, targets, mask = problem(6, (4, 5), 64, 700, torch.float32)
        want_loss, _, _ = reference(hidden, weight, targets, mask, "mean")
        with torch.no_grad():
            loss = tessl_torch.cross_entropy(hidden, weight, targets, mask)
        self.assertLessEqual(abs(loss.item() - want_loss), 1e-5 + 1e-5 * abs(want_loss))

    def test_gradients_accumulate_like_any_autograd_op(self):
        hidden, weight, targets, mask = problem(7, (2, 4), 64, 300, torch.float32)
        w = weight.detach().clone().requires_grad_(True)
        for _ in range(2):
            tessl_torch.cross_entropy(hidden, w, targets, mask).backward()
        _, _, want_gw = reference(hidden, weight, targets, mask, "mean", upstream=2.0)
        self.assertLessEqual(rel_err(w.grad, want_gw), 1e-4)

    def test_reads_what_pending_torch_work_writes(self):
        # hidden comes out of a chain of large matmuls still queued on torch's
        # stream when the call starts; tessl must see their result, and the
        # gradient tessl writes must be complete when torch reads it.
        g = torch.Generator().manual_seed(10)
        x = torch.randn(2048, 2048, generator=g).to(MPS) / 45.0
        weight = (torch.randn(1000, 64, generator=g) * 0.25).to(MPS)
        targets = torch.randint(0, 1000, (2, 8), generator=g)
        mask = torch.ones((2, 8), dtype=torch.bool)
        for trial in range(3):
            torch.mps.synchronize()
            y = x
            for _ in range(24):
                y = y @ x
            hidden = y[:16, :64].reshape(2, 8, 64).contiguous()
            w = weight.detach().clone().requires_grad_(True)
            loss = tessl_torch.cross_entropy(hidden, w, targets, mask)
            loss.backward()
            gw = w.grad.clone()
            torch.mps.synchronize()
            want_loss, _, want_gw = reference(hidden, weight, targets, mask, "mean")
            self.assertLessEqual(abs(loss.item() - want_loss), 1e-5 + 1e-5 * abs(want_loss), f"trial {trial}")
            self.assertLessEqual(rel_err(gw, want_gw), 1e-4, f"trial {trial}")

    def test_another_thread_gets_its_own_runtime(self):
        hidden, weight, targets, mask = problem(8, (2, 4), 64, 300, torch.float32)
        want_loss, _, _ = reference(hidden, weight, targets, mask, "mean")
        out = {}

        def work():
            with torch.no_grad():
                out["loss"] = tessl_torch.cross_entropy(hidden, weight, targets, mask).item()

        t = threading.Thread(target=work)
        t.start()
        t.join()
        self.assertLessEqual(abs(out["loss"] - want_loss), 1e-5 + 1e-5 * abs(want_loss))


class CrossEntropyRefuses(unittest.TestCase):
    def setUp(self):
        self.hidden, self.weight, self.targets, self.mask = problem(9, (2, 4), 64, 300, torch.float32)

    def refuses(self, needle, *args, **kw):
        with self.assertRaises(TesslError) as cm:
            tessl_torch.cross_entropy(*args, **kw)
        self.assertIn(needle, str(cm.exception))

    def test_an_empty_mask(self):
        empty = torch.zeros((2, 4), dtype=torch.bool)
        self.refuses("selects no positions", self.hidden, self.weight, self.targets, empty)

    def test_a_target_past_the_vocabulary(self):
        bad = self.targets.clone()
        bad[self.mask] = 300
        self.refuses("targets must lie in [0, 300)", self.hidden, self.weight, bad, self.mask)

    def test_unknown_operands(self):
        self.refuses(
            "operands must be 'f32' or 'bf16', not 'tf32'",
            self.hidden, self.weight, self.targets, self.mask, operands="tf32",
        )

    def test_cpu_tensors(self):
        self.refuses("mps device", self.hidden.cpu(), self.weight, self.targets, self.mask)

    def test_misshapen_targets_and_masks(self):
        self.refuses("shaped like", self.hidden, self.weight, self.targets[:, :3], self.mask)
        self.refuses("must be bool", self.hidden, self.weight, self.targets, self.mask.int())

    def test_a_hidden_width_that_does_not_match_the_weight(self):
        narrow = self.hidden[..., :32].contiguous()
        self.refuses("hidden width 32", narrow, self.weight, self.targets, self.mask)

    def test_an_f16_weight_is_refused_by_tessl(self):
        self.refuses("f32 or bf16", self.hidden, self.weight.half(), self.targets, self.mask)

    def test_a_view_past_its_storage_is_refused_before_tessl_sees_it(self):
        # torch will not build such a view (as_strided and set_ check), but the
        # MTLBuffer behind a storage is larger than the storage (torch rounds
        # allocations up), so tessl's own bound would pass it: the binding
        # checks the storage. A tensor-shaped stand-in reaches the check.
        real = torch.zeros(1000, device=MPS)

        class Overrun:
            device, dtype, shape = real.device, real.dtype, (1000,)

            def is_contiguous(self):
                return True

            def dim(self):
                return 1

            def untyped_storage(self):
                return real.untyped_storage()

            def storage_offset(self):
                return 4

            def element_size(self):
                return 4

            def numel(self):
                return 1000

        with self.assertRaises(TesslError) as cm:
            tessl_torch._ref(Overrun(), "hidden")
        self.assertIn("exceed its storage", str(cm.exception))

    def test_the_library_reports_its_abi(self):
        lib = tessl_torch._load()
        self.assertEqual(lib.tessl_abi_version(), 6)
        t = torch.zeros(1000, device=MPS)
        torch.mps.synchronize()
        # The storage pointer is the MTLBuffer (its length is torch's rounded bucket).
        self.assertGreaterEqual(lib.tessl_mtl_buffer_length(t.untyped_storage().data_ptr()), 4000)


if __name__ == "__main__":
    unittest.main()
