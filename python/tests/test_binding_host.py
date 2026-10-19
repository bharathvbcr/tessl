"""tessl_torch's handle ownership and argument ranges, with no GPU.

The library is replaced by a recording stand-in (``tessl_torch._load`` is
patched), so these run anywhere torch imports:

    python3 -m unittest python/tests/test_binding_host.py -v
"""

import copy
import ctypes
import pickle
import sys
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import tessl_torch  # noqa: E402
from tessl_torch import TesslError, TesslPanic, TesslPoisoned  # noqa: E402

RUNTIME, MODEL = 0xBEEF0000, 0xDEAD0000


class FakeLib:
    """Records every call; ``status`` is what the model's AdamW entry points
    and the GDN entries return."""

    def __init__(self):
        self.calls = []
        self.status = 0
        self.free_status = 0
        self.load_status = 0
        self.poisoned = 0
        self.poison_query_status = 0
        self.async_status = 0
        self.parameter_status = 0
        # Values read_parameter_host hands back, in storage order.
        self.parameter_values = [1.0, 2.0, 3.0, 4.0]
        self.written = None

    def _rec(self, name, *args):
        self.calls.append((name, args[0] if args else None))

    def frees(self):
        return [c for c in self.calls if c[0].endswith("_free")]

    def tessl_runtime_new(self, err, n):
        self.calls.append(("tessl_runtime_new", None))
        return RUNTIME

    def tessl_runtime_free(self, h):
        self._rec("tessl_runtime_free", h.value)
        return self.free_status

    def tessl_runtime_set_async_encode(self, h, on, err, n):
        self.calls.append(("tessl_runtime_set_async_encode", on))
        if self.async_status:
            err.value = b"async refused"
        return self.async_status

    def tessl_qwen35_read_parameter_host(self, h, index, dst, count, err, n):
        self.calls.append(("tessl_qwen35_read_parameter_host", (index, count)))
        if self.parameter_status:
            err.value = b"parameter refused"
            return self.parameter_status
        for k, v in enumerate(self.parameter_values[:count]):
            dst[k] = v
        return 0

    def tessl_qwen35_write_parameter_host(self, h, index, src, count, err, n):
        self.calls.append(("tessl_qwen35_write_parameter_host", (index, count)))
        if self.parameter_status:
            err.value = b"parameter refused"
            return self.parameter_status
        self.written = [src[k] for k in range(count)]
        return 0

    def tessl_qwen35_train_step_ragged(self, h, ids, n, cu, n_cu, operands, acc, loss, per_seq, n_seq, err, n_err):
        self.ragged = ([ids[k] for k in range(n)], [cu[k] for k in range(n_cu)], operands, acc, n_seq)
        self.calls.append(("tessl_qwen35_train_step_ragged", h.value))
        loss._obj.value = 1.5
        for k in range(n_seq):
            per_seq[k] = 0.25 * (k + 1)
        return 0

    def tessl_runtime_poisoned(self, h, out, err, n):
        self._rec("tessl_runtime_poisoned", h.value)
        out._obj.value = self.poisoned
        return self.poison_query_status

    def tessl_qwen35_load(self, rt, st, cfg, prefix, precision, out, err, n):
        if self.load_status:
            err.value = b"load failed"
            return self.load_status
        out._obj.value = MODEL
        return 0

    def tessl_qwen35_free(self, h):
        self._rec("tessl_qwen35_free", h.value)
        return self.free_status

    def tessl_qwen35_param_count(self, h, count, err, n):
        count._obj.value = 1
        return 0

    def tessl_qwen35_param_info(self, h, i, info, err, n):
        info._obj.name = b"norm.weight"
        info._obj.ndim = 1
        info._obj.shape[0] = 4
        return 0

    def tessl_qwen35_adamw_free(self, h, err, n):
        self._rec("tessl_qwen35_adamw_free", h)
        err.value = b"boom"
        return self.status

    def tessl_qwen35_adamw_init(self, h, update, seed, moments, err, n):
        self._rec("tessl_qwen35_adamw_init", h)
        return 0

    def tessl_gdn_train_forward(self, rt, args, err, n):
        self._rec("tessl_gdn_train_forward", rt)
        err.value = b"boom"
        return self.status


class HostCase(unittest.TestCase):
    def setUp(self):
        self.lib = FakeLib()
        patch = mock.patch.object(tessl_torch, "_load", return_value=self.lib)
        patch.start()
        self.addCleanup(patch.stop)
        # A fresh thread-local runtime cache for each test.
        local = mock.patch.object(tessl_torch, "_local", threading.local())
        local.start()
        self.addCleanup(local.stop)

    def model(self):
        return tessl_torch.Qwen35("m.safetensors", "c.json")

    def calls(self, name):
        return [c for c in self.lib.calls if c[0] == name]


class Ownership(HostCase):
    def test_copy_deepcopy_and_pickle_are_refused_and_each_handle_is_freed_once(self):
        model = self.model()
        runtime = tessl_torch._runtime()
        for obj in (model, runtime):
            with self.assertRaises(TypeError):
                copy.copy(obj)
            with self.assertRaises(TypeError):
                copy.deepcopy(obj)
            with self.assertRaises(TypeError):
                pickle.dumps(obj)
            obj.close()
            obj.close()
            del obj
        del model, runtime
        self.assertEqual(self.lib.frees(), [("tessl_qwen35_free", MODEL), ("tessl_runtime_free", RUNTIME)])

    def test_dropping_an_unclosed_object_frees_it_once(self):
        model = self.model()
        del model
        self.assertEqual(self.calls("tessl_qwen35_free"), [("tessl_qwen35_free", MODEL)])

    def test_context_manager_closes_and_a_closed_model_never_reaches_the_library(self):
        with self.model() as model:
            self.assertFalse(model.closed)
        self.assertTrue(model.closed)
        before = len(self.lib.calls)
        with self.assertRaisesRegex(TesslError, "closed"):
            model.adamw_free()
        self.assertEqual(len(self.lib.calls), before)

    def test_a_closed_runtime_is_replaced_on_the_next_call(self):
        first = tessl_torch._runtime()
        first.close()
        second = tessl_torch._runtime()
        self.assertIsNot(first, second)
        self.assertFalse(second.closed)


class AsyncEncode(HostCase):
    def test_a_new_runtime_turns_async_encode_on_before_anything_else(self):
        tessl_torch._runtime()
        self.assertEqual([c[0] for c in self.lib.calls[:2]], ["tessl_runtime_new", "tessl_runtime_set_async_encode"])
        self.assertEqual(self.calls("tessl_runtime_set_async_encode"), [("tessl_runtime_set_async_encode", 1)])

    def test_a_refused_switch_raises_and_frees_the_runtime(self):
        self.lib.async_status = 1
        with self.assertRaisesRegex(TesslError, "async refused"):
            tessl_torch._runtime()
        self.assertEqual(self.lib.frees(), [("tessl_runtime_free", RUNTIME)])


class OneParameter(HostCase):
    """read_parameter and write_parameter are the per-entry host entries, and
    touch nothing else: no whole-model copy, no wait on torch's queue."""

    def setUp(self):
        super().setUp()
        self.model_ = self.model()
        sync = mock.patch.object(torch.mps, "synchronize")
        self.sync = sync.start()
        self.addCleanup(sync.stop)

    def test_a_read_is_one_entry_in_its_shape_with_no_whole_model_copy(self):
        t = self.model_.read_parameter("norm.weight")
        self.assertEqual(t.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.calls("tessl_qwen35_read_parameter_host"), [("tessl_qwen35_read_parameter_host", (0, 4))])
        self.assertEqual(self.calls("tessl_qwen35_copy"), [])
        self.sync.assert_not_called()

    def test_a_write_is_one_entry_and_checks_the_shape_first(self):
        self.model_.write_parameter("norm.weight", torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.bfloat16))
        self.assertEqual(self.lib.written, [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(self.calls("tessl_qwen35_write_parameter_host"), [("tessl_qwen35_write_parameter_host", (0, 4))])
        with self.assertRaisesRegex(TesslError, "must be"):
            self.model_.write_parameter("norm.weight", torch.zeros(3))
        self.assertEqual(len(self.calls("tessl_qwen35_write_parameter_host")), 1)
        self.sync.assert_not_called()

    def test_an_unknown_name_and_a_refusal_are_tessl_errors(self):
        with self.assertRaisesRegex(TesslError, "no parameter named"):
            self.model_.read_parameter("nope")
        with self.assertRaisesRegex(TesslError, "must be a str"):
            self.model_.write_parameter(0, torch.zeros(4))
        self.lib.parameter_status = 1
        with self.assertRaisesRegex(TesslError, "parameter refused"):
            self.model_.read_parameter("norm.weight")
        self.assertFalse(self.model_.closed)


class RaggedStep(HostCase):
    """train_step_ragged hands the library u32 ids and prefix offsets (the
    int32 tensor DataCollatorWithFlattening's cu_seq_lens_q is) and one
    per-sequence slot for each sequence."""

    def test_a_flattened_batch_reaches_the_library_as_u32_with_its_losses_back(self):
        loss, per = self.model().train_step_ragged(
            torch.tensor([[7, 8, 9, 1, 2]]), torch.tensor([0, 3, 5], dtype=torch.int32), operands="bf16", accumulate=True
        )
        self.assertEqual((loss, per), (1.5, [0.25, 0.5]))
        self.assertEqual(self.lib.ragged, ([7, 8, 9, 1, 2], [0, 3, 5], 1, 1, 2))

    def test_a_malformed_argument_is_refused_before_the_library(self):
        m = self.model()
        ids = torch.tensor([1, 2, 3, 4])
        for kw, needle in (
            (dict(cu_seqlens=torch.tensor([[0, 4]])), "cu_seqlens must be 1-D"),
            (dict(cu_seqlens=torch.tensor([0.0, 4.0])), "cu_seqlens must be an integer tensor"),
            (dict(cu_seqlens=torch.tensor([-1, 4])), "cu_seqlens must lie in"),
            (dict(cu_seqlens=[0, 4], accumulate=2), "accumulate"),
            (dict(cu_seqlens=[0, 4], operands="fp8"), "operands"),
        ):
            with self.assertRaisesRegex(TesslError, needle):
                m.train_step_ragged(ids, **kw)
        with self.assertRaisesRegex(TesslError, "ids must be 1-D"):
            m.train_step_ragged(torch.zeros(2, 2, dtype=torch.long), [0, 4])
        self.assertEqual(self.calls("tessl_qwen35_train_step_ragged"), [])


class Panic(HostCase):
    def test_a_panic_status_raises_tesslpanic_and_closes_the_model(self):
        model = self.model()
        self.lib.status = 2
        with self.assertRaises(TesslPanic) as caught:
            model.adamw_free()
        self.assertIn("boom", str(caught.exception))
        self.assertTrue(model.closed)
        self.assertEqual(self.calls("tessl_qwen35_free"), [("tessl_qwen35_free", MODEL)])
        before = len(self.lib.calls)
        with self.assertRaises(TesslError) as later:
            model.adamw_free()
        self.assertNotIsInstance(later.exception, TesslPanic)
        self.assertEqual(len(self.lib.calls), before)
        model.close()
        self.assertEqual(len(self.calls("tessl_qwen35_free")), 1)

    def test_an_ordinary_refusal_is_not_a_panic_and_leaves_the_model_open(self):
        model = self.model()
        self.lib.status = 1
        with self.assertRaises(TesslError) as caught:
            model.adamw_free()
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertFalse(model.closed)

    def test_a_panic_in_a_runtime_call_drops_the_threads_runtime(self):
        rt = tessl_torch._runtime()
        self.lib.status = 2
        with self.assertRaises(TesslPanic):
            tessl_torch._gdn_call("tessl_gdn_train_forward", tessl_torch._GdnArgs())
        self.assertTrue(rt.closed)
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])
        self.assertFalse(tessl_torch._runtime().closed)

    def test_a_load_panic_closes_the_runtime_that_panicked(self):
        self.lib.load_status = 2
        with self.assertRaises(TesslPanic):
            self.model()
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])
        self.assertEqual(self.calls("tessl_qwen35_free"), [])
        self.assertFalse(tessl_torch._runtime().closed)

    def test_a_load_refusal_keeps_the_runtime(self):
        self.lib.load_status = 1
        with self.assertRaises(TesslError) as caught:
            self.model()
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertEqual(self.calls("tessl_runtime_free"), [])

    def test_a_model_panic_also_drops_the_threads_runtime(self):
        model = self.model()
        self.lib.status = 2
        with self.assertRaises(TesslPanic):
            model.adamw_free()
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])
        self.assertTrue(tessl_torch._runtime() is not model._rt)


class Poisoned(HostCase):
    def test_a_model_call_that_poisons_the_runtime_raises_tesslpoisoned_and_closes_everything(self):
        model = self.model()
        other = self.model()
        rt = model._rt
        self.assertIs(other._rt, rt)
        self.lib.status, self.lib.poisoned = 1, 1
        with self.assertRaises(TesslPoisoned) as caught:
            model.adamw_free()
        self.assertIsInstance(caught.exception, TesslError)
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertIn("boom", str(caught.exception))  # names the original error
        self.assertTrue(model.closed and other.closed and rt.closed)
        self.assertEqual(len(self.calls("tessl_qwen35_free")), 2)
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_the_next_call_builds_a_fresh_runtime(self):
        model = self.model()
        self.lib.status, self.lib.poisoned = 1, 1
        with self.assertRaises(TesslPoisoned):
            model.adamw_free()
        self.lib.status, self.lib.poisoned = 0, 0
        fresh = tessl_torch._runtime()
        self.assertFalse(fresh.closed)
        self.assertIsNot(fresh, model._rt)
        self.assertEqual(len(self.calls("tessl_runtime_new")), 2)
        self.assertFalse(self.model().closed)

    def test_a_runtime_call_that_poisons_the_runtime_raises_tesslpoisoned(self):
        rt = tessl_torch._runtime()
        self.lib.status, self.lib.poisoned = 1, 1
        with self.assertRaisesRegex(TesslPoisoned, "boom"):
            tessl_torch._gdn_call("tessl_gdn_train_forward", tessl_torch._GdnArgs())
        self.assertTrue(rt.closed)
        self.assertIsNot(tessl_torch._runtime(), rt)

    def test_a_load_that_poisons_the_runtime_raises_tesslpoisoned(self):
        self.lib.load_status, self.lib.poisoned = 1, 1
        with self.assertRaisesRegex(TesslPoisoned, "load failed"):
            self.model()
        self.assertEqual(self.calls("tessl_runtime_free"), [("tessl_runtime_free", RUNTIME)])

    def test_a_healthy_runtime_keeps_a_plain_error_and_everything_open(self):
        model = self.model()
        self.lib.status, self.lib.poisoned = 1, 0
        with self.assertRaises(TesslError) as caught:
            model.adamw_free()
        self.assertNotIsInstance(caught.exception, TesslPoisoned)
        self.assertFalse(model.closed or model._rt.closed)
        self.assertEqual(self.calls("tessl_runtime_free"), [])

    def test_a_refused_query_leaves_the_original_error(self):
        model = self.model()
        self.lib.status, self.lib.poisoned, self.lib.poison_query_status = 1, 1, 1
        with self.assertRaises(TesslError) as caught:
            model.adamw_free()
        self.assertNotIsInstance(caught.exception, TesslPoisoned)
        self.assertFalse(model.closed)

    def test_a_panic_stays_a_panic_without_the_query(self):
        model = self.model()
        self.lib.status, self.lib.poisoned = 2, 1
        with self.assertRaises(TesslPanic):
            model.adamw_free()
        self.assertEqual(self.calls("tessl_runtime_poisoned"), [])


class Close(HostCase):
    def test_a_panic_from_the_free_raises_tesslpanic_after_closing(self):
        model = self.model()
        self.lib.free_status = 2
        with self.assertRaises(TesslPanic):
            model.close()
        self.assertTrue(model.closed)
        self.lib.free_status = 0
        model.close()  # idempotent: nothing more is freed or raised
        self.assertEqual(len(self.calls("tessl_qwen35_free")), 1)

    def test_a_foreign_thread_free_raises_naming_the_leak(self):
        model = self.model()
        self.lib.free_status = 1
        with self.assertRaisesRegex(TesslError, "leaked") as caught:
            model.close()
        self.assertNotIsInstance(caught.exception, TesslPanic)
        self.assertTrue(model.closed)
        self.lib.free_status = 0

    def test_a_finalizer_warns_instead_of_raising(self):
        model = self.model()
        self.lib.free_status = 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.__del__()
        self.assertTrue(any(issubclass(w.category, ResourceWarning) and "leaked" in str(w.message) for w in caught))


class Ranges(HostCase):
    def test_chunk_is_checked_before_any_tensor_is_touched(self):
        # None for every tensor: any touch of them would be an AttributeError.
        for chunk in (2**32, -1, 2**70):
            with self.assertRaisesRegex(TesslError, "chunk"):
                tessl_torch.cross_entropy_rows(None, None, None, None, chunk=chunk)
            with self.assertRaisesRegex(TesslError, "chunk"):
                tessl_torch.cross_entropy(None, None, None, chunk=chunk)
        with self.assertRaisesRegex(TesslError, "chunk"):
            tessl_torch.cross_entropy_rows(None, None, None, None, chunk=1.5)
        self.assertEqual(self.calls("tessl_runtime_new"), [])

    def test_the_u32_edges_are_accepted_by_the_check(self):
        self.assertEqual(tessl_torch._check_uint("chunk", 0, 32), 0)
        self.assertEqual(tessl_torch._check_uint("chunk", 2**32 - 1, 32), 2**32 - 1)

    def test_bool_is_refused_as_an_integer_but_accepted_as_a_flag(self):
        with self.assertRaisesRegex(TesslError, "chunk"):
            tessl_torch._check_uint("chunk", True, 32)
        self.assertEqual(tessl_torch._flag("accumulate", True), 1)
        self.assertEqual(tessl_torch._flag("accumulate", 0), 0)
        for bad in (2, 2**32, -1, 1.0, None):
            with self.assertRaisesRegex(TesslError, "accumulate"):
                tessl_torch._flag("accumulate", bad)

    def test_a_non_integral_step_is_a_tessl_error(self):
        model = self.model()
        model.adamw_init()
        for step in (float("nan"), 3.9, "3", None, True):
            state = {"step": step, "exp_avg": {}, "exp_avg_sq": {}}
            with self.assertRaisesRegex(TesslError, "step"):
                model.load_adamw_state(state)

    def test_train_backward_accumulate_is_a_flag(self):
        model = self.model()
        with self.assertRaisesRegex(TesslError, "accumulate"):
            model.train_backward(accumulate=2**32)

    def test_float_and_bool_index_tensors_are_refused_and_empty_ones_are_not(self):
        for bad in (torch.tensor([3.5]), torch.tensor([True]), torch.tensor([1.0])):
            with self.assertRaisesRegex(TesslError, "ids must be an integer tensor"):
                tessl_torch._u32(bad, 10, "ids")
        self.assertEqual(tessl_torch._u32(torch.tensor([3, 4]), 10, "ids").t.tolist(), [3, 4])
        self.assertEqual(tessl_torch._u32(torch.as_tensor([]), 10, "ids").t.numel(), 0)
        with self.assertRaisesRegex(TesslError, "ids must be an integer tensor"):
            self.model().train_step(torch.tensor([1.5, 2.5]))

    def test_gdn_dims_are_range_checked(self):
        with self.assertRaisesRegex(TesslError, "seq"):
            tessl_torch._gdn_args(1, 2**32, 1, 1)

    def test_the_adamw_step_count_and_seed_are_range_checked(self):
        model = self.model()
        model.adamw_init()
        state = {"step": 2**64, "exp_avg": {}, "exp_avg_sq": {}}
        with self.assertRaisesRegex(TesslError, "step"):
            model.load_adamw_state(state)
        state["step"] = -1
        with self.assertRaisesRegex(TesslError, "step -1 is negative"):
            model.load_adamw_state(state)
        with self.assertRaisesRegex(TesslError, "seed"):
            self.model().adamw_init(seed=2**64)


if __name__ == "__main__":
    unittest.main()
