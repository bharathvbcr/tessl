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
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tessl_torch  # noqa: E402
from tessl_torch import TesslError, TesslPanic  # noqa: E402

RUNTIME, MODEL = 0xBEEF0000, 0xDEAD0000


class FakeLib:
    """Records every call; ``status`` is what the model's AdamW entry points
    and the GDN entries return."""

    def __init__(self):
        self.calls = []
        self.status = 0

    def _rec(self, name, *args):
        self.calls.append((name, args[0] if args else None))

    def frees(self):
        return [c for c in self.calls if c[0].endswith("_free")]

    def tessl_runtime_new(self, err, n):
        self.calls.append(("tessl_runtime_new", None))
        return RUNTIME

    def tessl_runtime_free(self, h):
        self._rec("tessl_runtime_free", h.value)
        return 0

    def tessl_qwen35_load(self, rt, st, cfg, prefix, precision, out, err, n):
        out._obj.value = MODEL
        return 0

    def tessl_qwen35_free(self, h):
        self._rec("tessl_qwen35_free", h.value)
        return 0

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
