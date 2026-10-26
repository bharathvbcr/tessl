#!/usr/bin/env python3
"""Host-only tests of the Python comparison lanes (no GPU, no MLX).

Needs numpy, plus torch and transformers for the CPU contracts of the torch
lanes (a tiny Qwen3.5, no device). The pins are
`tools/msl_emu/requirements.txt`, which CI installs before this file.

The numeric-parity scorer, the ladder and their contracts are Rust now
(`tests/gemm_parity.rs`). What stays here is what only Python can check: that
`flash_attn_torch_mlx.py` transcribes the kernels' masking rule and FLOP
accounting correctly, that `gemm_sweep_mlx.py --external-lanes` writes the
files the Rust scorer reads, and that the torch lane scripts keep the
argument, metric and CPU-step contracts `bench_paired` reads.

  python3 -I bench/test_comparator_lanes.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import embedgemma2_torch as eg2_lane  # noqa: E402
import gemm_sweep_mlx  # noqa: E402
import qwen35_gdn_patch_torch as gdn_lane  # noqa: E402
from flash_attn_torch_mlx import BY_LABEL, CFGS, keep_mask, live_pairs  # noqa: E402


class AttentionSemantics(unittest.TestCase):
    """The masking rule and GQA mapping the Python lane transcribes.

    If these were wrong, torch and MLX would be wrong in exactly the same way as
    each other and a speed ratio would still look plausible, which is why the
    parity run scores the comparison runtimes against the f64 reference too, and
    why the rule itself is checked here against hand-computed cases.
    """

    def test_global_causal_row_t_keeps_t_plus_1_keys(self):
        m = keep_mask(dict(tq=6, tkv=6, window=-1, q_off=0, kv_off=0))
        self.assertEqual(list(m.sum(axis=1)), [1, 2, 3, 4, 5, 6])

    def test_sliding_window_row_t_keeps_min_t_plus_1_w_keys(self):
        m = keep_mask(dict(tq=6, tkv=6, window=3, q_off=0, kv_off=0))
        self.assertEqual(list(m.sum(axis=1)), [1, 2, 3, 3, 3, 3])

    def test_offset_decode_window_lands_on_the_last_w_keys(self):
        # Decode with an offset: q_abs = q_off, so a window of w keeps the last w.
        m = keep_mask(dict(tq=1, tkv=100, window=8, q_off=99, kv_off=0))
        self.assertEqual(m.sum(), 8)
        self.assertTrue(m[0, 92] and m[0, 99] and not m[0, 91])

    def test_a_window_wider_than_the_history_is_inert(self):
        a = keep_mask(dict(tq=8, tkv=8, window=999, q_off=0, kv_off=0))
        b = keep_mask(dict(tq=8, tkv=8, window=-1, q_off=0, kv_off=0))
        self.assertTrue((a == b).all())

    def test_no_shipped_config_masks_a_row_completely(self):
        # A fully masked row is a legitimate decode state the kernels answer with
        # zeros, but torch SDPA returns NaN for it, so a config that hit it would
        # compare noise.
        empty = [c["label"] for c in CFGS if (keep_mask(c).sum(axis=1) == 0).any()]
        self.assertEqual(empty, [])

    def test_live_pair_counts_match_the_rust_flop_accounting(self):
        # Or the two sides report GFLOP/s against different denominators.
        want = {"swa128_prefill_512": 131328, "swa128_decode_1k": 1024,
                "global512_decode_4k": 4096, "swa128_prefill_2048": 1573376}
        got = {k: live_pairs(BY_LABEL[k]) for k in want}
        self.assertEqual(got, want)


M, N, K = 16, 12, 8


def exact(a, b, dtype):
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32), "f32"


def make_dump(root, seeds=3):
    """A dump as `bench_gemm_sweep --dump-parity` leaves it, without the lanes."""
    names = []
    for i in range(seeds):
        sd = os.path.join(root, f"seed_{i:02}")
        os.makedirs(sd)
        names.append(f"seed_{i:02}")
        rng = np.random.default_rng(100 + i)
        np.save(os.path.join(sd, "parity_a.npy"), rng.uniform(-1, 1, (M, K)).astype(np.float32))
        np.save(os.path.join(sd, "parity_b.npy"), rng.uniform(-1, 1, (K, N)).astype(np.float32))
    with open(os.path.join(root, "parity_manifest.json"), "w") as f:
        json.dump(dict(shape="synthetic", dist="uniform", m=M, n=N, k=K,
                       lanes=["tensorops-f32"], seeds=names), f)
    return root


def write(root, lanes):
    with contextlib.redirect_stderr(io.StringIO()):
        gemm_sweep_mlx.write_external_lanes(root, lanes)


def read_lanes(root):
    with open(os.path.join(root, "external_lanes.json")) as f:
        return json.load(f)["lanes"]


class ExternalLaneWriter(unittest.TestCase):
    """`--external-lanes DIR`: what the Rust scorer reads."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_dump(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_scored_lanes_write_one_npy_per_seed_and_record_the_output_dtype(self):
        write(self.root, [("mlx-f32", exact, "f32"), ("torch-mps-f32", exact, "f32")])
        lanes = read_lanes(self.root)
        self.assertEqual([e["lane"] for e in lanes], ["mlx-f32", "torch-mps-f32"])
        for e in lanes:
            self.assertEqual(e["dtype"], "f32")
            self.assertEqual(e["seeds"], [dict(seed=f"seed_{i:02}", out_dtype="f32") for i in range(3)])
        for i in range(3):
            c = np.load(os.path.join(self.root, f"seed_{i:02}", "parity_c_mlx-f32.npy"))
            self.assertEqual(c.shape, (M, N))
            self.assertEqual(c.dtype, np.float32)

    def test_the_dtype_a_runtime_returns_is_recorded_not_assumed(self):
        def bf16_like(a, b, dtype):
            c, _ = exact(a, b, dtype)
            return c, "bf16"
        write(self.root, [("mlx-bf16", bf16_like, "bf16")])
        entry = read_lanes(self.root)[0]
        self.assertEqual(entry["dtype"], "bf16")
        self.assertTrue(all(s["out_dtype"] == "bf16" for s in entry["seeds"]))

    def test_a_lane_that_raises_is_skipped_with_its_reason_and_the_others_still_run(self):
        def boom(a, b, dtype):
            raise RuntimeError("stub boom")
        write(self.root, [("mlx-f32", boom, "f32"), ("torch-mps-f32", exact, "f32")])
        lanes = read_lanes(self.root)
        self.assertEqual(lanes[0]["seeds"], [dict(seed=f"seed_{i:02}", skipped="stub boom") for i in range(3)])
        self.assertTrue(all("out_dtype" in s for s in lanes[1]["seeds"]))
        # A skipped (lane, seed) has no output file, so none can be scored by accident.
        self.assertFalse(os.path.exists(os.path.join(self.root, "seed_00", "parity_c_mlx-f32.npy")))

    def test_a_lane_that_fails_on_one_seed_only_is_recorded_per_seed(self):
        calls = {"n": 0}

        def flaky(a, b, dtype):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("stub seed-1 failure")
            return exact(a, b, dtype)
        write(self.root, [("mlx-f32", flaky, "f32")])
        seeds = read_lanes(self.root)[0]["seeds"]
        self.assertEqual([("skipped" in s) for s in seeds], [False, True, False])

    def test_two_entries_with_one_name_stay_two_entries_for_the_scorer_to_refuse(self):
        write(self.root, [("mlx-f32", exact, "f32"), ("mlx-f32", exact, "f32")])
        lanes = read_lanes(self.root)
        self.assertEqual(len(lanes), 2)
        self.assertEqual([len(e["seeds"]) for e in lanes], [3, 3])

    def test_a_dump_with_no_manifest_is_refused_and_writes_nothing(self):
        os.remove(os.path.join(self.root, "parity_manifest.json"))
        with self.assertRaises(SystemExit) as cm:
            write(self.root, [("mlx-f32", exact, "f32")])
        self.assertIn("parity_manifest.json", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.root, "external_lanes.json")))

    def test_a_dump_whose_operands_are_unreadable_leaves_no_manifest_of_lanes(self):
        os.remove(os.path.join(self.root, "seed_01", "parity_b.npy"))
        with self.assertRaises(SystemExit) as cm:
            write(self.root, [("mlx-f32", exact, "f32")])
        self.assertIn("seed_01", str(cm.exception))
        # Written last: a run that died halfway leaves none, and the scorer then
        # says that no comparison runtime was scored.
        self.assertFalse(os.path.exists(os.path.join(self.root, "external_lanes.json")))

    def test_a_stale_external_manifest_from_an_earlier_run_is_not_left_standing(self):
        write(self.root, [("mlx-f32", exact, "f32")])
        os.remove(os.path.join(self.root, "seed_02", "parity_a.npy"))
        with self.assertRaises(SystemExit):
            write(self.root, [("mlx-f32", exact, "f32")])
        self.assertFalse(os.path.exists(os.path.join(self.root, "external_lanes.json")))

    def test_the_default_lane_list_is_the_four_runtimes_the_scorer_knows(self):
        names = [lane for lane, _, _ in gemm_sweep_mlx.EXTERNAL_LANES]
        self.assertEqual(names, ["mlx-f32", "mlx-bf16", "torch-mps-f32", "torch-mps-bf16"])


class Qwen35GdnPatchLane(unittest.TestCase):
    """`qwen35_gdn_patch_torch.py` without MPS: its arguments, the METRIC lines
    `bench_paired --format metrics` reads, and the bound it applies to a patched
    step's gradients (scored here on synthetic CPU tensors)."""

    def test_exactly_one_of_arm_and_check(self):
        for argv in ([], ["--arm", "torch", "--check"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gdn_lane.parse_args(argv)
        self.assertEqual(gdn_lane.parse_args(["--arm", "tessl"]).arm, "tessl")
        self.assertTrue(gdn_lane.parse_args(["--check"]).check)

    def test_defaults_and_the_bench_env(self):
        old = {k: os.environ.pop(k, None) for k in ("BENCH_ITERS", "BENCH_WARMUP")}
        try:
            a = gdn_lane.parse_args(["--arm", "torch"])
            self.assertEqual((a.seq_len, a.layers, a.steps, a.warmup, a.init), (2048, 4, 5, 2, "real"))
            os.environ["BENCH_ITERS"], os.environ["BENCH_WARMUP"] = "7", "1"
            a = gdn_lane.parse_args(["--arm", "torch", "--seq-len", "512"])
            self.assertEqual((a.seq_len, a.steps, a.warmup), (512, 7, 1))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_nonsense_counts_are_refused(self):
        for bad in (["--steps", "0"], ["--seq-len", "0"], ["--layers", "0"], ["--warmup", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gdn_lane.parse_args(["--arm", "torch"] + bad)

    def test_metric_lines_are_name_value_pairs_with_the_min_and_median(self):
        lines = gdn_lane.metric_lines([30.0, 10.0, 20.0, 40.0], 1536.0, 1024.5)
        parsed = {}
        for line in lines:
            tag, name, value = line.split(" ")
            self.assertEqual(tag, "METRIC")
            parsed[name] = float(value)
        self.assertEqual(parsed, {
            "qwen35_gdn_step_ms_min": 10.0, "qwen35_gdn_step_ms_median": 25.0,
            "qwen35_gdn_tensor_mib_peak": 1536.0, "qwen35_gdn_tensor_mib_after": 1024.5})
        self.assertFalse([name for name in parsed if "driver" in name], "driver memory is not gated")
        with self.assertRaises(ValueError):
            gdn_lane.metric_lines([], 1.0, 1.0)

    def grads(self):
        import torch
        g = torch.Generator().manual_seed(0)
        return {"a": torch.randn(8, 4, generator=g), "b": torch.randn(16, generator=g)}

    def test_equal_gradients_pass_and_the_bound_is_relative_to_the_peak(self):
        want = self.grads()
        ok, _, err, why = gdn_lane.compare_grads(1.0, 1.0, {k: v.clone() for k, v in want.items()}, want)
        self.assertTrue(ok, why)
        self.assertEqual(err, 0.0)
        peak = want["b"].abs().max()
        near = {k: v.clone() for k, v in want.items()}
        near["b"][0] += 0.5 * gdn_lane.GRAD_REL * peak
        ok, name, err, _ = gdn_lane.compare_grads(1.0, 1.0, near, want)
        self.assertTrue(ok)
        self.assertEqual(name, "b")
        self.assertAlmostEqual(err, 0.5 * gdn_lane.GRAD_REL, delta=1e-6)
        far = {k: v.clone() for k, v in want.items()}
        far["b"][0] += 2 * gdn_lane.GRAD_REL * peak
        ok, name, err, why = gdn_lane.compare_grads(1.0, 1.0, far, want)
        self.assertFalse(ok)
        self.assertIn("b: gradient error", why[0])

    def test_loss_missing_extra_and_non_finite_gradients_each_fail(self):
        import torch
        want = self.grads()

        def clone():
            return {k: v.clone() for k, v in want.items()}

        ok, _, _, why = gdn_lane.compare_grads(1.0 + 2e-4, 1.0, clone(), want)
        self.assertFalse(ok)
        self.assertIn("loss", why[0])
        self.assertTrue(gdn_lane.compare_grads(1.0 + 5e-5, 1.0, clone(), want)[0])
        self.assertFalse(gdn_lane.compare_grads(float("nan"), 1.0, clone(), want)[0])
        short = clone()
        del short["a"]
        self.assertIn("a: no gradient in the patched model", gdn_lane.compare_grads(1.0, 1.0, short, want)[3])
        extra = clone()
        extra["c"] = torch.zeros(1)
        self.assertIn("c: no gradient in the unpatched model", gdn_lane.compare_grads(1.0, 1.0, extra, want)[3])
        bad = clone()
        bad["a"][0, 0] = float("nan")
        self.assertIn("a: non-finite gradient", gdn_lane.compare_grads(1.0, 1.0, bad, want)[3])

    def test_a_train_step_and_the_timing_loop_run_on_cpu(self):
        # A tiny Qwen3.5 (linear then full attention) through the lane's own step.
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        cfg = mq.Qwen3_5TextConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            vocab_size=64, tie_word_embeddings=True)
        torch.manual_seed(0)
        model = mq.Qwen3_5ForCausalLM(cfg).float()
        ids = torch.randint(0, 64, (1, 24))
        cpu = torch.device("cpu")
        ms = gdn_lane.time_steps(model, ids, cpu, warmup=1, steps=3)
        self.assertEqual(len(ms), 3)
        self.assertTrue(all(m > 0 for m in ms))
        # The same step twice is the same loss and the same gradients: what
        # --check relies on when only the GDN function differs.
        a = gdn_lane.train_step(model, ids, cpu)
        ga = gdn_lane.grads_of(model)
        b = gdn_lane.train_step(model, ids, cpu)
        self.assertEqual(a, b)
        self.assertTrue(gdn_lane.compare_grads(b, a, gdn_lane.grads_of(model), ga)[0])

    def fake_tessl(self, binds_early=False):
        """A stand-in for tessl_torch whose patch installs the module's current
        ``chunk_gated_delta_rule`` into a stand-in transformers module, as the
        real patch does by global lookup; ``binds_early`` keeps a reference taken
        before any spy, as a patch that bound at import would."""
        import types
        tessl = types.SimpleNamespace(chunk_gated_delta_rule=lambda *a, **k: "tessl")
        early = tessl.chunk_gated_delta_rule
        mq = types.SimpleNamespace(torch_chunk_gated_delta_rule=lambda *a, **k: "torch")

        def patch():
            previous = mq.torch_chunk_gated_delta_rule
            mq.torch_chunk_gated_delta_rule = early if binds_early else tessl.chunk_gated_delta_rule
            return previous

        tessl.patch_transformers_qwen3_5 = patch
        return tessl, mq

    def forward(self, mq, layers=3):
        for _ in range(layers):
            mq.torch_chunk_gated_delta_rule()

    def test_the_counter_sees_every_call_the_patch_routes_to_tessl(self):
        tessl, mq = self.fake_tessl()
        counter, original = gdn_lane.spy_on(tessl)
        self.forward(mq)
        self.assertEqual(counter.calls, 0)
        self.assertIsNone(gdn_lane.count_error(counter.calls, 0, "unpatched"))
        previous = tessl.patch_transformers_qwen3_5()
        self.forward(mq)
        self.assertEqual(counter.calls, 3)
        self.assertIsNone(gdn_lane.count_error(counter.calls, 3, "patched"))
        self.assertEqual(mq.torch_chunk_gated_delta_rule(), "tessl")
        self.assertEqual(previous(), "torch")
        setattr(tessl, "chunk_gated_delta_rule", original)
        self.assertIs(tessl.chunk_gated_delta_rule, original)

    def test_a_patch_that_takes_no_effect_is_an_error_not_a_pass(self):
        # The patch installed something the counter never sees (a binding made
        # before the spy): the model still runs, but zero calls reach tessl.
        tessl, mq = self.fake_tessl(binds_early=True)
        counter, _ = gdn_lane.spy_on(tessl)
        tessl.patch_transformers_qwen3_5()
        self.forward(mq)
        self.assertEqual(counter.calls, 0)
        msg = gdn_lane.count_error(counter.calls, 3, "patched step")
        self.assertIn("patch is not taking effect", msg)
        self.assertIn("0 tessl GDN calls", msg)

    def test_the_count_bounds(self):
        self.assertIsNone(gdn_lane.count_error(9, 9, "x"))
        self.assertIsNone(gdn_lane.count_error(12, 9, "x"))
        self.assertIn("expected at least 9", gdn_lane.count_error(8, 9, "x"))
        self.assertIn("expected 0", gdn_lane.count_error(1, 0, "unpatched step"))

    def test_gdn_layers_counts_the_linear_layers_of_the_cut(self):
        import types
        cfg = types.SimpleNamespace(layer_types=["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(gdn_lane.gdn_layers(types.SimpleNamespace(config=cfg)), 3)

    def test_the_sliced_rel_err_is_the_whole_tensor_rel_err(self):
        import torch
        g = torch.Generator().manual_seed(3)
        want = torch.randn(1000, generator=g)
        got = want + 1e-3 * torch.randn(1000, generator=g)
        whole = ((got.double() - want.double()).abs().max() / want.double().abs().max()).item()
        for n in (1, 7, 999, 1000, 5000):
            self.assertEqual(gdn_lane.rel_err(got, want, slice_elems=n), whole)
        self.assertEqual(gdn_lane.rel_err(torch.zeros(4), torch.zeros(4)), 0.0)
        self.assertEqual(gdn_lane.rel_err(torch.ones(2, 3), torch.ones(2, 3), slice_elems=2), 0.0)

    def test_host_copy_owns_its_storage_and_grads_of_matches_the_gradients(self):
        import torch
        t = torch.arange(4.0)
        c = gdn_lane.host_copy(t)
        c[0] = 9.0
        self.assertEqual(t[0].item(), 0.0)
        lin = torch.nn.Linear(2, 2)
        lin(torch.ones(1, 2)).sum().backward()
        grads = gdn_lane.grads_of(torch.nn.ModuleDict({"l": lin}))
        self.assertEqual(set(grads), {"l.weight", "l.bias"})
        self.assertTrue(torch.equal(grads["l.bias"], lin.bias.grad))

    def test_phase_log_line_has_the_phase_and_both_resident_sets(self):
        import torch
        err = io.StringIO()
        real = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False  # no MPS call in a device-free test
        try:
            with contextlib.redirect_stderr(err):
                gdn_lane.phase_log("model_built")
        finally:
            torch.backends.mps.is_available = real
        line = err.getvalue().strip()
        self.assertRegex(line, r"^phase=model_built rss_mib=\d+ rss_max_mib=\d+$")

    def test_real_weights_load_straight_into_the_cut_model_and_skip_later_layers(self):
        import tempfile
        import types
        import torch
        from safetensors.torch import save_file
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        cfg = mq.Qwen3_5TextConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            vocab_size=64, tie_word_embeddings=True)
        model = mq.Qwen3_5ForCausalLM(cfg).float()
        g = torch.Generator().manual_seed(5)
        tensors = {}
        for name, p in model.state_dict().items():
            if name.startswith("model.") and "rotary" not in name:
                tensors[gdn_lane._PREFIX + name[len("model."):]] = torch.randn(p.shape, generator=g).bfloat16()
        late = gdn_lane._PREFIX + "layers.7.mlp.down_proj.weight"
        tensors[late] = torch.zeros(1)
        tensors["model.visual.x"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors-00001-of-00001.safetensors")
            save_file(tensors, path)
            gdn_lane.load_real_weights(model, d, 2)
            for name, p in model.state_dict().items():
                key = gdn_lane._PREFIX + name[len("model."):]
                if key in tensors:
                    self.assertTrue(torch.equal(p, tensors[key].float()), name)
            del tensors[gdn_lane._PREFIX + "norm.weight"]
            save_file(tensors, path)
            with self.assertRaises(SystemExit) as cm:
                gdn_lane.load_real_weights(model, d, 2)
            self.assertIn("model.norm.weight", str(cm.exception))

    def test_the_cut_config_keeps_the_leading_layer_types(self):
        snap = gdn_lane.DEFAULT_SNAPSHOT
        if not (snap / "config.json").is_file():
            self.skipTest("no local Qwen3.5-2B-Base snapshot")
        cfg = gdn_lane.text_config(snap, 4)
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(list(cfg.layer_types), ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual((cfg.hidden_size, cfg.linear_num_value_heads, cfg.linear_value_head_dim), (2048, 16, 128))


import prefill_lane as pl  # noqa: E402
import qwen35_prefill_mlx as mlx_prefill  # noqa: E402
import qwen35_prefill_torch as torch_prefill  # noqa: E402

REPO = os.path.dirname(HERE)


def read_text(path):
    with open(path) as f:
        return f.read()


def tiny_ids(n=300, vocab=1000, answers=(3, 4, 5), lengths=(100, 200)):
    ids = [(i * 7919 + 13) % vocab for i in range(n)]
    return {"schema": pl.SCHEMA, "vocab": vocab, "ids": ids, "answers": list(answers),
            "digests": {str(t): pl.fnv1a32(ids[:t]) for t in lengths}}


def write_json(d, doc, name="ids.json"):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


class PrefillIdsContract(unittest.TestCase):
    """One ids file, read by tessl, torch and MLX alike; a mismatch is an error, not a shorter run."""

    def test_the_committed_file_verifies_at_every_head_to_head_length(self):
        doc = pl.load_ids(pl.DEFAULT_IDS)
        self.assertEqual((doc["vocab"], len(doc["answers"]), len(doc["ids"])), (248320, 17, 8192))
        for t in pl.LENGTHS:
            self.assertEqual(len(pl.prefix(doc, t)), t)

    def test_both_python_lanes_default_to_the_committed_file(self):
        for mod in (torch_prefill, mlx_prefill):
            args = mod.build_parser().parse_args([])
            self.assertEqual(os.path.realpath(args.ids), os.path.realpath(
                os.path.join(REPO, "tests", "fixtures", "qwen35_prefill_ids.json")))
            self.assertEqual(args.lengths, [200, 1024, 2048, 8192])

    def test_fnv_vectors_match_the_rust_reader(self):
        vectors = {(): 2166136261, (1, 2, 3): 2034659765, (248319,): 3941604598}
        for ids, want in vectors.items():
            self.assertEqual(pl.fnv1a32(list(ids)), want)
        src = read_text(os.path.join(REPO, "src", "bin", "common", "prefill_ids.rs"))
        for want in vectors.values():
            self.assertIn(f"{want:_}", src)

    def test_a_file_shorter_than_the_length_is_refused(self):
        doc = tiny_ids(n=150, lengths=(100,))
        doc["digests"]["200"] = 0
        with self.assertRaises(SystemExit) as cm:
            pl.prefix(doc, 200)
        self.assertIn("150 ids", str(cm.exception))

    def test_a_length_without_a_digest_or_with_a_wrong_one_is_refused(self):
        doc = tiny_ids()
        with self.assertRaises(SystemExit) as cm:
            pl.prefix(doc, 150)
        self.assertIn("no digest", str(cm.exception))
        doc["digests"]["100"] ^= 1
        with self.assertRaises(SystemExit) as cm:
            pl.prefix(doc, 100)
        self.assertIn("digest", str(cm.exception))

    def test_a_token_outside_the_vocabulary_or_a_wrong_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            doc = tiny_ids()
            doc["ids"][5] = 1000
            with self.assertRaises(SystemExit):
                pl.load_ids(write_json(d, doc))
            doc = tiny_ids()
            doc["schema"] = "other"
            with self.assertRaises(SystemExit):
                pl.load_ids(write_json(d, doc))
            doc = tiny_ids()
            doc["answers"] = []
            with self.assertRaises(SystemExit):
                pl.load_ids(write_json(d, doc))

    def test_prepare_returns_the_same_prefix_for_every_lane(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_json(d, tiny_ids())
            outs = []
            for mod in (torch_prefill, mlx_prefill):
                args = mod.build_parser().parse_args(["--ids", path, "--lengths", "100,200"])
                _, ids, digests = pl.prepare(args)
                outs.append((ids, digests))
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(outs[0][0][100], tiny_ids()["ids"][:100])

    def test_the_rust_lane_reads_the_same_file_and_flags(self):
        src = read_text(os.path.join(REPO, "src", "bin", "bench_qwen35_layers.rs"))
        for needle in ('"--prefill-score"', 'strip_prefix("--ids=")', "prefill_ids::PrefillIds::load",
                       "file.prefix(t)", "s.tokens() as usize != t", "prefill_score_t{t}_median"):
            self.assertIn(needle, src)


class PrefillArgs(unittest.TestCase):
    def test_defaults_and_overrides(self):
        args = torch_prefill.build_parser().parse_args([])
        self.assertEqual((args.dtype, args.device, args.attn), ("bf16", "gpu", None))
        args = torch_prefill.build_parser().parse_args(
            ["--lengths", "8,16", "--iters", "5", "--warmup", "1", "--dtype", "f32", "--device", "cpu"])
        self.assertEqual((args.lengths, args.iters, args.warmup, args.dtype, args.device),
                         ([8, 16], 5, 1, "f32", "cpu"))

    def test_bad_lengths_are_refused(self):
        for bad in ("0,5", "5,5", "a,b", "-3", ""):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mlx_prefill.build_parser().parse_args(["--lengths", bad])

    def test_iteration_env_and_flags(self):
        ns = type("A", (), {"iters": None, "warmup": None})()
        old = {k: os.environ.pop(k, None) for k in ("BENCH_ITERS", "BENCH_WARMUP")}
        try:
            self.assertEqual(pl.resolve_counts(ns), (9, 3))
            os.environ["BENCH_ITERS"], os.environ["BENCH_WARMUP"] = "5", "0"
            self.assertEqual(pl.resolve_counts(ns), (5, 0))
            for bad in ("0", "05", "-1", "x", ""):
                os.environ["BENCH_ITERS"] = bad
                with self.assertRaises(SystemExit):
                    pl.resolve_counts(ns)
            ns.iters, ns.warmup = 4, 2
            os.environ["BENCH_ITERS"] = "7"
            self.assertEqual(pl.resolve_counts(ns), (4, 2))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


class PrefillOutput(unittest.TestCase):
    """The lane's stdout must be what `bench_paired --format workloads` parses."""

    def lane_output(self, lengths=(100, 200), iters=4, warmup=2, ran=None, lp=None):
        ids = {t: list(range(t)) for t in lengths}
        digests = {t: pl.fnv1a32(ids[t]) for t in lengths}
        order, clock = [], iter(range(1, 10_000))

        def one(t):
            order.append(t)
            return float(next(clock)), [-1.0, -2.0, -3.0] if lp is None else lp, t if ran is None else ran

        samples = pl.run_rounds(list(lengths), iters, warmup, one)
        return order, pl.format_rows("fake", list(lengths), samples, iters, 3, digests)

    def test_lengths_alternate_and_reverse_every_other_round(self):
        order, _ = self.lane_output(lengths=(100, 200, 300), iters=2, warmup=1)
        self.assertEqual(order, [100, 200, 300, 300, 200, 100, 100, 200, 300])

    def test_rows_follow_the_workloads_grammar(self):
        _, rows = self.lane_output()
        line = json.loads(json.dumps(rows))  # what the last stdout line carries
        names = [r["workload"] for r in line]
        self.assertEqual(names, ["prefill_score_t100", "prefill_score_t100_median",
                                 "prefill_score_t200", "prefill_score_t200_median"])
        self.assertEqual(len(set(names)), len(names))
        for r in line:
            self.assertIsInstance(r["workload"], str)
            self.assertGreater(r["ms_min"], 0)
            self.assertEqual(r["iters"], 4)
            self.assertIn(r["tokens"], (100, 200))
            self.assertEqual(r["answers"], 3)
            self.assertEqual(r["ids_fnv1a32"], pl.fnv1a32(list(range(r["tokens"]))))

    def test_min_and_median_are_per_lane_statistics_of_the_timed_rounds_only(self):
        # 100 and 200 alternate; warmup rounds take clock ticks 1..4 and are dropped.
        _, rows = self.lane_output(iters=4, warmup=2)
        by = {r["workload"]: r for r in rows}
        # Even rounds put 100 first (ticks 5, 9), odd rounds put it second (ticks 8, 12).
        self.assertEqual(by["prefill_score_t100"]["ms_min"], 5)
        self.assertEqual(by["prefill_score_t100_median"]["ms_min"], (8 + 9) / 2)
        self.assertEqual(by["prefill_score_t100"]["ms_median"], (8 + 9) / 2)

    def test_a_forward_at_the_wrong_length_aborts_the_lane(self):
        with self.assertRaises(SystemExit) as cm:
            self.lane_output(ran=99)
        self.assertIn("ran on 99 tokens", str(cm.exception))

    def test_non_finite_or_all_zero_scores_abort_the_lane(self):
        for bad in ([float("nan"), -1.0], [0.0, 0.0], []):
            with self.assertRaises(SystemExit):
                self.lane_output(lp=bad)

    def test_equal_sample_counts_are_enforced(self):
        with self.assertRaises(SystemExit):
            pl.format_rows("x", [100], {100: [1.0, 2.0]}, 3, 3, {100: 1})

    def test_emit_prints_one_json_array_as_the_last_line(self):
        _, rows = self.lane_output()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print("log line")
            pl.emit(rows)
        last = [l for l in out.getvalue().splitlines() if l.strip()][-1]
        self.assertEqual(json.loads(last), rows)


class PrefillDocstringCommands(unittest.TestCase):
    """The commands the report quotes must parse under the real parsers."""

    def lane_lines(self):
        lines = [l.strip().rstrip("\\").strip().strip('"') for l in pl.__doc__.splitlines()
                 if l.strip().startswith(('"tessl:', '"torch:', '"mlx:'))]
        self.assertEqual(len(lines), 3)
        return lines

    def test_bench_paired_command_is_an_evidence_run_with_even_rounds(self):
        doc = pl.__doc__
        self.assertIn("target/release/bench_paired --evidence --rounds 6", doc)
        rounds = int(doc.split("--rounds ")[1].split()[0])
        self.assertTrue(rounds >= 4 and rounds % 2 == 0)
        self.assertIn("--out bench/results/", doc)
        self.assertIn("torch 2.15: NOT RUN", doc)

    def test_lane_lines_use_the_bench_paired_grammar_and_the_lane_parsers(self):
        import re
        import shlex
        sub = {"$SNAP": "/snap", "$IDS": "ids.json", "$PY": "python"}
        parsers = {"torch": torch_prefill.build_parser(), "mlx": mlx_prefill.build_parser()}
        names = []
        for line in self.lane_lines():
            m = re.fullmatch(r"(\w+):(BENCH_ITERS=9,BENCH_WARMUP=3)=(.+)", line)
            self.assertIsNotNone(m, line)
            name, _, cmd = m.groups()
            for k, v in sub.items():
                cmd = cmd.replace(k, v)
            argv = shlex.split(cmd)
            names.append(name)
            if name == "tessl":
                self.assertEqual(argv[0], "target/release/bench_qwen35_layers")
                self.assertIn("--prefill-score", argv)
                self.assertIn("--weights=/snap", argv)
                self.assertIn("--ids=ids.json", argv)
                self.assertEqual([a for a in argv if a.isdigit()], ["200", "1024", "2048", "8192"])
            else:
                self.assertTrue(argv[1].endswith(f"qwen35_prefill_{name}.py"))
                args = parsers[name].parse_args(argv[2:])
                self.assertEqual((args.snapshot, args.ids, args.lengths), ("/snap", "ids.json", [200, 1024, 2048, 8192]))
        self.assertEqual(names, ["tessl", "torch", "mlx"])


class Embedgemma2TorchMemory(unittest.TestCase):
    """The memory fields of `embedgemma2_torch.py` without MPS or sentence-transformers:
    `tensor_mib` (live tensor bytes, the metric) and `driver_mib` (recorded only)."""

    MIB = 1 << 20

    def toy(self):
        import torch
        return torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())

    def counter(self, values):
        it = iter(values)
        last = [values[0]]

        def current():
            last[0] = next(it, last[0])
            return last[0]
        return current

    def test_the_tensor_metric_is_the_peak_at_a_module_boundary_above_the_baseline(self):
        import torch
        model = self.toy()
        # The reads in order: the baseline, the sampler's opening read, six module
        # boundaries (Sequential in, Linear in and out, ReLU in and out, Sequential
        # out; the peak 400 is Linear's output), the closing read, the final read.
        current = self.counter([v * self.MIB for v in (100, 100, 100, 250, 400, 300, 200, 150, 150, 150)])
        driver_calls = []
        mem = eg2_lane.memory_pass(
            model, lambda: model(torch.zeros(1, 2)), current,
            lambda: driver_calls.append(1) or 9000 * self.MIB, lambda: None)
        self.assertEqual(mem, {"tensor_mib": 300.0, "tensor_base_mib": 100.0, "driver_mib": 9000.0})
        self.assertEqual(driver_calls, [1])

    def test_peak_sampler_reports_exactly_the_largest_sample_and_removes_its_hooks(self):
        import torch
        model = self.toy()
        n_before = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
        s = eg2_lane.PeakSampler(model, self.counter([5 * self.MIB, 2 * self.MIB, 40 * self.MIB, 7 * self.MIB]))
        with s:
            model(torch.zeros(1, 2))
        self.assertEqual(s.peak, 40 * self.MIB)
        self.assertEqual(sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()), n_before)

    def test_hooks_are_removed_when_the_forward_raises(self):
        import torch
        model = self.toy()
        with self.assertRaises(RuntimeError):
            with eg2_lane.PeakSampler(model, lambda: 0):
                model(torch.zeros(1, 3))  # wrong width
        self.assertEqual(sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()), 0)

    def test_tensor_mib_is_in_mib_above_the_baseline_and_never_negative(self):
        model = self.toy()
        base, peak = 1024 * self.MIB, 1536 * self.MIB
        seq = [base, base, peak, base]
        mem = eg2_lane.memory_pass(model, lambda: None, self.counter(seq), lambda: 0, lambda: None)
        self.assertEqual((mem["tensor_mib"], mem["tensor_base_mib"]), (512.0, 1024.0))
        low = eg2_lane.memory_pass(model, lambda: None, self.counter([2048 * self.MIB, 10 * self.MIB]),
                                   lambda: 0, lambda: None)
        self.assertEqual(low["tensor_mib"], 0.0)

    def test_the_driver_figure_never_moves_the_gated_metric(self):
        model = self.toy()
        a = eg2_lane.memory_pass(model, lambda: None, lambda: 7 * self.MIB, lambda: 1 * self.MIB, lambda: None)
        b = eg2_lane.memory_pass(model, lambda: None, lambda: 7 * self.MIB, lambda: 9000 * self.MIB, lambda: None)
        self.assertEqual(a["tensor_mib"], b["tensor_mib"])
        self.assertEqual((a["driver_mib"], b["driver_mib"]), (1.0, 9000.0))

    def test_the_row_is_what_bench_paired_reads_with_both_memory_fields(self):
        mem = {"tensor_mib": 812.5, "tensor_base_mib": 1990.2, "driver_mib": 4096.0}
        row = eg2_lane.workload_row(" 8x1024 ", "bf16", [2.5, 3.0, 4.0], 10, 1, mem)
        parsed = json.loads(json.dumps([row]))[0]
        self.assertEqual(parsed["workload"], "8x1024")
        self.assertEqual(parsed["backend"], "torch-mps-bf16")
        self.assertEqual((parsed["ms_min"], parsed["ms_median"], parsed["iters"], parsed["forwards"]),
                         (2.5, 3.0, 10, 1))
        for k, v in mem.items():
            self.assertEqual(parsed[k], v)
        self.assertGreater(parsed["ms_min"], 0)

    def test_the_script_imports_without_sentence_transformers_or_mps(self):
        self.assertTrue(callable(eg2_lane.main))
        self.assertNotIn("sentence_transformers", sys.modules.get("embedgemma2_torch").__dict__)


if __name__ == "__main__":
    unittest.main()
