#!/usr/bin/env python3
"""Host-only tests of the Python comparison lanes (numpy only, no GPU, no MLX/torch).

The numeric-parity scorer, the ladder and their contracts are Rust now
(`tests/gemm_parity.rs`). What stays here is what only Python can check: that
`flash_attn_torch_mlx.py` transcribes the kernels' masking rule and FLOP
accounting correctly, and that `gemm_sweep_mlx.py --external-lanes` writes the
files the Rust scorer reads.

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
            "qwen35_gdn_driver_mib_peak": 1536.0, "qwen35_gdn_driver_mib_after": 1024.5})
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

    def test_the_cut_config_keeps_the_leading_layer_types(self):
        snap = gdn_lane.DEFAULT_SNAPSHOT
        if not (snap / "config.json").is_file():
            self.skipTest("no local Qwen3.5-2B-Base snapshot")
        cfg = gdn_lane.text_config(snap, 4)
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(list(cfg.layer_types), ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual((cfg.hidden_size, cfg.linear_num_value_heads, cfg.linear_value_head_dim), (2048, 16, 128))


if __name__ == "__main__":
    unittest.main()
