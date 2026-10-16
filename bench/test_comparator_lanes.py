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


if __name__ == "__main__":
    unittest.main()
