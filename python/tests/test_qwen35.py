"""The Python surface after ABI 14 retired tessl's Qwen3.5 training step.

Host-only: no GPU, no checkpoint. The step's behaviour (loss, gradients, AdamW)
is ojas's now (the Tape over ojas-metal, gated against transformers by
`ojas-qwen35/tests/gpu_tape_2b.rs`); what is pinned here is that this binding
no longer offers it, and still offers the parameter table and its copies.
Before ABI 14 these assertions failed: `Qwen35` had every method listed as
retired, and `_ABI_VERSION` was 13.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tessl_torch  # noqa: E402

RETIRED_METHODS = (
    "train_step",
    "train_step_ragged",
    "train_forward",
    "train_backward",
    "train_discard",
    "hidden",
    "grads",
    "grad_sq_norm",
    "adamw_init",
    "adamw_free",
    "adamw_step",
    "adamw_step_count",
    "adamw_state",
    "load_adamw_state",
)

RETIRED_MODULE_NAMES = ("chunk_gated_delta_rule", "patch_transformers_qwen3_5")

KEPT_METHODS = (
    "parameters",
    "load_parameters",
    "read_parameter",
    "write_parameter",
    "describe",
    "shapes",
    "close",
)


class RetiredTrainingSurface(unittest.TestCase):
    def test_the_binding_expects_abi_14(self):
        self.assertEqual(tessl_torch._ABI_VERSION, 14)

    def test_qwen35_offers_no_training_step_gradients_or_adamw(self):
        for name in RETIRED_METHODS:
            self.assertFalse(hasattr(tessl_torch.Qwen35, name), f"Qwen35.{name} is retired")

    def test_the_gated_delta_rule_binding_is_retired(self):
        for name in RETIRED_MODULE_NAMES:
            self.assertFalse(hasattr(tessl_torch, name), f"tessl_torch.{name} is retired")
            self.assertNotIn(name, tessl_torch.__all__)

    def test_the_parameter_table_and_its_copies_stay(self):
        for name in KEPT_METHODS:
            self.assertTrue(hasattr(tessl_torch.Qwen35, name), f"Qwen35.{name} must stay")

    def test_cross_entropy_and_the_clip_coefficient_stay(self):
        for name in ("cross_entropy", "cross_entropy_rows", "clip_coef"):
            self.assertIn(name, tessl_torch.__all__)


if __name__ == "__main__":
    unittest.main()
