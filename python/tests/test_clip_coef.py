"""``tessl_torch.clip_coef`` (Rust's ``tessl_clip_coef``) against torch's own
``clip_grad_norm_`` coefficient, bit for bit.

torch computes ``torch.clamp(max_norm / (total_norm_f32 + 1e-6), max=1.0)``, and
``max_norm / tensor`` is ``tensor.reciprocal() * max_norm`` (``Tensor.__rtruediv__``),
which is not an f32 division: about one input in eight differs in the last bit.
The Rust function is that arithmetic; this sweep is the oracle for it (the
golden values in ``tests/capi_contract.rs`` are the Rust-side pin).

    cargo build --release
    python3 -m unittest python/tests/test_clip_coef.py -v
"""

import math
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import tessl_torch  # noqa: E402
import torch  # noqa: E402

LIB_BUILT = tessl_torch.library_path().exists()


def torch_coef(total_norm, max_norm):
    norm = torch.tensor(total_norm, dtype=torch.float32)
    return torch.clamp(max_norm / (norm + 1e-6), max=1.0).item()


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


@unittest.skipUnless(LIB_BUILT, "libtessl.dylib is not built: run `cargo build --release`")
class ClipCoef(unittest.TestCase):
    def test_a_sweep_of_norms_and_bounds_is_torchs_coefficient_bit_for_bit(self):
        rng = np.random.default_rng(0)
        n = 100_000
        norms = np.concatenate([10.0 ** rng.uniform(-8, 8, n), rng.uniform(0, 5, n)])
        bounds = np.concatenate([10.0 ** rng.uniform(-3, 3, n), rng.choice([0.5, 1.0, 2.0, 10.0], n)])
        differing_from_a_division = 0
        for t, m in zip(norms.tolist(), bounds.tolist()):
            want = torch_coef(t, m)
            got = tessl_torch.clip_coef(t, m)
            if bits(got) != bits(want):
                self.fail(f"clip_coef({t!r}, {m!r}) = {got!r}, torch says {want!r}")
            with np.errstate(all="ignore"):
                quotient = min(np.float32(m) / (np.float32(t) + np.float32(1e-6)), np.float32(1.0))
            differing_from_a_division += float(quotient) != want
        # The sweep can tell torch's formula from a plain division.
        self.assertGreater(differing_from_a_division, n // 20)

    def test_the_edges_are_torchs(self):
        for t, m in [
            (0.0, 1.0), (1e-6, 1.0), (0.1, 1.0), (1.0, 1.0), (3.0, 1.0), (1e30, 1.0), (4.0, 0.0), (5.0, 1e-9),
        ]:
            self.assertEqual(bits(tessl_torch.clip_coef(t, m)), bits(torch_coef(t, m)), (t, m))
        self.assertEqual(tessl_torch.clip_coef(0.1, 1.0), 1.0)
        self.assertEqual(tessl_torch.clip_coef(float("inf"), 1.0), 0.0)
        self.assertTrue(math.isnan(tessl_torch.clip_coef(float("nan"), 1.0)))
        self.assertTrue(math.isnan(torch_coef(float("nan"), 1.0)))

    def test_ints_and_zero_dimensional_tensors_are_numbers(self):
        self.assertEqual(tessl_torch.clip_coef(3, 1), tessl_torch.clip_coef(3.0, 1.0))
        self.assertEqual(tessl_torch.clip_coef(torch.tensor(3.0), 1.0), tessl_torch.clip_coef(3.0, 1.0))
        with self.assertRaises((TypeError, ValueError)):
            tessl_torch.clip_coef("3", [1.0])


if __name__ == "__main__":
    unittest.main()
