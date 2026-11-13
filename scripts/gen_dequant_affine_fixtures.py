#!/usr/bin/env python3
"""Golden fixtures for the MLX-affine dequant kernels (tests/dequant_affine.rs).

Fixture generator only: nothing here ships. Run isolated, with mlx on the GPU
stream (mx.cpu and mx.gpu quantise differently, so the stream is part of the
golden and is passed on every mlx call that defines one):

    /Users/bharath/.venvs/ml/bin/python -I scripts/gen_dequant_affine_fixtures.py

It must be run from a tree where this file is committed and unmodified: the
manifest records the commit that last changed it. Writes
tests/fixtures/dequant_affine/. Cases, all rows=17, cols=1024, for group sizes
32, 64 and 128 and for q8 and q4:

  quant_q8, quant_q4   weights -> mx.quantize(stream=mx.gpu) -> mx.dequantize(stream=mx.gpu)
  crafted_q8, crafted_q4
                       random packed codes with scales/biases drawn from the
                       corner values (zeros, subnormals, +-max, 2^125, a
                       subnormal-result pair) -> mx.dequantize(stream=mx.gpu)

Group size 64 keeps the original file names ({case}_q{bits}_{part}.bin); 32 and
128 are {case}_q{bits}_g{G}_{part}.bin. The weights and the G=64 crafted cases
come from one numpy stream, drawn in a fixed order that must not change (it
keeps the G=64 files byte-identical); each other group size's crafted case draws
from its own stream, default_rng([20261110, G]), over the group-size-64 corners
plus a NaN and both infinities. The script refuses to write a G=32/128 case whose
crafted golden lacks both infinities, a NaN, a maximum-magnitude finite value or a
zero of either sign, or whose device-model tally is not zero mismatches.

MANIFEST.json records the sha256 of every fixture file, so a test can tell the
bytes it loads are the bytes this script wrote.

The golden of every case is mx.dequantize's bf16 output as raw bits, so a test
compares bit patterns (NaN payloads, signed zeros and infinities included).
Host-side rounding to bf16 is done in numpy (round to nearest even), never by
an mlx op, so the only mlx computations are the two that define the goldens.
The host formulas the tallies compare against are the ones the Rust test uses.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import importlib.metadata

import mlx.core as mx
import numpy as np

ROWS, COLS = 17, 1024
GROUP = 64  # the original group size: its files keep their legacy names
OTHER_GROUPS = (32, 128)
SEED = 20261110
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "dequant_affine"
GENERATOR = "scripts/gen_dequant_affine_fixtures.py"
GPU = mx.gpu


def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):
    """float32 -> bf16 bits, round to nearest even; NaN becomes the quiet NaN 0x7fc0."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(x), np.uint16(0x7FC0), rounded).astype(np.uint16)


def bits_of(a):
    """The raw bits of an mlx bf16 array (a reinterpretation, no arithmetic)."""
    return np.array(a.view(mx.uint16)).astype(np.uint16)


def from_bits(bits):
    return mx.array(bits.astype(np.uint16)).view(mx.bfloat16)


def weights(rng):
    w = rng.standard_normal((ROWS, COLS)).astype(np.float32) * 0.02
    w[1] *= 1e3  # large magnitudes
    w[2] *= 1e-3  # small magnitudes
    w[3] = -np.abs(w[3])  # all negative
    w[4] = 0.5  # constant row
    w[5] = 0.0  # zero row
    w[6, :64] = 1e-30  # tiny group
    w[7] = np.linspace(-1.0, 1.0, COLS, dtype=np.float32)
    return w


# bf16 bit patterns that matter. The list is the group-size-64 stream's and its
# length is part of that stream (rng.integers(0, len(CORNERS))): changing it moves
# the group-size-64 files. 0x0000/0x8000 signed zeros; 0x0001 smallest subnormal,
# 0x007F largest, 0x8001 a negative one; 0x0080 smallest normal; 0x3F80/0xBF80 +-1;
# 0x7F7F/0xFF7F +-max finite; 0x7E00 is 2^125 (times a code it overflows f32
# itself, and an exact fma still lands finite against -max), NOT a NaN; 0x4B00 is
# 2^23; 0x3C23 a value near 0.01.
CORNERS = np.array(
    [0x0000, 0x8000, 0x0001, 0x007F, 0x8001, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7E00, 0x4B00, 0x3C23],
    dtype=np.uint16,
)

# The corners of the group sizes added later (32, 128), whose streams are their own:
# CORNERS plus a true NaN (0x7FC0, the quiet NaN: exponent all ones, mantissa non-zero)
# and both infinities (0x7F80, 0xFF80), so NaN and infinity propagation are compared
# against mlx as well.
CORNERS_WITH_NON_FINITE = np.concatenate([CORNERS, np.array([0x7FC0, 0x7F80, 0xFF80], dtype=np.uint16)])

# (scale, bias) pairs of normal operands whose code-1 result is a subnormal:
# 2^-126 - 2^-126 * (1 + 2^-7) = -2^-133, and its mirror and 2^-125 versions.
# Both operands are normal, so only the flush of the RESULT is exercised.
SUBNORMAL_RESULT_PAIRS = [(0x0080, 0x8081), (0x8080, 0x0081), (0x0100, 0x8101), (0x8100, 0x0101)]


def crafted(rng, bits, group, corners):
    words = COLS * bits // 32
    packed = rng.integers(0, 2**32, size=(ROWS, words), dtype=np.uint64).astype(np.uint32)
    n = ROWS * (COLS // group)
    pick = lambda: corners[rng.integers(0, len(corners), size=n)]
    sb, bb = pick(), pick()
    # Half of the groups draw an ordinary random scale/bias instead, so the
    # corner groups are mixed with arithmetic that is not degenerate.
    ordinary = rng.random(n) < 0.5
    rand_s = f32_to_bf16_bits(rng.standard_normal(n).astype(np.float32) * 0.05)
    rand_b = f32_to_bf16_bits(rng.standard_normal(n).astype(np.float32) * 0.5)
    sb = np.where(ordinary, rand_s, sb).astype(np.uint16)
    bb = np.where(ordinary, rand_b, bb).astype(np.uint16)
    for i, (s, b) in enumerate(SUBNORMAL_RESULT_PAIRS):
        sb[i], bb[i] = s, b
    return packed, sb.reshape(ROWS, -1), bb.reshape(ROWS, -1)


def flush_f32(x):
    """The device's f32 subnormal flush: a subnormal becomes a zero of its sign."""
    return np.where((np.abs(x) < np.float32(2.0**-126)) & (x != 0), np.copysign(np.float32(0), x), x).astype(np.float32)


def host_formula(packed, sbits, bbits, bits, group, ftz):
    """The dequant formula as an exact fused multiply-add rounded once to f32
    (the product is exact in f64), then once to bf16. With `ftz`, as the GPU
    runs it: subnormal scale, bias and f32 result flush to a zero of their sign.
    Returns bf16 bits."""
    per = 32 // bits
    shifts = (np.arange(per, dtype=np.uint32) * bits)[None, None, :]
    q = ((packed[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(ROWS, COLS).astype(np.float64)
    s = np.repeat(bf16_bits_to_f32(sbits), group, axis=1)
    b = np.repeat(bf16_bits_to_f32(bbits), group, axis=1)
    if ftz:
        s, b = flush_f32(s), flush_f32(b)
    with np.errstate(all="ignore"):
        r = (s.astype(np.float64) * q + b.astype(np.float64)).astype(np.float32)
    if ftz:
        r = flush_f32(r)
    return f32_to_bf16_bits(r)


def tally(packed, sbits, bbits, obits, bits, group):
    return {
        "ieee_fma_mismatches": int((host_formula(packed, sbits, bbits, bits, group, False) != obits).sum()),
        "ftz_model_mismatches": int((host_formula(packed, sbits, bbits, bits, group, True) != obits).sum()),
    }


def fname(case, bits, group, part):
    """Group size 64 keeps the original names."""
    if group == GROUP:
        return f"{case}_q{bits}_{part}.bin"
    return f"{case}_q{bits}_g{group}_{part}.bin"


def write(name, arr):
    (OUT / name).write_bytes(np.ascontiguousarray(arr).astype("<u4" if arr.dtype == np.uint32 else "<u2").tobytes())


def generator_commit():
    """The commit that last changed this script; refuses a modified or untracked one."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--", GENERATOR):
        sys.exit(f"{GENERATOR} has uncommitted changes; commit it first, the manifest names the commit")
    commit = git("log", "-1", "--format=%H", "--", GENERATOR)
    if len(commit) != 40:
        sys.exit(f"{GENERATOR} is not committed")
    return commit


def notes(group):
    return {
        "packed": "uint32 little-endian, [17, cols * bits / 32], MLX row-major layout",
        "scales": f"bf16 bits, uint16 little-endian, [17, {COLS // group}]",
        "biases": f"bf16 bits, uint16 little-endian, [17, {COLS // group}]",
        "out": f"mx.dequantize(..., group_size={group}, stream=mx.gpu) output as bf16 bits, uint16 little-endian, [17, 1024]",
    }


def write_manifest(commit):
    files = {}
    for group in (GROUP, *OTHER_GROUPS):
        for case in ("quant", "crafted"):
            for bits in (8, 4):
                for part in ("packed", "scales", "biases", "out"):
                    name = fname(case, bits, group, part)
                    role = "golden" if part == "out" else "input"
                    if case == "quant":
                        origin = "from mx.quantize(stream=mx.gpu)"
                    elif group == GROUP:
                        origin = f"crafted corner values, numpy default_rng({SEED})"
                    else:
                        origin = f"crafted corner values, numpy default_rng([{SEED}, {group}])"
                    files[name] = {
                        "role": role,
                        "published": True,
                        "note": f"{notes(group)[part]}; {origin}",
                        "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
                    }
    manifest = {
        "schema": 1,
        "generator": GENERATOR,
        "generator_commit": commit,
        "versions": {
            "torch": "not used",
            "transformers": "not used",
            "numpy": np.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "python": sys.version.split()[0],
        },
        "versions_role": "produced",
        "mlx_stream": "mx.gpu on every quantize and dequantize call that defines a golden (mx.cpu differs)",
        "inputs_source": "weights and the group-size-64 crafted scales/biases/codes are drawn from numpy default_rng(20261110) "
        "by this script, and the crafted ones of group sizes 32 and 128 from default_rng([20261110, group_size]); "
        "regenerating reproduces them for the same numpy, and the goldens for the same mlx on the same GPU family",
        "files": files,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")


def check_crafted_golden(name, obits, tallies):
    """A crafted golden must hold the corners it is there for, and the device model must reproduce it."""
    if not ((obits == 0x7F80).any() and (obits == 0xFF80).any()):
        sys.exit(f"{name}: the crafted golden lacks +inf or -inf; not writing it (do not reseed silently)")
    if not ((obits & 0x7FFF) > 0x7F80).any():
        sys.exit(f"{name}: the crafted golden holds no NaN; not writing it (do not reseed silently)")
    if not ((obits & 0x7FFF) == 0x7F7F).any():
        sys.exit(f"{name}: the crafted golden holds no maximum-magnitude finite value; not writing it")
    if not ((obits == 0x0000).any() and (obits == 0x8000).any()):
        sys.exit(f"{name}: the crafted golden lacks a zero of both signs; not writing it (do not reseed silently)")
    if tallies["ftz_model_mismatches"] != 0:
        sys.exit(f"{name}: the device model misses mlx {tallies['ftz_model_mismatches']} times; not writing it")


def main():
    commit = generator_commit()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    w = weights(rng)
    wbf = from_bits(f32_to_bf16_bits(w))
    report = {}
    # Group size 64 first, in the original draw order: the weights, then crafted(8) and crafted(4)
    # from the one stream, so the original 16 files do not move.
    for bits in (8, 4):
        wq, sc, bi = mx.quantize(wbf, group_size=GROUP, bits=bits, stream=GPU)
        out = mx.dequantize(wq, sc, bi, group_size=GROUP, bits=bits, stream=GPU)
        mx.eval(out)
        packed, sbits, bbits = np.array(wq), bits_of(sc), bits_of(bi)
        obits = bits_of(out)
        report[f"quant_q{bits}"] = tally(packed, sbits, bbits, obits, bits, GROUP)
        for suffix, a in (("packed", packed), ("scales", sbits), ("biases", bbits), ("out", obits)):
            write(fname("quant", bits, GROUP, suffix), a)

        packed, sbits, bbits = crafted(rng, bits, GROUP, CORNERS)
        out = mx.dequantize(mx.array(packed), from_bits(sbits), from_bits(bbits), group_size=GROUP, bits=bits, stream=GPU)
        mx.eval(out)
        obits = bits_of(out)
        report[f"crafted_q{bits}"] = tally(packed, sbits, bbits, obits, bits, GROUP)
        for suffix, a in (("packed", packed), ("scales", sbits), ("biases", bbits), ("out", obits)):
            write(fname("crafted", bits, GROUP, suffix), a)

    # The other group sizes: quantize the same weights, and draw the crafted case from a stream of its own.
    pending = []
    for group in OTHER_GROUPS:
        crng = np.random.default_rng([SEED, group])
        for bits in (8, 4):
            wq, sc, bi = mx.quantize(wbf, group_size=group, bits=bits, stream=GPU)
            out = mx.dequantize(wq, sc, bi, group_size=group, bits=bits, stream=GPU)
            mx.eval(out)
            packed, sbits, bbits = np.array(wq), bits_of(sc), bits_of(bi)
            obits = bits_of(out)
            key = f"quant_q{bits}_g{group}"
            report[key] = tally(packed, sbits, bbits, obits, bits, group)
            pending.append(("quant", bits, group, packed, sbits, bbits, obits))

            packed, sbits, bbits = crafted(crng, bits, group, CORNERS_WITH_NON_FINITE)
            out = mx.dequantize(mx.array(packed), from_bits(sbits), from_bits(bbits), group_size=group, bits=bits, stream=GPU)
            mx.eval(out)
            obits = bits_of(out)
            key = f"crafted_q{bits}_g{group}"
            report[key] = tally(packed, sbits, bbits, obits, bits, group)
            check_crafted_golden(key, obits, report[key])
            pending.append(("crafted", bits, group, packed, sbits, bbits, obits))
    for case, bits, group, packed, sbits, bbits, obits in pending:
        for suffix, a in (("packed", packed), ("scales", sbits), ("biases", bbits), ("out", obits)):
            write(fname(case, bits, group, suffix), a)
    write_manifest(commit)
    print(json.dumps({"host_formula_vs_mlx_gpu_bit_mismatches": report}, indent=1))


if __name__ == "__main__":
    main()
