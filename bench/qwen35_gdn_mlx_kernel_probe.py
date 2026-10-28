"""mlx-lm's gated-delta recurrence kernel, timed alone at the Qwen3.5-2B GDN shape.

The comparator for `tests/qwen35_gdn_simd_ab.rs`: the same kernel the mlx-lm
Qwen3.5 model calls (`mlx_lm.models.gated_delta.gated_delta_kernel`), on random
inputs prepared the way mlx-lm prepares them (q and k l2-normalised with q
scaled by Dk^-0.5, g = exp(-exp(A_log) softplus(a + dt_bias)), beta =
sigmoid(b)), batch 1, 16 key heads, 16 value heads, Dk = Dv = 128, zero state.
Only the kernel is inside the timed region; the inputs are evaluated first.

Both input dtypes are timed: bf16 is what mlx-lm runs the model in, f32 is the
width tessl's kernels read, so the pair separates the decomposition from the
operand bytes. Output is `MLX_KERNEL` lines for the results file.

    python bench/qwen35_gdn_mlx_kernel_probe.py [--rounds 15] [--lengths 1,200,1024,8192]
"""

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

B, HK, HV, DK, DV = 1, 16, 16, 128, 128


def l2norm(x, scale=1.0):
    ss = mx.sum(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(ss + 1e-6) * scale


def inputs(t, dtype, seed=0):
    mx.random.seed(seed + t)
    q = l2norm(mx.random.uniform(-1, 1, (B, t, HK, DK)), DK**-0.5).astype(dtype)
    k = l2norm(mx.random.uniform(-1, 1, (B, t, HK, DK))).astype(dtype)
    v = mx.random.uniform(-1, 1, (B, t, HV, DV)).astype(dtype)
    a = mx.random.uniform(-2, 2, (B, t, HV))
    b = mx.random.uniform(-2, 2, (B, t, HV))
    a_log = mx.random.uniform(-0.5, 1.0, (HV,))
    dt_bias = mx.random.uniform(-1, 1, (HV,))
    softplus = mx.logaddexp(a + dt_bias, 0.0)
    g = mx.exp(-mx.exp(a_log) * softplus)  # f32, as mlx-lm's compute_g
    beta = mx.sigmoid(b).astype(dtype)
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--lengths", default="1,200,1024,8192")
    args = ap.parse_args()
    for t in (int(x) for x in args.lengths.split(",")):
        for dtype in (mx.bfloat16, mx.float32):
            q, k, v, g, beta, state = inputs(t, dtype)

            def run(n):
                outs = []
                for _ in range(n):
                    y, s = gated_delta_kernel(q, k, v, g, beta, state)
                    outs += [y, s]
                mx.eval(*outs)

            run(1)
            t0 = time.perf_counter()
            run(1)
            pilot = (time.perf_counter() - t0) * 1e3
            n = max(1, min(200, int(args.target_ms / max(pilot, 0.05)) + 1))
            for _ in range(args.warmup):
                run(n)
            samples = []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                run(n)
                samples.append((time.perf_counter() - t0) * 1e3 / n)
            name = str(dtype).split(".")[-1]
            print(
                f"MLX_KERNEL t={t} dtype={name} min_ms={min(samples):.4f} "
                f"median_ms={statistics.median(samples):.4f} max_ms={max(samples):.4f} "
                f"rounds={args.rounds} launches_per_sample={n}"
            )


if __name__ == "__main__":
    main()
