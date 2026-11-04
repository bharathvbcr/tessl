# Metal Performance Primitives' `convolution2d`: semantics by probe, speed against MPSGraph (2026-11-04)

This is the evidence behind `tessl::conv` (`src/conv.rs`, `kernels/conv2d_tensorops.metal`). Apple ships
a convolution tensor op (`MPPTensorOpsConvolution2d.h`, Metal 4) that runs on the GPU's matrix units.
Its header documents the layouts (NHWC activations, HWIO weights, NHWO destination), and nothing
about padding, tile offsets or partial tiles. These were measured.

## Files

- `probe.metal` and `probe.swift`: a throwaway harness. It runs the op on random data, checks it
  against an f64 CPU reference (every point when the output is small, else 4,104 sampled points with
  every corner), and times it. Debug modes: `PROBE_DEBUG=kh,kw` encodes each input's index in its
  value, with identity weights at one tap. `PROBE_MASK` zeroes chosen weights. `PROBE_PREPAD` and
  `PROBE_ADJ` choose the padding and the offsets.
- `mpsgraph.swift` and `mpsgraph.log`: the MPSGraph `convolution2D` baseline (GPU time of the conv
  alone).
- `fused_ab.log`: the fused NCHW store against the NHWO scratch plus unpack pass, run with
  `examples/conv2d_bench.rs` before the unpack arm was deleted.

## What the op does [V, probe runs]

1. **It centres the kernel.** Destination `(h, w)` reads source `(h + kh - K/2, w + kw - K/2)` of
   the activation slice it is given. An identity weight at tap (1,1) read input `(h-1, w-1)` of a
   source padded by 1. At tap (0,0), every output whose read fell before the slice origin read zero.
2. **Reads before the slice origin are zero.** So a tile's halo on its top and left is lost unless
   the source is offset. Reads past the slice end reach the parent tensor (tap (2,2) was exact).
3. **`set_offsets(K/2, K/2)` cancels the centring.** On a source padded by P, this makes a
   destination tile at `(h0, w0)` the valid convolution at `(h0, w0)`, which is PyTorch's padding P.
   Result: relative L2 8.5e-7 in f32 at 40², 256 → 64. Offsets of (0,0) and (−1,−1) were wrong
   (1.29, 1.14).
4. **A partial destination tile is wrong.**
   - f16 was correct at widths 16 and 32 and wrong at 24, 33 and 40 (relative L2 0.39-0.56).
   - f32 was wrong at 37x45 (0.16) and right at 40x40.
   - Channel and spatial mapping were right in every debug run, so it is the partial tile.
   - **So `tessl::conv` pads the source and the scratch to a whole number of 32-wide tiles.** With
     the padding removed, all four parity tests in `tests/conv2d_tensorops.rs` fail (relative L2
     0.10-0.31), and they pass with it.
5. **A strided NCHW destination view is not written correctly.** Channel stride ≠ 1 gave 64,449
   unwritten outputs. So the op writes a cooperative destination (registers), and the kernel
   stores each element to NCHW itself with `get_multidimensional_index` (see below).
6. **The input channel count can be a runtime value** (`dynamic_length_v<int>`): correct at 32, 64,
   256 and 512. The kernel size and tile sizes are compile-time.
7. **Precision:** f16 relative L2 ≈ 2.8e-4 at K = 2304, f32 ≈ 8.5e-7.

## Speed: the op alone vs MPSGraph [V, GPU time, mac_heavy hold, load 5.6-9.1]

| 3x3 conv, stride 1, pad 1 | MPP op f16 (probe) | MPSGraph f16 NCHW | MPP op f32 (probe) | MPSGraph f32 NCHW |
|---|---|---|---|---|
| 288², 256 → 256 (97.8 GFLOP) | 3.26 ms (30.0 TFLOP/s) | 3.13 ms | 14.1 ms | 7.98 ms |
| 144², 256 → 256 | — | 0.86 ms | — | 2.01 ms |
| 72², 256 → 256 | — | 0.29 ms | — | 0.54 ms |
| 1152², 32 → 16 | — | 1.40 ms | — | 2.07 ms |

- **f16:** the MPP op and MPSGraph are within 5% at 288², where the op was probed.
- **f32:** MPSGraph is 1.8x faster than the f32 MPP op. It must use another algorithm; which one is
  unverified.
- **End to end**, `tessl::conv` adds the weight and pack passes. Its wall times through Ojas are in
  `ojas/bench/results/2026-11-04-conv2d-tensorops/`.

## Fused NCHW store vs scratch plus unpack [V, `fused_ab.log`]

Wall time per call through `conv2d_nchw` (encode, GPU and wait), bias on. Arms interleaved in 9
rounds of 10 calls after 2 warm-up rounds. Load 4.0-4.1. One session; the identical config was
not re-run, so gaps of a few percent are near the noise.

| 3x3 conv, pad 1 | f16 fused | f16 unpack | f32 fused | f32 unpack |
|---|---|---|---|---|
| 288², 256 → 256 | **4.63 ms** | 4.91 | **16.45** | 16.73 |
| 144², 256 → 256 | **1.46** | 1.51 | **4.74** | 4.84 |
| 72², 256 → 256 | 0.596 | **0.582** | 1.666 | **1.580** |
| 1152², 32 → 16 | **2.15** | 2.71 | **3.54** | 4.17 |
| 144², 256 → 1024 | **5.05** | 5.21 | **18.83** | 19.23 |

(medians; minima order the same way.)

- **The fused store won at 4 of 5 shapes**, by 2-6% at 256 output channels and 20% (f16) / 15%
  (f32) at the narrow 1152², 32 → 16 conv, where the NHWO scratch round trip was the larger part.
- **It lost at 72²** by 2.4% (f16) and 5.4% (f32). Unexplained: that grid is only 3 tiles wide, so
  the per-element NCHW stores may be the cost there.
- **The fused store was kept and the unpack path deleted** (its NHWO op kernels, the unpack
  kernel and the A/B switch). A per-shape choice would buy at most about 0.09 ms at 72².

## Not measured

- Strides above 1 (the probe's offset rule was found at stride 1 only), kernels other than 3x3,
  and grouped convolutions (the op asserts `groups == 1`).
- MPSGraph's algorithm choice per shape.
- A quiet machine: every number here ran at a load average between 5.6 and 9.1.
