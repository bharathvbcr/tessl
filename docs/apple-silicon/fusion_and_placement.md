# Fusion and placement: when a compiled graph or a megakernel pays

Inference evidence from two workloads on the M5 Pro, gathered 2026-11-03/04: SpikeTrack (SAM 3.1 vision
at batch 1, on Ojas + tessl, with Core ML placements) and gemma-metal (Gemma 4 E4B Q4 decode). Read this
before proposing a megakernel, a fused region or a Core ML export.

Related: [`coreml_metal_ml.md`](coreml_metal_ml.md) (Core ML is deploy only) ·
[`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) ·
[`../../../ojas/gemma-metal/DECISIONS.md`](../../../ojas/gemma-metal/DECISIONS.md) (D15-D18, D20).

Labels: **[V]** measured or read in code, with the source named; **[I]** inferred from those numbers.
SpikeTrack paths are in the sibling checkout `../../../SpikeTrack/`.

## The rule

Whole-graph compilation and megakernels remove two costs: the **launch** of each small kernel and the
**round trip of each intermediate** through memory. They pay only when those two costs are a large share
of the time. Measure that share first:

1. **Many small ops on large activations** (vision at batch 1: convs, norms and attention over
   72x72 to 288x288 maps): intermediates dominate, so fusion and compiled graphs pay.
2. **Weight streaming** (LLM decode, a GEMV per matrix): the weights are read once per token whatever
   the kernel boundaries are, so fusion cannot beat the GEMV's own bandwidth. It can still remove the
   time between kernels. In gemma-metal that time is most of the token (see LLM decode below), so
   measure kernel time against token time before deciding.
3. **A placement boundary** (Metal -> Core ML -> Metal) costs a host copy of every tensor that crosses it
   today. Offload only a region whose compute dwarfs its inputs and outputs, or remove the copies first
   (tessl ft-77620e66).

## Evidence

### Where a compiled graph won [V]

| region (SAM 3.1, batch 1) | per-op Metal (Ojas) | Core ML program, CPU_AND_GPU | in the frame | source |
|---|---|---|---|---|
| ViT trunk, 1008² in, 32 blocks | — | fp32 0.905 s, fp16 0.252 s | frame 0.80x (fp32), 0.55x (fp16); dispatches 6,284 -> 4,007 per frame | `SpikeTrack/bench/results/coreml-trunk/README.md` |
| memory attention | 252 ms/frame (stage profile) | fp16 program | with the trunk pipeline, frame 0.62-0.71x | `SpikeTrack/bench/results/e26-pipeline/README.md` |
| segmentation head | ~108 ms in-frame | fp32 24.8 ms, fp16 15.1 ms (rel L2 6.5e-6 / 4.3e-4) | 1.00-1.06x: no gain | `SpikeTrack/bench/results/coreml-seghead/`, `e26-pipeline/README.md` |
| detector + propagation necks | ~110 ms | fp32 56 ms | inline 0.97-1.07x; overlapped on the trunk's worker thread 0.98-1.03x (fp32); fp16 on the worker 0.89-0.92x on a quiet machine, adopted (E30) | `SpikeTrack/bench/results/e25-coreml-frame/README.md`, `e27-pipe-necks/README.md`, `e30-quiet/README.md` |

The segmentation head shows why per-op Metal loses there [V numbers, I cause]: Core ML runs the whole
head, which includes two 3x3 256->256 convs (144² and 288²), two GroupNorm+ReLU, an 8-head
cross-attention and the mask matmul, in 24.8 ms. Ojas's two convs alone take 26.9 + 6.2 ms in isolation
(`ojas/bench/results/2026-10-30-gemm-class/summary.md`). Ojas runs a 3x3 conv as im2col + GEMM + layout.
At 288² the im2col matrix is 82,944 x 2,304 f32 (~764 MB), written and then read on every call
(`ojas-metal/src/backend.rs`, `conv2d_routed`). Gap: ojas ft-d62b1248 (implicit-GEMM 3x3 conv).

### Where it did not, and why [V]

- **Placement boundaries.** The necks and the segmentation head are 2-4x faster as programs, but the
  frame pays host copies at the boundary: 111 MB of neck levels read back and re-uploaded twice per
  frame; 106 MB in and 66 MB out for the head. That is enough to erase the win [I, sizes from the
  tensor shapes]. The trunk wins because ~0.9 s of compute crosses 33 MB.
  - Moving the necks onto the trunk's own worker thread removes their boundary: they read the
    trunk there and their levels never pass through the main thread's Metal queue. At fp16 that paid
    (E30, 0.89-0.92x). At fp32 it did not (E27, 0.98-1.03x).
- **The Neural Engine.** The ViT trunk on CPU_AND_NE: 2.39 s per call against 0.26 s on CPU_AND_GPU,
  with twice the fp16 error (`SpikeTrack/bench/results/coreml-trunk/README.md`). Refuted for this model.
- **LLM decode.** gemma-metal E4B: ~600 dispatches/token, ~42 ms/token. Fusion removed 90 dispatches
  (launch tax ~3.3 ms, an ~8% ceiling) with no tok/s change; encode-once and true ICB were flat or slower
  (`ojas/gemma-metal/DECISIONS.md` D15, D16). A token reads ~2.86 GiB, which at 42 ms is ~70 GB/s
  effective [I]. mlx-lm runs the same model ~3x faster (D2; D10 notes the graphs are not identical).
  - **Measured 2026-11-04** (`ojas/gemma-metal/bench/results/gemv_fraction_of_peak_20261104/`): this
    file's GEMV (`gemv_q4_mlx_simd`) alone at E4B's shapes streams 83-89% of the 275-281 GB/s read
    ceiling at the MLP shapes and 96-102% at lm_head. MLX's `quantized_matmul` is no faster, except on a
    lone 512-row K or V (64 threadgroups underfill the GPU; MLX 2x).
  - So a token's GEMVs are ~13 ms of ~42 ms [I], and the gap is between kernels, not in the GEMV.
  - The next step is a per-dispatch GPU-timestamp profile of a real token, before the persistent-kernel
    probe (ojas ft-e6d8035e).
  - Note that removing 90 dispatches did not help (D15), so the gap may not scale with dispatch count.

### What paid without any new kernel [V]

- **Overlap across frames.** Running the trunk program for frame t+1 on its own thread while frame t is
  tracked (a Core ML model stays on the thread that loaded it): frame 0.78-0.91x, predictions
  byte-identical in all 3 rounds (`SpikeTrack/bench/results/e26-pipeline/README.md`).
- **fp16 where the task metric allows it.** The fp16 trunk + fp16 memory attention take the frame to
  0.57-0.60x of the fp32 default (`SpikeTrack/bench/results/e27-pipe-necks/README.md`). Per-op error is
  0.5-2% rel L2 on the trunk, which cosine > 0.999 does not catch, so the gate is the task metric:
  held-out HOTA 0.6025 unchanged on 32 frames x 5 videos and 0.5694 against 0.5700 on 1,500 frames
  (`SpikeTrack/bench/results/track/held300-coreml/README.md`). The pair became SpikeTrack's Metal
  default on 2026-11-04.
- **fp16 necks on the trunk's worker (E30).** They add 0.89-0.92x on top of the fp16 trunk + memory
  attention. Held-out HOTA is 0.5697 over 1,500 frames
  (`SpikeTrack/bench/results/e30-quiet/README.md`). They joined the default the same day. The first
  speed run was voided by machine load and repeated quietly under the same rule.

## Checklist before building a fused region or megakernel

1. Profile the region per op and per stage, synced, and record the launch count.
2. Estimate the intermediate bytes (sum of op outputs re-read by the next op) and the weight bytes.
   If weight bytes dominate, fix the kernel's bandwidth first.
3. Time the strongest baseline for the same region: a Core ML program (coremltools, fixed shapes) or
   MLX on the same shapes. If it is no faster, fusion will not be either.
4. If the region crosses a framework boundary, count the copies at the boundary before timing anything.
5. Pre-state the gate (interleaved rounds, min-of-N, noise floor of a repeated arm) and the task-metric
   gate for any precision change.
