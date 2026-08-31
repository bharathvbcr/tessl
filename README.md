<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="tessl" width="330">
  </picture>
</p>

<p align="center">
  <strong>Metal 4 GEMM for Apple silicon.</strong><br>
  Hand-written TensorOps kernels that match PyTorch MPS on bf16.
</p>

---

`tessl` is a Rust crate that runs matrix multiplication on Apple silicon through
Metal 4 and Metal Performance Primitives (MPP) `matmul2d`, targeting the neural
accelerators on M5-class hardware. It ships the GEMM kernels, the Metal 4 encode
path they run on, and the measurement harnesses used to tune them.

The name is short for *tessellation* — the whole design is about how an output
matrix is cut into tiles and in what order those tiles are walked.

## Where it stands

Measured against PyTorch MPS 2.13.0 and MLX 0.32.0 on an M5 Pro, using a
**paired** protocol that alternates the two lanes round by round so GPU clock
drift cancels (see [Benchmarking](docs/benchmarking.md) — this matters more than
it sounds):

| comparison | geomean of per-shape medians | worst shape | best shape |
| --- | --- | --- | --- |
| bf16 vs PyTorch MPS bf16 | **1.00×** | 0.84× | 1.12× |
| f32 exact vs PyTorch MPS f32 | 1.07× | 0.92× | 1.47× |
| tf32-relaxed vs PyTorch MPS f32 | 1.98× | 1.49× | 2.34× |
| bf16 vs MLX bf16 | 2.63× | 1.13× | 3.64× |

**Read that first row as parity, not a win.** On bf16 — the case that matters
for training and prefill — tessl and PyTorch MPS are within a few percent of
each other and the ranking flips shape to shape. Both plateau around 25 TFLOP/s
on large shapes, which is what you would expect if the same hardware unit, not
either library's tiling, is the limit.

Two caveats on the other rows, so the table is not read for more than it says.
The tf32 row is **not** like-for-like: relaxed precision truncates the mantissa,
so it measures what the `--tf32` opt-in buys, not an f32 result. And MLX bf16
measures within noise of MLX f32 on this machine (~6.5 vs ~6.7 TFLOP/s), which
suggests MLX is not reaching the neural accelerators for matmul here; that 2.63×
is reported as measured, not as a considered claim about MLX.

## The interesting part

The bf16 gap against PyTorch used to be a clean function of K:

| K | before | after | PyTorch MPS |
| --- | --- | --- | --- |
| 256 | 14,876 | 15,059 | 11,322 |
| 1024 | 23,347 | 24,821 | 24,748 |
| 4096 | 21,147 | **26,679** | 25,395 |
| 8192 | 18,666 | **23,182** | 23,746 |

*GFLOP/s at M=N=4096.* Throughput **fell** as K grew past 2048 — which a
compute-bound kernel should not do. The blocked kernel accumulated into a
device-memory C tile once per K block, so C traffic scaled with K/BK while the
useful work scaled with K. At K=8192 that is 32 passes over a 67 MB tile for a
GEMM that needs to write it once.

The fix holds the accumulator in registers across the whole K loop
(`get_destination_cooperative_tensor`) and stores once, making C traffic
independent of K. It applies to exactly the two kernels that had that structure;
[docs/architecture.md](docs/architecture.md) covers where it does **not** apply
and shows the evidence for that, which is more interesting than where it does.

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | Kernel selection, the cooperative-accumulator gate clause by clause, why TN/NT are excluded |
| [Benchmarking](docs/benchmarking.md) | The measurement protocol, and five ways these numbers went wrong before they went right |
| [Verification](docs/verification.md) | Static audit, seeded shape fuzz with coverage assertions, fault injection |

## Requirements

- Apple silicon, macOS 26 or newer
- Xcode 26 with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- Rust 1.82+

Metal 4 only — the classic `MTLCommandQueue` encode path is not used. The
TensorOps kernels require MPP, and the register-accumulator path assumes the
M5-generation neural accelerators.

## Status

Pre-release, and honest about it:

- **Every constant is tuned on one machine** (M5 Pro). Nothing has been checked
  on an M3, M4, or a base M5.
- **The API is not stable.** `Tensor`, `GpuRuntime` and the dispatch layer were
  extracted from a training codebase and still carry its shape.
- **No license yet.** This has to be settled before the crate can be published.

Code lands in this repository shortly; it currently lives inside a larger
research workspace and is being split out.

## Acknowledgements

Built on Apple's Metal Performance Primitives. Benchmarked against
[PyTorch](https://pytorch.org) MPS and [MLX](https://github.com/ml-explore/mlx),
whose numbers here were produced by the harnesses in `bench/` and are
reproducible with them.
