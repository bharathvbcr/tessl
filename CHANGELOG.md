# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`docs/verification.md` documented a command that ran zero tests and
  reported `ok`.** It named a test `gemm_randomized_shape_fuzz` and two
  environment variables `GEMM_FUZZ_SEED` / `GEMM_FUZZ_CASES`; none of the three
  exist, so `cargo test ... gemm_randomized_shape_fuzz` matched nothing and
  printed `test result: ok. 0 passed; 89 filtered out`. The real entry points
  are `gemm_fuzz_quick`, `gemm_fuzz_deep` and `STRESS_SEED`. The same section
  claimed the fuzzer asserts per-kernel coverage at a 1% floor; no such
  assertion is implemented, and the retraction is now in the document rather
  than only in the README.
- **`docs/benchmarking.md` named a binary that does not exist.** The reproducing
  block invoked `bench_gemm_coop_ab` with a `BENCH_ROUNDS` variable. This crate
  builds four binaries and neither the target nor the variable is among them.
- `docs/verification.md` reported 68 unit tests; the suite is 199 across 89 lib
  and 110 integration tests.

### Added

- **Docs cover the promoted library.** `docs/architecture.md` previously
  described only GEMM — no mention of `nn`, the fused epilogue, the reductions,
  or the integer GEMM. It now documents the validate-before-encode boundary,
  the row-stride-0 bias broadcast, softmax's masked-row behaviour, and the
  int32 wrap bound at `k = 131072`.
- **A measured GEMM sweep in `docs/benchmarking.md`**: bf16 TensorOps at 26,642
  GFLOP/s on 4096³, 11.8x the portable simdgroup fallback and 4.2x exact f32,
  with the 512³ inversion explained by the dispatch floor rather than left as
  an anomaly.

- **Quantized int8 GEMM with fused dequantization**: `nn::gemm_i8_dequant`.
  `int8 x int8` accumulates into `int32` natively on TensorOps, and every
  product fits, so the integer result carries **no rounding at all** — tested by
  exact equality against an integer reference, not a tolerance. The per-column
  dequantization is applied in registers between the accumulate and the store.
  `k` above 131072 is refused, past which a full-range accumulation could wrap
  the int32 silently.
- **Corrected a misdiagnosis this crate had been repeating.** Quantized
  TensorOps was documented as blocked because `MTLTensorDataType::Int4` is
  unbound in objc2-metal 0.3. That binding gates host-created `MTLTensor`
  descriptors, and every kernel here builds tensors from raw device pointers
  instead, so it never applied. TensorOps supports
  `uint8_t/int8_t/uint4b_format/int4b_format` per the header's own diagnostic;
  what actually blocks Int4 is the shader-side tensor constructor for a
  sub-byte element type.
- **Strided batched GEMM**: `gemm_batched` with `BatchedGemm` and
  `BatchStrides`. The batch is the grid's second dimension, so it costs a
  pointer offset per threadgroup and nothing else — every element is
  bit-identical to the `gemm` that would have produced it. Per-operand strides,
  because a **zero** stride is the useful case: batched activations against one
  shared weight matrix needs no copies of B. Dimensions are passed explicitly
  rather than read from tensor shapes, since a rank-2 shape cannot distinguish
  `[batch * m, k]` from `[m, k]` and a broadcast B genuinely is `[k, n]`.
- **IEEE binary16 (`DType::F16`)**: `alloc_tensor_f16`, `cast_f32_to_f16` /
  `cast_f16_to_f32`, host `f32_to_f16_bits` / `f16_bits_to_f32` /
  `f32_slice_to_f16`, `GpuBuffer::write_f16_bits`, and f16 GEMM kernels
  (`matmul2d_tensorops_f16_f32`, the 64x64 variant, and the epilogue variant).
  f16 and bf16 are both two bytes and both accumulate in f32, but their bit
  layouts differ, so `nn_coop_kernel` now selects on a three-way `CoopElem`
  rather than a boolean, and `ensure_bf16` refuses f16 operands instead of
  converting them — that conversion would lose three mantissa bits and change
  the exponent range to buy a path the caller did not ask for.
- **Row-wise reductions**: `nn::softmax_rows_f32`, `nn::row_sum_f32`,
  `nn::row_max_f32`. One threadgroup per row, striding, so `cols` is unbounded.
  Softmax subtracts the row maximum before exponentiating — without it a single
  logit above about 88 overflows `exp` in f32 and takes the row to NaN, which is
  an ordinary attention input rather than a pathological one. A fully masked row
  (`-inf` everywhere) yields a uniform distribution rather than NaN.
- **`gemm_epilogue` — fused GEMM epilogue.**
  `C = activation(alpha * A@B + beta * C_prev + bias)` in one dispatch, applied
  while the accumulator is still in registers, so `C` is written once and read
  at most once. Measured 1.57x to 2.43x cheaper than a single elementwise sweep
  over `C` — which is itself strictly less work than any real unfused epilogue.
- `Activation` (`None`, `Relu`, `GeluTanh`, `Silu`) and `Epilogue`, with
  `Epilogue::default()` as the identity, which dispatches to plain `gemm`.
- Per-column bias broadcasts through a row-stride-0 tensor view, reusing the
  cooperative `load` path that fetches `C_prev`.
- `matmul2d_tensorops_bf16_f32_epi` and `matmul2d_tensorops_f32_relaxed_epi`.
  Separate entry points rather than extra parameters on the existing kernels:
  Metal faults on a declared-but-unbound buffer, so widening those signatures
  would force every current caller to bind four operands it does not use. The
  epilogue is a template parameter, so both share one source and the plain path
  compiles to exactly what it did before.
- `examples/epilogue_cost.rs`.
- **`nn` module — 44 kernels promoted out of `gemma-metal`.** RMSNorm (f32,
  bf16, fused residual-add with layer scale), gated MLP activations (SiLU,
  `gelu_pytorch_tanh`), sliding-window and global flash attention, fused
  RMSNorm+QKV+RoPE, MLX-format Q4 GEMV and GEMM, Q8 GEMV, KV-cache timestep
  stores and ring densify, quantized embedding lookup, and softcap/argmax
  sampling. These had lived in one model's crate, reachable only as raw strings
  through an overlay metallib.
- Every `nn` entry point validates operand extents on the host before encoding.
  The kernels guard `gid >= n` and nothing else, so an undersized buffer was
  previously an unchecked out-of-bounds *device* read.
- A `_with_scalars` variant of each entry point, taking a closure that binds the
  scalar operands. Callers that need stable GPU addresses across encodes — an
  Indirect Command Buffer that froze its binds — supply their own persistent
  scalar pool without reimplementing the dispatch.
- `GpuRuntime::max_threadgroup_memory`, so callers can check a kernel's
  threadgroup-memory request before a dispatch-time failure that names neither
  the kernel nor the dimension.
- `examples/gemm.rs` and `examples/nn_layer.rs`, both run by CI. The README's
  snippets are these files.

### Fixed

- Removed a duplicate `cast_f32_to_bf16` that arrived with the promoted kernels.
  `GpuRuntime::pipeline` resolves the primary metallib before any overlay, so
  the copy in `rms_norm.metal` could never have been the one dispatched.
- `docs.rs` builds. The crate is Apple-silicon only and `build.rs` drives the
  Metal toolchain, neither of which exists on docs.rs's x86_64-linux builder;
  the build script now detects `DOCS_RS` and skips the shader compile, and
  `[package.metadata.docs.rs]` targets `aarch64-apple-darwin`.
- 18 rustdoc warnings: matrix-shape notation such as `C[M,N] = A[M,K] @ B[K,N]`
  was being parsed as intra-doc links, and one public doc linked a private item.
- README: the `add_metallib` / `from_metallib_path` snippets passed `&str` to
  functions that take `&Path` and would not have compiled.

### Changed

- `gemma-metal` now delegates seven wrappers to `tessl::nn` rather than
  reimplementing the dispatch, removing 52 lines of duplicate binding code.
- Documented the crate's `unsafe`. It splits into two classes: `objc2` message
  sends, where every site discharges the same obligation and which are covered
  once in the crate docs, and raw pointer/slice construction, where the
  obligation is site-specific. Every block in the second class now carries a
  `// SAFETY:` comment naming the invariant and where it is established —
  8 comments before, 22 now.
- `icb_smoke::verify_copy` bounds-checks its length against the buffer instead
  of relying on both call sites happening to pass the length the buffer was
  allocated from.

### Removed

- `mtl_tensor::try_quant_tensorops_prefill_gemm`. Its entire body was
  `Err("not wired yet")` with every parameter underscored, no caller and no
  test. The fact it encoded — that quantized TensorOps prefill GEMM does not
  exist, because `MTLTensorDataType::Int4` is unbound in objc2-metal 0.3 — is
  now the single `QUANT_PREFILL_GEMM_WIRED` constant that
  `nax_verify_readiness` reports, replacing a second copy of the same sentinel
  that existed alongside it.

[Unreleased]: https://github.com/bharathvbcr/tessl
