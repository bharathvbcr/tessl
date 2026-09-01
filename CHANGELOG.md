# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The README's bf16-vs-MPS claim was wrong, and it was the headline.** It read
  1.11x with a worst shape of 1.01x — "never loses". Re-measured with the same
  harness against torch 2.13 MPS on the same machine: **1.03x, losing on 4 of the
  8 shapes**, worst 0.86x at 8192x3072x768. The tf32 lane (2.11x, wins every
  shape) and the MLX comparison (2.55x) hold up; bf16 against Apple's own tuned
  GEMM is parity, and the table says so now.
- **The published f32 and tf32 peak throughputs were not reproducible.** 10,897
  and 18,040 GFLOP/s; the crate's own committed sweep in `bench/results/` records
  6,606 for f32 and a fresh run gives 6,431 — two independent sources agreeing
  against the README. Replaced with measured values (26,642 bf16 / 16,293 tf32 /
  6,431 f32) rather than averaged.
- **Two more references to a binary that does not exist.** `docs/benchmarking.md`
  named `bench_gemm_coop_ab` in its tool table and in the Pitfall 1 prose. An
  earlier pass corrected only the *Reproducing* block at the end of that file, so
  its note could truthfully say the phantom was fixed while two live references
  survived above it. The paired A/B lane is `bench_gemm_tnnt_tune`. Every command
  named in the README and docs is now checked to exist.

## [0.1.0] — 2026-08-31

First published release. The crate has not been on crates.io before, so
everything below ships in it. The entries were written as the work landed and
are kept in that form rather than reflowed, because each records why it was
done — several of them are defects found by giving a kernel its first numeric
test, and the reasoning is the useful part.

### Added

- **`tests/q4_interleaved.rs`** — numeric coverage for the seven `_i4`
  (Interleaved4) MLX Q4 kernels, which had only a name check. They read a
  different weight packing from their row-major twins, and `Q4MlxBank` carries
  no layout tag, so the wrong packing dispatches and returns numbers — the same
  hazard `gemv_q4_mlx_blocked` turned out to be. The packer is transcribed from
  `gemv_q4_mlx_simd_i4`'s indexing: weights at
  `((tile * packs + pack2) * 4 + r) * 8` bytes with `tile = row / 4`, scale/bias
  at `(tile * groups_per_row + g) * 4 + r`; the nibble order inside each 8-byte
  group is unchanged, only the placement moves. Each test checks the `_i4`
  kernel against the dense f64 reference *and* against its row-major twin over
  the same logical weights.

  **All seven kernels were correct.** Verified the tests can fail: feeding the
  `_i4` path a row-major bank fails all six, and mutating the kernel's scale
  stride fails the one test that covers it.

- **The GEMM residual arm.** `gemm_q4_mlx_simd_add` and `_add_i4` are reached
  only by passing `Some(resid)`, and every other GEMM test passed `None`, so
  they were the last two promoted kernels with a name and no number. The
  residual is the full `m x rows`, matching the output — `resid[m * rows + row]`
  in the kernel — not a per-row vector broadcast across `m`, which is what the
  first draft of the test assumed and what the host validation caught.

  With this, all 44 promoted kernels have a numeric test.

### Fixed

- **The sliding-window attention kernels read uninitialised threadgroup memory
  for half of every query block.** `flash_attn_swa_h128` and `_h256` zeroed
  their `scores[BR * BC]` scratch with `if (lid < BR * BC)` — 64 entries — while
  the host dispatches **32 threads per threadgroup**. Entries 32..63 were never
  zeroed, and those are query rows 4..7 of every block, which then accumulated
  QK products into whatever a previous dispatch had left there. The result was
  plausible numbers rather than NaN, so nothing looked wrong. Zeroing is now
  strided, which is correct for any relation between `tptg` and `BR * BC`.
  `flash_attn_global_h512` was unaffected only because its `BR = BC = 4` gives
  16 entries, under the 32 threads; it is fixed the same way so the property
  does not depend on the tile constants.
- **A fully masked block produced NaN in the online softmax.** The FA-2 rescale
  computes `alpha = exp(m_i - m_new)`, and when a row had seen nothing yet and
  the current block was entirely masked for it, both were `-inf` — so `alpha`
  was `exp(NaN)`, which then propagated through `Oacc` and `l_i` and poisoned
  the row. This is reachable whenever the block-level skip admits a block on
  behalf of another row in the same `BR` tile, which the union window makes
  routine at small `window`. Guarded with `m_i == -inf ? 0`, which is also the
  right value for the ordinary first-real-block case.
- **`out_bf16` demanded an output buffer sized for f32.** `flash_attn_global_h512`
  validated `o` through `validate_attn_dims`, which always required
  `require::<f32>`, and then added a `u16` check on top. A caller who sized the
  buffer for bf16 — the whole point of the flag, documented as "half-width act
  scratch" — got "buffer holds 2560 elements, kernel reads/writes 5120". The
  output width now follows `out_bf16`.

### Added

- **`tests/attention.rs`** — six tests against an f64 reference transcribed from
  the kernels' own masking rule: prefill at both sliding-window head dims with
  ragged query and key tails, window bounding (including `window = 1`, which
  must reduce each row to its own V), decode with the device-side position
  offsets, a fully masked decode row that must be zeros rather than NaN, GQA
  head grouping across four `H:Hkv` ratios, and the global kernel's causal rule
  plus its bf16 output arm.
- **`tests/qkv_rope.rs`** — four tests for the fused RMSNorm to QKV to RoPE
  kernels: the constant-position variant against an f64 reference at full and
  partial `rotary_dim`, V normalised but never rotated, `PosBuffer` agreeing
  bit for bit with `PosConst`, and `PosBufferKvStore` writing the rotated K and
  V into the cache at a device offset without touching anything outside the
  slot. All four passed on the first run; these kernels were correct.

- **`gemv_q4_mlx` with `Q4MlxRowVariant::Tiled` left most of its output
  unwritten** — the same defect as `gemv_q4_tiled`, in the sibling family.
  `gemv_q4_mlx_tiled` indexes its output row by `threadgroup_position_in_grid`
  and needs one threadgroup per row; all three row variants were dispatched with
  the one-thread-per-row grid, so `Tiled` wrote `rows / 128` rows and returned
  no error. Found by giving the three variants one shared numeric test instead
  of testing `Standard` alone: 508 of 512 rows never written.

### Documented

- **`gemv_q4_mlx_blocked` requires a block-interleaved bank, and nothing said
  so.** It takes the same `Q4MlxBank` as its row-major siblings — a type with no
  layout tag — and returns wrong numbers rather than an error when given a
  row-major one. The kernel reads scale/bias and nibbles at
  `block * groups_per_row * 16 + group * 16 + row_in_block`. Measured at 64x256
  with `group_size` 64: 63 of 64 rows wrong row-major, 0 of 64 repacked. The two
  layouts coincide only when `groups_per_row == 1`, which is exactly the shape a
  small smoke test would pick. `tests/promoted_numeric.rs` carries a reference
  repacking.

### Added

- **`tests/promoted_numeric.rs`** — numeric coverage for promoted kernels that
  had only a name check in `promoted_kernels.rs`. That file asserts each of the
  44 entry points resolves from tessl's own metallib, which is a real gate on
  the move and not a correctness test: `gemv_q4_tiled` resolved, had adversarial
  coverage, and wrote 4 rows of 512. Eight tests now cover the three MLX Q4 row
  variants, the blocked GEMV, the fused K/V GEMV, `gemm_q4_mlx` against the GEMV
  row by row, `mlp_gelu_tanh_bf16` against its f32 sibling,
  `kv_store_timestep_pair`, `kv_ring_densify`'s rotation, and
  `embed_lookup_q4_mlx` including out-of-range token ids. Every one uses `rows`
  above the 128 the row kernels group by, because below that the competing grids
  coincide and a mismatch is invisible. Shared Q4 scaffolding moved to
  `tests/common/mod.rs`.

- **`gemv_q4` with `tiled = true` silently left most of its output unwritten.**
  `gemv_q4_tiled` indexes its output row by `threadgroup_position_in_grid` and
  so needs one threadgroup per row, but it was dispatched with
  `rows.div_ceil(128)` groups — the geometry the one-thread-per-row `gemv_q4`
  needs. It wrote the first `rows / 128` rows and left the rest of `y` holding
  whatever was there before, with no error returned. Measured at 512 rows: 4
  written, 508 untouched. The dynamic threadgroup allocation and the `cols`
  ceiling that bounds it are now applied only to the kernel that caches `x`;
  the tiled kernel declares its scratch statically and never did.
  **Anyone who passed `tiled: true` was getting wrong results**, and with the
  grid corrected that variant is slower than the default one at every shape
  measured — its apparent speed was the work it was skipping.
- **Nothing tested `gemv_q4_tiled` numerically.** `promoted_kernels.rs` checks
  the pipeline name exists and `nn_adversarial.rs` checks error paths, so a
  kernel writing 0.8% of its rows passed both. Added a test that asserts the two
  variants agree row for row and that neither leaves a seeded sentinel behind,
  at 512x256 and a ragged 300x128.

- **Q8 GEMV ran one thread per row, uncoalesced.** `gemv_q8` dispatched `rows`
  threads, so adjacent threads read addresses `cols` bytes apart and a
  simdgroup's 32 loads touched 32 cache lines. Now one simdgroup per four rows
  with lanes striding K and `char4` loads covering a full cache line per
  instruction, reusing the `simd_gemv_threadgroups` / `SIMD_TPTG` geometry the
  MLX Q4 GEMVs already used. Measured on an M5 Pro: **2.9x at 4096x4096**
  (274.5 -> 95.8 us, 68.9 -> 197.4 GB/s) and 1.25x at 11008x4096. The tall
  case is bound by something else and is recorded as such rather than
  explained away.
- **The Q8 GEMV test covered neither new path.** rows = 24 is a multiple of the
  8 rows a threadgroup owns and group = 16 is divisible by 4, so the ragged row
  tail and the scalar fallback never ran; breaking either left it green. The new
  case sweeps rows 13/37/100 and groups 15/32/64, and seeds `y` past `rows` with
  a sentinel to catch a tail threadgroup writing rows it does not own.

- **RMSNorm ran one thread per row.** All three kernels (`rms_norm_f32`,
  `rms_norm_bf16`, `rms_norm_residual_add_f32`) dispatched `rows` threads, each
  walking its row serially twice, which capped parallelism at the row count and
  ran the entire kernel on a single GPU thread at the decode shape. Now one
  threadgroup per row with the sum of squares reduced as a tree, the pattern
  `reduce.metal` already used. Measured on an M5 Pro: **16.7x at 1x4096**
  (404.3 -> 24.2 us), 10.1x at 512x4096, 2.5x at 2048x4096, with effective
  bandwidth going from 87 GB/s to 216-305 GB/s. RMSNorm runs twice per
  transformer layer on every token. The reduction reassociates, so results
  differ in the low bits from the previous serial sum.
- **The RMSNorm tests could not see the bug they now cover.** Every existing
  case used `dim` of 16 to 64 against a 1024-thread group, so each lane's
  strided loop ran once and deleting the loop entirely left all three green.
  Added `dim` of 4096, a ragged 3000, and 8192 across all three variants; both
  new tests kill that mutation, as do removals of `REDUCE_TREE` and of `eps`.
- **`build.rs` tracked only `.metal` for `rerun-if-changed`.** `REDUCE_TREE` now
  lives in `kernels/reduce_tree.h`, shared by `reduce.metal` and
  `rms_norm.metal`; without tracking headers an edit to the shared reduction
  would leave both dependents stale in the metallib while the suite reported a
  pass. Verified by mutating only the header and confirming a rebuild and five
  failures.

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

