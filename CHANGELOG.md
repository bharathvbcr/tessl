# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`tests/gemm_flag_paths.rs`** — the four shipped accumulate kernels and the
  exact-f32 kernels' interior branch run under the flags that select them
  (`TESSL_GEMM_ACCUM`, `TESSL_GEMM_ACCUM_DX`, `TESSL_GEMM_INTERIOR`), each
  configuration in a child process, checked against the f64 reference with the
  previous C folded in, and with `TESSL_KERNEL_TRACE` proving the accumulate
  kernels ran and the temp-plus-`add_inplace_f32` fallback did not. Audit G7:
  under the default environment no test executed any of that code.

- **`clear_binder_encode_nop`** — the one public switch for binder-nop, and it
  can only disarm. The raw `set_binder_encode_nop(bool)` is crate-private now
  (see Removed); a downstream step that must never start under a stale replay
  flag calls the new function first.

- **Fail-closed cross-runtime performance evidence.**
  `bench/paired_cross_runtime.py` and `bench/attn_paired.py` now require an even
  outer-round count of at least 4 (default 6) for exact AB/BA balance and enough
  samples to estimate drift, default to a 1.10x paired max/min spread gate (the
  override is hard-capped at 1.25x), retain every outer-round child median and
  execution order under the explicit `outer_round_values` label, and make the
  median of per-round geometric means the
  canonical aggregate. Artifacts record the Git revision/dirty state and diff
  hash; exact driver and child invocations; executable, comparison script,
  evidence helper, and embedded metallib hashes; OS/device/runtime; load; and
  bounded thermal/power probes. Inherited benchmark/tuning prefixes are cleared,
  the secret-bearing full process environment is never serialized, and local
  home-directory paths are normalized to `$HOME/...` in publishable records
  while the real files are hashed. Output parents must already exist.
  Publication is an atomic same-directory replace
  only after every gate passes, with `<out>.attempt.json` marking an older output
  stale throughout a failed or interrupted rerun. The 2026-09-03
  `gemm_speed_ladder_m5pro_b.json` and `attn_speed_routed_m5pro_c.json` files are
  preserved as historical/non-current audit inputs: both predate this schema and
  contain measurements that breach the new hard spread ceiling.
  Thermal/power/load remain provenance-only because their portable probes do not
  provide a stable numeric acceptance signal; the paired ratio spread is the
  enforceable cross-platform stability gate.

- **`bench_gemm_tnnt_tune` interleaves, and gates on its own baseline.** Every
  arm is timed once per round in exact forward/reverse order pairs
  (`BENCH_ROUNDS`, even and at least 2; default 4), and
  the reported figure is the median of the **per-round ratios** with their
  spread — not a ratio of two blocked medians, which re-introduces exactly the
  drift interleaving removes. When the baseline moves more than 10% across
  rounds the run fails and names the spread rather than publishing ratios. One
  output buffer is now allocated per shape and shared by the baseline and every
  candidate; a fresh one per candidate let allocations pile up until the
  *baseline* drifted inside a single shape block.

  This capability was documented in three places and existed in none of them.
  `bench_gemm_coop_ab`, credited with the interleaving, was never a target in
  this crate — no source, no `[[bin]]`, and `cargo build --bin
  bench_gemm_coop_ab` fails; a stale executable from 2026-08-30 in
  `target/release/` was the only reason the name looked live. Both surviving
  Rust tuning binaries used the blocked protocol their own docs warned against.

- **Tests for the interleave statistics**, in that binary's module. Writing them
  found a defect in the gate as first written: `f64::min` and `f64::max` *ignore*
  a NaN operand rather than propagating it, so a run producing a NaN timing
  scored a spread of 1.00 and passed. `spread` now checks for non-finite and
  non-positive values before folding, and `degenerate_timings_fail_the_gate_
  rather_than_passing_it` holds it there.

- **`tests/docs_name_real_tools.rs`** — the suite now fails if the README or
  `docs/gemm_architecture.md` names a `bench_*` binary with no
  `src/bin/<name>.rs`, or a `bench/*.py` / `scripts/*.py` that is not present.
  Binaries the prose deliberately names as absent sit on a short, justified
  allowlist, and a third test fails if an allowlist entry ever names a binary
  that exists. Verified by injecting a fabricated tool reference and confirming
  the suite goes red. The document outside the crate is skipped with a printed
  notice when absent rather than passing as though it had been checked.

- **Quantized int8 GEMM with fused dequantization**: `nn::gemm_i8_dequant`.
  `int8 x int8` accumulates into `int32` natively on TensorOps, and every
  product fits, so the integer result carries **no rounding at all** — tested by
  exact equality against an integer reference, not a tolerance. The per-column
  dequantization is applied in registers between the accumulate and the store.
  `k` at or above 131072 is refused: `-128 * -128 * 131072` is exactly 2^31,
  one past `i32::MAX`, and would otherwise wrap the accumulator silently.
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

- **Attention tuning knobs, all measured rather than chosen.** `RowsGroups`
  (simdgroups per threadgroup in the row-parallel kernel, compiled per
  instantiation), `DecodeHeadBlock` (which query heads share a threadgroup in
  the KV-split partial pass), and a dispatch-parameter width for the KV-split
  reduce pass. [`bench/attn_tune.py`](bench/attn_tune.py) sweeps all six
  attention knobs in interleaved rounds and refuses to name a winner from a
  partial sweep; `--knob batched` decomposes wall clock into submit and kernel
  instead of tuning anything.
- **`nn::attn_kernel_for`** — the attention routing rule as a pure function, so
  a test can pin it. Both kernels compute the same thing, so a routing
  regression is invisible to every correctness test and shows up only as a
  slower clock.
- Four attention benchmark configs: two large-batch decodes that probe the
  routing rule out to `B*H = 2048`, one with GQA switched off (`Hkv = H`) to
  separate issued K/V traffic from unique, and one at `Hkv = 1` where two
  candidate cache layouts are byte-identical — which is what calibrates the
  run-to-run noise floor at ~3%.

### Fixed

- **The D=256 sliding-window wrapper owns the output-format slot.**
  `flash_attn_swa_with_scalars` binds scalar slot 13 (`out_bf16` for the D=256
  shader, the capacity slot for D=128) to 0 itself, after the callback, because
  it validated `o` as f32. A callback that set the slot, or left a stale 1
  from an earlier dispatch in the scope, had the kernel pack two-byte values
  into the four-byte output; `swa_d256_wrapper_owns_the_output_format_slot`
  does exactly that and reads the f64 reference back. (audit N14)

- **A nearly full constant arena no longer poisons the runtime.** A scope that
  opens with less than `CONST_ARENA_SCOPE_RESERVE` (1 MiB) free drains the GPU
  with a waiting commit, which rewinds the arena, instead of failing its first
  bind and latching `encode_failed` for the rest of the process. A waiting
  commit with no open batch now rewinds the arena too. (audit R8)

- **`self_weak` is a `OnceLock`** set once in `GpuRuntime::new`. The
  poison-tolerant read that could hand `alloc_buffer_kind` a dangling `Weak`,
  silently orphaning every later buffer from residency and the pool, is gone.
  (audit R15)

- **Threadgroup-memory slots have their own bound.** `set_threadgroup_memory`
  checks its index against `THREADGROUP_MEMORY_SLOTS` (32), not the buffer
  argument-table width. (audit R13)

- **`build.rs` sweeps stale metallibs** from `OUT_DIR` before writing the new
  `default-<build id>.metallib`; each build used to leave another ~1.1 MB
  behind. (audit R16)

- **`rms_qkv_rope` `q_only` corrupted Q rows whenever `T * Hq` was not a
  multiple of the SIMD width.** The q-only grid is `T * Hq` threads rounded
  up to the threadgroup width, and the kernel's bounds guard is
  `T*Hq + 2*T*Hkv` with the real `Hkv` bound, so the padding threads fell into
  the K and V branches — which q-only mode points at the Q buffer — and
  re-normalized and re-rotated rows other threads owned. The wrapper now
  rebinds slot 8 (`Hkv`) to zero after the scalar callback; adapters keep
  binding it as documented. `tests/qkv_rope.rs` runs 40 heads through both
  position variants against the f64 reference and checks K and V are
  untouched.

- **Interleaved4 MLX Q4 banks are validated at their tile-padded extent.**
  `Q4MlxBank` was checked against the row-major size, but the `_i4` kernels
  read `rows.div_ceil(4) * 4` rows of nibbles and scale pairs before the
  `row < rows` guard discards the surplus lanes, so a bank sized for six rows
  was accepted and read past by every `_i4` entry point. Validation now takes
  the layout and demands the padded extent (`I4_TILE_ROWS`).

- **Q4 group sizes the kernels cannot address are refused.** `gemv_q4` peels
  each group through 4-byte loads at `packed + row*cols/2 + g*group_size/2`,
  aligned only when `group_size % 8 == 0`; the MLX family peels through
  16-byte loads and the simdgroup kernels stride their scale pointer by
  `512 / group_size`, so its group must be one of 32, 64, 128, 256 or 512
  (MLX quantizes with 32/64/128). `QuantShape` only demanded
  `cols % group_size == 0`, so a group of 4 or 16 ran a misaligned load or a
  wrong stride and returned numbers. `Q4Bank::validate` and
  `Q4MlxBank::validate` enforce the domains; `tests/q4_shape_domain.rs`.

- **Multi-pass `argmax_f32_pass` applied the softcap on every pass.** The
  second pass re-capped the first pass's already-capped partial maxima, and
  `tanh` is not idempotent, so a capped two-pass argmax returned
  `30·tanh(29.65/30) ≈ 22.7` for a logit of 77 instead of 29.65 (the index was
  right, the value wrong). The kernel caps only when `has_idx_in == 0`.

- **Hazard mode left every cross-scope producer→consumer edge unordered.**
  With `TESSL_HAZARD_BARRIERS=1` the per-dispatch auto barrier is skipped and
  only edges *inside* one `with_binder` scope had explicit barriers, while
  consecutive scopes share one Metal 4 encoder in async mode — so
  `cast → cast → matmul` in `gemm_train`, `transpose → gemm`, `gemm →
  add_inplace`, split-K's last partition, and every caller-side op pair were
  unordered. The runtime now carries "an unbarriered dispatch sits behind the
  encoder's point" from scope to scope; the next scope's first dispatch (or
  ICB execute) emits one barrier first, and an explicit `Binder::barrier`
  clears the pending edge, so callers that already barrier their edges pay
  nothing extra. A decode capture records the barrier as the previous
  command's `barrier_after`. `tests/nn_decode_hazards.rs` pins both the edge
  and the no-doubling rule through `infer_trace` counters.

- **Split-K partition count is bounded.** `k_tile` was a fixed 256, so a legal
  deep TN GEMM encoded `ceil(K/256)` barriered dispatches (256 at K = 65536,
  4096 at K = 1M). `splitk_plan` caps the count at 32 and grows the partition
  (a multiple of the 32-wide K step) past that; below the cap the historical
  256-wide partitions are unchanged, so existing results are bit-identical.

- **A device whose simdgroup is not 32 wide is refused by the GEMM launch
  path** instead of being mis-sized silently: every TensorOps kernel here is
  written for `execution_simdgroups<N>` over 32 lanes (`tile_a[4][64]` in the
  simdgroup edge kernel), and `threads_per_tg` /
  `threadgroup_geometry_simdgroup` now return an error for any other width.

- **`bump_reset` returns `Result` instead of panicking** on a live host
  mapping, an open encoder, a poisoned runtime, or a failed replacement-slab
  allocation — the same conditions `bump_alloc_f32` reports as errors.

- **`alloc_temp_f32` no longer serves a busy or poisoned runtime from the
  pool.** Only an exhausted bump arena falls through; the other
  `bump_alloc_f32` errors are propagated, so a poisoned runtime stops handing
  out tensors that nothing can run.

- **`Tensor::read_f32` / `Tensor::write_f32` honour `byte_offset`.** Every
  `GpuBuffer` host accessor spans the whole allocation, so a bump view's
  window could only be reached by hand-offsetting through `buffer`; the bump
  lifecycle test wrote every view's mark to slab offset 0 through
  `write_f32_prefix` and its "reset aliased a live view" check was vacuous
  for seven of eight views. The views now read and write their own window,
  a wrong-length write is refused, and `tests/runtime_lifecycle.rs` pins
  that writing one view leaves its neighbour untouched.

- **A failed or panicking binder closure ends its encoder and command buffer.**
  The batch was left open with the runtime poisoned, so nothing could ever
  end it and the final drop released a Metal 4 encoder mid-recording. The
  runtime now ends both without committing (nothing was submitted, so the
  batch's retained anchors drop safely) on the failure path and again at
  final drop for a batch a caller never committed.

- **`build.rs` prints the metal4.0 diagnostic when it falls back to
  metal3.2**, and reports both diagnostics when the fallback fails too; the
  metal4.0 compiler output used to go to `/dev/null`, leaving a dialect
  downgrade with no visible cause and a fallback failure blamed on the wrong
  standard.

- **The dispatch counter counts every encode attempt.** It skipped failed
  closures in async mode, counted them in sync mode, and both the field and
  its reader could panic on a poisoned mutex; it now counts once per attempt
  in both modes, is private, and never panics.

- Docs that were wrong: the int8 GEMM claimed "no rounding at all" (the sum
  is exact in int32; the single int32→f32 conversion at the store rounds past
  2^24 — `tests/gemm_i8.rs` pins a 16,790,289 sum landing on 16,790,288, and
  the exactness test now compares integers so it can no longer launder the
  rounding); `gemm_batched` credited its bit-identity with `gemm` to sharing
  a tile geometry when the batched kernel is 128×64 only and `gemm` picks
  64×64 for `N <= 512` (the identity rests on MPP's tile-independent
  reduction order, which the test now pins at both a matching and a
  differing geometry); `utils.metal` called the bf16 cast a truncation when
  it is round-to-nearest-even like the host (`tests/bf16_cast.rs` pins
  ties-to-even, overflow and subnormal cases against `f32_to_bf16_bits`);
  the README flowchart named an `nt_splitk` kernel that does not exist and
  showed the column-panel swizzle under the wide geometry only.

- Dynamic threadgroup memory now has one explicit state model across live
  Binder dispatch, capture, direct replay, and frozen ICB replay. Every
  `set_pipeline` call and Binder-scope exit clears nonzero native encoder slots
  as well as host bookkeeping, while repeated dispatches with no intervening
  pipeline selection retain Metal's documented sticky state. Lengths must be
  16-byte aligned, and failures report static, dynamic, and device-limit bytes.
  `DecodeIcbCommand::tg_mem` is now a per-index vector rather than a single
  optional slot: capture previously kept only the last of multiple live slots,
  and public commands could bypass aggregate, duplicate-index, and alignment
  validation before native encoding. This is a source-breaking public-struct
  field change and belongs in the next compatible release decision.
- Tiled sliding-window D=128/D=256 and global D=512 attention now overwrite
  every live output row with zeros in their supported f32/bf16 output modes when
  the device-side live KV length is zero, matching the routed rows/decode paths
  instead of returning early and exposing stale bytes from a recycled output
  allocation.
- `build.rs` no longer copies or renames a generated `default.metallib` into
  `CARGO_MANIFEST_DIR`. Normal builds keep their immutable artifact in
  build-specific `OUT_DIR` storage and export it as `DEP_TESSL_METALLIB`;
  concurrent registry, vendored, and path-dependency builds therefore cannot
  mutate or race on shared source. Offline `TESSL_SKIP_AOT` builds must name an
  existing absolute `TESSL_PREBUILT_METALLIB`, and changes to that file or
  `DOCS_RS` invalidate cached build-script output. `DEP_TESSL_KERNELS` is now
  emitted on normal, offline, and docs.rs paths. The in-tree `tessl-arch02`
  consumer now selects only Tessl's GEMM/runtime owner files from that expanded
  directory instead of globbing inference kernels into its training library.
- NPY input now accepts exactly v1.0/v2.0 C-order ASCII headers, bounds metadata
  at 16 MiB, checks shape and byte products, preflights truncated payloads, and
  uses fallible allocation. Writers and transpose scratch allocation apply the
  same overflow checks; malformed shape tokens and hostile size fields return
  errors instead of panicking or attempting attacker-sized allocations.
- KV timestep stores and fused RMSNorm+QKV+RoPE cache stores now take an
  explicit logical cache capacity. Device-controlled offsets are checked with
  widened, subtraction-form arithmetic, so end, crossing, and `u32::MAX`
  offsets make the complete operation a no-op and cannot write into an adjacent
  slab allocation. Ring densification likewise clamps hostile `filled` state
  and widens cursor/address arithmetic.
- Every active raw `GpuBuffer` operand in `nn` is rejected when it belongs to a
  different `GpuRuntime`, before pipeline lookup or dispatch. The blocked
  gate/up GELU path also rejects `cols > 4096`, its fixed threadgroup x-cache,
  and int8 TensorOps GEMM rejects flattened extents beyond signed 32-bit shader
  indexing.
- Decode attention now encodes its partial and reduction passes in one binder
  scope with an explicit scratch RAW barrier whenever automatic barriers are
  disabled. Q-only fused RoPE no longer binds inactive K/V placeholders, and
  rejects the cache-store variant whose device guard could suppress Q.
- Decode ICB capture now rejects unrecordable raw buffer/resource-ID binds in
  every replay mode. `inheritBuffers` does not re-run the original binding
  callback, so the default prebuilt/sticky paths otherwise consumed missing or
  stale argument-table slots without owning the resource. Coarse range batching
  also treats an immediate-containing open span as unknown until a real barrier;
  it no longer forgets earlier buffers and elides a required dependency edge.
- Metal 4 object lifetimes now follow allocator completion rather than Rust
  closure scope. Raw buffers, pipelines, argument tables, and ICBs bound through
  `Binder`, plus `MTLTensor` objects bound by their resource IDs, are retained by
  the active batch, transferred to the submitted
  allocator slot, and released only after its SharedEvent completes; closure
  errors and panics preserve the same anchors. Raw buffers additionally fail
  before binding unless they belong to the runtime's device and residency set.
  Numeric-address/materialization and arbitrary ICB/table APIs, plus direct
  residency registration/removal, are now crate-private because their complete
  ownership graph cannot be expressed by the safe public signature.
- Final `GpuRuntime` drop waits a bounded 30 seconds for the latest in-flight
  allocator event. On a device timeout it emits one diagnostic and intentionally
  retains the queue, command/allocator state, residency and argument objects,
  constant arena, libraries/pipelines, pending retirements, active batch, and
  raw Binder anchors instead of releasing objects an unretained MTL4 command
  may still reference. Residency add/remove bookkeeping also recovers poisoned
  mutexes so a staged removal is never mistaken for a committed one.
- Final owners of transient Hot buffers, ICBs (including their private argument
  tables/pipelines), and device-backed `MTLTensor`s now retire through one
  post-completion path and leave residency without entering the Cold freelist.
  Re-encoding an `IcbCopySmoke` is rejected before mutating its tape.
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

- **The attention routing threshold was wrong.** `ATTN_SPLIT_KV_BELOW_TG` sent
  `B*H >= 128` decode dispatches to the row-parallel kernel on the evidence that
  the KV-split path lost 0.96 ms to 0.65 there — measured at one launch per
  submit, where ~88% of a decode call is the host round trip and the split path
  pays two submits to the row kernel's one. Measured kernel-only the split wins
  at every batch reachable, 1.4x to 9.9x, with no crossover. The threshold is
  gone.
- **The `tessl-decode` benchmark lane defaulted to chunk 256 for every head
  dim** while the library shipped 128, so a lane labelled "the decode kernel"
  was a kernel no caller reaches. Nothing timing-side could see it — both
  kernels are correct — and the parity dump caught it by showing the routed and
  forced lanes disagreeing where at `Tq == 1` they must be bit-identical.
- **`--dump-parity` wrote whichever implementation ran last** under
  `o_tessl.npy` while the manifest named a different kernel, so the scorer
  reported the row kernel's error under the routed path's name. Each
  implementation is now dumped from its own dispatch, and the dump refuses to
  run under a tuning override at all: a parity artifact describes the shipping
  configuration or it is not written.
- `attn_paired.py` recorded ratios only, so when a geomean moved there was no
  way to say which lane had moved. Both lanes' absolute medians now travel with
  every ratio.

### Changed

- **Constant arena: natural-width alignment.** A u32/f32 scalar costs the
  arena four bytes, an eight-byte payload eight, anything wider sixteen; every
  payload used to be rounded up to sixteen, charging the 16 MiB arena four
  times over for the scalars that make up nearly all of it.
  `a_kernel_reads_a_scalar_at_a_four_byte_offset` is the GPU-side evidence for
  the rule. (audit R8)

- **`setArgumentTable` once per encoder, not once per scope.** The open batch
  carries the latch across `with_binder` scopes on one encoder and clears it
  whenever the encoder is created; a tape-path adopt of another table or an
  indirect execute clears it too. Measured as host time per `copy_f32` scope,
  80k scopes per arm, seven ABBA rounds
  (`bench/results/arg_table_latch_m5pro.txt`): median 2.6% faster inside a
  0.62–1.13 spread, so kept for the removed send, not as a measured speedup.
  (audit R10c)

- **The pool lock no longer spans `addAllocation`.** `alloc_buffer_kind`
  releases the pool mutex before registering residency. (audit R14)

- **Row reductions are simdgroup-first, and rows get at most 256 lanes.**
  `softmax_rows_f32`, `row_sum_f32`, `row_max_f32` and the RMSNorm family
  reduced each row with a threadgroup-memory tree that paid a barrier per
  round — ten rounds and eleven barriers at 1024 lanes. `reduce_row_add` /
  `reduce_row_max` in `kernels/reduce_tree.h` fold each simdgroup with a
  shuffle, publish one partial per simdgroup, and fold the partials with a
  second shuffle across a single barrier; `REDUCE_TREE` is retired. Rows are
  launched with at most 256 lanes (`REDUCE_ROW_MAX_LANES`, was 1024). Paired
  A/B on an M5 Pro, twenty dispatches per command buffer, six order-balanced
  rounds (2026-09-05, `bench/results/row_reduction_simd_m5pro.txt`), old
  time ÷ new time at 4096 rows: softmax 1.13–2.32x up to 4096 columns and at
  the memory roof beyond; `row_sum`/`row_max` 1.4–4.35x; RMSNorm 1.02–1.72x
  (1.21x at the decode shape, 1.47x at 512 rows). Two things that did not
  work are recorded there too: a serial fold of the simdgroup partials lost
  to the tree from 32 simdgroups up, and a register-resident single-pass
  softmax (audit N11) never beat re-reading the row, because the re-reads
  are cache hits and the cost was the reduction. `tests/reductions.rs` and
  `tests/nn_kernels.rs` now cover every lane boundary of the new fold.

- **`gemv_q4` (signed nibble) is one simdgroup per four rows.** It was one
  thread per row with `x` staged in threadgroup memory, so adjacent threads
  read addresses `cols / 2` bytes apart and nothing coalesced — the geometry
  `gemv_q8` left on 2026-08-31. Lanes now stride K eight nibbles (one
  `uint`) at a time across group boundaries, so every lane is busy at every
  admitted group size and the four rows a simdgroup owns share `x` out of
  cache. 1.60–1.83x at 4096–11008 rows and every group size from 8 to 128,
  3.36x at 256 rows, 188–220 GB/s (`bench/results/gemv_q4_simd_m5pro.txt`).
  The 16 KiB `x` cache, the `cols` ceiling the host enforced for it, and the
  opaque 16-byte-alignment error a ragged `cols` used to hit (audit N15) are
  gone. `tests/quantized_gemv.rs` checks it and `gemv_q8` against an f64
  reference at every lane boundary; neither had a numeric test before.

- **The bf16 TN split-K gate opens at K = 12288, not 2048.** The gate rested
  on a sync-per-iteration measurement that its own findings file said could
  not resolve the difference (audit G8). Measured through `gemm_tn_train`
  with the packed encoder (`bench/results/splitk_gate_m5pro.txt`): the
  single coop dispatch beats split-K by 5–23% at K = 4096–8192 on every dW
  shape, split-K first wins at 12288 and pulls away with depth (0.70x at
  32768). `prefer_tn_splitk_bf16` carries the measured threshold for the bf16
  train and accumulate lanes; the f32 exact lane keeps its historical 2048
  gate, unmeasured and untouched as the golden-parity path.

- **`qdot16` and `load_x16_qdot` have one owner.** The MLX Q4 lane dot was
  copied into both `gemv_q4_mlx.metal` and `gemm_q4_mlx.metal`; both now
  include `kernels/q4_mlx_dot.h`, where `qdot16` reads its sixteen nibbles
  with one `uint2` load instead of four `ushort` loads (audit N13). Measured
  neutral (0.99–1.07x) because the simd GEMV already runs at the memory
  roof. `gemv_q8`'s vec4 path likewise takes one `float4` load of `x`
  instead of four scalar loads (audit N12), measured neutral (1.00–1.04x)
  and kept as an instruction-count reduction.

- **The NN coop GEMM tile is chosen by grid fill, not by `N` alone.**
  `nn_coop_kernel` picked the 64×64 sg4 tile for `N <= 512` and the 128×64
  tile for everything else, ignoring `M`; no NN shape below `M = 512` had
  ever been measured, although that is where every prefill chunk, LoRA and
  small-batch step lives. A paired, order-balanced A/B through `gemm` on an
  M5 Pro (`bench/results/bf16_smallm_coop_m5pro.txt`, twenty GEMMs per
  command buffer, six rounds of eight calls, 2026-09-04) found the 64×64
  tile 1.35–1.74× faster at `M <= 64` for every `N` and `K` tried, where a
  128-row tile is at least half padding, and 1.14–1.59× faster wherever
  fewer than 64 of the 128×64 tiles cover the output (`M = 128` at
  `N <= 3072`, `M <= 512` at `N = 768`); the default stays ahead once the
  grid fills (0.82× for the narrow tile at `M >= 256`, `N = K = 4096`). The
  selection is now `n <= 512 || m <= 64 || ceil(m/128) * ceil(n/64) < 64`,
  pinned by `nn_coop_kernel_selects_the_narrow_tile_by_grid_fill`. Shapes
  measured in the 2026-08-30 tune all have 96 default tiles or more and are
  unaffected.

- **The fused RMSNorm+QKV+RoPE kernels give each head row a simdgroup.**
  `rms_qkv_rope`, `rms_qkv_rope_posbuf` and `rms_qkv_rope_kv_store` were one
  *thread* per (token, head) row: at the decode shape the whole layer ran on
  `Hq + 2*Hkv` threads, each walking its 128-256 element row three times
  serially with uncoalesced neighbours, and at prefill the grid held
  `T * (Hq + 2*Hkv)` threads against rows of that width. Each row is now one
  32-lane simdgroup: lane `l` owns the pair `(p, p + D/2)` for every
  `p ≡ l (mod 32)`, so a rotation never needs a value another lane produced,
  adjacent lanes touch adjacent addresses, the sum of squares is a `simd_sum`,
  and the fused cache store writes from the register that holds the value
  instead of re-reading the row. Every element is read twice and written once
  (it was three reads and two writes). Paired, order-balanced, in-process A/B
  against the previous kernel on an M5 Pro, twenty dispatches per command
  buffer, six rounds of ten calls (2026-09-04), previous time ÷ new time:

  ```text
    T=1    Hq=8  Hkv=4 D=256   6.90x       T=64   Hq=16 Hkv=8 D=256  3.33x
    T=1    Hq=16 Hkv=8 D=256  11.46x       T=256  Hq=32 Hkv=8 D=128  1.26x
    T=1    Hq=32 Hkv=8 D=128   3.43x       T=1024 Hq=16 Hkv=8 D=256  1.71x
    T=8    Hq=16 Hkv=8 D=256   3.64x
  ```

  The simdgroup reduction reassociates the sum of squares, a change in the
  low bits; `tests/qkv_rope.rs` compares against an f64 reference (now also
  at `D = 96`, `7`, `130`, `1`, and Gemma's `D = 256, rotary = 64`) and pins
  that a row's result is bit-identical whether it is dispatched alone or
  inside a long prefill, which is what keeps cached keys consistent with the
  queries later compared against them. The host refuses a pipeline whose
  execution width is not 32 rather than reducing the wrong lanes.

- **Host access with no GPU work pending no longer commits residency.**
  `write_f32`, `read_f32`, `zero` and every `contents_*` mapping went through
  `commit_m4(true)`, which flushed the residency set and waited on the shared
  event even when no batch was open and nothing was in flight — so a
  load-then-write loop paid a `requestResidency()` over the whole committed
  set per tensor. Host access now checks for an open batch or an in-flight
  allocator first; with neither it takes the lease and returns. The set is
  still flushed before the first dispatch that uses it, and
  `tests/runtime_lifecycle.rs` pins zero residency flushes across a sixteen
  tensor load loop through `infer_trace`.

- **Crate-internal temporaries skip the host memset.** `cast_f32_to_bf16`,
  `cast_bf16_to_f32`, the transpose scratch in the fallback TN/NT paths and
  the accumulate fallback's GEMM output are written in full by the kernel
  that consumes them, so they are allocated through crate-private `_uninit`
  siblings of `alloc_tensor_*` / `alloc_temp_f32` instead of paying a
  single-threaded memset of the whole extent (50 MB per bf16 training GEMM
  at `mlp_up` size). Public allocations still zero; a recycled pool buffer's
  previous contents are visible only to code that overwrites every element
  before reading.

- **Tensor and buffer ownership checks no longer touch the refcounts.**
  `Tensor::validate` did `Arc::downgrade` (two atomic read-modify-writes) per
  bind and `GpuBuffer::belongs_to` an `upgrade()` CAS loop; both are now a
  liveness load plus a pointer compare.

- **Source-breaking `quant-prep` hardening:** `GpuTensor::tensor` is now
  private, with `GpuTensor::metal()` providing borrowed native interop. Safe
  callers can no longer replace the Objective-C tensor while leaving its
  original storage, runtime, and residency-retirement metadata in place.
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

- **Historical attention artifacts observed 1.2x to 1.7x changes.** K/V/Q reads in
  the row-parallel and KV-split kernels are `float4`, so a lane owns four
  consecutive head dims per step rather than issuing one scalar load each —
  at D=512 that was 32 memory instructions against 32 multiply-adds.
  Simdgroups per threadgroup is compiled per head dim instead of a fixed 8. The
  KV-split reduce pass keeps no accumulator array and takes its width as a
  dispatch parameter, rather than folding every chunk on one simdgroup per
  (batch, head). And the query heads that share a KV head now share a
  threadgroup, so they touch each K/V line while it is still in L1 instead of
  pulling it `H/Hkv` separate times — no threadgroup memory and no barrier.
  In the preserved schema-v1 artifacts, the changed path measured 0.90-0.97x
  MLX on prefill and 0.91-0.97x on decode kernel-only, from 11.1x and 37.9x.
  Those files predate the fail-closed provenance schema above and are retained
  as development history, not a current-tree performance claim.
- **Recorded that `gemma-metal` reaches none of this.** It dispatches
  `flash_attn_swa_h256` / `_h128` / `flash_attn_global_h512` by name through its
  own `KernelId`, with the original `BR = 8`, 32-thread geometry, because the
  scalar-binder form an indirect command buffer needs
  (`flash_attn_swa_with_scalars`) dispatches the *tiled* kernel and the fast
  paths have no `_with_scalars` variant. The KV-split path additionally
  allocates a partials scratch and issues two dispatches. Documented as a gap
  rather than fixed; no end-to-end figure is claimed for `gemma-metal` because
  none was measured.
- Retuned `DECODE_CHUNK_*`, `DECODE_LANES_D128`, `ROWS_GROUPS_*` and
  `DECODE_HEAD_BLOCK_*` against the changed kernels. Every checked-in tuning
  table is generated from its artifact JSON rather than transcribed; historical
  tables remain non-current until a gated artifact reaches `status: published`.

### Changed — source-breaking pre-1.0 surface

- `GpuRuntime::bump_reset` returns `Result<(), String>`.
- `GpuRuntime::metal4` is `pub(crate)`. Its fields reached safe `objc2-metal`
  methods (`setSignaledValue`, `removeAllAllocations`, `endCommandBuffer`,
  `reset`, ...) that can free storage the GPU is still reading, so a public
  field was a safe-Rust route to a GPU use-after-free. No dependent in this
  workspace used it.
- `GpuRuntime::dispatch_count` is private; `take_dispatch_count` is the
  reader.
- `Q4MlxBank` validation takes the bank's `Q4MlxLayout`; every public entry
  point passes the layout it dispatches for, so the change is invisible to
  callers of those functions.

This unreleased set must not ship as another `0.1.x` artifact. It requires at
least the next pre-1.0 minor version, plus a deliberate reconciliation with the
separate standalone tessl checkout. The complete known source-breaking surface
is:

- All 28 public `nn::*_with_scalars` entry points are now `unsafe fn`. Their
  caller-supplied closures can bind arbitrary resources and scalar slots, so
  callers must uphold each function's documented binding, lifetime, ownership,
  aliasing, and dispatch invariants. The corresponding typed safe wrappers
  remain the preferred API.
- `kv_store_timestep`, `kv_store_timestep_pair`, `rms_qkv_rope`, and their
  `_with_scalars` forms now take explicit logical cache-capacity arguments;
  scalar-binding callbacks receive the validated capacity where applicable.
- `DecodeIcbCommand::tg_mem` changed from one optional `(index, length)` pair to
  a per-index vector. Direct struct construction and field-pattern matches must
  migrate to the multi-slot representation.
- `GpuTensor::tensor` is no longer a public field. Read-only access migrates to
  `GpuTensor::metal()` so storage/runtime ownership cannot be bypassed.
- Raw ownership operations that could not express a safe lifetime graph are
  crate-private: Binder argument-table adoption, byte materialization, numeric
  address binds, ICB execution/optimization, and direct runtime residency
  registration/removal. External callers must use owned `GpuBuffer`, `Tensor`,
  `GpuTensor`, and typed dispatch paths.
- The public no-op `npy::seek_noop` and permanently unwired
  `mtl_tensor::try_quant_tensorops_prefill_gemm` functions were removed. NPY
  readers use `Seek::stream_position`; quantized TensorOps readiness is queried
  through `nax_verify_readiness`.

### Removed

- **Public `set_binder_encode_nop(bool)`.** While armed, every public encode
  API on the thread returned `Ok(())` having done nothing, and a caller then
  read stale device memory. Crate-private now; `clear_binder_encode_nop`
  (Added) is the public, disarm-only replacement, and gemma-metal's seven call
  sites use it. (audit R12)

- **The accumulator clear before every cooperative `op.run`.** All eight
  cooperative GEMM kernels zeroed their destination cooperative tensor before
  `run` under `matmul2d_descriptor::mode::multiply`, where `run` assigns every
  valid element and the accumulate and epilogue kernels load the previous C
  afterwards (audit G12). Measured with and without on the bf16 NN kernel
  (`bench/results/coop_clear_m5pro.txt`): within noise either way, so this is
  dead-code removal, not a speedup.


- `npy::seek_noop`, a public function that had no callers and performed no
  operation. NPY reads now use the standard `Seek::stream_position` API where
  the payload boundary must be verified.
- `ATTN_SPLIT_KV_BELOW_TG`. See *Fixed*: it was reading dispatch cost as kernel
  cost, and there is no batch at which the row kernel is the right choice for a
  single-query dispatch.
- `mtl_tensor::try_quant_tensorops_prefill_gemm`. Its entire body was
  `Err("not wired yet")` with every parameter underscored, no caller and no
  test. The host-side quantized prefill path remains unwired; that readiness
  state is now the single `QUANT_PREFILL_GEMM_WIRED` constant reported by
  `nax_verify_readiness`, replacing a callable stub and a second copy of the
  same sentinel. The missing Int4 host descriptor is a separate SDK-binding
  limitation and does not block the existing raw-address TensorOps formats.

[Unreleased]: https://github.com/bharathvbcr/tessl
