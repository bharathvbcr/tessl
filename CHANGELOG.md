# Changelog

All notable changes to `tessl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LM-head cross-entropy with gradients (`tessl::cross_entropy`)**:
  `cross_entropy_rows` over the supervised `(row, target)` pairs only, in
  vocabulary chunks, never forming `[rows, vocab]` logits. Loss (mean or sum),
  `dh` for the supplied rows and the full `dW`, exact f32, scratch bounded by
  a `CeWorkspace` sized by rows and chunk. Kernels `ce_gather_rows_{f32,bf16}`,
  `ce_lse_update`, `ce_softmax_grad`. `gemm::cast_bf16_to_f32_into` widens
  into a caller's buffer.
- **GDN training forward and backward (`tessl::gdn_train`)**: the gated
  delta rule at transformers' `torch_chunk_gated_delta_rule` seam (`g`,
  `beta` given, in-kernel q/k l2norm), saving only a state per 64 tokens
  (32 MiB/layer at the 2B's shapes, T = 2048, vs 435 MiB in torch), with a
  deterministic chunk-reverse backward. Kernels `gdn_train_fwd`,
  `gdn_train_bwd`, `gdn_train_bwd_finish`.
  From torch: `tessl_torch.chunk_gated_delta_rule` (transformers'
  signature) and `patch_transformers_qwen3_5()`; C ABI 3 adds
  `tessl_gdn_train_forward`/`_backward` and tensors up to rank 6.
- **Backward of the Qwen3.5 row-local ops (`tessl::qwen35_bwd`)**:
  `rms_norm_bwd` (with an accumulate flag for the residual stream),
  `gated_rms_norm_bwd` (the GDN output norm), `swiglu_bwd` and
  `attn_gate_bwd`, in the forward's own row windows: `dgate`/`dup` and the
  output gate's column land in the fused projections' gradient windows.
  Weight gradients are per-block partials summed in order, no atomics.
  Kernels `qwen35_rms_norm_bwd_f32`, `qwen35_col_sum_blocks_f32`,
  `qwen35_gated_rms_norm_bwd_f32`, `qwen35_swiglu_bwd_f32`,
  `qwen35_attn_gate_bwd_f32`.
- **Causal conv + SiLU backward (`qwen35_bwd::conv1d_silu_bwd`)**: the GDN
  conv as training runs it (zero state, as transformers' no-cache path),
  `dx` into a window of the fused projection's gradient and a
  deterministic `dw [C, KW]`; the pre-activation is recomputed from `x`.
  Kernels `qwen35_conv1d_silu_bwd_dx_f32`, `qwen35_conv1d_silu_bwd_dw_f32`.
- **Attention Q/K norm + partial RoPE backward
  (`qwen35_bwd::attn_qk_norm_rope_bwd`)**: from the q/k/v gradients into
  the fused projection's gradient (q, k, v columns; the gate's are
  `attn_gate_bwd`'s), with deterministic `(1 + w)` gradients for both norms.
  The RoPE angle is now one helper, `qwen35_rope_angle` in
  `kernels/qwen35_act.h`, shared by the forward and the backward. Kernel
  `qwen35_attn_qk_norm_rope_bwd_f32`.
- **Embedding backward (`qwen35_bwd::embed_rows_bwd`)**: `dW[id] += dh[r]`,
  added onto the tied LM head's gradient after `cross_entropy_rows`. The
  host checks the ids and groups the rows by id; each id's rows are summed
  in row order without atomics. `EmbedBwdWorkspace` holds the grouping.
  Kernel `qwen35_embed_rows_bwd_f32`.
- **A Qwen3.5 training step (`Qwen35Model::train_step`, `tessl::qwen35_train`)**:
  transformers' causal-LM loss for one sequence and every parameter's
  gradient, in f32, in each weight's own layout (packed fused projections,
  `[vocab, hidden]` for the tied embedding and head). The forward keeps
  only each layer's input and each layer is rebuilt just before its
  backward (0.38 GiB of activations at T = 2048 on the 2B, against 8.54 GiB
  saved, for 10% more time); it uses `gdn_train`, `attn_train` and
  `gdn_gates`, and the backward chains the CE, row-local, conv, gate, Q/K,
  attention and embedding backwards with exact-f32 GEMMs. Checked against
  transformers' autograd on a committed tiny model
  (`tests/fixtures/qwen35_train/`, from
  `tools/qwen35_ref/make_train_fixture.py tiny`), and on the real 2B by an
  ignored test (`make_train_fixture.py 2b`): loss within 4.6e-5, gradients
  within 3.9e-3 of each parameter's largest, a difference finite differences
  of tessl's own loss attribute to the two f32 forwards, not the backward
  (`real_2b_gradients_are_those_of_tessls_forward`);
  `tools/qwen35_ref/train_noise_floor.py` measures transformers' own
  run-to-run gradient disagreement for comparison.
- **A Qwen3.5 model's parameters under transformers' names
  (`tessl::qwen35_params`)**: `parameter_table`, `read_parameters`,
  `read_gradients` and `write_parameters` copy between the model and caller
  f32 tensors on the GPU, with transformers' shapes and values (norms as
  `w`, not the stored `1 + w`; linear weights as `[in, out]` windows of the
  packed projections), honouring a caller's byte offset and checking every
  tensor before writing any. `gemm::transpose_f32_into` is the checked GPU
  transpose (the TN/NT fallbacks now share it).
- **Training attention (`tessl::attn_train`)**: `attn_train_forward` is
  `attn_prefill`'s tiled kernel at its default geometry with each row's
  log-sum-exp written out (new entry point
  `qwen35_attn_tiled_lse_h256_q32_k32_sg4`; the four existing entry points
  are unchanged and bind no new slot), and bit-identical O.
  `attn_train_backward` is FlashAttention-2's backward on the matrix units
  (`kernels/qwen35_attn_bwd.metal`): dQ per query block, dK and dV per key
  block walking the KV head's query heads in order, each gradient written
  once, no atomics. Kernels `qwen35_attn_bwd_dvec_f32`,
  `qwen35_attn_bwd_{dq,dk,dv}_h256_q32_k32_sg4`.
- **GDN gates for training (`qwen35::gdn_gates`, `qwen35_bwd::gdn_gates_bwd`)**:
  `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)` written
  dense `[rows, heads]` for `gdn_train`, and their backward into the fused
  projection's a/b gradient columns with deterministic `dA_log`, `ddt_bias`.
  torch's softplus and the log decay move to `kernels/qwen35_act.h`
  (`qwen35_softplus`, `qwen35_log_decay`), shared with the inference loads.
  `qwen35_bwd::copy_cols` moves a column window between layouts. Kernels
  `qwen35_gdn_gates_f32`, `qwen35_gdn_gates_bwd_f32`, `qwen35_copy_cols_f32`.
- **qwen35_gdn.metal compiles under -Wall -Werror again**: the 16-column
  scan's threadgroup-memory constant gets the 32 KB static_assert its
  siblings have (the current compiler rejected it as unused).
- **`bench_qwen35_train`**: every training op at Qwen3.5-2B's shapes behind a
  NaN-poison gate, and `--step=N` for a whole `train_step` on the checkpoint
  (7.2 s at T = 2048 on an M5 Pro; per-op numbers in docs/qwen35.md).
- **Randomized-shape stress for the backward kernels**
  (`randomized_shapes_stress` in `tests/qwen35_bwd.rs`, `tests/attn_train.rs`):
  shapes drawn within each kernel's contract, each draw checked like the
  targeted tests; `TESSL_FUZZ_ITERS` / `TESSL_FUZZ_SEED` scale it (500 draws
  per sweep pass).
- **msl_emu host contract**: macro instantiations are expanded by a small
  preprocessor (nested macros, object-like parameter macros), the
  TensorOps backward's signatures and `src/attn_train.rs`'s binds are
  checked, and a host bind of a kernel no inspected source declares fails.
- **torch binding (`python/tessl_torch`) over a C ABI (`tessl::capi`)**:
  `tessl_torch.cross_entropy(hidden, weight, targets, mask)`, a
  `torch.autograd.Function` over MPS tensors, loaded with `ctypes` (no C++
  extension, no new dependency). The crate now also builds a `cdylib`
  (`libtessl.dylib`); the ABI is versioned, catches panics at the boundary,
  and refuses calls from a thread other than the handle's.
- **The Qwen3.5 text forward (`tessl::qwen35_model`)**, loaded from the
  Hugging Face checkpoint, checked end to end against transformers
  (`tests/qwen35_model.rs`, opt-in): F32 within 2.3e-6 per layer, Bf16 18x
  closer to fp32 than transformers' own bf16. See `docs/qwen35.md`.
- **`Qwen35Config::from_config_json` / `from_config_file`**: any Qwen3.5 text
  `config.json` (nested `text_config` or text-only), with every feature the
  forward lacks refused by name. The published 2B config parses to exactly
  `qwen35_2b()`; other sizes are parsed, not yet run against transformers.
  The JSON parser is the safetensors one, moved to a crate-private
  `json` module with per-format syntax limits.
- **`tessl::safetensors`**, a strict `.safetensors` reader (exact offsets,
  bounded header, JSON subset, reads through the handle it opened).
- **`qwen35::residual_add`** (`qwen35_residual_add_f32`), the exact-f32
  residual add for strided windows.

### Changed (breaking)

- **C ABI 5** (was 3; the binding and library refuse each other across
  versions, so rebuild `libtessl.dylib` with the binding). It adds a
  Qwen3.5 model handle: `tessl_qwen35_load`, `_train_step`,
  `_param_count`, `_param_info` (`TesslParamInfo`), `_copy` (read
  parameters, read gradients, write parameters) and `_free`, with the
  runtime's thread affinity. `tessl_torch.Qwen35` wraps it:
  `parameters()`, `train_step(ids)`, `grads()`, `load_parameters()`, under
  transformers' names and values, for a torch optimizer. Checked against
  transformers' own autograd before and after an AdamW step written back
  (`python/tests/test_qwen35.py`).
- `Tensor::from_mtl_buffer` is an `unsafe fn`: tessl cannot see another
  queue's work on a wrapped buffer, and its `# Safety` section states the
  cross-queue contract.
- `GpuRuntime::metal4` and `Metal4EncodePackage` are crate-private;
  `shared_event()` and `last_signaled_value()` remain the public handoff.
- `cross_entropy_rows` takes the hidden states and weight as `Tensor`s
  (`CeHidden { rows, off }`, `&Tensor`), so views at a storage offset work.
- `ParamsBuffer::push_u32`/`push_f32` take the host-access lease: they wait
  for encoded work and fail with "runtime busy" inside an encoder closure.

### Changed

- `Qwen35Model` in `Precision::F32` holds the tied embedding as one f32
  `[vocab, hidden]` table (the gather reads it through the new
  `qwen35_embed_rows_f32`, training's cross-entropy as its weight, and
  `forward()`'s head through `gemm_nt_f32`) instead of a bf16 gather table
  plus an f32 `[hidden, vocab]` head: 1 GB less on the 2B, a parameter write
  is exact, and gradients are taken at the parameters themselves rather
  than at a bf16 rounding of the embedding. `qwen35::embed_rows` accepts an
  f32 table. The bf16 model is unchanged.

### Fixed

- Host mappings of a GPU-private buffer are refused instead of building a
  slice over the null `contents()`.
- Two wraps of one `MTLBuffer` are one allocation: overlap checks compare the
  buffer, not the wrapper, and residency is counted per buffer, so dropping
  one wrap no longer evicts it from under another.
- A `ParamsBuffer` slot is no longer rewritten under an encoded dispatch that
  still reads it.
- `npy`: only versions 1-3, a bounded UTF-8 header, `fortran_order` exactly
  `True`/`False`, checked element counts, and a payload that must be exactly
  the rest of the file (checked before allocating).
- MLX Q4 tiled banks (`Interleaved4`, the blocked layout) must hold their
  padded last tile; a bank sized for the unpadded rows read past its end.
- `qwen35::pack_linear_weights` checks its width sum and skips empty output.
- `attn_qk_norm_rope` rejects a zero-capacity cache and cache positions past
  `u32`.
- `GdnWorkspace::check` computes its sizes with checked products.
- `Tensor::from_mtl_buffer` checks bounds before registering residency, so a
  rejected wrap leaves no residency entry.
- `execute_icb` requires the ICB to be in the residency set, as
  `optimize_icb` already did.

- **Qwen3.5 layer kernels (`tessl::qwen35`)**, the Metal replacement for the
  pure-torch GDN fallback transformers runs on MPS. See `docs/qwen35.md`.
  - Chunked gated delta rule as two dispatches: `qwen35_gdn_chunk_prep`
    (parallel over chunks: l2norm, Qwen3.5's gates, both 64x64 products, and
    the triangular solve in threadgroup memory) and `qwen35_gdn_chunk_scan`
    (the sequential state pass, on `simdgroup_matrix` tiles).
  - `qwen35_gdn_recurrent`, a token-by-token decode that reads a shared
    snapshot state without writing it, so one prefix serves many questions.
  - `qwen35_conv1d_silu` (prefill and decode, snapshot-capable),
    `qwen35_gated_rms_norm_{f32,bf16}`, `qwen35_attn_qk_norm_rope` (zero-centred
    norm, transformers' partial RoPE, K/V cache store),
    `qwen35_attn_gate_{f32,bf16}`, and `qwen35_score_rows_{f32,bf16}` (final
    norm + LM head for the answer tokens at the slot rows only).
  - Fused-projection helpers: weight packing, column layouts that every kernel
    reads in place, and `project_residual` (the residual add as a GEMM
    epilogue).
- **Prefill attention on the TensorOps matrix units** (`qwen35_attn_tiled_h256_*`,
  `qwen35::attn_prefill`): `flash_attn_rows`' contract at head_dim 256 and
  `window = 0`, with `Q·Kᵀ` and `P·V` on `matmul2d` in exact f32 over
  query-by-key blocks (four geometries, `qwen35::AttnTile`,
  `attn_prefill_with_tile`) and an f32 online softmax between them. It skips key blocks
  above the diagonal. `bench_qwen35_layers` now times it, and
  `--attn-tile=LABEL` / `--attn-rows` select another tile or the scalar
  kernel.
- **`qwen35_gdn_chunk_scan_bv16`**, the chunked GDN scan in 16-column value
  slices, selected by `GdnWorkspace::with_scan_slice(GdnScanSlice::Cols16)`.
  It launches twice the threadgroups and is bit-identical to the 32-column
  scan. `probe_gdn_scan` found the 32-column scan underfilling the GPU at
  batch 1; `bench_qwen35_layers --gdn-scan16` selects it. The default stays
  32 until a clean A/B.
- **`qwen35_swiglu_{f32,bf16}` / `qwen35::swiglu`**: Qwen3.5's MLP activation,
  `silu(gate) * up`, from f32 column windows straight to bf16 for the down
  GEMM, replacing `nn::mlp_silu` plus a cast pass in `bench_qwen35_layers`
  (`--mlp-unfused` times the old path).
- `bench_qwen35_layers` times the GDN chunked rule's two dispatches apart
  (`gdn chunk prep`, `gdn chunk scan`) through the doc-hidden
  `qwen35::gdn_chunk_phase`. Run in order, they are bit-identical to
  `gdn_chunk_forward`.
- **Shared-prefix attention for Qwen3.5** (`qwen35_attn_prefix_rows`,
  `qwen35::attn_prefix_rows`). Many questions over one prefilled context now
  share the 6 full-attention layers' KV prefix at batch stride 0, as the GDN
  and conv state already did through `StateIn::Snapshot`. Each row keeps its
  own suffix cache. The result is bit-identical to `nn::flash_attn_rows` over
  a per-row copy of the prefix. `qwen35_attn_prefix_decode_{partial,reduce}`
  (`qwen35::attn_prefix_decode`) are the split-KV single-query form, and are
  bit-identical to `nn::flash_attn_decode` the same way.
  `qwen35::attn_qk_norm_rope_suffix_posbuf` is the suffix writer with the
  position in a device buffer, for ICB replay.
  Ragged continuations take per-row lengths and positions:
  `attn_prefix_rows_varlen`, `attn_prefix_decode_varlen` and
  `attn_qk_norm_rope_suffix_rows`. Each row is bit-identical to that row
  alone. The shared-prefix kernels gained a `row_stride` slot, and both
  `qwen35_attn_qk_norm_rope` kernels gained `pos_stride` (buffer 21).
- **Ragged rows for the stateful layers:** `conv1d_silu_varlen`,
  `gdn_chunk_forward_varlen` and `gdn_recurrent_varlen` take per-row
  `seq_lens`, so the questions in a batch can have different lengths. Each
  row, final state included, is bit-identical to that row run alone. The
  kernels gained a `seq_lens` slot (flag bit 4; `use_lens` in the prep).
- **`qwen35_embed_rows_bf16` (`qwen35::embed_rows`)**, the embedding gather
  from the bf16 vocabulary table on the device. It is exact, and a bad id
  gives a NaN row. A forward no longer needs a host gather and a
  commit-and-wait before its first layer. `qwen35::attn_qk_norm_rope_suffix` caches a
  suffix at prefix-relative slots with absolute RoPE positions, through a new
  `slot_base` argument (buffer 20) to both `qwen35_attn_qk_norm_rope` kernels.
- **`tools/msl_emu`**, a CPU emulator that runs the kernel sources as C++ with
  real threadgroup barriers and simdgroup collectives, and a driver that checks
  them against transformers' own Qwen3.5 code.
- Hardening pass over the Qwen3.5 kernels after an independent audit:
  - **Numerics:** overflow-free sigmoid/SiLU. A `log1p`-accurate softplus (fast
    `log(1+e)` was 1-60% off where `a + dt_bias` routinely lands). Scoring keeps
    invalid answers out of the softmax and stores NaN as integer bits, which fast
    math cannot fold away.
  - **Scan and prep tiles:** the `const` diagonal matmul (a likely compile error)
    is gone. Threadgroup strides are padded against bank conflicts, the per-row
    decays are cached, and redundant tile loads are removed.
  - **Kernel structure:** thread counts are named constants with `static_assert`s
    on the lane mappings, and the conv runs with 256-thread groups.
  - **Host fixes:** the in-place GDN state is now checked against the inputs too.
    `score_answer_rows` rejects `rows == 0` / `vocab == 0`, which read out of
    bounds before. `AttnTargets` derives the KV capacity from the caches instead
    of taking one that could disagree with flash attention. Layouts get
    validating constructors and `GdnProjLayout::dims`. `seq = 0` with a
    `state_out` now copies the state through. Host arithmetic is checked
    throughout.
  - **Emulator:** threadgroup-order invariance, ASan/TSan builds with exactly
    sized allocations, fast-math ulp noise, a host-contract check of binds and
    constants, and a `kernel-emulator` CI job.
- `tools/msl_emu/check_qwen35_model.py`: a whole random `Qwen3_5ForCausalLM`
  through the kernels and tessl's own `flash_attn_rows` (emulated), against
  the model's logits, for prefill, cached decode and a shared snapshot. The
  Mac suite gains the same attention seam:
  `attention_layer_through_flash_attn_rows`.
- Scoring re-derives validity from the indices instead of storing flags. This
  restores the full `MAX_ANSWERS = 4096` inside 32 KB of threadgroup memory,
  and a unit test pins it. `attn_qk_norm_rope` accepts `batch = 0`. Softplus
  uses `precise::exp`.
- `qwen35_attn_qk_norm_rope_posbuf` / `attn_qk_norm_rope_posbuf`: the RoPE
  position comes from a device buffer, so a decode loop replayed from an ICB,
  which freezes scalar binds, advances correctly. Both variants form the
  position in 64 bits before the capacity check. In 32 bits, an offset near
  `u32::MAX` wrapped to slot 0 and passed the check.
- `tools/msl_emu/dialect_lint.py` (in CI): every MSL construct the Qwen3.5
  kernels use that no compiling tessl kernel uses must be on a reviewed list.
- Emulator cases at Qwen3.5's real head counts and at T=4096.
- `.github/workflows/metal-compile.yml`: Apple's Metal compiler on GitHub-hosted
  macOS, never the self-hosted runner. It compiles the Qwen3.5 sources under
  `-std=metal4.0 -Wall -Werror`, links them, checks every entry point is
  exported, and builds everything with no `metal3.2` fallback. Its first run
  found one unused constant; every other line compiled clean.
- `tests/qwen35_kernels.rs` with transformers-generated goldens in
  `tests/fixtures/qwen35/` (`scripts/gen_qwen35_fixtures.py`). The Qwen3.5
  sources join the widened-index-arithmetic inspection in
  `tests/shader_index_arithmetic.rs`.

## [0.2.0] — 2026-09-18

Fail-closed encode / attention / quantized paths, Tensor metadata hygiene, and
SharedEvent interop with sparsl. The minor version moves because public
`Tensor.shape` / `Tensor.byte_offset` fields become `pub(crate)` (use
`shape()` / `byte_offset()`), `BufferKind` gains `External` and is marked
`#[non_exhaustive]`, and several previously-accepted shapes / aliases are now
refused with errors.

### Added

- **Flash attention decode and rows kernels**, with host routing that picks the
  right specialization and refuses aliased / dimensionally inconsistent outputs.
- **`BufferKind::External`** and `Tensor::from_mtl_buffer` for zero-copy
  MTLBuffer handoff (BINN / sparsl SharedEvent paths).
- **Dispatch geometry validation** and traced kernel names so a bad grid fails
  closed instead of writing partial tiles.
- **Q4 / Q8 shape-domain checks** and widened index arithmetic in the Metal
  kernels that previously truncated under large extents.
- **Release tooling:** `scripts/ci_local.sh`, `scripts/check_release_ready.sh`,
  and a tag-triggered `.github/workflows/release.yml` that publishes to
  crates.io and creates the GitHub Release from this changelog.

### Changed

- **`Tensor.shape` / `Tensor.byte_offset` are no longer public fields.** Mutating
  them by assignment could desynchronize the logical view from the buffer;
  construct via allocators / `try_view` and read via accessors.
- **`BufferKind` is `#[non_exhaustive]`** so exhaustive matches stop breaking on
  every new kind.
- **Busy bump-reset and poisoned `alloc_temp` fail closed** instead of
  fallthrough.
- **Hazard tracking** (`hazard_pending`) so decode / attention writers cannot
  race a subsequent reader without a barrier.

### Performance (Apple M5 Pro, 2026-09-18)

Sequential A/B vs clean `v0.1.4` HEAD, `bench_gemm_sweep` with
`BENCH_WARMUP=10` / `BENCH_ITERS=30`, shapes `512³,1024³,2048³`, cooled
cur→prev pass:

| shape / backend | prev (ms) | 0.2.0 (ms) | Δ |
|---|---:|---:|---:|
| 1024³ tensorops-bf16 | 0.217 | 0.196 | **−10%** |
| 1024³ tensorops-f32 | 0.508 | 0.514 | +1% |
| 2048³ tensorops-f32 | 3.883 | 3.859 | ≈0% |
| 2048³ tensorops-bf16 | 0.876 | 0.888 | +1% |

Small square shapes sit near the ~0.25 ms dispatch floor and are not quoted as
wins. Host load was elevated during measurement; treat ±~10% as noise.

## [0.1.4] — 2026-09-14

Manifest metadata only. No code, API or behaviour change; the compiled crate is
identical to 0.1.3.

### Added

- **`homepage` in `Cargo.toml`.** 0.1.3 put a website badge in the README, which
  only reaches someone already reading the rendered README. crates.io shows a
  Homepage link in the crate page sidebar when the manifest declares one, and
  this crate declared none — so the showcase was missing from the one place a
  reader scanning the page looks for it. `sparsl` gained the same field in the
  same change.

## [0.1.3] — 2026-09-14

Documentation and repository hygiene only. No kernel, API or behaviour change,
so the compiled crate is identical to 0.1.2. It ships because the README is
rendered on the crate page, and the links below are the reason to publish it
there rather than only on GitHub.

### Added

- **`documentation` in `Cargo.toml`, and the published docs linked from the
  README.** The crate page fell back to docs.rs implicitly and the README named
  neither, so the one surface a GitHub reader lands on had no path to the API
  reference. The README now carries crates.io, docs.rs, CI and license badges,
  a nav line to docs.rs and the three `docs/` deep dives, and an API reference
  row at the top of the Documentation table.
- **The interactive benchmark showcase at
  [tessl.vbcr.dev](https://tessl.vbcr.dev/) linked from the README**, as a badge
  and as the first entry in the nav line. The measurements the README quotes in
  prose are explorable there; nothing else pointed to it.

### Fixed

- The README Status row said `0.1.0` while crates.io served `0.1.2`. Corrected,
  and linked to the crate page so the number has somewhere to be checked
  against.

## [0.1.2] — 2026-09-01

Documentation only.

### Fixed

- **Internal monorepo paths and audit numbering leaked into the published
  docs.** Eight doc comments referenced `arch_02_value_resid/metal-native`,
  `arch_02`, and "Audit 4 P0 / P1 / 4 6 / 7" — a directory layout and an
  internal review series that mean nothing to a reader arriving from crates.io,
  and cannot be looked up. Each cited note is replaced by the reasoning it stood
  for, so the *why* survives without the unresolvable citation: cold buffers
  recycle only after the command buffer completes because releasing earlier
  hands memory back while the GPU may still read it; one encoder is packed
  across dispatches because opening one per dispatch costs setup on every op.
  References to `gemma-metal` are kept, since that is a real crate and explains
  where the promoted kernels came from.

## [0.1.1] — 2026-09-01

One build-script fix, and documentation. No kernel or API changes.

### Fixed

- **A `DOCS_RS=1` build left the crate permanently broken until `cargo clean`.**
  That branch bakes `TESSL_METALLIB=""` through `cargo:rustc-env`, and
  `build.rs` never declared `rerun-if-env-changed=DOCS_RS` — so cargo had no
  reason to re-run it when the variable disappeared, and every later build kept
  the empty path. The crate still compiled; every `GpuRuntime::new()` then failed
  with `metallib missing at ` and an empty path, thousands of lines from the
  cause. Found by running the docs.rs simulation immediately before the test
  suite, which turned 88 passing lib tests into 62 failures on a tree whose only
  other change was doc comments.

- **The README's headline benchmark claim was wrong.** It read 1.11x against
  PyTorch MPS on bf16 with a worst shape of 1.01x, which reads as "never loses".
  Re-measured with the same paired harness on the same machine: **1.03x, losing
  on 4 of the 8 shapes**, worst 0.86x. The tf32 lane (2.11x, wins every shape)
  and the MLX comparison (2.55x) hold up. The table now carries a "shapes below
  1.0" column, because a geomean above 1.0 with half the ladder underneath it is
  not the same result as a clean win.
- **The published f32 and tf32 peak throughputs were not reproducible.** 10,897
  and 18,040 GFLOP/s; the crate's own committed sweep records 6,606 for f32 and
  a fresh run gives 6,431. Replaced with measured values rather than averaged.
- Two remaining references to `bench_gemm_coop_ab`, a binary that does not
  exist, in the benchmarking doc's tool table and prose. The paired A/B lane is
  `bench_gemm_tnnt_tune`.
- Stale counts in `docs/verification.md` (199 tests across 14 files; it is 228
  across 18) and in the README's known-gaps table.

### Added

- **A docs.rs landing page written for someone outside the monorepo.** Platform
  constraints first, since this crate does not build on Linux or Intel Macs at
  all, and the Metal Toolchain is a separately downloaded Xcode component rather
  than part of Xcode. Adds two `no_run` quickstarts, a module map, the feature
  table, the measured cross-runtime results, and the encode-model note that
  `async_encode` is off by default and worth roughly 49x on a small kernel.
- Contributor notes covering `--test-threads=1`, why `build.rs` tracks every
  `.metal` and `.h` individually, that a name check is not a correctness test,
  seeding outputs with a sentinel, and testing every arm of a kernel family.
- Real module docs for `npy`, `ops` and `infer_trace`, which had one line each.

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


[0.2.0]: https://github.com/bharathvbcr/tessl/releases/tag/v0.2.0
[0.1.4]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.4
[0.1.3]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.3
[0.1.2]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.2
[0.1.1]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.1
[0.1.0]: https://github.com/bharathvbcr/tessl/releases/tag/v0.1.0
