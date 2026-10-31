# MLX audit, 2026-10-30

Deeper pass after [`apple_ml_stack.md`](apple_ml_stack.md). The cheat sheet
and the Audit 6 steal list stay in [`mlx.md`](mlx.md). Live step-time gates
stay in [`optimization_map.md`](optimization_map.md). This note does not
re-time anything and does not change a kernel.

Tags: **mlx** means a line read in the MLX tree named below, **measured**
means this machine this day, **project** means a tessl source or doc.
No MLX wheel was imported. No benchmark was run.

## Versions

| Tree | What it is | Tag |
|------|------------|-----|
| Tag `v0.32.4` | Still the latest GitHub release. Commit [`36969c0`](https://github.com/ml-explore/mlx/commit/36969c094354086e15d2a327a9c50feb20b9cf11), released 2026-10-14 by zcbenz. Notes: [v0.32.4](https://github.com/ml-explore/mlx/releases/tag/v0.32.4). | mlx |
| `main` when read | [`1ba7f969`](https://github.com/ml-explore/mlx/commit/1ba7f969ad05d86decfeeb83787cc9b79e440962), 2026-10-30, “[jaccl] Detect dead RDMA links”. Compare API `v0.32.4...main`: `ahead_by` 58, `behind_by` 0. That count is the API field; the commit array in the payload was truncated. The first commit in the range is an untagged patch bump to 0.32.5 (`#4707`). The latest tag and the PyPI upload stay 0.32.4. | mlx |
| Docs site | The unified-memory page fetched for the stack survey is still labeled 0.32.4. | mlx |
| mlx-lm tag | [`v0.32.2`](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2), 2026-10-16, commit `93e6d4a`. Compare API `v0.32.2...main`: `ahead_by` 17. Tip [`736ab0d8`](https://github.com/ml-explore/mlx-lm/commit/736ab0d8c1d0edc79c3d798044fe6904e2fa23be) (2026-10-30) only edits a workflow file. | mlx |
| This GPU | `MTLDevice.architecture.name` is `applegpu_g17s`. `apple10` true, `apple11` false, `metal4` true. OS is macOS 27.0.1 (`26A434`). Re-measured 2026-10-30. | measured |

MLX reads that architecture string itself. Generation is the two digits
before the last character, and the last character is the size letter
(`device.cpp` around the `arch_gen_` assignment). On this name that is
generation **17**, letter **`s`**. A comment in `quantized.cpp` calls the
pair “M5 Max”. The check is the name, so this M5 Pro is in the same bucket.
What string an M5 Max or M5 Ultra reports was not measured here.

`is_nax_available()` (`device.cpp`) is true only when the OS is at least
26.2 and the generation is at least 17, or 18 when the letter is `p`, and
the build was not compiled with `MLX_METAL_NO_NAX`. This host meets the
version and the generation test. That is a reading of the function plus the
measured name, not a run of MLX.

JIT `newLibrary` on macOS 27 requests MSL `(4 << 16) + 1` (`get_metal_version`
in the **tag** `device.cpp`, the encoding from PR
[#4052](https://github.com/ml-explore/mlx/pull/4052)). The same function is
on `main`. The tag also includes “Fixing cooperative tensor build issues on
Mac OS 27” ([#4594](https://github.com/ml-explore/mlx/pull/4594)).

## What the 0.32.4 notes already contain

These were easy to file as “only on main”. They are in the
[v0.32.4](https://github.com/ml-explore/mlx/releases/tag/v0.32.4) list.

| Change | Why it matters here |
|--------|---------------------|
| SDPA VJP [#4563](https://github.com/ml-explore/mlx/pull/4563) | `ScaledDotProductAttentionVJP` accumulates through `sdpa_vjp_blocked`. Vector kernels throw if a logsumexp is requested. Sinks are `NYI` on the VJP. `MLX_SDPA_VJP_FALLBACK=1` forces the fallback. |
| Gated-delta VJP [#4565](https://github.com/ml-explore/mlx/pull/4565) | `GatedDeltaUpdateVJP` in `gated_delta_update.cpp`. Chunk 16 uses `gated_delta_fused_nax_*` and keeps gradient accumulators in float32. `GATED_DELTA_VJP_FALLBACK` forces the fallback. |
| Pre-M5 SDPA D=512 memory [#4518](https://github.com/ml-explore/mlx/pull/4518) | `use_blocked_d512` runs only when `is_nax_available()` is false. On this GPU that path is off. |
| Residency refresh [#4633](https://github.com/ml-explore/mlx/pull/4633) | `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS` defaults to 0. A positive value starts a thread that re-requests residency because, in MLX’s comment, macOS can drop it after the GPU goes idle. `MLX_RESIDENCY_SET_MAX_PCT` defaults to 5: each set’s cap is that percent of `recommendedMaxWorkingSetSize` (5% of the 48 GiB measured here is 2.4 GiB, arithmetic, not a trace). |
| 1-bit affine [#3161](https://github.com/ml-explore/mlx/pull/3161) | Tag `mlx/backend/metal/kernels/quantized.h` instantiates `bits == 1` beside 2, 3, 4, 5, 6, and 8. Python `nn.quantize` still defaults affine to group 64 and 4 bits. |
| Split-K quant partials in float32 [#4641](https://github.com/ml-explore/mlx/pull/4641) | `qmm_splitk` allocates the partials as float32 and rounds once after the sum. |
| Thin GEMM [#4640](https://github.com/ml-explore/mlx/pull/4640), thin M [#4654](https://github.com/ml-explore/mlx/pull/4654) | Small output tiles. `main` also has the NAX thin kernel described below. |
| `MLX_CONV_WINOGRAD` [#4639](https://github.com/ml-explore/mlx/pull/4639) | Switch to disable Winograd. Not a GEMM path. |
| Metal fence deadlock [#4689](https://github.com/ml-explore/mlx/pull/4689) | The commit the tag points at. |
| M5 / M6 split-K test fix [#4677](https://github.com/ml-explore/mlx/pull/4677) | A test, not a new kernel. |

Older tunings are already inside this tag’s dispatch, not a separate drop:
M5 Pro/Max matmul tuning [#3211](https://github.com/ml-explore/mlx/pull/3211)
(2026-03-11) and M5 Ultra non-quantized matmul [#4447](https://github.com/ml-explore/mlx/pull/4447)
(2026-09-05). The thresholds below are the file at `1ba7f969`, which still
contains that logic.

## What is only on main

Read at `1ba7f969`. None of these are in the 0.32.4 changelog.

| Commit | Date | What the message and the file say |
|--------|------|-----------------------------------|
| [#4712](https://github.com/ml-explore/mlx/pull/4712) `qmv_mma` / `qmv_nax` | 2026-10-22 | Few-row quantized GEMM, 6 to 16 rows, large N and K. Affine is excluded. See below. |
| [#4709](https://github.com/ml-explore/mlx/pull/4709) fp4 half loader | 2026-10-21 | `use_fp4_half_loader`: mode is not affine, no global scale, 4 bits, and the activation is float16, or bfloat16 when the mode is `nvfp4`. |
| [#4723](https://github.com/ml-explore/mlx/pull/4723) `sdpa_split` | 2026-10-25 | Mid-row attention for speculative decode. |
| [#4734](https://github.com/ml-explore/mlx/pull/4734) sliding window | 2026-10-23 | `window_size` is a function constant on the NAX full kernel. `sdpa_split` and `use_blocked_d512` both require `window_size <= 0`. |
| [#4728](https://github.com/ml-explore/mlx/pull/4728) wider `qmv_wide` | 2026-10-23 | Affine `tile_cap` is 8. |
| [#4760](https://github.com/ml-explore/mlx/pull/4760) | 2026-10-26 | Keep `g` and `beta` in float32 inside `gated_delta_update`. |
| [#4791](https://github.com/ml-explore/mlx/pull/4791) | 2026-10-29 | 4D per-channel gate indexing in the Metal gated-delta kernel. |
| [#4769](https://github.com/ml-explore/mlx/pull/4769) | 2026-10-29 | Precision loss in `addmm`. |
| [#4756](https://github.com/ml-explore/mlx/pull/4756) | 2026-10-28 | FP8 NaN preservation in `to_fp8` and E8M0 scale dequant. |
| [#4775](https://github.com/ml-explore/mlx/pull/4775) | 2026-10-28 | Affine quant of an all-negative group, Metal and CUDA. |
| [#4717](https://github.com/ml-explore/mlx/pull/4717) | 2026-10-26 | Softmax, logsumexp, RMSNorm, and layer-norm grids past 2^27 rows. |
| [#4767](https://github.com/ml-explore/mlx/pull/4767) | 2026-10-30 | JACCL reports a dead RDMA link instead of crashing. |
| [#4786](https://github.com/ml-explore/mlx/pull/4786) | 2026-10-30 | CI builds and tests for macOS 27. Not a kernel change. |

## Dispatch that decides a steal

Files at `1ba7f969` unless noted. Shapes are MLX’s: `M` is the activation
rows, `N` the output rows, `K` the reduction.

### Dense GEMM (`matmul.cpp`)

`use_nax` is `is_nax_available()` and the dtype is not complex, and it is
not float32 unless `enable_tf32()` is on.

1. NAX off, one batch, small `M×N` tiles (`(M+15)/16 * (N+15)/16` at most
   2048 when the size letter is `s` or `d`, else 1024) and `K` at least
   `max(M, N)` and at least 128: SIMD split-K.
2. NAX on, one batch, `A` not transposed, output not float32, and `N <= 64`
   or (`M <= 64` and `B` is transposed): `steel_gemm_thin_nax_*`.
3. NAX on, one batch, `K` at least three times `max(M, N)`, or
   `max(M, N) <= 1024` and `K` more than twice that: NAX split-K.
4. Otherwise the regular NAX or non-NAX Steel GEMM.

`gemv_wide` is generation 15 and later (M3 and after, by MLX’s own
comment), float16 or bfloat16, `M > 1`, at most three passes, so `M` at
most 15. `M == 1` stays on the ordinary GEMV. Decode is that case.

### Quantized matmul (`quantized.cpp`)

Python modes and defaults (`mlx/python/mlx/nn/layers/quantized.py` on `main`):
affine 64/4, mxfp4 32/4, nvfp4 16/4, mxfp8 32/8. Bits are a separate
argument. Tessl’s loader accepts affine, group 64, 4 bits only
(`docs/qwen35.md`).

`get_qmv_batch_limit` for generation ≥ 17 and a letter other than `d`
(this GPU): 33 when both `K` and `N` are ≤ 2048, 25 when both are ≤ 4096,
otherwise 13.

Order inside `QuantizedMatmul::eval_gpu` for a transposed weight:

1. `M` in 6..16 and `use_qmv_mma`. That function returns false for affine.
   With NAX it wants `N` and `K` ≥ 2048, `N` a multiple of 32, and not
   float32 unless TF32 is on. The header `fp_quantized_nax.h` says
   `qmv_nax` reads nvfp4, mxfp4, and mxfp8. The kernel name is
   `qmv_nax` when NAX is on and `qmv_mma` otherwise.
2. Affine NAX: `M >= 8` and `M` strictly below the batch limit, so at most
   12 when that limit is 13, plus `M * N >= 16384`, biases present, mode
   `affine`, `N` and `K` multiples of 64. A two-way `K` split must also fit
   in 256 threadgroups (`(N/64) * ceil(M/bm) * 2 <= 256`, `bm` 32 when
   `M <= 32`). A wide `N` fails that test and stays on `qmv_wide`. When the
   test passes, `qmm_splitk_nax` runs. Partials are float32. Inside it, `K`
   partitions double up to 4 while the grid stays within 256 threadgroups
   and each partition keeps at least 256 columns.
3. `M` at or above the batch limit: `qmm_splitk` for one batch, else `qmm`.
4. Otherwise `dispatch_qmv`: a quad kernel for tiny `K`, else `qmv_wide`
   when `M >= 2` (affine only from generation 15), else `qmv` / `qmv_fast`.

`qmv` narrows its output tile when the mode is `nvfp4`, `N >= 4096`,
generation is 17, and the letter is `s`. That is the comment’s “M5 Max”
line, and it matches this Pro’s name. It does not run for affine Q4.

### SDPA (`scaled_dot_product_attention.cpp`)

Query length ≤ 8 uses the vector kernels. On a size letter `d` or `s`,
1024 or more visible keys go to `sdpa_vector_2pass`, and so does grouped
query attention at 4096 or more keys.

`use_split` (the speculative path) needs query length ≤ 8, grouped rows in
(32, 128], a head dim in {64, 72, 80, 96, 128, 192, 256}, at least 128
keys, no array mask, no sinks, no window, and no logsumexp. On NAX the
query tile `bq` is 16 and the target grid is 320 threadgroups. Partials,
sums, and maxes are float32.

Query length above 8 uses full attention. On NAX, head dim in
{64, 96, 128, 256, 512} and a non-float32 dtype (or TF32) calls
`sdpa_full_self_attention_nax`. Head dim 256 or 512 uses
`steel_attention_dsplit_` with `bq` 32 at 512 and 64 otherwise, `bk` 32.
Head dims 72 and 80 can be padded to 96 for long unmasked half-precision
attention; the VJP path does not take that pad.

Qwen3.5’s compiled attention head dim is 256 (`docs/qwen35.md`). On this
GPU a bf16 or float16 prefill longer than 8 queries is the NAX
`steel_attention_dsplit_` kernel, not the pre-M5 D=512 blocked GEMM.
The 2026-10-24 profile’s “f32 SDPA versus MLX bf16 SDPA” gap is that
difference of kernels and dtypes. It was not remeasured for this note.
The attn-bf16 lane owns the tile (`Q32K32Sg4`). This audit does not change
it.

### Gated delta

The VJP above is the closest MLX analogue to tessl’s GDN backward. Chunk
16 is the NAX kernel. The forward/VJP sources on `main` also keep the
gate in shapes the 2026-10-29 fix calls out. Tessl’s GDN lane is
`gp/w27-gdn-cols32`. Nothing in that tree was opened or edited for this
audit.

## Steal, skip, already have

| MLX behavior | Tessl today | Decision |
|--------------|-------------|----------|
| Affine Q4, group 64, `w = scale * q + bias`, low nibble first | `Qwen35Model::load_mlx_q4`, pinned to the mlx-lm **0.32.0** layout in `docs/qwen35.md`. | Pin stays. `qwen3_5.py` last changed 2026-10-09 (`e7461f15`, parallel vocab / shard gather, +12/−0). That patch is not the affine Q4 bytes. `utils.py` after mlx-lm v0.32.2 changed only the nvfp4 global-scale path (`33fdd163`). |
| `M == 1` affine GEMV | `gemv_q4_mlx_simd` (`GEMM_Q4_MLX_MAX_M` is 8; `qwen35_q4.rs`) | Already the decode path. `qmv_nax` does not run for affine. |
| Affine `qmv_wide` for `M` in 2..7, `k_lanes` 8, one weight stream over several rows | `gemm_q4_mlx` for `M` in 2..=8, one weight read for those rows | Same job. A port needs a paired bench, not a second kernel beside this one. |
| Affine NAX `qmm_splitk_nax` for `M` from 8 through 12 on large `K`,`N`, and only when the two-way split fits in 256 threadgroups | `M > 8` goes to `gemm_q4_mlx_tensorops` | Overlap is `M` in 9..12 on shapes that pass MLX’s grid cap. Measure before moving the cutoff. The TensorOps kernel is parity-tested and not the live training step. |
| `qmv_nax` / mxfp4 / nvfp4 / mxfp8 / 1-bit | Loader refuses bits other than 4 and any mode other than affine | Skip until a checkpoint in that layout is a goal. |
| NAX full SDPA at head dim 256, half precision | Prefill attention is still the f32 tile in production. bf16 tiles live on `gp/w27-attn-bf16-prefill`. | The 8k gap’s MLX side is this kernel. Finish that lane’s paired bench. Do not retile from here. |
| `sdpa_split` for a handful of queries | No speculative-decode product path in this audit’s scope | Skip until that product exists. Partials in float32 match how tessl already accumulates. |
| SDPA VJP and GDN VJP | Custom training backward. GDN ownership is the other worktree. | Cite only. Do not train with MLX. Do not edit GDN sources from this note. |
| `gemv_wide` on dense half-precision, `M` in 2..15 | Training GEMM is TensorOps. Decode `M` is 1, which MLX itself leaves on GEMV. | Skip for the training step. |
| Residency refresh thread and 5% sets | Tessl has one `MTLResidencySet`. `BufferKind::Hot` stays registered until the last handle drops (`runtime.rs`). `requestResidency` runs at init and when the set is dirty, not on an idle timer. | Different mechanism. The idle-drop comment is a reason to keep requesting residency, not a reason to copy the environment variable. |
| MSL 4.1 on the runtime compiler | `build.rs` AOT-compiles with `-std=metal4.0` and falls back to `-std=metal3.2`. No `metal4.1` flag. | A JIT custom kernel on this OS can ask for 4.1. metal-cpp `8cdde68` (2026-09-22) sets `LanguageVersion4_1 = 262145`, the same `(4 << 16) + 1` encoding. The shipped metallib does not ask for 4.1. |
| JACCL / RDMA | Not a tessl device. | Still out of scope. `main` now fails a dead link instead of crashing; that does not make multi-Mac training a tessl feature. |

## mlx-lm, and the other stacks

mlx-lm `main` in the week of 2026-10-23 through 2026-10-30
([pulse](https://github.com/ml-explore/mlx-lm/pulse)): merged a LoRA
`--clear-cache-threshold`, a shared-memory fix in the KL/JS kernels, sparse
attention for DeepSeek-V3.2, a speculative-processor history fix, and
recursive Flint. The closed issue is a fused Metal KL overflow in the DWQ
backward for models past about 1B. Open work includes Flint’s sliding
window, quantized KV packed width, and a MemT/Qwen3.5 MoE. None of that
changes the affine Q4 bytes tessl loads.

mlx-swift 0.33.0 (published 2026-10-27, tag commit `7ede4fd6`) vendors
mlx `36969c0`, which is v0.32.4. `Package.swift` at that tag still defines
`MLX_VERSION` as the string `0.32.0`. One later commit on `main`
(`78d3cc1d`, JACCL package trait) does not move the mlx gitlink. Swift
matches the mlx release tag, not mlx `main`. It is not a second kernel
source.

Third-party quantizers (mlx-optiq, mlx-kquant, oMLX) showed up in the same
week’s package and release posts. They are not MLX’s dispatcher. A number
from one of those posts is not a tessl result.

## What this audit did not do

- No `import mlx`, no metallib from a wheel, no kernel timing.
- No edit under `kernels/`, `src/`, or the GDN and attn-bf16 worktrees.
- The 58.4 ms / 276 binder gate was not run and is not restated as new.
- M5 Max and M5 Ultra architecture strings were not read off a device.
- macOS 27.2 was not installed. The 0.32.4 and `main` sources above were
  fetched, not built on 27.2.
- Beta pages, metal-cpp, PyPI, and the version pins re-checked the same day
  are in [`beta_nightly_gap.md`](beta_nightly_gap.md).
- mlx-swift 0.33.0’s in-tree `Source/Cmlx/metal-cpp` was compared with
  apple/metal-cpp `8cdde68` on `MTLLibrary.hpp` and the README changelog
  only. Both headers set `LanguageVersion4_0 = 262144`.
  `LanguageVersion4_1 = 262145` is only in the upstream header. The vendored
  README’s newest row is macOS 26 / Apple10. Upstream’s newest row is
  macOS 27.0.1 / Apple11. That is not a file-by-file tree hash. Tessl does
  not compile that copy.
