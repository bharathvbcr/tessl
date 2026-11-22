# tessl limitations and known issues

_Read at tessl fba6fc0 plus the uncommitted working tree, 2026-11-22. Claims carry file:line evidence; where code and this page disagree, the code wins._

This page lists what tessl does not do, what it refuses, what was retired, and which issues were open in its own reports at the last reading. For how to use the crate see [developer-guide.md](developer-guide.md). Nothing here was re-run for this page: no build, test or GPU job was started, so every "open" below means "open in the cited source at the cited date, and not re-run". Where the current code was read and differs from an older report, the table says so.

Provenance of the report files. `BUILD_REPORT.md`, `BRANCH_REPORT.md`, `AUDIT_REPORT.md` and `tasks/` are listed in `.gitignore` (`.gitignore:117`, `:122`, `:155`, `:158`) and `git ls-files` returns none of them. They exist in the working directory of the checkout that was read and are not on GitHub, so this page cites them by name, not by link.

## 1. Platform limits

| Limit | Detail | Evidence |
|---|---|---|
| Apple silicon, macOS 26+, Xcode 26 Metal Toolchain | Nothing runs without a Metal 4 device. There is no CPU fallback, by decision: a silent CPU path would make every GPU benchmark meaningless. | `README.md`, "Requirements" and "Known Gaps" (No CPU fallback row); `AGENTS.md:39` |
| No Linux or Windows | `objc2-metal` is an unconditional dependency and `build.rs` drives `xcrun metal`. docs.rs is pointed at `aarch64-apple-darwin` and `build.rs` skips shader compilation there. | `Cargo.toml` (`[package.metadata.docs.rs]`); `build.rs:68`; `go/README.md`, "Not done" |
| Metal 4 only | The classic command-queue path is deliberately absent. | `Cargo.toml:86` |
| TensorOps needs hardware support | `GpuRuntime::has_tensorops()` gates it; `Qwen35Model::load_mlx_q4` returns an error when it is false. | `src/runtime.rs:1501`; `src/qwen35_model.rs:692-700` (the `has_tensorops` check) |
| Single thread per runtime | `GpuRuntime` is `!Send` and `!Sync`; host access fails instead of blocking when the runtime is busy or re-entered. | `src/runtime.rs:798-815` |
| Go binding: darwin/arm64, Go 1.27+, Rust 1.100+ | The package refuses to build elsewhere. The Go floor 1.27 is in the uncommitted `go/go.mod`; the committed file says 1.26. | `go/README.md`, "Requirements and the one command"; `go/go.mod:3` |
| Go binding is in-process | A SIGSEGV, abort or GPU driver fault ends the Go process; only a Rust panic is contained. | `go/README.md`, "What it does not contain" |
| Go cancellation | A call a worker has begun runs to completion; the caller is released at its deadline and the result is discarded. No streaming and no mid-flight cancel. | `go/README.md`, "Cancellation" and "Not done" |
| GPU CI is not unattended | Hosted `macos-26` runners can compile shaders but cannot execute them. The `gpu` and `bench` jobs are gated on `vars.HAS_METAL_RUNNER` and no runner is registered. | `.github/workflows/ci.yml:448`, `:640`; `README.md` Known Gaps (owner action, card ft-2ffbdcd5) |
| No off-device numeric check of TensorOps kernels | `tools/msl_emu` cannot execute TensorOps, so `attn_prefill` and `encoder_attn_tiled_*` are checked against f64 only on a device. | `README.md`, "Known Gaps" (attn_prefill row) |
| Sandbox hides the device in hosted tests | `scripts/hosted_tests.sh` runs device-free tests under a sandbox that denies IOKit. | `docs/WORKFLOWS.md`, tier 2 |
| Machine-load limits | GPU jobs are expected under a machine-wide lock with a 1-minute load gate of 12 and a process-footprint cap. These are conventions of the owner's machine, not crate behaviour. | `docs/WORKFLOWS.md`, section 2 |

## 2. Unsupported shapes, dtypes and formats

Refusals are by design: the entry points fail closed before encoding (developer guide, section 7.4). The list below is what the code refuses or the model docs name as unsupported.

### GEMM and tensors

| Refused | Evidence |
|---|---|
| Anything but non-empty rank-2 tensors in `gemm` and its variants | `src/gemm.rs:44-55` (`validate_gemm`) |
| An operand of more than `i32::MAX` elements (MPP uses signed 32-bit extents) | `src/gemm.rs:34-38` (`require_i32_extent`) |
| An operand view whose byte offset is not a multiple of 16 | `src/gemm.rs:102` (`GEMM_VIEW_ALIGN`) and `require_view_alignment` below it |
| Operands from different runtimes; an output that overlaps an input | `src/gemm.rs:61`, `:75` |
| A non-f32 `C`; f32-only paths with bf16 operands | `src/gemm.rs:64` |
| Tensor rank above 6 across the C ABI | `TESSL_MAX_DIMS = 6`, `src/capi.rs:94` |
| `DType` other than f32, bf16, f16 | `src/tensor.rs:19-28`; which kernels accept f16 was not enumerated (unverified) |

`TESSL_HAZARD_BARRIERS=1` is unsafe and must not be enabled: it removes the always-on dispatch barrier and produced wrong results in 300 of 300 repetitions on an M5 Pro in the measurement quoted by the README (`README.md`, "Environment Variables").

### Qwen3.5

| Refused or unsupported | Evidence |
|---|---|
| MoE, attention bias, an ungated attention output, a non-SiLU activation, `mlp_only_layers`, non-default RoPE, a key head dim other than 128, an attention head dim other than 256 | `src/qwen35_model.rs:190-191` (doc), `:241-242`, `:275-278`, key head dim `:319-325`, attention head dim `:408-415`; [qwen35.md](qwen35.md) "Not done" |
| A config that omits `tie_word_embeddings` | refused rather than guessed, `src/qwen35_model.rs:208-218` |
| Sizes other than the 2B were parsed, not validated against transformers with real weights | [qwen35.md](qwen35.md) "Not done" (Other model sizes) |
| mRoPE with image positions (text positions only); one sequence per `Decode` session | [qwen35.md](qwen35.md) "Not done" |
| Shared-prefix attention for any head_dim but 256 | [qwen35.md](qwen35.md) "Not done" |
| MLX quantization other than affine, 4 bits, group 64, uniform | `src/qwen35_q4.rs:200-204` |
| Training, parameter access, the optimizer and `score_answers` on an MLX Q4 model | `src/qwen35_model.rs:682-684` |
| Forward, `prefill` and `Staged::logits` on a bf16 `load_tower` model (no LM head) | `src/qwen35_model.rs:656-663` |
| `Decode::step` after `max_new` tokens, after a weight write since prefill, or after a failed step | `src/qwen35_model.rs:2418-2424` |
| `QUANT_PREFILL_GEMM_WIRED` is `false`: no native TensorOps Int4 prefill through a host `MTLTensor` descriptor | `src/mtl_tensor.rs:150`; `README.md` Known Gaps (Host `MTLTensor` Int4 row) |

### EmbeddingGemma 2, BERT, TimesFM 3

| Limit | Evidence |
|---|---|
| Audio per call: at most 4096 clips and 2^21 frames; per forward 256 clips and 16,384 frames; a call over a bound is refused by name before allocation | `src/embedgemma2_audio.rs:134`, `:138`, `:143`, `:154`; [embedgemma2.md](embedgemma2.md) "Batches and limits" |
| Audio attention takes at most 16 past frames; `AudioConfig::validate` refuses a config outside the bound | [embedgemma2.md](embedgemma2.md) "Batches and limits" |
| x86 / AVX2 behaviour of the native resize is unverified | [embedgemma2.md](embedgemma2.md) "Not done" |
| TimesFM 3: at most 8192 patch rows per forward and 32 variates per series; f32 weights and exact f32 GEMMs, relaxed precision must be off | `src/timesfm3.rs:63`, `:67`; [timesfm3.md](timesfm3.md) "Limits" |
| TimesFM 3 has no bf16 operands, no C ABI and no Go binding; timings are not claimed | [timesfm3.md](timesfm3.md) "Not done yet" |
| BERT: sparse document encoders only (`BertForMaskedLM`, `DistilBertForMaskedLM`) | [bert.md](bert.md) |

### Full decode-graph replay

`PingPongCbReplay::try_replay` returns `CbReplayError::NotWired` outside dry runs: `MTL4CommandBuffer` has no replay-prior-encoding API (`src/cb_replay.rs:455`, header `:1-30`). Mini and layer-level `DecodeIcb` replay is real but opt-in and default off (`src/decode_icb.rs:38-40`). `Decode::step` does not use it (developer guide, section 5.6).

## 3. Retired features

| Retired | When and why | Evidence |
|---|---|---|
| Whole-step Qwen3.5 training (`Qwen35Model::train_step*`, `train_forward`, `train_backward_into`, `Qwen35Grads`, `PendingStep`, `Supervise`) | ABI 13 to 14; owner decision D5: ojas owns Metal training through its Tape | CHANGELOG.md, Unreleased, "Removed"; `src/capi.rs:83-91` |
| Host AdamW object (`AdamW`, `AdamWConfig`, `UpdateRule`, `MomentStorage`, `Qwen35Model::adamw_step*`, `grad_sq_norm`) | same change; `adamw_step` (f32 launcher) and `AdamWHyper` stay | CHANGELOG.md; `src/qwen35_adamw.rs:29`, `:131` |
| 13 training C-ABI symbols and the gradient and AdamW directions of `tessl_qwen35_copy` | same change | `src/capi.rs:83-91` |
| `tessl_torch` training methods and `chunk_gated_delta_rule` / `patch_transformers_qwen3_5` | same change | banner at the top of `python/README.md`; `python/tessl_torch/__init__.py:728-930` has only load and parameter methods |
| Bins `bench_qwen35_train`, `bench_capi`, `probe_train_memory`, `probe_soak`, `probe_storage_memory` | same change | CHANGELOG.md; `README.md:575` marks `bench_qwen35_train` retired |
| tessl's own safetensors, JSON, NPY and plain parsers (-1,595 lines) | owner decision 2026-11-01, completed 2026-11-02 (Milestone M2); ojas owns IO | [io-ownership.md](io-ownership.md); `src/lib.rs:272`, `:285` |
| Classic freeze-binds, range-batch and coarse-range ICB modes | removed because product tok/s lost to the prebuilt-table path | `src/decode_icb.rs:1-13` |
| `GEMMA_METAL_*` and `METAL_NATIVE_*` env spellings | no longer read; legacy `METAL_RUNTIME_*` still accepted | `README.md`, "Environment Variables" |

What remains in tessl for training are per-op kernels: `src/qwen35_bwd.rs`, `src/gdn_train.rs`, `src/attn_train.rs`, `src/cross_entropy.rs`. A new optimizer, schedule or checkpoint format is added to ojas, not here ([architecture.md](architecture.md#what-tessl-should-not-grow)).

## 4. Open issues

Each source is dated by its own header or, for CHANGELOG, by its section. "Not re-run" applies to all rows.

### 4.1 BUILD_REPORT.md (wave 14, 2026-11-20/21, session tessl-4b)

| Item | State in the report | Check at fba6fc0 |
|---|---|---|
| Five w14 branches merged with no tests run (owner order 2026-11-21) | post-landing verify is card ft-1c848758: `hosted_tests.sh` bins count, clippy, doc, GPU suites including `kernel_cpu_check`, `gemm_correctness`, `gemm_determinism`, `qwen35_params_torn`, `config_json_parsers` (the names `kernel_cpu_check` and `qwen35_params_torn` are the report's; `rg -uu` finds neither in `src/`, `tests/` or `scripts/`, so they may be script labels or lane-only tests) | the five w14 refs are ancestors of HEAD (`git merge-base --is-ancestor`); the suites were not run |
| Hosted manifest miss: `lib embedgemma2_tok::tests::` 6 vs 7 | fixed forward in `fba6fc0`, from source, not run | HEAD is `fba6fc0` |
| w14-bench: clippy stopped at the first failing target; no passing run of the final code; manifest bins 437 + 39 = 476 (inferred) vs 488 after merge | open | not re-run |
| Residual cards: torn (ft-237a0a15), vacuous tests (ft-e2780f81), config bounds (ft-a6645275: qwen35 layers with no `layer_types` accepted `u32::MAX`, 4.3 GB RSS), bench (ft-274d53fc) | open on the board | not re-checked |
| Uncommitted WIP backed up under `tessl-backups/w14-2026-11-21/` (outside the repo): 5 attack tests, icb cross-element test, two-pass ledger tests, cfg rescale and exponent fixes (one test red and over-strict), media helpers | not on main | not checked |
| Cards ft-2a8d8f4b (media) and ft-5f875c33 (abi) left `in_progress` with a RELEASED note | owner action needed to move them back | not checked |

### 4.2 BRANCH_REPORT.md (2026-11-19, session tessl-64, main was `012eb02`)

The report merged and deleted nothing. Its branches are mostly still unmerged. Re-checked by `git merge-base --is-ancestor <ref> HEAD` on 2026-11-22:

- Not ancestors of HEAD: `gp/w11-int` (70 commits ahead), `gp/w11-fence` (8), `gp/w12-overflow-wip` (7), `gp/w13-faultword` (1), `gp/w13-go-lto` (4, the report said 1), `gp/w6-tiles` (4), and later lanes `gp/w15-flaky`, `gp/w15-validate`, `gp/w16-torn-adv`, `gp/w16-verify-manifest`, `gp/w17-bench`, `gp/w17-bench-adv`, `gp/w17-validate`, `gp/tfm3-tile-ab`, `gp/timesfm3-bf16`. There are 40 `gp/*` branches in total.
- `gp/timesfm3` no longer exists as a ref, but `src/timesfm3.rs` is on HEAD, so the content landed another way (inferred, not traced).
- Five w14 branches were merged by ref (`gp/w14-bench`, `-cfg`, `-torn`, `-vac-gemm`, `-vac-seeds`; merge commits `d14ac92`, `ab4f7d5`, `30ac93c`, `b4ee784`, `c63c92b`). `gp/w14-media` also reads as an ancestor of HEAD, but BUILD_REPORT.md records it as 0 commits (WIP uncommitted), so that is not a merge.
- What each unmerged branch holds is not summarized here; BRANCH_REPORT.md's inventory table was written against `012eb02` and is out of date.
- The report's environment notes (machine restored 2026-11-19; Hugging Face cache and `~/.venvs` gone; Qwen3.5-2B and Qwen3.8-27B weights not on disk) mean real-checkpoint tests skip or fail on that machine until weights are re-downloaded. Not re-checked.

### 4.3 AUDIT_REPORT.md (2026-11-01, at `a00521fb`) against the code at fba6fc0

Line numbers in the report are stale. Status below is from reading the named code today; nothing was executed.

| Id and concern | Code at fba6fc0 (read) |
|---|---|
| A-1 `TriFlag::get` could overwrite a concurrent `set` | looks fixed: the unset case uses `compare_exchange` (`src/ab_flags.rs:112-125`) |
| E-4 `VisionConfig::patch_dim` u32 overflow | looks fixed: `patch_dim` goes through `checked_patch_dim` (`src/embedgemma2_vision.rs:537-539`) |
| B-1 five `from_config_file` paths read config with no cap | no `read_to_string` remains in the five files named by the report (`rg -uu`, empty); the capped reader path was not read |
| A-2 `drain_cold_recycles` dropped charges on a poisoned pool lock | changed: the drain now holds items in `Requeue` guards (`src/runtime.rs:1965-1971`); not read further |
| D-1 SmallPar accepted an unpadded scratch | the gate comments now name the padded length (`src/gemm.rs:474`, `:559`); not verified further |
| E-2 `SetMaxTokens` race | reworked: a turn channel and a helper that updates the cache (`go/embedgemma2/model.go:281-303`); not verified further |
| E-3 `tessl-embed` EOS probe | the probe code changed (`go/cmd/tessl-embed/server.go:115`); not verified |
| E-1 video frame grid allocated before `max_frames` | `sample_frame_indices` validates `max_frames` (`src/embedgemma2_pre.rs:1384`); whether the allocation order changed was not read (open until checked) |
| ABI 14 prose still describing the retired step | partly fixed in `README.md` (ABI 14 notes at the top and in the feature list); still present in `docs/README.md`, `docs/qwen35.md`, `python/README.md` (see section 5) |
| Wait timeout "poisons forever" (ft-9a93d5ae) | the first two consecutive timeouts do not poison; the third does (`src/runtime.rs:1826-1833`) |
| DevMap Metal parse degraded, 46 files with ERROR/MISSING regions | not re-measured; treat empty Metal caller lists as unknown |

### 4.4 tasks/ (briefs committed-then-ignored 2026-10-25; each "Source" line says 2026-10-20)

These are 17 task briefs. They predate wave 14 and name removed things (`GEMMA_METAL_ICB_*`, `patch_transformers_qwen3_5`, `src/safetensors.rs`), so read each as a statement of intent, not current status.

| Brief | Intent (priority) | Note |
|---|---|---|
| gp-nax-int4-tensorops-and-exact-f32-epilogue | shader-side `int4b_format` prefill and an ExactF32 cooperative epilogue (p1) | `QUANT_PREFILL_GEMM_WIRED` is still `false` (`src/mtl_tensor.rs:150`) |
| gp-qwen35-q4-weights-and-qwen38-27b | MLX 4-bit in the Qwen3.5 tower; Qwen3.8-27B decode at MLX parity (p1) | `load_mlx_q4` exists (`src/qwen35_model.rs:692`); parity vs mlx-lm for 27B not verified |
| gp-qwen35-vs-current-torch-and-mlx-head-to-head | rerun against torch 2.14.1/2.15 and mlx-lm 0.32.x (p1) | not re-run; its `python/tests/test_gdn.py` does not exist |
| gp-qwen35-prefill-1k-2k-gap-vs-mlx | MLX led Qwen3.5 prefill at 1k and 2k tokens (p2) | open at that date |
| gp-bf16-gemm-losing-shapes-vs-mps | 4 of 8 bf16 GEMM shapes below parity vs torch 2.13 MPS (p2) | numbers are 2026-09-01; do not quote without rerun |
| gp-decode-icb-31b-and-range-batch-default | finish 31B decode-graph ICB, default-on replay (p2) | `decode_icb_enabled()` default still off |
| gp-qwen35-decode-fusion-concurrency-and-gdn-state | fusion landed as opt-in; barriers, GPU-resident loop, GDN state read/write once remain (p2) | `Decode::step` still waits per token |
| gp-metal4-fault-escalation-and-reader-bounds | third timeout to DeviceLost; reader payload caps (p2) | the DeviceLost escalation is in `src/runtime.rs:1832-1833`; reader caps not checked |
| gp-tokenizer-encode-batch-multithread | `encode_batch` was single-threaded: 0.32x of HF all-core on 200 documents (p2) | not re-run |
| gp-eg2-quiet-evidence-reruns-and-native-hf-tokenizer-lane | quiet-machine reruns of fusion, vision tower, tokenizer lane (p2) | not re-run |
| gp-resampler-licence-provenance | document provenance of `src/media/resample.rs` after LGPL removal (p1, high) | see [media_resample_cleanroom.md](media_resample_cleanroom.md); not re-read |
| gp-history-rewrite-sha-map | publish old-to-new SHA map for the 2026-10-18 rewrite (p2) | open per brief |
| gp-push-mains-before-article, gp-wave10-hosted-manifest-and-kernel-bench-coverage, gp-readme-generated-kernel-counts, gp-bench-evidence-holes, gp-qwen38-mixed-bits-and-sparse-ffn-gemv | release, coverage, docs and research chores | not re-checked |

### 4.5 CHANGELOG.md (Unreleased, last committed 2026-11-19)

- `bench_timesfm3` is added but not yet timed (`CHANGELOG.md:19`).
- The Rust 1.100 floor was verified on 2026-11-12 with `cargo +1.100.0 check`; results are in the changelog entry.
- Other sizes of Qwen3.5 "are parsed, not yet run against transformers" (`CHANGELOG.md:1581`).

### 4.6 `#[ignore]` tests

Most attributes are bare (no reason string). The reason below is taken from the doc comment or the test body I read. None were run.

| File | Why it is ignored |
|---|---|
| `tests/qwen35_model.rs:105`, `:149`, `:371`, `:566`, `:626` | real 2B checkpoint and transformers references (the 2B checkpoint and fixtures are needed; reason not stated in the attribute); `:371` is `logit_deviation_report`, which also needs the MLX Q4 checkpoint and prints "NOT RUN" when `QWEN35_REF_TEXT_DIR` is unset (`tests/qwen35_model.rs:365-377`) |
| `tests/qwen35_mlx_q4.rs:742`, `:864`, `:921` | real MLX Q4 checkpoint and mlx-lm references (reason not stated in the attribute) |
| `tests/safetensors.rs:593` | needs `QWEN35_2B_SAFETENSORS`; fails rather than passes when it is unset |
| `tests/embedgemma2_model.rs:131`, `tests/embedgemma2_e2e_model.rs:109`, `:249`, `:271`, `tests/embedgemma2_mm_model.rs:169`, `:258`, `tests/embedgemma2_multimodal_model.rs:99`, `tests/embedgemma2_video_model.rs` (six), `tests/media_e2e_model.rs:100`, `tests/media_e2e_video.rs:104` | compare against sentence-transformers references on the real snapshot (`EMBEDGEMMA2_SNAPSHOT`; several also read `EMBEDGEMMA2_REF_DIR` or `EMBEDGEMMA2_MM_REF_DIR`) |
| `tests/embedgemma2_tok_real.rs` (four) | real tokenizer files |
| `tests/media_yuv_exact.rs:211`, `tests/media_yuv_formats.rs` (four) | exhaustive comparison against a live `ffmpeg` (about a minute per matrix and range) |
| `tests/embedgemma2_pre.rs:238`, `:371`, `:514` | need `EMBEDGEMMA2_PRE_BIG_DIR` from the generator |
| `tests/media_jpeg.rs:693` | needs `JPEG_LIVE_FUZZ_DIR` |
| `src/gemm.rs:3604` | deep fuzz soak, run with `--ignored gemm_fuzz_deep` |
| `src/gemm.rs:4776` | prints the probe table; run by hand to refresh a committed result |
| `tests/cross_entropy.rs:1234`, `tests/qwen35_gdn_simd_ab.rs:149` | timing runs, not correctness gates |
| `tests/embedgemma2_memory.rs:206` | prints hashes for comparing two builds |
| `tests/hosted_completeness.rs:136`, `tests/hosted_selection.rs:1648` | run by `hosted_tests.sh` entries (`--outside`, sandbox only) |

The ignored model-parity tests are the main correctness evidence for real checkpoints, so a green `cargo test` without `--ignored` does not cover them.

## 5. Doc drift noticed

The developer guide's section 9 has the full table. In short, at fba6fc0:

- `docs/README.md` still lists training backward and fused AdamW under Qwen3.5 and "full model `train_step` integration" under PyTorch Interop. The code retired both at ABI 14.
- `docs/qwen35.md` "Not done" still says `train_step` and `train_step_ragged_into` run the whole step; no `fn train_step` exists in `src/` or `tests/`.
- `python/README.md` lists `test_gdn.py`; `python/tests/` has no such file. The `Qwen35` class docstring in `python/tessl_torch/__init__.py:728-729` still says it is "trained by tessl's ``train_step``".
- `AGENTS.md` names `HostLease`/`HostLeaseMut` as tessl leases (they are `ojas_core` traits) and treats decode ICB as part of the Qwen3.5 engine (not referenced from the Qwen3.5 modules).
- `AUDIT_REPORT.md` (2026-11-01) and `BRANCH_REPORT.md` (2026-11-19) describe states that have since moved; see 4.2 and 4.3.

## 6. Not verified here

- Every "open" item is open per its source, not per a run. No test, build or GPU job was run.
- The `.github/workflows/ci.yml` gates were read at the two `HAS_METAL_RUNNER` lines only.
- Whether each AUDIT_REPORT item is fully fixed (rows marked "not verified further").
- That `gp/timesfm3` content landed on main under another commit.
- Which kernels accept f16 operands.
