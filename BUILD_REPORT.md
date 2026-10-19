# BUILD_REPORT — GitPulse backlog, waves 13–19 (2026-10-17/19)

## Wave 19: a bench lane for every kernel (`main` = `1c4d03e`)

Owner decision: bench lanes for **all** kernel entry points. `kernel_coverage --check` now **exits 0**. Of 303 entry points, 301 are dispatched by bench lanes. The other 2 are allowlisted as test-only by owner decision: `mtl_tensor_fill_i8` / `mtl_tensor_read_i8` are only dispatched by a `#[cfg(test)]` test behind `quant-prep`. `--audit-allowlist` accepts both rows.

**Lane 1** added 25 kernels across four binaries:
- `bench_nn_kernels`: utils.
- `bench_gemm_variants`: TN split-K.
- `bench_qwen35_train`: GDN train, CE bf16, AdamW / window copies / q8 moments / bank views.
- `bench_qwen35_layers`: Qwen3.5 op lanes in the default run.

**Lane 2** added 22 kernels:
- `bench_embedgemma2{,_audio,_vision} --kernels`.
- `bench_encoder_attn`: the tiled variants.

Every lane checks its result before timing. Most compare against a host reference or a derived bound. Three EmbedGemma2 audio kernels get only finite and range checks, which do catch NaN.

The adversarial pass on both lanes found three problems, all fixed before landing:
- AdamW / moment write-backs that were never read back. They are now read back and bit-checked.
- Op lanes that checked only finite, non-zero output. They now have host references.
- `bench_encoder_attn` skipped its 10% spread guard whenever `TESSL_KERNEL_TRACE` was set. It now skips only on an explicit `--coverage-probe`, which `kernel_coverage` passes, and says so when it does.

Accepted as unobservable: `eg2t_scale_rms_norm_f32` can't detect an ignored scale, because `rms(x·s)` is scale-invariant. Accepted by convention: `bench_gemm_variants` lanes are timing-only, and their correctness is covered in `tests/`.

**Changed:** bench binaries, `src/bin/common`, `tests/kernel_coverage.rs`, the allowlist, and `hosted_tests.txt` `bins` 261 → 262. No library code, no kernels. No timing claims.

**Landing.** It was cherry-picked onto `09efe84`, the eg2 merge that tessl-a3 landed first (first green lands first). Gates were green after the last pick: fmt, build, clippy ×2, doc, hosted 1218 + completeness, dialect, IR, emulator ×4, comparator, torch host 110, Go unit, bench and lib tests. ojas acked it; it builds none of these files.

## History scrub (2026-10-18 ~20:41, by the eg2 session) and wave 18

**The scrub.** The eg2 session rewrote `main` and `eg2/integrate` with `git filter-branch` to remove an LGPL-derived `src/media/resample.rs` from history.
- Every sha changed, and the tip tree is unchanged: `41d8866` (ojas-verified) → `a083088`, `4e99ca6` → `a1fccca`. **The shas cited in the sections below were pre-scrub until eg2/integrate's `d8f9cc3` renamed them; since that merge they name the post-scrub commits.**
- `refs/original/refs/scrub/*` and earlier sessions' `refs/backup/*` still resolve them, and ojas cites pre-scrub `42f1d52` as provenance. Keep those refs until the scrubbing session publishes an old→new map.
- At your direction I deleted this run's own 54 `refs/backup/2026-10-17-w14/*` refs, because they kept the scrubbed blobs reachable.

**How wave 18 landed.** Its lanes had been cut from the pre-scrub `4e99ca6`. Each lane's own commits were **cherry-picked** onto `a1fccca`, never merged, so the old history stays unreachable from `main`. A second tessl session (tessl-a3) coordinated with me: it lands the eg2 session's last commits and removes my lane worktrees.

**`main` = `49d905e`**, ojas-acked before landing. The gates run after the last pick:
- fmt, build, clippy ×2, doc.
- Hosted: 106 invocations, 1210 tests.
- Dialect lint, kernel IR (39 kernels).
- Emulator: host contract plus the exact, model and fast-math cases.
- Comparator lanes, torch host tests, Go unit.
- 689 device tests in 60 binaries, plus `--lib` 272.

| Card | Lane | Result |
|---|---|---|
| ft-6d88defb kernel coverage | A | Both broken lanes were bench bugs. `softcap_sample` asked n=512 of a 256-wide single-threadgroup kernel; `bench_qwen35_train` used the 2B hidden size on a tiny model. `kernel_coverage --check` now completes: 303 entry points, **49 with no lane** (list on the card) — your "lanes for all" work list. |
| ft-fd8bde58 int4 TensorOps GEMM | B | Opt-in `nn::gemm_q4_mlx_tensorops` (G 32/64/128) with in-kernel xsum. Parity is within ~0.4% of a derived bound over 120 shapes. Refusals happen before any GPU work. A guard-region test proves partial tiles never write past `y`: the adversarial pass found that gap. **No timing:** the benchmark is still open. |
| ft-c8e7726b, ft-a13c82af | C | EmbedGemma2 config readers now use the shared `config_json`. A new `encode_batch` is the single owner of `upload_segments` and `check_batch_shape`. Behaviour is byte-identical on 49 configs. `TESSL_ATTN_TILED` is documented as the A/B baseline. |
| ft-afb1c42d bounds | D | Exact byte tests, using a test-only allocation counter, **found two under-counts** in the training OOM pre-flight: the GDN forward transients and the CE staging buffers. Fixed; the 2B bound rises 25–405 MB (≤3.8%). `qwen35_train_bytes_pin` re-pinned, old and new values recorded. A device-wide-peak flake already on `main` is fixed. |


`main` moved `4d3543f` → `02b1e93` → `bf0b7b9` → `527de20` → `8c51319` → `cef623e` → `e3dff5a` (this report) → **`a083088`**. Every move was a fast-forward, local only (nothing pushed). Each landing came after the gates below were green on the exact tree, and after the live ojas session acked the sha (ojas builds tessl by path). ojas independently verified `cef623e`: Metal 124/124 and ojas-qwen35 14/14, with its clip path switched to `parts_words`.

## Waves 16C/D and 17, landed at `a083088`

**Gates**, run after the last merge:
- fmt, release build, clippy ×2, doc.
- Hosted (selection and completeness).
- Dialect lint, kernel IR.
- Emulator host contract plus the exact, model and fast-math cases.
- Comparator 15, torch host 110, Go unit.
- 979 device tests across 78 binaries, serial, release, plus `--lib`.
- Python `test_qwen35` 21.
- Two hosted-manifest fixes (new tests' selection).

| Card | Wave | Result |
|---|---|---|
| ft-afb1c42d near-copies remainder | 16 C | gemm `_into` casts go through `cast_between`. They are bit-identical on all 65,536 bf16/f16 patterns and 393,216 f32 patterns recorded before the change. One `single_seq_gdn_dims`. Two bound tests: an exact structural test that the scratch bounds equal the bytes the allocators make, and a peak-vs-bound test. The adversarial pass showed the peak-vs-bound test alone missed a dropped `3·t·i` mirror, which led to the structural test. Layer fwd/bwd and CE bounds are not covered exactly (their GEMM temporaries live in no struct). |
| ft-c8e7726b config.json dedupe | 16 D | One private `config_json` reader for BERT and Qwen3.5. Drift fixed: BERT silently read a non-literal `sinusoidal_pos_embds` as false, so learned positions ran on a sinusoidal model, and it treated `position_embedding_type: null` as absolute. Both now refuse; these are the only error-text changes. No cached real config relied on the old leniency. EmbedGemma2 parsing is out of scope. |
| ft-9e79d8be ragged training, API | 17 E | Owner decision: new entry points. `tessl_qwen35_train_step_ragged` (C, ABI 12→13 additive; per-sequence losses as f64 to match the Rust API) and `tessl_torch` `Qwen35.train_step_ragged`. Every boundary attack refuses without writing outputs. Nothing pins ABI 12. |
| ft-1a0f8c18 AC5 RMSNorm convention | 17 F | Owner decisions: `precise::rsqrt` for the RMSNorm and gated-norm forwards, and for the shared qk-norm row kernel in both directions, including EmbedGemma2's use. See the deltas below. |

**ulp deltas from 17 F:**

| Output | Change |
|---|---|
| RMSNorm forward | ≤ 2 ulp |
| Gated-norm forward | ≤ 4 ulp |
| qk forward | ≤ 42 ulp at eps 1e-12 |
| qk backward dproj | ≤ 210 ulp at eps 1e-12 |
| EmbedGemma2 ExactF32 embeddings | ~1e-3 relative |
| Everything in bf16 | 0 |

Accuracy against torch and f64 is unchanged within noise, and slightly better in several cases. One golden was re-pinned with old and new values recorded: the single-sequence training pin.

An apparent "pre-existing flake" in that pin was a mis-measurement. The lane edited kernels during an in-flight cargo run. The pin was proven to move first at the lane's own commit, and 21 runs on `main` reproduced it.

Follow-up for the owner: EmbedGemma2's own norms (eg2 text, vision and audio) are still fast rsqrt beside the now-precise qk-norm. That's inference-only, so it isn't a break.

ft-c59ff950 `parts_words` landed in `cef623e`, and ojas switched to it.

**ojas verified `a083088`** (run 9, ojas main 4123dfb): qwen35 CPU 42/42, Tape provider CPU 14/14 and Metal 5/5, clippy clean. Nothing moved numerically. `gpu_parity` was 9/10: the miss, `gpu_real_2b_single_step_matches_the_transformers_reference`, is environmental, because the disk cleanup deleted `tessl/target/qwen35_train_ref`. Regenerating it is a 17 GB CPU job that must run under the `mac_heavy` lock; the command has been handed to the owner.

**Branches:** only `main` and the live eg2 session's fully merged `eg2/integrate` remain. Every deleted ref is under `refs/backup/2026-10-17-w14/`.

## How this run worked

- **GitPulse:** cards were read and written through the installed `gitpulse-mcp` over stdio (the MCP was not attached). Every card here went `in_progress` → `review` with evidence and `expected_revision`. None was requested `done`, which Manvi gates.
- **Roles:** Fable made the sequencing and judgement calls. Opus (this session) wrote every brief, reviewed every diff, and ran the final checks. Sonnet implemented each lane in its own worktree. A *separate* Sonnet attacked each lane, mutation-testing it where possible, and every break went back to the implementer.
- **GitPulse insights / collision risk** were load-shed for most of the run (load 80–230, "git spawn rate budget is low"). Collisions were checked from committed branch diffs instead, which cannot see uncommitted work in other worktrees. The live eg2 lane's files were avoided until its work was integrated.
- **DevMap:** I rebuilt the index (gen 1394). `impact`/`affected` walks stopped at depth 3 (`walk_incomplete`; 14 files with parse-error regions), so blast radii are lower bounds.
- **`dev verify`:** gates nothing here (no DevCouncil config). The gate run was `scripts/ci_local.sh`, minus its GPU-lock-only steps.
- **Branch base:** agent worktrees branch from `origin/main` (`60a6240`, 199 commits stale). Wave 13's lanes were built on it by mistake and rebased onto local `main`. After that, every lane started with `git merge --ff-only <main>`.

## Gates

Final run, on `cef623e`:
- fmt, release build (all targets), clippy `-D warnings` (default and `quant-prep`), doc `-D warnings`.
- Hosted device-free selection (103 invocations, 1196 tests) and completeness.
- Dialect lint, kernel IR check (38 kernels).
- Emulator host contract and the exact, model and fast-math kernel cases.
- Comparator lanes (15), torch binding host tests (108), Go unit (`make -C go unit`).
- Every targeted device test binary, serial, and `--lib` (265).

**NOT run:**
- The GPU-lock-only steps: the full serialized `cargo test --release`, the Python GPU tests and the Go device tests. The `mac_heavy` lock was held by ojas or Lappi jobs nearly the whole run.
- Examples, the `mtl_tensor` quant-prep run, and the release-ready gate.
- `kernel_coverage --check`: exits 77 on `main` already (see ft-6d88defb).

Earlier landings had the same gate set, each on its own tree.

## Tasks moved to review (evidence in each card's log)

| Card | Wave | Result |
|---|---|---|
| ft-14fa3c44 wave10 manifest/merge/kcov | 13 | Triage only. AC1–2 were already on `main` (5c54400, 2323add). AC3 is stale (eg2 was merged, not rebased). AC4 is ft-6d88defb. |
| ft-de6fa0ab Metal 4 faults + reader bounds | 13 | AC2–4 and the AC5 site list were already on `main`. `host_access_ban`'s "any receiver named `st`" exemption is now an exact counted site list: a first heuristic let 22 of 37 hostile cases through, so it was replaced. Added a commit-feedback error-path unit test. AC1 is closed by owner decision (timeouts stay retryable). |
| ft-c0bb2f03 decode ICB | 13 | AC4: `IcbCopySmoke::encode_copy` refuses while an execute is in flight. The token is read atomically under the batch lock; the adversarial pass found a TOCTOU in the first version. AC1–3 belong to ojas → filed ojas ft-4d9d3b55. |
| ft-20802330 NAX int4 + ExactF32 | 13 | The int4b constructor exists (compile probe `tools/probes/int4b/`, which fails on any deviation). The rest was split into ft-fd8bde58 and ft-36deadf7. |
| ft-a13c82af tiled FA-2 collapse | 14 | One `attn_fa2.h` template. Outputs are bit-identical, using goldens recorded on the old kernels. Slots are byte-identical, and ojas checked gemma-metal. Owner kept `TESSL_ATTN_TILED` as the A/B baseline. |
| ft-4b1313de runtime lease spike | 14 | Invariant table plus 9 pinning tests, mutation-verified (M1–M7 caught; M5b unobservable). Decision: no shared read guard now. AC1 measurement open (GPU). |
| ft-1a0f8c18 kernel templating | 15 | `gemv_q4_mlx` (10), `gemm_q4_mlx` (4) and utils copy/cast are templated. 132-row goldens and the IR-checked slots are identical. `qdot16` reloads its weight word per m (reported, not changed). Owner wants a real tuned `_wide` kernel (GPU-gated). The RMSNorm convention is not started: it changes numerics. |
| ft-afb1c42d host near-copies | 15 | Dispatch validator; poison-safe release queue (3 of 4 tests failed before the fix); loader/tensor/npy generics. Error texts are proven unchanged by running the new tests on the old code. Remainder is in progress (wave 16 C). |
| ft-fd8bde58 int4 TensorOps GEMM | 15 | Measured on the GPU: low nibble first, `int4b` is two's complement. A per-group MLX Q4 TensorOps GEMM matches `gemm_q4_mlx` and f64 within derived bounds. No production entry point and no benchmark yet. |
| ft-36deadf7 ExactF32 coop epilogue | 15 | Two new kernels. Bit-identical to the separate passes once FMA contraction was turned off (`STRICT`; it was 1 ulp off). 432+ cases. Census 298→300 with bench lanes. NN only. |
| ft-c59ff950 `SqNormPlan::parts_words` | 16 | ojas asked for it. One owner of the 2·pieces rule; all four mutants caught. |
| ft-9e79d8be ragged Qwen3.5 training | 16 | `train_step_ragged_into` (Rust core). Per-sequence losses are bit-identical to each sequence alone. Sequences are isolated. The memory bound now counts the ragged step: the adversarial pass found it undercounting. The single-sequence `train_step_bytes` is measured equal to `8c51319`'s. C API and Python wiring are deferred (public contract). |

Also merged without a card, at your request:
- **`ea630de`:** the prepared integration of `gp/clip-sq-norm`, `gp/w11-2bmem` and `gp/w12-kcov`. It brings `tessl::sq_norm` with overflow-safe per-piece scaling.
- **Your uncommitted work, committed:** the GDN speed tool and its result (recorded as `valid: false` by its own spread gate), the `bench/tok_hf_native` crate (a **new third-party dependency**: `tokenizers =0.23.2`, isolated in its own workspace), and `tasks/` (excluded from the crate).
- **The eg2 lane:** media decoding, plot_results, capi_embedgemma2 and a Go binding under `go/`. The Go binding depends on **gusset**, pinned to a commit, outside tessl's crate graph. eg2 also adds `THIRD_PARTY_NOTICES.md` for its libjpeg-turbo-derived JPEG decoder.

## Benchmarks

None. No hot path changed in a way that needed timing:
- The template refactors are bit-identical, and their IR was checked.
- ExactF32 adds a fused path; nothing that existed was slowed.
- No speedup is claimed anywhere.

Every benchmark-gated card stays open because the `mac_heavy` lock was busy.

## New tasks filed

- ojas **ft-4d9d3b55**: gemma-metal 31B decode-graph ICB migration.
- tessl **ft-36deadf7**: ExactF32 epilogue. Done in wave 15.
- tessl **ft-c59ff950**: `parts_words`. Done in wave 16.

Updated rather than duplicated:
- **ft-fd8bde58**: the int4 blocker turned out not to exist.
- **ft-6d88defb**: the exit-77 diagnosis. Two existing bench lanes fail inside the tool: `softcap_sample` n=512 exceeds 256 lanes, and `bench_qwen35_train` gets a wrong out shape.

## Fable's decisions

1. Wave 13: three lanes. Scope the `st` exemption; don't rename public API. Close timeout escalation as an owner decision. Run the int4 probe compile-only.
2. Wave 14: FA-2 collapse and the lease spike. Defer ragged training while the eg2 lane held `qwen35_train.rs`. Don't adopt the dead session's cards.
3. Wave 15: four low-risk lanes. Kernel templating on the FA-2 method, near-copies subset, int4 parity, ExactF32. No benchmark work.
4. Wave 16: `parts_words` and ragged training (Rust core only), then near-copies remainder and config parsing (BERT and Qwen3.5 only). Defer the RMSNorm numerics change. The nn.rs split lands last.

## Owner decisions (2026-10-18)

| Item | Decision | Remaining work |
|---|---|---|
| Tiled attention | Keep as the A/B baseline | Docs follow-up |
| Q4 `Wide` | Write a real tuned wide kernel | GPU benchmark first; announce to ojas |
| Kernel coverage | Bench lanes for every kernel | GPU; fix the two failing lanes first |
| GPU wait timeouts | Stay retryable | None |

## Still pending

- **Wave 16 C and D:** in flight.
- **For you:** the C API and Python signature for ragged batches.
- **Not started, for a later wave:** the RMSNorm rsqrt convention, which changes training numerics and needs an ojas announcement. ft-376ad5d4 public-surface narrowing (breaking).
- **Disk:** finished agent worktrees under `.claude/worktrees/` keep their `target/` (0.7–1.3 GB each). The gate holds `rm -rf` there for you.

## Branches and backups

- **Deleted:** every merged or contained branch of this run, the stale `gp/*` lanes, the three stale detached merge worktrees, and 12 stale eg2 agent branches.
- **Backups:** every deleted ref is under `refs/backup/2026-10-17-w14/`.
- **Kept:** `main`, the live `eg2/integrate`, the two running lanes, and five finished worktrees that are harness-locked or hold agent scratch.

## Incidents

- A shared `target/` was wiped by someone else mid-run.
- The disk hit 100% (ENOSPC) overnight from per-lane build dirs plus ojas's verify cache. No shell could run. I stopped my agents, and space was freed by the owner.
- One tessl fast-forward (`527de20`) landed during an ojas re-run. ojas had permitted it, but it mixed two sources of change. Since then, a fast-forward waits for ojas's "rerun done".

---

# Earlier run: waves 6–7 (2026-10-14/15)

Base: `main` @ `620939d` (which already held waves 1–5, see the earlier report below). All of this run's work is on **`gp/wave7`** (worktree `../tessl-wave6`): `gp/wave6` (wave 6 plus integration fixes) with the four wave-7 lanes merged on top. That is 56 commits and 96 files (+13.3k / −4.3k, report and CHANGELOG included) ahead of `main`. **Nothing is merged into `main` or pushed**, because that is your call.

## How this run worked

- **GitPulse was read *and* written** through the installed `gitpulse-mcp` over stdio (the MCP server wasn't attached to the session). Every card moved below went `in_progress` → `review` with a summary and `expected_revision`. No card was requested `done`. Done needs the work on `main`, and Manvi gates it.
- **Fable made the sequencing and judgement calls** (wave picks, the symlink rule, stopping after wave 7). Opus wrote every brief and reviewed every diff. Sonnet implemented each lane in its own worktree, and a *separate* Sonnet attacked it. Every break went back to the implementer before approval.
- **DevMap:** I rebuilt the index (gen 1359, 293 files) before wave 6, and again in the integration worktree before wave 7. `affected` walks stopped with `walk_incomplete` at depth 2, because 12 files have parse-error regions. I treated every blast radius as a lower bound, not as "safe". GitPulse's in-process DevMap still calls the index "payload obsolete" while the CLI calls it fresh (old report, item 5).
- **`dev verify`:** DevCouncil's `verification_mode` is `off` here and there is no `.devcouncil/config.yaml`, so `dev verify` gates nothing. `scripts/ci_local.sh` is the gate I ran.
- **Collisions:** `gitpulse_insights` / `gitpulse_collision_risk` before each wave and after the run found 0 overlapping files and no other live session or dirty worktree in tessl. No task was deferred for a collision.
- **Machine:** all heavy jobs ran under the `mac_heavy` lock. The disk reached 99% (11 GiB free) mid-run, and the lock refuses jobs below 15 GB free. I removed my merged lane worktrees and a scratch copy of ojas, and later the cargo cache was cleared elsewhere. Free space is now 95 GiB.

## Gate results on `gp/wave7` @ `b104ad4` (M5 Pro, under the lock)

| Gate | Result |
|---|---|
| `scripts/ci_local.sh` (full) | **`ci:local OK`** |
| Full serialized suite (`--no-fail-fast`) + hosted sandbox runs + mtl_tensor | 1877 test executions passed, **0 failed**, 18 ignored |
| Hosted device-free selection (device hidden by `sandbox-exec`) | OK: 59 invocations, 610 tests |
| Hosted manifest completeness | OK |
| fmt, clippy `-D warnings` (default + `quant-prep`), `cargo doc -D warnings` | clean |
| Dialect lint, kernel IR check (35 kernels), host contract, parity harness `--pure` (also with `TESSL_HOSTED=1`) | pass (harness: PASS WITH SKIPS; the skips are the `--pure` GPU CLI sections) |
| torch binding host + GPU tests, examples, release-ready, `cargo package --locked`, docs.rs check | pass |

Earlier runs of the same gate on the integration branches, and what each run caught:
- `gp/wave6`: 1171 passed and 1 failed. The failure was a `host_access_ban` false positive on a `#[cfg(test)]`-declared module. I fixed the scanner in `0949270`, and its new test fails when the module is ungated.
- `gp/wave7` (three lanes merged): 1227 passed, 0 failed.
- `gp/wave7` run 1 stopped at `cargo doc`: a public doc linked a private fn (fixed in `b7a2d8d`).
- `gp/wave7` run 2 stopped at the new hosted manifest's exact-count guard and completeness check. Merging the lanes had added 23 device-free tests that the manifest didn't list (fixed in `b104ad4`). The guards did their job.

## Tasks moved to review (10 cards, evidence in each card log)

| Card | Wave / branch | Result | Adversarial pass found (all fixed before approval) |
|---|---|---|---|
| ft-dc1fa0a3 per-entry host reads/writes | 6 / `gp/w6-hostmap` | AC1, AC2, AC4 met. **AC3 (the ojas-qwen35 `state.rs` conversion and the 2B peak measurement) is open** and waits on the merge, because ojas path-depends on tessl `main` | host bf16 writes kept NaN payloads and rounded subnormals where the GPU cast canonicalizes them. Now bit-identical to the GPU path; reads are verified over all 65,536 bf16 patterns |
| ft-04c475ac parser mutation tests | 6 / `gp/w6-parsers` | all ACs met (no new dependency: seeded splitmix64 harness) | `SafeTensors::open`/`read_npy` **hung forever on a FIFO**, now fixed (`O_NONBLOCK` + fstat). Consistency errors had counted as refusals; and CI could not detect a zero-test match |
| ft-01dd5d8a bench harness | 6 / `gp/w6-benchharness` | AC1–5 met; AC6 (moving the `tests/common` clones) deferred | duplicate env keys, `--opt=value` parsed as a lane, **no lane timeout**, negative METRIC values, trailing tokens, no per-round re-hash, lost raw-output archive |
| ft-5dd538e9 fast-math lint | 6 / `gp/w6-closers` | AC1–4 met; AC5 recorded (the four `nn_wiring` tests pass before and after) | the lint missed paren, cast, deref and multi-line self-compares and `isunordered`; the IR script missed bfloat/half/sNaN constants and passed when no IR was written |
| ft-0693aaa7 GemmOperands contract | 6 (confirm) | all ACs met on `main`; the slot-12 test gained a guard region (it catches a planted write one element past the output) | — |
| ft-2663702f prefill working set | 6 (confirm + docs) | all ACs met; the AC3 decision is recorded in `docs/qwen35.md` | — |
| ft-dde0a66a hosted CI (code half) | 7 / `gp/w7-hostedci` | AC1–3, 6, 7, 9 met; AC4/5 kept in Python with a written reason; AC8 partly met (`encoder_attn_tiled` documented) | **150 device-free tests were not selected**; added a completeness check and exact per-entry counts |
| ft-b7376d2e sharded checkpoints | 7 / `gp/w7-multifile` | AC1–2 met; AC3 is filed in ojas as ft-02256073; AC4 deferred (no 4B/9B weights) | no break. ojas `cargo check --workspace` passes against the branch |
| ft-9ac2a619 ICB correctness | 7 / `gp/w7-icb` | all 8 ACs met | **replay raced on sync + hazard captures** (40 of 40 replays wrong). Fixed at capture; the 16-cell matrix fails on the pre-fix code. ojas gemma-metal captures with always-on barriers, so it was not exposed (checked by reading the code) |
| ft-c9057a0d kernel_coverage | 7 / `gp/w7-kcov` | AC1–3 met; AC4 runs end to end and reports 150 of 266 kernels with no bench (filed as ft-6d88defb) | the regex scan invented and missed kernels; replaced by `xcrun metal -E`. "Could not run" and "found gaps" shared an exit code; the dump wasn't atomic and hung on a FIFO |

## Deferred or blocked

| Card | Why |
|---|---|
| ft-dc1fa0a3 AC3, ojas ft-02256073 | Need `gp/wave7` on tessl `main` (ojas path-depends on it; building against a worktree would be a pin bump) |
| ft-2238696a GPU suite + `#[ignore]` tests on an idle GPU | Wants a pushed `main`, plus hours of idle GPU |
| ft-380e34f4 fault-injection set | GPU-heavy (a metallib rebuild plus a GPU run per fault); queued for wave 8 |
| ft-51395822 reader bounds (new, P3) | Queued for wave 8 |
| ft-01dd5d8a AC6 | Touches about 60 test files; after the merge |
| ft-afb1c42d, ft-c8e7726b, ft-08a87448 | Refactors sequenced to land last |
| Benchmark cards (d0a49f5d, 62af7c2e, 805b6ea4, f3e14014, 285001b5, c1e556e2, ea33f477, 8f0c2ee3, ba1413a0, fd8bde58) | Need an idle GPU for timing |
| ft-12c31d56, ft-b7376d2e AC4 | Need 4B/9B weights (only 2B is local) |
| ft-b8561dbb, ft-2ffbdcd5, ft-376ad5d4, ft-a5491fbb, ft-1a0f8c18 AC2, ft-6d88defb | Owner decisions (below) |

## Benchmarks

No hot path changed, and no timing was taken. These are the only runtime-path changes:
- the ICB capture carries a hazard flag, which applies only while capturing;
- replay emits the captured markers, which means *fewer* barriers on tapes with no conflicting edges;
- host-only per-entry copies and the loader.

The bench-harness lane ran GPU smoke runs only. Statistic changes, documented in `docs/benchmarking.md`:
- paired metrics are the median of per-round ratios (`paired_bins` used a ratio of mins);
- `--rounds` must be at least 2;
- evidence drops `peak_gflops`/`geomean`.

## New tasks filed
- **ft-51395822** (tessl, P3): checkpoint readers have no payload cap (a sparse 512 GiB file fills memory until memguard kills the process), npy padding is unvalidated, the shape-overflow check depends on dimension order, and a file rewritten after open is read with stale offsets.
- **ft-6d88defb** (tessl, P3): 150 of 266 kernel entry points are dispatched by no benchmark, so `kernel_coverage.py --check` exits 1. It needs an owner decision on adding lanes or an allowlist.
- **ft-02256073** (ojas, P2): ojas-qwen35 should load through `CheckpointView` and retire the "one file" `Unsupported` branch at `names.rs:292-296`.
- Folded into card logs, not filed:
  - build.rs's Metal flags are now derived in two places (`check_kernel_ir.sh` mirrors them under a drift test; `kernel_coverage.py` parses `build.rs`);
  - `host_access_ban` still exempts calls by the receiver name `st`;
  - an uppercase `.JSON` index path is treated as a safetensors file.

## Fable's decisions (this run)
1. **Wave 6:** dc1fa0a3 (tessl half, additive API), 04c475ac (no-dependency mutation harness, so no ask is needed), 01dd5d8a AC1–5, 5dd538e9 plus confirming 0693aaa7 and 2663702f from a GPU run instead of reimplementing them.
2. **Wave 7:** dde0a66a, b7376d2e AC1–3, 9ac2a619, c9057a0d. dc1fa0a3 AC3 waits for the merge (pin-bump rule). 2238696a is deferred (idle GPU) and 380e34f4 moves to wave 8 (lock-hungry).
3. **Sharded checkpoint path rule:** weight_map names are checked *lexically* (a bare file name: no `/`, `\`, `..`, absolute path, or NUL), and symlinks are followed with no containment check. The brief's original rule ("resolves inside the directory") refused every Hugging Face cache snapshot, because their shards are links to `../../blobs/<sha>`. Security impact: the untrusted input, `index.json`, can still only name entries of the checkpoint directory. A symlink there was placed by whoever controls the directory and already controls its bytes.
4. **Stop after wave 7.** The integration branch already carries 10 cards the owner hasn't reviewed, including two public API additions ojas will build on. The highest-value next work (the ojas halves, the idle-GPU re-run) waits on the merge.

Opus calls during review, recorded in the card logs:
- the per-entry host API must be bit-identical to the GPU path;
- the ICB race is fixed in capture, not by making replay conservative again;
- the coverage census uses the real preprocessor, not a reimplemented one;
- the evidence enforcement is ported to Rust (so the bench-harness lane is net +1.1k lines rather than the card's expected −1k).

## Owner decisions pending
1. **Merge `gp/wave7` into `main` and push.** `ci:local OK` at `b104ad4` (plus a CHANGELOG commit). Hosted CI has not seen any of it, and the new hosted job (sandbox-exec, setup-python on macos-26) is unverified until a push.
2. **ft-6d88defb:** which kernel families need benchmark lanes, and which may be allowlisted.
3. Carried from before and still open: ft-b8561dbb (GDN golden provenance), ft-2ffbdcd5 (self-hosted M5 runner), ft-376ad5d4 (narrowing the public surface is a breaking change), ft-a5491fbb (env-alias contract), ft-1a0f8c18 AC2 (`Q4MlxRowVariant::Wide` is used by gemma-metal).
4. **After the merge (queued wave 8):**
   - dc1fa0a3 AC3 and ojas ft-02256073 (no collision on `state.rs` in any ojas worktree);
   - ft-2238696a on an idle GPU;
   - ft-380e34f4;
   - ft-51395822.

---

# Earlier run: waves 1–5

Base: `main` @ `2972027`. All work is on branch **`gp/wave1`** (worktree `../tessl-wave1`), which merges seven lane branches. **Nothing is merged into `main`**: merging to the default branch is your call.

### How this run differed from the brief

- **GitPulse was read, not written.** The MCP server was not connected to this session, and you said to start without it. I then reached the installed `gitpulse-mcp` directly over stdio, read-only: `list_tasks`, `get_task`, `insights` and `collision_risk`. No card was moved: there are no `in_progress`, `review` or `done` writes. The "Board actions to take" section lists the moves I would make.
- **No Fable model was available**, so Opus (this session) made the sequencing and judgement calls. They are listed under "Decisions taken in place of Fable".
- **DevMap** comes from the `devmap` CLI 0.2.4, with the index at `.devmap/codeintel/devmap.sqlite`. The CLI reports the index fresh, but GitPulse's in-process DevMap calls the same store "payload obsolete", which looks like an analyzer version mismatch. Blast-radius answers were partial: 10 files failed to parse, so every impact answer is a lower bound and was treated as "unknown", not "safe".
- **The machine-wide `mac_heavy` lock dominated wall time.** Other sessions' ojas jobs held it for long stretches; one LoRA run held it for about 1.5 h. Lanes waited 1–2 h per locked run. No benchmark timing was taken on a contended GPU.
- **`gitpulse_collision_risk` failed once under load** ("background work is shed"), so that answer was treated as unknown. No other worktree or agent session was active in tessl, per `gitpulse_insights`.

### Full GPU suite, main vs gp/wave1 (M5 Pro, `cargo test --release --no-fail-fast -- --test-threads=1`, under the lock)

| Tree | Passed | Failed | Ignored |
|---|---|---|---|
| `main` @ `2972027` | 825 | 4 | 14 |
| `gp/wave1` @ `80389a0` (waves 1-2) | 837 | 4 (the same four) | 14 |

Waves 1-2 introduced no failures and add 12 tests. The four failures **are already on `main`**. Nobody had seen them, because hosted CI never runs GPU tests (README Known Gaps; ft-2ffbdcd5 is the self-hosted runner card):
- `tests/attention.rs:2071` `global_h512_with_scalars_pins_out_bf16_after_the_callback` (added in d6a3c02)
- `tests/capi.rs:1821` `a_refused_train_step_keeps_the_previous_gradients`
- `tests/capi.rs:1344` `a_two_phase_step_through_the_abi_is_the_rust_one`
- `tests/capi.rs:660` `the_model_through_the_abi_is_the_rust_model`

The ft-f78dfed7 triage verdict "fixed on main" was therefore wrong: its own regression test fails on the GPU. Wave 3 takes these four.

`scripts/ci_local.sh` on gp/wave1: fmt, release build, clippy, clippy with `quant-prep`, `cargo doc -D warnings`, tile audit (33 kernels / 38 pairs), MSL dialect lint, emulator host contract and the parity harness all pass. The suite step stopped at the first failing binary (`attention`); the `--no-fail-fast` run above is the complete picture.

#### State of gp/wave1 @ 47f13c2 (waves 1-3)

| Gate | Result |
|---|---|
| `cargo test --release --no-fail-fast -- --test-threads=1` (full GPU suite, M5 Pro) | **841 passed, 0 failed**, 14 ignored |
| `cargo fmt --all --check` | clean |
| `cargo clippy --release --all-targets -- -D warnings`, and with `--features quant-prep` | clean |
| `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps` | clean |
| MSL dialect lint, emulator host contract, tile audit (33/38) | pass |
| `bench/test_parity_harness.py --pure` | pass (skips are the GPU CLI sections and `metal-nm`) |
| `python/tests/test_binding_host.py` | 21 passed |

`main` @ 2972027 was 825 passed / 4 failed / 14 ignored on the same machine.

#### Final state of gp/wave1 after wave 4 (`scripts/ci_local.sh`, M5 Pro, under the lock)

| Gate | Result |
|---|---|
| `cargo test --release --no-fail-fast -- --test-threads=1` | **864 passed, 0 failed**, 14 ignored. The one failure in the run was `docs_name_real_tools`, catching a wave-4 doc line that named `tools/mac_heavy.sh` without its repository; fixed in `1c3b7f9` and re-run (6/6). |
| fmt, clippy, clippy with `quant-prep`, `cargo doc -D warnings`, tile audit, dialect lint, host contract, parity harness | pass |
| torch-binding tests (host and GPU, `python/tests`) | 73 OK |
| release-ready gate, `cargo package --locked`, `DOCS_RS=1 cargo check` | pass |
| Hosted CI on gp/wave1 (dispatched) | Metal compile: success (run 38506954287). CI: MSRV, fmt, and build + clippy + docs succeed; the CPU-emulator job was still running at the time of writing (run 38506951621). The GPU jobs are skipped because no runner is registered. |

### Board triage at main 2972027 (2026-10-13)

The board was stale: most "ready" P1 cards were already closed by the 10-08..10-13 merges.
Verdicts are by reading code and commits (read-only Sonnet triage, 4 lanes); none is backed by a GPU run unless stated.

| Card | Rev | Verdict | Evidence / what remains |
|---|---|---|---|
| ft-7f859b95 CI not green | 5 | Fixed on main | 34badfc parser self-test; 671db5a ci_local; CI run 38434394530 green incl. emulator steps 1-11 |
| ft-e457f193 loader refuse_unread | 3 | Fixed on main | 2688fb9; `src/qwen35_model.rs:723`, `src/bert.rs:810`; tests `tests/qwen35_model.rs:1089`, `tests/bert_tiny.rs:214` |
| ft-f78dfed7 C ABI validate-before-mutate | 3 | Fixed on main, GPU run pending | 4551470; `tests/capi_oversized.rs`, `tests/capi.rs:1685/1727/1762/1853/1883` |
| ft-6b9d840b SharedEvent lifetime | 2 | Fixed on main, GPU run pending | 9163e1a; `tests/runtime_lifecycle.rs:539,893` |
| ft-c2faf159 Staged poison / relaxed re-check | 4 | Fixed on main, GPU run pending | 98f88c7, d6a3c02; `tests/qwen35_staged.rs:160,206` |
| ft-c125d853 bind past allocations | 3 | Mostly fixed, GPU + quant-prep build pending | c8554eb; tg_mem index check and re-encode-while-executing unconfirmed |
| ft-a3b7e231 async const arena | 3 | Fixed on main, GPU run pending | 23ca02c; `tests/runtime_lifecycle.rs:442` |
| ft-15bb74a2 eps == 0 | 3 | Fixed on main, GPU run pending | e1a64f9; `src/nn.rs:191` `eps <= 0.0` |
| ft-4b7950b2 overflowing SiLU | 3 | Fixed on main, GPU run pending | 2c3d84d; `kernels/act.h:34-37`; `tests/nn_kernels.rs:267` |
| ft-a650da95 tune rigs interleave | 2 | Fixed on main, GPU run pending | cbd83b7/8c68cef; `tests/tune_rigs_refuse_missing_variants.rs` |
| ft-9a93d5ae 30 s event wait | 2 | Fixed except one test | 17113fb; missing: a tiny-budget test driving the timeout path |
| ft-4dc74fad training waste wave 2 | 7 | Fixed; 3 criteria unconfirmed | 337fedd, 628f113, 04faf93; peak-drop size, T=256/2048 coverage, `train_step_bytes` pin not checked |
| ft-b4d2a884 GEMM tune/flags | 6 | Mostly fixed | `gemm_coop.h` included only by `matmul_tensorops.metal`; tune kernels and exact-f32 templating unconfirmed |
| ft-28a94d60 build.rs warnings | 4 | Partial | dbb8200, ff443a8; clean-build timing unrecorded; planted-warning check not run |
| ft-6ac668a5 parity harness --pure | 2 | Partial | passes on numpy 2.5.3, fails 3 assertions on numpy 2.4.6 → wave 1 |
| ft-f5d97431 panicking host access | 3 | Partial | `src/embedgemma2.rs:301`, `src/decode_icb.rs:913` still `write_u32` → wave 1 |
| ft-de70a58b require green CI | 2 | Open (+ owner decision) | main unprotected (404); release.yml has no CI gate, no `--target` → wave 1 minus protection |
| ft-06dcbdff static doc drift | 6 | Open | all rows still present → wave 1 minus AdamW (owner) |
| ft-691a7474 vacuous tests | 5 | Partial / mostly open | 8 criteria; two owner decisions |
| ft-e01d19fe tessl_torch ownership | 2 | Open | owner decision: `TESSL_POISONED` status (ABI bump) vs query fn |
| ft-a2151b52 bench on Qwen35Model | 6 | Partial / mostly open | GPU + real 2B weights |
| ft-4efcd42f EmbedGemma2 speed | 6 | Open | GPU benchmarks |
| ft-5622b8a3 GDN (ld, off) windows | 2 | Open, L | GPU bit-identity + bench |
| ft-dc1fa0a3 per-entry host mapping | 4 | Open, L | needs ojas change in the same work; owner decision on C API |
| ft-a9661824 bf16 default update rule | 7 | Owner decision | default rule and fall criterion |
| ft-6cd0fa99 objc2#782 | 5 | External | owner approves upstream text |

#### Follow-up confirmation (read-only lane W2d, gp/wave1)
- ft-c2faf159 Staged: every remaining criterion MET (`tests/qwen35_staged.rs:100-112` bit-compares f32 tower logits; `tests/qwen35_model.rs:927-964` asserts `position()` after poisoned steps; docs per brief).
- ft-b4d2a884 GEMM: MET. `gemm_coop.h` owns walks/views/epilogue, `mm_coop`, `mm_exact_f32`; the six exact-f32 kernels and the tune kernels use it; `bench/results/gemm_coop_header_bits_m5pro.txt` 0/543 lines differ. The MLP up/gate split-K route (0.90-0.95x of one panel-walked dispatch) is a recorded follow-up, because changing it changes exact-f32 bits.
- ft-c125d853 binding: MET except that re-encoding while an execute is in flight is documented, not refused (`src/icb_smoke.rs:182`). The brief allows that.
- ft-4dc74fad training waste: PARTIAL. The peak drop is at least the head-gradient tensor on 4B and 2B, `train_step_bytes` is pinned, and `grad_sq_norm` is one wait (`tests/qwen35_train_traffic.rs:579`). Clipped-step waits are committed only for bf16 at T=256; bf16 at T=2048 is missing, and f32 clipped is above memguard's 32 GiB cap.

### Lanes and what landed on gp/wave1

Each lane worked in its own worktree on a branch from `main` or `gp/wave1`. Every lane had an Opus brief and an Opus diff review; lanes that changed behaviour also had a separate Sonnet adversarial pass. Every break or gap found went back to the lane before merge.

| Wave | Branch | Card | Result |
|---|---|---|---|
| 1 | `gp/release-gate` | ft-de70a58b (P1) | `scripts/require_green_ci.sh` gates `release.yml` on CI and Metal compile succeeding for exactly `$GITHUB_SHA`, failing closed: no run, a run in progress, a failed run or an API error all block. Every publishing path needs a `v*` tag matching `Cargo.toml`, and `gh release create --target "$GITHUB_SHA"`. Clippy runs with `--features quant-prep` in ci.yml, metal-compile.yml and ci_local; it was clean, so no code fixes. Stale counts in ci.yml are reworded so they cannot drift. Dry runs against real runs: green sha → ok; failed sha (run 37944170626) → rc 1; unknown sha, unknown workflow or unknown repo → rc 1; short sha → rc 2. |
| 1 | `gp/host-access-fallible` | ft-f5d97431 | `upload_segments` goes through `alloc_buffer_from_u32`. `DecodeIcb::mini_copy_chain` writes all four buffers under one `acquire_access` lease, instead of `write_u32` plus raw writes that bypassed the lease, and refuses an `n` past `u32` or `n*4` before allocating. On pre-fix code both busy tests panic at `src/tensor.rs:317`. The `set_max_tokens` rustdoc is corrected (32768, not 262144). |
| 1 | `gp/parity-harness-numpy` | ft-6ac668a5 | Root cause was **not numpy**. Whichever interpreter had `mlx` importable scored the live MLX GPU f32 matmul, which is reduced precision on the M5 Pro (3.5e-3 against 7e-7 on MLX's CPU stream), 607x over budget. `parity(external_lanes=)` is injectable and validated, the tests use deterministic stub lanes, and an empty list is reported loudly. Budgets are unchanged. Passes 5 of 5 on python3.14/numpy 2.4.6 and 5 of 5 on the oracle venv; `audit_gemm_tiles` 9/9. |
| 1 | `gp/doc-drift` | ft-06dcbdff | Every number re-derived by command: tile audit 33 kernels / 38 pairs, 800 `#[test]`, 14 `#[ignore`. Pass counts are labelled "not re-run at 2972027". CHANGELOG headings are merged with every line kept (multiset check) and an `[Unreleased]` compare link added. docs/bert.md now shows the real API. gemv_q4, gemv_q4_mlx, GEMV_ROW_TPTG and GdnTrainWorkspace rustdoc are fixed. `bench/results/PROVENANCE.md` lists the 29 imported MLSystemsLab files, each place they are presented is labelled, and the imported attention ratios and figures are removed. The adversarial fact-check found one false claim and two unlabelled spots; all three fixed. The AdamW claim is left for you. |
| 2 | `gp/capi-ownership` | ft-e01d19fe | C ABI thread check before any `&mut`. tessl_torch gains copy/pickle refusal, `close()`, `TesslPanic`, and range checks on every integer sent to the ABI. 21 host tests: 10 of the first 11 and 9 of the next 10 fail on the code before each change. The adversarial pass found no break, and the in-scope gaps it found are fixed. |
| 2 | `gp/vacuous-tests` | ft-691a7474 (part) | The docs scan covers `docs/**` and fixed two real stale references. The promoted set is derived from sources. TensorOps probes assert. Setter tests added. Each converted check was fault-injected and caught its fault. |
| 2 | `gp/closers-w2` | ft-9a93d5ae, ft-28a94d60 | The timeout test already existed, though the triage said it was missing; it is tightened (submitted value, deadline, both range ends). Build criteria are proven on a real compile: a planted warning is forwarded, and with `TESSL_METAL_WERROR=1` the build fails. Toolchain rerun triggers are confirmed. Clean build is 14.1 s min-of-3, with no pre-parallel baseline. |
| 3 | `gp/attn-h512-scalars` | new | Test bug. The reference ran the routed rows kernel and `_with_scalars` the tiled one. The test still fails when the slot-12 override is removed, and each path is bitwise self-deterministic. |
| 3 | `gp/capi-grad-state` | ft-f78dfed7 (reopened) | Three stale tests. Two predate 4551470's documented "a refused call leaves the bank as it was" (`src/capi.rs:1126`); one read gradients in transformers' shape, not the storage shape. No ABI change. |
| 4 | `gp/capi-poisoned` | ft-e01d19fe (rest) | `tessl_runtime_poisoned` query (ABI 11); tessl_torch raises `TesslPoisoned` and rebuilds the runtime; the ABI-parsing contract test caught 4 of 4 planted mismatches and found the undeclared `tessl_synchronize`. |
| 4 | `gp/host-access-ban` | ft-f5d97431 (AC4) | Fallible twins added; every non-test library call converted, loads included; `tests/host_access_ban.rs` (12 violations before the change; catches a planted call; ignores test modules); no load-time regression. Weak spots: calls on a receiver named `st` are exempt by name, and `.zero()` is not covered. |
| 4 | `gp/owner-decisions` | ft-06dcbdff A1, ft-691a7474 (2 decisions), new issues | AdamW wording; `add_metallib` refuses duplicates (no tessl or ojas caller shadows a name: the compiled overlays define only `ple_*`, `persistent_interp_*` and `ojas_*`); `ci_local` gains `--no-fail-fast` and the python tests; MLX lane scored as tf32-class (a real `--parity-dir` run scored it at 0.37x of budget, with no abort); release-gate docs; SC2012 fixed. |
| 4 | `gp/bert-goldens` | ft-691a7474 (Bert) | Generator + manifest + `tests/bert_model.rs` (BERT and DistilBERT; bound fixed before the first run; about 3e-6 error against a 2e-4 bound; a swapped-layer test and a GELU-constant fault are both caught). |
| 5 | `gp/gdn-windows` | ft-5622b8a3 | The GDN training kernels read q/k/v through `(ld, off)` windows and share key heads (`h / (Hv/Hk)`), with new `*_windowed` entry points. The dense entry points delegate to them, so ojas-metal and the C ABI are unchanged (additive; no ojas branch needed). `copy_cols`, `repeat_heads` and `sum_heads` are gone from the GDN step. Gradients are bit-identical (253 hashes per fixture). The paired step bench is within ±1% at r=1 and 4-5% faster at r=2. The adversarial pass found no break (64-bit indices confirmed on a 17 GB buffer; conservative aliasing). It also found a gap, now fixed: a dense row width past u32 is refused. The full ci_local run showed one source-text test pinned to the old index line; it now pins the new 64-bit window offsets and catches a planted 32-bit `tok`. |

Diff `main..gp/wave1`: see `git diff --stat main..gp/wave1`; the CHANGELOG heading merge alone accounts for about 885 changed lines.

### Board actions to take (GitPulse was not written)

When you are ready to update the board, these are the moves the evidence supports, with the revision each card had when read:
- **Move to review now, done after `gp/wave1` merges:** ft-de70a58b (r2, except branch protection), ft-f5d97431 (r3, except the load-time ban scope), ft-6ac668a5 (r2), ft-06dcbdff (r6, except AdamW), ft-e01d19fe (r2, except the poisoned status and the ABI-parsing contract test), ft-9a93d5ae (r2), ft-28a94d60 (r4).
- **Done on main, now confirmed by the GPU suite** (the test that card names passes in the main run): ft-7f859b95 (r5), ft-e457f193 (r3), ft-6b9d840b (r2), ft-c2faf159 (r4), ft-c125d853 (r3), ft-a3b7e231 (r3), ft-15bb74a2 (r3), ft-4b7950b2 (r3), ft-a650da95 (r2), ft-b4d2a884 (r6).
  - Caveat: the 14 `#[ignore]` tests did not run. These are the real-weights and transformers comparisons, the deep GEMM fuzz and the alignment probe. Among them is ft-c2faf159's `f32_staged_and_decode_match_forward_on_the_real_2b`, though its non-ignored tests in `tests/qwen35_staged.rs` did run and pass. Running them belongs to ft-2238696; several exceed memguard's 32 GiB cap at f32.
- **ft-f78dfed7 (r3):** done once `gp/wave1` merges. Its regression tests failed on main until wave 3.
- **ft-691a7474 (r5):** stays in progress. The overlay policy, where `python/tests` runs, the Bert goldens and the GDN C-ABI success test remain.
- **ft-4dc74fad (r7):** stays in review. The clipped-step wait counts for bf16 at T=2048 need an idle GPU.

### Deferred or blocked

| Card | Why |
|---|---|
| ft-5622b8a3 GDN `(ld, off)` windows | Done in wave 5. |
| ft-dc1fa0a3 per-entry host mapping | Decided Rust-only. Blocked on another session's uncommitted `src/qwen35_adamw.rs` edits, and its ojas side has to land in the same work. |
| ft-a2151b52, ft-4efcd42f, ft-4dc74fad (bench part) | Benchmarks need an idle GPU. The lock was contended all session and benching beside other sessions would be contaminated. ft-a2151b52 also needs the real 2B weights at 8K. |
| ft-a9661824 bf16 default update rule | Decided (stochastic rounding); another session is implementing it, uncommitted in the main checkout. |
| ft-6cd0fa99 objc2#782 | Already done upstream: reply posted, PR #870 merged on 2026-10-11. Close the card. |
| ft-de70a58b branch protection | Decided: not enabled (see decision 2). |

### Benchmarks

Waves 1-4 changed no hot path: no kernel, GEMM, attention, training-step or decode-loop code was touched. The only library changes are host-side, in one-off setup paths (`upload_segments` now uses `alloc_buffer_from_u32`, and `mini_copy_chain` is a test/smoke ICB builder) and in C-ABI argument checks. So there is no benchmark delta to report, and none was taken on the contended GPU. Wave 5 changed the GDN training kernels; its paired step bench (`bench/results/qwen35_gdn_windows_m5pro.txt`, frozen binaries, interleaved, 5 rounds, min of medians, AC power) shows 2B r=1 at 0.974-1.004x and 4B-config r=2 at 0.946-0.962x of before, with no regression over 5%. The other new measurement is the clean build: 14.10 s min-of-3, `bench/results/build_time_m5pro.txt`.

### New tasks to file (grouped by concern)

1. **GPU tests have no unattended gate.** Four GPU tests stayed red on main for days (since d6a3c02 and 4551470). `scripts/ci_local.sh` stops at the first failing binary. **`--no-fail-fast` done in wave 4.** Still to do: make the GPU suite a required gate before merges until ft-2ffbdcd5's runner exists. Fold into ft-2238696 or ft-2ffbdcd5.
2. **Done in wave 4:** MLX f32 comparison lane is tf32-class on M5 Pro. A real `--parity-dir` run still aborts on any valid dump wherever `mlx` is installed. Decide whether a comparison-runtime-only breach is fatal, or whether to score MLX on its CPU stream or label it tf32 in `LANE_PRECISION`.
3. **Done in wave 4 (all non-test library code):** Panicking host-access wrappers. `write_f32` and `read_u32` have no `try_*` twin. Load-time `write_f32` sites remain at `loader.rs:123`, `bert.rs:665/690/760/769` and `qwen35_model.rs:777`, plus raw `contents()` writes at `runtime.rs:1744` and `tensor.rs:360/386`. Fold into ft-f5d97431 once you decide the ban scope.
4. **Done in wave 4:** `tessl_torch` ABI-parsing contract test. The `extern "C"` signatures against ctypes `argtypes`/`restype`/`_fields_`, and the literal `10` in `test_cross_entropy.py`. This is the remaining criterion of ft-e01d19fe.
5. **GitPulse's in-process DevMap reports the tessl index obsolete while the `devmap` 0.2.4 CLI reports it fresh.** This is a DevCouncil/GitPulse schema or analyzer parity gap; check with `bun run check:schema-parity` in GitPulse.
6. **Documented in wave 4:** Release gate edge cases. A tag pushed before CI finishes is refused (re-run the release job once CI is green). A tag on a commit outside `metal-compile.yml`'s path filter has no Metal-compile run and is refused. Document this or add a bounded wait.
7. Minor: `metal-compile.yml:86` SC2012 (`ls`) is fixed in wave 4. In W2c's test, the "refusal names the value" assertion is weak (it only checks for `'0'`).

### Decisions taken in place of Fable

- **Board state:** read GitPulse but don't write it, following "start without GitPulse". Record the moves here instead.
- **Sequencing:** wave 1 took the one open P1 (release gate) plus three non-overlapping P2/P3s. Cards that were already fixed were confirmed by the GPU suite, not reimplemented.
- **CHANGELOG ownership:** only the doc-drift lane edited `CHANGELOG.md` during a wave. The other lanes' entries were written by Opus at merge, to avoid conflicts.
- **Integration branch:** lanes merge into `gp/wave1`, not `main`, because merging to the default branch is your decision.
- **Deferring ft-5622b8a3:** it breaks a public signature and collides with the C-ABI lane.
- **"Is the test or the code wrong?":** for the four red GPU tests, the code's written contract decided it (`src/capi.rs:1126`; `route_attn`). Each corrected test was shown to still catch the fault it exists for.
- **bool in tessl_torch:** refused where an integer is expected, accepted only as a flag (`accumulate`).
- **Parity harness scope:** fix the test's dependence on installed runtimes in this card, and file the production MLX question separately rather than change budget or fatality semantics.

### Owner decisions, made on delegation ("choose the best", 2026-10-14)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Merge `gp/wave1` into `main` | **Yes, once the main checkout is clean.** Deferred for now; the branch is pushed and hosted CI and Metal compile are dispatched on it. | Another session has uncommitted ft-a96618241 work in the main checkout (`CHANGELOG.md`, `docs/qwen35.md`, `src/qwen35_adamw.rs`, `tests/qwen35_train.rs`, modified 17:06). Merging or fast-forwarding under it would mix its working tree with wave 1. |
| 2 | Branch protection on `main` | **Not enabled.** | You land work as local merges pushed straight to `main`, and required checks would block that. Required checks would only cover hosted jobs, which run no GPU tests, so they would not have caught the four red tests. `metal-compile` is path-filtered, so a required check from it would stall unrelated commits. Publishing is already gated by `require_green_ci.sh`. The real gap, unattended GPU testing, is the self-hosted runner (ft-2ffbdcd5) plus `ci_local.sh --no-fail-fast`. |
| 3 | AdamW "torch's bits" (`src/qwen35_adamw.rs:92`) | **Soften the doc.** | The moment and denominator order matches torch. The final association differs from `addcdiv_` by up to 1 ulp, unconfirmed. A bit-equality test would need the torch-CPU oracle in the GPU suite for a claim nobody depends on. |
| 4 | Poisoned runtime in the C ABI | **Additive query function `tessl_runtime_poisoned`, ABI 10 → 11.** | A new status code changes what every existing call can return, so callers that switch on status break. A query function is additive. The version bump is needed because tessl_torch requires an exact match. ojas does not call the C ABI. |
| 5 | Panicking host-access ban scope | **All non-test library code, load paths included.** `src/bin/` and `#[cfg(test)]` are exempt, and the panicking wrappers stay public, documented as test/bench conveniences. | A load that panics on a busy runtime is the same bug class. Removing the public wrappers would break the public API. |
| 6 | `add_metallib` duplicate entry-point names | **Refuse (fail closed).** The lane stops if any caller relies on shadowing. | First-match makes which kernel runs depend on load order. |
| 7 | Where `python/tests` runs | **`ci_local.sh`**: the host tests always run (oracle python, failing loudly if missing); the GPU Python tests run under the `mac_heavy` lock and otherwise print a visible SKIPPED line. `cargo test` moves to `--no-fail-fast`. | There is no GPU runner; ci_local is the gate that runs on the M5. |
| 8 | Bert torch goldens | **Yes**, with a committed generator and a version manifest in the oracle venv. | Bert had no external oracle. Python as a fixture generator is allowed by the language policy. |
| 9 | bf16 default update rule (ft-a96618241) | **Stochastic rounding.** It is being implemented by another session in the main checkout; not duplicated here. | It was the only rule inside the 2% per-step bound (worst 0.74% against round-to-nearest's 4.6%). |
| 10 | ft-5622b8a3 GDN `(ld, off)` windows (public API/ABI change) | **Approved** for the next wave, after the C-ABI lane lands. | tessl is 0.x, so semver allows a minor-version break. The change removes two copies and blocks two other cards. **Correction:** `ojas-metal/src/device.rs:25,784-786,2062` calls `gdn_train_forward/backward`, `GdnTrainDims` and `GdnTrainWorkspace` directly. The change therefore has to carry the ojas-metal edit in the same work (tessl+ojas co-develop) and prefer an additive shape (window fields with dense defaults, or a new entry point beside the dense one) over breaking the existing signature. |
| 11 | ft-dc1fa0a3 C API exposure | **Rust-only for now.** Waits until the other session's `qwen35_adamw.rs` edits are committed. | Keeps the ABI small, and Python can add it later. Its ojas side must land in the same work. |
| 12 | MLX comparison lane | **Classify `mlx-f32` by what it measures (tf32-class on M5 Pro).** | Its result was measured, not assumed. The lane is a comparison runtime, not the thing under test. |
| 13 | Release-gate edge cases | **Document, no added wait.** | A bounded wait would have to cover about 80 min of CI. Re-running the job is the simpler fix. |
| 14 | objc2#782 reply | **Nothing to do: the card was stale.** Checked read-only: your reply with probe results was posted 2026-10-11, PR #870 (marking `newTensorWithDescriptor*` and `tensorSizeAndAlignWithDescriptor` `unsafe`) merged the same day with madsmtm's approval, and #871 merged too. Close ft-6cd0fa99. Descriptor-level `unsafe` is **not pursued now**. | The maintainer deferred it ("we can consider ... in the future"), and tessl's calls already go through the now-`unsafe` methods. The one loose end is whether `objc2-generated` commit `893ea46c` landed upstream (unchecked). |

#### Still open after the decisions

- **ft-a96618241 (bf16 default):** being implemented by another session. Its 348-line diff sits uncommitted in the main checkout and applies cleanly on gp/wave1 (`git apply --check`); a copy is in this session's scratchpad. It is not committed here, because it is not this session's work.
- **ft-dc1fa0a3:** waits for that session's `src/qwen35_adamw.rs` edits to be committed, and needs its ojas side in the same work.
- **ft-5622b8a3:** done in wave 5 and merged.
- **Benchmark cards** (ft-a2151b52, ft-4efcd42f, ft-4dc74fad's T=2048 clipped run): need an idle GPU and the real 2B weights at 8K.
- **`main`:** fast-forwarded to gp/wave1 on origin only when hosted CI is green. The local `main` checkout is left untouched because of the uncommitted work above. Whoever owns that work: `git stash && git pull --ff-only && git stash pop` (the patch applies cleanly).
8. **Host-access ban gaps:** the `st` receiver exemption is name-based, and `GpuBuffer::zero()` and the `Tensor` view `read_f32`/`write_f32` are not covered.
9. **GPU Python tests** depend on a prior `cargo build --release`, because `python/tests` has no build step.
