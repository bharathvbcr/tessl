# BUILD_REPORT — GitPulse backlog, 2026-10-13/14

Base: `main` @ `2972027`. All work is on branch **`gp/wave1`** (worktree `../tessl-wave1`), which merges seven lane branches. **Nothing is merged into `main`**: merging to the default branch is your call.

## How this run differed from the brief

- **GitPulse was read, not written.** The MCP server was not connected to this session, and you said to start without it. I then reached the installed `gitpulse-mcp` directly over stdio, read-only: `list_tasks`, `get_task`, `insights` and `collision_risk`. No card was moved: there are no `in_progress`, `review` or `done` writes. The "Board actions to take" section lists the moves I would make.
- **No Fable model was available**, so Opus (this session) made the sequencing and judgement calls. They are listed under "Decisions taken in place of Fable".
- **DevMap** comes from the `devmap` CLI 0.2.4, with the index at `.devmap/codeintel/devmap.sqlite`. The CLI reports the index fresh, but GitPulse's in-process DevMap calls the same store "payload obsolete", which looks like an analyzer version mismatch. Blast-radius answers were partial: 10 files failed to parse, so every impact answer is a lower bound and was treated as "unknown", not "safe".
- **The machine-wide `mac_heavy` lock dominated wall time.** Other sessions' ojas jobs held it for long stretches; one LoRA run held it for about 1.5 h. Lanes waited 1–2 h per locked run. No benchmark timing was taken on a contended GPU.
- **`gitpulse_collision_risk` failed once under load** ("background work is shed"), so that answer was treated as unknown. No other worktree or agent session was active in tessl, per `gitpulse_insights`.

## Full GPU suite, main vs gp/wave1 (M5 Pro, `cargo test --release --no-fail-fast -- --test-threads=1`, under the lock)

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

### State of gp/wave1 @ 47f13c2 (waves 1-3)

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

### Final state of gp/wave1 after wave 4 (`scripts/ci_local.sh`, M5 Pro, under the lock)

| Gate | Result |
|---|---|
| `cargo test --release --no-fail-fast -- --test-threads=1` | **864 passed, 0 failed**, 14 ignored. The one failure in the run was `docs_name_real_tools`, catching a wave-4 doc line that named `tools/mac_heavy.sh` without its repository; fixed in `1c3b7f9` and re-run (6/6). |
| fmt, clippy, clippy with `quant-prep`, `cargo doc -D warnings`, tile audit, dialect lint, host contract, parity harness | pass |
| torch-binding tests (host and GPU, `python/tests`) | 73 OK |
| release-ready gate, `cargo package --locked`, `DOCS_RS=1 cargo check` | pass |
| Hosted CI on gp/wave1 (dispatched) | Metal compile: success (run 38506954287). CI: MSRV, fmt, and build + clippy + docs succeed; the CPU-emulator job was still running at the time of writing (run 38506951621). The GPU jobs are skipped because no runner is registered. |

## Board triage at main 2972027 (2026-10-13)

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

### Follow-up confirmation (read-only lane W2d, gp/wave1)
- ft-c2faf159 Staged: every remaining criterion MET (`tests/qwen35_staged.rs:100-112` bit-compares f32 tower logits; `tests/qwen35_model.rs:927-964` asserts `position()` after poisoned steps; docs per brief).
- ft-b4d2a884 GEMM: MET. `gemm_coop.h` owns walks/views/epilogue, `mm_coop`, `mm_exact_f32`; the six exact-f32 kernels and the tune kernels use it; `bench/results/gemm_coop_header_bits_m5pro.txt` 0/543 lines differ. The MLP up/gate split-K route (0.90-0.95x of one panel-walked dispatch) is a recorded follow-up, because changing it changes exact-f32 bits.
- ft-c125d853 binding: MET except that re-encoding while an execute is in flight is documented, not refused (`src/icb_smoke.rs:182`). The brief allows that.
- ft-4dc74fad training waste: PARTIAL. The peak drop is at least the head-gradient tensor on 4B and 2B, `train_step_bytes` is pinned, and `grad_sq_norm` is one wait (`tests/qwen35_train_traffic.rs:579`). Clipped-step waits are committed only for bf16 at T=256; bf16 at T=2048 is missing, and f32 clipped is above memguard's 32 GiB cap.

## Lanes and what landed on gp/wave1

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

## Board actions to take (GitPulse was not written)

When you are ready to update the board, these are the moves the evidence supports, with the revision each card had when read:
- **Move to review now, done after `gp/wave1` merges:** ft-de70a58b (r2, except branch protection), ft-f5d97431 (r3, except the load-time ban scope), ft-6ac668a5 (r2), ft-06dcbdff (r6, except AdamW), ft-e01d19fe (r2, except the poisoned status and the ABI-parsing contract test), ft-9a93d5ae (r2), ft-28a94d60 (r4).
- **Done on main, now confirmed by the GPU suite** (the test that card names passes in the main run): ft-7f859b95 (r5), ft-e457f193 (r3), ft-6b9d840b (r2), ft-c2faf159 (r4), ft-c125d853 (r3), ft-a3b7e231 (r3), ft-15bb74a2 (r3), ft-4b7950b2 (r3), ft-a650da95 (r2), ft-b4d2a884 (r6).
  - Caveat: the 14 `#[ignore]` tests did not run. These are the real-weights and transformers comparisons, the deep GEMM fuzz and the alignment probe. Among them is ft-c2faf159's `f32_staged_and_decode_match_forward_on_the_real_2b`, though its non-ignored tests in `tests/qwen35_staged.rs` did run and pass. Running them belongs to ft-2238696; several exceed memguard's 32 GiB cap at f32.
- **ft-f78dfed7 (r3):** done once `gp/wave1` merges. Its regression tests failed on main until wave 3.
- **ft-691a7474 (r5):** stays in progress. The overlay policy, where `python/tests` runs, the Bert goldens and the GDN C-ABI success test remain.
- **ft-4dc74fad (r7):** stays in review. The clipped-step wait counts for bf16 at T=2048 need an idle GPU.

## Deferred or blocked

| Card | Why |
|---|---|
| ft-5622b8a3 GDN `(ld, off)` windows | Done in wave 5. |
| ft-dc1fa0a3 per-entry host mapping | Decided Rust-only. Blocked on another session's uncommitted `src/qwen35_adamw.rs` edits, and its ojas side has to land in the same work. |
| ft-a2151b52, ft-4efcd42f, ft-4dc74fad (bench part) | Benchmarks need an idle GPU. The lock was contended all session and benching beside other sessions would be contaminated. ft-a2151b52 also needs the real 2B weights at 8K. |
| ft-a9661824 bf16 default update rule | Decided (stochastic rounding); another session is implementing it, uncommitted in the main checkout. |
| ft-6cd0fa99 objc2#782 | Already done upstream: reply posted, PR #870 merged on 2026-10-11. Close the card. |
| ft-de70a58b branch protection | Decided: not enabled (see decision 2). |

## Benchmarks

Waves 1-4 changed no hot path: no kernel, GEMM, attention, training-step or decode-loop code was touched. The only library changes are host-side, in one-off setup paths (`upload_segments` now uses `alloc_buffer_from_u32`, and `mini_copy_chain` is a test/smoke ICB builder) and in C-ABI argument checks. So there is no benchmark delta to report, and none was taken on the contended GPU. Wave 5 changed the GDN training kernels; its paired step bench (`bench/results/qwen35_gdn_windows_m5pro.txt`, frozen binaries, interleaved, 5 rounds, min of medians, AC power) shows 2B r=1 at 0.974-1.004x and 4B-config r=2 at 0.946-0.962x of before, with no regression over 5%. The other new measurement is the clean build: 14.10 s min-of-3, `bench/results/build_time_m5pro.txt`.

## New tasks to file (grouped by concern)

1. **GPU tests have no unattended gate.** Four GPU tests stayed red on main for days (since d6a3c02 and 4551470). `scripts/ci_local.sh` stops at the first failing binary. **`--no-fail-fast` done in wave 4.** Still to do: make the GPU suite a required gate before merges until ft-2ffbdcd5's runner exists. Fold into ft-2238696 or ft-2ffbdcd5.
2. **Done in wave 4:** MLX f32 comparison lane is tf32-class on M5 Pro. A real `--parity-dir` run still aborts on any valid dump wherever `mlx` is installed. Decide whether a comparison-runtime-only breach is fatal, or whether to score MLX on its CPU stream or label it tf32 in `LANE_PRECISION`.
3. **Done in wave 4 (all non-test library code):** Panicking host-access wrappers. `write_f32` and `read_u32` have no `try_*` twin. Load-time `write_f32` sites remain at `loader.rs:123`, `bert.rs:665/690/760/769` and `qwen35_model.rs:777`, plus raw `contents()` writes at `runtime.rs:1744` and `tensor.rs:360/386`. Fold into ft-f5d97431 once you decide the ban scope.
4. **Done in wave 4:** `tessl_torch` ABI-parsing contract test. The `extern "C"` signatures against ctypes `argtypes`/`restype`/`_fields_`, and the literal `10` in `test_cross_entropy.py`. This is the remaining criterion of ft-e01d19fe.
5. **GitPulse's in-process DevMap reports the tessl index obsolete while the `devmap` 0.2.4 CLI reports it fresh.** This is a DevCouncil/GitPulse schema or analyzer parity gap; check with `bun run check:schema-parity` in GitPulse.
6. **Documented in wave 4:** Release gate edge cases. A tag pushed before CI finishes is refused (re-run the release job once CI is green). A tag on a commit outside `metal-compile.yml`'s path filter has no Metal-compile run and is refused. Document this or add a bounded wait.
7. Minor: `metal-compile.yml:86` SC2012 (`ls`) is fixed in wave 4. In W2c's test, the "refusal names the value" assertion is weak (it only checks for `'0'`).

## Decisions taken in place of Fable

- **Board state:** read GitPulse but don't write it, following "start without GitPulse". Record the moves here instead.
- **Sequencing:** wave 1 took the one open P1 (release gate) plus three non-overlapping P2/P3s. Cards that were already fixed were confirmed by the GPU suite, not reimplemented.
- **CHANGELOG ownership:** only the doc-drift lane edited `CHANGELOG.md` during a wave. The other lanes' entries were written by Opus at merge, to avoid conflicts.
- **Integration branch:** lanes merge into `gp/wave1`, not `main`, because merging to the default branch is your decision.
- **Deferring ft-5622b8a3:** it breaks a public signature and collides with the C-ABI lane.
- **"Is the test or the code wrong?":** for the four red GPU tests, the code's written contract decided it (`src/capi.rs:1126`; `route_attn`). Each corrected test was shown to still catch the fault it exists for.
- **bool in tessl_torch:** refused where an integer is expected, accepted only as a flag (`accumulate`).
- **Parity harness scope:** fix the test's dependence on installed runtimes in this card, and file the production MLX question separately rather than change budget or fatality semantics.

## Owner decisions, made on delegation ("choose the best", 2026-10-14)

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

### Still open after the decisions

- **ft-a96618241 (bf16 default):** being implemented by another session. Its 348-line diff sits uncommitted in the main checkout and applies cleanly on gp/wave1 (`git apply --check`); a copy is in this session's scratchpad. It is not committed here, because it is not this session's work.
- **ft-dc1fa0a3:** waits for that session's `src/qwen35_adamw.rs` edits to be committed, and needs its ojas side in the same work.
- **ft-5622b8a3:** done in wave 5 and merged.
- **Benchmark cards** (ft-a2151b52, ft-4efcd42f, ft-4dc74fad's T=2048 clipped run): need an idle GPU and the real 2B weights at 8K.
- **`main`:** fast-forwarded to gp/wave1 on origin only when hosted CI is green. The local `main` checkout is left untouched because of the uncommitted work above. Whoever owns that work: `git stash && git pull --ff-only && git stash pop` (the patch applies cleanly).
8. **Host-access ban gaps:** the `st` receiver exemption is name-based, and `GpuBuffer::zero()` and the `Tensor` view `read_f32`/`write_f32` are not covered.
9. **GPU Python tests** depend on a prior `cargo build --release`, because `python/tests` has no build step.
