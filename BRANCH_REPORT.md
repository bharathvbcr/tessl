# Branch consolidation report

Date: 2026-10-16/17. Repository `/Users/bharath/Code/research/tessl`. Session `b1ba3cc0`.
Local only, nothing pushed. `origin/main` is still `60a6240`.

**Outcome:** `main` moved from `0b5f088` to **`1b900ec`**, a fast-forward from the integration line built in the detached worktree `../tessl-merge-all`. All 41 local branches were merged into `main`. Afterwards, at your request, **35 branches were deleted** with their worktrees, in two rounds: first 25 that were merged and idle, then the paused eg2 session's 10 after its uncommitted work was saved. **7 remain**: `main` and the live session's 6. 199 commits, 760 files, +97,202 / −6,523.

You overrode the brief's "never merge a branch in use or whose task is in progress" twice: "Ask fable for advice and merge all", then "merge all branches". So in-use branches were merged at their committed tips. **Uncommitted work in their worktrees was not touched and is not on `main`.**

The earlier report from this morning (session `31851a40`) is at `/Users/bharath/Code/research/tessl-backups/BRANCH_REPORT-2026-10-16-31851a40.md`.

## Inventory before the merge (2026-10-16 ~23:05Z)

`main` was `0b5f088`. No other branch was an ancestor of it, and `git cherry main <b>` had no `-` lines anywhere.
- **gp line:** `gp/wave10` contained the 8 w8/w9/w10-readers lanes. `gp/w11-2bmem` and `gp/w12-kcov` stacked on `gp/wave10`. `gp/w10-faults` and `gp/w10-gdnchunk` each had 8 patches not in `gp/wave10`. All owned by the live session `b46d2477` (pid 91618), with 5 GitPulse tasks in progress.
- **eg2 line:** `eg2/integrate` contained 25 of the 26 `worktree-agent-*` branches. `worktree-agent-a457b6d0b8ea41c76` had 6 patches not in `eg2/integrate`. All owned by session `ba5cba90`, paused on its weekly usage limit with lanes mid-flight.

The full per-branch table (tip, dates, ahead/behind, worktree, dirty, lock, owner) is in this session's first report, archived at `/Users/bharath/Code/research/tessl-backups/BRANCH_REPORT-2026-10-16-31851a40.md`. The only change since is that no tip moved: just before the fast-forward, every branch tip equalled its backup ref.

## Backups

- **Refs:** `refs/backup/2026-10-16T2305/<branch>` for all 41 branches, plus `stash-0` and `stash-1`. The list is in `/Users/bharath/Code/research/tessl-backups/tessl-backup-refs-2026-10-16T2305.txt`.
- **Bundle:** `/Users/bharath/Code/research/tessl-backups/tessl-refs-2026-10-16T2305.bundle` (33 MB, `--all`, 229 refs, sha256 `f97cead2…2aac`). `git bundle verify` reports okay and a complete history, and every branch head is present.
- **Pre-merge `main`:** `refs/backup/2026-10-16T2305/main` = `0b5f088`.

## How it was merged

- **Order:** Fable's sequence, smallest conflicts first:
  1. `gp/wave10`
  2. `gp/w11-2bmem`
  3. `gp/w12-kcov`
  4. `gp/w10-faults`
  5. `gp/w10-gdnchunk`
  6. `eg2/integrate`
  7. `worktree-agent-a457b6d0b8ea41c76`
  The other 33 branches came in through these seven.
- **Method:** each was a `git merge --no-ff` into the detached `tessl-merge-all` worktree, so the shared main tree, where `b46d2477` has its cwd, did not change until the final fast-forward.
- **Reviews:** Opus reviewed every merge that had a conflict before its commit, and every substantive fix.
- **CI:** `ci:local` steps were run on the merge line, as described under Verification.

## Merge commits on main

| Commit | What |
| --- | --- |
| `5c54400` | merge `gp/wave10` (clean) |
| `2323add` | `hosted_tests.txt`: `bins 178`. The committed wave10 tip failed the manifest count; this is the same one-line edit `b46d2477` held uncommitted in `tessl-int` |
| `057119e` | merge `gp/w11-2bmem` (clean) |
| `585a4ba` | merge `gp/w12-kcov` (clean) |
| `f0a9c4f` | `hosted_tests.txt`: select `qwen35_train::tests`. w11's new device-free test was unselected |
| `c148d38` | merge `gp/w10-faults` (clean) |
| `4b26182` | merge `gp/w10-gdnchunk` (1 conflict) |
| `8c80af7` | merge `eg2/integrate` (4 conflicts + semantic fixes) |
| `48ee95c` | `hosted_tests.txt`: select `fault_injection_script`'s 3 device-free tests (from w10-faults) |
| `e26ceae` | `msl_emu/build.sh`: a semantic conflict between `main` and eg2 |
| `1982ad7` | merge `worktree-agent-a457b6d0b8ea41c76` (5 conflicts) |
| `1b900ec` | port of w12's kernel-coverage allowlist gate into the Rust tool |

## Conflicts and how they were resolved

Both sides' intent was kept everywhere. Nothing was resolved wholesale as "ours" or "theirs".

**`gp/w10-gdnchunk` → `src/bin/bench_qwen35_train.rs`**
- Kept: wave10's `[a-z0-9_]` `op_key`, `OP_NAMES`, `--op` validation and `lane_key`, plus gdnchunk's `BENCH_GDN_KERNEL` doc and `gdn_kernel_from_env()`.
- Dropped: only gdnchunk's one-line doc for the `op_key` body that wave10 replaced.
- Opus: approved. The backward pass follows the workspace's kernel.
- Noted:
  - `bench/results/gdn_train_chunked_*` quote the pre-wave10 metric names (`op_gdn_train_forward_ms_t2048`, now `…_f32_ms_t2048`).
  - `BENCH_GDN_KERNEL` is validated after the GPU starts. That behaviour predates the merge.

**`eg2/integrate`**
- **`CHANGELOG.md`:** three `[Unreleased]` hunks, keeping both sides. eg2's "ABI 11 unchanged" claims were reworded to "ship at ABI 12", because wave10 bumped `TESSL_ABI_VERSION` to 12. The "282 kernel entry points" claim became 298.
- **`docs/benchmarking.md`:** wave10's `bench_embedgemma2` row plus eg2's new `_mm` row. eg2's older duplicate row was removed; Opus caught it.
- **`scripts/hosted_tests.txt`:** union of both sides' entries. `gdn_fixtures 12`; `bins 248` was measured, not taken from either side.
- **`src/bin/bench_embedgemma2.rs`:** union of the `use` lines.

Semantic conflicts that git did not flag, found by building and testing:
- **`tests/fixture_manifests.rs`** (wave10) matched `Json::Num { uint }`, but eg2 reshaped `Num`. It now uses `Json::exact_int() == Some(1)`, which accepts the same values; Opus checked.
- **README and `src/lib.rs`:** 298 kernel entry points. `docs_name_real_tools` reads the count from the build.
- **ABI wording:** `docs/embedgemma2.md` and `docs/embedgemma2_audit.md` now say ABI 12.
- **wave10's fixture-manifest rule vs eg2's new fixtures:**
  - `tests/fixtures/embedgemma2_tok/MANIFEST.json` uses `versions_role: reproduced`. The generator was rerun in place and all 60 oracle files came out byte-identical.
  - `results_report` is EXEMPT: its goldens came from the five Python scripts that eg2's commit `1b42066` deleted.

**main × eg2 → `tools/msl_emu/build.sh`**
- `main`'s emulator build cut `encoder_attn.metal` at a `///` marker and dropped `#include "attn_tiled.h"`. eg2 moved that code behind `encoder_attn_tiled.h` and made the marker a `//` comment.
- `build.sh` now cuts at the new marker and drops either include. The host contract passes.

**`worktree-agent-a457b6d0b8ea41c76`**
- **`bench/kernel_coverage.py` and its adversarial test:** w12 modified them, a457 deleted them for its Rust port. The deletion was taken, because the Rust port fully covers the pre-w12 Python (Opus checked the suites, labels, report, exit codes and all 14 tests). w12's additions were then ported to Rust in `1b900ec`, so there is one tool, not two.
- **`docs/benchmarking.md`:** a457's `attn_tune` row plus wave10's `bench_qwen35_layers` row, which carries `--attn-tiles`.
- **`scripts/ci_local.sh`:** `main`'s emulator-cases block is kept. The parity-harness step, whose script a457 deleted, is replaced by a457's comparator-lanes step.
- **`scripts/hosted_tests.txt`:** union of both sides' entries; `bins 256`, measured.
- **Semantic:** a457's `tests/fixtures/gemm_parity` is EXEMPT from the manifest rule. It is a captured GPU dump, scored once by `bench/gemm_sweep_mlx.py` at `b219aff`, and no generator run can reproduce device outputs. **Flag:** replace this with a manifest if you want a reproducible record.

**`1b900ec` (w12's allowlist gate in Rust):**
- `load_allowlist` and `audit_allowlist`, with the same format, classes and errors.
- The report gains the allowlisted/remaining lines and four JSON fields; stale and redundant entries are problems that fail `--check`.
- New flags `--allowlist` and `--audit-allowlist`.
- The 11 Qwen3.5 suites, `{crate}` substitution, and labels built from the full argument list.
- Tests: w12's 8 allowlist cases plus 5 more (41 → 54).
- `bench/kernel_coverage_allowlist.txt` is committed with its format header and no entries.
- Opus approved; its cheap nits are applied.

## Verification

- **After each merge step:** the device-free part of `ci:local`, meaning every step of `scripts/ci_local.sh` before the GPU `cargo test`, run against the merge worktree:
  - fmt, release build, clippy ×2, doc;
  - `hosted_tests.sh` and `--outside`;
  - dialect lint and kernel IR check;
  - emulator host contract, comparator lanes, torch binding host tests.
- **Grouping:** merges 1, 2–3 and 4–6 were each gated as one step after their fixes.
- **Final result on `1b900ec`:** see "Final CI run" below.
- **Not run: the GPU suites.** That is `cargo test --release` on the device, the GPU Python tests, and the examples. They need the machine-wide `mac_heavy` lock, which the ojas `w11-queue` has held since 17:28Z, with 4 `b46d2477` jobs queued behind it. **This means `main` now routes production GDN training through gdnchunk's chunked kernels (`d6e9dc5`) without a GPU run of the merged tree, and the owner of that task (ft-62af7c2e) still listed precision rework at strong decay as pending.** Run `tools/mac_heavy.sh` with `scripts/ci_local.sh` on `main` when the lock frees.
- **Before fast-forwarding:** all 40 branch tips were confirmed as ancestors of `1b900ec` and equal to their backup refs. No tracked path in `1b900ec` collides with the main tree's untracked `BRANCH_REPORT.md` and `tasks/`.

### Final CI run

**Every device-free step of `scripts/ci_local.sh` passed on `1b900ec`.** That commit's code is
what `main` carries; the later commits only add this report. The steps:
- fmt, release build, clippy ×2, doc;
- `hosted_tests.sh` (86 invocations, 1008 tests, device hidden) and `--outside` completeness;
- dialect lint, kernel IR check, emulator host contract;
- **emulator kernel cases**: exact, whole model and fast-math, 505 `[ok]`, no failure,
  including the chunked GDN kernels against f64;
- comparator lanes (15 tests) and the torch binding host tests (70).

The emulator cases ran in one run. The last two steps passed in a rerun with the emulator
skipped, because the first run's copy of the step list predated the a457 merge and called the
deleted `bench/test_parity_harness.py`. **Still not run:** the GPU part (`cargo test --release`,
the GPU Python tests, examples, release gate). Run `tools/mac_heavy.sh scripts/ci_local.sh` on
`main` when the lock frees.

## Branches deleted (25)

You asked to delete the old and stale branches. A branch was deleted only if it passed
every check at deletion time:
- merged into `main` (`git merge-base --is-ancestor`);
- its tip still equalled its backup ref;
- worktree clean, including any nested worktree;
- not git-locked;
- no process with its cwd inside;
- not a lane its owner last reported as running;
- no GitPulse task in progress.

Each worktree was removed with plain `git worktree remove` (no `--force`), then the branch
with `git branch -d`, which refuses unmerged work. Nothing was skipped or refused.

Since the earlier snapshot, the eight git locks on the eg2 agent worktrees, all naming the
dead pid 21579, were released by someone else; their lock files are gone. Six of those
worktrees were clean and are among the deletions.

| Branch | Tip | Backup ref | Worktree removed |
| --- | --- | --- | --- |
| `gp/w8-ci` | 343fe69 | `refs/backup/2026-10-16T2305/gp/w8-ci` | `tessl-w8-ci` |
| `gp/w8-gpurerun` | 5ac31d7 | `refs/backup/2026-10-16T2305/gp/w8-gpurerun` | `tessl-w8-gpurerun` |
| `gp/w8-untied` | 28d4907 | `refs/backup/2026-10-16T2305/gp/w8-untied` | `tessl-w8-untied` |
| `gp/w9-attn` | 3da2457 | `refs/backup/2026-10-16T2305/gp/w9-attn` | `tessl-w9-attn` |
| `gp/w9-capi` | c215dfb | `refs/backup/2026-10-16T2305/gp/w9-capi` | `tessl-w9-capi` |
| `gp/w9-gdnfix` | 2764c7c | `refs/backup/2026-10-16T2305/gp/w9-gdnfix` | `tessl-w9-gdnfix` |
| `gp/w9-icb` | 575923d | `refs/backup/2026-10-16T2305/gp/w9-icb` | `tessl-w9-icb` |
| `gp/w10-readers` | 19ceef0 | `refs/backup/2026-10-16T2305/gp/w10-readers` | `tessl-w10-readers` |
| `worktree-agent-a00e8fdf52347d85d` | fa448e0 | `refs/backup/2026-10-16T2305/worktree-agent-a00e8fdf52347d85d` | `tessl/.claude/worktrees/agent-a00e8fdf52347d85d` |
| `worktree-agent-a02cacbc66d10ac24` | 0fcfac4 | `refs/backup/2026-10-16T2305/worktree-agent-a02cacbc66d10ac24` | `tessl/.claude/worktrees/agent-a02cacbc66d10ac24` |
| `worktree-agent-a17ffa3c9da42a5f0` | c45f170 | `refs/backup/2026-10-16T2305/worktree-agent-a17ffa3c9da42a5f0` | `tessl/.claude/worktrees/agent-a17ffa3c9da42a5f0` |
| `worktree-agent-a3546229387828f05` | aeebd7b | `refs/backup/2026-10-16T2305/worktree-agent-a3546229387828f05` | `tessl/.claude/worktrees/agent-a3546229387828f05` |
| `worktree-agent-a403f81d87fc8f2cd` | d6903ab | `refs/backup/2026-10-16T2305/worktree-agent-a403f81d87fc8f2cd` | `tessl/.claude/worktrees/agent-a403f81d87fc8f2cd` |
| `worktree-agent-a4e0563c295471310` | 87b18b1 | `refs/backup/2026-10-16T2305/worktree-agent-a4e0563c295471310` | `tessl/.claude/worktrees/agent-a4e0563c295471310` |
| `worktree-agent-a52c15ece0674d300` | 40c8fdd | `refs/backup/2026-10-16T2305/worktree-agent-a52c15ece0674d300` | `tessl/.claude/worktrees/agent-a52c15ece0674d300` |
| `worktree-agent-a612242420f6baf71` | 2206fda | `refs/backup/2026-10-16T2305/worktree-agent-a612242420f6baf71` | `tessl/.claude/worktrees/agent-a612242420f6baf71` |
| `worktree-agent-a7751fb151981d840` | a645bdf | `refs/backup/2026-10-16T2305/worktree-agent-a7751fb151981d840` | `tessl/.claude/worktrees/agent-a7751fb151981d840` |
| `worktree-agent-a975c42e7f8ed0908` | a0e24b4 | `refs/backup/2026-10-16T2305/worktree-agent-a975c42e7f8ed0908` | `tessl/.claude/worktrees/agent-a975c42e7f8ed0908` |
| `worktree-agent-aa4d6ba1364869275` | c45be7b | `refs/backup/2026-10-16T2305/worktree-agent-aa4d6ba1364869275` | `tessl/.claude/worktrees/agent-aa4d6ba1364869275` |
| `worktree-agent-ad505cb5058a3b760` | 4ed97b7 | `refs/backup/2026-10-16T2305/worktree-agent-ad505cb5058a3b760` | `tessl/.claude/worktrees/agent-ad505cb5058a3b760` |
| `worktree-agent-ad79a641b2fc3d74c` | 967fc6b | `refs/backup/2026-10-16T2305/worktree-agent-ad79a641b2fc3d74c` | `tessl/.claude/worktrees/agent-ad79a641b2fc3d74c` |
| `worktree-agent-ae0500f286ff01707` | fb026e6 | `refs/backup/2026-10-16T2305/worktree-agent-ae0500f286ff01707` | `tessl/.claude/worktrees/agent-ae0500f286ff01707` |
| `worktree-agent-ae7a8e24557c0eb7f` | d66233e | `refs/backup/2026-10-16T2305/worktree-agent-ae7a8e24557c0eb7f` | `tessl/.claude/worktrees/agent-ae7a8e24557c0eb7f` |
| `worktree-agent-af207c3bb087f8d41` | 6765661 | `refs/backup/2026-10-16T2305/worktree-agent-af207c3bb087f8d41` | `tessl/.claude/worktrees/agent-af207c3bb087f8d41` |
| `worktree-agent-afcb4d0fa175439a7` | 7ae7371 | `refs/backup/2026-10-16T2305/worktree-agent-afcb4d0fa175439a7` | `tessl/.claude/worktrees/agent-afcb4d0fa175439a7` |

## Second deletion round: the paused eg2 session's 10 branches

You chose "eg2 lane, saved first". Session `ba5cba90` had no process, and its transcript was
last written at 14:54 local time on 2026-10-16. Its uncommitted work was reviewed: the file
lists, sizes (the largest snapshot is 2.8 MB, the largest file 108 KB) and a check for
secret-like paths. Then:

- **Saved as `wip(...)` commits on each lane's own branch:**
  - `a1445…`: SVG plot port, 36 files.
  - `ae38…`: media decode, 4 modified files plus untracked `src/media/`, bench and fixtures.
  - `acde…`: binding lane, 3 files.
  - `a42f…`: one untracked bench-lanes directory.
- **The four nested baseline worktrees**, saved as detached commits:
  - `refs/backup/2026-10-17/nested/a1bcd94-before` (`0f772c4`)
  - `refs/backup/2026-10-17/nested/a220451-before` (`219181d`)
  - `refs/backup/2026-10-17/nested/acae0f3-base` (`e59f8c3`)
  - `refs/backup/2026-10-17/nested/ad119ed-before` (`0de954f`)
- **Backups:** all 10 tips and the 4 nested snapshots are under `refs/backup/2026-10-17/`
  (list: `/Users/bharath/Code/research/tessl-backups/tessl-backup-refs-2026-10-17.txt`) and in
  `/Users/bharath/Code/research/tessl-backups/tessl-refs-2026-10-17.bundle`. `git bundle
  verify` reports okay and a complete history, and all 14 refs are present.
- **Removal:** nested worktrees first, then each lane's worktree, all with plain
  `git worktree remove`.
  - Six branches already in `main` were deleted with `git branch -d`.
  - **Four carried only their WIP snapshot, so `git branch -D` was needed.** That work is
    **not on `main`**; it lives only in the backup refs and the bundle.

| Branch | Tip | Kind | Backup ref | Worktree removed |
| --- | --- | --- | --- | --- |
| `worktree-agent-a1445f2b976322e5a` | 514bb88 | wip-only (-D) | `refs/backup/2026-10-17/worktree-agent-a1445f2b976322e5a` | `tessl/.claude/worktrees/agent-a1445f2b976322e5a` |
| `worktree-agent-a1bcd94eed1deb932` | d108ff4 | merged | `refs/backup/2026-10-17/worktree-agent-a1bcd94eed1deb932` | `tessl/.claude/worktrees/agent-a1bcd94eed1deb932` |
| `worktree-agent-a2204518a9d6e9786` | 845807c | merged | `refs/backup/2026-10-17/worktree-agent-a2204518a9d6e9786` | `tessl/.claude/worktrees/agent-a2204518a9d6e9786` |
| `worktree-agent-a42fc69fdc40fda3e` | 7a202b9 | wip-only (-D) | `refs/backup/2026-10-17/worktree-agent-a42fc69fdc40fda3e` | `tessl/.claude/worktrees/agent-a42fc69fdc40fda3e` |
| `worktree-agent-a457b6d0b8ea41c76` | a5fc7ae | merged | `refs/backup/2026-10-17/worktree-agent-a457b6d0b8ea41c76` | `tessl/.claude/worktrees/agent-a457b6d0b8ea41c76` |
| `worktree-agent-acae0f3c2690f3353` | 9cf2c87 | merged | `refs/backup/2026-10-17/worktree-agent-acae0f3c2690f3353` | `tessl/.claude/worktrees/agent-acae0f3c2690f3353` |
| `worktree-agent-acde1607dcb642397` | 0151fd9 | wip-only (-D) | `refs/backup/2026-10-17/worktree-agent-acde1607dcb642397` | `tessl/.claude/worktrees/agent-acde1607dcb642397` |
| `worktree-agent-ad119ed6f8584407f` | 7b8ac2f | merged | `refs/backup/2026-10-17/worktree-agent-ad119ed6f8584407f` | `tessl/.claude/worktrees/agent-ad119ed6f8584407f` |
| `worktree-agent-ae38ec46fbffd47ae` | 9dd757b | wip-only (-D) | `refs/backup/2026-10-17/worktree-agent-ae38ec46fbffd47ae` | `tessl/.claude/worktrees/agent-ae38ec46fbffd47ae` |
| `eg2/integrate` | 8967e31 | merged | `refs/backup/2026-10-17/eg2/integrate` | `tessl-eg2` |

To resume a saved lane: `git branch <b> refs/backup/2026-10-17/<b>`, then `git worktree add
<path> <b>`. The `wip` commit is the uncommitted state, as it was.

## Branches kept (7)

- **`main`.**
- **`gp/clip-sq-norm`:** new work by the live session `b46d2477`, created from `main` after
  the merge.
- **`gp/wave10`, `gp/w10-faults`, `gp/w11-2bmem`, `gp/w12-kcov`:** queued `mac_heavy` jobs in
  their worktrees; `gp/wave10` also has 1 dirty file.
- **`gp/w10-gdnchunk`:** its task (ft-62af7c2e) is in progress, with rework pending.

All 6 are ancestors of `main`. `git branch -d` works on each once `b46d2477` is done with it.

After round two: `git fsck` exits 0, every 2026-10-17 backup ref resolves, and
`gitpulse_insights` shows sessions 30794 and 91618 busy, with 8 worktrees.

## Follow-ups for the owners

1. **`b46d2477`:**
   - Its `gp/wave10` → `main` merge is done.
   - Its uncommitted `scripts/hosted_tests.txt` edit in `tessl-int` (`bins 178`) is already on `main`, and `main`'s count is now 256.
   - The 0.3.0 version bump and CHANGELOG header belong on `main`.
   - Its in-progress lanes (10A, 10B, 11A, 12A) should merge `main` before they continue.
2. **`ba5cba90`:** its branches are deleted (see the second deletion round); restore them from `refs/backup/2026-10-17/`, then merge `main` into `eg2/integrate` on resume. Its 4 dirty agent worktrees (`a1445…` plot port, `ae38…` media decode, `acde…` binding, `a42f…`) touch `docs/benchmarking.md`, `scripts/hosted_tests.txt`, `Cargo.toml` and `src/lib.rs`, so expect conflicts when they commit.
3. **ft-6d88defb:** the allowlist file is empty. The "150 of 266" figure predates a census of 298. Both need a GPU coverage run.
4. **Not done here:**
   - `tests/fixtures/embedgemma2` records its audio/vision fixtures in two more provenance files that the manifest gate does not check.
   - README still says "ABI version 10"; that was already stale before this work.
   - `src/qwen35_train.rs:1537` uses `async_encode_enabled()` where eg2 moved its siblings to `dispatch_batching()`.
5. **Stashes `stash@{0}` and `stash@{1}`** are untouched. `stash@{0}` describes itself as already landed.
6. **Earlier this session:** the stale `tessl-eg2/target/before` worktree entry was pruned, and the clean `tessl-ci-main` worktree was removed (both approved by you).

## Restore

```sh
# main as it was before the merge (run in the main tree; --keep refuses if it would lose local edits):
git -C /Users/bharath/Code/research/tessl reset --keep refs/backup/2026-10-16T2305/main
# any deleted branch (and its worktree, if wanted):
git branch <branch> refs/backup/2026-10-16T2305/<branch>
git worktree add ../tessl-<name> <branch>
# from the bundle if the refs are gone:
git fetch /Users/bharath/Code/research/tessl-backups/tessl-refs-2026-10-16T2305.bundle refs/heads/<branch>:refs/heads/<branch>
```

## Pass 3 folder cleanup (2026-10-17 ~11:55 CDT)

Goal: remove the `/Users/bharath/Code/research/tessl-*` worktree folders whose work is already on `main` (42f1d52).

**Checks run first:**
- `gitpulse_insights`: every facet `ok`, 14 of 14 worktrees scanned, all candidates clean.
- `gitpulse_collision_risk`: 0 overlaps, `sessions_ok`.
- Live sessions: pid 24500 (`tessl-eg2`) and this session. pid 91618 (`b46d2477`) is gone.
- `lsof -a -d cwd`: processes only in `tessl` and `tessl-eg2`.
- No `b46d2477` jobs are waiting on `mac_heavy`; only ojas `ojas-clip-rows` jobs are queued.

**Result: no sibling folder was removed.** Only the stale `old` entry went. Every candidate is either not on `main`, or is merged but its GitPulse task is still `in_progress`.

| Folder | Tip | On `main`? | Decision and reason | Backup ref (`refs/backup/2026-10-17-pass3/…`) |
| --- | --- | --- | --- | --- |
| `tessl-w10-faults` | `gp/w10-faults` f9039b8 | yes (ancestor) | **kept**: task ft-380e34f4 (fault-injection set) is `in_progress` | `gp/w10-faults` |
| `tessl-w10-gdnchunk` | `gp/w10-gdnchunk` d1175c7 | yes (ancestor) | **kept**: task ft-62af7c2e (chunked GDN) is `in_progress` | `gp/w10-gdnchunk` |
| `tessl-w11-2bmem` | `gp/w11-2bmem` fcb7a08 | **no**: fcb7a08 (23:02 on 10-16) is not on `main` | **kept**: unmerged; tasks ft-02b223f0 and ft-19f4bd16 are `in_progress` | `gp/w11-2bmem` |
| `tessl-w12-kcov` | `gp/w12-kcov` 1a6b606 | **no**: 1a6b606 (22:50 on 10-16) is not on `main` | **kept**: unmerged; task ft-6d88defb is `in_progress` | `gp/w12-kcov` |
| `tessl-merge2` | detached 4685388 | **no** | **kept**: `main` + `gp/w11-2bmem` + `gp/w12-kcov`, the ready-made merge of the two lines above | `detached/tessl-merge2` |
| `tessl-merge-w11` | detached 60ac94f | **no** | **kept**: despite its name, this is `main` + `gp/clip-sq-norm` (the clip line). It is not an ancestor of `tessl-merge-clip`'s 5cdf05d | `detached/tessl-merge-w11` |
| `tessl-merge-clip` | detached 5cdf05d | no | kept, as instructed (clip line; ojas `gp/clip-sq-norm` waits on it) | `detached/tessl-merge-clip` (its first backup) |
| `tessl-clipnorm` | `gp/clip-sq-norm` 2c02982 | no | kept, as instructed (clip line) | `gp/clip-sq-norm` |
| `tessl-eg2` | `eg2/integrate` eb69075 | — | kept: live session pid 24500 | — |
| `.claude/worktrees/agent-a02125…`, `agent-a86c7…` | — | — | kept: locked by pid 24500 | — |
| `.claude/worktrees/agent-af4e93…` | — | — | kept, as instructed | — |
| `old` (in `b46d2477`'s `/private/tmp` scratchpad) | detached 0fb1b30 (old `gp/wave10` tip) | yes (ancestor) | **removed** with `git worktree remove` (no `--force`). The session is gone, the tree was clean, and no task is open. The rest of the scratchpad (`run.sh`, logs, `old-target`) is untouched | `detached/old` |

**Notes:**
- **Branches deleted: none.** `gp/wave10` was already deleted before this pass; its tip is in `refs/backup/2026-10-17T0809/gp/wave10` (0fb1b30).
- **Bundle:** `tessl-backups/tessl-refs-2026-10-17-pass3.bundle` holds all 10 pass-3 refs, including `main` at 42f1d52, and `git bundle verify` passes. The ref list is in `tessl-backups/tessl-backup-refs-2026-10-17-pass3.txt`.
- **Not a worktree:** `/Users/bharath/Code/research/tessl-w2e` holds only a `target/` build-output folder from 2026-10-12. It is left for the owner to delete.
- **Correction to round two above:** that section says all 6 kept branches are ancestors of `main`. After it was written, `gp/w11-2bmem` and `gp/w12-kcov` each gained one commit (fcb7a08 and 1a6b606), so that is no longer true.

**To unblock the next pass:**
- `gp/w10-faults` and `gp/w10-gdnchunk`: once the owner closes or reassigns ft-380e34f4 and ft-62af7c2e (their owning session `b46d2477` is gone), run `git worktree remove ../tessl-w10-faults`, `git worktree remove ../tessl-w10-gdnchunk`, and `git branch -d gp/w10-faults gp/w10-gdnchunk`.
- `gp/w11-2bmem` and `gp/w12-kcov`: merge them first (`tessl-merge2` already holds that merge), then do the same.

### Restore (pass 3)

```sh
# the old worktree (detached at the old gp/wave10 tip):
git worktree add --detach <path> refs/backup/2026-10-17-pass3/detached/old
# any examined tip, if its branch or worktree is lost later:
git branch <branch> refs/backup/2026-10-17-pass3/gp/<name>
# from the bundle if the refs are gone:
git fetch /Users/bharath/Code/research/tessl-backups/tessl-refs-2026-10-17-pass3.bundle 'refs/backup/2026-10-17-pass3/*:refs/backup/2026-10-17-pass3/*'
```
