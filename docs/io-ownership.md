# Decision: ojas owns safetensors, JSON, and NPY IO; tessl re-exports ojas-io

Owner decision, 2026-11-01 (completed in Milestone M2, 2026-11-02). The ojas side of this record is
[`ojas/docs/io-ownership.md`](https://github.com/bharathvbcr/ojas/blob/main/docs/io-ownership.md),
which carries the full evidence; this is the same decision stated from tessl's
side. It is the IO slice of the ownership split: ojas is the training framework
and owns checkpoints and IO, tessl is the Metal 4 extension and the Mac
inference engine.

## What changed here

`src/safetensors.rs` held a complete, strict `.safetensors` parser — 950 lines
of header grammar, offset validation, tiling checks, index parsing and
positioned reads — alongside standalone parsers in `src/json.rs` (516 lines),
`src/npy.rs` (417 lines), and `src/plain.rs` (145 lines). `ojas-io` had its own
copies.

In the initial P8 step (2026-11-01, recorded below), `src/safetensors.rs` was reduced
to a 517-line adapter over `ojas_io::safetensors`. In Milestone M2 (2026-11-02),
`ojas-io` absorbed `.npy` v1.0/v2.0 reading/writing and the remaining safetensors
helper surface so that **`src/safetensors.rs`, `src/json.rs`, `src/npy.rs`, and
`src/plain.rs` (-1,595 LOC total) were deleted in full**. `src/lib.rs` now
re-exports `pub use ojas_io::{npy, safetensors};` directly and `src/config_json.rs`
parses model configs through `ojas_io::json::{JsonValue, JsonLimits}`.

The tree as a whole did **not** shrink, and it is worth saying so rather than
reporting only this side of it: ojas-io grew by 1,396 insertions against 82
deletions to take the format over, of which about 385 lines are new tests —
`ojas-io/tests/tessl_rejection_corpus.rs` (246),
`ojas-io/tests/header_amplification.rs` (101)
and the `widen_in_place` test that moved here with the code it covers (38).
What was removed is the duplicate; what was added is the discipline the
surviving reader did not have, with its tests.

`ojas-io` was already a declared path dependency of this crate and was used
nowhere, so this needed no `Cargo.toml` change and no MSRV ask.

## What the adapter keeps, and why

The adapter exists because three things here are tessl's, not the reader's.
Everything else — the grammar, the schema, the tiling, dtype widths, positioned
reads, the file-identity re-check, the index bound, shard validation — is
ojas-io's and is not duplicated.

- **The surface.** `String` errors, `Vec<usize>` shapes, and the method names
  the loaders already call. **46 of the 48** files outside `src/safetensors.rs`
  that reference `SafeTensors`, `CheckpointView`, `Dtype` or `TensorInfo`
  compile untouched (48 by `git grep -l` over tracked files, excluding `docs/`;
  the card counted 47). That was the design constraint, not an outcome: if the
  loaders had needed editing, the adapter would not have been faithful.

  The two edited are both tests of the reader itself, not callers of it:

  - `tests/safetensors.rs`, the rejection-parity gate: of the 52 malformed
    inputs it covers, 42 keep their exact needle and 10 are the same refusal in
    ojas-io's words, each annotated in place with the needle it had, and the
    sub-byte block gained two bit-exact inputs. No assertion was deleted or
    weakened.
  - `src/parser_mutation.rs`, whose `header_amplification_is_pinned` pinned the
    behaviour this change deliberately narrows. It is now
    `header_amplification_is_refused_not_merely_bounded`, which pins the
    refusal instead — a harder bound than the ratio check it replaced.
- **The dtype contract.** `Dtype` keeps its nineteen variants and no sub-byte
  float. ojas-io supports `F4`, `F6_E2M3` and `F6_E3M2` on a checked path,
  sizing them from their bit width and refusing a span that is not bit-exact;
  tessl declines them, because every path below this module is written in whole
  bytes. `TryFrom<StDtype> for Dtype` errors on all three, applied once per
  header at open, so the refusal is enforced by the type rather than
  documented. `pub type Dtype = StDtype` was rejected, but **not for the
  reason the card gives**, which was checked on 2026-11-02 and is wrong: the
  four matches at `src/qwen35.rs` :1774, :2779, :3518 and :3610 are on
  `DType`, ojas-core's compute dtype, each with a catch-all arm, and that file
  never names `safetensors::Dtype`; `const ALL_DTYPES: [Dtype; 19]` in
  `tests/qwen35_mlx_q4_adversarial.rs` is an array literal, which needs no
  exhaustiveness. The alias would compile.

  The reason it is rejected is behavioural, and survives that correction: the
  alias puts `F4`, `F6_E2M3` and `F6_E3M2` into this crate's namespace with no
  gate, so a checkpoint naming one would be **opened** here instead of refused,
  and every path below this module is written in whole bytes. It would also
  leave `ALL_DTYPES` sweeping 19 of 22 variants — three of them exactly the
  ones newly admitted.
- **The payload policy.** A `None` limit means this machine's physical memory
  here, and a checkpoint declaring more is refused at open. ojas-io treats
  `None` as unbounded deliberately — it validates a header without loading the
  file and reads positioned from disk, so opening a checkpoint larger than
  memory is legitimate there. The budget, its wording and the per-read
  allocation check all still go through `plain::check_payload_budget`, the one
  owner shared with the `.npy` reader, which is why `tests/reader_bounds.rs`
  passes with no edit at all.

## What tessl gives up

**Header amplification.** ojas-io budgets the JSON tree as it builds it
(`8 * input + 64 KiB`) and refuses a header that would exceed it. This reader
had no such bound and would open a 10 MB header of five million shape
dimensions, bounding only its largest single allocation. ojas-io is the
stricter of the two and stays that way, because it is the reader both
repositories now go through.

So a header made almost entirely of shape dimensions can no longer be opened
here, above about 8 KiB of dimension text — measured at 4,099 dimensions, in
`ojas-io/tests/header_amplification.rs`. Nothing writes such a header: real
tensors have rank 5 at most, a header's bytes are mostly tensor names and dtype
tags, and the Qwen3.5-2B's whole header is about 90 KB (the companion test
there opens a 630-tensor header to keep that honest).

`src/parser_mutation.rs`'s `header_amplification_is_pinned` asserted the old
behaviour and is now
`header_amplification_is_refused_not_merely_bounded`: it pins the refusal, that
the refusal costs no large allocation, and that a header below the budget still
reads. That is a harder bound than the ratio check it replaced, which is the
property the test was reaching for.

**Ratified by the owner on 2026-11-02: keep the bound.** It is a real
narrowing of what this crate can load, which is why it was put to the owner
rather than recorded as a detail; the answer is that the reader both
repositories go through stays the stricter of the two. Loosening it later is a
decision about `ojas-io`'s `JsonLimits`, not about this adapter, and
`ojas-io/tests/stream_writer.rs` will have to agree with whatever it becomes —
it refuses layouts the reader would refuse, which is what made the 32x
alternative fail.

## How this lands: the ojas half is public; the tessl half is not

This change spans two repositories and is **not landable one at a time.** The
constraint was found by the CI/MSRV session on 2026-11-02 and is now **half
satisfied**, because ojas was pushed later that night.

- `tessl/src/safetensors.rs:238` calls `ojas_io::SafeTensors::from_bytes`, which
  this work added in `3e45c70`.
- **As of 2026-11-02, ojas's `origin/main` is `0300578` and `3e45c70` is an
  ancestor of it** (`git merge-base --is-ancestor` exits 0), so
  `git grep -c 'fn from_bytes' origin/main -- ojas-io/src` now matches three
  times in `ojas-io/src/safetensors.rs`. The API the adapter needs is public.
- tessl's `origin/main` is still `a00521f`, ten commits behind. So the half that
  remains unpushed is the adapter itself, which is the safe order.

An earlier version of this section said `origin/main` was `dac0b5b`, 29 commits
behind, and that `from_bytes` matched zero files there. That was true when
written and is not now; the paragraph is corrected rather than deleted, because
the three-step table below was measured against `dac0b5b` and only makes sense
with its base stated. Local `main` is four commits ahead of `origin/main`: three
of this work's records, and one `fix(metal)` commit (`28edc3d`) belonging to
another session.

Two details worth keeping straight, because they are easy to conflate:

- At tessl `main` (`07c2e23`) the CI workflow does not mention ojas **at all**,
  so no cargo job there can resolve tessl's manifest today, with or without this
  change — tessl's path dependencies need ojas on disk. That is the subject of a
  separate card, not of this one.
- The `OJAS_REF: ${{ inputs.ojas_ref || 'main' }}` default lives on that card's
  branch (`fix/ci-ojas-checkout-msrv-floor`, `4f51e89`:42), mirroring ojas's own
  `TESSL_REF`.

An earlier draft of this section said that branch landing is what turns tessl
red, which is wrong, and the distinction matters because the wrong version
reads as "the CI fix must wait for an ojas push". The sequence is three steps,
not two, and the middle one was measured by that card's session in paired
worktrees (tessl at `4f51e89` beside ojas at `dac0b5b`, clean, own target dir)
and then on the hosted runner:

| at GitHub `main` | ojas it resolves | result |
| :--- | :--- | :--- |
| tessl `main` today | none — the workflow does not mention ojas | no cargo job resolves the manifest at all |
| that branch alone | `dac0b5b` | resolves and checks clean: `cargo +1.97 check --locked --all-targets` exit 0; `fmt`, Go binding and MSRV green on run 41392007235 |
| that branch **plus this adapter** | `dac0b5b` | red on `from_bytes` |

The reason the middle row is green is that the branch's base is `a00521f`, where
tessl still has its own reader: `git grep -c ojas_io <rev> -- src` matches zero
files at `a00521f` and at `4f51e89`, and one — this adapter — at `07c2e23`.

**Landing order: ojas before, or with, the adapter.** Not before that CI branch,
which does not need it. That order was met by circumstance rather than by plan —
ojas was pushed by another session on the owner's say-so, carrying this work's
IO commits with it. This work pushed neither repository itself, and tessl's ten
commits are still local.

One thing to weigh before pushing ojas, found while checking the backlog: of the
19 commits in `b2dd967..main`, five are another session's — `7087da0` (`to_host`
on a mapped buffer actually waits for the device, which is `SharedBuf` and
`Timeline` territory, not a docs-only ride-along), `32e4642`, `c681f92`,
`8a3df56` and `0300578`, written 22:34–22:58 and interleaved with this work's.
Pushing ojas `main` publishes them too. An IO-only split is possible but is a
cherry-pick of this work's fourteen commits, not a range.

## Verified by running

| Check | Result |
| :--- | :--- |
| `cargo test --release --test host_access_ban` | **6 passed, 0 failed** |
| `cargo test --release --test safetensors` | **12 passed, 0 failed** |
| `cargo test --release --test reader_bounds` | **12 passed, 0 failed**, 1 ignored, file unedited |
| `cargo test --release --lib parser_mutation::` | **16 passed, 0 failed** |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `rustfmt --check` on the three files touched | clean |
| `cargo test --release --no-fail-fast` (whole suite) | **2,252 passed, 0 failed**, 56 ignored, 206 binaries |

`tests/safetensors.rs` kept its rejection semantics. Of the 52 malformed inputs
it covers, 42 keep their exact needle and 10 are the same refusal in different
words, each annotated in place with the needle it had; the sub-byte case gained
two inputs rather than losing any. No test was deleted or weakened to make the
adapter pass.

`tests/reader_bounds.rs` matters most of the five: it pins this crate's budget
wording in twenty `contains(..)` assertions — "payload budget exceeded", "physical memory",
"the caller's limit of 1073741824", "declares 16 tensor bytes", "read
allocation" — and it passes with no edit at all. That is the evidence that
tessl's payload policy stayed tessl's rather than being pushed down into
ojas-io.

The ignored test is `opens_the_qwen35_2b_checkpoint`, which needs the 4.5 GB
file from the Hugging Face cache; it was ignored before this change too.

### The whole suite, and what it took to run it

**`cargo test --release --no-fail-fast`: 2,252 passed, 0 failed, 56 ignored over
206 binaries, exit 0** — at `d42305f`, from a detached worktree beside the real
`../ojas` with its own `CARGO_TARGET_DIR`. The live checkout could not give this
number: another session holds it with `src/config_json.rs` and
`src/qwen35_adamw.rs` part-way through a `Json` → `JsonValue` rename that does
not compile, and with `src/safetensors.rs` deleted from the working tree. Those
files are not this change's and were not touched. The worktree asserts before
testing that `enum Dtype` and `impl TryFrom<StDtype>` are present in
`src/safetensors.rs`, so the run cannot silently be of someone else's tree.

Getting there took four attempts, and the three that produced no number are
worth recording because each failed differently: one `SIGKILL`ed mid-compile at
load 38 from other work; one built for six seconds and ran nothing, on that
rename; and one ran 74 of the 206 binaries and stopped, because `cargo test`
abandons the remaining binaries at the first failure unless told otherwise. The
final run passes `--no-fail-fast` for exactly that reason — 128 binaries in an
earlier run was where it aborted, not the size of this suite.

Three of this repository's own gates caught defects in this work, each by
running rather than by review, and each is fixed:
`host_access_ban::library_code_never_calls_a_panicking_host_accessor` on the
adapter calling `.read_u32(` in method-call form; `cargo fmt --all --check` on
the adapter being wrapped at 100 columns against this repo's 120; and
`docs_name_real_tools::every_bench_binary_and_path_the_docs_name_exists` on this
very record naming two of ojas-io's test files as though they were this crate's.

**What no CI run covers, here or anywhere.** `GPU suite on bare metal` and
`GEMM sweep on bare metal` are both gated on `vars.HAS_METAL_RUNNER == 'true'`,
which is not set, so they **skip** — identically in the baseline, so this is
pre-existing rather than caused by any change. No hosted run of this repository
exercises the Metal path at all, and a green badge should not be read as
covering it. The 2,252 above were run locally on the real GPU, which is
currently the only place that path runs.

Separately and not caused by this work: `cargo fmt --all --check` fails on ten
files this change never touched — `src/shared_buf.rs`, `src/loader.rs`,
`src/runtime/host_fence.rs`, `tests/capi.rs`, `tests/capi_contract.rs`,
`tests/capi_oversized.rs`, `tests/kernel_coverage.rs`, `tests/qwen35_params.rs`,
`tests/shared_runtime.rs` and `tests/stress_trim_concurrency.rs` — all wrapped
at 100 columns where `rustfmt.toml` says 120. The `fmt` job in
`.github/workflows/ci.yml` is therefore red on `main` independently of this
change. `src/safetensors.rs` was in that state too, for one commit, and is not
any more.

`widen_in_place`, which was byte-for-byte identical to ojas-io's, is deleted
with the parser. Its only test moved to `ojas-io/tests/half_float.rs` rather
than being deleted with the duplicate it covered.

---

## Milestone M2 Update (2026-11-02): Full Retirement of `src/{safetensors,json,npy,plain}.rs`

Following the P8 adapter phase above, Milestone M2 completed the IO and optimizer-scalar unification across `ojas` and `tessl`:

1. **`src/safetensors.rs` (517 lines), `src/json.rs` (516 lines), `src/npy.rs` (417 lines), and `src/plain.rs` (145 lines) are deleted (-1,595 LOC):**
   - `ojas-io/src/npy.rs` absorbed `.npy` v1.0/v2.0 reading and writing (`read_f32`, `read_i32`, `read_u32`, `write_f32`, `write_u32`, `from_bytes`, and `read_*_with_limit`).
   - `ojas-io/src/safetensors.rs` absorbed `SafeTensors::header_bytes`, `HeaderEntry` / `TensorEntry` accessors, and `Dtype` / `StDtype` re-exports.
   - `src/lib.rs` re-exports `pub use ojas_io::{npy, safetensors};` directly.
   - `src/config_json.rs` uses `ojas_io::json::{JsonValue, JsonLimits}`.
2. **AdamW scalar deduplication:**
   - `src/qwen35_adamw.rs` delegates step bias-correction and weight-decay scalar calculation to `ojas_core::AdamWScalars::new`.
