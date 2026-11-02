# Decision: ojas owns safetensors IO; tessl's reader is an adapter

Owner decision, 2026-11-01. The ojas side of this record is
[`ojas/docs/io-ownership.md`](https://github.com/bharathvbcr/ojas/blob/main/docs/io-ownership.md),
which carries the full evidence; this is the same decision stated from tessl's
side. It is the IO slice of the ownership split: ojas is the training framework
and owns checkpoints and IO, tessl is the Metal 4 extension and the Mac
inference engine.

## What changed here

`src/safetensors.rs` held a complete, strict `.safetensors` parser — 950 lines
of header grammar, offset validation, tiling checks, index parsing and
positioned reads. ojas-io had another one. The two were kept in step by hand.

That parser is **deleted**. `src/safetensors.rs` is now a **517-line** adapter
over `ojas_io::safetensors`, and across the three source files touched the
change is **295 insertions against 663 deletions — a net 368 lines out of
tessl**, and one of two parsers for one format gone. With this record and the
doc alignment, the whole change here is 407 insertions against 663 deletions
over seven files.

The tree as a whole did **not** shrink, and it is worth saying so rather than
reporting only this side of it: ojas-io grew by 1,396 insertions against 82
deletions to take the format over, of which about 385 lines are new tests —
`tests/tessl_rejection_corpus.rs` (246), `tests/header_amplification.rs` (101)
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

## How this lands: ojas first, or the two together

This change spans two repositories and **is not landable one at a time.**
Found by the CI/MSRV session on 2026-11-02 and verified here by name:

- `tessl/src/safetensors.rs:238` (at `07c2e23`) calls
  `ojas_io::SafeTensors::from_bytes`.
- `fn from_bytes` exists in `ojas-io/src/safetensors.rs` only on ojas's **local**
  `main` (`0300578`). It was added by this work, in `3e45c70`.
  `git grep -c 'fn from_bytes' origin/main -- ojas-io/src` matches zero files.
- ojas's `origin/main` is `dac0b5b`, **29 commits behind** local `main`. tessl's
  `origin/main` is `a00521f`, six commits behind.

So tessl's adapter cannot compile against the ojas that is on GitHub. The other
API it uses — `StDtype`, `MAX_HEADER_BYTES`, `TensorSource`, `IoError::detail`,
`read_u32` — all exist at `dac0b5b`; `from_bytes` is the one that does not.

Two details worth keeping straight, because they are easy to conflate:

- At tessl `main` (`07c2e23`) the CI workflow does not mention ojas **at all**,
  so no cargo job there can resolve tessl's manifest today, with or without this
  change — tessl's path dependencies need ojas on disk. That is the subject of a
  separate card, not of this one.
- The `OJAS_REF: ${{ inputs.ojas_ref || 'main' }}` default lives on that card's
  branch (`fix/ci-ojas-checkout-msrv-floor`, `4f51e89`), mirroring ojas's own
  `TESSL_REF`. Once it lands, tessl's CI resolves ojas at GitHub `main`. If
  ojas's commits are still unpushed then, tessl's first green-looking push goes
  red on a missing API — a failure that would look like the sibling-checkout fix
  and would not be.

**Landing order: push ojas before, or in the same operation as, tessl.** Neither
repository was pushed by this work.

## Verified by running

| Check | Result |
| :--- | :--- |
| `cargo test --release --test host_access_ban` | **6 passed, 0 failed** |
| `cargo test --release --test safetensors` | **12 passed, 0 failed** |
| `cargo test --release --test reader_bounds` | **12 passed, 0 failed**, 1 ignored, file unedited |
| `cargo test --release --lib parser_mutation::` | **16 passed, 0 failed** |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `rustfmt --check` on the three files touched | clean |
| `cargo test --release` (whole suite) | **not re-run — see below** |

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

### The one check still outstanding

**The whole suite has not been run green end to end, and is not claimed to
have been.** It ran complete once on this work: **1,435 passed, 1 failed, 36
ignored over 128 binaries**. The single failure was
`host_access_ban::library_code_never_calls_a_panicking_host_accessor`, which
caught the adapter calling `.read_u32(` in method-call form — a real defect,
found by this repo's own guard rather than by review, and fixed in the commit
after. That guard is green now, with the two other binaries the fix touched.

Two attempts to re-run the whole suite after the fix produced no result. The
first was `SIGKILL`ed mid-compile with the machine at load 38 from other work.
The second failed to build in six seconds, with no test run, because another
session holds this checkout and has `src/config_json.rs` and
`src/qwen35_adamw.rs` part-way through a `Json` → `JsonValue` rename that does
not compile yet; neither file belongs to this change and neither was touched.
So one clean end-to-end run of this suite, once that refactor lands, is the
remaining check.

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
