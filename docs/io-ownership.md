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

That parser is **deleted**. `src/safetensors.rs` is now a 567-line adapter over
`ojas_io::safetensors`, and across the three files touched the change is 355
insertions against 673 deletions: a net 318 lines out of tessl, and one of two
parsers for one format gone.

`ojas-io` was already a declared path dependency of this crate and was used
nowhere, so this needed no `Cargo.toml` change and no MSRV ask.

## What the adapter keeps, and why

The adapter exists because three things here are tessl's, not the reader's.
Everything else — the grammar, the schema, the tiling, dtype widths, positioned
reads, the file-identity re-check, the index bound, shard validation — is
ojas-io's and is not duplicated.

- **The surface.** `String` errors, `Vec<usize>` shapes, and the method names
  the loaders already call. All 47 files outside `src/safetensors.rs` that
  reference `SafeTensors`, `CheckpointView`, `Dtype` or `TensorInfo` compile
  untouched. That was the design constraint, not an outcome: if they had needed
  editing, the adapter would not have been faithful.
- **The dtype contract.** `Dtype` keeps its nineteen variants and no sub-byte
  float. ojas-io supports `F4`, `F6_E2M3` and `F6_E3M2` on a checked path,
  sizing them from their bit width and refusing a span that is not bit-exact;
  tessl declines them, because every path below this module is written in whole
  bytes. `TryFrom<StDtype> for Dtype` errors on all three, applied once per
  header at open, so the refusal is enforced by the type rather than
  documented. `pub type Dtype = StDtype` was rejected: it would put three
  variants in this namespace and break the exhaustive matches in
  `src/qwen35.rs` (:1774, :2779, :3518, :3610) and the 19-variant array in
  `tests/qwen35_mlx_q4_adversarial.rs`.
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
property the test was reaching for — but it is a real narrowing of what this
crate can load, and it is the one thing in this change that wants the owner's
eye.

## Verified by running

| Suite | Result |
| :--- | :--- |
| `cargo test --release --test safetensors` | 12 passed, 0 failed, 1 ignored |
| `cargo test --release --test reader_bounds` | 12 passed, 0 failed (file unedited) |
| `cargo test --release --lib parser_mutation::` | 16 passed, 0 failed |
| `cargo check --all-targets` | clean, no warnings |

`tests/safetensors.rs` kept its rejection semantics. Of the 52 malformed inputs
it covers, 42 keep their exact needle and 10 are the same refusal in different
words, each annotated in place with the needle it had; the sub-byte case gained
two inputs rather than losing any. No test was deleted or weakened to make the
adapter pass.

The ignored test is `opens_the_qwen35_2b_checkpoint`, which needs the 4.5 GB
file from the Hugging Face cache; it was ignored before this change too.

`widen_in_place`, which was byte-for-byte identical to ojas-io's, is deleted
with the parser. Its only test moved to `ojas-io/tests/half_float.rs` rather
than being deleted with the duplicate it covered.
