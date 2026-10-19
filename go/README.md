# EmbeddingGemma 2 from Go

`github.com/bharathvbcr/tessl/go/embedgemma2` embeds text, images, audio, video
(with or without timestamps), mixed items, and files or bytes on Apple-silicon
Metal, in-process, with no Python. It is a binding over tessl's EmbeddingGemma 2
C ABI ([docs/embedgemma2.md](../docs/embedgemma2.md)), carried across the
Go/Rust boundary by [gusset](https://github.com/bharathvbcr/gusset).

```go
m, err := embedgemma2.Load(snapshotDir, embedgemma2.Options{}) // exact f32; BF16 and TextOnly are options
if err != nil { /* a *embedgemma2.Error carries tessl's message */ }
defer m.Close()

q, _ := m.EmbedTexts(ctx, []string{"how do I speed up attention?"}, "query", 0)
d, _ := m.EmbedTexts(ctx, []string{"Sliding-window attention ..."}, "document", 256) // Matryoshka prefix

img, _ := m.EmbedImageFiles(ctx, []string{"cat.png"}, 0)
mix, _ := m.EmbedMixed(ctx, [][]embedgemma2.Segment{{
    embedgemma2.Text("a photo of"), embedgemma2.ImageFileSegment("cat.png"),
}}, 0)
```

Rows are L2-normalized `[]float32`, one per item, `EmbeddingDim()` wide (the
`truncateDim` prefix, renormalized, when it is not 0).

## Requirements and the one command

Apple silicon (darwin/arm64) with a Metal device; Go 1.26 or newer (built and
tested with 1.27.2); a Rust toolchain for the engine archive (1.97 or newer);
a snapshot directory of `google/embeddinggemma-2`. The package refuses to build
anywhere else with a message that says so.

```sh
make -C go unit      # no GPU: engine tests, gofmt, go vet, the Go unit tests
make -C go device    # fixtures written by tessl's C ABI, then the device tests
make -C go test      # both
make -C go real      # the real-checkpoint tests (needs the three variables below)
make -C go bench     # the overhead benchmark; rewrites bench/results/embedgemma2_go_m5pro.json
```

`make -C go test` builds the archive (`cargo build --release --locked
--manifest-path go/engine/Cargo.toml`), generates the fixtures, then runs
`go vet` and `go test -count=1 ./...` with `TESSL_GO_REQUIRE_DEVICE=1`, which
turns missing fixtures from a skip into a failure. `scripts/ci_local.sh` runs
`make -C go unit` always and `make -C go device` under the heavy-job lock. In
`.github/workflows/ci.yml` the hosted `go` job runs `make -C go unit` and the
self-hosted `gpu` job runs `make -C go device`.

Go's build cache keys a cgo package on the text of its flags and not on the
bytes of the archive they name, so the Makefile passes the archive's hash in
`CGO_CFLAGS`. A bare `go test` after rebuilding the archive should pass
`-count=1 -a`.

`ld: warning` lines about a duplicate `-lgusset` and about gusset's own
`target/release` search paths are gusset's cgo flags meeting the archive's, and
are harmless.

## How it is built

```
go/engine/            the umbrella staticlib, libgusset.a = gusset + tessl, and nothing else
go/embedgemma2/       the Go package
go/embedgemma2/internal/wire   the request and reply frames (no cgo)
go/embedgemma2/internal/npy    a hand-written .npy reader for the reference tests
go/cmd/bench_overhead the overhead benchmark
tests/go_fixtures.rs  writes the fixtures the device tests compare against
```

A Go binary links exactly one gusset archive, so the Go package cannot link
gusset's own and tessl's separately: the umbrella crate (`go/engine`) is both in
one `libgusset.a`, exporting `tessl_go_engine_init` to register its operations
on gusset's opcode dispatch. A Rust staticlib carries none of the system
libraries its crates link, so the cgo preamble names them (Metal, CoreGraphics,
AudioToolbox, ImageIO, AVFoundation, CoreMedia, CoreVideo, VideoToolbox,
Foundation, libobjc). `tests/go_link_frameworks.rs` fails when a framework a
`#[link(kind = "framework")]` under `src/` names is missing from that line.

**Dependency, and its pin.** The binding adds one dependency, gusset, at commit
`984209b542885b10b43e3d9aaf2073fd871e9eb9` (the pushed `origin/main`, the same
commit Jarvis pins): a git `rev` in `go/engine/Cargo.toml` and the pseudo-version
`v0.0.3-0.20261010191510-984209b54288` in `go/go.mod`, with no `replace`. Gusset's
`init()` checks that the Go module and the archive agree on the ABI, so a
mismatched pair panics at start instead of misbehaving. Bumping the pin is a
commit that changes what this repository builds against; move both together.

### One thread per model

tessl's handles are bound to the thread that made them. Each `Model` therefore
owns one gusset handle with exactly one worker thread (`WithPoolSize(1)`), and
the tessl runtime and model live in a `thread_local` on that worker: `Load` and
`Close` are operations that run there. A `Model` is safe for concurrent use.
Calls from many goroutines queue on gusset's semaphore in Go, so 64 goroutines
cost one worker thread and not 64 blocked ones. Two `Model`s run on two workers
and two Metal runtimes.

### Frames

Gusset moves bytes and an opcode; the schema is this package's. Scalars are
64-bit words; blobs are a u64 length, the bytes, and zero padding to 8; typed
arrays are blobs. A reply is a 16-byte header (status, flags, the Rust-side time
of the tessl call) followed by the payload or tessl's error message. Requests
up to 4096 bytes travel inline; larger ones in a Rust-owned buffer. The frame is
capped by gusset's 1 GiB buffer limit, below tessl's own 4 GiB encoded-video
cap, so a video that large goes by path. Both halves decode one committed byte
string (`go/embedgemma2/internal/wire/testdata`).

## Cancellation

What a context can and cannot stop (also on `Model`'s doc comment):

- A call **waiting for the worker** is cancelled at once and never reaches Rust.
  Its deadline is also handed to the engine, which refuses a job that expired in
  the queue before it starts.
- A call the worker **has begun** runs to completion: tessl's entries are one
  synchronous GPU wait with no cancellation point. The caller is released at its
  deadline all the same, returning `ctx.Err()`; the result is discarded and the
  worker's permit returns only when the work stops, so abandoned work never
  exceeds one call in flight per `Model`, and `Close` waits for it.
- The request's memory stays valid for the abandoned work: gusset holds the
  buffer it travelled in until the job ends.

`TestUnitQueuedCallLeavesAtItsDeadline...` and
`TestUnitRunningCallReleasesItsCaller...` prove both with an engine operation
that holds the worker for a fixed time with no cancellation point, as a tessl
call has none.

## Errors

| Error | Means |
|---|---|
| `*ArgError` (matches `ErrInvalidArgument`) | refused in Go before anything crossed: a count, length, range or encoding that cannot be right. `Field` names it (`images[2]`). |
| `*Error` | tessl's refusal. `Message` is tessl's text, verbatim; `Status` is its status; `Poisoned` is set when the model's runtime latched a fault. The model stays usable unless it is poisoned. |
| `ErrPanic` | a tessl panic caught at its boundary, or a Rust panic caught by gusset: the model is unusable; `Close` it and `Load` again. |
| `ErrPoisoned`, `ErrNoModel` | the model's runtime is poisoned, or the worker holds no usable model. |
| `ErrClosed` | a call after `Close`. |
| `ErrProtocol` | a frame this package cannot read: a bug or a stale archive, never a caller's input. |

`context.Canceled` and `context.DeadlineExceeded` come back wrapped, so
`errors.Is` finds them. Every length is bounds-checked against tessl's own
limits before it crosses (the exported `Max...` constants); tessl checks them
again.

## What it does not contain

Gusset contains a Rust `panic!`. It cannot contain `SIGSEGV`, `abort`, a corrupt
C library or a GPU driver fault, which take the Go process with them; gusset's own
guidance puts an engine that can fault in a separate process. tessl is
`unsafe` Metal code, so this binding is in-process by decision of the
repository's owner, not because the library recommends it for a GPU engine. A
Metal device loss or a driver fault ends the process. A tessl panic does not.

## Results

All on the tiny checkpoint written by `tests/go_fixtures.rs`, unless a real
checkpoint is named. Labelled as verified (a command in this repository printed
it).

- **Bit for bit with Rust.** The Go device tests compare every embedding with
  what tessl's C ABI returns for the same input when called from Rust
  (`tests/go_fixtures.rs`): texts under each prompt and truncation, tokenize,
  decode, raw images, audio and video (with and without timestamps), files and
  bytes, mixed items, the bf16 model, and soft-token counts.
- **Real checkpoint, against sentence-transformers.** `make -C go real` with
  `EMBEDGEMMA2_SNAPSHOT`, `EMBEDGEMMA2_REF_DIR` and `EMBEDGEMMA2_MM_REF_DIR`
  (opted in, a missing reference fails). Max abs error and minimum cosine over
  the rows of each case (bound: 1e-4 and 0.99999):

  | case | rows | max abs | min cosine |
  |---|---|---|---|
  | texts, by `prompt_name` | 8 | 2.384e-07 | 1.000000000 |
  | texts, Matryoshka 512 / 256 / 128 | 8 | 2.980e-07 / 2.980e-07 / 3.576e-07 | 1.000000000 |
  | images | 2 | 2.086e-07 | 1.000000000 |
  | audio | 2 | 2.347e-07 | 1.000000000 |
  | video 0 | 1 | 3.725e-07 | 1.000000000 |
  | video 0, timestamps | 1 | 2.086e-07 | 1.000000000 |
  | video 1 | 1 | 1.490e-07 | 1.000000000 |

  The tokenizer ids equal the reference's for all 8 texts (7949 ids), and the
  soft-token counts equal the processor's. bf16 stays within its floors against
  float32 (cosine at least 0.9999) and against the reference's own bf16 run
  (at least 0.9996): the worst measured are 0.999985 and 0.999684.
- **Threads.** 8, 64 and 256 goroutines issuing the same 384 calls peak at 9-10,
  9-10 and 15-16 OS threads (`gusset.Threads()`, the Go scheduler's count, as
  gusset's docs measure it) over two runs, against 6 idle; a blocking cgo call per
  goroutine would add one thread each. Every result equals the serial one bit for
  bit. The growth from 64 to 256 goroutines is the Go runtime's own (more
  runnable goroutines doing Go-side work), not a thread per caller.
- **Overhead** ([figure](../bench/results/embedgemma2_go_m5pro.svg),
  [table](../bench/results/embedgemma2_go_m5pro.txt)). The cost of the boundary,
  measured per call as the Go wall time minus the engine's own timing of the same
  tessl call (so the GPU time is identical on both sides), is 9 us for one text
  and 13 us for one image, growing with the bytes copied to 61 us for 256 texts
  (about 790 KB of embeddings back) and 43 us for 8 224x224 images (1.2 MB in); a
  round trip with no work (`Ping`) is 1.3 us at best. Against direct calls of
  0.38 ms to 7.9 ms that is 0.8% to 2.4% (Go wall over the direct time, each a
  minimum from a separate process, is 1.008 to 1.042). Recorded 2026-10-18 at
  eg2/integrate cccdffd on a quiet machine (1-minute load average 3.9 before, 5.6
  after; the run waited for a load below 6, and no machine-wide lock was held),
  9 rounds of 150 calls per arm. The direct arm runs on a thread of its own with
  gusset's worker stack, which is what the engine's model lives on; the same arm on
  the process's main thread is within 1.8% of it, either way. An earlier recording
  at a load of 70 to 99 put the boundary at 32 to 113 us.

## Not done

- Linux and Windows: tessl is Metal.
- Streaming results, and cancelling a tessl call mid-flight: tessl has neither.
