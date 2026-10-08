# Kernel hardening and performance evidence — 2026-08-30 baseline, 2026-09-04 addendum

**Verified scope:** GEMM dispatch and its tensor boundaries in `crates/tessl`
and `arch_02_value_resid/metal-native`, on Apple M5 Pro / 64 GiB / macOS 27.0
(26A5421a), Rust 1.96.0. This is a bounded hardening pass, not certification of the
full training/inference kernel catalogue or a general-purpose PyTorch replacement.

Raw measurements, pre-fix failure output, and source hashes are in
[`kernel_hardening_evidence.json`](kernel_hardening_evidence.json).

> **Evidence chronology.** The original 2026-08-30 audit is retained below as a
> dated record. Its test counts and isolated timings are not current-tree
> results. The 2026-09-04 addendum records the broader Tessl/Sparsl hardening
> verification and links to each component's canonical performance tables.

## Changes and contracts

- **Verified:** public GEMM entry points validate rank, nonzero dimensions,
  compatible inner/output shapes, dtype, backing-buffer bounds, alignment, runtime
  ownership, signed 32-bit matrix indexing, and non-overlapping output before
  casts, scratch allocation, or GPU encoding. Errors use the existing `Result`
  interface. Disjoint views in the same allocation remain valid; input-input
  aliasing is allowed. See `src/gemm.rs::validate_gemm` in both crates.
- **Verified:** casts validate source and destination metadata, shape, dtype,
  runtime and overlap. Counts must be in `1..=u32::MAX`. No host readback was added
  to the launch path.
- **Verified:** allocation and view sizes use checked arithmetic. Invalid `view`
  calls retain the existing panic contract, including release builds; this is not
  a new fallible public view API. Views remain relative to the source offset and
  bounded by the backing allocation, rather than the parent view's logical span.
- **Verified:** CPU f32→BF16 preserves NaNs (quieting their payload) instead of
  converting small payloads to infinity or overflowing on negative payloads.
  Signed zero, infinities, nearest-even ties, and all finite BF16 round trips are
  tested. This does not impose identical NaN payloads on Metal casts.
- **Verified:** simdgroup GEMM supports odd dimensions using a separate guarded
  scratch-tile kernel. The aligned fast kernel remains unchanged. Both overwrite
  their complete logical output, so the redundant zero dispatch and its associated
  synchronization were removed. Offset guards and NaN-poisoned output verify this.
- **Verified:** GEMM's test error metric now rejects unequal lengths and nonfinite
  values instead of letting `zip` or `f32::max` hide them. Other modules' metrics
  have not been comprehensively audited.

**Preserved invariants:** row-major contiguous views, Metal 4-only encoding,
f32 output/accumulation, existing precision/accumulation opt-ins, TensorOps zeroing
where required, and existing synchronization/residency policy. No dependencies,
auth/access controls, manifests, credentials, or frozen reference ports were
changed by this pass. Checked sizes improve memory safety but do not make the
whole runtime memory-safe.

The source-only hardening diff captured before the later concurrent NN tuning
was **1,011 lines added / 216 removed** across eight files; most additions are
regression/adversarial tests mirrored in both implementations.

Training and inference currently have deliberately separate copies of these
modules. The fixes cover both; consolidating the runtimes is a separate migration
with caller, residency, synchronization, and training-quality gates.

## Final audit update

The earlier BF16 poisoned-output failure was reproduced and then passed after
the concurrent TensorOps edit; it is not reported as a current failure. The
runtime and training copies now both use an exclusive `HostMapping` guard,
runtime access serialization, poisoned-command-buffer handling, checked
allocation/view/copy geometry, retained bump slabs, and fail-closed bind and
constant-arena validation. The parity and Gemma activation reports reject empty,
unequal, non-finite, or invalid-tolerance evidence. A 128M checkpoint gate also
uses the same finite evidence comparator.

The build scripts emit a unique immutable library path per build. The original
hardening pass also published a legacy crate-root `default.metallib` by atomic
rename, which prevented inode truncation but still mutated dependency source
trees and allowed competing builds to replace the shared name. Tessl now keeps
all generated libraries in `OUT_DIR`, exposes the path as
`DEP_TESSL_METALLIB`, and requires an explicit absolute
`TESSL_PREBUILT_METALLIB` for offline reuse. This closes both the original live
inode failure and the remaining registry/vendor source-mutation race.

### 2026-09-04 whole-kernel audit addendum

The current audit broadened the scope from GEMM boundaries to Tessl's runtime,
attention, KV cache, DecodeICB, quantized/f16 GEMM, reductions and build
publication contracts, plus Sparsl's nine sparse/scan operation families. These
are current verification results, not evidence that one Apple GPU generalizes
to every device or OS:

| Current check | Observed result |
|---|---|
| Tessl release, all features and all targets, Metal API/GPU validation | **372 passed, 1 ignored**; the ignored deep GEMM fuzz lane was run separately |
| Tessl deep GEMM fuzz | **7,500 adversarial cases passed** |
| Tessl doctests | **9/9 passed** |
| Tessl benchmark kernel inventory | **147/147 entry points dispatched** |
| Sparsl physical-Metal randomized differential campaign | **50,000 operations passed** across nine operation families |
| Sparsl full Metal suite | **189 top-level tests passed**, plus **3** marker-verified isolated watchdog regressions |
| Sparsl CPU and doctest suites | **127 + 1 passed** |
| Arch02 TensorOps online-attention resource/parity probe | **25,088 B static + 0 B dynamic / 32,768 B limit**; T=256 max-abs **1.117587e-8**; ragged T=65 **7.451e-9**; 0 non-finite |

The Arch02 probe removed redundant Q/K/V staging and aliased disjoint S/P
lifetimes: explicit source storage fell from **28,928 to 12,544 bytes
(-56.6%)**. It remains opt-in and D=32-only with a supported fallback for other
head dimensions. This is a resource/correctness result, not a latency or
throughput claim; it stays off the hot path pending paired timing, backward
support, shape goldens, and Instruments NAX evidence.

#### Performance verdict

No checked-in current-tree timing artifact satisfies Tessl's new publication
schema with `status: published`, and Sparsl retains no raw output from a fresh
hardened timing run. The numbers below are therefore historical observations or
an explicitly identified direct optimization delta, not a universal routing
policy:

| Area | Recorded result | Evidence status |
|---|---:|---|
| Tessl exact-f32 GEMM, rep B | **1.045× torch / 0.999× MLX** throughput | Historical; parity; spread gate failed |
| Tessl relaxed-tf32 GEMM, rep B | **2.145× torch / 1.998× MLX f32** throughput | Historical; large but not precision-matched; spread gate failed |
| Tessl bf16 GEMM, rep B | **0.979× torch / 2.631× MLX** throughput | Historical; parity/slightly behind the stronger torch baseline; spread gate failed |
| Tessl routed attention, rep C | **0.624× torch / 1.150× MLX** latency | Historical; ~1.60× faster than torch, 15% slower than MLX; spread gate failed |
| Tessl kernel-amortized decode, three reps | **0.22× torch / 0.91–0.97× MLX** latency | Historical diagnostic; ~4.5× faster than torch and ~3–9% lower latency than MLX |
| Sparsl resident weights, 20M nonzeros | **3.03 → 1.01 ms** (~3.00×) | Retained historical measurement |
| Sparsl Metal SpMM vs repeated Metal SpMV | **8.4–22.5×** | Retained historical same-backend measurement |
| Sparsl packed spikes at the largest vector | **1.30–1.48×** | Retained historical measurement; no small-vector win |
| Sparsl scan reformat removal at 4.2M states | **13.53 → 8.99–10.55 ms** (1.28–1.50×) | Direct optimization delta; Metal still slower than 6.47 ms CPU sequential |

The canonical detailed tables and ratio conventions are in the
[Tessl performance summary](../../README.md#latest-checked-in-result-snapshot)
and [Sparsl performance section](../../../sparsl/README.md). A
current claim requires an idle-host, balanced paired rerun that passes the
spread and provenance gates; correctness/stress success is not substituted for
that timing evidence.

## Historical 2026-08-30 verification

Nine new regression tests failed against their pre-fix implementation, including:
`public Result API panicked`, alias accepted, wrong transpose dtype accepted,
`NaN payload 7f800001 became finite or infinity`, allocation overflow accepted,
and a release-mode view offset that did not panic.

| Check | Observed result |
|---|---|
| Hardened shared-runtime release library | 63 passed, 1 ignored manual benchmark |
| Hardened training release library | 95 passed |
| Gemma consumer release library | 134 passed; includes synthetic inference tests, not pretrained-model quality |
| Five serial runtime/native repetitions | 790 passing test executions; no skip or source-drift signal |
| Metal shader validation enabled | Runtime 63/1 ignored and native 95 passed |
| Build publication contract | 9/9 adversarial cases passed across 3 build scripts |
| 128M checkpoint/resume gate | Passed; 128,367,988 parameters, 1,679 dispatches |
| GEMM stress | 2,880 sampled cases per repeated runtime/native suite |
| Runtime debug tensor contracts | 3 passed |
| GEMM contracts with opt-in accumulation | Shared runtime 8 passed; training 8 passed |
| Metal shader validation, abort on fault | 8 GEMM contract tests passed; log confirms `Metal GPU Validation Enabled` |
| Manual paired timing test | Passed in three separate runs |

Test logs were inspected for skip messages; no runtime capability skips
were reported in these runs. The manual timing test is intentionally ignored in
normal test runs and was explicitly executed separately. Shader validation covers
the GEMM contract tests, not every shader. One initial validation command ran from
the repository root and failed to locate `Cargo.toml`; it was corrected to use
`--manifest-path` before reporting the suite result.

The expanded transpose test covers 64 combinations: four shapes (including
`17×31×2049` for split-K tails), two backends, two precision policies, and TN/NT
with overwrite/accumulate. Simdgroup under BF16 *policy* uses its f32 fallback;
it does not become a BF16 shader. This is explicit sampled coverage, not an
exhaustive shape/dtype/hardware proof.

Run from the repository root:

```sh
cargo test --release --manifest-path Cargo.toml --lib -- --test-threads=1 --nocapture
cargo test --release --manifest-path ../ojas/arch02/metal-native/Cargo.toml --lib -- --test-threads=1 --nocapture
cargo test --release --manifest-path ../ojas/gemma-metal/Cargo.toml --lib -- --test-threads=1 --nocapture
METAL_RUNTIME_GEMM_ACCUM=1 cargo test --release --manifest-path Cargo.toml --lib gemm::contract_tests -- --test-threads=1
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1 cargo test --release --manifest-path Cargo.toml --lib gemm::contract_tests -- --test-threads=1 --nocapture
cargo test --release --manifest-path Cargo.toml --lib benchmark_simdgroup_zero_cost -- --ignored --test-threads=1 --nocapture
```

The 2026-08-30 Gemma tests wrote benchmark JSON files. That run's generated
files were archived under `/tmp/mlsystems-gemma-test-artifacts`; pre-existing
tracked reports were restored to avoid bundling unrelated test artifacts.
Current tests do not replace tracked latest-slot files unless explicitly opted
in, and test output remains `diagnostic_only`; retained snapshots without a
fresh gated run are `historical_unverified`. See
[`gemma-metal/docs/dev.md`](../../../ojas/gemma-metal/docs/dev.md#benchmark-artifact-writes).

## Historical 2026-08-30 isolated performance measurement

**Verified isolated measurement:** alternating paired measurements compare
`zero_f32 + current simdgroup GEMM` against `current simdgroup GEMM` on the same
build, synchronizing every call. There are 10 warmup pairs and 100 measured pairs
per shape per process, repeated in three processes. This isolates the removed
clear; it is not a comparison of entire old/new framework binaries.

| Square dimension | Clear + GEMM median, µs (three runs) | GEMM-only median, µs | Speedup range |
|---|---|---|---|
| 16 | 449.000 / 466.417 / 491.750 | 221.938 / 230.938 / 238.291 | 2.02–2.06× |
| 128 | 402.188 / 399.688 / 423.896 | 191.146 / 199.084 / 202.584 | 2.01–2.10× |
| 512 | 500.625 / 457.625 / 506.396 | 267.917 / 259.250 / 265.417 | 1.76–1.91× |

**Verified structural change:** aligned simdgroup calls now require one binder
instead of two. The test asserts that count and verifies output plus untouched
input bank views.

A separate eight-shape × three-lane sweep (24 cases, 3 warmups, 9 samples) was
captured before/after. It is noisy, including variation in unchanged TensorOps
lanes; do not infer a general TensorOps speedup from it. No end-to-end training
speed or quality claim is made. The odd-shape path is correctness-oriented and
has not been competitively tuned.

## Publication gates after the 2026-09-04 addendum

1. **Verified package gate:** Tessl and Sparsl now carry repository, license,
   Rust-version and package metadata, and their local package verification
   passes. Gemma and the Arch02 trainer remain `publish = false`. Tessl's public
   source breaks require an explicit next-version/API decision before release.
   AOT builds require the Metal 4 toolchain; `*_SKIP_AOT` remains an offline
   escape hatch and must name a verified prebuilt artifact rather than silently
   selecting a stale one.
2. **Verified API boundary:** the high-level tensor/GEMM/copy paths are guarded,
   but the public low-level Metal binder and raw `MTLBuffer` access remain escape
   hatches. They accept shader-specific contracts that Rust cannot infer. Keep
   them out of a safe general-purpose package, or redesign them as explicitly
   unsafe APIs with typed kernel descriptors before publishing.
3. **Unverified hardware/shape universality:** the current Tessl benchmark
   inventory dispatches 147/147 shipped entry points, and the current suites
   cover runtime, GEMM, attention, KV, reductions, ICB and consumer paths. That
   does not prove every supported shape, allocation pressure, Apple GPU/OS
   version or raw low-level caller. Timeout/quarantine behavior is tested with
   watchdog-isolated regressions, but opaque driver hangs and additional
   hardware still need dedicated external gates.
4. **Unverified release quality:** these tests do not establish multi-seed
   training quality, sustained whole-model throughput, profiler evidence, or
   PyTorch-compatible API/semantics. The measured clear-removal speedup is a
   kernel-level result, not an end-to-end claim.

**Workspace provenance:** while this work ran, another process created commits
`7fc29d7` and `e445ce5`, including intermediate hardening edits, then continued
retuning native BF16 TensorOps NN. Those changes and commit history were preserved.
This task issued no commit, reset, push, or publication. Native TensorOps tuning
is not attributed to this pass; the raw timing sweep precedes that later tuning.

## Primary technical references checked

The edge kernel uses the SIMD-group load/store and barrier semantics in Apple's
[Metal Shading Language specification, §§6.8–6.9](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
The existing TensorOps execution-scope and tiling constraints were checked against
[Apple's MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
Validation environment variables follow
[Apple's shader validation documentation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage).
NaN bit handling follows the documented
[Rust f32 representation](https://doc.rust-lang.org/stable/core/primitive.f32.html).
