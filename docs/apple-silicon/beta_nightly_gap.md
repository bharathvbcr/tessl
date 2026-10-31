# Beta, nightly, and coverage gaps (2026-10-30)

Closes the pass after [`apple_ml_stack.md`](apple_ml_stack.md) and
[`mlx_audit.md`](mlx_audit.md). Three read-only agents re-opened the Apple
beta pages, the MLX/mlx-lm/mlx-swift/PyPI pins, and the tessl docs against
`src/`. This session also re-ran `sw_vers` and one `supportsFamily` query.
Nothing was installed. No kernel, gate time, or worktree source was edited.

Labels: **measured** is this machine this day. **agent** is a page that
agent opened and quoted. **source** is a tessl file line read here.

## Nothing newer than the 27.2 GM

**agent.** Releases feed
[releases.rss](https://developer.apple.com/news/releases/rss/releases.rss),
`lastBuildDate` Mon, 26 Oct 2026 10:00:00 PDT. Items on or after 15 Oct 2026:

| Item | Build | Date |
|------|-------|------|
| macOS 27.2 | 26B123 | 2026-10-26 |
| iOS 27.2 | 24B91020 and 24B119 | 2026-10-26 |
| iPadOS 27.2 | 24B119 | 2026-10-26 |
| tvOS 27.2 | 24K122 | 2026-10-26 |
| visionOS 27.2 | 24N120 | 2026-10-26 |
| watchOS 27.2 | 24S118 | 2026-10-26 |
| iPadOS 26.7.2 | 23H128 | 2026-10-26 |
| iOS / iPadOS 18.7.11 | 22H384 | 2026-10-26 |
| TestFlight 4.4.1 | | 2026-10-22 |
| App Store Connect | | 2026-10-20 |
| Xcode 27.2 RC | 27B48 | 2026-10-19 |
| Xcode 27.1 | 27A9275 | 2026-10-15 |

No 27.2.1, no 27.3, no later seed. The Xcode 27.3 and macOS 27.3 notes URLs,
and the 27.2.1 URLs, returned the site title only. The updates article’s top
heading is still “The 27 platform releases – June 2026”.

**measured.** Installed OS is macOS 27.0.1, build `26A434`. 27.2 stays
offered and not installed.

## 27.2 notes have no ML section

**agent.** [Xcode 27.2 notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27_2-release-notes.md):
“Xcode 27.2 includes Swift 6.4 and SDKs for iOS 27.2, iPadOS 27.2, tvOS 27.2,
watchOS 27.2, macOS 27.2, and visionOS 27.2.” No Metal, Core ML, Core AI,
Accelerate, shader, or GPU section.

Known issue 187160501, quoted:

> macOS, watchOS, tvOS, and visionOS SDKs in Xcode 27.2 incorrectly report 27.1 as a valid deployment target. Builds and other functionality might have unexpected behavior if this deployment target is used. Mac Catalyst builds with a 27.1 or 27.2 deployment target might be unable to use newly introduced API.

**source.** Tessl’s Metal compile passes `-mmacosx-version-min=26.0`
(`build.rs:487`). That flag is not 27.1.

**agent.** [macOS 27.2 notes](https://developer.apple.com/documentation/macos-release-notes/macos-27_2-release-notes.md)
headings: Overview, General, AppKit, Display Settings, Notifications,
Shortcuts, StoreKit Testing in Xcode. Metal, Core ML, Core AI, and Accelerate
do not appear.

## This GPU, Apple11, and MSL 4.1

**measured.** Device “Apple M5 Pro”, architecture `applegpu_g17s`.
`apple10` true, `apple11` false, `metal4` true.
`recommendedMaxWorkingSetSize` = 51539607552 (48 GiB).

**agent.** metal-cpp commit `8cdde68` (2026-09-22, “metal-cpp for macOS27.0.1,
iOS27.0.1”) is still the tip. No commit after 2026-09-22. `MTLLibrary.hpp`
at that commit:

- `LanguageVersion4_0 = 262144`
- `LanguageVersion4_1 = 262145` which is `(4 << 16) + 1`

Changelog row for that drop: “Add the Metal APIs to support the Apple11 GPU
family.” Those wrappers are for a family this GPU does not report.

**agent.** Symbol pages with empty behavior text, all macOS 27.0.0 unless
noted. A later fetch cannot invent what Apple left blank:

| Symbol | What the page contains |
|--------|------------------------|
| `optimizeForPersistentKernel` | `Bool` getter/setter only |
| `MTLContentionRelief` | cases `automatic`, `none` |
| `MTLForwardProgressUsage` | cases `automatic`, `simdGroupParallel`, `weak` |
| `MTLGPUFamily.apple10`, `.apple11` | case name only. The printed availability is the parent enum’s floor (macOS 10.15), which is not an introduction date |
| `MTLLanguageVersion.version4_1` | case name only, availability macOS 27.0.0 |
| `MTLLanguageVersion.version4_0` | case name only, availability macOS 26.0.0 |

`MTL4MachineLearningCommandEncoder` (macOS 26.0.0) does have a sentence:
“Encodes machine learning model inference commands for a single pass.”

**source.** `build.rs` asks for `-std=metal4.0` and falls back to
`-std=metal3.2` only when that compile fails and `TESSL_METAL_WERROR` is
unset. TensorOps sources do not take the fallback. There is no `metal4.1`
flag. MLX’s JIT on macOS 27 asks for the 4.1 encoding; the shipped tessl
metallib does not.

## Dtypes and graph APIs

**agent.** [`metalFloat8e5m2`](https://developer.apple.com/documentation/metal/mtltensordatatype/metalfloat8e5m2.md),
macOS 27.0.0: “An 8-bit floating point data type with 5 exponent bits, 2
mantissa bits, and 1 sign bit.” No kernel recipe on that page.

**agent.** `MPSGraphSDPADescriptor` is macOS 27.0.0. The page’s first sentence
is “A descriptor that configures a scaled dot product attention (SDPA)
operation.” It is a graph op. BNNSGraph’s symbol page is macOS 15 / iOS 18.
The Accelerate updates article’s newest section is still June 2024. vImage
has no 2026 section. The Core ML updates article’s newest section is still
June 2024. None of those change the training GEMM.

## Names that collide

**agent.** Dynamic Caching, Apple newsroom
[2023-10-30](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/):
“allocates the use of local memory in hardware in real time.” That is GPU
local memory. A transformer KV cache is model state. Foundation Models
Dynamic Profiles are a third API, already named in `apple_ml_stack.md`.
No later newsroom page was opened to see if Apple reused the phrase.

**source.** `MTLCaptureManager` does not appear under `docs/apple-silicon/`
or as a symbol this search found in `src/`. “Capture” in `src/decode_icb.rs`
is a frozen argument-table replay of a recorded command. Xcode GPU capture
is a debugger. The two are different tools.

Shader validation and Metal API validation are different Xcode debug layers.
The macOS 26 notes mention shader validation against MPP (149263281, listed
fixed there). The Xcode 27 notes mention Metal API validation failing a Core
AI path. Neither layer is a kernel change. The docs name both and do not
compare the two switches.

## Nightly pins

Compare counts are the GitHub API `ahead_by` field. The commit arrays in
those payloads were truncated, so the numbers were not re-counted by hand.
**agent**, 2026-10-30.

| Tree | Latest tag | `main` vs that tag |
|------|------------|--------------------|
| mlx | v0.32.4, `36969c0`, 2026-10-14 | `1ba7f969`, `ahead_by` 58. First commit in the range is an untagged patch bump to 0.32.5 (`#4707`). PyPI is still 0.32.4 |
| mlx-lm | v0.32.2, `93e6d4a`, 2026-10-16 | `736ab0d8`, `ahead_by` 17. That tip only edits a workflow file |
| mlx-swift | 0.33.0, published 2026-10-27, vendors mlx `36969c0` | `ahead_by` 1 (`78d3cc1d`, JACCL trait). The mlx gitlink does not move |

`Package.swift` at mlx-swift 0.33.0 still defines `MLX_VERSION` as the string
`0.32.0`. The gitlink and the release body are mlx v0.32.4. Swift matches
the mlx release, not mlx `main`.

`mlx_lm/models/qwen3_5.py` last changed 2026-10-09 (`e7461f15`, parallel
vocab, +12/−0). That patch is a shard gather. It is after mlx-lm v0.32.0 and
it does not change the affine Q4 bytes the tessl loader pins.
`mlx_lm/utils.py` after v0.32.2 has one commit, `33fdd163` (nvfp4 global
scale). The 0.32.0 loader pin stays.

PyPI RSS the same day: mlx 0.32.4 (14 Oct 2026), mlx-lm 0.32.2 (16 Oct 2026),
coremltools dev upload 9.1.dev1 (3 Aug 2026) while the JSON stable version is
9.0 (10 Nov 2025), torch 2.15.0 (28 Oct 2026). The MLX install page
(labeled 0.32.4, and `main`’s `install.rst`) names PyPI extras
`mlx`, `mlx[cuda12]`, `mlx[cuda13]`, `mlx[cpu]`. It does not name a nightly
index.

**agent.** https://pytorch.org/get-started/locally/ does not contain the
string `2.15` or `MPS`. Its stable selector label on the fetched HTML is
2.7.0. The macOS section says PyTorch can be installed on macOS 10.15 or
above. Wheel filenames were not opened, so a separate MPS build is
unconfirmed. PyTorch MPS as a tessl baseline stays in
`docs/benchmarking.md` and `gemm_architecture.md`, outside this Apple/MLX
set.

## Doc lines corrected in this pass

These contradicted `src/` or a page opened above. Gate numbers were not
touched.

| Was | Now | Evidence |
|-----|-----|----------|
| `metal4_mpp.md` “~1 MiB const arena” | 16 MiB, `METAL4_CONST_ARENA_BYTES` | `runtime.rs:42`, allocated at `:2923` |
| `mlx.md` “`alloc_buffer_hot` dead / unused” | live at `runtime.rs:1842`; Audit 6 marks the wiring DONE | callers include `loader.rs:170`, `qwen35_model.rs:1097`, `qwen35_train.rs:240`, `qwen35_q4.rs:471`, `qwen35_adamw.rs:250` |
| `machine_profile` still-open row “unused” | row removed; B=32/64 cell says the allocator is wired | same |
| `optimization_map.md` machine blurb “macOS 26.5.2” | 27.0.1 (`26A434`) | `sw_vers` this session. The TTFT clause on that line was left as written |
| `metal4_mpp.md` “Hot set vs cold set” as two residency sets | one `MTLResidencySet`; Hot and Cold are `BufferKind` | `runtime.rs:90-95` and `:380` |
| `metal4_mpp.md` opening “dialect may still use metal3.2” | 4.0 first, 3.2 is the fallback | `build.rs:153-217` |
| `mlx_audit.md` “mlx-swift was not re-read” | 0.33.0 vendors `36969c0`; version string still says 0.32.0 | agent, above |
| `apple_ml_stack.md` “e5m2 not opened” | opened; one-sentence definition, macOS 27.0 | agent, above |

`requestResidency` runs at init (`runtime.rs:2932`) and inside
`flush_residency` only when `residency_dirty` is set (`:1574`, `:1600`).
There is no idle-interval thread in `src/`. MLX’s refresh environment
variable is still not something to copy.

## Final pass, same day

Three more read-only agents, plus `system_profiler` and `ioreg` on this Mac.
The releases feed, metal-cpp tip, and mlx `ahead_by` 58 / tip `1ba7f969` were
unchanged.

**measured.** GPU cores are 20. `system_profiler SPDisplaysDataType` says
“Total Number of Cores: 20”. `ioreg` `gpu-core-count` is 20. `MTLDevice` still
does not print that count.

**source.** The default wired budget is `0.9 * recommendedMaxWorkingSetSize`
(`runtime.rs:1000`). On this boot that is 46,385,646,796 bytes, 43.2 GiB of
the 48 GiB recommendation. `set_wired_fraction` clamps to 0.5–0.95.
`machine_profile_m5_pro.md` now says that. The old “~41–48 GB” band is retired.
The 51.8 GiB figure in the optimization map is the 2026-07-12 banner, labeled
as that day.

**agent.** PyTorch 2.15.0 (PyPI, 28 Oct 2026) publishes
`macosx_14_0_arm64` wheels. No filename contains `mps`. The
[v2.15.0 release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
says the MPS backend requires macOS 15 or newer, and that on macOS 14
`torch.backends.mps.is_available()` returns false. This host is macOS 27.0.1,
which is inside that requirement. The wheel tag and the MPS OS floor are
different claims. No MPS kernel was run.

**agent.** Dynamic Caching on the M5 (2025-10-15) and M5 Pro (2026-03-03)
newsroom pages is lowercase “second-generation dynamic caching”, tied to the
shader core and graphics. The M6 / M5 Ultra page (2026-08-25) uses the
capitalised name for geometry rates, shader cores, and frame rates in games.
Same graphics feature as the 2023 M3 note. It is still GPU local memory, not
a KV cache.

**agent.** mlx-swift 0.33.0’s `Source/Cmlx/metal-cpp` is an in-tree copy, not
a submodule, and not apple/metal-cpp `8cdde68`. Both headers set
`LanguageVersion4_0 = 262144`. `LanguageVersion4_1 = 262145` is only in the
upstream header. The vendored README’s newest changelog row is macOS 26 /
Apple10. Upstream’s newest row is macOS 27.0.1 / Apple11. The comparison is
those two headers and the two README rows, not a file-by-file tree hash.
Tessl does not compile that copy. `build.rs` still asks for metal 4.0.

The profile’s “still open” list now matches the Audit 6 table: P1a–P1c DONE,
P1d REJECT as default, Hot DONE. `mlx.md`’s steal list keeps the original
estimates and points at that table for status. One-liners no longer say to
turn `multiply_accumulate` or the hazard-aware barrier on.

## Left open on purpose

- The Metal symbol pages in the table above still have no behavior text.
- M5 Max and M5 Ultra architecture strings were not read off a device.
- The 58.4 ms / 276 binder gate was not run.
- FineWeb B=32/64 under the 0.9 wired budget is still the open measurement
  on the optimization map. It was not started here. The board card is
  `gp-fineweb-b32-b64-rebench`
  (`ft-9a03a5d359aef49eaaf922f38badefe1`), backlog.
- mlx-swift’s vendored metal-cpp was not diffed file by file past
  `MTLLibrary.hpp` and the README changelog.
