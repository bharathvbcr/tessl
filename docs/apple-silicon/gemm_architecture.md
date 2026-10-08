# tessl GEMM: kernel selection, the coop gate, and how to check it

Covers the TensorOps (MPP `matmul2d`) GEMM path, which lives in `tessl`
(`crates/tessl`) and has exactly one copy.

`tessl-arch02` (`arch_02_value_resid/metal-native`) used to carry a byte-identical
fork of the kernels and of `gemm.rs` / `runtime.rs` / `tensor.rs` / `dispatch.rs`,
kept in step by a static audit. It now depends on `tessl` and compiles tessl's
kernel sources through `DEP_TESSL_KERNELS`, so there is nothing left to drift.

## Kernel selection

Every NN TensorOps shape resolves through one function, `tensorops_nn_kernel`
in `src/gemm.rs`. It is the single evaluation site for the cooperative gate, and
it is `pub(crate)` so tests can ask which path a shape takes rather than assume.

| dtype | gate | kernel |
|---|---|---|
| bf16 | `use_coop_nn(TILE_BF16_NN, …)` | `matmul2d_tensorops_bf16_f32_coop` |
| bf16 | otherwise | `matmul2d_tensorops_bf16_f32` |
| f32 relaxed (`--tf32`) | `use_coop_nn(TILE_F32R_NN, …)` | `matmul2d_tensorops_f32_relaxed_coop` |
| f32 relaxed | otherwise | `matmul2d_tensorops_f32_relaxed` |
| f32 exact | — | `matmul2d_tensorops_f32` |

TN, NT and the accumulating TN/NT paths are selected separately and have no
coop variant — see "Why TN/NT have no coop kernel" below.

## The two kernel shapes

**Blocked** (`matmul2d_tensorops_bf16_f32`, `…_f32_relaxed`): loops over K in
BK=256 blocks, accumulating into a **device-memory** C tile. Block 0 uses
`mode::multiply` so it seeds C and no host-side pre-zero is needed; later blocks
use `multiply_accumulate`. Handles ragged edges and any K.

**Cooperative** (`…_coop`): holds the C accumulator in registers
(`get_destination_cooperative_tensor`) across the whole K loop and stores once,
so C traffic is independent of K. Has **no ragged, short-K or tail branch** — it
trusts the host gate completely.

That difference is the whole point. In the blocked kernel C traffic scales with
K/BK while useful work scales with K, so throughput falls as K grows: measured
at M=N=4096 it peaked at K=2048 and then declined, reaching 0.79x of PyTorch MPS
at K=8192. The coop kernel removes that term.

## `use_coop_nn` — the only guard those kernels have

```rust
m >= tile.sm && m % tile.sm == 0
    && n >= tile.sn && n % tile.sn == 0
    && k >= COOP_MIN_K && k % COOP_BKC == 0
```

Each clause maps to something the kernel cannot do for itself:

- `m >= tile.sm`, `n >= tile.sn` — divisibility alone also admits `m == 0`.
- `% tile == 0` — every tile must be interior; there is no edge path.
- `k % COOP_BKC == 0` — the loop is `for k = 0; k + BKC <= K; k += BKC`. A K
  that is not a whole number of blocks silently drops the tail.
- `k >= COOP_MIN_K` (512) — **structural, not tuned.** The blocked kernels use
  BK=256, so below K=512 they run at most one full block plus a tail, which is
  already a single C store. Measurement agrees: at K=256 coop is 0.90x (bf16)
  and 0.93x (relaxed); it crosses over from K=512.

`COOP_BKC` is 128 for both kernels. Note it is deliberately *smaller* than the
blocked BK=256: 128 divides every K the presets produce (768, 1152, 2304, 3072,
4096) whereas 256 does not divide 1152. Paired measurement showed BK=256 and
BK=128 within noise of each other on the shapes both cover, so the one with
wider coverage wins.

The two NN tiles differ — `TILE_BF16_NN` is 64x64, `TILE_F32R_NN` is 128x64 —
so a shape can be gate-eligible for bf16 and not for relaxed. M=192 is the
canonical example and is covered by both the boundary test and the shape sweep.

## Why TN/NT have no coop kernel

They issue a **single full-K `matmul2d`**, so there is no host-visible C
round-trip to remove. Two controls establish this rather than assuming it:

- An explicit device-C K-blocked control (`mm_tnblk_*`, `mm_ntblk_*`) *regresses*
  to 0.34–0.86x of production.
- Every coop TN/NT variant is **bit-identical** to production (max_rel_err
  exactly `0.00e0`), while the device-C control differs by ~1.2e-6.

Identical bits mean MPP is already accumulating in registers inside its own K
loop; the device-C control differs precisely because it rounds to f32 once per
block. Coop variants measured 0.77–1.10x — no win. Same for the accumulating
TN/NT kernels: 0.88–1.12x against the real accumulate kernel.

(Measured against the *default* build the accumulate coop looked like 1.19–2.74x,
but `METAL_NATIVE_GEMM_ACCUM` defaults off, so that baseline was the
temp-buffer + `add_inplace` fallback. That fallback really is 1.19–2.74x slower
than the accumulate kernel, but enabling it is a training-quality decision
already made on numerics grounds.)

## Checking it

```bash
# Static: every Rust TileGeom vs the kernel's constexpr SM/SN and simdgroup
# count. Resolves paths from its own
# location, so it runs from any directory and survives `cargo package`. It also
# fails if tessl-arch02 grows a local copy of the kernels again.
python3 scripts/audit_gemm_tiles.py
```

```bash
# Runtime: the fixed 160-case seeded fuzz. It validates every sampled result but
# does not claim per-kernel coverage; use `bench/kernel_coverage.py --check` for
# the separate dispatch-coverage census.
# from the tessl root
cargo test --release --lib -- --test-threads=1 --nocapture gemm_fuzz_quick
```

```bash
# Deep soak: 2,500 fixed cases. STRESS_SEED accepts a non-zero decimal or
# `0x`-prefixed u64 and fails loudly on malformed, zero, or overflowing input.
STRESS_SEED=0xdeadbeef \
  cargo test --release --lib -- --ignored --test-threads=1 --nocapture gemm_fuzz_deep
```

```bash
# Cross-runtime numeric parity over the whole ladder. A job of its own, not a
# side effect of the timing sweep: every lane, every operand draw, scored
# against a per-draw f64 reference alongside MLX and torch. Covers f32 exact,
# tf32-relaxed and bf16 -- the dump was once filtered to `-f32` lane names,
# which left the two reduced-precision lanes unscored while the report still
# printed clean.
#
# The driver walks one label at a time (dump -> score -> delete). The ladder at
# eight draws is ~12 GB of .npy if dumped at once; streaming caps peak disk at
# one shape, ~3.5 GB at the widest.
# from the tessl root
python3 bench/parity_ladder.py --dists all --seeds 4 \
    --out bench/results/gemm_parity_grid_m5pro.json
```

```bash
# One shape, if that is all you need. The driver above is this, in a loop.
BENCH_PARITY_SHAPE=mlp_down BENCH_PARITY_DIST=heavy_tail BENCH_PARITY_SEEDS=8 \
  cargo run --release --bin bench_gemm_sweep -- --dump-parity /tmp/parity
python3 bench/gemm_sweep_mlx.py --parity-dir /tmp/parity
```

```bash
# Host-only harness: fabricated adversarial dumps, grid-merge cases, budget
# verdicts and static contracts. GPU-backed CLI sections are visibly skipped.
python3 bench/test_parity_harness.py --pure

# Physical-GPU gate: either CLI section being unavailable is a failing exit.
python3 bench/test_parity_harness.py --require-gpu
# Both modes emit a machine-readable HARNESS_SUMMARY. A non-required skip is
# `PASS WITH SKIPS`, never the same verdict as complete coverage.
```

Measured on M5 Pro over an **8 shape x 5 distribution x 4 seed grid**
(`bench/results/gemm_parity_grid_m5pro.json`), 320 scored cells per lane. Three
numbers per lane, because they answer different questions:

| lane | out | normwise | worst element | budget used | worst cell |
|---|---|---|---|---|---|
| `tensorops-f32` | f32 | 5.109e-06 | 1.04e+02 | 0.079x | `mlp_up`/heavy_tail |
| `simdgroup-f32` / `mlx-f32` / `torch-mps-f32` | f32 | 5.109e-06 | 1.04e+02 | 0.079x | `mlp_up`/heavy_tail |
| `tensorops-tf32` | f32 | 1.517e-03 | 2.94e+04 | 0.238x | `mlp_up`/heavy_tail |
| `tensorops-bf16` | f32 | 5.469e-03 | 3.08e+05 | 0.936x | `mlp_up`/heavy_tail |
| `mlx-bf16` / `torch-mps-bf16` | bf16 | 7.145e-03 | 3.08e+05 | 0.921x | `mlp_up`/heavy_tail |

`normwise` is max|err| / max|ref| -- the only number this harness used to
report. It cannot see a per-element failure: where cancellation drives an output
near zero, a bf16 result is wrong by **3.08e+05 relative** while normwise reads
5.5e-03. `budget used` is the fraction of `(gamma_{K+8} + 2*u_in)*sum|a.b| +
u_out*|ref|` consumed -- the same per-element bound `tests/common/mod.rs`
asserts -- and it is what decides pass or fail. Above 1.0 the run aborts.

**The operand distribution is a benchmark input and it dominates.** Uniform
operands, all this harness used to run, are the easiest case. Budget consumed,
worst cell per distribution:

| lane | uniform | normal | log_uniform | near_cancel | heavy_tail | spread |
|---|---|---|---|---|---|---|
| `tensorops-f32` | 0.009x | 0.011x | 0.035x | 0.050x | 0.079x | 9.0x |
| `tensorops-tf32` | 0.026x | 0.032x | 0.114x | 0.096x | 0.238x | 9.2x |
| `tensorops-bf16` | 0.076x | 0.103x | 0.413x | 0.063x | **0.936x** | **14.9x** |

bf16 runs at 93.6% of its error budget under heavy-tailed operands and 7.6%
under uniform -- a single-distribution benchmark understated it ~15x. MLX and
torch bf16 hit 0.921x on the same cell, so this is bf16 arithmetic reaching its
theoretical bound, not a tessl defect. The two axes move oppositely: for f32,
larger shapes consume *less* budget (0.009x at 512^3 to 0.002x at 4096^3,
because the gamma_K bound grows faster than the realised error) while heavier
tails consume ~9x more. Sweeping one axis alone misleads in either direction.

`tensorops-f32` is bit-identical to torch-MPS and to the simdgroup fallback at
every shape measured, and to MLX at 1024/2048/4096; MLX diverges only at
`square_512`, where it is slightly more accurate (5.62e-07 vs 9.85e-07) -- a
different kernel path at small sizes.

Bit-inequality is not scored: reduced-precision lanes cannot match f32
bit-for-bit by construction, and two f32 lanes differ constantly from summation
order alone while sitting at identical distance from the reference.

Every stage fails closed. `bench_gemm_sweep` pre-zeroes C, refuses non-finite
operands or results, rejects `BENCH_SHAPES` alongside `--dump-parity` and an
unknown `BENCH_PARITY_SHAPE` / `BENCH_PARITY_DIST`, and writes its manifest last
and only on success. The scorer exits non-zero on a missing lane, seed or grid
cell, a shape mismatch, a non-finite value, a degenerate reference, a lane with
no declared unit roundoff, or any lane over budget -- and a budget breach names
whether tessl alone, the comparison runtime alone, or every runtime exceeded it,
because those call for opposite responses. The ladder driver enforces the same
whole-grid rule across cells and cross-checks the Rust `SHAPES` ladder and the
requested distribution against every dump, which is what caught a stale binary
silently dumping `square_1024` for every label during development.

The fuzz fails if any selectable NN kernel is chosen for under 1% of cases. That
assertion is load-bearing: the first version of the fuzz passed three injected
coop-kernel faults because independent per-dimension sampling reached the gate
in well under 1% of cases and it never dispatched those kernels at all.

## Benchmarking

The A/B rig (`crates/tessl/kernels/tune/`) is **not** in the default metallib. It
sits in a subdirectory so neither build script's directory glob can pick it up by
accident. Measured with `xcrun metal-nm` on 2026-09-03: the default library holds
**147 entry points at 1.13 MB**, and the tuning build adds **34** `mm_bf16_*`
variants for **181 at 1.46 MB**. (This paragraph said "92 measurement-only
kernels" and "0.20 MB to 1.07 MB" until then; both predate the NN and attention
kernels landing in the default library.) Opt in:

```bash
TESSL_GEMM_TUNE=1 cargo build --release --bins
```

Both tuning binaries `exit(2)` with that command when the variants are absent,
rather than printing a page of `skip(pipe)` and exiting 0.

> **Evidence status.** All numeric performance results below come from
> historical schema-v1 artifacts. They lack the current revision, dirty-state,
> executable/metallib, host-load, and outer-round provenance contract, and none
> carries `status: published`. They document why the implementation changed;
> they do not establish current-tree speed.

| tool | what it is for |
|---|---|
| `bench_gemm_tnnt_tune` | TN / NT / TN-accum kernel A/B against the production baseline. **Interleaved**, with a baseline-drift gate — use this for kernel comparisons. |
| `bench_gemm_tile_tune` | Broad tile/BK ladder. Blocked-style timing — see the warning below. |
| `bench_gemm_sweep` | Cross-runtime timing lane (f32 exact / tf32 / bf16), JSON out. |
| `bench_gemm_sweep --dump-parity DIR` | Separate job, no timing: every lane x `BENCH_PARITY_SEEDS` draws at `BENCH_PARITY_SHAPE`, plus the manifest the scorer validates against. |
| `bench/parity_ladder.py` | Drives the above across the whole ladder, one shape at a time, and merges. Use this for the accuracy numbers. |
| `bench/test_parity_harness.py` | Adversarial cases for every harness, with machine-readable ran/skipped/failed accounting. Use `--pure` on host-only lanes and `--require-gpu` on physical Metal. |
| `bench_gemm_variants` | TN / NT / accumulate / split-K / batched / epilogue / f16 lanes -- the 18 GEMM kernels the cross-runtime sweep cannot reach. |
| `bench/kernel_coverage.py` | Measures which kernels the suite dispatches (`TESSL_KERNEL_TRACE`), cross-checked against `metal-nm` on the exact OUT_DIR metallib embedded in the release binary. `--check` gates on 100%; `--metallib` selects an explicit artifact. |
| `bench_flash_attn` + `bench/attn_paired.py` | The attention kernels, prefill and decode, over 14 configs. Historical runs first measured 11x (torch) / 20.5x (MLX) slower, geomean, worst case 271x. The row-parallel and KV-split rewrites, four throughput changes, and a routing fix followed; later schema-v1 snapshots measured 0.90-0.97x MLX prefill and 0.91-0.97x decode kernel-only. See the tessl README and do not treat those snapshots as current evidence. |
| `bench/paired_cross_runtime.py` | Alternates the tessl and torch/MLX lanes round by round. Aborts rather than reporting a geomean over part of the ladder; `--out` writes the artifact. |

The canonical numeric summary and evidence status live in the tessl README's
[latest checked-in result snapshot](../../README.md#latest-checked-in-result-snapshot).
Historical rep B measured exact f32 at **1.045× / 0.999×**, relaxed tf32 at
**2.145× / 1.998×**, and bf16 at **0.979× / 2.631×** against torch MPS / MLX.
All six comparisons fail the current hard 1.25× paired-spread gate, so those
values explain tuning decisions but do not establish current-tree speed. The
honest precision-matched conclusion is exact-f32 and bf16 parity with torch;
the roughly 2× tf32 result trades accuracy and is not a like-for-like f32
comparison. Absolute GFLOP/s also moved materially between runs and remains
same-run context rather than a portable claim.

**Timing protocol matters more than anything else here.** Measuring a baseline
block and a variant block minutes apart puts all GPU clock drift into the ratio:
run four times, `bench_gemm_tile_tune` showed the *production coop kernel against
itself* ranging 0.92x–1.46x.

> **Correction (2026-09-03).** This paragraph used to continue that
> `bench_gemm_coop_ab` "interleaves the two arms iteration by iteration", brings
> the spread to about ±5%, and flags `EXCEEDS 10%` when a run cannot support a
> comparison — and the tool table above named it as *the* tool for kernel
> comparisons. **No such binary exists in this crate.** There is no
> `src/bin/bench_gemm_coop_ab.rs` and no `[[bin]]` entry;
> `cargo build --release --bin bench_gemm_coop_ab` fails with `no bin target
> named`. A stale executable of that name survives in `target/release/` from a
> build on 2026-08-30, which is the only reason the name still looked live. The
> string `EXCEEDS` appears nowhere in the crate's sources.
>
> At the time of that audit, both surviving Rust tuning binaries used exactly
> the blocked protocol this paragraph warns against. `bench_gemm_tile_tune`
> still measures one candidate at a time and is therefore a tuning probe, not a
> publication harness; the following restoration changed
> `bench_gemm_tnnt_tune` into the counterbalanced comparison path.
>
> **Restored 2026-09-03.** `bench_gemm_tnnt_tune` now interleaves: every arm is
> timed once per round in exact forward/reverse pairs (`BENCH_ROUNDS`, even and
> at least 2; default 4), and it reports the median of the per-round ratios with
> their spread. If the baseline moves more than 10% across rounds, the process
> returns an error and publishes no comparison. It also allocates one output
> buffer per shape, shared by the baseline and every candidate — a fresh one per
> candidate is what let allocations pile up until the baseline drifted inside a
> single shape block.
>
> The median is taken *of the ratios*, not of the two medians; those are
> different statistics, and `median_of_ratios_is_not_ratio_of_medians` and
> `ratio_of_medians_can_invent_a_ratio_no_round_measured` in that binary's test
> module pin the difference. Writing those tests caught a real defect in the
> gate itself: `f64::min`/`f64::max` *ignore* a NaN operand rather than
> propagating it, so a run that produced a NaN timing scored a spread of 1.00 and
> sailed through. It now fails closed.
>
> `bench_gemm_tile_tune` is still blocked. Use `bench_gemm_tnnt_tune` for any
> ratio you intend to quote.

The same applies across runtimes. Two back-to-back runs of the identical
cross-runtime sweep disagreed by 16–21% on the torch lane alone, which is why
`paired_cross_runtime.py` exists and why single-run cross-runtime numbers should
not be quoted.

Below roughly 2 GFLOP of work both runtimes sit on a ~0.25 ms per-GEMM floor
under this submit-and-wait protocol — tessl's wall time is flat from 4 MFLOP to
2416 MFLOP. Ratios there measure dispatch latency, not the kernel.

## Adding a coop kernel

1. Compile-time `SM`/`SN` must equal the `TileGeom` the host dispatches with,
   and `constexpr int BKC` must equal `COOP_BKC` (if using explicit K loop blocking).
2. Pin it in `NN_PAIRS` in `scripts/audit_gemm_tiles.py`. The audit verifies both
   explicit `kernel void` and macro-instantiated (`NN_COOP_KERNEL`, `TN_NT_COOP_KERNEL`)
   tiles against Rust constants.
3. Route it through `nn_coop_kernel` / `src/gemm.rs` so tests and benches exercise it.
4. Verify the net can fail: inject a fault (tile drift, drop store, offset column)
   and confirm the test suite catches it before trusting a pass.

## Round 2 Evolution (2026-08-30)

In Round 2, cooperative destination tensors (`get_destination_cooperative_tensor`)
were expanded across all primary GEMM paths:
- **TN & NT bf16 descriptors:** Landed cooperative destination kernels (`TILE_COOP_TN_NT` 128×64 sg4),
  yielding 1.5–2.0× speedups over single dynamic-K multiply kernels.
- **Accumulate kernels:** Replaced `multiply_accumulate` with cooperative zero→run→load-add-store
  bias pattern (`TILE_COOP_ACCUM` 64×64 sg4), cutting memory traffic to 1 read + 1 write total.
- **NN Grid Swizzle:** Added column-panel swizzling (8 tile-rows per band) for large grids
  (`tiles_n * tiles_m >= 2048`). Recorded at 29,022 GFLOP/s at $4096^3$ on M5 Pro
  when it landed. That peak has **not** reproduced: the paired ladder measured
  26,702 GFLOP/s at that shape on 2026-09-01 and 27,734 on 2026-09-03. Absolute
  GFLOP/s is the quantity the paired design exists to avoid comparing across runs,
  so read this as same-run context for Round 2, not a standing figure.
- **Edge Slices:** Ragged boundary tiles execute origin-shifted slices with register accumulation,
  eliminating the need for separate pre-zeroing passes.
