# Benchmarking

The measurement protocol, and the five ways these numbers went wrong before they
went right. This page is longer than the architecture page on purpose: on this
hardware, at these sizes, **how you time it changes the answer more than what
you changed in the kernel.**

## The protocol

| tool | for |
| --- | --- |
| `bench_gemm_coop_ab` | Paired, interleaved kernel A/B. Use this for kernel comparisons. |
| `bench_gemm_tile_tune` | Broad tile/BK ladder. Blocked-style timing — see pitfall 1. |
| `bench_gemm_sweep` | Cross-runtime lane (f32 exact / tf32 / bf16), JSON out. |
| `bench/paired_cross_runtime.py` | Alternates the tessl and PyTorch/MLX lanes round by round. |

The A/B rig is 92 measurement-only kernels and is **not** in the default
metallib — linking it took the shipped artifact from 0.22 MB to 1.09 MB. Opt in:

```bash
TESSL_GEMM_TUNE=1 cargo build --release --bins
```

Both tuning binaries `exit(2)` with that exact command when the variants are
absent, rather than printing a page of `skip(pipe)` and exiting 0.

## Pitfall 1 — a noise floor wider than the effect

The tile-tune rig times a baseline block and a variant block minutes apart, so
GPU clock drift lands entirely in the ratio. Run four times, it showed the
**production kernel measured against itself** ranging **0.92×–1.46×**.

Any single-run conclusion from that rig is inside its own noise.
`bench_gemm_coop_ab` interleaves baseline and candidate iteration by iteration
(A,B,A,B,… alternating which goes first) and reports the ratio of per-round
medians across repeated rounds. That brought the spread to roughly ±5%.

## Pitfall 2 — a harness that contaminates its own baseline

The A/B harness allocated a fresh output tensor per candidate. Those piled up
across the run, and with nine candidates the **baseline kernel** drifted
0.268 → 0.665 ms *inside a single shape block*. Read naively, that is a 2.5×
candidate win invented entirely by allocation order.

Fixed by allocating once per shape. The harness now also prints its own baseline
spread across rows and flags

```
EXCEEDS 10%: ratios above are not comparable
```

because this failure is otherwise invisible — every number still looks like a
number. One shape (2048×768×768) still trips that guard and is excluded rather
than quoted.

## Pitfall 3 — cross-process drift

Running the Rust sweep once and the Python sweep once, minutes apart, puts all
the drift between them into the ratio. Two back-to-back runs of the *identical*
benchmark disagreed by **16–21% on the PyTorch lane alone** — larger than most
differences being reported.

`bench/paired_cross_runtime.py` alternates the lanes round by round and reports
the median of per-round ratios plus the spread. **Single-run cross-runtime
numbers should not be quoted**, including earlier ones from this project: a
"1.07× ahead of PyTorch on bf16" claim did not survive the paired protocol and
became 1.00×.

## Pitfall 4 — stale binaries

`cargo test --lib` rebuilds the library but **not** the binaries. Two rounds of
"the fix isn't working" were a stale `bench_gemm_sweep`. Worse, a conclusion was
drawn from those numbers — that in-kernel register pressure cost ~25% — and it
was never actually tested.

Rebuild binaries explicitly before benchmarking:

```bash
cargo build --release --bins
```

## Pitfall 5 — traversal order

The tune rig used row-major tile traversal while production uses Morton/Z-order
on square power-of-two grids. The control read 0.71–0.88× of production *despite
identical geometry*. A rig must mirror `tile_from_linear`, or every variant is
penalised on square shapes.

## The dispatch floor

Under this submit-and-wait protocol both tessl and PyTorch sit on a **~0.25 ms
per-GEMM floor**. tessl's wall time is flat from 4 MFLOP to 2416 MFLOP:

| shape (K=512) | MFLOP | tessl ms | PyTorch ms |
| --- | --- | --- | --- |
| 64×64 | 4 | 0.266 | 0.299 |
| 512×512 | 268 | 0.234 | 0.244 |
| 1536×1536 | 2416 | 0.267 | 0.396 |
| 2048×2048 | 4295 | 0.360 | 0.547 |

A 600× increase in work at constant wall time is not a GEMM measurement. Below
roughly 2 GFLOP these benchmarks measure submit-and-wait latency, so ratios there
— including bf16 `square_512` at 0.84× — say nothing about kernel throughput.
Real workloads batch many dispatches into one command buffer and do not pay this
per-GEMM.

## Reproducing

```bash
# Rust lanes, human table plus JSON to stdout. BENCH_SHAPES overrides the
# built-in ladder; BENCH_WARMUP and BENCH_ITERS default to 10 and 50.
cargo run --release --bin bench_gemm_sweep
BENCH_SHAPES="2048x2048x2048,4096x4096x1024" cargo run --release --bin bench_gemm_sweep

# Paired cross-runtime (needs torch and/or mlx importable)
python3 bench/paired_cross_runtime.py --rounds 5 --lanes torch,mlx

# Tile-geometry and TN/NT A/B lanes. These live behind TESSL_GEMM_TUNE because
# the 92 measurement kernels are excluded from the default metallib.
TESSL_GEMM_TUNE=1 cargo build --release --bins
cargo run --release --bin bench_gemm_tile_tune
cargo run --release --bin bench_gemm_tnnt_tune

# Bit-exact parity of TensorOps against the reference SIMD path
cargo run --release --bin probe_gemm_parity
```

> [!NOTE]
> This block previously named a binary `bench_gemm_coop_ab` and an environment
> variable `BENCH_ROUNDS`. Neither exists; the crate builds four binaries and
> they are the ones listed above. The command failed with "no bin target named
> `bench_gemm_coop_ab`" for anyone who tried it.

## Measured — M5 Pro, 2026-08-31

`cargo run --release --bin bench_gemm_sweep`, medians over 50 iterations after
10 warmup, single run rather than paired, so treat these as the machine's shape
rather than a cross-runtime claim:

| shape | f32 exact | tf32-relaxed | bf16 | simdgroup f32 |
| --- | ---: | ---: | ---: | ---: |
| 512³ | 796 | 1,284 | 1,358 | 912 |
| 1024³ | 4,167 | 7,291 | 8,410 | 2,380 |
| 2048³ | 6,431 | 13,383 | 21,220 | 2,735 |
| 4096³ | 6,292 | 13,624 | **26,642** | 2,254 |
| qkv_proj (2048×768×768) | 4,262 | 7,639 | 9,209 | 2,348 |
| mlp_up (8192×3072×768) | 6,127 | 16,293 | 24,776 | 2,639 |
| mlp_down (8192×768×3072) | 6,403 | 15,301 | 25,288 | 2,334 |
| tall_k1024 (4096×4096×1024) | 6,180 | 16,096 | 25,126 | 2,732 |

GFLOP/s. Three things in that table are worth reading rather than skimming.

**bf16 TensorOps is 11.8× the portable simdgroup fallback at 4096³** (26,642 vs
2,254) and 4.2× exact f32. That ratio is the entire argument for the
cooperative-destination path.

**At 512³ the simdgroup fallback beats TensorOps f32** — 912 against 796. Not a
regression: at that size the whole GEMM is inside the dispatch floor described
above, so the comparison measures submit-and-wait latency, not the kernels.

**f32 exact flattens at ~6,400 GFLOP/s from 2048³ onward** while bf16 keeps
climbing to 4096³. Exact f32 uses the 32×32 single-simdgroup kernel with no
register accumulator, so it is bandwidth-bound where the cooperative kernels are
not.

## The `nn` kernels — M5 Pro, 2026-08-31

```bash
cargo run --release --bin bench_nn_kernels
```

These kernels are small enough that the dispatch floor above dominates them
entirely. A 1×4096 RMSNorm moves 32 KB; issued alone with a `synchronize()` it
reports ~620 µs, which is the driver, not the shader. So each kernel is timed
twice and both columns are printed:

* **batched** — `set_async_encode(true)`, 64 dispatches accumulated into one
  command buffer, one `synchronize()`, divided by 64. What a decode loop pays.
* **solo** — `set_async_encode(false)`, one dispatch, one synchronize.

`async_encode` defaults to **off**, so the solo column is what a caller gets
without asking for anything. The gap between the columns is what the packed
encode path is worth, and at `mlp_silu n=4096` it is 4.1 µs against 203 µs —
**49×**.

| kernel | shape | batched µs | solo µs | GB/s |
| --- | ---: | ---: | ---: | ---: |
| `rms_norm_f32` | 1×4096 | 24.2 | 388.6 | 2.0 |
| `rms_norm_f32` | 512×4096 | 55.1 | 254.5 | 305.0 |
| `rms_norm_f32` | 2048×4096 | 310.5 | 512.7 | 216.2 |
| `mlp_silu` | n=4096 | 4.1 | 203.2 | 11.9 |
| `mlp_gelu_tanh` | n=4096 | 4.0 | 199.7 | 12.2 |
| `mlp_silu` | n=1M | 30.0 | 213.0 | 419.2 |
| `mlp_gelu_tanh` | n=1M | 30.9 | 216.4 | 406.7 |
| `mlp_silu` | n=8M | 440.1 | 582.4 | 228.8 |
| `softmax_rows_f32` | 512×4096 | 79.8 | 259.2 | 210.4 |
| `softmax_rows_f32` | 2048×8192 | 588.2 | 796.2 | 228.2 |
| `row_sum_f32` | 2048×8192 | 276.3 | 502.9 | **242.9** |
| `row_max_f32` | 2048×8192 | 277.7 | 435.3 | 241.7 |
| `gemv_q8` | 4096×4096 | 274.5 | 487.9 | 68.9 |
| `gemv_q8` | 11008×4096 | 383.6 | 721.8 | 132.4 |
| `gemm_i8_dequant` | 512³ | 18.9 | 210.6 | — |
| `gemm_i8_dequant` | 2048³ | 343.4 | 526.7 | — |

The two integer GEMMs are compute-bound rather than bandwidth-bound: 14,221 and
**50,032 GFLOP/s**. The latter is 1.9× the bf16 TensorOps peak in the table
above, which is the expected shape for int8 on these accelerators.

The `n=1M` row at 419 GB/s is the honest ceiling estimate for this machine —
above it the working set stops fitting and `n=8M` settles at 229 GB/s, which is
where the large reductions also land. Read ~240 GB/s as "saturating memory" for
these shapes.

### RMSNorm was one thread per row — fixed 2026-08-31

This benchmark's first run found `rms_norm_f32` peaking at **87 GB/s** where
`row_sum_f32` reached **243 GB/s** on identical traffic, and taking 404 µs to
move 32 KB at 1×4096.

Not an artifact. The kernel's own first line said "One thread per row", and
`dispatch_1d(rt, &p, rows)` launched exactly `rows` threads, each walking its
row serially twice — once to accumulate the sum of squares, once to scale. At
the decode shape `rows = 1`, so the whole kernel ran on a single GPU thread.
RMSNorm runs twice per transformer layer on every token, making decode the worst
point on that curve.

All three variants — `rms_norm_f32`, `rms_norm_bf16`,
`rms_norm_residual_add_f32` — now use one threadgroup per row with the sum of
squares reduced as a tree, which is the pattern `reduce.metal` already used:

| shape | before | after | speedup | GB/s before → after |
| --- | ---: | ---: | ---: | --- |
| 1×4096 (decode) | 404.3 µs | **24.2 µs** | **16.7×** | 0.1 → 2.0 |
| 512×4096 | 556.2 µs | **55.1 µs** | **10.1×** | 30.2 → 305.0 |
| 2048×4096 | 770.7 µs | **310.5 µs** | **2.5×** | 87.1 → 216.2 |

The 512×4096 case now reaches 305 GB/s, above the large reductions, because its
working set still fits cache where a 2048-row sweep does not.

The reduction reassociates where the serial loop did not, so results differ in
the low bits. That is why the new tests compare against an f64 reference rather
than a matching f32 accumulation — the old `rms_norm_ref` summed in f32 in the
same order the serial kernel did, which agreed with the kernel's rounding
instead of measuring it.

Two things this exposed beyond the kernel itself. The existing tests used `dim`
of 16 to 64, and `reduce_tptg` hands a 4096-wide row 1024 threads, so every
lane's strided loop ran exactly once — **deleting the loop entirely left all
three tests green**. `rms_norm_sums_rows_wider_than_one_threadgroup` and its
sibling now cover 4096, a ragged 3000, and 8192, and both kill that mutation.
And `REDUCE_TREE` moved into `kernels/reduce_tree.h` so the two callers share
one definition; `build.rs` had to start tracking `.h` for `rerun-if-changed`,
without which editing the shared reduction would leave both dependents stale in
the metallib while the tests reported a pass.

Every table in this repository is reproducible with the commands above, on an
M5 Pro. On different silicon expect different constants — see Status in the
README.
