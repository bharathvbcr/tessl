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
# Rust lanes, JSON to stdout
BENCH_SHAPES="2048x2048x2048,4096x4096x1024" cargo run --release --bin bench_gemm_sweep

# Paired cross-runtime (needs torch and/or mlx importable)
python3 bench/paired_cross_runtime.py --rounds 5 --lanes torch,mlx

# Kernel A/B, 7 interleaved rounds
TESSL_GEMM_TUNE=1 cargo build --release --bins
BENCH_ROUNDS=7 cargo run --release --bin bench_gemm_coop_ab
```

Every table in this repository is reproducible with these, on an M5 Pro. On
different silicon expect different constants — see Status in the README.
