# Architecture

How tessl picks a GEMM kernel, what the cooperative-accumulator gate guarantees,
and why three of the five layouts deliberately do not use it.

## Kernel selection

Every NN (non-transposed) TensorOps shape resolves through one function,
`tensorops_nn_kernel` in `src/gemm.rs`. One evaluation site for the gate, and it
is `pub(crate)` so tests can ask which path a shape actually takes instead of
assuming.

| dtype | gate | kernel |
| --- | --- | --- |
| bf16 | `use_coop_nn(TILE_BF16_NN, …)` | `matmul2d_tensorops_bf16_f32_coop` |
| bf16 | otherwise | `matmul2d_tensorops_bf16_f32` |
| f32 relaxed (`--tf32`) | `use_coop_nn(TILE_F32R_NN, …)` | `matmul2d_tensorops_f32_relaxed_coop` |
| f32 relaxed | otherwise | `matmul2d_tensorops_f32_relaxed` |
| f32 exact | — | `matmul2d_tensorops_f32` |

TN, NT and the accumulating TN/NT paths are selected separately and have no
cooperative variant.

## Two kernel shapes

**Blocked.** Loops over K in BK=256 blocks, accumulating into a device-memory C
tile. Block 0 uses `mode::multiply` so it seeds C and no host-side pre-zero
dispatch is needed; later blocks use `multiply_accumulate`. Handles ragged edges
and any K.

**Cooperative.** Holds the C accumulator in registers via
`get_destination_cooperative_tensor` across the whole K loop and stores once, so
C traffic is a single store regardless of K. It has **no ragged, short-K or tail
branch at all** — it trusts the host gate completely.

That difference is the point. In the blocked kernel, C traffic scales with K/BK
while useful work scales with K, so throughput falls as K grows.

## `use_coop_nn` — the only guard those kernels have

```rust
m >= tile.sm && m % tile.sm == 0
    && n >= tile.sn && n % tile.sn == 0
    && k >= COOP_MIN_K && k % COOP_BKC == 0
```

Each clause maps to something the kernel cannot do for itself:

- **`m >= tile.sm`, `n >= tile.sn`** — divisibility alone also accepts `m == 0`,
  since `0 % 64 == 0`. Found by a boundary test, not by reading the code.
- **`% tile == 0`** — every tile must be interior. There is no edge path, so a
  partial tile would read and write outside the logical matrix.
- **`k % COOP_BKC == 0`** — the loop is `for k = 0; k + BKC <= K; k += BKC`. A K
  that is not a whole number of blocks silently drops the tail and
  under-computes, with no error anywhere.
- **`k >= COOP_MIN_K`** (512) — **structural, not tuned.** Both blocked kernels
  use BK=256, so below K=512 they run at most one full block plus a tail, which
  is already a single C store; the cooperative kernel would only add the cost of
  zeroing the register accumulator. Measurement agrees: at K=256 it is 0.90×
  (bf16) and 0.93× (relaxed), crossing over from K=512.

`COOP_BKC` is 128 for both kernels, deliberately **smaller** than the blocked
BK=256: 128 divides every K the target workloads produce (768, 1152, 2304, 3072,
4096) whereas 256 does not divide 1152. Paired measurement put BK=256 and BK=128
within noise of each other on the shapes both cover, so the one with wider
coverage wins.

The two NN tiles differ — `TILE_BF16_NN` is 64×64, `TILE_F32R_NN` is 128×64 — so
a shape can be gate-eligible for bf16 and not for relaxed. M=192 is the canonical
example, and it is covered by both the boundary test and the adversarial shape
sweep precisely because using the wrong tile there is a live hazard.

## Why TN/NT are excluded

They issue a **single full-K `matmul2d`**, so there is no host-visible C
round-trip to remove. Two controls establish that rather than assuming it:

- An explicit device-C K-blocked control (`mm_tnblk_*`, `mm_ntblk_*`)
  **regresses** to 0.34–0.86× of production.
- Every cooperative TN/NT variant is **bit-identical** to production
  (`max_rel_err` exactly `0.00e0`), while the device-C control differs by
  ~1.2e-6.

Identical bits mean MPP is already accumulating in registers inside its own K
loop. The device-C control differs precisely *because* it rounds to f32 once per
block — which is the very traffic the fix removes elsewhere. Cooperative TN/NT
variants measured 0.77–1.10×: no win, and now with a reason rather than a shrug.

| path | structure | cooperative result | shipped |
| --- | --- | --- | --- |
| NN bf16 | explicit BK=256 loop into device C | 1.02–1.26× | yes |
| NN f32-relaxed | explicit BK=256 loop into device C | 1.05–1.13× | yes |
| TN bf16 | one full-K `matmul2d` | 0.84–1.06× | no |
| NT bf16 | one full-K `matmul2d` | 0.77–1.10× | no |
| TN/NT accumulate | one full-K `matmul2d` | 0.88–1.12× | no |

The accumulate row needed a correction of its own. Measured against the default
build, cooperative accumulate looked like a **1.19–2.74× win** — but the
accumulate kernels are off by default, so that baseline was a temp-buffer +
`add_inplace` fallback, not the accumulate kernel. Re-measured with the flag on,
it is 0.88–1.12×: noise. The gap between the fallback and the accumulate kernel
is real and reproducible, but enabling it is a numerics decision, not a
performance one.

## Tile ownership

The Rust `TileGeom` constants must equal the `constexpr int SM`/`SN` compiled
into the kernel they dispatch, because the host computes the threadgroup count
from them. A mismatch leaves output tiles unwritten with no error. The same
applies to `COOP_BKC` against each cooperative kernel's `BKC`.

Neither relationship is expressible in Rust's type system, so
`scripts/audit_gemm_tiles.py` checks both mechanically. It is pinned rather than
inferred: cooperative kernels are dispatched through a variable, never
`pipeline("literal")`, so a scanner cannot find them — and the audit fails on any
future `*_coop` kernel that is not pinned. See
[Verification](verification.md).
