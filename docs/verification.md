# Verification

Three layers, each of which has been shown to **fail** on an injected fault. A
check that has never failed is not known to work.

## 1. Static audit

```bash
python3 scripts/audit_gemm_tiles.py
```

Cross-checks, mechanically, the two relationships Rust's type system cannot
express:

- every Rust `TileGeom` against the `constexpr int SM`/`SN` compiled into the
  kernel it dispatches — a mismatch means the host launches the wrong
  threadgroup count and leaves output tiles unwritten;
- each cooperative kernel's `constexpr int BKC` against Rust's `COOP_BKC` — a
  drift there lets the host admit K values whose tail the kernel's
  `k + BKC <= K` loop silently drops.

It also **fails on any `*_coop` kernel that is not explicitly pinned**. Those
kernels are dispatched through a variable, never `pipeline("literal")`, so a
scanner cannot discover them; an unpinned one would escape the audit entirely
while appearing to pass.

Paths resolve from the script's own location, so it runs from any directory and
from inside an extracted `.crate`. Verified against three injected faults: a
tile drift, a BKC drift, and an unpinned kernel.

## 2. Adversarial shape sweep

Hand-picked shapes across every dispatch path: degenerate (1×1×1), primes,
one-off tile boundaries (63/65/127/129/257), exact tile multiples, extreme
aspect ratios, and shapes straddling each clause of the cooperative gate.

Output buffers are pre-seeded with a `1e30` sentinel, so a tile the kernel fails
to write is **caught** rather than read as a plausible number. Results are
checked against an f64 CPU reference; accumulating paths are checked for
`C0 + A@B` rather than `A@B`.

## 3. Randomized shape fuzz — with coverage assertions

```bash
cargo test --release --lib -- --test-threads=1 gemm_randomized_shape_fuzz

# deeper soak; seed accepts hex or decimal
GEMM_FUZZ_SEED=0xdeadbeef GEMM_FUZZ_CASES=1200 \
  cargo test --release --lib -- --test-threads=1 --nocapture gemm_randomized_shape_fuzz
```

Deterministic and seeded, so a failure prints the seed and shape and reproduces
exactly. Soak: 10 seeds × 1200 cases × 3 paths = **36,000 (path, shape)
combinations**, ~7,980 of them dispatching a cooperative kernel.

Two details are load-bearing, and both exist because the first version of this
test was quietly useless:

**It asserts its own coverage.** Every kernel `gemm` can select must be chosen
for at least 1% of cases or the run fails. The first version passed *three
injected cooperative-kernel faults* — because independent per-dimension sampling
needs M, N and K to satisfy the gate simultaneously, which happened in well under
1% of cases, so those kernels were never dispatched at all. One case in three is
now built to satisfy the gate by construction.

**Malformed env values panic rather than falling back.** `GEMM_FUZZ_SEED` used
to be parsed with `.ok()`, and `"0xdeadbeef".parse::<u64>()` fails — so a soak
across eight seeds silently re-ran one seed eight times and reported success
every time. A check that could not run must never report the same result as a
check that ran and passed.

## Fault injection

The suite is only worth its green tick if it can go red. Six faults injected
into the cooperative kernels, six caught:

| injected fault | caught by |
| --- | --- |
| `BKC` 128 → 256 (K tail silently dropped) | fuzz + sweep |
| tile `SM` 64 → 128 (rows unwritten) | fuzz + sweep |
| accumulator store removed | sentinel check |
| accumulator seeded non-zero | reference check |
| every other K block skipped | reference check |
| column offset off by one | sentinel check |

Two earlier "faults" were caught by nothing — and both were bad injections, not
gaps: one was a no-op (`if (K != 99999u)` is always true) and one perturbed the
result by less than the declared tolerance. They are recorded here because
"the test didn't catch it" and "there was nothing to catch" look identical in a
log.

## Current state

```
audit          PASS, 0 mismatches, fires on 3 injected faults
fuzz soak      36,000 (path, shape) combinations across 10 seeds
fault tests    6 of 6 caught
unit tests     68 passing
```
