# Verification

Three layers (§1-3), each of which was shown to **fail** on an injected fault. A
check that has never failed is not known to work — and a check that *can no
longer* fail has stopped working while still printing a pass, which is what §1
records about the static audit's retired `BKC` half.

```mermaid
flowchart TD
    subgraph VHierarchy["tessl Six-Tier Verification Hierarchy"]
        direction TB
        T1["Tier 1: Static Tile Audit<br/>(tests/audit_gemm_tiles.rs · Rust TileGeom vs Metal constexpr)"]
        T2["Tier 2: Adversarial Shape Sweeps<br/>(1e30 Sentinel-seeded buffers · 1x1, primes, tile edge boundaries)"]
        T3["Tier 3: Seeded Randomized Shape Fuzzing<br/>(gemm_fuzz_quick [160 cases] · gemm_fuzz_deep [2500 soak])"]
        T4["Tier 4: Hostile NN Input Boundary Suite<br/>(nn_adversarial.rs · overflow dims, capacity check, zero-dispatch)"]
        T5["Tier 5: Promoted Kernel Numeric Goldens<br/>(44 promoted kernels checked for exact values vs CPU references)"]
        T6["Tier 6: GDN Mathematical Oracles<br/>(f64 sequential reference · error damping &amp; saturation bounds)"]

        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end
```

## 1. Static audit

```bash
cargo test --release --test audit_gemm_tiles
```

It is a `#[test]` that needs no device, so the hosted CI job runs it on every push
(`scripts/hosted_tests.sh`, which hides the Metal device from every test it runs).

```mermaid
flowchart LR
    subgraph RustLand["Rust Host Side (TileGeom in src/gemm.rs, dispatch sites in src/)"]
        TileDef["TileGeom Constants<br/>• TILE_COOP_DEFAULT (128x64)<br/>• TILE_COOP_NARROW (64x64)<br/>• TILE_COOP_TN_NT (128x64)<br/>• TILE_F32 (32x32)"]
    end

    subgraph Script["tests/audit_gemm_tiles.rs"]
        Extractor["Source Reader<br/>Cross-checks SM, SN &amp; simdgroups"]
    end

    subgraph MetalLand["Metal Shader Side (kernels/matmul_tensorops.metal)"]
        ShaderDef["Compiled Kernel Constants<br/>• constexpr int SM = 128;<br/>• constexpr int SN = 64;<br/>• NN_COOP_KERNEL macros"]
    end

    RustLand --> Extractor
    MetalLand --> Extractor
    Extractor --> Pass["Every entry point checked<br/>Fails if it examined nothing"]
```

Cross-checks, mechanically, the relationship Rust's type system cannot express:
every Rust `TileGeom` against the `SM`/`SN` and simdgroup count compiled into the
kernel it dispatches. A mismatch means the host launches the wrong threadgroup
count and leaves output tiles unwritten. The kernel side is read from
`constexpr int SM`/`SN`, from the arguments of each `*_KERNEL(...)` macro
instantiation, or from a helper's template defaults when a kernel calls it with
no template argument list — `matmul2d_tensorops_bf16_f32`, the production NN
kernel, is that last case and takes its 128×64/sg4 from `mm_nn_coop_f32acc`.

The Rust side is read from the selectors, not from a hand-kept copy of them (the
`NN_PAIRS` list the Python script carried drifted from `epi_kernel`/`coop_kernel`
by construction): the `(kernel, TILE_*)` arms of `coop_kernel`, `epi_kernel`, the
batched `match` in `gemm_batched` and `SplitKPar::kernel`, plus every literal
`pipeline("matmul2d_tensorops_...")` call under `src/` paired with the `TILE_*`
within 12 lines of it.

It fails closed rather than skipping:

- every entry point in `matmul_tensorops.metal` must be reached by a pair; there is
  no exemption list. `matmul2d_tensorops_i8_f32` was exempt in the Python script
  while `src/nn.rs` dispatched it with local `SM`/`SN` constants; it now takes
  `TILE_COOP_DEFAULT`;
- a selector anchor it cannot find (a renamed `coop_kernel`), or one that pairs no
  kernel, fails: a selector it cannot see is a selector it did not check;
- a `pipeline("...")` call with no `TILE_*` within 12 lines fails, even when its
  kernel is checked through another site;
- a `*_KERNEL(` invocation it cannot parse, and a kernel whose geometry it cannot see each fail the audit;
- an audit that parsed no kernels, or checked no Rust dispatch pairs, fails;
- tessl-arch02 growing its own `matmul_tensorops.metal` fails (it must compile
  tessl's through `DEP_TESSL_KERNELS`); the check is skipped only when the sibling
  checkout is absent, as inside a published crate.

Its `PASS` line names what it checked (`cargo test --release --test
audit_gemm_tiles -- --nocapture clean_tree`):

```
PASS: tile geometry: 33 kernels, 34 Rust dispatch pairs
```

The counts move whenever a kernel or a dispatch site is added; the test's own
output, not this document, is the current figure.

The same file runs the audit against injected faults: a tile drift, a wrong tile
in one arm of each of `coop_kernel`, `epi_kernel`, the batched `match` and
`SplitKPar::kernel`, a renamed selector, a drifted template default, an
unaccounted kernel, an unparseable kernel macro, an empty kernel file, an empty
`src/`, a wrong tile at the `src/nn.rs` i8 dispatch, and that dispatch with no
`TileGeom` at all. The earlier Python script as it stood at `dade8e2` printed
`PASS` on the wrong i8 tile. Against the script as it stood at `52c091c`, the tile
drift failed, the empty `gemm.rs` crashed with a `KeyError`, and the other four
printed `PASS`.

### The retired `BKC` check (`GAP-TESSL-AUDIT-BKC-CHECK-DEAD`, closed)

The script used to claim a second relationship — each cooperative kernel's
`constexpr int BKC` against a Rust `COOP_BKC` — and printed `COOP_BKC = None`
followed by `PASS: 0 mismatch(es)` having examined none of it. Its fallback
guards keyed on a `_coop` entry-point suffix, so they could not fire either.

git history settles why, and the answer is neither a deliberate retirement nor
a rename: the check never applied to tessl. Across every commit, `COOP_BKC`
appears in no file under `src/`, `constexpr int BKC` in no file under
`kernels/`, and no entry point carries a `_coop` suffix (`git log -S` and
`git grep` over all 169 revisions, 2026-10-06). `COOP_BKC` appears only in docs from `84a52a1`, which predates tessl's
kernel sources, and in this script from `d4e4f2d`, where it arrived already
dead. Both describe the blocked kernels of the code tessl was extracted from,
whose `for k; k + BKC <= K` loop could drop a K tail. Every tessl kernel passes
K to `matmul2d_descriptor` as `dynamic_length_v<int>` and reduces all of it
inside `op.run()`, so there is no host-side K block for a gate to disagree with.

So the `BKC` half was deleted, not repaired. The `_coop`-suffix unpinned-kernel
guard was replaced by the every-entry-point rule above. Under the old script, 17
of the 27 entry points were checked; the other ten passed unexamined:
`matmul2d_tensorops_bf16_f32` (skipped as "no compile-time SM/SN"), the two
f16 NN kernels and three `*_batched` kernels (selected through a variable, never
pinned), the three split-K kernels (tile argument past the 8-line pairing
window, now 12), and `matmul2d_tensorops_i8_f32` (exempt, then checked; above).

tessl carries no gap ledger of its own; `GAP-TESSL-*` records live in the
consuming project's, at `~/Code/research/Lappi-decision/gaps.jsonl`, which is
also where `GAP-TESSL-NPY-READER-REFUSES-F64-REFERENCE` (cited from
`gdn_fixtures.rs`) is recorded. A record's current state is its last line under
that id.

## 2. Adversarial shape sweep

Hand-picked shapes across every dispatch path: degenerate (1×1×1), primes,
one-off tile boundaries (63/65/127/129/257), exact tile multiples, extreme
aspect ratios, and shapes straddling each clause of the cooperative gate.

```mermaid
flowchart TD
    subgraph SentinelPattern["Sentinel-Based Buffer Verification (Catch Silent Partial Writes)"]
        direction TB
        Seed["1. Pre-seed Output Buffer with 1e30 Sentinel<br/>(prevents zero-fill masking unwritten regions)"]
        Dispatch["2. Dispatch Kernel under Test<br/>(Adversarial shape: prime dimensions, ragged edges)"]
        Probe["3. Scan Output Buffer for Sentinel Survivals"]
        
        Seed --> Dispatch --> Probe
        Probe --> Check{"Any 1e30 Survives?"}
        Check -- Yes --> Fail["❌ Caught Silent Partial Write!<br/>(e.g., gemv_q4_tiled wrote 4 rows of 512)"]
        Check -- No --> Compare["Compare written values against f64 CPU reference"]
        Compare -- Match --> Pass["✅ Verified Numerically Sound"]
    end
```

Output buffers are pre-seeded with a `1e30` sentinel, so a tile the kernel fails
to write is **caught** rather than read as a plausible number. Results are
checked against an f64 CPU reference; accumulating paths are checked for
`C0 + A@B` rather than `A@B`.

## 3. Randomized shape fuzz

```bash
# 160 cases, part of the ordinary suite
cargo test --release --lib -- --test-threads=1 --nocapture gemm_fuzz_quick

# 2500-case soak, #[ignore]d so it stays out of the default run
cargo test --release --lib -- --ignored --test-threads=1 --nocapture gemm_fuzz_deep

# replay a failing seed
STRESS_SEED=0xdeadbeef cargo test --release --lib -- --test-threads=1 gemm_fuzz_quick
```

Deterministic and seeded, so a failure prints the seed and shape and reproduces
exactly.

> [!CAUTION]
> Everything this section said before 2026-08-31 was wrong, and wrong in the
> way this document exists to prevent. It documented a test named
> `gemm_randomized_shape_fuzz` and two environment variables `GEMM_FUZZ_SEED`
> and `GEMM_FUZZ_CASES`. None of the three exist. The command it told you to run
> therefore matched no test and printed:
>
> ```
> running 0 tests
> test result: ok. 0 passed; 0 failed; 89 filtered out
> ```
>
> A verification command that runs nothing and reports `ok` is worse than no
> command, because it converts an unexamined kernel into a documented green
> tick. It also claimed the fuzzer "asserts its own coverage — every kernel
> `gemm` can select must be chosen for at least 1% of cases or the run fails".
> No such assertion is implemented. Per-kernel coverage accounting would be
> worth having; until it exists, the fuzzer checks correctness on the shapes it
> happens to draw and nothing more.

**Malformed env values panic rather than falling back.** A seed parsed with
`.ok()` and silently discarded means a soak across eight seeds re-runs one seed
eight times and reports success every time. `STRESS_SEED` accepts hex or
decimal and refuses anything else loudly.

## 4. Hostile input across the `nn` surface

```bash
cargo test --release --test nn_adversarial -- --test-threads=1
```

Every entry point in `tessl::nn` is driven with undersized buffers, degenerate
dimensions, non-finite scalars, and dimension products chosen to overflow. Each
case asserts three things, and the third is the one that matters: the call
returns `Err`, it does not panic, and `take_dispatch_count()` is still zero.
Without that third assertion a kernel that validated *after* encoding would pass
while still having submitted work.

The checks are load-bearing rather than decorative. Compiling `nn` with the
`require` capacity checks disabled does not produce a clean failure — the suite
hangs the GPU past a 120-second timeout, against 0.06 s with them in place.

## 5. Numeric coverage of the promoted kernels

`promoted_kernels.rs` asserts each of the 44 promoted entry points resolves out
of tessl's own metallib. That is a gate on the *move*, not on correctness: a
kernel can resolve, dispatch, and return wrong numbers.

It did. `gemv_q4_tiled` resolved, had adversarial coverage of its error paths,
and wrote 4 rows of 512 because the host handed it the other Q4 kernel's grid.
Giving every promoted kernel a numeric test found six defects in total — two
grid mismatches, one undocumented weight layout, uninitialised threadgroup
scratch across half of every attention query block, a NaN in the online softmax,
and an output-width validation that made `out_bf16` unusable.

All 44 now have one, in `nn_kernels.rs`, `reductions.rs`, `nn_wiring.rs`,
`promoted_numeric.rs`, `attention.rs`, `qkv_rope.rs` and `q4_interleaved.rs`.
Where a family is selected by an enum or a bool, every arm is exercised: the
three Q4 MLX row variants share one reference, and both `Q4MlxLayout` packings
are checked against each other as well as against the dense reference.

## 6. CPU oracles for the GDN kernels (K1-K7)

No Metal kernel work is authorized yet — K1-K7 start after the shipping gate — so
this layer is the part that can be built first: the references the kernels will be
judged against, pinned before any kernel exists to shade them.

Two files, neither of which touches the GPU, so both run on any host:

- **`gdn_fixtures.rs`** — the published fixture corpus (13 cases) and its load
  path. Each case ships `y_chunked` (f32) and `y_seq_f64` (f64); only the second
  is an independent golden, because the first is the output of the algorithm
  under test. Agreement is held to `64 * u_f32`, derived from a worst observed
  1.473e-6 (24.7 u) on `L127c64`.
- **`gdn_gates.rs`** — **K7**, gate production in the *nanolab sigmoid-gate*
  form. The fixtures store *post*-gate `alpha`/`beta`, so the corpus cannot test
  K7 at all; this file is the golden for that operator only. It is not Qwen3.5's
  gate: the `qwen35_gdn_gates_f32` kernel (the `A_log`/softplus/`dt_bias` form)
  is held to `qwen35_gates_*` in `tests/fixtures/qwen35/`, captured from
  transformers' own module, by
  `gdn_gates_kernel_matches_the_transformers_module_golden` in
  `tests/qwen35_kernels.rs`.
- **`gdn_state.rs`** — **K3**, the state update, and the bound it must be judged
  by rather than the one the corpus currently uses.

`tests/common/gdn.rs` holds both references. It carries two deliberately wrong
operators alongside the right ones, so a test can *measure* each divergence
rather than assert its absence:

| the wrong thing | why it is in the file |
| --- | --- |
| `Rule::Repo` — the delta correction reads the undecayed state | nanolab's *default*. Diverges up to 31% of output magnitude, and **coincides exactly at `t = 0`** |
| `alpha_mamba2_refuted` — the build plan's K7 | the Mamba2/SSD decay from a different mixer in the same source file |

```mermaid
flowchart TD
    subgraph ErrorDynamics["GDN Numerical Error Dynamics: Saturation vs Recurrence Damping"]
        direction TB
        
        subgraph Saturation["Gate Saturation Hazard (K7: gdn_gates.rs)"]
            direction TB
            Sat_f32["f32 unit roundoff (2^-24):<br/>sigmoid saturates to 1.0 at x ≈ 16.6355"]
            Sat_f64["f64 precision:<br/>sigmoid saturates to 1.0 at x ≈ 36.7368"]
            Sat_Comp["⚠️ Between thresholds: f32 yields 1.0 while f64 yields 1.0 - eps.<br/>Gap multiplies state every step → compounds as O(L · u_f32)<br/>At L=8191: gap is ~8200 u_f32 (4.9e-4) — 4 orders above fixture bound!"]
            Sat_f32 --> Sat_Comp
            Sat_f64 --> Sat_Comp
        end

        subgraph Damping["State Recurrence Error Damping (K3: gdn_state.rs)"]
            direction TB
            Damp_Reassoc["Reassociation Divergence:<br/>Sequential vs PairwiseTree sum"]
            Damp_Alpha["Decay Gate alpha &lt;= 1.0<br/>Errors multiply by prod(alpha) &lt;= 1"]
            Damp_Bound["✅ Damped Geometric Sum: Error does NOT compound over sequence!<br/>Measured: 1.5 u_f64 at L=1 → 6.0 u_f64 at L=8191 (only 4.0× across 8191× steps)"]
            Damp_Reassoc --> Damp_Alpha --> Damp_Bound
        end
    end
```

### What K7's reference owns

Three things a kernel author would otherwise have to get right unaided, each of
which fails silently:

- **A layout change.** `a_gate`/`b_gate` are the tail of a split over the feature
  axis, so they arrive `[B, T, H]` — head index last. The recurrence wants
  `[B, H, L]`. Same element count, and every value still a sigmoid, so a missed
  transpose passes any shape, dtype or range check. `gates_published` consumes
  `[B,T,H]` and returns `[B,H,L]`, so a caller cannot skip it.
- **A per-head bias.** `decay_bias`/`update_bias` are `[H]`, broadcast over batch
  and position. An `[L]`-indexed bias is the same dtype and often the same
  length; it is refused by name.
- **The clamps, three of which cannot do anything.** `sigmoid` has range `(0, 1)`
  exactly and `[0, 1]` after rounding, so `alpha`'s ceiling and both of `beta`'s
  bounds can be attained but never exceeded — clamping there is arithmetically a
  no-op. The only clamp that changes a value is `alpha`'s floor of 1e-4, below
  `ln(1e-4/(1-1e-4)) = -9.21024`. That is why `L65_tinyalpha` is a fixture, and
  why clamping is applied in the reference rather than left to a caller: without
  it that case reproduces its golden to only 8.4e-5, with it to 1.9e-16.

### Two findings the plan did not carry

**The plan's K7 is a sign flip in disguise.** With the zero-initialised
parameters nanolab ships, `exp(-exp(0) * softplus(x))` is `exp(-ln(1 + e^x))`,
which is `1/(1 + e^x)`, which is **`sigmoid(-x)`** — an algebraic identity, not an
approximation. So the plan's formula and the real one agree *to the last bit* at
`a_gate = 0`, where an untrained checkpoint sits, and separate into mirror images
only once training moves the gate off zero. A kernel written from the plan and
smoke-tested on fresh weights would look correct, and would then rank every pair
of positions in reverse. This is the third defect in this project with that
shape: the two delta rules coincide at `t = 0`, and the conformal nonconformity
and probability scales coincide at `q_hat = 0.5`.

**Saturation is arithmetic-dependent, and it compounds past the fixture bound.**
f64 `sigmoid` returns exactly 1.0 above x ~= 36.7368; an f32 kernel does so above
x ~= 16.6355, because its unit roundoff is `2^-24`. Between those thresholds the
kernel yields `alpha == 1.0` while the reference yields `1.0 - eps` — at most one
`u_f32` per step, but `alpha` multiplies the state once per step, so across `L`
steps the gap on the decay product grows to about `L * u_f32`. At `L = 8191`,
tessl's longest fixture, that is 4.9e-4: **8200 `u_f32`, four orders above the
`64 * u_f32` bound `gdn_fixtures.rs` holds the chunked kernel to.** An
early-saturating K7 kernel is therefore not automatically wrong, and equally
cannot be judged by the fixture bound. Both thresholds are found by bisection in
`alpha_saturates_to_exactly_one_later_in_f64_than_in_f32`, not asserted from a
table.

A related limit on what a fixture can ever prove about K7: the corpus stores
post-gate `alpha`/`beta` as **f32**, so the fixture's own gate is already rounded.
K7 cannot be held to better than f32 against a fixture, whatever the kernel does.

### What K3 is, and what its reference owns

**First, a limit on this whole section.** The plan document that enumerates
K1–K7 is not in either repository. `K7` is named once —
`gdn_gate_prologue`, at Lappi-decision's `docs/plan-corrections.md:244`
(that repository, not this one) — and K3 is named nowhere. Its identity here
is reconstructed from seven cross-references, which agree but do not amount to
a specification:

| what the references pin | where |
| --- | --- |
| a GDN **state update**, reassociated across lanes, so it needs an f64 reference | `tessl-integration.md:572` |
| judged as a **K-term accumulation** — *"a delta-rule state update is"* one | `:639` |
| chained after K7 on one encoder, which gets ordering for free | `:275` |
| writes state, so it must call `require_disjoint_writes` | `:706` |
| asserts lane width against the pipeline rather than assuming it | `:525` |
| may want compile-time head-dim / tile variants, as K1/K2 do | `:1201`, `:1207` |

That is enough to build the **oracle**, because the recurrence is the same
operator whichever chunking variant K3 turns out to be. It is **not** enough to
write the Metal kernel: no entry-point name, signature, or tiling is recorded
anywhere. **Open: `GAP-TESSL-K3-IDENTITY-NOT-SPECIFIED`.**

`state_update_f64` therefore returns a `StateUpdate` rather than a bare `y`:

- `y` — `[B,H,L,D]`, bit-identical to what `sequential_f64` returned before
  (`sequential_f64` is now a one-line delegate, so there is one recurrence in the
  file and not two).
- `y_mag` — `sum_p |S[p][n] * q[p]|`, the quantity §C.3's bound is proportional
  to.
- `pred_mag` — `sum_p |S[p][n] * k[p]|`, the *other* `d`-term accumulation. It
  feeds `delta` and so the state, reaching it one step before `y`.

and `SumOrder` selects how the two dot products are summed: `Sequential`, or
`PairwiseTree` — recursive halving, the shape a simdgroup shuffle-down reduction
produces. Both orders in f64 means a divergence between them is *purely*
reassociation, with precision held constant. That is what makes the next finding
an answer rather than an estimate.

### Four findings from K3's measurements

**1. Reassociation does not compound over the sequence.** Worst divergence
between the two summation orders, relative to each element's own magnitude:
**1.5 `u_f64` at L=1, 6.0 `u_f64` at L=8191 — a factor of 4.0 across 8191× the
steps.** If it accumulated, the factor would be nearer 8191×. The mechanism is
the gate: an error entering the state at step *t* reaches `y` at step *t'*
multiplied by the product of `alpha` between them, which is at most 1, so the
contributions form a damped sum rather than a growing one. **A K3 bound therefore
does not need to scale with sequence length** — which is the opposite of the
saturation hazard in `gdn_gates.rs`, where the gap grows like `L * u_f32` because
`alpha` multiplies the state every step with nothing to damp it. Two GDN error
sources, opposite behaviour, same operator.

**2. The corpus gate is far too loose, not too tight.** `gdn_fixtures.rs` judges
the chunked output by `64 * u_f32 * max|y|`, a *global* scale. Against the derived
per-element `gamma_d(u_f32) * y_mag`:

- at every case's **widest** element the global bound is still the looser of the
  two, so the existing gate never rejects lawful f32 arithmetic — nothing here
  asks for it to be moved, and it is not moved (gates are read-only);
- at every case's **quietest** element it is looser by **528× to 75,184×**.

So a K3 defect confined to low-magnitude elements could be four orders of
magnitude worse than f32 arithmetic permits and still pass. The mechanism is
exact, not a correlation: `slack = (64/d) * (dynamic range of y_mag)`, verified to
within 1e-6 on all 13 cases — **8.0000× the range** at `d = 8`. An earlier draft
of this note blamed cancellation instead; the corpus refutes that directly, since
L65 cancels hardest (1.9e5 : 1) and has nearly the *least* slack, while
L65_tinyalpha cancels 58× less and has the most.

**3. Cancellation rules out the obvious bound.** Worst `y_mag / |y|` over the
corpus is **186,971× on L65**. A per-element `|y|`-proportional bound would there
demand about 17 significant bits beyond f32's 24, so a suite built on one would
fail on correct code. This is exactly the failure §C.3 warns about, with a
measured number attached.

**4. The corpus is `d = 8`; the model is `d = 128`.** Qwen3.5-2B's GDN state is
16×128×128. `gamma_128 / gamma_8 = 16.00`, so any bound calibrated on this corpus
understates the model's per-step accumulation budget by 16× **before** recurrence
depth is considered. A K3 kernel that passes at `d = 8` is not thereby known to
pass at `d = 128`.

### Mutation coverage

Same method as the fault-injection table below, run against both GDN references:
**19 defects injected into `tests/common/gdn.rs`, 19 caught.**

| injected fault | caught by |
| --- | --- |
| `[B,T,H]` read as `[B,H,L]` (no transpose) | 4 tests |
| bias indexed by position instead of head | `bias_is_per_head_and_broadcasts_over_batch_and_position` |
| `a_gate` wired into `alpha` twice | `a_gate_drives_alpha_and_b_gate_drives_beta` |
| `alpha`'s floor dropped | 3 tests |
| caller's clamp ignored, bounds hardcoded (`alpha`, then `beta`) | `the_clamp_is_a_parameter_not_a_constant` |
| NaN gate logit passed through instead of refused | `a_nan_gate_logit_is_refused_not_clamped` |
| non-finite bias passed through | `a_non_finite_bias_is_refused` |
| refuted reference implemented as a sigmoid (cannot show divergence) | 5 tests |
| `exp(A_log)` dropped from the refuted decay rate | 2 tests |
| sign error in `sigmoid`'s negative branch | 7 tests |
| `softplus` without the `max(x,0)` shift | `the_refuted_reference_holds_up_at_a_large_logit` |
| mis-shaped gate accepted rather than refused | `a_misshaped_gate_is_refused_rather_than_reinterpreted` |
| wrong-length bias accepted | `a_bias_indexed_by_position_is_refused` |
| `sum_tree` drops the odd tail | `a_single_step_matches_its_closed_form_at_odd_and_even_d` |
| `sum_tree` folds the same element each round | 2 tests |
| `PairwiseTree` silently reuses the sequential sum | `reassociation_error_does_not_compound_over_the_sequence` |
| state indexed `s[n*d + p]` instead of `s[p*d + n]` | 2 tests |
| magnitudes sum signed products, so they cancel | 6 tests |
| `pred_mag` deleted, or taken after the update | `the_second_step_pins_pred_mag_against_its_closed_form` |
| `pred_mag` returned in `y_mag`'s slot | 4 tests |
| `sequential_f64` delegates with the wrong order | `the_rust_f64_reference_reproduces_the_published_golden` |
| state not reset between `(b, h)` pairs | 2 tests |

**Four of those tests exist only because the mutation run demanded them.** For
K7, nothing exercised the refuted reference at a magnitude where the naive
`ln(1 + exp(x))` overflows, and nothing passed it a non-finite parameter. The
overflow is not merely imprecise — `exp(-1e-3 * inf)` is 0, so a gate that should
decay to 0.449 reads as *total* decay, making the divergence the file exists to
measure look larger than it is.

For K3 the two gaps were sharper. **Every fixture is `d = 8`, and 8 halves to 4,
2, 1 without ever leaving an odd count** — so the corpus cannot reach `sum_tree`'s
odd tail at all, and the claim that the tail is carried rather than dropped was
untested. `a_single_step_matches_its_closed_form_at_odd_and_even_d` reaches it at
`d = 5` and `d = 7`, against a closed form (`y[n] = beta * v[n] * <k, q>`, which
the recurrence collapses to at `t = 0`) rather than against another
implementation. And **`pred_mag` was verified by nothing**: the only test touching
it asserted it is zero at `t = 0`, which a `pred_mag` that is zero *everywhere*
also satisfies. Deleting the computation outright passed the whole suite. It is
now pinned at `t = 1`, where there is again a closed form, with operands whose
signs alternate so that `sum |k0 k1|` pulls clear of `|sum k0 k1|` — otherwise the
check would only be re-testing `pred`.

Two injected faults were caught by nothing and were bad injections rather than
gaps. Hardcoding the published clamp bounds in `sequential_f64` is a no-op for a
suite that only ever passes `GateClamp::PUBLISHED` to it. And removing the `.abs()`
from `abs_dot_state`'s *operand* changes nothing, because the *product* is absed
and `|a*b| == |a|*|b|` — that one found redundant code rather than a missing test,
and the redundancy was deleted. Both are recorded because "the test didn't catch
it" and "there was nothing to catch" look identical in a log.

## 7. Parser mutation tests

`src/json.rs`, `src/safetensors.rs` and `src/npy.rs` are hand-written parsers of
downloaded files. Besides their hand cases, `src/parser_mutation.rs` drives each
with seeded mutations of valid seeds (a Qwen3.5-style `config.json`, a four-tensor
safetensors file, npy v1/v2/v3 files): truncation at every byte, every single-bit
flip, boundary integers (0, 1, length +-1, `u16/u32/u64::MAX`, the header cap)
written at every position, and seeded random edits; plus hand-built hostile
headers (length fields at their extremes, reversed, overlapping and overflowing
offsets and shapes, bad magic and versions, lone surrogates, 100k-deep nesting,
huge exponents, duplicate keys, trailing garbage).

Decision (card ft-04c475ac): a dependency-free seeded mutation test, not
cargo-fuzz or a proptest dev-dependency, so no dependency is added and a failing
case replays from its seed and label. The property per input: `Ok` or a
non-empty `Err`, no panic, and no single allocation above `64 * input_len + 8 KiB`
(a counting global allocator under `cfg(test)`). A strict truncation of a valid
file must be refused, and `SafeTensors::open` must agree with `from_bytes`.

```bash
cargo test --release --lib parser_mutation:: -- --test-threads=1
# replay or widen: PARSER_MUTATION_SEED=0x... PARSER_MUTATION_CASES=20000
```

It builds no `GpuRuntime`, so the hosted CI `Host-only unit tests` step runs it.

## Fault injection

The suite is only worth its green tick if it can go red. Nineteen faults are
injected into tessl's own cooperative/TensorOps GEMM and attention kernels, one
at a time, and each is shown to make a named test fail on the GPU. **19 of 19
caught by a value assertion (2026-10-21; 18 at c2c1612, 1 at b0f307e).** No
fault went uncaught in this run, so there is no missed-fault row to record.

**Kernels covered.** `kernels/gemm_coop.h` (the cooperative-destination body,
its epilogue, the tile walks) and `kernels/matmul_tensorops.metal` (the bf16 NN
narrow tile, the bf16 NT kernels, the NN and TN split-K kernels, the bf16
parallel split-K); `kernels/attn_tiled.h` (the FlashAttention-2 body behind
`qwen35_attn_tiled_*`, the training `lse` forward and `encoder_attn_tiled_*`);
`kernels/attn_rows.h` (the one-simdgroup-per-row step behind `flash_attn_rows_*`
and `encoder_attn_rows_*`); and the split-KV reduce of
`kernels/flash_attn_decode.metal`. The shared `tile_from_linear_panel` walk is
also covered through its exact-f32 and i8 callers (the panel-walk fault is
caught by `gemm_ragged_shapes::exact_f32_column_panels_cover_every_tile` and
`gemm_i8::column_panels_cover_every_tile_exactly` as well). Not covered: the
exact-f32 NN/TN/NT kernel bodies beyond that walk and split-K, the i8-specific
kernel bodies, the q4 kernels, `gemm_q4_mlx`,
the GDN and backward attention kernels, and the SWA/global-h512 tiled variants
of `flash_attn_*` (their bodies are not `attn_tiled.h`).

**Method.** `scripts/fault_inject.sh` applies each `scripts/faults/<id>.patch`
(one hunk in one kernel source) to the working tree, rebuilds the metallib, runs
the row's candidate tests (`scripts/faults/manifest.tsv`) and restores the file
from a backup under a trap, then checks every file under `kernels/` against its
pre-run hash. A patch that does not apply is `NOT RUN (patch stale)`, a test name
that matches nothing is `NOT RUN (test not found)`, a build failure is
`NOT RUN (build failed)`: none of them counts as caught, and the self-test
(`scripts/fault_inject.sh --self-test`, also `tests/fault_injection_script.rs`)
proves it with a deliberately stale patch. `--check` reports whether every patch
still applies to the current kernels. Re-run with
`MAC_HEAVY_MAX_LOAD=60 bash <Lappi>/tools/mac_heavy.sh <label> bash scripts/fault_inject.sh --out <dir>`;
a mutated kernel is never committed.

**Runs.** Both on 2026-10-21, Apple M5 Pro (Mac17,8, macOS 27.0.1, AC power),
under the `mac_heavy` lock.

- Full run at `c2c1612` (clean tree), `bash scripts/fault_inject.sh --out <dir>`
  (idle load average at start 10.88 / 22.32 / 42.22): 18 of 19 caught by a value
  assertion, 0 by a non-assert failure; the 19th,
  `gemm-epilogue-alpha-dropped-with-beta`, was `NOT RUN (patch stale)`.
  `kernels/` was byte-identical to its pre-run hashes.
- Re-run of that row alone at `b0f307e` (`c2c1612` plus only the regenerated
  patch file), `--only gemm-epilogue-alpha-dropped-with-beta`: caught by
  `gemm_epilogue::fused_epilogue_matches_the_unfused_sequence`; `kernels/`
  byte-identical.

Why two: `d385e4f` added the STRICT path in `kernels/gemm_coop.h`, so the old
patch no longer applied; the regenerated patch mutates the same non-strict
epilogue expression. The table's rows come from those two runs'
`results.tsv` (kernel and fault from `scripts/faults/manifest.tsv`; the catching
test is the first candidate that failed with an assertion). The result logs are
not committed (the repo keeps no fault-run results files; bench results under
`bench/results/` are benchmark JSON). The earlier composite "19 of 19"
(2026-10-16, `f748ced` + `5b01c99`) is superseded by these runs.

| kernel | injected fault | caught by | observed failure |
| --- | --- | --- | --- |
| matmul2d_tensorops_bf16_f32_64x64_sg4 (coop NN, narrow tile) | tile geometry drift: SM 64 -> 32 while the host still sizes the grid for 64 rows | `gemm_ragged_shapes::nn_bf16_straddles_the_narrow_wide_kernel_boundary` | `bf16 NN boundary 67x33@33x511: element [33892] is 321351.666x its error budget (got 0, want -7.505443096160889, \|a.b\| sum 9.557201862335205, tol 2.336e-5)` |
| tile_from_linear_panel (every coop and exact GEMM walk) | dropped tile-grid tail: the last, shorter column-panel band is decoded as a full band (coop bf16/f16/relaxed with B >= 2^23 elements, exact f32, i8) | `gemm_ragged_shapes::nn_bf16_column_panels_cover_a_partial_band_at_the_b_size_gate` | `bf16 NN panel walk, 128x64 tile, 200x512@512x16384: element [2712774] is 9111.311x its error budget (got 0, want 35.65439246222377, \|a.b\| sum 126.25112960860133, tol 3.913e-3)` |
| matmul2d_tensorops_tn_splitk_par_bf16_f32 | dropped K tail: a final partial K partition is skipped, and every partition when K < k_tile (the whole output then stays unwritten) | `gemm_ragged_shapes::parallel_bf16_tn_partitions_cover_all_of_k` | `bf16 TN parallel split-K 100x300x2500, k_tile 256: element [5700] is 121.757x its error budget (got 0.34591174, want -5.218534645624459, \|a.b\| sum 305.67179442290217, tol 4.570e-2)` |
| matmul2d_tensorops_nn_splitk_f32 | split-K partial-sum error: each K partition overwrites C instead of accumulating | `gemm_ragged_shapes::long_k_nn_partitions_cover_all_of_k` | `f32 NN split-K 40x16200@16200x520: element [9360] is 30.939x its error budget (got -8.66033, want 52.4756461964785, \|a.b\| sum 2043.409290189152, tol 1.976e0)` |
| matmul2d_tensorops_tn_splitk_par_f32 | split-K partial-sum error: partition p stores into slice p-1, so two partitions share a slice | `gemm_ragged_shapes::parallel_tn_partitions_cover_all_of_k` | `f32 TN parallel split-K 12x768x4096, k_tile None: element [7008] is 51.764x its error budget (got -11.53454, want -8.320268433654267, \|a.b\| sum 253.78295958095663, tol 6.209e-2)` |
| coop_tile epilogue (every EPILOGUE coop kernel) | wrong epilogue: alpha is ignored whenever beta is nonzero | `gemm_epilogue::fused_epilogue_matches_the_unfused_sequence` | `128x64x64 act=None: worst \|delta\| = 2.580182 at 7591 (got -10.344316 want -7.7641335), tol 0.16` |
| coop_tile epilogue (every EPILOGUE coop kernel) | wrong epilogue: the beta == 0 guard is removed, so C_prev is read and multiplied by zero | `gemm_epilogue::beta_zero_ignores_whatever_c_held` | `beta=0: worst \|delta\| = 9.77573 at 4083 (got 0 want 9.77573), tol 0.16` |
| coop_tile epilogue, interior-tile bias view | swapped strides: the per-column bias view gets a row stride of N, so it indexes by row | `gemm_epilogue::bias_is_per_column_not_per_row` | `bias broadcast wrong at (1,0) for 256x128: got 0 want 1` |
| gemm_apply_activation (every EPILOGUE coop kernel) | wrong epilogue: the SiLU selector computes GELU | `gemm_epilogue::fused_epilogue_matches_the_unfused_sequence` | `128x64x64 act=Silu: worst \|delta\| = 0.19317436 at 2232 (got 1.8958186 want 1.7026442), tol 0.16` |
| matmul2d_tensorops_nt_bf16_f32 and nt_accum_bf16_f32 | swapped operands: the NT kernels read B untransposed | `gemm_correctness::nt_bf16_matches_cpu_reference` | `bf16 Nt 64x128@128x64: element [2356] is 78036.299x its error budget (got -10.705623, want 11.659647464752197, \|a.b\| sum 35.355334758758545, tol 2.866e-4)` |
| attn_tiled_body (qwen35_attn_tiled_*, encoder_attn_tiled_*) | online-softmax seed: the running max starts at 0 instead of -FLT_MAX | `attn_tiled_extreme_scores::the_tiled_kernels_survive_scores_far_below_and_above_zero` | `all scores far below -88, tile Q32K32Sg4: [0] got 0 want -0.3732729` |
| attn_tiled_body (qwen35_attn_tiled_*, encoder_attn_tiled_*) | online-softmax rescale: the O accumulator is not rescaled when the running max moves | `attn_tiled_extreme_scores::the_prefill_tiles_match_the_reference_across_many_blocks` | `multi-block causal prefill, tile Q32K32Sg4: [38656] got 0.28557223 want 0.04790396` |
| attn_tiled_body (qwen35_attn_tiled_*, encoder_attn_tiled_*) | online-softmax rescale: the running sum l is not rescaled when the running max moves | `attn_tiled_extreme_scores::the_prefill_tiles_match_the_reference_across_many_blocks` | `multi-block causal prefill, tile Q32K32Sg4: [38656] got 0.02310131 want 0.04790396` |
| attn_tiled_body (qwen35_attn_tiled_*, encoder_attn_tiled_*) | masked-row guard removed: a row with no live key is divided by l = 0 | `attention::rows_with_no_key_beside_rows_with_keys_are_zero_in_every_kernel` | `qwen35 tiled Q32K32Sg4[0]: non-finite NaN` |
| attn_tiled_body causal mode (qwen35_attn_tiled_*) | mask drift: the causal test is k < q instead of k <= q, so a query cannot see its own key | `attn_tiled_extreme_scores::the_prefill_tiles_match_the_reference_across_many_blocks` | `multi-block causal prefill, tile Q32K32Sg4: [0] got 0 want -0.24643564` |
| attn_tiled_body encoder mode (encoder_attn_tiled_*) | mask drift: the symmetric window loses its forward half | `embedgemma2_kernels::encoder_attn_matches_the_reference_at_the_real_window` | `Tiled random D=256 window=512 lens=[1100, 513, 1, 600]: max abs error 2.342e0 at 3379643 exceeds 2e-5 * 2.994e0 (got 0.8556084632873535, want -1.486357206975958)` |
| attn_tiled_body training forward (qwen35_attn_tiled_lse_*) | wrong epilogue: the saved log-sum-exp adds l instead of log(l) | `attn_train::training_attention_matches_the_reference` | `b=1 t=1 hq=1 hkv=1 lse: max err 1.000e0 at 0 (got -0.48906493 want -1.4890672876497488), 6.72e-1 of 1.489e0` |
| attn_rows_step (flash_attn_rows_*, encoder_attn_rows_*) | masked-row guard removed: a masked key's weight is exp(s - m) instead of 0 | `attention::rows_with_no_key_beside_rows_with_keys_are_zero_in_every_kernel` | `rows d=128 r=8 g=8[2048]: got 0.2879946 want 0` |
| flash_attn_decode_reduce_* (split-KV combine) | split-K partial-sum error: the reduce adds each chunk's l without exp(m_c - m_all) | `attention::decode_matches_the_reference_and_the_general_kernel` | `decode r=8 c=64 w=None sgs=None[0]: got -0.019816356 want -0.038298257` |

**The one miss of the 2026-10-16 run was a real gap, not a bad injection.** Seeding the online
softmax's running max at 0 instead of `-FLT_MAX` passed every attention test:
the softmax is shift-invariant, so with scores near zero the seed changes
nothing. It does change a row whose scores are all below about -88, where
`exp(s - 0)` underflows to 0 for every key, `l` never turns positive and the row
comes out as zeros. `attention::fast_paths_survive_extreme_score_magnitudes`
covers the row-parallel and decode kernels but not the tiled one, so
`tests/attn_tiled_extreme_scores.rs` runs all four `qwen35` tiles at scores near
-165 and +165 against an f64 reference. (Its tolerance is against the output
scale: on the clean kernel a score is an f32 K=256 dot product with an absolute
error of a few 1e-3, which `exp` turns into 0.3% per element.) It passes on the
clean kernel and fails with the fault (`got 0 want -0.3732729`).

**A restore that kept an old mtime poisoned later runs.** An early version of
the driver restored with `cp -p`; cargo compares mtimes, so it kept the metallib
it had built from the mutated source and the next baseline ran against the
fault (a new test failed its baseline, and an unrelated decode test failed with
`decode large positive d=128: got 0.1629 want 0.6517`, which looked like a flake).
The restore now bumps the mtime, the run touches every kernel before its
baseline, and the self-test fails on the old behaviour. The 2026-10-21 runs were
made with that fix.

Two earlier "faults" (from the first version of this table, which was never
run against tessl's kernels and is replaced) were caught by nothing and were bad
injections, not gaps: one was a no-op (`if (K != 99999u)` is always true) and one
perturbed the result by less than the declared tolerance. They stay recorded
because "the test didn't catch it" and "there was nothing to catch" look
identical in a log.

## Current state

> The pass counts below are the 2026-09-19 run and have not been re-run at
> `2972027`. At that commit `rg -c '^\s*#\[test\]' src tests` sums to 800
> `#[test]` functions (177 in `src/`, 623 in `tests/`), and 14 lines start
> `#[ignore` (`rg '^\s*#\[ignore' src tests | wc -l`). A `#[test]` count is
> what is collected, not what passes: no pass count is quoted for `2972027`.

Re-measured 2026-09-19 with `cargo test --release -- --test-threads=1`, which is
mandatory rather than tuning: GPU tests share default command encoders across
threads.

That command does not cover `mtl_tensor`: the module compiles only with the
`quant-prep` feature, so the default suite builds none of its tests. Run them
separately (the CI `gpu` job and `scripts/ci_local.sh` both do, and fail if the
filter matches nothing):

```bash
cargo test --release --features quant-prep --lib mtl_tensor:: -- --test-threads=1
```

On 2026-10-05 (M5 Pro, macOS 27.0.1) this ran 9 tests, 9 passing. Before
`7ffef42`, two of them were killed by SIGSEGV: `probe_tensor_support` and
`tensor_from_buffer` faulted on every call.

The torch binding's tests live in `python/tests` and are not part of `cargo test`
or the GitHub workflows. `scripts/ci_local.sh` runs the GPU-free
`test_binding_host.py` on every invocation with the torch oracle python (it
fails if there is none). It runs the GPU and MPS tests (`test_cross_entropy.py`,
`test_gdn.py`, `test_qwen35.py`) only when `MAC_HEAVY_LABEL` is set, that is
under Lappi-decision's `tools/mac_heavy.sh` (a script in that repository, not
this one); otherwise it prints a `SKIPPED` line. The script
also runs `cargo test` with `--no-fail-fast` and exits non-zero if any test
binary failed. It runs the CI kernel-emulator job's steps too: the MSL dialect
lint, the emulator host contract (`tools/msl_emu/check_qwen35.py -k
host_contract`) and the emulator kernel cases, exact and `--fast-math`. A step
that cannot run is named in the final line, `ci:local OK WITH SKIPS (not run:
...)`, never reported as a plain `OK`. The emulator kills a case that makes no
progress for `MSL_EMU_STALL_S` seconds (default 300) or runs past
`MSL_EMU_WALL_S` (default 14400).

GPU suite record (2026-10-15, commit 60a6240; hosted CI run 38739221987 ran on
that commit and its Rust jobs were green). Machine: Apple M5 Pro (hw.model
Mac17,8), macOS 27.0.1 (26A434). Run under Lappi-decision's `mac_heavy.sh` lock
(`MAC_HEAVY_MAX_LOAD=60`): the lock started the run at load average 59 and no peer held
it while it ran; no device-utilisation counter is read without sudo, so
idleness is evidenced by the lock alone.

```
cargo test --release --no-fail-fast -- --test-threads=1     exit 0
  1255 passed, 0 failed, 16 ignored, over 112 test binaries (per binary below)
cargo test --release --features quant-prep --lib mtl_tensor:: -- --test-threads=1
  11 passed, 0 failed (190 filtered out); exit 0, the filter matched
```

```
src/lib.rs                                    187 passed 0 failed 3 ignored
src/bin/bench_embedgemma2.rs                    8 passed 0 failed 0 ignored
src/bin/bench_flash_attn.rs                    11 passed 0 failed 0 ignored
src/bin/bench_gemm_coop_tile.rs                 7 passed 0 failed 0 ignored
src/bin/bench_gemm_routes.rs                    7 passed 0 failed 0 ignored
src/bin/bench_gemm_sweep.rs                     7 passed 0 failed 0 ignored
src/bin/bench_gemm_tile_tune.rs                 7 passed 0 failed 0 ignored
src/bin/bench_gemm_tnnt_tune.rs                 7 passed 0 failed 0 ignored
src/bin/bench_gemm_variants.rs                  7 passed 0 failed 0 ignored
src/bin/bench_nn_kernels.rs                     9 passed 0 failed 0 ignored
src/bin/bench_paired.rs                        35 passed 0 failed 0 ignored
src/bin/bench_qwen35_layers.rs                  7 passed 0 failed 0 ignored
src/bin/bench_qwen35_train.rs                   7 passed 0 failed 0 ignored
src/bin/probe_gdn_scan.rs                       7 passed 0 failed 0 ignored
src/bin/probe_gemm_bits.rs                      7 passed 0 failed 0 ignored
src/bin/probe_gemm_parity.rs                    7 passed 0 failed 0 ignored
src/bin/probe_load_memory.rs                    7 passed 0 failed 0 ignored
src/bin/probe_storage_memory.rs                 0 passed 0 failed 0 ignored
src/bin/probe_train_memory.rs                   0 passed 0 failed 0 ignored
tests/adamw_step.rs                             6 passed 0 failed 0 ignored
tests/attention.rs                             29 passed 0 failed 0 ignored
tests/attn_train.rs                             6 passed 0 failed 0 ignored
tests/audit_gemm_tiles.rs                      17 passed 0 failed 0 ignored
tests/bench_clip_2048_result.rs                11 passed 0 failed 0 ignored
tests/bench_env_validation.rs                   2 passed 0 failed 0 ignored
tests/bench_gemm_sweep_metallib.rs              5 passed 0 failed 0 ignored
tests/bench_paired_cli.rs                      29 passed 0 failed 0 ignored
tests/bench_qwen35_layers_cli.rs               14 passed 0 failed 0 ignored
tests/bert_kernels.rs                           9 passed 0 failed 0 ignored
tests/bert_model.rs                             3 passed 0 failed 0 ignored
tests/bert_tiny.rs                              4 passed 0 failed 0 ignored
tests/bf16_cast.rs                              2 passed 0 failed 0 ignored
tests/build_artifact_contract.rs                2 passed 0 failed 0 ignored
tests/capi.rs                                  21 passed 0 failed 0 ignored
tests/capi_contract.rs                         73 passed 0 failed 0 ignored
tests/capi_oversized.rs                         2 passed 0 failed 0 ignored
tests/check_kernel_ir_planted.rs                6 passed 0 failed 0 ignored
tests/checkpoint_view.rs                       15 passed 0 failed 0 ignored
tests/cross_entropy.rs                          8 passed 0 failed 0 ignored
tests/dialect_lint_script.rs                    5 passed 0 failed 0 ignored
tests/docs_name_real_tools.rs                   6 passed 0 failed 0 ignored
tests/embedded_metallib.rs                      1 passed 0 failed 0 ignored
tests/embedgemma2_kernels.rs                   20 passed 0 failed 0 ignored
tests/embedgemma2_model.rs                      0 passed 0 failed 1 ignored
tests/embedgemma2_tiny.rs                       9 passed 0 failed 0 ignored
tests/f16.rs                                    6 passed 0 failed 0 ignored
tests/flash_attn_rows_h64.rs                    1 passed 0 failed 0 ignored
tests/gdn_fixtures.rs                          11 passed 0 failed 0 ignored
tests/gdn_gates.rs                             22 passed 0 failed 0 ignored
tests/gdn_state.rs                             10 passed 0 failed 0 ignored
tests/gdn_train.rs                              7 passed 0 failed 0 ignored
tests/gemm_batched.rs                           9 passed 0 failed 0 ignored
tests/gemm_correctness.rs                      15 passed 0 failed 0 ignored
tests/gemm_determinism.rs                       0 passed 0 failed 0 ignored
tests/gemm_determinism.rs                       0 passed 0 failed 0 ignored
tests/gemm_determinism.rs                       8 passed 0 failed 0 ignored
tests/gemm_epilogue.rs                         10 passed 0 failed 0 ignored
tests/gemm_flag_paths.rs                        3 passed 0 failed 0 ignored
tests/gemm_i8.rs                                8 passed 0 failed 0 ignored
tests/gemm_operands_accumulate.rs               3 passed 0 failed 0 ignored
tests/gemm_ragged_shapes.rs                    19 passed 0 failed 0 ignored
tests/gemm_short_m_attack.rs                    5 passed 0 failed 0 ignored
tests/host_access_ban.rs                        4 passed 0 failed 0 ignored
tests/host_access_fallible.rs                  13 passed 0 failed 0 ignored
tests/host_access_load.rs                       2 passed 0 failed 0 ignored
tests/host_path_allocs.rs                       2 passed 0 failed 0 ignored
tests/hosted_completeness.rs                    0 passed 0 failed 1 ignored
tests/hosted_selection.rs                       6 passed 0 failed 1 ignored
tests/hosted_tests_script.rs                    9 passed 0 failed 0 ignored
tests/icb_adversarial.rs                        4 passed 0 failed 0 ignored
tests/icb_adversarial_lifecycle.rs              4 passed 0 failed 0 ignored
tests/icb_capture_matrix.rs                    16 passed 0 failed 0 ignored
tests/icb_flag_env.rs                           4 passed 0 failed 0 ignored
tests/icb_replay_barriers.rs                    2 passed 0 failed 0 ignored
tests/interop_mtl_buffer.rs                     5 passed 0 failed 0 ignored
tests/kernel_fast_math.rs                      13 passed 0 failed 0 ignored
tests/kv_safety.rs                              6 passed 0 failed 0 ignored
tests/metallib_clean_time_script.rs            18 passed 0 failed 0 ignored
tests/nn_adversarial.rs                         9 passed 0 failed 0 ignored
tests/nn_aliasing.rs                            7 passed 0 failed 0 ignored
tests/nn_decode_hazards.rs                      3 passed 0 failed 0 ignored
tests/nn_kernels.rs                            16 passed 0 failed 0 ignored
tests/nn_runtime_ownership.rs                   4 passed 0 failed 0 ignored
tests/nn_wiring.rs                             21 passed 0 failed 0 ignored
tests/npy_hardening.rs                          4 passed 0 failed 0 ignored
tests/parser_mutation.rs                       13 passed 0 failed 0 ignored
tests/promoted_kernels.rs                       5 passed 0 failed 0 ignored
tests/promoted_numeric.rs                       8 passed 0 failed 0 ignored
tests/public_api_errors.rs                     11 passed 0 failed 0 ignored
tests/q4_interleaved.rs                         7 passed 0 failed 0 ignored
tests/q4_shape_domain.rs                        3 passed 0 failed 0 ignored
tests/qk_norm_half_rope.rs                      2 passed 0 failed 0 ignored
tests/qkv_rope.rs                               7 passed 0 failed 0 ignored
tests/quantized_gemv.rs                         2 passed 0 failed 0 ignored
tests/qwen35_adamw.rs                          22 passed 0 failed 0 ignored
tests/qwen35_bwd.rs                            21 passed 0 failed 0 ignored
tests/qwen35_config.rs                          5 passed 0 failed 0 ignored
tests/qwen35_kernels.rs                        46 passed 0 failed 0 ignored
tests/qwen35_model.rs                          28 passed 0 failed 3 ignored
tests/qwen35_params.rs                         12 passed 0 failed 0 ignored
tests/qwen35_staged.rs                          4 passed 0 failed 0 ignored
tests/qwen35_train.rs                          23 passed 0 failed 6 ignored
tests/qwen35_train_traffic.rs                  13 passed 0 failed 0 ignored
tests/qwen35_weights_release.rs                 1 passed 0 failed 0 ignored
tests/reductions.rs                             7 passed 0 failed 0 ignored
tests/relaxed_predicate.rs                      1 passed 0 failed 0 ignored
tests/runtime_lifecycle.rs                     23 passed 0 failed 0 ignored
tests/safetensors.rs                           12 passed 0 failed 1 ignored
tests/shader_index_arithmetic.rs               20 passed 0 failed 0 ignored
tests/softcap_finite_adversarial.rs             1 passed 0 failed 0 ignored
tests/tune_rigs_refuse_missing_variants.rs      2 passed 0 failed 0 ignored
doc tests                                       8 passed 0 failed 0 ignored
```

Tests the cards asked to see by name, all `ok` in that run:
`prefill_then_decode_matches_forward_at_the_last_row` (tests/qwen35_model.rs:379),
`train_forward_hidden_states_are_the_inference_forwards` (tests/qwen35_train.rs:360),
`tiny_step_matches_transformers_autograd` (tests/qwen35_train.rs:284),
`stress_decode_matches_forward_on_random_shapes` (:964),
`stress_long_decode_stays_on_the_forward` (:1019),
`stress_interleaved_sessions_keep_their_own_state` (:1049),
`stress_relaxed_precision_mid_session_is_refused_not_fatal` (:1077),
`stress_hostile_rows_and_answers` (:1150) (all tests/qwen35_model.rs),
`async_encode_past_the_constant_arena_without_a_synchronize_stays_healthy`
(tests/runtime_lifecycle.rs:442), and in tests/nn_wiring.rs
`softcap_sample_rejects_an_all_nan_row`, `every_sampler_refuses_a_row_of_nan_and_negative_infinity`,
`argmax_f32_pass_refuses_a_nonfinite_row`,
`multi_pass_argmax_skips_a_group_with_no_finite_logit`.

The 16 `#[ignore]` tests, each run on its own with `--ignored --nocapture` and the
environment its doc comment names (`QWEN35_2B_SAFETENSORS` = the Qwen3.5-2B-Base
snapshot's `model.safetensors-00001-of-00001.safetensors`; `QWEN35_TRAIN_REF_DIR`
= a `make_train_fixture.py 2b --tokens 128` reference generated earlier that
day from the same snapshot; the other references generated in this worktree
with `make_reference.py`, `make_train_fixture.py tiny --layers 24` and
`embedgemma2_ref/make_reference.py`):

```
gemm_fuzz_deep                                             PASS (worst_err in the 1e-5 range)
alignment_probe_report                                     PASS (prints its table)
opens_the_qwen35_2b_checkpoint                             PASS
embedgemma2_matches_sentence_transformers                  PASS (traces ~1e-6 of max |ref|)
f32_forward_matches_transformers_fp32_layer_by_layer       PASS (final norm 2.62e-6, logits 1.92e-6, top-1 0/155)
bf16_forward_is_no_worse_than_transformers_bf16            PASS
f32_staged_and_decode_match_forward_on_the_real_2b         PASS
deep_tiny_step_matches_transformers                        PASS (worst 2.64e-5; A_log/dt_bias 9.35e-4)
real_2b_step_matches_transformers                          PASS (every parameter below)
real_2b_step_on_bf16_operands_stays_near_transformers      PASS (loss rel 3.72e-4, worst gradient 5.39e-2)
real_2b_step_on_bf16_storage_stays_near_the_f32_step       FAILED a bound (2026-10-21 run below; ft-8933ed6a)
real_2b_loss_curve_tracks_the_f32_step                     FAILED closed before its first arm (2026-10-21 run below; ft-dafde771)
real_2b_bank_accumulates_within_the_bound                  PASS (2026-10-21 run below)
real_2b_gradients_are_those_of_tessls_forward              FAILED a bound (2026-10-21 run below; ft-1d2ab5b0)
hosted_selection_has_no_device                             not run directly (needs the hosted sandbox)
every_device_free_test_is_on_the_hosted_manifest           PASS via scripts/hosted_tests.sh --outside
```

The earlier run of these 16 had three SIGTERM exits (the memguard cap, not a
test assertion), recorded then as not completed, not as passes, and one test
not run for a missing reference. The four rows above that changed are from the
2026-10-21 run below.

2026-10-21 (M5 Pro, 64 GiB): `real_2b_step_matches_transformers` was killed twice,
both times in its comparison phase after printing its loss, by the machine lock
`mac_heavy.sh` over its cap (the lock's kills, values as the lock reported them):

```
tree footprint 30,168 MiB (cap 28,672 MiB), 11.0 GiB host RSS
tree footprint 33,080 MiB (cap 31,744 MiB), 14.3 GiB host RSS
```

Host RSS excludes Metal buffers, so the growth is host memory. The cause read
from the code (inferred, not re-measured): `real_2b_results` read the whole
embedding gradient to host f64 and then `by_name` every gradient as f64, 1.88 B
parameters x 8 B = 14.0 GiB, though only the reference's files are compared.
`real_2b_results` now streams per part, reads only the kept embedding rows, and
`streamed_rel` compares the embedding a window at a time; the pre-start bound and
peak-footprint line of the loss-curve test (`footprint_requirement`,
`print_peak_footprint`) are in tests/qwen35_train.rs.

The locked run of gp/w20-2b-cap cfdb710, 2026-10-21 11:33-11:35Z, M5 Pro (Mac17,8,
macOS 27.0.1, AC power), under the `mac_heavy` lock with
`MAC_HEAVY_MAX_FOOTPRINT_MB=31744` (memguard cap 32 GiB).
`QWEN35_2B_SAFETENSORS` = the Qwen3.5-2B-Base snapshot b1485b2.
`QWEN35_TRAIN_REF_DIR` = a `make_train_fixture.py 2b --tokens 128` reference
regenerated 2026-10-21 at c2c1612 from that snapshot (loss 3.810166120529175).
Each test ran on its own with `--exact --ignored --nocapture --test-threads=1`
under `/usr/bin/time -l`; "peak" is `/usr/bin/time -l`'s peak memory footprint
in bytes.

```
test                                                        result   peak (bytes)
real_2b_step_matches_transformers                           PASS     16261980248
real_2b_step_on_bf16_operands_stays_near_transformers       PASS     16252428376
real_2b_step_on_bf16_storage_stays_near_the_f32_step        FAILED   20702084000
real_2b_bank_accumulates_within_the_bound                   PASS     25435753192
real_2b_loss_curve_tracks_the_f32_step                      FAILED   7604754312
qwen35_train::tests::real_2b_gradients_are_those_of_tessls_forward
                                                            FAILED   15818037632
```

- `real_2b_step_matches_transformers`: the test printed a peak footprint of
  15.15 GiB. Loss: train 3.81033988, transformers 3.81016612. Worst gradient
  4.05e-3 over 149 parameters. Earlier the same day, before this branch, it was
  killed over the lock cap at 30,168 and 33,080 MiB tree footprint (above).
- `real_2b_step_on_bf16_storage_stays_near_the_f32_step`: ran to its assertions
  and failed one bound. Loss: bf16 storage 2.745187 vs f32 2.745268 (rel 2.93e-5,
  the loss bound holds). `layers.1.post_attention_layernorm.weight` rel err
  7.048e-2 > 2^-4. Bit-identical on a rerun. With `GDN_TRAIN_PRODUCTION =
  Sequential` it is 8.429e-2, so the chunked GDN kernel is not the cause.
  Tracked as ft-8933ed6a.
- `real_2b_loss_curve_tracks_the_f32_step`: failed closed before the first arm:
  "REQUIREMENT UNMET (not a pass): seed 1: F32 storage with update=f32
  moments=f32 needs 34.91 GiB, over the 32.00 GiB cap (model 7.01 GiB +
  gradient bank 7.01 GiB + optimizer state 14.02 GiB + step workspace 4.45 GiB +
  host overhead 2.42 GiB)". Fitting the arm is tracked as ft-dafde771.
- `qwen35_train::tests::real_2b_gradients_are_those_of_tessls_forward`
  (ft-19f4bd16): ran to its assertions and failed.
  `layers.20.linear_attn.conv1d.weight`: tessl's loss moves as -8.820058e-1
  along v, its gradient says -8.807117e-1 (|fd - tessl| 1.29e-3 > 0.3 |d| =
  1.09e-3; |d| 3.636e-3). The second assertion (the finite difference closer to
  tessl than to torch) holds. It fails the same way on the sequential GDN
  kernel. Tracked as ft-1d2ab5b0.
  ft-1d2ab5b0 then replaced the static resolution (norms and conv1d assumed
  resolved) with one measured from the sweep: a parameter resolves only when
  three consecutive Richardson estimates over a 6-step halving sweep lie
  within 0.3 |d| of each other. A first version needing only two agreeing
  estimates (4 steps) was shown on the GPU to be coincidental at
  layers.20 conv1d.weight (successive-difference ratios 1.96 and 3.7, not 4),
  hence three; its result follows from the lead's next GPU run.

`real_2b_step_matches_transformers` (loss: train 3.81033944, inference 3.81033982,
transformers 3.81016612; train vs transformers 4.55e-5), the worst
|got - want| / max|want| of each of its 149 parameters (worst 4.05e-3,
layers.20.linear_attn.conv1d). The test's bound is 1e-2 of the parameter's
peak, 2.5x the observed worst:

```
embed_tokens-rows 3.22e-4;  norm 2.16e-3;  L0.input_layernorm 1.05e-3;  
L0.post_attention_layernorm 1.28e-3;  L0.mlp.gate_proj 1.75e-3;  L0.mlp.up_proj 1.08e-3;  
L0.mlp.down_proj 6.32e-4;  L0.gdn.in_proj_qkv 1.19e-3;  L0.gdn.in_proj_z 1.04e-3;  
L0.gdn.in_proj_b 5.82e-4;  L0.gdn.in_proj_a 6.80e-4;  L0.gdn.out_proj 7.30e-4;  
L0.gdn.conv1d 7.25e-4;  L0.gdn.A_log 1.27e-3;  L0.gdn.dt_bias 1.12e-3;  
L0.gdn.norm 1.03e-3;  L1.input_layernorm 2.28e-3;  L1.post_attention_layernorm 9.31e-4;  
L1.gdn.conv1d 7.55e-4;  L1.gdn.A_log 6.72e-4;  L1.gdn.dt_bias 4.09e-4;  
L1.gdn.norm 4.74e-4;  L2.input_layernorm 7.10e-4;  L2.post_attention_layernorm 6.57e-4;  
L2.gdn.conv1d 1.35e-3;  L2.gdn.A_log 1.05e-3;  L2.gdn.dt_bias 9.12e-4;  
L2.gdn.norm 5.61e-4;  L3.input_layernorm 3.40e-4;  L3.post_attention_layernorm 3.74e-4;  
L3.mlp.gate_proj 1.13e-3;  L3.mlp.up_proj 7.83e-4;  L3.mlp.down_proj 4.16e-4;  
L3.attn.q_proj 1.12e-3;  L3.attn.k_proj 5.44e-4;  L3.attn.v_proj 8.19e-4;  
L3.attn.o_proj 8.79e-4;  L3.attn.q_norm 7.01e-4;  L3.attn.k_norm 3.82e-4;  
L4.input_layernorm 7.98e-4;  L4.post_attention_layernorm 2.34e-4;  L4.gdn.conv1d 8.99e-4;  
L4.gdn.A_log 8.97e-4;  L4.gdn.dt_bias 9.48e-4;  L4.gdn.norm 5.00e-4;  
L5.input_layernorm 8.74e-4;  L5.post_attention_layernorm 1.66e-3;  L5.gdn.conv1d 3.28e-4;  
L5.gdn.A_log 1.99e-4;  L5.gdn.dt_bias 2.59e-4;  L5.gdn.norm 6.36e-4;  
L6.input_layernorm 3.09e-3;  L6.post_attention_layernorm 5.55e-4;  L6.gdn.conv1d 2.60e-4;  
L6.gdn.A_log 2.83e-4;  L6.gdn.dt_bias 2.87e-4;  L6.gdn.norm 1.43e-3;  
L7.input_layernorm 4.43e-4;  L7.post_attention_layernorm 5.73e-4;  L7.attn.q_norm 5.44e-4;  
L7.attn.k_norm 5.53e-4;  L8.input_layernorm 9.56e-4;  L8.post_attention_layernorm 3.05e-3;  
L8.gdn.conv1d 6.30e-4;  L8.gdn.A_log 2.24e-3;  L8.gdn.dt_bias 2.56e-3;  
L8.gdn.norm 1.36e-3;  L9.input_layernorm 2.03e-4;  L9.post_attention_layernorm 8.72e-4;  
L9.gdn.conv1d 8.88e-4;  L9.gdn.A_log 6.86e-4;  L9.gdn.dt_bias 3.32e-4;  
L9.gdn.norm 5.06e-4;  L10.input_layernorm 1.06e-3;  L10.post_attention_layernorm 2.91e-4;  
L10.gdn.conv1d 3.90e-4;  L10.gdn.A_log 4.06e-4;  L10.gdn.dt_bias 4.04e-4;  
L10.gdn.norm 7.96e-4;  L11.input_layernorm 7.45e-4;  L11.post_attention_layernorm 5.24e-4;  
L11.attn.q_norm 1.10e-3;  L11.attn.k_norm 4.80e-4;  L12.input_layernorm 2.80e-4;  
L12.post_attention_layernorm 7.89e-4;  L12.gdn.conv1d 1.13e-3;  L12.gdn.A_log 3.87e-4;  
L12.gdn.dt_bias 4.51e-4;  L12.gdn.norm 8.06e-4;  L13.input_layernorm 1.12e-3;  
L13.post_attention_layernorm 2.54e-4;  L13.gdn.conv1d 1.29e-3;  L13.gdn.A_log 3.01e-4;  
L13.gdn.dt_bias 2.35e-4;  L13.gdn.norm 8.97e-4;  L14.input_layernorm 1.01e-3;  
L14.post_attention_layernorm 3.80e-4;  L14.gdn.conv1d 1.01e-3;  L14.gdn.A_log 8.70e-4;  
L14.gdn.dt_bias 9.11e-4;  L14.gdn.norm 9.09e-4;  L15.input_layernorm 1.27e-3;  
L15.post_attention_layernorm 4.84e-4;  L15.attn.q_norm 1.91e-4;  L15.attn.k_norm 4.02e-4;  
L16.input_layernorm 1.30e-3;  L16.post_attention_layernorm 7.06e-4;  L16.gdn.conv1d 1.34e-3;  
L16.gdn.A_log 2.88e-4;  L16.gdn.dt_bias 2.60e-4;  L16.gdn.norm 1.58e-3;  
L17.input_layernorm 1.77e-3;  L17.post_attention_layernorm 5.56e-4;  L17.gdn.conv1d 1.20e-3;  
L17.gdn.A_log 5.59e-4;  L17.gdn.dt_bias 5.08e-4;  L17.gdn.norm 6.47e-4;  
L18.input_layernorm 8.80e-4;  L18.post_attention_layernorm 5.41e-4;  L18.gdn.conv1d 1.04e-3;  
L18.gdn.A_log 2.55e-4;  L18.gdn.dt_bias 2.40e-4;  L18.gdn.norm 7.88e-4;  
L19.input_layernorm 3.07e-4;  L19.post_attention_layernorm 5.86e-4;  L19.attn.q_norm 2.96e-4;  
L19.attn.k_norm 2.96e-4;  L20.input_layernorm 1.95e-3;  L20.post_attention_layernorm 1.01e-3;  
L20.gdn.conv1d 4.05e-3;  L20.gdn.A_log 2.04e-4;  L20.gdn.dt_bias 2.71e-4;  
L20.gdn.norm 2.71e-3;  L21.input_layernorm 1.49e-3;  L21.post_attention_layernorm 1.40e-3;  
L21.gdn.conv1d 3.09e-4;  L21.gdn.A_log 3.45e-4;  L21.gdn.dt_bias 4.43e-4;  
L21.gdn.norm 4.28e-4;  L22.input_layernorm 2.53e-3;  L22.post_attention_layernorm 1.86e-3;  
L22.gdn.conv1d 1.27e-3;  L22.gdn.A_log 1.01e-3;  L22.gdn.dt_bias 7.68e-4;  
L22.gdn.norm 1.37e-3;  L23.input_layernorm 3.21e-3;  L23.post_attention_layernorm 2.97e-3;  
L23.attn.q_norm 6.84e-4;  L23.attn.k_norm 7.41e-4;
```

Leak soak (src/bin/probe_soak.rs, the committed tiny fixture, under
`MallocStackLogging=1 leaks --atExit`): 16000 train steps (train_step_into plus a
waited AdamW step) and 160 decode sessions of 512 tokens (81920 decoded
tokens, each a prefill, 512 steps and a drop). `leaks` reports 0 leaks for 0
bytes at exit in both. The device's `currentAllocatedSize` is flat from
the first iteration (train 30900224 bytes) or the second session (decode
21282816 bytes; the first session's 19922944 is before the buffers recycle)
to the end. The process RSS is not monotonic: train 27 MB at the start,
177 MB at step 14000, 113 MB at the end; decode 25 MB, 222 MB at session 140, 31 MB
at the end, with a maximum resident set of 117 and 123 MB. Autorelease growth
under commit/synchronize is therefore bounded here; the sampling (every 2000
steps or 20 sessions) is coarse: RSS is read with `ps`, so these are samples of
a bounded range, not a per-step measurement.

```
audit          PASS (re-run at 2972027): tile geometry, 33 kernels, 38 Rust
               dispatch pairs. Tile geometry is all it checks (section 1)
GDN mutations  19 of 19 caught (section 6; 2026-10-06 pass, not re-run at 2972027)
fault tests    19 of 19 caught by a value assertion (2026-10-21; 18 at c2c1612, 1 at b0f307e; M5 Pro; section "Fault injection")
```

The fault-test line is a measured count over tessl's own kernels (a different
set from the mutation line above it, which targets the GDN references). Lines for
checks that were not re-run say so, for the reason section 6 gives: a check that
was not run must not be presented alongside checks that ran and passed.

## Releasing

`scripts/check_release_ready.sh` (clean tree, version triad) runs before a tag.
Pushing a `v*` tag starts `.github/workflows/release.yml`, which publishes only
if CI and Metal compile both concluded `success` for exactly the tagged commit
(`scripts/require_green_ci.sh`). Two edge cases are refused by design rather
than waited out:

- A tag pushed while CI or Metal compile on that commit is still running is
  refused. Push the tag after those runs have finished, or re-run the release
  job once they are green.
- `metal-compile.yml` runs only for pushes touching `kernels/`, `src/`,
  `tests/`, `build.rs`, `Cargo.toml`, `Cargo.lock` or its own file. A tag on a
  commit outside that filter has no Metal-compile run and is refused as "no
  run". Dispatch it on the tag first, then re-run the release job:
  `gh workflow run metal-compile.yml --ref vX.Y.Z`.
