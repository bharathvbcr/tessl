# Optimization map — Audit 6

**Living gate file for agents.** Update deltas here after each opt flip.
Code paths: [`arch_02_value_resid/metal-native`](../../../ojas/arch02/metal-native/).

Related: [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) · [`mlx.md`](mlx.md) ·
[`metal4_mpp.md`](metal4_mpp.md) · [`coreml_metal_ml.md`](coreml_metal_ml.md) ·
[`DECISIONS.md`](../../../ojas/arch02/metal-native/DECISIONS.md).

---

## Live baselines (gates)

Measured on **M5 Pro / FineWeb or synthetic / B=16** unless noted. Prefer these over any
README “~403 disp” or stale 91.60/417 wording.

| Gate | Value | Artifact / cite |
|------|-------|-----------------|
| **Encode** | **Metal 4 only** | `GpuRuntime::new` hard-fails if M4 init fails |
| **Step time (default)** | **56.6 ms/step** | Soft-harden: `multiply_accumulate` **off** (2026-07-12) |
| **Throughput** | **~72k tok/s** | same (`--bench --bench-steps 20 --f32`) |
| **Binders / step** | **250** | default temp+add dW/dx; was 211 with `METAL_NATIVE_GEMM_ACCUM=1` |
| **Speed A/B (accum on)** | **~55–56.5 ms / 211** | Soft regresses — opt-in only |
| **Prior gate (Audit 6 accum on)** | 55.1 ms / 211 / 74311 tok/s | pre–Soft-harden |
| **Prior gate (Audit 4)** | 58.4 ms / 276 / 70151 tok/s | packed encoder + residency |
| **Soft EMA BPB (default)** | **2.050** (2.0502) | `out/sota_f32_clipsoft_seed1337_harden/` — accum off |
| **Soft EMA BPB (quality)** | **2.037** (2.0369) hist.; **2.1063** current Soft-split+FA_TILED @3k | hist. `out/sota_f32_clipsoft_seed1337_fa_tiled/`; Phase G remeasure `out/phaseG_ctrl_softsplit_fatiled/` (DECISIONS M17) |
| **Soft EMA BPB (pre-Audit6)** | 2.0381 | `out/sota_f32_clipsoft_seed1337/` (404 binders era) |
| **Soft EMA BPB (Audit6 accum)** | 2.0580 | `out/sota_f32_clipsoft_seed1337_audit6/` — **REJECT** as default |
| **Soft EMA BPB (bf16)** | **2.037** (2.0370) | `out/sota_bf16_clipsoft_seed1337/` |
| **Phase G FA_BLOCKSOFT @3k** | **2.1044** (vs ctrl 2.1063) | Soft-split+FA_TILED+BLOCKSOFT — **REJECT** Soft BPB (M17) |
| CUDA ref BPB | 1.9944 | Soft@3k still open; Δ ≈ 0.11 under current Soft-split+FA_TILED |
| bf16 binders | **294** (was 361 post–weight-banks; was 594 cast-per-GEMM) | P1e act tape + bwd banks (2026-07-12) |
| Phase wall | fwd ~1 · **bwd ~48** · optim ~8 ms | Soft-harden default (accum off) |
| Lib gate | **57/57** green | `cargo test --release --lib`; includes native optimizer, checkpoint, telemetry, and NaN gates |
| RSS | stable ~680 MB after warmup (bench) / ~1.2 GB FineWeb | Hot weights/grads/optim + cold tape temps |
| **20k Soft recipe** | FA_TILED + `--warmdown 3500` | FINAL EMA **1.8969** (seed 1337 golden) |
| **100k Soft recipe** | WSD: `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000` | FINAL EMA **1.8828** (~5.6 h); last-10% alone **REJECT** |
| **Exact 128M replay** | B16/T256, Muon NS5, checkpoint v6 | **13.16 GiB current physical / 1707 dispatches / zero swap / PASS** |
| **Optimizer study** | 14/16 native; funnel **complete** through exact-128M two-seed 1000 | Winner **muon_polar_adamw @ matrix-lr 0.05**; MiMuon/SOAP systems-blocked |
| **128M champion (2000 steps)** | `out/champion_128m_seed1337_audit8` (WIN stack) | FINAL EMA sliding BPB **2.0107** (seed 1337); seed 2026 **2.0404**; ~1580–1680 ms/step mid-run, ~13.5 GiB phys. Supersedes audit7 @ 2.0156 / 2005 ms. **No same-shape CUDA 128M ref yet.** |
| **128M long20k (WIN)** | `out/long20k_128m_audit8` | FINAL EMA **1.8155**; ~1582 ms/step |
| **128M WIN phase wall** | fwd synced ~412 · bwd synced ~457 · optim ~413 ms | `out/blog/prof_{FWD,BWD,OPTIM}_win`; muon_banks **99.8%** of optim |

### Soft regression bisect + harden (2026-07-12)

| Flip | Soft EMA / late gnorm | Verdict |
|------|----------------------|---------|
| Audit 6 + `multiply_accumulate` on | **2.0580** / ~9 @2999 | **REJECT as default** — hotter late clip-soft |
| `METAL_NATIVE_GEMM_ACCUM=0` (now default) | **2.0502** / ~3.4 @2999 | **KEEP** — restores late gnorm; partial BPB |
| `METAL_NATIVE_RESID_BWD_FUSE=0` | mid-run BPB not better | **KEEP fuse on** |
| `METAL_NATIVE_FA_TILED=1` (+ accum off) | **2.0369** / ~4.0 @2999 | **KEEP opt-in** — best Soft; ~69 ms/step |
| `--flash-tensorops` (+ accum off) | **2.0462** / **~13** @2999 | **REJECT** — late gnorm explosion; M8 still blocks default |

Early Soft loss good vs Audit6 matched through ~1000; divergence is late dynamics.
P1a/P1a2 accum is the Soft late-gnorm root cause; remaining Δ to 2.038/1.994 is FA track.

### Audit 7 — exact-128M backward (2026-07-19, **measured on M5 Pro**)

Champion telemetry: backward ≈ 2.17 s of a ~2.94 s step at ~0.95 effective
TFLOPS (vs ~3.5 for fwd). Flags default OFF; harness:
[`scripts/bench_128m_ab.sh`](../../../ojas/arch02/metal-native/scripts/bench_128m_ab.sh).
Bench ladder (steps 6–7, per-step telemetry on, seed 1337):

| Flip | step ms | bwd ms | disp | Verdict |
|------|---------|--------|------|---------|
| baseline | ~3028 | ~2291 | 1702 | — |
| `BWD_CAST_ONCE=1` | ~2889 | ~2181 | 1554 | **KEEP** — Δloss 2.9e-6 vs off, inside measured off-vs-off run noise ~4e-6 (M13 replay atol 1e-5) |
| `GEMM_ACCUM_DX=1` | ~2955 | ~2218 | 1630 | **KEEP with CAST_ONCE** — dW banks untouched |
| **both** | **~2811** | **~2088** | **1482** | **−7.2% step / −220 disp; beats full-accum ref below** |
| `GEMM_ACCUM=1` ref | ~2872 | ~2162 | 1483 | stays Soft-REJECT (Audit 6) |
| `FA_TILED=1` @128M | ~2970 | ~2266 | 1702 | **KEEP for quality** — 500-step champion-config EMA BPB **2.4163 vs 2.4209** (−0.0046), speed cost ≈ noise |

### Audit 7b — the backward is one kernel (2026-07-19 `BWD_PROFILE` readout)

Section shares at exact-128M (synced; two consecutive steps agree):

| Section | Share | Note |
|---------|-------|------|
| **fa_dqdkv** | **~70%** (~1.1 s) | hand-written **scalar** row FA bwd × 24 layers |
| mlp_gemms | 11% | already bf16 TensorOps |
| qkv_gemms | 5.3% | already bf16 TensorOps |
| resid_glue | ~4.8% | fused megakernels |
| all 13 others | <3% each | glue / clip / stem / head |

**P2 (MLP coop postfix) is now formally dead** — it targets the 11% while the
70% is untouched, and the GEMM sections already run on the accelerators.
FA_TILED costs the *same* ~1.1 s as the row path, so BR=BC=32 tiling does not
change the regime.

Two structural defects found by reading the row kernels — both fixed in
[`kernels/flash_attn_bwd_fast.metal`](../../../ojas/arch02/metal-native/kernels/flash_attn_bwd_fast.metal):

1. **Loop-invariant reloads.** `dq_row` re-reads `Q`/`dO` and `dkv_row` re-reads
   `K`/`V` from device memory on *every* inner iteration. The compiler cannot
   hoist them — `device const float *Q` may alias the `device float *dQ` output.
2. **Accumulators in scratch.** `d_lim = min(D, 64u)` is a runtime bound, so
   `thread float dq[64]` is dynamically indexed and spills to private memory
   instead of registers (and wastes half its slots at head_dim 32).

| Flip | Mechanism | Gate |
|------|-----------|------|
| `METAL_NATIVE_FA_FAST=1` | `*_row_d32_f32`: hoisted operands + compile-time DH=32 + full unroll. **Identical numerics.** | pure speed win; `fa_bwd_row_d32_matches_generic` lib test must pass at ≤1e-5 |
| `METAL_NATIVE_FA_BF16=1` | as FA_FAST + bf16 Q/K/V, f32 dO/L/Δ/accum/grads | bf16-class parity in the same test + `fa-quality` 500-step probe |

**RETRACTED (2026-07-19):** an earlier revision of this section claimed bf16 bwd
was a fwd/bwd *consistency fix*. That was wrong. `model_fwd::use_bf16_flash`
returns `false` unconditionally, so the forward **always** runs f32 flash and
tapes an f32 LSE — `flash_attn_fwd_bf16` is unreachable. bf16 backward therefore
*introduces* a small precision mismatch rather than removing one. It is
justified **empirically only** (two-seed 500-step + 2000-step champion: no
measurable quality cost) and is now gated on `PrecisionMode::Bf16` so `--f32`
runs cannot break the 1e-4 bwd goldens.

**Dead code recovered:** `flash_attn_bwd_{dq,dkv,delta}_bf16` already existed in
`phase4_bf16.metal` but were referenced **zero** times from Rust. They are the
*tiled* bf16 twins and are drop-in signature-compatible with their f32
counterparts (only Q/K/V become `bfloat`). `FA_BF16` now reaches them, so bf16
FA bwd works on **both** paths — no new kernel needed for tiled+bf16:

| Combination | Kernels | head_dim |
|-------------|---------|----------|
| `FA_FAST` | `*_row_d32_f32` (new) | 32 only |
| `FA_BF16` | `*_row_d32_bf16` (new) | 32 only |
| `FA_TILED + FA_BF16` | `flash_attn_bwd_{dq,dkv}_bf16` (recovered) | any ≤ 64 |

FA_FAST/FA_BF16 on the row path require head_dim == 32; the host guard warns
once and falls back to the generic kernels otherwise.

**Same defect class elsewhere:** 32 `thread float x[64]` arrays across six
kernel files all use the runtime `d_lim = min(D, 64u)` bound. Only
`flash_attn_bwd.metal` is on the hot path; `flash_attn_fwd.metal` /
`phase4_bf16.metal` (forward ~300 ms/step, ~10%) are the next candidates if the
FA bwd fix confirms the spill theory. `qkv_post_bwd` (0.9%) and `xsa_bwd`
(0.4%) are not worth touching.

### Audit 7c — first quality readout (2026-07-19, **single seed — provisional**)

CAST_ONCE parity re-gated: step 0 bit-identical, max |Δloss| **7.15e-6** over 3
steps → **PASS** at the M13 replay atol of 1e-5.

500-step champion-config EMA BPB. **Caveat: the `fa-quality` probes stack
CAST_ONCE+ACCUM_DX**, so they compare against each other, not against the clean
baselines:

| Run | Flags | BPB |
|-----|-------|-----|
| row baseline | none | 2.4209 |
| FA_TILED | FA_TILED | 2.4163 |
| fa_fast | FA_FAST + CAST_ONCE + ACCUM_DX | 2.4249 |
| fa_bf16 | FA_BF16 + CAST_ONCE + ACCUM_DX | **2.4198** |
| fa_tiled_bf16 | FA_TILED + FA_BF16 + CAST_ONCE + ACCUM_DX | 2.4243 |

Two provisional reads:

- **ACCUM_DX looks like it costs quality.** FA_FAST is mathematically identical
  to the row path, so `fa_fast` acts as a control for the GEMM flags: 2.4249 vs
  2.4209 clean = **+0.0040**. CAST_ONCE is parity-clean, leaving ACCUM_DX. This
  contradicts the Audit 7 hypothesis that dW banks were the sole Audit 6
  regression source — **dX-only accumulate is apparently not safe either.**
- **bf16 FA bwd looks like it helps.** `fa_bf16` beats `fa_fast` by **−0.0051**
  under identical other flags — the cleanest pairing in the set, consistent with
  bf16 bwd removing the fwd/bwd LSE mismatch. Tiling's earlier edge does *not*
  survive bf16 (`fa_tiled_bf16` 2.4243 vs row `fa_bf16` 2.4198).

### Audit 7d — FA row fix CONFIRMED (2026-07-19, measured)

The spill/reload diagnosis holds. Bench ladder (8 steps, mean of last 4):

| Run | ms/step | bwd ms | disp | vs base |
|-----|---------|--------|------|---------|
| bench_baseline | 2804 | 2102 | 1702 | — |
| `FA_FAST` | **2069** | **1371** | 1702 | **−26.2%** |
| `FA_BF16` | **1919** | **1214** | 1774 | **−31.6%** |
| `FA_BF16 + CAST_ONCE + ACCUM_DX` | **1797** | **1094** | 1554 | **−35.9%** |
| `FA_TILED` | 2856 | 2152 | 1702 | +1.9% |
| `FA_TILED + FA_BF16` | 2863 | 2158 | 1774 | +2.1% |

**Backward 2102 → 1094 ms (−48%)** from removing loop-invariant reloads and
register spills alone — `FA_FAST` is exact-numerics and delivers 2/3 of it.
Steady state at 500 steps is better still: **1629 ms/step ≈ 2514 tok/s**, vs the
champion's 2941 ms / 1392 tok/s — **+80% throughput**.

**The tiled path is a different bottleneck.** Neither d32 specialization (not
applied there) nor bf16 moves it: FA_TILED and FA_TILED+FA_BF16 both sit at
baseline+2%. Tiled kernels are not spill-limited — threadgroup staging /
barriers / occupancy dominate. So the recovered Phase 4 bf16 twins are a
compatibility option (any head_dim), **not** a speed path, and tiling's quality
edge cannot be combined with the row speed win.

**Revised step budget** (at 1629 ms steady state): bwd ~1094 (61%) · optim ~460
(**26%**) · fwd ~260 (14%). The optimizer is now the #2 target, ahead of the
forward — reverse of the pre-fix ordering. Re-run `BWD_PROFILE=1` to get the new
intra-backward composition before choosing the next lever.

**None of this is decided.** These deltas (0.004–0.005) are the same magnitude
the funnel already called *statistically tied* at 1000 steps (MONA 2.1809 vs
Polar 2.1823), and FA_FAST is same-math but not bit-identical (merged
score/dp loops can change FMA contraction), so 500 steps of chaotic divergence
is expected. **Resolved below — the 7c reads did not survive two seeds.**

### Audit 7e — two-seed verdict (2026-07-19): quality effects are noise

`seed-repeat`, CAST_ONCE held fixed, 500 steps, seeds 42 / 2026:

| Arm | seed 42 | seed 2026 | mean | Δ vs ctl |
|-----|---------|-----------|------|----------|
| ctl (CAST_ONCE) | 2.4564 | 2.4245 | 2.4405 | — |
| + ACCUM_DX | 2.4667 | 2.4182 | 2.4425 | +0.0020 (**sign flips**: +0.0103 / −0.0063) |
| + FA_BF16 | 2.4527 | 2.4233 | 2.4380 | −0.0025 (both seeds negative) |

**Seed spread on the control alone is 0.0319** — roughly 10× every effect being
measured. Therefore:

- **ACCUM_DX is NOT a quality regression.** The Audit 7c reading of +0.0040 was
  single-seed noise; the sign reverses on seed 2026. That provisional claim is
  **retracted**. It is also not shown to be a *win* — it is simply unresolved,
  and now worth only ~3% step time.
- **FA_BF16's quality edge is suggestive but not significant.** Direction is
  consistent across both seeds, magnitude (−0.0025) is well inside seed noise.
  Adopt it for **speed**; make no BPB claim.
- 500-step BPB cannot resolve anything below ~0.03 at this scale. Do not gate
  future flips on it without ≥2 seeds.

**Speed reproducibility is excellent** and is the real result: CAST_ONCE+FA_BF16
gives 1685.7 / 1690.8 ms/step and bwd 756.0 / 756.7 ms across the two seeds —
sub-millisecond agreement on backward. **Backward 2102 → 756 ms (−64%)**,
step 2804 → 1688 ms (−40%).

**Decision: ship `CAST_ONCE + FA_BF16`. Drop ACCUM_DX** — no demonstrated
benefit in the post-FA regime and it inherits the unresolved Audit 6
accumulate-mode risk; not worth ~3% for that exposure.

### Audit 7f — champion rerun VALIDATED (2026-07-19)

`out/champion_128m_seed1337_audit7`, `CAST_ONCE + FA_BF16`, seed 1337, 2000
steps (means of last 10 logged steps):

| | old champion | audit7 | Δ |
|---|---|---|---|
| **FINAL EMA BPB** | 2.015756 | **2.015576** | −0.00018 (tie) |
| ms/step | 2895.0 | **2005.1** | **−30.7%** |
| tok/s | 1431 | **2071** | **+44.7%** |
| backward ms | 1963.9 | **1056.0** | **−46.2%** |
| forward ms | 449.7 | 436.8 | −2.9% |
| optim ms | 442.3 | 471.3 | +6.6% |
| dispatches | 1975 | 1899 | −76 |
| current physical | 13504 MB | 13510 MB | flat |
| nonfinite | 0 | 0 | ✓ |

Quality is a tie (−0.00018, vs a 500-step seed spread of 0.032) at **+44.7%
throughput**, zero nonfinite, flat memory. This replaces the champion.

**New budget: bwd 1056 (53%) · optim 471 (23%) · fwd 437 (22%).** The forward is
larger than the earlier 260 ms estimate and now rivals the optimizer.

### ⚠ Reference-scale correction (2026-07-20) — read before quoting any number

**The `~650–840 ms` and `BPB 1.9944` "3070 Ti reference" figures apply to
`--preset sota` only: 4 layers × 128 dim × mlp 384.** `arch02-128m` is
24 × 768 × 2304 — about **100× the parameters**. `train.rs` was printing both
strings on *every* bench and writing `"reference_3070ti":1.9944` into *every*
`metrics.jsonl`, including 128M runs, which implies a cross-scale comparison
that was never run.

Fixed: both are now gated on `num_layers == 4 && model_dim == 128`. Non-sota
runs print "no same-shape CUDA reference on record" and emit
`"reference": null`. Earlier revisions of this file quoted "Δ 0.021 vs CUDA"
for the 128M champion — **that comparison is withdrawn.**

`parameter-golf/train_gpt.py` takes the same env vars as the MLX trainer, so a
genuine same-shape CUDA number is one command on the 3070 Ti:
`./scripts/cuda_ref_128m.sh probe && … bench && … quality` (or
`./scripts/blog_results.sh cuda-ref` for the Mac-side instructions). Note 8 GB
VRAM may force `GRAD_ACCUM_STEPS>1` at 128M — disclose it if so, since it costs
CUDA speed. "Does not fit on 8 GB without accumulation, fits outright in 64 GB
unified" is itself a legitimate finding.

**Phase F status (Mac probe 2026-07-21):** this host is Apple M5 Pro / arm64 —
`nvidia-smi` absent, `torch.cuda.is_available() == False`. Probe/bench/quality
**must** run on the 3070 Ti box. Mac ingest path is ready:
`logs/cuda128m/` + `scripts/ingest_cuda128m.sh` + `scripts/score_cuda128m.py`.
Until those CUDA artifacts arrive, same-shape 128M CUDA speed/BPB = **null**.
Do not quote sota 1.9944 / 650–840 ms against 128M.

### Audit 8 — post-FA targets (2026-07-20; **measured** 2026-07-21)

**Defaults flipped (M16).** `CAST_ONCE`, `FA_FAST`, `FA_BF16` are now **default
ON**. Audit 9 Phase A also defaults **`FA_FWD_FAST` ON**. `ACCUM_DX` stays OFF
(two seeds could not separate it from noise). **`GLUE_ROWBLOCK` stays OFF** —
measured A/B ≈ noise (see below); keep flag for research only.

**`resid_glue` atomics hypothesis — WITHDRAWN as a speed claim.** Measured ~61–63 ms
in WIN bwd moving ~3.93 GB ≈ 63 GB/s. Rowblock (`METAL_NATIVE_GLUE_ROWBLOCK=1`)
cuts atomics ~128× and **did not move the clock** (`out/blog` glue sweep:
glue_off **1488.5** vs rb16 **1481.7** vs rb32 **1492.4** ms/step; rb128
**regresses** to 1580). Theory wrong or bound moved — do not treat glue atomics
as ~50 ms headroom. Re-diagnose under Audit 9 Phase D or accept ~60 ms floor.

- **Implemented** as `METAL_NATIVE_GLUE_ROWBLOCK=1`
  ([`block_glue_bwd_rowblock.metal`](../../../ojas/arch02/metal-native/kernels/block_glue_bwd_rowblock.metal))
  for research; **REJECT as WIN/default**.
- Bench: `./scripts/bench_128m_ab.sh glue` / `./scripts/blog_results.sh glue`.

**New instruments:** `METAL_NATIVE_FWD_PROFILE=1` (stem / skip_resid_glue /
qkv_gemms / qkv_post_ve / flash / xsa / attn_out_gemm / mixer_other / mlp /
layers_tail / head) and `METAL_NATIVE_OPTIM_PROFILE=1` (embed_adamw /
scalar_adamw / muon_banks). The optimizer is ~23% of step and had never been
sectioned.

**Harness hazard fixed:** every mode used `[ -x "$BIN" ] || build`, which
happily ran a **stale binary** after a source change — the same failure the
funnel guards against. All modes now call `ensure_build` (always incremental).
One `fwd` run was invalidated by this before it was caught.

### Audit 8b — forward d32 flash CONFIRMED (2026-07-20)

`METAL_NATIVE_FWD_PROFILE=1`, shipped bwd stack, 4-step bench:

| Section | generic | `FA_FWD_FAST` | Δ |
|---------|---------|---------------|---|
| **flash** | **264.28 ms (41.9%)** | **39.11 ms (10.2%)** | **−85%, 6.8×** |
| mlp | 176.55 | 169.48 (44.2%) | — |
| qkv_post_ve | 62.80 | 59.49 (15.5%) | — |
| qkv_gemms | 54.78 | 53.41 (13.9%) | — |
| attn_out_gemm | 31.81 | 26.13 | — |
| skip_resid_glue | 28.65 | 24.50 | — |
| **fwd total (synced)** | **630.2** | **383.6** | **−39%** |
| step ms | 1758.5 | **1496.7** | **−14.9%** |
| tok/s | 2329 | **2737** | **+17.5%** |

Loss identical at step 3 (6.6955); dispatches 1626 → 1698 (+72 = 3 bf16 casts ×
24 layers). The threadgroup-memory hypothesis was right: halving staging
(`[BC*64]` 16 KB → `[BC*DH]` 8 KB, 4 KB bf16) plus register-resident
accumulators moved a *tiled* kernel that bf16 alone could not. **This is also
the first execution of bf16 forward flash** — `use_bf16_flash` has been
hard-coded `false` since Phase 4.

**New forward budget:** mlp **169 ms (44%)** · qkv_post_ve 59 · qkv_gemms 53 ·
flash 39. The forward MLP block now runs at only **~4.1 TFLOP/s** (696 GFLOP in
169 ms) versus **10.7 TFLOP/s** for the *backward* MLP GEMMs — a 2.6× gap on
comparable shapes. That section includes mlp_act, bf16 casts and the residual
add, so it is not pure GEMM, but the gap is the largest unexplained item left.

**FA_FWD_FAST is MEASURED KEEP** (not “unmeasured”). Flash 264→39 ms (−85%);
step −14.9% on the Audit 8b table. Under Bf16 it also reaches `flash_attn_fwd_d32_bf16`.

**ACCUM_DX “KEEP −7.2%” as a ship claim — WITHDRAWN.** Early ladder suggested a
step win with CAST_ONCE; two-seed quality could not separate it from noise and it
inherits Audit 6 accumulate risk. **REJECT as default**; not in WIN.

**FA_TILED is not required for 128M quality.** Champions/long20k ran without it.

---

## Audit 9 — architecture truth (code-audited, 2026-07-21)

Homogeneous **Attention U-Net** (not MinGRU). Both `sota` and `arch02-128m`:

| | sota | arch02-128m |
|--|--|--|
| Shape | 4L × 128 | **24L × 768** |
| Heads / KV / D | 4 / 2 / 32 | **24 / 12 / 32** (GQA=2) |
| MLP | 3×C | **2304** |
| Params | ~0.78M | **128_367_988** |
| B×T | 16×256 = 4096 tok | same |

Per-layer: `resid_mix+RMS → QKV GEMM → qkv_post (VE? + vr_λ + QK-RMS + RoPE + q_gain)
→ flash → XSA? → out GEMM → resid_scale+RMS → MLP up → sq_leaky(0.5) → down → resid_scale`.

- “Gated attention” = per-head `q_gain[H]` in `qkv_post`.
- `ModelConfig.value_residual` is **false** on Attention presets (MinGRU banks only).
- MLP is **not SwiGLU** — `leaky_relu(x,0.5)` then square (`mlp_act_sq_leaky_f32`).

**Code defaults (`ab_flags.rs` after Audit 9A):** `CAST_ONCE` / `FA_FAST` /
`FA_BF16` / `RESID_BWD_FUSE` / `MUON_SG` / **`FA_FWD_FAST`** = ON.
`GLUE_ROWBLOCK` / `FA_TILED` / `GEMM_ACCUM*` = OFF. `use_bf16_flash()` hard-coded
**false** (bf16 fwd flash only via `FA_FWD_FAST` + Bf16).

**Blog WIN env (pre-A):** `CAST_ONCE=1 FA_FAST=1 FA_BF16=1 FA_FWD_FAST=1` —
**drop `GLUE_ROWBLOCK`** from WIN (noise). After Phase A, bare
`train --preset arch02-128m` matches WIN speed without env archaeology.

**WIN profile (measured, `out/blog/prof_*_win`):**

| Phase | ms | Dominant |
|--|--:|--|
| bwd synced | ~457–490 | mlp_gemms ~135 (29%), fa_dqdkv ~82, resid_glue ~62 |
| fwd synced | ~405–414 | mlp_down ~70, mlp_act ~45, flash ~41, mlp_up ~38 |
| optim | ~413 | **muon_banks 99.8%** |

**Primary leftover wall:** muon_banks ~400 ms → Audit 9 Phase B. Then MLP f32
act / bf16 cast sandwich (Phase C). Glue only if a new design beats the floor
(Phase D).

### Audit 9B — muon_banks (2026-07-21)

Sub-profile (`METAL_NATIVE_OPTIM_PROFILE=1`, steady step): **mlp_up ~202 ·
mlp_down ~192 · qo ~175 · kv ~43 ms**. Inside each bank, XXT / A² / BX+poly
dominate; prepare+finalize ≲5%. Polar and NS5 are both **5** poly steps (same
GEMM count). ~2.3 TFLOP of TensorOps GEMM / ~400 ms ≈ **~5.8 TFLOP/s** — GEMM-bound.

| Experiment | Result |
|--|--|
| Fuse zero+`mode::multiply` into one binder | **REJECT** — no step win |
| Inter-bank overlap | **blocked** — shared `muon_scratch` + already NAX-busy |
| NS3 (3 poly steps) speed probe | optim ↓ but step ≈ noise; **not** champion (quality) |

**Verdict: accept ~400 ms muon floor** for Polar@0.05 until a quality-safe
fewer-step orthogonalizer or larger NAX tiles land. Instrumentation kept
(per-bank + `muon_bank_detail`).

### Audit 9C — MLP sandwich (2026-07-21)

Fwd path was bf16 up → **f32** `hidden_pre` → f32 act → **cast bf16** → bf16 down.
Shipped `mlp_act_sq_leaky_f32_to_bf16` so act writes bf16 directly (pre-act stays
f32 for bwd). Disp −24/step (one cast × 24 layers). **Speed REJECT** — cast tax
≲ few ms at measured BW; step win ≪5%. Float4 act variant **regressed** act lap.

Pure GEMM floors (WIN FWD profile, synced): mlp_up ~38 ms · mlp_down ~70 ms ·
mlp_act ~45 ms — act is not “cast sandwich alone”; do not reopen
`MLP_COOP_POSTFIX`.

### Audit 9D — resid_glue floor (2026-07-21)

Rowblock (design 1) already **REJECT** as speed (A/B ≈ noise). Atomics were not
the clock. No second design clears ≥3% step without a new bound diagnosis
(Instruments occupancy/memory). **Accept ~60 ms resid_glue floor**; remove from
headroom narrative.

### Audit 9E — FA fwd/bwd precision (2026-07-21)

Under Bf16 + `FA_FWD_FAST` (now default), dispatch already selects
`flash_attn_fwd_d32_bf16` when the pipeline exists — fwd/bwd both bf16 QKV with
f32 accum. Soft@3k quality remains Phase G; no further 128M default change.

**Dead / unwired (do not resurrect without KEEP):** `use_bf16_flash` generic,
`flash_attn_fwd_bf16_nolse`, `flash_attn_tensorops_tile_f32`, `dkv_recompute`,
`MLP_COOP_POSTFIX` (not in `AbFlags`).

### Audit 6 code flips (2026-07-12)

| Flip | Result |
|------|--------|
| P1a QKV bwd NT/TN `multiply_accumulate` | landed then **Soft-REJECT as default** — flag `METAL_NATIVE_GEMM_ACCUM` (default off) |
| P1a2 dW accum-into-bank (MLP/out/stem) | same API; Soft-gated with P1a |
| P1b resid/RMS bwd megakernels + skip_resid bf16 twin | **KEEP** — Soft bisect no mid-BPB win when off |
| P1c out-of-place qkv_post / xsa / mlp_act | killed tape `deep_copy` |
| P1d hazard barriers A/B (`METAL_NATIVE_HAZARD_BARRIERS`) | default strict; drop duplicate pack barriers |
| P0b+ `BufferKind::Hot` for weights/grads/optim/EMA | Cold reserved for tape temps |
| P1e hot in-place bf16 bank refresh | banks in-place; act cast tax cut in residual A/B (below) |
| **Net gate (post Soft-harden)** | **~56.6 ms / 250 binders**; Soft EMA **2.050** (quality FA_TILED **2.037**) |

### Audit 6 residual A/B (2026-07-12, post–55.1 gate)

| Flip | Result |
|------|--------|
| P1d `METAL_NATIVE_HAZARD_BARRIERS=1` as default | **REJECT** — ~53.5 ms but NaN loss by step 3; keep opt-in only |
| P1e bf16 act tape + bwd weight banks | **KEEP** — bf16 **361 → 294** binders; ~53.2–53.5 ms (was ~53.9); f32 gate unchanged **55.1–55.5 / 211** |
| P2 `METAL_NATIVE_MLP_COOP_POSTFIX` | **DEFER** — flag still unread; no NAX util / no postfix kernel to wire without Instruments |
| P3 FA quality (Soft EMA → ~1.994) | **PARTIAL** — FA_TILED long Soft **2.0369** (KEEP opt-in); `--flash-tensorops` Soft **2.046** but late gnorm~13 **REJECT**; M8 still blocks TO default |
| P4 quantized TensorOps train | **SKIP** — deploy/Core ML only; training quant GEMM still non-goal |
| Soft-harden `GEMM_ACCUM` default off | **KEEP** — Soft **2.058→2.050**, late gnorm **9→3.4**; speed A/B via `METAL_NATIVE_GEMM_ACCUM=1` |

### Audit 4 code flips (2026-07-12)

| Flip | Result |
|------|--------|
| P0 cold recycle + `removeAllocation` after sync | RSS plateaus (no +440 MB/step) |
| P0b working-set probe + pool cache cap + banner | `recommended_ws=51.8 GiB`, wired_budget 0.9×, `--pool-cache-mb` / `--wired-fraction` |
| P1 packed compute encoder across `with_binder` | **80.5 → 58.4 ms** (−27%) |
| P1 skip dead non-XSA `add_inplace(dv,dv_flash)` | 278 → 276 binders |
| P1b persistent bf16 weight banks | enabled under `PrecisionMode::Bf16`; refresh after optim |

### B=32/64 rebench

| Batch | ms/step | tok/s | Notes |
|-------|---------|-------|-------|
| **B=16** (gate) | **55.1** | **74311** | Default; binder/bwd-bound |
| **B=32** (`--tok-mult 2`) | **105.2** | **77885** | Higher tok/s; RSS ~1.3 GB stable |
| **B=64** (`--tok-mult 4`) | **197.7** | **82876** | Stable; RSS ~2.5 GB |

B=32 improves **tokens/sec** despite higher ms/step. Prefer B=16 for step-time gate; consider B=32 for throughput runs.

### Machine

See [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md): M5 Pro, 20 GPU,
64 GB, macOS 26.5.2. Apple M5 “3–4× TTFT” is cross-chip inference — not free
headroom below this gate.

---

## Audit 4 verdict (historical)

**Encode + primary GEMM + packed encoder + cold residency recycle are M5-correct**
(landed 2026-07-12 → **58.4 / 276**). Audit 4’s former “remaining” items
(pack, residency leak, wired policy) are **DONE**. Current backlog =
[`Audit 6`](#audit-6--second-pass-current-backlog) (bwd fusion, dW accum,
barriers, Hot residency).

DualInputBuffers overlap and “just enable NAX” remain weak/false at current shape.
Keep training on custom Metal 4 + MPP (DECISIONS **M2/M3/M6**). MLX = doctrine only.

### What Audit 4 overturned

| Earlier assumption | Code reality |
|--------------------|--------------|
| DualInputBuffers / H2D overlap is a lever | Prefetch ≪1%; **always-sync every step**; ping-pong vestigial |
| Wired limits are #1 step-time win | Wired/cache = **stability + larger B**; B=16 is binder/bwd-bound |
| FA-3 / TensorOps flash next for speed | FA is **quality** track; row bwd already chosen; M8 still true |
| “More megakernels” vaguely | **Concrete ROI:** bwd QKV = **12 binders/layer ≈ 48/step** at [`model_bwd.rs` 619–639](../../../ojas/arch02/metal-native/src/model_bwd.rs) |
| Docs “sync only at log/eval” | **False** — sync every step (cross-step async poisoned ~step 2100) |
| AGENTS gate 91.60 / 417 | Stale; live **58.4 / 276 / BPB 2.038** (was 80.5/278 pre-pack) |

---

## Audit 6 — second pass (current backlog)

Extends Audit 5. **Encode + TensorOps GEMM + residency recycle + packed encoder
are M5-correct.** Audit 6 fusion flips landed → live gate **55.1 ms / 211 binders**.
Remaining open items are quality (P3), speculative NAX (P2), and deploy quant (P4).

### Audit 5 fusion (landed in Audit 6 flips)

| Pri | Work | Est. | Where |
|-----|------|------|-------|
| **P1a** | Fuse bwd QKV (`gemm_nt`+add+`gemm_tn`+`accum_bank` ×3) | **DONE** (− binders) | [`model_bwd.rs`](../../../ojas/arch02/metal-native/src/model_bwd.rs) |
| **P1b** | Mirror resid/RMS bwd megakernels + skip_resid bf16 twin | **DONE** | bwd megakernels |
| **P1c** | Kill tape `deep_copy` (q/k/v_pre, attn_y_flash, mlp_pre_act) | **DONE** | [`model_fwd.rs`](../../../ojas/arch02/metal-native/src/model_fwd.rs) |
| P0b | B=32/64 FineWeb rebench under wired policy | Throughput | `--tok-mult` |
| P2 | MLP coop postfix | Speculative | NAX util only |
| P3 | FA Soft EMA 2.038 → ~1.994 | Quality | Not step-time |

### Audit 6 — new core levers

| Pri | Work | Est. | Notes |
|-----|------|------|-------|
| **P1a2** | dW `multiply_accumulate` into bank views | **DONE** | MLP/out/stem (+ QKV fuse) |
| **P1d** | Hazard-aware Device barriers A/B | **REJECT as default** | Opt-in NaNs; keep always-on Device barriers |
| **P0b+** | Wire `BufferKind::Hot` residency | **DONE** | Hot = weights/grads/opt/EMA; Cold = tape |
| **P1e** | bf16 activation cast tax | **PARTIAL KEEP** | Tape keeps bf16 attn_in/mlp_in/Y/H; bwd uses bf16 banks. Grad operands still cast. FA QKV casts already off |

### Ranked backlog (Audit 6)

| Pri | Work | Status |
|-----|------|--------|
| **P0** | Residency leak (cold recycle) | **DONE** |
| **P0b** | Wired / WS / pool cache | **DONE** (FineWeb B=32/64 rebench still open) |
| **P1** | Encoder pack + dead-add skip | **DONE** — **58.4 / 276** |
| **P1a** | Fuse bwd QKV | **DONE** then Soft-gated — default temp+add |
| **P1a2** | dW accum-into-bank | **DONE** then Soft-gated with P1a |
| **P1b** | Resid/RMS bwd + skip_resid bf16 twin | **DONE / KEEP** |
| **P1c** | Kill tape `deep_copy` | **DONE** |
| **P1d** | Hazard-aware barriers | **REJECT as default** (opt-in NaNs); keep flag |
| **P0b+** | Hot residency wiring | **DONE** |
| **P1e** | bf16 activation casts | **PARTIAL** — **294** binders / ~53.5 ms; residual = f32 grad casts + mlp_act/FA O still f32 |
| **P2** | NAX util → MLP coop postfix | **DEFER** — no util evidence; flag unread; no kernel |
| **P3** | FA quality → CUDA ~1.994 | **PARTIAL** — FA_TILED Soft hist. **2.0369** / Soft-split hist. **2.0222** opt-in; current Soft-split+FA_TILED Soft@3k **2.1063**; FA_BLOCKSOFT Soft@3k **2.1044 REJECT**; TO flash Soft REJECT (gnorm); Soft@3k Δ vs 1.9944 remains |
| P4 | Core ML quant / deploy TensorOps | **SKIP** for train hot path |

**Implementation order (remaining):** Soft@3k FA gap still open after Phase G
FA_BLOCKSOFT **REJECT** (M17) — next Soft lever is **not** online-softmax
recurrence of this class → P1e residual grad casts (if bf16 train) → P2 only
after Instruments NAX.

### Smaller / conditional (Audit 6)

| Item | Verdict |
|------|---------|
| Host `zero()` on every `alloc_tensor_f32` | Prefer skip-zero when kernel overwrites; ~1–3 ms wall risk |
| Profiler mid-step `synchronize` on phase enter | Footgun for profiled steps only; bench gate OK |
| Microbatch / logit chunking | Memory for large B — weak at B=16 |
| FA pack delta+dq+dkv | Small alone; better with barrier A/B |
| `METAL_NATIVE_MLP_COOP_POSTFIX` | **Still unread** — defer until Instruments NAX util shows MLP-up epilogue BW-bound |
| Exact f32 TILE_V2 / `-ffast-math` | Golden risk — keep closed |

---

## Binder anatomy (f32 B=16, L=4 → **250** Soft-harden default; **211** with `GEMM_ACCUM=1`)

| Bucket | Binders | Notes |
|--------|---------|-------|
| Forward | ~82 | enc 16×2, dec 21×2, stem+head |
| **Backward** | **densest** | resid megakernels on; dW via temp+add (default) |
| Clip + optim | ~12 | Muon 4 banks packed; AdamW segment-packed |
| zero_grads | 1 | packed |

**Audit 6 binder ROI — landed:** QKV fuse path, resid/RMS bwd megakernels, kill
tape `deep_copy`. **P1a/P1a2 multiply_accumulate default OFF** after Soft
regression (late gnorm ~9 → ~3.4). **P1d barriers** rejected as default (NaNs).

**bf16:** weight banks **DONE**. P1e partial: stream acts + bwd banks → **294**
binders. Residual = f32 grad casts into `ensure_bf16` + mlp_act/FA O still f32.

---

## Memory / residency

| Item | Status |
|------|--------|
| Cold recycle + `removeAllocation` after sync | **DONE** — RSS ~683 MB stable |
| Wired / WS probe / pool cache CLI | **DONE** |
| Hot vs cold residency sets | **DONE** (Audit 6 P0b+) — Hot weights/grads/opt/EMA; Cold tape |
| B=32/64 FineWeb under wired policy | **OPEN** (synthetic B=32 already ~109.5 ms / 75k tok/s) |

Private heaps / ICB stay ruled out. Residency ≠ “bring back MTLHeap.”

### P2 — How to measure NAX util

1. Short `--bench` under Instruments (Metal System Trace / neural accelerator).
2. Inspect util on Muon banks, top GEMMs, FA.
3. Only port TensorOps-inside-Muon or MLP cooperative postfix if compute-bound and util is low.
4. Env flag placeholder: `METAL_NATIVE_MLP_COOP_POSTFIX=1` (**still unread** —
   defer until NAX util + postfix kernel exist).

### P3 — FA quality backlog

| Item | Status |
|------|--------|
| Row-wise FA bwd default @ T=256 | **Keep** (step-time); Soft EMA default **2.050** |
| Tiled FA bwd | **Soft KEEP opt-in** `METAL_NATIVE_FA_TILED=1` — hist. Soft EMA **2.0369** (~69 ms); Soft-split hist. **2.0222** |
| TensorOps flash | Opt-in `--flash-tensorops` / DECISIONS **M8** — Soft EMA 2.046 but **late gnorm~13 REJECT** |
| FA_BLOCKSOFT (Phase G) | Opt-in `METAL_NATIVE_FA_BLOCKSOFT=1` — Soft@3k **2.1044** vs ctrl **2.1063** — **REJECT Soft BPB** (M17); kernels kept for research |
| Soft EMA → ~1.994 | **Open** — current Soft-split+FA_TILED Soft@3k **2.1063** vs CUDA **1.9944** (Δ≈0.11); online-softmax recurrence tweaks exhausted for this gap |
| Low-risk numerics tweaks | FA_TILED recipe documented; FA_BLOCKSOFT measured null for Soft@3k |

### GEMM / NAX depth (skeptical)

| Item | Status |
|------|--------|
| Morton, packed zero+matmul, TN/NT, split-K | Done |
| `execution_simdgroups<4>` / BK=128 | bf16 + relaxed only |
| Interior offset tiles | Measured slower — keep off |
| Muon NS5 simdgroup inside bank | Toy/small fallback only |
| Exact-scale Muon matrix contractions | **Bank-batched TensorOps NN/TN/NT landed**; 4201→479 ms optimizer at 128M |
| TensorOps flash | Probe only; M8 blocks default |

---

## How to rebench (after each flip)

```bash
cd arch_02_value_resid/metal-native
cargo test --release --lib
cargo run --release --bin train -- --bench --bench-steps 20 --f32
# Compare: ms/step ≤ ~57 (default Soft-harden) or clear win; binders ~250
# Soft ladder (default): EMA ~2.050
# Soft quality: METAL_NATIVE_FA_TILED=1 → EMA ~2.037 (~69 ms)
# Speed A/B: METAL_NATIVE_GEMM_ACCUM=1 → ~211 binders (Soft regresses)
# Other: METAL_NATIVE_MUON_SG=0 | METAL_NATIVE_GEMM_INTERIOR=1
#         METAL_NATIVE_HAZARD_BARRIERS=1 (NaNs — do not default)
```

Record new numbers in this file’s “Live baselines” table.

---

## Ruled-out / false opportunities (keep closed)

| Idea | Why |
|------|-----|
| Default tiled FA bwd @ T=256 | Measured +~14–22 ms |
| Disable Muon simdgroup | Measured +~35 ms |
| f32 GEMM interior tiles | ≤1 ms / slower |
| TensorOps multi-block flash default | M8; no TO bwd |
| DualInputBuffers as step-time win under always-sync | Vestigial |
| Cross-step async (skip per-step sync) | Poisoned ~step 2100 |
| MTLHeap / ICB / Metal 3 restore / Core ML train / mlx-rs rewrite | Ruled out |
| “Bump wired → faster B=16 steps” | Pressure/B-scale, not FLOPs |
| Data loader I/O | ≪1% |
| More AdamW packing | Already complete |
| Expect M5 4× on this step | Already on M5 TensorOps GEMM |

---

## History — Audits 1–3 (archived)

### Phase 2 recovery context (still true)

| Metric | Pre | Phase 2 post | Recovery |
|--------|-----|--------------|----------|
| ms/step | 91.60 | 97.2 | **80.5** |
| tok/s | 44715 | 42139 | **50856** |
| binders | 417 | 294 | **278** |

**Landed (kept):** GEMM v2 zero-tax pack + Morton; bf16 `execution_simdgroups<4>` /
MLP+bf16 TN split-K; Muon NS5 simdgroup; residency deferred commit; packed
`zero_grads`; wired `skip_resid_mix_rms_norm_scale_f32`; bump-slab recycle.

**Root cause of Phase 2 latency regress:** tiled FA-2 bwd (BR=BC=32) at T=256.
Defaults: row FA + Muon SG on + interior off.

### Audit 1 — first docs/opt plan

Assumed ~111 ms and that “finish BindList / Metal 4 encode” was still P0.
Themes still valid as principles: MPP TensorOps (**M2**), megakernels (**M4**),
banked Muon (**M5**), Core ML deploy-only (**M6**).

**Overturned by Audit 2:** “finish Metal 4 BindList encode” as P0.

### Audit 2 — encode already landed

Metal 4 BindList encode was already default (~91.6 vs ~110.9 ms). Re-ranking
focused on GEMM v2, runtime density, tiled FA (later gated off), megakernels.

### Audit 3 — measured binders + Muon miss

Binders **417** (not ~403); GEMM+zeros ≈ 42% of binders; Muon NS5 never used
TensorOps (hand GEMM inside banks); Soft EMA demoted f32-vs-bf16 BPB myth.

### Final build audit (Metal 3 removal)

Encode path Metal 4 only; no `--metal3`; historical M3 bench frozen.
