# Apple-silicon reference docs (formerly MLSystemsLab `Rust_MLKit/docs`)

Agent-facing reference for optimizing
[`arch_02_value_resid/metal-native`](../../../ojas/arch02/metal-native/).
Ignore legacy README phase roadmaps as Apple-API truth; prefer these docs +
[`metal-native/DECISIONS.md`](../../../ojas/arch02/metal-native/DECISIONS.md).

| Doc | Use when |
|-----|----------|
| [`apple_ml_stack.md`](apple_ml_stack.md) | **Start here for Apple APIs.** Metal tensors, unified memory, Neural Accelerator vs Neural Engine vs SME, Core AI / Core ML / MLX, and the macOS 27.2 delta. Survey 2026-10-30. |
| [`gemm_architecture.md`](gemm_architecture.md) | GEMM kernel selection, the `use_coop_nn` gate, and how to verify and benchmark it |
| [`kernel_hardening.md`](kernel_hardening.md) | Tessl/Sparsl hardening verification, canonical performance-result links, measured clear removal, and publication gates |
| [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) | Host facts: M5 Pro 20 GPU / 64 GB / NAX / wired budgets / DONE vs open |
| [`mlx.md`](mlx.md) | MLX 0.32 cheat sheet (URLs → files; Audit 6 done vs steal next) |
| [`mlx_audit.md`](mlx_audit.md) | 2026-10-30 MLX tag versus `main`: NAX GEMM, affine Q4 versus fp4, SDPA head dim 256, GDN VJP. No new timings. |
| [`beta_nightly_gap.md`](beta_nightly_gap.md) | 2026-10-30 beta, nightly, and doc-gap check: no seed after 27.2, GPU cores 20, wired budget 0.9× the 48 GiB recommendation, mlx `ahead_by` 58, mlx-swift metal-cpp header is not `8cdde68`. |
| [`metal4_mpp.md`](metal4_mpp.md) | Metal 4 encode-only, MPP §2.3 DONE vs open, residency, NAX util gate |
| [`coreml_metal_ml.md`](coreml_metal_ml.md) | Deploy-only Core ML / ANE (QAT ≠ train) |
| [`optimization_map.md`](optimization_map.md) | **Start here for work** — live gates **58.4 / 276** + Audit 6 backlog |

## Stack root

```
../MLSystemsLab/research/parameter-golf/
  AGENTS.md                 ← points here
  docs/                     ← this folder
  crates/
    tessl/                  ← Metal 4 GEMM + encode runtime (owns the GEMM kernels and benches)
  gemma-metal/              ← Gemma inference; consumes tessl
  arch_02_value_resid/
    metal-native/           ← training hot path (Rust + MSL); `tessl-arch02`, depends on tessl
    burn-port/              ← frozen A/B reference (do not rewrite)
    mlx-baseline/           ← frozen (do not rewrite)
```

`sparsl` is a sibling checkout of this lab at `../../../sparsl`, not a crate in this tree.

## Hard rules for agents

1. **Encode is Metal 4 only** — no `--metal3`, no classic CB fallback.
2. **Do not implement opts from memory** — gate against
   [`optimization_map.md`](optimization_map.md) live numbers (**58.4 ms / 276 binders / Soft EMA BPB 2.038**). Audit 6 marks Hot residency DONE (`alloc_buffer_hot`). Open items stay in that map’s backlog.
3. **Cite Apple/MLX URLs** in PRs when claiming API behavior; do not invent from old READMEs.
4. Core ML / `MTL4MachineLearningCommandEncoder` = **inference only** (QAT ≠ train).
