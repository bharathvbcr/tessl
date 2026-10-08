# Apple-silicon reference docs (formerly MLSystemsLab `Rust_MLKit/docs`)

Agent-facing reference for optimizing
[`arch_02_value_resid/metal-native`](../../../ojas/arch02/metal-native/).
Ignore legacy README phase roadmaps as Apple-API truth; prefer these docs +
[`metal-native/DECISIONS.md`](../../../ojas/arch02/metal-native/DECISIONS.md).

| Doc | Use when |
|-----|----------|
| [`gemm_architecture.md`](gemm_architecture.md) | GEMM kernel selection, the `use_coop_nn` gate, and how to verify and benchmark it |
| [`kernel_hardening.md`](kernel_hardening.md) | Tessl/Sparsl hardening verification, canonical performance-result links, measured clear removal, and publication gates |
| [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) | Host facts: M5 Pro 20 GPU / 64 GB / NAX / wired budgets / DONE vs open |
| [`mlx.md`](mlx.md) | MLX 0.32 cheat sheet (URLs → files; Audit 6 done vs steal next) |
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
   [`optimization_map.md`](optimization_map.md) live numbers (**58.4 ms / 276 binders / Soft EMA BPB 2.038**). Audit 6 = next ROI (QKV fuse, dW accum, barriers, Hot residency).
3. **Cite Apple/MLX URLs** in PRs when claiming API behavior; do not invent from old READMEs.
4. Core ML / `MTL4MachineLearningCommandEncoder` = **inference only** (QAT ≠ train).
