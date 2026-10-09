# Documentation

Comprehensive architectural references, kernel documentation, benchmarks, and platform guides for `tessl`.

---

## Core Guides

| Document | Topic & Scope |
|---|---|
| [**Architecture**](architecture.md) | Deep dive into Metal 4 execution, kernel selection, cooperative destination register accumulation, memory hierarchies, and tile swizzling. |
| [**Benchmarking**](benchmarking.md) | Measurement methodology, paired sweep protocol, thermal and clock drift mitigations, and performance comparison against PyTorch MPS and MLX. |
| [**Verification**](verification.md) | Correctness guarantees: static tile geometry audits, shape fuzzing, CPU emulation (`tools/msl_emu`), and numerical parity ladders. |

---

## Models & Architectures

| Document | Topic & Scope |
|---|---|
| [**Qwen3.5**](qwen35.md) | Gated Delta Net (GDN) chunked prep and scan, read-only recurrent decode, training backward pass, vocabulary-chunked cross-entropy, and fused AdamW. |
| [**EmbeddingGemma 2**](embedgemma2.md) | Full text encoder execution from SafeTensors: symmetric sliding-window attention, per-layer inputs, mean pooling, and Matryoshka dimension truncation. |
| [**BERT & DistilBERT**](bert.md) | Learned sparse document encoders (`BertForMaskedLM` / `DistilBertForMaskedLM`): fused LayerNorms, exact erf GELU, and segment sparse max reduction. |
| [**PyTorch Interop**](../python/README.md) | `tessl_torch` C ABI bindings: drop-in cross-entropy, GDN seam patching, and full model `train_step` integration. |

---

## Apple Silicon Hardware & Optimization

The [`docs/apple-silicon/`](apple-silicon/README.md) directory contains specialized reference documentation:

| Document | Focus |
|---|---|
| [`machine_profile_m5_pro.md`](apple-silicon/machine_profile_m5_pro.md) | Apple M5 Pro hardware characteristics, GPU core counts, bandwidth, and cache topology. |
| [`gemm_architecture.md`](apple-silicon/gemm_architecture.md) | Low-level GEMM layout mechanics, $K$-partitioning, and cooperative tile selection. |
| [`metal4_mpp.md`](apple-silicon/metal4_mpp.md) | Metal 4 modern encode pipeline, argument tables, residency sets, and MPP TensorOps. |
| [`kernel_hardening.md`](apple-silicon/kernel_hardening.md) | Kernel robustness, bounds-checking guarantees, and memory hazard prevention. |
| [`optimization_map.md`](apple-silicon/optimization_map.md) | Ongoing optimization milestones, audit tracking, and runtime budgets. |
| [`mlx.md`](apple-silicon/mlx.md) | Architectural comparison with Apple's MLX framework. |
| [`coreml_metal_ml.md`](apple-silicon/coreml_metal_ml.md) | Trade-offs between Metal 4 compute, Core ML, and Apple Neural Engine (ANE). |
