---
id: "gp-nax-int4-tensorops-and-exact-f32-epilogue"
title: "Wire shader-side int4b_format TensorOps prefill on NAX and add cooperative ExactF32 GEMM epilogue"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "kernel"
  - "tensorops"
  - "nax"
  - "quantization"
  - "gemm"
  - "performance"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "src/mtl_tensor.rs"
  - "src/gemm.rs"
  - "src/nn.rs"
  - "kernels/matmul_tensorops.metal"
  - "kernels/gemm_coop.h"
  - "tests/mtl_tensor.rs"
acceptance_criteria:
  - "Author a shader-side sub-byte cooperative tensor constructor for uint4b_format / int4b_format on raw device pointers in kernels/, bypassing the missing host MTLTensorDataType::Int4 binding in objc2-metal 0.3 (src/mtl_tensor.rs:36-46, :78-108)"
  - "Wire native TensorOps Q4 prefill GEMM (flipping QUANT_PREFILL_GEMM_WIRED to true or exposing a dedicated nn::gemm_q4_tensorops entry point) and benchmark against hand-tuned simdgroup Q4 (gemm_q4_mlx_simd*) on M5 Pro to unblock DDTree speculative verification"
  - "Add a cooperative destination register epilogue path for ExactF32 in src/gemm.rs:933-941 and kernels/gemm_coop.h (or expose an explicit relaxed-f32 epilogue opt-in for ojas-metal) so f32 callers with bias + activation do not drop to unfused 32x32 1-simdgroup dispatches"
  - "Verify bit-exactness of the ExactF32 cooperative epilogue against separate GEMM + bias + activation passes across all 38 tile audit pairs in scripts/ci_local.sh"
---

# Task brief v1

## Title
Wire shader-side int4b_format TensorOps prefill on NAX and add cooperative ExactF32 GEMM epilogue

Task: gp-nax-int4-tensorops-and-exact-f32-epilogue
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: kernel, tensorops, nax, quantization, gemm, performance

## Repositories
- tessl
- ojas

## Description
Deep audit of `tessl` (0.2.0 @ `0b5f088`) and `ojas-metal` identified two high-leverage compute gaps in the Metal 4 MPP TensorOps pipeline:

1. **Unwired NAX `Int4` TensorOps Prefill (`src/mtl_tensor.rs:78-108`):** `QUANT_PREFILL_GEMM_WIRED` is currently `false` because `objc2-metal 0.3` does not bind `MTLTensorDataType::Int4` for host-created `MTLTensor` descriptors. However, as documented in `src/mtl_tensor.rs`, Metal 4 MPP TensorOps headers natively accept `uint4b_format` and `int4b_format` cooperative source types constructed from raw device pointers in shader code (the same pattern `nn::gemm_i8_dequant` already uses for `int8 x int8 -> int32`). Implementing the shader-side sub-byte tensor constructor unlocks Apple Neural Accelerator (NAX) hardware matmuls for Q4 prefill and `DDTree` verify(M) without waiting on `objc2-metal` SDK bindings.
2. **`ExactF32` Refusal in `gemm_epilogue` (`src/gemm.rs:933-941`, `ojas/AUDIT_REPORT.md` Risk #1 `G-V2 / E1`):** `ojas-metal` dispatches `GemmOperands::ExactF32`, which `gemm_epilogue` currently rejects ("Epilogue needs coop path") and routes to a 1-simdgroup 32x32 tile (`matmul2d_tensorops_f32`). Extending cooperative destination register accumulation and in-register bias/activation epilogues to `ExactF32` (and exposing a `tf32-relaxed` opt-in for `ojas` `Numerics::Fast`) removes the extra device memory round-trips that bottleneck `ojas-metal` linear and upcoming `conv2d` layers.

## Acceptance criteria
- [ ] Author a shader-side sub-byte cooperative tensor constructor for uint4b_format / int4b_format on raw device pointers in kernels/, bypassing the missing host MTLTensorDataType::Int4 binding in objc2-metal 0.3 (src/mtl_tensor.rs:36-46, :78-108)
- [ ] Wire native TensorOps Q4 prefill GEMM (flipping QUANT_PREFILL_GEMM_WIRED to true or exposing a dedicated nn::gemm_q4_tensorops entry point) and benchmark against hand-tuned simdgroup Q4 (gemm_q4_mlx_simd*) on M5 Pro to unblock DDTree speculative verification
- [ ] Add a cooperative destination register epilogue path for ExactF32 in src/gemm.rs:933-941 and kernels/gemm_coop.h (or expose an explicit relaxed-f32 epilogue opt-in for ojas-metal) so f32 callers with bias + activation do not drop to unfused 32x32 1-simdgroup dispatches
- [ ] Verify bit-exactness of the ExactF32 cooperative epilogue against separate GEMM + bias + activation passes across all 38 tile audit pairs in scripts/ci_local.sh

## Planned files
- src/mtl_tensor.rs
- src/gemm.rs
- src/nn.rs
- kernels/matmul_tensorops.metal
- kernels/gemm_coop.h
- tests/mtl_tensor.rs
