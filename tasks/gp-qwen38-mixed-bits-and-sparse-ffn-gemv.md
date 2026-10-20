---
id: "gp-qwen38-mixed-bits-and-sparse-ffn-gemv"
title: "Qwen3.8-27B formats MLX can't run: per-tensor mixed bits and TEAL-style activation-sparse FFN GEMV"
status: backlog
priority: 2
severity: low
type: research
owner: "unassigned"
due: "none"
labels:
  - "quant"
  - "sparsity"
  - "decode"
  - "27b"
  - "research"
  - "research-2026-10-20"
repositories:
  - "tessl"
  - "sparsl"
  - "LocalModelBench"
planned_files:
  - "docs/qwen35.md"
acceptance_criteria:
  - "Per-tensor bit allocation supported; recipes evaluated: (a) 4-bit everywhere, (b) gate/up 3-bit with down, attention, ssm_out and head at 4 or more bits, (c) a third chosen from (b)'s sensitivity sweep"
  - "Quality reference is Qwen/Qwen3.8-27B-FP8 (or the 8-bit MLX build): mean KLD on a fixed corpus plus the LocalModelBench agent task set; bounds written before the first run"
  - "A column-blocked FFN weight packing and Metal GEMV that skips weight blocks for activations under a per-layer calibrated threshold, at 30%, 40% and 50% sparsity, bit-exact against a dense reference with the same zeroed activations"
  - "For each recipe and sparsity: quality, decode tok/s paired against the 4-bit dense tessl path, and bytes per token"
  - "A Pareto table (tok/s vs quality) committed under bench/results/, with arms that lose recorded"
---

# Task brief v1

## Title
Qwen3.8-27B formats MLX can't run: per-tensor mixed bits and TEAL-style activation-sparse FFN GEMV

Task: gp-qwen38-mixed-bits-and-sparse-ffn-gemv
Type: research
Status: backlog
Priority: 2
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
Depends on `gp-qwen35-q4-weights-and-qwen38-27b`.

Decode at 27B is bytes-bound. MLX at 4-bit already reaches about 80–88% of the 307 GB/s peak (a third-party rMLX profile puts it at about 1.24x the roofline). The remaining lever is fewer bytes per token.

These projections are arithmetic at 250–270 GB/s achieved, to be tested:
- FFN 3-bit, rest 4-bit: 12.5 GB/token, 20–22 tok/s
- gate/up 3-bit with down kept at 4-bit (Unsloth found ffn_down more sensitive): 13.2 GB, 19–20 tok/s
- 40% activation sparsity on the FFN at 4-bit: 10.8 GB, 23–25 tok/s

TEAL reports 1.53x decode at 40% and 1.8x at 50% sparsity on Llama and Mistral (7–70B), with CUDA kernels. R-Sparse costs 1.5–3.2 points at 50% on 7–8B models. Neither has been tested on hybrid DeltaNet models or on Metal.

## Acceptance criteria
- [ ] Per-tensor bit allocation supported; recipes evaluated: (a) 4-bit everywhere, (b) gate/up 3-bit with down, attention, ssm_out and head at 4 or more bits, (c) a third chosen from (b)'s sensitivity sweep
- [ ] Quality reference is Qwen/Qwen3.8-27B-FP8 (or the 8-bit MLX build): mean KLD on a fixed corpus plus the LocalModelBench agent task set; bounds written before the first run
- [ ] A column-blocked FFN weight packing and Metal GEMV that skips weight blocks for activations under a per-layer calibrated threshold, at 30%, 40% and 50% sparsity, bit-exact against a dense reference with the same zeroed activations
- [ ] For each recipe and sparsity: quality, decode tok/s paired against the 4-bit dense tessl path, and bytes per token
- [ ] A Pareto table (tok/s vs quality) committed under bench/results/, with arms that lose recorded
