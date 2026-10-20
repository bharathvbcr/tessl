---
id: "gp-qwen35-q4-weights-and-qwen38-27b"
title: "MLX-format 4-bit weights in the Qwen3.5 tower, and load and decode Qwen3.8-27B at MLX parity"
status: ready
priority: 1
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "quant"
  - "decode"
  - "27b"
  - "research-2026-10-20"
repositories:
  - "tessl"
  - "LocalModelBench"
planned_files:
  - "src/qwen35.rs"
  - "src/qwen35_model.rs"
  - "docs/qwen35.md"
  - "src/bin/bench_qwen35_layers.rs"
acceptance_criteria:
  - "Packed projection windows accept MLX affine 4-bit (group 64, scale and bias) weights, loaded directly from mlx-community/Qwen3.8-27B-4bit safetensors without a bf16 intermediate; host peak footprint during load is recorded"
  - "The GDN chunk and recurrent kernels are verified (or extended) for 16 QK / 48 V heads and 64 layers; full attention for 24/4 heads of 256 with partial RoPE 0.25; untied 248,320-row head"
  - "Correctness against mlx-lm on the same 4-bit weights over fixed prompts: top-1 agreement and mean KL per position recorded, with bounds written before the first run"
  - "Decode tok/s paired and interleaved against mlx-lm 0.32.x on the same weights (tg256 at 1K and 8K context, quiet machine): within 5% of MLX, or the gap is profiled into kernel time and gaps"
  - "Bytes per token and achieved GB/s reported, with dispatches and waits per token from the trace"
---

# Task brief v1

## Title
MLX-format 4-bit weights in the Qwen3.5 tower, and load and decode Qwen3.8-27B at MLX parity

Task: gp-qwen35-q4-weights-and-qwen38-27b
Type: feature
Status: ready
Priority: 1
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
tessl's Qwen3.5 decode is already about as bandwidth-efficient as MLX:
- tessl, Qwen3.5-2B bf16: about 3.8 GB at 17.3 ms/token ≈ 217 GB/s (bench/results/qwen35_model_prefill_decode_m5pro.txt).
- MLX, the same model at 4-bit: 208.7 tok/s ≈ 221 GB/s (BaseRT paper, M5 Pro).

The gap is bytes: the Qwen3.5 path has no quantized weights.

Qwen3.8-27B uses the same class (`Qwen3_5ForConditionalGeneration`):
- 64 layers, 48 GatedDeltaNet and 16 gated-attention layers
- hidden 5120, FFN 17408
- GDN: 16 QK heads and 48 V heads of 128
- attention: 24 q heads and 4 kv heads of 256, partial RoPE 0.25
- vocab 248,320, untied head

MLX decodes it at 15.3–16.9 tok/s at 4-bit on this M5 Pro (LocalModelBench 2026-08-16/17, mlx-vlm path). tessl already has MLX-format Q4 GEMV kernels (Q4MlxBank) and an opt-in TensorOps Q4 GEMM.

This task establishes parity. The formats that only tessl can run come after, in `gp-qwen38-mixed-bits-and-sparse-ffn-gemv`. Prefill on TensorOps Q4 is tracked in `gp-nax-int4-tensorops-and-exact-f32-epilogue`; reuse it, don't duplicate it.

## Acceptance criteria
- [ ] Packed projection windows accept MLX affine 4-bit (group 64, scale and bias) weights, loaded directly from mlx-community/Qwen3.8-27B-4bit safetensors without a bf16 intermediate; host peak footprint during load is recorded
- [ ] The GDN chunk and recurrent kernels are verified (or extended) for 16 QK / 48 V heads and 64 layers; full attention for 24/4 heads of 256 with partial RoPE 0.25; untied 248,320-row head
- [ ] Correctness against mlx-lm on the same 4-bit weights over fixed prompts: top-1 agreement and mean KL per position recorded, with bounds written before the first run
- [ ] Decode tok/s paired and interleaved against mlx-lm 0.32.x on the same weights (tg256 at 1K and 8K context, quiet machine): within 5% of MLX, or the gap is profiled into kernel time and gaps
- [ ] Bytes per token and achieved GB/s reported, with dispatches and waits per token from the trace
