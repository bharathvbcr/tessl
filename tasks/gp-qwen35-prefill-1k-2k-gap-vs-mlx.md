---
id: "gp-qwen35-prefill-1k-2k-gap-vs-mlx"
title: "Close (or profile to a cause) MLX's Qwen3.5 prefill lead at 1k-2k tokens"
status: ready
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "mlx"
  - "prefill"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "docs/qwen35.md"
acceptance_criteria:
  - "Paired tessl vs mlx-lm Qwen3.5-2B prefill at 1k and 2k tokens: identical token ids, interleaved, equal samples, quiet machine"
  - "Gap closed (tessl >= MLX at 1k and 2k), or a stage-level profile names the cause (the earlier run had GEMM stages 10-20% slower)"
  - "docs/qwen35.md replaces 'unresolved until re-run' with the measured result"
---

# Task brief v1

## Title
Close (or profile to a cause) MLX's Qwen3.5 prefill lead at 1k-2k tokens

Task: gp-qwen35-prefill-1k-2k-gap-vs-mlx
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Source: research session 2026-10-20 (article audit gaps)

## Description
MLX leads Qwen3.5 prefill at 1k and 2k tokens, with 7,606 and 7,528 tok/s. `docs/qwen35.md` calls tessl's 1k/2k gap "unresolved until re-run", and in that run the GEMM stages were 10–20% slower.

Scope: `gp-qwen35-vs-current-torch-and-mlx-head-to-head` only measures the gap. This task closes it. Use that task's paired run as the starting point if it has landed.

## Acceptance criteria
- [ ] Paired tessl vs mlx-lm Qwen3.5-2B prefill at 1k and 2k tokens: identical token ids, interleaved, equal samples, quiet machine
- [ ] Gap closed (tessl >= MLX at 1k and 2k), or a stage-level profile names the cause (the earlier run had GEMM stages 10-20% slower)
- [ ] docs/qwen35.md replaces "unresolved until re-run" with the measured result
