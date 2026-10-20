---
id: "gp-bf16-gemm-losing-shapes-vs-mps"
title: "bf16 GEMM: re-run against torch 2.15 MPS and close or explain the shapes below 0.95x"
status: ready
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "gemm"
  - "bf16"
  - "torch-2.15"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "README.md"
acceptance_criteria:
  - "The 8-shape bf16 GEMM comparison re-run against torch 2.15 on MPS, interleaved, min-of-N, on a quiet machine, results committed under bench/results/"
  - "No shape below 0.95x, or each losing shape documented in the README with a profiled cause"
  - "README's bf16 table updated from the 2026-09-01 torch 2.13 numbers (1.03x geomean, 4 of 8 shapes losing, 0.86x at 8192x3072x768)"
---

# Task brief v1

## Title
bf16 GEMM: re-run against torch 2.15 MPS and close or explain the shapes below 0.95x

Task: gp-bf16-gemm-losing-shapes-vs-mps
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Source: research session 2026-10-20 (article audit gaps)

## Description
The README's bf16 GEMM row (2026-09-01, torch 2.13) shows a 1.03× geomean against PyTorch MPS. tessl loses on 4 of 8 shapes, worst at 0.86× on 8192×3072×768. That's below parity on half the shapes, and the comparison predates torch 2.15.

Run GPU benches under the `mac_heavy` lock.

## Acceptance criteria
- [ ] The 8-shape bf16 GEMM comparison re-run against torch 2.15 on MPS, interleaved, min-of-N, on a quiet machine, results committed under bench/results/
- [ ] No shape below 0.95x, or each losing shape documented in the README with a profiled cause
- [ ] README's bf16 table updated from the 2026-09-01 torch 2.13 numbers (1.03x geomean, 4 of 8 shapes losing, 0.86x at 8192x3072x768)
