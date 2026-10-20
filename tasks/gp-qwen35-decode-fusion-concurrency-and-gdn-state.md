---
id: "gp-qwen35-decode-fusion-concurrency-and-gdn-state"
title: "Qwen3.5 decode: safe fusion, memory-range barriers, a GPU-resident loop, and one read and write per GDN state"
status: ready
priority: 2
severity: low
type: perf
owner: "unassigned"
due: "none"
labels:
  - "decode"
  - "fusion"
  - "metal"
  - "gdn"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "src/qwen35.rs"
  - "src/dispatch.rs"
  - "docs/qwen35.md"
acceptance_criteria:
  - "Fused add+RMSNorm and gate/up GEMV+SiLU kernels for decode; dispatches per token before and after recorded from the trace"
  - "Hazard-tracked memory-range barriers in place of blanket barriers between independent decode dispatches, with the ordering proven by the existing hazard tests"
  - "The 3 per-token cold allocations and the per-token host wait are removed (GPU-resident decode loop), with a test that pins zero allocations per step"
  - "The recurrent GDN decode kernel is shown to read and write each state matrix once per token (counted bytes); a bf16-state variant is measured only behind a quality gate (KL against fp32 state on fixed prompts)"
  - "Paired A/B tok/s on Qwen3.5-2B (and on Qwen3.8-27B once gp-qwen35-q4-weights-and-qwen38-27b lands) with exactness or bounds passing before any speed is quoted"
---

# Task brief v1

## Title
Qwen3.5 decode: safe fusion, memory-range barriers, a GPU-resident loop, and one read and write per GDN state

Task: gp-qwen35-decode-fusion-concurrency-and-gdn-state
Type: perf
Status: ready
Priority: 2
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
Proven Apple-GPU gains come from fusion and fewer barriers, not megakernels:
- llama.cpp RMS_NORM+MUL+ADD fusion: +7–9% on M4 Max
- llama.cpp concurrency pass with memory-range barriers: about +20% on M2 Ultra
- MLX with fewer command-buffer boundaries: +5–8%

Qwen3.5-2B decode today runs 273 dispatches, 1 commit, 1 wait and 3 cold allocations per token (qwen35_model_prefill_decode_m5pro.txt). GDN recurrent state is 64 KiB per head in fp32; at 27B shapes that is 144 MiB per token, read and written.

## Acceptance criteria
- [ ] Fused add+RMSNorm and gate/up GEMV+SiLU kernels for decode; dispatches per token before and after recorded from the trace
- [ ] Hazard-tracked memory-range barriers in place of blanket barriers between independent decode dispatches, with the ordering proven by the existing hazard tests
- [ ] The 3 per-token cold allocations and the per-token host wait are removed (GPU-resident decode loop), with a test that pins zero allocations per step
- [ ] The recurrent GDN decode kernel is shown to read and write each state matrix once per token (counted bytes); a bf16-state variant is measured only behind a quality gate (KL against fp32 state on fixed prompts)
- [ ] Paired A/B tok/s on Qwen3.5-2B (and on Qwen3.8-27B once gp-qwen35-q4-weights-and-qwen38-27b lands) with exactness or bounds passing before any speed is quoted
