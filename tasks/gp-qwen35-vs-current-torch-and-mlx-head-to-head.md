---
id: "gp-qwen35-vs-current-torch-and-mlx-head-to-head"
title: "Re-run Qwen3.5 head-to-heads against current PyTorch and MLX: prefill+score, and the GDN training patch"
status: ready
priority: 1
severity: medium
type: bench
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "torch"
  - "mlx"
  - "qwen35"
  - "research-2026-10-20"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "python/tests/test_gdn.py"
  - "docs/qwen35.md"
acceptance_criteria:
  - "Prefill+score rerun: same real Qwen3.5-2B snapshot, identical token ids on every side, equal sample counts, lengths {200, 1k, 2k, 8k} interleaved, against torch 2.14.1 and 2.15 (RC or GA) on MPS and against mlx-lm 0.32.x; quiet machine, per-lane min and median"
  - "GDN training patch: a transformers Qwen3.5 forward+backward on MPS (2B config cut to 4 layers, or full 0.8B) with and without tessl_torch.patch_transformers_qwen3_5(), alternating; step time and MPS driver memory recorded; gradients match within test_gdn.py's bounds"
  - "Results committed under bench/results/ with every lane where tessl loses reported, including MLX at short lengths if it still leads"
---

# Task brief v1

## Title
Re-run Qwen3.5 head-to-heads against current PyTorch and MLX: prefill+score, and the GDN training patch

Task: gp-qwen35-vs-current-torch-and-mlx-head-to-head
Type: bench
Status: ready
Priority: 1
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
The 2.12–2.22x Qwen3.5-2B prefill+score result against PyTorch (ojas/bench/results/2026-10-03-lappi-inference-rerun/research-notes.md) has these limits:
- **Update 2026-10-23:** `bench/results/qwen35_prefill_score_h2h_m5pro.txt` (Oct 23) supersedes the 2.12–2.22x figure with 4.47–5.35x vs torch 2.14.1 (T=200 to 8192), and its paired run has mlx-lm 0.32 10–16% faster than tessl at every length, which also supersedes the unpaired 8k MLX comparison below. The limits listed here describe the old Oct 4 run.
- it ran against torch **2.12.1**
- one hand-run session
- 7 vs 10 samples per side
- different token streams (shapes matched)

The MLX comparison at 8k (tessl ~6.2k tok/s vs MLX 5,285) is unpaired, and MLX leads at 1k–2k (about 7.5k tok/s).

There is also no measured "PyTorch user swaps one function" number for `tessl_torch.patch_transformers_qwen3_5()`.

## Acceptance criteria
- [ ] Prefill+score rerun: same real Qwen3.5-2B snapshot, identical token ids on every side, equal sample counts, lengths {200, 1k, 2k, 8k} interleaved, against torch 2.14.1 and 2.15 (RC or GA) on MPS and against mlx-lm 0.32.x; quiet machine, per-lane min and median
- [ ] GDN training patch: a transformers Qwen3.5 forward+backward on MPS (2B config cut to 4 layers, or full 0.8B) with and without tessl_torch.patch_transformers_qwen3_5(), alternating; step time and MPS driver memory recorded; gradients match within test_gdn.py's bounds
- [ ] Results committed under bench/results/ with every lane where tessl loses reported, including MLX at short lengths if it still leads
