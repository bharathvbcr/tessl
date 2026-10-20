---
id: "gp-eg2-quiet-evidence-reruns-and-native-hf-tokenizer-lane"
title: "EmbeddingGemma 2: quiet evidence runs for the text fusion and vision tower, a native Rust tokenizers lane, and torch device memory"
status: ready
priority: 2
severity: low
type: bench
owner: "unassigned"
due: "none"
labels:
  - "embedgemma2"
  - "bench"
  - "tokenizer"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "docs/embedgemma2.md"
  - "bench/results/embedgemma2_tok_m5pro.txt"
  - "bench/embedgemma2_torch.py"
acceptance_criteria:
  - "`bench_paired --evidence --rounds 6 --max-spread 1.25 --clean-env` for the text fusion on an idle host (1-minute load < 6), published by the gate or refused with its numbers"
  - "A native Rust `tokenizers` crate lane, built in a separate bench crate so tessl gains no dependency, is added beside the Python lanes; all lanes rerun on a quiet machine with medians and best rounds; token-id hashes still equal"
  - "The vision tower bf16 vs ExactF32 rerun on a quiet machine, drift-gated"
  - "The torch lane of embedgemma2_vs_torch records MPS driver memory and current allocated memory per workload"
---

# Task brief v1

## Title
EmbeddingGemma 2: quiet evidence runs for the text fusion and vision tower, a native Rust tokenizers lane, and torch device memory

Task: gp-eg2-quiet-evidence-reruns-and-native-hf-tokenizer-lane
Type: bench
Status: ready
Priority: 2
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
Three EmbeddingGemma 2 claims still rest on loaded-machine runs or unmatched baselines:
- **Text-encoder fusion:** no end-to-end speedup is claimed. The paired evidence run was refused twice by its drift gate at load 66–108 (docs/embedgemma2.md, "Speed (text-encoder fusion)").
- **Tokenizer:** compared against Hugging Face tokenizers' *Python* bindings at load 110–135; only best rounds are reported.
- **Vision tower bf16:** 2.8–2.9x on clean rounds of a loaded run.

The vs-torch text run (embedgemma2_vs_torch_m5pro.txt) recorded no torch device memory.

## Acceptance criteria
- [ ] `bench_paired --evidence --rounds 6 --max-spread 1.25 --clean-env` for the text fusion on an idle host (1-minute load < 6), published by the gate or refused with its numbers
- [ ] A native Rust `tokenizers` crate lane, built in a separate bench crate so tessl gains no dependency, is added beside the Python lanes; all lanes rerun on a quiet machine with medians and best rounds; token-id hashes still equal
- [ ] The vision tower bf16 vs ExactF32 rerun on a quiet machine, drift-gated
- [ ] The torch lane of embedgemma2_vs_torch records MPS driver memory and current allocated memory per workload
