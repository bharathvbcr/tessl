---
id: "gp-bench-evidence-holes"
title: "Bench evidence holes: GDN training run without --evidence, EG2 audio kernels without reference checks, unchecked gemm_variants lanes"
status: ready
priority: 3
severity: low
type: bench
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "evidence"
  - "eg2"
  - "gdn"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files: []
acceptance_criteria:
  - "The chunked GDN training run redone with --evidence so the code state it ran on is recorded, and the result replaces the evidence-less one"
  - "The three EmbeddingGemma 2 audio kernels get host reference checks (not only finite/range checks)"
  - "bench_gemm_variants lanes either check their output against a reference or are documented as timing-only with the test that covers their correctness named"
---

# Task brief v1

## Title
Bench evidence holes: GDN training run without --evidence, EG2 audio kernels without reference checks, unchecked gemm_variants lanes

Task: gp-bench-evidence-holes
Type: bench
Status: ready
Priority: 3 (Low)
Severity: low
Source: research session 2026-10-20 (article audit gaps)

## Description
Three places where a recorded or running bench proves less than it appears to:
- The chunked GDN training run was recorded without `--evidence`, so the code state it ran on wasn't captured.
- Three EmbeddingGemma 2 audio kernels get only finite and range checks, and nothing compares them against a host reference.
- The `bench_gemm_variants` lanes only time. They don't check anything themselves.

## Acceptance criteria
- [ ] The chunked GDN training run redone with --evidence so the code state it ran on is recorded, and the result replaces the evidence-less one
- [ ] The three EmbeddingGemma 2 audio kernels get host reference checks (not only finite/range checks)
- [ ] bench_gemm_variants lanes either check their output against a reference or are documented as timing-only with the test that covers their correctness named
