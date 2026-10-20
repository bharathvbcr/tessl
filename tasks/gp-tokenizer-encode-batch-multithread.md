---
id: "gp-tokenizer-encode-batch-multithread"
title: "Spread tokenizer encode_batch across cores to match HF all-core throughput"
status: ready
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "tokenizer"
  - "perf"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files: []
acceptance_criteria:
  - "encode_batch splits texts across cores and produces token-ID hashes identical to the single-threaded path on the 200-document workload"
  - "On that workload, tessl matches or beats HF tokenizers' all-core encode_batch (6.62M tok/s in the audit run), measured in the same interleaved session"
  - "Thread count is bounded and configurable; single-text and empty-batch calls don't pay pool startup"
---

# Task brief v1

## Title
Spread tokenizer encode_batch across cores to match HF all-core throughput

Task: gp-tokenizer-encode-batch-multithread
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Source: research session 2026-10-20 (article audit gaps)

## Description
On 200 documents, HF's all-core `encode_batch` does 6.62M tok/s against tessl's 2.13M (0.32×), because tessl's batch encoder runs on one thread. The per-text speed is competitive, so the gap is in parallelism.

Related: `gp-eg2-quiet-evidence-reruns-and-native-hf-tokenizer-lane` adds the native HF tokenizer lane. Reuse that lane for the comparison instead of building a second one.

## Acceptance criteria
- [ ] encode_batch splits texts across cores and produces token-ID hashes identical to the single-threaded path on the 200-document workload
- [ ] On that workload, tessl matches or beats HF tokenizers' all-core encode_batch (6.62M tok/s in the audit run), measured in the same interleaved session
- [ ] Thread count is bounded and configurable; single-text and empty-batch calls don't pay pool startup
