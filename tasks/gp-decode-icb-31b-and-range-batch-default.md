---
id: "gp-decode-icb-31b-and-range-batch-default"
title: "Complete 31B decode-graph ICB migration and enable coarse range-batched DecodeIcb replay by default"
status: ready
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "icb"
  - "metal4"
  - "decode"
  - "performance"
  - "runtime"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "src/decode_icb.rs"
  - "src/cb_replay.rs"
  - "src/dispatch.rs"
  - "src/runtime.rs"
  - "tests/icb_smoke.rs"
acceptance_criteria:
  - "Audit all kernel bind sites in the 31B decode path and replace raw Binder::bind_resource_id calls with owned GpuBuffer binds (bind_tensor / bind_gpu_buf_offset) so DecodeIcbCommand::incomplete_binds == 0 across the full 31B decode graph (src/decode_icb.rs:19-25, :93-109)"
  - "Benchmark GEMMA_METAL_ICB_FREEZE_BINDS=1 and v0.5.9 coarse range-batching (GEMMA_METAL_ICB_RANGE_BATCH=1, eliding barrier_after across disjoint buffer sets into a single executeCommandsInBuffer:withRange:) on E4B and 31B decode loops"
  - "Enable decode_icb_enabled() and coarse range-batching by default once paired A/B decode tok/s on M5 Pro confirms a net throughput win with zero hazard regressions across the 16-cell ICB correctness matrix (ft-9ac2a619)"
  - "Refuse re-encoding an MTLIndirectCommandBuffer while an execute_icb is still in flight on the GPU (closing the documented-only caveat in src/icb_smoke.rs:182 / ft-c125d853)"
---

# Task brief v1

## Title
Complete 31B decode-graph ICB migration and enable coarse range-batched DecodeIcb replay by default

Task: gp-decode-icb-31b-and-range-batch-default
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: icb, metal4, decode, performance, runtime

## Repositories
- tessl
- ojas

## Description
Wave 7 (`ft-9ac2a619`) fixed the ICB hazard capture race by recording `barrier_after` markers at capture time and replaying them deterministically. However, `src/decode_icb.rs` (lines 6-25, 42-55) leaves `decode_icb_enabled()`, `GEMMA_METAL_ICB_FREEZE_BINDS=1`, and `GEMMA_METAL_ICB_RANGE_BATCH=1` (including v0.5.9 coarse range-batching across disjoint `Buf` sets) disabled by default (`OFF`) while 31B decode-graph migration remains open.

Completing the transition from raw `MTLResourceID` binds to owned `GpuBuffer` binds (`incomplete_binds == 0`) ensures the replay tape pins every operand `Arc` against freelist recycling and unlocks zero-host-reencode ICB token decoding by default.

## Acceptance criteria
- [ ] Audit all kernel bind sites in the 31B decode path and replace raw Binder::bind_resource_id calls with owned GpuBuffer binds (bind_tensor / bind_gpu_buf_offset) so DecodeIcbCommand::incomplete_binds == 0 across the full 31B decode graph (src/decode_icb.rs:19-25, :93-109)
- [ ] Benchmark GEMMA_METAL_ICB_FREEZE_BINDS=1 and v0.5.9 coarse range-batching (GEMMA_METAL_ICB_RANGE_BATCH=1, eliding barrier_after across disjoint buffer sets into a single executeCommandsInBuffer:withRange:) on E4B and 31B decode loops
- [ ] Enable decode_icb_enabled() and coarse range-batching by default once paired A/B decode tok/s on M5 Pro confirms a net throughput win with zero hazard regressions across the 16-cell ICB correctness matrix (ft-9ac2a619)
- [ ] Refuse re-encoding an MTLIndirectCommandBuffer while an execute_icb is still in flight on the GPU (closing the documented-only caveat in src/icb_smoke.rs:182 / ft-c125d853)

## Planned files
- src/decode_icb.rs
- src/cb_replay.rs
- src/dispatch.rs
- src/runtime.rs
- tests/icb_smoke.rs
