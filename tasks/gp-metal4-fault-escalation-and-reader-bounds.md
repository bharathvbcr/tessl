---
id: "gp-metal4-fault-escalation-and-reader-bounds"
title: "Escalate Metal 4 wait timeouts to sticky DeviceLost, drain commit feedback, and cap checkpoint reader payloads"
status: ready
priority: 2
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "faults"
  - "runtime"
  - "safetensors"
  - "io"
  - "security"
  - "metal4"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "src/runtime.rs"
  - "src/gemm.rs"
  - "src/safetensors.rs"
  - "src/npy.rs"
  - "src/loader.rs"
  - "tests/runtime_lifecycle.rs"
acceptance_criteria:
  - "Escalate repeated 30 s Metal command-buffer wait timeouts in src/runtime.rs to a sticky poisoned / DeviceLost state after N = 3 consecutive timeouts instead of stalling forever (G-GPUFAULT / M-1)"
  - "Consume MTL4CommitFeedback pending error queues on synchronize and validate GEMM extents in src/gemm.rs before allocating output buffers (G-GPUFAULT / M-2)"
  - "Enforce a bounded payload cap in checkpoint and safetensors readers so a sparse 512 GiB file is refused before filling memory and triggering memguard kills (ft-51395822)"
  - "Validate .npy header padding, make shape-overflow checks independent of dimension order, and detect post-open file truncation or size changes before reading stale offsets (ft-51395822)"
  - "Convert remaining load-time panicking host-access write_f32 and raw contents() sites (loader.rs:123, bert.rs:665/690/760/769, qwen35_model.rs:777, runtime.rs:1744, tensor.rs:360/386) and remove the receiver-name 'st' exemption in tests/host_access_ban.rs (ft-f5d97431)"
---

# Task brief v1

## Title
Escalate Metal 4 wait timeouts to sticky DeviceLost, drain commit feedback, and cap checkpoint reader payloads

Task: gp-metal4-fault-escalation-and-reader-bounds
Type: bug
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: faults, runtime, safetensors, io, security, metal4

## Repositories
- tessl
- ojas

## Description
Consolidates the GPU fault-boundary gaps (`G-GPUFAULT` in `ojas/AUDIT_REPORT.md`) and reader/host-access safety follow-ups (`ft-51395822` and `ft-f5d97431` in `tessl/BUILD_REPORT.md`):

1. **GPU Wait Timeout & Commit Feedback (`G-GPUFAULT` / `M-1`, `M-2`):** A hung GPU currently repeats a 30 s wait timeout forever without transitioning to `DeviceLost`, and retrying an in-place operation outside the repo applies it twice. Furthermore, GEMM extent validation runs after buffer allocation.
2. **Checkpoint & NPY Reader Bounds (`ft-51395822`):** Checkpoint readers currently have no payload ceiling (allowing a sparse 512 GiB file to exhaust unified memory until `memguard` kills the process), `.npy` padding bytes go unvalidated, shape-overflow checks depend on dimension order, and a file rewritten after `open` is read with stale offsets.
3. **Remaining Unchecked Host-Access Sites (`ft-f5d97431`):** `tests/host_access_ban.rs` still exempts calls on receivers named `st`, and load-time `write_f32` / raw `contents()` sites remain in `loader.rs`, `bert.rs`, `qwen35_model.rs`, `runtime.rs`, and `tensor.rs`.

## Acceptance criteria
- [ ] Escalate repeated 30 s Metal command-buffer wait timeouts in src/runtime.rs to a sticky poisoned / DeviceLost state after N = 3 consecutive timeouts instead of stalling forever (G-GPUFAULT / M-1)
- [ ] Consume MTL4CommitFeedback pending error queues on synchronize and validate GEMM extents in src/gemm.rs before allocating output buffers (G-GPUFAULT / M-2)
- [ ] Enforce a bounded payload cap in checkpoint and safetensors readers so a sparse 512 GiB file is refused before filling memory and triggering memguard kills (ft-51395822)
- [ ] Validate .npy header padding, make shape-overflow checks independent of dimension order, and detect post-open file truncation or size changes before reading stale offsets (ft-51395822)
- [ ] Convert remaining load-time panicking host-access write_f32 and raw contents() sites (loader.rs:123, bert.rs:665/690/760/769, qwen35_model.rs:777, runtime.rs:1744, tensor.rs:360/386) and remove the receiver-name 'st' exemption in tests/host_access_ban.rs (ft-f5d97431)

## Planned files
- src/runtime.rs
- src/gemm.rs
- src/safetensors.rs
- src/npy.rs
- src/loader.rs
- tests/runtime_lifecycle.rs
