---
id: "gp-wave10-hosted-manifest-and-kernel-bench-coverage"
title: "Fix gp/wave10 hosted test manifest completeness, merge wave10 to main, and close kernel_coverage.py gaps"
status: ready
priority: 1
severity: high
type: chore
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "merge"
  - "benchmarks"
  - "coverage"
  - "wave10"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "scripts/hosted_tests.txt"
  - "tests/hosted_completeness.rs"
  - "scripts/kernel_coverage.py"
  - "src/lib.rs"
  - "src/capi.rs"
  - "src/qwen35.rs"
acceptance_criteria:
  - "Add the four missing device-free CLI tests (bench_attn_cli: attn_kernel_refuses_bad_shapes_workloads_and_counts_before_any_metric, train_op_filter_is_validated_before_the_runtime_in_every_mode; bench_qwen35_layers_cli: attn_tiles_refuses_every_other_lane_and_the_model_options, attn_tiles_refuses_iters_it_would_not_honour) to scripts/hosted_tests.txt so every_device_free_test_is_on_the_hosted_manifest passes (BRANCH_REPORT.md)"
  - "Run scripts/hosted_tests.sh and scripts/ci_local.sh on gp/wave10 and merge gp/wave10 into tessl main so ojas gp/w10-readers (ft-02256073) and ft-dc1fa0a3 AC3 are unblocked"
  - "Rebase eg2/integrate (5229a1c) onto the new tessl main, resolving conflicts across src/lib.rs, src/capi.rs, src/qwen35.rs, and scripts/hosted_tests.txt, and clean up the merged w8/w9/w10-readers and worktree-agent-* branches"
  - "Resolve ft-6d88defb in scripts/kernel_coverage.py --check (150 of 266 xcrun metal -E kernel entry points currently lack a benchmark lane) by adding benchmark lanes for uncovered promoted families and an explicit documented allowlist for internal helper entry points"
---

# Task brief v1

## Title
Fix gp/wave10 hosted test manifest completeness, merge wave10 to main, and close kernel_coverage.py gaps

Task: gp-wave10-hosted-manifest-and-kernel-bench-coverage
Type: chore
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: ci, merge, benchmarks, coverage, wave10

## Repositories
- tessl
- ojas

## Description
Consolidates the immediate branch integration, CI manifest, and kernel benchmark coverage gaps documented in `tessl/BRANCH_REPORT.md` (2026-10-16) and `tessl/BUILD_REPORT.md` (`ft-6d88defb`):

1. **Hosted Manifest Failure Blocking `gp/wave10` Merge:** `every_device_free_test_is_on_the_hosted_manifest` (`tests/hosted_completeness.rs:189`) failed on `tessl-int` because four device-free CLI validation tests in `bench_attn_cli` and `bench_qwen35_layers_cli` are missing from `scripts/hosted_tests.txt`. This blocks `gp/wave10` from landing on `tessl` `main`, which in turn blocks `ojas` `gp/w10-readers` (`CheckpointView::tensor_info`, `ft-02256073`) and `ft-dc1fa0a3` AC3.
2. **`eg2/integrate` Rebase & Worktree Cleanup:** Once `gp/wave10` lands on `main`, `eg2/integrate` (`5229a1c`, 64 commits ahead) must be rebased across `src/lib.rs`, `src/capi.rs`, `src/qwen35.rs`, and `scripts/hosted_tests.txt`, retiring 26 merged worktree branches.
3. **Kernel Benchmark Coverage (`ft-6d88defb`):** `scripts/kernel_coverage.py --check` (using `xcrun metal -E`) reports 150 of 266 kernel entry points are not dispatched by any benchmark.

## Acceptance criteria
- [ ] Add the four missing device-free CLI tests (bench_attn_cli: attn_kernel_refuses_bad_shapes_workloads_and_counts_before_any_metric, train_op_filter_is_validated_before_the_runtime_in_every_mode; bench_qwen35_layers_cli: attn_tiles_refuses_every_other_lane_and_the_model_options, attn_tiles_refuses_iters_it_would_not_honour) to scripts/hosted_tests.txt so every_device_free_test_is_on_the_hosted_manifest passes (BRANCH_REPORT.md)
- [ ] Run scripts/hosted_tests.sh and scripts/ci_local.sh on gp/wave10 and merge gp/wave10 into tessl main so ojas gp/w10-readers (ft-02256073) and ft-dc1fa0a3 AC3 are unblocked
- [ ] Rebase eg2/integrate (5229a1c) onto the new tessl main, resolving conflicts across src/lib.rs, src/capi.rs, src/qwen35.rs, and scripts/hosted_tests.txt, and clean up the merged w8/w9/w10-readers and worktree-agent-* branches
- [ ] Resolve ft-6d88defb in scripts/kernel_coverage.py --check (150 of 266 xcrun metal -E kernel entry points currently lack a benchmark lane) by adding benchmark lanes for uncovered promoted families and an explicit documented allowlist for internal helper entry points

## Planned files
- scripts/hosted_tests.txt
- tests/hosted_completeness.rs
- scripts/kernel_coverage.py
- src/lib.rs
- src/capi.rs
- src/qwen35.rs
