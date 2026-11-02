# Accelerated Engineering Workflows & GPU Execution Protocol

This document establishes the canonical development, testing, and GPU execution workflows for `tessl` and its role as the Metal 4 inference engine and backend provider for `ojas` and `SpikeTrack`.

---

## Architecture Context

In the unified system architecture:
- `ojas-metal` delegates compute to `tessl` runtime kernels and shared memory pools.
- `tessl` specializes in Apple Silicon Metal 4 runtime execution, `StorageModeShared` unified memory pools, and optimized matrix kernels (TensorOps, ICB decode, EmbeddingGemma 2 multimodal towers).
- Dependency flow: `ojas-metal -> tessl -> ojas-core`.

---

## 1. Tiered Testing Pipeline

Tessl enforces a multi-stage testing pipeline designed to fail fast on syntax and host-contract checks before dispatching long-running GPU benchmarks.

| Tier | Gate | Scope | Command | Hardware Req |
|---|---|---|---|---|
| **Tier 0** | Fast Inner Loop (< 15s) | Incremental check, optional clippy, targeted unit test | `./scripts/dev_fast_check.sh [test_filter]` (`--check-only`, `--clippy`) | Host / serialized 1-thread (< 15s) |
| **Tier 1** | Fast Formatting & Compile | Workspace compile & clippy | `cargo check --all-targets && cargo clippy --release --all-targets -- -D warnings` | Host only (< 15s) |
| **Tier 2** | Device-Free Hosted Tests | Tests run under sandbox denying IOKit (`scripts/hosted_tests.sh`) | `./scripts/hosted_tests.sh` | Host only (< 30s) |
| **Tier 3** | Metal Dialect & IR Check | MSL syntax, fast-math bounds, no Inf/NaN constants | `./scripts/check_kernel_ir.sh` | Metal toolchain |
| **Tier 4** | Emulator & Contract Verification | Python MSL emulator contract checks vs PyTorch reference | `python3 tools/msl_emu/check_qwen35.py -k host_contract` | PyTorch oracle env |
| **Tier 5** | Serialized GPU Tests & Examples | Device execution on Metal under exclusive lock | `./scripts/ci_local.sh` (or Lappi-decision's `tools/mac_heavy.sh`) | Apple Silicon Metal |

---

## 2. Metal Execution Locking (`mac_heavy.sh`)

### Problem
Concurrent GPU execution across multiple coding agents or benchmark runs causes GPU memory thrashing, distorted timing metrics, allocator contention, and risk of macOS kernel panics under heavy unified memory pressure.

### The Machine-Wide Lock Contract
All heavy GPU jobs (Metal compilation, tensor tests, benchmarks, and multi-round evaluations) must be wrapped by the machine's heavy lock:

```bash
# Path to lock script
/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh <label> <command> [args...]
```

### Invariants Maintained
1. **Atomic Directory Lock**: Acquires an atomic lock directory (`mac-heavy.lock.d`). Waiters take FIFO tickets (`.q`) and never starve.
2. **Process Footprint Cap (`phys_footprint`)**: Monitors the job's entire process tree via `proc_pid_rusage`. If footprint exceeds threshold (default: 4 GiB below memguard threshold), it sends `SIGTERM` followed by `SIGKILL` to prevent unrecoverable OS lockups.
3. **Machine Load & Process Count Gate**: Refuses or waits while 1-minute load average exceeds 12 or active process count exceeds 2,500.
4. **Environment Propagation**: Exports `MAC_HEAVY_LABEL=<label>`. Tessl's `./scripts/ci_local.sh` checks for `MAC_HEAVY_LABEL`:
   - If present, GPU-dependent Python and Go tests execute.
   - If absent, GPU tests are cleanly skipped with an explicit notification rather than falsely reporting success.
5. **Thread Serialization**: Metal tests that query allocator statistics or device buffer gauges must be invoked with `--test-threads=1`.

---

## 3. Worktree Collision Management

### Multi-Workspace Topology
Edits in `tessl` directly impact downstream consumers:
- `ojas-metal` (path dependency `../tessl`)
- `SpikeTrack` (depends on `ojas` and `tessl`)

### Collision Prevention Protocol
1. **Check Collision Risk Before Editing**:
   Before modifying shared headers or runtime components (`src/runtime.rs`, `src/tensor.rs`, `kernels/`):
   ```bash
   # Via GitPulse MCP tool:
   gitpulse_collision_risk(repo_path: "/Users/bharath/Code/research/tessl")
   ```
2. **Review In-Flight Worktrees**:
   Use `gitpulse_insights` to identify other active worktrees holding uncommitted edits.
3. **No Breaking Interface Drift**:
   When updating C ABI (`src/capi.rs`) or HAL trait bindings, verify compatibility with `ojas-metal` before landing changes.
