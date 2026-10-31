# Machine profile — M5 Pro (training host)

Agent reference for the **live** training machine. Gate numbers and memory
budgets assume this profile unless a note says otherwise.

Related: [`optimization_map.md`](optimization_map.md) · [`metal4_mpp.md`](metal4_mpp.md) ·
[`mlx.md`](mlx.md).

---

## Hardware / OS

| Item | Value |
|------|-------|
| Chip | **Apple M5 Pro** |
| GPU cores | **20**, measured 2026-10-30 (`system_profiler SPDisplaysDataType` “Total Number of Cores: 20”; `ioreg` `gpu-core-count` = 20) |
| Unified memory | **64 GB** |
| OS | **macOS 27.0.1** (`26A434`), measured 2026-10-30. macOS 27.2 is offered and not installed; its developer notes have no Metal or Core ML delta. See [`apple_ml_stack.md`](apple_ml_stack.md). |
| Encode | Metal 4 only (hard-fail if M4 init fails) |
| NAX | Present — TensorOps GEMM is the correct primary path |

Apple’s public M5 vs M4 “3–4× TTFT” figures are **inference / prefill**
comparisons across chip generations. They are **not** free step-time headroom
below our live gate on this already-M5 host.

---

## Live training gate (B=16)

| Metric | Value |
|--------|-------|
| Step time | **58.4 ms/step** (was 80.5 pre–Audit 4 pack) |
| Throughput | **~70.2k tok/s** |
| Binders / step | **276** |
| Soft EMA BPB (f32) | **~2.038** |
| Phase wall (profiled, post-pack) | fwd ~13 ms · **bwd ~38 ms** · optim ~8 ms |

Default FineWeb bench shape: **B=16**, T=256, L=4, f32 masters unless noted.

---

## Memory / wired policy

MLX-style wired budgets map cleanly onto 64 GB unified memory:

| Knob | Guidance |
|------|----------|
| `recommendedMaxWorkingSetSize` | Probe at `GpuRuntime` init; log in train banner |
| Wired fraction | Default **0.9 ×** `recommendedMaxWorkingSetSize` (`src/runtime.rs` assemble). On this boot that is 46,385,646,796 bytes (43.2 GiB of the 48 GiB recommendation). `set_wired_fraction` stores a finite fraction clamped to 0.5–0.95. The arch02 banner prints that figure. It is not an admission cap, and it does not change `iogpu.wired_limit_mb` |
| Pool cache cap | Cap recycled buffer pool; CLI override |
| Storage | All pool allocs `StorageModeShared`; const arena **16 MiB** (`METAL4_CONST_ARENA_BYTES` in `src/runtime.rs`; 1 MiB exhausted mid-token); bump slab ~64 MiB mostly idle |

**Why this matters:** at B=16 the step is **binder / bwd-bound**, not
“out of RAM.” Wired + cache caps are primarily **stability + room for B=32/64**,
not a free FLOPs win at B=16.

### B targets

| Batch | Role |
|-------|------|
| **B=16** | Live gate / default |
| **B=32 / 64** | Throughput / RSS under wired policy; Hot residency is wired (`alloc_buffer_hot`) |

---

## NAX utilization gate

Further TensorOps ports (Muon inner GEMM rewrite, MLP cooperative postfix,
default TensorOps flash) must be gated on **Instruments neural-accelerator
utilization**, not on “M5 has NAX ⇒ must win.”

| Workload class | Expectation |
|----------------|-------------|
| Compute-bound prefill / large GEMM | NAX wins are real |
| Bandwidth-bound epilogues / small banks | May not move |

Measure on Muon banks, top training GEMMs, and FA before rewriting. See
[`metal4_mpp.md`](metal4_mpp.md) and [`optimization_map.md`](optimization_map.md) P2.

---

## DONE vs open

### DONE (correct for this machine)

| Item | Notes |
|------|-------|
| TensorOps / Morton / packed zero+matmul GEMM | Primary GEMM path |
| Metal 4 BindList encode | No M3 fallback |
| **P0** cold residency recycle + `removeAllocation` | RSS ~683 MB stable |
| **P0b** WS probe + wired fraction + pool cache | Banner / CLI live |
| **P1** packed compute encoder + dead-add skip | **58.4 ms / 276** |
| **P1b** persistent bf16 weight banks | Activations still cast (P1e) |
| Fwd resid/RMS megakernels; banked Muon NS5 | Simdgroup inside bank |
| Row-wise FA bwd @ T=256 | Tiled FA gated off |
| Soft EMA BPB ~2.038 f32 / ~2.037 bf16 | — |

### Still open

The status table is Audit 6 in [`optimization_map.md`](optimization_map.md).
These are the rows that table does not mark DONE:

| Item | Map status |
|------|------------|
| **P0b** FineWeb B=32/64 rebench | Still open. Wired limit, working-set probe, and pool cache are DONE. Board: `gp-fineweb-b32-b64-rebench` |
| **P1e** bf16 activation casts | **PARTIAL** |
| **P2** MLP coop postfix | **DEFER** until a neural-accelerator utilization measurement |
| **P3** FA quality | **PARTIAL** |

P1a, P1a2, P1b, and P1c are **DONE** on that table. P1d is **REJECT as default**.
Hot residency is **DONE**. Gate numbers stay on the map.

Private heaps / ICB stay ruled out. Do not re-open Audit 4 P0/P0b/P1 as “missing.”
