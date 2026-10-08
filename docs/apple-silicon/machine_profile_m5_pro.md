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
| GPU cores | **20** |
| Unified memory | **64 GB** |
| OS | **macOS 26.5.2** |
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
| Wired fraction | **~0.85–0.9 ×** recommended working set ≈ **~41–48 GB** on 64 GB |
| Pool cache cap | Cap recycled buffer pool; CLI override |
| Storage | All pool allocs `StorageModeShared`; const arena ~1 MiB; bump slab ~64 MiB mostly idle |

**Why this matters:** at B=16 the step is **binder / bwd-bound**, not
“out of RAM.” Wired + cache caps are primarily **stability + room for B=32/64**,
not a free FLOPs win at B=16.

### B targets

| Batch | Role |
|-------|------|
| **B=16** | Live gate / default |
| **B=32 / 64** | Throughput / RSS under wired policy; Hot residency still open |

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

### Still open (Audit 6)

| Gap | Impact |
|-----|--------|
| **P1a** bwd QKV fuse | **−48 binders** — top ROI |
| **P1a2** dW accum-into-bank | **−24–32 binders** (MLP/out/stem) |
| **P1b** resid/RMS bwd megakernels + skip_resid bf16 twin | **−16–20** |
| **P1c** tape `deep_copy` | **−14–16** |
| **P1d** hazard-aware Device barriers | 0 or ~2–8 ms (scheduling) |
| **Hot residency** (`alloc_buffer_hot` unused) | RSS / churn at B≥32 |
| **P0b** FineWeb B=32/64 rebench | Throughput docs |
| **P1e** bf16 activation / FA cast tax | ~5–15 ms if training bf16 |
| **P2** MLP coop postfix | Only after NAX util |
| **P3** FA → CUDA BPB ~1.994 | Quality |

Private heaps / ICB stay ruled out. Do not re-open Audit 4 P0/P0b/P1 as “missing.”
