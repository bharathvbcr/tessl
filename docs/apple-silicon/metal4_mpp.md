# Metal 4 + MPP (agent guide)

Training encode is **Metal 4 only**. Shader dialect may still use
`-std=metal3.2` for non-TensorOps kernels — that is **not** classic M3
command buffers.

Related: [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) ·
[`optimization_map.md`](optimization_map.md) · [`mlx.md`](mlx.md).

## Primary sources

| Topic | URL |
|-------|-----|
| MPP Programming Guide (GEMM §2.3) | https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf |
| Metal Shading Language Spec (TensorOps) | https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf |
| WWDC26-330 — Metal tensors / TensorOps / flash recipe | https://developer.apple.com/videos/play/wwdc2026/330/ |
| Metal 4 overview | https://developer.apple.com/metal/ |
| Residency sets | Apple Metal residency / resource guide (queue-level `useResidencySet`) |

Ignore legacy README claims about dual Metal3/Metal4 encode paths.

---

## Metal4-only encode (non-negotiable)

| Piece | File | Rule |
|-------|------|------|
| Init hard-fail | [`runtime.rs`](../../src/runtime.rs) `GpuRuntime::new` | If M4 package init fails → **error**, no classic CB fallback |
| Binder | [`dispatch.rs`](../../src/dispatch.rs) | Argument table + ~1 MiB const arena only |
| Train CLI | [`bin/train.rs`](../../../ojas/arch02/metal-native/src/bin/train.rs) | No `--metal3` / `--metal4`; always `encode=Metal4` |
| Decision | [`DECISIONS.md`](../../../ojas/arch02/metal-native/DECISIONS.md) **M3** | Overturned to Metal 4 training encode only |
| Historical M3 bench | `out/bench_m4_vs_m3/m3/` ~110.9 ms | **Frozen** — cannot re-run |

**Encoder policy:** one **compute encoder** packed across `with_binder` calls
within a CB (Audit 4 P1). Packed ops call `barrier()` explicitly. M4 auto-inserts
Dispatch→Dispatch Device barriers after each `Binder::dispatch`. Telemetry still
counts `with_binder` invocations as binders.

### CounterHeap profiling stamps

Timestamps: `MTL4CounterHeap` t0/t1 on the **training** CB — not stamp-only CBs.
Do not trust `MTLCounterSampleBuffer` on macOS 26 (zeros).

---

## MPP §2.3 — DONE vs open

Apple GEMM checklist → our status on M5 Pro:

| § | Optimization | Status | Notes |
|---|--------------|--------|-------|
| 2.3.1 | Threadgroup tile size | **DONE** (shape-tuned) | [`matmul_tensorops.metal`](../../kernels/matmul_tensorops.metal) |
| 2.3.2 | Simdgroup tile size | **DONE** / gated | Exact f32 stays 32×32 / 1 SG; bf16+relaxed uses `execution_simdgroups<4>` |
| 2.3.3 | Morton walk order | **DONE** | Compact launch in [`gemm.rs`](../../src/gemm.rs) |
| 2.3.4 | BK / accumulate sync | **DONE** | BK=128 on bf16+relaxed; exact f32 conservative |
| 2.3.5 | Static tensor extents | **DONE** where shapes fixed | `slice<DIM,…>` |
| — | Packed zero+matmul | **DONE** | Killed double-binder tax |
| — | TN/NT + split-K (attn/MLP dW) | **DONE** | |
| — | Interior offset tiles | **CLOSED** | Measured ≤1 ms / slower — `METAL_NATIVE_GEMM_INTERIOR=1` only |
| 3.x coop **inputs** | Cooperative tensor **inputs** | **macOS 26.3+** API available | Prefer for new epilogues when shapes fit |
| 3.x coop **postfix** | MLP-up cooperative postfix | **OPEN / optional** | A/B only if epilogue still BW-bound after NAX util gate |
| Muon NS5 | simdgroup **inside** bank | **DONE** (keep on) | Not TensorOps; profile NAX util before rewrite |
| TensorOps flash | Probe only | **OPEN** | M8 blockers; no TO bwd; opt-in only |

Simdgroup GEMM in [`matmul_simdgroup.metal`](../../kernels/matmul_simdgroup.metal)
remains portable fallback — TensorOps is primary (DECISIONS **M2**).

---

## Residency (queue-level)

| Do | Don’t |
|----|-------|
| Register every allocation + const arena | Assume unified memory ⇒ no residency set |
| `useResidencySet` after `beginCommandBuffer` (queue-level) | Forget mid-step temps |
| Batch `addAllocations`; **`removeAllocation` when CB complete** | Retain forever (Audit 4 **P0 leak**) |
| Hot set (weights/opt) vs cold set (tape temps) | One unbounded set that grows +~0.4 GB/step |
| Recycle temps via `recycle_buffer` on Tensor drop | Assume bump slab alone is enough |

**Heap note:** making an `MTLHeap` resident makes the **whole heap** resident.
Private heaps / ICB stay ruled out. Fix the leak with remove/recycle + hot/cold
sets — do not “bring back MTLHeap” as the residency fix.

Host `GpuBuffer::zero` on bump views is unsafe (whole-slab memset). Mid-step
zeros → GPU `zero_f32`.

---

## NAX utilization gate

Apple M5 blog: compute-bound prefill gets big NAX wins; BW-bound work does not.

| Before doing… | Measure… |
|---------------|----------|
| TensorOps-inside-Muon | Instruments neural-accelerator util on Muon banks |
| MLP cooperative postfix default | Util + BW on MLP-up epilogue |
| TensorOps flash default | Util + quality; M8 still blocks |

Do not assume “more TensorOps ⇒ faster step.”

**How to measure (Instruments):**

1. Profile a short `--bench` run on device.
2. Inspect neural-accelerator / NAX utilization on Muon bank kernels, top GEMMs, FA.
3. Only port if util is low **and** the kernel is compute-bound (not memory-bound).
4. Record before/after in [`optimization_map.md`](optimization_map.md).

Automation of Instruments is optional; documenting the gate is mandatory.

---

## Cooperative postfix + WWDC26 flash

WWDC26 session 330 shows:

`matmul2d` → `reduce_rows` / `map_iterator` → left-input ·V

That recipe is **single-tile** fused attention, not multi-block online FA-2.

| Path | Status | Files |
|------|--------|-------|
| Production FA | simdgroup FA-2 fwd + LSE tape | [`flash_attn_fwd.metal`](../../../ojas/arch02/metal-native/kernels/flash_attn_fwd.metal) |
| Production bwd | **Row-wise default** at T=256; tiled gated off (+14–22 ms) | [`flash_attn_bwd.metal`](../../../ojas/arch02/metal-native/kernels/flash_attn_bwd.metal) |
| TensorOps FA | Probe / opt-in only (`flash_attn_tensorops_*`) | [`flash_attn_tensorops.metal`](../../../ojas/arch02/metal-native/kernels/flash_attn_tensorops.metal) |

Cooperative **inputs** (26.3+) are fair game for new fused epilogues.
Cooperative **postfix** on MLP-up stays behind flag until NAX/BW evidence.

Do **not** flip TensorOps flash to default (DECISIONS **M8**).

---

## Build hazards (encode vs shader)

| Issue | Note |
|-------|------|
| TensorOps AIR soft-fail in `build.rs` | Encode hard-requires M4; silent simdgroup-only metallib blocks NAX work |
| `METAL_NATIVE_SKIP_AOT` | Trusts crate-root `default.metallib` — do not set in CI |
| `-std=metal3.2` on non-TensorOps | Shader dialect only; **not** M3 encode |

---

## False opportunities (do not reopen)

Private heaps, ICB, multi-thread encode, MTLTensor host bindings for training,
host-side NS5 as many GEMM launches, restoring `--metal3`, default tiled FA bwd,
disable Muon simdgroup, f32 interior tiles, DualInputBuffers-as-step-win under
always-sync, cross-step async.
