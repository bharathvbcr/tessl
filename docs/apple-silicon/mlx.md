# MLX 0.32 — agent cheat sheet

Quick map from **MLX 0.32** + Apple Metal/CoreML primary URLs onto
[`arch_02_value_resid/metal-native`](../../../ojas/arch02/metal-native/).
We do **not** train with mlx-rs / Python MLX. Keep custom Muon, XSA,
value-resid, and clip-soft parity.

**Live gate:** **58.4 ms / 276 binders / ~70k tok/s** (B=16 f32). See
[`optimization_map.md`](optimization_map.md) (Audit 6).

Related: [`machine_profile_m5_pro.md`](machine_profile_m5_pro.md) ·
[`metal4_mpp.md`](metal4_mpp.md) · [`coreml_metal_ml.md`](coreml_metal_ml.md).

---

## Primary URLs → metal-native files

| Topic | URL | Steal as | Files |
|-------|-----|----------|-------|
| **MLX index** | [mlx docs](https://ml-explore.github.io/mlx/build/html/index.html) | Doctrine only — not a runtime | — |
| **`mx.compile`** | [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) | Fuse graphs → megakernels + encoder packing on **bwd** | [`model_bwd.rs`](../../../ojas/arch02/metal-native/src/model_bwd.rs), [`block_megakernel.metal`](../../../ojas/arch02/metal-native/kernels/block_megakernel.metal), [`runtime.rs`](../../src/runtime.rs) |
| **Memory limits** | [memory_management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html) | `set_wired_limit` / cache → probe WS + pool cap | [`runtime.rs`](../../src/runtime.rs), [`bin/train.rs`](../../../ojas/arch02/metal-native/src/bin/train.rs) |
| **Unified memory** | [unified_memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | Shared buffers; streams ≠ skip residency / barriers | [`runtime.rs`](../../src/runtime.rs), [`dispatch.rs`](../../src/dispatch.rs), [`tensor.rs`](../../src/tensor.rs) |
| **`mx.fast.*`** | [fast](https://ml-explore.github.io/mlx/build/html/python/fast.html) | Fused FA / RoPE / RMS; `math_mode` ↔ `--tf32` | [`flash_attn_fwd.metal`](../../../ojas/arch02/metal-native/kernels/flash_attn_fwd.metal), [`qkv_post.metal`](../../../ojas/arch02/metal-native/kernels/qkv_post.metal) |
| **Transforms** | [transforms](https://ml-explore.github.io/mlx/build/html/python/transforms.html) | `checkpoint` / recompute → kill tape `deep_copy` | [`tape.rs`](../../../ojas/arch02/metal-native/src/tape.rs), [`model_fwd.rs`](../../../ojas/arch02/metal-native/src/model_fwd.rs) |
| **Custom Metal** | [custom_metal_kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) | AOT metallib once; never silent skip in release | [`build.rs`](../../../ojas/arch02/metal-native/build.rs) |
| **Metal** | [Metal](https://developer.apple.com/documentation/metal) | M4 encode, residency, barriers | [`dispatch.rs`](../../src/dispatch.rs), [`runtime.rs`](../../src/runtime.rs) |
| **MPP / TensorOps** | [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), WWDC26-330 | GEMM primary; coop postfix only after NAX util | [`matmul_tensorops.metal`](../../kernels/matmul_tensorops.metal), [`gemm.rs`](../../src/gemm.rs) |
| **Core ML** | [Core ML](https://developer.apple.com/documentation/coreml/) | Deploy / export only — never train | [`scripts/export_coreml.py`](../../../ojas/arch02/metal-native/scripts/export_coreml.py), [`coreml_metal_ml.md`](coreml_metal_ml.md) |

---

## Audit 6 — already done vs steal next

Ignore stale pre–Audit 4 “Missing” narrative (residency leak, no wired probe,
1-encoder-per-binder, 80.5/278 as live gate).

### Already done (do not re-litigate)

| Item | Status |
|------|--------|
| Metal 4 BindList encode | **DONE** — hard-fail if M4 init fails |
| TensorOps / Morton / packed zero+matmul GEMM | **DONE** |
| Audit 4 P0 cold recycle + `removeAllocation` | **DONE** — RSS stable |
| Audit 4 P0b wired / WS probe / pool cache | **DONE** |
| Audit 4 P1 packed compute encoder | **DONE** — **58.4 ms / 276** |
| Persistent bf16 weight banks | **DONE** (activations still cast; see P1e) |
| Fwd resid/RMS megakernels; banked Muon NS5 | **DONE** |
| Sync every step | **DONE** — do not revive cross-step async |

### Steal next (Audit 5 fusion + Audit 6 levers)

| Pri | Work | Est. | MLX / Apple analogy | Where |
|-----|------|------|---------------------|-------|
| **P1a** | Fuse bwd QKV | **−48 binders** | `mx.compile` graph fuse | [`model_bwd.rs`](../../../ojas/arch02/metal-native/src/model_bwd.rs) ~619–641 |
| **P1a2** | dW `multiply_accumulate` into bank views | **−24–32 binders** | Fuse away temp+`accum_bank` | [`model_bwd.rs`](../../../ojas/arch02/metal-native/src/model_bwd.rs) `accum_bank`; [`matmul_tensorops.metal`](../../kernels/matmul_tensorops.metal) `mode::multiply_accumulate` |
| **P1b** | Mirror resid/RMS bwd megakernels + skip_resid bf16 twin | **−16–20** | Same compile lesson on bwd | bwd ~279–363, ~644–676 |
| **P1c** | Kill tape `deep_copy` | **−14–16** | `checkpoint` / recompute | [`model_fwd.rs`](../../../ojas/arch02/metal-native/src/model_fwd.rs) ~411–546 |
| **P1d** | Hazard-aware Device barriers A/B | **0 or ~2–8 ms** | Stream deps without always-on serialize | [`dispatch.rs`](../../src/dispatch.rs) ~93–104 |
| **P0b+** | Wire `BufferKind::Hot` residency | RSS at B≥32 | Hot weights/opt vs cold tape | [`runtime.rs`](../../src/runtime.rs) `alloc_buffer_hot` (dead today) |
| **P1e** | Cut bf16 activation / FA cast tax | ~5–15 ms (bf16 only) | Persistent acts path | banks refresh + per-GEMM casts |
| **P2** | MLP coop postfix | Speculative | MPP coop destination | NAX util gate only |
| **P3** | FA Soft EMA 2.038 → ~1.994 | Quality | Not step-time | FA numerics |

**Top ROI order:** QKV fuse → dW accum-into-bank → resid/RMS bwd + deep_copy →
barrier A/B → Hot residency + B=32/64 → bf16 casts (if training bf16) → coop/FA.

---

## One-liners agents need

1. **Unified memory ≠ skip residency** — register, `useResidencySet`, recycle +
   `removeAllocation`. Hot/cold split still open (`alloc_buffer_hot` unused).
2. **`compile` ≈ megakernel + packed binder** — fwd packed; bwd still densest
   (~38 ms of 58.4). Finish QKV / resid / dW fuse before “more NAX.”
3. **Always-on Dispatch→Dispatch Device barrier** after every `Binder::dispatch`
   may over-serialize independent work — A/B carefully (P1d).
4. **dW today** = temp `dw` + `accum_bank` (`add_inplace`). TensorOps already
   has `multiply_accumulate` into bank views — use it (P1a2).
5. **Wired/cache** unlock B=32/64 stability, not B=16 FLOPs.
6. **Core ML / ANE / ML encoder** = inference only.

---

## Anti-patterns (keep closed)

| Mistake | Why |
|---------|-----|
| Train with mlx-rs / Core ML / ANE | Custom Muon, FA bwd, clip-soft need MSL |
| Host Muon GEMMs / explode banks | Defeats bank fusion |
| DualInputBuffers / ICB / MTLHeap / Metal 3 restore | False leads |
| Cross-step async | Poisoned ~step 2100 |
| Default TensorOps flash / tiled FA bwd | M8 / measured regress |
| Trust stale gates (91.60/417, **80.5/278**) | Live: **58.4 / 276 / BPB 2.038** |
| “More NAX” without Instruments util | Binder/bwd-bound at B=16 |
| `METAL_NATIVE_MLP_COOP_POSTFIX` without NAX gate | Documented, never read — wire last |
