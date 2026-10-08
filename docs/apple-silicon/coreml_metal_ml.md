# Core ML / Metal ML — deploy only

Training stays custom MSL in `metal-native`. Core ML, ANE, and
`MTL4MachineLearningCommandEncoder` are **inference / packaging** tools.

**QAT ≠ train.** Quantization-aware training, palettization passes, and Core ML
compute-unit assignment do **not** replace the Metal 4 training loop (Muon,
custom FA bwd, clip-soft, value-resid). They apply to **exported** graphs only.

Related: [`optimization_map.md`](optimization_map.md) ·
[`DECISIONS.md`](../../../ojas/arch02/metal-native/DECISIONS.md) **M6**.

## Primary sources

| Topic | URL |
|-------|-----|
| Core ML | https://developer.apple.com/documentation/coreml |
| coremltools | https://apple.github.io/coremltools/docs-guides/ |
| Metal / ML command encoder (inference on GPU timeline) | https://developer.apple.com/documentation/metal |
| WWDC26-330 (TensorOps → Core AI packaging context) | https://developer.apple.com/videos/play/wwdc2026/330/ |

---

## What agents may do

| Task | Where |
|------|-------|
| Export EMA / golden weights → `.mlpackage` | [`scripts/export_coreml.py`](../../../ojas/arch02/metal-native/scripts/export_coreml.py) |
| Palettize / int8 / int6 packages | same; outputs under `out/coreml_export/` |
| Stateful KV **Torch** reference | `--stateful-kv` → `arch02_sota_decode_step.pt`; see `out/coreml_export/STATEFUL_KV_CORE_AI.md` |
| Bench ANE / CPU_AND_NE forward | Core ML compute-unit sweeps (README numbers are deploy benches) |
| Quantized TensorOps GEMM for **deploy** kernels | Later P4 — export path only |

## What agents must not do

| Forbidden | Why |
|-----------|-----|
| Train with Core ML / ANE | Cannot express Muon, custom FA bwd, on-device clip-soft |
| Put `MTL4MachineLearningCommandEncoder` on the training hot path | Inference-only API |
| Treat QAT / palettize as a training BPB or step-time fix | Deploy graph opts ≠ training binder tax |
| Block training P0–P3 on Core ML `StateType` convert | Still blocked (dynamic slice lowering) — track separately |
| Use metal-package-builder ML passes to “optimize” [`gemm.rs`](../../src/gemm.rs) / optim | Wrong layer |

---

## metal-package-builder / ML passes

Apple’s packaging / graph-optimization passes (const folding, palettization,
compute-unit assignment) apply to **exported** models. Map them only when:

1. Training BPB/step-time gates are already green (**58.4 ms / 276 / BPB 2.038**), and
2. The change is in export scripts or `.mlpackage` post-process — not the training hot path.

---

## Quick export recipe (reference)

```bash
# From metal-native/; Python 3.12 + coremltools
python scripts/export_coreml.py \
  --weights golden/weights_init \
  --out out/coreml_export --seq-len 256
```

Packages: `arch02_sota_fp16.mlpackage`, `*_int8_palettized.mlpackage`,
`*_int6_palettized.mlpackage`.
