# AGENTS.md

Guidance for coding agents (and people) changing `tessl`.

## Role & Architecture

`tessl` is the **dedicated Apple Silicon / Metal 4 acceleration extension of [ojas](https://github.com/bharathvbcr/ojas)** and the high-throughput **Mac inference engine**.

1. **Metal Extension of ojas:** Provides the Metal 4 runtime, unified memory allocator (`SharedBuf`), and canonical compute kernels (`DEP_TESSL_KERNELS`) that `ojas-metal` drives for training and general ops.
2. **Mac Inference Engine:** Hand-tuned Qwen3.5 prefill and decode, decode ICB and ping-pong replay, KV caches, MLX Q4/Q8 quantization, EmbeddingGemma 2 multimodal towers, BERT, and fused kernels.

Two sibling checkouts are path dependencies and must exist for any build:
- `../ojas` (github.com/bharathvbcr/ojas): Universal training framework (path dependency for `ojas-core` HAL traits and `ojas-io`).
- `../../devtools/gusset` (github.com/bharathvbcr/gusset): In-process FFI bridge for Go bindings.

## Invariants a change must keep

- **Metal 4 Only:** Built strictly on Metal 4 primitives (`MTL4CommandBuffer`, `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable`, `MTLResidencySet`). Legacy `MTLCommandQueue` and command buffer paths are deliberately absent.
- **Zero Host Wait:** Packed command encoding with bump-allocated constant arenas (16 MiB) and `MTLSharedEvent` synchronization—host threads never block mid-step.
- **Unified Memory Safety via `SharedBuf`:** `SharedBuf` (`src/shared_buf.rs`) implements `ojas_core::MappedBuffer` over pooled `StorageModeShared` allocations. Host access acquires RAII leases (`HostLease`, `HostLeaseMut`) synchronized with GPU completion stamps (`last_use`, `last_write`) on the `Timeline`. Leases prevent concurrent GPU binds (`FenceError::Busy`). Memory allocations are tracked under `ojas_core::Budget` with `BudgetTrimHandler` clearing freelist caches under memory pressure.
- **No Generic Training Logic:** Optimizer scalar formation, gradient clipping policy, learning-rate schedules, loss scaling, checkpoint schemas, and autograd tape live in `ojas`. A step is driven through `ojas-model::TapeStepProvider`.
- **Fail-Closed Verification:** Entry points validate extents, bounds, and dtypes before command encoding (`validate_gemm`, `nn::require`). Failures return typed errors or descriptive error messages; they never fail silently or corrupt GPU state.

## Testing & Verification

```sh
cargo test --release -- --test-threads=1
cargo check
```

- Platform requirement: macOS 26+ with Apple Silicon GPU (M-series) and Xcode 26 Metal Toolchain.
- Run GPU tests with `--test-threads=1` to prevent GPU memory and command queue contention.
