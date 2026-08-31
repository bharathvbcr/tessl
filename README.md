<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/png/logo-dark@900.png">
    <img src="assets/png/logo@900.png" alt="tessl" width="420">
  </picture>
</p>

<p align="center">
  <strong>Low-overhead, zero-host-wait Metal 4 GEMM and encode runtime for Apple silicon.</strong><br>
  Powered by Metal Performance Primitives (MPP) TensorOps <code>matmul2d</code>.
</p>

---

`tessl` is a Rust GPU runtime substrate that executes high-performance matrix multiplication on Apple silicon through Metal 4 and Metal Performance Primitives (MPP) `matmul2d`, targeting the neural accelerators on Apple M-series hardware.

The name is short for *tessellation* — the design centers around how matrix operations are partitioned into tile geometries and the order in which those tiles are traversed.

---

## Key Highlights

- **Pure Metal 4 Architecture:** Built strictly on Metal 4 primitives (`MTL4CommandBuffer`, `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable`, `MTLResidencySet`). Legacy `MTLCommandQueue` and command buffer paths are deliberately absent.
- **Hardware-Accelerated GEMM:** Direct integration with MPP TensorOps `matmul2d` across NN, TN, and NT layouts in `f32`, `bf16` (with `f32` accumulate), and `tf32-relaxed` precision modes.
- **Cooperative Register Accumulators:** High-throughput cooperative destination kernels (`get_destination_cooperative_tensor`) holding `f32` accumulators in GPU registers across the entire $K$-reduction, eliminating device memory round-trips for NN, TN, NT, and accumulating paths.
- **In-Kernel Grid Swizzling & Bounds Checking:** Column-panel tile swizzling for large grids ($\ge 2048$ tiles) bounding operand rereads, combined with origin-shifted slice bounds checking for ragged edges.
- **Zero-Wait Execution Pipeline:** Packed command encoding with bump-allocated constant arenas (16 MiB) and `MTLSharedEvent` synchronization—host threads never block mid-step.
- **Decode ICB Capture & Replay:** Low-latency Indirect Command Buffer (ICB) capture and ping-pong execution with freeze-binds and range-batching for decode-shaped inference workloads.

> [!IMPORTANT]
> **Platform Requirements:**
> - **OS:** macOS 26+
> - **Toolchain:** Xcode 26 with the Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`).
> - **Hardware:** Apple Silicon GPU with Neural Accelerators (Apple M-series) for the MPP TensorOps path. A portable `simdgroup_matrix` fallback is available for A/B testing, but is 2–3× slower.

---

## System Architecture

```mermaid
graph TD
    subgraph Consumers["Downstream Consumers"]
        Gemma["gemma-metal<br/>(Gemma 4 Inference)"]
        Arch02["tessl-arch02<br/>(Value Residual Training)"]
    end

    subgraph TesslAPI["tessl Public API"]
        GpuRt["GpuRuntime"]
        GemmFn["gemm() / gemm_f32()"]
        TensorObj["Tensor / GpuBuffer"]
        IcbObj["DecodeIcb / PingPongCbReplay"]
    end

    subgraph CoreEngine["tessl Core Runtime Substrate"]
        RuntimeMod["runtime.rs<br/>MTL4 Buffers, Pools & Const Arena"]
        GemmMod["gemm.rs<br/>Validation, Layouts & Coop Dispatch"]
        DispatchMod["dispatch.rs<br/>Binder & Argument Table Encode"]
        IcbMod["decode_icb.rs / cb_replay.rs<br/>ICB Capture, Tape Replay & Coalescing"]
        MtlTensorMod["mtl_tensor.rs<br/>Quantized MTLTensor Prep (WWDC26-330)"]
    end

    subgraph Metal4Layer["Metal 4 Driver & Hardware Layer"]
        CmdBuf["MTL4CommandBuffer / Allocator"]
        ArgTable["MTL4ArgumentTable (32-slot)"]
        ResSet["MTLResidencySet (Hot / Cold Pools)"]
        SharedEvt["MTLSharedEvent (Zero-wait Sync)"]
    end

    subgraph Shaders["Compiled Metallib Shaders"]
        TensorOpsMetal["matmul_tensorops.metal (MPP matmul2d)"]
        SimdMetal["matmul_simdgroup.metal (Fallback)"]
        UtilsMetal["utils.metal (Elementwise & Softcap)"]
    end

    Gemma -->|Links & Overlays| TesslAPI
    Arch02 -->|DEP_TESSL_KERNELS| TesslAPI
    TesslAPI --> CoreEngine
    CoreEngine --> Metal4Layer
    Metal4Layer --> Shaders
```

---

## Performance vs. PyTorch MPS & MLX

Measurements taken on Apple M5 Pro utilizing `bench/paired_cross_runtime.py`. The benchmark harness interleaves `tessl` and PyTorch MPS iterations round-by-round to cancel GPU thermal throttling and frequency scaling drift (see [Benchmarking](docs/benchmarking.md)):

*Geomean of per-shape medians over 5 rounds across an 8-shape ladder:*

| Comparison | vs. Baseline | Worst Shape | Best Shape | Peak Throughput (M5 Pro) |
|---|---|---|---|---|
| **bf16 vs. PyTorch MPS bf16** | **1.11×** *(Outperforms MPS)* | 1.01× | 1.22× | **29,022 GFLOP/s** (`square_4096`) |
| **f32 exact vs. PyTorch MPS f32** | **1.07×** | 0.92× | 1.47× | **10,897 GFLOP/s** (`square_2048`) |
| **tf32-relaxed vs. PyTorch MPS f32** | **2.01×** | 1.49× | 2.90× | **18,040 GFLOP/s** (`square_4096`) |
| **bf16 vs. MLX bf16** | **2.63×** | 1.13× | 3.64× | — |

> [!WARNING]
> **Benchmarking Rigor:**
> - **Clock Drift:** Single-run cross-runtime benchmarks can fluctuate by 15–20% on identical workloads due to Apple Silicon dynamic power governor adjustments. Always use paired, interleaved sweeps (`bench_gemm_coop_ab` or `paired_cross_runtime.py`).
> - **Dispatch Floor:** Below ~2 GFLOP of total work, both runtimes hit a ~0.25 ms host submit-and-wait floor, measuring host driver dispatch latency rather than raw shader throughput.

---

## Metal 4 Memory & Residency Hierarchy

`tessl` manages GPU memory allocations explicitly to eliminate mid-command buffer host stalls and memory thrashing.

```mermaid
flowchart TD
    subgraph DeviceMemory["Unified System Memory (Metal 4 Device)"]
        subgraph Pools["tessl Managed Pools"]
            Hot["Hot Pool<br/>(Weights & Persistent State)<br/>Resident for lifetime of run"]
            Cold["Cold Pool<br/>(Intermediate Activations)<br/>Recycled + removeAllocation after CB"]
            Bump["Bump Pool<br/>(Per-step Ephemeral Slabs)<br/>Cursor reset on sync"]
        end

        subgraph Arenas["Low-Latency Arenas"]
            ConstArena["Constant Arena (16 MiB Bump)<br/>Scalar & Uniform Table Offsets"]
        end
    end

    subgraph DriverResidency["Metal 4 Driver Residency Management"]
        ResSet["MTLResidencySet"]
        ArgTable["MTL4ArgumentTable"]
    end

    Hot -->|Registered Once| ResSet
    Cold -->|Dynamic Register / Evict| ResSet
    Bump -->|Pre-allocated Slabs| ResSet
    ConstArena -->|Direct Table Offsets| ArgTable
```

- **`BufferKind::Hot`**: Persistent allocations (model weights, optimizer state, KV cache banks). Added to the `MTLResidencySet` once at initialization and retained across steps.
- **`BufferKind::Cold`**: Intermediate activations. Managed via an active freelist pool with a default 2 GiB cap (`DEFAULT_POOL_CACHE_BYTES`). Unused slabs are evicted via `removeAllocation` upon command buffer completion.
- **`BufferKind::Bump`**: Ephemeral scratch memory allocated linearly from pre-committed slabs. Bump cursors are reset at synchronization points without individual buffer deallocations.
- **Constant Arena (16 MiB)**: Eliminates per-dispatch host allocation overhead for scalars and small metadata buffers by writing directly into a shared staging buffer at 16-byte aligned offsets.

---

## GEMM Pipeline & Cooperative Destination Execution

```mermaid
flowchart TD
    Start["gemm(a, b, c, backend)"] --> Validate{"validate_gemm()<br/>Rank-2, Non-empty, Bounds &lt;= 2^31,<br/>Same Runtime, No In/Out Overlap"}
    Validate -- Fail --> Err["Return Err(String)"]
    Validate -- Pass --> BackendCheck{"Backend?"}

    BackendCheck -- SimdGroup --> SimdGroupKernel["matmul_simdgroup<br/>(Portable SIMD Fallback)"]
    BackendCheck -- TensorOps --> LayoutCheck{"Layout Resolution"}

    LayoutCheck -- "TN / NT Layout" --> SplitKCheck{"prefer_tn_splitk?<br/>(K &gt;= 2048, M,N &lt;= 384,<br/>min(M,N) &lt;= 128)"}
    SplitKCheck -- Yes --> SplitKKernel["matmul2d_tensorops_tn/nt_splitk_*<br/>(Split-K partial reductions)"]
    SplitKCheck -- No --> CoopTN["matmul2d_tensorops_tn/nt_bf16_f32<br/>(128x64 sg4 Cooperative Destination)"]

    LayoutCheck -- "NN Layout" --> PrecisionCheck{"Precision Mode"}
    
    PrecisionCheck -- "f32 exact" --> F32Exact["matmul2d_tensorops_f32<br/>(Tile: 32x32, 1 simdgroup)"]
    
    PrecisionCheck -- "bf16 / tf32-relaxed" --> NNTable{"nn_coop_kernel()<br/>N &lt;= 512?"}
    
    NNTable -- "N &lt;= 512 (Narrow)" --> NNNarrow["matmul2d_tensorops_*_64x64_sg4<br/>• TILE_COOP_NARROW (64x64, 4 simdgroups)<br/>• Register accumulator, cT.store<br/>• Edge bounds-checked slices"]
    
    NNTable -- "N &gt; 512 (Default)" --> NNDefault["matmul2d_tensorops_*<br/>• TILE_COOP_DEFAULT (128x64, 4 simdgroups)<br/>• Column-panel swizzle if grid &gt;= 2048 tiles<br/>• Register accumulator, cT.store<br/>• Edge bounds-checked slices"]
```

### Cooperative Destination Advantages

1. **Register Accumulation:** `op.template get_destination_cooperative_tensor<...>()` maintains the full `f32` accumulator in hardware SIMDgroup registers across the entire $K$-reduction loop.
2. **Zero Pre-Zero Overhead:** Register accumulators are initialized via `.set(i, 0.0f)` in shader code. The host-side `zero_f32(C)` pre-pass is completely eliminated.
3. **Single Store to Memory:** Device memory $C$ is written **exactly once** (`cT.store(tC)`) at threadgroup termination.
4. **Ragged Edge Handling:** Boundary tiles use origin-shifted full-extent tensor slices (`mA.slice(...)`, `mB.slice(...)`, `mC.slice(...)`), executing the same cooperative register accumulation without dropping tail elements.
5. **Column-Panel Grid Swizzling:** For large dispatch grids ($\text{tiles}_n \times \text{tiles}_m \ge 2048$), threadgroups are swizzled into 8-tile-row bands to bound operand $B$ cache rereads, delivering $+11\%$ throughput at $4096^3$.

---

## Indirect Command Buffer (ICB) Decode Pipeline

For auto-regressive generation where kernel execution times approach dispatch overheads, `tessl` provides Indirect Command Buffer (ICB) capture and tape replay.

```mermaid
sequenceDiagram
    autonumber
    participant Host as Host Runtime / Client
    participant Binder as Binder / Dispatcher
    participant Tape as DecodeIcb Capture Tape
    participant ICB as Metal 4 MTLIndirectCommandBuffer
    participant GPU as Apple Silicon GPU

    Note over Host,GPU: 1. Capture Phase (First Token / Warmup)
    Host->>Binder: begin_decode_icb_capture()
    loop Model Layers (Decode Graph)
        Host->>Binder: bind_buffer(), set_pipeline(), dispatch()
        Binder->>Tape: Record Command (PSO, ArgTable, Buffers, Grid Size)
    end
    Host->>Tape: take_decode_icb_capture() -> Bake ICB Tape
    Tape->>ICB: Encode ICB Commands (freeze-binds / range-batching)

    Note over Host,GPU: 2. Steady-State Replay Phase (Subsequent Tokens)
    loop Each Decode Token
        Host->>Tape: try_replay_icb(runtime)
        Tape->>ICB: executeCommandsInBuffer:withRange: (Zero setArgumentTable host tax)
        Host->>GPU: Submit MTL4CommandBuffer (Ping-Pong buffers)
        GPU-->>Host: Signal MTLSharedEvent (Zero-wait async execution)
    end
```

---

## Documentation

| Document | Topic & Scope |
|---|---|
| [**Architecture**](docs/architecture.md) | Deep dive into kernel selection, cooperative destination register mechanics, $K$-reduction bandwidth analysis, and TN/NT layout optimizations. |
| [**Benchmarking**](docs/benchmarking.md) | The paired measurement protocol, GPU thermal and frequency scaling mitigation, and five measurement pitfalls. |
| [**Verification**](docs/verification.md) | Static tile geometry audit, self-asserting randomized shape fuzzing, and fault injection test suites. |

---

## Requirements

- **OS:** Apple Silicon, macOS 26 or newer
- **Toolchain:** Xcode 26 with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- **Language:** Rust 1.82+

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
