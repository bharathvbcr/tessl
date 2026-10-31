# Apple ML stack for tessl

Survey date: 2026-10-30. This is the hardware and framework reference for
kernel work. Live training gates stay in
[`optimization_map.md`](optimization_map.md) and
[`machine_profile_m5_pro.md`](machine_profile_m5_pro.md). Encode rules stay in
[`metal4_mpp.md`](metal4_mpp.md). Deploy-only rules stay in
[`coreml_metal_ml.md`](coreml_metal_ml.md).

Every number below is tagged **measured** (this machine, this session),
**Apple** (a page quoted that day), or **project** (a constant or comment in
this repo). A press-release speedup is Apple's comparison, not headroom under
the live step-time gate.

## This host, and macOS 27.2

| Item | Value | Tag |
|------|--------|-----|
| Chip | Apple M5 Pro | measured, `machdep.cpu.brand_string` and `MTLCreateSystemDefaultDevice` |
| CPU cores | 18 | measured, `hw.physicalcpu` |
| Memory | 64 GiB (`68719476736`) | measured, `hw.memsize` |
| Page size | 16 KiB | measured, `hw.pagesize` |
| CPU cache line | 128 bytes | measured, `hw.cachelinesize` |
| OS installed | macOS 27.0.1, build `26A434` | measured, `sw_vers` |
| OS offered, not installed | macOS 27.2, label `macOS 27.2-26B123`, restart required | measured, `softwareupdate -l` |
| Xcode installed | 27.1 (`27A9275`) | measured, `xcodebuild -version` |
| Metal compiler | `32023.921`, target `air64-apple-darwin27.0.0` | measured, `xcrun metal --version` |
| GPU families | `apple9` and `apple10` true; `apple11` false; `metal4` true | measured 2026-10-30, `supportsFamily` |
| `recommendedMaxWorkingSetSize` | `51539607552` bytes (48 GiB) | measured |
| Threadgroup memory | 32,768 bytes | measured, `maxThreadgroupMemoryLength` |
| Max threads per threadgroup | 1024 × 1024 × 1024 | measured |
| Command Line Tools 27.2 | offered, not installed | measured, `softwareupdate -l` |

`docs/apple-silicon/machine_profile_m5_pro.md` records macOS 27.0.1
(`26A434`). The optimization map’s machine blurb had said 26.5.2; that OS
token is now 27.0.1. The gate times in those files were not re-run for
this survey.

### What 27.2 changes for this work

Apple's [macOS 27.2 Golden Gate developer release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27_2-release-notes)
(fetched 2026-10-30) have no Metal, Core ML, Core AI, MLX, or Accelerate
section. The user-facing [27.2 notes](https://support.apple.com/en-us/127257)
are Siri AI: conversation memory, more personal context, Excluded Apps, Write
with Siri, and French, Japanese, Korean, Portuguese, and Spanish. Shortcuts'
"Use Model" action can use the web in more languages. None of that is a kernel
or a training-runtime change.

Do not install 27.2 from a training session. It is a multi-gigabyte update
that restarts the machine. Xcode 27.2's [SDK note](https://developer.apple.com/documentation/xcode-release-notes/xcode-27_2-release-notes)
says the macOS, watchOS, tvOS, and visionOS SDKs incorrectly report 27.1 as a
valid deployment target (187160501). Leave `MACOSX_DEPLOYMENT_TARGET` off 27.1.

The ML behavior that matters on this OS is in the cumulative
[macOS 27 Golden Gate release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27-release-notes),
not in the 27.2 delta. Those notes do not say which 27.x build each fix
landed in.

## Which layer does what

| Layer | Use it for | Leave it out of |
|-------|------------|-----------------|
| Metal 4 tensors + MPP TensorOps | Custom GEMM, fused epilogues, the GPU Neural Accelerator | Training-graph orchestration |
| `MTL4ComputeCommandEncoder` + argument tables | The training and decode encode path this repo already uses | A second Metal 3 command-buffer path |
| MPS / MPSGraph / `MTL4MachineLearningCommandEncoder` | Running a compiled model on the GPU timeline next to other GPU work | Muon, custom attention backward, GDN |
| Core ML | Existing `.mlpackage` export | The training step |
| Core AI (`AIModel`, `.aimodel`) | New on-device deployment: specialization, state, zero-copy `NDArray` | The training step |
| Foundation Models / Evaluations | App prompts against Apple's on-device model, or a `LanguageModel` provider | Weights, kernels, grads |
| MLX | A separate research runtime. Metal 4, Neural Accelerators, and multi-Mac RDMA are its 2026 path | Replacing `tessl` or `ojas-metal` |
| Accelerate / BNNS Graph | CPU GEMM and small CPU graphs | GPU kernels |
| Apple Neural Engine | Core ML / Core AI placement, when the op and dtype fit | A programmable ISA. There is no public ANE kernel language |

## Three different matrix units

Apple's October 15, 2025 [M5 release](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
puts a Neural Accelerator in each GPU core and a separate 16-core Neural
Engine on the base M5. The same sentence also says the Neural Engine
complements "Neural Accelerators in the CPU and GPU." Apple does not document
a public CPU tensor ISA in that release. On this machine the CPU matrix unit
that `sysctl` exposes is ARM SME, measured below. Do not treat that sysctl
block as the GPU Neural Accelerator.

Programming path for the GPU Neural Accelerator, from Apple's
[inline tensor sample](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4):
Metal 4 tensor ops. The sample says `MTLGPUFamily.apple10` and later run those
ops on the per-core neural accelerator. This device reports `apple10`, so that
sentence applies here. The same boot reports `apple11` false. metal-cpp's
27.0.1 drop adds Apple11-family wrappers; those wrappers are not a feature
list for this M5 Pro. Dynamic Caching, in Apple's
[October 30, 2023 M3 note](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/),
is hardware allocation of GPU local memory. It is not a KV cache, and it is
not Foundation Models' Dynamic Profiles. The M5 (2025-10-15) and M5 Pro/Max
(2026-03-03) newsroom pages use the lowercase phrase “second-generation
dynamic caching” for the shader core. The M6 / M5 Ultra page (2026-08-25)
uses the capitalised name for geometry rates and game frame rates. Same
graphics feature. The quotes are in
[`beta_nightly_gap.md`](beta_nightly_gap.md).

The Neural Engine is the other block. Core AI's
[`ComputeUnitKind`](https://developer.apple.com/documentation/coreai/computeunitkind)
is `cpu`, `gpu`, or `neuralEngine`. Specialization uses all available units
unless you restrict it. The macOS 27 notes say some weight layouts do not run
on the Neural Engine and fall back to CPU or GPU: FP8-quantized weights and
activations, palettized weights whose values are not Float16, and sparse
weights (176210080). Background Neural Engine access needs the entitlement
`com.apple.developer.background-tasks.continued-processing.inference`
(179282606). A foreground Metal training process does not use that entitlement.

## Chip figures Apple has published

These are Apple's "up to" figures, not a measurement on this Mac. Peak-GPU-AI
multiples are Apple's own benchmarks against the chip they name.

| Chip | What Apple states | Source |
|------|-------------------|--------|
| M5 | 10-core GPU class with a Neural Accelerator per core; 16-core Neural Engine; 153 GB/s; up to 32 GB; over 4× peak GPU compute for AI vs M4 | [M5, 2025-10-15](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) |
| M5 Pro | Fusion Architecture (two dies); Neural Accelerator per GPU core; up to 64 GB; up to 307 GB/s; over 4× peak GPU compute for AI vs M4 Pro | [M5 Pro / M5 Max, 2026-03-03](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/) |
| M5 Max | Up to 40-core GPU; up to 128 GB; up to 614 GB/s. Apple ties the bandwidth increase to higher LLM token generation | same |
| M6 | First 2 nm M-series; 12-core CPU (2 super, 4 performance, 6 efficiency); 12-core GPU with a Neural Accelerator per core; Dual 16-core Neural Engine; up to 32 GB; up to 170 GB/s; about 30% more peak GPU AI compute than M5 | [M6 and M5 Ultra, 2026-08-25](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/) |
| M5 Ultra | Quad-die (two dual-die M5 Max) via UltraFusion, inter-die bandwidth over 4.4 TB/s; up to 36-core CPU (12 super + 24 performance); up to 80-core GPU with Neural Accelerators; 32-core Neural Engine; up to 512 GB; 1.2 TB/s | same |

This machine's 18 CPUs and 64 GiB match the high M5 Pro bin in those releases.
`MTLDevice` on this boot does not print a GPU core count. `system_profiler
SPDisplaysDataType` reports “Total Number of Cores: 20”, and `ioreg`
`gpu-core-count` is 20. Both were read 2026-10-30. That is the training
profile's figure.

An October 27, 2026 MacBook Pro with M6 exists as a separate newsroom item.
This survey did not open it, so it contributes no extra bandwidth or core
counts here.

M5's "over 4× peak GPU compute for AI versus M4" is a cross-generation peak.
It is not unused FLOPs under the live B=16 step, which the training profile
already treats as binder- and backward-bound.

## Unified memory

Apple's [storage-mode guide for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus):
the CPU and GPU share system memory, and the storage mode still decides who
can map it.

| Mode | Rule on this GPU |
|------|------------------|
| `shared` | Default for buffers and textures on Apple silicon. Both processors can access it. You still have to finish the CPU's writes before the GPU reads, and wait for the GPU before the CPU reads. Apple's words on [`MTLStorageMode.shared`](https://developer.apple.com/documentation/metal/mtlstoragemode/shared). |
| `private` | GPU only. On a unified-memory GPU the bytes still sit in system memory; the CPU cannot map them. Use it for data that is born on the GPU. Getting it back takes a blit or a shared staging buffer. |
| `memoryless` | Tile memory for a render pass. Higher bandwidth than DRAM, and gone when the pass ends. Not a place for weights or a GEMM operand. |
| `managed` | Deprecated. It was the discrete-GPU dual-copy mode. Do not allocate it. |

`cpuCacheModeWriteCombined` is the right CPU mapping for a buffer the CPU
fills once and never reads back. The default cached mapping is the right one
when the CPU will read the loss or a debug tensor.

### Residency is a second axis

[DTS, August 2026](https://developer.apple.com/forums/thread/842200): a
residency set says which allocations the GPU may access; the storage mode does
not. Shaders can use only resident resources.

From Apple's [residency-set article](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
and the [`useResidencySet`](https://developer.apple.com/documentation/metal/mtl4commandbuffer/useresidencyset(_:))
page:

- A command buffer holds at most 32 residency sets.
- The GPU sees the union. Removing an allocation from one set does nothing if
  another attached set still contains it.
- Making a heap resident makes the whole heap resident.
- The set does not track hazards. Fences, events, and encoder barriers still
  apply.
- `requestResidency()` after `commit()` asks Metal to prepare the set before
  the command buffer that uses it. Metal may defer that under memory pressure.
- Argument-buffer resources are not tracked automatically. The
  [argument-buffer residency article](https://developer.apple.com/documentation/metal/tracking-the-resource-residency-of-argument-buffers)
  says to call `useResource` / `useHeap` yourself.

Project rules already in [`metal4_mpp.md`](metal4_mpp.md): register every
allocation, attach the set on the command buffer, and `removeAllocation` only
after the command buffer completes. `MTL4` command buffers do not retain
those allocations (`src/decode_icb.rs`). The const arena is 16 MiB
(`METAL4_CONST_ARENA_BYTES` in `src/runtime.rs`). `metal4_mpp.md` names that
same 16 MiB arena. The older 1 MiB figure is the size that exhausted mid-token.

`recommendedMaxWorkingSetSize` on this boot is 48 GiB of the 64 GiB physical.
That is the driver's recommendation, not a license to pin 48 GiB. The training
profile's wired fraction is a policy on top of this probe. Re-read the probe
at `GpuRuntime` init; do not hard-code 48 GiB.

### Alignment

- CPU page: 16 KiB. Large allocations and residency churn should be page
  multiples.
- CPU cache line: 128 bytes. A shared atomic or a reduction slot that two
  cores write needs its own line.
- Tensor ops want extents indexed with `int`, and static tile extents when the
  tile size is known (Apple's inline-ML sample). Dynamic extents insert
  per-element bounds checks.

### What the bandwidth number is for

Apple's 307 GB/s for M5 Pro is a peak, not a sustained copy this session
measured. Decode of a weights-bound model is limited by how many bytes of
weights (and the growing KV cache) each new token rereads. A tensor core does
not speed up a kernel that is waiting on that read. Prefill and a large
training GEMM are where the Neural Accelerator shows up. The training profile
already says to confirm that with Instruments' neural-accelerator counter
before porting another op onto TensorOps.

## Kernel recipe

Apple's inline-ML sample is the pattern. Paraphrase, not a second copy of
their shader:

1. Bind rank-2 `tensor<device T, dextents<int, 2>>` values. The index type is
   `int`.
2. Slice a compile-time tile (`slice<Tile, Tile>`). Use `dynamic_extent` on
   the K side so the compiler takes K from the tensor.
3. Build `matmul2d_descriptor` and `matmul2d<descriptor, execution_simdgroups<N>>`.
   The sample uses a 64×64 tile and four SIMD groups.
4. Hold the product in `get_destination_cooperative_tensor()`. That storage is
   registers distributed across the execution scope, not device memory and not
   threadgroup memory.
5. Run the epilogue on `cooperativeTensor[element]` (their example is ReLU),
   then `store` the output slice.

Apple's note in that sample: Apple silicon GPUs do not need threadgroup-memory
staging for tensor ops, and they do not need a software pipeline to overlap
memory and math, because the cache hierarchy reuses the data. Tessl's tuned
tiles stay as they are until a paired bench says otherwise. The sample is a
license to stop staging, not a license to throw away a measured tile.

Cooperative-tensor **inputs** to matmul are the macOS 26.3 addition named in
Apple's [M5 tech talk](https://developer.apple.com/videos/play/tech-talks/111432/).
That is the dequant-in-kernel path: produce a cooperative tensor in registers
and hand it to matmul without a round trip through device memory. The same
talk says bfloat tensor support arrived in 26.1. The `MTLTensorDataType.bfloat16`
symbol page says the case exists since macOS 26.0. Both can be true: the enum
case and the TensorOps consumer landed on different builds. This host is 27.0.1,
so both are in the past.

[`MTL4MachineLearningCommandEncoder`](https://developer.apple.com/videos/play/wwdc2025/262/)
and `metal-package-builder` compile a Core ML package onto the GPU timeline.
That remains inference packaging. [`coreml_metal_ml.md`](coreml_metal_ml.md)
already forbids it on the training hot path.

Shader validation vs MPP: the macOS 26 notes say Metal Shader Validation might
not work with shaders that use Metal Performance Primitives (149263281, listed
as fixed there). If a validation failure appears only inside a TensorOps
kernel, check that note before rewriting the kernel.

The macOS 27 Metal notes fix a clamp-to-edge **sampler** returning zeros, on
Apple 10 GPUs in particular (172520325, 177318505). Compute buffers and tensor
ops do not go through that sampler. Do not cite it as a GEMM bug.

### Tensor dtypes, by the symbol page that was opened

| Case | Availability opened | Meaning |
|------|---------------------|---------|
| `bfloat16` | macOS 26.0 | 1/8/7 |
| `int8` | macOS 26.0 | 8-bit integer data plane |
| `int4` | macOS 26.4 | 4-bit integer data plane |
| `metalFloat8e4m3` | macOS 27.0 | 1/4/3 |
| `metalFloat8ue8m0` | macOS 27.0 | E8M0 scale, unsigned, no mantissa |
| `metalFloat4e2m1` | macOS 27.0 | 1/2/1 |
| `int2` | macOS 27.0 | 2-bit integer |
| `metalFloat8e5m2` | macOS 27.0 | 1 sign, 5 exponent, 2 mantissa. The symbol page is that sentence only. |

The [`MTLTensorDataType`](https://developer.apple.com/documentation/metal/mtltensordatatype)
page also lists `float16`, `float32`, the 16/32-bit integers, and the unsigned
4/8-bit integers. Those cases were not opened one-by-one.
`metalFloat8ue8m0` is not a legal **data-plane** type
([`dataType`](https://developer.apple.com/documentation/metal/mtltensordescriptor/datatype));
it is the scale plane.

WWDC 2026 session [330](https://developer.apple.com/videos/play/wwdc2026/330/)
shows the scale plane: an auxiliary plane of `metalFloat8ue8m0` with
`blockFactors` (their example is 32×1) on an `metalFloat8e4m3` data plane.
The MSL handle is `tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_handle, scales_plane>`.
That is an inference-weight layout. It is not a training-master layout.

Ojas dtype policy keeps parameters, master gradients, AdamW moments, and Muon
momentum in f32. A bf16 autocast region rounds matmul-class copies. FP8 and
int4 belong on an export path after a paired numeric check, not in the master
weights.

### What tessl already does

Do not re-implement these from this note. Status and files are in
[`metal4_mpp.md`](metal4_mpp.md):

- Metal 4 only. `GpuRuntime::new` fails if the Metal 4 package does not init.
- TensorOps GEMM is primary. Simdgroup GEMM stays the portable fallback.
- Morton order, packed zero+matmul, static extents where the shape is fixed.
- One compute encoder packed across binders. Metal 4 still inserts a
  device-scope barrier after each dispatch; hazard-aware barriers are an
  open A/B, not a default change.
- Residency add/remove is tied to command-buffer completion.
- Exact f32 stays on the conservative simdgroup tile. bf16+relaxed uses
  `execution_simdgroups<4>`.

## Deploy stack, for when a model leaves the trainer

### Core AI

[What's New in AI](https://developer.apple.com/machine-learning/whats-new/)
describes Core AI as the on-device path: Swift `AIModel`, specialization for
the current hardware, ahead-of-time compilation, zero-copy inputs, and state.
WWDC 2026 session [324](https://developer.apple.com/videos/play/wwdc2026/324/)
shows a KV cache as `register_buffer` in PyTorch, then `state_names` on
`coreai_torch.TorchConverter`. Session [330](https://developer.apple.com/videos/play/wwdc2026/330/)
says a custom Metal kernel can be the escape hatch inside that graph.

macOS 27 notes that change how an export behaves:

| Note | Why it matters here |
|------|---------------------|
| Control flow over dynamic-shape tensors can fail or crash. The example Apple gives is linear-attention LLMs such as Qwen3.5 / 3.6 (177354777), listed as fixed. | A Core AI export of Qwen3.5 is a suspect until it is run on the build you ship. The note does not name the point release. |
| `InferenceFunction.encode` on the GPU used to block until compute finished, unless the model was specialized with GPU as the preferred device (175789258), listed as fixed. | A latency measurement of a Core AI model specialized for "all units" is not a GPU-timeline measurement. |
| State arguments plus dynamic-shape outputs can fail shape inference (176807213), listed as fixed. | A stateful decode export needs an explicit output shape check. |
| Custom Metal kernels failed to load (178056451), listed as fixed. | Re-test any package that embeds a tessl kernel. "Fixed" in the 27 notes is not a test on this tree. |
| Metal API Validation could make Core AI models fail (177991751), listed as fixed. | A validation-on failure is not automatically a kernel bug. |
| `coreai-torch` v0.4.0 packages failed to specialize (177008303), listed as fixed. | Pin the converter version in the export script and record it. |
| AOT artifacts from Xcode 27 Beta 2 or earlier could fail to specialize (181264112), listed as fixed. | Rebuild `.aimodelc` with the SDK you actually ship. |

Core AI does not replace the Metal training loop.

### Foundation Models

The same What's New page: a Swift API over the on-device Apple model, plus any
type that conforms to `LanguageModel` (they name Claude and Gemini as
examples). Multimodal prompts, Vision tools, Dynamic Profiles, an Evaluations
framework, and Private Cloud Compute for apps under 2 million first-time App
Store downloads. WWDC 2026 sessions 241, 242, 319, 298.

macOS 27 fixes, all listed as resolved issues rather than new behavior to
depend on: excessive tool calls when tool calling and guided generation are
combined (177748926); `PrivateCloudComputeLanguageModel` always using greedy
decoding (178181782). A known issue remains: a default `SpotlightSearchTool`
does not fit the on-device model's context (183770678). Use `.focused()`.

Nothing in that framework launches a GEMM.

### MLX

The docs page opened for the stack survey is still labeled MLX 0.32.4.
[`mlx_audit.md`](mlx_audit.md) is the 2026-10-30 pass over that tag versus
`main`: NAX routing, affine Q4 versus fp4, SDPA at head dim 256, and the
gated-delta VJP. Its
[unified-memory page](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
is the model tessl already followed: an array has no copy between CPU and GPU;
you pick the stream; the scheduler inserts the dependency. MLX's
[distributed page on `main`](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst)
documents JACCL, RDMA over Thunderbolt, from macOS 26.2, Thunderbolt 5, and
enablement that cannot be done remotely. WWDC 2026 session
[233](https://developer.apple.com/videos/play/wwdc2026/233/) is the same stack.
Apple's What's New page also says this year's MLX uses Metal 4 and the GPU
Neural Accelerators.

MLX is a baseline and a source of ideas (`mx.compile` ≈ fewer dispatches).
It is not the trainer. The cheat sheet of what to steal is
[`mlx.md`](mlx.md). The routing audit is [`mlx_audit.md`](mlx_audit.md).

### CPU: Accelerate, and SME that you do not write by hand

Measured on this boot (`sysctl hw.optional`):

| Flag | Value |
|------|-------|
| `FEAT_SME`, `FEAT_SME2`, `FEAT_SME2p1` | 1 |
| `sme_max_svl_b` | 64 (512-bit streaming vector length) |
| `SME_F32F32`, `SME_F16F32`, `SME_B16F32`, `SME_I8I32` | 1 |
| `FEAT_SME_B16B16`, `FEAT_SME_F16F16` | 1 |
| `FEAT_SME_F8F16`, `FEAT_SME_F8F32` | 0 |
| `FEAT_BF16`, `FEAT_DotProd`, `FEAT_I8MM`, `FEAT_FP16` | 1 |

Large f32 GEMMs go through Accelerate, which can use this unit. FP8 outer
products are not in the SME feature set on this CPU. Small GEMMs lose to
launch overhead; ojas already keeps those on its own tile (see the ojas
guide). Do not add SME assembly. BNNS Graph (WWDC 2025 session
[276](https://developer.apple.com/videos/play/wwdc2025/276/)) is the supported
way to build a small CPU graph in Swift. It is not on the tessl hot path.
vImage’s symbol page has no 2026 section, and the Accelerate updates article’s
newest section is still June 2024. vImage is not a GEMM path.

## Adoption list

Worth a measured experiment on this OS (27.0.1, apple10):

1. Cooperative-tensor inputs for a dequant epilogue whose bytes dominate the
   kernel. Gate on the neural-accelerator counter and a paired bench, same
   rule as [`metal4_mpp.md`](metal4_mpp.md).
2. An export-only MXFP8 or int4 weight tensor, with the scale plane, compared
   against the f32 master. Not a training storage change.
3. Re-specializing any old Core AI package, because the 27 notes record
   several specialization and custom-kernel failures as fixed without naming
   the build.

Not worth doing because 27.2 exists:

- Retuning tiles for M6 or M5 Ultra bandwidth.
- Treating Siri AI language support as a runtime change.
- Setting the deployment target to 27.1 to "pick up the SDK."
- Moving training onto Core AI, Core ML, or the Neural Engine.

## Sources

Apple, fetched 2026-10-30 unless noted:

- [What's new in AI & machine learning](https://developer.apple.com/machine-learning/whats-new/)
- [MTLTensorDataType](https://developer.apple.com/documentation/metal/mtltensordatatype) and the case pages linked above
- [Running inline ML operations in a shader with Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
- [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
- [Simplifying GPU resource management with residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
- [macOS 27 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27-release-notes)
- [macOS 27.2 release notes](https://developer.apple.com/documentation/macos-release-notes/macos-27_2-release-notes)
- [Xcode 27.2 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27_2-release-notes)
- [What's new in macOS 27 updates](https://support.apple.com/en-us/127257)
- Newsroom: M5 (2025-10-15), M5 Pro/Max (2026-03-03), M6 and M5 Ultra (2026-08-25)
- WWDC 2026 sessions 324, 330, 233; WWDC 2025 sessions 262 and 276; tech talk 111432

This machine: `sw_vers`, `softwareupdate -l`, `sysctl`, `xcodebuild -version`,
`xcrun metal --version`, and one Swift `MTLCreateSystemDefaultDevice` query.
No training bench was re-run.
