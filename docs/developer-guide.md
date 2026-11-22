# tessl developer guide

_Read at tessl fba6fc0 plus the uncommitted working tree, 2026-11-22. Claims carry file:line evidence; where code and this page disagree, the code wins._

This guide is for developers, ML and infra engineers, and AI coding agents who use or change tessl. It says what tessl is, what it owns, how to build and test it, a tour of each feature with entry points, the contracts it keeps, and the traps that cost previous sessions time. Limits and open issues are in [limitations-and-known-issues.md](limitations-and-known-issues.md). Counts, coverage tables and benchmark numbers live in the canonical docs linked below and are not repeated here.

Working-tree note: at the read, four tessl files were dirty (`AGENTS.md`, `README.md`, `go/README.md`, `go/go.mod`; `git status --short`). The diff moves the Go floor from 1.26 to 1.27 (`go/go.mod:3`) and rewrites the sibling-checkout and gusset wording. This page follows the working tree.

## 1. Why tessl exists, and its two roles

tessl is a Rust crate (`Cargo.toml:2`, name `tessl`, edition 2021, `rust-version = "1.100"` at `Cargo.toml:17`) that drives Apple-silicon GPUs through Metal 4. It plays two roles, stated in `AGENTS.md:7-10` and [architecture.md](architecture.md#tessls-role-metal-extension-of-ojas-and-the-mac-inference-engine):

| Role | What it means | Where it shows |
|---|---|---|
| Metal 4 extension of ojas | The runtime, the unified-memory allocator (`SharedBuf`) and the compute kernels that ojas's Metal backend calls for training and general ops. | `ojas-metal` takes tessl as an optional path dependency, `tessl = { path = "../../tessl", optional = true }` (ojas repo, `ojas-metal/Cargo.toml:22`). |
| Mac inference engine | Hand-tuned Qwen3.5 prefill and decode, MLX Q4/Q8 quantization, EmbeddingGemma 2, BERT and DistilBERT, TimesFM 3, and the fused kernels under them. | The `src/` modules in section 5. |

Why a separate checkout (architecture.md, "The Modularity Rationale"): ojas builds on Linux, macOS and Windows; tessl needs Xcode 26, the Metal Toolchain and Objective-C bindings, tunes registers and tiles on its own cadence, and lets Mac applications use the engine without the training framework. The dependency direction is `ojas-metal -> tessl -> ojas-core` (`docs/WORKFLOWS.md`, "Architecture Context").

## 2. What tessl owns, and what ojas owns

The split is the owner decision recorded in [architecture.md](architecture.md#what-tessl-owns) and [io-ownership.md](io-ownership.md).

| Owned by tessl | Owned by ojas |
|---|---|
| `GpuRuntime` (one device, one Metal 4 queue, encoder lease, shared events, residency set, pipeline cache), `src/runtime.rs:1072` | `Tensor`, `DType`, `OjasError`, `Budget`, the `Backend` trait, shape validators, autocast (ojas-core) |
| The allocator pools (`BufferKind`, `src/runtime.rs:98`) and `SharedBuf` (`src/shared_buf.rs:134`) | The autograd tape, optimizers, training loops, data, checkpoints, the CPU reference backend |
| GEMM family, the `nn` kernel library, attention, quantized GEMV/GEMM, Qwen3.5 / EmbeddingGemma 2 / BERT / TimesFM 3 kernels | Optimizer scalar formation (`ojas_core::AdamWScalars`), clipping policy, LR schedules, loss scaling |
| The C ABI (`src/capi.rs`, `src/capi_embedgemma2.rs`), the `tessl_torch` binding, the Go binding | safetensors, JSON and NPY parsing (`ojas-io`), see [io-ownership.md](io-ownership.md) |
| Per-op training kernels: `src/qwen35_bwd.rs`, `src/gdn_train.rs`, `src/attn_train.rs`, `src/cross_entropy.rs` | Whole training steps: `TapeStepProvider` in `ojas-model/src/qwen35/step_provider.rs:627` (ojas repo) |

Evidence for the "retired to ojas" rows:

- `src/lib.rs:272` and `src/lib.rs:285` re-export `ojas_io::npy` and `ojas_io::safetensors`; tessl's own parsers were deleted (CHANGELOG.md, Unreleased, "Removed").
- ABI 14 removed the Qwen3.5 training step, the host AdamW object and 13 training C-ABI symbols (`src/capi.rs:82-91`; CHANGELOG.md, Unreleased, "Removed"). `src/qwen35_adamw.rs` keeps only the per-op f32 launcher `adamw_step` (`src/qwen35_adamw.rs:131`) with `AdamWHyper` (`:29`).
- Not-to-grow list: no generic training logic, no second autograd, no second `Tensor`/`Budget`/error model, no second safetensors/JSON/NPY parser, no portable-backend abstractions (architecture.md, "What tessl should not grow").

gusset, the in-process Go/Rust bridge, is not a path dependency of tessl. The Go binding's engine crate takes it as a git dependency (`go/engine/Cargo.toml:25`, `rev = "984209b5..."`) and `go/go.mod` requires `github.com/bharathvbcr/gusset v0.0.3` from the module proxy. A local checkout is needed only to build ojas itself (`AGENTS.md:15-21`). sparsl, the sparse and scan compute extension, is a companion project and not a dependency (`README.md`, "Related Projects").

## 3. Prerequisites

| Need | Detail | Evidence |
|---|---|---|
| Hardware and OS | Apple silicon, macOS 26 or newer. There is no CPU fallback; without a Metal 4 device nothing runs. | `README.md`, "Requirements" and "Known Gaps" |
| Toolchain | Xcode 26 with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`). `build.rs` compiles the shaders ahead of time with `xcrun metal`, never `xcrun -sdk macosx metal`. | `README.md` Requirements; `build.rs:10-11` |
| Rust | 1.100 or newer; it must not sit below the ojas workspace floor. | `Cargo.toml:3-17` |
| Sibling checkout | ojas at `../ojas`, because `ojas-core` and `ojas-io` are mandatory path dependencies. | `Cargo.toml:126-127` (the two `path = "../ojas/..."` lines) |
| Preflight | `./scripts/check_siblings.sh [--network]` reports a missing or misplaced ojas before cargo fails with a manifest error. | `scripts/check_siblings.sh:1-30`, `tests/check_siblings_script.rs` |

The `cargo add tessl` release on crates.io (0.2.0, 2026-09-18 per `README.md` Quickstart) predates the ojas path dependencies. Current `main` builds only from source with the two repositories side by side.

`build.rs` has two escape hatches: `DOCS_RS` skips shader compilation for docs.rs (`build.rs:68`), and `TESSL_SKIP_AOT` skips it offline but requires `TESSL_PREBUILT_METALLIB` (`build.rs:76-80`). The compiled metallib is embedded in the binary, so `GpuRuntime::new()` needs no file on disk (`src/runtime.rs:1343`, `:1380`; `tests/embedded_metallib.rs:1-9`).

## 4. Build and test

All commands below exist in `scripts/`, `AGENTS.md` or `README.md`. None were run to write this page.

```sh
./scripts/check_siblings.sh                 # sibling preflight
./scripts/dev_fast_check.sh                 # inner loop: cargo check + unit tests (default filter runtime::tests)
./scripts/dev_fast_check.sh --check-only    # type check only
./scripts/dev_fast_check.sh --clippy        # add clippy (off by default)
./scripts/dev_fast_check.sh --all           # the full lib test suite
cargo test --release -- --test-threads=1    # GPU tests, one at a time
cargo run --release --example gemm          # the README quickstart as a build target
cargo run --release --example nn_layer      # RMSNorm -> gate/up -> GELU -> residual
```

Options of `dev_fast_check.sh` are `--all`, `--check-only`, `--clippy`, `--skip-clippy`, `--skip-tests` (`scripts/dev_fast_check.sh:55-63` help text, `:73-98` option parsing). Run GPU tests with `--test-threads=1` to avoid GPU memory and queue contention (`AGENTS.md:39-40`); tests that read allocator gauges need it for correctness (`docs/WORKFLOWS.md`, "Invariants Maintained", item 5).

The tiered pipeline in [WORKFLOWS.md](WORKFLOWS.md) and [verification.md](verification.md):

| Tier | Gate | Command |
|---|---|---|
| 0 | fast loop | `./scripts/dev_fast_check.sh [filter]` |
| 1 | compile and lint | `cargo check --all-targets && cargo clippy --release --all-targets -- -D warnings` |
| 2 | device-free tests under a sandbox | `./scripts/hosted_tests.sh` (manifest `scripts/hosted_tests.txt`) |
| 3 | Metal dialect and IR scan | `./scripts/check_kernel_ir.sh` |
| 4 | emulator and host contract | `python3 tools/msl_emu/check_qwen35.py -k host_contract` |
| 5 | serialized GPU tests | `./scripts/ci_local.sh` |

`scripts/hosted_tests.sh` keeps an exact test count per manifest entry, so adding or renaming a test fails it until `scripts/hosted_tests.txt` is updated (`scripts/hosted_tests.sh:1-24`). `scripts/ci_local.sh` runs `check_siblings.sh` first (`:14-20`), then fmt, build and the rest; GPU-dependent Python and Go stages run only under the machine heavy lock (`MAC_HEAVY_LABEL` set, `docs/WORKFLOWS.md`) and print an explicit skip otherwise. GitHub's hosted runners cannot execute shaders, so the GPU suite is not run by CI (`README.md`, "Known Gaps", GPU CI row).

Other bindings:

- Go: `make -C go unit` (no GPU), `make -C go device`, `make -C go test`, `make -C go real`, `make -C go bench`, `make -C go serve` (`go/Makefile:44-101`, `go/README.md`, "Requirements and the one command").
- Python: `cargo build --release` then `python3 -m unittest discover -s python/tests -v`; `TESSL_LIB` overrides the library path (`python/README.md`, "Build and test").

## 5. Feature tour

Each row names an entry point and the page that holds the detail. Read the linked page for numbers.

### 5.1 GEMM and TensorOps `matmul2d`

- Entry points: `gemm` (`src/gemm.rs:445`), `gemm_bf16` (`:1567`), `gemm_nt_f32` (`:2269`), `gemm_nt_bf16` (`:2301`), `gemm_epilogue` (`:966`). `GemmBackend::TensorOps` selects the MPP `matmul2d` kernel `matmul2d_tensorops_f32`, `GemmBackend::Simdgroup` the portable `matmul_simdgroup_f32` (`src/gemm.rs:236-250`). `GemmOperands` picks exact f32 or bf16 operands with f32 accumulation (`src/gemm.rs:1371`).
- Example (lifted from `examples/gemm.rs:10-42`, the README quickstart as a build target that its header says CI compiles and runs; the `fn main` wrapper is dropped and the comments trimmed):

```rust
use tessl::{gemm, GemmBackend, GpuRuntime};

let rt = GpuRuntime::new()?;
let (m, k, n) = (512, 256, 128);
let a = rt.alloc_tensor_f32(&[m, k])?;
let b = rt.alloc_tensor_f32(&[k, n])?;
let c = rt.alloc_tensor_f32(&[m, n])?;
a.buffer.write_f32(&vec![1.0f32; m * k]);
b.buffer.write_f32(&vec![2.0f32; k * n]);
gemm(&a, &b, &c, GemmBackend::TensorOps)?;   // C = A @ B via MPP TensorOps
rt.synchronize()?;                          // wait for the GPU before reading back
let out = c.buffer.read_f32();              // every element is 2.0 * k
```

(The README's "Basic GEMM" block is the same call sequence at 4096 x 2304 x 768 without the fill and readback; the example is the compiled copy. Not compiled or run for this page.)
- Detail: [architecture.md](architecture.md) (kernel selection, cooperative destination accumulation, swizzling), [apple-silicon/gemm_architecture.md](apple-silicon/gemm_architecture.md), [apple-silicon/metal4_mpp.md](apple-silicon/metal4_mpp.md), [benchmarking.md](benchmarking.md).

### 5.2 Neural-network kernels (`nn`)

- `src/nn.rs` is the typed, validate-before-encode surface over model-agnostic kernels: RMSNorm (`rms_norm_f32` `src/nn.rs:486`), gated MLP activations, flash attention (`flash_attn_decode` `:1745`, `flash_attn_rows` `:2370`, `flash_attn_swa` `:2757`), softcap (`softcap_logits` `:3624`), quantized GEMV and embedding lookups (`gemv_q4_mlx` `:4397`, `embed_lookup_q4_mlx` `:4257`). The module comment says the kernels guard only `gid >= n`, so the host wrappers check buffer sizes before encoding (`src/nn.rs:9-17`).
- Each kernel has a plain entry that binds scalars through the runtime const arena and an `unsafe` `_with_scalars` entry for callers that need stable GPU addresses across encodes, as an ICB does (`src/nn.rs:19-30`).
- Detail: README "Neural-Network Kernels" and "Fused GEMM Epilogue", [architecture.md](architecture.md#the-kernel-library-and-the-nn-boundary).

### 5.3 `SharedBuf` and unified memory

- `SharedBuf` (`src/shared_buf.rs:134`) implements `ojas_core::MappedBuffer` (`:685`) over pooled `StorageModeShared` buffers. `SharedBuf::to_gpu_buffer()` (`:416`) returns the pooled `GpuBuffer` so ojas-metal can adopt it with no copy. `SharedAlloc::new` allocates one from a runtime (`:714`, `:721`).
- Host access is a lease. `SharedBuf::read` returns a `HostRead` (`:460`), `SharedBuf::write` a `HostWrite` (`:491`); the `MappedBuffer` methods `host_read` / `host_write` return boxed `ojas_core` lease traits (`:698-706`). Details in section 7.
- `src/pool_ledger.rs` defines the `PoolLedger` trait (`:123`) and `OjasBudgetLedger` (`:196`), which charges pool allocations to an `ojas_core::Budget`.
- Detail: [architecture.md](architecture.md#memory-allocation-lifecycle--residency-pools).

### 5.4 Qwen3.5 prefill and decode

- Load: `Qwen35Model::load` (`src/qwen35_model.rs:647`), `load_tower` (no LM head, `:665`), `load_mlx_q4` (`:692`). Config: `Qwen35Config::from_config_json` (`:201`) refuses by name what the forward lacks (attention bias at `:241`, `mlp_only_layers` at `:275`).
- Prefill and score: `forward` (`:1248`), `forward_rows` (`:1256`), `begin` returning a `Staged` snapshot (`:1293`), and `Staged::score_answers` (`:2315`).
- Decode: `Qwen35Model::prefill(ids, max_new)` returns a `Decode` session and the first-token logits (`:1307`); `Decode::step(id)` runs one token through every layer and returns `[vocab]` f32 logits (`:2425`). Fusion of add + RMSNorm and gate/up GEMV + SiLU is chosen per session with `Decode::set_fusion` (`:2404`) or `TESSL_QWEN35_DECODE_FUSION=1` (`:2047`).
- Kernels are in `src/qwen35.rs` (`gdn_chunk_forward` `:1100`, `gdn_recurrent` `:1343`, `attn_prefill` `:2858`) and `kernels/qwen35_*.metal`.
- Detail: [qwen35.md](qwen35.md). Its training sections (see "Doc drift noticed") describe retired ABI 13 behaviour; the prefill, decode, scoring, numerics and verification sections are the live ones.

### 5.5 KV caches and decode state

There is no `KvCache` type. Per-layer decode state is the private `Carry` enum (`src/qwen35_model.rs:2073`) held by each `Decode` session: attention layers keep `ids.len() + max_new` positions of K/V, GDN layers keep recurrent and conv state until the session is dropped (doc of `prefill`, `:1297-1306`). The attention K/V write is the `kv_store` kernel in `kernels/kv_store.metal`. A `Decode` session refuses to continue once `max_new` tokens are decoded, once the model's weights were written since the prefill, and for good after a step that failed part way (doc of `step`, `src/qwen35_model.rs:2418-2424`).

### 5.6 Decode ICB and ping-pong replay (opt-in, not on the Qwen3.5 path)

- `DecodeIcb` (`src/decode_icb.rs:258`) replays a captured chain of compute dispatches from an Indirect Command Buffer. It is off by default: `decode_icb_enabled()` reads `TESSL_DECODE_ICB` or `METAL_RUNTIME_DECODE_ICB` (`src/decode_icb.rs:38-40`). `PingPongCbReplay` (`src/cb_replay.rs:172`) is the dual-slot ledger; its `try_replay` returns `CbReplayError::NotWired` outside dry runs because Metal 4 has no "replay prior encoding" API (`src/cb_replay.rs:1-30`, the `NotWired` return at `:455`).
- Verified by search: `src/qwen35_model.rs`, `src/qwen35.rs` and `src/qwen35_q4.rs` contain no reference to `DecodeIcb` (`rg -uu -c`, no match). `Decode::step` encodes live. The ICB machinery targets the "mini" and "E4B" Hot layer graphs, and `src/cb_replay.rs:51` names the external `gemma-metal` crate as its only ledger consumer. Treat "decode ICB" in `AGENTS.md:10` as a capability of the crate, not a property of the Qwen3.5 decode loop.
- Env knobs live in `README.md`, "Environment Variables" (`TESSL_DECODE_ICB`, `TESSL_ICB_EXECUTE`, `TESSL_ICB_PIPELINES`, `TESSL_ICB_PREBUILT_TABLES`).

### 5.7 MLX Q4 and Q8 quantization, int8 GEMM

- MLX affine 4-bit: the loader accepts only affine, 4 bits, group size 64, uniform (`MlxQuant::from_config_json`, `src/qwen35_q4.rs:204`). Kernels: `gemv_q4_mlx` (`src/nn.rs:4397`), `gemv_q4_mlx_simd` (`:4647`), `gemm_q4_mlx_tensorops` (`:5350`). Decode uses the GEMV; prefill uses the small-batch kernel up to 8 rows and the TensorOps kernel above (`src/qwen35_model.rs:681-682`).
- Q8: `gemv_q8` (`src/nn.rs:825`) and the group-wise `gemm_q8_tensorops` (`:961`). Int8 and int4-weight pipelines: `gemm_i8` (`src/i8_pipeline.rs:203`), `gemm_w4a8` (`:326`); the legacy `nn::gemm_i8_dequant` is at `src/nn.rs:5551`.
- A model loaded with `load_mlx_q4` refuses training, parameter access and the optimizer (`src/qwen35_model.rs:682-683`).
- `QUANT_PREFILL_GEMM_WIRED` is `false` (`src/mtl_tensor.rs:150`); the host `MTLTensor` helpers are behind the `quant-prep` feature (`Cargo.toml:70`).
- Detail: [qwen35.md](qwen35.md) ("MLX 4-bit weights"), [architecture.md](architecture.md#integer-gemm).

### 5.8 EmbeddingGemma 2

- Text: `EmbedGemma2Model::load` (`src/embedgemma2.rs:1894`) and `encode` (`:2128`). Multimodal facade: `EmbedGemma2Multimodal::load` (`src/embedgemma2_mm.rs:256`) and `encode` (`:350`) over `MmItem` (`:153`: text, image, audio, video, timestamped video, mixed media). Towers: `src/embedgemma2_vision.rs` (`encode_images` `:1375`) and `src/embedgemma2_audio.rs` (`encode_audio` `:2300`); preprocessing, tokenizer and media decoding are `src/embedgemma2_pre.rs`, `src/embedgemma2_tok.rs`, `src/media/`.
- Detail, tolerances and measured errors: [embedgemma2.md](embedgemma2.md) and [embedgemma2_audit.md](embedgemma2_audit.md). Do not copy numbers from here.

### 5.9 BERT and DistilBERT

- `BertSparseModel::load` (`src/bert.rs:744`) and `encode` (`:840`), configured by `BertConfig` (`:427`): learned sparse document encoders with fused LayerNorms, exact erf GELU and a segment sparse max reduction. Detail: [bert.md](bert.md).

### 5.10 TimesFM 3

- `Timesfm3Model::load` (`src/timesfm3.rs:329`), `decode` (`:391`), `predict_batch` (`:422`), `forward` (`:472`); host steps in `src/timesfm3_host.rs`, batching in `src/timesfm3_forecast.rs`; CSV command line is the `timesfm3_forecast` binary (`Cargo.toml:163-164`, `src/bin/timesfm3_forecast.rs:232`). Detail and limits: [timesfm3.md](timesfm3.md).

### 5.11 PyTorch interop (`python/tessl_torch`)

- `python/tessl_torch/__init__.py` (`:58`) loads `libtessl.dylib` with `ctypes` and refuses a library whose `tessl_abi_version()` differs from `_ABI_VERSION = 14` (`python/tessl_torch/__init__.py:286-287`). Live surface: `cross_entropy` (`:677`), `cross_entropy_rows` (`:527`), `clip_coef` (`:209`), and `Qwen35` for loading a checkpoint and reading or writing parameters (`:728`, methods `parameters` `:858`, `load_parameters` `:864`, `read_parameter` `:885`, `describe` `:930`). `python/tessl_torch/embedgemma2.py` holds `EmbedGemma2` (`:827`).
- Example (lifted from `python/README.md`, top block, one comment line dropped):

```python
import sys; sys.path.insert(0, "python")
import tessl_torch
loss = tessl_torch.cross_entropy(hidden[:, :-1], weight, targets, mask)
loss.backward()
```

- Detail: [python/README.md](../python/README.md). Its banner says the Qwen3.5 training methods are retired; the sections below the banner are kept as the ABI 13 record.

### 5.12 Go binding and `tessl-embed`

- Package `github.com/bharathvbcr/tessl/go/embedgemma2`: `embedgemma2.Load` (`go/embedgemma2/model.go:154`), `Model.EmbedTexts` (`go/embedgemma2/texts.go:63`). It is a binding over the EmbeddingGemma 2 C ABI carried across the boundary by gusset; the umbrella staticlib `go/engine` is gusset plus tessl and exports `tessl_go_engine_init` (`go/engine/src/lib.rs:126`).
- Example (lifted from `go/README.md`, top block, trimmed: the error-check line and the later calls are dropped):

```go
m, err := embedgemma2.Load(snapshotDir, embedgemma2.Options{}) // exact f32; BF16 and TextOnly are options
defer m.Close()
q, _ := m.EmbedTexts(ctx, []string{"how do I speed up attention?"}, "query", 0)
```

- `tessl-embed` (`go/cmd/tessl-embed/main.go`) serves Ollama's `/api/embed` on `127.0.0.1`, default port 11435 (`go/cmd/tessl-embed/main.go:56`).
- Detail: [go/README.md](../go/README.md).

## 6. How to read the codebase

1. Start at `src/lib.rs:238-292` for the module list; the public re-exports follow at `:294-317`.
2. Runtime first (`src/runtime.rs`, `src/dispatch.rs`, `src/shared.rs`), then `src/tensor.rs`, then the kernel family you need.
3. For a model, read its module doc comment before its code; they are long and current (`src/qwen35_q4.rs:1-40` is a good example).
4. DevMap: the index for this checkout is partial. DevMap's Metal (`kernels/*.metal`) parse is degraded, so an empty caller list for a kernel means unknown, not none (AUDIT_REPORT.md, "Map limits", dated 2026-11-01, not re-run). Fall back to `rg -uu` with `-g '!target*'`.

## 7. Specifications and contracts

### 7.1 Metal 4 only

Encoding uses `MTL4CommandBuffer`, `MTL4ComputeCommandEncoder`, `MTL4ArgumentTable` and `MTLResidencySet`; the classic `MTLCommandQueue` family is pruned from the `objc2-metal` feature list (`Cargo.toml:86-94`, comment at `:86-87`). `MTLIndirectCommandBuffer` features are enabled for the opt-in ICB path (`Cargo.toml`, "Compute ICB (D16)" comment). `GpuRuntime::has_tensorops()` reports whether the device supports TensorOps (`src/runtime.rs:1501`).

### 7.2 Host waits

The contract in `AGENTS.md:26` ("host threads never block mid-step") holds on the batched path only. Evidence:

- The default is not batched: `async_encode` starts `false` (`src/runtime.rs:1472`), and [architecture.md](architecture.md#metal-4-execution-model--zero-host-wait-pipeline) labels solo submit-and-wait the default mode.
- Batching is opt-in: `set_async_encode(true)` (`src/runtime.rs:2234`) or an open `AsyncScope` (`async_scope` at `:2280`, doc above it); the host then waits once at `synchronize` (`:3347`). `set_async_encode(false)` synchronizes first (`:2235-2237`).
- Some entry points wait by design: `Decode::step` waits for its logits each token (`src/qwen35_model.rs:2420-2421`; [qwen35.md](qwen35.md) "Not done", "Decode, remaining gaps"), and `SharedBuf::read` / `write` wait for the GPU stamps (`src/shared_buf.rs:241-262`, `:279-292`).
- Scalar binds come from a bump-allocated constant arena of 16 MiB (`METAL4_CONST_ARENA_BYTES`, `src/runtime.rs:50`); synchronization uses `MTLSharedEvent` (`src/runtime/host_fence.rs:26`).
- Waits are bounded: `DEFAULT_EVENT_WAIT_MS = 30_000` (`src/runtime.rs:3475`), overridable with `TESSL_EVENT_WAIT_MS` (`:1804`); the doc at `:1832` describes a streak of timeouts.

### 7.3 `SharedBuf` leases

- Read lease: any number may be live; it waits until the GPU has completed every writer, and is refused with `HostAccessError::Busy` while a write lease is live (`src/shared_buf.rs:241-262`). Write lease: exclusive, waits for every reader and writer (`:279-292`). A refused call holds no lease.
- A GPU bind of a buffer under a conflicting host lease is refused, and a shared buffer is refused during an ICB capture (`src/shared_buf.rs:1047` test name `shared_binds_are_refused_during_an_icb_capture`; module doc `:1-60`).
- Errors: `HostAccessError` has `NotSubmitted`, `Busy`, `Runtime` (`:81-96`). Through `MappedBuffer` they surface as `ojas_core::FenceError` (`src/runtime/host_fence.rs:165-185`).
- `GpuRuntime` is `!Send` and `!Sync`; a `compile_fail` doctest pins it (`src/runtime.rs:810-815`). Host access fails instead of blocking when the runtime is busy or re-entered (`RuntimeAccess`, `:798-800`).

### 7.4 Fail-closed validation

- GEMM: `validate_gemm` (`src/gemm.rs:44`) requires rank-2, non-empty tensors, extents at most `i32::MAX` elements (`require_i32_extent`, `:34`), 16-byte-aligned views (`GEMM_VIEW_ALIGN`, `:102`), one runtime for all operands, `C` in f32, matching inner dimensions, and an output that does not overlap an input.
- `nn`: the wrappers reject undersized buffers before encoding (`src/nn.rs:9-17`; the shared helper `require` at `:136` is crate-private).
- Loaders: `Qwen35Model::load` checks every tensor shape against the config and a longer layer stack is refused through `Loader::refuse_unread` (the call is `src/qwen35_model.rs:936`; module doc `:34-35`).
- `tests/capi_contract.rs` and `tests/capi.rs` pin C-ABI refusals.

### 7.5 C ABI

There is no C header file in the repository: `git ls-files` lists only `kernels/*.h`, and there is no `include/` directory. The ABI is the `extern "C"` items with `#[no_mangle]` in two files, and the Python and Go bindings declare their own signatures. Treat the Rust source as the header.

- Version: `TESSL_ABI_VERSION = 14` (`src/capi.rs:91`), returned by `tessl_abi_version` (`:187`). A new version is required for any `#[repr(C)]` layout or signature change (`src/capi.rs:82-83`). Two places are verified to agree: the constant and `python/tessl_torch/__init__.py:58`; `tests/capi_contract.rs:109` fails when the Python line is stale and `tests/capi.rs:1345` asserts 14. How the Go engine tracks the version was not read (unverified).
- Status codes: `TESSL_OK = 0`, `TESSL_ERR = 1`, `TESSL_PANIC = 2` (`src/capi.rs:78-80`). Tensor dtype codes `TESSL_F32`, `TESSL_BF16`, `TESSL_F16`; `TESSL_MAX_DIMS = 6` (`:94-99`); tensors cross as `TesslTensorRef` (`:129`, a caller-owned `MTLBuffer` plus offset, dtype, shape).
- Exports in `src/capi.rs` (17 functions, each `#[no_mangle]`, `rg -c` count 17): `tessl_abi_version` (`:187`), `tessl_clip_coef` (`:202`), `tessl_runtime_new` (`:255`), `tessl_runtime_free` (`:290`), `tessl_runtime_poisoned` (`:226`), `tessl_synchronize` (`:300`), `tessl_runtime_set_async_encode` (`:317`), `tessl_mtl_buffer_length` (`:377`), `tessl_cross_entropy_rows` (`:398`), and the Qwen3.5 checkpoint family `tessl_qwen35_load` (`:858`), `_free`, `_param_count`, `_param_info`, `_copy`, `_read_parameter_host`, `_write_parameter_host`, `_describe` (`:959-1162`).
- Exports in `src/capi_embedgemma2.rs` (27 functions, each `#[no_mangle]`): load and lifetime (`tessl_embedgemma2_load` `:346`, `_load_multimodal` `:383`, `_free` `:422`), model queries (`_embedding_dim`, `_max_tokens`, `_set_max_tokens`), raw encode (`_encode` `:589`, `_encode_images` `:667`, `_encode_audio` `:761`), prompts (`_prompt_count`, `_prompt_name`, `_prompt_get`, `_default_prompt`), tokenizer (`_tokenize` `:1181`, `_decode` `:1246`), the `embed_*` family from `:1292` (texts, images, audio, videos, media, sources, segments) and the `*_soft_token_count` queries.
- Retired at ABI 14 (do not call or re-add): `tessl_qwen35_train_step`, `_step_ragged`, `_forward`, `_backward`, `_discard`, `tessl_qwen35_hidden`, the `tessl_qwen35_adamw_*` set, `tessl_qwen35_grad_sq_norm`, `tessl_gdn_train_forward`/`_backward`, and the gradient and AdamW directions of `tessl_qwen35_copy` (`src/capi.rs:85-90`).
- Handles are thread-affine: `TesslRuntime` records its owner thread (`src/capi.rs:175-181`). The embedgemma2 module says every entry runs through a shared `guarded` wrapper that refuses null and foreign-thread handles and catches panics (`src/capi_embedgemma2.rs:30-34`); `tessl_runtime_poisoned` reports a latched fault and is the one call that works on a poisoned runtime (`src/capi.rs:212-226`).

## 8. Notes for AI coding agents

### Invariants a change must keep

`AGENTS.md:23-29` lists them; condensed with this page's evidence:

1. Metal 4 only; do not add a legacy command-queue path (`Cargo.toml:86`).
2. Keep host waits out of the batched step path (section 7.2).
3. All host access to shared memory goes through leases; never hand out a raw pointer that bypasses the timeline (`src/shared_buf.rs`).
4. No training or IO logic that duplicates ojas (section 2). A new optimizer or schedule goes into ojas.
5. Validate before encoding; return a typed or descriptive error; never corrupt GPU state.
6. A C-ABI layout or signature change bumps `TESSL_ABI_VERSION`, the Python `_ABI_VERSION` and the Go side together, and needs the ojas handshake (`docs/WORKFLOWS.md`, "No Breaking Interface Drift").

### Verification discipline

- A check that cannot run must not read as a pass. The repo enforces this in scripts (`scripts/check_siblings.sh:32-33`, `scripts/hosted_tests.sh` exact counts, `scripts/ci_local.sh` named skips). Do the same in reports.
- Metal tests cannot run on hosted CI or off-device; `tools/msl_emu` is a CPU emulator that cannot execute TensorOps, so TensorOps kernels have no off-device numeric check (`README.md`, "Known Gaps", `attn_prefill` row).
- Wrap heavy GPU work in the machine lock named in `docs/WORKFLOWS.md` (an ojas-side script path outside this repo; not verified here) and run GPU tests single-threaded.
- Recent landings on main were merged with no tests run by owner order (git log, merges `ab4f7d5`, `d14ac92`, 2026-11-21). Do not assume main is green; BUILD_REPORT.md (local, gitignored, 2026-11-21) lists the unrun post-landing verifies.

### Pitfalls seen in the record

| Pitfall | Evidence |
|---|---|
| `TESSL_HAZARD_BARRIERS=1` removes the always-on dispatch barrier; the README row was inverted until 2026-08-31 and the result is wrong output | README.md, "Environment Variables" |
| Test and bench binaries live under `target/<profile>/build/<pkg>/<hash>/out/` with Cargo 1.100, not `deps/` | CHANGELOG.md, Unreleased, "Cargo 1.100 build-dir layout" |
| Adding or renaming a test breaks the hosted manifest until its count is updated; a miss on main was fixed forward in `fba6fc0` | `scripts/hosted_tests.sh:1-24`; BUILD_REPORT.md |
| Docs still describing `train_step` as live are stale; copying a present-tense fence calls a method ABI 14 removed | AUDIT_REPORT.md (2026-11-01, "ABI 14 prose"); section "Doc drift noticed" below |
| Do not cite a number from this repo's older docs without its date; the docs carry "read at" provenance for a reason | [benchmarking.md](benchmarking.md) |
| Worktrees of tessl need an `ojas` sibling at the matching depth, or the path dependency fails | `Cargo.toml:126-127`; BUILD_REPORT.md "Storage" paragraph |

## 9. Doc drift noticed

| Doc | Says | Code says |
|---|---|---|
| `docs/README.md` (Models table, Qwen3.5 row and PyTorch Interop row) | Qwen3.5 covers the "training backward pass" and "fused AdamW"; `tessl_torch` offers "full model `train_step` integration" and "GDN seam patching" | `Qwen35.train_step` and `patch_transformers_qwen3_5` do not exist in `python/tessl_torch/__init__.py` (`rg` finds only `cross_entropy`, `cross_entropy_rows`, `clip_coef`, `Qwen35` loading and parameter methods); `src/capi.rs:85-90` documents the retirement |
| `docs/qwen35.md`, "Not done", "Training, remaining gaps", and section "A training step: `Qwen35Model::train_step`" | `train_step` and `train_step_ragged_into` run the whole step on the GPU | no `fn train_step` anywhere in `src/` or `tests/` (`rg -uu`); CHANGELOG.md lists `Qwen35Model::train_step*` as removed |
| `python/README.md`, "Training a whole Qwen3.5 model" | shows `adamw_init`, `train_step`, `adamw_step` | removed at ABI 14; the file's own banner says so, but the block remains as the ABI 13 record |
| `python/README.md`, "Build and test" | lists `test_gdn.py` | `python/tests/` has no `test_gdn.py` |
| `Qwen35` class docstring, `python/tessl_torch/__init__.py:728-729` | "A Qwen3.5 text model trained by tessl's ``train_step``" | the class has no `train_step`; it only loads and exposes parameters |
| `AGENTS.md:10` and `:27` | "decode ICB and ping-pong replay" are listed in the Mac inference engine role beside Qwen3.5 prefill and decode; leases are `HostLease` / `HostLeaseMut` | `DecodeIcb` is unreferenced from the Qwen3.5 modules; `HostLease`, `HostLeaseMut` and `FenceError` are `ojas_core` types, tessl's own are `HostRead` / `HostWrite` and `HostAccessError` (`src/shared_buf.rs:86`, `:460`, `:491`) |
| `AGENTS.md:27` and `:29` | `PoolLedger` is named in `src/pool_ledger.rs` beside `BudgetTrimHandler`; `nn::require` is named as a validator | `PoolLedger` is a trait (`src/pool_ledger.rs:123`); the concrete type is `OjasBudgetLedger` (`:196`), and `nn::require` is crate-private (`src/nn.rs:136`) |
| `docs/architecture.md` mermaid diagram | `ojas-qwen35` hosts the `TapeStepProvider` | `TapeStepProvider` is in `ojas-model/src/qwen35/step_provider.rs:627` (ojas repo) |

## 10. Unverified on this page

- That any command in section 4 passes at `fba6fc0`: none was run (read-only task).
- The macOS 26 and Metal 4 device requirement beyond the README, Cargo comments and `build.rs` header; the device probe in `GpuRuntime::new` was not read end to end.
- `gusset` internals and the cancellation behaviour inside gusset (its `wait.go`, named in AUDIT_REPORT.md): read through `go/README.md` only.
- `ojas-metal`'s use of tessl beyond the Cargo dependency line (the "How ojas uses tessl" section of architecture.md is dated 2026-11-02 and was not re-checked against the ojas tree).
