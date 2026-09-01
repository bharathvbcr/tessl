//! Metal 4 GEMM, encode runtime, and neural-network kernel library for Apple
//! silicon, built on Metal Performance Primitives (MPP) TensorOps `matmul2d`.
//!
//! # Platform
//!
//! This crate is Apple-silicon only, and it does not degrade gracefully.
//! `objc2-metal` is an unconditional dependency and `build.rs` drives the Metal
//! shader compiler, so the crate **does not build at all** on Linux or on Intel
//! Macs. That is deliberate: a silent CPU fallback would make every "GPU"
//! benchmark here meaningless, so there isn't one.
//!
//! | Requirement | |
//! |---|---|
//! | OS | macOS 26 or newer |
//! | Hardware | Apple M-series with neural accelerators, for the TensorOps path |
//! | Toolchain | Xcode 26 with the Metal Toolchain component |
//! | Rust | 1.82+ |
//!
//! The Metal compiler is **not** part of Xcode. It is a separately downloaded
//! component, and without it `build.rs` fails at the shader compile step:
//!
//! ```sh
//! xcodebuild -downloadComponent MetalToolchain
//! ```
//!
//! # Quickstart
//!
//! ```no_run
//! use tessl::{gemm, GemmBackend, GpuRuntime};
//!
//! # fn main() -> Result<(), String> {
//! let rt = GpuRuntime::new()?;
//!
//! let a = rt.alloc_tensor_f32(&[4096, 2304])?;
//! let b = rt.alloc_tensor_f32(&[2304, 768])?;
//! let c = rt.alloc_tensor_f32(&[4096, 768])?;
//!
//! gemm(&a, &b, &c, GemmBackend::TensorOps)?;   // C = A @ B
//! rt.synchronize()?;
//! # Ok(())
//! # }
//! ```
//!
//! Neural-network kernels go through [`nn`], which validates every operand
//! before encoding anything:
//!
//! ```no_run
//! use tessl::{nn, GpuRuntime};
//!
//! # fn main() -> Result<(), String> {
//! let rt = GpuRuntime::new()?;
//! let (rows, dim) = (512u32, 4096u32);
//!
//! let x = rt.alloc_buffer(rows as usize * dim as usize * 4)?;
//! let weight = rt.alloc_buffer(dim as usize * 4)?;
//! let out = rt.alloc_buffer(rows as usize * dim as usize * 4)?;
//!
//! nn::rms_norm_f32(&rt, &x, &weight, &out, rows, dim, 1e-6)?;
//! rt.synchronize()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Module map
//!
//! | Module | What it holds |
//! |---|---|
//! | [`runtime`] | [`GpuRuntime`], buffer pools, residency sets, the encoder lease |
//! | [`gemm`](mod@gemm) | GEMM entry points, layout and precision selection, the fused epilogue |
//! | [`nn`] | Typed, shape-checked API over the promoted kernel library |
//! | [`tensor`] | [`Tensor`], [`GpuBuffer`], [`DType`], and the f16/bf16 conversions |
//! | [`dispatch`] | `Binder`, argument-table encoding, threadgroup dispatch helpers |
//! | [`decode_icb`], [`cb_replay`] | Indirect Command Buffer capture and ping-pong replay |
//! | [`ops`] | Elementwise helpers that are not part of a larger kernel family |
//! | `mtl_tensor` | Quantized `MTLTensor` prep, behind the `quant-prep` feature |
//! | [`ab_flags`], [`infer_trace`], [`icb_smoke`], [`npy`] | Tuning switches, tracing, smoke tests, and `.npy` I/O for benchmark parity |
//!
//! # Encode model
//!
//! Encoding is **Metal 4 only**: `MTL4CommandBuffer`, `MTL4ComputeCommandEncoder`,
//! `MTL4ArgumentTable`, `MTLResidencySet`. The legacy `MTLCommandQueue` path is
//! deliberately absent rather than merely unused.
//!
//! By default each dispatch gets its own command buffer and commits. That is
//! the safe shape for a caller who synchronizes after every step, and it is
//! also the expensive one: a submit-and-wait round trip is roughly 0.25 ms, so
//! a small kernel measures the driver rather than the shader. Call
//! [`runtime::GpuRuntime::set_async_encode`] with `true` and dispatches
//! accumulate into one command buffer until [`runtime::GpuRuntime::synchronize`].
//! On an elementwise kernel at n=4096 that is 203 µs against 4.1 µs, a factor
//! of about 49.
//!
//! # Kernels
//!
//! 18 Metal sources compile to 72 kernel entry points: RMSNorm, gated MLP
//! activations, flash attention (sliding-window and global), fused
//! RMSNorm+QKV+RoPE, MLX-format Q4 GEMV/GEMM, Q8 GEMV, an exact int8 GEMM, KV
//! cache stores, embedding lookup, row-wise softmax/sum/max, and softcap
//! sampling. [`nn`] exposes them through 62 shape-checked functions.
//!
//! 44 of these were promoted out of `gemma-metal`, where they were reachable
//! only as raw pipeline-name strings through an overlay metallib. All 44 now
//! carry a numeric test against an independent reference rather than a name
//! check; see the crate's `tests/` directory and `docs/verification.md`.
//!
//! Gemma-specific kernels stayed behind: Per-Layer Embeddings (`ple_lookup*`)
//! and the persistent-interpreter prototype (`persistent_interp*`).
//!
//! # Feature flags
//!
//! | Feature | Default | |
//! |---|---|---|
//! | `quant-prep` | off | Compiles `mtl_tensor`'s quantized `MTLTensor` path. Prep only: the prefill entry point returns an error and nothing dispatches it. Kept compiling behind a flag rather than shipped as public API that does not work. |
//!
//! # Performance
//!
//! Apple M5 Pro, paired interleaved rounds against torch 2.13 (MPS) and MLX,
//! geomean of per-shape medians over an eight-shape ladder:
//!
//! | Comparison | Geomean | Shapes below 1.0 |
//! |---|---|---|
//! | tf32-relaxed vs. MPS f32 | 2.11× | 0 of 8 |
//! | bf16 vs. MLX bf16 | 2.55× | 0 of 8 |
//! | f32 exact vs. MPS f32 | 1.12× | 2 of 8 |
//! | bf16 vs. MPS bf16 | 1.03× | 4 of 8 |
//!
//! bf16 against Apple's own bf16 GEMM is **parity, not a win**. The result
//! worth quoting is the tf32 lane. Peak observed throughput is 26,642 GFLOP/s
//! (bf16, 4096³), 16,293 (tf32) and 6,431 (f32 exact).
//!
//! Reproduce with `cargo run --release --bin bench_gemm_sweep` and
//! `python3 bench/paired_cross_runtime.py --rounds 5 --lanes torch,mlx`. Single
//! runs on this hardware fluctuate 15–20% as the power governor moves, so
//! cross-runtime claims come from paired sweeps only.
//!
//! # Notes for contributors
//!
//! **Tests need `--test-threads=1`.** GPU tests share default command encoders
//! across threads and are not safe to run concurrently. The integration suite
//! serializes itself with a mutex regardless, because a suite that only passes
//! when the caller remembers a flag is a suite that fails in CI.
//!
//! **A `.metal` edit must reach the build.** Kernel sources are compiled
//! ahead-of-time by `build.rs` into a metallib, and shader files reach that
//! build through a path Cargo does not infer. `build.rs` therefore emits
//! `rerun-if-changed` for every `.metal` **and `.h`** individually — not for the
//! directory, whose mtime does not move when a file is edited in place. Without
//! that, an edited kernel silently stays stale while the tests report a pass.
//!
//! **Name checks are not correctness tests.** `tests/promoted_kernels.rs`
//! asserts each promoted entry point resolves from this crate's own metallib.
//! That gates the migration, not the arithmetic: a kernel can resolve, dispatch,
//! and return wrong numbers, and one of them wrote 4 output rows out of 512
//! while passing that test and an adversarial error-path test. New kernels need
//! a numeric test against an independent reference.
//!
//! **Seed outputs with a sentinel when testing.** Zero-filling makes "never
//! written" indistinguishable from "written correctly", because zero is a
//! plausible result. Several tests here fill the output with an implausible
//! value first and assert none of it survives.
//!
//! **Where a kernel family is selected by an enum or a bool, test every arm.**
//! Two dispatch-geometry bugs lived in non-default arms of families whose
//! default arm was tested and correct.
//!
//! # Unsafe code
//!
//! The `unsafe` in this crate falls into two classes, and only one of them
//! carries an obligation worth writing down per call site.
//!
//! **Objective-C message sends.** `objc2` marks every Metal protocol method
//! `unsafe`, because the bindings cannot prove the receiver is live or that the
//! arguments satisfy a contract stated only in Apple's documentation. These are
//! the large majority. The obligation each one discharges is the same: the
//! receiver is a `Retained` handle this crate owns, so it is alive for the
//! call; the arguments are bounds-checked before the send wherever Metal states
//! a range; and encoder methods run inside [`runtime::GpuRuntime::with_binder`],
//! which holds the encoder lease that makes concurrent encoding impossible.
//! They are not individually annotated — 48 copies of that paragraph would bury
//! the ones that matter.
//!
//! **Raw pointer and slice construction.** Reinterpreting a Metal buffer's
//! `contents()` as a typed slice, or a `Vec`'s storage as bytes. These carry
//! real, site-specific obligations — pointer validity, length, alignment,
//! aliasing, and whether the GPU could be writing the same bytes — and every
//! one of them has a `// SAFETY:` comment stating which invariant makes it
//! sound and where that invariant is established.
//!
//! If you add an `unsafe` block that is not a bare message send, it belongs in
//! the second class: write the comment.

pub mod ab_flags;
pub mod cb_replay;
pub mod decode_icb;
pub mod dispatch;
pub mod gemm;
pub mod icb_smoke;
pub mod infer_trace;
#[cfg(feature = "quant-prep")]
pub mod mtl_tensor;
pub mod nn;
pub mod npy;
pub mod ops;
pub mod runtime;
pub mod tensor;

pub use cb_replay::{
    cb_replay_api_gap_summary, survey_cb_replay_api_gaps, ArgTableSlot, ArgTableSlotPlan,
    CbReplayApiGap, CbReplayError, CbReplayPhase, CbSlot, IcbCommandTypeHint, IcbReplayStub,
    IcbStubPhase, PingPongCbReplay,
};
pub use decode_icb::{
    begin_decode_icb_capture, binder_encode_nop, decode_icb_capture_active, decode_icb_enabled,
    icb_coarse_ranges_enabled, icb_freeze_binds_enabled, icb_pipelines_enabled,
    icb_range_batch_enabled, pipeline_icb, set_binder_encode_nop, set_decode_icb,
    set_icb_coarse_ranges, set_icb_freeze_binds, set_icb_pipelines, set_icb_range_batch,
    take_decode_icb_capture, BinderEncodeNopGuard, DecodeIcb, DecodeIcbBind, DecodeIcbCapture,
    DecodeIcbCommand,
};
pub use gemm::{
    cast_f16_to_f32, cast_f32_to_f16, gemm, gemm_batched, gemm_epilogue, gemm_f32, Activation,
    BatchStrides, BatchedGemm, Epilogue, GemmBackend,
};
pub use icb_smoke::{
    icb_smoke_enabled, run_copy_f32_smoke, set_icb_smoke, IcbBindBridge, IcbCopySmoke,
};
#[cfg(feature = "quant-prep")]
pub use mtl_tensor::{
    nax_verify_readiness, NaxVerifyReadiness, QuantDType, QUANT_PREFILL_GEMM_WIRED,
};
pub use ops::softcap_f32;
pub use runtime::{BufferKind, DeviceMemoryInfo, GpuRuntime, PrecisionMode};
pub use tensor::{DType, GpuBuffer, Tensor};

/// Metallib produced by `build.rs` (absolute path baked at compile time).
pub fn metallib_path() -> &'static str {
    env!("TESSL_METALLIB")
}
