//! Metal 4 GEMM, encode runtime, and neural-network kernel library for Apple
//! silicon.
//!
//! Extracted from `arch_02_value_resid/metal-native`: encode path, tensors,
//! TensorOps/simdgroup GEMM, util kernels. **Metal 4-only** encode — residency,
//! packed encoder, no host-zero mid-CB (Audit 4/6 lessons preserved).
//!
//! # Promoted kernels
//!
//! 44 model-agnostic kernels moved here out of `gemma-metal` — RMSNorm, gated
//! MLP activations, flash attention, quantized GEMV/GEMM, KV-cache stores,
//! embedding lookup, sampling. They had lived in one model's crate, reachable
//! only as raw strings through an overlay metallib. [`nn`] is the typed surface
//! over the subset with a stable host-side contract; the rest are dispatched by
//! name through [`runtime::GpuRuntime::pipeline`] until that surface grows.
//!
//! Gemma-specific kernels stayed behind: Per-Layer Embeddings (`ple_lookup*`)
//! and the persistent-interpreter prototype (`persistent_interp*`).
//!
//! Quantized `MTLTensor` hooks (WWDC26-330) live in `mtl_tensor`, behind the
//! off-by-default `quant-prep` feature — prep only, the prefill entry point
//! still returns an error.
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
