//! MTLTensor / quantized TensorOps prep (WWDC26-330).
//!
//! Metal 4 can feed native quantized tensors into TensorOps matmul (auto-dequant
//! on NAX). Prefer this for **prefill** GEMM once Q4 banks exist; keep hand GEMV
//! for decode (M=1) until proven otherwise.
//!
//! objc2-metal 0.3 exposes `MTLTensorDataType::{Int8, …}` today. Int4 / FP8 E8M0
//! scale planes land in later SDKs — callers should treat [`QuantDType`] as the
//! stable surface and map when bindings appear.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::NSInteger;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLDevice, MTLResourceID, MTLResourceOptions, MTLSizeAndAlign,
    MTLTensor, MTLTensorDataType, MTLTensorDescriptor, MTLTensorExtents, MTLTensorUsage,
};
use std::sync::Arc;

use crate::dispatch::Binder;
use crate::runtime::{GpuRuntime, ARGUMENT_TABLE_MAX_BUFFERS};
use crate::tensor::GpuBuffer;

/// Logical quantized element type for future TensorOps / GEMV paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantDType {
    /// Native `MTLTensorDataTypeInt8` (macOS 26+).
    Int8,
    /// Planned WWDC26-330 Int4 tensor path — not in objc2-metal 0.3 yet.
    Int4,
    /// Planned FP8 E8M0 scale-plane path (macOS 27+ per Apple notes).
    Fp8E8M0,
}

impl QuantDType {
    /// Map to objc2 `MTLTensorDataType` when the SDK binding exists.
    pub fn to_mtl(self) -> Result<MTLTensorDataType, String> {
        match self {
            QuantDType::Int8 => Ok(MTLTensorDataType::Int8),
            QuantDType::Int4 => Err(
                "MTLTensorDataType Int4 is not in objc2-metal 0.3 (WWDC26-330), so a \
                 host-created Int4 MTLTensor cannot be described. Note this gates only \
                 the host descriptor path: TensorOps itself accepts int4b_format, and \
                 kernels building tensors from device pointers are unaffected."
                    .into(),
            ),
            QuantDType::Fp8E8M0 => Err(
                "FP8 E8M0 MTLTensor scale planes require newer Metal SDK (macOS 27+ notes)".into(),
            ),
        }
    }

    pub fn bytes_per_elem_hint(self) -> f32 {
        match self {
            QuantDType::Int8 => 1.0,
            QuantDType::Int4 => 0.5,
            QuantDType::Fp8E8M0 => 1.0,
        }
    }
}

/// Snapshot of TensorOps / NAX readiness for verify(M) / prefill planning.
///
/// Verify(M) remains on hand-written simdgroup Q4 GEMM
/// (`gemm_q4_mlx_simd*`) because tessl does not yet ship the shader-side
/// sub-byte TensorOps constructor and wired host dispatch that raw-address Q4
/// needs. The missing Int4 host descriptor binding in objc2-metal 0.3 is
/// reported separately; it does not block that raw-address kernel design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NaxVerifyReadiness {
    pub int8_tensorops_dtype: bool,
    pub int4_tensorops_dtype: bool,
    pub fp8_e8m0_tensorops_dtype: bool,
    pub quant_prefill_gemm_wired: bool,
    pub note: &'static str,
}

/// Probe dtype binding + documented wire status (no device `newTensor` calls).
pub fn nax_verify_readiness() -> NaxVerifyReadiness {
    NaxVerifyReadiness {
        int8_tensorops_dtype: QuantDType::Int8.to_mtl().is_ok(),
        int4_tensorops_dtype: QuantDType::Int4.to_mtl().is_ok(),
        fp8_e8m0_tensorops_dtype: QuantDType::Fp8E8M0.to_mtl().is_ok(),
        // Const, not a call into a stub that returns `Err` so this can read
        // `.is_ok()` off it. There is one fact here — quantized TensorOps
        // prefill GEMM does not exist — and it is stated once.
        quant_prefill_gemm_wired: QUANT_PREFILL_GEMM_WIRED,
        note: "TensorOps Q4 is not shipped: raw-address kernels lack a shader-side sub-byte tensor constructor and wired host dispatch; the missing Int4 host descriptor binding is separate",
    }
}

/// Whether a quantized TensorOps prefill GEMM exists **through this module's
/// host-side `MTLTensor` path**.
///
/// It does not, and the distinction matters more than the flag. There was a
/// `try_quant_tensorops_prefill_gemm` whose entire body was `Err`, with no
/// caller and no test; it is gone.
///
/// What was missing was misdiagnosed here for some time. The note used to say
/// quantized TensorOps was blocked because `MTLTensorDataType::Int4` is unbound
/// in objc2-metal 0.3. That binding gates *host-created* `MTLTensor`
/// descriptors, which is what this module builds — and it is irrelevant to a
/// kernel that constructs its tensors from raw device pointers, which is what
/// every kernel in `kernels/` does.
///
/// So quantized TensorOps is **not** blocked in general:
/// [`crate::nn::gemm_i8_dequant`] ships an `int8 x int8 -> int32` GEMM with the
/// dequantization fused, needing nothing from this module. The header's own
/// diagnostic lists the supported cooperative source types as
/// `uint8_t/int8_t/uint4b_format/int4b_format/float/half/bfloat`, so Int4 is
/// supported by TensorOps too; what is missing there is the shader-side tensor
/// constructor for a sub-byte element type, not an objc2 binding.
pub const QUANT_PREFILL_GEMM_WIRED: bool = false;

/// Owned MTLTensor handle (device-allocated or buffer-backed).
///
/// The native handle is deliberately read-only through the safe API. Replacing
/// it would detach the object from the storage, runtime, and residency policy
/// recorded alongside it, so callers receive only a borrowed handle via
/// [`Self::metal`].
///
/// ```compile_fail,E0616
/// use tessl::mtl_tensor::GpuTensor;
/// fn cannot_replace_the_native_handle(tensor: &mut GpuTensor) {
///     let _replacement_slot = &mut tensor.tensor;
/// }
/// ```
pub struct GpuTensor {
    tensor: Retained<ProtocolObject<dyn MTLTensor>>,
    pub dtype: MTLTensorDataType,
    pub dims: Vec<usize>,
    // Lifetime anchors, never read: the MTLTensor above borrows this storage,
    // and the runtime owns the allocator that storage came from. Dropping
    // either while `tensor` is live is a use-after-free, so they are held, not
    // used. Scoped `allow` rather than a crate-level one, which would hide the
    // next genuinely dead field.
    #[allow(dead_code)]
    pub(crate) storage: Option<GpuBuffer>,
    #[allow(dead_code)]
    pub(crate) runtime: Arc<GpuRuntime>,
    /// Device-created tensors own a distinct `MTLAllocation` and therefore a
    /// distinct residency registration. Buffer-backed tensor views borrow the
    /// registration held by `storage` and must not double-register it.
    residency_registered: bool,
}

impl GpuTensor {
    /// Borrow the native tensor without allowing its ownership metadata to be
    /// separated from it.
    pub fn metal(&self) -> &ProtocolObject<dyn MTLTensor> {
        &self.tensor
    }

    pub fn gpu_resource_id(&self) -> MTLResourceID {
        self.tensor.gpuResourceID()
    }

    pub fn data_type(&self) -> MTLTensorDataType {
        self.tensor.dataType()
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        if self.residency_registered {
            // The runtime retains this allocation until every command that
            // could reference it has completed, then removes it from the
            // residency set without putting it into the buffer freelist.
            self.runtime.schedule_tensor_retirement(self.tensor.clone());
        }
    }
}

/// Bind an MTLTensor into the Metal 4 argument table via `setResource:atBufferIndex:`.
///
/// `index` is the buffer slot, and the table has
/// [`ARGUMENT_TABLE_MAX_BUFFERS`] of them — so the last valid slot is
/// `ARGUMENT_TABLE_MAX_BUFFERS - 1`, not `ARGUMENT_TABLE_MAX_BUFFERS`. Metal
/// does not range-check `setResource:atBufferIndex:`: an out-of-range slot
/// writes past the table (a large one segfaults outright), which is why this
/// safe wrapper rejects it instead of passing it through.
pub fn bind_mtl_tensor(bnd: &mut Binder<'_>, t: &GpuTensor, index: usize) -> Result<(), String> {
    if !bnd.belongs_to_runtime(t.runtime.as_ref()) {
        return Err("MTLTensor belongs to another runtime".into());
    }
    if index >= ARGUMENT_TABLE_MAX_BUFFERS {
        return Err(format!(
            "MTLTensor bind index {index} out of range: argument table has \
             {ARGUMENT_TABLE_MAX_BUFFERS} buffer slots (0..={})",
            ARGUMENT_TABLE_MAX_BUFFERS - 1
        ));
    }
    // The Binder takes its own +1 retain on the MTLTensor object before writing
    // the resource ID. Metal 4 command buffers use unretained resources, so the
    // caller may drop `t` (including a buffer-backed view) as soon as this
    // closure returns without invalidating the encoded ID. The anchor transfers
    // to the allocator slot on submission and releases only after completion.
    bnd.bind_mtl_tensor_resource(&t.tensor, index);
    Ok(())
}

/// Probe whether the device can size an MTLTensor with the given dtype/shape.
///
/// **Experimental:** some objc2 / SDK combinations have SIGSEGV'd on this
/// selector for unsupported layouts — gate behind Phase-2 smoke before use.
pub fn probe_tensor_support(
    rt: &GpuRuntime,
    dtype: QuantDType,
    dims: &[usize],
) -> Result<MTLSizeAndAlign, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    Ok(rt.device.tensorSizeAndAlignWithDescriptor(&desc))
}

/// Allocate a device-backed MTLTensor (no storage shared with [`GpuBuffer`]).
pub fn alloc_device_tensor(
    rt: &Arc<GpuRuntime>,
    dims: &[usize],
    dtype: QuantDType,
) -> Result<GpuTensor, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    let tensor = rt
        .device
        .newTensorWithDescriptor_error(&desc)
        .map_err(|e| format!("newTensorWithDescriptor: {e}"))?;
    rt.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*tensor));
    Ok(GpuTensor {
        tensor,
        dtype: mtl_dtype,
        dims: dims.to_vec(),
        storage: None,
        runtime: Arc::clone(rt),
        residency_registered: true,
    })
}

/// Wrap an existing shared buffer as an MTLTensor view (offset must satisfy align).
///
/// # Safety
/// The caller must ensure no incompatible live tensor view aliases the requested
/// buffer range. Alignment, bounds, size arithmetic, and runtime ownership are
/// validated here before the Metal selector is called.
pub unsafe fn tensor_from_buffer(
    rt: &Arc<GpuRuntime>,
    buf: &GpuBuffer,
    byte_offset: usize,
    dims: &[usize],
    dtype: QuantDType,
) -> Result<GpuTensor, String> {
    if !buf.belongs_to(rt) {
        return Err("MTLTensor buffer belongs to another runtime".into());
    }
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute)?;
    let align = rt.device.tensorSizeAndAlignWithDescriptor(&desc);
    validate_tensor_buffer_range(buf.nbytes(), byte_offset, align.size, align.align)?;
    let tensor = buf
        .metal()
        .newTensorWithDescriptor_offset_error(&desc, byte_offset as _)
        .map_err(|e| format!("buffer.newTensor: {e}"))?;
    Ok(GpuTensor {
        tensor,
        dtype: mtl_dtype,
        dims: dims.to_vec(),
        storage: Some(buf.clone()),
        runtime: Arc::clone(rt),
        residency_registered: false,
    })
}

fn make_descriptor(
    dims: &[usize],
    dtype: MTLTensorDataType,
    usage: MTLTensorUsage,
) -> Result<Retained<MTLTensorDescriptor>, String> {
    if dims.is_empty() || dims.len() > 16 {
        return Err(format!("MTLTensor rank {} out of range 1..=16", dims.len()));
    }
    if let Some(&extent) = dims.iter().find(|&&d| d > NSInteger::MAX as usize) {
        return Err(format!("MTLTensor extent {extent} does not fit NSInteger"));
    }
    let desc = MTLTensorDescriptor::new();
    let extents = extents_from_dims(dims)?;
    desc.setDimensions(&extents);
    desc.setDataType(dtype);
    desc.setUsage(usage);
    desc.setResourceOptions(MTLResourceOptions::StorageModeShared);
    Ok(desc)
}

fn validate_tensor_buffer_range(
    buffer_bytes: usize,
    byte_offset: usize,
    tensor_bytes: usize,
    alignment: usize,
) -> Result<(), String> {
    if alignment == 0 {
        return Err("MTLTensor reported zero buffer alignment".into());
    }
    if byte_offset % alignment != 0 {
        return Err(format!(
            "tensor buffer offset {byte_offset} not aligned to {alignment}"
        ));
    }
    if byte_offset
        .checked_add(tensor_bytes)
        .is_none_or(|end| end > buffer_bytes)
    {
        return Err(format!(
            "tensor buffer range offset={byte_offset} size={tensor_bytes} exceeds {buffer_bytes} bytes"
        ));
    }
    Ok(())
}

fn extents_from_dims(dims: &[usize]) -> Result<Retained<MTLTensorExtents>, String> {
    let values: Vec<NSInteger> = dims.iter().map(|&d| d as NSInteger).collect();
    let extents = unsafe {
        MTLTensorExtents::initWithRank_values(
            MTLTensorExtents::alloc(),
            values.len() as _,
            values.as_ptr(),
        )
    }
    .ok_or_else(|| "MTLTensorExtents::initWithRank_values failed".to_string())?;
    Ok(extents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dtype_int8_maps() {
        assert!(QuantDType::Int8.to_mtl().is_ok());
        let err = QuantDType::Int4.to_mtl().unwrap_err();
        assert!(err.contains("Int4"), "{err}");
        assert!(
            err.contains("unbound") || err.contains("not in objc2"),
            "{err}"
        );
        assert!(QuantDType::Fp8E8M0.to_mtl().is_err());
    }

    #[test]
    fn nax_verify_readiness_names_the_actual_q4_gap() {
        let r = nax_verify_readiness();
        assert!(r.int8_tensorops_dtype);
        assert!(!r.int4_tensorops_dtype);
        assert!(!r.fp8_e8m0_tensorops_dtype);
        assert!(!r.quant_prefill_gemm_wired);
        assert!(r.note.contains("shader-side sub-byte tensor constructor"));
        assert!(r.note.contains("host descriptor binding is separate"));
    }

    /// The argument table has 31 buffer slots, so 30 is the last valid index
    /// and 31 is already past the end. `setResource:atBufferIndex:` is not
    /// range-checked by Metal: before this wrapper validated, index 31 wrote
    /// past the table and `usize::MAX` took the process down with SIGSEGV.
    #[test]
    fn bind_mtl_tensor_rejects_out_of_range_index() {
        let rt = GpuRuntime::new().expect("runtime");
        let t = alloc_device_tensor(&rt, &[16, 16], QuantDType::Int8).expect("int8 tensor");
        let mut outcome = None;
        rt.with_binder(|bnd| {
            outcome = Some((
                bind_mtl_tensor(bnd, &t, 0),
                bind_mtl_tensor(bnd, &t, ARGUMENT_TABLE_MAX_BUFFERS - 1),
                bind_mtl_tensor(bnd, &t, ARGUMENT_TABLE_MAX_BUFFERS),
                bind_mtl_tensor(bnd, &t, usize::MAX),
            ));
            Ok(())
        })
        .expect("binder scope");
        let (first, last, past_end, huge) = outcome.expect("binder body ran");
        first.expect("index 0 is a valid slot");
        last.expect("the last slot must still bind");
        let err = past_end.expect_err("index 31 is past the end of a 31-slot table");
        assert!(err.contains("out of range"), "{err}");
        huge.expect_err("usize::MAX must never reach setResource:atBufferIndex:");
    }

    #[test]
    fn bind_mtl_tensor_rejects_foreign_runtime_before_mutation() {
        let owner = GpuRuntime::new().expect("owner runtime");
        let encoder = GpuRuntime::new().expect("encoder runtime");
        let foreign =
            alloc_device_tensor(&owner, &[16, 16], QuantDType::Int8).expect("foreign tensor");
        let local =
            alloc_device_tensor(&encoder, &[16, 16], QuantDType::Int8).expect("local tensor");

        let mut outcome = None;
        encoder
            .with_binder(|bnd| {
                let rejected = bind_mtl_tensor(bnd, &foreign, 0);
                // A rejected foreign resource must not poison or partially
                // mutate the binder. A valid local bind in the same slot and
                // scope must still succeed.
                let local_after_rejection = bind_mtl_tensor(bnd, &local, 0);
                outcome = Some((rejected, local_after_rejection));
                Ok(())
            })
            .expect("binder remains usable after a rejected foreign tensor");
        let (foreign_result, local_result) = outcome.expect("binder body ran");
        let err = foreign_result.expect_err("foreign runtime must be rejected");
        assert!(err.contains("another runtime"), "unexpected error: {err}");
        local_result.expect("local tensor must still bind");
    }

    #[test]
    fn device_tensor_residency_retires_after_completed_work() {
        let rt = GpuRuntime::new().expect("runtime");
        let baseline = *rt.metal4.residency_count.lock().unwrap();
        for _ in 0..32 {
            drop(alloc_device_tensor(&rt, &[16, 16], QuantDType::Int8).expect("device tensor"));
        }
        assert_eq!(
            *rt.metal4.residency_count.lock().unwrap(),
            baseline + 32,
            "final drop must retain residency until the completion drain"
        );

        rt.synchronize().expect("completed-work drain");
        assert_eq!(*rt.metal4.residency_count.lock().unwrap(), baseline);
    }

    /// Descriptor construction only (no device call). Full
    /// `tensorSizeAndAlign` / `newTensor` A/B is Phase 2 — objc2 bindings can
    /// SIGSEGV on some SDK/runtime combos when probing unsupported layouts.
    #[test]
    fn quant_descriptor_builds_for_int8() {
        let desc = make_descriptor(&[64, 64], MTLTensorDataType::Int8, MTLTensorUsage::Compute)
            .expect("descriptor");
        assert_eq!(desc.dataType(), MTLTensorDataType::Int8);
        assert_eq!(desc.dimensions().rank(), 2);
    }

    #[test]
    fn descriptor_rejects_extent_that_does_not_fit_nsinteger() {
        let err = make_descriptor(
            &[usize::MAX],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
        )
        .map(|_| ())
        .expect_err("usize::MAX would narrow to a negative NSInteger extent");
        assert!(err.contains("NSInteger"), "unexpected error: {err}");
    }

    #[test]
    fn tensor_buffer_range_checks_overflow_alignment_and_capacity() {
        validate_tensor_buffer_range(64, 16, 32, 16).expect("valid range");
        assert!(validate_tensor_buffer_range(64, 1, 32, 16).is_err());
        assert!(validate_tensor_buffer_range(64, 0, 65, 16).is_err());
        assert!(validate_tensor_buffer_range(usize::MAX, usize::MAX - 7, 16, 1).is_err());
        assert!(validate_tensor_buffer_range(64, 0, 1, 0).is_err());
    }
}
