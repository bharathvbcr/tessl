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
    MTLAllocation, MTLBuffer, MTLDevice, MTLResourceID, MTLResourceOptions, MTLSizeAndAlign, MTLTensor,
    MTLTensorDataType, MTLTensorDescriptor, MTLTensorExtents, MTLTensorUsage,
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
            QuantDType::Int4 => Err("MTLTensorDataType Int4 is not in objc2-metal 0.3 (WWDC26-330), so a \
                 host-created Int4 MTLTensor cannot be described. Note this gates only \
                 the host descriptor path: TensorOps itself accepts int4b_format, and \
                 kernels building tensors from device pointers are unaffected."
                .into()),
            QuantDType::Fp8E8M0 => {
                Err("FP8 E8M0 MTLTensor scale planes require newer Metal SDK (macOS 27+ notes)".into())
            }
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
/// Int4 is **unbound** in objc2-metal 0.3 — verify(M) remains on hand simdgroup Q4
/// GEMM (`gemm_q4_mlx_simd*`). DDTree stays parked until verify(M) flattens (it will
/// not with current simdgroup Q4 alone).
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
        note: "Int4 unbound in objc2-metal 0.3; verify(M) = hand simdgroup Q4; TensorOps Q4 not shipped",
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
/// A device-owned tensor is in its runtime's residency set from allocation.
/// Dropping the handle does not free the `MTLTensor` at once: Metal 4 command
/// buffers do not retain what they bind, so the runtime holds it until the
/// work already submitted has completed, and only then takes a device-owned
/// one out of the residency set.
pub struct GpuTensor {
    pub tensor: Retained<ProtocolObject<dyn MTLTensor>>,
    pub dtype: MTLTensorDataType,
    pub dims: Vec<usize>,
    /// A buffer-backed tensor's storage: the MTLTensor above borrows it, and
    /// its residency is the buffer's own. `None` for a device-owned tensor,
    /// which is registered for residency itself.
    pub(crate) storage: Option<GpuBuffer>,
    /// The runtime whose residency set and argument table this tensor is for.
    pub(crate) runtime: Arc<GpuRuntime>,
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        let alloc = ProtocolObject::<dyn MTLAllocation>::from_retained(self.tensor.clone());
        self.runtime.schedule_release(alloc, self.storage.is_none());
    }
}

impl GpuTensor {
    pub fn gpu_resource_id(&self) -> MTLResourceID {
        self.tensor.gpuResourceID()
    }

    pub fn data_type(&self) -> MTLTensorDataType {
        self.tensor.dataType()
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
///
/// A tensor from another runtime is refused (and fails the binder, so the
/// dispatch is not encoded even if the `Err` is ignored): it is outside this
/// runtime's residency set.
pub fn bind_mtl_tensor(bnd: &mut Binder<'_>, t: &GpuTensor, index: usize) -> Result<(), String> {
    if !std::ptr::eq(Arc::as_ptr(&t.runtime), bnd.runtime()) {
        let msg = "MTLTensor belongs to another runtime".to_string();
        bnd.fail(msg.clone());
        return Err(msg);
    }
    if index >= ARGUMENT_TABLE_MAX_BUFFERS {
        return Err(format!(
            "MTLTensor bind index {index} out of range: argument table has \
             {ARGUMENT_TABLE_MAX_BUFFERS} buffer slots (0..={})",
            ARGUMENT_TABLE_MAX_BUFFERS - 1
        ));
    }
    // SAFETY: `bind_resource_id` requires `index` to be within the argument
    // table's buffer bind count; the check above establishes that against the
    // same constant `runtime::try_init_metal4` builds the table with (the
    // DecodeIcb tape table is built with the same width). The resource id is
    // read from `t`, which owns its MTLTensor — and, for a buffer-backed
    // tensor, the storage behind it — for at least as long as this call.
    unsafe {
        bnd.bind_resource_id(t.gpu_resource_id(), index);
    }
    Ok(())
}

/// Size and alignment of a dense `dtype` tensor of shape `dims` laid out in a
/// buffer — what [`tensor_from_buffer`] needs the storage and offset to satisfy.
///
/// `dims` are in `MTLTensorExtents` order, innermost first: `dims[0]` is the
/// contiguous dimension (stride 1).
///
/// This used to fault on every call (EXC_BAD_ACCESS inside
/// `-[AGXG17XFamilyDevice tensorSizeAndAlignWithDescriptor:]`, M5 Pro, macOS
/// 27.0.1). The cause was not objc2 or an unsupported layout: the descriptor
/// left `strides` nil, and this selector sizes a tensor *in a buffer*, which
/// needs strides. The driver does not validate that — only `MTL_DEBUG_LAYER=1`
/// turns it into the assertion "strides should not be nil" — so it is the
/// descriptor layout, chosen in `make_descriptor`, that prevents it.
pub fn probe_tensor_support(rt: &GpuRuntime, dtype: QuantDType, dims: &[usize]) -> Result<MTLSizeAndAlign, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute, TensorLayout::DenseInBuffer)?;
    Ok(rt.device.tensorSizeAndAlignWithDescriptor(&desc))
}

/// Allocate a device-backed MTLTensor (no storage shared with [`GpuBuffer`]).
///
/// Metal picks the layout, so the tensor reports nil `strides`.
pub fn alloc_device_tensor(rt: &Arc<GpuRuntime>, dims: &[usize], dtype: QuantDType) -> Result<GpuTensor, String> {
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute, TensorLayout::DeviceOwned)?;
    let tensor = rt
        .device
        .newTensorWithDescriptor_error(&desc)
        .map_err(|e| format!("newTensorWithDescriptor: {e}"))?;
    // Metal 4 binds by resource ID and makes nothing resident on its own: a
    // kernel reading a tensor outside the residency set faults.
    rt.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*tensor));
    Ok(GpuTensor {
        tensor,
        dtype: mtl_dtype,
        dims: dims.to_vec(),
        storage: None,
        runtime: Arc::clone(rt),
    })
}

/// Wrap an existing shared buffer as a dense MTLTensor view starting at
/// `byte_offset` (offset must satisfy align).
///
/// `dims` are in `MTLTensorExtents` order, innermost first, and the view is
/// packed: strides `[1, dims[0], dims[0] * dims[1], …]` elements.
///
/// # Safety
/// No remaining caller obligation is known. What this section used to ask of
/// the caller — `byte_offset` aligned per `tensorSizeAndAlignWithDescriptor` —
/// is checked here, as are the window fitting inside `buf` and `buf` belonging
/// to `rt`; each failure is an `Err`. It stays an `unsafe fn` so that making it
/// safe is a deliberate API change, not a side effect of fixing its layout.
pub unsafe fn tensor_from_buffer(
    rt: &Arc<GpuRuntime>,
    buf: &GpuBuffer,
    byte_offset: usize,
    dims: &[usize],
    dtype: QuantDType,
) -> Result<GpuTensor, String> {
    // Same check as `Tensor::validate`: a buffer from another runtime is not in
    // this runtime's residency set, so binding it is a fault, not an error.
    if !buf.inner.runtime.ptr_eq(&Arc::downgrade(rt)) {
        return Err("tensor buffer belongs to a different runtime".into());
    }
    let mtl_dtype = dtype.to_mtl()?;
    let desc = make_descriptor(dims, mtl_dtype, MTLTensorUsage::Compute, TensorLayout::DenseInBuffer)?;
    let align = rt.device.tensorSizeAndAlignWithDescriptor(&desc);
    if align.align == 0 {
        return Err("tensorSizeAndAlignWithDescriptor reported zero alignment".into());
    }
    if byte_offset % align.align != 0 {
        return Err(format!(
            "tensor buffer offset {byte_offset} not aligned to {}",
            align.align
        ));
    }
    // Metal does not catch this one: on M5 Pro / macOS 27.0.1 it accepted a
    // window ending past `nbytes`. The pool rounds the MTLBuffer up to a power
    // of two (`BufferPool::bucket`), so it is longer than `nbytes`, and
    // whatever Metal checks does not protect the logical size.
    let end = byte_offset.checked_add(align.size);
    if end.is_none_or(|end| end > buf.nbytes()) {
        return Err(format!(
            "tensor view [{byte_offset}, +{}) runs past the end of a {}-byte buffer",
            align.size,
            buf.nbytes()
        ));
    }
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
    })
}

/// Which creation path a descriptor is for. Metal demands opposite things of
/// `strides` on the two, and neither failure is gentle, so the choice is
/// explicit at every call site rather than defaulted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorLayout {
    /// Metal allocates the storage and picks the layout
    /// (`-[MTLDevice newTensorWithDescriptor:error:]`). Strides stay nil: that
    /// selector rejects a descriptor that sets them (MTLTensorDomain code 2,
    /// "Strides should be nil when using newTensorWithDescriptor:error:").
    DeviceOwned,
    /// Dense layout over caller storage, for `tensorSizeAndAlignWithDescriptor:`
    /// and `-[MTLBuffer newTensorWithDescriptor:offset:error:]`. Strides are set:
    /// MTLTensor.h says to set them when creating tensors from a buffer, and
    /// the sizing selector faults in the driver when they are nil.
    ///
    /// Strides are in elements and follow MTLTensor.h's index order: index 0 is
    /// the innermost dimension and its stride is exactly 1, and each later
    /// stride is `strides[i - 1] * dims[i - 1]` — packed, no padding.
    DenseInBuffer,
}

/// `dims` are in `MTLTensorExtents` order — index 0 is the innermost
/// (contiguous) dimension — and are passed through unreordered.
fn make_descriptor(
    dims: &[usize],
    dtype: MTLTensorDataType,
    usage: MTLTensorUsage,
    layout: TensorLayout,
) -> Result<Retained<MTLTensorDescriptor>, String> {
    if dims.is_empty() || dims.len() > 16 {
        return Err(format!("MTLTensor rank {} out of range 1..=16", dims.len()));
    }
    let desc = MTLTensorDescriptor::new();
    let extents = extents_from_dims(dims)?;
    desc.setDimensions(&extents);
    match layout {
        TensorLayout::DeviceOwned => desc.setStrides(None),
        TensorLayout::DenseInBuffer => {
            let strides = extents_from_dims(&dense_strides(dims)?)?;
            desc.setStrides(Some(&strides));
        }
    }
    desc.setDataType(dtype);
    desc.setUsage(usage);
    desc.setResourceOptions(MTLResourceOptions::StorageModeShared);
    Ok(desc)
}

/// Packed strides, in elements, innermost first: `[1, d0, d0*d1, …]`.
fn dense_strides(dims: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for (i, &d) in dims.iter().enumerate() {
        strides.push(stride);
        if i + 1 < dims.len() {
            stride = stride
                .checked_mul(d)
                .ok_or_else(|| format!("MTLTensor stride overflows usize at dimension {}", i + 1))?;
        }
    }
    Ok(strides)
}

fn extents_from_dims(dims: &[usize]) -> Result<Retained<MTLTensorExtents>, String> {
    let values = dims
        .iter()
        .map(|&d| NSInteger::try_from(d).map_err(|_| format!("MTLTensor extent {d} exceeds NSInteger")))
        .collect::<Result<Vec<NSInteger>, String>>()?;
    let extents =
        unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), values.len() as _, values.as_ptr()) }
            .ok_or_else(|| "MTLTensorExtents::initWithRank_values failed".to_string())?;
    Ok(extents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    /// Env marker set on the isolated child process (see [`in_child`]).
    const CHILD_ENV: &str = "TESSL_MTL_TENSOR_CHILD";

    /// Run `body` in a child copy of this test binary, filtered to `test_name`.
    ///
    /// The calls under test go straight into the Metal driver, and a malformed
    /// descriptor there is not an `Err`: a stride-less descriptor handed to
    /// `tensorSizeAndAlignWithDescriptor:` faults inside the AGX driver
    /// (EXC_BAD_ACCESS). In-process, that SIGSEGV takes the whole harness down
    /// and every other test's result with it. In a child it is one failed test
    /// that names the signal.
    fn in_child(test_name: &str, body: impl FnOnce()) {
        if std::env::var_os(CHILD_ENV).is_some() {
            body();
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--test-threads=1", test_name])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn isolated child test");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(
            out.status.success(),
            "isolated child {test_name} failed ({}):\n{stdout}\n{stderr}",
            match out.status.signal() {
                Some(sig) => format!("killed by signal {sig}"),
                None => format!("exit status {:?}", out.status.code()),
            }
        );
        // A filter that matched nothing also exits 0 — make that loud.
        assert!(
            stdout.contains("1 passed"),
            "child ran no test (filter out of sync with {test_name}?):\n{stdout}"
        );
    }

    fn extent_values(e: &MTLTensorExtents) -> Vec<NSInteger> {
        // SAFETY: every index is below `rank()`, the bound Metal documents.
        (0..e.rank()).map(|i| unsafe { e.extentAtDimensionIndex(i) }).collect()
    }

    /// `tensorSizeAndAlignWithDescriptor:` sizes a tensor *in a buffer*, so it
    /// needs the buffer layout spelled out. Before `make_descriptor` took a
    /// [`TensorLayout`], this safe function faulted on every call.
    #[test]
    fn probe_tensor_support_sizes_int8() {
        in_child("mtl_tensor::tests::probe_tensor_support_sizes_int8", || {
            let rt = GpuRuntime::new().expect("runtime");
            let sa = probe_tensor_support(&rt, QuantDType::Int8, &[64, 64]).expect("int8 [64, 64] probe");
            assert!(
                sa.size >= 64 * 64,
                "int8 [64, 64] needs at least 4096 bytes, got {}",
                sa.size
            );
            assert!(sa.align > 0, "alignment must be non-zero");
        });
    }

    /// A buffer-backed tensor carries dense strides over the caller's storage,
    /// innermost (index 0) first, and a window the alignment rules forbid is an
    /// `Err` — not a fault.
    #[test]
    fn tensor_from_buffer_wraps_shared_storage() {
        in_child("mtl_tensor::tests::tensor_from_buffer_wraps_shared_storage", || {
            let rt = GpuRuntime::new().expect("runtime");
            let dims = [64usize, 32];
            let sa = probe_tensor_support(&rt, QuantDType::Int8, &dims).expect("probe");
            let align = sa.align;
            let buf = rt.alloc_buffer(sa.size + 2 * align).expect("buffer");

            // SAFETY: offset 0 and `align` are both multiples of the alignment
            // the device reported for this descriptor, and the buffer holds the
            // reported size past either offset.
            let t = unsafe { tensor_from_buffer(&rt, &buf, 0, &dims, QuantDType::Int8) }.expect("offset 0");
            let strides = t.tensor.strides().expect("buffer-backed tensor reports strides");
            assert_eq!(extent_values(&strides), [1, 64], "innermost stride 1, then dims[0]");
            assert_eq!(t.tensor.bufferOffset(), 0);

            let shifted =
                unsafe { tensor_from_buffer(&rt, &buf, align, &dims, QuantDType::Int8) }.expect("aligned offset");
            assert_eq!(shifted.tensor.bufferOffset(), align);

            // Aligned, but the tensor would run `align` bytes past the storage.
            let past_end = 3 * align;
            let err = unsafe { tensor_from_buffer(&rt, &buf, past_end, &dims, QuantDType::Int8) }
                .err()
                .expect("a view past the end of the buffer must be rejected");
            assert!(err.contains("past the end"), "{err}");

            let other = GpuRuntime::new().expect("second runtime");
            let err = unsafe { tensor_from_buffer(&other, &buf, 0, &dims, QuantDType::Int8) }
                .err()
                .expect("a buffer from another runtime must be rejected");
            assert!(err.contains("different runtime"), "{err}");

            if align > 1 {
                let err = unsafe { tensor_from_buffer(&rt, &buf, 1, &dims, QuantDType::Int8) }
                    .err()
                    .expect("misaligned offset must be rejected");
                assert!(err.contains("not aligned"), "{err}");
            }
        });
    }

    /// `newTensorWithDescriptor:error:` rejects a descriptor that sets strides
    /// (MTLTensorDomain code 2), so the device-owned layout must keep them nil.
    #[test]
    fn alloc_device_tensor_has_no_strides() {
        let rt = GpuRuntime::new().expect("runtime");
        let t = alloc_device_tensor(&rt, &[16, 16], QuantDType::Int8).expect("int8 tensor");
        assert!(
            t.tensor.strides().is_none(),
            "device-owned tensor must not carry strides"
        );
        assert_eq!(t.tensor.bufferOffset(), 0);

        // The other half of the contract: hand the device path the buffer
        // layout and Metal refuses it. If this ever starts succeeding, the
        // split in `TensorLayout` is no longer needed.
        let dense = make_descriptor(
            &[16, 16],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
            TensorLayout::DenseInBuffer,
        )
        .expect("buffer descriptor");
        let err = rt
            .device
            .newTensorWithDescriptor_error(&dense)
            .expect_err("newTensorWithDescriptor:error: must reject a strided descriptor");
        assert!(
            err.localizedDescription().to_string().contains("Strides should be nil"),
            "{err}"
        );
    }

    /// A kernel writes a device-owned tensor and another reads it back. Both
    /// dispatches are encoded and the handle dropped *before* the commit: the
    /// tensor must be resident from allocation and outlive the handle until
    /// the work completes, then leave the residency set. Isolated, because a
    /// tensor outside the set, or freed under the GPU, is a fault.
    #[test]
    fn a_kernel_reads_a_device_owned_tensor_dropped_before_commit() {
        in_child(
            "mtl_tensor::tests::a_kernel_reads_a_device_owned_tensor_dropped_before_commit",
            || {
                use crate::dispatch::{dispatch_2d, set_gpu_buf};
                use objc2_metal::MTLResidencySet;

                let rt = GpuRuntime::new().expect("runtime");
                let (cols, rows) = (40usize, 7usize);
                let t = alloc_device_tensor(&rt, &[cols, rows], QuantDType::Int8).expect("int8 tensor");
                let watch = t.tensor.clone();
                let alloc = || ProtocolObject::<dyn MTLAllocation>::from_ref(&*watch);
                assert!(
                    rt.metal4.residency.containsAllocation(alloc()),
                    "a device-owned tensor must be resident from allocation"
                );
                let out = rt.alloc_buffer(cols * rows * 4).expect("out");
                let fill = rt.pipeline("mtl_tensor_fill_i8").expect("fill pipeline");
                let read = rt.pipeline("mtl_tensor_read_i8").expect("read pipeline");

                rt.set_async_encode(true).expect("async encode");
                let mut binds = Vec::new();
                dispatch_2d(&rt, &fill, cols, rows, |bnd| binds.push(bind_mtl_tensor(bnd, &t, 0))).expect("fill");
                dispatch_2d(&rt, &read, cols, rows, |bnd| {
                    binds.push(bind_mtl_tensor(bnd, &t, 0));
                    set_gpu_buf(bnd, &out, 1);
                })
                .expect("read");
                drop(t);
                assert!(
                    rt.metal4.residency.containsAllocation(alloc()),
                    "a dropped tensor left residency while its work was still uncommitted"
                );
                rt.synchronize().expect("synchronize");
                rt.set_async_encode(false).expect("sync encode");
                for b in binds {
                    b.expect("bind");
                }

                let want: Vec<f32> = (0..cols * rows).map(|i| ((i % 127) as i32 - 63) as f32).collect();
                assert_eq!(out.read_f32()[..cols * rows], want[..]);
                assert!(
                    !rt.metal4.residency.containsAllocation(alloc()),
                    "a dropped tensor stayed resident after its work completed"
                );
            },
        );
    }

    /// A tensor allocated on one runtime is outside another's residency set:
    /// binding it there is refused, and the binder fails so nothing encodes.
    #[test]
    fn bind_mtl_tensor_refuses_a_foreign_runtime() {
        let rt = GpuRuntime::new().expect("runtime");
        let other = GpuRuntime::new().expect("second runtime");
        let t = alloc_device_tensor(&other, &[16, 16], QuantDType::Int8).expect("int8 tensor");
        let mut outcome = None;
        let scope = rt.with_binder(|bnd| {
            outcome = Some(bind_mtl_tensor(bnd, &t, 0));
            Ok(())
        });
        let err = outcome
            .expect("binder body ran")
            .expect_err("a foreign tensor must not bind");
        assert!(err.contains("another runtime"), "{err}");
        let err = scope.expect_err("the binder must fail with the bind");
        assert!(err.contains("another runtime"), "{err}");
    }

    #[test]
    fn quant_dtype_int8_maps() {
        assert!(QuantDType::Int8.to_mtl().is_ok());
        let err = QuantDType::Int4.to_mtl().unwrap_err();
        assert!(err.contains("Int4"), "{err}");
        assert!(err.contains("unbound") || err.contains("not in objc2"), "{err}");
        assert!(QuantDType::Fp8E8M0.to_mtl().is_err());
    }

    #[test]
    fn nax_verify_readiness_int4_unbound() {
        let r = nax_verify_readiness();
        assert!(r.int8_tensorops_dtype);
        assert!(!r.int4_tensorops_dtype);
        assert!(!r.fp8_e8m0_tensorops_dtype);
        assert!(!r.quant_prefill_gemm_wired);
        assert!(r.note.contains("Int4 unbound"));
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

    /// Descriptor construction only (no device call); the device paths are
    /// exercised by the child-process tests above. The two layouts must differ
    /// in exactly the property Metal checks: nil strides for device-owned
    /// tensors, dense strides for buffer-backed ones and sizing. Getting it
    /// backwards is a driver fault on one side and MTLTensorDomain code 2 on
    /// the other.
    #[test]
    fn quant_descriptor_builds_for_int8() {
        let owned = make_descriptor(
            &[64, 32],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
            TensorLayout::DeviceOwned,
        )
        .expect("device-owned descriptor");
        assert_eq!(owned.dataType(), MTLTensorDataType::Int8);
        assert_eq!(owned.dimensions().rank(), 2);
        assert!(
            owned.strides().is_none(),
            "device-owned descriptor must leave strides nil"
        );

        let dense = make_descriptor(
            &[64, 32],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
            TensorLayout::DenseInBuffer,
        )
        .expect("buffer descriptor");
        assert_eq!(extent_values(&dense.dimensions()), [64, 32]);
        assert_eq!(
            extent_values(&dense.strides().expect("buffer descriptor sets strides")),
            [1, 64]
        );
    }

    #[test]
    fn dense_strides_are_packed_innermost_first() {
        assert_eq!(dense_strides(&[7]).unwrap(), [1]);
        assert_eq!(dense_strides(&[64, 64]).unwrap(), [1, 64]);
        assert_eq!(dense_strides(&[3, 5, 7]).unwrap(), [1, 3, 15]);
        // The outermost extent never multiplies into a stride, so it cannot
        // overflow one; an inner extent that does is an error, not a wrap.
        assert_eq!(dense_strides(&[2, usize::MAX]).unwrap(), [1, 2]);
        let err = dense_strides(&[usize::MAX, 2, 2]).unwrap_err();
        assert!(err.contains("overflow"), "{err}");
    }

    /// An extent above `NSInteger::MAX` used to wrap negative through `as`.
    #[test]
    fn extents_reject_values_beyond_nsinteger() {
        let err = make_descriptor(
            &[usize::MAX],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
            TensorLayout::DeviceOwned,
        )
        .expect_err("extent past NSInteger::MAX");
        assert!(err.contains("exceeds NSInteger"), "{err}");
        // Every extent fits, but the outer stride is 2^31 * 2^32 = 2^63.
        let err = make_descriptor(
            &[1 << 31, 1 << 32, 2],
            MTLTensorDataType::Int8,
            MTLTensorUsage::Compute,
            TensorLayout::DenseInBuffer,
        )
        .expect_err("stride past NSInteger::MAX");
        assert!(err.contains("exceeds NSInteger"), "{err}");
    }
}
