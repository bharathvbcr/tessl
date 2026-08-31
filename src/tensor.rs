//! Thin GPU buffer + shape/dtype. No autodiff, no graph.
//!
//! Phase 4: optional byte offset for bank/slice views (no host round-trip),
//! and GPU blit copy for deep_copy (no host memcpy).
//!
//! Audit 4 P0: pooled buffers are Arc-owned; last drop schedules cold recycle +
//! `removeAllocation` after the in-flight CB completes (see [`GpuRuntime`]).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::sync::{Arc, Weak};

use crate::runtime::{BufferKind, GpuRuntime};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
}

impl DType {
    pub fn size_of(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
        }
    }
}

/// Checked storage size shared by allocation, views, and kernel validation.
pub(crate) fn checked_nbytes(shape: &[usize], dtype: DType) -> Result<usize, String> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| "tensor element count overflow".to_string())?;
    n.checked_mul(dtype.size_of())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| "tensor byte size overflow".to_string())
}

/// Shared Metal buffer with recycle / residency policy.
pub(crate) struct PooledBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) nbytes: usize,
    pub(crate) kind: BufferKind,
    pub(crate) runtime: Weak<GpuRuntime>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Bump views share the slab. Cold/Bump storage retires only after the
        // last owner drops and GPU work completes; Hot storage stays resident.
        if self.kind == BufferKind::Hot {
            return;
        }
        let Some(rt) = self.runtime.upgrade() else {
            return;
        };
        // Keep the MTLBuffer alive until after CB completion via pending queue.
        let buffer = self.buffer.clone();
        let nbytes = self.nbytes;
        rt.schedule_cold_recycle(buffer, nbytes);
    }
}

/// Owning handle to a shared-memory Metal buffer plus logical shape.
#[derive(Clone)]
pub struct GpuBuffer {
    pub(crate) inner: Arc<PooledBuffer>,
}

/// Exclusive mapped host view. GPU encoding/submission on this runtime is
/// rejected until the mapping drops. Mapping first waits for prior GPU work.
/// Use a short scope; an escaped `&mut` slice cannot outlive this guard.
pub struct HostMapping<'a, T> {
    buffer: &'a GpuBuffer,
    _runtime: Arc<GpuRuntime>,
    _access: crate::runtime::RuntimeAccess,
    _element: std::marker::PhantomData<T>,
}

impl<T> std::ops::Deref for HostMapping<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: only private map_host constructs this, checking type alignment
        // and length. The runtime lease excludes other host/GPU accesses.
        unsafe { std::slice::from_raw_parts(self.buffer.metal().contents().as_ptr().cast::<T>(),
            self.buffer.nbytes() / std::mem::size_of::<T>()) }
    }
}
impl<T> std::ops::DerefMut for HostMapping<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.metal().contents().as_ptr().cast::<T>(),
            self.buffer.nbytes() / std::mem::size_of::<T>()) }
    }
}

impl GpuBuffer {
    fn map_host<T>(&self) -> Result<HostMapping<'_, T>, String> {
        if self.nbytes() % std::mem::size_of::<T>() != 0 ||
            self.metal().contents().as_ptr() as usize % std::mem::align_of::<T>() != 0 {
            return Err("host mapping size/alignment mismatch".into());
        }
        let runtime = self.inner.runtime.upgrade()
            .ok_or_else(|| "host mapping runtime has been dropped".to_string())?;
        let access = runtime.host_access()?;
        Ok(HostMapping { buffer: self, _runtime: runtime, _access: access,
            _element: std::marker::PhantomData })
    }

    pub fn nbytes(&self) -> usize {
        self.inner.nbytes
    }

    pub fn metal(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.inner.buffer
    }

    /// Which residency pool this buffer came from.
    ///
    /// Callers choose a [`BufferKind`] at allocation time; this reads it back,
    /// which matters when a buffer is handed around and the recycling
    /// behaviour on drop (Cold recycles, Hot stays resident, Bump does not)
    /// affects what the holder may do with it.
    pub fn kind(&self) -> BufferKind {
        self.inner.kind
    }

    /// Host pointer into unified memory (StorageModeShared).
    pub fn try_contents_f32(&self) -> Result<HostMapping<'_, f32>, String> {
        self.map_host::<f32>()
    }

    pub fn contents_f32(&self) -> HostMapping<'_, f32> {
        self.try_contents_f32().expect("exclusive host mapping failed")
    }

    pub fn try_contents_u16(&self) -> Result<HostMapping<'_, u16>, String> {
        self.map_host::<u16>()
    }

    pub fn contents_u16(&self) -> HostMapping<'_, u16> {
        self.try_contents_u16().expect("exclusive host mapping failed")
    }

    pub fn write_f32(&self, data: &[f32]) {
        let mut dst = self.contents_f32();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    /// Write into the leading `data.len()` floats (buffer may be oversized scratch).
    pub fn write_f32_prefix(&self, data: &[f32]) {
        let mut dst = self.contents_f32();
        assert!(
            data.len() <= dst.len(),
            "write_f32_prefix: data {} > buf {}",
            data.len(),
            dst.len()
        );
        dst[..data.len()].copy_from_slice(data);
    }

    pub fn read_f32(&self) -> Vec<f32> {
        self.contents_f32().to_vec()
    }

    pub fn write_bf16_bits(&self, data: &[u16]) {
        let mut dst = self.contents_u16();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    pub fn try_contents_u8(&self) -> Result<HostMapping<'_, u8>, String> {
        self.map_host::<u8>()
    }

    pub fn contents_u8(&self) -> HostMapping<'_, u8> {
        self.try_contents_u8().expect("exclusive host mapping failed")
    }

    pub fn write_bytes(&self, data: &[u8]) {
        let mut dst = self.contents_u8();
        assert!(data.len() <= dst.len());
        dst[..data.len()].copy_from_slice(data);
    }

    pub fn try_contents_u32(&self) -> Result<HostMapping<'_, u32>, String> {
        self.map_host::<u32>()
    }

    pub fn contents_u32(&self) -> HostMapping<'_, u32> {
        self.try_contents_u32().expect("exclusive host mapping failed")
    }

    pub fn write_u32(&self, data: &[u32]) {
        let mut dst = self.contents_u32();
        assert_eq!(dst.len(), data.len());
        dst.copy_from_slice(data);
    }

    pub fn read_u32(&self) -> Vec<u32> {
        self.contents_u32().to_vec()
    }

    pub fn zero(&self) {
        self.map_host::<u8>().expect("exclusive host zero failed").fill(0);
    }

    /// # Safety
    /// Storage must be fresh or retired after GPU completion, with no live views.
    pub(crate) unsafe fn zero_unsubmitted(&self) {
        unsafe { std::ptr::write_bytes(self.metal().contents().as_ptr().cast::<u8>(), 0, self.nbytes()) };
    }
}

/// Logical tensor: shape + dtype over a GpuBuffer (row-major, contiguous view).
#[derive(Clone)]
pub struct Tensor {
    pub buffer: GpuBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Byte offset into `buffer` for bank / slice views.
    pub byte_offset: usize,
    pub(crate) runtime: Arc<GpuRuntime>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes_logical(&self) -> usize {
        self.numel() * self.dtype.size_of()
    }

    pub fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.runtime
    }

    /// Build a tensor over an existing buffer at `byte_offset`.
    ///
    /// The `runtime` field is private, so this is the only way to construct a
    /// `Tensor` from outside the crate. That is deliberate: the struct-literal
    /// form it replaces performed no checks at all, so a caller could describe a
    /// region larger than its buffer, or pair a buffer with a runtime that did
    /// not allocate it. This runs the same `validate()` every dispatch relies on
    /// before handing one back.
    pub fn from_buffer(
        runtime: &Arc<GpuRuntime>,
        buffer: GpuBuffer,
        shape: &[usize],
        dtype: DType,
        byte_offset: usize,
    ) -> Result<Tensor, String> {
        let t = Tensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
            byte_offset,
            runtime: Arc::clone(runtime),
        };
        t.validate()?;
        Ok(t)
    }

    /// View into the same buffer at an element offset (same dtype).
    pub fn view(&self, shape: &[usize], elem_offset: usize) -> Tensor {
        self.validate().expect("invalid source view");
        let nbytes = checked_nbytes(shape, self.dtype).expect("view size overflow");
        let off = elem_offset
            .checked_mul(self.dtype.size_of())
            .and_then(|off| self.byte_offset.checked_add(off))
            .expect("view offset overflow");
        assert!(
            off.checked_add(nbytes)
                .is_some_and(|end| end <= self.buffer.nbytes()),
            "view out of bounds"
        );
        Tensor {
            buffer: self.buffer.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            byte_offset: off,
            runtime: Arc::clone(&self.runtime),
        }
    }

    /// Validate public metadata before passing a view to a GPU kernel.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let bytes = checked_nbytes(&self.shape, self.dtype)?;
        if self.byte_offset % self.dtype.size_of() != 0
            || self
                .byte_offset
                .checked_add(bytes).is_none_or(|end| end > self.buffer.nbytes())
        {
            return Err("tensor view is misaligned or out of bounds".into());
        }
        if !self
            .buffer
            .inner
            .runtime
            .ptr_eq(&Arc::downgrade(&self.runtime))
        {
            return Err("tensor buffer belongs to a different runtime".into());
        }
        Ok(())
    }

    pub(crate) fn overlaps(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.buffer.inner, &other.buffer.inner)
            && self.byte_offset < other.byte_offset + other.nbytes_logical()
            && other.byte_offset < self.byte_offset + self.nbytes_logical()
    }

    /// Allocate a new buffer and GPU-copy contents (encoded into the active batch).
    pub fn deep_copy(&self) -> Result<Tensor, String> {
        self.validate()?;
        let t = match self.dtype {
            DType::F32 => self.runtime.alloc_tensor_f32(&self.shape)?,
            DType::BF16 => self.runtime.alloc_tensor_bf16(&self.shape)?,
        };
        gpu_copy(self, &t)?;
        Ok(t)
    }
}

/// Device blit: `dst = src` (same numel/dtype). Encodes into the active batch.
pub fn gpu_copy(src: &Tensor, dst: &Tensor) -> Result<(), String> {
    src.validate()?;
    dst.validate()?;
    if src.numel() != dst.numel() || src.dtype != dst.dtype ||
        !Arc::ptr_eq(src.runtime(), dst.runtime()) {
        return Err("copy requires equal element counts/dtypes and the same runtime".into());
    }
    if src.numel() > u32::MAX as usize {
        return Err("copy exceeds 32-bit kernel indexing".into());
    }
    if Arc::ptr_eq(&src.buffer.inner, &dst.buffer.inner) && src.byte_offset == dst.byte_offset {
        return Ok(());
    }
    if src.overlaps(dst) { return Err("copy source and destination overlap".into()); }
    let rt = src.runtime();
    let n = src.numel();
    let kernel = match src.dtype {
        DType::F32 => "copy_f32",
        DType::BF16 => "copy_bf16",
    };
    let p = rt.pipeline(kernel)?;
    crate::dispatch::dispatch_1d(rt, &p, n, |bnd| {
        crate::dispatch::set_tensor(bnd, src, 0);
        crate::dispatch::set_tensor(bnd, dst, 1);
        crate::dispatch::set_u32(bnd, n as u32, 2);
    })
}

/// Host f32 → bf16 bit pattern (round-to-nearest-even truncate).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    // Preserve NaNs even when all payload bits would be truncated. Quiet them
    // before rounding; negative payloads can otherwise overflow the u32 sum.
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    round as u16
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_slice_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_bf16_bits).collect()
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn bf16_preserves_special_values_and_ties() {
        for bits in [0x7f800001, 0x7fffffff, 0xff800001, 0xffffffff] {
            assert!(
                bf16_bits_to_f32(f32_to_bf16_bits(f32::from_bits(bits))).is_nan(),
                "NaN payload {bits:x} became finite or infinity"
            );
        }
        for bits in [0, 0x80000000, 0x7f800000, 0xff800000] {
            assert_eq!(f32_to_bf16_bits(f32::from_bits(bits)), (bits >> 16) as u16);
        }
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f818000)), 0x3f82);
        for b in 0..=u16::MAX {
            let x = bf16_bits_to_f32(b);
            if !x.is_nan() {
                assert_eq!(f32_to_bf16_bits(x), b);
            }
        }
    }

    #[test]
    fn allocation_rejects_overflow() {
        let rt = GpuRuntime::new().unwrap();
        assert!(rt.alloc_tensor_f32(&[usize::MAX, usize::MAX]).is_err());
        assert!(rt.alloc_tensor_bf16(&[usize::MAX, usize::MAX]).is_err());
        assert!(rt.alloc_tensor_f32_hot(&[usize::MAX / 4 + 2]).is_err());
        assert!(rt.alloc_temp_f32(&[usize::MAX, usize::MAX]).is_err());
    }

    #[test]
    #[should_panic(expected = "view")]
    fn view_rejects_offset_overflow() {
        let rt = GpuRuntime::new().unwrap();
        let t = rt.alloc_tensor_f32(&[4]).unwrap();
        let _ = t.view(&[1], usize::MAX / 4 + 1);
    }
}
#[cfg(test)]
mod audit_tests {
    use super::*;

    #[test]
    fn stress_mapping_reentry_and_queued_copies() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let a = rt.alloc_tensor_f32(&[257]).unwrap();
        let b = rt.alloc_tensor_f32(&[257]).unwrap();
        let values: Vec<f32> = (0..257).map(|i| i as f32 / 16.0).collect();
        a.buffer.write_f32(&values);
        for _ in 0..128 {
            gpu_copy(&a, &b).unwrap();
            gpu_copy(&b, &a).unwrap();
        }
        // Mapping itself must complete all queued work, without explicit sync.
        assert_eq!(b.buffer.read_f32(), values);
        let alias = b.buffer.clone();
        {
            let mut map = b.buffer.try_contents_f32().unwrap();
            assert!(alias.try_contents_u8().is_err());
            assert!(rt.commit(false).is_err());
            assert!(rt.synchronize().is_err());
            map[0] = 42.0;
        }
        assert_eq!(alias.contents_f32()[0], 42.0);
        rt.with_binder(|_| {
            assert!(rt.with_binder(|_| Ok(())).is_err());
            Ok(())
        }).unwrap();
        rt.synchronize().unwrap();
    }

    #[test]
    fn stress_retained_bump_views_and_pool_pressure() {
        let rt = GpuRuntime::new().unwrap();
        rt.ensure_bump(256).unwrap();
        let mut retained = Vec::new();
        for i in 0..128 {
            let view = rt.bump_alloc_f32(&[64]).unwrap();
            view.buffer.write_f32(&[i as f32; 64]);
            retained.push(view);
            rt.bump_reset();
            let pressure = rt.alloc_tensor_f32(&[64]).unwrap();
            pressure.buffer.write_f32(&[-999.0; 64]);
            drop(pressure);
        }
        assert!(rt.ensure_bump(usize::MAX).is_err());
        for (i, view) in retained.iter().enumerate() {
            assert!(view.buffer.read_f32().iter().all(|&x| x == i as f32));
        }
        drop(retained);
        rt.synchronize().unwrap();
        rt.bump_reset();
        assert!(rt.bump_alloc_f32(&[64]).is_ok());
    }

    #[test]
    fn bump_allocation_rejects_live_host_mapping() {
        let rt = GpuRuntime::new().unwrap();
        rt.ensure_bump(512).unwrap();
        let first = rt.bump_alloc_f32(&[4]).unwrap();
        let map = first.buffer.contents_f32();
        assert!(rt.bump_alloc_f32(&[4]).is_err(), "host mapping aliases slab initialization");
        assert_eq!(map[0], 0.0);
    }

    #[test]
    fn host_mapping_excludes_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let t = rt.alloc_tensor_f32(&[4]).unwrap();
        let map = t.buffer.contents_f32();
        assert!(rt.with_binder(|_| Ok(())).is_err(), "encoding accepted during host mapping");
        assert_eq!(map[0],0.0);
    }

    #[test]
    fn copy_rejects_overlap_before_encoding() {
        let rt = GpuRuntime::new().unwrap();
        let t = rt.alloc_tensor_f32(&[16]).unwrap();
        assert!(gpu_copy(&t.view(&[8],0),&t.view(&[8],4)).is_err());
        assert_eq!(rt.take_dispatch_count(),0);
    }

    #[test]
    fn copy_rejects_malformed_metadata_without_panicking() {
        let rt = GpuRuntime::new().unwrap();
        let a = rt.alloc_tensor_f32(&[4]).unwrap();
        let b = rt.alloc_tensor_f32(&[3]).unwrap();
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu_copy(&a,&b)));
        assert!(result.is_ok(), "Result API panicked");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn bump_growth_preserves_live_views() {
        let rt = GpuRuntime::new().unwrap();
        rt.ensure_bump(256).unwrap();
        let view = rt.bump_alloc_f32(&[64]).unwrap();
        view.buffer.write_f32(&[7.0;64]);
        rt.ensure_bump(512).unwrap();
        let new = rt.alloc_tensor_f32(&[64]).unwrap();
        new.buffer.write_f32(&[3.0;64]);
        assert!(view.buffer.read_f32().iter().all(|&x|x==7.0), "live bump allocation recycled");
    }

    #[test]
    fn bump_reset_preserves_live_views() {
        let rt = GpuRuntime::new().unwrap();
        rt.ensure_bump(256).unwrap();
        let a=rt.bump_alloc_f32(&[64]).unwrap();
        a.buffer.write_f32(&[7.0;64]);
        // Reset must either reject outstanding views, or move to a fresh slab.
        let reset=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.bump_reset()));
        if reset.is_ok() {
            let b=rt.bump_alloc_f32(&[64]).unwrap();
            b.buffer.write_f32(&[3.0;64]);
            assert!(a.buffer.read_f32().iter().all(|&x|x==7.0));
        }
    }
}
