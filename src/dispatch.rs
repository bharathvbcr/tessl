//! Metal 4 bind + dispatch helpers.
//!
//! [`Binder`] targets the Metal 4 argument table + const arena. Call sites use
//! `set_*` / `bind_*` sugar; [`GpuRuntime::with_binder`] opens the command
//! buffer encoder.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLBuffer, MTLComputePipelineState, MTLIndirectCommandBuffer, MTLResourceID, MTLSize,
    MTLStages,
};

use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{GpuBuffer, Tensor};

/// Metal 4 compute binder (argument table + const arena).
pub struct Binder<'a> {
    runtime: &'a GpuRuntime,
    error: Option<String>,
    max_buffers: usize,
    max_threads: Option<usize>,
    enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
    const_staging: &'a ProtocolObject<dyn MTLBuffer>,
    const_cursor: &'a mut usize,
    /// Last pipeline set (Retained) for DecodeIcb capture.
    last_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// A2: latch `setArgumentTable` once per binder scope (table is persistent).
    arg_table_latched: bool,
    /// Pointer identity of the last adopted / latched argument table (skip redundant
    /// `setArgumentTable` when the same table is reused across tape cmds).
    last_arg_table_ptr: Option<usize>,
    /// Auto-barrier mode latched for this binder scope. Reading the global once
    /// at construction keeps every dispatch and explicit-barrier decision in one
    /// scope consistent even if another thread changes the flag mid-encode.
    skip_auto_barriers: bool,
}

impl<'a> Binder<'a> {
    pub(crate) fn finish(&self) -> Result<(), String> {
        match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(message.into());
        }
    }
    fn valid_index(&mut self, index: usize) -> bool {
        if index >= self.max_buffers {
            self.fail("argument-table buffer index out of range");
        }
        self.error.is_none()
    }
    fn write_constants(&mut self, bytes: &[u8]) -> u64 {
        if self.error.is_some() {
            return 0;
        }
        let start = self.const_cursor.checked_add(15).map(|n| n & !15);
        let end = start.and_then(|n| n.checked_add(bytes.len().max(4)));
        let (Some(start), Some(end)) = (start, end) else {
            self.fail("constant arena offset overflow");
            return 0;
        };
        if bytes.is_empty() || end > self.const_staging.length() {
            self.fail("constant arena exhausted or empty payload");
            return 0;
        }
        // SAFETY: checked the entire destination range before writing.
        unsafe {
            let dst = self
                .const_staging
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(start);
            std::ptr::write_bytes(dst, 0, bytes.len().max(4));
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        *self.const_cursor = end;
        self.const_staging.gpuAddress() + start as u64
    }

    pub(crate) fn new(
        enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
        const_staging: &'a ProtocolObject<dyn MTLBuffer>,
        const_cursor: &'a mut usize,
        skip_auto_barriers: bool,
        max_buffers: usize,
        runtime: &'a GpuRuntime,
    ) -> Self {
        Self {
            runtime,
            error: None,
            max_buffers,
            max_threads: None,
            enc,
            table,
            const_staging,
            const_cursor,
            last_pipeline: None,
            arg_table_latched: false,
            last_arg_table_ptr: None,
            skip_auto_barriers,
        }
    }

    /// True when this scope skips the per-dispatch auto barrier, so packed
    /// multi-dispatch ops must insert [`Self::barrier`] at their RAW edges.
    /// Call sites must use this rather than re-reading the global flag — the
    /// two can disagree while another thread toggles the flag.
    #[inline]
    pub fn needs_explicit_barriers(&self) -> bool {
        self.skip_auto_barriers
    }

    /// Latch the persistent argument table onto the encoder (idempotent).
    ///
    /// No-op when any table is already latched (including a prebuilt table
    /// adopted via [`Self::adopt_argument_table`]) — do not overwrite.
    #[inline]
    pub fn latch_argument_table(&mut self) {
        if self.arg_table_latched {
            return;
        }
        self.enc.setArgumentTable(Some(self.table));
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(self.table as *const _ as usize);
    }

    /// Switch the encoder to a prebuilt argument table (DecodeIcb tape path).
    ///
    /// Marks the binder as latched so [`Self::dispatch`] will not overwrite with
    /// the runtime's persistent table. Returns `true` when a Metal
    /// `setArgumentTable` call was issued; `false` when the encoder already
    /// held this same table (pointer identity — A2 v0.5.7 sticky adopt).
    #[inline]
    pub fn adopt_argument_table(&mut self, table: &ProtocolObject<dyn MTL4ArgumentTable>) -> bool {
        let ptr = table as *const _ as usize;
        if self.arg_table_latched && self.last_arg_table_ptr == Some(ptr) {
            return false;
        }
        self.enc.setArgumentTable(Some(table));
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(ptr);
        true
    }

    /// Copy bytes into the const arena; return the GPU address.
    ///
    /// Does **not** call `setAddress` or capture — used when writing into a
    /// prebuilt per-command argument table (Immediate residual only).
    pub fn materialize_bytes(&mut self, bytes: &[u8]) -> u64 {
        self.write_constants(bytes)
    }

    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        if self.error.is_some() {
            return;
        }
        self.max_threads = Some(pipeline.maxTotalThreadsPerThreadgroup());
        self.enc.setComputePipelineState(pipeline);
        if crate::decode_icb::decode_icb_capture_active() {
            // Retain via pipeline cache clone path: callers pass cache Retained refs.
            // We re-lookup is unavailable here — store a raw retain if possible.
            // SAFETY: pipeline is a live Objective-C object retained by the caller
            // for the duration of with_binder; we retain an extra ref for the tape.
            let retained = unsafe {
                Retained::retain(pipeline as *const _ as *mut _).expect("retain pipeline")
            };
            self.last_pipeline = Some(retained);
            if let Some(ref p) = self.last_pipeline {
                crate::decode_icb::capture_note_pipeline(p.clone());
            }
        }
    }

    pub fn bind_buf(&mut self, buf: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize) {
        if !self.valid_index(index) {
            return;
        }
        if offset >= buf.length() {
            self.fail("buffer binding offset out of bounds");
            return;
        }
        let Some(addr) = buf.gpuAddress().checked_add(offset as u64) else {
            self.fail("GPU address overflow");
            return;
        };
        // A raw `MTLBuffer` carries no owning `GpuBuffer`, so there is nothing
        // for the capture tape to record or to pin. Mark the tape incomplete
        // instead of letting it silently omit the operand.
        crate::decode_icb::capture_note_unrecordable_bind();
        self.bind_addr(addr, index);
    }

    /// Bind a precomputed GPU address (DecodeIcb tape replay bind-tax cut).
    #[inline]
    pub fn bind_addr(&mut self, gpu_addr: u64, index: usize) {
        if !self.valid_index(index) {
            return;
        }
        if gpu_addr == 0 {
            self.fail("null GPU address");
            return;
        }
        unsafe {
            self.table.setAddress_atIndex(gpu_addr, index);
        }
    }

    pub fn bind_tensor(&mut self, t: &Tensor, index: usize) {
        if let Err(e) = t.validate() {
            self.fail(e);
            return;
        }
        if !std::ptr::eq(t.runtime().as_ref(), self.runtime) {
            self.fail("tensor belongs to another runtime");
            return;
        }
        self.bind_buf(t.buffer.metal(), t.byte_offset, index);
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_bind(index, &t.buffer, t.byte_offset);
        }
    }

    pub fn bind_gpu_buf(&mut self, b: &GpuBuffer, index: usize) {
        if !std::ptr::eq(b.inner.runtime.as_ptr(), self.runtime) {
            self.fail("buffer belongs to another runtime");
            return;
        }
        self.bind_buf(b.metal(), 0, index);
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_bind(index, b, 0);
        }
    }

    /// Bind an `MTLResourceID` (e.g. a `GpuTensor` from the `quant-prep`
    /// feature's `mtl_tensor` module) at a buffer index.
    ///
    /// # Safety
    /// `index` must be within the argument table's buffer bind count.
    /// # Safety
    ///
    /// `resource_id` must name a resource that stays alive and resident for the
    /// duration of the encode. That is the caller's to guarantee and cannot be
    /// checked here, which is why this stays `unsafe`.
    ///
    /// The range of `index` is *not* the caller's problem any more: it is
    /// checked below. It used to be part of this contract while
    /// `Binder::max_buffers` was private, so an out-of-crate caller had no way
    /// to satisfy it — and an out-of-range index reached
    /// `setResource:atBufferIndex:` on a 31-slot table. Probed directly: index
    /// 31 passed through silently, `usize::MAX` took the process down with
    /// SIGSEGV. Checking here fixes the class at the one place every caller
    /// goes through.
    pub unsafe fn bind_resource_id(&mut self, resource_id: MTLResourceID, index: usize) {
        if !self.valid_index(index) {
            return;
        }
        // As `bind_buf`: a bare `MTLResourceID` carries no owning handle, so a
        // capture that contains one cannot be replayed faithfully.
        crate::decode_icb::capture_note_unrecordable_bind();
        unsafe {
            self.table
                .setResource_atBufferIndex(resource_id, index as _);
        }
    }

    /// Bind raw bytes into the const arena; returns the GPU address written.
    pub fn bind_bytes(&mut self, bytes: &[u8], index: usize) -> u64 {
        if !self.valid_index(index) {
            return 0;
        }
        let addr = self.write_constants(bytes);
        if self.error.is_some() {
            return 0;
        }
        self.bind_addr(addr, index);
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_immediate(index, bytes);
        }
        addr
    }

    pub fn bind_u32(&mut self, v: u32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    pub fn bind_f32(&mut self, v: f32, index: usize) {
        self.bind_bytes(&v.to_ne_bytes(), index);
    }

    /// Dynamic threadgroup memory (`threadgroup T *ptr [[threadgroup(index)]]`).
    pub fn set_threadgroup_memory(&mut self, index: usize, length: usize) {
        // Same slot space as the buffer binds, and previously unchecked while
        // every `bind_*` validated. An out-of-range index reached Metal
        // directly.
        if !self.valid_index(index) {
            return;
        }
        // SAFETY: `index` is within the argument table's bind count, checked
        // immediately above; `length` is a byte count Metal validates against
        // the device's threadgroup memory limit; and `self.enc` is a live
        // encoder for the duration of the borrow.
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex(length as _, index as _);
        }
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_tg_mem(index, length);
        }
    }

    /// Dispatch threadgroups. Optionally inserts a Dispatch→Dispatch Device
    /// barrier after the dispatch (default on; skip via
    /// `METAL_RUNTIME_HAZARD_BARRIERS=1`). Packed multi-dispatch ops that need
    /// RAW/WAR still call [`Self::barrier`] explicitly.
    pub fn dispatch(&mut self, threadgroups: MTLSize, threads_per_tg: MTLSize) {
        if self.error.is_some() {
            return;
        }
        let lanes = threads_per_tg
            .width
            .checked_mul(threads_per_tg.height)
            .and_then(|n| n.checked_mul(threads_per_tg.depth));
        let valid = lanes
            .zip(self.max_threads)
            .is_some_and(|(n, max)| n > 0 && n <= max)
            && [threadgroups.width, threadgroups.height, threadgroups.depth]
                .iter()
                .all(|&n| n > 0 && n <= u32::MAX as usize);
        if !valid {
            self.fail("invalid dispatch geometry or missing pipeline");
            return;
        }

        self.latch_argument_table();
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        crate::infer_trace::on_dispatch();
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_dispatch(threadgroups, threads_per_tg);
        }
        if !self.skip_auto_barriers {
            self.enc
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            crate::infer_trace::on_barrier();
            // Freeze always-on auto-barrier into the DecodeIcb tape.
            if crate::decode_icb::decode_icb_capture_active() {
                crate::decode_icb::capture_note_barrier();
            }
        }
    }

    /// Explicit producer→consumer barrier inside a packed encoder
    /// (Dispatch→Dispatch Device).
    pub fn barrier(&mut self) {
        if self.error.is_some() {
            return;
        }
        self.enc
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        crate::infer_trace::on_barrier();
        // Shipping hazard skip-auto: RAW edges land here — capture for tape replay.
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_barrier();
        }
    }

    /// Execute a pre-encoded compute [`MTLIndirectCommandBuffer`] range.
    ///
    /// When `inherit_arg_table` is true, latches the current MTL4 argument table
    /// so `inheritBuffers=true` ICB cmds see host binds. Freeze-binds
    /// (`inheritBuffers=false` + classic `setKernelBuffer`) passes false — no
    /// `setArgumentTable` traffic.
    pub fn execute_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        self.execute_icb_ex(icb, start, count, true);
    }

    /// Like [`Self::execute_icb`] with explicit inherit-table control.
    pub fn execute_icb_ex(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
        inherit_arg_table: bool,
    ) {
        // Do not encode over a poisoned binder: a prior failure means the
        // argument table is not in the state this range assumes.
        if self.error.is_some() {
            return;
        }
        // A range past the end of the ICB is caller data, not a Metal detail.
        // `executeCommandsInBuffer` with an out-of-range range is undefined,
        // and `start`/`count` reach here straight from the caller.
        let icb_len = icb.size();
        match start.checked_add(count) {
            Some(end) if end <= icb_len as u64 => {}
            _ => {
                self.fail("ICB execute range out of bounds");
                return;
            }
        }
        let range = NSRange {
            location: start as _,
            length: count as _,
        };
        if inherit_arg_table {
            // Latch so ICB `inheritBuffers=true` sees MTL4 binds.
            self.latch_argument_table();
        }
        // SAFETY: `range` is within `icb`'s command count, checked above; `icb`
        // outlives this call through the borrow; and the argument table has
        // been latched when the commands inherit it.
        unsafe {
            self.enc.executeCommandsInBuffer_withRange(icb, range);
        }
        crate::infer_trace::on_dispatch();
        if !self.skip_auto_barriers {
            self.enc
                .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            crate::infer_trace::on_barrier();
            if crate::decode_icb::decode_icb_capture_active() {
                crate::decode_icb::capture_note_barrier();
            }
        }
    }

    /// Optimize an ICB range after CPU-side encode (recommended once before reuse).
    pub fn optimize_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        let range = NSRange {
            location: start as _,
            length: count as _,
        };
        unsafe {
            self.enc.optimizeIndirectCommandBuffer_withRange(icb, range);
        }
    }
}

// --- Free helpers (call-site sugar) -----------------------------------------

pub fn set_tensor(bnd: &mut Binder<'_>, t: &Tensor, index: usize) {
    bnd.bind_tensor(t, index);
}

pub fn set_gpu_buf(bnd: &mut Binder<'_>, buf: &GpuBuffer, index: usize) {
    bnd.bind_gpu_buf(buf, index);
}

/// Bind `buf` at a byte offset (slice / slot views without host round-trip).
pub fn set_gpu_buf_offset(bnd: &mut Binder<'_>, buf: &GpuBuffer, byte_offset: usize, index: usize) {
    bnd.bind_buf(buf.metal(), byte_offset, index);
    if crate::decode_icb::decode_icb_capture_active() {
        crate::decode_icb::capture_note_bind(index, buf, byte_offset);
    }
}

pub fn set_u32(bnd: &mut Binder<'_>, v: u32, index: usize) {
    bnd.bind_u32(v, index);
}

pub fn set_f32(bnd: &mut Binder<'_>, v: f32, index: usize) {
    bnd.bind_f32(v, index);
}

/// Dispatch `n` threads with automatic threadgroup sizing.
///
/// Callers bind `n as u32` for the kernels' `uint` element counts, so a count
/// past `u32::MAX` would silently wrap to a partial pass — reject it here at
/// the one seam every 1D op flows through.
pub fn dispatch_1d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if n > u32::MAX as usize {
        return Err("1D dispatch exceeds uint indexing".into());
    }
    let width = pipeline.threadExecutionWidth();
    let tpt = width.min(n).max(1);
    let groups = n.div_ceil(tpt);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// 1D-over-x grid with `ny` rows of threadgroups.
pub fn dispatch_2d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    nx: usize,
    ny: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if nx == 0 || ny == 0 {
        return Ok(());
    }
    if [nx, ny].iter().any(|&n| n > u32::MAX as usize) {
        return Err("dispatch extent exceeds uint indexing".into());
    }
    let width = pipeline.threadExecutionWidth();
    let tx = width.min(nx).max(1);
    let groups_x = nx.div_ceil(tx);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups_x, ny, 1), mtl_size(tx, 1, 1));
        Ok(())
    })
}

/// 1D-over-x grid with `ny` x `nz` planes of threadgroups.
pub fn dispatch_3d(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    nx: usize,
    ny: usize,
    nz: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Ok(());
    }
    if [nx, ny, nz].iter().any(|&n| n > u32::MAX as usize) {
        return Err("dispatch extent exceeds uint indexing".into());
    }
    let width = pipeline.threadExecutionWidth();
    let tx = width.min(nx).max(1);
    let groups_x = nx.div_ceil(tx);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(mtl_size(groups_x, ny, nz), mtl_size(tx, 1, 1));
        Ok(())
    })
}

/// 2D grid of threadgroups with fixed threads-per-threadgroup (FA-2 tiles).
pub fn dispatch_2d_tg(
    rt: &GpuRuntime,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    groups_x: usize,
    groups_y: usize,
    threads_per_tg: usize,
    encode_bufs: impl FnOnce(&mut Binder<'_>),
) -> Result<(), String> {
    if groups_x == 0 || groups_y == 0 || threads_per_tg == 0 {
        return Ok(());
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(pipeline);
        encode_bufs(bnd);
        bnd.dispatch(
            mtl_size(groups_x, groups_y, 1),
            mtl_size(threads_per_tg, 1, 1),
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binder_path_copy_via_dispatch_1d() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i as f32) * 2.0;
            }
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        dispatch_1d(&rt, &pipe, n, |bnd| {
            set_gpu_buf(bnd, &src, 0);
            set_gpu_buf(bnd, &dst, 1);
            set_u32(bnd, n as u32, 2);
        })
        .unwrap();
        rt.synchronize().unwrap();
        let out =
            unsafe { std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n) };
        for (i, v) in out.iter().take(n).enumerate() {
            assert_eq!(*v, (i as f32) * 2.0);
        }
    }

    #[test]
    fn dispatch_1d_rejects_u32_overflow_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.take_dispatch_count();
        let result = dispatch_1d(&rt, &pipe, u32::MAX as usize + 1, |_| {
            panic!("encode closure must not run for an oversized dispatch");
        });
        assert!(result.is_err(), "oversized dispatch_1d accepted");
        assert_eq!(rt.take_dispatch_count(), 0);
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    #[test]
    fn binder_rejects_foreign_runtime_storage() {
        let rt = GpuRuntime::new().unwrap();
        let other = GpuRuntime::new().unwrap();
        let t = other.alloc_tensor_f32(&[4]).unwrap();
        let map = t.buffer.contents_f32();
        assert!(rt
            .with_binder(|b| {
                b.bind_tensor(&t, 0);
                Ok(())
            })
            .is_err());
        assert_eq!(map[0], 0.0);
    }

    #[test]
    fn oversized_constants_return_error_instead_of_panicking() {
        let rt = GpuRuntime::new().unwrap();
        let bytes = vec![0u8; rt.metal4.const_staging.length() + 1];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.with_binder(|b| {
                b.bind_bytes(&bytes, 0);
                Ok(())
            })
        }));
        assert!(result.is_ok(), "Result API panicked on full arena");
        assert!(result.unwrap().is_err());
    }
    #[test]
    fn binder_rejects_bad_view_without_a_dispatch() {
        let rt = GpuRuntime::new().unwrap();
        let mut t = rt.alloc_tensor_f32(&[4]).unwrap();
        t.byte_offset = usize::MAX;
        assert!(rt
            .with_binder(|b| {
                b.bind_tensor(&t, 0);
                Ok(())
            })
            .is_err());
    }

    /// Moved here with `dispatch_2d`/`dispatch_3d`. The extent check must reject
    /// before the caller's encode closure runs — a closure that has already bound
    /// resources into a doomed dispatch is the failure this guards.
    #[test]
    fn oversized_extent_rejected_before_callback() {
        let rt = GpuRuntime::new().unwrap();
        let p = rt.pipeline("copy_f32").unwrap();
        let called = std::cell::Cell::new(false);
        assert!(dispatch_2d(&rt, &p, usize::MAX, 1, |_| called.set(true)).is_err());
        assert!(!called.get(), "oversized 2D dispatch reached encoder");
        assert!(dispatch_3d(&rt, &p, 1, 1, usize::MAX, |_| called.set(true)).is_err());
        assert!(!called.get(), "oversized 3D dispatch reached encoder");
    }
}
