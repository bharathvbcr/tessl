//! Metal 4 bind + dispatch helpers.
//!
//! [`Binder`] targets the Metal 4 argument table + const arena. Call sites use
//! `set_*` / `bind_*` sugar; [`GpuRuntime::with_binder`] opens the command
//! buffer encoder.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
#[cfg(feature = "quant-prep")]
use objc2_metal::MTLTensor;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLResidencySet, MTLResource, MTLResourceID, MTLSize, MTLStages,
};

use crate::runtime::{mtl_size, GpuRuntime, InFlightAnchor};
use crate::tensor::{GpuBuffer, Tensor};

/// `[[threadgroup(n)]]` slots a compute function can declare. Metal's own
/// index space for threadgroup memory arguments, unrelated to the argument
/// table's buffer slots.
pub const THREADGROUP_MEMORY_SLOTS: usize = 32;

#[cfg(test)]
thread_local! {
    /// `setArgumentTable` calls issued by binders on this thread; the
    /// once-per-encoder test reads it.
    pub(crate) static ARG_TABLE_SETS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How a [`Binder`] scope orders its dispatches, carried in from the runtime.
///
/// `skip_auto` is hazard mode for this scope (the value of
/// [`crate::ab_flags::hazard_barriers`] captured once at scope start, or the
/// scope's explicit override). `pending` says the previous scope on the same
/// encoder ended with an unbarriered dispatch, so this scope must order that
/// edge before its own first dispatch.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BarrierPolicy {
    pub(crate) skip_auto: bool,
    pub(crate) pending: bool,
}

/// Metal 4 compute binder (argument table + const arena).
///
/// Raw-address, raw resource-ID, argument-table adoption, and ICB replay are
/// deliberately unavailable to safe external callers: a numeric GPU address
/// and an ICB/table do not carry the complete ownership graph that an
/// unretained Metal 4 command buffer requires.
///
/// ```compile_fail
/// use tessl::dispatch::Binder;
/// fn cannot_bind_an_arbitrary_address(binder: &mut Binder<'_>) {
///     binder.bind_addr(1, 0);
/// }
/// ```
///
/// ```compile_fail
/// use tessl::dispatch::Binder;
/// fn cannot_materialize_an_unowned_address(binder: &mut Binder<'_>) {
///     binder.materialize_bytes(&[1, 2, 3, 4]);
/// }
/// ```
///
/// ```compile_fail
/// use objc2::runtime::ProtocolObject;
/// use objc2_metal::MTL4ArgumentTable;
/// use tessl::dispatch::Binder;
/// fn cannot_adopt_an_unowned_table(
///     binder: &mut Binder<'_>,
///     table: &ProtocolObject<dyn MTL4ArgumentTable>,
/// ) {
///     binder.adopt_argument_table(table);
/// }
/// ```
///
/// ```compile_fail
/// use objc2::runtime::ProtocolObject;
/// use objc2_metal::MTLIndirectCommandBuffer;
/// use tessl::dispatch::Binder;
/// fn cannot_execute_an_unowned_icb(
///     binder: &mut Binder<'_>,
///     icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
/// ) {
///     binder.execute_icb(icb, 0, 1);
/// }
/// ```
///
/// ```compile_fail
/// use objc2::runtime::ProtocolObject;
/// use objc2_metal::MTLIndirectCommandBuffer;
/// use tessl::dispatch::Binder;
/// fn cannot_execute_an_unowned_icb_with_raw_options(
///     binder: &mut Binder<'_>,
///     icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
/// ) {
///     binder.execute_icb_ex(icb, 0, 1, false);
/// }
/// ```
///
/// ```compile_fail
/// use objc2::runtime::ProtocolObject;
/// use objc2_metal::MTLIndirectCommandBuffer;
/// use tessl::dispatch::Binder;
/// fn cannot_optimize_an_unowned_icb(
///     binder: &mut Binder<'_>,
///     icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
/// ) {
///     binder.optimize_icb(icb, 0, 1);
/// }
/// ```
pub struct Binder<'a> {
    runtime: &'a GpuRuntime,
    error: Option<String>,
    max_buffers: usize,
    max_threads: Option<usize>,
    static_tg_memory: Option<usize>,
    dynamic_tg_memory: [usize; THREADGROUP_MEMORY_SLOTS],
    dynamic_tg_memory_total: usize,
    enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
    const_staging: &'a ProtocolObject<dyn MTLBuffer>,
    const_cursor: &'a mut usize,
    /// Raw Objective-C objects retained until this batch's allocator event
    /// completes. Metal 4 command buffers do not retain their object graph.
    in_flight_anchors: Vec<InFlightAnchor>,
    /// `setArgumentTable` has been issued on this encoder. Once per encoder,
    /// not per scope: the runtime carries the latch from one scope to the
    /// next (see [`Self::holds_persistent_table`]).
    arg_table_latched: bool,
    /// Pointer identity of the last adopted / latched argument table (skip redundant
    /// `setArgumentTable` when the same table is reused across tape cmds).
    last_arg_table_ptr: Option<usize>,
    /// Auto-barrier mode latched for this binder scope. Reading the global once
    /// at construction keeps every dispatch and explicit-barrier decision in one
    /// scope consistent even if another thread changes the flag mid-encode.
    skip_auto_barriers: bool,
    /// An unbarriered dispatch (or ICB execute) sits behind the encoder's
    /// current point — from an earlier scope on the same encoder, or from this
    /// one in hazard mode. The next dispatch emits a barrier first and an
    /// explicit [`Self::barrier`] clears it, so cross-scope producer→consumer
    /// edges are ordered without every caller knowing the graph, and a caller
    /// that already barriers its edges pays nothing extra. The runtime carries
    /// the flag from one scope to the next.
    hazard_pending: bool,
}

#[cfg(test)]
std::thread_local! {
    /// Test-only evidence that scope teardown reaches the native reset seam.
    /// Thread-local storage keeps focused assertions independent of parallel
    /// lib-test workers.
    static DYNAMIC_TG_NATIVE_CLEAR_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

impl<'a> Binder<'a> {
    /// Whether this binder encodes against `runtime`'s command queue and
    /// residency set.
    ///
    /// Resource IDs are device-global values, so accepting one from a
    /// different [`GpuRuntime`] can appear to work while bypassing the runtime
    /// that owns the resource's residency and completion lifetime. Safe
    /// resource-binding wrappers use this identity check before mutating the
    /// argument table.
    #[cfg(feature = "quant-prep")]
    #[inline]
    pub(crate) fn belongs_to_runtime(&self, runtime: &GpuRuntime) -> bool {
        std::ptr::eq(self.runtime, runtime)
    }

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

    fn valid_device(&mut self, device: &ProtocolObject<dyn MTLDevice>, object: &str) -> bool {
        if !std::ptr::eq(device, self.runtime.device.as_ref()) {
            self.fail(format!("{object} belongs to another Metal device"));
        }
        self.error.is_none()
    }

    /// Reset every dynamic threadgroup slot this binder installed.
    ///
    /// Metal encoder state survives individual dispatches (and, in async
    /// mode, binder scopes), so zeroing only the Rust totals would let a later
    /// pipeline inherit stale native lengths. Deliberately do not emit capture
    /// notes here: pipeline selection already starts a new captured command,
    /// while scope teardown is not itself a replayable operation.
    fn clear_dynamic_threadgroup_memory(&mut self) {
        for index in 0..self.dynamic_tg_memory.len() {
            if self.dynamic_tg_memory[index] == 0 {
                continue;
            }
            // SAFETY: only slots previously validated and installed by this
            // binder are reset, while its borrowed encoder is still live.
            unsafe {
                self.enc.setThreadgroupMemoryLength_atIndex(0, index as _);
            }
            self.dynamic_tg_memory[index] = 0;
            #[cfg(test)]
            DYNAMIC_TG_NATIVE_CLEAR_COUNT.with(|count| count.set(count.get() + 1));
        }
        self.dynamic_tg_memory_total = 0;
    }

    fn retain_buffer(&mut self, buffer: &ProtocolObject<dyn MTLBuffer>) -> bool {
        // SAFETY: the shared reference proves the Objective-C receiver is live
        // for this call. The returned +1 retain is transferred to the active
        // batch before any closure error or panic is propagated.
        let Some(buffer) = (unsafe {
            Retained::<ProtocolObject<dyn MTLBuffer>>::retain(buffer as *const _ as *mut _)
        }) else {
            self.fail("failed to retain Metal buffer");
            return false;
        };
        self.in_flight_anchors.push(InFlightAnchor::Buffer(buffer));
        true
    }

    fn retain_pipeline(
        &mut self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // SAFETY: as `retain_buffer`; the input borrow is a live receiver.
        let Some(pipeline) = (unsafe {
            Retained::<ProtocolObject<dyn MTLComputePipelineState>>::retain(
                pipeline as *const _ as *mut _,
            )
        }) else {
            self.fail("failed to retain Metal compute pipeline");
            return None;
        };
        Some(pipeline)
    }

    fn retain_argument_table(&mut self, table: &ProtocolObject<dyn MTL4ArgumentTable>) -> bool {
        // SAFETY: as `retain_buffer`; the input borrow is a live receiver.
        let Some(table) = (unsafe {
            Retained::<ProtocolObject<dyn MTL4ArgumentTable>>::retain(table as *const _ as *mut _)
        }) else {
            self.fail("failed to retain Metal argument table");
            return false;
        };
        self.in_flight_anchors
            .push(InFlightAnchor::ArgumentTable(table));
        true
    }

    fn retain_icb(&mut self, icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>) -> bool {
        // SAFETY: as `retain_buffer`; the input borrow is a live receiver.
        let Some(icb) = (unsafe {
            Retained::<ProtocolObject<dyn MTLIndirectCommandBuffer>>::retain(
                icb as *const _ as *mut _,
            )
        }) else {
            self.fail("failed to retain Metal indirect command buffer");
            return false;
        };
        self.in_flight_anchors.push(InFlightAnchor::Icb(icb));
        true
    }

    #[cfg(feature = "quant-prep")]
    fn retain_mtl_tensor(&mut self, tensor: &ProtocolObject<dyn MTLTensor>) -> bool {
        // SAFETY: as `retain_buffer`; the input borrow is a live receiver. The
        // resulting +1 retain follows the command-buffer completion anchor,
        // independently of the tensor's backing-allocation ownership policy.
        let Some(tensor) = (unsafe {
            Retained::<ProtocolObject<dyn MTLTensor>>::retain(tensor as *const _ as *mut _)
        }) else {
            self.fail("failed to retain Metal tensor");
            return false;
        };
        self.in_flight_anchors.push(InFlightAnchor::Tensor(tensor));
        true
    }

    pub(crate) fn take_in_flight_anchors(&mut self) -> Vec<InFlightAnchor> {
        std::mem::take(&mut self.in_flight_anchors)
    }

    /// Whether an unbarriered dispatch still sits behind the encoder's current
    /// point. The runtime carries it to the next scope on the same encoder.
    pub(crate) fn hazard_pending(&self) -> bool {
        self.hazard_pending
    }

    /// Adopt the runtime's persistent table as already bound on the encoder,
    /// left there by the previous scope. Pointer identity is recorded so a
    /// tape-path adopt of a different table still re-sets it.
    pub(crate) fn assume_persistent_table(&mut self) {
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(self.table as *const _ as usize);
    }

    /// Whether the encoder leaves this scope holding the runtime's persistent
    /// table, so the next scope on it can skip `setArgumentTable`. False after
    /// a tape-path adopt of another table or an indirect execute. A fresh
    /// binder per scope used to re-issue the set on every op even though the
    /// encoder is reused across scopes (audit R10c).
    pub(crate) fn holds_persistent_table(&self) -> bool {
        self.arg_table_latched && self.last_arg_table_ptr == Some(self.table as *const _ as usize)
    }

    /// Validate and bind one concrete buffer address without making any claim
    /// about whether the caller can retain that buffer for an ICB tape.
    ///
    /// The public raw-buffer path records an unretainable bind after this
    /// succeeds. The owned [`GpuBuffer`] / [`Tensor`] paths instead record the
    /// owning handle. Keeping those two cases out of this primitive prevents a
    /// fully recordable bind from also poisoning the tape as "incomplete".
    fn checked_buf_address(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) -> Option<u64> {
        if !self.valid_index(index) {
            return None;
        }
        if offset >= buf.length() {
            self.fail("buffer binding offset out of bounds");
            return None;
        }
        let Ok(offset) = u64::try_from(offset) else {
            self.fail("buffer binding offset does not fit a GPU address");
            return None;
        };
        let Some(addr) = buf.gpuAddress().checked_add(offset) else {
            self.fail("GPU address overflow");
            return None;
        };
        Some(addr)
    }

    fn bind_buf_address(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) -> bool {
        let Some(addr) = self.checked_buf_address(buf, offset, index) else {
            return false;
        };
        self.bind_addr(addr, index);
        self.error.is_none()
    }
    fn write_constants(&mut self, bytes: &[u8]) -> u64 {
        if self.error.is_some() {
            return 0;
        }
        // Natural alignment, four to sixteen bytes: a scalar takes four, a
        // pair eight, anything wider sixteen. Metal's constant address space
        // wants the payload's own alignment, and rounding every payload up to
        // sixteen charged the arena four times over for the u32/f32 scalars
        // that make up nearly all of it (audit R8).
        // `constant_arena_charges_natural_width` pins the rule;
        // `a_kernel_reads_a_scalar_at_a_four_byte_offset` shows the GPU reads
        // a four-aligned constant correctly.
        let len = bytes.len().max(4);
        let align = len.next_power_of_two().min(16);
        let start = self
            .const_cursor
            .checked_add(align - 1)
            .map(|n| n & !(align - 1));
        let end = start.and_then(|n| n.checked_add(len));
        let (Some(start), Some(end)) = (start, end) else {
            self.fail("constant arena offset overflow");
            return 0;
        };
        if bytes.is_empty() || end > self.const_staging.length() {
            self.fail("constant arena exhausted or empty payload");
            return 0;
        }
        let Ok(start_u64) = u64::try_from(start) else {
            self.fail("constant arena offset does not fit a GPU address");
            return 0;
        };
        let Some(gpu_addr) = self.const_staging.gpuAddress().checked_add(start_u64) else {
            self.fail("constant arena GPU address overflow");
            return 0;
        };
        if gpu_addr == 0 {
            self.fail("constant arena produced a null GPU address");
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
            std::ptr::write_bytes(dst, 0, len);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
        *self.const_cursor = end;
        gpu_addr
    }

    pub(crate) fn new(
        enc: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        table: &'a ProtocolObject<dyn MTL4ArgumentTable>,
        const_staging: &'a ProtocolObject<dyn MTLBuffer>,
        const_cursor: &'a mut usize,
        barriers: BarrierPolicy,
        max_buffers: usize,
        runtime: &'a GpuRuntime,
    ) -> Self {
        let BarrierPolicy {
            skip_auto: skip_auto_barriers,
            pending: hazard_pending,
        } = barriers;
        Self {
            runtime,
            error: None,
            max_buffers,
            max_threads: None,
            static_tg_memory: None,
            dynamic_tg_memory: [0; THREADGROUP_MEMORY_SLOTS],
            dynamic_tg_memory_total: 0,
            enc,
            table,
            const_staging,
            const_cursor,
            in_flight_anchors: Vec::new(),
            arg_table_latched: false,
            last_arg_table_ptr: None,
            skip_auto_barriers,
            hazard_pending,
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
    /// adopted via the crate-private `adopt_argument_table`) — do not overwrite.
    #[inline]
    pub fn latch_argument_table(&mut self) {
        if self.arg_table_latched {
            return;
        }
        self.enc.setArgumentTable(Some(self.table));
        #[cfg(test)]
        ARG_TABLE_SETS.with(|n| n.set(n.get() + 1));
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
    pub(crate) fn adopt_argument_table(
        &mut self,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
    ) -> bool {
        let ptr = table as *const _ as usize;
        if self.arg_table_latched && self.last_arg_table_ptr == Some(ptr) {
            return false;
        }
        let device = table.device();
        if !self.valid_device(&device, "argument table") || !self.retain_argument_table(table) {
            return false;
        }
        self.enc.setArgumentTable(Some(table));
        #[cfg(test)]
        ARG_TABLE_SETS.with(|n| n.set(n.get() + 1));
        self.arg_table_latched = true;
        self.last_arg_table_ptr = Some(ptr);
        true
    }

    /// Copy bytes into the const arena; return the GPU address.
    ///
    /// Does **not** call `setAddress` or capture — used when writing into a
    /// prebuilt per-command argument table (Immediate residual only).
    pub(crate) fn materialize_bytes(&mut self, bytes: &[u8]) -> u64 {
        self.write_constants(bytes)
    }

    /// Select a pipeline and retain it through this command buffer's completion.
    ///
    /// The retain is intentional even for pipelines normally returned by
    /// [`GpuRuntime::pipeline`]: this safe method is public and can receive an
    /// independently created same-device pipeline that the runtime cache does
    /// not own. Allocator-slot dedup keeps one stored reference per unique
    /// pipeline per command buffer without adding a mutex to each dispatch.
    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        if self.error.is_some() {
            return;
        }
        let device = pipeline.device();
        if !self.valid_device(&device, "compute pipeline") {
            return;
        }

        // Every explicit pipeline selection is a dynamic-memory reset
        // boundary, including re-selecting the same pipeline during ICB tape
        // replay. Consecutive dispatches with no intervening selection retain
        // the caller's explicit lengths.
        self.clear_dynamic_threadgroup_memory();
        self.max_threads = None;
        self.static_tg_memory = None;

        let static_tg_memory = pipeline.staticThreadgroupMemoryLength();
        let max_tg_memory = self.runtime.max_threadgroup_memory();
        if static_tg_memory
            .checked_add(self.dynamic_tg_memory_total)
            .is_none_or(|total| total > max_tg_memory)
        {
            self.fail(format!(
                "pipeline threadgroup memory exceeds device limit: static={static_tg_memory}, dynamic={}, limit={max_tg_memory}",
                self.dynamic_tg_memory_total
            ));
            return;
        }
        let Some(retained) = self.retain_pipeline(pipeline) else {
            return;
        };
        if crate::decode_icb::decode_icb_capture_active() {
            crate::decode_icb::capture_note_pipeline(retained.clone());
        }
        self.in_flight_anchors
            .push(InFlightAnchor::Pipeline(retained));
        self.max_threads = Some(pipeline.maxTotalThreadsPerThreadgroup());
        self.static_tg_memory = Some(static_tg_memory);
        self.enc.setComputePipelineState(pipeline);
    }

    pub(crate) fn bind_buf(&mut self, buf: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize) {
        // A raw `MTLBuffer` carries no owning `GpuBuffer`, so there is nothing
        // for the capture tape to record or to pin. Mark the tape incomplete
        // instead of letting it silently omit the operand. Invalid binds never
        // reached the encoder and therefore must not contaminate a capture.
        if self.error.is_some() {
            return;
        }
        let device = buf.device();
        let allocation = ProtocolObject::<dyn MTLAllocation>::from_ref(buf);
        if !self.valid_device(&device, "buffer") {
            return;
        }
        if !self.runtime.metal4.residency.containsAllocation(allocation) {
            self.fail("raw buffer is not registered with this runtime's residency set");
            return;
        }
        // Validate offset/address before retaining, then retain before mutating
        // the argument table. That makes every failure-before-bind path clean.
        let Some(addr) = self.checked_buf_address(buf, offset, index) else {
            return;
        };
        if !self.retain_buffer(buf) {
            return;
        }
        self.bind_addr(addr, index);
        if self.error.is_none() {
            crate::decode_icb::capture_note_unrecordable_bind();
        }
    }

    /// Bind a precomputed GPU address (DecodeIcb tape replay bind-tax cut).
    #[inline]
    pub(crate) fn bind_addr(&mut self, gpu_addr: u64, index: usize) {
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
        if self.bind_buf_address(t.buffer.metal(), t.byte_offset, index)
            && crate::decode_icb::decode_icb_capture_active()
        {
            crate::decode_icb::capture_note_bind(index, &t.buffer, t.byte_offset);
        }
    }

    pub fn bind_gpu_buf(&mut self, b: &GpuBuffer, index: usize) {
        self.bind_gpu_buf_offset(b, 0, index);
    }

    /// Bind an owned buffer at a byte offset and retain that exact bind in an
    /// active DecodeIcb capture.
    pub fn bind_gpu_buf_offset(&mut self, b: &GpuBuffer, byte_offset: usize, index: usize) {
        if !b.belongs_to(self.runtime) {
            self.fail("buffer belongs to another runtime");
            return;
        }
        if byte_offset >= b.nbytes() {
            self.fail("owned buffer binding offset out of logical bounds");
            return;
        }
        if self.bind_buf_address(b.metal(), byte_offset, index)
            && crate::decode_icb::decode_icb_capture_active()
        {
            crate::decode_icb::capture_note_bind(index, b, byte_offset);
        }
    }

    /// Bind an `MTLResourceID` (e.g. a `GpuTensor` from the `quant-prep`
    /// feature's `mtl_tensor` module) at a buffer index.
    ///
    /// # Safety
    ///
    /// `resource_id` must name a resource that stays alive and resident until
    /// every submitted command that uses it has completed. That is the caller's
    /// to guarantee and cannot be checked here, which is why this stays
    /// `unsafe`.
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

    /// Bind a concrete `MTLTensor` and retain that object until the submitted
    /// command buffer completes.
    ///
    /// This is the owned counterpart to [`Self::bind_resource_id`]. The latter
    /// cannot recover an Objective-C object from a numeric ID, so its lifetime
    /// remains an unsafe caller obligation; this seam carries the object into
    /// the Binder's allocator-slot anchors before mutating the argument table.
    #[cfg(feature = "quant-prep")]
    pub(crate) fn bind_mtl_tensor_resource(
        &mut self,
        tensor: &ProtocolObject<dyn MTLTensor>,
        index: usize,
    ) {
        if self.error.is_some() || !self.valid_index(index) {
            return;
        }
        let device = tensor.device();
        if !self.valid_device(&device, "Metal tensor") || !self.retain_mtl_tensor(tensor) {
            return;
        }
        crate::decode_icb::capture_note_unrecordable_bind();
        unsafe {
            self.table
                .setResource_atBufferIndex(tensor.gpuResourceID(), index as _);
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
    ///
    /// `[[threadgroup(n)]]` is its own index space, separate from
    /// `[[buffer(n)]]`, with its own limit ([`THREADGROUP_MEMORY_SLOTS`]).
    /// It used to be bounded by the buffer bind count, which was conservative
    /// but not a Metal fact, and would have moved for no reason with the
    /// argument-table size.
    pub fn set_threadgroup_memory(&mut self, index: usize, length: usize) {
        if index >= THREADGROUP_MEMORY_SLOTS {
            self.fail(format!(
                "threadgroup memory index {index} out of range (limit {THREADGROUP_MEMORY_SLOTS})"
            ));
            return;
        }
        let Some(static_bytes) = self.static_tg_memory else {
            self.fail("threadgroup memory set before a compute pipeline");
            return;
        };
        if length % 16 != 0 {
            self.fail(format!(
                "dynamic threadgroup memory length must be 16-byte aligned: index={index}, length={length}"
            ));
            return;
        }
        let old = self.dynamic_tg_memory[index];
        let Some(next_dynamic) = self
            .dynamic_tg_memory_total
            .checked_sub(old)
            .and_then(|total| total.checked_add(length))
        else {
            self.fail("dynamic threadgroup memory size overflow");
            return;
        };
        let Some(total) = static_bytes.checked_add(next_dynamic) else {
            self.fail("static plus dynamic threadgroup memory size overflow");
            return;
        };
        let limit = self.runtime.max_threadgroup_memory();
        if total > limit {
            self.fail(format!(
                "pipeline threadgroup memory exceeds device limit: static={static_bytes}, dynamic={next_dynamic}, limit={limit}"
            ));
            return;
        }
        // SAFETY: `index` is within the argument table's bind count and the sum
        // of this pipeline's static memory plus every dynamic slot is within
        // the device limit, checked above. `self.enc` is live for this borrow.
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex(length as _, index as _);
        }
        self.dynamic_tg_memory[index] = length;
        self.dynamic_tg_memory_total = next_dynamic;
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
        if let Err(error) =
            validate_dispatch_geometry(threadgroups, threads_per_tg, self.max_threads)
        {
            self.fail(error);
            return;
        }

        if self.hazard_pending {
            // A producer with no barrier after it precedes this dispatch on the
            // encoder (hazard mode, possibly from an earlier scope). Order it now.
            self.barrier();
        }
        self.latch_argument_table();
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        crate::infer_trace::on_dispatch();
        self.hazard_pending = self.skip_auto_barriers;
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
        self.hazard_pending = false;
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
    pub(crate) fn execute_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        self.execute_icb_ex(icb, start, count, true);
    }

    /// Like [`Self::execute_icb`] with explicit inherit-table control.
    pub(crate) fn execute_icb_ex(
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
        if let Err(error) = validate_icb_range(icb.size(), start, count) {
            self.fail(format!("ICB execute {error}"));
            return;
        }
        let device = icb.device();
        let allocation = ProtocolObject::<dyn MTLAllocation>::from_ref(icb);
        if !self.valid_device(&device, "indirect command buffer") {
            return;
        }
        if !self.runtime.metal4.residency.containsAllocation(allocation) {
            self.fail(
                "indirect command buffer is not registered with this runtime's residency set",
            );
            return;
        }
        if !self.retain_icb(icb) {
            return;
        }
        if self.hazard_pending {
            self.barrier();
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
        // The latch does not trust the table binding to survive an indirect
        // execute: the next dispatch, in this scope or a later one on the
        // same encoder, sets it again.
        self.arg_table_latched = false;
        self.last_arg_table_ptr = None;
        crate::infer_trace::on_dispatch();
        self.hazard_pending = self.skip_auto_barriers;
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
    pub(crate) fn optimize_icb(
        &mut self,
        icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
        start: u64,
        count: u64,
    ) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = validate_icb_range(icb.size(), start, count) {
            self.fail(format!("ICB optimize {error}"));
            return;
        }
        let device = icb.device();
        let allocation = ProtocolObject::<dyn MTLAllocation>::from_ref(icb);
        if !self.valid_device(&device, "indirect command buffer") {
            return;
        }
        if !self.runtime.metal4.residency.containsAllocation(allocation) {
            self.fail(
                "indirect command buffer is not registered with this runtime's residency set",
            );
            return;
        }
        if !self.retain_icb(icb) {
            return;
        }
        let range = NSRange {
            location: start as _,
            length: count as _,
        };
        unsafe {
            self.enc.optimizeIndirectCommandBuffer_withRange(icb, range);
        }
    }
}

impl Drop for Binder<'_> {
    fn drop(&mut self) {
        // The runtime may keep the same native encoder open for the next
        // `with_binder` scope, including after a closure error or panic.
        self.clear_dynamic_threadgroup_memory();
    }
}

/// Validate the geometry common to direct and captured Metal dispatches.
///
/// Tessl's shader-side coordinates and element counts are `uint`, so each grid
/// dimension is intentionally capped at `u32::MAX` even though `MTLSize` uses
/// the wider host `NSUInteger` type.
pub(crate) fn validate_dispatch_geometry(
    threadgroups: MTLSize,
    threads_per_tg: MTLSize,
    max_threads: Option<usize>,
) -> Result<(), String> {
    let grid_ok = [threadgroups.width, threadgroups.height, threadgroups.depth]
        .iter()
        .all(|&n| n > 0 && n <= u32::MAX as usize);
    let lanes = threads_per_tg
        .width
        .checked_mul(threads_per_tg.height)
        .and_then(|n| n.checked_mul(threads_per_tg.depth));
    let threads_ok = lanes
        .zip(max_threads)
        .is_some_and(|(n, max)| n > 0 && n <= max);
    if !grid_ok || !threads_ok {
        return Err("invalid dispatch geometry or missing pipeline".into());
    }
    Ok(())
}

fn validate_icb_range(icb_len: usize, start: u64, count: u64) -> Result<(), String> {
    if count == 0 {
        return Err("range must contain at least one command".into());
    }
    let len = u64::try_from(icb_len).map_err(|_| "command count does not fit u64")?;
    if start.checked_add(count).is_none_or(|end| end > len) {
        return Err("range out of bounds".into());
    }
    Ok(())
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
    bnd.bind_gpu_buf_offset(buf, byte_offset, index);
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
    if groups_x > u32::MAX as usize || groups_y > u32::MAX as usize {
        return Err("threadgroup grid exceeds uint shader indexing".into());
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if threads_per_tg > max_threads {
        return Err(format!(
            "threads per threadgroup {threads_per_tg} exceeds pipeline maximum {max_threads}"
        ));
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

    /// `[[threadgroup(n)]]` has its own slot space: the last slot Metal allows
    /// is accepted and the first past it is refused by name, independent of
    /// how many buffer slots the argument table has.
    #[test]
    fn threadgroup_memory_index_is_bounded_by_its_own_slot_space() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").unwrap();
        let err = rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&pipe);
                bnd.set_threadgroup_memory(THREADGROUP_MEMORY_SLOTS, 16);
                Ok(())
            })
            .expect_err("slot past the threadgroup limit");
        assert!(
            err.contains("threadgroup memory index 32 out of range"),
            "unexpected: {err}"
        );
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

    #[test]
    fn dispatch_2d_tg_rejects_unrepresentable_grid_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.take_dispatch_count();
        let result = dispatch_2d_tg(&rt, &pipe, u32::MAX as usize + 1, 1, 1, |_| {
            panic!("encode closure must not run for an oversized grid")
        });
        assert!(result.is_err(), "oversized threadgroup grid accepted");
        assert_eq!(rt.take_dispatch_count(), 0);
    }

    #[test]
    fn dispatch_2d_tg_rejects_pipeline_threadgroup_overflow_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.take_dispatch_count();
        let too_many = pipe.maxTotalThreadsPerThreadgroup() + 1;
        let result = dispatch_2d_tg(&rt, &pipe, 1, 1, too_many, |_| {
            panic!("encode closure must not run for an oversized threadgroup")
        });
        assert!(result.is_err(), "oversized threadgroup width accepted");
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

    #[test]
    fn owned_buffer_capture_is_not_marked_unrecordable() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").expect("pipeline");
        let buf = rt.alloc_buffer_hot(16).expect("buffer");
        crate::decode_icb::begin_decode_icb_capture();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&buf, 0);
            Ok(())
        })
        .expect("host-only binder encode");
        crate::decode_icb::capture_note_dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        let cap = crate::decode_icb::take_decode_icb_capture().expect("capture");
        assert_eq!(cap.commands.len(), 1);
        assert_eq!(
            cap.commands[0].incomplete_binds, 0,
            "an owned GpuBuffer bind is fully recordable"
        );
    }

    #[test]
    fn dynamic_threadgroup_memory_is_bounded_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").expect("pipeline");
        let too_large = (rt.max_threadgroup_memory() / 16 + 1) * 16;
        let err = rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&pipe);
                bnd.set_threadgroup_memory(0, too_large);
                Ok(())
            })
            .expect_err("oversized dynamic threadgroup storage must be rejected");
        assert!(
            err.contains("threadgroup memory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dynamic_threadgroup_memory_is_bounded_across_slots() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("copy_f32").expect("pipeline");
        let available = rt
            .max_threadgroup_memory()
            .checked_sub(pipe.staticThreadgroupMemoryLength())
            .expect("pipeline static memory fits device");
        let each = (available / 2 / 16 + 1) * 16;
        assert!(
            each <= available,
            "device limit is unexpectedly below 32 bytes"
        );
        let err = rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&pipe);
                bnd.set_threadgroup_memory(0, each);
                bnd.set_threadgroup_memory(1, each);
                Ok(())
            })
            .expect_err("individually valid slots must still respect the aggregate limit");
        assert!(
            err.contains("threadgroup memory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pipeline_selection_clears_dynamic_threadgroup_bookkeeping() {
        let rt = GpuRuntime::new().expect("runtime");
        let dynamic_pipe = rt.pipeline("gemv_q4").expect("dynamic pipeline");
        let next_pipe = rt.pipeline("copy_f32").expect("next pipeline");
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&dynamic_pipe);
            bnd.set_threadgroup_memory(0, 16);
            assert_eq!(bnd.dynamic_tg_memory[0], 16);
            assert_eq!(bnd.dynamic_tg_memory_total, 16);

            bnd.set_pipeline(&next_pipe);
            assert!(
                bnd.dynamic_tg_memory.iter().all(|&length| length == 0),
                "a pipeline boundary retained a dynamic threadgroup slot"
            );
            assert_eq!(bnd.dynamic_tg_memory_total, 0);
            Ok(())
        })
        .expect("pipeline transition");
    }

    #[test]
    fn dynamic_threadgroup_memory_requires_16_byte_alignment_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("gemv_q4").expect("dynamic pipeline");
        let bookkeeping_unchanged = std::cell::Cell::new(false);
        let err = rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&pipe);
                bnd.set_threadgroup_memory(0, 15);
                bookkeeping_unchanged.set(
                    bnd.dynamic_tg_memory.iter().all(|&length| length == 0)
                        && bnd.dynamic_tg_memory_total == 0,
                );
                Ok(())
            })
            .expect_err("an unaligned dynamic threadgroup length must be rejected");
        assert!(
            err.contains("16-byte aligned") && err.contains("index=0") && err.contains("length=15"),
            "unexpected error: {err}"
        );
        assert!(
            bookkeeping_unchanged.get(),
            "alignment rejection mutated dynamic threadgroup bookkeeping"
        );
    }

    #[test]
    fn dynamic_threadgroup_memory_persists_only_between_dispatches() {
        let rt = GpuRuntime::new().expect("runtime");
        let dynamic_pipe = rt.pipeline("gemv_q4").expect("dynamic pipeline");
        let next_pipe = rt.pipeline("copy_f32").expect("next pipeline");
        let packed = rt.alloc_buffer(4).expect("packed");
        packed.write_bytes(&[0; 4]);
        let scales = rt.alloc_buffer(4).expect("scales");
        scales.write_f32(&[1.0]);
        let zeros = rt.alloc_buffer(4).expect("zeros");
        zeros.write_f32(&[0.0]);
        let x = rt.alloc_buffer(8 * 4).expect("x");
        x.write_f32(&[1.0; 8]);
        let y = rt.alloc_buffer(4).expect("y");
        y.write_f32(&[f32::NAN]);

        rt.with_binder(|bnd| {
            bnd.set_pipeline(&dynamic_pipe);
            bnd.bind_gpu_buf(&packed, 0);
            bnd.bind_gpu_buf(&scales, 1);
            bnd.bind_gpu_buf(&zeros, 2);
            bnd.bind_gpu_buf(&x, 3);
            bnd.bind_gpu_buf(&y, 4);
            bnd.bind_u32(1, 5);
            bnd.bind_u32(8, 6);
            bnd.bind_u32(8, 7);
            bnd.set_threadgroup_memory(0, 32);

            for dispatch_index in 0..2 {
                bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
                assert_eq!(
                    bnd.dynamic_tg_memory[0], 32,
                    "dispatch {dispatch_index} dropped an explicit dynamic slot"
                );
                assert_eq!(bnd.dynamic_tg_memory_total, 32);
            }

            bnd.set_pipeline(&dynamic_pipe);
            assert!(
                bnd.dynamic_tg_memory.iter().all(|&length| length == 0),
                "explicitly re-selecting the same pipeline retained dynamic state"
            );
            assert_eq!(bnd.dynamic_tg_memory_total, 0);

            bnd.set_threadgroup_memory(0, 32);
            bnd.set_pipeline(&next_pipe);
            assert!(
                bnd.dynamic_tg_memory.iter().all(|&length| length == 0),
                "selecting a different pipeline retained dynamic state"
            );
            assert_eq!(bnd.dynamic_tg_memory_total, 0);
            Ok(())
        })
        .expect("two dynamic-memory dispatches and pipeline transitions");
        rt.synchronize()
            .expect("dynamic-memory dispatch completion");
        assert_eq!(y.contents_f32()[0], 0.0);
    }

    #[test]
    fn binder_scope_drop_clears_native_dynamic_threadgroup_slots() {
        DYNAMIC_TG_NATIVE_CLEAR_COUNT.with(|count| count.set(0));
        let rt = GpuRuntime::new().expect("runtime");
        let pipe = rt.pipeline("gemv_q4").expect("dynamic pipeline");
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.set_threadgroup_memory(0, 16);
            Ok(())
        })
        .expect("host-only dynamic-memory encode scope");
        DYNAMIC_TG_NATIVE_CLEAR_COUNT.with(|count| {
            assert_eq!(
                count.get(),
                1,
                "binder drop did not reset its one live native dynamic slot"
            );
            count.set(0);
        });
    }

    #[test]
    fn icb_ranges_reject_empty_overflowing_and_past_end_requests() {
        assert!(validate_icb_range(4, 0, 0).is_err());
        assert!(validate_icb_range(4, u64::MAX, 2).is_err());
        assert!(validate_icb_range(4, 3, 2).is_err());
        validate_icb_range(4, 0, 4).expect("whole nonempty range");
        validate_icb_range(4, 3, 1).expect("last command");
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
