//! Mini compute ICB smoke (opt-in): encode `copy_f32` once via classic
//! `setKernelBuffer`, then `MTL4ComputeCommandEncoder::executeCommandsInBuffer`
//! on later steps.
//!
//! ## Status
//!
//! Proves the SDK path works on MacOSX26 / objc2-metal 0.3 with Metal 4 encode.
//! Does **not** migrate the decode graph (argument-table + const-arena) — see
//! [`crate::cb_replay::survey_cb_replay_api_gaps`].
//!
//! ## Flag
//!
//! Default **OFF**. Enable with `METAL_RUNTIME_ICB_SMOKE=1` or
//! `GEMMA_METAL_ICB_SMOKE=1`, or [`set_icb_smoke`] in tests. The smoke helpers
//! themselves do not require the flag (tests call them directly); the flag is
//! the opt-in gate for any future session wiring.

use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{Arc, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4IndirectCommandBufferSupportState, MTL4LibraryFunctionDescriptor, MTLAllocation, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResourceOptions,
};

use crate::ab_flags::env_truthy;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

/// -1 = read env once, 0 = off, 1 = on.
static ICB_SMOKE: AtomicI8 = AtomicI8::new(-1);

/// Force ICB-smoke opt-in on/off (tests / harness). Overrides env.
pub fn set_icb_smoke(on: bool) {
    ICB_SMOKE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in gate for ICB smoke wiring. Default OFF.
pub fn icb_smoke_enabled() -> bool {
    let v = ICB_SMOKE.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_truthy(&[
        "TESSL_ICB_SMOKE",
        "METAL_RUNTIME_ICB_SMOKE",
        "GEMMA_METAL_ICB_SMOKE",
    ])
    .unwrap_or(false);
    ICB_SMOKE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Bind bridge mode for the mini ICB smoke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbBindBridge {
    /// Freeze classic `setKernelBuffer` into the ICB (Metal 3-style).
    /// Observed **not** to feed MTL4 pipelines on MacOSX26 — kept for A/B.
    ClassicKernelBuffer,
    /// Encode pipeline+dispatch in ICB; bind resources via MTL4 argument table
    /// at `executeCommandsInBuffer` time (`inheritBuffers=true`).
    InheritArgTable,
}

/// One-command ConcurrentDispatch ICB for `copy_f32` smoke.
pub struct IcbCopySmoke {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    runtime: Weak<GpuRuntime>,
    runtime_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Stable `u32` length bind (ICB / arg-table cannot use bump const-arena).
    n_buf: GpuBuffer,
    n: usize,
    bridge: IcbBindBridge,
    /// Retained for classic-bridge re-encode / inherit execute binds.
    src: Option<GpuBuffer>,
    dst: Option<GpuBuffer>,
    encoded: bool,
    optimized: bool,
    execute_count: u64,
}

impl IcbCopySmoke {
    /// Allocate ICB + ICB-capable `copy_f32` pipeline + stable `n` buffer.
    pub fn new(rt: &GpuRuntime, n: usize) -> Result<Self, String> {
        Self::new_with_bridge(rt, n, IcbBindBridge::InheritArgTable)
    }

    pub fn new_with_bridge(
        rt: &GpuRuntime,
        n: usize,
        bridge: IcbBindBridge,
    ) -> Result<Self, String> {
        let n_u32 = copy_count_u32(n, "icb smoke")?;
        let pipeline = pipeline_copy_f32_icb(rt)?;
        let n_buf = rt.alloc_buffer_hot(4)?;
        // SAFETY: `n_buf` is a 4-byte Hot buffer allocated on the line above
        // with no other handle to it, so this single `u32` write is unaliased
        // and exactly fills it. `contents()` on a shared-storage buffer is
        // Metal-aligned, well past `u32`'s requirement.
        unsafe {
            let p = n_buf.metal().contents().as_ptr() as *mut u32;
            *p = n_u32;
        }

        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        match bridge {
            IcbBindBridge::ClassicKernelBuffer => {
                desc.setInheritBuffers(false);
                desc.setInheritPipelineState(false);
                desc.setMaxKernelBufferBindCount(3);
            }
            IcbBindBridge::InheritArgTable => {
                // MTL4: resource binds come from the encoder's argument table.
                desc.setInheritBuffers(true);
                desc.setInheritPipelineState(false);
                desc.setMaxKernelBufferBindCount(0);
            }
        }

        let icb = unsafe {
            rt.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    1,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| "newIndirectCommandBuffer failed (ConcurrentDispatch)".to_string())?;

        // MTL4 residency: ICB must be registered alongside its bound buffers.
        rt.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*icb));

        Ok(Self {
            icb,
            runtime: rt.weak_handle(),
            runtime_table: rt.metal4.argument_table.table.clone(),
            pipeline,
            n_buf,
            n,
            bridge,
            src: None,
            dst: None,
            encoded: false,
            optimized: false,
            execute_count: 0,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn bridge(&self) -> IcbBindBridge {
        self.bridge
    }

    pub fn encoded(&self) -> bool {
        self.encoded
    }

    pub fn execute_count(&self) -> u64 {
        self.execute_count
    }

    pub fn status_line(&self) -> String {
        format!(
            "icb_copy_smoke n={} bridge={:?} encoded={} optimized={} executes={} cmds=1",
            self.n, self.bridge, self.encoded, self.optimized, self.execute_count
        )
    }

    /// CPU-encode command 0 once (pipeline + dispatch; binds per [`IcbBindBridge`]).
    pub fn encode_copy(&mut self, src: &GpuBuffer, dst: &GpuBuffer) -> Result<(), String> {
        // This object is explicitly an encode-once tape. Resetting its shared
        // ICB after execute may race an async GPU consumer, and replacing the
        // retained operands can release the resources that consumer still
        // references. Refuse before validation or any native mutation.
        if self.encoded {
            return Err("icb smoke: command is already encoded; construct a new tape".into());
        }
        let required = copy_nbytes(self.n, "icb smoke")?;
        if src.nbytes() < required || dst.nbytes() < required {
            return Err(format!(
                "icb smoke: copy buffer too small for {} bytes (src={}, dst={})",
                required,
                src.nbytes(),
                dst.nbytes()
            ));
        }
        let runtime = self
            .runtime
            .upgrade()
            .ok_or_else(|| "icb smoke: owning runtime has been dropped".to_string())?;
        if !src.belongs_to(&runtime) || !dst.belongs_to(&runtime) {
            return Err("icb smoke: buffers belong to another runtime".into());
        }
        let width = self.pipeline.threadExecutionWidth();
        let tpt = width.min(self.n).max(1);
        let groups = self.n.div_ceil(tpt);
        crate::dispatch::validate_dispatch_geometry(
            mtl_size(groups, 1, 1),
            mtl_size(tpt, 1, 1),
            Some(self.pipeline.maxTotalThreadsPerThreadgroup()),
        )
        .map_err(|e| format!("icb smoke: {e}"))?;

        let cmd: Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> =
            unsafe { self.icb.indirectComputeCommandAtIndex(0) };
        cmd.reset();
        cmd.setComputePipelineState(&self.pipeline);
        if self.bridge == IcbBindBridge::ClassicKernelBuffer {
            unsafe {
                cmd.setKernelBuffer_offset_atIndex(src.metal(), 0, 0);
                cmd.setKernelBuffer_offset_atIndex(dst.metal(), 0, 1);
                cmd.setKernelBuffer_offset_atIndex(self.n_buf.metal(), 0, 2);
            }
        }
        cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
            mtl_size(groups, 1, 1),
            mtl_size(tpt, 1, 1),
        );
        self.src = Some(src.clone());
        self.dst = Some(dst.clone());
        self.encoded = true;
        self.optimized = false;
        Ok(())
    }

    /// Optimize once (GPU) then execute the single ICB command via MTL4 encoder.
    pub fn execute(&mut self, rt: &GpuRuntime) -> Result<(), String> {
        if !self.encoded {
            return Err("icb smoke: encode_copy before execute".into());
        }
        if !std::ptr::eq(&*self.runtime_table, &*rt.metal4.argument_table.table) {
            return Err("icb smoke: execute runtime differs from capture runtime".into());
        }
        let need_opt = !self.optimized;
        let icb = self.icb.clone();
        let pipe = self.pipeline.clone();
        let bridge = self.bridge;
        let src = self
            .src
            .clone()
            .ok_or_else(|| "icb smoke: missing src".to_string())?;
        let dst = self
            .dst
            .clone()
            .ok_or_else(|| "icb smoke: missing dst".to_string())?;
        let n_buf = self.n_buf.clone();

        rt.with_binder(|bnd| {
            if need_opt {
                bnd.optimize_icb(&icb, 0, 1);
            }
            match bridge {
                IcbBindBridge::ClassicKernelBuffer => {
                    // Parent PSO required on MTL4 even with inheritPipelineState=false.
                    // Do NOT latch an argument table — setArgumentTable can override
                    // classic setKernelBuffer freezes in the ICB.
                    bnd.set_pipeline(&pipe);
                    bnd.execute_icb_ex(&icb, 0, 1, false);
                }
                IcbBindBridge::InheritArgTable => {
                    // Bind via MTL4 argument table; ICB inherits at execute time.
                    // Note: still host bind traffic per execute — freezes *dispatch*
                    // (pipeline + grid), not resource slots unless freeze-binds.
                    bnd.set_pipeline(&pipe);
                    bnd.bind_gpu_buf(&src, 0);
                    bnd.bind_gpu_buf(&dst, 1);
                    bnd.bind_gpu_buf(&n_buf, 2);
                    bnd.execute_icb(&icb, 0, 1);
                }
            }
            Ok(())
        })?;
        if need_opt {
            self.optimized = true;
        }
        self.execute_count = self.execute_count.saturating_add(1);
        Ok(())
    }
}

impl Drop for IcbCopySmoke {
    fn drop(&mut self) {
        if let Some(rt) = self.runtime.upgrade() {
            rt.schedule_icb_retirement(self.icb.clone(), Vec::new(), vec![self.pipeline.clone()]);
        }
    }
}

fn copy_count_u32(n: usize, context: &str) -> Result<u32, String> {
    if n == 0 {
        return Err(format!("{context}: n must be > 0"));
    }
    u32::try_from(n).map_err(|_| format!("{context}: n exceeds 32-bit kernel indexing"))
}

fn copy_nbytes(n: usize, context: &str) -> Result<usize, String> {
    copy_count_u32(n, context)?;
    n.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{context}: byte size overflow"))
}

fn pipeline_copy_f32_icb(
    rt: &GpuRuntime,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    // MTL4 encoder + executeCommandsInBuffer needs an MTL4 pipeline with ICB
    // support enabled (classic MTLComputePipelineDescriptor is insufficient).
    let compiler_desc = MTL4CompilerDescriptor::new();
    let compiler = rt
        .device
        .newCompilerWithDescriptor_error(&compiler_desc)
        .map_err(|e| format!("MTL4Compiler: {e}"))?;

    let func_desc = MTL4LibraryFunctionDescriptor::new();
    func_desc.setName(Some(&NSString::from_str("copy_f32")));
    func_desc.setLibrary(Some(&rt.library));

    let pipe_desc = MTL4ComputePipelineDescriptor::new();
    pipe_desc.setComputeFunctionDescriptor(Some(func_desc.as_super()));
    pipe_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

    let pipe = compiler
        .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
        .map_err(|e| format!("ICB MTL4 copy_f32 pipeline: {e}"))?;
    if !pipe.supportIndirectCommandBuffers() {
        return Err("ICB pipeline supportIndirectCommandBuffers=false".into());
    }
    Ok(pipe)
}

/// End-to-end smoke: encode once, execute twice, verify `dst == src`.
///
/// Independent of [`icb_smoke_enabled`] so unit tests can prove the API without
/// mutating global flags for other suites.
pub fn run_copy_f32_smoke(rt: &Arc<GpuRuntime>) -> Result<IcbCopySmoke, String> {
    let n = 64usize;
    let src = rt.alloc_buffer(n * 4)?;
    let dst = rt.alloc_buffer(n * 4)?;
    // SAFETY: `src` and `dst` were just allocated at `n * 4` bytes on this
    // thread and nothing else holds a handle to either, so these writes are
    // unaliased and in bounds for `i < n`. Both are shared-storage buffers with
    // a stable, Metal-aligned `contents()` pointer, and no GPU work has been
    // encoded against them yet — the first `execute` is below.
    unsafe {
        let p = src.metal().contents().as_ptr() as *mut f32;
        for i in 0..n {
            *p.add(i) = (i as f32) + 0.5;
        }
        let q = dst.metal().contents().as_ptr() as *mut f32;
        std::ptr::write_bytes(q as *mut u8, 0, n * 4);
    }

    let mut smoke = IcbCopySmoke::new(rt, n)?;
    smoke.encode_copy(&src, &dst)?;
    // First execute (optimize + run).
    smoke.execute(rt)?;
    rt.synchronize()?;
    verify_copy(&dst, n, "after first execute")?;

    // Second execute — proves ICB reuse without re-encoding.
    // Clear dst first so a no-op execute would fail the check.
    // SAFETY: as above, and `rt.synchronize()` on the line before returned, so
    // the GPU has finished every command touching `dst` and this write cannot
    // race one.
    unsafe {
        let q = dst.metal().contents().as_ptr() as *mut f32;
        std::ptr::write_bytes(q as *mut u8, 0xFF, n * 4);
    }
    smoke.execute(rt)?;
    rt.synchronize()?;
    verify_copy(&dst, n, "after second execute")?;
    if smoke.execute_count() != 2 {
        return Err(format!(
            "icb smoke expected 2 executes, got {}",
            smoke.execute_count()
        ));
    }
    Ok(smoke)
}

fn verify_copy(dst: &GpuBuffer, n: usize, label: &str) -> Result<(), String> {
    // Both call sites pass the `n` that `dst` was allocated from, so this
    // cannot fire today. It is here so the `unsafe` below is justified by a
    // check in this function rather than by reasoning about its callers — the
    // kind of invariant that holds until someone adds a third call site.
    let capacity = dst.nbytes() / std::mem::size_of::<f32>();
    if n > capacity {
        return Err(format!(
            "{label}: verify_copy asked for {n} floats from a buffer holding {capacity}"
        ));
    }
    // SAFETY: `dst` is a shared-storage `GpuBuffer`, so its contents pointer is
    // non-null and valid for `dst.nbytes()` bytes for as long as the buffer
    // lives — which outlives this borrow. `n <= capacity` was just checked, so
    // the slice cannot overrun, and `contents()` is 16-byte aligned by Metal,
    // well past `f32`'s requirement. Every bit pattern is a valid `f32`. The
    // caller synchronizes before verifying; a read racing the GPU would observe
    // a torn value rather than undefined behaviour, and the comparison below
    // would fail rather than the read.
    let out =
        unsafe { std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n) };
    for (i, &v) in out.iter().enumerate() {
        let expect = (i as f32) + 0.5;
        if v != expect {
            return Err(format!(
                "icb smoke mismatch {label} at {i}: got {v} want {expect}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residency_count(rt: &GpuRuntime) -> usize {
        *rt.metal4.residency_count.lock().unwrap()
    }

    #[test]
    fn icb_smoke_flag_default_off() {
        set_icb_smoke(false);
        assert!(!icb_smoke_enabled());
        set_icb_smoke(true);
        assert!(icb_smoke_enabled());
        set_icb_smoke(false);
    }

    #[test]
    fn icb_copy_rejects_counts_that_do_not_fit_the_shader_scalar() {
        let rt = GpuRuntime::new().expect("runtime");
        let err = IcbCopySmoke::new(&rt, u32::MAX as usize + 1)
            .map(|_| ())
            .expect_err("copy_f32 takes a uint count; truncation must be refused");
        assert!(err.contains("32-bit"), "unexpected error: {err}");
    }

    #[test]
    fn icb_copy_rejects_short_buffers_before_encoding() {
        let rt = GpuRuntime::new().expect("runtime");
        let mut smoke = IcbCopySmoke::new(&rt, 8).expect("smoke");
        let short = rt.alloc_buffer(4).expect("short buffer");
        let full = rt
            .alloc_buffer(8 * std::mem::size_of::<f32>())
            .expect("full buffer");
        let err = smoke
            .encode_copy(&short, &full)
            .expect_err("an undersized source must not reach the indirect command");
        assert!(err.contains("too small"), "unexpected error: {err}");
        assert!(!smoke.encoded());
    }

    #[test]
    fn icb_copy_is_encode_once_and_preserves_the_original_recipe() {
        let rt = GpuRuntime::new().expect("runtime");
        let mut smoke = IcbCopySmoke::new(&rt, 8).expect("smoke");
        let first_src = rt.alloc_buffer(8 * 4).expect("first src");
        let first_dst = rt.alloc_buffer(8 * 4).expect("first dst");
        let replacement_src = rt.alloc_buffer(8 * 4).expect("replacement src");
        let replacement_dst = rt.alloc_buffer(8 * 4).expect("replacement dst");
        smoke
            .encode_copy(&first_src, &first_dst)
            .expect("initial encode");

        let err = smoke
            .encode_copy(&replacement_src, &replacement_dst)
            .expect_err("an encode-once ICB must reject native command mutation");
        assert!(err.contains("already encoded"), "unexpected error: {err}");
        assert!(std::sync::Arc::ptr_eq(
            &smoke.src.as_ref().expect("retained src").inner,
            &first_src.inner
        ));
        assert!(std::sync::Arc::ptr_eq(
            &smoke.dst.as_ref().expect("retained dst").inner,
            &first_dst.inner
        ));
    }

    #[test]
    fn icb_copy_rejects_encoding_after_its_runtime_drops() {
        let rt = GpuRuntime::new().expect("runtime");
        let mut smoke = IcbCopySmoke::new(&rt, 8).expect("smoke");
        let src = rt.alloc_buffer(8 * 4).expect("src");
        let dst = rt.alloc_buffer(8 * 4).expect("dst");
        drop(rt);

        let err = smoke
            .encode_copy(&src, &dst)
            .expect_err("a safe API must not mutate an ICB after its runtime is gone");
        assert!(
            err.contains("runtime has been dropped"),
            "unexpected error: {err}"
        );
        assert!(!smoke.encoded());
    }

    #[test]
    fn repeated_icb_create_drop_balances_residency_after_completion() {
        let rt = GpuRuntime::new().expect("runtime");
        let baseline = residency_count(&rt);
        for _ in 0..16 {
            drop(IcbCopySmoke::new(&rt, 8).expect("smoke"));
        }
        // Each live construction registered one Hot scalar and one ICB. Drop
        // queues both, but completion is the only safe removal edge.
        assert_eq!(residency_count(&rt), baseline + 32);
        rt.synchronize().expect("completed-work retirement");
        assert_eq!(residency_count(&rt), baseline);
    }

    #[test]
    fn dropped_icb_and_operands_stay_resident_until_async_work_completes() {
        let rt = GpuRuntime::new().expect("runtime");
        let baseline = residency_count(&rt);
        let src = rt.alloc_buffer(8 * 4).expect("src");
        let dst = rt.alloc_buffer(8 * 4).expect("dst");
        let mut smoke = IcbCopySmoke::new(&rt, 8).expect("smoke");
        smoke.encode_copy(&src, &dst).expect("encode");

        rt.set_async_encode(true).expect("async encode");
        smoke.execute(&rt).expect("enqueue ICB");
        drop(smoke);
        drop(src);
        drop(dst);

        // One ICB, one Hot scalar and two Cold operands remain registered and
        // retained after a non-waiting submission. Immediate removal here can
        // race the queued executeCommandsInBuffer operation.
        rt.commit(false).expect("non-waiting submit");
        assert_eq!(residency_count(&rt), baseline + 4);

        rt.synchronize().expect("wait and retirement drain");
        assert_eq!(residency_count(&rt), baseline);
    }

    #[test]
    fn icb_mini_copy_smoke() {
        let rt = GpuRuntime::new().expect("runtime");
        let smoke = run_copy_f32_smoke(&rt).expect("icb copy_f32 smoke");
        assert!(smoke.encoded());
        assert_eq!(smoke.execute_count(), 2);
        assert_eq!(smoke.bridge(), IcbBindBridge::InheritArgTable);
        let mut stub = crate::IcbReplayStub::new();
        stub.mark_smoke_proven();
        assert!(stub.smoke_proven);
        assert_eq!(stub.phase, crate::IcbStubPhase::SmokeProven);
        eprintln!("icb_mini_copy_smoke: {}", smoke.status_line());
    }
}
