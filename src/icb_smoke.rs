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

use std::sync::{Arc, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand, MTLResourceOptions,
};

use crate::ab_flags::TriFlag;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

static ICB_SMOKE: TriFlag = TriFlag::new();

/// Force ICB-smoke opt-in on/off (tests / harness). Overrides env.
pub fn set_icb_smoke(on: bool) {
    ICB_SMOKE.set(on);
}

/// Opt-in gate for ICB smoke wiring. Default OFF.
pub fn icb_smoke_enabled() -> bool {
    ICB_SMOKE.get(
        &["TESSL_ICB_SMOKE", "METAL_RUNTIME_ICB_SMOKE", "GEMMA_METAL_ICB_SMOKE"],
        || false,
    )
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
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Stable `u32` length bind (ICB / arg-table cannot use bump const-arena).
    n_buf: GpuBuffer,
    n: usize,
    /// Address of the creating runtime, compared against each buffer's owner.
    rt_addr: usize,
    /// For the release on drop; weak so a smoke never keeps its runtime alive.
    rt: Weak<GpuRuntime>,
    bridge: IcbBindBridge,
    /// Retained for classic-bridge re-encode / inherit execute binds.
    src: Option<GpuBuffer>,
    dst: Option<GpuBuffer>,
    encoded: bool,
    optimized: bool,
    execute_count: u64,
    /// Whether the most recent `execute` was suppressed by binder-encode-nop.
    last_execute_was_nop: bool,
}

impl Drop for IcbCopySmoke {
    fn drop(&mut self) {
        crate::decode_icb::release_icb(&self.rt, &self.icb);
    }
}

impl IcbCopySmoke {
    /// Allocate ICB + ICB-capable `copy_f32` pipeline + stable `n` buffer.
    pub fn new(rt: &GpuRuntime, n: usize) -> Result<Self, String> {
        Self::new_with_bridge(rt, n, IcbBindBridge::InheritArgTable)
    }

    pub fn new_with_bridge(rt: &GpuRuntime, n: usize, bridge: IcbBindBridge) -> Result<Self, String> {
        if n == 0 {
            return Err("icb smoke: n must be > 0".into());
        }
        // `n` is bound as a `u32` and sizes `n * 4` byte buffers.
        if u32::try_from(n).is_err() || n.checked_mul(4).is_none() {
            return Err(format!("icb smoke: n = {n} does not fit the kernel's u32 length"));
        }
        let pipeline = crate::decode_icb::pipeline_icb(rt, "copy_f32")?;
        let n_buf = rt.alloc_buffer_hot(4)?;
        // SAFETY: `n_buf` is a 4-byte Hot buffer allocated on the line above
        // with no other handle to it, so this single `u32` write is unaliased
        // and exactly fills it. `contents()` on a shared-storage buffer is
        // Metal-aligned, well past `u32`'s requirement.
        unsafe {
            let p = n_buf.metal().contents().as_ptr() as *mut u32;
            *p = n as u32;
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
            pipeline,
            n_buf,
            n,
            rt_addr: rt as *const GpuRuntime as usize,
            rt: rt.weak_self(),
            bridge,
            src: None,
            dst: None,
            encoded: false,
            optimized: false,
            execute_count: 0,
            last_execute_was_nop: false,
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

    /// Whether the most recent [`Self::execute`] was a no-op because
    /// binder-encode-nop was in force: nothing was encoded and no counter moved.
    pub fn last_execute_was_nop(&self) -> bool {
        self.last_execute_was_nop
    }

    pub fn status_line(&self) -> String {
        format!(
            "icb_copy_smoke n={} bridge={:?} encoded={} optimized={} executes={} cmds=1",
            self.n, self.bridge, self.encoded, self.optimized, self.execute_count
        )
    }

    /// CPU-encode command 0 once (pipeline + dispatch; binds per [`IcbBindBridge`]).
    ///
    /// Refuses, before touching the ICB, a buffer owned by another runtime or
    /// shorter than `n * 4` bytes: the kernel copies `n` floats unchecked.
    ///
    /// Re-encoding rewrites command 0 in place, so the caller must not do it
    /// while an earlier [`Self::execute`] may still be in flight; synchronize
    /// the runtime first.
    pub fn encode_copy(&mut self, src: &GpuBuffer, dst: &GpuBuffer) -> Result<(), String> {
        let need = self.n * 4;
        for (name, b) in [("src", src), ("dst", dst)] {
            if !b.belongs_to_addr(self.rt_addr) {
                return Err(format!("icb smoke: {name} buffer belongs to another runtime"));
            }
            if b.nbytes() < need {
                return Err(format!(
                    "icb smoke: {name} buffer has {} bytes, but copying n = {} floats needs {need}",
                    b.nbytes(),
                    self.n
                ));
            }
        }
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
        let width = self.pipeline.threadExecutionWidth();
        let tpt = width.min(self.n).max(1);
        let groups = self.n.div_ceil(tpt);
        cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
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
        // Under binder-encode-nop `with_binder` skips its body and returns `Ok`,
        // so counting an execute here would report work that never happened.
        // Mirrors `DecodeIcb::execute`: refuse only on a poisoned runtime.
        if crate::decode_icb::binder_encode_nop() {
            rt.ensure_not_poisoned()?;
            self.last_execute_was_nop = true;
            return Ok(());
        }
        self.last_execute_was_nop = false;
        let need_opt = !self.optimized;
        let icb = self.icb.clone();
        let pipe = self.pipeline.clone();
        let bridge = self.bridge;
        let src = self.src.clone().ok_or_else(|| "icb smoke: missing src".to_string())?;
        let dst = self.dst.clone().ok_or_else(|| "icb smoke: missing dst".to_string())?;
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
        return Err(format!("icb smoke expected 2 executes, got {}", smoke.execute_count()));
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
    let out = unsafe { std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n) };
    for (i, &v) in out.iter().enumerate() {
        let expect = (i as f32) + 0.5;
        if v != expect {
            return Err(format!("icb smoke mismatch {label} at {i}: got {v} want {expect}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLResidencySet;

    #[test]
    fn icb_smoke_flag_default_off() {
        set_icb_smoke(false);
        assert!(!icb_smoke_enabled());
        set_icb_smoke(true);
        assert!(icb_smoke_enabled());
        set_icb_smoke(false);
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

    /// Against the pre-fix code `encode_copy` returns `Ok` for every case here.
    #[test]
    fn encode_copy_refuses_undersized_and_foreign_buffers() {
        let rt = GpuRuntime::new().expect("runtime");
        let other = GpuRuntime::new().expect("second runtime");
        let mut smoke = IcbCopySmoke::new(&rt, 1 << 20).expect("smoke");

        let small = rt.alloc_buffer(4).unwrap();
        let big = rt.alloc_buffer(4 << 20).unwrap();
        let err = smoke.encode_copy(&small, &big).unwrap_err();
        assert!(err.contains("src buffer has"), "{err}");
        let err = smoke.encode_copy(&big, &small).unwrap_err();
        assert!(err.contains("dst buffer has"), "{err}");
        assert!(!smoke.encoded(), "a refused encode must leave the ICB untouched");

        let foreign = other.alloc_buffer(4 << 20).unwrap();
        let err = smoke.encode_copy(&big, &foreign).unwrap_err();
        assert!(err.contains("another runtime"), "{err}");
        assert!(!smoke.encoded());

        smoke
            .encode_copy(&big, &big.clone())
            .expect("exactly-sized buffers encode");
        assert!(smoke.encoded());
    }

    /// Under binder-encode-nop `with_binder` skips its body and returns `Ok`;
    /// `execute` must not count that as an execute (or as an optimize).
    ///
    /// Against the pre-fix code `execute_count()` is 1.
    #[test]
    fn a_nop_execute_is_not_counted() {
        let _flags = crate::decode_icb::IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("runtime");
        let src = rt.alloc_buffer(64 * 4).unwrap();
        let dst = rt.alloc_buffer(64 * 4).unwrap();
        let mut smoke = IcbCopySmoke::new(&rt, 64).expect("smoke");
        smoke.encode_copy(&src, &dst).expect("encode");
        {
            let _nop = crate::BinderEncodeNopGuard::enter();
            smoke.execute(&rt).expect("a nop execute stays benign");
        }
        assert!(smoke.last_execute_was_nop());
        assert_eq!(smoke.execute_count(), 0);
        smoke.execute(&rt).expect("a live execute");
        rt.synchronize().unwrap();
        assert!(!smoke.last_execute_was_nop());
        assert_eq!(smoke.execute_count(), 1);
    }

    /// A nop scope on a poisoned runtime reports the poison instead of `Ok`.
    ///
    /// Against the pre-fix code `execute` returns `Ok` and counts.
    #[test]
    fn a_nop_execute_on_a_poisoned_runtime_fails() {
        let _flags = crate::decode_icb::IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("runtime");
        let src = rt.alloc_buffer(64 * 4).unwrap();
        let dst = rt.alloc_buffer(64 * 4).unwrap();
        let mut smoke = IcbCopySmoke::new(&rt, 64).expect("smoke");
        smoke.encode_copy(&src, &dst).expect("encode");
        assert!(rt.with_binder(|_| Err("injected encode failure".into())).is_err());
        let _nop = crate::BinderEncodeNopGuard::enter();
        assert!(
            smoke.execute(&rt).is_err(),
            "poison must not be hidden by the nop scope"
        );
        assert!(rt.with_binder(|_| Ok(())).is_err());
        assert_eq!(smoke.execute_count(), 0);
    }

    /// A dropped smoke leaves the residency set: 100 build-and-drops return the
    /// set to where it started. Against the pre-fix code each leaves its ICB
    /// registered and the count is 100 higher.
    #[test]
    fn dropping_smokes_returns_residency_to_baseline() {
        let rt = GpuRuntime::new().expect("runtime");
        rt.synchronize().unwrap();
        let baseline = rt.metal4.residency.allocationCount();
        for _ in 0..100 {
            let _smoke = IcbCopySmoke::new(&rt, 16).expect("smoke");
        }
        rt.synchronize().unwrap();
        assert_eq!(rt.metal4.residency.allocationCount(), baseline);
    }

    #[test]
    fn new_refuses_n_beyond_u32() {
        let rt = GpuRuntime::new().expect("runtime");
        let err = IcbCopySmoke::new(&rt, u32::MAX as usize + 1)
            .err()
            .expect("n > u32::MAX");
        assert!(err.contains("u32"), "{err}");
    }
}
