//! Mini decode ICB (compute Indirect Command Buffer).
//!
//! Extends [`crate::icb_smoke`] from one `copy_f32` to N ConcurrentDispatch
//! commands. The ICB is built with `inheritBuffers=true`, and execute uses MTL4
//! prebuilt argument tables (`setArgumentTable` x unique tables; sticky adopt
//! skips redundant switches). The classic freeze-binds, range-batch and
//! coarse-range modes were removed: product tok/s lost to the prebuilt path.
//!
//! Wired into [`crate::cb_replay::PingPongCbReplay::try_replay_icb`] so
//! `try_replay` -> `execute_icb` works for an attached mini or E4B Hot graph
//! (session eligibility is host-side). **31B** decode-graph migration remains
//! open. Opt-in; default OFF.

use objc2::rc::Retained;
use std::sync::Weak;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResourceOptions, MTLSize,
};

use crate::ab_flags::{env_truthy, TriFlag};
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::GpuBuffer;

static DECODE_ICB: TriFlag = TriFlag::new();

/// Force DecodeIcb replay opt-in (tests / harness). Overrides env.
pub fn set_decode_icb(on: bool) {
    DECODE_ICB.set(on);
}

/// Opt-in gate for DecodeIcb replay via ping-pong. Default **OFF**.
///
/// Env: `TESSL_DECODE_ICB=1` or `METAL_RUNTIME_DECODE_ICB=1`.
pub fn decode_icb_enabled() -> bool {
    DECODE_ICB.get(&["TESSL_DECODE_ICB", "METAL_RUNTIME_DECODE_ICB"], || false)
}

/// One argument-table bind for a captured / frozen command.
#[derive(Clone)]
pub enum DecodeIcbBind {
    /// Stable `GpuBuffer` (weights, activations, `IcbScalarPool` slots).
    Buf {
        index: usize,
        buf: GpuBuffer,
        byte_offset: usize,
        /// `buf.gpuAddress() + byte_offset` cached at capture (A2 bind-tax cut).
        /// Advisory only: [`DecodeIcb::from_commands`] recomputes it from
        /// `buf` and `byte_offset`, so a value that disagrees is discarded.
        gpu_addr: u64,
    },
    /// Const-arena scalar snapshotted at capture; materialized into a Hot buffer
    /// when building the ICB (frozen value — use `IcbScalarPool` for per-token).
    Immediate { index: usize, bytes: Vec<u8> },
}

#[inline]
fn buf_gpu_addr(buf: &GpuBuffer, byte_offset: usize) -> u64 {
    buf.gpu_address().wrapping_add(byte_offset as u64)
}

/// One frozen compute dispatch (pipeline + grid + bind recipe).
#[derive(Clone)]
pub struct DecodeIcbCommand {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub threadgroups: MTLSize,
    pub threads_per_tg: MTLSize,
    pub binds: Vec<DecodeIcbBind>,
    pub tg_mem: Option<(usize, usize)>,
    /// Insert a Dispatch→Dispatch Device barrier after this cmd on replay.
    ///
    /// Captured from live always-on auto-barriers and explicit binder barriers
    /// (`Binder::barrier`). Replay must honor these instead of forcing always-on for every cmd
    /// (product tok/s regression under shipping hazard skip-auto — D16).
    pub barrier_after: bool,
    /// Binds that reached the encoder but could NOT be recorded on the tape.
    ///
    /// `Binder::bind_resource_id` (and the test-only raw `bind_buf`) take an
    /// `MTLResourceID` / `MTLBuffer` with no owning `GpuBuffer`, so there is
    /// nothing for [`DecodeIcbBind::Buf`] to hold — and holding it is what pins
    /// the operand's `Arc`. When GEMMs bound A, B and C as raw buffers, a
    /// captured GEMM recorded only its scalar immediates: replay would rebind
    /// no operands at all, and the unrecorded buffers were pinned by nothing,
    /// free to be recycled and handed to a new tensor while the tape still
    /// logically referenced them. Owned binds (`bind_tensor`,
    /// `bind_gpu_buf_offset`) are recorded and never counted here.
    ///
    /// Replay does not depend on these binds (the tape is built with
    /// `inheritBuffers=true`, so the live argument table supplies them); the
    /// count is how a caller or test can see that capture was incomplete.
    pub incomplete_binds: usize,
}

/// MTL4 argument-table max buffer bind count (matches runtime table descriptor).
/// Re-export of the one authority on the argument table's size.
///
/// This was a second `= 31` literal sitting beside `runtime`'s. Two constants
/// for one hardware limit is how they drift, and this one is compared against
/// caller-supplied bind indices.
use crate::runtime::ARGUMENT_TABLE_MAX_BUFFERS as ARG_TABLE_SLOTS;

/// Sticky arg-table state for A2: skip `setAddress` when the slot already holds
/// the same GPU address across consecutive tape commands.
///
/// Used when prebuilt per-command tables are disabled; otherwise Buf binds are
/// frozen into dedicated tables at capture and execute only switches tables.
struct StickyArgTable {
    addr: [u64; ARG_TABLE_SLOTS],
    /// Bit i set ⇒ `addr[i]` is live in the table.
    valid: u32,
    /// `setAddress` calls actually issued this execute.
    set_calls: u64,
    /// Binds considered (Buf + Immediate) this execute.
    bind_total: u64,
    /// Buf binds satisfied by adopting a prebuilt table (no `setAddress`).
    prebuilt_elided: u64,
    /// `setArgumentTable` switches this execute (prebuilt path).
    set_table_calls: u64,
}

impl StickyArgTable {
    fn new() -> Self {
        Self {
            addr: [0; ARG_TABLE_SLOTS],
            valid: 0,
            set_calls: 0,
            bind_total: 0,
            prebuilt_elided: 0,
            set_table_calls: 0,
        }
    }

    #[inline]
    fn bind_addr(&mut self, bnd: &mut crate::dispatch::Binder<'_>, gpu_addr: u64, index: usize) {
        self.bind_total = self.bind_total.saturating_add(1);
        if index < ARG_TABLE_SLOTS {
            let bit = 1u32 << index;
            if (self.valid & bit) != 0 && self.addr[index] == gpu_addr {
                return;
            }
            self.addr[index] = gpu_addr;
            self.valid |= bit;
        }
        // SAFETY: `gpu_addr` comes from a `DecodeIcbBind::Buf` whose address
        // `DecodeIcb::from_commands` recomputed from `buf` + a bounds-checked
        // `byte_offset`; the tape holds `buf`, so the allocation outlives the
        // replay, and the runtime's residency set covers it.
        unsafe { bnd.bind_addr(gpu_addr, index) };
        self.set_calls = self.set_calls.saturating_add(1);
    }

    /// Immediate binds always materialize a fresh const-arena address.
    #[inline]
    fn bind_bytes(&mut self, bnd: &mut crate::dispatch::Binder<'_>, bytes: &[u8], index: usize) {
        self.bind_total = self.bind_total.saturating_add(1);
        let addr = bnd.bind_bytes(bytes, index);
        self.set_calls = self.set_calls.saturating_add(1);
        if index < ARG_TABLE_SLOTS {
            let bit = 1u32 << index;
            self.addr[index] = addr;
            self.valid |= bit;
        }
    }

    /// Prefill path: Buf already frozen in `table`; only Immediate needs
    /// `setAddress` (const-arena address changes each execute).
    #[inline]
    fn bind_prebuilt(
        &mut self,
        bnd: &mut crate::dispatch::Binder<'_>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        binds: &[DecodeIcbBind],
    ) {
        if bnd.adopt_argument_table(table) {
            self.set_table_calls = self.set_table_calls.saturating_add(1);
        }
        for b in binds {
            self.bind_total = self.bind_total.saturating_add(1);
            match b {
                DecodeIcbBind::Buf { .. } => {
                    self.prebuilt_elided = self.prebuilt_elided.saturating_add(1);
                }
                DecodeIcbBind::Immediate { index, bytes } => {
                    let addr = bnd.materialize_bytes(bytes);
                    unsafe {
                        table.setAddress_atIndex(addr, *index);
                    }
                    self.set_calls = self.set_calls.saturating_add(1);
                }
            }
        }
    }
}

/// Opt-out: `TESSL_ICB_PREBUILT_TABLES=0` / `METAL_RUNTIME_ICB_PREBUILT_TABLES=0`.
/// Default **ON** — build Buf binds into per-command MTL4 argument tables.
fn prebuilt_tables_enabled() -> bool {
    env_truthy(&["TESSL_ICB_PREBUILT_TABLES", "METAL_RUNTIME_ICB_PREBUILT_TABLES"]).unwrap_or(true)
}

static ICB_EXECUTE: TriFlag = TriFlag::new();

/// Force ICB execution of a replayed tape (tests / harness). Overrides env.
pub fn set_icb_execute(on: bool) {
    ICB_EXECUTE.set(on);
}

/// Execute replayed commands through the ICB (`executeCommandsInBuffer`)
/// instead of dispatching them directly. Default **OFF**. Read once, like the
/// other ICB flags, rather than on every replay.
///
/// Env: `TESSL_ICB_EXECUTE` or `METAL_RUNTIME_ICB_EXECUTE`.
pub fn icb_execute_enabled() -> bool {
    ICB_EXECUTE.get(&["TESSL_ICB_EXECUTE", "METAL_RUNTIME_ICB_EXECUTE"], || false)
}

static ICB_TRIAGE: TriFlag = TriFlag::new();

/// Force decode-ICB triage mode (tests / harness). Overrides env.
pub fn set_icb_triage(on: bool) {
    ICB_TRIAGE.set(on);
}

/// Triage mode: replay one command per binder scope with a sync after each,
/// to localize the first diverging dispatch. Default **OFF**. Read once.
///
/// Env: `TESSL_ICB_TRIAGE`.
pub fn icb_triage_enabled() -> bool {
    ICB_TRIAGE.get(&["TESSL_ICB_TRIAGE"], || false)
}

/// Fingerprint of Buf `(index, gpu_addr)` pairs for prebuilt-table dedup.
fn buf_bind_fingerprint(binds: &[DecodeIcbBind]) -> u64 {
    // FNV-1a 64 — stable, cheap, good enough for capture-time table sharing.
    let mut slots: Vec<(u16, u64)> = Vec::with_capacity(binds.len());
    for b in binds {
        if let DecodeIcbBind::Buf { index, gpu_addr, .. } = b {
            if *index < ARG_TABLE_SLOTS {
                slots.push((*index as u16, *gpu_addr));
            }
        }
    }
    slots.sort_unstable_by_key(|s| s.0);
    let mut h = 0xcbf29ce484222325u64;
    for &(idx, addr) in &slots {
        h ^= idx as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= addr;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Multi-command decode ICB (inheritBuffers + arg-table execute).
pub struct DecodeIcb {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    /// For the release on drop; weak so a tape never keeps its runtime alive.
    rt: Weak<GpuRuntime>,
    commands: Vec<DecodeIcbCommand>,
    /// Per-command MTL4 argument tables with Buf addresses frozen at capture
    /// (A2 residual: execute switches tables instead of re-`setAddress`).
    /// Empty when prebuilt tables are disabled.
    prebuilt_tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    /// Unique prebuilt tables after fingerprint dedup (≤ `prebuilt_tables.len()`).
    unique_prebuilt_tables: usize,
    encoded: bool,
    optimized: bool,
    execute_count: u64,
    /// Whether the most recent `execute` was suppressed by binder-encode-nop.
    ///
    /// Distinguishes "a replay ran" from "a replay was asked for while encoding
    /// was disabled", which the counters alone cannot express.
    last_execute_was_nop: bool,
    /// True when built from a Binder capture of the mini layer/head graph
    /// (not the `mini_copy_chain` smoke). Enables host-encode skip on replay.
    layer_graph: bool,
    /// Triage probe (`TESSL_ICB_TRIAGE=1`): buffer + f32 count sampled
    /// after every replayed command to localize the first diverging dispatch.
    triage_probe: Option<(crate::tensor::GpuBuffer, usize)>,
    /// Last execute: total binds considered / `setAddress` issued (A2 sticky).
    last_bind_total: u64,
    last_set_address_calls: u64,
    /// Last execute: `setArgumentTable` switches (prebuilt path).
    last_set_argument_table_calls: u64,
    /// Last execute: Buf binds elided via prebuilt tables.
    last_prebuilt_elided: u64,
    /// Last execute: `executeCommandsInBuffer` calls (ICB execute).
    last_execute_icb_calls: u64,
    /// Last execute: total ICB cmds covered by those calls.
    last_execute_icb_cmds: u64,
    /// Capture-time: Buf binds that would be sticky-skippable across cmds.
    sticky_skippable_binds: u64,
    total_buf_binds: u64,
    /// Every command's pipeline supports ICBs. Pipelines are fixed once the
    /// tape is built, so this is computed once instead of probing each
    /// pipeline (an Objective-C send) on every replay.
    all_icb: bool,
}

/// One argument table per encoded command, plus the number of distinct tables
/// actually built (commands with identical binds share one).
type PrebuiltTables = (Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>, usize);

/// Hand a dropped tape's ICB to the runtime's deferred release: it leaves the
/// residency set (and is freed) once every command already submitted or encoded
/// has completed, because MTL4 command buffers do not retain it. Nothing to do
/// when the runtime is already gone: its own drop frees the residency set.
pub(crate) fn release_icb(rt: &Weak<GpuRuntime>, icb: &Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>) {
    if let Some(rt) = rt.upgrade() {
        rt.schedule_release(ProtocolObject::<dyn MTLAllocation>::from_retained(icb.clone()), true);
    }
}

impl Drop for DecodeIcb {
    fn drop(&mut self) {
        release_icb(&self.rt, &self.icb);
    }
}

impl DecodeIcb {
    pub fn from_commands(rt: &GpuRuntime, commands: Vec<DecodeIcbCommand>) -> Result<Self, String> {
        if commands.is_empty() {
            return Err("DecodeIcb: empty command list".into());
        }
        // Reject bind indices the argument table cannot hold.
        //
        // The prebuilt path filters `index < ARG_TABLE_SLOTS` and
        // silently drops anything above, while `Binder::bind_addr` correctly
        // errors on the same index — so a tape could be built with `index: 40`,
        // return `Ok`, and replay with that operand simply absent. Worse,
        // `buf_bind_fingerprint` applies the same filter, so two commands
        // differing *only* in an out-of-range bind hash identically and share
        // one prebuilt argument table.
        //
        // The same pass makes the tape self-consistent, because `Buf`'s fields
        // are public and a caller can build one that disagrees with itself:
        // `gpu_addr` is recomputed from `buf` + `byte_offset` (whatever the
        // caller supplied is discarded), `byte_offset` must lie inside the
        // buffer, the buffer must belong to `rt`, and `tg_mem` must name a
        // slot the argument table has. Without this, safe code could replay a
        // forged address or a past-the-end offset into a kernel.
        let mut commands = commands;
        for (i, cmd) in commands.iter_mut().enumerate() {
            for b in cmd.binds.iter_mut() {
                match b {
                    DecodeIcbBind::Buf {
                        index,
                        buf,
                        byte_offset,
                        gpu_addr,
                    } => {
                        if *index >= ARG_TABLE_SLOTS {
                            return Err(format!(
                                "DecodeIcb: command {i} binds index {index}, but the argument table \
                                 has {ARG_TABLE_SLOTS} slots. Encoding would drop it silently.",
                                index = *index
                            ));
                        }
                        if !buf.belongs_to(rt) {
                            return Err(format!(
                                "DecodeIcb: command {i} binds a buffer at index {index} that belongs to another runtime",
                                index = *index
                            ));
                        }
                        if *byte_offset >= buf.nbytes() {
                            return Err(format!(
                                "DecodeIcb: command {i} binds index {index} at byte offset {byte_offset}, \
                                 outside its {} byte buffer",
                                buf.nbytes(),
                                index = *index,
                                byte_offset = *byte_offset
                            ));
                        }
                        *gpu_addr = buf.gpu_address().checked_add(*byte_offset as u64).ok_or_else(|| {
                            format!(
                                "DecodeIcb: command {i} bind at index {} overflows the GPU address space",
                                *index
                            )
                        })?;
                    }
                    DecodeIcbBind::Immediate { index, .. } => {
                        if *index >= ARG_TABLE_SLOTS {
                            return Err(format!(
                                "DecodeIcb: command {i} binds index {index}, but the argument table \
                                 has {ARG_TABLE_SLOTS} slots. Encoding would drop it silently.",
                                index = *index
                            ));
                        }
                    }
                }
            }
            if let Some((tg_idx, _)) = cmd.tg_mem {
                if tg_idx >= ARG_TABLE_SLOTS {
                    return Err(format!(
                        "DecodeIcb: command {i} sets threadgroup memory at slot {tg_idx}, but the \
                         argument table has {ARG_TABLE_SLOTS} slots"
                    ));
                }
            }
        }
        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        desc.setInheritPipelineState(false);
        // Working bridge on MacOSX26: inherit arg-table at execute time.
        desc.setInheritBuffers(true);
        desc.setMaxKernelBufferBindCount(0);

        let n = commands.len();
        let icb = unsafe {
            rt.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    n,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| "newIndirectCommandBuffer failed (DecodeIcb)".to_string())?;

        rt.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*icb));

        // Keep Immediate binds as snapshotted bytes. Re-pack via
        // `bind_bytes` at execute. Prefer `IcbScalarPool` Buf binds for
        // per-token scalars.
        let layer_graph = commands.len() >= Self::MIN_LAYER_GRAPH_COMMANDS;
        let (total_buf_binds, sticky_skippable_binds) = Self::analyze_sticky_buf_binds(&commands);
        let (prebuilt_tables, unique_prebuilt_tables) = if prebuilt_tables_enabled() {
            Self::build_prebuilt_tables(rt, &commands)?
        } else {
            (Vec::new(), 0)
        };
        let all_icb = commands.iter().all(|c| c.pipeline.supportIndirectCommandBuffers());
        let mut this = Self {
            icb,
            rt: rt.weak_self(),
            commands,
            prebuilt_tables,
            unique_prebuilt_tables,
            encoded: false,
            optimized: false,
            execute_count: 0,
            last_execute_was_nop: false,
            layer_graph,
            triage_probe: None,
            last_bind_total: 0,
            last_set_address_calls: 0,
            last_set_argument_table_calls: 0,
            last_prebuilt_elided: 0,
            last_execute_icb_calls: 0,
            last_execute_icb_cmds: 0,
            sticky_skippable_binds,
            total_buf_binds,
            all_icb,
        };
        this.encode_cpu()?;
        Ok(this)
    }

    /// Build Buf `gpu_addr`s into MTL4 argument tables; dedup by fingerprint
    /// so sticky adopt can skip redundant `setArgumentTable` switches.
    fn build_prebuilt_tables(rt: &GpuRuntime, commands: &[DecodeIcbCommand]) -> Result<PrebuiltTables, String> {
        let mut unique: Vec<(u64, Retained<ProtocolObject<dyn MTL4ArgumentTable>>)> = Vec::new();
        let mut tables = Vec::with_capacity(commands.len());
        for cmd in commands {
            let fp = buf_bind_fingerprint(&cmd.binds);
            if let Some((_, t)) = unique.iter().find(|(f, _)| *f == fp) {
                tables.push(t.clone());
                continue;
            }
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(ARG_TABLE_SLOTS);
            desc.setMaxTextureBindCount(16);
            desc.setMaxSamplerStateBindCount(8);
            let table = rt
                .device
                .newArgumentTableWithDescriptor_error(&desc)
                .map_err(|e| format!("DecodeIcb prebuilt newArgumentTable: {e}"))?;
            for b in &cmd.binds {
                if let DecodeIcbBind::Buf { index, gpu_addr, .. } = b {
                    unsafe {
                        table.setAddress_atIndex(*gpu_addr, *index);
                    }
                }
            }
            unique.push((fp, table.clone()));
            tables.push(table);
        }
        Ok((tables, unique.len()))
    }

    /// Count Buf binds and how many would skip `setAddress` under sticky replay.
    fn analyze_sticky_buf_binds(commands: &[DecodeIcbCommand]) -> (u64, u64) {
        let mut addr = [0u64; ARG_TABLE_SLOTS];
        let mut valid = 0u32;
        let mut total = 0u64;
        let mut skippable = 0u64;
        for cmd in commands {
            for b in &cmd.binds {
                match b {
                    DecodeIcbBind::Buf { index, gpu_addr, .. } => {
                        total = total.saturating_add(1);
                        if *index < ARG_TABLE_SLOTS {
                            let bit = 1u32 << *index;
                            if (valid & bit) != 0 && addr[*index] == *gpu_addr {
                                skippable = skippable.saturating_add(1);
                            } else {
                                addr[*index] = *gpu_addr;
                                valid |= bit;
                            }
                        }
                    }
                    DecodeIcbBind::Immediate { index, .. } => {
                        // Immediate always sets; invalidate sticky at index.
                        if *index < ARG_TABLE_SLOTS {
                            valid &= !(1u32 << *index);
                        }
                    }
                }
            }
        }
        (total, skippable)
    }

    /// Attach a triage probe: after each replayed command (when
    /// `TESSL_ICB_TRIAGE=1`), synchronize and print the probe buffer's
    /// absmean. One M5 run then pinpoints the first command whose output
    /// diverges from the live-encode trace. Costs a sync per command — never
    /// leave on for measurement.
    pub fn set_triage_probe(&mut self, buf: crate::tensor::GpuBuffer, n_f32: usize) {
        self.triage_probe = Some((buf, n_f32));
    }

    /// Minimum captured dispatches to treat as a mini layer/head graph (not copy-chain).
    pub const MIN_LAYER_GRAPH_COMMANDS: usize = 8;

    /// Convenience: two-hop `copy_f32` mini graph (a→b→c) for try_replay tests.
    pub fn mini_copy_chain(rt: &GpuRuntime, n: usize) -> Result<(Self, GpuBuffer), String> {
        const WHAT: &str = "DecodeIcb::mini_copy_chain";
        if n == 0 {
            return Err(format!("{WHAT}: n > 0"));
        }
        let n32 = u32::try_from(n).map_err(|_| format!("{WHAT}: n = {n} does not fit a u32"))?;
        let nbytes = n
            .checked_mul(4)
            .ok_or_else(|| format!("{WHAT}: n = {n} overflows n * 4 bytes"))?;
        // Hot/shared staging matches layer-graph residency.
        let a = rt.alloc_buffer_hot(nbytes)?;
        let b = rt.alloc_buffer_hot(nbytes)?;
        let c = rt.alloc_buffer_hot(nbytes)?;
        let n_buf = rt.alloc_buffer_hot(4)?;
        // One host-access lease for the whole initialisation: a live host
        // mapping (or an active encoder or submit) makes this fail as "busy"
        // instead of racing the writes below.
        {
            let _access = rt.acquire_access().map_err(|e| format!("{WHAT}: {e}"))?;
            // SAFETY: `_access` is held until the end of this block, so no
            // other host mapping, encoder or submit exists. `a`, `b`, `c` were
            // each allocated at `n * 4` bytes and `n_buf` at 4 just above and are
            // not yet shared with anything, so these writes are unaliased and in
            // bounds. They are Hot (shared-storage) buffers, so `contents()` is
            // a valid, Metal-aligned host pointer. No command referencing them
            // has been encoded yet.
            unsafe {
                n_buf.write_u32_unsubmitted(&[n32]);
                let p = a.metal().contents().as_ptr() as *mut f32;
                for i in 0..n {
                    *p.add(i) = (i as f32) + 1.0;
                }
                let zb = b.metal().contents().as_ptr() as *mut u8;
                let zc = c.metal().contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(zb, 0, nbytes);
                std::ptr::write_bytes(zc, 0, nbytes);
            }
        }
        let pipe = pipeline_icb(rt, "copy_f32")?;
        let tpt = pipe.threadExecutionWidth().min(n).max(1);
        let groups = n.div_ceil(tpt);
        let tg = mtl_size(groups, 1, 1);
        let tptg = mtl_size(tpt, 1, 1);
        let buf = |index, buf: &GpuBuffer| DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset: 0,
            gpu_addr: buf_gpu_addr(buf, 0),
        };
        let cmds = vec![
            DecodeIcbCommand {
                pipeline: pipe.clone(),
                threadgroups: tg,
                threads_per_tg: tptg,
                binds: vec![buf(0, &a), buf(1, &b), buf(2, &n_buf)],
                tg_mem: None,
                // a→b must drain before b→c (no live always-on during tape execute).
                barrier_after: true,
                // Built from GpuBuffers directly, so nothing is unrecordable.
                incomplete_binds: 0,
            },
            DecodeIcbCommand {
                pipeline: pipe,
                threadgroups: tg,
                threads_per_tg: tptg,
                binds: vec![buf(0, &b), buf(1, &c), buf(2, &n_buf)],
                tg_mem: None,
                barrier_after: false,
                incomplete_binds: 0,
            },
        ];
        let dec = Self::from_commands(rt, cmds)?;
        Ok((dec, c))
    }

    fn encode_cpu(&mut self) -> Result<(), String> {
        // Tape-only / direct-dispatch replay: allow non-ICB pipelines (default path).
        if !self.all_icb {
            self.encoded = true;
            self.optimized = true; // no ICB range to optimize
            return Ok(());
        }
        for (i, cmd) in self.commands.iter().enumerate() {
            let icmd: Retained<ProtocolObject<dyn MTLIndirectComputeCommand>> =
                unsafe { self.icb.indirectComputeCommandAtIndex(i) };
            icmd.reset();
            icmd.setComputePipelineState(&cmd.pipeline);
            icmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(cmd.threadgroups, cmd.threads_per_tg);
        }
        self.encoded = true;
        self.optimized = false;
        Ok(())
    }

    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// How many tape cmds have a captured `barrier_after`.
    pub fn barrier_after_count(&self) -> usize {
        self.commands.iter().filter(|c| c.barrier_after).count()
    }

    pub fn encoded(&self) -> bool {
        self.encoded
    }

    /// Whether the most recent [`DecodeIcb::execute`] was a no-op because
    /// binder-encode-nop was in force.
    pub fn last_execute_was_nop(&self) -> bool {
        self.last_execute_was_nop
    }

    pub fn execute_count(&self) -> u64 {
        self.execute_count
    }

    /// Captured mini layer/head graph (skips host encode on replay).
    pub fn is_layer_graph(&self) -> bool {
        self.layer_graph
    }

    pub fn status_line(&self) -> String {
        format!(
            "decode_icb cmds={} layer_graph={} encoded={} optimized={} executes={} \
             sticky_buf={}/{} prebuilt={}/{} barriers={} \
             last_setAddress={}/{} last_setArgTable={} elided={} last_execute_icb={}/{}",
            self.commands.len(),
            self.layer_graph,
            self.encoded,
            self.optimized,
            self.execute_count,
            self.sticky_skippable_binds,
            self.total_buf_binds,
            self.prebuilt_tables.len(),
            self.unique_prebuilt_tables,
            self.barrier_after_count(),
            self.last_set_address_calls,
            self.last_bind_total,
            self.last_set_argument_table_calls,
            self.last_prebuilt_elided,
            self.last_execute_icb_calls,
            self.last_execute_icb_cmds,
        )
    }

    /// Unique prebuilt argument tables after fingerprint dedup.
    pub fn unique_prebuilt_table_count(&self) -> usize {
        self.unique_prebuilt_tables
    }

    /// Capture-time Buf binds that sticky replay can skip (`setAddress`).
    pub fn sticky_skippable_binds(&self) -> u64 {
        self.sticky_skippable_binds
    }

    pub fn total_buf_binds(&self) -> u64 {
        self.total_buf_binds
    }

    /// Number of per-command prebuilt argument tables (0 if disabled).
    pub fn prebuilt_table_count(&self) -> usize {
        self.prebuilt_tables.len()
    }

    /// Last execute: binds considered vs `setAddress` calls issued.
    pub fn last_set_address_stats(&self) -> (u64, u64) {
        (self.last_set_address_calls, self.last_bind_total)
    }

    /// Last execute: `setArgumentTable` switches + Buf binds elided by prebuilt.
    pub fn last_prebuilt_stats(&self) -> (u64, u64) {
        (self.last_set_argument_table_calls, self.last_prebuilt_elided)
    }

    /// Last execute: `executeCommandsInBuffer` call count + cmds covered.
    pub fn last_execute_icb_stats(&self) -> (u64, u64) {
        (self.last_execute_icb_calls, self.last_execute_icb_cmds)
    }

    /// Replay each captured command with arg-table rebinds + recorded barriers.
    ///
    /// **Default (tape honesty):** direct `dispatch` with the Binder-tape
    /// pipeline+grid+binds. Opt-in true ICB execute via `TESSL_ICB_EXECUTE=1`
    /// (requires ICB-capable pipelines).
    ///
    /// **A2 residual (v0.5.4):** default builds Buf binds into dedup'd MTL4
    /// argument tables; execute switches `setArgumentTable` only when the table
    /// pointer changes (sticky adopt). Opt-out `TESSL_ICB_PREBUILT_TABLES=0`
    /// falls back to sticky `setAddress`.
    ///
    /// **Barriers (v0.5.6):** capture records `barrier_after` from always-on
    /// auto-barriers and explicit RAW `Binder::barrier` calls. Execute skips
    /// auto-barriers and replays those markers — do not force always-on for
    /// every cmd (that regressed shipping-hazard product tok/s).
    pub fn execute(&mut self, rt: &GpuRuntime) -> Result<(), String> {
        if !self.encoded {
            return Err("DecodeIcb: encode before execute".into());
        }
        // Under binder-encode-nop every `with_binder` body is skipped, so this
        // whole call is a no-op: no dispatch is encoded and the destination
        // buffers keep whatever they held. It used to return `Ok` while
        // incrementing `execute_count` and setting `optimized` — reporting a
        // completed, optimized replay for work that provably did not happen,
        // which `cb_replay` then counted as a successful tape replay.
        //
        // Control flow is deliberately unchanged (callers hold this guard
        // around encode loops and expect a benign no-op); only the bookkeeping
        // is corrected, and the fact is recorded so a caller can tell.
        if crate::decode_icb::binder_encode_nop() {
            // A poisoned runtime fails here as it would on a live replay.
            rt.ensure_not_poisoned()?;
            self.last_execute_was_nop = true;
            return Ok(());
        }
        self.last_execute_was_nop = false;
        let use_icb_exec = icb_execute_enabled();
        let need_opt = !self.optimized;
        let icb = self.icb.clone();
        let n = self.commands.len() as u64;
        let use_prebuilt = !self.prebuilt_tables.is_empty() && self.prebuilt_tables.len() == self.commands.len();

        // ICB execution cannot run a command whose pipeline was not built with
        // `supportIndirectCommandBuffers`. On a mixed tape `encode_cpu` returns
        // early without encoding anything — while still setting `encoded` and
        // `optimized` — and `execute` then decides per command, running the
        // ICB-capable ones and dropping the rest. Measured through the public
        // API: the surviving command's destination stayed [0,0,0,0] while this
        // returned Ok and reported last_execute_icb=1/1.
        //
        // Dropping half a tape is not a degraded mode, it is a wrong answer, so
        // it is refused.
        if use_icb_exec && !self.all_icb {
            return Err(format!(
                "DecodeIcb: ICB execution requested for a tape whose {} command(s) are not \
                 all ICB-capable. Those commands would be silently dropped. Capture with \
                 ICB-capable pipelines (TESSL_ICB_PIPELINES=1) or leave ICB execute off.",
                self.commands.len()
            ));
        }

        // `all_icb` holds here: the check above returned otherwise.
        if use_icb_exec && need_opt {
            rt.with_binder(|bnd| {
                bnd.optimize_icb(&icb, 0, n);
                Ok(())
            })?;
        }
        // Skip auto-barriers during tape encode; replay captured `barrier_after`
        // markers instead. Forcing always-on here inflated E4B barriers ~364→599
        // and regressed product tok/s despite lower host encode_us (D16).
        // Scope-local via with_binder_barriers — flipping the process-global
        // flag here would drop the trailing auto barrier of ops other threads
        // encode concurrently.
        let triage = icb_triage_enabled();
        let mut sticky = StickyArgTable::new();
        let mut execute_icb_calls = 0u64;
        let mut execute_icb_cmds = 0u64;
        let exec_result: Result<(), String> = if triage {
            // Per-cmd binder + sync so probe samples localize the first diverge.
            // Sticky/prebuilt resets each with_binder (new encoder scope).
            let total = self.commands.len();
            for i in 0..total {
                let cmd = &self.commands[i];
                rt.with_binder_barriers(Some(true), |bnd| {
                    let mut s = StickyArgTable::new();
                    // Eager `then_some(tables[i])` panics when the tables are disabled.
                    let prebuilt = if use_prebuilt {
                        Some(self.prebuilt_tables[i].as_ref())
                    } else {
                        None
                    };
                    let (icb_n, icb_cmds) = Self::encode_cmd(bnd, cmd, &icb, i, use_icb_exec, prebuilt, &mut s);
                    execute_icb_calls = execute_icb_calls.saturating_add(icb_n);
                    execute_icb_cmds = execute_icb_cmds.saturating_add(icb_cmds);
                    sticky.bind_total = sticky.bind_total.saturating_add(s.bind_total);
                    sticky.set_calls = sticky.set_calls.saturating_add(s.set_calls);
                    sticky.prebuilt_elided = sticky.prebuilt_elided.saturating_add(s.prebuilt_elided);
                    sticky.set_table_calls = sticky.set_table_calls.saturating_add(s.set_table_calls);
                    Ok(())
                })?;
                if let Some((probe, n)) = self.triage_probe.as_ref() {
                    rt.synchronize()?;
                    let host = probe.try_read_f32()?;
                    let take = (*n).min(host.len());
                    let absmean = if take > 0 {
                        host[..take].iter().map(|v| v.abs()).sum::<f32>() / take as f32
                    } else {
                        0.0
                    };
                    eprintln!("[icb_triage] cmd={i}/{total} probe_absmean={absmean:.6}");
                }
            }
            Ok(())
        } else {
            // Pack all tape cmds into one binder scope (A2 sticky / prebuilt).
            let tables = &self.prebuilt_tables;
            rt.with_binder_barriers(Some(true), |bnd| {
                for (i, cmd) in self.commands.iter().enumerate() {
                    let prebuilt = if use_prebuilt { Some(tables[i].as_ref()) } else { None };
                    let (icb_n, icb_cmds) = Self::encode_cmd(bnd, cmd, &icb, i, use_icb_exec, prebuilt, &mut sticky);
                    execute_icb_calls = execute_icb_calls.saturating_add(icb_n);
                    execute_icb_cmds = execute_icb_cmds.saturating_add(icb_cmds);
                    // A capture records every barrier the live encode emitted,
                    // including the lazy hazard-mode ones, so an unmarked edge
                    // inside the tape is one the capture did not need. The
                    // binder would otherwise re-add it (`hazard_pending` is set
                    // by every dispatch under skip-auto), making replay emit
                    // N-1 barriers whatever the markers say. The last command's
                    // pending hazard stays, to order it before whatever the
                    // runtime encodes next.
                    if !cmd.barrier_after && i + 1 < self.commands.len() {
                        bnd.forgo_pending_barrier();
                    }
                }
                Ok(())
            })
        };
        exec_result?;
        self.last_bind_total = sticky.bind_total;
        self.last_set_address_calls = sticky.set_calls;
        self.last_set_argument_table_calls = sticky.set_table_calls;
        self.last_prebuilt_elided = sticky.prebuilt_elided;
        self.last_execute_icb_calls = execute_icb_calls;
        self.last_execute_icb_cmds = execute_icb_cmds;
        if use_icb_exec && need_opt {
            self.optimized = true;
        } else if !use_icb_exec {
            self.optimized = true; // tape-only; no ICB optimize range
        }
        self.execute_count = self.execute_count.saturating_add(1);
        Ok(())
    }

    /// Encode one tape cmd. Returns `(execute_icb_calls, cmds_covered)`.
    #[inline]
    fn encode_cmd(
        bnd: &mut crate::dispatch::Binder<'_>,
        cmd: &DecodeIcbCommand,
        icb: &Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
        i: usize,
        use_icb_exec: bool,
        prebuilt: Option<&ProtocolObject<dyn MTL4ArgumentTable>>,
        sticky: &mut StickyArgTable,
    ) -> (u64, u64) {
        let mut icb_calls = 0u64;
        let mut icb_cmds = 0u64;
        bnd.set_pipeline(&cmd.pipeline);
        if let Some(table) = prebuilt {
            sticky.bind_prebuilt(bnd, table, &cmd.binds);
        } else {
            for b in &cmd.binds {
                match b {
                    DecodeIcbBind::Buf { index, gpu_addr, .. } => {
                        sticky.bind_addr(bnd, *gpu_addr, *index);
                    }
                    DecodeIcbBind::Immediate { index, bytes } => {
                        sticky.bind_bytes(bnd, bytes, *index);
                    }
                }
            }
        }
        if let Some((tg_idx, len)) = cmd.tg_mem {
            bnd.set_threadgroup_memory(tg_idx, len);
        }
        // `execute` only sets `use_icb_exec` once every command's pipeline
        // is ICB-capable (`DecodeIcb::all_icb`), so no per-command probe.
        if use_icb_exec {
            bnd.execute_icb(icb, i as u64, 1);
            icb_calls = 1;
            icb_cmds = 1;
        } else {
            let _ = (icb, i);
            bnd.dispatch(cmd.threadgroups, cmd.threads_per_tg);
        }
        if cmd.barrier_after {
            bnd.barrier();
        }
        (icb_calls, icb_cmds)
    }
}

/// The ICB-capable pipeline for `fn_name`, from the runtime's pipeline cache
/// (compiled on first use), whatever [`icb_pipelines_enabled`] says.
pub fn pipeline_icb(
    rt: &GpuRuntime,
    fn_name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    rt.pipeline_for_mode(fn_name, true)
}

// --- Capture tape (thread-local) --------------------------------------------

thread_local! {
    static CAPTURE: std::cell::RefCell<Option<DecodeIcbCapture>> = const { std::cell::RefCell::new(None) };
    /// Mirrors `CAPTURE.is_some()`. [`decode_icb_capture_active`] runs a dozen
    /// times per dispatch; a `Cell<bool>` has no destructor and no borrow
    /// flag, so the read is a plain thread-local load. Written only next to
    /// the writes of `CAPTURE` in begin / end / take.
    static CAPTURE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Host-side recording of dispatches for later [`DecodeIcb::from_commands`].
#[derive(Default)]
pub struct DecodeIcbCapture {
    pub commands: Vec<DecodeIcbCommand>,
    current_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    current_binds: Vec<DecodeIcbBind>,
    current_tg_mem: Option<(usize, usize)>,
    /// Binds seen since the last `note_pipeline` that could not be recorded.
    unrecordable_binds: usize,
    /// How many times `begin_decode_icb_capture` was called while this capture
    /// was already live. Non-zero means a previous capture was never taken.
    pub nested_begins: usize,
    /// The last scope ended with an unbarriered dispatch. See
    /// [`capture_take_carried_hazard`].
    carried_hazard: bool,
}

impl DecodeIcbCapture {
    pub fn note_pipeline(&mut self, p: Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
        self.current_pipeline = Some(p);
        self.current_binds.clear();
        self.current_tg_mem = None;
        self.unrecordable_binds = 0;
    }

    /// Note a bind that reached the encoder but carries no owning `GpuBuffer`.
    pub fn note_unrecordable_bind(&mut self) {
        self.unrecordable_binds += 1;
    }

    pub fn note_bind(&mut self, index: usize, buf: &GpuBuffer, byte_offset: usize) {
        let bind = DecodeIcbBind::Buf {
            index,
            buf: buf.clone(),
            byte_offset,
            gpu_addr: buf_gpu_addr(buf, byte_offset),
        };
        if let Some(slot) = self.current_binds.iter_mut().find(|b| match b {
            DecodeIcbBind::Buf { index: i, .. } | DecodeIcbBind::Immediate { index: i, .. } => *i == index,
        }) {
            *slot = bind;
        } else {
            self.current_binds.push(bind);
        }
    }

    pub fn note_immediate(&mut self, index: usize, bytes: &[u8]) {
        let bind = DecodeIcbBind::Immediate {
            index,
            bytes: bytes.to_vec(),
        };
        if let Some(slot) = self.current_binds.iter_mut().find(|b| match b {
            DecodeIcbBind::Buf { index: i, .. } | DecodeIcbBind::Immediate { index: i, .. } => *i == index,
        }) {
            *slot = bind;
        } else {
            self.current_binds.push(bind);
        }
    }

    pub fn note_tg_mem(&mut self, index: usize, length: usize) {
        self.current_tg_mem = Some((index, length));
    }

    pub fn note_dispatch(&mut self, threadgroups: MTLSize, threads_per_tg: MTLSize) {
        let Some(pipeline) = self.current_pipeline.clone() else {
            return;
        };
        // Clone rather than take. The argument table is a property of the
        // encoder, not of a single dispatch: under one `set_pipeline`, a second
        // dispatch inherits every bind the first one used. Taking them recorded
        // the second dispatch with zero binds — measured as
        // `cmd0: binds=3, cmd1: binds=0` for one pipeline and two dispatches,
        // which is exactly the shape of the split-K loop. That loop survived
        // only because it happens to re-bind all seven scalars every iteration.
        //
        // `current_binds` is already a slot table keyed by index, with
        // last-write-wins in `note_bind` / `note_immediate`, so carrying it
        // forward models the real argument table rather than accumulating
        // duplicates.
        self.commands.push(DecodeIcbCommand {
            pipeline,
            threadgroups,
            threads_per_tg,
            binds: self.current_binds.clone(),
            tg_mem: self.current_tg_mem,
            barrier_after: false,
            incomplete_binds: self.unrecordable_binds,
        });
    }

    /// Mark the most recent command as needing a post-dispatch Device barrier.
    pub fn note_barrier_after(&mut self) {
        if let Some(cmd) = self.commands.last_mut() {
            cmd.barrier_after = true;
        }
    }
}

/// Begin recording dispatches on this thread (Binder hooks).
pub fn begin_decode_icb_capture() {
    CAPTURE.with(|c| {
        let mut slot = c.borrow_mut();
        if let Some(existing) = slot.as_mut() {
            // Re-entry. There is no RAII guard on this API, so an early `?`
            // between begin and take leaves capture live for the rest of the
            // thread's life — growing without bound and holding a `GpuBuffer`
            // clone for every bind, which keeps those buffers out of the
            // freelist and pinned in the residency set forever.
            //
            // The first tape is kept rather than discarded: it is the one with
            // the recorded work, and throwing it away is how the leak stayed
            // invisible. The count rides along so `take` can report it instead
            // of handing back a tape that silently spans two capture attempts.
            existing.nested_begins += 1;
            return;
        }
        *slot = Some(DecodeIcbCapture::default());
        CAPTURE_ACTIVE.with(|a| a.set(true));
    });
}

/// Abandon an in-progress capture.
///
/// The counterpart `begin`/`take` pair has no RAII guard, so a caller that
/// bails between them leaks the tape. Call this on the error path.
pub fn end_decode_icb_capture() {
    CAPTURE.with(|c| {
        *c.borrow_mut() = None;
    });
    CAPTURE_ACTIVE.with(|a| a.set(false));
}

/// Finish capture; returns recorded commands (may be empty).
pub fn take_decode_icb_capture() -> Option<DecodeIcbCapture> {
    let taken = CAPTURE.with(|c| c.borrow_mut().take());
    CAPTURE_ACTIVE.with(|a| a.set(false));
    taken
}

#[inline]
pub fn decode_icb_capture_active() -> bool {
    CAPTURE_ACTIVE.with(|a| a.get())
}

pub(crate) fn capture_note_pipeline(p: Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_pipeline(p);
        }
    });
}

/// Record that a bind reached the encoder but could not be put on the tape.
pub(crate) fn capture_note_unrecordable_bind() {
    CAPTURE.with(|c| {
        if let Some(cap) = c.borrow_mut().as_mut() {
            cap.note_unrecordable_bind();
        }
    });
}

pub(crate) fn capture_note_bind(index: usize, buf: &GpuBuffer, byte_offset: usize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_bind(index, buf, byte_offset);
        }
    });
}

pub(crate) fn capture_note_immediate(index: usize, bytes: &[u8]) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_immediate(index, bytes);
        }
    });
}

pub(crate) fn capture_note_tg_mem(index: usize, length: usize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_tg_mem(index, length);
        }
    });
}

pub(crate) fn capture_note_dispatch(threadgroups: MTLSize, threads_per_tg: MTLSize) {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_dispatch(threadgroups, threads_per_tg);
        }
    });
}

/// While capturing, remember whether a binder scope ended with an unbarriered
/// dispatch (`pending`), so the next scope can order it even when the runtime
/// itself dropped that state.
///
/// Live execution orders consecutive scopes in one of two ways: the binder's
/// pending-hazard barrier inside a shared encoder (async encode), or the
/// command-buffer boundary (sync encode, or a mid-commit), which orders them
/// for free and so leaves no barrier for capture to see. Replay packs the whole
/// tape into one encoder, where only a marker orders anything. Carrying the
/// pending state across scopes while capturing makes every capture record what
/// the async path would have: the next scope's first dispatch emits the
/// barrier, and `Binder::barrier` lands it on the previous command.
pub(crate) fn capture_carry_hazard(pending: bool) {
    CAPTURE.with(|c| {
        if let Some(cap) = c.borrow_mut().as_mut() {
            cap.carried_hazard = pending;
        }
    });
}

/// The state [`capture_carry_hazard`] stored, cleared by the read. `false` when
/// not capturing.
pub(crate) fn capture_take_carried_hazard() -> bool {
    CAPTURE.with(|c| {
        c.borrow_mut()
            .as_mut()
            .is_some_and(|cap| std::mem::take(&mut cap.carried_hazard))
    })
}

/// Record that the last captured command needs a Device barrier after it.
pub(crate) fn capture_note_barrier() {
    CAPTURE.with(|c| {
        if let Some(ref mut cap) = *c.borrow_mut() {
            cap.note_barrier_after();
        }
    });
}

/// Force ICB-capable pipelines in the cache.
static ICB_PIPELINES: TriFlag = TriFlag::new();

pub fn set_icb_pipelines(on: bool) {
    ICB_PIPELINES.set(on);
}

pub fn icb_pipelines_enabled() -> bool {
    ICB_PIPELINES.get(&["TESSL_ICB_PIPELINES", "METAL_RUNTIME_ICB_PIPELINES"], || false)
}

// When set, `GpuRuntime::with_binder` is a no-op (no Metal encode).
// A process-global flag let one model's replay suppress encoding on another thread.
thread_local! {
    static BINDER_ENCODE_NOP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Arm or disarm binder-nop on this thread.
///
/// Crate-private on purpose: while it is armed every encode on this thread
/// returns `Ok(())` having done nothing. [`BinderEncodeNopGuard`] nests;
/// [`clear_binder_encode_nop`] is the only public switch, and it can only disarm.
pub(crate) fn set_binder_encode_nop(on: bool) {
    BINDER_ENCODE_NOP.with(|flag| flag.set(on));
}

/// Disarm binder-nop on this thread, whatever armed it.
///
/// Safe to call at any time: encoding resumes. A downstream step that must
/// never start under a stale replay flag calls this first.
pub fn clear_binder_encode_nop() {
    set_binder_encode_nop(false);
}

pub fn binder_encode_nop() -> bool {
    BINDER_ENCODE_NOP.with(std::cell::Cell::get)
}

/// RAII: enable binder encode nop; restore off on drop.
pub struct BinderEncodeNopGuard {
    /// The flag value this guard replaced, restored on drop.
    previous: bool,
}

impl BinderEncodeNopGuard {
    pub fn enter() -> Self {
        // Remember what was in effect. Restoring a constant `false` on drop
        // meant an inner guard ended an outer guard's scope: with two nested
        // guards, the inner drop re-enabled encoding while the outer guard was
        // still logically in force, so the layers it was meant to suppress got
        // encoded after all.
        let previous = binder_encode_nop();
        set_binder_encode_nop(true);
        Self { previous }
    }
}

impl Drop for BinderEncodeNopGuard {
    fn drop(&mut self) {
        set_binder_encode_nop(self.previous);
    }
}

/// Serializes tests whose behavior or assertions depend on the process-global
/// ICB / replay flags, and restores every flag on drop (including the -1
/// read-env state, which the raw setters cannot express). Cargo runs tests on
/// parallel threads in one process; without this, one test's `set_*` toggle is
/// visible to every other test mid-flight.
#[cfg(test)]
pub(crate) struct IcbFlagsTestGuard {
    saved: [i8; 4],
    saved_binder_encode_nop: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl IcbFlagsTestGuard {
    pub(crate) fn lock() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        // A panicking guard-holder already restored its flags in drop; the
        // poison carries no extra meaning here.
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        Self {
            saved: [
                DECODE_ICB.raw(),
                ICB_PIPELINES.raw(),
                ICB_EXECUTE.raw(),
                ICB_TRIAGE.raw(),
            ],
            saved_binder_encode_nop: binder_encode_nop(),
            _lock: lock,
        }
    }
}

#[cfg(test)]
impl Drop for IcbFlagsTestGuard {
    fn drop(&mut self) {
        DECODE_ICB.restore(self.saved[0]);
        ICB_PIPELINES.restore(self.saved[1]);
        ICB_EXECUTE.restore(self.saved[2]);
        ICB_TRIAGE.restore(self.saved[3]);
        set_binder_encode_nop(self.saved_binder_encode_nop);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLResidencySet;

    #[test]
    fn decode_icb_flag_default_off() {
        let _flags = IcbFlagsTestGuard::lock();
        set_decode_icb(false);
        assert!(!decode_icb_enabled());
        set_decode_icb(true);
        assert!(decode_icb_enabled());
        set_decode_icb(false);
    }

    #[test]
    fn decode_icb_multi_copy_f32() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("runtime");
        let (mut dec, c) = DecodeIcb::mini_copy_chain(&rt, 32).expect("mini copy chain");
        assert_eq!(dec.command_count(), 2);
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let n = 32usize;
        let out = unsafe { std::slice::from_raw_parts(c.metal().contents().as_ptr() as *const f32, n) };
        for (i, v) in out.iter().take(n).enumerate() {
            assert_eq!(*v, (i as f32) + 1.0, "mismatch at {i}");
        }
        // Second execute without re-encode.
        unsafe {
            let q = c.metal().contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(q, 0xFF, n * 4);
        }
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let out2 = unsafe { std::slice::from_raw_parts(c.metal().contents().as_ptr() as *const f32, n) };
        for (i, v) in out2.iter().take(n).enumerate() {
            assert_eq!(*v, (i as f32) + 1.0);
        }
        assert_eq!(dec.execute_count(), 2);
        let (set_calls, bind_total) = dec.last_set_address_stats();
        let (set_tables, elided) = dec.last_prebuilt_stats();
        // Prefbuilt path: Buf binds elided; mini_copy_chain has no Immediate.
        if dec.prebuilt_table_count() == 2 {
            assert_eq!(set_calls, 0, "prebuilt: no setAddress for Buf-only tape");
            // Two cmds have distinct Buf fingerprints → two setArgumentTable
            // (sticky adopt cannot skip). Dedup count == 2.
            assert_eq!(set_tables, 2, "one setArgumentTable per distinct table");
            assert_eq!(elided, bind_total);
            assert_eq!(dec.unique_prebuilt_table_count(), 2);
        }
        eprintln!("decode_icb_multi_copy_f32: {}", dec.status_line());
    }

    /// A second dispatch under one `set_pipeline` inherits the first one's binds.
    ///
    /// The argument table belongs to the encoder, not to a dispatch. Recording
    /// it with `mem::take` gave `cmd0: binds=N, cmd1: binds=0`, so a replayed
    /// tape would leave every slot unwritten for the second and later
    /// dispatches. The split-K loop has exactly this shape and survived only
    /// because it re-binds all seven scalars every iteration.
    ///
    /// Against the pre-fix code this asserts `3 == 0` on the second command.
    #[test]
    fn binds_persist_across_dispatches_under_one_pipeline() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let a = rt.alloc_buffer_hot(64 * 4).expect("a");
        let b = rt.alloc_buffer_hot(64 * 4).expect("b");
        let n = rt.alloc_buffer_hot(4).expect("n");
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");

        begin_decode_icb_capture();
        capture_note_pipeline(pipe);
        capture_note_bind(0, &a, 0);
        capture_note_bind(1, &b, 0);
        capture_note_bind(2, &n, 0);
        let tg = mtl_size(1, 1, 1);
        capture_note_dispatch(tg, tg);
        // No re-binding between the two dispatches: the table still holds them.
        capture_note_dispatch(tg, tg);
        let cap = take_decode_icb_capture().expect("capture");

        assert_eq!(cap.commands.len(), 2, "two dispatches were recorded");
        assert_eq!(cap.commands[0].binds.len(), 3);
        assert_eq!(
            cap.commands[1].binds.len(),
            3,
            "the second dispatch inherits the argument table; recording it with \
             zero binds would replay with unwritten slots"
        );
    }

    /// A suppressed execute must not be counted as a replay that happened.
    ///
    /// Under binder-encode-nop every `with_binder` body is skipped, so
    /// `execute` encodes nothing and the destination keeps its prior contents.
    /// It nonetheless returned `Ok`, incremented `execute_count` and set
    /// `optimized` — and `cb_replay` counted that as a successful tape replay.
    /// A token that did not happen was indistinguishable from one that did.
    ///
    /// Against the pre-fix code `execute_count` is 1 and `optimized` is true.
    #[test]
    fn an_execute_suppressed_by_nop_is_not_counted() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        set_decode_icb(true);
        let (mut dec, out) = DecodeIcb::mini_copy_chain(&rt, 64).expect("mini copy chain");
        let before = dec.execute_count();

        set_binder_encode_nop(true);
        let result = dec.execute(&rt);
        set_binder_encode_nop(false);

        result.expect("a suppressed execute stays benign for callers holding the guard");
        assert!(dec.last_execute_was_nop(), "the suppression must be observable");
        assert_eq!(
            dec.execute_count(),
            before,
            "no dispatch was encoded, so no execute may be counted"
        );
        // The destination is untouched, which is what makes the old accounting
        // a lie rather than a rounding difference.
        let _ = out;
    }

    /// A nop scope on a poisoned runtime reports the poison instead of `Ok`.
    ///
    /// Against the pre-fix code `execute` returns `Ok` and flags a nop.
    #[test]
    fn a_nop_execute_on_a_poisoned_runtime_fails() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let (mut dec, _out) = DecodeIcb::mini_copy_chain(&rt, 64).expect("mini copy chain");
        assert!(rt.with_binder(|_| Err("injected encode failure".into())).is_err());
        let _nop = BinderEncodeNopGuard::enter();
        assert!(dec.execute(&rt).is_err(), "poison must not be hidden by the nop scope");
        assert!(rt.with_binder(|_| Ok(())).is_err());
    }

    /// A dropped tape leaves the residency set: 100 build-and-drops return the
    /// set to where it started. Against the pre-fix code each leaves its ICB
    /// registered and the count is 100 higher.
    #[test]
    fn dropping_tapes_returns_residency_to_baseline() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        rt.synchronize().unwrap();
        let baseline = rt.metal4.residency.allocationCount();
        for _ in 0..100 {
            let (mut dec, out) = DecodeIcb::mini_copy_chain(&rt, 16).expect("mini copy chain");
            dec.execute(&rt).expect("execute");
            drop((dec, out));
        }
        rt.synchronize().unwrap();
        assert_eq!(rt.metal4.residency.allocationCount(), baseline);
    }

    /// A second `begin` must not silently discard the first tape.
    ///
    /// The begin/take pair has no RAII guard, so a caller that bails between
    /// them leaks the capture for the thread's lifetime — growing without bound
    /// and holding a `GpuBuffer` clone per bind, which keeps those buffers out
    /// of the freelist and pinned in residency. Silently replacing the tape on
    /// re-entry is what kept that invisible.
    ///
    /// Against the pre-fix code the recorded command is gone and
    /// `nested_begins` does not exist.
    #[test]
    fn a_second_capture_begin_is_recorded_not_silent() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let a = rt.alloc_buffer_hot(64 * 4).expect("a");
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");
        let tg = mtl_size(1, 1, 1);

        begin_decode_icb_capture();
        capture_note_pipeline(pipe);
        capture_note_bind(0, &a, 0);
        capture_note_dispatch(tg, tg);

        // A caller that forgot to take, then began again.
        begin_decode_icb_capture();

        let cap = take_decode_icb_capture().expect("capture");
        assert_eq!(
            cap.commands.len(),
            1,
            "the first tape's work must survive re-entry, not be discarded"
        );
        assert_eq!(cap.nested_begins, 1, "re-entry must be recorded so the leak is visible");

        // And the explicit escape hatch actually clears it.
        begin_decode_icb_capture();
        assert!(decode_icb_capture_active());
        end_decode_icb_capture();
        assert!(
            !decode_icb_capture_active(),
            "end_decode_icb_capture must release the tape and its pinned buffers"
        );
    }

    /// A nested nop guard must not end its parent's scope on drop.
    ///
    /// `Drop` restored a constant `false`, so an inner guard's drop re-enabled
    /// encoding while the outer guard was still logically in force — and the
    /// layers the outer guard existed to suppress got encoded after all.
    ///
    /// Against the pre-fix code the final assertion sees `false`.
    #[test]
    fn a_nested_nop_guard_restores_its_parent() {
        let _flags = IcbFlagsTestGuard::lock();
        set_binder_encode_nop(false);
        assert!(!binder_encode_nop());

        let outer = BinderEncodeNopGuard::enter();
        assert!(binder_encode_nop(), "the outer guard enables nop");
        {
            let _inner = BinderEncodeNopGuard::enter();
            assert!(binder_encode_nop());
        }
        assert!(
            binder_encode_nop(),
            "the inner guard's drop must restore the outer guard's state, not a constant"
        );
        drop(outer);
        assert!(!binder_encode_nop(), "the outer drop restores the original");
    }

    /// Replay suppression belongs to one host thread. A process-global switch
    /// made an unrelated model on another thread skip every `with_binder` body
    /// and return success over stale device memory.
    ///
    /// Against the process-global flag the spawned thread observes `true`.
    #[test]
    fn binder_encode_nop_does_not_cross_thread_boundaries() {
        let _flags = IcbFlagsTestGuard::lock();
        set_binder_encode_nop(false);
        let outer = BinderEncodeNopGuard::enter();
        assert!(binder_encode_nop());

        std::thread::spawn(|| {
            assert!(!binder_encode_nop(), "another thread inherited replay suppression");
            let local = BinderEncodeNopGuard::enter();
            assert!(binder_encode_nop());
            drop(local);
            assert!(!binder_encode_nop());
        })
        .join()
        .expect("thread-local replay-state probe");

        assert!(
            binder_encode_nop(),
            "another thread changed the original thread's guard"
        );
        drop(outer);
        assert!(!binder_encode_nop());
    }

    /// A bind the argument table cannot hold must be refused, not dropped.
    ///
    /// The prebuilt encode path filters `index < ARG_TABLE_SLOTS`
    /// and silently drop anything above, so a tape built with `index: 40` used
    /// to return `Ok` and replay with that operand absent. `buf_bind_fingerprint`
    /// applies the same filter, so two commands differing only in an
    /// out-of-range bind also collided onto one shared table.
    ///
    /// Against the pre-fix code `from_commands` returns `Ok` here.
    #[test]
    fn an_out_of_range_bind_index_is_refused() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let a = rt.alloc_buffer_hot(64 * 4).expect("a");
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");
        let tg = mtl_size(1, 1, 1);

        // ARG_TABLE_SLOTS is 31, so 31 is already one past the end.
        for index in [ARG_TABLE_SLOTS, 40] {
            let cmds = vec![DecodeIcbCommand {
                pipeline: pipe.clone(),
                threadgroups: tg,
                threads_per_tg: tg,
                binds: vec![DecodeIcbBind::Buf {
                    index,
                    buf: a.clone(),
                    byte_offset: 0,
                    gpu_addr: buf_gpu_addr(&a, 0),
                }],
                tg_mem: None,
                barrier_after: false,
                incomplete_binds: 0,
            }];
            let err = DecodeIcb::from_commands(&rt, cmds)
                .map(|_| ())
                .expect_err("index {index} is past the argument table");
            assert!(
                err.contains("argument table"),
                "the error must name the cause, got: {err}"
            );
        }
    }

    /// `Buf`'s fields are public, so a tape can disagree with itself. The
    /// builder must recompute the address and refuse offsets/slots outside what
    /// the buffer and the argument table hold.
    ///
    /// Against the pre-fix code every refusal below returns `Ok`, and the
    /// forged `gpu_addr` survives into the built tape.
    #[test]
    fn from_commands_refuses_forged_buf_addresses_and_offsets() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let other = GpuRuntime::new().expect("second runtime");
        let a = rt.alloc_buffer_hot(64 * 4).expect("a");
        let foreign = other.alloc_buffer_hot(64 * 4).expect("foreign");
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");
        let tg = mtl_size(1, 1, 1);
        let cmd = |bind: DecodeIcbBind, tg_mem: Option<(usize, usize)>| DecodeIcbCommand {
            pipeline: pipe.clone(),
            threadgroups: tg,
            threads_per_tg: tg,
            binds: vec![bind],
            tg_mem,
            barrier_after: false,
            incomplete_binds: 0,
        };
        let buf_at = |buf: &GpuBuffer, byte_offset: usize, gpu_addr: u64| DecodeIcbBind::Buf {
            index: 0,
            buf: buf.clone(),
            byte_offset,
            gpu_addr,
        };

        {
            let at_end = cmd(buf_at(&a, a.nbytes(), 1 << 32), None);
            let err = DecodeIcb::from_commands(&rt, vec![at_end])
                .map(|_| ())
                .expect_err("byte_offset == nbytes is outside the buffer");
            assert!(err.contains("byte offset"), "{err}");

            let err = DecodeIcb::from_commands(&rt, vec![cmd(buf_at(&foreign, 0, 0), None)])
                .map(|_| ())
                .expect_err("another runtime's buffer");
            assert!(err.contains("another runtime"), "{err}");

            let bad_tg = cmd(buf_at(&a, 0, buf_gpu_addr(&a, 0)), Some((ARG_TABLE_SLOTS, 16)));
            let err = DecodeIcb::from_commands(&rt, vec![bad_tg])
                .map(|_| ())
                .expect_err("threadgroup slot past the table");
            assert!(err.contains("threadgroup"), "{err}");

            // A forged address with a legal offset is replaced, not trusted.
            let forged = cmd(buf_at(&a, 16, 1 << 32), None);
            let dec = DecodeIcb::from_commands(&rt, vec![forged]).expect("legal offset builds");
            let DecodeIcbBind::Buf { gpu_addr, .. } = &dec.commands[0].binds[0] else {
                panic!("expected a Buf bind");
            };
            assert_eq!(*gpu_addr, buf_gpu_addr(&a, 16), "gpu_addr must be recomputed");
        }
    }

    /// Owned binds made through the real `Binder` are recorded, not counted as
    /// unrecordable, so a tape of them builds and replays.
    ///
    /// `bind_tensor` and `bind_gpu_buf_offset` used to finish in the raw
    /// `bind_buf`, which marks every bind unrecordable before the owned bind
    /// was noted. Any captured kernel then carried `incomplete_binds > 0`.
    #[test]
    fn owned_binds_capture_complete_and_replay() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let n = 64usize;
        let src = rt.alloc_buffer_hot(n * 4).expect("src");
        let dst = rt.alloc_tensor_f32(&[n]).expect("dst");
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = i as f32 + 1.0;
            }
        }
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");
        let tpt = pipe.threadExecutionWidth().min(n).max(1);
        let groups = n.div_ceil(tpt);

        begin_decode_icb_capture();
        let encoded = rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src, 0);
            bnd.bind_tensor(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        });
        let cap = take_decode_icb_capture().expect("capture");
        encoded.expect("captured dispatch");
        rt.synchronize().unwrap();

        assert_eq!(cap.commands.len(), 1);
        assert_eq!(cap.commands[0].incomplete_binds, 0, "owned binds are recordable");
        assert_eq!(cap.commands[0].binds.len(), 3);

        let mut dec = DecodeIcb::from_commands(&rt, cap.commands).expect("tape builds");
        unsafe {
            std::ptr::write_bytes(dst.buffer.metal().contents().as_ptr() as *mut u8, 0, n * 4);
        }
        dec.execute(&rt).unwrap();
        rt.synchronize().unwrap();
        let out = dst.read_f32().unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as f32 + 1.0, "mismatch at {i}");
        }
    }

    /// A capture that could not record an operand bind says so in
    /// `incomplete_binds`, and the tape still builds: it is inherit-buffers, so
    /// the live argument table supplies the bind at execute time.
    ///
    /// `bind_buf` takes a raw `MTLBuffer` with no owning `GpuBuffer`, so there
    /// is nothing to record and nothing to pin.
    #[test]
    fn a_tape_missing_operand_binds_is_counted_and_still_builds() {
        let _flags = IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("GpuRuntime::new");
        let a = rt.alloc_buffer_hot(64 * 4).expect("a");
        let pipe = pipeline_icb(&rt, "copy_f32").expect("copy_f32 icb pipeline");

        begin_decode_icb_capture();
        capture_note_pipeline(pipe);
        capture_note_bind(0, &a, 0);
        // Stand in for `Binder::bind_buf`, which reaches the encoder with no
        // owning handle for the tape to hold.
        capture_note_unrecordable_bind();
        let tg = mtl_size(1, 1, 1);
        capture_note_dispatch(tg, tg);
        let cap = take_decode_icb_capture().expect("capture");

        assert_eq!(cap.commands[0].incomplete_binds, 1);
        DecodeIcb::from_commands(&rt, cap.commands)
            .map(|_| ())
            .expect("inherit-buffers replay does not depend on the tape's binds");
    }
}
