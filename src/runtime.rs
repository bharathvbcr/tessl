//! objc2-metal runtime: device, Metal 4 encode path, pipeline cache, buffer pool,
//! persistent argument-table pattern.
//!
//! Encode is **Metal 4 only**: one `MTL4CommandBuffer` per step with
//! argument-table binds, a bump-allocated const arena (16 MiB), residency
//! registry, and SharedEvent sync. Steady-state work never host-waits except
//! at log / loss / eval boundaries via [`GpuRuntime::synchronize`].
//!
//! Audit 4 lessons preserved: cold-buffer recycle + `removeAllocation` after CB
//! complete; one compute encoder packed across `with_binder` calls (P1);
//! working-set probe; no host-zero mid-CB.

use core::ptr::NonNull;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_foundation::{NSData, NSRange, NSString, NSURL};
#[cfg(feature = "quant-prep")]
use objc2_metal::MTLTensor;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4ComputeCommandEncoder, MTL4ComputePipelineDescriptor, MTL4CounterHeap,
    MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTL4IndirectCommandBufferSupportState,
    MTL4LibraryFunctionDescriptor, MTL4TimestampHeapEntry, MTL4VisibilityOptions, MTLAllocation,
    MTLBuffer, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent,
    MTLIndirectCommandBuffer, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStages,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Const arena for Metal 4 scalar binds (distinct offsets; reset after sync).
// 31B dense decode packs hundreds of binder consts across mid-commits within a
// token before a waiting sync; 1 MiB exhausted mid-token. 16 MiB covers full
// product shapes with headroom (still tiny vs Hot weight residency).
const METAL4_CONST_ARENA_BYTES: usize = 16 * 1024 * 1024;

/// Constant-arena bytes a scope may bind. A scope that opens with less than
/// this free first drains the GPU with a waiting commit, which rewinds the
/// arena, so the scope never fails part-way through for lack of arena.
/// Exhaustion used to fail the scope and poison the runtime, with nothing
/// forcing the flush that would have freed it (audit R8). Reaching the
/// reserve takes hundreds of thousands of unsynchronized dispatches.
pub(crate) const CONST_ARENA_SCOPE_RESERVE: usize = 1 << 20;

/// Default pool freelist cap (~2 GiB of cached slabs).
const DEFAULT_POOL_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Buffer slots in the Metal 4 argument table.
///
/// Every argument table this crate builds is created with this bind count, so a
/// buffer index is in range iff it is `< ARGUMENT_TABLE_MAX_BUFFERS`. Public
/// because callers that bind by raw index (e.g. `mtl_tensor::bind_mtl_tensor`,
/// whose `setResource:atBufferIndex:` Metal does not range-check) have to check
/// against the same number the table was built with.
pub const ARGUMENT_TABLE_MAX_BUFFERS: usize = 31;

/// Residency / recycle policy for pooled buffers (Audit 4 P0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferKind {
    /// Mid-step temps — recycle + removeAllocation after CB complete.
    Cold,
    /// Weights / optim / long-lived — stay resident and never enter the
    /// freelist while owned; retire after their final handle drops.
    Hot,
    /// Bump slab — sub-allocated views share it and the cursor resets after
    /// sync. Storage retires exactly like [`Self::Cold`]: `Drop` schedules the
    /// slab for recycle, so it returns to the freelist once its last view has
    /// dropped and the CB that used it has completed.
    Bump,
    /// Caller-owned `MTLBuffer` wrapped via [`crate::Tensor::from_mtl_buffer`].
    /// Retires like [`Self::Hot`] (no freelist).
    External,
}

/// Probed device memory budget (logged in train banner).
#[derive(Clone, Copy, Debug)]
pub struct DeviceMemoryInfo {
    pub recommended_working_set: u64,
    pub memory_size: u64,
    pub wired_budget: u64,
    pub pool_cache_cap: usize,
}

/// Precision mode for the training hot path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Parity / `--f32`: all storage+compute f32.
    F32,
    /// Phase 4 default: bf16 storage/compute, f32 accum (GEMM/softmax/RMS/loss/optim).
    Bf16,
}

struct PipelineCache {
    map: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        overlays: &[Retained<ProtocolObject<dyn MTLLibrary>>],
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let icb = crate::decode_icb::icb_pipelines_enabled();
        let key = if icb {
            format!("icb:{name}")
        } else {
            name.to_string()
        };
        if let Some(p) = self.map.get(&key) {
            return Ok(p.clone());
        }
        let fname = NSString::from_str(name);
        let containing: &ProtocolObject<dyn MTLLibrary> =
            if library.newFunctionWithName(&fname).is_some() {
                library
            } else if let Some(lib) = overlays
                .iter()
                .find(|lib| lib.newFunctionWithName(&fname).is_some())
            {
                lib
            } else {
                return Err(format!("kernel '{name}' not found in metallib"));
            };

        let pipeline = if icb {
            let compiler_desc = MTL4CompilerDescriptor::new();
            let compiler = device
                .newCompilerWithDescriptor_error(&compiler_desc)
                .map_err(|e| format!("MTL4Compiler: {e}"))?;
            let func_desc = MTL4LibraryFunctionDescriptor::new();
            func_desc.setName(Some(&fname));
            func_desc.setLibrary(Some(containing));
            let pipe_desc = MTL4ComputePipelineDescriptor::new();
            pipe_desc.setComputeFunctionDescriptor(Some(func_desc.as_super()));
            pipe_desc
                .setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);
            let p = compiler
                .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
                .map_err(|e| format!("ICB pipeline '{name}': {e}"))?;
            if !p.supportIndirectCommandBuffers() {
                return Err(format!(
                    "ICB pipeline '{name}' supportIndirectCommandBuffers=false"
                ));
            }
            p
        } else {
            let func = containing
                .newFunctionWithName(&fname)
                .ok_or_else(|| format!("kernel '{name}' not found in metallib"))?;
            device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| format!("pipeline '{name}': {e}"))?
        };
        self.map.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

struct BufferPool {
    freelist: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    cached_bytes: usize,
    max_cache_bytes: usize,
}

impl BufferPool {
    fn new(max_cache_bytes: usize) -> Self {
        Self {
            freelist: HashMap::new(),
            cached_bytes: 0,
            max_cache_bytes,
        }
    }

    fn bucket(nbytes: usize) -> usize {
        nbytes.next_power_of_two().max(256)
    }

    fn alloc(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        nbytes: usize,
    ) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, bool), String> {
        if nbytes > isize::MAX as usize || nbytes > device.maxBufferLength() {
            return Err(format!(
                "buffer request {nbytes} exceeds host/device allocation limit"
            ));
        }
        let key = Self::bucket(nbytes);
        if key < nbytes || key > device.maxBufferLength() {
            return Err("rounded buffer size exceeds device limit".into());
        }
        if let Some(v) = self.freelist.get_mut(&key) {
            if let Some(b) = v.pop() {
                self.cached_bytes = self.cached_bytes.saturating_sub(key);
                // true = came from freelist (already resided previously; re-add)
                return Ok((b, true));
            }
        }
        let b = device
            .newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("newBuffer({key}) failed"))?;
        Ok((b, false))
    }

    fn recycle(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let key = Self::bucket(buffer.length());
        if key > self.max_cache_bytes.saturating_sub(self.cached_bytes) {
            // Drop buffer (let ARC release) — over cache cap.
            return;
        }
        self.cached_bytes += key;
        self.freelist.entry(key).or_default().push(buffer);
    }

    fn set_max_cache_bytes(&mut self, max_cache_bytes: usize) {
        self.max_cache_bytes = max_cache_bytes;
        // Trim if over (drop largest buckets first).
        if self.cached_bytes <= max_cache_bytes {
            return;
        }
        let mut keys: Vec<usize> = self.freelist.keys().copied().collect();
        keys.sort_unstable_by(|a, b| b.cmp(a));
        for key in keys {
            while self.cached_bytes > max_cache_bytes {
                let Some(v) = self.freelist.get_mut(&key) else {
                    break;
                };
                if v.pop().is_none() {
                    break;
                }
                self.cached_bytes = self.cached_bytes.saturating_sub(key);
            }
        }
    }
}

pub struct PersistentArgumentTable {
    pub table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    pub max_buffers: u64,
}

/// Strong references required by an unretained Metal 4 command buffer.
///
/// `Binder` records every raw Objective-C object that it places directly into
/// an encoder. The active batch transfers these references to its allocator
/// slot before submission, and the slot releases them only after its
/// `MTLSharedEvent` value completes. Owned Tessl buffers use the separate
/// pending-retirement path, but raw public buffer binds need this additional
/// anchor because the caller may release its handle immediately after encode.
pub(crate) enum InFlightAnchor {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    ArgumentTable(Retained<ProtocolObject<dyn MTL4ArgumentTable>>),
    Icb(Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>),
    /// An `MTLResourceID` does not retain the `MTLTensor` object it names.
    /// Buffer-backed tensor views therefore need their own completion anchor
    /// even though their storage allocation is independently retained.
    #[cfg(feature = "quant-prep")]
    Tensor(Retained<ProtocolObject<dyn MTLTensor>>),
}

impl InFlightAnchor {
    #[inline]
    fn identity(&self) -> (u8, usize) {
        match self {
            Self::Buffer(value) => (0, Retained::as_ptr(value) as *const () as usize),
            Self::Pipeline(value) => (1, Retained::as_ptr(value) as *const () as usize),
            Self::ArgumentTable(value) => (2, Retained::as_ptr(value) as *const () as usize),
            Self::Icb(value) => (3, Retained::as_ptr(value) as *const () as usize),
            #[cfg(feature = "quant-prep")]
            Self::Tensor(value) => (4, Retained::as_ptr(value) as *const () as usize),
        }
    }

    fn leak_clone(&self) {
        match self {
            Self::Buffer(value) => std::mem::forget(value.clone()),
            Self::Pipeline(value) => std::mem::forget(value.clone()),
            Self::ArgumentTable(value) => std::mem::forget(value.clone()),
            Self::Icb(value) => std::mem::forget(value.clone()),
            #[cfg(feature = "quant-prep")]
            Self::Tensor(value) => std::mem::forget(value.clone()),
        }
    }
}

#[derive(Default)]
struct InFlightAnchors {
    objects: Vec<InFlightAnchor>,
    identities: HashSet<(u8, usize)>,
}

impl InFlightAnchors {
    pub(crate) fn extend(&mut self, anchors: impl IntoIterator<Item = InFlightAnchor>) {
        for anchor in anchors {
            if self.identities.insert(anchor.identity()) {
                self.objects.push(anchor);
            }
        }
    }

    fn clear(&mut self) {
        self.objects.clear();
        self.identities.clear();
    }

    fn leak_clones(&self) {
        for anchor in &self.objects {
            anchor.leak_clone();
        }
    }
}

/// One MTL4 allocator slot (ping-pong for mid-token commit without wait).
struct AllocatorSlot {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    /// SharedEvent value that must land before reset+reuse; 0 = free.
    in_flight: u64,
    /// Raw objects referenced by the command encoded with this allocator.
    anchors: InFlightAnchors,
}

/// Metal 4 encode package (queue / dual allocators / argument table / CounterHeap).
pub struct Metal4EncodePackage {
    pub queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Dual allocators — GPU runs one CB while host encodes the next.
    allocators: Mutex<[AllocatorSlot; 2]>,
    active_alloc: Mutex<usize>,
    /// Legacy alias: allocator 0 (tests / callers that expected a single handle).
    pub allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pub command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub argument_table: PersistentArgumentTable,
    pub counter_heap: Option<Retained<ProtocolObject<dyn MTL4CounterHeap>>>,
    pub residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Scratch for scalar `[[buffer(N)]]` constants (M4 has no setBytes).
    /// 16 MiB bump arena; cursor advances per const pack, reset after sync.
    pub const_staging: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub const_cursor: Mutex<usize>,
    event_value: Mutex<u64>,
    /// Allocations registered into `residency` (debug / telemetry).
    pub residency_count: Mutex<usize>,
}

/// Soft mid-token commit threshold (dispatches since last commit).
/// Default off (`0`). Enable with `TESSL_MID_COMMIT=N` (e.g. 128–256);
/// `METAL_RUNTIME_MID_COMMIT` is still read for compatibility.
/// Free-allocator pick avoids the wait-storm when the peer slot is still busy.
fn mid_commit_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TESSL_MID_COMMIT")
            .or_else(|_| std::env::var("METAL_RUNTIME_MID_COMMIT"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Open Metal 4 command buffer for the current step (one allocator CB at a time).
/// Audit 4 P1: keep a single compute encoder open across `with_binder` calls.
struct ActiveMetal4Batch {
    dispatches: usize,
    /// Dispatches since last commit (mid-token overlap).
    since_commit: usize,
    cb_open: bool,
    stamped_t0: bool,
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    alloc_idx: usize,
    /// Raw objects referenced by the open, not-yet-submitted command buffer.
    anchors: InFlightAnchors,
    /// The previous scope on `encoder` ended with an unbarriered dispatch
    /// (hazard mode). The next scope opens with a barrier and clears it.
    hazard_pending: bool,
    /// `encoder` already holds the runtime's persistent argument table, left
    /// by the previous scope; the next one skips `setArgumentTable`. Cleared
    /// whenever the encoder is created (audit R10c).
    arg_table_latched: bool,
}

/// Leading text of the one `bump_alloc_f32` error that means "try the pool".
const BUMP_EXHAUSTED_PREFIX: &str = "bump arena exhausted:";

struct BumpState {
    buffer: crate::tensor::GpuBuffer,
    cursor: usize,
    capacity: usize,
}

/// Exclusive CPU/GPU access lease. Busy/reentrant access fails instead of blocking.
pub(crate) struct RuntimeAccess(Arc<AtomicBool>);
impl Drop for RuntimeAccess {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Restores the caller-visible async mode even when a binder callback unwinds.
struct AsyncEncodeRestore<'a> {
    mode: &'a Mutex<bool>,
    value: bool,
}

impl Drop for AsyncEncodeRestore<'_> {
    fn drop(&mut self) {
        *self
            .mode
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = self.value;
    }
}

/// A pooled buffer awaiting recycle, with the size it was allocated at.
type PendingRecycle = (Retained<ProtocolObject<dyn MTLBuffer>>, usize);

/// A non-recyclable allocation whose last owning handle has dropped while GPU
/// work may still reference it.
///
/// Keep the concrete retained type so objc2 can preserve the Objective-C
/// lifetime without an unchecked protocol-object cast. Every variant exposes
/// an `MTLAllocation` view when the completed-work drain removes residency.
enum PendingRetirement {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Icb {
        icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
        /// MTL4 command buffers use unretained-reference semantics. These
        /// per-tape tables and pipelines are not necessarily owned by the
        /// runtime's shared caches, so the ICB allocation alone is not a
        /// sufficient completion anchor.
        argument_tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    },
    #[cfg(feature = "quant-prep")]
    Tensor(Retained<ProtocolObject<dyn MTLTensor>>),
}

impl PendingRetirement {
    fn allocation(&self) -> &ProtocolObject<dyn MTLAllocation> {
        match self {
            Self::Buffer(buffer) => ProtocolObject::<dyn MTLAllocation>::from_ref(&**buffer),
            Self::Icb { icb, .. } => ProtocolObject::<dyn MTLAllocation>::from_ref(&**icb),
            #[cfg(feature = "quant-prep")]
            Self::Tensor(tensor) => ProtocolObject::<dyn MTLAllocation>::from_ref(&**tensor),
        }
    }

    /// Preserve a second strong reference when a bounded final-runtime wait
    /// times out. Intentionally leaking on a wedged device is safer than
    /// releasing an object an unretained MTL4 command may still dereference.
    fn leak_clone(&self) {
        match self {
            Self::Buffer(buffer) => std::mem::forget(buffer.clone()),
            Self::Icb {
                icb,
                argument_tables,
                pipelines,
            } => {
                std::mem::forget(icb.clone());
                for table in argument_tables {
                    std::mem::forget(table.clone());
                }
                for pipeline in pipelines {
                    std::mem::forget(pipeline.clone());
                }
            }
            #[cfg(feature = "quant-prep")]
            Self::Tensor(tensor) => std::mem::forget(tensor.clone()),
        }
    }
}

const FINAL_DROP_WAIT_MS: u64 = 30_000;

#[inline]
fn max_event_value(events: impl Iterator<Item = u64>) -> u64 {
    events.max().unwrap_or(0)
}

/// Recover the inner value during final destruction even if a prior panic
/// poisoned the mutex. Drop must never panic and abandon GPU lifetime anchors.
#[inline]
fn mutex_value_mut<T>(mutex: &mut Mutex<T>) -> &mut T {
    match mutex.get_mut() {
        Ok(value) => value,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Shared GPU runtime (Metal 4 encode required).
///
/// Metal 4 command buffers do not retain the objects they reference. Dropping
/// the final runtime owner therefore waits up to 30 seconds for any submitted
/// allocator event. A device that still has not completed after that bounded
/// wait causes the runtime to intentionally leak one strong reference to the
/// complete in-flight object graph; this trades memory on a wedged device for
/// freedom from GPU use-after-free.
///
/// Metal encoder objects are thread-affine; do not add unsafe Send/Sync. Host
/// mapping also excludes same-thread reentry into encode and submission.
///
/// ```compile_fail,E0277
/// use tessl::runtime::GpuRuntime;
/// fn require_send<T: Send>() {}
/// require_send::<GpuRuntime>();
/// ```

/// Persistent Hot scalar workspace (pos-buffer style). Bounded bump arena for
/// stable GPU addresses across encodes — not a full decode ICB scalar graph.
pub struct ParamsBuffer {
    buffer: crate::tensor::GpuBuffer,
    cursor: Mutex<usize>,
    capacity: usize,
}

impl ParamsBuffer {
    /// 4 KiB of Hot u32/f32 slots — enough for per-token dims without migrating
    /// the full decode graph into an ICB scalar pool.
    pub const CAPACITY_BYTES: usize = 4096;

    fn new(rt: &GpuRuntime) -> Result<Self, String> {
        let buffer = rt.alloc_buffer_kind(Self::CAPACITY_BYTES, BufferKind::Hot)?;
        // SAFETY: freshly allocated Hot buffer; no live views or in-flight CB.
        unsafe {
            buffer.zero_unsubmitted();
        }
        Ok(Self {
            buffer,
            cursor: Mutex::new(0),
            capacity: Self::CAPACITY_BYTES,
        })
    }

    /// Reset the bump cursor (call once per step before binding scalars).
    pub fn reset(&self) {
        *self.cursor.lock().unwrap_or_else(|p| p.into_inner()) = 0;
    }

    /// Push a `u32`; returns byte offset into the params buffer.
    pub fn push_u32(&self, v: u32) -> Result<usize, String> {
        let mut cursor = self.cursor.lock().map_err(|e| e.to_string())?;
        let offset = *cursor;
        let next = offset
            .checked_add(4)
            .ok_or_else(|| "params buffer cursor overflow".to_string())?;
        if next > self.capacity {
            return Err(format!(
                "params buffer exhausted (cap {} bytes)",
                self.capacity
            ));
        }
        let ptr = self.buffer.metal().contents().as_ptr() as *mut u8;
        unsafe {
            std::ptr::write_unaligned(ptr.add(offset) as *mut u32, v);
        }
        *cursor = next;
        Ok(offset)
    }

    /// Push an `f32`; returns byte offset into the params buffer.
    pub fn push_f32(&self, v: f32) -> Result<usize, String> {
        self.push_u32(v.to_bits())
    }

    pub fn buffer(&self) -> &crate::tensor::GpuBuffer {
        &self.buffer
    }
}

pub struct GpuRuntime {
    access_busy: Arc<AtomicBool>,
    encode_failed: AtomicBool,
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Extra metallibs (e.g. gemma-metal overlay) searched after [`Self::library`].
    overlay_libraries: Mutex<Vec<Retained<ProtocolObject<dyn MTLLibrary>>>>,
    /// Metal 4 encode package (required; init fails if unavailable).
    pub(crate) metal4: Metal4EncodePackage,
    pipelines: Mutex<PipelineCache>,
    pool: Mutex<BufferPool>,
    /// Per-step bump arena (single slab). Reset only after GPU work that used
    /// bump views has completed (or at the start of a step following a sync).
    bump: Mutex<Option<BumpState>>,
    has_tensorops: bool,
    /// When true, kernels accumulate into one CB until [`Self::synchronize`] / [`Self::commit`].
    async_encode: Mutex<bool>,
    active_m4: Mutex<Option<ActiveMetal4Batch>>,
    /// Last CounterHeap (t0, t1) resolved at synchronize.
    last_m4_stamps: Mutex<Option<(u64, u64)>>,
    /// `with_binder` calls since the last [`Self::take_dispatch_count`]
    /// (fusion/telemetry). Commits do not clear it; only the taker does.
    dispatch_count: Mutex<usize>,
    precision: Mutex<PrecisionMode>,
    /// Phase H bridge: TensorOps f32 GEMM with `relaxed_precision` (tf32-class).
    /// Off by default so f32 goldens stay exact; enable via `--tf32` / `set_relaxed_precision`.
    relaxed_precision: Mutex<bool>,
    /// Prefer TensorOps multi-block flash probe over simdgroup FA-2 (`--flash-tensorops`).
    flash_tensorops: Mutex<bool>,
    /// Uncommitted residency adds/removes — flushed before encode / synchronize.
    residency_dirty: Mutex<bool>,
    /// Cold buffers whose last Arc dropped mid-step; recycled after CB wait.
    pending_cold_recycle: Mutex<Vec<PendingRecycle>>,
    /// Hot buffers, ICBs, and device-backed tensors are not freelist entries,
    /// but their residency still has to be removed after the final GPU use.
    /// Retaining them here bridges last-handle drop to the next completed-work
    /// drain.
    pending_retirement: Mutex<Vec<PendingRetirement>>,
    /// Bounded Hot params workspace for stable scalar binds (pos-buffer style).
    params: Mutex<Option<ParamsBuffer>>,
    /// Self weak handle so Drop on pooled buffers can schedule recycle.
    /// Set exactly once in [`GpuRuntime::new`], right after the `Arc` exists;
    /// a `OnceLock` so no lock can be poisoned and no reader can observe an
    /// unset value as a silently dangling `Weak`.
    self_weak: OnceLock<Weak<GpuRuntime>>,
    /// Probed working-set / wired budget (P0b).
    memory_info: Mutex<DeviceMemoryInfo>,
}

/// Kernel-use trace, gated on `TESSL_KERNEL_TRACE=1`.
///
/// Benchmark coverage used to be an inference: grep a kernel's name out of the
/// bench sources and hope the path that dispatches it is the one being timed.
/// That mis-attributed both directions: `matmul2d_tensorops_*`, reached through
/// a dispatcher, was omitted while a name present only in a comment was
/// reported as measured. This records what a run actually dispatched, so
/// `bench/kernel_coverage.py` can gate on a measurement instead.
///
/// Off by default and read through a `OnceLock`, so the cost on the dispatch
/// path when disabled is one relaxed load.
static KERNEL_TRACE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static KERNEL_TRACE: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn record_kernel_use(name: &str) {
    if !*KERNEL_TRACE_ON.get_or_init(|| std::env::var_os("TESSL_KERNEL_TRACE").is_some()) {
        return;
    }
    if let Ok(mut set) = KERNEL_TRACE.lock() {
        if !set.contains(name) {
            set.insert(name.to_string());
        }
    }
}

/// Every distinct kernel this process has requested a pipeline for, sorted.
///
/// Empty unless `TESSL_KERNEL_TRACE` is set — a caller that forgets to set it
/// would otherwise read an empty trace as "nothing ran".
pub fn traced_kernels() -> Vec<String> {
    KERNEL_TRACE
        .lock()
        .map(|s| s.iter().cloned().collect())
        .unwrap_or_default()
}

/// Whether the trace is recording. Lets a caller distinguish "nothing
/// dispatched" from "tracing was never switched on".
pub fn kernel_trace_enabled() -> bool {
    *KERNEL_TRACE_ON.get_or_init(|| std::env::var_os("TESSL_KERNEL_TRACE").is_some())
}

impl GpuRuntime {
    fn acquire_access(&self) -> Result<RuntimeAccess, String> {
        if self.encode_failed.load(Ordering::Acquire) {
            return Err("runtime is poisoned after encode/submit failure; recreate it".into());
        }
        self.access_busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| {
                "runtime busy: another host mapping, encoder, or submit is active".to_string()
            })?;
        let access = RuntimeAccess(Arc::clone(&self.access_busy));
        if self.encode_failed.load(Ordering::Acquire) {
            return Err("runtime poisoned by an earlier encoding/submission failure".into());
        }
        Ok(access)
    }

    pub(crate) fn host_access(&self) -> Result<RuntimeAccess, String> {
        let access = self.acquire_access()?;
        // Host access needs completed GPU work, not residency. With no batch
        // open and no allocator in flight there is nothing to commit, wait
        // for, or make resident, and skipping `commit_m4` keeps a
        // load-then-write loop from paying a residency commit and an event
        // round trip per tensor.
        if self.gpu_work_pending() {
            if let Err(e) = self.commit_m4(true) {
                self.encode_failed.store(true, Ordering::Release);
                return Err(e);
            }
        }
        Ok(access)
    }

    /// Test-only: apply the same poison `encode_failed` bit that a SharedEvent
    /// wait timeout stores before returning.
    #[cfg(test)]
    pub fn poison_as_shared_event_timeout_for_test(&self) {
        self.encode_failed.store(true, Ordering::Release);
    }

    /// Whether a batch is open or any allocator slot still has work in flight.
    /// Poisoned bookkeeping counts as pending, so the conservative path runs.
    fn gpu_work_pending(&self) -> bool {
        let batch_present = self
            .active_m4
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(true);
        if batch_present {
            return true;
        }
        self.metal4
            .allocators
            .lock()
            .map(|slots| slots.iter().any(|slot| slot.in_flight != 0))
            .unwrap_or(true)
    }

    pub fn new() -> Result<Arc<Self>, String> {
        Self::from_metallib_path(Path::new(crate::metallib_path()))
    }

    /// Inference decode runtime: no CounterHeap timestamps (host encode tax).
    pub fn new_inference() -> Result<Arc<Self>, String> {
        Self::from_metallib_path_opts(Path::new(crate::metallib_path()), /*timestamps*/ false)
    }

    pub fn from_metallib_path(path: &Path) -> Result<Arc<Self>, String> {
        Self::from_metallib_path_opts(path, /*timestamps*/ true)
    }

    pub fn from_metallib_path_opts(path: &Path, timestamps: bool) -> Result<Arc<Self>, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_string())?;

        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-utf8 metallib path: {path:?}"))?;
        if !path.exists() {
            return Err(format!(
                "metallib missing at {path_str} (build.rs AOT failed?)"
            ));
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let library = device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("load metallib: {e}"))?;

        // Metal 4 encode package is required (Metal4-only doctrine).
        let metal4 = try_init_metal4(&device, timestamps).map_err(|err| {
            format!("Metal 4 encode package unavailable ({err}); tessl requires Metal 4")
        })?;

        let has_tensorops = library
            .newFunctionWithName(&NSString::from_str("matmul2d_tensorops_f32"))
            .is_some();

        let recommended = device.recommendedMaxWorkingSetSize();
        let memory_size = probe_system_memory_size();
        let wired_budget = ((recommended as f64) * 0.9) as u64;
        let mem_info = DeviceMemoryInfo {
            recommended_working_set: recommended,
            memory_size,
            wired_budget,
            pool_cache_cap: DEFAULT_POOL_CACHE_BYTES,
        };

        // clippy::arc_with_non_send_sync: `Retained<ProtocolObject<..>>` is not
        // marked Send/Sync by objc2, so this Arc trips the lint. `Rc` is not the
        // fix it suggests: `Arc<GpuRuntime>` is the type in every public
        // signature in this crate, and `self_weak` needs `Weak<GpuRuntime>` to
        // schedule buffer recycling from Drop. Marking the type Send/Sync would
        // be an unsafe assertion about Metal's threading that this crate has not
        // established, so the Arc stays and the lint is silenced here.
        #[allow(clippy::arc_with_non_send_sync)]
        let rt = Arc::new(Self {
            access_busy: Arc::new(AtomicBool::new(false)),
            encode_failed: AtomicBool::new(false),
            device,
            library,
            overlay_libraries: Mutex::new(Vec::new()),
            metal4,
            pipelines: Mutex::new(PipelineCache::new()),
            pool: Mutex::new(BufferPool::new(DEFAULT_POOL_CACHE_BYTES)),
            bump: Mutex::new(None),
            has_tensorops,
            async_encode: Mutex::new(false),
            active_m4: Mutex::new(None),
            last_m4_stamps: Mutex::new(None),
            dispatch_count: Mutex::new(0),
            precision: Mutex::new(PrecisionMode::F32),
            relaxed_precision: Mutex::new(false),
            flash_tensorops: Mutex::new(false),
            residency_dirty: Mutex::new(false),
            pending_cold_recycle: Mutex::new(Vec::new()),
            pending_retirement: Mutex::new(Vec::new()),
            params: Mutex::new(None),
            self_weak: OnceLock::new(),
            memory_info: Mutex::new(mem_info),
        });
        rt.self_weak
            .set(Arc::downgrade(&rt))
            .expect("self_weak is set once, here");
        Ok(rt)
    }

    pub fn has_tensorops(&self) -> bool {
        self.has_tensorops
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Bytes of threadgroup memory a single dispatch may request.
    ///
    /// Kernels that stage an operand tile in `threadgroup` memory size that
    /// request from a runtime dimension — the Q4 GEMV caches the whole `x`
    /// vector, i.e. `cols * 4` bytes. Exceeding the limit is a dispatch-time
    /// failure whose message names neither the kernel nor the dimension, so
    /// callers check against this first. Apple guarantees at least 32 KiB;
    /// current Apple silicon reports more.
    pub fn max_threadgroup_memory(&self) -> usize {
        self.device.maxThreadgroupMemoryLength()
    }

    pub fn set_precision(&self, mode: PrecisionMode) {
        *self.precision.lock().unwrap() = mode;
    }

    pub fn precision(&self) -> PrecisionMode {
        *self.precision.lock().unwrap()
    }

    /// Opt into TensorOps f32 `relaxed_precision` (tf32-class) GEMMs. Ignored when
    /// [`PrecisionMode::Bf16`] (bf16 path takes precedence) or TensorOps is absent.
    pub fn set_relaxed_precision(&self, on: bool) {
        *self.relaxed_precision.lock().unwrap() = on;
    }

    pub fn relaxed_precision(&self) -> bool {
        *self.relaxed_precision.lock().unwrap()
    }

    pub fn set_flash_tensorops(&self, on: bool) {
        *self.flash_tensorops.lock().unwrap() = on;
    }

    pub fn flash_tensorops(&self) -> bool {
        *self.flash_tensorops.lock().unwrap()
    }

    pub fn memory_info(&self) -> DeviceMemoryInfo {
        *self.memory_info.lock().unwrap()
    }

    /// Cap freelist cache bytes (CLI `--pool-cache-mb`).
    pub fn set_pool_cache_cap_bytes(&self, bytes: usize) {
        if let Ok(mut info) = self.memory_info.lock() {
            info.pool_cache_cap = bytes;
        }
        if let Ok(mut pool) = self.pool.lock() {
            pool.set_max_cache_bytes(bytes);
        }
    }

    /// Override wired budget fraction of `recommendedMaxWorkingSetSize` (logged only;
    /// raising the system `iogpu.wired_limit_mb` still requires sysctl).
    pub fn set_wired_fraction(&self, fraction: f64) {
        let frac = fraction.clamp(0.5, 0.95);
        if let Ok(mut info) = self.memory_info.lock() {
            info.wired_budget = ((info.recommended_working_set as f64) * frac) as u64;
        }
    }

    /// Last CounterHeap (t0, t1) from the most recent Metal 4 synchronize, if any.
    pub fn take_metal4_stamps(&self) -> Option<(u64, u64)> {
        self.last_m4_stamps.lock().ok().and_then(|mut g| g.take())
    }

    /// SharedEvent signaled on every Metal 4 commit (cross-crate handoff).
    pub fn shared_event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &*self.metal4.shared_event
    }

    /// Last timeline value this runtime has submitted a signal for.
    pub fn last_signaled_value(&self) -> u64 {
        *self
            .metal4
            .event_value
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// Register a buffer in the Metal 4 residency set (deferred commit).
    pub(crate) fn register_residency(&self, buf: &ProtocolObject<dyn MTLBuffer>) {
        self.register_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(buf));
    }

    /// Register any [`MTLAllocation`] (buffers, ICB, …) in the residency set.
    pub(crate) fn register_allocation(&self, alloc: &ProtocolObject<dyn MTLAllocation>) {
        let m4 = &self.metal4;
        m4.residency.addAllocation(alloc);
        let mut count = m4
            .residency_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *count = count.saturating_add(1);
        drop(count);
        *self
            .residency_dirty
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    }

    /// Mark allocation for removal on next residency commit (after CB complete).
    pub(crate) fn unregister_residency(&self, buf: &ProtocolObject<dyn MTLBuffer>) {
        self.unregister_allocation(ProtocolObject::<dyn MTLAllocation>::from_ref(buf));
    }

    /// Mark any allocation for removal on the next residency commit.
    ///
    /// Callers must only reach this after every command that can reference the
    /// allocation has completed. Last-owner drops use the pending retirement
    /// queue below rather than calling this directly.
    fn unregister_allocation(&self, alloc: &ProtocolObject<dyn MTLAllocation>) {
        let m4 = &self.metal4;
        m4.residency.removeAllocation(alloc);
        let mut count = m4
            .residency_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *count = count.saturating_sub(1);
        drop(count);
        *self
            .residency_dirty
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
    }

    /// Called from [`crate::tensor::PooledBuffer`] Drop for cold temps.
    pub(crate) fn schedule_cold_recycle(
        &self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        nbytes: usize,
    ) {
        let mut q = self
            .pending_cold_recycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.push((buffer, nbytes));
    }

    /// Retire a Hot buffer after all submitted work has completed.
    ///
    /// Unlike Cold/Bump storage, Hot storage never enters the freelist. It must
    /// nevertheless leave the residency set when its final handle drops; the
    /// old early return in `PooledBuffer::drop` made every transient Hot scalar
    /// and weight allocation permanent for the runtime's entire lifetime.
    pub(crate) fn schedule_hot_retirement(&self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let mut q = self
            .pending_retirement
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.push(PendingRetirement::Buffer(buffer));
    }

    /// Lazily create the persistent params workspace.
    pub fn params_buffer(&self) -> Result<(), String> {
        let mut guard = self.params.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            *guard = Some(ParamsBuffer::new(self)?);
        }
        Ok(())
    }

    /// Access the persistent params workspace, creating it on first use.
    pub fn with_params<R>(&self, f: impl FnOnce(&ParamsBuffer) -> R) -> Result<R, String> {
        self.params_buffer()?;
        let guard = self.params.lock().map_err(|e| e.to_string())?;
        let params = guard
            .as_ref()
            .ok_or_else(|| "params buffer missing after init".to_string())?;
        Ok(f(params))
    }

    /// Retire an indirect command buffer after all submitted work has completed.
    pub(crate) fn schedule_icb_retirement(
        &self,
        icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
        argument_tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) {
        let mut q = self
            .pending_retirement
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.push(PendingRetirement::Icb {
            icb,
            argument_tables,
            pipelines,
        });
    }

    /// Retire a device-backed MTLTensor after all submitted work has completed.
    #[cfg(feature = "quant-prep")]
    pub(crate) fn schedule_tensor_retirement(
        &self,
        tensor: Retained<ProtocolObject<dyn MTLTensor>>,
    ) {
        let mut q = self
            .pending_retirement
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        q.push(PendingRetirement::Tensor(tensor));
    }

    /// Weak handle used by allocation owners whose Drop implementations must
    /// not create an `Arc` cycle with the runtime that owns their residency set.
    pub(crate) fn weak_handle(&self) -> Weak<GpuRuntime> {
        // A buffer built on a dangling handle would never be unregistered from
        // residency or recycled, with no diagnostic; refusing loudly is the
        // only acceptable failure, and `new` makes it unreachable.
        self.self_weak
            .get()
            .cloned()
            .expect("GpuRuntime::new sets self_weak before any allocation")
    }

    /// Leak one strong reference to every object an in-flight unretained MTL4
    /// command can reach.
    ///
    /// This is only called after the final runtime owner's bounded completion
    /// wait times out. Apple documents MTL4 command buffers as not retaining
    /// their resource graph, so returning from Drop normally at that point
    /// would permit a GPU use-after-free. The intentionally leaked snapshot is
    /// a fail-safe for a wedged/lost device; a healthy queue takes the normal
    /// completion path and leaks nothing.
    fn leak_in_flight_anchors(&mut self) {
        // `allAllocations` is a copied NSArray and therefore owns strong
        // references to each current (including staged) residency member. Keep
        // it independently of the residency set rather than assuming that the
        // set itself retains its members as an undocumented side effect.
        let resident_allocations = self.metal4.residency.allAllocations();
        std::mem::forget(resident_allocations);

        std::mem::forget(self.device.clone());
        std::mem::forget(self.library.clone());
        for library in mutex_value_mut(&mut self.overlay_libraries).iter() {
            std::mem::forget(library.clone());
        }
        for pipeline in mutex_value_mut(&mut self.pipelines).map.values() {
            std::mem::forget(pipeline.clone());
        }

        std::mem::forget(self.metal4.queue.clone());
        for slot in mutex_value_mut(&mut self.metal4.allocators).iter() {
            std::mem::forget(slot.allocator.clone());
            slot.anchors.leak_clones();
        }
        std::mem::forget(self.metal4.allocator.clone());
        std::mem::forget(self.metal4.command_buffer.clone());
        std::mem::forget(self.metal4.argument_table.table.clone());
        if let Some(counter_heap) = &self.metal4.counter_heap {
            std::mem::forget(counter_heap.clone());
        }
        std::mem::forget(self.metal4.residency.clone());
        std::mem::forget(self.metal4.shared_event.clone());
        std::mem::forget(self.metal4.const_staging.clone());

        if let Some(batch) = mutex_value_mut(&mut self.active_m4).as_ref() {
            if let Some(encoder) = &batch.encoder {
                std::mem::forget(encoder.clone());
            }
            batch.anchors.leak_clones();
        }
        if let Some(bump) = mutex_value_mut(&mut self.bump).as_ref() {
            std::mem::forget(bump.buffer.clone());
        }
        for buffers in mutex_value_mut(&mut self.pool).freelist.values() {
            for buffer in buffers {
                std::mem::forget(buffer.clone());
            }
        }
        for (buffer, _) in mutex_value_mut(&mut self.pending_cold_recycle).iter() {
            std::mem::forget(buffer.clone());
        }
        for allocation in mutex_value_mut(&mut self.pending_retirement).iter() {
            allocation.leak_clone();
        }
    }

    /// After GPU catch-up, remove every allocation whose last logical owner has
    /// dropped. Cold/Bump buffers then enter the freelist; Hot buffers, ICBs,
    /// and device-backed tensors are simply released. This is the only drain
    /// for either class, keeping the completion-before-removal invariant in one
    /// place.
    fn drain_completed_allocations(&self) {
        let cold = std::mem::take(
            &mut *self
                .pending_cold_recycle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let retired = std::mem::take(
            &mut *self
                .pending_retirement
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        if cold.is_empty() && retired.is_empty() {
            return;
        }

        // Keep every retained object alive until after `flush_residency`
        // commits the removals. That avoids relying on the residency set's
        // ownership semantics for an allocation already marked for removal.
        for (buf, _) in &cold {
            self.unregister_residency(buf);
        }
        for allocation in &retired {
            self.unregister_allocation(allocation.allocation());
        }
        self.flush_residency();

        for (buf, _nbytes) in cold {
            self.pool
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .recycle(buf);
        }
        drop(retired);
    }

    /// Commit pending residency adds/removes and request residency (batched).
    pub fn flush_residency(&self) {
        let dirty = *self
            .residency_dirty
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !dirty {
            return;
        }
        crate::infer_trace::on_residency_flush();
        let m4 = &self.metal4;
        m4.residency.commit();
        m4.residency.requestResidency();
        *self
            .residency_dirty
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
        // try_lock: avoid re-entrancy when called under `active_m4`.
        if let Ok(guard) = self.active_m4.try_lock() {
            if guard.as_ref().map(|b| b.cb_open).unwrap_or(false) {
                m4.command_buffer.useResidencySet(&m4.residency);
            }
        }
    }

    /// Enable multi-kernel command buffers (training hot path).
    pub fn set_async_encode(&self, on: bool) -> Result<(), String> {
        if !on {
            self.synchronize()?;
        }
        *self.async_encode.lock().map_err(|e| e.to_string())? = on;
        Ok(())
    }

    pub fn async_encode_enabled(&self) -> bool {
        *self.async_encode.lock().unwrap()
    }

    pub fn take_dispatch_count(&self) -> usize {
        let mut g = self
            .dispatch_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = *g;
        *g = 0;
        n
    }

    /// Register an additional metallib (Gemma kernels, etc.). Pipeline names
    /// must be unique across primary + overlays.
    pub fn add_metallib(&self, path: &Path) -> Result<(), String> {
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-utf8 metallib path: {path:?}"))?;
        if !path.exists() {
            return Err(format!("metallib missing at {path_str}"));
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
        let library = self
            .device
            .newLibraryWithURL_error(&url)
            .map_err(|e| format!("load overlay metallib: {e}"))?;
        let mut libs = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        libs.push(library);
        Ok(())
    }

    /// Resolve (and cache) the pipeline state for a kernel name.
    ///
    /// The DecodeIcb binder-nop flag deliberately does *not* short-circuit this.
    /// It suppresses encoding, not name resolution: a typo'd or removed kernel
    /// must fail here on a replay step exactly as it does on a live one, and
    /// callers read `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup`
    /// off the returned handle, which is only meaningful if it is that kernel's
    /// own pipeline. A cache hit costs one uncontended lock (and no allocation
    /// off the ICB path) — the same lookup live encode pays per dispatch.
    pub fn pipeline(
        &self,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        record_kernel_use(name);
        // Cache hit without holding overlay lock or allocating a key String.
        let icb = crate::decode_icb::icb_pipelines_enabled();
        {
            let cache = self.pipelines.lock().map_err(|e| e.to_string())?;
            if !icb {
                if let Some(p) = cache.map.get(name) {
                    return Ok(p.clone());
                }
            } else {
                let key = format!("icb:{name}");
                if let Some(p) = cache.map.get(&key) {
                    return Ok(p.clone());
                }
            }
        }
        let overlays = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        let mut cache = self.pipelines.lock().map_err(|e| e.to_string())?;
        cache.get_or_create(&self.device, &self.library, &overlays, name)
    }

    /// Snapshot of overlay metallibs (for ICB pipeline construction).
    pub fn overlay_libraries_snapshot(
        &self,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLLibrary>>>, String> {
        let overlays = self.overlay_libraries.lock().map_err(|e| e.to_string())?;
        Ok(overlays.clone())
    }

    pub fn alloc_buffer(&self, nbytes: usize) -> Result<crate::tensor::GpuBuffer, String> {
        self.alloc_buffer_kind(nbytes, BufferKind::Cold)
    }

    pub fn alloc_buffer_hot(&self, nbytes: usize) -> Result<crate::tensor::GpuBuffer, String> {
        self.alloc_buffer_kind(nbytes, BufferKind::Hot)
    }

    pub fn alloc_buffer_kind(
        &self,
        nbytes: usize,
        kind: BufferKind,
    ) -> Result<crate::tensor::GpuBuffer, String> {
        if self.encode_failed.load(Ordering::Acquire) {
            return Err("runtime is poisoned after encode/submit failure; recreate it".into());
        }
        if kind == BufferKind::Cold {
            crate::infer_trace::on_cold_alloc();
        }
        let (buffer, _from_pool) = {
            // The pool lock covers only the pool: `register_residency` sends
            // an Objective-C message and takes two more locks, none of which
            // need the pool held (the recycle path takes them sequentially,
            // never nested).
            let mut pool = self.pool.lock().map_err(|e| e.to_string())?;
            pool.alloc(&self.device, nbytes)?
        };
        // Always (re)register — freelist buffers were removed on recycle.
        self.register_residency(&buffer);
        let weak = self.weak_handle();
        // Same reason as the runtime Arc above: the pooled buffer holds a
        // `Retained<ProtocolObject<dyn MTLBuffer>>`, and `GpuBuffer` is cloned
        // into every `Tensor` view that borrows it.
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(crate::tensor::GpuBuffer {
            inner: Arc::new(crate::tensor::PooledBuffer {
                buffer,
                nbytes,
                kind,
                runtime: weak,
            }),
        })
    }

    pub fn recycle_buffer(&self, buf: crate::tensor::GpuBuffer) {
        // Last Arc Drop schedules cold recycle after CB complete.
        drop(buf);
    }

    /// Ensure a bump slab of at least `capacity` bytes (power-of-two bucketed).
    pub fn ensure_bump(self: &Arc<Self>, capacity: usize) -> Result<(), String> {
        let cap = capacity
            .checked_next_power_of_two()
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| "bump capacity overflow".to_string())?
            .max(256);
        let mut bump = self.bump.lock().map_err(|e| e.to_string())?;
        if let Some(b) = bump.as_ref() {
            if b.capacity >= cap {
                return Ok(());
            }
        }
        // Allocate first: failure preserves the old arena. Replacing its owner
        // releases the old slab only after every outstanding view has dropped.
        let buffer = self.alloc_buffer_kind(cap, BufferKind::Bump)?;
        *bump = Some(BumpState {
            buffer,
            cursor: 0,
            capacity: cap,
        });
        Ok(())
    }

    /// Sub-allocate a zeroed f32 tensor from the bump slab (view with byte_offset).
    pub fn bump_alloc_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.bump_alloc_f32_with(shape, true)
    }

    /// The bump sub-allocation, zeroed or not. `zeroed == false` is for
    /// crate-internal temporaries whose every element a kernel overwrites
    /// before any read; the window then keeps whatever an earlier step left.
    fn bump_alloc_f32_with(
        self: &Arc<Self>,
        shape: &[usize],
        zeroed: bool,
    ) -> Result<crate::tensor::Tensor, String> {
        let _access = self.acquire_access()?;
        let nbytes = crate::tensor::checked_nbytes(shape, crate::tensor::DType::F32)?;
        let mut bump = self.bump.lock().map_err(|e| e.to_string())?;
        let state = bump
            .as_mut()
            .ok_or_else(|| "bump arena not initialized; call ensure_bump first".to_string())?;
        // Align to 16 bytes for TensorOps.
        let align = 16;
        let cursor = (state.cursor + align - 1) & !(align - 1);
        if cursor
            .checked_add(nbytes)
            .is_none_or(|end| end > state.capacity)
        {
            return Err(format!(
                "{BUMP_EXHAUSTED_PREFIX} need {} more bytes (cursor={cursor}, cap={})",
                nbytes, state.capacity
            ));
        }
        let off = cursor;
        state.cursor = cursor + nbytes;
        // Zero the logical window on the host (unified memory).
        if zeroed {
            let ptr = state.buffer.metal().contents().as_ptr() as *mut u8;
            // SAFETY: `off + nbytes == cursor` and the cursor was bounds-checked
            // against the arena's capacity before it advanced, so this window
            // lies inside `state.buffer`. `state` is held under the arena lock,
            // so no other host thread is writing it, and the window is past the
            // previous cursor — bytes no submitted command has been pointed at.
            unsafe {
                std::ptr::write_bytes(ptr.add(off), 0, nbytes);
            }
        }
        Ok(crate::tensor::Tensor {
            buffer: state.buffer.clone(),
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::F32,
            byte_offset: off,
            runtime: Arc::clone(self),
        })
    }

    /// Synchronize and reset the bump cursor. Retained views keep their old slab;
    /// a fresh slab is allocated when resetting would otherwise alias them.
    ///
    /// Fails, rather than panicking, on the same conditions
    /// [`Self::bump_alloc_f32`] fails on: a live host mapping or an open
    /// encoder (`busy`), a runtime poisoned by an earlier failure, or an
    /// allocation failure for the replacement slab.
    pub fn bump_reset(&self) -> Result<(), String> {
        let _access = self.host_access()?;
        let mut bump = self
            .bump
            .lock()
            .map_err(|_| "bump state poisoned".to_string())?;
        if let Some(b) = bump.as_mut() {
            // Keep the old arena alive until its last outstanding view drops.
            if Arc::strong_count(&b.buffer.inner) == 1 {
                b.cursor = 0;
            } else {
                let buffer = self.alloc_buffer_kind(b.capacity, BufferKind::Bump)?;
                b.buffer = buffer;
                b.cursor = 0;
            }
        }
        Ok(())
    }

    pub fn bump_enabled(&self) -> bool {
        self.bump.lock().ok().map(|b| b.is_some()).unwrap_or(false)
    }

    /// Prefer bump slab when initialized; otherwise pool-alloc a fresh tensor.
    ///
    /// Only an exhausted bump arena falls through to the pool. A busy runtime
    /// (a live host mapping, or a call from inside a binder closure) and a
    /// poisoned one are errors the caller has to see; silently serving them
    /// from the pool used to hand out tensors from a runtime that could no
    /// longer run anything.
    pub fn alloc_temp_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_temp_f32_with(shape, true)
    }

    /// [`Self::alloc_temp_f32`] without the host memset, for a temporary whose
    /// every element the caller's kernel writes before any read.
    pub(crate) fn alloc_temp_f32_uninit(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_temp_f32_with(shape, false)
    }

    fn alloc_temp_f32_with(
        self: &Arc<Self>,
        shape: &[usize],
        zeroed: bool,
    ) -> Result<crate::tensor::Tensor, String> {
        if self.bump_enabled() {
            match self.bump_alloc_f32_with(shape, zeroed) {
                Ok(t) => return Ok(t),
                Err(e) if e.starts_with(BUMP_EXHAUSTED_PREFIX) => {}
                Err(e) => return Err(e),
            }
        }
        self.alloc_tensor_f32_kind(shape, BufferKind::Cold, zeroed)
    }

    pub fn alloc_tensor_f32(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f32_kind(shape, BufferKind::Cold, true)
    }

    /// A Cold f32 tensor whose every element the caller overwrites before any
    /// read, so the host memset [`Self::alloc_tensor_f32`] pays is skipped.
    /// A recycled pool buffer's previous contents are visible until then.
    pub(crate) fn alloc_tensor_f32_uninit(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f32_kind(shape, BufferKind::Cold, false)
    }

    /// Persistent weights / grads / optim / EMA — stay in residency (no cold recycle).
    pub fn alloc_tensor_f32_hot(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f32_kind(shape, BufferKind::Hot, true)
    }

    fn alloc_tensor_f32_kind(
        self: &Arc<Self>,
        shape: &[usize],
        kind: BufferKind,
        zeroed: bool,
    ) -> Result<crate::tensor::Tensor, String> {
        let nbytes = crate::tensor::checked_nbytes(shape, crate::tensor::DType::F32)?;
        let buf = self.alloc_buffer_kind(nbytes, kind)?;
        if zeroed {
            unsafe { buf.zero_unsubmitted() };
        }
        Ok(crate::tensor::Tensor {
            buffer: buf,
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::F32,
            byte_offset: 0,
            runtime: Arc::clone(self),
        })
    }

    /// Allocate an IEEE binary16 tensor.
    ///
    /// Two bytes per element like bf16, but not interchangeable with it: the
    /// bit layouts differ, so a buffer written as one and read as the other is
    /// silently wrong rather than merely imprecise.
    pub fn alloc_tensor_f16(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f16_kind(shape, BufferKind::Cold, true)
    }

    pub fn alloc_tensor_f16_hot(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_f16_kind(shape, BufferKind::Hot, true)
    }

    fn alloc_tensor_f16_kind(
        self: &Arc<Self>,
        shape: &[usize],
        kind: BufferKind,
        zeroed: bool,
    ) -> Result<crate::tensor::Tensor, String> {
        let nbytes = crate::tensor::checked_nbytes(shape, crate::tensor::DType::F16)?;
        let buf = self.alloc_buffer_kind(nbytes, kind)?;
        if zeroed {
            unsafe { buf.zero_unsubmitted() };
        }
        Ok(crate::tensor::Tensor {
            buffer: buf,
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::F16,
            byte_offset: 0,
            runtime: Arc::clone(self),
        })
    }

    pub fn alloc_tensor_bf16(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_bf16_kind(shape, BufferKind::Cold, true)
    }

    /// A Cold bf16 tensor whose every element the caller overwrites before any
    /// read; see [`Self::alloc_tensor_f32_uninit`].
    pub(crate) fn alloc_tensor_bf16_uninit(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_bf16_kind(shape, BufferKind::Cold, false)
    }

    pub fn alloc_tensor_bf16_hot(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<crate::tensor::Tensor, String> {
        self.alloc_tensor_bf16_kind(shape, BufferKind::Hot, true)
    }

    fn alloc_tensor_bf16_kind(
        self: &Arc<Self>,
        shape: &[usize],
        kind: BufferKind,
        zeroed: bool,
    ) -> Result<crate::tensor::Tensor, String> {
        let nbytes = crate::tensor::checked_nbytes(shape, crate::tensor::DType::BF16)?;
        let buf = self.alloc_buffer_kind(nbytes, kind)?;
        if zeroed {
            unsafe { buf.zero_unsubmitted() };
        }
        Ok(crate::tensor::Tensor {
            buffer: buf,
            shape: shape.to_vec(),
            dtype: crate::tensor::DType::BF16,
            byte_offset: 0,
            runtime: Arc::clone(self),
        })
    }

    /// Encode a compute pass via [`crate::dispatch::Binder`] (Metal 4).
    ///
    /// Audit 4 P1: one compute encoder is kept open across calls within a CB
    /// (packed dispatches + per-dispatch barriers). Call sites still use one
    /// `with_binder` per op for telemetry (`dispatch_count`).
    pub fn with_binder<F>(&self, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        self.with_binder_barriers(None, f)
    }

    /// Like [`Self::with_binder`], with the binder's auto-barrier mode forced
    /// for this scope only. `None` latches the process-global flag once at
    /// binder construction; `Some(true)` skips per-dispatch auto barriers (the
    /// caller packs explicit RAW barriers — DecodeIcb tape encode). Scope-local
    /// on purpose: flipping the global instead would drop the trailing barrier
    /// of any op another thread encodes concurrently.
    pub fn with_binder_barriers<F>(&self, skip_auto: Option<bool>, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        // DecodeIcb replay prep: skip Metal encode; caller still runs push_u32 /
        // KV host bookkeeping outside the binder closure.
        if crate::decode_icb::binder_encode_nop() {
            return Ok(());
        }
        let _access = self.acquire_access()?;
        // Every encode attempt counts, in both modes and whether or not the
        // closure succeeds: the counter is telemetry for "how many ops reached
        // the encoder", and a failed op reached it.
        *self
            .dispatch_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        self.flush_residency();
        let async_on = self.async_encode_enabled();
        if async_on {
            if let Err(error) = self.encode_into_batch_m4(skip_auto, f) {
                self.encode_failed.store(true, Ordering::Release);
                return Err(error);
            }
            return Ok(());
        }
        let result = self.with_binder_sync(skip_auto, f);
        if result.is_err() {
            self.encode_failed.store(true, Ordering::Release);
        }
        result
    }

    fn ensure_m4_cb_open(&self, batch: &mut Option<ActiveMetal4Batch>) -> Result<(), String> {
        // Do not call flush_residency here — caller may already hold `active_m4`.
        let m4 = &self.metal4;
        let need_begin = match batch.as_ref() {
            Some(b) => !b.cb_open,
            None => true,
        };
        if need_begin {
            let alloc_idx = {
                let mut slots = m4.allocators.lock().map_err(|e| e.to_string())?;
                // Prefer a free allocator so mid-commit never blocks host encode
                // while the peer CB is still executing.
                let free = slots
                    .iter()
                    .position(|s| s.in_flight == 0)
                    .unwrap_or(usize::MAX);
                let i = if free != usize::MAX {
                    free
                } else {
                    // Both in flight — wait for the earlier signal (smaller event).
                    let (i0, v0) = (0usize, slots[0].in_flight);
                    let (i1, v1) = (1usize, slots[1].in_flight);
                    let (i, v) = if v0 <= v1 { (i0, v0) } else { (i1, v1) };
                    if !m4.shared_event.waitUntilSignaledValue_timeoutMS(v, 30_000) {
                        self.encode_failed.store(true, Ordering::Release);
                        return Err("Metal 4 allocator SharedEvent wait timed out".to_string());
                    }
                    // Reset every slot that has completed (event ≥ in_flight).
                    for s in slots.iter_mut() {
                        if s.in_flight != 0 && s.in_flight <= v {
                            s.allocator.reset();
                            s.anchors.clear();
                            s.in_flight = 0;
                        }
                    }
                    i
                };
                if let Ok(mut idx) = m4.active_alloc.lock() {
                    *idx = i;
                }
                i
            };
            let alloc = {
                let slots = m4.allocators.lock().map_err(|e| e.to_string())?;
                slots[alloc_idx].allocator.clone()
            };
            m4.command_buffer.beginCommandBufferWithAllocator(&alloc);
            m4.command_buffer.useResidencySet(&m4.residency);
            let mut stamped = false;
            if let Some(heap) = m4.counter_heap.as_ref() {
                unsafe {
                    m4.command_buffer.writeTimestampIntoHeap_atIndex(heap, 0);
                }
                stamped = true;
            }
            let since = batch.as_ref().map(|b| b.since_commit).unwrap_or(0);
            let total = batch.as_ref().map(|b| b.dispatches).unwrap_or(0);
            // Normally empty: a submitted batch transfers its anchors to the
            // allocator slot before setting `cb_open=false`. Preserve instead
            // of dropping if an error path ever leaves an unsubmitted anchor.
            let anchors = batch
                .as_mut()
                .map(|b| std::mem::take(&mut b.anchors))
                .unwrap_or_default();
            *batch = Some(ActiveMetal4Batch {
                dispatches: total,
                since_commit: since,
                cb_open: true,
                stamped_t0: stamped,
                encoder: None,
                alloc_idx,
                anchors,
                hazard_pending: false,
                arg_table_latched: false,
            });
        }
        Ok(())
    }

    fn encode_into_batch_m4<F>(&self, skip_auto: Option<bool>, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        autoreleasepool(|_| self.encode_into_batch_m4_inner(skip_auto, f))
    }

    fn encode_into_batch_m4_inner<F>(&self, skip_auto: Option<bool>, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        let m4 = &self.metal4;
        // A scope that opens on a nearly full constant arena drains the GPU
        // first; the waiting commit rewinds the arena and the scope gets all
        // of it. A stall, where the old fail-closed policy poisoned the
        // runtime (audit R8; `a_nearly_full_constant_arena_is_drained_before_
        // a_scope_opens`).
        if self.const_arena_free() < CONST_ARENA_SCOPE_RESERVE {
            self.commit_m4(true)?;
        }
        // Clone Retained encoder so we do not hold `active_m4` across `f`
        // (nested with_binder / flush would otherwise deadlock the Mutex).
        let (enc, pending_edge, table_latched) = {
            let mut guard = self.active_m4.lock().map_err(|e| e.to_string())?;
            self.ensure_m4_cb_open(&mut guard)?;
            let batch = guard
                .as_mut()
                .ok_or_else(|| "M4 batch missing".to_string())?;
            if batch.encoder.is_none() {
                let e = m4
                    .command_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| "MTL4 computeCommandEncoder failed".to_string())?;
                e.barrierAfterQueueStages_beforeStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
                batch.encoder = Some(e);
                batch.arg_table_latched = false;
            }
            let pending_edge = std::mem::take(&mut batch.hazard_pending);
            let table_latched = batch.arg_table_latched;
            let enc = batch
                .encoder
                .as_ref()
                .ok_or_else(|| "M4 encoder missing".to_string())?
                .clone();
            (enc, pending_edge, table_latched)
        };
        let (outcome, anchors, hazard_pending, holds_table) = {
            let mut cursor = m4.const_cursor.lock().map_err(|e| e.to_string())?;
            let mut binder = crate::dispatch::Binder::new(
                enc.as_ref(),
                &m4.argument_table.table,
                &m4.const_staging,
                &mut cursor,
                crate::dispatch::BarrierPolicy {
                    skip_auto: skip_auto.unwrap_or_else(crate::ab_flags::hazard_barriers),
                    // A previous scope on this encoder ended with an
                    // unbarriered dispatch (hazard mode). The binder orders it
                    // before this scope's first dispatch — lazily, so a scope
                    // whose first act is its own explicit barrier does not pay
                    // for two. Going through the binder also lands the barrier
                    // in a decode capture as the previous command's
                    // `barrier_after`.
                    pending: pending_edge,
                },
                m4.argument_table.max_buffers as usize,
                self,
            );
            if table_latched {
                binder.assume_persistent_table();
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(&mut binder).and_then(|_| binder.finish())
            }));
            let anchors = binder.take_in_flight_anchors();
            let hazard_pending = binder.hazard_pending();
            let holds_table = binder.holds_persistent_table();
            (outcome, anchors, hazard_pending, holds_table)
        };

        // A closure can fail or panic after it has already emitted Metal
        // commands. Transfer its raw-object retains before propagating either
        // outcome; dropping them here would turn an ordinary Result/panic into
        // a GPU use-after-free if the caller later submitted the partial CB.
        let successful = matches!(&outcome, Ok(Ok(())));
        let hit_mid = {
            let mut guard = self
                .active_m4
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(batch) = guard.as_mut() else {
                for anchor in &anchors {
                    anchor.leak_clone();
                }
                self.encode_failed.store(true, Ordering::Release);
                return Err("M4 batch missing after binder encode".to_string());
            };
            batch.anchors.extend(anchors);
            // Carried even when the closure failed: whatever it dispatched is
            // still in the encoder, and the batch is poisoned separately.
            batch.hazard_pending = hazard_pending;
            batch.arg_table_latched = holds_table;
            if successful {
                batch.dispatches += 1;
                batch.since_commit += 1;
                let threshold = mid_commit_threshold();
                // threshold==0 → mid-commit off (single CB / token). Hard cap
                // avoids unbounded CB growth if a client encodes without sync.
                (threshold > 0 && batch.since_commit >= threshold) || batch.dispatches >= 100_000
            } else {
                false
            }
        };
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                self.encode_failed.store(true, Ordering::Release);
                self.abort_open_batch();
                return Err(e);
            }
            Err(panic) => {
                self.encode_failed.store(true, Ordering::Release);
                self.abort_open_batch();
                std::panic::resume_unwind(panic);
            }
        }
        if hit_mid {
            self.commit_m4(/*wait*/ false)?;
        }
        Ok(())
    }

    /// End an open, uncommitted batch without submitting it.
    ///
    /// A closure that fails or panics after it has already encoded commands
    /// leaves the encoder and command buffer open. The runtime is poisoned, so
    /// nothing will ever commit them, and releasing a Metal 4 encoder or
    /// command buffer mid-recording is not a defined operation. Ending both
    /// here closes the allocator lifecycle cleanly. The batch's retained
    /// anchors drop with it: nothing was submitted, so no GPU work can be
    /// reading them. Also run at final drop for a batch the caller never
    /// committed.
    fn abort_open_batch(&self) {
        let mut guard = self
            .active_m4
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(batch) = guard.as_mut() else {
            return;
        };
        if !batch.cb_open {
            return;
        }
        if let Some(enc) = batch.encoder.take() {
            enc.endEncoding();
        }
        self.metal4.command_buffer.endCommandBuffer();
        batch.cb_open = false;
        *guard = None;
    }

    fn with_binder_sync<F>(&self, skip_auto: Option<bool>, f: F) -> Result<(), String>
    where
        F: FnOnce(&mut crate::dispatch::Binder<'_>) -> Result<(), String>,
    {
        // One autorelease pool per CB commit drains transient ObjC objects from
        // argument-table / encoder traffic that would otherwise accumulate.
        autoreleasepool(|_| {
            // Encode into the async-style batch then wait (keeps timestamps + residency).
            let was_async = *self
                .async_encode
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !was_async {
                *self
                    .async_encode
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            }
            let _restore = if was_async {
                None
            } else {
                Some(AsyncEncodeRestore {
                    mode: &self.async_encode,
                    value: was_async,
                })
            };
            (|| {
                self.encode_into_batch_m4(skip_auto, f)?;
                self.commit_m4(true)
            })()
        })
    }

    /// End encoding and commit without waiting (multi-step / low-sync path).
    pub fn commit(&self, wait: bool) -> Result<(), String> {
        let _access = self.acquire_access()?;
        let result = self.commit_m4(wait);
        if result.is_err() {
            self.encode_failed.store(true, Ordering::Release);
        }
        result
    }

    fn commit_m4(&self, wait: bool) -> Result<(), String> {
        // `with_binder` flushes before opening the callback, but allocation is
        // intentionally legal inside that callback. Such an allocation stages
        // `addAllocation` after the pre-encode flush, so every submission edge
        // must finalize residency again before it takes and closes the batch.
        // Do this before locking `active_m4`: `flush_residency` may re-attach the
        // updated set to the currently open command buffer via `try_lock`.
        self.flush_residency();
        let mut guard = self.active_m4.lock().map_err(|e| e.to_string())?;
        let Some(batch) = guard.as_mut() else {
            if wait {
                // No batch, but the arena may still hold a cursor from a run
                // that ended without one; rewind it once nothing is in flight.
                self.wait_all_allocators()?;
                self.reset_const_arena();
                self.drain_completed_allocations();
            }
            return Ok(());
        };
        if !batch.cb_open {
            if wait {
                // Still wait for any in-flight mid-commits.
                self.wait_all_allocators()?;
                // Reset the const arena, exactly as the cb_open path below does
                // after its own GPU catch-up.
                //
                // This branch used to skip it, and the omission was terminal
                // rather than merely wasteful: with `TESSL_MID_COMMIT` set,
                // `commit_m4(false)` deliberately preserves `const_cursor` and
                // leaves the batch present with `cb_open == false`, so every
                // subsequent `synchronize()` landed here and the cursor only
                // ever grew. Measured: +16 bytes per dispatch, reaching the
                // 16 MiB arena at ~1.05M dispatches, after which `with_binder`
                // fails with "constant arena exhausted" and the runtime is
                // poisoned for the rest of the process.
                //
                // The wait above dominates every in-flight allocator event, so
                // no GPU work can still be reading the arena here. (That
                // measurement predates natural-width alignment: a scalar now
                // costs four bytes, and a scope that opens under
                // `CONST_ARENA_SCOPE_RESERVE` drains first.)
                self.reset_const_arena();
                *guard = None;
                self.drain_completed_allocations();
            }
            return Ok(());
        }
        let m4 = &self.metal4;
        crate::infer_trace::on_commit();

        // Reserve the completion value before changing native CB state so an
        // exhausted counter leaves the open batch and its anchors intact.
        let next = {
            let mut value = m4
                .event_value
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let next = value
                .checked_add(1)
                .ok_or_else(|| "Metal 4 SharedEvent value overflow".to_string())?;
            *value = next;
            next
        };

        // Publish lifetime anchors to the allocator slot before native
        // submission. From this point through the event signal, even a panic or
        // poisoned bookkeeping mutex leaves final Drop able to find and retain
        // the complete object graph.
        {
            let mut slots = m4
                .allocators
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let slot = &mut slots[batch.alloc_idx];
            slot.anchors
                .extend(std::mem::take(&mut batch.anchors).objects);
            slot.in_flight = next;
            *m4.active_alloc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = 1 - batch.alloc_idx;
        }

        // Close packed compute encoder before ending the CB.
        if let Some(enc) = batch.encoder.take() {
            enc.endEncoding();
        }

        let write_t1 = wait && batch.stamped_t0;
        if write_t1 {
            if let Some(heap) = m4.counter_heap.as_ref() {
                unsafe {
                    m4.command_buffer.writeTimestampIntoHeap_atIndex(heap, 1);
                }
            }
        }
        m4.command_buffer.endCommandBuffer();
        // SAFETY: `Retained::as_ptr` yields a pointer to an object this struct
        // owns and keeps alive across the call, so the `NonNull` cannot dangle.
        // `NonNull::new` rejects null rather than assuming it. The commit takes
        // the pointer by value for the duration of the send and does not retain
        // it past that.
        unsafe {
            let mut cb =
                NonNull::new(Retained::as_ptr(&m4.command_buffer)
                    as *mut ProtocolObject<dyn MTL4CommandBuffer>)
                .ok_or_else(|| "null MTL4 command buffer".to_string())?;
            m4.queue
                .commit_count(NonNull::new_unchecked(&mut cb as *mut _), 1);
        }
        batch.cb_open = false;
        batch.since_commit = 0;

        m4.queue.signalEvent_value(
            ProtocolObject::<dyn MTLEvent>::from_ref(&*m4.shared_event),
            next,
        );
        if wait {
            let t0 = std::time::Instant::now();
            if !m4
                .shared_event
                .waitUntilSignaledValue_timeoutMS(next, 30_000)
            {
                self.encode_failed.store(true, Ordering::Release);
                return Err("Metal 4 SharedEvent wait timed out".to_string());
            }
            crate::infer_trace::record_sync_wait(t0);
            self.reset_all_allocators()?;
            if write_t1 {
                if let Some(heap) = m4.counter_heap.as_ref() {
                    if let Ok(stamps) = resolve_two_timestamps(heap) {
                        if let Ok(mut g) = self.last_m4_stamps.lock() {
                            *g = Some(stamps);
                        }
                    }
                }
            }
            // Reset const arena only after GPU catch-up.
            self.reset_const_arena();
            *guard = None;
            drop(guard);
            // Safe to removeAllocation + freelist now that CB completed.
            self.drain_completed_allocations();
        } else {
            // Keep const_cursor; do not reuse offsets until a waiting commit.
            // Next encode opens on the other allocator.
            batch.cb_open = false;
        }
        Ok(())
    }

    /// Bytes of the constant arena not yet written since its last rewind.
    fn const_arena_free(&self) -> usize {
        let cursor = *self
            .metal4
            .const_cursor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.metal4.const_staging.length().saturating_sub(cursor)
    }

    /// Rewind the constant arena. Only once every submitted command buffer
    /// has completed: the GPU reads scalars straight out of it.
    fn reset_const_arena(&self) {
        *self
            .metal4
            .const_cursor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = 0;
    }

    fn wait_all_allocators(&self) -> Result<(), String> {
        let m4 = &self.metal4;
        let slots = m4.allocators.lock().map_err(|e| e.to_string())?;
        let mut max_v = 0u64;
        for s in slots.iter() {
            max_v = max_v.max(s.in_flight);
        }
        drop(slots);
        if max_v == 0 {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        if !m4
            .shared_event
            .waitUntilSignaledValue_timeoutMS(max_v, 30_000)
        {
            self.encode_failed.store(true, Ordering::Release);
            return Err("Metal 4 SharedEvent wait timed out".to_string());
        }
        crate::infer_trace::record_sync_wait(t0);
        self.reset_all_allocators()
    }

    fn reset_all_allocators(&self) -> Result<(), String> {
        let m4 = &self.metal4;
        let mut slots = m4.allocators.lock().map_err(|e| e.to_string())?;
        for s in slots.iter_mut() {
            if s.in_flight != 0 {
                s.allocator.reset();
                s.anchors.clear();
                s.in_flight = 0;
            }
        }
        Ok(())
    }

    /// Commit + wait. Required before host readbacks. SharedEvent covers the
    /// training compute CB on the Metal 4 path (not a stamp-only CB).
    pub fn synchronize(&self) -> Result<(), String> {
        self.commit(true)
    }

    /// True when a timestamp `MTL4CounterHeap` was created with the M4 package.
    pub fn metal4_counter_heap_available(&self) -> bool {
        self.metal4.counter_heap.is_some()
    }
}

impl Drop for GpuRuntime {
    fn drop(&mut self) {
        // A batch that was never committed (a poisoning failure, or a caller
        // that dropped the runtime without synchronizing) must not have its
        // encoder and command buffer released mid-recording.
        self.abort_open_batch();
        let max_event = max_event_value(
            mutex_value_mut(&mut self.metal4.allocators)
                .iter()
                .map(|slot| slot.in_flight),
        );
        if max_event == 0 {
            return;
        }

        if self
            .metal4
            .shared_event
            .waitUntilSignaledValue_timeoutMS(max_event, FINAL_DROP_WAIT_MS)
        {
            // Completion makes allocator command memory and every unretained
            // resource safe to release. Resetting here is not needed for reuse,
            // but closes the allocator lifecycle before its final release.
            for slot in mutex_value_mut(&mut self.metal4.allocators).iter_mut() {
                if slot.in_flight != 0 && slot.in_flight <= max_event {
                    slot.allocator.reset();
                    slot.anchors.clear();
                    slot.in_flight = 0;
                }
            }
            return;
        }

        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr().lock(),
            "tessl: final GpuRuntime drop timed out after {FINAL_DROP_WAIT_MS} ms waiting for \
             Metal event {max_event}; leaking in-flight MTL4 anchors to prevent GPU use-after-free"
        );
        self.leak_in_flight_anchors();
    }
}

fn probe_system_memory_size() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            String::from_utf8(o.stdout)
                .ok()
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(0)
}

fn resolve_two_timestamps(
    heap: &ProtocolObject<dyn MTL4CounterHeap>,
) -> Result<(u64, u64), String> {
    let data: Retained<NSData> = unsafe {
        heap.resolveCounterRange(NSRange {
            location: 0,
            length: 2,
        })
    }
    .ok_or_else(|| "resolveCounterRange returned nil".to_string())?;
    let need = 2 * std::mem::size_of::<MTL4TimestampHeapEntry>();
    let bytes = data.length();
    if bytes < need {
        return Err(format!(
            "timestamp resolve too small: {bytes} bytes (need {need})"
        ));
    }
    let mut buf = vec![0u8; need];
    // SAFETY: `getBytes:length:` copies `need` bytes into the destination, and
    // `buf` is a live, uniquely borrowed allocation of exactly `need` bytes.
    unsafe {
        data.getBytes_length(NonNull::new(buf.as_mut_ptr().cast()).unwrap(), need);
    }
    decode_two_timestamps(&buf)
}

/// Decode two `MTL4TimestampHeapEntry` values out of resolved counter bytes.
///
/// Split out from [`resolve_two_timestamps`] so the byte decode is testable
/// without a device, and so the alignment contract lives in one place: the
/// staging buffer is a `Vec<u8>` (alignment 1), while the entry type is
/// 8-aligned, so every read here goes through `read_unaligned`.
fn decode_two_timestamps(bytes: &[u8]) -> Result<(u64, u64), String> {
    let need = std::mem::size_of::<[MTL4TimestampHeapEntry; 2]>();
    if bytes.len() < need {
        return Err(format!(
            "timestamp decode too small: {} bytes (need {need})",
            bytes.len()
        ));
    }
    // SAFETY: the length check above puts both entries fully inside `bytes`.
    // `read_unaligned` imposes no alignment requirement on the source, which is
    // what makes reading out of a `Vec<u8>` sound; `MTL4TimestampHeapEntry` is
    // a `#[repr(C)]` struct of one `u64`, so every bit pattern is a valid value
    // and the copy it makes is well defined.
    unsafe {
        let ptr = bytes.as_ptr().cast::<MTL4TimestampHeapEntry>();
        let a = std::ptr::read_unaligned(ptr).timestamp;
        let b = std::ptr::read_unaligned(ptr.add(1)).timestamp;
        Ok((a, b))
    }
}

fn try_init_metal4(
    device: &ProtocolObject<dyn MTLDevice>,
    timestamps: bool,
) -> Result<Metal4EncodePackage, String> {
    let queue = device
        .newMTL4CommandQueue()
        .ok_or_else(|| "newMTL4CommandQueue returned nil".to_string())?;
    let allocator_a = device
        .newCommandAllocator()
        .ok_or_else(|| "newCommandAllocator (A) returned nil".to_string())?;
    let allocator_b = device
        .newCommandAllocator()
        .ok_or_else(|| "newCommandAllocator (B) returned nil".to_string())?;
    let cmd = device
        .newCommandBuffer()
        .ok_or_else(|| "newCommandBuffer (MTL4) returned nil".to_string())?;

    let desc = MTL4ArgumentTableDescriptor::new();
    desc.setMaxBufferBindCount(ARGUMENT_TABLE_MAX_BUFFERS as _);
    desc.setMaxTextureBindCount(16);
    desc.setMaxSamplerStateBindCount(8);
    let table = device
        .newArgumentTableWithDescriptor_error(&desc)
        .map_err(|e| format!("newArgumentTable: {e}"))?;

    let counter_heap = if timestamps {
        let heap_desc = MTL4CounterHeapDescriptor::new();
        heap_desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe {
            heap_desc.setCount(64);
        }
        match device.newCounterHeapWithDescriptor_error(&heap_desc) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("[tessl] MTL4CounterHeap unavailable ({e}); timestamps off");
                None
            }
        }
    } else {
        None
    };

    let res_desc = MTLResidencySetDescriptor::new();
    let shared_event = device
        .newSharedEvent()
        .ok_or_else(|| "newSharedEvent returned nil".to_string())?;

    // Multi-slot const arena for batched argument-table encode (16 MiB).
    let const_staging = device
        .newBufferWithLength_options(
            METAL4_CONST_ARENA_BYTES,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| "const_staging buffer alloc failed".to_string())?;

    let residency = device
        .newResidencySetWithDescriptor_error(&res_desc)
        .map_err(|e| format!("newResidencySet: {e}"))?;
    // Const arena is always resident for M4 encode.
    residency.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(
        &*const_staging,
    ));
    residency.commit();
    residency.requestResidency();

    Ok(Metal4EncodePackage {
        queue,
        allocator: allocator_a.clone(),
        allocators: Mutex::new([
            AllocatorSlot {
                allocator: allocator_a,
                in_flight: 0,
                anchors: InFlightAnchors::default(),
            },
            AllocatorSlot {
                allocator: allocator_b,
                in_flight: 0,
                anchors: InFlightAnchors::default(),
            },
        ]),
        active_alloc: Mutex::new(0),
        command_buffer: cmd,
        argument_table: PersistentArgumentTable {
            table,
            max_buffers: ARGUMENT_TABLE_MAX_BUFFERS as u64,
        },
        counter_heap,
        residency,
        shared_event,
        const_staging,
        const_cursor: Mutex::new(0),
        event_value: Mutex::new(0),
        residency_count: Mutex::new(1),
    })
}

pub fn mtl_size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal4_encode_smoke_copy() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 64usize;
        let src = rt.alloc_buffer(n * 4).unwrap();
        let dst = rt.alloc_buffer(n * 4).unwrap();
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i + 1) as f32;
            }
            let q = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n * 4);
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        let width = pipe.threadExecutionWidth();
        let tpt = width.min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })
        .expect("metal4 smoke");
        rt.synchronize().unwrap();
        let out =
            unsafe { std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n) };
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "smoke mismatch at {i}");
        }
        if rt.metal4_counter_heap_available() {
            let stamps = rt.take_metal4_stamps().expect("expected timestamps");
            assert!(stamps.1 >= stamps.0, "timestamps not monotonic: {stamps:?}");
        }
    }

    #[test]
    fn metal4_batched_multi_dispatch_const_arena() {
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();

        let n1 = 16usize;
        let n2 = 24usize;
        let src1 = rt.alloc_buffer(n1 * 4).unwrap();
        let mid = rt.alloc_buffer(n1.max(n2) * 4).unwrap();
        let dst = rt.alloc_buffer(n2 * 4).unwrap();
        unsafe {
            let p = src1.metal().contents().as_ptr() as *mut f32;
            for i in 0..n1 {
                *p.add(i) = (i + 1) as f32;
            }
            let q = mid.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n1.max(n2) * 4);
            let r = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(r as *mut u8, 0, n2 * 4);
        }
        let src2 = rt.alloc_buffer(n2 * 4).unwrap();
        unsafe {
            let p = src2.metal().contents().as_ptr() as *mut f32;
            for i in 0..n2 {
                *p.add(i) = 100.0 + i as f32;
            }
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src1, 0);
            bnd.bind_gpu_buf(&mid, 1);
            bnd.bind_u32(n1 as u32, 2);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(n1, 1, 1));
            Ok(())
        })
        .unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src2, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n2 as u32, 2);
            bnd.dispatch(mtl_size(1, 1, 1), mtl_size(n2, 1, 1));
            Ok(())
        })
        .unwrap();
        rt.synchronize().unwrap();

        let mid_out = unsafe {
            std::slice::from_raw_parts(mid.metal().contents().as_ptr() as *const f32, n1).to_vec()
        };
        let dst_out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n2).to_vec()
        };
        for (i, v) in mid_out.iter().take(n1).enumerate() {
            assert_eq!(*v, (i + 1) as f32, "first copy mismatch at {i}");
        }
        for (i, v) in dst_out.iter().take(n2).enumerate() {
            assert_eq!(*v, 100.0 + i as f32, "second copy mismatch at {i}");
        }
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
        assert!(rt.metal4.const_staging.length() >= METAL4_CONST_ARENA_BYTES);
    }

    #[test]
    fn metal4_arg_table_offset_and_multi_const() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 32usize;
        let pad = 16usize;
        let src = rt.alloc_buffer((pad + n) * 4).expect("src");
        let dst = rt.alloc_buffer(n * 4).expect("dst");
        unsafe {
            let p = src.metal().contents().as_ptr() as *mut f32;
            for i in 0..(pad + n) {
                *p.add(i) = if i < pad { -1.0 } else { (i - pad + 1) as f32 };
            }
            let q = dst.metal().contents().as_ptr() as *mut f32;
            std::ptr::write_bytes(q as *mut u8, 0, n * 4);
        }
        let pipe = rt.pipeline("copy_f32").expect("pipe");
        let width = pipe.threadExecutionWidth();
        let tpt = width.min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf_offset(&src, pad * 4, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })
        .expect("m4 offset dispatch");
        rt.synchronize().unwrap();
        let out = unsafe {
            std::slice::from_raw_parts(dst.metal().contents().as_ptr() as *const f32, n).to_vec()
        };
        for (i, &v) in out.iter().enumerate() {
            let expect = (i + 1) as f32;
            assert!(
                (v - expect).abs() == 0.0,
                "offset bind mismatch at {i}: got {v} want {expect}"
            );
        }
        assert!(rt.metal4.const_staging.length() >= METAL4_CONST_ARENA_BYTES);
        // Multi-const staging via Binder const arena.
        rt.set_async_encode(true).unwrap();
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&dst, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(1, 2);
            bnd.bind_u32(2, 3);
            bnd.bind_f32(3.5, 4);
            // No dispatch needed for arena cursor check — but setArgumentTable
            // only happens on dispatch; just advance cursor via binds then barrier.
            bnd.barrier();
            Ok(())
        })
        .unwrap();
        let cursor = *rt.metal4.const_cursor.lock().unwrap();
        assert!(cursor > 0, "const arena cursor should advance");
        rt.synchronize().unwrap();
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
    }

    /// Fill `buf` with `1, 2, 3, ...` as f32.
    fn fill_ramp(buf: &crate::tensor::GpuBuffer, n: usize) {
        unsafe {
            let p = buf.metal().contents().as_ptr() as *mut f32;
            for i in 0..n {
                *p.add(i) = (i + 1) as f32;
            }
        }
    }

    fn read_f32s(buf: &crate::tensor::GpuBuffer, n: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(buf.metal().contents().as_ptr() as *const f32, n).to_vec()
        }
    }

    /// One `copy_f32` scope over `n` floats with `extra` unread scalars bound
    /// before `n`, so `n` itself lands at a small arena offset.
    fn copy_scope(
        rt: &GpuRuntime,
        src: &crate::tensor::GpuBuffer,
        dst: &crate::tensor::GpuBuffer,
        n: usize,
        extra: usize,
    ) -> Result<(), String> {
        let pipe = rt.pipeline("copy_f32")?;
        let tpt = pipe.threadExecutionWidth().min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(src, 0);
            bnd.bind_gpu_buf(dst, 1);
            for i in 0..extra {
                bnd.bind_u32(0xdead_beef, 3 + i);
            }
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })
    }

    /// Audit R8: a scalar costs the arena four bytes, a pair eight, anything
    /// wider sixteen. Every payload used to be rounded up to sixteen.
    #[test]
    fn constant_arena_charges_natural_width() {
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();
        let pipe = rt.pipeline("copy_f32").expect("pipe");
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_u32(1, 2);
            bnd.bind_u32(2, 3);
            bnd.bind_f32(3.5, 4);
            let _ = bnd.bind_bytes(&[7u8; 8], 5);
            let _ = bnd.bind_bytes(&[9u8; 12], 6);
            bnd.barrier();
            Ok(())
        })
        .unwrap();
        // Three scalars end at 12; the pair aligns to 16 and ends at 24; the
        // 12-byte payload aligns to 32 and ends at 44.
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 44);
        rt.synchronize().unwrap();
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
    }

    /// The GPU reads a constant that natural-width alignment puts at a
    /// four-byte offset. Not a regression test: it is the evidence the
    /// alignment rule in `write_constants` rests on.
    #[test]
    fn a_kernel_reads_a_scalar_at_a_four_byte_offset() {
        let rt = GpuRuntime::new().expect("runtime");
        let n = 48usize;
        let src = rt.alloc_buffer(n * 4).expect("src");
        let dst = rt.alloc_buffer(n * 4).expect("dst");
        fill_ramp(&src, n);
        // Sync mode: the arena is rewound after every scope, so the one unread
        // scalar puts `n` at offset 4 exactly.
        copy_scope(&rt, &src, &dst, n, 1).expect("copy");
        rt.synchronize().unwrap();
        assert_eq!(read_f32s(&dst, n), read_f32s(&src, n));
    }

    /// Audit R8: a scope that opens on a nearly full arena drains the GPU and
    /// starts on a rewound one; it used to fail the bind and poison the
    /// runtime.
    #[test]
    fn a_nearly_full_constant_arena_is_drained_before_a_scope_opens() {
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();
        let n = 64usize;
        let src = rt.alloc_buffer(n * 4).expect("src");
        let mid = rt.alloc_buffer(n * 4).expect("mid");
        let dst = rt.alloc_buffer(n * 4).expect("dst");
        fill_ramp(&src, n);
        // A batch is open with one scope in it; then pretend a long
        // unsynchronized run has used all but eight bytes of the arena.
        copy_scope(&rt, &src, &mid, n, 0).expect("first scope");
        *rt.metal4.const_cursor.lock().unwrap() = rt.metal4.const_staging.length() - 8;
        copy_scope(&rt, &mid, &dst, n, 3).expect("scope on a nearly full arena");
        rt.synchronize().unwrap();
        assert_eq!(read_f32s(&dst, n), read_f32s(&src, n));
        assert_eq!(*rt.metal4.const_cursor.lock().unwrap(), 0);
        // And the runtime is still usable.
        copy_scope(&rt, &src, &dst, n, 0).expect("later scope");
        rt.synchronize().unwrap();
    }

    /// Audit R10c: `setArgumentTable` is issued once per encoder, not once
    /// per scope. Two scopes on one open batch cost one set between them; a
    /// waiting commit ends the encoder and the next scope sets it again.
    #[test]
    fn argument_table_is_set_once_per_encoder_not_once_per_scope() {
        use crate::dispatch::ARG_TABLE_SETS;
        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).unwrap();
        let n = 32usize;
        let src = rt.alloc_buffer(n * 4).expect("src");
        let a = rt.alloc_buffer(n * 4).expect("a");
        let b = rt.alloc_buffer(n * 4).expect("b");
        fill_ramp(&src, n);
        let before = ARG_TABLE_SETS.with(|c| c.get());
        copy_scope(&rt, &src, &a, n, 0).expect("scope 1");
        copy_scope(&rt, &a, &b, n, 0).expect("scope 2");
        assert_eq!(
            ARG_TABLE_SETS.with(|c| c.get()) - before,
            1,
            "one set for two scopes"
        );
        rt.synchronize().unwrap();
        assert_eq!(read_f32s(&b, n), read_f32s(&src, n));
        copy_scope(&rt, &src, &b, n, 0).expect("scope on a new encoder");
        assert_eq!(
            ARG_TABLE_SETS.with(|c| c.get()) - before,
            2,
            "a new encoder sets it again"
        );
        rt.synchronize().unwrap();
    }
}

#[cfg(test)]
mod timestamp_decode_tests {
    use super::*;

    /// Resolved counter bytes are staged in a `Vec<u8>` (alignment 1) while
    /// `MTL4TimestampHeapEntry` is 8-aligned, so the decode must not assume the
    /// staging address happens to be 8-aligned.
    #[test]
    fn decodes_two_entries_from_an_unaligned_buffer() {
        let need = std::mem::size_of::<[MTL4TimestampHeapEntry; 2]>();
        assert_eq!(std::mem::align_of::<MTL4TimestampHeapEntry>(), 8);
        let mut raw = vec![0u8; need + 1];
        // Pick the offset that lands the payload on an odd address whatever the
        // allocator returned.
        let off = usize::from(raw.as_ptr() as usize % 2 == 0);
        raw[off..off + 8].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_ne_bytes());
        raw[off + 8..off + 16].copy_from_slice(&0x0fed_cba9_8765_4321u64.to_ne_bytes());
        let bytes = &raw[off..];
        assert_ne!(
            bytes.as_ptr() as usize % 8,
            0,
            "test did not actually produce a misaligned buffer"
        );
        let (a, b) = decode_two_timestamps(bytes).expect("decode");
        assert_eq!(a, 0x0123_4567_89ab_cdef);
        assert_eq!(b, 0x0fed_cba9_8765_4321);
    }

    #[test]
    fn short_buffer_is_rejected_not_read() {
        let need = std::mem::size_of::<[MTL4TimestampHeapEntry; 2]>();
        let short = vec![0u8; need - 1];
        let err = decode_two_timestamps(&short).expect_err("short buffer must not decode");
        assert!(err.contains("too small"), "{err}");
    }
}

#[cfg(test)]
mod nop_pipeline_tests {
    use super::*;

    /// Env marker set on the isolated child process (see below).
    const CHILD_ENV: &str = "TESSL_BINDER_NOP_CHILD";
    const SELF_NAME: &str =
        "runtime::nop_pipeline_tests::pipeline_under_binder_nop_resolves_the_named_kernel";

    /// binder-nop suppresses *encoding*; it must not suppress name resolution.
    ///
    /// The flag is process-global and turns every `with_binder` into a no-op,
    /// so setting it here would make any GEMM test running in parallel encode
    /// nothing and assert against stale memory. Restoring it on drop is not
    /// enough — the damage happens while it is set. So the parent invocation
    /// only re-runs this same test in a child process, filtered to itself and
    /// single-threaded, where nothing else can be encoding; the child (marked
    /// by `CHILD_ENV`) does the real work.
    #[test]
    fn pipeline_under_binder_nop_resolves_the_named_kernel() {
        if std::env::var_os(CHILD_ENV).is_some() {
            binder_nop_assertions();
            return;
        }
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--test-threads=1", SELF_NAME])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn isolated child test");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(
            out.status.success(),
            "isolated binder-nop child failed:\n{stdout}\n{stderr}"
        );
        // A filter that matched nothing also exits 0 — make that loud.
        assert!(
            stdout.contains("1 passed"),
            "child ran no test (filter out of sync with {SELF_NAME}?):\n{stdout}"
        );
    }

    fn binder_nop_assertions() {
        // Belt and braces inside the child: restore every ICB flag on drop.
        let _flags = crate::decode_icb::IcbFlagsTestGuard::lock();
        let rt = GpuRuntime::new().expect("runtime");
        // Warm the cache the way a replay step is reached: a live encode pass
        // ran first. An empty cache is not what this test is about.
        let live_copy = rt.pipeline("copy_f32").expect("copy_f32 live");
        rt.pipeline("zero_f32").expect("zero_f32 live");
        crate::decode_icb::set_binder_encode_nop(true);

        let err = rt
            .pipeline("this_kernel_does_not_exist_anywhere")
            .expect_err("binder-nop must still reject a kernel that is not in any metallib");
        assert!(err.contains("not found in metallib"), "{err}");

        // Two different kernels must not share one pipeline state: callers read
        // threadExecutionWidth / maxTotalThreadsPerThreadgroup off this handle.
        let copy = rt.pipeline("copy_f32").expect("copy_f32 under binder-nop");
        let zero = rt.pipeline("zero_f32").expect("zero_f32 under binder-nop");
        assert!(
            !std::ptr::eq(Retained::as_ptr(&copy), Retained::as_ptr(&zero)),
            "binder-nop handed two different kernels the same pipeline state"
        );

        // And the handle is the one the live path returns for that name.
        assert!(
            std::ptr::eq(Retained::as_ptr(&copy), Retained::as_ptr(&live_copy)),
            "binder-nop returned a stand-in instead of copy_f32's own pipeline"
        );
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    fn residency_count(rt: &GpuRuntime) -> usize {
        *rt.metal4.residency_count.lock().unwrap()
    }

    #[test]
    fn final_drop_wait_targets_the_latest_in_flight_event() {
        assert_eq!(max_event_value([].into_iter()), 0);
        assert_eq!(max_event_value([0, 4, 2, 9, 0, 7].into_iter()), 9);
        assert_eq!(max_event_value([u64::MAX, 1].into_iter()), u64::MAX);
    }

    #[test]
    fn final_runtime_drop_waits_for_an_async_commit() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let src = rt.alloc_buffer_hot(4).unwrap();
        let dst = rt.alloc_buffer_hot(4).unwrap();
        unsafe {
            *src.metal().contents().as_ptr().cast::<f32>() = 19.25;
            *dst.metal().contents().as_ptr().cast::<f32>() = 0.0;
        }
        let pipe = rt.pipeline("copy_f32").unwrap();
        let pipe_ptr = Retained::as_ptr(&pipe) as *const () as usize;
        let src_ptr = src.metal() as *const _ as *const () as usize;
        crate::dispatch::dispatch_1d(&rt, &pipe, 1, |bnd| {
            bnd.bind_buf(src.metal(), 0, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(1, 2);
        })
        .unwrap();
        drop(pipe);
        drop(src);
        rt.commit(false).unwrap();

        let allocators = rt.metal4.allocators.lock().unwrap();
        let in_flight = allocators
            .iter()
            .find(|slot| slot.in_flight != 0)
            .expect("submitted allocator slot");
        assert!(in_flight.anchors.objects.iter().any(|anchor| matches!(
            anchor,
            InFlightAnchor::Pipeline(value)
                if Retained::as_ptr(value) as *const () as usize == pipe_ptr
        )));
        assert!(in_flight.anchors.objects.iter().any(|anchor| matches!(
            anchor,
            InFlightAnchor::Buffer(value)
                if Retained::as_ptr(value) as *const () as usize == src_ptr
        )));
        drop(allocators);

        // This is the final Arc: GpuBuffer deliberately holds only a Weak
        // runtime. Drop must wait for the queue event because an MTL4 command
        // buffer does not retain `dst` or the other encoded objects itself.
        drop(rt);
        let got = unsafe { *dst.metal().contents().as_ptr().cast::<f32>() };
        assert_eq!(got, 19.25);
    }

    /// MTL4 argument tables store a tensor's resource ID, not an owning
    /// Objective-C reference. The retain lives in the common bind seam, with no
    /// device-created/buffer-backed branch; asserting it here therefore pins the
    /// completion anchor for both ownership modes. The physical buffer-backed
    /// constructor cannot be exercised portably in CI: Apple's
    /// `tensorSizeAndAlignWithDescriptor:` selector SIGSEGVs on this host before
    /// it can return an error, which is the pre-existing reason that selector is
    /// documented as an experimental Phase-2 probe.
    #[cfg(feature = "quant-prep")]
    #[test]
    fn mtl_tensor_bind_anchor_survives_drop_until_async_completion() {
        use crate::mtl_tensor::{alloc_device_tensor, bind_mtl_tensor, QuantDType};

        let rt = GpuRuntime::new().expect("runtime");
        rt.set_async_encode(true).expect("async encode");
        let tensor = alloc_device_tensor(&rt, &[16, 16], QuantDType::Int8).expect("int8 tensor");
        let tensor_ptr =
            tensor.metal() as *const ProtocolObject<dyn MTLTensor> as *const () as usize;

        let src = rt.alloc_buffer_hot(4).expect("copy source");
        let dst = rt.alloc_buffer_hot(4).expect("copy destination");
        unsafe {
            *src.metal().contents().as_ptr().cast::<f32>() = 23.5;
            *dst.metal().contents().as_ptr().cast::<f32>() = 0.0;
        }
        let pipe = rt.pipeline("copy_f32").expect("copy pipeline");
        rt.with_binder(|binder| {
            binder.set_pipeline(&pipe);
            binder.bind_gpu_buf(&src, 0);
            binder.bind_gpu_buf(&dst, 1);
            binder.bind_u32(1, 2);
            bind_mtl_tensor(binder, &tensor, ARGUMENT_TABLE_MAX_BUFFERS - 1)?;
            binder.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            Ok(())
        })
        .expect("encode async command with MTLTensor resource bind");

        drop(tensor);
        {
            let active = rt.active_m4.lock().expect("active batch");
            let anchors = &active.as_ref().expect("open batch").anchors.objects;
            assert!(
                anchors.iter().any(|anchor| matches!(
                    anchor,
                    InFlightAnchor::Tensor(value)
                        if Retained::as_ptr(value) as *const () as usize == tensor_ptr
                )),
                "MTLTensor was not anchored past binder-closure scope"
            );
        }

        rt.commit(false).expect("submit without waiting");
        {
            let allocators = rt.metal4.allocators.lock().expect("allocator slots");
            let submitted = allocators
                .iter()
                .find(|slot| slot.in_flight != 0)
                .expect("submitted allocator slot");
            assert!(
                submitted.anchors.objects.iter().any(|anchor| matches!(
                    anchor,
                    InFlightAnchor::Tensor(value)
                        if Retained::as_ptr(value) as *const () as usize == tensor_ptr
                )),
                "MTLTensor anchor did not transfer to the submitted allocator"
            );
        }

        rt.synchronize().expect("async completion");
        let got = unsafe { *dst.metal().contents().as_ptr().cast::<f32>() };
        assert_eq!(got, 23.5);
        assert!(rt
            .metal4
            .allocators
            .lock()
            .expect("allocator slots after completion")
            .iter()
            .all(|slot| slot.anchors.objects.is_empty()));
    }

    #[test]
    fn last_hot_owner_retires_without_entering_the_freelist() {
        let rt = GpuRuntime::new().unwrap();
        let baseline = residency_count(&rt);
        for _ in 0..64 {
            drop(rt.alloc_buffer_hot(64).unwrap());
        }
        assert_eq!(residency_count(&rt), baseline + 64);
        assert_eq!(rt.pending_retirement.lock().unwrap().len(), 64);

        // With no active command buffer, a waiting synchronize is an immediate
        // completed-work drain. Every transient Hot allocation must leave the
        // residency set, and none may appear in the reusable Cold freelist.
        rt.synchronize().unwrap();
        assert_eq!(residency_count(&rt), baseline);
        assert!(rt.pending_retirement.lock().unwrap().is_empty());
        assert_eq!(rt.pool.lock().unwrap().cached_bytes, 0);
    }

    #[test]
    fn hot_retirement_waits_for_an_async_consumer() {
        let rt = GpuRuntime::new().unwrap();
        let baseline = residency_count(&rt);
        let src = rt.alloc_buffer_hot(4).unwrap();
        let dst = rt.alloc_buffer_hot(4).unwrap();
        unsafe {
            *src.metal().contents().as_ptr().cast::<f32>() = 7.0;
        }
        rt.set_async_encode(true).unwrap();
        let pipe = rt.pipeline("copy_f32").unwrap();
        crate::dispatch::dispatch_1d(&rt, &pipe, 1, |bnd| {
            bnd.bind_gpu_buf(&src, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(1, 2);
        })
        .unwrap();
        drop(src);
        drop(dst);
        assert_eq!(rt.pending_retirement.lock().unwrap().len(), 2);

        // A non-waiting commit deliberately leaves the retirements pinned.
        rt.commit(false).unwrap();
        assert_eq!(residency_count(&rt), baseline + 2);
        assert_eq!(rt.pending_retirement.lock().unwrap().len(), 2);

        // The waiting edge owns removal and release.
        rt.synchronize().unwrap();
        assert_eq!(residency_count(&rt), baseline);
        assert!(rt.pending_retirement.lock().unwrap().is_empty());
    }

    #[test]
    fn callback_panic_poisoning_rejects_reuse() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.with_binder(|_| panic!("injected callback panic"))
        }));
        assert!(result.is_err());
        assert!(rt.commit(true).is_err());
        assert!(rt.with_binder(|_| Ok(())).is_err());
    }

    #[test]
    fn synchronous_callback_panic_restores_mode_and_discards_the_partial_batch() {
        let rt = GpuRuntime::new().unwrap();
        assert!(!rt.async_encode_enabled());
        let pipe = rt.pipeline("copy_f32").unwrap();

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), String> = rt.with_binder(|binder| {
                binder.set_pipeline(&pipe);
                panic!("injected callback panic after pipeline bind")
            });
        }));
        assert!(outcome.is_err());
        assert!(
            !rt.async_encode_enabled(),
            "temporary synchronous encode mode leaked across unwind"
        );

        // The partial batch was never submitted, so nothing on the GPU can
        // reference what it retained: the runtime ends its encoder and
        // command buffer and drops the batch with its anchors, rather than
        // keeping an open encoder alive for a submission that cannot happen.
        assert!(
            rt.active_m4.lock().unwrap().is_none(),
            "a panicked closure must leave no open partial batch behind"
        );
        assert!(
            rt.with_binder(|_| Ok(())).is_err(),
            "runtime stays poisoned"
        );
    }

    #[test]
    fn raw_buffer_rejects_nonresident_allocation_before_anchoring() {
        let rt = GpuRuntime::new().unwrap();
        let raw = rt
            .device
            .newBufferWithLength_options(16, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let error = rt
            .with_binder(|binder| {
                binder.bind_buf(&raw, 0, 0);
                Ok(())
            })
            .expect_err("nonresident raw buffer must fail closed");
        assert!(
            error.contains("not registered"),
            "unexpected error: {error}"
        );
        // The rejection happens before the bind retains anything, and the
        // failed batch is ended and discarded with it, so there is no open
        // batch left to have retained the buffer.
        assert!(
            rt.active_m4.lock().unwrap().is_none(),
            "a rejected raw buffer left an open batch behind"
        );
    }

    #[test]
    fn binder_callback_allocation_is_committed_before_submit() {
        let rt = GpuRuntime::new().expect("runtime");
        let src = rt.alloc_buffer_hot(4).expect("source");
        src.write_f32(&[37.25]);
        let pipe = rt.pipeline("copy_f32").expect("copy pipeline");
        rt.set_async_encode(true).expect("async encode");

        let mut late_dst = None;
        rt.with_binder(|binder| {
            let dst = rt
                .alloc_buffer_hot(4)
                .expect("allocation inside binder callback");
            binder.set_pipeline(&pipe);
            binder.bind_gpu_buf(&src, 0);
            binder.bind_gpu_buf(&dst, 1);
            binder.bind_u32(1, 2);
            binder.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
            late_dst = Some(dst);
            Ok(())
        })
        .expect("encode with late allocation");
        assert!(
            *rt.residency_dirty.lock().unwrap(),
            "the callback allocation did not stage a residency-set addition"
        );

        rt.commit(false).expect("submit late allocation");
        assert!(
            !*rt.residency_dirty.lock().unwrap(),
            "submission left the callback allocation uncommitted in the residency set"
        );
        rt.synchronize().expect("wait for late allocation copy");
        let output = late_dst.expect("callback destination").read_f32();
        assert_eq!(output, vec![37.25]);
    }

    #[test]
    fn poisoned_residency_bookkeeping_still_commits_retirement() {
        let rt = GpuRuntime::new().unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _dirty = rt.residency_dirty.lock().unwrap();
            panic!("poison dirty flag")
        }));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _count = rt.metal4.residency_count.lock().unwrap();
            panic!("poison residency count")
        }));

        let baseline = *rt
            .metal4
            .residency_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let buffer = rt.alloc_buffer_hot(16).unwrap();
        assert_eq!(
            *rt.metal4
                .residency_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            baseline + 1
        );
        drop(buffer);
        rt.synchronize().unwrap();
        assert_eq!(
            *rt.metal4
                .residency_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            baseline
        );
        assert!(
            !*rt.residency_dirty
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            "residency removals remained staged after a poisoned dirty flag"
        );
    }

    #[test]
    fn callback_failure_poisoning_prevents_partial_submission() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        assert!(rt
            .with_binder(|_| Err("injected encode failure".into()))
            .is_err());
        assert!(
            rt.synchronize().is_err(),
            "failed batch was submitted as success"
        );
    }

    #[test]
    fn shared_event_timeout_poison_rejects_further_encode_and_alloc() {
        let rt = GpuRuntime::new().unwrap();
        rt.poison_as_shared_event_timeout_for_test();
        let err = rt
            .with_binder(|_| Ok(()))
            .expect_err("poisoned runtime must refuse encode");
        assert!(
            err.contains("poison"),
            "expected poison refusal after SharedEvent-timeout latch, got {err}"
        );
        let alloc_err = rt.alloc_tensor_f32(&[4]).map(|_| ()).unwrap_err();
        assert!(
            alloc_err.contains("poison"),
            "expected alloc refusal after SharedEvent-timeout latch, got {alloc_err}"
        );
    }

    #[test]
    fn a_failed_callback_closes_the_open_encoder_and_command_buffer() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        assert!(rt
            .with_binder(|_| Err("injected encode failure".into()))
            .is_err());
        let guard = rt.active_m4.lock().unwrap();
        assert!(
            guard.is_none(),
            "the failed batch must be ended and discarded, not left open for \
             Drop to release mid-recording"
        );
    }

    #[test]
    fn oversized_raw_allocations_fail_without_panicking() {
        let rt = GpuRuntime::new().unwrap();
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.alloc_buffer(usize::MAX)));
        assert!(outcome.is_ok(), "allocation arithmetic panicked");
        assert!(outcome.unwrap().is_err());
    }
}
