//! Zero-copy CPU/GPU sharing of one pooled `StorageModeShared` buffer.
//!
//! Every pooled tessl buffer is `StorageModeShared`: the CPU and the GPU see
//! the same bytes. What was missing is a way for *CPU threads* to use those
//! bytes without copying them (`GpuBuffer::try_contents_f32` takes a
//! runtime-wide exclusive lease on the thread that owns the runtime, and
//! `read_f32`/`write_f32` copy). This module is that way.
//!
//! # Two types
//!
//! - [`SharedBuf`] is the `Send + Sync` core: the host pointer, the length, a
//!   retained `MTLBuffer`, two atomic stamps and a lease counter. It carries
//!   the crate's one `unsafe impl Send + Sync` for a Metal buffer wrapper
//!   (audited at the impl). CPU threads use it through [`SharedBuf::read`]
//!   and [`SharedBuf::write`].
//! - [`SharedAlloc`] owns the pooled [`GpuBuffer`] behind it and is what the
//!   encoding thread binds ([`crate::dispatch::Binder::bind_shared_ro`] /
//!   [`bind_shared_rw`](crate::dispatch::Binder::bind_shared_rw)). It is
//!   `!Send` like every `GpuBuffer`, and derefs to the core. A host lease
//!   borrows the core, so a lease cannot outlive the allocation, and the pool
//!   cannot hand the same `MTLBuffer` to another tensor while a lease is live.
//!
//! # The fence protocol
//!
//! Metal does not synchronise buffer access, and a CPU write racing a GPU
//! write leaves the whole buffer undefined, even where the ranges are
//! disjoint (`MTLBuffer.rs:15-18` in objc2-metal 0.3.2). So fencing is
//! whole-buffer:
//!
//! - Every GPU bind is *role-aware* and stamps the buffer with
//!   [`GpuRuntime::pending_commit_value`](crate::runtime::GpuRuntime): the
//!   timeline value of the commit that will carry the command. A read bind
//!   raises `last_use`; a write bind raises `last_use` and `last_write`.
//! - A host **read** lease waits until the GPU has completed `last_write`. A
//!   host **write** lease waits until it has completed `last_use` and is
//!   exclusive. Readers overlap with each other and with GPU reads.
//! - A stamp above what the runtime has *submitted* is not waited on (no
//!   commit exists that would signal it, so the wait could only spend the
//!   whole budget): the lease returns [`HostAccessError::NotSubmitted`], and
//!   the caller flushes through the thread that owns the runtime
//!   (`commit(false)`) and asks again.
//! - A GPU bind against a live host lease is refused, never blocked: a write
//!   bind against any lease, a read bind against a write lease. The check
//!   comes after the stamp (both sequentially consistent), the mirror of the
//!   lease taking its slot before reading the stamp, so one of the two always
//!   sees the other. A refused bind is a caller bug and, like every binder
//!   error, poisons the runtime; [`SharedBuf::check_gpu_access`] is the
//!   non-poisoning pre-flight.
//! - Every fence checks poison first, and waits go through the private
//!   lifetime event (see `runtime/host_fence.rs`), never the forgeable
//!   handoff event.
//!
//! # DecodeIcb replay
//!
//! A captured tape records binds and replays them without running the binder,
//! so a replayed command buffer would skip the stamp above. Rather than
//! re-stamp on replay, a `SharedAlloc` bind during an ICB capture is refused
//! (`bind_shared_*` fails the binder), and the `GpuBuffer` inside a
//! `SharedAlloc` is crate-private, so no other route puts the allocation on a
//! tape. A tape therefore never contains a shared buffer.

use std::any::Any;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};

use ojas_core::{BackendId, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer, OjasError, Timeline};

use crate::runtime::host_fence::HostFence;
use crate::runtime::GpuRuntime;
use crate::tensor::GpuBuffer;

/// Why a host lease or a GPU bind on a [`SharedBuf`] was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostAccessError {
    /// The GPU command that must finish first is still in an open command
    /// buffer: `needed` is its timeline value, `submitted` the last value any
    /// commit has been made for. Flush (`GpuRuntime::commit(false)`) on the
    /// thread that owns the runtime, then retry. Nothing waited.
    NotSubmitted { needed: u64, submitted: u64 },
    /// A conflicting lease is live (or, for a GPU bind, a conflicting host
    /// lease). The message says which.
    Busy(String),
    /// The runtime is poisoned, a wait ran out of budget, or the lifetime
    /// event was not honest. The runtime's own message.
    Runtime(String),
}

impl HostAccessError {
    /// Whether the caller should flush and retry.
    pub fn is_not_submitted(&self) -> bool {
        matches!(self, Self::NotSubmitted { .. })
    }
}

impl fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSubmitted { needed, submitted } => write!(
                f,
                "shared buffer: timeline value {needed} is not submitted (last submitted {submitted}); \
                 flush with commit(false) and retry"
            ),
            Self::Busy(why) => write!(f, "shared buffer busy: {why}"),
            Self::Runtime(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for HostAccessError {}

/// How a GPU command uses a [`SharedBuf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuRole {
    /// The kernel only reads the buffer.
    Read,
    /// The kernel writes (or reads and writes) the buffer.
    Write,
}

/// The `Send + Sync` core of a shared buffer. See the module docs.
///
/// There is no public constructor and the type is not `Clone`: it is reached
/// by `&` through [`SharedAlloc`], which is what keeps every lease inside the
/// allocation's life.
/// The `Send + Sync` core of a shared buffer. See the module docs.
#[derive(Clone)]
pub struct SharedBuf {
    pub(crate) inner: Arc<SharedBufInner>,
}

pub(crate) struct SharedBufInner {
    /// `MTLBuffer::contents()` read once at construction. The address is fixed
    /// for the buffer's life.
    ptr: NonNull<u8>,
    /// The logical length in bytes (at most the `MTLBuffer`'s length).
    len: usize,
    /// `MTLBuffer::gpuAddress()` read once at construction.
    gpu_addr: u64,
    /// Keeps the memory behind `ptr` mapped. Never messaged after
    /// construction; its release on drop is the only Objective-C message.
    _mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Retains the underlying pooled buffer so dropping a `SharedAlloc` does
    /// not recycle the memory while a `SharedBuf` or `HostLease` is still live.
    _pooled: Option<Arc<crate::tensor::PooledBuffer>>,
    /// Highest commit value of any GPU command that reads or writes the buffer.
    last_use: AtomicU64,
    /// Highest commit value of any GPU command that writes the buffer.
    last_write: AtomicU64,
    /// `0` free, `n > 0` that many read leases, `-1` one write lease.
    lease: AtomicI32,
    fence: HostFence,
}

// SAFETY: this is the one place the crate asserts thread-safety for a Metal
// buffer wrapper (decision D6 of the unification plan). objc2-metal marks
// `MTLEvent` `Send + Sync` (`MTLEvent.rs:14`) but declares `MTLBuffer` with no
// such supertrait (`MTLBuffer.rs:23`), because it cannot know per protocol;
// that is the only reason `Retained<ProtocolObject<dyn MTLBuffer>>` is not
// `Send`/`Sync`. Three facts discharge it:
//
// 1. No Objective-C message reaches the retained `MTLBuffer` after
//    construction except its release on drop, which is an atomic reference
//    count decrement that Objective-C permits from any thread. `contents()`
//    was read once into `ptr` and `length()` is not read again; the GPU
//    address and every bind go through the owning `GpuBuffer` on the encoding
//    thread, never through this struct.
// 2. The pointee is `MTLStorageMode.shared` memory: Apple documents it as
//    visible to CPU and GPU at one fixed address for the buffer's life, and
//    the buffer is kept alive by `_mtl` (and by `_pooled` which keeps the
//    pool from recycling it). Apple states no further thread-safety for `MTLBuffer`;
//    what Metal leaves to the caller is exactly the CPU/GPU ordering that the
//    stamps, the lease counter and the lifetime-event fence below enforce
//    (`MTLBuffer.rs:15-18`).
// 3. Rust-side data races on the bytes are excluded by `lease`: any number of
//    `&[u8]` readers or exactly one `&mut [u8]` writer, and a GPU bind is
//    refused while a conflicting lease is live. The remaining fields are
//    atomics and a `HostFence`, which is `Send + Sync` without `unsafe`
//    (asserted in `runtime/host_fence.rs`).
unsafe impl Send for SharedBufInner {}
// SAFETY: as `Send`; every `&self` method touches only atomics, the fence, or
// bytes guarded by `lease`.
unsafe impl Sync for SharedBufInner {}

impl SharedBuf {
    pub(crate) fn new(gpu: &GpuBuffer, len: usize, fence: HostFence) -> Result<Self, String> {
        let mtl = &gpu.inner.buffer;
        let mode = mtl.storageMode();
        if mode != MTLStorageMode::Shared {
            return Err(format!("shared buffer needs StorageModeShared storage, got {mode:?}"));
        }
        if len == 0 || len > gpu.nbytes() {
            return Err(format!(
                "shared buffer length {len} is outside the allocation ({} bytes)",
                gpu.nbytes()
            ));
        }
        let inner = SharedBufInner {
            ptr: mtl.contents().cast::<u8>(),
            len,
            gpu_addr: gpu.inner.gpu_addr,
            _mtl: mtl.clone(),
            _pooled: Some(Arc::clone(&gpu.inner)),
            last_use: AtomicU64::new(0),
            last_write: AtomicU64::new(0),
            lease: AtomicI32::new(0),
            fence,
        };
        Ok(Self { inner: Arc::new(inner) })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Always `false`: a shared buffer has at least one byte.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Highest commit value of any GPU command that read or wrote the buffer
    /// (`0` if none has).
    pub fn last_use(&self) -> u64 {
        self.inner.last_use.load(Ordering::SeqCst)
    }

    /// Highest commit value of any GPU command that wrote the buffer.
    pub fn last_write(&self) -> u64 {
        self.inner.last_write.load(Ordering::SeqCst)
    }

    /// A read lease: any number may be live at once, alongside GPU reads.
    ///
    /// Waits until the GPU has completed every command that wrote the buffer.
    /// Refused with [`HostAccessError::Busy`] while a write lease is live,
    /// with [`HostAccessError::NotSubmitted`] when a writer is still in an
    /// open command buffer, and with [`HostAccessError::Runtime`] when the
    /// runtime is poisoned. A refused call holds no lease.
    pub fn read(&self) -> Result<HostRead<'_>, HostAccessError> {
        self.inner
            .fence
            .ensure_not_poisoned()
            .map_err(HostAccessError::Runtime)?;
        let mut cur = self.inner.lease.load(Ordering::SeqCst);
        loop {
            if cur < 0 {
                return Err(HostAccessError::Busy("a host write lease is live".into()));
            }
            if cur == i32::MAX {
                return Err(HostAccessError::Busy("too many host read leases".into()));
            }
            match self
                .inner
                .lease
                .compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        // The slot is taken before the stamp is read; the guard releases it on
        // every error below.
        let lease = HostRead { buf: self };
        self.inner
            .fence
            .wait_stamp(self.inner.last_write.load(Ordering::SeqCst))?;
        Ok(lease)
    }

    /// The exclusive write lease.
    ///
    /// Waits until the GPU has completed every command that read or wrote the
    /// buffer. Refused with [`HostAccessError::Busy`] while any lease is live,
    /// and with the same `NotSubmitted` / `Runtime` errors as [`Self::read`].
    /// A refused call holds no lease.
    pub fn write(&self) -> Result<HostWrite<'_>, HostAccessError> {
        self.inner
            .fence
            .ensure_not_poisoned()
            .map_err(HostAccessError::Runtime)?;
        if self
            .inner
            .lease
            .compare_exchange(0, -1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(HostAccessError::Busy("a host lease is live".into()));
        }
        let lease = HostWrite { buf: self };
        self.inner
            .fence
            .wait_stamp(self.inner.last_use.load(Ordering::SeqCst))?;
        Ok(lease)
    }

    /// Acquire a boxed host read lease for `MappedBuffer`.
    fn acquire_read(&self) -> Result<SharedHostReadLease, FenceError> {
        self.inner
            .fence
            .ensure_not_poisoned()
            .map_err(|_| FenceError::Poisoned)?;
        let mut cur = self.inner.lease.load(Ordering::SeqCst);
        loop {
            if cur < 0 {
                return Err(FenceError::Busy {
                    detail: "a host write lease is live".into(),
                });
            }
            if cur == i32::MAX {
                return Err(FenceError::Busy {
                    detail: "too many host read leases".into(),
                });
            }
            match self
                .inner
                .lease
                .compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        let stamp = self.inner.last_write.load(Ordering::SeqCst);
        if let Err(e) = self.inner.fence.wait_stamp(stamp) {
            self.inner.lease.fetch_sub(1, Ordering::SeqCst);
            return Err(map_host_access_error(e, stamp, &self.inner.fence));
        }
        Ok(SharedHostReadLease {
            inner: Arc::clone(&self.inner),
        })
    }

    /// Acquire a boxed host write lease for `MappedBuffer`.
    fn acquire_write(&self) -> Result<SharedHostWriteLease, FenceError> {
        self.inner
            .fence
            .ensure_not_poisoned()
            .map_err(|_| FenceError::Poisoned)?;
        if self
            .inner
            .lease
            .compare_exchange(0, -1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(FenceError::Busy {
                detail: "a host lease is live".into(),
            });
        }
        let stamp = self.inner.last_use.load(Ordering::SeqCst);
        if let Err(e) = self.inner.fence.wait_stamp(stamp) {
            self.inner.lease.store(0, Ordering::SeqCst);
            return Err(map_host_access_error(e, stamp, &self.inner.fence));
        }
        Ok(SharedHostWriteLease {
            inner: Arc::clone(&self.inner),
        })
    }

    /// Whether a GPU command may use the buffer in `role` right now, without
    /// stamping or poisoning anything. Advisory: a lease can be taken the
    /// next instant. The bind itself is the authoritative check.
    pub fn check_gpu_access(&self, role: GpuRole) -> Result<(), HostAccessError> {
        Self::gpu_access(self.inner.lease.load(Ordering::SeqCst), role)
    }

    fn gpu_access(lease: i32, role: GpuRole) -> Result<(), HostAccessError> {
        match (role, lease) {
            (GpuRole::Read, l) if l >= 0 => Ok(()),
            (GpuRole::Write, 0) => Ok(()),
            (GpuRole::Read, _) => Err(HostAccessError::Busy(
                "a GPU read bind was refused: a host write lease is live".into(),
            )),
            (GpuRole::Write, _) => Err(HostAccessError::Busy(
                "a GPU write bind was refused: a host lease is live".into(),
            )),
        }
    }

    /// Record a GPU use at commit value `commit` and refuse it if a host lease
    /// conflicts. Stamps first, then checks (both sequentially consistent):
    /// the mirror of a lease taking its slot first and reading the stamp
    /// second. A refused bind keeps its stamp; the stamp only ever makes a
    /// later lease wait for a commit that is still to come.
    pub(crate) fn stamp_gpu(&self, commit: u64, role: GpuRole) -> Result<(), HostAccessError> {
        self.inner.last_use.fetch_max(commit, Ordering::SeqCst);
        if role == GpuRole::Write {
            self.inner.last_write.fetch_max(commit, Ordering::SeqCst);
        }
        Self::gpu_access(self.inner.lease.load(Ordering::SeqCst), role)
    }

    /// The bytes. Callers hold a lease.
    pub(crate) fn bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    /// The bytes, mutably. Callers hold the write lease.
    pub(crate) fn bytes_mut(&self) -> &mut [u8] {
        self.inner.bytes_mut()
    }

    /// The address of the first byte, for the zero-copy checks.
    #[doc(hidden)]
    pub fn host_addr(&self) -> usize {
        self.inner.ptr.as_ptr() as usize
    }
}

impl SharedBufInner {
    fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes while `_mtl` is retained, and
        // a caller holds a read or write lease, so no `&mut` to these bytes
        // exists and the fence has put every GPU writer behind us.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[allow(clippy::mut_from_ref)]
    fn bytes_mut(&self) -> &mut [u8] {
        // SAFETY: as `bytes`, and the caller holds the one write lease
        // (`lease == -1`), so no other reference to these bytes exists on any
        // thread, and the fence has put every GPU reader and writer behind us.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

mod sealed {
    pub trait Word: Copy {}
    impl Word for f32 {}
    impl Word for u32 {}
}

/// `bytes` viewed as `T`, if aligned and a whole number of elements.
fn words<T: sealed::Word>(bytes: &[u8]) -> Option<&[T]> {
    // SAFETY: `T` is `f32` or `u32` (sealed), for which every bit pattern is a
    // valid value; `align_to` splits off any misaligned head and tail.
    let (head, mid, tail) = unsafe { bytes.align_to::<T>() };
    (head.is_empty() && tail.is_empty()).then_some(mid)
}

/// [`words`], mutably.
fn words_mut<T: sealed::Word>(bytes: &mut [u8]) -> Option<&mut [T]> {
    // SAFETY: as `words`.
    let (head, mid, tail) = unsafe { bytes.align_to_mut::<T>() };
    (head.is_empty() && tail.is_empty()).then_some(mid)
}

/// A shared read lease: a `&[u8]` over the buffer's bytes, with no copy.
pub struct HostRead<'a> {
    buf: &'a SharedBuf,
}

impl HostRead<'_> {
    /// The bytes as `f32`, if aligned and a whole number of words.
    pub fn as_f32(&self) -> Option<&[f32]> {
        words(self.buf.bytes())
    }

    /// The bytes as `u32`, if aligned and a whole number of words.
    pub fn as_u32(&self) -> Option<&[u32]> {
        words(self.buf.bytes())
    }
}

impl Deref for HostRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.buf.bytes()
    }
}

impl Drop for HostRead<'_> {
    fn drop(&mut self) {
        self.buf.inner.lease.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The exclusive write lease: a `&mut [u8]` over the buffer's bytes, with no
/// copy. Split it with `chunks_mut` to fill it from several threads.
pub struct HostWrite<'a> {
    buf: &'a SharedBuf,
}

impl HostWrite<'_> {
    /// The bytes as `f32`, if aligned and a whole number of words.
    pub fn as_f32(&self) -> Option<&[f32]> {
        words(self.buf.bytes())
    }

    /// The bytes as mutable `f32`, if aligned and a whole number of words.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        words_mut(self.buf.bytes_mut())
    }

    /// The bytes as `u32`, if aligned and a whole number of words.
    pub fn as_u32(&self) -> Option<&[u32]> {
        words(self.buf.bytes())
    }

    /// The bytes as mutable `u32`, if aligned and a whole number of words.
    pub fn as_u32_mut(&mut self) -> Option<&mut [u32]> {
        words_mut(self.buf.bytes_mut())
    }
}

impl Deref for HostWrite<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.buf.bytes()
    }
}

impl DerefMut for HostWrite<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.buf.bytes_mut()
    }
}

impl Drop for HostWrite<'_> {
    fn drop(&mut self) {
        self.buf.inner.lease.store(0, Ordering::SeqCst);
    }
}

fn map_host_access_error(e: HostAccessError, stamp: u64, fence: &HostFence) -> FenceError {
    match e {
        HostAccessError::NotSubmitted { needed, submitted } => FenceError::NotSubmitted {
            value: needed,
            submitted,
        },
        HostAccessError::Busy(detail) => FenceError::Busy { detail },
        HostAccessError::Runtime(why) => {
            if why.contains("poisoned") || why.contains("DeviceLost") {
                FenceError::Poisoned
            } else if why.contains("timeout") || why.contains("exceeded") {
                FenceError::TimedOut {
                    value: stamp,
                    completed: fence.checked_value().unwrap_or(0),
                }
            } else {
                FenceError::Refused(OjasError::Backend {
                    id: BackendId::Metal,
                    detail: why,
                })
            }
        }
    }
}

/// A boxed, owning host read lease over a [`SharedBuf`].
pub struct SharedHostReadLease {
    inner: Arc<SharedBufInner>,
}

impl HostLease for SharedHostReadLease {
    fn as_bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_f32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32(&self) -> Result<&[u32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_u32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl Drop for SharedHostReadLease {
    fn drop(&mut self) {
        self.inner.lease.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A boxed, owning exclusive host write lease over a [`SharedBuf`].
pub struct SharedHostWriteLease {
    inner: Arc<SharedBufInner>,
}

impl HostLease for SharedHostWriteLease {
    fn as_bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_f32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32(&self) -> Result<&[u32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_u32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl HostLeaseMut for SharedHostWriteLease {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.bytes_mut()
    }

    fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError> {
        words_mut(self.inner.bytes_mut()).ok_or_else(|| OjasError::Shape {
            op: "HostLeaseMut::as_f32_mut",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32_mut(&mut self) -> Result<&mut [u32], OjasError> {
        words_mut(self.inner.bytes_mut()).ok_or_else(|| OjasError::Shape {
            op: "HostLeaseMut::as_u32_mut",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl Drop for SharedHostWriteLease {
    fn drop(&mut self) {
        self.inner.lease.store(0, Ordering::SeqCst);
    }
}

impl fmt::Debug for SharedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBuf")
            .field("len", &self.inner.len)
            .field("gpu_addr", &self.inner.gpu_addr)
            .field("last_use", &self.last_use())
            .field("last_write", &self.last_write())
            .field("lease", &self.inner.lease.load(Ordering::SeqCst))
            .finish()
    }
}

impl DeviceBuffer for SharedBuf {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.inner.len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        const OP: &str = "SharedBuf::read_bytes";
        let end = offset.checked_add(len).ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "offset + len overflows".into(),
        })?;
        if end > self.inner.len {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("range {offset}..{end} exceeds buffer length {}", self.inner.len),
            });
        }
        let lease = self.acquire_read().map_err(|e| e.into_ojas(BackendId::Metal))?;
        Ok(lease.as_bytes()[offset..end].to_vec())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for SharedBuf {
    fn host_ptr(&self) -> *mut u8 {
        self.inner.ptr.as_ptr()
    }

    fn gpu_address(&self) -> u64 {
        self.inner.gpu_addr
    }

    fn timeline(&self) -> &dyn Timeline {
        &self.inner.fence
    }

    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        let lease = self.acquire_read()?;
        Ok(Box::new(lease))
    }

    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        let lease = self.acquire_write()?;
        Ok(Box::new(lease))
    }
}

/// A pooled shared buffer and its [`SharedBuf`] core. See the module docs.
///
/// Field order matters only for tidiness: the core is dropped first, then the
/// pooled buffer, whose drop schedules its recycle after the next commit
/// completes, as for any `GpuBuffer`.
pub struct SharedAlloc {
    core: SharedBuf,
    gpu: GpuBuffer,
}

impl SharedAlloc {
    /// Allocate `nbytes` (at least one) from `rt`'s pool.
    pub fn new(rt: &GpuRuntime, nbytes: usize) -> Result<Self, String> {
        let gpu = rt.alloc_buffer(nbytes.max(1))?;
        let core = SharedBuf::new(&gpu, nbytes.max(1), rt.host_fence().clone())?;
        Ok(Self { core, gpu })
    }

    /// Convert this allocation into an owning `SharedBuf` (implements `MappedBuffer`),
    /// keeping the pool allocation alive as long as the `SharedBuf` is alive.
    pub fn into_shared_buf(self) -> SharedBuf {
        self.core
    }

    /// Access the underlying `SharedBuf` (implements `MappedBuffer`).
    pub fn shared_buf(&self) -> SharedBuf {
        self.core.clone()
    }

    /// The pooled buffer, for the binders. Crate-private on purpose: any other
    /// route would let the allocation onto a DecodeIcb tape unstamped.
    pub(crate) fn gpu_buffer(&self) -> &GpuBuffer {
        &self.gpu
    }
}

impl Deref for SharedAlloc {
    type Target = SharedBuf;
    fn deref(&self) -> &SharedBuf {
        &self.core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
    use objc2_metal::MTLSharedEvent;
    use std::time::{Duration, Instant};

    const N: usize = 1 << 18; // 1 MiB of u32

    fn bits(i: usize) -> u32 {
        0x3f80_0000 | (((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 41) as u32)
    }

    fn fill_from_threads(words: &mut [u32], salt: u32) {
        let chunk = words.len().div_ceil(4);
        std::thread::scope(|s| {
            for (t, c) in words.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (j, w) in c.iter_mut().enumerate() {
                        *w = bits(t * chunk + j) ^ salt;
                    }
                });
            }
        });
    }

    /// `dst = src` over `n` words, `src` read-only shared, `dst` plain.
    fn copy_from_shared(rt: &GpuRuntime, src: &SharedAlloc, dst: &GpuBuffer, n: usize) -> Result<(), String> {
        let p = rt.pipeline("copy_u32")?;
        dispatch_1d(rt, &p, n, |bnd| {
            bnd.bind_shared_ro(src, 0);
            set_gpu_buf(bnd, dst, 1);
            set_u32(bnd, n as u32, 2);
        })
    }

    /// `dst = src` over `n` words, `src` plain, `dst` written through the shared buffer.
    fn copy_into_shared(rt: &GpuRuntime, src: &GpuBuffer, dst: &SharedAlloc, n: usize) -> Result<(), String> {
        let p = rt.pipeline("copy_u32")?;
        dispatch_1d(rt, &p, n, |bnd| {
            set_gpu_buf(bnd, src, 0);
            bnd.bind_shared_rw(dst, 1);
            set_u32(bnd, n as u32, 2);
        })
    }

    #[test]
    fn cpu_threads_write_and_a_kernel_reads_the_same_bytes_bit_exact_with_no_copy() {
        let rt = GpuRuntime::new().unwrap();
        let shared = SharedAlloc::new(&rt, N * 4).unwrap();
        {
            let mut w = shared.write().unwrap();
            assert_eq!(
                w.as_ptr() as usize,
                shared.gpu_buffer().metal().contents().as_ptr() as usize
            );
            fill_from_threads(w.as_u32_mut().unwrap(), 0);
        }
        let dst = rt.alloc_buffer(N * 4).unwrap();
        copy_from_shared(&rt, &shared, &dst, N).unwrap();
        rt.synchronize().unwrap();
        let expect: Vec<u32> = (0..N).map(bits).collect();
        assert_eq!(dst.read_u32(), expect, "the kernel read what the CPU threads wrote");

        // And the other way: a kernel writes the shared buffer, CPU threads read it.
        let src = rt
            .alloc_buffer_from_u32(&expect.iter().map(|b| b ^ 0x5a5a_5a5a).collect::<Vec<_>>())
            .unwrap();
        copy_into_shared(&rt, &src, &shared, N).unwrap();
        rt.synchronize().unwrap();
        let r = shared.read().unwrap();
        assert_eq!(
            r.as_ptr() as usize,
            shared.host_addr(),
            "a lease is the buffer itself, not a copy"
        );
        let sum_bad: usize = std::thread::scope(|s| {
            let words = r.as_u32().unwrap();
            let chunk = words.len().div_ceil(4);
            let hs: Vec<_> = words
                .chunks(chunk)
                .enumerate()
                .map(|(t, c)| {
                    s.spawn(move || {
                        c.iter()
                            .enumerate()
                            .filter(|(j, w)| **w != bits(t * chunk + j) ^ 0x5a5a_5a5a)
                            .count()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        assert_eq!(sum_bad, 0);
    }

    #[test]
    fn stamps_are_the_pending_commit_value_by_role() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let other = rt.alloc_buffer(64).unwrap();
        assert_eq!((shared.last_use(), shared.last_write()), (0, 0));

        // Both dispatches share one command buffer: the mid-commit threshold
        // defaults to off (`TESSL_MID_COMMIT` unset), so no commit falls between them.
        let next = rt.last_signaled_value() + 1;
        copy_from_shared(&rt, &shared, &other, 16).unwrap();
        assert_eq!(
            (shared.last_use(), shared.last_write()),
            (next, 0),
            "a read bind stamps last_use only"
        );

        copy_into_shared(&rt, &other, &shared, 16).unwrap();
        assert_eq!(
            (shared.last_use(), shared.last_write()),
            (next, next),
            "a write bind stamps both"
        );
        rt.synchronize().unwrap();
    }

    #[test]
    fn readers_overlap_each_other_and_gpu_reads() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, N * 4).unwrap();
        shared.write().unwrap().as_u32_mut().unwrap().fill(7);
        let dst = rt.alloc_buffer(N * 4).unwrap();

        let a = shared.read().unwrap();
        let b = shared.read().unwrap();
        // A GPU read bind is fine while two readers hold the buffer.
        copy_from_shared(&rt, &shared, &dst, N).unwrap();
        assert!(shared.check_gpu_access(GpuRole::Read).is_ok());
        assert_eq!(a.as_u32().unwrap()[N - 1], 7);
        assert_eq!(b.as_u32().unwrap()[0], 7);
        drop((a, b));
        rt.synchronize().unwrap();
        assert!(dst.read_u32().iter().all(|&w| w == 7));
    }

    #[test]
    fn a_writer_is_exclusive() {
        let rt = GpuRuntime::new().unwrap();
        let shared = SharedAlloc::new(&rt, 256).unwrap();
        let w = shared.write().unwrap();
        assert!(matches!(shared.read(), Err(HostAccessError::Busy(_))));
        assert!(matches!(shared.write(), Err(HostAccessError::Busy(_))));
        drop(w);
        let r = shared.read().unwrap();
        assert!(
            matches!(shared.write(), Err(HostAccessError::Busy(_))),
            "a reader excludes a writer"
        );
        drop(r);
        // Refused calls left no lease behind.
        drop(shared.write().unwrap());
        drop(shared.read().unwrap());
    }

    #[test]
    fn a_stamp_above_submitted_returns_not_submitted_without_waiting() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, N * 4).unwrap();
        let src = rt.alloc_buffer(N * 4).unwrap();
        copy_into_shared(&rt, &src, &shared, N).unwrap();
        let needed = shared.last_write();
        assert!(
            needed > rt.last_signaled_value(),
            "the writer is in an open command buffer"
        );

        let t = Instant::now();
        let err = shared.read().err().expect("must refuse");
        assert_eq!(
            err,
            HostAccessError::NotSubmitted {
                needed,
                submitted: rt.last_signaled_value()
            }
        );
        assert!(t.elapsed() < Duration::from_secs(5), "no wait on an uncommitted value");
        assert!(shared.write().err().expect("must refuse").is_not_submitted());

        // The refused calls released their slots; flushing makes the lease available.
        rt.commit(false).unwrap();
        drop(
            shared
                .read()
                .expect("after the flush the read lease waits for the commit and succeeds"),
        );
        drop(shared.write().unwrap());
    }

    #[test]
    fn a_writer_waits_for_gpu_readers_but_a_reader_does_not() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let other = rt.alloc_buffer(64).unwrap();
        copy_from_shared(&rt, &shared, &other, 16).unwrap();
        // A GPU read in an open command buffer: a host read has nothing to wait for...
        drop(shared.read().expect("readers do not wait for GPU readers"));
        // ...a host write must, and the commit it needs is not submitted yet.
        assert!(shared.write().err().expect("must refuse").is_not_submitted());
        rt.synchronize().unwrap();
        drop(shared.write().unwrap());
    }

    #[test]
    fn a_poisoned_runtime_refuses_every_lease() {
        let rt = GpuRuntime::new().unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        drop(shared.read().unwrap());
        rt.poison_as_shared_event_timeout_for_test();
        for err in [shared.read().err(), shared.write().err()] {
            match err.expect("poisoned") {
                HostAccessError::Runtime(m) => assert!(m.contains("poisoned"), "{m}"),
                other => panic!("wanted the runtime's poison, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_dishonest_lifetime_event_poisons_instead_of_releasing_a_lease() {
        let rt = GpuRuntime::new().unwrap();
        let shared = SharedAlloc::new(&rt, N * 4).unwrap();
        let src = rt.alloc_buffer(N * 4).unwrap();
        copy_into_shared(&rt, &src, &shared, N).unwrap();
        rt.synchronize().unwrap();
        assert!(shared.last_write() > 0);
        let submitted = rt.last_signaled_value();
        // Foreign code puts a value above anything submitted on the private event.
        rt.metal4.lifetime_event.setSignaledValue(submitted + 5);
        match shared.read().err().expect("must refuse") {
            HostAccessError::Runtime(m) => assert!(!m.is_empty()),
            other => panic!("{other:?}"),
        }
        assert!(rt.is_poisoned());
    }

    #[test]
    fn a_foreign_signal_on_the_handoff_event_does_not_release_a_lease() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let src = rt.alloc_buffer(64).unwrap();
        copy_into_shared(&rt, &src, &shared, 16).unwrap();
        rt.shared_event().setSignaledValue(1 << 40);
        // Still an unsubmitted writer: the handoff value proves nothing.
        assert!(shared.read().err().expect("must refuse").is_not_submitted());
        rt.synchronize().unwrap();
    }

    #[test]
    fn a_gpu_write_bind_is_refused_under_a_live_reader() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let src = rt.alloc_buffer(64).unwrap();

        let reader = shared.read().unwrap();
        // The non-poisoning pre-flight says no to a write and yes to a read.
        assert!(matches!(
            shared.check_gpu_access(GpuRole::Write),
            Err(HostAccessError::Busy(_))
        ));
        assert!(shared.check_gpu_access(GpuRole::Read).is_ok());
        assert!(!rt.is_poisoned());

        let err = copy_into_shared(&rt, &src, &shared, 16).expect_err("the bind itself must refuse");
        assert!(err.contains("GPU write bind was refused"), "{err}");
        drop(reader);
    }

    #[test]
    fn a_gpu_read_bind_is_refused_under_a_live_writer() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let dst = rt.alloc_buffer(64).unwrap();
        let writer = shared.write().unwrap();
        assert!(matches!(
            shared.check_gpu_access(GpuRole::Read),
            Err(HostAccessError::Busy(_))
        ));
        let err = copy_from_shared(&rt, &shared, &dst, 16).expect_err("must refuse");
        assert!(err.contains("GPU read bind was refused"), "{err}");
        drop(writer);
    }

    #[test]
    fn shared_binds_are_refused_during_an_icb_capture() {
        let rt = GpuRuntime::new().unwrap();
        rt.set_async_encode(true).unwrap();
        let shared = SharedAlloc::new(&rt, 64).unwrap();
        let dst = rt.alloc_buffer(64).unwrap();
        crate::decode_icb::begin_decode_icb_capture();
        let result = copy_from_shared(&rt, &shared, &dst, 16);
        crate::decode_icb::end_decode_icb_capture();
        let err = result.expect_err("a replayed tape would skip the stamp");
        assert!(err.contains("ICB capture"), "{err}");
        assert_eq!(shared.last_use(), 0, "a refused capture bind stamps nothing");
    }

    #[test]
    fn the_core_and_its_errors_are_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<SharedBuf>();
        send_sync::<HostAccessError>();
    }

    /// Worker threads lease through `&SharedBuf` while the encoding thread
    /// keeps the `!Send` allocation: several read at once, then one writes.
    #[test]
    fn worker_threads_lease_through_the_core_reference() {
        let rt = GpuRuntime::new().unwrap();
        let shared = SharedAlloc::new(&rt, N * 4).unwrap();
        shared.write().unwrap().as_u32_mut().unwrap().fill(3);
        let core: &SharedBuf = &shared;
        let totals: Vec<u64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(move || {
                        let r = core.read().expect("readers overlap across threads");
                        let total = r.as_u32().unwrap().iter().map(|&w| u64::from(w)).sum::<u64>();
                        drop(r);
                        total
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(totals.iter().all(|&t| t == 3 * N as u64), "{totals:?}");
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut w = core.write().expect("the readers are gone");
                w.as_u32_mut().unwrap().fill(9);
            })
            .join()
            .unwrap();
        });
        assert!(shared.read().unwrap().as_u32().unwrap().iter().all(|&w| w == 9));
    }
}
