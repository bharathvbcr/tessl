//! Lightweight inference-path counters, off by default.
//!
//! Downstream inference crates need to know where a token's time went without
//! paying for that knowledge on every run. The atomics here are only touched
//! when [`set_enabled`] is true, so a production decode loop that never enables
//! tracing pays a single relaxed load per counter site.
//!
//! These are counters, not a profiler. They answer "how many dispatches, how
//! many bytes, how long in this phase", and they are the numbers to reach for
//! when a decode step is slower than the sum of its kernels — which usually
//! means encode overhead rather than shader time. See
//! [`crate::runtime::GpuRuntime::set_async_encode`] for the usual cause.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static BARRIERS: AtomicU64 = AtomicU64::new(0);
static COMMITS: AtomicU64 = AtomicU64::new(0);
static COLD_ALLOCS: AtomicU64 = AtomicU64::new(0);
static SYNC_WAIT_US: AtomicU64 = AtomicU64::new(0);
static RESIDENCY_FLUSHES: AtomicU64 = AtomicU64::new(0);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn reset_token_counters() {
    DISPATCHES.store(0, Ordering::Relaxed);
    BARRIERS.store(0, Ordering::Relaxed);
    COMMITS.store(0, Ordering::Relaxed);
    COLD_ALLOCS.store(0, Ordering::Relaxed);
    SYNC_WAIT_US.store(0, Ordering::Relaxed);
    RESIDENCY_FLUSHES.store(0, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub dispatches: u64,
    pub barriers: u64,
    pub commits: u64,
    pub cold_allocs: u64,
    pub sync_wait_us: u64,
    pub residency_flushes: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        dispatches: DISPATCHES.load(Ordering::Relaxed),
        barriers: BARRIERS.load(Ordering::Relaxed),
        commits: COMMITS.load(Ordering::Relaxed),
        cold_allocs: COLD_ALLOCS.load(Ordering::Relaxed),
        sync_wait_us: SYNC_WAIT_US.load(Ordering::Relaxed),
        residency_flushes: RESIDENCY_FLUSHES.load(Ordering::Relaxed),
    }
}

#[inline]
pub fn on_dispatch() {
    if enabled() {
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn on_barrier() {
    if enabled() {
        BARRIERS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn on_commit() {
    if enabled() {
        COMMITS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn on_cold_alloc() {
    if enabled() {
        COLD_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

#[inline]
pub fn on_residency_flush() {
    if enabled() {
        RESIDENCY_FLUSHES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Accumulate SharedEvent / synchronize wall time.
pub fn record_sync_wait(t0: Instant) {
    if enabled() {
        SYNC_WAIT_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}
