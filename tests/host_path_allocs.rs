//! What the per-dispatch host path allocates on success: nothing, where a
//! decode loop calls it every token.
//!
//! A counting global allocator, armed per thread, counts every Rust heap
//! allocation between `arm` and `disarm`. The pipeline-mode flag is
//! process-wide, which is why this is its own test binary and every check
//! runs in one test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    if ARMED.with(Cell::get) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` unchanged; the counter is a side
// effect that neither allocates nor touches the returned memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations `f` makes on this thread.
fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    (out, ALLOCS.load(Ordering::Relaxed) - before)
}

struct IcbModeRestore;

impl Drop for IcbModeRestore {
    fn drop(&mut self) {
        tessl::decode_icb::set_icb_pipelines(false);
    }
}

#[test]
fn a_pipeline_cache_hit_allocates_nothing_in_either_mode() {
    let rt = tessl::GpuRuntime::new().expect("GpuRuntime::new");
    let _restore = IcbModeRestore;
    for icb in [false, true] {
        tessl::decode_icb::set_icb_pipelines(icb);
        rt.pipeline("rms_norm_f32").expect("warm the cache");
        let (hits, n) = allocations(|| (0..64).map(|_| rt.pipeline("rms_norm_f32")).collect::<Vec<_>>());
        // The Vec holding the 64 results is the one allocation allowed.
        assert_eq!(n, 1, "icb = {icb}: {} allocations over 64 cache hits", n - 1);
        assert!(hits.iter().all(Result::is_ok));
    }
}
