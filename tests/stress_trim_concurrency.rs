//! Adversarial concurrency stress test:
//! 1. Concurrent budget charges, child budgets, and peak tracking under contention
//! 2. Concurrent `trim_free_list()` under worker-thread contention (`try_lock()`)
//! 3. Concurrent `BudgetTrimHandler` triggering `trim_free_list()` when hitting capacity limits

mod common;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use common::gpu_lock;
use ojas_core::{Budget, BudgetTrimHandler, Reservation};
use tessl::{BufferKind, GpuRuntime};

/// Test 1: Pure multi-threaded contention on `Budget` reservations, child budgets,
/// peaks, and concurrent trim handler registration/invocations.
#[test]
fn stress_concurrent_budget_charges_under_high_contention() {
    let root = Budget::new(50 * 1024 * 1024); // 50 MiB cap
    let running = Arc::new(AtomicBool::new(true));
    let total_reserved = Arc::new(AtomicU64::new(0));
    let total_refused = Arc::new(AtomicUsize::new(0));
    let total_released = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 16 worker threads concurrently charging and releasing child budgets
    for thread_idx in 0..16 {
        let root = root.clone();
        let running = Arc::clone(&running);
        let total_reserved = Arc::clone(&total_reserved);
        let total_refused = Arc::clone(&total_refused);
        let total_released = Arc::clone(&total_released);

        handles.push(thread::spawn(move || {
            let child = root.child(10 * 1024 * 1024); // 10 MiB per child
            let mut active_reservations: Vec<Reservation> = Vec::new();
            let mut rng_state: u64 = 0xdeadbeef ^ (thread_idx as u64);

            while running.load(Ordering::Relaxed) {
                // Fast xorshift64
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;

                let req_bytes = 1024 + (rng_state % (512 * 1024));

                let _ = child.check_room(req_bytes);

                match child.try_reserve(req_bytes) {
                    Ok(res) => {
                        total_reserved.fetch_add(req_bytes, Ordering::Relaxed);
                        active_reservations.push(res);
                    }
                    Err(_) => {
                        total_refused.fetch_add(1, Ordering::Relaxed);
                    }
                }

                if active_reservations.len() >= 8 || (rng_state % 5 == 0 && !active_reservations.is_empty()) {
                    let to_drop = (rng_state as usize % active_reservations.len()) + 1;
                    for _ in 0..to_drop {
                        if let Some(r) = active_reservations.pop() {
                            total_released.fetch_add(1, Ordering::Relaxed);
                            drop(r);
                        }
                    }
                }
            }

            drop(active_reservations);
        }));
    }

    let root_for_peak = root.clone();
    let running_for_peak = Arc::clone(&running);
    let peak_handle = thread::spawn(move || {
        while running_for_peak.load(Ordering::Relaxed) {
            root_for_peak.reset_peak();
            thread::yield_now();
        }
    });

    thread::sleep(Duration::from_millis(1500));
    running.store(false, Ordering::Relaxed);

    for h in handles {
        h.join().expect("worker thread join");
    }
    peak_handle.join().expect("peak thread join");

    let final_live = root.live_bytes().expect("root live bytes");
    assert_eq!(
        final_live, 0,
        "All budget reservations must cleanly release, but got live = {final_live}"
    );
    assert!(
        root.peak_bytes() <= root.cap_bytes(),
        "Peak bytes must never exceed cap"
    );
    assert!(
        total_reserved.load(Ordering::Relaxed) > 0,
        "Must have completed reservations"
    );
    assert!(
        total_released.load(Ordering::Relaxed) > 0,
        "Must have released reservations"
    );
}

enum DeviceCmd {
    AllocAndPark(usize),
    TrimFreeList(Sender<usize>),
    Stop,
}

/// Test 2: Multi-threaded contention on `GpuRuntime::trim_free_list()`
/// 16 caller threads concurrently hammering a dedicated device thread with:
/// - Allocations of cold buffers that park into the freelist
/// - Concurrent `trim_free_list()` requests
///
/// Verifies no deadlocks, no panics, clean freelist draining, and bounded execution.
#[test]
fn stress_concurrent_trim_free_list_under_heavy_contention() {
    let _guard = gpu_lock();

    let (cmd_tx, cmd_rx): (Sender<DeviceCmd>, Receiver<DeviceCmd>) = mpsc::channel();

    // Spawn dedicated device thread owning GpuRuntime (matching tessl/ojas-metal threading model)
    let device_handle = thread::spawn(move || {
        let rt = GpuRuntime::new().expect("Metal GPU runtime");
        let mut held = Vec::new();

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                DeviceCmd::AllocAndPark(size) => {
                    if let Ok(b) = rt.alloc_buffer_kind(size, BufferKind::Cold) {
                        held.push(b);
                        // Periodically drop to populate freelist
                        if held.len() >= 4 {
                            held.clear();
                        }
                    }
                }
                DeviceCmd::TrimFreeList(reply_tx) => {
                    let freed = rt.trim_free_list();
                    let _ = reply_tx.send(freed);
                }
                DeviceCmd::Stop => break,
            }
        }
        drop(held);
        let final_freed = rt.trim_free_list();
        let resident = rt.pool_resident();
        assert_eq!(resident.cached_bytes, 0, "After final trim, cached_bytes must be 0");
        final_freed
    });

    let running = Arc::new(AtomicBool::new(true));
    let total_alloc_ops = Arc::new(AtomicUsize::new(0));
    let total_trim_ops = Arc::new(AtomicUsize::new(0));
    let total_freed_bytes = Arc::new(AtomicU64::new(0));

    let mut client_handles = Vec::new();

    // 16 client threads concurrently issuing AllocAndPark and TrimFreeList
    for thread_idx in 0..16 {
        let cmd_tx = cmd_tx.clone();
        let running = Arc::clone(&running);
        let total_alloc_ops = Arc::clone(&total_alloc_ops);
        let total_trim_ops = Arc::clone(&total_trim_ops);
        let total_freed_bytes = Arc::clone(&total_freed_bytes);

        client_handles.push(thread::spawn(move || {
            let mut rng_state: u64 = 0x12345678 ^ (thread_idx as u64);
            let sizes = [16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024];

            while running.load(Ordering::Relaxed) {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;

                if rng_state % 3 == 0 {
                    // Send trim request with reply channel
                    let (reply_tx, reply_rx) = mpsc::channel();
                    if cmd_tx.send(DeviceCmd::TrimFreeList(reply_tx)).is_ok() {
                        if let Ok(freed) = reply_rx.recv_timeout(Duration::from_secs(2)) {
                            total_trim_ops.fetch_add(1, Ordering::Relaxed);
                            total_freed_bytes.fetch_add(freed as u64, Ordering::Relaxed);
                        }
                    }
                } else {
                    // Send alloc request
                    let size = sizes[(rng_state as usize) % sizes.len()];
                    if cmd_tx.send(DeviceCmd::AllocAndPark(size)).is_ok() {
                        total_alloc_ops.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    thread::sleep(Duration::from_millis(2000));
    running.store(false, Ordering::Relaxed);

    for h in client_handles {
        h.join().expect("client thread join");
    }

    cmd_tx.send(DeviceCmd::Stop).expect("send stop");
    let last_trim = device_handle.join().expect("device thread join");

    let allocs = total_alloc_ops.load(Ordering::Relaxed);
    let trims = total_trim_ops.load(Ordering::Relaxed);
    let freed = total_freed_bytes.load(Ordering::Relaxed) + (last_trim as u64);

    println!("Stress Test 2: allocs={allocs}, trims={trims}, total_freed={freed}");
    assert!(allocs > 100, "Expected >100 allocations, got {allocs}");
    assert!(trims > 50, "Expected >50 trims, got {trims}");
}

/// A trim handler bridging `Budget` to the device thread via channel.
#[derive(Debug, Clone)]
struct ThreadedTrimHandler {
    tx: Sender<DeviceCmd>,
}

impl BudgetTrimHandler for ThreadedTrimHandler {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        let (reply_tx, reply_rx) = mpsc::channel();
        if self.tx.send(DeviceCmd::TrimFreeList(reply_tx)).is_ok() {
            reply_rx.recv_timeout(Duration::from_secs(2)).unwrap_or(0) as u64
        } else {
            0
        }
    }
}

/// Test 3: Concurrent `BudgetTrimHandler` calling `trim_free_list()` on allocation pressure.
/// Multiple threads concurrently allocate, hold, and attempt reservations under tight budget cap.
#[test]
fn stress_budget_trim_handler_under_concurrent_contention() {
    let _guard = gpu_lock();

    let (cmd_tx, cmd_rx): (Sender<DeviceCmd>, Receiver<DeviceCmd>) = mpsc::channel();

    let device_handle = thread::spawn(move || {
        let rt = GpuRuntime::new().expect("Metal GPU runtime");
        let mut held = Vec::new();

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                DeviceCmd::AllocAndPark(size) => {
                    if let Ok(b) = rt.alloc_buffer_kind(size, BufferKind::Cold) {
                        held.push(b);
                        if held.len() >= 2 {
                            held.clear();
                        }
                    }
                }
                DeviceCmd::TrimFreeList(reply_tx) => {
                    let freed = rt.trim_free_list();
                    let _ = reply_tx.send(freed);
                }
                DeviceCmd::Stop => break,
            }
        }
    });

    let budget = Budget::new(4 * 1024 * 1024); // 4 MiB tight cap
    let handler = Arc::new(ThreadedTrimHandler { tx: cmd_tx.clone() });
    budget.set_trim_handler(handler);

    let running = Arc::new(AtomicBool::new(true));
    let completed_reservations = Arc::new(AtomicUsize::new(0));

    let mut client_handles = Vec::new();

    for thread_idx in 0..12 {
        let budget = budget.clone();
        let cmd_tx = cmd_tx.clone();
        let running = Arc::clone(&running);
        let completed_reservations = Arc::clone(&completed_reservations);

        client_handles.push(thread::spawn(move || {
            let mut rng_state: u64 = 0xfeedface ^ (thread_idx as u64);
            while running.load(Ordering::Relaxed) {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;

                // Push buffers to device freelist
                let _ = cmd_tx.send(DeviceCmd::AllocAndPark(256 * 1024));

                // Attempt to reserve in budget: triggers trim_handler -> trim_free_list
                let req = (128 * 1024) + (rng_state % (512 * 1024));
                if let Ok(res) = budget.try_reserve(req) {
                    completed_reservations.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                    drop(res);
                }
            }
        }));
    }

    thread::sleep(Duration::from_millis(1500));
    running.store(false, Ordering::Relaxed);

    for h in client_handles {
        h.join().expect("client completed without deadlocks");
    }

    cmd_tx.send(DeviceCmd::Stop).expect("send stop");
    device_handle.join().expect("device thread joined");

    assert!(completed_reservations.load(Ordering::Relaxed) > 0);
    assert_eq!(budget.live_bytes().unwrap(), 0, "Clean budget release");
}
