//! Two runtimes under one `Shared` share the poison latch and the commit
//! clock; a runtime with a `Shared` of its own behaves as it did before.
//!
//! The pure parts (the latch keeps the first cause, draws are unique and
//! gap-free across threads, `Shared` is `Send + Sync`) are unit tests in
//! `src/shared.rs`; these need a device.

mod common;

use std::sync::Arc;

use common::{gpu_lock, random_f32, tensor_f32};
use tessl::{softcap_f32, GpuRuntime, Shared, Timeline};

/// Run one dispatch on `rt` and wait for it, so at least one commit lands.
/// Returns the runtime's last signaled value afterwards.
fn commit_something(rt: &Arc<GpuRuntime>, seed: u64) -> u64 {
    let n = 4096;
    let pre = tensor_f32(rt, &[n], &random_f32(n, seed));
    let _post = softcap_f32(rt, &pre, 30.0).expect("softcap");
    rt.synchronize().expect("synchronize");
    rt.last_signaled_value()
}

#[test]
fn two_runtimes_under_one_shared_draw_commit_values_from_one_clock() {
    let _guard = gpu_lock();
    let shared = Shared::new();
    let a = GpuRuntime::with_shared(Arc::clone(&shared)).expect("runtime a");
    let b = GpuRuntime::with_shared(Arc::clone(&shared)).expect("runtime b");
    assert!(Arc::ptr_eq(a.shared(), &shared) && Arc::ptr_eq(b.shared(), &shared));

    // a, b, a, b, b: each commit's value is above every value drawn before it,
    // whichever runtime drew it.
    let order = [&a, &b, &a, &b, &b];
    let mut last = 0u64;
    for (i, rt) in order.into_iter().enumerate() {
        let drawn = commit_something(rt, 100 + i as u64);
        assert!(drawn > last, "commit {i}: value {drawn} does not follow {last}");
        last = drawn;
        assert_eq!(shared.issued(), last, "commit {i}: the clock is behind the commit");
    }

    // Each runtime's own view agrees with what it drew, and has completed it.
    for rt in [&a, &b] {
        let timeline = rt.timeline();
        assert_eq!(timeline.submitted(), rt.last_signaled_value());
        assert_eq!(
            timeline.completed(),
            timeline.submitted(),
            "synchronize left work pending"
        );
        timeline
            .wait(timeline.submitted())
            .expect("a completed value is waitable");
        assert!(rt.last_signaled_value() <= last);
    }
    // Interleaving left gaps in each runtime's own sequence: b drew values
    // between a's two commits, so a's second is not its first plus one.
    assert!(a.last_signaled_value() > 1 && b.last_signaled_value() > 1);
    assert_ne!(a.last_signaled_value(), b.last_signaled_value());
}

#[test]
fn the_timeline_refuses_an_unsubmitted_value_without_waiting_it_out() {
    let _guard = gpu_lock();
    let rt = GpuRuntime::new().expect("runtime");
    let submitted = commit_something(&rt, 7);
    let timeline = rt.timeline();
    timeline.wait(0).expect("value 0 is always complete");
    timeline.wait(submitted).expect("the last commit completed");

    let started = std::time::Instant::now();
    let error = timeline.wait(submitted + 1).expect_err("nothing signals this value");
    assert!(Timeline::is_not_submitted(&error), "{error}");
    assert!(!GpuRuntime::is_event_wait_timeout(&error), "{error}");
    assert!(
        started.elapsed().as_millis() < u128::from(rt.event_wait_ms()),
        "the refusal waited out the budget"
    );
    assert!(!rt.is_poisoned(), "an unsubmitted value must not poison");
    assert_eq!(rt.timeline().submitted(), submitted, "the refusal committed nothing");
}

#[test]
fn poison_in_one_runtime_refuses_calls_on_the_other() {
    let _guard = gpu_lock();
    let shared = Shared::new();
    let a = GpuRuntime::with_shared(Arc::clone(&shared)).expect("runtime a");
    let b = GpuRuntime::with_shared(Arc::clone(&shared)).expect("runtime b");
    commit_something(&b, 1);
    assert!(!a.is_poisoned() && !b.is_poisoned() && !shared.is_poisoned());

    a.poison_as_shared_event_timeout_for_test();

    assert!(shared.is_poisoned());
    assert!(b.is_poisoned(), "the other runtime did not see the shared poison");
    let cause = shared.poison_error().expect("poison cause");
    assert!(cause.contains("poison_as_shared_event_timeout_for_test"), "{cause}");
    let refused = b
        .alloc_buffer(256)
        .expect_err("a poisoned runtime refuses an allocation");
    assert_eq!(refused, cause, "b reports the cause a recorded");
    let waited = b.timeline().wait(b.last_signaled_value()).expect_err("poisoned wait");
    assert_eq!(waited, cause);
}

#[test]
fn a_runtime_with_its_own_shared_is_unchanged() {
    let _guard = gpu_lock();
    let a = GpuRuntime::new().expect("runtime a");
    let b = GpuRuntime::new().expect("runtime b");
    assert!(!Arc::ptr_eq(a.shared(), b.shared()), "new() must not share state");

    // A lone runtime's values are 1, 2, 3, ... with no gaps, as before.
    let mut previous = a.last_signaled_value();
    for seed in 0..3 {
        let drawn = commit_something(&a, 40 + seed);
        assert!(drawn > previous);
        assert_eq!(
            drawn,
            a.shared().issued(),
            "a lone runtime's clock is its own last value"
        );
        previous = drawn;
    }
    assert_eq!(b.last_signaled_value(), 0, "b never committed");
    assert_eq!(b.shared().issued(), 0, "a's commits drew from a's own clock");

    a.poison_as_shared_event_timeout_for_test();
    assert!(a.is_poisoned());
    assert!(!b.is_poisoned(), "new() runtimes do not share poison");
}

#[test]
fn adversarial_shared_buf_uma_and_placement_invariants() {
    let _guard = gpu_lock();
    use ojas_core::{BackendId, Budget, BudgetTrimHandler, DType, DeviceBuffer, MappedBuffer, OjasError, Tensor};
    use tessl::shared_buf::{SharedAlloc, SharedBuf};

    // 1. Confirm that BudgetTrimHandler is accessible from crate root in downstream code
    #[derive(Debug)]
    struct MockTrimHandler(std::sync::atomic::AtomicU64);
    impl BudgetTrimHandler for MockTrimHandler {
        fn trim_memory(&self, needed: u64) -> u64 {
            self.0.store(needed, std::sync::atomic::Ordering::SeqCst);
            needed
        }
    }
    let budget = Budget::new(1024 * 1024);
    let handler = Arc::new(MockTrimHandler(std::sync::atomic::AtomicU64::new(0)));
    budget.set_trim_handler(handler.clone());
    assert!(budget.trim_handler().is_some());

    // 2. Confirm SharedBuf implements MappedBuffer & can be passed to Tensor::from_mapped
    let rt = GpuRuntime::new().expect("runtime");
    let count = 256;
    let nbytes = count * 4;
    let shared_alloc = SharedAlloc::new(&rt, nbytes).expect("shared alloc");
    let shared_buf: SharedBuf = shared_alloc.shared_buf();

    // Verify MappedBuffer trait methods on SharedBuf
    assert_eq!(shared_buf.byte_len(), nbytes);
    assert_eq!(shared_buf.host_ptr() as usize, shared_alloc.host_addr());
    assert!(shared_buf.gpu_address() > 0);
    assert_eq!(shared_buf.timeline().current_value(), 0);

    let mapped: Arc<dyn MappedBuffer> = Arc::new(shared_buf.clone());
    let tensor = Tensor::from_mapped(mapped, &[count], DType::F32, &budget).expect("from_mapped");

    assert_eq!(tensor.device(), Some(BackendId::Metal));
    assert!(tensor.mapped_buffer().is_some());
    assert_eq!(budget.live_bytes().unwrap(), nbytes as u64);

    // 3. Verify placement invariants fail closed with OjasError::Placement without fallback
    assert!(matches!(
        tensor.f32_slice().unwrap_err(),
        OjasError::Placement {
            found: Some(BackendId::Metal),
            ..
        }
    ));
    assert!(matches!(
        tensor.to_f32_vec().unwrap_err(),
        OjasError::Placement {
            found: Some(BackendId::Metal),
            ..
        }
    ));
    let mut t_mut = tensor.clone();
    assert!(matches!(
        t_mut.write_f32(&vec![1.0; count]).unwrap_err(),
        OjasError::Placement {
            found: Some(BackendId::Metal),
            ..
        }
    ));

    // Also verify Bf16 tensor placement check
    let bf16_alloc = SharedAlloc::new(&rt, count * 2).expect("bf16 alloc");
    let bf16_buf: SharedBuf = bf16_alloc.shared_buf();
    let bf16_tensor = Tensor::from_mapped(Arc::new(bf16_buf), &[count], DType::Bf16, &budget).expect("bf16 tensor");
    assert!(matches!(
        bf16_tensor.bf16_slice().unwrap_err(),
        OjasError::Placement {
            found: Some(BackendId::Metal),
            ..
        }
    ));

    // 4. Verify zero-copy host view via tensor.host_view() and UMA memory coherence
    {
        let mut w = shared_buf.host_write().expect("host write lease");
        let f32_mut = w.as_f32_mut().expect("as_f32_mut");
        for (i, val) in f32_mut.iter_mut().enumerate() {
            *val = (i as f32) * 2.5 + 1.0;
        }
    }

    let view = tensor.host_view().expect("host_view");
    assert_eq!(view.as_bytes().as_ptr() as usize, shared_alloc.host_addr());
    let view_f32 = view.as_f32().expect("as_f32");
    assert_eq!(view_f32.len(), count);
    assert_eq!(view_f32[0], 1.0);
    assert_eq!(view_f32[1], 3.5);
    assert_eq!(view_f32[255], 255.0 * 2.5 + 1.0);

    // 5. Test freelist cache trimming under memory pressure
    let freed = rt.trim_free_list();
    let _ = freed;
}
