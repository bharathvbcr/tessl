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
        .err()
        .expect("a poisoned runtime refuses an allocation");
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
