//! `GpuRuntime::async_scope`: a library call queues its dispatches and waits
//! once, without touching the runtime's `async_encode` flag.
//!
//! The runtime is `!Send` and `!Sync` (`GpuRuntime`'s compile_fail doctest),
//! so "another thread dispatches while a scope is open" cannot be written;
//! its same-thread form is: code that runs while a scope is open and does not
//! know about it. These tests dispatch and read back from inside a scope the
//! way such code would.
//!
//! Counts come from `infer_trace`, which is process-wide, so every test takes
//! the GPU lock through `with_gpu` and the file needs no `--test-threads`.

mod common;

use common::with_gpu;
use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use tessl::infer_trace;
use tessl::runtime::GpuRuntime;
use tessl::tensor::GpuBuffer;

const N: usize = 64;

/// What `f` counted.
fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

fn buffer(rt: &GpuRuntime, fill: f32) -> GpuBuffer {
    let b = rt.alloc_buffer(N * 4).unwrap();
    b.try_write_f32(&[fill; N]).unwrap();
    b
}

/// `dst = src`, one dispatch of `copy_f32`.
fn copy(rt: &GpuRuntime, src: &GpuBuffer, dst: &GpuBuffer) -> Result<(), String> {
    let pipeline = rt.pipeline("copy_f32")?;
    dispatch_1d(rt, &pipeline, N, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        set_u32(bnd, N as u32, 2);
    })
}

fn first(b: &GpuBuffer) -> f32 {
    b.try_contents_f32().unwrap()[0]
}

#[test]
fn a_scope_queues_dispatches_and_leaves_the_flag_alone() {
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit and wait counts"
    );
    with_gpu(|rt| {
        let (a, b) = (buffer(rt, 7.0), buffer(rt, 0.0));
        copy(rt, &a, &b).unwrap(); // warm the pipeline
        assert!(!rt.async_encode_enabled());

        // Outside a scope a default runtime commits and waits per dispatch.
        let (_, plain) = traced(|| {
            for _ in 0..10 {
                copy(rt, &a, &b).unwrap();
            }
        });
        assert_eq!((plain.sync_waits, plain.commits), (10, 10));

        let ((), scoped) = traced(|| {
            let scope = rt.async_scope();
            assert_eq!(rt.async_scope_depth(), 1);
            assert!(rt.dispatch_batching());
            assert!(!rt.async_encode_enabled(), "a scope must not set the flag");
            for _ in 0..10 {
                copy(rt, &a, &b).unwrap();
            }
            rt.synchronize().unwrap();
            drop(scope);
        });
        assert_eq!(
            (scoped.sync_waits, scoped.commits),
            (1, 1),
            "ten scoped dispatches: one waited command buffer"
        );
        assert_eq!(rt.async_scope_depth(), 0);
        assert!(!rt.dispatch_batching());
        assert!(!rt.async_encode_enabled());
        assert_eq!(first(&b), 7.0);

        // And the runtime is synchronous again after it.
        let (_, after) = traced(|| copy(rt, &a, &b).unwrap());
        assert_eq!((after.sync_waits, after.commits), (1, 1));
    });
}

/// The same-thread form of "another caller's dispatch lands in the open
/// batch": a dispatch the scope's owner did not make is queued unwaited, yet
/// a host read of its result is correct, because a host mapping commits and
/// waits for everything queued, and the scope goes on batching afterwards.
#[test]
fn a_host_readback_inside_a_scope_waits_and_is_correct() {
    with_gpu(|rt| {
        let (a, b, c) = (buffer(rt, 3.0), buffer(rt, 0.0), buffer(rt, 0.0));
        copy(rt, &a, &b).unwrap(); // warm the pipeline
        let scope = rt.async_scope();

        let (_, queued) = traced(|| copy(rt, &a, &b).unwrap());
        assert_eq!(
            (queued.sync_waits, queued.commits),
            (0, 0),
            "the dispatch was waited for"
        );

        let (v, read) = traced(|| first(&b));
        assert_eq!(v, 3.0, "a readback inside the scope saw stale bytes");
        assert_eq!((read.sync_waits, read.commits), (1, 1), "the mapping did not wait");
        assert_eq!(rt.async_scope_depth(), 1, "a readback ended the scope");

        // Still batching: the next dispatch queues, and a second read is right.
        let (_, again) = traced(|| copy(rt, &b, &c).unwrap());
        assert_eq!((again.sync_waits, again.commits), (0, 0));
        assert_eq!(first(&c), 3.0);
        drop(scope);
        assert_eq!(rt.async_scope_depth(), 0);
    });
}

#[test]
fn scopes_nest_and_do_not_disturb_the_flag() {
    with_gpu(|rt| {
        assert_eq!(rt.async_scope_depth(), 0);
        let outer = rt.async_scope();
        let inner = rt.async_scope();
        assert_eq!(rt.async_scope_depth(), 2);
        drop(inner);
        assert_eq!(rt.async_scope_depth(), 1);
        assert!(
            rt.dispatch_batching(),
            "the inner scope's end ended the outer's batching"
        );
        drop(outer);
        assert!(!rt.dispatch_batching());

        // The caller's flag survives a scope opened and closed under it, and
        // a flag the caller turns off mid-scope does not end the scope.
        rt.set_async_encode(true).unwrap();
        let scope = rt.async_scope();
        drop(scope);
        assert!(rt.async_encode_enabled() && rt.dispatch_batching());
        let scope = rt.async_scope();
        rt.set_async_encode(false).unwrap();
        assert!(rt.dispatch_batching(), "turning the flag off ended an open scope");
        drop(scope);
        assert!(!rt.dispatch_batching());
    });
}

#[test]
fn serializing_dispatch_overrides_a_scope_and_the_flag() {
    with_gpu(|rt| {
        let (a, b) = (buffer(rt, 5.0), buffer(rt, 0.0));
        copy(rt, &a, &b).unwrap();
        rt.set_async_encode(true).unwrap();
        let scope = rt.async_scope();
        rt.set_serialized_dispatch(true).unwrap();
        assert!(!rt.dispatch_batching());
        let (_, t) = traced(|| {
            for _ in 0..5 {
                copy(rt, &a, &b).unwrap();
            }
        });
        assert_eq!((t.sync_waits, t.commits), (5, 5));
        rt.set_serialized_dispatch(false).unwrap();
        assert!(rt.dispatch_batching());
        drop(scope);
        rt.set_async_encode(false).unwrap();
        assert_eq!(first(&b), 5.0);
    });
}

/// The guard decrements on an error return, on an unwind, and on a poisoned
/// runtime, and a poisoned runtime still refuses work inside a scope.
#[test]
fn a_scope_ends_on_error_unwind_and_poison() {
    with_gpu(|rt| {
        let (a, b) = (buffer(rt, 1.0), buffer(rt, 0.0));
        copy(rt, &a, &b).unwrap();

        let failed = |rt: &GpuRuntime| -> Result<(), String> {
            let _scope = rt.async_scope();
            Err("early return".into())
        };
        assert!(failed(rt).is_err());
        assert_eq!(rt.async_scope_depth(), 0, "an error return leaked a scope");
        assert!(!rt.dispatch_batching());

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = rt.async_scope();
            let _ = rt.with_binder(|_| panic!("injected panic inside a scope"));
        }));
        assert!(unwound.is_err());
        assert_eq!(rt.async_scope_depth(), 0, "an unwind leaked a scope");
        assert!(rt.is_poisoned(), "a panicking binder closure must poison");

        let scope = rt.async_scope();
        assert!(
            copy(rt, &a, &b).is_err(),
            "a poisoned runtime dispatched inside a scope"
        );
        assert!(rt.synchronize().is_err());
        drop(scope);
        assert_eq!(rt.async_scope_depth(), 0);
    });
}

/// A scope is bounded like `set_async_encode(true)`: past the per-command-
/// buffer dispatch cap the runtime commits (without a host wait) and goes on.
/// 210k dispatches of a ping-pong copy cross the 100k cap twice; an even count
/// leaves the first buffer's value in it.
#[test]
fn a_long_scope_commits_at_the_dispatch_cap_and_stays_correct() {
    assert!(
        std::env::var_os("TESSL_MID_COMMIT").is_none(),
        "TESSL_MID_COMMIT changes the commit counts"
    );
    with_gpu(|rt| {
        let (a, b) = (buffer(rt, 9.0), buffer(rt, 0.0));
        copy(rt, &a, &b).unwrap();
        a.try_write_f32(&[9.0; N]).unwrap();
        b.try_write_f32(&[0.0; N]).unwrap();
        let (r, t) = traced(|| {
            let scope = rt.async_scope();
            for i in 0..210_000 {
                if i % 2 == 0 {
                    copy(rt, &a, &b)?;
                } else {
                    copy(rt, &b, &a)?;
                }
            }
            let waited = rt.synchronize();
            drop(scope);
            waited
        });
        r.unwrap();
        assert!(
            t.commits >= 3,
            "{} dispatches made {} command buffers",
            t.dispatches,
            t.commits
        );
        assert!(
            t.commits <= 4,
            "{} dispatches made {} command buffers",
            t.dispatches,
            t.commits
        );
        assert_eq!((first(&a), first(&b)), (9.0, 9.0));
    });
}
