//! Characterization of the runtime's exclusive access lease (card ft-4b1313de).
//!
//! `GpuRuntime::acquire_access` (src/runtime.rs) is one `AtomicBool` lease. A
//! `HostMapping`, an open encoder (`with_binder`), a submit (`commit`) and the
//! host-visible allocators all take it, and a second taker fails with
//! "runtime busy" instead of waiting. These tests pin what that lease does
//! today so a later change (for example a shared read guard) has to say which
//! behaviours it retires. They assert current behaviour and change none. The
//! invariant table they back is in docs/architecture.md ("Runtime access
//! lease").
//!
//! Every test needs a Metal device, so none is device-free.

mod common;

use std::panic::{catch_unwind, AssertUnwindSafe};

use common::{assert_within_bound, gpu_lock, random_f32, reference, tensor_f32, with_gpu, Layout};
use tessl::{gemm_f32, GemmBackend, GpuRuntime};

const BUSY: &str = "runtime busy: another host mapping, encoder, or submit is active";

/// The error of a result whose `Ok` side is a guard with no `Debug`.
fn refused<T>(what: &str, r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("{what} succeeded; it should have been refused"),
        Err(e) => e,
    }
}

/// The card's symptom: while one `HostMapping` is alive a second `map_host`
/// fails with the exact busy message, whether it names the same buffer (which
/// would alias `&mut [T]`) or another one, and whatever the element type. The
/// refusal leaves the first mapping intact and does not poison the runtime.
#[test]
fn a_second_host_mapping_while_one_is_alive_is_refused_as_busy() {
    with_gpu(|rt| {
        let a = rt.alloc_buffer(64).unwrap();
        let b = rt.alloc_buffer(64).unwrap();
        let mut first = a.try_contents_f32().unwrap();
        first[3] = 5.0;

        assert_eq!(refused("same buffer, same type", a.try_contents_f32()), BUSY);
        assert_eq!(refused("same buffer, other type", a.try_contents_u8()), BUSY);
        assert_eq!(refused("other buffer", b.try_contents_f32()), BUSY);
        assert!(!rt.is_poisoned(), "a busy refusal must not poison the runtime");
        assert_eq!(first[3], 5.0, "the refused mapping disturbed the live one");

        drop(first);
        let second = b.try_contents_f32().unwrap();
        drop(second);
        assert_eq!(a.try_contents_f32().unwrap()[3], 5.0);
    });
}

/// A live mapping excludes every other lease taker: encode, both submits, the
/// host-visible allocators. None of the refusals poisons the runtime, because
/// each returns before the poison latch can see its error.
#[test]
fn a_live_mapping_refuses_encode_submit_and_host_allocation() {
    with_gpu(|rt| {
        rt.ensure_bump(4096).unwrap();
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.try_contents_u8().unwrap();

        assert_eq!(refused("with_binder", rt.with_binder(|_| Ok(())).map(|_| ())), BUSY);
        assert_eq!(refused("commit(false)", rt.commit(false).map(|_| ())), BUSY);
        assert_eq!(refused("commit(true)", rt.commit(true).map(|_| ())), BUSY);
        assert_eq!(refused("synchronize", rt.synchronize().map(|_| ())), BUSY);
        assert_eq!(
            refused("alloc_buffer_from_u32", rt.alloc_buffer_from_u32(&[1]).map(|_| ())),
            BUSY
        );
        assert_eq!(refused("bump_alloc_f32", rt.bump_alloc_f32(&[4]).map(|_| ())), BUSY);
        assert!(!rt.is_poisoned());

        drop(mapping);
        rt.with_binder(|_| Ok(())).unwrap();
        rt.synchronize().unwrap();
        rt.alloc_buffer_from_u32(&[1]).unwrap();
        rt.bump_alloc_f32(&[4]).unwrap();
    });
}

/// The encoder holds the lease for the whole closure: a mapping, a nested
/// encode and a submit attempted from inside it are all refused, and when the
/// closure absorbs those refusals the runtime stays healthy.
#[test]
fn an_encoder_closure_refuses_mapping_nested_encode_and_submit() {
    with_gpu(|rt| {
        let buf = rt.alloc_buffer(64).unwrap();
        rt.with_binder(|_| {
            assert_eq!(refused("map in closure", buf.try_contents_u8()), BUSY);
            assert_eq!(
                refused("nested with_binder", rt.with_binder(|_| Ok(())).map(|_| ())),
                BUSY
            );
            assert_eq!(refused("commit in closure", rt.commit(true).map(|_| ())), BUSY);
            Ok(())
        })
        .unwrap();
        assert!(!rt.is_poisoned());
        drop(buf.try_contents_u8().unwrap());
    });
}

/// The other half of that nesting rule: a closure that propagates the busy
/// refusal with `?` fails its own encode, and a failed encode poisons the
/// runtime, with the busy message as the recorded first cause.
#[test]
fn a_busy_refusal_propagated_out_of_an_encoder_closure_poisons_the_runtime() {
    with_gpu(|rt| {
        let buf = rt.alloc_buffer(64).unwrap();
        let err = rt
            .with_binder(|_| {
                let _map = buf.try_contents_u8()?;
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err, BUSY);
        assert!(rt.is_poisoned());
        let later = refused("map after poison", buf.try_contents_u8());
        assert!(later.contains("poisoned"), "{later}");
        assert!(later.contains(BUSY), "the refusal lost the first cause: {later}");
    });
}

/// Poison is checked before the lease is tried: with a mapping live and the
/// runtime poisoned, a new mapping is refused as poisoned (not busy), the
/// live mapping keeps working, and dropping it does not clear the latch.
#[test]
fn a_poisoned_runtime_refuses_mapping_before_it_reports_busy() {
    with_gpu(|rt| {
        let buf = rt.alloc_buffer(16).unwrap();
        let mut live = buf.try_contents_u8().unwrap();
        rt.poison_as_shared_event_timeout_for_test();

        let err = refused("map while poisoned and busy", buf.try_contents_u8());
        assert!(err.contains("poisoned") && !err.contains("busy"), "{err}");
        live[0] = 9;
        assert_eq!(live[0], 9, "poison must not revoke a mapping already handed out");
        drop(live);

        let err = refused("map after the mapping dropped", buf.try_contents_u8());
        assert!(err.contains("poisoned"), "{err}");
        assert!(rt.is_poisoned());
    });
}

/// A mapping implies the GPU work encoded before it has finished: with async
/// encode the gemm sits in an open command buffer, and mapping the output
/// commits and waits (no `synchronize`) so the host reads the finished values.
#[test]
fn a_mapping_taken_over_open_encoded_work_reads_the_finished_result() {
    with_gpu(|rt| {
        const G: usize = 256;
        let (da, db) = (random_f32(G * G, 411), random_f32(G * G, 412));
        let a = tensor_f32(rt, &[G, G], &da);
        let b = tensor_f32(rt, &[G, G], &db);
        let c = rt.alloc_tensor_f32(&[G, G]).unwrap();
        rt.synchronize().unwrap();
        rt.set_async_encode(true).unwrap();
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();

        let mapped = c.buffer.try_contents_f32().unwrap();
        assert_eq!(rt.last_completed_value(), rt.last_signaled_value());
        let want = reference(Layout::Nn, &da, &db, G, G, G);
        assert_within_bound("mapped without synchronize", &mapped, &want, G, 0.0);
        drop(mapped);
        rt.set_async_encode(false).unwrap();
    });
}

/// Drop order. A mapping owns an `Arc<GpuRuntime>` and a lease flag that is
/// its own `Arc`: the runtime handle the caller held can go first, the mapping
/// still reads, and the last owner (the mapping) tearing the runtime down
/// while the lease flag is still held does not fault. The buffer, whose
/// runtime link is weak, outlives the runtime.
#[test]
fn a_mapping_outlives_the_callers_runtime_handle_and_releases_cleanly() {
    let _gpu = gpu_lock();
    let rt = GpuRuntime::new().expect("GpuRuntime::new");
    let buf = rt.alloc_buffer(32).unwrap();
    let mut mapping = buf.try_contents_u8().unwrap();
    mapping[0] = 7;
    drop(rt);
    assert_eq!(mapping[0], 7);
    assert_eq!(mapping.len(), 32);
    drop(mapping);
    drop(buf);
}

/// `map_host` validates the buffer (size and alignment for the element type)
/// before it asks for the lease, so a bad request is reported as what it is
/// even while another mapping is live, and never takes or disturbs the lease.
#[test]
fn a_size_mismatch_is_reported_before_the_lease_is_tried() {
    with_gpu(|rt| {
        let odd = rt.alloc_buffer(6).unwrap();
        let other = rt.alloc_buffer(64).unwrap();
        let live = other.try_contents_u8().unwrap();
        let err = refused("6-byte buffer as f32", odd.try_contents_f32());
        assert!(err.contains("size/alignment mismatch"), "{err}");
        assert_eq!(
            refused("the live mapping still holds the lease", other.try_contents_u8()),
            BUSY
        );
        drop(live);
        drop(odd.try_contents_u8().unwrap());
    });
}

/// The lease is RAII: a panic while a mapping is alive releases it on unwind,
/// so the next mapping succeeds and sees the write made before the panic.
#[test]
fn a_panic_with_a_live_mapping_releases_the_lease() {
    with_gpu(|rt| {
        let buf = rt.alloc_buffer(16).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut m = buf.try_contents_u8().unwrap();
            m[1] = 3;
            panic!("injected panic under a live mapping");
        }));
        assert!(outcome.is_err());
        assert!(!rt.is_poisoned(), "a host-side panic is not a GPU fault");
        assert_eq!(buf.try_contents_u8().unwrap()[1], 3);
    });
}
