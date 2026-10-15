//! Adversarial lifecycle checks for DecodeIcb (wave 7, ft-9ac2a619): drop while
//! in flight, drop after the runtime, drop on a poisoned runtime, nop scopes.

use std::sync::Mutex;

use tessl::decode_icb::set_icb_pipelines;
use tessl::{BinderEncodeNopGuard, DecodeIcb, GpuRuntime};

static LOCK: Mutex<()> = Mutex::new(());

struct Flags;
impl Drop for Flags {
    fn drop(&mut self) {
        set_icb_pipelines(false);
        tessl::set_icb_freeze_binds(false);
        tessl::set_icb_range_batch(false);
    }
}

/// Attack 3a: drop a tape whose ICB-executing replay is still in flight (async
/// encode, no wait). The copy it ran must still land, repeatedly.
#[test]
fn dropping_a_tape_with_its_execute_in_flight_is_safe() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    set_icb_pipelines(true);
    tessl::set_icb_freeze_binds(true);
    tessl::set_icb_range_batch(true);
    let rt = GpuRuntime::new().expect("runtime");
    rt.set_async_encode(true).unwrap();
    for round in 0..200 {
        let (mut dec, out) = DecodeIcb::mini_copy_chain(&rt, 1 << 16).expect("tape");
        dec.execute(&rt).expect("execute");
        drop(dec); // GPU has not been waited on
        rt.synchronize().expect("synchronize after dropping an in-flight tape");
        let _ = out.read_f32();
        assert!(!rt.is_poisoned(), "round {round}: runtime poisoned");
    }
}

/// Attack 3b: the runtime is gone before the tape (Weak upgrade fails).
#[test]
fn dropping_a_tape_after_its_runtime_does_not_crash() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    set_icb_pipelines(true);
    tessl::set_icb_freeze_binds(true);
    let rt = GpuRuntime::new().expect("runtime");
    let (mut dec, out) = DecodeIcb::mini_copy_chain(&rt, 64).expect("tape");
    dec.execute(&rt).expect("execute");
    rt.synchronize().unwrap();
    drop(rt);
    drop(dec);
    drop(out);
}

/// Attack 3c: drop on a poisoned runtime must not panic.
#[test]
fn dropping_a_tape_on_a_poisoned_runtime_does_not_panic() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    let rt = GpuRuntime::new().expect("runtime");
    let (dec, _out) = DecodeIcb::mini_copy_chain(&rt, 64).expect("tape");
    assert!(rt.with_binder(|_| Err("injected".into())).is_err());
    assert!(rt.is_poisoned());
    drop(dec);
}

/// Attack 5: under a nop scope no state changes on a healthy runtime; poison is
/// always reported.
#[test]
fn nop_scope_mutates_nothing_and_reports_poison() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _f = Flags;
    let rt = GpuRuntime::new().expect("runtime");
    let (mut dec, _out) = DecodeIcb::mini_copy_chain(&rt, 64).expect("tape");
    let (count, status) = (dec.execute_count(), dec.status_line());
    {
        let _nop = BinderEncodeNopGuard::enter();
        for _ in 0..3 {
            dec.execute(&rt).expect("nop execute is benign");
        }
        assert!(dec.last_execute_was_nop());
    }
    assert_eq!(dec.execute_count(), count);
    assert_eq!(dec.status_line(), status);
    assert!(!rt.is_poisoned());
    assert!(
        rt.with_binder(|_| Ok(())).is_ok(),
        "runtime still usable after nop scope"
    );
    assert!(rt.with_binder(|_| Err("x".into())).is_err());
    let _nop = BinderEncodeNopGuard::enter();
    assert!(rt.with_binder(|_| Ok(())).is_err(), "with_binder under nop on poison");
    assert!(rt.with_binder_barriers(Some(true), |_| Ok(())).is_err());
    assert!(dec.execute(&rt).is_err());
}
