//! The C ABI's handle contract: null handles, foreign threads, freed handles.
//!
//! The null-handle tests create no runtime and need no GPU. The foreign-thread
//! test needs one real runtime handle (a Metal device). The aliasing the
//! thread check used to cause (a `&mut` formed on a foreign thread while the
//! owner may be inside a call) is undefined behaviour Miri cannot exercise
//! through Metal, so what is pinned here is the observable half: the refusal
//! is a status, carries the thread message before any argument is looked at,
//! and leaves the handle usable.
//!
//! A freed handle is not tested for: the ABI keeps no registry of live
//! handles, so a call on a freed pointer is undefined behaviour by contract
//! (documented in `src/capi.rs`), and a test would have to commit it.

mod common;

use std::ffi::{c_char, CStr};
use std::ptr;

use tessl::capi::{
    tessl_cross_entropy_rows, tessl_gdn_train_backward, tessl_gdn_train_forward, tessl_qwen35_adamw_free,
    tessl_qwen35_adamw_set_step_count, tessl_qwen35_adamw_step_count, tessl_qwen35_free, tessl_qwen35_grad_sq_norm,
    tessl_qwen35_load, tessl_qwen35_param_count, tessl_qwen35_train_discard, tessl_qwen35_train_step,
    tessl_runtime_free, tessl_runtime_new, tessl_synchronize, TesslQwen35, TesslRuntime, TESSL_ABI_VERSION, TESSL_ERR,
    TESSL_F32, TESSL_OK,
};

const ERR_LEN: usize = 512;

type Call = (&'static str, i32, String);

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn run(out: &mut Vec<Call>, name: &'static str, f: &dyn Fn(*mut c_char, usize) -> i32) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = f(err.as_mut_ptr(), ERR_LEN);
    out.push((name, s, msg(&err)));
}

/// Every runtime-handle entry point called with `rt` and no valid arguments
/// beyond it: the handle check runs first, so status and message depend only
/// on the handle.
fn runtime_entries(rt: *mut TesslRuntime) -> Vec<Call> {
    let mut out = Vec::new();
    unsafe {
        run(&mut out, "tessl_synchronize", &|e, n| tessl_synchronize(rt, e, n));
        run(&mut out, "tessl_cross_entropy_rows", &|e, n| {
            tessl_cross_entropy_rows(rt, ptr::null(), ptr::null_mut(), ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_gdn_train_forward", &|e, n| {
            tessl_gdn_train_forward(rt, ptr::null(), e, n)
        });
        run(&mut out, "tessl_gdn_train_backward", &|e, n| {
            tessl_gdn_train_backward(rt, ptr::null(), e, n)
        });
        run(&mut out, "tessl_qwen35_load", &|e, n| {
            let mut m = ptr::null_mut();
            tessl_qwen35_load(rt, ptr::null(), ptr::null(), ptr::null(), TESSL_F32, &mut m, e, n)
        });
    }
    out
}

/// Every model-handle entry point, as [`runtime_entries`].
fn model_entries(model: *mut TesslQwen35) -> Vec<Call> {
    let mut out = Vec::new();
    unsafe {
        run(&mut out, "tessl_qwen35_param_count", &|e, n| {
            tessl_qwen35_param_count(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_train_step", &|e, n| {
            tessl_qwen35_train_step(model, ptr::null(), 0, 0, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_train_discard", &|e, n| {
            tessl_qwen35_train_discard(model, e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_free", &|e, n| {
            tessl_qwen35_adamw_free(model, e, n)
        });
        run(&mut out, "tessl_qwen35_grad_sq_norm", &|e, n| {
            tessl_qwen35_grad_sq_norm(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_step_count", &|e, n| {
            tessl_qwen35_adamw_step_count(model, ptr::null_mut(), e, n)
        });
        run(&mut out, "tessl_qwen35_adamw_set_step_count", &|e, n| {
            tessl_qwen35_adamw_set_step_count(model, 0, e, n)
        });
    }
    out
}

#[test]
fn a_null_handle_is_a_refusal_with_a_message_and_a_null_free_is_a_no_op() {
    let calls: Vec<Call> = runtime_entries(ptr::null_mut())
        .into_iter()
        .chain(model_entries(ptr::null_mut()))
        .collect();
    assert_eq!(calls.len(), 12);
    for (name, status, message) in calls {
        assert_eq!(status, TESSL_ERR, "{name}");
        assert!(message.starts_with("null "), "{name}: {message:?}");
        assert!(message.contains("handle"), "{name}: {message:?}");
    }
    assert_eq!(unsafe { tessl_runtime_free(ptr::null_mut()) }, TESSL_OK);
    assert_eq!(unsafe { tessl_qwen35_free(ptr::null_mut()) }, TESSL_OK);
}

#[test]
fn a_null_handle_with_no_error_buffer_is_still_a_status() {
    assert_eq!(
        unsafe { tessl_synchronize(ptr::null_mut(), ptr::null_mut(), 0) },
        TESSL_ERR
    );
}

#[test]
fn every_runtime_entry_refuses_a_foreign_thread_before_looking_at_its_arguments() {
    let _gpu = common::gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let raw = rt as usize;
    // The arguments are all null: were they looked at first, the message
    // would be "null args", not the thread's.
    let calls = std::thread::spawn(move || runtime_entries(raw as *mut TesslRuntime))
        .join()
        .unwrap();
    assert_eq!(calls.len(), 5);
    for (name, status, message) in &calls {
        assert_eq!(*status, TESSL_ERR, "{name}");
        assert!(
            message.contains("other than the one that created it"),
            "{name}: {message:?}"
        );
    }
    // A foreign free leaks rather than drops, and the owner still works.
    let freed = std::thread::spawn(move || unsafe { tessl_runtime_free(raw as *mut TesslRuntime) })
        .join()
        .unwrap();
    assert_eq!(freed, TESSL_ERR);
    assert_eq!(
        unsafe { tessl_synchronize(rt, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK,
        "{}",
        msg(&err)
    );
    assert_eq!(unsafe { tessl_runtime_free(rt) }, TESSL_OK);
}

#[test]
fn the_python_binding_expects_this_abi_version() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/python/tessl_torch/__init__.py")).unwrap();
    let line = src
        .lines()
        .find(|l| l.starts_with("_ABI_VERSION = "))
        .expect("_ABI_VERSION in the binding");
    assert_eq!(line, format!("_ABI_VERSION = {TESSL_ABI_VERSION}"));
}
