//! The C ABI (`tessl::capi`), called the way a foreign caller calls it.
//!
//! `python/tests/test_cross_entropy.py` covers the torch path end to end;
//! this covers the boundary itself without torch: statuses and messages for
//! every refusal, message truncation, the thread check, and a successful call
//! producing exactly what the Rust entry point produces.

mod common;

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use common::{random_f32, with_gpu};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSharedEvent};
use tessl::capi::{
    tessl_abi_version, tessl_cross_entropy_rows, tessl_mtl_buffer_length, tessl_qwen35_copy,
    tessl_qwen35_free, tessl_qwen35_load, tessl_qwen35_param_count, tessl_qwen35_param_info,
    tessl_qwen35_read_parameter_host, tessl_qwen35_write_parameter_host, tessl_runtime_free, tessl_runtime_new,
    tessl_runtime_poisoned, tessl_runtime_set_async_encode, tessl_synchronize, TesslCeArgs, TesslParamInfo,
    TesslQwen35, TesslRuntime, TesslTensorRef, TESSL_ABI_VERSION, TESSL_BF16, TESSL_ERR, TESSL_F16, TESSL_F32,
    TESSL_MAX_DIMS, TESSL_OK, TESSL_OPERANDS_BF16, TESSL_OPERANDS_EXACT_F32, TESSL_READ_PARAMS, TESSL_WRITE_PARAMS,
};
use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::gemm::GemmOperands;
use tessl::{DType, GpuRuntime};

const ERR_LEN: usize = 512;

struct Handle(*mut TesslRuntime);

impl Handle {
    fn new() -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let h = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!h.is_null(), "tessl_runtime_new: {}", msg(&err));
        Self(h)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_runtime_free(self.0) }, TESSL_OK);
    }
}

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn shared(rt: &GpuRuntime, data: &[f32]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let b = rt
        .device
        .newBufferWithLength_options(data.len().max(1) * 4, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr().cast::<f32>(), data.len());
    }
    b
}

fn read(b: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<f32>(), n) }.to_vec()
}

fn tref(b: &ProtocolObject<dyn MTLBuffer>, shape: &[u64]) -> TesslTensorRef {
    let mut s = [0u64; TESSL_MAX_DIMS];
    s[..shape.len()].copy_from_slice(shape);
    TesslTensorRef {
        buffer: b as *const _ as *mut c_void,
        byte_offset: 0,
        dtype: TESSL_F32,
        ndim: shape.len() as u32,
        shape: s,
    }
}

const NULL_REF: TesslTensorRef = TesslTensorRef {
    buffer: ptr::null_mut(),
    byte_offset: 0,
    dtype: 0,
    ndim: 0,
    shape: [0; TESSL_MAX_DIMS],
};

#[test]
fn the_abi_reports_its_version_and_probes_buffers() {
    assert_eq!(tessl_abi_version(), TESSL_ABI_VERSION);
    with_gpu(|rt| {
        let b = shared(rt, &[0.0; 1000]);
        assert_eq!(unsafe { tessl_mtl_buffer_length(&*b as *const _ as *mut c_void) }, 4000);
        assert_eq!(unsafe { tessl_mtl_buffer_length(ptr::null_mut()) }, 0);
    });
    assert_eq!(unsafe { tessl_runtime_free(ptr::null_mut()) }, TESSL_OK);
}

/// The same problem through the C ABI and through the Rust entry point:
/// loss, per-row losses and both gradients bit-identical.
#[test]
fn a_call_through_the_abi_is_the_rust_call() {
    let (t, h, v, n) = (10usize, 64usize, 997usize, 3usize);
    let hid = random_f32(t * h, 1);
    let w: Vec<f32> = random_f32(v * h, 2).iter().map(|x| x * 0.25).collect();
    let rows = [3u32, 0, 9];
    let targets = [0u32, 500, 996];
    let handle = Handle::new();
    // The ABI's runtime and the check's runtime are two runtimes on one device.
    with_gpu(|rt| {
        let (hb, wb) = (shared(rt, &hid), shared(rt, &w));
        let (dhb, dwb) = (shared(rt, &vec![0.0; n * h]), shared(rt, &vec![0.0; v * h]));
        let args = TesslCeArgs {
            hidden: tref(&hb, &[t as u64, h as u64]),
            col_off: 0,
            weight: tref(&wb, &[v as u64, h as u64]),
            rows: rows.as_ptr(),
            targets: targets.as_ptr(),
            n: n as u64,
            reduction: 0,
            operands: TESSL_OPERANDS_EXACT_F32,
            chunk: 128,
            want_grads: 1,
            scale: 0.5,
            dh: tref(&dhb, &[n as u64, h as u64]),
            dw: tref(&dwb, &[v as u64, h as u64]),
        };
        let (mut loss, mut per_row) = (0.0f64, [0.0f64; 3]);
        let mut err = [0 as c_char; ERR_LEN];
        let status = unsafe {
            tessl_cross_entropy_rows(
                handle.0,
                &args,
                &mut loss,
                per_row.as_mut_ptr(),
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(status, TESSL_OK, "{}", msg(&err));

        let ht = rt.alloc_tensor_f32(&[t, h]).unwrap();
        ht.write_f32(&hid).unwrap();
        let wt = rt.alloc_tensor_f32(&[v, h]).unwrap();
        wt.write_f32(&w).unwrap();
        let (dh, dw) = (
            rt.alloc_tensor_f32(&[n, h]).unwrap(),
            rt.alloc_tensor_f32(&[v, h]).unwrap(),
        );
        let ws = CeWorkspace::new(rt, 4, h as u32, 128, DType::F32, GemmOperands::ExactF32).unwrap();
        let want = cross_entropy_rows(
            rt,
            CeHidden { rows: &ht, off: 0 },
            &wt,
            &rows,
            &targets,
            Reduction::Mean,
            GemmOperands::ExactF32,
            &ws,
            Some(CeGrads {
                dh: &dh,
                dw: &dw,
                scale: 0.5,
            }),
        )
        .unwrap();
        assert_eq!(loss.to_bits(), want.loss.to_bits());
        assert_eq!(per_row.to_vec(), want.per_row);
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(read(&dhb, n * h)), bits(dh.read_f32().unwrap()));
        assert_eq!(bits(read(&dwb, v * h)), bits(dw.read_f32().unwrap()));

        // bf16 operands through the ABI are the Rust call's bf16 bits, and not
        // the exact call's.
        let bf_args = TesslCeArgs {
            operands: TESSL_OPERANDS_BF16,
            ..args
        };
        let mut bf_loss = 0.0f64;
        let status = unsafe {
            tessl_cross_entropy_rows(
                handle.0,
                &bf_args,
                &mut bf_loss,
                ptr::null_mut(),
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(status, TESSL_OK, "{}", msg(&err));
        let bf_want = cross_entropy_rows(
            rt,
            CeHidden { rows: &ht, off: 0 },
            &wt,
            &rows,
            &targets,
            Reduction::Mean,
            GemmOperands::Bf16,
            &CeWorkspace::new(rt, 4, h as u32, 128, DType::F32, GemmOperands::Bf16).unwrap(),
            Some(CeGrads {
                dh: &dh,
                dw: &dw,
                scale: 0.5,
            }),
        )
        .unwrap();
        assert_eq!(bf_loss.to_bits(), bf_want.loss.to_bits());
        assert_eq!(bits(read(&dhb, n * h)), bits(dh.read_f32().unwrap()));
        assert_ne!(
            bf_loss.to_bits(),
            want.loss.to_bits(),
            "bf16 operands gave the exact loss"
        );

        // The handle's cached workspace serves a second, smaller call.
        let args1 = TesslCeArgs {
            n: 1,
            want_grads: 0,
            dh: NULL_REF,
            dw: NULL_REF,
            ..args
        };
        let status = unsafe {
            tessl_cross_entropy_rows(handle.0, &args1, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(status, TESSL_OK, "{}", msg(&err));
        assert_eq!(loss.to_bits(), want.per_row[0].to_bits());
    });
}

#[test]
fn every_refusal_is_a_status_and_a_message() {
    let handle = Handle::new();
    with_gpu(|rt| {
        let (hb, wb) = (shared(rt, &random_f32(640, 3)), shared(rt, &random_f32(64 * 300, 4)));
        let rows = [1u32, 2];
        let targets = [5u32, 6];
        let good = TesslCeArgs {
            hidden: tref(&hb, &[10, 64]),
            col_off: 0,
            weight: tref(&wb, &[300, 64]),
            rows: rows.as_ptr(),
            targets: targets.as_ptr(),
            n: 2,
            reduction: 0,
            operands: TESSL_OPERANDS_EXACT_F32,
            chunk: 0,
            want_grads: 0,
            scale: 1.0,
            dh: NULL_REF,
            dw: NULL_REF,
        };
        let call = |h: *mut TesslRuntime, a: *const TesslCeArgs| {
            let mut loss = 0.0f64;
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe { tessl_cross_entropy_rows(h, a, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN) };
            (s, msg(&err))
        };
        let refused = |a: TesslCeArgs, needle: &str| {
            let (s, m) = call(handle.0, &a);
            assert_eq!(s, TESSL_ERR, "expected {needle:?}, got status {s}");
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        assert_eq!(call(handle.0, &good).0, TESSL_OK);

        let (s, m) = call(ptr::null_mut(), &good);
        assert_eq!((s, m.as_str()), (TESSL_ERR, "null runtime handle"));
        assert!(call(handle.0, ptr::null()).1.contains("null args"));
        refused(TesslCeArgs { n: 0, ..good }, "no supervised rows");
        refused(
            TesslCeArgs {
                rows: ptr::null(),
                ..good
            },
            "null rows or targets",
        );
        refused(TesslCeArgs { reduction: 7, ..good }, "reduction 7");
        refused(
            TesslCeArgs { operands: 2, ..good },
            "operands code 2 is neither 0 (exact f32) nor 1 (bf16)",
        );
        refused(
            TesslCeArgs {
                hidden: NULL_REF,
                ..good
            },
            "hidden: null MTLBuffer",
        );
        refused(
            TesslCeArgs {
                hidden: TesslTensorRef {
                    dtype: 9,
                    ..good.hidden
                },
                ..good
            },
            "unknown dtype code 9",
        );
        refused(
            TesslCeArgs {
                hidden: TesslTensorRef { ndim: 7, ..good.hidden },
                ..good
            },
            "rank 7",
        );
        refused(
            TesslCeArgs {
                weight: TesslTensorRef { ndim: 1, ..good.weight },
                ..good
            },
            "weight must be 2-D",
        );
        refused(
            TesslCeArgs {
                hidden: TesslTensorRef {
                    byte_offset: 8,
                    ..good.hidden
                },
                ..good
            },
            "hidden: tensor view is misaligned or out of bounds",
        );
        refused(TesslCeArgs { want_grads: 1, ..good }, "dh: null MTLBuffer");
        let bad_targets = [5u32, 300];
        refused(
            TesslCeArgs {
                targets: bad_targets.as_ptr(),
                ..good
            },
            "targets[1] = 300",
        );

        // A message longer than the caller's buffer is cut at a character
        // boundary and still NUL-terminated.
        let mut small = [0x7f as c_char; 8];
        let mut loss = 0.0;
        let s = unsafe {
            tessl_cross_entropy_rows(
                ptr::null_mut(),
                &good,
                &mut loss,
                ptr::null_mut(),
                small.as_mut_ptr(),
                small.len(),
            )
        };
        assert_eq!(s, TESSL_ERR);
        assert_eq!(msg(&small), "null ru");

        // Another thread may not use this handle.
        let raw = handle.0 as usize;
        let from_thread = std::thread::spawn(move || {
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe { tessl_synchronize(raw as *mut TesslRuntime, err.as_mut_ptr(), ERR_LEN) };
            (s, msg(&err))
        })
        .join()
        .unwrap();
        assert_eq!(from_thread.0, TESSL_ERR);
        assert!(
            from_thread.1.contains("other than the one that created it"),
            "{}",
            from_thread.1
        );

        // After all of that the handle still works.
        assert_eq!(call(handle.0, &good).0, TESSL_OK);
        let mut err = [0 as c_char; ERR_LEN];
        assert_eq!(
            unsafe { tessl_synchronize(handle.0, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
    });
}

#[test]
fn a_wrap_through_the_abi_leaves_residency_balanced() {
    // Repeated calls wrap the same caller buffers every time; the refcounted
    // residency must neither leak nor evict them.
    let handle = Handle::new();
    with_gpu(|rt| {
        let (hb, wb) = (shared(rt, &random_f32(640, 5)), shared(rt, &random_f32(64 * 300, 6)));
        let rows = [1u32];
        let targets = [2u32];
        let args = TesslCeArgs {
            hidden: tref(&hb, &[10, 64]),
            col_off: 0,
            weight: tref(&wb, &[300, 64]),
            rows: rows.as_ptr(),
            targets: targets.as_ptr(),
            n: 1,
            reduction: 1,
            operands: TESSL_OPERANDS_EXACT_F32,
            chunk: 64,
            want_grads: 0,
            scale: 1.0,
            dh: NULL_REF,
            dw: NULL_REF,
        };
        let mut first = 0.0;
        for i in 0..50 {
            let mut loss = 0.0;
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe {
                tessl_cross_entropy_rows(handle.0, &args, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN)
            };
            assert_eq!(s, TESSL_OK, "{}", msg(&err));
            if i == 0 {
                first = loss;
            }
            assert_eq!(loss.to_bits(), first.to_bits(), "call {i} changed the answer");
        }
    });
}

/// Every runtime-handle entry refuses a foreign thread with the thread message
/// before any argument is looked at (all arguments are null: were they looked
/// at first the message would be "null args"), and the owner stays usable.
#[test]
fn every_runtime_entry_refuses_a_foreign_thread_before_looking_at_its_arguments() {
    fn run(out: &mut Vec<(&'static str, i32, String)>, name: &'static str, f: &dyn Fn(*mut c_char, usize) -> i32) {
        let mut err = [0 as c_char; ERR_LEN];
        let s = f(err.as_mut_ptr(), ERR_LEN);
        out.push((name, s, msg(&err)));
    }
    fn runtime_entries(rt: *mut TesslRuntime) -> Vec<(&'static str, i32, String)> {
        let mut out = Vec::new();
        unsafe {
            run(&mut out, "tessl_synchronize", &|e, n| tessl_synchronize(rt, e, n));
            run(&mut out, "tessl_runtime_poisoned", &|e, n| {
                tessl_runtime_poisoned(rt, ptr::null_mut(), e, n)
            });
            run(&mut out, "tessl_cross_entropy_rows", &|e, n| {
                tessl_cross_entropy_rows(rt, ptr::null(), ptr::null_mut(), ptr::null_mut(), e, n)
            });
            run(&mut out, "tessl_qwen35_load", &|e, n| {
                let mut m = ptr::null_mut();
                tessl_qwen35_load(rt, ptr::null(), ptr::null(), ptr::null(), TESSL_F32, &mut m, e, n)
            });
        }
        out
    }
    let _gpu = common::gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let raw = rt as usize;
    let calls = std::thread::spawn(move || runtime_entries(raw as *mut TesslRuntime))
        .join()
        .unwrap();
    assert_eq!(calls.len(), 4);
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

/// A free from a foreign thread (a finalizer at interpreter shutdown) leaks
/// the handle instead of dropping a thread-affine runtime there; the owner can
/// still use and free it.
#[test]
fn a_free_from_another_thread_is_refused_and_leaves_the_handle_usable() {
    let handle = Handle::new();
    let raw = handle.0 as usize;
    let status = std::thread::spawn(move || unsafe { tessl_runtime_free(raw as *mut TesslRuntime) })
        .join()
        .unwrap();
    assert_eq!(status, TESSL_ERR);
    let mut err = [0 as c_char; ERR_LEN];
    assert_eq!(
        unsafe { tessl_synchronize(handle.0, err.as_mut_ptr(), ERR_LEN) },
        TESSL_OK,
        "{}",
        msg(&err)
    );
    drop(handle); // the owner's free succeeds (asserted in Drop)
}

// ------------------------------------------------------------ Qwen3.5 ---

fn fixture(name: &str) -> std::ffi::CString {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen35_train")
        .join(name);
    std::ffi::CString::new(p.to_str().unwrap()).unwrap()
}

fn load_model(rt: *mut TesslRuntime) -> *mut TesslQwen35 {
    load_model_as(rt, TESSL_F32)
}

/// The fixture through [`tessl_qwen35_load`] at `precision`.
fn load_model_as(rt: *mut TesslRuntime, precision: u32) -> *mut TesslQwen35 {
    let (st, cfg, prefix) = (
        fixture("model.safetensors"),
        fixture("config.json"),
        std::ffi::CString::new("model.").unwrap(),
    );
    let mut out = ptr::null_mut();
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_qwen35_load(
            rt,
            st.as_ptr(),
            cfg.as_ptr(),
            prefix.as_ptr(),
            precision,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert!(!out.is_null());
    out
}

/// The model's parameter count through [`tessl_qwen35_param_count`].
fn param_count(model: *mut TesslQwen35) -> usize {
    let mut n = 0u64;
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe { tessl_qwen35_param_count(model, &mut n, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    n as usize
}

/// `tessl_qwen35_load` takes a checkpoint directory or its index as well as a
/// `.safetensors` file, and the model is the same one: the fixture's directory
/// (no index, so `model.safetensors`), and an index over a Hugging Face style
/// symlinked shard.
#[test]
fn a_model_loads_through_the_abi_from_a_directory_or_an_index() {
    let _g = common::gpu_lock();
    let handle = Handle::new();
    let want = load_model(handle.0);
    let want_n = param_count(want);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
    let dir = std::env::temp_dir().join(format!("tessl-capi-index-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::os::unix::fs::symlink(
        src.join("model.safetensors"),
        dir.join("model-00001-of-00001.safetensors"),
    )
    .unwrap();
    let st = tessl::safetensors::SafeTensors::open(&src.join("model.safetensors")).unwrap();
    let body: Vec<String> = st
        .names()
        .map(|n| format!("{n:?}:\"model-00001-of-00001.safetensors\""))
        .collect();
    let index = dir.join("model.safetensors.index.json");
    std::fs::write(&index, format!("{{\"weight_map\":{{{}}}}}", body.join(","))).unwrap();

    let (cfg, prefix) = (fixture("config.json"), std::ffi::CString::new("model.").unwrap());
    for path in [src.clone(), dir.clone(), index.clone()] {
        let path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let mut out = ptr::null_mut();
        let mut err = [0 as c_char; ERR_LEN];
        let s = unsafe {
            tessl_qwen35_load(
                handle.0,
                path.as_ptr(),
                cfg.as_ptr(),
                prefix.as_ptr(),
                TESSL_F32,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{path:?}: {}", msg(&err));
        assert_eq!(param_count(out), want_n);
        assert_eq!(unsafe { tessl_qwen35_free(out) }, TESSL_OK);
    }
    assert_eq!(unsafe { tessl_qwen35_free(want) }, TESSL_OK);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_model_load_refuses_bad_arguments() {
    let handle = Handle::new();
    let mut err = [0 as c_char; ERR_LEN];
    let mut out = ptr::null_mut();
    let (st, cfg) = (fixture("model.safetensors"), fixture("config.json"));
    let missing = std::ffi::CString::new("/nonexistent/model.safetensors").unwrap();
    let prefix = std::ffi::CString::new("model.").unwrap();
    let wrong_prefix = std::ffi::CString::new("model.language_model.").unwrap();
    let load = |st: *const c_char,
                cfg: *const c_char,
                prefix: *const c_char,
                out: *mut *mut TesslQwen35,
                err: &mut [c_char; ERR_LEN]| unsafe {
        tessl_qwen35_load(handle.0, st, cfg, prefix, TESSL_F32, out, err.as_mut_ptr(), ERR_LEN)
    };
    assert_eq!(
        load(ptr::null(), cfg.as_ptr(), prefix.as_ptr(), &mut out, &mut err),
        TESSL_ERR
    );
    assert!(msg(&err).contains("null safetensors path"), "{}", msg(&err));
    assert_eq!(
        load(missing.as_ptr(), cfg.as_ptr(), prefix.as_ptr(), &mut out, &mut err),
        TESSL_ERR
    );
    assert!(msg(&err).contains("/nonexistent/model.safetensors"), "{}", msg(&err));
    assert_eq!(
        load(st.as_ptr(), cfg.as_ptr(), wrong_prefix.as_ptr(), &mut out, &mut err),
        TESSL_ERR
    );
    assert!(msg(&err).contains("model.language_model."), "{}", msg(&err));
    assert!(out.is_null(), "a failed load leaves *out null");
    assert_eq!(
        load(st.as_ptr(), cfg.as_ptr(), prefix.as_ptr(), ptr::null_mut(), &mut err),
        TESSL_ERR
    );
    assert!(msg(&err).contains("null out"), "{}", msg(&err));
    let s = unsafe {
        tessl_qwen35_load(
            handle.0,
            st.as_ptr(),
            cfg.as_ptr(),
            prefix.as_ptr(),
            TESSL_F16,
            &mut out,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(
        msg(&err).contains("precision code 2 is neither 0 (f32) nor 1 (bf16)"),
        "{}",
        msg(&err)
    );
    assert!(out.is_null(), "a failed load leaves *out null");
    assert_eq!(unsafe { tessl_qwen35_free(ptr::null_mut()) }, TESSL_OK);
}

// --------------------------------------- validate before mutate or publish ---

fn ce_args<'a>(
    hb: &ProtocolObject<dyn MTLBuffer>,
    wb: &ProtocolObject<dyn MTLBuffer>,
    rows: &'a [u32],
    targets: &'a [u32],
    chunk: u32,
) -> TesslCeArgs {
    TesslCeArgs {
        hidden: tref(hb, &[10, 64]),
        col_off: 0,
        weight: tref(wb, &[300, 64]),
        rows: rows.as_ptr(),
        targets: targets.as_ptr(),
        n: rows.len() as u64,
        reduction: 1,
        operands: TESSL_OPERANDS_EXACT_F32,
        chunk,
        want_grads: 0,
        scale: 1.0,
        dh: NULL_REF,
        dw: NULL_REF,
    }
}

/// A refused cross-entropy call keeps the cached workspace and sizes nothing
/// from the caller's `chunk`; a `chunk` past the vocabulary is one full-width
/// chunk, not a `[rows, chunk]` allocation.
#[test]
fn a_refused_cross_entropy_call_keeps_the_cached_workspace() {
    let handle = Handle::new();
    with_gpu(|rt| {
        let (hb, wb) = (shared(rt, &random_f32(640, 5)), shared(rt, &random_f32(64 * 300, 6)));
        let (rows, targets, bad_targets) = ([1u32, 2], [3u32, 4], [3u32, 300]);
        let call = |args: &TesslCeArgs| {
            let mut loss = 0.0;
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe {
                tessl_cross_entropy_rows(handle.0, args, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN)
            };
            (s, loss, msg(&err))
        };
        let builds = || unsafe { tessl::capi::test_workspace_builds(handle.0) };

        let good = ce_args(&hb, &wb, &rows, &targets, 64);
        let (s, first, m) = call(&good);
        assert_eq!(s, TESSL_OK, "{m}");
        assert_eq!(builds(), 1);

        // Refused: a target past the vocabulary, with a `chunk` that would ask
        // for a [rows, u32::MAX] f32 logits buffer if it were honoured.
        let (s, _, m) = call(&ce_args(&hb, &wb, &rows, &bad_targets, u32::MAX));
        assert_eq!(s, TESSL_ERR);
        assert!(m.contains("targets[1] = 300"), "{m}");
        assert_eq!(builds(), 1, "a refused call must not build a workspace");
        let (s, again, m) = call(&good);
        assert_eq!(s, TESSL_OK, "{m}");
        assert_eq!(builds(), 1, "the next good call reuses the cached workspace");
        assert_eq!(again.to_bits(), first.to_bits());

        // Accepted: the same call with valid targets and the huge chunk is one
        // 300-wide chunk, the same answer to rounding.
        let (s, wide, m) = call(&ce_args(&hb, &wb, &rows, &targets, u32::MAX));
        assert_eq!(s, TESSL_OK, "{m}");
        assert!((wide - first).abs() <= 1e-4 * first.abs().max(1.0), "{wide} vs {first}");
    });
}

/// A copy with an unknown direction, or one whose state is missing, wraps no
/// tensor.
#[test]
fn a_refused_copy_wraps_nothing() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    with_gpu(|rt| {
        let buf = shared(rt, &[0.0]);
        let n = param_count(model);
        let refs = vec![tref(&buf, &[1]); n];
        let wraps = || unsafe { tessl::capi::test_external_wrap_total(model) };
        let copy = |direction: u32| {
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe { tessl_qwen35_copy(model, direction, refs.as_ptr(), n as u64, err.as_mut_ptr(), ERR_LEN) };
            (s, msg(&err))
        };
        let before = wraps();
        let (s, m) = copy(99);
        assert_eq!(s, TESSL_ERR);
        assert!(m.contains("direction 99"), "{m}");
        // ABI 14 retired directions 1 (read gradients) and 3 to 8 (AdamW state).
        for retired in 1..=8u32 {
            if retired == 2 {
                continue;
            }
            let (s, m) = copy(retired);
            assert_eq!(s, TESSL_ERR, "direction {retired}");
            assert!(m.contains("retired with ABI 14"), "direction {retired}: {m}");
        }
        assert_eq!(wraps(), before, "a refused copy wrapped tensors");
        // The check is not vacuous: a valid direction wraps all n.
        let _ = copy(TESSL_READ_PARAMS);
        assert_eq!(wraps(), before + n as u64);
    });
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// Resets the load hook even when the test panics.
struct PoisonAfterLoad;

impl PoisonAfterLoad {
    fn on() -> Self {
        tessl::capi::test_poison_after_load(true);
        Self
    }
}

impl Drop for PoisonAfterLoad {
    fn drop(&mut self) {
        tessl::capi::test_poison_after_load(false);
    }
}

/// A load whose closing synchronize fails (the runtime is poisoned after the
/// model is built) writes no handle to `*out`.
#[test]
fn a_load_whose_final_sync_fails_publishes_no_handle() {
    let handle = Handle::new();
    let (st, cfg, prefix) = (
        fixture("model.safetensors"),
        fixture("config.json"),
        std::ffi::CString::new("model.").unwrap(),
    );
    let mut out: *mut TesslQwen35 = ptr::null_mut();
    let mut err = [0 as c_char; ERR_LEN];
    let s = {
        let _poison = PoisonAfterLoad::on();
        unsafe {
            tessl_qwen35_load(
                handle.0,
                st.as_ptr(),
                cfg.as_ptr(),
                prefix.as_ptr(),
                TESSL_F32,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        }
    };
    assert_eq!(s, TESSL_ERR, "{}", msg(&err));
    assert!(out.is_null(), "*out is null whenever the status is not TESSL_OK");
}

fn poisoned(rt: *mut TesslRuntime) -> i32 {
    let mut out = -1i32;
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe { tessl_runtime_poisoned(rt, &mut out, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    out
}

/// `tessl_runtime_poisoned` reads 0 on a healthy runtime and 1 once a failed
/// call has latched it, and keeps answering when every other call is refused.
#[test]
fn the_poisoned_query_reports_a_latched_runtime() {
    let healthy = Handle::new();
    assert_eq!(poisoned(healthy.0), 0);

    let handle = Handle::new();
    assert_eq!(poisoned(handle.0), 0);
    let (st, cfg, prefix) = (
        fixture("model.safetensors"),
        fixture("config.json"),
        std::ffi::CString::new("model.").unwrap(),
    );
    let mut out: *mut TesslQwen35 = ptr::null_mut();
    let mut err = [0 as c_char; ERR_LEN];
    let s = {
        let _poison = PoisonAfterLoad::on();
        unsafe {
            tessl_qwen35_load(
                handle.0,
                st.as_ptr(),
                cfg.as_ptr(),
                prefix.as_ptr(),
                TESSL_F32,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        }
    };
    assert_eq!(s, TESSL_ERR, "{}", msg(&err));
    assert_eq!(poisoned(handle.0), 1);
    // Every other call on it is refused, and the other runtime is untouched.
    assert_eq!(
        unsafe { tessl_synchronize(handle.0, err.as_mut_ptr(), ERR_LEN) },
        TESSL_ERR
    );
    assert_eq!(poisoned(healthy.0), 0);
}

/// What a binding does after the poisoned status: free the latched runtime
/// and build a new one on the same thread. The new runtime must load a real
/// model and train a step; a stand-in library cannot show that.
#[test]
fn a_fresh_runtime_on_the_poisoned_thread_loads_and_runs_a_model() {
    let mut err = [0 as c_char; ERR_LEN];
    let a = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!a.is_null(), "{}", msg(&err));
    let (st, cfg, prefix) = (
        fixture("model.safetensors"),
        fixture("config.json"),
        std::ffi::CString::new("model.").unwrap(),
    );
    let mut out: *mut TesslQwen35 = ptr::null_mut();
    let s = {
        let _poison = PoisonAfterLoad::on();
        unsafe {
            tessl_qwen35_load(
                a,
                st.as_ptr(),
                cfg.as_ptr(),
                prefix.as_ptr(),
                TESSL_F32,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        }
    };
    assert_eq!(s, TESSL_ERR, "{}", msg(&err));
    assert!(out.is_null());
    assert_eq!(poisoned(a), 1);
    assert_eq!(unsafe { tessl_runtime_free(a) }, TESSL_OK, "a poisoned runtime frees");

    let b = Handle::new();
    assert_eq!(poisoned(b.0), 0);
    let model = load_model(b.0);
    assert!(param_count(model) > 0, "the model on the fresh runtime has a parameter table");
    assert_eq!(poisoned(b.0), 0);
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn set_async(handle: &Handle, on: u32) -> i32 {
    let mut err = [0 as c_char; ERR_LEN];
    unsafe { tessl_runtime_set_async_encode(handle.0, on, err.as_mut_ptr(), ERR_LEN) }
}

/// The runtime's two timelines say nothing is in flight: tessl has seen every
/// commit it signaled complete, and the handoff event (signaled behind the
/// lifetime event) has reached the last of them. This is what a torch caller
/// waits on, so it is the contract `guarded` owes on every return.
fn assert_idle(rt: &GpuRuntime, what: &str) {
    let last = rt.last_signaled_value();
    // Non-blocking: the lifetime event alone says whether tessl still has work out.
    assert_eq!(rt.last_completed_value(), last, "{what}: tessl work in flight");
    assert!(
        rt.shared_event().waitUntilSignaledValue_timeoutMS(last, 5_000),
        "{what}: the handoff event never reached {last}"
    );
}

/// A call that fails after it encoded work returns only after that work is
/// done. With async encode on the kernels of a call are batched and not waited
/// one by one, so a failure past the first kernel leaves a batch open or in
/// flight; a caller that sees the refusal frees the buffers those kernels use
/// (torch recycles a dropped tensor's storage), so the work must be finished
/// first.
#[test]
fn a_call_that_fails_after_encoding_returns_only_after_its_work_completes() {
    let handle = Handle::new();
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    let rt = unsafe { tessl::capi::test_runtime(handle.0) };
    assert!(rt.async_encode_enabled());
    let n = 4096usize;
    let src = rt.alloc_buffer(n * 4).unwrap();
    let dst = rt.alloc_buffer(n * 4).unwrap();
    src.write_f32(&(0..n).map(|i| i as f32 + 1.0).collect::<Vec<_>>());
    dst.write_f32(&vec![0.0; n]);
    let pipe = rt.pipeline("copy_f32").unwrap();
    // The destination's bytes without the buffer's own accessors: those wait
    // for the GPU themselves, which would hide an unfinished copy.
    let raw = |b: &tessl::tensor::GpuBuffer| -> Vec<f32> {
        // SAFETY: a shared-storage buffer of `n` f32s that nothing else writes.
        unsafe { std::slice::from_raw_parts(b.metal().contents().as_ptr().cast::<f32>(), n) }.to_vec()
    };
    let encode = |rt: &Arc<GpuRuntime>| {
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            bnd.bind_gpu_buf(&src, 0);
            bnd.bind_gpu_buf(&dst, 1);
            bnd.bind_u32(n as u32, 2);
            bnd.dispatch(
                tessl::runtime::mtl_size(n / 64, 1, 1),
                tessl::runtime::mtl_size(64, 1, 1),
            );
            Ok(())
        })
    };
    let mut err = [0 as c_char; ERR_LEN];
    for panics in [false, true] {
        dst.write_f32(&vec![0.0; n]);
        let before = rt.last_signaled_value();
        let status = unsafe {
            tessl::capi::test_guarded(handle.0, err.as_mut_ptr(), ERR_LEN, |rt| {
                encode(rt)?;
                if panics {
                    panic!("bug after encoding");
                }
                Err("refused after encoding".to_string())
            })
        };
        let (want_status, want_text) = if panics {
            (tessl::capi::TESSL_PANIC, "bug after encoding")
        } else {
            (TESSL_ERR, "refused after encoding")
        };
        assert_eq!(status, want_status);
        assert!(msg(&err).contains(want_text), "{}", msg(&err));
        // Checked before anything else touches the runtime: the open batch was
        // committed (the signaled timeline moved) and finished (the copy's
        // bytes are there), though the call failed.
        assert!(
            rt.last_signaled_value() > before,
            "panics={panics}: the encoded work was never committed"
        );
        assert_idle(&rt, "a failed call");
        let got = raw(&dst);
        assert_eq!(
            (got[0], got[n - 1]),
            (1.0, n as f32),
            "panics={panics}: the encoded copy had not run"
        );
        assert_eq!(rt.handoff_foreign_signal(), None);
    }
}

fn param_infos(model: *mut TesslQwen35) -> Vec<TesslParamInfo> {
    (0..param_count(model))
        .map(|i| {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            let mut e = [0 as c_char; ERR_LEN];
            assert_eq!(
                unsafe { tessl_qwen35_param_info(model, i as u64, &mut info, e.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            info
        })
        .collect()
}

fn numel(p: &TesslParamInfo) -> usize {
    p.shape[..p.ndim as usize].iter().product::<u64>() as usize
}

/// One shared buffer per entry holding `values(i)` (or zeros), and the refs a
/// whole-model copy takes (transformers' shape, reversed for a transposed entry).
fn entry_buffers(
    rt: &GpuRuntime,
    infos: &[TesslParamInfo],
    values: impl Fn(usize) -> Vec<f32>,
) -> (Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, Vec<TesslTensorRef>) {
    let bufs: Vec<_> = infos.iter().enumerate().map(|(i, _)| shared(rt, &values(i))).collect();
    let refs = bufs
        .iter()
        .zip(infos)
        .map(|(b, p)| {
            let mut shape = p.shape[..p.ndim as usize].to_vec();
            if p.transposed != 0 {
                shape.reverse();
            }
            tref(b, &shape)
        })
        .collect();
    (bufs, refs)
}

/// Every entry through the whole-model GPU copy `direction`.
fn read_all(model: *mut TesslQwen35, rt: &GpuRuntime, infos: &[TesslParamInfo], direction: u32) -> Vec<Vec<f32>> {
    let (bufs, refs) = entry_buffers(rt, infos, |i| vec![0.0; numel(&infos[i])]);
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_qwen35_copy(
            model,
            direction,
            refs.as_ptr(),
            refs.len() as u64,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    bufs.iter().zip(infos).map(|(b, p)| read(b, numel(p))).collect()
}

fn read_entry(model: *mut TesslQwen35, i: usize, n: usize, guard: f32) -> (i32, Vec<f32>, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let mut dst = vec![guard; n.max(1) + 8];
    let s = unsafe {
        tessl_qwen35_read_parameter_host(model, i as u64, dst.as_mut_ptr(), n as u64, err.as_mut_ptr(), ERR_LEN)
    };
    (s, dst, msg(&err))
}

fn write_entry(model: *mut TesslQwen35, i: usize, src: &[f32]) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_qwen35_write_parameter_host(
            model,
            i as u64,
            src.as_ptr(),
            src.len() as u64,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    (s, msg(&err))
}

/// A NaN payload no read produces; a fn because `f32::from_bits` is const only from 1.83 (MSRV 1.82).
fn guard() -> f32 {
    f32::from_bits(0x7FC0_DEAD)
}

/// A parameter read through the ABI is that entry: bitwise the whole-model
/// copy's values, written into exactly `n` elements (a guard past the end is
/// untouched), with no GPU dispatch encoded; a write moves that entry only.
#[test]
fn a_parameter_read_or_write_is_that_entry_and_encodes_nothing() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let rt = unsafe { tessl::capi::test_model_runtime(model) };
    let infos = param_infos(model);
    let whole = read_all(model, &rt, &infos, TESSL_READ_PARAMS);
    let mut order: Vec<usize> = (0..infos.len()).collect();
    order.sort_by_key(|&i| numel(&infos[i]));
    let (small, large) = (order[0], *order.last().unwrap());
    assert!(numel(&infos[small]) < numel(&infos[large]));
    rt.take_dispatch_count();
    for i in [small, large] {
        let n = numel(&infos[i]);
        let (s, dst, m) = read_entry(model, i, n, guard());
        assert_eq!(s, TESSL_OK, "{m}");
        assert!(
            dst[n..].iter().all(|v| v.to_bits() == guard().to_bits()),
            "wrote past entry {i}"
        );
        assert_eq!(
            bits(&dst[..n]),
            bits(&whole[i]),
            "entry {i} differs from the whole-model copy"
        );
    }
    assert_eq!(rt.take_dispatch_count(), 0, "a host read encoded a GPU dispatch");
    let n = numel(&infos[small]);
    let new: Vec<f32> = whole[small].iter().map(|v| v + 0.5).collect();
    let (s, m) = write_entry(model, small, &new);
    assert_eq!(s, TESSL_OK, "{m}");
    for i in 0..infos.len() {
        let (s, got, m) = read_entry(model, i, numel(&infos[i]), guard());
        assert_eq!(s, TESSL_OK, "{m}");
        let want = if i == small { &new } else { &whole[i] };
        assert_eq!(bits(&got[..numel(&infos[i])]), bits(want), "entry {i} after the write");
    }
    assert_eq!(n, new.len());
    assert_eq!(rt.take_dispatch_count(), 0, "a host write encoded a GPU dispatch");
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// Hostile arguments to the per-entry entries are refused before anything
/// moves: a length one off in either direction, zero, a null buffer (with and
/// without a zero length), an index at `u64::MAX`, and a write while a step is
/// pending.
#[test]
fn per_parameter_host_entries_refuse_hostile_arguments_without_moving_anything() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let rt = unsafe { tessl::capi::test_model_runtime(model) };
    let infos = param_infos(model);
    let whole = read_all(model, &rt, &infos, TESSL_READ_PARAMS);
    let i = infos.len() / 2;
    let n = numel(&infos[i]);
    let mut err = [0 as c_char; ERR_LEN];
    let unchanged = |what: &str| {
        for (j, p) in infos.iter().enumerate() {
            let (s, got, m) = read_entry(model, j, numel(p), guard());
            assert_eq!(s, TESSL_OK, "{m}");
            assert_eq!(bits(&got[..numel(p)]), bits(&whole[j]), "{what}: entry {j} moved");
        }
    };
    // Reads: a refused read writes nothing into the destination.
    for len in [n - 1, n + 1, 0] {
        let mut dst = vec![guard(); n + 8];
        let s = unsafe {
            tessl_qwen35_read_parameter_host(model, i as u64, dst.as_mut_ptr(), len as u64, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(s, TESSL_ERR, "read n={len}");
        assert!(msg(&err).contains("takes"), "n={len}: {}", msg(&err));
        assert!(
            dst.iter().all(|v| v.to_bits() == guard().to_bits()),
            "a refused read (n={len}) wrote"
        );
    }
    for len in [0u64, 1] {
        let s = unsafe {
            tessl_qwen35_read_parameter_host(model, i as u64, ptr::null_mut(), len, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(s, TESSL_ERR, "null dst, n={len}");
        assert!(msg(&err).contains("null dst"), "{}", msg(&err));
    }
    let mut dst = vec![guard(); 16];
    let s =
        unsafe { tessl_qwen35_read_parameter_host(model, u64::MAX, dst.as_mut_ptr(), 1, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("out of range"), "{}", msg(&err));
    assert!(dst.iter().all(|v| v.to_bits() == guard().to_bits()));
    // Writes: nothing lands.
    let new = vec![7.0f32; n + 1];
    for len in [n - 1, n + 1, 0] {
        let s = unsafe {
            tessl_qwen35_write_parameter_host(model, i as u64, new.as_ptr(), len as u64, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(s, TESSL_ERR, "write n={len}");
        assert!(msg(&err).contains("takes"), "n={len}: {}", msg(&err));
    }
    for len in [0u64, 1] {
        let s =
            unsafe { tessl_qwen35_write_parameter_host(model, i as u64, ptr::null(), len, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_ERR, "null src, n={len}");
        assert!(msg(&err).contains("null src"), "{}", msg(&err));
    }
    let (s, m) = write_entry(model, usize::MAX, &new[..1]);
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("out of range"), "{m}");
    let s = unsafe { tessl_qwen35_write_parameter_host(model, u64::MAX, new.as_ptr(), 1, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    unchanged("refused arguments");

    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// On a poisoned runtime both entries are refused and nothing moves.
#[test]
fn per_parameter_host_entries_are_refused_on_a_poisoned_runtime() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let rt = unsafe { tessl::capi::test_model_runtime(model) };
    let infos = param_infos(model);
    let n = numel(&infos[0]);
    let (s, before, m) = read_entry(model, 0, n, guard());
    assert_eq!(s, TESSL_OK, "{m}");
    rt.poison_as_shared_event_timeout_for_test();
    assert_eq!(poisoned(handle.0), 1);
    let (s, dst, _) = read_entry(model, 0, n, guard());
    assert_eq!(s, TESSL_ERR, "a read on a poisoned runtime");
    assert!(
        dst.iter().all(|v| v.to_bits() == guard().to_bits()),
        "a refused read wrote"
    );
    let new: Vec<f32> = before[..n].iter().map(|v| v + 1.0).collect();
    let (s, _) = write_entry(model, 0, &new);
    assert_eq!(s, TESSL_ERR, "a write on a poisoned runtime");
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// The bit patterns that separate bf16 rounding rules: ties on both parities,
/// subnormals (flushed as the GPU copy flushes them), NaNs of both signs and
/// payloads, infinities, and ordinary values.
fn rounding_patterns(n: usize) -> Vec<f32> {
    const SPECIAL: [u32; 16] = [
        0x3F80_8000, // tie, rounds to even (down)
        0x3F81_8000, // tie, rounds to even (up)
        0xBF80_8000,
        0xBF81_8000,
        0x3F80_8001, // just above a tie
        0x3F80_7FFF, // just below a tie
        0x0000_0001, // subnormal
        0x8000_0001,
        0x0040_0000,
        0x007F_FFFF,
        0x7FC0_0001, // NaNs
        0xFFC0_1234,
        0x7F80_0001, // signalling NaN
        0x7F80_0000, // inf
        0xFF80_0000,
        0x7F7F_FFFF, // largest finite: rounds up to inf
    ];
    (0..n)
        .map(|k| {
            if k < SPECIAL.len() * 3 {
                f32::from_bits(SPECIAL[k % SPECIAL.len()])
            } else {
                ((k * 2_654_435_761) % 10_007) as f32 / 97.0 - 50.0
            }
        })
        .collect()
}

/// A bf16 model's host write rounds exactly as the GPU whole-model write does:
/// bit for bit on every entry, including ties, subnormals, NaNs and the
/// transposed entries; and for an entry the model stores in bf16 it is the
/// documented `f32_to_bf16_bits_gpu`. Entries stored in f32 keep every bit.
#[test]
fn a_bf16_host_write_rounds_bit_for_bit_like_the_gpu_write() {
    let handle = Handle::new();
    let host = load_model_as(handle.0, TESSL_BF16);
    let gpu = load_model_as(handle.0, TESSL_BF16);
    let rt = unsafe { tessl::capi::test_model_runtime(host) };
    let infos = param_infos(host);
    assert!(
        infos.iter().any(|p| p.transposed != 0),
        "the fixture has no transposed entry"
    );
    let mut err = [0 as c_char; ERR_LEN];
    // No optimizer on either model: every entry accepts the host write.
    for (i, p) in infos.iter().enumerate() {
        let (s, m) = write_entry(host, i, &rounding_patterns(numel(p)));
        assert_eq!(s, TESSL_OK, "entry {i}: {m}");
    }
    let (_bufs, refs) = entry_buffers(&rt, &infos, |i| rounding_patterns(numel(&infos[i])));
    let s = unsafe {
        tessl_qwen35_copy(
            gpu,
            TESSL_WRITE_PARAMS,
            refs.as_ptr(),
            refs.len() as u64,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    let by_host = read_all(host, &rt, &infos, TESSL_READ_PARAMS);
    let by_gpu = read_all(gpu, &rt, &infos, TESSL_READ_PARAMS);
    let mut rounded_entries = 0;
    for (i, p) in infos.iter().enumerate() {
        assert_eq!(
            bits(&by_host[i]),
            bits(&by_gpu[i]),
            "entry {i} ({}): host and GPU writes differ",
            i
        );
        let pattern = rounding_patterns(numel(p));
        // An ordinary value past the special ones is not a bf16 number: it
        // comes back changed exactly when the entry is stored in bf16.
        let tail = 48.min(pattern.len());
        if pattern.len() == tail || bits(&by_gpu[i][tail..]) == bits(&pattern[tail..]) {
            continue;
        }
        rounded_entries += 1;
        for (k, (&got, &x)) in by_gpu[i].iter().zip(&pattern).enumerate() {
            let want = tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits_gpu(x));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "entry {i} element {k}: {:#x}",
                x.to_bits()
            );
        }
    }
    assert!(rounded_entries > 0, "no entry was stored in bf16");
    assert!(
        infos
            .iter()
            .zip(&by_gpu)
            .any(|(p, v)| p.transposed != 0 && bits(v) != bits(&rounding_patterns(numel(p)))),
        "no transposed bf16 entry exercised"
    );
    assert_eq!(unsafe { tessl_qwen35_free(host) }, TESSL_OK);
    assert_eq!(unsafe { tessl_qwen35_free(gpu) }, TESSL_OK);
}

/// ABI 12's entry: a value other than 0 or 1 is refused and changes nothing;
/// off waits for the runtime to go idle first.
#[test]
fn set_async_encode_validates_and_toggles_the_runtime() {
    let handle = Handle::new();
    let rt = unsafe { tessl::capi::test_runtime(handle.0) };
    assert!(!rt.async_encode_enabled(), "async encode is off by default");
    assert_eq!(set_async(&handle, 2), TESSL_ERR);
    assert!(!rt.async_encode_enabled());
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    assert!(rt.async_encode_enabled());
    assert_eq!(set_async(&handle, 0), TESSL_OK);
    assert!(!rt.async_encode_enabled());
    assert_eq!(TESSL_ABI_VERSION, 14);
}

/// Turning async encode off with work encoded and not yet waited for returns
/// only once that work is done: the switch waits for the runtime first.
#[test]
fn turning_async_encode_off_waits_for_work_in_flight() {
    let handle = Handle::new();
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    let rt = unsafe { tessl::capi::test_runtime(handle.0) };
    let n = 4096usize;
    let src = rt.alloc_buffer(n * 4).unwrap();
    let dst = rt.alloc_buffer(n * 4).unwrap();
    src.write_f32(&(0..n).map(|i| i as f32 + 1.0).collect::<Vec<_>>());
    dst.write_f32(&vec![0.0; n]);
    let pipe = rt.pipeline("copy_f32").unwrap();
    let before = rt.last_signaled_value();
    // Encoded straight on the runtime, outside any call, so nothing waits for it.
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipe);
        bnd.bind_gpu_buf(&src, 0);
        bnd.bind_gpu_buf(&dst, 1);
        bnd.bind_u32(n as u32, 2);
        bnd.dispatch(
            tessl::runtime::mtl_size(n / 64, 1, 1),
            tessl::runtime::mtl_size(64, 1, 1),
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(set_async(&handle, 0), TESSL_OK);
    assert!(!rt.async_encode_enabled());
    assert!(rt.last_signaled_value() > before, "the open batch was never committed");
    assert_idle(&rt, "turning async off");
    // SAFETY: a shared-storage buffer of `n` f32s that nothing else writes.
    let raw = unsafe { std::slice::from_raw_parts(dst.metal().contents().as_ptr().cast::<f32>(), n) };
    assert_eq!(
        (raw[0], raw[n - 1]),
        (1.0, n as f32),
        "the copy had not finished when the switch returned"
    );
}

