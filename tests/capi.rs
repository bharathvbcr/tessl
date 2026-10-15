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
    tessl_abi_version, tessl_cross_entropy_rows, tessl_gdn_train_backward, tessl_gdn_train_forward,
    tessl_mtl_buffer_length, tessl_qwen35_adamw_free, tessl_qwen35_adamw_init, tessl_qwen35_adamw_set_step_count,
    tessl_qwen35_adamw_step, tessl_qwen35_adamw_step_count, tessl_qwen35_copy, tessl_qwen35_describe,
    tessl_qwen35_free, tessl_qwen35_grad_sq_norm, tessl_qwen35_hidden, tessl_qwen35_load, tessl_qwen35_param_count,
    tessl_qwen35_param_info, tessl_qwen35_read_parameter_host, tessl_qwen35_train_backward, tessl_qwen35_train_discard,
    tessl_qwen35_train_forward, tessl_qwen35_train_step, tessl_qwen35_write_parameter_host, tessl_runtime_free,
    tessl_runtime_new, tessl_runtime_poisoned, tessl_runtime_set_async_encode, tessl_synchronize, TesslCeArgs,
    TesslGdnArgs, TesslParamInfo, TesslQwen35, TesslRuntime, TesslTensorRef, TESSL_ABI_VERSION, TESSL_BF16, TESSL_ERR,
    TESSL_F16, TESSL_F32, TESSL_MAX_DIMS, TESSL_MOMENTS_BLOCK8, TESSL_MOMENTS_F32, TESSL_OK, TESSL_OPERANDS_BF16,
    TESSL_OPERANDS_EXACT_F32, TESSL_READ_ADAMW_AUX, TESSL_READ_ADAMW_M, TESSL_READ_ADAMW_V, TESSL_READ_GRADS,
    TESSL_READ_PARAMS, TESSL_SUPERVISE_CAUSAL, TESSL_SUPERVISE_ROWS, TESSL_UPDATE_BF16_KAHAN,
    TESSL_UPDATE_BF16_STOCHASTIC, TESSL_UPDATE_F32, TESSL_UPDATE_F32_MASTER, TESSL_WRITE_ADAMW_AUX,
    TESSL_WRITE_ADAMW_M, TESSL_WRITE_ADAMW_V, TESSL_WRITE_PARAMS,
};
use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::gemm::GemmOperands;
use tessl::qwen35_adamw::{excluded_from_weight_decay, AdamW, AdamWHyper, Moment};
use tessl::qwen35_train::{Qwen35Grads, Supervise};
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
    let _gpu = common::gpu_lock();
    let mut err = [0 as c_char; ERR_LEN];
    let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
    assert!(!rt.is_null(), "{}", msg(&err));
    let raw = rt as usize;
    let calls = std::thread::spawn(move || runtime_entries(raw as *mut TesslRuntime))
        .join()
        .unwrap();
    assert_eq!(calls.len(), 6);
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

fn fixture_ids() -> Vec<u32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train/ids.npy");
    tessl::npy::read_npy(&p)
        .unwrap()
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&x| x as u32)
        .collect()
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

/// The whole model surface through the ABI against the Rust API on the tiny
/// training fixture: the table, the step's loss, the gradients and the
/// parameters bit for bit, and a write through the ABI moving the model.
#[test]
fn the_model_through_the_abi_is_the_rust_model() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    with_gpu(|rt| {
        let st =
            tessl::safetensors::SafeTensors::open(std::path::Path::new(fixture("model.safetensors").to_str().unwrap()))
                .unwrap();
        let cfg = tessl::qwen35_model::Qwen35Config::from_config_file(std::path::Path::new(
            fixture("config.json").to_str().unwrap(),
        ))
        .unwrap();
        let rust = tessl::qwen35_model::Qwen35Model::load(rt, &st, "model.", cfg, tessl::qwen35_model::Precision::F32)
            .unwrap();
        let table = rust.parameter_table().unwrap();

        let mut n = 0u64;
        assert_eq!(
            unsafe { tessl_qwen35_param_count(model, &mut n, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(n as usize, table.len());
        for (i, p) in table.iter().enumerate() {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { tessl_qwen35_param_info(model, i as u64, &mut info, err.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            assert_eq!(msg(&info.name), p.name);
            assert_eq!(
                &info.shape[..info.ndim as usize],
                p.shape.iter().map(|&d| d as u64).collect::<Vec<_>>().as_slice()
            );
            assert_eq!(info.transposed != 0, p.transposed);
        }
        let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { tessl_qwen35_param_info(model, n, &mut info, err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains(&format!("index {n} is outside")), "{}", msg(&err));

        // Caller buffers, one per entry, of each entry's storage shape.
        let bufs: Vec<_> = table
            .iter()
            .map(|p| shared(rt, &vec![0.0; p.shape.iter().product()]))
            .collect();
        let refs: Vec<TesslTensorRef> = table
            .iter()
            .zip(&bufs)
            .map(|(p, b)| tref(b, &p.storage_shape().iter().map(|&d| d as u64).collect::<Vec<_>>()))
            .collect();
        let copy = |dir: u32, refs: &[TesslTensorRef], err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_copy(model, dir, refs.as_ptr(), refs.len() as u64, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(copy(TESSL_READ_GRADS, &refs, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("no gradients yet"), "{}", msg(&err));
        assert_eq!(copy(TESSL_READ_PARAMS, &refs[1..], &mut err), TESSL_ERR);
        assert!(
            msg(&err).contains(&format!("{} tensors for {} parameters", n - 1, n)),
            "{}",
            msg(&err)
        );
        assert_eq!(copy(9, &refs, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("direction 9"), "{}", msg(&err));

        let mut loss = 0.0f64;
        let s = unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                &mut loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        let want = rust.train_step(&ids, GemmOperands::ExactF32).unwrap();
        assert_eq!(loss.to_bits(), want.loss.to_bits());

        let local: Vec<tessl::Tensor> = table
            .iter()
            .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
            .collect();
        for (dir, fill) in [(TESSL_READ_GRADS, true), (TESSL_READ_PARAMS, false)] {
            assert_eq!(copy(dir, &refs, &mut err), TESSL_OK, "{}", msg(&err));
            if fill {
                rust.read_gradients(&want.grads, &local).unwrap();
            } else {
                rust.read_parameters(&local).unwrap();
            }
            for ((p, b), t) in table.iter().zip(&bufs).zip(&local) {
                let got = read(b, p.shape.iter().product());
                let exp = t.read_f32().unwrap();
                assert!(
                    got.iter().zip(&exp).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} (direction {dir})",
                    p.name
                );
            }
        }

        // A bf16-operand step through the ABI is the Rust bf16 step: its loss
        // and gradients bit for bit, and not the exact step's loss.
        let mut bf_loss = 0.0f64;
        let s = unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_BF16,
                &mut bf_loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        let bf_want = rust.train_step(&ids, GemmOperands::Bf16).unwrap();
        assert_eq!(bf_loss.to_bits(), bf_want.loss.to_bits());
        assert_ne!(bf_loss.to_bits(), loss.to_bits(), "bf16 operands gave the exact loss");
        assert_eq!(copy(TESSL_READ_GRADS, &refs, &mut err), TESSL_OK, "{}", msg(&err));
        rust.read_gradients(&bf_want.grads, &local).unwrap();
        for ((p, b), t) in table.iter().zip(&bufs).zip(&local) {
            let got = read(b, p.shape.iter().product());
            let exp = t.read_f32().unwrap();
            assert!(
                got.iter().zip(&exp).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{} (bf16 gradients)",
                p.name
            );
        }
        // An unknown operands code is refused before the bank is touched
        // (4551470: validate before mutate), so the bf16 step's gradients
        // above are still readable.
        let s = unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                2,
                &mut bf_loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_ERR);
        assert!(
            msg(&err).contains("operands code 2 is neither 0 (exact f32) nor 1 (bf16)"),
            "{}",
            msg(&err)
        );
        assert_eq!(
            copy(TESSL_READ_GRADS, &refs, &mut err),
            TESSL_OK,
            "a step refused for its operands code discarded the gradients: {}",
            msg(&err)
        );
        // Put the buffers back to the parameters for the write below.
        assert_eq!(copy(TESSL_READ_PARAMS, &refs, &mut err), TESSL_OK, "{}", msg(&err));

        // Halve every parameter through the ABI: the next loss moves, and is
        // the Rust model's after the same write.
        for (p, b) in table.iter().zip(&bufs) {
            let n = p.shape.iter().product();
            let half: Vec<f32> = read(b, n).iter().map(|x| x * 0.5).collect();
            unsafe { ptr::copy_nonoverlapping(half.as_ptr(), b.contents().as_ptr().cast::<f32>(), n) };
        }
        assert_eq!(copy(TESSL_WRITE_PARAMS, &refs, &mut err), TESSL_OK, "{}", msg(&err));
        for (t, b) in local.iter().zip(&bufs) {
            t.write_f32(&read(b, t.numel())).unwrap();
        }
        rust.write_parameters(&local).unwrap();
        let mut moved = 0.0f64;
        let s = unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                &mut moved,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        assert_ne!(moved.to_bits(), loss.to_bits());
        assert_eq!(
            moved.to_bits(),
            rust.train_step(&ids, GemmOperands::ExactF32).unwrap().loss.to_bits()
        );
    });

    // Refusals: a bad id, a null handle, another thread.
    let bad = [1u32, 64];
    let mut loss = 0.0;
    let s = unsafe {
        tessl_qwen35_train_step(
            model,
            bad.as_ptr(),
            2,
            TESSL_OPERANDS_EXACT_F32,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("token id 64 >= vocab 64"), "{}", msg(&err));
    let mut n = 0u64;
    assert_eq!(
        unsafe { tessl_qwen35_param_count(ptr::null_mut(), &mut n, err.as_mut_ptr(), ERR_LEN) },
        TESSL_ERR
    );
    assert_eq!(msg(&err), "null model handle");
    let raw = model as usize;
    let (s, m, freed) = std::thread::spawn(move || {
        let mut err = [0 as c_char; ERR_LEN];
        let mut n = 0u64;
        let s = unsafe { tessl_qwen35_param_count(raw as *mut TesslQwen35, &mut n, err.as_mut_ptr(), ERR_LEN) };
        (s, msg(&err), unsafe { tessl_qwen35_free(raw as *mut TesslQwen35) })
    })
    .join()
    .unwrap();
    assert_eq!((s, freed), (TESSL_ERR, TESSL_ERR));
    assert!(
        m.contains("tessl model used from a thread other than the one that created it"),
        "{m}"
    );
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
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

/// AdamW through the ABI is the Rust AdamW: after two steps on the same
/// gradients every parameter has the same bits; the default decay flags are
/// the Rust rule; the step count follows; every refusal is a status and a
/// message, and a refused step moves nothing.
#[test]
fn adamw_through_the_abi_is_the_rust_adamw() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    with_gpu(|rt| {
        let st =
            tessl::safetensors::SafeTensors::open(std::path::Path::new(fixture("model.safetensors").to_str().unwrap()))
                .unwrap();
        let cfg = tessl::qwen35_model::Qwen35Config::from_config_file(std::path::Path::new(
            fixture("config.json").to_str().unwrap(),
        ))
        .unwrap();
        let rust = tessl::qwen35_model::Qwen35Model::load(rt, &st, "model.", cfg, tessl::qwen35_model::Precision::F32)
            .unwrap();
        let table = rust.parameter_table().unwrap();
        let n = table.len() as u64;

        // The decay flags are the Rust rule, and they build the default decays.
        let mut wd = Vec::new();
        for (i, p) in table.iter().enumerate() {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { tessl_qwen35_param_info(model, i as u64, &mut info, err.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            assert_eq!(
                info.decay_excluded != 0,
                excluded_from_weight_decay(&p.name),
                "{}",
                p.name
            );
            wd.push(if info.decay_excluded != 0 { 0.0f32 } else { 0.1 });
        }
        assert_eq!(wd, rust.default_weight_decay(0.1).unwrap());

        // Refusals before there is anything to step.
        let step = |w: *const f32, n: u64, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_adamw_step(model, 1e-2, 0.9, 0.999, 1e-8, 1.0, w, n, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(step(wd.as_ptr(), n, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("no gradients yet"), "{}", msg(&err));
        let mut count = 0u64;
        assert_eq!(
            unsafe { tessl_qwen35_adamw_step_count(model, &mut count, err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains("no AdamW state"), "{}", msg(&err));
        let mut loss = 0.0f64;
        let train = |loss: &mut f64, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(train(&mut loss, &mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(step(wd.as_ptr(), n, &mut err), TESSL_ERR);
        assert!(
            msg(&err).contains("no AdamW state; call tessl_qwen35_adamw_init first"),
            "{}",
            msg(&err)
        );
        assert_eq!(
            unsafe {
                tessl_qwen35_adamw_init(model, TESSL_UPDATE_F32, 0, TESSL_MOMENTS_F32, err.as_mut_ptr(), ERR_LEN)
            },
            TESSL_OK,
            "{}",
            msg(&err)
        );
        assert_eq!(
            unsafe {
                tessl_qwen35_adamw_init(model, TESSL_UPDATE_F32, 0, TESSL_MOMENTS_F32, err.as_mut_ptr(), ERR_LEN)
            },
            TESSL_ERR
        );
        assert!(msg(&err).contains("already has AdamW state"), "{}", msg(&err));
        assert_eq!(step(wd.as_ptr(), n - 1, &mut err), TESSL_ERR);
        assert!(
            msg(&err).contains(&format!("{} weight decays for {n} parameters", n - 1)),
            "{}",
            msg(&err)
        );
        assert_eq!(step(ptr::null(), n, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("null weight_decay"), "{}", msg(&err));
        let bad_beta = unsafe {
            tessl_qwen35_adamw_step(
                model,
                1e-2,
                1.0,
                0.999,
                1e-8,
                1.0,
                wd.as_ptr(),
                n,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(bad_beta, TESSL_ERR);
        assert!(msg(&err).contains("beta1 1 must lie in [0, 1)"), "{}", msg(&err));
        assert_eq!(
            unsafe { tessl_qwen35_adamw_step_count(model, &mut count, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(count, 0, "a refused step advanced the count");

        // Two steps each side, on the same gradients, the second with its
        // gradients scaled by 0.3 (a clip coefficient, inexact in binary): the same
        // gradient norm and the same bits.
        let mut state = AdamW::new(&rust).unwrap();
        let mut hyper = AdamWHyper {
            lr: 1e-2,
            ..AdamWHyper::default()
        };
        let mut sq = 0.0f64;
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains("null out"), "{}", msg(&err));
        for k in 1..=2u64 {
            if k > 1 {
                assert_eq!(train(&mut loss, &mut err), TESSL_OK, "{}", msg(&err));
            }
            let s = rust.train_step(&ids, GemmOperands::ExactF32).unwrap();
            assert_eq!(
                unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
                TESSL_OK,
                "{}",
                msg(&err)
            );
            assert_eq!(sq.to_bits(), rust.grad_sq_norm(&s.grads).unwrap().to_bits());
            if k == 1 {
                assert_eq!(step(wd.as_ptr(), n, &mut err), TESSL_OK, "{}", msg(&err));
            } else {
                hyper.grad_scale = 0.3;
                let status = unsafe {
                    tessl_qwen35_adamw_step(
                        model,
                        1e-2,
                        0.9,
                        0.999,
                        1e-8,
                        hyper.grad_scale,
                        wd.as_ptr(),
                        n,
                        err.as_mut_ptr(),
                        ERR_LEN,
                    )
                };
                assert_eq!(status, TESSL_OK, "{}", msg(&err));
            }
            rust.adamw_step(&s.grads, &mut state, &hyper, &wd).unwrap();
            assert_eq!(
                unsafe { tessl_qwen35_adamw_step_count(model, &mut count, err.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            assert_eq!(count, k);
        }
        let bufs: Vec<_> = table
            .iter()
            .map(|p| shared(rt, &vec![0.0; p.shape.iter().product()]))
            .collect();
        let refs: Vec<TesslTensorRef> = table
            .iter()
            .zip(&bufs)
            .map(|(p, b)| tref(b, &p.storage_shape().iter().map(|&d| d as u64).collect::<Vec<_>>()))
            .collect();
        assert_eq!(
            unsafe { tessl_qwen35_copy(model, TESSL_READ_PARAMS, refs.as_ptr(), n, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        let local: Vec<tessl::Tensor> = table
            .iter()
            .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
            .collect();
        rust.read_parameters(&local).unwrap();
        for ((p, b), t) in table.iter().zip(&bufs).zip(&local) {
            let got = read(b, p.shape.iter().product());
            let want = t.read_f32().unwrap();
            assert!(
                got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{}",
                p.name
            );
        }

        // The moments read through the ABI are the Rust state's, bit for bit.
        // Each write direction sets its own moment and no other: m written
        // from the parameters and v from the gradients read back as those
        // bits. The step count is set, not counted.
        let copy = |dir: u32, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_copy(model, dir, refs.as_ptr(), n, err.as_mut_ptr(), ERR_LEN)
        };
        let moment_bits = || -> Vec<Vec<u32>> {
            table
                .iter()
                .zip(&bufs)
                .map(|(p, b)| read(b, p.shape.iter().product()).iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        let mut moments = Vec::new();
        for (dir, which) in [
            (TESSL_READ_ADAMW_M, Moment::First),
            (TESSL_READ_ADAMW_V, Moment::Second),
        ] {
            assert_eq!(copy(dir, &mut err), TESSL_OK, "{}", msg(&err));
            rust.read_adamw_moment(&state, which, &local).unwrap();
            for ((p, b), t) in table.iter().zip(&bufs).zip(&local) {
                let got = read(b, p.shape.iter().product());
                let want = t.read_f32().unwrap();
                assert!(
                    got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} (direction {dir})",
                    p.name
                );
            }
            moments.push(moment_bits());
        }
        assert_ne!(moments[0], moments[1]);
        let read_bits = |dir: u32, err: &mut [c_char; ERR_LEN]| {
            assert_eq!(copy(dir, err), TESSL_OK, "{}", msg(err));
            moment_bits()
        };
        let params = read_bits(TESSL_READ_PARAMS, &mut err);
        assert_eq!(copy(TESSL_WRITE_ADAMW_M, &mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(
            read_bits(TESSL_READ_ADAMW_M, &mut err),
            params,
            "m did not take the written values"
        );
        assert_eq!(read_bits(TESSL_READ_ADAMW_V, &mut err), moments[1], "writing m moved v");
        let grads = read_bits(TESSL_READ_GRADS, &mut err);
        assert_ne!(grads, params);
        assert_eq!(copy(TESSL_WRITE_ADAMW_V, &mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(
            read_bits(TESSL_READ_ADAMW_V, &mut err),
            grads,
            "v did not take the written values"
        );
        assert_eq!(read_bits(TESSL_READ_ADAMW_M, &mut err), params, "writing v moved m");
        assert_eq!(
            unsafe { tessl_qwen35_adamw_set_step_count(model, 7, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK,
            "{}",
            msg(&err)
        );
        assert_eq!(
            unsafe { tessl_qwen35_adamw_step_count(model, &mut count, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(count, 7);

        // Freed state is gone: a step, a moment copy and a set count are
        // refused again.
        assert_eq!(
            unsafe { tessl_qwen35_adamw_free(model, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(step(wd.as_ptr(), n, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("no AdamW state"), "{}", msg(&err));
        for dir in [
            TESSL_READ_ADAMW_M,
            TESSL_READ_ADAMW_V,
            TESSL_WRITE_ADAMW_M,
            TESSL_WRITE_ADAMW_V,
        ] {
            assert_eq!(copy(dir, &mut err), TESSL_ERR);
            assert!(
                msg(&err).contains("no AdamW state; call tessl_qwen35_adamw_init first"),
                "{}",
                msg(&err)
            );
        }
        assert_eq!(
            unsafe { tessl_qwen35_adamw_set_step_count(model, 1, err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains("no AdamW state"), "{}", msg(&err));
    });
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// Every gradient read through the ABI, as bits, one Vec per table entry.
fn abi_grads(model: *mut TesslQwen35, rt: &GpuRuntime, table: &[tessl::qwen35_params::ParamInfo]) -> Vec<Vec<u32>> {
    let bufs: Vec<_> = table
        .iter()
        .map(|p| shared(rt, &vec![0.0; p.shape.iter().product()]))
        .collect();
    let refs: Vec<TesslTensorRef> = table
        .iter()
        .zip(&bufs)
        .map(|(p, b)| tref(b, &p.storage_shape().iter().map(|&d| d as u64).collect::<Vec<_>>()))
        .collect();
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe {
        tessl_qwen35_copy(
            model,
            TESSL_READ_GRADS,
            refs.as_ptr(),
            refs.len() as u64,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    table
        .iter()
        .zip(&bufs)
        .map(|(p, b)| read(b, p.shape.iter().product()).iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// A bank's gradients through the Rust API, as bits, laid out as `abi_grads`.
fn rust_grads(
    rust: &tessl::qwen35_model::Qwen35Model,
    bank: &Qwen35Grads,
    rt: &Arc<GpuRuntime>,
    table: &[tessl::qwen35_params::ParamInfo],
) -> Vec<Vec<u32>> {
    let local: Vec<tessl::Tensor> = table
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    rust.read_gradients(bank, &local).unwrap();
    local
        .iter()
        .map(|t| t.read_f32().unwrap().iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// The two-phase step through the ABI is the Rust one: a rows forward's
/// loss, the hidden rows, and a backward with an outside gradient, then a
/// causal step accumulated onto it, each bit for bit. While a step is
/// pending, another step, the gradients and AdamW are refused; a refused
/// backward keeps the step pending; a discard drops it; accumulating onto a
/// bank a refused step marked is refused.
#[test]
fn a_two_phase_step_through_the_abi_is_the_rust_one() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    with_gpu(|rt| {
        let st =
            tessl::safetensors::SafeTensors::open(std::path::Path::new(fixture("model.safetensors").to_str().unwrap()))
                .unwrap();
        let cfg = tessl::qwen35_model::Qwen35Config::from_config_file(std::path::Path::new(
            fixture("config.json").to_str().unwrap(),
        ))
        .unwrap();
        let (h, vocab) = (cfg.hidden as usize, cfg.vocab);
        let rust = tessl::qwen35_model::Qwen35Model::load(rt, &st, "model.", cfg, tessl::qwen35_model::Precision::F32)
            .unwrap();
        let table = rust.parameter_table().unwrap();
        let mm = GemmOperands::ExactF32;
        let (positions, targets) = ([1u32, 4], [ids[2], ids[5]]);
        let forward = |sup: u32, pos: &[u32], tg: &[u32], loss: &mut f64, err: &mut [c_char; ERR_LEN]| unsafe {
            let (pp, tp) = if pos.is_empty() {
                (ptr::null(), ptr::null())
            } else {
                (pos.as_ptr(), tg.as_ptr())
            };
            tessl_qwen35_train_forward(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                sup,
                pp,
                tp,
                pos.len() as u64,
                0.5,
                loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };

        // Nothing pending yet.
        let out_buf = shared(rt, &vec![0.0; 2 * h]);
        let out_ref = tref(&out_buf, &[2, h as u64]);
        let hidden = |pos: &[u32], r: &TesslTensorRef, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_hidden(model, pos.as_ptr(), pos.len() as u64, r, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(hidden(&[2, 0], &out_ref, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("no step is pending"), "{}", msg(&err));

        let mut loss = 0.0f64;
        assert_eq!(
            forward(TESSL_SUPERVISE_ROWS, &positions, &targets, &mut loss, &mut err),
            TESSL_OK,
            "{}",
            msg(&err)
        );
        let sup = Supervise::Rows {
            positions: &positions,
            targets: &targets,
            scale: 0.5,
        };
        let p = rust.train_forward(&ids, mm, sup).unwrap();
        assert_eq!(loss.to_bits(), p.loss().to_bits());

        // The hidden rows are the Rust step's.
        assert_eq!(hidden(&[2, 0], &out_ref, &mut err), TESSL_OK, "{}", msg(&err));
        let want = rt.alloc_tensor_f32(&[2, h]).unwrap();
        p.hidden(&[2, 0], &want).unwrap();
        let bits = |v: Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(read(&out_buf, 2 * h)), bits(want.read_f32().unwrap()));

        // While pending: no second step, no gradients, no AdamW.
        let mut other = 0.0f64;
        assert_eq!(
            forward(TESSL_SUPERVISE_CAUSAL, &[], &[], &mut other, &mut err),
            TESSL_ERR
        );
        assert!(msg(&err).contains("a step is pending"), "{}", msg(&err));
        let train = |loss: &mut f64, ids: &[u32], err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(train(&mut other, &ids, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("a step is pending"), "{}", msg(&err));
        let mut sq = 0.0f64;
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains("a step is pending"), "{}", msg(&err));
        // Nor a parameter write: the backward rebuilds each layer with the
        // weights it finds.
        let zeros: Vec<_> = table
            .iter()
            .map(|p| shared(rt, &vec![0.0; p.shape.iter().product()]))
            .collect();
        let zero_refs: Vec<TesslTensorRef> = table
            .iter()
            .zip(&zeros)
            .map(|(p, b)| tref(b, &p.storage_shape().iter().map(|&d| d as u64).collect::<Vec<_>>()))
            .collect();
        let s = unsafe {
            tessl_qwen35_copy(
                model,
                TESSL_WRITE_PARAMS,
                zero_refs.as_ptr(),
                zero_refs.len() as u64,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_ERR);
        assert!(msg(&err).contains("a step is pending"), "{}", msg(&err));

        // An outside gradient at two positions; a refused one keeps the step.
        let dh_v: Vec<f32> = (0..2 * h).map(|i| ((i % 7) as f32 - 3.0) * 1e-2).collect();
        let dh_buf = shared(rt, &dh_v);
        let dh_ref = tref(&dh_buf, &[2, h as u64]);
        let backward = |pos: &[u32], r: *const TesslTensorRef, acc: u32, err: &mut [c_char; ERR_LEN]| unsafe {
            let pp = if r.is_null() { ptr::null() } else { pos.as_ptr() };
            tessl_qwen35_train_backward(model, pp, pos.len() as u64, r, acc, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(backward(&[3, 3], &dh_ref, 0, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("position 3 appears twice"), "{}", msg(&err));
        assert_eq!(backward(&[3, 0], &dh_ref, 0, &mut err), TESSL_OK, "{}", msg(&err));
        let dh_t = rt.alloc_tensor_f32(&[2, h]).unwrap();
        dh_t.buffer.write_f32(&dh_v);
        let bank = Qwen35Grads::zeros_like(&rust).unwrap();
        rust.train_backward_into(p, Some((&[3, 0], &dh_t)), &bank, false)
            .unwrap();
        assert_eq!(abi_grads(model, rt, &table), rust_grads(&rust, &bank, rt, &table));

        // A causal step accumulated onto it.
        assert_eq!(
            forward(TESSL_SUPERVISE_CAUSAL, &[], &[], &mut loss, &mut err),
            TESSL_OK,
            "{}",
            msg(&err)
        );
        assert_eq!(backward(&[], ptr::null(), 1, &mut err), TESSL_OK, "{}", msg(&err));
        let p = rust.train_forward(&ids, mm, Supervise::Causal).unwrap();
        assert_eq!(loss.to_bits(), p.loss().to_bits());
        rust.train_backward_into(p, None, &bank, true).unwrap();
        assert_eq!(abi_grads(model, rt, &table), rust_grads(&rust, &bank, rt, &table));

        // Refusals of the forward's arguments.
        assert_eq!(forward(7, &[], &[], &mut loss, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("supervise 7 is not"), "{}", msg(&err));
        assert_eq!(
            forward(TESSL_SUPERVISE_CAUSAL, &positions, &targets, &mut loss, &mut err),
            TESSL_ERR
        );
        assert!(msg(&err).contains("takes no positions"), "{}", msg(&err));

        // A discard drops a pending step.
        assert_eq!(forward(TESSL_SUPERVISE_CAUSAL, &[], &[], &mut loss, &mut err), TESSL_OK);
        assert_eq!(
            unsafe { tessl_qwen35_train_discard(model, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(hidden(&[0], &tref(&out_buf, &[1, h as u64]), &mut err), TESSL_ERR);
        assert!(msg(&err).contains("no step is pending"), "{}", msg(&err));

        // A refused step marks the bank: no gradients, and no accumulating onto it.
        assert_eq!(train(&mut loss, &[5, vocab], &mut err), TESSL_ERR);
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
            TESSL_ERR
        );
        assert!(msg(&err).contains("no gradients yet"), "{}", msg(&err));
        assert_eq!(forward(TESSL_SUPERVISE_CAUSAL, &[], &[], &mut loss, &mut err), TESSL_OK);
        assert_eq!(backward(&[], ptr::null(), 1, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("start again with accumulate = 0"), "{}", msg(&err));
        assert_eq!(backward(&[], ptr::null(), 0, &mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );

        // A step refused for its operands code leaves the bank as it was
        // (Ready from the backward above), unlike one that starts and fails.
        let bad_operands = unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                9,
                &mut loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(bad_operands, TESSL_ERR);
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK,
            "{}",
            msg(&err)
        );
        // A step refused because one is pending leaves them as they were.
        assert_eq!(train(&mut loss, &ids, &mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(forward(TESSL_SUPERVISE_CAUSAL, &[], &[], &mut loss, &mut err), TESSL_OK);
        assert_eq!(train(&mut loss, &ids, &mut err), TESSL_ERR);
        assert_eq!(
            unsafe { tessl_qwen35_train_discard(model, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        assert_eq!(
            unsafe { tessl_qwen35_grad_sq_norm(model, &mut sq, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK,
            "{}",
            msg(&err)
        );
    });
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// A bf16 model through the ABI is the Rust bf16 model (`load_tower`,
/// `AdamW::with_config`): two steps into the bank and two Kahan / 8-bit
/// AdamW steps leave the parameters, both moments and the compensations at
/// the Rust bits; the auxiliary state written back reads as the same bits;
/// the description names the stored precision and the optimizer; every
/// refusal is a status and a message.
#[test]
fn a_bf16_model_through_the_abi_is_the_rust_bf16_model() {
    use tessl::qwen35_adamw::{AdamWConfig, MomentStorage, UpdateRule};
    let handle = Handle::new();
    let model = load_model_as(handle.0, TESSL_BF16);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    with_gpu(|rt| {
        let st =
            tessl::safetensors::SafeTensors::open(std::path::Path::new(fixture("model.safetensors").to_str().unwrap()))
                .unwrap();
        let cfg = tessl::qwen35_model::Qwen35Config::from_config_file(std::path::Path::new(
            fixture("config.json").to_str().unwrap(),
        ))
        .unwrap();
        let rust =
            tessl::qwen35_model::Qwen35Model::load_tower(rt, &st, "model.", cfg, tessl::qwen35_model::Precision::Bf16)
                .unwrap();
        let table = rust.parameter_table().unwrap();
        let n = table.len() as u64;

        let describe = |len: usize, err: &mut [c_char; ERR_LEN]| -> (i32, String, u64) {
            let mut out = vec![0 as c_char; len.max(1)];
            let mut needed = 0u64;
            let s = unsafe {
                tessl_qwen35_describe(
                    model,
                    out.as_mut_ptr(),
                    len as u64,
                    &mut needed,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            (s, msg(&out), needed)
        };
        let (s, line, needed) = describe(1024, &mut err);
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        assert_eq!(line, rust.describe());
        assert!(line.contains("matrices bf16"), "{line}");
        assert_eq!(needed as usize, line.len() + 1);
        let (s, _, short) = describe(8, &mut err);
        assert_eq!((s, short), (TESSL_ERR, needed));
        assert!(
            msg(&err).contains(&format!("needs {needed} bytes, out has 8")),
            "{}",
            msg(&err)
        );

        let init = |update: u32, seed: u64, moments: u32, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_adamw_init(model, update, seed, moments, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(init(TESSL_UPDATE_F32, 0, TESSL_MOMENTS_F32, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("does not apply to a Bf16 model"), "{}", msg(&err));
        assert_eq!(init(9, 0, TESSL_MOMENTS_F32, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("update code 9 is not"), "{}", msg(&err));
        assert_eq!(init(TESSL_UPDATE_BF16_KAHAN, 0, 7, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("moments code 7 is not"), "{}", msg(&err));
        assert_eq!(
            init(TESSL_UPDATE_BF16_KAHAN, 5, TESSL_MOMENTS_BLOCK8, &mut err),
            TESSL_ERR
        );
        assert!(
            msg(&err).contains("seed 5 given to a rule that takes none"),
            "{}",
            msg(&err)
        );
        assert_eq!(
            init(TESSL_UPDATE_BF16_KAHAN, 0, TESSL_MOMENTS_BLOCK8, &mut err),
            TESSL_OK,
            "{}",
            msg(&err)
        );
        let config = AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        };
        let mut state = AdamW::with_config(&rust, config).unwrap();
        let (_, line, _) = describe(1024, &mut err);
        assert_eq!(line, format!("{}; {}", rust.describe(), state.describe()));
        assert!(
            line.ends_with("AdamW update=bf16-kahan moments=block8/256 step=0"),
            "{line}"
        );

        let train = |operands: u32, loss: &mut f64, err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_train_step(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                operands,
                loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        let mut loss = 0.0f64;
        assert_eq!(train(TESSL_OPERANDS_EXACT_F32, &mut loss, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("trains on GemmOperands::Bf16"), "{}", msg(&err));
        let bank = Qwen35Grads::zeros_like(&rust).unwrap();
        let wd = rust.default_weight_decay(0.1).unwrap();
        let hyper = AdamWHyper {
            lr: 1e-2,
            ..AdamWHyper::default()
        };
        for _ in 0..2 {
            assert_eq!(
                train(TESSL_OPERANDS_BF16, &mut loss, &mut err),
                TESSL_OK,
                "{}",
                msg(&err)
            );
            let want = rust
                .train_step_into(&ids, GemmOperands::Bf16, Supervise::Causal, &bank, false)
                .unwrap();
            assert_eq!(loss.to_bits(), want.to_bits());
            let s = unsafe {
                tessl_qwen35_adamw_step(
                    model,
                    hyper.lr,
                    hyper.beta1,
                    hyper.beta2,
                    hyper.eps,
                    hyper.grad_scale,
                    wd.as_ptr(),
                    n,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            assert_eq!(s, TESSL_OK, "{}", msg(&err));
            rust.adamw_step(&bank, &mut state, &hyper, &wd).unwrap();
        }
        let (_, line, _) = describe(1024, &mut err);
        assert!(line.ends_with("step=2"), "{line}");

        // Parameters, moments and compensations: the Rust state's bits.
        let bufs: Vec<_> = table
            .iter()
            .map(|p| shared(rt, &vec![0.0; p.shape.iter().product()]))
            .collect();
        let refs: Vec<TesslTensorRef> = table
            .iter()
            .zip(&bufs)
            .map(|(p, b)| tref(b, &p.storage_shape().iter().map(|&d| d as u64).collect::<Vec<_>>()))
            .collect();
        let local: Vec<tessl::Tensor> = table
            .iter()
            .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
            .collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let abi = |dir: u32, err: &mut [c_char; ERR_LEN]| -> Vec<Vec<u32>> {
            let s = unsafe { tessl_qwen35_copy(model, dir, refs.as_ptr(), n, err.as_mut_ptr(), ERR_LEN) };
            assert_eq!(s, TESSL_OK, "direction {dir}: {}", msg(err));
            table
                .iter()
                .zip(&bufs)
                .map(|(p, b)| bits(&read(b, p.shape.iter().product())))
                .collect()
        };
        let mine =
            |local: &[tessl::Tensor]| -> Vec<Vec<u32>> { local.iter().map(|t| bits(&t.read_f32().unwrap())).collect() };
        rust.read_parameters(&local).unwrap();
        assert_eq!(abi(TESSL_READ_PARAMS, &mut err), mine(&local), "parameters");
        for (dir, which) in [
            (TESSL_READ_ADAMW_M, Moment::First),
            (TESSL_READ_ADAMW_V, Moment::Second),
        ] {
            rust.read_adamw_moment(&state, which, &local).unwrap();
            assert_eq!(abi(dir, &mut err), mine(&local), "{which:?}");
        }
        rust.read_adamw_aux(&state, &local).unwrap();
        let aux = abi(TESSL_READ_ADAMW_AUX, &mut err);
        assert_eq!(aux, mine(&local), "compensations");
        assert!(aux.iter().flatten().any(|&b| b != 0), "two steps left no compensation");

        // Written back, the compensations read as the same bits; zeros
        // written read as zeros.
        let s = unsafe {
            tessl_qwen35_copy(
                model,
                TESSL_WRITE_ADAMW_AUX,
                refs.as_ptr(),
                n,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        assert_eq!(abi(TESSL_READ_ADAMW_AUX, &mut err), aux);
        for (p, b) in table.iter().zip(&bufs) {
            let z = vec![0.0f32; p.shape.iter().product()];
            unsafe { ptr::copy_nonoverlapping(z.as_ptr(), b.contents().as_ptr().cast::<f32>(), z.len()) };
        }
        let s = unsafe {
            tessl_qwen35_copy(
                model,
                TESSL_WRITE_ADAMW_AUX,
                refs.as_ptr(),
                n,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        assert!(abi(TESSL_READ_ADAMW_AUX, &mut err).iter().flatten().all(|&b| b == 0));
    });
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);

    // An f32 model's AdamW keeps no auxiliary state.
    let f32_model = load_model(handle.0);
    let mut err = [0 as c_char; ERR_LEN];
    assert_eq!(
        unsafe {
            tessl_qwen35_adamw_init(
                f32_model,
                TESSL_UPDATE_F32,
                0,
                TESSL_MOMENTS_F32,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        },
        TESSL_OK,
        "{}",
        msg(&err)
    );
    with_gpu(|rt| {
        let mut n = 0u64;
        assert_eq!(
            unsafe { tessl_qwen35_param_count(f32_model, &mut n, err.as_mut_ptr(), ERR_LEN) },
            TESSL_OK
        );
        let mut infos = Vec::new();
        for i in 0..n {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { tessl_qwen35_param_info(f32_model, i, &mut info, err.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            infos.push(info);
        }
        let bufs: Vec<_> = infos
            .iter()
            .map(|i| {
                shared(
                    rt,
                    &vec![0.0; i.shape[..i.ndim as usize].iter().product::<u64>() as usize],
                )
            })
            .collect();
        let refs: Vec<TesslTensorRef> = infos
            .iter()
            .zip(&bufs)
            .map(|(i, b)| {
                let mut shape = i.shape[..i.ndim as usize].to_vec();
                if i.transposed != 0 {
                    shape.reverse();
                }
                tref(b, &shape)
            })
            .collect();
        let s = unsafe {
            tessl_qwen35_copy(
                f32_model,
                TESSL_READ_ADAMW_AUX,
                refs.as_ptr(),
                n,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_ERR);
        assert!(
            msg(&err).contains("update rule f32 keeps no auxiliary state"),
            "{}",
            msg(&err)
        );
    });
    assert_eq!(unsafe { tessl_qwen35_free(f32_model) }, TESSL_OK);
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
        let builds = || unsafe { tessl::capi::test_workspace_builds(handle.0) }.0;

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
        let (s, m) = copy(TESSL_READ_ADAMW_M);
        assert_eq!(s, TESSL_ERR);
        assert!(m.contains("no AdamW state"), "{m}");
        let (s, m) = copy(TESSL_READ_GRADS);
        assert_eq!(s, TESSL_ERR);
        assert!(m.contains("no gradients yet"), "{m}");
        assert_eq!(wraps(), before, "a refused copy wrapped tensors");
        // The check is not vacuous: a valid direction wraps all n.
        let _ = copy(TESSL_READ_PARAMS);
        assert_eq!(wraps(), before + n as u64);
    });
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// A step refused for its arguments leaves the bank as it was: the previous
/// step's gradients are still readable. (A step that starts and fails leaves
/// none; see the function's doc.)
#[test]
fn a_refused_train_step_keeps_the_previous_gradients() {
    let handle = Handle::new();
    let model = load_model(handle.0);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    let mut loss = 0.0f64;
    let step = |operands: u32, loss: &mut f64, err: &mut [c_char; ERR_LEN]| unsafe {
        tessl_qwen35_train_step(
            model,
            ids.as_ptr(),
            ids.len() as u64,
            operands,
            loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(
        step(TESSL_OPERANDS_EXACT_F32, &mut loss, &mut err),
        TESSL_OK,
        "{}",
        msg(&err)
    );
    with_gpu(|rt| {
        let n = param_count(model);
        let mut infos = Vec::new();
        for i in 0..n {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            let mut e = [0 as c_char; ERR_LEN];
            assert_eq!(
                unsafe { tessl_qwen35_param_info(model, i as u64, &mut info, e.as_mut_ptr(), ERR_LEN) },
                TESSL_OK
            );
            infos.push(info);
        }
        let bufs: Vec<_> = infos
            .iter()
            .map(|p| {
                shared(
                    rt,
                    &vec![0.0; p.shape[..p.ndim as usize].iter().product::<u64>() as usize],
                )
            })
            .collect();
        let refs: Vec<_> = bufs
            .iter()
            .zip(&infos)
            .map(|(b, p)| {
                // A copy reads into the storage shape: transformers' shape,
                // reversed for a `transposed` entry (see TesslParamInfo).
                let mut shape = p.shape[..p.ndim as usize].to_vec();
                if p.transposed != 0 {
                    shape.reverse();
                }
                tref(b, &shape)
            })
            .collect();
        let read = |err: &mut [c_char; ERR_LEN]| unsafe {
            tessl_qwen35_copy(
                model,
                TESSL_READ_GRADS,
                refs.as_ptr(),
                n as u64,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(read(&mut err), TESSL_OK, "{}", msg(&err));
        assert_eq!(step(99, &mut loss, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("operands"), "{}", msg(&err));
        assert_eq!(
            read(&mut err),
            TESSL_OK,
            "a step refused for its operands code discarded the gradients: {}",
            msg(&err)
        );
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
    let ids = fixture_ids();
    let mut loss = f64::NAN;
    let s = unsafe {
        tessl_qwen35_train_step(
            model,
            ids.as_ptr(),
            ids.len() as u64,
            TESSL_OPERANDS_EXACT_F32,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert!(loss.is_finite(), "loss {loss}");
    assert_eq!(poisoned(b.0), 0);
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}

/// A GDN backward refused for its tensors keeps the cached workspace.
#[test]
fn a_refused_gdn_backward_keeps_the_cached_workspace() {
    use tessl::gdn_train::GdnTrainDims;
    let handle = Handle::new();
    with_gpu(|rt| {
        let dims = GdnTrainDims {
            batch: 1,
            seq: 16,
            heads: 2,
            v_dim: 128,
        };
        let (b, t, h, dv) = (1usize, 16usize, 2usize, 128usize);
        let buf = |n: usize, seed: u64| shared(rt, &random_f32(n, seed));
        let (q, k) = (buf(b * t * h * 128, 1), buf(b * t * h * 128, 2));
        let (v, g, beta) = (buf(b * t * h * dv, 3), buf(b * t * h, 4), buf(b * t * h, 5));
        let ck = dims.checkpoint_shape();
        let ckpt = buf(ck.iter().product(), 6);
        let (o, d_o) = (buf(b * t * h * dv, 7), buf(b * t * h * dv, 8));
        let (dq, dk, dvv) = (
            buf(b * t * h * 128, 9),
            buf(b * t * h * 128, 10),
            buf(b * t * h * dv, 11),
        );
        let (dg, db) = (buf(b * t * h, 12), buf(b * t * h, 13));
        let (bu, tu, hu, du) = (b as u64, t as u64, h as u64, dv as u64);
        let ck64: Vec<u64> = ck.iter().map(|&d| d as u64).collect();
        let args = TesslGdnArgs {
            batch: 1,
            seq: 16,
            heads: 2,
            v_dim: 128,
            q: tref(&q, &[bu, tu, hu, 128]),
            k: tref(&k, &[bu, tu, hu, 128]),
            v: tref(&v, &[bu, tu, hu, du]),
            g: tref(&g, &[bu, tu, hu]),
            beta: tref(&beta, &[bu, tu, hu]),
            s0: NULL_REF,
            ckpt: tref(&ckpt, &ck64),
            o: tref(&o, &[bu, tu, hu, du]),
            s_fin: NULL_REF,
            d_o: tref(&d_o, &[bu, tu, hu, du]),
            d_fin: NULL_REF,
            dq: tref(&dq, &[bu, tu, hu, 128]),
            dk: tref(&dk, &[bu, tu, hu, 128]),
            dv: tref(&dvv, &[bu, tu, hu, du]),
            dg: tref(&dg, &[bu, tu, hu]),
            dbeta: tref(&db, &[bu, tu, hu]),
            ds0: NULL_REF,
        };
        let run = |f: unsafe extern "C" fn(*mut TesslRuntime, *const TesslGdnArgs, *mut c_char, usize) -> i32,
                   a: &TesslGdnArgs| {
            let mut err = [0 as c_char; ERR_LEN];
            let s = unsafe { f(handle.0, a, err.as_mut_ptr(), ERR_LEN) };
            (s, msg(&err))
        };
        let builds = || unsafe { tessl::capi::test_workspace_builds(handle.0) }.1;
        let (s, m) = run(tessl_gdn_train_forward, &args);
        assert_eq!(s, TESSL_OK, "{m}");
        let (s, m) = run(tessl_gdn_train_backward, &args);
        assert_eq!(s, TESSL_OK, "{m}");
        assert_eq!(builds(), 1);

        // Other dims over the same tensors: refused, and the cached workspace
        // (for the dims above) is neither dropped nor replaced.
        let (s, m) = run(tessl_gdn_train_backward, &TesslGdnArgs { seq: 32, ..args });
        assert_eq!(s, TESSL_ERR, "{m}");
        assert_eq!(builds(), 1, "a refused call must not build a workspace: {m}");
        let (s, m) = run(tessl_gdn_train_backward, &args);
        assert_eq!(s, TESSL_OK, "{m}");
        assert_eq!(builds(), 1, "the next good call reuses the cached workspace");
    });
}

type GdnFn = unsafe extern "C" fn(*mut TesslRuntime, *const TesslGdnArgs, *mut c_char, usize) -> i32;

fn run_gdn(f: GdnFn, handle: &Handle, a: &TesslGdnArgs) -> (i32, String) {
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe { f(handle.0, a, err.as_mut_ptr(), ERR_LEN) };
    (s, msg(&err))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every GDN output through the C ABI is the Rust call's, bit for bit, with
/// every optional tensor (`s0`, `s_fin`, `d_fin`, `ds0`) present, and the
/// optional ones are really read. A mis-wired field (q for k, dg for dbeta,
/// a dropped `s_fin`) fails by name. `T = 80` crosses a checkpoint
/// (`GDN_TRAIN_CKPT` = 64), `Dv = 32` is not the 128 a swapped dimension
/// would hide behind, and `B = 2` exercises the batch stride.
///
/// This replaces `python/tests/test_gdn.py`'s ABI-versus-Rust role: that file
/// needs MPS, so no hosted runner runs it. Its transformers-oracle role is
/// not replaced; `tests/gdn_train.rs` holds the kernels to an f64 reference.
#[test]
fn the_gdn_abi_is_the_rust_gdn() {
    use tessl::gdn_train::{
        gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace,
    };
    const SENTINEL: f32 = -7.25e27;
    let (t, h, dv) = (80usize, 2usize, 32usize);
    let handle = Handle::new();
    with_gpu(|rt| {
        for b in [1usize, 2] {
            let n = b * t * h;
            let seed = 100 * b as u64;
            let q = random_f32(n * 128, seed + 1);
            let k = random_f32(n * 128, seed + 2);
            let v = random_f32(n * dv, seed + 3);
            let g: Vec<f32> = random_f32(n, seed + 4)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect();
            let beta: Vec<f32> = random_f32(n, seed + 5).iter().map(|&x| 0.5 + 0.45 * x).collect();
            let s0 = random_f32(b * h * 128 * dv, seed + 6);
            let d_o = random_f32(n * dv, seed + 7);
            let d_fin = random_f32(b * h * 128 * dv, seed + 8);
            let dims = GdnTrainDims {
                batch: b as u32,
                seq: t as u32,
                heads: h as u32,
                v_dim: dv as u32,
            };
            let ck = dims.checkpoint_shape();
            assert_eq!(ck[2], 2, "the case must cross a checkpoint");
            let ck64: Vec<u64> = ck.iter().map(|&d| d as u64).collect();
            let sizes = [
                ("o", n * dv),
                ("s_fin", b * h * 128 * dv),
                ("ckpt", ck.iter().product()),
                ("dq", n * 128),
                ("dk", n * 128),
                ("dv", n * dv),
                ("dg", n),
                ("dbeta", n),
                ("ds0", b * h * 128 * dv),
            ];

            // ---- the ABI side
            let sentinel = |i: usize| shared(rt, &vec![SENTINEL; sizes[i].1]);
            let outs: Vec<_> = (0..sizes.len()).map(sentinel).collect();
            let (qb, kb, vb, gb, bb) = (
                shared(rt, &q),
                shared(rt, &k),
                shared(rt, &v),
                shared(rt, &g),
                shared(rt, &beta),
            );
            let (s0b, dob, dfb) = (shared(rt, &s0), shared(rt, &d_o), shared(rt, &d_fin));
            let (bu, tu, hu, du) = (b as u64, t as u64, h as u64, dv as u64);
            let o_shape = [bu, tu, hu, du];
            let st_shape = [bu, hu, 128, du];
            let args = TesslGdnArgs {
                batch: b as u32,
                seq: t as u32,
                heads: h as u32,
                v_dim: dv as u32,
                q: tref(&qb, &[bu, tu, hu, 128]),
                k: tref(&kb, &[bu, tu, hu, 128]),
                v: tref(&vb, &o_shape),
                g: tref(&gb, &[bu, tu, hu]),
                beta: tref(&bb, &[bu, tu, hu]),
                s0: tref(&s0b, &st_shape),
                ckpt: tref(&outs[2], &ck64),
                o: tref(&outs[0], &o_shape),
                s_fin: tref(&outs[1], &st_shape),
                d_o: tref(&dob, &o_shape),
                d_fin: tref(&dfb, &st_shape),
                dq: tref(&outs[3], &[bu, tu, hu, 128]),
                dk: tref(&outs[4], &[bu, tu, hu, 128]),
                dv: tref(&outs[5], &o_shape),
                dg: tref(&outs[6], &[bu, tu, hu]),
                dbeta: tref(&outs[7], &[bu, tu, hu]),
                ds0: tref(&outs[8], &st_shape),
            };
            let (s, m) = run_gdn(tessl_gdn_train_forward, &handle, &args);
            assert_eq!(s, TESSL_OK, "B={b} forward: {m}");
            let (s, m) = run_gdn(tessl_gdn_train_backward, &handle, &args);
            assert_eq!(s, TESSL_OK, "B={b} backward: {m}");

            // ---- the Rust side
            let up = |d: &[f32], shape: &[usize]| {
                let x = rt.alloc_tensor_f32(shape).unwrap();
                x.write_f32(d).unwrap();
                x
            };
            let (qt, kt) = (up(&q, &[b, t, h, 128]), up(&k, &[b, t, h, 128]));
            let (vt, gt, bt) = (up(&v, &[b, t, h, dv]), up(&g, &[b, t, h]), up(&beta, &[b, t, h]));
            let s0t = up(&s0, &[b, h, 128, dv]);
            let (dot, dft) = (up(&d_o, &[b, t, h, dv]), up(&d_fin, &[b, h, 128, dv]));
            let shapes: [&[usize]; 9] = [
                &[b, t, h, dv],
                &[b, h, 128, dv],
                &ck,
                &[b, t, h, 128],
                &[b, t, h, 128],
                &[b, t, h, dv],
                &[b, t, h],
                &[b, t, h],
                &[b, h, 128, dv],
            ];
            let want: Vec<_> = (0..sizes.len())
                .map(|i| up(&vec![SENTINEL; sizes[i].1], shapes[i]))
                .collect();
            let inputs = GdnTrainInputs {
                q: &qt,
                k: &kt,
                v: &vt,
                g: &gt,
                beta: &bt,
                s0: Some(&s0t),
            };
            gdn_train_forward(rt, dims, inputs, &want[0], Some(&want[1]), &want[2]).unwrap();
            let ws = GdnTrainWorkspace::new(rt, dims).unwrap();
            gdn_train_backward(
                rt,
                dims,
                inputs,
                &want[2],
                &dot,
                Some(&dft),
                &ws,
                GdnTrainGrads {
                    dq: &want[3],
                    dk: &want[4],
                    dv: &want[5],
                    dg: &want[6],
                    dbeta: &want[7],
                    ds0: Some(&want[8]),
                },
            )
            .unwrap();

            for (i, (name, len)) in sizes.iter().enumerate() {
                let (got, exp) = (read(&outs[i], *len), want[i].read_f32().unwrap());
                assert!(
                    exp.iter().all(|x| x.is_finite() && *x != SENTINEL),
                    "B={b} {name}: the Rust call left a sentinel or a non-finite value"
                );
                assert!(
                    got.iter().all(|x| x.is_finite() && *x != SENTINEL),
                    "B={b} {name}: the ABI left a sentinel or a non-finite value"
                );
                assert_eq!(bits(&got), bits(&exp), "B={b} {name}: ABI differs from the Rust call");
            }

            if b != 1 {
                continue;
            }
            // ---- the optional tensors are read, so equality above is not vacuous
            let fresh = |len: usize| shared(rt, &vec![SENTINEL; len]);
            let (o2, ck2) = (fresh(sizes[0].1), fresh(sizes[2].1));
            let no_s0 = TesslGdnArgs {
                s0: NULL_REF,
                s_fin: NULL_REF,
                o: tref(&o2, &o_shape),
                ckpt: tref(&ck2, &ck64),
                ..args
            };
            let (s, m) = run_gdn(tessl_gdn_train_forward, &handle, &no_s0);
            assert_eq!(s, TESSL_OK, "{m}");
            assert_ne!(
                bits(&read(&o2, sizes[0].1)),
                bits(&read(&outs[0], sizes[0].1)),
                "o must depend on s0"
            );
            let dv2 = fresh(sizes[5].1);
            let no_dfin = TesslGdnArgs {
                d_fin: NULL_REF,
                s0: tref(&s0b, &st_shape),
                dv: tref(&dv2, &o_shape),
                ..args
            };
            let (s, m) = run_gdn(tessl_gdn_train_backward, &handle, &no_dfin);
            assert_eq!(s, TESSL_OK, "{m}");
            assert_ne!(
                bits(&read(&dv2, sizes[5].1)),
                bits(&read(&outs[5], sizes[5].1)),
                "dv must depend on d_fin (it flows through the state gradient)"
            );

            // ---- a required tensor that is null is refused by name
            let bad = TesslGdnArgs { q: NULL_REF, ..args };
            for f in [tessl_gdn_train_forward, tessl_gdn_train_backward] {
                let (s, m) = run_gdn(f, &handle, &bad);
                assert_eq!(s, TESSL_ERR);
                assert!(m.contains("q: null MTLBuffer"), "{m}");
            }
        }
    });
}

/// Hostile GDN arguments are refused with an error, never a panic or a
/// poisoned handle, and a refused call writes nothing: zero and overflowing
/// dimensions, `s0` without `ds0` and the reverse, a wrong-shaped `ckpt`, and
/// a v_dim the tensors do not carry. A NaN input flows through without
/// crashing, and the handle still serves a good call afterwards.
#[test]
fn hostile_gdn_args_are_refused_and_leave_outputs_untouched() {
    use tessl::gdn_train::GdnTrainDims;
    const SENTINEL: f32 = -7.25e27;
    let handle = Handle::new();
    with_gpu(|rt| {
        let (b, t, h, dv) = (1usize, 16usize, 2usize, 32usize);
        let buf = |n: usize, seed: u64| shared(rt, &random_f32(n, seed));
        let sent = |n: usize| shared(rt, &vec![SENTINEL; n]);
        let dims = GdnTrainDims {
            batch: 1,
            seq: 16,
            heads: 2,
            v_dim: 32,
        };
        let ck = dims.checkpoint_shape();
        let ck64: Vec<u64> = ck.iter().map(|&d| d as u64).collect();
        let (q, k) = (buf(b * t * h * 128, 1), buf(b * t * h * 128, 2));
        let (v, g, beta) = (buf(b * t * h * dv, 3), buf(b * t * h, 4), buf(b * t * h, 5));
        let s0 = buf(b * h * 128 * dv, 6);
        let ckpt = sent(ck.iter().product());
        let (o, s_fin) = (sent(b * t * h * dv), sent(b * h * 128 * dv));
        let d_o = buf(b * t * h * dv, 7);
        let (dq, dk, dvv) = (sent(b * t * h * 128), sent(b * t * h * 128), sent(b * t * h * dv));
        let (dg, db, ds0) = (sent(b * t * h), sent(b * t * h), sent(b * h * 128 * dv));
        let (bu, tu, hu, du) = (b as u64, t as u64, h as u64, dv as u64);
        let st = [bu, hu, 128, du];
        let os = [bu, tu, hu, du];
        let args = TesslGdnArgs {
            batch: 1,
            seq: 16,
            heads: 2,
            v_dim: 32,
            q: tref(&q, &[bu, tu, hu, 128]),
            k: tref(&k, &[bu, tu, hu, 128]),
            v: tref(&v, &os),
            g: tref(&g, &[bu, tu, hu]),
            beta: tref(&beta, &[bu, tu, hu]),
            s0: tref(&s0, &st),
            ckpt: tref(&ckpt, &ck64),
            o: tref(&o, &os),
            s_fin: tref(&s_fin, &st),
            d_o: tref(&d_o, &os),
            d_fin: NULL_REF,
            dq: tref(&dq, &[bu, tu, hu, 128]),
            dk: tref(&dk, &[bu, tu, hu, 128]),
            dv: tref(&dvv, &os),
            dg: tref(&dg, &[bu, tu, hu]),
            dbeta: tref(&db, &[bu, tu, hu]),
            ds0: tref(&ds0, &st),
        };
        let untouched = |what: &str| {
            for (name, b, n) in [
                ("o", &o, b * t * h * dv),
                ("s_fin", &s_fin, b * h * 128 * dv),
                ("ckpt", &ckpt, ck.iter().product()),
                ("dq", &dq, b * t * h * 128),
                ("dv", &dvv, b * t * h * dv),
                ("dg", &dg, b * t * h),
                ("ds0", &ds0, b * h * 128 * dv),
            ] {
                assert!(read(b, n).iter().all(|x| *x == SENTINEL), "{what}: {name} was written");
            }
        };
        let cases: Vec<(&str, TesslGdnArgs)> = vec![
            ("batch 0", TesslGdnArgs { batch: 0, ..args }),
            ("seq 0", TesslGdnArgs { seq: 0, ..args }),
            ("heads 0", TesslGdnArgs { heads: 0, ..args }),
            ("v_dim 0", TesslGdnArgs { v_dim: 0, ..args }),
            (
                "batch max",
                TesslGdnArgs {
                    batch: u32::MAX,
                    ..args
                },
            ),
            ("seq max", TesslGdnArgs { seq: u32::MAX, ..args }),
            (
                "heads max",
                TesslGdnArgs {
                    heads: u32::MAX,
                    ..args
                },
            ),
            (
                "v_dim max",
                TesslGdnArgs {
                    v_dim: u32::MAX,
                    ..args
                },
            ),
            ("v_dim 33", TesslGdnArgs { v_dim: 33, ..args }),
            ("v_dim 64 on 32 tensors", TesslGdnArgs { v_dim: 64, ..args }),
            (
                "ckpt too small",
                TesslGdnArgs {
                    ckpt: tref(&ckpt, &[bu, hu, 0, 128, du]),
                    ..args
                },
            ),
            (
                "s0 shape wrong",
                TesslGdnArgs {
                    s0: tref(&s0, &[bu, hu, du, 128]),
                    ..args
                },
            ),
        ];
        let run = |f: GdnFn, a: &TesslGdnArgs| run_gdn(f, &handle, a);
        for (name, a) in &cases {
            for f in [tessl_gdn_train_forward, tessl_gdn_train_backward] {
                let (s, m) = run(f, a);
                assert_eq!(s, TESSL_ERR, "{name}: accepted ({m})");
                assert!(!m.is_empty(), "{name}: refused without a message");
            }
            untouched(name);
            assert_eq!(poisoned(handle.0), 0, "{name}: poisoned the handle");
        }
        // ds0 exactly when s0 is given, both directions of the mismatch.
        let (s, m) = run(tessl_gdn_train_backward, &TesslGdnArgs { ds0: NULL_REF, ..args });
        assert_eq!(s, TESSL_ERR, "s0 without ds0 accepted ({m})");
        let (s, m) = run(tessl_gdn_train_backward, &TesslGdnArgs { s0: NULL_REF, ..args });
        assert_eq!(s, TESSL_ERR, "ds0 without s0 accepted ({m})");
        untouched("ds0 mismatch");
        assert_eq!(poisoned(handle.0), 0);

        // NaN in q: the call completes, the NaN reaches o, the handle survives.
        let mut qn = random_f32(b * t * h * 128, 1);
        qn[0] = f32::NAN;
        let qnb = shared(rt, &qn);
        let a = TesslGdnArgs {
            q: tref(&qnb, &[bu, tu, hu, 128]),
            ..args
        };
        let (s, m) = run(tessl_gdn_train_forward, &a);
        assert_eq!(s, TESSL_OK, "{m}");
        assert!(
            read(&o, b * t * h * dv).iter().any(|x| x.is_nan()),
            "NaN q must reach o"
        );
        assert_eq!(poisoned(handle.0), 0);

        // The same handle still serves a good call.
        let (s, m) = run(tessl_gdn_train_forward, &args);
        assert_eq!(s, TESSL_OK, "{m}");
        let (s, m) = run(tessl_gdn_train_backward, &args);
        assert_eq!(s, TESSL_OK, "{m}");
        assert!(read(&dq, b * t * h * 128).iter().all(|x| x.is_finite()));
    });
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

/// A bad `dh` shape on `train_backward` after a forward, async encode on, is
/// refused and the pending step survives it: the real backward still runs.
/// This does **not** show the error-path wait: the arguments are checked
/// before any kernel is encoded, so it passes with or without that wait. The
/// test above is the one that fails without it.
#[test]
fn a_refused_backward_keeps_the_pending_step_under_async_encode() {
    let handle = Handle::new();
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    let model = load_model(handle.0);
    let rt = unsafe { tessl::capi::test_model_runtime(model) };
    assert!(rt.async_encode_enabled());
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    let mut loss = 0.0f64;
    let s = unsafe {
        tessl_qwen35_train_forward(
            model,
            ids.as_ptr(),
            ids.len() as u64,
            TESSL_OPERANDS_EXACT_F32,
            TESSL_SUPERVISE_CAUSAL,
            ptr::null(),
            ptr::null(),
            0,
            1.0,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert_idle(&rt, "a forward");
    let wrong = shared(&rt, &[0.0; 3]);
    let dh = tref(&wrong, &[1, 3]);
    let s = unsafe { tessl_qwen35_train_backward(model, [0u32].as_ptr(), 1, &dh, 0, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("src must be f32 [1, 64]"), "{}", msg(&err));
    assert_idle(&rt, "a refused backward");
    // The refusal kept the step pending: the real backward still runs.
    let s = unsafe { tessl_qwen35_train_backward(model, ptr::null(), 0, ptr::null(), 0, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert_idle(&rt, "a backward");
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
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

const GUARD: f32 = f32::from_bits(0x7FC0_DEAD);

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
        let (s, dst, m) = read_entry(model, i, n, GUARD);
        assert_eq!(s, TESSL_OK, "{m}");
        assert!(
            dst[n..].iter().all(|v| v.to_bits() == GUARD.to_bits()),
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
        let (s, got, m) = read_entry(model, i, numel(&infos[i]), GUARD);
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
            let (s, got, m) = read_entry(model, j, numel(p), GUARD);
            assert_eq!(s, TESSL_OK, "{m}");
            assert_eq!(bits(&got[..numel(p)]), bits(&whole[j]), "{what}: entry {j} moved");
        }
    };
    // Reads: a refused read writes nothing into the destination.
    for len in [n - 1, n + 1, 0] {
        let mut dst = vec![GUARD; n + 8];
        let s = unsafe {
            tessl_qwen35_read_parameter_host(model, i as u64, dst.as_mut_ptr(), len as u64, err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(s, TESSL_ERR, "read n={len}");
        assert!(msg(&err).contains("takes"), "n={len}: {}", msg(&err));
        assert!(
            dst.iter().all(|v| v.to_bits() == GUARD.to_bits()),
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
    let mut dst = vec![GUARD; 16];
    let s =
        unsafe { tessl_qwen35_read_parameter_host(model, u64::MAX, dst.as_mut_ptr(), 1, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("out of range"), "{}", msg(&err));
    assert!(dst.iter().all(|v| v.to_bits() == GUARD.to_bits()));
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

    // A write during a pending step is refused, and the step survives it.
    let ids = fixture_ids();
    let mut loss = 0.0f64;
    let s = unsafe {
        tessl_qwen35_train_forward(
            model,
            ids.as_ptr(),
            ids.len() as u64,
            TESSL_OPERANDS_EXACT_F32,
            TESSL_SUPERVISE_CAUSAL,
            ptr::null(),
            ptr::null(),
            0,
            1.0,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    let (s, m) = write_entry(model, i, &whole[i].iter().map(|v| v + 1.0).collect::<Vec<_>>());
    assert_eq!(s, TESSL_ERR);
    assert!(m.contains("pending"), "{m}");
    unchanged("a write during a pending step");
    let s = unsafe { tessl_qwen35_train_backward(model, ptr::null(), 0, ptr::null(), 0, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_OK, "the refused write consumed the step: {}", msg(&err));
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
    let (s, before, m) = read_entry(model, 0, n, GUARD);
    assert_eq!(s, TESSL_OK, "{m}");
    rt.poison_as_shared_event_timeout_for_test();
    assert_eq!(poisoned(handle.0), 1);
    let (s, dst, _) = read_entry(model, 0, n, GUARD);
    assert_eq!(s, TESSL_ERR, "a read on a poisoned runtime");
    assert!(
        dst.iter().all(|v| v.to_bits() == GUARD.to_bits()),
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

/// A per-entry host write is refused, before anything moves, for an entry
/// whose optimizer keeps auxiliary state (a bf16 entry under f32-master or
/// bf16-kahan): the next step would rebuild the weight from that state. The
/// weight and the auxiliary state are untouched, and entries without aux
/// state (f32 entries), the other rules and a model with no optimizer still
/// accept the write.
#[test]
fn a_host_write_that_the_optimizer_would_overwrite_is_refused() {
    for (rule, name, update, seed) in [
        ("f32-master", "f32-master", TESSL_UPDATE_F32_MASTER, 0u64),
        ("bf16-kahan", "bf16-kahan", TESSL_UPDATE_BF16_KAHAN, 0),
        ("bf16-stochastic", "", TESSL_UPDATE_BF16_STOCHASTIC, 7),
    ] {
        let handle = Handle::new();
        let model = load_model_as(handle.0, TESSL_BF16);
        let rt = unsafe { tessl::capi::test_model_runtime(model) };
        let infos = param_infos(model);
        let mut err = [0 as c_char; ERR_LEN];
        let s = unsafe { tessl_qwen35_adamw_init(model, update, seed, TESSL_MOMENTS_F32, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_OK, "{rule}: {}", msg(&err));
        let keeps_aux = update != TESSL_UPDATE_BF16_STOCHASTIC;
        let weights = read_all(model, &rt, &infos, TESSL_READ_PARAMS);
        let aux = keeps_aux.then(|| read_all(model, &rt, &infos, TESSL_READ_ADAMW_AUX));
        let (mut refused, mut accepted) = (0, 0);
        for (i, p) in infos.iter().enumerate() {
            let changed: Vec<f32> = weights[i].iter().map(|v| v + 1.0).collect();
            let (s, m) = write_entry(model, i, &changed);
            if s == TESSL_OK {
                accepted += 1;
                continue;
            }
            refused += 1;
            assert_eq!(s, TESSL_ERR);
            assert!(
                m.contains(&format!("index {i}")) && m.contains(name) && m.contains("TESSL_WRITE_ADAMW_AUX"),
                "entry {i}: {m}"
            );
            let (s, got, e) = read_entry(model, i, numel(p), GUARD);
            assert_eq!(s, TESSL_OK, "{e}");
            assert_eq!(
                bits(&got[..numel(p)]),
                bits(&weights[i]),
                "{rule}: a refused write moved entry {i}"
            );
        }
        if let Some(aux) = aux {
            assert!(refused > 0, "{rule}: no write was refused");
            assert!(accepted > 0, "{rule}: the f32 entries (no aux state) were refused too");
            let after = read_all(model, &rt, &infos, TESSL_READ_ADAMW_AUX);
            for i in 0..infos.len() {
                assert_eq!(
                    bits(&after[i]),
                    bits(&aux[i]),
                    "{rule}: a refused write moved entry {i}'s aux state"
                );
            }
        } else {
            assert_eq!(refused, 0, "{rule} keeps no aux state, yet a write was refused");
        }
        assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
    }
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
    assert_eq!(TESSL_ABI_VERSION, 12);
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

/// Switching async encode between a forward and its backward leaves the
/// pending step intact: the gradients are bit for bit those of an untouched
/// step, whether the backward then runs async or synchronous.
#[test]
fn toggling_async_encode_during_a_pending_step_changes_nothing() {
    let handle = Handle::new();
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    let model = load_model(handle.0);
    let rt = unsafe { tessl::capi::test_model_runtime(model) };
    let infos = param_infos(model);
    let ids = fixture_ids();
    let mut err = [0 as c_char; ERR_LEN];
    let forward = |err: &mut [c_char; ERR_LEN]| -> f64 {
        let mut loss = 0.0f64;
        let s = unsafe {
            tessl_qwen35_train_forward(
                model,
                ids.as_ptr(),
                ids.len() as u64,
                TESSL_OPERANDS_EXACT_F32,
                TESSL_SUPERVISE_CAUSAL,
                ptr::null(),
                ptr::null(),
                0,
                1.0,
                &mut loss,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(err));
        loss
    };
    let backward = |err: &mut [c_char; ERR_LEN]| {
        let s =
            unsafe { tessl_qwen35_train_backward(model, ptr::null(), 0, ptr::null(), 0, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_OK, "{}", msg(err));
    };
    let loss = forward(&mut err);
    backward(&mut err);
    let reference = read_all(model, &rt, &infos, TESSL_READ_GRADS);
    // Off and back on in between.
    assert_eq!(forward(&mut err).to_bits(), loss.to_bits());
    assert_eq!(set_async(&handle, 0), TESSL_OK);
    assert_eq!(set_async(&handle, 1), TESSL_OK);
    backward(&mut err);
    let toggled = read_all(model, &rt, &infos, TESSL_READ_GRADS);
    // Off for the backward.
    assert_eq!(forward(&mut err).to_bits(), loss.to_bits());
    assert_eq!(set_async(&handle, 0), TESSL_OK);
    backward(&mut err);
    let sync = read_all(model, &rt, &infos, TESSL_READ_GRADS);
    for i in 0..infos.len() {
        assert_eq!(bits(&toggled[i]), bits(&reference[i]), "entry {i} after off and on");
        assert_eq!(
            bits(&sync[i]),
            bits(&reference[i]),
            "entry {i} with the backward synchronous"
        );
    }
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
}
