//! The C ABI (`tessl::capi`), called the way a foreign caller calls it.
//!
//! `python/tests/test_cross_entropy.py` covers the torch path end to end;
//! this covers the boundary itself without torch: statuses and messages for
//! every refusal, message truncation, the thread check, and a successful call
//! producing exactly what the Rust entry point produces.

mod common;

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use common::{random_f32, with_gpu};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use tessl::capi::{
    tessl_abi_version, tessl_cross_entropy_rows, tessl_mtl_buffer_length, tessl_qwen35_copy, tessl_qwen35_free,
    tessl_qwen35_load, tessl_qwen35_param_count, tessl_qwen35_param_info, tessl_qwen35_train_step, tessl_runtime_free,
    tessl_runtime_new, tessl_synchronize, TesslCeArgs, TesslRuntime, TesslTensorRef, TESSL_ABI_VERSION,
    TESSL_ERR, TESSL_F32, TESSL_MAX_DIMS, TESSL_OK, TESSL_READ_GRADS, TESSL_READ_PARAMS, TESSL_WRITE_PARAMS,
    TesslParamInfo, TesslQwen35,
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
            chunk: 128,
            want_grads: 1,
            scale: 0.5,
            dh: tref(&dhb, &[n as u64, h as u64]),
            dw: tref(&dwb, &[v as u64, h as u64]),
        };
        let (mut loss, mut per_row) = (0.0f64, [0.0f64; 3]);
        let mut err = [0 as c_char; ERR_LEN];
        let status = unsafe {
            tessl_cross_entropy_rows(handle.0, &args, &mut loss, per_row.as_mut_ptr(), err.as_mut_ptr(), ERR_LEN)
        };
        assert_eq!(status, TESSL_OK, "{}", msg(&err));

        let ht = rt.alloc_tensor_f32(&[t, h]).unwrap();
        ht.write_f32(&hid).unwrap();
        let wt = rt.alloc_tensor_f32(&[v, h]).unwrap();
        wt.write_f32(&w).unwrap();
        let (dh, dw) = (rt.alloc_tensor_f32(&[n, h]).unwrap(), rt.alloc_tensor_f32(&[v, h]).unwrap());
        let ws = CeWorkspace::new(rt, 4, h as u32, 128, DType::F32).unwrap();
        let want = cross_entropy_rows(
            rt,
            CeHidden { rows: &ht, off: 0 },
            &wt,
            &rows,
            &targets,
            Reduction::Mean,
            GemmOperands::ExactF32,
            &ws,
            Some(CeGrads { dh: &dh, dw: &dw, scale: 0.5 }),
        )
        .unwrap();
        assert_eq!(loss.to_bits(), want.loss.to_bits());
        assert_eq!(per_row.to_vec(), want.per_row);
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(read(&dhb, n * h)), bits(dh.read_f32().unwrap()));
        assert_eq!(bits(read(&dwb, v * h)), bits(dw.read_f32().unwrap()));

        // The handle's cached workspace serves a second, smaller call.
        let args1 = TesslCeArgs { n: 1, want_grads: 0, dh: NULL_REF, dw: NULL_REF, ..args };
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
        refused(TesslCeArgs { rows: ptr::null(), ..good }, "null rows or targets");
        refused(TesslCeArgs { reduction: 7, ..good }, "reduction 7");
        refused(TesslCeArgs { hidden: NULL_REF, ..good }, "hidden: null MTLBuffer");
        refused(TesslCeArgs { hidden: TesslTensorRef { dtype: 9, ..good.hidden }, ..good }, "unknown dtype code 9");
        refused(TesslCeArgs { hidden: TesslTensorRef { ndim: 7, ..good.hidden }, ..good }, "rank 7");
        refused(TesslCeArgs { weight: TesslTensorRef { ndim: 1, ..good.weight }, ..good }, "weight must be 2-D");
        refused(
            TesslCeArgs { hidden: TesslTensorRef { byte_offset: 8, ..good.hidden }, ..good },
            "hidden: tensor view is misaligned or out of bounds",
        );
        refused(TesslCeArgs { want_grads: 1, ..good }, "dh: null MTLBuffer");
        let bad_targets = [5u32, 300];
        refused(TesslCeArgs { targets: bad_targets.as_ptr(), ..good }, "targets[1] = 300");

        // A message longer than the caller's buffer is cut at a character
        // boundary and still NUL-terminated.
        let mut small = [0x7f as c_char; 8];
        let mut loss = 0.0;
        let s = unsafe {
            tessl_cross_entropy_rows(ptr::null_mut(), &good, &mut loss, ptr::null_mut(), small.as_mut_ptr(), small.len())
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
        assert!(from_thread.1.contains("other than the one that created it"), "{}", from_thread.1);

        // After all of that the handle still works.
        assert_eq!(call(handle.0, &good).0, TESSL_OK);
        let mut err = [0 as c_char; ERR_LEN];
        assert_eq!(unsafe { tessl_synchronize(handle.0, err.as_mut_ptr(), ERR_LEN) }, TESSL_OK);
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
            let s = unsafe { tessl_cross_entropy_rows(handle.0, &args, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN) };
            assert_eq!(s, TESSL_OK, "{}", msg(&err));
            if i == 0 {
                first = loss;
            }
            assert_eq!(loss.to_bits(), first.to_bits(), "call {i} changed the answer");
        }
    });
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
    assert_eq!(unsafe { tessl_synchronize(handle.0, err.as_mut_ptr(), ERR_LEN) }, TESSL_OK, "{}", msg(&err));
    drop(handle); // the owner's free succeeds (asserted in Drop)
}

// ------------------------------------------------------------ Qwen3.5 ---

fn fixture(name: &str) -> std::ffi::CString {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train").join(name);
    std::ffi::CString::new(p.to_str().unwrap()).unwrap()
}

fn fixture_ids() -> Vec<u32> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train/ids.npy");
    tessl::npy::read_npy(&p).unwrap().i64_slice().unwrap().iter().map(|&x| x as u32).collect()
}

fn load_model(rt: *mut TesslRuntime) -> *mut TesslQwen35 {
    let (st, cfg, prefix) = (fixture("model.safetensors"), fixture("config.json"), std::ffi::CString::new("model.").unwrap());
    let mut out = ptr::null_mut();
    let mut err = [0 as c_char; ERR_LEN];
    let s = unsafe { tessl_qwen35_load(rt, st.as_ptr(), cfg.as_ptr(), prefix.as_ptr(), &mut out, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_OK, "{}", msg(&err));
    assert!(!out.is_null());
    out
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
        let st = tessl::safetensors::SafeTensors::open(std::path::Path::new(fixture("model.safetensors").to_str().unwrap())).unwrap();
        let cfg = tessl::qwen35_model::Qwen35Config::from_config_file(std::path::Path::new(fixture("config.json").to_str().unwrap())).unwrap();
        let rust = tessl::qwen35_model::Qwen35Model::load(rt, &st, "model.", cfg, tessl::qwen35_model::Precision::F32).unwrap();
        let table = rust.parameter_table().unwrap();

        let mut n = 0u64;
        assert_eq!(unsafe { tessl_qwen35_param_count(model, &mut n, err.as_mut_ptr(), ERR_LEN) }, TESSL_OK);
        assert_eq!(n as usize, table.len());
        for (i, p) in table.iter().enumerate() {
            let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { tessl_qwen35_param_info(model, i as u64, &mut info, err.as_mut_ptr(), ERR_LEN) }, TESSL_OK);
            assert_eq!(msg(&info.name), p.name);
            assert_eq!(&info.shape[..info.ndim as usize], p.shape.iter().map(|&d| d as u64).collect::<Vec<_>>().as_slice());
            assert_eq!(info.transposed != 0, p.transposed);
        }
        let mut info: TesslParamInfo = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { tessl_qwen35_param_info(model, n, &mut info, err.as_mut_ptr(), ERR_LEN) }, TESSL_ERR);
        assert!(msg(&err).contains(&format!("index {n} is outside")), "{}", msg(&err));

        // Caller buffers, one per entry, of each entry's storage shape.
        let bufs: Vec<_> = table.iter().map(|p| shared(rt, &vec![0.0; p.shape.iter().product()])).collect();
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
        assert!(msg(&err).contains(&format!("{} tensors for {} parameters", n - 1, n)), "{}", msg(&err));
        assert_eq!(copy(7, &refs, &mut err), TESSL_ERR);
        assert!(msg(&err).contains("direction 7"), "{}", msg(&err));

        let mut loss = 0.0f64;
        let s = unsafe { tessl_qwen35_train_step(model, ids.as_ptr(), ids.len() as u64, &mut loss, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        let want = rust.train_step(&ids, GemmOperands::ExactF32).unwrap();
        assert_eq!(loss.to_bits(), want.loss.to_bits());

        let local: Vec<tessl::Tensor> = table.iter().map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap()).collect();
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
                assert!(got.iter().zip(&exp).all(|(a, b)| a.to_bits() == b.to_bits()), "{} (direction {dir})", p.name);
            }
        }

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
        let s = unsafe { tessl_qwen35_train_step(model, ids.as_ptr(), ids.len() as u64, &mut moved, err.as_mut_ptr(), ERR_LEN) };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        assert_ne!(moved.to_bits(), loss.to_bits());
        assert_eq!(moved.to_bits(), rust.train_step(&ids, GemmOperands::ExactF32).unwrap().loss.to_bits());
    });

    // Refusals: a bad id, a null handle, another thread.
    let bad = [1u32, 64];
    let mut loss = 0.0;
    let s = unsafe { tessl_qwen35_train_step(model, bad.as_ptr(), 2, &mut loss, err.as_mut_ptr(), ERR_LEN) };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("token id 64 >= vocab 64"), "{}", msg(&err));
    let mut n = 0u64;
    assert_eq!(unsafe { tessl_qwen35_param_count(ptr::null_mut(), &mut n, err.as_mut_ptr(), ERR_LEN) }, TESSL_ERR);
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
    assert!(m.contains("tessl model used from a thread other than the one that created it"), "{m}");
    assert_eq!(unsafe { tessl_qwen35_free(model) }, TESSL_OK);
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
    let load = |st: *const c_char, cfg: *const c_char, prefix: *const c_char, out: *mut *mut TesslQwen35, err: &mut [c_char; ERR_LEN]| unsafe {
        tessl_qwen35_load(handle.0, st, cfg, prefix, out, err.as_mut_ptr(), ERR_LEN)
    };
    assert_eq!(load(ptr::null(), cfg.as_ptr(), prefix.as_ptr(), &mut out, &mut err), TESSL_ERR);
    assert!(msg(&err).contains("null safetensors path"), "{}", msg(&err));
    assert_eq!(load(missing.as_ptr(), cfg.as_ptr(), prefix.as_ptr(), &mut out, &mut err), TESSL_ERR);
    assert!(msg(&err).contains("/nonexistent/model.safetensors"), "{}", msg(&err));
    assert_eq!(load(st.as_ptr(), cfg.as_ptr(), wrong_prefix.as_ptr(), &mut out, &mut err), TESSL_ERR);
    assert!(msg(&err).contains("model.language_model."), "{}", msg(&err));
    assert!(out.is_null(), "a failed load leaves *out null");
    assert_eq!(load(st.as_ptr(), cfg.as_ptr(), prefix.as_ptr(), ptr::null_mut(), &mut err), TESSL_ERR);
    assert!(msg(&err).contains("null out"), "{}", msg(&err));
    assert_eq!(unsafe { tessl_qwen35_free(ptr::null_mut()) }, TESSL_OK);
}
