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
    tessl_abi_version, tessl_cross_entropy_rows, tessl_mtl_buffer_length, tessl_runtime_free,
    tessl_runtime_new, tessl_synchronize, TesslCeArgs, TesslRuntime, TesslTensorRef, TESSL_ABI_VERSION,
    TESSL_ERR, TESSL_F32, TESSL_OK,
};
use tessl::cross_entropy::{cross_entropy_rows, CeGrads, CeHidden, CeWorkspace, Reduction};
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
    let mut s = [0u64; 4];
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
    shape: [0; 4],
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
        refused(TesslCeArgs { hidden: TesslTensorRef { ndim: 5, ..good.hidden }, ..good }, "rank 5");
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
