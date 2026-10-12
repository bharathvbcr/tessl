//! A caller's `n` too large for a slice is a refusal, not undefined behaviour.
//!
//! Its own test binary: before the byte cap, an oversized `n` made the standard
//! library's `from_raw_parts` precondition check abort the process under a
//! debug build (a non-unwinding panic `catch_unwind` does not catch), which
//! would otherwise take the rest of `tests/capi.rs` down with it.

mod common;

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

use common::{random_f32, with_gpu};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use tessl::capi::{
    tessl_cross_entropy_rows, tessl_qwen35_free, tessl_qwen35_hidden, tessl_qwen35_load, tessl_qwen35_train_backward,
    tessl_qwen35_train_forward, tessl_qwen35_train_step, tessl_runtime_free, tessl_runtime_new, TesslCeArgs,
    TesslQwen35, TesslRuntime, TesslTensorRef, TESSL_ERR, TESSL_F32, TESSL_MAX_DIMS, TESSL_OK,
    TESSL_OPERANDS_EXACT_F32, TESSL_SUPERVISE_ROWS,
};
use tessl::GpuRuntime;

const ERR_LEN: usize = 512;
/// Past `isize::MAX` bytes for any element of 2 or more bytes.
const HUGE: u64 = u64::MAX / 2;

fn msg(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned()
}

fn shared(rt: &GpuRuntime, data: &[f32]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let b = rt
        .device
        .newBufferWithLength_options(data.len().max(1) * 4, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe { ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr().cast::<f32>(), data.len()) };
    b
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

fn fixture(name: &str) -> std::ffi::CString {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen35_train")
        .join(name);
    std::ffi::CString::new(p.to_str().unwrap()).unwrap()
}

struct Handles(*mut TesslRuntime, *mut TesslQwen35);

impl Handles {
    fn new() -> Self {
        let mut err = [0 as c_char; ERR_LEN];
        let rt = unsafe { tessl_runtime_new(err.as_mut_ptr(), ERR_LEN) };
        assert!(!rt.is_null(), "{}", msg(&err));
        let (st, cfg, prefix) = (
            fixture("model.safetensors"),
            fixture("config.json"),
            std::ffi::CString::new("model.").unwrap(),
        );
        let mut model = ptr::null_mut();
        let s = unsafe {
            tessl_qwen35_load(
                rt,
                st.as_ptr(),
                cfg.as_ptr(),
                prefix.as_ptr(),
                TESSL_F32,
                &mut model,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        assert_eq!(s, TESSL_OK, "{}", msg(&err));
        Self(rt, model)
    }
}

impl Drop for Handles {
    fn drop(&mut self) {
        assert_eq!(unsafe { tessl_qwen35_free(self.1) }, TESSL_OK);
        assert_eq!(unsafe { tessl_runtime_free(self.0) }, TESSL_OK);
    }
}

#[test]
fn an_oversized_n_is_refused_not_undefined() {
    let hs = Handles::new();
    let ids = [1u32, 2, 3, 4];
    let mut err = [0 as c_char; ERR_LEN];
    let mut loss = 0.0f64;
    let too_big = |what: &str, s: i32, err: &[c_char]| {
        assert_eq!(s, TESSL_ERR, "{what}: {}", msg(err));
        let m = msg(err);
        assert!(m.contains(what), "{what}: {m}");
    };

    let s = unsafe {
        tessl_qwen35_train_step(
            hs.1,
            ids.as_ptr(),
            HUGE,
            TESSL_OPERANDS_EXACT_F32,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    too_big("tessl_qwen35_train_step", s, &err);
    // The largest u32 count that still fits a slice is not the cap's business,
    // the first count past it is: isize::MAX / 4 + 1 u32s.
    let s = unsafe {
        tessl_qwen35_train_step(
            hs.1,
            ids.as_ptr(),
            (isize::MAX as u64) / 4 + 1,
            TESSL_OPERANDS_EXACT_F32,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    too_big("largest slice", s, &err);

    let s = unsafe {
        tessl_qwen35_train_forward(
            hs.1,
            ids.as_ptr(),
            HUGE,
            TESSL_OPERANDS_EXACT_F32,
            0,
            ptr::null(),
            ptr::null(),
            0,
            1.0,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    too_big("tessl_qwen35_train_forward", s, &err);
    let s = unsafe {
        tessl_qwen35_train_forward(
            hs.1,
            ids.as_ptr(),
            ids.len() as u64,
            TESSL_OPERANDS_EXACT_F32,
            TESSL_SUPERVISE_ROWS,
            ids.as_ptr(),
            ids.as_ptr(),
            HUGE,
            1.0,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    too_big("positions", s, &err);

    with_gpu(|rt| {
        let out = shared(rt, &[0.0; 8]);
        let out_ref = tref(&out, &[1, 8]);
        let s = unsafe { tessl_qwen35_hidden(hs.1, ids.as_ptr(), HUGE, &out_ref, err.as_mut_ptr(), ERR_LEN) };
        too_big("tessl_qwen35_hidden", s, &err);
        let s =
            unsafe { tessl_qwen35_train_backward(hs.1, ids.as_ptr(), HUGE, &out_ref, 0, err.as_mut_ptr(), ERR_LEN) };
        too_big("tessl_qwen35_train_backward", s, &err);

        let (hb, wb) = (shared(rt, &random_f32(640, 5)), shared(rt, &random_f32(64 * 300, 6)));
        let args = TesslCeArgs {
            hidden: tref(&hb, &[10, 64]),
            col_off: 0,
            weight: tref(&wb, &[300, 64]),
            rows: ids.as_ptr(),
            targets: ids.as_ptr(),
            n: HUGE,
            reduction: 1,
            operands: TESSL_OPERANDS_EXACT_F32,
            chunk: 64,
            want_grads: 0,
            scale: 1.0,
            dh: tref(&hb, &[10, 64]),
            dw: tref(&wb, &[300, 64]),
        };
        let s = unsafe { tessl_cross_entropy_rows(hs.0, &args, &mut loss, ptr::null_mut(), err.as_mut_ptr(), ERR_LEN) };
        too_big("tessl_cross_entropy_rows", s, &err);
    });
}

/// A null pointer with elements is refused, not read.
#[test]
fn a_null_slice_with_elements_is_refused() {
    let hs = Handles::new();
    let mut err = [0 as c_char; ERR_LEN];
    let mut loss = 0.0f64;
    let s = unsafe {
        tessl_qwen35_train_forward(
            hs.1,
            [1u32].as_ptr(),
            1,
            TESSL_OPERANDS_EXACT_F32,
            TESSL_SUPERVISE_ROWS,
            ptr::null(),
            ptr::null(),
            3,
            1.0,
            &mut loss,
            err.as_mut_ptr(),
            ERR_LEN,
        )
    };
    assert_eq!(s, TESSL_ERR);
    assert!(msg(&err).contains("null"), "{}", msg(&err));
}
