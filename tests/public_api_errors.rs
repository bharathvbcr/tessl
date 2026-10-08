//! Rejection paths, asserted by message rather than by `is_err()`.
//!
//! Every one of these returns `Result<_, String>` with a specific sentence, and
//! a caller that only checks `is_err()` cannot tell "your K does not match" from
//! "your tensor belongs to another runtime" -- so a validator that started
//! failing for the wrong reason would keep every such test green. Pinning the
//! message is also what keeps the checks in `validate_gemm` ordered the way the
//! callers depend on: cheap metadata first, aliasing last, and nothing encoded
//! before all of them have run.
//!
//! Each test also asserts that a rejected call encoded nothing. A validator
//! that rejects *after* allocating scratch or opening a binder leaks work on
//! every bad call, and `take_dispatch_count` is the public way to see it.

mod common;

use common::{with_gpu, with_two_gpus};
use tessl::gemm::{gemm_nt_f32, gemm_tn_f32};
use tessl::tensor::gpu_copy;
use tessl::{gemm, gemm_f32, softcap_f32, DType, GemmBackend, GpuRuntime, Tensor};

#[track_caller]
fn expect_err<T>(what: &str, result: Result<T, String>, expected: &str) {
    match result {
        Ok(_) => panic!("{what}: expected rejection with {expected:?}, but the call succeeded"),
        Err(e) => assert_eq!(e, expected, "{what}: wrong rejection message"),
    }
}

#[track_caller]
fn expect_err_starting<T>(what: &str, result: Result<T, String>, prefix: &str) {
    match result {
        Ok(_) => panic!("{what}: expected rejection starting {prefix:?}, but the call succeeded"),
        Err(e) => assert!(
            e.starts_with(prefix),
            "{what}: expected a message starting {prefix:?}, got {e:?}"
        ),
    }
}

const NOT_RANK2: &str = "GEMM requires nonempty rank-2 tensors";
const DIM_MISMATCH: &str = "GEMM inner dimensions or output shape do not match";
const CROSS_RUNTIME: &str = "GEMM tensors must belong to the same runtime";
const BAD_DTYPE: &str = "GEMM operand dtype does not match the selected precision path";
const OVERLAP: &str = "GEMM output must not overlap either input";
const MIXED_OPERANDS: &str =
    "GEMM requires matching operand dtypes; bf16 and f16 require TensorOps";

#[test]
fn gemm_rejects_mismatched_dimensions() {
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let a = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let c = rt.alloc_tensor_f32(&[32, 32]).unwrap();

        // K disagrees between A's columns and B's rows.
        let b_short = rt.alloc_tensor_f32(&[16, 32]).unwrap();
        expect_err(
            "NN inner dim",
            gemm_f32(&a, &b_short, &c, GemmBackend::TensorOps),
            DIM_MISMATCH,
        );

        // K agrees but C is the wrong width, which a caller reusing a scratch
        // buffer across two different N gets wrong far more often than K.
        let b = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let c_narrow = rt.alloc_tensor_f32(&[32, 16]).unwrap();
        expect_err(
            "NN output shape",
            gemm_f32(&a, &b, &c_narrow, GemmBackend::TensorOps),
            DIM_MISMATCH,
        );

        // TN and NT read the same buffers with different axes, so a shape that
        // is valid for one is a defect for the other and each needs its own check.
        expect_err(
            "TN inner dim",
            gemm_tn_f32(&a, &b_short, &c, GemmBackend::TensorOps),
            DIM_MISMATCH,
        );
        expect_err(
            "NT inner dim",
            gemm_nt_f32(&a, &b_short, &c, GemmBackend::TensorOps),
            DIM_MISMATCH,
        );

        assert_eq!(rt.take_dispatch_count(), 0, "rejected GEMM still encoded");
    });
}

#[test]
fn gemm_rejects_non_rank2_and_empty_extents() {
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let a = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let b = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let c = rt.alloc_tensor_f32(&[32, 32]).unwrap();

        let vector = rt.alloc_tensor_f32(&[1024]).unwrap();
        expect_err(
            "rank-1 operand",
            gemm_f32(&vector, &b, &c, GemmBackend::TensorOps),
            NOT_RANK2,
        );
        let cube = rt.alloc_tensor_f32(&[4, 8, 32]).unwrap();
        expect_err(
            "rank-3 operand",
            gemm_f32(&cube, &b, &c, GemmBackend::TensorOps),
            NOT_RANK2,
        );

        // A zero extent is rank-2 and passes every bounds check, so it has to be
        // rejected by name; a kernel handed it would dispatch a zero-tile grid.
        let empty = a.view(&[0, 32], 0);
        expect_err(
            "empty operand",
            gemm_f32(&empty, &b, &c, GemmBackend::TensorOps),
            NOT_RANK2,
        );
        let empty_out = c.view(&[32, 0], 0);
        expect_err(
            "empty output",
            gemm_f32(&a, &b, &empty_out, GemmBackend::TensorOps),
            NOT_RANK2,
        );

        assert_eq!(rt.take_dispatch_count(), 0, "rejected GEMM still encoded");
    });
}

#[test]
fn gemm_rejects_operands_from_another_runtime() {
    with_two_gpus(|first, second| {
        first.take_dispatch_count();
        second.take_dispatch_count();
        let a = first.alloc_tensor_f32(&[32, 32]).unwrap();
        let b = first.alloc_tensor_f32(&[32, 32]).unwrap();
        let foreign_c = second.alloc_tensor_f32(&[32, 32]).unwrap();
        expect_err(
            "foreign output",
            gemm_f32(&a, &b, &foreign_c, GemmBackend::TensorOps),
            CROSS_RUNTIME,
        );
        let foreign_b = second.alloc_tensor_f32(&[32, 32]).unwrap();
        let c = first.alloc_tensor_f32(&[32, 32]).unwrap();
        expect_err(
            "foreign operand",
            gemm_f32(&a, &foreign_b, &c, GemmBackend::TensorOps),
            CROSS_RUNTIME,
        );
        assert_eq!(first.take_dispatch_count(), 0);
        assert_eq!(second.take_dispatch_count(), 0);
    });
}

#[test]
fn gemm_rejects_dtypes_the_selected_path_cannot_run() {
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let a32 = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let b32 = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let a16 = rt.alloc_tensor_bf16(&[32, 32]).unwrap();
        let b16 = rt.alloc_tensor_bf16(&[32, 32]).unwrap();
        let c32 = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let c16 = rt.alloc_tensor_bf16(&[32, 32]).unwrap();

        // C is always f32: bf16 GEMMs accumulate in f32 and store f32.
        expect_err(
            "bf16 output",
            gemm(&a16, &b16, &c16, GemmBackend::TensorOps),
            BAD_DTYPE,
        );

        // One bf16 operand and one f32 operand is the classic half-migrated
        // call site, and it must not be silently promoted or demoted.
        expect_err(
            "mixed operands",
            gemm(&a16, &b32, &c32, GemmBackend::TensorOps),
            MIXED_OPERANDS,
        );

        // The portable simdgroup kernels have no bf16 variant at all.
        expect_err(
            "bf16 on simdgroup",
            gemm(&a16, &b16, &c32, GemmBackend::Simdgroup),
            MIXED_OPERANDS,
        );

        // TN / NT f32 entry points do not take bf16 in either operand slot,
        // regardless of runtime precision -- `gemm_tn_train` is that door.
        expect_err(
            "TN f32 given bf16",
            gemm_tn_f32(&a16, &b16, &c32, GemmBackend::TensorOps),
            BAD_DTYPE,
        );
        expect_err(
            "NT f32 given bf16",
            gemm_nt_f32(&a16, &b16, &c32, GemmBackend::TensorOps),
            BAD_DTYPE,
        );

        assert!(gemm(&a32, &b32, &c32, GemmBackend::TensorOps).is_ok());
        rt.synchronize().unwrap();
        assert_eq!(
            rt.take_dispatch_count(),
            1,
            "exactly the one accepted GEMM should have encoded"
        );
    });
}

#[test]
fn gemm_rejects_output_aliasing_an_input() {
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let a = rt.alloc_tensor_f32(&[32, 32]).unwrap();
        let b = rt.alloc_tensor_f32(&[32, 32]).unwrap();

        // In-place C = A @ B would read A after the first tile overwrote it.
        expect_err(
            "C is A",
            gemm_f32(&a, &b, &a, GemmBackend::TensorOps),
            OVERLAP,
        );
        expect_err(
            "C is B",
            gemm_f32(&a, &b, &b, GemmBackend::TensorOps),
            OVERLAP,
        );

        // Partial aliasing through views of one allocation is the realistic
        // version: two banks carved from the same buffer that happen to touch.
        let bank = rt.alloc_tensor_f32(&[64, 32]).unwrap();
        let lower = bank.view(&[32, 32], 0);
        let upper = bank.view(&[32, 32], 16 * 32);
        expect_err(
            "overlapping views",
            gemm_f32(&lower, &b, &upper, GemmBackend::TensorOps),
            OVERLAP,
        );

        // Disjoint views of the same allocation must still be accepted, or the
        // overlap check would be rejecting legitimate bank slicing.
        let disjoint = bank.view(&[32, 32], 32 * 32);
        assert!(gemm_f32(&lower, &b, &disjoint, GemmBackend::TensorOps).is_ok());
        rt.synchronize().unwrap();
        assert_eq!(rt.take_dispatch_count(), 1);
    });
}

#[test]
fn tensor_from_buffer_rejects_windows_it_cannot_back() {
    with_gpu(|rt| {
        // `Tensor::from_buffer` is the only way an external crate builds a
        // tensor over storage it allocated itself, so it is the boundary where
        // an unchecked shape would turn into an out-of-bounds kernel write.
        let buf = rt.alloc_buffer(16).unwrap();
        expect_err(
            "shape larger than buffer",
            Tensor::from_buffer(rt, buf.clone(), &[8], DType::F32, 0),
            "tensor view is misaligned or out of bounds",
        );
        expect_err(
            "offset past the end",
            Tensor::from_buffer(rt, buf.clone(), &[2], DType::F32, 12),
            "tensor view is misaligned or out of bounds",
        );
        expect_err(
            "offset not a multiple of the element size",
            Tensor::from_buffer(rt, buf.clone(), &[1], DType::F32, 2),
            "tensor view is misaligned or out of bounds",
        );
        expect_err(
            "element count overflows usize",
            Tensor::from_buffer(rt, buf.clone(), &[usize::MAX, usize::MAX], DType::F32, 0),
            "tensor element count overflow",
        );
        // A window that fits is accepted, so the checks above are rejecting the
        // defect and not the whole entry point.
        assert!(Tensor::from_buffer(rt, buf, &[2], DType::F32, 8).is_ok());
    });
}

#[test]
fn tensor_from_buffer_rejects_storage_from_another_runtime() {
    with_two_gpus(|first, second| {
        // Pairing a buffer with the wrong runtime binds an address the other
        // device's residency set never registered -- the failure mode is a
        // wrong result or a fault, not an allocation error, so it is caught here.
        let foreign = second.alloc_buffer(64).unwrap();
        expect_err(
            "buffer from the other runtime",
            Tensor::from_buffer(first, foreign, &[4], DType::F32, 0),
            "tensor buffer belongs to a different runtime",
        );
    });
}

#[test]
fn gpu_copy_rejects_shape_dtype_runtime_and_overlap() {
    const COPY_MISMATCH: &str = "copy requires equal element counts/dtypes and the same runtime";
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let a = rt.alloc_tensor_f32(&[16]).unwrap();
        let short = rt.alloc_tensor_f32(&[8]).unwrap();
        let half = rt.alloc_tensor_bf16(&[16]).unwrap();
        expect_err("element count", gpu_copy(&a, &short), COPY_MISMATCH);
        expect_err("dtype", gpu_copy(&a, &half), COPY_MISMATCH);
        expect_err(
            "overlapping windows",
            gpu_copy(&a.view(&[8], 0), &a.view(&[8], 4)),
            "copy source and destination overlap",
        );
        assert_eq!(rt.take_dispatch_count(), 0, "rejected copy still encoded");
    });
    with_two_gpus(|first, second| {
        let src = first.alloc_tensor_f32(&[16]).unwrap();
        let dst = second.alloc_tensor_f32(&[16]).unwrap();
        expect_err("cross runtime", gpu_copy(&src, &dst), COPY_MISMATCH);
    });
}

#[test]
fn softcap_rejects_inputs_it_cannot_cap() {
    const SOFTCAP_BAD: &str =
        "softcap requires f32, matching runtime, uint count, and a finite positive cap";
    with_gpu(|rt| {
        rt.take_dispatch_count();
        let t = rt.alloc_tensor_f32(&[16]).unwrap();
        // A non-positive or non-finite cap makes `cap * tanh(x / cap)` undefined
        // or a sign flip, neither of which the kernel guards against itself.
        for cap in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            expect_err(&format!("cap {cap}"), softcap_f32(rt, &t, cap), SOFTCAP_BAD);
        }
        let half = rt.alloc_tensor_bf16(&[16]).unwrap();
        expect_err("bf16 input", softcap_f32(rt, &half, 30.0), SOFTCAP_BAD);
        assert_eq!(
            rt.take_dispatch_count(),
            0,
            "rejected softcap still encoded"
        );
    });
    with_two_gpus(|first, second| {
        let foreign = second.alloc_tensor_f32(&[16]).unwrap();
        expect_err(
            "tensor from another runtime",
            softcap_f32(first, &foreign, 30.0),
            SOFTCAP_BAD,
        );
    });
}

#[test]
fn missing_kernels_and_metallibs_are_named_in_the_error() {
    with_gpu(|rt| {
        // Downstream crates overlay their own metallib and then ask for kernels
        // by name; the failure has to say which name and which file, because
        // the caller's next move depends on whether the build or the lookup broke.
        expect_err(
            "unknown kernel",
            rt.pipeline("definitely_not_a_kernel").map(|_| ()),
            "kernel 'definitely_not_a_kernel' not found in metallib",
        );
        expect_err_starting(
            "missing overlay metallib",
            rt.add_metallib(std::path::Path::new("/nonexistent/overlay.metallib")),
            "metallib missing at /nonexistent/overlay.metallib",
        );
        expect_err_starting(
            "missing metallib at construction",
            GpuRuntime::from_metallib_path(std::path::Path::new("/nonexistent/base.metallib"))
                .map(|_| ()),
            "metallib missing at /nonexistent/base.metallib",
        );
    });
}
