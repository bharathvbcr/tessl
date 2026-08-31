//! GEMM parity against a CPU reference, across every layout / precision lane a
//! downstream crate can reach through the public API.
//!
//! The unit tests in `src/gemm.rs` check a handful of square shapes against an
//! f32 CPU loop with a hand-picked `1e-4`. These check the same kernels the way
//! `gemma-metal` reaches them — through `tessl::gemm::*` with no crate-private
//! help — against an f64 reference and a per-element bound derived from the
//! accumulator width (see `common::tolerance`), so the tolerance tightens with
//! K instead of being one constant that is too loose for K=32 and too tight
//! for K=2048.

mod common;

use common::{
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, with_gpu,
    Layout, U_BF16,
};
use tessl::gemm::{gemm_nt_f32, gemm_nt_train, gemm_tn_f32, gemm_tn_train};
use tessl::{gemm, gemm_f32, GemmBackend, GpuRuntime, PrecisionMode};

/// Operand extents for each layout, given the logical (M, N, K).
fn operand_shapes(layout: Layout, m: usize, n: usize, k: usize) -> ([usize; 2], [usize; 2]) {
    match layout {
        Layout::Nn => ([m, k], [k, n]),
        Layout::Tn => ([k, m], [k, n]),
        Layout::Nt => ([m, k], [n, k]),
    }
}

/// One f32 case end to end: upload, dispatch, read back, compare.
fn check_f32(rt: &std::sync::Arc<GpuRuntime>, layout: Layout, m: usize, n: usize, k: usize) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let a_host = random_f32(m * k, 0x51ed ^ (m * 31 + k) as u64);
    let b_host = random_f32(k * n, 0xb0b0 ^ (n * 17 + k) as u64);
    let expect = reference(layout, &a_host, &b_host, m, n, k);

    let a = tensor_f32(rt, &a_shape, &a_host);
    let b = tensor_f32(rt, &b_shape, &b_host);
    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
    match layout {
        Layout::Nn => gemm_f32(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Tn => gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nt => gemm_nt_f32(&a, &b, &c, GemmBackend::TensorOps),
    }
    .unwrap();
    rt.synchronize().unwrap();

    assert_within_bound(
        &format!("f32 {layout:?} {m}x{k}@{k}x{n}"),
        &c.buffer.read_f32(),
        &expect,
        k,
        // Operands are already f32; the kernel narrows nothing.
        0.0,
    );
}

/// One bf16 case. Operands are rounded to bf16 on the host first so the
/// reference and the GPU consume identical values and the only error left to
/// bound is the f32 accumulation.
fn check_bf16(rt: &std::sync::Arc<GpuRuntime>, layout: Layout, m: usize, n: usize, k: usize) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let a_host = round_trip_bf16(&random_f32(m * k, 0x2f11 ^ (m * 13 + k) as u64));
    let b_host = round_trip_bf16(&random_f32(k * n, 0x77aa ^ (n * 29 + k) as u64));
    let expect = reference(layout, &a_host, &b_host, m, n, k);

    let a = tensor_bf16(rt, &a_shape, &a_host);
    let b = tensor_bf16(rt, &b_shape, &b_host);
    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
    match layout {
        Layout::Nn => gemm(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Tn => gemm_tn_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nt => gemm_nt_train(&a, &b, &c, GemmBackend::TensorOps),
    }
    .unwrap();
    rt.synchronize().unwrap();

    assert_within_bound(
        &format!("bf16 {layout:?} {m}x{k}@{k}x{n}"),
        &c.buffer.read_f32(),
        &expect,
        k,
        // Zero: the host already rounded, so bf16 x bf16 -> f32 is exact here.
        0.0,
    );
}

#[test]
fn nn_f32_matches_cpu_reference() {
    with_gpu(|rt| {
        // 32x32 is exactly one TILE_F32; 96x48 and 130x257 sit inside and
        // across it, which is where a wrong grid or a missing bounds check on
        // the trailing tile shows up.
        for &(m, n, k) in &[(32, 32, 32), (96, 48, 64), (130, 257, 96)] {
            check_f32(rt, Layout::Nn, m, n, k);
        }
    });
}

#[test]
fn tn_f32_matches_cpu_reference() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(32, 32, 32), (64, 96, 48), (130, 257, 96)] {
            check_f32(rt, Layout::Tn, m, n, k);
        }
    });
}

#[test]
fn nt_f32_matches_cpu_reference() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(32, 32, 32), (64, 96, 48), (130, 257, 96)] {
            check_f32(rt, Layout::Nt, m, n, k);
        }
    });
}

#[test]
fn nn_bf16_matches_cpu_reference() {
    with_gpu(|rt| {
        // N=512 and N=520 straddle the `nn_coop_kernel` switch: at or below 512
        // the 64x64 narrow tile is selected, above it the 128x64 default. Both
        // sides of that decision have to be right, and only N moves it.
        for &(m, n, k) in &[(64, 64, 128), (96, 512, 64), (96, 520, 64), (129, 520, 130)] {
            check_bf16(rt, Layout::Nn, m, n, k);
        }
    });
}

#[test]
fn tn_bf16_matches_cpu_reference() {
    with_gpu(|rt| {
        // The bf16 TN/NT descriptor kernels only engage under PrecisionMode::Bf16;
        // in F32 mode `gemm_tn_train` would quietly route to the f32 path and
        // this test would be checking a kernel it does not name.
        rt.set_precision(PrecisionMode::Bf16);
        assert_eq!(rt.precision(), PrecisionMode::Bf16);
        for &(m, n, k) in &[(64, 64, 128), (130, 200, 96)] {
            check_bf16(rt, Layout::Tn, m, n, k);
        }
    });
}

#[test]
fn nt_bf16_matches_cpu_reference() {
    with_gpu(|rt| {
        rt.set_precision(PrecisionMode::Bf16);
        for &(m, n, k) in &[(64, 64, 128), (130, 200, 96)] {
            check_bf16(rt, Layout::Nt, m, n, k);
        }
    });
}

#[test]
fn tn_splitk_matches_cpu_reference() {
    with_gpu(|rt| {
        // `prefer_tn_splitk` fires at K >= 2048 with M, N <= 384 and min <= 128.
        // That lane reduces partial sums in a second pass, so it is the one
        // shape class where a dropped or double-counted partial is possible;
        // nothing else in this file reaches it.
        check_f32(rt, Layout::Tn, 128, 128, 2048);

        rt.set_precision(PrecisionMode::Bf16);
        check_bf16(rt, Layout::Tn, 128, 128, 2048);
    });
}

#[test]
fn simdgroup_backend_matches_cpu_reference() {
    with_gpu(|rt| {
        // The portable fallback is what a machine without TensorOps runs, and
        // it dispatches a different kernel once any extent is unaligned
        // (matmul_simdgroup_edges_f32) -- so both sides of that split are here.
        for &(m, n, k) in &[(32, 32, 32), (17, 19, 13)] {
            let a_host = random_f32(m * k, 0xabc ^ m as u64);
            let b_host = random_f32(k * n, 0xdef ^ n as u64);
            let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
            let a = tensor_f32(rt, &[m, k], &a_host);
            let b = tensor_f32(rt, &[k, n], &b_host);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_f32(&a, &b, &c, GemmBackend::Simdgroup).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("simdgroup {m}x{k}@{k}x{n}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                0.0,
            );
        }
    });
}

#[test]
fn relaxed_precision_stays_within_a_bf16_class_bound() {
    with_gpu(|rt| {
        // `set_relaxed_precision` swaps NN onto the tf32-class kernels while the
        // runtime still reports PrecisionMode::F32 and callers still pass f32
        // buffers. That is a silent accuracy change, so it gets an explicit
        // bound: tf32 keeps 11 significand bits, and U_BF16 (2^-8) is a safe
        // over-estimate of that narrowing regardless of the exact format the
        // hardware uses internally.
        rt.set_relaxed_precision(true);
        assert!(rt.relaxed_precision());
        assert_eq!(rt.precision(), PrecisionMode::F32);

        for &(m, n, k) in &[(96, 512, 64), (96, 520, 64)] {
            let a_host = random_f32(m * k, 0x9911 ^ n as u64);
            let b_host = random_f32(k * n, 0x1199 ^ n as u64);
            let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
            let a = tensor_f32(rt, &[m, k], &a_host);
            let b = tensor_f32(rt, &[k, n], &b_host);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("relaxed {m}x{k}@{k}x{n}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                U_BF16,
            );
        }
    });
}

#[test]
fn bf16_gemm_beats_the_relaxed_bound_it_is_allowed() {
    with_gpu(|rt| {
        // Guards the claim the tolerance derivation rests on: the bf16 kernels
        // accumulate in f32, not in bf16. If they ever accumulated narrow, the
        // zero operand-width term in `check_bf16` would be wrong, and the cheap
        // way to notice is that the result would need the *bf16-wide* budget it
        // is denied here.
        let (m, n, k) = (64, 64, 512);
        let a_host = round_trip_bf16(&random_f32(m * k, 7));
        let b_host = round_trip_bf16(&random_f32(k * n, 11));
        let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
        let a = tensor_bf16(rt, &[m, k], &a_host);
        let b = tensor_bf16(rt, &[k, n], &b_host);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        assert_within_bound("bf16 f32-accumulate", &c.buffer.read_f32(), &expect, k, 0.0);
    });
}
