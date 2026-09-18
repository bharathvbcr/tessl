//! The accumulate kernels and the interior-offset branch, under the flags
//! that select them.
//!
//! `TESSL_GEMM_ACCUM`, `TESSL_GEMM_ACCUM_DX` and `TESSL_GEMM_INTERIOR` default
//! off and latch on first read, so with the default environment the four
//! shipped `*_accum_*` kernels never run (`gemm_*_accum_train` takes the
//! temp-plus-`add_inplace_f32` fallback) and the `use_interior` branch of the
//! exact-f32 kernels is dead. Nothing in the suite set them (audit G7): the
//! only place that did was a benchmark script asserting the kernels were
//! *dispatched*, which checks no output value.
//!
//! The flags are process-global and the environment cannot be mutated safely
//! under a concurrent `getenv`, so each configuration re-runs its own test in
//! a child process with the variables set, filtered to itself and
//! single-threaded. The child checks numbers against an f64 reference and,
//! with `TESSL_KERNEL_TRACE` on, that the accumulate kernels themselves ran
//! and the fallback did not.

mod common;

use common::{
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, with_gpu,
    Layout, Reference,
};
use std::sync::Arc;
use tessl::gemm::{gemm_nt_accum_train, gemm_nt_f32, gemm_tn_accum_train, gemm_tn_f32};
use tessl::runtime::traced_kernels;
use tessl::{gemm_f32, GemmBackend, GpuRuntime, PrecisionMode};

const CHILD_ENV: &str = "TESSL_GEMM_FLAG_PATHS_CHILD";

/// Re-run `test_name` in a child process with `vars` set; the child body is
/// `body`. Returns in the parent after asserting the child passed.
fn in_child(test_name: &str, vars: &[(&str, &str)], body: impl FnOnce()) {
    if std::env::var_os(CHILD_ENV).is_some() {
        body();
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--exact", "--test-threads=1", test_name])
        .env(CHILD_ENV, "1")
        .env("TESSL_KERNEL_TRACE", "1");
    for (k, v) in vars {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn isolated child test");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "isolated child {test_name} failed:\n{stdout}\n{stderr}"
    );
    // A filter that matched nothing also exits 0 — make that loud.
    assert!(
        stdout.contains("1 passed"),
        "child ran no test (filter out of sync with {test_name}?):\n{stdout}"
    );
}

fn operand_shapes(layout: Layout, m: usize, n: usize, k: usize) -> ([usize; 2], [usize; 2]) {
    match layout {
        Layout::Nn => ([m, k], [k, n]),
        Layout::Tn => ([k, m], [k, n]),
        Layout::Nt => ([m, k], [n, k]),
    }
}

/// `expect += c0` for an accumulate: the previous C is part of the answer.
fn with_previous(mut r: Reference, c0: &[f32]) -> Reference {
    for (want, prev) in r.c.iter_mut().zip(c0) {
        *want += *prev as f64;
    }
    r
}

/// One accumulate case: C0 random, `C = C0 + A^T B` (TN) or `C0 + A B^T` (NT).
fn check_accum(rt: &Arc<GpuRuntime>, layout: Layout, bf16: bool, m: usize, n: usize, k: usize) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let mut a_host = random_f32(m * k, 0xacc0 ^ (m * 31 + k) as u64);
    let mut b_host = random_f32(k * n, 0xacc1 ^ (n * 17 + k) as u64);
    if bf16 {
        a_host = round_trip_bf16(&a_host);
        b_host = round_trip_bf16(&b_host);
    }
    let c0 = random_f32(m * n, 0xacc2 ^ (m * n) as u64);
    let expect = with_previous(reference(layout, &a_host, &b_host, m, n, k), &c0);

    let (a, b) = if bf16 {
        (
            tensor_bf16(rt, &a_shape, &a_host),
            tensor_bf16(rt, &b_shape, &b_host),
        )
    } else {
        (
            tensor_f32(rt, &a_shape, &a_host),
            tensor_f32(rt, &b_shape, &b_host),
        )
    };
    let c = tensor_f32(rt, &[m, n], &c0);
    match layout {
        Layout::Tn => gemm_tn_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nt => gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nn => unreachable!("no NN accumulate entry point"),
    }
    .unwrap();
    rt.synchronize().unwrap();
    assert_within_bound(
        &format!(
            "{} accum {layout:?} {m}x{n}x{k}",
            if bf16 { "bf16" } else { "f32" }
        ),
        &c.buffer.read_f32(),
        &expect,
        k,
        0.0,
    );
}

fn assert_traced(kernel: &str) {
    let traced = traced_kernels();
    assert!(
        traced.iter().any(|name| name == kernel),
        "{kernel} never ran; traced: {traced:?}"
    );
}

fn assert_not_traced(kernel: &str) {
    let traced = traced_kernels();
    assert!(
        !traced.iter().any(|name| name == kernel),
        "{kernel} ran, so the flagged path fell back; traced: {traced:?}"
    );
}

/// `TESSL_GEMM_ACCUM=1`: the bf16 and f32 accumulate kernels, TN and NT,
/// on full-tile and ragged shapes, with the previous C folded in.
#[test]
fn accumulate_kernels_add_into_c_under_the_accum_flag() {
    in_child(
        "accumulate_kernels_add_into_c_under_the_accum_flag",
        &[("TESSL_GEMM_ACCUM", "1")],
        || {
            with_gpu(|rt| {
                if !rt.has_tensorops() {
                    return;
                }
                for &(m, n, k) in &[(128usize, 128usize, 256usize), (96, 48, 64), (200, 72, 96)] {
                    rt.set_precision(PrecisionMode::F32);
                    check_accum(rt, Layout::Tn, false, m, n, k);
                    check_accum(rt, Layout::Nt, false, m, n, k);
                    rt.set_precision(PrecisionMode::Bf16);
                    check_accum(rt, Layout::Tn, true, m, n, k);
                    check_accum(rt, Layout::Nt, true, m, n, k);
                }
                rt.set_precision(PrecisionMode::F32);
                for kernel in [
                    "matmul2d_tensorops_tn_accum_f32",
                    "matmul2d_tensorops_nt_accum_f32",
                    "matmul2d_tensorops_tn_accum_bf16_f32",
                    "matmul2d_tensorops_nt_accum_bf16_f32",
                ] {
                    assert_traced(kernel);
                }
                assert_not_traced("add_inplace_f32");
            });
        },
    );
}

/// `TESSL_GEMM_ACCUM_DX=1` alone: only the NT (dX-class) lane accumulates in
/// place; TN still takes the fallback.
#[test]
fn only_the_nt_lane_accumulates_under_the_dx_flag() {
    in_child(
        "only_the_nt_lane_accumulates_under_the_dx_flag",
        &[("TESSL_GEMM_ACCUM_DX", "1")],
        || {
            with_gpu(|rt| {
                if !rt.has_tensorops() {
                    return;
                }
                for &(m, n, k) in &[(128usize, 128usize, 256usize), (96, 48, 64)] {
                    rt.set_precision(PrecisionMode::F32);
                    check_accum(rt, Layout::Nt, false, m, n, k);
                    check_accum(rt, Layout::Tn, false, m, n, k);
                    rt.set_precision(PrecisionMode::Bf16);
                    check_accum(rt, Layout::Nt, true, m, n, k);
                    check_accum(rt, Layout::Tn, true, m, n, k);
                }
                rt.set_precision(PrecisionMode::F32);
                assert_traced("matmul2d_tensorops_nt_accum_f32");
                assert_traced("matmul2d_tensorops_nt_accum_bf16_f32");
                assert_not_traced("matmul2d_tensorops_tn_accum_f32");
                assert_not_traced("matmul2d_tensorops_tn_accum_bf16_f32");
                // TN fell back to temp + add, as documented.
                assert_traced("add_inplace_f32");
            });
        },
    );
}

/// `TESSL_GEMM_INTERIOR=1` (with `TESSL_GEMM_ACCUM=1` so the f32 accumulate
/// kernels also take their interior branch): every exact-f32 kernel on a
/// shape whose tiles are all interior and on a ragged one where only some are.
#[test]
fn exact_f32_kernels_agree_with_the_reference_on_the_interior_branch() {
    in_child(
        "exact_f32_kernels_agree_with_the_reference_on_the_interior_branch",
        &[("TESSL_GEMM_INTERIOR", "1"), ("TESSL_GEMM_ACCUM", "1")],
        || {
            with_gpu(|rt| {
                if !rt.has_tensorops() {
                    return;
                }
                rt.set_precision(PrecisionMode::F32);
                // 32x32 tiles: 256x256 is all interior, 96x48 is partly edge.
                for &(m, n, k) in &[(256usize, 256usize, 128usize), (96, 48, 64), (64, 160, 32)] {
                    for layout in [Layout::Nn, Layout::Tn, Layout::Nt] {
                        let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
                        let a_host = random_f32(m * k, 0x1ed ^ (m * 7 + k) as u64);
                        let b_host = random_f32(k * n, 0x2ed ^ (n * 5 + k) as u64);
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
                            &format!("interior f32 {layout:?} {m}x{n}x{k}"),
                            &c.buffer.read_f32(),
                            &expect,
                            k,
                            0.0,
                        );
                    }
                    check_accum(rt, Layout::Tn, false, m, n, k);
                    check_accum(rt, Layout::Nt, false, m, n, k);
                }
                assert_traced("matmul2d_tensorops_f32");
                assert_traced("matmul2d_tensorops_tn_accum_f32");
                assert_traced("matmul2d_tensorops_nt_accum_f32");
            });
        },
    );
}
