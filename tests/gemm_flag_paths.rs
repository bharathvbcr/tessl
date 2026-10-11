//! The accumulate kernels behind `gemm_{tn,nt}_accum_train`, and the
//! interior-offset branch under the flag that selects it.
//!
//! `gemm_{tn,nt}_accum_train` took a temp-plus-`add_inplace_f32` fallback
//! unless `TESSL_GEMM_ACCUM` / `TESSL_GEMM_ACCUM_DX` were set, while
//! `GemmOperands::{tn_acc, nt_acc}` ran the accumulate kernels with no flag.
//! The flags are retired: one path, pinned here bit for bit against
//! `GemmOperands`. `TESSL_GEMM_INTERIOR` defaults off and latches on first
//! read, so with the default environment the `use_interior` branch of the
//! exact-f32 kernels is dead.
//!
//! The flag is process-global and the environment cannot be mutated safely
//! under a concurrent `getenv`, so each configuration re-runs its own test in
//! a child process with the variables set, filtered to itself and
//! single-threaded. The child checks numbers against an f64 reference and,
//! with `TESSL_KERNEL_TRACE` on, which kernels ran.

mod common;

use common::{
    assert_within_bound, random_f32, rank_one_case, reference, round_trip_bf16, tensor_bf16_at, tensor_f32,
    tensor_f32_at, with_gpu, Layout, Reference, F32_PANEL_SHAPES,
};
use std::sync::Arc;
use tessl::gemm::{gemm_nt_accum_train, gemm_nt_f32, gemm_tn_accum_train, gemm_tn_f32, GemmOperands};
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

/// `expect += c0` for an accumulate: the previous C is part of the answer,
/// and one more term of the sum, so `|c0|` joins the magnitude the error
/// budget scales with. Without it a small-K accumulate onto a large C0 is
/// held to the rounding of `a.b` alone and fails on the final f32 add.
fn with_previous(mut r: Reference, c0: &[f32]) -> Reference {
    for ((want, mag), prev) in r.c.iter_mut().zip(r.mag.iter_mut()).zip(c0) {
        *want += *prev as f64;
        *mag += (*prev as f64).abs();
    }
    r
}

/// One accumulate case: C0 random, `C = C0 + A^T B` (TN) or `C0 + A B^T` (NT).
fn check_accum(rt: &Arc<GpuRuntime>, layout: Layout, bf16: bool, m: usize, n: usize, k: usize) {
    check_accum_at(rt, layout, bf16, (m, n, k), 0);
}

/// [`check_accum`] with A, B and C each viewed `pad_bytes` into its buffer.
fn check_accum_at(
    rt: &Arc<GpuRuntime>,
    layout: Layout,
    bf16: bool,
    (m, n, k): (usize, usize, usize),
    pad_bytes: usize,
) {
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
            tensor_bf16_at(rt, &a_shape, &a_host, pad_bytes),
            tensor_bf16_at(rt, &b_shape, &b_host, pad_bytes),
        )
    } else {
        (
            tensor_f32_at(rt, &a_shape, &a_host, pad_bytes),
            tensor_f32_at(rt, &b_shape, &b_host, pad_bytes),
        )
    };
    let c = tensor_f32_at(rt, &[m, n], &c0, pad_bytes);
    let label = format!(
        "{} accum {layout:?} {m}x{n}x{k} at a {pad_bytes}-byte offset",
        if bf16 { "bf16" } else { "f32" }
    );
    match layout {
        Layout::Tn => gemm_tn_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nt => gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nn => unreachable!("no NN accumulate entry point"),
    }
    .unwrap_or_else(|e| panic!("{label}: {e}"));
    rt.synchronize().unwrap();
    assert_within_bound(&label, &c.read_f32().unwrap(), &expect, k, 0.0);
}

/// An f32 accumulate on a column-panel shape, with rank-one operands so the
/// reference stays O(M·N) (see `common::rank_one_case`).
fn check_accum_panel(rt: &Arc<GpuRuntime>, layout: Layout, m: usize, n: usize, k: usize) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let (a_host, b_host, r) = rank_one_case(layout, m, n, k, 0xacc3 ^ (m * n) as u64);
    let c0 = random_f32(m * n, 0xacc4 ^ (m * n) as u64);
    let expect = with_previous(r, &c0);
    let (a, b) = (tensor_f32(rt, &a_shape, &a_host), tensor_f32(rt, &b_shape, &b_host));
    let c = tensor_f32(rt, &[m, n], &c0);
    match layout {
        Layout::Tn => gemm_tn_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nt => gemm_nt_accum_train(&a, &b, &c, GemmBackend::TensorOps),
        Layout::Nn => unreachable!("no NN accumulate entry point"),
    }
    .unwrap();
    rt.synchronize().unwrap();
    assert_within_bound(
        &format!("f32 accum {layout:?} panel walk {m}x{n}x{k}"),
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

/// `gemm_*_accum_train` and the `GemmOperands` accumulate on one `C0`, bit
/// for bit: the same kernels, whatever precision the runtime is in.
fn check_same_as_operands(rt: &Arc<GpuRuntime>, layout: Layout, bf16: bool, (m, n, k): (usize, usize, usize)) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let a = tensor_f32(rt, &a_shape, &random_f32(m * k, 0x0e1 ^ (m * 13 + k) as u64));
    let b = tensor_f32(rt, &b_shape, &random_f32(k * n, 0x0e2 ^ (n * 11 + k) as u64));
    let c0 = random_f32(m * n, 0x0e3 ^ (m * n) as u64);
    let (train, ops) = (tensor_f32(rt, &[m, n], &c0), tensor_f32(rt, &[m, n], &c0));
    let lane = if bf16 {
        GemmOperands::Bf16
    } else {
        GemmOperands::ExactF32
    };
    rt.set_precision(if bf16 { PrecisionMode::Bf16 } else { PrecisionMode::F32 });
    match layout {
        Layout::Tn => {
            gemm_tn_accum_train(&a, &b, &train, GemmBackend::TensorOps).unwrap();
            lane.tn_acc(&a, &b, &ops).unwrap();
        }
        Layout::Nt => {
            gemm_nt_accum_train(&a, &b, &train, GemmBackend::TensorOps).unwrap();
            lane.nt_acc(&a, &b, &ops).unwrap();
        }
        Layout::Nn => unreachable!("no NN accumulate train entry point"),
    }
    rt.synchronize().unwrap();
    let (t, o) = (train.buffer.read_f32(), ops.buffer.read_f32());
    if let Some(i) = (0..m * n).find(|&i| t[i].to_bits() != o[i].to_bits()) {
        panic!(
            "{layout:?} bf16={bf16} {m}x{n}x{k}: accum_train and GemmOperands differ at {i}: {} vs {}",
            t[i], o[i]
        );
    }
}

/// The bf16 and f32 accumulate kernels, TN and NT, on full-tile, ragged,
/// K = 3, split-K-gated and column-panel shapes, with the previous C folded
/// in. The retired variables are set to `0`, the value that used to select
/// the fallback, and change nothing.
#[test]
fn accumulate_train_entries_run_the_accumulate_kernels() {
    in_child(
        "accumulate_train_entries_run_the_accumulate_kernels",
        &[
            ("TESSL_GEMM_ACCUM", "0"),
            ("TESSL_GEMM_ACCUM_DX", "0"),
            ("METAL_NATIVE_GEMM_ACCUM", "0"),
            ("METAL_NATIVE_GEMM_ACCUM_DX", "0"),
        ],
        || {
            with_gpu(|rt| {
                if !rt.has_tensorops() {
                    return;
                }
                // K = 3 keeps a·b small against C0, so the final f32 add's
                // rounding has to be in the budget (`with_previous`).
                // (128, 384, 4096) takes the split-K TN lanes.
                for &(m, n, k) in &[
                    (128usize, 128usize, 256usize),
                    (96, 48, 64),
                    (200, 72, 96),
                    (96, 48, 3),
                    (128, 384, 4096),
                ] {
                    rt.set_precision(PrecisionMode::F32);
                    check_accum(rt, Layout::Tn, false, m, n, k);
                    check_accum(rt, Layout::Nt, false, m, n, k);
                    rt.set_precision(PrecisionMode::Bf16);
                    check_accum(rt, Layout::Tn, true, m, n, k);
                    check_accum(rt, Layout::Nt, true, m, n, k);
                    for layout in [Layout::Tn, Layout::Nt] {
                        for bf16 in [false, true] {
                            check_same_as_operands(rt, layout, bf16, (m, n, k));
                        }
                    }
                }
                rt.set_precision(PrecisionMode::F32);
                for &(m, n, k) in F32_PANEL_SHAPES {
                    check_accum_panel(rt, Layout::Tn, m, n, k);
                    check_accum_panel(rt, Layout::Nt, m, n, k);
                }
                for kernel in [
                    "matmul2d_tensorops_tn_accum_f32",
                    "matmul2d_tensorops_nt_accum_f32",
                    "matmul2d_tensorops_tn_accum_bf16_f32",
                    "matmul2d_tensorops_nt_accum_bf16_f32",
                    "matmul2d_tensorops_tn_splitk_f32",
                    "matmul2d_tensorops_tn_splitk_bf16_f32",
                ] {
                    assert_traced(kernel);
                }
                assert_not_traced("add_inplace_f32");
            });
        },
    );
}

/// Accumulate views 16, 32 and 48 bytes into their buffers.
/// The exact-f32 accumulate kernels are pointer-tensor kernels like plain f32,
/// which takes these views; they used to be refused under the cooperative
/// path's 64-byte rule, so `gemm_tn_f32` accepted a view `gemm_tn_accum_train`
/// rejected. One 16-byte rule now covers every family (see
/// `bench/results/gemm_align_probe_m5pro.txt`), bf16 accumulate included.
#[test]
fn accumulate_kernels_take_views_on_any_16_byte_boundary() {
    in_child("accumulate_kernels_take_views_on_any_16_byte_boundary", &[], || {
        with_gpu(|rt| {
            if !rt.has_tensorops() {
                return;
            }
            for pad in [16, 32, 48] {
                for shape in [(96usize, 48usize, 64usize), (130, 97, 70)] {
                    rt.set_precision(PrecisionMode::F32);
                    check_accum_at(rt, Layout::Tn, false, shape, pad);
                    check_accum_at(rt, Layout::Nt, false, shape, pad);
                    rt.set_precision(PrecisionMode::Bf16);
                    check_accum_at(rt, Layout::Tn, true, shape, pad);
                    check_accum_at(rt, Layout::Nt, true, shape, pad);
                }
            }
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
    });
}

/// `TESSL_GEMM_INTERIOR=1` (the f32 accumulate kernels take their interior
/// branch too): every exact-f32 kernel on a shape whose tiles are all
/// interior and on a ragged one where only some are.
#[test]
fn exact_f32_kernels_agree_with_the_reference_on_the_interior_branch() {
    in_child(
        "exact_f32_kernels_agree_with_the_reference_on_the_interior_branch",
        &[("TESSL_GEMM_INTERIOR", "1")],
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
                // The interior branch under the column-panel walk.
                for &(m, n, k) in F32_PANEL_SHAPES {
                    for layout in [Layout::Nn, Layout::Tn, Layout::Nt] {
                        let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
                        let (a_host, b_host, expect) = rank_one_case(layout, m, n, k, 0x3ed ^ (m + n) as u64);
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
                            &format!("interior f32 {layout:?} panel walk {m}x{n}x{k}"),
                            &c.buffer.read_f32(),
                            &expect,
                            k,
                            0.0,
                        );
                    }
                    check_accum_panel(rt, Layout::Tn, m, n, k);
                    check_accum_panel(rt, Layout::Nt, m, n, k);
                }
                assert_traced("matmul2d_tensorops_f32");
                assert_traced("matmul2d_tensorops_tn_accum_f32");
                assert_traced("matmul2d_tensorops_nt_accum_f32");
            });
        },
    );
}
