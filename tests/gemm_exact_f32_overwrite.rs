//! The exact-f32 `mode::multiply` kernels overwrite every element of C.
//!
//! `matmul2d_tensorops_f32` (NN), `_tn_f32` and `_nt_f32` are dispatched with no
//! zero pass over C, so the kernel itself has to write every in-bounds element:
//! interior tiles, edge tiles, and any K, however short. This test calls each
//! kernel directly, without the library's dispatch wrappers, over a C pre-filled
//! with NaN, with large finite sentinels, and with zeros, and requires:
//!
//! - the three runs to be bit-identical (so nothing of C's prior contents leaks
//!   into the result, not even as `x + 0` or `NaN * 0`),
//! - the result to equal the exact integer product (the operands are small
//!   integers, so every partial sum is exact in f32 and any reduction order gives
//!   the same bits), and
//! - a guard region past the end of C to be untouched.
//!
//! The second test pins what that buys: the public exact-f32 NN, TN and NT GEMMs
//! are one dispatch each (they were a `zero_f32` and a matmul), and still give
//! the exact product into a C that held NaN.

mod common;

use common::{tensor_f32, with_gpu};
use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;
use tessl::gemm::{gemm_f32, gemm_nt_f32, gemm_tn_f32, GemmBackend};
use tessl::infer_trace;
use tessl::runtime::mtl_size;
use tessl::{GpuRuntime, Tensor};

const TILE: usize = 32;
/// A public exact-f32 GEMM entry: A, B, C, backend.
type PublicGemm = fn(&Tensor, &Tensor, &Tensor, GemmBackend) -> Result<(), String>;

/// Elements past C's last element that the kernel must not write.
const GUARD: usize = 64;
const GUARD_SENTINEL: f32 = -7.5e30;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Nn,
    Tn,
    Nt,
}

impl Layout {
    fn kernel(self) -> &'static str {
        match self {
            Layout::Nn => "matmul2d_tensorops_f32",
            Layout::Tn => "matmul2d_tensorops_tn_f32",
            Layout::Nt => "matmul2d_tensorops_nt_f32",
        }
    }
}

/// Small-integer operands: every product and partial sum is an integer far below
/// 2^24, so the f32 result is exact whatever the order.
fn operands(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k).map(|i| ((i * 7 + 3) % 7) as f32 - 3.0).collect();
    let b = (0..k * n).map(|i| ((i * 5 + 1) % 5) as f32 - 2.0).collect();
    (a, b)
}

fn exact_product(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = src[r * cols + c];
        }
    }
    t
}

/// One dispatch of `layout`'s multiply kernel over a C pre-filled with `fill`.
/// Returns C and the guard region after it.
fn run(
    rt: &Arc<GpuRuntime>,
    layout: Layout,
    (m, n, k): (usize, usize, usize),
    (a, b): (&[f32], &[f32]),
    fill: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (a_t, b_t): (Tensor, Tensor) = match layout {
        Layout::Nn => (tensor_f32(rt, &[m, k], a), tensor_f32(rt, &[k, n], b)),
        Layout::Tn => (tensor_f32(rt, &[k, m], &transpose(a, m, k)), tensor_f32(rt, &[k, n], b)),
        Layout::Nt => (tensor_f32(rt, &[m, k], a), tensor_f32(rt, &[n, k], &transpose(b, k, n))),
    };
    let mut host = vec![fill; m * n];
    host.extend(std::iter::repeat_n(GUARD_SENTINEL, GUARD));
    let bank = tensor_f32(rt, &[m * n + GUARD], &host);
    let c = bank.view(&[m, n], 0);

    let pipeline = rt.pipeline(layout.kernel()).expect("exact-f32 multiply pipeline");
    let tiles_n = n.div_ceil(TILE);
    let tiles_m = m.div_ceil(TILE);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipeline);
        bnd.bind_tensor(&a_t, 0);
        bnd.bind_tensor(&b_t, 1);
        bnd.bind_tensor(&c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.dispatch(
            mtl_size(tiles_n * tiles_m, 1, 1),
            mtl_size(pipeline.threadExecutionWidth(), 1, 1),
        );
        Ok(())
    })
    .expect("dispatch");
    rt.synchronize().expect("synchronize");
    let all = bank.buffer.read_f32();
    (all[..m * n].to_vec(), all[m * n..m * n + GUARD].to_vec())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn exact_f32_multiply_overwrites_every_element_of_c() {
    let shapes: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 1, 4096),
        (1, 33, 7),
        (33, 1, 7),
        (1, 4096, 64),
        (4096, 1, 64),
        (7, 7, 1),
        (100, 50, 1),
        (5, 3, 31),
        (31, 33, 5),
        (32, 32, 32),
        (33, 65, 31),
        (64, 64, 64),
        (65, 97, 129),
        (128, 96, 32),
        (257, 129, 300),
        (48, 48, 4097),
        (512, 512, 512),
        // B holds K * N = 8_392_704 >= 2^23 elements, so tile_walk takes column panels.
        (33, 2049, 4096),
    ];
    let fills = [("nan", f32::NAN), ("+big", 3.0e38), ("-big", -3.0e38), ("zero", 0.0)];
    with_gpu(|rt| {
        for layout in [Layout::Nn, Layout::Tn, Layout::Nt] {
            for &(m, n, k) in shapes {
                let (a, b) = operands(m, n, k);
                let want = exact_product(&a, &b, m, n, k);
                let mut runs = Vec::new();
                for (name, fill) in fills {
                    let (c, guard) = run(rt, layout, (m, n, k), (&a, &b), fill);
                    let case = format!("{layout:?} {m}x{n}x{k} fill {name}");
                    assert!(
                        guard.iter().all(|g| g.to_bits() == GUARD_SENTINEL.to_bits()),
                        "{case}: the kernel wrote past the end of C"
                    );
                    if let Some(i) = c.iter().position(|v| !v.is_finite() || v.abs() > 1.0e30) {
                        panic!("{case}: C[{i}] = {} was never overwritten", c[i]);
                    }
                    runs.push((name, c));
                }
                let zeroed = runs.last().expect("a zero-filled run").1.clone();
                assert_eq!(
                    bits(&zeroed),
                    bits(&want),
                    "{layout:?} {m}x{n}x{k}: zero-filled run is not the exact product"
                );
                for (name, c) in &runs {
                    assert_eq!(
                        bits(c),
                        bits(&zeroed),
                        "{layout:?} {m}x{n}x{k}: fill {name} differs from the zero-filled run"
                    );
                }
            }
        }
    });
}

#[test]
fn public_exact_f32_gemms_are_one_dispatch_into_a_nan_filled_c() {
    let (m, n, k) = (70usize, 45usize, 37usize);
    let (a, b) = operands(m, n, k);
    let want = exact_product(&a, &b, m, n, k);
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let nan_c = || tensor_f32(rt, &[m, n], &vec![f32::NAN; m * n]);
        let cases: [(&str, Tensor, Tensor, PublicGemm); 3] = [
            ("NN", tensor_f32(rt, &[m, k], &a), tensor_f32(rt, &[k, n], &b), gemm_f32),
            (
                "TN",
                tensor_f32(rt, &[k, m], &transpose(&a, m, k)),
                tensor_f32(rt, &[k, n], &b),
                gemm_tn_f32,
            ),
            (
                "NT",
                tensor_f32(rt, &[m, k], &a),
                tensor_f32(rt, &[n, k], &transpose(&b, k, n)),
                gemm_nt_f32,
            ),
        ];
        for (name, a_t, b_t, gemm) in cases {
            let c = nan_c();
            infer_trace::set_enabled(true);
            let before = infer_trace::snapshot();
            gemm(&a_t, &b_t, &c, GemmBackend::TensorOps).expect(name);
            let dispatches = infer_trace::snapshot().since(&before).dispatches;
            infer_trace::set_enabled(false);
            rt.synchronize().expect("synchronize");
            assert_eq!(dispatches, 1, "{name}: exact-f32 GEMM dispatches");
            assert_eq!(bits(&c.buffer.read_f32()), bits(&want), "{name}: product");
        }
    });
}
