//! `GemmOperands::{nn_acc, tn_acc, nt_acc}`: `C += op(A) op(B)` on both
//! operand lanes, against an f64 reference with the previous C folded in.
//!
//! Exact f32 accumulates the products into C as it goes; bf16 operands add
//! the f32 product to C once. Either way the error is within the recursive
//! summation bound over the K products and C0 (`common::assert_within_bound`
//! with `|c0|` in the magnitude). Shapes cover full and ragged tiles, a
//! 3-deep K (so C0 dominates and the final add's rounding is in the budget),
//! views at a 16-byte offset, the TN and NN split-K lanes, and the
//! column-panel walk (rank-one operands keep that reference O(M·N)).

mod common;

use std::sync::Arc;

use common::{
    assert_within_bound, random_f32, rank_one_case, reference, round_trip_bf16, tensor_bf16_at, tensor_f32,
    tensor_f32_at, with_gpu, Layout, Reference, F32_PANEL_SHAPES,
};
use tessl::gemm::GemmOperands;
use tessl::{GpuRuntime, Tensor};

fn operand_shapes(layout: Layout, m: usize, n: usize, k: usize) -> ([usize; 2], [usize; 2]) {
    match layout {
        Layout::Nn => ([m, k], [k, n]),
        Layout::Tn => ([k, m], [k, n]),
        Layout::Nt => ([m, k], [n, k]),
    }
}

fn with_previous(mut r: Reference, c0: &[f32]) -> Reference {
    for ((want, mag), prev) in r.c.iter_mut().zip(r.mag.iter_mut()).zip(c0) {
        *want += f64::from(*prev);
        *mag += f64::from(*prev).abs();
    }
    r
}

fn run(op: GemmOperands, layout: Layout, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), String> {
    match layout {
        Layout::Nn => op.nn_acc(a, b, c),
        Layout::Tn => op.tn_acc(a, b, c),
        Layout::Nt => op.nt_acc(a, b, c),
    }
}

/// One case: random operands (bf16-representable on the bf16 lane, stored
/// as bf16 when `bf16_storage`), C0 random, at `pad_bytes` into each buffer.
fn check(
    rt: &Arc<GpuRuntime>,
    op: GemmOperands,
    layout: Layout,
    (m, n, k): (usize, usize, usize),
    pad_bytes: usize,
    bf16_storage: bool,
) {
    let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
    let mut a_host = random_f32(m * k, 0x5acc ^ (m * 31 + k) as u64);
    let mut b_host = random_f32(k * n, 0x6acc ^ (n * 17 + k) as u64);
    if op == GemmOperands::Bf16 {
        a_host = round_trip_bf16(&a_host);
        b_host = round_trip_bf16(&b_host);
    }
    let c0 = random_f32(m * n, 0x7acc ^ (m * n) as u64);
    let expect = with_previous(reference(layout, &a_host, &b_host, m, n, k), &c0);
    let (a, b) = if bf16_storage {
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
    let label = format!("{op:?} {layout:?}_acc {m}x{n}x{k} at +{pad_bytes} B (bf16 storage {bf16_storage})");
    run(op, layout, &a, &b, &c).unwrap_or_else(|e| panic!("{label}: {e}"));
    rt.synchronize().unwrap();
    assert_within_bound(&label, &c.read_f32().unwrap(), &expect, k, 0.0);
}

#[test]
fn accumulate_operands_add_the_product_into_c() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "tessl requires TensorOps");
        let shapes = [
            (128usize, 128usize, 256usize),
            (96, 48, 64),
            (200, 72, 96),
            (96, 48, 3),
            // TN's sequential split-K (K >= 2048, M and N <= 384, one <= 128).
            (96, 200, 2304),
        ];
        for layout in [Layout::Nn, Layout::Tn, Layout::Nt] {
            for &dims in &shapes {
                check(rt, GemmOperands::ExactF32, layout, dims, 0, false);
                check(rt, GemmOperands::Bf16, layout, dims, 0, false);
            }
            check(rt, GemmOperands::ExactF32, layout, (200, 72, 96), 16, false);
            check(rt, GemmOperands::Bf16, layout, (200, 72, 96), 16, true);
        }
    });
}

/// The NN split-K lane (`K * N >= 2^23`, M past one tile) without its zero,
/// and the column-panel walk, on rank-one operands.
#[test]
fn accumulate_operands_on_the_split_and_panel_lanes() {
    with_gpu(|rt| {
        let mut cases: Vec<(Layout, (usize, usize, usize))> = vec![(Layout::Nn, (64, 1024, 8192))];
        for &dims in F32_PANEL_SHAPES {
            cases.extend([(Layout::Nn, dims), (Layout::Tn, dims), (Layout::Nt, dims)]);
        }
        for (layout, (m, n, k)) in cases {
            let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
            let (a_host, b_host, r) = rank_one_case(layout, m, n, k, 0x8acc ^ (m * n) as u64);
            let c0 = random_f32(m * n, 0x9acc ^ (m * n) as u64);
            let expect = with_previous(r, &c0);
            let (a, b) = (tensor_f32(rt, &a_shape, &a_host), tensor_f32(rt, &b_shape, &b_host));
            let c = tensor_f32(rt, &[m, n], &c0);
            let label = format!("ExactF32 {layout:?}_acc {m}x{n}x{k}");
            run(GemmOperands::ExactF32, layout, &a, &b, &c).unwrap_or_else(|e| panic!("{label}: {e}"));
            rt.synchronize().unwrap();
            assert_within_bound(&label, &c.read_f32().unwrap(), &expect, k, 0.0);
        }
    });
}

/// C may not overlap an operand (it is read and written), and the exact
/// lane refuses a runtime with relaxed precision on.
#[test]
fn accumulate_operands_refuse_what_they_cannot_do() {
    with_gpu(|rt| {
        let a = tensor_f32(rt, &[64, 64], &random_f32(64 * 64, 1));
        let b = tensor_f32(rt, &[64, 64], &random_f32(64 * 64, 2));
        let err = GemmOperands::ExactF32.nn_acc(&a, &b, &a).unwrap_err();
        assert!(err.contains("overlap"), "{err}");
        let c = tensor_f32(rt, &[64, 64], &[0.0; 64 * 64]);
        rt.set_relaxed_precision(true);
        let err = GemmOperands::ExactF32.tn_acc(&a, &b, &c).unwrap_err();
        rt.set_relaxed_precision(false);
        assert!(err.contains("relaxed"), "{err}");
    });
}
