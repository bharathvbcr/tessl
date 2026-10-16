//! The small-M bf16 NN lane: the bf16 form of `gemm_small_m.rs`'s exact-f32
//! lane. A skinny bf16 A over a C of few 64x64 tiles runs its K as parallel
//! partitions (`matmul2d_tensorops_nn_splitk_par_bf16_f32`, then
//! `reduce_partitions_f32`) instead of one cooperative 64x64 walk per tile.
//!
//! Same shapes, same rows (1, 7, 16, 63, 64) and the same bounds, fixed
//! before the first run:
//!
//! * every element within `common::tolerance(K, |a.b| sum, 0)`: the operands
//!   are rounded to bf16 before upload and the reference reads those same
//!   values, a bf16 x bf16 product is exact in f32, so what remains is f32
//!   accumulation, `gamma_(K+8) * sum_k |a_k b_k|`, the bound every bf16 GEMM
//!   here meets (the `+8` covers the split-K second pass);
//! * the lane repeats bit for bit;
//! * a poisoned scratch changes nothing;
//! * routing, by the traced kernel counts per shape, in a child process.

mod common;

use common::{
    assert_within_bound, in_traced_child, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_bf16_at,
    tolerance, with_gpu, Layout, Reference,
};
use std::sync::Arc;
use tessl::gemm::{gemm_bf16, gemm_nn_splitk_par_bf16, GemmOperands};
use tessl::runtime::traced_kernel_count;
use tessl::{gemm, GemmBackend, GpuRuntime, Tensor};

const PAR: &str = "matmul2d_tensorops_nn_splitk_par_bf16_f32";
const REDUCE: &str = "reduce_partitions_f32";
const NARROW: &str = "matmul2d_tensorops_bf16_f32_64x64_sg4";
const WIDE: &str = "matmul2d_tensorops_bf16_f32";

const SHAPES: &[(&str, usize, usize)] = &[
    ("down / o global", 2048, 512),
    ("o sliding", 1024, 512),
    ("ple_gate / ple_out", 512, 512),
];
const ROWS: &[usize] = &[1, 7, 16, 63, 64];

/// bf16 operands and the f32 values they hold, for the reference.
fn operands(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize, seed: u64) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let a = random_f32(m * k, seed);
    let b = random_f32(k * n, seed ^ 0x9e37);
    (
        tensor_bf16(rt, &[m, k], &a),
        tensor_bf16(rt, &[k, n], &b),
        round_trip_bf16(&a),
        round_trip_bf16(&b),
    )
}

fn used_budget(got: &[f32], want: &Reference, k: usize) -> f64 {
    got.iter()
        .zip(want.c.iter().zip(&want.mag))
        .map(|(&g, (&c, &mag))| (g as f64 - c).abs() / tolerance(k, mag, 0.0).max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

/// The lane at the real shapes through `gemm_bf16`, `GemmOperands::Bf16` and
/// the forced entry at two widths. Prints the largest share of the error
/// budget any result used (`--nocapture`).
#[test]
fn the_lane_matches_f64_at_the_forward_shapes() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "needs TensorOps");
        let mut worst = 0.0f64;
        for &(name, k, n) in SHAPES {
            for &m in ROWS {
                let (a, b, a_h, b_h) = operands(rt, m, n, k, 0x5ca1 ^ (m * 31 + k) as u64);
                let want = reference(Layout::Nn, &a_h, &b_h, m, n, k);
                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();

                gemm_bf16(&a, &b, &c).unwrap();
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                assert_within_bound(&format!("gemm_bf16 {name} M={m}"), &got, &want, k, 0.0);
                worst = worst.max(used_budget(&got, &want, k));

                GemmOperands::Bf16.nn(&a, &b, &c).unwrap();
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                assert_within_bound(&format!("Bf16.nn {name} M={m}"), &got, &want, k, 0.0);
                worst = worst.max(used_budget(&got, &want, k));

                for k_tile in [128, 256] {
                    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                    gemm_nn_splitk_par_bf16(&a, &b, &c, k_tile).unwrap();
                    rt.synchronize().unwrap();
                    let got = c.buffer.read_f32();
                    assert_within_bound(&format!("forced k_tile={k_tile} {name} M={m}"), &got, &want, k, 0.0);
                    worst = worst.max(used_budget(&got, &want, k));
                }
            }
        }
        eprintln!("bf16 small-M lane: the worst result used {worst:.4} of its error budget");
    });
}

/// Ragged extents, a short last partition, K under one partition's worth, M
/// and N off the 64 tile, and bf16 operands 16 bytes into their buffers (the
/// view alignment every GEMM operand must keep).
#[test]
fn ragged_partitions_and_offset_views_stay_within_the_bound() {
    with_gpu(|rt| {
        for &(m, n, k, k_tile) in &[
            (7usize, 520usize, 2051usize, 256usize),
            (65, 63, 700, 128),
            (1, 1, 100, 12),
            (16, 512, 300, 128),
            (63, 513, 129, 8),
            (64, 512, 2048, 2040),
        ] {
            let a_h = round_trip_bf16(&random_f32(m * k, 0xa11 ^ (m * k) as u64));
            let b_h = round_trip_bf16(&random_f32(k * n, 0xb22 ^ (n * k) as u64));
            let want = reference(Layout::Nn, &a_h, &b_h, m, n, k);
            let a = tensor_bf16_at(rt, &[m, k], &a_h, 16);
            let b = tensor_bf16_at(rt, &[k, n], &b_h, 16);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_nn_splitk_par_bf16(&a, &b, &c, k_tile).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("{m}x{n}x{k} k_tile={k_tile}"),
                &c.buffer.read_f32(),
                &want,
                k,
                0.0,
            );
        }
    });
}

/// The scratch comes from the unzeroed allocator; under poisoning it is NaN,
/// so an element of a slice the kernel did not write reaches C.
#[test]
fn the_lane_writes_every_slice_element() {
    with_gpu(|rt| {
        rt.set_poison_unzeroed(true);
        for &(name, k, n) in SHAPES {
            for &m in ROWS {
                let (a, b, a_h, b_h) = operands(rt, m, n, k, 0xdead ^ (m + n) as u64);
                let want = reference(Layout::Nn, &a_h, &b_h, m, n, k);
                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                gemm_bf16(&a, &b, &c).unwrap();
                rt.synchronize().unwrap();
                assert_within_bound(&format!("poisoned {name} M={m}"), &c.buffer.read_f32(), &want, k, 0.0);
            }
        }
        rt.set_poison_unzeroed(false);
    });
}

#[test]
fn the_lane_repeats_bit_identically() {
    with_gpu(|rt| {
        let (m, n, k) = (16, 512, 2048);
        let (a, b, _, _) = operands(rt, m, n, k, 0x77);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let mut first: Option<Vec<f32>> = None;
        for run in 0..48 {
            gemm_bf16(&a, &b, &c).unwrap();
            rt.synchronize().unwrap();
            let got = c.buffer.read_f32();
            match &first {
                None => first = Some(got),
                Some(f) => assert!(
                    f.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "run {run} differs from run 0"
                ),
            }
        }
    });
}

/// The forced entry refuses a width the kernel cannot take, and an output
/// that is not f32.
#[test]
fn the_forced_entry_refuses_a_bad_width() {
    with_gpu(|rt| {
        let (m, n, k) = (16, 64, 256);
        let (a, b, _, _) = operands(rt, m, n, k, 3);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        for bad in [0usize, 3, 130] {
            let err = gemm_nn_splitk_par_bf16(&a, &b, &c, bad).unwrap_err();
            assert!(err.contains("K partition"), "k_tile {bad}: {err}");
        }
    });
}

/// Which kernels one bf16 `gemm` call dispatches, by pipeline requests:
/// (parallel split, reduce, narrow 64x64, wide 128x64).
fn route(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize) -> (u64, u64, u64, u64) {
    let (a, b, _, _) = operands(rt, m, n, k, 0x1 ^ (m + n + k) as u64);
    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
    let count = |name| traced_kernel_count(name);
    let before = (count(PAR), count(REDUCE), count(NARROW), count(WIDE));
    gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
    rt.synchronize().unwrap();
    (
        count(PAR) - before.0,
        count(REDUCE) - before.1,
        count(NARROW) - before.2,
        count(WIDE) - before.3,
    )
}

/// Which path each shape takes. A shape on the lane dispatches the parallel
/// kernel and one reduce and neither single-dispatch kernel; every other
/// shape keeps the tile it had.
#[test]
fn routing_by_shape() {
    let _ = in_traced_child("routing_by_shape", || {
        with_gpu(|rt| {
            let lane = (1, 1, 0, 0);
            let narrow = (0, 0, 1, 0);
            let wide = (0, 0, 0, 1);
            for &(m, n, k, want, why) in &[
                (16usize, 512usize, 2048usize, lane, "down at 1x16"),
                (1, 512, 2048, lane, "down at one row"),
                (64, 512, 2048, lane, "down at 64 rows"),
                (16, 512, 1024, lane, "o sliding at 1x16"),
                (16, 512, 512, lane, "ple_gate / ple_out at 1x16"),
                (65, 512, 2048, narrow, "past 64 rows the C is 16 tiles"),
                (16, 2048, 512, narrow, "qkv sliding at 1x16: 32 tiles"),
                (16, 4096, 512, narrow, "gate|up at 1x16: short M keeps 64x64"),
                (2048, 512, 2048, narrow, "down at 64x32: N <= 512 keeps 64x64"),
                (2048, 4096, 512, wide, "gate|up at 64x32: 128x64"),
                (16, 512, 128, narrow, "K within one partition"),
            ] {
                assert_eq!(route(rt, m, n, k), want, "{why}: {m}x{n}x{k}");
            }
        });
    });
}
