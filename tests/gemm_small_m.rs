//! The small-M exact-f32 NN lane: a skinny A over a C of few tiles runs its K
//! as parallel partitions (`matmul2d_tensorops_nn_splitk_par_f32`, then
//! `reduce_partitions_f32`) instead of one single-simdgroup walk per tile.
//!
//! The shapes are EmbeddingGemma 2's at one 16-token sequence: the down and
//! global-o GEMMs (K 2048, N 512), the sliding-o GEMM (K 1024, N 512) and the
//! two per-layer-input GEMMs (K 512, N 512), at the row counts the lane is for
//! (1, 7, 16, 63, 64).
//!
//! Bounds, fixed before the first run and not loosened since:
//!
//! * every element within `common::tolerance(K, |a.b| sum, 0)`, the f32
//!   dot-product bound `gamma_(K+8) * sum_k |a_k b_k|` (`tests/common`). The
//!   `+8` is that bound's own allowance for a split-K lane's second summation
//!   pass, so the lane is held to the bound every other exact-f32 GEMM meets,
//!   not a looser one. A different summation order is the whole difference
//!   from the single dispatch; this is what keeps it honest;
//! * the lane repeats bit for bit (no atomics, a fixed reduction order);
//! * a poisoned scratch changes nothing (every element of every slice is
//!   written before the reduce reads it);
//! * routing: the traced kernel counts per shape, in a child process.

mod common;

use common::{
    assert_within_bound, in_traced_child, random_f32, rank_one_b_case, reference, tensor_f32, tensor_f32_at, tolerance,
    with_gpu, Layout, Reference,
};
use std::sync::Arc;
use tessl::gemm::{gemm_nn_splitk_par_f32, GemmOperands};
use tessl::runtime::traced_kernel_count;
use tessl::{gemm, GemmBackend, GpuRuntime, Tensor};

const PAR: &str = "matmul2d_tensorops_nn_splitk_par_f32";
const REDUCE: &str = "reduce_partitions_f32";
const PLAIN: &str = "matmul2d_tensorops_f32";
const SEQ_SPLIT: &str = "matmul2d_tensorops_nn_splitk_f32";

/// (name, K, N): the forward's GEMMs whose C is under 64 tiles at 16 rows.
const SHAPES: &[(&str, usize, usize)] = &[
    ("down / o global", 2048, 512),
    ("o sliding", 1024, 512),
    ("ple_gate / ple_out", 512, 512),
];
const ROWS: &[usize] = &[1, 7, 16, 63, 64];

fn operands(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize, seed: u64) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let a = random_f32(m * k, seed);
    let b = random_f32(k * n, seed ^ 0x9e37);
    (tensor_f32(rt, &[m, k], &a), tensor_f32(rt, &[k, n], &b), a, b)
}

/// The largest error over its element's budget (`tolerance`), for the record:
/// the bound passes at 1.0, this says how much of it a result used.
fn used_budget(got: &[f32], want: &Reference, k: usize) -> f64 {
    got.iter()
        .zip(want.c.iter().zip(&want.mag))
        .map(|(&g, (&c, &mag))| (g as f64 - c).abs() / tolerance(k, mag, 0.0).max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

/// The lane at the real shapes, through `gemm` and through the forced entry
/// at two widths, against the f64 reference. Prints the largest share of the
/// error budget any result used (`--nocapture`).
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

                gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                assert_within_bound(&format!("gemm {name} M={m}"), &got, &want, k, 0.0);
                worst = worst.max(used_budget(&got, &want, k));

                GemmOperands::ExactF32.nn(&a, &b, &c).unwrap();
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                assert_within_bound(&format!("ExactF32.nn {name} M={m}"), &got, &want, k, 0.0);
                worst = worst.max(used_budget(&got, &want, k));

                for k_tile in [128, 256] {
                    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                    gemm_nn_splitk_par_f32(&a, &b, &c, k_tile).unwrap();
                    rt.synchronize().unwrap();
                    let got = c.buffer.read_f32();
                    assert_within_bound(&format!("forced k_tile={k_tile} {name} M={m}"), &got, &want, k, 0.0);
                    worst = worst.max(used_budget(&got, &want, k));
                }
            }
        }
        eprintln!("small-M lane: the worst result used {worst:.4} of its error budget");
    });
}

/// A rank-one B makes every K term count while the reference stays O(M*K):
/// a partition the kernel skips or reads from the wrong column of A changes
/// the answer, which random operands can hide behind the bound.
#[test]
fn a_skipped_or_misplaced_partition_changes_the_answer() {
    with_gpu(|rt| {
        for &(name, k, n) in SHAPES {
            for &m in ROWS {
                let (a_h, b_h, want) = rank_one_b_case(m, n, k, 0xb0b ^ (m + k) as u64);
                let (a, b) = (tensor_f32(rt, &[m, k], &a_h), tensor_f32(rt, &[k, n], &b_h));
                for k_tile in [128, 256, 1024] {
                    if k_tile >= k {
                        continue;
                    }
                    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                    gemm_nn_splitk_par_f32(&a, &b, &c, k_tile).unwrap();
                    rt.synchronize().unwrap();
                    assert_within_bound(
                        &format!("rank-one k_tile={k_tile} {name} M={m}"),
                        &c.buffer.read_f32(),
                        &want,
                        k,
                        0.0,
                    );
                }
            }
        }
    });
}

/// Ragged extents and the widest ways a partition can fall: a short last
/// partition (K = 2051 over 256), K under one partition's worth, M and N off
/// the 32 tile, and operands that start 16 bytes into their buffers.
#[test]
fn ragged_partitions_and_offset_views_stay_within_the_bound() {
    with_gpu(|rt| {
        for &(m, n, k, k_tile) in &[
            (7usize, 520usize, 2051usize, 256usize),
            (33, 31, 700, 128),
            (1, 1, 100, 12),
            (16, 512, 300, 128),
            (63, 513, 129, 4),
            (64, 512, 2048, 2044),
        ] {
            let a_h = random_f32(m * k, 0xa11 ^ (m * k) as u64);
            let b_h = random_f32(k * n, 0xb22 ^ (n * k) as u64);
            let want = reference(Layout::Nn, &a_h, &b_h, m, n, k);
            // 16 bytes = the alignment every GEMM operand view must keep.
            let a = tensor_f32_at(rt, &[m, k], &a_h, 16);
            let b = tensor_f32_at(rt, &[k, n], &b_h, 16);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_nn_splitk_par_f32(&a, &b, &c, k_tile).unwrap();
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
/// so an element of a slice that the kernel did not write reaches C.
#[test]
fn the_lane_writes_every_slice_element() {
    with_gpu(|rt| {
        rt.set_poison_unzeroed(true);
        for &(name, k, n) in SHAPES {
            for &m in ROWS {
                let (a, b, a_h, b_h) = operands(rt, m, n, k, 0xdead ^ (m + n) as u64);
                let want = reference(Layout::Nn, &a_h, &b_h, m, n, k);
                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
                rt.synchronize().unwrap();
                assert_within_bound(&format!("poisoned {name} M={m}"), &c.buffer.read_f32(), &want, k, 0.0);
            }
        }
        rt.set_poison_unzeroed(false);
    });
}

/// Same kernel, same partitions, same reduction order: the same bits, over
/// enough repeats to cross command-buffer and allocator-slot boundaries, and
/// with the scratch recycled from the pool between calls.
#[test]
fn the_lane_repeats_bit_identically() {
    with_gpu(|rt| {
        let (m, n, k) = (16, 512, 2048);
        let (a, b, _, _) = operands(rt, m, n, k, 0x77);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        let mut first: Option<Vec<f32>> = None;
        for run in 0..48 {
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
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

/// The forced entry refuses what the kernel cannot do, rather than running it.
#[test]
fn the_forced_entry_refuses_a_bad_width_and_dtype() {
    with_gpu(|rt| {
        let (m, n, k) = (16, 64, 256);
        let (a, b, _, _) = operands(rt, m, n, k, 3);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        for bad in [0usize, 3, 130] {
            let err = gemm_nn_splitk_par_f32(&a, &b, &c, bad).unwrap_err();
            assert!(err.contains("K partition"), "k_tile {bad}: {err}");
        }
        let a_bf = tessl::gemm::cast_f32_to_bf16(&a).unwrap();
        let b_bf = tessl::gemm::cast_f32_to_bf16(&b).unwrap();
        assert!(gemm_nn_splitk_par_f32(&a_bf, &b_bf, &c, 128).is_err());
    });
}

/// Pipeline requests for `name` since `before` (one per dispatch: `gemm`
/// asks the runtime for its pipeline on every call).
fn count(name: &str) -> u64 {
    traced_kernel_count(name)
}

/// Which kernels one `gemm` call dispatches, by pipeline requests.
fn route(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize) -> (u64, u64, u64, u64) {
    let (a, b, _, _) = operands(rt, m, n, k, 0x1 ^ (m + n + k) as u64);
    let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
    let before = (count(PAR), count(REDUCE), count(PLAIN), count(SEQ_SPLIT));
    gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
    rt.synchronize().unwrap();
    (
        count(PAR) - before.0,
        count(REDUCE) - before.1,
        count(PLAIN) - before.2,
        count(SEQ_SPLIT) - before.3,
    )
}

/// Which path each shape takes. A shape on the lane dispatches the parallel
/// kernel and one reduce and neither the plain kernel nor the sequential
/// split-K; every other shape keeps the route it had.
#[test]
fn routing_by_shape() {
    let _ = in_traced_child("routing_by_shape", || {
        with_gpu(|rt| {
            // (m, n, k, expected (par, reduce, plain, sequential split-K))
            let lane = (1, 1, 0, 0);
            let plain = (0, 0, 1, 0);
            let seq = (0, 0, 0, 1);
            for &(m, n, k, want, why) in &[
                (16usize, 512usize, 2048usize, lane, "down at 1x16"),
                (1, 512, 2048, lane, "down at one row"),
                (64, 512, 2048, lane, "down at 64 rows, 32 tiles"),
                (16, 512, 1024, lane, "o sliding at 1x16"),
                (16, 512, 512, lane, "ple_gate / ple_out at 1x16"),
                (
                    16,
                    2048,
                    512,
                    plain,
                    "qkv sliding at 1x16: 64 tiles already fill the GPU",
                ),
                (16, 4096, 512, plain, "gate|up at 1x16: 128 tiles"),
                (2048, 512, 2048, plain, "down at 64x32: 1024 tiles"),
                (8192, 512, 2048, plain, "down at 8x1024"),
                (128, 512, 2048, plain, "down at 128 rows: 64 tiles"),
                (16, 512, 128, plain, "K within one partition"),
                (
                    40,
                    520,
                    16_200,
                    seq,
                    "a long K over a large B keeps the sequential split-K",
                ),
                (4096, 768, 50_304, seq, "the LM head's input gradient"),
            ] {
                assert_eq!(route(rt, m, n, k), want, "{why}: {m}x{n}x{k}");
            }
        });
    });
}
