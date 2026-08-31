//! Ragged extents: shapes that do not divide the tile geometries the README
//! documents.
//!
//! Every kernel here dispatches a whole number of tiles and then relies on
//! origin-shifted bounds-checked slices to keep the trailing partial tile from
//! reading or writing past the operands. That tail is invisible on the square
//! power-of-two shapes the benchmarks use, and a mis-derived grid or a dropped
//! edge slice leaves output elements simply unwritten -- which reads as a
//! plausible-looking matrix, not as a crash. These cases put at least one
//! extent on each side of every documented tile boundary.
//!
//! Boundaries covered (from README "GEMM Pipeline & Kernel Selection"):
//!   32x32   TILE_F32          -- f32 exact NN / TN / NT
//!   64x64   TILE_COOP_NARROW  -- bf16 and relaxed NN with N <= 512
//!   128x64  TILE_COOP_DEFAULT -- bf16 and relaxed NN with N > 512
//!   128x64  TILE_COOP_TN_NT   -- bf16 TN / NT descriptors
//!   tiles_n * tiles_m >= 2048 -- column-panel swizzle

mod common;

use common::{
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, with_gpu,
    Layout,
};
use tessl::gemm::{gemm_nt_f32, gemm_nt_train, gemm_tn_f32, gemm_tn_train};
use tessl::{gemm, gemm_f32, GemmBackend, GpuRuntime, PrecisionMode};

/// Degenerate and boundary-straddling (M, N, K).
///
/// The three degenerate rows come first because a single row, a single column
/// and a single reduction step are the shapes where "one tile" and "one
/// element" coincide, so any off-by-one in the grid is unambiguous.
const RAGGED: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 129, 65), // 1xN: a decode-shaped row vector against a ragged N
    (129, 1, 65), // Mx1: a single output column
    (65, 65, 1),  // K=1: one reduction step, so C is a rank-1 outer product
    (31, 31, 31), // just under the 32x32 f32 tile
    (33, 33, 33), // just over it
    (63, 65, 33), // straddles 64 on M and N in opposite directions
    (65, 63, 31),
    (127, 129, 63), // just under / over the 128x64 coop default tile
    (129, 127, 65),
    (130, 257, 96), // both extents ragged, several tiles deep
];

fn check_f32(rt: &std::sync::Arc<GpuRuntime>, layout: Layout, backend: GemmBackend) {
    for &(m, n, k) in RAGGED {
        let (a_shape, b_shape) = match layout {
            Layout::Nn => ([m, k], [k, n]),
            Layout::Tn => ([k, m], [k, n]),
            Layout::Nt => ([m, k], [n, k]),
        };
        let a_host = random_f32(m * k, 0x1234 ^ (m * 7 + k * 3 + n) as u64);
        let b_host = random_f32(k * n, 0x5678 ^ (n * 5 + k * 11 + m) as u64);
        let expect = reference(layout, &a_host, &b_host, m, n, k);
        let a = tensor_f32(rt, &a_shape, &a_host);
        let b = tensor_f32(rt, &b_shape, &b_host);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        match layout {
            Layout::Nn => gemm_f32(&a, &b, &c, backend),
            Layout::Tn => gemm_tn_f32(&a, &b, &c, backend),
            Layout::Nt => gemm_nt_f32(&a, &b, &c, backend),
        }
        .unwrap();
        rt.synchronize().unwrap();
        assert_within_bound(
            &format!("f32 {layout:?} {backend:?} {m}x{k}@{k}x{n}"),
            &c.buffer.read_f32(),
            &expect,
            k,
            0.0,
        );
    }
}

fn check_bf16(rt: &std::sync::Arc<GpuRuntime>, layout: Layout) {
    for &(m, n, k) in RAGGED {
        let (a_shape, b_shape) = match layout {
            Layout::Nn => ([m, k], [k, n]),
            Layout::Tn => ([k, m], [k, n]),
            Layout::Nt => ([m, k], [n, k]),
        };
        let a_host = round_trip_bf16(&random_f32(m * k, 0x9abc ^ (m * 3 + k + n) as u64));
        let b_host = round_trip_bf16(&random_f32(k * n, 0xdef0 ^ (n * 9 + k + m) as u64));
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
            0.0,
        );
    }
}

#[test]
fn nn_f32_tensorops_handles_ragged_extents() {
    with_gpu(|rt| check_f32(rt, Layout::Nn, GemmBackend::TensorOps));
}

#[test]
fn tn_f32_tensorops_handles_ragged_extents() {
    with_gpu(|rt| check_f32(rt, Layout::Tn, GemmBackend::TensorOps));
}

#[test]
fn nt_f32_tensorops_handles_ragged_extents() {
    with_gpu(|rt| check_f32(rt, Layout::Nt, GemmBackend::TensorOps));
}

#[test]
fn nn_f32_simdgroup_handles_ragged_extents() {
    // The fallback picks matmul_simdgroup_edges_f32 whenever M%16, N%16 or K%8
    // is nonzero, so this table exercises a second kernel entirely.
    with_gpu(|rt| check_f32(rt, Layout::Nn, GemmBackend::Simdgroup));
}

#[test]
fn nn_bf16_handles_ragged_extents() {
    with_gpu(|rt| check_bf16(rt, Layout::Nn));
}

#[test]
fn tn_nt_bf16_handle_ragged_extents() {
    with_gpu(|rt| {
        rt.set_precision(PrecisionMode::Bf16);
        check_bf16(rt, Layout::Tn);
        check_bf16(rt, Layout::Nt);
    });
}

#[test]
fn nn_bf16_straddles_the_narrow_wide_kernel_boundary() {
    with_gpu(|rt| {
        // `nn_coop_kernel` switches tile geometry on N alone, at exactly 512.
        // A ragged M on both sides makes sure the boundary is not merely
        // "reachable" but correct with a partial trailing M tile in each.
        for &(m, n, k) in &[(67, 511, 33), (67, 512, 33), (67, 513, 33), (193, 576, 65)] {
            let a_host = round_trip_bf16(&random_f32(m * k, 0x4242 ^ n as u64));
            let b_host = round_trip_bf16(&random_f32(k * n, 0x2424 ^ n as u64));
            let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
            let a = tensor_bf16(rt, &[m, k], &a_host);
            let b = tensor_bf16(rt, &[k, n], &b_host);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("bf16 NN boundary {m}x{k}@{k}x{n}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                0.0,
            );
        }
    });
}

#[test]
fn nn_bf16_column_panel_swizzle_covers_a_partial_band() {
    with_gpu(|rt| {
        // At tiles_n * tiles_m >= 2048 the coop NN kernel stops walking tiles
        // linearly and remaps them into 8-tile-row bands. The last band is
        // short whenever tiles_m is not a multiple of 8, and the remap has to
        // clamp to it -- get that wrong and a strip of C is never written while
        // every other shape in this file still passes.
        //
        // N = 512 selects the 64x64 narrow tile, the cheapest geometry that can
        // reach 2048 tiles at all; M = 16545 gives tiles_m = 259 (not a
        // multiple of 8) for tiles_n * tiles_m = 2072. K = 1 keeps the check
        // to one reduction step so the cost is in coverage, not arithmetic.
        let (m, n, k) = (16_545usize, 512usize, 1usize);
        assert_eq!(m.div_ceil(64) * n.div_ceil(64), 2072, "swizzle not engaged");
        assert_ne!(m.div_ceil(64) % 8, 0, "final band is not partial");

        let a_host = round_trip_bf16(&random_f32(m * k, 0xfeed));
        let b_host = round_trip_bf16(&random_f32(k * n, 0xf00d));
        let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);
        let a = tensor_bf16(rt, &[m, k], &a_host);
        let b = tensor_bf16(rt, &[k, n], &b_host);
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        assert_within_bound("bf16 NN swizzle", &c.buffer.read_f32(), &expect, k, 0.0);
    });
}

#[test]
fn output_views_at_a_byte_offset_stay_inside_their_window() {
    with_gpu(|rt| {
        // `Tensor::view` is how consumers slice a bank out of one allocation,
        // and a GEMM writing a view has to respect the offset on C as well as
        // on A and B. Writing the middle third of an oversized buffer and then
        // asserting the untouched thirds are still zero catches a kernel that
        // ignores byte_offset and writes from element 0.
        let (m, n, k) = (33usize, 45usize, 17usize);
        let a_host = random_f32(m * k, 3);
        let b_host = random_f32(k * n, 4);
        let expect = reference(Layout::Nn, &a_host, &b_host, m, n, k);

        let a = tensor_f32(rt, &[m, k], &a_host);
        let b = tensor_f32(rt, &[k, n], &b_host);
        let big = rt.alloc_tensor_f32(&[3 * m * n]).unwrap();
        let c = big.view(&[m, n], m * n);
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let all = big.buffer.read_f32();
        assert_within_bound(
            "f32 NN offset view",
            &all[m * n..2 * m * n],
            &expect,
            k,
            0.0,
        );
        assert!(
            all[..m * n].iter().all(|&x| x == 0.0) && all[2 * m * n..].iter().all(|&x| x == 0.0),
            "GEMM wrote outside the destination view's window"
        );
    });
}
