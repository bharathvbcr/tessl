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
//!   64x64   TILE_COOP_NARROW  -- NN with N <= 512; bf16 also when M < 128
//!   128x64  TILE_COOP_DEFAULT -- N > 512, except bf16 with M < 128
//!   128x64  TILE_COOP_TN_NT   -- bf16 TN / NT descriptors
//!   tiles_n * tiles_m >= 2048 -- column-panel swizzle (coop NN, 8-row bands)
//!   N * K >= 2^23             -- column-panel walk (exact f32, 16-row bands)

mod common;

use common::{
    assert_within_bound, random_f32, rank_one_b_case, rank_one_case, reference, round_trip_bf16, tensor_bf16,
    tensor_bf16_at, tensor_f32, tensor_f32_at, with_gpu, Layout, F32_PANEL_SHAPES,
};
use tessl::gemm::{
    gemm_batched, gemm_epilogue, gemm_nt_bf16, gemm_nt_f32, gemm_nt_train, gemm_tn_accum_train, gemm_tn_bf16,
    gemm_tn_f32, gemm_tn_splitk_par_f32, gemm_tn_train, Activation, BatchStrides, BatchedGemm, Epilogue,
};
use tessl::{gemm, gemm_f32, GemmBackend, GpuRuntime, PrecisionMode, Tensor};

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
        // N = 511 and 512 select 64×64. N = 513 with M = 67 is the short-M
        // bf16 path (64×64 despite N > 512). M = 193, N = 576 fills a 128-row
        // tile and stays on 128×64. Each row keeps a partial trailing M tile.
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
fn exact_f32_column_panels_cover_every_tile() {
    with_gpu(|rt| {
        // Once B (N×K) reaches 2^23 elements the exact-f32 kernels stop walking
        // tiles row-major and remap them into bands of 16 tile rows. A remap
        // that drops or repeats a tile leaves that 32x32 block of C at the
        // zero the host writes first, which only a reference comparison sees.
        for &(m, n, k) in F32_PANEL_SHAPES {
            assert!(n * k >= 1 << 23, "panel walk not engaged at N={n} K={k}");
            for layout in [Layout::Nn, Layout::Tn, Layout::Nt] {
                let (a_shape, b_shape) = match layout {
                    Layout::Nn => ([m, k], [k, n]),
                    Layout::Tn => ([k, m], [k, n]),
                    Layout::Nt => ([m, k], [n, k]),
                };
                let (a_host, b_host, expect) = rank_one_case(layout, m, n, k, 0x9a7e ^ (m * 13 + n) as u64);
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
                    &format!("f32 {layout:?} panel walk {m}x{k}@{k}x{n}"),
                    &c.buffer.read_f32(),
                    &expect,
                    k,
                    0.0,
                );
            }
        }
    });
}

#[test]
fn long_k_nn_partitions_cover_all_of_k() {
    with_gpu(|rt| {
        // An exact-f32 NN whose K is long and whose B does not fit in cache
        // runs as zero-then-accumulate K partitions
        // (`matmul2d_tensorops_nn_splitk_f32`); the routing for these two
        // shapes is pinned by `long_k_nn_with_a_large_b_takes_k_partitions` in
        // src/gemm.rs. Both end on a short partition, and M and N are ragged.
        // With a rank-one B every k contributes, so a partition skipped,
        // repeated, or read from the wrong offset of A or B misses the budget.
        for &(m, n, k) in &[(40usize, 520usize, 16_200usize), (33, 768, 11_008)] {
            let (a_host, b_host, expect) = rank_one_b_case(m, n, k, 0x5917 ^ (m * n + k) as u64);
            let a = tensor_f32(rt, &[m, k], &a_host);
            let b = tensor_f32(rt, &[k, n], &b_host);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("f32 NN split-K {m}x{k}@{k}x{n}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                0.0,
            );
        }
    });
}

/// [`rank_one_b_case`] with A laid out `[K, M]` for a TN: the same product
/// and reference, with every k contributing.
fn rank_one_b_case_tn(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>, common::Reference) {
    let (a_mk, b, expect) = rank_one_b_case(m, n, k, seed);
    let mut a_km = vec![0.0f32; k * m];
    for i in 0..m {
        for p in 0..k {
            a_km[p * m + i] = a_mk[i * k + p];
        }
    }
    (a_km, b, expect)
}

#[test]
fn parallel_tn_partitions_cover_all_of_k() {
    with_gpu(|rt| {
        let check = |m: usize, n: usize, k: usize, k_tile: Option<usize>| {
            let (a_host, b_host, expect) = rank_one_b_case_tn(m, n, k, 0x7a11 ^ (m * n + k) as u64);
            let a = tensor_f32(rt, &[k, m], &a_host);
            let b = tensor_f32(rt, &[k, n], &b_host);
            let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
            match k_tile {
                None => gemm_tn_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap(),
                Some(w) => gemm_tn_splitk_par_f32(&a, &b, &c, w).unwrap(),
            }
            rt.synchronize().unwrap();
            assert_within_bound(
                &format!("f32 TN parallel split-K {m}x{n}x{k}, k_tile {k_tile:?}"),
                &c.buffer.read_f32(),
                &expect,
                k,
                0.0,
            );
        };
        // Through the router (pinned by `few_tile_long_k_tn_takes_parallel_partitions`
        // in src/gemm.rs): the gate weight gradient, and a ragged shape whose
        // last partition is short. With a rank-one B every k contributes, so a
        // partition skipped, repeated, or read from the wrong offset of A or B
        // misses the budget.
        check(12, 768, 4096, None);
        check(13, 520, 3000, None);
        // Shapes the sequential split-K used to take: attention dW, and a
        // ragged one (10 partitions, the last 196 long).
        check(128, 128, 4096, None);
        check(100, 300, 2500, None);
        // The entry at explicit widths, with M·N odd so the slices are padded:
        // one partition (k_tile > K); 150 partitions of 4 (every start off a
        // 256 boundary); two tile rows with a short last partition.
        check(13, 99, 600, Some(1024));
        check(13, 99, 600, Some(4));
        check(33, 77, 1000, Some(256));
    });
}

#[test]
fn parallel_tn_refuses_bad_widths_and_an_oversized_scratch() {
    with_gpu(|rt| {
        let (m, n, k) = (12usize, 768usize, 4096usize);
        let a = rt.alloc_tensor_f32(&[k, m]).unwrap();
        let b = rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
        rt.take_dispatch_count();
        for k_tile in [0usize, 6, 258] {
            let e = gemm_tn_splitk_par_f32(&a, &b, &c, k_tile).expect_err("width accepted");
            assert!(e.contains("multiple of 4"), "k_tile {k_tile}: {e}");
        }
        // 1024 partitions of 12x768 is 9.4M floats, over the 4M cap.
        let e = gemm_tn_splitk_par_f32(&a, &b, &c, 4).expect_err("oversized scratch accepted");
        assert!(e.contains("over the cap"), "{e}");
        assert_eq!(rt.take_dispatch_count(), 0, "a refused call dispatched");
    });
}

#[test]
fn parallel_tn_is_bit_identical_across_100_calls_in_flight() {
    with_gpu(|rt| {
        // Each call takes a fresh scratch and drops it while its work is still
        // queued; 100 calls before one synchronize would expose a scratch
        // handed to the next call too early, or a reduction whose order
        // depended on scheduling.
        let (m, n, k) = (12usize, 768usize, 4096usize);
        let a = tensor_f32(rt, &[k, m], &random_f32(k * m, 0x51));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 0x52));
        let outs: Vec<_> = (0..100).map(|_| rt.alloc_tensor_f32(&[m, n]).unwrap()).collect();
        for c in &outs {
            gemm_tn_f32(&a, &b, c, GemmBackend::TensorOps).unwrap();
        }
        rt.synchronize().unwrap();
        let first: Vec<u32> = outs[0].buffer.read_f32().iter().map(|v| v.to_bits()).collect();
        for (i, c) in outs.iter().enumerate().skip(1) {
            let bits: Vec<u32> = c.buffer.read_f32().iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, first, "call {i} differs from call 0");
        }
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
        // Pad the middle window start to a 16-byte (4-element) boundary so
        // validate_gemm's alignment gate is not what this test exercises.
        let off = (m * n).div_ceil(4) * 4;
        let big = rt.alloc_tensor_f32(&[off + m * n + off]).unwrap();
        let c = big.view(&[m, n], off);
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();

        let all = big.buffer.read_f32();
        assert_within_bound("f32 NN offset view", &all[off..off + m * n], &expect, k, 0.0);
        assert!(
            all[..off].iter().all(|&x| x == 0.0) && all[off + m * n..].iter().all(|&x| x == 0.0),
            "GEMM wrote outside the destination view's window"
        );
    });
}

/// One rule for every kernel family: an operand view on any 16-byte boundary
/// runs (`GEMM_VIEW_ALIGN` in `src/gemm.rs`, from the probe in
/// `bench/results/gemm_align_probe_m5pro.txt`). The cooperative entries used
/// to refuse anything off a 64-byte boundary, so turning relaxed precision on
/// made a view that plain f32 accepted fail. Each entry here runs on views
/// 16, 32 and 48 bytes into their buffers and must reproduce its 0-offset
/// bits exactly.
#[test]
fn cooperative_entries_accept_views_on_any_16_byte_boundary() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "requires the TensorOps metallib");
        // Ragged against every tile; K = 70 also puts each row of A off a
        // 16-byte boundary, as interior tiles already address it.
        let (m, n, k) = (130usize, 97usize, 70usize);
        let a_host = round_trip_bf16(&random_f32(m * k, 0xa1));
        let b_host = round_trip_bf16(&random_f32(k * n, 0xb1));
        let c0 = random_f32(m * n, 0xc1);
        let bias_host = random_f32(n, 0xd1);

        type Entry = fn(&Tensor, &Tensor, &Tensor, &Tensor) -> Result<(), String>;
        let entries: [(&str, bool, Entry); 5] = [
            ("gemm relaxed f32", false, |a, b, c, _| {
                gemm(a, b, c, GemmBackend::TensorOps)
            }),
            ("gemm bf16", true, |a, b, c, _| gemm(a, b, c, GemmBackend::TensorOps)),
            ("gemm_tn_bf16", true, |a, b, c, _| gemm_tn_bf16(a, b, c)),
            ("gemm_nt_bf16", true, |a, b, c, _| gemm_nt_bf16(a, b, c)),
            ("gemm_epilogue bf16", true, |a, b, c, bias| {
                let epi = Epilogue {
                    alpha: 0.5,
                    beta: 0.25,
                    bias: Some(bias),
                    activation: Activation::None,
                };
                gemm_epilogue(a, b, c, GemmBackend::TensorOps, epi)
            }),
        ];
        rt.set_relaxed_precision(true);
        for (name, bf16, entry) in entries {
            let (a_shape, b_shape) = match name {
                "gemm_tn_bf16" => ([k, m], [k, n]),
                "gemm_nt_bf16" => ([m, k], [n, k]),
                _ => ([m, k], [k, n]),
            };
            let run = |pad: usize| -> Vec<u32> {
                let (a, b) = if bf16 {
                    (
                        tensor_bf16_at(rt, &a_shape, &a_host, pad),
                        tensor_bf16_at(rt, &b_shape, &b_host, pad),
                    )
                } else {
                    (
                        tensor_f32_at(rt, &a_shape, &a_host, pad),
                        tensor_f32_at(rt, &b_shape, &b_host, pad),
                    )
                };
                let c = tensor_f32_at(rt, &[m, n], &c0, pad);
                let bias = tensor_f32_at(rt, &[n], &bias_host, pad);
                entry(&a, &b, &c, &bias).unwrap_or_else(|e| panic!("{name} at a {pad}-byte offset: {e}"));
                rt.synchronize().unwrap();
                c.read_f32().unwrap().iter().map(|v| v.to_bits()).collect()
            };
            let aligned = run(0);
            for pad in [16, 32, 48] {
                assert_eq!(run(pad), aligned, "{name}: a {pad}-byte offset changed the result");
            }
        }
    });
}

/// Three contiguous batches whose matrices are whole 16-byte units in bf16
/// and f32, so only the offset under test moves a batch start.
const BATCHED: BatchedGemm = BatchedGemm {
    m: 144,
    n: 96,
    k: 80,
    batch: 3,
    strides: BatchStrides {
        a: 144 * 80,
        b: 80 * 96,
        c: 144 * 96,
    },
};

/// `gemm_batched` starts batch `i` at `base + i * stride`. With a 16-byte base
/// it must run, bit-identical to the aligned base; with an aligned base and a
/// stride that is not a whole number of 16-byte units, batch 1 is misaligned
/// and the call is refused (next test).
#[test]
fn batched_accepts_a_base_on_any_16_byte_boundary() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "requires the TensorOps metallib");
        let (spec, batch) = (BATCHED, BATCHED.batch);
        let (m, n, k) = (spec.m, spec.n, spec.k);
        let a_host = round_trip_bf16(&random_f32(batch * m * k, 0xa2));
        let b_host = round_trip_bf16(&random_f32(batch * k * n, 0xb2));
        let run = |pad: usize| -> Vec<u32> {
            let a = tensor_bf16_at(rt, &[batch * m, k], &a_host, pad);
            let b = tensor_bf16_at(rt, &[batch * k, n], &b_host, pad);
            let c = tensor_f32_at(rt, &[batch * m, n], &vec![f32::NAN; batch * m * n], pad);
            gemm_batched(&a, &b, &c, GemmBackend::TensorOps, spec)
                .unwrap_or_else(|e| panic!("batched at a {pad}-byte base: {e}"));
            rt.synchronize().unwrap();
            c.read_f32().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        let aligned = run(0);
        for pad in [16, 32, 48] {
            assert_eq!(run(pad), aligned, "a {pad}-byte batched base changed the result");
        }
    });
}

/// Aligned bases; one stride per operand two elements past a whole matrix, so
/// batch 1 of that operand sits 4 (bf16) or 8 (f32) bytes off a 16-byte
/// boundary. Each is refused before encoding, naming the entry and operand.
#[test]
fn batched_refuses_a_stride_that_misaligns_a_later_batch() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "requires the TensorOps metallib");
        rt.set_relaxed_precision(true);
        let (spec, batch) = (BATCHED, BATCHED.batch);
        let (m, n, k) = (spec.m, spec.n, spec.k);
        for (operand, strides) in [
            (
                "A",
                BatchStrides {
                    a: m * k + 2,
                    ..spec.strides
                },
            ),
            (
                "B",
                BatchStrides {
                    b: k * n + 2,
                    ..spec.strides
                },
            ),
            (
                "C",
                BatchStrides {
                    c: m * n + 2,
                    ..spec.strides
                },
            ),
        ] {
            let pad_elems = 2 * batch;
            let a = tensor_bf16_at(
                rt,
                &[batch * m * k + pad_elems],
                &vec![0.0; batch * m * k + pad_elems],
                0,
            );
            let b = tensor_bf16_at(
                rt,
                &[batch * k * n + pad_elems],
                &vec![0.0; batch * k * n + pad_elems],
                0,
            );
            let c = tensor_f32_at(
                rt,
                &[batch * m * n + pad_elems],
                &vec![0.0; batch * m * n + pad_elems],
                0,
            );
            rt.take_dispatch_count();
            let err = gemm_batched(&a, &b, &c, GemmBackend::TensorOps, BatchedGemm { strides, ..spec })
                .expect_err("a misaligned batch stride must be refused");
            assert!(
                err.contains("gemm_batched")
                    && err.contains(&format!("operand {operand} stride"))
                    && err.contains("16-byte"),
                "error must name the entry, the operand and the rule: {err}"
            );
            assert_eq!(rt.take_dispatch_count(), 0, "refused before encoding");
        }
    });
}

/// A misaligned view's error names the public entry point, the operand and
/// the 16-byte rule, not an internal path.
#[test]
fn misaligned_views_name_the_entry_and_the_rule() {
    with_gpu(|rt| {
        assert!(rt.has_tensorops(), "requires the TensorOps metallib");
        rt.set_relaxed_precision(true);
        let (m, n, k) = (64usize, 64usize, 64usize);
        let a = tensor_f32(rt, &[m, k], &random_f32(m * k, 1));
        let b = tensor_f32(rt, &[k, n], &random_f32(k * n, 2));
        let b_off = tensor_f32_at(rt, &[k, n], &random_f32(k * n, 2), 8);
        let c = tensor_f32(rt, &[m, n], &vec![0.0; m * n]);
        let a_km = tensor_f32_at(rt, &[k, m], &random_f32(m * k, 3), 4);
        let epi = Epilogue {
            alpha: 2.0,
            ..Epilogue::default()
        };
        for (entry, operand, result) in [
            ("gemm", "B", gemm(&a, &b_off, &c, GemmBackend::TensorOps)),
            (
                "gemm_epilogue",
                "B",
                gemm_epilogue(&a, &b_off, &c, GemmBackend::TensorOps, epi),
            ),
            // The identity epilogue runs the plain GEMM; the refusal still
            // names the entry the caller used.
            (
                "gemm_epilogue",
                "B",
                gemm_epilogue(&a, &b_off, &c, GemmBackend::TensorOps, Epilogue::default()),
            ),
            (
                "gemm_tn_accum_train",
                "A",
                gemm_tn_accum_train(&a_km, &b, &c, GemmBackend::TensorOps),
            ),
        ] {
            let err = result.expect_err("an 8- or 4-byte offset is under the 16-byte rule");
            assert!(
                err.starts_with(&format!("{entry}: operand {operand} byte_offset")) && err.contains("16-byte"),
                "{entry}: {err}"
            );
        }
    });
}
