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
    assert_within_bound, random_f32, reference, round_trip_bf16, tensor_bf16, tensor_f32, tolerance, with_gpu, Layout,
    U_BF16,
};
use tessl::gemm::{
    gemm_bf16, gemm_nt_bf16, gemm_nt_f32, gemm_nt_train, gemm_tiled, gemm_tn_bf16, gemm_tn_f32, gemm_tn_train, EpiTile,
    GemmOperands,
};
use tessl::qwen35::GdnProjLayout;
use tessl::tensor::f32_slice_to_f16;
use tessl::{gemm, gemm_f32, GemmBackend, GpuRuntime, PrecisionMode, Tensor};

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
        // N=512 still selects 64×64. N=520 with M=96 is the short-M bf16
        // exception (64×64 even though N > 512). M=129, N=520 stays on 128×64.
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

/// `gemm_bf16` / `gemm_tn_bf16` / `gemm_nt_bf16`, directly and through
/// `GemmOperands::Bf16`, round f32 operands to bf16 whatever the runtime's
/// mode. The runtime stays in `PrecisionMode::F32`, where the `*_train`
/// functions run exact f32: each result must match the reference on the
/// bf16-rounded operands within the accumulation bound, and must not be the
/// exact-f32 result (some element outside that result's own bound). The TN
/// shape `128 x 384 x 2048` takes the split-K lane.
#[test]
fn bf16_entry_points_round_f32_operands_in_any_runtime_mode() {
    with_gpu(|rt| {
        assert_eq!(rt.precision(), PrecisionMode::F32);
        let cases = [
            (Layout::Nn, 64, 64, 128),
            (Layout::Nn, 130, 520, 96),
            (Layout::Tn, 130, 200, 96),
            (Layout::Tn, 128, 384, 2048),
            (Layout::Nt, 130, 200, 96),
        ];
        for (layout, m, n, k) in cases {
            let (a_shape, b_shape) = operand_shapes(layout, m, n, k);
            let a_host = random_f32(m * k, 0x6b1f ^ (m * 7 + k) as u64);
            let b_host = random_f32(k * n, 0x3c3c ^ (n * 11 + k) as u64);
            let rounded = reference(layout, &round_trip_bf16(&a_host), &round_trip_bf16(&b_host), m, n, k);
            let exact = reference(layout, &a_host, &b_host, m, n, k);
            let a = tensor_f32(rt, &a_shape, &a_host);
            let b = tensor_f32(rt, &b_shape, &b_host);
            for via_enum in [false, true] {
                let c = rt.alloc_tensor_f32(&[m, n]).unwrap();
                match (layout, via_enum) {
                    (Layout::Nn, false) => gemm_bf16(&a, &b, &c),
                    (Layout::Tn, false) => gemm_tn_bf16(&a, &b, &c),
                    (Layout::Nt, false) => gemm_nt_bf16(&a, &b, &c),
                    (Layout::Nn, true) => GemmOperands::Bf16.nn(&a, &b, &c),
                    (Layout::Tn, true) => GemmOperands::Bf16.tn(&a, &b, &c),
                    (Layout::Nt, true) => GemmOperands::Bf16.nt(&a, &b, &c),
                }
                .unwrap();
                rt.synchronize().unwrap();
                let got = c.buffer.read_f32();
                let label = format!("bf16 operands {layout:?} {m}x{n}x{k} enum={via_enum}");
                assert_within_bound(&label, &got, &rounded, k, 0.0);
                let off_exact = got
                    .iter()
                    .zip(exact.c.iter().zip(&exact.mag))
                    .any(|(&g, (&w, &mag))| (f64::from(g) - w).abs() > tolerance(k, mag, 0.0));
                assert!(
                    off_exact,
                    "{label}: equals the exact-f32 product, so nothing was rounded"
                );
            }
        }
    });
}

/// `GemmOperands::ExactF32` means exact: it refuses a runtime whose
/// relaxed-precision (tf32-class) GEMMs are on, in every layout, before
/// dispatching anything.
#[test]
fn exact_f32_operands_refuse_relaxed_precision() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let a = tensor_f32(rt, &[64, 64], &random_f32(64 * 64, 1));
        let b = tensor_f32(rt, &[64, 64], &random_f32(64 * 64, 2));
        let c = rt.alloc_tensor_f32(&[64, 64]).unwrap();
        rt.take_dispatch_count();
        for r in [
            GemmOperands::ExactF32.nn(&a, &b, &c),
            GemmOperands::ExactF32.tn(&a, &b, &c),
            GemmOperands::ExactF32.nt(&a, &b, &c),
        ] {
            let e = r.expect_err("relaxed precision accepted as exact");
            assert!(e.contains("relaxed precision is on"), "{e}");
        }
        assert_eq!(rt.take_dispatch_count(), 0);
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

fn f16_tensor(rt: &std::sync::Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f16(shape).expect("alloc_tensor_f16");
    t.buffer.write_f16_bits(&f32_slice_to_f16(data));
    t
}

/// `|narrow - wide|` against [`tolerance`] with `operand_u = 0`, the bound
/// [`assert_within_bound`] uses for bf16 GEMM. Bit-identical tiles never
/// consult a magnitude: the difference is zero, which is inside that bound.
#[allow(clippy::too_many_arguments)]
fn assert_tiles_within_bf16_bound(
    label: &str,
    narrow: &[f32],
    wide: &[f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(narrow.len(), m * n, "{label}");
    assert_eq!(wide.len(), m * n, "{label}");
    let mut worst = 0.0f32;
    let mut bits_differ = 0usize;
    let mut worst_scaled = 0.0f64;
    let mut worst_at = 0usize;
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let (g, w) = (narrow[idx], wide[idx]);
            assert!(g.is_finite() && w.is_finite(), "{label}: non-finite at {idx}");
            let err = (g - w).abs();
            if err > worst {
                worst = err;
                worst_at = idx;
            }
            if g.to_bits() == w.to_bits() {
                continue;
            }
            bits_differ += 1;
            let mut mag = 0.0f64;
            for p in 0..k {
                mag += (a[i * k + p] as f64).abs() * (b[p * n + j] as f64).abs();
            }
            let tol = tolerance(k, mag, 0.0);
            let scaled = if tol > 0.0 {
                err as f64 / tol
            } else if err == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            worst_scaled = worst_scaled.max(scaled);
        }
    }
    println!(
        "{label}: worst |64x64-128x64| = {worst:.6e} at {worst_at} bits_differ {bits_differ}/{} scaled {worst_scaled:.3} (bf16 GEMM tolerance, operand_u=0)",
        m * n
    );
    assert!(
        worst_scaled <= 1.0,
        "{label}: element {worst_at} is {worst_scaled:.3}x the bf16 GEMM tolerance"
    );
}

fn same_bits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label}: diverged at {i} ({g} vs {w})");
    }
}

/// Qwen3.5-2B GDN fused in-projection: `gemm` of `[M, hidden]` by
/// `[hidden, GdnProjLayout::width]`. N is not a multiple of 64, so the last
/// tile is partial on both geometries.
#[test]
fn short_m_bf16_plain_gemm_matches_the_wide_tile() {
    with_gpu(|rt| {
        let n = GdnProjLayout::new(16, 16, 128).unwrap().width() as usize;
        let k = 2048usize;
        assert_eq!(n, 8224, "Qwen3.5-2B GDN fused in-proj width");
        assert!(n > 512);
        assert_ne!(n % 64, 0, "the shape must keep a partial N tile");

        let b_h = round_trip_bf16(&random_f32(k * n, 0xB164));
        let b16 = tensor_bf16(rt, &[k, n], &b_h);
        let b_f16 = f16_tensor(rt, &[k, n], &random_f32(k * n, 0xF164));

        for (case, &m) in [1usize, 127, 128, 200].iter().enumerate() {
            let a_h = round_trip_bf16(&random_f32(m * k, 0xA164 + case as u64));
            let a16 = tensor_bf16(rt, &[m, k], &a_h);
            let wide = rt.alloc_tensor_f32(&[m, n]).unwrap();
            let narrow = rt.alloc_tensor_f32(&[m, n]).unwrap();
            let auto = rt.alloc_tensor_f32(&[m, n]).unwrap();
            for c in [&wide, &narrow, &auto] {
                c.buffer.write_f32(&vec![f32::NAN; m * n]);
            }
            gemm_tiled(&a16, &b16, &wide, GemmBackend::TensorOps, EpiTile::Wide).expect("wide bf16");
            gemm_tiled(&a16, &b16, &narrow, GemmBackend::TensorOps, EpiTile::Narrow).expect("narrow bf16");
            gemm(&a16, &b16, &auto, GemmBackend::TensorOps).expect("auto bf16");
            rt.synchronize().unwrap();
            let wide_out = wide.buffer.read_f32();
            let narrow_out = narrow.buffer.read_f32();
            let auto_out = auto.buffer.read_f32();
            assert!(
                wide_out[..m * n].iter().any(|v| v.abs() > 1e-3),
                "M={m} wide bf16 wrote nothing"
            );
            assert_tiles_within_bf16_bound(
                &format!("bf16 M={m} {m}x{n}x{k}"),
                &narrow_out[..m * n],
                &wide_out[..m * n],
                &a_h,
                &b_h,
                m,
                n,
                k,
            );
            let followed = if m < 128 { &narrow_out } else { &wide_out };
            same_bits(&format!("bf16 auto M={m}"), &auto_out[..m * n], &followed[..m * n]);

            // f16 keeps the N rule: N > 512 stays on 128×64 for every M here.
            let a_f = f16_tensor(rt, &[m, k], &random_f32(m * k, 0xF16A + case as u64));
            let f_wide = rt.alloc_tensor_f32(&[m, n]).unwrap();
            let f_auto = rt.alloc_tensor_f32(&[m, n]).unwrap();
            for c in [&f_wide, &f_auto] {
                c.buffer.write_f32(&vec![f32::NAN; m * n]);
            }
            gemm_tiled(&a_f, &b_f16, &f_wide, GemmBackend::TensorOps, EpiTile::Wide).expect("wide f16");
            gemm(&a_f, &b_f16, &f_auto, GemmBackend::TensorOps).expect("auto f16");
            rt.synchronize().unwrap();
            let f_wide_out = f_wide.buffer.read_f32();
            let f_auto_out = f_auto.buffer.read_f32();
            assert!(
                f_wide_out[..m * n].iter().all(|v| v.is_finite()),
                "M={m} f16 wide non-finite"
            );
            assert!(
                f_wide_out[..m * n].iter().any(|v| v.abs() > 1e-3),
                "M={m} f16 wide wrote nothing"
            );
            same_bits(
                &format!("f16 auto stays on 128x64 M={m}"),
                &f_auto_out[..m * n],
                &f_wide_out[..m * n],
            );

            // Exact f32 does not use the cooperative selector. K is short so
            // the reference stays cheap; N is still past 512.
            check_f32(rt, Layout::Nn, m, n, 8);
        }

        rt.take_dispatch_count();
        let a = tensor_bf16(rt, &[1, 64], &vec![1.0f32; 64]);
        let b_bad = tensor_bf16(rt, &[32, 128], &vec![1.0f32; 32 * 128]);
        let c = rt.alloc_tensor_f32(&[1, 128]).unwrap();
        let err = gemm(&a, &b_bad, &c, GemmBackend::TensorOps).expect_err("mismatched K");
        assert_eq!(err, "GEMM inner dimensions or output shape do not match");
        let empty = a.view(&[0, 64], 0);
        let b = tensor_bf16(rt, &[64, 128], &vec![1.0f32; 64 * 128]);
        let err = gemm(&empty, &b, &c, GemmBackend::TensorOps).expect_err("empty M");
        assert_eq!(err, "GEMM requires nonempty rank-2 tensors");
        assert_eq!(rt.take_dispatch_count(), 0, "rejected GEMM still encoded");
    });
}
