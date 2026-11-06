//! Quantized int8 GEMM with fused dequantization.
//!
//! The property worth testing here is not "close to a reference" but **exact**.
//! `int8 x int8` accumulates into `int32`, and every product fits, so the
//! integer result carries no rounding whatever. The only approximation is the
//! final multiply by the scales. That makes an exact integer reference the
//! right oracle, and it catches errors a float tolerance would absorb.

mod common;

use std::sync::Arc;

use common::{buf, with_gpu};
use tessl::nn;
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn i8_buf(rt: &Arc<GpuRuntime>, data: &[i8]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1)).expect("alloc");
    b.write_bytes(&data.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    b
}

/// Deterministic int8 spread over the full range, including the extremes.
fn i8_data(n: usize, seed: u64) -> Vec<i8> {
    let mut x = seed;
    (0..n)
        .map(|i| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            match i % 16 {
                0 => -128,
                1 => 127,
                _ => ((x >> 33) as i32 % 256 - 128) as i8,
            }
        })
        .collect()
}

#[test]
fn the_integer_accumulation_is_exact() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(128usize, 64usize, 64usize), (96, 80, 128), (37, 45, 33)] {
            let a = i8_data(m * k, 0x1_8000 + k as u64);
            let b = i8_data(k * n, 0x2_8000 + n as u64);
            let ab = i8_buf(rt, &a);
            let bb = i8_buf(rt, &b);
            let cb = buf(rt, &vec![0.0f32; m * n]);

            // a_scale of 1 and no b_scale, so the output *is* the integer sum
            // and any deviation is a real arithmetic error, not rounding.
            nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None).expect("gemm_i8_dequant");
            rt.synchronize().unwrap();

            let got = cb.read_f32();
            for i in 0..m {
                for j in 0..n {
                    let mut acc: i32 = 0;
                    for p in 0..k {
                        acc += a[i * k + p] as i32 * b[p * n + j] as i32;
                    }
                    // Every such sum is exactly representable in f32 here, so
                    // this is equality, not a tolerance.
                    assert_eq!(
                        got[i * n + j],
                        acc as f32,
                        "{m}x{n}x{k} at ({i},{j}): got {} want {acc}",
                        got[i * n + j]
                    );
                }
            }
        }
    });
}

/// Once B (K×N) reaches the shader's `PANEL_MIN_B_ELEMS` = 2^23 elements on a
/// grid that is not square with a power-of-two side, the kernel walks its
/// 128x64 tiles in column panels of 4 tile rows (`tile_walk` in
/// `kernels/gemm_coop.h`). M = 545 is 5 tile rows (a full band
/// and a partial one) and M = 33 one short band; both Ns leave a ragged last
/// tile column. A is zero past its first column, so the exact answer is
/// A[:, 0] times B[0, :], O(M·N) to check at any K. C starts as NaN, so a
/// tile the walk skips or misplaces shows.
#[test]
fn column_panels_cover_every_tile_exactly() {
    assert!(
        include_str!("../kernels/gemm_coop.h").contains("constexpr constant ulong PANEL_MIN_B_ELEMS = 1ul << 23;"),
        "the shader's panel gate moved; update these shapes"
    );
    with_gpu(|rt| {
        for &(m, n, k) in &[(545usize, 8200usize, 1024usize), (33, 8193, 1024)] {
            assert!(n * k >= 1 << 23, "panel walk not engaged at N={n} K={k}");
            let col = i8_data(m, 0x3_8000 + m as u64);
            let mut a = vec![0i8; m * k];
            for (i, &v) in col.iter().enumerate() {
                a[i * k] = v;
            }
            let b = i8_data(k * n, 0x4_8000 + n as u64);
            let ab = i8_buf(rt, &a);
            let bb = i8_buf(rt, &b);
            let cb = buf(rt, &vec![f32::NAN; m * n]);

            nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None).expect("gemm_i8_dequant");
            rt.synchronize().unwrap();

            let got = cb.read_f32();
            let mut bad = 0usize;
            let mut first = None;
            for i in 0..m {
                for j in 0..n {
                    let want = (col[i] as i32 * b[j] as i32) as f32;
                    if got[i * n + j] != want {
                        bad += 1;
                        first.get_or_insert((i, j, got[i * n + j], want));
                    }
                }
            }
            assert_eq!(bad, 0, "{m}x{n}x{k}: {bad} outputs wrong, first {first:?}");
        }
    });
}

#[test]
fn the_per_column_scale_is_applied_per_column() {
    with_gpu(|rt| {
        let (m, n, k) = (128usize, 64usize, 64usize);
        // A of all ones and B of all ones make every integer sum exactly k, so
        // the output is the scale alone — a transposed broadcast is obvious.
        let a = vec![1i8; m * k];
        let b = vec![1i8; k * n];
        let scale: Vec<f32> = (0..n).map(|j| (j as f32 + 1.0) * 0.25).collect();
        let ab = i8_buf(rt, &a);
        let bb = i8_buf(rt, &b);
        let sb = buf(rt, &scale);
        let cb = buf(rt, &vec![0.0f32; m * n]);

        nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 0.5, Some(&sb)).expect("gemm_i8_dequant");
        rt.synchronize().unwrap();

        let got = cb.read_f32();
        for i in 0..m {
            for j in 0..n {
                let want = k as f32 * 0.5 * scale[j];
                assert!(
                    (got[i * n + j] - want).abs() <= 1e-3,
                    "scale wrong at ({i},{j}): got {} want {want}",
                    got[i * n + j]
                );
            }
        }
    });
}

#[test]
fn full_range_operands_do_not_overflow_the_accumulator() {
    with_gpu(|rt| {
        // Every product at its extreme: -128 * -128 = 16384, summed k times.
        let (m, n, k) = (128usize, 64usize, 1024usize);
        let a = vec![-128i8; m * k];
        let b = vec![-128i8; k * n];
        let ab = i8_buf(rt, &a);
        let bb = i8_buf(rt, &b);
        let cb = buf(rt, &vec![0.0f32; m * n]);

        nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None).expect("gemm_i8_dequant");
        rt.synchronize().unwrap();

        let want = (k as i64 * 16384) as f32;
        let got = cb.read_f32();
        for (e, g) in got.iter().take(m * n).enumerate() {
            assert_eq!(*g, want, "full-range accumulation at {e}");
        }
    });
}

#[test]
fn the_largest_accepted_k_is_exact_at_full_int8_range() {
    with_gpu(|rt| {
        // 131071 = floor(i32::MAX / 16384), the largest k at which k products of
        // (-128)*(-128) still fit an int32. The sum, 16384 * 131071, has 17
        // significant bits, so it survives the f32 store exactly.
        let (m, n, k) = (8usize, 8usize, 131_071usize);
        let ab = i8_buf(rt, &vec![-128i8; m * k]);
        let bb = i8_buf(rt, &vec![-128i8; k * n]);
        let cb = buf(rt, &vec![0.0f32; m * n]);

        nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None).expect("k = 131071");
        rt.synchronize().unwrap();

        let want = 16384i64 * k as i64;
        assert!(want <= i32::MAX as i64);
        let got = cb.read_f32();
        for (e, g) in got.iter().take(m * n).enumerate() {
            assert_eq!(*g as i64, want, "full-range accumulation at k = {k}, element {e}");
        }
    });
}

#[test]
fn k_at_two_to_the_seventeen_is_refused() {
    with_gpu(|rt| {
        // 16384 * 131072 = 2^31, one past i32::MAX: all -128 operands would wrap
        // the accumulator to -2^31. Every buffer is large enough that only the
        // k bound can refuse.
        let (m, n, k) = (8usize, 8usize, 131_072usize);
        let ab = rt.alloc_buffer(m * k).unwrap();
        let bb = rt.alloc_buffer(k * n).unwrap();
        let cb = rt.alloc_buffer(m * n * 4).unwrap();
        let err = nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None)
            .expect_err("k = 2^17 wraps an int32 accumulator");
        assert!(err.contains("overflow"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn a_k_that_could_overflow_int32_is_refused() {
    with_gpu(|rt| {
        let b = rt.alloc_buffer(1 << 20).unwrap();
        let c = rt.alloc_buffer(1 << 20).unwrap();
        // Past this k, full-range int8 products can wrap the int32 accumulator
        // silently. Refusing keeps the exactness claim true rather than nearly.
        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 8, 8, 200_000, 1.0, None).expect_err("k past the exact range");
        assert!(err.contains("overflow"), "{err}");

        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 0, 8, 8, 1.0, None).expect_err("m = 0");
        assert!(err.contains("non-zero"), "{err}");

        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 8, 8, 8, f32::NAN, None).expect_err("non-finite scale");
        assert!(err.contains("finite"), "{err}");
    });
}

#[test]
fn undersized_operands_are_refused_before_dispatch() {
    with_gpu(|rt| {
        let small = rt.alloc_buffer(16).unwrap();
        let big = rt.alloc_buffer(1 << 20).unwrap();
        let err = nn::gemm_i8_dequant(rt, &small, &big, &big, 128, 64, 64, 1.0, None).expect_err("A too small");
        assert!(err.contains("buffer holds"), "{err}");
        let scale = rt.alloc_buffer(4).unwrap();
        let err =
            nn::gemm_i8_dequant(rt, &big, &big, &big, 128, 64, 64, 1.0, Some(&scale)).expect_err("scale too short");
        assert!(err.contains("buffer holds"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

fn seq_i8(n: usize, seed: i32) -> Vec<i8> {
    (0..n)
        .map(|i| ((i as i32).wrapping_mul(13).wrapping_add(seed) % 17 - 8) as i8)
        .collect()
}

fn f32_buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    buf(rt, data)
}

fn out_buf(rt: &Arc<GpuRuntime>, elems: usize, out: nn::I8Out) -> GpuBuffer {
    let bytes = elems * if out == nn::I8Out::F32 { 4 } else { 2 };
    let b = rt.alloc_buffer(bytes.max(4)).expect("alloc");
    b.write_bytes(&vec![0u8; bytes.max(1)]);
    b
}

fn read_bits(buf: &GpuBuffer, n: usize) -> Vec<u16> {
    let mapped = buf.contents_u16();
    mapped[..n].to_vec()
}

/// Independent of the shader: int32 dots, then the kernel's f32 flush order.
fn oracle_full(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    nt: bool,
    a_rows: Option<&[f32]>,
    a_scalar: f32,
    b_col: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let ascl = a_rows.map(|rows| rows[i]).unwrap_or(a_scalar);
        for j in 0..n {
            let mut s: i32 = 0;
            for p in 0..k {
                let bv = if nt { b[j * k + p] } else { b[p * n + j] };
                s += a[i * k + p] as i32 * bv as i32;
            }
            let bscl = b_col.map(|sc| sc[j]).unwrap_or(1.0);
            let mut v = (s as f32 * ascl) * bscl;
            if let Some(bias) = bias {
                v += bias[j];
            }
            out[i * n + j] = v;
        }
    }
    out
}

fn oracle_group(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    group: usize,
    nt: bool,
    a_rows: Option<&[f32]>,
    a_scalar: f32,
    b_scale: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let groups = k / group;
    for i in 0..m {
        let ascl = a_rows.map(|rows| rows[i]).unwrap_or(a_scalar);
        for j in 0..n {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let mut s: i32 = 0;
                for p in 0..group {
                    let kk = g * group + p;
                    let bv = if nt { b[j * k + kk] } else { b[kk * n + j] };
                    s += a[i * k + kk] as i32 * bv as i32;
                }
                acc += s as f32 * b_scale[g * n + j];
            }
            acc *= ascl;
            if let Some(bias) = bias {
                acc += bias[j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what} [{i}]: got {g} want {w}");
    }
}

struct Case<'a> {
    a: &'a [i8],
    b: &'a [i8],
    m: u32,
    n: u32,
    k: u32,
    layout: nn::I8BLayout,
    a_scale: nn::I8AScale<'a>,
    b_scale: nn::I8BScale<'a>,
    bias: Option<&'a GpuBuffer>,
    out: nn::I8Out,
}

fn run_case(rt: &Arc<GpuRuntime>, case: Case<'_>) -> GpuBuffer {
    let elems = case.m as usize * case.n as usize;
    let ab = i8_buf(rt, case.a);
    let bb = i8_buf(rt, case.b);
    let cb = out_buf(rt, elems, case.out);
    nn::gemm_i8(
        rt,
        nn::I8Gemm {
            a: &ab,
            b: &bb,
            c: &cb,
            m: case.m,
            n: case.n,
            k: case.k,
            a_scale: case.a_scale,
            b_scale: case.b_scale,
            bias: case.bias,
            b_layout: case.layout,
            out: case.out,
        },
    )
    .expect("gemm_i8");
    cb
}

fn quant_row(row: &[f32]) -> Result<(Vec<i8>, f32), ()> {
    if row.iter().any(|v| !v.is_finite()) {
        return Err(());
    }
    let mut absmax = 0.0f32;
    for v in row {
        absmax = absmax.max(v.abs());
    }
    let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
    let q = row
        .iter()
        .map(|v| {
            let r = (*v / scale).round_ties_even();
            r.clamp(-127.0, 127.0) as i8
        })
        .collect();
    Ok((q, scale))
}

#[test]
fn pipeline_names_select_gemv_small_tile_and_wide_tile() {
    assert_eq!(
        nn::i8_gemm_pipeline_name(1, nn::I8BLayout::RowMajorKN, 0).unwrap(),
        "i8_gemv_dequant"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(1, nn::I8BLayout::TransposedNK, 128).unwrap(),
        "i8_gemv_dequant"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(2, nn::I8BLayout::RowMajorKN, 0).unwrap(),
        "i8_w8a8_64"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(64, nn::I8BLayout::TransposedNK, 0).unwrap(),
        "i8_w8a8_64_nt"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(65, nn::I8BLayout::RowMajorKN, 0).unwrap(),
        "i8_w8a8_128"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(96, nn::I8BLayout::TransposedNK, 64).unwrap(),
        "i8_w8a8_g64_128_nt"
    );
    assert_eq!(
        nn::i8_gemm_pipeline_name(64, nn::I8BLayout::RowMajorKN, 256).unwrap(),
        "i8_w8a8_g256_64"
    );
    let err = nn::i8_gemm_pipeline_name(0, nn::I8BLayout::RowMajorKN, 0).unwrap_err();
    assert!(err.contains("non-zero"), "{err}");
    let err = nn::i8_gemm_pipeline_name(4, nn::I8BLayout::RowMajorKN, 32).unwrap_err();
    assert!(err.contains("32"), "{err}");
    assert!(err.contains("group"), "{err}");
}

#[test]
fn per_row_quantize_matches_the_round_ties_even_oracle() {
    with_gpu(|rt| {
        let rows = [vec![0.0f32, 0.0, 0.0, 0.0], vec![127.0, -63.5, 63.5, 0.0]];
        let cols = 4u32;
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let x = f32_buf(rt, &flat);
        let q = rt.alloc_buffer(flat.len()).unwrap();
        q.write_bytes(&vec![0x7fu8; flat.len()]);
        let scales = f32_buf(rt, &[99.0, 99.0]);
        nn::quantize_i8_rows(rt, &x, &q, &scales, rows.len() as u32, cols).expect("quantize");
        let got_q: Vec<u8> = q.contents_u8()[..flat.len()].to_vec();
        let got_s = scales.read_f32();
        for (r, row) in rows.iter().enumerate() {
            let (want_q, want_s) = quant_row(row).unwrap();
            assert_eq!(got_s[r].to_bits(), want_s.to_bits(), "row {r} scale");
            for (c, wq) in want_q.iter().enumerate() {
                assert_eq!(got_q[r * 4 + c] as i8, *wq, "row {r} col {c}");
            }
        }

        let mut wide = vec![0.0f32; 300];
        wide[0] = 1.0;
        wide[1] = 3.0;
        wide[299] = 254.0;
        let x = f32_buf(rt, &wide);
        let q = rt.alloc_buffer(wide.len()).unwrap();
        let scales = f32_buf(rt, &[0.0]);
        nn::quantize_i8_rows(rt, &x, &q, &scales, 1, 300).expect("wide quantize");
        let (want_q, want_s) = quant_row(&wide).unwrap();
        let got: Vec<u8> = q.contents_u8()[..wide.len()].to_vec();
        assert_eq!(scales.read_f32()[0].to_bits(), want_s.to_bits());
        for (i, wq) in want_q.iter().enumerate() {
            assert_eq!(got[i] as i8, *wq, "wide col {i}");
        }

        let bad = f32_buf(rt, &[1.0, f32::NAN]);
        let q = rt.alloc_buffer(8).unwrap();
        let scales = f32_buf(rt, &[0.0]);
        let err = nn::quantize_i8_rows(rt, &bad, &q, &scales, 1, 2).expect_err("nan row");
        assert!(err.contains("non-finite"), "{err}");

        rt.take_dispatch_count();
        let err = nn::quantize_i8_rows(rt, &x, &q, &scales, 0, 4).expect_err("zero rows");
        assert!(err.contains("non-zero"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn scalar_f32_matches_legacy_dequant_and_the_integer_oracle() {
    with_gpu(|rt| {
        for &(m, n, k) in &[(37u32, 45, 33), (96, 45, 33), (64, 17, 13), (128, 20, 15)] {
            let a = i8_data((m * k) as usize, 0x11 + k as u64);
            let b = i8_data((k * n) as usize, 0x22 + n as u64);
            let ab = i8_buf(rt, &a);
            let bb = i8_buf(rt, &b);
            let legacy = buf(rt, &vec![0.0f32; (m * n) as usize]);
            nn::gemm_i8_dequant(rt, &ab, &bb, &legacy, m, n, k, 1.0, None).expect("legacy");
            let fresh = run_case(
                rt,
                Case {
                    a: &a,
                    b: &b,
                    m,
                    n,
                    k,
                    layout: nn::I8BLayout::RowMajorKN,
                    a_scale: nn::I8AScale::Scalar(1.0),
                    b_scale: nn::I8BScale::None,
                    bias: None,
                    out: nn::I8Out::F32,
                },
            );
            let want = oracle_full(&a, &b, m as usize, n as usize, k as usize, false, None, 1.0, None, None);
            assert_exact(&legacy.read_f32(), &want, &format!("legacy {m}x{n}x{k}"));
            assert_exact(&fresh.read_f32(), &want, &format!("gemm_i8 {m}x{n}x{k}"));
        }
    });
}

#[test]
fn per_row_scale_transposed_b_bias_and_per_column_match_the_integer_oracle() {
    with_gpu(|rt| {
        for &(m, n, k, nt) in &[
            (1u32, 5, 9, false),
            (7, 5, 13, true),
            (64, 6, 11, false),
            (65, 6, 11, true),
            (96, 7, 17, true),
        ] {
            let a = seq_i8((m * k) as usize, 3);
            let b = seq_i8((n * k) as usize, 9);
            let rows: Vec<f32> = (0..m).map(|i| [0.5, 1.0, 2.0, 4.0][i as usize % 4]).collect();
            let cols: Vec<f32> = (0..n).map(|j| [1.0, 0.5, 2.0][j as usize % 3]).collect();
            let bias_v: Vec<f32> = (0..n).map(|j| if j % 2 == 0 { 0.25 } else { -0.5 }).collect();
            let row_buf = f32_buf(rt, &rows);
            let col_buf = f32_buf(rt, &cols);
            let bias = f32_buf(rt, &bias_v);
            let layout = if nt {
                nn::I8BLayout::TransposedNK
            } else {
                nn::I8BLayout::RowMajorKN
            };
            let b_matrix = if nt {
                b.clone()
            } else {
                let mut kn = vec![0i8; (k * n) as usize];
                for p in 0..k as usize {
                    for j in 0..n as usize {
                        kn[p * n as usize + j] = b[j * k as usize + p];
                    }
                }
                kn
            };
            let got = run_case(
                rt,
                Case {
                    a: &a,
                    b: &b_matrix,
                    m,
                    n,
                    k,
                    layout,
                    a_scale: nn::I8AScale::Rows(&row_buf),
                    b_scale: nn::I8BScale::PerColumn(&col_buf),
                    bias: Some(&bias),
                    out: nn::I8Out::F32,
                },
            );
            let want = oracle_full(
                &a,
                &b_matrix,
                m as usize,
                n as usize,
                k as usize,
                nt,
                Some(&rows),
                1.0,
                Some(&cols),
                Some(&bias_v),
            );
            assert_exact(&got.read_f32(), &want, &format!("rows/cols {m}x{n}x{k} nt={nt}"));
            assert!(rt.take_dispatch_count() > 0);
        }
    });
}

#[test]
fn group_scales_flush_in_ascending_group_order() {
    with_gpu(|rt| {
        for &(m, n, k, group, nt) in &[
            (2u32, 3, 128, 64, false),
            (2, 3, 128, 64, true),
            (1, 4, 256, 128, true),
            (80, 3, 256, 256, false),
            (70, 5, 256, 64, true),
            (3, 4, 256, 128, false),
        ] {
            let a = vec![1i8; (m * k) as usize];
            let b_logical = vec![1i8; (n * k) as usize];
            let groups = (k / group) as usize;
            let mut scales = vec![0.0f32; groups * n as usize];
            for g in 0..groups {
                for j in 0..n as usize {
                    scales[g * n as usize + j] = [1.0, 0.1, 0.5, 0.25][g % 4];
                }
            }
            let b_matrix = if nt {
                b_logical.clone()
            } else {
                let mut kn = vec![0i8; (k * n) as usize];
                for p in 0..k as usize {
                    for j in 0..n as usize {
                        kn[p * n as usize + j] = b_logical[j * k as usize + p];
                    }
                }
                kn
            };
            let sb = f32_buf(rt, &scales);
            let layout = if nt {
                nn::I8BLayout::TransposedNK
            } else {
                nn::I8BLayout::RowMajorKN
            };
            let got = run_case(
                rt,
                Case {
                    a: &a,
                    b: &b_matrix,
                    m,
                    n,
                    k,
                    layout,
                    a_scale: nn::I8AScale::Scalar(1.0),
                    b_scale: nn::I8BScale::PerGroup { scales: &sb, group },
                    bias: None,
                    out: nn::I8Out::F32,
                },
            );
            let want = oracle_group(
                &a,
                &b_matrix,
                m as usize,
                n as usize,
                k as usize,
                group as usize,
                nt,
                None,
                1.0,
                &scales,
                None,
            );
            assert_exact(&got.read_f32(), &want, &format!("group {group} {m}x{n}x{k} nt={nt}"));
            if groups > 1 {
                let summed_scales: f32 = scales.iter().step_by(n as usize).take(groups).sum();
                let collapsed = (k as f32) * summed_scales;
                assert_ne!(
                    want[0].to_bits(),
                    collapsed.to_bits(),
                    "group flush collapsed into one multiply"
                );
            }
        }
    });
}

#[test]
fn f16_and_bf16_stores_match_the_gpu_cast() {
    with_gpu(|rt| {
        let (m, n, k) = (3u32, 5, 4);
        let a = vec![1i8; (m * k) as usize];
        let b = vec![1i8; (k * n) as usize];
        let bias_v = vec![0.5f32; n as usize];
        let bias = f32_buf(rt, &bias_v);
        let want = oracle_full(
            &a,
            &b,
            m as usize,
            n as usize,
            k as usize,
            false,
            None,
            1.0,
            None,
            Some(&bias_v),
        );
        for out in [nn::I8Out::F16, nn::I8Out::Bf16] {
            let got = run_case(
                rt,
                Case {
                    a: &a,
                    b: &b,
                    m,
                    n,
                    k,
                    layout: nn::I8BLayout::RowMajorKN,
                    a_scale: nn::I8AScale::Scalar(1.0),
                    b_scale: nn::I8BScale::None,
                    bias: Some(&bias),
                    out,
                },
            );
            let bits = read_bits(&got, (m * n) as usize);
            for (i, w) in want.iter().enumerate() {
                let expect = if out == nn::I8Out::F16 {
                    tessl::tensor::f32_to_f16_bits(*w)
                } else {
                    tessl::tensor::f32_to_bf16_bits_gpu(*w)
                };
                assert_eq!(bits[i], expect, "{out:?} element {i}: f32 {w}");
            }
        }
    });
}

#[test]
fn m1_gemv_and_small_m_tile_match_the_integer_oracle() {
    let Some(out) = common::in_traced_child("m1_gemv_and_small_m_tile_match_the_integer_oracle", || {
        with_gpu(|rt| {
            let specs = [
                (1u32, 6u32, 9u32, false, "i8_gemv_dequant"),
                (64, 5, 7, false, "i8_w8a8_64"),
                (65, 5, 7, false, "i8_w8a8_128"),
                (7, 4, 9, true, "i8_w8a8_64_nt"),
                (3, 4096, 2048, false, "i8_w8a8_64"),
            ];
            for (m, n, k, nt, name) in specs {
                let before = tessl::runtime::traced_kernel_count(name);
                let a = if k > 64 {
                    vec![1i8; (m * k) as usize]
                } else {
                    seq_i8((m * k) as usize, 1)
                };
                let b = if k > 64 {
                    let mut kn = vec![1i8; (k * n) as usize];
                    if nt {
                        kn = vec![1i8; (n * k) as usize];
                    }
                    kn
                } else if nt {
                    seq_i8((n * k) as usize, 2)
                } else {
                    let logical = seq_i8((n * k) as usize, 2);
                    let mut kn = vec![0i8; (k * n) as usize];
                    for p in 0..k as usize {
                        for j in 0..n as usize {
                            kn[p * n as usize + j] = logical[j * k as usize + p];
                        }
                    }
                    kn
                };
                let got = run_case(
                    rt,
                    Case {
                        a: &a,
                        b: &b,
                        m,
                        n,
                        k,
                        layout: if nt {
                            nn::I8BLayout::TransposedNK
                        } else {
                            nn::I8BLayout::RowMajorKN
                        },
                        a_scale: nn::I8AScale::Scalar(1.0),
                        b_scale: nn::I8BScale::None,
                        bias: None,
                        out: nn::I8Out::F32,
                    },
                );
                let want = oracle_full(&a, &b, m as usize, n as usize, k as usize, nt, None, 1.0, None, None);
                assert_exact(&got.read_f32(), &want, name);
                let after = tessl::runtime::traced_kernel_count(name);
                println!("TRACE {name} delta={}", after - before);
                assert!(after > before, "{name} was not requested");
                assert_eq!(tessl::runtime::traced_kernel_count("matmul2d_tensorops_i8_f32"), 0);
            }
        });
    }) else {
        return;
    };
    assert!(out.contains("TRACE i8_gemv_dequant delta=1"), "{out}");
    assert!(
        out.contains("TRACE i8_w8a8_64 delta=1") || out.contains("TRACE i8_w8a8_64 delta=2"),
        "{out}"
    );
    assert!(out.contains("TRACE i8_w8a8_128 delta=1"), "{out}");
    assert!(out.contains("TRACE i8_w8a8_64_nt delta=1"), "{out}");
}

#[test]
fn two_runs_are_bit_identical() {
    with_gpu(|rt| {
        let (m, n, k) = (9u32, 6, 128);
        let a = seq_i8((m * k) as usize, 4);
        let b = seq_i8((k * n) as usize, 8);
        let groups = (k / 64) as usize;
        let mut scales = vec![0.0f32; groups * n as usize];
        for j in 0..n as usize {
            scales[j] = 1.0;
            scales[n as usize + j] = 0.1;
        }
        let sb = f32_buf(rt, &scales);
        let mut first = Vec::new();
        for _ in 0..2 {
            let got = run_case(
                rt,
                Case {
                    a: &a,
                    b: &b,
                    m,
                    n,
                    k,
                    layout: nn::I8BLayout::RowMajorKN,
                    a_scale: nn::I8AScale::Scalar(0.5),
                    b_scale: nn::I8BScale::PerGroup { scales: &sb, group: 64 },
                    bias: None,
                    out: nn::I8Out::F32,
                },
            );
            first.push(got.read_f32());
        }
        assert_exact(&first[0], &first[1], "rerun");
        let want = oracle_group(
            &a, &b, m as usize, n as usize, k as usize, 64, false, None, 0.5, &scales, None,
        );
        assert_exact(&first[0], &want, "group rerun vs oracle");
    });
}

#[test]
fn non_finite_scale_is_refused_without_a_dispatch() {
    with_gpu(|rt| {
        let b = rt.alloc_buffer(64).unwrap();
        let c = rt.alloc_buffer(64).unwrap();
        rt.take_dispatch_count();
        let err = nn::gemm_i8(
            rt,
            nn::I8Gemm {
                a: &b,
                b: &b,
                c: &c,
                m: 2,
                n: 2,
                k: 2,
                a_scale: nn::I8AScale::Scalar(f32::NAN),
                b_scale: nn::I8BScale::None,
                bias: None,
                b_layout: nn::I8BLayout::RowMajorKN,
                out: nn::I8Out::F32,
            },
        )
        .expect_err("nan scale");
        assert!(err.contains("finite"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::gemm_i8(
            rt,
            nn::I8Gemm {
                a: &b,
                b: &b,
                c: &c,
                m: 0,
                n: 2,
                k: 2,
                a_scale: nn::I8AScale::Scalar(1.0),
                b_scale: nn::I8BScale::None,
                bias: None,
                b_layout: nn::I8BLayout::RowMajorKN,
                out: nn::I8Out::F32,
            },
        )
        .expect_err("empty");
        assert!(err.contains("non-zero"), "{err}");

        let scales = f32_buf(rt, &[1.0]);
        let err = nn::gemm_i8(
            rt,
            nn::I8Gemm {
                a: &b,
                b: &b,
                c: &c,
                m: 2,
                n: 2,
                k: 32,
                a_scale: nn::I8AScale::Scalar(1.0),
                b_scale: nn::I8BScale::PerGroup {
                    scales: &scales,
                    group: 48,
                },
                bias: None,
                b_layout: nn::I8BLayout::RowMajorKN,
                out: nn::I8Out::F32,
            },
        )
        .expect_err("bad group");
        assert!(err.contains("48"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::gemm_i8(
            rt,
            nn::I8Gemm {
                a: &b,
                b: &b,
                c: &c,
                m: 2,
                n: 2,
                k: 131_072,
                a_scale: nn::I8AScale::Scalar(1.0),
                b_scale: nn::I8BScale::None,
                bias: None,
                b_layout: nn::I8BLayout::RowMajorKN,
                out: nn::I8Out::F32,
            },
        )
        .expect_err("k past exact");
        assert!(err.contains("131071") || err.contains("exceeds"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
