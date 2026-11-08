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
#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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

/// The refusal wording is a contract other crates may match on. Pinned
/// byte-for-byte, so moving the checks into a shared validator cannot reword it.
#[test]
fn refusal_messages_are_pinned_byte_for_byte() {
    with_gpu(|rt| {
        let big = rt.alloc_buffer(1 << 20).unwrap();
        big.write_bytes(&vec![0u8; 1 << 20]);
        let tiny = rt.alloc_buffer(64).unwrap();
        tiny.write_bytes(&[0u8; 64]);
        let spec = |a, b, c, m, n, k| nn::I8Gemm {
            a,
            b,
            c,
            m,
            n,
            k,
            a_scale: nn::I8AScale::Scalar(1.0),
            b_scale: nn::I8BScale::None,
            bias: None,
            b_layout: nn::I8BLayout::RowMajorKN,
            out: nn::I8Out::F32,
        };
        rt.take_dispatch_count();
        let e = nn::gemm_i8(rt, spec(&big, &big, &big, 2, 2, 131_072)).expect_err("k bound");
        assert_eq!(
            e,
            "gemm_i8: k = 131072 exceeds 131071, past which an int32 accumulator can overflow at full int8 range"
        );
        let e = nn::gemm_i8(rt, spec(&big, &big, &big, 2, 2, 32)).expect_err("alias");
        assert_eq!(e, "gemm_i8: writable buffer c overlaps read-only buffer a");
        let e = nn::gemm_i8(rt, spec(&tiny, &tiny, &tiny, 65_536, 8, 32_768)).expect_err("extent");
        assert_eq!(e, "gemm_i8 a exceeds signed 32-bit kernel indexing");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

// ---------------------------------------------------------------------------
// Coverage of every pipeline `i8_gemm_pipeline_name` can return.
// ---------------------------------------------------------------------------

const GROUP_DOMAIN: [u32; 4] = [0, 64, 128, 256];
const LAYOUT_DOMAIN: [nn::I8BLayout; 2] = [nn::I8BLayout::RowMajorKN, nn::I8BLayout::TransposedNK];
/// Upper bound of the M scan. Large enough to hold a tile row past the first
/// (200 > 128) so a wide kernel runs with `tiles_m > 1`.
const M_SCAN_MAX: u32 = 200;

/// Every `(m, layout, group)` the host can send to each pipeline name, derived
/// from `i8_gemm_pipeline_name` itself. For each `(layout, group)` the M scan
/// `1..=M_SCAN_MAX` is cut into runs of constant name; the first and last M of
/// every run are kept, so each threshold is crossed on both sides. No name is
/// written here: a pipeline the function gains tomorrow is covered by it.
fn pipeline_domain() -> std::collections::BTreeMap<&'static str, Vec<(u32, nn::I8BLayout, u32)>> {
    let mut by_name: std::collections::BTreeMap<&'static str, Vec<(u32, nn::I8BLayout, u32)>> = Default::default();
    for &layout in &LAYOUT_DOMAIN {
        for &group in &GROUP_DOMAIN {
            let names: Vec<&'static str> = (1..=M_SCAN_MAX)
                .map(|m| nn::i8_gemm_pipeline_name(m, layout, group).expect("domain call"))
                .collect();
            for (idx, name) in names.iter().enumerate() {
                let first = idx == 0 || names[idx - 1] != *name;
                let last = idx + 1 == names.len() || names[idx + 1] != *name;
                if first || last {
                    by_name.entry(name).or_default().push((idx as u32 + 1, layout, group));
                }
            }
        }
    }
    by_name
}

fn nk_to_kn(logical: &[i8], n: usize, k: usize) -> Vec<i8> {
    let mut kn = vec![0i8; k * n];
    for p in 0..k {
        for j in 0..n {
            kn[p * n + j] = logical[j * k + p];
        }
    }
    kn
}

/// Run one `(m, layout, group)` with seeded full-range A and B, per-column
/// (or per-group-per-column) distinct scales, per-row A scales, a bias, and
/// every output type; compare each to the CPU oracle bit for bit.
fn run_domain_point(rt: &Arc<GpuRuntime>, m: u32, layout: nn::I8BLayout, group: u32) {
    // 70 columns: two 64-wide tile columns with a ragged second one.
    let (n, k) = (70u32, 512u32);
    let nt = layout == nn::I8BLayout::TransposedNK;
    let seed = 0x5eed ^ ((m as u64) << 8) ^ ((group as u64) << 20) ^ nt as u64;
    let a = i8_data((m * k) as usize, seed);
    let logical = i8_data((n * k) as usize, seed ^ 0xb0b);
    let b = if nt {
        logical
    } else {
        nk_to_kn(&logical, n as usize, k as usize)
    };
    let rows: Vec<f32> = (0..m)
        .map(|i| [0.25, 0.5, 0.75, 1.0, 1.5, 0.125, 2.0][i as usize % 7])
        .collect();
    let bias_v: Vec<f32> = (0..n).map(|j| (j as f32 - 35.0) * 0.375).collect();
    let groups = k.checked_div(group).unwrap_or(1);
    // Distinct per column and per group, exactly representable.
    let scales: Vec<f32> = (0..groups * n)
        .map(|i| ((i * 7 + 3) % 251 + 1) as f32 / 2048.0)
        .collect();
    let row_buf = f32_buf(rt, &rows);
    let bias = f32_buf(rt, &bias_v);
    let sb = f32_buf(rt, &scales);
    let tag = format!("m={m} {layout:?} group={group}");

    for out in [nn::I8Out::F32, nn::I8Out::F16, nn::I8Out::Bf16] {
        let b_scale = if group == 0 {
            nn::I8BScale::PerColumn(&sb)
        } else {
            nn::I8BScale::PerGroup { scales: &sb, group }
        };
        let got = run_case(
            rt,
            Case {
                a: &a,
                b: &b,
                m,
                n,
                k,
                layout,
                a_scale: nn::I8AScale::Rows(&row_buf),
                b_scale,
                bias: Some(&bias),
                out,
            },
        );
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let want = if group == 0 {
            oracle_full(&a, &b, mu, nu, ku, nt, Some(&rows), 1.0, Some(&scales), Some(&bias_v))
        } else {
            oracle_group(
                &a,
                &b,
                mu,
                nu,
                ku,
                group as usize,
                nt,
                Some(&rows),
                1.0,
                &scales,
                Some(&bias_v),
            )
        };
        rt.synchronize().unwrap();
        match out {
            nn::I8Out::F32 => assert_exact(&got.read_f32(), &want, &format!("{tag} f32")),
            nn::I8Out::F16 | nn::I8Out::Bf16 => {
                let bits = read_bits(&got, want.len());
                for (i, w) in want.iter().enumerate() {
                    let expect = if out == nn::I8Out::F16 {
                        tessl::tensor::f32_to_f16_bits(*w)
                    } else {
                        tessl::tensor::f32_to_bf16_bits_gpu(*w)
                    };
                    assert_eq!(bits[i], expect, "{tag} {out:?} [{i}]: f32 {w}");
                }
            }
        }
    }

    // Scalar activation scale, no bias, f32: the other arm of each enum.
    let b_scale = if group == 0 {
        nn::I8BScale::None
    } else {
        nn::I8BScale::PerGroup { scales: &sb, group }
    };
    let got = run_case(
        rt,
        Case {
            a: &a,
            b: &b,
            m,
            n,
            k,
            layout,
            a_scale: nn::I8AScale::Scalar(0.75),
            b_scale,
            bias: None,
            out: nn::I8Out::F32,
        },
    );
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let want = if group == 0 {
        oracle_full(&a, &b, mu, nu, ku, nt, None, 0.75, None, None)
    } else {
        oracle_group(&a, &b, mu, nu, ku, group as usize, nt, None, 0.75, &scales, None)
    };
    assert_exact(&got.read_f32(), &want, &format!("{tag} scalar scale"));
}

/// Every pipeline name the host can pick dispatches, and matches the oracle.
/// The names come from [`pipeline_domain`], not a list written here. The child
/// runs with `TESSL_KERNEL_TRACE=1` and prints one `COVERED <name>` line per
/// name whose dispatch count rose; the parent compares those lines to the
/// domain it derives itself. Wide kernels run with `tiles_m > 1` and
/// `tiles_n > 1` together (M = 200, N = 70).
#[test]
fn every_int8_pipeline_name_dispatches_and_matches_the_oracle() {
    let domain = pipeline_domain();
    // 1 GEMV + {64, 128} tile x {NN, NT} x groups {0, 64, 128, 256}.
    assert_eq!(domain.len(), 17, "pipeline names: {:?}", domain.keys());
    assert!(
        domain
            .values()
            .flatten()
            .any(|&(m, _, _)| m.div_ceil(128) > 1 && m > 64),
        "no wide case with tiles_m > 1"
    );
    let Some(out) = common::in_traced_child("every_int8_pipeline_name_dispatches_and_matches_the_oracle", || {
        with_gpu(|rt| {
            assert!(tessl::runtime::kernel_trace_enabled());
            for (name, points) in &domain {
                let before = tessl::runtime::traced_kernel_count(name);
                for &(m, layout, group) in points {
                    assert_eq!(nn::i8_gemm_pipeline_name(m, layout, group).unwrap(), *name);
                    run_domain_point(rt, m, layout, group);
                }
                let after = tessl::runtime::traced_kernel_count(name);
                assert!(
                    after - before >= points.len() as u64,
                    "{name}: {} points but {} dispatches traced",
                    points.len(),
                    after - before
                );
                println!("COVERED {name}");
            }
        });
    }) else {
        return;
    };
    for name in domain.keys() {
        assert!(out.contains(&format!("COVERED {name}\n")), "{name} not covered:\n{out}");
    }
}

// ---------------------------------------------------------------------------
// f16 output overflows; documented on `I8Out::F16`, pinned here.
// ---------------------------------------------------------------------------

#[test]
fn f16_output_overflows_to_infinity_for_finite_inputs() {
    with_gpu(|rt| {
        let n = 2u32;
        for m in [1u32, 2, 65] {
            for (fill, label) in [(1i8, "+"), (-1i8, "-")] {
                let run = |k_use: u32, out: nn::I8Out| {
                    let a_k = vec![fill; (m * k_use) as usize];
                    let b_k = vec![1i8; (k_use * n) as usize];
                    let got = run_case(
                        rt,
                        Case {
                            a: &a_k,
                            b: &b_k,
                            m,
                            n,
                            k: k_use,
                            layout: nn::I8BLayout::RowMajorKN,
                            a_scale: nn::I8AScale::Scalar(1.0),
                            b_scale: nn::I8BScale::None,
                            bias: None,
                            out,
                        },
                    );
                    rt.synchronize().unwrap();
                    got
                };
                let sign: f32 = if fill > 0 { 1.0 } else { -1.0 };
                // 65504 is the largest finite half and is exact.
                let at_max = read_bits(&run(65_504, nn::I8Out::F16), (m * n) as usize);
                let want_max = tessl::tensor::f32_to_f16_bits(sign * 65_504.0);
                assert!(
                    at_max.iter().all(|&v| v == want_max),
                    "{label} m={m} k=65504: {at_max:?}"
                );
                assert_eq!(want_max & 0x7fff, 0x7bff);
                // Past it the finite f32 sum 70000 stores as +-inf. No flag, no error.
                let over = read_bits(&run(70_000, nn::I8Out::F16), (m * n) as usize);
                let inf_bits: u16 = if fill > 0 { 0x7c00 } else { 0xfc00 };
                assert!(
                    over.iter().all(|&v| v == inf_bits),
                    "{label} m={m} k=70000 f16: {over:?}"
                );
                // The same sum is finite and exact in f32, and finite in bf16.
                let f32_out = run(70_000, nn::I8Out::F32).read_f32();
                assert!(
                    f32_out.iter().take((m * n) as usize).all(|&v| v == sign * 70_000.0),
                    "{f32_out:?}"
                );
                let bf = read_bits(&run(70_000, nn::I8Out::Bf16), (m * n) as usize);
                let want_bf = tessl::tensor::f32_to_bf16_bits_gpu(sign * 70_000.0);
                assert!(bf.iter().all(|&v| v == want_bf), "{label} m={m} bf16: {bf:?}");
            }
        }
    });
}

// ---------------------------------------------------------------------------
// i8_quantize_rows on general data, and under many threadgroups.
// ---------------------------------------------------------------------------

/// Seeded rows spanning magnitudes and signs; every 11th row all zeros.
fn random_rows(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let mag = [0.001f32, 0.37, 1.0, 13.0, 4096.0][r % 5];
        for _ in 0..cols {
            out.push(if r % 11 == 0 { 0.0 } else { next() * mag });
        }
    }
    out
}

/// Check one quantized row against the IEEE host reference.
///
/// Tolerance, and why. The kernel is built with fast math. Measured on this
/// device: `red[0] / 127.0f` is not the correctly rounded quotient for every
/// absmax (a reciprocal multiply), so the scale may differ from the host
/// `absmax / 127` by 1 ulp; it is checked as at most 1 ulp apart, never more.
/// `q` is then checked against the reference built from the scale the kernel
/// actually wrote, so the scale's own error is not double counted. The
/// kernel's `x / scale` may likewise be off by a few ulp of the quotient
/// `r = x / scale`. That can move `r` across a `.5` boundary and no other
/// integer boundary, so `q` may differ from `rint(r)` by 1, and only when
/// `|r - (floor(r) + 0.5)| <= 8 * EPSILON * |r|` (8 ulp, at most 1.2e-4 for
/// |r| <= 127). Everywhere else `q` must be identical. Returns how many
/// elements used that allowance.
fn check_quant_row(row: &[f32], got_q: &[u8], got_scale: f32, tag: &str) -> usize {
    let (_, host_s) = quant_row(row).expect("finite row");
    let ulps = (got_scale.to_bits() as i64 - host_s.to_bits() as i64).abs();
    assert!(ulps <= 1, "{tag} scale {got_scale} vs host {host_s}: {ulps} ulp");
    if ulps == 1 {
        // Classification: the absmax is exact (an fmax of |x|), so the gap is
        // the divide. `recip` is the reciprocal-multiply form fast math emits.
        let absmax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let recip = absmax * (1.0f32 / 127.0);
        println!(
            "SCALE1ULP {tag}: absmax bits {:#010x} kernel scale {:#010x} host absmax/127 {:#010x} absmax*(1/127) {:#010x} kernel==recip {}",
            absmax.to_bits(),
            got_scale.to_bits(),
            host_s.to_bits(),
            recip.to_bits(),
            got_scale.to_bits() == recip.to_bits()
        );
    }
    let mut slack = 0usize;
    for (c, x) in row.iter().enumerate() {
        let got = got_q[c] as i8;
        let r = *x / got_scale;
        let want = r.round_ties_even().clamp(-127.0, 127.0) as i8;
        if got == want {
            continue;
        }
        let dist = (r - (r.floor() + 0.5)).abs();
        assert!(
            (got as i32 - want as i32).abs() == 1 && dist <= 8.0 * f32::EPSILON * r.abs(),
            "{tag} col {c}: x {x} scale {got_scale} ratio {r} got {got} want {want} (dist to .5 {dist})"
        );
        slack += 1;
    }
    slack
}

#[test]
fn per_row_quantize_on_random_rows_is_within_the_stated_tolerance() {
    with_gpu(|rt| {
        let (rows, cols) = (64usize, 777usize);
        let flat = random_rows(rows, cols, 0xfeed);
        let x = f32_buf(rt, &flat);
        let q = rt.alloc_buffer(flat.len()).unwrap();
        let scales = f32_buf(rt, &vec![f32::NAN; rows]);
        nn::quantize_i8_rows(rt, &x, &q, &scales, rows as u32, cols as u32).expect("quantize");
        let got_q = q.contents_u8()[..flat.len()].to_vec();
        let got_s = scales.read_f32();
        let mut slack = 0;
        for r in 0..rows {
            slack += check_quant_row(
                &flat[r * cols..(r + 1) * cols],
                &got_q[r * cols..(r + 1) * cols],
                got_s[r],
                &format!("row {r}"),
            );
        }
        println!("elements using the 1-step allowance: {slack} of {}", rows * cols);
    });
}

/// Many threadgroups, repeated, with the outputs pre-filled with poison. The
/// kernel's second phase must see the scale and the bad flag from the
/// reduction of its own row. A stale scale (poison NaN) or a stale bad flag
/// (the host zero-fills `bad`, so a not-yet-visible flag reads 0 and a bad row
/// would be quantized) shows as a wrong q, scale or a written bad row.
///
/// Honest limit: the read-back-from-device-memory ordering gap this guards is
/// a memory-model hazard, not something Apple GPUs are known to expose in
/// practice; on a device where thread 0's device write is visible to its
/// threadgroup in time this test passes against the old kernel too. It checks
/// the contract at volume; the fix stands on the memory model. `bad` is
/// allocated inside `quantize_i8_rows`, so the poison goes into `q` and
/// `scales` only.
#[test]
fn quantize_rows_under_many_threadgroups_repeats_clean() {
    with_gpu(|rt| {
        let (rows, cols) = (4096usize, 515usize);
        let clean = random_rows(rows, cols, 0xc0ffee);
        let mut dirty = clean.clone();
        let bad_rows: Vec<usize> = (0..rows).filter(|r| r % 97 == 5).collect();
        for &r in &bad_rows {
            dirty[r * cols + (r * 31) % cols] = if r % 2 == 0 { f32::NAN } else { f32::INFINITY };
        }
        for iter in 0..6 {
            let with_bad = iter % 2 == 1;
            let data = if with_bad { &dirty } else { &clean };
            let x = f32_buf(rt, data);
            let q = rt.alloc_buffer(data.len()).unwrap();
            q.write_bytes(&vec![0x7fu8; data.len()]);
            let scales = f32_buf(rt, &vec![f32::NAN; rows]);
            let res = nn::quantize_i8_rows(rt, &x, &q, &scales, rows as u32, cols as u32);
            if with_bad {
                assert!(res.expect_err("bad rows").contains("non-finite"));
            } else {
                res.expect("clean");
            }
            let got_q = q.contents_u8()[..data.len()].to_vec();
            let got_s = scales.read_f32();
            for r in 0..rows {
                let tag = format!("iter {iter} row {r}");
                let qrow = &got_q[r * cols..(r + 1) * cols];
                if with_bad && bad_rows.contains(&r) {
                    assert!(qrow.iter().all(|&v| v == 0x7f), "{tag}: a non-finite row was written");
                } else {
                    check_quant_row(&data[r * cols..(r + 1) * cols], qrow, got_s[r], &tag);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Adversarial review (card ft-70f2141f): quantize edges, non-finite rows, a
// race hunt, and the f16 rounding boundary.
// ---------------------------------------------------------------------------

/// Quantize `flat` (`rows x cols`) with `q` pre-filled 0x55 and `scales`
/// pre-filled NaN, so an unwritten element is recognisable.
fn quantize_poisoned(
    rt: &Arc<GpuRuntime>,
    flat: &[f32],
    rows: usize,
    cols: usize,
) -> (Result<(), String>, Vec<u8>, Vec<f32>) {
    let x = f32_buf(rt, flat);
    let q = rt.alloc_buffer(flat.len()).unwrap();
    q.write_bytes(&vec![0x55u8; flat.len()]);
    let scales = f32_buf(rt, &vec![f32::NAN; rows]);
    let res = nn::quantize_i8_rows(rt, &x, &q, &scales, rows as u32, cols as u32);
    let got_q = q.contents_u8()[..flat.len()].to_vec();
    (res, got_q, scales.read_f32())
}

/// Rows built to hit the quantize corner cases at width `cols`.
fn edge_rows(cols: usize) -> Vec<(&'static str, Vec<f32>)> {
    let mut rows: Vec<(&'static str, Vec<f32>)> = Vec::new();
    rows.push(("zeros", vec![0.0; cols]));
    rows.push(("negative zeros", vec![-0.0; cols]));
    let mut huge_tiny = vec![1e-30f32; cols];
    huge_tiny[cols / 2] = 1e30;
    rows.push(("one huge, rest tiny", huge_tiny));
    let mut single = vec![0.0f32; cols];
    single[cols - 1] = -3.25;
    rows.push(("single nonzero in the last column", single));
    let ties = [127.0f32, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -126.5, 63.5, -64.5];
    rows.push((
        "half-way values at scale 1",
        (0..cols).map(|c| ties[c % ties.len()]).collect(),
    ));
    rows.push((
        "alternating +-max",
        (0..cols).map(|c| if c % 2 == 0 { 1000.0 } else { -1000.0 }).collect(),
    ));
    rows.push(("ramp", (0..cols).map(|c| c as f32 - cols as f32 / 2.0).collect()));
    rows
}

#[test]
fn adv_quantize_cols_sweep_holds_the_stated_tolerance_and_never_emits_minus_128() {
    with_gpu(|rt| {
        for cols in [1usize, 2, 3, 17, 255, 256, 257, 300, 511, 512, 513, 1000, 4097] {
            let rows = edge_rows(cols);
            let flat: Vec<f32> = rows.iter().flat_map(|(_, r)| r.iter().copied()).collect();
            let (res, got_q, got_s) = quantize_poisoned(rt, &flat, rows.len(), cols);
            res.unwrap_or_else(|e| panic!("cols {cols}: {e}"));
            for (i, (name, row)) in rows.iter().enumerate() {
                let qrow = &got_q[i * cols..(i + 1) * cols];
                println!(
                    "cols {cols} row `{name}`: scale {:#010x} ({}), q[0..4] {:?}",
                    got_s[i].to_bits(),
                    got_s[i],
                    &qrow[..qrow.len().min(4)]
                );
                assert!(qrow.iter().all(|&v| v != 0x80), "cols {cols} `{name}`: q holds -128");
                check_quant_row(row, qrow, got_s[i], &format!("cols {cols} `{name}`"));
            }
        }
    });
}

/// Rows whose `absmax` is below `127 * f32::MIN_POSITIVE` have a scale
/// (`absmax / 127`) below the smallest normal f32. Metal flushes f32 subnormals
/// to zero under fast and safe math alike (hardware; GitPulse card
/// ft-02eec315), so the kernel cannot produce that scale. Documented contract
/// (`quantize_i8_rows`, card ft-c68232d9): the written scale is a finite 0.0 (the
/// row's `q` saturate to +-127 or are 0) or 1.0 (an all-subnormal row read as
/// absmax 0, `q` all 0), and either way the dequantised row `q * scale` is
/// exactly 0, so the row contributes exactly 0 to a GEMM and never NaN or inf.
/// Rows at or above the threshold still match the host reference.
#[test]
fn adv_quantize_rows_below_the_normal_scale_range_contribute_exactly_zero() {
    with_gpu(|rt| {
        let cols = 300usize;
        // A subnormal f32.
        let sub = f32::from_bits(0x0000_1234);

        // absmax < 127 * f32::MIN_POSITIVE: the documented flush-to-zero range.
        let below: Vec<(&str, Vec<f32>)> = vec![
            ("all subnormal", (0..cols).map(|c| sub * (1 + c % 7) as f32).collect()),
            (
                "absmax is f32::MIN_POSITIVE",
                (0..cols)
                    .map(|c| f32::MIN_POSITIVE * (1 + c % 3) as f32 / 3.0)
                    .collect(),
            ),
            (
                "absmax 4e-37 (scale is subnormal)",
                (0..cols).map(|c| 1e-37 * ((c % 9) as f32 - 4.0)).collect(),
            ),
        ];
        // absmax >= 127 * f32::MIN_POSITIVE: host parity holds.
        let at_or_above: Vec<(&str, Vec<f32>)> = vec![
            (
                "subnormal with one normal 1e-3",
                (0..cols).map(|c| if c == 5 { 1e-3 } else { sub }).collect(),
            ),
            (
                "absmax 4e-36 (scale is normal)",
                (0..cols).map(|c| 1e-36 * ((c % 9) as f32 - 4.0)).collect(),
            ),
        ];
        let threshold = 127.0 * f32::MIN_POSITIVE;
        for (name, row) in &below {
            let absmax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(absmax < threshold, "`{name}` is not below the threshold");
            let (res, got_q, got_s) = quantize_poisoned(rt, row, 1, cols);
            res.unwrap_or_else(|e| panic!("`{name}`: {e}"));
            let scale = got_s[0];
            let head = &got_q[..6];
            println!("`{name}`: kernel scale {:#010x}; q[0..6] {head:?}", scale.to_bits());
            assert!(scale.is_finite(), "`{name}`: scale {scale} is not finite");
            assert!(
                scale == 0.0 || scale == 1.0,
                "`{name}`: scale {scale} is neither 0 nor 1"
            );
            assert!(got_q.iter().all(|&v| v != 0x80), "`{name}`: q holds -128 (or poison)");
            for (c, &q) in got_q.iter().enumerate() {
                let dq = f32::from(q as i8) * scale;
                assert!(
                    dq == 0.0,
                    "`{name}`: dequantised element {c} is {dq} (q {}, scale {scale})",
                    q as i8
                );
            }
        }
        for (name, row) in &at_or_above {
            let absmax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(absmax >= threshold, "`{name}` is not at or above the threshold");
            let (res, got_q, got_s) = quantize_poisoned(rt, row, 1, cols);
            res.unwrap_or_else(|e| panic!("`{name}`: {e}"));
            check_quant_row(row, &got_q, got_s[0], name);
        }
    });
}

#[test]
fn adv_quantize_absmax_at_f32_max_is_finite_and_in_range() {
    with_gpu(|rt| {
        let cols = 260usize;
        let row: Vec<f32> = (0..cols)
            .map(|c| match c % 4 {
                0 => f32::MAX,
                1 => -f32::MAX,
                2 => f32::MAX / 2.0,
                _ => 1.0,
            })
            .collect();
        let (res, got_q, got_s) = quantize_poisoned(rt, &row, 1, cols);
        res.expect("finite row");
        let head = &got_q[..4];
        println!("f32::MAX row: scale {:#010x} q[0..4] {head:?}", got_s[0].to_bits());
        assert!(got_s[0].is_finite() && got_s[0] > 0.0, "scale {}", got_s[0]);
        assert!(got_q.iter().all(|&v| v != 0x80));
        check_quant_row(&row, &got_q, got_s[0], "f32::MAX row");
    });
}

/// One non-finite element at the first, last or middle column, in a row of a
/// later threadgroup, at several widths. The error is returned; every clean
/// row is fully written and correct; a refused row's `q` is untouched; its
/// scale is the one of its finite elements (written, not poison).
#[test]
fn adv_quantize_single_non_finite_element_is_refused_and_isolated() {
    with_gpu(|rt| {
        for cols in [1usize, 2, 255, 256, 257, 777] {
            let rows = 3001usize;
            let clean = random_rows(rows, cols, 0xbad + cols as u64);
            let mut dirty = clean.clone();
            let mut bad: Vec<usize> = Vec::new();
            for (r, c, v) in [
                (0usize, 0usize, f32::NAN),
                (1, cols - 1, f32::INFINITY),
                (2, cols / 2, f32::NEG_INFINITY),
                (1500, cols - 1, f32::NAN),
                (2999, 0, f32::NEG_INFINITY),
                (3000, cols / 2, f32::INFINITY),
            ]
            .into_iter()
            {
                dirty[r * cols + c] = v;
                bad.push(r);
            }
            let (res, got_q, got_s) = quantize_poisoned(rt, &dirty, rows, cols);
            let err = res.expect_err("non-finite input must return an error");
            assert!(err.contains("non-finite"), "{err}");
            for r in 0..rows {
                let qrow = &got_q[r * cols..(r + 1) * cols];
                let row = &dirty[r * cols..(r + 1) * cols];
                if bad.contains(&r) {
                    assert!(
                        qrow.iter().all(|&v| v == 0x55),
                        "cols {cols} row {r}: a refused row was written"
                    );
                    let finite: Vec<f32> = row.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
                    let (_, want) = quant_row(&finite).unwrap();
                    let ulps = (got_s[r].to_bits() as i64 - want.to_bits() as i64).abs();
                    assert!(
                        ulps <= 1,
                        "cols {cols} row {r}: refused row's scale {} vs finite-elements scale {want}",
                        got_s[r]
                    );
                } else {
                    check_quant_row(row, qrow, got_s[r], &format!("cols {cols} row {r}"));
                }
            }
        }
    });
}

/// A row of only non-finite values: refused, not written, scale 1 (no finite
/// element, so absmax stays 0).
#[test]
fn adv_quantize_row_of_only_non_finite_values_is_refused_with_scale_one() {
    with_gpu(|rt| {
        let cols = 300usize;
        let flat: Vec<f32> = (0..cols)
            .map(|c| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][c % 3])
            .collect();
        let (res, got_q, got_s) = quantize_poisoned(rt, &flat, 1, cols);
        assert!(res.expect_err("refused").contains("non-finite"));
        assert!(got_q.iter().all(|&v| v == 0x55));
        assert_eq!(got_s[0], 1.0);
    });
}

/// Race hunt. 8192 rows (one threadgroup each), cols just over one 256-wide
/// stride, many iterations, outputs poisoned each time, clean and dirty
/// matrices alternating. Run against the pre-fix kernel (d8174da reverted)
/// and the current one; the result of the pre-fix run is in the review report.
#[test]
fn adv_quantize_race_hunt_many_rows_cols_just_over_one_stride() {
    with_gpu(|rt| {
        for cols in [257usize, 513, 300] {
            let rows = 8192usize;
            let clean = random_rows(rows, cols, 0x5eed ^ cols as u64);
            let mut dirty = clean.clone();
            let bad_rows: Vec<usize> = (0..rows).filter(|r| r % 13 == 4).collect();
            for &r in &bad_rows {
                dirty[r * cols + (r * 7) % cols] = if r % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
            }
            let is_bad: Vec<bool> = (0..rows).map(|r| r % 13 == 4).collect();
            for iter in 0..24 {
                let with_bad = iter % 2 == 1;
                let data = if with_bad { &dirty } else { &clean };
                let (res, got_q, got_s) = quantize_poisoned(rt, data, rows, cols);
                assert_eq!(res.is_err(), with_bad, "cols {cols} iter {iter}");
                for r in 0..rows {
                    let qrow = &got_q[r * cols..(r + 1) * cols];
                    if with_bad && is_bad[r] {
                        assert!(
                            qrow.iter().all(|&v| v == 0x55),
                            "cols {cols} iter {iter} row {r}: a refused row was written"
                        );
                    } else {
                        check_quant_row(
                            &data[r * cols..(r + 1) * cols],
                            qrow,
                            got_s[r],
                            &format!("cols {cols} iter {iter} row {r}"),
                        );
                    }
                }
            }
        }
    });
}

/// f16 rounding at the top of the range, through the real kernels: 65504 is
/// the largest finite half, 65519 (and the f32 just below 65520) rounds to
/// 65504, 65520 is the tie that rounds to even = infinity.
#[test]
fn adv_f16_store_boundary_is_exactly_65520() {
    with_gpu(|rt| {
        let below = f32::from_bits(65520.0f32.to_bits() - 1);
        let above = f32::from_bits(65520.0f32.to_bits() + 1);
        let values: [(f32, u16); 7] = [
            (65504.0, 0x7bff),
            (65505.0, 0x7bff),
            (65519.0, 0x7bff),
            (below, 0x7bff),
            (65520.0, 0x7c00),
            (above, 0x7c00),
            (70000.0, 0x7c00),
        ];
        for m in [1u32, 2, 65] {
            for (v, want) in values {
                for sign in [1.0f32, -1.0] {
                    let want = if sign < 0.0 { want | 0x8000 } else { want };
                    let n = 2u32;
                    let got = run_case(
                        rt,
                        Case {
                            a: &vec![1i8; m as usize],
                            b: &vec![1i8; n as usize],
                            m,
                            n,
                            k: 1,
                            layout: nn::I8BLayout::RowMajorKN,
                            a_scale: nn::I8AScale::Scalar(sign * v),
                            b_scale: nn::I8BScale::None,
                            bias: None,
                            out: nn::I8Out::F16,
                        },
                    );
                    rt.synchronize().unwrap();
                    let bits = read_bits(&got, (m * n) as usize);
                    println!(
                        "m={m} scalar scale {:+} -> {:#06x} (want {want:#06x})",
                        sign * v,
                        bits[0]
                    );
                    assert!(
                        bits.iter().all(|&b| b == want),
                        "m={m} scale {}: {bits:#06x?} want {want:#06x}",
                        sign * v
                    );
                    // The host conversion agrees with the claim.
                    assert_eq!(
                        tessl::tensor::f32_to_f16_bits(sign * v),
                        want,
                        "host cast of {}",
                        sign * v
                    );
                }
            }
        }
    });
}
