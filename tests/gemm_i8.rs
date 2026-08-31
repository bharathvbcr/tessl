//! Quantized int8 GEMM with fused dequantization.
//!
//! The property worth testing here is not "close to a reference" but **exact**.
//! `int8 x int8` accumulates into `int32`, and every product fits, so the
//! integer result carries no rounding whatever. The only approximation is the
//! final multiply by the scales. That makes an exact integer reference the
//! right oracle, and it catches errors a float tolerance would absorb.

mod common;

use std::sync::Arc;

use common::with_gpu;
use tessl::nn;
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn i8_buf(rt: &Arc<GpuRuntime>, data: &[i8]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1)).expect("alloc");
    b.write_bytes(&data.iter().map(|v| *v as u8).collect::<Vec<u8>>());
    b
}

fn f32_buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

/// Deterministic int8 spread over the full range, including the extremes.
fn i8_data(n: usize, seed: u64) -> Vec<i8> {
    let mut x = seed;
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
            let cb = f32_buf(rt, &vec![0.0f32; m * n]);

            // a_scale of 1 and no b_scale, so the output *is* the integer sum
            // and any deviation is a real arithmetic error, not rounding.
            nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None)
                .expect("gemm_i8_dequant");
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
        let sb = f32_buf(rt, &scale);
        let cb = f32_buf(rt, &vec![0.0f32; m * n]);

        nn::gemm_i8_dequant(
            rt,
            &ab,
            &bb,
            &cb,
            m as u32,
            n as u32,
            k as u32,
            0.5,
            Some(&sb),
        )
        .expect("gemm_i8_dequant");
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
        let cb = f32_buf(rt, &vec![0.0f32; m * n]);

        nn::gemm_i8_dequant(rt, &ab, &bb, &cb, m as u32, n as u32, k as u32, 1.0, None)
            .expect("gemm_i8_dequant");
        rt.synchronize().unwrap();

        let want = (k as i64 * 16384) as f32;
        let got = cb.read_f32();
        for (e, g) in got.iter().take(m * n).enumerate() {
            assert_eq!(*g, want, "full-range accumulation at {e}");
        }
    });
}

#[test]
fn a_k_that_could_overflow_int32_is_refused() {
    with_gpu(|rt| {
        let b = rt.alloc_buffer(1 << 20).unwrap();
        let c = rt.alloc_buffer(1 << 20).unwrap();
        // Past this k, full-range int8 products can wrap the int32 accumulator
        // silently. Refusing keeps the exactness claim true rather than nearly.
        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 8, 8, 200_000, 1.0, None)
            .expect_err("k past the exact range");
        assert!(err.contains("overflow"), "{err}");

        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 0, 8, 8, 1.0, None).expect_err("m = 0");
        assert!(err.contains("non-zero"), "{err}");

        let err = nn::gemm_i8_dequant(rt, &b, &b, &c, 8, 8, 8, f32::NAN, None)
            .expect_err("non-finite scale");
        assert!(err.contains("finite"), "{err}");
    });
}

#[test]
fn undersized_operands_are_refused_before_dispatch() {
    with_gpu(|rt| {
        let small = rt.alloc_buffer(16).unwrap();
        let big = rt.alloc_buffer(1 << 20).unwrap();
        let err = nn::gemm_i8_dequant(rt, &small, &big, &big, 128, 64, 64, 1.0, None)
            .expect_err("A too small");
        assert!(err.contains("buffer holds"), "{err}");
        let scale = rt.alloc_buffer(4).unwrap();
        let err = nn::gemm_i8_dequant(rt, &big, &big, &big, 128, 64, 64, 1.0, Some(&scale))
            .expect_err("scale too short");
        assert!(err.contains("buffer holds"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
