//! Row-wise reductions: softmax, sum, max.
//!
//! Softmax's whole reason for existing in a stable form is the overflow it
//! avoids, so the tests feed it the inputs that would overflow a naive one —
//! logits far above `ln(f32::MAX)`, and a fully masked row of `-inf` — rather
//! than only well-scaled random data where naive and stable agree.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::nn;
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

fn buf(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 4).expect("alloc");
    b.write_f32(data);
    b
}

fn empty(rt: &Arc<GpuRuntime>, n: usize) -> GpuBuffer {
    let b = rt.alloc_buffer(n.max(1) * 4).expect("alloc");
    b.zero();
    b
}

/// f64 reference, so the comparison is against better arithmetic than the
/// kernel's rather than against the same rounding.
fn softmax_ref(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return vec![1.0 / row.len() as f32; row.len()];
    }
    let exps: Vec<f64> = row.iter().map(|v| ((*v - m) as f64).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / s) as f32).collect()
}

#[test]
fn softmax_matches_a_f64_reference() {
    with_gpu(|rt| {
        for &(rows, cols) in &[(1usize, 1usize), (4, 17), (8, 512), (3, 4096)] {
            let x: Vec<f32> = random_f32(rows * cols, 0x50F + cols as u64)
                .iter()
                .map(|v| v * 8.0)
                .collect();
            let xb = buf(rt, &x);
            let ob = empty(rt, rows * cols);
            nn::softmax_rows_f32(rt, &xb, &ob, rows as u32, cols as u32).unwrap();
            rt.synchronize().unwrap();

            let got = ob.read_f32();
            for r in 0..rows {
                let want = softmax_ref(&x[r * cols..(r + 1) * cols]);
                let mut sum = 0.0f64;
                for c in 0..cols {
                    let g = got[r * cols + c];
                    assert!(
                        (g - want[c]).abs() <= 1e-6,
                        "softmax[{r},{c}] = {g}, want {} ({rows}x{cols})",
                        want[c]
                    );
                    sum += g as f64;
                }
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "row {r} of {rows}x{cols} sums to {sum}, not 1"
                );
            }
        }
    });
}

#[test]
fn softmax_survives_logits_that_overflow_a_naive_exp() {
    with_gpu(|rt| {
        // exp(89) is already infinity in f32. A softmax that does not subtract
        // the row max returns NaN for every one of these rows.
        let cols = 64usize;
        let rows = 4usize;
        let mut x = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                x[r * cols + c] = 100.0 + (r * cols + c) as f32 * 0.5;
            }
        }
        let xb = buf(rt, &x);
        let ob = empty(rt, rows * cols);
        nn::softmax_rows_f32(rt, &xb, &ob, rows as u32, cols as u32).unwrap();
        rt.synchronize().unwrap();

        let got = ob.read_f32();
        assert!(
            got[..rows * cols].iter().all(|v| v.is_finite()),
            "large logits produced a non-finite softmax"
        );
        for r in 0..rows {
            let want = softmax_ref(&x[r * cols..(r + 1) * cols]);
            for c in 0..cols {
                assert!(
                    (got[r * cols + c] - want[c]).abs() <= 1e-6,
                    "row {r} col {c}: {} vs {}",
                    got[r * cols + c],
                    want[c]
                );
            }
        }
    });
}

#[test]
fn a_fully_masked_row_is_uniform_not_nan() {
    with_gpu(|rt| {
        // Every position masked out is what an attention row looks like when
        // nothing is visible. Dividing by a zero denominator would give NaN and
        // silently poison whatever consumes it.
        let cols = 32usize;
        let x = vec![f32::NEG_INFINITY; cols];
        let xb = buf(rt, &x);
        let ob = empty(rt, cols);
        nn::softmax_rows_f32(rt, &xb, &ob, 1, cols as u32).unwrap();
        rt.synchronize().unwrap();

        let got = ob.read_f32();
        let uniform = 1.0 / cols as f32;
        for (c, g) in got.iter().take(cols).enumerate() {
            assert!(
                (g - uniform).abs() < 1e-6,
                "masked row position {c} = {g}, want uniform {uniform}"
            );
        }
    });
}

#[test]
fn softmax_in_place_matches_out_of_place() {
    with_gpu(|rt| {
        let (rows, cols) = (5usize, 129usize);
        let x = random_f32(rows * cols, 0xA1A1);
        let a = buf(rt, &x);
        let out = empty(rt, rows * cols);
        nn::softmax_rows_f32(rt, &a, &out, rows as u32, cols as u32).unwrap();

        let b = buf(rt, &x);
        nn::softmax_rows_f32(rt, &b, &b, rows as u32, cols as u32).unwrap();
        rt.synchronize().unwrap();

        // In place, the kernel re-reads `x` after writing part of it. It only
        // works because each pass finishes before the next begins; if that
        // stopped holding, the two would diverge here and nowhere else.
        let (o, i) = (out.read_f32(), b.read_f32());
        for k in 0..rows * cols {
            assert_eq!(o[k].to_bits(), i[k].to_bits(), "in-place differs at {k}");
        }
    });
}

#[test]
fn row_sum_and_row_max_match_a_f64_reference() {
    with_gpu(|rt| {
        let (rows, cols) = (7usize, 1000usize);
        let x = random_f32(rows * cols, 0x5115);
        let xb = buf(rt, &x);
        let sums = empty(rt, rows);
        let maxes = empty(rt, rows);
        nn::row_sum_f32(rt, &xb, &sums, rows as u32, cols as u32).unwrap();
        nn::row_max_f32(rt, &xb, &maxes, rows as u32, cols as u32).unwrap();
        rt.synchronize().unwrap();

        let (gs, gm) = (sums.read_f32(), maxes.read_f32());
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let want_sum: f64 = row.iter().map(|v| *v as f64).sum();
            let want_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // A tree reduction reassociates against a sequential f64 sum; the
            // bound is the usual n*eps*max|term|.
            let bound =
                8.0 * f32::EPSILON * cols as f32 * row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (gs[r] as f64 - want_sum).abs() <= bound as f64,
                "row_sum[{r}] = {} want {want_sum} (bound {bound})",
                gs[r]
            );
            assert_eq!(gm[r], want_max, "row_max[{r}]");
        }
    });
}

#[test]
fn reductions_reject_empty_and_undersized_operands() {
    with_gpu(|rt| {
        let full = empty(rt, 4 * 16);
        let short = empty(rt, 3);
        let err = nn::softmax_rows_f32(rt, &full, &full, 4, 0).expect_err("zero cols");
        assert!(err.contains("non-zero"), "{err}");
        let err = nn::row_sum_f32(rt, &full, &short, 4, 16).expect_err("short out");
        assert!(err.contains("buffer holds"), "{err}");
        let err = nn::softmax_rows_f32(rt, &short, &full, 4, 16).expect_err("short x");
        assert!(err.contains("buffer holds"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
