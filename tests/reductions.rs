//! Row-wise reductions: softmax, sum, max.
//!
//! Softmax's whole reason for existing in a stable form is the overflow it
//! avoids, so the tests feed it the inputs that would overflow a naive one —
//! logits far above `ln(f32::MAX)`, and a fully masked row of `-inf` — rather
//! than only well-scaled random data where naive and stable agree.

mod common;

use common::{buf, empty, random_f32, reduction_bound, tensor_f32, with_gpu};
use tessl::gemm::{gemm, GemmBackend};
use tessl::nn;
use tessl::tensor::{f16_bits_to_f32, f32_slice_to_f16, f32_to_f16_bits};

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
            let bound = 8.0 * f32::EPSILON * cols as f32 * row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (gs[r] as f64 - want_sum).abs() <= bound as f64,
                "row_sum[{r}] = {} want {want_sum} (bound {bound})",
                gs[r]
            );
            assert_eq!(gm[r], want_max, "row_max[{r}]");
        }
    });
}

/// The max reductions seed from the row's own data, not from an infinity.
///
/// Kernels compile with fast math, which may assume no value is infinite, so
/// the lane seed and the padding of lanes past the last simdgroup must not be
/// `-INFINITY`. Neither may they be `-FLT_MAX`: a row of only `-inf` would
/// then report `-FLT_MAX`. Widths around one simdgroup (lanes with no element,
/// a partial last simdgroup) and past one threadgroup, rows of all `-inf`, and
/// rows whose one finite entry is `-FLT_MAX` itself.
#[test]
fn row_max_and_softmax_of_masked_rows_never_invent_a_value() {
    with_gpu(|rt| {
        for cols in [1usize, 31, 33, 100, 1000, 5000] {
            let rows = 3usize;
            let mut x = vec![f32::NEG_INFINITY; rows * cols];
            x[cols + cols / 2] = f32::MIN;
            x[2 * cols + cols - 1] = -2.5;
            let xb = buf(rt, &x);
            let maxes = empty(rt, rows);
            let soft = empty(rt, rows * cols);
            nn::row_max_f32(rt, &xb, &maxes, rows as u32, cols as u32).unwrap();
            nn::softmax_rows_f32(rt, &xb, &soft, rows as u32, cols as u32).unwrap();
            rt.synchronize().unwrap();
            let gm = maxes.read_f32();
            assert_eq!(gm[0], f32::NEG_INFINITY, "cols={cols}: max of an all -inf row");
            assert_eq!(gm[1], f32::MIN, "cols={cols}: max whose only finite entry is f32::MIN");
            assert_eq!(gm[2], -2.5, "cols={cols}: max whose only finite entry is -2.5");
            let got = soft.read_f32();
            let uniform = 1.0 / cols as f32;
            for (c, g) in got[..cols].iter().enumerate() {
                assert!(
                    (g - uniform).abs() <= 1e-6 * uniform.max(1.0),
                    "cols={cols}: masked row [{c}] = {g}, want {uniform}"
                );
            }
            for r in 1..rows {
                let want = softmax_ref(&x[r * cols..(r + 1) * cols]);
                for (c, (g, w)) in got[r * cols..(r + 1) * cols].iter().zip(&want).enumerate() {
                    assert!((g - w).abs() <= 1e-6, "cols={cols} row {r} [{c}] = {g}, want {w}");
                }
            }
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

/// Bit patterns of a read-back row, so "untouched" means bit-identical (NaN included).
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// One run of the three row reductions over `rows x cols`: `(sum, max, softmax)`.
fn run_reductions(
    rt: &std::sync::Arc<tessl::GpuRuntime>,
    x: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let xb = buf(rt, x);
    let (s, m, sm) = (empty(rt, rows), empty(rt, rows), empty(rt, rows * cols));
    nn::row_sum_f32(rt, &xb, &s, rows as u32, cols as u32).unwrap();
    nn::row_max_f32(rt, &xb, &m, rows as u32, cols as u32).unwrap();
    nn::softmax_rows_f32(rt, &xb, &sm, rows as u32, cols as u32).unwrap();
    rt.synchronize().unwrap();
    (s.read_f32(), m.read_f32(), sm.read_f32())
}

/// NaN, +inf and -inf planted at known positions of known rows.
///
/// Rows: 0 clean, 1 one NaN, 2 one +inf, 3 one -inf (not at the maximum), 4 +inf and -inf,
/// 5 clean. The same data without the plants is run first; rows 0 and 5 must come back
/// bit-identical to it, so a poisoned row cannot leak into a neighbour through a shared
/// scratch or a mis-strided read.
///
/// Kernels compile with fast math (`build.rs`), so the compiler is allowed to assume no NaN or
/// inf: IEEE propagation is therefore pinned only where the arithmetic is unambiguous (the sum
/// and the +inf maximum). Where a fast-math `fmax` or a NaN denominator may legitimately take
/// either of two results, the test pins the set of acceptable results and nothing else, so a
/// value invented from nowhere still fails.
#[test]
fn planted_non_finite_inputs_poison_exactly_their_rows() {
    with_gpu(|rt| {
        for cols in [33usize, 1000, 5000] {
            let rows = 6usize;
            let clean = random_f32(rows * cols, 0xBAD + cols as u64);
            let mut x = clean.clone();
            let at = |r: usize, c: usize| r * cols + c;
            let (c_nan, c_pinf, c_ninf, c_a, c_b) = (cols / 3, cols - 1, 1, 0, cols / 2);
            x[at(1, c_nan)] = f32::NAN;
            x[at(2, c_pinf)] = f32::INFINITY;
            x[at(3, c_ninf)] = f32::NEG_INFINITY;
            x[at(4, c_a)] = f32::INFINITY;
            x[at(4, c_b)] = f32::NEG_INFINITY;

            let (cs, cm, csm) = run_reductions(rt, &clean, rows, cols);
            let (s, m, sm) = run_reductions(rt, &x, rows, cols);
            let row = |v: &[f32], r: usize| v[r * cols..(r + 1) * cols].to_vec();
            let label = format!("cols={cols}");

            for r in [0usize, 5] {
                assert_eq!(s[r].to_bits(), cs[r].to_bits(), "{label}: sum of clean row {r}");
                assert_eq!(m[r].to_bits(), cm[r].to_bits(), "{label}: max of clean row {r}");
                assert_eq!(
                    bits(&row(&sm, r)),
                    bits(&row(&csm, r)),
                    "{label}: softmax of clean row {r}"
                );
            }

            // Sum: IEEE, row by row.
            assert!(s[1].is_nan(), "{label}: NaN + finite = NaN, got {}", s[1]);
            assert_eq!(s[2], f32::INFINITY, "{label}: sum of a +inf row");
            assert_eq!(s[3], f32::NEG_INFINITY, "{label}: sum of a -inf row");
            assert!(s[4].is_nan(), "{label}: +inf + -inf = NaN, got {}", s[4]);

            // Max: +inf wins; a planted -inf never is the maximum of a row that has finite
            // entries (compared to the row's own finite maximum); a NaN gives NaN or the
            // finite maximum.
            assert_eq!(m[2], f32::INFINITY, "{label}: max of a +inf row");
            assert_eq!(m[4], f32::INFINITY, "{label}: max of a +inf and -inf row");
            let finite_max_3 = row(&x, 3)
                .iter()
                .cloned()
                .filter(|v| v.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(m[3], finite_max_3, "{label}: a planted -inf cannot be a row's maximum");
            let finite_max_1 = row(&x, 1)
                .iter()
                .cloned()
                .filter(|v| v.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                m[1].is_nan() || m[1] == finite_max_1,
                "{label}: max of a NaN row is NaN or the finite maximum {finite_max_1}, got {}",
                m[1]
            );

            // Softmax of a poisoned row: no invented probability. Each entry is NaN, or the
            // row is the uniform fallback the kernel documents for a non-positive
            // denominator (a NaN denominator compares false against zero), or, for -inf
            // alone, the finite entries' own softmax with a zero at the -inf position.
            let uniform = 1.0 / cols as f32;
            for r in 1..=4usize {
                let out = row(&sm, r);
                let all_uniform = out.iter().all(|p| (p - uniform).abs() <= 1e-6 * uniform.max(1.0));
                let any_nan = out.iter().any(|p| p.is_nan());
                let finite_valid = out.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                    && (out.iter().map(|p| f64::from(*p)).sum::<f64>() - 1.0).abs() < 1e-4;
                eprintln!(
                    "OBSERVED softmax {label} row {r}: nan_count={} all_uniform={all_uniform} finite_valid={finite_valid}",
                    out.iter().filter(|p| p.is_nan()).count()
                );
                assert!(
                    any_nan || all_uniform || finite_valid,
                    "{label}: softmax of poisoned row {r} is neither NaN, uniform nor a distribution: {:?}",
                    &out[..4.min(cols)]
                );
            }
            // The -inf row: its -inf position carries exactly zero mass.
            assert_eq!(sm[at(3, c_ninf)], 0.0, "{label}: softmax at a planted -inf");
        }
    });
}

/// Subnormal inputs. Fast math may flush them to zero, so the result is either the f32
/// reference or the flushed one; what it must never be is a magnitude the input does not
/// have. Budget: `reduction_bound(n, scale)` with scale = sum of |terms|.
#[test]
fn subnormal_inputs_are_summed_or_flushed_never_invented() {
    with_gpu(|rt| {
        let cols = 1000usize;
        let sub = 1.0e-40f32;
        assert!(sub > 0.0 && sub < f32::MIN_POSITIVE, "1e-40 must be subnormal in f32");
        let rows = 3usize;
        let mut x = vec![0.0f32; rows * cols];
        x[..cols].fill(sub); // row 0: all subnormal
        x[cols..2 * cols].fill(1.0); // row 1: normal, with one subnormal that must not matter
        x[cols + 7] = sub;
        x[2 * cols + 11] = -sub; // row 2: one negative subnormal in zeros
        let (s, m, _) = run_reductions(rt, &x, rows, cols);

        let ref0 = f64::from(sub) * cols as f64;
        let b0 = reduction_bound(cols, ref0);
        let branch = |got: f32, want: f64, b: f64| {
            if got == 0.0 {
                "flushed-to-zero"
            } else if (f64::from(got) - want).abs() <= b {
                "reference"
            } else {
                "neither"
            }
        };
        eprintln!(
            "OBSERVED subnormal sum row0 = {:e} ({}), row2 = {:e}, max row0 = {:e} ({})",
            s[0],
            branch(s[0], ref0, b0),
            s[2],
            m[0],
            if m[0] == 0.0 { "flushed-to-zero" } else { "kept" }
        );
        assert!(
            s[0] == 0.0 || (f64::from(s[0]) - ref0).abs() <= b0,
            "sum of {cols} x {sub:e} = {:e}: want {ref0:e} +- {b0:e} or flushed 0",
            s[0]
        );
        let ref1 = (cols - 1) as f64 + f64::from(sub);
        let b1 = reduction_bound(cols, ref1);
        assert!(
            (f64::from(s[1]) - ref1).abs() <= b1,
            "normal row with a subnormal: {} want {ref1} +- {b1}",
            s[1]
        );
        assert!(
            s[2] == 0.0 || (f64::from(s[2]) + f64::from(sub)).abs() <= reduction_bound(cols, f64::from(sub)),
            "sum of zeros and one -{sub:e} = {:e}",
            s[2]
        );
        assert!(
            m[0] == sub || m[0] == 0.0,
            "max of a subnormal row is itself or flushed, got {:e}",
            m[0]
        );
        assert_eq!(m[1], 1.0, "max of the normal row");
        assert!(
            m[2] == 0.0,
            "max of zeros and one negative subnormal is 0, got {:e}",
            m[2]
        );
    });
}

/// What an fp16 running sum does with `terms`: every partial sum rounded to f16.
fn f16_running_sum(terms: &[f32]) -> f32 {
    terms
        .iter()
        .fold(0.0f32, |acc, t| f16_bits_to_f32(f32_to_f16_bits(acc + t)))
}

/// A K-long dot product of f16 operands accumulates in f32 (the contract: f16/bf16 operands
/// "accumulate in f32", `tests/f16.rs`). Two inputs on which an f16 accumulator fails
/// visibly, each first checked on the host to *discriminate* (an f16 running sum is off by
/// far more than the budget), so a vacuous input cannot pass:
///
/// * K = 8192 products of 1.0: f16's running sum stalls at 2048 (ulp 2 absorbs +1).
/// * K = 4096 products of 256.0: the true 2^20 exceeds f16's 65504, so an f16 sum is +inf.
///
/// Reference: f64 over the same f16-rounded operands; budget `reduction_bound(K, sum |terms|)`.
#[test]
fn f16_operands_accumulate_in_f32() {
    with_gpu(|rt| {
        for &(k, a_val, b_val) in &[(8192usize, 1.0f32, 1.0f32), (4096, 16.0, 16.0)] {
            let (m, n) = (128usize, 64usize);
            let terms: Vec<f32> = vec![a_val * b_val; k];
            let abs_sum: f64 = terms.iter().map(|t| f64::from(t.abs())).sum();
            let want = terms.iter().map(|t| f64::from(*t)).sum::<f64>();
            let bound = reduction_bound(k, abs_sum);
            let f16_sum = f64::from(f16_running_sum(&terms));
            assert!(
                !f16_sum.is_finite() || (f16_sum - want).abs() > 100.0 * bound,
                "k={k}: input does not discriminate (f16 sum {f16_sum}, f32 {want}, bound {bound})"
            );

            let a = rt.alloc_tensor_f16(&[m, k]).unwrap();
            a.buffer.write_f16_bits(&f32_slice_to_f16(&vec![a_val; m * k]));
            let b = rt.alloc_tensor_f16(&[k, n]).unwrap();
            b.buffer.write_f16_bits(&f32_slice_to_f16(&vec![b_val; k * n]));
            let c = tensor_f32(rt, &[m, n], &vec![0.0; m * n]);
            gemm(&a, &b, &c, GemmBackend::TensorOps).expect("f16 gemm");
            rt.synchronize().unwrap();
            for (i, got) in c.buffer.read_f32().iter().enumerate() {
                assert!(
                    (f64::from(*got) - want).abs() <= bound,
                    "k={k}: c[{i}] = {got}, f32-accumulated reference {want} +- {bound} (an f16 accumulator gives {f16_sum})"
                );
            }
        }
    });
}
