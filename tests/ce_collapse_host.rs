//! Softmax collapse in the cross-entropy gradient, on the host (no device).
//!
//! Barron, "A Fix for Softmax and Cross-Entropy": past the f32 precision
//! cliff (a margin near 16.6) `1 + exp(-margin)` rounds to 1, the target's
//! softmax probability rounds to 1, and `softmax_t - 1` is exactly 0 while the
//! other classes' gradients stay positive, so a row no longer sums to 0.
//!
//! Two f32 mirrors of the kernels' arithmetic, one per formula, are held to
//! the f64 oracle of `tests/common/ce.rs`:
//!
//! - `shifted_softmax_grad`, the max-shifted `softmax - onehot` this crate's
//!   kernel used to compute (`p -= 1.0f` at the target). It documents the
//!   failure: its target gradient is 0 on the paper's row and its row sums to
//!   a positive number.
//! - `excluded_target_grad`, what `ce_lse_update` + `ce_softmax_grad` compute
//!   now: the log-sum-exp over the non-target classes, the target gradient
//!   `-sigmoid(-margin)`, the others `exp(z_j - L) * sigmoid(-margin)`.
//!
//! The Metal kernels are checked against the same oracle on the device in
//! `tests/cross_entropy.rs`.

mod common;

use common::ce::ce_oracle;

/// The old kernel's arithmetic for one row, in f32: running max `m`, sum
/// `s = sum(exp(z - m))` over every class, `lse = m + log(s)`, then
/// `exp(z_j - lse)` with 1 subtracted at the target.
fn shifted_softmax_grad(z: &[f32], t: usize) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::MIN, f32::max);
    let s: f32 = z.iter().map(|&l| (l - m).exp()).sum();
    let lse = m + s.ln();
    z.iter()
        .enumerate()
        .map(|(j, &l)| {
            let mut p = (l - lse).exp();
            if j == t {
                p -= 1.0;
            }
            p
        })
        .collect()
}

/// The new kernels' arithmetic for one row, in f32 (the vocabulary walk's
/// chunking changes only the order of the sum). A row with no finite
/// non-target class has `s == 0` and a zero gradient.
fn excluded_target_grad(z: &[f32], t: usize) -> Vec<f32> {
    let others = || z.iter().enumerate().filter(|&(j, _)| j != t).map(|(_, &l)| l);
    let m = others().fold(f32::MIN, f32::max);
    let s: f32 = others().map(|l| (l - m).exp()).sum();
    if s == 0.0 {
        return vec![0.0; z.len()];
    }
    let lo = m + s.ln();
    let margin = z[t] - lo;
    let e = (-margin.abs()).exp();
    let q = if margin >= 0.0 { e / (1.0 + e) } else { 1.0 / (1.0 + e) };
    z.iter()
        .enumerate()
        .map(|(j, &l)| if j == t { -q } else { (l - lo).exp() * q })
        .collect()
}

fn as_f64(z: &[f32]) -> Vec<f64> {
    z.iter().map(|&x| f64::from(x)).collect()
}

/// The paper's row: the old formula has no target gradient and a row sum of
/// `+3e-8`; the oracle has `g_t = -3.0e-8` and a zero sum.
#[test]
fn the_shifted_softmax_loses_the_target_gradient_past_the_cliff() {
    let z = [9.0f32, -9.0, -9.0];
    let (_, want) = ce_oracle(&as_f64(&z), 0);
    let q = 1.0 / (1.0 + (9.0f64 - (-9.0 + 2f64.ln())).exp());
    assert!((want[0] + q).abs() < 1e-12 * q, "oracle target gradient {:e}", want[0]);
    assert!(want.iter().sum::<f64>().abs() < 1e-15 * q);

    let old = shifted_softmax_grad(&z, 0);
    assert_eq!(old[0], 0.0, "the old target gradient should have collapsed to 0");
    assert!(old[1] > 0.0 && old[2] > 0.0);
    let old_sum: f32 = old.iter().sum();
    assert!(
        f64::from(old_sum) > 0.9 * 2.0 * f64::from(old[1]),
        "the old row sums to {old_sum:e}, not 0"
    );
}

/// The fixed arithmetic keeps the target gradient and sums to zero within a
/// few ulps, from the paper's row to margins of +-100, with the target first,
/// in the middle and last, and for ties.
#[test]
fn the_excluded_target_gradient_matches_the_oracle_and_sums_to_zero() {
    let others = [0.0f32, -1.0, -2.5, 0.5, 3.0, -0.25];
    let lo: f64 = others.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
    let mut cases: Vec<(Vec<f32>, usize)> = vec![
        (vec![9.0, -9.0, -9.0], 0),
        (vec![-9.0, 9.0, -9.0], 1),
        (vec![-9.0, -9.0, 9.0], 2),
        (vec![3.0; 7], 4),
    ];
    for m in [
        0.0f64, 16.0, -16.0, 17.0, -17.0, 20.0, -20.0, 40.0, -40.0, 100.0, -100.0,
    ] {
        for t in [0usize, 3, 6] {
            let mut z = others.to_vec();
            z.insert(t, (lo + m) as f32);
            cases.push((z, t));
        }
    }
    for (z, t) in cases {
        let (_, want) = ce_oracle(&as_f64(&z), t);
        let got = excluded_target_grad(&z, t);
        let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let mut sum = 0.0f64;
        for (j, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() <= 3e-5 * peak + 1e-36,
                "{z:?} target {t} class {j}: {g:e} want {w:e}"
            );
            sum += f64::from(g);
        }
        if want[t].abs() > 1e-30 {
            assert!(
                got[t] < 0.0,
                "{z:?} target {t}: the target gradient collapsed to {:e}",
                got[t]
            );
        }
        let l = ce_oracle_other_lse(&z, t);
        let tol = 4.0 * f64::from(f32::EPSILON) * (1.0 + l.abs()) * peak;
        assert!(sum.abs() <= tol, "{z:?} target {t}: row sums to {sum:e}, tol {tol:e}");
    }
}

fn ce_oracle_other_lse(z: &[f32], t: usize) -> f64 {
    common::ce::other_classes_lse(&as_f64(z), t)
}

/// A row whose other classes are all `-inf`: loss 0 and zero gradient, by
/// definition, in the oracle and the mirror alike.
#[test]
fn a_row_of_only_the_target_has_no_loss_and_no_gradient() {
    let z = [f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
    let (loss, want) = ce_oracle(&as_f64(&z), 1);
    assert_eq!(loss, 0.0);
    assert!(want.iter().all(|&g| g == 0.0));
    assert!(excluded_target_grad(&z, 1).iter().all(|&g| g == 0.0));
}

/// The oracle itself, against the textbook `softmax - onehot` where that is
/// well conditioned (small margins) so the two forms are the same function.
#[test]
fn the_oracle_is_softmax_minus_onehot_where_that_is_stable() {
    let z = [0.3f64, -1.2, 2.0, 0.7];
    for t in 0..4 {
        let m = z.iter().copied().fold(f64::MIN, f64::max);
        let lse = m + z.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        let (loss, g) = ce_oracle(&z, t);
        assert!((loss - (lse - z[t])).abs() < 1e-13);
        for (j, &gj) in g.iter().enumerate() {
            let want = (z[j] - lse).exp() - f64::from(u8::from(j == t));
            assert!((gj - want).abs() < 1e-13, "t={t} j={j}: {gj} want {want}");
        }
    }
}
