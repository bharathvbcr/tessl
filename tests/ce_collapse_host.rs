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

/// `exp` as the kernels may not rely on it: they compile under fast math
/// (`-fmetal-math-mode=fast`), where an infinite or NaN argument is outside
/// what `precise::exp` must handle. The mirror makes that hostile case
/// explicit (NaN), so it passes only where no infinity ever reaches `exp`.
fn kexp(x: f32) -> f32 {
    if x.is_finite() {
        x.exp()
    } else {
        f32::NAN
    }
}

/// How `ce_lse_update` keeps `-inf` away from `exp`. Three versions of the
/// kernel, each modelled as the device runs it:
/// - `None`: the first version fed every logit to `exp`.
/// - `Compare`: the second skipped `l > -FLT_MAX`. Under fast math (`ninf`,
///   `nnan`) the compare is false for NaN, so a NaN logit is silently dropped
///   instead of poisoning the sum (the device run's `a_nan_logit_is_an_error`
///   and `rejects_what_it_cannot_compute` failures). The mirror models that
///   NaN behaviour; the device also lost finite logits in row 1 of the `-inf`
///   test, which the mirror cannot reproduce (a compare folded by the
///   optimiser has no host model).
/// - `Bits`: the kernel's: skip exactly the bit pattern `0xff800000`, let NaN
///   through to `exp`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Guard {
    None,
    Compare,
    Bits,
}

const NEG_INF_BITS: u32 = 0xff80_0000;

/// `ce_is_zero`: the bits of `x`, sign dropped, are 0.
fn bits_is_zero(x: f32) -> bool {
    x.to_bits() << 1 == 0
}

/// `ce_lse_update` over `chunk`-wide pieces of the row, in f32: the running
/// maximum `m` (identity `f32::MIN`, never an infinity) and `s` over the
/// non-target classes, with the given [`Guard`]. A zero `s` is not rescaled
/// under `Compare` and `Bits`.
fn lse_walk(z: &[f32], t: usize, chunk: usize, guard: Guard) -> (f32, f32) {
    let (mut m, mut s) = (f32::MIN, 0.0f32);
    for (k, piece) in z.chunks(chunk).enumerate() {
        let v0 = k * chunk;
        let live = |c: usize, l: f32| {
            v0 + c != t
                && match guard {
                    Guard::None => true,
                    Guard::Compare => l > f32::MIN,
                    Guard::Bits => l.to_bits() != NEG_INF_BITS,
                }
        };
        let mx = piece
            .iter()
            .enumerate()
            .filter(|&(c, _)| v0 + c != t)
            .fold(f32::MIN, |a, (_, &l)| a.max(l));
        let m_old = if k == 0 { mx } else { m };
        let m_new = m_old.max(mx);
        let acc: f32 = piece
            .iter()
            .enumerate()
            .filter(|&(c, &l)| live(c, l))
            .map(|(_, &l)| kexp(l - m_new))
            .sum();
        let s_old = if k == 0 { 0.0 } else { s };
        let carried = if guard != Guard::None && bits_is_zero(s_old) {
            0.0
        } else {
            s_old * kexp(m_old - m_new)
        };
        s = carried + acc;
        m = m_new;
    }
    (m, s)
}

/// The new kernels' arithmetic for one row, in f32 (the vocabulary walk's
/// chunking changes only the order of the sum). A row with no finite
/// non-target class has `s == 0` and a zero gradient.
fn excluded_target_grad(z: &[f32], t: usize) -> Vec<f32> {
    let (m, s) = lse_walk(z, t, z.len(), Guard::Bits);
    if bits_is_zero(s) {
        return vec![0.0; z.len()];
    }
    let lo = m + s.ln();
    let margin = z[t] - lo;
    let e = (-margin.abs()).exp();
    let q = if margin >= 0.0 { e / (1.0 + e) } else { 1.0 / (1.0 + e) };
    z.iter()
        .enumerate()
        .map(|(j, &l)| {
            if j == t {
                -q
            } else if l.to_bits() == NEG_INF_BITS {
                0.0
            } else {
                kexp(l - lo) * q
            }
        })
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

/// `-inf` logits across a chunked walk: no infinity may reach `exp`, whatever
/// the chunk boundaries (a chunk of only `-inf`, of only the target, a row
/// whose other classes are all `-inf`). The unguarded first version, which fed
/// `-inf` to `exp`, ends with `s = NaN` on every such row under the hostile
/// `kexp`: the failure `minus_infinite_logits_and_a_row_of_only_the_target`
/// reported on the device ("row 0's loss is not finite").
#[test]
fn minus_infinite_logits_never_reach_exp_in_any_chunking() {
    let ninf = f32::NEG_INFINITY;
    let rows: [(&[f32], usize); 5] = [
        (&[ninf, 2.0, ninf, ninf, ninf], 1),
        (&[1.0, ninf, 2.0, ninf, 0.0], 2),
        (&[0.5, ninf, ninf, ninf, ninf], 0),
        (&[1.0, 2.0, 3.0, ninf, ninf], 0),
        (&[ninf, ninf, 4.0, 1.0, ninf], 3),
    ];
    for (z, t) in rows {
        let zd = as_f64(z);
        let (loss, want) = ce_oracle(&zd, t);
        let finite_others = z.iter().enumerate().any(|(j, &l)| j != t && l.is_finite());
        let neg_inf_other = z.iter().enumerate().any(|(j, &l)| j != t && l == ninf);
        for chunk in [5, 3, 2, 1] {
            let (m, s) = lse_walk(z, t, chunk, Guard::Bits);
            assert!(s.is_finite(), "{z:?} t={t} chunk={chunk}: s = {s}");
            if finite_others {
                let lo = f64::from(m) + f64::from(s).ln();
                assert!(
                    (lo - common::ce::other_classes_lse(&zd, t)).abs() < 1e-5,
                    "{z:?} t={t} chunk={chunk}"
                );
            } else {
                assert_eq!(s, 0.0, "{z:?} t={t} chunk={chunk}: nothing finite to sum");
                assert_eq!(loss, 0.0);
            }
            // The guard is what keeps `s` finite: without it a `-inf` other
            // class makes `s` NaN, exactly when there is one.
            for guard in [Guard::None, Guard::Compare] {
                let s = lse_walk(z, t, chunk, guard).1;
                assert_eq!(
                    s.is_nan(),
                    neg_inf_other && guard == Guard::None,
                    "{z:?} t={t} chunk={chunk} {guard:?}: s = {s}"
                );
            }
        }
        let got = excluded_target_grad(z, t);
        let peak = want.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        for (j, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "{z:?} t={t} class {j}: {g}");
            assert!(
                (f64::from(g) - w).abs() <= 3e-5 * peak + 1e-36,
                "{z:?} t={t} class {j}: {g} want {w}"
            );
        }
    }
}

/// A NaN logit must reach the sum and poison it, in every chunking and in
/// every position (non-target, in a chunk of its own, next to `-inf`), so the
/// host reports the row as not finite. A float compare as the `-inf` guard
/// (`Compare`) drops it and `s` comes out a clean number: the mirror of the
/// device failures "a NaN logit gave a number" and "got loss 0".
#[test]
fn a_nan_logit_poisons_the_sum_and_a_compare_guard_would_drop_it() {
    let (nan, ninf) = (f32::NAN, f32::NEG_INFINITY);
    let rows: [(&[f32], usize); 4] = [
        (&[1.0, nan, 2.0, 0.0], 2),
        (&[nan, 3.0, 1.0, 0.0], 1),
        (&[1.0, ninf, nan, ninf], 0),
        (&[ninf, nan, ninf, ninf], 0),
    ];
    for (z, t) in rows {
        for chunk in [4, 2, 1] {
            let s = lse_walk(z, t, chunk, Guard::Bits).1;
            assert!(s.is_nan(), "{z:?} t={t} chunk={chunk}: s = {s}, the NaN was dropped");
            let dropped = lse_walk(z, t, chunk, Guard::Compare).1;
            assert!(
                !dropped.is_nan(),
                "{z:?} t={t} chunk={chunk}: the compare guard model should drop NaN"
            );
        }
        assert!(excluded_target_grad(z, t).iter().any(|g| g.is_nan()) || z[t].is_nan());
    }
}

/// Ties the mirror to the kernel: the guards the mirror models are the ones
/// `kernels/cross_entropy.metal` spells, and the float compare they replaced
/// is gone. Removing a guard from the kernel fails here, on any host.
#[test]
fn the_kernel_spells_the_guards_the_mirror_models() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/cross_entropy.metal");
    let src = std::fs::read_to_string(&path).expect("kernel source");
    let code: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    for needle in [
        "as_type<uint>(x) == 0xff800000u",
        "(as_type<uint>(x) << 1) == 0u",
        "v0 + c != t && !ce_is_neg_inf(row[c])",
        "ce_is_zero(s_old) ? 0.0f",
        "if (ce_is_zero(sn))",
        "ce_is_neg_inf(z) ? 0.0f : precise::exp(z - lo) * q",
        "if (v0 + c != t)",
    ] {
        assert!(code.contains(needle), "kernel no longer spells `{needle}`");
    }
    assert!(!code.contains("> -FLT_MAX"), "a float compare with -FLT_MAX is back");
    assert!(!code.contains("== 0.0f"), "a float compare of s with 0 is back");
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
