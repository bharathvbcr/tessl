//! **K3 — the GDN state update.** Its f64 oracle, and the bound it must be
//! judged by.
//!
//! `AUDIT/tessl-integration.md:572` is the requirement: *"a GDN state update
//! reassociated across lanes needs an f64 reference too."* `:639` says what shape
//! the bound takes: *"a delta-rule state update is a K-term accumulation"*, so it
//! is the GEMM idiom — `tolerance(k, mag, operand_u)` scaled by
//! `sum_k |a_k b_k|`, not a flat relative figure.
//!
//! `gdn_fixtures.rs` does not do that. It judges the chunked output by
//! `64 * u_f32 * max|y|`, where `max|y|` is the maximum over the **whole
//! tensor**. That is a global scale factor, not the per-element magnitude the
//! accumulation error is actually proportional to, and its 64 was calibrated from
//! an observed worst case rather than derived. This file measures what the
//! derived bound would be instead, and what the difference costs.
//!
//! It runs no GPU code, so it runs on any host. It is the oracle a K3 kernel will
//! be compared against; it compares no kernel to anything yet.

mod common;
use common::gdn::{
    case_names, load_case, state_update_f64, Case, Dims, GateClamp, Problem, Rule, StateUpdate, SumOrder,
};
use common::U_F32;

/// f64 unit roundoff, 2^-53.
const U_F64: f64 = 1.110_223_024_625_157e-16;

/// `gamma_n = n*u / (1 - n*u)` from the classical dot-product error analysis.
///
/// The same function `tests/common/mod.rs` uses for the GEMMs, restated here
/// because that one adds `k + 8` for tessl's split-K lanes — a GEMM-specific
/// allowance that has no meaning for this recurrence.
fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    assert!(nu < 1.0, "error bound degenerate at n={n}");
    nu / (1.0 - nu)
}

fn run(c: &Case, order: SumOrder) -> StateUpdate {
    state_update_f64(&c.problem(), Rule::Published, GateClamp::PUBLISHED, order)
}

/// The corpus, loaded once per test that needs all of it.
fn all_cases() -> Vec<Case> {
    case_names().iter().map(|n| load_case(n)).collect()
}

// ------------------------------------------------ the oracle is the old one ---

#[test]
fn the_sequential_order_is_bit_identical_to_the_reference_it_replaced() {
    // `sequential_f64` now delegates to `state_update_f64`. If that delegation
    // changed a single bit, the fixture suite's bit-exact golden test would fail
    // too -- but it would fail there without saying why, so the invariant is
    // pinned at its source as well.
    for c in all_cases() {
        let got = run(&c, SumOrder::Sequential);
        assert_eq!(got.y.len(), c.golden.len(), "{}: length against the golden", c.name);
        let mismatches = got
            .y
            .iter()
            .zip(c.golden.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "{}: {mismatches} elements differ from the published f64 golden \
             in their bit pattern",
            c.name
        );
    }
}

#[test]
fn the_magnitude_never_understates_its_own_result() {
    // |sum_p x_p| <= sum_p |x_p| exactly; both sides are d-term f64 sums here, so
    // rounding can put the computed `y_mag` a few ulps under a computed |y| that
    // happens to sit right at the bound. A few ulps of slack, not a free pass.
    for c in all_cases() {
        let u = run(&c, SumOrder::Sequential);
        for (i, (&y, &mag)) in u.y.iter().zip(u.y_mag.iter()).enumerate() {
            let slack = 8.0 * U_F64 * mag.max(y.abs());
            assert!(
                y.abs() <= mag + slack,
                "{}: |y[{i}]| = {} exceeds its magnitude {mag} by more than {slack:.3e}",
                c.name,
                y.abs()
            );
        }
    }
}

#[test]
fn the_state_starts_at_zero_so_the_first_step_reads_nothing() {
    // At t = 0 the state is zero, so `pred = S^T k` is exactly zero and the two
    // rules coincide -- the same t = 0 blind spot the rule divergence has. Pinned
    // here so a reference that silently carried state across the (b, h) loop, or
    // failed to reset it, is caught by an exact zero rather than by a tolerance.
    for c in all_cases() {
        let u = run(&c, SumOrder::Sequential);
        let d = c.dims.d;
        for bi in 0..c.dims.b {
            for hi in 0..c.dims.h {
                let base = ((bi * c.dims.h + hi) * c.dims.l) * d;
                for n in 0..d {
                    assert_eq!(
                        u.pred_mag[base + n],
                        0.0,
                        "{}: pred_mag at t=0 [b{bi} h{hi} n{n}] must be exactly zero",
                        c.name
                    );
                }
            }
        }
    }
}

// ------------------------------------------- closed form, and the odd tail ---

#[test]
fn a_single_step_matches_its_closed_form_at_odd_and_even_d() {
    // At t = 0 the state is zero, so the whole recurrence collapses to something
    // checkable by hand:
    //   pred = 0,  delta[n] = beta * v[n],  S[p][n] = beta * k[p] * v[n]
    //   y[n]  = sum_p S[p][n] * q[p] = beta * v[n] * <k, q>
    // An independent expression for the same number, which the recurrence cannot
    // satisfy by accident.
    //
    // Every fixture is d = 8, and 8 halves to 4, 2, 1 without ever leaving an odd
    // count -- so the corpus never exercises `sum_tree`'s odd tail, and the claim
    // that the tail is carried rather than dropped would be untested. d = 7 and
    // d = 5 reach it; d = 8 keeps the even path covered here too.
    for d in [5usize, 7, 8] {
        let dims = Dims { b: 1, h: 1, l: 1, d };
        let q: Vec<f32> = (0..d).map(|i| 0.5 + i as f32 * 0.25).collect();
        let k: Vec<f32> = (0..d).map(|i| 1.0 - i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..d).map(|i| 0.3 * (i as f32 + 1.0)).collect();
        let alpha = vec![0.75f32];
        let beta = vec![0.5f32];
        let p = Problem {
            q: &q,
            k: &k,
            v: &v,
            alpha: &alpha,
            beta: &beta,
            dims,
        };

        let kq: f64 = (0..d).map(|i| f64::from(k[i]) * f64::from(q[i])).sum();
        for order in [SumOrder::Sequential, SumOrder::PairwiseTree] {
            let u = state_update_f64(&p, Rule::Published, GateClamp::PUBLISHED, order);
            for (n, &vn) in v.iter().enumerate() {
                let want = f64::from(beta[0]) * f64::from(vn) * kq;
                let tol = 16.0 * U_F64 * want.abs().max(1.0);
                assert!(
                    (u.y[n] - want).abs() <= tol,
                    "d={d} {order:?} y[{n}]: got {}, closed form {want}, gap {:.3e}",
                    u.y[n],
                    (u.y[n] - want).abs()
                );
            }
            // The tail being dropped would lose one product from every dot
            // product, so the closed form above already catches it -- but only if
            // the value it drops is nonzero. Assert that it is.
            assert!(
                f64::from(k[d - 1]) * f64::from(q[d - 1]) != 0.0,
                "d={d}: the last term is zero, so dropping it would be invisible"
            );
        }
    }
}

#[test]
fn the_second_step_pins_pred_mag_against_its_closed_form() {
    // `pred_mag` needs a check of its own. The t = 0 assertion above is satisfied
    // by a `pred_mag` that is zero everywhere, so on its own it cannot tell a
    // correct one from an absent one -- a mutation run proved exactly that by
    // deleting the computation and watching every test still pass.
    //
    // At t = 1 there is a closed form. After step 0 the state is
    // `S[p][n] = beta0 * k0[p] * v0[n]`, so
    //   pred[n]     = sum_p S[p][n] * k1[p] = beta0 * v0[n] * <k0, k1>
    //   pred_mag[n] = sum_p |S[p][n] * k1[p]| = |beta0 * v0[n]| * sum_p |k0[p] k1[p]|
    // The two differ whenever <k0, k1> is not already a sum of like signs, which
    // is what makes this a check on the magnitude and not a second check on
    // `pred`. `k0` and `k1` below are chosen with mixed signs so the two diverge.
    for d in [5usize, 8] {
        let dims = Dims { b: 1, h: 1, l: 2, d };
        // Step 0 operands, then step 1 operands, laid out [B,H,L,D].
        let k0: Vec<f32> = (0..d).map(|i| 1.0 - i as f32 * 0.3).collect();
        // Alternating signs, so the products of `k0` and `k1` largely cancel and
        // `sum |k0 k1|` pulls far away from `|sum k0 k1|`. Without that the two
        // closed forms coincide and the test would only be re-checking `pred`.
        let k1: Vec<f32> = (0..d)
            .map(|i| {
                let m = 0.5 + i as f32 * 0.2;
                if i % 2 == 0 {
                    m
                } else {
                    -m
                }
            })
            .collect();
        let v0: Vec<f32> = (0..d).map(|i| 0.25 * (i as f32 + 1.0)).collect();
        let mut k = k0.clone();
        k.extend_from_slice(&k1);
        let mut v = v0.clone();
        v.extend(std::iter::repeat_n(0.1f32, d));
        let q: Vec<f32> = std::iter::repeat_n(0.5f32, 2 * d).collect();
        let alpha = vec![1.0f32, 1.0];
        let beta = vec![0.5f32, 0.5];
        let p = Problem {
            q: &q,
            k: &k,
            v: &v,
            alpha: &alpha,
            beta: &beta,
            dims,
        };

        let b0 = f64::from(beta[0]);
        let signed: f64 = (0..d).map(|i| f64::from(k0[i]) * f64::from(k1[i])).sum();
        let absolute: f64 = (0..d).map(|i| (f64::from(k0[i]) * f64::from(k1[i])).abs()).sum();
        assert!(
            absolute > 1.5 * signed.abs(),
            "d={d}: the test is degenerate unless sum|k0 k1| ({absolute}) clearly \
             exceeds |sum k0 k1| ({}); otherwise pred_mag and |pred| coincide and \
             this proves nothing",
            signed.abs()
        );

        let u = state_update_f64(&p, Rule::Published, GateClamp::PUBLISHED, SumOrder::Sequential);
        for (n, &v0n) in v0.iter().enumerate() {
            let want = (b0 * f64::from(v0n)).abs() * absolute;
            let got = u.pred_mag[d + n]; // t = 1
            let tol = 16.0 * U_F64 * want.max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "d={d} pred_mag[t=1][{n}]: got {got}, closed form {want}, gap {:.3e}",
                (got - want).abs()
            );
            assert!(want > 0.0, "d={d} n={n}: closed form is zero, so the check is vacuous");
        }
    }
}

// ------------------------------------------------------- the reassociation ---

#[test]
fn reassociating_the_dot_products_is_measured_not_assumed() {
    // The requirement from AUDIT/tessl-integration.md:572. Both runs are f64, so
    // any difference here is *purely* summation order -- precision is held
    // constant. That is what makes this an answer about reassociation rather than
    // about f32.
    println!(
        "\n{:<16} {:>5} {:>6} {:>12} {:>12} {:>12}",
        "case", "d", "L", "worst |dy|", "/y_mag", "in u_f64"
    );
    let mut corpus_worst_rel = 0.0f64;
    let mut corpus_worst_case = String::new();
    for c in all_cases() {
        let seq = run(&c, SumOrder::Sequential);
        let tree = run(&c, SumOrder::PairwiseTree);
        let mut worst_abs = 0.0f64;
        let mut worst_rel = 0.0f64;
        for i in 0..seq.y.len() {
            let err = (seq.y[i] - tree.y[i]).abs();
            worst_abs = worst_abs.max(err);
            // Judged against the magnitude the accumulation error scales with,
            // which is the whole point of carrying `y_mag`.
            let mag = seq.y_mag[i].max(tree.y_mag[i]);
            if mag > 0.0 {
                worst_rel = worst_rel.max(err / mag);
            } else {
                assert_eq!(err, 0.0, "{}: nonzero error at [{i}] with zero magnitude", c.name);
            }
        }
        println!(
            "{:<16} {:>5} {:>6} {:>12.3e} {:>12.3e} {:>12.1}",
            c.name,
            c.dims.d,
            c.dims.l,
            worst_abs,
            worst_rel,
            worst_rel / U_F64
        );
        if worst_rel > corpus_worst_rel {
            corpus_worst_rel = worst_rel;
            corpus_worst_case = c.name.clone();
        }
    }
    println!(
        "\nworst over the corpus: {:.3e} = {:.1} u_f64 on {corpus_worst_case}",
        corpus_worst_rel,
        corpus_worst_rel / U_F64
    );
    // Measured: worst is 6.0 u_f64 on L8191; the quietest is 3.0 u on
    // L65_tinyalpha. 32 u_f64 leaves 5x headroom over the observed worst, and the
    // two real defects this corpus has already caught clear it by eleven orders
    // and more -- a wrong rule diverges by 1.4e-1 and a missed alpha clamp by
    // 8.4e-5, against a bound of 3.6e-15.
    const REASSOC_BOUND: f64 = 32.0 * U_F64;
    assert!(
        corpus_worst_rel < REASSOC_BOUND,
        "reassociation alone moved the f64 answer by {corpus_worst_rel:.3e} relative to \
         its own magnitude on {corpus_worst_case}, over the {REASSOC_BOUND:.3e} bound. \
         For a reordering of f64 sums that is too large to be rounding, and means the \
         two orders are not computing the same thing."
    );
}

#[test]
fn reassociation_error_does_not_compound_over_the_sequence() {
    // The question a K3 author actually has to answer: does a lane reassociation
    // need a longer-sequence allowance? Measured: no, and by a wide margin.
    //
    // An error introduced into the state at step t reaches y at step t' scaled by
    // the product of alpha over the steps between, which is at most 1 and in
    // practice well below it. So the contributions form a damped sum rather than
    // an accumulating one, and the total stays bounded no matter how long the
    // sequence is. This is the opposite of the saturation hazard in
    // `gdn_gates.rs`, where the gap grows like L * u because alpha multiplies the
    // state every step with nothing to damp it.
    let worst_for = |name: &str| -> f64 {
        let c = load_case(name);
        let seq = run(&c, SumOrder::Sequential);
        let tree = run(&c, SumOrder::PairwiseTree);
        let mut worst = 0.0f64;
        for i in 0..seq.y.len() {
            let mag = seq.y_mag[i].max(tree.y_mag[i]);
            if mag > 0.0 {
                worst = worst.max((seq.y[i] - tree.y[i]).abs() / mag);
            }
        }
        worst
    };

    let short = worst_for("L1");
    let long = worst_for("L8191");
    let growth = long / short;
    println!(
        "\nL1 ({:.1} u_f64) -> L8191 ({:.1} u_f64): growth {growth:.2}x over 8191x the steps",
        short / U_F64,
        long / U_F64
    );
    assert!(short > 0.0, "L1 shows no reassociation at all; nothing to compare");
    assert!(
        growth < 10.0,
        "reassociation error grew {growth:.1}x from L=1 to L=8191. If it accumulated \
         rather than damping, the factor would be nearer 8191x, and a K3 bound would \
         have to scale with sequence length."
    );
}

// --------------------------------------------------- the shape of the bound ---

#[test]
fn the_derived_bound_and_the_corpus_global_bound_are_different_quantities() {
    // `gdn_fixtures.rs` uses `64 * u_f32 * max|y|` over the whole tensor. The
    // derived form is `gamma_d(u_f32) * y_mag` per element. Neither dominates the
    // other a priori: the global one is looser where |y| is small and the terms
    // cancel, and tighter where one element's magnitude greatly exceeds the
    // tensor's peak result. Both numbers are reported so a K3 author picks
    // knowingly.
    const CORPUS_FLAT: f64 = 64.0 * U_F32;
    let mut slack: Vec<(String, f64, f64)> = Vec::new();
    println!(
        "\n{:<16} {:>5} {:>11} {:>11} {:>11} {:>9}",
        "case", "d", "global bnd", "max derived", "min derived", "cancel"
    );
    for c in all_cases() {
        let u = run(&c, SumOrder::Sequential);
        let peak = u.y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let global = CORPUS_FLAT * peak;
        let g = gamma(c.dims.d, U_F32);

        let mut max_derived = 0.0f64;
        let mut min_derived = f64::INFINITY;
        let mut worst_cancel = 0.0f64;
        for (&y, &mag) in u.y.iter().zip(u.y_mag.iter()) {
            let derived = g * mag;
            max_derived = max_derived.max(derived);
            min_derived = min_derived.min(derived);
            if y.abs() > 0.0 {
                worst_cancel = worst_cancel.max(mag / y.abs());
            }
        }
        println!(
            "{:<16} {:>5} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.1}x",
            c.name, c.dims.d, global, max_derived, min_derived, worst_cancel
        );
        assert!(
            peak > 0.0,
            "{}: the whole output tensor is zero, so no bound means anything",
            c.name
        );
        assert!(
            min_derived > 0.0 && max_derived.is_finite(),
            "{}: derived bound is not a usable number",
            c.name
        );

        // The global bound is looser than arithmetic requires even at the element
        // where the derived bound is widest -- so the existing gate is not too
        // tight, and nothing here asks for it to be moved.
        assert!(
            global > max_derived,
            "{}: the global bound {global:.3e} is tighter than the derived bound at its \
             widest element {max_derived:.3e}. That would make the existing fixture gate \
             reject lawful f32 arithmetic, which is a different and worse problem than \
             the one this test is about.",
            c.name
        );

        // And far looser at the quiet end. What drives that slack is the dynamic
        // range of `y_mag` inside the tensor -- how far the smallest per-element
        // magnitude sits below the tensor's peak result -- and NOT how hard the
        // terms cancel. Those are different elements and the corpus separates
        // them plainly: L65 cancels hardest (1.9e5 : 1) yet has nearly the least
        // slack, while L65_tinyalpha cancels 58x less and has the most, because a
        // tiny alpha decays its state until some elements' magnitudes are
        // vanishing.
        let quiet_mag = u.y_mag.iter().cloned().fold(f64::INFINITY, f64::min);
        slack.push((c.name.clone(), global / min_derived, peak / quiet_mag));
    }

    slack.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nslack of the global bound over arithmetic, at each case's quietest element:");
    for (name, s, range) in &slack {
        println!("  {name:<16} {s:>12.1}x   (y_mag dynamic range {range:.1}x)");
    }
    let (worst_name, worst_slack, _) = &slack[0];
    let (floor_name, floor_slack, _) = &slack[slack.len() - 1];
    println!("\nmost slack: {worst_slack:.1}x on {worst_name}; least: {floor_slack:.1}x on {floor_name}");

    // The finding. A K3 defect confined to low-magnitude elements could be four
    // orders of magnitude worse than f32 arithmetic permits and still pass the
    // existing fixture gate.
    assert!(
        *worst_slack > 10_000.0,
        "expected the global bound to be >10000x looser than arithmetic requires \
         somewhere in the corpus, measured {worst_slack:.1}x on {worst_name}. If this \
         has dropped, the corpus changed and this file's header note is stale."
    );

    // Prove the mechanism rather than assert a correlation. Algebraically,
    //   slack = (64 u peak) / (gamma_d * min_mag),  gamma_d = d*u / (1 - d*u)
    //         = (64/d) * (peak / min_mag) * (1 - d*u)
    // and at d = 8 with d*u = 4.8e-7 the last factor is 1 to within 5e-7. So the
    // slack IS eight times the dynamic range, exactly, and nothing else.
    let d = 8usize;
    let expected_factor = 64.0 / d as f64 * (1.0 - d as f64 * U_F32);
    for (name, s, range) in &slack {
        let implied = s / range;
        assert!(
            (implied / expected_factor - 1.0).abs() < 1e-6,
            "{name}: slack/range is {implied:.6}, expected {expected_factor:.6}. The \
             slack is then not simply 64/d times the dynamic range, and the derivation \
             in the comment above is wrong."
        );
    }
    println!(
        "slack = {expected_factor:.4} x dynamic range, exactly, on all {} cases",
        slack.len()
    );
}

#[test]
fn cancellation_in_this_corpus_is_extreme_enough_to_decide_the_bound_shape() {
    // Why the bound shape is not academic. If the terms barely cancelled, `|y|`
    // and `sum_p |products|` would agree to within a small factor and any of the
    // three candidate bounds would do. Measured, they disagree by up to five
    // orders of magnitude, which is what rules out a per-element |y|-proportional
    // bound outright: at 1.9e5 : 1 it would demand accuracy no f32 kernel can
    // reach, and a suite built on it would fail on correct code.
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    for c in all_cases() {
        let u = run(&c, SumOrder::Sequential);
        for (&y, &mag) in u.y.iter().zip(u.y_mag.iter()) {
            if y.abs() > 0.0 && mag / y.abs() > worst {
                worst = mag / y.abs();
                worst_case = c.name.clone();
            }
        }
    }
    println!("\nworst cancellation: {worst:.1}x on {worst_case}");
    assert!(
        worst > 1_000.0,
        "worst cancellation over the corpus is only {worst:.1}x on {worst_case}. Below \
         about 1000x the three candidate bounds stop being meaningfully different and \
         this file's argument for carrying `y_mag` weakens."
    );
    // f32 cannot represent a cancellation of this depth in its result at all:
    // 1/1.9e5 is 5e-6, which is 88 u_f32, so the surviving result is only about
    // seven significant bits wide.
    let surviving_bits = -(1.0f64 / worst).log2();
    println!(
        "a result cancelled {worst:.0}x has lost {surviving_bits:.1} of its 24 f32 \
         significand bits"
    );
    assert!(
        surviving_bits > 10.0,
        "expected heavy bit loss from cancellation, got {surviving_bits:.1} bits"
    );
}

#[test]
fn the_corpus_is_d8_so_its_calibrated_bound_does_not_transfer_to_the_model() {
    // Qwen3.5-2B's GDN state is 16 x 128 x 128 (`docs/plan-corrections.md`), so
    // the model's inner dimension is 128. Every fixture here is d = 8. The
    // per-step accumulation budget scales with gamma_d, so a bound calibrated on
    // this corpus understates the model's by about 16x -- before the recurrence
    // depth is considered at all. A K3 kernel that passes at d = 8 is not thereby
    // known to pass at d = 128.
    const MODEL_D: usize = 128;
    let cases = all_cases();
    for c in &cases {
        assert_eq!(
            c.dims.d, 8,
            "{}: this test's conclusion is about a d = 8 corpus; a case at d = {} \
             means the corpus grew and the note above needs rewriting",
            c.name, c.dims.d
        );
    }
    let at_fixture = gamma(8, U_F32);
    let at_model = gamma(MODEL_D, U_F32);
    let ratio = at_model / at_fixture;
    println!(
        "\ngamma_8 = {at_fixture:.3e} ({:.1} u_f32), gamma_128 = {at_model:.3e} \
         ({:.1} u_f32), ratio {ratio:.2}x",
        at_fixture / U_F32,
        at_model / U_F32
    );
    assert!(
        ratio > 15.0 && ratio < 17.0,
        "expected the d = 8 -> d = 128 budget ratio to be about 16, got {ratio}"
    );
}
