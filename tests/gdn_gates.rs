//! **K7 — gate production.** The CPU oracle for `alpha` / `beta`, and the traps
//! around it.
//!
//! The published fixture corpus stores *post*-gate `alpha` and `beta`, so it
//! cannot test gate production at all: every case in `gdn_fixtures.rs` consumes
//! gates that were already computed somewhere else. K7 is the kernel that
//! computes them, which makes it the one kernel in the set with no golden. This
//! file is that golden.
//!
//! It runs no GPU code — no `with_gpu` — so it runs on any host, and it compares
//! no kernel to anything. It pins the reference a kernel will later be judged by.
//!
//! **Why this reference exists at all.** The build plan specifies K7 as
//!
//! ```text
//! alpha = exp(-exp(A_log) * softplus(a_gate + dt_bias))
//! ```
//!
//! which is the Mamba2 / SSD decay from a *different* mixer in the same source
//! file. The gated delta rule's gate is a plain sigmoid of a per-head-biased
//! logit (`nanolab/mixers.py:803-804`, recorded in
//! `qwen-decision/AUDIT/gdn-reference-and-contracts.md` §4.2). `A_log` and
//! `dt_bias` are not GDN parameters; `decay_bias` is.
//!
//! The audit calls this the most dangerous single error in the plan, and the
//! tests below show why it earns that. With the zero-initialised parameters
//! nanolab ships, the refuted formula is **exactly `sigmoid(-a_gate)`** — an
//! algebraic identity, not an approximation. So the two formulas agree to the
//! last bit at `a_gate = 0`, which is where an untrained checkpoint sits, and
//! separate into mirror images only once training moves the gate off zero. A
//! kernel written from the plan and smoke-tested at initialisation would look
//! correct.
//!
//! That is the same shape as two other defects already recorded in this project:
//! the published and repo delta rules coincide exactly at `t = 0`, and the
//! conformal nonconformity and probability scales coincide exactly at
//! `q_hat = 0.5`. Three separate defects, each invisible at the one input a
//! smoke test reaches for first.

mod common;
use common::gdn::{alpha_mamba2_refuted, gates_published, sequential_f64, Dims, GateBias, GateClamp, Problem, Rule};
use common::{random_f32, U_F32};

/// `ln(1e-4 / (1 - 1e-4))` — the logit at which `alpha`'s floor starts to bite.
/// Stated to full f64 precision so `alpha_floor_engages_at_the_logit_of_the_bound`
/// can bisect onto it instead of accepting a rounded neighbourhood.
const ALPHA_FLOOR_LOGIT: f64 = -9.210_240_366_975_85;

fn dims(b: usize, h: usize, l: usize) -> Dims {
    Dims { b, h, l, d: 2 }
}

/// Zero biases of the right length, which is what nanolab initialises.
fn zeros(h: usize) -> Vec<f64> {
    vec![0.0; h]
}

/// Gates laid out `[B, T, H]`, the layout the input projection's split produces.
fn bth(b: usize, l: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
    let mut v = vec![0.0f32; b * l * h];
    for bi in 0..b {
        for t in 0..l {
            for hi in 0..h {
                v[(bi * l + t) * h + hi] = f(bi, t, hi);
            }
        }
    }
    v
}

fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// What a K7 kernel computing in f32 would produce. Not a reference — a model of
/// the thing being measured, used only by the saturation tests.
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn assert_close(what: &str, got: f64, want: f64, rel: f64) {
    let tol = rel * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, gap {:.3e} > tol {tol:.3e}",
        (got - want).abs()
    );
}

// ------------------------------------------------------ the formula itself ---

#[test]
fn published_gate_is_the_sigmoid_of_the_biased_logit() {
    // One batch, one head, so the layout question cannot confound this.
    let d = dims(1, 1, 5);
    let logits: [f32; 5] = [0.0, 1.0, -1.0, 2.0, -3.5];
    let a = logits.to_vec();
    let b: Vec<f32> = logits.iter().map(|x| -x).collect();
    let (alpha, beta) = gates_published(
        &a,
        &b,
        d,
        GateBias {
            decay: &zeros(1),
            update: &zeros(1),
        },
        GateClamp::PUBLISHED,
    );

    for (i, &x) in logits.iter().enumerate() {
        assert_close(&format!("alpha[{i}]"), alpha[i], sigmoid_f64(f64::from(x)), 0.0);
        assert_close(&format!("beta[{i}]"), beta[i], sigmoid_f64(f64::from(-x)), 0.0);
    }
    // Spot-check against values computed outside this file, so the test is not
    // just `sigmoid` compared with itself.
    assert_close("alpha at logit 0", alpha[0], 0.5, 0.0);
    assert_close("alpha at logit 1", alpha[1], 0.731_058_578_630_004_9, 1e-15);
    assert_close("alpha at logit -1", alpha[2], 0.268_941_421_369_995_1, 1e-15);
    assert_close("alpha at logit 2", alpha[3], 0.880_797_077_977_882_3, 1e-15);
}

#[test]
fn a_gate_drives_alpha_and_b_gate_drives_beta() {
    // Swapping the two gate tensors must change the answer, or the reference is
    // reading one of them twice.
    let d = dims(1, 1, 3);
    let a = vec![2.0f32, 2.0, 2.0];
    let b = vec![-2.0f32, -2.0, -2.0];
    let bias = GateBias {
        decay: &zeros(1),
        update: &zeros(1),
    };
    let (alpha, beta) = gates_published(&a, &b, d, bias, GateClamp::PUBLISHED);
    let (alpha_swapped, beta_swapped) = gates_published(&b, &a, d, bias, GateClamp::PUBLISHED);
    assert_close("alpha", alpha[0], sigmoid_f64(2.0), 0.0);
    assert_close("beta", beta[0], sigmoid_f64(-2.0), 0.0);
    assert_close("alpha swapped", alpha_swapped[0], beta[0], 0.0);
    assert_close("beta swapped", beta_swapped[0], alpha[0], 0.0);
    assert!(
        alpha[0] != beta[0],
        "test is degenerate: pick gate values whose sigmoids differ"
    );
}

// ------------------------------------------------------------- the layout ---

#[test]
fn gate_layout_maps_bth_input_onto_bhl_output() {
    // A value that encodes its own coordinates, so a wrong index is not merely
    // detectable but readable from the failure message.
    let (b, h, l) = (2usize, 3usize, 5usize);
    let d = dims(b, h, l);
    let code = |bi: usize, t: usize, hi: usize| (bi * 100 + t * 10 + hi) as f32;
    let a = bth(b, l, h, code);
    let (alpha, _) = gates_published(
        &a,
        &a,
        d,
        GateBias {
            decay: &zeros(h),
            update: &zeros(h),
        },
        GateClamp::PUBLISHED,
    );

    for bi in 0..b {
        for hi in 0..h {
            for t in 0..l {
                let want = sigmoid_f64(f64::from(code(bi, t, hi)));
                let got = alpha[(bi * h + hi) * l + t];
                assert_close(&format!("alpha[b{bi} h{hi} t{t}]"), got, want, 0.0);
            }
        }
    }
}

#[test]
fn reading_the_gate_without_transposing_gives_a_different_answer() {
    // `[B,T,H]` and `[B,H,L]` hold the same number of elements and the same set
    // of values; only the pairing changes. Both are in range, both are finite,
    // and no shape or dtype check can tell them apart. That is the whole reason
    // `gates_published` owns the conversion rather than trusting a caller.
    let (b, h, l) = (1usize, 4usize, 4usize);
    let d = dims(b, h, l);
    let a = bth(b, l, h, |_, t, hi| (t as f32) - (hi as f32) * 0.5);
    let bias = GateBias {
        decay: &zeros(h),
        update: &zeros(h),
    };
    let (alpha, _) = gates_published(&a, &a, d, bias, GateClamp::PUBLISHED);

    // The mistake: treat the `[B,T,H]` buffer as if it were already `[B,H,L]`.
    let naive: Vec<f64> = a.iter().map(|&x| sigmoid_f64(f64::from(x))).collect();

    assert_eq!(
        alpha.len(),
        naive.len(),
        "the two readings are the same size, which is the hazard"
    );
    let mut same = 0usize;
    for (i, (&got, &wrong)) in alpha.iter().zip(naive.iter()).enumerate() {
        if got == wrong {
            same += 1;
        }
        assert!(
            (0.0..=1.0).contains(&wrong),
            "the wrong reading is still in range at [{i}]: {wrong}"
        );
    }
    // With b == 1 and this generator the transposed and untransposed readings
    // coincide exactly on the diagonal (t == hi) and nowhere else.
    assert_eq!(
        same,
        h.min(l),
        "only the diagonal should coincide; {same} of {} elements matched",
        alpha.len()
    );
}

#[test]
#[should_panic(expected = "expected 12 for [2,3,2]")]
fn a_misshaped_gate_is_refused_rather_than_reinterpreted() {
    let d = dims(2, 2, 3);
    let short = vec![0.0f32; 6];
    let _ = gates_published(
        &short,
        &short,
        d,
        GateBias {
            decay: &zeros(2),
            update: &zeros(2),
        },
        GateClamp::PUBLISHED,
    );
}

// --------------------------------------------------------------- the bias ---

#[test]
fn bias_is_per_head_and_broadcasts_over_batch_and_position() {
    let (b, h, l) = (2usize, 3usize, 4usize);
    let d = dims(b, h, l);
    let a = vec![0.0f32; b * l * h]; // every logit zero, so only the bias moves alpha
    let decay = vec![-1.0, 0.0, 2.5];
    let update = vec![0.5, -0.5, 0.0];
    let (alpha, beta) = gates_published(
        &a,
        &a,
        d,
        GateBias {
            decay: &decay,
            update: &update,
        },
        GateClamp::PUBLISHED,
    );

    for bi in 0..b {
        for hi in 0..h {
            for t in 0..l {
                let idx = (bi * h + hi) * l + t;
                assert_close(
                    &format!("alpha[b{bi} h{hi} t{t}]"),
                    alpha[idx],
                    sigmoid_f64(decay[hi]),
                    0.0,
                );
                assert_close(
                    &format!("beta[b{bi} h{hi} t{t}]"),
                    beta[idx],
                    sigmoid_f64(update[hi]),
                    0.0,
                );
            }
        }
    }
    // And the heads must actually differ, or the loop above is vacuous.
    assert!(
        alpha[0] != alpha[l] && alpha[l] != alpha[2 * l],
        "distinct per-head biases must give distinct alphas"
    );
}

#[test]
#[should_panic(expected = "indexed by position instead of head")]
fn a_bias_indexed_by_position_is_refused() {
    // `[L]` instead of `[H]`. Same dtype, plausible length, silently wrong.
    let (b, h, l) = (1usize, 2usize, 6usize);
    let d = dims(b, h, l);
    let a = vec![0.0f32; b * l * h];
    let _ = gates_published(
        &a,
        &a,
        d,
        GateBias {
            decay: &zeros(l),
            update: &zeros(h),
        },
        GateClamp::PUBLISHED,
    );
}

// ------------------------------------------------------------- the clamps ---

#[test]
fn only_alphas_floor_can_change_a_value() {
    // Sweep well past both saturation thresholds in both directions. If any of
    // the other three bounds could alter a value, an unclamped and a clamped
    // sweep would differ somewhere other than at alpha's floor.
    let n = 4001usize;
    let logits: Vec<f32> = (0..n).map(|i| -50.0 + 100.0 * (i as f32) / ((n - 1) as f32)).collect();
    let d = dims(1, 1, n);
    let bias = GateBias {
        decay: &zeros(1),
        update: &zeros(1),
    };
    let wide = GateClamp {
        alpha: (f64::NEG_INFINITY, f64::INFINITY),
        beta: (f64::NEG_INFINITY, f64::INFINITY),
    };
    let (alpha_clamped, beta_clamped) = gates_published(&logits, &logits, d, bias, GateClamp::PUBLISHED);
    let (alpha_raw, beta_raw) = gates_published(&logits, &logits, d, bias, wide);

    let mut floor_fired = 0usize;
    for i in 0..n {
        assert_eq!(
            beta_clamped[i], beta_raw[i],
            "beta[{i}] changed at logit {}: neither of its bounds can be exceeded",
            logits[i]
        );
        if alpha_clamped[i] != alpha_raw[i] {
            floor_fired += 1;
            assert_eq!(
                alpha_clamped[i], 1e-4,
                "the only clamp with work to do is alpha's floor, at logit {}",
                logits[i]
            );
            assert!(
                alpha_raw[i] < 1e-4,
                "alpha's floor fired at logit {} on a value already above it",
                logits[i]
            );
        }
        assert!(
            alpha_raw[i] <= 1.0 && beta_raw[i] <= 1.0,
            "sigmoid exceeded 1.0 at logit {} — the ceiling clamps would then matter",
            logits[i]
        );
        assert!(
            alpha_raw[i] >= 0.0 && beta_raw[i] >= 0.0,
            "sigmoid fell below 0.0 at logit {}",
            logits[i]
        );
    }
    assert!(
        floor_fired > 0,
        "the sweep never reached alpha's floor, so it proves nothing"
    );
}

#[test]
fn alpha_floor_engages_at_the_logit_of_the_bound() {
    let clamped_at = |x: f64| -> bool {
        let d = dims(1, 1, 1);
        let g = [x as f32];
        let (alpha, _) = gates_published(
            &g,
            &g,
            d,
            GateBias {
                decay: &zeros(1),
                update: &zeros(1),
            },
            GateClamp::PUBLISHED,
        );
        alpha[0] <= 1e-4
    };

    assert!(clamped_at(-9.3), "clamp must fire below the logit");
    assert!(!clamped_at(-9.0), "clamp must not fire above the logit");

    // Bisect for the crossing rather than assert a hand-derived value against
    // itself. The gate is stored as f32, which caps the resolution.
    let (mut lo, mut hi) = (-10.0f64, -8.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if clamped_at(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let crossing = 0.5 * (lo + hi);
    assert!(
        (crossing - ALPHA_FLOOR_LOGIT).abs() < 1e-6,
        "floor engages at {crossing:.9}, expected ln(1e-4/(1-1e-4)) = {ALPHA_FLOOR_LOGIT:.9}"
    );
}

#[test]
fn the_clamp_is_a_parameter_not_a_constant() {
    // A caller-supplied clamp must be honoured, or `GateClamp::PUBLISHED` is
    // decoration and a fixture generated under other bounds cannot be read.
    let d = dims(1, 1, 3);
    let g = [0.0f32, 0.0, 0.0];
    let tight = GateClamp {
        alpha: (0.6, 0.9),
        beta: (0.1, 0.4),
    };
    let (alpha, beta) = gates_published(
        &g,
        &g,
        d,
        GateBias {
            decay: &zeros(1),
            update: &zeros(1),
        },
        tight,
    );
    assert_close("alpha raised to its floor", alpha[0], 0.6, 0.0);
    assert_close("beta lowered to its ceiling", beta[0], 0.4, 0.0);
}

// --------------------------------------------------------- the saturation ---

#[test]
fn alpha_saturates_to_exactly_one_later_in_f64_than_in_f32() {
    let bisect_f64 = || {
        let (mut lo, mut hi) = (0.0f64, 100.0f64);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if sigmoid_f64(mid) == 1.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let bisect_f32 = || {
        let (mut lo, mut hi) = (0.0f32, 100.0f32);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if sigmoid_f32(mid) == 1.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    };

    let t64 = bisect_f64();
    let t32 = f64::from(bisect_f32());

    // ln(1 / 2^-53) = 36.7368, ln(1 / 2^-24) = 16.6355.
    assert!(
        (t64 - 36.736_800_569_677_1).abs() < 1e-3,
        "f64 saturation at {t64:.6}, expected ~36.7368"
    );
    assert!(
        (t32 - 16.635_532_333_438_684).abs() < 1e-3,
        "f32 saturation at {t32:.6}, expected ~16.6355"
    );
    assert!(
        t32 < t64,
        "an f32 kernel must saturate first, or this hazard does not exist"
    );

    // In the window between them the two disagree, and the reference is the one
    // that has not saturated.
    let x = 0.5 * (t32 + t64);
    assert_eq!(sigmoid_f32(x as f32), 1.0, "f32 should have saturated at {x}");
    assert!(sigmoid_f64(x) < 1.0, "f64 must not have saturated at {x}");
}

#[test]
fn a_saturated_alpha_gap_compounds_past_the_fixture_bound() {
    // The worst case is a logit just past f32's threshold, where the f64 value is
    // still `1 - u_f32` but f32 has already rounded to 1.0. alpha multiplies the
    // state once per step, so the gap on the decay product is `(1-g)^L`.
    const L: i32 = 8191; // the longest published fixture
    let x = 17.0f64; // just past 16.6355
    let a_f32 = f64::from(sigmoid_f32(x as f32));
    let a_f64 = sigmoid_f64(x);
    assert_eq!(a_f32, 1.0, "premise: f32 has saturated at {x}");
    assert!(a_f64 < 1.0, "premise: f64 has not saturated at {x}");

    let per_step = a_f32 - a_f64;
    let product_gap = 1.0 - a_f64.powi(L);
    let in_u = product_gap / U_F32;

    assert!(
        per_step < U_F32,
        "per-step gap {per_step:.3e} should be under one u_f32 ({U_F32:.3e})"
    );
    assert!(
        in_u > 64.0,
        "the compounded gap is {in_u:.1} u_f32 over {L} steps; were it under the 64 u_f32 \
         fixture bound this hazard would need no separate note"
    );
    // Roughly L * per_step, which is what makes the growth predictable.
    let linear = f64::from(L) * per_step;
    assert!(
        (product_gap / linear - 1.0).abs() < 0.01,
        "compounded gap {product_gap:.3e} should track L*per_step {linear:.3e} to 1%"
    );
}

// ----------------------------------------------- the plan's refuted form ---

#[test]
fn the_refuted_formula_is_the_sign_flipped_sigmoid_at_zero_init() {
    // exp(-exp(0) * softplus(x)) = exp(-ln(1 + e^x)) = 1 / (1 + e^x) = sigmoid(-x).
    // An identity, so the divergence is exactly characterisable: the plan's
    // formula decays fastest precisely where the published one decays slowest.
    let logits: Vec<f32> = (0..161).map(|i| -8.0 + 0.1 * i as f32).collect();
    let d = dims(1, 1, logits.len());
    let got = alpha_mamba2_refuted(&logits, d, &zeros(1), &zeros(1));
    for (i, &x) in logits.iter().enumerate() {
        assert_close(
            &format!("refuted[{i}] at logit {x}"),
            got[i],
            sigmoid_f64(f64::from(-x)),
            1e-14,
        );
    }
}

#[test]
fn the_two_formulas_agree_exactly_at_a_zero_gate() {
    // The trap. nanolab zero-initialises decay_bias, so an untrained checkpoint
    // whose a_gate sits near zero cannot distinguish the plan's K7 from the real
    // one. A smoke test on fresh weights would pass either way.
    let d = dims(2, 2, 3);
    let zero = vec![0.0f32; d.gate_len()];
    let (published, _) = gates_published(
        &zero,
        &zero,
        d,
        GateBias {
            decay: &zeros(2),
            update: &zeros(2),
        },
        GateClamp::PUBLISHED,
    );
    let refuted = alpha_mamba2_refuted(&zero, d, &zeros(2), &zeros(2));
    assert_eq!(published.len(), refuted.len());
    for (i, (&p, &r)) in published.iter().zip(refuted.iter()).enumerate() {
        assert_close(&format!("published vs refuted at [{i}]"), r, p, 1e-15);
        assert_close(&format!("published[{i}]"), p, 0.5, 0.0);
    }
}

#[test]
fn the_two_formulas_diverge_once_the_gate_leaves_zero() {
    // Trained-scale logits. The mirror-image identity means the worst case is the
    // largest |logit| in the sample, and the divergence is order-1 — not a
    // tolerance question.
    let (b, h, l) = (1usize, 2usize, 64usize);
    let d = dims(b, h, l);
    // random_f32 is uniform in [-1, 1); scale to a plausible post-training range.
    let logits: Vec<f32> = random_f32(d.gate_len(), 0x00C0_FFEE).iter().map(|x| x * 4.0).collect();
    let (published, _) = gates_published(
        &logits,
        &logits,
        d,
        GateBias {
            decay: &zeros(h),
            update: &zeros(h),
        },
        GateClamp::PUBLISHED,
    );
    let refuted = alpha_mamba2_refuted(&logits, d, &zeros(h), &zeros(h));

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for (i, (&p, &r)) in published.iter().zip(refuted.iter()).enumerate() {
        let gap = (p - r).abs();
        if gap > worst {
            worst = gap;
            worst_at = i;
        }
    }
    assert!(
        worst > 0.9,
        "worst absolute gap is only {worst:.6} at [{worst_at}] (published {}, refuted {}); \
         with |logit| up to 4 the mirror identity should push it near 1.0",
        published[worst_at],
        refuted[worst_at]
    );

    // And it is a *mirror*, not an offset: the two orderings are opposed, so a
    // kernel using the wrong formula ranks every pair of positions in reverse.
    let sign = |v: &[f64], i: usize, j: usize| (v[i] - v[j]).signum();
    let mut opposed = 0usize;
    let mut compared = 0usize;
    for i in 0..published.len() {
        for j in (i + 1)..published.len() {
            if published[i] == published[j] {
                continue;
            }
            compared += 1;
            if sign(&published, i, j) != sign(&refuted, i, j) {
                opposed += 1;
            }
        }
    }
    assert!(compared > 0, "no comparable pairs; the sample is degenerate");
    assert_eq!(
        opposed, compared,
        "every pair should be ordered oppositely; {opposed} of {compared} were"
    );
}

#[test]
fn a_nonzero_a_log_scales_the_refuted_decay() {
    // Present so the refuted reference is a faithful model of the Mamba2 form
    // rather than only its zero-init special case: A_log multiplies the rate, so
    // a larger A_log decays harder.
    let d = dims(1, 1, 4);
    let logits = [0.0f32, 1.0, 2.0, 3.0];
    let slow = alpha_mamba2_refuted(&logits, d, &[-1.0], &zeros(1));
    let fast = alpha_mamba2_refuted(&logits, d, &[1.0], &zeros(1));
    for i in 0..4 {
        assert!(
            fast[i] < slow[i],
            "exp(A_log)=e should decay harder than exp(A_log)=1/e at [{i}]: {} vs {}",
            fast[i],
            slow[i]
        );
        assert!(
            (0.0..=1.0).contains(&fast[i]),
            "decay out of range at [{i}]: {}",
            fast[i]
        );
    }
    // dt_bias shifts the softplus argument, so it also moves the result.
    let shifted = alpha_mamba2_refuted(&logits, d, &[-1.0], &[2.0]);
    assert!(shifted[0] < slow[0], "a positive dt_bias must increase the decay rate");
}

#[test]
fn the_refuted_reference_holds_up_at_a_large_logit() {
    // A trained `a_gate` is not bounded, and the textbook `ln(1 + exp(x))`
    // overflows above x = 709. Paired with a small `exp(A_log)` the overflow is
    // not merely imprecise, it inverts the answer: `exp(-1e-3 * inf)` is 0, so a
    // gate that should decay to 0.449 reads as total decay instead.
    let d = dims(1, 1, 1);
    let big = [800.0f32];
    let rate = 1e-3f64;
    let got = alpha_mamba2_refuted(&big, d, &[rate.ln()], &zeros(1));

    // softplus(800) = 800 to well within f64 precision, so the expected value is
    // exp(-rate * 800).
    let want = (-rate * 800.0f64).exp();
    assert_close("refuted alpha at logit 800", got[0], want, 1e-12);
    assert!(
        got[0] > 0.44 && got[0] < 0.46,
        "expected ~0.449, got {} — an overflowing softplus returns 0.0 here",
        got[0]
    );
    assert!(got[0].is_finite(), "non-finite decay");
}

#[test]
fn gates_stay_in_bounds_at_the_extremes_of_f32() {
    // A kernel that produces a gate logit at the edge of the format must still
    // yield a usable gate. Nothing here may be NaN: alpha multiplies the state
    // once per step, so a single NaN gate erases the whole sequence.
    let logits: Vec<f32> = vec![
        f32::MIN,
        -1e30,
        -800.0,
        -100.0,
        -9.21,
        0.0,
        9.21,
        100.0,
        800.0,
        1e30,
        f32::MAX,
    ];
    let d = dims(1, 1, logits.len());
    let (alpha, beta) = gates_published(
        &logits,
        &logits,
        d,
        GateBias {
            decay: &zeros(1),
            update: &zeros(1),
        },
        GateClamp::PUBLISHED,
    );
    for (i, &x) in logits.iter().enumerate() {
        assert!(
            (1e-4..=1.0).contains(&alpha[i]),
            "alpha at logit {x} is {} — outside the clamped range",
            alpha[i]
        );
        assert!(
            (0.0..=1.0).contains(&beta[i]),
            "beta at logit {x} is {} — outside [0, 1]",
            beta[i]
        );
    }
    // The extremes must actually reach the ends, or the sweep is not extreme.
    assert_eq!(alpha[0], 1e-4, "the most negative logit must hit the floor");
    assert_eq!(alpha[logits.len() - 1], 1.0, "the most positive logit must saturate");
}

#[test]
#[should_panic(expected = "non-finite gate logit at [b0 t1 h0]")]
fn a_nan_gate_logit_is_refused_not_clamped() {
    // `f64::clamp` returns NaN for a NaN input, so the bounds are no defence.
    // Refusing here is the difference between a named bad input and a kernel
    // comparison that reports a mismatch on every element of the sequence.
    let d = dims(1, 1, 3);
    let g = [0.0f32, f32::NAN, 0.0];
    let _ = gates_published(
        &g,
        &g,
        d,
        GateBias {
            decay: &zeros(1),
            update: &zeros(1),
        },
        GateClamp::PUBLISHED,
    );
}

#[test]
#[should_panic(expected = "decay bias[1] is not finite")]
fn a_non_finite_bias_is_refused() {
    // A trainable parameter that has diverged. Checking the gate logits alone
    // would not catch it, because the bias is added *after* they are read, and a
    // per-head bias poisons every position of that head rather than one element.
    let d = dims(1, 3, 2);
    let g = vec![0.0f32; d.gate_len()];
    let _ = gates_published(
        &g,
        &g,
        d,
        GateBias {
            decay: &[0.0, f64::NAN, 0.0],
            update: &zeros(3),
        },
        GateClamp::PUBLISHED,
    );
}

#[test]
#[should_panic(expected = "dt_bias[0] is not finite")]
fn a_non_finite_refuted_parameter_is_refused() {
    let d = dims(1, 1, 2);
    let g = vec![0.0f32; d.gate_len()];
    let _ = alpha_mamba2_refuted(&g, d, &zeros(1), &[f64::INFINITY]);
}

// --------------------------------------------- composition with the rule ---

#[test]
fn produced_gates_drive_the_recurrence() {
    // End to end: produce gates from logits, then run the published delta rule on
    // them. This establishes that the gate reference's output layout is the one
    // `sequential_f64` consumes — a transpose bug would survive every test above
    // if the two halves disagreed here.
    let (b, h, l, dd) = (1usize, 2usize, 6usize, 4usize);
    let d = Dims { b, h, l, d: dd };
    let a_logits = random_f32(d.gate_len(), 0x51DE);
    let b_logits = random_f32(d.gate_len(), 0x51DF);
    let (alpha64, beta64) = gates_published(
        &a_logits,
        &b_logits,
        d,
        GateBias {
            decay: &zeros(h),
            update: &zeros(h),
        },
        GateClamp::PUBLISHED,
    );

    // The published fixtures store post-gate alpha/beta as f32, so this narrowing
    // is not a shortcut in the test — it is the precision the corpus itself
    // carries. A K7 kernel cannot be held to better than f32 against a fixture,
    // because the fixture's own alpha is already rounded.
    let alpha: Vec<f32> = alpha64.iter().map(|&x| x as f32).collect();
    let beta: Vec<f32> = beta64.iter().map(|&x| x as f32).collect();

    let q = random_f32(d.qkv_len(), 1);
    let k = random_f32(d.qkv_len(), 2);
    let v = random_f32(d.qkv_len(), 3);
    let p = Problem {
        q: &q,
        k: &k,
        v: &v,
        alpha: &alpha,
        beta: &beta,
        dims: d,
    };
    let y = sequential_f64(&p, Rule::Published, GateClamp::PUBLISHED);
    assert_eq!(y.len(), d.qkv_len());
    assert!(y.iter().all(|x| x.is_finite()), "non-finite output");
    assert!(
        y.iter().any(|&x| x != 0.0),
        "the recurrence produced all zeros, so the gates cannot have reached it"
    );

    // Substituting the refuted gates changes the output, so K7's formula is
    // observable downstream and is not a free choice.
    let refuted64 = alpha_mamba2_refuted(&a_logits, d, &zeros(h), &zeros(h));
    let refuted: Vec<f32> = refuted64.iter().map(|&x| x as f32).collect();
    let p_wrong = Problem { alpha: &refuted, ..p };
    let y_wrong = sequential_f64(&p_wrong, Rule::Published, GateClamp::PUBLISHED);
    let worst = y
        .iter()
        .zip(y_wrong.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let scale = y.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(
        worst > 0.01 * scale,
        "the refuted gate changed the output by only {worst:.3e} against a scale of \
         {scale:.3e}; a test that cannot see K7's formula cannot guard it"
    );
}
