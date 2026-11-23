//! Attack on `adamw_step`'s hyperparameter checks (lane F, 3ce999f): hyperparameters at and past the
//! limits of an f32 word, the empty-tensor path, and the device's subnormal flush.
//!
//! The property most cases check is "refused, or the parameter stays finite": a step that is
//! accepted must not poison a parameter.

mod common;

use std::sync::Arc;

use common::with_gpu;
use tessl::qwen35_adamw::{adamw_step, AdamWHyper};
use tessl::{GpuRuntime, Tensor};

fn filled(rt: &Arc<GpuRuntime>, n: usize, x: f32) -> Tensor {
    let t = rt.alloc_tensor_f32(&[n]).unwrap();
    t.write_f32(&vec![x; n]).unwrap();
    t
}

/// One step on `p = 1`, gradient `g`, zero moments. `Ok` carries the parameter after the step.
fn run(hyper: &AdamWHyper, at: u64, weight_decay: f32, g: f32) -> Result<Vec<f32>, String> {
    with_gpu(|rt| {
        let mut p = filled(rt, 8, 1.0);
        let grad = filled(rt, 8, g);
        let mut m = filled(rt, 8, 0.0);
        let mut v = filled(rt, 8, 0.0);
        rt.take_dispatch_count();
        let r = adamw_step(rt, &mut p, &grad, &mut m, &mut v, hyper, at, weight_decay);
        let dispatched = rt.take_dispatch_count();
        rt.synchronize().unwrap();
        match r {
            Ok(()) => Ok(p.read_f32().unwrap()),
            Err(e) => {
                assert_eq!(dispatched, 0, "a refused step encoded {dispatched} dispatches: {e}");
                Err(e)
            }
        }
    })
}

/// Refused with an `adamw_step` error, or accepted with every parameter finite.
fn refused_or_finite(hyper: AdamWHyper, at: u64, weight_decay: f32, g: f32) {
    match run(&hyper, at, weight_decay, g) {
        Ok(after) => assert!(
            after.iter().all(|x| x.is_finite()),
            "accepted {hyper:?} step {at} wd {weight_decay} g {g}, parameter is now {after:?}"
        ),
        Err(e) => assert!(e.contains("adamw_step"), "{e}"),
    }
}

fn accepted_and_finite(hyper: AdamWHyper, at: u64, weight_decay: f32, g: f32) -> Vec<f32> {
    let after = run(&hyper, at, weight_decay, g).unwrap_or_else(|e| panic!("must be accepted: {e}"));
    assert!(after.iter().all(|x| x.is_finite()), "parameter is now {after:?}");
    after
}

// ---------------------------------------------------------------- subnormal eps ---

/// A subnormal eps survives the host's non-zero test (`w[6] == 0.0`) but the device flushes it to
/// zero, so a zero gradient gives `0 / (0 + 0)`: NaN in every parameter.
#[test]
fn subnormal_eps_with_a_zero_gradient_does_not_poison_the_parameter() {
    refused_or_finite(
        AdamWHyper {
            eps: 1e-40,
            ..AdamWHyper::default()
        },
        1,
        0.0,
        0.0,
    );
}

#[test]
fn smallest_subnormal_eps_with_a_zero_gradient_does_not_poison_the_parameter() {
    refused_or_finite(
        AdamWHyper {
            eps: 1e-45,
            ..AdamWHyper::default()
        },
        1,
        0.0,
        0.0,
    );
}

/// The same flush with a gradient whose square underflows: v is 0, the denominator is the flushed eps.
#[test]
fn subnormal_eps_with_a_tiny_gradient_does_not_poison_the_parameter() {
    refused_or_finite(
        AdamWHyper {
            eps: 1e-40,
            ..AdamWHyper::default()
        },
        1,
        0.0,
        1e-30,
    );
}

/// Positive control: the smallest normal eps with a zero gradient leaves the parameter at 1.
#[test]
fn smallest_normal_eps_with_a_zero_gradient_leaves_the_parameter_alone() {
    let after = accepted_and_finite(
        AdamWHyper {
            eps: f64::from(f32::MIN_POSITIVE),
            ..AdamWHyper::default()
        },
        1,
        0.0,
        0.0,
    );
    assert!(after.iter().all(|&x| x == 1.0), "{after:?}");
}

// ---------------------------------------------------------------- betas near 1 ---

/// beta 1 ulp below 1 in f64 (1 - 2^-53): 1 - beta is 1.1e-16, bc is 1.1e-16 at step 1, beta2 rounds to
/// 1.0 in f32. The step is lr * sign(g) up to rounding.
#[test]
fn betas_one_ulp_below_one_take_a_finite_step_of_about_lr() {
    let below_one = 1.0 - f64::EPSILON / 2.0;
    assert!(below_one < 1.0);
    for at in [1u64, 2, 1000] {
        let hyper = AdamWHyper {
            beta1: below_one,
            beta2: below_one,
            ..AdamWHyper::default()
        };
        let after = accepted_and_finite(hyper, at, 0.0, 0.5);
        if at == 1 {
            for x in &after {
                assert!((x - 0.999).abs() < 1e-4, "step {at}: {x}, want about 1 - lr");
            }
        }
    }
}

#[test]
fn betas_zero_take_a_finite_step() {
    let hyper = AdamWHyper {
        beta1: 0.0,
        beta2: 0.0,
        ..AdamWHyper::default()
    };
    accepted_and_finite(hyper, 1, 0.0, 0.5);
}

// ---------------------------------------------------------------- step ---

#[test]
fn step_u64_max_is_accepted_and_finite() {
    accepted_and_finite(AdamWHyper::default(), u64::MAX, 0.0, 0.5);
}

#[test]
fn step_u64_max_with_betas_one_ulp_below_one_is_refused_or_finite() {
    let below_one = 1.0 - f64::EPSILON / 2.0;
    refused_or_finite(
        AdamWHyper {
            beta1: below_one,
            beta2: below_one,
            ..AdamWHyper::default()
        },
        u64::MAX,
        0.0,
        0.5,
    );
}

#[test]
fn step_zero_is_refused() {
    let e = run(&AdamWHyper::default(), 0, 0.0, 0.5).unwrap_err();
    assert!(e.contains("adamw_step"), "{e}");
}

// ---------------------------------------------------------------- lr and decay ---

/// lr / bc1 stays finite at f32::MAX / 2 with beta1 0.9 at step 1 (5e38 would overflow; 1.5e37 / 0.1 does not).
#[test]
fn a_large_lr_whose_step_size_just_fits_is_accepted() {
    let hyper = AdamWHyper {
        lr: 3.0e37,
        ..AdamWHyper::default()
    };
    let r = run(&hyper, 1, 0.0, 0.5);
    match r {
        Ok(after) => assert!(after.iter().all(|x| x.is_finite()), "{after:?}"),
        Err(e) => panic!("lr 3e37 / bc1 0.1 = 3e38 fits f32 and must be accepted: {e}"),
    }
}

#[test]
fn a_subnormal_step_size_is_refused_or_still_finite() {
    refused_or_finite(
        AdamWHyper {
            lr: 1e-40,
            ..AdamWHyper::default()
        },
        1,
        0.0,
        0.5,
    );
}

#[test]
fn a_subnormal_weight_decay_is_refused_or_finite() {
    refused_or_finite(AdamWHyper::default(), 1, 1e-40, 0.5);
}

#[test]
fn weight_decay_past_one_over_lr_flips_the_sign_and_stays_finite() {
    // decay = 1 - 1e-3 * 5e3 = -4: legal in torch.
    refused_or_finite(AdamWHyper::default(), 1, 5.0e3, 0.5);
}

#[test]
fn weight_decay_nan_inf_and_negative_zero() {
    for wd in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        let e = run(&AdamWHyper::default(), 1, wd, 0.5).unwrap_err();
        assert!(e.contains("adamw_step"), "wd {wd}: {e}");
    }
    // -0.0 is not negative.
    accepted_and_finite(AdamWHyper::default(), 1, -0.0, 0.5);
}

// ---------------------------------------------------------------- grad_scale ---

#[test]
fn grad_scale_non_finite_or_negative_is_refused() {
    for gs in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, -f64::MIN_POSITIVE] {
        let hyper = AdamWHyper {
            grad_scale: gs,
            ..AdamWHyper::default()
        };
        let e = run(&hyper, 1, 0.0, 0.5).unwrap_err();
        assert!(e.contains("adamw_step"), "grad_scale {gs}: {e}");
    }
}

#[test]
fn grad_scale_subnormal_or_underflowing_to_zero_is_refused_or_finite() {
    for gs in [1e-40, 1e-50, f64::MIN_POSITIVE] {
        refused_or_finite(
            AdamWHyper {
                grad_scale: gs,
                ..AdamWHyper::default()
            },
            1,
            0.0,
            0.5,
        );
    }
}

// ---------------------------------------------------------------- empty tensors ---

fn empties(rt: &Arc<GpuRuntime>) -> (Tensor, Tensor, Tensor, Tensor) {
    let base = rt.alloc_tensor_f32(&[4]).unwrap();
    let e = || base.try_view(&[0], 0).unwrap();
    (e(), e(), e(), e())
}

#[test]
fn an_empty_tensor_refuses_every_bad_hyperparameter_and_encodes_nothing() {
    with_gpu(|rt| {
        let bad: Vec<(&str, AdamWHyper, u64, f32)> = vec![
            (
                "eps underflow",
                AdamWHyper {
                    eps: 1e-50,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
            (
                "grad_scale NaN",
                AdamWHyper {
                    grad_scale: f64::NAN,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
            (
                "grad_scale overflow",
                AdamWHyper {
                    grad_scale: 1e300,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
            ("step 0", AdamWHyper::default(), 0, 0.0),
            ("wd NaN", AdamWHyper::default(), 1, f32::NAN),
            (
                "wd overflow",
                AdamWHyper {
                    lr: 1e20,
                    ..AdamWHyper::default()
                },
                1,
                1e30,
            ),
            (
                "lr overflow",
                AdamWHyper {
                    lr: 1e300,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
            (
                "beta1 = 1",
                AdamWHyper {
                    beta1: 1.0,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
            (
                "eps 0",
                AdamWHyper {
                    eps: 0.0,
                    ..AdamWHyper::default()
                },
                1,
                0.0,
            ),
        ];
        for (what, hyper, at, wd) in bad {
            let (mut p, g, mut m, mut v) = empties(rt);
            rt.take_dispatch_count();
            let r = adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, at, wd);
            assert!(r.is_err(), "{what}: an empty tensor accepted it");
            assert_eq!(rt.take_dispatch_count(), 0, "{what}");
        }
    });
}

#[test]
fn an_empty_tensor_with_good_hyperparameters_is_a_no_op() {
    with_gpu(|rt| {
        let (mut p, g, mut m, mut v) = empties(rt);
        rt.take_dispatch_count();
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &AdamWHyper::default(), 1, 0.0).unwrap();
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn an_empty_tensor_on_another_runtime_or_with_another_shape_is_still_refused() {
    with_gpu(|rt| {
        let (mut p, g, mut m, mut v) = empties(rt);
        let other = rt.alloc_tensor_f32(&[3]).unwrap();
        let r = adamw_step(rt, &mut p, &other, &mut m, &mut v, &AdamWHyper::default(), 1, 0.0);
        assert!(r.is_err(), "shape [0] against [3]");
        drop(g);
    });
}

/// Lead's probe: zero gradient, zero moments, eps 1e-40 (subnormal, non-zero in f32), step 1. The
/// parameter value is logged; a refusal is reported and is not a failure.
#[test]
fn probe_subnormal_eps_zero_gradient_step_one_parameter_value() {
    let hyper = AdamWHyper {
        eps: 1e-40,
        ..AdamWHyper::default()
    };
    match run(&hyper, 1, 0.0, 0.0) {
        Ok(after) => {
            eprintln!("PROBE accepted; parameter after = {after:?}");
            assert!(
                after.iter().all(|x| x.is_finite()),
                "parameter went non-finite: {after:?}"
            );
        }
        Err(e) => eprintln!("PROBE refused: {e}"),
    }
}
