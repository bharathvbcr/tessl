//! `adamw_step` refuses a hyperparameter whose f32 word overflows, with an `Err` and before any
//! dispatch; it still accepts the largest and the smallest words an f32 can hold.
//!
//! The kernel reads seven f32 scalars the host forms in f64 (decay, 1 - beta1, beta2, 1 - beta2,
//! lr / bc1, sqrt(bc2), eps) and `grad_scale`. Both betas lie in [0, 1), so their words cannot
//! overflow; lr, weight decay, eps and grad_scale can. One test per case, so that a case that
//! panics (the behaviour before the fix) does not hide the cases after it.

mod common;

use std::sync::Arc;

use common::with_gpu;
use tessl::qwen35_adamw::{adamw_step, AdamWHyper};
use tessl::{GpuRuntime, Tensor};

fn operands(rt: &Arc<GpuRuntime>) -> (Tensor, Tensor, Tensor, Tensor) {
    let t = |x: f32| {
        let t = rt.alloc_tensor_f32(&[8]).unwrap();
        t.write_f32(&[x; 8]).unwrap();
        t
    };
    (t(1.0), t(0.5), t(0.0), t(0.0))
}

/// One step: its result, the dispatches it encoded, and the parameter before and after.
fn step(
    rt: &Arc<GpuRuntime>,
    hyper: &AdamWHyper,
    at: u64,
    weight_decay: f32,
) -> (Result<(), String>, usize, Vec<f32>, Vec<f32>) {
    let (mut p, g, mut m, mut v) = operands(rt);
    let before = p.read_f32().unwrap();
    rt.take_dispatch_count();
    let result = adamw_step(rt, &mut p, &g, &mut m, &mut v, hyper, at, weight_decay);
    let dispatched = rt.take_dispatch_count();
    rt.synchronize().unwrap();
    (result, dispatched, before, p.read_f32().unwrap())
}

fn refused(hyper: AdamWHyper, at: u64, weight_decay: f32) {
    with_gpu(|rt| {
        let (result, dispatched, before, after) = step(rt, &hyper, at, weight_decay);
        let err = result.expect_err("the step must be refused");
        assert!(err.contains("adamw_step"), "{err}");
        assert_eq!(dispatched, 0, "a refused step must encode nothing");
        assert_eq!(after, before, "the parameter moved");
    });
}

fn accepted(hyper: AdamWHyper, at: u64, weight_decay: f32) {
    with_gpu(|rt| {
        let (result, dispatched, _, _) = step(rt, &hyper, at, weight_decay);
        result.unwrap_or_else(|e| panic!("must be accepted: {e}"));
        // The positive control for the refusals' zero: an accepted step is counted.
        assert!(dispatched >= 1, "an accepted step must encode its dispatch");
    });
}

#[test]
fn lr_whose_step_size_overflows_is_refused() {
    refused(
        AdamWHyper {
            lr: 1e300,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn lr_f32_max_at_step_one_is_refused_because_lr_over_bc1_overflows() {
    refused(
        AdamWHyper {
            lr: f64::from(f32::MAX),
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn weight_decay_whose_decay_word_overflows_is_refused() {
    // lr 1e20 / bc1 (0.1) is a finite step size; only 1 - lr * weight_decay overflows.
    refused(
        AdamWHyper {
            lr: 1e20,
            ..AdamWHyper::default()
        },
        1,
        1e30,
    );
}

#[test]
fn eps_that_overflows_f32_is_refused() {
    refused(
        AdamWHyper {
            eps: 1e300,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn eps_that_underflows_to_zero_in_f32_is_refused() {
    refused(
        AdamWHyper {
            eps: 1e-50,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn grad_scale_that_overflows_f32_is_refused() {
    refused(
        AdamWHyper {
            grad_scale: 1e300,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn grad_scale_just_past_f32_max_is_refused() {
    refused(
        AdamWHyper {
            grad_scale: 3.5e38,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn an_overflowing_word_is_refused_even_when_nothing_would_move() {
    with_gpu(|rt| {
        let base = rt.alloc_tensor_f32(&[4]).unwrap();
        let empty = || base.try_view(&[0], 0).unwrap();
        let (mut p, g, mut m, mut v) = (empty(), empty(), empty(), empty());
        let hyper = AdamWHyper {
            lr: 1e300,
            ..AdamWHyper::default()
        };
        let err = adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 1, 0.0).expect_err("lr 1e300, empty");
        assert!(err.contains("adamw_step"), "{err}");
    });
}

#[test]
fn eps_f32_max_is_accepted() {
    accepted(
        AdamWHyper {
            eps: f64::from(f32::MAX),
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn a_subnormal_eps_is_accepted() {
    accepted(
        AdamWHyper {
            eps: 1e-40,
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn grad_scale_f32_max_is_accepted() {
    accepted(
        AdamWHyper {
            grad_scale: f64::from(f32::MAX),
            ..AdamWHyper::default()
        },
        1,
        0.0,
    );
}

#[test]
fn lr_f32_max_is_accepted_once_bc1_has_rounded_to_one() {
    accepted(
        AdamWHyper {
            lr: f64::from(f32::MAX),
            ..AdamWHyper::default()
        },
        1000,
        0.0,
    );
}

#[test]
fn weight_decay_f32_max_at_lr_one_is_accepted() {
    accepted(
        AdamWHyper {
            lr: 1.0,
            ..AdamWHyper::default()
        },
        1,
        f32::MAX,
    );
}
