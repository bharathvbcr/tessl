//! Adversarial checks for the shared float home. No GPU, no extra crate.

use numcheck::{
    check_subnormal, compare_oracle, plant_at, plant_f32, preserving_equal, reduction_tolerance, CheckError, NonFinite,
    Plant, Side,
};

fn pin(n: usize, scale: f64) -> f64 {
    4.0 * f64::from(f32::EPSILON) * (n as f64).sqrt() * scale
}

#[test]
fn plants_nan_both_infinities_and_a_real_subnormal() {
    let nan = plant_f32(Plant::Nan);
    let pos = plant_f32(Plant::PosInf);
    let neg = plant_f32(Plant::NegInf);
    let sub = plant_f32(Plant::Subnormal);

    assert!(nan.is_nan());
    assert_eq!(pos, f32::INFINITY);
    assert!(pos.is_sign_positive());
    assert_eq!(neg, f32::NEG_INFINITY);
    assert!(neg.is_sign_negative());
    assert!(sub.is_finite(), "a subnormal is finite");
    assert!(sub.is_subnormal());
    assert_ne!(sub, 0.0);
    assert_eq!(sub.to_bits(), 1);

    assert!(check_subnormal(sub).is_ok());
    assert_eq!(check_subnormal(0.0), Err(CheckError::NotSubnormal));
    assert_eq!(check_subnormal(f32::MIN_POSITIVE), Err(CheckError::NotSubnormal));
    assert_eq!(
        check_subnormal(f32::NAN),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::Nan,
            side: Side::Left,
        })
    );
    assert_eq!(
        check_subnormal(f32::INFINITY),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::PosInf,
            side: Side::Left,
        })
    );
    assert_eq!(
        check_subnormal(f32::NEG_INFINITY),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::NegInf,
            side: Side::Left,
        })
    );
}

#[test]
fn plant_at_writes_only_the_named_slot_and_rejects_a_bad_index() {
    let mut buf = [1.0, 2.0, 3.0];
    plant_at(&mut buf, 2, Plant::Nan).expect("last index");
    assert_eq!(buf[0], 1.0);
    assert_eq!(buf[1], 2.0);
    assert!(buf[2].is_nan());

    plant_at(&mut buf, 0, Plant::Subnormal).expect("first index");
    assert!(buf[0].is_subnormal());
    assert_eq!(buf[1], 2.0);

    assert_eq!(
        plant_at(&mut buf, 3, Plant::PosInf),
        Err(CheckError::OutOfRange { index: 3, len: 3 })
    );
    assert_eq!(
        plant_at(&mut [], 0, Plant::NegInf),
        Err(CheckError::OutOfRange { index: 0, len: 0 })
    );
    assert_eq!(buf[1], 2.0, "a refused plant does not write");
}

#[test]
fn tolerance_matches_the_four_epsilon_sqrt_formula_and_does_not_widen() {
    for (n, scale) in [(0usize, 1.0f64), (1, 1.0), (4096, 2.5), (1000, 1e-40)] {
        let got = reduction_tolerance(n, scale);
        assert_eq!(got.to_bits(), pin(n, scale).to_bits(), "n={n} scale={scale}");
    }
    assert!(reduction_tolerance(4, -1.0) < 0.0);
    assert!(reduction_tolerance(1, f64::NAN).is_nan());
    assert!(reduction_tolerance(10_000_000, 1.0).is_finite());
    let wide = 8.0 * f64::from(f32::EPSILON) * 4096.0_f64.sqrt() * 2.5;
    assert!(reduction_tolerance(4096, 2.5) < wide);
}

#[test]
fn oracle_accepts_an_f32_sum_inside_the_bound_and_rejects_the_rest() {
    let terms: Vec<f32> = (0..128).map(|i| 0.01 * i as f32).collect();
    let scale: f64 = terms.iter().map(|v| f64::from(v.abs())).sum();
    let oracle: f64 = terms.iter().map(|v| f64::from(*v)).sum();
    let got: f32 = terms.iter().copied().sum();
    compare_oracle(&[got], &[oracle], terms.len(), scale).expect("f32 sum vs f64 sum");

    let tol = reduction_tolerance(1, 1.0);
    assert!(compare_oracle(&[1.0], &[1.0 + tol], 1, 1.0).is_ok());
    assert_eq!(
        compare_oracle(&[1.0], &[1.0 + tol + tol], 1, 1.0),
        Err(CheckError::OutsideTolerance { index: 0 })
    );

    let sub = plant_f32(Plant::Subnormal);
    assert_eq!(
        compare_oracle(&[0.0], &[f64::from(sub)], 1, f64::from(sub)),
        Err(CheckError::OutsideTolerance { index: 0 }),
        "flushing a subnormal to zero does not pass the oracle"
    );

    assert_eq!(
        compare_oracle(&[f32::NAN], &[0.0], 1, 1.0),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::Nan,
            side: Side::Left,
        })
    );
    assert_eq!(
        compare_oracle(&[f32::INFINITY], &[f64::INFINITY], 1, 1.0),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::PosInf,
            side: Side::Left,
        })
    );
    assert_eq!(
        compare_oracle(&[f32::NEG_INFINITY], &[0.0], 1, 1.0),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::NegInf,
            side: Side::Left,
        })
    );
    assert_eq!(
        compare_oracle(&[0.0], &[f64::NAN], 1, 1.0),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::Nan,
            side: Side::Right,
        })
    );
    assert_eq!(compare_oracle(&[], &[], 1, 1.0), Err(CheckError::Empty));
    assert_eq!(
        compare_oracle(&[1.0, 2.0], &[1.0], 1, 1.0),
        Err(CheckError::LenMismatch { left: 2, right: 1 })
    );
    assert_eq!(
        compare_oracle(&[1.0], &[1.0], 4, -1.0),
        Err(CheckError::ToleranceNotUsable)
    );
    assert_eq!(
        compare_oracle(&[1.0], &[1.0], 1, f64::NAN),
        Err(CheckError::ToleranceNotUsable)
    );
    assert_eq!(
        compare_oracle(&[1.0], &[1.0], 1, f64::INFINITY),
        Err(CheckError::ToleranceNotUsable)
    );
}

#[test]
fn preserving_pair_is_bitwise_and_a_non_finite_result_fails() {
    let values = [0.0f32, -2.5, 1.0e-20, f32::MIN_POSITIVE];
    preserving_equal(&values, &values).expect("same bits");

    let mut flipped = values;
    flipped[2] = f32::from_bits(values[2].to_bits() + 1);
    assert_eq!(
        preserving_equal(&values, &flipped),
        Err(CheckError::BitsDiffer { index: 2 })
    );
    assert_eq!(
        preserving_equal(&[0.0], &[-0.0]),
        Err(CheckError::BitsDiffer { index: 0 }),
        "signed zero is a different bit pattern"
    );

    let nan = plant_f32(Plant::Nan);
    assert_eq!(
        preserving_equal(&[nan], &[nan]),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::Nan,
            side: Side::Left,
        })
    );
    assert_eq!(
        preserving_equal(&[1.0], &[f32::INFINITY]),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::PosInf,
            side: Side::Right,
        })
    );
    assert_eq!(
        preserving_equal(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY]),
        Err(CheckError::NonFinite {
            index: 0,
            kind: NonFinite::NegInf,
            side: Side::Left,
        })
    );
    assert_eq!(preserving_equal(&[], &[]), Err(CheckError::Empty));
    assert_eq!(
        preserving_equal(&[1.0], &[1.0, 1.0]),
        Err(CheckError::LenMismatch { left: 1, right: 2 })
    );
}
