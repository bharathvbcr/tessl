//! The forecaster's host side (`tessl::timesfm3_forecast`) against upstream's
//! `TimesFM3Forecaster.predict_batch()` on the tiny fixtures, device-free.
//!
//! `make_reference.py` recorded every `decode()` call upstream's forecaster
//! made (inputs and output) and its final forecast. [`plan`] must make the
//! same calls (shapes and masks exactly, values to rounding), and [`finish`]
//! must turn upstream's own decode outputs into upstream's forecast, so the
//! model is not involved in either check.
//!
//! The cases: ragged contexts over two groups with a length-1 context and
//! leading, interior, trailing and all-NaN series; z-normalisation, symmetric
//! averaging and make_positive with an odd group size (a series and its
//! negation in different groups); two target variates with past-only and
//! past+future covariates on some series only and an all-NaN variate; edge
//! padding with unsorted quantiles; contexts past the global context.

#[path = "common/timesfm3.rs"]
mod tfm;

use tessl::timesfm3::Forecast;
use tessl::timesfm3::Timesfm3Config;
use tessl::timesfm3_forecast::{finish, plan, PredictOptions, Rows, Series};
use tessl::timesfm3_host::DecodeInput;

/// Interpolation and z-normalisation sum in f64 where numpy sums float32
/// pairwise: rounding apart, nothing more.
const TOL: f64 = 1e-6;

fn config() -> Timesfm3Config {
    Timesfm3Config::from_config_file(&tfm::tiny_dir().join("config.json")).unwrap()
}

fn mask_of(a: &tfm::Arr) -> Vec<u8> {
    tfm::mask(a)
}

#[test]
fn plan_makes_upstreams_decode_calls() {
    let cfg = config();
    let dir = tfm::tiny_dir();
    for case in tfm::FORECAST_TINY {
        let fc = tfm::forecast_case(&dir, case);
        let pl = plan(&cfg.host, &fc.series, fc.horizon, &fc.opts).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(pl.calls.len(), fc.call_horizons.len(), "{case}: decode calls");
        for (k, call) in pl.calls.iter().enumerate() {
            let what = format!("{case} call {k}");
            let t = tfm::load(&dir, case, &format!("call_{k}.target"));
            let (g, u, c) = (t.shape[0], t.shape[1], t.shape[2]);
            // The horizon decode() ran: the forecaster passes the patch-rounded
            // one, which decode() replaces by the past+future width when there
            // is one; its output's length says which.
            let ran = tfm::load(&dir, case, &format!("call_{k}.out")).shape[2];
            if tfm::maybe(&dir, case, &format!("call_{k}.pf")).is_none() {
                assert_eq!(
                    ran, fc.call_horizons[k],
                    "{what}: horizon without past+future covariates"
                );
            }
            assert_eq!(
                (call.batch, call.num_target, call.context, call.horizon),
                (g, u, c, ran),
                "{what}: (batch, targets, context, horizon)"
            );
            tfm::assert_close(&format!("{what} target"), &call.target, &t.data, TOL);
            // decode()'s global mask is the group padding; plan() folds it
            // into every variate's context mask.
            let pad = mask_of(&tfm::load(&dir, case, &format!("call_{k}.mask")));
            let want: Vec<u8> = (0..g)
                .flat_map(|b| (0..u).flat_map(move |_| (0..c).map(move |t| (b, t))))
                .map(|(b, t)| pad[b * c + t])
                .collect();
            assert_eq!(
                call.target_mask.as_deref(),
                Some(want.as_slice()),
                "{what}: target mask"
            );
            for (short, vals, msk, n, width) in [
                ("po", &call.past_only, &call.past_only_mask, call.num_past_only, c),
                (
                    "pf",
                    &call.past_future,
                    &call.past_future_mask,
                    call.num_past_future,
                    c + call.horizon,
                ),
            ] {
                match tfm::maybe(&dir, case, &format!("call_{k}.{short}")) {
                    None => assert!(vals.is_none() && msk.is_none(), "{what}: {short} upstream did not pass"),
                    Some(rec) => {
                        assert_eq!(rec.shape, vec![g, n, width], "{what}: {short} shape");
                        tfm::assert_close(&format!("{what} {short}"), vals.as_ref().unwrap(), &rec.data, TOL);
                        let rm = mask_of(&tfm::load(&dir, case, &format!("call_{k}.{short}_mask")));
                        let want: Vec<u8> = (0..g * n * width)
                            .map(|i| {
                                let (b, t) = (i / (n * width), i % width);
                                rm[i] | if t < c { pad[b * c + t] } else { 0 }
                            })
                            .collect();
                        assert_eq!(msk.as_deref(), Some(want.as_slice()), "{what}: {short} mask");
                    }
                }
            }
        }
    }
}

#[test]
fn finish_turns_upstreams_decode_outputs_into_its_forecast() {
    let cfg = config();
    let dir = tfm::tiny_dir();
    for case in tfm::FORECAST_TINY {
        let fc = tfm::forecast_case(&dir, case);
        let pl = plan(&cfg.host, &fc.series, fc.horizon, &fc.opts).unwrap();
        let outs: Vec<Forecast> = (0..pl.calls.len())
            .map(|k| {
                let o = tfm::load(&dir, case, &format!("call_{k}.out"));
                Forecast {
                    batch: o.shape[0],
                    variates: o.shape[1],
                    horizon: o.shape[2],
                    quantiles: o.shape[3],
                    values: o.data,
                }
            })
            .collect();
        let got = finish(&pl, &outs).unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(got.len(), fc.series.len(), "{case}: series");
        for (i, f) in got.iter().enumerate() {
            let want = tfm::load(&dir, case, &format!("out_{i}"));
            assert_eq!(
                f.variates * f.horizon * f.quantiles,
                want.data.len(),
                "{case} series {i}: shape {:?}",
                want.shape
            );
            assert_eq!(
                (f.horizon, f.quantiles),
                (fc.horizon, cfg.host.num_quantiles),
                "{case} series {i}"
            );
            tfm::assert_close(&format!("{case} series {i} quantiles"), &f.values, &want.data, TOL);
            tfm::assert_close(
                &format!("{case} series {i} point"),
                &f.point(),
                &tfm::load(&dir, case, &format!("point_{i}")).data,
                TOL,
            );
        }
    }
}

fn one(len: usize) -> Series {
    Series {
        target: Rows::one((0..len).map(|t| t as f32).collect()),
        ..Series::default()
    }
}

#[test]
fn plan_refuses_malformed_requests_by_name() {
    let cfg = config();
    let ok = PredictOptions::default();
    let bad_cov = |po: Rows, pf: Rows| Series {
        past_only: Some(po),
        past_future: Some(pf),
        ..one(8)
    };
    let cases: Vec<(Vec<Series>, usize, PredictOptions, &str)> = vec![
        (vec![one(8)], 0, ok.clone(), "horizon must be a positive integer"),
        (vec![one(8)], usize::MAX, ok.clone(), "overflows"),
        (
            vec![one(8)],
            4,
            PredictOptions {
                per_core_batch_size: 0,
                ..ok.clone()
            },
            "per_core_batch_size",
        ),
        (vec![one(0)], 4, ok.clone(), "at least one time step"),
        (
            vec![Series {
                target: Rows {
                    rows: 0,
                    values: vec![],
                },
                ..Series::default()
            }],
            4,
            ok.clone(),
            "at least one time step",
        ),
        (
            vec![
                one(8),
                Series {
                    target: Rows {
                        rows: 2,
                        values: vec![0.0; 16],
                    },
                    ..Series::default()
                },
            ],
            4,
            ok.clone(),
            "same number of target variates",
        ),
        (
            vec![bad_cov(Rows::one(vec![0.0; 7]), Rows::one(vec![0.0; 12]))],
            4,
            ok.clone(),
            "past_only holds 7",
        ),
        (
            vec![bad_cov(Rows::one(vec![0.0; 8]), Rows::one(vec![0.0; 11]))],
            4,
            ok.clone(),
            "past_future holds 11",
        ),
        (
            vec![Series {
                target: Rows::one(vec![1.0, f32::INFINITY, 2.0]),
                ..Series::default()
            }],
            4,
            ok.clone(),
            "is infinite",
        ),
        (
            vec![
                Series {
                    past_only: Some(Rows::one(vec![0.0; 8])),
                    ..one(8)
                },
                Series {
                    past_only: Some(Rows {
                        rows: 2,
                        values: vec![0.0; 16],
                    }),
                    ..one(8)
                },
            ],
            4,
            ok.clone(),
            "same count",
        ),
    ];
    for (series, horizon, opts, want) in cases {
        let e = plan(&cfg.host, &series, horizon, &opts)
            .err()
            .unwrap_or_else(|| panic!("{want}: accepted"));
        assert!(e.contains(want), "expected {want:?} in {e:?}");
    }
    // Upstream returns nothing for no series; so does plan().
    assert!(plan(&cfg.host, &[], 4, &ok).unwrap().calls.is_empty());
}

#[test]
fn slice_batch_is_the_batch_in_pieces() {
    let full = DecodeInput {
        batch: 3,
        context: 4,
        horizon: 2,
        num_target: 2,
        target: (0..24).map(|x| x as f32).collect(),
        target_mask: Some((0..24).map(|x| (x % 3 == 0) as u8).collect()),
        num_past_future: 1,
        past_future: Some((0..18).map(|x| -(x as f32)).collect()),
        past_future_mask: Some(vec![0; 18]),
        ..DecodeInput::default()
    };
    let a = full.slice_batch(0, 2).unwrap();
    let b = full.slice_batch(2, 3).unwrap();
    assert_eq!((a.batch, b.batch), (2, 1));
    assert_eq!([a.target.clone(), b.target.clone()].concat(), full.target);
    assert_eq!(
        [a.target_mask.clone().unwrap(), b.target_mask.clone().unwrap()].concat(),
        full.target_mask.clone().unwrap()
    );
    assert_eq!(
        [a.past_future.clone().unwrap(), b.past_future.clone().unwrap()].concat(),
        full.past_future.clone().unwrap()
    );
    assert!(a.past_only.is_none() && a.past_only_mask.is_none());
    for (s, e) in [(0, 0), (2, 1), (0, 4)] {
        assert!(full.slice_batch(s, e).is_err(), "{s}..{e} accepted");
    }
    let mut bad = full.clone();
    bad.target.pop();
    assert!(bad.slice_batch(0, 1).unwrap_err().contains("not a multiple of batch"));
}
