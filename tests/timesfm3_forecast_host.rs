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
use tessl::timesfm3_forecast::{finish, pack, plan, PredictOptions, Rows, Series};
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

// ---- packing -----------------------------------------------------------------

/// A `decode()` input of `batch` elements whose every value says which call
/// and element it is (`call * 1000 + element` plus its place in the array), so
/// a slice that lands on the wrong element changes the result.
fn input(call: usize, batch: usize, context: usize, horizon: usize, cov: (usize, usize), masks: bool) -> DecodeInput {
    let fill = |n: usize, salt: usize| -> Vec<f32> {
        (0..batch * n)
            .map(|i| (call * 1000 + i / n.max(1) + salt) as f32 + (i % n.max(1)) as f32 * 0.001)
            .collect()
    };
    let mask = |n: usize| masks.then(|| (0..batch * n).map(|i| ((i / n.max(1) + call) % 2) as u8).collect());
    let (po, pf) = cov;
    DecodeInput {
        batch,
        context,
        horizon,
        num_target: 2,
        target: fill(2 * context, 0),
        target_mask: mask(2 * context),
        num_past_only: po,
        past_only: (po > 0).then(|| fill(po * context, 300)),
        past_only_mask: if po > 0 { mask(po * context) } else { None },
        num_past_future: pf,
        past_future: (pf > 0).then(|| fill(pf * (context + horizon), 600)),
        past_future_mask: if pf > 0 { mask(pf * (context + horizon)) } else { None },
    }
}

/// A stand-in for `decode()` that, like it, treats every batch element on its
/// own: each output value is a function of that element's inputs alone.
fn fake_decode(i: &DecodeInput) -> Forecast {
    const Q: usize = 3;
    let v = i.variates();
    let per_target = i.num_target * i.context;
    let mut values = Vec::new();
    for e in 0..i.batch {
        let mut seed = i.target[e * per_target..(e + 1) * per_target].iter().sum::<f32>();
        if let Some(p) = &i.past_only {
            let n = i.num_past_only * i.context;
            seed += p[e * n..(e + 1) * n].iter().sum::<f32>();
        }
        if let Some(p) = &i.past_future {
            let n = i.num_past_future * (i.context + i.horizon);
            seed += p[e * n..(e + 1) * n].iter().sum::<f32>();
        }
        if let Some(m) = &i.target_mask {
            seed += m[e * per_target..(e + 1) * per_target]
                .iter()
                .map(|&x| f32::from(x))
                .sum::<f32>()
                * 0.5;
        }
        for vi in 0..v {
            for h in 0..i.horizon {
                for q in 0..Q {
                    values.push(seed + (vi * 100 + h * 10 + q) as f32);
                }
            }
        }
    }
    Forecast {
        batch: i.batch,
        variates: v,
        horizon: i.horizon,
        quantiles: Q,
        values,
    }
}

/// Packed and run through `fake_decode`, then unpacked: must be, bit for bit,
/// one `fake_decode` per call.
fn packed_equals_unpacked(what: &str, calls: &[DecodeInput], max_rows: usize) -> usize {
    let cfg = config();
    let packing = pack(&cfg.host, calls, max_rows).unwrap_or_else(|e| panic!("{what}: {e}"));
    let outs: Vec<Forecast> = packing.forwards.iter().map(fake_decode).collect();
    let got = packing.unpack(&outs).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(got.len(), calls.len(), "{what}: forecasts");
    for (k, (g, call)) in got.iter().zip(calls).enumerate() {
        let want = fake_decode(call);
        assert_eq!(
            (g.batch, g.variates, g.horizon, g.quantiles),
            (want.batch, want.variates, want.horizon, want.quantiles),
            "{what} call {k}"
        );
        tfm::assert_bit_equal(&format!("{what} call {k}"), &g.values, &want.values);
    }
    for (f, (fwd, pieces)) in packing.forwards.iter().zip(&packing.pieces).enumerate() {
        let rows: usize = pieces
            .iter()
            .map(|p| cfg.host.rows_per_series(fwd.variates(), fwd.context, fwd.horizon) * (p.end - p.start))
            .sum();
        assert!(
            rows <= max_rows.min(tessl::timesfm3::MAX_ROWS),
            "{what}: forward {f} has {rows} rows"
        );
        assert!(
            pieces.iter().all(|p| calls[p.call].same_shape(fwd)),
            "{what}: forward {f} mixes shapes"
        );
        assert_eq!(
            fwd.batch,
            pieces.iter().map(|p| p.end - p.start).sum::<usize>(),
            "{what}: forward {f} batch"
        );
    }
    packing.forwards.len()
}

#[test]
fn equal_shapes_share_one_forward_and_unpack_to_their_calls() {
    // Four calls of 4, 4, 4 and 2 elements: upstream's groups of one request.
    let calls: Vec<DecodeInput> = [4, 4, 4, 2]
        .iter()
        .enumerate()
        .map(|(k, &b)| input(k, b, 64, 64, (0, 0), true))
        .collect();
    assert_eq!(packed_equals_unpacked("equal shapes", &calls, 8192), 1);
    let pk = pack(&config().host, &calls, 8192).unwrap();
    assert_eq!(pk.forwards[0].batch, 14);
}

#[test]
fn shapes_that_differ_in_any_field_never_share_a_forward() {
    let base = |k| input(k, 3, 64, 64, (0, 0), true);
    let variants = [
        ("context", input(1, 3, 96, 64, (0, 0), true)),
        ("horizon", input(2, 3, 64, 128, (0, 0), true)),
        ("past-only channels", input(3, 3, 64, 64, (1, 0), true)),
        ("past+future channels", input(4, 3, 64, 64, (0, 2), true)),
        ("covariate layout", input(5, 3, 64, 64, (2, 1), true)),
        ("masks present", input(6, 3, 64, 64, (0, 0), false)),
    ];
    for (field, other) in variants {
        let calls = [base(0), other, base(7)];
        let n = packed_equals_unpacked(field, &calls, 8192);
        assert_eq!(
            n, 2,
            "{field}: the two equal calls share one forward, the odd one runs apart"
        );
        let pk = pack(&config().host, &calls, 8192).unwrap();
        assert_eq!(
            pk.pieces[0].iter().map(|p| p.call).collect::<Vec<_>>(),
            [0, 2],
            "{field}: non-adjacent equal calls merge"
        );
        assert_eq!(pk.pieces[1].iter().map(|p| p.call).collect::<Vec<_>>(), [1], "{field}");
    }
}

#[test]
fn packing_respects_the_row_cap_and_cuts_whole_elements() {
    let cfg = config();
    let one = cfg.host.rows_per_series(2, 64, 64);
    // Calls of 3, 2 and 4 elements at a cap of 4 elements: forwards of 4, 4, 1,
    // with the middle call cut across the first two.
    let calls = [
        input(0, 3, 64, 64, (0, 0), true),
        input(1, 2, 64, 64, (0, 0), true),
        input(2, 4, 64, 64, (0, 0), true),
    ];
    assert_eq!(packed_equals_unpacked("capped", &calls, 4 * one), 3);
    let pk = pack(&cfg.host, &calls, 4 * one).unwrap();
    assert_eq!(pk.forwards.iter().map(|f| f.batch).collect::<Vec<_>>(), [4, 4, 1]);
    // A cap below one element is refused by name, and one element exactly fits.
    let empty = [input(0, 2, 64, 64, (0, 0), true), input(1, 0, 64, 64, (0, 0), true)];
    assert!(pack(&cfg.host, &empty, 8192)
        .unwrap_err()
        .contains("call 1 has an empty batch"));
    let e = pack(&cfg.host, &calls, one - 1).unwrap_err();
    assert!(e.contains("one series needs") && e.contains(&one.to_string()), "{e}");
    assert_eq!(packed_equals_unpacked("one element", &calls, one), 9);
}

/// The forward counts the bench used to compute by hand (`forwards_for`).
#[test]
fn forwards_follow_the_row_cap() {
    let cfg = config();
    let p = cfg.host.input_patch_len;
    // A univariate-target call of `batch` elements of `per` patch rows each.
    let call = |batch: usize, per: usize| {
        let horizon = cfg.host.output_patch_len;
        let hp = cfg.host.rows_per_series(1, p, horizon) - 1;
        let mut c = input(0, batch, p * (per - hp), horizon, (0, 0), false);
        c.num_target = 1;
        c.target = vec![0.5; batch * c.context];
        assert_eq!(
            cfg.host.rows_per_series(c.variates(), c.context, c.horizon),
            per,
            "test bug"
        );
        c
    };
    let forwards = |batch, per| pack(&cfg.host, &[call(batch, per)], 8192).map(|p| p.forwards.len());
    assert_eq!(forwards(4, 100).unwrap(), 1);
    assert_eq!(forwards(4, 4096).unwrap(), 2);
    assert_eq!(forwards(5, 3000).unwrap(), 3);
    assert_eq!(forwards(1, 8192).unwrap(), 1);
    assert!(forwards(1, 8193).unwrap_err().contains("8193"));
}

#[test]
fn unpack_refuses_forecasts_that_do_not_fit_the_packing() {
    let cfg = config();
    let calls = [input(0, 2, 64, 64, (0, 0), true), input(1, 3, 64, 64, (0, 0), true)];
    let pk = pack(&cfg.host, &calls, 8192).unwrap();
    let ok = fake_decode(&pk.forwards[0]);
    assert!(pk.unpack(&[]).unwrap_err().contains("0 forecasts for 1 forwards"));
    let mut short = fake_decode(&pk.forwards[0]);
    short.batch -= 1;
    short
        .values
        .truncate(short.batch * short.variates * short.horizon * short.quantiles);
    assert!(pk.unpack(&[short]).unwrap_err().contains("expected a batch of 5"));
    let mut cut = ok;
    cut.values.pop();
    assert!(pk.unpack(&[cut]).is_err());
}

#[test]
fn concat_and_slice_batches_keep_every_array_aligned() {
    let a = input(0, 5, 64, 64, (1, 2), true);
    let b = input(1, 3, 64, 64, (1, 2), true);
    let c = DecodeInput::concat_batches(&[(&a, 1, 4), (&b, 0, 2), (&a, 4, 5)]).unwrap();
    assert_eq!(c.batch, 6);
    for (k, (src, at)) in [(&a, 1), (&a, 2), (&a, 3), (&b, 0), (&b, 1), (&a, 4)]
        .into_iter()
        .enumerate()
    {
        let one = c.slice_batch(k, k + 1).unwrap();
        let want = src.slice_batch(at, at + 1).unwrap();
        assert_eq!(one, want, "element {k}");
    }
    let other = input(2, 3, 96, 64, (1, 2), true);
    assert!(DecodeInput::concat_batches(&[(&a, 0, 1), (&other, 0, 1)])
        .unwrap_err()
        .contains("different shapes"));
    assert!(DecodeInput::concat_batches(&[(&a, 0, 6)])
        .unwrap_err()
        .contains("0..6 of 5"));
    assert!(DecodeInput::concat_batches(&[]).is_err());
    let mut ragged = a.clone();
    ragged.target.pop();
    assert!(DecodeInput::concat_batches(&[(&ragged, 0, 1)])
        .unwrap_err()
        .contains("target holds"));
    assert!(a.slice_batch(2, 2).is_err() && a.slice_batch(0, 6).is_err());
}
