//! `Timesfm3Model` on the real `google/timesfm-3.0-pytorch` checkpoint
//! against upstream's PyTorch decode(). Both live outside the repository (the
//! weights are under the TimesFM Non-Commercial License v1.0, so neither they
//! nor anything computed from them is committed), so the test is ignored by
//! default and fails, never passes, when asked to run without them:
//!
//!   ~/.venvs/ml/bin/python tools/timesfm3_ref/make_reference.py \
//!       --model-dir $TIMESFM3_DIR --out target/timesfm3_ref_real
//!   TIMESFM3_DIR=<snapshot dir> TIMESFM3_REF_DIR=target/timesfm3_ref_real \
//!       cargo test --release --test timesfm3_real -- --ignored --test-threads=1
//!
//! `real_predict_batch_matches_upstream_forecaster` holds the forecaster to
//! upstream's `TimesFM3Forecaster.predict_batch()`, including contexts at and
//! past the 15360 global context and one group of about 15.4k patch rows,
//! which runs as several forwards (`MAX_ROWS`).

mod common;
#[path = "common/timesfm3.rs"]
mod tfm;

use std::path::PathBuf;

use common::with_gpu;
use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_forecast::{self as forecast, PredictOptions};
use tessl::timesfm3_host::{prepare, preprocess};

fn env_dir(k: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var_os(k).unwrap_or_else(|| panic!("set {k} (see the module docs)")));
    assert!(p.is_dir(), "{k}={} is not a directory", p.display());
    p
}

#[test]
#[ignore]
fn real_checkpoint_matches_upstream() {
    let (wdir, rdir) = (env_dir("TIMESFM3_DIR"), env_dir("TIMESFM3_REF_DIR"));
    with_gpu(|rt| {
        let cfg = Timesfm3Config::from_config_file(&wdir.join("config.json")).unwrap();
        let st = SafeTensors::open(&wdir.join("model.safetensors")).unwrap();
        let m = Timesfm3Model::load(rt, &st, cfg).unwrap();
        for case in tfm::REAL_CASES {
            let inp = tfm::decode_input(&rdir, case);
            let pr = prepare(&m.config().host, &inp).unwrap();
            let pre = preprocess(&m.config().host, &pr).unwrap();
            let rin = tfm::load(&rdir, case, "resblock_input");
            let out = m
                .forward(pr.batch, pr.variates, pr.patches, &rin.data, &pre.leading_masked, true)
                .unwrap();
            for l in [0usize, 10, 19] {
                let want = tfm::load(&rdir, case, &format!("layer_{l}"));
                tfm::assert_close(&format!("{case} layer {l}"), &out.trace[l + 1], &want.data, 1e-3);
            }
            tfm::assert_close(
                &format!("{case} raw logits"),
                &out.raw_logits,
                &tfm::load(&rdir, case, "raw_logits").data,
                1e-3,
            );
            let f = m.decode(&inp).unwrap();
            tfm::assert_close(
                &format!("{case} forecast"),
                &f.values,
                &tfm::load(&rdir, case, "out").data,
                1e-3,
            );
        }
    });
}

#[test]
#[ignore]
fn real_predict_batch_matches_upstream_forecaster() {
    let (wdir, rdir) = (env_dir("TIMESFM3_DIR"), env_dir("TIMESFM3_REF_DIR"));
    with_gpu(|rt| {
        let cfg = Timesfm3Config::from_config_file(&wdir.join("config.json")).unwrap();
        let st = SafeTensors::open(&wdir.join("model.safetensors")).unwrap();
        let m = Timesfm3Model::load(rt, &st, cfg).unwrap();
        for case in tfm::FORECAST_REAL {
            let fc = tfm::forecast_case(&rdir, case);
            let got = m
                .predict_batch(&fc.series, fc.horizon, &fc.opts)
                .unwrap_or_else(|e| panic!("{case}: {e}"));
            assert_eq!(got.len(), fc.series.len(), "{case}: series");
            for (i, f) in got.iter().enumerate() {
                let want = tfm::load(&rdir, case, &format!("out_{i}"));
                tfm::assert_close(&format!("{case} series {i} forecast"), &f.values, &want.data, 1e-3);
            }
        }
    });
}

fn load_real(rt: &std::sync::Arc<tessl::runtime::GpuRuntime>, wdir: &std::path::Path) -> Timesfm3Model {
    let cfg = Timesfm3Config::from_config_file(&wdir.join("config.json")).unwrap();
    let st = SafeTensors::open(&wdir.join("model.safetensors")).unwrap();
    Timesfm3Model::load(rt, &st, cfg).unwrap()
}

/// The premise of group packing at the real widths: a batch element's output
/// bits do not depend on how many elements share its forward (the GEMM M, from
/// one row group up to a few thousand rows). One call decoded whole equals
/// the same call decoded in slices of several sizes, bit for bit.
#[test]
#[ignore]
fn real_decode_bits_do_not_depend_on_batch_size() {
    let wdir = env_dir("TIMESFM3_DIR");
    with_gpu(|rt| {
        let m = load_real(rt, &wdir);
        // (series, variates, context, horizon): the bench's three predict shapes.
        for (n, v, ctx, horizon) in [
            (64usize, 1usize, 512usize, 128usize),
            (8, 1, 4096, 256),
            (16, 4, 1024, 96),
        ] {
            let series = tfm::synthetic_series(n, v, |_| ctx, 7);
            let opts = PredictOptions {
                per_core_batch_size: n,
                ..PredictOptions::default()
            };
            let pl = forecast::plan(&m.config().host, &series, horizon, &opts).unwrap();
            assert_eq!(pl.calls.len(), 1, "test bug: one call of {n} series");
            let call = &pl.calls[0];
            let whole = m.decode(call).unwrap();
            for size in [1usize, 3, 4, 5, 7, 13, 16] {
                let mut values = Vec::new();
                for start in (0..call.batch).step_by(size) {
                    let piece = call.slice_batch(start, (start + size).min(call.batch)).unwrap();
                    values.extend(m.decode(&piece).unwrap().values);
                }
                tfm::assert_bit_equal(
                    &format!("{n}x{v}x{ctx}:{horizon} in slices of {size}"),
                    &values,
                    &whole.values,
                );
            }
        }
    });
}

/// Group packing on the real checkpoint: `predict_batch` (calls of equal
/// shape run as shared forwards) against one `decode()` per plan call, bit
/// for bit, on the recorded forecaster cases (ragged, covariates,
/// multivariate, over the row cap) and on ragged synthetic workloads shaped
/// like the bench's. Both sides are live runs of this checkpoint; nothing
/// computed from it is stored.
#[test]
#[ignore]
fn real_packed_predict_batch_is_bit_equal_to_one_decode_per_call() {
    let (wdir, rdir) = (env_dir("TIMESFM3_DIR"), env_dir("TIMESFM3_REF_DIR"));
    with_gpu(|rt| {
        let m = load_real(rt, &wdir);
        let host = &m.config().host;
        let ragged = |longest: usize| move |i: usize| longest - (i * 61) % (longest / 4 + 1);
        let mut cases: Vec<(String, Vec<_>, usize, PredictOptions)> = Vec::new();
        for case in tfm::FORECAST_REAL {
            let fc = tfm::forecast_case(&rdir, case);
            cases.push((case.to_string(), fc.series, fc.horizon, fc.opts));
        }
        cases.push((
            "ragged 64x512:128".into(),
            tfm::synthetic_series(64, 1, ragged(512), 11),
            128,
            PredictOptions::default(),
        ));
        cases.push((
            "ragged 16x4x1024:96+1+1".into(),
            tfm::with_covariates(tfm::synthetic_series(16, 4, ragged(1024), 12), 96, 1, 1, |_| true),
            96,
            PredictOptions::default(),
        ));
        cases.push((
            "ragged 9x700:200 symmetric, covariates on some".into(),
            tfm::with_covariates(tfm::synthetic_series(9, 1, ragged(700), 13), 200, 0, 2, |i| i % 3 != 1),
            200,
            PredictOptions {
                use_symmetric_averaging: true,
                use_znorm: true,
                per_core_batch_size: 3,
                ..PredictOptions::default()
            },
        ));
        for (name, series, horizon, opts) in cases {
            let pl = forecast::plan(host, &series, horizon, &opts).unwrap();
            let forwards = forecast::pack(host, &pl.calls, tessl::timesfm3::MAX_ROWS)
                .unwrap()
                .forwards
                .len();
            let got = m
                .predict_batch(&series, horizon, &opts)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let want = tfm::predict_one_decode_per_call(&m, &series, horizon, &opts, tessl::timesfm3::MAX_ROWS);
            tfm::assert_forecasts_bit_equal(&name, &got, &want);
            eprintln!("{name}: {} plan calls in {forwards} forwards", pl.calls.len());
        }
    });
}
