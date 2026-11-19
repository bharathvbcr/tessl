//! `Timesfm3Model` on the tiny seeded checkpoint against upstream's own
//! PyTorch forward (`tests/fixtures/timesfm3_tiny`, written by
//! `tools/timesfm3_ref/make_reference.py`): the residual-block output, every
//! layer's residual stream, the head's raw logits, and the whole decode().
//!
//! The cases cover one variate (upstream's single-variate shortcut), several
//! variates with covariates, a left-padded context, a fully-masked leading
//! patch (sequence keys masked, an all-masked query row, variate keys
//! masked), detrending on and off, a constant series and a horizon past one
//! output patch (stitching). The off-branch cases hold each `config.json`
//! switch the port reads (stitching, linear detrending, CPM refinement,
//! variate attention, the memory-efficient-attention scale) off, one at a time
//! and all together, to upstream built with that switch off.

mod common;
#[path = "common/timesfm3.rs"]
mod tfm;

use common::with_gpu;
use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_host::{prepare, preprocess};

/// Exact-f32 GEMMs against torch's CPU fp32: differences are operation order
/// only. Layers compound them, the forecast de-normalises them.
const LAYER_TOL: f64 = 2e-5;
const OUT_TOL: f64 = 5e-5;

fn model(rt: &std::sync::Arc<tessl::runtime::GpuRuntime>) -> Timesfm3Model {
    let dir = tfm::tiny_dir();
    let cfg = Timesfm3Config::from_config_file(&dir.join("config.json")).unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    Timesfm3Model::load(rt, &st, cfg).unwrap()
}

/// Forward pass of `case` on `m`, driven from upstream's own residual-block
/// input, held to every layer and the raw logits.
fn check_forward(m: &Timesfm3Model, case: &str) {
    let dir = tfm::tiny_dir();
    let inp = tfm::decode_input(&dir, case);
    let pr = prepare(&m.config().host, &inp).unwrap();
    let pre = preprocess(&m.config().host, &pr).unwrap();
    // Drive the device from upstream's own residual-block input, so a
    // host difference cannot hide or cause a device one.
    let rin = tfm::load(&dir, case, "resblock_input");
    let out = m
        .forward(pr.batch, pr.variates, pr.patches, &rin.data, &pre.leading_masked, true)
        .unwrap_or_else(|e| panic!("{case}: {e}"));
    assert_eq!(out.trace.len(), 1 + m.config().layers as usize);
    tfm::assert_close(
        &format!("{case} transformer_input"),
        &out.trace[0],
        &tfm::load(&dir, case, "transformer_input").data,
        LAYER_TOL,
    );
    for l in 0..m.config().layers as usize {
        let want = tfm::load(&dir, case, &format!("layer_{l}"));
        tfm::assert_close(&format!("{case} layer {l}"), &out.trace[l + 1], &want.data, LAYER_TOL);
    }
    tfm::assert_close(
        &format!("{case} raw logits"),
        &out.raw_logits,
        &tfm::load(&dir, case, "raw_logits").data,
        LAYER_TOL,
    );
}

/// The whole decode() of `case` on `m`, held to upstream's fp32 forecast and
/// to no further from its f64 run than its fp32 run is.
fn check_decode(m: &Timesfm3Model, case: &str) {
    let dir = tfm::tiny_dir();
    let f = m
        .decode(&tfm::decode_input(&dir, case))
        .unwrap_or_else(|e| panic!("{case}: {e}"));
    let want = tfm::load(&dir, case, "out");
    assert_eq!(
        [f.batch, f.variates, f.horizon, f.quantiles].as_slice(),
        want.shape.as_slice(),
        "{case}"
    );
    tfm::assert_close(&format!("{case} forecast vs fp32"), &f.values, &want.data, OUT_TOL);
    tfm::assert_no_worse_than_upstream(
        &format!("{case} forecast"),
        &f.values,
        &want.data,
        &tfm::load64(&dir, case, "out"),
    );
}

#[test]
fn tiny_forward_matches_upstream_layer_by_layer() {
    with_gpu(|rt| {
        let m = model(rt);
        for case in tfm::TINY_CASES {
            check_forward(&m, case);
        }
    });
}

#[test]
fn tiny_decode_matches_upstream_forecast() {
    with_gpu(|rt| {
        let m = model(rt);
        for case in tfm::TINY_CASES {
            check_decode(&m, case);
        }
    });
}

/// The model of an off-branch case: its own config (one or more switches
/// off) and weights.
fn off_model(rt: &std::sync::Arc<tessl::runtime::GpuRuntime>, case: &str) -> Timesfm3Model {
    let dir = tfm::tiny_dir();
    let cfg = tfm::off_config(&dir, case);
    let st = SafeTensors::open(&tfm::off_weights_path(&dir, case)).unwrap();
    Timesfm3Model::load(rt, &st, cfg).unwrap_or_else(|e| panic!("{case}: {e}"))
}

/// Every `config.json` switch the port reads, off one at a time and all
/// together, against upstream's own forward on a model built with it off.
#[test]
fn off_branch_forward_matches_upstream_layer_by_layer() {
    with_gpu(|rt| {
        for case in tfm::OFF_CASES {
            check_forward(&off_model(rt, case), case);
        }
    });
}

#[test]
fn off_branch_decode_matches_upstream_forecast() {
    with_gpu(|rt| {
        for case in tfm::OFF_CASES {
            check_decode(&off_model(rt, case), case);
        }
    });
}

#[test]
fn forward_refuses_malformed_calls() {
    with_gpu(|rt| {
        let m = model(rt);
        let w = m.config().host.resblock_in();
        let ok = vec![0.0f32; 2 * w];
        assert!(m.forward(1, 1, 2, &ok, &[0], false).is_ok());
        for (b, v, n, rin, lead, want) in [
            (1, 1, 2, vec![0.0f32; w], vec![0u32], "resblock_input holds"),
            (1, 1, 2, ok.clone(), vec![0, 0], "leading_masked holds"),
            (1, 1, 2, ok.clone(), vec![3], "exceeds 2 patches"),
            (
                1,
                1,
                2,
                {
                    let mut x = ok.clone();
                    x[5] = f32::NAN;
                    x
                },
                vec![0],
                "non-finite",
            ),
            (0, 1, 2, Vec::new(), Vec::new(), "empty"),
            (1, 33, 1, vec![0.0f32; 33 * w], vec![0; 33], "variates exceed"),
        ] {
            let e = m
                .forward(b, v, n, &rin, &lead, false)
                .err()
                .unwrap_or_else(|| panic!("{want}: accepted"));
            assert!(e.contains(want), "expected {want:?} in {e:?}");
        }
    });
}

#[test]
fn tiny_predict_batch_matches_upstream_forecaster() {
    with_gpu(|rt| {
        let m = model(rt);
        let dir = tfm::tiny_dir();
        for case in tfm::FORECAST_TINY {
            let fc = tfm::forecast_case(&dir, case);
            let got = m
                .predict_batch(&fc.series, fc.horizon, &fc.opts)
                .unwrap_or_else(|e| panic!("{case}: {e}"));
            assert_eq!(got.len(), fc.series.len(), "{case}: series");
            for (i, f) in got.iter().enumerate() {
                let want = tfm::load(&dir, case, &format!("out_{i}"));
                assert_eq!(f.values.len(), want.data.len(), "{case} series {i}: {:?}", want.shape);
                tfm::assert_close(&format!("{case} series {i} forecast"), &f.values, &want.data, OUT_TOL);
                let point = tfm::load(&dir, case, &format!("point_{i}"));
                tfm::assert_close(&format!("{case} series {i} point"), &f.point(), &point.data, OUT_TOL);
            }
        }
    });
}

/// A group run as several forwards of whole series gives what one forward
/// gives (series in a batch never interact), and a cap below one series is
/// refused by name.
#[test]
fn predict_batch_in_pieces_matches_one_forward() {
    with_gpu(|rt| {
        let m = model(rt);
        let dir = tfm::tiny_dir();
        for case in ["fa_ragged", "fc_multivariate_cov", "fe_long"] {
            let fc = tfm::forecast_case(&dir, case);
            let whole = m.predict_batch(&fc.series, fc.horizon, &fc.opts).unwrap();
            let pl = tessl::timesfm3_forecast::plan(&m.config().host, &fc.series, fc.horizon, &fc.opts).unwrap();
            assert!(pl.calls.iter().any(|c| c.batch > 1), "{case}: no group to split");
            let one = pl
                .calls
                .iter()
                .map(|c| m.config().host.rows_per_series(c.variates(), c.context, c.horizon))
                .max()
                .unwrap();
            let pieces = m.predict_batch_capped(&fc.series, fc.horizon, &fc.opts, one).unwrap();
            for (i, (a, b)) in whole.iter().zip(&pieces).enumerate() {
                tfm::assert_close(&format!("{case} series {i} in pieces"), &b.values, &a.values, 1e-6);
            }
            let e = m
                .predict_batch_capped(&fc.series, fc.horizon, &fc.opts, one - 1)
                .err()
                .unwrap_or_else(|| panic!("{case}: a cap below one series was accepted"));
            assert!(e.contains("one series needs"), "{case}: {e}");
        }
    });
}
