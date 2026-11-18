//! `Timesfm3Model` on the tiny seeded checkpoint against upstream's own
//! PyTorch forward (`tests/fixtures/timesfm3_tiny`, written by
//! `tools/timesfm3_ref/make_reference.py`): the residual-block output, every
//! layer's residual stream, the head's raw logits, and the whole decode().
//!
//! The cases cover one variate (upstream's single-variate shortcut), several
//! variates with covariates, a left-padded context, a fully-masked leading
//! patch (sequence keys masked, an all-masked query row, variate keys
//! masked), detrending on and off, a constant series and a horizon past one
//! output patch (stitching).

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

#[test]
fn tiny_forward_matches_upstream_layer_by_layer() {
    with_gpu(|rt| {
        let m = model(rt);
        let dir = tfm::tiny_dir();
        for case in tfm::TINY_CASES {
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
            tfm::assert_close(&format!("{case} transformer_input"), &out.trace[0], &tfm::load(&dir, case, "transformer_input").data, LAYER_TOL);
            for l in 0..m.config().layers as usize {
                let want = tfm::load(&dir, case, &format!("layer_{l}"));
                tfm::assert_close(&format!("{case} layer {l}"), &out.trace[l + 1], &want.data, LAYER_TOL);
            }
            tfm::assert_close(&format!("{case} raw logits"), &out.raw_logits, &tfm::load(&dir, case, "raw_logits").data, LAYER_TOL);
        }
    });
}

#[test]
fn tiny_decode_matches_upstream_forecast() {
    with_gpu(|rt| {
        let m = model(rt);
        let dir = tfm::tiny_dir();
        for case in tfm::TINY_CASES {
            let f = m.decode(&tfm::decode_input(&dir, case)).unwrap_or_else(|e| panic!("{case}: {e}"));
            let want = tfm::load(&dir, case, "out");
            assert_eq!([f.batch, f.variates, f.horizon, f.quantiles].as_slice(), want.shape.as_slice(), "{case}");
            tfm::assert_close(&format!("{case} forecast vs fp32"), &f.values, &want.data, OUT_TOL);
            tfm::assert_no_worse_than_upstream(&format!("{case} forecast"), &f.values, &want.data, &tfm::load64(&dir, case, "out"));
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
            (1, 1, 2, { let mut x = ok.clone(); x[5] = f32::NAN; x }, vec![0], "non-finite"),
            (0, 1, 2, Vec::new(), Vec::new(), "empty"),
            (1, 33, 1, vec![0.0f32; 33 * w], vec![0; 33], "variates exceed"),
        ] {
            let e = m.forward(b, v, n, &rin, &lead, false).err().unwrap_or_else(|| panic!("{want}: accepted"));
            assert!(e.contains(want), "expected {want:?} in {e:?}");
        }
    });
}
