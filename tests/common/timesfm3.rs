//! Fixture reading shared by the TimesFM 3 tests: the directories
//! `tools/timesfm3_ref/make_reference.py` writes (one flat directory, files
//! named `<case>.<name>.npy`, masks as f32 0/1).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tessl::npy;
use tessl::timesfm3_forecast::{CovariatePadding, PredictOptions, Rows, Series};
use tessl::timesfm3_host::DecodeInput;

pub struct Arr {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load(dir: &Path, case: &str, name: &str) -> Arr {
    let p = dir.join(format!("{case}.{name}.npy"));
    let a = npy::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    Arr {
        shape: a.shape.clone(),
        data: a.f32s().unwrap_or_else(|e| panic!("{}: {e}", p.display())).to_vec(),
    }
}

/// A case's float64 reference (the f64 pass of the same model and inputs).
pub fn load64(dir: &Path, case: &str, name: &str) -> Vec<f64> {
    let p = dir.join(format!("{case}.f64.{name}.npy"));
    let a = npy::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    a.f64s().unwrap_or_else(|e| panic!("{}: {e}", p.display())).to_vec()
}

fn max_abs_diff(what: &str, got: &[f32], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f64, |m, &x| m.max(x.abs())).max(1e-6);
    let worst = got.iter().zip(want).enumerate().fold(0.0f64, |m, (i, (&g, &w))| {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        m.max((f64::from(g) - w).abs())
    });
    (worst, peak)
}

/// The port is no further from upstream's f64 run than upstream's own fp32
/// run is: `max|got - f64| <= 2 * max|torch32 - f64| + 1e-6 * peak`. The
/// fixed term covers stages where fp32 happens to be exact.
pub fn assert_no_worse_than_upstream(what: &str, got: &[f32], torch32: &[f32], f64_ref: &[f64]) {
    let (ours, peak) = max_abs_diff(what, got, f64_ref);
    let (theirs, _) = max_abs_diff(what, torch32, f64_ref);
    let bound = 2.0 * theirs + 1e-6 * peak;
    eprintln!("{what}: |ours - f64| = {ours:.3e}, |torch fp32 - f64| = {theirs:.3e} (peak {peak:.3e})");
    assert!(
        ours <= bound,
        "{what}: |ours - f64| = {ours:.3e} > 2 * |torch fp32 - f64| + 1e-6 peak = {bound:.3e}"
    );
}

pub fn maybe(dir: &Path, case: &str, name: &str) -> Option<Arr> {
    dir.join(format!("{case}.{name}.npy"))
        .is_file()
        .then(|| load(dir, case, name))
}

pub fn mask(a: &Arr) -> Vec<u8> {
    a.data
        .iter()
        .map(|&x| {
            assert!(x == 0.0 || x == 1.0, "mask value {x} is not 0 or 1");
            x as u8
        })
        .collect()
}

pub fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/timesfm3_tiny")
}

pub const TINY_CASES: [&str; 3] = ["a_univariate", "b_covariates", "c_constant"];
pub const REAL_CASES: [&str; 5] = [
    "a_univariate",
    "b_covariates",
    "c_constant",
    "d_batch4_ctx1024",
    "e_multivariate8_ctx512",
];

/// The decode() input a case recorded; the horizon is the forecast's length.
pub fn decode_input(dir: &Path, case: &str) -> DecodeInput {
    let t = load(dir, case, "in_target");
    let out = load(dir, case, "out");
    let (b, u, ctx) = (t.shape[0], t.shape[1], t.shape[2]);
    let mut inp = DecodeInput {
        batch: b,
        context: ctx,
        horizon: out.shape[2],
        num_target: u,
        target: t.data,
        target_mask: Some(mask(&load(dir, case, "in_target_mask"))),
        ..DecodeInput::default()
    };
    if let Some(po) = maybe(dir, case, "in_po") {
        inp.num_past_only = po.shape[1];
        inp.past_only = Some(po.data);
        inp.past_only_mask = Some(mask(&load(dir, case, "in_po_mask")));
    }
    if let Some(pf) = maybe(dir, case, "in_pf") {
        inp.num_past_future = pf.shape[1];
        inp.past_future = Some(pf.data);
        inp.past_future_mask = Some(mask(&load(dir, case, "in_pf_mask")));
    }
    inp
}

/// `max |got - want| / max(max |want|, 1e-6)`, asserting equal lengths and
/// finite values.
pub fn rel_peak(what: &str, got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())).max(1e-6);
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        let d = (f64::from(g) - f64::from(w)).abs();
        if d > worst {
            (worst, at) = (d, i);
        }
    }
    if worst > 0.0 {
        eprintln!(
            "{what}: worst at element {at}: got {} want {} (peak {peak})",
            got[at], want[at]
        );
    }
    worst / peak
}

pub fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f64) {
    let r = rel_peak(what, got, want);
    eprintln!("{what}: max |diff| / peak = {r:.3e}");
    assert!(r <= tol, "{what}: max |diff| / peak = {r:.3e} > {tol:.0e}");
}

/// The forecaster cases `make_reference.py` records on the tiny model, and
/// on the real one (which adds a group over `MAX_ROWS`).
pub const FORECAST_TINY: [&str; 5] = ["fa_ragged", "fb_options", "fc_multivariate_cov", "fd_edge", "fe_long"];
pub const FORECAST_REAL: [&str; 6] = [
    "fa_ragged",
    "fb_options",
    "fc_multivariate_cov",
    "fd_edge",
    "fe_long",
    "ff_over_cap",
];

/// One recorded `predict_batch()` call: its series, horizon and options, and
/// the horizon of each `decode()` call it made.
pub struct ForecastCase {
    pub horizon: usize,
    pub series: Vec<Series>,
    pub opts: PredictOptions,
    pub call_horizons: Vec<usize>,
}

fn rows(a: Arr) -> Rows {
    match a.shape.len() {
        1 => Rows {
            rows: 1,
            values: a.data,
        },
        2 => Rows {
            rows: a.shape[0],
            values: a.data,
        },
        _ => panic!("a series is 1-D or 2-D, got {:?}", a.shape),
    }
}

/// Reads `<case>.forecast.json` and the case's inputs. The options are
/// what upstream ran with, read from the record rather than restated here.
pub fn forecast_case(dir: &Path, case: &str) -> ForecastCase {
    let p = dir.join(format!("{case}.forecast.json"));
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let j = ojas_io::json::parse_json(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let u = |k: &str| j.get_u64(k).unwrap_or_else(|e| panic!("{case} {k}: {e}")) as usize;
    let mut opts = PredictOptions {
        per_core_batch_size: u("per_core_batch_size"),
        ..PredictOptions::default()
    };
    let o = j.get_object("options").unwrap();
    o.deny_unknown_keys(&[
        "use_symmetric_averaging",
        "make_positive",
        "sort_quantiles",
        "use_znorm",
        "padding_mode",
    ])
    .unwrap();
    let flag = |k: &str, d: bool| o.get(k).map_or(d, |v| v.as_bool().unwrap());
    opts.use_symmetric_averaging = flag("use_symmetric_averaging", false);
    opts.make_positive = flag("make_positive", false);
    opts.sort_quantiles = flag("sort_quantiles", true);
    opts.use_znorm = flag("use_znorm", false);
    opts.padding = match o.get("padding_mode").map(|v| v.as_str().unwrap()) {
        None | Some("none") => CovariatePadding::None,
        Some("edge") => CovariatePadding::Edge,
        Some(m) => panic!("{case}: padding_mode {m:?}"),
    };
    let series = (0..u("series"))
        .map(|i| Series {
            target: rows(load(dir, case, &format!("in_{i}"))),
            past_only: maybe(dir, case, &format!("in_po_{i}")).map(rows),
            past_future: maybe(dir, case, &format!("in_pf_{i}")).map(rows),
        })
        .collect();
    let call_horizons = j
        .get_array("calls")
        .unwrap()
        .iter()
        .map(|c| c.get_u64("horizon").unwrap() as usize)
        .collect();
    ForecastCase {
        horizon: u("horizon"),
        series,
        opts,
        call_horizons,
    }
}
