//! Fixture reading shared by the TimesFM 3 tests: the directories
//! `tools/timesfm3_ref/make_reference.py` writes (one flat directory, files
//! named `<case>.<name>.npy`, masks as f32 0/1).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tessl::npy;
use tessl::timesfm3::{Forecast, Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_forecast::{finish, plan, CovariatePadding, PredictOptions, Rows, Series, SeriesForecast};
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
/// The off-branch cases: the inputs of an all-on case run on a model with the
/// switches named in the case's `case.json` `off` turned off (the rest on).
pub const OFF_CASES: [&str; 6] = [
    "off_stitching",
    "off_detrending",
    "off_cpm_refine",
    "off_variate_attention",
    "off_mea_scale",
    "off_all",
];

/// The config an off-branch case ran upstream with.
pub fn off_config_path(dir: &Path, case: &str) -> PathBuf {
    dir.join(format!("{case}.config.json"))
}

/// Its weights: its own file when its state dict differs (no variate
/// attention), else the base model's.
pub fn off_weights_path(dir: &Path, case: &str) -> PathBuf {
    let own = dir.join(format!("{case}.model.safetensors"));
    if own.is_file() {
        own
    } else {
        dir.join("model.safetensors")
    }
}

/// The `config.json` keys the case turned off, from its `case.json`.
pub fn off_switches(dir: &Path, case: &str) -> Vec<String> {
    let p = dir.join(format!("{case}.case.json"));
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let j = ojas_io::json::parse_json(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    j.get_array("off")
        .unwrap_or_else(|e| panic!("{case} off: {e}"))
        .iter()
        .map(|k| k.as_str().unwrap().to_string())
        .collect()
}

/// Loads the case's config and checks that it reads back as the case says:
/// the named switches off and every other switch on, so a fixture whose
/// config and `case.json` disagree cannot pass.
pub fn off_config(dir: &Path, case: &str) -> Timesfm3Config {
    let cfg = Timesfm3Config::from_config_file(&off_config_path(dir, case)).unwrap_or_else(|e| panic!("{case}: {e}"));
    let off = off_switches(dir, case);
    assert!(!off.is_empty(), "{case}: no switch is off");
    for (key, on) in [
        ("use_stitching", cfg.host.use_stitching),
        ("use_linear_detrending", cfg.host.use_linear_detrending),
        ("use_iterative_cpm_revin", cfg.host.use_iterative_cpm_revin),
        ("use_variate_attention", cfg.use_variate_attention),
        ("use_memory_efficient_attention", cfg.mea_scale),
    ] {
        assert_eq!(
            on,
            !off.iter().any(|k| k == key),
            "{case}: {key} in config.json vs case.json"
        );
    }
    cfg
}

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
pub const FORECAST_TINY: [&str; 6] = [
    "fa_ragged",
    "fb_options",
    "fc_multivariate_cov",
    "fd_edge",
    "fe_long",
    "fg_options_cov",
];
pub const FORECAST_REAL: [&str; 7] = [
    "fa_ragged",
    "fb_options",
    "fc_multivariate_cov",
    "fd_edge",
    "fe_long",
    "fg_options_cov",
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

/// `n` deterministic series of `rows` target variates, series `i` over
/// `len_of(i)` steps (a sinusoid of its own period and phase plus seeded
/// noise), for tests that need real-shaped inputs without a recorded fixture.
pub fn synthetic_series(n: usize, rows: usize, len_of: impl Fn(usize) -> usize, seed: u64) -> Vec<Series> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    };
    (0..n)
        .map(|i| {
            let len = len_of(i);
            let values = (0..rows * len)
                .map(|k| {
                    let (r, t) = (k / len, k % len);
                    let period = 11.0 + (i % 7) as f32 * 3.0 + r as f32;
                    10.0 + i as f32 * 0.25
                        + (t as f32 * std::f32::consts::TAU / period + i as f32).sin() * 2.0
                        + noise()
                })
                .collect();
            Series {
                target: Rows { rows, values },
                past_only: None,
                past_future: None,
            }
        })
        .collect()
}

/// `series` with `po` past-only and `pf` past+future covariate channels on
/// every series whose index passes `has`, derived from its own target (the
/// past+future ones run on through `horizon` steps).
pub fn with_covariates(
    mut series: Vec<Series>,
    horizon: usize,
    po: usize,
    pf: usize,
    has: impl Fn(usize) -> bool,
) -> Vec<Series> {
    for (i, s) in series.iter_mut().enumerate().filter(|(i, _)| has(*i)) {
        let len = s.target.values.len() / s.target.rows;
        let channel = |k: usize, n: usize| {
            (0..n)
                .map(|t| ((t + 3 * k + i) as f32 * 0.37).cos() + k as f32)
                .collect::<Vec<f32>>()
        };
        if po > 0 {
            s.past_only = Some(Rows {
                rows: po,
                values: (0..po).flat_map(|k| channel(k, len)).collect(),
            });
        }
        if pf > 0 {
            s.past_future = Some(Rows {
                rows: pf,
                values: (0..pf).flat_map(|k| channel(k + 5, len + horizon)).collect(),
            });
        }
    }
    series
}

/// `predict_batch` as it ran before group packing: one `decode()` per plan call
/// (cut by hand into slices of whole series when the call exceeds `max_rows`),
/// then `finish`. The reference packed forecasts are held to bit for bit.
pub fn predict_one_decode_per_call(
    m: &Timesfm3Model,
    series: &[Series],
    horizon: usize,
    opts: &PredictOptions,
    max_rows: usize,
) -> Vec<SeriesForecast> {
    let host = &m.config().host;
    let pl = plan(host, series, horizon, opts).unwrap();
    let outs: Vec<Forecast> = pl
        .calls
        .iter()
        .map(|call| {
            let per = host.rows_per_series(call.variates(), call.context, call.horizon);
            let chunk = (max_rows / per).max(1);
            let mut values = Vec::new();
            let mut shape = (0, 0, 0);
            for start in (0..call.batch).step_by(chunk) {
                let f = m
                    .decode(&call.slice_batch(start, (start + chunk).min(call.batch)).unwrap())
                    .unwrap();
                shape = (f.variates, f.horizon, f.quantiles);
                values.extend(f.values);
            }
            Forecast {
                batch: call.batch,
                variates: shape.0,
                horizon: shape.1,
                quantiles: shape.2,
                values,
            }
        })
        .collect();
    finish(&pl, &outs).unwrap()
}

/// Holds `got` to `want`, series by series, bit for bit.
pub fn assert_forecasts_bit_equal(what: &str, got: &[SeriesForecast], want: &[SeriesForecast]) {
    assert_eq!(got.len(), want.len(), "{what}: series");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            (g.variates, g.horizon, g.quantiles),
            (w.variates, w.horizon, w.quantiles),
            "{what} series {i}"
        );
        assert_bit_equal(&format!("{what} series {i}"), &g.values, &w.values);
    }
}

/// Refuses two f32 slices that differ in any element's bits (stricter than
/// `==`: it also tells 0.0 from -0.0 and refuses NaN on either side), or in
/// length.
pub fn assert_bit_equal(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() && !g.is_nan(),
            "{what}: element {i} is {g:?} ({:#010x}), expected {w:?} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}
