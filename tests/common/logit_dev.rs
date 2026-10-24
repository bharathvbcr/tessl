//! Logit deviation between two forwards of the same prompt, and the real-2B
//! transformers reference files it is usually measured against.
//!
//! One owner for the numbers `qwen35_model.rs`'s accuracy tests and its
//! `logit_deviation_report` print: max and mean |d| (absolute and as a
//! fraction of the reference's largest logit), top-1 agreement, and per-position
//! KL divergence. Pure host arithmetic in f64, so it needs no device.
//!
//! It fails closed. A NaN or an infinity in either side is counted and makes
//! every deviation `INFINITY` and the agreement zero, so no bound written as
//! `dev <= limit` can pass on it (`f64::max` would otherwise skip a NaN and
//! report a clean-looking number). A shape that cannot be compared panics.

use std::path::{Path, PathBuf};

use tessl::npy::read_npy;

/// The reference directory: `QWEN35_REF_DIR`, else `target/qwen35_ref`.
pub fn ref_dir() -> PathBuf {
    std::env::var_os("QWEN35_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_ref"))
}

/// `name.npy` from `dir`, widened to f64 (f32 and i64 files only).
pub fn load_ref_in(dir: &Path, name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = dir.join(format!("{name}.npy"));
    let a = read_npy(&path).unwrap_or_else(|e| panic!("{e}\n(run python3 tools/qwen35_ref/make_reference.py first)"));
    let data = if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| x as f64).collect()
    } else if let Some(i) = &a.data_i64 {
        i.iter().map(|&x| x as f64).collect()
    } else {
        panic!("{name}: unexpected dtype")
    };
    (a.shape, data)
}

/// `name.npy` from [`ref_dir`].
pub fn load_ref(name: &str) -> (Vec<usize>, Vec<f64>) {
    load_ref_in(&ref_dir(), name)
}

pub fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0
}

pub fn log_softmax(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// KL(p || q) from logits, per row.
pub fn kl_rows(p_logits: &[f64], q_logits: &[f64], vocab: usize) -> Vec<f64> {
    p_logits
        .chunks(vocab)
        .zip(q_logits.chunks(vocab))
        .map(|(p, q)| {
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
        })
        .collect()
}

/// How far `got` is from `want`, over `positions` rows of `vocab` logits.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitDev {
    /// max |got - want|.
    pub max_abs: f64,
    /// mean |got - want|.
    pub mean_abs: f64,
    /// max |want|: the scale `max_abs` and `mean_abs` are quoted against.
    pub ref_scale: f64,
    /// Positions whose argmax is the same in both.
    pub top1_agree: usize,
    pub positions: usize,
    /// Over positions, KL(want || got).
    pub max_kl: f64,
    pub mean_kl: f64,
    /// NaN or infinite values in `got` / `want`. Non-zero makes every
    /// deviation above `INFINITY` and `top1_agree` zero.
    pub nonfinite_got: usize,
    pub nonfinite_want: usize,
}

impl LogitDev {
    pub fn is_finite(&self) -> bool {
        self.nonfinite_got == 0 && self.nonfinite_want == 0
    }

    /// `max_abs / ref_scale`, the figure the accuracy bounds are written in.
    /// A zero reference scale gives 0 only for an exact match.
    pub fn max_rel(&self) -> f64 {
        self.rel(self.max_abs)
    }

    pub fn mean_rel(&self) -> f64 {
        self.rel(self.mean_abs)
    }

    fn rel(&self, abs: f64) -> f64 {
        if self.ref_scale.is_nan() || self.ref_scale <= 0.0 {
            return if abs == 0.0 { 0.0 } else { f64::INFINITY };
        }
        abs / self.ref_scale
    }

    /// True only for finite input whose largest deviation is within `bound`
    /// of the reference scale. False for any NaN or infinity in the logits, in
    /// `ref_scale`, in the deviation, or in `bound`: every operand is checked
    /// finite explicitly, so no comparison's NaN behaviour decides the result.
    pub fn within_rel(&self, bound: f64) -> bool {
        let rel = self.max_rel();
        self.is_finite() && self.ref_scale.is_finite() && bound.is_finite() && rel.is_finite() && rel <= bound
    }

    pub fn top1_misses(&self) -> usize {
        self.positions - self.top1_agree
    }

    /// One printable row: `label`, then the deviations the report compares.
    pub fn row(&self, label: &str) -> String {
        let mut s = format!(
            "{label:<34} max|d| {:.3e} ({:.3e} rel)  mean|d| {:.3e} ({:.3e} rel)  top-1 {}/{}  KL mean {:.3e} max {:.3e}",
            self.max_abs,
            self.max_rel(),
            self.mean_abs,
            self.mean_rel(),
            self.top1_agree,
            self.positions,
            self.mean_kl,
            self.max_kl,
        );
        if !self.is_finite() {
            s.push_str(&format!(
                "  NON-FINITE: {} in got, {} in want",
                self.nonfinite_got, self.nonfinite_want
            ));
        }
        s
    }
}

/// Compare `got` with `want`, both `[positions, vocab]` row-major.
///
/// Panics when `vocab` is zero, the lengths differ, the input is empty or
/// `vocab` does not divide the length: a shape that cannot be compared is not
/// a deviation of any size.
pub fn logit_dev(got: &[f32], want: &[f64], vocab: usize) -> LogitDev {
    assert!(vocab > 0, "logit_dev: vocab is 0");
    assert_eq!(got.len(), want.len(), "logit_dev: got and want differ in length");
    assert!(!got.is_empty(), "logit_dev: no logits to compare");
    assert_eq!(
        got.len() % vocab,
        0,
        "logit_dev: vocab {vocab} does not divide the {} logits",
        got.len()
    );
    let positions = got.len() / vocab;
    let nonfinite_got = got.iter().filter(|x| !x.is_finite()).count();
    let nonfinite_want = want.iter().filter(|x| !x.is_finite()).count();
    if nonfinite_got + nonfinite_want > 0 {
        return LogitDev {
            max_abs: f64::INFINITY,
            mean_abs: f64::INFINITY,
            ref_scale: want
                .iter()
                .filter(|x| x.is_finite())
                .fold(0.0, |m, x| f64::max(m, x.abs())),
            top1_agree: 0,
            positions,
            max_kl: f64::INFINITY,
            mean_kl: f64::INFINITY,
            nonfinite_got,
            nonfinite_want,
        };
    }
    let got64: Vec<f64> = got.iter().map(|&x| f64::from(x)).collect();
    let (mut max_abs, mut sum_abs, mut ref_scale) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got64.iter().zip(want) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        ref_scale = ref_scale.max(w.abs());
    }
    let top1_agree = got64
        .chunks(vocab)
        .zip(want.chunks(vocab))
        .filter(|(g, w)| argmax(g) == argmax(w))
        .count();
    let kl = kl_rows(want, &got64, vocab);
    let kl_max = kl.iter().cloned().fold(0.0, f64::max);
    let kl_mean = kl.iter().sum::<f64>() / kl.len() as f64;
    // The KL of finite logits is finite; a NaN here is an arithmetic failure
    // and must not read as a small number.
    let (max_kl, mean_kl) = if kl_max.is_finite() && kl_mean.is_finite() && kl.iter().all(|k| k.is_finite()) {
        (kl_max, kl_mean)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    LogitDev {
        max_abs,
        mean_abs: sum_abs / got.len() as f64,
        ref_scale,
        top1_agree,
        positions,
        max_kl,
        mean_kl,
        nonfinite_got,
        nonfinite_want,
    }
}

/// The bar SLQ calls "effectively lossless" for the Expected Acceptance Rate.
pub const EAR_LOSSLESS_BAR: f64 = 0.99;

/// SLQ's top-K width for EAR (`eval/run_ear.py --top-k`, default 10).
pub const EAR_TOP_K: usize = 10;

/// Expected Acceptance Rate of `got` against the reference `want`, per
/// position and aggregated.
///
/// The definition is SLQ's (IST-DASLab/SLQ, `eval/run_ear.py` calling
/// `compute_baseline_topk` / `compute_kl_and_ear` in `src/metrics/evaluation.py`,
/// read 2026-10-23): at each position take the reference's top-`k` logits and
/// their indices; gather the candidate's logits at those same indices;
/// `p = softmax(reference top-k logits)` and `q = softmax(gathered candidate
/// logits)`, both over the `k` entries only; `EAR = sum_i min(p_i, q_i)`,
/// averaged over positions. SLQ weights positions by a calibration mask; with no
/// mask every position has weight 1, which is what is computed here. Ties in
/// the reference break to the lower index.
#[derive(Debug, Clone, PartialEq)]
pub struct Ear {
    pub per_position: Vec<f64>,
    pub mean: f64,
    pub min: f64,
    pub positions: usize,
    pub k: usize,
    /// NaN or infinite values in `got` / `want`: any makes `mean` and `min` NaN.
    pub nonfinite: usize,
}

impl Ear {
    /// True only for finite input whose mean reaches `bar`.
    pub fn clears(&self, bar: f64) -> bool {
        self.nonfinite == 0 && self.mean.is_finite() && bar.is_finite() && self.mean >= bar
    }

    pub fn row(&self, label: &str) -> String {
        let mut s = format!(
            "{label:<34} EAR top-{} mean {:.6} min {:.6} over {} positions  clears {EAR_LOSSLESS_BAR}: {}",
            self.k,
            self.mean,
            self.min,
            self.positions,
            self.clears(EAR_LOSSLESS_BAR),
        );
        if self.nonfinite > 0 {
            s.push_str(&format!("  NON-FINITE: {}", self.nonfinite));
        }
        s
    }
}

fn softmax_f64(xs: &[f64]) -> Vec<f64> {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = xs.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// EAR of `got` against `want`, both `[positions, vocab]` row-major, with `k`
/// candidates per position. Panics on a shape that cannot be compared or `k`
/// outside `1..=vocab`.
pub fn ear(got: &[f32], want: &[f64], vocab: usize, k: usize) -> Ear {
    assert!(vocab > 0, "ear: vocab is 0");
    assert!((1..=vocab).contains(&k), "ear: k {k} outside 1..={vocab}");
    assert_eq!(got.len(), want.len(), "ear: got and want differ in length");
    assert!(!got.is_empty(), "ear: no logits to compare");
    assert_eq!(
        got.len() % vocab,
        0,
        "ear: vocab {vocab} does not divide the {} logits",
        got.len()
    );
    let positions = got.len() / vocab;
    let nonfinite = got.iter().filter(|x| !x.is_finite()).count() + want.iter().filter(|x| !x.is_finite()).count();
    if nonfinite > 0 {
        return Ear {
            per_position: vec![f64::NAN; positions],
            mean: f64::NAN,
            min: f64::NAN,
            positions,
            k,
            nonfinite,
        };
    }
    let per_position: Vec<f64> = got
        .chunks(vocab)
        .zip(want.chunks(vocab))
        .map(|(g, w)| {
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| w[b].total_cmp(&w[a]).then(a.cmp(&b)));
            idx.truncate(k);
            let p = softmax_f64(&idx.iter().map(|&i| w[i]).collect::<Vec<_>>());
            let q = softmax_f64(&idx.iter().map(|&i| f64::from(g[i])).collect::<Vec<_>>());
            p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum()
        })
        .collect();
    let mean = per_position.iter().sum::<f64>() / positions as f64;
    let min = per_position.iter().cloned().fold(f64::INFINITY, f64::min);
    Ear {
        per_position,
        mean,
        min,
        positions,
        k,
        nonfinite: 0,
    }
}
