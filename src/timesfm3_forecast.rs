//! TimesFM 3's forecaster, device-free: upstream's
//! `TimesFM3Forecaster.predict_batch()` (google-research/timesfm at 918df02,
//! `src/timesfm3/torch/timesfm3_forecaster.py`; Apache-2.0) around
//! [`crate::timesfm3::Timesfm3Model::decode`].
//!
//! [`plan`] turns ragged series into the `decode()` calls upstream makes:
//! drop leading time steps where every target variate is NaN, fill the other
//! NaNs by linear interpolation, optionally z-normalise each row and append
//! each series' negation (symmetric averaging), optionally edge-pad the
//! past+future covariates to the patch-rounded horizon, then cut the queries
//! into groups of `per_core_batch_size`. Each group is left-padded (masked)
//! or truncated to its own context, the longest in the group rounded up to a
//! whole patch and capped at [`MAX_CONTEXT_LENGTH`]; a group where any series
//! has covariates gives the others masked zero placeholders.
//!
//! [`finish`] turns the calls' forecasts into one quantile forecast per
//! series: keep the targets and the requested horizon, sort each step's
//! quantiles, average a series with its negation (quantiles reversed), undo
//! the z-normalisation and clamp at zero the variates whose input was
//! non-negative.
//!
//! The grouping is upstream's and changes the numbers (a group's context and
//! horizon follow from its members), so it is reproduced exactly; how a group
//! is executed is not (`Timesfm3Model::predict_batch` may run it in pieces).

use crate::timesfm3::Forecast;
use crate::timesfm3_host::{DecodeInput, HostConfig};

/// Upstream's `_MAX_CONTEXT_LENGTH`: a series longer than this (rounded up
/// to a whole patch) keeps only its last points.
pub const MAX_CONTEXT_LENGTH: usize = 15360;
/// Upstream's `_SIGMA_THRESHOLD`: a row whose standard deviation is below
/// it is z-normalised with sigma 1.
const SIGMA_THRESHOLD: f64 = 1e-7;
/// Upstream's `_ModelConfig.median_quantile_index` default.
const MEDIAN_QUANTILE_INDEX: usize = 4;

/// The quantile [`SeriesForecast::point`] reports for `quantiles` outputs:
/// upstream's index 4, or the middle one when there are no more than 4.
pub fn median_quantile_index(quantiles: usize) -> usize {
    if MEDIAN_QUANTILE_INDEX < quantiles {
        MEDIAN_QUANTILE_INDEX
    } else {
        quantiles / 2
    }
}

/// Upstream's `padding_mode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CovariatePadding {
    /// `"none"`: the past+future covariates cover the requested horizon only.
    #[default]
    None,
    /// `"edge"`: they are extended to the patch-rounded horizon by repeating
    /// their last value.
    Edge,
}

/// `predict_batch()`'s options, with upstream's defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictOptions {
    /// Queries per `decode()` call (upstream's `per_core_batch_size`).
    pub per_core_batch_size: usize,
    pub use_symmetric_averaging: bool,
    pub make_positive: bool,
    pub sort_quantiles: bool,
    pub use_znorm: bool,
    pub padding: CovariatePadding,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            per_core_batch_size: 4,
            use_symmetric_averaging: false,
            make_positive: false,
            sort_quantiles: true,
            use_znorm: false,
            padding: CovariatePadding::None,
        }
    }
}

/// `rows` series of equal length, row-major `[rows, len]`; NaN is a
/// missing point.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rows {
    pub rows: usize,
    pub values: Vec<f32>,
}

impl Rows {
    /// One row.
    pub fn one(values: Vec<f32>) -> Self {
        Self { rows: 1, values }
    }

    fn len(&self) -> usize {
        self.values.len() / self.rows.max(1)
    }

    fn row(&self, r: usize) -> &[f32] {
        let n = self.len();
        &self.values[r * n..(r + 1) * n]
    }
}

/// One series to forecast: its target variates over the context and,
/// optionally, past-only covariates (same length) and past+future covariates
/// (the context plus the horizon).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Series {
    pub target: Rows,
    pub past_only: Option<Rows>,
    pub past_future: Option<Rows>,
}

/// One series' forecast: `[variates, horizon, quantiles]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesForecast {
    pub variates: usize,
    pub horizon: usize,
    pub quantiles: usize,
    pub values: Vec<f32>,
}

impl SeriesForecast {
    /// The point forecast, `[variates, horizon]`: quantile
    /// [`median_quantile_index`] (upstream's `ForecastOutput.forecast`).
    pub fn point(&self) -> Vec<f32> {
        let m = median_quantile_index(self.quantiles);
        self.values.chunks_exact(self.quantiles).map(|q| q[m]).collect()
    }
}

/// [`plan`]'s output: the `decode()` calls, in order, and what [`finish`]
/// needs to turn their forecasts into per-series ones.
#[derive(Clone, Debug)]
pub struct Plan {
    /// One per group of up to `per_core_batch_size` queries. Every mask
    /// already has the group's padding folded in.
    pub calls: Vec<DecodeInput>,
    horizon: usize,
    targets: usize,
    opts: PredictOptions,
    /// Per input series, per target row: the z-normalisation `(mu, sigma)`
    /// (`(0, 1)` without it).
    znorm: Vec<Vec<(f32, f32)>>,
    /// Per input series, per target row: at least one finite point and none
    /// negative (`make_positive` clamps those rows).
    nonneg: Vec<Vec<bool>>,
}

impl Plan {
    /// Series in the request (queries are twice as many with symmetric
    /// averaging).
    pub fn series(&self) -> usize {
        self.znorm.len()
    }
}

/// One query as the group formatting sees it: `[rows, len]` arrays.
struct Query {
    len: usize,
    target: Vec<Vec<f32>>,
    past_only: Option<Vec<Vec<f32>>>,
    past_future: Option<Vec<Vec<f32>>>,
}

/// numpy's `interp` over the NaN positions of `row` from its finite ones (in
/// f64, rounded once), the end values held flat beyond the first and last;
/// an all-NaN row becomes zeros (upstream's fallback).
fn interpolate(row: &mut [f32]) {
    let valid: Vec<usize> = (0..row.len()).filter(|&i| !row[i].is_nan()).collect();
    if valid.len() == row.len() {
        return;
    }
    let (Some(&first), Some(&last)) = (valid.first(), valid.last()) else {
        row.fill(0.0);
        return;
    };
    let mut j = 0usize;
    for i in 0..row.len() {
        if !row[i].is_nan() {
            continue;
        }
        if i < first {
            row[i] = row[first];
        } else if i > last {
            row[i] = row[last];
        } else {
            while valid[j + 1] < i {
                j += 1;
            }
            let (x0, x1) = (valid[j], valid[j + 1]);
            let (y0, y1) = (f64::from(row[x0]), f64::from(row[x1]));
            let slope = (y1 - y0) / (x1 - x0) as f64;
            row[i] = (slope * (i - x0) as f64 + y0) as f32;
        }
    }
}

/// Upstream's `_znorm_stats` on an interpolated row: mean and population
/// standard deviation (summed in f64, rounded to f32 as numpy's float32
/// results are), sigma 1 below the threshold.
fn znorm_stats(row: &[f32]) -> (f32, f32) {
    let n = row.len() as f64;
    let mu = row.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = row.iter().map(|&x| (f64::from(x) - mu).powi(2)).sum::<f64>() / n;
    let (mu, sigma) = (mu as f32, var.sqrt() as f32);
    let mu = if mu.is_finite() { mu } else { 0.0 };
    let sigma = if !sigma.is_finite() || f64::from(sigma) < SIGMA_THRESHOLD {
        1.0
    } else {
        sigma
    };
    (mu, sigma)
}

fn normalise(rows: &mut [Vec<f32>]) -> Vec<(f32, f32)> {
    rows.iter_mut()
        .map(|r| {
            let (mu, sigma) = znorm_stats(r);
            for x in r.iter_mut() {
                *x = (*x - mu) / sigma;
            }
            (mu, sigma)
        })
        .collect()
}

fn split_rows(what: &str, i: usize, r: &Rows, want_len: usize) -> Result<Vec<Vec<f32>>, String> {
    if r.rows == 0 || r.values.len() != r.rows * want_len {
        return Err(format!(
            "timesfm3 predict: series {i}: {what} holds {} values in {} rows, expected {} rows of {want_len}",
            r.values.len(),
            r.rows,
            r.rows.max(1)
        ));
    }
    if let Some(t) = r.values.iter().position(|x| x.is_infinite()) {
        return Err(format!("timesfm3 predict: series {i}: {what} value {t} is infinite"));
    }
    Ok((0..r.rows).map(|k| r.row(k).to_vec()).collect())
}

/// Upstream's `predict_batch()` up to the `decode()` calls. Validates every
/// series first (lengths, matching target counts, no infinities).
pub fn plan(cfg: &HostConfig, series: &[Series], horizon: usize, opts: &PredictOptions) -> Result<Plan, String> {
    if horizon == 0 {
        return Err("timesfm3 predict: horizon must be a positive integer".into());
    }
    if opts.per_core_batch_size == 0 {
        return Err("timesfm3 predict: per_core_batch_size must be at least 1".into());
    }
    let p = cfg.input_patch_len;
    if p == 0 || cfg.output_patch_len == 0 {
        return Err("timesfm3 predict: patch lengths must be non-zero".into());
    }
    let global_horizon = horizon.div_ceil(cfg.output_patch_len) * cfg.output_patch_len;
    let global_context = MAX_CONTEXT_LENGTH.div_ceil(p) * p;
    let targets = series.first().map_or(1, |s| s.target.rows);

    let mut queries: Vec<Query> = Vec::with_capacity(series.len() * 2);
    let mut znorm = Vec::with_capacity(series.len());
    let mut nonneg = Vec::with_capacity(series.len());
    for (i, s) in series.iter().enumerate() {
        let len = s.target.len();
        if s.target.rows == 0 || len == 0 || s.target.values.len() != s.target.rows * len {
            return Err(format!(
                "timesfm3 predict: series {i} must have at least one time step and one variate"
            ));
        }
        if s.target.rows != targets {
            return Err(format!(
                "timesfm3 predict: every series needs the same number of target variates; series 0 has \
                 {targets}, series {i} has {}",
                s.target.rows
            ));
        }
        let mut target = split_rows("target", i, &s.target, len)?;
        let mut po = s
            .past_only
            .as_ref()
            .map(|r| split_rows("past_only", i, r, len))
            .transpose()?;
        let mut pf = s
            .past_future
            .as_ref()
            .map(|r| split_rows("past_future", i, r, len + horizon))
            .transpose()?;
        nonneg.push(
            target
                .iter()
                .map(|r| {
                    let mut any = false;
                    for &x in r.iter().filter(|x| !x.is_nan()) {
                        if x < 0.0 {
                            return false;
                        }
                        any = true;
                    }
                    any
                })
                .collect::<Vec<bool>>(),
        );

        // Leading steps where every target variate is NaN are dropped (from
        // the covariates too); a series that is NaN throughout becomes zeros.
        let first_valid = (0..len).find(|&t| target.iter().any(|r| !r[t].is_nan()));
        match first_valid {
            None => target.iter_mut().for_each(|r| r.fill(0.0)),
            Some(0) => {}
            Some(f) => {
                for rows in [Some(&mut target), po.as_mut(), pf.as_mut()].into_iter().flatten() {
                    for r in rows.iter_mut() {
                        r.drain(..f);
                    }
                }
            }
        }
        for rows in [Some(&mut target), po.as_mut(), pf.as_mut()].into_iter().flatten() {
            rows.iter_mut().for_each(|r| interpolate(r));
        }
        if opts.use_znorm {
            znorm.push(normalise(&mut target));
            for rows in [po.as_mut(), pf.as_mut()].into_iter().flatten() {
                normalise(rows);
            }
        } else {
            znorm.push(vec![(0.0, 1.0); targets]);
        }
        let q = Query {
            len: target[0].len(),
            target,
            past_only: po,
            past_future: pf,
        };
        if opts.use_symmetric_averaging {
            let neg = |rows: &Vec<Vec<f32>>| rows.iter().map(|r| r.iter().map(|&x| -x).collect()).collect();
            let mirror = Query {
                len: q.len,
                target: neg(&q.target),
                past_only: q.past_only.as_ref().map(neg),
                past_future: q.past_future.as_ref().map(neg),
            };
            queries.push(q);
            queries.push(mirror);
        } else {
            queries.push(q);
        }
    }
    if opts.padding == CovariatePadding::Edge && global_horizon > horizon {
        for q in &mut queries {
            for r in q.past_future.iter_mut().flatten() {
                let last = *r.last().expect("a past+future row spans the horizon");
                r.resize(r.len() + global_horizon - horizon, last);
            }
        }
    }

    let mut calls = Vec::new();
    for (gi, group) in queries.chunks(opts.per_core_batch_size).enumerate() {
        calls.push(group_call(gi, group, targets, p, global_context, global_horizon)?);
    }
    Ok(Plan {
        calls,
        horizon,
        targets,
        opts: opts.clone(),
        znorm,
        nonneg,
    })
}

/// One group's `decode()` input: every query left-padded (masked) or
/// truncated to the group's context; masked zero placeholders for a query
/// without the covariates another one has.
fn group_call(
    gi: usize,
    group: &[Query],
    u: usize,
    p: usize,
    global_context: usize,
    global_horizon: usize,
) -> Result<DecodeInput, String> {
    let g = group.len();
    let longest = group.iter().map(|q| q.len).max().unwrap_or(1);
    let c = (longest.div_ceil(p) * p).min(global_context).max(p);
    let channels = |pick: fn(&Query) -> Option<&Vec<Vec<f32>>>, what: &str| -> Result<Option<usize>, String> {
        let mut found: Option<usize> = None;
        for q in group {
            if let Some(rows) = pick(q) {
                match found {
                    Some(n) if n != rows.len() => {
                        return Err(format!(
                            "timesfm3 predict: group {gi} mixes {what} with {n} and {} channels; upstream stacks \
                             them, so every series in a group needs the same count",
                            rows.len()
                        ))
                    }
                    _ => found = Some(rows.len()),
                }
            }
        }
        Ok(found)
    };
    let po_ch = channels(|q| q.past_only.as_ref(), "past-only covariates")?;
    let pf_ch = channels(|q| q.past_future.as_ref(), "past+future covariates")?;
    // decode() reads the horizon off the past+future width when there is one.
    let mut future: Option<usize> = None;
    for q in group {
        if let Some(rows) = &q.past_future {
            let f = rows[0].len() - q.len;
            if future.is_some_and(|x| x != f) {
                return Err(format!(
                    "timesfm3 predict: group {gi} mixes past+future covariate horizons"
                ));
            }
            future = Some(f);
        }
    }
    let h = future.unwrap_or(global_horizon);

    // Where each query's points land in the group's context, and the padding
    // mask (true where padded) every variate's context mask is ORed with.
    let place = |q: &Query| -> (usize, usize) {
        if q.len > c {
            (q.len - c, 0) // (skip, pad)
        } else {
            (0, c - q.len)
        }
    };
    let mut target = vec![0.0f32; g * u * c];
    let mut target_mask = vec![0u8; g * u * c];
    let mut past_only = po_ch.map(|n| (n, vec![0.0f32; g * n * c], vec![0u8; g * n * c]));
    let mut past_future = pf_ch.map(|n| (n, vec![0.0f32; g * n * (c + h)], vec![0u8; g * n * (c + h)]));
    for (b, q) in group.iter().enumerate() {
        let (skip, pad) = place(q);
        let width = q.len - skip;
        for (r, row) in q.target.iter().enumerate() {
            let at = (b * u + r) * c;
            target[at + pad..at + c].copy_from_slice(&row[skip..skip + width]);
            target_mask[at..at + pad].fill(1);
        }
        if let Some((n, vals, mask)) = past_only.as_mut() {
            for r in 0..*n {
                let at = (b * *n + r) * c;
                match &q.past_only {
                    Some(rows) => {
                        vals[at + pad..at + c].copy_from_slice(&rows[r][skip..skip + width]);
                        mask[at..at + pad].fill(1);
                    }
                    None => mask[at..at + c].fill(1),
                }
            }
        }
        if let Some((n, vals, mask)) = past_future.as_mut() {
            let wd = c + h;
            for r in 0..*n {
                let at = (b * *n + r) * wd;
                match &q.past_future {
                    Some(rows) => {
                        let src = &rows[r][skip..];
                        vals[at + pad..at + wd].copy_from_slice(src);
                        mask[at..at + pad].fill(1);
                    }
                    None => mask[at..at + wd].fill(1),
                }
            }
        }
    }
    let (num_past_only, past_only, past_only_mask) = match past_only {
        Some((n, v, m)) => (n, Some(v), Some(m)),
        None => (0, None, None),
    };
    let (num_past_future, past_future, past_future_mask) = match past_future {
        Some((n, v, m)) => (n, Some(v), Some(m)),
        None => (0, None, None),
    };
    Ok(DecodeInput {
        batch: g,
        context: c,
        horizon: h,
        num_target: u,
        target,
        target_mask: Some(target_mask),
        num_past_only,
        past_only,
        past_only_mask,
        num_past_future,
        past_future,
        past_future_mask,
    })
}

/// The rest of upstream's `predict_batch()`: `outs` holds one forecast per
/// [`Plan::calls`] entry, in order.
pub fn finish(plan: &Plan, outs: &[Forecast]) -> Result<Vec<SeriesForecast>, String> {
    if outs.len() != plan.calls.len() {
        return Err(format!(
            "timesfm3 predict: {} forecasts for {} decode calls",
            outs.len(),
            plan.calls.len()
        ));
    }
    let (u, h) = (plan.targets, plan.horizon);
    let q = outs.first().map_or(0, |f| f.quantiles);
    // Per query: [u, h, q], the targets and the requested horizon only.
    let mut per_query: Vec<Vec<f32>> = Vec::new();
    for (k, (f, call)) in outs.iter().zip(&plan.calls).enumerate() {
        if f.batch != call.batch || f.horizon != call.horizon || f.variates < u || f.quantiles != q || q == 0 {
            return Err(format!(
                "timesfm3 predict: forecast {k} is [{}, {}, {}, {}], expected [{}, >= {u}, {}, {q}]",
                f.batch, f.variates, f.horizon, f.quantiles, call.batch, call.horizon
            ));
        }
        if f.values.len() != f.batch * f.variates * f.horizon * q || f.horizon < h {
            return Err(format!(
                "timesfm3 predict: forecast {k} holds {} values",
                f.values.len()
            ));
        }
        for b in 0..f.batch {
            let mut one = Vec::with_capacity(u * h * q);
            for r in 0..u {
                let at = ((b * f.variates + r) * f.horizon) * q;
                one.extend_from_slice(&f.values[at..at + h * q]);
            }
            per_query.push(one);
        }
    }
    if plan.opts.sort_quantiles {
        for one in &mut per_query {
            for step in one.chunks_exact_mut(q) {
                step.sort_by(f32::total_cmp);
            }
        }
    }
    let mut out: Vec<Vec<f32>> = if plan.opts.use_symmetric_averaging {
        let (pairs, odd) = per_query.as_chunks::<2>();
        if !odd.is_empty() {
            return Err(format!("timesfm3 predict: {} queries cannot pair up", per_query.len()));
        }
        pairs
            .iter()
            .map(|[pos, neg]| {
                let mut avg = vec![0.0f32; pos.len()];
                for ((a, ps), ns) in avg
                    .chunks_exact_mut(q)
                    .zip(pos.chunks_exact(q))
                    .zip(neg.chunks_exact(q))
                {
                    for k in 0..q {
                        let n = if q > 1 { ns[q - 1 - k] } else { ns[k] };
                        a[k] = (ps[k] - n) / 2.0;
                    }
                }
                avg
            })
            .collect()
    } else {
        per_query
    };
    if out.len() != plan.series() {
        return Err(format!(
            "timesfm3 predict: {} forecasts for {} series",
            out.len(),
            plan.series()
        ));
    }
    for (i, one) in out.iter_mut().enumerate() {
        for r in 0..u {
            let row = &mut one[r * h * q..(r + 1) * h * q];
            if plan.opts.use_znorm {
                let (mu, sigma) = plan.znorm[i][r];
                row.iter_mut().for_each(|x| *x = *x * sigma + mu);
            }
            // numpy's maximum keeps NaN, which f32::max would not.
            if plan.opts.make_positive && plan.nonneg[i][r] {
                row.iter_mut().filter(|x| **x < 0.0).for_each(|x| *x = 0.0);
            }
        }
    }
    Ok(out
        .into_iter()
        .map(|values| SeriesForecast {
            variates: u,
            horizon: h,
            quantiles: q,
            values,
        })
        .collect())
}
