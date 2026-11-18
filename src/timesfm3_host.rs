//! TimesFM 3's host-side forecasting steps, device-free: everything
//! upstream's `TimesFM3Torch.decode()` does around the transformer.
//!
//! Upstream (google-research/timesfm at 918df02, `src/timesfm3/torch/model.py`,
//! `util.py`, `cpm_revin_refine.py`; Apache-2.0) runs these as torch ops on the
//! device. They are `O(batch * variates * time)` scalar work with a sequential
//! scan over patches (the running RevIN statistics and the CPM refinement), so
//! here they run on the host, between one upload and one read-back:
//!
//! 1. [`prepare`]: left-pad the context to a whole patch, fit and (where it
//!    explains enough variance) remove a linear trend per variate, lay out the
//!    horizon, and cut everything into 32-point patches.
//! 2. [`preprocess`]: running RevIN statistics per patch, the normalised
//!    input patch, the next `rolls` patches as future covariates, and the
//!    192-wide residual-block input; the per-row count of leading fully-masked
//!    patches that attention masks.
//! 3. The device runs the residual block, the stacked transformer and the
//!    output head on the residual-block input ([`crate::timesfm3`]).
//! 4. [`postprocess`]: refine the RevIN statistics over the horizon from the
//!    model's own median (CPM iterative RevIN), de-normalise, clip, stitch the
//!    overlapping patch forecasts and add the trend back.
//!
//! Shapes follow upstream's names: `b` batch, `v` variates (targets, then
//! past-only covariates, then past-and-future covariates), `n` patches, `p`
//! the input patch length (32), `o` the output patch length (64), `q` the
//! quantiles (9). Every array is row-major f32 (bool as `u8`, 1 = masked).

/// The model constants the host steps need (from `config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct HostConfig {
    pub input_patch_len: usize,
    pub output_patch_len: usize,
    pub num_quantiles: usize,
    pub value_clip: f32,
    pub use_stitching: bool,
    pub use_linear_detrending: bool,
    pub linear_detrending_threshold: f32,
    pub use_iterative_cpm_revin: bool,
}

impl HostConfig {
    /// `output_patch_len / input_patch_len`: future patches per input patch.
    pub fn rolls(&self) -> usize {
        self.output_patch_len / self.input_patch_len
    }

    /// The residual block's input width, `2 * (p + o)`.
    pub fn resblock_in(&self) -> usize {
        2 * (self.input_patch_len + self.output_patch_len)
    }

    /// The output head's width, `o * q`.
    pub fn head_out(&self) -> usize {
        self.output_patch_len * self.num_quantiles
    }

    fn validate(&self) -> Result<(), String> {
        let p = self.input_patch_len;
        let o = self.output_patch_len;
        if p == 0 || o == 0 || self.num_quantiles == 0 {
            return Err("timesfm3: patch lengths and quantile count must be non-zero".into());
        }
        if o % p != 0 {
            return Err(format!("timesfm3: output patch {o} is not a multiple of input patch {p}"));
        }
        if self.use_stitching && o <= p {
            return Err("timesfm3: stitching needs output_patch_len > input_patch_len".into());
        }
        if !(self.value_clip > 0.0) {
            return Err(format!("timesfm3: value_clip must be positive, got {}", self.value_clip));
        }
        Ok(())
    }
}

/// One batch of series to forecast, as `decode()` takes it.
///
/// `target` is `[b, u, context]`; `past_only` `[b, po, context]`;
/// `past_future` `[b, w, context + horizon]`. A mask is 1 where a point is
/// missing; a missing mask means every point is present. NaN values are
/// allowed only where masked (upstream replaces NaN by 0 before the model,
/// and a masked point is zeroed anyway).
#[derive(Clone, Debug, Default)]
pub struct DecodeInput {
    pub batch: usize,
    pub context: usize,
    pub horizon: usize,
    pub num_target: usize,
    pub target: Vec<f32>,
    pub target_mask: Option<Vec<u8>>,
    pub num_past_only: usize,
    pub past_only: Option<Vec<f32>>,
    pub past_only_mask: Option<Vec<u8>>,
    pub num_past_future: usize,
    pub past_future: Option<Vec<f32>>,
    pub past_future_mask: Option<Vec<u8>>,
}

/// [`prepare`]'s output: the forward's patched input and what
/// [`postprocess`] needs to turn its logits back into a forecast.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub batch: usize,
    pub variates: usize,
    pub patches: usize,
    /// Context patches (the context left-padded to a whole patch).
    pub context_patches: usize,
    /// The padded context length, `context_patches * p`.
    pub context: usize,
    pub horizon: usize,
    /// `[b, v, n, p]`.
    pub values: Vec<f32>,
    /// `[b, v, n, p]`, 1 = masked.
    pub masks: Vec<u8>,
    /// `[b, v, n]`: 1 for targets and past-only covariates (no known future).
    pub patch_is_target: Vec<u8>,
    /// `[b, n]`: 1 for horizon patches.
    pub cpm: Vec<u8>,
    /// `[b, v]` linear trend slope, intercept and whether it was removed.
    pub trend_m: Vec<f32>,
    pub trend_c: Vec<f32>,
    pub detrended: Vec<u8>,
}

/// [`preprocess`]'s output.
#[derive(Clone, Debug)]
pub struct Preprocessed {
    /// `[b, v, n, 2 * (p + o)]`: the residual block's input.
    pub resblock_input: Vec<f32>,
    /// `[b, v, n]` running count, mean and standard deviation.
    pub running_n: Vec<f32>,
    pub running_mean: Vec<f32>,
    pub running_std: Vec<f32>,
    /// `[b * v]`: leading fully-masked patches per row. Sequence attention
    /// masks those keys (upstream's `cumprod` of the all-masked flag).
    pub leading_masked: Vec<u32>,
}

const TOLERANCE: f32 = 1e-6;

fn safe_sigma(s: f32) -> f32 {
    if s < TOLERANCE {
        1.0
    } else {
        s
    }
}

fn check_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        return Err(format!("timesfm3: {what} holds {got} values, expected {want}"));
    }
    Ok(())
}

fn checked_mul(dims: &[usize], what: &str) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("timesfm3: {what} size {dims:?} overflows"))
}

/// One running-statistics step (upstream `util.update_running_stats`): fold
/// the unmasked points of `x` into `(n, mu, sigma)`.
///
/// The two sums over the patch accumulate in f64 and round once: torch's f32
/// sums are pairwise, which a sequential f32 loop is not, and f64 is closer to
/// both than they are to each other.
pub fn update_running_stats(n: f32, mu: f32, sigma: f32, x: &[f32], mask: &[u8]) -> (f32, f32, f32) {
    let mut inc_n = 0.0f32;
    let mut sum = 0.0f64;
    for (&xi, &m) in x.iter().zip(mask) {
        if m == 0 {
            inc_n += 1.0;
            sum += f64::from(xi);
        }
    }
    let inc_sum = sum as f32;
    let safe_inc_n = inc_n.max(1.0);
    let inc_mu = inc_sum / safe_inc_n;
    let mut sq = 0.0f64;
    for (&xi, &m) in x.iter().zip(mask) {
        if m == 0 {
            let d = f64::from(xi - inc_mu);
            sq += d * d;
        }
    }
    let inc_var = sq as f32 / safe_inc_n;
    let new_n = n + inc_n;
    let safe_new_n = new_n.max(1.0);
    let new_mu = (n * mu + inc_sum) / safe_new_n;
    let new_sigma = ((n * sigma * sigma
        + inc_n * inc_var
        + n * (mu - new_mu) * (mu - new_mu)
        + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu))
        / safe_new_n)
        .sqrt();
    (new_n, new_mu, new_sigma)
}

/// Upstream `decode()` up to the `forward()` call: pad, detrend, lay out the
/// horizon and patch. Validates every shape first.
pub fn prepare(cfg: &HostConfig, inp: &DecodeInput) -> Result<Prepared, String> {
    cfg.validate()?;
    let (b, u, ctx0, horizon) = (inp.batch, inp.num_target, inp.context, inp.horizon);
    if b == 0 || u == 0 || ctx0 == 0 {
        return Err("timesfm3: batch, targets and context must be non-zero".into());
    }
    if horizon == 0 {
        return Err("timesfm3: decode needs horizon > 0".into());
    }
    let po = if inp.past_only.is_some() { inp.num_past_only } else { 0 };
    let w = if inp.past_future.is_some() { inp.num_past_future } else { 0 };
    if inp.past_only.is_some() && po == 0 || inp.past_future.is_some() && w == 0 {
        return Err("timesfm3: a covariate array was given with zero channels".into());
    }
    check_len("target", inp.target.len(), checked_mul(&[b, u, ctx0], "target")?)?;
    if let Some(m) = &inp.target_mask {
        check_len("target_mask", m.len(), inp.target.len())?;
    }
    if let Some(x) = &inp.past_only {
        check_len("past_only", x.len(), checked_mul(&[b, po, ctx0], "past_only")?)?;
        if let Some(m) = &inp.past_only_mask {
            check_len("past_only_mask", m.len(), x.len())?;
        }
    } else if inp.past_only_mask.is_some() {
        return Err("timesfm3: past_only_mask without past_only".into());
    }
    let pf_len = ctx0.checked_add(horizon).ok_or("timesfm3: context + horizon overflows")?;
    if let Some(x) = &inp.past_future {
        check_len("past_future", x.len(), checked_mul(&[b, w, pf_len], "past_future")?)?;
        if let Some(m) = &inp.past_future_mask {
            check_len("past_future_mask", m.len(), x.len())?;
        }
    } else if inp.past_future_mask.is_some() {
        return Err("timesfm3: past_future_mask without past_future".into());
    }

    let p = cfg.input_patch_len;
    let pad = (p - ctx0 % p) % p;
    let context = ctx0 + pad;
    let v = u + po + w;

    // Horizon layout (stitching: forecast patches of p, plus rolls - 1 so the
    // last forecast patch's output patch lies inside the sequence).
    let (num_horizon_patches, padded_horizon) = if cfg.use_stitching {
        let extract = (2 * p).min(cfg.output_patch_len);
        let overlap = extract - p;
        let nfp = horizon.saturating_sub(overlap).div_ceil(p).max(1);
        let nhp = nfp + cfg.rolls() - 1;
        (nhp, nhp * p)
    } else {
        let padded = horizon.div_ceil(cfg.output_patch_len) * cfg.output_patch_len;
        (padded / p, padded)
    };
    let ncp = context / p;
    let n = ncp + num_horizon_patches;
    let total = context + padded_horizon;

    // Context values and masks per variate, left-padded; the global mask is
    // just the padding (decode()'s `mask` argument is not exposed).
    let mut ctx_vals = vec![0.0f32; b * v * context];
    let mut ctx_masks = vec![0u8; b * v * context];
    let fill = |dst_v: &mut [f32], dst_m: &mut [u8], src: &[f32], src_m: Option<&[u8]>| {
        for t in 0..context {
            if t < pad {
                dst_v[t] = 0.0;
                dst_m[t] = 1;
            } else {
                let s = t - pad;
                dst_v[t] = src[s];
                dst_m[t] = src_m.map_or(0, |m| (m[s] != 0) as u8);
            }
        }
    };
    for bi in 0..b {
        for vi in 0..v {
            let row = (bi * v + vi) * context;
            let (dv, dm) = (&mut ctx_vals[row..row + context], &mut ctx_masks[row..row + context]);
            if vi < u {
                let s = (bi * u + vi) * ctx0;
                fill(dv, dm, &inp.target[s..s + ctx0], inp.target_mask.as_ref().map(|m| &m[s..s + ctx0]));
            } else if vi < u + po {
                let s = (bi * po + (vi - u)) * ctx0;
                let x = inp.past_only.as_ref().expect("po > 0");
                fill(dv, dm, &x[s..s + ctx0], inp.past_only_mask.as_ref().map(|m| &m[s..s + ctx0]));
            } else {
                let s = (bi * w + (vi - u - po)) * pf_len;
                let x = inp.past_future.as_ref().expect("w > 0");
                fill(dv, dm, &x[s..s + ctx0], inp.past_future_mask.as_ref().map(|m| &m[s..s + ctx0]));
            }
        }
    }

    // Linear detrending over the valid context points, t in (-(C-1)..=0)/C.
    let mut trend_m = vec![0.0f32; b * v];
    let mut trend_c = vec![0.0f32; b * v];
    let mut detrended = vec![0u8; b * v];
    let cf = context as f32;
    let tn = |t: usize| (t as f32 - (context as f32 - 1.0)) / cf;
    if cfg.use_linear_detrending {
        for r in 0..b * v {
            let vals = &mut ctx_vals[r * context..(r + 1) * context];
            let msk = &ctx_masks[r * context..(r + 1) * context];
            // Sums over up to the whole context: f64 accumulation, one
            // rounding each (see update_running_stats). The products are
            // formed in f32, as torch forms them.
            let (mut n_v, mut st, mut st2, mut sy, mut sty, mut sy2) = (0.0f32, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for t in 0..context {
                if msk[t] == 0 {
                    let (x, y) = (tn(t), vals[t]);
                    n_v += 1.0;
                    st += f64::from(x);
                    st2 += f64::from(x * x);
                    sy += f64::from(y);
                    sty += f64::from(x * y);
                    sy2 += f64::from(y * y);
                }
            }
            let (st, st2, sy, sty, sy2) = (st as f32, st2 as f32, sy as f32, sty as f32, sy2 as f32);
            let det = n_v * st2 - st * st;
            let nv1 = n_v.max(1.0);
            let m = if det == 0.0 { 0.0 } else { (n_v * sty - st * sy) / det };
            let c = if det == 0.0 {
                if n_v > 0.0 {
                    sy / nv1
                } else {
                    0.0
                }
            } else {
                (sy - m * st) / nv1
            };
            let mean_y = sy / nv1;
            let std_orig = (sy2 / nv1 - mean_y * mean_y).max(0.0).sqrt();
            let (mut syd, mut syd2) = (0.0f64, 0.0f64);
            for t in 0..context {
                if msk[t] == 0 {
                    let yd = vals[t] - (m * tn(t) + c);
                    syd += f64::from(yd);
                    syd2 += f64::from(yd * yd);
                }
            }
            let (syd, syd2) = (syd as f32, syd2 as f32);
            let mean_yd = syd / nv1;
            let std_det = (syd2 / nv1 - mean_yd * mean_yd).max(0.0).sqrt();
            trend_m[r] = m;
            trend_c[r] = c;
            if std_det < cfg.linear_detrending_threshold * std_orig {
                detrended[r] = 1;
                for t in 0..context {
                    vals[t] -= m * tn(t) + c;
                }
            }
        }
    }

    // Assemble [b, v, total] values and masks: context (masked -> 0), then the
    // horizon (targets and past-only: unknown, masked; past+future: given).
    let mut values = vec![0.0f32; checked_mul(&[b, v, total], "values")?];
    let mut masks = vec![1u8; values.len()];
    for bi in 0..b {
        for vi in 0..v {
            let r = bi * v + vi;
            let dst = r * total;
            for t in 0..context {
                let m = ctx_masks[r * context + t];
                masks[dst + t] = m;
                values[dst + t] = if m != 0 { 0.0 } else { ctx_vals[r * context + t] };
            }
            if vi >= u + po {
                let s = (bi * w + (vi - u - po)) * pf_len;
                let x = inp.past_future.as_ref().expect("w > 0");
                for h in 0..horizon {
                    let mut y = x[s + ctx0 + h];
                    let m = inp.past_future_mask.as_ref().map_or(0, |mm| (mm[s + ctx0 + h] != 0) as u8);
                    if cfg.use_linear_detrending && detrended[r] != 0 {
                        y -= trend_m[r] * ((h + 1) as f32 / cf) + trend_c[r];
                    }
                    masks[dst + context + h] = m;
                    values[dst + context + h] = if m != 0 { 0.0 } else { y };
                }
            }
        }
    }
    let mut patch_is_target = vec![0u8; b * v * n];
    for bi in 0..b {
        for vi in 0..u + po {
            patch_is_target[(bi * v + vi) * n..(bi * v + vi + 1) * n].fill(1);
        }
    }
    let mut cpm = vec![0u8; b * n];
    for bi in 0..b {
        cpm[bi * n + ncp..(bi + 1) * n].fill(1);
    }
    Ok(Prepared {
        batch: b,
        variates: v,
        patches: n,
        context_patches: ncp,
        context,
        horizon,
        values,
        masks,
        patch_is_target,
        cpm,
        trend_m,
        trend_c,
        detrended,
    })
}

/// Upstream `forward()`'s preprocessing (`_preprocess` and the effective
/// patch mask), from a [`Prepared`] batch.
pub fn preprocess(cfg: &HostConfig, pr: &Prepared) -> Result<Preprocessed, String> {
    cfg.validate()?;
    let (b, v, n, p) = (pr.batch, pr.variates, pr.patches, cfg.input_patch_len);
    let o = cfg.output_patch_len;
    let rolls = cfg.rolls();
    let rows = checked_mul(&[b, v, n], "rows")?;
    check_len("values", pr.values.len(), rows * p)?;
    check_len("masks", pr.masks.len(), rows * p)?;
    check_len("patch_is_target", pr.patch_is_target.len(), rows)?;
    check_len("cpm", pr.cpm.len(), b * n)?;
    // forward(): nan_to_num (NaN -> 0, +-inf -> +-f32::MAX), then clip.
    let clip = cfg.value_clip;
    let values: Vec<f32> = pr
        .values
        .iter()
        .map(|&x| {
            let x = if x.is_nan() {
                0.0
            } else if x == f32::INFINITY {
                f32::MAX
            } else if x == f32::NEG_INFINITY {
                f32::MIN
            } else {
                x
            };
            x.clamp(-clip, clip)
        })
        .collect();

    let mut running_n = vec![0.0f32; rows];
    let mut running_mean = vec![0.0f32; rows];
    let mut running_std = vec![0.0f32; rows];
    for r in 0..b * v {
        let (mut cn, mut cm, mut cs) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let off = (r * n + i) * p;
            (cn, cm, cs) = update_running_stats(cn, cm, cs, &values[off..off + p], &pr.masks[off..off + p]);
            running_n[r * n + i] = cn;
            running_mean[r * n + i] = cm;
            running_std[r * n + i] = cs;
        }
    }

    // CPM: target variates at horizon patches are masked (all points).
    let mut masks = pr.masks.clone();
    for bi in 0..b {
        for vi in 0..v {
            for i in 0..n {
                if pr.cpm[bi * n + i] != 0 && pr.patch_is_target[(bi * v + vi) * n + i] != 0 {
                    let off = ((bi * v + vi) * n + i) * p;
                    masks[off..off + p].fill(1);
                }
            }
        }
    }

    let width = cfg.resblock_in();
    let half = p + o;
    let mut resblock_input = vec![0.0f32; checked_mul(&[rows, width], "resblock_input")?];
    let mut leading_masked = vec![0u32; b * v];
    for r in 0..b * v {
        let mut leading = true;
        for i in 0..n {
            let row = r * n + i;
            let (mu, sd) = (running_mean[row], safe_sigma(running_std[row]));
            let is_target = pr.patch_is_target[row] != 0;
            let dst = &mut resblock_input[row * width..(row + 1) * width];
            let mut all_masked = true;
            // The patch itself.
            for j in 0..p {
                let m = masks[row * p + j];
                dst[j] = if m != 0 { 0.0 } else { (values[row * p + j] - mu) / sd };
                dst[half + j] = m as f32;
                all_masked &= m != 0;
            }
            // The next `rolls` patches (wrapping, with the wrap masked) as
            // future covariates, normalised by this patch's statistics.
            for k in 0..rolls {
                let src_patch = i + 1 + k;
                let wrap = src_patch >= n;
                let src = r * n + src_patch % n;
                for j in 0..p {
                    let m = masks[src * p + j] != 0 || is_target || wrap;
                    let col = p + k * p + j;
                    dst[col] = if m { 0.0 } else { (values[src * p + j] - mu) / sd };
                    dst[half + col] = m as u8 as f32;
                    all_masked &= m;
                }
            }
            if leading && all_masked {
                leading_masked[r] += 1;
            } else {
                leading = false;
            }
        }
    }
    Ok(Preprocessed {
        resblock_input,
        running_n,
        running_mean,
        running_std,
        leading_masked,
    })
}

/// CPM iterative RevIN refinement (upstream `cpm_iterative_revin_refine`):
/// at each horizon patch, statistics that also fold in the model's median
/// estimate of every earlier horizon patch. Returns the refined `(mean,
/// std)`, `[b, v, n]`, equal to the inputs outside the CPM region.
pub fn cpm_refine(
    cfg: &HostConfig,
    pr: &Prepared,
    pre: &Preprocessed,
    raw_logits: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (b, v, n, p, q) = (pr.batch, pr.variates, pr.patches, cfg.input_patch_len, cfg.num_quantiles);
    let rolls = cfg.rolls();
    let width = cfg.head_out();
    check_len("raw logits", raw_logits.len(), checked_mul(&[b, v, n, width], "logits")?)?;
    let median = q / 2;
    let no_mask = vec![0u8; p];
    let mut mu_out = pre.running_mean.clone();
    let mut sd_out = pre.running_std.clone();
    for bi in 0..b {
        for vi in 0..v {
            let r = bi * v + vi;
            let (mut cn, mut cm, mut cs) = (0.0f32, 0.0f32, 0.0f32);
            let mut anchor = vec![0.0f32; rolls * p];
            let mut offset = 0usize;
            for i in 0..n {
                let row = r * n + i;
                let is_cpm = pr.cpm[bi * n + i] != 0;
                let step = &anchor[offset * p..(offset + 1) * p];
                let (nn, nm, ns) = update_running_stats(cn, cm, cs, step, &no_mask);
                let (on, om, os) = if is_cpm {
                    (nn, nm, ns)
                } else {
                    (pre.running_n[row], pre.running_mean[row], pre.running_std[row])
                };
                let new_offset = if is_cpm { (offset + 1) % rolls } else { 0 };
                if new_offset == 0 {
                    let sd = safe_sigma(os);
                    let lg = &raw_logits[row * width..(row + 1) * width];
                    for k in 0..rolls * p {
                        anchor[k] = (lg[k * q + median] * sd + om).clamp(-cfg.value_clip, cfg.value_clip);
                    }
                }
                (cn, cm, cs) = (on, om, os);
                offset = new_offset;
                if is_cpm {
                    mu_out[row] = om;
                    sd_out[row] = os;
                }
            }
        }
    }
    Ok((mu_out, sd_out))
}

/// Upstream `forward()` after the output head plus `decode()` after
/// `forward()`: RevIN refinement, de-normalisation, clipping, stitching and
/// the trend. `raw_logits` is the head's output, `[b, v, n, o * q]`.
/// Returns `[b, v, horizon, q]`.
pub fn postprocess(cfg: &HostConfig, pr: &Prepared, pre: &Preprocessed, raw_logits: &[f32]) -> Result<Vec<f32>, String> {
    let (b, v, n, p, q) = (pr.batch, pr.variates, pr.patches, cfg.input_patch_len, cfg.num_quantiles);
    let (o, width, horizon) = (cfg.output_patch_len, cfg.head_out(), pr.horizon);
    let (mu, sd) = if cfg.use_iterative_cpm_revin {
        cpm_refine(cfg, pr, pre, raw_logits)?
    } else {
        check_len("raw logits", raw_logits.len(), checked_mul(&[b, v, n, width], "logits")?)?;
        (pre.running_mean.clone(), pre.running_std.clone())
    };
    let logit = |row: usize, k: usize, qi: usize| -> f32 {
        let s = safe_sigma(sd[row]);
        (raw_logits[row * width + k * q + qi] * s + mu[row]).clamp(-cfg.value_clip, cfg.value_clip)
    };
    let ncp = pr.context_patches;
    let mut out = vec![0.0f32; checked_mul(&[b, v, horizon, q], "forecast")?];
    for r in 0..b * v {
        let base = r * n;
        let dst = &mut out[r * horizon * q..(r + 1) * horizon * q];
        if cfg.use_stitching {
            let extract = (2 * p).min(o);
            let overlap = extract - p;
            let nfp = horizon.saturating_sub(overlap).div_ceil(p).max(1);
            // torch.linspace(1, 0, overlap) in f32.
            let weights: Vec<f32> = linspace_desc(overlap);
            for t in 0..horizon {
                // Position t of the stitched series [nfp * p + overlap].
                let (patch, k) = (t / p, t % p);
                for qi in 0..q {
                    let val = if nfp == 1 {
                        logit(base + ncp - 1, t, qi)
                    } else if patch == 0 {
                        logit(base + ncp - 1, k, qi)
                    } else if patch < nfp {
                        // Inside the stitched middle: the overlap blends the
                        // previous patch's tail with this patch's head.
                        if k < overlap {
                            let wgt = weights[k];
                            let prev = logit(base + ncp - 1 + patch - 1, p + k, qi);
                            let next = logit(base + ncp - 1 + patch, k, qi);
                            wgt * prev + (1.0 - wgt) * next
                        } else {
                            logit(base + ncp - 1 + patch, k, qi)
                        }
                    } else {
                        // The tail: the last patch's points past p.
                        logit(base + ncp - 1 + nfp - 1, p + (t - nfp * p), qi)
                    };
                    dst[t * q + qi] = val;
                }
            }
        } else {
            let rolls = cfg.rolls();
            for t in 0..horizon {
                let (chunk, k) = (t / o, t % o);
                let row = base + chunk * rolls + ncp - 1;
                for qi in 0..q {
                    dst[t * q + qi] = logit(row, k, qi);
                }
            }
        }
        if cfg.use_linear_detrending && pr.detrended[r] != 0 {
            let cf = pr.context as f32;
            for t in 0..horizon {
                let trend = pr.trend_m[r] * ((t + 1) as f32 / cf) + pr.trend_c[r];
                for qi in 0..q {
                    dst[t * q + qi] += trend;
                }
            }
        }
    }
    Ok(out)
}

/// `torch.linspace(1.0, 0.0, steps)` in f32: torch fills the first half
/// from the start and the second half from the end.
fn linspace_desc(steps: usize) -> Vec<f32> {
    match steps {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let step = (0.0f32 - 1.0) / (steps - 1) as f32;
            let half = steps / 2;
            (0..steps)
                .map(|i| if i < half { 1.0 + step * i as f32 } else { 0.0 - step * (steps - i - 1) as f32 })
                .collect()
        }
    }
}
