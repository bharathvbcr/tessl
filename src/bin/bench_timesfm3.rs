//! End-to-end forecast latency of `tessl::timesfm3` on the real checkpoint:
//! the model's `decode` (`BENCH_WORKLOADS`), the user path
//! `Timesfm3Model::predict_batch` (`BENCH_PREDICT`), and a per-stage split of
//! either (`BENCH_STAGES=1`), and a per-kernel-class GPU profile of the
//! forward (`BENCH_STAGES=kernels`).
//!
//! ```text
//! TIMESFM3_DIR=<google/timesfm-3.0-pytorch snapshot> \
//!   cargo run --release --bin bench_timesfm3
//! ```
//!
//! # Decode workloads
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,`, each `BxVxC:H`: `B`
//! series of `V` target variates, `C` context points, horizon `H` (one
//! `Timesfm3Model::decode` call). The series are synthetic and identical in
//! every lane (`series` below; `bench/timesfm3_mlx.py` and
//! `bench/timesfm3_torch.py` compute the same values), so lanes do the same
//! work: detrending decisions and masks are data-dependent.
//!
//! What is timed: one `decode` from host arrays to host forecast (padding,
//! detrending, RevIN, upload, the forward, read-back, CPM refinement,
//! stitching). Not timed: the model load. `BENCH_ITERS` (default 10) and
//! `BENCH_WARMUP` (default 3) are per workload.
//!
//! # Predict workloads (the user path)
//!
//! `BENCH_PREDICT` lists workloads separated by `,`, each `NxL:H` (`N`
//! univariate series of at most `L` points) or `NxVxL:H` (`N` series of `V`
//! target variates), optionally followed by `+PO+PF`: `PO` past-only and `PF`
//! past+future covariate channels per series (`+0+0` is refused: leave the
//! suffix off). Series are ragged, deterministically: series `i` has
//! `L - (i * 61) % (L / 4 + 1)` points, so `L` is the longest and none is
//! shorter than three quarters of it. Target row `r` of series `i` is
//! `series(i * V + r, t, len_i)`; covariate channel `k` is
//! `series(i + 7 * k + 1000, t, len_i)` (past-only) or
//! `series(i + 7 * k + 2000, t, len_i + H)` (past+future). The Python lanes
//! build the same arrays. The options are upstream's defaults, with
//! `per_core_batch_size` from `BENCH_PREDICT_BATCH` (default 4, upstream's).
//!
//! What is timed: one `predict_batch` over all the series, host arrays in to
//! per-series quantile forecasts out (planning, one `decode` per group of
//! `BENCH_PREDICT_BATCH` series, post-processing). When only `BENCH_PREDICT`
//! is set the decode workloads do not run; set `BENCH_WORKLOADS` as well to run
//! both, and neither for the default decode set.
//!
//! # Stage split
//!
//! `BENCH_STAGES=1` (`0`, the default, or `1`) times, in place of one opaque
//! call, the library's public pieces one after the other in the same
//! iteration: `timesfm3_forecast::plan`, then per `decode` call
//! `timesfm3_host::prepare`, `preprocess`, `Timesfm3Model::forward` (the
//! device: upload, kernels, read-back) and `timesfm3_host::postprocess`, then
//! `timesfm3_forecast::finish`. A decode workload has no plan or finish
//! stage. A stage that repeats over the calls is summed per iteration. The
//! row's `ms_min`/`ms_median` are the wall time of the whole staged
//! iteration, and `stages` lists `{stage, ms_min, ms_median, share}` where
//! `share` is the stage's median over the sum of the stage medians. Before
//! timing, the staged result is compared with the plain call's (elementwise,
//! 1e-4 relative) and the run refuses on a difference: the split must be the
//! same work. A stage-mode call runs one forward per `decode` call, so one
//! needing more than `MAX_ROWS` patch rows is refused (lower
//! `BENCH_PREDICT_BATCH`); the plain `predict_batch` splits it.
//!
//! # Kernel profile
//!
//! `BENCH_STAGES=kernels` splits the forward itself. The runtime is built with
//! per-dispatch GPU timestamps (`tessl::dispatch_stamps`), armed only around
//! each `Timesfm3Model::forward`. Every stamped dispatch is aggregated by kernel
//! name and by a class derived from the name alone (`classify`: gemm,
//! attention, qk_prep_rope, norm, softmax, copy_transpose, elementwise, other).
//! A stamp delta brackets one dispatch including its launch, drain and
//! barrier, and stamps serialise dispatches, so the stamped time is an upper
//! bound on the unstamped GPU time. The kernel names do not say which layer
//! kind (sequence attention, variate attention, feed-forward) a GEMM serves and
//! the forward does not tag layers, so the split is by kernel class only.
//!
//! Each iteration runs the staged call twice, stamps off then stamps on, so
//! the overhead of the stamps is measured on the same work: `ms_min` and
//! `ms_median` are the stamps-off wall time, and the row's `kernels` object
//! holds `forward_ms_off`, `forward_ms_on`, `overhead_forward` (on over off),
//! `wall_ms_on`, `overhead_wall`, `dispatches_per_forward`,
//! `commits_per_forward` (waiting commits of the forward's command buffers),
//! `gpu_stamped_ms` (the summed per-kernel median stamp deltas),
//! `host_outside_stamped_ms` (the stamps-on forward wall minus that: host
//! allocation, RoPE tables, upload, encoding, commit latency, read-back),
//! `proxy_upload_ms` and `proxy_readback_ms` (see below), `max_abs_diff` and
//! `bit_equal` (stamped against unstamped forecast), then `classes` (class,
//! dispatches per iteration, ms, share of the stamped time) and `top_kernels`
//! (the ten costliest kernels). Per-kernel times are the median over the
//! iterations, so the shares sum to one over the medians.
//!
//! The run refuses (non-zero exit) when a command buffer overflowed the
//! 1,023-dispatch stamp heap, a batch was dropped, no stamps came back, or the
//! dispatch counts change between iterations: a profile never covers part of
//! the forward. A forward that overflows needs a smaller group (lower
//! `BENCH_PREDICT_BATCH`) or workload.
//!
//! Upload and read-back happen inside `Timesfm3Model::forward`, which the bench
//! cannot instrument, so they are not timed in place.
//! `proxy_upload_ms` (allocate and write a buffer of one forward's input size)
//! and `proxy_readback_ms` (copy one forward's logits out of a buffer) are
//! standalone minima of the same host operations, an estimate of their order
//! of magnitude, not a slice of the forward.
//!
//! `BENCH_PREDICT_SWEEP` (kernels mode only; a list such as `1,4,16,64`) runs
//! every predict workload once per `per_core_batch_size`, so the rows of one
//! workload differ only in the patch rows per forward: forward ms against
//! rows separates the fixed per-forward cost from the per-row cost, e.g.
//! `BENCH_PREDICT=64x512:128 BENCH_PREDICT_SWEEP=1,4,16,64`.
//!
//! # Output
//!
//! One JSON array on stdout, one object per workload: `{workload, backend,
//! ms_min, ms_median, iters, rows, check, peak_footprint_mib}`, where `rows`
//! is `b * v * n` patch rows (summed over the forwards for a predict workload)
//! and `check` the mean of the forecast's median quantile over target variates
//! (lanes should agree to ~1e-3; a mismatch means they did different work). A
//! predict row is named `predict:<spec>` and adds `mode`, `series`,
//! `per_core_batch_size`, `groups` (`decode` calls) and `forwards` (device
//! forwards; more than `groups` only when a group exceeds `MAX_ROWS`). Before
//! anything is timed every forecast must be finite. The check value comes from
//! the licensed checkpoint: it is not a result to commit.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use tessl::dispatch_stamps::{DispatchStampBatch, STAMP_HEAP_SLOTS};
use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Forecast, Timesfm3Config, Timesfm3Model, MAX_ROWS};
use tessl::timesfm3_forecast::{self as forecast, PredictOptions, Rows, Series, SeriesForecast};
use tessl::timesfm3_host::{self as host, DecodeInput};
use tessl::GpuRuntime;

use common::{env_usize, median, parse_count, peak_footprint};

const DEFAULT_WORKLOADS: &str = "1x1x512:128,16x1x2048:128,64x1x512:64,1x8x1024:96,4x4x2048:256";
/// Most input elements (targets and covariates) one predict workload may
/// build: 1 GiB of f32. A typo that adds a digit is an error, not a swap storm.
const MAX_PREDICT_ELEMENTS: usize = 1 << 28;
/// Staged and plain forecasts must agree to this, relative to `1 + |x|`.
const STAGE_AGREE: f32 = 1e-4;

struct Workload {
    name: String,
    b: usize,
    v: usize,
    ctx: usize,
    horizon: usize,
}

fn parse(w: &str) -> Result<Workload, String> {
    let bad = || format!("workload {w:?}: expected BxVxC:H");
    let (shape, h) = w.split_once(':').ok_or_else(bad)?;
    let dims: Vec<usize> = shape
        .split('x')
        .map(|x| x.parse().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    if dims.len() != 3 || dims.contains(&0) {
        return Err(bad());
    }
    let horizon = h.parse().map_err(|_| bad())?;
    if horizon == 0 {
        return Err(bad());
    }
    Ok(Workload {
        name: w.to_string(),
        b: dims[0],
        v: dims[1],
        ctx: dims[2],
        horizon,
    })
}

/// One `BENCH_PREDICT` workload: `n` series of `v` target variates, at most
/// `len` points, horizon `horizon`, `po`/`pf` covariate channels.
#[derive(Debug, PartialEq)]
struct Predict {
    name: String,
    n: usize,
    v: usize,
    len: usize,
    horizon: usize,
    po: usize,
    pf: usize,
}

fn parse_predict(w: &str) -> Result<Predict, String> {
    const VAR: &str = "BENCH_PREDICT";
    let bad = |why: &str| format!("{VAR}: workload {w:?}: {why}; expected NxL:H or NxVxL:H, optionally +PO+PF");
    let mut parts = w.split('+');
    let main = parts.next().unwrap_or_default();
    let cov: Vec<&str> = parts.collect();
    if !cov.is_empty() && cov.len() != 2 {
        return Err(bad("covariates are given as both +PO and +PF"));
    }
    let (shape, h) = main.split_once(':').ok_or_else(|| bad("no :H"))?;
    let dims = shape
        .split('x')
        .map(|x| parse_count(VAR, x, 1))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| bad(&e))?;
    let (n, v, len) = match dims[..] {
        [n, len] => (n, 1, len),
        [n, v, len] => (n, v, len),
        _ => return Err(bad("the shape has two or three dimensions")),
    };
    let horizon = parse_count(VAR, h, 1).map_err(|e| bad(&e))?;
    let (po, pf) = match cov[..] {
        [] => (0, 0),
        [po, pf] => (
            parse_count(VAR, po, 0).map_err(|e| bad(&e))?,
            parse_count(VAR, pf, 0).map_err(|e| bad(&e))?,
        ),
        _ => unreachable!("cov.len() is 0 or 2"),
    };
    if !cov.is_empty() && po == 0 && pf == 0 {
        return Err(bad("+0+0 has no covariates; leave the suffix off"));
    }
    let channels = v.checked_add(po).and_then(|c| c.checked_add(pf));
    let elements = channels
        .and_then(|c| c.checked_mul(n))
        .and_then(|c| c.checked_mul(len.checked_add(horizon)?));
    if !elements.is_some_and(|e| e <= MAX_PREDICT_ELEMENTS) {
        return Err(bad(&format!("more than {MAX_PREDICT_ELEMENTS} input elements")));
    }
    Ok(Predict {
        name: format!("predict:{w}"),
        n,
        v,
        len,
        horizon,
        po,
        pf,
    })
}

/// A `BENCH_STAGES` flag value: `0` or `1`, nothing else.
fn parse_flag(name: &str, raw: &str) -> Result<bool, String> {
    match raw {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{name}={raw:?} must be 0 or 1")),
    }
}

/// What `BENCH_STAGES` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stages {
    /// One opaque call per iteration.
    Off,
    /// The host and forward stage split.
    Split,
    /// The per-kernel-class GPU profile.
    Kernels,
}

/// `BENCH_STAGES`: `0`, `1` or `kernels`, nothing else.
fn parse_stages(raw: &str) -> Result<Stages, String> {
    if raw == "kernels" {
        return Ok(Stages::Kernels);
    }
    match parse_flag("BENCH_STAGES", raw) {
        Ok(on) => Ok(if on { Stages::Split } else { Stages::Off }),
        Err(_) => Err(format!("BENCH_STAGES={raw:?} must be 0, 1 or kernels")),
    }
}

fn env_stages() -> Result<Stages, String> {
    match std::env::var("BENCH_STAGES") {
        Ok(v) => parse_stages(&v),
        Err(std::env::VarError::NotPresent) => Ok(Stages::Off),
        Err(e) => Err(format!("BENCH_STAGES: {e}")),
    }
}

/// `BENCH_PREDICT_SWEEP`: comma-separated `per_core_batch_size` values, each a
/// count of at least one, no repeats.
fn parse_sweep(raw: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let v = parse_count("BENCH_PREDICT_SWEEP", part, 1)?;
        if out.contains(&v) {
            return Err(format!("BENCH_PREDICT_SWEEP lists {v} twice"));
        }
        out.push(v);
    }
    Ok(out)
}

fn env_opt(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

/// Which workloads run: the decode set (`BENCH_WORKLOADS`, else the default
/// unless `BENCH_PREDICT` is set) and the predict set.
fn select(workloads: Option<&str>, predict: Option<&str>) -> Result<(Vec<Workload>, Vec<Predict>), String> {
    let decode = match (workloads, predict) {
        (None, Some(_)) => Vec::new(),
        (w, _) => w
            .unwrap_or(DEFAULT_WORKLOADS)
            .split(',')
            .map(parse)
            .collect::<Result<_, _>>()?,
    };
    let predict = match predict {
        Some(p) => p.split(',').map(parse_predict).collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    Ok((decode, predict))
}

/// Series `i` of a workload, point `t`: a sine whose period depends on `i`,
/// a slope on every third series (so detrending fires on some), an offset.
/// Computed in f64 and rounded once, as the Python lanes do.
fn series(i: usize, t: usize, ctx: usize) -> f32 {
    let period = 7.0 + (i % 29) as f64;
    let slope = if i.is_multiple_of(3) {
        3.0 * t as f64 / ctx as f64
    } else {
        0.0
    };
    ((2.0 * std::f64::consts::PI * t as f64 / period).sin() + slope + (i % 5) as f64) as f32
}

fn input(w: &Workload) -> DecodeInput {
    let mut target = Vec::with_capacity(w.b * w.v * w.ctx);
    for i in 0..w.b * w.v {
        target.extend((0..w.ctx).map(|t| series(i, t, w.ctx)));
    }
    DecodeInput {
        batch: w.b,
        context: w.ctx,
        horizon: w.horizon,
        num_target: w.v,
        target,
        ..Default::default()
    }
}

/// Points of series `i` of a predict workload: the longest has `len`, the
/// shortest more than three quarters of it.
fn series_len(i: usize, len: usize) -> usize {
    len - (i * 61) % (len / 4 + 1)
}

/// `channels` rows of `series(base + 7 * k, ·, n)`, row-major.
fn channels(base: usize, rows: usize, n: usize) -> Rows {
    let mut values = Vec::with_capacity(rows * n);
    for k in 0..rows {
        values.extend((0..n).map(|t| series(base + 7 * k, t, n)));
    }
    Rows { rows, values }
}

fn predict_series(p: &Predict) -> Vec<Series> {
    (0..p.n)
        .map(|i| {
            let len = series_len(i, p.len);
            let mut target = Vec::with_capacity(p.v * len);
            for r in 0..p.v {
                target.extend((0..len).map(|t| series(i * p.v + r, t, len)));
            }
            Series {
                target: Rows {
                    rows: p.v,
                    values: target,
                },
                past_only: (p.po > 0).then(|| channels(i + 1000, p.po, len)),
                past_future: (p.pf > 0).then(|| channels(i + 2000, p.pf, len + p.horizon)),
            }
        })
        .collect()
}

/// Per-stage wall time of one iteration, in the order each stage first ran;
/// a stage that runs once per `decode` call is summed.
type StageMs = Vec<(&'static str, f64)>;

fn stage<T>(acc: &mut StageMs, name: &'static str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let t0 = Instant::now();
    let v = f()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    match acc.iter_mut().find(|(n, _)| *n == name) {
        Some(e) => e.1 += ms,
        None => acc.push((name, ms)),
    }
    Ok(v)
}

/// `Timesfm3Model::decode`, from its public pieces (the same calls in the same
/// order), each timed into `acc`; the forward goes through `probe` when there
/// is one. Refuses more than `MAX_ROWS` patch rows.
fn staged_decode(
    model: &Timesfm3Model,
    inp: &DecodeInput,
    acc: &mut StageMs,
    probe: Option<&mut Probe<'_>>,
) -> Result<Forecast, String> {
    let hc = &model.config().host;
    let pr = stage(acc, "prepare", || host::prepare(hc, inp))?;
    let pre = stage(acc, "preprocess", || host::preprocess(hc, &pr))?;
    let rows = pr.batch * pr.variates * pr.patches;
    if rows > MAX_ROWS {
        return Err(format!(
            "BENCH_STAGES: one decode call has {rows} patch rows (batch {}, {} variates, {} patches), above MAX_ROWS = \
             {MAX_ROWS}; predict_batch would split it, the stage split does not: lower BENCH_PREDICT_BATCH",
            pr.batch, pr.variates, pr.patches
        ));
    }
    let fwd = stage(acc, "forward", || {
        let run = || {
            model.forward(
                pr.batch,
                pr.variates,
                pr.patches,
                &pre.resblock_input,
                &pre.leading_masked,
                false,
            )
        };
        match probe {
            Some(p) => p.forward(run),
            None => run(),
        }
    })?;
    let values = stage(acc, "postprocess", || host::postprocess(hc, &pr, &pre, &fwd.raw_logits))?;
    Ok(Forecast {
        batch: pr.batch,
        variates: pr.variates,
        horizon: pr.horizon,
        quantiles: hc.num_quantiles,
        values,
    })
}

/// `Timesfm3Model::predict_batch`, from its public pieces, each timed into
/// `acc`.
fn staged_predict(
    model: &Timesfm3Model,
    series: &[Series],
    horizon: usize,
    opts: &PredictOptions,
    acc: &mut StageMs,
    mut probe: Option<&mut Probe<'_>>,
) -> Result<Vec<SeriesForecast>, String> {
    let plan = stage(acc, "plan", || {
        forecast::plan(&model.config().host, series, horizon, opts)
    })?;
    let mut outs = Vec::with_capacity(plan.calls.len());
    for call in &plan.calls {
        outs.push(staged_decode(model, call, acc, probe.as_deref_mut())?);
    }
    stage(acc, "finish", || forecast::finish(&plan, &outs))
}

/// Refuses two forecasts that differ by more than [`STAGE_AGREE`], or in
/// length: the stage split must do the plain call's work.
fn agree(what: &str, staged: &[f32], plain: &[f32]) -> Result<(), String> {
    if staged.len() != plain.len() {
        return Err(format!(
            "{what}: staged forecast has {} values, plain {}",
            staged.len(),
            plain.len()
        ));
    }
    for (i, (a, b)) in staged.iter().zip(plain).enumerate() {
        // A NaN on either side fails here, which is why this is not `>`.
        let within = (a - b).abs() <= STAGE_AGREE * (1.0 + b.abs());
        if !within {
            return Err(format!(
                "{what}: staged and plain forecasts differ at element {i}: {a} vs {b}"
            ));
        }
    }
    Ok(())
}

/// One device forward as the probe saw it: the host wall time of
/// `Timesfm3Model::forward` and, when stamping, the dispatch stamps it left.
struct ForwardRecord {
    host_ms: f64,
    batch: Option<DispatchStampBatch>,
}

/// Wraps each `Timesfm3Model::forward` of a staged call. With `stamps` it arms
/// the per-dispatch timestamps for exactly that forward and takes the batch
/// afterwards; without, it only times the forward and checks that no stamps
/// leaked.
struct Probe<'a> {
    rt: &'a GpuRuntime,
    stamps: bool,
    records: Vec<ForwardRecord>,
}

impl Probe<'_> {
    fn forward<T>(&mut self, run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        if self.stamps {
            drop(self.rt.take_dispatch_stamps());
            tessl::ab_flags::set_dispatch_stamps(true);
        }
        let t0 = Instant::now();
        let out = run();
        let host_ms = t0.elapsed().as_secs_f64() * 1e3;
        if self.stamps {
            tessl::ab_flags::set_dispatch_stamps(false);
        }
        let out = out?;
        let batch = match (self.stamps, self.rt.take_dispatch_stamps()) {
            (true, Some(Ok(b))) => Some(b),
            (true, Some(Err(e))) => {
                return Err(format!("BENCH_STAGES=kernels: the stamps could not be read back: {e}"))
            }
            (true, None) => {
                return Err(
                    "BENCH_STAGES=kernels: no stamps came back from a forward (was the runtime built with \
                            stamps on, and did the forward reach a waiting commit?)"
                        .to_string(),
                )
            }
            (false, None) => None,
            (false, Some(_)) => {
                return Err("BENCH_STAGES=kernels: a forward with stamps off produced stamps".to_string())
            }
        };
        self.records.push(ForwardRecord { host_ms, batch });
        Ok(out)
    }
}

/// Refuses a stamp batch that does not cover the whole forward.
fn validate_batch(b: &DispatchStampBatch) -> Result<(), String> {
    if b.overflowed {
        return Err(format!(
            "BENCH_STAGES=kernels: a command buffer overflowed the stamp heap ({} stamps kept, at most {} fit per \
             command buffer): the profile would cover part of the forward; lower BENCH_PREDICT_BATCH or use a \
             smaller workload",
            b.stamps.len(),
            STAMP_HEAP_SLOTS - 1
        ));
    }
    if b.dropped_batches > 0 {
        return Err(format!(
            "BENCH_STAGES=kernels: {} armed command buffer(s) ended without a waiting commit and their stamps were \
             dropped: the profile would cover part of the forward",
            b.dropped_batches
        ));
    }
    if b.stamps.is_empty() || b.command_buffers == 0 {
        return Err("BENCH_STAGES=kernels: the stamp batch holds no dispatches".to_string());
    }
    Ok(())
}

/// The kernel class of a dispatch, from its name alone. The first rule that
/// matches wins, so a fused `rms_norm_residual_add` is a norm, not an add.
fn classify(kernel: &str) -> &'static str {
    const RULES: &[(&str, &[&str])] = &[
        ("qk_prep_rope", &["qk_prep", "rope"]),
        ("attention", &["attn", "attention"]),
        ("norm", &["norm"]),
        ("softmax", &["softmax"]),
        ("gemm", &["gemm", "matmul", "gemv"]),
        ("copy_transpose", &["copy", "transpose", "zero_", "fill"]),
        (
            "elementwise",
            &["add_", "scale_", "mul_", "relu", "gelu", "silu", "softcap", "mlp_"],
        ),
    ];
    RULES
        .iter()
        .find(|(_, needles)| needles.iter().any(|n| kernel.contains(n)))
        .map_or("other", |(class, _)| class)
}

/// One iteration's stamped dispatches over all its forwards: per kernel name
/// the dispatch count and summed stamp deltas (ns).
#[derive(Clone, Debug, PartialEq)]
struct Profile {
    kernels: BTreeMap<String, (usize, u64)>,
    forwards: usize,
    commits: usize,
}

fn profile_of(records: &[ForwardRecord]) -> Result<Profile, String> {
    if records.is_empty() {
        return Err("BENCH_STAGES=kernels: the iteration ran no forward".to_string());
    }
    let mut p = Profile {
        kernels: BTreeMap::new(),
        forwards: records.len(),
        commits: 0,
    };
    for r in records {
        let b = r
            .batch
            .as_ref()
            .ok_or("BENCH_STAGES=kernels: a forward has no stamp batch")?;
        validate_batch(b)?;
        p.commits += b.command_buffers as usize;
        for s in &b.stamps {
            let e = p.kernels.entry(s.name.clone()).or_default();
            e.0 += 1;
            e.1 += s.dt_ns;
        }
    }
    Ok(p)
}

#[derive(Debug)]
struct KernelRow {
    name: String,
    class: &'static str,
    dispatches: usize,
    ms: f64,
    share: f64,
}

#[derive(Debug)]
struct ClassRow {
    class: &'static str,
    dispatches: usize,
    ms: f64,
    share: f64,
}

/// The profile over the iterations: per-kernel medians, by kernel and class.
#[derive(Debug)]
struct Summary {
    /// Costliest first.
    kernels: Vec<KernelRow>,
    /// Costliest first.
    classes: Vec<ClassRow>,
    forwards: usize,
    commits: usize,
    dispatches: usize,
    /// Sum of the per-kernel medians.
    stamped_ms: f64,
}

fn summarize(profiles: &[Profile]) -> Result<Summary, String> {
    let first = profiles.first().ok_or("BENCH_STAGES=kernels: no profiled iteration")?;
    for p in profiles {
        let same_counts = p.kernels.len() == first.kernels.len()
            && p.kernels
                .iter()
                .zip(&first.kernels)
                .all(|((na, (ca, _)), (nb, (cb, _)))| na == nb && ca == cb);
        if p.forwards != first.forwards || p.commits != first.commits || !same_counts {
            return Err("BENCH_STAGES=kernels: the dispatches or commits changed between iterations".to_string());
        }
    }
    let mut kernels = Vec::new();
    for (name, (count, _)) in &first.kernels {
        let ns: Vec<f64> = profiles.iter().map(|p| p.kernels[name].1 as f64).collect();
        kernels.push(KernelRow {
            name: name.clone(),
            class: classify(name),
            dispatches: *count,
            ms: median(ns)? / 1e6,
            share: 0.0,
        });
    }
    let stamped_ms: f64 = kernels.iter().map(|k| k.ms).sum();
    if stamped_ms.is_nan() || stamped_ms <= 0.0 {
        return Err("BENCH_STAGES=kernels: the stamped time sums to zero".to_string());
    }
    let mut by_class: BTreeMap<&'static str, (usize, f64)> = BTreeMap::new();
    for k in &mut kernels {
        k.share = k.ms / stamped_ms;
        let e = by_class.entry(k.class).or_default();
        e.0 += k.dispatches;
        e.1 += k.ms;
    }
    let mut classes: Vec<ClassRow> = by_class
        .into_iter()
        .map(|(class, (dispatches, ms))| ClassRow {
            class,
            dispatches,
            ms,
            share: ms / stamped_ms,
        })
        .collect();
    classes.sort_by(|a, b| b.ms.total_cmp(&a.ms).then(a.class.cmp(b.class)));
    kernels.sort_by(|a, b| b.ms.total_cmp(&a.ms).then(a.name.cmp(&b.name)));
    Ok(Summary {
        dispatches: first.kernels.values().map(|v| v.0).sum(),
        forwards: first.forwards,
        commits: first.commits,
        kernels,
        classes,
        stamped_ms,
    })
}

/// Everything a kernel-profile row needs from the timed iterations.
struct KernelRun {
    off_wall: Vec<f64>,
    on_wall: Vec<f64>,
    off_forward: Vec<f64>,
    on_forward: Vec<f64>,
    profiles: Vec<Profile>,
    max_abs_diff: f32,
    bit_equal: bool,
}

/// Largest absolute difference between two equal-length forecasts, and
/// whether they are bit-identical. A NaN anywhere makes the difference
/// infinite.
fn forecast_diff(a: &[f32], b: &[f32]) -> (f32, bool) {
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max = if d.is_nan() { f32::INFINITY } else { max.max(d) };
    }
    (max, a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()))
}

/// One staged call with stamps on or off: its forecast values, wall ms and
/// forward records.
fn kernel_arm(
    rt: &GpuRuntime,
    stamps: bool,
    run: &mut impl FnMut(Option<&mut Probe<'_>>) -> Result<Vec<f32>, String>,
) -> Result<(Vec<f32>, f64, Vec<ForwardRecord>), String> {
    let mut probe = Probe {
        rt,
        stamps,
        records: Vec::new(),
    };
    let t0 = Instant::now();
    let values = run(Some(&mut probe))?;
    Ok((values, t0.elapsed().as_secs_f64() * 1e3, probe.records))
}

/// `warmup` untimed then `iters` timed iterations, each a stamps-off then a
/// stamps-on staged call. The stamped forecast must match the unstamped one
/// and every stamped forward must be fully covered ([`validate_batch`]).
fn run_kernels(
    rt: &GpuRuntime,
    what: &str,
    warmup: usize,
    iters: usize,
    mut run: impl FnMut(Option<&mut Probe<'_>>) -> Result<Vec<f32>, String>,
) -> Result<KernelRun, String> {
    let (off, _, _) = kernel_arm(rt, false, &mut run)?;
    let (on, _, _) = kernel_arm(rt, true, &mut run)?;
    agree(&format!("{what} (stamps on against off)"), &on, &off)?;
    for _ in 0..warmup {
        kernel_arm(rt, false, &mut run)?;
        kernel_arm(rt, true, &mut run)?;
    }
    let mut k = KernelRun {
        off_wall: Vec::with_capacity(iters),
        on_wall: Vec::with_capacity(iters),
        off_forward: Vec::with_capacity(iters),
        on_forward: Vec::with_capacity(iters),
        profiles: Vec::with_capacity(iters),
        max_abs_diff: 0.0,
        bit_equal: true,
    };
    for _ in 0..iters {
        let (off, off_wall, off_rec) = kernel_arm(rt, false, &mut run)?;
        let (on, on_wall, on_rec) = kernel_arm(rt, true, &mut run)?;
        agree(&format!("{what} (stamps on against off)"), &on, &off)?;
        let (d, bit) = forecast_diff(&on, &off);
        k.max_abs_diff = k.max_abs_diff.max(d);
        k.bit_equal &= bit;
        k.off_wall.push(off_wall);
        k.on_wall.push(on_wall);
        k.off_forward.push(off_rec.iter().map(|r| r.host_ms).sum());
        k.on_forward.push(on_rec.iter().map(|r| r.host_ms).sum());
        k.profiles.push(profile_of(&on_rec)?);
    }
    Ok(k)
}

/// Minimum of five standalone runs of the host-side upload (allocate a buffer
/// and write `up` floats) and read-back (copy `down` floats out of a buffer):
/// estimates of what `Timesfm3Model::forward` does inside, which the bench
/// cannot time in place.
fn proxy_host_ms(rt: &GpuRuntime, up: usize, down: usize) -> Result<(f64, f64), String> {
    let data = vec![0.5f32; up.max(1)];
    let src = rt.alloc_buffer(down.max(1) * 4)?;
    src.try_write_f32(&vec![0.5f32; down.max(1)])?;
    let (mut upload, mut readback) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        let t0 = Instant::now();
        let b = rt.alloc_buffer(data.len() * 4)?;
        b.try_write_f32(&data)?;
        upload = upload.min(t0.elapsed().as_secs_f64() * 1e3);
        let t0 = Instant::now();
        let copy = src.try_contents_f32()?[..down.max(1)].to_vec();
        readback = readback.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(copy);
    }
    Ok((upload, readback))
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `"kernels":{...}` for a row, with the row's stamps-off wall min and median.
fn kernels_json(k: &KernelRun, proxy: (f64, f64), rows_per_forward: usize) -> Result<(f64, f64, String), String> {
    let s = summarize(&k.profiles)?;
    let mut off_wall = k.off_wall.clone();
    off_wall.sort_by(f64::total_cmp);
    let (off_fwd, on_fwd) = (median(k.off_forward.clone())?, median(k.on_forward.clone())?);
    let (off_w, on_w) = (median(k.off_wall.clone())?, median(k.on_wall.clone())?);
    if off_fwd.is_nan() || off_w.is_nan() || off_fwd <= 0.0 || off_w <= 0.0 {
        return Err("BENCH_STAGES=kernels: a stamps-off time is zero".to_string());
    }
    let classes: Vec<String> = s
        .classes
        .iter()
        .map(|c| {
            format!(
                "{{\"class\":{},\"dispatches\":{},\"ms\":{:.4},\"share\":{:.4}}}",
                json_str(c.class),
                c.dispatches,
                c.ms,
                c.share
            )
        })
        .collect();
    let top: Vec<String> = s
        .kernels
        .iter()
        .take(10)
        .map(|c| {
            format!(
                "{{\"kernel\":{},\"class\":{},\"dispatches\":{},\"ms\":{:.4},\"share\":{:.4}}}",
                json_str(&c.name),
                json_str(c.class),
                c.dispatches,
                c.ms,
                c.share
            )
        })
        .collect();
    let json = format!(
        ",\"kernels\":{{\"forward_ms_off\":{off_fwd:.4},\"forward_ms_on\":{on_fwd:.4},\"overhead_forward\":{:.4},\"wall_ms_on\":{on_w:.4},\"overhead_wall\":{:.4},\"rows_per_forward\":{rows_per_forward},\"dispatches_per_forward\":{:.2},\"commits_per_forward\":{:.2},\"gpu_stamped_ms\":{:.4},\"host_outside_stamped_ms\":{:.4},\"proxy_upload_ms\":{:.4},\"proxy_readback_ms\":{:.4},\"max_abs_diff\":{:e},\"bit_equal\":{},\"classes\":[{}],\"top_kernels\":[{}]}}",
        on_fwd / off_fwd,
        on_w / off_w,
        s.dispatches as f64 / s.forwards as f64,
        s.commits as f64 / s.forwards as f64,
        s.stamped_ms,
        on_fwd - s.stamped_ms,
        proxy.0,
        proxy.1,
        k.max_abs_diff,
        k.bit_equal,
        classes.join(","),
        top.join(",")
    );
    Ok((off_wall[0], median(off_wall)?, json))
}

/// The kernel-profile row tail for one workload: runs it and renders it.
fn kernel_profile(
    rt: &GpuRuntime,
    model: &Timesfm3Model,
    what: &str,
    (warmup, iters): (usize, usize),
    rows_per_forward: usize,
    run: impl FnMut(Option<&mut Probe<'_>>) -> Result<Vec<f32>, String>,
) -> Result<(f64, f64, String), String> {
    let k = run_kernels(rt, what, warmup, iters, run)?;
    let host = &model.config().host;
    let proxy = proxy_host_ms(
        rt,
        rows_per_forward * host.resblock_in(),
        rows_per_forward * host.head_out(),
    )?;
    kernels_json(&k, proxy, rows_per_forward)
}

/// Device forwards `decode_capped` runs for a call of `batch` series of
/// `per` patch rows each, at most `cap` rows a forward.
fn forwards_for(batch: usize, per: usize, cap: usize) -> Result<usize, String> {
    if per > cap {
        return Err(format!(
            "one series needs {per} patch rows, more than one forward runs ({cap})"
        ));
    }
    let chunk = (cap / per.max(1)).max(1);
    Ok(batch.div_ceil(chunk))
}

struct Timing {
    wall_ms: Vec<f64>,
    /// `[stage][iteration]`, empty without `BENCH_STAGES`.
    stages: Vec<(&'static str, Vec<f64>)>,
}

/// `warmup` untimed runs, then `iters` timed ones; `run` returns that run's
/// stage times (empty when it does not split).
fn measure(warmup: usize, iters: usize, mut run: impl FnMut() -> Result<StageMs, String>) -> Result<Timing, String> {
    for _ in 0..warmup {
        run()?;
    }
    let mut t = Timing {
        wall_ms: Vec::with_capacity(iters),
        stages: Vec::new(),
    };
    for _ in 0..iters {
        let t0 = Instant::now();
        let st = run()?;
        t.wall_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        if t.stages.is_empty() {
            t.stages = st.iter().map(|(n, _)| (*n, Vec::with_capacity(iters))).collect();
        }
        if st.len() != t.stages.len() || st.iter().zip(&t.stages).any(|((a, _), (b, _))| a != b) {
            return Err("the stages of an iteration changed between iterations".to_string());
        }
        for ((_, ms), (_, samples)) in st.iter().zip(&mut t.stages) {
            samples.push(*ms);
        }
    }
    Ok(t)
}

/// `"stages":[...]`: each stage's min, median and share of the sum of medians.
fn stages_json(stages: &[(&'static str, Vec<f64>)]) -> Result<String, String> {
    let mut stats = Vec::new();
    for (name, samples) in stages {
        let mut s = samples.clone();
        s.sort_by(f64::total_cmp);
        stats.push((*name, *s.first().ok_or("no samples")?, median(s)?));
    }
    let total: f64 = stats.iter().map(|s| s.2).sum();
    if total <= 0.0 {
        return Err("the stage medians sum to zero".to_string());
    }
    let items: Vec<String> = stats
        .iter()
        .map(|(n, lo, med)| {
            format!(
                "{{\"stage\":\"{n}\",\"ms_min\":{lo:.4},\"ms_median\":{med:.4},\"share\":{:.4}}}",
                med / total
            )
        })
        .collect();
    Ok(format!("[{}]", items.join(",")))
}

/// Common tail of a row: timing fields, then `stages` when split.
fn timing_json(t: &mut Timing) -> Result<(f64, f64, String), String> {
    t.wall_ms.sort_by(f64::total_cmp);
    let med = median(t.wall_ms.clone())?;
    let stages = if t.stages.is_empty() {
        String::new()
    } else {
        format!(",\"stages\":{}", stages_json(&t.stages)?)
    };
    Ok((t.wall_ms[0], med, stages))
}

fn main() -> Result<(), String> {
    let dir = PathBuf::from(std::env::var_os("TIMESFM3_DIR").ok_or("set TIMESFM3_DIR to the checkpoint directory")?);
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
    let per_core = env_usize("BENCH_PREDICT_BATCH", PredictOptions::default().per_core_batch_size, 1)?;
    let stages = env_stages()?;
    let staged = stages != Stages::Off;
    let sweep = env_opt("BENCH_PREDICT_SWEEP")?
        .map(|raw| parse_sweep(&raw))
        .transpose()?;
    let (workloads, predicts) = select(
        env_opt("BENCH_WORKLOADS")?.as_deref(),
        env_opt("BENCH_PREDICT")?.as_deref(),
    )?;
    if sweep.is_some() && (stages != Stages::Kernels || predicts.is_empty()) {
        return Err("BENCH_PREDICT_SWEEP needs BENCH_STAGES=kernels and BENCH_PREDICT".to_string());
    }
    // The stamp heap exists only if the runtime is built with stamps on; each
    // forward arms them itself (`Probe::forward`), so they stay off otherwise.
    if stages == Stages::Kernels {
        tessl::ab_flags::set_dispatch_stamps(true);
    }
    let rt = GpuRuntime::new();
    if stages == Stages::Kernels {
        tessl::ab_flags::set_dispatch_stamps(false);
    }
    let rt = rt?;
    if stages == Stages::Kernels && !rt.dispatch_stamp_heap_available() {
        return Err("BENCH_STAGES=kernels: the runtime has no dispatch stamp heap".to_string());
    }
    let cfg = Timesfm3Config::from_config_file(&dir.join("config.json"))?;
    let st = SafeTensors::open(&dir.join("model.safetensors")).map_err(|e| e.to_string())?;
    let model = Timesfm3Model::load(&rt, &st, cfg)?;
    drop(st);
    let q = model.config().host.num_quantiles;
    let mut rows_out = Vec::new();
    for w in &workloads {
        let inp = input(w);
        let first = model.decode(&inp)?;
        if let Some(bad) = first.values.iter().position(|x| !x.is_finite()) {
            return Err(format!("{}: forecast element {bad} is not finite", w.name));
        }
        let (h, med) = (first.horizon, q / 2);
        let mut sum = 0.0f64;
        for bi in 0..w.b {
            for vi in 0..w.v {
                for t in 0..h {
                    sum += f64::from(first.values[((bi * first.variates + vi) * h + t) * q + med]);
                }
            }
        }
        let check = sum / (w.b * w.v * h) as f64;
        if staged {
            let f = staged_decode(&model, &inp, &mut Vec::new(), None)?;
            agree(&w.name, &f.values, &first.values)?;
        }
        let (ms_min, ms_median, tail) = if stages == Stages::Kernels {
            let rows_per_forward = w.b * model.config().host.rows_per_series(w.v, w.ctx, w.horizon);
            kernel_profile(&rt, &model, &w.name, (warmup, iters), rows_per_forward, |probe| {
                Ok(staged_decode(&model, &inp, &mut Vec::new(), probe)?.values)
            })?
        } else {
            let mut t = measure(warmup, iters, || {
                let mut acc = Vec::new();
                if staged {
                    std::hint::black_box(staged_decode(&model, &inp, &mut acc, None)?.values);
                } else {
                    std::hint::black_box(model.decode(&inp)?.values);
                }
                Ok(acc)
            })?;
            timing_json(&mut t)?
        };
        let peak_mib = peak_footprint()? as f64 / (1u64 << 20) as f64;
        rows_out.push(format!(
            "{{\"workload\":\"{}\",\"backend\":\"tessl-f32\",\"ms_min\":{ms_min:.4},\"ms_median\":{ms_median:.4},\"iters\":{},\"rows\":{},\"check\":{:.6},\"peak_footprint_mib\":{:.1}{tail}}}",
            w.name,
            iters,
            w.b * model.config().host.rows_per_series(w.v, w.ctx, w.horizon),
            check,
            peak_mib
        ));
    }
    let pcbs = sweep.unwrap_or_else(|| vec![per_core]);
    for (p, per_core) in predicts.iter().flat_map(|p| pcbs.iter().map(move |&c| (p, c))) {
        let opts = PredictOptions {
            per_core_batch_size: per_core,
            ..PredictOptions::default()
        };
        let ser = predict_series(p);
        let plan = forecast::plan(&model.config().host, &ser, p.horizon, &opts)?;
        let (mut rows, mut forwards) = (0usize, 0usize);
        for call in &plan.calls {
            let per = model
                .config()
                .host
                .rows_per_series(call.variates(), call.context, call.horizon);
            rows += call.batch * per;
            forwards += forwards_for(call.batch, per, MAX_ROWS)?;
        }
        let first = model.predict_batch(&ser, p.horizon, &opts)?;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (i, f) in first.iter().enumerate() {
            let point = f.point();
            if let Some(bad) = f.values.iter().position(|x| !x.is_finite()) {
                return Err(format!("{}: series {i} forecast element {bad} is not finite", p.name));
            }
            sum += point.iter().map(|&x| f64::from(x)).sum::<f64>();
            count += point.len();
        }
        let check = sum / count as f64;
        if staged {
            let f = staged_predict(&model, &ser, p.horizon, &opts, &mut Vec::new(), None)?;
            for (i, (a, b)) in f.iter().zip(&first).enumerate() {
                agree(&format!("{} series {i}", p.name), &a.values, &b.values)?;
            }
        }
        let (ms_min, ms_median, tail) = if stages == Stages::Kernels {
            let label = format!("{} (per_core_batch_size {per_core})", p.name);
            kernel_profile(&rt, &model, &label, (warmup, iters), rows / forwards, |probe| {
                let f = staged_predict(&model, &ser, p.horizon, &opts, &mut Vec::new(), probe)?;
                Ok(f.into_iter().flat_map(|s| s.values).collect())
            })?
        } else {
            let mut t = measure(warmup, iters, || {
                let mut acc = Vec::new();
                if staged {
                    std::hint::black_box(staged_predict(&model, &ser, p.horizon, &opts, &mut acc, None)?);
                } else {
                    std::hint::black_box(model.predict_batch(&ser, p.horizon, &opts)?);
                }
                Ok(acc)
            })?;
            timing_json(&mut t)?
        };
        let peak_mib = peak_footprint()? as f64 / (1u64 << 20) as f64;
        rows_out.push(format!(
            "{{\"workload\":\"{}\",\"mode\":\"predict\",\"backend\":\"tessl-f32\",\"ms_min\":{ms_min:.4},\"ms_median\":{ms_median:.4},\"iters\":{},\"series\":{},\"per_core_batch_size\":{per_core},\"groups\":{},\"forwards\":{forwards},\"rows\":{rows},\"check\":{check:.6},\"peak_footprint_mib\":{peak_mib:.1}{tail}}}",
            p.name,
            iters,
            plan.series(),
            plan.calls.len(),
        ));
    }
    println!("[{}]", rows_out.join(","));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(w: &str) -> Predict {
        parse_predict(w).unwrap_or_else(|e| panic!("{w}: {e}"))
    }

    #[test]
    fn predict_forms_parse() {
        assert_eq!(
            p("8x512:64"),
            Predict {
                name: "predict:8x512:64".into(),
                n: 8,
                v: 1,
                len: 512,
                horizon: 64,
                po: 0,
                pf: 0
            }
        );
        let m = p("3x2x1024:128+1+2");
        assert_eq!((m.n, m.v, m.len, m.horizon, m.po, m.pf), (3, 2, 1024, 128, 1, 2));
        let only_pf = p("2x256:32+0+3");
        assert_eq!((only_pf.v, only_pf.po, only_pf.pf), (1, 0, 3));
    }

    #[test]
    fn predict_refuses_bad_specs() {
        for bad in [
            "",
            "8",
            "8x512",
            "8x512:",
            ":64",
            "x512:64",
            "8x:64",
            "0x512:64",
            "8x0:64",
            "8x512:0",
            "8x0x512:64",
            "8x512:64:1",
            "8x1x2x512:64",
            "8x512:-1",
            "8x512:+1",
            "8x512:6 4",
            " 8x512:64",
            "08x512:64",
            "8x512:064",
            "8x512:64+1",
            "8x512:64+1+2+3",
            "8x512:64+0+0",
            "8x512:64+a+1",
            "8x512:64+1+",
            "8x512:64++1",
            "8x512:64+-1+1",
            "8x512:64+01+1",
            "8X512:64",
            "8x512:6.4",
            "8x512:1e2",
            // Above MAX_COUNT, and past 2^64.
            "1000000x512:64",
            "8x1000000:64",
            "8x512:1000000",
            "8x512:18446744073709551616",
            "8x512:64+1000000+1",
            // Under MAX_COUNT each but over the element cap.
            "999999x999999:64",
        ] {
            assert!(parse_predict(bad).is_err(), "{bad:?} accepted");
        }
        let e = parse_predict("8x512:64+0+0").unwrap_err();
        assert!(e.contains("BENCH_PREDICT") && e.contains("leave the suffix off"), "{e}");
        assert!(parse_predict("999999x999999:64")
            .unwrap_err()
            .contains("input elements"));
    }

    #[test]
    fn predict_list_and_selection() {
        let (d, p) = select(None, None).unwrap();
        assert_eq!((d.len(), p.len()), (5, 0));
        // BENCH_PREDICT alone drops the default decode set; both set keep both.
        let (d, p) = select(None, Some("2x256:32,4x2x128:64+1+1")).unwrap();
        assert_eq!((d.len(), p.len()), (0, 2));
        let (d, p) = select(Some("1x1x64:32"), Some("2x256:32")).unwrap();
        assert_eq!((d.len(), p.len()), (1, 1));
        let (d, p) = select(Some("1x1x64:32"), None).unwrap();
        assert_eq!((d.len(), p.len()), (1, 0));
        assert!(select(None, Some("")).is_err());
        assert!(select(None, Some("2x256:32,")).is_err());
        assert!(select(None, Some("2x256:32,bad")).is_err());
        assert!(select(Some(""), Some("2x256:32")).is_err());
        assert!(select(Some("1x1x64:32,bad"), None).is_err());
    }

    #[test]
    fn stage_flag_is_exactly_zero_or_one() {
        assert!(!parse_flag("BENCH_STAGES", "0").unwrap());
        assert!(parse_flag("BENCH_STAGES", "1").unwrap());
        for bad in ["", "2", "01", "true", "yes", "-1", " 1", "1 ", "18446744073709551616"] {
            let e = parse_flag("BENCH_STAGES", bad).unwrap_err();
            assert!(e.contains("BENCH_STAGES") && e.contains("0 or 1"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn predict_batch_knob_is_a_strict_count_of_at_least_one() {
        assert_eq!(parse_count("BENCH_PREDICT_BATCH", "4", 1).unwrap(), 4);
        for bad in ["0", "", "x", "-1", "04", "1000000", "18446744073709551616"] {
            assert!(parse_count("BENCH_PREDICT_BATCH", bad, 1).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn ragged_lengths_are_deterministic_and_bounded() {
        for len in [1usize, 3, 4, 31, 512, 2048] {
            let lens: Vec<usize> = (0..200).map(|i| series_len(i, len)).collect();
            assert_eq!(lens[0], len, "series 0 is the longest");
            assert!(
                lens.iter().all(|&l| l <= len && 4 * l >= 3 * len),
                "len {len}: {lens:?}"
            );
            assert_eq!(lens, (0..200).map(|i| series_len(i, len)).collect::<Vec<_>>());
        }
        assert!(
            (0..50)
                .map(|i| series_len(i, 512))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 10
        );
        // Pinned, so the Python lane's formula cannot drift from this one.
        assert_eq!(
            (0..5).map(|i| series_len(i, 512)).collect::<Vec<_>>(),
            [512, 451, 390, 458, 397]
        );
    }

    #[test]
    fn predict_series_have_the_documented_shapes() {
        let w = p("5x2x64:32+1+3");
        let s = predict_series(&w);
        assert_eq!(s.len(), 5);
        for (i, x) in s.iter().enumerate() {
            let len = series_len(i, 64);
            assert_eq!((x.target.rows, x.target.values.len()), (2, 2 * len));
            let po = x.past_only.as_ref().unwrap();
            let pf = x.past_future.as_ref().unwrap();
            assert_eq!((po.rows, po.values.len()), (1, len));
            assert_eq!((pf.rows, pf.values.len()), (3, 3 * (len + 32)));
            assert!(x
                .target
                .values
                .iter()
                .chain(&po.values)
                .chain(&pf.values)
                .all(|v| v.is_finite()));
        }
        assert_eq!(s[1].target.values[0], series(2, 0, series_len(1, 64)));
        assert_eq!(s[1].target.values[series_len(1, 64)], series(3, 0, series_len(1, 64)));
        let u = predict_series(&p("2x64:32"));
        assert!(u
            .iter()
            .all(|x| x.past_only.is_none() && x.past_future.is_none() && x.target.rows == 1));
    }

    #[test]
    fn forwards_follow_the_row_cap() {
        assert_eq!(forwards_for(4, 100, 8192).unwrap(), 1);
        assert_eq!(forwards_for(4, 4096, 8192).unwrap(), 2);
        assert_eq!(forwards_for(5, 3000, 8192).unwrap(), 3);
        assert_eq!(forwards_for(1, 8192, 8192).unwrap(), 1);
        assert!(forwards_for(1, 8193, 8192).unwrap_err().contains("8193"));
    }

    #[test]
    fn staged_and_plain_forecasts_must_agree() {
        agree("w", &[1.0, 2.0], &[1.0, 2.00001]).unwrap();
        assert!(agree("w", &[1.0, 2.0], &[1.0]).unwrap_err().contains("2 values"));
        assert!(agree("w", &[1.0, 2.5], &[1.0, 2.0]).unwrap_err().contains("element 1"));
        assert!(agree("w", &[f32::NAN], &[1.0]).is_err());
    }

    #[test]
    fn stage_json_shares_sum_to_one() {
        let j = stages_json(&[("plan", vec![1.0, 3.0, 2.0]), ("forward", vec![8.0, 6.0, 6.0])]).unwrap();
        assert!(
            j.contains("\"stage\":\"plan\",\"ms_min\":1.0000,\"ms_median\":2.0000,\"share\":0.2500"),
            "{j}"
        );
        assert!(
            j.contains("\"stage\":\"forward\",\"ms_min\":6.0000,\"ms_median\":6.0000,\"share\":0.7500"),
            "{j}"
        );
        assert!(stages_json(&[("plan", vec![0.0])]).is_err());
        assert!(stages_json(&[("plan", vec![f64::NAN])]).is_err());
    }

    #[test]
    fn stages_accumulate_and_keep_first_run_order() {
        let mut acc = StageMs::new();
        stage(&mut acc, "a", || Ok(())).unwrap();
        stage(&mut acc, "b", || Ok(())).unwrap();
        stage(&mut acc, "a", || Ok(())).unwrap();
        assert_eq!(acc.iter().map(|s| s.0).collect::<Vec<_>>(), ["a", "b"]);
        assert!(stage(&mut acc, "c", || -> Result<(), String> { Err("boom".into()) }).is_err());
        assert_eq!(acc.len(), 2, "a failed stage records nothing");
    }

    #[test]
    fn measure_refuses_a_changing_stage_list() {
        let mut n = 0;
        let r = measure(0, 2, || {
            n += 1;
            Ok(if n == 1 { vec![("a", 1.0)] } else { vec![("b", 1.0)] })
        });
        assert!(r.is_err());
        let t = measure(1, 3, || Ok(vec![("a", 1.0), ("b", 2.0)])).unwrap();
        assert_eq!((t.wall_ms.len(), t.stages.len(), t.stages[1].1.len()), (3, 2, 3));
    }

    #[test]
    fn stages_accept_zero_one_and_kernels_only() {
        assert_eq!(parse_stages("0").unwrap(), Stages::Off);
        assert_eq!(parse_stages("1").unwrap(), Stages::Split);
        assert_eq!(parse_stages("kernels").unwrap(), Stages::Kernels);
        for bad in ["", "2", "Kernels", "kernel", "true", " kernels"] {
            let e = parse_stages(bad).unwrap_err();
            assert!(
                e.contains("BENCH_STAGES") && e.contains("0, 1 or kernels"),
                "{bad:?}: {e}"
            );
        }
    }

    #[test]
    fn sweep_is_a_list_of_distinct_counts() {
        assert_eq!(parse_sweep("1,4,16,64").unwrap(), [1, 4, 16, 64]);
        assert_eq!(parse_sweep("8").unwrap(), [8]);
        for bad in ["", "0", "1,,2", "1,1", "1, 2", "x", "1,"] {
            assert!(parse_sweep(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn kernels_are_classified_by_name() {
        for (name, class) in [
            ("matmul2d_tensorops_nn_f32", "gemm"),
            ("matmul2d_tensorops_nn_accum_f32", "gemm"),
            ("gemm_epilogue_tiled", "gemm"),
            ("timesfm3_attn_h80_r4_g16", "attention"),
            ("timesfm3_qk_prep_h80", "qk_prep_rope"),
            ("rms_norm_f32", "norm"),
            ("rms_norm_residual_add_f32", "norm"),
            ("zero_f32", "copy_transpose"),
            ("copy_f32", "copy_transpose"),
            ("transpose2d_f32", "copy_transpose"),
            ("add_inplace_f32", "elementwise"),
            ("mlp_silu", "elementwise"),
            ("execute_icb", "other"),
            ("?", "other"),
        ] {
            assert_eq!(classify(name), class, "{name}");
        }
    }

    fn stamp(name: &str, dt_ns: u64) -> tessl::dispatch_stamps::DispatchStamp {
        tessl::dispatch_stamps::DispatchStamp {
            name: name.to_string(),
            layer: -1,
            barrier_before: true,
            t_end_ns: dt_ns,
            dt_ns,
        }
    }

    fn batch(stamps: &[(&str, u64)]) -> DispatchStampBatch {
        DispatchStampBatch {
            freq_hz: 1_000_000_000,
            t0_ns: 0,
            stamps: stamps.iter().map(|&(n, dt)| stamp(n, dt)).collect(),
            overflowed: false,
            dropped_batches: 0,
            command_buffers: 1,
        }
    }

    fn record(b: Option<DispatchStampBatch>) -> ForwardRecord {
        ForwardRecord { host_ms: 1.0, batch: b }
    }

    #[test]
    fn aggregation_sums_by_kernel_and_class() {
        let p = profile_of(&[
            record(Some(batch(&[
                ("gemm", 3_000_000),
                ("rms_norm_f32", 1_000_000),
                ("gemm", 4_000_000),
            ]))),
            record(Some(batch(&[("timesfm3_attn_h80_r4_g16", 2_000_000)]))),
        ])
        .unwrap();
        assert_eq!((p.forwards, p.commits), (2, 2));
        assert_eq!(p.kernels["gemm"], (2, 7_000_000));
        let s = summarize(&[p]).unwrap();
        assert_eq!((s.dispatches, s.forwards, s.commits), (4, 2, 2));
        assert!((s.stamped_ms - 10.0).abs() < 1e-9);
        let classes: Vec<_> = s.classes.iter().map(|c| (c.class, c.dispatches)).collect();
        assert_eq!(classes, [("gemm", 2), ("attention", 1), ("norm", 1)]);
        assert!((s.classes.iter().map(|c| c.share).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((s.classes[0].share - 0.7).abs() < 1e-12);
        assert_eq!(s.kernels[0].name, "gemm");
    }

    #[test]
    fn summary_takes_the_median_per_kernel_and_refuses_a_changing_profile() {
        let one = |ns| profile_of(&[record(Some(batch(&[("gemm", ns), ("copy_f32", 10)])))]).unwrap();
        let s = summarize(&[one(1_000_000), one(9_000_000), one(2_000_000)]).unwrap();
        assert!((s.kernels[0].ms - 2.0).abs() < 1e-9, "{}", s.kernels[0].ms);
        let other = profile_of(&[record(Some(batch(&[("gemm", 1)])))]).unwrap();
        assert!(summarize(&[one(1), other])
            .unwrap_err()
            .contains("changed between iterations"));
        assert!(summarize(&[]).is_err());
    }

    #[test]
    fn a_profile_that_misses_part_of_the_forward_is_refused() {
        let good = batch(&[("gemm", 1)]);
        assert!(profile_of(&[record(Some(good.clone()))]).is_ok());
        let mut over = good.clone();
        over.overflowed = true;
        let e = profile_of(&[record(Some(over))]).unwrap_err();
        assert!(
            e.contains("overflowed") && e.contains("1023") && e.contains("BENCH_PREDICT_BATCH"),
            "{e}"
        );
        let mut dropped = good.clone();
        dropped.dropped_batches = 1;
        assert!(profile_of(&[record(Some(dropped))]).unwrap_err().contains("dropped"));
        assert!(profile_of(&[record(None)]).unwrap_err().contains("no stamp batch"));
        assert!(profile_of(&[record(Some(batch(&[])))])
            .unwrap_err()
            .contains("no dispatches"));
        assert!(profile_of(&[]).is_err());
        // One bad forward among good ones refuses the whole profile.
        let mut bad = good.clone();
        bad.overflowed = true;
        assert!(profile_of(&[record(Some(good)), record(Some(bad))]).is_err());
    }

    #[test]
    fn forecast_diff_reports_max_and_bit_equality() {
        assert_eq!(forecast_diff(&[1.0, 2.0], &[1.0, 2.0]), (0.0, true));
        let (d, bit) = forecast_diff(&[1.0, 2.5], &[1.0, 2.0]);
        assert_eq!((d, bit), (0.5, false));
        assert!(forecast_diff(&[f32::NAN], &[1.0]).0.is_infinite());
    }

    #[test]
    fn kernel_row_reports_classes_overhead_and_top_kernels() {
        let prof = |g| profile_of(&[record(Some(batch(&[("gemm", g), ("rms_norm_f32", 1_000_000)])))]).unwrap();
        let k = KernelRun {
            off_wall: vec![12.0, 10.0, 11.0],
            on_wall: vec![24.0, 20.0, 22.0],
            off_forward: vec![8.0, 8.0, 8.0],
            on_forward: vec![16.0, 16.0, 16.0],
            profiles: vec![prof(3_000_000), prof(3_000_000), prof(3_000_000)],
            max_abs_diff: 0.0,
            bit_equal: true,
        };
        let (lo, med, j) = kernels_json(&k, (0.25, 0.5), 64).unwrap();
        assert_eq!((lo, med), (10.0, 11.0));
        for want in [
            "\"overhead_forward\":2.0000",
            "\"overhead_wall\":2.0000",
            "\"dispatches_per_forward\":2.00",
            "\"commits_per_forward\":1.00",
            "\"gpu_stamped_ms\":4.0000",
            "\"host_outside_stamped_ms\":12.0000",
            "\"proxy_upload_ms\":0.2500",
            "\"rows_per_forward\":64",
            "\"bit_equal\":true",
            "{\"class\":\"gemm\",\"dispatches\":1,\"ms\":3.0000,\"share\":0.7500}",
            "{\"kernel\":\"rms_norm_f32\",\"class\":\"norm\",\"dispatches\":1,\"ms\":1.0000,\"share\":0.2500}",
        ] {
            assert!(j.contains(want), "{want} not in {j}");
        }
        let mut zero = KernelRun {
            off_forward: vec![0.0],
            ..k
        };
        zero.off_wall = vec![1.0];
        assert!(kernels_json(&zero, (0.0, 0.0), 1).is_err());
    }
}
