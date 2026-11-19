//! End-to-end forecast latency of `tessl::timesfm3` on the real checkpoint.
//!
//! ```text
//! TIMESFM3_DIR=<google/timesfm-3.0-pytorch snapshot> \
//!   cargo run --release --bin bench_timesfm3
//! ```
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
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! backend, ms_min, ms_median, iters, rows, check, peak_footprint_mib}`, where
//! `rows` is `b * v * n` patch rows and `check` the mean of the forecast's
//! median quantile over target variates (lanes should agree to ~1e-3; a
//! mismatch means they did different work). Before anything is timed every
//! forecast must be finite.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::safetensors::SafeTensors;
use tessl::timesfm3::{Timesfm3Config, Timesfm3Model};
use tessl::timesfm3_host::DecodeInput;
use tessl::GpuRuntime;

use common::{env_usize, peak_footprint};

const DEFAULT_WORKLOADS: &str = "1x1x512:128,16x1x2048:128,64x1x512:64,1x8x1024:96,4x4x2048:256";

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

fn main() -> Result<(), String> {
    let dir = PathBuf::from(std::env::var_os("TIMESFM3_DIR").ok_or("set TIMESFM3_DIR to the checkpoint directory")?);
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
    let workloads: Vec<Workload> = std::env::var("BENCH_WORKLOADS")
        .unwrap_or_else(|_| DEFAULT_WORKLOADS.to_string())
        .split(',')
        .map(parse)
        .collect::<Result<_, _>>()?;
    let rt = GpuRuntime::new()?;
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
        for _ in 0..warmup {
            model.decode(&inp)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let f = model.decode(&inp)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&f.values);
        }
        ms.sort_by(f64::total_cmp);
        let peak_mib = peak_footprint()? as f64 / (1u64 << 20) as f64;
        rows_out.push(format!(
            "{{\"workload\":\"{}\",\"backend\":\"tessl-f32\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{},\"rows\":{},\"check\":{:.6},\"peak_footprint_mib\":{:.1}}}",
            w.name,
            ms[0],
            ms[ms.len() / 2],
            iters,
            w.b * model.config().host.rows_per_series(w.v, w.ctx, w.horizon),
            check,
            peak_mib
        ));
    }
    println!("[{}]", rows_out.join(","));
    Ok(())
}
