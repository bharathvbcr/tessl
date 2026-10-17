//! Time the sample-rate conversion of `tessl::media::audio`: libswresample's
//! algorithm in Rust (`tessl::media::resample`) against the system
//! `AudioConverter` it replaced as the default. No device, no checkpoint.
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python bench/make_media_bench_inputs.py     # once
//! cargo run --release --bin bench_media_resample
//! BENCH_MEDIA_DIR=/path/to/files BENCH_ITERS=20 cargo run --release --bin bench_media_resample
//! ```
//!
//! For every audio file in `BENCH_MEDIA_DIR` (default `target/media_bench`,
//! `wav`, `flac`, `m4a` and `mp3`) whose rate is not 16 kHz, `decode_audio_file`
//! to 16 kHz mono is timed four ways:
//!
//! * `native decode`: the same decode asked for the file's own rate, so it only
//!   decodes and mixes. What every other row has in common.
//! * `swresample`: the default, `Resampler::Swresample`.
//! * `audiotoolbox`: `Resampler::AudioToolbox`, the system converter.
//! * `swresample converter alone`: [`tessl::media::resample::resample`] on the
//!   one mixed channel the native decode returned (no file, no decoder).
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 2) are per row.
//! Prints a table, then one JSON array on the last stdout line: `{workload,
//! stage, ms_min, ms_median, iters, audio_s}`. Minimum-of-N wall times on
//! whatever else the machine is doing: indicative, not a paired comparison.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tessl::media::audio::{decode_audio_file, AudioOptions, Resampler};
use tessl::media::resample::resample;

struct Row {
    workload: String,
    stage: &'static str,
    ms_min: f64,
    ms_median: f64,
    iters: usize,
    audio_s: f64,
}

fn env_count(name: &str, default: usize, min: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(format!("{name}: {e}")),
        Ok(v) => {
            let n: usize = v.trim().parse().map_err(|_| format!("{name}={v:?}: not a count"))?;
            if n < min {
                return Err(format!("{name}={n}: at least {min} expected"));
            }
            Ok(n)
        }
    }
}

fn time(warmup: usize, iters: usize, mut f: impl FnMut() -> Result<(), String>) -> Result<(f64, f64), String> {
    for _ in 0..warmup {
        f()?;
    }
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f()?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    Ok((ms[0], ms[ms.len() / 2]))
}

fn run() -> Result<(), String> {
    let iters = env_count("BENCH_ITERS", 10, 1)?;
    let warmup = env_count("BENCH_WARMUP", 2, 0)?;
    let dir = std::env::var("BENCH_MEDIA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/media_bench"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e} (run bench/make_media_bench_inputs.py)", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "wav" | "flac" | "m4a" | "mp3"))
        })
        .collect();
    files.sort();
    let mut rows = Vec::new();
    for path in &files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let probe = decode_audio_file(path, &AudioOptions::default()).map_err(|e| format!("{name}: {e}"))?;
        if probe.source_rate == f64::from(probe.sample_rate) {
            continue;
        }
        let rate = probe.source_rate as u32;
        let audio_s = probe.samples.len() as f64 / f64::from(probe.sample_rate);
        let native_opts = AudioOptions {
            sample_rate: rate,
            ..AudioOptions::default()
        };
        let apple_opts = AudioOptions {
            resampler: Resampler::AudioToolbox,
            ..AudioOptions::default()
        };
        let native = decode_audio_file(path, &native_opts).map_err(|e| format!("{name}: {e}"))?;
        let mut push = |stage: &'static str, (min, med): (f64, f64)| {
            rows.push(Row {
                workload: name.clone(),
                stage,
                ms_min: min,
                ms_median: med,
                iters,
                audio_s,
            });
        };
        push(
            "native decode",
            time(warmup, iters, || {
                black_box(decode_audio_file(path, &native_opts).map_err(|e| e.to_string())?);
                Ok(())
            })?,
        );
        push(
            "swresample",
            time(warmup, iters, || {
                black_box(decode_audio_file(path, &AudioOptions::default()).map_err(|e| e.to_string())?);
                Ok(())
            })?,
        );
        push(
            "audiotoolbox",
            time(warmup, iters, || {
                black_box(decode_audio_file(path, &apple_opts).map_err(|e| e.to_string())?);
                Ok(())
            })?,
        );
        push(
            "swresample converter alone",
            time(warmup, iters, || {
                black_box(resample(&native.samples, rate, probe.sample_rate).map_err(|e| e.to_string())?);
                Ok(())
            })?,
        );
    }
    if rows.is_empty() {
        return Err(format!(
            "{}: no audio file with a rate other than 16 kHz (run bench/make_media_bench_inputs.py)",
            dir.display()
        ));
    }
    println!(
        "{:32} {:28} {:>10} {:>10} {:>8}",
        "workload", "stage", "ms min", "ms median", "audio s"
    );
    for r in &rows {
        println!(
            "{:32} {:28} {:>10.3} {:>10.3} {:>8.2}",
            r.workload, r.stage, r.ms_min, r.ms_median, r.audio_s
        );
    }
    let json: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"stage\":\"{}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{},\"audio_s\":{:.4}}}",
                r.workload, r.stage, r.ms_min, r.ms_median, r.iters, r.audio_s
            )
        })
        .collect();
    println!("[{}]", json.join(","));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bench_media_resample: {e}");
        std::process::exit(1);
    }
}
