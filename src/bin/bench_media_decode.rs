//! Time `tessl::media`: native image, audio and video decoding for EmbeddingGemma
//! 2 on the CPU (ImageIO, AudioToolbox, AVFoundation). No device, no checkpoint.
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python bench/make_media_bench_inputs.py     # once
//! cargo run --release --bin bench_media_decode
//! BENCH_MEDIA_DIR=/path/to/files BENCH_ITERS=20 cargo run --release --bin bench_media_decode
//! ```
//!
//! `BENCH_MEDIA_DIR` (default `target/media_bench`) holds the files that
//! `bench/make_media_bench_inputs.py` writes. Every file in it is timed by
//! extension: images (`png`, `jpg`, `tiff`) with `decode_image_file`; audio
//! (`wav`, `flac`, `m4a`, `mp3`) with `decode_audio_file` to 16 kHz mono; video
//! (`mp4`) as `video_open` (`VideoSource::open`: the first pass, which reads the
//! compressed samples to count frames and read their times), then twice under
//! each colour path (`:exact`, the default: VideoToolbox planes converted by
//! libswscale's arithmetic in Rust; `:bgra`: AVFoundation's own conversion) as
//! `video_sampled_N` (decoding the N frames the processor's 1 fps sampling
//! picks, N being 9 for a 10 s clip) and `video_first_K` (decoding the first K
//! = 30 frames, so the row divides into a cost per frame), each from a freshly
//! opened source. A run in which a path fell back to the other is refused.
//! `bench/make_video_bench_inputs.py` writes four clips for these rows
//! (`BENCH_MEDIA_DIR=target/media_bench_video`), plotted by
//! `plot_results video-color`.
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 2) are per workload.
//! Prints a table, then one JSON array on the last stdout line: `{workload,
//! ms_min, ms_median, iters, units, unit}` where `units` is how many of `unit`
//! (`megapixel`, `audio_s`, `frame`) the call produces. The numbers are
//! minimum-of-N wall times on whatever else the machine is doing; they are
//! indicative, not a paired comparison.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tessl::embedgemma2_pre::PreConfig;
use tessl::media::audio::{decode_audio_file, AudioOptions};
use tessl::media::image::{decode_image_file, ImageOptions};
use tessl::media::video::{ColorPath, Conversion, VideoOptions, VideoSource};

struct Row {
    workload: String,
    ms_min: f64,
    ms_median: f64,
    iters: usize,
    units: f64,
    unit: &'static str,
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

/// How many leading frames the `video_first_N` rows decode.
const FIRST_FRAMES: usize = 30;

/// A timing is only a timing of the path it names: refuse a run in which the
/// exact path fell back to AVFoundation's conversion (a stream it does not
/// cover) or the other way round.
fn check_path(got: &Conversion, exact: bool) -> Result<(), String> {
    match (got, exact) {
        (Conversion::Swscale { .. }, true) | (Conversion::AvFoundation, false) => Ok(()),
        _ => Err(format!("the {} path ran {got:?}", if exact { "exact" } else { "bgra" })),
    }
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
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "{}: no files (run bench/make_media_bench_inputs.py)",
            dir.display()
        ));
    }
    let cfg = PreConfig::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2/eg2p_processor_config.json"),
    )?;
    let mut rows = Vec::new();
    for path in &files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "png" | "jpg" | "tiff" => {
                let d = decode_image_file(path, &ImageOptions::default()).map_err(|e| format!("{name}: {e}"))?;
                let mp = (d.width * d.height) as f64 / 1e6;
                let (min, med) = time(warmup, iters, || {
                    black_box(decode_image_file(path, &ImageOptions::default()).map_err(|e| e.to_string())?);
                    Ok(())
                })?;
                rows.push(Row {
                    workload: name,
                    ms_min: min,
                    ms_median: med,
                    iters,
                    units: mp,
                    unit: "megapixel",
                });
            }
            "wav" | "flac" | "m4a" | "mp3" => {
                let d = decode_audio_file(path, &AudioOptions::default()).map_err(|e| format!("{name}: {e}"))?;
                let secs = d.samples.len() as f64 / f64::from(d.sample_rate);
                let (min, med) = time(warmup, iters, || {
                    black_box(decode_audio_file(path, &AudioOptions::default()).map_err(|e| e.to_string())?);
                    Ok(())
                })?;
                rows.push(Row {
                    workload: name,
                    ms_min: min,
                    ms_median: med,
                    iters,
                    units: secs,
                    unit: "audio_s",
                });
            }
            "mp4" => {
                let opts = VideoOptions::default();
                let src = VideoSource::open(path, &opts).map_err(|e| format!("{name}: {e}"))?;
                let total = src.info().total_frames;
                let (min, med) = time(warmup, iters, || {
                    black_box(VideoSource::open(path, &opts).map_err(|e| e.to_string())?);
                    Ok(())
                })?;
                rows.push(Row {
                    workload: format!("{name}:video_open"),
                    ms_min: min,
                    ms_median: med,
                    iters,
                    units: total as f64,
                    unit: "frame",
                });
                // The same two calls under each colour path: `exact` is the
                // default (VideoToolbox planes, libswscale's arithmetic in Rust),
                // `bgra` AVFoundation's own conversion. `sampled` decodes the
                // frames the processor's sampling picks; `first` the first
                // FIRST_FRAMES, so the rows divide into a cost per frame.
                let first: Vec<usize> = (0..total.min(FIRST_FRAMES)).collect();
                for (label, color, conversion_is_exact) in [
                    ("exact", ColorPath::default(), true),
                    ("bgra", ColorPath::AvFoundation, false),
                ] {
                    let opts = VideoOptions { color, ..opts };
                    let mut sampled = 0usize;
                    let (min, med) = time(warmup, iters, || {
                        let v = VideoSource::open(path, &opts).map_err(|e| e.to_string())?;
                        let d = v.decode_sampled(&cfg.video).map_err(|e| e.to_string())?;
                        check_path(&d.conversion, conversion_is_exact)?;
                        sampled = d.frames.len();
                        black_box(d);
                        Ok(())
                    })?;
                    rows.push(Row {
                        workload: format!("{name}:video_sampled_{sampled}:{label}"),
                        ms_min: min,
                        ms_median: med,
                        iters,
                        units: sampled as f64,
                        unit: "frame",
                    });
                    let (min, med) = time(warmup, iters, || {
                        let v = VideoSource::open(path, &opts).map_err(|e| e.to_string())?;
                        let d = v.decode_frames(&first).map_err(|e| e.to_string())?;
                        check_path(&d.conversion, conversion_is_exact)?;
                        black_box(d);
                        Ok(())
                    })?;
                    rows.push(Row {
                        workload: format!("{name}:video_first_{}:{label}", first.len()),
                        ms_min: min,
                        ms_median: med,
                        iters,
                        units: first.len() as f64,
                        unit: "frame",
                    });
                }
            }
            _ => eprintln!("skipping {name}: not a media extension this bench knows"),
        }
    }
    println!(
        "{:44} {:>10} {:>10} {:>8} {:>12}",
        "workload", "ms min", "ms median", "units", "ms/unit(min)"
    );
    for r in &rows {
        println!(
            "{:44} {:>10.3} {:>10.3} {:>8.2} {:>12.3}  {}",
            r.workload,
            r.ms_min,
            r.ms_median,
            r.units,
            r.ms_min / r.units,
            r.unit
        );
    }
    let json: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{},\"units\":{:.4},\"unit\":\"{}\"}}",
                r.workload, r.ms_min, r.ms_median, r.iters, r.units, r.unit
            )
        })
        .collect();
    println!("[{}]", json.join(","));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bench_media_decode: {e}");
        std::process::exit(1);
    }
}
