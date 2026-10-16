//! Time `tessl::embedgemma2_pre`: native image, audio and video preprocessing for
//! EmbeddingGemma 2, on the CPU. No device, no checkpoint.
//!
//! ```text
//! cargo run --release --bin bench_embedgemma2_pre
//! BENCH_WORKLOADS=image_640x480,audio_10s BENCH_ITERS=30 cargo run --release --bin bench_embedgemma2_pre
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,` (default all):
//!
//! * `image_WxH`: one decoded RGB image, `preprocess_image` at the checkpoint's
//!   geometry (patch 16, pooling 3, 280 soft tokens);
//! * `audio_Ns`: N seconds of 16 kHz mono PCM, `AudioFrontEnd::extract` (the
//!   window, filterbank and FFT plan are built once, outside the timing);
//! * `video_40f_WxH`: 40 decoded RGB frames at 1 fps, so the processor samples
//!   40 and caps them at 32 (`preprocess_video`, 140 soft tokens per frame).
//!
//! `BENCH_ITERS` (default 15) and `BENCH_WARMUP` (default 3) are per
//! workload. `BENCH_PROCESSOR_CONFIG` names a `processor_config.json` (default:
//! the snapshot's, from `EMBEDGEMMA2_SNAPSHOT`, else the committed copy in
//! `tests/fixtures/embedgemma2`).
//!
//! Prints one JSON array on the last stdout line, the format `bench_paired
//! --format workloads` reads: `{workload, ms_min, ms_median, iters, units,
//! unit, threads, input_crc}` where `units` is how many of `unit` (`image`,
//! `audio_s`, `frame`) the call processes and `input_crc` the CRC-32 of the
//! input bytes, which `bench/embedgemma2_pre_hf.py` computes over the same
//! deterministic input, so the two lanes are shown to time the same data. The
//! inputs are integer formulas (no floating-point trig), identical in both
//! languages. Not timed: input construction. `threads` is the most the
//! resize and the FFT front end use here (`available_parallelism`, at most 8).

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_pre::{preprocess_image, preprocess_video, AudioFrontEnd, PreConfig, RawImage};

const SAMPLE_RATE: u32 = 16_000;
const VIDEO_FRAMES: usize = 40;
const IMAGES: [(usize, usize); 4] = [(224, 224), (640, 480), (1920, 1080), (4000, 3000)];
const AUDIO_SECONDS: [usize; 3] = [1, 10, 30];
const VIDEOS: [(usize, usize); 2] = [(640, 360), (1280, 720)];

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

/// CRC-32 (IEEE), the checksum Python's `zlib.crc32` computes.
fn crc32(chunks: &[&[u8]]) -> u32 {
    let table: Vec<u32> = (0..256u32)
        .map(|n| (0..8).fold(n, |c, _| if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }))
        .collect();
    let mut c = !0u32;
    for chunk in chunks {
        for &b in *chunk {
            c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    !c
}

/// Byte `c` of the pixel at row `y`, column `x` of frame `f`: an integer formula
/// that `bench/embedgemma2_pre_hf.py` repeats with numpy.
fn pixel(f: usize, y: usize, x: usize, c: usize) -> u8 {
    ((y * 7 + x * 13 + c * 29 + ((x * y) >> 4) + f * 11) & 255) as u8
}

fn frame(f: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out.push(pixel(f, y, x, c));
            }
        }
    }
    out
}

/// Sample `i`: a 24-bit hash of the index scaled by 2^-25, so every value is
/// exact in f32 and the signal is noise-like, within +-0.25.
fn sample(i: usize) -> f32 {
    let h = ((i as u64).wrapping_mul(2_654_435_761) & 0xFFFF_FFFF) >> 8;
    (h as i64 - (1 << 23)) as f32 * (1.0 / 33_554_432.0)
}

struct Row {
    workload: String,
    ms: Vec<f64>,
    units: f64,
    unit: &'static str,
    crc: u32,
}

fn time(warmup: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    for _ in 0..warmup {
        run();
    }
    let mut ms: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            run();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    ms
}

fn main() -> Result<(), String> {
    let config_path = match std::env::var("BENCH_PROCESSOR_CONFIG") {
        Ok(p) => PathBuf::from(p),
        Err(_) => match std::env::var("EMBEDGEMMA2_SNAPSHOT") {
            Ok(dir) => PathBuf::from(dir).join("processor_config.json"),
            Err(_) => {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2/eg2p_processor_config.json")
            }
        },
    };
    let cfg = PreConfig::from_file(&config_path)?;
    let iters = env_count("BENCH_ITERS", 15, 1)?;
    let warmup = env_count("BENCH_WARMUP", 3, 0)?;
    let all: Vec<String> = IMAGES
        .iter()
        .map(|(w, h)| format!("image_{w}x{h}"))
        .chain(AUDIO_SECONDS.iter().map(|s| format!("audio_{s}s")))
        .chain(VIDEOS.iter().map(|(w, h)| format!("video_{VIDEO_FRAMES}f_{w}x{h}")))
        .collect();
    let wanted: Vec<String> = match std::env::var("BENCH_WORKLOADS") {
        Err(_) => all.clone(),
        Ok(s) => s
            .split(',')
            .map(|w| w.trim().to_string())
            .filter(|w| !w.is_empty())
            .collect(),
    };
    if let Some(bad) = wanted.iter().find(|w| !all.contains(w)) {
        return Err(format!(
            "BENCH_WORKLOADS: unknown workload {bad:?}; known: {}",
            all.join(",")
        ));
    }
    let front_end = AudioFrontEnd::new(&cfg.audio)?;

    let mut rows: Vec<Row> = Vec::new();
    for name in &wanted {
        if let Some(rest) = name.strip_prefix("image_") {
            let (w, h) = rest
                .split_once('x')
                .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                .unwrap();
            let data = frame(0, w, h);
            let img = RawImage::rgb(w, h, &data);
            let ms = time(warmup, iters, || {
                black_box(preprocess_image(&cfg.image, black_box(&img)).expect("preprocess_image"));
            });
            rows.push(Row {
                workload: name.clone(),
                ms,
                units: 1.0,
                unit: "image",
                crc: crc32(&[&data]),
            });
        } else if let Some(rest) = name.strip_prefix("audio_") {
            let seconds: usize = rest.trim_end_matches('s').parse().unwrap();
            let pcm: Vec<f32> = (0..seconds * SAMPLE_RATE as usize).map(sample).collect();
            let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ms = time(warmup, iters, || {
                black_box(front_end.extract(black_box(&pcm), SAMPLE_RATE).expect("extract"));
            });
            rows.push(Row {
                workload: name.clone(),
                ms,
                units: seconds as f64,
                unit: "audio_s",
                crc: crc32(&[&bytes]),
            });
        } else {
            let rest = name.strip_prefix(&format!("video_{VIDEO_FRAMES}f_")).unwrap();
            let (w, h): (usize, usize) = rest
                .split_once('x')
                .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                .unwrap();
            let data: Vec<Vec<u8>> = (0..VIDEO_FRAMES).map(|f| frame(f, w, h)).collect();
            let frames: Vec<RawImage<'_>> = data.iter().map(|d| RawImage::rgb(w, h, d)).collect();
            let sampled = preprocess_video(&cfg.video, &frames, 1.0)?.frames.len();
            let ms = time(warmup, iters, || {
                black_box(preprocess_video(&cfg.video, black_box(&frames), 1.0).expect("preprocess_video"));
            });
            let chunks: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
            rows.push(Row {
                workload: name.clone(),
                ms,
                units: sampled as f64,
                unit: "frame",
                crc: crc32(&chunks),
            });
        }
    }

    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(8);
    let json: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"units\":{},\"unit\":\"{}\",\"threads\":{threads},\"input_crc\":\"{:08x}\"}}",
                r.workload,
                r.ms[0],
                r.ms[r.ms.len() / 2],
                r.units,
                r.unit,
                r.crc
            )
        })
        .collect();
    for r in &rows {
        eprintln!(
            "{:<22} min {:>9.3} ms  median {:>9.3} ms  per {} {:.3} ms",
            r.workload,
            r.ms[0],
            r.ms[r.ms.len() / 2],
            r.unit,
            r.ms[0] / r.units
        );
    }
    println!("[{}]", json.join(","));
    Ok(())
}
