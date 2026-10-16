//! End-to-end latency of one video through
//! `tessl::embedgemma2_mm::EmbedGemma2Multimodal::encode` on the real checkpoint:
//! 32 decoded frames at 640x360 (the processor's frame cap), resized to
//! 384x720 (24 x 45 patches of 16), 120 soft tokens a frame (the budget is up
//! to 140), 2 + 32 x 122 = 3906 ids.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo run --release --bin bench_embedgemma2_video
//! ```
//!
//! The frames are synthetic and deterministic (no checkpoint goldens are read).
//! Both operand modes run in one process, one after the other
//! (`BENCH_PRECISION=f32|bf16|both`, default `both`); each loads its own model
//! and drops it before the next. `BENCH_ITERS` (default 10) and `BENCH_WARMUP`
//! (default 3) are per workload.
//!
//! Workloads, per mode:
//!
//! * `preprocess`: `preprocess_video` of the 32 raw frames (CPU only, the same
//!   for both modes; reported once, under `f32`);
//! * `tower`: `EmbedGemma2VisionModel::encode_images` of the 32 prepared frames
//!   (the vision tower alone, soft tokens read back to the host);
//! * `encode`: `EmbedGemma2Multimodal::encode` of one `MmItem::Video` (the tower
//!   and the text encoder behind one wait, the soft tokens staying on the
//!   device).
//!
//! What is timed: the call, from host pixels to host embeddings, min and median
//! of `BENCH_ITERS` after `BENCH_WARMUP` untimed calls. Not timed: the load.
//!
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! precision, ms_min, ms_median, iters, sync_waits, load_avg_1m}`. `sync_waits`
//! is the host waits of one call (`infer_trace`, on a separate call before the
//! timed ones; `null` for `preprocess`). `load_avg_1m` is the machine's
//! one-minute load average when the workload started: on a shared machine the
//! timings are indicative only. stderr carries `HASH <precision> <fnv1a64>`
//! over the bits of the embedding.

mod common;

use std::time::Instant;

use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::{preprocess_video, PreConfig, PreparedImage, PreparedVideo, RawImage};
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::GpuRuntime;

use common::env_usize;

const FRAMES: usize = 32;
const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn fnv1a(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

/// `FRAMES` frames of a drifting gradient with a moving bar and a little
/// noise: `[FRAMES][HEIGHT * WIDTH * 3]` RGB bytes.
fn synthetic_frames() -> Vec<Vec<u8>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..FRAMES)
        .map(|f| {
            let mut px = Vec::with_capacity(HEIGHT * WIDTH * 3);
            let bar = (f * 17) % WIDTH;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let noise = ((state >> 59) & 7) as usize;
                    let on_bar = x.abs_diff(bar) < 12;
                    px.push(if on_bar {
                        250
                    } else {
                        ((x + f * 3) * 255 / WIDTH).min(255) as u8
                    });
                    px.push(((y * 255 / HEIGHT) + noise).min(255) as u8);
                    px.push((((x + y) * 255 / (WIDTH + HEIGHT)) + noise).min(255) as u8);
                }
            }
            px
        })
        .collect()
}

fn load_avg() -> f64 {
    let mut avg = [0.0f64; 3];
    // SAFETY: `getloadavg` writes at most 3 doubles into the array it is given.
    let n = unsafe { libc::getloadavg(avg.as_mut_ptr(), 3) };
    if n >= 1 {
        avg[0]
    } else {
        f64::NAN
    }
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "null".to_string()
    }
}

struct Timing {
    ms_min: f64,
    ms_median: f64,
}

fn time(warmup: usize, iters: usize, mut f: impl FnMut() -> Result<(), String>) -> Result<Timing, String> {
    for _ in 0..warmup {
        f()?;
    }
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    Ok(Timing {
        ms_min: ms[0],
        ms_median: ms[ms.len() / 2],
    })
}

fn row(workload: &str, precision: &str, t: &Timing, iters: usize, waits: Option<u64>, load: f64) -> String {
    format!(
        "{{\"workload\":\"{workload}\",\"precision\":\"{precision}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"sync_waits\":{},\"load_avg_1m\":{}}}",
        t.ms_min,
        t.ms_median,
        waits.map_or("null".to_string(), |w| w.to_string()),
        json_num(load),
    )
}

fn main() -> Result<(), String> {
    let snap =
        std::path::PathBuf::from(std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| {
            "EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory".to_string()
        })?);
    let modes: Vec<(&str, GemmOperands)> = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("both") => vec![("f32", GemmOperands::ExactF32), ("bf16", GemmOperands::Bf16)],
        Ok("f32") => vec![("f32", GemmOperands::ExactF32)],
        Ok("bf16") => vec![("bf16", GemmOperands::Bf16)],
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32, bf16 or both")),
    };
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    let cfg = PreConfig::from_snapshot(&snap)?;
    let bytes = synthetic_frames();
    let raw: Vec<RawImage<'_>> = bytes.iter().map(|b| RawImage::rgb(WIDTH, HEIGHT, b)).collect();
    // One source frame a second, so the processor's 1 fps keeps all 32.
    let video: PreparedVideo = preprocess_video(&cfg.video, &raw, 1.0)?;
    if video.frames.len() != FRAMES {
        return Err(format!("{} frames kept, expected {FRAMES}", video.frames.len()));
    }
    let frames: Vec<ImageInput<'_>> = video.frames.iter().map(PreparedImage::input).collect();
    eprintln!(
        "video: {FRAMES} frames of {WIDTH}x{HEIGHT} resized to {:?}, {} soft tokens a frame, {} ids",
        video.frames[0].resized,
        video.soft_tokens_per_frame,
        2 + FRAMES * (video.soft_tokens_per_frame + 2)
    );

    let mut rows = Vec::new();
    for (k, (precision, operands)) in modes.into_iter().enumerate() {
        if k == 0 {
            let load = load_avg();
            let t = time(warmup, iters, || preprocess_video(&cfg.video, &raw, 1.0).map(drop))?;
            rows.push(row("preprocess", "f32", &t, iters, None, load));
        }
        let rt = GpuRuntime::new_inference()?;
        let model = EmbedGemma2Multimodal::load(&rt, &snap, operands)?;
        let item = [MmItem::Video(&frames)];
        let out = model.encode(&item, None)?;
        if !out.embeddings.iter().all(|v| v.is_finite()) {
            return Err(format!("{precision}: the embedding is not finite"));
        }
        eprintln!("HASH {precision} {:016x}", fnv1a(&out.embeddings));

        let load = load_avg();
        let t = time(warmup, iters, || model.vision().encode_images(&frames, false).map(drop))?;
        rows.push(row("tower", precision, &t, iters, None, load));

        infer_trace::set_enabled(true);
        let s0 = infer_trace::snapshot();
        model.encode(&item, None)?;
        let waits = infer_trace::snapshot().since(&s0).sync_waits;
        infer_trace::set_enabled(false);
        let load = load_avg();
        let t = time(warmup, iters, || model.encode(&item, None).map(drop))?;
        rows.push(row("encode", precision, &t, iters, Some(waits), load));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
