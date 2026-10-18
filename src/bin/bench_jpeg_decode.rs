//! Time the native JPEG decoder (`tessl::media::jpeg`, reached through
//! `decode_image`) against the ImageIO path it replaced (`decode_image_imageio`)
//! on the same files. CPU only, no device, no checkpoint.
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python bench/make_jpeg_bench_inputs.py    # once
//! cargo run --release --bin bench_jpeg_decode
//! BENCH_JPEG_DIR=/path/to/files BENCH_ITERS=20 cargo run --release --bin bench_jpeg_decode
//! ```
//!
//! `BENCH_JPEG_DIR` (default `target/jpeg_bench`) holds the files that
//! `bench/make_jpeg_bench_inputs.py` writes. Each `.jpg` is first decoded by both
//! paths once, and the run stops if either fails. Then the two paths are timed
//! interleaved, one call of each in turn per iteration, so a change in machine
//! load lands on both; `BENCH_ITERS` (default 15) iterations after
//! `BENCH_WARMUP` (default 3) unrecorded ones.
//!
//! Prints a table, then one JSON array on the last stdout line:
//! `{workload, megapixels, native_ms_min, native_ms_median, imageio_ms_min,
//! imageio_ms_median, iters}`. The numbers are minimum-of-N wall times on
//! whatever else the machine is doing: **indicative, not a paired comparison**.
//! The two paths do not produce the same pixels (ImageIO differs from
//! libjpeg-turbo by up to 43 levels per channel); the native one is the reference's.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tessl::media::image::{decode_image, decode_image_imageio, ImageOptions};

struct Row {
    workload: String,
    megapixels: f64,
    native: (f64, f64),
    imageio: (f64, f64),
    iters: usize,
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

fn min_median(mut ms: Vec<f64>) -> (f64, f64) {
    ms.sort_by(f64::total_cmp);
    (ms[0], ms[ms.len() / 2])
}

fn run() -> Result<(), String> {
    let iters = env_count("BENCH_ITERS", 15, 1)?;
    let warmup = env_count("BENCH_WARMUP", 3, 0)?;
    let dir = std::env::var("BENCH_JPEG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/jpeg_bench"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e} (run bench/make_jpeg_bench_inputs.py)", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg")))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "{}: no .jpg files (run bench/make_jpeg_bench_inputs.py)",
            dir.display()
        ));
    }
    let opts = ImageOptions::default();
    let mut rows = Vec::new();
    for path in &files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string();
        let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
        let native = decode_image(&bytes, &opts).map_err(|e| format!("{name}: native: {e}"))?;
        let imageio = decode_image_imageio(&bytes, &opts).map_err(|e| format!("{name}: ImageIO: {e}"))?;
        if (native.width, native.height) != (imageio.width, imageio.height) {
            return Err(format!("{name}: the two paths disagree on the size"));
        }
        let mut a = Vec::with_capacity(iters);
        let mut b = Vec::with_capacity(iters);
        for i in 0..warmup + iters {
            let t = Instant::now();
            black_box(decode_image(black_box(&bytes), &opts).map_err(|e| e.to_string())?);
            let na = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            black_box(decode_image_imageio(black_box(&bytes), &opts).map_err(|e| e.to_string())?);
            let nb = t.elapsed().as_secs_f64() * 1e3;
            if i >= warmup {
                a.push(na);
                b.push(nb);
            }
        }
        rows.push(Row {
            workload: name,
            megapixels: (native.width * native.height) as f64 / 1e6,
            native: min_median(a),
            imageio: min_median(b),
            iters,
        });
    }
    println!(
        "{:30} {:>6} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "workload", "MP", "native min", "native med", "ImageIO min", "ImageIO med", "native/ImageIO(min)"
    );
    for r in &rows {
        println!(
            "{:30} {:>6.2} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>8.2}x",
            r.workload,
            r.megapixels,
            r.native.0,
            r.native.1,
            r.imageio.0,
            r.imageio.1,
            r.native.0 / r.imageio.0
        );
    }
    let json: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"workload\":\"{}\",\"megapixels\":{:.4},\"native_ms_min\":{:.4},\"native_ms_median\":{:.4},\
                 \"imageio_ms_min\":{:.4},\"imageio_ms_median\":{:.4},\"iters\":{}}}",
                r.workload, r.megapixels, r.native.0, r.native.1, r.imageio.0, r.imageio.1, r.iters
            )
        })
        .collect();
    println!("[{}]", json.join(","));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bench_jpeg_decode: {e}");
        std::process::exit(1);
    }
}
