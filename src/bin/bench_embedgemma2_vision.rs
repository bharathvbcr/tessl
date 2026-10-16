//! End-to-end latency of `tessl::embedgemma2_vision` on the real checkpoint.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo run --release --bin bench_embedgemma2_vision
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,`; a workload is a count
//! of images (default `1,8`), one `encode_images` call. The images are the
//! goldens' two real processor outputs (`image0_*`: 224 x 224, 2304 real
//! patches; `image1_*`: 640 x 360, 2376), both padded to 2520 patches,
//! cycled to the count, so `8` is a ragged batch of four of each that
//! `encode_images` cuts into two forwards (`forwards` in the output).
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 3) are per workload.
//! `BENCH_PRECISION=f32|bf16` (default `f32`) picks the operand precision the
//! tower is loaded with ([`EmbedGemma2VisionModel::load_with`]: `GemmOperands::ExactF32`
//! or `Bf16`); the object's `backend` says which (`tessl-f32`, `tessl-bf16`).
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! backend, ms_min, ms_median, iters, forwards, weight_mib, loaded_mib, device_peak_mib,
//! peak_footprint_mib}`, the format `bench_paired --format workloads` reads.
//! `weight_mib` is the model's weight bytes ([`EmbedGemma2VisionModel::weight_bytes`]);
//! `loaded_mib` is the device bytes allocated once it is loaded (those, and the
//! pool's rounding; no activations). The base build (433e659), which has neither
//! `load_with` nor `weight_bytes`, runs this file as of f4a9ece, which differs
//! from this one in those two lines and the `BENCH_PRECISION` knob only.
//! stderr carries `HASH <workload> <fnv1a64>` per workload, over the bits of
//! every soft value in call order: two builds with equal hashes produced
//! bit-identical outputs.
//!
//! What is timed: one `encode_images` call from the host's pixel values to the
//! host's soft tokens: the pixel scaling and upload, every forward, the wait
//! and the read back. Not timed: the model load and the goldens' read.
//!
//! Before anything is timed every workload's soft tokens must be finite; the
//! run aborts otherwise, so a forward that wrote nothing cannot post the best
//! time.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

use common::{env_usize, peak_footprint};

const DEFAULT_WORKLOADS: &str = "1,8";
const MIB: f64 = (1u64 << 20) as f64;

/// One image's processor tensors from the goldens.
struct Golden {
    pixels: Vec<f32>,
    positions: Vec<i32>,
}

fn golden(dir: &std::path::Path, n: usize) -> Result<Golden, String> {
    let read = |name: String| read_npy(&dir.join(&name)).map_err(|e| format!("{name}: {e}"));
    let pixels = read(format!("image{n}_pixel_values.npy"))?;
    let positions = read(format!("image{n}_image_position_ids.npy"))?;
    let positions = positions
        .data_i64
        .as_ref()
        .ok_or("image position ids are not i64")?
        .iter()
        .map(|&v| i32::try_from(v).map_err(|_| format!("position {v} exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Golden {
        pixels: pixels.f32_slice()?.to_vec(),
        positions,
    })
}

fn main() -> Result<(), String> {
    let snap = PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT").map_err(|_| "EMBEDGEMMA2_SNAPSHOT must name the snapshot directory")?,
    );
    let refs = PathBuf::from(
        std::env::var("EMBEDGEMMA2_MM_REF_DIR")
            .map_err(|_| "EMBEDGEMMA2_MM_REF_DIR must name the make_mm_reference.py output directory")?,
    );
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| DEFAULT_WORKLOADS.into());
    let counts = spec
        .split(',')
        .map(|w| {
            w.trim()
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("workload {w:?} is not a positive image count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    let cfg = VisionConfig::from_config_file(&snap.join("config.json"))?;
    let st = SafeTensors::open(&snap.join("model.safetensors"))?;
    let goldens = [golden(&refs, 0)?, golden(&refs, 1)?];
    let rt = GpuRuntime::new_inference()?;
    let (operands, backend) = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("f32") => (GemmOperands::ExactF32, "tessl-f32"),
        Ok("bf16") => (GemmOperands::Bf16, "tessl-bf16"),
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32 or bf16")),
    };
    let model = EmbedGemma2VisionModel::load_with(&rt, &st, cfg, operands)?;
    let weight_mib = model.weight_bytes() as f64 / MIB;
    let loaded = rt.current_allocated_bytes();
    let loaded_mib = loaded as f64 / MIB;

    let mut rows = Vec::new();
    for &n in &counts {
        let batch: Vec<ImageInput<'_>> = (0..n)
            .map(|i| {
                let g = &goldens[i % goldens.len()];
                ImageInput {
                    pixel_values: &g.pixels,
                    position_ids: &g.positions,
                }
            })
            .collect();
        rt.reset_peak_allocated_bytes();
        let out = model.encode_images(&batch, false)?;
        if let Some((i, _)) = out
            .soft
            .iter()
            .enumerate()
            .find(|(_, s)| s.is_empty() || !s.iter().all(|v| v.is_finite()))
        {
            return Err(format!("{n} images: image {i}'s soft tokens are empty or not finite"));
        }
        // FNV-1a over every soft value's bits, images in call order: equal
        // hashes across two builds are bit-identical outputs.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for v in out.soft.iter().flatten() {
            for byte in v.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        eprintln!("HASH {n}img {hash:016x}");
        for _ in 0..warmup {
            model.encode_images(&batch, false)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode_images(&batch, false)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let device_peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{n}img\",\"backend\":\"{backend}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"forwards\":{},\"weight_mib\":{weight_mib:.1},\"loaded_mib\":{loaded_mib:.1},\"device_peak_mib\":{:.1},\"peak_footprint_mib\":{:.1}}}",
            ms[0],
            ms[ms.len() / 2],
            out.forwards,
            device_peak as f64 / MIB,
            peak_footprint()? as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
