//! End-to-end latency of `tessl::embedgemma2_mm::EmbedGemma2Multimodal::encode`
//! on the real checkpoint: the call that takes texts, images and clips.
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//! EMBEDGEMMA2_REF_DIR=/path/to/target/embedgemma2_ref \
//!   cargo run --release --bin bench_embedgemma2_mm
//! ```
//!
//! `BENCH_WORKLOADS` lists workloads separated by `,` (default all); each is
//! one `encode` call over the goldens of the reference directories:
//!
//! * `mixed_full`: image 0, clip 0, the eight reference texts (10 to 6147
//!   tokens), image 1, clip 1: the batch of `tests/embedgemma2_multimodal_model.rs`
//!   (the text side needs several forwards);
//! * `mixed_short`: the same twelve items with every text cut to its first 48
//!   ids, so the towers are most of the call;
//! * `media_only`: the two images and the two clips.
//!
//! `BENCH_ITERS` (default 10) and `BENCH_WARMUP` (default 3) are per workload.
//! `BENCH_PRECISION=f32|bf16` (default `f32`) is the operand precision of the
//! text encoder and the vision tower ([`GemmOperands`]).
//!
//! What is timed: one `encode` call from host ids, pixels and features to host
//! embeddings. Not timed: the model load and the goldens' read.
//!
//! Prints one JSON array on stdout, one object per workload: `{workload,
//! ms_min, ms_median, iters, items, sync_waits, forwards, emb_err_max,
//! device_peak_mib}`, the format `bench_paired --format workloads` reads.
//! `sync_waits` is the host waits of one call (`infer_trace`, counted on a
//! separate call before the timed ones). stderr carries `HASH <workload>
//! <fnv1a64>` per workload over the bits of every embedding: two builds with
//! equal hashes produced bit-identical outputs. Before anything is timed every
//! embedding must be finite and, for `mixed_full`, the eight text rows and the
//! four media rows within `1e-4` max abs of the references (the bound of
//! `tests/embedgemma2_multimodal_model.rs`); `emb_err_max` is the observed
//! worst (`null` for the workloads without a reference).
//!
//! The binary uses only the library's public `EmbedGemma2Multimodal::load` and
//! `encode`, so it builds unchanged against older revisions of the crate.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::npy::read_npy;
use tessl::GpuRuntime;

use common::env_usize;

const MIB: f64 = (1u64 << 20) as f64;
const WORKLOADS: [&str; 3] = ["mixed_full", "mixed_short", "media_only"];
const EMB_BOUND: f64 = 1e-4;
const SHORT_TOKENS: usize = 48;

fn env_dir(var: &str, what: &str) -> Result<PathBuf, String> {
    std::env::var(var)
        .map(PathBuf::from)
        .map_err(|_| format!("{var} must name {what}"))
}

fn f32s(dir: &std::path::Path, name: &str) -> Result<Vec<f32>, String> {
    Ok(read_npy(&dir.join(name))?.f32_slice()?.to_vec())
}

fn i64s(dir: &std::path::Path, name: &str) -> Result<Vec<i64>, String> {
    Ok(read_npy(&dir.join(name))?.i64_slice()?.to_vec())
}

fn fnv1a(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

struct Goldens {
    texts: Vec<Vec<u32>>,
    /// `[texts.len(), dim]`.
    text_emb: Vec<f32>,
    images: Vec<(Vec<f32>, Vec<i32>)>,
    clips: Vec<AudioInput>,
    /// Image 0, image 1, clip 0, clip 1.
    media_emb: Vec<Vec<f32>>,
}

impl Goldens {
    fn image(&self, k: usize) -> MmItem<'_> {
        MmItem::Image(ImageInput {
            pixel_values: &self.images[k].0,
            position_ids: &self.images[k].1,
        })
    }

    /// Image 0, clip 0, `texts`, image 1, clip 1.
    fn layout<'a>(&'a self, texts: &'a [Vec<u32>]) -> Vec<MmItem<'a>> {
        let mut items = vec![self.image(0), MmItem::Audio(&self.clips[0])];
        items.extend(texts.iter().map(|t| MmItem::Text(t)));
        items.extend([self.image(1), MmItem::Audio(&self.clips[1])]);
        items
    }
}

fn goldens() -> Result<Goldens, String> {
    let rd = env_dir("EMBEDGEMMA2_REF_DIR", "make_reference.py's output")?;
    let md = env_dir("EMBEDGEMMA2_MM_REF_DIR", "make_mm_reference.py's (sdpa) output")?;
    let ids = i64s(&rd, "ids.npy")?;
    let off = i64s(&rd, "offsets.npy")?;
    let texts = off
        .windows(2)
        .map(|w| {
            ids[w[0] as usize..w[1] as usize]
                .iter()
                .map(|&i| u32::try_from(i).map_err(|_| format!("token id {i}")))
                .collect::<Result<Vec<u32>, String>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let images = ["image0_", "image1_"]
        .iter()
        .map(|p| {
            let pos = i64s(&md, &format!("{p}image_position_ids.npy"))?
                .into_iter()
                .map(|v| i32::try_from(v).map_err(|_| format!("position {v}")))
                .collect::<Result<Vec<i32>, String>>()?;
            Ok((f32s(&md, &format!("{p}pixel_values.npy"))?, pos))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let clips = ["audio0_", "audio1_"]
        .iter()
        .map(|p| {
            Ok(AudioInput {
                input_features: f32s(&md, &format!("{p}input_features.npy"))?,
                mask: i64s(&md, &format!("{p}input_features_mask.npy"))?
                    .into_iter()
                    .map(|m| m != 0)
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let media_emb = ["image0_", "image1_", "audio0_", "audio1_"]
        .iter()
        .map(|p| f32s(&md, &format!("{p}emb_f32.npy")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Goldens {
        texts,
        text_emb: f32s(&rd, "emb_f32.npy")?,
        images,
        clips,
        media_emb,
    })
}

fn main() -> Result<(), String> {
    let snap = env_dir("EMBEDGEMMA2_SNAPSHOT", "the google/embeddinggemma-2 snapshot directory")?;
    let spec = std::env::var("BENCH_WORKLOADS").unwrap_or_else(|_| WORKLOADS.join(","));
    let names: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).collect();
    if let Some(bad) = names.iter().find(|n| !WORKLOADS.contains(&n.as_str())) {
        return Err(format!("unknown workload {bad:?}; expected one of {WORKLOADS:?}"));
    }
    let iters = env_usize("BENCH_ITERS", 10, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;
    let operands = match std::env::var("BENCH_PRECISION").as_deref() {
        Err(_) | Ok("f32") => GemmOperands::ExactF32,
        Ok("bf16") => GemmOperands::Bf16,
        Ok(other) => return Err(format!("BENCH_PRECISION={other:?}: expected f32 or bf16")),
    };

    let g = goldens()?;
    let rt = GpuRuntime::new_inference()?;
    let model = EmbedGemma2Multimodal::load(&rt, &snap, operands)?;
    let dim = model.text().config().embedding_dim as usize;
    let loaded = rt.current_allocated_bytes();

    let short: Vec<Vec<u32>> = g
        .texts
        .iter()
        .map(|t| t[..t.len().min(SHORT_TOKENS)].to_vec())
        .collect();

    let mut rows = Vec::new();
    for name in &names {
        let items: Vec<MmItem<'_>> = match name.as_str() {
            "mixed_full" => g.layout(&g.texts),
            "mixed_short" => g.layout(&short),
            _ => vec![
                g.image(0),
                g.image(1),
                MmItem::Audio(&g.clips[0]),
                MmItem::Audio(&g.clips[1]),
            ],
        };
        rt.reset_peak_allocated_bytes();
        let out = model.encode(&items, None)?;
        if out.embeddings.len() != items.len() * dim || !out.embeddings.iter().all(|v| v.is_finite()) {
            return Err(format!("{name}: the embeddings are missing or not finite"));
        }
        let err = if name == "mixed_full" {
            let row = |k: usize| &out.embeddings[k * dim..(k + 1) * dim];
            let n = g.texts.len();
            let mut worst = max_abs(&out.embeddings[2 * dim..(2 + n) * dim], &g.text_emb);
            // Image 0, clip 0, ..., image 1, clip 1: the reference order is
            // [image0, image1, audio0, audio1].
            for (k, at) in [0, 2 + n, 1, 3 + n].into_iter().enumerate() {
                worst = worst.max(max_abs(row(at), &g.media_emb[k]));
            }
            if worst.is_nan() || worst > EMB_BOUND {
                return Err(format!(
                    "{name}: an embedding is {worst:.3e} from the reference (bound {EMB_BOUND:e})"
                ));
            }
            format!("{worst:e}")
        } else {
            "null".to_string()
        };
        eprintln!("HASH {name} {:016x}", fnv1a(&out.embeddings));
        infer_trace::set_enabled(true);
        let s0 = infer_trace::snapshot();
        model.encode(&items, None)?;
        let waits = infer_trace::snapshot().since(&s0).sync_waits;
        infer_trace::set_enabled(false);
        for _ in 0..warmup {
            model.encode(&items, None)?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            model.encode(&items, None)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        let peak = rt.peak_allocated_bytes().saturating_sub(loaded);
        rows.push(format!(
            "{{\"workload\":\"{name}\",\"ms_min\":{:.4},\"ms_median\":{:.4},\"iters\":{iters},\"items\":{},\"sync_waits\":{waits},\"forwards\":{},\"emb_err_max\":{err},\"device_peak_mib\":{:.1}}}",
            ms[0],
            ms[ms.len() / 2],
            items.len(),
            out.forwards,
            peak as f64 / MIB,
        ));
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
