//! Media files on the real checkpoint: a file path through `tessl::media`
//! (ImageIO / AudioToolbox decoding), `tessl::embedgemma2_pre` and the towers to
//! the embedding, against sentence-transformers' (transformers 5.x, fp32, sdpa,
//! CPU). Opt-in:
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_media_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MEDIA_REF_DIR=/path/to/target/media_e2e_ref \
//!   cargo test --release --test media_e2e_model -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_MEDIA_REF_DIR` defaults to `target/media_e2e_ref`. A missing
//! variable or file fails the test rather than skipping it.
//!
//! The five inputs and what the reference was given are listed in
//! `make_media_reference.py`. Two of them decode to exactly what the reference
//! decoded (`photo.png`: lossless, same bytes; `clip16.wav`: 16 kHz mono PCM16,
//! same floats), and hold to the bounds of `tests/embedgemma2_e2e_model.rs`:
//! `1e-4` max abs and cosine `0.99999` per row. The other three differ in the
//! decode by a measured amount (a JPEG IDCT and chroma upsampler, a different
//! resampler, a different AAC decoder), and their bounds are the measurements
//! printed by this test, rounded up.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::{preprocess_audio, preprocess_image, PreConfig, PreparedImage, RawImage};
use tessl::gemm::GemmOperands;
use tessl::media::audio::{decode_audio_file, AudioOptions};
use tessl::media::image::{decode_image_file, ImageOptions};
use tessl::npy::read_npy;

/// `(file, max abs bound, cosine bound)`.
const CASES: [(&str, f64, f64); 5] = [
    ("photo.png", 1e-4, 0.99999),
    ("clip16.wav", 1e-4, 0.99999),
    ("photo.jpg", 1e-2, 0.998),
    ("clip44.wav", 2e-2, 0.99),
    ("clip44.m4a", 2e-2, 0.99),
];

fn env_dir(var: &str, what: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} must name {what}")))
}

fn ref_dir() -> PathBuf {
    std::env::var("EMBEDGEMMA2_MEDIA_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/media_e2e_ref"))
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

enum Prepared {
    Image(PreparedImage),
    Audio(AudioInput),
}

fn prepare(cfg: &PreConfig, name: &str) -> Prepared {
    let path = ref_dir().join(name);
    if name.ends_with(".png") || name.ends_with(".jpg") {
        let d = decode_image_file(&path, &ImageOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        Prepared::Image(
            preprocess_image(&cfg.image, &RawImage::rgb(d.width, d.height, &d.data))
                .unwrap_or_else(|e| panic!("{name}: {e}")),
        )
    } else {
        let a = decode_audio_file(&path, &AudioOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        Prepared::Audio(
            preprocess_audio(&cfg.audio, &a.samples, a.sample_rate).unwrap_or_else(|e| panic!("{name}: {e}")),
        )
    }
}

#[test]
#[ignore]
fn media_files_through_the_towers_match_sentence_transformers() {
    let snap = env_dir("EMBEDGEMMA2_SNAPSHOT", "the google/embeddinggemma-2 snapshot directory");
    let cfg = PreConfig::from_snapshot(&snap).expect("the snapshot's processor_config.json");
    let prepared: Vec<Prepared> = CASES.iter().map(|(name, ..)| prepare(&cfg, name)).collect();
    let items: Vec<MmItem<'_>> = prepared
        .iter()
        .map(|p| match p {
            Prepared::Image(i) => MmItem::Image(i.input()),
            Prepared::Audio(a) => MmItem::Audio(a),
        })
        .collect();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::ExactF32).unwrap();
        let dim = m.text().config().embedding_dim as usize;
        let out = m.encode(&items, None).unwrap();
        assert_eq!(out.embeddings.len(), items.len() * dim);
        let mut failures = Vec::new();
        for (k, &(name, abs_bound, cos_bound)) in CASES.iter().enumerate() {
            let p = ref_dir().join(format!("{name}.emb_f32.npy"));
            let want = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run make_media_reference.py)", p.display()));
            let want = want.f32_slice().unwrap();
            assert_eq!(want.len(), dim, "{name} emb_f32");
            let got = &out.embeddings[k * dim..(k + 1) * dim];
            let (err, cos) = (max_abs(got, want), cosine(got, want));
            eprintln!("{name}: file to embedding, max abs {err:.3e} (bound {abs_bound:e}), cosine {cos:.9} (bound {cos_bound})");
            if err > abs_bound || cos < cos_bound {
                failures.push(format!(
                    "{name}: max abs {err:.3e} (bound {abs_bound:e}), cosine {cos:.9} (bound {cos_bound})"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "file to embedding out of bounds:\n{}",
            failures.join("\n")
        );
    });
}
