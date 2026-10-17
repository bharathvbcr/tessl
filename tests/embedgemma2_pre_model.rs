//! Native preprocessing on the real checkpoint: a raw image and a raw clip, no
//! Python processor, through `tessl::embedgemma2_pre` and the towers to the
//! embedding, against sentence-transformers' (transformers 5.19, fp32, sdpa,
//! CPU). Opt-in:
//!
//! ```text
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_pre_model -- --ignored --test-threads=1
//! ```
//!
//! `make_mm_reference.py` saves each input's raw form (`image{n}_raw.npy`,
//! `audio{n}_raw.npy`) beside what the processor made of it. A missing variable
//! or file fails the test rather than skipping it.
//!
//! * `native_pixels_and_features_equal_the_processors`: no device. The pixels of
//!   both images and the positions are equal to the processor's bit for bit; the
//!   log-mel features of both clips are within `AUDIO_ABS` of its (measured in
//!   `tests/embedgemma2_pre.rs`), with the same mask; the config is read from the
//!   snapshot's own `processor_config.json`.
//! * `raw_inputs_through_the_towers_match_sentence_transformers`: the four raw
//!   inputs, preprocessed natively, encoded in one call, each row within `1e-4`
//!   max abs of `emb_f32.npy` and at cosine `0.99999` or better: the bounds of
//!   `tests/embedgemma2_e2e_model.rs`, which feeds the processor's tensors.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::{preprocess_audio, preprocess_image, PreConfig, PreparedImage, RawImage};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;

const AUDIO_ABS: f64 = 2e-6;
const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const IMAGES: [&str; 2] = ["image0_", "image1_"];
const CLIPS: [&str; 2] = ["audio0_", "audio1_"];

fn env_dir(var: &str, what: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} must name {what}")))
}

fn npy(name: &str) -> tessl::npy::NpyArray {
    let p = env_dir("EMBEDGEMMA2_MM_REF_DIR", "make_mm_reference.py's (sdpa) output").join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_mm_reference.py)", p.display()))
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

fn config() -> PreConfig {
    PreConfig::from_snapshot(&env_dir(
        "EMBEDGEMMA2_SNAPSHOT",
        "the google/embeddinggemma-2 snapshot directory",
    ))
    .expect("the snapshot's processor_config.json")
}

/// The raw image of reference input `prefix`, preprocessed natively.
fn native_image(cfg: &PreConfig, prefix: &str) -> PreparedImage {
    let raw = npy(&format!("{prefix}raw.npy"));
    assert_eq!(raw.shape.len(), 3, "{prefix} raw {:?}", raw.shape);
    assert_eq!(raw.shape[2], 3);
    let bytes: Vec<u8> = raw
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&v| u8::try_from(v).unwrap())
        .collect();
    preprocess_image(&cfg.image, &RawImage::rgb(raw.shape[1], raw.shape[0], &bytes)).unwrap()
}

/// The raw clip of reference input `prefix`, preprocessed natively.
fn native_clip(cfg: &PreConfig, prefix: &str) -> AudioInput {
    let raw = npy(&format!("{prefix}raw.npy"));
    preprocess_audio(&cfg.audio, raw.f32_slice().unwrap(), 16000).unwrap()
}

#[test]
#[ignore]
fn native_pixels_and_features_equal_the_processors() {
    let cfg = config();
    for prefix in IMAGES {
        let got = native_image(&cfg, prefix);
        let px = npy(&format!("{prefix}pixel_values.npy"));
        let pos = npy(&format!("{prefix}image_position_ids.npy"));
        let want_px = px.f32_slice().unwrap();
        assert_eq!(got.pixel_values.len(), want_px.len(), "{prefix} pixel count");
        let bad = got
            .pixel_values
            .iter()
            .zip(want_px)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let want_pos: Vec<i32> = pos
            .i64_slice()
            .unwrap()
            .iter()
            .map(|&v| i32::try_from(v).unwrap())
            .collect();
        eprintln!(
            "{prefix}: {} patches, resized {:?}, {bad} of {} pixel values differ",
            got.patches,
            got.resized,
            want_px.len()
        );
        assert_eq!(bad, 0, "{prefix}: pixel_values differ from the processor's");
        assert_eq!(got.position_ids, want_pos, "{prefix}: positions");
    }
    for prefix in CLIPS {
        let got = native_clip(&cfg, prefix);
        let feats = npy(&format!("{prefix}input_features.npy"));
        let mask: Vec<bool> = npy(&format!("{prefix}input_features_mask.npy"))
            .i64_slice()
            .unwrap()
            .iter()
            .map(|&m| m != 0)
            .collect();
        assert_eq!(got.mask, mask, "{prefix}: mask");
        let err = max_abs(&got.input_features, feats.f32_slice().unwrap());
        eprintln!("{prefix}: {} frames, features max abs {err:.3e}", got.mask.len());
        assert!(
            err <= AUDIO_ABS,
            "{prefix}: features max abs {err:.3e} (bound {AUDIO_ABS:e})"
        );
    }
}

#[test]
#[ignore]
fn raw_inputs_through_the_towers_match_sentence_transformers() {
    let snap = env_dir("EMBEDGEMMA2_SNAPSHOT", "the google/embeddinggemma-2 snapshot directory");
    let cfg = config();
    let images: Vec<PreparedImage> = IMAGES.iter().map(|p| native_image(&cfg, p)).collect();
    let clips: Vec<AudioInput> = CLIPS.iter().map(|p| native_clip(&cfg, p)).collect();
    let items: Vec<MmItem<'_>> = images
        .iter()
        .map(|i| MmItem::Image(i.input()))
        .chain(clips.iter().map(MmItem::Audio))
        .collect();
    let names: Vec<&str> = IMAGES.iter().chain(CLIPS.iter()).copied().collect();

    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::ExactF32).unwrap();
        let dim = m.text().config().embedding_dim as usize;
        let out = m.encode(&items, None).unwrap();
        assert_eq!(out.embeddings.len(), items.len() * dim);
        let mut failures = Vec::new();
        for (k, name) in names.iter().enumerate() {
            let want = npy(&format!("{name}emb_f32.npy"));
            let want = want.f32_slice().unwrap();
            assert_eq!(want.len(), dim, "{name} emb_f32");
            let got = &out.embeddings[k * dim..(k + 1) * dim];
            let (err, cos) = (max_abs(got, want), cosine(got, want));
            eprintln!("{name}: raw input to embedding, max abs {err:.3e}, cosine {cos:.9}");
            if err > EMB_ABS || cos < EMB_COS {
                failures.push(format!(
                    "{name}: max abs {err:.3e} (bound {EMB_ABS:e}), cosine {cos:.9} (bound {EMB_COS})"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "raw input to embedding out of bounds:\n{}",
            failures.join("\n")
        );
    });
}
