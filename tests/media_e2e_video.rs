//! Video files on the real checkpoint: an MP4 path through `tessl::media::video`
//! (VideoToolbox decode, libswscale-exact colour conversion, the processor's
//! frame sampling), `tessl::embedgemma2_pre` and the towers to the embedding,
//! against sentence-transformers' (transformers 5.x, fp32, sdpa, CPU). Opt-in:
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_video_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MEDIA_REF_DIR=/path/to/target/media_e2e_ref \
//!   cargo test --release --test media_e2e_video -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_MEDIA_REF_DIR` defaults to `target/media_e2e_ref`. A missing
//! variable or file fails the test rather than skipping it.
//!
//! The inputs are the committed files of `tests/fixtures/media_video/`. The
//! reference was given the frames the reference loader would produce from the
//! same file (the `ffmpeg` CLI's decode, which equals torchcodec's, with
//! torchcodec's exact-mode frame count, duration and frame rate:
//! `make_video_reference.py`), and tessl's frames equal those byte for byte
//! (`tests/media_video_exact.rs`), so the bounds are the ones of
//! `tests/embedgemma2_e2e_model.rs`: `1e-4` max abs and cosine `0.99999` per
//! row. One case runs with `add_timestamps` on, which puts each frame's
//! `mm:ss` label in the sequence.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::{PreConfig, PreparedImage, PreparedVideo};
use tessl::embedgemma2_tok::Tokenizer;
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::media::video::{decode_video_file, preprocess_decoded_video, ColorPath, VideoOptions};
use tessl::npy::read_npy;

const ABS_BOUND: f64 = 1e-4;
const COS_BOUND: f64 = 0.99999;

/// `(case, fixture, timestamps on)`, the cases of `make_video_reference.py`.
const CASES: [(&str, &str, bool); 4] = [
    ("e2e_video", "e2e_video", false),
    ("e2e_video_ts", "e2e_video", true),
    ("rot90", "h264_709_tv_64x48_rot90", false),
    ("hevc_pc", "hevc_601_pc_34x26", false),
];

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot"),
    )
}

fn ref_dir() -> PathBuf {
    std::env::var("EMBEDGEMMA2_MEDIA_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/media_e2e_ref"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/media_video/{name}.mp4"))
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

/// The snapshot's processor config, with `add_timestamps` as asked.
fn config(snap: &std::path::Path, timestamps: bool) -> PreConfig {
    let text = std::fs::read_to_string(snap.join("processor_config.json")).expect("processor_config.json");
    assert_eq!(
        text.matches(r#""add_timestamps": false"#).count(),
        1,
        "the snapshot's setting"
    );
    let want = if timestamps { "true" } else { "false" };
    PreConfig::from_json(&text.replace(r#""add_timestamps": false"#, &format!(r#""add_timestamps": {want}"#)))
        .expect("processor config")
}

fn inputs(v: &PreparedVideo) -> Vec<ImageInput<'_>> {
    v.frames.iter().map(PreparedImage::input).collect()
}

#[test]
#[ignore]
fn video_files_through_the_towers_match_sentence_transformers() {
    let snap = snapshot();
    let tok = Tokenizer::from_snapshot(&snap).expect("the snapshot's tokenizer");
    let prepared: Vec<(PreparedVideo, bool)> = CASES
        .iter()
        .map(|&(case, file, stamps)| {
            let cfg = config(&snap, stamps);
            let decoded = decode_video_file(&fixture(file), &cfg.video, &VideoOptions::default())
                .unwrap_or_else(|e| panic!("{case}: {e}"));
            let p = ref_dir().join(format!("video_{case}.indices.npy"));
            let want = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run make_video_reference.py)", p.display()));
            let want: Vec<usize> = want
                .i64_slice()
                .unwrap()
                .iter()
                .map(|&i| usize::try_from(i).unwrap())
                .collect();
            assert_eq!(decoded.indices, want, "{case}: the frames the processor samples");
            (
                preprocess_decoded_video(&cfg.video, &decoded).unwrap_or_else(|e| panic!("{case}: {e}")),
                stamps,
            )
        })
        .collect();
    // A control: the first case again with AVFoundation's own colour conversion
    // (not libswscale's), to show the bounds can tell the two paths apart.
    let control = {
        let options = VideoOptions {
            color: ColorPath::AvFoundation,
            ..VideoOptions::default()
        };
        let cfg = config(&snap, false);
        let decoded = decode_video_file(&fixture(CASES[0].1), &cfg.video, &options).expect("control decode");
        preprocess_decoded_video(&cfg.video, &decoded).expect("control preprocess")
    };
    let mut prepared = prepared;
    prepared.push((control, false));
    let frames: Vec<Vec<ImageInput<'_>>> = prepared.iter().map(|(v, _)| inputs(v)).collect();
    let items: Vec<MmItem<'_>> = prepared
        .iter()
        .zip(&frames)
        .map(|((v, stamps), f)| {
            if *stamps {
                MmItem::TimestampedVideo {
                    frames: f,
                    timestamps: &v.timestamps,
                    tokenizer: &tok,
                }
            } else {
                MmItem::Video(f)
            }
        })
        .collect();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::ExactF32).unwrap();
        let dim = m.text().config().embedding_dim as usize;
        let out = m.encode(&items, None).unwrap();
        assert_eq!(out.embeddings.len(), items.len() * dim);
        let mut failures = Vec::new();
        for (k, &(case, ..)) in CASES.iter().enumerate() {
            let p = ref_dir().join(format!("video_{case}.emb_f32.npy"));
            let want = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run make_video_reference.py)", p.display()));
            let want = want.f32_slice().unwrap();
            assert_eq!(want.len(), dim, "{case} emb_f32");
            let got = &out.embeddings[k * dim..(k + 1) * dim];
            let (err, cos) = (max_abs(got, want), cosine(got, want));
            eprintln!(
                "{case}: file to embedding, max abs {err:.3e} (bound {ABS_BOUND:e}), cosine {cos:.9} (bound {COS_BOUND})"
            );
            if err > ABS_BOUND || cos < COS_BOUND {
                failures.push(format!("{case}: max abs {err:.3e}, cosine {cos:.9}"));
            }
        }
        let want = read_npy(&ref_dir().join(format!("video_{}.emb_f32.npy", CASES[0].0))).unwrap();
        let got = &out.embeddings[CASES.len() * dim..(CASES.len() + 1) * dim];
        let (err, cos) = (
            max_abs(got, want.f32_slice().unwrap()),
            cosine(got, want.f32_slice().unwrap()),
        );
        eprintln!("control, AVFoundation's own colour conversion: max abs {err:.3e}, cosine {cos:.9}");
        assert!(
            err > ABS_BOUND,
            "the bound no longer tells libswscale's conversion from AVFoundation's: {err:.3e}"
        );
        assert!(
            failures.is_empty(),
            "file to embedding out of bounds:\n{}",
            failures.join("\n")
        );
    });
}
