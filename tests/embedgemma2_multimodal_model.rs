//! `EmbedGemma2Multimodal` on the real checkpoint: the text encoder and both
//! towers from one snapshot directory, a batch of eight texts, two images and
//! two clips in one call, against sentence-transformers' fp32 embeddings
//! (transformers 5.19, sdpa, CPU). Opt-in:
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_reference.py      # once
//! ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//! EMBEDGEMMA2_REF_DIR=/path/to/target/embedgemma2_ref \
//!   cargo test --release --test embedgemma2_multimodal_model -- --ignored --test-threads=1
//! ```
//!
//! A missing variable or file fails the test rather than skipping it.
//!
//! Bounds, fixed before the first run (`tests/embedgemma2_e2e_model.rs`'s and
//! `tests/embedgemma2_model.rs`'s):
//!
//! * every row of the mixed call within `1e-4` max abs of `emb_f32.npy` and at
//!   cosine `0.99999` or better;
//! * each image and clip alone vs inside the mixed call within `1e-5` (the
//!   mixed call needs several text forwards, since one text has 6147 tokens);
//! * host waits: the mixed call and a call of media alone make exactly 2, a
//!   call of texts alone exactly 1;
//! * `GemmOperands::Bf16` for the text encoder (towers unchanged): the four
//!   media rows at cosine `0.9999` or better against the fp32 reference, the
//!   floor `tests/embedgemma2_mm_model.rs` holds a bf16 forward to.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::npy::read_npy;

const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const BATCH_ABS: f64 = 1e-5;
const BF16_COS: f64 = 0.9999;
const IMAGES: [&str; 2] = ["image0_", "image1_"];
const CLIPS: [&str; 2] = ["audio0_", "audio1_"];

fn env_dir(var: &str, what: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} must name {what}")))
}

fn npy(dir: &str, name: &str) -> tessl::npy::NpyArray {
    let base = if dir == "mm" {
        env_dir("EMBEDGEMMA2_MM_REF_DIR", "make_mm_reference.py's (sdpa) output")
    } else {
        env_dir("EMBEDGEMMA2_REF_DIR", "make_reference.py's output")
    };
    let p = base.join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_*reference.py)", p.display()))
}

fn f32s(dir: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let a = npy(dir, name);
    (a.shape.clone(), a.f32_slice().expect("f32 reference").to_vec())
}

fn i64s(dir: &str, name: &str) -> Vec<i64> {
    npy(dir, name).i64_slice().expect("i64 reference").to_vec()
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

fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

#[test]
#[ignore]
fn one_call_over_texts_images_and_clips_matches_sentence_transformers() {
    let snap = env_dir("EMBEDGEMMA2_SNAPSHOT", "the google/embeddinggemma-2 snapshot directory");

    // Eight texts (make_reference.py), two images and two clips (make_mm_reference.py).
    let ids = i64s("text", "ids.npy");
    let off = i64s("text", "offsets.npy");
    let texts: Vec<Vec<u32>> = off
        .windows(2)
        .map(|w| {
            ids[w[0] as usize..w[1] as usize]
                .iter()
                .map(|&i| u32::try_from(i).unwrap())
                .collect()
        })
        .collect();
    let (shape, text_ref) = f32s("text", "emb_f32.npy");
    let dim = shape[1];
    assert_eq!(shape[0], texts.len());
    let pixels: Vec<(Vec<f32>, Vec<i32>)> = IMAGES
        .iter()
        .map(|p| {
            let (_, px) = f32s("mm", &format!("{p}pixel_values.npy"));
            let pos = i64s("mm", &format!("{p}image_position_ids.npy"))
                .into_iter()
                .map(|v| i32::try_from(v).unwrap())
                .collect();
            (px, pos)
        })
        .collect();
    let clips: Vec<AudioInput> = CLIPS
        .iter()
        .map(|p| AudioInput {
            input_features: f32s("mm", &format!("{p}input_features.npy")).1,
            mask: i64s("mm", &format!("{p}input_features_mask.npy"))
                .into_iter()
                .map(|m| m != 0)
                .collect(),
        })
        .collect();
    let media_ref: Vec<Vec<f32>> = IMAGES
        .iter()
        .chain(CLIPS.iter())
        .map(|p| f32s("mm", &format!("{p}emb_f32.npy")).1)
        .collect();
    let image_item = |k: usize| {
        MmItem::Image(ImageInput {
            pixel_values: &pixels[k].0,
            position_ids: &pixels[k].1,
        })
    };
    // Media first and last, texts between: the output order is the call's.
    let mut items: Vec<MmItem<'_>> = vec![image_item(0), MmItem::Audio(&clips[0])];
    items.extend(texts.iter().map(|t| MmItem::Text(t)));
    items.extend([image_item(1), MmItem::Audio(&clips[1])]);
    // Row of each reference input, in the order [image0, image1, audio0, audio1].
    let media_at = [0usize, 2 + texts.len(), 1, 3 + texts.len()];

    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::ExactF32).unwrap();
        assert_eq!(m.text().config().embedding_dim as usize, dim);
        let (out, t) = traced(|| m.encode(&items, None).unwrap());
        assert_eq!(out.embeddings.len(), items.len() * dim);
        eprintln!(
            "mixed call: {} text forwards, {} host waits",
            out.forwards, t.sync_waits
        );
        assert!(out.forwards >= 2, "test bug: the batch is one forward");
        assert_eq!(t.sync_waits, 2, "texts, images and clips in one call");

        let mut failures = Vec::new();
        let mut check = |what: String, got: &[f32], want: &[f32]| {
            let (err, cos) = (max_abs(got, want), cosine(got, want));
            eprintln!("{what}: max abs {err:.3e}, cosine {cos:.9}");
            if err > EMB_ABS || cos < EMB_COS {
                failures.push(format!(
                    "{what}: max abs {err:.3e} (bound {EMB_ABS:e}), cosine {cos:.9} (bound {EMB_COS})"
                ));
            }
        };
        let row = |k: usize| &out.embeddings[k * dim..(k + 1) * dim];
        for (k, name) in IMAGES.iter().chain(CLIPS.iter()).enumerate() {
            check(format!("{name} in the mixed call"), row(media_at[k]), &media_ref[k]);
        }
        for i in 0..texts.len() {
            check(
                format!("text {i} in the mixed call"),
                row(2 + i),
                &text_ref[i * dim..(i + 1) * dim],
            );
        }
        assert!(failures.is_empty(), "out of bounds:\n{}", failures.join("\n"));

        // Each media item alone, and a call of media alone.
        for (k, name) in IMAGES.iter().chain(CLIPS.iter()).enumerate() {
            let item = &items[media_at[k]];
            let alone = m.encode(std::slice::from_ref(item), None).unwrap();
            let d = max_abs(row(media_at[k]), &alone.embeddings);
            eprintln!("{name} alone vs in the mixed call: {d:.3e}");
            assert!(d <= BATCH_ABS, "{name}: alone vs mixed {d:.3e} (bound {BATCH_ABS:e})");
        }
        let media: Vec<MmItem<'_>> = media_at.iter().map(|&k| items[k]).collect();
        let (only, t) = traced(|| m.encode(&media, None).unwrap());
        assert_eq!(t.sync_waits, 2, "media alone");
        for k in 0..4 {
            let d = max_abs(&only.embeddings[k * dim..(k + 1) * dim], row(media_at[k]));
            assert!(d <= BATCH_ABS, "media row {k}: media-only call vs mixed call {d:.3e}");
        }
        let text_items: Vec<MmItem<'_>> = texts.iter().map(|t| MmItem::Text(t)).collect();
        let (txt, t) = traced(|| m.encode(&text_items, None).unwrap());
        assert_eq!(t.sync_waits, 1, "texts alone");
        for i in 0..texts.len() {
            let want = &text_ref[i * dim..(i + 1) * dim];
            assert!(
                max_abs(&txt.embeddings[i * dim..(i + 1) * dim], want) <= EMB_ABS,
                "text {i}"
            );
        }

        // A Matryoshka prefix of an image, normalized over its own columns.
        let cut = m.encode(&[image_item(0)], Some(256)).unwrap();
        let full = row(0);
        let norm = full[..256].iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let want: Vec<f32> = full[..256].iter().map(|v| (f64::from(*v) / norm) as f32).collect();
        assert!(max_abs(&cut.embeddings, &want) <= BATCH_ABS, "truncate_dim 256");
    });

    // The text encoder in bf16, the towers as they were.
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::Bf16).unwrap();
        assert_eq!(m.text().operands(), GemmOperands::Bf16);
        let media: Vec<MmItem<'_>> = vec![
            image_item(0),
            image_item(1),
            MmItem::Audio(&clips[0]),
            MmItem::Audio(&clips[1]),
        ];
        let out = m.encode(&media, None).unwrap();
        for (k, name) in IMAGES.iter().chain(CLIPS.iter()).enumerate() {
            let cos = cosine(&out.embeddings[k * dim..(k + 1) * dim], &media_ref[k]);
            eprintln!("{name} bf16 text encoder: cosine vs fp32 {cos:.9}");
            assert!(cos >= BF16_COS, "{name}: bf16 cosine {cos:.9} (bound {BF16_COS})");
        }
    });
}
