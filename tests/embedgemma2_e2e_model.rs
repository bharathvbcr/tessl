//! EmbeddingGemma 2 end to end on the real checkpoint: the processor's pixels
//! and log-mel features through tessl's vision and audio towers, their soft
//! tokens through tessl's text encoder, against sentence-transformers' final
//! embedding (transformers 5.19, fp32, sdpa, CPU).
//!
//! Each piece is held to its own goldens elsewhere: the towers' soft tokens in
//! `tests/embedgemma2_vision_model.rs` and `tests/embedgemma2_audio_model.rs`,
//! and the text encoder fed the *golden* soft tokens in
//! `tests/embedgemma2_mm_model.rs`. Only this test composes tessl's towers with
//! tessl's text encoder, so it is the one that sees a disagreement between them
//! about how many soft tokens an input has or the order they come in. Opt-in:
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_e2e_model -- --ignored --test-threads=1
//! ```
//!
//! A missing variable or file fails the test rather than skipping it.
//!
//! Checks, with bounds fixed before the first run:
//!
//! * per input, the tower's soft-token count equals the placeholder count of
//!   `ids.npy`, and the config's sequence builder lays out exactly `ids.npy`
//!   for that count;
//! * per input, the embedding of all four inputs encoded in one call within
//!   `1e-4` max abs of `emb_f32.npy` and at cosine `0.99999` or better: the
//!   text test's bounds. The towers' soft tokens are within `3.3e-5` of their
//!   largest magnitude of the goldens, and the text encoder is held to the same
//!   bound from golden soft tokens, so the composition is expected inside it.
//!
//! The text path from a string, with no Python tokenizer, is held to the text
//! goldens of `tools/embedgemma2_ref/make_reference.py` (`EMBEDGEMMA2_REF_DIR`,
//! default `target/embedgemma2_ref`; its eight texts, each with its prompt
//! name):
//!
//! * `the_rust_tokenizer_reproduces_the_reference_ids`: `ids.npy` and
//!   `offsets.npy` (what sentence-transformers fed the model) equal
//!   [`Tokenizer::encode_with_prompt`]'s ids exactly;
//! * `text_to_embedding_with_no_python_matches_sentence_transformers`:
//!   [`EmbedGemma2Model::encode_texts`] (tokenize, then encode) within the same
//!   `1e-4` and cosine `0.99999` of `emb_f32.npy`, each text alone, and the
//!   same call equals `encode` of the reference ids bit for bit.

mod common;

use std::path::PathBuf;

use common::embedgemma2_tok::{brief, first_diff, reference_texts};
use common::with_gpu;
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, MmSequence};
use tessl::embedgemma2_audio::{AudioConfig, AudioInput, EmbedGemma2Audio};
use tessl::embedgemma2_tok::Tokenizer;
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, VisionConfig};
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const PREFIX: &str = "language_model.";
const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const IMAGES: [&str; 2] = ["image0_", "image1_"];
const CLIPS: [&str; 2] = ["audio0_", "audio1_"];

fn ref_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_MM_REF_DIR")
            .expect("EMBEDGEMMA2_MM_REF_DIR must name make_mm_reference.py's (sdpa) output"),
    )
}

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory"),
    )
}

fn npy(name: &str) -> tessl::npy::NpyArray {
    let p = ref_dir().join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_mm_reference.py)", p.display()))
}

fn npy_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let a = npy(name);
    (a.shape.clone(), a.f32_slice().expect("f32 reference").to_vec())
}

fn npy_i64(name: &str) -> Vec<i64> {
    npy(name).i64_slice().expect("i64 reference").to_vec()
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

#[test]
#[ignore]
fn pixels_and_features_through_tessl_match_sentence_transformers() {
    let snap = snapshot();
    let config = snap.join("config.json");
    let text_cfg = EmbedGemma2Config::from_config_file(&config).unwrap();
    let vision_cfg = VisionConfig::from_config_file(&config).unwrap();
    let audio_cfg = AudioConfig::from_config_file(&config).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let h = text_cfg.hidden as usize;
    let dim = text_cfg.embedding_dim as usize;

    let pixels: Vec<(Vec<f32>, Vec<i32>)> = IMAGES
        .iter()
        .map(|p| {
            let (shape, px) = npy_f32(&format!("{p}pixel_values.npy"));
            assert_eq!(shape.len(), 3, "{p} pixel_values {shape:?}");
            let pos: Vec<i32> = npy_i64(&format!("{p}image_position_ids.npy"))
                .into_iter()
                .map(|v| i32::try_from(v).unwrap())
                .collect();
            (px, pos)
        })
        .collect();
    let clips: Vec<AudioInput> = CLIPS
        .iter()
        .map(|p| {
            let (shape, feats) = npy_f32(&format!("{p}input_features.npy"));
            assert_eq!(shape.len(), 3, "{p} input_features {shape:?}");
            let mask: Vec<bool> = npy_i64(&format!("{p}input_features_mask.npy"))
                .into_iter()
                .map(|m| m != 0)
                .collect();
            assert_eq!(mask.len(), shape[1], "{p} mask vs frames");
            AudioInput {
                input_features: feats,
                mask,
            }
        })
        .collect();

    with_gpu(|rt| {
        let vision = EmbedGemma2VisionModel::load(rt, &st, vision_cfg.clone()).unwrap();
        let audio = EmbedGemma2Audio::load(rt, &st, "", audio_cfg.clone()).unwrap();
        let text = EmbedGemma2Model::load(rt, &st, PREFIX, text_cfg.clone()).unwrap();

        let images: Vec<ImageInput<'_>> = pixels
            .iter()
            .map(|(px, pos)| ImageInput {
                pixel_values: px,
                position_ids: pos,
            })
            .collect();
        let v = vision.encode_images(&images, false).unwrap();
        let a = audio.encode_audio(&clips, false).unwrap();
        let soft: Vec<&[f32]> = v.soft.iter().chain(a.soft_tokens.iter()).map(Vec::as_slice).collect();

        let names: Vec<&str> = IMAGES.iter().chain(CLIPS.iter()).copied().collect();
        let mut ids: Vec<Vec<u32>> = Vec::new();
        for (k, (&name, rows)) in names.iter().zip(&soft).enumerate() {
            let want: Vec<u32> = npy_i64(&format!("{name}ids.npy"))
                .into_iter()
                .map(|i| u32::try_from(i).unwrap())
                .collect();
            assert_eq!(rows.len() % h, 0, "{name}: soft width");
            let n = rows.len() / h;
            let tm = text_cfg.special_tokens.multimodal(name).unwrap();
            let placeholders = want.iter().filter(|&&i| tm.is_placeholder(i)).count();
            assert_eq!(n, placeholders, "{name}: tower soft tokens vs placeholders in ids.npy");
            let built = if k < IMAGES.len() {
                text_cfg.image_sequence(n)
            } else {
                text_cfg.audio_sequence(n)
            }
            .unwrap();
            assert_eq!(built, want, "{name}: built ids vs ids.npy");
            ids.push(built);
        }

        let seqs: Vec<MmSequence<'_>> = ids
            .iter()
            .zip(&soft)
            .map(|(ids, soft)| MmSequence { ids, soft })
            .collect();
        let out = text.encode_multimodal(&seqs, None, false).unwrap();
        assert_eq!(out.embeddings.len(), names.len() * dim);

        let mut failures = Vec::new();
        for (k, name) in names.iter().enumerate() {
            let (shape, want) = npy_f32(&format!("{name}emb_f32.npy"));
            assert_eq!(shape, [dim], "{name} emb_f32");
            let got = &out.embeddings[k * dim..(k + 1) * dim];
            let (err, cos) = (max_abs(got, &want), cosine(got, &want));
            eprintln!("{name}: end to end max abs {err:.3e}, cosine {cos:.9}");
            if err > EMB_ABS || cos < EMB_COS {
                failures.push(format!(
                    "{name}: max abs {err:.3e} (bound {EMB_ABS:e}), cosine {cos:.9} (bound {EMB_COS})"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "end to end out of bounds:\n{}",
            failures.join("\n")
        );
    });
}

fn text_ref_dir() -> PathBuf {
    std::env::var_os("EMBEDGEMMA2_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/embedgemma2_ref"))
}

fn text_npy(name: &str) -> tessl::npy::NpyArray {
    let p = text_ref_dir().join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_reference.py)", p.display()))
}

/// The reference texts with their prompt names, and the ids sentence-transformers fed the model.
fn reference_inputs() -> Vec<(String, String, Vec<u32>)> {
    let json = std::fs::read_to_string(text_ref_dir().join("texts.json"))
        .unwrap_or_else(|e| panic!("texts.json: {e} (run tools/embedgemma2_ref/make_reference.py)"));
    let texts = reference_texts(&json);
    let ids = text_npy("ids.npy").i64_slice().expect("i64 ids").to_vec();
    let off = text_npy("offsets.npy").i64_slice().expect("i64 offsets").to_vec();
    assert_eq!(off.len(), texts.len() + 1, "offsets vs texts");
    texts
        .into_iter()
        .zip(off.windows(2))
        .map(|((prompt, text), w)| {
            let row = ids[w[0] as usize..w[1] as usize]
                .iter()
                .map(|&i| u32::try_from(i).unwrap())
                .collect();
            (prompt, text, row)
        })
        .collect()
}

#[test]
#[ignore]
fn the_rust_tokenizer_reproduces_the_reference_ids() {
    let tok = Tokenizer::from_snapshot(&snapshot()).unwrap();
    let inputs = reference_inputs();
    assert_eq!(inputs.len(), 8);
    for (n, (prompt, text, want)) in inputs.iter().enumerate() {
        let got = tok.encode_with_prompt(Some(prompt), text).unwrap();
        eprintln!("text {n} ({prompt}): {} ids, {}", got.len(), brief(text));
        assert_eq!(&got, want, "text {n} ({prompt}): {}", first_diff(&got, want));
    }
    // And as one batch of the eight, grouped by prompt as the facade takes them.
    for prompt in ["query", "document", "CodeRetrieval", "Classification"] {
        let members: Vec<&(String, String, Vec<u32>)> = inputs.iter().filter(|i| i.0 == prompt).collect();
        let texts: Vec<&str> = members.iter().map(|m| m.1.as_str()).collect();
        let got = tok.encode_batch_with_prompt(Some(prompt), &texts).unwrap();
        for (g, m) in got.iter().zip(&members) {
            assert_eq!(g, &m.2, "batched {prompt}");
        }
    }
}

#[test]
#[ignore]
fn text_to_embedding_with_no_python_matches_sentence_transformers() {
    let snap = snapshot();
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let tok = Tokenizer::from_snapshot(&snap).unwrap();
    let inputs = reference_inputs();
    let dim = cfg.embedding_dim as usize;
    let want = text_npy("emb_f32.npy");
    assert_eq!(want.shape, [inputs.len(), dim], "emb_f32 shape");
    let want = want.f32_slice().unwrap().to_vec();

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        let mut failures = Vec::new();
        for (n, (prompt, text, ids)) in inputs.iter().enumerate() {
            let got = model
                .encode_texts(&tok, Some(prompt), &[text], None)
                .unwrap()
                .embeddings;
            let reference = &want[n * dim..(n + 1) * dim];
            let (err, cos) = (max_abs(&got, reference), cosine(&got, reference));
            eprintln!(
                "text {n} ({prompt}): {} tokens, max abs {err:.3e}, cosine {cos:.9}",
                ids.len()
            );
            if err > EMB_ABS || cos < EMB_COS {
                failures.push(format!(
                    "text {n}: max abs {err:.3e} (bound {EMB_ABS:e}), cosine {cos:.9} (bound {EMB_COS})"
                ));
            }
            // The same ids through `encode` are the same forward: the facade adds nothing.
            let direct = model.encode(&[ids.as_slice()], None, false).unwrap().embeddings;
            assert_eq!(got, direct, "text {n}: encode_texts vs encode of the reference ids");
        }
        assert!(failures.is_empty(), "text path out of bounds:\n{}", failures.join("\n"));

        // A Matryoshka prefix goes through the same call.
        let short = model.encode_texts(&tok, Some("query"), &["x"], Some(128)).unwrap();
        assert_eq!(short.embeddings.len(), 128);
        // A text the tokenizer refuses is an error before any GPU work, naming the text.
        let err = model
            .encode_texts(&tok, None, &["fine", "has <|image|> in it"], None)
            .err()
            .expect("refused");
        assert!(err.contains("encode_texts") && err.contains("text 1"), "{err}");
        let err = model
            .encode_texts(&tok, Some("no such prompt"), &["x"], None)
            .err()
            .expect("refused");
        assert!(err.contains("no prompt named"), "{err}");
    });
}
