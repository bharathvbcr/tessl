//! Video on the real checkpoint: raw decoded frames, no Python processor,
//! through `tessl::embedgemma2_pre` (sampling, resize, patches), the vision
//! tower (one batch of frames), the video soft tokens in the text sequence and
//! the text encoder, against sentence-transformers' embedding (transformers
//! 5.19, fp32, sdpa, CPU). Opt-in:
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=/path/to/target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_video_model -- --ignored --test-threads=1
//! ```
//!
//! `make_mm_reference.py` saves each video's raw frames, source frame rate,
//! the frames the processor kept, its tensors, ids and soft tokens, and the
//! embeddings, for two synthetic videos: `video0` (24 frames at 4 fps, 96x176,
//! sampled to 6) and `video1` (100 frames at 2.5 fps, 72x104, 40 sampled and
//! cut to 32 by the frame cap), and `video0t`, `video0` with the processor's
//! `add_timestamps` on. A missing variable or file fails the test rather than
//! skipping it.
//!
//! * `native_video_preprocessing_equals_the_processors` (no device): the
//!   sampled indices, the timestamps, every frame's pixels and positions equal
//!   the processor's bit for bit.
//! * `video_ids_equal_the_processors`: `video_sequence` of the soft-token counts
//!   is `ids.npy`; the timestamped sequence is the processor's with the
//!   tokenizer's text between the frames' blocks, and
//!   `video_sequence_with_timestamps` (the snapshot's tokenizer, this crate's
//!   timestamps and the golden's) builds `video0t`'s `ids.npy` exactly.
//! * `the_vision_tower_makes_the_processors_soft_tokens`: the frames' soft
//!   tokens against the captured ones, frame by frame.
//! * `raw_video_to_embedding_matches_sentence_transformers`: raw frames to
//!   embedding in one call of both videos, within `1e-4` max abs of
//!   `emb_f32.npy` and cosine `0.99999` (the bounds of
//!   `tests/embedgemma2_e2e_model.rs`), and `video0t` through `MmItem::Media`
//!   (the golden's ids) and through `MmItem::TimestampedVideo` (a config with
//!   `add_timestamps` on, native timestamps, the snapshot's tokenizer).
//! * `bf16_video_stays_near_the_fp32_embedding`: the same with both models in
//!   `GemmOperands::Bf16`.
//! * `a_mixed_call_gives_each_item_its_own_embedding`: two videos, an image and
//!   a clip in one call, each against `emb_f32.npy` and alone.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::{
    preprocess_audio, preprocess_image, preprocess_video, PreConfig, PreparedImage, PreparedVideo, RawImage,
};
use tessl::embedgemma2_tok::Tokenizer;
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;

const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
/// A frame's soft tokens against the processor's, within this of their largest
/// magnitude (the vision tower's bound, `tests/embedgemma2_vision_model.rs`).
const SOFT_REL: f64 = 1e-4;
/// Bf16 against the fp32 embedding, per item: measured `1 - cos` was 3.2e-6 and
/// 7.0e-6 for the two videos and at most 1.2e-5 for the image and clip of the
/// mixed call (the earlier image and clip tests saw the same), so `3e-5` is
/// 2.6x the worst of them.
const BF16_COS: f64 = 0.99997;
/// Bf16 against transformers' own bf16 run (which differs from its fp32 by
/// 3.0e-5 and 1.4e-5 here), the bound of `tests/embedgemma2_mm_model.rs`.
const BF16_VS_TRANSFORMERS_COS: f64 = 0.9996;
const VIDEOS: [&str; 2] = ["video0_", "video1_"];

fn env_dir(var: &str, what: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("{var} must name {what}")))
}

fn npy(name: &str) -> tessl::npy::NpyArray {
    let p = env_dir("EMBEDGEMMA2_MM_REF_DIR", "make_mm_reference.py's (sdpa) output").join(name);
    read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_mm_reference.py)", p.display()))
}

fn npy_i64(name: &str) -> Vec<i64> {
    npy(name).i64_slice().unwrap().to_vec()
}

fn npy_f32(name: &str) -> Vec<f32> {
    npy(name).f32_slice().unwrap().to_vec()
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

fn snapshot() -> PathBuf {
    env_dir("EMBEDGEMMA2_SNAPSHOT", "the google/embeddinggemma-2 snapshot directory")
}

fn config() -> PreConfig {
    PreConfig::from_snapshot(&snapshot()).expect("the snapshot's processor_config.json")
}

/// The snapshot's `processor_config.json` with `add_timestamps` switched on, as
/// the reference run of `video0t` does to its processor.
fn timestamped_config() -> PreConfig {
    let text = std::fs::read_to_string(snapshot().join("processor_config.json")).unwrap();
    assert_eq!(
        text.matches(r#""add_timestamps": false"#).count(),
        1,
        "the snapshot's setting"
    );
    PreConfig::from_json(&text.replace(r#""add_timestamps": false"#, r#""add_timestamps": true"#)).unwrap()
}

/// The raw frames of reference video `prefix`, preprocessed natively.
fn native_video(cfg: &PreConfig, prefix: &str) -> PreparedVideo {
    let raw = npy(&format!("{prefix}raw.npy"));
    assert_eq!(raw.shape.len(), 4, "{prefix} raw {:?}", raw.shape);
    let (t, h, w) = (raw.shape[0], raw.shape[1], raw.shape[2]);
    assert_eq!(raw.shape[3], 3);
    let bytes: Vec<u8> = raw
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&v| u8::try_from(v).unwrap())
        .collect();
    let frames: Vec<RawImage<'_>> = (0..t)
        .map(|i| RawImage::rgb(w, h, &bytes[i * h * w * 3..(i + 1) * h * w * 3]))
        .collect();
    let fps = npy(&format!("{prefix}fps.npy")).f64_slice().unwrap()[0];
    preprocess_video(&cfg.video, &frames, fps).unwrap()
}

fn inputs(v: &PreparedVideo) -> Vec<ImageInput<'_>> {
    v.frames.iter().map(PreparedImage::input).collect()
}

fn soft_per_frame(prefix: &str) -> usize {
    usize::try_from(npy_i64(&format!("{prefix}soft_per_frame.npy"))[0]).unwrap()
}

#[test]
#[ignore]
fn native_video_preprocessing_equals_the_processors() {
    let cfg = config();
    for prefix in VIDEOS {
        let got = native_video(&cfg, prefix);
        let want_idx: Vec<usize> = npy_i64(&format!("{prefix}indices.npy"))
            .into_iter()
            .map(|i| usize::try_from(i).unwrap())
            .collect();
        assert_eq!(got.indices, want_idx, "{prefix}: sampled indices");
        let want_ts = npy(&format!("{prefix}timestamps.npy"));
        assert_eq!(got.timestamps, want_ts.f64_slice().unwrap(), "{prefix}: timestamps");
        assert_eq!(
            got.soft_tokens_per_frame,
            soft_per_frame(prefix),
            "{prefix}: soft tokens"
        );

        let px = npy(&format!("{prefix}pixel_values_videos.npy"));
        let pos = npy(&format!("{prefix}video_position_ids.npy"));
        let (want_px, want_pos) = (px.f32_slice().unwrap(), pos.i64_slice().unwrap());
        let (pn, qn) = (px.shape[1] * px.shape[2], pos.shape[1] * 2);
        assert_eq!(px.shape[0], got.frames.len(), "{prefix}: frame count");
        let mut bad = 0usize;
        for (f, frame) in got.frames.iter().enumerate() {
            assert_eq!(frame.pixel_values.len(), pn, "{prefix} frame {f} pixel count");
            bad += frame
                .pixel_values
                .iter()
                .zip(&want_px[f * pn..(f + 1) * pn])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let want: Vec<i32> = want_pos[f * qn..(f + 1) * qn]
                .iter()
                .map(|&v| i32::try_from(v).unwrap())
                .collect();
            assert_eq!(frame.position_ids, want, "{prefix} frame {f}: positions");
        }
        eprintln!(
            "{prefix}: {} of {} frames kept, indices {:?}, {bad} of {} pixel values differ",
            got.frames.len(),
            npy(&format!("{prefix}raw.npy")).shape[0],
            got.indices,
            px.shape[0] * pn
        );
        assert_eq!(bad, 0, "{prefix}: pixel_values_videos differ from the processor's");
    }
}

/// The processor's ids for a timestamped video with the text between the
/// blocks split out: `prefixes[f]` is what precedes frame `f`'s `<boi>`. The
/// structure is checked, and the text ids are the tokenizer's (this crate has
/// no tokenizer), so they are taken from the golden.
fn split_timestamped(ids: &[u32], boi: u32, eoi: u32, video: u32, bos: u32, eos: u32) -> Vec<Vec<u32>> {
    assert_eq!((ids[0], ids[ids.len() - 1]), (bos, eos), "bos and eos");
    let mut prefixes = Vec::new();
    let mut i = 1;
    while i < ids.len() - 1 {
        let start = i;
        while ids[i] != boi {
            assert!(
                ids[i] != video && ids[i] != eoi,
                "a placeholder or eoi at {i} outside a block"
            );
            i += 1;
        }
        prefixes.push(ids[start..i].to_vec());
        i += 1;
        while ids[i] == video {
            i += 1;
        }
        assert_eq!(ids[i], eoi, "a block ends with eoi at {i}");
        i += 1;
    }
    assert_eq!(i, ids.len() - 1, "the blocks cover the sequence");
    prefixes
}

#[test]
#[ignore]
fn video_ids_equal_the_processors() {
    let cfg = config();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snapshot(), GemmOperands::ExactF32).unwrap();
        let tc = m.text().config();
        for prefix in VIDEOS {
            let v = native_video(&cfg, prefix);
            let want: Vec<u32> = npy_i64(&format!("{prefix}ids.npy"))
                .into_iter()
                .map(|i| u32::try_from(i).unwrap())
                .collect();
            let counts = vec![v.soft_tokens_per_frame; v.frames.len()];
            let built = tc.video_sequence(&counts).unwrap();
            assert_eq!(built, want, "{prefix}: video_sequence vs ids.npy");
            eprintln!(
                "{prefix}: {} frames x {} soft tokens: {} ids, as the processor's",
                v.frames.len(),
                v.soft_tokens_per_frame,
                built.len()
            );
        }

        // With timestamps: the processor writes `mm:ss <boi>...<eoi>` per frame,
        // joined by spaces; the tokenizer's ids for the `mm:ss ` text are the
        // golden's, and the blocks are the same blocks.
        let t = tc.special_tokens;
        let (boi, eoi, video) = (t.boi.unwrap(), t.eoi.unwrap(), t.video.unwrap());
        let (bos, eos) = (t.bos.unwrap(), t.eos.unwrap());
        let ids: Vec<u32> = npy_i64("video0t_ids.npy")
            .into_iter()
            .map(|i| u32::try_from(i).unwrap())
            .collect();
        let plain: Vec<u32> = npy_i64("video0_ids.npy")
            .into_iter()
            .map(|i| u32::try_from(i).unwrap())
            .collect();
        let prefixes = split_timestamped(&ids, boi, eoi, video, bos, eos);
        let frames = npy_i64("video0_indices.npy").len();
        assert_eq!(prefixes.len(), frames, "one label per frame");
        assert!(prefixes.iter().all(|p| !p.is_empty()), "every frame is labelled");
        // The timestamped sequence is the plain one with those labels inserted
        // before each block and nothing else changed.
        let n = soft_per_frame("video0_");
        let mut rebuilt = vec![bos];
        for p in &prefixes {
            rebuilt.extend(p);
            rebuilt.extend(tc.video_sequence(&[n]).unwrap()[1..n + 3].iter());
        }
        rebuilt.push(eos);
        assert_eq!(rebuilt, ids, "timestamped ids = labels + the plain blocks");
        assert_ne!(ids, plain);

        // Natively: the tokenizer's `mm:ss ` ids and the config's blocks make
        // the golden's ids exactly, from the timestamps of this crate's own
        // preprocessing and from the golden's.
        let tok = Tokenizer::from_snapshot(&snapshot()).unwrap();
        let stamped_cfg = timestamped_config();
        assert!(stamped_cfg.video.add_timestamps && !cfg.video.add_timestamps);
        let v0 = native_video(&stamped_cfg, "video0_");
        let counts = vec![v0.soft_tokens_per_frame; v0.frames.len()];
        for (what, stamps) in [
            ("native timestamps", v0.timestamps.clone()),
            (
                "golden timestamps",
                npy("video0_timestamps.npy").f64_slice().unwrap().to_vec(),
            ),
        ] {
            let built = tc.video_sequence_with_timestamps(&tok, &stamps, &counts).unwrap();
            assert_eq!(built, ids, "video0t: {what}: video_sequence_with_timestamps vs ids.npy");
        }
        eprintln!(
            "video0t: {} ids built through the tokenizer, equal to ids.npy",
            ids.len()
        );
        eprintln!(
            "video0t: labels of {:?} ids before each of {frames} blocks",
            prefixes.iter().map(Vec::len).collect::<Vec<_>>()
        );
    });
}

#[test]
#[ignore]
fn the_vision_tower_makes_the_processors_soft_tokens() {
    let cfg = config();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snapshot(), GemmOperands::ExactF32).unwrap();
        let h = m.text().config().hidden as usize;
        for prefix in VIDEOS {
            let v = native_video(&cfg, prefix);
            let frames = inputs(&v);
            let out = m.vision().encode_images(&frames, false).unwrap();
            let want = npy_f32(&format!("{prefix}soft.npy"));
            let n = v.soft_tokens_per_frame;
            assert_eq!(want.len(), frames.len() * n * h, "{prefix}: golden soft tokens");
            let mut worst = 0.0f64;
            for (f, soft) in out.soft.iter().enumerate() {
                assert_eq!(soft.len(), n * h, "{prefix} frame {f}: soft tokens");
                let golden = &want[f * n * h..(f + 1) * n * h];
                let scale = golden.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
                let err = max_abs(soft, golden);
                worst = worst.max(err / scale);
                assert!(
                    err <= SOFT_REL * scale,
                    "{prefix} frame {f}: soft max abs {err:.3e} > {SOFT_REL:e} * {scale:.3e}"
                );
            }
            eprintln!(
                "{prefix}: {} frames in {} tower forward(s), worst soft error {worst:.2e} of the largest magnitude",
                frames.len(),
                out.forwards
            );
        }
    });
}

fn check_embeddings(what: &str, names: &[&str], got: &[f32], dim: usize, failures: &mut Vec<String>) {
    for (k, name) in names.iter().enumerate() {
        let want = npy_f32(&format!("{name}emb_f32.npy"));
        assert_eq!(want.len(), dim, "{name} emb_f32");
        let row = &got[k * dim..(k + 1) * dim];
        let (err, cos) = (max_abs(row, &want), cosine(row, &want));
        eprintln!("{what} {name}: max abs {err:.3e}, cosine {cos:.9}");
        if err > EMB_ABS || cos < EMB_COS {
            failures.push(format!(
                "{what} {name}: max abs {err:.3e} (bound {EMB_ABS:e}), cosine {cos:.9} (bound {EMB_COS})"
            ));
        }
    }
}

#[test]
#[ignore]
fn raw_video_to_embedding_matches_sentence_transformers() {
    let cfg = config();
    let videos: Vec<PreparedVideo> = VIDEOS.iter().map(|p| native_video(&cfg, p)).collect();
    let frames: Vec<Vec<ImageInput<'_>>> = videos.iter().map(inputs).collect();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snapshot(), GemmOperands::ExactF32).unwrap();
        let dim = m.text().config().embedding_dim as usize;
        let items: Vec<MmItem<'_>> = frames.iter().map(|f| MmItem::Video(f)).collect();
        let out = m.encode(&items, None).unwrap();
        assert_eq!(out.embeddings.len(), items.len() * dim);
        let mut failures = Vec::new();
        check_embeddings("raw video", &VIDEOS, &out.embeddings, dim, &mut failures);

        // Alone, each is the same embedding as in the call.
        for (k, item) in items.iter().enumerate() {
            let alone = m.encode(std::slice::from_ref(item), None).unwrap();
            let d = max_abs(&alone.embeddings, &out.embeddings[k * dim..(k + 1) * dim]);
            eprintln!("{}: alone vs in the call, max abs {d:.3e}", VIDEOS[k]);
            assert!(d <= 1e-5, "{}: alone vs in the call {d:.3e}", VIDEOS[k]);
        }

        // With timestamps: video0's frames again, the ids carrying the
        // processor's `mm:ss` text, through a Media item.
        let ids: Vec<u32> = npy_i64("video0t_ids.npy")
            .into_iter()
            .map(|i| u32::try_from(i).unwrap())
            .collect();
        let vids: [&[ImageInput<'_>]; 1] = [&frames[0]];
        let stamped = m
            .encode(
                &[MmItem::Media {
                    ids: &ids,
                    images: &[],
                    audio: &[],
                    videos: &vids,
                }],
                None,
            )
            .unwrap();
        check_embeddings(
            "timestamped (Media)",
            &["video0t_"],
            &stamped.embeddings,
            dim,
            &mut failures,
        );

        // And natively: the config with `add_timestamps` on, this crate's own
        // timestamps and the snapshot's tokenizer, in one item.
        let tok = Tokenizer::from_snapshot(&snapshot()).unwrap();
        let stamped_cfg = timestamped_config();
        let v0t = native_video(&stamped_cfg, "video0_");
        let f0t = inputs(&v0t);
        assert!(
            stamped_cfg.video.add_timestamps,
            "test bug: the config must ask for timestamps"
        );
        let item = MmItem::TimestampedVideo {
            frames: &f0t,
            timestamps: &v0t.timestamps,
            tokenizer: &tok,
        };
        let native = m.encode(&[item], None).unwrap();
        check_embeddings(
            "timestamped (TimestampedVideo)",
            &["video0t_"],
            &native.embeddings,
            dim,
            &mut failures,
        );
        let d = max_abs(&native.embeddings, &stamped.embeddings);
        eprintln!("TimestampedVideo vs Media with the golden's ids: max abs {d:.3e}");
        assert!(d <= 1e-5, "TimestampedVideo vs Media: {d:.3e}");
        assert!(
            failures.is_empty(),
            "raw video to embedding out of bounds:\n{}",
            failures.join("\n")
        );
    });
}

#[test]
#[ignore]
fn bf16_video_stays_near_the_fp32_embedding() {
    let cfg = config();
    let videos: Vec<PreparedVideo> = VIDEOS.iter().map(|p| native_video(&cfg, p)).collect();
    let frames: Vec<Vec<ImageInput<'_>>> = videos.iter().map(inputs).collect();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snapshot(), GemmOperands::Bf16).unwrap();
        let dim = m.text().config().embedding_dim as usize;
        let items: Vec<MmItem<'_>> = frames.iter().map(|f| MmItem::Video(f)).collect();
        let out = m.encode(&items, None).unwrap();
        let mut failures = Vec::new();
        for (k, name) in VIDEOS.iter().enumerate() {
            let row = &out.embeddings[k * dim..(k + 1) * dim];
            let fp32 = npy_f32(&format!("{name}emb_f32.npy"));
            let tf_bf16 = npy_f32(&format!("{name}emb_bf16.npy"));
            let (c32, c16) = (cosine(row, &fp32), cosine(row, &tf_bf16));
            let tf_vs_fp32 = cosine(&tf_bf16, &fp32);
            eprintln!(
                "{name}: bf16 vs fp32 cosine {c32:.9}, vs transformers bf16 {c16:.9} \
                 (transformers bf16 vs fp32 {tf_vs_fp32:.9})"
            );
            if c32 < BF16_COS {
                failures.push(format!("{name}: bf16 vs fp32 cosine {c32:.9} (bound {BF16_COS})"));
            }
            if c16 < BF16_VS_TRANSFORMERS_COS {
                failures.push(format!(
                    "{name}: bf16 vs transformers bf16 cosine {c16:.9} (bound {BF16_VS_TRANSFORMERS_COS})"
                ));
            }
            // Closer to the fp32 reference than transformers' own bf16 run is.
            if 1.0 - c32 > 1.0 - tf_vs_fp32 {
                failures.push(format!(
                    "{name}: bf16 1 - cos {:.3e} is worse than transformers' bf16 {:.3e}",
                    1.0 - c32,
                    1.0 - tf_vs_fp32
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "bf16 video out of bounds:\n{}",
            failures.join("\n")
        );
    });
}

#[test]
#[ignore]
fn a_mixed_call_gives_each_item_its_own_embedding() {
    let cfg = config();
    let videos: Vec<PreparedVideo> = VIDEOS.iter().map(|p| native_video(&cfg, p)).collect();
    let frames: Vec<Vec<ImageInput<'_>>> = videos.iter().map(inputs).collect();
    let raw_image = npy("image1_raw.npy");
    let bytes: Vec<u8> = raw_image
        .i64_slice()
        .unwrap()
        .iter()
        .map(|&v| u8::try_from(v).unwrap())
        .collect();
    let image = preprocess_image(
        &cfg.image,
        &RawImage::rgb(raw_image.shape[1], raw_image.shape[0], &bytes),
    )
    .unwrap();
    let clip: AudioInput = preprocess_audio(&cfg.audio, npy("audio1_raw.npy").f32_slice().unwrap(), 16000).unwrap();
    let names = ["video1_", "image1_", "video0_", "audio1_"];
    with_gpu(|rt| {
        for operands in [GemmOperands::ExactF32, GemmOperands::Bf16] {
            let m = EmbedGemma2Multimodal::load(rt, &snapshot(), operands).unwrap();
            let dim = m.text().config().embedding_dim as usize;
            let items = [
                MmItem::Video(&frames[1]),
                MmItem::Image(image.input()),
                MmItem::Video(&frames[0]),
                MmItem::Audio(&clip),
            ];
            let out = m.encode(&items, None).unwrap();
            let mut failures = Vec::new();
            match operands {
                GemmOperands::ExactF32 => check_embeddings("mixed call", &names, &out.embeddings, dim, &mut failures),
                GemmOperands::Bf16 => {
                    for (k, name) in names.iter().enumerate() {
                        let want = npy_f32(&format!("{name}emb_f32.npy"));
                        let cos = cosine(&out.embeddings[k * dim..(k + 1) * dim], &want);
                        eprintln!("bf16 mixed call {name}: cosine vs fp32 {cos:.9}");
                        if cos < BF16_COS {
                            failures.push(format!("bf16 mixed call {name}: cosine {cos:.9} (bound {BF16_COS})"));
                        }
                    }
                }
            }
            for (k, item) in items.iter().enumerate() {
                let alone = m.encode(std::slice::from_ref(item), None).unwrap();
                let d = max_abs(&alone.embeddings, &out.embeddings[k * dim..(k + 1) * dim]);
                eprintln!("{operands:?} {}: alone vs in the call, max abs {d:.3e}", names[k]);
                if d > 1e-5 {
                    failures.push(format!("{operands:?} {}: alone vs in the call {d:.3e}", names[k]));
                }
            }
            assert!(
                failures.is_empty(),
                "mixed call out of bounds:\n{}",
                failures.join("\n")
            );
        }
    });
}
