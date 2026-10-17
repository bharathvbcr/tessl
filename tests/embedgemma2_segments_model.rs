//! Items built from text interleaved with media, on the real
//! `google/embeddinggemma-2` checkpoint, against the transformers processor.
//!
//! `tools/embedgemma2_ref/make_segments_reference.py` renders each case's message the way the
//! snapshot's chat template does (pieces joined with no separator, each medium its placeholder),
//! runs `EmbeddingGemma2Processor` on it and the model on the result, and writes into
//! `target/embedgemma2_seg_ref/` (override `EMBEDGEMMA2_SEG_REF_DIR`), per case:
//!
//! * `{case}_ids.npy`: the ids the processor made;
//! * `{case}_emb_f32.npy`: the model's mean-pooled, L2-normalized embedding of them (fp32, CPU);
//!
//! and the media: `seg_image.png`, `seg_clip.wav` and `seg_video_raw.npy` (decoded frames, source
//! rate 4 fps).
//!
//! * `segment_layout_reproduces_the_processors_ids`: [`MmItem::Segments`]' layout (the library's
//!   `mixed_sequence` with the soft-token counts of the preprocessing, no placeholder counted by
//!   hand) equals `{case}_ids.npy` exactly, for an image, a clip and a video each between texts,
//!   two videos in a row, all three kinds, and a timestamped video. The media are the files: the
//!   image and clip go through `tessl::media`'s decoders and `embedgemma2_media::prepare_*`.
//! * `segment_built_items_match_the_processor_built_reference`: the embedding of every case, built
//!   as segments, is within max abs `1e-4` and cosine `0.99999` of `{case}_emb_f32.npy`
//!   (the bounds of `tests/embedgemma2_e2e_model.rs`), the cases in one call (one GPU wait) and each
//!   alone.
//!
//! Both are `#[ignore]`d:
//!
//! ```text
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo test --release --test embedgemma2_segments_model -- --ignored --test-threads=1
//! ```

mod common;

use std::path::{Path, PathBuf};

use common::with_gpu;
use tessl::embedgemma2::{EmbedGemma2Config, LayoutSegment};
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_media::{prepare_audio, prepare_image, MediaSource};
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem, MmSegment};
use tessl::embedgemma2_pre::{
    audio_soft_tokens, image_soft_tokens, preprocess_video, video_soft_tokens, AudioFrontEnd, PreConfig, PreparedImage,
    PreparedVideo, RawImage,
};
use tessl::embedgemma2_tok::Tokenizer;
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;

const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
/// The source rate the reference gave its video (`make_segments_reference.py`).
const VIDEO_FPS: f64 = 4.0;

/// The cases of `make_segments_reference.py`: text pieces and media, in order.
const CASES: [(&str, &[&str]); 7] = [
    ("t_i_t", &["a photo of", "IMAGE", "on a table"]),
    ("t_a_t", &["listen:", "AUDIO", "(a chirp)"]),
    ("t_v_t", &["watch", "VIDEO", "now"]),
    ("i_t", &["IMAGE", "alone with text"]),
    ("t_i_a_v", &["x", "IMAGE", "y", "AUDIO", "z", "VIDEO"]),
    ("v_v", &["VIDEO", "VIDEO"]),
    ("t_vt_t", &["at", "VIDEO_T", "end"]),
];

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory"),
    )
}

fn ref_dir() -> PathBuf {
    std::env::var("EMBEDGEMMA2_SEG_REF_DIR").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("embedgemma2_seg_ref")
        },
        PathBuf::from,
    )
}

fn npy(name: &str) -> tessl::npy::NpyArray {
    let p = ref_dir().join(name);
    read_npy(&p).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run tools/embedgemma2_ref/make_segments_reference.py)",
            p.display()
        )
    })
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

/// The media of the reference, prepared by this crate.
struct Media {
    image: PreparedImage,
    clip: AudioInput,
    video: PreparedVideo,
    /// The same video, `(frames, height, width)` as the reference stored it.
    video_shape: (usize, usize, usize),
}

fn media(pre: &PreConfig) -> Media {
    let dir = ref_dir();
    let image = prepare_image(&pre.image, MediaSource::Path(&dir.join("seg_image.png"))).unwrap();
    let fe = AudioFrontEnd::new(&pre.audio).unwrap();
    let clip = prepare_audio(&fe, MediaSource::Path(&dir.join("seg_clip.wav"))).unwrap();
    let raw = npy("seg_video_raw.npy");
    let (t, h, w) = (raw.shape[0], raw.shape[1], raw.shape[2]);
    assert_eq!(raw.shape, [t, h, w, 3], "seg_video_raw.npy");
    let pixels: Vec<u8> = raw
        .i64_slice()
        .expect("seg_video_raw.npy holds int64")
        .iter()
        .map(|&v| u8::try_from(v).expect("a uint8 value"))
        .collect();
    let frames: Vec<RawImage<'_>> = (0..t)
        .map(|f| RawImage::rgb(w, h, &pixels[f * h * w * 3..(f + 1) * h * w * 3]))
        .collect();
    let video = preprocess_video(&pre.video, &frames, VIDEO_FPS).unwrap();
    Media {
        image,
        clip,
        video,
        video_shape: (t, h, w),
    }
}

/// The audio placeholders of the processor's ids for the clip alone (`t_a_t`: one clip between two
/// texts): what the tower makes for a 1 s clip, counted from the reference and not from this crate.
fn clip_soft_tokens(cfg: &EmbedGemma2Config) -> usize {
    let audio = cfg.special_tokens.audio.expect("config.json's audio_token_id");
    npy("t_a_t_ids.npy")
        .i64_slice()
        .expect("int64 ids")
        .iter()
        .filter(|&&i| i == i64::from(audio))
        .count()
}

/// The counts the C ABI's `*_soft_token_count` entries compute from sizes alone are the ones the
/// preprocessing and the tower's mask make.
fn assert_counts_from_sizes(pre: &PreConfig, m: &Media, clip_soft: usize) {
    let img = image_soft_tokens(&pre.image, 224, 224).unwrap();
    assert_eq!(img, m.image.soft_tokens, "image_soft_tokens vs preprocessing");
    let (t, h, w) = m.video_shape;
    let (kept, per_frame) = video_soft_tokens(&pre.video, h, w, t, VIDEO_FPS).unwrap();
    assert_eq!(
        (kept, per_frame),
        (m.video.frames.len(), m.video.soft_tokens_per_frame),
        "video_soft_tokens vs preprocessing"
    );
    // The clip's samples: the reference built it as 1.0 s at 16 kHz.
    assert_eq!(
        audio_soft_tokens(&pre.audio, 16000),
        clip_soft,
        "audio_soft_tokens vs the processor's audio placeholders"
    );
}

fn layout<'a>(pieces: &[&'a str], m: &'a Media, clip_soft: usize, per_frame: &'a [usize]) -> Vec<LayoutSegment<'a>> {
    pieces
        .iter()
        .map(|&p| match p {
            "IMAGE" => LayoutSegment::Image {
                soft: m.image.soft_tokens,
            },
            "AUDIO" => LayoutSegment::Audio { soft: clip_soft },
            "VIDEO" => LayoutSegment::Video {
                soft_per_frame: per_frame,
                timestamps: None,
            },
            "VIDEO_T" => LayoutSegment::Video {
                soft_per_frame: per_frame,
                timestamps: Some(&m.video.timestamps),
            },
            text => LayoutSegment::Text(text),
        })
        .collect()
}

#[test]
#[ignore]
fn segment_layout_reproduces_the_processors_ids() {
    let snap = snapshot();
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json")).unwrap();
    let tok = Tokenizer::from_snapshot(&snap).unwrap();
    let pre = PreConfig::from_snapshot(&snap).unwrap();
    let m = media(&pre);
    let clip_soft = clip_soft_tokens(&cfg);
    assert_counts_from_sizes(&pre, &m, clip_soft);
    let per_frame = vec![m.video.soft_tokens_per_frame; m.video.frames.len()];
    for (name, pieces) in CASES {
        let want: Vec<u32> = npy(&format!("{name}_ids.npy"))
            .i64_slice()
            .expect("int64 ids")
            .iter()
            .map(|&i| u32::try_from(i).unwrap())
            .collect();
        let got = cfg
            .mixed_sequence(&tok, &layout(pieces, &m, clip_soft, &per_frame))
            .unwrap();
        assert_eq!(got, want, "{name}: the library's layout vs the processor's ids");
        eprintln!("{name}: {} ids, equal to the processor's", want.len());
    }
}

#[test]
#[ignore]
fn segment_built_items_match_the_processor_built_reference() {
    let snap = snapshot();
    let tok = Tokenizer::from_snapshot(&snap).unwrap();
    let pre = PreConfig::from_snapshot(&snap).unwrap();
    let m = media(&pre);
    let frames: Vec<ImageInput<'_>> = m.video.frames.iter().map(PreparedImage::input).collect();
    let segments: Vec<Vec<MmSegment<'_>>> = CASES
        .iter()
        .map(|(_, pieces)| {
            pieces
                .iter()
                .map(|&p| match p {
                    "IMAGE" => MmSegment::Image(m.image.input()),
                    "AUDIO" => MmSegment::Audio(&m.clip),
                    "VIDEO" => MmSegment::Video(&frames),
                    "VIDEO_T" => MmSegment::TimestampedVideo {
                        frames: &frames,
                        timestamps: &m.video.timestamps,
                    },
                    text => MmSegment::Text(text),
                })
                .collect()
        })
        .collect();
    let items: Vec<MmItem<'_>> = segments
        .iter()
        .map(|s| MmItem::Segments {
            segments: s,
            tokenizer: &tok,
        })
        .collect();
    let wants: Vec<Vec<f32>> = CASES
        .iter()
        .map(|(name, _)| {
            npy(&format!("{name}_emb_f32.npy"))
                .f32_slice()
                .expect("f32 reference")
                .to_vec()
        })
        .collect();

    with_gpu(|rt| {
        let model = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::ExactF32).unwrap();
        let dim = model.text().config().embedding_dim as usize;
        let all = model.encode(&items, None).unwrap().embeddings;
        assert_eq!(all.len(), CASES.len() * dim);
        let (mut worst, mut worst_cos) = (0.0f64, 1.0f64);
        let mut failures = Vec::new();
        for (k, (name, _)) in CASES.iter().enumerate() {
            let alone = model.encode(std::slice::from_ref(&items[k]), None).unwrap().embeddings;
            for (what, got) in [("in one call", &all[k * dim..(k + 1) * dim]), ("alone", &alone[..])] {
                let (err, cos) = (max_abs(got, &wants[k]), cosine(got, &wants[k]));
                eprintln!("{name} {what}: max abs {err:.3e}, cosine {cos:.8}");
                worst = worst.max(err);
                worst_cos = worst_cos.min(cos);
                if err > EMB_ABS || cos < EMB_COS {
                    failures.push(format!("{name} {what}: max abs {err:.3e}, cosine {cos:.8}"));
                }
            }
        }
        eprintln!(
            "worst over {} cases: max abs {worst:.3e}, cosine {worst_cos:.8}",
            CASES.len()
        );
        assert!(
            failures.is_empty(),
            "outside max abs {EMB_ABS} / cosine {EMB_COS}:\n{}",
            failures.join("\n")
        );
    });
}
