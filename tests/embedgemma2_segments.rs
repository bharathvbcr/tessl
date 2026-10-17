//! Text interleaved with media, laid out by the library: `EmbedGemma2Config::mixed_sequence`,
//! `MmItem::Segments` and the soft-token counts that let a caller skip counting placeholders.
//!
//! The tiny checkpoint of `tests/common/embedgemma2_mm.rs` and a tokenizer with every marker
//! and placeholder as an added token (`full_tokenizer_json`, with merges so that a space gluing
//! to a digit and `0 0` joining make stretch boundaries observable). The oracle for the ids is
//! the processor's way of working, written out: the snapshot's chat template joins the pieces
//! with nothing between them and writes each medium as a placeholder, the processor expands
//! each placeholder to its block, and the tokenizer tokenizes that whole string
//! (`encode_raw(string, true)`) -- so the layout, which never builds the string, must equal it
//! for every message below. The real processor's output for the real checkpoint is
//! `tests/embedgemma2_segments_model.rs`.
//!
//! Device-free (hosted selection): the ids, the refusals and the counts. On a device: an item
//! built from segments is the `MmItem::Media` item with the whole-string ids bit for bit, in
//! both text modes, in one wait.

mod common;

use std::sync::Arc;

use common::embedgemma2::{open, PREFIX};
use common::embedgemma2_mm::{
    checkpoint_bytes, clip, config_json, config_json_without_ids, configs, full_tokenizer_json, Configs, Image,
};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{LayoutSegment, MAX_TIMESTAMP_SECONDS};
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem, MmSegment};
use tessl::embedgemma2_pre::{
    audio_soft_tokens, image_soft_tokens, preprocess_image, preprocess_video, timestamp_label, video_soft_tokens,
    AudioFrontEnd, PreConfig, RawImage,
};
use tessl::embedgemma2_tok::{Tokenizer, MAX_TEXT_BYTES};
use tessl::embedgemma2_vision::ImageInput;
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;

const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];

const BOS: u32 = 2;
const EOS: u32 = 1;
const BOI: u32 = 40;
const EOI: u32 = 41;
const BOA: u32 = 42;
const EOA: u32 = 43;
const IMAGE: u32 = 60;
const AUDIO: u32 = 61;
const VIDEO: u32 = 62;

fn tokenizer() -> Tokenizer {
    Tokenizer::from_json(&full_tokenizer_json()).unwrap()
}

/// One piece of a message, with the soft-token counts the media would make.
#[derive(Clone, Copy)]
enum P<'a> {
    T(&'a str),
    Img(usize),
    Aud(usize),
    Vid(&'a [usize]),
    VidT(&'a [usize], &'a [f64]),
}

impl<'a> P<'a> {
    fn layout(self) -> LayoutSegment<'a> {
        match self {
            P::T(t) => LayoutSegment::Text(t),
            P::Img(soft) => LayoutSegment::Image { soft },
            P::Aud(soft) => LayoutSegment::Audio { soft },
            P::Vid(soft_per_frame) => LayoutSegment::Video {
                soft_per_frame,
                timestamps: None,
            },
            P::VidT(soft_per_frame, stamps) => LayoutSegment::Video {
                soft_per_frame,
                timestamps: Some(stamps),
            },
        }
    }
}

/// The processor's whole string for a message: the text as it is, each medium
/// as its block (a video's blocks joined by a space under timestamps, each
/// block behind its `mm:ss` label), nothing between the pieces.
fn processor_string(pieces: &[P<'_>]) -> String {
    let (boi, eoi, boa, eoa) = ("<|image>", "<image|>", "<|audio>", "<audio|>");
    let (image, audio, video) = ("<|image|>", "<|audio|>", "<|video|>");
    pieces
        .iter()
        .map(|p| match *p {
            P::T(t) => t.to_string(),
            P::Img(n) => format!("{boi}{}{eoi}", image.repeat(n)),
            P::Aud(n) => format!("{boa}{}{eoa}", audio.repeat(n)),
            P::Vid(soft) => soft
                .iter()
                .map(|&n| format!("{boi}{}{eoi}", video.repeat(n)))
                .collect::<String>(),
            P::VidT(soft, stamps) => stamps
                .iter()
                .zip(soft)
                .map(|(&t, &n)| format!("{} {boi}{}{eoi}", timestamp_label(t), video.repeat(n)))
                .collect::<Vec<_>>()
                .join(" "),
        })
        .collect()
}

/// The messages the layout is held to. They put digits next to the stretches' edges (`0 0`
/// merges, a space glues to the digit after it), so a boundary in the wrong place changes ids.
fn messages() -> Vec<(&'static str, Vec<P<'static>>)> {
    const ONE: &[usize] = &[2];
    const THREE: &[usize] = &[3, 3, 3];
    const MIXED: &[usize] = &[1, 1];
    const STAMPS: &[f64] = &[0.0, 1.0, 59.9];
    const STAMPS2: &[f64] = &[0.0, 61.0];
    vec![
        (
            "text image text",
            vec![P::T("a photo of"), P::Img(6), P::T("on a table")],
        ),
        ("image first", vec![P::Img(3), P::T("1: 2")]),
        ("image last", vec![P::T("0 1"), P::Img(2)]),
        ("image alone", vec![P::Img(5)]),
        ("text alone", vec![P::T("12 0 00")]),
        ("empty text", vec![P::T(""), P::Img(1), P::T("")]),
        (
            "adjacent texts merge",
            vec![P::T("0"), P::T("0"), P::Img(1), P::T("0"), P::T(" 0")],
        ),
        ("clip", vec![P::T("listen:"), P::Aud(4), P::T("(a chirp)")]),
        ("video", vec![P::T("watch"), P::Vid(THREE), P::T("now")]),
        ("video alone", vec![P::Vid(ONE)]),
        ("two videos in a row", vec![P::Vid(MIXED), P::Vid(ONE)]),
        (
            "image clip video",
            vec![P::T("x"), P::Img(2), P::T("y"), P::Aud(3), P::T("z"), P::Vid(MIXED)],
        ),
        ("two images", vec![P::Img(1), P::Img(2), P::T("0"), P::Img(1)]),
        (
            "timestamped video",
            vec![P::T("at"), P::VidT(THREE, STAMPS), P::T("end")],
        ),
        (
            "text glued to the first label",
            vec![P::T("0"), P::VidT(MIXED, STAMPS2), P::T("0")],
        ),
        ("timestamped video alone", vec![P::VidT(THREE, STAMPS)]),
        (
            "everything",
            vec![
                P::T("0 "),
                P::Img(2),
                P::T(" 0"),
                P::VidT(MIXED, STAMPS2),
                P::T("0:1"),
                P::Aud(1),
                P::Vid(ONE),
            ],
        ),
    ]
}

#[test]
fn mixed_sequence_is_the_tokenizers_ids_of_the_processors_whole_string() {
    let cfg = configs().text;
    let tok = tokenizer();
    for (name, pieces) in messages() {
        let layout: Vec<LayoutSegment<'_>> = pieces.iter().map(|p| p.layout()).collect();
        let got = cfg
            .mixed_sequence(&tok, &layout)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = tok.encode_raw(&processor_string(&pieces), true);
        assert_eq!(got, want, "{name}");
        assert_eq!((got[0], *got.last().unwrap()), (BOS, EOS), "{name}");
    }
}

/// Written out, so the oracle above is not only itself: the ids of three messages.
#[test]
fn mixed_sequence_ids_written_out() {
    let cfg = configs().text;
    let tok = tokenizer();
    let seq = |p: &[P<'_>]| {
        let layout: Vec<LayoutSegment<'_>> = p.iter().map(|p| p.layout()).collect();
        cfg.mixed_sequence(&tok, &layout).unwrap()
    };
    // The digit ids are 4 + digit; "00" is one token (16), "▁0" is 17, "▁" is 15, ":" is 14.
    assert_eq!(
        seq(&[P::T("1"), P::Img(2), P::T("2")]),
        [BOS, 5, BOI, IMAGE, IMAGE, EOI, 6, EOS]
    );
    // Two texts with nothing between them are one stretch: `0` and `0` merge.
    assert_eq!(seq(&[P::T("0"), P::T("0")]), [BOS, 16, EOS]);
    assert_eq!(seq(&[P::T("0"), P::T("0"), P::Img(1)]), [BOS, 16, BOI, IMAGE, EOI, EOS]);
    // The text before a video joins its first label's stretch: "0" + "00:00 " is "000:00 ",
    // `00`, `0` ... where the label alone would be `00`, `:`, `00`, `▁`.
    assert_eq!(
        seq(&[P::T("0"), P::VidT(&[1], &[0.0])]),
        [BOS, 16, 4, 14, 16, 15, BOI, VIDEO, EOI, EOS]
    );
    assert_eq!(
        seq(&[P::Aud(1), P::T("1"), P::Img(1)]),
        [BOS, BOA, AUDIO, EOA, 5, BOI, IMAGE, EOI, EOS]
    );
}

/// `video_sequence` and `video_sequence_with_timestamps` are the one-segment case of
/// the same builder.
#[test]
fn a_video_alone_is_the_video_sequences() {
    let cfg = configs().text;
    let tok = tokenizer();
    let soft = [3usize, 3];
    let stamps = [0.0, 61.5];
    assert_eq!(
        cfg.mixed_sequence(
            &tok,
            &[LayoutSegment::Video {
                soft_per_frame: &soft,
                timestamps: None
            }]
        )
        .unwrap(),
        cfg.video_sequence(&soft).unwrap()
    );
    assert_eq!(
        cfg.mixed_sequence(
            &tok,
            &[LayoutSegment::Video {
                soft_per_frame: &soft,
                timestamps: Some(&stamps)
            }]
        )
        .unwrap(),
        cfg.video_sequence_with_timestamps(&tok, &stamps, &soft).unwrap()
    );
    assert_eq!(
        cfg.mixed_sequence(&tok, &[LayoutSegment::Image { soft: 4 }]).unwrap(),
        cfg.image_sequence(4).unwrap()
    );
    assert_eq!(
        cfg.mixed_sequence(&tok, &[LayoutSegment::Audio { soft: 4 }]).unwrap(),
        cfg.audio_sequence(4).unwrap()
    );
}

#[test]
fn mixed_sequence_refusals_are_named() {
    let cfg = configs().text;
    let tok = tokenizer();
    let soft = [2usize];
    let stamps = [0.0];
    let video = |soft_per_frame, timestamps| LayoutSegment::Video {
        soft_per_frame,
        timestamps,
    };
    let big = String::from_utf8(vec![b'a'; MAX_TEXT_BYTES / 2]).unwrap();
    let cases: Vec<(&str, Vec<LayoutSegment<'_>>, &str)> = vec![
        ("no segments", vec![], "no segments"),
        (
            "an image of no soft tokens",
            vec![LayoutSegment::Text("1"), LayoutSegment::Image { soft: 0 }],
            "segment 1: the medium has no soft tokens",
        ),
        (
            "a clip of no soft tokens",
            vec![LayoutSegment::Audio { soft: 0 }],
            "segment 0: the medium has no soft tokens",
        ),
        (
            "a video of no frames",
            vec![video(&[], None)],
            "the video has no frames",
        ),
        (
            "a frame of no soft tokens",
            vec![video(&[2, 0], None)],
            "frame 1 has no soft tokens",
        ),
        (
            "timestamps for other frames",
            vec![video(&[2, 2], Some(&stamps))],
            "1 timestamps for 2 frames",
        ),
        (
            "a negative time",
            vec![video(&soft, Some(&[-1.0]))],
            "is not a time in 0..=",
        ),
        (
            "a time past the limit",
            vec![video(&soft, Some(&[MAX_TIMESTAMP_SECONDS * 2.0]))],
            "is not a time in 0..=",
        ),
        (
            "a NaN time",
            vec![video(&soft, Some(&[f64::NAN]))],
            "is not a time in 0..=",
        ),
        (
            "a huge count",
            vec![LayoutSegment::Image { soft: usize::MAX }],
            "overflows",
        ),
        (
            "text past the tokenizer's limit",
            vec![
                LayoutSegment::Text(&big),
                LayoutSegment::Image { soft: 1 },
                LayoutSegment::Text(&big),
                LayoutSegment::Text(&big),
            ],
            "segment 3: the text between two media",
        ),
    ];
    for (what, layout, want) in cases {
        let e = cfg.mixed_sequence(&tok, &layout).expect_err(what);
        assert!(e.contains(want), "{what}: {e}");
        assert!(e.starts_with("mixed_sequence"), "{what}: {e}");
    }
    // A config without the ids names every key the segments place, once.
    let bare = tessl::embedgemma2::EmbedGemma2Config::from_config_json(&config_json_without_ids()).unwrap();
    let e = bare
        .mixed_sequence(
            &tok,
            &[
                LayoutSegment::Text("1"),
                LayoutSegment::Image { soft: 1 },
                LayoutSegment::Audio { soft: 1 },
                LayoutSegment::Video {
                    soft_per_frame: &soft,
                    timestamps: None,
                },
            ],
        )
        .expect_err("accepted");
    for key in [
        "text_config.bos_token_id",
        "text_config.eos_token_id",
        "boi_token_id",
        "image_token_id",
        "eoi_token_id",
        "boa_token_id",
        "audio_token_id",
        "eoa_token_index",
        "video_token_id",
    ] {
        assert_eq!(e.matches(key).count(), 1, "{e} names {key} once");
    }
    // Only the ids a segment places are required.
    assert!(
        bare.mixed_sequence(&tok, &[LayoutSegment::Text("1")]).is_err(),
        "bos and eos"
    );
}

// ---------------------------------------------------------------------------
// Soft-token counts, without the media
// ---------------------------------------------------------------------------

fn pixels(len: usize, seed: u64) -> Vec<u8> {
    let mut r = SplitMix::new(seed);
    (0..len).map(|_| ((r.unit() + 1.0) * 127.5) as u8).collect()
}

#[test]
fn image_soft_tokens_are_what_preprocessing_makes() {
    let pre = PreConfig::from_json(&common::embedgemma2_mm::processor_config_json(false, 70)).unwrap();
    for (h, w) in [
        (8, 8),
        (13, 40),
        (200, 31),
        (3, 400),
        (64, 64),
        (1000, 1000),
        (16, 1),
        (1, 9),
    ] {
        let data = pixels(h * w * 3, (h * 1000 + w) as u64);
        let raw = RawImage::rgb(w, h, &data);
        match preprocess_image(&pre.image, &raw) {
            Ok(p) => assert_eq!(image_soft_tokens(&pre.image, h, w).unwrap(), p.soft_tokens, "{h}x{w}"),
            Err(e) => assert_eq!(image_soft_tokens(&pre.image, h, w).unwrap_err(), e, "{h}x{w}"),
        }
    }
    assert!(image_soft_tokens(&pre.image, 0, 5).is_err());
}

#[test]
fn audio_soft_tokens_are_the_towers_mask() {
    let pre = PreConfig::from_json(&common::embedgemma2_mm::processor_config_json(false, 70)).unwrap();
    let fe = AudioFrontEnd::new(&pre.audio).unwrap();
    let kept_max = pre.audio.max_samples.div_ceil(128) * 128;
    // Around every boundary of the frame arithmetic, and past the extractor's cut.
    let mut lengths: Vec<usize> = (1..=1100).collect();
    lengths.extend([
        1599,
        1600,
        3199,
        3200,
        16_000,
        47_999,
        48_000,
        kept_max - 1,
        kept_max,
        kept_max + 1,
        3 * kept_max,
    ]);
    for n in lengths {
        let pcm: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let want = fe.extract(&pcm, pre.audio.sampling_rate).unwrap();
        let mask = tessl::embedgemma2_audio::out_mask(&want.mask);
        assert_eq!(
            audio_soft_tokens(&pre.audio, n),
            mask.iter().filter(|&&k| k).count(),
            "{n} samples"
        );
    }
    assert_eq!(audio_soft_tokens(&pre.audio, 1), 0, "too short for a frame");
    // The 30 s cut caps it, whatever audio_seq_length says of the cap.
    assert_eq!(
        audio_soft_tokens(&pre.audio, 10 * kept_max),
        audio_soft_tokens(&pre.audio, kept_max)
    );
}

#[test]
fn video_soft_tokens_are_what_preprocessing_makes() {
    let pre = PreConfig::from_json(&common::embedgemma2_mm::processor_config_json(false, 70)).unwrap();
    for (frames, h, w, fps) in [
        (6usize, 16usize, 24usize, 2.0f64),
        (3, 40, 8, 1.0),
        (12, 9, 9, 24.0),
        (40, 8, 8, 1.0),
        (1, 12, 12, 30.0),
    ] {
        let data = pixels(frames * h * w * 3, frames as u64);
        let raws: Vec<RawImage<'_>> = (0..frames)
            .map(|f| RawImage::rgb(w, h, &data[f * h * w * 3..(f + 1) * h * w * 3]))
            .collect();
        let got = preprocess_video(&pre.video, &raws, fps).unwrap();
        assert_eq!(
            video_soft_tokens(&pre.video, h, w, frames, fps).unwrap(),
            (got.frames.len(), got.soft_tokens_per_frame),
            "{frames} frames of {h}x{w} at {fps}"
        );
    }
    let mut every = pre.video.clone();
    every.do_sample_frames = false;
    assert_eq!(video_soft_tokens(&every, 8, 8, 7, 1.0).unwrap().0, 7);
    assert!(video_soft_tokens(&pre.video, 8, 8, 0, 1.0).is_err());
    assert!(video_soft_tokens(&pre.video, 8, 8, 5, f64::NAN).is_err());
    assert!(video_soft_tokens(&pre.video, 8, 8, usize::MAX, 1.0).is_err());
}

// ---------------------------------------------------------------------------
// On a device: segments are the whole-string ids as a `Media` item
// ---------------------------------------------------------------------------

struct Fixture {
    c: Configs,
    st: SafeTensors,
}

fn fixture(tag: &str, operands: GemmOperands) -> Fixture {
    let c = configs();
    let st = open(tag, checkpoint_bytes(&c, 17, operands));
    Fixture { c, st }
}

impl Fixture {
    fn facade(&self, rt: &Arc<tessl::GpuRuntime>, operands: GemmOperands) -> EmbedGemma2Multimodal {
        let _ = PREFIX;
        EmbedGemma2Multimodal::load_from(rt, &self.st, &config_json(), operands).unwrap()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Media with the counts the tower will make, and the segments naming them.
struct Media {
    images: Vec<Image>,
    clips: Vec<AudioInput>,
    /// Two videos of frames.
    videos: Vec<Vec<Image>>,
}

fn media(c: &Configs) -> Media {
    let pd = c.vision.patch_dim() as usize;
    Media {
        // 3x2 patches with one padding patch: 6 patches, 1 soft token (pool 2x2) ... sized below.
        images: vec![
            Image::new(4, 2, 0, pd, 101),
            Image::new(2, 2, 4, pd, 102),
            Image::new(6, 2, 0, pd, 103),
        ],
        clips: vec![clip(&c.audio, 32, 201, |f| f < 28), clip(&c.audio, 24, 202, |f| f < 24)],
        videos: vec![
            (0..3).map(|k| Image::new(4, 4, 0, pd, 300 + k)).collect(),
            (0..2).map(|k| Image::new(2, 4, 0, pd, 310 + k)).collect(),
        ],
    }
}

fn counts(m: &EmbedGemma2Multimodal, media: &Media) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>) {
    fn inputs(images: &[Image]) -> Vec<ImageInput<'_>> {
        images.iter().map(Image::input).collect()
    }
    let images = m.vision().check_images(&inputs(&media.images)).unwrap();
    let clips = m.audio().check_clips(&media.clips).unwrap();
    let videos = media
        .videos
        .iter()
        .map(|v| m.vision().check_images(&inputs(v)).unwrap())
        .collect();
    (images, clips, videos)
}

#[test]
fn segment_items_are_the_whole_string_ids_as_a_media_item_bit_for_bit() {
    with_gpu(|rt| {
        let tok = tokenizer();
        for operands in MODES {
            let f = fixture("segments_media", operands);
            let m = f.facade(rt, operands);
            let md = media(&f.c);
            let (img, clp, vid) = counts(&m, &md);
            let stamps: Vec<Vec<f64>> = vec![vec![0.0, 1.0, 2.0], vec![0.0, 75.0]];
            let image_in: Vec<ImageInput<'_>> = md.images.iter().map(Image::input).collect();
            let frames: Vec<Vec<ImageInput<'_>>> =
                md.videos.iter().map(|v| v.iter().map(Image::input).collect()).collect();

            // (name, segments, the same message as pieces with counts, the media in order)
            struct Case<'a> {
                name: &'static str,
                segments: Vec<MmSegment<'a>>,
                pieces: Vec<P<'a>>,
                images: Vec<ImageInput<'a>>,
                clips: Vec<AudioInput>,
                videos: Vec<&'a [ImageInput<'a>]>,
            }
            let cases = vec![
                Case {
                    name: "text image text",
                    segments: vec![
                        MmSegment::Text("a 0"),
                        MmSegment::Image(image_in[0]),
                        MmSegment::Text("0 b"),
                    ],
                    pieces: vec![P::T("a 0"), P::Img(img[0]), P::T("0 b")],
                    images: vec![image_in[0]],
                    clips: vec![],
                    videos: vec![],
                },
                Case {
                    name: "clip",
                    segments: vec![
                        MmSegment::Text("1:"),
                        MmSegment::Audio(&md.clips[0]),
                        MmSegment::Text(" 2"),
                    ],
                    pieces: vec![P::T("1:"), P::Aud(clp[0]), P::T(" 2")],
                    images: vec![],
                    clips: vec![md.clips[0].clone()],
                    videos: vec![],
                },
                Case {
                    name: "images and a clip in segment order",
                    segments: vec![
                        MmSegment::Image(image_in[2]),
                        MmSegment::Text("0"),
                        MmSegment::Audio(&md.clips[1]),
                        MmSegment::Image(image_in[1]),
                        MmSegment::Text("00"),
                        MmSegment::Image(image_in[0]),
                    ],
                    pieces: vec![
                        P::Img(img[2]),
                        P::T("0"),
                        P::Aud(clp[1]),
                        P::Img(img[1]),
                        P::T("00"),
                        P::Img(img[0]),
                    ],
                    images: vec![image_in[2], image_in[1], image_in[0]],
                    clips: vec![md.clips[1].clone()],
                    videos: vec![],
                },
                Case {
                    name: "video",
                    segments: vec![
                        MmSegment::Text("go"),
                        MmSegment::Video(&frames[0]),
                        MmSegment::Text("0"),
                    ],
                    pieces: vec![P::T("go"), P::Vid(&vid[0]), P::T("0")],
                    images: vec![],
                    clips: vec![],
                    videos: vec![&frames[0]],
                },
                Case {
                    name: "timestamped video then an image then another video",
                    segments: vec![
                        MmSegment::Text("0"),
                        MmSegment::TimestampedVideo {
                            frames: &frames[0],
                            timestamps: &stamps[0],
                        },
                        MmSegment::Image(image_in[1]),
                        MmSegment::Video(&frames[1]),
                    ],
                    pieces: vec![P::T("0"), P::VidT(&vid[0], &stamps[0]), P::Img(img[1]), P::Vid(&vid[1])],
                    images: vec![image_in[1]],
                    clips: vec![],
                    videos: vec![&frames[0], &frames[1]],
                },
                Case {
                    name: "everything",
                    segments: vec![
                        MmSegment::Text("x "),
                        MmSegment::Image(image_in[0]),
                        MmSegment::Video(&frames[1]),
                        MmSegment::Text("0"),
                        MmSegment::Audio(&md.clips[0]),
                        MmSegment::TimestampedVideo {
                            frames: &frames[1],
                            timestamps: &stamps[1],
                        },
                    ],
                    pieces: vec![
                        P::T("x "),
                        P::Img(img[0]),
                        P::Vid(&vid[1]),
                        P::T("0"),
                        P::Aud(clp[0]),
                        P::VidT(&vid[1], &stamps[1]),
                    ],
                    images: vec![image_in[0]],
                    clips: vec![md.clips[0].clone()],
                    videos: vec![&frames[1], &frames[1]],
                },
            ];
            for case in &cases {
                let ids = tok.encode_raw(&processor_string(&case.pieces), true);
                let by_ids = m
                    .encode(
                        &[MmItem::Media {
                            ids: &ids,
                            images: &case.images,
                            audio: &case.clips,
                            videos: &case.videos,
                        }],
                        None,
                    )
                    .unwrap();
                let (out, trace) = {
                    infer_trace::set_enabled(true);
                    let s0 = infer_trace::snapshot();
                    let out = m
                        .encode(
                            &[MmItem::Segments {
                                segments: &case.segments,
                                tokenizer: &tok,
                            }],
                            None,
                        )
                        .unwrap();
                    let s1 = infer_trace::snapshot();
                    infer_trace::set_enabled(false);
                    (out, s1.since(&s0))
                };
                assert_eq!(
                    bits(&out.embeddings),
                    bits(&by_ids.embeddings),
                    "{} ({operands:?}): segments differ from the whole-string ids",
                    case.name
                );
                assert_eq!(trace.sync_waits, 1, "{} ({operands:?}): one wait", case.name);
            }

            // A batch of them equals each alone when the text side packs one forward.
            let batch: Vec<MmItem<'_>> = cases
                .iter()
                .map(|c| MmItem::Segments {
                    segments: &c.segments,
                    tokenizer: &tok,
                })
                .collect();
            let all = m.encode(&batch, Some(24)).unwrap();
            let dim = all.embeddings.len() / cases.len();
            for (i, c) in cases.iter().enumerate() {
                let alone = m.encode(&[batch[i]], Some(24)).unwrap();
                let worst = all.embeddings[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(&alone.embeddings)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(worst <= 1e-5, "{} ({operands:?}): in a batch vs alone: {worst}", c.name);
            }
        }
    });
}

#[test]
fn segment_refusals_come_before_any_gpu_work() {
    with_gpu(|rt| {
        let tok = tokenizer();
        let f = fixture("segments_refusals", GemmOperands::ExactF32);
        let m = f.facade(rt, GemmOperands::ExactF32);
        let md = media(&f.c);
        let img = md.images[0].input();
        let ok = [MmSegment::Text("0"), MmSegment::Image(img)];
        let frames_a: Vec<ImageInput<'_>> = md.videos[0].iter().map(Image::input).collect();
        // One frame the size of another video's: different soft-token counts in one video.
        let mixed_frames = [md.videos[0][0].input(), md.images[1].input()];
        fn item<'a>(segments: &'a [MmSegment<'a>], tokenizer: &'a Tokenizer) -> MmItem<'a> {
            MmItem::Segments { segments, tokenizer }
        }
        let cases: Vec<(&str, Vec<MmSegment<'_>>, &str)> = vec![
            ("no segments", vec![], "no segments"),
            (
                "a video with frames of different sizes",
                vec![MmSegment::Video(&mixed_frames)],
                "frame 1 makes",
            ),
            ("no frames", vec![MmSegment::Video(&[])], "the video has no frames"),
            (
                "timestamps for the wrong number of frames",
                vec![MmSegment::TimestampedVideo {
                    frames: &frames_a,
                    timestamps: &[0.0],
                }],
                "1 timestamps for 3 frames",
            ),
            (
                "a negative time",
                vec![MmSegment::TimestampedVideo {
                    frames: &frames_a[..1],
                    timestamps: &[-3.0],
                }],
                "is not a time",
            ),
        ];
        for (what, segments, want) in cases {
            infer_trace::set_enabled(true);
            let s0 = infer_trace::snapshot();
            let e = match m.encode(&[item(&segments, &tok)], None) {
                Err(e) => e,
                Ok(_) => panic!("{what}: accepted"),
            };
            let d = infer_trace::snapshot().since(&s0);
            infer_trace::set_enabled(false);
            assert!(e.contains(want), "{what}: {e}");
            assert!(e.contains("item 0"), "{what}: {e}");
            assert_eq!(
                (d.cold_allocs, d.sync_waits),
                (0, 0),
                "{what}: a refusal allocated or waited"
            );
        }
        // The facade works afterwards.
        m.encode(&[item(&ok, &tok)], None).unwrap();
    });
}
