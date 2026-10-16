//! Video through `EmbedGemma2Multimodal` (`MmItem::Video`, and `MmItem::Media`
//! with `videos`) on the tiny random checkpoint of
//! `tests/common/embedgemma2_mm.rs`. It runs wherever `cargo test` reaches a
//! GPU, with no real checkpoint (`tests/embedgemma2_video_model.rs` is the real
//! one, against sentence-transformers).
//!
//! A video is a list of frames, each the processor's patch tensors, and each
//! frame is an image to the vision tower: its `S` soft tokens are the rows of
//! one `<boi> <video> x S <eoi>` block of the sequence
//! `[<bos>, block, block, ..., <eos>]` (`EmbedGemma2Config::video_sequence`).
//! The oracle is the composition done by hand: `vision.encode_images` on the
//! frames, the ids written out below, `text.encode_multimodal` on host soft
//! rows. Bounds, fixed before the first run:
//!
//! * a video item through the facade is the by-hand composition bit for bit
//!   (`0`) in both text modes, including a video whose frames the tower cuts
//!   into several forwards, and a `Media` item whose ids carry text between the
//!   frames' blocks (where a caller puts timestamps) and two videos;
//! * the frame order is observable (reversed frames give another embedding) and
//!   so is the order of videos in a `Media` item;
//! * the ids are the processor's: bos, then per frame boi, `S` video
//!   placeholders, eoi, then eos, with no separator (checked against a layout
//!   written out here, and for the real processor in `embedgemma2_video_model`);
//! * a video, an image, a clip, caller-built ids and two texts in one call each
//!   equal themselves alone bit for bit when the text side packs one forward,
//!   and the call waits for the GPU once (3 for the by-hand composition);
//! * a refused call (no frames, frames of different sizes, a placeholder count
//!   that is not the frames' soft tokens, more frames than the tower's batch,
//!   a sequence past the limit) allocates nothing and waits for nothing, and
//!   the facade works afterwards.

mod common;

use std::sync::Arc;

use common::embedgemma2::{open, PREFIX};
use common::embedgemma2_mm::{checkpoint_bytes, clip, config_json, config_json_without_ids, configs, Configs, Image};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Model, MmSequence};
use tessl::embedgemma2_audio::AudioInput;
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_pre::PreConfig;
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput, MAX_IMAGES};
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;

const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];
/// The text encoder's batch bound (`tests/embedgemma2_mm_tiny.rs`).
const BATCH_ABS: f64 = 1e-5;

/// The ids `common::embedgemma2_mm::config_json` carries.
const BOS: u32 = 2;
const EOS: u32 = 1;
const BOI: u32 = 40;
const EOI: u32 = 41;
const IMAGE: u32 = 60;
const VIDEO: u32 = 62;

fn traced<T>(f: impl FnOnce() -> T) -> (T, infer_trace::Snapshot) {
    infer_trace::set_enabled(true);
    let s0 = infer_trace::snapshot();
    let out = f();
    let s1 = infer_trace::snapshot();
    infer_trace::set_enabled(false);
    (out, s1.since(&s0))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn drift(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    f64::from(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max))
}

/// `n` frames of a `w x h` patch grid with `pad` padding patches, each with
/// its own pixels.
fn frames(c: &Configs, n: usize, (w, h): (i32, i32), pad: usize, seed: u64) -> Vec<Image> {
    let pd = c.vision.patch_dim() as usize;
    (0..n).map(|k| Image::new(w, h, pad, pd, seed + k as u64)).collect()
}

fn inputs(images: &[Image]) -> Vec<ImageInput<'_>> {
    images.iter().map(Image::input).collect()
}

/// The processor's ids for a video, written out (not through the builder).
fn hand_ids(soft_per_frame: &[usize]) -> Vec<u32> {
    let mut ids = vec![BOS];
    for &n in soft_per_frame {
        ids.push(BOI);
        ids.extend(std::iter::repeat_n(VIDEO, n));
        ids.push(EOI);
    }
    ids.push(EOS);
    ids
}

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
        EmbedGemma2Multimodal::load_from(rt, &self.st, &config_json(), operands).unwrap()
    }
}

/// The by-hand composition: the tower's soft rows on the host, written under
/// the ids' placeholders, then the text encoder.
struct ByHand {
    text: EmbedGemma2Model,
    vision: EmbedGemma2VisionModel,
}

impl ByHand {
    fn new(f: &Fixture, rt: &Arc<tessl::GpuRuntime>, operands: GemmOperands) -> Self {
        Self {
            text: EmbedGemma2Model::load_with(rt, &f.st, PREFIX, f.c.text.clone(), operands).unwrap(),
            vision: EmbedGemma2VisionModel::load_with(rt, &f.st, f.c.vision.clone(), operands).unwrap(),
        }
    }

    /// `ids` with the soft rows of `images` under the image placeholders and of
    /// `frames` under the video placeholders, each laid end to end.
    fn embed(&self, ids: &[u32], images: &[ImageInput<'_>], frames: &[ImageInput<'_>]) -> Vec<f32> {
        let h = self.text.config().hidden as usize;
        let rows = |pics: &[ImageInput<'_>]| -> Vec<f32> {
            if pics.is_empty() {
                Vec::new()
            } else {
                self.vision.encode_images(pics, false).unwrap().soft.concat()
            }
        };
        let (img, vid) = (rows(images), rows(frames));
        let (mut i, mut v) = (0usize, 0usize);
        let mut soft = Vec::new();
        for &id in ids {
            if id == IMAGE {
                soft.extend_from_slice(&img[i * h..(i + 1) * h]);
                i += 1;
            } else if id == VIDEO {
                soft.extend_from_slice(&vid[v * h..(v + 1) * h]);
                v += 1;
            }
        }
        assert_eq!((i * h, v * h), (img.len(), vid.len()), "test bug: placeholder count");
        self.text
            .encode_multimodal(&[MmSequence { ids, soft: &soft }], None, false)
            .unwrap()
            .embeddings
    }
}

// ---------------------------------------------------------------------------
// Device-free: the id layout and the config.
// ---------------------------------------------------------------------------

#[test]
fn video_sequence_is_the_processors_layout() {
    let cfg = configs().text;
    assert_eq!(
        cfg.video_sequence(&[3, 3]).unwrap(),
        [BOS, BOI, VIDEO, VIDEO, VIDEO, EOI, BOI, VIDEO, VIDEO, VIDEO, EOI, EOS]
    );
    // One frame is an image's block with the video placeholder.
    assert_eq!(
        cfg.video_sequence(&[2]).unwrap(),
        cfg.image_sequence(2)
            .unwrap()
            .iter()
            .map(|&t| if t == IMAGE { VIDEO } else { t })
            .collect::<Vec<_>>()
    );
    // The real checkpoint's counts: 32 frames of 140.
    let ids = cfg.video_sequence(&[140; 32]).unwrap();
    assert_eq!(ids.len(), 2 + 32 * 142);
    assert_eq!(ids, hand_ids(&[140; 32]));
    // Frames may differ (a caller's own sizes).
    assert_eq!(cfg.video_sequence(&[1, 2]).unwrap(), hand_ids(&[1, 2]));

    for (what, counts, want) in [
        ("no frames", &[][..], "the video has no frames"),
        (
            "a frame without soft tokens",
            &[4, 0, 4][..],
            "frame 1 has no soft tokens",
        ),
        ("a length that overflows", &[usize::MAX, 1][..], "overflows"),
    ] {
        let e = cfg
            .video_sequence(counts)
            .err()
            .unwrap_or_else(|| panic!("{what}: accepted"));
        assert!(e.contains(want), "{what}: {e}");
    }
    // A config without the ids names what is missing.
    let bare = tessl::embedgemma2::EmbedGemma2Config::from_config_json(&config_json_without_ids()).unwrap();
    let e = bare.video_sequence(&[2]).expect_err("accepted");
    for key in [
        "text_config.bos_token_id",
        "boi_token_id",
        "video_token_id",
        "eoi_token_id",
    ] {
        assert!(e.contains(key), "{e} does not name {key}");
    }
}

/// The snapshot's `processor_config.json`, with `add_timestamps` as given.
fn processor_config(add_timestamps: &str) -> String {
    format!(
        r#"{{
      "audio_ms_per_token": 40, "audio_seq_length": 280,
      "feature_extractor": {{"dither": 0.0, "feature_extractor_type": "Gemma4AudioFeatureExtractor",
        "feature_size": 128, "fft_length": 512, "fft_overdrive": false, "frame_length": 320,
        "hop_length": 160, "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001,
        "min_frequency": 0.0, "padding_side": "right", "padding_value": 0.0, "per_bin_mean": null,
        "per_bin_stddev": null, "preemphasis": 0.0, "preemphasis_htk_flavor": true,
        "return_attention_mask": true, "sampling_rate": 16000}},
      "image_processor": {{"do_convert_rgb": true, "do_normalize": false, "do_rescale": true,
        "do_resize": true, "image_mean": [0.0, 0.0, 0.0], "image_processor_type": "Gemma4ImageProcessor",
        "image_seq_length": 280, "image_std": [1.0, 1.0, 1.0], "max_soft_tokens": 280, "patch_size": 16,
        "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098}},
      "video_processor": {{"add_timestamps": {add_timestamps}, "do_convert_rgb": true, "do_normalize": false,
        "do_rescale": true, "do_resize": true, "do_sample_frames": true, "fps": 1,
        "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0], "max_frames": 32,
        "max_soft_tokens": 140, "overflow_strategy": "uniform", "patch_size": 16,
        "pooling_kernel_size": 3, "resample": 3, "rescale_factor": 0.00392156862745098,
        "return_metadata": false, "video_processor_type": "EmbeddingGemma2VideoProcessor"}}
    }}"#
    )
}

/// `add_timestamps` puts `mm:ss` text in the ids, which takes the tokenizer; the
/// layout `video_sequence` builds is the one without, so a config asking for
/// timestamps is refused by name instead of read as if it did not.
#[test]
fn a_config_with_timestamps_on_is_refused_by_name() {
    PreConfig::from_json(&processor_config("false")).expect("the snapshot's own setting");
    let e = PreConfig::from_json(&processor_config("true")).expect_err("timestamps on were read as off");
    assert!(
        e.contains("video_processor.add_timestamps") && e.contains("not supported"),
        "{e}"
    );
}

// ---------------------------------------------------------------------------
// On the device.
// ---------------------------------------------------------------------------

#[test]
fn a_video_item_equals_the_by_hand_composition() {
    for operands in MODES {
        let f = fixture("video_item", operands);
        let vid = frames(&f.c, 5, (6, 4), 8, 100);
        let fr = inputs(&vid);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let counts = by_hand.vision.check_images(&fr).unwrap();
            assert_eq!(counts, [6; 5], "test bug: 6x4 patches pooled 2x2");
            let ids = hand_ids(&counts);
            assert_eq!(m.text().config().video_sequence(&counts).unwrap(), ids);
            let want = by_hand.embed(&ids, &[], &fr);

            let (got, t) = traced(|| m.encode(&[MmItem::Video(&fr)], None).unwrap());
            assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}");
            assert_eq!(
                t.sync_waits, 1,
                "{operands:?}: the tower and the text encoder wait once"
            );

            // The soft rows of a frame are the same alone as in the video.
            let together = by_hand.vision.encode_images(&fr, false).unwrap();
            for (k, one) in fr.iter().enumerate() {
                let alone = by_hand.vision.encode_images(&[*one], false).unwrap();
                assert_eq!(bits(&alone.soft[0]), bits(&together.soft[k]), "{operands:?}: frame {k}");
            }

            // The order of the frames is observable.
            let reversed: Vec<ImageInput<'_>> = fr.iter().rev().copied().collect();
            let other = m.encode(&[MmItem::Video(&reversed)], None).unwrap();
            assert_ne!(
                bits(&other.embeddings),
                bits(&got.embeddings),
                "{operands:?}: frame order is not observable"
            );
            // And it is the by-hand composition of the reversed frames.
            let want_rev = by_hand.embed(&ids, &[], &reversed);
            assert_eq!(bits(&other.embeddings), bits(&want_rev), "{operands:?}: reversed");

            // The Matryoshka prefix goes through.
            let d = 40u32;
            let pre = m.encode(&[MmItem::Video(&fr)], Some(d)).unwrap();
            let full = by_hand
                .text
                .encode_multimodal(
                    &[MmSequence {
                        ids: &ids,
                        soft: &together.soft.concat(),
                    }],
                    Some(d),
                    false,
                )
                .unwrap();
            assert_eq!(bits(&pre.embeddings), bits(&full.embeddings), "{operands:?}: prefix");
        });
    }
}

/// More frames than one tower forward holds: the soft rows sit in two output
/// buffers, in frame order only by the placeholder walk, not by position.
#[test]
fn frames_spread_over_several_tower_forwards_keep_their_order() {
    for operands in MODES {
        let f = fixture("video_forwards", operands);
        // 800 patches a frame: 20 fit one forward (16,384 padded patches).
        let vid = frames(&f.c, 25, (20, 40), 0, 300);
        let fr = inputs(&vid);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let tower = by_hand.vision.encode_images(&fr, false).unwrap();
            assert!(tower.forwards >= 2, "test bug: {} tower forwards", tower.forwards);
            let counts = by_hand.vision.check_images(&fr).unwrap();
            assert_eq!(counts, [200; 25]);
            let ids = hand_ids(&counts);
            let want = by_hand.embed(&ids, &[], &fr);
            let (got, t) = traced(|| m.encode(&[MmItem::Video(&fr)], None).unwrap());
            assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}");
            assert_eq!(
                t.sync_waits, 1,
                "{operands:?}: {} tower forwards, one wait",
                tower.forwards
            );
        });
    }
}

/// Caller-built ids: text between the frames' blocks (where timestamps go), an
/// image among them, and two videos whose frames the placeholders take end to
/// end, so a facade that gave each video its own run, or took the frames in
/// another order, would come out different.
#[test]
fn media_items_take_video_frames_in_placeholder_order() {
    for operands in MODES {
        let f = fixture("video_media", operands);
        let a = frames(&f.c, 3, (6, 4), 8, 10);
        let b = frames(&f.c, 2, (4, 4), 0, 20);
        let img = frames(&f.c, 1, (2, 2), 0, 30);
        let (fa, fb, fi) = (inputs(&a), inputs(&b), inputs(&img));
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let ca = by_hand.vision.check_images(&fa).unwrap();
            let cb = by_hand.vision.check_images(&fb).unwrap();
            let ci = by_hand.vision.check_images(&fi).unwrap();
            assert_eq!((ca[0], cb[0], ci[0]), (6, 4, 1), "test bug: soft tokens per frame");

            // [bos, "00:00", block(a0), "00:01", block(a1), image, "00:02",
            //  block(a2) block(b0) (no text between), "00:03", block(b1), eos]
            let (t0, t1, t2, t3) = ([7u32, 8], [9u32, 10], [11u32, 12], [13u32, 14]);
            let block = |n: usize| {
                let mut v = vec![BOI];
                v.extend(std::iter::repeat_n(VIDEO, n));
                v.push(EOI);
                v
            };
            let mut ids = vec![BOS];
            ids.extend(t0);
            ids.extend(block(ca[0]));
            ids.extend(t1);
            ids.extend(block(ca[1]));
            ids.extend([BOI, IMAGE, EOI]);
            ids.extend(t2);
            ids.extend(block(ca[2]));
            ids.extend(block(cb[0]));
            ids.extend(t3);
            ids.extend(block(cb[1]));
            ids.push(EOS);

            let all_frames: Vec<ImageInput<'_>> = fa.iter().chain(&fb).copied().collect();
            let want = by_hand.embed(&ids, &fi, &all_frames);
            let videos: [&[ImageInput<'_>]; 2] = [&fa, &fb];
            let item = MmItem::Media {
                ids: &ids,
                images: &fi,
                audio: &[],
                videos: &videos,
            };
            let (got, t) = traced(|| m.encode(&[item], None).unwrap());
            assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}");
            assert_eq!(t.sync_waits, 1);

            // The order of the videos is observable.
            let swapped: [&[ImageInput<'_>]; 2] = [&fb, &fa];
            // The counts are checked as totals, so the swap is accepted and the
            // embedding must move.
            let other = m
                .encode(
                    &[MmItem::Media {
                        ids: &ids,
                        images: &fi,
                        audio: &[],
                        videos: &swapped,
                    }],
                    None,
                )
                .unwrap();
            assert_ne!(
                bits(&other.embeddings),
                bits(&got.embeddings),
                "{operands:?}: the order of videos is not observable"
            );

            // A Video item is a Media item with the processor's ids.
            let vid_ids = hand_ids(&ca);
            let as_media = m
                .encode(
                    &[MmItem::Media {
                        ids: &vid_ids,
                        images: &[],
                        audio: &[],
                        videos: &[&fa],
                    }],
                    None,
                )
                .unwrap();
            let as_video = m.encode(&[MmItem::Video(&fa)], None).unwrap();
            assert_eq!(bits(&as_media.embeddings), bits(&as_video.embeddings), "{operands:?}");
        });
    }
}

/// A batch of every kind of item equals each item alone.
#[test]
fn a_mixed_batch_with_video_equals_each_item_alone() {
    for operands in MODES {
        let f = fixture("video_mixed", operands);
        let a = frames(&f.c, 4, (6, 4), 8, 40);
        let b = frames(&f.c, 3, (4, 4), 4, 50);
        let img = frames(&f.c, 2, (5, 4), 28, 60);
        let (fa, fb, fi) = (inputs(&a), inputs(&b), inputs(&img));
        let au: AudioInput = clip(&f.c.audio, 41, 1, |_| true);
        let mut r = SplitMix::new(21);
        let text: Vec<u32> = (0..34).map(|_| r.range(3, 46) as u32).collect();
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let cb = by_hand.vision.check_images(&fb).unwrap();
            let ci = by_hand.vision.check_images(&fi).unwrap();
            let mut media_ids = vec![BOS, 5];
            media_ids.extend(std::iter::repeat_n(IMAGE, ci[0]));
            media_ids.push(6);
            media_ids.extend(std::iter::repeat_n(VIDEO, cb.iter().sum::<usize>()));
            media_ids.push(EOS);
            let vids: [&[ImageInput<'_>]; 1] = [&fb];
            let items = [
                MmItem::Text(&text),
                MmItem::Video(&fa),
                MmItem::Image(fi[1]),
                MmItem::Audio(&au),
                MmItem::Media {
                    ids: &media_ids,
                    images: &fi[..1],
                    audio: &[],
                    videos: &vids,
                },
                MmItem::Video(&fb),
                MmItem::Text(&text[..30]),
            ];
            let (batch, t) = traced(|| m.encode(&items, None).unwrap());
            assert_eq!(t.sync_waits, 1, "{operands:?}: a mixed call waits once");
            assert_eq!(
                batch.forwards, 1,
                "test bug: pick a batch the packer runs as one forward"
            );
            let dim = m.text().config().embedding_dim as usize;
            for (k, item) in items.iter().enumerate() {
                let alone = m.encode(std::slice::from_ref(item), None).unwrap();
                assert_eq!(
                    bits(&batch.embeddings[k * dim..(k + 1) * dim]),
                    bits(&alone.embeddings),
                    "{operands:?}: item {k} in the batch vs alone"
                );
            }
            // The video items are also the by-hand composition.
            for (k, fr) in [(1usize, &fa), (5, &fb)] {
                let counts = by_hand.vision.check_images(fr).unwrap();
                let want = by_hand.embed(&hand_ids(&counts), &[], fr);
                assert_eq!(
                    bits(&batch.embeddings[k * dim..(k + 1) * dim]),
                    bits(&want),
                    "{operands:?}: item {k} vs by hand"
                );
            }
            // Poisoning every unzeroed activation changes no bit.
            rt.set_poison_unzeroed(true);
            let poisoned = m.encode(&items, None);
            rt.set_poison_unzeroed(false);
            assert_eq!(
                bits(&poisoned.unwrap().embeddings),
                bits(&batch.embeddings),
                "{operands:?}: poisoned"
            );
        });
    }
}

/// A batch whose text side needs several forwards still waits once, and every
/// item agrees with itself alone within the text encoder's batch bound.
#[test]
fn a_batch_of_several_forwards_with_video_agrees_within_the_batch_bound() {
    for operands in MODES {
        let f = fixture("video_forwards_mixed", operands);
        let long = frames(&f.c, 25, (20, 40), 0, 70);
        let short = frames(&f.c, 3, (6, 4), 8, 80);
        let (fl, fs) = (inputs(&long), inputs(&short));
        let mut r = SplitMix::new(33);
        let text: Vec<u32> = (0..7).map(|_| r.range(3, 46) as u32).collect();
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let items = [MmItem::Video(&fl), MmItem::Text(&text), MmItem::Video(&fs)];
            let (batch, t) = traced(|| m.encode(&items, None).unwrap());
            assert!(batch.forwards >= 2, "test bug: {} text forwards", batch.forwards);
            assert_eq!(t.sync_waits, 1, "{operands:?}: several forwards, still one wait");
            let dim = m.text().config().embedding_dim as usize;
            for (k, item) in items.iter().enumerate() {
                let alone = m.encode(std::slice::from_ref(item), None).unwrap();
                let d = drift(&batch.embeddings[k * dim..(k + 1) * dim], &alone.embeddings);
                eprintln!(
                    "{operands:?}: item {k} in a {}-forward batch vs alone: {d:.3e}",
                    batch.forwards
                );
                assert!(d <= BATCH_ABS, "{operands:?}: item {k}: {d:.3e}");
            }
        });
    }
}

#[test]
fn video_items_are_refused_before_any_gpu_work() {
    let f = fixture("video_refuse", GemmOperands::ExactF32);
    let good = frames(&f.c, 3, (6, 4), 8, 90);
    let other_size = frames(&f.c, 1, (5, 4), 0, 95);
    let fg = inputs(&good);
    let mixed_sizes: Vec<ImageInput<'_>> = fg.iter().copied().chain(inputs(&other_size)).collect();
    let pd = f.c.vision.patch_dim() as usize;
    let many: Vec<Image> = (0..MAX_IMAGES as u64 + 1).map(|k| Image::new(1, 1, 3, pd, k)).collect();
    let many_in = inputs(&many);
    let mut bad_pos = Image::new(2, 2, 0, pd, 7);
    bad_pos.position_ids[2] = 999;
    let with_bad: Vec<ImageInput<'_>> = fg.iter().copied().chain([bad_pos.input()]).collect();
    let text_with_video = [BOS, 5, VIDEO, EOS];
    let one_video_ph = [BOS, VIDEO, EOS];
    let no_ph = [BOS, 5, EOS];
    let no_frames: [ImageInput<'_>; 0] = [];
    let one_video: [&[ImageInput<'_>]; 1] = [&fg];

    with_gpu(|rt| {
        let mut m = f.facade(rt, GemmOperands::ExactF32);
        let ok = [MmItem::Video(&fg)];
        let want = m.encode(&ok, None).unwrap();
        let sizes = m.vision().check_images(&mixed_sizes).unwrap();
        assert_ne!(sizes[3], sizes[0], "test bug: the last frame must differ in size");
        let different = format!("frame 3 makes {} soft tokens but frame 0 makes {}", sizes[3], sizes[0]);

        let cases: Vec<(&str, Vec<MmItem<'_>>, &str)> = vec![
            (
                "a video of no frames",
                vec![MmItem::Video(&no_frames)],
                "the video has no frames",
            ),
            (
                "a video of no frames beside good items",
                vec![MmItem::Video(&fg), MmItem::Video(&no_frames)],
                "item 1",
            ),
            (
                "frames of different sizes",
                vec![MmItem::Video(&mixed_sizes)],
                &different,
            ),
            (
                "more frames than the tower's batch",
                vec![MmItem::Video(&many_in)],
                "exceeds the batch limit",
            ),
            (
                "a bad frame after good ones",
                vec![MmItem::Video(&with_bad)],
                "outside the position table",
            ),
            (
                "a text item holding a video placeholder",
                vec![MmItem::Text(&text_with_video)],
                "item 0 is text but its ids hold 1 placeholder ids",
            ),
            (
                "video placeholders without frames",
                vec![MmItem::Media {
                    ids: &one_video_ph,
                    images: &[],
                    audio: &[],
                    videos: &[],
                }],
                "has 1 video placeholder ids",
            ),
            (
                "frames without video placeholders",
                vec![MmItem::Media {
                    ids: &no_ph,
                    images: &[],
                    audio: &[],
                    videos: &one_video,
                }],
                "has 0 video placeholder ids",
            ),
            (
                "too few video placeholders",
                vec![MmItem::Media {
                    ids: &one_video_ph,
                    images: &[],
                    audio: &[],
                    videos: &one_video,
                }],
                "make 18 soft tokens",
            ),
        ];
        for (what, items, want_msg) in cases {
            let (res, t) = traced(|| m.encode(&items, None));
            let e = res.err().unwrap_or_else(|| panic!("{what}: accepted"));
            assert!(e.contains(want_msg), "{what}: {e:?} does not mention {want_msg:?}");
            assert!(
                e.starts_with("EmbedGemma2Multimodal::encode"),
                "{what}: names its entry: {e}"
            );
            assert_eq!(
                (t.cold_allocs, t.sync_waits),
                (0, 0),
                "{what}: did GPU work before refusing"
            );
        }
        // Still usable, and unchanged.
        let again = m.encode(&ok, None).unwrap();
        assert_eq!(bits(&again.embeddings), bits(&want.embeddings));
        // A tightened limit refuses a video whose sequence no longer fits.
        m.text_mut().set_max_tokens(20).unwrap();
        let (res, t) = traced(|| m.encode(&ok, None));
        let e = res.err().expect("accepted");
        assert!(e.contains("the limit is"), "{e}");
        assert_eq!((t.cold_allocs, t.sync_waits), (0, 0));
    });
}
