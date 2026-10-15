//! `EmbedGemma2Multimodal` (`tessl::embedgemma2_mm`): the text encoder and both
//! towers behind one snapshot and one call, on the tiny random checkpoint of
//! `tests/common/embedgemma2_mm.rs`. It runs wherever `cargo test` reaches a
//! GPU, with no real checkpoint (`tests/embedgemma2_multimodal_model.rs` is the
//! real one).
//!
//! The facade is a composition, so its oracle is the composition done by hand:
//! `vision.encode_images` / `audio.encode_audio`, the placeholder layout built
//! here from the ids, `text.encode_multimodal`. The pieces are held to f64
//! references in `embedgemma2_{tiny,vision_tiny,audio_tiny,mm_tiny}.rs`; what
//! only this file can see is that the glue (the soft-token counts, their order
//! across modalities, the batching, the waits) is right.
//!
//! Bounds, fixed before the first run:
//!
//! * an item through the facade equals the by-hand composition of the same
//!   models bit for bit (`0`), in both text modes (`ExactF32`, `Bf16`);
//! * a text item equals `EmbedGemma2Model::encode` bit for bit;
//! * an item inside a mixed batch equals itself alone bit for bit when the text
//!   encoder packs both calls into one forward (the towers are bit-for-bit
//!   batch-independent), and within `1e-5` max abs otherwise (the text
//!   encoder's own batch bound, `embedgemma2_mm_tiny.rs`);
//! * host waits (`infer_trace`): a call of text items makes exactly 1, a call
//!   with media exactly 2 (one for both towers' queued forwards, one for the
//!   text encoder's), against 3 for the by-hand composition (vision, audio,
//!   text each wait on their own);
//! * a refused call allocates nothing and waits for nothing, and the facade
//!   works afterwards.

mod common;

use common::embedgemma2::{open, PREFIX};
use common::embedgemma2_mm::{
    self as mm, checkpoint_bytes, clip, config_json, config_json_without_ids, configs, Configs, Image, AUDIO, IMAGE,
    VIDEO,
};
use common::{with_gpu, with_two_gpus, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Model, MmSequence, MAX_BATCH};
use tessl::embedgemma2_audio::{AudioInput, EmbedGemma2Audio};
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::embedgemma2_vision::{EmbedGemma2VisionModel, ImageInput};
use tessl::gemm::GemmOperands;
use tessl::infer_trace;
use tessl::safetensors::SafeTensors;

const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];
/// The text encoder's batch bound (`tests/embedgemma2_mm_tiny.rs`).
const BATCH_ABS: f64 = 1e-5;

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

/// Everything a test needs: the three models loaded the facade's way and the
/// same models for the by-hand composition.
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
    /// The facade, loaded from the checkpoint and its config.
    fn facade(&self, rt: &std::sync::Arc<tessl::GpuRuntime>, operands: GemmOperands) -> EmbedGemma2Multimodal {
        EmbedGemma2Multimodal::load_from(rt, &self.st, &config_json(), operands).unwrap()
    }
}

/// The by-hand composition of one item's soft tokens: the same three models,
/// the facade's pieces called directly, the placeholder walk written out here.
struct ByHand {
    text: EmbedGemma2Model,
    vision: EmbedGemma2VisionModel,
    audio: EmbedGemma2Audio,
}

impl ByHand {
    fn new(f: &Fixture, rt: &std::sync::Arc<tessl::GpuRuntime>, operands: GemmOperands) -> Self {
        Self {
            text: EmbedGemma2Model::load_with(rt, &f.st, PREFIX, f.c.text.clone(), operands).unwrap(),
            vision: EmbedGemma2VisionModel::load(rt, &f.st, f.c.vision.clone()).unwrap(),
            audio: EmbedGemma2Audio::load(rt, &f.st, "", f.c.audio.clone()).unwrap(),
        }
    }

    /// `ids` with the soft tokens of `images` and `clips` over their
    /// placeholders: image placeholders take the images' rows end to end, audio
    /// placeholders the clips'.
    fn embed(&self, ids: &[u32], images: &[ImageInput<'_>], clips: &[AudioInput]) -> Vec<f32> {
        let h = self.text.config().hidden as usize;
        let img: Vec<f32> = if images.is_empty() {
            Vec::new()
        } else {
            self.vision.encode_images(images, false).unwrap().soft.concat()
        };
        let aud: Vec<f32> = if clips.is_empty() {
            Vec::new()
        } else {
            self.audio.encode_audio(clips, false).unwrap().soft_tokens.concat()
        };
        let (mut i, mut a) = (0usize, 0usize);
        let mut soft = Vec::new();
        for &id in ids {
            if id == IMAGE {
                soft.extend_from_slice(&img[i * h..(i + 1) * h]);
                i += 1;
            } else if id == AUDIO {
                soft.extend_from_slice(&aud[a * h..(a + 1) * h]);
                a += 1;
            }
        }
        assert_eq!((i * h, a * h), (img.len(), aud.len()), "test bug: placeholder count");
        self.text
            .encode_multimodal(&[MmSequence { ids, soft: &soft }], None, false)
            .unwrap()
            .embeddings
    }
}

fn text_ids(r: &mut SplitMix, n: usize) -> Vec<u32> {
    (0..n).map(|_| r.range(3, 46) as u32).collect()
}

/// The images and clips the tests use: ragged, with padding, an odd grid
/// width, a single patch, interior mask padding and an all-masked clip.
struct Media {
    images: Vec<Image>,
    clips: Vec<AudioInput>,
}

fn media(c: &Configs) -> Media {
    let pd = c.vision.patch_dim() as usize;
    Media {
        images: vec![
            Image::new(8, 6, 16, pd, 1), // 48 real, P = 64
            Image::new(5, 4, 28, pd, 2), // 20 real, odd width
            Image::new(1, 1, 3, pd, 3),  // 1 real
            Image::new(2, 2, 0, pd, 4),  // 4 real, no padding
        ],
        clips: vec![
            clip(&c.audio, 41, 1, |_| true),
            clip(&c.audio, 58, 2, |f| f < 45),
            clip(&c.audio, 30, 3, |f| !(8..16).contains(&f)),
            clip(&c.audio, 3, 5, |_| false),
        ],
    }
}

// ---------------------------------------------------------------------------

#[test]
fn text_items_are_encode_bit_for_bit_with_one_wait() {
    for operands in MODES {
        let f = fixture("mm_text", operands);
        let mut r = SplitMix::new(9);
        let seqs: Vec<Vec<u32>> = [9usize, 20, 5, 14].iter().map(|&n| text_ids(&mut r, n)).collect();
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let items: Vec<MmItem<'_>> = seqs.iter().map(|s| MmItem::Text(s)).collect();
            let (got, t) = traced(|| m.encode(&items, None).unwrap());
            let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();
            let want = by_hand.text.encode(&refs, None, false).unwrap();
            assert_eq!(bits(&got.embeddings), bits(&want.embeddings), "{operands:?}");
            assert_eq!(got.forwards, want.forwards);
            assert_eq!(t.sync_waits, 1, "{operands:?}: a text call waits once");
            // The Matryoshka prefix goes through.
            let d = 40u32;
            let got = m.encode(&items, Some(d)).unwrap();
            let want = by_hand.text.encode(&refs, Some(d), false).unwrap();
            assert_eq!(bits(&got.embeddings), bits(&want.embeddings), "{operands:?}: prefix");
        });
    }
}

#[test]
fn image_and_audio_items_equal_the_by_hand_composition() {
    for operands in MODES {
        let f = fixture("mm_items", operands);
        let md = media(&f.c);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let cfg = m.text().config().clone();
            for (k, img) in md.images.iter().enumerate() {
                let s = by_hand.vision.check_images(&[img.input()]).unwrap()[0];
                let ids = cfg.image_sequence(s).unwrap();
                let want = by_hand.embed(&ids, &[img.input()], &[]);
                let (got, t) = traced(|| m.encode(&[MmItem::Image(img.input())], None).unwrap());
                assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}: image {k}");
                assert_eq!(
                    t.sync_waits, 2,
                    "{operands:?}: image {k}: towers wait once, the text encoder once"
                );
            }
            for (k, c) in md.clips.iter().enumerate() {
                let s = by_hand.audio.check_clips(std::slice::from_ref(c)).unwrap()[0];
                if s == 0 {
                    // Every frame masked: no soft token, so no sequence.
                    let e = m.encode(&[MmItem::Audio(c)], None).err().expect("accepted");
                    assert!(e.contains("item 0") && e.contains("n_soft is zero"), "{e}");
                    continue;
                }
                let ids = cfg.audio_sequence(s).unwrap();
                let want = by_hand.embed(&ids, &[], std::slice::from_ref(c));
                let (got, t) = traced(|| m.encode(&[MmItem::Audio(c)], None).unwrap());
                assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}: clip {k}");
                assert_eq!(t.sync_waits, 2, "{operands:?}: clip {k}");
            }
        });
    }
}

/// Caller-built ids: a prompt around two images and a clip, the image
/// placeholders broken into runs that do not line up with the images' ends and
/// interleaved with the audio ones, so a facade that concatenated images then
/// audio, or took each image's rows as one run, would come out different.
#[test]
fn media_items_take_each_modality_in_placeholder_order() {
    for operands in MODES {
        let f = fixture("mm_media", operands);
        let md = media(&f.c);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let (a, b) = (&md.images[0], &md.images[2]);
            let imgs = [a.input(), b.input()];
            let counts = by_hand.vision.check_images(&imgs).unwrap();
            let total_img: usize = counts.iter().sum();
            let clips = [md.clips[0].clone()];
            let n_aud = by_hand.audio.check_clips(&clips).unwrap()[0];
            assert!(
                total_img >= 8 && n_aud >= 4,
                "test bug: {total_img} image, {n_aud} audio tokens"
            );
            // [bos, text, <5 image>, <2 audio>, text, <rest of the images>, <rest of audio>, eos]
            let mut ids = vec![2u32, 7, 9];
            ids.extend([IMAGE; 5]);
            ids.extend([AUDIO; 2]);
            ids.push(11);
            ids.extend(std::iter::repeat_n(IMAGE, total_img - 5));
            ids.push(13);
            ids.extend(std::iter::repeat_n(AUDIO, n_aud - 2));
            ids.push(1);
            let want = by_hand.embed(&ids, &imgs, &clips);
            let item = MmItem::Media {
                ids: &ids,
                images: &imgs,
                audio: &clips,
            };
            let (got, t) = traced(|| m.encode(&[item], None).unwrap());
            assert_eq!(bits(&got.embeddings), bits(&want), "{operands:?}");
            assert_eq!(t.sync_waits, 2);

            // The order is observable: the same ids with the images swapped
            // give a different embedding.
            let swapped = [b.input(), a.input()];
            let other = m
                .encode(
                    &[MmItem::Media {
                        ids: &ids,
                        images: &swapped,
                        audio: &clips,
                    }],
                    None,
                )
                .unwrap();
            assert_ne!(
                bits(&other.embeddings),
                bits(&got.embeddings),
                "{operands:?}: order is not observable"
            );

            // An image-only Media item is the Image item (same sequence).
            let s = counts[0];
            let one = cfg_image_ids(&f, s);
            let as_media = m
                .encode(
                    &[MmItem::Media {
                        ids: &one,
                        images: &imgs[..1],
                        audio: &[],
                    }],
                    None,
                )
                .unwrap();
            let as_image = m.encode(&[MmItem::Image(a.input())], None).unwrap();
            assert_eq!(bits(&as_media.embeddings), bits(&as_image.embeddings));
        });
    }
}

fn cfg_image_ids(f: &Fixture, s: usize) -> Vec<u32> {
    f.c.text.image_sequence(s).unwrap()
}

/// A batch of every kind of item equals each item alone.
#[test]
fn a_mixed_batch_equals_each_item_alone() {
    for operands in MODES {
        let f = fixture("mm_mixed", operands);
        let md = media(&f.c);
        let mut r = SplitMix::new(21);
        let t0 = text_ids(&mut r, 8);
        let t1 = text_ids(&mut r, 14);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let by_hand = ByHand::new(&f, rt, operands);
            let s_img = by_hand.vision.check_images(&[md.images[3].input()]).unwrap()[0];
            let mut ids = vec![2u32, 5];
            ids.extend(std::iter::repeat_n(IMAGE, s_img));
            ids.extend([6, 1]);
            let imgs = [md.images[3].input()];
            let items = [
                MmItem::Text(&t0),
                MmItem::Image(md.images[2].input()),
                MmItem::Audio(&md.clips[2]),
                MmItem::Text(&t1),
                MmItem::Media {
                    ids: &ids,
                    images: &imgs,
                    audio: &[],
                },
                MmItem::Image(md.images[3].input()),
                MmItem::Audio(&md.clips[1]),
            ];
            let (batch, t) = traced(|| m.encode(&items, None).unwrap());
            assert_eq!(t.sync_waits, 2, "{operands:?}: a mixed call waits twice");
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
            // Poisoning every unzeroed activation of all three models changes
            // no bit: nothing reads a byte its forward did not write.
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

/// A batch whose text side needs several forwards: the long item runs alone,
/// every item agrees with itself alone within the text encoder's batch bound,
/// and the call still waits twice.
#[test]
fn a_batch_of_several_text_forwards_agrees_within_the_batch_bound() {
    for operands in MODES {
        let f = fixture("mm_forwards", operands);
        let md = media(&f.c);
        let mut r = SplitMix::new(33);
        let long = text_ids(&mut r, 900);
        let short = text_ids(&mut r, 6);
        with_gpu(|rt| {
            let m = f.facade(rt, operands);
            let items = [
                MmItem::Text(&long),
                MmItem::Image(md.images[0].input()),
                MmItem::Audio(&md.clips[0]),
                MmItem::Text(&short),
                MmItem::Image(md.images[2].input()),
            ];
            let (batch, t) = traced(|| m.encode(&items, None).unwrap());
            assert!(batch.forwards >= 2, "test bug: {} text forwards", batch.forwards);
            assert_eq!(t.sync_waits, 2, "{operands:?}: several forwards, still two waits");
            let dim = m.text().config().embedding_dim as usize;
            for (k, item) in items.iter().enumerate() {
                let alone = m.encode(std::slice::from_ref(item), None).unwrap();
                let d = drift(&batch.embeddings[k * dim..(k + 1) * dim], &alone.embeddings);
                assert!(d <= BATCH_ABS, "{operands:?}: item {k}: {d:.3e}");
            }
        });
    }
}

/// The waits and the allocations: one tower call is not two, the facade saves
/// the by-hand composition's third wait, and a text call allocates what
/// `encode` does.
#[test]
fn waits_are_one_for_text_and_two_with_media_against_three_by_hand() {
    let f = fixture("mm_waits", GemmOperands::ExactF32);
    let md = media(&f.c);
    let mut r = SplitMix::new(2);
    let t0 = text_ids(&mut r, 8);
    with_gpu(|rt| {
        let m = f.facade(rt, GemmOperands::ExactF32);
        let by_hand = ByHand::new(&f, rt, GemmOperands::ExactF32);
        let cfg = m.text().config().clone();
        let s_img = by_hand.vision.check_images(&[md.images[0].input()]).unwrap()[0];
        let s_aud = by_hand.audio.check_clips(std::slice::from_ref(&md.clips[0])).unwrap()[0];
        let items = [
            MmItem::Text(&t0),
            MmItem::Image(md.images[0].input()),
            MmItem::Audio(&md.clips[0]),
        ];
        // Warm every pool, then count.
        m.encode(&items, None).unwrap();
        let (_, t) = traced(|| m.encode(&items, None).unwrap());
        assert_eq!(t.sync_waits, 2, "facade, text + image + audio");
        let (_, t) = traced(|| {
            let v = by_hand.vision.encode_images(&[md.images[0].input()], false).unwrap();
            let a = by_hand
                .audio
                .encode_audio(std::slice::from_ref(&md.clips[0]), false)
                .unwrap();
            let ids_i = cfg.image_sequence(s_img).unwrap();
            let ids_a = cfg.audio_sequence(s_aud).unwrap();
            by_hand
                .text
                .encode_multimodal(
                    &[
                        MmSequence {
                            ids: &ids_i,
                            soft: &v.soft[0],
                        },
                        MmSequence {
                            ids: &ids_a,
                            soft: &a.soft_tokens[0],
                        },
                    ],
                    None,
                    false,
                )
                .unwrap()
        });
        assert_eq!(t.sync_waits, 3, "by hand: vision, audio and text each wait");

        // A tower alone waits once, whatever the runtime's async flag says,
        // and `queue_images` + the caller's wait + `collect` is that call.
        for async_on in [false, true] {
            rt.set_async_encode(async_on).unwrap();
            let (_, t) = traced(|| by_hand.vision.encode_images(&[md.images[0].input()], false).unwrap());
            assert_eq!(t.sync_waits, 1, "encode_images, async_encode {async_on}");
            let (_, t) = traced(|| {
                by_hand
                    .audio
                    .encode_audio(std::slice::from_ref(&md.clips[0]), false)
                    .unwrap()
            });
            assert_eq!(t.sync_waits, 1, "encode_audio, async_encode {async_on}");
            let (_, t) = traced(|| m.encode(&items, None).unwrap());
            assert_eq!(t.sync_waits, 2, "facade, async_encode {async_on}");
        }
        rt.set_async_encode(false).unwrap();
        let flag = rt.async_encode_enabled();
        m.encode(&items, None).unwrap();
        assert_eq!(
            rt.async_encode_enabled(),
            flag,
            "the facade leaves the async flag alone"
        );
        assert_eq!(rt.async_scope_depth(), 0, "and no scope open");
    });
}

/// `queue_images` / `queue_audio` hand back a pending call whose result is
/// `encode_*`'s, bit for bit, whether the caller waits or not, and dropping
/// it without collecting waits for the queued work.
#[test]
fn queued_tower_calls_collect_what_encode_returns() {
    let f = fixture("mm_queue", GemmOperands::ExactF32);
    let md = media(&f.c);
    with_gpu(|rt| {
        let by_hand = ByHand::new(&f, rt, GemmOperands::ExactF32);
        let imgs: Vec<ImageInput<'_>> = md.images.iter().map(Image::input).collect();
        let want_v = by_hand.vision.encode_images(&imgs, false).unwrap();
        let want_a = by_hand.audio.encode_audio(&md.clips, false).unwrap();

        // Both queued, one wait for both, then collected: no further wait.
        let (got, t) = traced(|| {
            let v = by_hand.vision.queue_images(&imgs, false).unwrap();
            let a = by_hand.audio.queue_audio(&md.clips, false).unwrap();
            rt.synchronize().unwrap();
            (v.collect().unwrap(), a.collect().unwrap())
        });
        assert_eq!(t.sync_waits, 1, "two towers queued behind one wait");
        for i in 0..imgs.len() {
            assert_eq!(bits(&got.0.soft[i]), bits(&want_v.soft[i]), "image {i}");
        }
        for i in 0..md.clips.len() {
            assert_eq!(bits(&got.1.soft_tokens[i]), bits(&want_a.soft_tokens[i]), "clip {i}");
            assert_eq!(got.1.out_masks[i], want_a.out_masks[i]);
        }

        // `collect` without the caller's wait waits itself.
        let v = by_hand.vision.queue_images(&imgs, false).unwrap();
        let out = v.collect().unwrap();
        assert_eq!(bits(&out.soft[0]), bits(&want_v.soft[0]));

        // A pending call dropped uncollected leaves the runtime idle and usable.
        drop(by_hand.vision.queue_images(&imgs, false).unwrap());
        drop(by_hand.audio.queue_audio(&md.clips, false).unwrap());
        let again = by_hand.vision.encode_images(&imgs, false).unwrap();
        assert_eq!(bits(&again.soft[1]), bits(&want_v.soft[1]));

        // `check_*` do no GPU work and agree with the outputs.
        let (checked, t) = traced(|| {
            (
                by_hand.vision.check_images(&imgs).unwrap(),
                by_hand.audio.check_clips(&md.clips).unwrap(),
            )
        });
        assert_eq!((t.cold_allocs, t.sync_waits), (0, 0));
        assert_eq!(checked.0, want_v.tokens);
        let kept: Vec<usize> = want_a
            .out_masks
            .iter()
            .map(|m| m.iter().filter(|&&k| k).count())
            .collect();
        assert_eq!(checked.1, kept);
    });
}

#[test]
fn refusals_do_no_gpu_work_and_leave_the_facade_usable() {
    let f = fixture("mm_refuse", GemmOperands::ExactF32);
    let md = media(&f.c);
    let mut r = SplitMix::new(4);
    let good_text = text_ids(&mut r, 8);
    with_gpu(|rt| {
        let mut m = f.facade(rt, GemmOperands::ExactF32);
        let cfg = m.text().config().clone();
        let good = [MmItem::Text(&good_text), MmItem::Image(md.images[0].input())];
        let want = m.encode(&good, None).unwrap();

        let s_img = m.vision().check_images(&[md.images[0].input()]).unwrap()[0];
        let imgs = [md.images[0].input()];
        let too_few: Vec<u32> = [vec![2u32], vec![IMAGE; s_img - 1], vec![1]].concat();
        let too_many: Vec<u32> = [vec![2u32], vec![IMAGE; s_img + 1], vec![1]].concat();
        let no_media = [2u32, IMAGE, 1];
        let audio_for_none = [2u32, 5, 1];
        let with_video = [2u32, VIDEO, 1];
        let with_audio_ph = [2u32, AUDIO, 1];
        let text_with_ph = [2u32, 5, IMAGE, 1];
        let oov = [2u32, 5, 48, 1];
        let empty: [u32; 0] = [];
        let long = vec![5u32; m.text().max_tokens() as usize + 1];

        // A bad position after a good image: no tower may have run for the good one.
        let mut bad_positions = Image::new(2, 2, 0, f.c.vision.patch_dim() as usize, 8);
        bad_positions.position_ids[2] = 999;
        let mut nan_features = clip(&f.c.audio, 20, 3, |_| true);
        nan_features.input_features[7] = f32::NAN;
        let ragged = AudioInput {
            input_features: vec![0.0; 5],
            mask: vec![true; 3],
        };
        let silent = clip(&f.c.audio, 3, 5, |_| false);
        let too_many_items: Vec<MmItem<'_>> = (0..MAX_BATCH + 1).map(|_| MmItem::Text(&good_text)).collect();
        let media_ok = MmItem::Image(md.images[0].input());

        let cases: Vec<(&str, Vec<MmItem<'_>>, Option<u32>, &str)> = vec![
            ("empty batch", vec![], None, "empty batch"),
            ("too many items", too_many_items, None, "exceeds the batch limit"),
            (
                "truncate_dim 0",
                vec![MmItem::Text(&good_text)],
                Some(0),
                "truncate_dim",
            ),
            (
                "truncate_dim past the width",
                vec![MmItem::Text(&good_text)],
                Some(cfg.embedding_dim + 1),
                "truncate_dim",
            ),
            ("empty text", vec![MmItem::Text(&empty)], None, "sequence 0 is empty"),
            (
                "token id past the vocab",
                vec![MmItem::Text(&oov)],
                None,
                "token id 48 >= vocab 48",
            ),
            ("text past max_tokens", vec![MmItem::Text(&long)], None, "the limit is"),
            (
                "text holding a placeholder",
                vec![MmItem::Text(&text_with_ph)],
                None,
                "item 0 is text but its ids hold 1 placeholder ids",
            ),
            (
                "too few image placeholders",
                vec![MmItem::Media {
                    ids: &too_few,
                    images: &imgs,
                    audio: &[],
                }],
                None,
                "image placeholder ids",
            ),
            (
                "too many image placeholders",
                vec![MmItem::Media {
                    ids: &too_many,
                    images: &imgs,
                    audio: &[],
                }],
                None,
                "image placeholder ids",
            ),
            (
                "placeholders without media",
                vec![MmItem::Media {
                    ids: &no_media,
                    images: &[],
                    audio: &[],
                }],
                None,
                "has 1 image placeholder ids",
            ),
            (
                "audio placeholders without a clip",
                vec![MmItem::Media {
                    ids: &with_audio_ph,
                    images: &[],
                    audio: &[],
                }],
                None,
                "audio placeholder ids",
            ),
            (
                "media without placeholders",
                vec![MmItem::Media {
                    ids: &audio_for_none,
                    images: &imgs,
                    audio: &[],
                }],
                None,
                "image placeholder ids",
            ),
            (
                "a video placeholder",
                vec![MmItem::Media {
                    ids: &with_video,
                    images: &[],
                    audio: &[],
                }],
                None,
                "no video tower",
            ),
            (
                "a bad image after a good one",
                vec![media_ok, MmItem::Image(bad_positions.input())],
                None,
                "outside the position table",
            ),
            (
                "non-finite audio after a good image",
                vec![media_ok, MmItem::Audio(&nan_features)],
                None,
                "is not finite",
            ),
            (
                "ragged audio features",
                vec![MmItem::Audio(&ragged)],
                None,
                "feature values for 3 frames",
            ),
            (
                "an all-masked clip",
                vec![media_ok, MmItem::Audio(&silent)],
                None,
                "n_soft is zero",
            ),
            (
                "a bad text item after good media",
                vec![media_ok, MmItem::Text(&oov)],
                None,
                "token id 48 >= vocab 48",
            ),
        ];
        for (what, items, trunc, want_msg) in cases {
            let (res, t) = traced(|| m.encode(&items, trunc));
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
        // The runtime's relaxed f32 GEMMs are refused by name before any work.
        rt.set_relaxed_precision(true);
        let (res, t) = traced(|| m.encode(&good, None));
        rt.set_relaxed_precision(false);
        let e = res.err().expect("accepted under relaxed precision");
        assert!(e.contains("exact-f32"), "{e}");
        assert_eq!((t.cold_allocs, t.sync_waits), (0, 0));

        // Still usable, and unchanged.
        let again = m.encode(&good, None).unwrap();
        assert_eq!(bits(&again.embeddings), bits(&want.embeddings));
        // A tightened limit refuses an image sequence that no longer fits.
        m.text_mut().set_max_tokens(s_img as u32 + 3).unwrap();
        let e = m
            .encode(&[MmItem::Image(md.images[0].input())], None)
            .err()
            .expect("accepted");
        assert!(e.contains("the limit is"), "{e}");
    });
}

#[test]
fn loading_refuses_what_it_cannot_serve_before_reading_weights() {
    let f = fixture("mm_load", GemmOperands::ExactF32);
    with_gpu(|rt| {
        // A text-only config (no multimodal ids): refused naming the keys, with
        // no allocation (nothing was loaded).
        let (res, t) =
            traced(|| EmbedGemma2Multimodal::load_from(rt, &f.st, &config_json_without_ids(), GemmOperands::ExactF32));
        let e = res.err().expect("accepted a config without the multimodal ids");
        for key in ["image_token_id", "audio_token_id", "boi_token_id", "eoa_token_index"] {
            assert!(e.contains(key), "{e}");
        }
        assert_eq!((t.cold_allocs, t.sync_waits), (0, 0));

        // A snapshot missing a tower's tensors is refused by the tower.
        let c = &f.c;
        let mut tensors =
            common::embedgemma2::text_tensors(&common::embedgemma2::weights_for(&c.text, 17, GemmOperands::ExactF32));
        tensors.extend(mm::vision_tensors(&c.vision, 18));
        let no_audio = open("mm_no_audio", common::embedgemma2::pack_safetensors(&tensors));
        let e = EmbedGemma2Multimodal::load_from(rt, &no_audio, &config_json(), GemmOperands::ExactF32)
            .err()
            .expect("accepted a checkpoint without the audio tower");
        assert!(e.contains("audio_tower"), "{e}");
    });
}

#[test]
fn from_parts_refuses_mismatched_widths_and_runtimes() {
    let f = fixture("mm_parts", GemmOperands::ExactF32);
    // A vision tower whose soft tokens are 48 wide against a 64-wide text encoder.
    let v48_json = format!(
        r#"{{"text_config":{{"hidden_size":48}},"vision_config":{}}}"#,
        mm::VISION_JSON
    );
    let v48_cfg = tessl::embedgemma2_vision::VisionConfig::from_config_json(&v48_json).unwrap();
    assert_eq!(v48_cfg.text_hidden, 48);
    let v48 = open(
        "mm_v48",
        common::embedgemma2::pack_safetensors(&mm::vision_tensors(&v48_cfg, 5)),
    );
    let a24_json = format!(
        r#"{{"text_config":{{"hidden_size":24}},"audio_config":{}}}"#,
        mm::AUDIO_JSON
    );
    let a24_cfg = tessl::embedgemma2_audio::AudioConfig::from_config_json(&a24_json).unwrap();
    let a24 = open(
        "mm_a24",
        common::embedgemma2::pack_safetensors(&mm::audio_tensors(&a24_cfg, 6)),
    );
    with_gpu(|rt| {
        let text = EmbedGemma2Model::load(rt, &f.st, PREFIX, f.c.text.clone()).unwrap();
        let vision = EmbedGemma2VisionModel::load(rt, &f.st, f.c.vision.clone()).unwrap();
        let audio = EmbedGemma2Audio::load(rt, &f.st, "", f.c.audio.clone()).unwrap();
        let vision48 = EmbedGemma2VisionModel::load(rt, &v48, v48_cfg.clone()).unwrap();
        let audio24 = EmbedGemma2Audio::load(rt, &a24, "", a24_cfg.clone()).unwrap();
        let e = EmbedGemma2Multimodal::from_parts(
            EmbedGemma2Model::load(rt, &f.st, PREFIX, f.c.text.clone()).unwrap(),
            vision48,
            EmbedGemma2Audio::load(rt, &f.st, "", f.c.audio.clone()).unwrap(),
        )
        .err()
        .expect("accepted a 48-wide vision tower");
        assert!(e.contains("vision tower's soft tokens are 48 wide"), "{e}");
        let e = EmbedGemma2Multimodal::from_parts(
            EmbedGemma2Model::load(rt, &f.st, PREFIX, f.c.text.clone()).unwrap(),
            EmbedGemma2VisionModel::load(rt, &f.st, f.c.vision.clone()).unwrap(),
            audio24,
        )
        .err()
        .expect("accepted a 24-wide audio tower");
        assert!(e.contains("audio tower's soft tokens are 24 wide"), "{e}");
        // The parts that agree compose.
        assert!(EmbedGemma2Multimodal::from_parts(text, vision, audio).is_ok());
    });
    with_two_gpus(|a, b| {
        let text = EmbedGemma2Model::load(a, &f.st, PREFIX, f.c.text.clone()).unwrap();
        let vision = EmbedGemma2VisionModel::load(b, &f.st, f.c.vision.clone()).unwrap();
        let audio = EmbedGemma2Audio::load(a, &f.st, "", f.c.audio.clone()).unwrap();
        let e = EmbedGemma2Multimodal::from_parts(text, vision, audio)
            .err()
            .expect("accepted parts of two runtimes");
        assert!(e.contains("different runtimes"), "{e}");
    });
}
