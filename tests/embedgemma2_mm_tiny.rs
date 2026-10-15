//! `EmbedGemma2Model::encode_multimodal` on the tiny random checkpoint of
//! `tests/embedgemma2_tiny.rs`, against the same f64 host forward with an
//! `inputs_embeds` override (`tests/common/embedgemma2.rs`): the placeholder
//! ids are gathered as `pad_token_id` through the scaled embedding, then the
//! soft rows are written over them, unscaled, in order, as
//! `EmbeddingGemma2Model.forward`'s `masked_scatter` does, and the per-layer
//! inputs and the residual stream both start from the result. Needs a GPU, no
//! checkpoint; the real one is `tests/embedgemma2_mm_model.rs`.
//!
//! The config adds the multimodal ids to the tiny one. The placeholders
//! (image 60, audio 61, video 62) are past the tiny vocabulary of 48 on
//! purpose: a placeholder is never looked up, so it must not be bounded by it.
//!
//! Bounds, the ones `tests/embedgemma2_tiny.rs` fixed for each mode
//! (`common::embedgemma2::bounds`), set before this file's first run:
//! each layer's residual and the final norm within `layer_rel` of their
//! largest magnitude, embeddings within `emb_abs` max abs and `emb_cos`
//! cosine of the f64 forward; each sequence alone vs inside a one-forward batch
//! bit for bit (the GEMMs sum K in the same order whatever M is), and vs inside a
//! batch of several forwards, in buffers sized for another, within
//! `1e-5`; the multimodal entry with no soft rows bit for bit `encode`; a
//! queued encode bit for bit the serialized one with every unzeroed
//! activation poisoned to NaN, after one host wait. (The first run, with
//! the one-forward comparison at `1e-5`, measured 0 in both modes, which is
//! when it was made exact.)

mod common;

use common::embedgemma2::{
    bounds, cosine, host_forward_embeds, l2_normalize, max_abs, mm_embeds, open, safetensors_bytes, weights_for,
    Bounds, Weights, CONFIG, PREFIX,
};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model, MmSequence};
use tessl::gemm::GemmOperands;
use tessl::infer_trace;

const BATCH_ABS: f64 = 1e-5;
const MODES: [GemmOperands; 2] = [GemmOperands::ExactF32, GemmOperands::Bf16];

const PAD: u32 = 0;
const IMAGE: u32 = 60;
const AUDIO: u32 = 61;
const VIDEO: u32 = 62;
const PLACEHOLDERS: [u32; 3] = [IMAGE, VIDEO, AUDIO];

/// The tiny config with the multimodal ids (the real config's layout: the
/// markers and placeholders at the top level, `pad`, `bos` and `eos` in
/// `text_config`).
fn mm_config_json() -> String {
    let top = r#"{"image_token_id":60,"audio_token_id":61,"video_token_id":62,"boi_token_id":40,"eoi_token_id":41,"boa_token_id":42,"eoa_token_index":43,"#;
    let text = r#""pad_token_id":0,"bos_token_id":2,"eos_token_id":1,"hidden_size":64,"#;
    let json = CONFIG.replacen("{", top, 1).replacen(r#""hidden_size":64,"#, text, 1);
    assert!(
        json.contains("eoa_token_index") && json.contains("bos_token_id"),
        "test bug"
    );
    json
}

fn mm_config() -> EmbedGemma2Config {
    EmbedGemma2Config::from_config_json(&mm_config_json()).unwrap()
}

/// A text sequence of `n` ids from the vocabulary.
fn text(r: &mut SplitMix, n: usize, vocab: u32) -> Vec<u32> {
    (0..n).map(|_| r.range(3, vocab as usize - 1) as u32).collect()
}

/// `n` soft rows of the tiny hidden size, each element in [-1, 1).
fn soft(r: &mut SplitMix, n: usize, hidden: usize) -> Vec<f32> {
    (0..n * hidden).map(|_| r.unit()).collect()
}

struct Case {
    ids: Vec<u32>,
    soft: Vec<f32>,
}

impl Case {
    fn seq(&self) -> MmSequence<'_> {
        MmSequence {
            ids: &self.ids,
            soft: &self.soft,
        }
    }

    /// The f64 forward of this case: (per-layer trace, embedding).
    fn host(&self, cfg: &EmbedGemma2Config, w: &Weights) -> (Vec<Vec<f64>>, Vec<f64>) {
        let soft64: Vec<f64> = self.soft.iter().map(|&v| f64::from(v)).collect();
        host_forward_embeds(cfg, w, mm_embeds(cfg, w, &self.ids, &PLACEHOLDERS, PAD, &soft64))
    }
}

/// Sequences that put the soft rows at different places: none, a processor's
/// image and audio layouts, placeholders of all three kinds scattered in runs
/// of different lengths (one at the first position, one at the last, over the
/// window of 3), and a pair of equal-length sequences whose placeholders touch
/// across the boundary (one ends with a placeholder, the next starts with
/// one).
fn cases(cfg: &EmbedGemma2Config) -> Vec<Case> {
    let (h, vocab) = (cfg.hidden as usize, cfg.vocab);
    let mut r = SplitMix::new(404);
    let mut ids_with = |n: usize, at: &[(usize, u32)]| {
        let mut ids = text(&mut r, n, vocab);
        for &(i, p) in at {
            ids[i] = p;
        }
        ids
    };
    let scattered = ids_with(
        14,
        &[
            (0, IMAGE),
            (1, IMAGE),
            (5, AUDIO),
            (6, AUDIO),
            (7, AUDIO),
            (8, VIDEO),
            (13, IMAGE),
        ],
    );
    let end_a = ids_with(14, &[(0, VIDEO), (13, IMAGE)]);
    let start_b = ids_with(14, &[(0, AUDIO), (1, AUDIO)]);
    let plain = text(&mut r, 9, vocab);
    let mut out = vec![Case {
        ids: plain,
        soft: Vec::new(),
    }];
    for (ids, n) in [
        (cfg.image_sequence(6).unwrap(), 6),
        (cfg.audio_sequence(5).unwrap(), 5),
        (scattered, 7),
        (end_a, 2),
        (start_b, 2),
    ] {
        out.push(Case {
            soft: soft(&mut r, n, h),
            ids,
        });
    }
    for c in &out {
        let n = c.ids.iter().filter(|id| PLACEHOLDERS.contains(id)).count();
        assert_eq!(n * h, c.soft.len(), "test bug: placeholders and soft rows differ");
    }
    out
}

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
    f64::from(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max))
}

#[test]
fn mm_tiny_matches_the_f64_forward() {
    matches_the_f64_forward(GemmOperands::ExactF32);
}

#[test]
fn mm_tiny_bf16_matches_the_f64_forward() {
    matches_the_f64_forward(GemmOperands::Bf16);
}

fn matches_the_f64_forward(operands: GemmOperands) {
    let Bounds {
        layer_rel,
        emb_abs,
        emb_cos,
    } = bounds(operands);
    let cfg = mm_config();
    let w = weights_for(&cfg, 2026, operands);
    let st = open("mm", safetensors_bytes(&w, &[("vision_tower.unused.weight", &[3])]));
    let cases = cases(&cfg);
    let (h, dim) = (cfg.hidden as usize, cfg.embedding_dim as usize);

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();

        // 1. The soft rows matter: the f64 forward with them differs from the
        //    one that leaves the placeholders as pad rows, by far more than a
        //    bound, so a scatter that wrote nothing, or elsewhere, fails below.
        let image = &cases[1];
        let zeros = vec![0.0; image.soft.len()];
        let (_, ignored) = host_forward_embeds(&cfg, &w, mm_embeds(&cfg, &w, &image.ids, &PLACEHOLDERS, PAD, &zeros));
        let (_, want) = image.host(&cfg, &w);
        let cos = ignored.iter().zip(&want).map(|(a, b)| a * b).sum::<f64>();
        assert!(
            cos < 0.99,
            "test bug: the soft rows move the embedding by only cosine {cos:.6}"
        );

        // 2. Layer by layer, the image (the soft rows reach every layer
        //    through the residual stream and the per-layer inputs).
        let (want_trace, _) = image.host(&cfg, &w);
        let out = model.encode_multimodal(&[image.seq()], None, true).unwrap();
        assert_eq!(out.trace.len(), cfg.layers.len() + 1);
        for (i, (got, want)) in out.trace.iter().zip(&want_trace).enumerate() {
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let err = max_abs(&got[..image.ids.len() * h], want);
            eprintln!("{operands:?} trace {i}: max abs {err:.3e} of scale {scale:.3e}");
            assert!(
                err <= layer_rel * scale,
                "trace {i}: max abs {err:.3e} > {layer_rel:.0e} * {scale:.3e}"
            );
        }

        // 3. Every case alone, against the f64 forward.
        let mut alone = Vec::new();
        for (n, c) in cases.iter().enumerate() {
            let (_, want) = c.host(&cfg, &w);
            let e = model.encode_multimodal(&[c.seq()], None, false).unwrap().embeddings;
            let (err, cos) = (max_abs(&e, &want), cosine(&e, &want));
            eprintln!(
                "{operands:?} case {n} ({} tokens, {} soft) alone: max abs {err:.3e}, cosine {cos:.9}",
                c.ids.len(),
                c.soft.len() / h
            );
            assert!(
                err <= emb_abs && cos >= emb_cos,
                "case {n}: max abs {err:.3e}, cosine {cos:.9}"
            );
            alone.push((e, want));
        }

        // 4. All of them in one call, text beside soft: each against the f64
        //    forward and against itself alone.
        let seqs: Vec<MmSequence<'_>> = cases.iter().map(Case::seq).collect();
        let out = model.encode_multimodal(&seqs, None, false).unwrap();
        assert_eq!(out.forwards, 1, "test bug: the batch must share one forward");
        for (n, (e, want)) in alone.iter().enumerate() {
            let got = &out.embeddings[n * dim..(n + 1) * dim];
            let (err, cos, d) = (max_abs(got, want), cosine(got, want), drift(got, e));
            eprintln!("{operands:?} case {n} batched: max abs {err:.3e}, cosine {cos:.9}, vs alone {d:.3e}");
            assert!(
                err <= emb_abs && cos >= emb_cos,
                "case {n} batched: max abs {err:.3e}, cosine {cos:.9}"
            );
            assert_eq!(bits(got), bits(e), "case {n}: batched vs alone differ by {d:.3e}");
        }

        // 5. A Matryoshka prefix of the soft sequences.
        for d in [64usize, 17] {
            let out = model.encode_multimodal(&seqs, Some(d as u32), false).unwrap();
            for (n, (_, want)) in alone.iter().enumerate() {
                let want = l2_normalize(&want[..d]);
                let got = &out.embeddings[n * d..(n + 1) * d];
                assert!(
                    max_abs(got, &want) <= emb_abs && cosine(got, &want) >= emb_cos,
                    "prefix {d}, case {n}"
                );
            }
        }

        // 6. Text is the special case: with no soft rows the entry is `encode`,
        //    bit for bit, and a text id equal to a placeholder id stays an
        //    ordinary token under `encode` (the tiny vocab has 48 ids, so use
        //    the largest).
        let texts: Vec<&[u32]> = vec![&cases[0].ids, &cases[0].ids[..4]];
        let mm: Vec<MmSequence<'_>> = texts.iter().map(|&ids| MmSequence { ids, soft: &[] }).collect();
        let (a, b) = (
            model.encode(&texts, None, false).unwrap(),
            model.encode_multimodal(&mm, None, false).unwrap(),
        );
        assert_eq!(bits(&a.embeddings), bits(&b.embeddings), "text through either entry");
        assert_eq!(a.forwards, b.forwards);
        assert!(
            model.encode(&[&[IMAGE]], None, false).is_err(),
            "encode bounds every id by the vocab"
        );
    });
}

/// A call that waits for the GPU once however many forwards it queues, whose
/// bits are the serialized dispatch's and do not depend on what an unzeroed
/// activation held: soft rows in a long sequence alone, mid-length ones and
/// short ones in later forwards of buffers sized for the first. Each sequence
/// matches itself alone.
#[test]
fn mm_multi_forward_waits_once_and_matches_serialized() {
    for operands in MODES {
        multi_forward_at(operands);
    }
}

fn multi_forward_at(operands: GemmOperands) {
    let cfg = mm_config();
    let w = weights_for(&cfg, 11, operands);
    let st = open("mm_async", safetensors_bytes(&w, &[]));
    let h = cfg.hidden as usize;
    let vocab = cfg.vocab;
    let mut r = SplitMix::new(5);
    // Placeholders in a run of `run` at `at`, and an audio one at the end.
    let seq = |r: &mut SplitMix, n: usize, at: usize, run: usize| -> Case {
        let mut ids = text(r, n, vocab);
        ids[at..at + run].fill(IMAGE);
        ids[n - 1] = AUDIO;
        let rows = run + 1;
        Case {
            ids,
            soft: soft(r, rows, h),
        }
    };
    let mut cases = vec![seq(&mut r, 4000, 100, 256)];
    cases.extend((0..3).map(|_| seq(&mut r, 300, 20, 64)));
    cases.extend((0..7).map(|_| seq(&mut r, 30, 3, 5)));
    cases.push(Case {
        ids: text(&mut r, 30, vocab),
        soft: Vec::new(),
    });
    let batch: Vec<MmSequence<'_>> = cases.iter().map(Case::seq).collect();
    let dim = cfg.embedding_dim as usize;
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
        rt.set_serialized_dispatch(true).unwrap();
        let (sync, t_sync) = traced(|| model.encode_multimodal(&batch, None, false).unwrap());
        rt.set_serialized_dispatch(false).unwrap();
        assert!(
            sync.forwards >= 3,
            "test bug: {} forwards, need at least 3",
            sync.forwards
        );
        assert!(
            t_sync.sync_waits >= u64::from(sync.forwards),
            "test bug: serialized run did not wait per forward"
        );

        rt.set_poison_unzeroed(true);
        let (queued, t) = traced(|| model.encode_multimodal(&batch, None, false));
        rt.set_poison_unzeroed(false);
        let queued = queued.unwrap();
        assert_eq!(queued.forwards, sync.forwards);
        assert_eq!(
            t.sync_waits, 1,
            "{operands:?}: {} forwards made {} host waits",
            queued.forwards, t.sync_waits
        );
        assert_eq!(
            bits(&queued.embeddings),
            bits(&sync.embeddings),
            "{operands:?}: queueing the forwards, or an unzeroed activation, changed an embedding"
        );
        assert!(queued.embeddings.iter().all(|v| v.is_finite()));

        for (n, c) in cases.iter().enumerate() {
            let alone = model.encode_multimodal(&[c.seq()], None, false).unwrap().embeddings;
            let d = drift(&queued.embeddings[n * dim..(n + 1) * dim], &alone);
            assert!(
                d <= BATCH_ABS,
                "{operands:?}: sequence {n} ({} tokens) vs alone: {d:.3e}",
                c.ids.len()
            );
        }
    });
}

/// Poisoned activations change no bit of a one-forward mixed batch either, and
/// the entry's host waits and allocations are `encode`'s: a text-only forward
/// allocates nothing for soft rows, a soft one exactly one buffer.
#[test]
fn mm_poisoned_activations_and_allocation_counts() {
    for operands in MODES {
        let cfg = mm_config();
        let w = weights_for(&cfg, 7, operands);
        let st = open("mm_poison", safetensors_bytes(&w, &[]));
        let cases = cases(&cfg);
        let batch: Vec<MmSequence<'_>> = cases.iter().map(Case::seq).collect();
        let texts: Vec<MmSequence<'_>> = batch.iter().filter(|s| s.soft.is_empty()).copied().collect();
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            let (plain, t_plain) = traced(|| model.encode_multimodal(&batch, None, false).unwrap());
            assert_eq!(t_plain.sync_waits, 1, "{operands:?}");
            rt.set_poison_unzeroed(true);
            let poisoned = model.encode_multimodal(&batch, None, false);
            rt.set_poison_unzeroed(false);
            assert_eq!(
                bits(&poisoned.unwrap().embeddings),
                bits(&plain.embeddings),
                "{operands:?}: an unzeroed activation was read before it was written"
            );
            let text_ids: Vec<&[u32]> = texts.iter().map(|s| s.ids).collect();
            let (a, ta) = traced(|| model.encode(&text_ids, None, false).unwrap());
            let (b, tb) = traced(|| model.encode_multimodal(&texts, None, false).unwrap());
            assert_eq!(
                ta.cold_allocs, tb.cold_allocs,
                "{operands:?}: a text-only forward allocated for soft rows"
            );
            assert_eq!(bits(&a.embeddings), bits(&b.embeddings));
            let (_, t_one) = traced(|| model.encode_multimodal(&batch[1..2], None, false).unwrap());
            let (_, t_text) = traced(|| model.encode_multimodal(&batch[..1], None, false).unwrap());
            assert_eq!(
                t_one.cold_allocs,
                t_text.cold_allocs + 1,
                "{operands:?}: soft rows cost exactly one more buffer"
            );
        });
    }
}

/// Every refusal comes before any GPU work (no allocation, no wait), and the
/// model is usable afterwards.
#[test]
fn mm_refusals_do_no_gpu_work() {
    let cfg = mm_config();
    let w = weights_for(&cfg, 3, GemmOperands::ExactF32);
    let st = open("mm_refuse", safetensors_bytes(&w, &[]));
    let h = cfg.hidden as usize;
    let good = cfg.image_sequence(3).unwrap();
    let rows = |n: usize| vec![0.25f32; n * h];
    let (r1, r2, r3, r4) = (rows(1), rows(2), rows(3), rows(4));
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), GemmOperands::ExactF32).unwrap();
        let ok = model.encode_multimodal(&[MmSequence { ids: &good, soft: &r3 }], None, false);
        assert!(ok.is_ok(), "{:?}", ok.as_ref().err());

        let mut nan = r3.clone();
        nan[h + 5] = f32::NAN;
        let mut inf = r3.clone();
        inf[0] = f32::INFINITY;
        let mut neg_inf = r3.clone();
        neg_inf[3 * h - 1] = f32::NEG_INFINITY;
        let ragged = vec![0.25f32; 3 * h - 1];
        let bad_id = [2, IMAGE, 48, 1];
        let oov_other = [2, IMAGE, 5000, 1];
        let long = vec![5u32; model.max_tokens() as usize + 1];
        let a = MmSequence { ids: &good, soft: &r3 };
        let cases: Vec<(&str, Vec<MmSequence<'_>>, &str)> = vec![
            (
                "too few soft rows",
                vec![MmSequence { ids: &good, soft: &r2 }],
                "3 placeholder ids but 2 soft rows",
            ),
            (
                "too many soft rows",
                vec![MmSequence { ids: &good, soft: &r4 }],
                "3 placeholder ids but 4 soft rows",
            ),
            (
                "no soft rows for placeholders",
                vec![MmSequence { ids: &good, soft: &[] }],
                "3 placeholder ids but 0 soft rows",
            ),
            (
                "soft rows without a placeholder",
                vec![MmSequence {
                    ids: &[2, 7, 1],
                    soft: &r1,
                }],
                "0 placeholder ids but 1 soft rows",
            ),
            (
                "ragged soft",
                vec![MmSequence {
                    ids: &good,
                    soft: &ragged,
                }],
                "not a multiple of hidden",
            ),
            (
                "NaN soft",
                vec![MmSequence { ids: &good, soft: &nan }],
                "(row 1, column 5) is not finite",
            ),
            (
                "+inf soft",
                vec![MmSequence { ids: &good, soft: &inf }],
                "(row 0, column 0) is not finite",
            ),
            (
                "-inf soft",
                vec![MmSequence {
                    ids: &good,
                    soft: &neg_inf,
                }],
                "(row 2, column 63) is not finite",
            ),
            (
                "the second sequence is wrong",
                vec![a, MmSequence { ids: &good, soft: &r1 }],
                "sequence 1",
            ),
            (
                "an id past the vocab beside a placeholder",
                vec![MmSequence {
                    ids: &bad_id,
                    soft: &r1,
                }],
                "token id 48 >= vocab 48",
            ),
            (
                "another id past the vocab",
                vec![MmSequence {
                    ids: &oov_other,
                    soft: &r1,
                }],
                "token id 5000 >= vocab 48",
            ),
            (
                "empty sequence",
                vec![MmSequence { ids: &[], soft: &[] }],
                "sequence 0 is empty",
            ),
            (
                "over the token limit",
                vec![MmSequence { ids: &long, soft: &[] }],
                "the limit is",
            ),
            ("empty batch", vec![], "empty batch"),
        ];
        for (what, batch, want) in cases {
            let (r, t) = traced(|| model.encode_multimodal(&batch, None, false));
            let e = r.err().unwrap_or_else(|| panic!("{what}: accepted"));
            assert!(e.contains(want), "{what}: {e:?} does not mention {want:?}");
            assert!(e.starts_with("EmbedGemma2Model::encode_multimodal"), "{what}: {e}");
            assert_eq!(
                (t.cold_allocs, t.sync_waits),
                (0, 0),
                "{what}: did GPU work before refusing"
            );
        }
        // The batch-level refusals of `encode`, through this entry.
        let two = [a, a];
        for (what, r) in [
            ("trace of two", model.encode_multimodal(&two, None, true)),
            ("truncate_dim 0", model.encode_multimodal(&[a], Some(0), false)),
            (
                "truncate_dim past the embedding",
                model.encode_multimodal(&[a], Some(cfg.embedding_dim + 1), false),
            ),
        ] {
            assert!(r.is_err(), "{what}: accepted");
        }
        let too_many: Vec<MmSequence<'_>> = vec![
            MmSequence {
                ids: &[2, 3],
                soft: &[]
            };
            tessl::embedgemma2::MAX_BATCH + 1
        ];
        assert!(
            model.encode_multimodal(&too_many, None, false).is_err(),
            "over MAX_BATCH"
        );

        // An OOV placeholder with its soft row is fine; so is a video one.
        let video = [2, VIDEO, 1];
        assert!(model
            .encode_multimodal(&[MmSequence { ids: &video, soft: &r1 }], None, false)
            .is_ok());
        // Still usable, and the same answer as before the refusals.
        let again = model.encode_multimodal(&[a], None, false).unwrap();
        let first = ok.unwrap();
        assert_eq!(bits(&again.embeddings), bits(&first.embeddings));
    });
}

/// The entry needs the config's ids, named by their `config.json` keys; a
/// text-only config still loads and runs `encode`.
#[test]
fn mm_entry_needs_the_multimodal_ids() {
    // Text-only: the tiny CONFIG carries none of them.
    let text_only = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    assert_eq!(text_only.special_tokens, tessl::embedgemma2::SpecialTokens::default());
    let w = weights_for(&text_only, 3, GemmOperands::ExactF32);
    let st = open("mm_text_only", safetensors_bytes(&w, &[]));
    let one = [2u32, 7, 9];
    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, text_only.clone()).unwrap();
        assert!(
            model.encode(&[&one], None, false).is_ok(),
            "a text-only config still encodes"
        );
        let (r, t) = traced(|| model.encode_multimodal(&[MmSequence { ids: &one, soft: &[] }], None, false));
        let e = r.err().expect("accepted");
        for key in [
            "text_config.pad_token_id",
            "image_token_id",
            "video_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
            "boa_token_id",
            "eoa_token_index",
        ] {
            assert!(e.contains(key), "{e:?} does not name {key}");
        }
        assert_eq!((t.cold_allocs, t.sync_waits), (0, 0));

        // One missing id is named alone (the checkpoint's spelling).
        let json = mm_config_json().replacen(r#""eoa_token_index":43,"#, "", 1);
        let partial = EmbedGemma2Config::from_config_json(&json).unwrap();
        assert_eq!(partial.special_tokens.eoa, None);
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, partial).unwrap();
        let e = model
            .encode_multimodal(&[MmSequence { ids: &one, soft: &[] }], None, false)
            .err()
            .expect("accepted");
        assert!(e.contains("eoa_token_index") && !e.contains("image_token_id"), "{e}");

        // A pad id outside the vocabulary cannot be looked up.
        let json = mm_config_json().replacen(r#""pad_token_id":0"#, r#""pad_token_id":48"#, 1);
        let model =
            EmbedGemma2Model::load(rt, &st, PREFIX, EmbedGemma2Config::from_config_json(&json).unwrap()).unwrap();
        let e = model
            .encode_multimodal(&[MmSequence { ids: &one, soft: &[] }], None, false)
            .err()
            .expect("accepted");
        assert!(e.contains("pad_token_id 48 >= vocab 48"), "{e}");
    });
}

/// The config keeps the ids as the file spells them, and the builders lay out
/// the processor's ids. Device-free.
#[test]
fn config_keeps_the_ids_and_the_builders_lay_out_the_processor_ids() {
    let cfg = mm_config();
    let t = cfg.special_tokens;
    assert_eq!(
        (t.pad, t.bos, t.eos, t.image, t.video, t.audio),
        (Some(0), Some(2), Some(1), Some(60), Some(62), Some(61))
    );
    assert_eq!((t.boi, t.eoi, t.boa, t.eoa), (Some(40), Some(41), Some(42), Some(43)));
    assert_eq!(cfg.image_sequence(3).unwrap(), [2, 40, 60, 60, 60, 41, 1]);
    assert_eq!(cfg.audio_sequence(2).unwrap(), [2, 42, 61, 61, 43, 1]);
    assert_eq!(cfg.image_sequence(1).unwrap(), [2, 40, 60, 41, 1]);
    assert_eq!(cfg.image_sequence(280).unwrap().len(), 284);
    for e in [cfg.image_sequence(0), cfg.audio_sequence(0)] {
        assert!(e.unwrap_err().contains("n_soft is zero"));
    }
    assert!(
        cfg.image_sequence(usize::MAX).is_err(),
        "an overflowing run is refused, not wrapped"
    );

    // Text-only: none, and the builders name what they lack.
    let text_only = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let e = text_only.image_sequence(3).unwrap_err();
    assert!(
        e.contains("boi_token_id") && e.contains("image_token_id") && e.contains("bos_token_id"),
        "{e}"
    );
    let e = text_only.audio_sequence(3).unwrap_err();
    assert!(
        e.contains("boa_token_id") && e.contains("eoa_token_index") && e.contains("audio_token_id"),
        "{e}"
    );
    // The audio builder does not need the image markers.
    let json = mm_config_json().replacen(r#""boi_token_id":40,"#, "", 1);
    let no_boi = EmbedGemma2Config::from_config_json(&json).unwrap();
    assert!(no_boi.audio_sequence(2).is_ok());
    assert!(no_boi.image_sequence(2).unwrap_err().contains("boi_token_id"));

    // A null is absent; a present id must be a u32.
    let null = mm_config_json().replacen(r#""image_token_id":60"#, r#""image_token_id":null"#, 1);
    assert_eq!(
        EmbedGemma2Config::from_config_json(&null).unwrap().special_tokens.image,
        None
    );
    for (from, to) in [
        (r#""image_token_id":60"#, r#""image_token_id":"60""#),
        (r#""image_token_id":60"#, r#""image_token_id":-1"#),
        (r#""image_token_id":60"#, r#""image_token_id":6.5"#),
        (r#""image_token_id":60"#, r#""image_token_id":4294967296"#),
        (r#""pad_token_id":0"#, r#""pad_token_id":"0""#),
    ] {
        let e = EmbedGemma2Config::from_config_json(&mm_config_json().replacen(from, to, 1)).unwrap_err();
        assert!(e.contains("token_id"), "{to}: {e}");
    }
}
