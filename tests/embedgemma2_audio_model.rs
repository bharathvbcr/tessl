//! The whole EmbeddingGemma 2 audio tower and embedder against the
//! transformers fp32 forward on the real checkpoint.
//!
//! `tests/embedgemma2_audio_tiny.rs` checks the composition against an f64
//! forward on random weights; this checks the real weights, the real
//! geometry (12 layers of 8 heads of 128, chunk 12, context left 13) and
//! transformers itself. It needs the checkpoint and the references
//! `tools/embedgemma2_ref/make_mm_reference.py` writes (sdpa, its default), so
//! it is opt-in:
//!
//! ```text
//! ~/.venvs/ml-2026-10/bin/python tools/embedgemma2_ref/make_mm_reference.py        # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//! EMBEDGEMMA2_MM_REF_DIR=target/embedgemma2_mm_ref \
//!   cargo test --release --test embedgemma2_audio_model -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_MM_REF_DIR` names sdpa references (the bool mask the model is
//! specified to apply). Under eager, transformers 5.19 builds a float mask and
//! then calls `logical_not()` on it, which inverts the local attention (see
//! `src/embedgemma2_audio.rs`). The second test shows an eager reference
//! disagreeing: `EMBEDGEMMA2_MM_REF_EAGER_DIR`, also opt-in, names a directory
//! written with `EMBEDGEMMA2_MM_ATTN=eager`. A missing variable or file
//! fails the test rather than skipping it.
//!
//! Bounds, fixed before the first run (the forward is transformers' fp32
//! forward up to operation order, with exact-f32 GEMMs; the observed errors are
//! recorded in `docs/embedgemma2.md`):
//!
//! * the subsample output, the residual stream after every layer and the
//!   tower output within `1e-4` of their largest magnitude, over the kept rows
//!   of both clips (99 frames: 25 rows; 330 frames of which 329 are valid: 83);
//! * the soft tokens within `1e-4` of their largest magnitude too;
//! * each clip alone and inside one batch bit for bit equal.
//!
//! `GemmOperands::Bf16` has its own two tests below (the tower alone against the
//! fp32 goldens by cosine, batch-independent; the clip embeddings end to end
//! against fp32 and transformers' bf16 forward), with the floors measured on
//! this checkpoint and recorded beside the constants.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::{AudioConfig, AudioInput, AudioOutput, EmbedGemma2Audio};
use tessl::embedgemma2_mm::{EmbedGemma2Multimodal, MmItem};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const LAYER_REL: f64 = 1e-4;
// The bf16 bounds. Measured on this checkpoint (Apple M5 Pro, sdpa goldens): the
// tower's soft tokens against fp32 at cosine 0.999951 and 0.999922 (clips 0 and
// 1); the embeddings end to end at 0.99999343 and 0.99998571 against fp32 and
// 0.99980337 and 0.99968055 against transformers' bf16 forward (which is itself
// at 0.99982002 and 0.99968966 from fp32). The floors sit below the worst clip:
// the soft tokens about 4x further from 1 than the worst clip, the embeddings the floors
// `tests/embedgemma2_mm_model.rs` and the vision test hold a bf16 forward to.
const SOFT_COS_FLOOR: f64 = 0.9997;
// Two 90 s clips (2250 tokens each), bf16 against the exact forward, token by
// token: whole clips at cosine 0.999904 and 0.999911, the worst single token
// at 0.994434 and 0.996582. The floor leaves the worst token's distance from 1
// a factor of 1.8 of room.
const LONG_TOKEN_COS_FLOOR: f64 = 0.99;
const EMB_COS_FLOOR: f64 = 0.9999;
const HF_COS_FLOOR: f64 = 0.9996;
const BATCH_ABS: f64 = 1e-5;
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

fn npy_f32(dir: &std::path::Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let p = dir.join(name);
    let a = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    (a.shape.clone(), a.f32_slice().expect("f32 reference").to_vec())
}

/// A `|b1` (bool) npy, which `read_npy` does not read: version 1 header, then
/// one byte per element.
fn npy_bool(dir: &std::path::Path, name: &str) -> Vec<bool> {
    let p = dir.join(name);
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{name}");
    assert_eq!(b[6], 1, "{name}: only npy v1");
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = std::str::from_utf8(&b[10..10 + hlen]).expect("utf-8 header");
    assert!(header.contains("'|b1'"), "{name}: not a bool array: {header}");
    b[10 + hlen..].iter().map(|&v| v != 0).collect()
}

fn clip_input(dir: &std::path::Path, prefix: &str) -> AudioInput {
    let (shape, feats) = npy_f32(dir, &format!("{prefix}input_features.npy"));
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 1);
    let mask = read_npy(&dir.join(format!("{prefix}input_features_mask.npy"))).unwrap();
    let mask: Vec<bool> = mask.i64_slice().expect("int64 mask").iter().map(|&m| m != 0).collect();
    assert_eq!(mask.len(), shape[1]);
    AudioInput {
        input_features: feats,
        mask,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn scale_of(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn load(rt: &std::sync::Arc<tessl::GpuRuntime>) -> (AudioConfig, EmbedGemma2Audio) {
    let snap = snapshot();
    let cfg = AudioConfig::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let model = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap();
    (cfg, model)
}

/// `(stage, kept-row values we got, reference values)` for one clip.
fn stages(
    cfg: &AudioConfig,
    dir: &std::path::Path,
    prefix: &str,
    out: &AudioOutput,
    i: usize,
) -> Vec<(String, Vec<f32>, Vec<f32>, usize)> {
    let tr = &out.trace[i];
    let (h, od) = (cfg.hidden as usize, cfg.out_dims as usize);
    let mut v = vec![(
        "subsample".to_string(),
        tr.subsample.clone(),
        npy_f32(dir, &format!("{prefix}subsample.npy")).1,
        h,
    )];
    for (l, got) in tr.layers.iter().enumerate() {
        v.push((
            format!("layer{l}"),
            got.clone(),
            npy_f32(dir, &format!("{prefix}layer{l}.npy")).1,
            h,
        ));
    }
    v.push((
        "tower_out".to_string(),
        tr.tower_out.clone(),
        npy_f32(dir, &format!("{prefix}tower_out.npy")).1,
        od,
    ));
    v
}

#[test]
#[ignore]
fn audio_tower_matches_transformers_sdpa() {
    let dir = ref_dir();
    let inputs: Vec<AudioInput> = CLIPS.iter().map(|p| clip_input(&dir, p)).collect();
    with_gpu(|rt| {
        let (cfg, model) = load(rt);
        // Every numeric bound is checked and reported before the test fails,
        // so one run shows the whole picture.
        let mut failures: Vec<String> = Vec::new();
        let mut alone = Vec::new();
        for (i, (prefix, input)) in CLIPS.iter().zip(&inputs).enumerate() {
            let out = model.encode_audio(std::slice::from_ref(input), true).unwrap();
            let want_mask = npy_bool(&dir, &format!("{prefix}out_mask.npy"));
            assert_eq!(out.out_masks[0], want_mask, "{prefix}: the tower's mask");
            let t = want_mask.len();
            for (name, got, want, width) in stages(&cfg, &dir, prefix, &out, 0) {
                assert_eq!(got.len(), t * width, "{prefix}{name}: rows");
                assert_eq!(want.len(), got.len(), "{prefix}{name}: reference rows");
                let (err, scale) = (max_abs(&got, &want), scale_of(&want));
                eprintln!(
                    "{prefix}{name}: {t} rows, max abs {err:.3e} = {:.3e} of max |ref| {scale:.3e}",
                    err / scale
                );
                if err > LAYER_REL * scale {
                    failures.push(format!(
                        "{prefix}{name}: max abs {err:.3e} > {LAYER_REL:.0e} * {scale:.3e}"
                    ));
                }
            }
            let (shape, want_soft) = npy_f32(&dir, &format!("{prefix}soft.npy"));
            let kept = want_mask.iter().filter(|&&m| m).count();
            assert_eq!(shape, [kept, cfg.embed_dim as usize], "{prefix}soft shape");
            let err = max_abs(&out.soft_tokens[0], &want_soft);
            let cos: f64 = {
                let dot: f64 = out.soft_tokens[0]
                    .iter()
                    .zip(&want_soft)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let na = out.soft_tokens[0]
                    .iter()
                    .map(|a| f64::from(*a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nb = want_soft.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
                dot / (na * nb)
            };
            eprintln!(
                "{prefix}soft: {kept} tokens, max abs {err:.3e} (max |ref| {:.3e}), cosine {cos:.9}",
                scale_of(&want_soft)
            );
            let soft_scale = scale_of(&want_soft);
            if err > LAYER_REL * soft_scale {
                failures.push(format!(
                    "{prefix}soft: max abs {err:.3e} > {LAYER_REL:.0e} * {soft_scale:.3e}"
                ));
            }
            assert!(
                out.soft_tokens[0].iter().all(|v| v.is_finite()),
                "{prefix}: non-finite soft token"
            );
            alone.push((out, i));
        }

        // Both clips in one batch: each is bit for bit what it is alone.
        let batch = model.encode_audio(&inputs, true).unwrap();
        assert_eq!(batch.forwards, 1);
        for (a, i) in &alone {
            let b = &batch;
            let mut same = bits(&a.soft_tokens[0]) == bits(&b.soft_tokens[*i]);
            let (ta, tb) = (&a.trace[0], &b.trace[*i]);
            same &= bits(&ta.subsample) == bits(&tb.subsample) && bits(&ta.tower_out) == bits(&tb.tower_out);
            same &= ta.layers.iter().zip(&tb.layers).all(|(x, y)| bits(x) == bits(y));
            let drift = max_abs(&a.soft_tokens[0], &b.soft_tokens[*i]);
            eprintln!(
                "{}: alone vs batched soft max abs {drift:.3e}, bit-identical {same}",
                CLIPS[*i]
            );
            if !same {
                failures.push(format!(
                    "{}: alone vs batched differ (soft max abs {drift:.3e})",
                    CLIPS[*i]
                ));
            }
        }
        // The reverse order is the same two answers.
        let rev: Vec<AudioInput> = inputs.iter().rev().cloned().collect();
        let r = model.encode_audio(&rev, false).unwrap();
        for (pos, i) in [(0usize, 1usize), (1, 0)] {
            if bits(&r.soft_tokens[pos]) != bits(&batch.soft_tokens[i]) {
                failures.push(format!("{}: reordered batch differs", CLIPS[i]));
            }
        }

        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// `GemmOperands::Bf16` on the real checkpoint, the tower alone: the weights
/// are bf16 on the device, each clip's error against the fp32 goldens is
/// reported per stage and its soft tokens are gated by cosine (the
/// measurement is quoted in `docs/embedgemma2.md`), and each clip alone, both
/// in one forward and in the reverse order are bit for bit equal (the bf16 GEMM
/// sums K in the same order whatever M is, and no row reads another's).
#[test]
#[ignore]
fn embedgemma2_audio_bf16_is_batch_independent_and_close_to_fp32() {
    let dir = ref_dir();
    let snap = snapshot();
    let inputs: Vec<AudioInput> = CLIPS.iter().map(|p| clip_input(&dir, p)).collect();
    with_gpu(|rt| {
        let cfg = AudioConfig::from_config_file(&snap.join("config.json")).unwrap();
        let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
        let exact_bytes = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap().weight_bytes();
        let model = EmbedGemma2Audio::load_with(rt, &st, "", cfg.clone(), GemmOperands::Bf16).unwrap();
        assert_eq!(model.operands(), GemmOperands::Bf16);
        eprintln!(
            "weight bytes: bf16 {} ({:.1} MiB), exact f32 {} ({:.1} MiB), {:.1}%",
            model.weight_bytes(),
            model.weight_bytes() as f64 / (1 << 20) as f64,
            exact_bytes,
            exact_bytes as f64 / (1 << 20) as f64,
            100.0 * model.weight_bytes() as f64 / exact_bytes as f64
        );
        assert!(model.weight_bytes() < exact_bytes);

        let mut failures: Vec<String> = Vec::new();
        let mut alone = Vec::new();
        for (prefix, input) in CLIPS.iter().zip(&inputs) {
            let out = model.encode_audio(std::slice::from_ref(input), true).unwrap();
            let want_mask = npy_bool(&dir, &format!("{prefix}out_mask.npy"));
            assert_eq!(out.out_masks[0], want_mask, "{prefix}: the tower's mask");
            let t = want_mask.len();
            let per_stage: Vec<String> = stages(&cfg, &dir, prefix, &out, 0)
                .into_iter()
                .map(|(name, got, want, width)| {
                    assert_eq!(got.len(), t * width, "{prefix}{name}: rows");
                    format!("{name} {:.1e}", max_abs(&got, &want) / scale_of(&want))
                })
                .collect();
            eprintln!(
                "{prefix}bf16 vs fp32 goldens, max abs over max |ref|: {}",
                per_stage.join(", ")
            );
            let (_, want_soft) = npy_f32(&dir, &format!("{prefix}soft.npy"));
            let cos = cosine(&out.soft_tokens[0], &want_soft);
            let rel = max_abs(&out.soft_tokens[0], &want_soft) / scale_of(&want_soft);
            eprintln!("{prefix}bf16 soft tokens vs fp32: cosine {cos:.9}, max abs over max |ref| {rel:.3e}");
            if cos < SOFT_COS_FLOOR {
                failures.push(format!("{prefix}soft: cosine {cos:.9} < {SOFT_COS_FLOOR}"));
            }
            assert!(
                out.soft_tokens[0].iter().all(|v| v.is_finite()),
                "{prefix}: non-finite soft token"
            );
            alone.push(out.soft_tokens.into_iter().next().unwrap());
        }
        let batch = model.encode_audio(&inputs, false).unwrap();
        assert_eq!(batch.forwards, 1, "the two clips must share one forward");
        let rev: Vec<AudioInput> = inputs.iter().rev().cloned().collect();
        let r = model.encode_audio(&rev, false).unwrap();
        for (i, a) in alone.iter().enumerate() {
            if bits(a) != bits(&batch.soft_tokens[i]) {
                failures.push(format!(
                    "{}: batched differs from alone (max abs {:.3e})",
                    CLIPS[i],
                    max_abs(a, &batch.soft_tokens[i])
                ));
            }
            if bits(a) != bits(&r.soft_tokens[1 - i]) {
                failures.push(format!("{}: reversed call differs from alone", CLIPS[i]));
            }
        }

        // Long clips, which the goldens are not: 90 s of frames (hundreds of
        // attention blocks), two of them, which together pass the frame budget
        // of one forward (two forwards). Each token of the bf16 forward against
        // the same token of the exact one.
        let exact = EmbedGemma2Audio::load(rt, &st, "", cfg.clone()).unwrap();
        let long = |frames: usize, shift: usize| {
            let (base, mels) = (&inputs[1], cfg.mel_bins() as usize);
            let mut f = Vec::with_capacity(frames * mels);
            for r in 0..frames {
                let src = (r + shift) % base.mask.len();
                f.extend_from_slice(&base.input_features[src * mels..(src + 1) * mels]);
            }
            AudioInput {
                input_features: f,
                mask: vec![true; frames],
            }
        };
        let clips = [long(9000, 0), long(9000, 41)];
        let (want, got) = (
            exact.encode_audio(&clips, false).unwrap(),
            model.encode_audio(&clips, false).unwrap(),
        );
        assert_eq!(
            (want.forwards, got.forwards),
            (2, 2),
            "18,000 frames pass one forward's budget"
        );
        let width = cfg.embed_dim as usize;
        for (i, (w, g)) in want.soft_tokens.iter().zip(&got.soft_tokens).enumerate() {
            assert_eq!(w.len(), g.len(), "long clip {i}: tokens");
            let worst = w
                .chunks(width)
                .zip(g.chunks(width))
                .map(|(a, b)| cosine(a, b))
                .fold(f64::INFINITY, f64::min);
            let whole = cosine(g, w);
            eprintln!(
                "long clip {i} ({} tokens): bf16 vs exact cosine {whole:.9}, worst token {worst:.9}",
                w.len() / width
            );
            if worst < LONG_TOKEN_COS_FLOOR {
                failures.push(format!(
                    "long clip {i}: a token is at cosine {worst:.9} from the exact forward's (floor {LONG_TOKEN_COS_FLOOR})"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}

/// The clip embeddings end to end with every model in bf16 (`GemmOperands::Bf16`
/// through the facade: the audio tower and the text encoder; no image in these
/// calls), against sentence-transformers' fp32 embeddings and its own bf16
/// forward: the pattern `tests/embedgemma2_vision_model.rs` holds the images
/// to. Bounds (the constants above, from the measurement quoted in
/// `docs/embedgemma2.md`):
///
/// * each clip at cosine `EMB_COS_FLOOR` or better against `audio{n}_emb_f32.npy`;
/// * each clip at cosine `HF_COS_FLOOR` or better against `audio{n}_emb_bf16.npy`;
/// * over the two clips the mean of `1 - cos` against fp32 at most that of
///   transformers' bf16 forward (`emb_bf16.npy` against `emb_f32.npy`);
/// * each clip alone within `1e-5` max abs of itself inside one call of both.
#[test]
#[ignore]
fn embedgemma2_multimodal_bf16_clips_are_no_worse_than_transformers_bf16() {
    let dir = ref_dir();
    let snap = snapshot();
    let inputs: Vec<AudioInput> = CLIPS.iter().map(|p| clip_input(&dir, p)).collect();
    let emb = |n: usize, kind: &str| npy_f32(&dir, &format!("audio{n}_emb_{kind}.npy")).1;
    let (f32_ref, bf16_ref): (Vec<_>, Vec<_>) = (0..2).map(|n| (emb(n, "f32"), emb(n, "bf16"))).unzip();
    with_gpu(|rt| {
        let m = EmbedGemma2Multimodal::load(rt, &snap, GemmOperands::Bf16).unwrap();
        assert_eq!(m.text().operands(), GemmOperands::Bf16);
        assert_eq!(m.audio().operands(), GemmOperands::Bf16);
        let dim = f32_ref[0].len();
        let mut failures: Vec<String> = Vec::new();
        let (mut deficit_ours, mut deficit_hf) = (0.0f64, 0.0f64);
        let mut alone: Vec<Vec<f32>> = Vec::new();
        for n in 0..2 {
            let e = m.encode(&[MmItem::Audio(&inputs[n])], None).unwrap().embeddings;
            let (cos_f, cos_h, cos_hf) = (
                cosine(&e, &f32_ref[n]),
                cosine(&e, &bf16_ref[n]),
                cosine(&bf16_ref[n], &f32_ref[n]),
            );
            eprintln!(
                "clip {n}: tessl bf16 (tower + text) vs fp32 cosine {cos_f:.8} (max abs {:.2e}); vs transformers bf16 \
                 cosine {cos_h:.8}; transformers bf16 vs fp32 cosine {cos_hf:.8}",
                max_abs(&e, &f32_ref[n])
            );
            deficit_ours += 1.0 - cos_f;
            deficit_hf += 1.0 - cos_hf;
            if cos_f < EMB_COS_FLOOR {
                failures.push(format!("clip {n}: cosine vs fp32 {cos_f:.8} < {EMB_COS_FLOOR}"));
            }
            if cos_h < HF_COS_FLOOR {
                failures.push(format!(
                    "clip {n}: cosine vs transformers bf16 {cos_h:.8} < {HF_COS_FLOOR}"
                ));
            }
            let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "clip {n}: norm {norm}");
            alone.push(e);
        }
        eprintln!(
            "mean 1 - cos vs fp32: tessl bf16 {:.3e}, transformers bf16 {:.3e}",
            deficit_ours / 2.0,
            deficit_hf / 2.0
        );
        if deficit_ours > deficit_hf {
            failures.push(format!(
                "tessl bf16 is further from fp32 (mean 1 - cos {:.3e}) than transformers' bf16 forward ({:.3e})",
                deficit_ours / 2.0,
                deficit_hf / 2.0
            ));
        }
        let both = m
            .encode(&[MmItem::Audio(&inputs[0]), MmItem::Audio(&inputs[1])], None)
            .unwrap()
            .embeddings;
        for (n, a) in alone.iter().enumerate() {
            let drift = max_abs(a, &both[n * dim..(n + 1) * dim]);
            eprintln!(
                "clip {n}: alone vs both in one call max abs {drift:.3e} (bit-identical: {})",
                bits(a) == bits(&both[n * dim..(n + 1) * dim])
            );
            if drift > BATCH_ABS {
                failures.push(format!(
                    "clip {n}: alone vs inside one call max abs {drift:.3e} > {BATCH_ABS:.0e}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}

/// Transformers' eager references (`make_mm_reference.py` with
/// `EMBEDGEMMA2_MM_ATTN=eager`) agree with this
/// forward through the subsampling, which has no attention, and disagree from
/// layer 0: eager inverts the attention mask. Opt-in on a second directory.
#[test]
#[ignore]
fn eager_references_encode_the_inverted_mask() {
    let eager = PathBuf::from(std::env::var("EMBEDGEMMA2_MM_REF_EAGER_DIR").expect(
        "EMBEDGEMMA2_MM_REF_EAGER_DIR must name make_mm_reference.py's output under EMBEDGEMMA2_MM_ATTN=eager",
    ));
    let sdpa = ref_dir();
    let input = clip_input(&sdpa, CLIPS[0]);
    with_gpu(|rt| {
        let (cfg, model) = load(rt);
        let out = model.encode_audio(std::slice::from_ref(&input), true).unwrap();
        let mut first_bad = None;
        for (name, got, want, _) in stages(&cfg, &eager, CLIPS[0], &out, 0) {
            let (err, scale) = (max_abs(&got, &want), scale_of(&want));
            eprintln!(
                "eager {}{name}: max abs {err:.3e} = {:.3e} of max |ref| {scale:.3e}",
                CLIPS[0],
                err / scale
            );
            if name == "subsample" {
                assert!(
                    err <= LAYER_REL * scale,
                    "the subsampling does not touch attention: {err:.3e}"
                );
            } else if err > LAYER_REL * scale && first_bad.is_none() {
                first_bad = Some((name, err / scale));
            }
        }
        let (name, rel) = first_bad.expect(
            "the eager references agree: the transformers 5.19.0 eager mask bug is fixed upstream, retire this test",
        );
        assert_eq!(name, "layer0", "the eager references first disagree at {name}");
        assert!(rel > 1e-2, "layer0 differs by only {rel:.3e}");
    });
}
