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

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2_audio::{AudioConfig, AudioInput, AudioOutput, EmbedGemma2Audio};
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const LAYER_REL: f64 = 1e-4;
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
