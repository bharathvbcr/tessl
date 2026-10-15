//! The audio tower's f64 reference is transformers.
//!
//! `tests/fixtures/embedgemma2/eg2a_audio_*` are the outputs of a tiny random
//! `Gemma4AudioModel` and `EmbeddingGemma2MultimodalEmbedder` under sdpa
//! (`scripts/gen_embedgemma2_audio_fixtures.py`): the checkpoint's attention
//! geometry (chunk 12, context left 13) at hidden 32, finite clip bounds, a
//! gradient clip and a logit cap that bite, on three clips (all frames valid,
//! trailing frames masked, a hole in the middle). The f64 reference of
//! `tests/common/embedgemma2_audio.rs` (written the way transformers writes the
//! chunked attention, and the base of `tests/embedgemma2_audio_kernels.rs` and
//! `tests/embedgemma2_audio_tiny.rs`) is held to them here, on the CPU.
//!
//! Bound, fixed before any run: transformers' f32 output against the f64
//! reference within `1e-5 * max|ref|` (the f32 forward's own rounding through a
//! layer, the same kind of bound as `tests/embedgemma2_kernels.rs`'s `1e-5`
//! for a single op).
//!
//! The eager output of one clip is pinned too: transformers' eager attention
//! inverts the local mask (see `src/embedgemma2_audio.rs`), and the reference
//! must be far from it.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use common::embedgemma2_audio::{audio_forward, Weights};
use tessl::embedgemma2_audio::AudioConfig;
use tessl::npy::read_npy;

const FIXTURE_REL: f64 = 1e-5;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embedgemma2")
}

fn f32s(name: &str) -> Vec<f64> {
    let p = dir().join(format!("eg2a_audio_{name}.npy"));
    let a = read_npy(&p).unwrap_or_else(|e| panic!("load {}: {e}", p.display()));
    a.f32_slice()
        .expect("f32 fixture")
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn ints(name: &str) -> Vec<i64> {
    let p = dir().join(format!("eg2a_audio_{name}.npy"));
    read_npy(&p)
        .unwrap_or_else(|e| panic!("load {}: {e}", p.display()))
        .i64_slice()
        .expect("i64 fixture")
        .to_vec()
}

fn config() -> AudioConfig {
    let text = std::fs::read_to_string(dir().join("eg2a_audio_config.json")).unwrap();
    AudioConfig::from_config_json(&text).unwrap()
}

/// Every tensor of the model by its checkpoint name: `index.txt` lists them
/// (name, dims) in the order `weights.npy` concatenates them.
fn weights() -> Weights {
    let flat = f32s("weights");
    let index = std::fs::read_to_string(dir().join("eg2a_audio_index.txt")).unwrap();
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();
    let mut at = 0;
    for line in index.lines() {
        let mut it = line.split(' ');
        let name = it.next().unwrap().to_string();
        let n: usize = it.map(|d| d.parse::<usize>().unwrap()).product();
        out.insert(name, flat[at..at + n].to_vec());
        at += n;
    }
    assert_eq!(at, flat.len(), "index.txt covers the whole weights file");
    out
}

fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale
}

#[test]
fn the_f64_reference_matches_transformers() {
    let cfg = config();
    assert_eq!((cfg.hidden, cfg.chunk, cfg.past(), cfg.mel_bins()), (32, 12, 12, 16));
    let w = weights();
    for n in 0..3 {
        let feats = f32s(&format!("c{n}_feats"));
        let mask: Vec<bool> = ints(&format!("c{n}_mask")).iter().map(|&m| m != 0).collect();
        let got = audio_forward(&cfg, &w, &feats, &mask);
        let want_mask: Vec<bool> = ints(&format!("c{n}_out_mask")).iter().map(|&m| m != 0).collect();
        assert_eq!(got.out_mask, want_mask, "clip {n}: the tower's mask");
        let stages = [
            ("subsample", &got.subsample, f32s(&format!("c{n}_subsample"))),
            ("layer0", &got.layers[0], f32s(&format!("c{n}_layer0"))),
            ("tower", &got.tower, f32s(&format!("c{n}_tower"))),
            ("soft", &got.soft, f32s(&format!("c{n}_soft"))),
        ];
        for (name, g, want) in stages {
            let rel = max_rel(g, &want);
            eprintln!(
                "clip {n} ({} frames, {} valid) {name}: max abs error {rel:.2e} of max |ref|",
                mask.len(),
                mask.iter().filter(|&&m| m).count()
            );
            assert!(rel <= FIXTURE_REL, "clip {n} {name}: {rel:.3e} > {FIXTURE_REL:.0e}");
        }
    }
}

/// Eager attention applies `logical_not()` to a float mask: the local window
/// is inverted, and the output is not the sdpa (specified) one.
#[test]
fn transformers_eager_differs_from_the_specified_mask() {
    let cfg = config();
    let (feats, mask) = (
        f32s("c1_feats"),
        ints("c1_mask").iter().map(|&m| m != 0).collect::<Vec<_>>(),
    );
    let got = audio_forward(&cfg, &weights(), &feats, &mask);
    let eager = f32s("c1_eager_tower");
    let rel = max_rel(&got.tower, &eager);
    eprintln!("eager vs specified tower output: {rel:.2e} of max |ref|");
    assert!(
        rel > 1e-2,
        "eager agrees with the specified mask ({rel:.2e}): retire this test"
    );
}
