//! The whole EmbeddingGemma 2 text encoder against sentence-transformers.
//!
//! Every kernel is checked against transformers on its own
//! (`tests/embedgemma2_kernels.rs`); this checks the composition
//! (`tessl::embedgemma2`): layer order, the per-layer inputs, the layer
//! scalars, the window per layer type, pooling and normalization. It needs the
//! real checkpoint and the references `tools/embedgemma2_ref/make_reference.py`
//! writes, so it is opt-in:
//!
//! ```text
//! ~/.venvs/ml/bin/python tools/embedgemma2_ref/make_reference.py   # once
//! EMBEDGEMMA2_SNAPSHOT=/path/to/google/embeddinggemma-2/snapshot \
//!   cargo test --release --test embedgemma2_model -- --ignored --test-threads=1
//! ```
//!
//! `EMBEDGEMMA2_REF_DIR` overrides `target/embedgemma2_ref`. A missing
//! variable or file fails the test rather than skipping it.
//!
//! Bounds, fixed before the first run (the forward is transformers' fp32
//! forward up to operation order, with exact-f32 GEMMs); the observed errors
//! are recorded in `docs/embedgemma2.md`:
//!
//! * the residual stream after every layer within `1e-4` of that layer's
//!   largest magnitude, and the final norm's output likewise, for text 0 and
//!   for the 1658-token text (past twice the window, so the sliding and full
//!   layers see different keys, and far enough that RoPE's frequency table
//!   matters: `nn::rope_inv_freq`);
//! * every text's embedding within `1e-4` (max abs, on unit vectors) of the
//!   fp32 reference, cosine at least `0.99999`;
//! * every text embedded alone and inside one ragged batch agree within
//!   `1e-5` (padding is masked, so only GEMM tiling can move them); the
//!   batch needs more than one forward, so the split is checked too;
//! * the Matryoshka prefixes (512, 256, 128) within the embedding bounds of
//!   sentence-transformers' `encode(truncate_dim=d, normalize_embeddings=True)`.
//!
//! The bf16 reference is reported, not gated: it is how far the checkpoint's
//! own dtype moves the embedding, the scale against which `1e-4` is small.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model};
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const PREFIX: &str = "language_model.";
const LAYER_REL: f64 = 1e-4;
const EMB_ABS: f64 = 1e-4;
const EMB_COS: f64 = 0.99999;
const BATCH_ABS: f64 = 1e-5;
/// `make_reference.py`'s TRACE_LONG: the 1658-token text.
const TRACE_LONG: usize = 6;

fn ref_dir() -> PathBuf {
    std::env::var_os("EMBEDGEMMA2_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/embedgemma2_ref"))
}

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("EMBEDGEMMA2_SNAPSHOT")
            .expect("EMBEDGEMMA2_SNAPSHOT must name the google/embeddinggemma-2 snapshot directory"),
    )
}

fn npy_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let p = ref_dir().join(name);
    let a =
        read_npy(&p).unwrap_or_else(|e| panic!("{}: {e} (run tools/embedgemma2_ref/make_reference.py)", p.display()));
    (a.shape.clone(), a.f32_slice().expect("f32 reference").to_vec())
}

fn npy_i64(name: &str) -> Vec<i64> {
    let p = ref_dir().join(name);
    let a = read_npy(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    a.data_i64.clone().expect("i64 reference")
}

fn texts() -> Vec<Vec<u32>> {
    let ids = npy_i64("ids.npy");
    let off = npy_i64("offsets.npy");
    off.windows(2)
        .map(|w| {
            ids[w[0] as usize..w[1] as usize]
                .iter()
                .map(|&i| u32::try_from(i).unwrap())
                .collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore]
fn embedgemma2_matches_sentence_transformers() {
    let snap = snapshot();
    let cfg = EmbedGemma2Config::from_config_file(&snap.join("config.json")).unwrap();
    let st = SafeTensors::open(&snap.join("model.safetensors")).unwrap();
    let texts = texts();
    let (es, want) = npy_f32("emb_f32.npy");
    let (_, bf16) = npy_f32("emb_bf16.npy");
    let dim = es[1];
    assert_eq!(es[0], texts.len(), "reference rows vs texts");

    with_gpu(|rt| {
        let model = EmbedGemma2Model::load(rt, &st, PREFIX, cfg.clone()).unwrap();
        // Every numeric bound is checked and reported before the test fails,
        // so one run shows the whole picture.
        let mut failures: Vec<String> = Vec::new();

        // 1. Layer by layer: text 0, then the long text, which spans more
        //    than two windows, so its sliding and full layers differ.
        let h = cfg.hidden as usize;
        for (n, prefix) in [(0usize, "trace"), (TRACE_LONG, "trace_long")] {
            let t = texts[n].len();
            if prefix == "trace_long" {
                assert!(t > 2 * cfg.sliding_window as usize, "the long trace has {t} tokens");
            }
            let out = model.encode(&[&texts[n]], None, true).unwrap();
            assert_eq!(out.trace.len(), cfg.layers.len() + 1);
            for (i, got) in out.trace.iter().enumerate() {
                let name = if i < cfg.layers.len() {
                    format!("{prefix}_l{i}.npy")
                } else {
                    format!("{prefix}_final.npy")
                };
                let (shape, want_l) = npy_f32(&name);
                assert_eq!(shape, [t, h], "{name}");
                let scale = want_l.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
                let err = max_abs(&got[..t * h], &want_l);
                eprintln!(
                    "{name}: {t} tokens, max abs {err:.3e} = {:.3e} of max |ref| {scale:.3e}",
                    err / scale
                );
                if err > LAYER_REL * scale {
                    failures.push(format!("{name}: max abs {err:.3e} > {LAYER_REL:.0e} * {scale:.3e}"));
                }
            }
        }

        // 2. Every text alone, against fp32 (gated) and bf16 (reported).
        let mut alone = Vec::new();
        for (n, ids) in texts.iter().enumerate() {
            let e = model.encode(&[ids], None, false).unwrap().embeddings;
            let w = &want[n * dim..(n + 1) * dim];
            let (err, cos) = (max_abs(&e, w), cosine(&e, w));
            let cos16 = cosine(w, &bf16[n * dim..(n + 1) * dim]);
            eprintln!(
                "text {n}: {} tokens, max abs {err:.2e}, cosine {cos:.8}; bf16 ref vs fp32 ref cosine {cos16:.6}",
                ids.len()
            );
            if !(err <= EMB_ABS && cos >= EMB_COS) {
                failures.push(format!("text {n}: max abs {err:.3e}, cosine {cos:.8}"));
            }
            let norm = e.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "text {n}: norm {norm}");
            alone.push(e);
        }

        // 3. The same texts as one ragged batch.
        // The long document cannot share a forward with the rest under
        // MAX_BATCH_TOKENS, so this also checks the split and the reordering.
        let refs: Vec<&[u32]> = texts.iter().map(Vec::as_slice).collect();
        let out = model.encode(&refs, None, false).unwrap();
        assert!(
            out.forwards >= 2,
            "expected a split batch, got {} forward(s)",
            out.forwards
        );
        let batched = out.embeddings;
        for (n, e) in alone.iter().enumerate() {
            let err = max_abs(e, &batched[n * dim..(n + 1) * dim]);
            eprintln!("text {n}: alone vs batched max abs {err:.2e}");
            if err > BATCH_ABS {
                failures.push(format!("text {n}: alone vs batched max abs {err:.3e}"));
            }
        }

        // 4. Matryoshka prefixes, batched, against sentence-transformers'
        //    encode(truncate_dim=d, normalize_embeddings=True).
        for d in [512usize, 256, 128] {
            let (shape, want) = npy_f32(&format!("emb_f32_trunc{d}.npy"));
            assert_eq!(shape, [texts.len(), d]);
            let got = model.encode(&refs, Some(d as u32), false).unwrap().embeddings;
            for n in 0..texts.len() {
                let (g, w) = (&got[n * d..(n + 1) * d], &want[n * d..(n + 1) * d]);
                let (err, cos) = (max_abs(g, w), cosine(g, w));
                eprintln!("prefix {d}, text {n}: max abs {err:.2e}, cosine {cos:.8}");
                if !(err <= EMB_ABS && cos >= EMB_COS) {
                    failures.push(format!("prefix {d}, text {n}: max abs {err:.3e}, cosine {cos:.8}"));
                }
            }
        }

        // 5. The refusals the API promises.
        assert!(model.encode(&[], None, false).is_err(), "empty batch");
        assert!(model.encode(&[&[]], None, false).is_err(), "empty sequence");
        assert!(model.encode(&[&[cfg.vocab]], None, false).is_err(), "id past vocab");
        assert!(
            model.encode(&[&texts[0], &texts[1]], None, true).is_err(),
            "trace takes one sequence"
        );
        let too_long = vec![2u32; model.max_tokens() as usize + 1];
        assert!(model.encode(&[&too_long], None, false).is_err(), "past max_tokens");
        assert!(
            model.encode(&[&texts[0]], Some(769), false).is_err(),
            "truncate_dim past 768"
        );

        assert!(
            failures.is_empty(),
            "{} bound(s) exceeded:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
