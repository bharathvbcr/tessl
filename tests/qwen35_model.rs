//! The whole Qwen3.5-2B text forward against transformers.
//!
//! Every kernel is checked against transformers on its own; this checks their
//! composition (`tessl::qwen35_model`): layer order, norm offsets, weight
//! layouts, the tied LM head. It needs the real checkpoint and the reference
//! outputs `tools/qwen35_ref/make_reference.py` writes, so it is opt-in:
//!
//! ```text
//! python3 tools/qwen35_ref/make_reference.py      # once: target/qwen35_ref/
//! QWEN35_2B_SAFETENSORS=/path/to/model.safetensors-00001-of-00001.safetensors \
//!   cargo test --release --test qwen35_model -- --ignored --test-threads=1
//! ```
//!
//! `QWEN35_REF_DIR` overrides `target/qwen35_ref`. A missing variable or file
//! fails the test rather than skipping it.
//!
//! Bounds, fixed before any run:
//!
//! * `F32`: the same arithmetic as transformers' fp32 forward up to operation
//!   order. Each layer's residual stream within 1e-4 of its largest magnitude,
//!   the logits within 1e-3 of the largest logit, and the same top-1 token at
//!   every position. (The kernels agree with f64 to ~1e-7 relative each; 24
//!   layers of reordering leave orders of magnitude of margin.)
//! * `Bf16`: bf16 GEMM inputs are a real change of numerics, so it is held to
//!   transformers' own bf16 forward instead: its mean KL divergence from the
//!   fp32 reference and its top-1 disagreements with it may not exceed the
//!   bf16 reference's.

mod common;

use std::path::{Path, PathBuf};

use tessl::npy::read_npy;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

const PREFIX: &str = "model.language_model.";

fn ref_dir() -> PathBuf {
    std::env::var_os("QWEN35_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/qwen35_ref"))
}

fn load_ref(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = ref_dir().join(format!("{name}.npy"));
    let a = read_npy(&path).unwrap_or_else(|e| panic!("{e}\n(run python3 tools/qwen35_ref/make_reference.py first)"));
    let data = if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| x as f64).collect()
    } else if let Some(i) = &a.data_i64 {
        i.iter().map(|&x| x as f64).collect()
    } else {
        panic!("{name}: unexpected dtype")
    };
    (a.shape, data)
}

fn checkpoint() -> SafeTensors {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .expect("set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B-Base .safetensors file");
    SafeTensors::open(Path::new(&path)).unwrap()
}

fn ids() -> Vec<u32> {
    let (_, ids) = load_ref("ids");
    ids.iter()
        .map(|&x| u32::try_from(x as i64).expect("token id fits u32"))
        .collect()
}

/// max |got - want| / max |want|.
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()));
    err / scale.max(f64::MIN_POSITIVE)
}

fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        )
        .0
}

fn log_softmax(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// KL(p || q) from logits, per row.
fn kl_rows(p_logits: &[f64], q_logits: &[f64], vocab: usize) -> Vec<f64> {
    p_logits
        .chunks(vocab)
        .zip(q_logits.chunks(vocab))
        .map(|(p, q)| {
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
        })
        .collect()
}

fn run(precision: Precision) -> (Vec<f32>, Vec<Vec<f32>>) {
    let rt = GpuRuntime::new().unwrap();
    let st = checkpoint();
    let model = Qwen35Model::load(&rt, &st, PREFIX, Qwen35Config::qwen35_2b().unwrap(), precision).unwrap();
    let out = model.forward(&ids(), true).unwrap();
    assert!(out.logits.iter().all(|x| x.is_finite()), "non-finite logits");
    (out.logits, out.trace)
}

#[test]
#[ignore]
fn f32_forward_matches_transformers_fp32_layer_by_layer() {
    let (logits, trace) = run(Precision::F32);
    let cfg = Qwen35Config::qwen35_2b().unwrap();
    let n_layers = cfg.layers.len();
    assert_eq!(trace.len(), n_layers + 1, "trace: every layer, then the final norm");
    let mut worst = 0.0f64;
    for (l, got) in trace[..n_layers - 1].iter().enumerate() {
        let (_, want) = load_ref(&format!("hidden_f32_{l}"));
        let r = rel(got, &want);
        eprintln!("layer {l:2} ({:?}): rel err {r:.2e}", cfg.layers[l]);
        worst = worst.max(r);
        assert!(r <= 1e-4, "layer {l}: residual stream rel err {r:.3e} > 1e-4");
    }
    // transformers reports the final norm's output in place of the last
    // layer's residual stream.
    let (_, want) = load_ref("final_norm_f32");
    let r = rel(&trace[n_layers], &want);
    eprintln!("final norm: rel err {r:.2e} (worst layer {worst:.2e})");
    assert!(r <= 1e-4, "final norm rel err {r:.3e} > 1e-4");

    let (shape, want) = load_ref("logits_f32");
    let vocab = shape[1];
    let r = rel(&logits, &want);
    let got64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let mismatched: Vec<usize> = got64
        .chunks(vocab)
        .zip(want.chunks(vocab))
        .enumerate()
        .filter(|(_, (g, w))| argmax(g) != argmax(w))
        .map(|(t, _)| t)
        .collect();
    let kl = kl_rows(&want, &got64, vocab);
    let kl_max = kl.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "logits: rel err {r:.2e}, top-1 mismatches {}/{}, max KL {kl_max:.2e}",
        mismatched.len(),
        shape[0]
    );
    assert!(r <= 1e-3, "logits rel err {r:.3e} > 1e-3");
    assert!(mismatched.is_empty(), "top-1 differs at positions {mismatched:?}");
}

#[test]
#[ignore]
fn bf16_forward_is_no_worse_than_transformers_bf16() {
    let (logits, _) = run(Precision::Bf16);
    let (shape, fp32) = load_ref("logits_f32");
    let (_, hf_bf16) = load_ref("logits_bf16");
    let vocab = shape[1];
    let got64: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let kl_tessl = kl_rows(&fp32, &got64, vocab);
    let kl_hf = kl_rows(&fp32, &hf_bf16, vocab);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let top1_miss = |q: &[f64]| {
        fp32.chunks(vocab)
            .zip(q.chunks(vocab))
            .filter(|(p, q)| argmax(p) != argmax(q))
            .count()
    };
    let (m_t, m_h) = (top1_miss(&got64), top1_miss(&hf_bf16));
    eprintln!(
        "vs fp32 over {} positions: tessl bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_t}; \
         transformers bf16 mean KL {:.3e} (max {:.3e}), top-1 misses {m_h}",
        shape[0],
        mean(&kl_tessl),
        kl_tessl.iter().cloned().fold(0.0, f64::max),
        mean(&kl_hf),
        kl_hf.iter().cloned().fold(0.0, f64::max),
    );
    assert!(
        mean(&kl_tessl) <= mean(&kl_hf),
        "tessl bf16 is further from fp32 than transformers' bf16 forward"
    );
    assert!(
        m_t <= m_h,
        "tessl bf16 misses more top-1 tokens ({m_t}) than transformers bf16 ({m_h})"
    );
}
