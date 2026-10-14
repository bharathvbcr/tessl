//! `BertSparseModel::encode` against transformers, on a tiny random `BertForMaskedLM`
//! and `DistilBertForMaskedLM` (tests/fixtures/bert_tiny/, written by
//! scripts/gen_bert_fixtures.py; manifest.json records the torch / transformers /
//! numpy versions, the seed and the sha256 of every file).
//!
//! The goldens are transformers' fp32 eager forward of each sequence alone, with no
//! padding. The Rust side encodes all six as one padded batch, so padding masks and
//! the pooling's treatment of padded rows are held to the unpadded truth as well.
//! They cover layer order, head wiring, the tied decoder, DistilBERT's tensor
//! names, and the weight layouts: a wrong one moves the outputs by O(1).
//!
//! Bounds, fixed before the first run and never widened after seeing results:
//!
//! * Every compared value (`pooled`, the embedding LayerNorm output, each layer's
//!   output) must satisfy `|got - want| <= 2e-4 + 2e-4 * |want|`.
//! * Why that size: the forward is exact-f32 GEMMs of depth K <= 96 (u = 2^-24, so
//!   one dot product is off by at most about K * u * sum|a*b| ~ 6e-6 relative
//!   to the operands' magnitude), then LayerNorm, an erf GELU and softmax, about 15 stages
//!   to the logits. Summing 15 worst-case stage errors with a LayerNorm gain of a few
//!   gives roughly 1e-4; 2e-4 is that bound with the transcendental implementation
//!   differences (Metal erf/exp/log vs libm) inside it. The typical error is far below
//!   the worst case. A fault the test must catch (swapped layers, a transposed weight,
//!   a wrong constant) moves values by 1e-2 or more.
//! * The goldens must carry signal (generator asserts it too): more than 20% of the
//!   pooled entries above 0.5, so "close to zero" cannot pass.

mod common;

use std::path::PathBuf;

use common::with_gpu;
use tessl::bert::{BertConfig, BertFamily, BertSparseModel};
use tessl::npy::read_npy;
use tessl::safetensors::SafeTensors;

const ABS: f32 = 2e-4;
const REL: f32 = 2e-4;

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bert_tiny")
        .join(name)
}

fn npy_f32(d: &std::path::Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let a = read_npy(&d.join(name)).unwrap_or_else(|e| panic!("{name}: {e} (run scripts/gen_bert_fixtures.py)"));
    (a.shape.clone(), a.f32_slice().expect("f32 golden").to_vec())
}

fn npy_i64(d: &std::path::Path, name: &str) -> Vec<i64> {
    let a = read_npy(&d.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    a.i64_slice().expect("i64 golden").to_vec()
}

/// The first element outside the bound, as a message.
fn first_violation(label: &str, got: &[f32], want: &[f32]) -> Option<String> {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = (0.0f32, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        if err.is_nan() || err > ABS + REL * w.abs() {
            return Some(format!(
                "{label}[{i}]: got {g}, want {w} (|err| {err} > {ABS} + {REL}*|want|)"
            ));
        }
        if err > worst.0 {
            worst = (err, i);
        }
    }
    eprintln!("{label}: max |err| {:e} at {}", worst.0, worst.1);
    None
}

/// Load the family's fixture (or `st`, replacement weights for the fault check)
/// and compare `encode` to the goldens.
fn check(family: BertFamily, name: &str, st: Option<SafeTensors>) -> Result<(), String> {
    let d = dir(name);
    let cfg = BertConfig::from_config_file(&d.join("config.json"))?;
    assert_eq!(cfg.family, family);
    let st = match st {
        Some(st) => st,
        None => SafeTensors::open(&d.join("model.safetensors"))?,
    };
    let ids = npy_i64(&d, "ids.npy");
    let lens = npy_i64(&d, "lens.npy");
    let (pshape, pooled) = npy_f32(&d, "pooled.npy");
    let (tshape, trace) = npy_f32(&d, "trace.npy");
    let vocab = cfg.vocab as usize;
    assert_eq!(pshape, vec![lens.len(), vocab]);
    assert_eq!(
        tshape,
        vec![cfg.layers as usize + 1, lens[0] as usize, cfg.hidden as usize]
    );
    assert!(
        pooled.iter().filter(|&&v| v > 0.5).count() * 5 > pooled.len(),
        "{name}: pooled goldens carry too little signal"
    );

    let mut seqs: Vec<Vec<u32>> = Vec::new();
    let mut at = 0usize;
    for &n in &lens {
        seqs.push(ids[at..at + n as usize].iter().map(|&i| i as u32).collect());
        at += n as usize;
    }
    assert_eq!(at, ids.len());
    let refs: Vec<&[u32]> = seqs.iter().map(Vec::as_slice).collect();

    with_gpu(|rt| {
        let model = BertSparseModel::load(rt, &st, cfg.clone())?;
        let out = model.encode(&refs, false)?;
        if let Some(v) = first_violation(&format!("{name} pooled"), &out.pooled, &pooled) {
            return Err(v);
        }
        // The layer-by-layer trace, sequence 0 alone: localizes a failure to a layer.
        let t = model.encode(&refs[..1], true)?;
        assert_eq!(t.trace.len(), cfg.layers as usize + 1, "{name}: trace stages");
        let stage = lens[0] as usize * cfg.hidden as usize;
        for (s, got) in t.trace.iter().enumerate() {
            let label = if s == 0 {
                format!("{name} embeddings")
            } else {
                format!("{name} layer {}", s - 1)
            };
            if let Some(v) = first_violation(&label, got, &trace[s * stage..(s + 1) * stage]) {
                return Err(v);
            }
        }
        // The single-sequence run pools to the same vector as in the batch.
        if let Some(v) = first_violation(&format!("{name} pooled (alone)"), &t.pooled, &pooled[..vocab]) {
            return Err(v);
        }
        Ok(())
    })
}

#[test]
fn bert_matches_transformers() {
    check(BertFamily::Bert, "bert", None).unwrap();
}

#[test]
fn distilbert_matches_transformers() {
    check(BertFamily::DistilBert, "distilbert", None).unwrap();
}

/// Rewrite `model.safetensors` of `name` with the two tensors' data exchanged
/// (same shape), as safetensors bytes.
fn with_swapped(name: &str, a: &str, b: &str) -> SafeTensors {
    let st = SafeTensors::open(&dir(name).join("model.safetensors")).unwrap();
    let mut names: Vec<String> = st.names().map(str::to_string).collect();
    names.sort();
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let src = if n == a {
            b
        } else if n == b {
            a
        } else {
            n
        };
        let (shape, vals) = st.read_f32(src).unwrap();
        let start = data.len();
        for v in &vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        if i > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        header.push_str(&format!(
            "\"{n}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            dims.join(","),
            data.len()
        ));
    }
    header.push('}');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    SafeTensors::from_bytes("swapped", bytes).unwrap()
}

/// The comparison has teeth: exchanging two layers' FFN output weights (same
/// shape, so it still loads) must fail it.
#[test]
fn swapping_two_layers_weights_fails_the_comparison() {
    let cases = [
        (
            BertFamily::Bert,
            "bert",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.1.output.dense.weight",
        ),
        (
            BertFamily::DistilBert,
            "distilbert",
            "distilbert.transformer.layer.0.ffn.lin2.weight",
            "distilbert.transformer.layer.1.ffn.lin2.weight",
        ),
    ];
    for (family, name, a, b) in cases {
        let swapped = with_swapped(name, a, b);
        let err = check(family, name, Some(swapped)).expect_err(&format!("{name}: swapped layers still matched"));
        assert!(err.contains("got"), "{name}: unexpected failure: {err}");
    }
}
