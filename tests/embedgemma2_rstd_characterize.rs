//! EmbeddingGemma 2 outputs that pass through the shared qk-norm row kernel
//! (`qwen35_attn_qk_norm_rope*`), as bits: the tiny checkpoint's per-layer
//! residual stream (trace) and embeddings, at both GEMM operand modes.
//!
//! `EG2_BITS_OUT=<file>` records them; `EG2_BITS_BASE=<file>` prints the max
//! ulp change per tensor against an earlier recording. Without either it only
//! checks that every value is finite. It exists to measure what moving the
//! kernel's `rsqrt` to `precise::rsqrt` does to EmbeddingGemma 2.

mod common;

use std::fs;

use common::embedgemma2::{open, safetensors_bytes, weights_for, CONFIG, PREFIX};
use common::{with_gpu, SplitMix};
use tessl::embedgemma2::{EmbedGemma2Config, EmbedGemma2Model};
use tessl::gemm::GemmOperands;

fn ulp_dist(a: f32, b: f32) -> u64 {
    let key = |v: f32| -> i64 {
        let b = v.to_bits();
        if b >> 31 == 1 {
            -i64::from(b & 0x7fff_ffff)
        } else {
            i64::from(b)
        }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn characterize_embedgemma2_through_the_qk_norm() {
    let cfg = EmbedGemma2Config::from_config_json(CONFIG).unwrap();
    let mut r = SplitMix::new(77);
    let seqs: Vec<Vec<u32>> = [9usize, 4, 1, 7]
        .iter()
        .map(|&n| (0..n).map(|_| r.range(0, cfg.vocab as usize - 1) as u32).collect())
        .collect();
    let h = cfg.hidden as usize;
    let mut segs: Vec<(String, Vec<f32>)> = Vec::new();
    for operands in [GemmOperands::ExactF32, GemmOperands::Bf16] {
        let w = weights_for(&cfg, 2026, operands);
        let st = open("ok", safetensors_bytes(&w, &[("vision_tower.unused.weight", &[3])]));
        with_gpu(|rt| {
            let model = EmbedGemma2Model::load_with(rt, &st, PREFIX, cfg.clone(), operands).unwrap();
            let out = model.encode(&[&seqs[0]], None, true).unwrap();
            for (i, t) in out.trace.iter().enumerate() {
                segs.push((format!("{operands:?} trace {i}"), t[..seqs[0].len() * h].to_vec()));
            }
            for (n, ids) in seqs.iter().enumerate() {
                let e = model.encode(&[ids], None, false).unwrap().embeddings;
                segs.push((format!("{operands:?} embedding {n}"), e));
            }
        });
    }
    for (name, v) in &segs {
        assert!(v.iter().all(|f| f.is_finite()), "{name}: non-finite value");
    }
    let bytes: Vec<u8> = segs
        .iter()
        .flat_map(|(_, v)| v.iter().flat_map(|f| f.to_bits().to_le_bytes()))
        .collect();
    if let Ok(path) = std::env::var("EG2_BITS_OUT") {
        fs::write(&path, &bytes).unwrap();
        eprintln!("recorded {} bytes to {path}", bytes.len());
    }
    if let Ok(path) = std::env::var("EG2_BITS_BASE") {
        let base = fs::read(&path).unwrap();
        assert_eq!(base.len(), bytes.len(), "baseline {path} has a different shape");
        let mut at = 0usize;
        for (name, v) in &segs {
            let (mut max, mut differ) = (0u64, 0usize);
            for (i, f) in v.iter().enumerate() {
                let o = u32::from_le_bytes(base[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                let dist = ulp_dist(f32::from_bits(o), *f);
                max = max.max(dist);
                differ += usize::from(dist != 0);
            }
            at += 4 * v.len();
            eprintln!("ULP {name}: max {max} ulp, {differ}/{} elements differ", v.len());
        }
    }
}
