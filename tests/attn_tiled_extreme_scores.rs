//! The tiled FlashAttention-2 body (`attn_tiled.h`) at score magnitudes far from
//! zero, in both directions.
//!
//! The fault-injection run (docs/verification.md, "Fault injection") found that
//! seeding the running max at 0 instead of -FLT_MAX passed every tiled test:
//! with scores near zero the seed changes nothing, because the softmax is
//! shift-invariant. It is *not* an equivalent mutant. A row whose scores are all
//! below about -88 gets `exp(s - 0) == 0` for every key, so `l` stays 0, the
//! "has seen a key" flag never sets, and the row comes out as zeros where the
//! correct answer is a weighted mean of V. `fast_paths_survive_extreme_score_
//! magnitudes` covers the row-parallel and decode kernels only; this covers the
//! qwen35 tiled kernels, which share the body with the encoder.

mod common;

use common::{buf, buf_u32, random_f32, seeded, with_gpu};
use tessl::nn::AttnDims;
use tessl::qwen35;

const UNWRITTEN: f32 = -6.5e28;
const D: usize = 256;

/// f64 causal reference with a max-subtracted softmax.
fn reference(q: &[f32], k: &[f32], v: &[f32], t: usize, h: usize, hkv: usize, scale: f32) -> Vec<f32> {
    let group = h / hkv;
    let mut out = vec![0.0f32; t * h * D];
    for tq in 0..t {
        for hi in 0..h {
            let hk = hi / group;
            let qb = (tq * h + hi) * D;
            let scores: Vec<f64> = (0..=tq)
                .map(|tk| {
                    let kb = (tk * hkv + hk) * D;
                    (0..D).map(|x| q[qb + x] as f64 * k[kb + x] as f64).sum::<f64>() * scale as f64
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f64 = ps.iter().sum();
            for x in 0..D {
                let acc: f64 = ps
                    .iter()
                    .enumerate()
                    .map(|(tk, p)| p * v[(tk * hkv + hk) * D + x] as f64)
                    .sum();
                out[qb + x] = (acc / l) as f32;
            }
        }
    }
    out
}

#[test]
fn the_tiled_kernels_survive_scores_far_below_and_above_zero() {
    with_gpu(|rt| {
        let (t, h, hkv) = (40usize, 4usize, 2usize);
        // (label, q scale, k scale, softmax scale). Scores are q.k * scale with
        // q, k of one sign, so every score has the sign of the k scale.
        let cases: &[(&str, f32, f32, f32)] = &[
            ("all scores far below -88", 1.0, -1.0, 2.5),
            ("all scores far above +88", 1.0, 1.0, 2.5),
        ];
        for &(label, qs, ks, scale) in cases {
            let q: Vec<f32> = random_f32(t * h * D, 0xA001).iter().map(|x| x.abs() * qs).collect();
            let k: Vec<f32> = random_f32(t * hkv * D, 0xA002).iter().map(|x| x.abs() * ks).collect();
            let v = random_f32(t * hkv * D, 0xA003);
            let want = reference(&q, &k, &v, t, h, hkv, scale);
            let first = (0..D).map(|x| q[x] as f64 * k[x] as f64).sum::<f64>() * scale as f64;
            assert!(first.abs() > 100.0, "{label}: the case is not extreme enough ({first})");
            let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
            let (tkv, zero) = (buf_u32(rt, &[t as u32]), buf_u32(rt, &[0]));
            let dims = AttnDims {
                batch: 1,
                tq: t as u32,
                heads: h as u32,
                heads_kv: hkv as u32,
                window: 0,
                scale,
            };
            for tile in qwen35::AttnTile::ALL {
                let ob = seeded(rt, t * h * D, UNWRITTEN);
                qwen35::attn_prefill_with_tile(rt, &qb, &kb, &vb, &ob, &tkv, &zero, &zero, dims, false, tile).unwrap();
                rt.synchronize().unwrap();
                let got = ob.read_f32();
                // A score is a K=256 f32 dot product at |s| ~ 160, so its absolute error is a
                // few 1e-3, and exp amplifies that into a weight: compare against the
                // output scale, not each element. A row that comes out as zeros misses by
                // the whole scale.
                let span = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
                for (i, w) in want.iter().enumerate() {
                    let tol = 2e-3 * w.abs() + 5e-3 * span;
                    assert!(
                        got[i].is_finite() && (got[i] - w).abs() <= tol,
                        "{label}, tile {tile:?}: [{i}] got {} want {w}",
                        got[i]
                    );
                }
            }
        }
    });
}
