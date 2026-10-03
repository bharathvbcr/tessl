//! Causal flash attention at head dimension 64.
//!
//! The row kernel is the same `ROWS_KERNEL` macro as 128/256/512. D=64 is
//! compiled only at R=8, SGT=8, where `D % (4*R) == 0`.

mod common;

use common::{buf, seeded, with_gpu};
use tessl::nn::{self, AttnDims};

const D: usize = 64;
const T: usize = 4;

/// f64 causal SDPA. Position `t` attends to keys `0..=t`. Scale is `1/sqrt(D)`.
fn causal_reference(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (D as f64).sqrt();
    let mut out = vec![0.0f32; T * D];
    for t_q in 0..T {
        let mut scores = [f64::NEG_INFINITY; T];
        for t_k in 0..=t_q {
            let mut dot = 0.0f64;
            for x in 0..D {
                dot += q[t_q * D + x] as f64 * k[t_k * D + x] as f64;
            }
            scores[t_k] = dot * scale;
        }
        let m = scores[..t_q + 1].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0.0f64;
        let mut acc = vec![0.0f64; D];
        for t_k in 0..=t_q {
            let p = (scores[t_k] - m).exp();
            l += p;
            for x in 0..D {
                acc[x] += p * v[t_k * D + x] as f64;
            }
        }
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for x in 0..D {
            out[t_q * D + x] = (acc[x] * inv) as f32;
        }
    }
    out
}

fn pattern(seed: f32) -> Vec<f32> {
    (0..T * D)
        .map(|i| {
            let t = (i / D) as f32;
            let d = (i % D) as f32;
            ((d * 0.17 + t * 0.31 + seed).sin()) * 0.25
        })
        .collect()
}

fn run(rt: &std::sync::Arc<tessl::GpuRuntime>, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let qb = buf(rt, q);
    let kb = buf(rt, k);
    let vb = buf(rt, v);
    let o = seeded(rt, T * D, -6.5e28);
    let tkv = common::buf_u32(rt, &[T as u32]);
    let zero = common::buf_u32(rt, &[0]);
    let dims = AttnDims {
        batch: 1,
        tq: T as u32,
        heads: 1,
        heads_kv: 1,
        window: 0,
        scale: 1.0 / (D as f32).sqrt(),
    };
    nn::flash_attn_rows(rt, &qb, &kb, &vb, &o, &tkv, &zero, &zero, dims, D as u32, false)
        .expect("head dim 64 causal attention");
    rt.synchronize().expect("sync");
    o.read_f32()[..T * D].to_vec()
}

#[test]
fn head_dim_64_causal_matches_cpu_and_ignores_a_future_key() {
    with_gpu(|rt| {
        let q = pattern(0.2);
        let k = pattern(1.7);
        let v = pattern(3.1);
        let got = run(rt, &q, &k, &v);
        let want = causal_reference(&q, &k, &v);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "[{i}] non-finite {g}");
            let tol = 2e-4 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "[{i}] got {g} want {w} tol {tol}");
        }

        // Key at t=3 is in the future for query 0. A value large enough to
        // dominate softmax must not move position 0.
        let mut k_future = k.clone();
        k_future[3 * D] = 1.0e6;
        let got_future = run(rt, &q, &k_future, &v);
        let pos0 = &got[..D];
        let pos0_future = &got_future[..D];
        for (i, (a, b)) in pos0.iter().zip(pos0_future).enumerate() {
            assert_eq!(a, b, "position 0 dim {i} changed after a future key");
        }
        let want_future = causal_reference(&q, &k_future, &v);
        let changed = got_future[3 * D..]
            .iter()
            .zip(&want[3 * D..])
            .any(|(g, w)| (g - w).abs() > 1e-3);
        assert!(
            changed,
            "the future key was invisible to position 3 as well, so the mask was not exercised"
        );
        for (i, (g, w)) in got_future.iter().zip(&want_future).enumerate() {
            let tol = 2e-4 * w.abs().max(1.0);
            assert!((g - w).abs() <= tol, "future-key case [{i}] got {g} want {w}");
        }

        // A head dim outside {64 at r8/g8, 128, 256, 512} keeps the old error.
        let err = nn::flash_attn_rows(
            rt,
            &buf(rt, &q),
            &buf(rt, &k),
            &buf(rt, &v),
            &seeded(rt, T * D, 0.0),
            &tkv_of(rt),
            &common::buf_u32(rt, &[0]),
            &common::buf_u32(rt, &[0]),
            AttnDims {
                batch: 1,
                tq: T as u32,
                heads: 1,
                heads_kv: 1,
                window: 0,
                scale: 1.0,
            },
            32,
            false,
        )
        .expect_err("head dim 32");
        assert!(
            err.contains("has no kernel (128, 256 or 512)"),
            "unexpected refusal: {err}"
        );
    });
}

fn tkv_of(rt: &std::sync::Arc<tessl::GpuRuntime>) -> tessl::tensor::GpuBuffer {
    common::buf_u32(rt, &[T as u32])
}
