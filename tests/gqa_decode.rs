//! Decode-only grouped-query attention. Query length is 1.
//!
//! Every head dimension the entry point accepts is checked against a CPU
//! reference. A dimension outside that set is an error that names it and
//! does not dispatch. Prefill kernels are not this path.

mod common;

use common::{buf, with_gpu};
use tessl::nn;

fn wave(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.017 + seed).sin() * 0.05).collect()
}

fn gqa_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    kv_len: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let group = q_heads / kv_heads;
    let mut o = vec![0.0f32; batch * q_heads * d];
    for b in 0..batch {
        for h in 0..q_heads {
            let kv_h = h / group;
            let qs = (b * q_heads + h) * d;
            let mut dots = vec![0.0f32; kv_len];
            let mut m = f32::NEG_INFINITY;
            for t in 0..kv_len {
                let ks = ((b * kv_len + t) * kv_heads + kv_h) * d;
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q[qs + i] * k[ks + i];
                }
                dots[t] = dot * scale;
                m = m.max(dots[t]);
            }
            let mut se = 0.0f32;
            let mut w = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                w[t] = (dots[t] - m).exp();
                se += w[t];
            }
            for i in 0..d {
                let mut acc = 0.0f32;
                for t in 0..kv_len {
                    let vs = ((b * kv_len + t) * kv_heads + kv_h) * d;
                    acc += (w[t] / se) * v[vs + i];
                }
                o[qs + i] = acc;
            }
        }
    }
    o
}

fn close(got: f32, want: f32) -> bool {
    let abs = (got - want).abs();
    abs <= 1.0e-5 || abs <= 2.0e-4 * want.abs()
}

#[test]
fn every_accepted_head_dim_matches_the_cpu_reference() {
    with_gpu(|rt| {
        for d in [32u32, 64, 128, 256, 512] {
            assert!(nn::ATTENTION_HEAD_DIMS.contains(&d), "contract dropped {d}");
        }
        let (batch, q_heads, kv_heads, kv_len) = (2u32, 4, 2, 5);
        for d in nn::ATTENTION_HEAD_DIMS {
            let dsz = d as usize;
            let q = wave((batch * q_heads) as usize * dsz, 0.2);
            let k = wave((batch * kv_len * kv_heads) as usize * dsz, 0.7);
            let v = wave((batch * kv_len * kv_heads) as usize * dsz, 1.3);
            let qb = buf(rt, &q);
            let kb = buf(rt, &k);
            let vb = buf(rt, &v);
            let o1 = buf(rt, &vec![0.0f32; q.len()]);
            let o2 = buf(rt, &vec![0.0f32; q.len()]);
            nn::gqa_decode(rt, &qb, &kb, &vb, &o1, batch, q_heads, kv_heads, kv_len, d).expect("gqa");
            nn::gqa_decode(rt, &qb, &kb, &vb, &o2, batch, q_heads, kv_heads, kv_len, d).expect("gqa rerun");
            let g1 = o1.read_f32();
            let g2 = o2.read_f32();
            for (i, (a, b)) in g1.iter().zip(&g2).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "head {d} rerun [{i}]");
            }
            let want = gqa_oracle(
                &q,
                &k,
                &v,
                batch as usize,
                q_heads as usize,
                kv_heads as usize,
                kv_len as usize,
                dsz,
            );
            for (i, (got, w)) in g1.iter().zip(&want).enumerate() {
                assert!(close(*got, *w), "head {d} [{i}]: got {got} want {w}");
            }
            assert!(rt.take_dispatch_count() > 0, "head {d} did not dispatch");
        }
    });
}

#[test]
fn an_unsupported_head_dim_is_named_and_not_dispatched() {
    with_gpu(|rt| {
        let b = rt.alloc_buffer(64).unwrap();
        rt.take_dispatch_count();
        let err = nn::gqa_decode(rt, &b, &b, &b, &b, 1, 1, 1, 1, 96).expect_err("dim 96");
        assert!(err.contains("96"), "{err}");
        assert!(err.contains("head dim"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::gqa_decode(rt, &b, &b, &b, &b, 1, 1, 1, 0, 64).expect_err("kv_len 0");
        assert!(err.contains("non-zero"), "{err}");
        let err = nn::gqa_decode(rt, &b, &b, &b, &b, 1, 3, 2, 1, 64).expect_err("heads");
        assert!(err.contains("3"), "{err}");
        assert!(err.contains("not divisible"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
