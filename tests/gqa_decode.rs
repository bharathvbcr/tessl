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

#[allow(clippy::too_many_arguments)]
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
            for (t, slot) in dots.iter_mut().enumerate() {
                let ks = ((b * kv_len + t) * kv_heads + kv_h) * d;
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q[qs + i] * k[ks + i];
                }
                *slot = dot * scale;
                m = m.max(*slot);
            }
            let mut se = 0.0f32;
            let mut w = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                w[t] = (dots[t] - m).exp();
                se += w[t];
            }
            for i in 0..d {
                let mut acc = 0.0f32;
                for (t, &wt) in w.iter().enumerate() {
                    let vs = ((b * kv_len + t) * kv_heads + kv_h) * d;
                    acc += (wt / se) * v[vs + i];
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

#[test]
fn extreme_finite_scores_match_the_cpu_reference() {
    with_gpu(|rt| {
        // batch 1, two q heads on two kv heads (group 1), head_dim 64, four keys.
        let (q_heads, kv_heads, kv_len, d) = (2usize, 2usize, 4usize, 64usize);
        let mut q = vec![0.0f32; q_heads * d];
        let mut k = vec![0.0f32; kv_len * kv_heads * d];
        // Head 0: every score very negative. q = 1e18, k_t = -c_t * 1e18, so
        // dot = -64e36 * c_t and the scaled score (x 0.125) is -8e36 * c_t,
        // between -8e36 and -2e37.
        // Head 1: mixed large positive and negative scores, c_t in [-1, 1].
        let neg_c = [1.0f32, 1.5, 2.5, 2.0];
        let mix_c = [-1.0f32, 0.5, 1.0, -0.25];
        for i in 0..d {
            q[i] = 1.0e18;
            q[d + i] = 1.0e18;
        }
        for t in 0..kv_len {
            for i in 0..d {
                k[((t * kv_heads) * d) + i] = -neg_c[t] * 1.0e18;
                k[((t * kv_heads + 1) * d) + i] = mix_c[t] * 1.0e18;
            }
        }
        // Finiteness check, in f64 so it cannot itself overflow: the largest
        // |dot| (x scale) must be below f32::MAX, and every partial sum is
        // bounded by the final one because all terms in a dot share a sign.
        let scale = 1.0f64 / (d as f64).sqrt();
        let mut worst = 0.0f64;
        for h in 0..q_heads {
            for t in 0..kv_len {
                let dot: f64 = (0..d)
                    .map(|i| q[h * d + i] as f64 * k[(t * kv_heads + h) * d + i] as f64)
                    .sum();
                worst = worst.max(dot.abs());
                assert!(
                    (dot * scale).abs() < f32::MAX as f64,
                    "score head {h} key {t} overflows f32"
                );
            }
        }
        assert!(worst < f32::MAX as f64, "a raw dot overflows f32: {worst:e}");
        let v = wave(kv_len * kv_heads * d, 1.3);
        let want = gqa_oracle(&q, &k, &v, 1, q_heads, kv_heads, kv_len, d);
        assert!(want.iter().all(|x| x.is_finite()), "oracle output is not finite");

        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let ob = buf(rt, &vec![0.0f32; q.len()]);
        nn::gqa_decode(
            rt,
            &qb,
            &kb,
            &vb,
            &ob,
            1,
            q_heads as u32,
            kv_heads as u32,
            kv_len as u32,
            d as u32,
        )
        .expect("gqa");
        let got = ob.read_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(close(*g, *w), "[{i}]: got {g} want {w}");
        }
    });
}

#[test]
fn a_non_finite_head_does_not_leak_into_other_heads() {
    with_gpu(|rt| {
        let (q_heads, kv_heads, kv_len, d) = (4usize, 2usize, 3usize, 64usize);
        let q = wave(q_heads * d, 0.2);
        let mut k = wave(kv_len * kv_heads * d, 0.7);
        let v = wave(kv_len * kv_heads * d, 1.3);
        // kv head 0, key 1, element 0.
        k[(kv_heads) * d] = f32::INFINITY;
        let want = gqa_oracle(&q, &k, &v, 1, q_heads, kv_heads, kv_len, d);

        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let ob = buf(rt, &vec![0.0f32; q.len()]);
        nn::gqa_decode(
            rt,
            &qb,
            &kb,
            &vb,
            &ob,
            1,
            q_heads as u32,
            kv_heads as u32,
            kv_len as u32,
            d as u32,
        )
        .expect("gqa");
        let got = ob.read_f32();
        // Heads 0 and 1 (kv head 0) are not asserted: the kernel is built with fast math, so its non-finite behaviour is unspecified.
        for h in 2..q_heads {
            for i in 0..d {
                let (g, w) = (got[h * d + i], want[h * d + i]);
                assert!(close(g, w), "head {h} [{i}]: got {g} want {w}");
            }
        }
    });
}
