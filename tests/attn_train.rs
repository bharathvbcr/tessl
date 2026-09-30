//! Training attention (`tessl::attn_train`): the forward with its saved
//! log-sum-exp, and the FlashAttention-2 backward.
//!
//! An f64 reference forward and backward (the backward checked against
//! central finite differences of the forward), then the kernels against it
//! within `1e-4` of the largest reference magnitude across block edges,
//! grouped heads and batch rows. The forward's output is also the inference
//! `attn_prefill`'s, bit for bit (same kernel body and geometry), and every
//! gradient is bit-identical on a rerun.

mod common;

use std::sync::Arc;

use common::{buf, buf_u32, random_f32, seeded, with_gpu};
use tessl::attn_train::{
    attn_train_backward, attn_train_forward, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace, ATTN_TRAIN_HEAD_DIM,
};
use tessl::nn::AttnDims;
use tessl::qwen35;
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.25e27;
const D: usize = ATTN_TRAIN_HEAD_DIM as usize;

struct Shape {
    b: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    scale: f64,
}

impl Shape {
    fn qi(&self, b: usize, t: usize, h: usize) -> usize {
        ((b * self.t + t) * self.hq + h) * self.d
    }
    fn ki(&self, b: usize, t: usize, h: usize) -> usize {
        ((b * self.t + t) * self.hkv + h) * self.d
    }
    fn kv_head(&self, h: usize) -> usize {
        h / (self.hq / self.hkv)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `(o, lse)` of causal grouped attention, in f64.
fn attn_fwd(s: &Shape, q: &[f64], k: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let d = s.d;
    let mut o = vec![0.0; q.len()];
    let mut lse = vec![0.0; s.b * s.hq * s.t];
    for b in 0..s.b {
        for h in 0..s.hq {
            let g = s.kv_head(h);
            for t in 0..s.t {
                let qr = &q[s.qi(b, t, h)..][..d];
                let sc: Vec<f64> = (0..=t).map(|j| s.scale * dot(qr, &k[s.ki(b, j, g)..][..d])).collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = sc.iter().map(|x| (x - m).exp()).sum();
                lse[(b * s.hq + h) * s.t + t] = m + z.ln();
                for (j, x) in sc.iter().enumerate() {
                    let p = (x - m).exp() / z;
                    for c in 0..d {
                        o[s.qi(b, t, h) + c] += p * v[s.ki(b, j, g) + c];
                    }
                }
            }
        }
    }
    (o, lse)
}

/// `(dq, dk, dv)` of [`attn_fwd`]'s `o` for the upstream `d_o`.
fn attn_bwd(s: &Shape, q: &[f64], k: &[f64], v: &[f64], d_o: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = s.d;
    let (o, lse) = attn_fwd(s, q, k, v);
    let (mut dq, mut dk, mut dv) = (vec![0.0; q.len()], vec![0.0; k.len()], vec![0.0; v.len()]);
    for b in 0..s.b {
        for h in 0..s.hq {
            let g = s.kv_head(h);
            for t in 0..s.t {
                let (qo, l) = (s.qi(b, t, h), lse[(b * s.hq + h) * s.t + t]);
                let dr = dot(&d_o[qo..][..d], &o[qo..][..d]);
                for j in 0..=t {
                    let ko = s.ki(b, j, g);
                    let p = (s.scale * dot(&q[qo..][..d], &k[ko..][..d]) - l).exp();
                    let dp = dot(&d_o[qo..][..d], &v[ko..][..d]);
                    let ds = p * (dp - dr) * s.scale;
                    for c in 0..d {
                        dv[ko + c] += p * d_o[qo + c];
                        dq[qo + c] += ds * k[ko + c];
                        dk[ko + c] += ds * q[qo + c];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn close(label: &str, got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    let mut worst = (0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g != SENTINEL, "{label}[{i}]: never written");
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let e = (f64::from(g) - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    let rel = worst.0 / peak;
    eprintln!("{label}: {rel:.2e} of max|ref|");
    assert!(
        rel <= 1e-4,
        "{label}: max err {:.3e} at {} (got {} want {}), {rel:.2e} of {peak:.3e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
    rel
}

#[test]
fn reference_backward_is_the_derivative() {
    let s = Shape {
        b: 2,
        t: 4,
        hq: 4,
        hkv: 2,
        d: 3,
        scale: 0.7,
    };
    let q = f64s(&random_f32(s.b * s.t * s.hq * s.d, 1));
    let k = f64s(&random_f32(s.b * s.t * s.hkv * s.d, 2));
    let v = f64s(&random_f32(s.b * s.t * s.hkv * s.d, 3));
    let g = f64s(&random_f32(q.len(), 4));
    let (dq, dk, dv) = attn_bwd(&s, &q, &k, &v, &g);
    let loss = |q: &[f64], k: &[f64], v: &[f64]| dot(&attn_fwd(&s, q, k, v).0, &g);
    let fd = |name: &str, x: &[f64], grad: &[f64], f: &dyn Fn(&[f64]) -> f64| {
        let h = 1e-6;
        let mut worst = 0.0f64;
        for i in 0..x.len() {
            let (mut p, mut m) = (x.to_vec(), x.to_vec());
            p[i] += h;
            m[i] -= h;
            let want = (f(&p) - f(&m)) / (2.0 * h);
            worst = worst.max((want - grad[i]).abs() / (1.0 + want.abs()));
        }
        assert!(worst < 1e-7, "{name}: worst {worst:.2e} against finite differences");
    };
    fd("dq", &q, &dq, &|x| loss(x, &k, &v));
    fd("dk", &k, &dk, &|x| loss(&q, x, &v));
    fd("dv", &v, &dv, &|x| loss(&q, &k, x));
}

fn dims(b: usize, t: usize, hq: usize, hkv: usize) -> AttnTrainDims {
    AttnTrainDims {
        batch: b as u32,
        seq: t as u32,
        q_heads: hq as u32,
        kv_heads: hkv as u32,
        scale: 1.0 / 16.0,
    }
}

fn run(rt: &Arc<GpuRuntime>, b: usize, t: usize, hq: usize, hkv: usize, seed: u64) {
    let dm = dims(b, t, hq, hkv);
    let s = Shape {
        b,
        t,
        hq,
        hkv,
        d: D,
        scale: f64::from(dm.scale),
    };
    let (nq, nkv) = (b * t * hq * D, b * t * hkv * D);
    // Scores of a few units, so rows are neither uniform nor one-hot.
    let q: Vec<f32> = random_f32(nq, seed).iter().map(|x| 3.0 * x).collect();
    let k: Vec<f32> = random_f32(nkv, seed + 1).iter().map(|x| 3.0 * x).collect();
    let v = random_f32(nkv, seed + 2);
    let d_o = random_f32(nq, seed + 3);
    let (qb, kb, vb, dob) = (buf(rt, &q), buf(rt, &k), buf(rt, &v), buf(rt, &d_o));
    let (ob, lseb) = (seeded(rt, nq, SENTINEL), seeded(rt, dm.lse_len(), SENTINEL));
    let ws = AttnTrainWorkspace::new(rt, dm).unwrap();
    attn_train_forward(rt, &dm, &qb, &kb, &vb, &ob, &lseb, &ws).unwrap();
    let (dqb, dkb, dvb) = (
        seeded(rt, nq, SENTINEL),
        seeded(rt, nkv, SENTINEL),
        seeded(rt, nkv, SENTINEL),
    );
    let grads = AttnTrainGrads {
        dq: &dqb,
        dk: &dkb,
        dv: &dvb,
    };
    attn_train_backward(rt, &dm, &qb, &kb, &vb, &ob, &lseb, &dob, &grads, &ws).unwrap();
    rt.synchronize().unwrap();

    let (q64, k64, v64) = (f64s(&q), f64s(&k), f64s(&v));
    let (want_o, want_lse) = attn_fwd(&s, &q64, &k64, &v64);
    let (want_dq, want_dk, want_dv) = attn_bwd(&s, &q64, &k64, &v64, &f64s(&d_o));
    let label = format!("b={b} t={t} hq={hq} hkv={hkv}");
    let o = ob.read_f32();
    close(&format!("{label} o"), &o, &want_o);
    close(&format!("{label} lse"), &lseb.read_f32(), &want_lse);
    let first = [dqb.read_f32(), dkb.read_f32(), dvb.read_f32()];
    close(&format!("{label} dq"), &first[0], &want_dq);
    close(&format!("{label} dk"), &first[1], &want_dk);
    close(&format!("{label} dv"), &first[2], &want_dv);

    // The inference forward at its default geometry gives the same bits.
    let prefill = seeded(rt, nq, SENTINEL);
    let (tkv, zero) = (buf_u32(rt, &[t as u32]), buf_u32(rt, &[0]));
    let ad = AttnDims {
        batch: b as u32,
        tq: t as u32,
        heads: hq as u32,
        heads_kv: hkv as u32,
        window: 0,
        scale: dm.scale,
    };
    qwen35::attn_prefill(rt, &qb, &kb, &vb, &prefill, &tkv, &zero, &zero, ad, false).unwrap();
    // A rerun of the backward gives the same bits.
    attn_train_backward(rt, &dm, &qb, &kb, &vb, &ob, &lseb, &dob, &grads, &ws).unwrap();
    rt.synchronize().unwrap();
    assert!(
        prefill
            .read_f32()
            .iter()
            .zip(&o)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: o differs from attn_prefill"
    );
    for (name, (got, was)) in ["dq", "dk", "dv"]
        .iter()
        .zip([dqb.read_f32(), dkb.read_f32(), dvb.read_f32()].iter().zip(&first))
    {
        assert!(
            got.iter().zip(was).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label}: {name} changed on a rerun"
        );
    }
}

#[test]
fn training_attention_matches_the_reference() {
    with_gpu(|rt| {
        run(rt, 1, 1, 1, 1, 10); // one token: P = 1, dq = dk = 0
        run(rt, 1, 31, 2, 1, 11); // one partial block
        run(rt, 1, 64, 2, 2, 12); // exactly two blocks
        run(rt, 2, 33, 4, 2, 13); // a one-row block, two batch rows, grouped heads
        run(rt, 1, 100, 8, 2, 14); // the 2B's heads
        run(rt, 2, 130, 2, 1, 15);
    });
}

#[test]
fn training_attention_refuses_bad_calls() {
    with_gpu(|rt| {
        let e = |r: Result<(), String>, needle: &str| {
            let m = r.expect_err(needle);
            assert!(m.contains(needle), "{m:?} lacks {needle:?}");
        };
        e(
            AttnTrainWorkspace::new(rt, dims(1, 4, 3, 2)).map(|_| ()),
            "q_heads must be a non-zero multiple of kv_heads",
        );
        e(
            AttnTrainWorkspace::new(
                rt,
                AttnTrainDims {
                    scale: 0.0,
                    ..dims(1, 4, 2, 1)
                },
            )
            .map(|_| ()),
            "scale",
        );
        let dm = dims(1, 4, 2, 1);
        let ws = AttnTrainWorkspace::new(rt, dm).unwrap();
        let (nq, nkv) = (4 * 2 * D, 4 * D);
        let (q, k, v, o) = (
            buf(rt, &vec![0.0; nq]),
            buf(rt, &vec![0.0; nkv]),
            buf(rt, &vec![0.0; nkv]),
            buf(rt, &vec![0.0; nq]),
        );
        let lse = buf(rt, &[0.0; 8]);
        attn_train_forward(rt, &dm, &q, &k, &v, &o, &lse, &ws).expect("a valid call");
        e(
            attn_train_forward(rt, &dims(1, 5, 2, 1), &q, &k, &v, &o, &lse, &ws),
            "the workspace was made for",
        );
        e(
            attn_train_forward(rt, &dm, &q, &k, &v, &o, &buf(rt, &[0.0; 7]), &ws),
            "lse",
        );
        e(attn_train_forward(rt, &dm, &q, &k, &v, &q, &lse, &ws), "o");
        let (dq, dk, dv) = (
            buf(rt, &vec![0.0; nq]),
            buf(rt, &vec![0.0; nkv]),
            buf(rt, &vec![0.0; nkv]),
        );
        let grads = AttnTrainGrads {
            dq: &dq,
            dk: &dk,
            dv: &dv,
        };
        attn_train_backward(rt, &dm, &q, &k, &v, &o, &lse, &o, &grads, &ws).expect("a valid call");
        e(
            attn_train_backward(
                rt,
                &dm,
                &q,
                &k,
                &v,
                &o,
                &lse,
                &o,
                &AttnTrainGrads { dq: &q, ..grads },
                &ws,
            ),
            "dq",
        );
        e(
            attn_train_backward(
                rt,
                &dm,
                &q,
                &k,
                &v,
                &o,
                &lse,
                &o,
                &AttnTrainGrads { dk: &dq, ..grads },
                &ws,
            ),
            "dk",
        );
        e(
            attn_train_backward(
                rt,
                &dm,
                &q,
                &k,
                &v,
                &o,
                &lse,
                &o,
                &AttnTrainGrads {
                    dv: &buf(rt, &[0.0; 8]),
                    ..grads
                },
                &ws,
            ),
            "dv",
        );
    });
}

/// The training attention over randomly drawn shapes (batch rows, lengths
/// across block edges, head grouping), each checked as `run` checks it.
/// `TESSL_FUZZ_ITERS` / `TESSL_FUZZ_SEED` scale it up.
#[test]
fn randomized_shapes_stress() {
    let (iters, seed) = common::fuzz_plan(4);
    with_gpu(|rt| {
        for it in 0..iters {
            let s = seed.wrapping_mul(1_000_003).wrapping_add(it as u64);
            let mut r = common::SplitMix::new(s);
            let (hkv, group) = (r.range(1, 2), r.range(1, 4));
            let (b, t) = (r.range(1, 2), r.range(1, 140));
            eprintln!(
                "stress iteration {it} (seed {s}): b={b} t={t} hq={} hkv={hkv}",
                hkv * group
            );
            run(rt, b, t, hkv * group, hkv, s);
        }
    });
}
