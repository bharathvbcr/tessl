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

use common::{buf, buf_u32, random_f32, reduction_bound, seeded, with_gpu};
use tessl::attn_train::{
    attn_train_backward, attn_train_forward, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace, ATTN_TRAIN_HEAD_DIM,
};
use tessl::dispatch::{dispatch_2d_tg, set_f32, set_gpu_buf, set_u32};
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
    attn_fwd_kv_off(s, q, k, v, 0)
}

/// [`attn_fwd`] with key `j` at position `kv_off + j`: row `t` sees keys
/// `j <= t - kv_off`. A row that sees none keeps `o = 0` and gets `lse = NaN`,
/// which no kernel value equals, so a caller must check those rows itself.
fn attn_fwd_kv_off(s: &Shape, q: &[f64], k: &[f64], v: &[f64], kv_off: usize) -> (Vec<f64>, Vec<f64>) {
    let d = s.d;
    let mut o = vec![0.0; q.len()];
    let mut lse = vec![0.0; s.b * s.hq * s.t];
    for b in 0..s.b {
        for h in 0..s.hq {
            let g = s.kv_head(h);
            for t in 0..s.t {
                if t < kv_off {
                    lse[(b * s.hq + h) * s.t + t] = f64::NAN;
                    continue;
                }
                let qr = &q[s.qi(b, t, h)..][..d];
                let sc: Vec<f64> = (0..=t - kv_off)
                    .map(|j| s.scale * dot(qr, &k[s.ki(b, j, g)..][..d]))
                    .collect();
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

/// The log-sum-exp the forward saves for a row with no key: `f32::MAX`, not
/// `+inf`, because the kernels compile with fast math, which may assume no
/// value is infinite (`qwen35_attn_tiled.metal`).
const EMPTY_ROW_LSE: f32 = f32::MAX;

/// Keys of [`EmptyRowCase`] start at this position, so query rows `0..5`
/// have no key.
const EMPTY_KV_OFF: usize = 5;

/// One batch row, T = 40 (one full query block and a 24-row padded tail),
/// two query heads on one KV head, scores of a few units.
struct EmptyRowCase {
    s: Shape,
    dm: AttnTrainDims,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qb: tessl::GpuBuffer,
    kb: tessl::GpuBuffer,
    vb: tessl::GpuBuffer,
}

impl EmptyRowCase {
    fn new(rt: &Arc<GpuRuntime>) -> Self {
        let (b, t, hq, hkv) = (1usize, 40usize, 2usize, 1usize);
        let dm = dims(b, t, hq, hkv);
        let s = Shape {
            b,
            t,
            hq,
            hkv,
            d: D,
            scale: f64::from(dm.scale),
        };
        let q: Vec<f32> = random_f32(b * t * hq * D, 70).iter().map(|x| 3.0 * x).collect();
        let k: Vec<f32> = random_f32(b * t * hkv * D, 71).iter().map(|x| 3.0 * x).collect();
        let v = random_f32(b * t * hkv * D, 72);
        let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
        Self {
            s,
            dm,
            q,
            k,
            v,
            qb,
            kb,
            vb,
        }
    }

    /// The training forward's kernel with keys from [`EMPTY_KV_OFF`]:
    /// `(o, lse)`. `attn_train_forward` always passes zero offsets, under
    /// which every causal row sees key 0, so a direct dispatch is the only
    /// way to make its empty-row log-sum-exp reachable.
    fn forward_kv_off(&self, rt: &Arc<GpuRuntime>) -> (Vec<f32>, Vec<f32>) {
        let Shape { b, t, hq, hkv, .. } = self.s;
        let ob = seeded(rt, self.q.len(), SENTINEL);
        let lseb = seeded(rt, self.dm.lse_len(), SENTINEL);
        let (tkv, zero, kv_off) = (
            buf_u32(rt, &[t as u32]),
            buf_u32(rt, &[0]),
            buf_u32(rt, &[EMPTY_KV_OFF as u32]),
        );
        let p = rt.pipeline("qwen35_attn_tiled_lse_h256_q32_k32_sg4").unwrap();
        dispatch_2d_tg(rt, &p, t.div_ceil(32), b * hq, 4 * 32, |bnd| {
            set_gpu_buf(bnd, &self.qb, 0);
            set_gpu_buf(bnd, &self.kb, 1);
            set_gpu_buf(bnd, &self.vb, 2);
            set_gpu_buf(bnd, &ob, 3);
            set_u32(bnd, b as u32, 4);
            set_u32(bnd, t as u32, 5);
            set_gpu_buf(bnd, &tkv, 6);
            set_u32(bnd, hq as u32, 7);
            set_u32(bnd, hkv as u32, 8);
            set_u32(bnd, 0, 9);
            set_f32(bnd, self.dm.scale, 10);
            set_gpu_buf(bnd, &zero, 11);
            set_gpu_buf(bnd, &kv_off, 12);
            set_u32(bnd, 0, 13);
            set_u32(bnd, t as u32, 14);
            set_gpu_buf(bnd, &lseb, 15);
        })
        .unwrap();
        rt.synchronize().unwrap();
        (ob.read_f32(), lseb.read_f32())
    }
}

/// The training forward on rows with no key: exact zeros out and
/// [`EMPTY_ROW_LSE`] saved, beside rows with keys and a padded tail that
/// match the f64 reference.
#[test]
fn training_forward_saves_a_finite_lse_for_a_row_with_no_key() {
    with_gpu(|rt| {
        let c = EmptyRowCase::new(rt);
        let (s, t) = (&c.s, c.s.t);
        let (o, lse) = c.forward_kv_off(rt);
        let (want_o, want_lse) = attn_fwd_kv_off(s, &f64s(&c.q), &f64s(&c.k), &f64s(&c.v), EMPTY_KV_OFF);
        for h in 0..s.hq {
            for row in 0..EMPTY_KV_OFF {
                let at = s.qi(0, row, h);
                assert!(
                    o[at..][..D].iter().all(|&x| x.to_bits() == 0),
                    "head {h} row {row} has no key: o must be +0.0, got {:?}",
                    &o[at..][..4]
                );
                let l = lse[h * t + row];
                assert_eq!(
                    l.to_bits(),
                    EMPTY_ROW_LSE.to_bits(),
                    "head {h} row {row} has no key: lse must be f32::MAX, got {l}"
                );
            }
        }
        close("kv_off=5 o", &o, &want_o);
        let (live_lse, live_want): (Vec<f32>, Vec<f64>) = lse
            .iter()
            .zip(&want_lse)
            .filter(|(_, w)| !w.is_nan())
            .map(|(&g, &w)| (g, w))
            .unzip();
        close("kv_off=5 lse", &live_lse, &live_want);
    });
}

/// The training backward on a padded tail and a row saved as empty.
///
/// T = 40, so the backward's dead tail rows (`tid >= nq`) take part in every
/// block walk. One live row's saved log-sum-exp is replaced by the value the
/// forward writes for a row with no key (read back from the kernel, not
/// assumed). That row must then contribute nothing — dq exactly zero there,
/// dk and dv equal to the reference with that row's upstream gradient zeroed —
/// and no gradient may be non-finite.
#[test]
fn training_backward_gives_a_row_saved_as_empty_no_gradient() {
    with_gpu(|rt| {
        let c = EmptyRowCase::new(rt);
        let (s, dm, t) = (&c.s, c.dm, c.s.t);
        let (nq, nkv) = (c.q.len(), c.k.len());
        let empty_lse = c.forward_kv_off(rt).1[0];
        let ws = AttnTrainWorkspace::new(rt, dm).unwrap();
        let (ob, lseb) = (seeded(rt, nq, SENTINEL), seeded(rt, dm.lse_len(), SENTINEL));
        attn_train_forward(rt, &dm, &c.qb, &c.kb, &c.vb, &ob, &lseb, &ws).unwrap();
        rt.synchronize().unwrap();
        let (dead_h, dead_t) = (1usize, 7usize);
        let mut saved = lseb.read_f32();
        saved[dead_h * t + dead_t] = empty_lse;
        lseb.write_f32(&saved);
        let mut d_o = random_f32(nq, 73);
        let dob = buf(rt, &d_o);
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
        attn_train_backward(rt, &dm, &c.qb, &c.kb, &c.vb, &ob, &lseb, &dob, &grads, &ws).unwrap();
        rt.synchronize().unwrap();
        let dq = dqb.read_f32();
        let at = s.qi(0, dead_t, dead_h);
        assert!(
            dq[at..][..D].iter().all(|&x| x == 0.0),
            "a row saved as empty must get no dq, got {:?}",
            &dq[at..][..4]
        );
        // The reference for "row contributes nothing" is that row's upstream
        // gradient set to zero: its P·(dP - D) and Pᵀ dO terms both vanish.
        d_o[at..][..D].fill(0.0);
        let (want_dq, want_dk, want_dv) = attn_bwd(s, &f64s(&c.q), &f64s(&c.k), &f64s(&c.v), &f64s(&d_o));
        close("empty-row dq", &dq, &want_dq);
        close("empty-row dk", &dkb.read_f32(), &want_dk);
        close("empty-row dv", &dvb.read_f32(), &want_dv);
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

/// Runs the training forward and backward once and returns `(o, dq, dk, dv)`.
fn fwd_bwd(
    rt: &Arc<GpuRuntime>,
    dm: &AttnTrainDims,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_o: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (nq, nkv) = (q.len(), k.len());
    let (qb, kb, vb, dob) = (buf(rt, q), buf(rt, k), buf(rt, v), buf(rt, d_o));
    let (ob, lseb) = (seeded(rt, nq, SENTINEL), seeded(rt, dm.lse_len(), SENTINEL));
    let ws = AttnTrainWorkspace::new(rt, *dm).unwrap();
    attn_train_forward(rt, dm, &qb, &kb, &vb, &ob, &lseb, &ws).unwrap();
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
    attn_train_backward(rt, dm, &qb, &kb, &vb, &ob, &lseb, &dob, &grads, &ws).unwrap();
    rt.synchronize().unwrap();
    (ob.read_f32(), dqb.read_f32(), dkb.read_f32(), dvb.read_f32())
}

/// GProj's "one-key row" test (arXiv:2609.34272, Appendix I.3, and the first-row identity of
/// I.2): a causal first query row has a single legal key, so `P = 1`, the output is exactly
/// `V[0]` and the query and key gradients are exactly zero, whatever the scores. The paper's
/// prescribed construction: every query coordinate `c * 2^k` and every key coordinate 1, for
/// `c` in {3, 5, 7}, `k` in 0..16, at sequence lengths 1 and 128, with the incoming derivative
/// nonzero only at the first query. (The paper uses head dim 64; this kernel's is
/// `ATTN_TRAIN_HEAD_DIM`.)
///
/// Pinned bit for bit: `o[0] == V[0]` and `dV[0] == dO[0]` (both multiply by `P = exp(0) = 1`),
/// and every gradient row the derivative does not reach is exactly 0. Budgeted with
/// `reduction_bound`: `dQ[0]` and `dK[0]` (`dS = scale * P * (dO.V - dO.o)`, the difference of
/// two D-term dots of the same operands, so scale = `scale * sum |dO_c V_c|` for `dS`, times
/// `|k|` = 1 for `dQ` and times `|q|` for `dK`), which the paper requires to be zero.
#[test]
fn one_key_row_gives_output_v_and_zero_query_and_key_gradients() {
    with_gpu(|rt| {
        for t in [1usize, 128] {
            let dm = dims(1, t, 1, 1);
            let scale = f64::from(dm.scale);
            let v = random_f32(t * D, 0x0E0 + t as u64);
            let mut d_o = vec![0.0f32; t * D];
            d_o[..D].copy_from_slice(&random_f32(D, 0x0E1));
            let ds_scale = scale * (0..D).map(|c| f64::from((d_o[c] * v[c]).abs())).sum::<f64>();
            for c in [3.0f32, 5.0, 7.0] {
                for k in 0..16 {
                    let qv = c * (1u32 << k) as f32;
                    let q = vec![qv; t * D];
                    let key = vec![1.0f32; t * D];
                    let (o, dq, dk, dv) = fwd_bwd(rt, &dm, &q, &key, &v, &d_o);
                    let label = format!("t={t} c={c} k={k}");
                    let peak = |g: &[f32]| g.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                    eprintln!(
                        "OBSERVED one-key {label}: max|dQ[0]|={:e} max|dK[0]|={:e}",
                        peak(&dq[..D]),
                        peak(&dk[..D])
                    );
                    assert!(
                        o[..D].iter().zip(&v[..D]).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{label}: o[0] != V[0]"
                    );
                    assert!(
                        dv[..D].iter().zip(&d_o[..D]).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{label}: dV[0] != dO[0]"
                    );
                    let (bq, bk) = (
                        reduction_bound(D, ds_scale),
                        reduction_bound(D, ds_scale * f64::from(qv)),
                    );
                    for (i, g) in dq[..D].iter().enumerate() {
                        assert!(f64::from(g.abs()) <= bq, "{label}: dQ[0][{i}] = {g:e}, bound {bq:e}");
                    }
                    for (i, g) in dk[..D].iter().enumerate() {
                        assert!(f64::from(g.abs()) <= bk, "{label}: dK[0][{i}] = {g:e}, bound {bk:e}");
                    }
                    // Rows past the first receive no derivative: exactly zero, not "small".
                    assert!(dq[D..].iter().all(|g| *g == 0.0), "{label}: dQ past row 0 is not 0");
                    assert!(dk[D..].iter().all(|g| *g == 0.0), "{label}: dK past row 0 is not 0");
                    assert!(dv[D..].iter().all(|g| *g == 0.0), "{label}: dV past row 0 is not 0");
                }
            }
        }
    });
}

/// GProj's exact "-24 witness" (arXiv:2609.34272, Section 3.4 and Appendix A.1). Four keys
/// `k_j = (65536, j-1)`, `q = 0` (uniform attention), `alpha = 1/8`, `eta = 3/1024`, values
/// `(4, 4*eta)`, `(-4, 0)`, `(-4*eta, 0)`, `(0, 0)`, upstream derivative `u = (1, 1)`, all
/// exactly representable in bf16 and f32. The score gradient is `g = (1+eta, -1, -eta, 0)`,
/// which sums to zero, so the true first query-gradient coordinate is exactly 0. Casting `g`
/// to bf16 rounds `1+eta` to 1, leaves a row mass `rho = -eta` and a first coordinate of
/// `alpha * rho * 65536 = -24`; this test checks that arithmetic on the host, then that the
/// f32 backward does not leak it: `dQ[3][0]` is within `reduction_bound(4, sum_j |alpha g_j
/// k_j0|)` of 0 (scale: the four terms' magnitudes, about 1.6e4), which is far below 24.
/// The remaining coordinates and `o` are checked against the f64 reference. Head-dim
/// coordinates past the second are zero. Tokens 0-2 are the first three keys with `dO = 0`.
#[test]
fn the_minus_24_witness_leaves_no_query_gradient_in_the_first_coordinate() {
    use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
    let (alpha, eta) = (0.125f32, 3.0f32 / 1024.0);
    let g = [1.0 + eta, -1.0, -eta, 0.0f32];
    assert_eq!(g.iter().sum::<f32>(), 0.0, "g sums to zero, so the true dQ[0] is 0");
    let t: Vec<f32> = g.iter().map(|x| bf16_bits_to_f32(f32_to_bf16_bits(*x))).collect();
    let rho: f32 = t.iter().sum();
    assert_eq!(t, [1.0, -1.0, -eta, 0.0], "bf16 rounds 1+eta to 1");
    assert_eq!(rho, -eta);
    assert_eq!(alpha * rho * 65536.0, -24.0, "the witness value");

    with_gpu(|rt| {
        let tn = 4usize;
        let dm = AttnTrainDims {
            batch: 1,
            seq: tn as u32,
            q_heads: 1,
            kv_heads: 1,
            scale: alpha,
        };
        let s = Shape {
            b: 1,
            t: tn,
            hq: 1,
            hkv: 1,
            d: D,
            scale: f64::from(alpha),
        };
        let mut k = vec![0.0f32; tn * D];
        let mut v = vec![0.0f32; tn * D];
        let vals = [[4.0, 4.0 * eta], [-4.0, 0.0], [-4.0 * eta, 0.0], [0.0, 0.0]];
        for j in 0..tn {
            k[j * D] = 65536.0;
            k[j * D + 1] = j as f32;
            v[j * D..j * D + 2].copy_from_slice(&vals[j]);
        }
        let q = vec![0.0f32; tn * D];
        let mut d_o = vec![0.0f32; tn * D];
        d_o[3 * D] = 1.0;
        d_o[3 * D + 1] = 1.0;

        let (o, dq, dk, dv) = fwd_bwd(rt, &dm, &q, &k, &v, &d_o);
        let (q64, k64, v64, do64) = (f64s(&q), f64s(&k), f64s(&v), f64s(&d_o));
        let (want_o, _) = attn_fwd(&s, &q64, &k64, &v64);
        let (want_dq, want_dk, want_dv) = attn_bwd(&s, &q64, &k64, &v64, &do64);
        assert_eq!(want_dq[3 * D], 0.0, "reference: the first coordinate is exactly 0");

        eprintln!(
            "OBSERVED witness dQ[3][0]={:e} dQ[3][1]={:e} (reference {:e})",
            dq[3 * D],
            dq[3 * D + 1],
            want_dq[3 * D + 1]
        );
        let term_scale: f64 = g
            .iter()
            .map(|x| (f64::from(alpha) * f64::from(*x) * 65536.0).abs())
            .sum();
        let bound = reduction_bound(tn, term_scale);
        assert!(bound < 1.0, "the budget must sit far below the witness 24, got {bound}");
        assert!(
            f64::from(dq[3 * D]).abs() <= bound,
            "dQ[3][0] = {} (the bf16-cast witness would be -24), bound {bound:e}",
            dq[3 * D]
        );
        // Everything else, against the f64 reference, with the same stated scale.
        let dq1_scale: f64 = g
            .iter()
            .enumerate()
            .map(|(j, x)| (f64::from(alpha) * f64::from(*x) * j as f64).abs())
            .sum();
        let b1 = reduction_bound(tn, dq1_scale);
        assert!(
            (f64::from(dq[3 * D + 1]) - want_dq[3 * D + 1]).abs() <= b1,
            "dQ[3][1] = {}, want {} +- {b1:e}",
            dq[3 * D + 1],
            want_dq[3 * D + 1]
        );
        assert!(dq[..3 * D].iter().all(|x| *x == 0.0), "rows 0-2 get no derivative");
        close("witness o", &o, &want_o);
        close("witness dk", &dk, &want_dk);
        close("witness dv", &dv, &want_dv);
    });
}
