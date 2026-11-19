//! The two TimesFM 3 kernels (kernels/timesfm3.metal), each launched alone through the
//! raw dispatch API with the buffer slots, constants and grid src/timesfm3.rs uses, against
//! f64 references written from the kernels' documented semantics. Nothing here calls
//! src/timesfm3.rs: that path requests both pipelines for every forward, so the kernel trace
//! could not tell which kernel a check exercised.
//!
//! Every comparison is against a bound derived from the inputs of that row or head, never
//! fitted to a measured error. The largest measured error of each case is printed next to
//! its bound.

use std::sync::Arc;

use crate::common::{buf, buf_u32, random_f32};
use tessl::dispatch::{dispatch_1d, dispatch_2d_tg};
use tessl::GpuRuntime;

use super::Ledger;

const D: usize = 80;
const HALF: usize = 40;
/// f32 unit roundoff, 2^-24.
const U: f64 = 5.960_464_477_539_063e-8;
/// Query rows one threadgroup covers: 16 simdgroups of eight rows (src/timesfm3.rs).
const ROWS_PER_TG: usize = 16 * (32 / 4);

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().copied().map(f64::from).collect()
}

/// A positive gain around one, so a per-dim gain is neither 0 nor a no-op.
fn gains(seed: u64) -> Vec<f32> {
    random_f32(D, seed)
        .into_iter()
        .map(|x| 1.0 + 0.5 * x.clamp(-1.0, 1.0))
        .collect()
}

// ---------------------------------------------------------------------------
// qk prep
// ---------------------------------------------------------------------------

struct PrepCase {
    label: &'static str,
    rows: usize,
    h: usize,
    /// Patches per sequence: row m has position m % n.
    n: usize,
    rope: bool,
    eps: f32,
    /// Input magnitude; small values make eps matter against the mean square.
    mag: f32,
    seed: u64,
}

/// Half-split RoPE tables, [n, 40] each: angle = pos / 10000^(i / 40). Written in f64 and
/// rounded to f32, the table the kernel reads.
fn rope_tables(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sin = Vec::with_capacity(n * HALF);
    let mut cos = Vec::with_capacity(n * HALF);
    for pos in 0..n {
        for i in 0..HALF {
            let angle = pos as f64 / 10000f64.powf(i as f64 / HALF as f64);
            sin.push(angle.sin() as f32);
            cos.push(angle.cos() as f32);
        }
    }
    (sin, cos)
}

/// One head: optional rotation by (sin, cos) of its position, RMSNorm over 80 with eps, gain.
/// Returns the result and its error bound.
///
/// Bound: the f32 sum of 80 positive squares, taken in order, is off by at most 79 u relative;
/// the reciprocal root halves that (about 4e-6 / 2 = 2.4e-6 at 79 u = 4.7e-6), the
/// precise:: divide and rsqrt add a few ulp, and the rotation, the two multiplies and the
/// gain add about 10 u in total (6e-7). 4e-6 of the head's largest
/// |input| * r * max gain covers the 3.1e-6 sum with margin, and the scale is the head's own
/// largest term, so a cancelling rotation (an output far below the inputs) is bounded by what
/// it was computed from.
fn prep_head(x: &[f64], rot: Option<(&[f32], &[f32])>, eps: f64, g: &[f32]) -> (Vec<f64>, f64) {
    let mut v = x.to_vec();
    if let Some((sin, cos)) = rot {
        for i in 0..HALF {
            let (c, s) = (f64::from(cos[i]), f64::from(sin[i]));
            let (a, b) = (x[i], x[i + HALF]);
            v[i] = a * c - b * s;
            v[i + HALF] = b * c + a * s;
        }
    }
    let mean = v.iter().map(|t| t * t).sum::<f64>() / D as f64;
    let r = 1.0 / (mean + eps).sqrt();
    let want: Vec<f64> = v.iter().zip(g).map(|(t, &gi)| t * r * f64::from(gi)).collect();
    let gmax = g.iter().fold(0.0f32, |m, &t| m.max(t.abs()));
    let vmax = x
        .iter()
        .fold(0.0f64, |m, t| m.max(t.abs()))
        .max(v.iter().fold(0.0f64, |m, t| m.max(t.abs())));
    (want, 4.0e-6 * vmax * r * f64::from(gmax))
}

fn run_prep(rt: &Arc<GpuRuntime>, c: &PrepCase) {
    let name = "timesfm3_qk_prep_h80";
    let (rows, h, n) = (c.rows, c.h, c.n);
    let width = h * D;
    let scale_in = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| x * c.mag).collect() };
    let q0 = scale_in(random_f32(rows * width, c.seed));
    let k0 = scale_in(random_f32(rows * width, c.seed + 1));
    let (gq, gk) = (gains(c.seed + 2), gains(c.seed + 3));
    let (sin, cos) = rope_tables(n);
    // Without rotation the kernel must not read the tables: poison them.
    let (sin_dev, cos_dev) = if c.rope {
        (sin.clone(), cos.clone())
    } else {
        (vec![f32::NAN; sin.len()], vec![f32::NAN; cos.len()])
    };
    let (qb, kb) = (buf(rt, &q0), buf(rt, &k0));
    let (gqb, gkb, sb, cb) = (buf(rt, &gq), buf(rt, &gk), buf(rt, &sin_dev), buf(rt, &cos_dev));
    let p = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    dispatch_1d(rt, &p, 2 * rows * h, |bnd| {
        bnd.bind_gpu_buf(&qb, 0);
        bnd.bind_gpu_buf(&kb, 1);
        bnd.bind_gpu_buf(&gqb, 2);
        bnd.bind_gpu_buf(&gkb, 3);
        bnd.bind_gpu_buf(&sb, 4);
        bnd.bind_gpu_buf(&cb, 5);
        bnd.bind_u32(rows as u32, 6);
        bnd.bind_u32(h as u32, 7);
        bnd.bind_u32(n as u32, 8);
        bnd.bind_u32(u32::from(c.rope), 9);
        bnd.bind_f32(c.eps, 10);
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("{name}: {e}"));
    let got_q = qb.try_contents_f32().expect("read Q")[..rows * width].to_vec();
    let got_k = kb.try_contents_f32().expect("read K")[..rows * width].to_vec();

    let mut worst = 0.0f64;
    let mut worst_ratio = 0.0f64;
    for (which, input, gain, got) in [("Q", &q0, &gq, &got_q), ("K", &k0, &gk, &got_k)] {
        for m in 0..rows {
            let pos = m % n;
            let rot = c
                .rope
                .then(|| (&sin[pos * HALF..(pos + 1) * HALF], &cos[pos * HALF..(pos + 1) * HALF]));
            for hh in 0..h {
                let off = m * width + hh * D;
                let (want, tol) = prep_head(&f64s(&input[off..off + D]), rot, f64::from(c.eps), gain);
                for i in 0..D {
                    let g = got[off + i];
                    let err = (f64::from(g) - want[i]).abs();
                    assert!(
                        g.is_finite() && err <= tol,
                        "{name}[{}] {which} row {m} head {hh} dim {i}: got {g} want {} (err {err:.3e} > bound {tol:.3e})",
                        c.label,
                        want[i]
                    );
                    worst = worst.max(err);
                    worst_ratio = worst_ratio.max(err / tol);
                }
            }
        }
    }
    println!(
        "kcheck timesfm3 {name}[{}]: max err {worst:.3e}, worst err/bound {worst_ratio:.3}",
        c.label
    );
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

struct AttnCase {
    label: &'static str,
    causal: bool,
    /// Sequences, positions per sequence, heads.
    nseq: usize,
    t: usize,
    h: usize,
    /// (P, S1, S2, S3): sequence s, position t is model row (s / P) * S1 + (s % P) * S2 + t * S3.
    strides: (usize, usize, usize, usize),
    /// Total model rows the buffers hold.
    total_rows: usize,
    /// Causal: leading masked keys per sequence. Variate: key bitmask per sequence.
    info: Vec<u32>,
    scale: f32,
    seed: u64,
}

struct AttnWant {
    out: Vec<f64>,
    tol: Vec<f64>,
    /// Elements whose query sees no key: the kernel must write exact zeros.
    empty: Vec<bool>,
}

/// Softmax attention in f64 over the keys each query is allowed to see, written from the
/// kernel's documented mask semantics.
///
/// Bound per output element (weighted average of values, so an error eps in every weight
/// moves it by at most 2 * eps * max|v|): eps = scale * 80 u * S (an f32 dot of 80 terms, S the
/// largest sum of |q_i k_i| over the visible keys; 80 u is the sequential worst case) + 2 u *
/// xmax (fast-math exp: argument rounding costs u * |x| relative, used for the weight and for the
/// running-max rescale, xmax the widest score gap seen) + 4 u * T (one rescale and one
/// accumulate round per key, linear worst case, 4 u each including the exp's own ulps) + 8 u
/// (final reciprocal and store).
fn attn_ref(c: &AttnCase, q: &[f32], k: &[f32], v: &[f32]) -> AttnWant {
    let (p, s1, s2, s3) = c.strides;
    let width = c.h * D;
    let n = c.total_rows * width;
    let mut out = vec![0.0f64; n];
    let mut tol = vec![0.0f64; n];
    let mut empty = vec![false; n];
    let scale = f64::from(c.scale);
    for s in 0..c.nseq {
        let info = c.info[s];
        let row = |t: usize| (s / p) * s1 + (s % p) * s2 + t * s3;
        for hh in 0..c.h {
            let at = |t: usize| row(t) * width + hh * D;
            for tq in 0..c.t {
                let keys: Vec<usize> = (0..c.t)
                    .filter(|&kk| {
                        if c.causal {
                            kk >= info as usize && kk <= tq
                        } else {
                            kk < 32 && (info >> kk) & 1 == 1
                        }
                    })
                    .collect();
                let o = at(tq);
                if keys.is_empty() {
                    empty[o..o + D].fill(true);
                    continue;
                }
                let mut scores = Vec::with_capacity(keys.len());
                let mut sum_abs = 0.0f64;
                for &kk in &keys {
                    let (qo, ko) = (at(tq), at(kk));
                    let mut dot = 0.0f64;
                    let mut abs = 0.0f64;
                    for d in 0..D {
                        let term = f64::from(q[qo + d]) * f64::from(k[ko + d]);
                        dot += term;
                        abs += term.abs();
                    }
                    scores.push(scale * dot);
                    sum_abs = sum_abs.max(abs);
                }
                let smax = scores.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x));
                let smin = scores.iter().fold(f64::INFINITY, |m, &x| m.min(x));
                let w: Vec<f64> = scores.iter().map(|x| (x - smax).exp()).collect();
                let l: f64 = w.iter().sum();
                let mut vmax = 0.0f64;
                for (&kk, wi) in keys.iter().zip(&w) {
                    let vo = at(kk);
                    for d in 0..D {
                        out[o + d] += wi / l * f64::from(v[vo + d]);
                        vmax = vmax.max(f64::from(v[vo + d]).abs());
                    }
                }
                let eps = scale * 80.0 * U * sum_abs + 2.0 * U * (smax - smin) + 4.0 * U * c.t as f64 + 8.0 * U;
                tol[o..o + D].fill(2.0 * eps * vmax);
            }
        }
    }
    AttnWant { out, tol, empty }
}

fn run_attn(rt: &Arc<GpuRuntime>, c: &AttnCase) {
    let name = "timesfm3_attn_h80_r4_g16";
    let (p, s1, s2, s3) = c.strides;
    let n = c.total_rows * c.h * D;
    let (q, k, v) = (
        random_f32(n, c.seed),
        random_f32(n, c.seed + 1),
        random_f32(n, c.seed + 2),
    );
    let want = attn_ref(c, &q, &k, &v);
    let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
    // Unwritten output elements stay NaN and fail the finite check.
    let ob = buf(rt, &vec![f32::NAN; n]);
    let ib = buf_u32(rt, &c.info);
    let pipe = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    dispatch_2d_tg(rt, &pipe, c.t.div_ceil(ROWS_PER_TG), c.nseq * c.h, 16 * 32, |bnd| {
        bnd.bind_gpu_buf(&qb, 0);
        bnd.bind_gpu_buf(&kb, 1);
        bnd.bind_gpu_buf(&vb, 2);
        bnd.bind_gpu_buf(&ob, 3);
        bnd.bind_u32(c.t as u32, 4);
        bnd.bind_u32(c.h as u32, 5);
        bnd.bind_u32(p as u32, 6);
        bnd.bind_u32(s1 as u32, 7);
        bnd.bind_u32(s2 as u32, 8);
        bnd.bind_u32(s3 as u32, 9);
        bnd.bind_u32(u32::from(c.causal), 10);
        bnd.bind_gpu_buf(&ib, 11);
        bnd.bind_f32(c.scale, 12);
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("{name}: {e}"));
    let got = ob.try_contents_f32().expect("read O")[..n].to_vec();

    let (mut worst, mut worst_ratio, mut empties) = (0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        let g = got[i];
        assert!(
            g.is_finite(),
            "{name}[{}] element {i}: not written or non-finite ({g})",
            c.label
        );
        if want.empty[i] {
            empties += 1;
            assert!(
                g.to_bits() == 0,
                "{name}[{}] element {i}: query sees no key, want exact +0.0, got {g}",
                c.label
            );
            continue;
        }
        let err = (f64::from(g) - want.out[i]).abs();
        assert!(
            err <= want.tol[i],
            "{name}[{}] element {i}: got {g} want {} (err {err:.3e} > bound {:.3e})",
            c.label,
            want.out[i],
            want.tol[i]
        );
        worst = worst.max(err);
        worst_ratio = worst_ratio.max(err / want.tol[i]);
    }
    println!(
        "kcheck timesfm3 {name}[{}]: max err {worst:.3e}, worst err/bound {worst_ratio:.3}, {empties} zero-row elements",
        c.label
    );
}

fn causal(label: &'static str, nseq: usize, t: usize, h: usize, info: Vec<u32>, seed: u64) -> AttnCase {
    AttnCase {
        label,
        causal: true,
        nseq,
        t,
        h,
        // S2 is unused while P is 1; a stray value shows a kernel that reads it.
        strides: (1, t, 12_345, 1),
        total_rows: nseq * t,
        info,
        scale: 1.0,
        seed,
    }
}

/// Variate attention over b batches of v variates and n patches: model rows are [b, v, n].
fn variate(
    label: &'static str,
    (b, v, n): (usize, usize, usize),
    h: usize,
    info: Vec<u32>,
    scale: f32,
    seed: u64,
) -> AttnCase {
    AttnCase {
        label,
        causal: false,
        nseq: b * n,
        t: v,
        h,
        strides: (n, v * n, 1, n),
        total_rows: b * v * n,
        info,
        scale,
        seed,
    }
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let name = "timesfm3_qk_prep_h80";
    ledger.credit(rt, name, &[name], || {
        let case = |label, rows, h, n, rope, eps, mag, seed| PrepCase {
            label,
            rows,
            h,
            n,
            rope,
            eps,
            mag,
            seed,
        };
        let eps = f32::EPSILON;
        run_prep(rt, &case("rope h3 n5", 37, 3, 5, true, eps, 1.0, 100));
        run_prep(rt, &case("no-rope h3 n5", 37, 3, 5, false, eps, 1.0, 200));
        run_prep(rt, &case("rope h1 n1", 1, 1, 1, true, eps, 1.0, 300));
        run_prep(rt, &case("rope h2 n7 rows 23", 23, 2, 7, true, eps, 1.0, 400));
        // Mean square near 3e-7 against eps 1.2e-7: eps is a third of the denominator.
        run_prep(rt, &case("rope small-input", 11, 2, 4, true, eps, 1.0e-3, 500));
        run_prep(rt, &case("no-rope large eps", 9, 2, 3, false, 0.05, 1.0, 600));
    });

    let name = "timesfm3_attn_h80_r4_g16";
    ledger.credit(rt, name, &[name], || {
        // T = 150: two threadgroups, the second holding 22 rows (two full simdgroups and a
        // partial third). info: no lead, a short lead, T (every key masked), past T, a long lead.
        run_attn(
            rt,
            &causal("causal t150 h2", 5, 150, 2, vec![0, 7, 150, 155, 100], 1000),
        );
        // T = 129: one row past a full threadgroup.
        run_attn(rt, &causal("causal t129 h1", 3, 129, 1, vec![0, 3, 200], 1100));
        // T = 3: a single partial simdgroup.
        run_attn(rt, &causal("causal t3 h2", 4, 3, 2, vec![0, 1, 2, 3], 1200));
        // Variate: 20 variates (two full simdgroups and a partial third), mask word 0 (empty),
        // a single key, all keys, and bits past T that must be ignored.
        run_attn(
            rt,
            &variate(
                "variate v20 n3",
                (2, 20, 3),
                2,
                vec![0xA5A5_A5A5, 0, 1 << 19, 0x000F_FFFF, 0xFFFF_FFFF, 0x0012_3457],
                1.0,
                2000,
            ),
        );
        // Full 32-variate width: bit 31 is a key, and the scale is not 1.
        run_attn(
            rt,
            &variate(
                "variate v32 n2 scale",
                (2, 32, 2),
                1,
                vec![0x8000_0001, 0xFFFF_FFFF, 0, 0x7FFF_FFFE],
                0.35,
                2100,
            ),
        );
        // One variate: query and key are the same row.
        run_attn(
            rt,
            &variate("variate v1 n4", (2, 1, 4), 2, vec![1, 0, 1, 1, 0, 1, 1, 1], 1.0, 2200),
        );
    });
}
