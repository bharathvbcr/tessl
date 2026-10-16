//! The GDN training op (`tessl::gdn_train`): transformers' chunk-rule seam,
//! forward and backward.
//!
//! Evidence, bottom up:
//!
//! 1. The f64 reference's forward equals `common::qwen35::gdn_f64`, the
//!    transformers-anchored recurrence, given the same gates.
//! 2. The f64 reference's hand-derived backward equals central finite
//!    differences of its forward for every input, the initial state included.
//! 3. The kernels match that reference: every output within `1e-4` of the
//!    reference's largest magnitude (the kernels are f32 with `precise::`
//!    exponentials and square roots), across chunk edges (T = 1, 63, 64, 65,
//!    130), with and without an initial state and a final-state gradient, and
//!    at Qwen3.5-2B's head count and value dim. The backward is deterministic.

mod common;

use std::sync::Arc;

use common::gdn_train::{gdn_train_bwd_f64, gdn_train_f64, Inputs, Shape};
use common::qwen35::{gdn_f64, sigmoid, softplus, GdnProblem, GdnShape};
use common::{random_f32, with_gpu};
use tessl::gdn_train::{
    gdn_train_backward, gdn_train_forward, gdn_train_forward_with, GdnTrainDims, GdnTrainGrads, GdnTrainInputs,
    GdnTrainKernel, GdnTrainWorkspace, GDN_TRAIN_PRODUCTION,
};

/// Both kernels, each held to the f64 reference: the token-sequential one is
/// the oracle the chunked one is also compared with.
const KERNELS: [GdnTrainKernel; 2] = [GdnTrainKernel::Sequential, GdnTrainKernel::Chunked];
use tessl::{GpuRuntime, Tensor};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

struct Owned {
    s: Shape,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    s0: Option<Vec<f64>>,
}

impl Owned {
    fn random(s: Shape, seed: u64, with_state: bool) -> Self {
        let n = s.b * s.t * s.h;
        Self {
            s,
            q: f64s(&random_f32(n * s.dk, seed)),
            k: f64s(&random_f32(n * s.dk, seed + 1)),
            v: f64s(&random_f32(n * s.dv, seed + 2)),
            // log decays in [-1.5, 0): exp(g) in (0.22, 1]
            g: random_f32(n, seed + 3)
                .iter()
                .map(|&x| -0.75 * (f64::from(x) + 1.0) - 1e-3)
                .collect(),
            beta: random_f32(n, seed + 4)
                .iter()
                .map(|&x| 0.5 + 0.45 * f64::from(x))
                .collect(),
            s0: with_state.then(|| f64s(&random_f32(s.b * s.h * s.dk * s.dv, seed + 5))),
        }
    }

    fn inputs(&self) -> Inputs<'_> {
        Inputs {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: self.s0.as_deref(),
        }
    }
}

#[test]
fn the_reference_forward_is_the_transformers_recurrence() {
    let (b, t, h, dv) = (2, 9, 2, 32);
    let n = b * t * h;
    let q = random_f32(n * 128, 1);
    let k = random_f32(n * 128, 2);
    let v = random_f32(n * dv, 3);
    let a = random_f32(n, 4);
    let bl = random_f32(n, 5);
    let a_log = random_f32(h, 6);
    let dt_bias = random_f32(h, 7);
    let s0 = random_f32(b * h * 128 * dv, 8);
    let (want_o, want_s) = gdn_f64(&GdnProblem {
        s: GdnShape { b, t, hk: h, hv: h, dv },
        q: &q,
        k: &k,
        v: &v,
        a: &a,
        b: &bl,
        a_log: &a_log,
        dt_bias: &dt_bias,
        state0: Some(&s0),
        snapshot: false,
    });
    let g: Vec<f64> = (0..n)
        .map(|i| -f64::from(a_log[i % h]).exp() * softplus(f64::from(a[i]) + f64::from(dt_bias[i % h])))
        .collect();
    let beta: Vec<f64> = bl.iter().map(|&x| sigmoid(f64::from(x))).collect();
    let (qd, kd, vd, sd) = (f64s(&q), f64s(&k), f64s(&v), f64s(&s0));
    let (o, s) = gdn_train_f64(&Inputs {
        s: Shape { b, t, h, dk: 128, dv },
        q: &qd,
        k: &kd,
        v: &vd,
        g: &g,
        beta: &beta,
        s0: Some(&sd),
    });
    let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| (a - b).abs() <= 1e-12 * (1.0 + b.abs()));
    assert!(close(&o, &want_o), "output");
    assert!(close(&s, &want_s), "final state");
}

/// Central differences, h = 1e-6, of `L = sum(W_o * o) + sum(W_s * final)`
/// for fixed random weights, against the analytic gradient. In f64 the
/// truncation and rounding error at this step is ~1e-9 relative.
#[test]
fn the_reference_backward_is_the_derivative_of_its_forward() {
    let s = Shape {
        b: 2,
        t: 7,
        h: 2,
        dk: 6,
        dv: 4,
    };
    let base = Owned::random(s, 11, true);
    let n = s.b * s.t * s.h;
    let w_o = f64s(&random_f32(n * s.dv, 90));
    let w_s = f64s(&random_f32(s.b * s.h * s.dk * s.dv, 91));
    let loss = |p: &Owned| -> f64 {
        let (o, fin) = gdn_train_f64(&p.inputs());
        o.iter().zip(&w_o).map(|(a, b)| a * b).sum::<f64>() + fin.iter().zip(&w_s).map(|(a, b)| a * b).sum::<f64>()
    };
    let g = gdn_train_bwd_f64(&base.inputs(), &w_o, Some(&w_s));
    let hstep = 1e-6;
    let check = |name: &str, analytic: &[f64], field: &dyn Fn(&mut Owned) -> &mut Vec<f64>| {
        let mut worst = 0.0f64;
        for (i, &an) in analytic.iter().enumerate() {
            let mut plus = Owned {
                s0: base.s0.clone(),
                q: base.q.clone(),
                k: base.k.clone(),
                v: base.v.clone(),
                g: base.g.clone(),
                beta: base.beta.clone(),
                s,
            };
            let mut minus = Owned {
                s0: base.s0.clone(),
                q: base.q.clone(),
                k: base.k.clone(),
                v: base.v.clone(),
                g: base.g.clone(),
                beta: base.beta.clone(),
                s,
            };
            field(&mut plus)[i] += hstep;
            field(&mut minus)[i] -= hstep;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * hstep);
            let err = (fd - an).abs() / (1.0 + fd.abs());
            worst = worst.max(err);
        }
        assert!(worst < 1e-7, "{name}: worst relative error {worst:.2e}");
        eprintln!("{name}: worst relative error vs finite differences {worst:.2e}");
    };
    check("dq", &g.dq, &|p| &mut p.q);
    check("dk", &g.dk, &|p| &mut p.k);
    check("dv", &g.dv, &|p| &mut p.v);
    check("dg", &g.dg, &|p| &mut p.g);
    check("dbeta", &g.dbeta, &|p| &mut p.beta);
    check("ds0", &g.ds0, &|p| p.s0.as_mut().unwrap());
}

// ---------------------------------------------------------------- GPU ---

/// Sentinel written to every output first, so an element never written shows.
const SENTINEL: f32 = -7.25e27;

struct Case {
    s: Shape,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    s0: Option<Vec<f32>>,
    d_o: Vec<f32>,
    dfin: Option<Vec<f32>>,
}

impl Case {
    fn new(b: usize, t: usize, h: usize, dv: usize, seed: u64, with_s0: bool, with_dfin: bool) -> Self {
        let s = Shape { b, t, h, dk: 128, dv };
        let n = b * t * h;
        Self {
            s,
            q: random_f32(n * 128, seed),
            k: random_f32(n * 128, seed + 1),
            v: random_f32(n * dv, seed + 2),
            g: random_f32(n, seed + 3)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect(),
            beta: random_f32(n, seed + 4).iter().map(|&x| 0.5 + 0.45 * x).collect(),
            s0: with_s0.then(|| random_f32(b * h * 128 * dv, seed + 5)),
            d_o: random_f32(n * dv, seed + 6),
            dfin: with_dfin.then(|| random_f32(b * h * 128 * dv, seed + 7)),
        }
    }
}

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).unwrap();
    t.write_f32(data).unwrap();
    t
}

fn sentinel(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Tensor {
    tensor(rt, shape, &vec![SENTINEL; shape.iter().product()])
}

fn assert_close(label: &str, got: &[f32], want: &[f64]) -> f64 {
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
    assert!(
        rel <= 1e-4,
        "{label}: max err {:.3e} at {} (got {} want {}), {rel:.2e} of max|ref| {peak:.3e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
    rel
}

/// Run the forward and backward on the GPU and check both against f64.
/// Returns the worst relative error seen.
fn check(rt: &Arc<GpuRuntime>, c: &Case, kernel: GdnTrainKernel) -> f64 {
    let Shape { b, t, h, dv, .. } = c.s;
    let label = format!(
        "{kernel:?} B={b} T={t} H={h} Dv={dv} s0={} dfin={}",
        c.s0.is_some(),
        c.dfin.is_some()
    );
    let dims = GdnTrainDims {
        batch: b as u32,
        seq: t as u32,
        heads: h as u32,
        v_dim: dv as u32,
    };
    let (q, k, v) = (
        tensor(rt, &[b, t, h, 128], &c.q),
        tensor(rt, &[b, t, h, 128], &c.k),
        tensor(rt, &[b, t, h, dv], &c.v),
    );
    let (g, beta) = (tensor(rt, &[b, t, h], &c.g), tensor(rt, &[b, t, h], &c.beta));
    let s0 = c.s0.as_ref().map(|s| tensor(rt, &[b, h, 128, dv], s));
    let x = GdnTrainInputs {
        q: &q,
        k: &k,
        v: &v,
        g: &g,
        beta: &beta,
        s0: s0.as_ref(),
    };
    let o = sentinel(rt, &[b, t, h, dv]);
    let sfin = sentinel(rt, &[b, h, 128, dv]);
    let ckpt = sentinel(rt, &dims.checkpoint_shape());
    gdn_train_forward_with(kernel, rt, dims, x, &o, Some(&sfin), &ckpt).unwrap_or_else(|e| panic!("{label}: {e}"));

    let (q64, k64, v64, g64, b64) = (f64s(&c.q), f64s(&c.k), f64s(&c.v), f64s(&c.g), f64s(&c.beta));
    let s064 = c.s0.as_ref().map(|s| f64s(s));
    let inp = Inputs {
        s: c.s,
        q: &q64,
        k: &k64,
        v: &v64,
        g: &g64,
        beta: &b64,
        s0: s064.as_deref(),
    };
    let (want_o, want_fin) = gdn_train_f64(&inp);
    let mut worst = 0.0f64;
    worst = worst.max(assert_close(&format!("{label} o"), &o.read_f32().unwrap(), &want_o));
    worst = worst.max(assert_close(
        &format!("{label} final state"),
        &sfin.read_f32().unwrap(),
        &want_fin,
    ));

    let d_o = tensor(rt, &[b, t, h, dv], &c.d_o);
    let dfin = c.dfin.as_ref().map(|d| tensor(rt, &[b, h, 128, dv], d));
    let ws = GdnTrainWorkspace::new_for(rt, dims, kernel).unwrap();
    let (dq, dk, dvv) = (
        sentinel(rt, &[b, t, h, 128]),
        sentinel(rt, &[b, t, h, 128]),
        sentinel(rt, &[b, t, h, dv]),
    );
    let (dg, db) = (sentinel(rt, &[b, t, h]), sentinel(rt, &[b, t, h]));
    let ds0 = c.s0.as_ref().map(|_| sentinel(rt, &[b, h, 128, dv]));
    let grads = || GdnTrainGrads {
        dq: &dq,
        dk: &dk,
        dv: &dvv,
        dg: &dg,
        dbeta: &db,
        ds0: ds0.as_ref(),
    };
    gdn_train_backward(rt, dims, x, &ckpt, &d_o, dfin.as_ref(), &ws, grads())
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let d_o64 = f64s(&c.d_o);
    let dfin64 = c.dfin.as_ref().map(|d| f64s(d));
    let want = gdn_train_bwd_f64(&inp, &d_o64, dfin64.as_deref());
    let first: Vec<Vec<f32>> = [&dq, &dk, &dvv, &dg, &db]
        .iter()
        .map(|t| t.read_f32().unwrap())
        .collect();
    for (name, got, w) in [
        ("dq", &first[0], &want.dq),
        ("dk", &first[1], &want.dk),
        ("dv", &first[2], &want.dv),
        ("dg", &first[3], &want.dg),
        ("dbeta", &first[4], &want.dbeta),
    ] {
        worst = worst.max(assert_close(&format!("{label} {name}"), got, w));
    }
    if let Some(ds0) = &ds0 {
        worst = worst.max(assert_close(
            &format!("{label} ds0"),
            &ds0.read_f32().unwrap(),
            &want.ds0,
        ));
    }
    // Deterministic: a second backward is bit-identical.
    gdn_train_backward(rt, dims, x, &ckpt, &d_o, dfin.as_ref(), &ws, grads()).unwrap();
    for (i, tns) in [&dq, &dk, &dvv, &dg, &db].iter().enumerate() {
        let again = tns.read_f32().unwrap();
        assert!(
            again.iter().zip(&first[i]).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label}: backward output {i} changed on a rerun"
        );
    }
    eprintln!("{label}: worst {worst:.2e} of max|ref|");
    worst
}

#[test]
fn the_kernels_match_the_reference_across_chunk_edges() {
    with_gpu(|rt| {
        for (t, s0, dfin) in [
            (1, false, false),
            (63, true, false),
            (64, false, true),
            (65, true, true),
            (130, true, true),
        ] {
            for kernel in KERNELS {
                check(rt, &Case::new(2, t, 3, 32, 100 + t as u64, s0, dfin), kernel);
            }
        }
    });
}

#[test]
fn the_kernels_match_the_reference_at_qwen35_2b_heads() {
    with_gpu(|rt| {
        for kernel in KERNELS {
            check(rt, &Case::new(1, 200, 16, 128, 7, false, false), kernel);
        }
    });
}

#[test]
fn the_host_refuses_what_the_kernels_cannot_run() {
    with_gpu(|rt| {
        let dims = GdnTrainDims {
            batch: 1,
            seq: 5,
            heads: 2,
            v_dim: 32,
        };
        let z = |shape: &[usize]| rt.alloc_tensor_f32(shape).unwrap();
        let (q, k, v) = (z(&[1, 5, 2, 128]), z(&[1, 5, 2, 128]), z(&[1, 5, 2, 32]));
        let (g, beta) = (z(&[1, 5, 2]), z(&[1, 5, 2]));
        let x = GdnTrainInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            s0: None,
        };
        let (o, ckpt) = (z(&[1, 5, 2, 32]), z(&dims.checkpoint_shape()));
        let fwd = |d: GdnTrainDims, x: GdnTrainInputs<'_>, o: &Tensor, sf: Option<&Tensor>, c: &Tensor| {
            gdn_train_forward(rt, d, x, o, sf, c)
        };
        let refused = |r: Result<(), String>, needle: &str| {
            let e = r.expect_err(needle);
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        };
        fwd(dims, x, &o, None, &ckpt).expect("valid call");
        refused(
            fwd(GdnTrainDims { v_dim: 24, ..dims }, x, &o, None, &ckpt),
            "multiple of 16",
        );
        refused(fwd(GdnTrainDims { seq: 0, ..dims }, x, &o, None, &ckpt), "non-zero");
        // heads * 128 = 2^32: the dense window's u32 row stride would wrap.
        refused(
            fwd(
                GdnTrainDims {
                    heads: 1 << 25,
                    seq: 1,
                    ..dims
                },
                x,
                &o,
                None,
                &ckpt,
            ),
            "row width past u32",
        );
        refused(
            fwd(GdnTrainDims { seq: 6, ..dims }, x, &o, None, &ckpt),
            "q must be f32 [1, 6, 2, 128]",
        );
        let k64 = z(&[1, 5, 2, 64]);
        refused(
            fwd(dims, GdnTrainInputs { k: &k64, ..x }, &o, None, &ckpt),
            "k must be f32",
        );
        let bf = rt.alloc_tensor_bf16(&[1, 5, 2]).unwrap();
        refused(
            fwd(dims, GdnTrainInputs { g: &bf, ..x }, &o, None, &ckpt),
            "g must be f32",
        );
        refused(fwd(dims, x, &o, None, &z(&[1, 2, 2, 128, 32])), "ckpt must be f32");
        refused(
            fwd(dims, x, &q.try_view(&[1, 5, 2, 32], 0).unwrap(), None, &ckpt),
            "output o overlaps input q",
        );
        refused(
            fwd(dims, x, &o, Some(&z(&[1, 2, 128, 16])), &ckpt),
            "s_fin must be f32 [1, 2, 128, 32]",
        );
        let other = GpuRuntime::new().unwrap();
        let foreign = other.alloc_tensor_f32(&[1, 5, 2, 32]).unwrap();
        refused(fwd(dims, x, &foreign, None, &ckpt), "o belongs to another runtime");

        let ws = GdnTrainWorkspace::new(rt, dims).unwrap();
        let d_o = z(&[1, 5, 2, 32]);
        let (dq, dk, dvv, dg, db) = (
            z(&[1, 5, 2, 128]),
            z(&[1, 5, 2, 128]),
            z(&[1, 5, 2, 32]),
            z(&[1, 5, 2]),
            z(&[1, 5, 2]),
        );
        let grads = GdnTrainGrads {
            dq: &dq,
            dk: &dk,
            dv: &dvv,
            dg: &dg,
            dbeta: &db,
            ds0: None,
        };
        let bwd = |ws: &GdnTrainWorkspace, x: GdnTrainInputs<'_>, gr: GdnTrainGrads<'_>| {
            gdn_train_backward(rt, dims, x, &ckpt, &d_o, None, ws, gr)
        };
        bwd(&ws, x, GdnTrainGrads { ..grads }).expect("valid backward");
        let ws_other = GdnTrainWorkspace::new(rt, GdnTrainDims { seq: 6, ..dims }).unwrap();
        refused(bwd(&ws_other, x, GdnTrainGrads { ..grads }), "the workspace is for");
        let s0 = z(&[1, 2, 128, 32]);
        refused(
            bwd(&ws, GdnTrainInputs { s0: Some(&s0), ..x }, GdnTrainGrads { ..grads }),
            "ds0 is required",
        );
        let ds0 = z(&[1, 2, 128, 32]);
        refused(
            bwd(
                &ws,
                x,
                GdnTrainGrads {
                    ds0: Some(&ds0),
                    ..grads
                },
            ),
            "no initial state",
        );
        refused(
            bwd(&ws, x, GdnTrainGrads { dk: &dq, ..grads }),
            "outputs dq and dk overlap",
        );
        refused(
            bwd(&ws, x, GdnTrainGrads { dv: &d_o, ..grads }),
            "output dv overlaps input d_o",
        );
        refused(
            GdnTrainWorkspace::new(rt, GdnTrainDims { heads: 0, ..dims }).map(|_| ()),
            "non-zero",
        );
        // The production (chunked) workspace for B = 1, T = 5 (one padded
        // chunk of 64 rows), H = 2, Dv = 32 (two slices): kh, qh; gc, bt; wm,
        // aq; dsbuf; dbuf, drbuf, dop; dscore, das, gtot, dbs; dgpart, dbpart, xpart;
        // dqh, dkh.
        let (bh, tp, ns) = (2usize, 64usize, 2usize);
        assert_eq!(
            GdnTrainWorkspace::bytes_for(dims),
            4 * (2 * bh * tp * 128
                + 2 * bh * tp
                + 2 * bh * 64 * 64
                + bh * 128 * 32
                + 3 * bh * tp * 32
                + 4 * bh * 64 * 64
                + 3 * ns * bh * tp
                + 2 * 10 * 128)
        );
    });
}

/// The grouped-heads form: `q`, `k`, `v` read, and `dq`, `dk`, `dv` written,
/// as windows of one `[tokens, ld]` matrix (the Qwen3.5 conv output's layout,
/// behind a pad column block), with `key_heads < heads`. Every output is
/// bit-identical to what the dense kernels give on the host-repeated operands
/// (each key head copied to its `r` value heads), with `dq` and `dk` the host
/// sums of the dense per-value-head gradients in value-head order: what
/// `repeat_heads` and `sum_heads` computed. Columns outside the windows are
/// never written.
#[test]
fn windows_with_shared_key_heads_equal_the_repeated_dense_path() {
    use tessl::gdn_train::{gdn_train_backward_windowed, gdn_train_forward_windowed};
    use tessl::gdn_train::{GdnTrainWindowedGrads, GdnTrainWindowedInputs, GdnWindow};
    const PAD: usize = 8;
    with_gpu(|rt| {
        for (hk, hv) in [(2usize, 2usize), (2, 4), (1, 3)] {
            let (b, t, dv, dk) = (2usize, 70usize, 32usize, 128usize);
            let r = hv / hk;
            let tokens = b * t;
            let (qw, vw) = (hk * dk, hv * dv);
            let ld = PAD + 2 * qw + vw;
            let seed = 500 + hv as u64;
            let conv = random_f32(tokens * ld, seed);
            let col = |tok: usize, c: usize| conv[tok * ld + c];
            // The dense operands: value head h reads key head h / r.
            let (mut qd, mut kd, mut vd) = (Vec::new(), Vec::new(), Vec::new());
            for tok in 0..tokens {
                for h in 0..hv {
                    let hkh = h / r;
                    qd.extend((0..dk).map(|i| col(tok, PAD + hkh * dk + i)));
                    kd.extend((0..dk).map(|i| col(tok, PAD + qw + hkh * dk + i)));
                    vd.extend((0..dv).map(|j| col(tok, PAD + 2 * qw + h * dv + j)));
                }
            }
            let n = tokens * hv;
            let g: Vec<f32> = random_f32(n, seed + 1)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect();
            let beta: Vec<f32> = random_f32(n, seed + 2).iter().map(|&x| 0.5 + 0.45 * x).collect();
            let s0 = random_f32(b * hv * dk * dv, seed + 3);
            let d_o = random_f32(n * dv, seed + 4);
            let dfin = random_f32(b * hv * dk * dv, seed + 5);
            let dims = GdnTrainDims {
                batch: b as u32,
                seq: t as u32,
                heads: hv as u32,
                v_dim: dv as u32,
            };
            let label = format!("hk={hk} hv={hv}");

            let convt = tensor(rt, &[tokens, ld], &conv);
            let (gt, bt) = (tensor(rt, &[b, t, hv], &g), tensor(rt, &[b, t, hv], &beta));
            let s0t = tensor(rt, &[b, hv, dk, dv], &s0);
            let (d_ot, dfint) = (tensor(rt, &[b, t, hv, dv], &d_o), tensor(rt, &[b, hv, dk, dv], &dfin));
            let ws = GdnTrainWorkspace::new(rt, dims).unwrap();

            // Dense reference on the repeated operands.
            let (q, k, v) = (
                tensor(rt, &[b, t, hv, dk], &qd),
                tensor(rt, &[b, t, hv, dk], &kd),
                tensor(rt, &[b, t, hv, dv], &vd),
            );
            let xd = GdnTrainInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &gt,
                beta: &bt,
                s0: Some(&s0t),
            };
            let (od, sfd, ckd) = (
                sentinel(rt, &[b, t, hv, dv]),
                sentinel(rt, &[b, hv, dk, dv]),
                sentinel(rt, &dims.checkpoint_shape()),
            );
            gdn_train_forward(rt, dims, xd, &od, Some(&sfd), &ckd).unwrap();
            let (dq, dkk, dvd) = (
                sentinel(rt, &[b, t, hv, dk]),
                sentinel(rt, &[b, t, hv, dk]),
                sentinel(rt, &[b, t, hv, dv]),
            );
            let (dgd, dbd, ds0d) = (
                sentinel(rt, &[b, t, hv]),
                sentinel(rt, &[b, t, hv]),
                sentinel(rt, &[b, hv, dk, dv]),
            );
            gdn_train_backward(
                rt,
                dims,
                xd,
                &ckd,
                &d_ot,
                Some(&dfint),
                &ws,
                GdnTrainGrads {
                    dq: &dq,
                    dk: &dkk,
                    dv: &dvd,
                    dg: &dgd,
                    dbeta: &dbd,
                    ds0: Some(&ds0d),
                },
            )
            .unwrap();

            // Windowed.
            macro_rules! win {
                ($t:expr, $off:expr) => {
                    GdnWindow {
                        tensor: $t,
                        ld: ld as u32,
                        off: $off as u32,
                    }
                };
            }
            let xw = GdnTrainWindowedInputs {
                q: win!(&convt, PAD),
                k: win!(&convt, PAD + qw),
                v: win!(&convt, PAD + 2 * qw),
                key_heads: hk as u32,
                g: &gt,
                beta: &bt,
                s0: Some(&s0t),
            };
            let (ow, sfw, ckw) = (
                sentinel(rt, &[b, t, hv, dv]),
                sentinel(rt, &[b, hv, dk, dv]),
                sentinel(rt, &dims.checkpoint_shape()),
            );
            gdn_train_forward_windowed(rt, dims, xw, &ow, Some(&sfw), &ckw).unwrap();
            let dconv = sentinel(rt, &[tokens, ld]);
            let (dgw, dbw, ds0w) = (
                sentinel(rt, &[b, t, hv]),
                sentinel(rt, &[b, t, hv]),
                sentinel(rt, &[b, hv, dk, dv]),
            );
            gdn_train_backward_windowed(
                rt,
                dims,
                xw,
                &ckw,
                &d_ot,
                Some(&dfint),
                &ws,
                GdnTrainWindowedGrads {
                    dq: win!(&dconv, PAD),
                    dk: win!(&dconv, PAD + qw),
                    dv: win!(&dconv, PAD + 2 * qw),
                    dg: &dgw,
                    dbeta: &dbw,
                    ds0: Some(&ds0w),
                },
            )
            .unwrap();

            let same = |name: &str, a: &Tensor, bb: &Tensor| {
                let (x, y) = (a.read_f32().unwrap(), bb.read_f32().unwrap());
                assert_eq!(x.len(), y.len(), "{label} {name}: length");
                if let Some(i) = (0..x.len()).find(|&i| x[i].to_bits() != y[i].to_bits()) {
                    panic!("{label} {name}[{i}]: windowed {} dense {}", x[i], y[i]);
                }
            };
            same("o", &ow, &od);
            same("final state", &sfw, &sfd);
            same("ckpt", &ckw, &ckd);
            same("dg", &dgw, &dgd);
            same("dbeta", &dbw, &dbd);
            same("ds0", &ds0w, &ds0d);
            let dc = dconv.read_f32().unwrap();
            let (dqh, dkh, dvh) = (dq.read_f32().unwrap(), dkk.read_f32().unwrap(), dvd.read_f32().unwrap());
            for tok in 0..tokens {
                let row = &dc[tok * ld..(tok + 1) * ld];
                assert!(
                    row[..PAD].iter().all(|&x| x == SENTINEL),
                    "{label}: the pad columns were written"
                );
                for (name, off, dense) in [("dq", PAD, &dqh), ("dk", PAD + qw, &dkh)] {
                    for hkh in 0..hk {
                        for i in 0..dk {
                            // Value heads' gradients added in value-head order.
                            let at = |j: usize| dense[(tok * hv + hkh * r + j) * dk + i];
                            let want = (1..r).fold(at(0), |acc, j| acc + at(j));
                            let got = row[off + hkh * dk + i];
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "{label} {name} tok {tok} head {hkh} [{i}]"
                            );
                        }
                    }
                }
                for h in 0..hv {
                    for j in 0..dv {
                        let got = row[PAD + 2 * qw + h * dv + j];
                        let want = dvh[(tok * hv + h) * dv + j];
                        assert_eq!(got.to_bits(), want.to_bits(), "{label} dv tok {tok} head {h} [{j}]");
                    }
                }
            }
            eprintln!("{label}: windowed == repeated dense, bit for bit");
        }
    });
}

/// A window that does not fit its matrix, or windows of one output matrix
/// that overlap, or an output window over an input, is refused by name.
#[test]
fn the_host_refuses_bad_windows() {
    use tessl::gdn_train::{gdn_train_backward_windowed, gdn_train_forward_windowed};
    use tessl::gdn_train::{GdnTrainWindowedGrads, GdnTrainWindowedInputs, GdnWindow};
    with_gpu(|rt| {
        let dims = GdnTrainDims {
            batch: 1,
            seq: 5,
            heads: 4,
            v_dim: 32,
        };
        let z = |shape: &[usize]| rt.alloc_tensor_f32(shape).unwrap();
        // [q (1 key head) | k | v (4 heads)] = 128 + 128 + 128 columns.
        let ld = 2 * 128 + 4 * 32;
        let (conv, dconv) = (z(&[5, ld]), z(&[5, ld]));
        let (g, beta) = (z(&[1, 5, 4]), z(&[1, 5, 4]));
        macro_rules! w {
            ($t:expr, $off:expr) => {
                GdnWindow {
                    tensor: $t,
                    ld: ld as u32,
                    off: $off,
                }
            };
        }
        let x = GdnTrainWindowedInputs {
            q: w!(&conv, 0),
            k: w!(&conv, 128),
            v: w!(&conv, 256),
            key_heads: 1,
            g: &g,
            beta: &beta,
            s0: None,
        };
        let (o, ckpt) = (z(&[1, 5, 4, 32]), z(&dims.checkpoint_shape()));
        let refused = |r: Result<(), String>, needle: &str| {
            let e = r.expect_err(needle);
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        };
        gdn_train_forward_windowed(rt, dims, x, &o, None, &ckpt).expect("valid windows");
        refused(
            gdn_train_forward_windowed(rt, dims, GdnTrainWindowedInputs { key_heads: 3, ..x }, &o, None, &ckpt),
            "not a multiple of 3 key heads",
        );
        refused(
            gdn_train_forward_windowed(rt, dims, GdnTrainWindowedInputs { key_heads: 0, ..x }, &o, None, &ckpt),
            "0 key heads",
        );
        refused(
            gdn_train_forward_windowed(
                rt,
                dims,
                GdnTrainWindowedInputs { v: w!(&conv, 300), ..x },
                &o,
                None,
                &ckpt,
            ),
            "does not fit its row stride",
        );
        let small = z(&[4, ld]);
        refused(
            gdn_train_forward_windowed(
                rt,
                dims,
                GdnTrainWindowedInputs { q: w!(&small, 0), ..x },
                &o,
                None,
                &ckpt,
            ),
            "q holds",
        );
        let ws = GdnTrainWorkspace::new(rt, dims).unwrap();
        let (d_o, dg, db) = (z(&[1, 5, 4, 32]), z(&[1, 5, 4]), z(&[1, 5, 4]));
        let bwd = |dq: GdnWindow<'_>, dk: GdnWindow<'_>, dv: GdnWindow<'_>| {
            gdn_train_backward_windowed(
                rt,
                dims,
                x,
                &ckpt,
                &d_o,
                None,
                &ws,
                GdnTrainWindowedGrads {
                    dq,
                    dk,
                    dv,
                    dg: &dg,
                    dbeta: &db,
                    ds0: None,
                },
            )
        };
        bwd(w!(&dconv, 0), w!(&dconv, 128), w!(&dconv, 256)).expect("disjoint windows of one matrix");
        refused(
            bwd(w!(&dconv, 0), w!(&dconv, 64), w!(&dconv, 256)),
            "outputs dq and dk overlap",
        );
        refused(
            bwd(w!(&dconv, 0), w!(&dconv, 128), w!(&dconv, 200)),
            "outputs dk and dv overlap",
        );
        refused(
            bwd(w!(&conv, 0), w!(&dconv, 128), w!(&dconv, 256)),
            "output dq overlaps input",
        );
    });
}

// ---------------------------------------------------------------------------
// The chunked kernels against the token-sequential ones
// ---------------------------------------------------------------------------

/// The tolerance between the two kernels, as a fraction of the sequential
/// output's largest magnitude. Both are f32 with `precise::` exponentials and
/// square roots; they differ only in summation order and association. The
/// sequential kernel adds a value column's 128 key-row products one simdgroup
/// at a time and walks 64 dependent steps per chunk; the chunked one forms the
/// same sums as 8-wide matrix products over a 64-row triangular solve. A sum
/// of n terms in f32 is off by at most n/2 ulp (6e-8 relative to the sum of
/// magnitudes), and the longest chain here is the solve's 63 forward
/// substitutions of 128-term dot products, so about 1e-5 worst case. The
/// bound is checked, not assumed: this test prints every case's worst ratio
/// (`chunked vs sequential worst ...`), and on the M5 Pro they were 2.7e-7 to
/// 4.4e-6 (T = 1 to 2048, grouped heads included; the log of that run is not
/// committed, run the test with `--nocapture` to reproduce). 1e-5 is the
/// worst-case bound: a defect in either kernel's algebra moves outputs by
/// O(1e-2) of their magnitude or more, and the f64 checks hold each kernel to
/// 1e-4 independently.
const PARITY_TOL: f64 = 1e-5;

fn parity(label: &str, chunked: &[f32], oracle: &[f32]) -> f64 {
    assert_eq!(chunked.len(), oracle.len(), "{label}: length");
    let peak = oracle.iter().fold(0.0f64, |m, x| m.max(f64::from(x.abs()))).max(1e-30);
    let mut worst = (0.0f64, 0usize);
    for (i, (&c, &o)) in chunked.iter().zip(oracle).enumerate() {
        assert!(c.is_finite() && c != SENTINEL, "{label}[{i}]: chunked wrote {c}");
        let e = (f64::from(c) - f64::from(o)).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    let rel = worst.0 / peak;
    assert!(
        rel <= PARITY_TOL,
        "{label}: chunked and sequential differ by {:.3e} at {} (chunked {} sequential {}), {rel:.2e} of max|seq| {peak:.3e}",
        worst.0,
        worst.1,
        chunked[worst.1],
        oracle[worst.1]
    );
    rel
}

/// Every output of one forward and backward, in a fixed order.
struct Outputs {
    names: Vec<&'static str>,
    data: Vec<Vec<f32>>,
}

/// Forward on `fwd_kernel`, backward on `bwd_kernel` (the checkpoints are the
/// same layout and content, so the two compose), dense operands.
fn run_dense(rt: &Arc<GpuRuntime>, c: &Case, fwd_kernel: GdnTrainKernel, bwd_kernel: GdnTrainKernel) -> Outputs {
    let Shape { b, t, h, dv, .. } = c.s;
    let dims = GdnTrainDims {
        batch: b as u32,
        seq: t as u32,
        heads: h as u32,
        v_dim: dv as u32,
    };
    let (q, k, v) = (
        tensor(rt, &[b, t, h, 128], &c.q),
        tensor(rt, &[b, t, h, 128], &c.k),
        tensor(rt, &[b, t, h, dv], &c.v),
    );
    let (g, beta) = (tensor(rt, &[b, t, h], &c.g), tensor(rt, &[b, t, h], &c.beta));
    let s0 = c.s0.as_ref().map(|s| tensor(rt, &[b, h, 128, dv], s));
    let x = GdnTrainInputs {
        q: &q,
        k: &k,
        v: &v,
        g: &g,
        beta: &beta,
        s0: s0.as_ref(),
    };
    let (o, sfin, ckpt) = (
        sentinel(rt, &[b, t, h, dv]),
        sentinel(rt, &[b, h, 128, dv]),
        sentinel(rt, &dims.checkpoint_shape()),
    );
    gdn_train_forward_with(fwd_kernel, rt, dims, x, &o, Some(&sfin), &ckpt).unwrap();
    let d_o = tensor(rt, &[b, t, h, dv], &c.d_o);
    let dfin = c.dfin.as_ref().map(|d| tensor(rt, &[b, h, 128, dv], d));
    let ws = GdnTrainWorkspace::new_for(rt, dims, bwd_kernel).unwrap();
    let (dq, dk, dvv) = (
        sentinel(rt, &[b, t, h, 128]),
        sentinel(rt, &[b, t, h, 128]),
        sentinel(rt, &[b, t, h, dv]),
    );
    let (dg, db) = (sentinel(rt, &[b, t, h]), sentinel(rt, &[b, t, h]));
    let ds0 = c.s0.as_ref().map(|_| sentinel(rt, &[b, h, 128, dv]));
    gdn_train_backward(
        rt,
        dims,
        x,
        &ckpt,
        &d_o,
        dfin.as_ref(),
        &ws,
        GdnTrainGrads {
            dq: &dq,
            dk: &dk,
            dv: &dvv,
            dg: &dg,
            dbeta: &db,
            ds0: ds0.as_ref(),
        },
    )
    .unwrap();
    let mut out = Outputs {
        names: vec!["o", "final state", "ckpt", "dq", "dk", "dv", "dg", "dbeta"],
        data: [&o, &sfin, &ckpt, &dq, &dk, &dvv, &dg, &db]
            .iter()
            .map(|t| t.read_f32().unwrap())
            .collect(),
    };
    if let Some(ds0) = ds0 {
        out.names.push("ds0");
        out.data.push(ds0.read_f32().unwrap());
    }
    out
}

fn assert_parity(label: &str, chunked: &Outputs, oracle: &Outputs) -> f64 {
    let mut worst = 0.0f64;
    for ((name, c), o) in chunked.names.iter().zip(&chunked.data).zip(&oracle.data) {
        worst = worst.max(parity(&format!("{label} {name}"), c, o));
    }
    eprintln!("{label}: chunked vs sequential worst {worst:.2e} of max|seq|");
    worst
}

/// The chunked kernels' every output (o, final state, checkpoints, dq, dk, dv,
/// dg, dbeta, ds0) equals the sequential kernels' within `PARITY_TOL`, across
/// the chunk edges, with and without an initial state and a final-state
/// gradient, at Qwen3.5-2B's shapes (16 heads of 128 by 128, T = 2048, B = 1),
/// and with the two kernels mixed across the forward/backward seam.
#[test]
fn the_chunked_kernels_match_the_sequential_ones() {
    use GdnTrainKernel::{Chunked, Sequential};
    with_gpu(|rt| {
        for (b, t, h, dv, s0, dfin) in [
            (1, 1, 2, 16, false, false),
            (2, 63, 3, 32, true, false),
            (1, 64, 2, 48, false, true),
            (2, 65, 3, 32, true, true),
            (1, 130, 4, 16, true, true),
            (1, 2048, 16, 128, false, false),
            (1, 2048, 16, 128, true, true),
        ] {
            let c = Case::new(b, t, h, dv, 900 + t as u64, s0, dfin);
            let label = format!("B={b} T={t} H={h} Dv={dv} s0={s0} dfin={dfin}");
            let oracle = run_dense(rt, &c, Sequential, Sequential);
            assert_parity(&label, &run_dense(rt, &c, Chunked, Chunked), &oracle);
            if t <= 130 {
                assert_parity(
                    &format!("{label} (chunked fwd, sequential bwd)"),
                    &run_dense(rt, &c, Chunked, Sequential),
                    &oracle,
                );
                assert_parity(
                    &format!("{label} (sequential fwd, chunked bwd)"),
                    &run_dense(rt, &c, Sequential, Chunked),
                    &oracle,
                );
            }
        }
    });
}

/// Grouped heads (several value heads per key head), through windows of one
/// conv-layout matrix: the chunked forward and backward equal the sequential
/// ones within `PARITY_TOL`, dq and dk (summed over each key head's value
/// heads) included.
#[test]
fn the_chunked_kernels_match_the_sequential_ones_on_grouped_windows() {
    use tessl::gdn_train::{
        gdn_train_backward_windowed, gdn_train_forward_windowed_with, GdnTrainWindowedGrads, GdnTrainWindowedInputs,
        GdnWindow,
    };
    const PAD: usize = 8;
    with_gpu(|rt| {
        for (hk, hv, t) in [(2usize, 4usize, 130usize), (1, 3, 65), (4, 16, 200)] {
            let (b, dv, dk) = (2usize, 32usize, 128usize);
            let (tokens, qw, vw) = (b * t, hk * dk, hv * dv);
            let ld = PAD + 2 * qw + vw;
            let seed = 700 + hv as u64;
            let conv = tensor(rt, &[tokens, ld], &random_f32(tokens * ld, seed));
            let n = tokens * hv;
            let g: Vec<f32> = random_f32(n, seed + 1)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect();
            let beta: Vec<f32> = random_f32(n, seed + 2).iter().map(|&x| 0.5 + 0.45 * x).collect();
            let (gt, bt) = (tensor(rt, &[b, t, hv], &g), tensor(rt, &[b, t, hv], &beta));
            let s0 = tensor(rt, &[b, hv, dk, dv], &random_f32(b * hv * dk * dv, seed + 3));
            let d_o = tensor(rt, &[b, t, hv, dv], &random_f32(n * dv, seed + 4));
            let dfin = tensor(rt, &[b, hv, dk, dv], &random_f32(b * hv * dk * dv, seed + 5));
            let dims = GdnTrainDims {
                batch: b as u32,
                seq: t as u32,
                heads: hv as u32,
                v_dim: dv as u32,
            };
            macro_rules! win {
                ($t:expr, $off:expr) => {
                    GdnWindow {
                        tensor: $t,
                        ld: ld as u32,
                        off: $off as u32,
                    }
                };
            }
            let xw = GdnTrainWindowedInputs {
                q: win!(&conv, PAD),
                k: win!(&conv, PAD + qw),
                v: win!(&conv, PAD + 2 * qw),
                key_heads: hk as u32,
                g: &gt,
                beta: &bt,
                s0: Some(&s0),
            };
            let run = |kernel: GdnTrainKernel| {
                let (o, sf, ck) = (
                    sentinel(rt, &[b, t, hv, dv]),
                    sentinel(rt, &[b, hv, dk, dv]),
                    sentinel(rt, &dims.checkpoint_shape()),
                );
                gdn_train_forward_windowed_with(kernel, rt, dims, xw, &o, Some(&sf), &ck).unwrap();
                let dconv = sentinel(rt, &[tokens, ld]);
                let (dg, db, ds0) = (
                    sentinel(rt, &[b, t, hv]),
                    sentinel(rt, &[b, t, hv]),
                    sentinel(rt, &[b, hv, dk, dv]),
                );
                // The chunked workspace sized by key heads (the production
                // constructor for a grouped model); never larger than the dense one.
                let ws = if kernel == GdnTrainKernel::Chunked && GDN_TRAIN_PRODUCTION == GdnTrainKernel::Chunked {
                    assert!(
                        GdnTrainWorkspace::bytes_for_grouped(dims, hk as u32) < GdnTrainWorkspace::bytes_for(dims)
                            || hk == hv
                    );
                    GdnTrainWorkspace::new_grouped(rt, dims, hk as u32).unwrap()
                } else {
                    GdnTrainWorkspace::new_for(rt, dims, kernel).unwrap()
                };
                gdn_train_backward_windowed(
                    rt,
                    dims,
                    xw,
                    &ck,
                    &d_o,
                    Some(&dfin),
                    &ws,
                    GdnTrainWindowedGrads {
                        dq: win!(&dconv, PAD),
                        dk: win!(&dconv, PAD + qw),
                        dv: win!(&dconv, PAD + 2 * qw),
                        dg: &dg,
                        dbeta: &db,
                        ds0: Some(&ds0),
                    },
                )
                .unwrap();
                // The window columns only: the pad is never written.
                let dc = dconv.read_f32().unwrap();
                let cols = |from: usize, to: usize| -> Vec<f32> {
                    (0..tokens)
                        .flat_map(|r| dc[r * ld + from..r * ld + to].to_vec())
                        .collect()
                };
                assert!(
                    (0..tokens).all(|r| dc[r * ld..r * ld + PAD].iter().all(|&x| x == SENTINEL)),
                    "{kernel:?}: the pad columns were written"
                );
                Outputs {
                    names: vec!["o", "final state", "ckpt", "dq", "dk", "dv", "dg", "dbeta", "ds0"],
                    data: vec![
                        o.read_f32().unwrap(),
                        sf.read_f32().unwrap(),
                        ck.read_f32().unwrap(),
                        cols(PAD, PAD + qw),
                        cols(PAD + qw, PAD + 2 * qw),
                        cols(PAD + 2 * qw, ld),
                        dg.read_f32().unwrap(),
                        db.read_f32().unwrap(),
                        ds0.read_f32().unwrap(),
                    ],
                }
            };
            let oracle = run(GdnTrainKernel::Sequential);
            assert_parity(
                &format!("hk={hk} hv={hv} T={t}"),
                &run(GdnTrainKernel::Chunked),
                &oracle,
            );
            if hk > 1 && GDN_TRAIN_PRODUCTION == GdnTrainKernel::Chunked {
                // A workspace sized for fewer key heads than the call has is refused.
                let small = GdnTrainWorkspace::new_grouped(rt, dims, 1).unwrap();
                let ck = sentinel(rt, &dims.checkpoint_shape());
                let dconv = sentinel(rt, &[tokens, ld]);
                let (dg, db, ds0) = (
                    sentinel(rt, &[b, t, hv]),
                    sentinel(rt, &[b, t, hv]),
                    sentinel(rt, &[b, hv, dk, dv]),
                );
                let e = gdn_train_backward_windowed(
                    rt,
                    dims,
                    xw,
                    &ck,
                    &d_o,
                    Some(&dfin),
                    &small,
                    GdnTrainWindowedGrads {
                        dq: win!(&dconv, PAD),
                        dk: win!(&dconv, PAD + qw),
                        dv: win!(&dconv, PAD + 2 * qw),
                        dg: &dg,
                        dbeta: &db,
                        ds0: Some(&ds0),
                    },
                )
                .expect_err("a workspace for one key head must refuse two");
                assert!(e.contains("key heads"), "{e}");
            }
        }
    });
}

/// "Once" means once per call, per (token, key head): one `gdn_train_c_norm`
/// dispatch in the forward and one in the backward (which rebuilds the
/// normalized q and k into its workspace instead of keeping the forward's, to
/// keep memory down), over a grid of chunks x batch x *key* heads, however
/// many value heads and 16-column slices share them. The backward's final
/// l2norm-backward pass (`gdn_train_c_finish`) re-derives each row's two
/// norms from the raw q and k it reads, once per (token, key head) as well.
/// The token-sequential kernels normalize inside every one of the Dv / 16
/// slices' threadgroups, and the backward again per slice, and never dispatch
/// the norm stage.
#[test]
fn q_and_k_are_normalized_once_per_call_per_key_head() {
    common::in_traced_child("q_and_k_are_normalized_once_per_call_per_key_head", || {
        let count = tessl::runtime::traced_kernel_count;
        with_gpu(|rt| {
            for (hv, dv) in [(1usize, 16usize), (4, 128), (3, 64)] {
                let c = Case::new(1, 100, hv, dv, 11, false, false);
                let norm = count("gdn_train_c_norm");
                let before = |n: &str| count(n);
                let (prep, fwd, bwd) = (
                    before("gdn_train_c_prep"),
                    before("gdn_train_c_fwd"),
                    before("gdn_train_c_bwd_a"),
                );
                run_dense(rt, &c, GdnTrainKernel::Chunked, GdnTrainKernel::Chunked);
                assert_eq!(
                    count("gdn_train_c_norm") - norm,
                    2,
                    "hv={hv} dv={dv}: one norm per forward and per backward"
                );
                assert_eq!(
                    count("gdn_train_c_prep") - prep,
                    2,
                    "hv={hv} dv={dv}: prep per forward and per backward"
                );
                assert_eq!(count("gdn_train_c_fwd") - fwd, 1);
                assert_eq!(count("gdn_train_c_bwd_a") - bwd, 1);
                let norm = count("gdn_train_c_norm");
                run_dense(rt, &c, GdnTrainKernel::Sequential, GdnTrainKernel::Sequential);
                assert_eq!(
                    count("gdn_train_c_norm"),
                    norm,
                    "the sequential kernels dispatch no norm stage"
                );
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Adversarial inputs against the f64 reference
// ---------------------------------------------------------------------------

/// One adversarial problem: B = 1 (T, H, Dv as given), gates and operand
/// scales from closures over (flat index, a uniform draw in [-1, 1)).
struct Adv {
    name: String,
    t: usize,
    h: usize,
    hk: usize,
    dv: usize,
    s0: bool,
    dfin: bool,
    g: Box<dyn Fn(usize, f32) -> f32>,
    beta: Box<dyn Fn(usize, f32) -> f32>,
    q_scale: f32,
    k_scale: f32,
    /// Fraction of every key that is one shared vector: 1.0 makes all keys
    /// identical, so the chunk's (I + A) solve is as ill-conditioned as it gets.
    k_corr: f32,
    /// Whether the problem itself is ill-conditioned, so that the f64
    /// reference bound is not asserted (the ratio to the sequential kernel is).
    ill_conditioned: bool,
}

fn adv(name: &str, t: usize) -> Adv {
    Adv {
        name: name.into(),
        t,
        h: 2,
        hk: 2,
        dv: 32,
        s0: true,
        dfin: true,
        g: Box::new(|_, x| -0.75 * (x + 1.0) - 1e-3),
        beta: Box::new(|_, x| 0.5 + 0.45 * x),
        q_scale: 1.0,
        k_scale: 1.0,
        k_corr: 0.0,
        ill_conditioned: false,
    }
}

const ADV_OUTPUTS: [&str; 8] = ["o", "final state", "dq", "dk", "dv", "dg", "dbeta", "ds0"];

/// Each output's worst error as a fraction of that output's own largest
/// reference magnitude, floored at 1e-30 (so an output whose reference is
/// exactly 0 must come back as ~0, not as 1e-7 of rounding noise). The measure
/// is per output because the outputs differ in scale by many orders at strong
/// decay: dg of a problem with g = -50 is O(e^-50) throughout, and a kernel
/// that is right to 1e-7 of the *largest* gradient is wrong by 1.0 relative to
/// dg's own scale.
fn adv_errors(rt: &Arc<GpuRuntime>, p: &Adv, kernel: GdnTrainKernel, seed: u64) -> Vec<f64> {
    let (b, t, h, hk, dv) = (1usize, p.t, p.h, p.hk, p.dv);
    let r = h / hk;
    let n = b * t * h;
    let q: Vec<f32> = random_f32(n * 128, seed).iter().map(|x| x * p.q_scale).collect();
    let shared = random_f32(128, seed + 100);
    let k: Vec<f32> = random_f32(n * 128, seed + 1)
        .iter()
        .enumerate()
        .map(|(i, x)| p.k_scale * (p.k_corr * shared[i % 128] + (1.0 - p.k_corr) * x))
        .collect();
    let v = random_f32(n * dv, seed + 2);
    let g: Vec<f32> = random_f32(n, seed + 3)
        .iter()
        .enumerate()
        .map(|(i, &x)| (p.g)(i, x))
        .collect();
    let beta: Vec<f32> = random_f32(n, seed + 4)
        .iter()
        .enumerate()
        .map(|(i, &x)| (p.beta)(i, x))
        .collect();
    let s0 = p.s0.then(|| random_f32(b * h * 128 * dv, seed + 5));
    let d_o = random_f32(n * dv, seed + 6);
    let dfin = p.dfin.then(|| random_f32(b * h * 128 * dv, seed + 7));
    let dims = GdnTrainDims {
        batch: 1,
        seq: t as u32,
        heads: h as u32,
        v_dim: dv as u32,
    };
    // Grouped problems hand the kernels one key head per `r` value heads; the
    // f64 reference gets the repeated operands, as transformers does.
    let qk_small = |x: &[f32]| -> Vec<f32> {
        (0..t * hk)
            .flat_map(|row| x[(row / hk * h + row % hk * r) * 128..][..128].to_vec())
            .collect()
    };
    let (q_in, k_in) = if r == 1 {
        (q.clone(), k.clone())
    } else {
        (qk_small(&q), qk_small(&k))
    };
    let (qt, kt) = (tensor(rt, &[1, t, hk, 128], &q_in), tensor(rt, &[1, t, hk, 128], &k_in));
    let vt = tensor(rt, &[1, t, h, dv], &v);
    let (gt, bt) = (tensor(rt, &[b, t, h], &g), tensor(rt, &[b, t, h], &beta));
    let s0t = s0.as_ref().map(|s| tensor(rt, &[b, h, 128, dv], s));
    let x = tessl::gdn_train::GdnTrainWindowedInputs {
        q: tessl::gdn_train::GdnWindow::dense(&qt, (hk * 128) as u32),
        k: tessl::gdn_train::GdnWindow::dense(&kt, (hk * 128) as u32),
        v: tessl::gdn_train::GdnWindow::dense(&vt, (h * dv) as u32),
        key_heads: hk as u32,
        g: &gt,
        beta: &bt,
        s0: s0t.as_ref(),
    };
    let (o, sfin, ckpt) = (
        rt.alloc_tensor_f32(&[b, t, h, dv]).unwrap(),
        rt.alloc_tensor_f32(&[b, h, 128, dv]).unwrap(),
        rt.alloc_tensor_f32(&dims.checkpoint_shape()).unwrap(),
    );
    tessl::gdn_train::gdn_train_forward_windowed_with(kernel, rt, dims, x, &o, Some(&sfin), &ckpt).unwrap();
    let dot = tensor(rt, &[b, t, h, dv], &d_o);
    let dft = dfin.as_ref().map(|d| tensor(rt, &[b, h, 128, dv], d));
    let ws = GdnTrainWorkspace::new_for(rt, dims, kernel).unwrap();
    let (dq, dk, dvv) = (
        rt.alloc_tensor_f32(&[1, t, hk, 128]).unwrap(),
        rt.alloc_tensor_f32(&[1, t, hk, 128]).unwrap(),
        rt.alloc_tensor_f32(&[b, t, h, dv]).unwrap(),
    );
    let (dg, db) = (
        rt.alloc_tensor_f32(&[b, t, h]).unwrap(),
        rt.alloc_tensor_f32(&[b, t, h]).unwrap(),
    );
    let ds0 = p.s0.then(|| rt.alloc_tensor_f32(&[b, h, 128, dv]).unwrap());
    tessl::gdn_train::gdn_train_backward_windowed(
        rt,
        dims,
        x,
        &ckpt,
        &dot,
        dft.as_ref(),
        &ws,
        tessl::gdn_train::GdnTrainWindowedGrads {
            dq: tessl::gdn_train::GdnWindow::dense(&dq, (hk * 128) as u32),
            dk: tessl::gdn_train::GdnWindow::dense(&dk, (hk * 128) as u32),
            dv: tessl::gdn_train::GdnWindow::dense(&dvv, (h * dv) as u32),
            dg: &dg,
            dbeta: &db,
            ds0: ds0.as_ref(),
        },
    )
    .unwrap();
    // The f64 reference on the repeated (dense) operands.
    let rep = |x: &[f32], w: usize| -> Vec<f32> {
        (0..t * h)
            .flat_map(|row| x[(row / h * hk + row % h / r) * w..][..w].to_vec())
            .collect()
    };
    let (qd, kd) = (rep(&q_in, 128), rep(&k_in, 128));
    let s = Shape { b, t, h, dk: 128, dv };
    let (q64, k64, v64, g64, b64) = (f64s(&qd), f64s(&kd), f64s(&v), f64s(&g), f64s(&beta));
    let s064 = s0.as_ref().map(|x| f64s(x));
    let inp = Inputs {
        s,
        q: &q64,
        k: &k64,
        v: &v64,
        g: &g64,
        beta: &b64,
        s0: s064.as_deref(),
    };
    let (wo, wf) = gdn_train_f64(&inp);
    let df64 = dfin.as_ref().map(|d| f64s(d));
    let w = gdn_train_bwd_f64(&inp, &f64s(&d_o), df64.as_deref());
    // dq, dk of a grouped problem are the sums over each key head's value heads.
    let fold = |x: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; t * hk * 128];
        for row in 0..t * h {
            let (tok, hv) = (row / h, row % h);
            for i in 0..128 {
                out[(tok * hk + hv / r) * 128 + i] += x[row * 128 + i];
            }
        }
        out
    };
    let rel = |got: &[f32], want: &[f64]| -> f64 {
        let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
        got.iter().zip(want).fold(0.0f64, |m, (&g, &r)| {
            if g.is_finite() {
                m.max((f64::from(g) - r).abs())
            } else {
                f64::INFINITY
            }
        }) / peak
    };
    let mut errs = vec![
        rel(&o.read_f32().unwrap(), &wo),
        rel(&sfin.read_f32().unwrap(), &wf),
        rel(&dq.read_f32().unwrap(), &fold(&w.dq)),
        rel(&dk.read_f32().unwrap(), &fold(&w.dk)),
        rel(&dvv.read_f32().unwrap(), &w.dv),
        rel(&dg.read_f32().unwrap(), &w.dg),
        rel(&db.read_f32().unwrap(), &w.dbeta),
    ];
    if let Some(d) = &ds0 {
        errs.push(rel(&d.read_f32().unwrap(), &w.ds0));
    }
    errs
}

/// The bound for every output of every adversarial problem, each as a fraction
/// of that output's own largest reference magnitude: the same 1e-4 the other
/// f64 checks use.
const ADV_F64_TOL: f64 = 1e-4;
/// The chunked kernels must be no worse than the sequential ones by more than
/// this factor on any output (plus `ADV_RATIO_FLOOR`, below which both are
/// rounding): they reorder the same f32 sums, so a larger ratio is a
/// cancellation the other kernel does not have.
const ADV_RATIO: f64 = 10.0;
const ADV_RATIO_FLOOR: f64 = 2e-6;

fn run_adv(rt: &Arc<GpuRuntime>, cases: &[Adv], seed: u64) {
    for (i, p) in cases.iter().enumerate() {
        let seed = seed + 17 * i as u64;
        let seq = adv_errors(rt, p, GdnTrainKernel::Sequential, seed);
        let chk = adv_errors(rt, p, GdnTrainKernel::Chunked, seed);
        let show = |e: &[f64]| {
            ADV_OUTPUTS
                .iter()
                .zip(e)
                .map(|(n, x)| format!("{n}={x:.1e}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!("ADV {:44} SEQ {}\nADV {:44} CHK {}", p.name, show(&seq), "", show(&chk));
        for ((name, s), c) in ADV_OUTPUTS.iter().zip(&seq).zip(&chk) {
            assert!(
                *c <= ADV_RATIO * s + ADV_RATIO_FLOOR,
                "{}: chunked {name} error {c:.2e} vs sequential {s:.2e} (allowed {ADV_RATIO}x + {ADV_RATIO_FLOOR:.0e}) of that output's max|ref|",
                p.name
            );
            if !p.ill_conditioned {
                assert!(*c <= ADV_F64_TOL, "{}: chunked {name} error {c:.2e} vs f64", p.name);
                assert!(*s <= ADV_F64_TOL, "{}: sequential {name} error {s:.2e} vs f64", p.name);
            }
        }
    }
}

/// The gate sweep that exposed a cancellation in dg: with strong decay every
/// gradient except the within-token diagonal is below the rounding of an O(1)
/// term, and a kernel that adds and subtracts that diagonal flushes dg to 0
/// (error 1.0 of dg's own scale at g = -50, where dg is O(e^-50)).
#[test]
fn dg_survives_strong_decay() {
    with_gpu(|rt| {
        let cases: Vec<Adv> = [-0.5f32, -2.0, -5.0, -10.0, -20.0, -50.0, -1e4]
            .iter()
            .map(|&g| Adv {
                g: Box::new(move |_, _| g),
                ..adv(&format!("g={g} T=200"), 200)
            })
            .collect();
        run_adv(rt, &cases, 3000);
    });
}

/// Chunk-edge lengths, degenerate gates, beta at its extremes, operand scales,
/// ill-conditioned solves and every s0 / dfin combination, against f64.
#[test]
fn the_kernels_survive_adversarial_inputs() {
    with_gpu(|rt| {
        let mut cases: Vec<Adv> = vec![];
        for t in [1, 63, 64, 65, 127, 129, 1023, 1025, 4097] {
            cases.push(adv(&format!("default T={t}"), t));
        }
        for (g, label) in [(0.0f32, "g=0"), (-1e-6, "g=-1e-6"), (-1e4, "g=-1e4")] {
            cases.push(Adv {
                g: Box::new(move |_, _| g),
                ..adv(label, 200)
            });
        }
        for (label, f) in [
            ("beta=0", Box::new(|_, _| 0.0f32) as Box<dyn Fn(usize, f32) -> f32>),
            ("beta=1", Box::new(|_, _| 1.0)),
            ("beta alternating 0/1", Box::new(|i, _| (i % 2) as f32)),
        ] {
            cases.push(Adv {
                beta: f,
                ..adv(label, 200)
            });
        }
        for (label, qs, ks) in [
            ("q,k x1e3", 1e3f32, 1e3f32),
            ("q,k x1e-3", 1e-3, 1e-3),
            ("k x1e-4", 1.0, 1e-4),
        ] {
            cases.push(Adv {
                q_scale: qs,
                k_scale: ks,
                ..adv(label, 200)
            });
        }
        for corr in [0.9f32, 0.99, 1.0] {
            cases.push(Adv {
                g: Box::new(|_, _| 0.0),
                beta: Box::new(|_, _| 1.0),
                k_corr: corr,
                ill_conditioned: true,
                ..adv(&format!("g=0 beta=1 k correlation {corr}"), 200)
            });
        }
        for (s0, dfin) in [(false, false), (true, false), (false, true), (true, true)] {
            cases.push(Adv {
                s0,
                dfin,
                ..adv(&format!("s0={s0} dfin={dfin} T=129"), 129)
            });
        }
        // Several value heads per key head, against f64 on the repeated operands.
        for (h, hk) in [(4usize, 2usize), (3, 1)] {
            cases.push(Adv {
                h,
                hk,
                ..adv(&format!("grouped hv={h} hk={hk} T=130"), 130)
            });
            cases.push(Adv {
                h,
                hk,
                g: Box::new(|_, _| -50.0),
                ..adv(&format!("grouped hv={h} hk={hk} g=-50 T=130"), 130)
            });
        }
        run_adv(rt, &cases, 4000);
    });
}
