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
    gdn_train_backward, gdn_train_forward, GdnTrainDims, GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace,
};
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
fn check(rt: &Arc<GpuRuntime>, c: &Case) -> f64 {
    let Shape { b, t, h, dv, .. } = c.s;
    let label = format!(
        "B={b} T={t} H={h} Dv={dv} s0={} dfin={}",
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
    gdn_train_forward(rt, dims, x, &o, Some(&sfin), &ckpt).unwrap_or_else(|e| panic!("{label}: {e}"));

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
    let ws = GdnTrainWorkspace::new(rt, dims).unwrap();
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
            check(rt, &Case::new(2, t, 3, 32, 100 + t as u64, s0, dfin));
        }
    });
}

#[test]
fn the_kernels_match_the_reference_at_qwen35_2b_heads() {
    with_gpu(|rt| {
        check(rt, &Case::new(1, 200, 16, 128, 7, false, false));
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
        assert_eq!(
            GdnTrainWorkspace::bytes_for(dims),
            4 * (2 * 2 * 64 * 128 * 16 + 2 * 2 * 10 * 128 + 2 * 2 * 10)
        );
    });
}
