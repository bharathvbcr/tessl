//! Qwen3.5 numeric kernels against host references.
//!
//! Each `credit` dispatches the named pipeline and compares the tensor it writes.
//! Storage kernels have no safe public launcher; they are bound from
//! `kernels/qwen35_train_storage.metal`.

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use crate::common::qwen35::{
    conv1d_silu_f64, gated_rms_norm_f64, gdn_f64, norm_rope_row_f64, score_row_f64, softplus, GdnProblem, GdnShape,
};
use crate::common::{buf, buf_u32, gelu_pytorch_tanh, random_f32, sigmoid, silu, tensor_f32};
use tessl::attn_prefill_bf16::{self, AttnProbTile};
use tessl::attn_train::{self, AttnTrainDims, AttnTrainGrads, AttnTrainWorkspace};
use tessl::dispatch::{self, dispatch_1d, dispatch_2d, dispatch_2d_tg};
use tessl::nn::{AttnDims, DecodeScratch};
use tessl::qwen35::{self, AttnShape, AttnTile, Cols, GdnDims, GdnScanSlice, GdnSimdVariant, StateIn};
use tessl::qwen35_bwd::{self, AttnQkvGrads};
use tessl::sq_norm::{self, SqWindow};
use tessl::{DType, GpuBuffer, GpuRuntime};

use super::Ledger;

const D: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f32 = 10_000.0;
const SENTINEL: f32 = -7.25e27;

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().copied().map(f64::from).collect()
}

fn take_f32(b: &GpuBuffer, n: usize) -> Vec<f32> {
    b.read_f32()[..n].to_vec()
}

fn take_bf16(b: &GpuBuffer, n: usize) -> Vec<f32> {
    b.contents_u16()[..n]
        .iter()
        .copied()
        .map(tessl::tensor::bf16_bits_to_f32)
        .collect()
}

fn gpu_bf16(rt: &Arc<GpuRuntime>, data: &[f32]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len().max(1) * 2).expect("bf16 alloc");
    let bits: Vec<u16> = data.iter().copied().map(tessl::tensor::f32_to_bf16_bits_gpu).collect();
    b.write_bf16_bits(&bits);
    b
}

fn widen_gpu(data: &[f32]) -> Vec<f32> {
    data.iter()
        .copied()
        .map(|x| tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits_gpu(x)))
        .collect()
}

fn round_gpu(x: f32) -> f32 {
    tessl::tensor::bf16_bits_to_f32(tessl::tensor::f32_to_bf16_bits_gpu(x))
}

fn close_attn(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}] {g}");
        let tol = 2e-4 * w.abs().max(1.0);
        let e = (f64::from(g) - w).abs();
        assert!(e <= tol, "{label}[{i}] got {g} want {w} tol {tol}");
    }
}

fn close_wide(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}] {g}");
        let tol = 2e-2 * w.abs().max(1.0);
        let e = (f64::from(g) - w).abs();
        assert!(e <= tol, "{label}[{i}] got {g} want {w} tol {tol}");
    }
}

fn close_peak(label: &str, got: &[f32], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}");
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3);
    let atol = rel * peak;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}] {g}");
        let e = (f64::from(g) - w).abs();
        assert!(e <= atol, "{label}[{i}] got {g} want {w} atol {atol}");
    }
}

fn close_elem(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}] {g}");
        let tol = 1e-5 * w.abs().max(1.0);
        let e = (f64::from(g) - w).abs();
        assert!(e <= tol, "{label}[{i}] got {g} want {w} tol {tol}");
    }
}

fn bits_eq(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label}[{i}] got {g} want {w}");
    }
}

fn sync(rt: &GpuRuntime) {
    rt.synchronize().expect("synchronize");
}

fn dense<'a>(b: &'a GpuBuffer, width: u32) -> Cols<'a> {
    Cols::dense(b, width)
}

fn out_f32(b: &GpuBuffer, width: u32) -> qwen35::OutCols<'_> {
    qwen35::OutCols {
        cols: dense(b, width),
        dtype: DType::F32,
    }
}

fn out_bf16(b: &GpuBuffer, width: u32) -> qwen35::OutCols<'_> {
    qwen35::OutCols {
        cols: dense(b, width),
        dtype: DType::BF16,
    }
}

// ----------------------------------------------------------------- attention

struct AttnShapeHost {
    b: usize,
    tq: usize,
    tkv: usize,
    h: usize,
    hkv: usize,
    d: usize,
    scale: f64,
    q_pos: usize,
    kv_pos: usize,
}

fn attn_o(s: &AttnShapeHost, q: &[f32], k: &[f32], v: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let group = s.h / s.hkv;
    let mut o = vec![0.0f64; s.b * s.tq * s.h * s.d];
    let mut lse = vec![0.0f64; s.b * s.h * s.tq];
    for b in 0..s.b {
        for h in 0..s.h {
            let hk = h / group;
            for t in 0..s.tq {
                let q_abs = s.q_pos + t;
                let qs = ((b * s.tq + t) * s.h + h) * s.d;
                let mut scores = Vec::with_capacity(s.tkv);
                let mut best = f64::NEG_INFINITY;
                for j in 0..s.tkv {
                    let k_abs = s.kv_pos + j;
                    if k_abs > q_abs {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let ks = ((b * s.tkv + j) * s.hkv + hk) * s.d;
                    let mut dot = 0.0f64;
                    for c in 0..s.d {
                        dot += f64::from(q[qs + c]) * f64::from(k[ks + c]);
                    }
                    let sc = dot * s.scale;
                    best = best.max(sc);
                    scores.push(sc);
                }
                let mut denom = 0.0f64;
                let mut acc = vec![0.0f64; s.d];
                if best.is_finite() {
                    for j in 0..s.tkv {
                        if !scores[j].is_finite() {
                            continue;
                        }
                        let p = (scores[j] - best).exp();
                        denom += p;
                        let vs = ((b * s.tkv + j) * s.hkv + hk) * s.d;
                        for c in 0..s.d {
                            acc[c] += p * f64::from(v[vs + c]);
                        }
                    }
                }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let os = ((b * s.tq + t) * s.h + h) * s.d;
                for c in 0..s.d {
                    o[os + c] = acc[c] * inv;
                }
                lse[(b * s.h + h) * s.tq + t] = if denom > 0.0 { best + denom.ln() } else { f64::NAN };
            }
        }
    }
    (o, lse)
}

fn attn_bwd_f64(q: &[f64], k: &[f64], v: &[f64], d_o: &[f64], s: &AttnShapeHost) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (b, t, hq, hkv, d) = (s.b, s.tq, s.h, s.hkv, s.d);
    let scale = s.scale;
    let group = hq / hkv;
    let mut o = vec![0.0f64; q.len()];
    let mut lse64 = vec![0.0f64; b * hq * t];
    for bi in 0..b {
        for h in 0..hq {
            let hk = h / group;
            for ti in 0..t {
                let qs = ((bi * t + ti) * hq + h) * d;
                let mut scores = vec![0.0f64; ti + 1];
                let mut best = f64::NEG_INFINITY;
                for j in 0..=ti {
                    let ks = ((bi * t + j) * hkv + hk) * d;
                    let dot: f64 = (0..d).map(|c| q[qs + c] * k[ks + c]).sum();
                    scores[j] = dot * scale;
                    best = best.max(scores[j]);
                }
                let z: f64 = scores.iter().map(|x| (x - best).exp()).sum();
                lse64[(bi * hq + h) * t + ti] = best + z.ln();
                for j in 0..=ti {
                    let p = (scores[j] - best).exp() / z;
                    let vs = ((bi * t + j) * hkv + hk) * d;
                    for c in 0..d {
                        o[qs + c] += p * v[vs + c];
                    }
                }
            }
        }
    }
    let (mut dq, mut dk, mut dv) = (vec![0.0f64; q.len()], vec![0.0f64; k.len()], vec![0.0f64; v.len()]);
    for bi in 0..b {
        for h in 0..hq {
            let hk = h / group;
            for ti in 0..t {
                let qs = ((bi * t + ti) * hq + h) * d;
                let l = lse64[(bi * hq + h) * t + ti];
                let dr: f64 = (0..d).map(|c| d_o[qs + c] * o[qs + c]).sum();
                for j in 0..=ti {
                    let ks = ((bi * t + j) * hkv + hk) * d;
                    let dot: f64 = (0..d).map(|c| q[qs + c] * k[ks + c]).sum();
                    let p = (dot * scale - l).exp();
                    let dp: f64 = (0..d).map(|c| d_o[qs + c] * v[ks + c]).sum();
                    let ds = p * (dp - dr) * scale;
                    for c in 0..d {
                        dv[ks + c] += p * d_o[qs + c];
                        dq[qs + c] += ds * k[ks + c];
                        dk[ks + c] += ds * q[qs + c];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

fn rms_rows(x: &[f32], w: &[f32], dim: usize, eps: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(dim) {
        let ss: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let inv = 1.0 / (ss / dim as f64 + eps).sqrt();
        for (v, wi) in row.iter().zip(w) {
            out.push(f64::from(*v) * inv * (1.0 + f64::from(*wi)));
        }
    }
    out
}

fn silu_grad(g: f64) -> f64 {
    let s = sigmoid(g);
    s * (1.0 + g * (1.0 - s))
}

fn softplus_grad(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else {
        sigmoid(x)
    }
}

fn rms_bwd(x: &[f64], w: &[f64], dy: &[f64], d: usize, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mut dx = vec![0.0; x.len()];
    let mut dw = vec![0.0; d];
    for (r, (xr, gr)) in x.chunks(d).zip(dy.chunks(d)).enumerate() {
        let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        let dot: f64 = (0..d).map(|j| gr[j] * (1.0 + w[j]) * xr[j]).sum();
        for j in 0..d {
            dx[r * d + j] = rstd * gr[j] * (1.0 + w[j]) - xr[j] * rstd.powi(3) * dot / d as f64;
            dw[j] += gr[j] * xr[j] * rstd;
        }
    }
    (dx, dw)
}

fn gated_bwd(x: &[f64], z: &[f64], w: &[f64], dy: &[f64], d: usize, eps: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut dx, mut dz, mut dw) = (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; d]);
    for u in 0..x.len() / d {
        let (xr, zr, gr) = (&x[u * d..(u + 1) * d], &z[u * d..(u + 1) * d], &dy[u * d..(u + 1) * d]);
        let rstd = 1.0 / (xr.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        let xn: Vec<f64> = xr.iter().map(|v| v * rstd).collect();
        let dxn: Vec<f64> = (0..d).map(|j| gr[j] * w[j] * silu(zr[j])).collect();
        let m: f64 = (0..d).map(|j| dxn[j] * xn[j]).sum::<f64>() / d as f64;
        for j in 0..d {
            dx[u * d + j] = rstd * (dxn[j] - xn[j] * m);
            dz[u * d + j] = gr[j] * w[j] * xn[j] * silu_grad(zr[j]);
            dw[j] += gr[j] * xn[j] * silu(zr[j]);
        }
    }
    (dx, dz, dw)
}

fn conv_bwd(x: &[f64], w: &[f64], dy: &[f64], b: usize, t: usize, c: usize, kw: usize) -> (Vec<f64>, Vec<f64>) {
    let hist = kw - 1;
    let (mut dx, mut dw) = (vec![0.0; b * t * c], vec![0.0; c * kw]);
    for bi in 0..b {
        for ti in 0..t {
            for ci in 0..c {
                let at = |j: usize| (bi * t + ti + j - hist) * c + ci;
                let taps: Vec<usize> = (0..kw).filter(|&j| ti + j >= hist).collect();
                let pre: f64 = taps.iter().map(|&j| w[ci * kw + j] * x[at(j)]).sum();
                let dpre = dy[(bi * t + ti) * c + ci] * silu_grad(pre);
                for &j in &taps {
                    dx[at(j)] += dpre * w[ci * kw + j];
                    dw[ci * kw + j] += dpre * x[at(j)];
                }
            }
        }
    }
    (dx, dw)
}

fn rope_cs(p: usize, rot: usize, pos: usize, theta: f64) -> (f64, f64) {
    let inv_freq = 1.0f32 / (theta as f32).powf((2 * p) as f32 / rot as f32);
    let angle = f64::from(pos as f32 * inv_freq);
    (angle.cos(), angle.sin())
}

fn norm_rope_bwd(
    x: &[f64],
    w: &[f64],
    g: &[f64],
    rot: usize,
    pos: usize,
    theta: f64,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let d = x.len();
    let rstd = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
    let mut dn = g.to_vec();
    let half = rot / 2;
    for p in 0..half {
        let (c, s) = rope_cs(p, rot, pos, theta);
        dn[p] = g[p] * c + g[p + half] * s;
        dn[p + half] = g[p + half] * c - g[p] * s;
    }
    let xn: Vec<f64> = x.iter().map(|v| v * rstd).collect();
    let dxn: Vec<f64> = dn.iter().zip(w).map(|(a, wi)| a * (1.0 + wi)).collect();
    let m = dxn.iter().zip(&xn).map(|(a, b)| a * b).sum::<f64>() / d as f64;
    let dx = dxn.iter().zip(&xn).map(|(a, b)| rstd * (a - b * m)).collect();
    let dw = dn.iter().zip(&xn).map(|(a, b)| a * b).collect();
    (dx, dw)
}

fn q8m_encode(x: f32, s: f32) -> u8 {
    if !(s > 0.0) {
        return 0;
    }
    let r = ((x.abs() / s).min(1.0).sqrt() * 127.0).round_ties_even();
    let mag = r as i32;
    let signed = if x < 0.0 { -mag } else { mag };
    signed as i8 as u8
}

fn q8m_decode(q: u8, s: f32) -> f32 {
    let t = f32::from(q as i8) / 127.0;
    s * t * t.abs()
}

// ----------------------------------------------------------------- elementwise

fn credit_conv(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(rt, "qwen35_conv1d_silu", &["qwen35_conv1d_silu"], || {
        let (b, t, c, kw) = (1usize, 3usize, 4usize, 4usize);
        let x = random_f32(b * t * c, 11);
        let w = random_f32(c * kw, 12);
        let (want, _) = conv1d_silu_f64(&x, &w, None, b, t, c, kw);
        let (xb, wb) = (buf(rt, &x), buf(rt, &w));
        let yb = buf(rt, &vec![SENTINEL; b * t * c]);
        qwen35::conv1d_silu(
            rt,
            dense(&xb, c as u32),
            &wb,
            kw as u32,
            StateIn::Zero,
            &yb,
            None,
            b as u32,
            t as u32,
            c as u32,
        )
        .unwrap();
        sync(rt);
        close_elem("conv1d_silu", &take_f32(&yb, want.len()), &want);
    });
}

fn credit_gates(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(rt, "qwen35_gdn_gates_f32", &["qwen35_gdn_gates_f32"], || {
        let (rows, heads) = (3usize, 2usize);
        let ld = heads * 2 + 1;
        let p = random_f32(rows * ld, 21);
        let a_off = 1usize;
        let b_off = 1 + heads;
        let a_log = random_f32(heads, 22);
        let dt = random_f32(heads, 23);
        let (pb, alb, dtb) = (buf(rt, &p), buf(rt, &a_log), buf(rt, &dt));
        let (gb, bb) = (
            buf(rt, &vec![SENTINEL; rows * heads]),
            buf(rt, &vec![SENTINEL; rows * heads]),
        );
        let logits = qwen35::GdnGateLogits {
            buf: &pb,
            ld: ld as u32,
            a_off: a_off as u32,
            b_off: b_off as u32,
        };
        let params = qwen35::GdnParams {
            a_log: &alb,
            dt_bias: &dtb,
        };
        qwen35::gdn_gates(rt, &logits, &params, &gb, &bb, rows as u32, heads as u32).unwrap();
        sync(rt);
        let mut wg = Vec::new();
        let mut wbeta = Vec::new();
        for r in 0..rows {
            for h in 0..heads {
                let a = f64::from(p[r * ld + a_off + h]);
                let b = f64::from(p[r * ld + b_off + h]);
                wg.push(-f64::from(a_log[h]).exp() * softplus(a + f64::from(dt[h])));
                wbeta.push(sigmoid(b));
            }
        }
        close_elem("gdn g", &take_f32(&gb, wg.len()), &wg);
        close_elem("gdn beta", &take_f32(&bb, wbeta.len()), &wbeta);
    });
}

struct GdnCase {
    dims: GdnDims,
    ld: u32,
    qkv: GpuBuffer,
    gates: GpuBuffer,
    a_log: GpuBuffer,
    dt: GpuBuffer,
    out: GpuBuffer,
    state: GpuBuffer,
    y: Vec<f64>,
    st: Vec<f64>,
    out_n: usize,
    st_n: usize,
}

fn gdn_case(rt: &Arc<GpuRuntime>, seed: u64) -> GdnCase {
    let (b, t, hk, hv, dv) = (1usize, 2usize, 1usize, 1usize, 32usize);
    let dims = GdnDims {
        batch: b as u32,
        seq: t as u32,
        k_heads: hk as u32,
        v_heads: hv as u32,
        v_dim: dv as u32,
    };
    let q = random_f32(b * t * hk * 128, seed);
    let k = random_f32(b * t * hk * 128, seed + 1);
    let v = random_f32(b * t * hv * dv, seed + 2);
    let a = random_f32(b * t * hv, seed + 3);
    let bts = random_f32(b * t * hv, seed + 4);
    let a_log = random_f32(hv, seed + 5);
    let dt = random_f32(hv, seed + 6);
    let (y, st) = gdn_f64(&GdnProblem {
        s: GdnShape { b, t, hk, hv, dv },
        q: &q,
        k: &k,
        v: &v,
        a: &a,
        b: &bts,
        a_log: &a_log,
        dt_bias: &dt,
        state0: None,
        snapshot: false,
    });
    let ld = (2 * hk * 128 + hv * dv) as u32;
    let rows = b * t;
    let mut packed = vec![0.0f32; rows * ld as usize];
    let mut gates = vec![0.0f32; rows * 2 * hv];
    for r in 0..rows {
        let q0 = r * hk * 128;
        packed[r * ld as usize..r * ld as usize + hk * 128].copy_from_slice(&q[q0..q0 + hk * 128]);
        packed[r * ld as usize + hk * 128..r * ld as usize + 2 * hk * 128].copy_from_slice(&k[q0..q0 + hk * 128]);
        let v0 = r * hv * dv;
        packed[r * ld as usize + 2 * hk * 128..r * ld as usize + 2 * hk * 128 + hv * dv]
            .copy_from_slice(&v[v0..v0 + hv * dv]);
        for h in 0..hv {
            gates[r * 2 * hv + h] = a[r * hv + h];
            gates[r * 2 * hv + hv + h] = bts[r * hv + h];
        }
    }
    let out_n = rows * hv * dv;
    let st_n = b * hv * 128 * dv;
    GdnCase {
        dims,
        ld,
        qkv: buf(rt, &packed),
        gates: buf(rt, &gates),
        a_log: buf(rt, &a_log),
        dt: buf(rt, &dt),
        out: buf(rt, &vec![SENTINEL; out_n]),
        state: buf(rt, &vec![SENTINEL; st_n]),
        y,
        st,
        out_n,
        st_n,
    }
}

fn gdn_views(
    c: &GdnCase,
) -> (
    qwen35::GdnQkv<'_>,
    qwen35::GdnGateLogits<'_>,
    qwen35::GdnParams<'_>,
    Cols<'_>,
) {
    let hv = c.dims.v_heads;
    let qkv = qwen35::GdnQkv {
        buf: &c.qkv,
        ld: c.ld,
        q_off: 0,
        k_off: c.dims.k_heads * 128,
        v_off: c.dims.k_heads * 256,
    };
    let gates = qwen35::GdnGateLogits {
        buf: &c.gates,
        ld: hv * 2,
        a_off: 0,
        b_off: hv,
    };
    let params = qwen35::GdnParams {
        a_log: &c.a_log,
        dt_bias: &c.dt,
    };
    let out = dense(&c.out, hv * c.dims.v_dim);
    (qkv, gates, params, out)
}

fn check_gdn(label: &str, c: &GdnCase) {
    close_peak(&format!("{label} y"), &take_f32(&c.out, c.out_n), &c.y, 1e-4);
    close_peak(&format!("{label} state"), &take_f32(&c.state, c.st_n), &c.st, 1e-4);
}

fn credit_gdn(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(
        rt,
        "gdn_chunk_cols32",
        &["qwen35_gdn_chunk_prep", "qwen35_gdn_chunk_scan"],
        || {
            let c = gdn_case(rt, 31);
            let (qkv, gates, params, out) = gdn_views(&c);
            let ws = qwen35::GdnWorkspace::new(rt, &c.dims)
                .unwrap()
                .with_scan_slice(GdnScanSlice::Cols32);
            qwen35::gdn_chunk_forward(
                rt,
                &c.dims,
                &qkv,
                &gates,
                &params,
                StateIn::Zero,
                &ws,
                out,
                Some(&c.state),
            )
            .unwrap();
            sync(rt);
            check_gdn("chunk32", &c);
        },
    );
    ledger.credit(rt, "gdn_chunk_cols16", &["qwen35_gdn_chunk_scan_bv16"], || {
        let c = gdn_case(rt, 32);
        let (qkv, gates, params, out) = gdn_views(&c);
        let ws = qwen35::GdnWorkspace::new(rt, &c.dims)
            .unwrap()
            .with_scan_slice(GdnScanSlice::Cols16);
        qwen35::gdn_chunk_forward(
            rt,
            &c.dims,
            &qkv,
            &gates,
            &params,
            StateIn::Zero,
            &ws,
            out,
            Some(&c.state),
        )
        .unwrap();
        sync(rt);
        check_gdn("chunk16", &c);
    });
    ledger.credit(rt, "gdn_recurrent", &["qwen35_gdn_recurrent"], || {
        let c = gdn_case(rt, 33);
        let (qkv, gates, params, out) = gdn_views(&c);
        qwen35::gdn_recurrent_with_slice(
            rt,
            &c.dims,
            &qkv,
            &gates,
            &params,
            StateIn::Zero,
            out,
            Some(&c.state),
            GdnScanSlice::Cols32,
        )
        .unwrap();
        sync(rt);
        check_gdn("recurrent", &c);
    });
    ledger.credit(rt, "gdn_recurrent_bv16", &["qwen35_gdn_recurrent_bv16"], || {
        let c = gdn_case(rt, 34);
        let (qkv, gates, params, out) = gdn_views(&c);
        qwen35::gdn_recurrent_with_slice(
            rt,
            &c.dims,
            &qkv,
            &gates,
            &params,
            StateIn::Zero,
            out,
            Some(&c.state),
            GdnScanSlice::Cols16,
        )
        .unwrap();
        sync(rt);
        check_gdn("recurrent16", &c);
    });
    ledger.credit(rt, "gdn_simd", &["qwen35_gdn_recurrent_simd"], || {
        let c = gdn_case(rt, 35);
        let (qkv, gates, params, out) = gdn_views(&c);
        qwen35::gdn_recurrent_simd(rt, &c.dims, &qkv, &gates, &params, StateIn::Zero, out, Some(&c.state)).unwrap();
        sync(rt);
        check_gdn("simd", &c);
    });
    ledger.credit(rt, "gdn_simd_l8", &["qwen35_gdn_recurrent_simd_l8_vec"], || {
        let c = gdn_case(rt, 36);
        let (qkv, gates, params, out) = gdn_views(&c);
        qwen35::gdn_recurrent_simd_with(
            rt,
            &c.dims,
            &qkv,
            &gates,
            &params,
            StateIn::Zero,
            out,
            Some(&c.state),
            None,
            GdnSimdVariant::Fused8,
            None,
        )
        .unwrap();
        sync(rt);
        check_gdn("simd_l8", &c);
    });
    ledger.credit(
        rt,
        "gdn_simd_pre",
        &["qwen35_gdn_simd_prep", "qwen35_gdn_recurrent_simd_pre_l8_vec"],
        || {
            let c = gdn_case(rt, 37);
            let (qkv, gates, params, out) = gdn_views(&c);
            let scalars = qwen35::GdnSimdScalars::new(rt, &c.dims).unwrap();
            qwen35::gdn_recurrent_simd_with(
                rt,
                &c.dims,
                &qkv,
                &gates,
                &params,
                StateIn::Zero,
                out,
                Some(&c.state),
                None,
                GdnSimdVariant::Prepped8,
                Some(&scalars),
            )
            .unwrap();
            sync(rt);
            check_gdn("simd_pre", &c);
        },
    );
}

fn credit_gated_norm(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let run = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, heads, dim) = (2u32, 2u32, 8u32);
            let width = heads * dim;
            let n = (rows * width) as usize;
            let x = random_f32(n, 41);
            let z = random_f32(n, 42);
            let w = random_f32(dim as usize, 43);
            let want = gated_rms_norm_f64(&x, &z, &w, dim as usize, f64::from(EPS));
            let (xb, zb, wb) = (buf(rt, &x), buf(rt, &z), buf(rt, &w));
            if bf16 {
                let yb = gpu_bf16(rt, &vec![SENTINEL; n]);
                qwen35::gated_rms_norm(
                    rt,
                    dense(&xb, width),
                    dense(&zb, width),
                    &wb,
                    out_bf16(&yb, width),
                    rows,
                    heads,
                    dim,
                    EPS,
                )
                .unwrap();
                sync(rt);
                close_wide(name, &take_bf16(&yb, n), &want);
            } else {
                let yb = buf(rt, &vec![SENTINEL; n]);
                qwen35::gated_rms_norm(
                    rt,
                    dense(&xb, width),
                    dense(&zb, width),
                    &wb,
                    out_f32(&yb, width),
                    rows,
                    heads,
                    dim,
                    EPS,
                )
                .unwrap();
                sync(rt);
                close_elem(name, &take_f32(&yb, n), &want);
            }
        });
    };
    run(ledger, false, "qwen35_gated_rms_norm_f32");
    run(ledger, true, "qwen35_gated_rms_norm_bf16");
}

fn qk_shape() -> AttnShape {
    AttnShape {
        batch: 1,
        seq: 2,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 8,
        rotary_dim: 4,
    }
}

fn qk_want(proj: &[f32], qw: &[f32], kw: &[f32], shape: &AttnShape) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (d, rot) = (shape.head_dim as usize, shape.rotary_dim as usize);
    let width = qwen35::AttnProjLayout::new(shape.q_heads, shape.kv_heads, shape.head_dim)
        .unwrap()
        .width() as usize;
    let rows = (shape.batch * shape.seq) as usize;
    let mut q = Vec::new();
    let mut k = Vec::new();
    let mut v = Vec::new();
    for t in 0..rows {
        let row = &proj[t * width..(t + 1) * width];
        q.extend(norm_rope_row_f64(
            &row[..d],
            qw,
            rot,
            t as u64,
            f64::from(THETA),
            f64::from(EPS),
        ));
        let k_off = (2 * shape.q_heads * shape.head_dim) as usize;
        k.extend(norm_rope_row_f64(
            &row[k_off..k_off + d],
            kw,
            rot,
            t as u64,
            f64::from(THETA),
            f64::from(EPS),
        ));
        let v_off = k_off + d;
        v.extend(row[v_off..v_off + d].iter().copied().map(f64::from));
    }
    (q, k, v)
}

fn credit_qk(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let shape = qk_shape();
    let width = qwen35::AttnProjLayout::new(shape.q_heads, shape.kv_heads, shape.head_dim)
        .unwrap()
        .width();
    let rows = (shape.batch * shape.seq) as usize;
    let qn = rows * shape.q_heads as usize * shape.head_dim as usize;
    let kn = rows * shape.kv_heads as usize * shape.head_dim as usize;
    let prep = |seed: u64| {
        let proj = random_f32(rows * width as usize, seed);
        let qw = random_f32(shape.head_dim as usize, seed + 1);
        let kw = random_f32(shape.head_dim as usize, seed + 2);
        let want = qk_want(&proj, &qw, &kw, &shape);
        (proj, qw, kw, want)
    };
    ledger.credit(rt, "qwen35_attn_qk_norm_rope", &["qwen35_attn_qk_norm_rope"], || {
        let (proj, qw, kw, (wq, wk, wv)) = prep(51);
        let (pb, qwb, kwb) = (buf(rt, &proj), buf(rt, &qw), buf(rt, &kw));
        let (qo, kc, vc) = (
            buf(rt, &vec![SENTINEL; qn]),
            buf(rt, &vec![SENTINEL; kn]),
            buf(rt, &vec![SENTINEL; kn]),
        );
        qwen35::attn_qk_norm_rope(
            rt,
            &shape,
            dense(&pb, width),
            &qwb,
            &kwb,
            &qwen35::AttnTargets {
                q_out: &qo,
                k_cache: &kc,
                v_cache: &vc,
            },
            0,
            THETA,
            EPS,
        )
        .unwrap();
        sync(rt);
        close_elem("qk q", &take_f32(&qo, qn), &wq);
        close_elem("qk k", &take_f32(&kc, kn), &wk);
        bits_eq(
            "qk v",
            &take_f32(&vc, kn),
            &wv.iter().map(|x| *x as f32).collect::<Vec<_>>(),
        );
    });
    ledger.credit(
        rt,
        "qwen35_attn_qk_norm_rope_bf16",
        &["qwen35_attn_qk_norm_rope_bf16"],
        || {
            let (proj, qw, kw, (wq, wk, wv)) = prep(52);
            let layout = qwen35::AttnProjLayout::new(shape.q_heads, shape.kv_heads, shape.head_dim).unwrap();
            let (pb, qwb, kwb) = (buf(rt, &proj), buf(rt, &qw), buf(rt, &kw));
            let (qo, kc, vc) = (
                gpu_bf16(rt, &vec![0.0; qn]),
                gpu_bf16(rt, &vec![0.0; kn]),
                gpu_bf16(rt, &vec![0.0; kn]),
            );
            qwen35::attn_qk_norm_rope_columns(
                rt,
                &shape,
                dense(&pb, width),
                qwen35::QkvColumns {
                    q_head_stride: shape.head_dim * 2,
                    k_col: layout.k_off(),
                    v_col: layout.v_off(),
                    v_norm: false,
                },
                1.0,
                &qwb,
                &kwb,
                &qwen35::AttnTargets {
                    q_out: &qo,
                    k_cache: &kc,
                    v_cache: &vc,
                },
                DType::BF16,
                0,
                THETA,
                EPS,
            )
            .unwrap();
            sync(rt);
            close_wide("qk bf16 q", &take_bf16(&qo, qn), &wq);
            close_wide("qk bf16 k", &take_bf16(&kc, kn), &wk);
            close_wide("qk bf16 v", &take_bf16(&vc, kn), &wv);
        },
    );
    ledger.credit(
        rt,
        "qwen35_attn_qk_norm_rope_posbuf",
        &["qwen35_attn_qk_norm_rope_posbuf"],
        || {
            let (proj, qw, kw, (wq, wk, wv)) = prep(53);
            let (pb, qwb, kwb, pos) = (buf(rt, &proj), buf(rt, &qw), buf(rt, &kw), buf_u32(rt, &[0]));
            let (qo, kc, vc) = (
                buf(rt, &vec![SENTINEL; qn]),
                buf(rt, &vec![SENTINEL; kn]),
                buf(rt, &vec![SENTINEL; kn]),
            );
            qwen35::attn_qk_norm_rope_posbuf(
                rt,
                &shape,
                dense(&pb, width),
                &qwb,
                &kwb,
                &qwen35::AttnTargets {
                    q_out: &qo,
                    k_cache: &kc,
                    v_cache: &vc,
                },
                &pos,
                THETA,
                EPS,
            )
            .unwrap();
            sync(rt);
            close_elem("qk pos q", &take_f32(&qo, qn), &wq);
            close_elem("qk pos k", &take_f32(&kc, kn), &wk);
            bits_eq(
                "qk pos v",
                &take_f32(&vc, kn),
                &wv.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            );
        },
    );
}

fn credit_gate_act(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (rows, heads, dim) = (2u32, 1u32, 8u32);
    let width = heads * dim;
    let ld = width * 2;
    let n = (rows * width) as usize;
    ledger.credit(rt, "qwen35_attn_gate_f32", &["qwen35_attn_gate_f32"], || {
        let p = random_f32((rows * ld) as usize, 61);
        let attn = random_f32(n, 62);
        let (pb, ab) = (buf(rt, &p), buf(rt, &attn));
        let yb = buf(rt, &vec![SENTINEL; n]);
        qwen35::attn_output_gate(rt, &ab, dense(&pb, ld), out_f32(&yb, width), rows, heads, dim).unwrap();
        sync(rt);
        let mut want = Vec::new();
        for r in 0..rows as usize {
            for h in 0..heads as usize {
                for j in 0..dim as usize {
                    let g = f64::from(p[r * ld as usize + h * 2 * dim as usize + dim as usize + j]);
                    let a = f64::from(attn[r * width as usize + h * dim as usize + j]);
                    want.push(a * sigmoid(g));
                }
            }
        }
        close_elem("attn gate", &take_f32(&yb, n), &want);
    });
    ledger.credit(rt, "qwen35_attn_gate_bf16", &["qwen35_attn_gate_bf16"], || {
        let p = random_f32((rows * ld) as usize, 63);
        let attn = random_f32(n, 64);
        let (pb, ab) = (buf(rt, &p), buf(rt, &attn));
        let yb = gpu_bf16(rt, &vec![0.0; n]);
        qwen35::attn_output_gate(rt, &ab, dense(&pb, ld), out_bf16(&yb, width), rows, heads, dim).unwrap();
        sync(rt);
        let mut want = Vec::new();
        for r in 0..rows as usize {
            for h in 0..heads as usize {
                for j in 0..dim as usize {
                    let g = f64::from(p[r * ld as usize + h * 2 * dim as usize + dim as usize + j]);
                    let a = f64::from(attn[r * width as usize + h * dim as usize + j]);
                    want.push(a * sigmoid(g));
                }
            }
        }
        close_wide("attn gate bf16", &take_bf16(&yb, n), &want);
    });
    let glu = |ledger: &mut Ledger, gelu: bool, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, width) = (2u32, 4u32);
            let n = (rows * width) as usize;
            let gate = random_f32(n, 71 + u64::from(gelu));
            let up = random_f32(n, 81);
            let want: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    let a = if gelu {
                        gelu_pytorch_tanh(f64::from(g))
                    } else {
                        silu(f64::from(g))
                    };
                    a * f64::from(u)
                })
                .collect();
            let (gb, ub) = (buf(rt, &gate), buf(rt, &up));
            if bf16 {
                let yb = gpu_bf16(rt, &vec![0.0; n]);
                if gelu {
                    qwen35::gated_act(
                        rt,
                        qwen35::GatedAct::GeluTanh,
                        dense(&gb, width),
                        dense(&ub, width),
                        out_bf16(&yb, width),
                        rows,
                        width,
                    )
                    .unwrap();
                } else {
                    qwen35::swiglu(
                        rt,
                        dense(&gb, width),
                        dense(&ub, width),
                        out_bf16(&yb, width),
                        rows,
                        width,
                    )
                    .unwrap();
                }
                sync(rt);
                close_wide(name, &take_bf16(&yb, n), &want);
            } else {
                let yb = buf(rt, &vec![SENTINEL; n]);
                if gelu {
                    qwen35::gated_act(
                        rt,
                        qwen35::GatedAct::GeluTanh,
                        dense(&gb, width),
                        dense(&ub, width),
                        out_f32(&yb, width),
                        rows,
                        width,
                    )
                    .unwrap();
                } else {
                    qwen35::swiglu(
                        rt,
                        dense(&gb, width),
                        dense(&ub, width),
                        out_f32(&yb, width),
                        rows,
                        width,
                    )
                    .unwrap();
                }
                sync(rt);
                close_elem(name, &take_f32(&yb, n), &want);
            }
        });
    };
    glu(ledger, true, false, "qwen35_gelu_tanh_glu_f32");
    glu(ledger, true, true, "qwen35_gelu_tanh_glu_bf16");
    glu(ledger, false, false, "qwen35_swiglu_f32");
    glu(ledger, false, true, "qwen35_swiglu_bf16");
}

fn credit_residual_norm(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(rt, "qwen35_residual_add_f32", &["qwen35_residual_add_f32"], || {
        let (rows, width) = (2u32, 4u32);
        let n = (rows * width) as usize;
        let y = random_f32(n, 91);
        let r = random_f32(n, 92);
        let want: Vec<f32> = y.iter().zip(&r).map(|(a, b)| a + b).collect();
        let (yb, rb) = (buf(rt, &y), buf(rt, &r));
        qwen35::residual_add(rt, dense(&yb, width), dense(&rb, width), rows, width).unwrap();
        sync(rt);
        bits_eq("residual", &take_f32(&rb, n), &want);
    });
    let rms = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, dim) = (2u32, 8u32);
            let n = (rows * dim) as usize;
            let x = random_f32(n, 93);
            let w = random_f32(dim as usize, 94);
            let want = rms_rows(&x, &w, dim as usize, f64::from(EPS));
            let (xb, wb) = (buf(rt, &x), buf(rt, &w));
            if bf16 {
                let yb = gpu_bf16(rt, &vec![0.0; n]);
                qwen35::rms_norm(rt, &xb, &wb, &yb, DType::BF16, rows, dim, EPS).unwrap();
                sync(rt);
                close_wide(name, &take_bf16(&yb, n), &want);
            } else {
                let yb = buf(rt, &vec![SENTINEL; n]);
                qwen35::rms_norm(rt, &xb, &wb, &yb, DType::F32, rows, dim, EPS).unwrap();
                sync(rt);
                close_elem(name, &take_f32(&yb, n), &want);
            }
        });
    };
    rms(ledger, false, "qwen35_rms_norm_f32");
    rms(ledger, true, "qwen35_rms_norm_bf16");
    let add = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, dim) = (2u32, 8u32);
            let n = (rows * dim) as usize;
            let y = random_f32(n, 95);
            let r = random_f32(n, 96);
            let w = random_f32(dim as usize, 97);
            let z: Vec<f32> = y.iter().zip(&r).map(|(a, b)| a + b).collect();
            let want = rms_rows(&z, &w, dim as usize, f64::from(EPS));
            let (yb, rb, wb) = (buf(rt, &y), buf(rt, &r), buf(rt, &w));
            if bf16 {
                let ob = gpu_bf16(rt, &vec![0.0; n]);
                qwen35::add_rms_norm(rt, &yb, &rb, &wb, &ob, DType::BF16, rows, dim, EPS).unwrap();
                sync(rt);
                close_wide(name, &take_bf16(&ob, n), &want);
            } else {
                let ob = buf(rt, &vec![SENTINEL; n]);
                qwen35::add_rms_norm(rt, &yb, &rb, &wb, &ob, DType::F32, rows, dim, EPS).unwrap();
                sync(rt);
                close_elem(name, &take_f32(&ob, n), &want);
            }
        });
    };
    add(ledger, false, "qwen35_add_rms_norm_f32");
    add(ledger, true, "qwen35_add_rms_norm_bf16");
    let gate_up = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (k, n) = (4usize, 4usize);
            let x = random_f32(k, 101);
            let wg = random_f32(k * n, 102);
            let wu = random_f32(k * n, 103);
            let (xw, gw, uw) = if bf16 {
                (widen_gpu(&x), widen_gpu(&wg), widen_gpu(&wu))
            } else {
                (x.clone(), wg.clone(), wu.clone())
            };
            let want: Vec<f64> = (0..n)
                .map(|j| {
                    let mut g = 0.0f64;
                    let mut u = 0.0f64;
                    for i in 0..k {
                        g += f64::from(xw[i]) * f64::from(gw[i * n + j]);
                        u += f64::from(xw[i]) * f64::from(uw[i * n + j]);
                    }
                    silu(g) * u
                })
                .collect();
            let dtype = if bf16 { DType::BF16 } else { DType::F32 };
            let (xb, wgb, wub) = if bf16 {
                (gpu_bf16(rt, &x), gpu_bf16(rt, &wg), gpu_bf16(rt, &wu))
            } else {
                (buf(rt, &x), buf(rt, &wg), buf(rt, &wu))
            };
            if bf16 {
                let ob = gpu_bf16(rt, &vec![0.0; n]);
                qwen35::gate_up_silu(rt, &xb, &wgb, &wub, &ob, dtype, k as u32, n as u32).unwrap();
                sync(rt);
                close_wide(name, &take_bf16(&ob, n), &want);
            } else {
                let ob = buf(rt, &vec![SENTINEL; n]);
                qwen35::gate_up_silu(rt, &xb, &wgb, &wub, &ob, dtype, k as u32, n as u32).unwrap();
                sync(rt);
                close_elem(name, &take_f32(&ob, n), &want);
            }
        });
    };
    gate_up(ledger, false, "qwen35_gate_up_silu_f32");
    gate_up(ledger, true, "qwen35_gate_up_silu_bf16");
}

fn prefill_dims(tq: u32) -> AttnDims {
    AttnDims {
        batch: 1,
        tq,
        heads: 2,
        heads_kv: 1,
        window: 0,
        scale: 0.0625,
    }
}

fn credit_prefill(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let host = AttnShapeHost {
        b: 1,
        tq: 4,
        tkv: 4,
        h: 2,
        hkv: 1,
        d: D,
        scale: 0.0625,
        q_pos: 0,
        kv_pos: 0,
    };
    let qn = host.b * host.tq * host.h * host.d;
    let kn = host.b * host.tkv * host.hkv * host.d;
    let tiles = [
        (AttnTile::Q32K32Sg4, "qwen35_attn_tiled_h256_q32_k32_sg4"),
        (AttnTile::Q32K64Sg4, "qwen35_attn_tiled_h256_q32_k64_sg4"),
        (AttnTile::Q64K32Sg8, "qwen35_attn_tiled_h256_q64_k32_sg8"),
        (AttnTile::Q64K64Sg8, "qwen35_attn_tiled_h256_q64_k64_sg8"),
    ];
    for (i, (tile, name)) in tiles.into_iter().enumerate() {
        ledger.credit(rt, name, &[name], || {
            let q = random_f32(qn, 110 + i as u64);
            let k = random_f32(kn, 120 + i as u64);
            let v = random_f32(kn, 130 + i as u64);
            let (want, _) = attn_o(&host, &q, &k, &v);
            let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
            let ob = buf(rt, &vec![SENTINEL; qn]);
            let (tkv, zero) = (buf_u32(rt, &[host.tkv as u32]), buf_u32(rt, &[0]));
            qwen35::attn_prefill_with_tile(
                rt,
                &qb,
                &kb,
                &vb,
                &ob,
                &tkv,
                &zero,
                &zero,
                prefill_dims(host.tq as u32),
                false,
                tile,
            )
            .unwrap();
            sync(rt);
            close_attn(name, &take_f32(&ob, qn), &want);
        });
    }
    let bf = [
        (
            AttnTile::Q32K32Sg4,
            AttnProbTile::Bf16,
            "qwen35_attn_tiled_bf16_h256_q32_k32_sg4",
        ),
        (
            AttnTile::Q32K64Sg4,
            AttnProbTile::Bf16,
            "qwen35_attn_tiled_bf16_h256_q32_k64_sg4",
        ),
        (
            AttnTile::Q64K32Sg8,
            AttnProbTile::Bf16,
            "qwen35_attn_tiled_bf16_h256_q64_k32_sg8",
        ),
        (
            AttnTile::Q64K64Sg8,
            AttnProbTile::Bf16,
            "qwen35_attn_tiled_bf16_h256_q64_k64_sg8",
        ),
        (
            AttnTile::Q32K32Sg4,
            AttnProbTile::F32,
            "qwen35_attn_tiled_bf16pf32_h256_q32_k32_sg4",
        ),
    ];
    for (i, (tile, prob, name)) in bf.into_iter().enumerate() {
        ledger.credit(rt, name, &[name], || {
            let q = random_f32(qn, 140 + i as u64);
            let k = random_f32(kn, 150 + i as u64);
            let v = random_f32(kn, 160 + i as u64);
            let (qr, kr, vr) = (widen_gpu(&q), widen_gpu(&k), widen_gpu(&v));
            let (want, _) = attn_o(&host, &qr, &kr, &vr);
            let (qb, kb, vb) = (gpu_bf16(rt, &q), gpu_bf16(rt, &k), gpu_bf16(rt, &v));
            let ob = buf(rt, &vec![SENTINEL; qn]);
            let (tkv, zero) = (buf_u32(rt, &[host.tkv as u32]), buf_u32(rt, &[0]));
            attn_prefill_bf16::attn_prefill_bf16_with_tile(
                rt,
                &qb,
                &kb,
                &vb,
                &ob,
                &tkv,
                &zero,
                &zero,
                prefill_dims(host.tq as u32),
                false,
                tile,
                prob,
            )
            .unwrap();
            sync(rt);
            close_wide(name, &take_f32(&ob, qn), &want);
        });
    }
}

fn credit_train(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (b, t, hq, hkv) = (1usize, 4usize, 2usize, 1usize);
    let scale = 0.0625f64;
    let qn = b * t * hq * D;
    let kn = b * t * hkv * D;
    let dims = AttnTrainDims {
        batch: b as u32,
        seq: t as u32,
        q_heads: hq as u32,
        kv_heads: hkv as u32,
        scale: scale as f32,
    };
    let host = AttnShapeHost {
        b,
        tq: t,
        tkv: t,
        h: hq,
        hkv,
        d: D,
        scale,
        q_pos: 0,
        kv_pos: 0,
    };
    ledger.credit(
        rt,
        "qwen35_attn_tiled_lse_h256_q32_k32_sg4",
        &["qwen35_attn_tiled_lse_h256_q32_k32_sg4"],
        || {
            let q = random_f32(qn, 171);
            let k = random_f32(kn, 172);
            let v = random_f32(kn, 173);
            let (want_o, want_lse) = attn_o(&host, &q, &k, &v);
            let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
            let ob = buf(rt, &vec![SENTINEL; qn]);
            let lse = buf(rt, &vec![SENTINEL; dims.lse_len()]);
            let ws = AttnTrainWorkspace::new(rt, dims).unwrap();
            attn_train::attn_train_forward(rt, &dims, &qb, &kb, &vb, &ob, &lse, &ws).unwrap();
            sync(rt);
            close_attn("train o", &take_f32(&ob, qn), &want_o);
            close_attn("train lse", &take_f32(&lse, dims.lse_len()), &want_lse);
        },
    );
    ledger.credit(rt, "qwen35_attn_bwd_dvec_f32", &["qwen35_attn_bwd_dvec_f32"], || {
        let o = random_f32(qn, 174);
        let d_o = random_f32(qn, 175);
        let mut want = vec![0.0f64; b * hq * t];
        for bi in 0..b {
            for h in 0..hq {
                for ti in 0..t {
                    let row = ((bi * t + ti) * hq + h) * D;
                    let s: f64 = (0..D).map(|c| f64::from(o[row + c]) * f64::from(d_o[row + c])).sum();
                    want[(bi * hq + h) * t + ti] = s;
                }
            }
        }
        let (ob, dob) = (buf(rt, &o), buf(rt, &d_o));
        let dvec = buf(rt, &vec![SENTINEL; want.len()]);
        let p = rt.pipeline("qwen35_attn_bwd_dvec_f32").unwrap();
        let bh = (b * hq) as u32;
        dispatch_2d(rt, &p, t, bh as usize, |bnd| {
            dispatch::set_gpu_buf(bnd, &ob, 0);
            dispatch::set_gpu_buf(bnd, &dob, 1);
            dispatch::set_gpu_buf(bnd, &dvec, 2);
            dispatch::set_u32(bnd, t as u32, 3);
            dispatch::set_u32(bnd, hq as u32, 4);
            dispatch::set_u32(bnd, bh, 5);
        })
        .unwrap();
        sync(rt);
        close_attn("dvec", &take_f32(&dvec, want.len()), &want);
    });
    ledger.credit(
        rt,
        "attn_bwd_qkv",
        &[
            "qwen35_attn_bwd_dq_h256_q32_k32_sg4",
            "qwen35_attn_bwd_dk_h256_q32_k32_sg4",
            "qwen35_attn_bwd_dv_h256_q32_k32_sg4",
        ],
        || {
            let q = random_f32(qn, 176);
            let k = random_f32(kn, 177);
            let v = random_f32(kn, 178);
            let d_o = random_f32(qn, 179);
            let (want_dq, want_dk, want_dv) = attn_bwd_f64(&f64s(&q), &f64s(&k), &f64s(&v), &f64s(&d_o), &host);
            let (qb, kb, vb, dob) = (buf(rt, &q), buf(rt, &k), buf(rt, &v), buf(rt, &d_o));
            let ob = buf(rt, &vec![0.0; qn]);
            let lse = buf(rt, &vec![0.0; dims.lse_len()]);
            let (dq, dk, dv) = (
                buf(rt, &vec![SENTINEL; qn]),
                buf(rt, &vec![SENTINEL; kn]),
                buf(rt, &vec![SENTINEL; kn]),
            );
            let ws = AttnTrainWorkspace::new(rt, dims).unwrap();
            attn_train::attn_train_forward(rt, &dims, &qb, &kb, &vb, &ob, &lse, &ws).unwrap();
            attn_train::attn_train_backward(
                rt,
                &dims,
                &qb,
                &kb,
                &vb,
                &ob,
                &lse,
                &dob,
                &AttnTrainGrads {
                    dq: &dq,
                    dk: &dk,
                    dv: &dv,
                },
                &ws,
            )
            .unwrap();
            sync(rt);
            close_attn("dq", &take_f32(&dq, qn), &want_dq);
            close_attn("dk", &take_f32(&dk, kn), &want_dk);
            close_attn("dv", &take_f32(&dv, kn), &want_dv);
        },
    );
}

fn credit_prefix(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let (h, hkv, d) = (2usize, 1usize, D);
    let (p, s_live, s_cap) = (2usize, 2usize, 4usize);
    let scale = 0.0625f64;
    let pk = random_f32(p * hkv * d, 181);
    let pv = random_f32(p * hkv * d, 182);
    let sk = random_f32(s_cap * hkv * d, 183);
    let sv = random_f32(s_cap * hkv * d, 184);
    let mut kc = Vec::new();
    let mut vc = Vec::new();
    kc.extend_from_slice(&pk);
    kc.extend_from_slice(&sk[..s_live * hkv * d]);
    vc.extend_from_slice(&pv);
    vc.extend_from_slice(&sv[..s_live * hkv * d]);
    ledger.credit(rt, "qwen35_attn_prefix_rows", &["qwen35_attn_prefix_rows"], || {
        let tq = 2usize;
        let q = random_f32(tq * h * d, 185);
        let host = AttnShapeHost {
            b: 1,
            tq,
            tkv: p + s_live,
            h,
            hkv,
            d,
            scale,
            q_pos: p,
            kv_pos: 0,
        };
        let (want, _) = attn_o(&host, &q, &kc, &vc);
        let (qb, pkb, pvb, skb, svb) = (buf(rt, &q), buf(rt, &pk), buf(rt, &pv), buf(rt, &sk), buf(rt, &sv));
        let ob = buf(rt, &vec![SENTINEL; tq * h * d]);
        let (slen, qpos) = (buf_u32(rt, &[s_live as u32]), buf_u32(rt, &[p as u32]));
        let dims = AttnDims {
            batch: 1,
            tq: tq as u32,
            heads: h as u32,
            heads_kv: hkv as u32,
            window: 0,
            scale: scale as f32,
        };
        qwen35::attn_prefix_rows(
            rt,
            &qb,
            qwen35::SharedPrefix {
                k: &pkb,
                v: &pvb,
                len: p as u32,
            },
            &skb,
            &svb,
            &slen,
            &qpos,
            &ob,
            dims,
            false,
        )
        .unwrap();
        sync(rt);
        close_attn("prefix rows", &take_f32(&ob, want.len()), &want);
    });
    ledger.credit(
        rt,
        "prefix_decode",
        &["qwen35_attn_prefix_decode_partial", "qwen35_attn_prefix_decode_reduce"],
        || {
            let q = random_f32(h * d, 186);
            let host = AttnShapeHost {
                b: 1,
                tq: 1,
                tkv: p + s_live,
                h,
                hkv,
                d,
                scale,
                q_pos: p + s_live - 1,
                kv_pos: 0,
            };
            let (want, _) = attn_o(&host, &q, &kc, &vc);
            let (qb, pkb, pvb, skb, svb) = (buf(rt, &q), buf(rt, &pk), buf(rt, &pv), buf(rt, &sk), buf(rt, &sv));
            let ob = buf(rt, &vec![SENTINEL; h * d]);
            let (slen, qpos) = (buf_u32(rt, &[s_live as u32]), buf_u32(rt, &[(p + s_live - 1) as u32]));
            let dims = AttnDims {
                batch: 1,
                tq: 1,
                heads: h as u32,
                heads_kv: hkv as u32,
                window: 0,
                scale: scale as f32,
            };
            let scratch = DecodeScratch::new(rt, 1, h as u32, p + s_cap, d as u32).unwrap();
            qwen35::attn_prefix_decode(
                rt,
                &qb,
                qwen35::SharedPrefix {
                    k: &pkb,
                    v: &pvb,
                    len: p as u32,
                },
                &skb,
                &svb,
                &slen,
                &qpos,
                &ob,
                &scratch,
                dims,
                false,
            )
            .unwrap();
            sync(rt);
            close_attn("prefix decode", &take_f32(&ob, want.len()), &want);
        },
    );
}

fn credit_score_embed(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let score = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, hidden, vocab, n_ans) = (3usize, 8usize, 4usize, 2usize);
            let hidden_s = random_f32(rows * hidden, 191);
            let norm_w = random_f32(hidden, 192);
            let table = random_f32(vocab * hidden, 193);
            let table_w = if bf16 { widen_gpu(&table) } else { table.clone() };
            let emb0 = &table_w[0..hidden];
            let emb2 = &table_w[2 * hidden..3 * hidden];
            let (logits, logp) = score_row_f64(
                &hidden_s[hidden..2 * hidden],
                &norm_w,
                1.0,
                f64::from(EPS),
                &[emb0, emb2],
            );
            let (hb, nb, slots, answers) = (
                buf(rt, &hidden_s),
                buf(rt, &norm_w),
                buf_u32(rt, &[1]),
                buf_u32(rt, &[0, 2]),
            );
            let tb = if bf16 { gpu_bf16(rt, &table) } else { buf(rt, &table) };
            let (lb, pb) = (buf(rt, &vec![SENTINEL; n_ans]), buf(rt, &vec![SENTINEL; n_ans]));
            qwen35::score_answer_rows(
                rt,
                &hb,
                rows as u32,
                hidden as u32,
                &slots,
                1,
                &nb,
                1.0,
                EPS,
                qwen35::LmHead {
                    weight: &tb,
                    dtype: if bf16 { DType::BF16 } else { DType::F32 },
                    vocab: vocab as u32,
                },
                &answers,
                n_ans as u32,
                &lb,
                &pb,
            )
            .unwrap();
            sync(rt);
            close_peak(&format!("{name} logits"), &take_f32(&lb, n_ans), &logits, 1e-4);
            close_peak(&format!("{name} logp"), &take_f32(&pb, n_ans), &logp, 1e-4);
        });
    };
    score(ledger, false, "qwen35_score_rows_f32");
    score(ledger, true, "qwen35_score_rows_bf16");
    ledger.credit(rt, "qwen35_embed_rows_f32", &["qwen35_embed_rows_f32"], || {
        let (vocab, hidden) = (4usize, 8usize);
        let table = random_f32(vocab * hidden, 201);
        let ids = [1u32, 3, 0];
        let mut want = Vec::new();
        for &id in &ids {
            want.extend_from_slice(&table[id as usize * hidden..(id as usize + 1) * hidden]);
        }
        let (ib, tb) = (buf_u32(rt, &ids), buf(rt, &table));
        let ob = buf(rt, &vec![SENTINEL; ids.len() * hidden]);
        qwen35::embed_rows(
            rt,
            &ib,
            ids.len() as u32,
            qwen35::LmHead {
                weight: &tb,
                dtype: DType::F32,
                vocab: vocab as u32,
            },
            hidden as u32,
            &ob,
        )
        .unwrap();
        sync(rt);
        bits_eq("embed f32", &take_f32(&ob, want.len()), &want);
    });
    ledger.credit(rt, "qwen35_embed_rows_bf16", &["qwen35_embed_rows_bf16"], || {
        let (vocab, hidden) = (4usize, 8usize);
        let table = random_f32(vocab * hidden, 202);
        let wide = widen_gpu(&table);
        let ids = [2u32, 0];
        let mut want = Vec::new();
        for &id in &ids {
            want.extend_from_slice(&wide[id as usize * hidden..(id as usize + 1) * hidden]);
        }
        let (ib, tb) = (buf_u32(rt, &ids), gpu_bf16(rt, &table));
        let ob = buf(rt, &vec![SENTINEL; ids.len() * hidden]);
        qwen35::embed_rows(
            rt,
            &ib,
            ids.len() as u32,
            qwen35::LmHead {
                weight: &tb,
                dtype: DType::BF16,
                vocab: vocab as u32,
            },
            hidden as u32,
            &ob,
        )
        .unwrap();
        sync(rt);
        bits_eq("embed bf16", &take_f32(&ob, want.len()), &want);
    });
    ledger.credit(
        rt,
        "qwen35_embed_rows_bf16_scaled",
        &["qwen35_embed_rows_bf16_scaled"],
        || {
            let (vocab, hidden) = (4usize, 8usize);
            let scale = 1.25f32;
            let table = random_f32(vocab * hidden, 203);
            let wide = widen_gpu(&table);
            let ids = [0u32, 1];
            let want: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    wide[id as usize * hidden..(id as usize + 1) * hidden]
                        .iter()
                        .map(move |v| v * scale)
                })
                .collect();
            let (ib, tb) = (buf_u32(rt, &ids), gpu_bf16(rt, &table));
            let ob = buf(rt, &vec![SENTINEL; ids.len() * hidden]);
            qwen35::embed_rows_scaled(
                rt,
                &ib,
                ids.len() as u32,
                qwen35::LmHead {
                    weight: &tb,
                    dtype: DType::BF16,
                    vocab: vocab as u32,
                },
                hidden as u32,
                scale,
                &ob,
            )
            .unwrap();
            sync(rt);
            bits_eq("embed scaled", &take_f32(&ob, want.len()), &want);
        },
    );
}

fn credit_bwd(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(
        rt,
        "rms_norm_bwd",
        &["qwen35_rms_norm_bwd_f32", "qwen35_col_sum_blocks_f32"],
        || {
            let (rows, dim) = (2usize, 16usize);
            let x = random_f32(rows * dim, 211);
            let w = random_f32(dim, 212);
            let dy = random_f32(rows * dim, 213);
            let (want_dx, want_dw) = rms_bwd(&f64s(&x), &f64s(&w), &f64s(&dy), dim, f64::from(EPS));
            let (xb, wb, dyb) = (buf(rt, &x), buf(rt, &w), buf(rt, &dy));
            let (dx, dw) = (buf(rt, &vec![SENTINEL; rows * dim]), buf(rt, &vec![SENTINEL; dim]));
            let part = buf(
                rt,
                &vec![0.0; qwen35_bwd::rms_norm_bwd_part_len(rows as u32, dim as u32)],
            );
            qwen35_bwd::rms_norm_bwd(rt, &xb, &wb, &dyb, &dx, &dw, &part, rows as u32, dim as u32, EPS, false).unwrap();
            sync(rt);
            close_peak("rms dx", &take_f32(&dx, rows * dim), &want_dx, 1e-4);
            close_peak("rms dw", &take_f32(&dw, dim), &want_dw, 1e-4);
        },
    );
    ledger.credit(
        rt,
        "qwen35_gated_rms_norm_bwd_f32",
        &["qwen35_gated_rms_norm_bwd_f32"],
        || {
            let (rows, heads, dim) = (2usize, 2usize, 8usize);
            let width = (heads * dim) as u32;
            let n = rows * heads * dim;
            let x = random_f32(n, 221);
            let z = random_f32(n, 222);
            let w = random_f32(dim, 223);
            let dy = random_f32(n, 224);
            let (dx, dz, dw) = gated_bwd(&f64s(&x), &f64s(&z), &f64s(&w), &f64s(&dy), dim, f64::from(EPS));
            let (xb, zb, wb, dyb) = (buf(rt, &x), buf(rt, &z), buf(rt, &w), buf(rt, &dy));
            let (dxb, dzb, dwb) = (
                buf(rt, &vec![SENTINEL; n]),
                buf(rt, &vec![SENTINEL; n]),
                buf(rt, &vec![SENTINEL; dim]),
            );
            let part = buf(
                rt,
                &vec![0.0; qwen35_bwd::gated_rms_norm_bwd_part_len(rows as u32, heads as u32, dim as u32)],
            );
            qwen35_bwd::gated_rms_norm_bwd(
                rt,
                dense(&xb, width),
                dense(&zb, width),
                &wb,
                dense(&dyb, width),
                dense(&dxb, width),
                dense(&dzb, width),
                &dwb,
                &part,
                rows as u32,
                heads as u32,
                dim as u32,
                EPS,
            )
            .unwrap();
            sync(rt);
            close_peak("gated dx", &take_f32(&dxb, n), &dx, 1e-4);
            close_peak("gated dz", &take_f32(&dzb, n), &dz, 1e-4);
            close_peak("gated dw", &take_f32(&dwb, dim), &dw, 1e-4);
        },
    );
    ledger.credit(rt, "qwen35_swiglu_bwd_f32", &["qwen35_swiglu_bwd_f32"], || {
        let (rows, width) = (2usize, 4usize);
        let gate = random_f32(rows * width, 231);
        let up = random_f32(rows * width, 232);
        let dy = random_f32(rows * width, 233);
        let mut dgate = vec![0.0f64; gate.len()];
        let mut dup = vec![0.0f64; up.len()];
        for i in 0..gate.len() {
            dgate[i] = f64::from(dy[i]) * f64::from(up[i]) * silu_grad(f64::from(gate[i]));
            dup[i] = f64::from(dy[i]) * silu(f64::from(gate[i]));
        }
        let (gb, ub, dyb) = (buf(rt, &gate), buf(rt, &up), buf(rt, &dy));
        let (dgb, dub) = (buf(rt, &vec![SENTINEL; gate.len()]), buf(rt, &vec![SENTINEL; up.len()]));
        qwen35_bwd::swiglu_bwd(
            rt,
            dense(&gb, width as u32),
            dense(&ub, width as u32),
            dense(&dyb, width as u32),
            dense(&dgb, width as u32),
            dense(&dub, width as u32),
            rows as u32,
            width as u32,
        )
        .unwrap();
        sync(rt);
        close_peak("swiglu dgate", &take_f32(&dgb, dgate.len()), &dgate, 1e-4);
        close_peak("swiglu dup", &take_f32(&dub, dup.len()), &dup, 1e-4);
    });
    ledger.credit(rt, "qwen35_attn_gate_bwd_f32", &["qwen35_attn_gate_bwd_f32"], || {
        let (rows, heads, dim) = (2usize, 1usize, 8usize);
        let width = heads * dim;
        let ld = width * 2;
        let p = random_f32(rows * ld, 241);
        let attn = random_f32(rows * width, 242);
        let dy = random_f32(rows * width, 243);
        let (pb, ab, dyb) = (buf(rt, &p), buf(rt, &attn), buf(rt, &dy));
        let (dab, dpb) = (
            buf(rt, &vec![SENTINEL; rows * width]),
            buf(rt, &vec![SENTINEL; rows * ld]),
        );
        qwen35_bwd::attn_gate_bwd(
            rt,
            &ab,
            dense(&pb, ld as u32),
            dense(&dyb, width as u32),
            &dab,
            &dpb,
            rows as u32,
            heads as u32,
            dim as u32,
        )
        .unwrap();
        sync(rt);
        let dp = take_f32(&dpb, rows * ld);
        let mut wa = Vec::new();
        let mut wg = Vec::new();
        let mut got_g = Vec::new();
        for r in 0..rows {
            for h in 0..heads {
                for j in 0..dim {
                    let gi = r * ld + h * 2 * dim + dim + j;
                    let ai = r * width + h * dim + j;
                    let s = sigmoid(f64::from(p[gi]));
                    wa.push(f64::from(dy[ai]) * s);
                    wg.push(f64::from(dy[ai]) * f64::from(attn[ai]) * s * (1.0 - s));
                    got_g.push(dp[gi]);
                }
            }
        }
        close_peak("gate d_attn", &take_f32(&dab, wa.len()), &wa, 1e-4);
        close_peak("gate d_gate", &got_g, &wg, 1e-4);
    });
    ledger.credit(
        rt,
        "conv_bwd",
        &["qwen35_conv1d_silu_bwd_dx_f32", "qwen35_conv1d_silu_bwd_dw_f32"],
        || {
            let (b, t, c, kw) = (1usize, 3usize, 4usize, 3usize);
            let x = random_f32(b * t * c, 251);
            let w = random_f32(c * kw, 252);
            let dy = random_f32(b * t * c, 253);
            let (dx, dw) = conv_bwd(&f64s(&x), &f64s(&w), &f64s(&dy), b, t, c, kw);
            let (xb, wb, dyb) = (buf(rt, &x), buf(rt, &w), buf(rt, &dy));
            let (dxb, dwb) = (buf(rt, &vec![SENTINEL; x.len()]), buf(rt, &vec![SENTINEL; w.len()]));
            let part = buf(
                rt,
                &vec![0.0; qwen35_bwd::conv1d_silu_bwd_part_len(b as u32, t as u32, c as u32, kw as u32).max(1)],
            );
            qwen35_bwd::conv1d_silu_bwd(
                rt,
                dense(&xb, c as u32),
                &wb,
                kw as u32,
                dense(&dyb, c as u32),
                dense(&dxb, c as u32),
                &dwb,
                &part,
                b as u32,
                t as u32,
                c as u32,
            )
            .unwrap();
            sync(rt);
            close_peak("conv dx", &take_f32(&dxb, dx.len()), &dx, 1e-4);
            close_peak("conv dw", &take_f32(&dwb, dw.len()), &dw, 1e-4);
        },
    );
    ledger.credit(
        rt,
        "qwen35_attn_qk_norm_rope_bwd_f32",
        &["qwen35_attn_qk_norm_rope_bwd_f32"],
        || {
            let shape = qk_shape();
            let d = shape.head_dim as usize;
            let rot = shape.rotary_dim as usize;
            let rows = (shape.batch * shape.seq) as usize;
            let width = qwen35::AttnProjLayout::new(shape.q_heads, shape.kv_heads, shape.head_dim)
                .unwrap()
                .width() as usize;
            let proj = random_f32(rows * width, 261);
            let (qw, kw) = (random_f32(d, 262), random_f32(d, 263));
            let (dq, dk, dv) = (
                random_f32(rows * d, 264),
                random_f32(rows * d, 265),
                random_f32(rows * d, 266),
            );
            let k_off = 2 * d;
            let v_off = 3 * d;
            let mut dproj = vec![SENTINEL; rows * width];
            let (mut dqw, mut dkw) = (vec![0.0f64; d], vec![0.0f64; d]);
            for t in 0..rows {
                let row = &proj[t * width..(t + 1) * width];
                let (dx, dw) = norm_rope_bwd(
                    &f64s(&row[..d]),
                    &f64s(&qw),
                    &f64s(&dq[t * d..(t + 1) * d]),
                    rot,
                    t,
                    f64::from(THETA),
                    f64::from(EPS),
                );
                for j in 0..d {
                    dproj[t * width + j] = dx[j] as f32;
                    dqw[j] += dw[j];
                }
                let (dxk, dwk) = norm_rope_bwd(
                    &f64s(&row[k_off..k_off + d]),
                    &f64s(&kw),
                    &f64s(&dk[t * d..(t + 1) * d]),
                    rot,
                    t,
                    f64::from(THETA),
                    f64::from(EPS),
                );
                for j in 0..d {
                    dproj[t * width + k_off + j] = dxk[j] as f32;
                    dkw[j] += dwk[j];
                }
                for j in 0..d {
                    dproj[t * width + v_off + j] = dv[t * d + j];
                }
            }
            let (pb, qwb, kwb) = (buf(rt, &proj), buf(rt, &qw), buf(rt, &kw));
            let (dqb, dkb, dvb) = (buf(rt, &dq), buf(rt, &dk), buf(rt, &dv));
            let dpb = buf(rt, &vec![SENTINEL; rows * width]);
            let (dqwb, dkwb) = (buf(rt, &vec![SENTINEL; d]), buf(rt, &vec![SENTINEL; d]));
            let part = buf(rt, &vec![0.0; qwen35_bwd::attn_qk_norm_rope_bwd_part_len(&shape)]);
            qwen35_bwd::attn_qk_norm_rope_bwd(
                rt,
                &shape,
                dense(&pb, width as u32),
                &qwb,
                &kwb,
                &AttnQkvGrads {
                    dq: &dqb,
                    dk: &dkb,
                    dv: &dvb,
                },
                &dpb,
                &dqwb,
                &dkwb,
                &part,
                THETA,
                EPS,
            )
            .unwrap();
            sync(rt);
            let got = take_f32(&dpb, rows * width);
            let mut got_q = Vec::new();
            let mut got_k = Vec::new();
            let mut got_v = Vec::new();
            let mut want_q = Vec::new();
            let mut want_k = Vec::new();
            let mut want_v = Vec::new();
            for t in 0..rows {
                for j in 0..d {
                    got_q.push(got[t * width + j]);
                    want_q.push(f64::from(dproj[t * width + j]));
                    got_k.push(got[t * width + k_off + j]);
                    want_k.push(f64::from(dproj[t * width + k_off + j]));
                    got_v.push(got[t * width + v_off + j]);
                    want_v.push(f64::from(dproj[t * width + v_off + j]));
                }
            }
            close_peak("qk bwd q", &got_q, &want_q, 1e-4);
            close_peak("qk bwd k", &got_k, &want_k, 1e-4);
            bits_eq("qk bwd v", &got_v, &dv);
            close_peak("qk bwd dqw", &take_f32(&dqwb, d), &dqw, 1e-4);
            close_peak("qk bwd dkw", &take_f32(&dkwb, d), &dkw, 1e-4);
        },
    );
    ledger.credit(rt, "qwen35_embed_rows_bwd_f32", &["qwen35_embed_rows_bwd_f32"], || {
        let (vocab, hidden) = (4usize, 8usize);
        let ids = [1u32, 0, 1];
        let dh = random_f32(ids.len() * hidden, 271);
        let prior = random_f32(vocab * hidden, 272);
        let mut want = prior.clone();
        for (r, &id) in ids.iter().enumerate() {
            for c in 0..hidden {
                want[id as usize * hidden + c] += dh[r * hidden + c];
            }
        }
        let (dhb, dwb) = (buf(rt, &dh), buf(rt, &prior));
        let ws = qwen35_bwd::EmbedBwdWorkspace::new(rt, 8).unwrap();
        qwen35_bwd::embed_rows_bwd(rt, &ids, &dhb, &dwb, vocab as u32, hidden as u32, &ws).unwrap();
        sync(rt);
        bits_eq("embed bwd", &take_f32(&dwb, want.len()), &want);
    });
    ledger.credit(rt, "qwen35_gdn_gates_bwd_f32", &["qwen35_gdn_gates_bwd_f32"], || {
        let (rows, heads) = (3usize, 2usize);
        let (a_off, b_off) = (0usize, heads);
        let ld = heads * 2;
        let p = random_f32(rows * ld, 281);
        let a_log = random_f32(heads, 282);
        let dt = random_f32(heads, 283);
        let (dg, dbeta) = (random_f32(rows * heads, 284), random_f32(rows * heads, 285));
        let (mut da, mut db) = (vec![0.0f64; rows * heads], vec![0.0f64; rows * heads]);
        let (mut dal, mut ddt) = (vec![0.0f64; heads], vec![0.0f64; heads]);
        for r in 0..rows {
            for h in 0..heads {
                let a = f64::from(p[r * ld + a_off + h]);
                let b = f64::from(p[r * ld + b_off + h]);
                let al = f64::from(a_log[h]);
                let d = f64::from(dt[h]);
                let g = -al.exp() * softplus(a + d);
                let o = r * heads + h;
                let dai = f64::from(dg[o]) * -al.exp() * softplus_grad(a + d);
                da[o] = dai;
                db[o] = f64::from(dbeta[o]) * sigmoid(b) * (1.0 - sigmoid(b));
                dal[h] += f64::from(dg[o]) * g;
                ddt[h] += dai;
            }
        }
        let (pb, alb, dtb) = (buf(rt, &p), buf(rt, &a_log), buf(rt, &dt));
        let (dgb, dbb) = (buf(rt, &dg), buf(rt, &dbeta));
        let dpb = buf(rt, &vec![SENTINEL; rows * ld]);
        let (dalb, ddtb) = (buf(rt, &vec![SENTINEL; heads]), buf(rt, &vec![SENTINEL; heads]));
        let part = buf(
            rt,
            &vec![0.0; qwen35_bwd::gdn_gates_bwd_part_len(rows as u32, heads as u32).max(1)],
        );
        let logits = qwen35::GdnGateLogits {
            buf: &pb,
            ld: ld as u32,
            a_off: a_off as u32,
            b_off: b_off as u32,
        };
        qwen35_bwd::gdn_gates_bwd(
            rt,
            &logits,
            &qwen35::GdnParams {
                a_log: &alb,
                dt_bias: &dtb,
            },
            &dgb,
            &dbb,
            &dpb,
            &dalb,
            &ddtb,
            &part,
            rows as u32,
            heads as u32,
        )
        .unwrap();
        sync(rt);
        let dp = take_f32(&dpb, rows * ld);
        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        for r in 0..rows {
            for h in 0..heads {
                got_a.push(dp[r * ld + a_off + h]);
                got_b.push(dp[r * ld + b_off + h]);
            }
        }
        close_peak("gates da", &got_a, &da, 1e-4);
        close_peak("gates db", &got_b, &db, 1e-4);
        close_peak("gates dA", &take_f32(&dalb, heads), &dal, 1e-4);
        close_peak("gates ddt", &take_f32(&ddtb, heads), &ddt, 1e-4);
    });
    ledger.credit(
        rt,
        "qwen35_scatter_add_rows_f32",
        &["qwen35_scatter_add_rows_f32"],
        || {
            let (width, pos) = (3usize, [1u32, 3]);
            let src = random_f32(pos.len() * width, 291);
            let mut dst = random_f32(4 * width, 292);
            let src_t = tensor_f32(rt, &[pos.len(), width], &src);
            let dst_t = tensor_f32(rt, &[4, width], &dst);
            qwen35_bwd::scatter_add_rows(rt, &src_t, &pos, &dst_t).unwrap();
            sync(rt);
            for (i, &p) in pos.iter().enumerate() {
                for c in 0..width {
                    dst[p as usize * width + c] += src[i * width + c];
                }
            }
            bits_eq("scatter", &dst_t.read_f32().unwrap(), &dst);
        },
    );
    ledger.credit(rt, "qwen35_copy_cols_f32", &["qwen35_copy_cols_f32"], || {
        let (rows, width, src_ld, src_off, dst_ld, dst_off) = (2usize, 3usize, 6usize, 1usize, 5usize, 2usize);
        let src = random_f32(rows * src_ld, 293);
        let mut dst = vec![SENTINEL; rows * dst_ld];
        let (sb, db) = (buf(rt, &src), buf(rt, &dst));
        qwen35_bwd::copy_cols(
            rt,
            Cols {
                buf: &sb,
                ld: src_ld as u32,
                off: src_off as u32,
            },
            Cols {
                buf: &db,
                ld: dst_ld as u32,
                off: dst_off as u32,
            },
            rows as u32,
            width as u32,
        )
        .unwrap();
        sync(rt);
        let got = take_f32(&db, dst.len());
        for r in 0..rows {
            for c in 0..width {
                dst[r * dst_ld + dst_off + c] = src[r * src_ld + src_off + c];
            }
        }
        bits_eq("copy_cols", &got, &dst);
    });
}

fn credit_sq(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    ledger.credit(rt, "sq_norm", &["qwen35_sq_sum_pieces", "qwen35_sum_exact_f32"], || {
        let x = random_f32(17, 301);
        let want: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let xb = buf(rt, &x);
        let got = sq_norm::sq_norm(rt, &[SqWindow::dense(&xb, 0, DType::F32, x.len())], "kcheck sq").unwrap();
        let tol = want.abs().max(1e-6) * 1e-5;
        assert!((got - want).abs() <= tol, "sq_norm got {got} want {want}");
    });
}

fn credit_storage(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let window = |ledger: &mut Ledger, kind: u8, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (rows, width, src_ld, src_off, dst_ld, dst_off) = (3usize, 5usize, 8usize, 1usize, 7usize, 2usize);
            let src_f = random_f32(rows * src_ld, 311 + u64::from(kind));
            let p = rt.pipeline(name).unwrap();
            match kind {
                0 => {
                    let mut dst = vec![SENTINEL; rows * dst_ld];
                    let (sb, db) = (buf(rt, &src_f), buf(rt, &dst));
                    dispatch_2d(rt, &p, width, rows, |bnd| {
                        dispatch::set_gpu_buf(bnd, &sb, 0);
                        dispatch::set_gpu_buf(bnd, &db, 1);
                        dispatch::set_u32(bnd, rows as u32, 2);
                        dispatch::set_u32(bnd, width as u32, 3);
                        dispatch::set_u32(bnd, src_ld as u32, 4);
                        dispatch::set_u32(bnd, src_off as u32, 5);
                        dispatch::set_u32(bnd, dst_ld as u32, 6);
                        dispatch::set_u32(bnd, dst_off as u32, 7);
                    })
                    .unwrap();
                    sync(rt);
                    let got = take_f32(&db, dst.len());
                    for r in 0..rows {
                        for c in 0..width {
                            dst[r * dst_ld + dst_off + c] = src_f[r * src_ld + src_off + c];
                        }
                    }
                    bits_eq(name, &got, &dst);
                }
                1 => {
                    let sb = gpu_bf16(rt, &src_f);
                    let wide = widen_gpu(&src_f);
                    let mut dst = vec![SENTINEL; rows * dst_ld];
                    let db = buf(rt, &dst);
                    dispatch_2d(rt, &p, width, rows, |bnd| {
                        dispatch::set_gpu_buf(bnd, &sb, 0);
                        dispatch::set_gpu_buf(bnd, &db, 1);
                        dispatch::set_u32(bnd, rows as u32, 2);
                        dispatch::set_u32(bnd, width as u32, 3);
                        dispatch::set_u32(bnd, src_ld as u32, 4);
                        dispatch::set_u32(bnd, src_off as u32, 5);
                        dispatch::set_u32(bnd, dst_ld as u32, 6);
                        dispatch::set_u32(bnd, dst_off as u32, 7);
                    })
                    .unwrap();
                    sync(rt);
                    let got = take_f32(&db, dst.len());
                    for r in 0..rows {
                        for c in 0..width {
                            dst[r * dst_ld + dst_off + c] = wide[r * src_ld + src_off + c];
                        }
                    }
                    bits_eq(name, &got, &dst);
                }
                _ => {
                    let sb = buf(rt, &src_f);
                    let n = rows * dst_ld;
                    let init = vec![0.0f32; n];
                    let db = gpu_bf16(rt, &init);
                    dispatch_2d(rt, &p, width, rows, |bnd| {
                        dispatch::set_gpu_buf(bnd, &sb, 0);
                        dispatch::set_gpu_buf(bnd, &db, 1);
                        dispatch::set_u32(bnd, rows as u32, 2);
                        dispatch::set_u32(bnd, width as u32, 3);
                        dispatch::set_u32(bnd, src_ld as u32, 4);
                        dispatch::set_u32(bnd, src_off as u32, 5);
                        dispatch::set_u32(bnd, dst_ld as u32, 6);
                        dispatch::set_u32(bnd, dst_off as u32, 7);
                    })
                    .unwrap();
                    sync(rt);
                    let got = db.contents_u16()[..n].to_vec();
                    let mut want = vec![tessl::tensor::f32_to_bf16_bits_gpu(0.0); n];
                    for r in 0..rows {
                        for c in 0..width {
                            want[r * dst_ld + dst_off + c] =
                                tessl::tensor::f32_to_bf16_bits_gpu(src_f[r * src_ld + src_off + c]);
                        }
                    }
                    assert_eq!(got, want, "{name}");
                }
            }
        });
    };
    window(ledger, 0, "qwen35_window_copy_f32_f32");
    window(ledger, 1, "qwen35_window_copy_bf16_f32");
    window(ledger, 2, "qwen35_window_copy_f32_bf16");

    ledger.credit(
        rt,
        "qwen35_deliver_f32_to_bf16",
        &["qwen35_deliver_f32_to_bf16"],
        || {
            let n = 8usize;
            let src = random_f32(n, 321);
            let dst0 = random_f32(n, 322);
            let dst_bits: Vec<u16> = dst0.iter().copied().map(tessl::tensor::f32_to_bf16_bits_gpu).collect();
            let want: Vec<u16> = src
                .iter()
                .zip(&dst_bits)
                .map(|(&s, &d)| tessl::tensor::f32_to_bf16_bits_gpu(tessl::tensor::bf16_bits_to_f32(d) + s))
                .collect();
            let sb = buf(rt, &src);
            let db = gpu_bf16(rt, &dst0.iter().copied().map(round_gpu).collect::<Vec<_>>());
            let p = rt.pipeline("qwen35_deliver_f32_to_bf16").unwrap();
            dispatch_1d(rt, &p, n, |bnd| {
                dispatch::set_gpu_buf(bnd, &sb, 0);
                dispatch::set_gpu_buf(bnd, &db, 1);
                dispatch::set_u32(bnd, n as u32, 2);
                dispatch::set_u32(bnd, 1, 3);
            })
            .unwrap();
            sync(rt);
            assert_eq!(db.contents_u16()[..n], want[..], "deliver");
        },
    );

    ledger.credit(
        rt,
        "moment_q8",
        &["qwen35_moment_q8_encode", "qwen35_moment_q8_decode"],
        || {
            const N: usize = 256;
            let mut x = random_f32(N, 331);
            x[0] = 1.75;
            x[3] = -1.25;
            let scale = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let codes: Vec<u8> = x.iter().copied().map(|v| q8m_encode(v, scale)).collect();
            let decoded: Vec<f32> = codes.iter().copied().map(|c| q8m_decode(c, scale)).collect();
            let xb = buf(rt, &x);
            let qb = rt.alloc_buffer(N).unwrap();
            qb.write_bytes(&vec![0u8; N]);
            let sb = buf(rt, &[0.0f32]);
            let yb = buf(rt, &vec![SENTINEL; N]);
            let enc = rt.pipeline("qwen35_moment_q8_encode").unwrap();
            let dec = rt.pipeline("qwen35_moment_q8_decode").unwrap();
            dispatch_2d_tg(rt, &enc, 1, 1, 256, |bnd| {
                dispatch::set_gpu_buf(bnd, &xb, 0);
                dispatch::set_gpu_buf(bnd, &qb, 1);
                dispatch::set_gpu_buf(bnd, &sb, 2);
                dispatch::set_u32(bnd, N as u32, 3);
                dispatch::set_u32(bnd, 1, 4);
            })
            .unwrap();
            dispatch_2d_tg(rt, &dec, 1, 1, 256, |bnd| {
                dispatch::set_gpu_buf(bnd, &qb, 0);
                dispatch::set_gpu_buf(bnd, &sb, 1);
                dispatch::set_gpu_buf(bnd, &yb, 2);
                dispatch::set_u32(bnd, N as u32, 3);
                dispatch::set_u32(bnd, 1, 4);
            })
            .unwrap();
            sync(rt);
            assert_eq!(qb.contents_u8()[..N], codes[..], "q8 codes");
            assert_eq!(take_f32(&sb, 1)[0].to_bits(), scale.to_bits(), "q8 scale");
            // Codes and the scale are exact. The decode is `s * (q/127) * |q/127|`
            // and Metal fast-math division lands within an ulp of the host divide.
            let got = take_f32(&yb, N);
            for (i, (&g, &w)) in got.iter().zip(&decoded).enumerate() {
                let tol = 2.0 * f32::EPSILON * g.abs().max(w.abs());
                assert!((g - w).abs() <= tol, "q8 decode[{i}] got {g} want {w}");
            }
        },
    );

    let head = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (hidden, w, v0) = (4usize, 4usize, 1usize);
            let uniq = [1u32, 3];
            let rows = v0 + w;
            let src = random_f32(w * hidden, 341);
            let bank0 = random_f32(rows * hidden, 342);
            let bank_f: Vec<f32> = if bf16 {
                bank0.iter().copied().map(round_gpu).collect()
            } else {
                bank0.clone()
            };
            let mut side = vec![SENTINEL; uniq.len() * hidden];
            let mut bank = bank_f.clone();
            for y in 0..w {
                let row = v0 + y;
                let u = uniq.iter().position(|&q| q as usize == row);
                for x in 0..hidden {
                    let s = src[y * hidden + x];
                    if let Some(u) = u {
                        side[u * hidden + x] = s;
                    } else if bf16 {
                        bank[row * hidden + x] = round_gpu(bank[row * hidden + x] + s);
                    } else {
                        bank[row * hidden + x] += s;
                    }
                }
            }
            let sb = buf(rt, &src);
            let bb = if bf16 { gpu_bf16(rt, &bank_f) } else { buf(rt, &bank_f) };
            let ub = buf_u32(rt, &uniq);
            let side_b = buf(rt, &vec![SENTINEL; uniq.len() * hidden]);
            let p = rt.pipeline(name).unwrap();
            dispatch_2d(rt, &p, hidden, w, |bnd| {
                dispatch::set_gpu_buf(bnd, &sb, 0);
                dispatch::set_gpu_buf(bnd, &bb, 1);
                dispatch::set_gpu_buf(bnd, &ub, 2);
                dispatch::set_gpu_buf(bnd, &side_b, 3);
                dispatch::set_u32(bnd, uniq.len() as u32, 4);
                dispatch::set_u32(bnd, v0 as u32, 5);
                dispatch::set_u32(bnd, w as u32, 6);
                dispatch::set_u32(bnd, hidden as u32, 7);
                dispatch::set_u32(bnd, 1, 8);
                dispatch::set_u32(bnd, 1, 9);
            })
            .unwrap();
            sync(rt);
            let got_side = take_f32(&side_b, side.len());
            bits_eq(&format!("{name} side"), &got_side, &side);
            if bf16 {
                bits_eq(name, &take_bf16(&bb, bank.len()), &bank);
            } else {
                bits_eq(name, &take_f32(&bb, bank.len()), &bank);
            }
        });
    };
    head(ledger, false, "qwen35_head_rows_to_bank_f32");
    head(ledger, true, "qwen35_head_rows_to_bank_bf16");

    let token = |ledger: &mut Ledger, bf16: bool, name: &'static str| {
        ledger.credit(rt, name, &[name], || {
            let (hidden, rows) = (4usize, 4usize);
            let uniq = [0u32, 2];
            let side = random_f32(uniq.len() * hidden, 351);
            let bank0 = random_f32(rows * hidden, 352);
            let bank_f: Vec<f32> = if bf16 {
                bank0.iter().copied().map(round_gpu).collect()
            } else {
                bank0
            };
            let mut bank = bank_f.clone();
            for (u, &row) in uniq.iter().enumerate() {
                for x in 0..hidden {
                    let s = side[u * hidden + x];
                    let i = row as usize * hidden + x;
                    bank[i] = if bf16 { round_gpu(bank[i] + s) } else { bank[i] + s };
                }
            }
            let sb = buf(rt, &side);
            let bb = if bf16 { gpu_bf16(rt, &bank_f) } else { buf(rt, &bank_f) };
            let ub = buf_u32(rt, &uniq);
            let p = rt.pipeline(name).unwrap();
            dispatch_2d(rt, &p, hidden, uniq.len(), |bnd| {
                dispatch::set_gpu_buf(bnd, &sb, 0);
                dispatch::set_gpu_buf(bnd, &bb, 1);
                dispatch::set_gpu_buf(bnd, &ub, 2);
                dispatch::set_u32(bnd, uniq.len() as u32, 3);
                dispatch::set_u32(bnd, hidden as u32, 4);
                dispatch::set_u32(bnd, 1, 5);
            })
            .unwrap();
            sync(rt);
            if bf16 {
                bits_eq(name, &take_bf16(&bb, bank.len()), &bank);
            } else {
                bits_eq(name, &take_f32(&bb, bank.len()), &bank);
            }
        });
    };
    token(ledger, false, "qwen35_token_rows_to_bank_f32");
    token(ledger, true, "qwen35_token_rows_to_bank_bf16");
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    credit_conv(rt, ledger);
    credit_gates(rt, ledger);
    credit_gdn(rt, ledger);
    credit_gated_norm(rt, ledger);
    credit_qk(rt, ledger);
    credit_gate_act(rt, ledger);
    credit_residual_norm(rt, ledger);
    credit_prefill(rt, ledger);
    credit_train(rt, ledger);
    credit_prefix(rt, ledger);
    credit_score_embed(rt, ledger);
    credit_bwd(rt, ledger);
    credit_sq(rt, ledger);
    credit_storage(rt, ledger);
}
