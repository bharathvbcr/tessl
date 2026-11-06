//! Head-size pairs llama.cpp's Metal tensor flash attention ships.
//!
//! Equal sizes 64, 128, 256 and 512 go through the shipped prefill and decode
//! entries. 192/128 and 576/512 go through the key/value entries. Head size 32
//! stays on `gqa_decode`. Each result is checked against an f64 attention
//! reference. An unsupported pair is named and does not dispatch.

mod common;

use std::sync::Arc;

use common::{buf, buf_u32, seeded, with_gpu};
use tessl::nn::{self, AttnDims};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const B: usize = 1;
const H: usize = 2;
const HKV: usize = 1;
const T: usize = 4;

fn wave(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013 + seed).sin() * 0.2).collect()
}

/// Causal scaled dot-product attention in f64. Query `t` attends to keys whose
/// absolute position is at most the query's. Scale is `1/sqrt(key_dim)`.
fn causal_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tq: usize,
    tkv: usize,
    dk: usize,
    dv: usize,
    q_pos_offset: usize,
    kv_pos_offset: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dk as f64).sqrt();
    let group = H / HKV;
    let mut out = vec![0.0f32; B * tq * H * dv];
    for t in 0..tq {
        for h in 0..H {
            let hk = h / group;
            let q_abs = q_pos_offset + t;
            let qs = ((t * H) + h) * dk;
            let mut best = f64::NEG_INFINITY;
            let mut scores = vec![f64::NEG_INFINITY; tkv];
            for tk in 0..tkv {
                let k_abs = kv_pos_offset + tk;
                if k_abs > q_abs {
                    continue;
                }
                let ks = (tk * HKV + hk) * dk;
                let mut dot = 0.0f64;
                for x in 0..dk {
                    dot += f64::from(q[qs + x]) * f64::from(k[ks + x]);
                }
                let s = dot * scale;
                scores[tk] = s;
                best = best.max(s);
            }
            let mut denom = 0.0f64;
            let mut acc = vec![0.0f64; dv];
            if best.is_finite() {
                for tk in 0..tkv {
                    if !scores[tk].is_finite() {
                        continue;
                    }
                    let p = (scores[tk] - best).exp();
                    denom += p;
                    let vs = (tk * HKV + hk) * dv;
                    for x in 0..dv {
                        acc[x] += p * f64::from(v[vs + x]);
                    }
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let os = (t * H + h) * dv;
            for x in 0..dv {
                out[os + x] = (acc[x] * inv) as f32;
            }
        }
    }
    out
}

/// Full-sequence grouped-query attention in f64. No causal window. Scale is
/// `1/sqrt(head_dim)`, which is what `gqa_decode` applies inside the kernel.
fn gqa_reference(q: &[f32], k: &[f32], v: &[f32], d: usize, kv_len: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f64).sqrt();
    let group = H / HKV;
    let mut out = vec![0.0f32; H * d];
    for h in 0..H {
        let hk = h / group;
        let qs = h * d;
        let mut best = f64::NEG_INFINITY;
        let mut scores = vec![0.0f64; kv_len];
        for t in 0..kv_len {
            let ks = (t * HKV + hk) * d;
            let mut dot = 0.0f64;
            for x in 0..d {
                dot += f64::from(q[qs + x]) * f64::from(k[ks + x]);
            }
            scores[t] = dot * scale;
            best = best.max(scores[t]);
        }
        let mut denom = 0.0f64;
        let mut acc = vec![0.0f64; d];
        for t in 0..kv_len {
            let p = (scores[t] - best).exp();
            denom += p;
            let vs = (t * HKV + hk) * d;
            for x in 0..d {
                acc[x] += p * f64::from(v[vs + x]);
            }
        }
        let inv = 1.0 / denom;
        for x in 0..d {
            out[h * d + x] = (acc[x] * inv) as f32;
        }
    }
    out
}

fn positions(buf: &GpuBuffer, dim: usize) -> usize {
    (buf.nbytes() / std::mem::size_of::<f32>()) / (HKV * dim)
}

fn shared_capacity(k: &GpuBuffer, v: &GpuBuffer, dk: usize, dv: usize) -> usize {
    let kp = positions(k, dk);
    let vp = positions(v, dv);
    assert!(kp >= T && vp >= T, "bucket left k={kp} v={vp} positions, need {T}");
    if dk == dv {
        assert_eq!(kp, vp, "equal head widths disagreed on KV capacity");
        kp
    } else {
        kp.min(vp)
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f32], wide: bool) {
    assert_eq!(got.len(), want.len(), "{what} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what} [{i}] non-finite {g}");
        let tol = if wide {
            1.0e-3 * w.abs().max(1.0) + 1.0e-4
        } else {
            2.0e-4 * w.abs().max(1.0)
        };
        assert!((*g - *w).abs() <= tol, "{what} [{i}] got {g} want {w} tol {tol}");
    }
}

fn check_rows(rt: &Arc<GpuRuntime>, dk: usize, dv: usize, kv_api: bool) {
    let q = wave(T * H * dk, 0.2);
    let k = wave(T * HKV * dk, 1.1);
    let v = wave(T * HKV * dv, 2.4);
    let qb = buf(rt, &q);
    let kb = buf(rt, &k);
    let vb = buf(rt, &v);
    let o = seeded(rt, T * H * dv, -3.0e20);
    let tkv = buf_u32(rt, &[T as u32]);
    let zero = buf_u32(rt, &[0]);
    let dims = AttnDims {
        batch: B as u32,
        tq: T as u32,
        heads: H as u32,
        heads_kv: HKV as u32,
        window: 0,
        scale: 1.0 / (dk as f32).sqrt(),
    };
    if kv_api {
        nn::flash_attn_rows_kv(
            rt, &qb, &kb, &vb, &o, &tkv, &zero, &zero, dims, dk as u32, dv as u32, false,
        )
        .unwrap_or_else(|e| panic!("rows {dk}/{dv}: {e}"));
    } else {
        nn::flash_attn_rows(rt, &qb, &kb, &vb, &o, &tkv, &zero, &zero, dims, dk as u32, false)
            .unwrap_or_else(|e| panic!("rows {dk}: {e}"));
    }
    rt.synchronize().unwrap_or_else(|e| panic!("rows sync {dk}/{dv}: {e}"));
    let want = causal_sdpa(&q, &k, &v, T, T, dk, dv, 0, 0);
    let got = o.read_f32();
    assert_close(&format!("rows {dk}/{dv}"), &got[..want.len()], &want, dk.max(dv) >= 256);
}

fn check_decode(rt: &Arc<GpuRuntime>, dk: usize, dv: usize, kv_api: bool) {
    let q = wave(H * dk, 0.4);
    let k = wave(T * HKV * dk, 0.9);
    let v = wave(T * HKV * dv, 1.8);
    let qb = buf(rt, &q);
    let kb = buf(rt, &k);
    let vb = buf(rt, &v);
    let o = seeded(rt, H * dv, -3.0e20);
    let tkv = buf_u32(rt, &[T as u32]);
    let q_pos = buf_u32(rt, &[(T - 1) as u32]);
    let zero = buf_u32(rt, &[0]);
    let cap = shared_capacity(&kb, &vb, dk, dv);
    let scratch = nn::DecodeScratch::new(rt, B as u32, H as u32, cap, dv as u32)
        .unwrap_or_else(|e| panic!("scratch {dk}/{dv}: {e}"));
    let dims = AttnDims {
        batch: B as u32,
        tq: 1,
        heads: H as u32,
        heads_kv: HKV as u32,
        window: 0,
        scale: 1.0 / (dk as f32).sqrt(),
    };
    if kv_api {
        nn::flash_attn_decode_kv(
            rt, &qb, &kb, &vb, &o, &scratch, &tkv, &q_pos, &zero, dims, dk as u32, dv as u32, cap, false,
        )
        .unwrap_or_else(|e| panic!("decode {dk}/{dv}: {e}"));
    } else {
        nn::flash_attn_decode(
            rt, &qb, &kb, &vb, &o, &scratch, &tkv, &q_pos, &zero, dims, dk as u32, cap, false,
        )
        .unwrap_or_else(|e| panic!("decode {dk}: {e}"));
    }
    rt.synchronize()
        .unwrap_or_else(|e| panic!("decode sync {dk}/{dv}: {e}"));
    let want = causal_sdpa(&q, &k, &v, 1, T, dk, dv, T - 1, 0);
    let got = o.read_f32();
    assert_close(
        &format!("decode {dk}/{dv}"),
        &got[..want.len()],
        &want,
        dk.max(dv) >= 256,
    );
}

fn check_gqa32(rt: &Arc<GpuRuntime>) {
    let d = 32;
    let q = wave(H * d, 0.3);
    let k = wave(T * HKV * d, 1.4);
    let v = wave(T * HKV * d, 2.2);
    let qb = buf(rt, &q);
    let kb = buf(rt, &k);
    let vb = buf(rt, &v);
    let o = seeded(rt, H * d, -3.0e20);
    nn::gqa_decode(
        rt, &qb, &kb, &vb, &o, B as u32, H as u32, HKV as u32, T as u32, d as u32,
    )
    .unwrap_or_else(|e| panic!("gqa 32: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("gqa sync: {e}"));
    let want = gqa_reference(&q, &k, &v, d, T);
    let got = o.read_f32();
    assert_close("gqa 32", &got[..want.len()], &want, false);
}

fn refused_pair(rt: &Arc<GpuRuntime>) {
    let b = rt.alloc_buffer(256).expect("alloc");
    let scratch = nn::DecodeScratch::new(rt, 1, 1, 4, 80).expect("scratch");
    let tkv = buf_u32(rt, &[T as u32]);
    let zero = buf_u32(rt, &[0]);
    let rows = AttnDims {
        batch: 1,
        tq: T as u32,
        heads: 1,
        heads_kv: 1,
        window: 0,
        scale: 1.0,
    };
    rt.take_dispatch_count();
    let err =
        nn::flash_attn_rows_kv(rt, &b, &b, &b, &b, &tkv, &zero, &zero, rows, 96, 80, false).expect_err("rows 96/80");
    assert!(err.contains("96/80"), "rows error does not name the pair: {err}");
    assert_eq!(rt.take_dispatch_count(), 0, "rows 96/80 encoded: {err}");
    let decode = AttnDims { tq: 1, ..rows };
    let err = nn::flash_attn_decode_kv(
        rt, &b, &b, &b, &b, &scratch, &tkv, &zero, &zero, decode, 96, 80, 4, false,
    )
    .expect_err("decode 96/80");
    assert!(err.contains("96/80"), "decode error does not name the pair: {err}");
    assert_eq!(rt.take_dispatch_count(), 0, "decode 96/80 encoded: {err}");
}

#[test]
fn llama_cpp_head_pairs_match_an_f64_reference() {
    with_gpu(|rt| {
        for d in [64usize, 128, 256, 512] {
            check_rows(rt, d, d, false);
            check_decode(rt, d, d, false);
        }
        for (dk, dv) in [(192usize, 128usize), (576, 512)] {
            check_rows(rt, dk, dv, true);
            check_decode(rt, dk, dv, true);
        }
        check_gqa32(rt);
        refused_pair(rt);
    });
}
