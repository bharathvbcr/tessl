//! Flash-attention rows, decode, tiled SWA/global, and gqa_decode.

use std::sync::Arc;

use crate::common::{buf, buf_u32, seeded};
use tessl::nn::{self, AttnDims, AttnHeadDim, DecodeChunk, RowsGroups, RowsLanes};
use tessl::GpuRuntime;

use super::Ledger;

const B: usize = 1;
const H: usize = 2;
const HKV: usize = 1;
const T: usize = 4;

fn wave(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013 + seed).sin() * 0.2).collect()
}

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

fn positions(buf: &tessl::GpuBuffer, dim: usize) -> usize {
    (buf.nbytes() / std::mem::size_of::<f32>()) / (HKV * dim)
}

fn shared_capacity(k: &tessl::GpuBuffer, v: &tessl::GpuBuffer, dk: usize, dv: usize) -> usize {
    let kp = positions(k, dk);
    let vp = positions(v, dv);
    assert!(kp >= T && vp >= T, "bucket left k={kp} v={vp} positions, need {T}");
    if dk == dv {
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
        assert!((g - w).abs() <= tol, "{what} [{i}] got {g} want {w} tol {tol}");
    }
}

fn dims(tq: u32, dk: usize) -> AttnDims {
    AttnDims {
        batch: B as u32,
        tq,
        heads: H as u32,
        heads_kv: HKV as u32,
        window: 0,
        scale: 1.0 / (dk as f32).sqrt(),
    }
}

fn lane_n(r: RowsLanes) -> u32 {
    match r {
        RowsLanes::R8 => 8,
        RowsLanes::R16 => 16,
        RowsLanes::R32 => 32,
    }
}

fn group_n(g: RowsGroups) -> u32 {
    match g {
        RowsGroups::G8 => 8,
        RowsGroups::G16 => 16,
        RowsGroups::G32 => 32,
    }
}

fn chunk_n(c: DecodeChunk) -> u32 {
    match c {
        DecodeChunk::C64 => 64,
        DecodeChunk::C128 => 128,
        DecodeChunk::C256 => 256,
    }
}

fn rows_equal(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, d: usize, lanes: RowsLanes, groups: RowsGroups) {
    let name = format!("flash_attn_rows_h{d}_r{}_g{}", lane_n(lanes), group_n(groups));
    let q = wave(T * H * d, 0.2);
    let k = wave(T * HKV * d, 1.1);
    let v = wave(T * HKV * d, 2.4);
    let want = causal_sdpa(&q, &k, &v, T, T, d, d, 0, 0);
    let wide = d >= 256;
    let label = name.clone();
    ledger.credit(rt, &label, &[name], || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, T * H * d, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let zero = buf_u32(rt, &[0]);
        nn::flash_attn_rows_with_lanes(
            rt,
            &qb,
            &kb,
            &vb,
            &o,
            &tkv,
            &zero,
            &zero,
            dims(T as u32, d),
            d as u32,
            lanes,
            groups,
            false,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{label} sync: {e}"));
        assert_close(&label, &o.read_f32()[..want.len()], &want, wide);
    });
}

fn rows_kv(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, dk: usize, dv: usize, name: &str) {
    let q = wave(T * H * dk, 0.2);
    let k = wave(T * HKV * dk, 1.1);
    let v = wave(T * HKV * dv, 2.4);
    let want = causal_sdpa(&q, &k, &v, T, T, dk, dv, 0, 0);
    let wide = dk.max(dv) >= 256;
    ledger.credit(rt, name, &[name], || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, T * H * dv, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let zero = buf_u32(rt, &[0]);
        nn::flash_attn_rows_kv(
            rt,
            &qb,
            &kb,
            &vb,
            &o,
            &tkv,
            &zero,
            &zero,
            dims(T as u32, dk),
            dk as u32,
            dv as u32,
            false,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{name} sync: {e}"));
        assert_close(name, &o.read_f32()[..want.len()], &want, wide);
    });
}

fn decode_equal(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, d: usize, chunk: DecodeChunk, lanes: RowsLanes) {
    let partial = format!("flash_attn_decode_partial_h{d}_c{}_r{}", chunk_n(chunk), lane_n(lanes));
    let reduce = format!("flash_attn_decode_reduce_h{d}_c{}", chunk_n(chunk));
    let names = [partial, reduce];
    let q = wave(H * d, 0.4);
    let k = wave(T * HKV * d, 0.9);
    let v = wave(T * HKV * d, 1.8);
    let want = causal_sdpa(&q, &k, &v, 1, T, d, d, T - 1, 0);
    let wide = d >= 256;
    let label = names[0].clone();
    ledger.credit(rt, &label, &names, || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, H * d, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let q_pos = buf_u32(rt, &[(T - 1) as u32]);
        let zero = buf_u32(rt, &[0]);
        let cap = shared_capacity(&kb, &vb, d, d);
        let scratch = nn::DecodeScratch::with_chunk(rt, B as u32, H as u32, cap, d as u32, chunk)
            .unwrap_or_else(|e| panic!("{label} scratch: {e}"));
        nn::flash_attn_decode_with_chunk(
            rt,
            &qb,
            &kb,
            &vb,
            &o,
            &scratch,
            &tkv,
            &q_pos,
            &zero,
            dims(1, d),
            d as u32,
            cap,
            chunk,
            lanes,
            None,
            None,
            false,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{label} sync: {e}"));
        assert_close(&label, &o.read_f32()[..want.len()], &want, wide);
    });
}

fn decode_kv(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, dk: usize, dv: usize, partial: &str, reduce: &str) {
    let names = [partial, reduce];
    let q = wave(H * dk, 0.4);
    let k = wave(T * HKV * dk, 0.9);
    let v = wave(T * HKV * dv, 1.8);
    let want = causal_sdpa(&q, &k, &v, 1, T, dk, dv, T - 1, 0);
    let wide = dk.max(dv) >= 256;
    ledger.credit(rt, partial, &names, || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, H * dv, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let q_pos = buf_u32(rt, &[(T - 1) as u32]);
        let zero = buf_u32(rt, &[0]);
        let cap = shared_capacity(&kb, &vb, dk, dv);
        let scratch = nn::DecodeScratch::new(rt, B as u32, H as u32, cap, dv as u32)
            .unwrap_or_else(|e| panic!("{partial} scratch: {e}"));
        nn::flash_attn_decode_kv(
            rt,
            &qb,
            &kb,
            &vb,
            &o,
            &scratch,
            &tkv,
            &q_pos,
            &zero,
            dims(1, dk),
            dk as u32,
            dv as u32,
            cap,
            false,
        )
        .unwrap_or_else(|e| panic!("{partial}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{partial} sync: {e}"));
        assert_close(partial, &o.read_f32()[..want.len()], &want, wide);
    });
}

fn swa(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, head: AttnHeadDim) {
    let d = match head {
        AttnHeadDim::D128 => 128,
        AttnHeadDim::D256 => 256,
    };
    let name = head.kernel();
    let q = wave(T * H * d, 0.25);
    let k = wave(T * HKV * d, 0.7);
    let v = wave(T * HKV * d, 1.3);
    let want = causal_sdpa(&q, &k, &v, T, T, d, d, 0, 0);
    ledger.credit(rt, name, &[name], || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, T * H * d, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let zero = buf_u32(rt, &[0]);
        nn::flash_attn_swa_tiled(rt, head, &qb, &kb, &vb, &o, &tkv, &zero, &zero, dims(T as u32, d))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{name} sync: {e}"));
        assert_close(name, &o.read_f32()[..want.len()], &want, d >= 256);
    });
}

fn global_h512(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let d = 512;
    let name = "flash_attn_global_h512";
    let q = wave(T * H * d, 0.25);
    let k = wave(T * HKV * d, 0.7);
    let v = wave(T * HKV * d, 1.3);
    let want = causal_sdpa(&q, &k, &v, T, T, d, d, 0, 0);
    ledger.credit(rt, name, &[name], || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, T * H * d, -3.0e20);
        let tkv = buf_u32(rt, &[T as u32]);
        let zero = buf_u32(rt, &[0]);
        nn::flash_attn_global_h512_tiled(rt, &qb, &kb, &vb, &o, &tkv, &zero, &zero, dims(T as u32, d), false)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{name} sync: {e}"));
        assert_close(name, &o.read_f32()[..want.len()], &want, true);
    });
}

fn gqa(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, d: usize) {
    let name = format!("gqa_decode_h{d}");
    let q = wave(H * d, 0.3);
    let k = wave(T * HKV * d, 1.4);
    let v = wave(T * HKV * d, 2.2);
    let want = gqa_reference(&q, &k, &v, d, T);
    let label = name.clone();
    ledger.credit(rt, &label, &[name], || {
        let qb = buf(rt, &q);
        let kb = buf(rt, &k);
        let vb = buf(rt, &v);
        let o = seeded(rt, H * d, -3.0e20);
        nn::gqa_decode(
            rt, &qb, &kb, &vb, &o, B as u32, H as u32, HKV as u32, T as u32, d as u32,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        rt.synchronize().unwrap_or_else(|e| panic!("{label} sync: {e}"));
        assert_close(&label, &o.read_f32()[..want.len()], &want, d >= 256);
    });
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let lanes = [RowsLanes::R8, RowsLanes::R16, RowsLanes::R32];
    let groups = [RowsGroups::G8, RowsGroups::G16, RowsGroups::G32];
    let chunks = [DecodeChunk::C64, DecodeChunk::C128, DecodeChunk::C256];
    for d in [128usize, 256, 512] {
        for lanes in lanes {
            for groups in groups {
                rows_equal(rt, ledger, d, lanes, groups);
            }
        }
        for chunk in chunks {
            for lanes in lanes {
                decode_equal(rt, ledger, d, chunk, lanes);
            }
        }
    }
    rows_equal(rt, ledger, 64, RowsLanes::R8, RowsGroups::G8);
    decode_equal(rt, ledger, 64, DecodeChunk::C256, RowsLanes::R8);
    rows_kv(rt, ledger, 192, 128, "flash_attn_rows_dk192_dv128_r8_g8");
    rows_kv(rt, ledger, 576, 512, "flash_attn_rows_dk576_dv512_r8_g8");
    decode_kv(
        rt,
        ledger,
        192,
        128,
        "flash_attn_decode_partial_dk192_dv128_c256_r8",
        "flash_attn_decode_reduce_h128_c256",
    );
    decode_kv(
        rt,
        ledger,
        576,
        512,
        "flash_attn_decode_partial_dk576_dv512_c128_r16",
        "flash_attn_decode_reduce_h512_c128",
    );
    swa(rt, ledger, AttnHeadDim::D128);
    swa(rt, ledger, AttnHeadDim::D256);
    global_h512(rt, ledger);
    for d in [32usize, 64, 128, 256, 512] {
        gqa(rt, ledger, d);
    }
}
