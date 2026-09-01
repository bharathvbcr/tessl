//! Numeric tests for the fused RMSNorm → QKV → RoPE kernels.
//!
//! Three entry points had a name check and error-path coverage, and nothing
//! that ran them for a number.
//!
//! The reference transcribes the kernel's own definitions:
//!
//! * RMSNorm per head row: `x * rsqrt(mean(x^2) + eps) * weight`, weights
//!   shared across heads.
//! * RoPE is proportional NeoX / MLX `traditional=False`: pair `x[i]` with
//!   `x[i + D/2]` for `i < rotary_dim/2`, with `inv_freq = theta^(-2i/D)` —
//!   the denominator is the **full** head dim, not `rotary_dim`, which is what
//!   makes it proportional rather than a plain truncation.
//! * V is normalized only. No RoPE, no attention scale.
//!
//! Layout is `q [T, Hq, D]` and `k`/`v` `[T, Hkv, D]`, all rewritten in place.

mod common;

use common::{buf, random_f32, seeded, with_gpu};
use std::sync::Arc;
use tessl::nn::{self, KvStoreTarget, QkvBuffers, QkvRopeDims, QkvRopeVariant};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const UNWRITTEN: f32 = -5.5e28;

#[derive(Clone, Copy)]
struct Dims {
    t: usize,
    hq: usize,
    hkv: usize,
    d: usize,
    rotary: usize,
    theta: f32,
    eps: f32,
}

fn rms_norm(row: &mut [f32], weight: &[f32], eps: f32) {
    let d = row.len();
    let ss: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = 1.0 / (ss / d as f64 + eps as f64).sqrt();
    for (x, w) in row.iter_mut().zip(weight) {
        *x = (*x as f64 * inv * *w as f64) as f32;
    }
}

fn rope(row: &mut [f32], rotary_dim: usize, pos: usize, theta: f32) {
    let d = row.len();
    let half = d / 2;
    for i in 0..rotary_dim / 2 {
        let inv_freq = 1.0 / (theta as f64).powf(2.0 * i as f64 / d as f64);
        let angle = pos as f64 * inv_freq;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (row[i] as f64, row[i + half] as f64);
        row[i] = (x0 * c - x1 * s) as f32;
        row[i + half] = (x0 * s + x1 * c) as f32;
    }
}

/// Returns the expected `(q, k, v)` after the fused pass.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qw: &[f32],
    kw: &[f32],
    vw: &[f32],
    dm: Dims,
    pos_offset: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut q, mut k, mut v) = (q.to_vec(), k.to_vec(), v.to_vec());
    for t in 0..dm.t {
        for h in 0..dm.hq {
            let row = &mut q[(t * dm.hq + h) * dm.d..(t * dm.hq + h + 1) * dm.d];
            rms_norm(row, qw, dm.eps);
            rope(row, dm.rotary, pos_offset + t, dm.theta);
        }
        for h in 0..dm.hkv {
            let row = &mut k[(t * dm.hkv + h) * dm.d..(t * dm.hkv + h + 1) * dm.d];
            rms_norm(row, kw, dm.eps);
            rope(row, dm.rotary, pos_offset + t, dm.theta);
            // V is normalized and left unrotated.
            let row = &mut v[(t * dm.hkv + h) * dm.d..(t * dm.hkv + h + 1) * dm.d];
            rms_norm(row, vw, dm.eps);
        }
    }
    (q, k, v)
}

fn u32_buf(rt: &Arc<GpuRuntime>, v: u32) -> GpuBuffer {
    let b = rt.alloc_buffer(4).unwrap();
    b.write_u32(&[v]);
    b
}

fn close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        assert!(*g != UNWRITTEN, "{label}[{i}]: never written");
        let tol = 3e-5 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{label}[{i}]: got {g} want {w}");
    }
}

struct Fixture {
    dm: Dims,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qw: Vec<f32>,
    kw: Vec<f32>,
    vw: Vec<f32>,
}

fn fixture(dm: Dims, seed: u64) -> Fixture {
    Fixture {
        dm,
        q: random_f32(dm.t * dm.hq * dm.d, seed),
        k: random_f32(dm.t * dm.hkv * dm.d, seed + 1),
        v: random_f32(dm.t * dm.hkv * dm.d, seed + 2),
        // Weights away from zero, so a dropped weight multiply is visible
        // rather than hidden by a near-unit factor.
        qw: random_f32(dm.d, seed + 3).iter().map(|w| 1.0 + w).collect(),
        kw: random_f32(dm.d, seed + 4).iter().map(|w| 0.5 + w).collect(),
        vw: random_f32(dm.d, seed + 5).iter().map(|w| 1.5 + w).collect(),
    }
}

impl Fixture {
    fn upload(&self, rt: &Arc<GpuRuntime>) -> (GpuBuffer, GpuBuffer, GpuBuffer) {
        (buf(rt, &self.q), buf(rt, &self.k), buf(rt, &self.v))
    }
    fn dims(&self) -> QkvRopeDims {
        QkvRopeDims {
            t: self.dm.t as u32,
            heads_q: self.dm.hq as u32,
            heads_kv: self.dm.hkv as u32,
            head_dim: self.dm.d as u32,
            rotary_dim: self.dm.rotary as u32,
            theta: self.dm.theta,
            eps: self.dm.eps,
        }
    }
}

/// The constant-position variant against the f64 reference, at full rotary and
/// at a partial one.
#[test]
fn rms_qkv_rope_matches_an_f64_reference() {
    with_gpu(|rt| {
        for &(t, hq, hkv, d, rotary) in &[
            (3usize, 4usize, 2usize, 64usize, 64usize),
            // rotary_dim < head_dim is the proportional-RoPE case: only the
            // first rotary_dim/2 pairs rotate and the rest must be left alone,
            // which a reference that used rotary_dim in the inv_freq
            // denominator would get wrong in the rotated half too.
            (2, 2, 1, 128, 64),
            (1, 8, 8, 32, 16),
        ] {
            let dm = Dims {
                t,
                hq,
                hkv,
                d,
                rotary,
                theta: 10_000.0,
                eps: 1e-6,
            };
            let f = fixture(dm, 0x11 + d as u64 + rotary as u64);
            let pos = 7usize;
            let (qb, kb, vb) = f.upload(rt);
            let (qwb, kwb, vwb) = (buf(rt, &f.qw), buf(rt, &f.kw), buf(rt, &f.vw));

            nn::rms_qkv_rope(
                rt,
                QkvRopeVariant::PosConst,
                QkvBuffers {
                    q: &qb,
                    k: &kb,
                    v: &vb,
                    q_weight: &qwb,
                    k_weight: &kwb,
                    v_weight: &vwb,
                },
                f.dims(),
                pos as u32,
                None,
                None,
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();

            let (wq, wk, wv) = reference(&f.q, &f.k, &f.v, &f.qw, &f.kw, &f.vw, dm, pos);
            let label = format!("d={d} rotary={rotary}");
            close(&format!("q {label}"), &qb.read_f32()[..wq.len()], &wq);
            close(&format!("k {label}"), &kb.read_f32()[..wk.len()], &wk);
            close(&format!("v {label}"), &vb.read_f32()[..wv.len()], &wv);
        }
    });
}

/// V must be normalized and *not* rotated. A kernel that applied RoPE to V
/// would still produce finite, plausible numbers, so this is checked directly
/// rather than left to the combined comparison above.
#[test]
fn v_is_normalized_but_never_rotated() {
    with_gpu(|rt| {
        let dm = Dims {
            t: 4,
            hq: 2,
            hkv: 2,
            d: 64,
            rotary: 64,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let f = fixture(dm, 0x22);
        let (qb, kb, vb) = f.upload(rt);
        let (qwb, kwb, vwb) = (buf(rt, &f.qw), buf(rt, &f.kw), buf(rt, &f.vw));

        // A non-zero position: with pos = 0 every angle is 0 and RoPE is the
        // identity, so V would look unrotated no matter what the kernel did.
        nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosConst,
            QkvBuffers {
                q: &qb,
                k: &kb,
                v: &vb,
                q_weight: &qwb,
                k_weight: &kwb,
                v_weight: &vwb,
            },
            f.dims(),
            13,
            None,
            None,
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let mut want = f.v.clone();
        for t in 0..dm.t {
            for h in 0..dm.hkv {
                let row = &mut want[(t * dm.hkv + h) * dm.d..(t * dm.hkv + h + 1) * dm.d];
                rms_norm(row, &f.vw, dm.eps);
            }
        }
        close("v norm-only", &vb.read_f32()[..want.len()], &want);

        // And the same values *with* RoPE must differ, or the check above is
        // satisfied by a rotation that happens to be the identity.
        let mut rotated = want.clone();
        for t in 0..dm.t {
            for h in 0..dm.hkv {
                let row = &mut rotated[(t * dm.hkv + h) * dm.d..(t * dm.hkv + h + 1) * dm.d];
                rope(row, dm.rotary, 13 + t, dm.theta);
            }
        }
        assert!(
            rotated.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-4),
            "the fixture's RoPE is the identity here, so this test proves nothing"
        );
    });
}

/// `PosBuffer` reads the position from a device `u32`. Same math as
/// `PosConst`, and the two must agree bit for bit at the same position — that
/// is the contract the ICB path relies on.
#[test]
fn pos_buffer_variant_agrees_bit_for_bit_with_the_constant_one() {
    with_gpu(|rt| {
        let dm = Dims {
            t: 3,
            hq: 4,
            hkv: 2,
            d: 64,
            rotary: 64,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let f = fixture(dm, 0x33);
        let (qwb, kwb, vwb) = (buf(rt, &f.qw), buf(rt, &f.kw), buf(rt, &f.vw));
        let pos = 21u32;

        let mut outs = Vec::new();
        for use_buf in [false, true] {
            let (qb, kb, vb) = f.upload(rt);
            let pb = u32_buf(rt, pos);
            nn::rms_qkv_rope(
                rt,
                if use_buf {
                    QkvRopeVariant::PosBuffer
                } else {
                    QkvRopeVariant::PosConst
                },
                QkvBuffers {
                    q: &qb,
                    k: &kb,
                    v: &vb,
                    q_weight: &qwb,
                    k_weight: &kwb,
                    v_weight: &vwb,
                },
                f.dims(),
                pos,
                if use_buf { Some(&pb) } else { None },
                None,
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();
            outs.push((qb.read_f32(), kb.read_f32(), vb.read_f32()));
        }
        for (name, a, b) in [
            ("q", &outs[0].0, &outs[1].0),
            ("k", &outs[0].1, &outs[1].1),
            ("v", &outs[0].2, &outs[1].2),
        ] {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name}[{i}]: PosConst {x} vs PosBuffer {y}"
                );
            }
        }
    });
}

/// `PosBufferKvStore` does the same work and additionally writes the rotated K
/// and V into the cache at a device-side offset. The cache contents must equal
/// what the two-pass path would have stored, and nothing outside the slot may
/// be touched.
#[test]
fn kv_store_variant_writes_the_rotated_k_and_v_into_the_cache() {
    with_gpu(|rt| {
        let dm = Dims {
            t: 2,
            hq: 4,
            hkv: 2,
            d: 64,
            rotary: 64,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let f = fixture(dm, 0x44);
        let (qwb, kwb, vwb) = (buf(rt, &f.qw), buf(rt, &f.kw), buf(rt, &f.vw));
        let kv_elems = dm.t * dm.hkv * dm.d;
        let pos = 5u32;
        let offset = kv_elems; // one full block in, so the guard has both sides

        let (qb, kb, vb) = f.upload(rt);
        let pb = u32_buf(rt, pos);
        let dst_k = seeded(rt, kv_elems * 3, UNWRITTEN);
        let dst_v = seeded(rt, kv_elems * 3, UNWRITTEN);
        let off = u32_buf(rt, offset as u32);

        nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosBufferKvStore,
            QkvBuffers {
                q: &qb,
                k: &kb,
                v: &vb,
                q_weight: &qwb,
                k_weight: &kwb,
                v_weight: &vwb,
            },
            f.dims(),
            pos,
            Some(&pb),
            Some(KvStoreTarget {
                dst_k: &dst_k,
                dst_v: &dst_v,
                dst_offset: &off,
            }),
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let (_, wk, wv) = reference(&f.q, &f.k, &f.v, &f.qw, &f.kw, &f.vw, dm, pos as usize);
        let (gk, gv) = (dst_k.read_f32(), dst_v.read_f32());
        close("cache k", &gk[offset..offset + kv_elems], &wk);
        close("cache v", &gv[offset..offset + kv_elems], &wv);
        for (name, got) in [("k", &gk), ("v", &gv)] {
            let touched = got[..offset]
                .iter()
                .chain(&got[offset + kv_elems..kv_elems * 3])
                .filter(|p| **p != UNWRITTEN)
                .count();
            assert_eq!(
                touched, 0,
                "cache {name}: wrote {touched} elements outside its slot"
            );
        }
    });
}
