//! `nn::gemm_q8_tensorops` (kernels/gemm_q8_tensorops.metal).
//!
//! Parity is against an f64 product of the loaded f32 activations and int8
//! codes, and, for M = 1, against `nn::gemv_q8` on the same bytes. Activations
//! are multiples of 1/64, so `x * q` is exact in f32 and the bound has no
//! operand-rounding term. Two runs of one shape are bit-identical. A group
//! the descriptor does not instantiate, a ragged K, an alias, and an i32
//! extent are refused with no dispatch and no GEMV fallback.

mod common;

use common::{buf, seeded, with_gpu, SplitMix};
use std::sync::Arc;
use tessl::nn::{self, GEMM_Q8_TENSOROPS_GROUP_SIZES};
use tessl::runtime::{kernel_trace_enabled, traced_kernel_count};
use tessl::GpuRuntime;

const SENTINEL: f32 = -7.7e28;

fn gamma(terms: usize) -> f64 {
    let tu = terms as f64 * 2f64.powi(-24);
    tu / (1.0 - tu)
}

#[allow(clippy::too_many_arguments)]
fn reference(
    x: &[f32],
    q: &[i8],
    scales: &[f32],
    zeros: &[f32],
    m: usize,
    n: usize,
    k: usize,
    group: usize,
) -> (Vec<f64>, Vec<f64>) {
    let groups = k / group;
    let mut want = vec![0.0f64; m * n];
    let mut mag = vec![0.0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let (mut acc, mut s_mag) = (0.0f64, 0.0f64);
            for g in 0..groups {
                let s = scales[ni * groups + g] as f64;
                let z = zeros[ni * groups + g] as f64;
                let (mut qx, mut aqx, mut xsum, mut ax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for j in 0..group {
                    let xv = x[mi * k + g * group + j] as f64;
                    let qv = q[ni * k + g * group + j] as f64;
                    qx += qv * xv;
                    aqx += qv.abs() * xv.abs();
                    xsum += xv;
                    ax += xv.abs();
                }
                acc += s * qx + (-s * z) * xsum;
                s_mag += s.abs() * aqx + (s * z).abs() * ax;
            }
            want[mi * n + ni] = acc;
            mag[mi * n + ni] = s_mag;
        }
    }
    (want, mag)
}

fn upload_i8(rt: &Arc<GpuRuntime>, q: &[i8]) -> tessl::GpuBuffer {
    let b = rt.alloc_buffer(q.len().max(1)).unwrap();
    let bytes: Vec<u8> = if q.is_empty() {
        vec![0]
    } else {
        q.iter().map(|v| *v as u8).collect()
    };
    b.write_bytes(&bytes);
    b
}

struct Case {
    to_ref: f64,
    gemv_ref: f64,
    to_gemv: f64,
}

fn run_case(rt: &Arc<GpuRuntime>, m: usize, n: usize, k: usize, group: usize) -> Case {
    let groups = k / group;
    let mut rng = SplitMix::new(0x0C08 ^ (m * 1_000_003 + n * 4099 + k * 17 + group) as u64);
    let x: Vec<f32> = (0..m * k)
        .map(|_| {
            let v = rng.unit();
            (v * 64.0).round() / 64.0
        })
        .collect();
    let q: Vec<i8> = (0..n * k).map(|_| (rng.range(0, 255) as i32 - 128) as i8).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| 0.01 + 0.04 * ((rng.unit() + 1.0) * 0.5))
        .collect();
    let zeros: Vec<f32> = (0..n * groups)
        .map(|_| -1.5 + 3.0 * ((rng.unit() + 1.0) * 0.5))
        .collect();
    let (want, mag) = reference(&x, &q, &scales, &zeros, m, n, k, group);

    let packed = upload_i8(rt, &q);
    let sb = buf(rt, &scales);
    let zb = buf(rt, &zeros);
    let xb = buf(rt, &x);
    let yb = seeded(rt, m * n, SENTINEL);
    nn::gemm_q8_tensorops(
        rt,
        &packed,
        &sb,
        &zb,
        &xb,
        &yb,
        n as u32,
        k as u32,
        group as u32,
        m as u32,
    )
    .unwrap();
    rt.synchronize().unwrap();
    let to: Vec<f32> = yb.read_f32()[..m * n].to_vec();
    assert!(
        to.iter().all(|v| *v != SENTINEL && v.is_finite()),
        "Q8 TensorOps left outputs unwritten or non-finite for M={m} N={n} K={k} G={group}"
    );

    let g = gamma(k + 2 * groups + group);
    let mut out = Case {
        to_ref: 0.0,
        gemv_ref: 0.0,
        to_gemv: 0.0,
    };
    let mut gemv: Option<Vec<f32>> = None;
    if m == 1 {
        let yv = seeded(rt, n, SENTINEL);
        nn::gemv_q8(rt, &packed, &sb, &zb, &xb, &yv, n as u32, k as u32, group as u32).unwrap();
        rt.synchronize().unwrap();
        let gv = yv.read_f32()[..n].to_vec();
        assert!(gv.iter().all(|v| *v != SENTINEL && v.is_finite()), "gemv unwritten");
        gemv = Some(gv);
    }
    for i in 0..m * n {
        let e = (to[i] as f64 - want[i]).abs();
        let bound = g * mag[i];
        let frac = if bound == 0.0 {
            if e == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            e / bound
        };
        out.to_ref = out.to_ref.max(frac);
        if let Some(gv) = gemv.as_ref() {
            let eg = (gv[i] as f64 - want[i]).abs();
            let ep = (to[i] as f64 - gv[i] as f64).abs();
            let gfrac = if bound == 0.0 {
                if eg == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                eg / bound
            };
            let pbound = 2.0 * bound;
            let pfrac = if pbound == 0.0 {
                if ep == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                ep / pbound
            };
            out.gemv_ref = out.gemv_ref.max(gfrac);
            out.to_gemv = out.to_gemv.max(pfrac);
        }
    }
    out
}

#[test]
fn q8_tensorops_matches_the_f64_affine_and_the_m1_gemv() {
    with_gpu(|rt| {
        let mut failures = Vec::new();
        for &group in &GEMM_Q8_TENSOROPS_GROUP_SIZES {
            let group = group as usize;
            for m in [1usize, 8, 33] {
                for n in [32usize, 33] {
                    let mut ks = vec![group * 2];
                    if 256 != group * 2 {
                        ks.push(256);
                    }
                    for k in ks {
                        let c = run_case(rt, m, n, k, group);
                        eprintln!(
                            "M={m} N={n} K={k} G={group}: TO/f64 {:.3} GEMV/f64 {:.3} TO/GEMV {:.3}",
                            c.to_ref, c.gemv_ref, c.to_gemv
                        );
                        if c.to_ref > 1.0 || c.gemv_ref > 1.0 || c.to_gemv > 1.0 {
                            failures.push(format!(
                                "M={m} N={n} K={k} G={group}: {:.3} / {:.3} / {:.3}",
                                c.to_ref, c.gemv_ref, c.to_gemv
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    });
}

#[test]
fn two_runs_are_bit_identical() {
    with_gpu(|rt| {
        let (m, n, k, group) = (8usize, 40, 128, 64);
        let groups = k / group;
        let mut rng = SplitMix::new(0xB17);
        let x: Vec<f32> = (0..m * k).map(|_| (rng.unit() * 64.0).round() / 64.0).collect();
        let q: Vec<i8> = (0..n * k).map(|_| (rng.range(0, 255) as i32 - 128) as i8).collect();
        let scales: Vec<f32> = (0..n * groups).map(|_| 0.02 + 0.01 * rng.unit()).collect();
        let zeros: Vec<f32> = (0..n * groups).map(|_| rng.unit()).collect();
        let packed = upload_i8(rt, &q);
        let sb = buf(rt, &scales);
        let zb = buf(rt, &zeros);
        let xb = buf(rt, &x);
        let mut bits = Vec::new();
        for _ in 0..2 {
            let yb = seeded(rt, m * n, SENTINEL);
            nn::gemm_q8_tensorops(
                rt,
                &packed,
                &sb,
                &zb,
                &xb,
                &yb,
                n as u32,
                k as u32,
                group as u32,
                m as u32,
            )
            .unwrap();
            rt.synchronize().unwrap();
            bits.push(yb.read_f32()[..m * n].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert_eq!(bits[0], bits[1]);
        assert!(bits[0].iter().all(|b| *b != SENTINEL.to_bits()));
    });
}

#[test]
fn the_q8_group_selects_its_matmul2d_kernel() {
    let Some(_) = common::in_traced_child("the_q8_group_selects_its_matmul2d_kernel", || {
        with_gpu(|rt| {
            assert!(kernel_trace_enabled());
            let (m, n, k, group) = (4u32, 32, 128, 64);
            let q = vec![1i8; (n * k) as usize];
            let scales = vec![0.05f32; (n * (k / group)) as usize];
            let zeros = vec![0.0f32; scales.len()];
            let x = vec![0.25f32; (m * k) as usize];
            let packed = upload_i8(rt, &q);
            let sb = buf(rt, &scales);
            let zb = buf(rt, &zeros);
            let xb = buf(rt, &x);
            let yb = seeded(rt, (m * n) as usize, SENTINEL);
            let before = traced_kernel_count("gemm_q8_tensorops_g64");
            let gemv_before = traced_kernel_count("gemv_q8");
            nn::gemm_q8_tensorops(rt, &packed, &sb, &zb, &xb, &yb, n, k, group, m).unwrap();
            rt.synchronize().unwrap();
            assert_eq!(traced_kernel_count("gemm_q8_tensorops_g64") - before, 1);
            assert_eq!(traced_kernel_count("gemv_q8"), gemv_before);
            assert_eq!(traced_kernel_count("gemm_q8_tensorops_g32"), 0);
            assert_eq!(traced_kernel_count("gemm_q8_tensorops_g128"), 0);
        });
    }) else {
        return;
    };
}

#[test]
fn bad_group_ragged_k_alias_and_extent_are_refused_before_dispatch() {
    with_gpu(|rt| {
        let scratch = |n: usize| rt.alloc_buffer(n.max(4)).unwrap();
        let packed = scratch(16);
        let scales = scratch(16);
        let zeros = scratch(16);
        let x = scratch(16);
        let y = scratch(16);

        rt.take_dispatch_count();
        let err = nn::gemm_q8_tensorops(rt, &packed, &scales, &zeros, &x, &y, 4, 64, 16, 2).unwrap_err();
        assert!(err.contains("16"), "{err}");
        assert!(err.contains("group_size"), "{err}");
        assert!(!err.contains("host"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::gemm_q8_tensorops(rt, &packed, &scales, &zeros, &x, &y, 4, 96, 64, 2).unwrap_err();
        assert!(err.contains("96"), "{err}");
        assert!(err.contains("multiple"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::gemm_q8_tensorops(rt, &packed, &scales, &zeros, &x, &y, 4, 0, 64, 2).unwrap_err();
        assert!(err.contains("non-zero"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        // Real sizes so the refusal is the alias, not a short buffer.
        let (m, n, k, group) = (2u32, 4, 64, 64);
        let q = vec![1i8; (n * k) as usize];
        let scales_v = vec![0.1f32; (n * (k / group)) as usize];
        let zeros_v = vec![0.0f32; scales_v.len()];
        let xv = vec![0.5f32; (m * k) as usize];
        let packed = upload_i8(rt, &q);
        let sb = buf(rt, &scales_v);
        let zb = buf(rt, &zeros_v);
        let xb = buf(rt, &xv);
        rt.take_dispatch_count();
        let err = nn::gemm_q8_tensorops(rt, &packed, &sb, &zb, &xb, &xb, n, k, group, m).unwrap_err();
        assert!(err.contains("overlap"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let y0 = seeded(rt, 1, SENTINEL);
        rt.take_dispatch_count();
        nn::gemm_q8_tensorops(rt, &packed, &sb, &zb, &xb, &y0, n, k, group, 0).unwrap();
        assert_eq!(rt.take_dispatch_count(), 0);
        assert_eq!(y0.read_f32()[0].to_bits(), SENTINEL.to_bits());

        let wide = 1000u32;
        let k_bad = nn::gemm_q8_tensorops_max_k(wide, wide, group) + group;
        rt.take_dispatch_count();
        let err = nn::gemm_q8_tensorops(rt, &packed, &sb, &zb, &xb, &y0, wide, k_bad, group, wide).unwrap_err();
        assert!(err.contains("signed 32-bit"), "{err}");
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}
