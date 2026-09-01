//! Numeric tests for the three flash-attention kernels.
//!
//! These had a name check in `promoted_kernels.rs` and error-path coverage in
//! `nn_wiring.rs`, and nothing that ran them for a number.
//!
//! The reference is a direct transcription of the kernels' own masking rule,
//! computed in f64. From the sources:
//!
//! * `q_abs = q_pos_offset + t_q`, `k_abs = kv_pos_offset + t_k`.
//! * Sliding window keeps `max(0, q_abs - window + 1) <= k_abs <= q_abs`.
//! * Global keeps `k_abs <= q_abs` and ignores `window` entirely.
//! * A row with nothing unmasked divides by `l_i == 0`, and both kernels
//!   special-case that to `inv_l = 0` — so it is zeros, not NaN.
//!
//! Softmax is computed in the reference as a plain max-subtracted pass; the
//! kernels use the FlashAttention-2 online rescaling, so agreement between them
//! is evidence the streaming update is right, not a restatement of it.

mod common;

use common::{buf, empty, random_f32, seeded, with_gpu};
use std::sync::Arc;
use tessl::nn::{self, AttnDims, AttnHeadDim};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const UNWRITTEN: f32 = -6.5e28;

#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    tq: usize,
    tkv: usize,
    h: usize,
    hkv: usize,
    d: usize,
}

/// f64 attention reference. `window == None` is the global (causal) rule.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: Shape,
    window: Option<usize>,
    q_off: usize,
    kv_off: usize,
    scale: f32,
) -> Vec<f32> {
    let Shape {
        b,
        tq,
        tkv,
        h,
        hkv,
        d,
    } = s;
    let group = (h / hkv).max(1);
    let mut out = vec![0.0f32; b * tq * h * d];
    for bi in 0..b {
        for t_q in 0..tq {
            let q_abs = (q_off + t_q) as i64;
            for hi in 0..h {
                let hk = hi / group;
                let q_base = ((bi * tq + t_q) * h + hi) * d;

                let mut scores = vec![f64::NEG_INFINITY; tkv];
                for (t_k, sc) in scores.iter_mut().enumerate() {
                    let k_abs = (kv_off + t_k) as i64;
                    let keep = match window {
                        Some(w) => k_abs >= (q_abs - w as i64 + 1).max(0) && k_abs <= q_abs,
                        None => k_abs <= q_abs,
                    };
                    if !keep {
                        continue;
                    }
                    let k_base = ((bi * tkv + t_k) * hkv + hk) * d;
                    let dot: f64 = (0..d)
                        .map(|x| q[q_base + x] as f64 * k[k_base + x] as f64)
                        .sum();
                    *sc = dot * scale as f64;
                }

                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if !m.is_finite() {
                    // Every position masked. Both kernels emit zeros here
                    // rather than NaN, and a caller masking a whole row is an
                    // ordinary decode state, not a pathological one.
                    continue;
                }
                let mut l = 0.0f64;
                let mut acc = vec![0.0f64; d];
                for (t_k, sc) in scores.iter().enumerate() {
                    if !sc.is_finite() {
                        continue;
                    }
                    let p = (sc - m).exp();
                    l += p;
                    let v_base = ((bi * tkv + t_k) * hkv + hk) * d;
                    for x in 0..d {
                        acc[x] += p * v[v_base + x] as f64;
                    }
                }
                let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                for x in 0..d {
                    out[q_base + x] = (acc[x] * inv) as f32;
                }
            }
        }
    }
    out
}

fn u32_buf(rt: &Arc<GpuRuntime>, v: u32) -> GpuBuffer {
    let b = rt.alloc_buffer(4).unwrap();
    b.write_u32(&[v]);
    b
}

fn check(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        assert!(*g != UNWRITTEN, "{label}[{i}]: never written");
        // Head sums run to D terms in f32 against an f64 reference.
        let tol = 2e-4 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{label}[{i}]: got {g} want {w}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_swa(
    rt: &Arc<GpuRuntime>,
    head: AttnHeadDim,
    s: Shape,
    window: usize,
    q_off: u32,
    kv_off: u32,
    scale: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let q = random_f32(s.b * s.tq * s.h * s.d, seed);
    let k = random_f32(s.b * s.tkv * s.hkv * s.d, seed + 1);
    let v = random_f32(s.b * s.tkv * s.hkv * s.d, seed + 2);
    let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
    let ob = seeded(rt, s.b * s.tq * s.h * s.d, UNWRITTEN);
    let tkv = u32_buf(rt, s.tkv as u32);
    let qo = u32_buf(rt, q_off);
    let ko = u32_buf(rt, kv_off);

    nn::flash_attn_swa(
        rt,
        head,
        &qb,
        &kb,
        &vb,
        &ob,
        &tkv,
        &qo,
        &ko,
        AttnDims {
            batch: s.b as u32,
            tq: s.tq as u32,
            heads: s.h as u32,
            heads_kv: s.hkv as u32,
            window: window as u32,
            scale,
        },
    )
    .unwrap();
    rt.synchronize().unwrap();

    let want = reference(
        &q,
        &k,
        &v,
        s,
        Some(window),
        q_off as usize,
        kv_off as usize,
        scale,
    );
    (ob.read_f32()[..want.len()].to_vec(), want)
}

/// Prefill: `Tq == Tkv`, both offsets zero, window wide enough to be inert —
/// so this is plain causal attention and isolates the tiling from the masking.
#[test]
fn swa_prefill_matches_an_f64_reference_at_both_head_dims() {
    with_gpu(|rt| {
        for (head, d) in [(AttnHeadDim::D128, 128usize), (AttnHeadDim::D256, 256)] {
            // Tq spans several BR=8 query blocks and Tkv several BC=8 key
            // blocks, with a ragged tail in both so the `n_k = min(BC, ...)`
            // and `t_q < Tq` guards are exercised.
            let s = Shape {
                b: 2,
                tq: 19,
                tkv: 19,
                h: 4,
                hkv: 2,
                d,
            };
            let (got, want) = run_swa(rt, head, s, 4096, 0, 0, 0.125, 0xA1 + d as u64);
            check(&format!("swa prefill d={d}"), &got, &want);
        }
    });
}

/// The window must actually bound the key range. A wide window and a narrow one
/// over identical inputs have to disagree, or the mask is not being applied.
#[test]
fn swa_window_restricts_the_key_range() {
    with_gpu(|rt| {
        let s = Shape {
            b: 1,
            tq: 17,
            tkv: 17,
            h: 2,
            hkv: 1,
            d: 128,
        };
        let (wide, wide_ref) = run_swa(rt, AttnHeadDim::D128, s, 4096, 0, 0, 0.125, 0xB2);
        check("swa wide", &wide, &wide_ref);
        for w in [1usize, 2, 5] {
            let (got, want) = run_swa(rt, AttnHeadDim::D128, s, w, 0, 0, 0.125, 0xB2);
            check(&format!("swa window={w}"), &got, &want);
            // Same seed, so the only difference is the mask.
            assert!(
                got.iter().zip(&wide).any(|(a, b)| (a - b).abs() > 1e-6),
                "window={w} produced the same output as an unbounded window; \
                 the mask is not doing anything"
            );
        }
        // window = 1 keeps only k_abs == q_abs, so each row is exactly its own
        // V vector — softmax over a single element is 1.
        let s1 = Shape {
            b: 1,
            tq: 5,
            tkv: 5,
            h: 1,
            hkv: 1,
            d: 128,
        };
        let (got, _) = run_swa(rt, AttnHeadDim::D128, s1, 1, 0, 0, 0.125, 0xB3);
        let v = random_f32(s1.b * s1.tkv * s1.hkv * s1.d, 0xB3 + 2);
        check("swa window=1 is the diagonal", &got, &v[..got.len()]);
    });
}

/// Decode: one query position against a filled cache, with the offsets that
/// make `q_abs` and `k_abs` disagree. This is the shape the buffers exist for —
/// `tkv` and both offsets are device `u32`s precisely so decode can advance
/// them without re-encoding.
#[test]
fn swa_decode_positions_come_from_the_device_offsets() {
    with_gpu(|rt| {
        let s = Shape {
            b: 1,
            tq: 1,
            tkv: 40,
            h: 4,
            hkv: 2,
            d: 128,
        };
        // The cache holds positions 0..40 and the new token is at 40.
        let (got, want) = run_swa(rt, AttnHeadDim::D128, s, 16, 40, 0, 0.125, 0xC3);
        check("swa decode", &got, &want);

        // A query far past everything in the cache leaves the window empty.
        // `l_i == 0` then, and the kernel must emit zeros rather than NaN.
        let (got, want) = run_swa(rt, AttnHeadDim::D128, s, 4, 10_000, 0, 0.125, 0xC4);
        assert!(
            got.iter().all(|v| *v == 0.0),
            "a fully masked decode row must be zeros, not {:?}",
            &got[..4]
        );
        check("swa fully masked", &got, &want);
    });
}

/// Grouped-query attention: `H > Hkv` means head `h` must read KV head
/// `h / (H / Hkv)`. Getting that division wrong still produces plausible
/// numbers, so the reference indexes it independently.
#[test]
fn swa_maps_query_heads_onto_their_kv_group() {
    with_gpu(|rt| {
        for (h, hkv) in [(8usize, 1usize), (8, 2), (8, 4), (8, 8)] {
            let s = Shape {
                b: 1,
                tq: 9,
                tkv: 9,
                h,
                hkv,
                d: 128,
            };
            let (got, want) = run_swa(rt, AttnHeadDim::D128, s, 4096, 0, 0, 0.125, 0xD4);
            check(&format!("gqa H={h} Hkv={hkv}"), &got, &want);
        }
    });
}

/// The global kernel is causal with no lower bound, and `window` is ignored.
#[test]
fn global_h512_is_causal_and_ignores_the_window() {
    with_gpu(|rt| {
        let s = Shape {
            b: 1,
            tq: 9,
            tkv: 9,
            h: 2,
            hkv: 1,
            d: 512,
        };
        let q = random_f32(s.b * s.tq * s.h * s.d, 0xE5);
        let k = random_f32(s.b * s.tkv * s.hkv * s.d, 0xE6);
        let v = random_f32(s.b * s.tkv * s.hkv * s.d, 0xE7);
        let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
        let tkv = u32_buf(rt, s.tkv as u32);
        let zero = u32_buf(rt, 0);
        let want = reference(&q, &k, &v, s, None, 0, 0, 0.125);

        // Two different windows: the global kernel must not react to either.
        let mut seen = Vec::new();
        for window in [1u32, 4096] {
            let ob = seeded(rt, s.b * s.tq * s.h * s.d, UNWRITTEN);
            nn::flash_attn_global_h512(
                rt,
                &qb,
                &kb,
                &vb,
                &ob,
                &tkv,
                &zero,
                &zero,
                AttnDims {
                    batch: s.b as u32,
                    tq: s.tq as u32,
                    heads: s.h as u32,
                    heads_kv: s.hkv as u32,
                    window,
                    scale: 0.125,
                },
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let got = ob.read_f32()[..want.len()].to_vec();
            check(&format!("global window={window}"), &got, &want);
            seen.push(got);
        }
        assert_eq!(
            seen[0], seen[1],
            "the global kernel changed with `window`, which it documents as ignored"
        );
    });
}

/// `out_bf16` writes O as bfloat. Same math, narrower store.
#[test]
fn global_h512_bf16_output_matches_the_f32_one_within_bf16_resolution() {
    with_gpu(|rt| {
        let s = Shape {
            b: 1,
            tq: 5,
            tkv: 5,
            h: 2,
            hkv: 1,
            d: 512,
        };
        let n = s.b * s.tq * s.h * s.d;
        let q = random_f32(n, 0xF6);
        let k = random_f32(s.b * s.tkv * s.hkv * s.d, 0xF7);
        let v = random_f32(s.b * s.tkv * s.hkv * s.d, 0xF8);
        let (qb, kb, vb) = (buf(rt, &q), buf(rt, &k), buf(rt, &v));
        let tkv = u32_buf(rt, s.tkv as u32);
        let zero = u32_buf(rt, 0);
        let dims = AttnDims {
            batch: s.b as u32,
            tq: s.tq as u32,
            heads: s.h as u32,
            heads_kv: s.hkv as u32,
            window: 4096,
            scale: 0.125,
        };

        let f32_out = empty(rt, n);
        nn::flash_attn_global_h512(rt, &qb, &kb, &vb, &f32_out, &tkv, &zero, &zero, dims, false)
            .unwrap();
        let bf = rt.alloc_buffer(n * 2).unwrap();
        bf.zero();
        nn::flash_attn_global_h512(rt, &qb, &kb, &vb, &bf, &tkv, &zero, &zero, dims, true).unwrap();
        rt.synchronize().unwrap();

        let want = f32_out.read_f32();
        let got: Vec<f32> = bf.read_u32()[..n / 2]
            .iter()
            .flat_map(|p| {
                [
                    tessl::tensor::bf16_bits_to_f32((*p & 0xffff) as u16),
                    tessl::tensor::bf16_bits_to_f32((*p >> 16) as u16),
                ]
            })
            .collect();
        for (i, (g, w)) in got.iter().zip(&want[..n]).enumerate() {
            let tol = 1e-2 * w.abs().max(1e-3);
            assert!((g - w).abs() <= tol, "global bf16 [{i}]: {g} vs {w}");
        }
    });
}
