//! Interleaved A/B of the GDN core at the Qwen3.5-2B shape: the chunked prefill
//! path (`gdn_chunk_forward`), the token-by-token `gdn_recurrent`, and
//! the four kernels of `gdn_recurrent_simd`'s family, with `gdn_recurrent`
//! timed a second time as the noise floor.
//!
//! This is a measurement, not a gate: it is `#[ignore]`d, and run under the
//! machine lock by `target-gp/` scripts (see `bench/results/
//! qwen35_gdn_simd_ab_m5pro.txt` for the protocol and the numbers).
//!
//! Protocol. Same inputs to every arm (batch 1, 16 key heads, 16 value heads,
//! Dk = Dv = 128, a non-zero per-batch start state, `state_out` written), the
//! gates, norms and q scale folded into each kernel's loads (equal work: the
//! chunked arm's prep dispatch is inside its time). A sample is `N` launches
//! between two `synchronize`s, divided by `N`; every arm gets `WARMUP`
//! untimed launches before its first sample. Rounds rotate the arm order (round
//! r starts at arm r mod the number of arms), so no arm is always first or always after the same
//! neighbour. Per arm: min and median over rounds. Noise floor: the ratio of
//! the two `gdn_recurrent` arms, per round, then its median and range, which is
//! the resolution at which any other ratio below can be read.
//!
//! Correctness runs first: the simd kernel's output and state are compared
//! with `gdn_recurrent`'s on the same inputs, and the run aborts if they are
//! not within 1e-4 of the output scale. A fast wrong kernel is not a result.

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::{buf, empty, random_f32, with_gpu};
use tessl::qwen35::{
    gdn_chunk_forward, gdn_recurrent, gdn_recurrent_simd_with, Cols, GdnDims, GdnGateLogits, GdnParams, GdnQkv,
    GdnSimdScalars, GdnSimdVariant, GdnWorkspace, StateIn,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

const DK: usize = 128;
const HK: usize = 16;
const HV: usize = 16;
const DV: usize = 128;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().map_or(default, |v| {
        v.parse().unwrap_or_else(|_| panic!("{key}={v} is not a number"))
    })
}

struct Ops {
    qkv: GpuBuffer,
    ld_qkv: u32,
    k_off: u32,
    v_off: u32,
    gates: GpuBuffer,
    ld_gates: u32,
    b_off: u32,
    a_log: GpuBuffer,
    dt_bias: GpuBuffer,
    state0: GpuBuffer,
}

fn ops(rt: &Arc<GpuRuntime>, t: usize, seed: u64) -> Ops {
    let key_w = HK * DK;
    let val_w = HV * DV;
    let ld_qkv = 2 * key_w + val_w;
    let q = random_f32(t * key_w, seed);
    let k = random_f32(t * key_w, seed + 1);
    let v = random_f32(t * val_w, seed + 2);
    let mut qkv = vec![0.0f32; t * ld_qkv];
    for r in 0..t {
        qkv[r * ld_qkv..r * ld_qkv + key_w].copy_from_slice(&q[r * key_w..(r + 1) * key_w]);
        qkv[r * ld_qkv + key_w..r * ld_qkv + 2 * key_w].copy_from_slice(&k[r * key_w..(r + 1) * key_w]);
        qkv[r * ld_qkv + 2 * key_w..(r + 1) * ld_qkv].copy_from_slice(&v[r * val_w..(r + 1) * val_w]);
    }
    let ld_gates = 2 * HV;
    let a = random_f32(t * HV, seed + 3);
    let b = random_f32(t * HV, seed + 4);
    let mut gates = vec![0.0f32; t * ld_gates];
    for r in 0..t {
        for h in 0..HV {
            gates[r * ld_gates + h] = a[r * HV + h] * 2.0;
            gates[r * ld_gates + HV + h] = b[r * HV + h] * 2.0;
        }
    }
    let a_log: Vec<f32> = random_f32(HV, seed + 5).into_iter().map(|x| -0.5 + 1.5 * x).collect();
    let state0: Vec<f32> = random_f32(HV * DK * DV, seed + 7)
        .into_iter()
        .map(|x| x * 0.1)
        .collect();
    Ops {
        qkv: buf(rt, &qkv),
        ld_qkv: ld_qkv as u32,
        k_off: key_w as u32,
        v_off: (2 * key_w) as u32,
        gates: buf(rt, &gates),
        ld_gates: ld_gates as u32,
        b_off: HV as u32,
        a_log: buf(rt, &a_log),
        dt_bias: buf(rt, &random_f32(HV, seed + 6)),
        state0: buf(rt, &state0),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Chunked,
    Recurrent,
    Simd(GdnSimdVariant),
    RecurrentAgain,
}

/// Chunked, recurrent, every kernel of the simd family, recurrent again.
fn arms() -> Vec<Arm> {
    let mut v = vec![Arm::Chunked, Arm::Recurrent];
    v.extend(GdnSimdVariant::ALL.map(Arm::Simd));
    v.push(Arm::RecurrentAgain);
    v
}

const CHUNKED: usize = 0;
const RECURRENT: usize = 1;
const SIMD: usize = 2;

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[test]
#[ignore = "timing run; execute under the machine lock (target-gp scripts)"]
fn gdn_simd_recurrent_ab_at_the_2b_shape() {
    let rounds = env_usize("GDN_AB_ROUNDS", 22);
    let warmup = env_usize("GDN_AB_WARMUP", 3);
    let lengths: Vec<usize> = std::env::var("GDN_AB_T")
        .unwrap_or_else(|_| "1,200,1024,8192".into())
        .split(',')
        .map(|x| x.parse().expect("GDN_AB_T is a comma list of lengths"))
        .collect();
    assert!(rounds >= 9, "fewer than 9 rounds is not a result");
    with_gpu(|rt| {
        let max_t = *lengths.iter().max().unwrap();
        let big = GdnDims {
            batch: 1,
            seq: max_t as u32,
            k_heads: HK as u32,
            v_heads: HV as u32,
            v_dim: DV as u32,
        };
        let arms = arms();
        let recurrent_again = arms.len() - 1;
        let ws = GdnWorkspace::new(rt, &big).unwrap();
        let scalars = GdnSimdScalars::new(rt, &big).unwrap();
        for &t in &lengths {
            let o = ops(rt, t, 3 + t as u64);
            let d = GdnDims { seq: t as u32, ..big };
            let width = (HV * DV) as u32;
            let state_n = HV * DK * DV;
            let outs: Vec<GpuBuffer> = arms.iter().map(|_| empty(rt, t * width as usize)).collect();
            let sts: Vec<GpuBuffer> = arms.iter().map(|_| empty(rt, state_n)).collect();
            let launch = |arm: Arm| {
                let i = arms.iter().position(|&a| a == arm).unwrap();
                let q = GdnQkv {
                    buf: &o.qkv,
                    ld: o.ld_qkv,
                    q_off: 0,
                    k_off: o.k_off,
                    v_off: o.v_off,
                };
                let g = GdnGateLogits {
                    buf: &o.gates,
                    ld: o.ld_gates,
                    a_off: 0,
                    b_off: o.b_off,
                };
                let p = GdnParams {
                    a_log: &o.a_log,
                    dt_bias: &o.dt_bias,
                };
                let cols = Cols::dense(&outs[i], width);
                let st = StateIn::PerBatch(&o.state0);
                let r = match arm {
                    Arm::Chunked => gdn_chunk_forward(rt, &d, &q, &g, &p, st, &ws, cols, Some(&sts[i])),
                    Arm::Recurrent | Arm::RecurrentAgain => gdn_recurrent(rt, &d, &q, &g, &p, st, cols, Some(&sts[i])),
                    Arm::Simd(v) => {
                        gdn_recurrent_simd_with(rt, &d, &q, &g, &p, st, cols, Some(&sts[i]), v, Some(&scalars))
                    }
                };
                r.expect("launch");
            };
            // Correctness before timing.
            for &arm in &arms {
                launch(arm);
            }
            rt.synchronize().unwrap();
            let read = |i: usize, b: &[GpuBuffer], n: usize| b[i].read_f32()[..n].to_vec();
            let n_out = t * width as usize;
            let (y_r, s_r) = (read(RECURRENT, &outs, n_out), read(RECURRENT, &sts, state_n));
            let scale = y_r.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-3);
            let sscale = s_r.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-3);
            let dev = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            let dc = dev(&y_r, &read(CHUNKED, &outs, n_out));
            println!(
                "GDN_AB_CHECK t={t}: chunked vs recurrent max|dy|/scale {:.3e}",
                dc / scale
            );
            for i in SIMD..recurrent_again {
                let (y_s, s_s) = (read(i, &outs, n_out), read(i, &sts, state_n));
                let (dy, ds) = (dev(&y_r, &y_s), dev(&s_r, &s_s));
                println!(
                    "GDN_AB_CHECK t={t} {:?}: vs recurrent max|dy|/scale {:.3e} max|dstate|/scale {:.3e}",
                    arms[i],
                    dy / scale,
                    ds / sscale
                );
                assert!(
                    dy <= 1e-4 * scale && ds <= 1e-4 * sscale,
                    "{:?} disagrees with gdn_recurrent at t={t}",
                    arms[i]
                );
                assert!(y_s.iter().any(|&x| x != 0.0), "{:?}: output is all zeros", arms[i]);
            }

            // Launches per sample: enough that a sample is well above the host
            // submit-and-wait floor (about 0.25 ms) at every length.
            let n = (env_usize("GDN_AB_TARGET_MS", 20) as f64 / 1.0).max(1.0);
            let pilot = {
                let t0 = Instant::now();
                launch(Arm::Chunked);
                rt.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e3
            };
            let per_sample = ((n / pilot.max(0.05)).ceil() as usize).clamp(1, 200);
            for &arm in &arms {
                for _ in 0..warmup {
                    launch(arm);
                }
            }
            rt.synchronize().unwrap();
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            for round in 0..rounds {
                for step in 0..arms.len() {
                    let ai = (round + step) % arms.len();
                    let arm = arms[ai];
                    let t0 = Instant::now();
                    for _ in 0..per_sample {
                        launch(arm);
                    }
                    rt.synchronize().unwrap();
                    samples[ai].push(t0.elapsed().as_secs_f64() * 1e3 / per_sample as f64);
                }
            }
            let ratio = |num: usize, den: usize| -> Vec<f64> {
                samples[num].iter().zip(&samples[den]).map(|(a, b)| a / b).collect()
            };
            for (i, arm) in arms.iter().enumerate() {
                println!(
                    "GDN_AB t={t} arm={arm:?} min_ms={:.4} median_ms={:.4} max_ms={:.4} rounds={rounds} launches_per_sample={per_sample}",
                    min(&samples[i]),
                    median(&samples[i]),
                    max(&samples[i])
                );
            }
            let mut pairs = vec![
                (
                    "noise recurrent/recurrent_again".to_string(),
                    RECURRENT,
                    recurrent_again,
                ),
                ("recurrent/chunked".to_string(), RECURRENT, CHUNKED),
            ];
            for i in SIMD..recurrent_again {
                pairs.push((format!("{:?}/recurrent", arms[i]), i, RECURRENT));
                pairs.push((format!("{:?}/chunked", arms[i]), i, CHUNKED));
            }
            for (name, num, den) in pairs {
                let r = ratio(num, den);
                println!(
                    "GDN_AB_RATIO t={t} {name}: median {:.3} min {:.3} max {:.3}",
                    median(&r),
                    min(&r),
                    max(&r)
                );
            }
        }
    });
}
