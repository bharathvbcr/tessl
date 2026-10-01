//! Generic device AdamW against a non-contracted f32 reference.
//!
//! The bound is the one `tests/qwen35_adamw.rs` uses for the model step:
//! `2e-6` absolute. Values are large enough that an f32 contraction changes
//! bits by more than that, so a kernel compiled with contraction on fails
//! this test.

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::{buf, empty, random_f32, with_gpu};
use tessl::qwen35_adamw::{adamw_step, AdamWHyper};
use tessl::{nn, GpuRuntime, Tensor};

const ADAMW_ABS_TOL: f32 = 2e-6;

fn adamw_f32_ref(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    hyper: &AdamWHyper,
    step: u64,
    weight_decay: f32,
) {
    let t = step as f64;
    let bc1 = 1.0 - hyper.beta1.powf(t);
    let bc2 = 1.0 - hyper.beta2.powf(t);
    let step_size = (hyper.lr / bc1) as f32;
    let bc2_sqrt = bc2.powf(0.5) as f32;
    let decay = (1.0 - hyper.lr * f64::from(weight_decay)) as f32;
    let lerp_w = (1.0 - hyper.beta1) as f32;
    let beta2 = hyper.beta2 as f32;
    let one_minus_beta2 = (1.0 - hyper.beta2) as f32;
    let eps = hyper.eps as f32;
    let scale = hyper.grad_scale as f32;
    for i in 0..p.len() {
        let mut w = p[i] * decay;
        let gi = g[i] * scale;
        let mut mi = m[i];
        let diff = gi - mi;
        mi = if lerp_w < 0.5 {
            mi + lerp_w * diff
        } else {
            gi - diff * (1.0 - lerp_w)
        };
        let vi = v[i] * beta2 + (one_minus_beta2 * gi) * gi;
        let denom = vi.sqrt() / bc2_sqrt + eps;
        w += (-step_size) * (mi / denom);
        p[i] = w;
        m[i] = mi;
        v[i] = vi;
    }
}

struct StepTensors {
    p: Tensor,
    g: Tensor,
    m: Tensor,
    v: Tensor,
    hp: Vec<f32>,
    hg: Vec<f32>,
    hm: Vec<f32>,
    hv: Vec<f32>,
}

fn tensors(rt: &Arc<GpuRuntime>, n: usize, seed: u64) -> StepTensors {
    let hp = random_f32(n, seed).into_iter().map(|x| x * 1.0e4).collect::<Vec<_>>();
    let hg = random_f32(n, seed ^ 1)
        .into_iter()
        .map(|x| x * 1.0e4)
        .collect::<Vec<_>>();
    let hm = random_f32(n, seed ^ 2)
        .into_iter()
        .map(|x| x * 1.0e2)
        .collect::<Vec<_>>();
    let hv = random_f32(n, seed ^ 3)
        .into_iter()
        .map(|x| x.abs() * 1.0e2)
        .collect::<Vec<_>>();
    let p = rt.alloc_tensor_f32(&[n]).unwrap();
    let g = rt.alloc_tensor_f32(&[n]).unwrap();
    let m = rt.alloc_tensor_f32(&[n]).unwrap();
    let v = rt.alloc_tensor_f32(&[n]).unwrap();
    p.write_f32(&hp).unwrap();
    g.write_f32(&hg).unwrap();
    m.write_f32(&hm).unwrap();
    v.write_f32(&hv).unwrap();
    StepTensors {
        p,
        g,
        m,
        v,
        hp,
        hg,
        hm,
        hv,
    }
}

#[test]
fn device_adamw_matches_noncontracted_f32_within_adamw_abs_tol() {
    with_gpu(|rt| {
        let hyper = AdamWHyper {
            lr: 1e-2,
            ..AdamWHyper::default()
        };
        let wd = 0.1f32;
        let StepTensors {
            mut p,
            g,
            mut m,
            mut v,
            mut hp,
            hg,
            mut hm,
            mut hv,
        } = tensors(rt, 4096, 0xA4);
        let mut worst = 0.0f32;
        for step in 1..=3 {
            adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, step, wd).unwrap();
            rt.synchronize().unwrap();
            adamw_f32_ref(&mut hp, &hg, &mut hm, &mut hv, &hyper, step, wd);
            let got = p.read_f32().unwrap();
            for (i, (a, b)) in got.iter().zip(&hp).enumerate() {
                let err = (a - b).abs();
                if err > worst {
                    worst = err;
                }
                assert!(err <= ADAMW_ABS_TOL, "step {step} [{i}]: gpu {a} ref {b} err {err:.3e}");
            }
        }
        eprintln!("adamw f32 parity max abs err {worst:.3e} (tol {ADAMW_ABS_TOL:.3e})");
    });
}

#[test]
fn device_adamw_refuses_a_bad_step_length_or_dtype_and_accepts_empty() {
    with_gpu(|rt| {
        let hyper = AdamWHyper::default();
        let StepTensors {
            mut p,
            g,
            mut m,
            mut v,
            hp: p0,
            ..
        } = tensors(rt, 8, 0x11);
        let before = p.read_f32().unwrap();
        let err = adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 0, 0.0).unwrap_err();
        assert!(err.contains("step 0") && err.contains("u64::MAX"), "{err}");
        rt.synchronize().unwrap();
        assert_eq!(p.read_f32().unwrap(), before);

        let short = rt.alloc_tensor_f32(&[7]).unwrap();
        let err = adamw_step(rt, &mut p, &short, &mut m, &mut v, &hyper, 1, 0.0).unwrap_err();
        assert!(err.contains("shapes differ"), "{err}");

        let bf = rt.alloc_tensor_bf16(&[8]).unwrap();
        let err = adamw_step(rt, &mut p, &bf, &mut m, &mut v, &hyper, 1, 0.0).unwrap_err();
        assert!(err.contains("not f32"), "{err}");
        assert_eq!(p.read_f32().unwrap(), p0);

        let empty = |seed_shape: &[usize]| {
            let base = rt.alloc_tensor_f32(seed_shape).unwrap();
            base.try_view(&[0], 0).unwrap()
        };
        let mut pe = empty(&[4]);
        let ge = empty(&[4]);
        let mut me = empty(&[4]);
        let mut ve = empty(&[4]);
        adamw_step(rt, &mut pe, &ge, &mut me, &mut ve, &hyper, 1, 0.1).unwrap();

        let err = adamw_step(rt, &mut pe, &ge, &mut me, &mut ve, &hyper, 1, f32::NAN).unwrap_err();
        assert!(err.contains("weight decay"), "{err}");
    });
}

#[test]
fn device_adamw_honours_a_byte_offset_and_rejects_an_overlapping_grad() {
    with_gpu(|rt| {
        let hyper = AdamWHyper {
            lr: 1e-3,
            ..AdamWHyper::default()
        };
        let parent = rt.alloc_tensor_f32(&[8]).unwrap();
        let payload = random_f32(4, 0x22).into_iter().map(|x| x * 10.0).collect::<Vec<_>>();
        let mut host = vec![7.0f32; 4];
        host.extend_from_slice(&payload);
        parent.write_f32(&host).unwrap();
        let mut param = parent.try_view(&[4], 4).unwrap();
        let g = rt.alloc_tensor_f32(&[4]).unwrap();
        let mut m = rt.alloc_tensor_f32(&[4]).unwrap();
        let mut v = rt.alloc_tensor_f32(&[4]).unwrap();
        let grad = random_f32(4, 0x23);
        g.write_f32(&grad).unwrap();
        m.write_f32(&[0.0; 4]).unwrap();
        v.write_f32(&[0.0; 4]).unwrap();
        let mut href = payload.clone();
        let mut hm = vec![0.0; 4];
        let mut hv = vec![0.0; 4];
        adamw_step(rt, &mut param, &g, &mut m, &mut v, &hyper, 1, 0.0).unwrap();
        rt.synchronize().unwrap();
        adamw_f32_ref(&mut href, &grad, &mut hm, &mut hv, &hyper, 1, 0.0);
        let full = parent.read_f32().unwrap();
        assert_eq!(&full[..4], &[7.0; 4], "the view's step wrote before its offset");
        for (i, (a, b)) in full[4..].iter().zip(&href).enumerate() {
            assert!((a - b).abs() <= ADAMW_ABS_TOL, "view[{i}]: {a} vs {b}");
        }

        let alias = param.clone();
        let err = adamw_step(rt, &mut param, &alias, &mut m, &mut v, &hyper, 2, 0.0).unwrap_err();
        assert!(err.contains("overlaps"), "{err}");
    });
}

#[test]
fn device_adamw_step_at_u64_max_is_applied_and_a_foreign_runtime_is_refused() {
    with_gpu(|rt| {
        let hyper = AdamWHyper::default();
        let StepTensors {
            mut p,
            g,
            mut m,
            mut v,
            mut hp,
            hg,
            mut hm,
            mut hv,
        } = tensors(rt, 32, 0x44);
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, u64::MAX, 0.0).unwrap();
        rt.synchronize().unwrap();
        adamw_f32_ref(&mut hp, &hg, &mut hm, &mut hv, &hyper, u64::MAX, 0.0);
        let got = p.read_f32().unwrap();
        for (i, (a, b)) in got.iter().zip(&hp).enumerate() {
            assert!((a - b).abs() <= ADAMW_ABS_TOL, "u64::MAX [{i}]: {a} vs {b}");
        }

        let other = GpuRuntime::new().unwrap();
        let foreign = other.alloc_tensor_f32(&[32]).unwrap();
        let err = adamw_step(rt, &mut p, &foreign, &mut m, &mut v, &hyper, 1, 0.0).unwrap_err();
        assert!(err.contains("different runtime"), "{err}");
    });
}

/// Times one large step so the contraction flag's cost can be recorded.
/// The number is the GPU work plus one submit, not a host loop.
#[test]
fn device_adamw_step_time_on_a_million_elements() {
    with_gpu(|rt| {
        let n = 1 << 20;
        let StepTensors {
            mut p, g, mut m, mut v, ..
        } = tensors(rt, n, 0x55);
        let hyper = AdamWHyper::default();
        // Warm the pipeline and the pool.
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 1, 0.01).unwrap();
        rt.synchronize().unwrap();
        rt.set_async_encode(true).unwrap();
        let start = Instant::now();
        const STEPS: u64 = 32;
        for step in 2..2 + STEPS {
            adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, step, 0.01).unwrap();
        }
        rt.synchronize().unwrap();
        let per = start.elapsed().as_secs_f64() / STEPS as f64 * 1.0e3;
        eprintln!("adamw_step {n} elements, {STEPS} batched steps: {per:.4} ms/step");
    });
}

/// RMSNorm versus a serial non-contracted f32 sum. The reduction tree already
/// reassociates, so this records the gap rather than requiring bit equality.
#[test]
fn rms_norm_gap_against_serial_f32() {
    with_gpu(|rt| {
        let (rows, dim, eps) = (8usize, 256usize, 1e-6f32);
        let x = random_f32(rows * dim, 0x71);
        let w = random_f32(dim, 0x72);
        let xb = buf(rt, &x);
        let wb = buf(rt, &w);
        let ob = empty(rt, rows * dim);
        let start = Instant::now();
        const REPS: u32 = 64;
        rt.set_async_encode(true).unwrap();
        for _ in 0..REPS {
            nn::rms_norm_f32(rt, &xb, &wb, &ob, rows as u32, dim as u32, eps).unwrap();
        }
        rt.synchronize().unwrap();
        let per = start.elapsed().as_secs_f64() / f64::from(REPS) * 1.0e3;
        let got = ob.read_f32();
        let mut worst = 0.0f32;
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
            for d in 0..dim {
                let want = row[d] * inv * w[d];
                worst = worst.max((got[r * dim + d] - want).abs());
            }
        }
        eprintln!("rms_norm_f32 {rows}x{dim}, {REPS} batched: {per:.4} ms/step, max abs err {worst:.3e}");
        assert!(worst <= 1e-5, "rms gap {worst:.3e} exceeds the existing 1e-5 bound");
    });
}
