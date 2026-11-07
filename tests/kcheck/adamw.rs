//! AdamW entry points against an independent f32 reference.
//!
//! `qwen35_adamw_f32` uses the same absolute bound as `tests/adamw_step.rs`
//! (`2e-6`): large values, so a contracted fma misses it. The 21 stored
//! kernels run one dyadic step whose f32 result is exact, then the kernel's
//! own store rule (bf16 nearest, Kahan, stochastic, or q8 block encode).

use std::sync::Arc;

use objc2_metal::MTLBuffer;

use crate::common::{buf, random_f32};
use tessl::dispatch::{dispatch_2d_tg, set_gpu_buf};
use tessl::qwen35_adamw::{adamw_step, AdamWHyper};
use tessl::tensor::{bf16_bits_to_f32, f32_to_bf16_bits_gpu, GpuBuffer};
use tessl::GpuRuntime;

use super::Ledger;

const N: usize = 8;
const ABS: f32 = 2e-6;

fn gpu_addr(b: &GpuBuffer) -> u64 {
    b.metal().gpuAddress()
}

fn f32_step(a: &[f32; 8], w: f32, g: f32, m: f32, v: f32) -> (f32, f32, f32, f32) {
    let gi = g * a[7];
    let diff = gi - m;
    let mi = if a[1] < 0.5 {
        m + a[1] * diff
    } else {
        gi - diff * (1.0 - a[1])
    };
    let vi = v * a[2] + (a[3] * gi) * gi;
    let denom = vi.sqrt() / a[5] + a[6];
    let dir = mi / denom;
    let updated = w * a[0] + (-a[4]) * dir;
    let delta = w * (a[0] - 1.0) + (-a[4]) * dir;
    (updated, delta, mi, vi)
}

fn dyadic_scalars() -> [f32; 8] {
    // step 1, lr 1/4, beta1 1/2, beta2 3/4, eps 0, decay 1.
    // denom = sqrt(v) / sqrt(1/4) = 2, so the direction is an exact f32.
    let beta2 = 0.75f32;
    let lr = 0.25f32;
    let beta1 = 0.5f32;
    [
        1.0,
        (1.0 - beta1),
        beta2,
        (1.0 - beta2),
        lr / (1.0 - beta1),
        (1.0 - beta2).sqrt(),
        0.0,
        1.0,
    ]
}

fn words() -> [f32; 8] {
    [1.0, 2.0, 4.0, 8.0, 0.5, 1.5, 2.5, 3.5]
}

fn put_u64(slot: &mut [u8], off: usize, v: u64) {
    slot[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(slot: &mut [u8], off: usize, v: u32) {
    slot[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_f32(slot: &mut [u8], off: usize, v: f32) {
    slot[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn bf16_buf(rt: &GpuRuntime, data: &[f32]) -> GpuBuffer {
    let bits: Vec<u16> = data.iter().copied().map(f32_to_bf16_bits_gpu).collect();
    let b = rt.alloc_buffer(bits.len() * 2).unwrap();
    b.write_bf16_bits(&bits);
    b
}

fn u8_buf(rt: &GpuRuntime, data: &[u8]) -> GpuBuffer {
    let b = rt.alloc_buffer(data.len()).unwrap();
    b.write_bytes(data);
    b
}

fn bits_f32(label: &str, got: &[f32], want: &[f32]) {
    assert!(got.len() >= want.len(), "{label}: short read");
    for (i, w) in want.iter().enumerate() {
        assert_eq!(got[i].to_bits(), w.to_bits(), "{label}[{i}] {} vs {w}", got[i]);
    }
}

fn bits_u16(label: &str, got: &[u16], want: &[u16]) {
    assert!(got.len() >= want.len(), "{label}: short read");
    for (i, w) in want.iter().enumerate() {
        assert_eq!(got[i], *w, "{label}[{i}] {:#06x} vs {w:#06x}", got[i]);
    }
}

fn q8m_encode(x: f32, s: f32) -> u8 {
    if s.is_nan() || s <= 0.0 {
        return 0;
    }
    let r = ((x.abs() / s).min(1.0).sqrt() * 127.0).round_ties_even();
    let signed = if x < 0.0 { -r } else { r };
    signed as i8 as u8
}

fn q8v_encode(u: f32, s: f32) -> u8 {
    if s.is_nan() || s <= 0.0 {
        return 0;
    }
    let mut r = ((u / s).min(1.0).sqrt() * 255.0).round_ties_even();
    if u > 0.0 && r < 1.0 {
        r = 1.0;
    }
    r as u8
}

fn sr_bits(i: u32) -> u32 {
    fn mix(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846ca68b);
        x ^= x >> 16;
        x
    }
    // The key words the slot carries (raw offsets 112..132 in `stored`).
    let (seed_lo, seed_hi, step_lo, step_hi, salt) = (1u32, 2u32, 1u32, 0u32, 7u32);
    let mut h = mix(seed_lo ^ mix(seed_hi));
    h = mix(h ^ step_lo);
    h = mix(h ^ step_hi);
    h = mix(h ^ salt);
    mix(h ^ i)
}

fn round_stochastic(x: f32, r: u32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return f32_to_bf16_bits_gpu(x);
    }
    ((bits + (r & 0xffff)) >> 16) as u16
}

#[derive(Clone, Copy)]
struct Store {
    param_bf16: bool,
    grad_bf16: bool,
    mom: Mom,
    rule: Rule,
}

#[derive(Clone, Copy)]
enum Mom {
    F32,
    Bf16,
    Q8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    Plain,
    Master,
    Kahan,
    Sr,
}

impl Store {
    fn name(self) -> String {
        let mom = match self.mom {
            Mom::F32 => "mf32",
            Mom::Bf16 => "mbf16",
            Mom::Q8 => "mq8",
        };
        let g = if self.grad_bf16 { "gbf16" } else { "gf32" };
        match self.rule {
            Rule::Plain => format!("qwen35_adamw_f32_plain_gf32_{mom}"),
            Rule::Master => format!("qwen35_adamw_bf16_master_{g}_{mom}"),
            Rule::Kahan => format!("qwen35_adamw_bf16_kahan_{g}_{mom}"),
            Rule::Sr => format!("qwen35_adamw_bf16_sr_{g}_{mom}"),
        }
    }
}

fn launch_stored(rt: &GpuRuntime, name: &str, slot: &GpuBuffer, n_slots: &GpuBuffer) {
    let p = rt.pipeline(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    dispatch_2d_tg(rt, &p, 1, 1, 256, |bnd| {
        set_gpu_buf(bnd, slot, 0);
        set_gpu_buf(bnd, n_slots, 1);
    })
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    rt.synchronize().unwrap_or_else(|e| panic!("{name} sync: {e}"));
}

fn stored(rt: &Arc<GpuRuntime>, ledger: &mut Ledger, spec: Store) {
    let name = spec.name();
    let a = dyadic_scalars();
    let w = words();
    let g = [2.0f32; N];
    let zeros = [0.0f32; N];
    let grad_f = if spec.grad_bf16 {
        g.iter()
            .copied()
            .map(|x| bf16_bits_to_f32(f32_to_bf16_bits_gpu(x)))
            .collect::<Vec<_>>()
    } else {
        g.to_vec()
    };
    let mut want_p = Vec::with_capacity(N);
    let mut want_m = Vec::with_capacity(N);
    let mut want_v = Vec::with_capacity(N);
    let mut want_delta = Vec::with_capacity(N);
    for i in 0..N {
        // Master and the other rules start from the same value: the master copy
        // `aux_f` is seeded with `w`, and every `w` word is bf16-exact.
        let (updated, delta, mi, vi) = f32_step(&a, w[i], grad_f[i], 0.0, 0.0);
        want_p.push(updated);
        want_delta.push(delta);
        want_m.push(mi);
        want_v.push(vi);
    }

    let dummy = buf(rt, &[0.0]);
    let p_f = buf(rt, &w);
    let p_b = bf16_buf(rt, &w);
    let g_f = buf(rt, &g);
    let g_b = bf16_buf(rt, &g);
    let m_f = buf(rt, &zeros);
    let v_f = buf(rt, &zeros);
    let m_b = bf16_buf(rt, &zeros);
    let v_b = bf16_buf(rt, &zeros);
    let m_q = u8_buf(rt, &[0u8; N]);
    let v_q = u8_buf(rt, &[0u8; N]);
    let m_scale = buf(rt, &[0.0]);
    let v_scale = buf(rt, &[0.0]);
    let aux_f = buf(rt, &w);
    let aux_b = bf16_buf(rt, &zeros);
    let n_slots = {
        let b = rt.alloc_buffer(4).unwrap();
        b.write_u32(&[1]);
        b
    };

    let (p_addr, g_addr, m_addr, v_addr, aux_addr) = match (spec.rule, spec.mom, spec.grad_bf16) {
        (_, Mom::F32, false) => (
            gpu_addr(&p_f),
            gpu_addr(&g_f),
            gpu_addr(&m_f),
            gpu_addr(&v_f),
            gpu_addr(&dummy),
        ),
        (_, Mom::Bf16, false) => (
            gpu_addr(&p_f),
            gpu_addr(&g_f),
            gpu_addr(&m_b),
            gpu_addr(&v_b),
            gpu_addr(&dummy),
        ),
        (_, Mom::Q8, false) => (
            gpu_addr(&p_f),
            gpu_addr(&g_f),
            gpu_addr(&m_q),
            gpu_addr(&v_q),
            gpu_addr(&dummy),
        ),
        (_, Mom::F32, true) => (
            gpu_addr(&p_f),
            gpu_addr(&g_b),
            gpu_addr(&m_f),
            gpu_addr(&v_f),
            gpu_addr(&dummy),
        ),
        (_, Mom::Bf16, true) => (
            gpu_addr(&p_f),
            gpu_addr(&g_b),
            gpu_addr(&m_b),
            gpu_addr(&v_b),
            gpu_addr(&dummy),
        ),
        (_, Mom::Q8, true) => (
            gpu_addr(&p_f),
            gpu_addr(&g_b),
            gpu_addr(&m_q),
            gpu_addr(&v_q),
            gpu_addr(&dummy),
        ),
    };
    let p_addr = if spec.param_bf16 { gpu_addr(&p_b) } else { p_addr };
    let aux_addr = match spec.rule {
        Rule::Master => gpu_addr(&aux_f),
        Rule::Kahan => gpu_addr(&aux_b),
        _ => aux_addr,
    };

    let mut raw = vec![0u8; 136];
    put_u64(&mut raw, 0, p_addr);
    put_u64(&mut raw, 8, g_addr);
    put_u64(&mut raw, 16, m_addr);
    put_u64(&mut raw, 24, v_addr);
    put_u64(&mut raw, 32, aux_addr);
    put_u64(&mut raw, 40, gpu_addr(&m_scale));
    put_u64(&mut raw, 48, gpu_addr(&v_scale));
    put_u32(&mut raw, 56, N as u32);
    put_u32(&mut raw, 60, N as u32);
    put_u32(&mut raw, 64, N as u32);
    put_u32(&mut raw, 68, 0);
    put_u32(&mut raw, 72, 0);
    for (i, s) in a.iter().enumerate() {
        put_f32(&mut raw, 80 + i * 4, *s);
    }
    put_u32(&mut raw, 112, 1);
    put_u32(&mut raw, 116, 2);
    put_u32(&mut raw, 120, 1);
    put_u32(&mut raw, 124, 0);
    put_u32(&mut raw, 128, 7);
    let slot = rt.alloc_buffer(raw.len()).unwrap();
    slot.write_bytes(&raw);

    let label = name.clone();
    ledger.credit(rt, &label, &[label.as_str()], || {
        launch_stored(rt, &label, &slot, &n_slots);
        match spec.rule {
            Rule::Plain | Rule::Master => {
                if spec.param_bf16 {
                    let want: Vec<u16> = want_p.iter().copied().map(f32_to_bf16_bits_gpu).collect();
                    bits_u16(&label, &p_b.contents_u16(), &want);
                } else {
                    bits_f32(&label, &p_f.read_f32(), &want_p);
                }
                if spec.rule == Rule::Master {
                    bits_f32(&format!("{label} aux"), &aux_f.read_f32(), &want_p);
                }
            }
            Rule::Kahan => {
                let want_bits: Vec<u16> = want_p.iter().copied().map(f32_to_bf16_bits_gpu).collect();
                bits_u16(&label, &p_b.contents_u16(), &want_bits);
                let aux_want: Vec<u16> = (0..N)
                    .map(|i| {
                        let pn = bf16_bits_to_f32(want_bits[i]);
                        let c = want_delta[i];
                        f32_to_bf16_bits_gpu(c - (pn - w[i]))
                    })
                    .collect();
                bits_u16(&format!("{label} aux"), &aux_b.contents_u16(), &aux_want);
            }
            Rule::Sr => {
                let want: Vec<u16> = (0..N)
                    .map(|i| round_stochastic(w[i] + want_delta[i], sr_bits(i as u32)))
                    .collect();
                bits_u16(&label, &p_b.contents_u16(), &want);
            }
        }
        match spec.mom {
            Mom::F32 => {
                bits_f32(&format!("{label} m"), &m_f.read_f32(), &want_m);
                bits_f32(&format!("{label} v"), &v_f.read_f32(), &want_v);
            }
            Mom::Bf16 => {
                let mb: Vec<u16> = want_m.iter().copied().map(f32_to_bf16_bits_gpu).collect();
                let vb: Vec<u16> = want_v.iter().copied().map(f32_to_bf16_bits_gpu).collect();
                bits_u16(&format!("{label} m"), &m_b.contents_u16(), &mb);
                bits_u16(&format!("{label} v"), &v_b.contents_u16(), &vb);
            }
            Mom::Q8 => {
                let sm = want_m.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let sv = want_v.iter().map(|x| x.sqrt()).fold(0.0f32, f32::max);
                let mc: Vec<u8> = want_m.iter().copied().map(|x| q8m_encode(x, sm)).collect();
                let vc: Vec<u8> = want_v.iter().copied().map(|x| q8v_encode(x.sqrt(), sv)).collect();
                let got_m = m_q.contents_u8()[..N].to_vec();
                let got_v = v_q.contents_u8()[..N].to_vec();
                assert_eq!(got_m, mc, "{label} m codes");
                assert_eq!(got_v, vc, "{label} v codes");
                bits_f32(&format!("{label} m scale"), &m_scale.read_f32()[..1], &[sm]);
                bits_f32(&format!("{label} v scale"), &v_scale.read_f32()[..1], &[sv]);
            }
        }
    });
}

fn dense_f32(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    let n = 64usize;
    let hyper = AdamWHyper {
        lr: 1e-2,
        ..AdamWHyper::default()
    };
    let wd = 0.1f32;
    let mut hp: Vec<f32> = random_f32(n, 0xA4).into_iter().map(|x| x * 1.0e4).collect();
    let hg: Vec<f32> = random_f32(n, 0xA5).into_iter().map(|x| x * 1.0e4).collect();
    let mut hm: Vec<f32> = random_f32(n, 0xA6).into_iter().map(|x| x * 1.0e2).collect();
    let mut hv: Vec<f32> = random_f32(n, 0xA7).into_iter().map(|x| x.abs() * 1.0e2).collect();
    let t = 1u64;
    let bc1 = 1.0 - hyper.beta1.powf(t as f64);
    let bc2 = 1.0 - hyper.beta2.powf(t as f64);
    let step_size = (hyper.lr / bc1) as f32;
    let bc2_sqrt = bc2.powf(0.5) as f32;
    let decay = (1.0 - hyper.lr * f64::from(wd)) as f32;
    let lerp_w = (1.0 - hyper.beta1) as f32;
    let beta2 = hyper.beta2 as f32;
    let one_minus = (1.0 - hyper.beta2) as f32;
    let eps = hyper.eps as f32;
    let scale = hyper.grad_scale as f32;
    for i in 0..n {
        let gi = hg[i] * scale;
        let diff = gi - hm[i];
        hm[i] = if lerp_w < 0.5 {
            hm[i] + lerp_w * diff
        } else {
            gi - diff * (1.0 - lerp_w)
        };
        hv[i] = hv[i] * beta2 + (one_minus * gi) * gi;
        let denom = hv[i].sqrt() / bc2_sqrt + eps;
        hp[i] = hp[i] * decay + (-step_size) * (hm[i] / denom);
    }
    let want_p = hp.clone();
    let want_m = hm.clone();
    let want_v = hv.clone();
    ledger.credit(rt, "qwen35_adamw_f32", &["qwen35_adamw_f32"], || {
        let mut p = rt.alloc_tensor_f32(&[n]).unwrap();
        let g = rt.alloc_tensor_f32(&[n]).unwrap();
        let mut m = rt.alloc_tensor_f32(&[n]).unwrap();
        let mut v = rt.alloc_tensor_f32(&[n]).unwrap();
        // Rebuild the pre-step hosts. `hp` was overwritten by the reference.
        let p0: Vec<f32> = random_f32(n, 0xA4).into_iter().map(|x| x * 1.0e4).collect();
        let g0: Vec<f32> = random_f32(n, 0xA5).into_iter().map(|x| x * 1.0e4).collect();
        let m0: Vec<f32> = random_f32(n, 0xA6).into_iter().map(|x| x * 1.0e2).collect();
        let v0: Vec<f32> = random_f32(n, 0xA7).into_iter().map(|x| x.abs() * 1.0e2).collect();
        p.write_f32(&p0).unwrap();
        g.write_f32(&g0).unwrap();
        m.write_f32(&m0).unwrap();
        v.write_f32(&v0).unwrap();
        adamw_step(rt, &mut p, &g, &mut m, &mut v, &hyper, 1, wd).unwrap_or_else(|e| panic!("adamw_f32: {e}"));
        rt.synchronize().unwrap();
        let gp = p.read_f32().unwrap();
        let gm = m.read_f32().unwrap();
        let gv = v.read_f32().unwrap();
        for i in 0..n {
            assert!(
                (gp[i] - want_p[i]).abs() <= ABS,
                "adamw p[{i}] {} vs {}",
                gp[i],
                want_p[i]
            );
            assert!(
                (gm[i] - want_m[i]).abs() <= ABS,
                "adamw m[{i}] {} vs {}",
                gm[i],
                want_m[i]
            );
            // One f32 ulp of a ~1e5 second moment is ~1e-2, past the 2e-6
            // parameter bound. 1e-6 relative still rejects a wrong beta.
            let tol_v = want_v[i].abs() * 1.0e-6;
            assert!(
                (gv[i] - want_v[i]).abs() <= tol_v.max(ABS),
                "adamw v[{i}] {} vs {}",
                gv[i],
                want_v[i]
            );
        }
    });
}

pub fn run(rt: &Arc<GpuRuntime>, ledger: &mut Ledger) {
    dense_f32(rt, ledger);
    for mom in [Mom::F32, Mom::Bf16, Mom::Q8] {
        stored(
            rt,
            ledger,
            Store {
                param_bf16: false,
                grad_bf16: false,
                mom,
                rule: Rule::Plain,
            },
        );
    }
    for rule in [Rule::Master, Rule::Kahan, Rule::Sr] {
        for grad_bf16 in [false, true] {
            for mom in [Mom::F32, Mom::Bf16, Mom::Q8] {
                stored(
                    rt,
                    ledger,
                    Store {
                        param_bf16: true,
                        grad_bf16,
                        mom,
                        rule,
                    },
                );
            }
        }
    }
}
