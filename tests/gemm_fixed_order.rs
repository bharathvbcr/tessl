//! Fixed-order f32 GEMM against `gemm_f32_cpu`'s ascending-k left fold.
//!
//! The TensorOps 32×32 tile and the simdgroup tile match that fold on short K
//! and on integer products. They do not match it on order-sensitive products
//! whose K fills a hardware chunk (8×16×16). These kernels are the ones
//! `Numerics::Exact` dispatches, and the comparison is `to_bits`.

mod common;

use common::{tensor_f32, with_gpu};
use tessl::gemm::{gemm_f32_cpu, gemm_fixed_order_nn, gemm_fixed_order_nt, gemm_fixed_order_tn};

fn order_sensitive(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for i in 0..m {
        for p in 0..k {
            a[i * k + p] = match p {
                0 => 1.0,
                1 => -1.0,
                _ => (p as f32) * 1.0e-6,
            };
        }
    }
    for p in 0..k {
        for j in 0..n {
            b[p * n + j] = 1.0 + (j as f32) * 1.0e-3;
        }
    }
    (a, b)
}

fn integers(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0f32; m * k];
    let mut b = vec![0.0f32; k * n];
    for (i, x) in a.iter_mut().enumerate() {
        *x = ((i % 7) as f32) - 3.0;
    }
    for (i, x) in b.iter_mut().enumerate() {
        *x = ((i % 5) as f32) - 2.0;
    }
    (a, b)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn nt_stored(b_nn: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; n * k];
    for j in 0..n {
        for p in 0..k {
            b[j * k + p] = b_nn[p * n + j];
        }
    }
    b
}

fn tn_stored(a_nn: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; k * m];
    for p in 0..k {
        for i in 0..m {
            a[p * m + i] = a_nn[i * k + p];
        }
    }
    a
}

#[test]
fn fixed_order_matches_the_cpu_left_fold_on_order_sensitive_products() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let shapes = [(3usize, 5usize, 7usize), (8, 16, 16), (4, 8, 9), (2, 2, 2)];
        for &(m, n, k) in &shapes {
            for (label, (a_nn, b_nn)) in [("order", order_sensitive(m, n, k)), ("integer", integers(m, n, k))] {
                let want = gemm_f32_cpu(&a_nn, &b_nn, m, n, k);

                let a = tensor_f32(rt, &[m, k], &a_nn);
                let b = tensor_f32(rt, &[k, n], &b_nn);
                let c = rt.alloc_tensor_f32(&[m, n]).expect("c");
                gemm_fixed_order_nn(&a, &b, &c).expect("nn");
                rt.synchronize().expect("sync");
                let got = c.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got), bits(&want), "nn {label} {m}x{n}x{k}");

                let b_nt = nt_stored(&b_nn, n, k);
                let bn = tensor_f32(rt, &[n, k], &b_nt);
                let c_nt = rt.alloc_tensor_f32(&[m, n]).expect("c nt");
                gemm_fixed_order_nt(&a, &bn, &c_nt).expect("nt");
                rt.synchronize().expect("sync");
                let got_nt = c_nt.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got_nt), bits(&want), "nt {label} {m}x{n}x{k}");

                let a_tn = tn_stored(&a_nn, m, k);
                let at = tensor_f32(rt, &[k, m], &a_tn);
                let c_tn = rt.alloc_tensor_f32(&[m, n]).expect("c tn");
                gemm_fixed_order_tn(&at, &b, &c_tn).expect("tn");
                rt.synchronize().expect("sync");
                let got_tn = c_tn.buffer.read_f32()[..m * n].to_vec();
                assert_eq!(bits(&got_tn), bits(&want), "tn {label} {m}x{n}x{k}");
            }
        }
    });
}

/// The kernels' own fold on the host: `acc = +0`, `acc += a * b` for `k`
/// ascending, two f32 roundings and no FMA. aarch64 does not flush subnormals,
/// so this is the IEEE answer.
fn ieee_fold(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let prod = a[i * k + p] * b[p * n + j];
                acc += prod;
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Runs `a[m,k] @ b[k,n]` through nn, nt and tn and returns the three results.
fn run_all_layouts(
    rt: &std::sync::Arc<tessl::GpuRuntime>,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> [(&'static str, Vec<f32>); 3] {
    let at = tensor_f32(rt, &[m, k], a);
    let bt = tensor_f32(rt, &[k, n], b);
    let c_nn = rt.alloc_tensor_f32(&[m, n]).expect("c nn");
    gemm_fixed_order_nn(&at, &bt, &c_nn).expect("nn");

    let bn = tensor_f32(rt, &[n, k], &nt_stored(b, n, k));
    let c_nt = rt.alloc_tensor_f32(&[m, n]).expect("c nt");
    gemm_fixed_order_nt(&at, &bn, &c_nt).expect("nt");

    let a_tn = tensor_f32(rt, &[k, m], &tn_stored(a, m, k));
    let c_tn = rt.alloc_tensor_f32(&[m, n]).expect("c tn");
    gemm_fixed_order_tn(&a_tn, &bt, &c_tn).expect("tn");

    rt.synchronize().expect("sync");
    let read = |c: &tessl::Tensor| c.buffer.read_f32()[..m * n].to_vec();
    [("nn", read(&c_nn)), ("nt", read(&c_nt)), ("tn", read(&c_tn))]
}

/// xorshift64 stream of the scratch control that measured this behaviour, so
/// the mixed-operand case is the same construction (card ft-02eec315).
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A normal f32 in 2^-20 .. 2^20, either sign.
    fn normal(&mut self) -> f32 {
        let r = self.next();
        let exp = 127 - 20 + ((r >> 24) % 41) as u32;
        f32::from_bits((((r >> 40) & 1) as u32) << 31 | exp << 23 | (r & 0x7f_ffff) as u32)
    }
    /// A subnormal f32 (mantissa never 0), either sign.
    fn subnormal(&mut self) -> f32 {
        let r = self.next();
        f32::from_bits((((r >> 40) & 1) as u32) << 31 | ((r & 0x7f_ffff) as u32).max(1))
    }
}

/// Pins observed hardware behaviour, not a design goal: the fixed-order GEMM
/// kernels flush f32 subnormal operands, products and partial sums to zero on
/// the GPU, so they differ from the IEEE CPU fold exactly where one appears.
/// Measured on an M5 Pro on 2026-11-08 and identical under
/// `-fmetal-math-mode=fast` and `safe`, so the flush is not the compiler flag.
/// The contract is documented on [`tessl::gemm::gemm_fixed_order_nn`]
/// (`# Subnormals`); GitPulse card ft-02eec315. If this test fails, the
/// hardware, the driver or the OS compile changed: read the failing case, then
/// revisit that doc section and every caller that relies on `Numerics::Exact`.
/// Each case also asserts the host fold's IEEE bits, so a mistake in the
/// construction cannot masquerade as a hardware change.
#[test]
fn fixed_order_flushes_f32_subnormals_to_zero_on_this_gpu() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let sub = f32::from_bits;
        // Built from bits: `f32::powi(2.0, -130)` overflowed in the control.
        let p2 = |e: i32| f32::from_bits(((127 + e) as u32) << 23);
        let two_m130 = sub(1 << (149 - 130));
        assert_eq!(two_m130.to_bits(), 0x0008_0000);

        // (label, a, b, k, IEEE bits, GPU bits): 1 x 1 x k, one output each.
        type OneOut = (&'static str, Vec<f32>, Vec<f32>, usize, u32, u32);
        let cases: Vec<OneOut> = vec![
            ("a: 0x00000001 * 1.0", vec![sub(1)], vec![1.0], 1, 0x0000_0001, 0),
            (
                "a: 0x00400000 * 1.0",
                vec![sub(0x0040_0000)],
                vec![1.0],
                1,
                0x0040_0000,
                0,
            ),
            (
                "a: 0x007fffff * 1.0",
                vec![sub(0x007f_ffff)],
                vec![1.0],
                1,
                0x007f_ffff,
                0,
            ),
            (
                "a: 0x80000001 * 1.0 (negative flushes to +0)",
                vec![sub(0x8000_0001)],
                vec![1.0],
                1,
                0x8000_0001,
                0,
            ),
            (
                "a: 1.0 * 0x00400000 (subnormal B)",
                vec![1.0],
                vec![sub(0x0040_0000)],
                1,
                0x0040_0000,
                0,
            ),
            (
                "a: min normal 0x00800000 * 1.0 survives",
                vec![sub(0x0080_0000)],
                vec![1.0],
                1,
                0x0080_0000,
                0x0080_0000,
            ),
            (
                "b1: 2^-130 * 2^10 (normal product)",
                vec![two_m130],
                vec![p2(10)],
                1,
                0x0380_0000,
                0,
            ),
            (
                "b2: 2^-100 * 2^-30 (subnormal product)",
                vec![p2(-100)],
                vec![p2(-30)],
                1,
                0x0008_0000,
                0,
            ),
            (
                "b3: 4 x (0x1 * 1.0) (sum of subnormals)",
                vec![sub(1); 4],
                vec![1.0; 4],
                4,
                0x0000_0004,
                0,
            ),
            (
                "b4: 2 x (0x007fffff * 1.0) (sum is normal)",
                vec![sub(0x007f_ffff); 2],
                vec![1.0; 2],
                2,
                0x00ff_fffe,
                0,
            ),
            (
                "b6: 0x00800001 + (-0x00800000) (subnormal sum)",
                vec![sub(0x0080_0001), sub(0x0080_0000)],
                vec![1.0, -1.0],
                2,
                0x0000_0001,
                0,
            ),
        ];
        for (label, a, b, k, ieee, gpu) in &cases {
            let want = ieee_fold(a, b, 1, 1, *k);
            assert_eq!(want[0].to_bits(), *ieee, "{label}: host fold is not the IEEE bits");
            for (layout, got) in run_all_layouts(rt, a, b, 1, 1, *k) {
                assert_eq!(got[0].to_bits(), *gpu, "{label}: {layout} (IEEE would be {ieee:#010x})");
            }
        }

        // 1 x 4097 x 1: A = [0.75], B mixed normal / subnormal (every tenth
        // element subnormal). Each subnormal B element reads as zero, so its
        // output is +0 where the fold keeps a subnormal; the other outputs are
        // bit-identical to the fold. The control measured 410 of 4097 zeroed.
        let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
        let mixed: Vec<f32> = (0..4097)
            .map(|i| if i % 10 == 0 { rng.subnormal() } else { rng.normal() })
            .collect();
        let want = ieee_fold(&[0.75], &mixed, 1, 4097, 1);
        let subnormal_outputs = want.iter().filter(|x| x.is_subnormal()).count();
        assert_eq!(subnormal_outputs, 410, "the construction changed");
        for (layout, got) in run_all_layouts(rt, &[0.75], &mixed, 1, 4097, 1) {
            let zeroed = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() == 0 && w.is_subnormal())
                .count();
            assert_eq!(zeroed, 410, "c1 {layout}: zeroed outputs");
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                let expect = if w.is_subnormal() { 0 } else { w.to_bits() };
                assert_eq!(g.to_bits(), expect, "c1 {layout} output {j}");
            }
        }

        // Normal-range control: no operand, product or partial sum is
        // subnormal, so the kernels stay bit-identical to the host fold.
        let an: Vec<f32> = (0..2 * 257).map(|_| rng.normal()).collect();
        let bn: Vec<f32> = (0..257 * 64).map(|_| rng.normal()).collect();
        let want = ieee_fold(&an, &bn, 2, 64, 257);
        assert!(
            want.iter().all(|x| x.is_normal()),
            "control must stay in the normal range"
        );
        for (layout, got) in run_all_layouts(rt, &an, &bn, 2, 64, 257) {
            assert_eq!(bits(&got), bits(&want), "normal-range control {layout}");
        }
    });
}
