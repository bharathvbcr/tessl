//! Fixed-order f32 GEMM against `gemm_f32_cpu`'s ascending-k left fold.
//!
//! The TensorOps 32×32 tile and the simdgroup tile match that fold on short K
//! and on integer products. They do not match it on order-sensitive products
//! whose K fills a hardware chunk (8×16×16). These kernels are the ones
//! `Numerics::Exact` dispatches, and the comparison is `to_bits`.
//!
//! The fold is bit-identical only while no operand, product or partial sum is an
//! f32 subnormal: the GPU flushes those to zero. See the `# Subnormals` section of
//! `gemm_fixed_order_nn`; `fixed_order_flushes_f32_subnormals_to_zero_on_this_gpu`
//! pins it.

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

// ---------------------------------------------------------------------------
// Adversarial extension (card ft-02eec315, lane w5-adv-ftz). The model under
// test is the one the `# Subnormals` doc states: the GPU flushes every f32
// subnormal operand, every subnormal product and every subnormal partial sum
// to zero, and is IEEE otherwise. These tests assert the model's exact bits
// over randomized and structured inputs on every layout, and assert on the
// host that the model differs from both the IEEE fold and an operand-only
// flush on the same inputs, so the pin would notice either change.
// ---------------------------------------------------------------------------

fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        // Sign-preserving: a flushed negative partial sum is -0.0 on this GPU.
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

/// Flush operands, every product and every partial sum (the documented model).
fn flush_fold(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let prod = ftz(ftz(a[i * k + p]) * ftz(b[p * n + j]));
                acc = ftz(acc + prod);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// IEEE arithmetic on operands that were flushed first (flush "only operands").
fn operand_flush_fold(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let fa: Vec<f32> = a.iter().map(|x| ftz(*x)).collect();
    let fb: Vec<f32> = b.iter().map(|x| ftz(*x)).collect();
    ieee_fold(&fa, &fb, m, n, k)
}

fn differing(x: &[f32], y: &[f32]) -> usize {
    x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
}

/// Asserts the GPU equals the documented flush model on every layout, and (when
/// `sensitive`) that the host models distinguish it from IEEE and operand-only.
fn assert_flush_model(
    rt: &std::sync::Arc<tessl::GpuRuntime>,
    label: &str,
    a: &[f32],
    b: &[f32],
    (m, n, k): (usize, usize, usize),
    sensitive: bool,
) {
    let model = flush_fold(a, b, m, n, k);
    if sensitive {
        let d_ieee = differing(&model, &ieee_fold(a, b, m, n, k));
        let d_op = differing(&model, &operand_flush_fold(a, b, m, n, k));
        assert!(d_ieee > 0, "{label}: construction does not separate flush from IEEE");
        assert!(
            d_op > 0,
            "{label}: construction does not separate full flush from operand-only flush"
        );
    }
    for (layout, got) in run_all_layouts(rt, a, b, m, n, k) {
        let bad = differing(&got, &model);
        if bad != 0 {
            let first = got
                .iter()
                .zip(&model)
                .position(|(g, w)| g.to_bits() != w.to_bits())
                .unwrap();
            panic!(
                "{label}: {layout}: {bad} of {} outputs differ from the flush model; first at {first}: gpu {:#010x} model {:#010x}",
                m * n,
                got[first].to_bits(),
                model[first].to_bits()
            );
        }
    }
}

impl Xorshift {
    /// Operand mix chosen so products and partial sums underflow often: normals in
    /// 2^-68 .. 2^-58 (products 2^-136 .. 2^-116), 15% subnormal (either sign),
    /// 5% signed zero, so subnormal operands, products and sums all occur.
    fn tiny_mix(&mut self) -> f32 {
        let sel = self.next() % 100;
        let r = self.next();
        let sign = (((r >> 40) & 1) as u32) << 31;
        match sel {
            0..=14 => f32::from_bits(sign | ((r & 0x7f_ffff) as u32).max(1)),
            15..=19 => f32::from_bits(sign),
            _ => {
                let exp = 127 - 68 + ((r >> 24) % 11) as u32;
                f32::from_bits(sign | exp << 23 | (r & 0x7f_ffff) as u32)
            }
        }
    }
}

#[test]
fn attack_random_mix_matches_flush_model_all_layouts() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let mut rng = Xorshift(0xdead_beef_cafe_f00d);
        // (m, n, k): odd extents that straddle 8/16/32 tiles and a 130-long K.
        for (m, n, k) in [
            (1usize, 1usize, 1usize),
            (3, 5, 7),
            (19, 37, 130),
            (33, 65, 33),
            (2, 3, 1025),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.tiny_mix()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.tiny_mix()).collect();
            let sensitive = m * n * k >= 1000;
            assert_flush_model(rt, &format!("random {m}x{n}x{k}"), &a, &b, (m, n, k), sensitive);
        }
    });
}

#[test]
fn attack_partial_sum_flush_with_normal_operands_and_products() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let sub = f32::from_bits;
        // Every operand and product is normal; only the running sum goes subnormal.
        // 0x00800001 + (-0x00800000) = 0x00000001 (exact, subnormal) -> flushed to 0.
        // A later +0x00800000 then shows whether the flushed 0 persisted:
        // IEEE gives 0x00800001, a flushed partial sum gives 0x00800000.
        type SumCase = (&'static str, Vec<f32>, Vec<f32>, u32, u32);
        let cases: [SumCase; 3] = [
            (
                "sum subnormal then end",
                vec![sub(0x0080_0001), sub(0x0080_0000)],
                vec![1.0, -1.0],
                0x0000_0001,
                0,
            ),
            (
                "flushed partial persists into a later add",
                vec![sub(0x0080_0001), sub(0x0080_0000), sub(0x0080_0000)],
                vec![1.0, -1.0, 1.0],
                0x0080_0001,
                0x0080_0000,
            ),
            (
                "negative subnormal partial sum flushes to -0.0 (sign kept)",
                vec![sub(0x0080_0000), sub(0x0080_0001)],
                vec![1.0, -1.0],
                0x8000_0001,
                0x8000_0000,
            ),
        ];
        for (label, a, b, ieee, gpu) in &cases {
            assert!(
                a.iter().chain(b.iter()).all(|x| x.is_normal()),
                "{label}: operands must be normal"
            );
            let k = a.len();
            let want = ieee_fold(a, b, 1, 1, k);
            assert_eq!(want[0].to_bits(), *ieee, "{label}: host fold");
            for (layout, got) in run_all_layouts(rt, a, b, 1, 1, k) {
                assert_eq!(got[0].to_bits(), *gpu, "{label}: {layout}");
            }
        }
    });
}

#[test]
fn attack_product_underflow_from_normal_operands_and_sign() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let p2 = |e: i32| f32::from_bits(((127 + e) as u32) << 23);
        // (label, a, b, IEEE bits, GPU bits). Every operand is normal.
        type ProdCase = (&'static str, f32, f32, u32, u32);
        let cases: [ProdCase; 6] = [
            (
                "2^-63 * 2^-63 = min normal survives",
                p2(-63),
                p2(-63),
                0x0080_0000,
                0x0080_0000,
            ),
            ("2^-63 * 2^-64 = 2^-127 flushes", p2(-63), p2(-64), 0x0040_0000, 0),
            (
                "2^-100 * 2^-49 = 2^-149 (smallest subnormal) flushes",
                p2(-100),
                p2(-49),
                0x0000_0001,
                0,
            ),
            (
                "negative product -2^-100 * 2^-30 flushes to +0",
                -p2(-100),
                p2(-30),
                0x8008_0000,
                0,
            ),
            (
                "both negative 2^-100 * 2^-30 flushes",
                -p2(-100),
                -p2(-30),
                0x0008_0000,
                0,
            ),
            (
                "product below the smallest subnormal rounds to 0 either way",
                p2(-100),
                p2(-60),
                0,
                0,
            ),
        ];
        for (label, a, b, ieee, gpu) in &cases {
            assert!(a.is_normal() && b.is_normal(), "{label}");
            let want = ieee_fold(&[*a], &[*b], 1, 1, 1);
            assert_eq!(want[0].to_bits(), *ieee, "{label}: host fold");
            for (layout, got) in run_all_layouts(rt, &[*a], &[*b], 1, 1, 1) {
                assert_eq!(got[0].to_bits(), *gpu, "{label}: {layout}");
            }
        }
    });
}

#[test]
fn attack_signed_zero_and_subnormal_operand_extremes_every_layout() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let sub = f32::from_bits;
        // Both operand slots, both signs, smallest / largest subnormal and -0.0.
        // No output can be -0.0 (the accumulator starts at +0 and +0 + -0 = +0),
        // so the documented "(positive) zero" is asserted over all of these.
        let vals = [
            0x0000_0001u32,
            0x007f_ffff,
            0x8000_0001,
            0x807f_ffff,
            0x8000_0000,
            0x0000_0000,
        ];
        for &v in &vals {
            for (label, a, b) in [("a", sub(v), 1.0f32), ("b", 1.0f32, sub(v)), ("neg-b", -1.0f32, sub(v))] {
                let want_gpu = 0u32; // flushed or zero: +0 in every case
                let want = ieee_fold(&[a], &[b], 1, 1, 1);
                // IEEE keeps the subnormal except for true zeros, where the fold gives +0 too.
                assert!(want[0].is_subnormal() || want[0].to_bits() == 0, "{label} {v:#010x}");
                for (layout, got) in run_all_layouts(rt, &[a], &[b], 1, 1, 1) {
                    assert_eq!(got[0].to_bits(), want_gpu, "{label} {v:#010x}: {layout}");
                }
            }
        }
    });
}

#[test]
fn attack_subnormal_at_k_positions_across_chunk_and_long_k() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let sub = f32::from_bits;
        // One output at (row i, col j) inside a grid, so a layout that mis-indexes
        // the subnormal's operand moves the flush to a different output.
        let (m, n) = (3usize, 5usize);
        for k in [15usize, 16, 17, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097, 70_000] {
            // Row 2 of A is zero except the planted entries; column 4 of B is 1.0
            // except the planted one; every other row and column is zero. Only
            // output (2, 4) is non-zero: sub(1) + sub(0x7fffff) - sub(1) in IEEE.
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; k * n];
            for p in 0..k {
                b[p * n + 4] = 1.0;
            }
            a[2 * k] = sub(1);
            a[2 * k + k / 2] = 1.0;
            b[(k / 2) * n + 4] = sub(0x007f_ffff);
            a[2 * k + (k - 1)] = sub(0x8000_0001);
            assert_flush_model(rt, &format!("k={k} planted subnormals"), &a, &b, (m, n, k), false);
            let model = flush_fold(&a, &b, m, n, k);
            assert!(
                model[2 * n + 4].to_bits() != ieee_fold(&a, &b, m, n, k)[2 * n + 4].to_bits(),
                "k={k}: planted subnormals must change the IEEE result"
            );

            // Late cancellation: sum goes subnormal at p = k-3 and a normal add follows.
            if k >= 8 {
                let mut a2 = vec![0.0f32; k];
                let mut b2 = vec![1.0f32; k];
                a2[1] = sub(0x0080_0001);
                a2[k - 3] = sub(0x0080_0000);
                b2[k - 3] = -1.0;
                a2[k - 1] = sub(0x0080_0000);
                assert_flush_model(rt, &format!("k={k} late partial-sum flush"), &a2, &b2, (1, 1, k), true);
            }
        }
    });
}

/// The doc says a flushed result "comes back as (positive) zero". The GPU
/// keeps the sign of a flushed partial sum: normal 0x00800000 + (-0x00800001)
/// gives -0.0 (0x80000000), observed in w5-adv-ftz-run1. Asserts the doc.
#[test]
#[ignore = "defect: gemm_fixed_order docs claim a flushed result is +0, but a negative subnormal partial sum flushes to -0.0 (0x80000000); see w5-adv-ftz-run1"]
fn defect_flushed_negative_partial_sum_is_positive_zero() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let a = [f32::from_bits(0x0080_0000), f32::from_bits(0x0080_0001)];
        for (layout, got) in run_all_layouts(rt, &a, &[1.0, -1.0], 1, 1, 2) {
            assert_eq!(got[0].to_bits(), 0, "{layout}: doc says +0");
        }
    });
}
