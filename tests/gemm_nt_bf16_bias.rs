//! `gemm_nt_bf16_bias`: the bf16 NT GEMM whose register epilogue adds an f32
//! per-column bias and optionally rounds to bf16, against the composition a
//! caller writes today, every piece its own dispatch on today's code:
//! `gemm_nt_bf16`, then `bert::bias_add` (one f32 add per element), then
//! `cast_f32_to_bf16_into` and `cast_bf16_to_f32_into`.
//!
//! The gate is bit equality, `to_bits()` on every element, with no tolerance:
//! the fusion must change when the arithmetic happens, not what it is. That
//! holds only if the fused kernel reduces K in the order of the plain NT
//! kernel, so the shapes include SAM's trunk linears (a long K, a wide N) and
//! ragged edges, where the bounds-checked tile path runs.
//!
//! The operands carry a zero row of A (so a row of C is the bias alone, which
//! lets the bias be any f32 bit pattern), a NaN, an Inf, a product near
//! bf16's range and biases around the bf16 overflow and f32 subnormal
//! boundaries, so the rounding step's corner cases are compared too.

mod common;

use std::sync::Arc;

use common::{random_f32, tensor_bf16, tensor_f32, with_gpu};
use tessl::bert::bias_add;
use tessl::gemm::{cast_bf16_to_f32_into, cast_f32_to_bf16_into, gemm_nt_bf16, gemm_nt_bf16_bias};
use tessl::tensor::Tensor;
use tessl::GpuRuntime;

fn read(rt: &Arc<GpuRuntime>, t: &Tensor) -> Vec<f32> {
    rt.synchronize().expect("synchronize");
    t.buffer.read_f32()[..t.numel()].to_vec()
}

/// Operands of one case: A `[m, k]`, B `[n, k]` (both stored bf16) and the bias.
struct Case {
    a: Tensor,
    b: Tensor,
    bias: Tensor,
}

fn case(rt: &Arc<GpuRuntime>, (m, n, k): (usize, usize, usize)) -> Case {
    let mut a_h = random_f32(m * k, 0xa11 ^ (m * 31 + k) as u64);
    let mut b_h = random_f32(n * k, 0xb22 ^ (n * 17 + k) as u64);
    let mut bias_h: Vec<f32> = random_f32(n, 0xc33 ^ n as u64).iter().map(|x| x * 40.0 + 3.0).collect();
    // A row of zeros: C[0, j] = bias[j] exactly, whatever bit pattern it is.
    // (0 * B is 0 unless B is Inf or NaN, which B never is here.)
    a_h[..k].fill(0.0);
    if m > 3 {
        a_h[k] = f32::NAN;
        a_h[2 * k] = f32::INFINITY;
        // 1e30 against a B column of 1e8 is a 1e38 product: a finite value
        // within a factor of 3.4 of bf16's largest.
        a_h[3 * k] = 1.0e30;
        for j in 0..n {
            b_h[j * k] = 1.0e8;
        }
    }
    let edge_biases: [f32; 14] = [
        3.4e38,                          // rounds up to Inf as a bf16
        3.39e38,                         // stays finite as a bf16
        -3.4e38,                         // rounds to -Inf
        f32::from_bits(0x0000_0001),     // smallest f32 subnormal
        f32::from_bits(0x007f_ffff),     // largest f32 subnormal
        f32::from_bits(0x8040_0000),     // a negative subnormal
        f32::from_bits(0x0080_0000),     // smallest normal
        -0.0,                            // a signed zero
        f32::NAN,                        // a NaN bias
        f32::from_bits(0x3f80_8000),     // 1 + 2^-8: a tie, rounds to even (1.0)
        f32::from_bits(0x3f81_8000),     // 1 + 3 * 2^-8: a tie, rounds up to even
        f32::from_bits(0x3f80_8001),     // just above a tie
        f32::INFINITY,                   // an infinite bias
        1.0e-30,                         // a small normal
    ];
    for (j, &v) in edge_biases.iter().enumerate().take(n) {
        bias_h[j] = v;
    }
    Case {
        a: tensor_bf16(rt, &[m, k], &a_h),
        b: tensor_bf16(rt, &[n, k], &b_h),
        bias: tensor_f32(rt, &[n], &bias_h),
    }
}

/// The three-step sequence, each step its own dispatch.
fn three_step(rt: &Arc<GpuRuntime>, c: &Case, (m, n): (usize, usize), round: bool) -> Vec<f32> {
    let out = rt.alloc_tensor_f32(&[m, n]).expect("alloc");
    gemm_nt_bf16(&c.a, &c.b, &out).expect("gemm_nt_bf16");
    bias_add(rt, &out.buffer, &c.bias.buffer, m as u32, n as u32).expect("bias_add");
    if round {
        let halves = rt.alloc_tensor_bf16(&[m, n]).expect("alloc bf16");
        cast_f32_to_bf16_into(&out, &halves).expect("cast to bf16");
        cast_bf16_to_f32_into(&halves, &out).expect("cast to f32");
    }
    read(rt, &out)
}

/// The fused kernel into a C that starts as a NaN pattern no result takes, so
/// an element the kernel never writes cannot match.
fn fused(rt: &Arc<GpuRuntime>, c: &Case, (m, n): (usize, usize), round: bool) -> Vec<f32> {
    let out = tensor_f32(rt, &[m, n], &vec![f32::from_bits(0x7fa5_a5a5); m * n]);
    gemm_nt_bf16_bias(&c.a, &c.b, &out, &c.bias, round).expect("gemm_nt_bf16_bias");
    read(rt, &out)
}

fn assert_bit_identical(tag: &str, n: usize, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{tag}: lengths");
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
    if let Some(&i) = bad.first() {
        panic!(
            "{tag}: {} of {} elements differ; first at [{}, {}]: fused {} ({:#010x}) vs three-step {} ({:#010x})",
            bad.len(),
            got.len(),
            i / n,
            i % n,
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

fn check(shape: (usize, usize, usize)) {
    with_gpu(|rt| {
        let (m, n, k) = shape;
        let c = case(rt, shape);
        for round in [false, true] {
            let tag = format!("{m}x{n}x{k} round_bf16={round}");
            let want = three_step(rt, &c, (m, n), round);
            let got = fused(rt, &c, (m, n), round);
            assert_bit_identical(&tag, n, &got, &want);
            // The comparison is not between two empty or all-NaN results. (The special
            // rows and columns are a few of a big C, but most of a 5-row one.)
            let finite = want.iter().filter(|x| x.is_finite()).count();
            assert!(
                m * n < 4096 || finite * 10 >= want.len() * 9,
                "{tag}: only {finite} finite elements of {}",
                want.len()
            );
            // Rounded outputs are bf16 values: the low 16 bits are zero.
            if round {
                assert!(
                    got.iter().all(|x| x.is_nan() || x.to_bits() & 0xffff == 0),
                    "{tag}: a rounded element carries f32 mantissa bits"
                );
            }
        }
        // The bias is applied: without the epilogue the result differs.
        let plain = rt.alloc_tensor_f32(&[m, n]).expect("alloc");
        gemm_nt_bf16(&c.a, &c.b, &plain).expect("gemm_nt_bf16");
        let plain = read(rt, &plain);
        let with_bias = fused(rt, &c, (m, n), false);
        assert!(
            plain.iter().zip(&with_bias).any(|(p, f)| p.to_bits() != f.to_bits()),
            "{m}x{n}x{k}: the bias changed nothing"
        );
    });
}

/// SAM's fc1: M = 5184 tokens, N = 4736, K = 1024.
#[test]
fn sam_fc1_matches_the_three_step_sequence_bit_for_bit() {
    check((5184, 4736, 1024));
}

/// SAM's fc2: the long K.
#[test]
fn sam_fc2_matches_the_three_step_sequence_bit_for_bit() {
    check((5184, 1024, 4736));
}

/// SAM's fused qkv.
#[test]
fn sam_qkv_matches_the_three_step_sequence_bit_for_bit() {
    check((5184, 3072, 1024));
}

/// Shapes whose M and N are not multiples of the 128x64 tile, so the edge
/// (bounds-checked) path runs, beside ones that fill whole tiles.
#[test]
fn ragged_and_tile_sized_shapes_match_the_three_step_sequence_bit_for_bit() {
    for shape in [
        (201, 256, 256),
        (61, 100, 96),
        (128, 64, 64),
        (130, 70, 40),
        (257, 129, 33),
        (5, 20, 16),
    ] {
        check(shape);
    }
}

/// A caller's mistakes fail at the boundary, naming the entry.
#[test]
fn refuses_a_bias_that_is_not_f32_n_long_or_disjoint_from_c() {
    with_gpu(|rt| {
        let (m, n, k) = (64, 64, 64);
        let c = case(rt, (m, n, k));
        let out = rt.alloc_tensor_f32(&[m, n]).expect("alloc");
        let refused = |bias: &Tensor, out: &Tensor| {
            gemm_nt_bf16_bias(&c.a, &c.b, out, bias, true).expect_err("must refuse")
        };
        let short = tensor_f32(rt, &[n - 1], &vec![0.0; n - 1]);
        assert!(refused(&short, &out).contains("exactly N = 64"), "short bias");
        let long = tensor_f32(rt, &[n + 1], &vec![0.0; n + 1]);
        assert!(refused(&long, &out).contains("exactly N = 64"), "long bias");
        let half = tensor_bf16(rt, &[n], &vec![0.0; n]);
        assert!(refused(&half, &out).contains("bias must be f32"), "bf16 bias");
        // A bias that is a slice of C itself.
        let aliased = out.view(&[n], 0);
        assert!(refused(&aliased, &out).contains("must not overlap"), "aliased bias");
        // C of the wrong shape.
        let wrong_c = rt.alloc_tensor_f32(&[m, n + 1]).expect("alloc");
        assert!(refused(&c.bias, &wrong_c).contains("GEMM"), "wrong C shape");
    });
}
