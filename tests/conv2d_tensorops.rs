//! `conv::conv2d_nchw` (the matrix-unit convolution) against an f64 CPU reference.
//!
//! The shapes are the ones the op's probe found fragile: widths that are not a whole number of
//! 32-wide tiles (the op computes wrong values in a partial tile, so the host pads), a channel
//! count below the 64-channel tile, input channels that are not a multiple of the 32-wide pack
//! tile, batches above one, and padding other than 1. Scratch is allocated poisoned, so an element
//! the kernels fail to write reads as NaN.

mod common;

use common::with_gpu;
use tessl::conv::{conv2d_nchw, conv2d_tensorops_supports, ConvPrecision};
use tessl::runtime::GpuRuntime;
use tessl::tensor::Tensor;

use std::sync::Arc;

struct Case {
    n: usize,
    ci: usize,
    co: usize,
    h: usize,
    w: usize,
    pad: usize,
    bias: bool,
}

fn rng(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

fn upload(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).unwrap();
    t.write_f32(data).unwrap();
    t
}

/// f64 reference: PyTorch's conv2d, stride 1, 3x3.
fn reference(c: &Case, x: &[f32], wt: &[f32], b: &[f32]) -> Vec<f64> {
    let (oh, ow) = (c.h + 2 * c.pad - 2, c.w + 2 * c.pad - 2);
    let mut y = vec![0f64; c.n * c.co * oh * ow];
    for n in 0..c.n {
        for o in 0..c.co {
            for i in 0..oh {
                for j in 0..ow {
                    let mut s = if c.bias { b[o] as f64 } else { 0.0 };
                    for ch in 0..c.ci {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let (hh, ww) = ((i + kh) as isize - c.pad as isize, (j + kw) as isize - c.pad as isize);
                                if hh < 0 || ww < 0 || hh >= c.h as isize || ww >= c.w as isize {
                                    continue;
                                }
                                let xv = x[((n * c.ci + ch) * c.h + hh as usize) * c.w + ww as usize];
                                s += xv as f64 * wt[((o * c.ci + ch) * 3 + kh) * 3 + kw] as f64;
                            }
                        }
                    }
                    y[((n * c.co + o) * oh + i) * ow + j] = s;
                }
            }
        }
    }
    y
}

/// Runs one case in `precision` and returns (relative L2 error, max abs error, count of non-finite).
fn run(rt: &Arc<GpuRuntime>, c: &Case, precision: ConvPrecision) -> (f64, f64, usize) {
    let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ (c.ci * 131 + c.co * 7 + c.h) as u64;
    let x: Vec<f32> = (0..c.n * c.ci * c.h * c.w).map(|_| rng(&mut seed)).collect();
    let wt: Vec<f32> = (0..c.co * c.ci * 9).map(|_| rng(&mut seed) * 0.05).collect();
    let b: Vec<f32> = (0..c.co).map(|_| rng(&mut seed)).collect();
    let (oh, ow) = (c.h + 2 * c.pad - 2, c.w + 2 * c.pad - 2);
    rt.set_poison_unzeroed(true);
    let xt = upload(rt, &[c.n, c.ci, c.h, c.w], &x);
    let wt_t = upload(rt, &[c.co, c.ci, 3, 3], &wt);
    let bt = upload(rt, &[c.co], &b);
    let yt = rt.alloc_tensor_f32(&[c.n, c.co, oh, ow]).unwrap();
    yt.write_f32(&vec![f32::NAN; c.n * c.co * oh * ow]).unwrap();
    conv2d_nchw(&xt, &wt_t, c.bias.then_some(&bt), &yt, c.pad, precision).unwrap();
    rt.synchronize().unwrap();
    rt.set_poison_unzeroed(false);
    let got = yt.read_f32().unwrap();
    let want = reference(c, &x, &wt, &b);
    let (mut err, mut norm, mut max_abs, mut bad) = (0f64, 0f64, 0f64, 0usize);
    for (g, r) in got.iter().zip(&want) {
        if !g.is_finite() {
            bad += 1;
            continue;
        }
        let d = *g as f64 - r;
        err += d * d;
        norm += r * r;
        max_abs = max_abs.max(d.abs());
    }
    ((err / norm).sqrt(), max_abs, bad)
}

fn check(c: Case) {
    with_gpu(|rt| {
        for (precision, tol) in [(ConvPrecision::F32, 1e-5), (ConvPrecision::F16, 2e-3)] {
            let (rel, max_abs, bad) = run(rt, &c, precision);
            assert_eq!(
                bad, 0,
                "{precision:?} n{} ci{} co{} {}x{} pad{}: {bad} outputs unwritten or non-finite",
                c.n, c.ci, c.co, c.h, c.w, c.pad
            );
            assert!(
                rel < tol,
                "{precision:?} n{} ci{} co{} {}x{} pad{} bias {}: relative L2 {rel:.3e} (max abs {max_abs:.3e}) over {tol:e}",
                c.n,
                c.ci,
                c.co,
                c.h,
                c.w,
                c.pad,
                c.bias
            );
        }
    });
}

#[test]
fn partial_width_tiles_match_the_reference() {
    // 40 = one full 32-wide tile and 8 columns of the next: the probe's f16 failure.
    check(Case {
        n: 1,
        ci: 256,
        co: 64,
        h: 40,
        w: 40,
        pad: 1,
        bias: true,
    });
}

#[test]
fn narrow_output_batched_odd_sizes_match_the_reference() {
    // c_out 16 takes the 16-channel tile; 45 wide and 37 high are neither tile multiples; n = 2.
    check(Case {
        n: 2,
        ci: 32,
        co: 16,
        h: 37,
        w: 45,
        pad: 1,
        bias: false,
    });
}

#[test]
fn input_channels_off_the_pack_tile_match_the_reference() {
    check(Case {
        n: 1,
        ci: 20,
        co: 48,
        h: 19,
        w: 33,
        pad: 1,
        bias: true,
    });
}

#[test]
fn padding_zero_and_two_match_the_reference() {
    check(Case {
        n: 1,
        ci: 64,
        co: 64,
        h: 24,
        w: 30,
        pad: 0,
        bias: true,
    });
    check(Case {
        n: 1,
        ci: 64,
        co: 128,
        h: 17,
        w: 17,
        pad: 2,
        bias: false,
    });
}

#[test]
fn scope_and_refusals() {
    assert!(conv2d_tensorops_supports(256, 256, 3, 3, (1, 1), (1, 1), 1));
    assert!(
        !conv2d_tensorops_supports(256, 24, 3, 3, (1, 1), (1, 1), 1),
        "c_out not a multiple of 16"
    );
    assert!(!conv2d_tensorops_supports(256, 256, 5, 5, (1, 1), (1, 1), 1), "5x5");
    assert!(
        !conv2d_tensorops_supports(256, 256, 3, 3, (2, 2), (1, 1), 1),
        "stride 2"
    );
    assert!(
        !conv2d_tensorops_supports(256, 256, 3, 3, (1, 1), (2, 2), 1),
        "dilation 2"
    );
    assert!(
        !conv2d_tensorops_supports(256, 256, 3, 3, (1, 1), (1, 1), 256),
        "depthwise"
    );
    with_gpu(|rt| {
        let x = rt.alloc_tensor_f32(&[1, 8, 6, 6]).unwrap();
        let w = rt.alloc_tensor_f32(&[16, 8, 3, 3]).unwrap();
        let w24 = rt.alloc_tensor_f32(&[24, 8, 3, 3]).unwrap();
        let y = rt.alloc_tensor_f32(&[1, 16, 6, 6]).unwrap();
        let y_bad = rt.alloc_tensor_f32(&[1, 16, 4, 4]).unwrap();
        let bias_bad = rt.alloc_tensor_f32(&[8]).unwrap();
        let f = ConvPrecision::F16;
        assert!(conv2d_nchw(&x, &w, None, &y, 1, f).is_ok());
        assert!(conv2d_nchw(&x, &w24, None, &y, 1, f).unwrap_err().contains("scope"));
        assert!(conv2d_nchw(&x, &w, None, &y_bad, 1, f)
            .unwrap_err()
            .contains("y must be"));
        assert!(conv2d_nchw(&x, &w, Some(&bias_bad), &y, 1, f)
            .unwrap_err()
            .contains("bias must be"));
        assert!(conv2d_nchw(&x, &w, None, &x, 1, f).is_err(), "y aliasing x");
        rt.synchronize().unwrap();
    });
}
