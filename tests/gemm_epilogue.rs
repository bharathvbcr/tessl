//! Fused GEMM epilogue: `C = act(alpha * A@B + beta * C_prev + bias)`.
//!
//! The gate is equivalence with the unfused sequence a caller writes today —
//! a GEMM, then a scale, then a bias add, then an activation, each its own
//! dispatch. Fusing must change *when* the arithmetic happens, not what it is.
//!
//! Bias is the part most likely to be subtly wrong. It reaches the kernel
//! through a row-stride-0 tensor view so that one `load` broadcasts a
//! per-column vector across every row of the tile. If that view is wrong the
//! result still looks like a plausible matrix, so the tests below check bias
//! against a reference that indexes it explicitly, at shapes that are not tile
//! multiples, where the edge path takes a different code branch.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::gemm::{gemm, gemm_epilogue, Activation, Epilogue, GemmBackend};
use tessl::tensor::Tensor;
use tessl::GpuRuntime;

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc");
    t.buffer.write_f32(data);
    t
}

/// The scalar terms of the fused expression, grouped so the reference below
/// takes seven arguments rather than eight.
#[derive(Clone, Copy)]
struct RefEpi {
    alpha: f32,
    beta: f32,
    act: Activation,
}

/// CPU reference for the whole fused expression.
fn reference(
    a: &[f32],
    b: &[f32],
    c_prev: &[f32],
    bias: Option<&[f32]>,
    (m, n, k): (usize, usize, usize),
    e: RefEpi,
) -> Vec<f32> {
    let (alpha, beta, act) = (e.alpha, e.beta, e.act);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            let mut v = alpha * acc + beta * c_prev[i * n + j];
            if let Some(bias) = bias {
                v += bias[j];
            }
            out[i * n + j] = match act {
                Activation::None => v,
                Activation::Relu => v.max(0.0),
                Activation::GeluTanh => {
                    let xc = v.clamp(-20.0, 20.0);
                    let inner =
                        0.7978845608028654f64 * (xc as f64 + 0.044715 * (xc as f64).powi(3));
                    (0.5 * xc as f64 * (1.0 + inner.clamp(-10.0, 10.0).tanh())) as f32
                }
                Activation::Silu => v / (1.0 + (-v).exp()),
            };
        }
    }
    out
}

/// Relaxed f32 is required for the cooperative path, and it is tf32-class: the
/// product carries far less mantissa than an f32 reference does.
/// Every activation here is non-expansive on the range these operands reach —
/// ReLU can only shrink, GELU and SiLU are contractive near zero and roughly
/// identity far from it — so the bound does not depend on which one ran.
fn tol(k: usize) -> f32 {
    2e-2 * (k as f32).sqrt()
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: worst |delta| = {worst} at {at} (got {} want {}), tol {tol}",
        got[at],
        want[at]
    );
}

const SHAPES: &[(usize, usize, usize)] = &[
    // Tile multiples (128x64 sg4) and deliberately not: the edge path is a
    // different branch, and the bias view is rebuilt differently there.
    (128, 64, 64),
    (256, 128, 128),
    (100, 70, 96),
    (37, 45, 64),
];

#[test]
fn fused_epilogue_matches_the_unfused_sequence() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        for &(m, n, k) in SHAPES {
            for act in [
                Activation::None,
                Activation::Relu,
                Activation::GeluTanh,
                Activation::Silu,
            ] {
                let a_h = random_f32(m * k, 0x11 + k as u64);
                let b_h = random_f32(k * n, 0x22 + n as u64);
                let c_h = random_f32(m * n, 0x33 + m as u64);
                let bias_h = random_f32(n, 0x44);

                let a = tensor(rt, &[m, k], &a_h);
                let b = tensor(rt, &[k, n], &b_h);
                let c = tensor(rt, &[m, n], &c_h);
                let bias = tensor(rt, &[n], &bias_h);

                let epi = Epilogue {
                    alpha: 0.75,
                    beta: 0.5,
                    bias: Some(&bias),
                    activation: act,
                };
                gemm_epilogue(&a, &b, &c, GemmBackend::TensorOps, epi).expect("gemm_epilogue");
                rt.synchronize().unwrap();

                let want = reference(
                    &a_h,
                    &b_h,
                    &c_h,
                    Some(&bias_h),
                    (m, n, k),
                    RefEpi {
                        alpha: 0.75,
                        beta: 0.5,
                        act,
                    },
                );
                close(
                    &format!("{m}x{n}x{k} act={act:?}"),
                    &c.buffer.read_f32()[..m * n],
                    &want,
                    tol(k),
                );
            }
        }
    });
}

#[test]
fn bias_is_per_column_not_per_row() {
    // The single likeliest way to get the stride-0 broadcast wrong is to
    // transpose it. A bias that varies only along N, checked on a
    // deliberately non-square shape, catches that: a per-row reading would
    // need a length-M vector and would produce visibly different columns.
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        // Two shapes on purpose. The tile is 128x64, so 96x48 has *no* interior
        // tile and exercises only the bounds-checked edge path, while 256x128
        // is all interior. The two build the bias view differently — an offset
        // pointer against a sliced tensor — and a transposed stride in one of
        // them is invisible to a test that only reaches the other. Found by
        // mutation: transposing the interior stride left this test green.
        for &(m, n, k) in &[(96usize, 48usize, 64usize), (256, 128, 64)] {
            // A and B zero, so C is exactly the bias: no product to hide behind.
            let a = tensor(rt, &[m, k], &vec![0.0f32; m * k]);
            let b = tensor(rt, &[k, n], &vec![0.0f32; k * n]);
            let c = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
            let bias_h: Vec<f32> = (0..n).map(|j| j as f32 + 1.0).collect();
            let bias = tensor(rt, &[n], &bias_h);

            gemm_epilogue(
                &a,
                &b,
                &c,
                GemmBackend::TensorOps,
                Epilogue {
                    bias: Some(&bias),
                    ..Default::default()
                },
            )
            .expect("gemm_epilogue");
            rt.synchronize().unwrap();

            let got = c.buffer.read_f32();
            for i in 0..m {
                for j in 0..n {
                    assert!(
                        (got[i * n + j] - bias_h[j]).abs() < 1e-4,
                        "bias broadcast wrong at ({i},{j}) for {m}x{n}: got {} want {}",
                        got[i * n + j],
                        bias_h[j]
                    );
                }
            }
        }
    });
}

#[test]
fn beta_zero_ignores_whatever_c_held() {
    // `beta == 0` skips reading C entirely, which is a bandwidth decision. If
    // it were implemented as a multiply by zero instead, a C holding NaN or
    // infinity would poison the result rather than being ignored.
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k) = (128usize, 64usize, 64usize);
        let a_h = random_f32(m * k, 0x55);
        let b_h = random_f32(k * n, 0x66);
        let a = tensor(rt, &[m, k], &a_h);
        let b = tensor(rt, &[k, n], &b_h);
        let c = tensor(rt, &[m, n], &vec![f32::NAN; m * n]);

        gemm_epilogue(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: None,
                activation: Activation::Relu,
            },
        )
        .expect("gemm_epilogue");
        rt.synchronize().unwrap();

        let got = c.buffer.read_f32();
        assert!(
            got[..m * n].iter().all(|v| v.is_finite()),
            "a NaN-filled C leaked through beta = 0"
        );
        let want = reference(
            &a_h,
            &b_h,
            &vec![0.0; m * n],
            None,
            (m, n, k),
            RefEpi {
                alpha: 1.0,
                beta: 0.0,
                act: Activation::Relu,
            },
        );
        close("beta=0", &got[..m * n], &want, tol(k));
    });
}

#[test]
fn an_identity_epilogue_equals_a_plain_gemm_bit_for_bit() {
    // The identity dispatches to `gemm`, so this is not an approximation: any
    // difference means the fast path stopped being taken.
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k) = (128usize, 64usize, 128usize);
        let a_h = random_f32(m * k, 0x77);
        let b_h = random_f32(k * n, 0x88);
        let a = tensor(rt, &[m, k], &a_h);
        let b = tensor(rt, &[k, n], &b_h);

        let plain = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
        gemm(&a, &b, &plain, GemmBackend::TensorOps).expect("gemm");
        let fused = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
        gemm_epilogue(&a, &b, &fused, GemmBackend::TensorOps, Epilogue::default())
            .expect("gemm_epilogue");
        rt.synchronize().unwrap();

        let (p, f) = (plain.buffer.read_f32(), fused.buffer.read_f32());
        for i in 0..m * n {
            assert_eq!(
                p[i].to_bits(),
                f[i].to_bits(),
                "identity epilogue differs at {i}"
            );
        }
    });
}

#[test]
fn the_epilogue_refuses_paths_with_nothing_to_fuse_into() {
    with_gpu(|rt| {
        let (m, n, k) = (64usize, 64usize, 64usize);
        let a = tensor(rt, &[m, k], &vec![1.0f32; m * k]);
        let b = tensor(rt, &[k, n], &vec![1.0f32; k * n]);
        let c = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
        let epi = Epilogue {
            activation: Activation::Relu,
            ..Default::default()
        };

        // Simdgroup writes C straight from the matmul: no register accumulator.
        let err = gemm_epilogue(&a, &b, &c, GemmBackend::Simdgroup, epi)
            .expect_err("simdgroup has nowhere to fuse");
        assert!(err.contains("cooperative-destination"), "{err}");

        // Exact f32 TensorOps likewise.
        rt.set_relaxed_precision(false);
        let err = gemm_epilogue(&a, &b, &c, GemmBackend::TensorOps, epi)
            .expect_err("exact f32 has nowhere to fuse");
        assert!(err.contains("cooperative-destination"), "{err}");
    });
}

#[test]
fn a_short_bias_is_refused() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k) = (128usize, 64usize, 64usize);
        let a = tensor(rt, &[m, k], &vec![1.0f32; m * k]);
        let b = tensor(rt, &[k, n], &vec![1.0f32; k * n]);
        let c = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
        // Per-column means length N. A length-M bias is the transposed
        // mistake, and on a non-square shape it is also the wrong size.
        let short = tensor(rt, &[n - 1], &vec![0.0f32; n - 1]);
        let err = gemm_epilogue(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                bias: Some(&short),
                ..Default::default()
            },
        )
        .expect_err("short bias");
        assert!(err.contains("per-column"), "{err}");

        let err = gemm_epilogue(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            Epilogue {
                alpha: f32::NAN,
                ..Default::default()
            },
        )
        .expect_err("non-finite alpha");
        assert!(err.contains("finite"), "{err}");
    });
}
