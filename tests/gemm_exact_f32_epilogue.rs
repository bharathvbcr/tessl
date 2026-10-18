//! Fused GEMM epilogue on exact-f32 operands: `C = act(alpha * A@B + beta * C_prev + bias)`.
//!
//! The gate is bit equality with the composition a caller writes today, every
//! piece its own dispatch on today's code: the exact-f32 [`gemm`], then
//! `nn::scale_f32_inplace` for alpha, the `beta * C_prev` accumulate (host f32
//! arithmetic between two dispatches, which is what separate passes compute:
//! each product and sum rounded on its own), `bert::bias_add`, and the
//! activation through the kernel `nn::mlp_*` runs with an all-ones `up`
//! (`x * 1` is exact, so it is the activation function body alone).
//!
//! Fusing must change when the arithmetic happens, not what it is, so there is
//! no tolerance here. Exact f32 means exact f32 products: this runs with
//! relaxed precision off, where the relaxed (tf32-class) epilogue is not
//! reachable at all.
//!
//! Shapes keep K small enough that `gemm` takes its single-dispatch kernel:
//! the K-split routes sum in another order by design.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::bert::bias_add;
use tessl::gemm::{gemm, gemm_epilogue, Activation, Epilogue, GemmBackend};
use tessl::nn::{mlp_gelu_tanh, mlp_silu, scale_f32_inplace};
use tessl::tensor::Tensor;
use tessl::GpuRuntime;

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc");
    t.buffer.write_f32(data);
    t
}

fn read(rt: &Arc<GpuRuntime>, t: &Tensor, n: usize) -> Vec<f32> {
    rt.synchronize().expect("synchronize");
    t.buffer.read_f32()[..n].to_vec()
}

/// The separate-dispatch composition of the epilogue, on today's ops.
fn unfused(
    rt: &Arc<GpuRuntime>,
    (a, b): (&Tensor, &Tensor),
    c_prev: &[f32],
    bias: Option<&Tensor>,
    (m, n): (usize, usize),
    (alpha, beta, act): (f32, f32, Activation),
) -> Vec<f32> {
    let mn = m * n;
    let t = rt.alloc_tensor_f32(&[m, n]).expect("alloc");
    gemm(a, b, &t, GemmBackend::TensorOps).expect("exact gemm");
    // The kernel scales only when alpha != 1 or beta != 0, and when beta != 0
    // it always forms alpha * p + beta * prev.
    if alpha != 1.0 {
        scale_f32_inplace(rt, &t.buffer, alpha, mn as u32).expect("scale");
    }
    if beta != 0.0 {
        let mut v = read(rt, &t, mn);
        for (x, &p) in v.iter_mut().zip(c_prev) {
            let bp = beta * p;
            *x += bp;
        }
        t.buffer.write_f32(&v);
    }
    if let Some(bias) = bias {
        bias_add(rt, &t.buffer, &bias.buffer, m as u32, n as u32).expect("bias_add");
    }
    match act {
        Activation::None => {}
        Activation::Relu => {
            let v: Vec<f32> = read(rt, &t, mn).into_iter().map(|x| x.max(0.0)).collect();
            t.buffer.write_f32(&v);
        }
        Activation::GeluTanh | Activation::Silu => {
            let ones = tensor(rt, &[m, n], &vec![1.0f32; mn]);
            let out = rt.alloc_tensor_f32(&[m, n]).expect("alloc");
            let f = if act == Activation::Silu {
                mlp_silu
            } else {
                mlp_gelu_tanh
            };
            f(rt, &t.buffer, &ones.buffer, &out.buffer, mn as u32).expect("activation pass");
            return read(rt, &out, mn);
        }
    }
    read(rt, &t, mn)
}

const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (31, 33, 17),
    (64, 64, 64),
    (100, 37, 129),
    // Whole 128x64 and 64x64 tiles, and an interior block beside an edge.
    (128, 64, 64),
    (130, 70, 40),
];

const SCALARS: &[(f32, f32)] = &[
    (1.0, 0.0),
    (0.75, 0.0),
    (1.0, 1.0),
    (0.75, 0.5),
    (1.0, 0.5),
    (-1.5, -0.25),
    (0.7, 0.3),
    (-0.9, 1.1),
];

#[test]
fn exact_f32_fused_epilogue_is_bit_identical_to_the_separate_passes() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let acts = [
            Activation::None,
            Activation::Relu,
            Activation::GeluTanh,
            Activation::Silu,
        ];
        let mut cases = 0usize;
        for &(m, n, k) in SHAPES {
            let a_h = random_f32(m * k, 0x11 + k as u64);
            let b_h = random_f32(k * n, 0x22 + n as u64);
            let c_h = random_f32(m * n, 0x33 + m as u64);
            let bias_h = random_f32(n, 0x44);
            let a = tensor(rt, &[m, k], &a_h);
            let b = tensor(rt, &[k, n], &b_h);
            let bias = tensor(rt, &[n], &bias_h);
            for &(alpha, beta) in SCALARS {
                for &act in &acts {
                    for with_bias in [false, true] {
                        let bias_t = with_bias.then_some(&bias);
                        // C_prev is NaN where the kernel must not read it.
                        let init = if beta == 0.0 {
                            vec![f32::NAN; m * n]
                        } else {
                            c_h.clone()
                        };
                        let c = tensor(rt, &[m, n], &init);
                        let epi = Epilogue {
                            alpha,
                            beta,
                            bias: bias_t,
                            activation: act,
                        };
                        let tag = format!("{m}x{n}x{k} alpha={alpha} beta={beta} bias={with_bias} act={act:?}");
                        gemm_epilogue(&a, &b, &c, GemmBackend::TensorOps, epi)
                            .unwrap_or_else(|e| panic!("{tag}: gemm_epilogue refused: {e}"));
                        let got = read(rt, &c, m * n);
                        let want = unfused(rt, (&a, &b), &c_h, bias_t, (m, n), (alpha, beta, act));
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert_eq!(
                                g.to_bits(),
                                w.to_bits(),
                                "{tag}: element {i} fused {g} ({:#010x}) vs separate {w} ({:#010x})",
                                g.to_bits(),
                                w.to_bits()
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, SHAPES.len() * SCALARS.len() * acts.len() * 2);
    });
}

/// Relaxed precision on still takes the relaxed epilogue, whose products are
/// tf32-class: the exact route must be chosen by the setting, not by the call.
#[test]
fn relaxed_setting_keeps_the_relaxed_epilogue() {
    with_gpu(|rt| {
        let (m, n, k) = (64, 64, 64);
        let a_h = random_f32(m * k, 5);
        let b_h = random_f32(k * n, 6);
        let a = tensor(rt, &[m, k], &a_h);
        let b = tensor(rt, &[k, n], &b_h);
        let epi = Epilogue {
            alpha: 0.5,
            activation: Activation::Relu,
            ..Epilogue::default()
        };
        let run = |relaxed: bool| {
            rt.set_relaxed_precision(relaxed);
            let c = tensor(rt, &[m, n], &vec![f32::NAN; m * n]);
            gemm_epilogue(&a, &b, &c, GemmBackend::TensorOps, epi).expect("epilogue");
            read(rt, &c, m * n)
        };
        let (exact, relaxed) = (run(false), run(true));
        assert!(
            exact.iter().zip(&relaxed).any(|(x, y)| x.to_bits() != y.to_bits()),
            "relaxed and exact epilogues produced identical bits on 64x64x64: the setting is not selecting a kernel"
        );
    });
}
