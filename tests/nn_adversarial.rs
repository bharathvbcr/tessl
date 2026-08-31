//! Hostile input against every `tessl::nn` entry point.
//!
//! These kernels index device memory with no bounds check of their own — that
//! is deliberate and documented, and it is exactly why the host-side
//! validation has to be right. A missed check here is not a wrong number, it
//! is a read or write of arbitrary device memory.
//!
//! So the assertion throughout is threefold: the call returns `Err`, it does
//! **not** panic, and it encodes nothing. The last is what separates "refused"
//! from "refused after already dispatching", and `take_dispatch_count` is what
//! makes it checkable.

mod common;

use std::sync::Arc;

use common::with_gpu;
use tessl::nn::{
    self, AttnDims, AttnHeadDim, Q4Bank, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant, QkvBuffers,
    QkvRopeDims, QkvRopeVariant, QuantShape,
};
use tessl::tensor::GpuBuffer;
use tessl::GpuRuntime;

/// A buffer far too small for anything the sweep asks of it.
fn tiny(rt: &Arc<GpuRuntime>) -> GpuBuffer {
    let b = rt.alloc_buffer(4).expect("alloc");
    b.zero();
    b
}

/// Generous, so a failure is the shape check firing and not an extent one.
fn roomy(rt: &Arc<GpuRuntime>) -> GpuBuffer {
    let b = rt.alloc_buffer(1 << 20).expect("alloc");
    b.zero();
    b
}

/// Every call below must refuse without encoding anything.
macro_rules! refuses {
    ($rt:expr, $what:expr, $call:expr) => {{
        let r = $call;
        assert!(r.is_err(), "{} was accepted; expected a refusal", $what);
        assert_eq!(
            $rt.take_dispatch_count(),
            0,
            "{} encoded work before refusing",
            $what
        );
    }};
}

#[test]
fn undersized_buffers_are_refused_across_the_whole_surface() {
    with_gpu(|rt| {
        let t = tiny(rt);
        let r = roomy(rt);

        // Norms and gated activations: `rows * dim` far exceeds `t`.
        refuses!(
            rt,
            "rms_norm_f32",
            nn::rms_norm_f32(rt, &t, &t, &t, 1024, 1024, 1e-6)
        );
        refuses!(
            rt,
            "rms_norm_bf16",
            nn::rms_norm_bf16(rt, &t, &t, &t, 1024, 1024, 1e-6)
        );
        refuses!(
            rt,
            "rms_norm_residual_add_f32",
            nn::rms_norm_residual_add_f32(rt, &t, &t, &t, 1024, 1024, 1e-6, 1.0)
        );
        refuses!(rt, "mlp_silu", nn::mlp_silu(rt, &t, &t, &t, 1 << 20));
        refuses!(
            rt,
            "mlp_gelu_tanh",
            nn::mlp_gelu_tanh(rt, &t, &t, &t, 1 << 20)
        );
        refuses!(
            rt,
            "mlp_gelu_tanh_bf16",
            nn::mlp_gelu_tanh_bf16(rt, &t, &t, &t, 1 << 20)
        );
        refuses!(
            rt,
            "scale_f32_inplace",
            nn::scale_f32_inplace(rt, &t, 2.0, 1 << 20)
        );

        // Quantized GEMV: the bank extents are derived from the shape.
        let shape = QuantShape {
            rows: 4096,
            cols: 4096,
            group_size: 32,
        };
        refuses!(
            rt,
            "gemv_q8",
            nn::gemv_q8(rt, &t, &t, &t, &t, &t, 4096, 4096, 32)
        );
        refuses!(
            rt,
            "gemv_q4",
            nn::gemv_q4(
                rt,
                Q4Bank {
                    packed: &t,
                    scales: &t,
                    zeros: &t
                },
                &t,
                &t,
                shape,
                false
            )
        );
        refuses!(
            rt,
            "gemv_q4_mlx",
            nn::gemv_q4_mlx(
                rt,
                Q4MlxBank {
                    packed: &t,
                    scales_biases: &t
                },
                &t,
                &t,
                shape,
                Q4MlxRowVariant::Standard
            )
        );
        refuses!(
            rt,
            "gemv_q4_mlx_simd",
            nn::gemv_q4_mlx_simd(
                rt,
                Q4MlxBank {
                    packed: &t,
                    scales_biases: &t
                },
                &t,
                &t,
                shape,
                Q4MlxLayout::RowMajor,
                None
            )
        );
        refuses!(
            rt,
            "gemm_q4_mlx",
            nn::gemm_q4_mlx(
                rt,
                Q4MlxBank {
                    packed: &t,
                    scales_biases: &t
                },
                &t,
                &t,
                shape,
                4,
                Q4MlxLayout::RowMajor,
                None
            )
        );

        // KV cache and embeddings.
        refuses!(
            rt,
            "kv_store_timestep",
            nn::kv_store_timestep(rt, &t, &t, &t, 1 << 20)
        );
        refuses!(
            rt,
            "kv_ring_densify",
            nn::kv_ring_densify(rt, &t, &t, &t, &t, 4096, 4096)
        );
        refuses!(
            rt,
            "embed_lookup_q4",
            nn::embed_lookup_q4(
                rt,
                Q4Bank {
                    packed: &t,
                    scales: &t,
                    zeros: &t
                },
                &t,
                &t,
                4096,
                4096,
                32,
                64
            )
        );

        // Sampling and reductions.
        refuses!(
            rt,
            "softmax_rows_f32",
            nn::softmax_rows_f32(rt, &t, &t, 1024, 1024)
        );
        refuses!(rt, "row_sum_f32", nn::row_sum_f32(rt, &t, &t, 1024, 1024));
        refuses!(rt, "row_max_f32", nn::row_max_f32(rt, &t, &t, 1024, 1024));
        refuses!(
            rt,
            "softcap_logits",
            nn::softcap_logits(rt, &t, &t, 1 << 20)
        );
        refuses!(
            rt,
            "argmax_f32_pass",
            nn::argmax_f32_pass(rt, &t, &t, &t, None, &t, 1 << 20)
        );
        refuses!(
            rt,
            "softcap_argmax_one_pass",
            nn::softcap_argmax_one_pass(rt, &t, &t, &t, 1 << 20)
        );
        refuses!(
            rt,
            "gemm_i8_dequant",
            nn::gemm_i8_dequant(rt, &t, &t, &t, 1024, 1024, 1024, 1.0, None)
        );

        // Attention: Q and O are checked against `B * Tq * H * D`.
        let dims = AttnDims {
            batch: 2,
            tq: 8,
            heads: 8,
            heads_kv: 2,
            window: 64,
            scale: 0.1,
        };
        refuses!(
            rt,
            "flash_attn_swa",
            nn::flash_attn_swa(rt, AttnHeadDim::D128, &t, &t, &t, &t, &r, &r, &r, dims)
        );
        refuses!(
            rt,
            "flash_attn_global_h512",
            nn::flash_attn_global_h512(rt, &t, &t, &t, &t, &r, &r, &r, dims, false)
        );

        // Fused QKV+RoPE.
        let qkv = QkvBuffers {
            q: &t,
            k: &t,
            v: &t,
            q_weight: &t,
            k_weight: &t,
            v_weight: &t,
        };
        let rope = QkvRopeDims {
            t: 64,
            heads_q: 32,
            heads_kv: 8,
            head_dim: 128,
            rotary_dim: 128,
            theta: 10_000.0,
            eps: 1e-6,
        };
        refuses!(
            rt,
            "rms_qkv_rope",
            nn::rms_qkv_rope(
                rt,
                QkvRopeVariant::PosConst,
                qkv,
                rope,
                0,
                None,
                None,
                false
            )
        );
    });
}

#[test]
fn zero_and_degenerate_dimensions_never_panic() {
    with_gpu(|rt| {
        let r = roomy(rt);
        // Zero work is either a clean no-op or a refusal — never a panic and
        // never a dispatch with a zero-sized grid, which Metal rejects.
        let outcomes: Vec<Result<(), String>> = vec![
            nn::mlp_silu(rt, &r, &r, &r, 0),
            nn::scale_f32_inplace(rt, &r, 1.0, 0),
            nn::rms_norm_f32(rt, &r, &r, &r, 0, 16, 1e-6),
            nn::softmax_rows_f32(rt, &r, &r, 0, 16),
            nn::row_sum_f32(rt, &r, &r, 0, 16),
            nn::gemv_q8(rt, &r, &r, &r, &r, &r, 0, 64, 32),
            nn::kv_store_timestep(rt, &r, &r, &r, 0),
        ];
        for (i, o) in outcomes.iter().enumerate() {
            assert!(o.is_ok(), "zero-work call {i} errored unexpectedly: {o:?}");
        }
        rt.synchronize()
            .expect("a zero-work sweep must leave the runtime usable");

        // Zero *columns* is different: there is no row to reduce, and silently
        // returning would report a result for work that cannot be defined.
        assert!(nn::softmax_rows_f32(rt, &r, &r, 4, 0).is_err());
        assert!(nn::row_max_f32(rt, &r, &r, 4, 0).is_err());
        assert!(nn::gemv_q8(rt, &r, &r, &r, &r, &r, 4, 64, 0).is_err());
    });
}

#[test]
fn non_finite_scalars_are_refused_rather_than_propagated() {
    with_gpu(|rt| {
        let r = roomy(rt);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Attention scale multiplies every logit; a NaN there silently
            // turns the whole output to NaN rather than failing.
            let dims = AttnDims {
                batch: 1,
                tq: 1,
                heads: 1,
                heads_kv: 1,
                window: 8,
                scale: bad,
            };
            refuses!(
                rt,
                "flash_attn_swa scale",
                nn::flash_attn_swa(rt, AttnHeadDim::D128, &r, &r, &r, &r, &r, &r, &r, dims)
            );
            refuses!(
                rt,
                "gemm_i8_dequant a_scale",
                nn::gemm_i8_dequant(rt, &r, &r, &r, 8, 8, 8, bad, None)
            );
            let rope = QkvRopeDims {
                t: 1,
                heads_q: 1,
                heads_kv: 1,
                head_dim: 8,
                rotary_dim: 8,
                theta: bad,
                eps: 1e-6,
            };
            let qkv = QkvBuffers {
                q: &r,
                k: &r,
                v: &r,
                q_weight: &r,
                k_weight: &r,
                v_weight: &r,
            };
            refuses!(
                rt,
                "rms_qkv_rope theta",
                nn::rms_qkv_rope(
                    rt,
                    QkvRopeVariant::PosConst,
                    qkv,
                    rope,
                    0,
                    None,
                    None,
                    false
                )
            );
        }
    });
}

#[test]
fn dimension_products_that_overflow_are_refused_not_wrapped() {
    with_gpu(|rt| {
        let r = roomy(rt);
        // `rows * dim` in usize. On a 64-bit host these do not overflow usize,
        // but they do exceed any buffer, so the extent check must fire rather
        // than the multiply wrapping into a small number that passes.
        refuses!(
            rt,
            "rms_norm_f32 huge",
            nn::rms_norm_f32(rt, &r, &r, &r, u32::MAX, u32::MAX, 1e-6)
        );
        refuses!(
            rt,
            "softmax huge",
            nn::softmax_rows_f32(rt, &r, &r, u32::MAX, u32::MAX)
        );
        refuses!(
            rt,
            "gemv_q8 huge",
            // The largest multiple of 32 below u32::MAX, so the ragged-group
            // check passes and the extent check is what has to fire.
            nn::gemv_q8(rt, &r, &r, &r, &r, &r, u32::MAX, u32::MAX - 31, 32)
        );
        refuses!(
            rt,
            "gemm_i8_dequant huge",
            nn::gemm_i8_dequant(rt, &r, &r, &r, u32::MAX, u32::MAX, 8, 1.0, None)
        );
    });
}

#[test]
fn the_runtime_is_still_usable_after_the_whole_hostile_sweep() {
    // The point of refusing before encoding: a rejected call must leave no
    // partial state behind. If any of the above had half-encoded, this would
    // fail or hang rather than passing.
    with_gpu(|rt| {
        let t = tiny(rt);
        let _ = nn::rms_norm_f32(rt, &t, &t, &t, 4096, 4096, 1e-6);
        let _ = nn::softmax_rows_f32(rt, &t, &t, 4096, 4096);
        let _ = nn::gemm_i8_dequant(rt, &t, &t, &t, 4096, 4096, 4096, 1.0, None);

        let n = 256usize;
        let x = rt.alloc_buffer(n * 4).unwrap();
        x.write_f32(&vec![2.0f32; n]);
        let out = rt.alloc_buffer(n * 4).unwrap();
        out.zero();
        nn::scale_f32_inplace(rt, &x, 0.5, n as u32).expect("real work after the sweep");
        nn::mlp_silu(rt, &x, &x, &out, n as u32).expect("real work after the sweep");
        rt.synchronize().expect("synchronize after the sweep");
        assert!(out.read_f32()[..n].iter().all(|v| v.is_finite()));
    });
}
