//! Buffer-alias contracts for the raw-buffer neural-network API.
//!
//! The rejection cases deliberately stop before encoding. Running the old
//! implementation against them would launch unordered reads/writes or overwrite
//! an operand the API promises to preserve. The regression therefore observes
//! the host boundary (error + zero dispatches) rather than attempting to detect
//! nondeterministic corruption after the fact.

mod common;

use std::sync::Arc;

use common::with_gpu;
use tessl::nn::{
    self, GateUpDispatch, KvStoreTarget, Q4Bank, Q4MlxBank, Q4MlxLayout, Q4MlxRowVariant,
    QkvBuffers, QkvOutputs, QkvRopeDims, QkvRopeVariant, QuantShape,
};
use tessl::{GpuBuffer, GpuRuntime};

fn empty(rt: &Arc<GpuRuntime>, bytes: usize) -> GpuBuffer {
    let buffer = rt.alloc_buffer(bytes).expect("allocate test buffer");
    buffer.zero();
    buffer
}

fn f32_buffer(rt: &Arc<GpuRuntime>, values: &[f32]) -> GpuBuffer {
    let buffer = empty(rt, std::mem::size_of_val(values));
    buffer.write_f32(values);
    buffer
}

fn assert_alias_refused(rt: &GpuRuntime, label: &str, result: Result<(), String>) {
    let err = result.expect_err("aliased output must be refused");
    assert!(
        err.contains("overlap"),
        "{label}: expected an overlap error, got {err:?}"
    );
    assert_eq!(
        rt.take_dispatch_count(),
        0,
        "{label}: encoded GPU work before refusing the alias"
    );
}

fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{label}[{i}]: got {got}, want {want}"
        );
    }
}

#[test]
fn rms_and_bf16_activation_reject_unordered_aliases() {
    with_gpu(|rt| {
        let x = empty(rt, 256);
        let weight = empty(rt, 256);
        let other = empty(rt, 256);

        assert_alias_refused(
            rt,
            "rms_norm_f32 out/weight",
            nn::rms_norm_f32(rt, &x, &weight, &weight, 2, 8, 1e-6),
        );
        assert_alias_refused(
            rt,
            "rms_norm_bf16 out/x",
            nn::rms_norm_bf16(rt, &x, &weight, &x, 2, 8, 1e-6),
        );
        assert_alias_refused(
            rt,
            "rms_norm_residual_add_f32 resid/weight",
            nn::rms_norm_residual_add_f32(rt, &x, &weight, &weight, 2, 8, 1e-6, 1.0),
        );
        assert_alias_refused(
            rt,
            "mlp_gelu_tanh_bf16 out/gate",
            nn::mlp_gelu_tanh_bf16(rt, &x, &other, &x, 16),
        );
    });
}

#[test]
fn documented_f32_in_place_norm_and_activation_paths_remain_correct() {
    with_gpu(|rt| {
        let input = [1.0f32, -2.0, 3.0, -4.0];
        let weight = [0.5f32, 1.0, 1.5, 2.0];
        let eps = 1e-6f32;
        let mean_square = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let inv = (mean_square + eps).sqrt().recip();
        let norm: Vec<_> = input
            .iter()
            .zip(weight)
            .map(|(&x, w)| x * inv * w)
            .collect();

        let x = f32_buffer(rt, &input);
        let w = f32_buffer(rt, &weight);
        nn::rms_norm_f32(rt, &x, &w, &x, 1, input.len() as u32, eps)
            .expect("rms_norm_f32 supports out == x");
        rt.synchronize().unwrap();
        assert_close("rms_norm_f32 in place", &x.read_f32(), &norm, 2e-5);

        let resid_x = f32_buffer(rt, &input);
        nn::rms_norm_residual_add_f32(rt, &resid_x, &w, &resid_x, 1, input.len() as u32, eps, 1.0)
            .expect("rms residual supports resid == x");
        rt.synchronize().unwrap();
        let want_resid: Vec<_> = input.iter().zip(&norm).map(|(x, n)| x + n).collect();
        assert_close(
            "rms_norm_residual_add_f32 in place",
            &resid_x.read_f32(),
            &want_resid,
            2e-5,
        );

        let gate_values = [-2.0f32, -0.5, 0.5, 2.0];
        let up_values = [0.25f32, 2.0, -3.0, 0.75];
        let want_silu: Vec<_> = gate_values
            .iter()
            .zip(up_values)
            .map(|(&gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        let gate = f32_buffer(rt, &gate_values);
        let up = f32_buffer(rt, &up_values);
        nn::mlp_silu(rt, &gate, &up, &gate, gate_values.len() as u32)
            .expect("mlp_silu supports out == gate");
        rt.synchronize().unwrap();
        assert_close("mlp_silu in place", &gate.read_f32(), &want_silu, 2e-5);

        let gate_values = [-1.5f32, -0.25, 0.75, 1.5];
        let up_values = [0.5f32, -2.0, 1.25, 3.0];
        let want_gelu: Vec<_> = gate_values
            .iter()
            .zip(up_values)
            .map(|(&x, up)| {
                let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh()) * up
            })
            .collect();
        let gate = f32_buffer(rt, &gate_values);
        let up = f32_buffer(rt, &up_values);
        nn::mlp_gelu_tanh(rt, &gate, &up, &up, gate_values.len() as u32)
            .expect("mlp_gelu_tanh supports out == up");
        rt.synchronize().unwrap();
        assert_close("mlp_gelu_tanh in place", &up.read_f32(), &want_gelu, 2e-5);
    });
}

#[test]
fn kv_cache_writes_reject_aliases_before_dispatch() {
    with_gpu(|rt| {
        let src_k = empty(rt, 256);
        let src_v = empty(rt, 256);
        let dst = empty(rt, 256);
        let offset = empty(rt, 4);

        assert_alias_refused(
            rt,
            "kv_store_timestep dst/src",
            nn::kv_store_timestep(rt, &src_k, &src_k, &offset, 8, 16),
        );
        assert_alias_refused(
            rt,
            "kv_store_timestep_pair destinations",
            nn::kv_store_timestep_pair(rt, &src_k, &src_v, &dst, &dst, &offset, 8, 16),
        );
        assert_alias_refused(
            rt,
            "kv_ring_densify dst/src",
            nn::kv_ring_densify(rt, &src_k, &src_k, &offset, &offset, 8, 4),
        );
    });
}

#[test]
fn sampling_outputs_reject_input_and_output_aliases() {
    with_gpu(|rt| {
        let logits = empty(rt, 256);
        let softcap = empty(rt, 4);
        let out = empty(rt, 256);

        assert_alias_refused(
            rt,
            "softcap_logits scalar/logits",
            nn::softcap_logits(rt, &logits, &logits, 16),
        );
        assert_alias_refused(
            rt,
            "argmax_f32_pass outputs",
            nn::argmax_f32_pass(rt, &logits, &out, &out, None, &softcap, 16),
        );
        assert_alias_refused(
            rt,
            "softcap_sample token/logits",
            nn::softcap_sample(rt, &logits, &logits, &softcap, 16),
        );
        assert_alias_refused(
            rt,
            "softcap_argmax_one_pass token/logits",
            nn::softcap_argmax_one_pass(rt, &logits, &logits, &softcap, 16),
        );
    });
}

#[test]
fn quantized_projections_and_embeddings_reject_output_aliases() {
    with_gpu(|rt| {
        let a = empty(rt, 4096);
        let b = empty(rt, 4096);
        let c = empty(rt, 4096);
        let shape = QuantShape {
            rows: 8,
            cols: 32,
            group_size: 32,
        };
        let q4 = Q4Bank {
            packed: &a,
            scales: &b,
            zeros: &b,
        };
        let mlx = Q4MlxBank {
            packed: &a,
            scales_biases: &b,
        };

        assert_alias_refused(
            rt,
            "gemv_q8 y/x",
            nn::gemv_q8(rt, &a, &b, &b, &c, &c, 8, 32, 32),
        );
        assert_alias_refused(
            rt,
            "gemv_q4 y/packed",
            nn::gemv_q4(rt, q4, &c, &a, shape, false),
        );
        assert_alias_refused(
            rt,
            "embed_lookup_q4 out/token_ids",
            nn::embed_lookup_q4(rt, q4, &c, &c, 8, 32, 32, 2),
        );
        assert_alias_refused(
            rt,
            "embed_lookup_q4_mlx out/packed",
            nn::embed_lookup_q4_mlx(rt, mlx, &c, &a, 8, 32, 32, 2),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx y/x",
            nn::gemv_q4_mlx(rt, mlx, &c, &c, shape, Q4MlxRowVariant::Standard),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx_blocked y/packed",
            nn::gemv_q4_mlx_blocked(rt, mlx, &c, &a, shape),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx_simd y/x",
            nn::gemv_q4_mlx_simd(rt, mlx, &c, &c, shape, Q4MlxLayout::RowMajor, None),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx_gate_up_gelu mid/x",
            nn::gemv_q4_mlx_gate_up_gelu(
                rt,
                mlx,
                mlx,
                &c,
                &c,
                shape,
                GateUpDispatch::Blocked,
                false,
            ),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx_kv outputs",
            nn::gemv_q4_mlx_kv(rt, mlx, mlx, &c, &c, &c, shape, Q4MlxLayout::RowMajor),
        );
        assert_alias_refused(
            rt,
            "gemv_q4_mlx_qkv outputs",
            nn::gemv_q4_mlx_qkv(
                rt,
                mlx,
                mlx,
                mlx,
                &c,
                QkvOutputs {
                    q_out: &c,
                    k_out: &c,
                    v_out: &c,
                },
                8,
                8,
                32,
                32,
                Q4MlxLayout::RowMajor,
            ),
        );
        assert_alias_refused(
            rt,
            "gemm_q4_mlx y/x",
            nn::gemm_q4_mlx(rt, mlx, &c, &c, shape, 2, Q4MlxLayout::RowMajor, None),
        );
        assert_alias_refused(
            rt,
            "gemm_i8_dequant c/a",
            nn::gemm_i8_dequant(rt, &c, &b, &c, 2, 2, 2, 1.0, None),
        );
    });
}

#[test]
fn fused_q4_residual_paths_preserve_the_supported_in_place_operation() {
    with_gpu(|rt| {
        let shape = QuantShape {
            rows: 8,
            cols: 32,
            group_size: 32,
        };
        let packed = empty(rt, shape.rows as usize * shape.cols as usize / 2);
        let scales_biases = empty(rt, shape.rows as usize * 4);
        let bank = Q4MlxBank {
            packed: &packed,
            scales_biases: &scales_biases,
        };
        let x = empty(rt, shape.cols as usize * 2);
        let residual: Vec<_> = (0..shape.rows).map(|i| i as f32 - 3.0).collect();
        let y = f32_buffer(rt, &residual);

        nn::gemv_q4_mlx_simd(rt, bank, &x, &y, shape, Q4MlxLayout::RowMajor, Some(&y))
            .expect("simd GEMV supports y == resid");
        rt.synchronize().unwrap();
        assert_close(
            "gemv_q4_mlx_simd in-place resid",
            &y.read_f32(),
            &residual,
            0.0,
        );

        let m = 2u32;
        let x = empty(rt, m as usize * shape.cols as usize * 2);
        let residual: Vec<_> = (0..m * shape.rows).map(|i| 0.25 * i as f32 - 1.0).collect();
        let y = f32_buffer(rt, &residual);
        nn::gemm_q4_mlx(rt, bank, &x, &y, shape, m, Q4MlxLayout::RowMajor, Some(&y))
            .expect("Q4 GEMM supports y == resid");
        rt.synchronize().unwrap();
        assert_close("gemm_q4_mlx in-place resid", &y.read_f32(), &residual, 0.0);
    });
}

#[test]
fn fused_qkv_rope_rejects_active_in_out_and_cache_aliases() {
    with_gpu(|rt| {
        let q = empty(rt, 256);
        let k = empty(rt, 256);
        let v = empty(rt, 256);
        let weight = empty(rt, 256);
        let other = empty(rt, 256);
        let offset = empty(rt, 4);
        let dims = QkvRopeDims {
            t: 1,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };

        assert_alias_refused(
            rt,
            "rms_qkv_rope q/k",
            nn::rms_qkv_rope(
                rt,
                QkvRopeVariant::PosConst,
                QkvBuffers {
                    q: &q,
                    k: &q,
                    v: &v,
                    q_weight: &weight,
                    k_weight: &weight,
                    v_weight: &weight,
                },
                dims,
                0,
                None,
                None,
                false,
            ),
        );
        assert_alias_refused(
            rt,
            "rms_qkv_rope cache/q",
            nn::rms_qkv_rope(
                rt,
                QkvRopeVariant::PosBufferKvStore,
                QkvBuffers {
                    q: &q,
                    k: &k,
                    v: &v,
                    q_weight: &weight,
                    k_weight: &weight,
                    v_weight: &weight,
                },
                dims,
                0,
                Some(&offset),
                Some(KvStoreTarget {
                    dst_k: &q,
                    dst_v: &other,
                    dst_offset: &offset,
                    capacity: 16,
                }),
                false,
            ),
        );

        // q_only dispatches no K/V work, so inactive aliases are not rejected.
        // Keeping this accepted prevents a conservative check from restricting
        // operands the selected grid provably never reaches.
        nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosConst,
            QkvBuffers {
                q: &q,
                k: &q,
                v: &q,
                q_weight: &weight,
                k_weight: &q,
                v_weight: &q,
            },
            dims,
            0,
            None,
            None,
            true,
        )
        .expect("q_only permits aliases among inactive K/V operands");
        rt.synchronize().unwrap();
    });
}
