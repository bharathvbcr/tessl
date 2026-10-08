//! Runtime-identity and fixed-scratch contracts for raw NN buffers.
//!
//! `GpuBuffer` has no element type in its Rust type, so each public wrapper is
//! the boundary that must establish both capacity and allocator/runtime
//! identity. A buffer from another runtime can still be a valid Metal object;
//! accepting it would make ownership and synchronization depend on an
//! unrelated command queue.

mod common;

use std::sync::Arc;

use common::{empty, with_gpu, with_two_gpus};
use tessl::nn::{
    self, AttnDims, GateUpDispatch, KvStoreTarget, Q4Bank, Q4MlxBank, QkvBuffers, QkvRopeDims,
    QkvRopeVariant, QuantShape,
};
use tessl::{GpuBuffer, GpuRuntime};

fn assert_foreign_rejected(
    primary: &Arc<GpuRuntime>,
    foreign: &Arc<GpuRuntime>,
    label: &str,
    result: Result<(), String>,
) {
    let err = result.expect_err(label);
    assert!(
        err.contains("belongs to another runtime"),
        "{label}: unexpected error: {err}"
    );
    assert_eq!(
        primary.take_dispatch_count(),
        0,
        "{label}: rejection happened after a dispatch"
    );
    assert_eq!(
        foreign.take_dispatch_count(),
        0,
        "{label}: foreign runtime unexpectedly dispatched"
    );
}

fn u32_buf(rt: &Arc<GpuRuntime>, value: u32) -> GpuBuffer {
    let buffer = rt.alloc_buffer(4).expect("allocate u32 device scalar");
    buffer.write_u32(&[value]);
    buffer
}

#[test]
fn every_nn_buffer_family_rejects_foreign_runtime_storage_before_encode() {
    with_two_gpus(|primary, foreign_rt| {
        let local = empty(primary, 4096);
        let local_2 = empty(primary, 4096);
        let local_3 = empty(primary, 4096);
        let foreign = empty(foreign_rt, 4096);
        let scalar = u32_buf(primary, 0);

        assert_foreign_rejected(
            primary,
            foreign_rt,
            "normalization input",
            nn::rms_norm_f32(primary, &foreign, &local, &local_2, 1, 4, 1e-6),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "elementwise output",
            nn::mlp_silu(primary, &local, &local_2, &foreign, 4),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "in-place elementwise buffer",
            nn::scale_f32_inplace(primary, &foreign, 0.5, 4),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "quantized bank",
            nn::gemv_q8(
                primary, &foreign, &local, &local_2, &local_3, &local, 1, 32, 32,
            ),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "KV device offset",
            nn::kv_store_timestep(primary, &local, &local_2, &foreign, 4, 16),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "KV pair destination",
            nn::kv_store_timestep_pair(
                primary, &local, &local_2, &local_3, &foreign, &scalar, 4, 16,
            ),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "ring metadata",
            nn::kv_ring_densify(primary, &local, &local_2, &foreign, &scalar, 4, 4),
        );

        let attn_dims = AttnDims {
            batch: 1,
            tq: 1,
            heads: 1,
            heads_kv: 1,
            window: 0,
            scale: 0.125,
        };
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "attention K",
            nn::flash_attn_rows(
                primary, &local, &foreign, &local_2, &local_3, &scalar, &scalar, &scalar,
                attn_dims, 128, false,
            ),
        );

        let qkv_dims = QkvRopeDims {
            t: 1,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "fused QKV cache destination",
            nn::rms_qkv_rope(
                primary,
                QkvRopeVariant::PosBufferKvStore,
                QkvBuffers {
                    q: &local,
                    k: &local_2,
                    v: &local_3,
                    q_weight: &local,
                    k_weight: &local_2,
                    v_weight: &local_3,
                },
                qkv_dims,
                0,
                Some(&scalar),
                Some(KvStoreTarget {
                    dst_k: &local,
                    dst_v: &foreign,
                    dst_offset: &scalar,
                    capacity: 16,
                }),
                false,
            ),
        );

        assert_foreign_rejected(
            primary,
            foreign_rt,
            "sampling scalar",
            nn::softcap_logits(primary, &local, &foreign, 4),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "Q4 bank",
            nn::gemv_q4(
                primary,
                Q4Bank {
                    packed: &foreign,
                    scales: &local,
                    zeros: &local_2,
                },
                &local_3,
                &local,
                QuantShape {
                    rows: 1,
                    cols: 32,
                    group_size: 32,
                },
                false,
            ),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "row reduction output",
            nn::row_sum_f32(primary, &local, &foreign, 1, 4),
        );
        assert_foreign_rejected(
            primary,
            foreign_rt,
            "I8 matrix",
            nn::gemm_i8_dequant(primary, &local, &foreign, &local_2, 1, 1, 1, 1.0, None),
        );
    });
}

#[test]
fn blocked_gate_up_rejects_cols_beyond_its_fixed_x_cache() {
    with_gpu(|rt| {
        // This check must precede bank/input extent validation: allocating the
        // impossible matrix just to discover the shader has only 4096 floats
        // of threadgroup x-cache would defeat a host safety boundary.
        let tiny = empty(rt, 1);
        let bank = Q4MlxBank {
            packed: &tiny,
            scales_biases: &tiny,
        };
        let err = nn::gemv_q4_mlx_gate_up_gelu(
            rt,
            bank,
            bank,
            &tiny,
            &tiny,
            QuantShape {
                rows: 16,
                cols: 4096 + 32,
                group_size: 32,
            },
            GateUpDispatch::Blocked,
            false,
        )
        .expect_err("blocked x-cache overflow");
        assert!(
            err.contains("exceeds the blocked kernel x-cache capacity 4096"),
            "unexpected error: {err}"
        );
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn q_only_never_binds_inactive_foreign_buffers_and_rejects_cache_store_cleanly() {
    with_two_gpus(|primary, foreign_rt| {
        let q = empty(primary, 16);
        let q_weight = empty(primary, 16);
        let foreign = empty(foreign_rt, 16);
        let dims = QkvRopeDims {
            t: 1,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let inactive_foreign = QkvBuffers {
            q: &q,
            k: &foreign,
            v: &foreign,
            q_weight: &q_weight,
            k_weight: &foreign,
            v_weight: &foreign,
        };

        nn::rms_qkv_rope(
            primary,
            QkvRopeVariant::PosConst,
            inactive_foreign,
            dims,
            0,
            None,
            None,
            true,
        )
        .expect("inactive q_only placeholders must never reach Binder");
        primary.synchronize().expect("q_only completion");
        assert!(primary.take_dispatch_count() > 0);
        assert_eq!(foreign_rt.take_dispatch_count(), 0);

        // A cache-store shader checks its live cache offset before branching
        // to Q, so exposing it as q_only could silently turn the requested Q
        // transform into a no-op. Reject that contradictory combination
        // before any buffer validation or encoding, and prove the runtime is
        // still usable afterwards.
        let offset = u32_buf(primary, 0);
        let err = nn::rms_qkv_rope(
            primary,
            QkvRopeVariant::PosBufferKvStore,
            inactive_foreign,
            dims,
            0,
            Some(&offset),
            Some(KvStoreTarget {
                dst_k: &foreign,
                dst_v: &foreign,
                dst_offset: &foreign,
                capacity: 4,
            }),
            true,
        )
        .expect_err("q_only cache store must be rejected");
        assert!(
            err.contains("q_only cannot use PosBufferKvStore"),
            "unexpected error: {err}"
        );
        assert_eq!(primary.take_dispatch_count(), 0);
        nn::scale_f32_inplace(primary, &q, 1.0, 4)
            .expect("early q_only rejection must not poison the runtime");
        primary.synchronize().expect("post-rejection completion");
        assert!(primary.take_dispatch_count() > 0);
        assert_eq!(foreign_rt.take_dispatch_count(), 0);
    });
}

#[test]
fn raw_buffer_runtime_check_is_canonical_and_precedes_capacity() {
    let host = include_str!("../src/nn.rs");
    let require = host
        .split("fn require<T>")
        .nth(1)
        .and_then(|tail| tail.split("/// Reject unordered").next())
        .expect("canonical NN require helper");
    assert!(require.contains("require_runtime(rt, buf, what)?"));
    assert!(require.contains("require_capacity::<T>(buf, need, what)"));
    assert!(
        require.find("require_runtime").unwrap() < require.find("require_capacity").unwrap(),
        "ownership must fail before capacity inspection"
    );

    let blocked = include_str!("../kernels/gemv_q4_mlx.metal");
    assert!(blocked.contains("constant uint GEMV_X_TILE = 4096"));
    assert!(host.contains("shape.cols as usize > GEMV_X_TILE"));
}
