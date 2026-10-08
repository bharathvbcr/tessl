//! Adversarial contracts for device-controlled KV state.
//!
//! The offset, ring cursor, filled count, and RoPE position are deliberately
//! device buffers so an encode-once/ICB path can mutate them between replays.
//! That also means the host cannot trust their values. These tests keep guard
//! regions around every logical cache and exercise values that used to wrap
//! `uint` arithmetic into apparently valid device addresses.

mod common;

use std::sync::Arc;

use common::{buf, seeded, with_gpu};
use tessl::nn::{self, KvStoreTarget, QkvBuffers, QkvRopeDims, QkvRopeVariant};
use tessl::{GpuBuffer, GpuRuntime};

const SENTINEL: f32 = -7.25e27;

fn u32_buf(rt: &Arc<GpuRuntime>, value: u32) -> GpuBuffer {
    let buffer = rt.alloc_buffer(4).expect("allocate u32 device scalar");
    buffer.write_u32(&[value]);
    buffer
}

fn assert_all_sentinel(label: &str, buffer: &GpuBuffer, len: usize) {
    let got = buffer.read_f32();
    assert!(
        got[..len].iter().all(|&value| value == SENTINEL),
        "{label}: invalid device state changed the destination: {:?}",
        &got[..len]
    );
}

fn assert_guard(label: &str, got: &[f32], range: std::ops::Range<usize>) {
    assert!(
        got[range.clone()].iter().all(|&value| value == SENTINEL),
        "{label}: guard {range:?} was modified: {:?}",
        &got[range.clone()]
    );
}

#[test]
fn timestep_store_device_offsets_cannot_cross_logical_capacity() {
    with_gpu(|rt| {
        let src = [1.25, -2.5, 3.75];
        let src_k = buf(rt, &src);
        let src_v = buf(rt, &[-1.0, 4.0, 9.0]);
        let capacity = 8u32;
        let physical = 12usize;

        // offset == capacity, offset > capacity, a non-wrapping span that
        // crosses the end, and the u32 wrapping boundary must all be complete
        // no-ops. A per-thread bound would allow a partial store here.
        for offset in [
            capacity,
            capacity + 1,
            capacity - src.len() as u32 + 1,
            u32::MAX,
        ] {
            let off = u32_buf(rt, offset);
            let dst = seeded(rt, physical, SENTINEL);
            nn::kv_store_timestep(rt, &src_k, &dst, &off, src.len() as u32, capacity).unwrap();
            rt.synchronize().unwrap();
            assert_all_sentinel(&format!("single offset={offset}"), &dst, physical);

            let dst_k = seeded(rt, physical, SENTINEL);
            let dst_v = seeded(rt, physical, SENTINEL);
            nn::kv_store_timestep_pair(
                rt,
                &src_k,
                &src_v,
                &dst_k,
                &dst_v,
                &off,
                src.len() as u32,
                capacity,
            )
            .unwrap();
            rt.synchronize().unwrap();
            assert_all_sentinel(&format!("pair K offset={offset}"), &dst_k, physical);
            assert_all_sentinel(&format!("pair V offset={offset}"), &dst_v, physical);
        }

        // The last exactly fitting span remains valid, while the physical slab
        // beyond the declared suballocation is still a canary.
        let valid_offset = capacity - src.len() as u32;
        let off = u32_buf(rt, valid_offset);
        let dst_k = seeded(rt, physical, SENTINEL);
        let dst_v = seeded(rt, physical, SENTINEL);
        nn::kv_store_timestep_pair(
            rt,
            &src_k,
            &src_v,
            &dst_k,
            &dst_v,
            &off,
            src.len() as u32,
            capacity,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let got_k = dst_k.read_f32();
        let got_v = dst_v.read_f32();
        let start = valid_offset as usize;
        assert_eq!(&got_k[start..start + src.len()], &src);
        assert_eq!(&got_v[start..start + src.len()], &[-1.0, 4.0, 9.0]);
        assert_guard("valid pair K prefix", &got_k, 0..start);
        assert_guard("valid pair V prefix", &got_v, 0..start);
        assert_guard(
            "valid pair K slab tail",
            &got_k,
            capacity as usize..physical,
        );
        assert_guard(
            "valid pair V slab tail",
            &got_v,
            capacity as usize..physical,
        );
    });
}

#[test]
fn fused_multi_token_store_is_atomic_at_every_offset_boundary() {
    with_gpu(|rt| {
        let dims = QkvRopeDims {
            t: 2,
            heads_q: 2,
            heads_kv: 2,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let q_values: Vec<f32> = (1..=16).map(|i| i as f32 * 0.125).collect();
        let k_values: Vec<f32> = (1..=16).map(|i| -(i as f32) * 0.25).collect();
        let v_values: Vec<f32> = (1..=16).map(|i| i as f32 * 0.375).collect();
        let weight = buf(rt, &[1.0, 0.75, 1.25, 0.5]);
        let pos = u32_buf(rt, 11);
        let span = 16u32; // T * Hkv * D; intentionally more than one token.
        let capacity = 24u32;
        let physical = 32usize;

        for offset in [capacity, capacity + 1, capacity - span + 1, u32::MAX] {
            let q = buf(rt, &q_values);
            let k = buf(rt, &k_values);
            let v = buf(rt, &v_values);
            let dst_k = seeded(rt, physical, SENTINEL);
            let dst_v = seeded(rt, physical, SENTINEL);
            let off = u32_buf(rt, offset);

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
                Some(&pos),
                Some(KvStoreTarget {
                    dst_k: &dst_k,
                    dst_v: &dst_v,
                    dst_offset: &off,
                    capacity,
                }),
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();

            assert_eq!(
                q.read_f32()[..q_values.len()],
                q_values,
                "Q offset={offset}"
            );
            assert_eq!(
                k.read_f32()[..k_values.len()],
                k_values,
                "K offset={offset}"
            );
            assert_eq!(
                v.read_f32()[..v_values.len()],
                v_values,
                "V offset={offset}"
            );
            assert_all_sentinel(&format!("fused K offset={offset}"), &dst_k, physical);
            assert_all_sentinel(&format!("fused V offset={offset}"), &dst_v, physical);
        }

        // Exact-fit at the logical tail must still execute and must not touch
        // either the prefix or the physical slab beyond logical capacity.
        let q = buf(rt, &q_values);
        let k = buf(rt, &k_values);
        let v = buf(rt, &v_values);
        let dst_k = seeded(rt, physical, SENTINEL);
        let dst_v = seeded(rt, physical, SENTINEL);
        let offset = capacity - span;
        let off = u32_buf(rt, offset);
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
            Some(&pos),
            Some(KvStoreTarget {
                dst_k: &dst_k,
                dst_v: &dst_v,
                dst_offset: &off,
                capacity,
            }),
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let got_k = dst_k.read_f32();
        let got_v = dst_v.read_f32();
        let transformed_k = k.read_f32();
        let transformed_v = v.read_f32();
        let start = offset as usize;
        assert_eq!(
            &got_k[start..start + span as usize],
            &transformed_k[..span as usize]
        );
        assert_eq!(
            &got_v[start..start + span as usize],
            &transformed_v[..span as usize]
        );
        assert_guard("valid fused K prefix", &got_k, 0..start);
        assert_guard("valid fused V prefix", &got_v, 0..start);
        assert_guard(
            "valid fused K slab tail",
            &got_k,
            capacity as usize..physical,
        );
        assert_guard(
            "valid fused V slab tail",
            &got_v,
            capacity as usize..physical,
        );
    });
}

#[test]
fn invalid_declared_capacities_fail_before_any_dispatch() {
    with_gpu(|rt| {
        let src = buf(rt, &[1.0, 2.0, 3.0]);
        let src_v = buf(rt, &[4.0, 5.0, 6.0]);
        let dst = seeded(rt, 4, SENTINEL);
        let dst_v = seeded(rt, 4, SENTINEL);
        let off = u32_buf(rt, 0);

        let err =
            nn::kv_store_timestep(rt, &src, &dst, &off, 3, 2).expect_err("capacity smaller than n");
        assert!(err.contains("smaller than n"), "unexpected error: {err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::kv_store_timestep(rt, &src, &dst, &off, 3, 5)
            .expect_err("declared capacity exceeds physical storage");
        assert!(err.contains("buffer holds"), "unexpected error: {err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::kv_store_timestep_pair(rt, &src, &src_v, &dst, &dst_v, &off, 3, 2)
            .expect_err("pair capacity smaller than n");
        assert!(err.contains("smaller than n"), "unexpected error: {err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::kv_store_timestep_pair(rt, &src, &src_v, &dst, &dst_v, &off, 3, 5)
            .expect_err("pair capacity exceeds physical storage");
        assert!(err.contains("buffer holds"), "unexpected error: {err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        let dims = QkvRopeDims {
            t: 2,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let q = buf(rt, &[1.0; 8]);
        let k = buf(rt, &[2.0; 8]);
        let v = buf(rt, &[3.0; 8]);
        let weight = buf(rt, &[1.0; 4]);
        let pos = u32_buf(rt, 0);
        let dst_k = seeded(rt, 16, SENTINEL);
        let dst_v = seeded(rt, 16, SENTINEL);
        let qkv = QkvBuffers {
            q: &q,
            k: &k,
            v: &v,
            q_weight: &weight,
            k_weight: &weight,
            v_weight: &weight,
        };

        let err = nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosBufferKvStore,
            qkv,
            dims,
            0,
            Some(&pos),
            Some(KvStoreTarget {
                dst_k: &dst_k,
                dst_v: &dst_v,
                dst_offset: &off,
                capacity: 7,
            }),
            false,
        )
        .expect_err("fused capacity smaller than T*Hkv*D");
        assert!(
            err.contains("smaller than the K/V span"),
            "unexpected error: {err}"
        );
        assert_eq!(rt.take_dispatch_count(), 0);

        let err = nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosBufferKvStore,
            qkv,
            dims,
            0,
            Some(&pos),
            Some(KvStoreTarget {
                dst_k: &dst_k,
                dst_v: &dst_v,
                dst_offset: &off,
                capacity: 17,
            }),
            false,
        )
        .expect_err("fused capacity exceeds physical storage");
        assert!(err.contains("buffer holds"), "unexpected error: {err}");
        assert_eq!(rt.take_dispatch_count(), 0);

        // This grid cannot be represented by the shader's uint gid, and is
        // rejected before the deliberately tiny buffers are inspected.
        let filled = u32_buf(rt, 0);
        let start = u32_buf(rt, 0);
        let err = nn::kv_ring_densify(rt, &src, &dst, &filled, &start, 2, u32::MAX)
            .expect_err("ring grid larger than Metal uint");
        assert!(
            err.contains("exceeds Metal uint"),
            "unexpected error: {err}"
        );
        assert_eq!(rt.take_dispatch_count(), 0);
    });
}

#[test]
fn ring_densify_clamps_filled_and_normalizes_wrapping_start() {
    with_gpu(|rt| {
        let capacity = 6u32;
        let n_slot = 2u32;
        let ring_elems = (capacity * n_slot) as usize;
        let physical = ring_elems + 5;
        let mut source_values: Vec<f32> = (0..ring_elems).map(|i| i as f32 + 0.5).collect();
        source_values.extend(std::iter::repeat_n(SENTINEL, physical - ring_elems));
        let src = buf(rt, &source_values);

        for (label, filled_value, start_value) in [
            ("filled beyond capacity", capacity + 3, 1u32),
            ("u32 max metadata", u32::MAX, u32::MAX),
        ] {
            let filled = u32_buf(rt, filled_value);
            let start = u32_buf(rt, start_value);
            let dst = seeded(rt, physical, SENTINEL);
            nn::kv_ring_densify(rt, &src, &dst, &filled, &start, n_slot, capacity).unwrap();
            rt.synchronize().unwrap();

            let got = dst.read_f32();
            let mut expected = Vec::with_capacity(ring_elems);
            for t in 0..capacity as u64 {
                let src_t = (start_value as u64 + t) % capacity as u64;
                for e in 0..n_slot as usize {
                    expected.push(source_values[src_t as usize * n_slot as usize + e]);
                }
            }
            assert_eq!(&got[..ring_elems], &expected, "{label}");
            assert_guard(label, &got, ring_elems..physical);
            assert_guard(label, &src.read_f32(), ring_elems..physical);
        }
    });
}

#[test]
fn rope_position_addition_does_not_wrap_at_u32_max() {
    with_gpu(|rt| {
        let dims = QkvRopeDims {
            t: 2,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 4,
            rotary_dim: 4,
            theta: 10_000.0,
            eps: 1e-6,
        };
        let q_values = [1.0, -0.5, 0.25, 1.5, 0.5, -0.25, 1.25, -1.0];
        let k_values = [-0.75, 0.5, 1.0, -1.25, 1.5, 0.25, -0.5, 0.75];
        let v_values = [0.25, 1.0, -0.75, 1.5, -1.0, 0.5, 0.75, -0.25];
        let weight = buf(rt, &[1.0; 4]);

        let run = |variant: QkvRopeVariant| {
            let q = buf(rt, &q_values);
            let k = buf(rt, &k_values);
            let v = buf(rt, &v_values);
            let pos = u32_buf(rt, u32::MAX);
            let dst_k = seeded(rt, 16, SENTINEL);
            let dst_v = seeded(rt, 16, SENTINEL);
            let off = u32_buf(rt, 4);
            let pos_arg = (variant != QkvRopeVariant::PosConst).then_some(&pos);
            let target = (variant == QkvRopeVariant::PosBufferKvStore).then_some(KvStoreTarget {
                dst_k: &dst_k,
                dst_v: &dst_v,
                dst_offset: &off,
                capacity: 12,
            });
            nn::rms_qkv_rope(
                rt,
                variant,
                QkvBuffers {
                    q: &q,
                    k: &k,
                    v: &v,
                    q_weight: &weight,
                    k_weight: &weight,
                    v_weight: &weight,
                },
                dims,
                u32::MAX,
                pos_arg,
                target,
                false,
            )
            .unwrap();
            rt.synchronize().unwrap();
            (q.read_f32(), k.read_f32(), v.read_f32())
        };

        let constant = run(QkvRopeVariant::PosConst);
        let pos_buffer = run(QkvRopeVariant::PosBuffer);
        let fused = run(QkvRopeVariant::PosBufferKvStore);
        for (name, lhs, rhs) in [
            ("const vs posbuf Q", &constant.0, &pos_buffer.0),
            ("const vs posbuf K", &constant.1, &pos_buffer.1),
            ("const vs posbuf V", &constant.2, &pos_buffer.2),
            ("posbuf vs fused Q", &pos_buffer.0, &fused.0),
            ("posbuf vs fused K", &pos_buffer.1, &fused.1),
            ("posbuf vs fused V", &pos_buffer.2, &fused.2),
        ] {
            assert_eq!(lhs, rhs, "{name}");
        }

        // Before the widening fix, MAX + token-index wrapped to position zero
        // for the second token. Compare that row with an explicit position-zero
        // run of the same input: the fixed path must apply a real rotation.
        let q_zero = buf(rt, &q_values[4..]);
        let k_zero = buf(rt, &k_values[4..]);
        let v_zero = buf(rt, &v_values[4..]);
        let zero_dims = QkvRopeDims { t: 1, ..dims };
        nn::rms_qkv_rope(
            rt,
            QkvRopeVariant::PosConst,
            QkvBuffers {
                q: &q_zero,
                k: &k_zero,
                v: &v_zero,
                q_weight: &weight,
                k_weight: &weight,
                v_weight: &weight,
            },
            zero_dims,
            0,
            None,
            None,
            false,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let zero = q_zero.read_f32();
        assert!(
            constant.0[4..8]
                .iter()
                .zip(&zero[..4])
                .any(|(high, at_zero)| (high - at_zero).abs() > 1e-3),
            "u32::MAX + token index behaved like wrapped position zero"
        );
    });
}

/// Lock the three capacity ABI slots, wide shader arithmetic, and both Rust
/// binders together. This is allocation-free coverage for extents too large to
/// materialize in a test process.
#[test]
fn kv_shader_and_host_capacity_abis_are_locked() {
    let kv = include_str!("../kernels/kv_store.metal");
    let qkv = include_str!("../kernels/rms_qkv_rope.metal");
    let host = include_str!("../src/nn.rs");
    let gemma = include_str!("../../../gemma-metal/src/kernels.rs");
    let model = include_str!("../../../gemma-metal/src/gpu_model.rs");

    assert!(kv.contains("constant uint &dst_capacity [[buffer(4)]]"));
    assert!(kv.contains("constant uint &dst_capacity [[buffer(6)]]"));
    assert_eq!(
        kv.matches("count > capacity - dst_offset").count(),
        2,
        "both store kernels need subtraction-form bounds checks"
    );
    assert!(kv.contains("const ulong live = (ulong)min(*filled_ptr, capacity)"));
    assert!(kv.contains("const ulong total = live * slot_width"));
    assert!(kv.contains("((ulong)*start_ptr + t) % (ulong)capacity"));
    assert!(kv.contains("src_t * slot_width + e"));

    assert!(qkv.contains("    ulong pos,"));
    assert!(qkv.contains("constant uint &kv_capacity [[buffer(17)]]"));
    assert!(qkv.contains("kv_span > capacity - kv_dst_offset"));
    assert!(qkv.matches("const ulong total_q").count() >= 3);
    assert!(qkv.matches("* (ulong)D").count() >= 8);
    assert!(!qkv.contains("const uint base = kv_dst_offset"));

    assert!(host.contains("set_u32(bnd, capacity, 4)"));
    assert!(host.contains("set_u32(bnd, capacity, 6)"));
    assert!(host.contains("set_u32(bnd, capacity, 17)"));
    assert!(host.contains("require_runtime(rt, buf, what)?"));
    let fused_host = host
        .split("pub unsafe fn rms_qkv_rope_with_scalars")
        .nth(1)
        .and_then(|tail| {
            tail.split("// -------------------------------------------------------------- Sampling")
                .next()
        })
        .expect("fused QKV host wrapper");
    assert!(
        fused_host.find("set_gpu_buf(bnd, t.dst_offset, 16);").unwrap()
            < fused_host.find("scalars(bnd, kv_capacity);").unwrap(),
        "the unsafe callback must run after the validated default offset bind so a stable pool can rebind slot 16"
    );
    assert!(host.contains("pub fn validate_rms_qkv_rope("));
    assert!(
        fused_host.find("validate_rms_qkv_rope(").unwrap()
            < fused_host.find("let p = rt.pipeline").unwrap(),
        "the canonical QKV preflight must run before pipeline lookup and scalar callback"
    );

    assert!(gemma.contains("bind_u32(bnd, capacity_off, 4)"));
    assert!(gemma.contains("bind_u32(bnd, capacity_off, 6)"));
    assert_eq!(
        gemma.matches("validate_kv_capacity(").count(),
        4,
        "the direct single/pair adapters must validate every destination against the declared capacity"
    );
    assert!(gemma.contains("checked_mul(n_slot as usize)"));
    assert!(gemma.contains("fixed grid {n} exceeds Metal uint indexing"));
    assert!(!gemma.contains("if n == 0 || filled == 0"));
    assert!(!gemma.contains("let n = (t * hq + 2 * t * hkv) as usize"));

    let fused_gemma = gemma
        .split("pub fn rms_qkv_rope_kv_store(")
        .nth(1)
        .and_then(|tail| tail.split("\npub fn ple_lookup(").next())
        .expect("Gemma fused QKV adapter");
    let preflight = fused_gemma
        .find("tessl::nn::validate_rms_qkv_rope(")
        .expect("Gemma fused QKV canonical preflight");
    let pipeline = fused_gemma
        .find(".pipeline(KernelId::RmsQkvRopeKvStore.entry_name())")
        .expect("Gemma fused QKV pipeline preflight");
    let first_push = fused_gemma
        .find("gpu.icb_scalars.push_u32(")
        .expect("Gemma fused QKV stable-scalar allocation");
    let delegated = fused_gemma
        .find("tessl::nn::rms_qkv_rope_with_scalars(")
        .expect("Gemma fused QKV delegated encode");
    assert!(
        preflight < pipeline && pipeline < first_push && first_push < delegated,
        "validation and fallible pipeline lookup must precede scalar-pool mutation, then the canonical Tessl seam must encode"
    );
    assert!(fused_gemma.contains("dst_offset: &gpu.icb_scalars.u32s"));
    assert!(fused_gemma.contains("bind_u32(bnd, kv_dst_off, 16)"));
    assert!(fused_gemma.contains("bind_u32(bnd, kv_capacity_off, 17)"));
    assert!(model.contains("elem_capacity_u32()"));
}
