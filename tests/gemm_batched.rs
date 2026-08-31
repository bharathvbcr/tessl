//! Strided batched GEMM.
//!
//! The oracle is the single-matrix path: batch element `i` must equal a plain
//! `gemm` over the same operands, **bit for bit**. Batching changes which
//! threadgroup does the work and nothing about the arithmetic, so anything
//! short of bit-identity means a stride or an offset is wrong.
//!
//! A zero stride — one shared B against a batch of A — is checked separately,
//! because it is the case the API exists for and the one where an off-by-one
//! in the offset produces a plausible matrix rather than a crash.

mod common;

use std::sync::Arc;

use common::{random_f32, with_gpu};
use tessl::gemm::{gemm, gemm_batched, BatchStrides, BatchedGemm, GemmBackend};
use tessl::tensor::Tensor;
use tessl::GpuRuntime;

fn tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc");
    t.buffer.write_f32(data);
    t
}

#[test]
fn every_batch_element_equals_a_single_gemm_bit_for_bit() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        for &(m, n, k, batch) in &[(128usize, 64usize, 64usize, 4usize), (96, 80, 128, 3)] {
            let a_h = random_f32(m * k * batch, 0xBA7 + k as u64);
            let b_h = random_f32(k * n * batch, 0xC4E + n as u64);
            let a = tensor(rt, &[batch * m, k], &a_h);
            let b = tensor(rt, &[batch * k, n], &b_h);
            let c = tensor(rt, &[batch * m, n], &vec![0.0f32; batch * m * n]);

            gemm_batched(
                &a,
                &b,
                &c,
                GemmBackend::TensorOps,
                BatchedGemm {
                    m,
                    n,
                    k,
                    batch,
                    strides: BatchStrides::contiguous(m, n, k),
                },
            )
            .expect("gemm_batched");
            rt.synchronize().unwrap();
            let batched = c.buffer.read_f32();

            for i in 0..batch {
                let ai = tensor(rt, &[m, k], &a_h[i * m * k..(i + 1) * m * k]);
                let bi = tensor(rt, &[k, n], &b_h[i * k * n..(i + 1) * k * n]);
                let ci = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
                gemm(&ai, &bi, &ci, GemmBackend::TensorOps).expect("gemm");
                rt.synchronize().unwrap();
                let single = ci.buffer.read_f32();
                for e in 0..m * n {
                    assert_eq!(
                        batched[i * m * n + e].to_bits(),
                        single[e].to_bits(),
                        "{m}x{n}x{k} batch {i} element {e} differs from a single gemm"
                    );
                }
            }
        }
    });
}

#[test]
fn a_zero_stride_broadcasts_one_shared_operand() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k, batch) = (128usize, 64usize, 64usize, 5usize);
        let a_h = random_f32(m * k * batch, 0x5_A7ED);
        let b_h = random_f32(k * n, 0x5EED);
        let a = tensor(rt, &[batch * m, k], &a_h);
        // One B, not `batch` copies of it. That is the saving.
        let b = tensor(rt, &[k, n], &b_h);
        let c = tensor(rt, &[batch * m, n], &vec![0.0f32; batch * m * n]);

        gemm_batched(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            BatchedGemm {
                m,
                n,
                k,
                batch,
                strides: BatchStrides::shared_b(m, n, k),
            },
        )
        .expect("gemm_batched");
        rt.synchronize().unwrap();
        let batched = c.buffer.read_f32();

        for i in 0..batch {
            let ai = tensor(rt, &[m, k], &a_h[i * m * k..(i + 1) * m * k]);
            let bi = tensor(rt, &[k, n], &b_h);
            let ci = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
            gemm(&ai, &bi, &ci, GemmBackend::TensorOps).expect("gemm");
            rt.synchronize().unwrap();
            let single = ci.buffer.read_f32();
            for e in 0..m * n {
                assert_eq!(
                    batched[i * m * n + e].to_bits(),
                    single[e].to_bits(),
                    "shared-B batch {i} element {e}"
                );
            }
        }
    });
}

#[test]
fn a_batch_of_one_is_a_plain_gemm() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k) = (128usize, 64usize, 128usize);
        let a_h = random_f32(m * k, 0x0_10E3);
        let b_h = random_f32(k * n, 0x0_10E4);
        let a = tensor(rt, &[m, k], &a_h);
        let b = tensor(rt, &[k, n], &b_h);
        let plain = tensor(rt, &[m, n], &vec![0.0f32; m * n]);
        let one = tensor(rt, &[m, n], &vec![0.0f32; m * n]);

        gemm(&a, &b, &plain, GemmBackend::TensorOps).expect("gemm");
        gemm_batched(
            &a,
            &b,
            &one,
            GemmBackend::TensorOps,
            BatchedGemm {
                m,
                n,
                k,
                batch: 1,
                strides: BatchStrides::contiguous(m, n, k),
            },
        )
        .expect("gemm_batched");
        rt.synchronize().unwrap();

        let (p, o) = (plain.buffer.read_f32(), one.buffer.read_f32());
        for e in 0..m * n {
            assert_eq!(
                p[e].to_bits(),
                o[e].to_bits(),
                "batch of one differs at {e}"
            );
        }
    });
}

#[test]
fn a_batch_that_reaches_past_an_operand_is_refused() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(true);
        let (m, n, k) = (128usize, 64usize, 64usize);
        // Sized for two batches, asked for four: without the check the kernel
        // reads past the end of device memory and returns whatever is there.
        let a = tensor(rt, &[2 * m, k], &vec![1.0f32; 2 * m * k]);
        let b = tensor(rt, &[2 * k, n], &vec![1.0f32; 2 * k * n]);
        let c = tensor(rt, &[2 * m, n], &vec![0.0f32; 2 * m * n]);
        let err = gemm_batched(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            BatchedGemm {
                m,
                n,
                k,
                batch: 4,
                strides: BatchStrides::contiguous(m, n, k),
            },
        )
        .expect_err("batch overruns the operands");
        assert!(err.contains("reaches"), "{err}");

        // And a batch of zero is a no-op, not an error and not a fault.
        assert!(gemm_batched(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            BatchedGemm {
                m,
                n,
                k,
                batch: 0,
                strides: BatchStrides::contiguous(m, n, k)
            }
        )
        .is_ok());
    });
}

#[test]
fn batched_refuses_paths_without_a_register_accumulator() {
    with_gpu(|rt| {
        rt.set_relaxed_precision(false);
        let (m, n, k) = (128usize, 64usize, 64usize);
        let a = tensor(rt, &[2 * m, k], &vec![1.0f32; 2 * m * k]);
        let b = tensor(rt, &[2 * k, n], &vec![1.0f32; 2 * k * n]);
        let c = tensor(rt, &[2 * m, n], &vec![0.0f32; 2 * m * n]);
        let err = gemm_batched(
            &a,
            &b,
            &c,
            GemmBackend::TensorOps,
            BatchedGemm {
                m,
                n,
                k,
                batch: 2,
                strides: BatchStrides::contiguous(m, n, k),
            },
        )
        .expect_err("exact f32 has no batched kernel");
        assert!(err.contains("cooperative-destination"), "{err}");
    });
}
