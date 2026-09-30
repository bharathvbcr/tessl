//! External `MTLBuffer` wrap and SharedEvent handoff surface.
//!
//! Tessl and sparsl stay on separate queues (Metal 4 vs Metal 3). This suite
//! locks the tessl side: wrap a buffer as a GEMM operand, reject device /
//! bounds mistakes, and expose `(shared_event, last_signaled_value)`.

mod common;

use common::{random_f32, tensor_f32, with_gpu};
use objc2_metal::{MTLDevice, MTLResourceOptions, MTLSharedEvent};
use tessl::{gemm_f32, BufferKind, DType, GemmBackend, Tensor};

#[test]
fn from_mtl_buffer_gemm_operand_matches_native_tensor() {
    with_gpu(|rt| {
        let (m, n, k) = (32, 48, 64);
        let a_host = random_f32(m * k, 41);
        let b_host = random_f32(k * n, 42);
        let a = tensor_f32(rt, &[m, k], &a_host);
        let b = tensor_f32(rt, &[k, n], &b_host);
        let c_native = rt.alloc_tensor_f32(&[m, n]).unwrap();
        gemm_f32(&a, &b, &c_native, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let want = c_native.buffer.read_f32();

        let nbytes = (m * n) * 4;
        let raw = rt
            .device
            .newBufferWithLength_options(nbytes, MTLResourceOptions::StorageModeShared)
            .expect("alloc raw MTLBuffer");
        // SAFETY: a fresh buffer only this runtime touches.
        let c = unsafe { Tensor::from_mtl_buffer(rt, raw, &[m, n], DType::F32, 0) }.unwrap();
        assert_eq!(c.buffer.kind(), BufferKind::External);
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        assert_eq!(want.len(), got.len());
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "external C diverged at [{i}]: {w} vs {g}");
        }
        assert!(want.iter().any(|&x| x != 0.0));
        // Commit advanced the SharedEvent timeline used for cross-crate handoff.
        assert!(rt.last_signaled_value() > 0);
        let _ = rt.shared_event().signaledValue();
    });
}

#[test]
fn from_mtl_buffer_rejects_short_buffer_and_bad_offset() {
    with_gpu(|rt| {
        let raw = rt
            .device
            .newBufferWithLength_options(16, MTLResourceOptions::StorageModeShared)
            .expect("16-byte buffer");
        // SAFETY (this and the next wrap): a fresh buffer only this runtime touches.
        let err =
            unsafe { Tensor::from_mtl_buffer(rt, raw.clone(), &[8], DType::F32, 0) }.expect_err("8 f32 need 32 bytes");
        assert!(
            err.contains("out of bounds") || err.contains("misaligned"),
            "unexpected: {err}"
        );
        let ok = unsafe { Tensor::from_mtl_buffer(rt, raw, &[4], DType::F32, 0) };
        assert!(ok.is_ok());
    });
}

#[test]
fn from_mtl_buffer_rejects_foreign_device_when_available() {
    with_gpu(|rt| {
        let devices = objc2_metal::MTLCopyAllDevices();
        if devices.count() < 2 {
            return;
        }
        let Some(foreign) = devices.iter().find(|d| d.registryID() != rt.device.registryID()) else {
            return;
        };
        let raw = foreign
            .newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared)
            .expect("foreign buffer");
        // SAFETY: a fresh buffer; the wrap is refused before any use.
        let err = unsafe { Tensor::from_mtl_buffer(rt, raw, &[4], DType::F32, 0) }.unwrap_err();
        assert!(err.contains("registryID"), "expected registryID rejection, got {err}");
    });
}

/// A GPU-private buffer has no CPU mapping: `contents()` is nil. tessl's
/// own pools only hand out shared storage, but a wrapped foreign buffer can be
/// private, and a host read of it has to fail rather than build a slice over
/// a null pointer. The same wrap stays usable as a GPU operand.
#[test]
fn a_wrapped_private_buffer_refuses_host_access_but_serves_the_gpu() {
    with_gpu(|rt| {
        let raw = rt
            .device
            .newBufferWithLength_options(64, MTLResourceOptions::StorageModePrivate)
            .expect("private buffer");
        // SAFETY: a fresh buffer only this runtime touches.
        let t = unsafe { Tensor::from_mtl_buffer(rt, raw, &[4, 4], DType::F32, 0) }
            .expect("a private buffer is a valid GPU operand");
        let err = t
            .buffer
            .try_contents_u8()
            .err()
            .expect("host mapping of private storage");
        assert!(err.contains("private"), "{err}");
        assert!(t.read_f32().is_err(), "Tensor::read_f32 of private storage");
        // GPU use: C = A * B into the private buffer, then copied out on the GPU.
        let a = tensor_f32(rt, &[4, 4], &random_f32(16, 5));
        let b = tensor_f32(rt, &[4, 4], &random_f32(16, 6));
        gemm_f32(&a, &b, &t, GemmBackend::TensorOps).expect("gemm into private");
        let back = rt.alloc_tensor_f32(&[4, 4]).unwrap();
        tessl::tensor::gpu_copy(&t, &back).expect("copy out");
        rt.synchronize().unwrap();
        let want = rt.alloc_tensor_f32(&[4, 4]).unwrap();
        gemm_f32(&a, &b, &want, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(back.buffer.read_f32(), want.buffer.read_f32());
    });
}

/// Two wraps of one `MTLBuffer` are the same memory. Aliasing checks compared
/// the wrappers, so separately wrapped views of one torch storage (which is
/// how a binding sees a tensor and its gradient buffer) passed as disjoint.
#[test]
fn separate_wraps_of_one_buffer_are_seen_to_overlap() {
    with_gpu(|rt| {
        let raw = rt
            .device
            .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
            .expect("buffer");
        // SAFETY: a fresh buffer only this runtime touches.
        let (a, b) = unsafe {
            (
                Tensor::from_mtl_buffer(rt, raw.clone(), &[4], DType::F32, 0).unwrap(),
                Tensor::from_mtl_buffer(rt, raw.clone(), &[4], DType::F32, 8).unwrap(),
            )
        };
        let err = tessl::tensor::gpu_copy(&a, &b).expect_err("[0,16) and [8,24) overlap");
        assert!(err.contains("overlap"), "{err}");
        // Disjoint windows of the one buffer still copy.
        let c = unsafe { Tensor::from_mtl_buffer(rt, raw, &[4], DType::F32, 16) }.unwrap();
        a.buffer.write_f32(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        tessl::tensor::gpu_copy(&a, &c).expect("disjoint copy");
        rt.synchronize().unwrap();
        assert_eq!(c.read_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    });
}
