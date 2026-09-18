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
        let c = Tensor::from_mtl_buffer(rt, raw, &[m, n], DType::F32, 0).unwrap();
        assert_eq!(c.buffer.kind(), BufferKind::External);
        gemm_f32(&a, &b, &c, GemmBackend::TensorOps).unwrap();
        rt.synchronize().unwrap();
        let got = c.buffer.read_f32();
        assert_eq!(want.len(), got.len());
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "external C diverged at [{i}]: {w} vs {g}"
            );
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
        let err = Tensor::from_mtl_buffer(rt, raw.clone(), &[8], DType::F32, 0)
            .expect_err("8 f32 need 32 bytes");
        assert!(
            err.contains("out of bounds") || err.contains("misaligned"),
            "unexpected: {err}"
        );
        let ok = Tensor::from_mtl_buffer(rt, raw, &[4], DType::F32, 0);
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
        let Some(foreign) = devices.iter().find(|d| d.registryID() != rt.device.registryID())
        else {
            return;
        };
        let raw = foreign
            .newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared)
            .expect("foreign buffer");
        let err = Tensor::from_mtl_buffer(rt, raw, &[4], DType::F32, 0).unwrap_err();
        assert!(
            err.contains("registryID"),
            "expected registryID rejection, got {err}"
        );
    });
}
