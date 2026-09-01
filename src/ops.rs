//! Small elementwise device ops that do not belong to a larger kernel family.
//!
//! These live in `utils.metal` rather than in one of the named kernel sources,
//! and they are the operations a caller needs *between* the big kernels: a
//! logit softcap, an in-place scale, a buffer zero.
//!
//! Anything with a shape contract worth checking belongs in [`crate::nn`]
//! instead, which validates operands before encoding. The split is by whether
//! there is an invariant to enforce, not by kernel size.

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, set_f32, set_tensor, set_u32};
use crate::runtime::GpuRuntime;
use crate::tensor::Tensor;

/// `post = softcap * tanh(pre / softcap)` (elementwise).
pub fn softcap_f32(rt: &Arc<GpuRuntime>, pre: &Tensor, softcap: f32) -> Result<Tensor, String> {
    pre.validate()?;
    if pre.dtype != crate::tensor::DType::F32
        || !Arc::ptr_eq(rt, pre.runtime())
        || !softcap.is_finite()
        || softcap <= 0.0
        || pre.numel() > u32::MAX as usize
    {
        return Err(
            "softcap requires f32, matching runtime, uint count, and a finite positive cap".into(),
        );
    }
    let post = rt.alloc_tensor_f32(&pre.shape)?;
    let p = rt.pipeline("softcap_f32")?;
    let n = pre.numel();
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, pre, 0);
        set_tensor(bnd, &post, 1);
        set_f32(bnd, softcap, 2);
        set_u32(bnd, n as u32, 3);
    })?;
    Ok(post)
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    #[test]
    fn softcap_rejects_invalid_boundary() {
        let rt = GpuRuntime::new().unwrap();
        let bf = rt.alloc_tensor_bf16(&[4]).unwrap();
        assert!(softcap_f32(&rt, &bf, 1.0).is_err());
        let t = rt.alloc_tensor_f32(&[4]).unwrap();
        for cap in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(softcap_f32(&rt, &t, cap).is_err());
        }
        assert_eq!(rt.take_dispatch_count(), 0);
    }
}
