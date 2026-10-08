//! The README's quickstart, as a build target.
//!
//! A snippet that only lives in a README drifts the moment a signature changes
//! and nothing notices. This is the same code, compiled and run by CI.
//!
//! ```text
//! cargo run --release --example gemm
//! ```

use tessl::{gemm, GemmBackend, GpuRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the Metal 4 GPU runtime.
    let rt = GpuRuntime::new()?;
    println!("device: {}", rt.device_name());

    // 2. Allocate operands on the GPU.
    let (m, k, n) = (512, 256, 128);
    let a = rt.alloc_tensor_f32(&[m, k])?;
    let b = rt.alloc_tensor_f32(&[k, n])?;
    let c = rt.alloc_tensor_f32(&[m, n])?;

    // Fill A and B so the result is something checkable rather than zeros.
    a.buffer.write_f32(&vec![1.0f32; m * k]);
    b.buffer.write_f32(&vec![2.0f32; k * n]);

    // 3. C = A @ B through MPP TensorOps.
    gemm(&a, &b, &c, GemmBackend::TensorOps)?;

    // 4. Wait for the GPU before reading back.
    rt.synchronize()?;

    // Every element is a sum of `k` products of 1.0 and 2.0.
    let out = c.buffer.read_f32();
    let expected = 2.0 * k as f32;
    assert!(
        (out[0] - expected).abs() < 1e-3,
        "C[0] = {} but every element should be {expected}",
        out[0]
    );
    println!(
        "C[0] = {} (expected {expected}) over {m}x{k} @ {k}x{n}",
        out[0]
    );
    Ok(())
}
