//! A transformer feed-forward block using the kernels promoted from
//! `gemma-metal`, driven entirely through [`tessl::nn`].
//!
//! ```text
//! cargo run --release --example nn_layer
//! ```
//!
//! RMSNorm the hidden state, run two projections, gate them with
//! `gelu_pytorch_tanh`, and fold the result back into the residual stream —
//! four dispatches, no host round-trip between them.

use tessl::{gemm_f32, GemmBackend, GpuRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = GpuRuntime::new()?;
    println!("device: {}", rt.device_name());

    let (rows, hidden, ffn) = (8usize, 256usize, 512usize);
    let eps = 1e-6f32;

    // Hidden state and the residual stream it will be folded back into.
    let x = rt.alloc_tensor_f32(&[rows, hidden])?;
    let residual = rt.alloc_tensor_f32(&[rows, hidden])?;
    let norm_weight = rt.alloc_buffer(hidden * 4)?;
    x.buffer.write_f32(
        &(0..rows * hidden)
            .map(|i| (i % 17) as f32 * 0.1 - 0.8)
            .collect::<Vec<_>>(),
    );
    residual.buffer.write_f32(&vec![0.25f32; rows * hidden]);
    norm_weight.write_f32(&vec![1.0f32; hidden]);

    // 1. Normalize into a scratch tensor.
    let normed = rt.alloc_tensor_f32(&[rows, hidden])?;
    tessl::nn::rms_norm_f32(
        &rt,
        &x.buffer,
        &norm_weight,
        &normed.buffer,
        rows as u32,
        hidden as u32,
        eps,
    )?;

    // 2. Gate and up projections. Dense here for clarity; swap in
    //    `nn::gemv_q4_mlx_simd` for a quantized decode step.
    let w_gate = rt.alloc_tensor_f32(&[hidden, ffn])?;
    let w_up = rt.alloc_tensor_f32(&[hidden, ffn])?;
    w_gate.buffer.write_f32(&vec![0.02f32; hidden * ffn]);
    w_up.buffer.write_f32(&vec![0.03f32; hidden * ffn]);
    let gate = rt.alloc_tensor_f32(&[rows, ffn])?;
    let up = rt.alloc_tensor_f32(&[rows, ffn])?;
    gemm_f32(&normed, &w_gate, &gate, GemmBackend::TensorOps)?;
    gemm_f32(&normed, &w_up, &up, GemmBackend::TensorOps)?;

    // 3. Fused gating: mid = gelu(gate) * up, one dispatch, no intermediate.
    let mid = rt.alloc_tensor_f32(&[rows, ffn])?;
    tessl::nn::mlp_gelu_tanh(
        &rt,
        &gate.buffer,
        &up.buffer,
        &mid.buffer,
        (rows * ffn) as u32,
    )?;

    // 4. Down projection, then fold into the residual with a fused
    //    norm-and-add. `layer_scale` of 1.0 is the plain residual add.
    let w_down = rt.alloc_tensor_f32(&[ffn, hidden])?;
    w_down.buffer.write_f32(&vec![0.01f32; ffn * hidden]);
    let down = rt.alloc_tensor_f32(&[rows, hidden])?;
    gemm_f32(&mid, &w_down, &down, GemmBackend::TensorOps)?;
    tessl::nn::rms_norm_residual_add_f32(
        &rt,
        &down.buffer,
        &norm_weight,
        &residual.buffer,
        rows as u32,
        hidden as u32,
        eps,
        1.0,
    )?;

    rt.synchronize()?;

    let out = residual.buffer.read_f32();
    assert!(
        out[..rows * hidden].iter().all(|v| v.is_finite()),
        "the block produced a non-finite value"
    );
    println!(
        "{rows}x{hidden} block through 4 dispatches; residual[0] = {:.6}",
        out[0]
    );
    Ok(())
}
