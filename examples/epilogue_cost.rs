//! What does a fused epilogue actually save?
//!
//! Three arms over the same shapes:
//!
//!   * `gemm` alone — the floor.
//!   * `gemm_epilogue` with bias and GELU — the fused form.
//!   * `gemm` plus one `add_inplace_f32` over C — a *lower bound* on the
//!     unfused form. That kernel reads C, reads a second operand and writes C,
//!     which is strictly less work than a real bias-broadcast pass and far less
//!     than bias plus a separate activation. If fusing does not beat even this,
//!     it is not worth having.
//!
//! Measurement rules are the ones the crossover benchmarks in this workspace
//! settled on: every arm is exercised before any arm is timed, and each is
//! timed twice in opposite orders with the spread reported. A spread far from
//! 1.00 means ordering still matters and the numbers should not be quoted.
//!
//! Run: `cargo run --release --example epilogue_cost`

use std::time::{Duration, Instant};

use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32};
use tessl::gemm::{gemm, gemm_epilogue, Activation, Epilogue, GemmBackend};
use tessl::{GpuRuntime, Tensor};

const SHAPES: &[(usize, usize, usize)] = &[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 512)];
const RAMP: usize = 10;
const ITERS: usize = 30;

fn ms(total: Duration, iters: usize) -> f64 {
    total.as_secs_f64() * 1000.0 / iters as f64
}

fn filled(rt: &std::sync::Arc<GpuRuntime>, shape: &[usize], v: f32) -> Tensor {
    let t = rt.alloc_tensor_f32(shape).expect("alloc");
    t.buffer
        .write_f32(&vec![v; shape.iter().product::<usize>()]);
    t
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = GpuRuntime::new()?;
    // The cooperative-destination path, which is the only one with a register
    // accumulator to fuse into.
    rt.set_relaxed_precision(true);
    println!("device: {}\n", rt.device_name());
    println!(
        "  {:>16} {:>10} {:>12} {:>12} {:>12} {:>9}",
        "shape", "gemm (ms)", "fused (ms)", "+1 pass (ms)", "epi cost", "spread"
    );

    for &(m, n, k) in SHAPES {
        let a = filled(&rt, &[m, k], 0.01);
        let b = filled(&rt, &[k, n], 0.02);
        let c = filled(&rt, &[m, n], 0.0);
        let bias = filled(&rt, &[n], 0.5);
        let other = filled(&rt, &[m, n], 1.0);
        let epi = Epilogue {
            alpha: 1.0,
            beta: 0.0,
            bias: Some(&bias),
            activation: Activation::GeluTanh,
        };
        let add = rt.pipeline("add_inplace_f32")?;
        let elems = m * n;

        let plain = |iters: usize| -> Duration {
            let t0 = Instant::now();
            for _ in 0..iters {
                gemm(&a, &b, &c, GemmBackend::TensorOps).expect("gemm");
            }
            rt.synchronize().expect("sync");
            t0.elapsed()
        };
        let fused = |iters: usize| -> Duration {
            let t0 = Instant::now();
            for _ in 0..iters {
                gemm_epilogue(&a, &b, &c, GemmBackend::TensorOps, epi).expect("epilogue");
            }
            rt.synchronize().expect("sync");
            t0.elapsed()
        };
        let one_pass = |iters: usize| -> Duration {
            let t0 = Instant::now();
            for _ in 0..iters {
                gemm(&a, &b, &c, GemmBackend::TensorOps).expect("gemm");
                dispatch_1d(&rt, &add, elems, |bnd| {
                    set_gpu_buf(bnd, &c.buffer, 0);
                    set_gpu_buf(bnd, &other.buffer, 1);
                    set_u32(bnd, elems as u32, 2);
                })
                .expect("add");
            }
            rt.synchronize().expect("sync");
            t0.elapsed()
        };

        // Ramp all three before timing any.
        plain(RAMP);
        fused(RAMP);
        one_pass(RAMP);

        let (p1, f1, o1) = (plain(ITERS), fused(ITERS), one_pass(ITERS));
        let (o2, f2, p2) = (one_pass(ITERS), fused(ITERS), plain(ITERS));

        let p = ms(p1, ITERS).min(ms(p2, ITERS));
        let f = ms(f1, ITERS).min(ms(f2, ITERS));
        let o = ms(o1, ITERS).min(ms(o2, ITERS));
        let spread = (ms(f1, ITERS) / ms(f2, ITERS)).max(ms(f2, ITERS) / ms(f1, ITERS));

        println!(
            "  {:>16} {:>10.3} {:>12.3} {:>12.3} {:>11.3} {:>9.2}",
            format!("{m}x{n}x{k}"),
            p,
            f,
            o,
            f - p,
            spread
        );
    }
    println!(
        "\n`epi cost` is fused minus plain: what bias + GELU cost when applied in registers.\n\
         `+1 pass` is plain plus a single elementwise sweep over C, which is the cheapest\n\
         any unfused epilogue can possibly be."
    );
    Ok(())
}
