//! A/B the bf16 NN GEMM tile geometry against the production kernel.
//!
//! Tests two suspects behind the ~2× gap vs PyTorch MPS bf16: output-tile
//! arithmetic intensity, and the zero_f32(C) pre-pass that only exists because
//! the production kernel accumulates on the first K block. Every variant is
//! checked against the production kernel's output before its time is reported.

// A tuning sweep takes the full GEMM shape plus the tile parameters it is
// sweeping. Bundling them would add a struct that every call site unpacks.
#![allow(clippy::too_many_arguments)]

use objc2_metal::MTLComputePipelineState;
use std::time::Instant;
use tessl::gemm::{cast_f32_to_bf16, gemm, GemmBackend};
use tessl::runtime::{mtl_size, GpuRuntime};
use tessl::tensor::Tensor;

struct Variant {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    bk: usize,
    nsg: usize,
    /// Mirrors production: first K block accumulates, so C must be pre-zeroed.
    needs_zero: bool,
}

const VARIANTS: &[Variant] = &[
    Variant {
        kernel: "mm_bf16_64x32_bk128_sg4_accf",
        sm: 64,
        sn: 32,
        bk: 128,
        nsg: 4,
        needs_zero: true,
    },
    Variant {
        kernel: "mm_bf16_64x64_bk256_sg4",
        sm: 64,
        sn: 64,
        bk: 256,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_128x64_bk256_sg8",
        sm: 128,
        sn: 64,
        bk: 256,
        nsg: 8,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_128x128_bk256_sg4",
        sm: 128,
        sn: 128,
        bk: 256,
        nsg: 4,
        needs_zero: false,
    },
    // Cooperative destination tensor: register accumulator, C written once,
    // single dynamic-K run. bk: 1 so the divisibility gate never skips on K.
    Variant {
        kernel: "mm_bf16_coop_64x32_sg4",
        sm: 64,
        sn: 32,
        bk: 1,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_coop_64x64_sg4",
        sm: 64,
        sn: 64,
        bk: 1,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_coop_128x64_sg4",
        sm: 128,
        sn: 64,
        bk: 1,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_coop_128x64_sg8",
        sm: 128,
        sn: 64,
        bk: 1,
        nsg: 8,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_coop_128x128_sg8",
        sm: 128,
        sn: 128,
        bk: 1,
        nsg: 8,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_bf16_coop_256x64_sg8",
        sm: 256,
        sn: 64,
        bk: 1,
        nsg: 8,
        needs_zero: false,
    },
];

const SHAPES: &[(usize, usize, usize, &str)] = &[
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
    // Small / narrow shapes: do big tiles starve the grid?
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (4096, 128, 2048, "narrow_n128"),
    (1024, 256, 1024, "narrow_n256"),
];

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn run_variant(
    rt: &std::sync::Arc<GpuRuntime>,
    v: &Variant,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    let p = rt.pipeline(v.kernel)?;
    let zero_p = rt.pipeline("zero_f32")?;
    let tiles_n = n / v.sn;
    let tg = tiles_n * (m / v.sm);
    let tpt = p.threadExecutionWidth() * v.nsg;
    let numel = c.numel();
    let z_tpt = zero_p.threadExecutionWidth().min(numel).max(1);
    let z_groups = numel.div_ceil(z_tpt);
    let needs_zero = v.needs_zero;
    rt.with_binder(|bnd| {
        if needs_zero {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(z_groups, 1, 1), mtl_size(z_tpt, 1, 1));
            bnd.barrier();
        }
        bnd.set_pipeline(&p);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    println!(
        "{:<32}{:>12}{:>14}{:>16}",
        "kernel", "maxTPTG", "tgMem(B)", "requested TPTG"
    );
    for v in VARIANTS {
        match rt.pipeline(v.kernel) {
            Ok(p) => {
                let w = p.threadExecutionWidth();
                println!(
                    "{:<32}{:>12}{:>14}{:>16}",
                    v.kernel,
                    p.maxTotalThreadsPerThreadgroup(),
                    p.staticThreadgroupMemoryLength(),
                    w * v.nsg
                );
            }
            Err(e) => println!("{:<32}  pipeline error: {e}", v.kernel),
        }
    }

    for &(m, n, k, label) in SHAPES {
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        a.buffer.write_f32(&fill(m * k, 1));
        b.buffer.write_f32(&fill(k * n, 2));
        let a_bf = cast_f32_to_bf16(&a)?;
        let b_bf = cast_f32_to_bf16(&b)?;

        // Production reference.
        let c_ref = rt.alloc_tensor_f32(&[m, n])?;
        gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
        rt.synchronize()?;
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
        let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prod = {
            for _ in 0..warmup {
                gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
            }
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            median(s)
        };
        println!(
            "\n{label}  M={m} N={n} K={k}   production {prod:.3} ms  {:.0} GFLOP/s",
            flop / (prod * 1e6)
        );
        println!(
            "  {:<32}{:>10}{:>12}{:>9}{:>12}",
            "variant", "ms", "GFLOP/s", "vs prod", "max_rel_err"
        );

        for v in VARIANTS {
            if m % v.sm != 0 || n % v.sn != 0 || k % v.bk != 0 {
                println!("  {:<32}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            let c = rt.alloc_tensor_f32(&[m, n])?;
            if run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k).is_err() {
                println!("  {:<32}{:>10}", v.kernel, "skip(pipe)");
                continue;
            }
            rt.synchronize()?;
            let got = c.buffer.read_f32()[..m * n].to_vec();
            let err = got
                .iter()
                .zip(&refv)
                .map(|(x, y)| (*x as f64 - *y as f64).abs())
                .fold(0.0, f64::max)
                / refmax;

            for _ in 0..warmup {
                run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k)?;
                rt.synchronize()?;
            }
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                run_variant(&rt, v, &a_bf, &b_bf, &c, m, n, k)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let med = median(s);
            println!(
                "  {:<32}{:>10.3}{:>12.0}{:>8.2}×{:>12.2e}",
                v.kernel,
                med,
                flop / (med * 1e6),
                prod / med,
                err
            );
        }

        let prod_after = {
            let mut s = Vec::new();
            for _ in 0..iters {
                let t0 = Instant::now();
                gemm(&a_bf, &b_bf, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                s.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            median(s)
        };
        println!(
            "  production re-measured after: {prod_after:.3} ms ({:.0} GFLOP/s) \
— drift vs before: {:+.1}%",
            flop / (prod_after * 1e6),
            (prod_after / prod - 1.0) * 100.0
        );
    }
    Ok(())
}
