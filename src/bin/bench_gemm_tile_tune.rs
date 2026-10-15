//! A/B the bf16 NN GEMM tile geometry against the production kernel.
//!
//! Tests two suspects behind the ~2× gap vs PyTorch MPS bf16: output-tile
//! arithmetic intensity, and the zero_f32(C) pre-pass that only exists because
//! the production kernel accumulates on the first K block. Every variant is
//! checked against the production kernel's output before its time is reported.
//!
//! Production and the variants of a shape are timed as the arms of one
//! interleaved A/B (`common::run_paired`, `BENCH_ROUNDS` x `BENCH_ITERS`);
//! a variant absent from the metallib (a build without `TESSL_GEMM_TUNE=1`)
//! stops the run with exit status 2 before anything is timed.

// A tuning sweep takes the full GEMM shape plus the tile parameters it is
// sweeping. Bundling them would add a struct that every call site unpacks.
#![allow(clippy::too_many_arguments)]

mod common;

use objc2_metal::MTLComputePipelineState;
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

/// Exact-f32 (non-relaxed) coop variants, measured against production's exact
/// f32 NN (32x32, one simdgroup, zero pre-pass) on f32 operands. Qwen35's
/// `train_step` runs every GEMM on that kernel.
const F32_VARIANTS: &[Variant] = &[
    Variant {
        kernel: "mm_f32_exact_coop_64x64_sg4",
        sm: 64,
        sn: 64,
        bk: 1,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_f32_exact_coop_128x64_sg4",
        sm: 128,
        sn: 64,
        bk: 1,
        nsg: 4,
        needs_zero: false,
    },
    Variant {
        kernel: "mm_f32_exact_coop_128x64_sg8",
        sm: 128,
        sn: 64,
        bk: 1,
        nsg: 8,
        needs_zero: false,
    },
];

const SHAPES: &[(usize, usize, usize, &str)] = &[
    (2048, 2048, 2048, "square_2048"),
    // Qwen3.5-2B at T = 2048: MLP up/gate and down.
    (2048, 6144, 2048, "q35_mlp_up"),
    (2048, 2048, 6144, "q35_mlp_down"),
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
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
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
    let warmup = common::env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = common::env_usize("BENCH_ITERS", 5, 1)?;
    let rounds = common::env_usize("BENCH_ROUNDS", 6, 1)?;
    common::require_kernels(
        &rt,
        "bench_gemm_tile_tune",
        VARIANTS.iter().chain(F32_VARIANTS).map(|v| v.kernel),
    );

    println!(
        "{:<32}{:>12}{:>14}{:>16}",
        "kernel", "maxTPTG", "tgMem(B)", "requested TPTG"
    );
    for v in VARIANTS.iter().chain(F32_VARIANTS) {
        let p = rt.pipeline(v.kernel)?;
        println!(
            "{:<32}{:>12}{:>14}{:>16}",
            v.kernel,
            p.maxTotalThreadsPerThreadgroup(),
            p.staticThreadgroupMemoryLength(),
            p.threadExecutionWidth() * v.nsg
        );
    }
    println!(
        "\ninterleaved A/B: {rounds} rounds (order reversed every other round), {iters} timed calls per arm \
         per round after {warmup} warmup; times are medians of per-round medians, ratios are medians of \
         per-round production/variant ratios [range]"
    );

    // `BENCH_ONLY=a,b` runs only the shapes whose label contains one of them.
    let only: Option<Vec<String>> = std::env::var("BENCH_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    // `BENCH_LANES=bf16` or `=f32` runs one lane; both by default.
    let lanes = std::env::var("BENCH_LANES").unwrap_or_else(|_| "bf16,f32".into());
    let mut unstable = Vec::new();
    for &(m, n, k, label) in SHAPES {
        if let Some(only) = &only {
            if !only.iter().any(|p| label.contains(p.as_str())) {
                continue;
            }
        }
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        a.buffer.write_f32(&common::fill(m * k, 1));
        b.buffer.write_f32(&common::fill(k * n, 2));
        if lanes.split(',').any(|l| l == "bf16") {
            let a_bf = cast_f32_to_bf16(&a)?;
            let b_bf = cast_f32_to_bf16(&b)?;
            if !sweep(
                &rt,
                "bf16",
                VARIANTS,
                &a_bf,
                &b_bf,
                (m, n, k),
                label,
                warmup,
                iters,
                rounds,
            )? {
                unstable.push(format!("{label} [bf16]"));
            }
        }
        if lanes.split(',').any(|l| l == "f32")
            && !sweep(
                &rt,
                "exact f32",
                F32_VARIANTS,
                &a,
                &b,
                (m, n, k),
                label,
                warmup,
                iters,
                rounds,
            )?
        {
            unstable.push(format!("{label} [exact f32]"));
        }
    }
    common::exit_if_unstable("bench_gemm_tile_tune", &unstable);
    Ok(())
}

/// Production `gemm` on `a`, `b` (bf16 or f32 operands pick the kernel) and
/// every variant, each checked against production's output first, timed as
/// the arms of one interleaved A/B. `false` when production's own spread
/// across rounds broke the 10% limit.
fn sweep(
    rt: &std::sync::Arc<GpuRuntime>,
    lane: &str,
    variants: &[Variant],
    a: &Tensor,
    b: &Tensor,
    (m, n, k): (usize, usize, usize),
    label: &str,
    warmup: usize,
    iters: usize,
    rounds: usize,
) -> Result<bool, String> {
    let c_ref = rt.alloc_tensor_f32(&[m, n])?;
    gemm(a, b, &c_ref, GemmBackend::TensorOps)?;
    rt.synchronize()?;
    let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
    let refmax = refv.iter().fold(0f32, |acc, x| acc.max(x.abs())) as f64;
    let flop = 2.0 * m as f64 * n as f64 * k as f64;

    let mut runnable: Vec<(&Variant, Tensor, f64)> = Vec::new();
    let mut skipped: Vec<&Variant> = Vec::new();
    for v in variants {
        if m % v.sm != 0 || n % v.sn != 0 || k % v.bk != 0 {
            skipped.push(v);
            continue;
        }
        let c = rt.alloc_tensor_f32(&[m, n])?;
        run_variant(rt, v, a, b, &c, m, n, k)?;
        rt.synchronize()?;
        let err = c.buffer.read_f32()[..m * n]
            .iter()
            .zip(&refv)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0, f64::max)
            / refmax;
        runnable.push((v, c, err));
    }

    let mut arms: Vec<common::Arm> = Vec::with_capacity(1 + runnable.len());
    arms.push(common::Arm {
        name: "production".into(),
        reps: 1,
        run: Box::new(|| {
            gemm(a, b, &c_ref, GemmBackend::TensorOps)?;
            rt.synchronize()
        }),
    });
    for (v, c, _) in &runnable {
        arms.push(common::Arm {
            name: v.kernel.into(),
            reps: 1,
            run: Box::new(move || {
                run_variant(rt, v, a, b, c, m, n, k)?;
                rt.synchronize()
            }),
        });
    }
    let paired = common::run_paired(&mut arms, warmup, iters, rounds)?;
    drop(arms);

    let prod_rounds = &paired.rounds[0];
    let prod = common::median(prod_rounds.clone())?;
    let spread = common::self_spread(prod_rounds);
    println!(
        "\n{label} [{lane}]  M={m} N={n} K={k}   production {prod:.3} ms  {:.0} GFLOP/s  (self-spread {:.1}%)",
        flop / (prod * 1e6),
        spread * 100.0
    );
    if spread > common::SELF_SPREAD_LIMIT {
        println!("  EXCEEDS 10%: ratios below are not comparable");
    }
    println!(
        "  {:<32}{:>10}{:>12}{:>9}{:>15}{:>12}",
        "variant", "ms", "GFLOP/s", "vs prod", "range", "max_rel_err"
    );
    for ((v, _, err), rounds_ms) in runnable.iter().zip(&paired.rounds[1..]) {
        let med = common::median(rounds_ms.clone())?;
        let (ratio, lo, hi) = common::ratio_stats(prod_rounds, rounds_ms)?;
        println!(
            "  {:<32}{:>10.3}{:>12.0}{:>8.2}×   [{lo:.2}–{hi:.2}]{:>12.2e}",
            v.kernel,
            med,
            flop / (med * 1e6),
            ratio,
            err
        );
    }
    for v in skipped {
        println!(
            "  {:<32}{:>10}",
            v.kernel, "skip(div): the shape is not a multiple of the tile"
        );
    }
    Ok(spread <= common::SELF_SPREAD_LIMIT)
}
