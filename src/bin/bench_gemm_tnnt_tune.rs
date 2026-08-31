//! Coop-round A/B for the lanes the NN landing left open: TN, NT, TN-accum
//! bf16 kernels (training backward), and an NN grid swizzle for the
//! large-square operand-reread question. Production baselines go through the
//! public gemm_* API under PrecisionMode::Bf16 with pre-cast operands (the
//! BWD_CAST_ONCE steady state); variants dispatch raw kernels.

use tessl::gemm::{
    cast_f32_to_bf16, gemm, gemm_tn_train, gemm_nt_train, GemmBackend,
};
use tessl::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
    }).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

struct Variant {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    nsg: usize,
    /// Kernel takes tiles_m at buffer(7) (swizzle signature).
    binds_tiles_m: bool,
}

fn dispatch_variant(
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
    let tiles_n = n / v.sn;
    let tiles_m = m / v.sm;
    let tg = tiles_n * tiles_m;
    let tpt = p.threadExecutionWidth() as usize * v.nsg;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_buf(a.buffer.metal(), a.byte_offset, 0);
        bnd.bind_buf(b.buffer.metal(), b.byte_offset, 1);
        bnd.bind_buf(c.buffer.metal(), c.byte_offset, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        if v.binds_tiles_m {
            bnd.bind_u32(tiles_m as u32, 7);
        }
        bnd.dispatch(mtl_size(tg, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

fn time_it(mut f: impl FnMut() -> Result<(), String>, warmup: usize, iters: usize)
    -> Result<f64, String>
{
    for _ in 0..warmup { f()?; }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        s.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median(s))
}

fn rel_err(got: &[f32], reference: &[f32]) -> f64 {
    let scale = reference.iter().fold(0f32, |a, x| a.max(x.abs())) as f64;
    got.iter().zip(reference)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max) / scale.max(1e-12)
}

#[derive(Clone, Copy, PartialEq)]
enum Lane { Nn, Tn, Nt, TnAccum, NtAccum }

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    rt.set_precision(PrecisionMode::Bf16);
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);

    // (lane, m, n, k, label)
    let cases: &[(Lane, usize, usize, usize, &str)] = &[
        // NN swizzle question (production = landed coop + selection table).
        (Lane::Nn, 4096, 4096, 4096, "nn_square_4096"),
        (Lane::Nn, 2048, 2048, 2048, "nn_square_2048"),
        (Lane::Nn, 8192, 3072, 768, "nn_mlp_up"),
        (Lane::Nn, 4096, 4096, 1024, "nn_tall_k1024"),
        (Lane::Nn, 8192, 768, 3072, "nn_mlp_down"),
        // TN: non-split-K descriptor shapes, then the split-K-gated dW shapes.
        (Lane::Tn, 2048, 2048, 2048, "tn_square_2048"),
        (Lane::Tn, 1024, 1024, 4096, "tn_1024_k4096"),
        (Lane::Tn, 512, 768, 4096, "tn_512x768_k4096"),
        (Lane::Tn, 128, 128, 4096, "tn_dw_attn(splitk)"),
        (Lane::Tn, 128, 384, 4096, "tn_dw_mlp(splitk)"),
        // NT: the dx shapes every backward layer runs, plus generic.
        (Lane::Nt, 4096, 128, 384, "nt_dx_mlp_in"),
        (Lane::Nt, 4096, 384, 128, "nt_dx_mlp_hid"),
        (Lane::Nt, 4096, 128, 128, "nt_dx_attn"),
        (Lane::Nt, 2048, 2048, 2048, "nt_square_2048"),
        (Lane::Nt, 8192, 768, 3072, "nt_wide"),
        // TN accumulate kernels (raw A/B; production kernel is the
        // GEMM_ACCUM=1 path, default-off in training).
        (Lane::TnAccum, 512, 768, 4096, "tnacc_512x768_k4096"),
        (Lane::TnAccum, 2048, 2048, 2048, "tnacc_square_2048"),
        (Lane::NtAccum, 4096, 128, 384, "ntacc_dx_mlp_in"),
        (Lane::NtAccum, 2048, 2048, 2048, "ntacc_square_2048"),
    ];

    let nn_variants = &[
        Variant { kernel: "mm_bf16_coop_128x64_sg4_swz4", sm: 128, sn: 64, nsg: 4, binds_tiles_m: true },
        Variant { kernel: "mm_bf16_coop_128x64_sg4_swz8", sm: 128, sn: 64, nsg: 4, binds_tiles_m: true },
        Variant { kernel: "mm_bf16_coop_256x64_sg8_swz4", sm: 256, sn: 64, nsg: 8, binds_tiles_m: true },
    ];
    let tn_variants = &[
        Variant { kernel: "mm_bf16_tn_coop_64x64_sg4", sm: 64, sn: 64, nsg: 4, binds_tiles_m: false },
        Variant { kernel: "mm_bf16_tn_coop_128x64_sg4", sm: 128, sn: 64, nsg: 4, binds_tiles_m: false },
    ];
    let nt_variants = &[
        Variant { kernel: "mm_bf16_nt_coop_64x64_sg4", sm: 64, sn: 64, nsg: 4, binds_tiles_m: false },
        Variant { kernel: "mm_bf16_nt_coop_128x64_sg4", sm: 128, sn: 64, nsg: 4, binds_tiles_m: false },
    ];
    let tnacc_variants = &[
        Variant { kernel: "matmul2d_tensorops_tn_accum_bf16_f32", sm: 64, sn: 32, nsg: 4, binds_tiles_m: true },
        Variant { kernel: "mm_bf16_tn_accum_coop_64x64_sg4", sm: 64, sn: 64, nsg: 4, binds_tiles_m: false },
    ];
    let ntacc_variants = &[
        Variant { kernel: "matmul2d_tensorops_nt_accum_bf16_f32", sm: 64, sn: 32, nsg: 4, binds_tiles_m: true },
        Variant { kernel: "mm_bf16_nt_accum_coop_64x64_sg4", sm: 64, sn: 64, nsg: 4, binds_tiles_m: false },
    ];

    for &(lane, m, n, k, label) in cases {
        // Operand storage per lane: NN A[M,K] B[K,N]; TN A[K,M] B[K,N]; NT A[M,K] B[N,K].
        let (a_shape, b_shape) = match lane {
            Lane::Nn => ([m, k], [k, n]),
            Lane::Tn | Lane::TnAccum => ([k, m], [k, n]),
            Lane::Nt | Lane::NtAccum => ([m, k], [n, k]),
        };
        let a_f = rt.alloc_tensor_f32(&a_shape)?;
        let b_f = rt.alloc_tensor_f32(&b_shape)?;
        a_f.buffer.write_f32(&fill(a_shape[0] * a_shape[1], 1));
        b_f.buffer.write_f32(&fill(b_shape[0] * b_shape[1], 2));
        let a = cast_f32_to_bf16(&a_f)?;
        let b = cast_f32_to_bf16(&b_f)?;
        let c_ref = rt.alloc_tensor_f32(&[m, n])?;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prefill = 0.25f32;

        // Production baseline through the public API (accum: raw kernel, since
        // the public accum path is flag-gated and includes a temp GEMM).
        let prod_ms = match lane {
            Lane::Nn => {
                gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(|| { gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?; rt.synchronize() },
                        warmup, iters)?
            }
            Lane::Tn => {
                gemm_tn_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(|| { gemm_tn_train(&a, &b, &c_ref, GemmBackend::TensorOps)?; rt.synchronize() },
                        warmup, iters)?
            }
            Lane::Nt => {
                gemm_nt_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(|| { gemm_nt_train(&a, &b, &c_ref, GemmBackend::TensorOps)?; rt.synchronize() },
                        warmup, iters)?
            }
            Lane::TnAccum | Lane::NtAccum => {
                // Reference: prefill + product via the matching train path.
                let tmp = rt.alloc_tensor_f32(&[m, n])?;
                if lane == Lane::TnAccum {
                    gemm_tn_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
                } else {
                    gemm_nt_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
                }
                rt.synchronize()?;
                let base = tmp.buffer.read_f32();
                c_ref.buffer.write_f32(&base.iter().map(|x| x + prefill).collect::<Vec<_>>());
                f64::NAN // no API baseline; variants compared against each other below
            }
        };
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
        if prod_ms.is_nan() {
            println!("\n{label}  M={m} N={n} K={k}   (raw accum kernels; ref = TN product + {prefill})");
        } else {
            println!("\n{label}  M={m} N={n} K={k}   production {prod_ms:.3} ms  {:.0} GFLOP/s",
                     flop / (prod_ms * 1e6));
        }

        let variants: &[Variant] = match lane {
            Lane::Nn => nn_variants,
            Lane::Tn => tn_variants,
            Lane::Nt => nt_variants,
            Lane::TnAccum => tnacc_variants,
            Lane::NtAccum => ntacc_variants,
        };
        for v in variants {
            if m % v.sm != 0 || n % v.sn != 0 {
                println!("  {:<36}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            let c = rt.alloc_tensor_f32(&[m, n])?;
            let is_accum = matches!(lane, Lane::TnAccum | Lane::NtAccum);
            let prefill_host = vec![prefill; m * n];
            if is_accum {
                c.buffer.write_f32(&prefill_host);
            }
            if dispatch_variant(&rt, v, &a, &b, &c, m, n, k).is_err() {
                println!("  {:<36}{:>10}", v.kernel, "skip(pipe)");
                continue;
            }
            rt.synchronize()?;
            let err = rel_err(&c.buffer.read_f32()[..m * n], &refv);

            let med = time_it(|| {
                if is_accum {
                    // Accum kernels mutate C; reset so every iter does the same work.
                    c.buffer.write_f32(&prefill_host);
                }
                dispatch_variant(&rt, v, &a, &b, &c, m, n, k)?;
                rt.synchronize()
            }, warmup, iters)?;
            let vs = if prod_ms.is_nan() { String::from("      —") } else { format!("{:>6.2}×", prod_ms / med) };
            println!("  {:<36}{:>10.3}{:>12.0}{}{:>12.2e}",
                     v.kernel, med, flop / (med * 1e6), vs, err);
        }
    }
    Ok(())
}
