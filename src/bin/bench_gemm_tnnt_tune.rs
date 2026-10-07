//! Coop-round A/B for the lanes the NN landing left open: TN, NT, TN-accum
//! bf16 kernels (training backward), and an NN grid swizzle for the
//! large-square operand-reread question. Production baselines go through the
//! public gemm_* API under PrecisionMode::Bf16 with pre-cast operands (the
//! BWD_CAST_ONCE steady state); variants dispatch raw kernels.

// A tuning sweep takes the full GEMM shape plus the tile parameters it is
// sweeping. Bundling them would add a struct that every call site unpacks.
#![allow(clippy::too_many_arguments)]

use objc2_metal::MTLComputePipelineState;
use std::time::Instant;
use tessl::gemm::{cast_f32_to_bf16, gemm, gemm_nt_f32, gemm_nt_train, gemm_tn_f32, gemm_tn_train, GemmBackend};
use tessl::nn::gemm_i8_dequant;
use tessl::runtime::{mtl_size, GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    let tpt = p.threadExecutionWidth() * v.nsg;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
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

fn time_it(mut f: impl FnMut() -> Result<(), String>, warmup: usize, iters: usize) -> Result<f64, String> {
    for _ in 0..warmup {
        f()?;
    }
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
    got.iter()
        .zip(reference)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
        / scale.max(1e-12)
}

#[derive(Clone, Copy, PartialEq)]
enum Lane {
    Nn,
    Tn,
    Nt,
    TnAccum,
    NtAccum,
    /// Large-B TN/NT: the row-major coop walk against the column-panel walk
    /// at 4, 8 and 16 tile rows per band.
    TnPanel,
    NtPanel,
    /// The same for the 64x64 accumulate kernels. Unlike `TnAccum` and
    /// `NtAccum`, C is not reset inside the timed loop: at these sizes the
    /// host write would be most of the time. Each iteration does the same
    /// GPU work; only the values C grows to differ.
    TnAccPanel,
    NtAccPanel,
    /// The exact-f32 TN/NT kernels through `gemm_tn_f32` / `gemm_nt_f32` on
    /// f32 operands: production only, for an A/B of two builds' tile walks.
    TnF32,
    NtF32,
    /// The int8 NN kernel through `nn::gemm_i8_dequant`: production only, as
    /// for the f32 lanes.
    NnI8,
}

/// Time `nn::gemm_i8_dequant` (`C[M,N] = A[M,K] · B[K,N]`, int8 operands
/// filled from [`fill`], no per-column scale) and print its line in the
/// production format of the other lanes.
fn bench_i8(
    rt: &std::sync::Arc<GpuRuntime>,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
    warmup: usize,
    iters: usize,
) -> Result<(), String> {
    let int8 = |len: usize, seed: u64| -> Vec<u8> {
        fill(len, seed)
            .iter()
            .map(|&x| ((x * 127.0).round() as i8) as u8)
            .collect()
    };
    let a = rt.alloc_buffer(m * k)?;
    a.write_bytes(&int8(m * k, 1));
    let b = rt.alloc_buffer(k * n)?;
    b.write_bytes(&int8(k * n, 2));
    let c = rt.alloc_buffer(m * n * 4)?;
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);
    let run = || -> Result<(), String> {
        gemm_i8_dequant(rt, &a, &b, &c, mu, nu, ku, 0.01, None)?;
        rt.synchronize()
    };
    run()?;
    let ms = time_it(run, warmup, iters)?;
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "\n{label}  M={m} N={n} K={k}   production {ms:.3} ms  {:.0} GFLOP/s",
        flop / (ms * 1e6)
    );
    Ok(())
}

impl Lane {
    fn is_tn(self) -> bool {
        matches!(
            self,
            Lane::Tn | Lane::TnAccum | Lane::TnPanel | Lane::TnAccPanel | Lane::TnF32
        )
    }

    fn is_f32(self) -> bool {
        matches!(self, Lane::TnF32 | Lane::NtF32)
    }

    fn is_accum(self) -> bool {
        matches!(
            self,
            Lane::TnAccum | Lane::NtAccum | Lane::TnAccPanel | Lane::NtAccPanel
        )
    }
}

fn variant(kernel: &'static str, sm: usize, sn: usize, binds_tiles_m: bool) -> Variant {
    Variant {
        kernel,
        sm,
        sn,
        nsg: 4,
        binds_tiles_m,
    }
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    rt.set_precision(PrecisionMode::Bf16);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

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
        // The tied LM head at 1024 rows: the NN product over the packed
        // [hidden, vocab] copy, and the NT product over the [vocab, hidden]
        // table the same logits could come from.
        (Lane::Nn, 1024, 248320, 2048, "nn_lm_head"),
        (Lane::Nt, 1024, 248320, 2048, "nt_lm_head"),
        // TN accumulate kernels (raw A/B; production kernel is the
        // GEMM_ACCUM=1 path, default-off in training).
        (Lane::TnAccum, 512, 768, 4096, "tnacc_512x768_k4096"),
        (Lane::TnAccum, 2048, 2048, 2048, "tnacc_square_2048"),
        (Lane::NtAccum, 4096, 128, 384, "ntacc_dx_mlp_in"),
        (Lane::NtAccum, 2048, 2048, 2048, "ntacc_square_2048"),
        // Column-panel walk for the TN/NT coop kernels. B (the operand each
        // tile row re-reads) from 16 MB to 1 GB of bf16, either side of the
        // 32 MiB at which the exact-f32 kernels switch to panels.
        (Lane::NtPanel, 4096, 50304, 768, "ntp_lmhead_50304"),
        (Lane::NtPanel, 4096, 32768, 768, "ntp_vocab_32768"),
        (Lane::NtPanel, 4096, 8192, 2048, "ntp_b32mib"),
        (Lane::NtPanel, 4096, 16384, 768, "ntp_vocab_16384"),
        (Lane::NtPanel, 4096, 6144, 2048, "ntp_mlp_6144"),
        (Lane::NtPanel, 4096, 4096, 2048, "ntp_b16mib"),
        (Lane::NtPanel, 1024, 248320, 2048, "ntp_lm_head_248320"),
        // Long K: a band's A slab (PH * 128 rows of K) outgrows the cache.
        (Lane::NtPanel, 4096, 8192, 8192, "ntp_longk_8192"),
        (Lane::TnPanel, 2048, 8192, 16384, "tnp_longk_16384"),
        (Lane::TnPanel, 768, 50304, 4096, "tnp_lmhead_dw"),
        (Lane::TnPanel, 2048, 8192, 4096, "tnp_b64mib"),
        (Lane::TnPanel, 2048, 4096, 4096, "tnp_b32mib"),
        (Lane::TnPanel, 2048, 2048, 4096, "tnp_b16mib"),
        // A square power-of-two tile grid, which production walks in Morton
        // order rather than row-major, at a B of 32 MiB.
        (Lane::NtPanel, 8192, 4096, 4096, "ntp_morton_sq64"),
        (Lane::TnPanel, 8192, 4096, 4096, "tnp_morton_sq64"),
        (Lane::NtAccPanel, 4096, 32768, 768, "ntap_vocab_32768"),
        (Lane::NtAccPanel, 4096, 4096, 2048, "ntap_b16mib"),
        (Lane::NtAccPanel, 4096, 4096, 4096, "ntap_morton_sq64"),
        (Lane::TnAccPanel, 768, 50304, 4096, "tnap_lmhead_dw"),
        (Lane::TnAccPanel, 2048, 2048, 4096, "tnap_b16mib"),
        (Lane::TnAccPanel, 4096, 4096, 4096, "tnap_morton_sq64"),
        // The B-size gate's bracket, 12 to 24 MiB, on grids that are not
        // square (M = 3072: 24 tile rows of 128, 48 of 64), so production
        // walks them row-major.
        (Lane::NtPanel, 3072, 3072, 2048, "ntp_g12mib"),
        (Lane::NtPanel, 3072, 4096, 2048, "ntp_g16mib"),
        (Lane::NtPanel, 3072, 5120, 2048, "ntp_g20mib"),
        (Lane::NtPanel, 3072, 8192, 768, "ntp_g12mib_k768"),
        (Lane::NtPanel, 3072, 13696, 768, "ntp_g20mib_k768"),
        (Lane::TnPanel, 3072, 1536, 4096, "tnp_g12mib"),
        (Lane::TnPanel, 3072, 2560, 4096, "tnp_g20mib"),
        (Lane::TnPanel, 3072, 3072, 4096, "tnp_g24mib"),
        (Lane::NtAccPanel, 3072, 3072, 2048, "ntap_g12mib"),
        (Lane::NtAccPanel, 3072, 4096, 2048, "ntap_g16mib"),
        (Lane::NtAccPanel, 3072, 5120, 2048, "ntap_g20mib"),
        (Lane::TnAccPanel, 3072, 1536, 4096, "tnap_g12mib"),
        (Lane::TnAccPanel, 3072, 2048, 4096, "tnap_g16mib"),
        (Lane::TnAccPanel, 3072, 2560, 4096, "tnap_g20mib"),
        // Exact f32 at a 32 MiB B: square power-of-two grids of 32x32 tiles
        // (Morton order unless the panel walk overrides it), and a
        // non-square control that takes panels in either build.
        (Lane::NtF32, 4096, 4096, 2048, "f32nt_morton_sq128"),
        (Lane::TnF32, 4096, 4096, 2048, "f32tn_morton_sq128"),
        (Lane::NtF32, 2048, 2048, 4096, "f32nt_morton_sq64"),
        (Lane::TnF32, 2048, 2048, 4096, "f32tn_morton_sq64"),
        (Lane::NtF32, 4096, 16384, 768, "f32nt_vocab_16384"),
        // The int8 NN kernel (128x64 tiles, B = K x N bytes): an LM-head and
        // a vocabulary B, the gate's bracket on grids that are not square,
        // long K, and two controls the walk leaves alone (a Morton grid, and
        // a large grid over a small B).
        (Lane::NnI8, 4096, 50304, 768, "i8nn_lmhead_50304"),
        (Lane::NnI8, 4096, 32768, 768, "i8nn_vocab_32768"),
        (Lane::NnI8, 4096, 8192, 2048, "i8nn_b16m"),
        (Lane::NnI8, 3072, 6144, 2048, "i8nn_b12m"),
        (Lane::NnI8, 3072, 4096, 2048, "i8nn_b8m"),
        (Lane::NnI8, 3072, 3072, 2048, "i8nn_b6m"),
        (Lane::NnI8, 4096, 8192, 8192, "i8nn_longk_8192"),
        (Lane::NnI8, 8192, 4096, 4096, "i8nn_morton_sq64"),
        (Lane::NnI8, 8192, 3072, 768, "i8nn_grid_small_b"),
    ];
    // `BENCH_ONLY=a,b` runs only the cases whose label contains one of them.
    let only: Option<Vec<String>> = std::env::var("BENCH_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    // `BENCH_PRODUCTION_ONLY=1` times the public API and skips the raw kernel
    // variants, for an A/B of two builds' production kernels.
    let production_only = std::env::var_os("BENCH_PRODUCTION_ONLY").is_some();
    let ntp_variants = &[
        variant("mm_bf16_nt_coop_128x64_sg4", 128, 64, false),
        variant("mm_bf16_nt_coop_128x64_sg4_ph4", 128, 64, true),
        variant("mm_bf16_nt_coop_128x64_sg4_ph8", 128, 64, true),
        variant("mm_bf16_nt_coop_128x64_sg4_ph16", 128, 64, true),
    ];
    let tnp_variants = &[
        variant("mm_bf16_tn_coop_128x64_sg4", 128, 64, false),
        variant("mm_bf16_tn_coop_128x64_sg4_ph4", 128, 64, true),
        variant("mm_bf16_tn_coop_128x64_sg4_ph8", 128, 64, true),
        variant("mm_bf16_tn_coop_128x64_sg4_ph16", 128, 64, true),
    ];
    // The production accumulate kernels have no public API that reaches them
    // without an A/B flag, so they lead their lists as raw variants.
    let ntap_variants = &[
        variant("matmul2d_tensorops_nt_accum_bf16_f32", 64, 64, true),
        variant("mm_bf16_nt_accum_coop_64x64_sg4", 64, 64, false),
        variant("mm_bf16_nt_accum_coop_64x64_sg4_ph8", 64, 64, true),
        variant("mm_bf16_nt_accum_coop_64x64_sg4_ph16", 64, 64, true),
    ];
    let tnap_variants = &[
        variant("matmul2d_tensorops_tn_accum_bf16_f32", 64, 64, true),
        variant("mm_bf16_tn_accum_coop_64x64_sg4", 64, 64, false),
        variant("mm_bf16_tn_accum_coop_64x64_sg4_ph8", 64, 64, true),
        variant("mm_bf16_tn_accum_coop_64x64_sg4_ph16", 64, 64, true),
    ];

    let nn_variants = &[
        Variant {
            kernel: "mm_bf16_coop_128x64_sg4_swz4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_coop_128x64_sg4_swz8",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_coop_256x64_sg8_swz4",
            sm: 256,
            sn: 64,
            nsg: 8,
            binds_tiles_m: true,
        },
    ];
    let tn_variants = &[
        Variant {
            kernel: "mm_bf16_tn_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_tn_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];
    let nt_variants = &[
        Variant {
            kernel: "mm_bf16_nt_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_64x128_sg4",
            sm: 64,
            sn: 128,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x128_sg4",
            sm: 128,
            sn: 128,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x128_sg8",
            sm: 128,
            sn: 128,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x64_sg8",
            sm: 256,
            sn: 64,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_64x256_sg8",
            sm: 64,
            sn: 256,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x256_sg8",
            sm: 128,
            sn: 256,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x128_sg8",
            sm: 256,
            sn: 128,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x64_sg4",
            sm: 256,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x32_sg8",
            sm: 256,
            sn: 32,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x32_sg8",
            sm: 512,
            sn: 32,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x64_sg8",
            sm: 512,
            sn: 64,
            nsg: 8,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x64_sg16",
            sm: 512,
            sn: 64,
            nsg: 16,
            binds_tiles_m: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_1024x32_sg16",
            sm: 1024,
            sn: 32,
            nsg: 16,
            binds_tiles_m: false,
        },
    ];
    let tnacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_tn_accum_bf16_f32",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_tn_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];
    let ntacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_nt_accum_bf16_f32",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
        },
        Variant {
            kernel: "mm_bf16_nt_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
        },
    ];

    for &(lane, m, n, k, label) in cases {
        if let Some(only) = &only {
            if !only.iter().any(|p| label.contains(p.as_str())) {
                continue;
            }
        }
        if lane == Lane::NnI8 {
            bench_i8(&rt, m, n, k, label, warmup, iters)?;
            continue;
        }
        // Operand storage per lane: NN A[M,K] B[K,N]; TN A[K,M] B[K,N]; NT A[M,K] B[N,K].
        let (a_shape, b_shape) = match lane {
            Lane::Nn => ([m, k], [k, n]),
            _ if lane.is_tn() => ([k, m], [k, n]),
            _ => ([m, k], [n, k]),
        };
        let a_f = rt.alloc_tensor_f32(&a_shape)?;
        let b_f = rt.alloc_tensor_f32(&b_shape)?;
        a_f.buffer.write_f32(&fill(a_shape[0] * a_shape[1], 1));
        b_f.buffer.write_f32(&fill(b_shape[0] * b_shape[1], 2));
        let (a, b) = if lane.is_f32() {
            (a_f, b_f)
        } else {
            (cast_f32_to_bf16(&a_f)?, cast_f32_to_bf16(&b_f)?)
        };
        let c_ref = rt.alloc_tensor_f32(&[m, n])?;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prefill = 0.25f32;

        // Production baseline through the public API (accum: raw kernel, since
        // the public accum path is flag-gated and includes a temp GEMM).
        let prod_ms = match lane {
            Lane::Nn => {
                gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(
                    || {
                        gemm(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                        rt.synchronize()
                    },
                    warmup,
                    iters,
                )?
            }
            Lane::Tn | Lane::TnPanel => {
                gemm_tn_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(
                    || {
                        gemm_tn_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                        rt.synchronize()
                    },
                    warmup,
                    iters,
                )?
            }
            Lane::Nt | Lane::NtPanel => {
                gemm_nt_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(
                    || {
                        gemm_nt_train(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                        rt.synchronize()
                    },
                    warmup,
                    iters,
                )?
            }
            Lane::TnF32 => {
                gemm_tn_f32(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(
                    || {
                        gemm_tn_f32(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                        rt.synchronize()
                    },
                    warmup,
                    iters,
                )?
            }
            Lane::NtF32 => {
                gemm_nt_f32(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                rt.synchronize()?;
                time_it(
                    || {
                        gemm_nt_f32(&a, &b, &c_ref, GemmBackend::TensorOps)?;
                        rt.synchronize()
                    },
                    warmup,
                    iters,
                )?
            }
            Lane::NnI8 => unreachable!("int8 cases are timed by bench_i8 and skip this loop body"),
            Lane::TnAccum | Lane::NtAccum | Lane::TnAccPanel | Lane::NtAccPanel => {
                // Reference: prefill + product via the matching train path.
                let tmp = rt.alloc_tensor_f32(&[m, n])?;
                if lane.is_tn() {
                    gemm_tn_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
                } else {
                    gemm_nt_train(&a, &b, &tmp, GemmBackend::TensorOps)?;
                }
                rt.synchronize()?;
                let base = tmp.buffer.read_f32();
                c_ref
                    .buffer
                    .write_f32(&base.iter().map(|x| x + prefill).collect::<Vec<_>>());
                f64::NAN // no API baseline; variants compared against each other below
            }
        };
        let refv = c_ref.buffer.read_f32()[..m * n].to_vec();
        if prod_ms.is_nan() {
            println!("\n{label}  M={m} N={n} K={k}   (raw accum kernels; ref = TN product + {prefill})");
        } else {
            println!(
                "\n{label}  M={m} N={n} K={k}   production {prod_ms:.3} ms  {:.0} GFLOP/s",
                flop / (prod_ms * 1e6)
            );
        }

        let variants: &[Variant] = match lane {
            Lane::Nn => nn_variants,
            Lane::Tn => tn_variants,
            Lane::Nt => nt_variants,
            Lane::TnAccum => tnacc_variants,
            Lane::NtAccum => ntacc_variants,
            Lane::TnPanel => tnp_variants,
            Lane::NtPanel => ntp_variants,
            Lane::TnAccPanel => tnap_variants,
            Lane::NtAccPanel => ntap_variants,
            Lane::TnF32 | Lane::NtF32 | Lane::NnI8 => &[],
        };
        // Only the old accumulate lanes reset C inside the timed loop.
        let reset_each_iter = matches!(lane, Lane::TnAccum | Lane::NtAccum);
        for v in variants
            .iter()
            .filter(|v| !production_only || v.kernel.starts_with("matmul2d_tensorops"))
        {
            if m % v.sm != 0 || n % v.sn != 0 {
                println!("  {:<36}{:>10}", v.kernel, "skip(div)");
                continue;
            }
            let c = rt.alloc_tensor_f32(&[m, n])?;
            let is_accum = lane.is_accum();
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

            let med = time_it(
                || {
                    if reset_each_iter {
                        // Accum kernels mutate C; reset so every iter does the same work.
                        c.buffer.write_f32(&prefill_host);
                    }
                    dispatch_variant(&rt, v, &a, &b, &c, m, n, k)?;
                    rt.synchronize()
                },
                warmup,
                iters,
            )?;
            let vs = if prod_ms.is_nan() {
                String::from("      —")
            } else {
                format!("{:>6.2}×", prod_ms / med)
            };
            println!(
                "  {:<36}{:>10.3}{:>12.0}{}{:>12.2e}",
                v.kernel,
                med,
                flop / (med * 1e6),
                vs,
                err
            );
        }
    }
    Ok(())
}
