//! Coop-round A/B for the lanes the NN landing left open: TN, NT, TN-accum
//! bf16 kernels (training backward), and an NN grid swizzle for the
//! large-square operand-reread question. Production baselines go through the
//! public gemm_* API under PrecisionMode::Bf16 with pre-cast operands (the
//! BWD_CAST_ONCE steady state), the accumulate lanes through
//! `GemmOperands::Bf16`; variants dispatch raw kernels.
//!
//! Production and every variant of a shape are arms of one interleaved A/B
//! (`common::run_paired`): `BENCH_ROUNDS` rounds, the arm order reversed
//! every other round, `BENCH_ITERS` timed calls per arm per round. A variant
//! line's ratio is the median of its per-round production/variant ratios.
//! Variants absent from the metallib (a build without `TESSL_GEMM_TUNE=1`)
//! stop the run with exit status 2 before anything is timed.

// A tuning sweep takes the full GEMM shape plus the tile parameters it is
// sweeping. Bundling them would add a struct that every call site unpacks.
#![allow(clippy::too_many_arguments)]

mod common;

use objc2_metal::MTLComputePipelineState;
use tessl::gemm::{
    cast_f32_to_bf16, gemm, gemm_nt_f32, gemm_nt_train, gemm_tn_f32, gemm_tn_train, GemmBackend, GemmOperands,
};
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

struct Variant {
    kernel: &'static str,
    sm: usize,
    sn: usize,
    nsg: usize,
    /// Kernel takes tiles_m at buffer(7) (swizzle signature).
    binds_tiles_m: bool,
    /// Zero C first in the same binder, as production does for the
    /// exact-f32 `mode::multiply` kernels.
    zero_first: bool,
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
    let zero_p = rt.pipeline("zero_f32")?;
    let tiles_n = n / v.sn;
    let tiles_m = m / v.sm;
    let tg = tiles_n * tiles_m;
    let tpt = p.threadExecutionWidth() * v.nsg;
    let numel = c.numel();
    let z_tpt = zero_p.threadExecutionWidth().min(numel).max(1);
    rt.with_binder(|bnd| {
        if v.zero_first {
            bnd.set_pipeline(&zero_p);
            bnd.bind_tensor(c, 0);
            bnd.bind_u32(numel as u32, 1);
            bnd.dispatch(mtl_size(numel.div_ceil(z_tpt), 1, 1), mtl_size(z_tpt, 1, 1));
            if bnd.needs_explicit_barriers() {
                bnd.barrier();
            }
        }
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
    /// The exact-f32 kernels on f32 operands: production through `gemm`,
    /// `gemm_tn_f32`, `gemm_nt_f32` and `GemmOperands::ExactF32`'s
    /// accumulates, against the production kernel dispatched raw in one
    /// dispatch (production may split K).
    NnF32,
    TnF32,
    NtF32,
    NnAccF32,
    TnAccF32,
    NtAccF32,
    /// One panel-walk policy: production's walk (`tile_walk` for every
    /// family) against the production coop kernel under the grid gate the NN
    /// kernels used until 2026-10-11 (`_walkg`).
    WalkNn,
    WalkTn,
    WalkNt,
    WalkTnAcc,
    WalkNtAcc,
    /// Exact-f32 NN: production (`gemm`) against the production 32x32 kernel
    /// in one dispatch under each walk, and the cooperative exact-f32 tiles.
    F32NnWalk,
    /// The int8 NN kernel through `nn::gemm_i8_dequant`: production only, as
    /// for the f32 lanes.
    NnI8,
}

/// Time `nn::gemm_i8_dequant` (`C[M,N] = (A[M,K] · B[K,N]) · a_scale ·
/// b_scale[n]`, int8 operands filled from [`fill`]) without and with the
/// per-column `b_scale`, as two arms of one interleaved A/B, after checking
/// both outputs exactly against an integer reference at sampled elements
/// (the edges of C and a spread of interior ones): int8 products accumulate
/// exactly in int32, and the scales are the kernel's two f32 multiplies in
/// its order, so anything but equality is an error.
fn bench_i8(
    rt: &std::sync::Arc<GpuRuntime>,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
    warmup: usize,
    iters: usize,
    rounds: usize,
) -> Result<(), String> {
    const A_SCALE: f32 = 0.01;
    let int8 =
        |len: usize, seed: u64| -> Vec<i8> { fill(len, seed).iter().map(|&x| (x * 127.0).round() as i8).collect() };
    let (a_host, b_host) = (int8(m * k, 1), int8(k * n, 2));
    let scale_host: Vec<f32> = fill(n, 3).iter().map(|x| 0.5 + x.abs()).collect();
    let bytes = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    let a = rt.alloc_buffer(m * k)?;
    a.write_bytes(&bytes(&a_host));
    let b = rt.alloc_buffer(k * n)?;
    b.write_bytes(&bytes(&b_host));
    let scale = rt.alloc_buffer(n * 4)?;
    scale.write_f32(&scale_host);
    let (c, c_scaled) = (rt.alloc_buffer(m * n * 4)?, rt.alloc_buffer(m * n * 4)?);
    let (mu, nu, ku) = (m as u32, n as u32, k as u32);

    gemm_i8_dequant(rt, &a, &b, &c, mu, nu, ku, A_SCALE, None)?;
    gemm_i8_dequant(rt, &a, &b, &c_scaled, mu, nu, ku, A_SCALE, Some(&scale))?;
    rt.synchronize()?;
    let (got, got_scaled) = (c.read_f32(), c_scaled.read_f32());
    let mut g = common::Lcg::new((m * n + k) as u64);
    let mut points: Vec<(usize, usize)> = vec![(0, 0), (m - 1, n - 1), (0, n - 1), (m - 1, 0)];
    points.extend((0..m).step_by(m.div_ceil(16).max(1)).map(|i| (i, n - 1)));
    points.extend((0..n).step_by(n.div_ceil(16).max(1)).map(|j| (m - 1, j)));
    points.extend((0..1024).map(|_| {
        (
            (g.next_u01() * (m - 1) as f64) as usize,
            (g.next_u01() * (n - 1) as f64) as usize,
        )
    }));
    for &(i, j) in &points {
        let acc: i32 = (0..k)
            .map(|p| a_host[i * k + p] as i32 * b_host[p * n + j] as i32)
            .sum();
        let want = acc as f32 * A_SCALE;
        let want_scaled = acc as f32 * A_SCALE * scale_host[j];
        if got[i * n + j] != want || got_scaled[i * n + j] != want_scaled {
            return Err(format!(
                "{label}: int8 output wrong at ({i}, {j}): {} (want {want}), with b_scale {} (want {want_scaled})",
                got[i * n + j],
                got_scaled[i * n + j]
            ));
        }
    }

    let mut arms = [
        common::Arm {
            name: "production".into(),
            reps: 1,
            run: Box::new(|| {
                gemm_i8_dequant(rt, &a, &b, &c, mu, nu, ku, A_SCALE, None)?;
                rt.synchronize()
            }),
        },
        common::Arm {
            name: "b_scale".into(),
            reps: 1,
            run: Box::new(|| {
                gemm_i8_dequant(rt, &a, &b, &c_scaled, mu, nu, ku, A_SCALE, Some(&scale))?;
                rt.synchronize()
            }),
        },
    ];
    let paired = common::run_paired(&mut arms, warmup, iters, rounds)?;
    let ms = common::median(paired.rounds[0].clone())?;
    let ms_scaled = common::median(paired.rounds[1].clone())?;
    let (ratio, lo, hi) = common::ratio_stats(&paired.rounds[0], &paired.rounds[1])?;
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "\n{label}  M={m} N={n} K={k}   production {ms:.3} ms  {:.0} GOP/s  (self-spread {:.1}%, exact at {} sampled outputs)",
        flop / (ms * 1e6),
        common::self_spread(&paired.rounds[0]) * 100.0,
        points.len()
    );
    println!(
        "  {:<36}{:>10.3}{:>12.0}{:>8.2}× [{lo:.2}–{hi:.2}]",
        "b_scale (per-column scale)",
        ms_scaled,
        flop / (ms_scaled * 1e6),
        ratio
    );
    Ok(())
}

impl Lane {
    fn is_tn(self) -> bool {
        matches!(
            self,
            Lane::Tn
                | Lane::TnAccum
                | Lane::TnPanel
                | Lane::TnAccPanel
                | Lane::TnF32
                | Lane::TnAccF32
                | Lane::WalkTn
                | Lane::WalkTnAcc
        )
    }

    fn is_f32(self) -> bool {
        matches!(
            self,
            Lane::NnF32
                | Lane::TnF32
                | Lane::NtF32
                | Lane::NnAccF32
                | Lane::TnAccF32
                | Lane::NtAccF32
                | Lane::F32NnWalk
        )
    }

    fn is_accum(self) -> bool {
        matches!(
            self,
            Lane::TnAccum
                | Lane::NtAccum
                | Lane::TnAccPanel
                | Lane::NtAccPanel
                | Lane::NnAccF32
                | Lane::TnAccF32
                | Lane::NtAccF32
                | Lane::WalkTnAcc
                | Lane::WalkNtAcc
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
        zero_first: false,
    }
}

/// A tune exact-f32 32x32 kernel (`mm_f32_exact_32x32_*`): C zeroed first, as
/// production's `mode::multiply` kernel needs.
fn exact_f32_walk(kernel: &'static str) -> Variant {
    Variant {
        kernel,
        sm: 32,
        sn: 32,
        nsg: 1,
        binds_tiles_m: true,
        zero_first: true,
    }
}

/// An exact-f32 production kernel dispatched raw in one dispatch: 32x32
/// tiles on one simdgroup, C zeroed first unless it accumulates.
fn exact_f32(kernel: &'static str, accumulates: bool) -> Variant {
    Variant {
        kernel,
        sm: 32,
        sn: 32,
        nsg: 1,
        binds_tiles_m: true,
        zero_first: !accumulates,
    }
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    rt.set_precision(PrecisionMode::Bf16);
    let warmup = common::env_usize("BENCH_WARMUP", 8, 0)?;
    let iters = common::env_usize("BENCH_ITERS", 5, 1)?;
    let rounds = common::env_usize("BENCH_ROUNDS", 6, 1)?;

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
        // TN/NT accumulate kernels; production is `GemmOperands::Bf16`'s
        // `tn_acc` / `nt_acc`, the path training takes.
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
        // The exact-f32 kernels at the Qwen3.5-2B training shapes (T = 2048,
        // hidden 2048, MLP 6144): forward NN (up/gate, down, a square
        // projection), the weight gradients (TN) and input gradients (NT),
        // and the accumulate forms the backward adds into banks with.
        (Lane::NnF32, 2048, 6144, 2048, "f32int_nn_mlp_up"),
        (Lane::NnF32, 2048, 2048, 6144, "f32int_nn_mlp_down"),
        (Lane::NnF32, 2048, 2048, 2048, "f32int_nn_proj"),
        (Lane::TnF32, 2048, 6144, 2048, "f32int_tn_dw_up"),
        (Lane::TnF32, 6144, 2048, 2048, "f32int_tn_dw_down"),
        (Lane::NtF32, 2048, 2048, 6144, "f32int_nt_dx_up"),
        (Lane::NtF32, 2048, 6144, 2048, "f32int_nt_dx_down"),
        (Lane::NnAccF32, 2048, 2048, 2048, "f32int_nnacc_proj"),
        (Lane::TnAccF32, 2048, 6144, 2048, "f32int_tnacc_dw_up"),
        (Lane::NtAccF32, 2048, 2048, 6144, "f32int_ntacc_dx_up"),
        // One panel-walk policy. NN coop: where the grid gate (>= 2048 tiles
        // of 128x64 -> 8-row panels) and `tile_walk` (B >= 2^23 elements ->
        // 4-row panels; Morton kept) disagree, and a control where they agree.
        (Lane::WalkNn, 4096, 4096, 1024, "walk_nn_grid_only"),
        (Lane::WalkNn, 4096, 4096, 4096, "walk_nn_both"),
        (Lane::WalkNn, 8192, 4096, 4096, "walk_nn_morton_sq64"),
        (Lane::WalkNn, 2048, 8192, 2048, "walk_nn_b16m_grid2048"),
        (Lane::WalkNn, 1024, 8192, 2048, "walk_nn_b_only"),
        (Lane::WalkNn, 2048, 6144, 2048, "walk_nn_q35_mlp_up"),
        (Lane::WalkNn, 2048, 2048, 6144, "walk_nn_q35_mlp_down"),
        (Lane::WalkNn, 8192, 3072, 768, "walk_nn_mlp_up"),
        (Lane::WalkNn, 4096, 50304, 768, "walk_nn_lmhead_50304"),
        (Lane::WalkNn, 1024, 248320, 2048, "walk_nn_lm_head"),
        (Lane::WalkNn, 2048, 2048, 2048, "walk_nn_control"),
        // TN / NT / accumulate: production `tile_walk` against the grid gate.
        (Lane::WalkNt, 4096, 8192, 2048, "walk_nt_both"),
        (Lane::WalkNt, 4096, 4096, 1024, "walk_nt_grid_only"),
        (Lane::WalkNt, 4096, 50304, 768, "walk_nt_lmhead_50304"),
        (Lane::WalkNt, 2048, 6144, 2048, "walk_nt_q35_dx_down"),
        (Lane::WalkTn, 2048, 8192, 4096, "walk_tn_both"),
        (Lane::WalkTn, 768, 50304, 4096, "walk_tn_lmhead_dw"),
        (Lane::WalkTn, 2048, 6144, 2048, "walk_tn_q35_dw_up"),
        (Lane::WalkTn, 2048, 4096, 1024, "walk_tn_control"),
        (Lane::WalkNtAcc, 4096, 4096, 1024, "walk_ntacc_morton_grid"),
        (Lane::WalkNtAcc, 3072, 3072, 1024, "walk_ntacc_grid_only"),
        (Lane::WalkTnAcc, 4096, 4096, 1024, "walk_tnacc_morton_grid"),
        (Lane::WalkTnAcc, 3072, 3072, 1024, "walk_tnacc_grid_only"),
        // Exact f32 NN at the 2B training shapes (T = 2048): the MLP up/gate
        // shape against the square and down shapes.
        (Lane::F32NnWalk, 2048, 6144, 2048, "f32walk_q35_mlp_up"),
        (Lane::F32NnWalk, 2048, 2048, 2048, "f32walk_square_2048"),
        (Lane::F32NnWalk, 2048, 2048, 6144, "f32walk_q35_mlp_down"),
        (Lane::F32NnWalk, 2048, 8224, 2048, "f32walk_q35_gdn_in"),
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
    // Each accumulate list leads with the production kernel dispatched raw: a
    // control whose ratio to the `GemmOperands` production arm shows the noise.
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
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_coop_128x64_sg4_swz8",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_coop_256x64_sg8_swz4",
            sm: 256,
            sn: 64,
            nsg: 8,
            binds_tiles_m: true,
            zero_first: false,
        },
    ];
    let tn_variants = &[
        Variant {
            kernel: "mm_bf16_tn_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_tn_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
    ];
    let nt_variants = &[
        Variant {
            kernel: "mm_bf16_nt_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x64_sg4",
            sm: 128,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_64x128_sg4",
            sm: 64,
            sn: 128,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x128_sg4",
            sm: 128,
            sn: 128,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x128_sg8",
            sm: 128,
            sn: 128,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x64_sg8",
            sm: 256,
            sn: 64,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_64x256_sg8",
            sm: 64,
            sn: 256,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_128x256_sg8",
            sm: 128,
            sn: 256,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x128_sg8",
            sm: 256,
            sn: 128,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x64_sg4",
            sm: 256,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_256x32_sg8",
            sm: 256,
            sn: 32,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x32_sg8",
            sm: 512,
            sn: 32,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x64_sg8",
            sm: 512,
            sn: 64,
            nsg: 8,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_512x64_sg16",
            sm: 512,
            sn: 64,
            nsg: 16,
            binds_tiles_m: false,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_coop_1024x32_sg16",
            sm: 1024,
            sn: 32,
            nsg: 16,
            binds_tiles_m: false,
            zero_first: false,
        },
    ];
    let tnacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_tn_accum_bf16_f32",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_tn_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
    ];
    let ntacc_variants = &[
        Variant {
            kernel: "matmul2d_tensorops_nt_accum_bf16_f32",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: true,
            zero_first: false,
        },
        Variant {
            kernel: "mm_bf16_nt_accum_coop_64x64_sg4",
            sm: 64,
            sn: 64,
            nsg: 4,
            binds_tiles_m: false,
            zero_first: false,
        },
    ];

    // Each exact-f32 production kernel raw, in one dispatch. (These lanes
    // decided `TESSL_GEMM_INTERIOR`, with a second arm per kernel that set
    // the retired interior branch: bench/results/gemm_interior_m5pro.txt.)
    let pair = |kernel: &'static str, acc: bool| [exact_f32(kernel, acc)];
    let f32_nn_variants = pair("matmul2d_tensorops_f32", false);
    let f32_tn_variants = pair("matmul2d_tensorops_tn_f32", false);
    let f32_nt_variants = pair("matmul2d_tensorops_nt_f32", false);
    let f32_nnacc_variants = pair("matmul2d_tensorops_nn_accum_f32", true);
    let f32_tnacc_variants = pair("matmul2d_tensorops_tn_accum_f32", true);
    let f32_ntacc_variants = pair("matmul2d_tensorops_nt_accum_f32", true);
    let walk_nn_variants = [
        variant("mm_bf16_coop_128x64_sg4_walkg", 128, 64, true),
        variant("mm_bf16_coop_64x64_sg4_walkg", 64, 64, true),
    ];
    let walk_tn_variants = [variant("mm_bf16_tn_coop_128x64_sg4_walkg", 128, 64, true)];
    let walk_nt_variants = [variant("mm_bf16_nt_coop_128x64_sg4_walkg", 128, 64, true)];
    let walk_tnacc_variants = [variant("mm_bf16_tn_accum_coop_64x64_sg4_walkg", 64, 64, true)];
    let walk_ntacc_variants = [variant("mm_bf16_nt_accum_coop_64x64_sg4_walkg", 64, 64, true)];
    let f32_walk_variants = [
        exact_f32("matmul2d_tensorops_f32", false),
        exact_f32_walk("mm_f32_exact_32x32_linear"),
        exact_f32_walk("mm_f32_exact_32x32_rowmajor"),
        exact_f32_walk("mm_f32_exact_32x32_ph8"),
        exact_f32_walk("mm_f32_exact_32x32_ph16"),
        exact_f32_walk("mm_f32_exact_32x32_ph32"),
        Variant {
            nsg: 4,
            ..variant("mm_f32_exact_coop_128x64_sg4", 128, 64, false)
        },
        Variant {
            nsg: 8,
            ..variant("mm_f32_exact_coop_128x64_sg8", 128, 64, false)
        },
        Variant {
            nsg: 4,
            ..variant("mm_f32_exact_coop_64x64_sg4", 64, 64, false)
        },
    ];
    let lane_variants = |lane: Lane| -> &[Variant] {
        match lane {
            Lane::Nn => nn_variants,
            Lane::Tn => tn_variants,
            Lane::Nt => nt_variants,
            Lane::TnAccum => tnacc_variants,
            Lane::NtAccum => ntacc_variants,
            Lane::TnPanel => tnp_variants,
            Lane::NtPanel => ntp_variants,
            Lane::TnAccPanel => tnap_variants,
            Lane::NtAccPanel => ntap_variants,
            Lane::NnF32 => &f32_nn_variants,
            Lane::TnF32 => &f32_tn_variants,
            Lane::NtF32 => &f32_nt_variants,
            Lane::NnAccF32 => &f32_nnacc_variants,
            Lane::TnAccF32 => &f32_tnacc_variants,
            Lane::NtAccF32 => &f32_ntacc_variants,
            Lane::WalkNn => &walk_nn_variants,
            Lane::WalkTn => &walk_tn_variants,
            Lane::WalkNt => &walk_nt_variants,
            Lane::WalkTnAcc => &walk_tnacc_variants,
            Lane::WalkNtAcc => &walk_ntacc_variants,
            Lane::F32NnWalk => &f32_walk_variants,
            Lane::NnI8 => &[],
        }
    };
    let selected = |label: &str| {
        only.as_ref()
            .is_none_or(|only| only.iter().any(|p| label.contains(p.as_str())))
    };
    let runs = |v: &Variant| !production_only || v.kernel.starts_with("matmul2d_tensorops");
    // Every variant this run will dispatch must be in the metallib before
    // anything is timed.
    common::require_kernels(
        &rt,
        "bench_gemm_tnnt_tune",
        cases
            .iter()
            .filter(|c| selected(c.4))
            .flat_map(|c| lane_variants(c.0).iter().filter(|v| runs(v)).map(|v| v.kernel)),
    );
    println!(
        "interleaved A/B: {rounds} rounds (order reversed every other round), {iters} timed calls per arm \
         per round after {warmup} warmup; times are medians of per-round medians, ratios are medians of \
         per-round production/variant ratios [range]"
    );

    let mut unstable = Vec::new();
    for &(lane, m, n, k, label) in cases {
        if !selected(label) {
            continue;
        }
        if lane == Lane::NnI8 {
            bench_i8(&rt, m, n, k, label, warmup, iters, rounds)?;
            continue;
        }
        // Operand storage per lane: NN A[M,K] B[K,N]; TN A[K,M] B[K,N]; NT A[M,K] B[N,K].
        let (a_shape, b_shape) = match lane {
            Lane::Nn | Lane::NnF32 | Lane::NnAccF32 | Lane::WalkNn | Lane::F32NnWalk => ([m, k], [k, n]),
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

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let prefill = 0.25f32;
        let is_accum = lane.is_accum();
        let prefill_host = vec![prefill; m * n];
        // Only the old accumulate lanes reset C inside the timed loop; every
        // arm of the shape does, so the host write is in all of them alike.
        let reset_each_iter = matches!(lane, Lane::TnAccum | Lane::NtAccum);

        // Reference: production's own output, or for an accumulate lane the
        // prefill plus the product of the matching overwriting train path.
        let refv = {
            let r = rt.alloc_tensor_f32(&[m, n])?;
            match lane {
                Lane::Nn | Lane::NnF32 | Lane::NnAccF32 | Lane::WalkNn | Lane::F32NnWalk => {
                    gemm(&a, &b, &r, GemmBackend::TensorOps)?
                }
                Lane::TnF32 | Lane::TnAccF32 => gemm_tn_f32(&a, &b, &r, GemmBackend::TensorOps)?,
                Lane::NtF32 | Lane::NtAccF32 => gemm_nt_f32(&a, &b, &r, GemmBackend::TensorOps)?,
                _ if lane.is_tn() => gemm_tn_train(&a, &b, &r, GemmBackend::TensorOps)?,
                _ => gemm_nt_train(&a, &b, &r, GemmBackend::TensorOps)?,
            }
            rt.synchronize()?;
            let base = r.buffer.read_f32()[..m * n].to_vec();
            if is_accum {
                base.iter().map(|x| x + prefill).collect::<Vec<_>>()
            } else {
                base
            }
        };

        // Production through the public API: the accumulate lanes through
        // `GemmOperands::Bf16`, the path training takes.
        let prod_c = rt.alloc_tensor_f32(&[m, n])?;
        let production = |c: &Tensor| -> Result<(), String> {
            match lane {
                Lane::Nn | Lane::WalkNn | Lane::F32NnWalk => gemm(&a, &b, c, GemmBackend::TensorOps),
                Lane::Tn | Lane::TnPanel | Lane::WalkTn => gemm_tn_train(&a, &b, c, GemmBackend::TensorOps),
                Lane::Nt | Lane::NtPanel | Lane::WalkNt => gemm_nt_train(&a, &b, c, GemmBackend::TensorOps),
                Lane::TnF32 => gemm_tn_f32(&a, &b, c, GemmBackend::TensorOps),
                Lane::NtF32 => gemm_nt_f32(&a, &b, c, GemmBackend::TensorOps),
                Lane::NnF32 => gemm(&a, &b, c, GemmBackend::TensorOps),
                Lane::NnAccF32 => GemmOperands::ExactF32.nn_acc(&a, &b, c),
                Lane::TnAccF32 => GemmOperands::ExactF32.tn_acc(&a, &b, c),
                Lane::NtAccF32 => GemmOperands::ExactF32.nt_acc(&a, &b, c),
                Lane::TnAccum | Lane::TnAccPanel | Lane::WalkTnAcc => GemmOperands::Bf16.tn_acc(&a, &b, c),
                Lane::NtAccum | Lane::NtAccPanel | Lane::WalkNtAcc => GemmOperands::Bf16.nt_acc(&a, &b, c),
                Lane::NnI8 => unreachable!("int8 cases are timed by bench_i8"),
            }
        };

        let mut variants: Vec<(&Variant, Tensor, f64)> = Vec::new();
        let mut skipped: Vec<&Variant> = Vec::new();
        for v in lane_variants(lane).iter().filter(|v| runs(v)) {
            if m % v.sm != 0 || n % v.sn != 0 {
                skipped.push(v);
                continue;
            }
            variants.push((v, rt.alloc_tensor_f32(&[m, n])?, 0.0));
        }

        // One untimed call of every arm, checked against the reference.
        if is_accum {
            prod_c.buffer.write_f32(&prefill_host);
        }
        production(&prod_c)?;
        rt.synchronize()?;
        let prod_err = rel_err(&prod_c.buffer.read_f32()[..m * n], &refv);
        for (v, c, err) in variants.iter_mut() {
            if is_accum {
                c.buffer.write_f32(&prefill_host);
            }
            dispatch_variant(&rt, v, &a, &b, c, m, n, k)?;
            rt.synchronize()?;
            *err = rel_err(&c.buffer.read_f32()[..m * n], &refv);
        }

        let mut arms: Vec<common::Arm> = Vec::with_capacity(1 + variants.len());
        arms.push(common::Arm {
            name: "production".into(),
            reps: 1,
            run: Box::new(|| {
                if reset_each_iter {
                    prod_c.buffer.write_f32(&prefill_host);
                }
                production(&prod_c)?;
                rt.synchronize()
            }),
        });
        for (v, c, _) in &variants {
            let (rt, a, b, prefill_host) = (&rt, &a, &b, &prefill_host);
            arms.push(common::Arm {
                name: v.kernel.into(),
                reps: 1,
                run: Box::new(move || {
                    if reset_each_iter {
                        c.buffer.write_f32(prefill_host);
                    }
                    dispatch_variant(rt, v, a, b, c, m, n, k)?;
                    rt.synchronize()
                }),
            });
        }
        let paired = common::run_paired(&mut arms, warmup, iters, rounds)?;
        drop(arms);

        let prod_rounds = &paired.rounds[0];
        let prod_ms = common::median(prod_rounds.clone())?;
        let spread = common::self_spread(prod_rounds);
        println!(
            "\n{label}  M={m} N={n} K={k}   production {prod_ms:.3} ms  {:.0} GFLOP/s  (self-spread {:.1}%, err {prod_err:.2e})",
            flop / (prod_ms * 1e6),
            spread * 100.0
        );
        if spread > common::SELF_SPREAD_LIMIT {
            println!("  EXCEEDS 10%: ratios below are not comparable");
            unstable.push(label.to_string());
        }
        for ((v, _, err), rounds_ms) in variants.iter().zip(&paired.rounds[1..]) {
            let med = common::median(rounds_ms.clone())?;
            let (ratio, lo, hi) = common::ratio_stats(prod_rounds, rounds_ms)?;
            println!(
                "  {:<36}{:>10.3}{:>12.0}{:>8.2}× [{lo:.2}–{hi:.2}]{:>12.2e}",
                v.kernel,
                med,
                flop / (med * 1e6),
                ratio,
                err
            );
        }
        for v in skipped {
            println!(
                "  {:<36}{:>10}",
                v.kernel, "skip(div): the shape is not a multiple of the tile"
            );
        }
    }
    common::exit_if_unstable("bench_gemm_tnnt_tune", &unstable);
    Ok(())
}
