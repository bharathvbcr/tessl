//! Paired A/B of the bf16 GEMM routing rules at the shapes they gate on, on
//! this tree's kernels.
//!
//! ```text
//! cargo run --release --bin bench_gemm_routes            # every lane
//! BENCH_LANES=splitk cargo run --release --bin bench_gemm_routes
//! ```
//!
//! The method of `bench/results/splitk_gate_m5pro.txt` and
//! `bf16_smallm_coop_m5pro.txt`, which were measured on the MLSystemsLab copy
//! of this crate (imported; see bench/results/PROVENANCE.md): async encode, `BENCH_REPS` GEMMs per command buffer (20) so
//! the clock sees the kernels and not the submit, `BENCH_ITERS` calls per arm
//! per round (8), `BENCH_ROUNDS` rounds (6) with the arm order reversed
//! every other round, after `BENCH_WARMUP` calls of each arm (2). A ratio is
//! the median of the per-round ratios; its range is printed beside it. A
//! shape whose first arm moved more than 10% against itself across rounds is
//! flagged `EXCEEDS 10%`, and the run then exits 3 (as the tune rigs do).
//!
//! Lanes (`BENCH_LANES`, comma-separated, default all):
//! - `splitk`: the TN shapes `prefer_tn_splitk` gates (K ≥ 2048, M, N ≤ 384,
//!   min(M, N) ≤ 128), overwriting (`gemm_tn_bf16`) and accumulating
//!   (`GemmOperands::Bf16::tn_acc`). Arms: production, one cooperative
//!   dispatch over all of K (128x64 plain, 64x64 accumulate), and the
//!   parallel split-K (every partition one 64x64 cooperative tile set into a
//!   scratch slice, then the partition reduce) at three widths. Every arm but
//!   production is dispatched raw here, so the comparison does not depend on
//!   which one production routes to. (Until the parallel lane replaced it,
//!   a fourth arm timed the sequential 64x32 split-K;
//!   bench/results/splitk_par_bf16_m5pro.txt.)
//! - `smallm`: plain bf16 NN at M = 16..1024, the 128x64 and 64x64 tiles
//!   (`gemm_tiled`) against production's choice.
//! - `batched`: strided-batched NN (`gemm_batched`, batch > 1) on bf16, f16
//!   and relaxed f32 at N <= 512 and bf16 short M, where the single-matrix
//!   path takes 64x64: production against the 64x64 batched kernel.
//! - `epi`: the fused epilogue (`beta = 1`, bias, SiLU) at N <= 512, where
//!   the plain path takes 64x64: production against the 64x64 epilogue
//!   kernel, for bf16, f16 and relaxed f32.

// A paired lane takes its operands, its shape and the knobs of the run.
#![allow(clippy::too_many_arguments)]

mod common;

use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;
use tessl::gemm::{
    cast_f32_to_bf16, cast_f32_to_f16, gemm, gemm_batched, gemm_epilogue, gemm_tiled, gemm_tn_bf16, Activation,
    BatchStrides, BatchedGemm, EpiTile, Epilogue, GemmBackend, GemmOperands,
};
use tessl::runtime::{mtl_size, GpuRuntime};
use tessl::tensor::Tensor;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut g = common::Lcg::new(seed);
    (0..n).map(|_| (g.next_u01() * 2.0 - 1.0) as f32).collect()
}

fn bf16(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(&fill(shape.iter().product(), seed));
    cast_f32_to_bf16(&t)
}

/// One cooperative TN dispatch over all of K: `matmul2d_tensorops_tn_bf16_f32`
/// (128x64, C overwritten) or `_tn_accum_bf16_f32` (64x64, C added into).
fn coop_tn(
    rt: &Arc<GpuRuntime>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    (m, n, k): (usize, usize, usize),
    acc: bool,
) -> Result<(), String> {
    let (kernel, sm, sn) = if acc {
        ("matmul2d_tensorops_tn_accum_bf16_f32", 64, 64)
    } else {
        ("matmul2d_tensorops_tn_bf16_f32", 128, 64)
    };
    let p = rt.pipeline(kernel)?;
    let (tiles_n, tiles_m) = (n.div_ceil(sn), m.div_ceil(sm));
    let tpt = p.threadExecutionWidth() * 4;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.dispatch(mtl_size(tiles_n * tiles_m, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// The parallel bf16 split-K at width `k_tile`: every partition's 64x64
/// cooperative tiles in one `matmul2d_tensorops_tn_splitk_par_bf16_f32`
/// dispatch into a scratch slice each, then `reduce_partitions_f32` (or
/// `_add_f32` when accumulating) in partition order.
fn par_splitk(
    rt: &Arc<GpuRuntime>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    (m, n, k): (usize, usize, usize),
    k_tile: usize,
    acc: bool,
) -> Result<(), String> {
    // The kernel and the tile it is compiled for (64x64 sg4), together.
    const PAR: (&str, usize) = ("matmul2d_tensorops_tn_splitk_par_bf16_f32", 64);
    let p = rt.pipeline(PAR.0)?;
    let r = rt.pipeline(if acc {
        "reduce_partitions_add_f32"
    } else {
        "reduce_partitions_f32"
    })?;
    let (tiles_n, tiles_m) = (n.div_ceil(PAR.1), m.div_ceil(PAR.1));
    let parts = k.div_ceil(k_tile);
    let numel = m * n;
    let slice = numel.div_ceil(4) * 4;
    let scratch = rt.alloc_temp_f32(&[slice * parts])?;
    let tpt = p.threadExecutionWidth() * 4;
    let r_tpt = r.threadExecutionWidth().min(numel).max(1);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(&scratch, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(k_tile as u32, 6);
        bnd.bind_u32(tiles_n as u32, 7);
        bnd.bind_u32(tiles_m as u32, 8);
        bnd.bind_u32(parts as u32, 9);
        bnd.bind_u32(slice as u32, 10);
        bnd.dispatch(mtl_size(tiles_n * tiles_m * parts, 1, 1), mtl_size(tpt, 1, 1));
        if bnd.needs_explicit_barriers() {
            bnd.barrier();
        }
        bnd.set_pipeline(&r);
        bnd.bind_tensor(&scratch, 0);
        bnd.bind_tensor(c, 1);
        bnd.bind_u32(numel as u32, 2);
        bnd.bind_u32(parts as u32, 3);
        bnd.bind_u32(slice as u32, 4);
        bnd.dispatch(mtl_size(numel.div_ceil(r_tpt), 1, 1), mtl_size(r_tpt, 1, 1));
        Ok(())
    })
}

/// One batched NN dispatch on `kernel` (a `NN_COOP_BATCHED_KERNEL`
/// instantiation of `sm x 64` tiles).
fn batched_raw(
    rt: &Arc<GpuRuntime>,
    kernel: &str,
    sm: usize,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    spec: BatchedGemm,
) -> Result<(), String> {
    let p = rt.pipeline(kernel)?;
    let (tiles_n, tiles_m) = (spec.n.div_ceil(64), spec.m.div_ceil(sm));
    let tpt = p.threadExecutionWidth() * 4;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(spec.m as u32, 3);
        bnd.bind_u32(spec.n as u32, 4);
        bnd.bind_u32(spec.k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.bind_u32(spec.strides.a as u32, 8);
        bnd.bind_u32(spec.strides.b as u32, 9);
        bnd.bind_u32(spec.strides.c as u32, 10);
        bnd.dispatch(mtl_size(tiles_n * tiles_m, spec.batch, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// One epilogue NN dispatch on `kernel` (a `NN_COOP_EPI_KERNEL`
/// instantiation of `sm x 64` tiles), `beta = 1` with a bias and SiLU.
fn epi_raw(
    rt: &Arc<GpuRuntime>,
    kernel: &str,
    sm: usize,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    bias: &Tensor,
    (m, n, k): (usize, usize, usize),
) -> Result<(), String> {
    let p = rt.pipeline(kernel)?;
    let (tiles_n, tiles_m) = (n.div_ceil(64), m.div_ceil(sm));
    let tpt = p.threadExecutionWidth() * 4;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        bnd.bind_tensor(a, 0);
        bnd.bind_tensor(b, 1);
        bnd.bind_tensor(c, 2);
        bnd.bind_u32(m as u32, 3);
        bnd.bind_u32(n as u32, 4);
        bnd.bind_u32(k as u32, 5);
        bnd.bind_u32(tiles_n as u32, 6);
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.bind_tensor(bias, 8);
        bnd.bind_f32(1.0, 9);
        bnd.bind_f32(1.0, 10);
        bnd.bind_u32(Activation::Silu as u32, 11);
        bnd.bind_u32(1, 12);
        bnd.dispatch(mtl_size(tiles_n * tiles_m, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// The operand element types the batched and epilogue lanes cover.
#[derive(Clone, Copy)]
enum Elem {
    Bf16,
    F16,
    Relaxed,
}

impl Elem {
    fn name(self) -> &'static str {
        match self {
            Elem::Bf16 => "bf16",
            Elem::F16 => "f16",
            Elem::Relaxed => "f32_relaxed",
        }
    }

    /// The operand as this lane takes it, from f32 values; relaxed f32 also
    /// turns the runtime's relaxed precision on.
    fn operand(self, rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
        let t = rt.alloc_tensor_f32(shape)?;
        t.buffer.write_f32(&fill(shape.iter().product(), seed));
        rt.set_relaxed_precision(matches!(self, Elem::Relaxed));
        match self {
            Elem::Bf16 => cast_f32_to_bf16(&t),
            Elem::F16 => cast_f32_to_f16(&t),
            Elem::Relaxed => Ok(t),
        }
    }

    /// `matmul2d_tensorops_<this>_f32` (`_relaxed` for f32): the stem of the
    /// cooperative kernel names.
    fn stem(self) -> &'static str {
        match self {
            Elem::Bf16 => "matmul2d_tensorops_bf16_f32",
            Elem::F16 => "matmul2d_tensorops_f16_f32",
            Elem::Relaxed => "matmul2d_tensorops_f32_relaxed",
        }
    }
}

const ELEMS: [Elem; 3] = [Elem::Bf16, Elem::F16, Elem::Relaxed];

fn batched_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    // (m, n, k, batch): attention-style heads, a short-M bf16 batch, and
    // N = 512 at the edge of the rule.
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        (256, 64, 64, 16),
        (512, 128, 512, 8),
        (1024, 256, 256, 4),
        (128, 512, 2048, 4),
        (2048, 128, 128, 8),
        (64, 3072, 768, 8),
    ];
    for &(m, n, k, batch) in SHAPES {
        for elem in ELEMS {
            if n > 512 && !matches!(elem, Elem::Bf16) {
                continue;
            }
            let a = elem.operand(rt, &[batch * m, k], 0x7a1 ^ (m * n) as u64)?;
            let b = elem.operand(rt, &[batch * k, n], 0x7b2 ^ (n * k) as u64)?;
            let spec = BatchedGemm {
                m,
                n,
                k,
                batch,
                strides: BatchStrides::contiguous(m, n, k),
            };
            let cs: Vec<Tensor> = (0..3)
                .map(|_| rt.alloc_tensor_f32(&[batch * m, n]))
                .collect::<Result<_, _>>()?;
            let wide = format!("{}_batched", elem.stem());
            let narrow = format!("{}_batched_64x64_sg4", elem.stem());
            gemm_batched(&a, &b, &cs[0], GemmBackend::TensorOps, spec)?;
            batched_raw(rt, &wide, 128, &a, &b, &cs[1], spec)?;
            batched_raw(rt, &narrow, 64, &a, &b, &cs[2], spec)?;
            rt.synchronize()?;
            let base = cs[0].buffer.read_f32();
            for c in &cs[1..] {
                if c.buffer.read_f32()[..batch * m * n] != base[..batch * m * n] {
                    return Err(format!(
                        "batched {} {m}x{n}x{k}x{batch}: the tiles' outputs differ",
                        elem.name()
                    ));
                }
            }
            let reps = run.reps;
            let mut arms = vec![
                arm("production", rt, reps, || {
                    gemm_batched(&a, &b, &cs[0], GemmBackend::TensorOps, spec)
                }),
                arm("batched_128x64", rt, reps, || {
                    batched_raw(rt, &wide, 128, &a, &b, &cs[1], spec)
                }),
                arm("batched_64x64", rt, reps, || {
                    batched_raw(rt, &narrow, 64, &a, &b, &cs[2], spec)
                }),
            ];
            report(
                &format!(
                    "batched {} M={m} N={n} K={k} batch={batch}  (outputs bit-identical)",
                    elem.name()
                ),
                &mut arms,
                run,
            )?;
        }
    }
    rt.set_relaxed_precision(false);
    Ok(())
}

fn epi_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    const SHAPES: &[(usize, usize, usize)] = &[
        (2048, 384, 2048),
        (4096, 128, 768),
        (1024, 256, 1024),
        (512, 512, 512),
        (8192, 512, 768),
        // bf16 grid fill: fewer than 64 tiles of 128x64 with M >= 128.
        (256, 768, 3072),
        (128, 2048, 2048),
        (512, 768, 3072),
        (128, 4096, 4096),
    ];
    for &(m, n, k) in SHAPES {
        for elem in ELEMS {
            let a = elem.operand(rt, &[m, k], 0x8a1 ^ (m * n) as u64)?;
            let b = elem.operand(rt, &[k, n], 0x8b2 ^ (n * k) as u64)?;
            let bias = rt.alloc_tensor_f32(&[n])?;
            bias.buffer.write_f32(&fill(n, 0x8c3));
            let c0 = fill(m * n, 0x8d4);
            let cs: Vec<Tensor> = (0..3)
                .map(|_| {
                    let t = rt.alloc_tensor_f32(&[m, n])?;
                    t.buffer.write_f32(&c0);
                    Ok(t)
                })
                .collect::<Result<_, String>>()?;
            let epi = Epilogue {
                alpha: 1.0,
                beta: 1.0,
                bias: Some(&bias),
                activation: Activation::Silu,
            };
            let wide = format!("{}_epi", elem.stem());
            let narrow = format!("{}_epi_64x64_sg4", elem.stem());
            gemm_epilogue(&a, &b, &cs[0], GemmBackend::TensorOps, epi)?;
            epi_raw(rt, &wide, 128, &a, &b, &cs[1], &bias, (m, n, k))?;
            epi_raw(rt, &narrow, 64, &a, &b, &cs[2], &bias, (m, n, k))?;
            rt.synchronize()?;
            let base = cs[0].buffer.read_f32();
            for c in &cs[1..] {
                if c.buffer.read_f32()[..m * n] != base[..m * n] {
                    return Err(format!(
                        "epilogue {} {m}x{n}x{k}: the tiles' outputs differ",
                        elem.name()
                    ));
                }
            }
            let reps = run.reps;
            let mut arms = vec![
                arm("production", rt, reps, || {
                    gemm_epilogue(&a, &b, &cs[0], GemmBackend::TensorOps, epi)
                }),
                arm("epi_128x64", rt, reps, || {
                    epi_raw(rt, &wide, 128, &a, &b, &cs[1], &bias, (m, n, k))
                }),
                arm("epi_64x64", rt, reps, || {
                    epi_raw(rt, &narrow, 64, &a, &b, &cs[2], &bias, (m, n, k))
                }),
            ];
            report(
                &format!(
                    "epi {} M={m} N={n} K={k} beta=1 bias silu  (outputs bit-identical)",
                    elem.name()
                ),
                &mut arms,
                run,
            )?;
        }
    }
    rt.set_relaxed_precision(false);
    Ok(())
}

/// The knobs of one run.
struct Run {
    reps: usize,
    warmup: usize,
    iters: usize,
    rounds: usize,
    /// Labels whose first arm broke [`common::SELF_SPREAD_LIMIT`]; a
    /// non-empty list ends the run with exit 3.
    unstable: std::cell::RefCell<Vec<String>>,
}

/// Time `arms` (each one call = `reps` dispatches, then a wait) as one
/// interleaved A/B and print every arm against the first.
fn report(label: &str, arms: &mut [common::Arm<'_>], run: &Run) -> Result<Vec<f64>, String> {
    let paired = common::run_paired(arms, run.warmup, run.iters, run.rounds)?;
    let base = &paired.rounds[0];
    let spread = common::self_spread(base);
    if spread > common::SELF_SPREAD_LIMIT {
        run.unstable.borrow_mut().push(label.to_string());
    }
    println!(
        "\n{label}  (self-spread of {} {:.1}%{})",
        paired.names[0],
        spread * 100.0,
        if spread > common::SELF_SPREAD_LIMIT {
            ", EXCEEDS 10%"
        } else {
            ""
        }
    );
    let mut medians = Vec::with_capacity(paired.names.len());
    for (name, rounds) in paired.names.iter().zip(&paired.rounds) {
        let med = common::median(rounds.clone())?;
        let (ratio, lo, hi) = common::ratio_stats(base, rounds)?;
        println!(
            "  {name:<22} {:>9.1} us   {}/this {ratio:.2} [{lo:.2}–{hi:.2}]",
            med * 1e3,
            paired.names[0]
        );
        medians.push(med);
    }
    Ok(medians)
}

/// `f` repeated `reps` times, then one wait: one call of an arm.
fn arm<'a>(
    name: &str,
    rt: &'a Arc<GpuRuntime>,
    reps: usize,
    mut f: impl FnMut() -> Result<(), String> + 'a,
) -> common::Arm<'a> {
    common::Arm {
        name: name.into(),
        reps,
        run: Box::new(move || {
            for _ in 0..reps {
                f()?;
            }
            rt.synchronize()
        }),
    }
}

/// Largest absolute difference between two arms' outputs, relative to the
/// first's largest magnitude: every arm computes the same product.
fn rel_diff(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())) as f64;
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
        / scale.max(1e-30)
}

fn splitk_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    const SHAPES: &[(usize, usize, usize)] = &[
        (128, 128, 4096),
        (128, 384, 4096),
        (384, 128, 4096),
        (64, 64, 8192),
        (128, 128, 2048),
        (128, 384, 8192),
        (128, 384, 12288),
        (128, 384, 16384),
        (128, 384, 32768),
        (128, 128, 8192),
        (128, 128, 16384),
        (128, 128, 32768),
        (64, 64, 16384),
        (64, 64, 32768),
        (384, 128, 16384),
        // Not gated: every TN takes the cooperative kernel. The control.
        (384, 384, 4096),
    ];
    const WIDTHS: [usize; 3] = [256, 512, 1024];
    for &(m, n, k) in SHAPES {
        let dims = (m, n, k);
        let a = bf16(rt, &[k, m], 0x5a1 ^ (m * n + k) as u64)?;
        let b = bf16(rt, &[k, n], 0x5b2 ^ (m * n + k) as u64)?;
        for acc in [false, true] {
            // Outputs, checked against each other before anything is timed.
            let c0 = fill(m * n, 0x5c3 ^ k as u64);
            let fresh = || -> Result<Tensor, String> {
                let t = rt.alloc_tensor_f32(&[m, n])?;
                t.buffer.write_f32(&c0);
                Ok(t)
            };
            let cs: Vec<Tensor> = (0..2 + WIDTHS.len()).map(|_| fresh()).collect::<Result<_, _>>()?;
            let production = |c: &Tensor| {
                if acc {
                    GemmOperands::Bf16.tn_acc(&a, &b, c)
                } else {
                    gemm_tn_bf16(&a, &b, c)
                }
            };
            production(&cs[0])?;
            coop_tn(rt, &a, &b, &cs[1], dims, acc)?;
            for (i, &w) in WIDTHS.iter().enumerate() {
                par_splitk(rt, &a, &b, &cs[2 + i], dims, w, acc)?;
            }
            rt.synchronize()?;
            let base = cs[0].buffer.read_f32()[..m * n].to_vec();
            let worst = cs[1..]
                .iter()
                .map(|c| rel_diff(&base, &c.buffer.read_f32()[..m * n]))
                .fold(0.0, f64::max);
            if worst.is_nan() || worst >= 1e-4 {
                return Err(format!(
                    "{m}x{n}x{k} acc={acc}: arms disagree by {worst:.2e} of the output's scale"
                ));
            }

            let reps = run.reps;
            let mut arms = vec![
                arm("production", rt, reps, || production(&cs[0])),
                arm(if acc { "coop_64x64" } else { "coop_128x64" }, rt, reps, || {
                    coop_tn(rt, &a, &b, &cs[1], dims, acc)
                }),
            ];
            for (i, &w) in WIDTHS.iter().enumerate() {
                let c = &cs[2 + i];
                let (a, b) = (&a, &b);
                arms.push(arm(&format!("par_splitk_{w}"), rt, reps, move || {
                    par_splitk(rt, a, b, c, dims, w, acc)
                }));
            }
            let label = format!(
                "splitk {} M={m} N={n} K={k}  (arms agree within {worst:.1e})",
                if acc { "accumulate" } else { "overwrite" }
            );
            report(&label, &mut arms, run)?;
        }
    }
    Ok(())
}

fn smallm_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    const MS: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];
    const NK: [(usize, usize); 4] = [(3072, 768), (768, 3072), (4096, 4096), (2048, 2048)];
    for &(n, k) in &NK {
        for &m in &MS {
            let a = bf16(rt, &[m, k], 0x6a1 ^ (m * n + k) as u64)?;
            let b = bf16(rt, &[k, n], 0x6b2 ^ (n + k) as u64)?;
            let cs: Vec<Tensor> = (0..3).map(|_| rt.alloc_tensor_f32(&[m, n])).collect::<Result<_, _>>()?;
            gemm(&a, &b, &cs[0], GemmBackend::TensorOps)?;
            gemm_tiled(&a, &b, &cs[1], GemmBackend::TensorOps, EpiTile::Wide)?;
            gemm_tiled(&a, &b, &cs[2], GemmBackend::TensorOps, EpiTile::Narrow)?;
            rt.synchronize()?;
            let base = cs[0].buffer.read_f32();
            for c in &cs[1..] {
                if c.buffer.read_f32()[..m * n] != base[..m * n] {
                    return Err(format!("M={m} N={n} K={k}: the tiles' outputs differ"));
                }
            }
            let reps = run.reps;
            let mut arms = vec![
                arm("wide_128x64", rt, reps, || {
                    gemm_tiled(&a, &b, &cs[1], GemmBackend::TensorOps, EpiTile::Wide)
                }),
                arm("narrow_64x64", rt, reps, || {
                    gemm_tiled(&a, &b, &cs[2], GemmBackend::TensorOps, EpiTile::Narrow)
                }),
                arm("production", rt, reps, || gemm(&a, &b, &cs[0], GemmBackend::TensorOps)),
            ];
            let wide_tiles = m.div_ceil(128) * n.div_ceil(64);
            report(
                &format!("smallm M={m} N={n} K={k}  ({wide_tiles} tiles of 128x64; outputs bit-identical)"),
                &mut arms,
                run,
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("bf16 GEMM needs TensorOps".into());
    }
    let run = Run {
        reps: common::env_usize("BENCH_REPS", 20, 1)?,
        warmup: common::env_usize("BENCH_WARMUP", 2, 0)?,
        iters: common::env_usize("BENCH_ITERS", 8, 1)?,
        rounds: common::env_usize("BENCH_ROUNDS", 6, 1)?,
        unstable: Default::default(),
    };
    let lanes = std::env::var("BENCH_LANES").unwrap_or_else(|_| "splitk,smallm,batched,epi".into());
    println!("device: {}", rt.device_name());
    println!(
        "async encode, {} GEMMs per command buffer, {} calls per arm per round, {} rounds \
         (order reversed every other round) after {} warmup calls; us per GEMM, median of \
         per-round medians; ratio = first arm / this arm (> 1: this arm is faster), median of \
         per-round ratios [range]",
        run.reps, run.iters, run.rounds, run.warmup
    );
    rt.set_async_encode(true)?;
    for lane in lanes.split(',') {
        match lane {
            "splitk" => splitk_lane(&rt, &run)?,
            "smallm" => smallm_lane(&rt, &run)?,
            "batched" => batched_lane(&rt, &run)?,
            "epi" => epi_lane(&rt, &run)?,
            other => return Err(format!("BENCH_LANES: no lane {other:?} (splitk, smallm, batched, epi)")),
        }
    }
    rt.set_async_encode(false)?;
    common::emit_kernel_trace();
    common::exit_if_unstable("bench_gemm_routes", &run.unstable.borrow());
    Ok(())
}
