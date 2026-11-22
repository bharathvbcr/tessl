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
//!   (`gemm_tiled`) against production's choice. `BENCH_SMALLM_ELEMS` (comma
//!   list of `bf16`, `f16`, `f32_relaxed`; unset = bf16 only, the committed
//!   lane) runs the same table on f16 and relaxed f32 (lead 3, below).
//! - `nnsmall`: the small-M exact-f32 NN lane, at EmbeddingGemma 2's GEMM shapes
//!   (`BENCH_SHAPES`: names, default the forward's; the `audio *` ones are the small-B shapes) and `BENCH_ROWS` rows
//!   (default 1,16,32,64), weights cycled over 24 copies as in a forward.
//!   Arms: the plain single dispatch (raw), production (`gemm_small_m`, whatever it
//!   routes to), and the parallel split-K at `BENCH_WIDTHS` (default
//!   64,128,256,512,1024, those under K). Every arm's output is checked against
//!   the plain one before anything is timed.
//! - `batched`: strided-batched NN (`gemm_batched`, batch > 1) on bf16, f16
//!   and relaxed f32 at N <= 512 and bf16 short M, where the single-matrix
//!   path takes 64x64: production against the 64x64 batched kernel.
//! - `epi`: the fused epilogue (`beta = 1`, bias, SiLU) at N <= 512, where
//!   the plain path takes 64x64: production against the 64x64 epilogue
//!   kernel, for bf16, f16 and relaxed f32.
//! - `sam` (not in the default set): SpikeTrack's SAM 3.1 trunk linears at
//!   M = 5184 and 2304: Ojas's NT bf16 and exact-f32 calls against NN on
//!   pre-transposed weights (production tile, both tiles), f16 operands and
//!   the bias fused in the epilogue; and the NT bf16 call with a bias, and with
//!   a bias and a bf16 round, as the three separate dispatches Ojas runs today
//!   (`gemm_nt_bf16`, `bert::bias_add`, the casts through a bf16 scratch)
//!   against `gemm_nt_bf16_bias` (one dispatch; bit-identical, checked before
//!   anything is timed).
//! - `relaxedplain` (not in the default set): plain relaxed-f32 NN at N <= 512 on
//!   large M, 128x64 and 64x64 (`gemm_tiled`) against production, which takes
//!   64x64 there (lead 4, below).
//!
//! # The GEMM-routing leads (card ft-ea33f4774ab30b7b2d423564f7c76dbe)
//!
//! Measurement only: nothing here changes a routing rule, and none of the
//! three runs unless asked for. Each needs `TESSL_KERNEL_TRACE=1` (the proof of
//! which kernel ran reads the pipeline-request trace, latched at process
//! start) and refuses to time anything without it.
//!
//! 1. Lead 2, parallel bf16 split-K beyond `prefer_tn_splitk`:
//!    `BENCH_LANES=splitk BENCH_SPLITK_SET=ungated` runs `UNGATED_SHAPES`, TN
//!    shapes the gate does not take (dW of the Qwen3.5-2B and EmbeddingGemma 2
//!    linears, and the gated attention/MLP dW at a K below 2048), with the lane's
//!    own arms. Widths are those below K whose scratch fits production's own cap.
//!    Production is the cooperative kernel on every one of these, so the
//!    `coop_*` arm is a null comparison and its ratio is the noise floor.
//! 2. Lead 3, grid-fill tile for f16 and relaxed f32 on the plain path:
//!    `BENCH_LANES=smallm BENCH_SMALLM_ELEMS=bf16,f16,f32_relaxed` (the bf16
//!    rows reproduce `bf16_smallm_coop_reconciled_m5pro.txt` on the same run).
//! 3. Lead 4, relaxed f32 at 64x64 against 128x64 on large grids, which the
//!    plain rule gives 64x64 by N <= 512: `BENCH_LANES=relaxedplain`.
//!
//! Before any timing, every arm of these runs (a) leaves an output that is
//! finite and not all zero, (b) agrees with an f64 host reference on the
//! operands as the kernel read them, at 256 sampled elements within
//! 2e-3 of the reference's largest sampled magnitude (relaxed f32: 2^-8, the
//! bound the tree itself holds the tf32-class kernels to; this bin has no other
//! host-reference gate, and the arms' mutual 1e-4 and bit-identity gates stay), and
//! (c) is proven to have run its intended kernel: the pipeline-request trace
//! shows the expected kernel name and not the other tile's (or the split-K's),
//! and the raw arms are exactly one `with_binder` dispatch. A shape that fails
//! its checks prints `REFUSED`, the run goes on, and it exits 4 at the end.
//!
//! **Win rule, stated before any of these ran.** A route change is indicated
//! only if the alternative (the arm production does not take, or for lead 2
//! the parallel split-K) has a median per-round production/alternative time
//! ratio beyond 1 + max(0.05, 2 x spread), spread being the production arm's
//! self-spread across rounds, and is faster in at least 80% of rounds
//! (`route_verdict` prints this per arm), in both of two invocations of the
//! identical command. A shape whose production arm exceeds the 10% self-spread
//! limit is excluded, not counted either way. For lead 2 the same split-K
//! width has to win in both invocations; a width picked after seeing both does
//! not count. A verdict of `production faster` is the symmetric rule and
//! confirms the current route. Anything else is inconclusive, and the route
//! stays.

// A paired lane takes its operands, its shape and the knobs of the run.
#![allow(clippy::too_many_arguments)]

mod common;

use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;
use tessl::bert::bias_add;
use tessl::gemm::{
    cast_bf16_to_f32, cast_bf16_to_f32_into, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_bf16_into, cast_f32_to_f16,
    gemm, gemm_batched, gemm_epilogue, gemm_nn_splitk_par_f32, gemm_nt_bf16, gemm_nt_bf16_bias, gemm_nt_f32,
    gemm_small_m, gemm_tiled, gemm_tn_bf16, Activation, BatchStrides, BatchedGemm, EpiTile, Epilogue, GemmBackend,
    GemmOperands,
};
use tessl::runtime::{mtl_size, GpuRuntime};
use tessl::tensor::Tensor;

fn bf16(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(&common::fill(shape.iter().product(), seed));
    cast_f32_to_bf16(&t)
}

/// The cooperative TN kernel [`coop_tn`] dispatches, and the kernels of the
/// parallel split-K ([`par_splitk`]), by name: what the arms request and what
/// the kernel proof looks for.
fn coop_tn_kernel(acc: bool) -> &'static str {
    if acc {
        "matmul2d_tensorops_tn_accum_bf16_f32"
    } else {
        "matmul2d_tensorops_tn_bf16_f32"
    }
}

const PAR_SPLITK_KERNEL: &str = "matmul2d_tensorops_tn_splitk_par_bf16_f32";

fn reduce_kernel(acc: bool) -> &'static str {
    if acc {
        "reduce_partitions_add_f32"
    } else {
        "reduce_partitions_f32"
    }
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
        (coop_tn_kernel(true), 64, 64)
    } else {
        (coop_tn_kernel(false), 128, 64)
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
    const PAR: (&str, usize) = (PAR_SPLITK_KERNEL, 64);
    let p = rt.pipeline(PAR.0)?;
    let r = rt.pipeline(reduce_kernel(acc))?;
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

/// The operand element types the batched, epilogue and plain-tile lanes cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        t.buffer.write_f32(&common::fill(shape.iter().product(), seed));
        rt.set_relaxed_precision(matches!(self, Elem::Relaxed));
        match self {
            Elem::Bf16 => cast_f32_to_bf16(&t),
            Elem::F16 => cast_f32_to_f16(&t),
            Elem::Relaxed => Ok(t),
        }
    }

    fn parse(name: &str) -> Option<Elem> {
        ELEMS.into_iter().find(|e| e.name() == name)
    }

    /// The plain cooperative NN kernel on the 128x64 tile, or with `narrow` the
    /// 64x64 one (src/gemm.rs `coop_kernel`).
    fn tile_kernel(self, narrow: bool) -> String {
        if narrow {
            format!("{}_64x64_sg4", self.stem())
        } else {
            self.stem().to_string()
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
            bias.buffer.write_f32(&common::fill(n, 0x8c3));
            let c0 = common::fill(m * n, 0x8d4);
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
    /// Shapes of the opt-in lead runs that failed a pre-timing check; the run
    /// goes on without them and a non-empty list ends it with exit 4.
    refused: std::cell::RefCell<Vec<String>>,
}

/// Exit status of a run that refused a shape (3 is a drifting baseline).
const EXIT_REFUSED: i32 = 4;

/// Run one shape of an opt-in lead lane. A failed check is recorded, printed
/// as `REFUSED` and does not end the run: the queue is hours long, and one bad
/// shape must not take the rest of the step with it.
fn guarded(run: &Run, label: &str, shape: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    if let Err(e) = shape() {
        eprintln!("REFUSED {label}: {e}");
        println!("\nREFUSED {label}: {e}");
        run.refused.borrow_mut().push(label.to_string());
    }
    Ok(())
}

/// The kernel proof reads the pipeline-request trace, which is latched from
/// the environment at process start: without it a "proof" would be nothing.
fn require_trace(what: &str) -> Result<(), String> {
    if tessl::runtime::kernel_trace_enabled() {
        Ok(())
    } else {
        Err(format!(
            "{what}: needs TESSL_KERNEL_TRACE=1 (the proof of which kernel ran reads the pipeline-request trace); refusing to time arms whose kernel is unproven"
        ))
    }
}

/// [`prove`]'s verdict from the trace deltas around one call: every `expect`ed
/// kernel was requested, no `forbid`den one was, and (when `want` is given) the
/// call was exactly that many `with_binder` dispatches.
fn judge_proof(
    what: &str,
    expect: &[(&str, u64)],
    forbid: &[(&str, u64)],
    want: Option<usize>,
    got: usize,
) -> Result<(), String> {
    for &(name, delta) in expect {
        if delta == 0 {
            return Err(format!(
                "{what}: kernel {name} was not requested, so another kernel ran"
            ));
        }
    }
    for &(name, delta) in forbid {
        if delta != 0 {
            return Err(format!(
                "{what}: kernel {name} was requested, so the route is not the intended one"
            ));
        }
    }
    match want {
        Some(w) if w != got => Err(format!("{what}: {got} dispatches, expected {w}")),
        _ => Ok(()),
    }
}

/// Run `call` once between two trace readings and judge them ([`judge_proof`]).
/// Returns the dispatch count of the call.
fn prove(
    rt: &Arc<GpuRuntime>,
    what: &str,
    expect: &[&str],
    forbid: &[&str],
    want: Option<usize>,
    call: impl FnOnce() -> Result<(), String>,
) -> Result<usize, String> {
    let counts =
        |names: &[&str]| -> Vec<u64> { names.iter().map(|n| tessl::runtime::traced_kernel_count(n)).collect() };
    rt.synchronize()?;
    rt.take_dispatch_count();
    let (e0, f0) = (counts(expect), counts(forbid));
    call()?;
    rt.synchronize()?;
    let got = rt.take_dispatch_count();
    let delta = |names: &[&str], before: &[u64]| -> Vec<u64> {
        names
            .iter()
            .zip(before)
            .map(|(n, b)| tessl::runtime::traced_kernel_count(n).saturating_sub(*b))
            .collect()
    };
    let er: Vec<(&str, u64)> = expect.iter().copied().zip(delta(expect, &e0)).collect();
    let fr: Vec<(&str, u64)> = forbid.iter().copied().zip(delta(forbid, &f0)).collect();
    judge_proof(what, &er, &fr, want, got)?;
    Ok(got)
}

/// Time `arms` (each one call = `reps` dispatches, then a wait) as one
/// interleaved A/B and print every arm against the first.
fn report(label: &str, arms: &mut [common::Arm<'_>], run: &Run) -> Result<Vec<f64>, String> {
    Ok(report_paired(label, arms, run)?.1)
}

/// [`report`], also returning the per-round samples (`rounds[arm][round]`).
fn report_paired(label: &str, arms: &mut [common::Arm<'_>], run: &Run) -> Result<(common::Paired, Vec<f64>), String> {
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
    Ok((paired, medians))
}

/// What a lead's measurement says about one alternative arm (the module docs'
/// win rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// Beyond the band and faster in at least 80% of rounds.
    AltFaster,
    /// The mirror: production is beyond the band and faster in 80% of rounds.
    ProductionFaster,
    Inconclusive,
    /// Production's self-spread broke [`common::SELF_SPREAD_LIMIT`]: not evidence.
    Excluded,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::AltFaster => "ALTERNATIVE FASTER",
            Outcome::ProductionFaster => "production faster",
            Outcome::Inconclusive => "inconclusive",
            Outcome::Excluded => "excluded (EXCEEDS 10%)",
        }
    }
}

/// The band's floor, and its multiple of the production arm's self-spread.
const ROUTE_MIN_EFFECT: f64 = 0.05;
const ROUTE_SPREAD_MULT: f64 = 2.0;

struct Verdict {
    outcome: Outcome,
    /// Median of the per-round production / alternative time ratios.
    median: f64,
    /// Rounds in which the alternative was faster.
    faster: usize,
    rounds: usize,
    band: f64,
}

/// The win rule on the per-round times of production and one alternative:
/// median ratio beyond 1 +- max(0.05, 2 x spread) and the same direction in at
/// least 80% of rounds (integer arithmetic, so 4 of 5 passes and 7 of 9 does
/// not). `spread` is the production arm's self-spread.
fn route_verdict(production: &[f64], alt: &[f64], spread: f64) -> Result<Verdict, String> {
    if production.is_empty() || production.len() != alt.len() {
        return Err(format!(
            "route verdict over {} production rounds and {} alternative rounds",
            production.len(),
            alt.len()
        ));
    }
    if !spread.is_finite() {
        return Err("route verdict with a non-finite self-spread".into());
    }
    let ratios: Vec<f64> = production.iter().zip(alt).map(|(p, a)| p / a).collect();
    let median = common::median(ratios.clone())?;
    let rounds = ratios.len();
    let faster = ratios.iter().filter(|&&r| r > 1.0).count();
    let slower = ratios.iter().filter(|&&r| r < 1.0).count();
    let band = ROUTE_MIN_EFFECT.max(ROUTE_SPREAD_MULT * spread);
    let outcome = if spread > common::SELF_SPREAD_LIMIT {
        Outcome::Excluded
    } else if median > 1.0 + band && faster * 5 >= rounds * 4 {
        Outcome::AltFaster
    } else if median < 1.0 - band && slower * 5 >= rounds * 4 {
        Outcome::ProductionFaster
    } else {
        Outcome::Inconclusive
    };
    Ok(Verdict {
        outcome,
        median,
        faster,
        rounds,
        band,
    })
}

/// Print [`route_verdict`] of every arm of `paired` against the production arm.
fn print_verdicts(paired: &common::Paired, production: usize) -> Result<(), String> {
    let spread = common::self_spread(&paired.rounds[production]);
    for (i, name) in paired.names.iter().enumerate() {
        if i == production {
            continue;
        }
        let v = route_verdict(&paired.rounds[production], &paired.rounds[i], spread)?;
        println!(
            "  verdict {name:<20} {:<24} (median production/this {:.2}, this faster in {}/{} rounds, band +-{:.2}, production self-spread {:.1}%)",
            v.outcome.label(),
            v.median,
            v.faster,
            v.rounds,
            v.band,
            spread * 100.0
        );
    }
    Ok(())
}

/// Tolerance of the host-reference gate, relative to the reference's largest
/// sampled magnitude, and how many elements it samples. Relaxed f32 runs the
/// tf32-class kernels, which the tree itself holds only to a bf16-class bound
/// (tests/gemm_correctness.rs `relaxed_precision_stays_within_a_bf16_class_bound`,
/// U_BF16 = 2^-8), so its gate is that bound and not the 2e-3 of the exact-in-f32
/// element types: a gate tighter than the tree's own would refuse every relaxed shape.
const HOST_REF_TOL: f64 = 2e-3;
const HOST_REF_TOL_RELAXED: f64 = 1.0 / 256.0;
const HOST_REF_SAMPLES: usize = 256;

fn host_ref_tol(elem: Elem) -> f64 {
    match elem {
        Elem::Bf16 | Elem::F16 => HOST_REF_TOL,
        Elem::Relaxed => HOST_REF_TOL_RELAXED,
    }
}

/// `got` (`[m, n]`, row-major) against an f64 host product of the operands as
/// the kernel read them: `a_at(i, p)` and `b_at(p, j)` are the logical
/// operands, `base` the C a `C +=` started from. The four corners and
/// [`HOST_REF_SAMPLES`] seeded positions are checked within `tol` (see
/// [`host_ref_tol`]) of the largest sampled magnitude; a NaN anywhere sampled,
/// a reference of all zeros and a short `got` are refused. Returns the worst
/// sampled error over that magnitude.
fn host_ref_check(
    lane: &str,
    got: &[f32],
    (m, n, k): (usize, usize, usize),
    a_at: &dyn Fn(usize, usize) -> f32,
    b_at: &dyn Fn(usize, usize) -> f32,
    base: Option<&[f32]>,
    tol: f64,
) -> Result<f64, String> {
    if m == 0 || n == 0 || k == 0 || got.len() < m * n || base.is_some_and(|b| b.len() < m * n) {
        return Err(format!(
            "{lane}: host reference of {m}x{n}x{k} against {} output values",
            got.len()
        ));
    }
    let mut rng = common::Lcg::new(0x40_57 ^ (m * n + k) as u64);
    let mut pick = |extent: usize| ((rng.next_u01() * extent as f64) as usize).min(extent - 1);
    let mut at = vec![(0, 0), (m - 1, n - 1), (0, n - 1), (m - 1, 0)];
    for _ in 0..HOST_REF_SAMPLES {
        let (i, j) = (pick(m), pick(n));
        at.push((i, j));
    }
    let want: Vec<f64> = at
        .iter()
        .map(|&(i, j)| {
            let dot: f64 = (0..k).map(|p| f64::from(a_at(i, p)) * f64::from(b_at(p, j))).sum();
            dot + base.map_or(0.0, |b| f64::from(b[i * n + j]))
        })
        .collect();
    let scale = want.iter().fold(0.0f64, |s, w| s.max(w.abs()));
    if !scale.is_finite() || scale == 0.0 {
        return Err(format!("{lane}: the host reference has no usable scale ({scale})"));
    }
    let sampled: Vec<f32> = at.iter().map(|&(i, j)| got[i * n + j]).collect();
    common::check_against(lane, &sampled, &want, tol * scale, 0.0)?;
    let worst = sampled
        .iter()
        .zip(&want)
        .fold(0.0f64, |s, (&g, &w)| s.max((f64::from(g) - w).abs()));
    Ok(worst / scale)
}

/// `t`'s elements as the kernel reads them, widened to f32 (a bf16 or f16
/// tensor through the exact GPU widening cast; relaxed f32 is read as is).
fn operand_values(rt: &Arc<GpuRuntime>, elem: Elem, t: &Tensor) -> Result<Vec<f32>, String> {
    let widened;
    let src = match elem {
        Elem::Bf16 => {
            widened = cast_bf16_to_f32(t)?;
            &widened
        }
        Elem::F16 => {
            widened = cast_f16_to_f32(t)?;
            &widened
        }
        Elem::Relaxed => t,
    };
    rt.synchronize()?;
    Ok(src.buffer.read_f32()[..t.numel()].to_vec())
}

/// A copy of `prefer_tn_splitk` (src/gemm.rs, private): the shapes whose TN
/// takes the parallel split-K. The unit tests pin this copy to the committed
/// gated table; on the device, the production arm's kernel name is the check.
#[cfg(test)]
fn prefer_tn_splitk(m: usize, n: usize, k: usize) -> bool {
    k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128
}

/// Read an env knob: unset is `None`, non-unicode is an error.
fn env_knob(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

/// `BENCH_SPLITK_SET`: which TN shapes the `splitk` lane runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitkSet {
    /// The shapes `prefer_tn_splitk` gates, and the one control: the committed lane.
    Gated,
    /// `UNGATED_SHAPES`: lead 2.
    Ungated,
}

impl SplitkSet {
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("gated") => Ok(Self::Gated),
            Some("ungated") => Ok(Self::Ungated),
            Some(other) => Err(format!(
                "BENCH_SPLITK_SET={other:?}: expected gated (the default) or ungated"
            )),
        }
    }
}

/// `BENCH_SMALLM_ELEMS`: the element types of the `smallm` lane, in order.
fn parse_elems(raw: Option<&str>) -> Result<Vec<Elem>, String> {
    let Some(raw) = raw else {
        return Ok(vec![Elem::Bf16]);
    };
    let mut out: Vec<Elem> = Vec::new();
    for name in raw.split(',').map(str::trim) {
        let elem = Elem::parse(name)
            .ok_or_else(|| format!("BENCH_SMALLM_ELEMS={raw:?}: no element type {name:?} (bf16, f16, f32_relaxed)"))?;
        if out.contains(&elem) {
            return Err(format!("BENCH_SMALLM_ELEMS={raw:?}: {name} twice"));
        }
        out.push(elem);
    }
    Ok(out)
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

/// The TN shapes the `splitk` lane has always run: every one `prefer_tn_splitk`
/// gates, and (last) the ungated 384x384x4096 control.
const GATED_SHAPES: &[(usize, usize, usize)] = &[
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

/// Lead 2: TN shapes `prefer_tn_splitk` does not gate (a copy of the predicate
/// is pinned against these by the unit tests), `(m, n, k, source)`. A TN call
/// is `C[m, n] = A[k, m]^T B[k, n]`, a weight gradient `dW = X^T dY` with `k` the
/// tokens of the step. None of them was taken from a caller that runs in this
/// tree at that shape. The only non-test TN caller in this tree (DevMap and rg
/// over src/, bins excluded) is the cross-entropy dW of `src/cross_entropy.rs`
/// (`dlogits^T h`, M = the vocab chunk, N = 2048 hidden, K = the row count): a
/// grid of 1280 tiles or more whose scratch is over production's own split-K cap
/// at any width, so a route could not take it, and it is left out. Training
/// backward callers of `gemm_tn_train`, `gemm_tn_accum_train` and
/// `GemmOperands::tn` live outside it, in Ojas (../ojas/arch02/metal-native/src/model_bwd.rs,
/// e.g. :458, :1321-1327, :1706-1719; ../ojas/ojas-metal/src/device.rs `gemm_tn`, e.g. :3001, :4576).
/// Those call sites were found by rg, but the dims of Ojas's own arch02 models were
/// not read, so the dims below are *derived* from the layers: Qwen3.5-2B's from its config
/// (tests/qwen35_model.rs, `the_real_2b_config_shape_still_loads`: hidden 2048,
/// 2 KV heads x head_dim 256, 16 linear-attention value heads), EmbeddingGemma
/// 2's from this bin's own `nnsmall` forward table.
const UNGATED_SHAPES: &[(usize, usize, usize, &str)] = &[
    (
        2048,
        16,
        4096,
        "derived: Qwen3.5-2B linear-attention gate dW (hidden 2048 x 16 value heads), 4096 tokens",
    ),
    (
        16,
        2048,
        4096,
        "derived: the same dW with the operands swapped (M = the 16 heads)",
    ),
    (
        2048,
        512,
        4096,
        "derived: Qwen3.5-2B k|v_proj dW (2048 x 2 x 256); EmbeddingGemma 2 down / o-global dW (K 2048 to N 512)",
    ),
    (
        512,
        512,
        4096,
        "derived: EmbeddingGemma 2 ple_gate / ple_out dW (512 x 512)",
    ),
    (1024, 512, 4096, "derived: EmbeddingGemma 2 o-sliding dW (1024 to 512)"),
    (
        512,
        2048,
        4096,
        "derived: EmbeddingGemma 2 qkv-sliding dW (512 to 2048)",
    ),
    (
        512,
        128,
        4096,
        "gate-edge probe, not a caller: the attention dW with M just past the gate's 384",
    ),
    (
        128,
        512,
        4096,
        "gate-edge probe, not a caller: the attention dW with N just past the gate's 384",
    ),
    (
        128,
        128,
        1024,
        "the gated attention dW (prefer_tn_splitk's doc: M = N = 128) at 1024 tokens, below its K >= 2048",
    ),
    (
        128,
        384,
        1024,
        "the gated MLP dW (one side mlp_dim 384) at 1024 tokens, below its K >= 2048",
    ),
    (384, 128, 1024, "the gated MLP dW, other orientation, at 1024 tokens"),
];

/// The parallel split-K widths of the lane.
const WIDTHS: [usize; 3] = [256, 512, 1024];

/// Production's cap on the parallel split-K's scratch, in f32s (src/gemm.rs
/// `TN_PAR_MAX_SCRATCH`, private): a width over it is a route that could not exist.
const PAR_SCRATCH_CAP: usize = 1 << 22;

/// Scratch floats of the parallel split-K at width `w`: one padded slice of C per partition.
fn par_scratch_floats((m, n, k): (usize, usize, usize), w: usize) -> usize {
    (m * n).div_ceil(4) * 4 * k.div_ceil(w)
}

/// The widths the lane times at this shape: all of them for the committed set,
/// as it always ran; for the ungated one those below K (two partitions or more)
/// whose scratch fits [`PAR_SCRATCH_CAP`].
fn live_widths(set: SplitkSet, dims: (usize, usize, usize)) -> Vec<usize> {
    match set {
        SplitkSet::Gated => WIDTHS.to_vec(),
        SplitkSet::Ungated => WIDTHS
            .into_iter()
            .filter(|&w| w < dims.2 && par_scratch_floats(dims, w) <= PAR_SCRATCH_CAP)
            .collect(),
    }
}

fn splitk_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    let set = SplitkSet::parse(env_knob("BENCH_SPLITK_SET")?.as_deref())?;
    match set {
        SplitkSet::Gated => {
            for &dims in GATED_SHAPES {
                splitk_shape(rt, run, set, dims, None)?;
            }
        }
        SplitkSet::Ungated => {
            require_trace("BENCH_SPLITK_SET=ungated")?;
            println!("\nsplitk, the ungated set: TN shapes prefer_tn_splitk does not take; kernels proven, outputs host-checked");
            for &(m, n, k, src) in UNGATED_SHAPES {
                let label = format!("splitk ungated M={m} N={n} K={k}");
                guarded(run, &label, || splitk_shape(rt, run, set, (m, n, k), Some(src)))?;
            }
        }
    }
    Ok(())
}

/// One TN shape of the `splitk` lane, overwriting then accumulating. `source`
/// is given for the opt-in ungated set, which adds the proofs and the host
/// reference ahead of the timing; the committed set runs as it always did.
fn splitk_shape(
    rt: &Arc<GpuRuntime>,
    run: &Run,
    set: SplitkSet,
    dims: (usize, usize, usize),
    source: Option<&str>,
) -> Result<(), String> {
    let (m, n, k) = dims;
    let widths = live_widths(set, dims);
    let checked = source.is_some();
    if checked {
        for w in WIDTHS.iter().filter(|w| !widths.contains(w)) {
            println!(
                "  (par_splitk_{w} not run at {m}x{n}x{k}: {})",
                if *w >= k {
                    "the width is not below K".to_string()
                } else {
                    format!(
                        "{} scratch floats, over production's cap of {PAR_SCRATCH_CAP}",
                        par_scratch_floats(dims, *w)
                    )
                }
            );
        }
    }
    let a = bf16(rt, &[k, m], 0x5a1 ^ (m * n + k) as u64)?;
    let b = bf16(rt, &[k, n], 0x5b2 ^ (m * n + k) as u64)?;
    let operands = if checked {
        Some((operand_values(rt, Elem::Bf16, &a)?, operand_values(rt, Elem::Bf16, &b)?))
    } else {
        None
    };
    for acc in [false, true] {
        // Outputs, checked against each other before anything is timed.
        let c0 = common::fill(m * n, 0x5c3 ^ k as u64);
        let fresh = || -> Result<Tensor, String> {
            let t = rt.alloc_tensor_f32(&[m, n])?;
            t.buffer.write_f32(&c0);
            Ok(t)
        };
        let cs: Vec<Tensor> = (0..2 + widths.len()).map(|_| fresh()).collect::<Result<_, _>>()?;
        let production = |c: &Tensor| {
            if acc {
                GemmOperands::Bf16.tn_acc(&a, &b, c)
            } else {
                gemm_tn_bf16(&a, &b, c)
            }
        };
        let what = format!("splitk {} {m}x{n}x{k}", if acc { "accumulate" } else { "overwrite" });
        if checked {
            let coop = coop_tn_kernel(acc);
            prove(
                rt,
                &format!("{what}: production"),
                &[coop],
                &[PAR_SPLITK_KERNEL],
                None,
                || production(&cs[0]),
            )?;
            prove(
                rt,
                &format!("{what}: coop"),
                &[coop],
                &[PAR_SPLITK_KERNEL],
                Some(1),
                || coop_tn(rt, &a, &b, &cs[1], dims, acc),
            )?;
            for (i, &w) in widths.iter().enumerate() {
                let kernels = [PAR_SPLITK_KERNEL, reduce_kernel(acc)];
                prove(rt, &format!("{what}: par_splitk_{w}"), &kernels, &[], Some(1), || {
                    par_splitk(rt, &a, &b, &cs[2 + i], dims, w, acc)
                })?;
            }
        } else {
            production(&cs[0])?;
            coop_tn(rt, &a, &b, &cs[1], dims, acc)?;
            for (i, &w) in widths.iter().enumerate() {
                par_splitk(rt, &a, &b, &cs[2 + i], dims, w, acc)?;
            }
        }
        rt.synchronize()?;
        let base = cs[0].buffer.read_f32()[..m * n].to_vec();
        let mut worst = 0.0f64;
        for c in &cs[1..] {
            let lane = format!("{m}x{n}x{k} acc={acc}");
            worst = worst.max(common::max_rel_diff(
                &lane,
                c.buffer.try_read_f32()?.get(..m * n).unwrap_or(&[]),
                &base,
            )?);
        }
        if worst >= 1e-4 {
            return Err(format!(
                "{m}x{n}x{k} acc={acc}: arms disagree by {worst:.2e} of the output's scale"
            ));
        }
        let mut host = String::new();
        if let Some((av, bv)) = &operands {
            for (i, c) in cs.iter().enumerate() {
                common::check_lane_output(&format!("{what} arm {i}"), &c.buffer.read_f32()[..m * n])?;
            }
            let err = host_ref_check(
                &what,
                &base,
                dims,
                &|i, p| av[p * m + i],
                &|p, j| bv[p * n + j],
                acc.then_some(c0.as_slice()),
                host_ref_tol(Elem::Bf16),
            )?;
            host = format!("; production within {err:.1e} of the host reference");
        }

        let reps = run.reps;
        let mut arms = vec![
            arm("production", rt, reps, || production(&cs[0])),
            arm(if acc { "coop_64x64" } else { "coop_128x64" }, rt, reps, || {
                coop_tn(rt, &a, &b, &cs[1], dims, acc)
            }),
        ];
        for (i, &w) in widths.iter().enumerate() {
            let c = &cs[2 + i];
            let (a, b) = (&a, &b);
            arms.push(arm(&format!("par_splitk_{w}"), rt, reps, move || {
                par_splitk(rt, a, b, c, dims, w, acc)
            }));
        }
        let label = format!(
            "splitk {} M={m} N={n} K={k}  (arms agree within {worst:.1e}{host}){}",
            if acc { "accumulate" } else { "overwrite" },
            source.map_or(String::new(), |s| format!("  [{s}]"))
        );
        let (paired, _) = report_paired(&label, &mut arms, run)?;
        if checked {
            print_verdicts(&paired, 0)?;
        }
    }
    Ok(())
}

/// The exact-f32 NN as it ran before the small-M lane: C zeroed, then one
/// `matmul2d_tensorops_f32` dispatch (32x32 tiles, one simdgroup each) over
/// all of K. Raw, so the comparison does not depend on what `gemm` routes to.
fn plain_nn(
    rt: &Arc<GpuRuntime>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    (m, n, k): (usize, usize, usize),
) -> Result<(), String> {
    // The kernel and the tile it is compiled for, together.
    const PLAIN_F32: (&str, usize, usize) = ("matmul2d_tensorops_f32", 32, 1);
    let (kernel, tile, simdgroups) = PLAIN_F32;
    let z = rt.pipeline("zero_f32")?;
    let p = rt.pipeline(kernel)?;
    let numel = c.numel();
    let z_tpt = z.threadExecutionWidth().min(numel).max(1);
    let (tiles_n, tiles_m) = (n.div_ceil(tile), m.div_ceil(tile));
    let tpt = p.threadExecutionWidth() * simdgroups;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&z);
        bnd.bind_tensor(c, 0);
        bnd.bind_u32(numel as u32, 1);
        bnd.dispatch(mtl_size(numel.div_ceil(z_tpt), 1, 1), mtl_size(z_tpt, 1, 1));
        if bnd.needs_explicit_barriers() {
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
        bnd.bind_u32(tiles_m as u32, 7);
        bnd.dispatch(mtl_size(tiles_n * tiles_m, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

/// The small-M exact-f32 NN (`gemm_nn_splitk_par_f32`) at the EmbeddingGemma 2
/// forward's shapes and row counts: the plain single dispatch, production
/// (`gemm`, whatever it routes to), and the parallel split-K at forced widths.
/// A layer's weight is one of 24 distinct copies cycled, as in a forward, so
/// the weights stream from memory rather than sit in cache.
fn nnsmall_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    let tile = 32;
    // (name, K, N): the forward's GEMMs.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("down", 2048, 512),
        ("o global", 2048, 512),
        ("o sliding", 1024, 512),
        ("ple_gate/out", 512, 512),
        ("qkv sliding", 512, 2048),
        ("qkv global", 512, 3072),
        ("gate|up", 512, 4096),
        // The audio tower's GEMMs in tests/embedgemma2_audio_tiny.rs's two
        // configurations: a B small enough to sit in cache.
        ("audio tiny ffn2", 256, 64),
        ("audio wide ffn2", 1024, 256),
        ("audio wide attn", 256, 256),
    ];
    const LAYERS: usize = 24;
    let only: Option<Vec<String>> = std::env::var("BENCH_SHAPES")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let ms: Vec<usize> = match std::env::var("BENCH_ROWS") {
        Ok(v) => v
            .split(',')
            .map(|r| r.trim().parse::<usize>().ok().filter(|&r| r > 0))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("BENCH_ROWS={v:?}: expected positive integers separated by commas"))?,
        Err(_) => vec![1, 16, 32, 64],
    };
    let widths: Vec<usize> = match std::env::var("BENCH_WIDTHS") {
        Ok(v) => v
            .split(',')
            .map(|r| r.trim().parse::<usize>().ok().filter(|&r| r > 0 && r % 4 == 0))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("BENCH_WIDTHS={v:?}: expected positive multiples of 4 separated by commas"))?,
        Err(_) => vec![64, 128, 256, 512, 1024],
    };
    for &(name, k, n) in SHAPES {
        // Without BENCH_SHAPES the forward's shapes only, as committed.
        let wanted = match &only {
            Some(o) => o.iter().any(|s| s == name),
            None => !name.starts_with("audio"),
        };
        if !wanted {
            continue;
        }
        for &m in &ms {
            let a = rt.alloc_tensor_f32(&[m, k])?;
            a.buffer.write_f32(&common::fill(m * k, 0x7a1 ^ (m * n + k) as u64));
            let ws: Vec<Tensor> = (0..LAYERS)
                .map(|l| {
                    let w = rt.alloc_tensor_f32(&[k, n])?;
                    w.buffer.write_f32(&common::fill(k * n, 0x7b2 ^ l as u64));
                    Ok(w)
                })
                .collect::<Result<_, String>>()?;
            let live: Vec<usize> = widths.iter().copied().filter(|&w| w < k).collect();
            let cs: Vec<Tensor> = (0..2 + live.len())
                .map(|_| rt.alloc_tensor_f32(&[m, n]))
                .collect::<Result<_, _>>()?;
            // Every arm computes the same product, up to the summation order.
            plain_nn(rt, &a, &ws[0], &cs[0], (m, n, k))?;
            gemm_small_m(&a, &ws[0], &cs[1], GemmBackend::TensorOps, None)?;
            for (i, &w) in live.iter().enumerate() {
                gemm_nn_splitk_par_f32(&a, &ws[0], &cs[2 + i], w)?;
            }
            rt.synchronize()?;
            let base = cs[0].buffer.read_f32()[..m * n].to_vec();
            let mut worst = 0.0f64;
            for c in &cs[1..] {
                let lane = format!("{name} M={m} N={n} K={k}");
                worst = worst.max(common::max_rel_diff(
                    &lane,
                    c.buffer.try_read_f32()?.get(..m * n).unwrap_or(&[]),
                    &base,
                )?);
            }
            if worst >= 1e-5 {
                return Err(format!(
                    "{name} M={m} N={n} K={k}: arms disagree by {worst:.2e} of the output's scale"
                ));
            }
            let reps = run.reps.div_ceil(LAYERS).max(1) * LAYERS;
            let mut arms = Vec::new();
            let mut i = 0usize;
            let (a_ref, ws_ref) = (&a, &ws);
            let c_plain = &cs[0];
            arms.push(arm("plain", rt, reps, move || {
                i += 1;
                plain_nn(rt, a_ref, &ws_ref[i % LAYERS], c_plain, (m, n, k))
            }));
            let mut i = 0usize;
            let c_prod = &cs[1];
            arms.push(arm("production", rt, reps, move || {
                i += 1;
                gemm_small_m(a_ref, &ws_ref[i % LAYERS], c_prod, GemmBackend::TensorOps, None)
            }));
            for (j, &w) in live.iter().enumerate() {
                let mut i = 0usize;
                let c = &cs[2 + j];
                arms.push(arm(&format!("par_splitk_{w}"), rt, reps, move || {
                    i += 1;
                    gemm_nn_splitk_par_f32(a_ref, &ws_ref[i % LAYERS], c, w)
                }));
            }
            let tiles = m.div_ceil(tile) * n.div_ceil(tile);
            report(
                &format!(
                    "nnsmall f32 {name} M={m} N={n} K={k}  ({tiles} tiles of {tile}x{tile}; arms agree within {worst:.1e})",
                ),
                &mut arms,
                run,
            )?;
        }
    }
    Ok(())
}

/// `values` (`[rows, cols]` row-major) uploaded as f32, then narrowed to `elem`.
fn upload_as(rt: &Arc<GpuRuntime>, elem: Elem, values: &[f32], shape: &[usize]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(values);
    match elem {
        Elem::Bf16 => cast_f32_to_bf16(&t),
        Elem::F16 => cast_f32_to_f16(&t),
        Elem::Relaxed => Ok(t),
    }
}

/// A named arm of the `sam` lane and the call it times.
type SamArm<'a> = (&'static str, Box<dyn Fn() -> Result<(), String> + 'a>);

/// SpikeTrack's SAM 3.1 PE ViT-L trunk linears, `x [M, K] @ W [N, K]^T + b`, at the
/// 1008 input (M = 72 x 72 tokens) and the 672 one (M = 48 x 48). Ojas runs them as
/// NT on `[N, K]` weights (`gemm_nt_f32`, or `gemm_nt_bf16` under Autocast) and adds
/// the bias in a second pass. The arms ask what the same product costs as NN on
/// weights transposed once at load (`gemm`, its tile rule, both tiles forced), on
/// f16 operands, and with the bias fused (`gemm_epilogue`, `beta = 0`, no
/// activation: GELU-erf is not an epilogue activation). Every arm computes the same
/// product of the same values, so outputs are compared to the bf16 NT arm (max
/// difference relative to its largest magnitude) rather than required bit-equal.
/// The last four arms are the exception: `gemm_nt_bf16_bias` is required bit-equal
/// to the three dispatches it replaces, with and without the bf16 round.
fn sam_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    // (N, K, label): q/k/v (one 1024-wide linear each), the fused qkv, proj, fc1, fc2.
    const NK: [(usize, usize, &str); 5] = [
        (1024, 1024, "q|k|v|proj"),
        (3072, 1024, "qkv fused"),
        (4736, 1024, "fc1"),
        (1024, 4736, "fc2"),
        (256, 1024, "neck 1x1"),
    ];
    for m in [5184usize, 2304] {
        for &(n, k, label) in &NK {
            let x = common::fill(m * k, 0x5a1 ^ (m * k) as u64);
            let w_nk = common::fill(n * k, 0x5b2 ^ (n * k) as u64);
            let mut w_kn = vec![0f32; n * k];
            for (r, row) in w_nk.chunks_exact(k).enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    w_kn[c * n + r] = v;
                }
            }
            let bias = rt.alloc_tensor_f32(&[n])?;
            bias.buffer.write_f32(&common::fill(n, 0x5c3));
            let (xf, wf_nk) = (
                upload_as(rt, Elem::Relaxed, &x, &[m, k])?,
                upload_as(rt, Elem::Relaxed, &w_nk, &[n, k])?,
            );
            rt.set_relaxed_precision(false);
            let (xb, wb_nk, wb_kn) = (
                upload_as(rt, Elem::Bf16, &x, &[m, k])?,
                upload_as(rt, Elem::Bf16, &w_nk, &[n, k])?,
                upload_as(rt, Elem::Bf16, &w_kn, &[k, n])?,
            );
            let (xh, wh_kn) = (
                upload_as(rt, Elem::F16, &x, &[m, k])?,
                upload_as(rt, Elem::F16, &w_kn, &[k, n])?,
            );
            // Zeroed: the epilogue arms read C (times beta = 0, and 0 x NaN is NaN).
            let cs: Vec<Tensor> = (0..12)
                .map(|_| {
                    let t = rt.alloc_tensor_f32(&[m, n])?;
                    t.buffer.write_f32(&vec![0f32; m * n]);
                    Ok(t)
                })
                .collect::<Result<_, String>>()?;
            let epi = Epilogue {
                alpha: 1.0,
                beta: 0.0,
                bias: Some(&bias),
                activation: Activation::None,
            };
            let halves = rt.alloc_tensor_bf16(&[m, n])?;
            let calls: [SamArm<'_>; 12] = [
                ("nt_bf16 (ojas)", Box::new(|| gemm_nt_bf16(&xb, &wb_nk, &cs[0]))),
                (
                    "nn_bf16",
                    Box::new(|| gemm(&xb, &wb_kn, &cs[1], GemmBackend::TensorOps)),
                ),
                (
                    "nn_bf16_128x64",
                    Box::new(|| gemm_tiled(&xb, &wb_kn, &cs[2], GemmBackend::TensorOps, EpiTile::Wide)),
                ),
                (
                    "nn_bf16_64x64",
                    Box::new(|| gemm_tiled(&xb, &wb_kn, &cs[3], GemmBackend::TensorOps, EpiTile::Narrow)),
                ),
                ("nn_f16", Box::new(|| gemm(&xh, &wh_kn, &cs[4], GemmBackend::TensorOps))),
                (
                    "nn_bf16+bias epi",
                    Box::new(|| gemm_epilogue(&xb, &wb_kn, &cs[5], GemmBackend::TensorOps, epi)),
                ),
                (
                    "nn_f16+bias epi",
                    Box::new(|| gemm_epilogue(&xh, &wh_kn, &cs[6], GemmBackend::TensorOps, epi)),
                ),
                (
                    "nt_f32 exact (ojas)",
                    Box::new(|| gemm_nt_f32(&xf, &wf_nk, &cs[7], GemmBackend::TensorOps)),
                ),
                (
                    "nt_bf16+bias 3-step",
                    Box::new(|| {
                        gemm_nt_bf16(&xb, &wb_nk, &cs[8])?;
                        bias_add(rt, &cs[8].buffer, &bias.buffer, m as u32, n as u32)
                    }),
                ),
                (
                    "nt_bf16+bias fused",
                    Box::new(|| gemm_nt_bf16_bias(&xb, &wb_nk, &cs[9], &bias, false)),
                ),
                (
                    "nt_bf16+bias+round 3-step",
                    Box::new(|| {
                        gemm_nt_bf16(&xb, &wb_nk, &cs[10])?;
                        bias_add(rt, &cs[10].buffer, &bias.buffer, m as u32, n as u32)?;
                        cast_f32_to_bf16_into(&cs[10], &halves)?;
                        cast_bf16_to_f32_into(&halves, &cs[10])
                    }),
                ),
                (
                    "nt_bf16+bias+round fused",
                    Box::new(|| gemm_nt_bf16_bias(&xb, &wb_nk, &cs[11], &bias, true)),
                ),
            ];
            for (_, f) in &calls {
                f()?;
            }
            rt.synchronize()?;
            // The fused kernel replaces the three dispatches bit for bit, or its time means nothing.
            for (three, fused) in [(8, 9), (10, 11)] {
                let (want, got) = (cs[three].buffer.read_f32(), cs[fused].buffer.read_f32());
                if let Some(i) = (0..m * n).find(|&i| want[i].to_bits() != got[i].to_bits()) {
                    return Err(format!(
                        "sam {label} M={m} N={n} K={k}: {} differs from {} at element {i}: {} vs {}",
                        calls[fused].0, calls[three].0, got[i], want[i]
                    ));
                }
            }
            let base = cs[0].buffer.read_f32();
            let mut diffs: Vec<String> = Vec::new();
            for ((name, _), c) in calls.iter().zip(&cs).skip(1) {
                let lane = format!("sam {label} M={m} N={n} K={k}: {name}");
                let e = common::max_rel_diff(
                    &lane,
                    c.buffer.try_read_f32()?.get(..m * n).unwrap_or(&[]),
                    &base[..m * n],
                )?;
                diffs.push(format!("{name} {e:.1e}"));
            }
            let reps = run.reps;
            let mut arms: Vec<common::Arm<'_>> = calls.iter().map(|(name, f)| arm(name, rt, reps, f)).collect();
            let medians = report(
                &format!(
                    "sam {label} M={m} N={n} K={k}  (max |diff| / max |nt_bf16|: {})",
                    diffs.join(", ")
                ),
                &mut arms,
                run,
            )?;
            let flop = 2.0 * (m * n * k) as f64;
            let rates: Vec<String> = calls
                .iter()
                .zip(&medians)
                .map(|((name, _), ms)| format!("{name} {:.1}", flop / (ms * 1e-3) / 1e12))
                .collect();
            println!("  TFLOP/s: {}", rates.join(", "));
        }
    }
    Ok(())
}

/// The plain-NN tile table of the `smallm` lane: M, and (N, K).
const SMALLM_MS: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];
const SMALLM_NK: [(usize, usize); 4] = [(3072, 768), (768, 3072), (4096, 4096), (2048, 2048)];

/// Lead 4's shapes `(m, n, k)`: relaxed-f32 plain NN at N <= 512 with large M,
/// where the plain rule takes 64x64. The first two are the batched shapes of
/// `gemm_coop_tiles_m5pro.txt` that lost at 64x64 (1024x256 x4: 0.90x, 2048x128
/// x8: 0.93x) as single matrices; the rest are the epilogue lane's N <= 512
/// shapes and larger M at the same N, K.
const RELAXED_PLAIN_SHAPES: &[(usize, usize, usize)] = &[
    (1024, 256, 256),
    (2048, 128, 128),
    (1024, 256, 1024),
    (4096, 128, 768),
    (2048, 384, 2048),
    (8192, 512, 768),
    (4096, 256, 256),
    (8192, 128, 128),
];

/// Wide (128x64) tiles of a C: the grid-fill measure of `coop_narrow`.
fn wide_tiles(m: usize, n: usize) -> usize {
    m.div_ceil(128) * n.div_ceil(64)
}

/// Whether production's plain NN takes the 64x64 tile: a copy of src/gemm.rs
/// `coop_narrow` (private) at the committed rule. The unit tests pin it to the
/// rule's documented cases; on the device the production arm's kernel name is
/// checked against it, so a rule that moves fails the run loudly.
fn production_narrow(elem: Elem, m: usize, n: usize) -> bool {
    n <= 512 || (elem == Elem::Bf16 && (m < 128 || wide_tiles(m, n) < 64))
}

/// One shape of the plain tile lanes (`smallm`, `relaxedplain`): 128x64 and 64x64
/// through `gemm_tiled` against production. With `checked` (the opt-in runs) it
/// first proves each arm's kernel and checks the output against a host
/// reference, and prints the win rule's verdict after the timings.
fn plain_tiles(
    rt: &Arc<GpuRuntime>,
    run: &Run,
    lane: &str,
    elem: Elem,
    (m, n, k): (usize, usize, usize),
    seeds: (u64, u64),
    checked: bool,
) -> Result<(), String> {
    let a = elem.operand(rt, &[m, k], seeds.0)?;
    let b = elem.operand(rt, &[k, n], seeds.1)?;
    let cs: Vec<Tensor> = (0..3).map(|_| rt.alloc_tensor_f32(&[m, n])).collect::<Result<_, _>>()?;
    let production = || gemm(&a, &b, &cs[0], GemmBackend::TensorOps);
    let wide = || gemm_tiled(&a, &b, &cs[1], GemmBackend::TensorOps, EpiTile::Wide);
    let narrow = || gemm_tiled(&a, &b, &cs[2], GemmBackend::TensorOps, EpiTile::Narrow);
    let what = format!("{lane} {} {m}x{n}x{k}", elem.name());
    if checked {
        let (wide_k, narrow_k) = (elem.tile_kernel(false), elem.tile_kernel(true));
        let (wide_k, narrow_k) = (wide_k.as_str(), narrow_k.as_str());
        let (took, other) = if production_narrow(elem, m, n) {
            (narrow_k, wide_k)
        } else {
            (wide_k, narrow_k)
        };
        prove(
            rt,
            &format!("{what}: production"),
            &[took],
            &[other],
            Some(1),
            production,
        )?;
        prove(
            rt,
            &format!("{what}: wide_128x64"),
            &[wide_k],
            &[narrow_k],
            Some(1),
            wide,
        )?;
        prove(
            rt,
            &format!("{what}: narrow_64x64"),
            &[narrow_k],
            &[wide_k],
            Some(1),
            narrow,
        )?;
    } else {
        production()?;
        wide()?;
        narrow()?;
    }
    rt.synchronize()?;
    let base = cs[0].buffer.read_f32();
    for c in &cs[1..] {
        if c.buffer.read_f32()[..m * n] != base[..m * n] {
            return Err(format!("M={m} N={n} K={k}: the tiles' outputs differ"));
        }
    }
    let mut host = String::new();
    if checked {
        for (i, c) in cs.iter().enumerate() {
            common::check_lane_output(&format!("{what} arm {i}"), &c.buffer.read_f32()[..m * n])?;
        }
        let (av, bv) = (operand_values(rt, elem, &a)?, operand_values(rt, elem, &b)?);
        let err = host_ref_check(
            &what,
            &base[..m * n],
            (m, n, k),
            &|i, p| av[i * k + p],
            &|p, j| bv[p * n + j],
            None,
            host_ref_tol(elem),
        )?;
        host = format!("; within {err:.1e} of the host reference");
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
    // The committed lane is plain bf16 and says so by saying nothing.
    let tag = if checked || elem != Elem::Bf16 {
        format!(" {}", elem.name())
    } else {
        String::new()
    };
    let label = format!(
        "{lane}{tag} M={m} N={n} K={k}  ({} tiles of 128x64; outputs bit-identical{host})",
        wide_tiles(m, n)
    );
    let (paired, _) = report_paired(&label, &mut arms, run)?;
    if checked {
        print_verdicts(&paired, 2)?;
    }
    Ok(())
}

fn smallm_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    let knob = env_knob("BENCH_SMALLM_ELEMS")?;
    let checked = knob.is_some();
    let elems = parse_elems(knob.as_deref())?;
    if checked {
        require_trace("BENCH_SMALLM_ELEMS")?;
        println!(
            "\nsmallm on {}: kernels proven, outputs host-checked",
            knob.as_deref().unwrap_or_default()
        );
    }
    for elem in elems {
        for &(n, k) in &SMALLM_NK {
            for &m in &SMALLM_MS {
                let seeds = (0x6a1 ^ (m * n + k) as u64, 0x6b2 ^ (n + k) as u64);
                if checked {
                    let label = format!("smallm {} M={m} N={n} K={k}", elem.name());
                    guarded(run, &label, || {
                        plain_tiles(rt, run, "smallm", elem, (m, n, k), seeds, true)
                    })?;
                } else {
                    plain_tiles(rt, run, "smallm", elem, (m, n, k), seeds, false)?;
                }
            }
        }
    }
    rt.set_relaxed_precision(false);
    Ok(())
}

/// Lead 4: relaxed-f32 plain NN, 128x64 against production's 64x64, on large M at N <= 512.
fn relaxedplain_lane(rt: &Arc<GpuRuntime>, run: &Run) -> Result<(), String> {
    require_trace("lane relaxedplain")?;
    println!("\nrelaxedplain: relaxed f32, N <= 512 on large M; kernels proven, outputs host-checked");
    for &(m, n, k) in RELAXED_PLAIN_SHAPES {
        let seeds = (0x9a1 ^ (m * n + k) as u64, 0x9b2 ^ (n + k) as u64);
        let label = format!("relaxedplain f32_relaxed M={m} N={n} K={k}");
        guarded(run, &label, || {
            plain_tiles(rt, run, "relaxedplain", Elem::Relaxed, (m, n, k), seeds, true)
        })?;
    }
    rt.set_relaxed_precision(false);
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
        refused: Default::default(),
    };
    let lanes = std::env::var("BENCH_LANES").unwrap_or_else(|_| "splitk,smallm,nnsmall,batched,epi".into());
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
            "relaxedplain" => relaxedplain_lane(&rt, &run)?,
            "nnsmall" => nnsmall_lane(&rt, &run)?,
            "batched" => batched_lane(&rt, &run)?,
            "epi" => epi_lane(&rt, &run)?,
            "sam" => sam_lane(&rt, &run)?,
            other => {
                return Err(format!(
                    "BENCH_LANES: no lane {other:?} (splitk, smallm, nnsmall, batched, epi, sam, relaxedplain)"
                ))
            }
        }
    }
    rt.set_async_encode(false)?;
    common::emit_kernel_trace();
    let refused = run.refused.borrow();
    if !refused.is_empty() {
        eprintln!(
            "bench_gemm_routes: {} shape(s) failed a pre-timing check and were not timed:\n  {}",
            refused.len(),
            refused.join("\n  ")
        );
        std::process::exit(EXIT_REFUSED);
    }
    common::exit_if_unstable("bench_gemm_routes", &run.unstable.borrow());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// src/gemm.rs as committed: the copies in this bin are pinned to its text.
    const GEMM_RS: &str = include_str!("../gemm.rs");

    #[test]
    fn the_prefer_tn_splitk_copy_matches_the_committed_gated_table() {
        let (control, gated) = GATED_SHAPES.split_last().unwrap();
        for &(m, n, k) in gated {
            assert!(prefer_tn_splitk(m, n, k), "{m}x{n}x{k} is a gated shape");
        }
        assert!(
            !prefer_tn_splitk(control.0, control.1, control.2),
            "the control is not gated"
        );
        // Each clause of the predicate decides something.
        assert!(!prefer_tn_splitk(128, 128, 2047));
        assert!(!prefer_tn_splitk(385, 128, 4096));
        assert!(!prefer_tn_splitk(128, 385, 4096));
        assert!(!prefer_tn_splitk(129, 129, 4096));
        assert!(prefer_tn_splitk(384, 128, 2048));
    }

    #[test]
    fn the_copies_are_pinned_to_the_text_of_src_gemm_rs() {
        for needle in [
            "k >= 2048 && m <= 384 && n <= 384 && m.min(n) <= 128",
            "const TN_PAR_MAX_SCRATCH: usize = 1 << 22;",
            "const COOP_GRID_FILL_TILES: usize = 64;",
            "if n <= 512 {",
            "elem == CoopElem::Bf16",
            "< COOP_GRID_FILL_TILES",
            "m < TILE_COOP_DEFAULT.sm",
            "\"matmul2d_tensorops_tn_splitk_par_bf16_f32\"",
            "\"matmul2d_tensorops_tn_bf16_f32\"",
            "\"matmul2d_tensorops_tn_accum_bf16_f32\"",
            "\"reduce_partitions_f32\"",
            "\"reduce_partitions_add_f32\"",
        ] {
            assert!(
                GEMM_RS.contains(needle),
                "src/gemm.rs no longer contains {needle:?}: re-pin the copy"
            );
        }
        for elem in ELEMS {
            for narrow in [false, true] {
                let name = elem.tile_kernel(narrow);
                assert!(
                    GEMM_RS.contains(&format!("\"{name}\"")),
                    "src/gemm.rs has no kernel {name}"
                );
            }
        }
    }

    #[test]
    fn every_ungated_shape_is_outside_the_gate_sourced_distinct_and_runnable() {
        let mut seen = std::collections::HashSet::new();
        for &(m, n, k, source) in UNGATED_SHAPES {
            assert!(!prefer_tn_splitk(m, n, k), "{m}x{n}x{k} is gated");
            assert!(!source.is_empty(), "{m}x{n}x{k} has no source");
            assert!(seen.insert((m, n, k)), "{m}x{n}x{k} twice");
            assert!(
                !GATED_SHAPES.contains(&(m, n, k)),
                "{m}x{n}x{k} is in the committed set"
            );
            assert_eq!(k % 4, 0, "{m}x{n}x{k}: the kernel takes K in multiples of 4");
            assert!(
                !live_widths(SplitkSet::Ungated, (m, n, k)).is_empty(),
                "{m}x{n}x{k}: no width fits"
            );
        }
        // The callers the lead names: the 2B's gate dW both ways round, and EmbeddingGemma 2's.
        for (m, n) in [
            (2048, 16),
            (16, 2048),
            (2048, 512),
            (512, 512),
            (1024, 512),
            (512, 2048),
        ] {
            assert!(
                UNGATED_SHAPES.iter().any(|s| (s.0, s.1) == (m, n)),
                "no {m}x{n} shape in the ungated set"
            );
        }
    }

    #[test]
    fn ungated_widths_stay_below_k_and_inside_production_scratch_cap() {
        assert_eq!(live_widths(SplitkSet::Ungated, (2048, 512, 4096)), vec![1024]);
        assert_eq!(live_widths(SplitkSet::Ungated, (512, 512, 4096)), vec![256, 512, 1024]);
        assert_eq!(live_widths(SplitkSet::Ungated, (128, 128, 1024)), vec![256, 512]);
        // The committed set always ran every width, over the cap or not.
        assert!(par_scratch_floats((128, 384, 32768), 256) > PAR_SCRATCH_CAP);
        assert_eq!(live_widths(SplitkSet::Gated, (128, 384, 32768)), WIDTHS.to_vec());
    }

    #[test]
    fn the_splitk_set_knob_defaults_to_the_committed_set_and_refuses_anything_else() {
        assert_eq!(SplitkSet::parse(None), Ok(SplitkSet::Gated));
        assert_eq!(SplitkSet::parse(Some("gated")), Ok(SplitkSet::Gated));
        assert_eq!(SplitkSet::parse(Some("ungated")), Ok(SplitkSet::Ungated));
        for bad in ["", "Ungated", "all", "ungated,gated", " ungated"] {
            assert!(SplitkSet::parse(Some(bad)).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn the_smallm_elems_knob_defaults_to_bf16_keeps_order_and_refuses_unknown_or_repeated() {
        assert_eq!(parse_elems(None), Ok(vec![Elem::Bf16]));
        assert_eq!(
            parse_elems(Some("f32_relaxed, f16,bf16")),
            Ok(vec![Elem::Relaxed, Elem::F16, Elem::Bf16])
        );
        for bad in ["", "relaxed", "f32", "bf16,bf16", "bf16,", "f16;bf16"] {
            assert!(parse_elems(Some(bad)).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn the_smallm_table_keeps_the_committed_shapes_and_covers_the_reconciled_grid_fill_ones() {
        assert_eq!(SMALLM_MS, [16, 32, 64, 128, 256, 512, 1024]);
        assert_eq!(SMALLM_NK, [(3072, 768), (768, 3072), (4096, 4096), (2048, 2048)]);
        // bf16_smallm_coop_reconciled_m5pro.txt: 64x64 won under 64 wide tiles at M >= 128 ...
        for (m, n, tiles) in [(128, 3072, 48), (128, 768, 12), (256, 768, 24), (512, 768, 48)] {
            assert!(SMALLM_MS.contains(&m) && SMALLM_NK.iter().any(|&(nn, _)| nn == n));
            assert_eq!(wide_tiles(m, n), tiles);
            assert!(wide_tiles(m, n) < 64 && m >= 128 && n > 512);
        }
        // ... and tied or lost from 64 up: the controls.
        for (m, n, tiles) in [(128, 4096, 64), (1024, 4096, 512), (1024, 768, 96)] {
            assert!(SMALLM_MS.contains(&m) && SMALLM_NK.iter().any(|&(nn, _)| nn == n));
            assert_eq!(wide_tiles(m, n), tiles);
        }
    }

    #[test]
    fn the_production_tile_replica_follows_the_documented_rule() {
        let (bf, h, r) = (Elem::Bf16, Elem::F16, Elem::Relaxed);
        // bf16: N <= 512, M < 128, or under 64 wide tiles.
        assert!(production_narrow(bf, 128, 3072)); // 48 tiles
        assert!(production_narrow(bf, 256, 768)); // 24
        assert!(production_narrow(bf, 64, 3072)); // M < 128
        assert!(!production_narrow(bf, 128, 4096)); // exactly 64 tiles
        assert!(!production_narrow(bf, 1024, 4096));
        assert!(!production_narrow(bf, 200, 4096)); // 128 tiles
                                                    // f16 and relaxed f32 keep the N rule: the lead.
        for e in [h, r] {
            assert!(!production_narrow(e, 128, 3072));
            assert!(!production_narrow(e, 16, 3072));
            assert!(!production_narrow(e, 256, 768));
            assert!(!production_narrow(e, 128, 513));
            assert!(production_narrow(e, 128, 512));
            assert!(production_narrow(e, 8192, 512));
            assert!(production_narrow(e, 2048, 128));
        }
    }

    #[test]
    fn the_relaxedplain_shapes_are_the_n_512_territory_with_the_two_named_ones() {
        for want in [(1024, 256, 256), (2048, 128, 128)] {
            assert!(RELAXED_PLAIN_SHAPES.contains(&want), "{want:?} is missing");
        }
        for &(m, n, k) in RELAXED_PLAIN_SHAPES {
            assert!(n <= 512, "{m}x{n}x{k} is past the N rule");
            assert!(
                production_narrow(Elem::Relaxed, m, n),
                "production takes 64x64 at {m}x{n}x{k}"
            );
            assert!(m >= 1024 && k > 0);
        }
        assert!(
            RELAXED_PLAIN_SHAPES
                .iter()
                .filter(|&&(m, n, _)| wide_tiles(m, n) >= 64)
                .count()
                >= 3,
            "not enough large grids"
        );
    }

    #[test]
    fn the_tile_kernel_names_are_the_ones_the_arms_and_proofs_expect() {
        assert_eq!(Elem::Bf16.tile_kernel(false), "matmul2d_tensorops_bf16_f32");
        assert_eq!(Elem::Bf16.tile_kernel(true), "matmul2d_tensorops_bf16_f32_64x64_sg4");
        assert_eq!(Elem::F16.tile_kernel(false), "matmul2d_tensorops_f16_f32");
        assert_eq!(Elem::F16.tile_kernel(true), "matmul2d_tensorops_f16_f32_64x64_sg4");
        assert_eq!(Elem::Relaxed.tile_kernel(false), "matmul2d_tensorops_f32_relaxed");
        assert_eq!(
            Elem::Relaxed.tile_kernel(true),
            "matmul2d_tensorops_f32_relaxed_64x64_sg4"
        );
        assert_eq!(coop_tn_kernel(false), "matmul2d_tensorops_tn_bf16_f32");
        assert_eq!(coop_tn_kernel(true), "matmul2d_tensorops_tn_accum_bf16_f32");
        assert_eq!(reduce_kernel(false), "reduce_partitions_f32");
        assert_eq!(reduce_kernel(true), "reduce_partitions_add_f32");
        assert_eq!(PAR_SPLITK_KERNEL, "matmul2d_tensorops_tn_splitk_par_bf16_f32");
    }

    #[test]
    fn the_kernel_proof_demands_the_kernel_forbids_the_other_and_counts_dispatches() {
        let ok =
            |expect: &[(&str, u64)], forbid: &[(&str, u64)], want, got| judge_proof("w", expect, forbid, want, got);
        assert!(ok(&[("a", 1)], &[("b", 0)], Some(1), 1).is_ok());
        assert!(ok(&[("a", 3)], &[], None, 7).is_ok(), "no count asked, none checked");
        assert!(
            ok(&[("a", 0)], &[("b", 0)], Some(1), 1).is_err(),
            "the expected kernel never ran"
        );
        assert!(ok(&[("a", 1)], &[("b", 1)], Some(1), 1).is_err(), "the other tile ran");
        assert!(
            ok(&[("a", 1), ("c", 0)], &[], None, 1).is_err(),
            "every expected kernel counts"
        );
        assert!(ok(&[("a", 1)], &[], Some(1), 2).is_err());
        assert!(ok(&[("a", 1)], &[], Some(1), 0).is_err());
    }

    #[test]
    fn the_win_rule_needs_the_band_the_rounds_and_a_calm_baseline() {
        let prod = |t: f64| vec![t; 6];
        // 1.25x faster in every round, no spread: the alternative wins.
        assert_eq!(
            route_verdict(&prod(10.0), &prod(8.0), 0.0).unwrap().outcome,
            Outcome::AltFaster
        );
        // 1.04x is inside the 0.05 floor; 1.06x is outside it.
        assert_eq!(
            route_verdict(&prod(10.4), &prod(10.0), 0.0).unwrap().outcome,
            Outcome::Inconclusive
        );
        assert_eq!(
            route_verdict(&prod(10.6), &prod(10.0), 0.0).unwrap().outcome,
            Outcome::AltFaster
        );
        // The band widens to twice the spread: 1.12x is noise at 8%, a win at 2%.
        assert_eq!(
            route_verdict(&prod(11.2), &prod(10.0), 0.08).unwrap().outcome,
            Outcome::Inconclusive
        );
        assert_eq!(
            route_verdict(&prod(11.2), &prod(10.0), 0.02).unwrap().outcome,
            Outcome::AltFaster
        );
        // The mirror confirms production.
        assert_eq!(
            route_verdict(&prod(8.0), &prod(10.0), 0.0).unwrap().outcome,
            Outcome::ProductionFaster
        );
        // Beyond the band on the median, but faster in 3 of 5 rounds: below 80%.
        let (p, a) = ([13.0, 13.0, 13.0, 9.0, 9.0], [10.0; 5]);
        let v = route_verdict(&p, &a, 0.0).unwrap();
        assert_eq!((v.outcome, v.faster, v.rounds), (Outcome::Inconclusive, 3, 5));
        // 4 of 5 passes.
        let p = [13.0, 13.0, 13.0, 13.0, 9.0];
        assert_eq!(route_verdict(&p, &a, 0.0).unwrap().outcome, Outcome::AltFaster);
        // A baseline past the 10% self-spread limit is not evidence.
        assert_eq!(
            route_verdict(&prod(20.0), &prod(10.0), 0.11).unwrap().outcome,
            Outcome::Excluded
        );
        // Refusals: nothing, mismatched rounds, a NaN spread, a zero time.
        assert!(route_verdict(&[], &[], 0.0).is_err());
        assert!(route_verdict(&[1.0, 1.0], &[1.0], 0.0).is_err());
        assert!(route_verdict(&[1.0], &[1.0], f64::NAN).is_err());
        assert!(route_verdict(&[1.0], &[0.0], 0.0).is_err());
    }

    #[test]
    fn the_host_reference_gate_accepts_the_product_and_refuses_everything_else() {
        let (m, n, k) = (2usize, 2usize, 3usize);
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [m, k]
        let b = [0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5]; // [k, n]
        let a_at = |i: usize, p: usize| a[i * k + p];
        let b_at = |p: usize, j: usize| b[p * n + j];
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| a_at(i, p) * b_at(p, j)).sum();
            }
        }
        let dims = (m, n, k);
        assert_eq!(
            host_ref_check("t", &c, dims, &a_at, &b_at, None, HOST_REF_TOL).unwrap(),
            0.0
        );
        // A sampled element off by 5% of the scale, a NaN, a short output.
        let mut off = c.clone();
        off[3] += 0.05 * c.iter().fold(0f32, |s, v| s.max(v.abs()));
        assert!(host_ref_check("t", &off, dims, &a_at, &b_at, None, HOST_REF_TOL).is_err());
        let mut nan = c.clone();
        nan[1] = f32::NAN;
        assert!(host_ref_check("t", &nan, dims, &a_at, &b_at, None, HOST_REF_TOL).is_err());
        assert!(host_ref_check("t", &c[..3], dims, &a_at, &b_at, None, HOST_REF_TOL).is_err());
        // Within the 2e-3 band: 1e-4 of the scale passes.
        let mut near = c.clone();
        near[0] += 1e-4 * c.iter().fold(0f32, |s, v| s.max(v.abs()));
        assert!(host_ref_check("t", &near, dims, &a_at, &b_at, None, HOST_REF_TOL).is_ok());
        // C += starts from a base; leaving the base out is caught.
        let base = [10.0f32, -20.0, 30.0, -40.0];
        let summed: Vec<f32> = c.iter().zip(&base).map(|(x, y)| x + y).collect();
        assert!(host_ref_check("t", &summed, dims, &a_at, &b_at, Some(&base), HOST_REF_TOL).is_ok());
        assert!(host_ref_check("t", &summed, dims, &a_at, &b_at, None, HOST_REF_TOL).is_err());
        // A reference of all zeros has no scale.
        assert!(host_ref_check("t", &[0.0; 4], dims, &|_, _| 0.0, &b_at, None, HOST_REF_TOL).is_err());
        // Relaxed f32 is held to the tree's bf16-class bound, the others to 2e-3: 3e-3 of the scale
        // passes only the relaxed gate.
        let mut loose = c.clone();
        loose[0] += 3e-3 * c.iter().fold(0f32, |s, v| s.max(v.abs()));
        assert!(host_ref_check("t", &loose, dims, &a_at, &b_at, None, host_ref_tol(Elem::Relaxed)).is_ok());
        assert!(host_ref_check("t", &loose, dims, &a_at, &b_at, None, host_ref_tol(Elem::Bf16)).is_err());
        assert!(host_ref_check("t", &loose, dims, &a_at, &b_at, None, host_ref_tol(Elem::F16)).is_err());
        assert_eq!(host_ref_tol(Elem::Relaxed), 1.0 / 256.0);
    }
}
