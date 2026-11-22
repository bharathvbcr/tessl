//! Timing lanes for the GEMM kernels the cross-runtime sweep never reaches.
//!
//! `bench_gemm_sweep` covers the NN layout, which is the only one torch and MLX
//! expose a comparable primitive for. Everything else in
//! `matmul_tensorops.metal` — TN, NT, their accumulating and split-K forms,
//! strided-batched, the fused epilogue, and the f16 operand path — had no
//! timing lane at all: 18 of the 84 kernel entry points, including the two
//! layouts the backward pass runs on.
//!
//! Correctness for these paths lives in `tests/`, not here: the lanes are
//! timing and coverage only, except the three below that no test pins to an
//! absolute reference. Each lane's covering test (a `#[test]` of the file
//! named, run with `cargo test --release --test <file> <name>`; `--lib` for the
//! one in `src/gemm.rs`):
//!
//! | lane | test |
//! |---|---|
//! | `tn_f32_simdgroup` | `--lib gemm::tests::transpose_edges_precision_and_accumulation` (TN, `GemmBackend::Simdgroup`, F32) |
//! | `tn_f32`, `nt_f32` | `gemm_correctness`: `tn_f32_matches_cpu_reference`, `nt_f32_matches_cpu_reference` |
//! | `tn_bf16`, `nt_bf16` | `gemm_correctness`: `tn_bf16_matches_cpu_reference`, `nt_bf16_matches_cpu_reference` |
//! | `tn_accum_f32`, `nt_accum_f32`, `tn_accum_bf16`, `nt_accum_bf16` | `gemm_operands_accumulate`: `accumulate_operands_add_the_product_into_c` (`GemmOperands::{tn,nt}_acc`, which the `gemm_*_accum_train` entry points reach through the same `tn_accum` / `nt_accum`); also `--lib gemm::tests::transpose_edges_precision_and_accumulation` |
//! | `tn_splitk_f32` | `gemm_correctness`: `tn_splitk_matches_cpu_reference` (`128x128x2048` takes `gemm_tn_splitk_par_f32`, as `128x256x4096` does); `gemm_ragged_shapes`: `parallel_tn_partitions_cover_all_of_k` |
//! | `tn_splitk_bf16` | `gemm_correctness`: `tn_splitk_matches_cpu_reference` (its bf16 half) |
//! | `tn_accum_splitk_f32` | `gemm_operands_accumulate`: `accumulate_operands_add_the_product_into_c`, the `96x200x2304` TN case (sequential exact-f32 split-K) |
//! | `nn_f16_narrow` | `f16`: `f16_gemm_matches_an_f32_reference_within_f16_resolution` (N = 64, 80, the 64x64 tile) |
//! | `batched_f32r`, `batched_bf16`, `batched_f16` | `gemm_batched`: `every_batch_element_equals_a_single_gemm_bit_for_bit` (relaxed f32), `bf16_and_f16_batch_elements_match_single_gemm_bit_for_bit`; each batch element is bit-equal to a single GEMM, whose own tests are the NN ones above and in `gemm_correctness` (`relaxed_precision_stays_within_a_bf16_class_bound`, `nn_bf16_matches_cpu_reference`) and `f16` |
//! | `batched_bf16_accum` | the same call as `batched_bf16` (same operands, entry point and precision mode), so the same test; the lane is a duplicate timing of it |
//! | `epi_f32r` | `gemm_epilogue`: `fused_epilogue_matches_the_unfused_sequence` (relaxed f32, every activation, against an f32 host reference) |
//! | `epi_f32` | `gemm_exact_f32_epilogue`: `exact_f32_wide_tile_by_the_production_rule_is_bit_identical` (N > 512, the 128x64 exact kernel, as N = 1024 here) |
//! | `epi_f32_narrow` | `gemm_exact_f32_epilogue`: `exact_f32_fused_epilogue_is_bit_identical_to_the_separate_passes` (N <= 512, the 64x64 exact kernel) |
//! | `q4_mlx_tensorops_g32/g64/g128` | `gemm_q4_mlx_tensorops`: `matches_gemm_q4_mlx_and_f64_on_random_activations`, `matches_at_a_long_k` (K = 2048, all three group sizes, against `gemm_q4_mlx` and f64) |
//!
//! Three lanes have no test at their shape against an absolute reference, so
//! this binary checks them before timing, at 512 sampled elements against an f64
//! host reference on operands rounded to the lane's 16-bit format
//! (`check_narrow_nn`): `nn_f16` (N = 1024, the wide f16 tile: `f16.rs` reaches
//! only N <= 80), `epi_f16` (`gemm_short_m_attack` and `gemm_epilogue` pin the f16
//! epilogue's 64x64 tile to the 128x64 tile's bits, not to a reference) and
//! `epi_bf16` at the bench's `1024x1024x1024` GELU shape (`gemm_epilogue`'s bf16
//! cases compare the two tiles with each other, and a GELU probe at `K = 64`;
//! `gemm_short_m_attack`'s Silu reference stops at M = 128). `nn_f16_narrow` is
//! checked too, cheaply, since it shares the helper. The device command for all
//! of it is `cargo run --release --bin bench_gemm_variants`, which fails before
//! timing on a mismatch. The timings themselves are not a correctness claim.
//!
//! There is no cross-runtime lane here on purpose: `a.T @ b` in torch or MLX
//! may materialise the transpose rather than fuse it, so a ratio would compare
//! tessl's fused kernel against transpose-plus-GEMM and read as a kernel result.

mod common;

use common::{env_usize, fill_dist, median, Dist};
use std::sync::Arc;
use std::time::Instant;
use tessl::gemm::{
    cast_f32_to_bf16, cast_f32_to_f16, gemm, gemm_batched, gemm_epilogue, gemm_nt_accum_train, gemm_nt_f32,
    gemm_nt_train, gemm_tn_accum_train, gemm_tn_f32, gemm_tn_train, Activation, BatchStrides, BatchedGemm, Epilogue,
    GemmBackend,
};
use tessl::nn::{gemm_q4_mlx, gemm_q4_mlx_tensorops, gemm_q8_tensorops, Q4MlxBank, Q4MlxLayout, QuantShape};
use tessl::runtime::{GpuRuntime, PrecisionMode};
use tessl::tensor::{f32_slice_to_bf16, GpuBuffer, Tensor};

/// (M, N, K). One general shape, plus a split-K-eligible one: `prefer_tn_splitk`
/// wants K >= 2048, M and N <= 384 and min(M, N) <= 128, so a single shape
/// cannot reach both the descriptor and the split-K TN kernels.
const GENERAL: (usize, usize, usize) = (1024, 1024, 1024);
const SPLITK: (usize, usize, usize) = (128, 256, 4096);
const BATCHED: (usize, usize, usize, usize) = (256, 256, 256, 8);
/// (M, N, K) of the MLX Q4 TensorOps lanes: M past `GEMM_Q4_MLX_MAX_M`, N and K
/// multiples of every group size in `GEMM_Q4_MLX_TENSOROPS_GROUP_SIZES`.
const Q4_TENSOROPS: (usize, usize, usize) = (256, 1024, 2048);

/// An MLX Q4 bank over seeded random nibbles with finite bf16 scale/bias pairs
/// (`Q4MlxBank` packing: two nibbles per byte, low first; one `bfloat2` per
/// group), plus a bf16 activation buffer. Values are inputs to a timing lane,
/// not a reference.
fn q4_mlx_operands(
    rt: &Arc<GpuRuntime>,
    (m, n, k): (usize, usize, usize),
    group: usize,
) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer), String> {
    let packed = rt.alloc_buffer(n * k / 2)?;
    let bytes: Vec<u8> = fill_dist(n * k / 2, 21, Dist::Uniform)
        .iter()
        .map(|v| ((v + 1.0) * 127.5) as u8)
        .collect();
    packed.write_bytes(&bytes);
    let sb = rt.alloc_buffer(n * (k / group) * 4)?;
    sb.write_bf16_bits(&f32_slice_to_bf16(&fill_dist(n * (k / group) * 2, 22, Dist::Uniform)));
    let x = rt.alloc_buffer(m * k * 2)?;
    x.write_bf16_bits(&f32_slice_to_bf16(&fill_dist(m * k, 23, Dist::Uniform)));
    Ok((packed, sb, x))
}

fn f32_tensor(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let n: usize = shape.iter().product();
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(&fill_dist(n, seed, Dist::Uniform));
    Ok(t)
}

/// What an f16 NN lane's result is checked against: C_prev, bias and the
/// scalars of the fused epilogue (`act(alpha * A@B + beta * C_prev + bias)`,
/// GELU-tanh), or nothing for the plain GEMM.
struct NarrowEpilogue<'a> {
    alpha: f64,
    beta: f64,
    c_prev: &'a [f32],
    bias: &'a [f32],
}

/// The 16-bit operand format of a checked lane.
#[derive(Clone, Copy)]
enum Narrow {
    F16,
    Bf16,
}

/// `got` (an `[m, n]` f32 result of 16-bit operands `a_f32 [m, k]` and `b_f32
/// [k, n]`) against an f64 reference at 512 sampled elements. The operands are
/// rounded to `narrow` on the host first, as the kernel sees them, so the only
/// error left is the f32 accumulation order: `tests/f16.rs` bounds it at `8 *
/// eps * k` and so does this, scaled by `max(1, |alpha|)` and 1.2 (the GELU slope
/// bound) when an epilogue is fused. Sampled, not exhaustive: a kernel that is
/// wrong on a whole tile, a stride or a column is caught; one wrong at a single
/// unsampled element is not.
fn check_narrow_nn(
    lane: &str,
    narrow: Narrow,
    (a_f32, b_f32): (&[f32], &[f32]),
    got: &[f32],
    (m, n, k): (usize, usize, usize),
    epi: Option<&NarrowEpilogue>,
) -> Result<(), String> {
    let round = |v: &[f32]| -> Vec<f64> {
        match narrow {
            Narrow::F16 => tessl::tensor::f32_slice_to_f16(v)
                .into_iter()
                .map(|h| f64::from(tessl::tensor::f16_bits_to_f32(h)))
                .collect(),
            Narrow::Bf16 => f32_slice_to_bf16(v)
                .into_iter()
                .map(|h| f64::from(tessl::tensor::bf16_bits_to_f32(h)))
                .collect(),
        }
    };
    let (a, b) = (round(a_f32), round(b_f32));
    if got.len() < m * n {
        return Err(format!("{lane}: {} elements read back, need {}", got.len(), m * n));
    }
    let gelu = |x: f64| 0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh());
    let scale = epi.map_or(1.0, |e| e.alpha.abs().max(1.0) * 1.2);
    let bound = (8.0 * f64::from(f32::EPSILON) * k as f64).max(1e-3) * scale;
    // Rows and columns walk co-prime strides, so the samples cover every row
    // residue and the last (ragged-edge) rows and columns are included.
    let mut samples: Vec<(usize, usize)> = (0..508).map(|s| ((s * 37 + 11) % m, (s * 101 + 5) % n)).collect();
    samples.extend([(0, 0), (m - 1, 0), (0, n - 1), (m - 1, n - 1)]);
    let mut worst = 0.0f64;
    for (i, j) in samples {
        let dot: f64 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        let want = match epi {
            None => dot,
            Some(e) => gelu(e.alpha * dot + e.beta * f64::from(e.c_prev[i * n + j]) + f64::from(e.bias[j])),
        };
        let err = (f64::from(got[i * n + j]) - want).abs();
        // NaN result fails too.
        if err.is_nan() || err > bound {
            return Err(format!(
                "{lane}: C[{i},{j}] is {}, host reference {want} (|delta| {err:e} > {bound:e})",
                got[i * n + j]
            ));
        }
        if err > worst {
            worst = err;
        }
    }
    eprintln!("{lane}: host reference ok at 512 samples, worst |delta| {worst:.3e} (bound {bound:.3e})");
    Ok(())
}

/// A named lane and the closure that dispatches it once.
struct Lane<'a> {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    run: Box<dyn Fn() -> Result<(), String> + 'a>,
}

fn time_lane(rt: &Arc<GpuRuntime>, l: &Lane, warmup: usize, iters: usize) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        (l.run)()?;
        rt.synchronize()?;
    }
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        (l.run)()?;
        rt.synchronize()?;
        s.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(s)
}

fn run() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("TensorOps absent from the metallib; these lanes are TensorOps-only".into());
    }
    let warmup = env_usize("BENCH_WARMUP", 10, 0)?;
    let iters = env_usize("BENCH_ITERS", 50, 1)?;

    let (gm, gn, gk) = GENERAL;
    let (sm, sn, sk) = SPLITK;
    let (bm, bn, bk, bb) = BATCHED;

    // NN operands, plus their bf16 and f16 casts.
    let a = f32_tensor(&rt, &[gm, gk], 1)?;
    let b = f32_tensor(&rt, &[gk, gn], 2)?;
    let c = rt.alloc_tensor_f32(&[gm, gn])?;
    let a_bf = cast_f32_to_bf16(&a)?;
    let b_bf = cast_f32_to_bf16(&b)?;
    let a_h = cast_f32_to_f16(&a)?;
    let b_h = cast_f32_to_f16(&b)?;
    let bias = f32_tensor(&rt, &[gn], 9)?;
    // `nn_coop_kernel` switches tile geometry on N alone, at 512. A single wide
    // shape reaches only the 128x64 tile, leaving every `_64x64_sg4` variant
    // undispatched -- which is how the f16 narrow tile stayed uncovered.
    let nn_narrow: usize = 256;
    let b_narrow = f32_tensor(&rt, &[gk, nn_narrow], 10)?;
    let c_narrow = rt.alloc_tensor_f32(&[gm, nn_narrow])?;
    let b_narrow_h = cast_f32_to_f16(&b_narrow)?;

    // TN wants A as [K, M]; NT wants B as [N, K].
    let a_tn = f32_tensor(&rt, &[gk, gm], 3)?;
    let b_nt = f32_tensor(&rt, &[gn, gk], 4)?;
    // Split-K shapes, separately sized.
    let a_tn_s = f32_tensor(&rt, &[sk, sm], 5)?;
    let b_tn_s = f32_tensor(&rt, &[sk, sn], 6)?;
    let c_s = rt.alloc_tensor_f32(&[sm, sn])?;
    // Batched.
    let a_b = f32_tensor(&rt, &[bb * bm, bk], 7)?;
    let b_b = f32_tensor(&rt, &[bb * bk, bn], 8)?;
    let c_b = rt.alloc_tensor_f32(&[bb * bm, bn])?;
    let a_b_bf = cast_f32_to_bf16(&a_b)?;
    let b_b_bf = cast_f32_to_bf16(&b_b)?;
    let a_b_h = cast_f32_to_f16(&a_b)?;
    let b_b_h = cast_f32_to_f16(&b_b)?;

    let spec = BatchedGemm {
        m: bm,
        n: bn,
        k: bk,
        batch: bb,
        strides: BatchStrides {
            a: bm * bk,
            b: bk * bn,
            c: bm * bn,
        },
    };
    let epi = || Epilogue {
        alpha: 1.5,
        beta: 0.5,
        bias: Some(&bias),
        activation: Activation::GeluTanh,
    };
    let to = GemmBackend::TensorOps;

    // MLX Q4 TensorOps GEMM, one lane per instantiated group size.
    let (qm, qn, qk) = Q4_TENSOROPS;
    let q4_ops = [32usize, 64, 128]
        .map(|g| q4_mlx_operands(&rt, Q4_TENSOROPS, g))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let q4_y = rt.alloc_buffer(qm * qn * 4)?;
    let q4_lane = |idx: usize, name: &'static str, group: u32| -> Lane {
        let (packed, sb, x) = &q4_ops[idx];
        let y = &q4_y;
        let rt = &rt;
        Lane {
            name,
            m: qm,
            n: qn,
            k: qk,
            batch: 1,
            run: Box::new(move || {
                let bank = Q4MlxBank {
                    packed,
                    scales_biases: sb,
                };
                let shape = QuantShape {
                    rows: qn as u32,
                    cols: qk as u32,
                    group_size: group,
                };
                gemm_q4_mlx_tensorops(rt, bank, x, y, shape, qm as u32)
            }),
        }
    };

    let mut lanes: Vec<Lane> = vec![
        // Simdgroup backend for TN: the explicit transpose + NN fallback, which
        // is the only path that dispatches `transpose2d_f32`.
        Lane {
            name: "tn_f32_simdgroup",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_tn_f32(&a_tn, &b, &c, GemmBackend::Simdgroup)),
        },
        Lane {
            name: "tn_f32",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_tn_f32(&a_tn, &b, &c, to)),
        },
        Lane {
            name: "tn_bf16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_tn_train(&a_tn, &b, &c, to)),
        },
        Lane {
            name: "nt_f32",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_nt_f32(&a, &b_nt, &c, to)),
        },
        Lane {
            name: "nt_bf16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_nt_train(&a, &b_nt, &c, to)),
        },
        Lane {
            name: "tn_accum_f32",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_tn_accum_train(&a_tn, &b, &c, to)),
        },
        Lane {
            name: "nt_accum_f32",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_nt_accum_train(&a, &b_nt, &c, to)),
        },
        Lane {
            name: "tn_accum_bf16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_tn_accum_train(&a_tn, &b, &c, to)),
        },
        Lane {
            name: "nt_accum_bf16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_nt_accum_train(&a, &b_nt, &c, to)),
        },
        Lane {
            name: "tn_splitk_f32",
            m: sm,
            n: sn,
            k: sk,
            batch: 1,
            run: Box::new(|| gemm_tn_f32(&a_tn_s, &b_tn_s, &c_s, to)),
        },
        // `C +=` at a split-K shape is the only caller of the sequential exact-f32
        // TN split-K (`matmul2d_tensorops_tn_splitk_f32`).
        Lane {
            name: "tn_accum_splitk_f32",
            m: sm,
            n: sn,
            k: sk,
            batch: 1,
            run: Box::new(|| gemm_tn_accum_train(&a_tn_s, &b_tn_s, &c_s, to)),
        },
        Lane {
            name: "tn_splitk_bf16",
            m: sm,
            n: sn,
            k: sk,
            batch: 1,
            run: Box::new(|| gemm_tn_train(&a_tn_s, &b_tn_s, &c_s, to)),
        },
        Lane {
            name: "nn_f16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm(&a_h, &b_h, &c, to)),
        },
        Lane {
            name: "nn_f16_narrow",
            m: gm,
            n: nn_narrow,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm(&a_h, &b_narrow_h, &c_narrow, to)),
        },
        Lane {
            name: "batched_bf16_accum",
            m: bm,
            n: bn,
            k: bk,
            batch: bb,
            run: Box::new(|| gemm_batched(&a_b_bf, &b_b_bf, &c_b, to, spec)),
        },
        Lane {
            name: "batched_bf16",
            m: bm,
            n: bn,
            k: bk,
            batch: bb,
            run: Box::new(|| gemm_batched(&a_b_bf, &b_b_bf, &c_b, to, spec)),
        },
        Lane {
            name: "batched_f16",
            m: bm,
            n: bn,
            k: bk,
            batch: bb,
            run: Box::new(|| gemm_batched(&a_b_h, &b_b_h, &c_b, to, spec)),
        },
        Lane {
            name: "batched_f32r",
            m: bm,
            n: bn,
            k: bk,
            batch: bb,
            run: Box::new(|| gemm_batched(&a_b, &b_b, &c_b, to, spec)),
        },
        Lane {
            name: "epi_bf16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_epilogue(&a_bf, &b_bf, &c, to, epi())),
        },
        Lane {
            name: "epi_f16",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_epilogue(&a_h, &b_h, &c, to, epi())),
        },
        Lane {
            name: "epi_f32r",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_epilogue(&a, &b, &c, to, epi())),
        },
        // Relaxed precision is off for this name: the exact-f32 epilogue.
        Lane {
            name: "epi_f32",
            m: gm,
            n: gn,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_epilogue(&a, &b, &c, to, epi())),
        },
        // N <= 512 takes the 64x64 exact-f32 epilogue tile.
        Lane {
            name: "epi_f32_narrow",
            m: gm,
            n: nn_narrow,
            k: gk,
            batch: 1,
            run: Box::new(|| gemm_epilogue(&a, &b_narrow, &c_narrow, to, epi())),
        },
    ];
    lanes.push(q4_lane(0, "q4_mlx_tensorops_g32", 32));
    lanes.push(q4_lane(1, "q4_mlx_tensorops_g64", 64));
    lanes.push(q4_lane(2, "q4_mlx_tensorops_g128", 128));

    // The lanes with no absolute-reference test at their shape (module docs:
    // `nn_f16` at N = 1024, `epi_f16`, `epi_bf16`) are checked here once, on
    // fresh outputs, in the precision mode their lane runs in.
    rt.set_relaxed_precision(false);
    {
        let (a_h, b_h) = (
            fill_dist(gm * gk, 1, Dist::Uniform),
            fill_dist(gk * gn, 2, Dist::Uniform),
        );
        let b_narrow_host = fill_dist(gk * nn_narrow, 10, Dist::Uniform);
        let (bias_h, c_prev) = (fill_dist(gn, 9, Dist::Uniform), fill_dist(gm * gn, 12, Dist::Uniform));
        let cast = |narrow: Narrow, data: &[f32], r: usize, c: usize| -> Result<Tensor, String> {
            let t = rt.alloc_tensor_f32(&[r, c])?;
            t.buffer.write_f32(data);
            match narrow {
                Narrow::F16 => cast_f32_to_f16(&t),
                Narrow::Bf16 => cast_f32_to_bf16(&t),
            }
        };
        let read = |t: &Tensor| -> Result<Vec<f32>, String> {
            rt.synchronize()?;
            Ok(t.buffer.read_f32())
        };

        rt.set_precision(PrecisionMode::F32);
        let f16 = |data: &[f32], r: usize, c: usize| cast(Narrow::F16, data, r, c);
        let c_nn = rt.alloc_tensor_f32(&[gm, gn])?;
        gemm(&f16(&a_h, gm, gk)?, &f16(&b_h, gk, gn)?, &c_nn, to)?;
        check_narrow_nn("nn_f16", Narrow::F16, (&a_h, &b_h), &read(&c_nn)?, (gm, gn, gk), None)?;

        let c_narrow_chk = rt.alloc_tensor_f32(&[gm, nn_narrow])?;
        gemm(
            &f16(&a_h, gm, gk)?,
            &f16(&b_narrow_host, gk, nn_narrow)?,
            &c_narrow_chk,
            to,
        )?;
        check_narrow_nn(
            "nn_f16_narrow",
            Narrow::F16,
            (&a_h, &b_narrow_host),
            &read(&c_narrow_chk)?,
            (gm, nn_narrow, gk),
            None,
        )?;

        let e = epi();
        let fused = NarrowEpilogue {
            alpha: f64::from(e.alpha),
            beta: f64::from(e.beta),
            c_prev: &c_prev,
            bias: &bias_h,
        };
        for (lane, narrow, mode) in [
            ("epi_f16", Narrow::F16, PrecisionMode::F32),
            ("epi_bf16", Narrow::Bf16, PrecisionMode::Bf16),
        ] {
            rt.set_precision(mode);
            let c_epi = f32_tensor(&rt, &[gm, gn], 12)?;
            gemm_epilogue(
                &cast(narrow, &a_h, gm, gk)?,
                &cast(narrow, &b_h, gk, gn)?,
                &c_epi,
                to,
                epi(),
            )?;
            check_narrow_nn(lane, narrow, (&a_h, &b_h), &read(&c_epi)?, (gm, gn, gk), Some(&fused))?;
        }
        rt.set_precision(PrecisionMode::F32);
    }

    let mut rows: Vec<String> = Vec::new();
    for l in &lanes {
        // The relaxed-precision lanes are a runtime mode, not a dtype, so the
        // name selects it — and it is cleared afterwards rather than latched.
        rt.set_relaxed_precision(l.name.ends_with("f32r"));
        // The `_train` entry points consult the runtime precision mode, so a
        // bf16 lane called under F32 silently measures the f32 kernel and
        // reports it under a bf16 name.
        rt.set_precision(if l.name.contains("bf16") {
            PrecisionMode::Bf16
        } else {
            PrecisionMode::F32
        });
        let samples = time_lane(&rt, l, warmup, iters)?;
        let med = median(samples.clone())?;
        if med <= 0.0 {
            rt.set_relaxed_precision(false);
            rt.set_precision(PrecisionMode::F32);
            return Err(format!("{}: median {med} ms is not positive", l.name));
        }
        let flop = 2.0 * l.m as f64 * l.n as f64 * l.k as f64 * l.batch as f64;
        let gflops = flop / (med * 1e6);
        eprintln!(
            "{:<16} M={} N={} K={} batch={}  {med:8.3} ms  {gflops:9.1} GFLOP/s",
            l.name, l.m, l.n, l.k, l.batch
        );
        rows.push(format!(
            r#"{{"lane":"{}","runtime":"tessl","m":{},"n":{},"k":{},"batch":{},"median_ms":{med:.6},"gflops":{gflops:.3}}}"#,
            l.name, l.m, l.n, l.k, l.batch
        ));
    }
    rt.set_relaxed_precision(false);
    rt.set_precision(PrecisionMode::F32);
    println!("[{}]", rows.join(","));
    Ok(())
}

fn time_calls(
    rt: &Arc<GpuRuntime>,
    warmup: usize,
    repeat: usize,
    mut once: impl FnMut() -> Result<(), String>,
) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        once()?;
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        once()?;
        rt.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if !ms.is_finite() || ms <= 0.0 {
            return Err(format!(
                "cooperative quant GEMM sample {ms} is not a positive finite time"
            ));
        }
        samples.push(ms);
    }
    Ok(samples)
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn tool_capture(program: &str, args: &[&str]) -> (bool, String) {
    match std::process::Command::new(program).args(args).output() {
        Ok(out) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&out.stdout));
            if !out.stderr.is_empty() {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            (out.status.success(), text)
        }
        Err(e) => (false, format!("{program}: {e}")),
    }
}

fn quant_row(kernel: &str, m: usize, group: u32, repeat: usize, samples: &[f64]) -> Result<String, String> {
    let med = median(samples.to_vec())?;
    let times = samples.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",");
    Ok(format!(
        r#"{{"kernel":"{kernel}","runtime":"tessl","device_rows":true,"m":{m},"n":4096,"k":4096,"group_size":{group},"repeat":{repeat},"times_ms":[{times}],"median_ms":{med:.6}}}"#
    ))
}

/// `(M, 4096, 4096)` for the eight M in the quant-coop grid.
///
/// `cargo run --release --bin bench_gemm_variants -- --quant-coop`
///
/// Times `gemm_q4_mlx_tensorops` (group 64) and `gemm_q8_tensorops` (group 32)
/// on device. MLX `nax`, torch MPS, and a fan pin are attempted in the same
/// process; a tool that cannot run contributes its own error text.
fn quant_coop_grid() -> Result<(), String> {
    const N: usize = 4096;
    const K: usize = 4096;
    const MAX_M: usize = 2048;
    const MS: [usize; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];
    const Q4_GROUP: u32 = 64;
    const Q8_GROUP: u32 = 32;
    const Q4_KERNEL: &str = "gemm_q4_mlx_tensorops_g64";
    const Q8_KERNEL: &str = "gemm_q8_tensorops_g32";

    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("TensorOps is unavailable; the cooperative quant GEMM has no host path".into());
    }
    let device = rt.device_name();
    let warmup = env_usize("BENCH_WARMUP", 2, 0)?;
    let repeat = env_usize("BENCH_ITERS", 5, 1)?;

    let q4_packed = rt.alloc_buffer(N * K / 2)?;
    q4_packed.try_write_bytes(&vec![0x11u8; N * K / 2])?;
    let q4_pairs = N * (K / Q4_GROUP as usize);
    let q4_sb = rt.alloc_buffer(q4_pairs * 4)?;
    q4_sb.try_write_bf16_bits(&vec![0x3f80u16; q4_pairs * 2])?;
    let q4_x = rt.alloc_buffer(MAX_M * K * 2)?;
    q4_x.try_write_bf16_bits(&vec![0x3f00u16; MAX_M * K])?;
    let y = rt.alloc_buffer(MAX_M * N * 4)?;
    y.try_write_f32(&vec![0.0f32; MAX_M * N])?;

    let q8_packed = rt.alloc_buffer(N * K)?;
    q8_packed.try_write_bytes(&vec![1u8; N * K])?;
    let q8_groups = K / Q8_GROUP as usize;
    let q8_scale = rt.alloc_buffer(N * q8_groups * 4)?;
    q8_scale.try_write_f32(&vec![0.01f32; N * q8_groups])?;
    let q8_zero = rt.alloc_buffer(N * q8_groups * 4)?;
    q8_zero.try_write_f32(&vec![0.0f32; N * q8_groups])?;
    let q8_x = rt.alloc_buffer(MAX_M * K * 4)?;
    q8_x.try_write_f32(&vec![0.25f32; MAX_M * K])?;

    let mut rows = Vec::with_capacity(MS.len() * 2);
    for &m in &MS {
        eprintln!("timing {Q4_KERNEL} M={m}");
        let samples = time_calls(&rt, warmup, repeat, || {
            let bank = Q4MlxBank {
                packed: &q4_packed,
                scales_biases: &q4_sb,
            };
            let shape = QuantShape {
                rows: N as u32,
                cols: K as u32,
                group_size: Q4_GROUP,
            };
            gemm_q4_mlx_tensorops(&rt, bank, &q4_x, &y, shape, m as u32)
        })?;
        rows.push(quant_row(Q4_KERNEL, m, Q4_GROUP, repeat, &samples)?);
        eprintln!("timing {Q8_KERNEL} M={m}");
        let samples = time_calls(&rt, warmup, repeat, || {
            gemm_q8_tensorops(
                &rt, &q8_packed, &q8_scale, &q8_zero, &q8_x, &y, N as u32, K as u32, Q8_GROUP, m as u32,
            )
        })?;
        rows.push(quant_row(Q8_KERNEL, m, Q8_GROUP, repeat, &samples)?);
    }

    let (nax_ok, nax_text) = tool_capture("python3", &["-c", NAX_ATTEMPT]);
    let (mps_ok, mps_text) = tool_capture("python3", &["-c", MPS_ATTEMPT]);
    let (fan_ok, fan_text) = tool_capture("sh", &["-c", FAN_ATTEMPT]);
    let fan_pinned = fan_ok && fan_text.contains("FAN_PIN_EXIT:0");

    println!(
        r#"{{"device":{},"kernels":["{Q4_KERNEL}","{Q8_KERNEL}"],"warmup":{warmup},"repeat":{repeat},"shapes":"M x 4096 x 4096","m":[1,16,64,128,256,512,1024,2048],"tessl":[{}],"nax":{{"ok":{nax_ok},"output":{}}},"mps":{{"ok":{mps_ok},"output":{}}},"fan_pin":{{"ok":{fan_pinned},"output":{}}}}}"#,
        json_string(&device),
        rows.join(","),
        json_string(&nax_text),
        json_string(&mps_text),
        json_string(&fan_text),
    );
    Ok(())
}

/// Q4 weights at M = 2..=8: the small-batch kernel `gemm_q4_mlx` (the route
/// `Q4Weight::matmul` took for these M until the Nov 5 work) against
/// `gemm_q4_mlx_tensorops` (the route it takes now), at the Qwen3.5-2B
/// projection shapes and 4096x4096, group 64.
///
/// `cargo run --release --bin bench_gemm_variants -- --q4-small-m`
///
/// Method, fixed before any run: per (shape, M), first a parity check (the two
/// arms' outputs agree within `1e-3` of the largest magnitude, f32 accumulation
/// in different orders over bf16 inputs) and one dispatch per call for each
/// arm; then `BENCH_ROUNDS` paired rounds, each timing `BENCH_CALLS`
/// back-to-back calls of both arms (ms per call), alternating which arm goes
/// first. Reported per (shape, M): each arm's median, the new/old median ratio,
/// the share of rounds where the new arm was slower, and each arm's own spread
/// `(p90 - p10) / median` as its noise floor.
///
/// Decision rule (stated before the run): the old route wins at (shape, M) when
/// the new/old ratio exceeds `1 + max(0.05, 2 * spread)` (spread = the larger
/// of the two arms') and the new arm was slower in at least 80% of rounds. A
/// win on any 2B shape at any M is a routing regression worth a card; a ratio
/// inside the band is "no difference measured"; a ratio below `1 - band` is a
/// win for the new route.
fn q4_small_m() -> Result<(), String> {
    const GROUP: u32 = 64;
    const SHAPES: [(&str, usize, usize); 4] = [
        ("2b_attn_2048x2048", 2048, 2048),
        ("2b_mlp_up_6144x2048", 6144, 2048),
        ("2b_mlp_down_2048x6144", 2048, 6144),
        ("sq_4096x4096", 4096, 4096),
    ];
    const MS: std::ops::RangeInclusive<usize> = 2..=8;

    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("TensorOps is unavailable; the new route has no device path".into());
    }
    let device = rt.device_name();
    let rounds = env_usize("BENCH_ROUNDS", 15, 3)?;
    let calls = env_usize("BENCH_CALLS", 20, 1)?;
    let warmup = env_usize("BENCH_WARMUP", 3, 0)?;

    // A small LCG so both arms read the same varied data.
    let mut seed = 0x9e37_79b9_u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    let mut rows_out: Vec<String> = Vec::new();
    for (label, rows, cols) in SHAPES {
        let packed = rt.alloc_buffer(rows * cols / 2)?;
        packed.try_write_bytes(&(0..rows * cols / 2).map(|_| (next() >> 24) as u8).collect::<Vec<_>>())?;
        let pairs = rows * (cols / GROUP as usize);
        let sb: Vec<f32> = (0..pairs * 2)
            .map(|i| {
                let r = (next() >> 8) as f32 / (1u32 << 24) as f32;
                if i.is_multiple_of(2) {
                    0.01 + 0.02 * r
                } else {
                    -0.05 * r
                }
            })
            .collect();
        let sb_buf = rt.alloc_buffer(pairs * 2 * 2)?;
        sb_buf.try_write_bf16_bits(&f32_slice_to_bf16(&sb))?;
        let x: Vec<f32> = (0..8 * cols)
            .map(|_| ((next() >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0)
            .collect();
        let x_buf = rt.alloc_buffer(8 * cols * 2)?;
        x_buf.try_write_bf16_bits(&f32_slice_to_bf16(&x))?;
        let y_old = rt.alloc_buffer(8 * rows * 4)?;
        let y_new = rt.alloc_buffer(8 * rows * 4)?;
        let shape = QuantShape {
            rows: rows as u32,
            cols: cols as u32,
            group_size: GROUP,
        };
        let bank = || Q4MlxBank {
            packed: &packed,
            scales_biases: &sb_buf,
        };
        for m in MS {
            let old = || {
                gemm_q4_mlx(
                    &rt,
                    bank(),
                    &x_buf,
                    &y_old,
                    shape,
                    m as u32,
                    Q4MlxLayout::RowMajor,
                    None,
                )
            };
            let new = || gemm_q4_mlx_tensorops(&rt, bank(), &x_buf, &y_new, shape, m as u32);

            // Correctness and path before speed.
            rt.synchronize()?;
            rt.take_dispatch_count();
            old()?;
            let old_dispatches = rt.take_dispatch_count();
            new()?;
            let new_dispatches = rt.take_dispatch_count();
            rt.synchronize()?;
            if old_dispatches != 1 || new_dispatches != 1 {
                return Err(format!(
                    "{label} M={m}: expected one dispatch per call, old {old_dispatches}, new {new_dispatches}"
                ));
            }
            let a = y_old.try_read_f32()?;
            let b = y_new.try_read_f32()?;
            let n = m * rows;
            let scale = a[..n].iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1e-30);
            let worst = common::check_against(
                &format!("{label} M={m}"),
                a.get(..n).unwrap_or(&[]),
                b.get(..n).unwrap_or(&[]),
                common::UNBOUNDED,
                0.0,
            )?;
            if worst > f64::from(1e-3 * scale) {
                return Err(format!(
                    "{label} M={m}: arms disagree, max |old - new| {worst:e} against max |old| {scale:e}"
                ));
            }

            let per_call = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
                let t0 = Instant::now();
                for _ in 0..calls {
                    f()?;
                }
                rt.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / calls as f64;
                if !ms.is_finite() || ms <= 0.0 {
                    return Err(format!("{label} M={m}: sample {ms} is not a positive finite time"));
                }
                Ok(ms)
            };
            for _ in 0..warmup {
                per_call(&old)?;
                per_call(&new)?;
            }
            let (mut ts_old, mut ts_new) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
            for r in 0..rounds {
                if r.is_multiple_of(2) {
                    ts_old.push(per_call(&old)?);
                    ts_new.push(per_call(&new)?);
                } else {
                    ts_new.push(per_call(&new)?);
                    ts_old.push(per_call(&old)?);
                }
            }
            let spread = |v: &[f64]| -> Result<f64, String> {
                let mut s = v.to_vec();
                s.sort_by(f64::total_cmp);
                let at = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
                Ok((at(0.9) - at(0.1)) / median(v.to_vec())?)
            };
            let (med_old, med_new) = (median(ts_old.clone())?, median(ts_new.clone())?);
            let ratio = med_new / med_old;
            let slower = ts_old.iter().zip(&ts_new).filter(|(o, n)| n > o).count() as f64 / rounds as f64;
            let noise = spread(&ts_old)?.max(spread(&ts_new)?);
            let band = 0.05f64.max(2.0 * noise);
            let verdict = if ratio > 1.0 + band && slower >= 0.8 {
                "old_route_wins"
            } else if ratio < 1.0 - band && slower <= 0.2 {
                "new_route_wins"
            } else {
                "no_difference_measured"
            };
            eprintln!(
                "{label:<24} M={m}  old {med_old:.4} ms  new {med_new:.4} ms  new/old {ratio:.3}  band {band:.3}  new slower {:.0}%  {verdict}",
                slower * 100.0
            );
            let list = |v: &[f64]| v.iter().map(|t| format!("{t:.6}")).collect::<Vec<_>>().join(",");
            rows_out.push(format!(
                r#"{{"shape":"{label}","rows":{rows},"cols":{cols},"m":{m},"old_ms":[{}],"new_ms":[{}],"old_median_ms":{med_old:.6},"new_median_ms":{med_new:.6},"new_over_old":{ratio:.4},"new_slower_share":{slower:.3},"band":{band:.4},"max_abs_diff":{worst:e},"verdict":"{verdict}"}}"#,
                list(&ts_old),
                list(&ts_new)
            ));
        }
    }
    println!(
        r#"{{"device":{},"old":"gemm_q4_mlx (RowMajor)","new":"gemm_q4_mlx_tensorops_g64","group_size":{GROUP},"rounds":{rounds},"calls_per_sample":{calls},"warmup":{warmup},"rule":"old wins if new/old > 1 + max(0.05, 2*spread) and new slower in >= 80% of rounds","rows":[{}]}}"#,
        json_string(&device),
        rows_out.join(",")
    );
    Ok(())
}

const NAX_ATTEMPT: &str = r#"
import os, sys, time
try:
    import mlx.core as mx
except Exception as e:
    sys.stderr.write("%s: %s\n" % (type(e).__name__, e))
    sys.exit(1)
nax = getattr(mx, "nax", None)
if nax is None:
    sys.stderr.write("AttributeError: module 'mlx.core' has no attribute 'nax'\n")
    sys.exit(1)
matmul = getattr(nax, "matmul", None)
if matmul is None:
    sys.stderr.write("AttributeError: mlx.core.nax has no matmul; names: %s\n" % ", ".join(dir(nax)))
    sys.exit(1)
repeat = int(os.environ.get("BENCH_ITERS", "5"))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
ms = [1, 16, 64, 128, 256, 512, 1024, 2048]
out = []
for m in ms:
    a = mx.random.uniform(shape=(m, 4096))
    b = mx.random.uniform(shape=(4096, 4096))
    mx.eval(a, b)
    for _ in range(warmup):
        mx.eval(matmul(a, b))
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        mx.eval(matmul(a, b))
        samples.append((time.perf_counter() - t0) * 1000.0)
    out.append({"m": m, "times_ms": samples})
print(out)
"#;

const MPS_ATTEMPT: &str = r#"
import os, sys, time
try:
    import torch
except Exception as e:
    sys.stderr.write("%s: %s\n" % (type(e).__name__, e))
    sys.exit(1)
if not torch.backends.mps.is_available():
    sys.stderr.write("RuntimeError: torch.backends.mps.is_available() is False\n")
    sys.exit(1)
repeat = int(os.environ.get("BENCH_ITERS", "5"))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
ms = [1, 16, 64, 128, 256, 512, 1024, 2048]
out = []
for m in ms:
    a = torch.randn(m, 4096, device="mps")
    b = torch.randn(4096, 4096, device="mps")
    torch.mps.synchronize()
    for _ in range(warmup):
        torch.matmul(a, b)
        torch.mps.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    out.append({"m": m, "times_ms": samples})
print(out)
"#;

const FAN_ATTEMPT: &str = r#"
sudo -n powermetrics -n 1 -i 100 --samplers smc
echo FAN_PIN_EXIT:$?
command -v smc || echo "smc: command not found"
command -v smcif || echo "smcif: command not found"
exit 0
"#;

fn main() -> Result<(), String> {
    let outcome = if std::env::args().any(|a| a == "--quant-coop") {
        quant_coop_grid()
    } else if std::env::args().any(|a| a == "--q4-small-m") {
        q4_small_m()
    } else {
        run()
    };
    common::emit_kernel_trace();
    outcome
}
