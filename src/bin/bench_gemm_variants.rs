//! Timing lanes for the GEMM kernels the cross-runtime sweep never reaches.
//!
//! `bench_gemm_sweep` covers the NN layout, which is the only one torch and MLX
//! expose a comparable primitive for. Everything else in
//! `matmul_tensorops.metal` — TN, NT, their accumulating and split-K forms,
//! strided-batched, the fused epilogue, and the f16 operand path — had no
//! timing lane at all: 18 of the 84 kernel entry points, including the two
//! layouts the backward pass runs on.
//!
//! Correctness for these paths already lives in `tests/gemm_correctness.rs`,
//! `gemm_batched.rs`, `gemm_epilogue.rs` and `f16.rs`, all against an f64
//! reference with the per-element bound in `tests/common/mod.rs`. This binary
//! is deliberately timing and coverage only, and does not restate them.
//!
//! There is no cross-runtime lane here on purpose: `a.T @ b` in torch or MLX
//! may materialise the transpose rather than fuse it, so a ratio would compare
//! tessl's fused kernel against transpose-plus-GEMM and read as a kernel result.

mod common;

use common::{env_usize, fill_dist, median, Dist};
use std::sync::Arc;
use std::time::Instant;
use tessl::gemm::{
    cast_f32_to_bf16, cast_f32_to_f16, gemm, gemm_batched, gemm_epilogue, gemm_nt_accum_train,
    gemm_nt_f32, gemm_nt_train, gemm_tn_accum_train, gemm_tn_f32, gemm_tn_train, Activation,
    BatchStrides, BatchedGemm, Epilogue, GemmBackend,
};
use tessl::runtime::{GpuRuntime, PrecisionMode};
use tessl::tensor::Tensor;

/// (M, N, K). One general shape, plus a split-K-eligible one: `prefer_tn_splitk`
/// wants K >= 2048, M and N <= 384 and min(M, N) <= 128, so a single shape
/// cannot reach both the descriptor and the split-K TN kernels.
const GENERAL: (usize, usize, usize) = (1024, 1024, 1024);
const SPLITK: (usize, usize, usize) = (128, 256, 4096);
const BATCHED: (usize, usize, usize, usize) = (256, 256, 256, 8);

fn f32_tensor(rt: &Arc<GpuRuntime>, shape: &[usize], seed: u64) -> Result<Tensor, String> {
    let n: usize = shape.iter().product();
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(&fill_dist(n, seed, Dist::Uniform));
    Ok(t)
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

fn time_lane(
    rt: &Arc<GpuRuntime>,
    l: &Lane,
    warmup: usize,
    iters: usize,
) -> Result<Vec<f64>, String> {
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

/// Emit the kernel trace for `bench/kernel_coverage.py`.
fn emit_kernel_trace() {
    if tessl::runtime::kernel_trace_enabled() {
        eprintln!(
            "KERNEL_TRACE {}",
            tessl::runtime::traced_kernels().join(",")
        );
    }
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

    let lanes: Vec<Lane> = vec![
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
    ];

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

fn main() -> Result<(), String> {
    let outcome = run();
    emit_kernel_trace();
    outcome
}
