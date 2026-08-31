//! GEMM shape sweep: TensorOps vs simdgroup, JSON out for cross-runtime compare.
//!
//! Companion to `bench/gemm_sweep_mlx.py` (MLX + PyTorch MPS lanes). Protocol is
//! pinned to match: same shapes, same warmup/iters, **synchronize every iteration**
//! so no lane hides dispatch cost behind pipelining, median over iters.
//!
//! `--dump-parity DIR` writes A/B/C as .npy for the numeric check in the Python lane.

use tessl::gemm::{cast_f32_to_bf16, gemm, GemmBackend};
use tessl::npy::write_npy_f32;
use tessl::runtime::GpuRuntime;
use tessl::tensor::Tensor;
use std::path::Path;
use std::time::Instant;

/// (M, N, K, label). Square ladder + the projection shapes arch_02 actually runs.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
];

/// Deterministic LCG fill in u64, mapped to [-1, 1). The Python timing lanes
/// use constant 0.5 operands instead — GEMM timing is data-independent, so the
/// lanes only share data where it matters: the parity check reads A/B from the
/// .npy files this binary dumps.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 32) as u32) as f64 / (u32::MAX as f64);
        out.push((u * 2.0 - 1.0) as f32);
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn time_backend(
    rt: &std::sync::Arc<GpuRuntime>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
    warmup: usize,
    iters: usize,
) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        gemm(a, b, c, backend)?;
        rt.synchronize()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        gemm(a, b, c, backend)?;
        rt.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(samples)
}

/// `BENCH_SHAPES="MxNxK,MxNxK,..."` overrides the built-in ladder so this lane
/// and the Python lane can be pointed at an identical diagnostic grid.
fn shapes_from_env() -> Option<Vec<(usize, usize, usize, String)>> {
    let raw = std::env::var("BENCH_SHAPES").ok()?;
    let mut out = Vec::new();
    for spec in raw.split(',').filter(|s| !s.trim().is_empty()) {
        let d: Vec<usize> = spec.trim().split('x').map(|v| v.parse().unwrap()).collect();
        assert_eq!(d.len(), 3, "BENCH_SHAPES entry must be MxNxK, got {spec}");
        out.push((d[0], d[1], d[2], format!("{}x{}x{}", d[0], d[1], d[2])));
    }
    Some(out)
}

fn main() -> Result<(), String> {
    let rt = GpuRuntime::new()?;
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);

    let args: Vec<String> = std::env::args().collect();
    let dump_dir = args.iter().position(|a| a == "--dump-parity").map(|i| args[i + 1].clone());

    let mut backends = vec![("simdgroup", GemmBackend::Simdgroup)];
    if rt.has_tensorops() {
        backends.insert(0, ("tensorops", GemmBackend::TensorOps));
    } else {
        eprintln!("warning: TensorOps absent from metallib; simdgroup lane only");
    }

    let owned = shapes_from_env();
    let shapes: Vec<(usize, usize, usize, String)> = match owned {
        Some(v) => v,
        None => SHAPES.iter().map(|&(m, n, k, l)| (m, n, k, l.to_string())).collect(),
    };

    let mut rows: Vec<String> = Vec::new();
    for (m, n, k, label) in shapes.iter().map(|(m, n, k, l)| (*m, *n, *k, l.as_str())) {
        let a = rt.alloc_tensor_f32(&[m, k])?;
        let b = rt.alloc_tensor_f32(&[k, n])?;
        let c = rt.alloc_tensor_f32(&[m, n])?;
        let a_host = fill(m * k, 1);
        let b_host = fill(k * n, 2);
        a.buffer.write_f32(&a_host);
        b.buffer.write_f32(&b_host);

        // 2*M*N*K FLOP per GEMM.
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let mut lanes: Vec<(String, GemmBackend, Tensor, Tensor)> = Vec::new();
        for &(bname, backend) in &backends {
            lanes.push((format!("{bname}-f32"), backend, a.view(&[m, k], 0), b.view(&[k, n], 0)));
        }
        if rt.has_tensorops() {
            // bf16 operands, f32 accumulate — the path `gemm_train` takes under
            // PrecisionMode::Bf16. Cast is hoisted out of the timed region.
            lanes.push((
                "tensorops-bf16".to_string(),
                GemmBackend::TensorOps,
                cast_f32_to_bf16(&a)?,
                cast_f32_to_bf16(&b)?,
            ));
            // tf32-class relaxed precision on f32 operands (opt-in --tf32 path).
            lanes.push((
                "tensorops-tf32".to_string(),
                GemmBackend::TensorOps,
                a.view(&[m, k], 0),
                b.view(&[k, n], 0),
            ));
        }

        for (bname, backend, la, lb) in &lanes {
            let (bname, backend) = (bname.as_str(), *backend);
            // The tf32-class path is opt-in at runtime, so it is invisible to a
            // sweep that only toggles dtype. Without this lane the relaxed-f32
            // kernels never appear in any cross-runtime comparison.
            rt.set_relaxed_precision(bname == "tensorops-tf32");
            let samples = time_backend(&rt, la, lb, &c, backend, warmup, iters)?;
            let med = median(samples.clone());
            let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let gflops = flop / (med * 1e6);
            eprintln!("{label:<12} {bname:<10} M={m} N={n} K={k}  {med:8.3} ms  {gflops:8.1} GFLOP/s");
            rows.push(format!(
                r#"{{"shape":"{label}","backend":"{bname}","runtime":"metal-native","m":{m},"n":{n},"k":{k},"median_ms":{med:.6},"best_ms":{best:.6},"gflops":{gflops:.3}}}"#
            ));

            if let Some(dir) = &dump_dir {
                if label == "square_1024" && bname.ends_with("-f32") {
                    let d = Path::new(dir);
                    std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                    write_npy_f32(&d.join("parity_a.npy"), &[m, k], &a_host)?;
                    write_npy_f32(&d.join("parity_b.npy"), &[k, n], &b_host)?;
                    write_npy_f32(
                        &d.join(format!("parity_c_{bname}.npy")),
                        &[m, n],
                        &c.buffer.read_f32()[..m * n],
                    )?;
                }
            }
        }
    }
    println!("[{}]", rows.join(","));
    Ok(())
}
