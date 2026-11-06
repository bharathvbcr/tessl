//! Time the shipped int8 TensorOps path on `(M, 4096, 4096)`.
//!
//! `bench_gemm_sweep` times f32, tf32 and bf16. This binary times
//! `nn::gemm_i8` and the quantize-then-dequant pipeline
//! (`nn::quantize_i8_rows` plus per-row scales). MLX and MPS are not linked
//! here. Protocol matches the sweep: warmup, then `BENCH_ITERS` samples,
//! `synchronize` every iteration, median. Defaults are warmup 2 and 5 iters.

mod common;

use std::time::Instant;

use common::{env_usize, jarr, jnum, jstr, median};
use tessl::nn::{self, I8AScale, I8BLayout, I8BScale, I8Gemm, I8Out};
use tessl::runtime::GpuRuntime;

const MS: [u32; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];
const N: u32 = 4096;
const K: u32 = 4096;

fn main() {
    if let Err(err) = run() {
        eprintln!("bench_gemm_i8: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let warmup = env_usize("BENCH_WARMUP", 2, 1)?;
    let iters = env_usize("BENCH_ITERS", 5, 1)?;
    let rt = GpuRuntime::new()?;
    let device = rt.device_name();
    let m_max = 2048usize;
    let a = rt.alloc_buffer(m_max * K as usize)?;
    let b = rt.alloc_buffer((K as usize) * (N as usize))?;
    let c = rt.alloc_buffer(m_max * (N as usize) * 4)?;
    let x = rt.alloc_buffer(m_max * (K as usize) * 4)?;
    let q = rt.alloc_buffer(m_max * K as usize)?;
    let scales = rt.alloc_buffer(m_max * 4)?;

    let mut a_bytes = vec![0u8; m_max * K as usize];
    for (i, byte) in a_bytes.iter_mut().enumerate() {
        *byte = i.wrapping_mul(17).wrapping_add(3) as u8;
    }
    a.try_write_bytes(&a_bytes)?;
    let mut b_bytes = vec![0u8; (K as usize) * (N as usize)];
    for (i, byte) in b_bytes.iter_mut().enumerate() {
        *byte = i.wrapping_mul(31).wrapping_add(9) as u8;
    }
    b.try_write_bytes(&b_bytes)?;
    let mut xf = vec![0f32; m_max * K as usize];
    for (i, value) in xf.iter_mut().enumerate() {
        *value = ((i % 251) as f32 - 125.0) * 0.01;
    }
    x.try_write_f32(&xf)?;

    call_gemm(&rt, &a, &b, &c, 1)?;
    rt.synchronize()?;
    nn::quantize_i8_rows(&rt, &x, &q, &scales, 1, K)?;
    call_rows(&rt, &q, &b, &c, &scales, 1)?;
    rt.synchronize()?;

    let mut gemm_rows = Vec::new();
    let mut pipe_rows = Vec::new();
    for &m in &MS {
        let samples = time(warmup, iters, || {
            call_gemm(&rt, &a, &b, &c, m)?;
            rt.synchronize()
        })?;
        let line = row("tessl-i8-gemm", m, &samples)?;
        eprintln!("{line}");
        gemm_rows.push(json_row("tessl-i8-gemm", m, &samples)?);

        let samples = time(warmup, iters, || {
            nn::quantize_i8_rows(&rt, &x, &q, &scales, m, K)?;
            call_rows(&rt, &q, &b, &c, &scales, m)?;
            rt.synchronize()
        })?;
        let line = row("tessl-i8-pipeline", m, &samples)?;
        eprintln!("{line}");
        pipe_rows.push(json_row("tessl-i8-pipeline", m, &samples)?);
    }

    let head = c.read_f32();
    let first = head.first().copied().unwrap_or(f32::NAN);
    if !first.is_finite() {
        return Err(format!("tessl-i8 produced a non-finite output ({first})"));
    }

    println!("{{");
    println!("  \"entry\": \"bench_gemm_i8\",");
    println!("  \"device\": {},", jstr(&device));
    println!("  \"n\": {N},");
    println!("  \"k\": {K},");
    println!("  \"warmup\": {warmup},");
    println!("  \"repeat_count\": {iters},");
    println!("  \"synchronize_every_iter\": true,");
    println!("  \"runtime\": \"GpuRuntime::new (counter timestamps on)\",");
    println!("  \"arms\": [");
    println!("    {{");
    println!("      \"name\": \"tessl-i8-gemm\",");
    println!("      \"what\": \"nn::gemm_i8 scalar a_scale 1, no b scale, f32 store, B [K,N]\",");
    println!("      \"rows\": [");
    println!("        {}", gemm_rows.join(",\n        "));
    println!("      ]");
    println!("    }},");
    println!("    {{");
    println!("      \"name\": \"tessl-i8-pipeline\",");
    println!("      \"what\": \"nn::quantize_i8_rows then nn::gemm_i8 with per-row a_scale, f32 store\",");
    println!("      \"rows\": [");
    println!("        {}", pipe_rows.join(",\n        "));
    println!("      ]");
    println!("    }}");
    println!("  ]");
    println!("}}");
    Ok(())
}

fn call_gemm(
    rt: &std::sync::Arc<GpuRuntime>,
    a: &tessl::tensor::GpuBuffer,
    b: &tessl::tensor::GpuBuffer,
    c: &tessl::tensor::GpuBuffer,
    m: u32,
) -> Result<(), String> {
    nn::gemm_i8(
        rt,
        I8Gemm {
            a,
            b,
            c,
            m,
            n: N,
            k: K,
            a_scale: I8AScale::Scalar(1.0),
            b_scale: I8BScale::None,
            bias: None,
            b_layout: I8BLayout::RowMajorKN,
            out: I8Out::F32,
        },
    )
}

fn call_rows(
    rt: &std::sync::Arc<GpuRuntime>,
    a: &tessl::tensor::GpuBuffer,
    b: &tessl::tensor::GpuBuffer,
    c: &tessl::tensor::GpuBuffer,
    scales: &tessl::tensor::GpuBuffer,
    m: u32,
) -> Result<(), String> {
    nn::gemm_i8(
        rt,
        I8Gemm {
            a,
            b,
            c,
            m,
            n: N,
            k: K,
            a_scale: I8AScale::Rows(scales),
            b_scale: I8BScale::None,
            bias: None,
            b_layout: I8BLayout::RowMajorKN,
            out: I8Out::F32,
        },
    )
}

fn time(warmup: usize, iters: usize, mut body: impl FnMut() -> Result<(), String>) -> Result<Vec<f64>, String> {
    for _ in 0..warmup {
        body()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        body()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(samples)
}

fn row(name: &str, m: u32, samples: &[f64]) -> Result<String, String> {
    let med = median(samples.to_vec())?;
    let gflops = (2.0 * f64::from(m) * f64::from(N) * f64::from(K)) / (med * 1.0e6);
    Ok(format!(
        "{name} M={m} N={N} K={K}  {med:.3} ms  {gflops:.1} GFLOP/s  samples {}",
        jarr(samples)
    ))
}

fn json_row(name: &str, m: u32, samples: &[f64]) -> Result<String, String> {
    let med = median(samples.to_vec())?;
    let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
    if !best.is_finite() {
        return Err(format!("{name} M={m}: non-finite sample"));
    }
    let gflops = (2.0 * f64::from(m) * f64::from(N) * f64::from(K)) / (med * 1.0e6);
    Ok(format!(
        "{{\"arm\": {}, \"m\": {m}, \"n\": {N}, \"k\": {K}, \"median_ms\": {}, \"min_ms\": {}, \"gflops\": {}, \"samples_ms\": {}}}",
        jstr(name),
        jnum(med),
        jnum(best),
        jnum(gflops),
        jarr(samples)
    ))
}
