//! Paired 128×64 vs 64×64 plain bf16 GEMM at Qwen3.5-2B's GDN fused in-projection.
//!
//! ```text
//! cargo run --release --bin bench_gemm_coop_tile
//! ```
//!
//! One process, one pair of buffers per M. `gemm` (not the epilogue) launches
//! this shape from `Qwen35Model::gdn`: A is `[M, hidden]` bf16, B is
//! `[hidden, width]` bf16, C is `[M, width]` f32. hidden = 2048.
//! `GdnProjLayout::width` for 16/16/128 is the fused in-proj N. M = 61 does
//! not fill a 128-row tile; M = 200 does.
//!
//! ABBA, three warmup rounds, nine timed rounds. Each sample is many launches
//! in one command buffer so the submit floor is not the measurement.

use std::sync::Arc;
use std::time::Instant;

use tessl::gemm::{gemm_tiled, EpiTile, GemmBackend};
use tessl::qwen35::GdnProjLayout;
use tessl::tensor::{f32_slice_to_bf16, Tensor};
use tessl::GpuRuntime;

const K: usize = 2048;
const MS: [usize; 2] = [61, 200];
const REPS: usize = 256;
const WARMUP_ROUNDS: usize = 3;
const ROUNDS: usize = 9;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bf16(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_bf16(shape)?;
    t.buffer.write_bf16_bits(&f32_slice_to_bf16(data));
    Ok(t)
}

fn f32_tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(data);
    Ok(t)
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

fn main() -> Result<(), String> {
    let n = GdnProjLayout::new(16, 16, 128)?.width() as usize;
    if n <= 512 {
        return Err(format!("GDN in-proj N={n} does not exercise the N>512 path"));
    }
    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("bf16 GEMM needs TensorOps".into());
    }
    println!("device: {}", rt.device_name());
    println!("paired plain GEMM tiles: 128x64 (wide) vs 64x64 (narrow), bf16, f32 accumulate");
    println!("Qwen3.5-2B GDN fused in-projection: K={K} N={n}, M={MS:?}");
    println!(
        "one process, same A/B/C per M, ABBA (first tile rotates each round), \
         warmup {WARMUP_ROUNDS}, timed {ROUNDS}, {REPS} launches per sample"
    );

    rt.set_async_encode(true)?;
    for &m in &MS {
        let a = bf16(&rt, &[m, K], &fill(m * K, 0xA100 + m as u64))?;
        let b = bf16(&rt, &[K, n], &fill(K * n, 0xB200))?;
        let c_init = fill(m * n, 0xC300 + m as u64);
        let wide_c = f32_tensor(&rt, &[m, n], &c_init)?;
        let narrow_c = f32_tensor(&rt, &[m, n], &c_init)?;

        gemm_tiled(&a, &b, &wide_c, GemmBackend::TensorOps, EpiTile::Wide)?;
        gemm_tiled(&a, &b, &narrow_c, GemmBackend::TensorOps, EpiTile::Narrow)?;
        rt.synchronize()?;
        let wide_out = wide_c.buffer.read_f32();
        let narrow_out = narrow_c.buffer.read_f32();
        if let Some(i) = wide_out[..m * n].iter().position(|v| !v.is_finite()) {
            return Err(format!("M={m} wide: element {i} is non-finite"));
        }
        if let Some(i) = narrow_out[..m * n].iter().position(|v| !v.is_finite()) {
            return Err(format!("M={m} narrow: element {i} is non-finite"));
        }
        let mut worst = 0.0f32;
        let mut bits_differ = 0usize;
        for (w, nv) in wide_out[..m * n].iter().zip(narrow_out[..m * n].iter()) {
            worst = worst.max((w - nv).abs());
            if w.to_bits() != nv.to_bits() {
                bits_differ += 1;
            }
        }
        println!(
            "M={m}: max |narrow-wide| = {worst:.4e} bits_differ {bits_differ}/{}",
            m * n
        );
        if wide_out[..m * n].iter().all(|v| v.abs() < 1e-3) {
            return Err(format!("M={m}: wide output is ~0"));
        }

        let c = f32_tensor(&rt, &[m, n], &c_init)?;
        let mut wide_ms = Vec::with_capacity(ROUNDS);
        let mut narrow_ms = Vec::with_capacity(ROUNDS);
        for round in 0..(WARMUP_ROUNDS + ROUNDS) {
            let record = round >= WARMUP_ROUNDS;
            let order = if round % 2 == 0 {
                [EpiTile::Wide, EpiTile::Narrow]
            } else {
                [EpiTile::Narrow, EpiTile::Wide]
            };
            for tile in order {
                rt.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..REPS {
                    gemm_tiled(&a, &b, &c, GemmBackend::TensorOps, tile)?;
                }
                rt.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
                if record {
                    let which = if round % 2 == 0 { "wide-first" } else { "narrow-first" };
                    let name = match tile {
                        EpiTile::Wide => "wide128",
                        EpiTile::Narrow => "narrow64",
                    };
                    println!(
                        "sample M={m} round={} {which} {name} {ms:.4} ms (buffer {:.2} ms, {REPS} launches)",
                        round - WARMUP_ROUNDS,
                        ms * REPS as f64
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                    match tile {
                        EpiTile::Wide => wide_ms.push(ms),
                        EpiTile::Narrow => narrow_ms.push(ms),
                    }
                }
            }
        }
        for (name, samples) in [("wide128", &wide_ms), ("narrow64", &narrow_ms)] {
            let med = median(samples.clone());
            let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "PAIRED m={m} tile={name} n={} reps={REPS} median_ms={med:.4} min_ms={lo:.4} max_ms={hi:.4}",
                samples.len()
            );
        }
        let n_lo = narrow_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let n_hi = narrow_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w_lo = wide_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let w_hi = wide_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let overlap = n_hi >= w_lo && w_hi >= n_lo;
        let narrow_wins = n_hi < w_lo;
        println!(
            "PAIRED m={m} ratio_narrow_over_wide={:.4} ranges_overlap={overlap} narrow_range_entirely_faster={narrow_wins}",
            median(narrow_ms) / median(wide_ms)
        );
    }
    rt.set_async_encode(false)?;
    Ok(())
}
