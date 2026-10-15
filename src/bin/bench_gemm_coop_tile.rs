//! Paired 128x64 vs 64x64 bf16 GEMM tiles at Qwen3.5-2B's two N>512 projections.
//!
//! ```text
//! cargo run --release --bin bench_gemm_coop_tile -- --family plain      # default
//! cargo run --release --bin bench_gemm_coop_tile -- --family epilogue
//! ```
//!
//! One process, one pair of buffers per M. A is `[M, K]` bf16, B is `[K, N]`
//! bf16, C is `[M, N]` f32. M = 61 does not fill a 128-row tile; M = 200 does.
//!
//! * `plain`: `gemm_tiled`, which `Qwen35Model::gdn` launches for the fused
//!   in-projection. K = hidden = 2048, N = `GdnProjLayout::width` for 16/16/128.
//! * `epilogue`: `gemm_epilogue_tiled` with the residual add (`beta = 1`),
//!   which `project_residual` dispatches. K = 8 query heads x 256, N = hidden
//!   2048. The two tiles must agree within the unfused-sequence tolerance.
//!
//! ABBA, three warmup rounds, nine timed rounds. Each sample is many launches
//! in one command buffer so the submit floor is not the measurement.

mod common;

use std::sync::Arc;
use std::time::Instant;

use tessl::gemm::{gemm_epilogue_tiled, gemm_tiled, EpiTile, Epilogue, GemmBackend};
use tessl::qwen35::GdnProjLayout;
use tessl::tensor::{f32_slice_to_bf16, Tensor};
use tessl::GpuRuntime;

const K: usize = 2048;
const MS: [usize; 2] = [61, 200];
const REPS: usize = 256;
const WARMUP_ROUNDS: usize = 3;
const ROUNDS: usize = 9;

/// Which projection's launch the tiles are compared on.
#[derive(Clone, Copy)]
enum Family {
    Plain,
    Epilogue,
}

impl Family {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "plain" => Ok(Family::Plain),
            "epilogue" => Ok(Family::Epilogue),
            other => Err(format!(
                "--family {other:?} is not a family; expected plain or epilogue"
            )),
        }
    }

    fn from_args() -> Result<Self, String> {
        let mut family = Family::Plain;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let value = match arg.strip_prefix("--family=") {
                Some(v) => v.to_string(),
                None if arg == "--family" => args.next().ok_or("--family needs plain or epilogue")?,
                None => return Err(format!("unknown argument {arg:?}; expected --family plain|epilogue")),
            };
            family = Family::parse(&value)?;
        }
        Ok(family)
    }

    /// The output width: the fused GDN in-projection's, or the hidden size.
    fn n(self) -> Result<usize, String> {
        match self {
            Family::Plain => {
                let n = GdnProjLayout::new(16, 16, 128)?.width() as usize;
                if n <= 512 {
                    return Err(format!("GDN in-proj N={n} does not exercise the N>512 path"));
                }
                Ok(n)
            }
            Family::Epilogue => Ok(2048),
        }
    }

    fn launch(self, a: &Tensor, b: &Tensor, c: &Tensor, tile: EpiTile) -> Result<(), String> {
        match self {
            Family::Plain => gemm_tiled(a, b, c, GemmBackend::TensorOps, tile),
            Family::Epilogue => {
                let residual_add = Epilogue {
                    beta: 1.0,
                    ..Epilogue::default()
                };
                gemm_epilogue_tiled(a, b, c, GemmBackend::TensorOps, residual_add, tile)
            }
        }
    }
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

fn main() -> Result<(), String> {
    let family = Family::from_args()?;
    let n = family.n()?;
    let rt = GpuRuntime::new()?;
    if !rt.has_tensorops() {
        return Err("bf16 GEMM needs TensorOps".into());
    }
    println!("device: {}", rt.device_name());
    match family {
        Family::Plain => {
            println!("paired plain GEMM tiles: 128x64 (wide) vs 64x64 (narrow), bf16, f32 accumulate");
            println!("Qwen3.5-2B GDN fused in-projection: K={K} N={n}, M={MS:?}");
        }
        Family::Epilogue => {
            println!("paired epilogue tiles: 128x64 (wide) vs 64x64 (narrow), bf16, f32 accumulate, beta=1");
            println!("Qwen3.5-2B output projection: K={K} N={n}, M={MS:?}");
        }
    }
    println!(
        "one process, same A/B/C per M, ABBA (first tile rotates each round), \
         warmup {WARMUP_ROUNDS}, timed {ROUNDS}, {REPS} launches per sample"
    );

    rt.set_async_encode(true)?;
    for &m in &MS {
        let a = bf16(&rt, &[m, K], &common::fill(m * K, 0xA100 + m as u64))?;
        let b = bf16(&rt, &[K, n], &common::fill(K * n, 0xB200))?;
        let c_init = common::fill(m * n, 0xC300 + m as u64);
        let wide_c = f32_tensor(&rt, &[m, n], &c_init)?;
        let narrow_c = f32_tensor(&rt, &[m, n], &c_init)?;

        family.launch(&a, &b, &wide_c, EpiTile::Wide)?;
        family.launch(&a, &b, &narrow_c, EpiTile::Narrow)?;
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
        if let Family::Epilogue = family {
            let tol = 2e-2 * (K as f32).sqrt();
            if worst > tol {
                return Err(format!(
                    "M={m}: tiles differ by {worst}, above the unfused-sequence tol {tol}"
                ));
            }
        }
        if wide_out[..m * n].iter().all(|v| v.abs() < 1e-3) {
            return Err(format!("M={m}: wide output is ~0"));
        }

        // Both tiles write the same C, so any epilogue traffic matches.
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
                    family.launch(&a, &b, &c, tile)?;
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
            let med = common::median(samples.clone())?;
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
            common::median(narrow_ms)? / common::median(wide_ms)?
        );
    }
    rt.set_async_encode(false)?;
    Ok(())
}
