//! Where Qwen3.5-2B training's device memory goes.
//!
//! ```text
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin probe_train_memory
//! QWEN35_2B_SAFETENSORS=... cargo run --release --bin probe_train_memory -- --steps=128,2048,8192
//! ```
//!
//! Prints `MTLDevice::currentAllocatedSize`
//! ([`GpuRuntime::current_allocated_bytes`]) and the process's `ps` RSS after
//! each resident table appears: the f32 2B load, a gradient bank
//! ([`Qwen35Grads::zeros_like`]) and AdamW's two moments ([`AdamW::new`]),
//! each against its logical bytes. Then the gradient read-back that
//! ojas-qwen35 staged on the device: one exact-f32 causal step at 128 tokens
//! into the bank, a waited commit so the step's freed buffers are recycled,
//! and one whole table of fresh f32 tensors filled by
//! [`Qwen35Model::read_gradients`]. The staging is refused (dropped before
//! any GPU work reads it) if it leaves the device over its recommended
//! working set. `--no-staging` skips it.
//!
//! `--steps=T,...` then drops the moments and the staging and, per T, runs
//! one causal step into the bank from an empty freelist, printing its peak
//! allocation over what was allocated before it
//! ([`GpuRuntime::peak_allocated_bytes`]), the forward's and the backward's
//! apart, against [`Qwen35Model::train_step_bytes`]. `--bf16` runs it on bf16
//! GEMM operands, `--async` with async encode on (one command buffer per wait
//! instead of one per dispatch).

use tessl::gemm::GemmOperands;
use tessl::qwen35_adamw::AdamW;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::{GpuRuntime, Tensor};

type Res<T> = Result<T, String>;

const GB: f64 = 1e9;
const VOCAB: u32 = 248_320;

fn gb(b: u64) -> String {
    format!("{:.2} GB", b as f64 / GB)
}

/// This process's resident set as `ps` reports it, in bytes.
fn ps_rss() -> Res<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .map_err(|e| format!("ps: {e}"))?;
    let kib: u64 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .map_err(|e| format!("ps rss: {e}"))?;
    Ok(kib * 1024)
}

fn point(rt: &GpuRuntime, what: &str, prev: &mut u64, logical: Option<u64>) -> Res<()> {
    let now = rt.current_allocated_bytes();
    let delta = now as i128 - *prev as i128;
    let sign = if delta < 0 { "-" } else { "+" };
    let moved = gb(delta.unsigned_abs() as u64);
    let rss = gb(ps_rss()?);
    match logical {
        Some(l) => println!(
            "{what:<34} {:>10}  {sign}{moved:>9}  rss {rss:>9}  (logical {}, x{:.3})",
            gb(now),
            gb(l),
            delta as f64 / l as f64
        ),
        None => println!("{what:<34} {:>10}  {sign}{moved:>9}  rss {rss:>9}", gb(now)),
    }
    *prev = now;
    Ok(())
}

fn ids(t: usize) -> Vec<u32> {
    (0..t as u32).map(|i| (i * 104_729 + 17) % VOCAB).collect()
}

fn main() -> Res<()> {
    let path = std::env::var("QWEN35_2B_SAFETENSORS")
        .map_err(|_| "set QWEN35_2B_SAFETENSORS to the Qwen3.5-2B-Base .safetensors".to_string())?;
    let (mut staging_on, mut steps, mut operands) = (true, Vec::new(), GemmOperands::ExactF32);
    let mut batched = false;
    for arg in std::env::args().skip(1) {
        if arg == "--no-staging" {
            staging_on = false;
        } else if arg == "--bf16" {
            operands = GemmOperands::Bf16;
        } else if arg == "--async" {
            batched = true;
        } else if let Some(v) = arg.strip_prefix("--steps=") {
            for t in v.split(',') {
                let t: usize = t
                    .parse()
                    .map_err(|_| format!("--steps expects token counts, got {v:?}"))?;
                if t < 2 {
                    return Err("a causal step needs at least two tokens".into());
                }
                steps.push(t);
            }
        } else {
            return Err(format!(
                "expected --no-staging, --bf16, --async or --steps=T,..., got {arg:?}"
            ));
        }
    }
    let rt = GpuRuntime::new()?;
    rt.set_async_encode(batched)?;
    let ws = rt.memory_info().recommended_working_set;
    println!("device: {}, recommended working set {}", rt.device_name(), gb(ws));
    let mut prev = rt.current_allocated_bytes();
    point(&rt, "runtime", &mut prev, None)?;

    let st = SafeTensors::open(std::path::Path::new(&path))?;
    let model = Qwen35Model::load(
        &rt,
        &st,
        "model.language_model.",
        Qwen35Config::qwen35_2b()?,
        Precision::F32,
    )?;
    drop(st);
    rt.synchronize()?;
    let table = model.parameter_table()?;
    let logical: u64 = table.iter().map(|p| p.shape.iter().product::<usize>() as u64 * 4).sum();
    point(&rt, "load (f32 weights)", &mut prev, Some(logical))?;
    let bank = Qwen35Grads::zeros_like(&model)?;
    point(&rt, "Qwen35Grads::zeros_like (bank)", &mut prev, Some(logical))?;
    let adamw = AdamW::new(&model)?;
    point(&rt, "AdamW::new (two moments)", &mut prev, Some(2 * logical))?;

    // ft-dc1fa0's gradient read-back: one exact-f32 step at 128 tokens.
    let loss = model.train_step_into(&ids(128), GemmOperands::ExactF32, Supervise::Causal, &bank, false)?;
    if !loss.is_finite() {
        return Err(format!("step loss {loss} is not finite"));
    }
    point(&rt, "after a T = 128 step", &mut prev, None)?;
    rt.synchronize()?;
    point(&rt, "recycle done (waited commit)", &mut prev, None)?;
    if staging_on {
        let staging = table
            .iter()
            .map(|p| rt.alloc_tensor_f32(&p.storage_shape()))
            .collect::<Res<Vec<Tensor>>>()?;
        let with_staging = rt.current_allocated_bytes();
        point(&rt, "with staging (one f32 table)", &mut prev, Some(logical))?;
        if with_staging > ws {
            drop(staging);
            rt.synchronize()?;
            println!(
                "staging refused: {} allocated is over the working set by {}; nothing read",
                gb(with_staging),
                gb(with_staging - ws)
            );
        } else {
            model.read_gradients(&bank, &staging)?;
            point(&rt, "read_gradients done", &mut prev, None)?;
            println!("headroom {} under the working set", gb(ws - with_staging));
        }
    }
    drop(adamw);
    rt.synchronize()?;
    point(&rt, "moments and staging dropped", &mut prev, None)?;

    let cap = rt.memory_info().pool_cache_cap;
    for t in steps {
        // Empty the freelist, so the step allocates as on a fresh runtime.
        rt.set_pool_cache_cap_bytes(0);
        rt.set_pool_cache_cap_bytes(cap);
        rt.synchronize()?;
        let before = rt.current_allocated_bytes();
        let estimate = model.train_step_bytes(t as u32, operands);
        rt.reset_peak_allocated_bytes();
        let p = model.train_forward(&ids(t), operands, Supervise::Causal)?;
        let loss = p.loss();
        rt.synchronize()?;
        let forward = rt.peak_allocated_bytes();
        rt.reset_peak_allocated_bytes();
        model.train_backward_into(p, None, &bank, false)?;
        rt.synchronize()?;
        if !loss.is_finite() {
            return Err(format!("T = {t}: step loss {loss} is not finite"));
        }
        let (peak, pool) = (rt.peak_allocated_bytes().max(forward), cap as u64);
        let grew = peak - before;
        println!(
            "step T = {t:>5} {operands:?}: before {}, peak {} (+{}; forward +{}, backward +{}); estimate +{} \
             (over by {}, {} of it the freelist cap), rss {}",
            gb(before),
            gb(peak),
            gb(grew),
            gb(forward - before),
            gb(rt.peak_allocated_bytes().saturating_sub(before)),
            gb(estimate),
            gb(estimate.saturating_sub(grew)),
            gb(pool),
            gb(ps_rss()?)
        );
        if estimate < grew {
            return Err(format!(
                "T = {t}: the estimate {estimate} B is under the measured {grew} B"
            ));
        }
    }
    Ok(())
}
