//! What Qwen3.5 training holds at a model size with no local checkpoint, per
//! stored precision, against the device's recommended working set.
//!
//! ```text
//! cargo run --release --bin probe_storage_memory -- --config=PATH/config.json
//! cargo run --release --bin probe_storage_memory -- --config=PATH/config.json --steps=512,2048
//! cargo run --release --bin probe_storage_memory -- --config=PATH/config.json --step=512
//! ```
//!
//! The model is [`Qwen35Model::random_tower`] at the config's exact shapes:
//! allocation, not values, is what is measured. For each variant it prints
//! the resident tables' allocated bytes: the weights, a gradient bank
//! ([`Qwen35Grads::zeros_like`], in the weights' dtype) and the optimizer
//! ([`AdamW::allocated_bytes_for`]). Every variant that fits beside what is
//! already allocated is then allocated for real and
//! `MTLDevice::currentAllocatedSize` ([`GpuRuntime::current_allocated_bytes`])
//! is printed after each table; one that does not fit is reported from the
//! computed sizes and not allocated. The f32 rows are computed only: they do
//! not fit a 64 GB Mac by construction.
//!
//! `--steps=T,...` adds [`Qwen35Model::train_step_bytes`], the bound the
//! step's pre-flight gate uses, for each length.
//!
//! `--step=T` runs one training step and one AdamW step at T tokens on the
//! config's model, into a bf16 bank, with a Kahan + 8-bit optimizer resident,
//! and prints the measured peak ([`GpuRuntime::peak_allocated_bytes`]) beside
//! the step's bound and the working set.

use std::sync::Arc;

use tessl::gemm::GemmOperands;
use tessl::qwen35_adamw::{AdamW, AdamWConfig, AdamWHyper, MomentStorage, UpdateRule};
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{Qwen35Grads, Supervise};
use tessl::GpuRuntime;

type Res<T> = Result<T, String>;

const GIB: f64 = (1u64 << 30) as f64;

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / GIB)
}

/// The variants: label, weight precision, optimizer configuration.
fn variants() -> Vec<(&'static str, Precision, AdamWConfig)> {
    let c = |update, moments| AdamWConfig { update, moments };
    vec![
        ("f32 everywhere", Precision::F32, AdamWConfig::F32),
        (
            "bf16 weights+bank, f32 master, f32 moments",
            Precision::Bf16,
            c(UpdateRule::F32Master, MomentStorage::F32),
        ),
        (
            "bf16 weights+bank, f32 master, 8-bit moments",
            Precision::Bf16,
            c(UpdateRule::F32Master, MomentStorage::Block8),
        ),
        (
            "bf16 weights+bank, Kahan, bf16 moments (all bf16)",
            Precision::Bf16,
            c(UpdateRule::Bf16Kahan, MomentStorage::Bf16),
        ),
        (
            "bf16 weights+bank, Kahan, 8-bit moments",
            Precision::Bf16,
            c(UpdateRule::Bf16Kahan, MomentStorage::Block8),
        ),
        (
            "bf16 weights+bank, stochastic rounding, 8-bit moments",
            Precision::Bf16,
            c(UpdateRule::Bf16Stochastic { seed: 0 }, MomentStorage::Block8),
        ),
    ]
}

/// Allocated bytes of a resident table of `n` elements of `size` bytes.
fn hot(n: usize, size: usize) -> u64 {
    GpuRuntime::allocated_bytes_for(n * size, tessl::runtime::BufferKind::Hot)
}

/// The weights' (or a bank's) allocated bytes with matrices of `size`
/// bytes per element: every 1-D, conv and norm parameter is its own f32
/// buffer, and the embedding and packed projections are [`packed_bytes`].
fn table_bytes(model: &Qwen35Model, size: usize) -> Res<u64> {
    let vectors: u64 = model
        .parameter_table()?
        .iter()
        .filter(|p| !p.transposed && p.name != "embed_tokens.weight")
        .map(|p| hot(p.shape.iter().product(), 4))
        .sum();
    Ok(vectors + packed_bytes(model, size))
}

/// Allocated bytes of the embedding and every packed projection, at `size`
/// bytes per element.
fn packed_bytes(model: &Qwen35Model, size: usize) -> u64 {
    let c = model.config();
    let (h, i, v) = (c.hidden as usize, c.intermediate as usize, c.vocab as usize);
    let mut total = hot(v * h, size);
    for kind in &c.layers {
        let (w_in, w_out) = match kind {
            tessl::qwen35_model::LayerKind::LinearAttention => {
                (h * c.gdn.width() as usize, c.gdn.value_dim() as usize * h)
            }
            tessl::qwen35_model::LayerKind::FullAttention => (
                h * c.attn.width() as usize,
                (c.attn.q_heads() * c.attn.head_dim()) as usize * h,
            ),
        };
        total += hot(w_in, size) + hot(w_out, size) + 3 * hot(h * i, size);
    }
    total
}

fn main() -> Res<()> {
    let mut config_path = None;
    let (mut steps, mut step) = (Vec::new(), None);
    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--config=") {
            config_path = Some(std::path::PathBuf::from(v));
        } else if let Some(v) = arg.strip_prefix("--steps=") {
            for t in v.split(',') {
                steps.push(
                    t.parse::<u32>()
                        .map_err(|_| format!("--steps expects token counts, got {v:?}"))?,
                );
            }
        } else if let Some(v) = arg.strip_prefix("--step=") {
            step = Some(
                v.parse::<u32>()
                    .map_err(|_| format!("--step expects a token count, got {v:?}"))?,
            );
        } else {
            return Err(format!(
                "expected --config=PATH, --steps=T,... or --step=T, got {arg:?}"
            ));
        }
    }
    let path = config_path.ok_or("--config=PATH/config.json is required")?;
    let cfg = Qwen35Config::from_config_file(&path)?;
    let rt = GpuRuntime::new()?;
    let ws = rt.memory_info().recommended_working_set;
    println!(
        "device {}, recommended working set {} ({ws} B)",
        rt.device_name(),
        gib(ws)
    );

    // One bf16 model serves every bf16 variant; its own allocation is
    // measured, then each optimizer is allocated and dropped in turn.
    let base = rt.current_allocated_bytes();
    let model = Qwen35Model::random_tower(&rt, cfg.clone(), Precision::Bf16, 1)?;
    rt.synchronize()?;
    let weights_bf16 = rt.current_allocated_bytes() - base;
    println!("{}", model.describe());
    println!(
        "parameters {}",
        model
            .parameter_table()?
            .iter()
            .map(|p| p.shape.iter().product::<usize>() as u64)
            .sum::<u64>()
    );
    println!(
        "bf16 weights: {} allocated (computed {})",
        gib(weights_bf16),
        gib(table_bytes(&model, 2)?)
    );
    let bank = Qwen35Grads::zeros_like(&model)?;
    rt.synchronize()?;
    let bank_bf16 = rt.current_allocated_bytes() - base - weights_bf16;
    println!("bf16 gradient bank: {} allocated", gib(bank_bf16));
    let f32_tables = table_bytes(&model, 4)?;

    println!();
    println!(
        "{:<56} {:>11} {:>11} {:>11} {:>11}  fits",
        "variant", "weights", "bank", "optimizer", "total"
    );
    for (label, precision, config) in variants() {
        let opt = if precision == Precision::F32 {
            // AdamW::allocated_bytes_for reads the model's layout; an f32
            // model's moments are two f32 tables.
            2 * f32_tables
        } else {
            AdamW::allocated_bytes_for(&model, config)?
        };
        let (w, b) = if precision == Precision::F32 {
            (f32_tables, f32_tables)
        } else {
            (weights_bf16, bank_bf16)
        };
        let total = w + b + opt;
        let fits = total <= ws;
        let measured = if precision == Precision::Bf16 && rt.current_allocated_bytes() + opt <= ws {
            let before = rt.current_allocated_bytes();
            let state = AdamW::with_config(&model, config)?;
            rt.synchronize()?;
            let got = rt.current_allocated_bytes() - before;
            drop(state);
            rt.synchronize()?;
            format!(
                "measured optimizer {}, device total {}",
                gib(got),
                gib(before - base + got)
            )
        } else if precision == Precision::Bf16 {
            "computed only (not allocated: over the working set)".to_string()
        } else {
            "computed only".to_string()
        };
        println!(
            "{label:<56} {:>11} {:>11} {:>11} {:>11}  {}  {measured}",
            gib(w),
            gib(b),
            gib(opt),
            gib(total),
            if fits { "yes" } else { "NO" }
        );
    }

    if !steps.is_empty() {
        println!();
        for &t in &steps {
            println!(
                "train_step_bytes(T={t}, bf16): {}",
                gib(model.train_step_bytes(t, GemmOperands::Bf16))
            );
        }
    }
    drop(bank);
    drop(model);
    rt.synchronize()?;

    if let Some(t) = step {
        measure_step(&rt, &cfg, t)?;
    }
    Ok(())
}

/// One training step and one AdamW step at `t` tokens on the config's
/// model, Kahan and 8-bit moments resident.
fn measure_step(rt: &Arc<GpuRuntime>, cfg: &Qwen35Config, t: u32) -> Res<()> {
    println!();
    let base = rt.current_allocated_bytes();
    let model = Qwen35Model::random_tower(rt, cfg.clone(), Precision::Bf16, 2)?;
    let bank = Qwen35Grads::zeros_like(&model)?;
    let mut state = AdamW::with_config(
        &model,
        AdamWConfig {
            update: UpdateRule::Bf16Kahan,
            moments: MomentStorage::Block8,
        },
    )?;
    rt.synchronize()?;
    let resident = rt.current_allocated_bytes();
    let bound = model.train_step_bytes(t, GemmOperands::Bf16);
    let ids: Vec<u32> = (0..t).map(|i| (i * 104_729 + 17) % model.config().vocab).collect();
    rt.reset_peak_allocated_bytes();
    let loss = model.train_step_into(&ids, GemmOperands::Bf16, Supervise::Causal, &bank, false)?;
    let step_peak = rt.peak_allocated_bytes();
    model.adamw_step(
        &bank,
        &mut state,
        &AdamWHyper::default(),
        &model.default_weight_decay(0.0)?,
    )?;
    rt.synchronize()?;
    let peak = rt.peak_allocated_bytes();
    println!(
        "{}; resident {} (weights, bank, Kahan + 8-bit AdamW)",
        state.describe(),
        gib(resident - base)
    );
    println!(
        "step T={t}: loss {loss:.4}, peak {} over resident, bound {}; device peak {} \
         (with the AdamW step), recommended working set {}",
        gib(step_peak.saturating_sub(resident)),
        gib(bound),
        gib(peak - base),
        gib(rt.memory_info().recommended_working_set)
    );
    Ok(())
}
