//! The OOM pre-flight (`train_step_bytes`, `train_step_ragged_bytes`) mirrors
//! the step's allocations by hand, so a buffer added to the step without a
//! mirror makes the bound too low. The bound's last term is the pool's freelist
//! cap, which dwarfs these tiny steps and would hide a missing mirror, so each
//! measured peak is held against the bound *without* that term as well.
//!
//! Two shapes each, for the single-sequence and the ragged step, on the f32
//! model and the bf16 tower. Peaks are device-wide, so this file's one test
//! runs everything serially.

use std::path::{Path, PathBuf};

use tessl::gemm::GemmOperands;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{Qwen35Grads, Supervise};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load(bf16_tower: bool) -> (std::sync::Arc<GpuRuntime>, Qwen35Model) {
    let dir = fixture();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let model = if bf16_tower {
        Qwen35Model::load_tower(&rt, &st, "model.", cfg, Precision::Bf16).unwrap()
    } else {
        Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap()
    };
    (rt, model)
}

fn ids(n: usize, vocab: u32, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 7_919 + salt * 31 + 3) % vocab).collect()
}

/// `(what, measured peak above the step's start, bound, the bound's pool term)`.
type Row = (String, u64, u64, u64);

#[test]
fn the_step_bounds_cover_the_measured_peaks_without_their_pool_term() {
    let mut rows: Vec<Row> = Vec::new();
    for (bf16, mm) in [(false, GemmOperands::ExactF32), (true, GemmOperands::Bf16)] {
        let (rt, model) = load(bf16);
        let vocab = model.config().vocab;
        let bank = Qwen35Grads::zeros_like(&model).unwrap();
        let cap = rt.memory_info().pool_cache_cap as u64;
        // Run once untimed (lazily built, step-independent state is not the
        // step's own allocation), then empty the freelist so the measured run
        // starts from nothing cached and its peak is its own bytes.
        let measure = |run: &dyn Fn()| {
            run();
            rt.synchronize().unwrap();
            rt.set_pool_cache_cap_bytes(0);
            rt.set_pool_cache_cap_bytes(cap as usize);
            let before = rt.current_allocated_bytes();
            rt.reset_peak_allocated_bytes();
            run();
            rt.synchronize().unwrap();
            rt.peak_allocated_bytes().saturating_sub(before)
        };
        for t in [17usize, 256] {
            let x = ids(t, vocab, 1);
            let peak = measure(&|| {
                model.train_step_into(&x, mm, Supervise::Causal, &bank, false).unwrap();
            });
            let bound = model.train_step_bytes(t as u32, mm);
            rows.push((format!("single bf16={bf16} t={t}"), peak, bound, cap));
        }
        for lens in [vec![5usize, 17, 30], vec![64usize, 3, 40, 100]] {
            let mut cu = vec![0u32];
            for l in &lens {
                cu.push(cu.last().unwrap() + *l as u32);
            }
            let total = *cu.last().unwrap() as usize;
            let x = ids(total, vocab, 2);
            let in_place = !bf16;
            let peak = measure(&|| {
                model.train_step_ragged_into(&x, &cu, mm, &bank, false).unwrap();
            });
            let bound = model.train_step_ragged_bytes(&cu, total, mm, in_place).unwrap();
            rows.push((format!("ragged bf16={bf16} lens={lens:?}"), peak, bound, cap));
        }
    }
    let mut bad = Vec::new();
    for (what, peak, bound, cap) in &rows {
        eprintln!("{what}: peak {peak} B, bound {bound} B, pool term {cap} B");
        if bound < cap || *peak > bound - cap {
            bad.push(format!(
                "{what}: peak {peak} over the bound's own part {}",
                bound.saturating_sub(*cap)
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
