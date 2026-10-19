//! `Qwen35Model::train_step_bytes` for one sequence is what it was before the
//! ragged step existed: the pre-flight bound ojas-qwen35's `train_step`
//! relies on. The values were printed by 8c51319's `src/qwen35_train.rs` on
//! the tiny fixture (default dispatch, this GPU's tensor-op capability and
//! pool cap, which the bound reads), then pinned.

use std::path::Path;

use tessl::gemm::GemmOperands;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

/// `(t, bf16 operands, bytes)`.
const PINNED: [(u32, bool, u64); 8] = [
    (2, false, 2149185792),
    (2, true, 2150339840),
    (17, false, 2150411008),
    (17, true, 2151967488),
    (256, false, 2163621888),
    (256, true, 2169847808),
    (4096, false, 2378167296),
    (4096, true, 2436969472),
];

#[test]
fn the_single_sequence_step_bound_is_unchanged() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train");
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap();
    let mut bad = Vec::new();
    for (t, bf16, want) in PINNED {
        let mm = if bf16 {
            GemmOperands::Bf16
        } else {
            GemmOperands::ExactF32
        };
        let got = model.train_step_bytes(t, mm);
        eprintln!("PIN ({t}, {bf16}, {got}),");
        if got != want {
            bad.push((t, bf16, got, want));
        }
    }
    assert!(bad.is_empty(), "(t, bf16, got, pinned): {bad:?}");
}
