//! `Qwen35Model::train_step_bytes` for one sequence is pinned: the
//! pre-flight bound ojas-qwen35's `train_step` relies on. The values were
//! first printed by 8c51319's `src/qwen35_train.rs` on the tiny fixture
//! (default dispatch, this GPU's tensor-op capability and pool cap, which the
//! bound reads), and re-pinned when the bound began counting two allocations
//! it had omitted (see `PINNED`).

use std::path::Path;

use tessl::gemm::GemmOperands;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::GpuRuntime;

/// `(t, bf16 operands, bytes)`.
///
/// Re-pinned: the bound omitted the GDN forward's kernel transients and the
/// cross-entropy's two id staging buffers (found by the exact byte tests in
/// `src/qwen35_train.rs`, which measure what each phase allocates). Old ->
/// new, per `(t, bf16 operands)`:
///
/// The delta at these `t` is entirely the GDN forward transient term: the
/// cross-entropy staging term does not bind in the step's max here, so only
/// `cross_entropy_bound_equals_the_bytes_the_cross_entropy_allocates` (in
/// `src/qwen35_train.rs`) guards it.
///
/// * (2, false): 2149185792 -> 2149284608
/// * (2, true): 2150339840 -> 2150438656
/// * (17, false): 2150411008 -> 2150509824
/// * (17, true): 2151967488 -> 2152066304
/// * (256, false): 2163621888 -> 2164017152
/// * (256, true): 2169847808 -> 2170243072
/// * (4096, false): 2378167296 -> 2384491520
/// * (4096, true): 2436969472 -> 2443293696
const PINNED: [(u32, bool, u64); 8] = [
    (2, false, 2149284608),
    (2, true, 2150438656),
    (17, false, 2150509824),
    (17, true, 2152066304),
    (256, false, 2164017152),
    (256, true, 2170243072),
    (4096, false, 2384491520),
    (4096, true, 2443293696),
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
