//! `Qwen35Model`'s weights are long-lived (`BufferKind::Hot`): dropping the
//! model hands their memory back to the device. Allocated as mid-step
//! temporaries (`Cold`) they would park in the runtime's freelist instead, and
//! the device would still hold them after the model is gone. Runs on the tiny
//! two-layer checkpoint in `tests/fixtures/qwen35_train/`, at both precisions,
//! with and without the LM head.

mod common;

use std::path::{Path, PathBuf};

use common::with_gpu;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

#[test]
fn dropping_the_model_releases_its_weights() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        for precision in [Precision::F32, Precision::Bf16] {
            for tower in [false, true] {
                let what = format!("{precision:?}{}", if tower { " tower" } else { "" });
                rt.synchronize().unwrap();
                let before = rt.current_allocated_bytes();
                let model = if tower {
                    Qwen35Model::load_tower(rt, &st, "model.", cfg.clone(), precision)
                } else {
                    Qwen35Model::load(rt, &st, "model.", cfg.clone(), precision)
                }
                .unwrap();
                assert!(
                    rt.current_allocated_bytes() > before,
                    "{what}: loading allocated nothing the device charges"
                );
                drop(model);
                rt.synchronize().unwrap();
                let after = rt.current_allocated_bytes();
                assert!(
                    after <= before,
                    "{what}: {} bytes still allocated after the model dropped",
                    after.saturating_sub(before)
                );
            }
        }
    });
}
