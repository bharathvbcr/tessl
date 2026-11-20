//! A failed `Qwen35Model::write_parameters` part way leaves some parameters
//! written and the rest not: the model must refuse every later forward and
//! write (`mark_torn`), while a failure before the first copy, or a write
//! that succeeds, leaves it usable.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::{GpuRuntime, Tensor};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load() -> (Arc<GpuRuntime>, Qwen35Model, SafeTensors<'static>) {
    let rt = GpuRuntime::new().unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap();
    (rt, model, st)
}

/// The table's current values, as tensors a write takes back.
fn current(rt: &Arc<GpuRuntime>, model: &Qwen35Model) -> Vec<Tensor> {
    let ts: Vec<Tensor> = model
        .parameter_table()
        .unwrap()
        .iter()
        .map(|p| rt.alloc_tensor_f32(&p.storage_shape()).unwrap())
        .collect();
    model.read_parameters(&ts).unwrap();
    ts
}

const IDS: [u32; 3] = [1, 2, 3];

fn assert_torn(model: &Qwen35Model, what: &str) {
    let forward = model.forward(&IDS, false).err().expect("forward after a torn write");
    assert!(
        forward.contains("part of the way through a step that failed"),
        "{what}: {forward}"
    );
    let host = model
        .write_parameter_host(0, &[])
        .expect_err("host write after a torn write");
    assert!(
        host.contains("part of the way through a step that failed"),
        "{what}: {host}"
    );
    let begin = model.begin(&IDS).err().expect("begin after a torn write");
    assert!(
        begin.contains("part of the way through a step that failed"),
        "{what}: {begin}"
    );
}

#[test]
fn a_write_that_fails_part_way_tears_the_model() {
    let (rt, model, _st) = load();
    let ts = current(&rt, &model);
    let k = ts.len() / 2;
    assert!(k > 0);
    model.fail_parameter_copy_after_for_test(k);
    let e = model.write_parameters(&ts).expect_err("injected copy fault");
    assert!(e.contains("injected fault"), "{e}");
    assert_torn(&model, "after a fault on copy k");
    // The latch is not cleared by a later, clean write.
    let again = model.write_parameters(&ts).expect_err("write on a torn model");
    assert!(again.contains("part of the way through a step that failed"), "{again}");
}

#[test]
fn a_write_that_fails_before_any_copy_leaves_the_model_usable() {
    let (rt, model, _st) = load();
    let before = model.forward(&IDS, false).unwrap().logits;
    let ts = current(&rt, &model);
    model.fail_parameter_copy_after_for_test(0);
    let e = model.write_parameters(&ts).expect_err("injected copy fault");
    assert!(e.contains("injected fault"), "{e}");
    let after = model
        .forward(&IDS, false)
        .expect("a failure before any write must not tear the model");
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        after.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "nothing was written, yet the forward moved"
    );
    model
        .write_parameters(&ts)
        .expect("the fault fired once; a retry writes");
}

#[test]
fn a_successful_write_does_not_tear_the_model() {
    let (rt, model, _st) = load();
    let before = model.forward(&IDS, false).unwrap().logits;
    let ts = current(&rt, &model);
    model.write_parameters(&ts).unwrap();
    let after = model.forward(&IDS, false).expect("forward after a clean write").logits;
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        after.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    model.write_parameters(&ts).unwrap();
}
