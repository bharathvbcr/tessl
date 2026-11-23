//! A failed `Qwen35Model::write_parameters` part way leaves some parameters
//! written and the rest not: the model must refuse every later forward and
//! write (`mark_torn`), while a failure before the first copy, or a write
//! that succeeds, leaves it usable.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::with_gpu;
use tessl::qwen35_model::{LogitRows, Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;
use tessl::{GpuRuntime, Tensor};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn load(rt: &Arc<GpuRuntime>) -> (Qwen35Model, SafeTensors<'static>) {
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let model = Qwen35Model::load(rt, &st, "model.", cfg, Precision::F32).unwrap();
    (model, st)
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
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let ts = current(rt, &model);
        let k = ts.len() / 2;
        assert!(k > 0);
        model.fail_parameter_copy_after_for_test(k);
        let e = model.write_parameters(&ts).expect_err("injected copy fault");
        assert!(e.contains("injected fault"), "{e}");
        assert_torn(&model, "after a fault on copy k");
        // The latch is not cleared by a later, clean write.
        let again = model.write_parameters(&ts).expect_err("write on a torn model");
        assert!(again.contains("part of the way through a step that failed"), "{again}");
    });
}

#[test]
fn a_write_that_fails_before_any_copy_leaves_the_model_usable() {
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let before = model.forward(&IDS, false).unwrap().logits;
        let ts = current(rt, &model);
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
    });
}

#[test]
fn a_successful_write_does_not_tear_the_model() {
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let before = model.forward(&IDS, false).unwrap().logits;
        let ts = current(rt, &model);
        model.write_parameters(&ts).unwrap();
        let after = model.forward(&IDS, false).expect("forward after a clean write").logits;
        assert_eq!(
            before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            after.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        model.write_parameters(&ts).unwrap();
    });
}

const TORN: &str = "part of the way through a step that failed";

/// A failure on the second entry, and on the last, has encoded earlier
/// copies: both tear. (Entry 0 is the other test; `Window::tensor_as` and
/// `window_copy` cannot fail on entry i > 0 after `check` passed except
/// through the runtime, which is the same failure the hook stands for.)
#[test]
fn a_fault_on_the_second_and_on_the_last_entry_tears_the_model() {
    with_gpu(|rt| {
        let (probe, _st) = load(rt);
        let n = probe.parameter_table().unwrap().len();
        drop(probe);
        for k in [1, n - 1] {
            let (model, _st) = load(rt);
            let ts = current(rt, &model);
            model.fail_parameter_copy_after_for_test(k);
            let e = model.write_parameters(&ts).expect_err("injected copy fault");
            assert!(e.contains("injected fault"), "k={k}: {e}");
            assert_torn(&model, &format!("fault after {k} of {n} copies"));
        }
    });
}

/// A write refused by its checks (a wrong tensor in the last position) has
/// written nothing: it does not tear the model and does not retire a session
/// ("every tensor is checked before anything is written").
#[test]
fn a_write_refused_by_its_checks_leaves_model_and_sessions_alone() {
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let layers = model.config().layers.len();
        let mut ts = current(rt, &model);
        let mut staged = model.begin(&IDS).unwrap();
        staged.advance_to(1).unwrap();
        *ts.last_mut().unwrap() = rt.alloc_tensor_f32(&[1]).unwrap();
        let e = model
            .write_parameters(&ts)
            .expect_err("a wrong shape in the last tensor");
        assert!(!e.contains(TORN) && !e.contains("injected"), "{e}");
        staged
            .advance_to(layers)
            .expect("a refused write must not retire the session");
        staged.logits(LogitRows::Last).unwrap();
        model
            .forward(&IDS, false)
            .expect("a refused write must not tear the model");
    });
}

/// A copy that fails on entry 0 wrote nothing and does not tear the model,
/// but the generation moved first (write_parameters' doc: a session from
/// before the write refuses), so a session begun before it is retired
/// conservatively and a fresh one works. Pins the bump-before-copy order.
#[test]
fn a_copy_zero_failure_retires_an_earlier_session_but_not_the_model() {
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let ts = current(rt, &model);
        let mut staged = model.begin(&IDS).unwrap();
        model.fail_parameter_copy_after_for_test(0);
        model.write_parameters(&ts).expect_err("injected copy fault");
        let m = staged.advance_to(1).expect_err("session across a (failed) write");
        assert!(m.contains("parameters were written"), "{m}");
        model.begin(&IDS).unwrap().advance_to(1).expect("a fresh session works");
    });
}

/// The hook is keyed to the model it was armed on: a second model on the
/// same thread writes cleanly while the first one's hook is armed.
#[test]
fn a_hook_armed_on_one_model_does_not_fire_on_another() {
    with_gpu(|rt| {
        let (a, _sa) = load(rt);
        let (b, _sb) = load(rt);
        let (ta, tb) = (current(rt, &a), current(rt, &b));
        a.fail_parameter_copy_after_for_test(1);
        b.write_parameters(&tb)
            .expect("another model's hook must not fire here");
        b.forward(&IDS, false).expect("and must not tear it");
        a.write_parameters(&ta)
            .expect_err("the armed model's write still fails");
        assert_torn(&a, "the armed model");
    });
}

/// A real runtime failure (not the hook): with the runtime poisoned, the
/// write fails on its first copy, before anything is encoded, so the weights
/// are untouched and the model is not torn (the refusal is the poison).
#[test]
fn a_poisoned_runtime_fails_the_first_copy_without_tearing() {
    with_gpu(|rt| {
        let (model, _st) = load(rt);
        let ts = current(rt, &model);
        rt.poison_as_shared_event_timeout_for_test();
        let e = model.write_parameters(&ts).expect_err("a poisoned runtime");
        assert!(!e.contains(TORN), "{e}");
        let f = model.forward(&IDS, false).err().expect("forward on a poisoned runtime");
        assert!(!f.contains(TORN), "a first-copy failure tore the model: {f}");
    });
}
