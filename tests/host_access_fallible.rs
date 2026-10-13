//! Library paths that write a small host-visible buffer report a busy runtime
//! (a live `HostMapping`) as an error instead of panicking, and the runtime is
//! usable once the mapping is dropped.

mod common;

use common::with_gpu;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use tessl::decode_icb::DecodeIcb;
use tessl::embedgemma2::upload_segments;
use tessl::qwen35_model::{LogitRows, Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| (i * 37 + 11) % 64).collect()
}

/// `Staged::logits`, `Staged::score_answers` and `forward_rows` (the
/// `read_rows` path) report a busy runtime as an error, and the same session
/// and model work once the mapping is dropped. (These already returned `Err`
/// before this lane: coverage, not a regression for a code change.)
#[test]
fn staged_readbacks_report_a_busy_runtime_instead_of_panicking() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    let n = cfg.layers.len();
    with_gpu(|rt| {
        let model = Qwen35Model::load(rt, &st, "model.", cfg.clone(), Precision::F32).unwrap();
        let mut s = model.begin(&ids(4)).unwrap();
        s.advance_to(n).unwrap();
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let busy = |what: &str, r: std::thread::Result<Result<(), String>>| {
            let e = r.unwrap_or_else(|_| panic!("{what} panicked on a busy runtime")).unwrap_err();
            assert!(e.contains("busy"), "{what}: {e}");
        };
        busy("logits", catch_unwind(AssertUnwindSafe(|| s.logits(LogitRows::Last).map(|_| ()))));
        busy("score_answers", catch_unwind(AssertUnwindSafe(|| s.score_answers(&[0], &[1, 2]).map(|_| ()))));
        busy(
            "forward_rows",
            catch_unwind(AssertUnwindSafe(|| model.forward_rows(&ids(4), LogitRows::Last, false).map(|_| ()))),
        );
        drop(mapping);
        assert!(!s.logits(LogitRows::Last).unwrap().is_empty());
        assert!(s.score_answers(&[0], &[1, 2]).is_ok());
        assert!(model.forward_rows(&ids(4), LogitRows::Last, false).is_ok());
    });
}

#[test]
fn upload_segments_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let outcome = catch_unwind(AssertUnwindSafe(|| upload_segments(rt, &[(0, 2), (2, 5)], 5)));
        let err = outcome.expect("upload_segments panicked on a busy runtime").unwrap_err();
        assert!(err.contains("busy") && err.contains("upload_segments"), "{err}");
        drop(mapping);
        let ok = upload_segments(rt, &[(0, 2), (2, 5)], 5).unwrap();
        assert_eq!(ok.read_u32(), vec![0, 2, 2, 5]);
    });
}

/// An `n` that cannot be a u32 is refused by name before anything is
/// allocated (a 16 GiB-per-buffer allocation would otherwise be attempted).
#[test]
fn mini_copy_chain_refuses_an_n_past_u32_before_allocating() {
    with_gpu(|rt| {
        let n = u32::MAX as usize + 1;
        let err = DecodeIcb::mini_copy_chain(rt, n).map(|_| ()).unwrap_err();
        assert!(err.contains("mini_copy_chain") && err.contains("u32"), "{err}");
        let err = DecodeIcb::mini_copy_chain(rt, usize::MAX).map(|_| ()).unwrap_err();
        assert!(err.contains("u32"), "{err}");
        assert!(DecodeIcb::mini_copy_chain(rt, 0).is_err());
    });
}

#[test]
fn mini_copy_chain_reports_a_busy_runtime_instead_of_panicking() {
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let outcome = catch_unwind(AssertUnwindSafe(|| DecodeIcb::mini_copy_chain(rt, 8).map(|_| ())));
        let err = outcome.expect("mini_copy_chain panicked on a busy runtime").unwrap_err();
        assert!(err.contains("busy"), "{err}");
        drop(mapping);
        let (mut dec, out) = DecodeIcb::mini_copy_chain(rt, 8).unwrap();
        dec.execute(rt).unwrap();
        rt.synchronize().unwrap();
        assert_eq!(out.read_f32(), (1..=8).map(|i| i as f32).collect::<Vec<_>>());
    });
}
