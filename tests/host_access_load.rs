//! The fallible twins of the panicking host accessors, and the load paths that
//! use them: a live `HostMapping` makes a weight upload an `Err("busy")`, not a
//! panic, and the same runtime loads fine once the mapping is dropped.

mod common;

use common::with_gpu;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::safetensors::SafeTensors;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train")
}

/// A live mapping is `Err("busy")` (not a panic), a wrong length is `Err`, and
/// the buffer works afterwards.
#[test]
fn try_write_f32_and_try_read_u32_report_busy_and_bad_length() {
    with_gpu(|rt| {
        let f = rt.alloc_buffer(16).unwrap();
        let u = rt.alloc_buffer(16).unwrap();
        u.write_u32(&[1, 2, 3, 4]);
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        let e = f.try_write_f32(&[0.0; 4]).unwrap_err();
        assert!(e.contains("busy"), "{e}");
        let e = u.try_read_u32().unwrap_err();
        assert!(e.contains("busy"), "{e}");
        drop(mapping);
        let e = f.try_write_f32(&[0.0; 3]).unwrap_err();
        assert!(e.contains("write_f32") && e.contains('3') && e.contains('4'), "{e}");
        f.try_write_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(f.read_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(u.try_read_u32().unwrap(), vec![1, 2, 3, 4]);
    });
}

/// Every weight upload (`Loader::buf` norms, the embedding table,
/// `random_tower`'s tensors) reports a busy runtime as an error.
#[test]
fn load_paths_report_a_busy_runtime_instead_of_panicking() {
    let cfg = Qwen35Config::from_config_file(&fixture().join("config.json")).unwrap();
    let st = SafeTensors::open(&fixture().join("model.safetensors")).unwrap();
    with_gpu(|rt| {
        let held = rt.alloc_buffer(64).unwrap();
        let mapping = held.contents_u8();
        for p in [Precision::F32, Precision::Bf16] {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                Qwen35Model::load(rt, &st, "model.", cfg.clone(), p).map(|_| ())
            }));
            let e = outcome
                .unwrap_or_else(|_| panic!("Qwen35Model::load({p:?}) panicked on a busy runtime"))
                .unwrap_err();
            assert!(e.contains("busy"), "load {p:?}: {e}");
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                Qwen35Model::random_tower(rt, cfg.clone(), p, 7).map(|_| ())
            }));
            let e = outcome
                .unwrap_or_else(|_| panic!("random_tower({p:?}) panicked on a busy runtime"))
                .unwrap_err();
            assert!(e.contains("busy"), "random_tower {p:?}: {e}");
        }
        drop(mapping);
        for p in [Precision::F32, Precision::Bf16] {
            Qwen35Model::load(rt, &st, "model.", cfg.clone(), p).unwrap();
            Qwen35Model::random_tower(rt, cfg.clone(), p, 7).unwrap();
        }
    });
}
