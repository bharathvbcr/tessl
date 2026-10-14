//! `bench_qwen35_layers` command line: bad inputs are refused before the GPU
//! runtime starts, and `--iters=N` reaches the per-layer table's label.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_bench_qwen35_layers");

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn bench_qwen35_layers");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A scratch directory under the target dir, removed on drop.
struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let p = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create scratch dir");
        Scratch(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_refused_before_gpu(args: &[&str], needle: &str) {
    let (ok, stdout, stderr) = run(args);
    assert!(!ok, "{args:?} must fail");
    assert!(stderr.contains(needle), "{args:?}: stderr lacks {needle:?}:\n{stderr}");
    assert!(
        !stdout.contains("device:"),
        "{args:?} started the GPU runtime before refusing:\n{stdout}"
    );
}

#[test]
fn bad_weights_dirs_are_refused_before_any_gpu_work() {
    assert_refused_before_gpu(&["--weights=/nonexistent-tessl-dir", "1024"], "--weights");
    assert_refused_before_gpu(&["--weights=/tmp", "1024"], "--weights");
    let empty = Scratch::new("bq35_empty_ckpt");
    let d = empty.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={d}"), "1024"], "config.json");
    std::fs::write(empty.0.join("config.json"), "{}").unwrap();
    assert_refused_before_gpu(&[&format!("--weights={d}"), "1024"], "found 0");
    std::fs::write(empty.0.join("a.safetensors"), b"").unwrap();
    std::fs::write(empty.0.join("b.safetensors"), b"").unwrap();
    assert_refused_before_gpu(&[&format!("--weights={d}"), "1024"], "found 2");
}

#[test]
fn bad_flags_are_refused_before_any_gpu_work() {
    assert_refused_before_gpu(&["--iters=0", "16"], "--iters");
    assert_refused_before_gpu(&["--bogus"], "expected a token count");
    assert_refused_before_gpu(&["0"], "T must be positive");
}

/// GPU: the per-layer table's header reports the sample count that ran.
#[test]
fn iters_flag_reaches_the_layer_table_label() {
    let (ok, stdout, stderr) = run(&["--iters=1", "16"]);
    assert!(ok, "bench failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("per layer (ms, median of 1,"),
        "layer table label ignores --iters:\n{stdout}"
    );
    assert!(!stdout.contains("median of 7"), "stale default in output:\n{stdout}");
}
