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

#[test]
fn config_json_that_is_a_directory_is_refused_before_any_gpu_work() {
    let d = Scratch::new("bq35_cfg_dir");
    std::fs::create_dir(d.0.join("config.json")).unwrap();
    std::fs::write(d.0.join("a.safetensors"), b"").unwrap();
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "config.json");
}

#[test]
fn unicode_and_spaced_weights_paths_are_refused_cleanly() {
    let d = Scratch::new("bq35_ü ñ dir");
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "config.json");
}

#[test]
fn bad_weights_with_a_paired_lane_is_refused_before_any_gpu_work() {
    assert_refused_before_gpu(&["--weights=/nonexistent-tessl-dir", "--paired-attn"], "--weights");
}

#[test]
fn a_later_bad_weights_flag_wins_over_nothing_and_a_later_bad_iters_still_refuses() {
    assert_refused_before_gpu(&["--weights=/nonexistent-tessl-dir", "--iters=0", "16"], "--weights");
    assert_refused_before_gpu(&["--iters=-1", "16"], "--iters");
    assert_refused_before_gpu(&["--iters=", "16"], "--iters");
    assert_refused_before_gpu(&["--iters=99999999999999999999999", "16"], "--iters");
}

#[test]
fn directory_named_safetensors_and_unloadable_checkpoints_are_refused_before_any_gpu_work() {
    let d = Scratch::new("bq35_dir_st");
    std::fs::create_dir(d.0.join("x.safetensors")).unwrap();
    std::fs::write(d.0.join("config.json"), "{}").unwrap();
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "found 0");
    // A real file but an unusable config / empty checkpoint.
    let e = Scratch::new("bq35_bad_content");
    std::fs::write(e.0.join("config.json"), "{}").unwrap();
    std::fs::write(e.0.join("a.safetensors"), b"").unwrap();
    let q = e.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={q}"), "16"], "--weights");
}

#[test]
fn huge_iters_is_refused_before_any_gpu_work() {
    assert_refused_before_gpu(&["--iters=18446744073709551615", "16"], "--iters");
    assert_refused_before_gpu(&["--iters=1000000000", "16"], "--iters");
    assert_refused_before_gpu(&["--iters=10001", "16"], "--iters");
}

fn valid_config_dir(name: &str) -> Scratch {
    let d = Scratch::new(name);
    let cfg = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35/config_qwen35_2b_base.json");
    std::fs::copy(cfg, d.0.join("config.json")).expect("copy fixture config");
    d
}

#[test]
fn valid_config_with_corrupt_safetensors_is_refused_before_any_gpu_work() {
    let d = valid_config_dir("bq35_corrupt_st");
    let p = d.0.to_str().unwrap().to_string();
    // Lying header length (2^63-1), truncated header, non-JSON header, and an 8-byte-only file.
    let cases: [&[u8]; 4] = [
        b"\xff\xff\xff\xff\xff\xff\xff\x7f{}",
        b"\x10\0\0\0\0\0\0\0{}",
        b"\x02\0\0\0\0\0\0\0zz",
        b"\0\0\0\0\0\0\0\0",
    ];
    for bytes in cases {
        std::fs::write(d.0.join("m.safetensors"), bytes).unwrap();
        assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "--weights");
    }
}

#[test]
fn broken_and_directory_symlinked_safetensors_are_not_counted() {
    let d = valid_config_dir("bq35_symlinks");
    std::os::unix::fs::symlink("/nonexistent-tessl-target", d.0.join("a.safetensors")).unwrap();
    std::os::unix::fs::symlink("/tmp", d.0.join("b.safetensors")).unwrap();
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "found 0");
}

#[test]
fn non_utf8_config_and_iters_forms_are_refused_before_any_gpu_work() {
    let d = Scratch::new("bq35_nonutf8_cfg");
    std::fs::write(d.0.join("config.json"), b"\xff\xfe\x00").unwrap();
    std::fs::write(d.0.join("a.safetensors"), b"\x02\0\0\0\0\0\0\0{}").unwrap();
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(&[&format!("--weights={p}"), "16"], "--weights");
    for bad in [
        "--iters= 5",
        "--iters=1e3",
        "--iters=10000x",
        "--iters=0x10",
        "--iters=\u{661}",
    ] {
        assert_refused_before_gpu(&[bad, "16"], "--iters");
    }
}

#[test]
fn token_counts_above_u32_are_refused_before_any_gpu_work() {
    // 2^32 + 1 used to wrap to a 1-token sequence while labels kept the big T.
    for big in ["4294967297", "4294967295", "18446744073709551615"] {
        assert_refused_before_gpu(&["--check-only", big], "at most");
        assert_refused_before_gpu(&[big], "at most");
        assert_refused_before_gpu(&["--model-forward", big], "at most");
        assert_refused_before_gpu(&[&format!("--decode={big}")], "at most");
    }
    assert_refused_before_gpu(&["--paired-gdn-recurrent", "4294967297"], "at most");
}
