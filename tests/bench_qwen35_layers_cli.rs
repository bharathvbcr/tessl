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
    assert_refused_before_gpu(
        &["--attn-tiles", "--check-only"],
        "--attn-tiles takes token counts only",
    );
    assert_refused_before_gpu(
        &["--attn-tiles", "--paired-attn"],
        "--attn-tiles takes token counts only",
    );
}

/// A `--weights` directory that passes `checkpoint()`: the fixture config and a
/// safetensors file with an empty header, so the later guards are reached.
fn parsing_weights_dir(name: &str) -> Scratch {
    let d = valid_config_dir(name);
    std::fs::write(d.0.join("m.safetensors"), b"\x02\0\0\0\0\0\0\0{}").unwrap();
    d
}

#[test]
fn attn_tiles_refuses_every_other_lane_and_the_model_options() {
    // The `--weights`, `--gdn-simd` and `--gdn-scan32` cases must get past the checkpoint
    // parse to the `--paired-*` guard, so the directory is one that parses.
    let d = parsing_weights_dir("bq35_tiles_weights");
    let p = d.0.to_str().unwrap();
    assert_refused_before_gpu(
        &["--attn-tiles", &format!("--weights={p}")],
        "--weights, --gdn-simd and --gdn-scan32 apply to the model lanes",
    );
    assert_refused_before_gpu(
        &["--attn-tiles", "--gdn-scan32"],
        "--weights, --gdn-simd and --gdn-scan32 apply to the model lanes",
    );
    assert_refused_before_gpu(
        &["--attn-tiles", "--gdn-simd"],
        "--weights, --gdn-simd and --gdn-scan32 apply to the model lanes",
    );
    for lane in [
        "--check-only",
        "--paired-attn",
        "--model-forward",
        "--decode",
        "--decode=16",
        "--paired-gdn-recurrent",
        "--paired-gate-up",
    ] {
        assert_refused_before_gpu(&["--attn-tiles", lane], "--attn-tiles takes token counts only");
        assert_refused_before_gpu(&[lane, "--attn-tiles"], "--attn-tiles takes token counts only");
    }
}

#[test]
fn attn_tiles_refuses_iters_it_would_not_honour() {
    for args in [["--attn-tiles", "--iters=3"], ["--iters=3", "--attn-tiles"]] {
        assert_refused_before_gpu(&args, "--iters does not apply");
    }
    assert_refused_before_gpu(&["--attn-tiles", "--iters=0"], "--iters");
}

#[test]
fn paired_attn_bf16_refuses_every_other_lane_and_iters() {
    for lane in [
        "--attn-tiles",
        "--check-only",
        "--paired-attn",
        "--model-forward",
        "--decode",
        "--decode=16",
        "--paired-gdn-recurrent",
        "--paired-gate-up",
    ] {
        assert_refused_before_gpu(
            &["--paired-attn-bf16", lane],
            "--paired-attn-bf16 takes token counts only",
        );
        assert_refused_before_gpu(
            &[lane, "--paired-attn-bf16"],
            "--paired-attn-bf16 takes token counts only",
        );
    }
    for args in [["--paired-attn-bf16", "--iters=3"], ["--iters=3", "--paired-attn-bf16"]] {
        assert_refused_before_gpu(&args, "--iters does not apply");
    }
    assert_refused_before_gpu(
        &["--paired-attn-bf16", "--gdn-scan32"],
        "--weights, --gdn-simd and --gdn-scan32 apply to the model lanes",
    );
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
    // The bench times the library's default prefill GDN, not an override of it.
    assert!(
        stdout.contains("gdn prefill: Chunked(Cols16)"),
        "bench overrode the default GDN prefill:\n{stdout}"
    );
}

/// GPU: `--gdn-simd` reaches the model, which then runs the simd recurrence.
#[test]
fn gdn_simd_flag_reaches_the_model() {
    let (ok, stdout, stderr) = run(&["--gdn-simd", "--iters=1", "16"]);
    assert!(ok, "bench failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("gdn prefill: Simd"),
        "model did not get --gdn-simd:\n{stdout}"
    );
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

#[test]
fn token_count_cap_boundary_and_odd_forms_are_refused_before_any_gpu_work() {
    // One past the cap is refused in every lane that takes a token count.
    let over = "16777217";
    assert_refused_before_gpu(&[over], "at most 16777216");
    assert_refused_before_gpu(&["--check-only", over], "at most 16777216");
    assert_refused_before_gpu(&["--paired-attn", over], "at most 16777216");
    assert_refused_before_gpu(&["--paired-gate-up", over], "at most 16777216");
    assert_refused_before_gpu(&["--paired-gdn-recurrent", over], "at most 16777216");
    assert_refused_before_gpu(&[&format!("--decode={over}")], "at most 16777216");
    // A valid count earlier or later in the list does not mask an oversized one.
    assert_refused_before_gpu(&["1024", "4294967297"], "at most");
    assert_refused_before_gpu(&["4294967297", "1024"], "at most");
    assert_refused_before_gpu(&["--model-forward", "16", "8589934592", "16"], "at most");
    // Zero-extended / odd spellings never slip past the parser as a small count.
    for odd in ["-1", " 16", "16 ", "1_000", "0x10", "1e3", "\u{ff11}\u{ff16}", "", "٣"] {
        let (ok, stdout, stderr) = run(&["--check-only", odd]);
        assert!(!ok, "{odd:?} must be refused");
        assert!(!stdout.contains("device:"), "{odd:?} reached the GPU:\n{stdout}");
        assert!(stderr.contains("expected a token count"), "{odd:?}: {stderr}");
    }
    // Leading zeros are plain decimal and still bounded.
    assert_refused_before_gpu(&["00000000000000004294967297"], "at most");
    assert_refused_before_gpu(&["+4294967297"], "at most");
    // Numbers beyond usize are a parse refusal, not a wrap.
    assert_refused_before_gpu(&["99999999999999999999999999"], "expected a token count");
    assert_refused_before_gpu(&["--decode=99999999999999999999999999"], "--decode=P wants");
    assert_refused_before_gpu(&["--decode=0"], "must be positive");
}
