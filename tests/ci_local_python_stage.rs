//! The Python GPU stage of `scripts/ci_local.sh` must not pass with its tiny fixtures missing.
//!
//! Wave 5's integration run printed `Ran 173 tests ... OK (skipped=42)`: 20 skips for the real
//! checkpoint (expected) and 22 for `run cargo test --release --test capi_embedgemma2... to write
//! the tiny snapshot/checkpoint`, because the stage ran before the release tests that write those
//! fixtures. `scripts/unittest_fixture_skips.sh` reads the unittest output and fails on the 22; the
//! stage now runs after the release tests. No GPU, no torch, no Metal toolchain: the detector reads
//! text, and the order is read from the script's source. `tests/ci_local_python_stage_w5.txt` is
//! lines 4150-4285 of that run's log (target-gp/w5-int-run2/ci_local.log), unedited.
#![cfg(unix)]

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const W5_LOG: &str = include_str!("ci_local_python_stage_w5.txt");
const CI_LOCAL: &str = include_str!("../scripts/ci_local.sh");

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Runs the detector on `input`; returns (exit code, stdout, stderr).
fn detect(input: &str) -> (i32, String, String) {
    let mut child = Command::new("bash")
        .arg(root().join("scripts/unittest_fixture_skips.sh"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the detector");
    child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn is_fixture_skip(line: &str) -> bool {
    line.contains("skipped 'run `cargo test")
}

#[test]
fn the_real_w5_output_fails_and_names_all_22_fixture_skips() {
    assert_eq!(W5_LOG.lines().filter(|l| is_fixture_skip(l)).count(), 22, "the excerpt is the real log");
    let (code, stdout, _) = detect(W5_LOG);
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("fixture_skips=22"), "{stdout}");
    for capi in ["capi_embedgemma2`", "capi_embedgemma2_media`", "capi_embedgemma2_raw`", "capi_embedgemma2_multimodal`"] {
        assert!(stdout.contains(capi), "the {capi} skips are named: {stdout}");
    }
    assert!(!stdout.contains("EMBEDGEMMA2_SNAPSHOT"), "real-checkpoint skips are not listed: {stdout}");
}

#[test]
fn only_real_checkpoint_skips_pass() {
    let clean: String = W5_LOG.lines().filter(|l| !is_fixture_skip(l)).map(|l| format!("{l}\n")).collect();
    let real = clean.lines().filter(|l| l.contains("skipped 'set EMBEDGEMMA2_SNAPSHOT")).count();
    assert_eq!(real, 20, "the 20 expected skips stay in the input");
    let (code, stdout, stderr) = detect(&clean);
    assert_eq!(code, 0, "{stdout}{stderr}");
    assert!(stdout.contains("fixture_skips=0"), "{stdout}");
}

#[test]
fn empty_output_is_not_a_pass() {
    let (code, _, stderr) = detect("");
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("nothing was examined"), "{stderr}");
}

#[test]
fn output_with_no_summary_is_not_a_pass() {
    // The stage died before unittest finished: the lines before the crash are all it has.
    let partial: String = W5_LOG.lines().filter(|l| !l.starts_with("Ran ")).map(|l| format!("{l}\n")).collect();
    let (code, _, stderr) = detect(&partial);
    assert_eq!(code, 2, "{stderr}");
}

#[test]
fn a_truncated_skip_line_still_counts() {
    let cut = "test_x (test_embedgemma2.Library.test_x) ... skipped 'run `cargo test --release --te";
    let (code, stdout, _) = detect(&format!("{cut}\n\nRan 1 test in 0.1s\n\nOK (skipped=1)\n"));
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("fixture_skips=1"), "{stdout}");
    // Cut before the word `skipped`, the line names nothing and is not a fixture skip.
    let (code, _, _) = detect("test_x (test_embedgemma2.Library.test_x) ... ski\n\nRan 1 test in 0.1s\n");
    assert_eq!(code, 0);
}

#[test]
fn a_library_that_was_not_built_is_a_fixture_skip() {
    let line = "test_y (test_clip_coef.T.test_y) ... skipped 'libtessl.dylib is not built: run `cargo build --release`'";
    let (code, stdout, _) = detect(&format!("{line}\n\nRan 1 test in 0.1s\n\nOK (skipped=1)\n"));
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("fixture_skips=1"), "{stdout}");
}

/// The detector keys on the start of the skip reason the Python tests write. If a test's reason is
/// reworded, this fails here rather than the detector going quiet.
#[test]
fn the_python_tests_still_word_their_fixture_skips_as_the_detector_expects() {
    let mut files = 0;
    for entry in fs::read_dir(root().join("python/tests")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "py") {
            continue;
        }
        let src = fs::read_to_string(&path).unwrap();
        if src.contains("to write the tiny") {
            files += 1;
            assert!(src.contains("\"run `cargo test --release --test "), "{} rewords the skip", path.display());
        }
    }
    assert_eq!(files, 4, "the four capi_embedgemma2* fixture users");
}

/// The tiny fixtures are written by the release `cargo test` step, so the Python GPU stage runs
/// after it, and consults the detector. Pre-fix the stage came first.
#[test]
fn the_python_gpu_stage_runs_after_the_release_tests_and_checks_for_fixture_skips() {
    let at = |needle: &str| {
        let hits: Vec<usize> = CI_LOCAL
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with('#') && l.contains(needle))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hits.len(), 1, "{needle:?} appears once as a command: {hits:?}");
        hits[0]
    };
    let release = at("cargo test --release --no-fail-fast");
    let python = at("unittest discover -s python/tests");
    let detector = at("scripts/unittest_fixture_skips.sh");
    assert!(release < python, "python stage (line {}) after release tests (line {})", python + 1, release + 1);
    assert!(python < detector);
    assert!(CI_LOCAL.contains("FIXTURE_SKIPS_FAILED"), "a fixture skip ends the run non-zero");
    assert!(CI_LOCAL.contains("SKIPPED: python/tests GPU tests (MAC_HEAVY_LABEL unset"), "the unlocked wording stays");
}
