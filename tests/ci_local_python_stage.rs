//! The Python GPU stage of `scripts/ci_local.sh` must not pass with tests skipped for a cause
//! this run could have fixed, and the code that decides that is the code these tests run.
//!
//! Wave 5's integration run printed `Ran 173 tests ... OK (skipped=42)`: 20 skips for the real
//! checkpoint (expected) and 22 for `run cargo test --release --test capi_embedgemma2... to write
//! the tiny snapshot/checkpoint`, because the stage ran before the release tests that write those
//! fixtures. Three pieces, each executed here and not re-implemented:
//! - `scripts/unittest_fixture_skips.sh`, an allow-list over unittest's output (stdin);
//! - `scripts/ci_local_python_stage.sh`, which runs python, keeps the log and applies the detector;
//! - the `[python-stage]` and `[verdict]` fragments of `scripts/ci_local.sh`, cut out by their
//!   marker comments and run in a driver with a stand-in python (the guard, the early exit, the
//!   named SKIPS entries and the last line).
//!
//! No GPU, no torch, no Metal toolchain. `tests/ci_local_python_stage_w5.txt` is the python
//! section of wave 5's `ci_local.log` (lines 4106-4285 of target-gp/w5-int-run2), unedited.
#![cfg(unix)]

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const W5: &str = include_str!("ci_local_python_stage_w5.txt");
const CI_LOCAL: &str = include_str!("../scripts/ci_local.sh");

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn text(o: &Output) -> String {
    format!(
        "--- stdout\n{}--- stderr\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

// ---------------------------------------------------------------------------------------------
// The detector, on text.

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

fn is_real_checkpoint_skip(line: &str) -> bool {
    line.contains("skipped 'set EMBEDGEMMA2_")
}

/// The wave-5 output with the 22 fixture skips taken out, its summary rewritten to match.
fn w5_without_fixture_skips() -> String {
    let kept: String = W5
        .lines()
        .filter(|l| !is_fixture_skip(l))
        .map(|l| format!("{l}\n"))
        .collect();
    let out = kept
        .replace("Ran 173 tests", "Ran 151 tests")
        .replace("OK (skipped=42)", "OK (skipped=20)");
    assert_ne!(out, kept, "the summary was rewritten");
    out
}

/// A unittest report: these lines, then the `Ran` line and `summary` as unittest ends a run.
fn report(lines: &[&str], summary: &str) -> String {
    let mut s: String = lines.iter().map(|l| format!("{l}\n")).collect();
    s.push_str(&format!("\n{}\nRan 1 test in 0.001s\n\n{summary}\n", "-".repeat(70)));
    s
}

#[test]
fn the_real_w5_output_fails_and_names_exactly_the_22_fixture_skips() {
    assert_eq!(
        W5.lines().filter(|l| is_fixture_skip(l)).count(),
        22,
        "the excerpt is the real log"
    );
    assert_eq!(W5.lines().filter(|l| is_real_checkpoint_skip(l)).count(), 20);
    let (code, stdout, _) = detect(W5);
    assert_eq!(code, 1, "{stdout}");
    assert!(stdout.contains("unexpected_skips=22"), "{stdout}");
    for capi in [
        "capi_embedgemma2`",
        "capi_embedgemma2_media`",
        "capi_embedgemma2_raw`",
        "capi_embedgemma2_multimodal`",
    ] {
        assert!(stdout.contains(capi), "the {capi} skips are named: {stdout}");
    }
    assert!(
        !stdout.contains("EMBEDGEMMA2_SNAPSHOT"),
        "real-checkpoint skips are not listed: {stdout}"
    );
}

#[test]
fn the_20_real_checkpoint_skips_alone_pass() {
    let clean = w5_without_fixture_skips();
    assert_eq!(clean.lines().filter(|l| is_real_checkpoint_skip(l)).count(), 20);
    let (code, stdout, stderr) = detect(&clean);
    assert_eq!(code, 0, "{stdout}{stderr}");
    assert!(stdout.contains("unexpected_skips=0"), "{stdout}");
}

#[test]
fn a_run_with_no_skips_passes() {
    let (code, stdout, _) = detect("test_a (m.T.test_a) ... ok\n\nRan 1 test in 0.1s\n\nOK\n");
    assert_eq!(code, 0, "{stdout}");
}

#[test]
fn nothing_examined_is_not_a_pass() {
    for (name, input) in [
        ("empty", String::new()),
        (
            "no Ran line",
            w5_without_fixture_skips().replace("Ran 151 tests in 4.082s", ""),
        ),
        ("no verdict", w5_without_fixture_skips().replace("OK (skipped=20)", "")),
        (
            "only a crash",
            "Traceback (most recent call last):\n  File \"x\", line 1\n".to_string(),
        ),
    ] {
        let (code, stdout, stderr) = detect(&input);
        assert_eq!(code, 2, "{name}: {stdout}{stderr}");
        assert!(stderr.contains("nothing was examined"), "{name}: {stderr}");
    }
}

#[test]
fn a_parsed_count_that_differs_from_the_summary_is_not_examined() {
    // A skip line lost (20 -> 19 lines, summary still says 20) and one the summary does not count.
    let clean = w5_without_fixture_skips();
    let mut lines: Vec<&str> = clean.lines().collect();
    let at = lines.iter().position(|l| is_real_checkpoint_skip(l)).unwrap();
    lines.remove(at);
    let lost = lines.join("\n") + "\n";
    let (code, _, stderr) = detect(&lost);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("parsed 19 skip lines but the summary says skipped=20"),
        "{stderr}"
    );
    let extra = w5_without_fixture_skips().replace("OK (skipped=20)", "OK (skipped=21)");
    assert_eq!(detect(&extra).0, 2);
    // A summary with no skipped= at all but a skip line in the output.
    let none = w5_without_fixture_skips().replace("OK (skipped=20)", "OK");
    assert_eq!(detect(&none).0, 2);
}

#[test]
fn every_skip_that_is_not_an_external_checkpoint_one_fails() {
    for (name, reason) in [
        (
            "tiny fixture",
            "'run `cargo test --release --test capi_embedgemma2` to write the tiny checkpoint'",
        ),
        ("truncated", "'run `cargo test --release --te"),
        ("capital run", "'Run `cargo test` to write the tiny snapshot'"),
        ("capital Set", "'Set EMBEDGEMMA2_SNAPSHOT to the snapshot directory'"),
        ("mid-sentence set", "'please set EMBEDGEMMA2_SNAPSHOT to run this'"),
        (
            "lib not built",
            "'libtessl.dylib is not built: run `cargo build --release`'",
        ),
        ("lib mid-sentence", "'the libtessl.dylib is not built'"),
        (
            "apostrophe",
            "\"the reference script's imports (torch, PIL) are not installed\"",
        ),
        (
            "apostrophe fixture",
            "\"run `cargo test --test x` to write the tiny snapshot; it's missing\"",
        ),
        ("package", "'the transformers package is not installed'"),
        ("escaped both quotes", "'it\\'s \"quoted\" and unlisted'"),
    ] {
        let line = format!("test_x (m.T.test_x) ... skipped {reason}");
        let (code, stdout, stderr) = detect(&report(&[&line], "OK (skipped=1)"));
        assert_eq!(code, 1, "{name}: {stdout}{stderr}");
        assert!(stdout.contains("unexpected_skips=1"), "{name}: {stdout}");
    }
}

#[test]
fn external_checkpoint_skips_pass_in_either_quote_style_and_in_a_failed_run() {
    let single = "test_x (m.T.test_x) ... skipped 'set EMBEDGEMMA2_SNAPSHOT to the snapshot directory'";
    let double = "test_y (m.T.test_y) ... skipped \"set EMBEDGEMMA2_SNAPSHOT (it's required)\"";
    // setUpClass and module-level skips print without a test name, or with a loader name.
    let bare = "skipped 'set EMBEDGEMMA2_REF_DIR for the references'";
    let loader = "test_q (unittest.loader.ModuleSkipped) ... skipped 'set EMBEDGEMMA2_COMPARE_ST=1 to run it'";
    let (code, stdout, stderr) = detect(&report(&[single, double, bare, loader], "OK (skipped=4)"));
    assert_eq!(code, 0, "{stdout}{stderr}");
    let (code, stdout, stderr) = detect(&report(&[single, double], "FAILED (failures=1, skipped=2)"));
    assert_eq!(
        code, 0,
        "unittest's own failure is the python stage's to report: {stdout}{stderr}"
    );
}

// ---------------------------------------------------------------------------------------------
// The reasons python/tests can print, derived from its source, held to the classification.

/// Every skip reason literal in `python/tests/*.py`: the last string literal inside the
/// parentheses of `skipTest(`, `skipUnless(`, `skipIf(`, `skip(` and `SkipTest(`.
fn skip_reasons_in_source() -> BTreeSet<String> {
    let mut reasons = BTreeSet::new();
    for entry in fs::read_dir(root().join("python/tests")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "py") {
            continue;
        }
        let src = fs::read_to_string(&path).unwrap();
        for needle in ["skipTest(", "skipUnless(", "skipIf(", "unittest.skip(", "SkipTest("] {
            let mut from = 0;
            while let Some(i) = src[from..].find(needle) {
                let open = from + i + needle.len();
                from = open;
                if src[..open]
                    .lines()
                    .last()
                    .is_some_and(|l| l.trim_start().starts_with("def "))
                {
                    continue;
                }
                let literal = last_literal_in_parens(&src[open..])
                    .unwrap_or_else(|| panic!("{}: no literal reason in {needle}...: classify it", path.display()));
                reasons.insert(literal);
            }
        }
    }
    reasons
}

/// The last quoted literal before the parenthesis that closes the one already open.
fn last_literal_in_parens(s: &str) -> Option<String> {
    let (mut depth, mut last, mut chars) = (1usize, None, s.char_indices());
    while let Some((_, c)) = chars.next() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return last;
                }
            }
            '"' | '\'' => {
                let mut lit = String::new();
                for (_, d) in chars.by_ref() {
                    if d == c {
                        break;
                    }
                    lit.push(d);
                }
                last = Some(lit);
            }
            _ => {}
        }
    }
    None
}

/// `(reason, allowed)`: allowed means its cause is outside this ci_local run (an external
/// checkpoint or reference). Everything else is a skip this run can produce, so it fails.
const REASONS: [(&str, bool); 14] = [
    ("libtessl.dylib is not built: run `cargo build --release`", false), // the build step makes it
    (
        "run `cargo test --release --test capi_embedgemma2` to write the tiny checkpoint",
        false,
    ), // release tests write it
    (
        "run `cargo test --release --test capi_embedgemma2_multimodal` to write the tiny checkpoint",
        false,
    ),
    (
        "run `cargo test --release --test capi_embedgemma2_media` to write the tiny snapshot",
        false,
    ),
    (
        "run `cargo test --release --test capi_embedgemma2_raw` to write the tiny snapshot",
        false,
    ),
    (
        "run tools/embedgemma2_ref/make_segments_reference.py to write the references",
        false,
    ), // a repo script writes it
    ("the tokenizers package is not installed", false), // only reachable with a checkpoint; the oracle env must carry it
    ("the transformers package is not installed", false),
    (
        "the reference script's imports (torch, PIL, sentence-transformers) are not installed",
        false,
    ),
    (
        "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_REF_DIR) to run against the real checkpoint",
        true,
    ),
    (
        "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_MM_REF_DIR) to run against the real checkpoint",
        true,
    ),
    (
        "set EMBEDGEMMA2_SNAPSHOT (and EMBEDGEMMA2_REF_DIR, EMBEDGEMMA2_MM_REF_DIR) to run against the real checkpoint",
        true,
    ),
    (
        "set EMBEDGEMMA2_SNAPSHOT to the google/embeddinggemma-2 snapshot directory",
        true,
    ),
    ("set EMBEDGEMMA2_COMPARE_ST=1 to run sentence-transformers live", true),
];

#[test]
fn the_classification_covers_exactly_the_reasons_python_tests_can_print() {
    let table: BTreeSet<String> = REASONS.iter().map(|(r, _)| r.to_string()).collect();
    let source = skip_reasons_in_source();
    assert_eq!(
        source, table,
        "a reason was added, removed or reworded in python/tests: classify it"
    );
}

#[test]
fn the_detector_agrees_with_the_classification_of_every_reason() {
    for (reason, allowed) in REASONS {
        // unittest prints repr(reason): double quotes when the reason holds a single quote.
        let quoted = if reason.contains('\'') {
            format!("\"{reason}\"")
        } else {
            format!("'{reason}'")
        };
        let line = format!("test_x (m.T.test_x) ... skipped {quoted}");
        let (code, stdout, stderr) = detect(&report(&[&line], "OK (skipped=1)"));
        assert_eq!(code, if allowed { 0 } else { 1 }, "{reason}: {stdout}{stderr}");
    }
}

// ---------------------------------------------------------------------------------------------
// The stage as ci_local.sh runs it: ci_local's own fragments, the helper, a stand-in python.

fn fragment(name: &str) -> String {
    let (begin, end) = (format!("# [{name}:begin]"), format!("# [{name}:end]"));
    let mut inside = false;
    let mut out = String::new();
    for line in CI_LOCAL.lines() {
        if line.starts_with(&end) {
            inside = false;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
        if line.starts_with(&begin) {
            inside = true;
        }
    }
    assert!(!out.is_empty(), "ci_local.sh has no [{name}] fragment");
    out
}

#[derive(Default)]
struct Case {
    /// The stand-in python prints this, then exits `rc` or kills itself with `sig`.
    output: String,
    rc: i32,
    sig: Option<i32>,
    /// Set `MAC_HEAVY_LABEL` (ci_local under the lock) or not.
    unlocked: bool,
    /// Value of `TEST_FAILED` when the stage starts.
    test_failed: u8,
    /// Prepares the case's checkout before the stage runs.
    setup: Option<fn(&Path)>,
    /// Runs `ORACLE_PY` as this path instead of the stand-in.
    oracle: Option<&'static str>,
}

struct Ran {
    out: Output,
    dir: PathBuf,
}

impl Ran {
    fn stdout(&self) -> String {
        String::from_utf8_lossy(&self.out.stdout).into_owned()
    }
    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.out.stderr).into_owned()
    }
    fn code(&self) -> i32 {
        self.out.status.code().unwrap_or(-1)
    }
    fn reached_the_verdict(&self) -> bool {
        self.stdout().contains("STAGE_DONE")
    }
}

fn make_writable(dir: &Path) {
    if dir.exists() {
        let _ = Command::new("chmod").args(["-R", "u+rwX"]).arg(dir).status();
    }
}

fn run_stage(tag: &str, case: Case) -> Ran {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ci_local_python_stage")
        .join(tag);
    make_writable(&dir);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("scripts")).unwrap();
    for f in ["ci_local_python_stage.sh", "unittest_fixture_skips.sh"] {
        fs::copy(root().join("scripts").join(f), dir.join("scripts").join(f)).unwrap();
    }
    let canned = dir.join("canned.txt");
    fs::write(&canned, &case.output).unwrap();
    let marker = dir.join("python_ran");
    let fake = dir.join("fake_python.sh");
    fs::write(
        &fake,
        "#!/usr/bin/env bash\ntouch \"$FAKE_MARKER\"\ncat \"$FAKE_OUT\"\n\
         [[ -n \"${FAKE_SIG:-}\" ]] && kill -\"$FAKE_SIG\" $$\nexit \"${FAKE_RC:-0}\"\n",
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
    if let Some(setup) = case.setup {
        setup(&dir);
    }
    let driver = format!(
        "set -euo pipefail\ncd \"$ROOT\"\nSKIPS=()\nTEST_FAILED=\"$TF\"\n{}echo STAGE_DONE\n{}",
        fragment("python-stage"),
        fragment("verdict")
    );
    let mut cmd = Command::new("bash");
    cmd.arg("-c")
        .arg(&driver)
        .env("ROOT", &dir)
        .env("ORACLE_PY", case.oracle.map_or(fake.clone(), PathBuf::from))
        .env("TF", case.test_failed.to_string())
        .env("FAKE_OUT", &canned)
        .env("FAKE_MARKER", &marker)
        .env("FAKE_RC", case.rc.to_string())
        .env_remove("FAKE_SIG");
    if let Some(sig) = case.sig {
        cmd.env("FAKE_SIG", sig.to_string());
    }
    if case.unlocked {
        cmd.env_remove("MAC_HEAVY_LABEL");
    } else {
        cmd.env("MAC_HEAVY_LABEL", "test-lock");
    }
    let out = cmd.output().unwrap();
    make_writable(&dir);
    let ran = Ran { out, dir };
    eprintln!("=== {tag}: rc={}\n{}", ran.code(), text(&ran.out));
    ran
}

fn last_line(s: &str) -> &str {
    s.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("")
}

#[test]
fn the_wave_5_fixture_skips_fail_the_run_and_the_last_line_names_them() {
    let ran = run_stage(
        "fixture22",
        Case {
            output: W5.to_string(),
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    assert!(
        ran.reached_the_verdict(),
        "the stage ends in the verdict, not an early exit"
    );
    assert!(ran.stdout().contains("unexpected_skips=22"), "the skips are listed");
    let last = last_line(&ran.stderr()).to_string();
    assert!(last.starts_with("ci:local FAILED:"), "{last}");
    assert!(
        last.contains("stage exit 3") && last.contains("(not run: python/tests skipped tests"),
        "{last}"
    );
}

#[test]
fn checkpoint_skips_only_and_a_clean_run_pass() {
    let ran = run_stage(
        "ckpt_only",
        Case {
            output: w5_without_fixture_skips(),
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 0, "{}", text(&ran.out));
    assert_eq!(last_line(&ran.stdout()), "ci:local OK");
    let clean = "test_a (m.T.test_a) ... ok\n\nRan 1 test in 0.1s\n\nOK\n";
    let ran = run_stage(
        "clean",
        Case {
            output: clean.to_string(),
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 0);
    assert_eq!(last_line(&ran.stdout()), "ci:local OK");
}

#[test]
fn output_with_no_summary_is_not_examined_and_fails() {
    let partial: String = W5
        .lines()
        .filter(|l| !l.starts_with("Ran "))
        .map(|l| format!("{l}\n"))
        .collect();
    let ran = run_stage(
        "nosummary",
        Case {
            output: partial,
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    let last = last_line(&ran.stderr()).to_string();
    assert!(last.contains("stage exit 4") && last.contains("not examined"), "{last}");
    assert!(ran.stderr().contains("nothing was examined"));
}

#[test]
fn a_failing_python_ends_the_run_at_once_whatever_the_cargo_result_was() {
    let clean = "Ran 1 test in 0.1s\n\nFAILED (failures=1)\n";
    let ran = run_stage(
        "py_fail_tf0",
        Case {
            output: clean.into(),
            rc: 1,
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    assert!(!ran.reached_the_verdict());
    assert!(
        ran.stderr().contains("python/tests failed (python exit 1)"),
        "{}",
        ran.stderr()
    );
    let ran = run_stage(
        "py_fail_tf1",
        Case {
            output: clean.into(),
            rc: 1,
            test_failed: 1,
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    assert!(!ran.reached_the_verdict());
    assert!(
        ran.stderr().contains("one or more cargo test binaries failed"),
        "{}",
        ran.stderr()
    );
}

#[test]
fn a_python_killed_by_a_signal_or_missing_fails() {
    let clean = "test_a (m.T.test_a) ... ok\n\nRan 1 test in 0.1s\n\nOK\n";
    for sig in [9, 15] {
        let ran = run_stage(
            &format!("sig{sig}"),
            Case {
                output: clean.into(),
                sig: Some(sig),
                ..Case::default()
            },
        );
        assert_eq!(ran.code(), 1, "signal {sig}");
        assert!(!ran.reached_the_verdict());
        assert!(
            ran.stderr().contains(&format!("python exit {}", 128 + sig)),
            "{}",
            ran.stderr()
        );
    }
    let ran = run_stage(
        "nopython",
        Case {
            oracle: Some("/nonexistent/python"),
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    assert!(ran.stderr().contains("python exit 127"), "{}", ran.stderr());
}

fn read_only_dir(p: &Path) {
    fs::set_permissions(p, fs::Permissions::from_mode(0o555)).unwrap();
}

fn stale_clean_log(dir: &Path) {
    let target = dir.join("target");
    fs::create_dir_all(&target).unwrap();
    let log = target.join("python_gpu_unittest.log");
    fs::write(&log, "Ran 3 tests in 0.1s\n\nOK\n").unwrap();
    fs::set_permissions(&log, fs::Permissions::from_mode(0o444)).unwrap();
}

#[test]
fn an_unwritable_log_is_its_own_named_failure_with_or_without_a_stale_log() {
    // The adversary's reproductions: 22 fixture skips, and a stale clean log that must not be read.
    let skips = || W5.to_string();
    for (tag, setup) in [
        (
            "tee_ro_target",
            (|d: &Path| {
                fs::create_dir_all(d.join("target")).unwrap();
                read_only_dir(&d.join("target"));
            }) as fn(&Path),
        ),
        ("tee_ro_target_stale", |d: &Path| {
            stale_clean_log(d);
            read_only_dir(&d.join("target"));
        }),
        ("tee_log_is_dir", |d: &Path| {
            fs::create_dir_all(d.join("target/python_gpu_unittest.log")).unwrap();
        }),
    ] {
        let ran = run_stage(
            tag,
            Case {
                output: skips(),
                setup: Some(setup),
                ..Case::default()
            },
        );
        assert_eq!(ran.code(), 1, "{tag}");
        let last = last_line(&ran.stderr()).to_string();
        assert!(last.starts_with("ci:local FAILED:"), "{tag}: {last}");
        assert!(
            last.contains("output could not be kept") && last.contains("stage exit 2"),
            "{tag}: {last}"
        );
        assert!(
            !last.contains("stage exit 3"),
            "a tee failure is not blamed on skips: {last}"
        );
    }
}

#[test]
fn a_stale_read_only_log_in_a_writable_target_is_replaced_not_read() {
    let ran = run_stage(
        "stale_log_writable_target",
        Case {
            output: W5.to_string(),
            setup: Some(stale_clean_log),
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1, "{}", text(&ran.out));
    assert!(
        ran.stdout().contains("unexpected_skips=22"),
        "the fresh output was examined"
    );
    let log = fs::read_to_string(ran.dir.join("target/python_gpu_unittest.log")).unwrap();
    assert!(log.contains("Ran 173 tests"), "the log is this run's");
}

#[test]
fn outside_the_lock_python_is_not_run_and_the_skip_is_named() {
    let ran = run_stage(
        "unlocked",
        Case {
            output: W5.to_string(),
            unlocked: true,
            test_failed: 0,
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 0);
    assert!(
        !ran.dir.join("python_ran").exists(),
        "python must not run without the lock"
    );
    assert!(ran
        .stdout()
        .contains("SKIPPED: python/tests GPU tests (MAC_HEAVY_LABEL unset"));
    assert_eq!(
        last_line(&ran.stdout()),
        "ci:local OK WITH SKIPS (not run: python/tests GPU tests)"
    );
}

#[test]
fn a_cargo_failure_and_unexamined_skips_both_reach_the_last_line() {
    let ran = run_stage(
        "both",
        Case {
            output: W5.to_string(),
            test_failed: 1,
            ..Case::default()
        },
    );
    assert_eq!(ran.code(), 1);
    let last = last_line(&ran.stderr()).to_string();
    assert!(
        last.contains("cargo test binaries failed") && last.contains("stage exit 3"),
        "{last}"
    );
}

// ---------------------------------------------------------------------------------------------
// Where the stage sits.

/// The tiny fixtures are written by the release `cargo test` step, so the Python GPU stage runs
/// after it. Pre-fix the stage came first.
#[test]
fn the_python_gpu_stage_runs_after_the_release_tests() {
    let mut hits = CI_LOCAL
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with('#'))
        .filter(|(_, l)| {
            l.contains("unittest discover -s python/tests") || l.contains("scripts/ci_local_python_stage.sh")
        })
        .map(|(i, _)| i);
    let python = hits.next().expect("ci_local.sh runs the python stage");
    assert!(hits.next().is_none(), "the python stage is run once");
    let release = CI_LOCAL
        .lines()
        .position(|l| !l.trim_start().starts_with('#') && l.contains("cargo test --release --no-fail-fast"))
        .expect("ci_local.sh runs the release tests");
    assert!(
        release < python,
        "python stage (line {}) after release tests (line {})",
        python + 1,
        release + 1
    );
}
