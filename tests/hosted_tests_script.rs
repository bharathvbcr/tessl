//! `scripts/hosted_tests.sh` is what turns "cargo test matched nothing and exited
//! 0" into a red hosted step. These tests drive it with a fake cargo (no GPU, no
//! Metal toolchain, no sandbox) and pin that every way an invocation can run
//! nothing, or the wrong number of tests, fails the script.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/hosted_tests.sh")
}

fn scratch(tag: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("hosted_tests_script-{tag}"));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A cargo that prints `result` for every `test` invocation that is not
/// `--no-run`, and exits `code`.
fn fake_cargo(dir: &Path, result: &str, code: i32) -> PathBuf {
    let p = dir.join("fake-cargo");
    let body = format!(
        "#!/bin/sh\ncase \" $* \" in *' --no-run '*) exit 0;; esac\n\
         echo 'running tests'\n{result}\nexit {code}\n"
    );
    fs::write(&p, body).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn run(dir: &Path, cargo: &Path, manifest: &str) -> Output {
    run_with(dir, cargo, manifest, &[])
}

fn run_with(dir: &Path, cargo: &Path, manifest: &str, flags: &[&str]) -> Output {
    let m = dir.join("manifest.txt");
    fs::write(&m, manifest).unwrap();
    Command::new("bash")
        .arg(script())
        .args(flags)
        .arg(&m)
        .env("TESSL_HOSTED_CARGO", cargo)
        .env("TESSL_HOSTED_WRAP", "env")
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn ok_line(n: usize) -> String {
    format!("echo 'test result: ok. {n} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'")
}

#[test]
fn a_whole_binary_that_ran_tests_passes() {
    let d = scratch("bin_ok");
    let o = run(&d, &fake_cargo(&d, &ok_line(3), 0), "bin gdn_gates 3\n");
    assert!(o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("OK (1 invocations"), "{}", text(&o));
}

/// The hazard this script exists for: a selector that matches nothing exits 0.
#[test]
fn a_filter_that_matches_nothing_fails() {
    let d = scratch("zero");
    for entry in ["bin gdn_gates 3\n", "lib json::tests:: 3\n", "bins 3\n"] {
        let o = run(&d, &fake_cargo(&d, &ok_line(0), 0), entry);
        assert!(!o.status.success(), "{entry:?} passed with 0 tests run:\n{}", text(&o));
        assert!(text(&o).contains("matched nothing"), "{}", text(&o));
    }
}

#[test]
fn an_exact_name_that_matches_nothing_fails_even_when_another_ran() {
    let d = scratch("partial");
    let o = run(
        &d,
        &fake_cargo(&d, &ok_line(1), 0),
        "exact attention a_real_test a_renamed_test\n",
    );
    assert!(!o.status.success(), "one of two names ran and it passed:\n{}", text(&o));
    assert!(text(&o).contains("expected exactly 2"), "{}", text(&o));
}

#[test]
fn every_exact_name_running_passes() {
    let d = scratch("exact_ok");
    let o = run(&d, &fake_cargo(&d, &ok_line(2), 0), "exact lib a::b b::c\n");
    assert!(o.status.success(), "{}", text(&o));
}

#[test]
fn a_run_with_no_result_line_is_not_a_pass() {
    let d = scratch("no_result");
    let o = run(&d, &fake_cargo(&d, "echo 'nothing to see'", 0), "bin gdn_gates 3\n");
    assert!(!o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("nothing ran"), "{}", text(&o));
}

#[test]
fn a_failing_cargo_fails() {
    let d = scratch("cargo_fails");
    let o = run(&d, &fake_cargo(&d, &ok_line(3), 101), "bin gdn_gates 3\n");
    assert!(!o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("cargo test failed"), "{}", text(&o));
}

#[test]
fn a_malformed_or_empty_manifest_fails() {
    let d = scratch("malformed");
    let cargo = fake_cargo(&d, &ok_line(3), 0);
    for bad in [
        "frobnicate x\n",
        "bin\n",
        "bin a\n",
        "bin a b\n",
        "bin a 0\n",
        "bins\n",
        "lib a b\n",
        "outside a\n",
        "exact onlytarget\n",
        "ignored a\n",
        "# only a comment\n\n",
        "",
    ] {
        let o = run(&d, &cargo, bad);
        assert!(!o.status.success(), "{bad:?} was accepted:\n{}", text(&o));
    }
}

/// COUNT is exact in both directions: a deleted test and an unlisted new one
/// each fail until the manifest is updated.
#[test]
fn a_count_that_moved_fails_either_way() {
    let d = scratch("count_moved");
    for (ran, entry) in [
        (2, "bin gdn_gates 3\n"),
        (4, "bin gdn_gates 3\n"),
        (9, "bins 10\n"),
        (11, "lib x:: 10\n"),
    ] {
        let o = run(&d, &fake_cargo(&d, &ok_line(ran), 0), entry);
        assert!(
            !o.status.success(),
            "{ran} ran against {entry:?} and passed:\n{}",
            text(&o)
        );
        assert!(text(&o).contains("expected exactly"), "{}", text(&o));
    }
    let o = run(&d, &fake_cargo(&d, &ok_line(10), 0), "bins 10\n");
    assert!(o.status.success(), "{}", text(&o));
}

/// `--outside` runs only the `outside` entries (the completeness check, which
/// launches its own sandboxed children), and the default run skips them.
#[test]
fn outside_entries_run_only_with_the_outside_flag() {
    let d = scratch("outside");
    let manifest = "bin gdn_gates 3\noutside hosted_completeness the_check\n";
    let o = run_with(&d, &fake_cargo(&d, &ok_line(1), 0), manifest, &["--outside"]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("outside hosted_completeness"), "{}", text(&o));
    assert!(!text(&o).contains("bin gdn_gates"), "{}", text(&o));
    let o = run(&d, &fake_cargo(&d, &ok_line(3), 0), manifest);
    assert!(o.status.success(), "{}", text(&o));
    assert!(!text(&o).contains("outside hosted_completeness"), "{}", text(&o));
    let o = run_with(&d, &fake_cargo(&d, &ok_line(3), 0), "bin gdn_gates 3\n", &["--outside"]);
    assert!(
        !o.status.success(),
        "--outside with no outside entry passed:\n{}",
        text(&o)
    );
}
