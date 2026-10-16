//! `src/bin/release_check.rs` is the release gate's text checks (README,
//! CHANGELOG and Cargo.lock versions, and the CHANGELOG section the GitHub
//! Release publishes). It replaced four inline Python heredocs; each case here
//! feeds it a fixture, so a gate that stopped failing would show up.
//!
//! No GPU, no Metal toolchain: the binary reads three text files.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct Run {
    code: Option<i32>,
    out: String,
    err: String,
}

fn fixture(case: &str, name: &str, text: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("release_check")
        .join(case);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, text).unwrap();
    path
}

fn run(args: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_release_check"))
        .args(args)
        .output()
        .unwrap();
    Run {
        code: out.status.code(),
        out: String::from_utf8(out.stdout).unwrap(),
        err: String::from_utf8(out.stderr).unwrap(),
    }
}

fn passes(r: &Run) {
    assert_eq!(r.code, Some(0), "stderr: {}", r.err);
    assert_eq!(r.out, "");
    assert_eq!(r.err, "");
}

fn fails_with(r: &Run, message: &str) {
    assert_eq!(r.code, Some(1), "stdout: {}", r.out);
    assert_eq!(r.out, "");
    assert_eq!(r.err, format!("{message}\n"));
}

fn section(case: &str, changelog: &str, version: &str) -> Run {
    let path = fixture(case, "CHANGELOG.md", changelog);
    run(&["changelog-section", path.to_str().unwrap(), version])
}

const CHANGELOG: &str = "# Changelog\n\n## [Unreleased]\n\nnothing yet\n\n\
## [0.2.0] - 2026-10-01\n\n### Added\n\n- the new thing\n\n### Fixed\n\n- the old thing\n\n\
## [0.1.9] - 2026-09-01\n\n- an older note\n\n\
[0.2.0]: https://example.invalid/0.2.0\n[0.1.9]: https://example.invalid/0.1.9\n";

#[test]
fn the_readme_badge_must_state_the_cargo_version() {
    let p = fixture(
        "readme_ok",
        "README.md",
        "intro\n| **Status** | [`0.2.0`](https://x) | tail\n",
    );
    let p = p.to_str().unwrap();
    passes(&run(&["readme", p, "0.2.0"]));
    fails_with(
        &run(&["readme", p, "0.2.1"]),
        "error: README Status is '0.2.0' but Cargo.toml is '0.2.1'",
    );
    for (case, text) in [
        ("readme_none", "no badge here\n"),
        ("readme_unquoted", "**Status** | [0.2.0](x)\n"),
        ("readme_short", "**Status** | [`0.2`](x)\n"),
        ("readme_open", "**Status** | [`0.2.0\n"),
    ] {
        let p = fixture(case, "README.md", text);
        fails_with(
            &run(&["readme", p.to_str().unwrap(), "0.2.0"]),
            "error: README Status is '<missing>' but Cargo.toml is '0.2.0'",
        );
    }
}

#[test]
fn the_first_readme_badge_decides_and_a_malformed_one_is_skipped() {
    let p = fixture(
        "readme_first",
        "README.md",
        "**Status** | [`0.2.0.1`]\n**Status** | [`0.1.9`](u)\n**Status** | [`0.2.0`](u)\n",
    );
    // `0.2.0.1` is not a version triple followed by "`]"; the next badge is the first.
    fails_with(
        &run(&["readme", p.to_str().unwrap(), "0.2.0"]),
        "error: README Status is '0.1.9' but Cargo.toml is '0.2.0'",
    );
    let p = fixture("readme_crlf", "README.md", "a\r\n**Status** | [`0.2.0`](u)\r\n");
    passes(&run(&["readme", p.to_str().unwrap(), "0.2.0"]));
}

#[test]
fn the_newest_changelog_heading_must_state_the_cargo_version() {
    let p = fixture("changelog_ok", "CHANGELOG.md", CHANGELOG);
    let p = p.to_str().unwrap();
    // `[Unreleased]` is not a version; the first numbered heading is the newest.
    passes(&run(&["changelog", p, "0.2.0"]));
    fails_with(
        &run(&["changelog", p, "0.1.9"]),
        "error: newest CHANGELOG version is '0.2.0' but Cargo.toml is '0.1.9'",
    );
}

#[test]
fn a_changelog_without_a_numbered_heading_or_with_it_indented_is_missing() {
    for (case, text) in [
        ("changelog_empty", ""),
        ("changelog_unreleased", "## [Unreleased]\n\n- x\n"),
        ("changelog_indented", "  ## [0.2.0]\n### [0.2.0]\nmid ## [0.2.0]\n"),
        ("changelog_two_parts", "## [0.2]\n## [0.2.0.1]\n"),
    ] {
        let p = fixture(case, "CHANGELOG.md", text);
        fails_with(
            &run(&["changelog", p.to_str().unwrap(), "0.2.0"]),
            "error: newest CHANGELOG version is '<missing>' but Cargo.toml is '0.2.0'",
        );
    }
}

#[test]
fn the_lock_file_must_agree_and_an_absent_entry_does_not_fail() {
    let lock = "[[package]]\nname = \"objc2\"\nversion = \"0.6.1\"\n\n\
[[package]]\nname = \"tessl-extra\"\nversion = \"9.9.9\"\n\n\
[[package]]\nname = \"tessl\"\nversion = \"0.2.0\"\ndependencies = [\"objc2\"]\n";
    let p = fixture("lock", "Cargo.lock", lock);
    let p = p.to_str().unwrap();
    passes(&run(&["lock", p, "tessl", "0.2.0"]));
    fails_with(
        &run(&["lock", p, "tessl", "0.2.1"]),
        "error: Cargo.lock has tessl 0.2.0 but Cargo.toml is 0.2.1",
    );
    // No entry for the crate, or a `version` that is not on the very next line,
    // is not a stated version: nothing to compare, as the shell script had it.
    passes(&run(&["lock", p, "absent", "0.2.0"]));
    let gap = fixture(
        "lock_gap",
        "Cargo.lock",
        "[[package]]\nname = \"tessl\"\nsource = \"x\"\nversion = \"0.0.1\"\n",
    );
    passes(&run(&["lock", gap.to_str().unwrap(), "tessl", "0.2.0"]));
    // The name is matched whole: `tessl` does not read `tessl-extra`'s version.
    let prefix = fixture(
        "lock_prefix",
        "Cargo.lock",
        "[[package]]\nname = \"tessl-extra\"\nversion = \"9.9.9\"\n",
    );
    passes(&run(&["lock", prefix.to_str().unwrap(), "tessl", "0.2.0"]));
}

#[test]
fn the_section_is_the_body_between_its_heading_and_the_next_one() {
    let r = section("section_middle", CHANGELOG, "0.2.0");
    assert_eq!(r.code, Some(0), "{}", r.err);
    assert_eq!(r.out, "### Added\n\n- the new thing\n\n### Fixed\n\n- the old thing\n");
    assert_eq!(r.err, "");
}

#[test]
fn the_last_section_runs_to_the_end_of_the_file_and_is_trimmed() {
    let r = section("section_last", CHANGELOG, "0.1.9");
    assert_eq!(r.code, Some(0));
    // The footer links are part of the last section's body, as before.
    assert_eq!(
        r.out,
        "- an older note\n\n[0.2.0]: https://example.invalid/0.2.0\n[0.1.9]: https://example.invalid/0.1.9\n"
    );
    let r = section("section_trim", "## [1.0.0]\n\n\t \n body \n\n\n", "1.0.0");
    assert_eq!(r.out, "body\n");
    let r = section("section_empty", "## [1.0.0]\n## [0.9.0]\n", "0.9.0");
    assert_eq!(r.code, Some(0));
    assert_eq!(r.out, "\n");
}

#[test]
fn a_missing_section_is_a_failure_naming_the_version() {
    for (case, text, version) in [
        ("missing_plain", CHANGELOG, "0.3.0"),
        // `]` ends the version: 0.2.0 is not 0.2.01.
        ("missing_prefix", "## [0.2.01]\n\nbody\n", "0.2.0"),
        // Dots are literal, not wildcards.
        ("missing_dots", "## [0x2x0]\n\nbody\n", "0.2.0"),
        // The heading line must end in a newline.
        ("missing_no_newline", "## [0.2.0]", "0.2.0"),
        ("missing_empty_file", "", "0.2.0"),
    ] {
        let r = section(case, text, version);
        assert_eq!(r.code, Some(1), "{case}: {}", r.out);
        assert_eq!(r.out, "", "{case}");
        assert_eq!(r.err, format!("CHANGELOG section for {version} not found\n"), "{case}");
    }
}

#[test]
fn a_heading_with_no_newline_at_the_end_does_not_hide_an_earlier_one() {
    // The first occurrence has a newline after it, the second does not: the
    // first decides, and the second only ends its body.
    let r = section("two_headings", "intro\n## [1.0.0] a\nbody\n## [1.0.0]", "1.0.0");
    assert_eq!(r.code, Some(0));
    assert_eq!(r.out, "body\n");
}

/// Two headings on consecutive lines (no blank between) make the first
/// section swallow the second: the lookahead needs a `\n` in front of `## [`
/// and the first section's own heading line has already consumed it. This is
/// what the Python extractor did; the release workflow's CHANGELOG has a blank
/// line between sections, so it has not mattered. It is pinned here so a change
/// to it is a decision rather than a side effect.
#[test]
fn consecutive_headings_run_together_as_the_python_extractor_did() {
    let r = section(
        "consecutive",
        "## [2.0.0]\n## [1.0.0]\nolder\n\n## [0.9.0]\nx\n",
        "2.0.0",
    );
    assert_eq!(r.code, Some(0));
    assert_eq!(r.out, "## [1.0.0]\nolder\n");
}

#[test]
fn a_heading_is_found_anywhere_on_a_line_as_the_python_extractor_did() {
    let r = section("anywhere", "### [3.0.0] sub\nsub body\n\n## [2.0.0]\nx\n", "3.0.0");
    assert_eq!(r.code, Some(0));
    assert_eq!(r.out, "sub body\n");
}

#[test]
fn windows_line_endings_are_read_as_newlines() {
    let r = section(
        "crlf",
        "## [1.0.0] - d\r\n\r\n- one\r\n- two\r\n\r\n## [0.9.0]\r\nx\r\n",
        "1.0.0",
    );
    assert_eq!(r.code, Some(0));
    assert_eq!(r.out, "- one\n- two\n");
}

#[test]
fn unicode_white_space_is_trimmed_like_python_strip() {
    let r = section("unicode_space", "## [1.0.0]\n\u{2003}\u{1c}body\u{85}\u{a0}\n", "1.0.0");
    assert_eq!(r.out, "body\n");
}

#[test]
fn bad_invocations_and_unreadable_files_fail_loudly() {
    let r = run(&[]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.starts_with("usage: release_check"), "{}", r.err);
    assert_eq!(run(&["readme", "only-one-arg"]).code, Some(1));
    assert_eq!(run(&["lock", "Cargo.lock", "tessl"]).code, Some(1));
    assert_eq!(run(&["frobnicate", "a", "b"]).code, Some(1));

    let missing = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("release_check/does-not-exist.md");
    let r = run(&["changelog-section", missing.to_str().unwrap(), "1.0.0"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.starts_with("error: cannot read "), "{}", r.err);

    let dir = fixture("not_utf8", "CHANGELOG.md", "");
    fs::write(&dir, b"## [1.0.0]\n\xff\xfe\n").unwrap();
    let r = run(&["changelog", dir.to_str().unwrap(), "1.0.0"]);
    assert_eq!(r.code, Some(1));
    assert!(r.err.contains("is not UTF-8"), "{}", r.err);
}
