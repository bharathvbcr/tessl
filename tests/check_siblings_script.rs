//! `scripts/check_siblings.sh` holds the sibling layout to resolving. This
//! holds the script to failing.
//!
//! It exists because the script shipped once with a coverage check that could
//! not fire: the guard counted every `path` containing `..`, which includes the
//! in-repository `go/engine -> ../..`, so a parser blind to BOTH ojas
//! dependencies still reported success. A guard nobody has watched fail is not
//! a guard, and reading the script is not watching it fail.
//!
//! Each case builds a throwaway tree, runs the real script in it, and asserts
//! the exit status and the words the author has to read. std-only, no device,
//! no cargo in the fixtures that must not reach cargo: the three layout cases
//! fail before step 4, which is the only step that runs cargo.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script() -> PathBuf {
    repo().join("scripts/check_siblings.sh")
}

/// A fixture tree: a primary git checkout holding the real script and whatever
/// manifests the case needs. Under the target directory, so a case leaves
/// nothing in the working tree.
fn fixture(name: &str, manifests: &[(&str, &str)], git_init: bool) -> PathBuf {
    let dir = repo().join("target/check_siblings_fixtures").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("scripts")).expect("create fixture");
    fs::copy(script(), dir.join("scripts/check_siblings.sh")).expect("copy script");
    for (path, body) in manifests {
        let p = dir.join(path);
        fs::create_dir_all(p.parent().unwrap()).expect("create manifest dir");
        fs::write(&p, body).expect("write manifest");
    }
    if git_init {
        run(&dir, "git", &["init", "-q"]);
        run(&dir, "git", &["add", "-A"]);
    }
    dir
}

fn run(dir: &Path, program: &str, args: &[&str]) -> Output {
    Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn {program}: {e}"))
}

struct Ran {
    code: i32,
    text: String,
}

fn check_siblings(dir: &Path, args: &[&str]) -> Ran {
    let out = Command::new(dir.join("scripts/check_siblings.sh"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("run check_siblings.sh");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ran {
        code: out.status.code().unwrap_or(-1),
        text,
    }
}

const ROOT_WITH_SIBLING: &str = r#"[package]
name = "fake"
version = "0.0.0"

[dependencies]
ojas-core = { version = "0.1.0", path = "../ojas/ojas-core" }
"#;

/// The real repository passes. Not a tautology: it is the only case that proves
/// the parser still matches the manifests it is pointed at, which is what every
/// failing case below is defined against.
#[test]
fn the_real_checkout_passes() {
    let out = Command::new(script())
        .current_dir(repo())
        .output()
        .expect("run check_siblings.sh");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the real checkout must pass: {}{}",
        text,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("of them leaving this repository"),
        "the summary must report the sibling count: {text}"
    );
}

/// An absent sibling is a layout diagnosis, and must be reported as one rather
/// than as cargo's package error.
#[test]
fn an_absent_sibling_fails_and_names_the_layout() {
    let dir = fixture("absent-sibling", &[("Cargo.toml", ROOT_WITH_SIBLING)], true);
    let ran = check_siblings(&dir, &[]);
    assert_eq!(ran.code, 1, "must fail: {}", ran.text);
    assert!(ran.text.contains("MISSING"), "{}", ran.text);
    assert!(
        ran.text.contains("do not resolve"),
        "must name the unresolved dependencies: {}",
        ran.text
    );
    assert!(
        ran.text.contains("<parent>/ojas"),
        "must print the layout it needs: {}",
        ran.text
    );
}

/// The regression this test file was written for. A parser that matches no
/// out-of-repository path must fail, even though an in-repository `../..`
/// resolves: counting that one is how the guard came to be unable to fire.
#[test]
fn a_parser_blind_to_the_siblings_fails_even_though_an_in_repo_path_resolves() {
    let dir = fixture(
        "parser-blind",
        &[
            // Single-quoted TOML: valid, and invisible to a double-quote-only
            // parser. This is the shape that demonstrated the hole.
            (
                "Cargo.toml",
                "[package]\nname = \"fake\"\nversion = \"0.0.0\"\n\n[dependencies]\nojas-core = { path = '../ojas/ojas-core' }\n",
            ),
            // An in-repository `..` dependency that always resolves.
            (
                "go/engine/Cargo.toml",
                "[package]\nname = \"engine\"\nversion = \"0.0.0\"\n\n[dependencies]\nfake = { path = \"../..\" }\n",
            ),
        ],
        true,
    );
    // With the sibling absent AND the parser seeing it, this fails at MISSING.
    // The point of the case is that it must not pass when the parser does not
    // see it, so assert on the guard's own words.
    let ran = check_siblings(&dir, &[]);
    assert_eq!(ran.code, 1, "must fail: {}", ran.text);
    assert!(
        ran.text.contains("MISSING") || ran.text.contains("not one of them leaves this"),
        "must either see the sibling or refuse to report success: {}",
        ran.text
    );
    assert!(
        !ran.text.contains("==> siblings ok"),
        "must never reach the success line: {}",
        ran.text
    );
}

/// Both TOML string forms are parsed. The sibling exists here, so a parser that
/// sees the single-quoted path reaches the resolve steps; one that does not
/// trips the coverage guard. Either way it must not report success.
#[test]
fn a_single_quoted_path_is_parsed() {
    let dir = fixture(
        "single-quoted",
        &[(
            "Cargo.toml",
            "[package]\nname = \"fake\"\nversion = \"0.0.0\"\n\n[dependencies]\nojas-core = { path = '../ojas/ojas-core' }\n",
        )],
        true,
    );
    let ran = check_siblings(&dir, &[]);
    assert!(
        ran.text.contains("MISSING   Cargo.toml -> ../ojas/ojas-core"),
        "the single-quoted path must be seen and reported: {}",
        ran.text
    );
}

/// No manifests at all is a broken walk, not a pass.
#[test]
fn no_tracked_manifests_fails() {
    let dir = fixture("no-manifests", &[], true);
    let ran = check_siblings(&dir, &[]);
    assert_eq!(ran.code, 1, "must fail: {}", ran.text);
    assert!(
        ran.text.contains("found no Cargo.toml"),
        "must name the broken walk: {}",
        ran.text
    );
}

/// A bad flag is a usage error (2), distinct from a failed check (1), so a
/// mistyped invocation cannot read as a layout verdict.
#[test]
fn a_bad_flag_is_a_usage_error_not_a_verdict() {
    let dir = fixture("bad-flag", &[("Cargo.toml", ROOT_WITH_SIBLING)], true);
    let ran = check_siblings(&dir, &["--bogus"]);
    assert_eq!(ran.code, 2, "usage errors exit 2: {}", ran.text);
    assert!(ran.text.contains("usage:"), "{}", ran.text);
}

/// The script's own resolve set must match the lockfiles the repository tracks,
/// which is what stops the named subset in step 4 from going stale. Read from
/// the script's text rather than re-run, so this is a source check with no
/// cargo in it.
#[test]
fn the_expected_lockfile_set_is_the_tracked_set() {
    let text = fs::read_to_string(script()).expect("read script");
    let expected: Vec<&str> = text
        .lines()
        .find_map(|l| l.strip_prefix("EXPECTED_LOCKS=\""))
        .expect("EXPECTED_LOCKS in the script")
        .trim_end_matches('"')
        .split_whitespace()
        .collect();
    let out = run(&repo(), "git", &["ls-files", "*Cargo.lock"]);
    let mut tracked: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();
    tracked.sort();
    let mut want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(
        tracked, want,
        "EXPECTED_LOCKS and the tracked lockfiles have diverged; see step 5 in the script"
    );

    let resolve: Vec<&str> = text
        .lines()
        .find_map(|l| l.strip_prefix("RESOLVE_LOCKS=\""))
        .expect("RESOLVE_LOCKS in the script")
        .trim_end_matches('"')
        .split_whitespace()
        .collect();
    for lock in &resolve {
        assert!(
            want.iter().any(|w| w == lock),
            "{lock} is resolved but not accounted for in EXPECTED_LOCKS"
        );
        assert!(
            repo().join(lock).is_file(),
            "{lock} is in RESOLVE_LOCKS but not on disk"
        );
    }
    assert!(
        resolve.contains(&"go/engine/Cargo.lock"),
        "go/engine has its own [workspace] and the go job builds it --locked; \
         it must stay in the resolve set"
    );
}
