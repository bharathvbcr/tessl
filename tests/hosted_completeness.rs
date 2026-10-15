//! ADVERSARIAL REGRESSION TEST (wave-7 hosted-CI tester). This test FAILS on
//! the current manifest: it lists every omission.
//!
//! `scripts/hosted_tests.txt` is a hand-kept list, and nothing else notices a
//! device-free test that is not on it: a new test in a mixed file, a renamed
//! one, or a whole `src/bin` unit-test binary. Such a test just never runs on
//! hosted CI. This check derives the truth the way the manifest was derived:
//! list every test of every test binary cargo builds (integration tests, the
//! library, and the `src/bin` unit tests), run the ones the manifest does not
//! select under the same IOKit-denying sandbox, and require that none of them
//! passes. A test that passes with the device hidden is device-free and
//! belongs on the manifest.
//!
//! It runs nested `cargo test --no-run` (already built, so a few seconds) and
//! every binary once under the sandbox (about a minute), so it must not be
//! listed in the manifest itself: `sandbox-exec` does not nest.
#![cfg(target_os = "macos")]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

const MANIFEST: &str = include_str!("../scripts/hosted_tests.txt");
const PROFILE: &str = "(version 1)(allow default)(deny iokit-open)";

/// `(kind, target name, executable)`; kind is "test", "bin" or "lib".
fn test_binaries() -> Vec<(String, String, PathBuf)> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let profile = if cfg!(debug_assertions) {
        vec![]
    } else {
        vec!["--release"]
    };
    let out = Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["test", "--no-run", "--message-format=json"])
        .args(&profile)
        .output()
        .expect("run cargo");
    assert!(out.status.success(), "cargo test --no-run failed");
    let mut found = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if !line.contains("\"compiler-artifact\"") || !line.contains("\"test\":true") {
            continue;
        }
        let field = |key: &str| -> Option<String> {
            let k = format!("\"{key}\":\"");
            let at = line.find(&k)? + k.len();
            Some(line[at..at + line[at..].find('"')?].to_string())
        };
        let (Some(exe), Some(name)) = (field("executable"), field("name")) else {
            continue;
        };
        let kind = if line.contains("\"kind\":[\"test\"]") {
            "test"
        } else if line.contains("\"kind\":[\"bin\"]") {
            "bin"
        } else {
            "lib"
        };
        // Never run this binary from itself: it would recurse without end.
        if name == "hosted_completeness" {
            continue;
        }
        found.push((kind.to_string(), name, PathBuf::from(exe)));
    }
    assert!(found.len() > 50, "found only {} test binaries", found.len());
    found
}

fn run(exe: &PathBuf, args: &[&str], sandboxed: bool) -> String {
    let mut c = if sandboxed {
        let mut c = Command::new("sandbox-exec");
        c.args(["-p", PROFILE]).arg(exe);
        c
    } else {
        Command::new(exe)
    };
    let out = c
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("TESSL_HOSTED_NO_DEVICE", "1")
        .args(args)
        .output()
        .expect("run test binary");
    // stdout only: a should_panic test's panic message goes to stderr and
    // would otherwise split a `test x ... ok` line.
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `(kind, target, test)` triples the manifest selects.
fn selected(all: &BTreeSet<(String, String, String)>) -> BTreeSet<(String, String, String)> {
    let mut sel = BTreeSet::new();
    for raw in MANIFEST.lines() {
        let f: Vec<&str> = raw.split('#').next().unwrap().split_whitespace().collect();
        match f.as_slice() {
            ["bin", n] => sel.extend(all.iter().filter(|(k, t, _)| k == "test" && t == n).cloned()),
            ["lib", prefix] => sel.extend(
                all.iter()
                    .filter(|(k, _, t)| k == "lib" && t.starts_with(prefix))
                    .cloned(),
            ),
            ["exact", target, names @ ..] => sel.extend(
                all.iter()
                    .filter(|(k, t, n)| {
                        names.contains(&n.as_str())
                            && if *target == "lib" {
                                k == "lib"
                            } else {
                                k == "test" && t == target
                            }
                    })
                    .cloned(),
            ),
            _ => {}
        }
    }
    sel
}

#[test]
fn every_device_free_test_is_on_the_hosted_manifest() {
    let mut all = BTreeSet::new();
    let mut passing = BTreeSet::new();
    for (kind, name, exe) in test_binaries() {
        for line in run(&exe, &["--list"], false).lines() {
            if let Some(t) = line.strip_suffix(": test") {
                all.insert((kind.clone(), name.clone(), t.to_string()));
            }
        }
        for line in run(&exe, &["--test-threads=4"], true).lines() {
            let ok = line.strip_prefix("test ").and_then(|l| l.strip_suffix(" ... ok"));
            // A #[should_panic] test prints `name - should panic ... ok`.
            if let Some(t) = ok.map(|t| t.strip_suffix(" - should panic").unwrap_or(t)) {
                passing.insert((kind.clone(), name.clone(), t.to_string()));
            }
        }
    }
    let sel = selected(&all);
    let omitted: Vec<_> = passing.difference(&sel).collect();
    assert!(
        omitted.is_empty(),
        "{} test(s) pass with the Metal device hidden but are not selected by scripts/hosted_tests.txt, \
         so hosted CI never runs them:\n{}",
        omitted.len(),
        omitted
            .iter()
            .map(|(k, t, n)| format!("  {k} {t}: {n}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
