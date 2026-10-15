//! Completeness of `scripts/hosted_tests.txt`: every test that passes with the
//! Metal device hidden must be selected by it.
//!
//! The manifest is a hand-kept list, and nothing else notices a device-free test
//! that is not on it: a new test in a mixed file, a renamed one, or a `src/bin`
//! unit-test binary. Such a test just never runs on hosted CI. This check
//! derives the truth the way the manifest was derived: list every test of every
//! test binary cargo builds (integration tests, the library, the `src/bin` unit
//! tests), run the ones the manifest does not select under the IOKit-denying
//! sandbox, and require that none of them passes.
//!
//! It is `#[ignore]`d and run explicitly, as the `outside` entry of the manifest
//! (`scripts/hosted_tests.sh --outside`, after the sandboxed step, in ci_local
//! and in hosted CI): it starts sandboxed children itself and `sandbox-exec`
//! does not nest, so it cannot be one of the sandboxed entries. It runs only the
//! unselected tests, by exact name, reusing the binaries cargo already built.
//! It never runs its own binary, and `TESSL_COMPLETENESS_CHILD` (set on every
//! child) makes a second level of it refuse to start.
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
        if !line.contains("\"compiler-artifact\"") {
            continue;
        }
        // The test-harness build of a target, not the ordinary build of a bin
        // (whose `target.test` is also true: running that executes the bench).
        let in_test_profile = line.find("\"profile\":{").is_some_and(|at| {
            line[at..]
                .split('}')
                .next()
                .is_some_and(|p| p.contains("\"test\":true"))
        });
        if !in_test_profile {
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
        .env("TESSL_COMPLETENESS_CHILD", "1")
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
            ["bin", n, _count] => sel.extend(all.iter().filter(|(k, t, _)| k == "test" && t == n).cloned()),
            ["bins", _count] => sel.extend(all.iter().filter(|(k, _, _)| k == "bin").cloned()),
            ["lib", prefix, _count] => sel.extend(
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
#[ignore = "run by `scripts/hosted_tests.sh --outside` (the manifest's `outside` entry): it launches sandboxed children, and sandbox-exec does not nest"]
fn every_device_free_test_is_on_the_hosted_manifest() {
    assert!(
        std::env::var_os("TESSL_COMPLETENESS_CHILD").is_none(),
        "the completeness check started inside one of its own children; refusing to recurse"
    );
    let started = std::time::Instant::now();
    let mut all = BTreeSet::new();
    let mut by_binary = Vec::new();
    for (kind, name, exe) in test_binaries() {
        let mut names = Vec::new();
        let t0 = std::time::Instant::now();
        let listing = run(&exe, &["--list"], false);
        if t0.elapsed().as_secs() >= 2 {
            eprintln!("completeness: listing {name} took {:?}", t0.elapsed());
        }
        for line in listing.lines() {
            if let Some(t) = line.strip_suffix(": test") {
                all.insert((kind.clone(), name.clone(), t.to_string()));
                names.push(t.to_string());
            }
        }
        by_binary.push((kind, name, exe, names));
    }
    eprintln!("completeness: listed {} tests in {:?}", all.len(), started.elapsed());
    let sel = selected(&all);
    assert!(
        !sel.is_empty(),
        "the manifest selects nothing; the parser is out of sync with its syntax"
    );
    let mut passing = BTreeSet::new();
    for (kind, name, exe, names) in &by_binary {
        let unselected: Vec<&String> = names
            .iter()
            .filter(|n| !sel.contains(&(kind.clone(), name.clone(), (*n).clone())))
            .collect();
        for chunk in unselected.chunks(100) {
            let mut args = vec!["--test-threads=4", "--exact"];
            args.extend(chunk.iter().map(|n| n.as_str()));
            for line in run(exe, &args, true).lines() {
                let ok = line.strip_prefix("test ").and_then(|l| l.strip_suffix(" ... ok"));
                // A #[should_panic] test prints `name - should panic ... ok`.
                if let Some(t) = ok.map(|t| t.strip_suffix(" - should panic").unwrap_or(t)) {
                    passing.insert((kind.clone(), name.clone(), t.to_string()));
                }
            }
        }
    }
    eprintln!(
        "completeness: ran the unselected tests, done at {:?}",
        started.elapsed()
    );
    let omitted: Vec<_> = passing.iter().collect();
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
