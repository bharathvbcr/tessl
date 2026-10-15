//! The hosted CI job runs `scripts/hosted_tests.sh` over `scripts/hosted_tests.txt`
//! on a runner with no usable Metal device. These tests keep that list honest
//! from the source tree, with no device of their own:
//!
//! * every `exact` and `ignored` name is a real `#[test]` in the file it is
//!   listed under, and every `bin` / `lib` entry names something that exists,
//!   so a rename shows up here, on a laptop, not only as a red hosted step;
//! * every `exact` test is device-free by source: neither its body nor any
//!   function it calls in the same file mentions a runtime or the GPU helpers.
//!   The sandbox the hosted job runs under is the real enforcement; this is
//!   the early, local half of it;
//! * the scanner flags a planted device test, so a clean report means
//!   something.
//!
//! `hosted_selection_has_no_device` is the one `#[ignore]`d test: the hosted
//! script runs it first, under the sandbox, to prove the device really is hidden.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST: &str = include_str!("../scripts/hosted_tests.txt");

/// Source text that means "this opens, or needs, a device".
const DEVICE_MARKERS: &[&str] = &[
    "with_gpu",
    "with_two_gpus",
    "gpu_lock",
    "in_traced_child",
    "GpuRuntime",
    "new_inference",
    "Gemm::new",
    "MTLCreateSystemDefaultDevice",
];

#[derive(Debug, PartialEq)]
enum Entry {
    Bin(String),
    Bins,
    Lib(String),
    Outside { bin: String, name: String },
    Exact { target: String, names: Vec<String> },
    Ignored { bin: String, name: String },
}

fn parse(manifest: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for (i, raw) in manifest.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        let f: Vec<&str> = line.split_whitespace().collect();
        let Some(kind) = f.first() else { continue };
        let at = i + 1;
        out.push(match (*kind, f.len()) {
            ("bin", 3) if f[2].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Bin(f[1].into()),
            ("bins", 2) if f[1].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Bins,
            ("lib", 3) if f[2].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Lib(f[1].into()),
            ("outside", 3) => Entry::Outside {
                bin: f[1].into(),
                name: f[2].into(),
            },
            ("exact", n) if n >= 3 => Entry::Exact {
                target: f[1].into(),
                names: f[2..].iter().map(|s| s.to_string()).collect(),
            },
            ("ignored", 3) => Entry::Ignored {
                bin: f[1].into(),
                name: f[2].into(),
            },
            _ => panic!("scripts/hosted_tests.txt:{at}: malformed entry {raw:?}"),
        });
    }
    out
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `{...}` body of `fn name(`, or `None` if there is no such function.
fn fn_body<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("fn {name}(");
    let at = src.match_indices(&key).map(|(i, _)| i).find(|&i| {
        src[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'))
    })?;
    let open = at + src[at..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open..open + i + 1]);
                }
            }
            _ => {}
        }
    }
    // Braces inside string or char literals can leave the count open. Fall back
    // to the text up to the next item: a longer span can only add markers, so
    // the scan stays conservative.
    let rest = &src[at..];
    let end = rest[1..].find("\n#[test]").map_or(rest.len(), |i| i + 1);
    Some(&rest[..end])
}

/// Every function name defined in `src`.
fn defined_fns(src: &str) -> HashSet<String> {
    src.match_indices("fn ")
        .filter_map(|(i, _)| {
            let name: String = src[i + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// A device marker reachable from `test` through functions defined in `src`.
fn device_marker(src: &str, test: &str) -> Result<Option<String>, String> {
    let src = strip_comments(src);
    let known = defined_fns(&src);
    let mut seen = HashSet::new();
    let mut work = vec![test.to_string()];
    while let Some(name) = work.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(body) = fn_body(&src, &name) else {
            if name == test {
                return Err(format!("no `fn {test}(` found"));
            }
            continue;
        };
        for m in DEVICE_MARKERS {
            if body.contains(m) {
                return Ok(Some(format!("{name} mentions {m}")));
            }
        }
        for callee in &known {
            if body.contains(&format!("{callee}(")) {
                work.push(callee.clone());
            }
        }
    }
    Ok(None)
}

/// The file a manifest target lives in: `tests/BIN.rs`, or for `lib` the
/// source file of a `module::path::name`.
fn source_of(target: &str, name: &str) -> PathBuf {
    if target != "lib" {
        return root().join("tests").join(format!("{target}.rs"));
    }
    let mut parts: Vec<&str> = name.split("::").collect();
    parts.pop();
    parts.retain(|p| !p.ends_with("tests") && *p != "tests");
    let rel = parts.join("/");
    let flat = root().join("src").join(format!("{rel}.rs"));
    if flat.exists() {
        flat
    } else {
        root().join("src").join(&rel).join("mod.rs")
    }
}

fn test_attribute_precedes(src: &str, name: &str) -> bool {
    let key = format!("fn {name}(");
    let Some(at) = src.find(&key) else { return false };
    let head: Vec<&str> = src[..at].lines().rev().take(6).collect();
    head.iter().any(|l| l.trim_start().starts_with("#[test]"))
}

fn every_name() -> Vec<(String, String, bool)> {
    let mut out = Vec::new();
    for e in parse(MANIFEST) {
        match e {
            Entry::Exact { target, names } => {
                for n in names {
                    out.push((target.clone(), n, false));
                }
            }
            Entry::Ignored { bin, name } | Entry::Outside { bin, name } => out.push((bin, name, true)),
            Entry::Bin(_) | Entry::Bins | Entry::Lib(_) => {}
        }
    }
    out
}

#[test]
fn the_manifest_is_not_empty_and_has_no_duplicate_names() {
    let entries = parse(MANIFEST);
    assert!(!entries.is_empty(), "scripts/hosted_tests.txt lists nothing");
    let mut seen = HashSet::new();
    for (target, name, _) in every_name() {
        assert!(
            seen.insert((target.clone(), name.clone())),
            "{target} {name} is listed twice"
        );
    }
    let mut bins = HashSet::new();
    for e in &entries {
        if let Entry::Bin(b) = e {
            assert!(bins.insert(b.clone()), "bin {b} is listed twice");
        }
    }
}

#[test]
fn every_listed_name_is_a_real_test_where_it_is_listed() {
    for (target, name, ignored) in every_name() {
        let leaf = name.rsplit("::").next().unwrap();
        let path = source_of(&target, &name);
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{target} {name}: cannot read {}: {e}", path.display()));
        assert!(
            test_attribute_precedes(&src, leaf),
            "{target} {name}: no #[test] fn {leaf} in {}",
            path.display()
        );
        if ignored {
            assert!(
                src.contains("#[ignore"),
                "{target} {name}: listed as ignored but {} has no #[ignore]",
                path.display()
            );
        }
    }
}

#[test]
fn every_whole_binary_and_lib_filter_names_something() {
    for e in parse(MANIFEST) {
        match e {
            Entry::Bin(b) => {
                let path = root().join("tests").join(format!("{b}.rs"));
                assert!(path.exists(), "bin {b}: {} does not exist", path.display());
            }
            Entry::Lib(filter) => {
                let module = filter.trim_end_matches("::").split("::").next().unwrap();
                let flat = root().join("src").join(format!("{module}.rs"));
                let dir = root().join("src").join(module).join("mod.rs");
                assert!(flat.exists() || dir.exists(), "lib {filter}: no module src/{module}");
            }
            _ => {}
        }
    }
}

#[test]
fn every_exact_test_is_device_free_by_source() {
    for e in parse(MANIFEST) {
        let Entry::Exact { target, names } = e else { continue };
        for name in names {
            let leaf = name.rsplit("::").next().unwrap();
            let src = fs::read_to_string(source_of(&target, &name)).unwrap();
            match device_marker(&src, leaf) {
                Ok(None) => {}
                Ok(Some(why)) => panic!("{target} {name} is in the hosted selection but is not device-free: {why}"),
                Err(e) => panic!("{target} {name}: {e}"),
            }
        }
    }
}

#[test]
fn the_scanner_flags_a_planted_device_test() {
    let direct = "#[test]\nfn planted() { with_gpu(|rt| drop(rt)); }\n";
    assert!(device_marker(direct, "planted").unwrap().is_some());
    let through_helper = "fn helper() { let _rt = GpuRuntime::new(); }\n#[test]\nfn planted() { helper(); }\n";
    assert!(
        device_marker(through_helper, "planted").unwrap().is_some(),
        "a device reached through a helper went unseen"
    );
    let in_comment = "#[test]\nfn fine() { let x = 1; // with_gpu is not called here\n let _ = x; }\n";
    assert!(device_marker(in_comment, "fine").unwrap().is_none());
    assert!(
        device_marker("fn other() {}", "missing").is_err(),
        "a missing test must be an error, not a pass"
    );
}

#[test]
fn the_manifest_parser_refuses_a_malformed_line() {
    for bad in [
        "exact tests_only_a_target\n",
        "bin gdn_gates\n",
        "bin gdn_gates 0\n",
        "lib a::\n",
        "bins\n",
    ] {
        let result = std::panic::catch_unwind(|| parse(bad));
        assert!(result.is_err(), "a malformed manifest line was accepted: {bad:?}");
    }
    assert_eq!(
        parse("# comment\n\nbin gdn_gates 3 # trailing\nbins 4\n"),
        vec![Entry::Bin("gdn_gates".into()), Entry::Bins]
    );
}

/// Run by `scripts/hosted_tests.sh` before it trusts the sandbox: with the
/// device hidden, no runtime can be created. Ignored so a laptop's full run
/// (which has a device) does not see it.
#[test]
#[ignore = "run only under the hosted sandbox: asserts the device is hidden"]
fn hosted_selection_has_no_device() {
    assert!(
        std::env::var_os("TESSL_HOSTED_NO_DEVICE").is_some(),
        "TESSL_HOSTED_NO_DEVICE is not set: this is not the hosted selection"
    );
    assert!(
        tessl::GpuRuntime::new().is_err(),
        "GpuRuntime::new succeeded: the sandbox did not hide the Metal device"
    );
    assert!(
        tessl::GpuRuntime::new_inference().is_err(),
        "GpuRuntime::new_inference succeeded: the sandbox did not hide the Metal device"
    );
}
