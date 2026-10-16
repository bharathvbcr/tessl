//! Device-free checks on the GPU fault-injection set (`scripts/fault_inject.sh`,
//! `scripts/faults/`, docs/verification.md "Fault injection").
//!
//! The GPU run itself needs the shared device and a metallib rebuild per fault, so
//! it is not a test. What can rot without a device is the set around it: a
//! manifest row whose test was renamed (the driver would then report it NOT RUN,
//! which is honest but loses the row), a patch that no longer has exactly one hunk
//! in a kernel source, and the driver's own rule that a patch which cannot be
//! applied is never reported as caught. Whether a patch still applies to the
//! *current* kernels is deliberately not asserted here: a kernel edit landing
//! should not turn an unrelated suite red. `scripts/fault_inject.sh --check`
//! answers that, and the GPU run reports a stale patch as NOT RUN.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Row {
    id: String,
    sources: String,
    candidates: Vec<(String, String)>,
}

fn rows() -> Vec<Row> {
    let text = fs::read_to_string(root().join("scripts/faults/manifest.tsv")).expect("manifest");
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split('|').collect();
        assert_eq!(f.len(), 5, "manifest row needs 5 `|` fields: {line}");
        let candidates = f[4]
            .split_whitespace()
            .map(|c| {
                let (bin, name) = c
                    .split_once(':')
                    .unwrap_or_else(|| panic!("candidate `{c}` is not bin:test"));
                (bin.to_string(), name.to_string())
            })
            .collect::<Vec<_>>();
        assert!(!candidates.is_empty(), "{}: no candidate tests", f[0]);
        out.push(Row {
            id: f[0].into(),
            sources: f[1].into(),
            candidates,
        });
    }
    out
}

#[test]
fn the_driver_refuses_a_stale_patch_and_never_counts_a_run_that_did_not_run() {
    let out = Command::new("bash")
        .arg(root().join("scripts/fault_inject.sh"))
        .arg("--self-test")
        .current_dir(root())
        .output()
        .expect("bash runs the driver");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "driver self-test failed:\n{text}");
    assert!(text.contains("a stale patch is refused"), "{text}");
    assert!(text.contains("is NOT FOUND, not a pass"), "{text}");
}

#[test]
fn every_manifest_row_has_a_one_hunk_kernel_patch_and_real_tests() {
    let rows = rows();
    assert!(rows.len() >= 12, "the set is 12 to 20 faults, found {}", rows.len());
    let ids: BTreeSet<_> = rows.iter().map(|r| r.id.clone()).collect();
    assert_eq!(ids.len(), rows.len(), "duplicate fault id");

    for r in &rows {
        let patch = fs::read_to_string(root().join(format!("scripts/faults/{}.patch", r.id)))
            .unwrap_or_else(|e| panic!("{}: patch missing: {e}", r.id));
        let hunks = patch.lines().filter(|l| l.starts_with("@@ ")).count();
        assert_eq!(hunks, 1, "{}: a fault is one hunk", r.id);
        let files: Vec<&str> = patch.lines().filter_map(|l| l.strip_prefix("+++ b/")).collect();
        assert_eq!(files.len(), 1, "{}: a fault patches one file", r.id);
        assert!(files[0].starts_with("kernels/"), "{}: patches {}", r.id, files[0]);
        assert!(
            r.sources.contains(files[0].trim_start_matches("kernels/")),
            "{}: manifest says {}, patch edits {}",
            r.id,
            r.sources,
            files[0]
        );

        for (bin, name) in &r.candidates {
            let src = fs::read_to_string(root().join(format!("tests/{bin}.rs")))
                .unwrap_or_else(|e| panic!("{}: tests/{bin}.rs: {e}", r.id));
            let decl = format!("fn {name}(");
            let at = src
                .find(&decl)
                .unwrap_or_else(|| panic!("{}: tests/{bin}.rs has no `{decl}`", r.id));
            // The attributes right above the fn: it must be a plain #[test].
            let head: Vec<&str> = src[..at].lines().rev().take(4).collect();
            assert!(
                head.iter().any(|l| l.trim() == "#[test]"),
                "{}: {bin}:{name} is not a #[test]",
                r.id
            );
            assert!(
                !head.iter().any(|l| l.trim_start().starts_with("#[ignore")),
                "{}: {bin}:{name} is #[ignore]d",
                r.id
            );
        }
    }

    // No orphan patches: every .patch but the self-test's own is a manifest row.
    for e in fs::read_dir(root().join("scripts/faults")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "patch") {
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            assert!(stem == "selftest-stale" || ids.contains(&stem), "orphan patch {stem}");
        }
    }
}

#[test]
fn the_self_test_patch_is_stale_by_construction() {
    // If someone "fixes" it so it applies, the driver's own rule goes untested.
    let out = Command::new("git")
        .args(["apply", "--check", "scripts/faults/selftest-stale.patch"])
        .current_dir(root())
        .output()
        .expect("git runs");
    assert!(!out.status.success(), "selftest-stale.patch applies; it must not");
}
