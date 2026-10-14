//! Every tool the docs tell you to run must exist.
//!
//! This exists because it did not. `bench_gemm_coop_ab` was named in three
//! places — the tessl README's benchmarking-rigor note and two spots in
//! `docs/gemm_architecture.md`, where it was *the* recommended tool for kernel
//! A/B — for long enough that a stale executable of that name sitting in
//! `target/release/` from an earlier build was the only thing making the
//! reference look live. `cargo build --bin bench_gemm_coop_ab` had been failing
//! with `no bin target named` the whole time, and nothing said so.
//!
//! A doc that names a command which cannot run is worse than one that says
//! nothing: it sends a reader to reproduce a measurement they cannot reproduce,
//! and it hides that the capability the paragraph describes is gone.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Binaries the prose deliberately names as *absent*.
///
/// Keep this list short and each entry justified. An entry here is a promise
/// that the surrounding text says the tool does not exist; it is not a way to
/// silence a broken reference.
const DELIBERATELY_ABSENT: &[&str] = &[
    // Named only inside the corrections recording that it was removed. If a
    // real interleaved coop A/B binary ever lands under this name, delete this
    // entry rather than the text.
    "bench_gemm_coop_ab",
];

/// Pull `` `bench_foo` `` style identifiers out of markdown.
fn bench_binaries_named_in(text: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for chunk in text.split('`').skip(1).step_by(2) {
        let name = chunk.trim();
        if name.starts_with("bench_") && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            found.insert(name.to_string());
        }
    }
    found
}

/// Every tracked-or-not `.md` under `docs/`, plus the crate README, as
/// `(label, path)`. A missing `docs/` is a failure: the directory sits in the
/// repo root and a scan that found nothing must not read as a clean one.
fn doc_files() -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("could not read {}: {e}", dir.display()));
        for entry in entries.map(|e| e.expect("directory entry")) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|x| x == "md") {
                out.push(path);
            }
        }
    }
    let root = crate_root();
    let readme = root.join("README.md");
    assert!(readme.is_file(), "the crate README must be present to check");
    let mut found = Vec::new();
    walk(&root.join("docs"), &mut found);
    assert!(
        found.len() >= 10,
        "found only {} markdown files under docs/; the scan is not reaching them",
        found.len()
    );
    found.sort();
    let mut docs = vec![("README.md".to_string(), readme)];
    for path in found {
        let label = path
            .strip_prefix(&root)
            .expect("under crate root")
            .display()
            .to_string();
        docs.push((label, path));
    }
    docs
}

/// Repo-relative directories a backticked path may start with.
const PATH_ROOTS: &[&str] = &[
    "bench/",
    "scripts/",
    "tools/",
    "src/",
    "kernels/",
    "tests/",
    "docs/",
    "examples/",
    "python/",
];

/// References that name files in a *different* repository, as
/// `(path prefix, marker)`. A reference is exempt only when its path starts
/// with the prefix **and its own line says which repository it lives in** (the
/// marker), so a typo'd in-repo path cannot hide behind an entry. Keep each
/// entry justified.
const EXTERNAL_REFERENCES: &[(&str, &str)] = &[
    // docs/qwen35.md: the memory cap in Lappi-decision's lock script, "a
    // script in that repository and not in this one".
    ("tools/mac_heavy.sh", "Lappi-decision"),
    // docs/verification.md cites the planning log, which lives in
    // Lappi-decision, and says so.
    ("docs/plan-corrections.md", "Lappi-decision"),
    // docs/apple-silicon/*.md name scripts and kernels of the ojas
    // arch02 metal-native crate; each is a link whose target spells that path.
    ("scripts/", "ojas/arch02/metal-native"),
    ("kernels/", "ojas/arch02/metal-native"),
];

/// Backticked tokens that are repo-relative file paths: `(line, path)`.
/// Globs, placeholders, URLs and `:line` / `#anchor` suffixes are not paths
/// to resolve, so the suffixes are stripped and the rest are skipped.
fn paths_named_in(text: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (i, line) in text.lines().enumerate() {
        for chunk in line.split('`').skip(1).step_by(2) {
            let name = chunk.trim();
            let name = name.split(['#', ':']).next().unwrap_or(name);
            let plain = !name.contains(|c: char| "*{}<>$ ()[]|,=".contains(c));
            let has_ext = Path::new(name).extension().is_some();
            if plain && has_ext && PATH_ROOTS.iter().any(|r| name.starts_with(r)) {
                found.push((i + 1, name.to_string()));
            }
        }
    }
    found
}

/// Dangling references in one document's text, as `label:line: message`.
fn dangling_in(label: &str, text: &str, root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for (i, line) in text.lines().enumerate() {
        for name in bench_binaries_named_in(line) {
            if DELIBERATELY_ABSENT.contains(&name.as_str()) {
                continue;
            }
            let src = root.join("src/bin").join(format!("{name}.rs"));
            if !src.exists() {
                missing.push(format!(
                    "{label}:{}: names `{name}`, but {} does not exist",
                    i + 1,
                    src.display()
                ));
            }
        }
    }
    for (line, name) in paths_named_in(text) {
        if root.join(&name).exists() {
            continue;
        }
        let here = text.lines().nth(line - 1).unwrap_or("");
        let external = EXTERNAL_REFERENCES
            .iter()
            .any(|(prefix, marker)| name.starts_with(prefix) && here.contains(marker));
        // A path the prose names as deliberately absent (its stem is on the list).
        let absent = Path::new(&name)
            .file_stem()
            .is_some_and(|s| DELIBERATELY_ABSENT.contains(&s.to_string_lossy().as_ref()));
        if !external && !absent {
            missing.push(format!("{label}:{line}: names path `{name}`, which does not exist"));
        }
    }
    missing
}

#[test]
fn every_bench_binary_and_path_the_docs_name_exists() {
    let root = crate_root();
    let mut missing = Vec::new();
    for (label, path) in doc_files() {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{label}: could not read: {e}"));
        missing.extend(dangling_in(&label, &text, &root));
    }
    assert!(
        missing.is_empty(),
        "dangling tool or path references:\n  {}",
        missing.join("\n  ")
    );
}

/// The check itself must detect a fault: a doc naming a missing path, a
/// missing binary, or a local `tools/` script must fail; an external-repo
/// reference with its repo named must not.
#[test]
fn the_dangling_reference_check_detects_faults() {
    let root = crate_root();
    let bad = dangling_in("x.md", "ok\nrun `scripts/no_such_file.py` now", &root);
    assert_eq!(bad.len(), 1, "{bad:?}");
    assert!(bad[0].starts_with("x.md:2:"), "{bad:?}");
    assert_eq!(dangling_in("x.md", "see `bench_no_such_tool`", &root).len(), 1);
    assert_eq!(dangling_in("x.md", "see `tools/mac_heavy.sh` here", &root).len(), 1);
    assert!(dangling_in("x.md", "Lappi-decision's `tools/mac_heavy.sh`", &root).is_empty());
    // The marker must be on the reference's own line, and match its prefix.
    assert_eq!(
        dangling_in("x.md", "Lappi-decision\n`tools/mac_heavy.sh`", &root).len(),
        1
    );
    assert_eq!(dangling_in("x.md", "Lappi-decision's `tools/other.sh`", &root).len(), 1);
    assert!(dangling_in("x.md", "see `README.md` and `src/lib.rs:10`", &root).is_empty());
}

/// The allowlist must not outlive its justification: an entry naming a binary
/// that now exists is a stale exemption hiding a real check.
#[test]
fn the_absent_list_does_not_name_binaries_that_exist() {
    for name in DELIBERATELY_ABSENT {
        let src = crate_root().join("src/bin").join(format!("{name}.rs"));
        assert!(
            !src.exists(),
            "`{name}` is on the deliberately-absent list but {} exists — remove the exemption",
            src.display()
        );
    }
}

/// The README may not say "no stubs" while the public API ships one.
///
/// `tessl::IcbReplayStub` is a host-side scaffold: its `try_allocate` fails
/// with `NotWired` for the full decode graph. The Known Gaps section said
/// "there are no stubs" regardless, and a reader who trusts that goes looking
/// for a full-graph ICB replay that does not exist. So the claim is pinned to
/// the behaviour: while the stub still refuses, Known Gaps must name it.
#[test]
fn known_gaps_names_the_icb_scaffold_the_crate_still_exports() {
    let mut stub = tessl::IcbReplayStub::new();
    assert_eq!(
        stub.try_allocate(),
        Err(tessl::CbReplayError::NotWired),
        "full-graph ICB allocate is wired now — retire the Known Gaps row and this test"
    );

    let readme = std::fs::read_to_string(crate_root().join("README.md")).expect("README");
    assert!(
        !readme.to_ascii_lowercase().contains("no stubs"),
        "README claims there are no stubs, but `tessl::IcbReplayStub` is a NotWired scaffold"
    );
    let gaps = readme
        .split("## Known Gaps")
        .nth(1)
        .expect("README has a Known Gaps section")
        .split("\n---")
        .next()
        .expect("Known Gaps section is delimited");
    for needle in ["IcbReplayStub", "IcbStubPhase", "NotWired"] {
        assert!(gaps.contains(needle), "Known Gaps must name `{needle}`");
    }
}

/// The kernel counts the crate docs and README quote are the build's, not a
/// snapshot: they said 21 sources and 83 entry points long after the Qwen3.5
/// and cross-entropy kernels had joined the build.
#[test]
fn the_kernel_counts_the_docs_quote_are_the_builds() {
    let sources = std::fs::read_dir(crate_root().join("kernels"))
        .expect("kernels/")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "metal"))
        .count();
    let nm = std::process::Command::new("xcrun")
        .args(["metal-nm", tessl::metallib_path()])
        .output()
        .expect("xcrun metal-nm");
    assert!(
        nm.status.success(),
        "metal-nm failed: {}",
        String::from_utf8_lossy(&nm.stderr)
    );
    let entries = String::from_utf8_lossy(&nm.stdout)
        .lines()
        .filter(|l| l.split_whitespace().nth(1) == Some("T"))
        .count();
    let nn = std::fs::read_to_string(crate_root().join("src/nn.rs")).expect("nn.rs");
    let nn_fns = nn
        .lines()
        .filter(|l| l.starts_with("pub fn ") || l.starts_with("pub unsafe fn "))
        .count();
    let lib = std::fs::read_to_string(crate_root().join("src/lib.rs")).expect("lib.rs");
    let readme = std::fs::read_to_string(crate_root().join("README.md")).expect("README");
    let lib_claim = format!("{sources} Metal sources compile to {entries} kernel entry points");
    let nn_claim = format!("through {nn_fns} shape-checked functions");
    let readme_claim = format!("{sources} Metal source files providing {entries} kernel entry points");
    let readme_nn_claim = format!("through {nn_fns} shape-checked entry points in `tessl::nn`");
    let lib_flat = lib.replace("\n//! ", " ");
    assert!(lib_flat.contains(&lib_claim), "src/lib.rs should say `{lib_claim}`");
    assert!(lib_flat.contains(&nn_claim), "src/lib.rs should say `{nn_claim}`");
    assert!(readme.contains(&readme_claim), "README.md should say `{readme_claim}`");
    assert!(
        readme.contains(&readme_nn_claim),
        "README.md should say `{readme_nn_claim}`"
    );
}

/// Entry points the GEMM tune rig adds, from a compile of `kernels/tune/`.
/// Entry-point names do not depend on math mode, so the flags here are the
/// bare minimum, not build.rs's.
///
/// No toolchain is a failure, not a skip: this crate does not build without
/// the Metal 4 compiler (build.rs refuses a simdgroup-only metallib), and a
/// count that could not be taken must not read as one that matched.
fn tune_entry_points() -> usize {
    let root = crate_root();
    let sdk = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .expect("spawn xcrun --show-sdk-path");
    assert!(
        sdk.status.success(),
        "no macOS SDK, so the tune-kernel count cannot be taken: {}",
        String::from_utf8_lossy(&sdk.stderr)
    );
    let sdk = String::from_utf8_lossy(&sdk.stdout).trim().to_string();
    let dir = std::env::temp_dir().join(format!("tessl-tune-count-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut total = 0;
    let mut sources: Vec<_> = std::fs::read_dir(root.join("kernels/tune"))
        .expect("kernels/tune")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "metal"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "kernels/tune holds no .metal source");
    for src in &sources {
        let air = dir.join(format!("{}.air", src.file_stem().unwrap().to_string_lossy()));
        let cc = std::process::Command::new("xcrun")
            .args([
                "metal",
                "-std=metal4.0",
                "-O2",
                "-isysroot",
                &sdk,
                "-mmacosx-version-min=26.0",
                "-c",
            ])
            .arg(src)
            .arg("-o")
            .arg(&air)
            .output()
            .expect("spawn xcrun metal");
        assert!(
            cc.status.success(),
            "{} failed to compile under -std=metal4.0 (a missing Metal Toolchain fails here too): {}",
            src.display(),
            String::from_utf8_lossy(&cc.stderr)
        );
        let nm = std::process::Command::new("xcrun")
            .arg("metal-nm")
            .arg(&air)
            .output()
            .expect("xcrun metal-nm");
        assert!(nm.status.success(), "metal-nm: {}", String::from_utf8_lossy(&nm.stderr));
        total += String::from_utf8_lossy(&nm.stdout)
            .lines()
            .filter(|l| l.split_whitespace().nth(1) == Some("T"))
            .count();
    }
    let _ = std::fs::remove_dir_all(&dir);
    total
}

/// Every integer a doc puts directly before a tune-kernel phrase, with the
/// phrase it sits before.
fn quoted_tune_counts<'a>(text: &'a str, phrases: &[&'a str]) -> Vec<(&'a str, usize)> {
    let mut found = Vec::new();
    for &phrase in phrases {
        for (at, _) in text.match_indices(phrase) {
            let head = text[..at].trim_end_matches([' ', '-']);
            let digits = head.len() - head.trim_end_matches(|c: char| c.is_ascii_digit()).len();
            if digits > 0 {
                found.push((phrase, head[head.len() - digits..].parse().expect("digits")));
            }
        }
    }
    found
}

#[test]
fn the_tune_kernel_count_the_docs_quote_is_the_builds() {
    let tune = tune_entry_points();
    assert!(tune > 0, "kernels/tune/ compiled to no entry points");
    const PHRASES: &[&str] = &[
        " variants)",
        "-kernel A/B",
        " A/B Kernels",
        " measurement-only kernels",
        " measurement-only entry points",
        " measurement kernels",
    ];
    let mut quoted = 0;
    for doc in [
        "README.md",
        "docs/benchmarking.md",
        "docs/apple-silicon/gemm_architecture.md",
    ] {
        let text = std::fs::read_to_string(crate_root().join(doc)).expect(doc);
        for (phrase, n) in quoted_tune_counts(&text, PHRASES) {
            quoted += 1;
            assert_eq!(
                n, tune,
                "{doc} quotes {n}{phrase:?}, but kernels/tune/ builds {tune} entry points"
            );
        }
    }
    assert!(
        quoted >= 6,
        "found only {quoted} tune-count claims; did the docs' wording move?"
    );
}
