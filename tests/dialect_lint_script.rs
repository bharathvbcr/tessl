//! `tools/msl_emu/dialect_lint.py` is a gate only if it can go red. Each case
//! runs a copy of the script against a copy of `kernels/` with one fault
//! planted, so the lint audits the faulty tree and never this checkout's.
//!
//! The lint needs only the standard library, so these tests need only a
//! `python3`; no GPU, no Metal toolchain, no torch.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const LINT: &str = include_str!("../tools/msl_emu/dialect_lint.py");
const NEW_KERNELS: [&str; 4] = ["qwen35_gdn", "qwen35_attn", "qwen35_mlp", "qwen35_score"];

struct Run {
    ok: bool,
    out: String,
}

/// A fresh crate-shaped tree: a copy of `kernels/` (the lint's input) and the
/// lint script (with `edit` applied) at `tools/msl_emu/`.
fn tree(case: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dialect_lint").join(case);
    let _ = fs::remove_dir_all(&root);
    let kernels = root.join("kernels");
    fs::create_dir_all(&kernels).unwrap();
    copy_kernels(&Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"), &kernels);
    root
}

fn copy_kernels(from: &Path, to: &Path) {
    for entry in fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let dest = to.join(path.file_name().unwrap());
        if path.is_dir() {
            fs::create_dir_all(&dest).unwrap();
            copy_kernels(&path, &dest);
        } else if path.extension().is_some_and(|e| e == "metal" || e == "h") {
            fs::copy(&path, &dest).unwrap();
        }
    }
}

fn lint(root: &Path, script: &str) -> Run {
    let path = root.join("tools/msl_emu/dialect_lint.py");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, script).unwrap();
    let out = Command::new("python3")
        .arg("-I")
        .arg(&path)
        .arg(root)
        .output()
        .unwrap_or_else(|e| panic!("cannot run python3 for the dialect lint: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("--- {} (exit {:?})\n{text}", root.display(), out.status.code());
    Run {
        ok: out.status.success(),
        out: text,
    }
}

fn append(root: &Path, kernel: &str, text: &str) {
    let path = root.join("kernels").join(format!("{kernel}.metal"));
    let mut src = fs::read_to_string(&path).unwrap();
    src.push_str(text);
    fs::write(path, src).unwrap();
}

#[test]
fn the_unmodified_tree_is_clean() {
    let run = lint(&tree("clean"), LINT);
    assert!(run.ok, "the lint fails on the unmodified tree:\n{}", run.out);
    assert!(run.out.contains("all reviewed"), "{}", run.out);
    assert!(
        !run.out.contains("no longer needed"),
        "a stale entry is still reported:\n{}",
        run.out
    );
}

#[test]
fn an_unreviewed_construct_fails() {
    let root = tree("unreviewed");
    append(
        &root,
        "qwen35_score",
        "\ninline float lint_probe(float x) { return frobnicate_probe(x); }\n",
    );
    let run = lint(&root, LINT);
    assert!(!run.ok, "an unreviewed construct passed:\n{}", run.out);
    assert!(
        run.out.contains("frobnicate_probe") && run.out.contains("not reviewed"),
        "{}",
        run.out
    );
}

/// Ten of these printed a note and exited 0 before: an allowlist entry that no
/// longer vouches for anything is an unchecked claim.
#[test]
fn a_reviewed_entry_that_is_no_longer_used_fails() {
    let root = tree("stale_entry");
    // `fabs` is used by tessl's other kernels, so it is not new to them.
    let script = inject(
        LINT,
        "REVIEWED = {\n",
        "REVIEWED = {\n    \"fabs\": \"MSL common math function\",\n",
    );
    let run = lint(&root, &script);
    assert!(!run.ok, "a stale allowlist entry passed:\n{}", run.out);
    assert!(run.out.contains("FAIL") && run.out.contains("fabs"), "{}", run.out);
}

#[test]
fn the_shipped_allowlist_has_no_stale_entry() {
    // The clean-tree case above fails on any stale entry; this names the
    // property so a reader of a failure knows which gate tripped.
    let run = lint(&tree("shipped_allowlist"), LINT);
    assert!(run.ok && !run.out.contains("FAIL"), "{}", run.out);
}

#[test]
fn a_tree_without_the_qwen35_sources_is_not_a_pass() {
    let root = tree("no_sources");
    for k in NEW_KERNELS {
        fs::remove_file(root.join("kernels").join(format!("{k}.metal"))).unwrap();
    }
    let run = lint(&root, LINT);
    assert!(!run.ok, "a tree with nothing to lint passed:\n{}", run.out);
}

fn inject(src: &str, from: &str, to: &str) -> String {
    assert_eq!(src.matches(from).count(), 1, "injection site not unique: {from:?}");
    src.replacen(from, to, 1)
}
