//! `src/bin/dialect_lint.rs` is a gate only if it can go red. Each case runs
//! the binary against a copy of `kernels/` with one fault planted, so the lint
//! audits the faulty tree and never this checkout's.
//!
//! The lint needs only the standard library (the Linux kernel-emulator CI job
//! compiles it with a bare `rustc`), so these tests need no GPU, no Metal
//! toolchain and no torch. The file keeps the name of the Python script it
//! replaced, `tools/msl_emu/dialect_lint.py`, which the manifest and the
//! history refer to.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const NEW_KERNELS: [&str; 4] = ["qwen35_gdn", "qwen35_attn", "qwen35_mlp", "qwen35_score"];

struct Run {
    ok: bool,
    out: String,
}

/// A fresh crate-shaped tree: a copy of `kernels/`, the lint's input.
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

fn lint_with(root: &Path, extra: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_dialect_lint"))
        .args(extra)
        .arg(root)
        .output()
        .unwrap_or_else(|e| panic!("cannot run the dialect lint: {e}"));
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

fn lint(root: &Path) -> Run {
    lint_with(root, &[])
}

fn append(root: &Path, kernel: &str, text: &str) {
    let path = root.join("kernels").join(format!("{kernel}.metal"));
    let mut src = fs::read_to_string(&path).unwrap();
    src.push_str(text);
    fs::write(path, src).unwrap();
}

/// The lines of the report that name `construct`.
fn rows_for<'a>(run: &'a Run, construct: &str) -> Vec<&'a str> {
    run.out.lines().filter(|l| l.contains(construct)).collect()
}

#[test]
fn the_unmodified_tree_is_clean() {
    let run = lint(&tree("clean"));
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
    let run = lint(&root);
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
    let run = lint_with(&root, &["--reviewed", "fabs=MSL common math function"]);
    assert!(!run.ok, "a stale allowlist entry passed:\n{}", run.out);
    assert!(run.out.contains("FAIL") && run.out.contains("fabs"), "{}", run.out);
}

#[test]
fn the_shipped_allowlist_has_no_stale_entry() {
    // The clean-tree case above fails on any stale entry; this names the
    // property so a reader of a failure knows which gate tripped.
    let run = lint(&tree("shipped_allowlist"));
    assert!(run.ok && !run.out.contains("FAIL"), "{}", run.out);
}

#[test]
fn a_tree_without_the_qwen35_sources_is_not_a_pass() {
    let root = tree("no_sources");
    for k in NEW_KERNELS {
        fs::remove_file(root.join("kernels").join(format!("{k}.metal"))).unwrap();
    }
    let run = lint(&root);
    assert!(!run.ok, "a tree with nothing to lint passed:\n{}", run.out);
    assert!(run.out.contains("cannot read"), "{}", run.out);
}

#[test]
fn one_missing_or_undecodable_qwen35_source_is_a_named_failure() {
    let root = tree("one_missing");
    fs::remove_file(root.join("kernels/qwen35_mlp.metal")).unwrap();
    let run = lint(&root);
    assert!(!run.ok, "{}", run.out);
    assert!(
        run.out.contains("qwen35_mlp.metal") && run.out.contains("cannot read"),
        "{}",
        run.out
    );

    let root = tree("not_utf8");
    let path = root.join("kernels/qwen35_gdn.metal");
    let mut bytes = fs::read(&path).unwrap();
    bytes.extend_from_slice(b"\xff\xfe");
    fs::write(&path, bytes).unwrap();
    let run = lint(&root);
    assert!(!run.ok, "{}", run.out);
    assert!(run.out.contains("is not UTF-8"), "{}", run.out);
}

/// A tree whose Qwen3.5 sources add nothing the other kernels do not already
/// use has been examined for nothing, which is not a pass: here the shipped
/// allowlist entries all go stale, so that is what the lint names.
#[test]
fn a_tree_with_nothing_new_is_not_a_pass() {
    let root = tree("nothing_new");
    let mut all = String::new();
    for k in NEW_KERNELS {
        all.push_str(&fs::read_to_string(root.join("kernels").join(format!("{k}.metal"))).unwrap());
        all.push('\n');
    }
    fs::write(root.join("kernels/copycat.metal"), all).unwrap();
    let run = lint(&root);
    assert!(!run.ok, "{}", run.out);
    assert!(run.out.contains("no longer needed"), "{}", run.out);
    assert!(!run.out.contains("all reviewed"), "{}", run.out);
}

/// What counts as a construct: calls, qualified names, attributes (with their
/// arguments normalised), pointer casts (with their spacing normalised) and
/// template calls; what does not: comments.
#[test]
fn the_construct_kinds_are_read_and_comments_are_not() {
    let root = tree("kinds");
    append(
        &root,
        "qwen35_gdn",
        "\nvoid probe_kinds() {\n\
         \x20   plain_call_probe(1);\n\
         \x20   probe_ns::qualified_probe(2);\n\
         \x20   template_call_probe<float, 4>(3);\n\
         \x20   auto p = (device  const  probe_t *)q;\n\
         \x20   // line_comment_probe(1);\n\
         \x20   /* block_comment_probe(1); */\n\
         }\n\
         struct S { float a [[probe_attr(7)]]; };\n",
    );
    let run = lint(&root);
    assert!(!run.ok, "{}", run.out);
    for want in [
        "plain_call_probe",
        "probe_ns::qualified_probe",
        "template_call_probe",
        "[[probe_attr(n)]]",
        "cast (device const probe_t *)",
    ] {
        let rows = rows_for(&run, want);
        assert_eq!(rows.len(), 1, "{want}: {rows:?}\n{}", run.out);
        assert!(
            rows[0].contains("[FAIL]") && rows[0].contains("qwen35_gdn.metal"),
            "{want}: {}",
            rows[0]
        );
    }
    for hidden in ["line_comment_probe", "block_comment_probe"] {
        assert!(
            rows_for(&run, hidden).is_empty(),
            "{hidden} was read from a comment:\n{}",
            run.out
        );
    }
}

/// A helper, kernel or macro the Qwen3.5 sources define themselves is their
/// own name and not a construct to review.
#[test]
fn names_the_new_sources_define_themselves_are_not_flagged() {
    let root = tree("own_names");
    append(
        &root,
        "qwen35_mlp",
        "\ninline float own_helper_probe(float x) { return x; }\n\
         #define OWN_MACRO_PROBE(NAME_PROBE, ACT_PROBE) \\\n\
         \x20   kernel void NAME_PROBE(device float *y [[buffer(0)]]) { y[0] = ACT_PROBE(y[0]); }\n\
         OWN_MACRO_PROBE(own_kernel_probe, own_helper_probe)\n\
         kernel void own_kernel_two_probe(device float *y [[buffer(0)]]) { y[0] = own_helper_probe(y[0]); }\n",
    );
    let run = lint(&root);
    assert!(run.ok, "own names were flagged:\n{}", run.out);
    for own in [
        "own_helper_probe",
        "OWN_MACRO_PROBE",
        "own_kernel_probe",
        "own_kernel_two_probe",
        "NAME_PROBE",
        "ACT_PROBE",
    ] {
        assert!(rows_for(&run, own).is_empty(), "{own} was flagged:\n{}", run.out);
    }
}

/// A construct another kernel (in a subdirectory too) already uses is not new;
/// a hidden directory's files are not read.
#[test]
fn constructs_other_kernels_use_are_known_and_hidden_directories_are_skipped() {
    let root = tree("known_elsewhere");
    fs::create_dir_all(root.join("kernels/sub/deeper")).unwrap();
    fs::write(
        root.join("kernels/sub/deeper/x.metal"),
        "void a() { nested_probe(1); }\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("kernels/.hidden")).unwrap();
    fs::write(root.join("kernels/.hidden/y.metal"), "void a() { hidden_probe(1); }\n").unwrap();
    // A header is read at the top level only.
    fs::write(root.join("kernels/sub/z.h"), "void a() { nested_header_probe(1); }\n").unwrap();
    fs::write(root.join("kernels/top.h"), "void a() { top_header_probe(1); }\n").unwrap();
    append(
        &root,
        "qwen35_score",
        "\nvoid use_them() { nested_probe(1); hidden_probe(2); nested_header_probe(3); top_header_probe(4); }\n",
    );
    let run = lint(&root);
    assert!(!run.ok, "{}", run.out);
    assert!(rows_for(&run, "nested_probe").is_empty(), "{}", run.out);
    assert!(rows_for(&run, "top_header_probe").is_empty(), "{}", run.out);
    assert_eq!(rows_for(&run, "hidden_probe").len(), 1, "{}", run.out);
    assert_eq!(rows_for(&run, "nested_header_probe").len(), 1, "{}", run.out);
}

/// The Linux CI job has no tessl build, only a bare `rustc`: the lint must
/// compile from its one source file with nothing else, and agree with the
/// binary cargo built on the same tree.
#[test]
fn the_source_compiles_standalone_with_a_bare_rustc() {
    let out_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dialect_lint_standalone");
    fs::create_dir_all(&out_dir).unwrap();
    let exe = out_dir.join("dialect_lint");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let build = Command::new(rustc)
        .args(["--edition", "2021", "-O", "-o"])
        .arg(&exe)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/dialect_lint.rs"))
        .output()
        .expect("cannot run rustc");
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let root = tree("standalone");
    let standalone = Command::new(&exe).arg(&root).output().unwrap();
    let cargo_built = lint(&root);
    assert!(standalone.status.success() && cargo_built.ok);
    assert_eq!(String::from_utf8_lossy(&standalone.stdout), cargo_built.out);
}

#[test]
fn a_malformed_reviewed_flag_is_refused() {
    let run = lint_with(&tree("bad_flag"), &["--reviewed", "no-equals-sign"]);
    assert!(!run.ok && run.out.contains("usage:"), "{}", run.out);
}
