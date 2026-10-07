//! `scripts/audit_gemm_tiles.py` is only worth its `PASS` if it can print
//! `FAIL`. Each case copies the script and the two files it reads into a fresh
//! crate-shaped directory, injects one fault, and runs the copy — the script
//! resolves its inputs from its own location, so the copy audits the faulty
//! files, never this checkout's.
//!
//! An earlier version printed `PASS: 0 mismatch(es)` while half of it examined
//! nothing (`GAP-TESSL-AUDIT-BKC-CHECK-DEAD`); the cases below are the ways a
//! check can stop examining things without anything going red.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCRIPT: &str = include_str!("../scripts/audit_gemm_tiles.py");
const METAL: &str = include_str!("../kernels/matmul_tensorops.metal");
const GEMM_RS: &str = include_str!("../src/gemm.rs");

struct Run {
    ok: bool,
    stdout: String,
}

fn audit(case: &str, metal: &str, gemm_rs: &str) -> Run {
    let root: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("audit_gemm_tiles")
        .join(case);
    if root.exists() {
        fs::remove_dir_all(&root).unwrap();
    }
    for (rel, body) in [
        ("scripts/audit_gemm_tiles.py", SCRIPT),
        ("kernels/matmul_tensorops.metal", metal),
        ("src/gemm.rs", gemm_rs),
    ] {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, body).unwrap();
    }
    let out = Command::new("python3")
        .arg(root.join("scripts/audit_gemm_tiles.py"))
        .output()
        .unwrap_or_else(|e| panic!("cannot run python3 for the tile audit: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("--- {case} (exit {:?})\n{stdout}{stderr}", out.status.code());
    Run {
        ok: out.status.success(),
        stdout,
    }
}

/// Replace exactly one occurrence, so a fault that silently failed to inject
/// cannot pass as a fault the audit missed.
fn inject(src: &str, from: &str, to: &str) -> String {
    assert_eq!(src.matches(from).count(), 1, "injection site not unique: {from:?}");
    src.replacen(from, to, 1)
}

#[test]
fn clean_tree_passes_and_names_what_it_checked() {
    let run = audit("clean", METAL, GEMM_RS);
    assert!(run.ok, "the audit fails on the unmodified tree");
    let last = run.stdout.lines().rev().find(|l| !l.is_empty()).unwrap();
    assert!(
        last.starts_with("PASS: tile geometry: ") && last.contains(" kernels"),
        "PASS line does not name what it checked: {last:?}"
    );
    assert!(!run.stdout.contains("BKC"), "the retired BKC check is still reported");
}

#[test]
fn tile_drift_fails() {
    let gemm = inject(
        GEMM_RS,
        "const TILE_COOP_NARROW: TileGeom = TileGeom {\n    sm: 64,",
        "const TILE_COOP_NARROW: TileGeom = TileGeom {\n    sm: 32,",
    );
    let run = audit("tile_drift", METAL, &gemm);
    assert!(!run.ok);
    assert!(run.stdout.contains("MISMATCH"));
}

/// `matmul2d_tensorops_bf16_f32` is the production NN kernel and takes its
/// geometry from `mm_nn_coop_f32acc`'s template defaults, not from a
/// `constexpr` or a macro argument. It was once skipped as "no compile-time
/// SM/SN" — and a skipped kernel counted as passed.
#[test]
fn template_default_drift_fails() {
    let metal = inject(
        METAL,
        "template <typename ElemT, int SM = 128, int SN = 64, int NSG = 4,",
        "template <typename ElemT, int SM = 256, int SN = 64, int NSG = 4,",
    );
    let run = audit("template_default_drift", &metal, GEMM_RS);
    assert!(!run.ok);
    assert!(run.stdout.contains("matmul2d_tensorops_bf16_f32 "));
}

/// A kernel the host dispatches through a variable never appears next to a
/// `pipeline("literal")`, so nothing checks its tile unless something insists
/// every entry point is accounted for.
#[test]
fn unaccounted_kernel_fails() {
    let metal = format!("{METAL}\nNN_COOP_KERNEL(matmul2d_tensorops_audit_probe, bfloat, 64, 64, 4, false)\n");
    let run = audit("unaccounted_kernel", &metal, GEMM_RS);
    assert!(!run.ok);
    assert!(run.stdout.contains("matmul2d_tensorops_audit_probe"));
}

/// A new cooperative macro whose arguments the parser does not understand
/// must stop the audit, not drop its kernels from it.
#[test]
fn unparsed_kernel_macro_fails() {
    let metal = format!("{METAL}\nNN_COOP_SPLIT_KERNEL(matmul2d_tensorops_audit_probe, bfloat, 64, 64)\n");
    let run = audit("unparsed_kernel_macro", &metal, GEMM_RS);
    assert!(!run.ok);
    assert!(run.stdout.contains("NN_COOP_SPLIT_KERNEL"));
}

/// Nothing to examine is a failure, never `PASS: 0 mismatch(es)`.
#[test]
fn no_kernels_fails() {
    let run = audit("no_kernels", "", GEMM_RS);
    assert!(!run.ok);
    assert!(run.stdout.contains("examined nothing"));
}

#[test]
fn no_rust_dispatch_fails() {
    let run = audit("no_rust_dispatch", METAL, "");
    assert!(!run.ok);
    assert!(run.stdout.contains("examined nothing"));
}
