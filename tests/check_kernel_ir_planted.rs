//! Plants kernels in a scratch directory and runs
//! `scripts/check_kernel_ir.sh` on them. Needs `xcrun metal` (like the build);
//! the tests say so and return when the script exits 77 (SKIPPED).
//!
//! The script's constant check parses the exponent, so it also sees what LLVM
//! prints for half and bfloat (`0xH7C00`, `0xR7F80`) and a signalling-NaN f32
//! (`0x7FF4000000000000`). The fake-`xcrun` tests need no toolchain.

use std::path::PathBuf;
use std::process::Command;

fn run_on(name: &str, kernel_body: &str) -> Option<(i32, String)> {
    let dir: PathBuf = std::env::temp_dir().join(format!("tessl_ir_planted_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = format!("#include <metal_stdlib>\nusing namespace metal;\n{kernel_body}\n");
    std::fs::write(dir.join(format!("{name}.metal")), src).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/check_kernel_ir.sh");
    let out = Command::new("bash").arg(&script).arg(&dir).output().expect("bash runs");
    std::fs::remove_dir_all(&dir).unwrap();
    let code = out.status.code().expect("script exited normally");
    let text = String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    if code == 77 {
        eprintln!("SKIPPED (no Metal toolchain): {text}");
        return None;
    }
    Some((code, text))
}

const F32: &str = "device float *o [[buffer(0)]], device const float *a [[buffer(1)]]";
const BF: &str = "device bfloat *o [[buffer(0)]], device const bfloat *a [[buffer(1)]]";
const HF: &str = "device half *o [[buffer(0)]], device const half *a [[buffer(1)]]";

fn kernel(args: &str, body: &str) -> String {
    format!("kernel void k({args}, uint i [[thread_position_in_grid]]) {{ {body} }}")
}

#[test]
fn control_f32_infinity_compare_is_rejected_and_a_clean_kernel_passes() {
    if let Some((code, text)) = run_on("inf", &kernel(F32, "o[i] = (a[i] == INFINITY) ? 1.f : 0.f;")) {
        assert_eq!(code, 1, "{text}");
    }
    if let Some((code, text)) = run_on("clean", &kernel(F32, "o[i] = a[i] * 2.f;")) {
        assert_eq!(code, 0, "{text}");
    }
}

#[test]
fn bfloat_infinity_constant_is_rejected() {
    let k = kernel(BF, "o[i] = (a[i] == (bfloat)INFINITY) ? bfloat(1.f) : bfloat(0.f);");
    if let Some((code, text)) = run_on("bf_inf", &k) {
        assert_eq!(code, 1, "bfloat 0xR7F80 slipped through: {text}");
    }
}

#[test]
fn half_infinity_constant_is_rejected() {
    let k = kernel(HF, "o[i] = (a[i] == (half)INFINITY) ? 1.h : 0.h;");
    if let Some((code, text)) = run_on("half_inf", &k) {
        assert_eq!(code, 1, "half 0xH7C00 slipped through: {text}");
    }
}

#[test]
fn signalling_nan_constant_is_rejected() {
    let k = kernel(F32, "o[i] = (a[i] > 0.f) ? as_type<float>(0x7fa00000u) : 0.f;");
    if let Some((code, text)) = run_on("snan", &k) {
        assert_eq!(code, 1, "float 0x7FF4000000000000 slipped through: {text}");
    }
}

/// Run the script on a directory holding one trivial source with a fake `xcrun`
/// (first on `PATH`) whose `metal` exits 0 and leaves `mode`'s output: nothing
/// (`"none"`) or an empty file (`"empty"`).
fn run_with_fake_metal(mode: &str) -> (i32, String) {
    use std::os::unix::fs::PermissionsExt;
    let dir: PathBuf = std::env::temp_dir().join(format!("tessl_ir_fake_{}_{mode}", std::process::id()));
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::write(dir.join("kernels/k.metal"), "kernel void k() {}\n").unwrap();
    let write_exe = |name: &str, body: &str| {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    };
    let metal = dir.join("metal");
    write_exe(
        "metal",
        &format!(
            "#!/bin/sh\nwhile [ $# -gt 0 ]; do [ \"$1\" = -o ] && out=\"$2\"; shift; done\n[ {mode} = empty ] && : > \"$out\"\nexit 0\n"
        ),
    );
    write_exe(
        "xcrun",
        &format!(
            "#!/bin/sh\ncase \"$1\" in -f) echo {};; *) echo /nonexistent-sdk;; esac\n",
            metal.display()
        ),
    );
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/check_kernel_ir.sh");
    let out = Command::new("bash")
        .arg(&script)
        .arg(dir.join("kernels"))
        .env("PATH", format!("{}:/usr/bin:/bin", dir.display()))
        .output()
        .expect("bash runs");
    std::fs::remove_dir_all(&dir).unwrap();
    let text = String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout);
    (out.status.code().expect("script exited normally"), text)
}

/// A `metal` that exits 0 without writing IR must fail the check, never pass it.
#[test]
fn a_compile_that_leaves_no_ir_fails_closed() {
    for mode in ["none", "empty"] {
        let (code, text) = run_with_fake_metal(mode);
        assert_eq!(code, 1, "{mode}: {text}");
        assert!(text.contains("wrote no IR"), "{mode}: {text}");
    }
}

/// NaN payloads and a negative infinity are caught by the exponent, not by a
/// fixed spelling.
#[test]
fn nan_payloads_are_rejected_by_exponent() {
    for (name, bits) in [
        ("nan_lo", "0x7f800001u"),
        ("nan_neg", "0xffc12345u"),
        ("inf_neg", "0xff800000u"),
    ] {
        let k = kernel(F32, &format!("o[i] = (a[i] > 0.f) ? as_type<float>({bits}) : 0.f;"));
        if let Some((code, text)) = run_on(name, &k) {
            assert_eq!(code, 1, "{bits} slipped through: {text}");
        }
    }
}
