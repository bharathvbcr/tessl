//! Tester-written: plants kernels in a scratch directory and runs
//! `scripts/check_kernel_ir.sh` on them. Needs `xcrun metal` (like the build);
//! the tests say so and return when the script exits 77 (SKIPPED).
//!
//! `known_gap_*` tests FAIL on the implementation as committed: the script's
//! constant pattern knows only the f32/f64 spellings `0x7FF0...`, `0xFFF0...`,
//! `0x7FF8...`, `0xFFF8...`. LLVM prints half and bfloat constants as
//! `0xH7C00` / `0xR7F80`, and a signalling-NaN f32 as `0x7FF4000000000000`.

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
fn known_gap_bfloat_infinity_constant_is_rejected() {
    let k = kernel(BF, "o[i] = (a[i] == (bfloat)INFINITY) ? bfloat(1.f) : bfloat(0.f);");
    if let Some((code, text)) = run_on("bf_inf", &k) {
        assert_eq!(code, 1, "bfloat 0xR7F80 slipped through: {text}");
    }
}

#[test]
fn known_gap_half_infinity_constant_is_rejected() {
    let k = kernel(HF, "o[i] = (a[i] == (half)INFINITY) ? 1.h : 0.h;");
    if let Some((code, text)) = run_on("half_inf", &k) {
        assert_eq!(code, 1, "half 0xH7C00 slipped through: {text}");
    }
}

#[test]
fn known_gap_signalling_nan_constant_is_rejected() {
    let k = kernel(F32, "o[i] = (a[i] > 0.f) ? as_type<float>(0x7fa00000u) : 0.f;");
    if let Some((code, text)) = run_on("snan", &k) {
        assert_eq!(code, 1, "float 0x7FF4000000000000 slipped through: {text}");
    }
}
