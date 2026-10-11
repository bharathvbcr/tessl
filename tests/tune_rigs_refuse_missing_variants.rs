//! The GEMM tuning binaries on a metallib without the tune kernels.
//!
//! `bench_gemm_tnnt_tune` and `bench_gemm_tile_tune` dispatch measurement
//! kernels that only a `TESSL_GEMM_TUNE=1` build links. Without them both used
//! to print `skip(pipe)` for every variant and exit 0, so a run that measured
//! nothing looked like a run that passed. Each must instead exit 2 before
//! timing anything and name the build command.

mod common;

use std::process::Command;

const TUNE_BUILD: &str = "TESSL_GEMM_TUNE=1 cargo build --release --bins";

/// Whether this build's metallib carries the tune kernels; one entry point
/// from `kernels/tune/matmul_tensorops_tune.metal` stands for the file.
fn tune_kernels_linked() -> bool {
    common::with_gpu(|rt| rt.pipeline("mm_bf16_coop_64x64_sg4").is_ok())
}

fn assert_refuses(bin: &str, exe: &str) {
    let out = Command::new(exe)
        .env("BENCH_WARMUP", "0")
        .env("BENCH_ITERS", "1")
        .env("BENCH_ROUNDS", "1")
        .output()
        .unwrap_or_else(|e| panic!("spawn {bin}: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{bin} without the tune kernels must exit 2, not {:?}:\n{stdout}\n{stderr}",
        out.status
    );
    assert!(
        stderr.contains(TUNE_BUILD),
        "{bin} must name the build command:\n{stderr}"
    );
    assert!(
        !stdout.contains("skip(pipe)"),
        "{bin} printed skip(pipe) lines:\n{stdout}"
    );
    assert!(
        !stdout.contains("production"),
        "{bin} timed something before refusing:\n{stdout}"
    );
}

#[test]
fn tuning_binaries_exit_2_without_the_tune_kernels() {
    if tune_kernels_linked() {
        // A TESSL_GEMM_TUNE=1 build: the refusal has nothing to refuse.
        eprintln!("tune kernels are linked in this build; the missing-variant refusal is not reachable");
        return;
    }
    assert_refuses("bench_gemm_tnnt_tune", env!("CARGO_BIN_EXE_bench_gemm_tnnt_tune"));
    assert_refuses("bench_gemm_tile_tune", env!("CARGO_BIN_EXE_bench_gemm_tile_tune"));
}
