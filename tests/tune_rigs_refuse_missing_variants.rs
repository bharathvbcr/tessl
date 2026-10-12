//! The GEMM tuning binaries on a metallib without the tune kernels.
//!
//! `bench_gemm_tnnt_tune` and `bench_gemm_tile_tune` dispatch measurement
//! kernels that only a `TESSL_GEMM_TUNE=1` build links. Without them both used
//! to print `skip(pipe)` for every variant and exit 0, so a run that measured
//! nothing looked like a run that passed. Each must instead exit 2 before
//! timing anything and name the build command. Likewise a rig that flags a
//! drifting baseline (`EXCEEDS 10%`) must not then exit 0.

mod common;

use std::process::Command;

/// `TESSL_GEMM_TUNE` goes on the `cargo run` itself: it is a build-time input,
/// so naming only a `cargo build` sends the reader to a `cargo run` that
/// rebuilds the metallib without the variants.
const TUNE_RUN: &str = "TESSL_GEMM_TUNE=1 cargo run --release --bin";

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
    let command = format!("{TUNE_RUN} {bin}");
    assert!(
        stderr.contains(&command),
        "{bin} must name `{command}`:\n{stderr}"
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

/// Every bench binary that flags a baseline drifting past
/// `SELF_SPREAD_LIMIT` must end such a run with `exit_if_unstable` (exit 3).
/// `bench_gemm_routes` printed the flag and exited 0, so its results files
/// read as clean runs.
#[test]
fn every_rig_that_flags_drift_exits_3() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin");
    let mut flagging = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("src/bin") {
        let path = entry.expect("src/bin entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("bench source");
        if text.contains("SELF_SPREAD_LIMIT") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            assert!(
                text.contains("common::exit_if_unstable("),
                "{name} flags drift past SELF_SPREAD_LIMIT but never calls common::exit_if_unstable"
            );
            flagging.push(name);
        }
    }
    // A new rig that calls exit_if_unstable passes; the scan must still see
    // the three it was written for.
    for rig in [
        "bench_gemm_routes.rs",
        "bench_gemm_tile_tune.rs",
        "bench_gemm_tnnt_tune.rs",
    ] {
        assert!(
            flagging.iter().any(|f| f == rig),
            "the scan must see {rig}; it saw {flagging:?}"
        );
    }
}
