//! A malformed benchmark count must stop the run, not change it.
//!
//! `BENCH_ITERS=abc` used to fall through `.ok().and_then(parse).unwrap_or(50)`
//! and run 50 iterations while the log claimed whatever the operator typed, and
//! `BENCH_ITERS=0` panicked inside `median` on an empty sample set. Both now go
//! through `common::env_usize`, which names the variable and exits non-zero
//! before the GPU is touched.

use std::process::Command;

fn assert_refuses(exe: &str, var: &str, value: &str) {
    let out = Command::new(exe)
        .env("BENCH_SHAPES", "64x64x64")
        .env(var, value)
        .output()
        .unwrap_or_else(|e| panic!("spawn {exe}: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{var}={value} must fail {exe}, got {:?}:\n{}\n{stderr}",
        out.status,
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(stderr.contains(var), "{exe} must name {var} in its error:\n{stderr}");
}

#[test]
fn benches_refuse_malformed_counts() {
    for exe in [
        env!("CARGO_BIN_EXE_bench_gemm_sweep"),
        env!("CARGO_BIN_EXE_bench_nn_kernels"),
    ] {
        for var in ["BENCH_ITERS", "BENCH_WARMUP"] {
            assert_refuses(exe, var, "abc");
        }
        assert_refuses(exe, "BENCH_ITERS", "0");
    }
}

#[test]
fn tile_bench_refuses_an_unknown_family() {
    let out = Command::new(env!("CARGO_BIN_EXE_bench_gemm_coop_tile"))
        .args(["--family", "bogus"])
        .output()
        .expect("spawn bench_gemm_coop_tile");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("plain or epilogue"));
}
