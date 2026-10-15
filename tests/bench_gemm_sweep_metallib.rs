//! `bench_gemm_sweep --dump-metallib PATH`: how `bench/kernel_coverage.py`
//! gets the metallib the benchmarks run. The runtime embeds the library's bytes
//! (`include_bytes!`), so no binary carries a path to find. The dump must be
//! the build artifact's bytes and must not need a GPU.

use std::process::Command;

fn sweep() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bench_gemm_sweep"));
    // Were the flag ignored, a tiny sweep rather than the default ladder runs.
    cmd.env("BENCH_SHAPES", "16x16x16")
        .env("BENCH_ITERS", "1")
        .env("BENCH_WARMUP", "0");
    cmd
}

#[test]
fn dump_metallib_writes_the_embedded_library_bytes() {
    let dir = std::env::temp_dir().join(format!("tessl-dump-metallib-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("embedded.metallib");
    let status = sweep().arg("--dump-metallib").arg(&out).output().unwrap();
    assert!(
        status.status.success(),
        "dump failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let dumped = std::fs::read(&out).expect("--dump-metallib wrote no file");
    let artifact = std::fs::read(tessl::metallib_path()).expect("build artifact");
    assert!(!dumped.is_empty() && dumped.starts_with(b"MTLB"), "not a metallib");
    assert_eq!(dumped, artifact, "the dump is not the library build.rs produced");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dump_metallib_without_a_path_fails_loudly() {
    let out = sweep().arg("--dump-metallib").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs a PATH"));
}
