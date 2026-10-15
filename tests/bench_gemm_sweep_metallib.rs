//! `bench_gemm_sweep --dump-metallib PATH`: how `bench/kernel_coverage.py`
//! gets the metallib the benchmarks run. The runtime embeds the library's bytes
//! (`include_bytes!`), so no binary carries a path to find. The dump must be
//! the build artifact's bytes and must not need a GPU.

use std::process::Command;
use std::time::{Duration, Instant};

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

/// Run `--dump-metallib target` and fail the test if it has not exited within
/// ten seconds: a dump into a FIFO used to block forever in `open`.
fn dump_with_deadline(target: &std::path::Path) -> std::process::Output {
    let mut child = sweep()
        .arg("--dump-metallib")
        .arg(target)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() > deadline {
            let _ = child.kill(); // our own child, by handle
            let _ = child.wait();
            panic!(
                "--dump-metallib {} did not exit: it blocked on a non-regular target",
                target.display()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tessl-dump-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn dump_metallib_refuses_a_fifo_target_without_blocking() {
    let dir = scratch("fifo");
    let fifo = dir.join("lib.fifo");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0, "mkfifo");
    let out = dump_with_deadline(&fifo);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a regular file"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dump_metallib_refuses_a_directory_target() {
    let dir = scratch("dir");
    let out = dump_with_deadline(&dir);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a regular file"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dump_metallib_replaces_a_regular_file_atomically_and_leaves_no_staging_file() {
    let dir = scratch("atomic");
    let target = dir.join("lib.metallib");
    std::fs::write(&target, b"old contents").unwrap();
    let out = dump_with_deadline(&target);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        std::fs::read(&target).unwrap(),
        std::fs::read(tessl::metallib_path()).unwrap()
    );
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "staging file left behind: {names:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
