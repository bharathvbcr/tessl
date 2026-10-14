//! `bench/metallib_clean_time.sh` is the benchmark behind the clean-build
//! numbers in `bench/results/build_time_m5pro.txt`. These tests drive it with
//! fake build scripts (no GPU, no Metal toolchain) to pin that a malformed run
//! fails loudly instead of printing a METRIC line that looks like a result.
//! The script reads `sysctl vm.loadavg`, so the tests are macOS-only.
#![cfg(target_os = "macos")]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/metallib_clean_time.sh")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-mct-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A fake build script: writes `airs` .air files, `keys` .air.key files and a
/// metallib (unless `no_lib`), then prints the TESSL_METALLIB line.
fn fake_bs(dir: &Path, airs: usize, keys: usize, no_lib: bool, exit: i32) -> PathBuf {
    let p = dir.join("fake bs");
    let body = format!(
        "#!/bin/sh\ni=0; while [ $i -lt {airs} ]; do i=$((i+1)); : > \"$OUT_DIR/k$i.air\"; done\n\
         i=0; while [ $i -lt {keys} ]; do i=$((i+1)); : > \"$OUT_DIR/k$i.air.key\"; done\n\
         {lib}exit {exit}\n",
        lib = if no_lib {
            String::new()
        } else {
            "echo x > \"$OUT_DIR/default.metallib\"\necho \"cargo:rustc-env=TESSL_METALLIB=$OUT_DIR/default.metallib\"\n"
                .to_string()
        },
    );
    fs::write(&p, body).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn run(bs: &Path, manifest: &Path) -> Output {
    Command::new("bash").arg(script()).arg(bs).arg(manifest).output().unwrap()
}

fn manifest(dir: &Path) -> PathBuf {
    let m = dir.join("man ifest");
    fs::create_dir_all(m.join("kernels")).unwrap();
    m
}

#[test]
fn a_well_formed_run_prints_the_metrics_even_with_spaces_in_paths() {
    let d = scratch("ok");
    let out = run(&fake_bs(&d, 35, 0, false, 0), &manifest(&d));
    let so = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(so.contains("METRIC metallib_clean_s "), "{so}");
    assert!(so.contains("METRIC air_keys 0"), "{so}");
    let d2 = scratch("ok35");
    let out = run(&fake_bs(&d2, 35, 35, false, 0), &manifest(&d2));
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("METRIC air_keys 35"));
}

#[test]
fn malformed_runs_exit_nonzero_and_print_no_metric() {
    for (tag, airs, keys, no_lib, exit) in [
        ("few", 34, 0, false, 0),
        ("many", 36, 0, false, 0),
        ("halfkeys", 35, 10, false, 0),
        ("nolib", 35, 0, true, 0),
        ("fail", 35, 0, false, 3),
    ] {
        let d = scratch(tag);
        let out = run(&fake_bs(&d, airs, keys, no_lib, exit), &manifest(&d));
        assert!(!out.status.success(), "{tag}: must fail");
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("METRIC"),
            "{tag}: a failed run must not print METRIC lines"
        );
    }
}

#[test]
fn bad_arguments_are_a_usage_error() {
    let d = scratch("args");
    let bs = fake_bs(&d, 35, 0, false, 0);
    let st = Command::new("bash").arg(script()).status().unwrap();
    assert_eq!(st.code(), Some(2));
    let out = run(&bs, &d.join("no-such-manifest"));
    assert_eq!(out.status.code(), Some(2));
    let out = run(&d.join("absent"), &manifest(&d));
    assert_eq!(out.status.code(), Some(2));
}
