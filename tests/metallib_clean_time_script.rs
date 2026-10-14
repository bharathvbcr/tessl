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
    Command::new("bash")
        .arg(script())
        .arg(bs)
        .arg(manifest)
        .output()
        .unwrap()
}

/// A manifest whose `kernels/` holds `n` top-level `.metal` sources (plus one
/// in a subdirectory, which the build script does not compile).
fn manifest_with(dir: &Path, n: usize) -> PathBuf {
    let m = dir.join("man ifest");
    fs::create_dir_all(m.join("kernels/sub")).unwrap();
    fs::write(m.join("kernels/sub/inc.metal"), "").unwrap();
    for i in 0..n {
        fs::write(m.join(format!("kernels/k{i}.metal")), "").unwrap();
    }
    m
}

fn manifest(dir: &Path) -> PathBuf {
    manifest_with(dir, 35)
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
fn the_expected_air_count_follows_the_kernel_sources() {
    // Adding a kernel must not make a correct build fail the benchmark.
    let d = scratch("n36");
    let out = run(&fake_bs(&d, 36, 0, false, 0), &manifest_with(&d, 36));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // ...and a build that dropped a source still fails.
    let d = scratch("n36short");
    let out = run(&fake_bs(&d, 35, 0, false, 0), &manifest_with(&d, 36));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("expected 36 .air"));
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

#[test]
fn paired_bins_accepts_newline_separated_paths_with_spaces() {
    let d = scratch("pb");
    let mk = |name: &str, v: &str| {
        let p = d.join(name);
        fs::write(&p, format!("#!/bin/sh\necho METRIC m {v}\n")).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        p
    };
    let a = mk("bin a", "2");
    let b = mk("bin b", "1");
    let bins = format!("a={}\nb={}", a.display(), b.display());
    let log = d.join("out.log");
    let out = Command::new("bash")
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/paired_bins.sh"))
        .args(["2"])
        .arg(&log)
        .env("BENCH_BINS", bins)
        .output()
        .unwrap();
    let so = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{so}{}", String::from_utf8_lossy(&out.stderr));
    assert!(so.contains("a") && so.contains("b"), "{so}");
}

// ---- adversarial round (tester) ----

fn run_pb(bins: &str, rounds: &str, log: &Path) -> Output {
    Command::new("bash")
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/paired_bins.sh"))
        .arg(rounds)
        .arg(log)
        .env("BENCH_BINS", bins)
        .output()
        .unwrap()
}

fn mk_metric_bin(d: &Path, name: &str, v: &str) -> PathBuf {
    let p = d.join(name);
    fs::write(&p, format!("#!/bin/sh\necho METRIC m {v}\n")).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    p
}

#[test]
fn paired_bins_tolerates_trailing_newline_and_blank_lines() {
    let d = scratch("pbnl");
    let a = mk_metric_bin(&d, "bin a", "2");
    let b = mk_metric_bin(&d, "bin b", "1");
    let bins = format!("\na={}\n\n\nb={}\n\n", a.display(), b.display());
    let out = run_pb(&bins, "2", &d.join("o.log"));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // Exactly two labels: the ratio column only appears with exactly two bins.
    assert!(String::from_utf8_lossy(&out.stdout).contains("b/a"));
}

#[test]
fn paired_bins_single_spaced_path_with_trailing_newline_is_one_entry() {
    let d = scratch("pb1");
    let a = mk_metric_bin(&d, "bin a", "2");
    let out = run_pb(&format!("a={}\n", a.display()), "1", &d.join("o.log"));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn paired_bins_missing_binary_is_a_usage_error_not_a_metric() {
    let d = scratch("pbmiss");
    let a = mk_metric_bin(&d, "bin a", "2");
    let bins = format!("a={}\nb={}/nope", a.display(), d.display());
    let out = run_pb(&bins, "1", &d.join("o.log"));
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn paired_bins_whitespace_only_bins_is_rejected() {
    let d = scratch("pbws");
    let out = run_pb("   ", "1", &d.join("o.log"));
    assert_eq!(out.status.code(), Some(2), "no binaries must be a usage error");
    assert!(String::from_utf8_lossy(&out.stderr).contains("names no binaries"));
}

#[test]
fn paired_bins_a_binary_without_metrics_fails_the_run() {
    let d = scratch("pbnom");
    let p = d.join("quiet");
    fs::write(&p, "#!/bin/sh\necho hi\n").unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    let out = run_pb(&format!("q={}", p.display()), "1", &d.join("o.log"));
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn a_symlinked_kernel_source_counts_as_a_source() {
    // Sources reached through a symlink are still sources.
    let d = scratch("sym");
    let m = manifest_with(&d, 3);
    std::os::unix::fs::symlink(m.join("kernels/k0.metal"), m.join("kernels/link.metal")).unwrap();
    let out = run(&fake_bs(&d, 4, 0, false, 0), &m);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn zero_kernel_sources_is_an_error_with_no_metric() {
    let d = scratch("zero");
    let out = run(&fake_bs(&d, 0, 0, false, 0), &manifest_with(&d, 0));
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("METRIC"));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no top-level .metal"));
}

#[test]
fn a_directory_named_like_a_kernel_is_not_a_source() {
    let d = scratch("dirmetal");
    let m = manifest_with(&d, 3);
    fs::create_dir(m.join("kernels/fake.metal")).unwrap();
    let out = run(&fake_bs(&d, 3, 0, false, 0), &m);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn a_metallib_outside_the_scratch_out_dir_is_rejected() {
    let d = scratch("stale");
    let lib = d.join("stale.metallib");
    fs::write(&lib, "x").unwrap();
    let bs = d.join("fake bs");
    let body = format!(
        "#!/bin/sh\ni=0; while [ $i -lt 3 ]; do i=$((i+1)); : > \"$OUT_DIR/k$i.air\"; done\necho \"cargo:rustc-env=TESSL_METALLIB={}\"\n",
        lib.display()
    );
    fs::write(&bs, body).unwrap();
    fs::set_permissions(&bs, fs::Permissions::from_mode(0o755)).unwrap();
    let out = run(&bs, &manifest_with(&d, 3));
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("METRIC"));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not under the scratch OUT_DIR"));
}

#[test]
fn only_exact_dot_metal_names_count_as_sources() {
    // `.metal.bak` and a nested source are not top-level kernels.
    let d = scratch("names");
    let m = manifest_with(&d, 3);
    fs::write(m.join("kernels/k0.metal.bak"), "").unwrap();
    fs::write(m.join("kernels/notes.txt"), "").unwrap();
    let out = run(&fake_bs(&d, 3, 0, false, 0), &m);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn a_missing_kernels_dir_with_a_valid_build_is_a_usage_error() {
    let d = scratch("nokern");
    let m = d.join("m");
    fs::create_dir_all(&m).unwrap();
    let out = run(&fake_bs(&d, 3, 0, false, 0), &m);
    assert_eq!(out.status.code(), Some(2));
}

// ---- amendment round: ROUNDS validation and resolved-path containment ----

#[test]
fn paired_bins_rejects_a_rounds_that_is_not_a_positive_integer() {
    let d = scratch("rounds");
    let a = mk_metric_bin(&d, "a", "1");
    let bins = format!("a={}", a.display());
    let long = "9".repeat(10);
    for bad in ["0", "00", "-1", "", "abc", "1.5", "+2", " 2", "2 ", &long] {
        let out = run_pb(&bins, bad, &d.join("log"));
        assert_eq!(out.status.code(), Some(2), "ROUNDS={bad:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("ROUNDS"), "ROUNDS={bad:?}: {err}");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("summary"));
        assert!(!err.contains("round 0"), "ran despite ROUNDS={bad:?}");
    }
    let ok = run_pb(&bins, "2", &d.join("log"));
    assert_eq!(ok.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&ok.stdout).contains("summary"));
}

/// A fake build script that makes `$OUT_DIR/<setup>` then prints a metallib path.
fn escaping_bs(dir: &Path, setup: &str, lib_expr: &str) -> PathBuf {
    let bs = dir.join("fake bs");
    let body = format!(
        "#!/bin/sh\ni=0; while [ $i -lt 3 ]; do i=$((i+1)); : > \"$OUT_DIR/k$i.air\"; done\n\
         {setup}\necho \"cargo:rustc-env=TESSL_METALLIB={lib_expr}\"\n"
    );
    fs::write(&bs, body).unwrap();
    fs::set_permissions(&bs, fs::Permissions::from_mode(0o755)).unwrap();
    bs
}

fn assert_rejected_as_escape(out: &Output) {
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("METRIC"));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not under the scratch OUT_DIR"));
}

#[test]
fn a_dotdot_traversal_out_of_the_scratch_out_dir_is_rejected() {
    let d = scratch("dotdot");
    // OUT_DIR is <scratch>/o, so ../x.metallib is textually under it but lands beside it.
    let bs = escaping_bs(&d, "echo x > \"$OUT_DIR/../x.metallib\"", "$OUT_DIR/../o/../x.metallib");
    assert_rejected_as_escape(&run(&bs, &manifest_with(&d, 3)));
}

#[test]
fn a_symlinked_directory_inside_out_dir_pointing_outside_is_rejected() {
    let d = scratch("symdir");
    let outside = d.join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("x.metallib"), "x").unwrap();
    let bs = escaping_bs(
        &d,
        &format!("ln -s '{}' \"$OUT_DIR/link\"", outside.display()),
        "$OUT_DIR/link/x.metallib",
    );
    assert_rejected_as_escape(&run(&bs, &manifest_with(&d, 3)));
}

#[test]
fn a_symlinked_metallib_pointing_outside_is_rejected() {
    let d = scratch("symfile");
    let lib = d.join("outside.metallib");
    fs::write(&lib, "x").unwrap();
    let bs = escaping_bs(
        &d,
        &format!("ln -s '{}' \"$OUT_DIR/default.metallib\"", lib.display()),
        "$OUT_DIR/default.metallib",
    );
    assert_rejected_as_escape(&run(&bs, &manifest_with(&d, 3)));
}
