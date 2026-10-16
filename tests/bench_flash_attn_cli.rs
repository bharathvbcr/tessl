//! `bench_flash_attn`'s argument and environment contract.
//!
//! These are the cases of the retired `ATTN_CLI_CASES` /
//! `attn_cli_contract` in `bench/test_parity_harness.py`. That harness had to
//! probe whether a Metal device was available and skip the whole contract when
//! it was not, because the binary opened the device before it validated
//! `BENCH_ATTN_CFGS`. The bin now rejects every bad argument and environment
//! before it opens the device, so these cases run on a host with no GPU and a
//! skip is no longer a way for the contract to pass unchecked.
//!
//! Each case must exit nonzero (not by signal, not by panic), and say why.

use std::path::PathBuf;
use std::process::Command;

/// The variables a case may set; every one a case does not set is removed, so
/// the caller's own shell cannot change what is being tested.
const SCOPED: [&str; 12] = [
    "BENCH_ITERS",
    "BENCH_WARMUP",
    "BENCH_ATTN_DIST",
    "BENCH_ATTN_CFGS",
    "BENCH_ATTN_DECODE_CHUNK",
    "BENCH_ATTN_DECODE_R",
    "BENCH_ATTN_ROWS_R",
    "BENCH_ATTN_ROWS_SGT",
    "BENCH_ATTN_REDUCE_W",
    "BENCH_ATTN_DECODE_SGS",
    "BENCH_ATTN_BATCHED",
    "TESSL_ATTN_TILED",
];

fn refused_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-attn-refused-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Run one case and check it was refused for the stated reason.
fn refuses(label: &str, args: &[&str], env: &[(&str, &str)], needle: &str) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bench_flash_attn"));
    cmd.args(args);
    for var in SCOPED {
        cmd.env_remove(var);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let o = cmd.output().unwrap();
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stderr),
        String::from_utf8_lossy(&o.stdout)
    );
    let code = o.status.code();
    assert!(
        code.is_some() && !out.contains("panicked at"),
        "{label}: died instead of refusing: {code:?}\n{out}"
    );
    assert_ne!(code, Some(0), "{label}: exited 0, accepting bad input\n{out}");
    assert!(
        out.to_lowercase().contains(&needle.to_lowercase()),
        "{label}: the error does not say {needle:?}\n{out}"
    );
}

#[test]
fn dump_parity_needs_a_directory() {
    refuses(
        "--dump-parity with no directory",
        &["--dump-parity"],
        &[],
        "requires a directory",
    );
    // A following flag is not a directory.
    refuses(
        "--dump-parity followed by a flag",
        &["--dump-parity", "--other"],
        &[],
        "requires a directory",
    );
}

#[test]
fn iteration_counts_are_checked() {
    refuses("BENCH_ITERS=0", &[], &[("BENCH_ITERS", "0")], "below the minimum");
    refuses(
        "BENCH_ITERS=abc",
        &[],
        &[("BENCH_ITERS", "abc")],
        "not a non-negative integer",
    );
    refuses(
        "BENCH_WARMUP=abc",
        &[],
        &[("BENCH_WARMUP", "abc")],
        "not a non-negative integer",
    );
}

#[test]
fn the_distribution_is_checked() {
    refuses(
        "BENCH_ATTN_DIST=gaussian",
        &[],
        &[("BENCH_ATTN_DIST", "gaussian")],
        "not a distribution",
    );
}

/// The case that needed a device before: an unknown configuration name.
#[test]
fn configurations_are_checked_without_a_device() {
    refuses(
        "BENCH_ATTN_CFGS=nope",
        &[],
        &[("BENCH_ATTN_CFGS", "nope")],
        "not a configuration",
    );
    refuses(
        "BENCH_ATTN_CFGS=swa128_prefill_5120",
        &[],
        &[("BENCH_ATTN_CFGS", "swa128_prefill_5120")],
        "not a configuration",
    );
    refuses(
        "BENCH_ATTN_CFGS names one real and one unknown configuration",
        &[],
        &[("BENCH_ATTN_CFGS", "swa128_prefill_512,nope")],
        "not a configuration",
    );
    refuses(
        "BENCH_ATTN_CFGS=,",
        &[],
        &[("BENCH_ATTN_CFGS", " , ")],
        "names no configurations",
    );
}

/// A parity artifact must describe the shipping configuration. Under a tuning
/// override it would describe a kernel no caller reaches, which is how the
/// forced-decode lane came to be dumped at chunk 256 while the library shipped
/// 128. The four overrides the Python cases covered are here, and so are the
/// three it did not (`BENCH_ATTN_ROWS_SGT`, `BENCH_ATTN_REDUCE_W`,
/// `BENCH_ATTN_DECODE_SGS`) that the bin refuses by the same rule. The refusal
/// happens before the output directory is made.
#[test]
fn dump_parity_refuses_every_tuning_override() {
    let overrides = [
        ("BENCH_ATTN_DECODE_CHUNK", "64"),
        ("BENCH_ATTN_DECODE_R", "32"),
        ("BENCH_ATTN_ROWS_R", "32"),
        ("BENCH_ATTN_ROWS_SGT", "8"),
        ("BENCH_ATTN_REDUCE_W", "32"),
        ("BENCH_ATTN_DECODE_SGS", "all"),
        ("TESSL_ATTN_TILED", "1"),
    ];
    for (var, value) in overrides {
        let dir = refused_dir(var);
        let d = dir.to_str().unwrap();
        refuses(
            &format!("--dump-parity with {var}"),
            &["--dump-parity", d],
            &[(var, value)],
            "shipping configuration",
        );
        refuses(
            &format!("--dump-parity with {var} names the variable"),
            &["--dump-parity", d],
            &[(var, value)],
            var,
        );
        assert!(!dir.exists(), "{var}: the refusal came after the directory was made");
    }
}

/// The refusal points at the tool that sweeps tuning values, by its current name.
#[test]
fn the_refusal_names_the_attn_tune_bin() {
    refuses(
        "--dump-parity with BENCH_ATTN_ROWS_R names the sweep tool",
        &["--dump-parity", "unused"],
        &[("BENCH_ATTN_ROWS_R", "8")],
        "attn_tune",
    );
    let src = include_str!("../src/bin/bench_flash_attn.rs");
    assert!(
        !src.contains("attn_tune.py"),
        "bench_flash_attn.rs still names the retired script"
    );
}
