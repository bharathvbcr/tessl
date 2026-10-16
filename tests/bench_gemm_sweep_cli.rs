//! `bench_gemm_sweep`'s argument and environment contract, with no device.
//!
//! Every case here either panicked or silently did the wrong thing before it
//! was pinned: a malformed count fell through to a default, `BENCH_SHAPES`
//! beside `--dump-parity` quietly produced an empty dump, an unknown
//! `BENCH_PARITY_SHAPE` scored a different shape than the one asked for.
//! `bench_gemm_sweep` rejects all of them before it initialises the GPU, so the
//! contract holds on a machine with no Metal device and the hosted job runs it.
//! (It used to be a GPU-gated section of the Python parity harness that skipped
//! itself wherever no device was present.)

use std::process::Command;

/// Variables a case may set; every other case must not inherit them from the
/// caller's shell, or a stale value would silently change what it tests.
const CONTROLLED: &[&str] = &[
    "BENCH_ITERS",
    "BENCH_WARMUP",
    "BENCH_SHAPES",
    "BENCH_PARITY_SEEDS",
    "BENCH_PARITY_SHAPE",
    "BENCH_PARITY_DIST",
];

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-sweep-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run one case and assert it fails cleanly (a plain non-zero exit, no panic)
/// with `needle` somewhere in its output.
fn refused(label: &str, argv: &[&str], env: &[(&str, &str)], needle: &str) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bench_gemm_sweep"));
    cmd.args(argv);
    for var in CONTROLLED {
        cmd.env_remove(var);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("{label}: spawn: {e}"));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !text.contains("panicked at") && out.status.code().is_some(),
        "{label}: panicked instead of erroring cleanly:\n{text}"
    );
    assert!(!out.status.success(), "{label}: exited 0, accepting bad input:\n{text}");
    assert!(
        text.to_lowercase().contains(&needle.to_lowercase()),
        "{label}: error lacks {needle:?}:\n{text}"
    );
}

#[test]
fn dump_parity_needs_a_directory() {
    refused(
        "--dump-parity with no directory",
        &["--dump-parity"],
        &[],
        "requires a directory",
    );
    refused(
        "--dump-parity followed by a flag",
        &["--dump-parity", "--x"],
        &[],
        "requires a directory",
    );
}

#[test]
fn malformed_counts_are_refused_before_the_gpu() {
    refused("BENCH_ITERS=0", &[], &[("BENCH_ITERS", "0")], "below the minimum");
    refused(
        "BENCH_ITERS=abc",
        &[],
        &[("BENCH_ITERS", "abc")],
        "not a non-negative integer",
    );
    refused(
        "BENCH_ITERS=-1",
        &[],
        &[("BENCH_ITERS", "-1")],
        "not a non-negative integer",
    );
    refused(
        "BENCH_WARMUP=xyz",
        &[],
        &[("BENCH_WARMUP", "xyz")],
        "not a non-negative integer",
    );
}

#[test]
fn malformed_shape_grids_name_the_entry() {
    refused(
        "BENCH_SHAPES=garbage",
        &[],
        &[("BENCH_SHAPES", "garbage")],
        "must be MxNxK",
    );
    refused("BENCH_SHAPES=64x64", &[], &[("BENCH_SHAPES", "64x64")], "must be MxNxK");
    refused(
        "BENCH_SHAPES with a zero dim",
        &[],
        &[("BENCH_SHAPES", "0x64x64")],
        "zero dimension",
    );
    refused(
        "BENCH_SHAPES=64xAx64",
        &[],
        &[("BENCH_SHAPES", "64xAx64")],
        "not an integer",
    );
}

#[test]
fn parity_dump_knobs_are_validated() {
    let tmp = scratch("knobs");
    let dir = tmp.to_str().unwrap();
    let dump = ["--dump-parity", dir];
    refused(
        "BENCH_PARITY_SEEDS=0",
        &dump,
        &[("BENCH_PARITY_SEEDS", "0")],
        "below the minimum",
    );
    refused(
        "BENCH_PARITY_SEEDS=abc",
        &dump,
        &[("BENCH_PARITY_SEEDS", "abc")],
        "not a non-negative integer",
    );
    refused(
        "BENCH_SHAPES alongside --dump-parity",
        &dump,
        &[("BENCH_SHAPES", "64x64x64")],
        "does not apply",
    );
    refused(
        "BENCH_PARITY_SHAPE=nonsense",
        &dump,
        &[("BENCH_PARITY_SHAPE", "nonsense")],
        "not a ladder label",
    );
    refused(
        "BENCH_PARITY_SHAPE=512x512x512 (dims, not a label)",
        &dump,
        &[("BENCH_PARITY_SHAPE", "512x512x512")],
        "not a ladder label",
    );
    refused(
        "BENCH_PARITY_SHAPE empty",
        &dump,
        &[("BENCH_PARITY_SHAPE", "")],
        "not a ladder label",
    );
    refused(
        "BENCH_PARITY_DIST=gaussian",
        &dump,
        &[("BENCH_PARITY_DIST", "gaussian")],
        "not a distribution",
    );
    refused(
        "BENCH_PARITY_DIST empty",
        &dump,
        &[("BENCH_PARITY_DIST", "")],
        "not a distribution",
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn a_refused_dump_leaves_no_manifest() {
    // The manifest is written last and only on success, so a dump that never
    // started cannot be scored as a complete one.
    let tmp = scratch("nomanifest");
    let out = Command::new(env!("CARGO_BIN_EXE_bench_gemm_sweep"))
        .args(["--dump-parity", tmp.to_str().unwrap()])
        .env("BENCH_PARITY_SHAPE", "nonsense")
        .env_remove("BENCH_SHAPES")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let left: Vec<_> = std::fs::read_dir(&tmp).unwrap().collect();
    assert!(left.is_empty(), "a refused dump wrote {left:?}");
    let _ = std::fs::remove_dir_all(&tmp);
}
