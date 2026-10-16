//! `bench_qwen35_train --op=` and `bench_embedgemma2 --attn-kernel=` refuse bad
//! input before the GPU runtime starts and before any `METRIC` line.

use std::process::Command;

fn assert_refused(bin: &str, envs: &[(&str, &str)], args: &[&str], needle: &str) {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .env("EMBEDGEMMA2_SNAPSHOT", "/nonexistent-tessl-snapshot");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().unwrap_or_else(|e| panic!("spawn {bin}: {e}"));
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(!out.status.success(), "{args:?} {envs:?} must fail:\n{stdout}");
    assert!(
        stderr.contains(needle),
        "{args:?} {envs:?}: stderr lacks {needle:?}:\n{stderr}"
    );
    assert!(
        !stdout.contains("device:"),
        "{args:?} started the GPU runtime:\n{stdout}"
    );
    assert!(
        !stdout.contains("METRIC"),
        "{args:?} printed a METRIC line before refusing:\n{stdout}"
    );
}

const TRAIN: &str = env!("CARGO_BIN_EXE_bench_qwen35_train");
const EG2: &str = env!("CARGO_BIN_EXE_bench_embedgemma2");

#[test]
fn train_op_filter_is_validated_before_the_runtime_in_every_mode() {
    for extra in [&[][..], &["--check-only"], &["--step-only"], &["--bf16"]] {
        let mut a = vec!["--op=no_such_op"];
        a.extend_from_slice(extra);
        assert_refused(TRAIN, &[], &a, "no such op");
        let mut a = vec!["--op=swiglu_bwd", "--op=swiglu_bwd"];
        a.extend_from_slice(extra);
        assert_refused(TRAIN, &[], &a, "given twice");
        // A repeat spelled two ways is still a repeat.
        let mut a = vec!["--op=attn_train forward", "--op=attn_train_forward"];
        a.extend_from_slice(extra);
        assert_refused(TRAIN, &[], &a, "given twice");
    }
    assert_refused(TRAIN, &[], &["--op="], "no such op");
}

#[test]
fn attn_kernel_refuses_bad_shapes_workloads_and_counts_before_any_metric() {
    let args = ["--attn-kernel=tiled"];
    assert_refused(EG2, &[("BENCH_ATTN_SHAPES", "64x32,bogus")], &args, "BENCH_ATTN_SHAPES");
    assert_refused(EG2, &[("BENCH_ATTN_SHAPES", "64x32,")], &args, "BENCH_ATTN_SHAPES");
    assert_refused(EG2, &[("BENCH_ATTN_SHAPES", "8x64-1024,8x64-1024")], &args, "repeats");
    assert_refused(EG2, &[("BENCH_WORKLOADS", "1x32")], &args, "BENCH_WORKLOADS");
    assert_refused(EG2, &[("BENCH_ITERS", "0")], &args, "BENCH_ITERS");
    assert_refused(EG2, &[("BENCH_ATTN_REPS", "x")], &args, "BENCH_ATTN_REPS");
    assert_refused(EG2, &[], &["--attn-kernel=bogus"], "--attn-kernel");
    assert_refused(EG2, &[], &["--attn-kernel=rows", "extra"], "unexpected argument");
}
