//! `bench_paired` end to end, on fake lanes (shell scripts), no GPU.
//!
//! These carry the properties the retired `paired_bins.sh` tests pinned
//! (paths with spaces, a lane that prints no metric fails the run, a bad round
//! count is a usage error before anything runs) and the evidence contract the
//! retired `paired_cross_runtime.py` / `attn_paired.py` drivers had (stale
//! marker, pre-launch rejection, drift gate, whole-ladder coverage).

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tessl-bp-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn metric_bin(dir: &Path, name: &str, v: &str) -> PathBuf {
    script(dir, name, &format!("echo METRIC m {v}"))
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bench_paired"))
        .args(args)
        .output()
        .unwrap()
}

fn strs(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn metric_lanes_accept_quoted_paths_with_spaces_and_report_the_ratio() {
    let d = scratch("spaces");
    let a = metric_bin(&d, "bin a", "2");
    let b = metric_bin(&d, "bin b", "1");
    let out = run(&[
        "--format".into(),
        "metrics".into(),
        "--rounds".into(),
        "2".into(),
        format!("a=\"{}\"", a.display()),
        format!("b='{}'", b.display()),
    ]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("0.500x [0.500, 0.500]"), "{so}");
}

#[test]
fn a_lane_without_metrics_fails_the_run() {
    let d = scratch("nometric");
    let quiet = script(&d, "quiet", "echo hi");
    let ok = metric_bin(&d, "ok", "1");
    let out = run(&[
        "--format".into(),
        "metrics".into(),
        format!("q={}", quiet.display()),
        format!("o={}", ok.display()),
    ]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("no METRIC lines"), "{}", stderr(&out));
}

#[test]
fn a_missing_lane_binary_is_a_usage_error_not_a_metric() {
    let d = scratch("missing");
    let a = metric_bin(&d, "a", "2");
    let out = run(&[
        "--format".into(),
        "metrics".into(),
        format!("a={}", a.display()),
        format!("b={}/nope", d.display()),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        !stderr(&out).contains("round 1/"),
        "a lane ran despite the missing binary"
    );
}

#[test]
fn a_bad_round_count_is_a_usage_error_before_any_lane_runs() {
    let d = scratch("rounds");
    let a = metric_bin(&d, "a", "1");
    let b = metric_bin(&d, "b", "1");
    let long = "9".repeat(10);
    for bad in [
        "0", "1", "00", "-1", "", "abc", "1.5", "+2", " 2", "2 ", "007", "0x10", "1000000", &long, "1\n2",
    ] {
        let out = run(&[
            "--format".into(),
            "metrics".into(),
            "--rounds".into(),
            bad.to_string(),
            format!("a={}", a.display()),
            format!("b={}", b.display()),
        ]);
        assert_eq!(out.status.code(), Some(2), "--rounds {bad:?}: {}", stderr(&out));
        assert!(
            stdout(&out).is_empty() && !stderr(&out).contains("round 1/"),
            "ran despite --rounds {bad:?}"
        );
    }
}

#[test]
fn per_lane_environment_reaches_only_its_lane_and_an_empty_value_unsets() {
    let d = scratch("env");
    let echo = script(&d, "echo", "echo METRIC knob ${KNOB:-1}; echo METRIC gone ${GONE:-7}");
    let out = run(&[
        "--format".into(),
        "metrics".into(),
        "--rounds".into(),
        "2".into(),
        format!("base={}", echo.display()),
        format!("tuned:KNOB=4,GONE==\"{}\"", echo.display()),
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    let so = stdout(&out);
    let knob = so.lines().find(|l| l.trim_start().starts_with("knob")).unwrap();
    assert!(knob.contains("4.000x"), "{knob}");
    let gone = so.lines().find(|l| l.trim_start().starts_with("gone")).unwrap();
    assert!(gone.contains("1.000x"), "{gone}");
}

#[test]
fn clean_env_drops_inherited_bench_variables_but_not_the_lane_line() {
    let d = scratch("clean");
    let echo = script(&d, "echo", "echo METRIC iters ${BENCH_ITERS:-1}");
    let go = |extra: &[&str], inherited: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_bench_paired"));
        c.args(["--format", "metrics", "--rounds", "2"]).args(extra);
        c.arg(format!("a={}", echo.display()));
        c.arg(format!("b:BENCH_ITERS=2={}", echo.display()));
        if inherited {
            c.env("BENCH_ITERS", "8");
        }
        c.output().unwrap()
    };
    // Inherited BENCH_ITERS=8 reaches lane a; the lane line overrides it for b.
    assert!(stdout(&go(&[], true)).contains("0.250x"), "inherited 8 against 2");
    // With --clean-env lane a sees nothing, so the default 1 is used.
    assert!(stdout(&go(&["--clean-env"], true)).contains("2.000x"));
}

#[test]
fn a_drifting_ratio_past_max_spread_exits_3_and_publishes_nothing() {
    let d = scratch("drift");
    let counter = d.join("n");
    fs::write(&counter, "0").unwrap();
    // The second lane's value doubles every round it runs: ratios 1, 2, 4, 8.
    let base = metric_bin(&d, "base", "1");
    let grow = script(
        &d,
        "grow",
        &format!(
            "n=$(cat '{c}'); echo $((n+1)) > '{c}'; v=1; i=0; while [ $i -lt $n ]; do v=$((v*2)); i=$((i+1)); done; echo METRIC m $v",
            c = counter.display()
        ),
    );
    let out_json = d.join("o.json");
    let out = run(&[
        "--format".into(),
        "metrics".into(),
        "--rounds".into(),
        "4".into(),
        "--max-spread".into(),
        "1.10".into(),
        "--out".into(),
        out_json.display().to_string(),
        format!("base={}", base.display()),
        format!("grow={}", grow.display()),
    ]);
    assert_eq!(out.status.code(), Some(3), "{}", stderr(&out));
    assert!(stderr(&out).contains("exceeds bounded limit"), "{}", stderr(&out));
    assert!(stdout(&out).is_empty(), "no number may be presented: {}", stdout(&out));
    assert!(!out_json.exists());
}

#[test]
fn evidence_marks_stale_and_rejects_bad_policy_before_launching_a_lane() {
    let d = scratch("evidence");
    let a = metric_bin(&d, "a", "1");
    let b = metric_bin(&d, "b", "1");
    for (tag, bad, needle) in [
        ("cap", &["--max-spread", "9"][..], "ratio-spread limit"),
        ("round3", &["--rounds", "3"][..], "even integer >= 4"),
        ("round5", &["--rounds", "5"][..], "even integer >= 4"),
        ("iters", &[][..], "BENCH_ITERS must be >= 3"),
    ] {
        let out_json = d.join(format!("{tag}.json"));
        fs::write(&out_json, "{\"generation\":\"old\"}").unwrap();
        let mut args = strs(&["--format", "metrics", "--evidence", "--out"]);
        args.push(out_json.display().to_string());
        args.extend(bad.iter().map(|s| s.to_string()));
        let (la, lb) = if tag == "iters" {
            (format!("a:BENCH_ITERS=2={}", a.display()), format!("b={}", b.display()))
        } else {
            (format!("a={}", a.display()), format!("b={}", b.display()))
        };
        args.push(la);
        args.push(lb);
        let out = run(&args);
        let text = format!("{}{}", stdout(&out), stderr(&out));
        let marker = fs::read_to_string(format!("{}.attempt.json", out_json.display())).unwrap();
        assert_ne!(out.status.code(), Some(0), "{tag}: {text}");
        assert!(marker.contains("\"status\": \"not_published\""), "{tag}: {marker}");
        assert_eq!(
            fs::read_to_string(&out_json).unwrap(),
            "{\"generation\":\"old\"}",
            "{tag}"
        );
        assert!(text.contains(needle), "{tag}: {text}");
        assert!(!text.contains("round 1/"), "{tag}: a lane launched: {text}");
    }
}

fn rows_lane(dir: &Path, name: &str, rows: &str) -> PathBuf {
    script(dir, name, &format!("cat <<'EOF'\n{rows}\nEOF"))
}

const ROWS_ARGS: [&str; 8] = [
    "--format",
    "rows:shape,backend,gflops",
    "--compare",
    "t:o",
    "--rounds",
    "4",
    "--keys",
    "s1,s2",
];

#[test]
fn rows_lanes_compare_variants_across_processes() {
    let d = scratch("rows");
    let t = rows_lane(
        &d,
        "t",
        "[{\"shape\":\"s1\",\"backend\":\"t\",\"gflops\":400},{\"shape\":\"s2\",\"backend\":\"t\",\"gflops\":300}]",
    );
    // Multi-line output, as the Python lanes print it.
    let o = rows_lane(
        &d,
        "o",
        "[\n {\"shape\":\"s1\",\"backend\":\"o\",\"gflops\":200},\n {\"shape\":\"s2\",\"backend\":\"o\",\"gflops\":100}\n]",
    );
    let out_json = d.join("rows.json");
    let mut args = strs(&ROWS_ARGS);
    args.extend(strs(&["--out", out_json.to_str().unwrap()]));
    args.push(format!("tessl={}", t.display()));
    args.push(format!("other={}", o.display()));
    let out = run(&args);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let so = stdout(&out);
    assert!(so.contains("2.000x") && so.contains("3.000x"), "{so}");
    assert!(so.contains("PAIRED GEOMEAN") && so.contains("2.449x"), "{so}");
    let doc = fs::read_to_string(&out_json).unwrap();
    assert!(doc.contains("\"comparisons\"") && doc.contains("\"paired_geomean\""));
}

#[test]
fn rows_lanes_refuse_a_ladder_with_a_dropped_shape() {
    let d = scratch("rowsgap");
    let t = rows_lane(
        &d,
        "t",
        "[{\"shape\":\"s1\",\"backend\":\"t\",\"gflops\":400},{\"shape\":\"s2\",\"backend\":\"t\",\"gflops\":300}]",
    );
    let o = rows_lane(&d, "o", "[{\"shape\":\"s1\",\"backend\":\"o\",\"gflops\":200}]");
    let mut args = strs(&ROWS_ARGS);
    args.push(format!("tessl={}", t.display()));
    args.push(format!("other={}", o.display()));
    let out = run(&args);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("s2/o") && err.contains("Refusing to report a geomean"),
        "{err}"
    );
    assert!(stdout(&out).is_empty());
}

#[test]
fn rows_lanes_refuse_mismatched_batching() {
    let d = scratch("rowsbatch");
    let t = rows_lane(
        &d,
        "t",
        "[{\"shape\":\"s1\",\"backend\":\"t\",\"gflops\":400,\"batched\":32}]",
    );
    let o = rows_lane(
        &d,
        "o",
        "[{\"shape\":\"s1\",\"backend\":\"o\",\"gflops\":200,\"batched\":1}]",
    );
    let mut args = strs(&ROWS_ARGS);
    args.extend(strs(&["--require", "batched=32"]));
    args.push(format!("tessl={}", t.display()));
    args.push(format!("other={}", o.display()));
    let out = run(&args);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("batched=1"), "{}", stderr(&out));
}

#[test]
fn evidence_publishes_provenance_atomically_and_hashes_every_lane_input() {
    let d = scratch("publish");
    let t = rows_lane(&d, "t", "[{\"shape\":\"s1\",\"backend\":\"t\",\"gflops\":300}]");
    let o = rows_lane(&d, "o", "[{\"shape\":\"s1\",\"backend\":\"o\",\"gflops\":100}]");
    let out_json = d.join("evidence.json");
    let out = run(&[
        "--format".into(),
        "rows:shape,backend,gflops".into(),
        "--compare".into(),
        "t:o".into(),
        "--keys".into(),
        "s1".into(),
        "--evidence".into(),
        "--clean-env".into(),
        "--out".into(),
        out_json.display().to_string(),
        format!("tessl:BENCH_ITERS=5={}", t.display()),
        format!("other:BENCH_ITERS=5={}", o.display()),
    ]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let doc = fs::read_to_string(&out_json).unwrap();
    for needle in [
        "\"provenance\"",
        "\"tracked_diff_sha256\"",
        "\"lane:tessl:t\"",
        "\"lane:other:o\"",
        "\"methodology\"",
        "\"max_ratio_spread\": 1.1",
        "\"rounds\": 6",
    ] {
        assert!(doc.contains(needle), "evidence lacks {needle}");
    }
    let marker = fs::read_to_string(format!("{}.attempt.json", out_json.display())).unwrap();
    assert!(marker.contains("\"status\": \"published\""), "{marker}");
}
