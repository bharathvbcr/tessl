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

// ---------------------------------------------------------------------------
// Adversarial-review regression tests. Tests marked `// pin a defect in the implementation as reviewed (commit 6f9d1f3): they FAIL
// today (`cargo test --test bench_paired_cli -- --ignored`) and should be
// un-ignored by the commit that fixes the defect.
// ---------------------------------------------------------------------------

fn counter_lane(dir: &Path, name: &str, body_after_count: &str) -> PathBuf {
    let cnt = dir.join(format!("{name}.cnt"));
    fs::write(&cnt, "0").unwrap();
    script(
        dir,
        name,
        &format!(
            "n=$(cat '{c}'); n=$((n+1)); echo $n > '{c}'\n{body_after_count}",
            c = cnt.display()
        ),
    )
}

#[test]
fn a_lane_killed_by_a_signal_fails_the_run_and_names_it() {
    let d = scratch("sigkill");
    let k = script(&d, "k", "echo METRIC m 1\nkill -9 $$");
    let ok = metric_bin(&d, "ok", "1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        &format!("k={}", k.display()),
        &format!("o={}", ok.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("signal"), "{}", stderr(&out));
    assert!(
        !stdout(&out).contains("vs base"),
        "a result was printed: {}",
        stdout(&out)
    );
}

#[test]
fn non_utf8_lane_output_is_refused() {
    let d = scratch("nonutf8");
    let bad = script(&d, "bad", "printf 'METRIC m 1\\n\\377\\376\\n'");
    let ok = metric_bin(&d, "ok", "1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        &format!("b={}", bad.display()),
        &format!("o={}", ok.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(!stdout(&out).contains("vs base"));
}

#[test]
fn a_metric_dropped_in_one_later_round_refuses_the_statistic() {
    let d = scratch("dropped");
    let flaky = counter_lane(
        &d,
        "flaky",
        "echo METRIC m 1\nif [ \"$n\" -ne 3 ]; then echo METRIC extra 1; fi",
    );
    let ok = script(&d, "ok", "echo METRIC m 1\necho METRIC extra 1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        "--rounds",
        "4",
        &format!("ok={}", ok.display()),
        &format!("f={}", flaky.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("extra") && stderr(&out).contains("round"),
        "{}",
        stderr(&out)
    );
    assert!(!stdout(&out).contains("vs base"), "partial data produced a number");
}

#[test]
fn a_spread_exactly_at_the_limit_passes_and_one_ulp_past_it_exits_3() {
    // Ratios b/a are 2.0 and 2.2 alternately: max/min is exactly 1.1.
    let run_with = |hi: &str| {
        let d = scratch(&format!("bound-{hi}"));
        let a = metric_bin(&d, "a", "1");
        let b = counter_lane(
            &d,
            "b",
            &format!("if [ $((n % 2)) -eq 1 ]; then echo METRIC m 2.0; else echo METRIC m {hi}; fi"),
        );
        run(&strs(&[
            "--format",
            "metrics",
            "--rounds",
            "4",
            "--max-spread",
            "1.10",
            &format!("a={}", a.display()),
            &format!("b={}", b.display()),
        ]))
    };
    let at = run_with("2.2");
    assert!(at.status.success(), "exactly at the limit must pass: {}", stderr(&at));
    let past = run_with("2.2000001");
    assert_eq!(past.status.code(), Some(3), "{}", stderr(&past));
    assert!(!stdout(&past).contains("vs base"));
}

#[test]
fn evidence_paths_with_spaces_and_unicode_publish_and_overwrite_atomically() {
    let d = scratch("unicode").join("dir \u{fc} \u{65e5}\u{672c}");
    fs::create_dir_all(&d).unwrap();
    let (a, b) = (metric_bin(&d, "a", "1"), metric_bin(&d, "b", "2"));
    let out_path = d.join("e v.json");
    for _ in 0..2 {
        let o = run(&strs(&[
            "--format",
            "metrics",
            "--evidence",
            "--rounds",
            "4",
            "--out",
            out_path.to_str().unwrap(),
            &format!("a=\"{}\"", a.display()),
            &format!("b='{}'", b.display()),
        ]));
        assert!(o.status.success(), "{}", stderr(&o));
    }
    let marker = fs::read_to_string(format!("{}.attempt.json", out_path.display())).unwrap();
    assert!(marker.contains("\"published\""), "{marker}");
    let leftovers: Vec<_> = fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".e v.json."))
        .collect();
    assert!(leftovers.is_empty(), "staging files left behind: {leftovers:?}");
}

#[test]
fn a_negative_metric_is_refused_not_ratioed() {
    let d = scratch("negmetric");
    let a = metric_bin(&d, "a", "-1");
    let b = metric_bin(&d, "b", "-2");
    let out = run(&strs(&[
        "--format",
        "metrics",
        &format!("a={}", a.display()),
        &format!("b={}", b.display()),
    ]));
    assert!(!out.status.success(), "reported {}", stdout(&out));
    assert!(!stdout(&out).contains("2.000x"));
}

#[test]
fn a_metric_line_with_trailing_tokens_is_malformed() {
    let d = scratch("trailing");
    let a = script(&d, "a", "echo METRIC m 1 2 3");
    let b = metric_bin(&d, "b", "2");
    let out = run(&strs(&[
        "--format",
        "metrics",
        &format!("a={}", a.display()),
        &format!("b={}", b.display()),
    ]));
    assert!(!out.status.success(), "accepted: {}", stdout(&out));
}

#[test]
fn a_hung_lane_is_bounded_by_a_timeout() {
    let d = scratch("hang");
    let hang = script(&d, "hang", "exec sleep 40");
    let ok = metric_bin(&d, "ok", "1");
    let mut child = Command::new(env!("CARGO_BIN_EXE_bench_paired"))
        .args(["--format", "metrics", "--lane-timeout", "2"])
        .arg(format!("h={}", hang.display()))
        .arg(format!("o={}", ok.display()))
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    let finished = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break Some(st);
        }
        if start.elapsed().as_secs() >= 15 {
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    if finished.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let st = finished.expect("bench_paired was still waiting on a hung lane after 15 s");
    assert_eq!(st.code(), Some(1));
}

#[test]
fn equals_style_options_are_options_or_usage_errors_never_lanes() {
    let d = scratch("eqopt");
    let (a, b) = (metric_bin(&d, "a", "1"), metric_bin(&d, "b", "2"));
    let lanes = [format!("a={}", a.display()), format!("b={}", b.display())];
    // The `=` form is an option, not a lane named `--rounds`.
    let o = run(&[strs(&["--format", "metrics", "--rounds=3"]), lanes.to_vec()].concat());
    assert!(o.status.success(), "--rounds=3: {}", stderr(&o));
    assert!(
        stderr(&o).contains("round 3/3 done"),
        "--rounds=3 was not read: {}",
        stderr(&o)
    );
    for bad in [
        "--rounds=0",
        "--rounds=",
        "--bogus=1",
        "--evidence=1",
        "--clean-env=yes",
    ] {
        let o = run(&[strs(&["--format", "metrics", bad]), lanes.to_vec()].concat());
        assert_eq!(o.status.code(), Some(2), "{bad}: {}", stderr(&o));
    }
    let out_path = d.join("o.json");
    let o = run(&[
        strs(&[
            "--format",
            "metrics",
            "--rounds",
            "2",
            &format!("--out={}", out_path.display()),
        ]),
        lanes.to_vec(),
    ]
    .concat());
    assert!(
        o.status.success() && out_path.exists(),
        "--out= ignored or misparsed: {}",
        stderr(&o)
    );
}

#[test]
fn evidence_refuses_a_lane_executable_that_changes_between_rounds() {
    let d = scratch("swap");
    let swap = d.join("swap");
    fs::write(&swap, "0").unwrap();
    let body = format!(
        "n=$(cat '{c}'); n=$((n+1)); echo $n > '{c}'\n\
         if [ $n -eq 3 ]; then printf '#!/bin/sh\\necho METRIC m 1\\n# swapped\\n' > '{s}'; fi\necho METRIC m 1",
        c = d.join("swap.cnt").display(),
        s = swap.display()
    );
    fs::write(d.join("swap.cnt"), "0").unwrap();
    script(&d, "swap", &body);
    let ok = metric_bin(&d, "ok", "2");
    let out_path = d.join("ev.json");
    let o = run(&strs(&[
        "--format",
        "metrics",
        "--evidence",
        "--rounds",
        "4",
        "--out",
        out_path.to_str().unwrap(),
        &format!("a={}", swap.display()),
        &format!("b={}", ok.display()),
    ]));
    assert!(
        !o.status.success() && !out_path.exists(),
        "published evidence for a binary that changed mid-run"
    );
}

#[test]
fn a_timed_out_lane_is_killed_with_its_children_and_named_with_its_round() {
    let d = scratch("timeout");
    let pidfile = d.join("child.pid");
    // The lane's own child outlives a plain kill of the lane; the group kill reaps it.
    let hang = script(
        &d,
        "hang",
        &format!("sleep 60 &\necho $! > '{}'\nwait", pidfile.display()),
    );
    let ok = metric_bin(&d, "ok", "1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        "--lane-timeout=1",
        &format!("ok={}", ok.display()),
        &format!("hang={}", hang.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("lane hang") && err.contains("timed out after 1s") && err.contains("round 1/5"),
        "{err}"
    );
    assert!(stdout(&out).is_empty(), "a result was printed");
    let pid: i32 = fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
    let alive = Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .unwrap()
        .success();
    assert!(!alive, "the lane's child {pid} survived the timeout");
}

#[test]
fn a_malformed_evidence_command_line_leaves_no_marker() {
    let d = scratch("nomarker");
    let (a, b) = (metric_bin(&d, "a", "1"), metric_bin(&d, "b", "1"));
    let out_json = d.join("o.json");
    for bad in [
        vec!["--bogus"],
        vec!["--max-spread=9"],
        vec!["--rounds=x"],
        vec!["--evidence=1"],
    ] {
        let mut args = strs(&[
            "--format",
            "metrics",
            "--evidence",
            &format!("--out={}", out_json.display()),
        ]);
        args.extend(strs(&bad));
        args.push(format!("a={}", a.display()));
        args.push(format!("b={}", b.display()));
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{bad:?}: {}", stderr(&o));
        assert!(!d.join("o.json.attempt.json").exists(), "{bad:?} wrote a marker");
    }
}

#[test]
fn the_out_equals_form_marks_the_older_output_stale_like_the_spaced_form() {
    let d = scratch("eqmarker");
    let (a, b) = (metric_bin(&d, "a", "1"), metric_bin(&d, "b", "1"));
    let out_json = d.join("o.json");
    fs::write(&out_json, "{\"generation\":\"old\"}").unwrap();
    let o = run(&strs(&[
        "--format",
        "metrics",
        "--evidence",
        "--rounds=3",
        &format!("--out={}", out_json.display()),
        &format!("a={}", a.display()),
        &format!("b={}", b.display()),
    ]));
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    let marker = fs::read_to_string(d.join("o.json.attempt.json")).unwrap();
    assert!(marker.contains("not_published"), "{marker}");
    assert_eq!(fs::read_to_string(&out_json).unwrap(), "{\"generation\":\"old\"}");
}

#[test]
fn out_keeps_every_lane_process_output_load_and_timestamp_per_round() {
    let d = scratch("archive");
    let noisy = script(
        &d,
        "noisy",
        "echo METRIC m 2\necho to-stdout-marker\necho to-stderr-marker >&2",
    );
    let ok = metric_bin(&d, "ok", "1");
    let out_json = d.join("o.json");
    let o = run(&strs(&[
        "--format",
        "metrics",
        "--rounds",
        "2",
        "--out",
        out_json.to_str().unwrap(),
        &format!("base={}", ok.display()),
        &format!("odd name={}", noisy.display()),
    ]));
    assert!(o.status.success(), "{}", stderr(&o));
    let lanes = d.join("o.json.lanes");
    let mut names: Vec<String> = fs::read_dir(&lanes)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    // 2 rounds x 2 lanes x (stdout, stderr, meta).
    assert_eq!(names.len(), 12, "{names:?}");
    assert!(names.contains(&"r02-1-odd_name.stdout".to_string()), "{names:?}");
    let out1 = fs::read_to_string(lanes.join("r01-1-odd_name.stdout")).unwrap();
    assert!(
        out1.contains("METRIC m 2") && out1.contains("to-stdout-marker"),
        "{out1}"
    );
    assert!(fs::read_to_string(lanes.join("r01-1-odd_name.stderr"))
        .unwrap()
        .contains("to-stderr-marker"));
    let meta = fs::read_to_string(lanes.join("r02-0-base.meta.json")).unwrap();
    for needle in [
        "started_at_utc",
        "finished_at_utc",
        "loadavg_1m",
        "\"exit_code\": 0",
        "\"round\": 2",
    ] {
        assert!(meta.contains(needle), "meta lacks {needle}: {meta}");
    }
    assert!(fs::read_to_string(&out_json).unwrap().contains("lane_archive"));
    // A failing run still leaves what ran.
    let bad = script(&d, "bad", "echo partial-output; echo boom >&2; exit 4");
    let out2 = d.join("o2.json");
    let o = run(&strs(&[
        "--format",
        "metrics",
        "--out",
        out2.to_str().unwrap(),
        &format!("base={}", ok.display()),
        &format!("bad={}", bad.display()),
    ]));
    assert_eq!(o.status.code(), Some(1));
    assert!(!out2.exists());
    let stderr_file = fs::read_to_string(d.join("o2.json.lanes/r01-1-bad.stderr")).unwrap();
    assert!(stderr_file.contains("boom"), "{stderr_file}");
}

#[test]
fn a_metric_dropped_in_round_two_fails_at_the_end_of_round_two() {
    let d = scratch("failfast");
    let flaky = counter_lane(
        &d,
        "flaky",
        "echo METRIC m 1\nif [ \"$n\" -ne 2 ]; then echo METRIC extra 1; fi",
    );
    let base = script(&d, "base", "echo METRIC m 1\necho METRIC extra 1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        "--rounds",
        "6",
        &format!("base={}", base.display()),
        &format!("f={}", flaky.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("round 1/6 done") && err.contains("\"extra\"") && err.contains("round 2"),
        "{err}"
    );
    assert!(
        !err.contains("round 2/6 done") && !err.contains("round 3/6 done"),
        "kept running: {err}"
    );
    // The lane ran twice, not six times.
    assert_eq!(fs::read_to_string(d.join("flaky.cnt")).unwrap().trim(), "2");
}

#[test]
fn an_unparsable_lane_fails_in_the_round_it_prints_it() {
    let d = scratch("failfast-parse");
    let bad = counter_lane(
        &d,
        "bad",
        "if [ \"$n\" -eq 2 ]; then echo METRIC m oops; else echo METRIC m 1; fi",
    );
    let ok = metric_bin(&d, "ok", "1");
    let out = run(&strs(&[
        "--format",
        "metrics",
        "--rounds",
        "6",
        &format!("ok={}", ok.display()),
        &format!("b={}", bad.display()),
    ]));
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert_eq!(fs::read_to_string(d.join("bad.cnt")).unwrap().trim(), "2");
}

#[test]
fn counts_are_one_strict_bounded_grammar_for_rounds_and_bench_iters() {
    let d = scratch("counts");
    let (a, b) = (metric_bin(&d, "a", "1"), metric_bin(&d, "b", "1"));
    let lanes = |iters: &str| {
        strs(&[
            "--format",
            "metrics",
            "--evidence",
            "--out",
            d.join("o.json").to_str().unwrap(),
            &format!("a:BENCH_ITERS={iters}={}", a.display()),
            &format!("b={}", b.display()),
        ])
    };
    for bad in ["18446744073709551615", "1000000", "+5", "005"] {
        let o = run(&lanes(bad));
        assert_eq!(o.status.code(), Some(2), "BENCH_ITERS={bad:?}: {}", stderr(&o));
        assert!(stderr(&o).contains("BENCH_ITERS"), "{}", stderr(&o));
    }
    for bad in ["18446744073709551615", "+4", " 4"] {
        let o = run(&strs(&[
            "--format",
            "metrics",
            "--rounds",
            bad,
            &format!("a={}", a.display()),
            &format!("b={}", b.display()),
        ]));
        assert_eq!(o.status.code(), Some(2), "--rounds {bad:?}");
    }
}
