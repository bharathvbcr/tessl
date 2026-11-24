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

mod scratch_support;
use scratch_support::Scratch;
mod signal_support;

fn scratch(tag: &str) -> Scratch {
    Scratch::dir(&format!("tessl-bp-{tag}"))
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
fn the_label_is_written_to_the_json_and_absent_without_one() {
    let d = scratch("label");
    let a = metric_bin(&d, "a", "1");
    let b = metric_bin(&d, "b", "2");
    for (tag, label) in [("with", Some("42f1d52 against head")), ("without", None)] {
        let out_json = d.join(format!("{tag}.json"));
        let mut args = strs(&["--format", "metrics", "--rounds", "2", "--out"]);
        args.push(out_json.display().to_string());
        if let Some(l) = label {
            args.extend(strs(&["--label", l]));
        }
        args.push(format!("a={}", a.display()));
        args.push(format!("b={}", b.display()));
        let out = run(&args);
        assert!(out.status.success(), "{tag}: {}", stderr(&out));
        let text = fs::read_to_string(&out_json).unwrap();
        match label {
            Some(l) => assert!(text.contains(&format!("\"label\": \"{l}\"")), "{tag}: {text}"),
            None => assert!(!text.contains("\"label\""), "{tag}: {text}"),
        }
    }
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
    let root = scratch("unicode");
    let d = root.join("dir \u{fc} \u{65e5}\u{672c}");
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

/// An interrupted run settles its staged archive instead of leaving a
/// `.lanes.staging-*` directory behind, and still dies of the signal. The lane
/// signals that it is running by writing a file; the test waits for that file,
/// interrupts the `bench_paired` it spawned and waits for it to exit, so
/// nothing here depends on timing (the lane timeout only bounds a hang).
#[test]
fn an_interrupted_run_leaves_no_staging_dir_and_dies_of_the_signal() {
    signal_support::expect_ignored_if_rerun();
    use std::os::unix::process::ExitStatusExt;
    use std::time::{Duration, Instant};
    let d = scratch("interrupt");
    let started = d.join("started");
    let sleeper_pid = d.join("sleeper.pid");
    // The lane's own child outlives a plain kill of the lane, so its death is what
    // shows the interrupt ended the lane's whole group. The pid file is written
    // before `started`, which is what the test waits for.
    let hang = script(
        &d,
        "hang",
        &format!(
            "sleep 300 &\necho $! > '{}'\necho started > '{}'\nwait",
            sleeper_pid.display(),
            started.display()
        ),
    );
    let out_json = d.join("o.json");
    // SIGINT at its default whatever this harness inherited (see signal_support).
    let mut child = signal_support::default_signals(&mut Command::new(env!("CARGO_BIN_EXE_bench_paired")))
        .args(strs(&[
            "--format",
            "metrics",
            "--rounds=2",
            "--lane-timeout=120",
            &format!("--out={}", out_json.display()),
            &format!("ok={}", metric_bin(&d, "ok", "1").display()),
            &format!("hang={}", hang.display()),
        ]))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(100);
    while !started.exists() {
        if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the lane never started");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let sleeper: i32 = fs::read_to_string(&sleeper_pid).unwrap().trim().parse().unwrap();
    let sent = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(sent.success());
    let status = child.wait().unwrap();
    assert_eq!(status.signal(), Some(2), "the run did not die of SIGINT: {status:?}");
    // The group kill reaches the sleeper before bench_paired dies; launchd reaps it.
    let alive = |pid: i32| {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .unwrap()
            .success()
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(sleeper) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let survived = alive(sleeper);
    if survived {
        // The sleep this test's own lane script started.
        let _ = Command::new("kill").args(["-KILL", &sleeper.to_string()]).status();
    }
    assert!(!survived, "the lane's child {sleeper} survived the interrupt");
    let mut left: Vec<String> = fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("o.json"))
        .collect();
    left.sort();
    assert!(
        left.iter().all(|n| !n.contains(".staging-")),
        "a staging dir was left behind: {left:?}"
    );
    // No earlier JSON, so the interrupted run's lane output takes the final name.
    assert_eq!(left, ["o.json.lanes"], "{left:?}");
    assert!(
        fs::read_dir(d.join("o.json.lanes")).unwrap().count() > 0,
        "the interrupted run's lane output was not kept"
    );
}

fn signal_pid(pid: u32, name: &str) {
    let sent = Command::new("kill")
        .args([format!("-{name}"), pid.to_string()])
        .status()
        .unwrap();
    assert!(sent.success(), "kill -{name} {pid}");
}

/// Wait (bounded) until the lane has written its marker; fail if the run is already over.
fn wait_for_file(child: &mut std::process::Child, path: &Path, what: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(100);
    while !path.exists() {
        if std::time::Instant::now() > deadline || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{what}");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Wait (bounded) until the child this test spawned is in the stopped state: a
/// condition, not a delay. `waitpid` with `WUNTRACED` reports the stop to the parent;
/// `ps` would do too, but the hosted sandbox denies running it.
fn wait_stopped(pid: u32) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let mut status = 0;
        // SAFETY: `pid` is a child of this test process; `status` is a valid out-parameter.
        let r = unsafe { libc::waitpid(pid as libc::pid_t, &mut status, libc::WUNTRACED | libc::WNOHANG) };
        assert!(r >= 0, "waitpid {pid}: {}", std::io::Error::last_os_error());
        if r == pid as libc::pid_t {
            assert!(
                libc::WIFSTOPPED(status),
                "pid {pid} ended instead of stopping: status {status}"
            );
            return;
        }
        assert!(std::time::Instant::now() < deadline, "pid {pid} never stopped");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Wait (bounded) for the child this test spawned to end; kill it if it hangs.
fn wait_exit(child: &mut std::process::Child) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if let Some(s) = child.try_wait().unwrap() {
            return s;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("bench_paired did not exit after the signal");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn names_with_prefix(d: &Path, prefix: &str) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(prefix))
        .collect();
    v.sort();
    v
}

/// Start `bench_paired` with SIGINT, SIGQUIT and SIGHUP at their defaults, so a
/// signal the test sends acts as it would from a terminal (see signal_support).
fn spawn_bp(args: Vec<String>) -> std::process::Child {
    signal_support::default_signals(&mut Command::new(env!("CARGO_BIN_EXE_bench_paired")))
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap()
}

/// Owns the `bench_paired` a test stopped and its gate lane, and ends both when
/// dropped, on every path including a panic: a stopped process is not ended by its
/// parent dying, so without this a failing run leaves it (and the lane's shell
/// loop) behind for good. Signals only the pids this test spawned: the child, and
/// the lane pid its own gate script recorded.
struct Reaper {
    child: std::process::Child,
    lane_pid_file: PathBuf,
}

impl Drop for Reaper {
    fn drop(&mut self) {
        // Already reaped (the test ran to its end): nothing is left, and the pids
        // may be recycled. No assertions here: a panic in drop aborts.
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        let _ = Command::new("kill")
            .args(["-CONT", &self.child.id().to_string()])
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(pid) = fs::read_to_string(&self.lane_pid_file)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            let _ = Command::new("kill").args(["-KILL", &pid.to_string()]).status();
        }
    }
}

/// Signal "name" arrives while bench_paired is stopped and the gate lane's block_call-th run is held
/// until a release file exists (a file, not a FIFO: the hosted sandbox denies mkfifo); the lane is
/// released and has exited before bench_paired continues, so the signal is delivered after the
/// lane ended and before bench_paired has looked at it. No timing involved.
fn signal_after_gate_call_ended(
    tag: &str,
    block_call: u32,
    name: &str,
) -> (std::process::ExitStatus, PathBuf, Scratch) {
    let keep = scratch(tag);
    let d = keep.to_path_buf();
    let release = d.join("release");
    let up = d.join("up");
    let gate_pid = d.join("gate.pid");
    let gate_done = d.join("gate.done");
    let count = d.join("count");
    let gate = script(
        &d,
        "gate",
        &format!(
            "echo METRIC m 1\nn=$(cat '{c}' 2>/dev/null || echo 0)\nn=$((n+1))\necho $n > '{c}'\nif [ $n = {k} ]; then echo $$ > '{p}'; echo up > '{u}'; while [ ! -e '{f}' ]; do sleep 0.02; done; echo done > '{x}'; fi",
            c = count.display(),
            k = block_call,
            p = gate_pid.display(),
            u = up.display(),
            f = release.display(),
            x = gate_done.display()
        ),
    );
    let b_ran = d.join("b_ran");
    let b = script(&d, "b", &format!("echo ran > '{}'\necho METRIC m 2", b_ran.display()));
    let out_json = d.join("o.json");
    let mut reaper = Reaper {
        child: spawn_bp(strs(&[
            "--format",
            "metrics",
            "--rounds=2",
            "--lane-timeout=120",
            &format!("--out={}", out_json.display()),
            &format!("gate={}", gate.display()),
            &format!("b={}", b.display()),
        ])),
        lane_pid_file: gate_pid.clone(),
    };
    let child = &mut reaper.child;
    wait_for_file(child, &up, "the gating lane never reached its block");
    signal_pid(child.id(), "STOP");
    wait_stopped(child.id());
    signal_pid(child.id(), name);
    fs::write(&release, "").unwrap();
    // The lane's last act before it exits. Whether it has fully exited when the
    // signal lands does not change the outcome under test (a lane killed instead
    // of finished also ends the run without publishing), and `ps`, which would
    // show the exit, cannot run in the hosted sandbox.
    wait_for_file(&mut reaper.child, &gate_done, "the gating lane never finished");
    let child = &mut reaper.child;
    signal_pid(child.id(), "CONT");
    let status = wait_exit(child);
    (status, d, keep)
}

/// a signal that lands after the last lane ended and before the JSON is published.
#[test]
fn a_run_that_dies_of_a_signal_after_its_last_lane_publishes_no_result() {
    signal_support::expect_ignored_if_rerun();
    use std::os::unix::process::ExitStatusExt;
    let (status, d, _keep) = signal_after_gate_call_ended("sig_publish", 2, "INT");
    let left = names_with_prefix(&d, "o.json");
    assert!(left.iter().all(|n| !n.contains(".staging-")), "{left:?}");
    assert_eq!(status.signal(), Some(2), "{status:?}");
    assert!(
        !d.join("o.json").exists(),
        "the run died of SIGINT yet published its result: {left:?}"
    );
}

/// The two signal tests above do not depend on the SIGINT disposition this
/// harness inherited: run again with SIGINT, SIGQUIT and SIGHUP ignored on entry
/// (as under a background job, the cargo slot gate or `nohup`), both still pass.
#[test]
fn the_signal_tests_pass_when_the_harness_inherits_the_signals_ignored() {
    signal_support::rerun_with_signals_ignored(
        &[
            "an_interrupted_run_leaves_no_staging_dir_and_dies_of_the_signal",
            "a_run_that_dies_of_a_signal_after_its_last_lane_publishes_no_result",
        ],
        &[],
    );
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

#[test]
fn workload_lanes_carry_tensor_mib_into_the_report_and_leave_unlisted_fields_out() {
    let d = scratch("tensor-mib");
    let row = |ms: &str, tensor: &str| {
        format!("echo '[{{\"workload\":\"w\",\"ms_min\":{ms},\"tensor_mib\":{tensor},\"driver_mib\":9999.0}}]'")
    };
    let a = script(&d, "a", &row("2.0", "1024.5"));
    let b = script(&d, "b", &row("1.0", "512.25"));
    let json = d.join("report.json");
    let out = run(&strs(&[
        "--rounds",
        "2",
        "--out",
        json.to_str().unwrap(),
        &format!("a={}", a.display()),
        &format!("b={}", b.display()),
    ]));
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains("0.500x [0.500, 0.500]"), "{}", stdout(&out));
    let text: String = fs::read_to_string(&json).unwrap().split_whitespace().collect();
    assert!(text.contains("\"tensor_mib_max\":1024.5"), "{text}");
    assert!(text.contains("\"tensor_mib_max\":512.25"), "{text}");
    assert!(
        !text.contains("driver_mib_max"),
        "driver memory is recorded in the raw rows, not summarised: {text}"
    );
    assert!(
        text.contains("\"driver_mib\":9999.0"),
        "the raw row keeps the driver figure: {text}"
    );
}
