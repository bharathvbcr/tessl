//! Env semantics of the tri-state getters (wave 7, ft-9ac2a619, attack 4).
//!
//! Each case runs the getters in a fresh child process (this test binary,
//! re-invoked) so the process-global latches start unset and no test mutates
//! its own environment. The oracle is the pre-refactor logic from gp/wave6:
//! the first name whose value is exactly one of 1/true/TRUE/yes (on) or
//! 0/false/FALSE/no (off) decides; otherwise the default (false), except
//! `icb_coarse_ranges_enabled`, which then follows `icb_range_batch_enabled`,
//! and `coarse_barriers`, which then inherits `hazard_barriers`.

use std::process::Command;

fn parse(v: Option<&str>) -> Option<bool> {
    match v {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") => Some(true),
        Some("0") | Some("false") | Some("FALSE") | Some("no") => Some(false),
        _ => None,
    }
}

fn first(names: &[&str], env: &[(&str, &str)]) -> Option<bool> {
    names
        .iter()
        .find_map(|n| parse(env.iter().find(|(k, _)| k == n).map(|(_, v)| *v)))
}

const GETTERS: [&str; 10] = [
    "decode_icb",
    "freeze",
    "range_batch",
    "coarse_ranges",
    "execute",
    "triage",
    "pipelines",
    "smoke",
    "hazard",
    "coarse_barriers",
];

fn report() -> Vec<bool> {
    vec![
        tessl::decode_icb_enabled(),
        tessl::icb_freeze_binds_enabled(),
        tessl::icb_range_batch_enabled(),
        tessl::icb_coarse_ranges_enabled(),
        tessl::icb_execute_enabled(),
        tessl::icb_triage_enabled(),
        tessl::icb_pipelines_enabled(),
        tessl::icb_smoke_enabled(),
        tessl::ab_flags::hazard_barriers(),
        tessl::ab_flags::coarse_barriers(),
    ]
}

/// The child entry point; a no-op in the parent run.
#[test]
fn child_report() {
    if std::env::var("TESSL_ADV_CHILD").is_err() {
        return;
    }
    let r: Vec<String> = report().iter().map(|b| b.to_string()).collect();
    println!("REPORT {}", r.join(","));
}

fn run_child(env: &[(&str, &str)]) -> Vec<bool> {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "child_report", "--nocapture", "--test-threads=1"]);
    for var in [
        "TESSL_DECODE_ICB", "METAL_RUNTIME_DECODE_ICB", "GEMMA_METAL_DECODE_ICB",
        "TESSL_ICB_FREEZE_BINDS", "METAL_RUNTIME_ICB_FREEZE_BINDS", "GEMMA_METAL_ICB_FREEZE_BINDS",
        "TESSL_ICB_RANGE_BATCH", "METAL_RUNTIME_ICB_RANGE_BATCH", "GEMMA_METAL_ICB_RANGE_BATCH",
        "TESSL_ICB_COARSE_RANGES", "METAL_RUNTIME_ICB_COARSE_RANGES", "GEMMA_METAL_ICB_COARSE_RANGES",
        "TESSL_ICB_EXECUTE", "METAL_RUNTIME_ICB_EXECUTE", "GEMMA_METAL_ICB_EXECUTE",
        "TESSL_ICB_TRIAGE", "GEMMA_METAL_ICB_TRIAGE",
        "TESSL_ICB_PIPELINES", "METAL_RUNTIME_ICB_PIPELINES", "GEMMA_METAL_ICB_PIPELINES",
        "TESSL_ICB_SMOKE", "METAL_RUNTIME_ICB_SMOKE", "GEMMA_METAL_ICB_SMOKE",
        "TESSL_HAZARD_BARRIERS", "METAL_RUNTIME_HAZARD_BARRIERS", "METAL_NATIVE_HAZARD_BARRIERS",
        "TESSL_COARSE_BARRIERS", "GEMMA_METAL_COARSE_BARRIERS", "METAL_RUNTIME_COARSE_BARRIERS",
    ] {
        cmd.env_remove(var);
    }
    cmd.env("TESSL_ADV_CHILD", "1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn child");
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find_map(|l| l.split_once("REPORT ").map(|(_, r)| r))
        .unwrap_or_else(|| panic!("no REPORT in child output: {text}\n{}", String::from_utf8_lossy(&out.stderr)));
    line.split(',').map(|s| s == "true").collect()
}

const VALUES: [&str; 14] = [
    "", "0", "1", "true", "TRUE", "True", "yes", "no", "false", "FALSE", "2", "garbage", " 1", "on",
];

/// (getter index, names in precedence order, default).
fn table() -> Vec<(usize, Vec<&'static str>)> {
    vec![
        (0, vec!["TESSL_DECODE_ICB", "METAL_RUNTIME_DECODE_ICB", "GEMMA_METAL_DECODE_ICB"]),
        (1, vec!["TESSL_ICB_FREEZE_BINDS", "METAL_RUNTIME_ICB_FREEZE_BINDS", "GEMMA_METAL_ICB_FREEZE_BINDS"]),
        (2, vec!["TESSL_ICB_RANGE_BATCH", "METAL_RUNTIME_ICB_RANGE_BATCH", "GEMMA_METAL_ICB_RANGE_BATCH"]),
        (4, vec!["TESSL_ICB_EXECUTE", "METAL_RUNTIME_ICB_EXECUTE", "GEMMA_METAL_ICB_EXECUTE"]),
        (5, vec!["TESSL_ICB_TRIAGE", "GEMMA_METAL_ICB_TRIAGE"]),
        (6, vec!["TESSL_ICB_PIPELINES", "METAL_RUNTIME_ICB_PIPELINES", "GEMMA_METAL_ICB_PIPELINES"]),
        (7, vec!["TESSL_ICB_SMOKE", "METAL_RUNTIME_ICB_SMOKE", "GEMMA_METAL_ICB_SMOKE"]),
        (8, vec!["TESSL_HAZARD_BARRIERS", "METAL_RUNTIME_HAZARD_BARRIERS", "METAL_NATIVE_HAZARD_BARRIERS"]),
    ]
}

#[test]
fn every_getter_matches_the_old_env_semantics() {
    for (idx, names) in table() {
        for v in VALUES {
            // Each name alone, then the first name garbage with a later name set
            // (the first PARSING name wins), then two parsing names (first wins).
            let mut cases: Vec<Vec<(&str, &str)>> = names.iter().map(|n| vec![(*n, v)]).collect();
            cases.push(vec![(names[0], "garbage"), (names[1], v)]);
            cases.push(vec![(names[0], v), (names[1], "1")]);
            cases.push(vec![(names[0], v), (names[1], "0")]);
            for env in cases {
                let got = run_child(&env)[idx];
                // Both oracle paths are the same fold the old getters used.
                let want = first(&names, &env).unwrap_or(false);
                assert_eq!(got, want, "{}: env {env:?}", GETTERS[idx]);
            }
        }
    }
}

#[test]
fn coarse_ranges_follows_range_batch_only_when_its_env_does_not_parse() {
    let coarse = ["TESSL_ICB_COARSE_RANGES", "METAL_RUNTIME_ICB_COARSE_RANGES", "GEMMA_METAL_ICB_COARSE_RANGES"];
    for cv in VALUES {
        for rv in VALUES {
            for name in coarse {
                let env = [(name, cv), ("TESSL_ICB_RANGE_BATCH", rv)];
                let r = run_child(&env);
                let range = first(&["TESSL_ICB_RANGE_BATCH"], &env).unwrap_or(false);
                let want = first(&coarse, &env).unwrap_or(range);
                assert_eq!(r[3], want, "coarse_ranges: env {env:?}");
                assert_eq!(r[2], range, "range_batch: env {env:?}");
            }
        }
    }
}

#[test]
fn coarse_barriers_inherits_hazard_only_when_its_env_does_not_parse() {
    let coarse = ["TESSL_COARSE_BARRIERS", "GEMMA_METAL_COARSE_BARRIERS", "METAL_RUNTIME_COARSE_BARRIERS"];
    for cv in VALUES {
        for hv in VALUES {
            for name in coarse {
                let env = [(name, cv), ("TESSL_HAZARD_BARRIERS", hv)];
                let r = run_child(&env);
                let hazard = first(&["TESSL_HAZARD_BARRIERS"], &env).unwrap_or(false);
                let want = first(&coarse, &env).unwrap_or(hazard);
                assert_eq!(r[9], want, "coarse_barriers: env {env:?}");
            }
        }
    }
}
