//! Env semantics of the tri-state getters (wave 7, ft-9ac2a619, attack 4).
//!
//! Each case runs the getters in a fresh child process (this test binary,
//! re-invoked) so the process-global latches start unset and no test mutates
//! its own environment. The oracle is the pre-refactor logic from gp/wave6:
//! the first name whose value is exactly one of 1/true/TRUE/yes (on) or
//! 0/false/FALSE/no (off) decides; otherwise the default (false), except
//! `coarse_barriers`, which then inherits `hazard_barriers`. The retired
//! `GEMMA_METAL_*` / `METAL_NATIVE_*` spellings must not decide any getter.

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

const GETTERS: [&str; 7] = [
    "decode_icb",
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
        "TESSL_DECODE_ICB",
        "METAL_RUNTIME_DECODE_ICB",
        "TESSL_ICB_EXECUTE",
        "METAL_RUNTIME_ICB_EXECUTE",
        "TESSL_ICB_TRIAGE",
        "TESSL_ICB_PIPELINES",
        "METAL_RUNTIME_ICB_PIPELINES",
        "TESSL_ICB_SMOKE",
        "METAL_RUNTIME_ICB_SMOKE",
        "TESSL_HAZARD_BARRIERS",
        "METAL_RUNTIME_HAZARD_BARRIERS",
        "TESSL_COARSE_BARRIERS",
        "METAL_RUNTIME_COARSE_BARRIERS",
        // Retired aliases: still scrubbed so an ambient value cannot leak into
        // the child, and covered by `retired_aliases_decide_nothing`.
        "GEMMA_METAL_DECODE_ICB",
        "GEMMA_METAL_ICB_EXECUTE",
        "GEMMA_METAL_ICB_TRIAGE",
        "GEMMA_METAL_ICB_PIPELINES",
        "GEMMA_METAL_ICB_SMOKE",
        "GEMMA_METAL_COARSE_BARRIERS",
        "METAL_NATIVE_HAZARD_BARRIERS",
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
        .unwrap_or_else(|| {
            panic!(
                "no REPORT in child output: {text}\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    line.split(',').map(|s| s == "true").collect()
}

/// Aliases removed with the GEMMA_METAL_/METAL_NATIVE_ retirement, paired with
/// the getter index each one used to feed.
const RETIRED: [(usize, &str); 7] = [
    (0, "GEMMA_METAL_DECODE_ICB"),
    (1, "GEMMA_METAL_ICB_EXECUTE"),
    (2, "GEMMA_METAL_ICB_TRIAGE"),
    (3, "GEMMA_METAL_ICB_PIPELINES"),
    (4, "GEMMA_METAL_ICB_SMOKE"),
    (5, "METAL_NATIVE_HAZARD_BARRIERS"),
    (6, "GEMMA_METAL_COARSE_BARRIERS"),
];

const VALUES: [&str; 14] = [
    "", "0", "1", "true", "TRUE", "True", "yes", "no", "false", "FALSE", "2", "garbage", " 1", "on",
];

/// (getter index, names in precedence order). Getter 6 (`coarse_barriers`)
/// inherits hazard and is covered separately below.
fn table() -> Vec<(usize, Vec<&'static str>)> {
    vec![
        (0, vec!["TESSL_DECODE_ICB", "METAL_RUNTIME_DECODE_ICB"]),
        (1, vec!["TESSL_ICB_EXECUTE", "METAL_RUNTIME_ICB_EXECUTE"]),
        (2, vec!["TESSL_ICB_TRIAGE"]),
        (3, vec!["TESSL_ICB_PIPELINES", "METAL_RUNTIME_ICB_PIPELINES"]),
        (4, vec!["TESSL_ICB_SMOKE", "METAL_RUNTIME_ICB_SMOKE"]),
        (5, vec!["TESSL_HAZARD_BARRIERS", "METAL_RUNTIME_HAZARD_BARRIERS"]),
    ]
}

#[test]
fn every_getter_matches_the_old_env_semantics() {
    for (idx, names) in table() {
        for v in VALUES {
            // Each name alone, then the first name garbage with a later name set
            // (the first PARSING name wins), then two parsing names (first wins).
            let mut cases: Vec<Vec<(&str, &str)>> = names.iter().map(|n| vec![(*n, v)]).collect();
            if let Some(second) = names.get(1) {
                cases.push(vec![(names[0], "garbage"), (*second, v)]);
                cases.push(vec![(names[0], v), (*second, "1")]);
                cases.push(vec![(names[0], v), (*second, "0")]);
            }
            for env in cases {
                let got = run_child(&env)[idx];
                // Both oracle paths are the same fold the old getters used.
                let want = first(&names, &env).unwrap_or(false);
                assert_eq!(got, want, "{}: env {env:?}", GETTERS[idx]);
            }
        }
    }
}

/// A retired alias, set alone to every value, must leave its getter at the
/// default. Against the pre-removal code `GEMMA_METAL_DECODE_ICB=1` turns
/// `decode_icb_enabled` on.
#[test]
fn retired_aliases_decide_nothing() {
    for (idx, name) in RETIRED {
        for v in ["1", "true", "yes", "0"] {
            let got = run_child(&[(name, v)]);
            let default = vec![false; GETTERS.len()];
            assert_eq!(got, default, "{name}={v} must not move {}", GETTERS[idx]);
        }
    }
}

#[test]
fn coarse_barriers_inherits_hazard_only_when_its_env_does_not_parse() {
    let coarse = ["TESSL_COARSE_BARRIERS", "METAL_RUNTIME_COARSE_BARRIERS"];
    for cv in VALUES {
        for hv in VALUES {
            for name in coarse {
                let env = [(name, cv), ("TESSL_HAZARD_BARRIERS", hv)];
                let r = run_child(&env);
                let hazard = first(&["TESSL_HAZARD_BARRIERS"], &env).unwrap_or(false);
                let want = first(&coarse, &env).unwrap_or(hazard);
                assert_eq!(r[6], want, "coarse_barriers: env {env:?}");
            }
        }
    }
}
