//! Every self-hosted job in `.github/workflows/ci.yml` must be fork-safe.
//!
//! A self-hosted runner executes whatever a pull request's workflow says. On a
//! public repository any fork can open one, so each self-hosted job carries the
//! `HAS_METAL_RUNNER` variable gate (a job whose labels match nothing queues
//! forever) and a same-repository guard for `pull_request` events.

use std::path::PathBuf;

const VARIABLE_GATE: &str = "vars.HAS_METAL_RUNNER == 'true'";
const FORK_GUARD: &str = "github.event_name != 'pull_request' || \
     github.event.pull_request.head.repo.full_name == github.repository";

/// `(job id, its text)` for each entry under the top-level `jobs:` key.
fn jobs(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut in_jobs = false;
    for line in text.lines() {
        if !line.starts_with(' ') && !line.is_empty() && !line.starts_with('#') {
            in_jobs = line.trim_end() == "jobs:";
            continue;
        }
        if !in_jobs {
            continue;
        }
        let is_id = line.starts_with("  ")
            && !line.starts_with("   ")
            && !line.trim_start().starts_with('#')
            && line.trim_end().ends_with(':');
        if is_id {
            out.push((line.trim().trim_end_matches(':').to_string(), String::new()));
        } else if let Some((_, body)) = out.last_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    out
}

/// The job's own `runs-on:` value and `if:` value (comments excluded).
fn field<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    body.lines()
        .find(|l| l.starts_with(&format!("    {key}:")))
        .map(|l| l[4 + key.len() + 1..].trim())
}

fn self_hosted_jobs() -> Vec<(String, String)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yml");
    let text = std::fs::read_to_string(&path).expect("read ci.yml");
    jobs(&text)
        .into_iter()
        .filter(|(_, body)| field(body, "runs-on").is_some_and(|r| r.contains("self-hosted")))
        .collect()
}

#[test]
fn every_self_hosted_job_is_gated_and_fork_safe() {
    let found = self_hosted_jobs();
    let ids: Vec<&str> = found.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        ids.contains(&"gpu") && ids.contains(&"bench"),
        "parser must find the known self-hosted jobs, found {ids:?}"
    );
    for (id, body) in &found {
        let cond = field(body, "if").unwrap_or("");
        assert!(
            cond.contains(VARIABLE_GATE),
            "self-hosted job `{id}` lacks the HAS_METAL_RUNNER gate (if: {cond:?})"
        );
        assert!(
            cond.contains(FORK_GUARD),
            "self-hosted job `{id}` lacks the same-repository pull_request guard (if: {cond:?})"
        );
    }
}

#[test]
fn the_job_parser_sees_only_self_hosted_runs_on() {
    let text = "name: X\njobs:\n  a:\n    runs-on: ubuntu-latest\n    # runs-on: self-hosted\n  b:\n    runs-on: [self-hosted, macOS]\n";
    let ids: Vec<String> = jobs(text)
        .into_iter()
        .filter(|(_, b)| field(b, "runs-on").is_some_and(|r| r.contains("self-hosted")))
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, ["b"]);
}
