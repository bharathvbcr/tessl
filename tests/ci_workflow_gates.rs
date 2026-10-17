//! Every workflow job that is not provably GitHub-hosted must carry the guard.
//!
//! A self-hosted runner executes whatever a pull request's workflow says. Each
//! job in `.github/workflows/*.yml|yaml` therefore has to be one of:
//!   * `runs-on` made only of explicit GitHub-hosted labels (`macos-*`,
//!     `ubuntu-*`, `windows-*`), or
//!   * guarded: its `if:` is exactly [`GUARD`] (whitespace-normalised, optionally
//!     wrapped in `${{ }}`).
//!
//! Anything else (custom labels, a group/labels map, a `${{ }}` expression, a
//! reusable-workflow `uses:` job, a missing `runs-on`) is not provably hosted and
//! fails closed without the guard. The guard is an accident filter, not a
//! security boundary: a fork's PR can edit the workflow file. The owner settings
//! named in ci.yml are the real controls. The parser is deliberately small and
//! std-only; what it cannot read it treats as not hosted.

use std::path::PathBuf;

/// The canonical `if:` expression, whitespace-normalised.
const GUARD: &str = "vars.HAS_METAL_RUNNER == 'true' && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)";

const HOSTED_PREFIXES: &[&str] = &["macos-", "ubuntu-", "windows-"];

#[derive(Debug)]
struct Job {
    id: String,
    hosted: bool,
    guarded: bool,
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

fn is_blank_or_comment(line: &str) -> bool {
    let t = line.trim();
    t.is_empty() || t.starts_with('#')
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Drop a trailing ` # comment` that is outside quotes.
fn strip_comment(s: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut prev = ' ';
    for (i, c) in s.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' && prev.is_whitespace() => return &s[..i],
            None => {}
        }
        prev = c;
    }
    s
}

/// Split `key: value` at the key colon, unquoting the key. None if not a mapping line.
fn split_key(line: &str) -> Option<(String, String)> {
    let t = line.trim_start();
    if let Some(q) = t.chars().next().filter(|c| *c == '"' || *c == '\'') {
        let end = t[1..].find(q)? + 1;
        let rest = t[end + 1..].trim_start().strip_prefix(':')?;
        return Some((t[1..end].to_string(), rest.trim().to_string()));
    }
    let (k, v) = t.split_once(':')?;
    if k.contains(' ') || k.starts_with('-') {
        return None;
    }
    // `key:value` without a space is a plain scalar in YAML, not a mapping.
    if !v.is_empty() && !v.starts_with(' ') {
        return None;
    }
    Some((k.to_string(), v.trim().to_string()))
}

/// The value of a property whose key line is `lines[0]`: inline text plus every
/// following more-indented line. Block scalars (`|`, `>`) keep their content;
/// everything else drops comment-only lines and trailing comments.
fn value_of(lines: &[&str], key_indent: usize, inline: &str) -> String {
    let body: Vec<&str> = lines[1..]
        .iter()
        .take_while(|l| l.trim().is_empty() || indent(l) > key_indent)
        .copied()
        .collect();
    if inline.starts_with('|') || inline.starts_with('>') {
        return norm(&body.join(" "));
    }
    let mut parts = vec![strip_comment(inline).trim().to_string()];
    for l in body {
        if !is_blank_or_comment(l) {
            parts.push(strip_comment(l).trim().to_string());
        }
    }
    norm(&parts.join(" "))
}

fn runs_on_is_hosted(value: &str) -> bool {
    if value.is_empty() || value.contains("${{") {
        return false;
    }
    let cleaned: String = value
        .chars()
        .map(|c| {
            if matches!(c, '[' | ']' | ',' | '"' | '\'') {
                ' '
            } else {
                c
            }
        })
        .collect();
    let mut any = false;
    for tok in cleaned.split_whitespace() {
        if tok == "-" {
            continue;
        }
        let label_ok = HOSTED_PREFIXES.iter().any(|p| tok.starts_with(p))
            && tok
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'));
        if !label_ok {
            return false;
        }
        any = true;
    }
    any
}

fn guard_matches(value: &str) -> bool {
    let v = value.trim();
    let v = v
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .unwrap_or(v)
        .trim();
    let v = match v.strip_prefix("${{").and_then(|x| x.strip_suffix("}}")) {
        Some(inner) => inner.trim(),
        None => v,
    };
    norm(v) == GUARD
}

fn parse_jobs(text: &str) -> Vec<Job> {
    let lines: Vec<&str> = text.lines().collect();
    let Some(start) = lines.iter().position(|l| {
        indent(l) == 0 && matches!(split_key(l), Some((k, v)) if k == "jobs" && strip_comment(&v).trim().is_empty())
    }) else {
        return Vec::new();
    };
    let section: Vec<&str> = lines[start + 1..]
        .iter()
        .take_while(|l| is_blank_or_comment(l) || indent(l) > 0)
        .copied()
        .collect();
    let Some(job_indent) = section.iter().find(|l| !is_blank_or_comment(l)).map(|l| indent(l)) else {
        return Vec::new();
    };
    let mut jobs = Vec::new();
    let mut i = 0;
    while i < section.len() {
        let l = section[i];
        let head = if !is_blank_or_comment(l) && indent(l) == job_indent {
            split_key(l).filter(|(_, v)| strip_comment(v).trim().is_empty())
        } else {
            None
        };
        let Some((id, _)) = head else {
            i += 1;
            continue;
        };
        let end = section[i + 1..]
            .iter()
            .position(|l| !is_blank_or_comment(l) && indent(l) <= job_indent)
            .map_or(section.len(), |n| i + 1 + n);
        jobs.push(analyse(&id, &section[i + 1..end]));
        i = end;
    }
    jobs
}

fn analyse(id: &str, body: &[&str]) -> Job {
    let prop_indent = body.iter().find(|l| !is_blank_or_comment(l)).map(|l| indent(l));
    let (mut hosted, mut guarded, mut uses) = (false, false, false);
    if let Some(pi) = prop_indent {
        for (n, l) in body.iter().enumerate() {
            if is_blank_or_comment(l) || indent(l) != pi {
                continue;
            }
            let Some((key, inline)) = split_key(l) else { continue };
            match key.as_str() {
                "runs-on" => hosted = runs_on_is_hosted(&value_of(&body[n..], pi, &inline)),
                "if" => guarded = guard_matches(&value_of(&body[n..], pi, &inline)),
                "uses" => uses = true,
                _ => {}
            }
        }
    }
    Job {
        id: id.to_string(),
        hosted: hosted && !uses,
        guarded,
    }
}

/// Jobs that are neither provably hosted nor guarded.
fn violations(text: &str) -> Vec<String> {
    parse_jobs(text)
        .into_iter()
        .filter(|j| !j.hosted && !j.guarded)
        .map(|j| j.id)
        .collect()
}

fn workflow_files() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut out: Vec<(String, String)> = std::fs::read_dir(&dir)
        .expect("read .github/workflows")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("yml" | "yaml")))
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("read workflow");
            (p.file_name().unwrap().to_string_lossy().into_owned(), text)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn every_workflow_job_is_hosted_or_guarded() {
    let files = workflow_files();
    assert!(files.iter().any(|(n, _)| n == "ci.yml"), "ci.yml must be scanned");
    for (name, text) in &files {
        let v = violations(text);
        assert!(v.is_empty(), "{name}: jobs neither GitHub-hosted nor guarded: {v:?}");
    }
}

#[test]
fn the_real_ci_workflow_guards_gpu_and_bench_and_hosts_the_rest() {
    let files = workflow_files();
    let text = &files.iter().find(|(n, _)| n == "ci.yml").expect("ci.yml").1;
    let jobs = parse_jobs(text);
    for id in ["gpu", "bench"] {
        let j = jobs
            .iter()
            .find(|j| j.id == id)
            .unwrap_or_else(|| panic!("job {id} not parsed"));
        assert!(j.guarded && !j.hosted, "{j:?}");
    }
    for id in ["fmt", "kernel-emulator", "check", "go", "msrv"] {
        let j = jobs
            .iter()
            .find(|j| j.id == id)
            .unwrap_or_else(|| panic!("job {id} not parsed"));
        assert!(j.hosted && !j.guarded, "{j:?}");
    }
}

fn wf(job: &str) -> String {
    format!("name: x\non: push\njobs:\n{job}")
}

const G: &str = GUARD;

#[test]
fn adversarial_variants_are_rejected() {
    let bad: Vec<(&str, String)> = vec![
        ("custom label", wf("  a:\n    runs-on: [self-hosted, macOS]\n")),
        ("inline custom label", wf("  a:\n    runs-on: my-box\n")),
        (
            "continuation-line label",
            wf("  a:\n    runs-on: [macos-26,\n      self-hosted]\n"),
        ),
        (
            "list on following lines",
            wf("  a:\n    runs-on:\n      - macos-26\n      - self-hosted\n"),
        ),
        (
            "group map",
            wf("  a:\n    runs-on:\n      group: metal\n      labels: macos-26\n"),
        ),
        ("matrix expression", wf("  a:\n    runs-on: ${{ matrix.os }}\n")),
        ("quoted key", wf("  a:\n    \"runs-on\": self-hosted\n")),
        ("quoted key and value", wf("  a:\n    'runs-on': 'self-hosted'\n")),
        (
            "4-space indent",
            "name: x\njobs:\n    a:\n        runs-on: self-hosted\n".to_string(),
        ),
        ("uses job", wf("  a:\n    uses: ./.github/workflows/gpu.yml\n")),
        (
            "uses job with hosted runs-on",
            wf("  a:\n    runs-on: macos-26\n    uses: o/r/.github/workflows/w.yml@v1\n"),
        ),
        ("missing runs-on", wf("  a:\n    steps: []\n")),
        (
            "true bypass",
            wf(&format!("  a:\n    if: true || {G}\n    runs-on: self-hosted\n")),
        ),
        (
            "bypass inside wrapper",
            wf(&format!(
                "  a:\n    if: ${{{{ true || {G} }}}}\n    runs-on: self-hosted\n"
            )),
        ),
        (
            "variable gate only",
            wf("  a:\n    if: ${{ vars.HAS_METAL_RUNNER == 'true' }}\n    runs-on: self-hosted\n"),
        ),
        (
            "guard in a comment",
            wf(&format!("  a:\n    # if: {G}\n    runs-on: self-hosted\n")),
        ),
        (
            "guard in a trailing comment",
            wf("  a:\n    if: true # vars.HAS_METAL_RUNNER == 'true'\n    runs-on: self-hosted\n"),
        ),
        (
            "guard in a step",
            wf(&format!(
                "  a:\n    runs-on: self-hosted\n    steps:\n      - if: {G}\n        run: x\n"
            )),
        ),
        (
            "hosted label with a self-hosted label",
            wf("  a:\n    runs-on: ubuntu-latest, self-hosted\n"),
        ),
        ("lookalike prefix", wf("  a:\n    runs-on: macosx\n")),
    ];
    for (what, text) in &bad {
        assert_eq!(violations(text), ["a"], "{what} must be rejected");
    }
}

#[test]
fn correct_variants_pass() {
    let ok: Vec<(&str, String)> = vec![
        ("hosted macos", wf("  a:\n    runs-on: macos-26\n")),
        ("hosted flow list", wf("  a:\n    runs-on: [ubuntu-latest, \"windows-2022\"]\n")),
        ("hosted list lines", wf("  a:\n    runs-on:\n      - ubuntu-24.04\n")),
        (
            "hosted 4-space",
            "name: x\njobs:\n    a:\n        runs-on: ubuntu-latest\n".to_string(),
        ),
        ("guarded plain", wf(&format!("  a:\n    if: {G}\n    runs-on: self-hosted\n"))),
        (
            "guarded wrapper",
            wf(&format!("  a:\n    if: ${{{{ {G} }}}}\n    runs-on: [self-hosted]\n")),
        ),
        (
            "guarded wrapper with comment",
            wf(&format!("  a:\n    if: ${{{{ {G} }}}} # why\n    runs-on: [self-hosted]\n")),
        ),
        (
            "guarded folded >-",
            wf("  a:\n    if: >-\n      ${{ vars.HAS_METAL_RUNNER == 'true' &&\n      (github.event_name != 'pull_request' ||\n      github.event.pull_request.head.repo.full_name == github.repository) }}\n    runs-on: self-hosted\n"),
        ),
        (
            "guarded literal |",
            wf(&format!("  a:\n    if: |\n      {G}\n    runs-on: self-hosted\n")),
        ),
        (
            "guarded group map",
            wf(&format!("  a:\n    if: {G}\n    runs-on:\n      group: metal\n")),
        ),
    ];
    for (what, text) in &ok {
        assert!(violations(text).is_empty(), "{what} must pass: {:?}", violations(text));
    }
}

#[test]
fn every_job_is_found_across_a_file() {
    let text = wf("  a:\n    runs-on: macos-26\n  b:\n    runs-on: self-hosted\n  c:\n    runs-on: ubuntu-latest\n");
    assert_eq!(parse_jobs(&text).len(), 3);
    assert_eq!(violations(&text), ["b"]);
}

#[test]
fn a_second_workflow_file_is_checked_too() {
    let files = [
        ("ok.yml", wf("  a:\n    runs-on: macos-26\n")),
        ("bad.yaml", wf("  a:\n    runs-on: self-hosted\n")),
    ];
    let bad: Vec<&str> = files
        .iter()
        .filter(|(_, t)| !violations(t).is_empty())
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(bad, ["bad.yaml"]);
    let real: Vec<String> = workflow_files().into_iter().map(|(n, _)| n).collect();
    assert!(real.len() >= 3, "every workflow file is scanned, saw {real:?}");
}
