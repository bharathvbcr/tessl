//! The text checks `scripts/check_release_ready.sh` and the release workflow
//! make on README.md, CHANGELOG.md and Cargo.lock.
//!
//! ```text
//! release_check readme            README.md    VERSION
//! release_check changelog         CHANGELOG.md VERSION
//! release_check lock              Cargo.lock   CRATE VERSION
//! release_check changelog-section CHANGELOG.md VERSION
//! ```
//!
//! The first three fail (exit 1, `error: ...` on stderr) when the version the
//! file states is missing or is not `VERSION`. `changelog-section` prints the
//! body of the `## [VERSION]` section, trimmed, and fails when there is none;
//! the release workflow publishes it as the GitHub Release notes.
//!
//! Standalone on purpose: standard library only, no `tessl` import, so the
//! gate compiles with a bare `rustc --edition 2021` before anything else in the
//! crate (the Metal toolchain included) has been built.
//!
//! These replace three inline `python3 -` heredocs and one in release.yml.
//! Their matching rules are the old regular expressions, written out:
//!
//! * README: the first `**Status** | [`X.Y.Z`]` anywhere in the file;
//! * CHANGELOG: the first line that begins `## [X.Y.Z]`;
//! * Cargo.lock: the first line `name = "CRATE"` whose very next line starts
//!   `version = "`, up to the next quote;
//! * section: the first `## [VERSION]` anywhere (not only at a line start),
//!   through the end of that line, up to the next `\n## [` or the end of the
//!   file. Two headings on consecutive lines therefore make the first section
//!   swallow the second, as the Python did.
//!
//! Files are read the way Python's `Path.read_text()` read them: UTF-8, with
//! `\r\n` and lone `\r` folded to `\n`.

use std::process::ExitCode;

/// Python's `str.isspace`: Unicode white space plus the four ASCII separators
/// `\x1c`..`\x1f`.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// Universal-newline decoding: `\r\n` and `\r` both become `\n`.
fn universal_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Where each line of `text` starts, as byte offsets (`^` under `re.M`).
fn line_starts(text: &str) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0).chain(text.match_indices('\n').map(|(i, _)| i + 1))
}

/// `N+.N+.N+` (ASCII digits) at the start of `s`, greedy; the text it spans.
fn dotted_triple(s: &str) -> Option<&str> {
    let b = s.as_bytes();
    let mut i = 0;
    for part in 0..3 {
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return None;
        }
        if part < 2 {
            if b.get(i) != Some(&b'.') {
                return None;
            }
            i += 1;
        }
    }
    Some(&s[..i])
}

/// `**Status** | [`X.Y.Z`]`: the version of the first such badge.
fn readme_version(text: &str) -> Option<String> {
    const OPEN: &str = "**Status** | [`";
    text.match_indices(OPEN).find_map(|(at, _)| {
        let rest = &text[at + OPEN.len()..];
        let v = dotted_triple(rest)?;
        rest[v.len()..].starts_with("`]").then(|| v.to_string())
    })
}

/// `## [X.Y.Z]` at the start of the first line that has one.
fn changelog_version(text: &str) -> Option<String> {
    line_starts(text).find_map(|at| {
        let rest = text[at..].strip_prefix("## [")?;
        let v = dotted_triple(rest)?;
        rest[v.len()..].starts_with(']').then(|| v.to_string())
    })
}

/// The `version` of the first `[[package]]` named `crate`: the line
/// `name = "<crate>"` immediately followed by a `version = "..."` line.
fn lock_version(text: &str, krate: &str) -> Option<String> {
    let head = format!("name = \"{krate}\"\nversion = \"");
    line_starts(text).find_map(|at| {
        let rest = text[at..].strip_prefix(head.as_str())?;
        let end = rest.find('"')?;
        (end > 0).then(|| rest[..end].to_string())
    })
}

/// The trimmed body of the first `## [version]` section, or `None`.
fn changelog_section(text: &str, version: &str) -> Option<String> {
    let heading = format!("## [{version}]");
    text.match_indices(heading.as_str()).find_map(|(at, _)| {
        let after = at + heading.len();
        // `[^\n]*\n`: the rest of the heading line, which must end in a newline.
        let body = after + text[after..].find('\n')? + 1;
        // `(.*?)(?=\n## \[|\Z)`: the shortest body followed by a new heading or the end.
        let end = text[body..].find("\n## [").map_or(text.len(), |i| body + i);
        Some(py_strip(&text[body..end]).to_string())
    })
}

fn read(path: &str) -> Result<String, String> {
    let raw = std::fs::read(path).map_err(|e| format!("error: cannot read {path}: {e}"))?;
    let text = String::from_utf8(raw).map_err(|e| format!("error: {path} is not UTF-8: {e}"))?;
    Ok(universal_newlines(&text))
}

/// `$(...)` in the shell script dropped trailing newlines before comparing; a
/// version cannot hold one except through the lock file's `[^"]+`, which can.
fn shell_value(v: &str) -> &str {
    v.trim_end_matches('\n')
}

fn run(args: &[String]) -> Result<Option<String>, String> {
    let arg = |i: usize| args.get(i).map(String::as_str);
    match (arg(0), args.len()) {
        (Some("readme"), 3) => {
            let found = readme_version(&read(&args[1])?).unwrap_or_default();
            if found != args[2] {
                let shown = if found.is_empty() { "<missing>" } else { &found };
                return Err(format!(
                    "error: README Status is '{shown}' but Cargo.toml is '{}'",
                    args[2]
                ));
            }
            Ok(None)
        }
        (Some("changelog"), 3) => {
            let found = changelog_version(&read(&args[1])?).unwrap_or_default();
            if found != args[2] {
                let shown = if found.is_empty() { "<missing>" } else { &found };
                return Err(format!(
                    "error: newest CHANGELOG version is '{shown}' but Cargo.toml is '{}'",
                    args[2]
                ));
            }
            Ok(None)
        }
        (Some("lock"), 4) => {
            // An absent entry (or one the shell script cannot compare) passes,
            // as in the script: only a stated, different version is drift.
            let found = lock_version(&read(&args[1])?, &args[2]).unwrap_or_default();
            let found = shell_value(&found);
            if !found.is_empty() && found != args[3] {
                return Err(format!(
                    "error: Cargo.lock has {} {found} but Cargo.toml is {}",
                    args[2], args[3]
                ));
            }
            Ok(None)
        }
        (Some("changelog-section"), 3) => match changelog_section(&read(&args[1])?, &args[2]) {
            Some(body) => Ok(Some(body)),
            None => Err(format!("CHANGELOG section for {} not found", args[2])),
        },
        _ => Err(
            "usage: release_check readme README VERSION | changelog CHANGELOG VERSION | \
                  lock CARGO_LOCK CRATE VERSION | changelog-section CHANGELOG VERSION"
                .to_string(),
        ),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(Some(body)) => {
            println!("{body}");
            ExitCode::SUCCESS
        }
        Ok(None) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}
