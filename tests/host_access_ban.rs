//! Source-text ban: non-test library code never calls the panicking host
//! accessors of `GpuBuffer` (`write_f32`, `read_f32`, `contents_f32`, ...). A
//! load or inference path that panics on a busy runtime (a live `HostMapping`)
//! is the same bug as any other; library code uses the `try_*` twins and
//! propagates the error. Binaries (`src/bin/**`) and `#[cfg(test)]` items are
//! exempt, and so are the definitions in `src/tensor.rs` (a `fn` item is not a
//! `.name(` call). CPU-only.

use std::path::{Path, PathBuf};

const BANNED: &[&str] = &[
    ".write_f32(",
    ".write_u32(",
    ".read_f32(",
    ".read_u32(",
    ".contents_f32(",
    ".contents_u16(",
    ".contents_u8(",
    ".contents_u32(",
    ".write_bf16_bits(",
    ".write_f16_bits(",
    ".write_f32_prefix(",
    ".write_bytes(",
];

/// `SafeTensors::read_f32` / `read_u32` share two banned names but return
/// `Result` (they read a file, not a mapping); the loaders call them on a
/// receiver named `st`.
fn is_safetensors_receiver(line: &str, at: usize) -> bool {
    let before = &line[..at];
    let ident_start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map_or(0, |p| p + 1);
    &before[ident_start..] == "st"
}

/// Blank out comments, string and char literals (keeping newlines so line
/// numbers survive), then blank every `#[cfg(test)]` item (brace-matched).
fn strip(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = vec![b' '; b.len()];
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            out[i] = b'\n';
            i += 1;
        } else if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            let mut depth = 0;
            while i < b.len() {
                if b[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if b[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    if b[i] == b'\n' {
                        out[i] = b'\n';
                    }
                    i += 1;
                }
            }
        } else if c == b'r' && {
            let h = b[i + 1..].iter().take_while(|&&x| x == b'#').count();
            b.get(i + 1 + h) == Some(&b'"')
        } {
            let h = b[i + 1..].iter().take_while(|&&x| x == b'#').count();
            let mut close = vec![b'"'];
            close.extend(std::iter::repeat_n(b'#', h));
            i += 2 + h;
            while i < b.len() && !b[i..].starts_with(&close) {
                if b[i] == b'\n' {
                    out[i] = b'\n';
                }
                i += 1;
            }
            i += close.len();
        } else if c == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                if i < b.len() && b[i] == b'\n' {
                    out[i] = b'\n';
                }
                i += 1;
            }
            i += 1;
        } else if c == b'\'' {
            // A char literal ('x', '\n', '\'') vs a lifetime ('a): a literal
            // closes within a few bytes.
            let close = if b.get(i + 1) == Some(&b'\\') {
                b[i + 2..].iter().take(10).position(|&x| x == b'\'').map(|p| i + 3 + p)
            } else if b.get(i + 2) == Some(&b'\'') {
                Some(i + 3)
            } else {
                None
            };
            if let Some(end) = close {
                i = end;
            } else {
                out[i] = c;
                i += 1;
            }
        } else {
            out[i] = c;
            i += 1;
        }
    }
    // Everything kept is ASCII-or-original bytes at their original offsets and
    // everything else is a space or newline, so this stays valid UTF-8 only if
    // we never split a multi-byte char: lossy conversion covers the rest.
    let mut s = String::from_utf8_lossy(&out).into_owned();
    // Blank `#[cfg(test)]` items: from the attribute through the matching `}`
    // (or the `;` of a body-less item such as `use`).
    let attr = "#[cfg(test)]";
    let mut from = 0;
    while let Some(rel) = s[from..].find(attr) {
        let start = from + rel;
        let bytes = s.as_bytes();
        let mut j = start + attr.len();
        let end = loop {
            assert!(j < bytes.len(), "unterminated #[cfg(test)] item");
            match bytes[j] {
                b';' => break j + 1,
                b'{' => {
                    let mut depth = 0;
                    while j < bytes.len() {
                        match bytes[j] {
                            b'{' => depth += 1,
                            b'}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        j += 1;
                    }
                    break (j + 1).min(bytes.len());
                }
                _ => j += 1,
            }
        };
        let blanked: String = s[start..end]
            .chars()
            .map(|c| if c == '\n' { '\n' } else { ' ' })
            .collect();
        s.replace_range(start..end, &blanked);
        from = start + blanked.len();
    }
    s
}

/// `file:line: .accessor` for every banned call in `src`.
fn violations(file: &str, src: &str) -> Vec<String> {
    let stripped = strip(src);
    let mut out = Vec::new();
    for (n, line) in stripped.lines().enumerate() {
        for pat in BANNED {
            for (at, _) in line.match_indices(pat) {
                if !is_safetensors_receiver(line, at) {
                    out.push(format!("{file}:{}: {}", n + 1, pat.trim_end_matches('(')));
                }
            }
        }
    }
    out
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "bin") && p.parent().is_some_and(|d| d.ends_with("src")) {
                continue;
            }
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn library_code_never_calls_a_panicking_host_accessor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rs_files(&root.join("src"), &mut files);
    assert!(files.len() > 10, "found only {} source files", files.len());
    let mut bad = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(root).unwrap().display().to_string();
        bad.extend(violations(&rel, &std::fs::read_to_string(f).unwrap()));
    }
    assert!(
        bad.is_empty(),
        "panicking host accessors on non-test library paths (use the try_* twin):\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_scanner_catches_a_planted_call() {
    let src = "fn load(b: &GpuBuffer) {\n    let x = 1;\n    b.write_f32(&[1.0]);\n}\n";
    assert_eq!(
        violations("planted.rs", src),
        vec!["planted.rs:3: .write_f32".to_string()]
    );
    for pat in BANNED {
        let src = format!("fn f() {{\n    b{pat}x);\n}}\n");
        assert_eq!(violations("p.rs", &src).len(), 1, "{pat}");
    }
}

#[test]
fn the_scanner_ignores_tests_comments_strings_and_definitions() {
    let src = r##"
// b.write_f32(&x);
/* b.read_f32() /* nested */ b.read_u32() */
fn ok(b: &GpuBuffer) -> Result<(), String> {
    let s = "b.write_f32(&x)";
    let r = r#"b.contents_f32()"#;
    let c = '(';
    b.try_write_f32(&[1.0])?;
    Ok(())
}
pub fn write_f32(&self, d: &[f32]) {}
#[cfg(test)]
mod tests {
    fn t(b: &GpuBuffer) {
        b.write_f32(&[1.0]);
        if true { b.read_f32(); }
    }
}
fn after_tests(b: &GpuBuffer) {
    b.contents_u8();
}
"##;
    let v = violations("x.rs", src);
    assert_eq!(v, vec!["x.rs:20: .contents_u8".to_string()], "{v:?}");
    // A SafeTensors read (a `Result`-returning file read) is not a mapping.
    assert!(violations("y.rs", "fn f() { self.st.read_f32(&n)?; }").is_empty());
    assert_eq!(violations("y.rs", "fn f() { self.st2.read_f32(&n)?; }").len(), 1);
}
