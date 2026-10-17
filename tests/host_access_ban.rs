//! Source-text ban: non-test library code never calls the panicking host
//! accessors of `GpuBuffer` (`write_f32`, `read_f32`, `contents_f32`, ...). A
//! load or inference path that panics on a busy runtime (a live `HostMapping`)
//! is the same bug as any other; library code uses the `try_*` twins and
//! propagates the error. Binaries (`src/bin/**`), `#[cfg(test)]` items and
//! whole files whose `mod` declaration is `#[cfg(test)]` (e.g. `src/lib.rs`'s
//! `#[cfg(test)] mod parser_mutation;`) are exempt, and so are the definitions in `src/tensor.rs` (a `fn` item is not a
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

/// One reviewed exemption: `count` calls of `method` on exactly `receiver`
/// (the identifier chain right before the dot) in `file`.
struct Site {
    file: &'static str,
    receiver: &'static str,
    method: &'static str,
    count: usize,
}

/// Why this is a site list. `SafeTensors::read_f32` / `read_u32` and the
/// `CheckpointSource` / `TensorSource` trait methods share names with the
/// panicking `GpuBuffer` accessors but return `Result` (they read a file, not a
/// mapping). Renaming them is a public API change the owner has not approved,
/// and text scanning cannot infer a receiver's type soundly (aliases, wrapper
/// types, shadowing, pattern bindings), so no heuristic is attempted: every
/// exempt call is a reviewed site, counted exactly.
///
/// To add a site: confirm by reading the call that the receiver is a
/// `SafeTensors` / `CheckpointSource` / `TensorSource` value, then add a row
/// with the file, the receiver text as written (`self.st`, `st`) and the number
/// of such calls. A new call beyond the count fails the test and names the
/// call; a count above what the file holds (a stale row) fails and names the row.
const ALLOWED_SITES: &[Site] = &[
    Site {
        file: "src/loader.rs",
        receiver: "self.st",
        method: ".read_f32(",
        count: 2,
    },
    Site {
        file: "src/safetensors.rs",
        receiver: "st",
        method: ".read_f32(",
        count: 1,
    },
];

/// The identifier chain immediately before the dot (`self.st`, `st`); empty
/// when the receiver is a call, index or other expression, so those never match.
fn receiver_text(line: &str, at: usize) -> &str {
    let before = &line[..at];
    let start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .map_or(0, |p| p + 1);
    let chain = &before[start..];
    if chain.starts_with('.') {
        ""
    } else {
        chain
    }
}

/// Blank out comments, string and char literals (keeping newlines so line
/// numbers survive), then blank every `#[cfg(test)]` item (brace-matched).
fn strip(src: &str) -> String {
    blank_cfg_test_items(strip_comments_and_strings(src))
}

/// Blank out comments, string and char literals, keeping newlines.
fn strip_comments_and_strings(src: &str) -> String {
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
    String::from_utf8_lossy(&out).into_owned()
}

/// Blank every `#[cfg(test)]` item of comment-stripped source.
fn blank_cfg_test_items(mut s: String) -> String {
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

/// `file:line: .accessor` for every banned call in `src` that is not covered by
/// a `sites` row, plus one line per row whose count the file does not match:
/// calls beyond the count are reported individually, a shortfall names the row.
fn violations_with(file: &str, src: &str, sites: &[Site]) -> Vec<String> {
    let stripped = strip(src);
    let mut out = Vec::new();
    let mut seen = vec![0usize; sites.len()];
    for (n, line) in stripped.lines().enumerate() {
        let mut calls: Vec<(usize, &str)> = BANNED
            .iter()
            .flat_map(|pat| line.match_indices(pat).map(move |(at, _)| (at, *pat)))
            .collect();
        calls.sort_unstable();
        for (at, pat) in calls {
            let recv = receiver_text(line, at);
            let site = sites
                .iter()
                .position(|s| s.file == file && s.receiver == recv && s.method == pat);
            match site {
                Some(i) if seen[i] < sites[i].count => seen[i] += 1,
                Some(i) => {
                    out.push(format!(
                        "{file}:{}: {} on `{recv}` exceeds the {} reviewed call(s) in ALLOWED_SITES",
                        n + 1,
                        pat.trim_end_matches('('),
                        sites[i].count
                    ));
                }
                None => out.push(format!("{file}:{}: {}", n + 1, pat.trim_end_matches('('))),
            }
        }
    }
    for (s, &got) in sites.iter().zip(&seen) {
        if s.file == file && got < s.count {
            out.push(format!(
                "{file}: stale ALLOWED_SITES row: `{}{}` expected {} call(s), found {got}",
                s.receiver, s.method, s.count
            ));
        }
    }
    out
}

fn violations(file: &str, src: &str) -> Vec<String> {
    violations_with(file, src, ALLOWED_SITES)
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

/// Module names that `src` declares out of line under `#[cfg(test)]`
/// (`#[cfg(test)] mod name;`, other attributes allowed between). Comments and
/// strings are blanked first, so a commented-out declaration does not count.
fn cfg_test_out_of_line_mods(src: &str) -> Vec<String> {
    let s = strip_comments_and_strings(src);
    let mut out = Vec::new();
    let attr = "#[cfg(test)]";
    let mut from = 0;
    while let Some(i) = s[from..].find(attr) {
        let mut rest = s[from + i + attr.len()..].trim_start();
        // Further attributes on the same item.
        while rest.starts_with("#[") {
            match rest.find(']') {
                Some(j) => rest = rest[j + 1..].trim_start(),
                None => break,
            }
        }
        let rest = rest
            .strip_prefix("pub(crate) ")
            .or_else(|| rest.strip_prefix("pub "))
            .unwrap_or(rest);
        if let Some(decl) = rest.strip_prefix("mod ") {
            let name: String = decl.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() && decl[name.len()..].trim_start().starts_with(';') {
                out.push(name);
            }
        }
        from += i + attr.len();
    }
    out
}

/// The file (`name.rs`) or directory (`name/`) an out-of-line `mod name;` in
/// `decl_file` refers to: a sibling for `lib.rs`/`main.rs`/`mod.rs`, otherwise
/// under the declaring file's own directory (`foo.rs` → `foo/name.rs`).
fn mod_paths(decl_file: &Path, name: &str) -> [PathBuf; 2] {
    let parent = decl_file.parent().unwrap();
    let stem = decl_file.file_stem().unwrap();
    let dir = if ["lib", "main", "mod"].iter().any(|s| stem == *s) {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    [dir.join(format!("{name}.rs")), dir.join(name)]
}

/// Every source path compiled only under `cfg(test)` because its `mod`
/// declaration is gated: a file, or a whole directory module.
fn cfg_test_only_paths(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for f in files {
        for name in cfg_test_out_of_line_mods(&std::fs::read_to_string(f).unwrap()) {
            out.extend(mod_paths(f, &name).into_iter().filter(|p| p.exists()));
        }
    }
    out
}

#[test]
fn library_code_never_calls_a_panicking_host_accessor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rs_files(&root.join("src"), &mut files);
    assert!(files.len() > 10, "found only {} source files", files.len());
    let test_only = cfg_test_only_paths(&files);
    let mut bad = Vec::new();
    let mut scanned = Vec::new();
    for f in files.iter().filter(|f| !test_only.iter().any(|t| f.starts_with(t))) {
        let rel = f.strip_prefix(root).unwrap().display().to_string();
        bad.extend(violations(&rel, &std::fs::read_to_string(f).unwrap()));
        scanned.push(rel);
    }
    for s in ALLOWED_SITES.iter().filter(|s| !scanned.iter().any(|r| r == s.file)) {
        bad.push(format!(
            "{}: stale ALLOWED_SITES row: file is not scanned library code",
            s.file
        ));
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
}

const PLANTED: &[Site] = &[Site {
    file: "p.rs",
    receiver: "self.st",
    method: ".read_f32(",
    count: 2,
}];

/// A `Result`-returning checkpoint read is exempt only at a reviewed site, by
/// exact file and receiver text, and the per-file count must match exactly.
#[test]
fn the_exemption_is_an_exact_counted_site_list() {
    let v = |file: &str, src: &str| violations_with(file, src, PLANTED);
    let two = "fn f(&self) {\n    self.st.read_f32(&a)?;\n    self.st.read_f32(&b)?;\n}\n";
    assert!(v("p.rs", two).is_empty());
    // A third call with the allowlisted receiver text is an unreviewed call.
    let three = format!("{two}fn g(&self) {{ self.st.read_f32(&c)?; }}\n");
    assert_eq!(
        v("p.rs", &three),
        vec!["p.rs:5: .read_f32 on `self.st` exceeds the 2 reviewed call(s) in ALLOWED_SITES"]
    );
    // A missing site is a stale row.
    let one = "fn f(&self) { self.st.read_f32(&a)?; }\n";
    assert_eq!(
        v("p.rs", one),
        vec!["p.rs: stale ALLOWED_SITES row: `self.st.read_f32(` expected 2 call(s), found 1"]
    );
    // Same receiver in another file, another method, or another receiver text.
    assert_eq!(v("q.rs", two).len(), 2);
    assert_eq!(v("p.rs", "fn f(&self) { self.st.read_u32(&a)?; }\n").len(), 2);
    assert_eq!(v("p.rs", &format!("{two}fn g() {{ st.read_f32(&c); }}\n")).len(), 1);
    // Multiple calls on one line are counted individually.
    assert_eq!(
        v(
            "p.rs",
            "fn f(&self, b: &GpuBuffer) { self.st.read_f32(&a)?; b.read_f32(); self.st.read_f32(&c)?; }\n"
        ),
        vec!["p.rs:1: .read_f32"]
    );
}

/// The regression the old `st` name rule missed, plus type-mentioning shapes a
/// text-level type inference would wrongly clear: none has a site row.
#[test]
fn receivers_without_a_reviewed_site_are_flagged() {
    for src in [
        "fn f(st: &GpuBuffer) { st.read_f32(); }",
        "fn f(rt: &R) { let st = rt.alloc(4); st.read_f32(); }",
        "struct L { st: GpuBuffer }\nfn f(l: &L) { l.st.read_u32(); }",
        "fn f(st: Result<GpuBuffer, SafeTensors>) { st.read_f32(); }",
        "type SafeTensors = GpuBuffer;\nfn f(st: &SafeTensors) { st.read_f32(); }",
        "use x::GpuBuffer as SafeTensors;\nfn f(st: &SafeTensors) { st.read_f32(); }",
        "fn f(p: &str, buf: GpuBuffer) { let st = SafeTensors::open(p).map(|_| buf).unwrap(); st.read_f32(); }",
        "fn f(o: &O) { if let Some(st) = o.get() { st.read_f32(); } }",
        "fn f(c: &C) { c.checkpoint().read_f32(&n); }",
        "fn f(v: &[SafeTensors]) { v[0].read_f32(&n); }",
        "fn f(ckpt: &SafeTensors) { ckpt.read_f32(&n)?; }",
    ] {
        assert_eq!(violations("y.rs", src).len(), 1, "{src}");
    }
}

#[test]
fn a_cfg_test_mod_declaration_exempts_its_file_and_nothing_else() {
    let src = r##"
pub mod tensor;
#[cfg(test)]
mod parser_mutation;
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod helpers;
#[cfg(test)]
mod inline { fn t() {} }
// #[cfg(test)] mod commented;
const S: &str = "#[cfg(test)] mod in_a_string;";
#[cfg(not(test))]
mod real;
"##;
    assert_eq!(cfg_test_out_of_line_mods(src), vec!["parser_mutation", "helpers"]);
    let lib = Path::new("/r/src/lib.rs");
    assert_eq!(
        mod_paths(lib, "parser_mutation"),
        [
            PathBuf::from("/r/src/parser_mutation.rs"),
            PathBuf::from("/r/src/parser_mutation")
        ]
    );
    assert_eq!(
        mod_paths(Path::new("/r/src/nn.rs"), "t"),
        [PathBuf::from("/r/src/nn/t.rs"), PathBuf::from("/r/src/nn/t")]
    );
    // The real tree: the gated module is found, and an ordinary library
    // module is not exempt.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rs_files(&root.join("src"), &mut files);
    let test_only = cfg_test_only_paths(&files);
    assert!(
        test_only.contains(&root.join("src/parser_mutation.rs")),
        "{test_only:?}"
    );
    assert!(
        !test_only.iter().any(|t| root.join("src/tensor.rs").starts_with(t)),
        "{test_only:?}"
    );
}
