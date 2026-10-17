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

/// Types whose `read_f32` / `read_u32` return `Result` (a file read, not a
/// `GpuBuffer` mapping): `SafeTensors` and the checkpoint traits.
const SAFE_TYPES: &[&str] = &["SafeTensors", "CheckpointSource", "TensorSource"];

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte offsets of every whole-word occurrence of `word` in `s`.
fn word_positions<'a>(s: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    s.match_indices(word)
        .map(|(i, _)| i)
        .filter(move |&i| !s[..i].ends_with(is_ident_char) && !s[i + word.len()..].starts_with(is_ident_char))
}

/// A type or initializer expression: up to the first `,` `=` `;` `{` at depth
/// zero, or the first unbalanced closer.
fn clause(s: &str) -> &str {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => {
                depth -= 1;
                if depth < 0 {
                    return &s[..i];
                }
            }
            ',' | '=' | ';' | '{' if depth == 0 => return &s[..i],
            _ => {}
        }
    }
    s
}

fn mentions_safe_type(text: &str) -> bool {
    SAFE_TYPES.iter().any(|t| word_positions(text, t).next().is_some())
}

/// Does the file declare `fn name` with a safe type in its return type?
fn fn_returns_safe_type(stripped: &str, name: &str) -> bool {
    word_positions(stripped, name).any(|i| {
        stripped[..i].trim_end().ends_with("fn") && {
            let sig = stripped[i..].split(['{', ';']).next().unwrap_or("");
            sig.split_once("->").is_some_and(|(_, ret)| mentions_safe_type(ret))
        }
    })
}

/// Is `ident` bound to a safe type everywhere this file binds it? A binding is
/// `ident: Type` (parameter, field, typed `let`), `let [mut] ident = init`, or a
/// `for` / closure-parameter binding. Safe means: the type mentions a safe
/// type, or names a generic whose own bound does (`S: CheckpointSource`); an
/// initializer is `SafeTensors::...` or a call to a file-local `fn` returning
/// one. One non-safe binding anywhere in the file revokes the exemption
/// (shadowing is not resolved by text scanning), and so does having no
/// recognisable binding. `depth` bounds the generic-bound recursion.
fn is_safe_binding(stripped: &str, ident: &str, depth: u8) -> bool {
    let mut found = false;
    for i in word_positions(stripped, ident) {
        let before = stripped[..i].trim_end();
        let before = before
            .strip_suffix("mut")
            .filter(|r| !r.ends_with(is_ident_char))
            .map_or(before, str::trim_end);
        let after = stripped[i + ident.len()..].trim_start();
        let is_let = before.ends_with("let") && !before[..before.len() - 3].ends_with(is_ident_char);
        let safe = if after.starts_with(':') && !after.starts_with("::") {
            let ty = clause(&after[1..]);
            mentions_safe_type(ty)
                || (depth < 2
                    && ty
                        .split(|c: char| !is_ident_char(c))
                        .filter(|t| *t != ident && t.starts_with(char::is_uppercase))
                        .any(|t| is_safe_binding(stripped, t, depth + 1)))
        } else if is_let {
            let init = after.strip_prefix('=').unwrap_or("").split(';').next().unwrap_or("");
            let init = init.trim_start_matches(|c: char| c == '&' || c.is_whitespace());
            let init = init.strip_prefix("self.").unwrap_or(init);
            let callee = init.split('(').next().unwrap_or("").trim();
            init.contains("SafeTensors::")
                || (!callee.is_empty() && callee.chars().all(is_ident_char) && fn_returns_safe_type(stripped, callee))
        } else if before.ends_with("for") || before.ends_with('|') {
            false
        } else {
            continue;
        };
        if !safe {
            return false;
        }
        found = true;
    }
    found
}

/// `SafeTensors::read_f32` / `read_u32` (and the `CheckpointSource` /
/// `TensorSource` trait methods) share two banned names but return `Result`.
/// The exemption applies only when the receiver is a plain identifier (or
/// `self.ident`) that this file binds to such a type. A chained receiver such
/// as `checkpoint().read_f32(` or `a[0].read_f32(` has no identifier and is
/// flagged.
fn is_safetensors_receiver(stripped: &str, line: &str, at: usize) -> bool {
    let before = &line[..at];
    let ident = &before[before.rfind(|c: char| !is_ident_char(c)).map_or(0, |p| p + 1)..];
    !ident.is_empty() && is_safe_binding(stripped, ident, 0)
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

/// `file:line: .accessor` for every banned call in `src`.
fn violations(file: &str, src: &str) -> Vec<String> {
    let stripped = strip(src);
    let mut out = Vec::new();
    for (n, line) in stripped.lines().enumerate() {
        for pat in BANNED {
            for (at, _) in line.match_indices(pat) {
                if !is_safetensors_receiver(&stripped, line, at) {
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
    for f in files.iter().filter(|f| !test_only.iter().any(|t| f.starts_with(t))) {
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
}

/// A checkpoint read (a `Result`-returning file read) is not a mapping, but the
/// exemption follows the receiver's declared type, never its name.
#[test]
fn the_safetensors_exemption_follows_the_receiver_type_not_its_name() {
    let flagged = |src: &str| violations("y.rs", src).len();
    // `st` bound to a GpuBuffer is a panicking accessor (regression: the old
    // scanner exempted any receiver literally named `st`).
    assert_eq!(flagged("fn f(st: &GpuBuffer) { st.read_f32(); }"), 1);
    assert_eq!(flagged("fn f(rt: &R) { let st = rt.alloc(4); st.read_f32(); }"), 1);
    assert_eq!(
        flagged("struct L { st: GpuBuffer }\nfn f(l: &L) { l.st.read_u32(); }"),
        1
    );
    // Unbound receiver names are flagged, including the old `st`.
    assert_eq!(flagged("fn f() { st.read_f32(&n)?; }"), 1);
    assert_eq!(flagged("fn f() { self.st.read_f32(&n)?; }"), 1);
    // SafeTensors / CheckpointSource receivers pass under any name.
    assert_eq!(flagged("fn f(ckpt: &SafeTensors) { ckpt.read_f32(&n)?; }"), 0);
    assert_eq!(flagged("fn f(c: &dyn CheckpointSource) { c.read_u32(&n)?; }"), 0);
    assert_eq!(flagged("fn f(c: &impl CheckpointSource) { c.read_f32(&n)?; }"), 0);
    assert_eq!(flagged("fn f<S: CheckpointSource>(c: &S) { c.read_f32(&n)?; }"), 0);
    assert_eq!(
        flagged("fn f<S>(c: &S) where S: CheckpointSource { c.read_f32(&n)?; }"),
        0
    );
    assert_eq!(
        flagged("fn f() { let st = SafeTensors::open(p)?; st.read_f32(&n)?; }"),
        0
    );
    assert_eq!(
        flagged("struct L<'a> { st: &'a dyn TensorSource }\nfn f(l: &L) { l.st.read_f32(&n)?; }"),
        0
    );
    assert_eq!(
        flagged("fn shard(&self) -> Result<&SafeTensors, String> { x }\nfn f(&self) { let st = self.shard(n)?; st.read_f32(n) }"),
        0
    );
    // A chained receiver has no identifier to resolve: flagged.
    assert_eq!(flagged("fn f() { checkpoint().read_f32(&n)?; }"), 1);
    assert_eq!(flagged("fn f(v: &[SafeTensors]) { v[0].read_f32(&n)?; }"), 1);
    // Shadowing is not resolved: one non-safe binding in the file revokes the
    // exemption for every use of that name (conservative, may over-flag).
    assert_eq!(
        flagged("fn a(st: &SafeTensors) { st.read_f32(&n)?; }\nfn b(st: &GpuBuffer) { st.read_f32(); }"),
        2
    );
    // Several calls on one line are judged independently.
    assert_eq!(
        flagged(
            "fn f(st: &SafeTensors, b: &GpuBuffer) { st.read_f32(&n)?; b.read_f32(); st.read_u32(&n)?; b.read_u32(); }"
        ),
        2
    );
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
