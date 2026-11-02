//! The hosted CI job runs `scripts/hosted_tests.sh` over `scripts/hosted_tests.txt`
//! on a runner with no usable Metal device. These tests keep that list honest
//! from the source tree, with no device of their own:
//!
//! * every `exact` and `ignored` name is a real `#[test]` in the file it is
//!   listed under, and every `bin` / `lib` entry names something that exists,
//!   so a rename shows up here, on a laptop, not only as a red hosted step;
//! * every `exact` test is device-free by source: neither its body nor any
//!   function it calls in the same file mentions a runtime or the GPU helpers.
//!   The sandbox the hosted job runs under is the real enforcement; this is
//!   the early, local half of it;
//! * every `lib FILTER COUNT` entry is resolved to the tests libtest would run
//!   for that filter, and each of them gets the same device-free scan. A `lib`
//!   entry selects a whole module, so without this it reported clean because
//!   nothing asked. The resolved set is also counted against `COUNT`, so a
//!   filter that quietly gains or loses a test fails here first;
//! * the scanner flags a planted device test, both as an `exact` name and
//!   behind a `lib` filter, so a clean report means something.
//!
//! `hosted_selection_has_no_device` is the one `#[ignore]`d test: the hosted
//! script runs it first, under the sandbox, to prove the device really is hidden.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

const MANIFEST: &str = include_str!("../scripts/hosted_tests.txt");

/// Source text that means "this opens, or needs, a device".
const DEVICE_MARKERS: &[&str] = &[
    "with_gpu",
    "with_two_gpus",
    "gpu_lock",
    "in_traced_child",
    "GpuRuntime",
    "new_inference",
    "Gemm::new",
    "MTLCreateSystemDefaultDevice",
];

/// Call paths that name a device type but provably open nothing: a pure
/// associated function on it. Without this, `GpuRuntime::is_event_wait_timeout`
/// — which classifies an error string and takes no `self` — reads as a device
/// marker, and a module filter over a test that calls it is refused although
/// every test it selects is device-free.
///
/// Each entry is `(the path a call site writes, the file that defines it, its
/// own name there)`. The last two are not decoration:
/// `the_device_marker_exemptions_are_still_pure` re-reads each body and refuses
/// the exemption the moment it stops being true. An exemption asserted once and
/// never re-checked would go on passing the day its function grew a device
/// call, and would be the very thing hiding it.
const DEVICE_MARKER_EXEMPT: &[(&str, &str, &str)] = &[(
    "GpuRuntime::is_event_wait_timeout",
    "src/runtime.rs",
    "is_event_wait_timeout",
)];

#[derive(Debug, PartialEq)]
enum Entry {
    Bin(String),
    Bins,
    Lib { filter: String, count: usize },
    Outside { bin: String, name: String },
    Exact { target: String, names: Vec<String> },
    Ignored { bin: String, name: String },
}

fn parse(manifest: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for (i, raw) in manifest.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        let f: Vec<&str> = line.split_whitespace().collect();
        let Some(kind) = f.first() else { continue };
        let at = i + 1;
        out.push(match (*kind, f.len()) {
            ("bin", 3) if f[2].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Bin(f[1].into()),
            ("bins", 2) if f[1].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Bins,
            ("lib", 3) if f[2].parse::<usize>().is_ok_and(|n| n > 0) => Entry::Lib {
                filter: f[1].into(),
                count: f[2].parse().unwrap(),
            },
            ("outside", 3) => Entry::Outside {
                bin: f[1].into(),
                name: f[2].into(),
            },
            ("exact", n) if n >= 3 => Entry::Exact {
                target: f[1].into(),
                names: f[2..].iter().map(|s| s.to_string()).collect(),
            },
            ("ignored", 3) => Entry::Ignored {
                bin: f[1].into(),
                name: f[2].into(),
            },
            _ => panic!("scripts/hosted_tests.txt:{at}: malformed entry {raw:?}"),
        });
    }
    out
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `{...}` body of `fn name(`, or `None` if there is no such function.
fn fn_body<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("fn {name}(");
    let at = src.match_indices(&key).map(|(i, _)| i).find(|&i| {
        src[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'))
    })?;
    let open = at + src[at..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open..open + i + 1]);
                }
            }
            _ => {}
        }
    }
    // Braces inside string or char literals can leave the count open. Fall back
    // to the text up to the next item: a longer span can only add markers, so
    // the scan stays conservative.
    let rest = &src[at..];
    let end = rest[1..].find("\n#[test]").map_or(rest.len(), |i| i + 1);
    Some(&rest[..end])
}

/// Every function name defined in `src`.
fn defined_fns(src: &str) -> HashSet<String> {
    src.match_indices("fn ")
        .filter_map(|(i, _)| {
            let name: String = src[i + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// `body` with every `DEVICE_MARKER_EXEMPT` call path replaced, so the marker
/// scan does not read a device type's name out of a call that opens nothing.
/// Nothing else is touched: a mention of the type in any other form still
/// counts, which is what keeps this narrower than retiring the marker.
fn without_exempt_calls(body: &str) -> String {
    let mut out = body.to_string();
    for (path, _, _) in DEVICE_MARKER_EXEMPT {
        out = out.replace(path, "exempt_pure_call");
    }
    out
}

/// A device marker reachable from `test` through functions defined in `src`.
fn device_marker(src: &str, test: &str) -> Result<Option<String>, String> {
    let src = strip_comments(src);
    let known = defined_fns(&src);
    let mut seen = HashSet::new();
    let mut work = vec![test.to_string()];
    while let Some(name) = work.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(body) = fn_body(&src, &name) else {
            if name == test {
                return Err(format!("no `fn {test}(` found"));
            }
            continue;
        };
        // Only the exempt call paths are blanked, and only for the marker scan;
        // every other mention of a device type still counts, and the call-graph
        // walk below still sees the body as written.
        let scanned = without_exempt_calls(body);
        for m in DEVICE_MARKERS {
            if scanned.contains(m) {
                return Ok(Some(format!("{name} mentions {m}")));
            }
        }
        for callee in &known {
            if body.contains(&format!("{callee}(")) {
                work.push(callee.clone());
            }
        }
    }
    Ok(None)
}

/// The file a manifest target lives in: `tests/BIN.rs`, or for `lib` the
/// source file of a `module::path::name`.
fn source_of(root: &Path, target: &str, name: &str) -> PathBuf {
    if target != "lib" {
        return root.join("tests").join(format!("{target}.rs"));
    }
    let mut parts: Vec<&str> = name.split("::").collect();
    parts.pop();
    parts.retain(|p| !p.ends_with("tests") && *p != "tests");
    let rel = parts.join("/");
    let flat = root.join("src").join(format!("{rel}.rs"));
    if flat.exists() {
        flat
    } else {
        root.join("src").join(&rel).join("mod.rs")
    }
}

// --------------------------------------------- resolving a `lib FILTER` entry
//
// `lib FILTER COUNT` is run as `cargo test --lib -- FILTER`, and libtest
// matches a bare filter as a *substring* of the whole `module::...::name` path,
// not as a prefix (`--exact` is what the `exact` entries add). So checking a
// `lib` entry at all — for its count or for device-freeness — means resolving
// the filter against every test path the library target has. That is the module
// tree, which is read here from `src/lib.rs` down.

/// One `#[test]` function of the library target.
#[derive(Debug)]
struct LibTest {
    /// The path libtest prints, and matches a filter against.
    path: String,
    /// The function's own name, which `device_marker` scans for.
    leaf: String,
    /// The file it is written in.
    file: PathBuf,
}

/// `src` with every comment and every string, byte-string and char literal
/// replaced by spaces, newlines kept.
///
/// Brace depth over the result is the real module nesting: a `{` inside a
/// shader source, a JSON fixture or a doc comment cannot shift it. This is what
/// `strip_comments` cannot do — it cuts at the first `//`, which a `https://`
/// inside a string literal also has.
///
/// Byte for byte as long as `src`, so an offset into one indexes the other: the
/// structure is read off this, and an attribute's own text — a `#[path = "…"]`
/// — off the source it came from.
fn blank_literals(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        let end = literal_end(&c, i);
        if end > i {
            for &ch in &c[i..end] {
                if ch == '\n' {
                    out.push('\n');
                } else {
                    // One space per byte, so offsets do not drift.
                    for _ in 0..ch.len_utf8() {
                        out.push(' ');
                    }
                }
            }
            i = end;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    debug_assert_eq!(out.len(), src.len());
    out
}

/// The end of the comment or literal starting at `i`, or `i` itself if none
/// starts there. Anything that does not parse as one is left alone, so a
/// surprise in the source cannot swallow the rest of a file.
fn literal_end(c: &[char], i: usize) -> usize {
    let at = |k: usize| c.get(k).copied().unwrap_or('\0');
    let name_char = |ch: char| ch.is_alphanumeric() || ch == '_';
    match at(i) {
        '/' if at(i + 1) == '/' => {
            let mut j = i + 2;
            while j < c.len() && at(j) != '\n' {
                j += 1;
            }
            j
        }
        // Block comments nest in Rust.
        '/' if at(i + 1) == '*' => {
            let mut depth = 1usize;
            let mut j = i + 2;
            while j < c.len() && depth > 0 {
                if at(j) == '/' && at(j + 1) == '*' {
                    depth += 1;
                    j += 2;
                } else if at(j) == '*' && at(j + 1) == '/' {
                    depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            j
        }
        '"' => quoted_end(c, i + 1),
        // `b"…"`, `r"…"`, `br"…"` and their `#`-hashed raw forms. A name
        // character in front means this is the tail of an identifier.
        'b' | 'r' if i == 0 || !name_char(c[i - 1]) => {
            let mut j = i;
            if at(j) == 'b' {
                j += 1;
            }
            if at(j) == 'r' {
                j += 1;
                let hashes = (0..).take_while(|k| at(j + k) == '#').count();
                if at(j + hashes) != '"' {
                    return i; // a raw identifier such as `r#type`
                }
                return raw_end(c, j + hashes + 1, hashes);
            }
            if at(j) == '"' {
                quoted_end(c, j + 1)
            } else {
                i
            }
        }
        // A char literal, against a lifetime: `'a'` against `'a`.
        '\'' if at(i + 1) == '\\' => {
            let mut j = i + 2;
            if at(j) == 'u' && at(j + 1) == '{' {
                while j < c.len() && at(j) != '}' {
                    j += 1;
                }
            }
            j += 1;
            if at(j) == '\'' {
                j + 1
            } else {
                i
            }
        }
        '\'' if at(i + 2) == '\'' => i + 3,
        _ => i,
    }
}

/// The end of a `"`-terminated literal whose body starts at `i`.
fn quoted_end(c: &[char], mut i: usize) -> usize {
    while i < c.len() {
        match c[i] {
            '\\' => i += 2,
            '"' => return i + 1,
            _ => i += 1,
        }
    }
    i
}

/// The end of a raw literal whose body starts at `i` and is closed by `"` and
/// `hashes` `#`s. Raw literals have no escapes.
fn raw_end(c: &[char], mut i: usize, hashes: usize) -> usize {
    while i < c.len() {
        if c[i] == '"' && (1..=hashes).all(|k| c.get(i + k) == Some(&'#')) {
            return i + hashes + 1;
        }
        i += 1;
    }
    i
}

/// True if a name starts at `i` rather than continuing one.
fn at_word_start(src: &str, i: usize) -> bool {
    src[..i]
        .chars()
        .next_back()
        .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
}

/// The name in `KEYWORD NAME` at `i`, and the offset just past that name.
fn declared_name(src: &str, i: usize, keyword: &str) -> Option<(String, usize)> {
    let rest = src.get(i..)?.strip_prefix(keyword)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let after_keyword = rest.trim_start();
    let name: String = after_keyword
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    let after_name = src.len() - after_keyword.len() + name.len();
    Some((name, after_name))
}

/// The span of the attribute lines attached to the item at `i`: every line
/// above it up to the first that is neither an attribute nor blank. Comments
/// read as blank, because `blank_literals` has already emptied them, so an
/// attribute above a doc comment still belongs to the item.
///
/// A span rather than the text, because the caller needs both halves: the
/// blanked source to recognise an attribute, the real one to read its string.
fn attrs_range(blank: &str, i: usize) -> Range<usize> {
    let mut start = blank[..i].rfind('\n').map_or(0, |n| n + 1);
    let end = start;
    while start > 0 {
        let line_start = blank[..start - 1].rfind('\n').map_or(0, |n| n + 1);
        let line = blank[line_start..start - 1].trim();
        if !(line.is_empty() || line.starts_with("#[")) {
            break;
        }
        start = line_start;
    }
    start..end
}

/// The index of the `}` closing the `{` at `open`.
fn matching_brace(src: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (k, ch) in src.as_bytes().iter().enumerate().skip(open) {
        match ch {
            b'{' => depth += 1,
            // A `}` with nothing open means `open` was not a `{`: say so rather
            // than wrapping the count and answering confidently.
            b'}' if depth == 0 => return None,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(k);
                }
            }
            _ => {}
        }
    }
    None
}

/// The directory a `mod NAME;` inside `file` resolves against.
fn module_dir(file: &Path) -> PathBuf {
    match file.file_stem().and_then(|s| s.to_str()) {
        Some("lib") | Some("mod") => file.parent().unwrap().to_path_buf(),
        _ => file.with_extension(""),
    }
}

/// The file named by a `#[path = "…"]` among `attrs`, if there is one. Two
/// modules in `src/` use it (`runtime::pool_ledger_tests` and the footprint
/// helper `qwen35_q4` shares with the bench binaries), and without reading it
/// the walk stops at a `mod` whose file is not where the name says.
fn path_attribute(attrs: &str) -> Option<String> {
    let at = attrs.find("#[path")?;
    let rest = &attrs[at..];
    let open = rest.find('"')?;
    let close = rest[open + 1..].find('"')?;
    Some(rest[open + 1..open + 1 + close].to_string())
}

/// Walk one module body — or a whole file, at `prefix` — collecting the
/// `#[test]` functions declared directly in it, recursing into its
/// `mod NAME { … }` blocks, and reporting its `mod NAME;` declarations in
/// `children` for the caller to open as files.
fn walk_scope(
    blank: &str,
    src: &str,
    scope: Range<usize>,
    prefix: &str,
    file: &Path,
    tests: &mut Vec<LibTest>,
    children: &mut Vec<(String, Option<String>)>,
) {
    let bytes = blank.as_bytes();
    let mut depth = 0usize;
    let mut i = scope.start;
    while i < scope.end {
        match bytes[i] {
            b'{' => {
                depth += 1;
                i += 1;
                continue;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
                continue;
            }
            _ => {}
        }
        // Only what this module declares itself. A nested body is walked by the
        // recursive call that owns it, and a `macro_rules!` body — whose braces
        // are balanced — is skipped the same way.
        if depth > 0 || !at_word_start(blank, i) {
            i += 1;
            continue;
        }
        if let Some((name, after)) = declared_name(blank, i, "mod") {
            let span = attrs_range(blank, i);
            // The hosted job builds no features, so a module behind one has no
            // tests to run. A filter that named into it resolves to nothing,
            // and the count check then fails rather than passing quietly.
            let gated = blank[span.clone()].contains("feature");
            let tail = blank[after..].trim_start();
            if tail.starts_with('{') {
                let open = after + blank[after..].find('{').unwrap();
                let close = matching_brace(blank, open)
                    .unwrap_or_else(|| panic!("{}: `mod {name} {{` is never closed", file.display()));
                if !gated {
                    let into = format!("{prefix}{name}::");
                    walk_scope(blank, src, open + 1..close, &into, file, tests, children);
                }
                i = close + 1;
            } else {
                if tail.starts_with(';') && !gated {
                    children.push((name, path_attribute(&src[span])));
                }
                i = after;
            }
            continue;
        }
        if let Some((name, after)) = declared_name(blank, i, "fn") {
            let attrs = &blank[attrs_range(blank, i)];
            // An `#[ignore]`d test is listed but never runs, and `COUNT` is the
            // number that passed, so it is not part of either answer.
            if attrs.contains("#[test]") && !attrs.contains("#[ignore") {
                tests.push(LibTest {
                    path: format!("{prefix}{name}"),
                    leaf: name,
                    file: file.to_path_buf(),
                });
            }
            i = after;
            continue;
        }
        i += 1;
    }
}

/// Every `#[test]` the library target of the crate at `root` runs, with the
/// path libtest gives it.
fn lib_tests_in(root: &Path) -> Vec<LibTest> {
    let mut tests = Vec::new();
    let mut queue = vec![(String::new(), root.join("src").join("lib.rs"))];
    while let Some((prefix, file)) = queue.pop() {
        let src = fs::read_to_string(&file).unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
        let blank = blank_literals(&src);
        let mut children = Vec::new();
        walk_scope(&blank, &src, 0..blank.len(), &prefix, &file, &mut tests, &mut children);
        for (name, explicit) in children {
            // `#[path]` at the top of a file is relative to the directory that
            // file sits in, not to the directory the module name would imply.
            let child = match explicit {
                Some(rel) => {
                    let named = file.parent().unwrap().join(&rel);
                    assert!(
                        named.exists(),
                        "{}: `#[path = {rel:?}] mod {name};` points at no file ({})",
                        file.display(),
                        named.display()
                    );
                    named
                }
                None => {
                    let dir = module_dir(&file);
                    let flat = dir.join(format!("{name}.rs"));
                    let nested = dir.join(&name).join("mod.rs");
                    assert!(
                        flat.exists() || nested.exists(),
                        "{}: `mod {name};` resolves to neither {} nor {}",
                        file.display(),
                        flat.display(),
                        nested.display()
                    );
                    if flat.exists() {
                        flat
                    } else {
                        nested
                    }
                }
            };
            queue.push((format!("{prefix}{name}::"), child));
        }
    }
    tests.sort_by(|a, b| a.path.cmp(&b.path));
    tests
}

/// The tests `cargo test --lib -- FILTER` would run, in path order.
fn lib_tests_matching<'a>(all: &'a [LibTest], filter: &str) -> Vec<&'a LibTest> {
    all.iter().filter(|t| t.path.contains(filter)).collect()
}

/// How many `fn NAME(` definitions `src` has, counted at a name boundary.
fn fn_definitions(src: &str, name: &str) -> usize {
    let key = format!("fn {name}(");
    src.match_indices(&key).filter(|&(i, _)| at_word_start(src, i)).count()
}

/// Scan one resolved test, reading each file at most once. A leaf name defined
/// twice in the same file is refused rather than guessed at: `fn_body` takes
/// the first match, and in a file with a dozen test modules that need not be
/// the one the filter selected.
fn scan_resolved(test: &LibTest, cache: &mut HashMap<PathBuf, String>, what: &str) {
    let src = cache.entry(test.file.clone()).or_insert_with(|| {
        fs::read_to_string(&test.file).unwrap_or_else(|e| panic!("cannot read {}: {e}", test.file.display()))
    });
    let found = fn_definitions(&blank_literals(src), &test.leaf);
    assert_eq!(
        found,
        1,
        "{what} selects {}, but {} defines `fn {}(` {found} time(s): the scanner cannot tell which body is the test",
        test.path,
        test.file.display(),
        test.leaf
    );
    match device_marker(src, &test.leaf) {
        Ok(None) => {}
        Ok(Some(why)) => panic!(
            "{what} selects {}, which is in the hosted selection but is not device-free: {why}",
            test.path
        ),
        Err(e) => panic!("{what} selects {}: {e}", test.path),
    }
}

fn test_attribute_precedes(src: &str, name: &str) -> bool {
    let key = format!("fn {name}(");
    let Some(at) = src.find(&key) else { return false };
    let head: Vec<&str> = src[..at].lines().rev().take(6).collect();
    head.iter().any(|l| l.trim_start().starts_with("#[test]"))
}

fn every_name() -> Vec<(String, String, bool)> {
    let mut out = Vec::new();
    for e in parse(MANIFEST) {
        match e {
            Entry::Exact { target, names } => {
                for n in names {
                    out.push((target.clone(), n, false));
                }
            }
            Entry::Ignored { bin, name } | Entry::Outside { bin, name } => out.push((bin, name, true)),
            Entry::Bin(_) | Entry::Bins | Entry::Lib { .. } => {}
        }
    }
    out
}

#[test]
fn the_manifest_is_not_empty_and_has_no_duplicate_names() {
    let entries = parse(MANIFEST);
    assert!(!entries.is_empty(), "scripts/hosted_tests.txt lists nothing");
    let mut seen = HashSet::new();
    for (target, name, _) in every_name() {
        assert!(
            seen.insert((target.clone(), name.clone())),
            "{target} {name} is listed twice"
        );
    }
    let mut bins = HashSet::new();
    for e in &entries {
        if let Entry::Bin(b) = e {
            assert!(bins.insert(b.clone()), "bin {b} is listed twice");
        }
    }
}

#[test]
fn every_listed_name_is_a_real_test_where_it_is_listed() {
    for (target, name, ignored) in every_name() {
        let leaf = name.rsplit("::").next().unwrap();
        let path = source_of(root(), &target, &name);
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{target} {name}: cannot read {}: {e}", path.display()));
        assert!(
            test_attribute_precedes(&src, leaf),
            "{target} {name}: no #[test] fn {leaf} in {}",
            path.display()
        );
        if ignored {
            assert!(
                src.contains("#[ignore"),
                "{target} {name}: listed as ignored but {} has no #[ignore]",
                path.display()
            );
        }
    }
}

#[test]
fn every_whole_binary_and_lib_filter_names_something() {
    for e in parse(MANIFEST) {
        match e {
            Entry::Bin(b) => {
                let path = root().join("tests").join(format!("{b}.rs"));
                assert!(path.exists(), "bin {b}: {} does not exist", path.display());
            }
            Entry::Lib { filter, .. } => {
                let module = filter.trim_end_matches("::").split("::").next().unwrap();
                let flat = root().join("src").join(format!("{module}.rs"));
                let dir = root().join("src").join(module).join("mod.rs");
                assert!(flat.exists() || dir.exists(), "lib {filter}: no module src/{module}");
            }
            _ => {}
        }
    }
}

/// Every test every `exact` entry names is device-free by source.
///
/// Parameterised for the same reason the two `lib` checks are: the planted test
/// has to drive *this*, not a copy of it. `the_scanner_flags_a_planted_device_test`
/// proves `device_marker` works, but it calls it on literal strings, so it would
/// stay green if this loop stopped feeding it — the very shape this change is
/// about, one path over.
fn check_exact_device_free(manifest: &str, root: &Path) {
    for e in parse(manifest) {
        let Entry::Exact { target, names } = e else { continue };
        for name in names {
            let leaf = name.rsplit("::").next().unwrap();
            let path = source_of(root, &target, &name);
            let src = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{target} {name}: cannot read {}: {e}", path.display()));
            match device_marker(&src, leaf) {
                Ok(None) => {}
                Ok(Some(why)) => panic!("{target} {name} is in the hosted selection but is not device-free: {why}"),
                Err(e) => panic!("{target} {name}: {e}"),
            }
        }
    }
}

#[test]
fn every_exact_test_is_device_free_by_source() {
    check_exact_device_free(MANIFEST, root());
}

// Both `lib` checks are written against a manifest and a crate root rather than
// against `MANIFEST` and `root()` directly, so that the planted test below can
// run *these* functions over a fixture. A planted test that exercised only the
// resolver would still pass if either check went back to skipping `Entry::Lib`,
// which is the bug that was here.

/// Every `lib` entry selects exactly the number of tests it claims.
fn check_lib_counts(manifest: &str, root: &Path) {
    let all = lib_tests_in(root);
    assert!(!all.is_empty(), "no library tests were found at all under src/");
    for e in parse(manifest) {
        let Entry::Lib { filter, count } = e else { continue };
        let got = lib_tests_matching(&all, &filter);
        assert_eq!(
            got.len(),
            count,
            "lib {filter} claims {count} test(s) but selects {}: {}",
            got.len(),
            got.iter().map(|t| t.path.as_str()).collect::<Vec<_>>().join(" ")
        );
    }
}

/// Every test every `lib` entry selects is device-free by source.
fn check_lib_device_free(manifest: &str, root: &Path) {
    let all = lib_tests_in(root);
    let mut cache = HashMap::new();
    for e in parse(manifest) {
        let Entry::Lib { filter, .. } = e else { continue };
        let got = lib_tests_matching(&all, &filter);
        assert!(!got.is_empty(), "lib {filter} selects nothing");
        for test in got {
            scan_resolved(test, &mut cache, &format!("lib {filter}"));
        }
    }
}

#[test]
fn every_lib_filter_selects_the_number_of_tests_it_claims() {
    check_lib_counts(MANIFEST, root());
}

#[test]
fn every_lib_filter_selects_only_device_free_tests() {
    check_lib_device_free(MANIFEST, root());
}

/// A planted manifest entry naming a device-opening test must fail the real
/// checks — and this drives `check_lib_device_free`, `check_lib_counts` and
/// `check_exact_device_free` themselves rather than reimplementing them.
///
/// That indirection is the whole point. The bug being closed was the manifest
/// loop never looking at `Entry::Lib`, not anything wrong with the scanner, and
/// a check that passes everything looks identical from outside to a check that
/// is never asked. `the_scanner_flags_a_planted_device_test` has the same blind
/// spot for the `exact` path: it calls `device_marker` on literal strings, so
/// gutting `check_exact_device_free` leaves it green. Both paths are covered
/// here, through a fixture crate that gives the resolvers a real layout —
/// `src/lib.rs` into a file module, then an inline `#[cfg(test)] mod`, plus a
/// `tests/` binary for the `exact` arm.
#[test]
fn a_planted_manifest_entry_over_a_device_test_is_refused() {
    let dir = std::env::temp_dir().join(format!("tessl-hosted-selection-planted-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/lib.rs"), "pub mod planted;\npub mod clean;\n").unwrap();
    fs::write(
        dir.join("src/planted.rs"),
        concat!(
            "fn helper() {\n    let _rt = GpuRuntime::new();\n}\n",
            "#[cfg(test)]\nmod tests {\n",
            "    use super::*;\n",
            "    /// A doc comment, so the attribute walk has to step over one.\n",
            "    #[test]\n    fn is_clean() {\n        assert_eq!(1 + 1, 2);\n    }\n",
            "    #[test]\n    fn opens_a_device() {\n        helper();\n    }\n",
            "    #[test]\n    #[ignore = \"never runs, so it is not in the count\"]\n",
            "    fn is_ignored() {\n        with_gpu(|rt| drop(rt));\n    }\n",
            "}\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("src/clean.rs"),
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn adds_up() {\n        assert_eq!(2 + 2, 4);\n    }\n}\n",
    )
    .unwrap();

    // The resolver reads the layout, and leaves the `#[ignore]`d test out.
    assert_eq!(
        lib_tests_in(&dir).iter().map(|t| t.path.as_str()).collect::<Vec<_>>(),
        [
            "clean::tests::adds_up",
            "planted::tests::is_clean",
            "planted::tests::opens_a_device",
        ],
        "the resolver did not read the planted crate's module tree"
    );

    let refused =
        |manifest: &'static str, check: fn(&str, &Path)| std::panic::catch_unwind(|| check(manifest, &dir)).is_err();

    // The device-free check must refuse the module that opens one, and pass the
    // one that does not — so a clean report still means something.
    assert!(
        refused("lib planted::tests:: 2\n", check_lib_device_free),
        "a `lib` entry selecting a device-opening test was reported clean"
    );
    check_lib_device_free("lib clean::tests:: 1\n", &dir);

    // And the count: a pin that does not match what the filter selects.
    assert!(
        refused("lib planted::tests:: 1\n", check_lib_counts),
        "a `lib` entry whose COUNT is wrong was accepted"
    );
    check_lib_counts("lib planted::tests:: 2\n", &dir);

    // Substring, not prefix. `ted::tests::` is a prefix of nothing, yet libtest
    // would select both planted tests with it, because a bare filter matches
    // anywhere in the path. Pinned here so that "tidying" the resolver to
    // `starts_with` fails rather than quietly scanning a smaller set than the
    // gate runs — the same shape as the planted entries above, making the
    // wiring's assumption explicit instead of incidental.
    assert_eq!(
        lib_tests_matching(&lib_tests_in(&dir), "ted::tests::")
            .iter()
            .map(|t| t.path.as_str())
            .collect::<Vec<_>>(),
        ["planted::tests::is_clean", "planted::tests::opens_a_device"],
        "a filter that is a prefix of nothing selected nothing: this is matching by prefix"
    );
    assert!(
        refused("lib ted::tests:: 2\n", check_lib_device_free),
        "a device test reached by a mid-path filter was reported clean"
    );

    // The `exact` arm, through its own manifest loop rather than the scanner.
    fs::create_dir_all(dir.join("tests")).unwrap();
    fs::write(
        dir.join("tests/planted_bin.rs"),
        concat!(
            "fn helper() {\n    with_gpu(|rt| drop(rt));\n}\n",
            "#[test]\nfn opens_a_device() {\n    helper();\n}\n",
            "#[test]\nfn is_clean() {\n    assert_eq!(2 + 2, 4);\n}\n",
        ),
    )
    .unwrap();
    assert!(
        refused("exact planted_bin opens_a_device\n", check_exact_device_free),
        "an `exact` entry naming a device-opening test was reported clean"
    );
    check_exact_device_free("exact planted_bin is_clean\n", &dir);

    fs::remove_dir_all(&dir).unwrap();
}

/// Every `DEVICE_MARKER_EXEMPT` entry is exempt because its own body opens
/// nothing — re-read here rather than taken on trust.
///
/// This is the half that keeps the exemption honest over time. A name in a list
/// is a claim someone checked once; it would go on passing the day its function
/// grew a device call, and the exemption is exactly what would hide that from
/// every test calling it. So the claim is re-derived on every run, from the same
/// `fn_body` and `DEVICE_MARKERS` the scan itself uses.
#[test]
fn the_device_marker_exemptions_are_still_pure() {
    assert!(
        !DEVICE_MARKER_EXEMPT.is_empty(),
        "no exemptions to check: if the list is emptied, drop this test with it \
         rather than leaving one that asserts nothing"
    );
    for (path, file, name) in DEVICE_MARKER_EXEMPT {
        let full = root().join(file);
        let src = fs::read_to_string(&full).unwrap_or_else(|e| panic!("{path}: cannot read {}: {e}", full.display()));
        let stripped = strip_comments(&src);

        // One definition, or `fn_body` would be reading whichever came first.
        let defined = fn_definitions(&blank_literals(&src), name);
        assert_eq!(
            defined,
            1,
            "{path}: {} defines `fn {name}(` {defined} time(s), so which body the \
             exemption rests on is not decidable",
            full.display()
        );

        let body = fn_body(&stripped, name).unwrap_or_else(|| panic!("{path}: no `fn {name}(` in {}", full.display()));
        for m in DEVICE_MARKERS {
            assert!(
                !body.contains(m),
                "{path} is exempt from the device scan, but its body in {} now mentions \
                 {m}. The exemption is no longer true, and leaving it would hide a device \
                 call from every test that calls this.",
                full.display()
            );
        }

        // And the exemption must still be able to fire: a path that matches no
        // call form exempts nothing and quietly stops carrying its weight.
        assert_eq!(
            without_exempt_calls(&format!("let _ = {path}(x);"))
                .matches(path)
                .count(),
            0,
            "{path}: the exemption did not rewrite its own call form"
        );
    }
}

#[test]
fn the_scanner_flags_a_planted_device_test() {
    let direct = "#[test]\nfn planted() { with_gpu(|rt| drop(rt)); }\n";
    assert!(device_marker(direct, "planted").unwrap().is_some());
    let through_helper = "fn helper() { let _rt = GpuRuntime::new(); }\n#[test]\nfn planted() { helper(); }\n";
    assert!(
        device_marker(through_helper, "planted").unwrap().is_some(),
        "a device reached through a helper went unseen"
    );
    let in_comment = "#[test]\nfn fine() { let x = 1; // with_gpu is not called here\n let _ = x; }\n";
    assert!(device_marker(in_comment, "fine").unwrap().is_none());
    assert!(
        device_marker("fn other() {}", "missing").is_err(),
        "a missing test must be an error, not a pass"
    );
}

#[test]
fn the_manifest_parser_refuses_a_malformed_line() {
    for bad in [
        "exact tests_only_a_target\n",
        "bin gdn_gates\n",
        "bin gdn_gates 0\n",
        "lib a::\n",
        "lib a:: 0\n",
        "lib a:: many\n",
        "bins\n",
    ] {
        let result = std::panic::catch_unwind(|| parse(bad));
        assert!(result.is_err(), "a malformed manifest line was accepted: {bad:?}");
    }
    assert_eq!(
        parse("# comment\n\nbin gdn_gates 3 # trailing\nbins 4\n"),
        vec![Entry::Bin("gdn_gates".into()), Entry::Bins]
    );
}

/// Run by `scripts/hosted_tests.sh` before it trusts the sandbox: with the
/// device hidden, no runtime can be created. Ignored so a laptop's full run
/// (which has a device) does not see it.
#[test]
#[ignore = "run only under the hosted sandbox: asserts the device is hidden"]
fn hosted_selection_has_no_device() {
    assert!(
        std::env::var_os("TESSL_HOSTED_NO_DEVICE").is_some(),
        "TESSL_HOSTED_NO_DEVICE is not set: this is not the hosted selection"
    );
    assert!(
        tessl::GpuRuntime::new().is_err(),
        "GpuRuntime::new succeeded: the sandbox did not hide the Metal device"
    );
    assert!(
        tessl::GpuRuntime::new_inference().is_err(),
        "GpuRuntime::new_inference succeeded: the sandbox did not hide the Metal device"
    );
}
